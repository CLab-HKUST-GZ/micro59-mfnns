#include <hnswlib/space_l2_dynamic_precision_et.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
using Label = hnswlib::labeltype;

struct FbinHeader {
    uint64_t rows = 0;
    uint64_t dim = 0;
};

enum class VectorFormat {
    FBIN,
    FVECS,
};

struct IndexHeader {
    size_t offset_level0 = 0;
    size_t max_elements = 0;
    size_t current_elements = 0;
    size_t bytes_per_element = 0;
    size_t label_offset = 0;
    size_t data_offset = 0;
    int max_level = 0;
    hnswlib::tableint entrypoint = 0;
    size_t max_m = 0;
    size_t max_m0 = 0;
    size_t m = 0;
    double level_multiplier = 0.0;
    size_t ef_construction = 0;
};

class Options {
 public:
    Options(int argc, char** argv, int begin) {
        for (int i = begin; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help" || key == "-h") {
                values_["help"] = "1";
                continue;
            }
            if (key.rfind("--", 0) != 0) {
                throw std::runtime_error("Expected an option beginning with '--', got: " + key);
            }
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for option: " + key);
            }
            const std::string name = key.substr(2);
            if (values_.count(name) != 0) {
                throw std::runtime_error("Duplicate option: " + key);
            }
            values_[name] = argv[++i];
        }
    }

    bool has(const std::string& name) const {
        return values_.count(name) != 0;
    }

    std::string require(const std::string& name) const {
        const auto it = values_.find(name);
        if (it == values_.end() || it->second.empty()) {
            throw std::runtime_error("Missing required option: --" + name);
        }
        return it->second;
    }

    std::string get(const std::string& name, const std::string& fallback) const {
        const auto it = values_.find(name);
        return it == values_.end() ? fallback : it->second;
    }

    uint64_t get_u64(const std::string& name, uint64_t fallback) const {
        const auto it = values_.find(name);
        if (it == values_.end()) {
            return fallback;
        }
        size_t consumed = 0;
        const unsigned long long value = std::stoull(it->second, &consumed);
        if (consumed != it->second.size()) {
            throw std::runtime_error("Invalid integer for --" + name + ": " + it->second);
        }
        return static_cast<uint64_t>(value);
    }

    bool get_bool(const std::string& name, bool fallback) const {
        const auto it = values_.find(name);
        if (it == values_.end()) {
            return fallback;
        }
        const std::string value = lowercase(it->second);
        if (value == "1" || value == "true" || value == "yes") {
            return true;
        }
        if (value == "0" || value == "false" || value == "no") {
            return false;
        }
        throw std::runtime_error("Invalid boolean for --" + name + ": " + it->second);
    }

    void reject_unknown(const std::unordered_set<std::string>& allowed) const {
        for (const auto& item : values_) {
            if (allowed.count(item.first) == 0) {
                throw std::runtime_error("Unknown option: --" + item.first);
            }
        }
    }

    static std::string lowercase(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

 private:
    std::unordered_map<std::string, std::string> values_;
};

template <typename T>
void read_pod(std::istream& input, T& value, const std::string& description) {
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("Failed to read " + description);
    }
}

uint64_t checked_product(uint64_t lhs, uint64_t rhs, const std::string& description) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        throw std::runtime_error("Integer overflow while computing " + description);
    }
    return lhs * rhs;
}

FbinHeader read_fbin_header(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open FBIN file: " + path.string());
    }

    int32_t rows = 0;
    int32_t dim = 0;
    read_pod(input, rows, "FBIN row count from " + path.string());
    read_pod(input, dim, "FBIN dimension from " + path.string());
    if (rows <= 0 || dim <= 0) {
        throw std::runtime_error("Invalid FBIN shape in " + path.string());
    }

    const uint64_t payload_values =
        checked_product(static_cast<uint64_t>(rows), static_cast<uint64_t>(dim), "FBIN payload");
    const uint64_t expected_bytes =
        8 + checked_product(payload_values, sizeof(float), "FBIN byte size");
    const uint64_t actual_bytes = fs::file_size(path);
    if (actual_bytes != expected_bytes) {
        std::ostringstream message;
        message << "FBIN size mismatch for " << path << ": expected " << expected_bytes
                << " bytes, found " << actual_bytes;
        throw std::runtime_error(message.str());
    }
    return {static_cast<uint64_t>(rows), static_cast<uint64_t>(dim)};
}

VectorFormat parse_vector_format(const std::string& raw_format) {
    const std::string format = Options::lowercase(raw_format);
    if (format == "fbin") {
        return VectorFormat::FBIN;
    }
    if (format == "fvecs") {
        return VectorFormat::FVECS;
    }
    throw std::runtime_error(
        "Unsupported --base-format: " + raw_format + " (expected fbin or fvecs)");
}

const char* vector_format_name(VectorFormat format) {
    return format == VectorFormat::FBIN ? "fbin" : "fvecs";
}

FbinHeader read_fvecs_header(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open FVECS file: " + path.string());
    }

    int32_t dim = 0;
    read_pod(input, dim, "first FVECS dimension from " + path.string());
    if (dim <= 0) {
        throw std::runtime_error("Invalid FVECS dimension in " + path.string());
    }

    const uint64_t record_bytes =
        sizeof(int32_t) +
        checked_product(static_cast<uint64_t>(dim), sizeof(float), "FVECS record size");
    const uint64_t actual_bytes = fs::file_size(path);
    if (actual_bytes == 0 || actual_bytes % record_bytes != 0) {
        std::ostringstream message;
        message << "FVECS size mismatch for " << path << ": " << actual_bytes
                << " bytes is not divisible by record size " << record_bytes;
        throw std::runtime_error(message.str());
    }
    return {actual_bytes / record_bytes, static_cast<uint64_t>(dim)};
}

FbinHeader read_vector_header(const fs::path& path, VectorFormat format) {
    return format == VectorFormat::FBIN
               ? read_fbin_header(path)
               : read_fvecs_header(path);
}

IndexHeader read_index_header(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open HNSW index: " + path.string());
    }

    IndexHeader header;
    read_pod(input, header.offset_level0, "index offsetLevel0");
    read_pod(input, header.max_elements, "index max_elements");
    read_pod(input, header.current_elements, "index current_elements");
    read_pod(input, header.bytes_per_element, "index bytes_per_element");
    read_pod(input, header.label_offset, "index label_offset");
    read_pod(input, header.data_offset, "index data_offset");
    read_pod(input, header.max_level, "index max_level");
    read_pod(input, header.entrypoint, "index entrypoint");
    read_pod(input, header.max_m, "index maxM");
    read_pod(input, header.max_m0, "index maxM0");
    read_pod(input, header.m, "index M");
    read_pod(input, header.level_multiplier, "index level multiplier");
    read_pod(input, header.ef_construction, "index ef_construction");

    if (header.current_elements == 0 || header.current_elements > header.max_elements) {
        throw std::runtime_error("Invalid element counts in index header: " + path.string());
    }
    if (header.data_offset > header.label_offset ||
        header.label_offset + sizeof(Label) > header.bytes_per_element) {
        throw std::runtime_error("Invalid data/label offsets in index header: " + path.string());
    }
    if (header.m == 0 || header.max_m != header.m || header.max_m0 != 2 * header.m) {
        throw std::runtime_error("Invalid HNSW M fields in index header: " + path.string());
    }
    return header;
}

uint64_t infer_index_dim(const IndexHeader& header) {
    const uint64_t vector_bytes = header.label_offset - header.data_offset;
    if (vector_bytes == 0 || vector_bytes % sizeof(float) != 0) {
        throw std::runtime_error("Index does not contain an integral number of FP32 dimensions");
    }
    return vector_bytes / sizeof(float);
}

std::vector<float> read_fbin_rows(const fs::path& path,
                                  const FbinHeader& header,
                                  uint64_t start,
                                  uint64_t count) {
    if (start > header.rows || count > header.rows - start) {
        throw std::runtime_error("Requested FBIN row range is out of bounds");
    }
    const uint64_t values = checked_product(count, header.dim, "selected FBIN rows");
    if (values > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Selected FBIN row range is too large for this process");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open FBIN file: " + path.string());
    }
    const uint64_t value_offset = checked_product(start, header.dim, "FBIN row offset");
    const uint64_t byte_offset =
        8 + checked_product(value_offset, sizeof(float), "FBIN byte offset");
    input.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
    if (!input) {
        throw std::runtime_error("Failed to seek in FBIN file: " + path.string());
    }

    std::vector<float> result(static_cast<size_t>(values));
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(values * sizeof(float)));
    if (!input) {
        throw std::runtime_error("Failed to read selected rows from FBIN file: " + path.string());
    }
    return result;
}

std::vector<float> read_fvecs_rows(const fs::path& path,
                                   const FbinHeader& header,
                                   uint64_t start,
                                   uint64_t count) {
    if (start > header.rows || count > header.rows - start) {
        throw std::runtime_error("Requested FVECS row range is out of bounds");
    }
    const uint64_t values = checked_product(count, header.dim, "selected FVECS rows");
    if (values > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Selected FVECS row range is too large for this process");
    }

    const uint64_t record_bytes =
        sizeof(int32_t) +
        checked_product(header.dim, sizeof(float), "FVECS record size");
    const uint64_t byte_offset = checked_product(start, record_bytes, "FVECS row offset");
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open FVECS file: " + path.string());
    }
    input.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
    if (!input) {
        throw std::runtime_error("Failed to seek in FVECS file: " + path.string());
    }

    std::vector<float> result(static_cast<size_t>(values));
    for (uint64_t row = 0; row < count; ++row) {
        int32_t row_dim = 0;
        read_pod(input, row_dim, "FVECS row dimension from " + path.string());
        if (row_dim <= 0 || static_cast<uint64_t>(row_dim) != header.dim) {
            throw std::runtime_error(
                "Inconsistent FVECS row dimension in " + path.string());
        }
        float* destination = result.data() + static_cast<size_t>(row * header.dim);
        input.read(
            reinterpret_cast<char*>(destination),
            static_cast<std::streamsize>(header.dim * sizeof(float)));
        if (!input) {
            throw std::runtime_error(
                "Failed to read selected rows from FVECS file: " + path.string());
        }
    }
    return result;
}

std::vector<float> read_vector_rows(const fs::path& path,
                                    const FbinHeader& header,
                                    VectorFormat format,
                                    uint64_t start,
                                    uint64_t count) {
    return format == VectorFormat::FBIN
               ? read_fbin_rows(path, header, start, count)
               : read_fvecs_rows(path, header, start, count);
}

constexpr uint64_t FNV1A_OFFSET_BASIS = 14695981039346656037ULL;
constexpr uint64_t FNV1A_PRIME = 1099511628211ULL;

void update_fingerprint(uint64_t& fingerprint,
                        const std::vector<float>& values) {
    const auto* bytes =
        reinterpret_cast<const unsigned char*>(values.data());
    const size_t byte_count = values.size() * sizeof(float);
    for (size_t index = 0; index < byte_count; ++index) {
        fingerprint ^= bytes[index];
        fingerprint *= FNV1A_PRIME;
    }
}

void normalize_vectors(std::vector<float>& data, uint64_t rows, uint64_t dim, int threads) {
    std::atomic<uint64_t> first_invalid{rows};
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
        float norm_squared = 0.0f;
        const size_t offset = static_cast<size_t>(row) * dim;
        for (uint64_t col = 0; col < dim; ++col) {
            const float value = data[offset + col];
            norm_squared += value * value;
        }
        if (!std::isfinite(norm_squared) || norm_squared <= 1.0e-24f) {
            uint64_t current = first_invalid.load(std::memory_order_relaxed);
            while (static_cast<uint64_t>(row) < current &&
                   !first_invalid.compare_exchange_weak(
                       current, static_cast<uint64_t>(row),
                       std::memory_order_relaxed)) {
            }
            continue;
        }
        const float norm = std::sqrt(norm_squared);
        for (uint64_t col = 0; col < dim; ++col) {
            data[offset + col] /= norm;
        }
    }
    if (first_invalid.load(std::memory_order_relaxed) != rows) {
        throw std::runtime_error(
            "Cannot L2-normalize zero-norm or non-finite vector at selected row " +
            std::to_string(first_invalid.load(std::memory_order_relaxed)));
    }
}

void verify_unit_vectors(const std::vector<float>& data,
                         uint64_t rows,
                         uint64_t dim,
                         double tolerance,
                         const std::string& description) {
    for (uint64_t row = 0; row < rows; ++row) {
        double norm_squared = 0.0;
        const size_t offset = static_cast<size_t>(row) * dim;
        for (uint64_t col = 0; col < dim; ++col) {
            const float value = data[offset + col];
            norm_squared += static_cast<double>(value) * value;
        }
        const double norm = std::sqrt(norm_squared);
        if (!std::isfinite(norm) || std::abs(norm - 1.0) > tolerance) {
            std::ostringstream message;
            message << description << " row " << row
                    << " is not unit-normalized: norm=" << std::setprecision(10)
                    << norm << ", tolerance=" << tolerance;
            throw std::runtime_error(message.str());
        }
    }
}

void ensure_parent_directory(const fs::path& path) {
    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }
}

void write_build_metadata(const fs::path& path,
                          const fs::path& index_path,
                          const fs::path& base_path,
                          VectorFormat base_format,
                          uint64_t rows,
                          uint64_t dim,
                          uint64_t m,
                          uint64_t ef_construction,
                          uint64_t seed,
                          bool normalize,
                          uint64_t base_fingerprint) {
    ensure_parent_directory(path);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot create index metadata: " + path.string());
    }
    output << "field\tvalue\n"
           << "format\tmfnns_hnsw_index_v1\n"
           << "index_path\t" << fs::absolute(index_path).string() << '\n'
           << "base_path\t" << fs::absolute(base_path).string() << '\n'
           << "base_format\t" << vector_format_name(base_format) << '\n'
           << "rows\t" << rows << '\n'
           << "dimension\t" << dim << '\n'
           << "M\t" << m << '\n'
           << "ef_construction\t" << ef_construction << '\n'
           << "seed\t" << seed << '\n'
           << "base_fingerprint_fnv1a64\t" << base_fingerprint << '\n'
           << "normalization\t" << (normalize ? "l2" : "none") << '\n';
    if (!output) {
        throw std::runtime_error("Failed while writing index metadata: " + path.string());
    }
}

int checked_thread_count(uint64_t value) {
    if (value == 0 || value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("--threads must be in [1, INT_MAX]");
    }
    return static_cast<int>(value);
}

void print_usage(std::ostream& output) {
    output
        << "mfnns_hnsw_tool: build/inspect indexes and prepare/evaluate data\n\n"
        << "Usage:\n"
        << "  mfnns_hnsw_tool build --base BASE --index OUT.bin [options]\n"
        << "  mfnns_hnsw_tool prepare --base BASE --queries QUERY "
           "--output-dir DIR [options]\n"
        << "  mfnns_hnsw_tool inspect --index INDEX.bin\n"
        << "  mfnns_hnsw_tool evaluate --base BASE.fbin --queries QUERY.fbin "
           "--index INDEX.bin [options]\n\n"
        << "Build options:\n"
        << "  --base-format FORMAT     fbin or fvecs (default: fbin)\n"
        << "  --normalize 0|1          L2-normalize each vector (default: 1)\n"
        << "  --limit N                Build the first N rows (default: all)\n"
        << "  --m N                    HNSW M (default: 16)\n"
        << "  --ef-construction N      HNSW ef_construction (default: 500)\n"
        << "  --threads N              OpenMP normalization threads (default: 1)\n"
        << "  --insertion-threads N    HNSW construction threads (default: --threads)\n"
        << "  --batch-size N           Streaming batch rows (default: 100000)\n"
        << "  --seed N                 HNSW random seed (default: 100)\n"
        << "  --force 0|1              Replace an existing output index (default: 0)\n"
        << "  --metadata FILE          Write normalization/build provenance after success\n\n"
        << "Prepare options:\n"
        << "  --base-format FORMAT     fbin or fvecs (default: fbin)\n"
        << "  --query-format FORMAT    fbin or fvecs (default: fbin)\n"
        << "  --query-count N          Output query rows (default: 1000)\n"
        << "  --seed N                 Deterministic query-selection seed (default: 42)\n"
        << "  --gt-k-list LIST         Exact-GT widths (default: 5,10,100)\n"
        << "  --normalize 0|1          L2-normalize base/query vectors (default: 1)\n"
        << "  --threads N              Normalization/exact-GT threads (default: 1)\n"
        << "  --batch-size N           Streaming exact-GT base rows (default: 100000)\n"
        << "  --force 0|1              Replace a complete/partial bundle (default: 0)\n\n"
        << "Evaluate options:\n"
        << "  --normalize 0|1          L2-normalize base/query vectors (default: 1)\n"
        << "  --query-limit N          Number of query rows (default: 10)\n"
        << "  --query-offset N         First query row (default: 0)\n"
        << "  --k N                    Recall@k (default: 10)\n"
        << "  --ef N                   HNSW ef_search (default: 100)\n"
        << "  --threads N              Exact-GT/query threads (default: 1)\n"
        << "  --batch-size N           Exact-GT base batch rows (default: 100000)\n"
        << "  --precisions LIST        Comma-separated modes; FP32 is always added first\n"
        << "                           (default: fp32,fp16_true,fp16_fpma,fp8_e4m3,int16,int8)\n"
        << "  --output FILE.csv        Optional result CSV\n";
}

void run_inspect(const Options& options) {
    options.reject_unknown({"help", "index"});
    if (options.has("help")) {
        print_usage(std::cout);
        return;
    }
    const fs::path index_path = options.require("index");
    const IndexHeader header = read_index_header(index_path);
    std::cout << "index=" << fs::absolute(index_path) << '\n'
              << "file_bytes=" << fs::file_size(index_path) << '\n'
              << "max_elements=" << header.max_elements << '\n'
              << "current_elements=" << header.current_elements << '\n'
              << "dimension=" << infer_index_dim(header) << '\n'
              << "bytes_per_element=" << header.bytes_per_element << '\n'
              << "M=" << header.m << '\n'
              << "maxM0=" << header.max_m0 << '\n'
              << "ef_construction=" << header.ef_construction << '\n'
              << "max_level=" << header.max_level << '\n'
              << "entrypoint=" << header.entrypoint << '\n';
}

void run_build(const Options& options) {
    options.reject_unknown({"help", "base", "base-format", "index", "normalize",
                            "limit", "m", "ef-construction", "threads",
                            "insertion-threads", "batch-size", "seed", "force",
                            "metadata"});
    if (options.has("help")) {
        print_usage(std::cout);
        return;
    }

    const fs::path base_path = options.require("base");
    const fs::path index_path = options.require("index");
    const VectorFormat base_format =
        parse_vector_format(options.get("base-format", "fbin"));
    const FbinHeader base_header = read_vector_header(base_path, base_format);
    const bool normalize = options.get_bool("normalize", true);
    const bool force = options.get_bool("force", false);
    const uint64_t requested_limit = options.get_u64("limit", 0);
    const uint64_t rows =
        requested_limit == 0 ? base_header.rows : std::min(requested_limit, base_header.rows);
    const uint64_t m = options.get_u64("m", 16);
    const uint64_t ef_construction = options.get_u64("ef-construction", 500);
    const uint64_t seed = options.get_u64("seed", 100);
    const uint64_t batch_size = options.get_u64("batch-size", 100000);
    const int threads = checked_thread_count(options.get_u64("threads", 1));
    const int insertion_threads = checked_thread_count(
        options.get_u64("insertion-threads", static_cast<uint64_t>(threads)));

    if (rows == 0 || batch_size == 0 || m < 2 || m > 10000 ||
        ef_construction < m || seed > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Invalid build parameter");
    }
    if (fs::exists(index_path) && !force) {
        throw std::runtime_error("Output index already exists; pass --force 1 to replace it: " +
                                 index_path.string());
    }
    ensure_parent_directory(index_path);

    std::cout << "[build] base=" << fs::absolute(base_path) << '\n'
              << "[build] base_format=" << vector_format_name(base_format) << '\n'
              << "[build] shape=" << rows << "x" << base_header.dim << '\n'
              << "[build] normalize=" << (normalize ? "true" : "false") << '\n'
              << "[build] M=" << m << " ef_construction=" << ef_construction
              << " seed=" << seed << " preprocess_threads=" << threads
              << " insertion_threads=" << insertion_threads
              << " batch_size=" << batch_size << '\n'
              << "[build] output=" << fs::absolute(index_path) << std::endl;

    omp_set_dynamic(0);
    hnswlib::L2SpaceDynamicPrecision space(base_header.dim, hnswlib::PrecisionType::FP32);
    hnswlib::HierarchicalNSW<float> index(
        &space, rows, m, ef_construction, static_cast<size_t>(seed));

    const auto started = Clock::now();
    uint64_t inserted = 0;
    uint64_t base_fingerprint = FNV1A_OFFSET_BASIS;
    while (inserted < rows) {
        const uint64_t count = std::min(batch_size, rows - inserted);
        std::vector<float> batch =
            read_vector_rows(base_path, base_header, base_format, inserted, count);
        update_fingerprint(base_fingerprint, batch);
        if (normalize) {
            normalize_vectors(batch, count, base_header.dim, threads);
        }

        std::atomic<bool> insertion_failed{false};
        std::atomic<uint64_t> next_row{0};
        std::exception_ptr insertion_error;
        std::mutex insertion_error_mutex;
        auto insert_rows = [&]() {
            while (!insertion_failed.load(std::memory_order_relaxed)) {
                const uint64_t row =
                    next_row.fetch_add(1, std::memory_order_relaxed);
                if (row >= count) {
                    break;
                }
                try {
                    index.addPoint(
                        batch.data() + row * base_header.dim,
                        static_cast<Label>(inserted + row));
                } catch (...) {
                    bool expected = false;
                    if (insertion_failed.compare_exchange_strong(
                            expected, true, std::memory_order_relaxed)) {
                        std::lock_guard<std::mutex> lock(insertion_error_mutex);
                        insertion_error = std::current_exception();
                    }
                }
            }
        };

        std::vector<std::thread> workers;
        const int worker_count =
            static_cast<int>(std::min<uint64_t>(count, insertion_threads));
        workers.reserve(static_cast<size_t>(worker_count - 1));
        try {
            for (int thread_id = 1; thread_id < worker_count; ++thread_id) {
                workers.emplace_back(insert_rows);
            }
        } catch (...) {
            insertion_failed.store(true, std::memory_order_relaxed);
            for (auto& worker : workers) {
                worker.join();
            }
            throw;
        }
        insert_rows();
        for (auto& worker : workers) {
            worker.join();
        }
        if (insertion_error) {
            std::rethrow_exception(insertion_error);
        }
        inserted += count;
        const double elapsed =
            std::chrono::duration<double>(Clock::now() - started).count();
        std::cout << "[build] inserted=" << inserted << "/" << rows
                  << " elapsed_s=" << std::fixed << std::setprecision(3) << elapsed
                  << std::endl;
    }

    const auto save_started = Clock::now();
    index.saveIndex(index_path.string());
    const double save_seconds =
        std::chrono::duration<double>(Clock::now() - save_started).count();
    if (!fs::exists(index_path) || fs::file_size(index_path) == 0) {
        throw std::runtime_error("Index serialization did not produce a non-empty file");
    }

    const IndexHeader output_header = read_index_header(index_path);
    if (output_header.current_elements != rows || infer_index_dim(output_header) != base_header.dim ||
        output_header.m != m || output_header.ef_construction != ef_construction) {
        throw std::runtime_error("Serialized index header does not match requested build");
    }
    if (options.has("metadata")) {
        write_build_metadata(
            options.require("metadata"), index_path, base_path, base_format, rows,
            base_header.dim, m, ef_construction, seed, normalize,
            base_fingerprint);
    }

    const double total_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    std::cout << "RESULT command=build rows=" << rows
              << " dim=" << base_header.dim
              << " M=" << m
              << " ef_construction=" << ef_construction
              << " preprocess_threads=" << threads
              << " insertion_threads=" << insertion_threads
              << " total_s=" << std::fixed << std::setprecision(6) << total_seconds
              << " save_s=" << save_seconds
              << " index_bytes=" << fs::file_size(index_path) << std::endl;
}

using DistanceLabel = std::pair<float, Label>;
using MaxHeap = std::priority_queue<DistanceLabel>;

std::vector<std::vector<Label>> compute_exact_ground_truth(
    const fs::path& base_path,
    const FbinHeader& base_header,
    VectorFormat base_format,
    uint64_t rows,
    const std::vector<float>& queries,
    uint64_t query_count,
    uint64_t k,
    uint64_t batch_size,
    bool normalize,
    int threads,
    uint64_t* base_fingerprint) {
    std::vector<MaxHeap> heaps(query_count);
    hnswlib::L2SpaceDynamicPrecision exact_space(
        base_header.dim, hnswlib::PrecisionType::FP32);
    hnswlib::DISTFUNC<float> exact_distance = exact_space.get_dist_func();
    void* exact_distance_parameter = exact_space.get_dist_func_param();
    uint64_t processed = 0;
    const auto started = Clock::now();
    while (processed < rows) {
        const uint64_t count = std::min(batch_size, rows - processed);
        std::vector<float> batch =
            read_vector_rows(base_path, base_header, base_format, processed, count);
        if (base_fingerprint != nullptr) {
            update_fingerprint(*base_fingerprint, batch);
        }
        if (normalize) {
            normalize_vectors(batch, count, base_header.dim, threads);
        }

#pragma omp parallel for num_threads(threads) schedule(static)
        for (int64_t query_id = 0; query_id < static_cast<int64_t>(query_count); ++query_id) {
            MaxHeap& heap = heaps[static_cast<size_t>(query_id)];
            const float* query =
                queries.data() + static_cast<size_t>(query_id) * base_header.dim;
            for (uint64_t row = 0; row < count; ++row) {
                const float* point = batch.data() + row * base_header.dim;
                const DistanceLabel candidate{
                    exact_distance(query, point, exact_distance_parameter),
                    static_cast<Label>(processed + row)};
                if (heap.size() < k) {
                    heap.push(candidate);
                } else if (candidate < heap.top()) {
                    heap.pop();
                    heap.push(candidate);
                }
            }
        }
        processed += count;
        const double elapsed =
            std::chrono::duration<double>(Clock::now() - started).count();
        std::cout << "[gt] processed=" << processed << "/" << rows
                  << " elapsed_s=" << std::fixed << std::setprecision(3) << elapsed
                  << std::endl;
    }

    std::vector<std::vector<Label>> ground_truth(query_count);
    for (uint64_t query_id = 0; query_id < query_count; ++query_id) {
        auto& labels = ground_truth[query_id];
        labels.reserve(k);
        while (!heaps[query_id].empty()) {
            labels.push_back(heaps[query_id].top().second);
            heaps[query_id].pop();
        }
        std::reverse(labels.begin(), labels.end());
    }
    return ground_truth;
}

std::vector<uint64_t> parse_positive_integer_list(const std::string& text,
                                                  const std::string& option_name) {
    std::vector<uint64_t> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            throw std::runtime_error("Empty integer in --" + option_name);
        }
        size_t consumed = 0;
        const unsigned long long value = std::stoull(token, &consumed);
        if (consumed != token.size() || value == 0) {
            throw std::runtime_error(
                "Invalid positive integer in --" + option_name + ": " + token);
        }
        values.push_back(static_cast<uint64_t>(value));
    }
    if (values.empty()) {
        throw std::runtime_error("--" + option_name + " cannot be empty");
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::string join_integer_list(const std::vector<uint64_t>& values) {
    std::ostringstream output;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    return output.str();
}

std::vector<uint64_t> select_query_indices(uint64_t available,
                                           uint64_t requested,
                                           uint64_t seed) {
    if (available == 0 || requested == 0) {
        throw std::runtime_error("Query source and requested query count must be nonzero");
    }
    if (seed > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("--seed must fit uint32 for deterministic selection");
    }
    std::vector<uint64_t> indices;
    indices.reserve(static_cast<size_t>(requested));
    if (requested <= available) {
        indices.resize(static_cast<size_t>(available));
        std::iota(indices.begin(), indices.end(), uint64_t{0});
        std::mt19937 random(static_cast<uint32_t>(seed));
        std::shuffle(indices.begin(), indices.end(), random);
        indices.resize(static_cast<size_t>(requested));
        std::sort(indices.begin(), indices.end());
    } else {
        for (uint64_t index = 0; index < requested; ++index) {
            indices.push_back(index % available);
        }
    }
    return indices;
}

void write_float_matrix(const fs::path& path,
                        const std::vector<float>& values,
                        uint64_t rows,
                        uint64_t cols) {
    if (rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        cols > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        values.size() != checked_product(rows, cols, "float matrix values")) {
        throw std::runtime_error("Float matrix shape is not serializable: " + path.string());
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot create float matrix: " + path.string());
    }
    const int32_t output_rows = static_cast<int32_t>(rows);
    const int32_t output_cols = static_cast<int32_t>(cols);
    output.write(reinterpret_cast<const char*>(&output_rows), sizeof(output_rows));
    output.write(reinterpret_cast<const char*>(&output_cols), sizeof(output_cols));
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("Failed while writing float matrix: " + path.string());
    }
}

void write_query_indices(const fs::path& path,
                         const std::vector<uint64_t>& indices) {
    if (indices.size() >
        static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Too many query indices to serialize");
    }
    std::vector<int32_t> serialized;
    serialized.reserve(indices.size());
    for (uint64_t index : indices) {
        if (index > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("Query source index does not fit int32");
        }
        serialized.push_back(static_cast<int32_t>(index));
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot create query-index file: " + path.string());
    }
    const int32_t count = static_cast<int32_t>(serialized.size());
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    output.write(
        reinterpret_cast<const char*>(serialized.data()),
        static_cast<std::streamsize>(serialized.size() * sizeof(int32_t)));
    if (!output) {
        throw std::runtime_error("Failed while writing query-index file: " + path.string());
    }
}

void write_ground_truth(const fs::path& path,
                        const std::vector<std::vector<Label>>& ground_truth,
                        uint64_t k) {
    if (ground_truth.empty() ||
        ground_truth.size() >
            static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        k > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Ground-truth shape is not serializable");
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot create ground-truth file: " + path.string());
    }
    const int32_t rows = static_cast<int32_t>(ground_truth.size());
    const int32_t cols = static_cast<int32_t>(k);
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    for (const auto& labels : ground_truth) {
        if (labels.size() < k) {
            throw std::runtime_error("Ground-truth row is shorter than requested output");
        }
        for (uint64_t column = 0; column < k; ++column) {
            const Label label = labels[static_cast<size_t>(column)];
            if (label > static_cast<Label>(std::numeric_limits<uint32_t>::max())) {
                throw std::runtime_error("Ground-truth label does not fit uint32");
            }
            const uint32_t serialized = static_cast<uint32_t>(label);
            output.write(
                reinterpret_cast<const char*>(&serialized), sizeof(serialized));
        }
    }
    if (!output) {
        throw std::runtime_error("Failed while writing ground-truth file: " + path.string());
    }
}

std::vector<uint32_t> read_uint_matrix(const fs::path& path,
                                       uint64_t expected_rows,
                                       uint64_t expected_cols) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open uint32 matrix: " + path.string());
    }
    int32_t rows = 0;
    int32_t cols = 0;
    read_pod(input, rows, "matrix row count from " + path.string());
    read_pod(input, cols, "matrix column count from " + path.string());
    if (rows <= 0 || cols <= 0 ||
        static_cast<uint64_t>(rows) != expected_rows ||
        static_cast<uint64_t>(cols) != expected_cols) {
        throw std::runtime_error("Unexpected matrix shape in " + path.string());
    }
    const uint64_t count = checked_product(expected_rows, expected_cols, "uint32 matrix");
    const uint64_t expected_bytes =
        8 + checked_product(count, sizeof(uint32_t), "uint32 matrix bytes");
    if (fs::file_size(path) != expected_bytes ||
        count > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Unexpected matrix size in " + path.string());
    }
    std::vector<uint32_t> values(static_cast<size_t>(count));
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(uint32_t)));
    if (!input) {
        throw std::runtime_error("Failed while reading uint32 matrix: " + path.string());
    }
    return values;
}

fs::path query_output_path(const fs::path& output_dir,
                           uint64_t query_count,
                           uint64_t seed) {
    return output_dir /
           ("query_vectors_n" + std::to_string(query_count) + "_seed" +
            std::to_string(seed) + ".bin");
}

fs::path query_indices_output_path(const fs::path& output_dir,
                                   uint64_t query_count,
                                   uint64_t seed) {
    return output_dir /
           ("query_indices_n" + std::to_string(query_count) + "_seed" +
            std::to_string(seed) + ".bin");
}

fs::path ground_truth_output_path(const fs::path& output_dir,
                                  uint64_t k,
                                  uint64_t query_count,
                                  uint64_t seed) {
    return output_dir /
           ("gt_labels_topk" + std::to_string(k) + "_n" +
            std::to_string(query_count) + "_seed" + std::to_string(seed) +
            ".bin");
}

fs::path bundle_metadata_path(const fs::path& output_dir,
                              uint64_t query_count,
                              uint64_t seed) {
    return output_dir /
           ("bundle_metadata_n" + std::to_string(query_count) + "_seed" +
            std::to_string(seed) + ".tsv");
}

void write_bundle_metadata(const fs::path& path,
                           const fs::path& base_path,
                           const fs::path& query_path,
                           VectorFormat base_format,
                           VectorFormat query_format,
                           const FbinHeader& base_header,
                           const FbinHeader& query_header,
                           uint64_t query_count,
                           uint64_t unique_query_count,
                           uint64_t seed,
                           const std::vector<uint64_t>& gt_k_values,
                           bool normalize,
                           uint64_t base_fingerprint,
                           uint64_t query_source_fingerprint) {
    ensure_parent_directory(path);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot create bundle metadata: " + path.string());
    }
    output << "field\tvalue\n"
           << "format\tmfnns_normalized_query_gt_v1\n"
           << "base_path\t" << fs::absolute(base_path).string() << '\n'
           << "query_source_path\t" << fs::absolute(query_path).string() << '\n'
           << "base_format\t" << vector_format_name(base_format) << '\n'
           << "query_source_format\t" << vector_format_name(query_format) << '\n'
           << "base_rows\t" << base_header.rows << '\n'
           << "query_source_rows\t" << query_header.rows << '\n'
           << "dimension\t" << base_header.dim << '\n'
           << "query_count\t" << query_count << '\n'
           << "unique_query_count\t" << unique_query_count << '\n'
           << "seed\t" << seed << '\n'
           << "gt_k_list\t" << join_integer_list(gt_k_values) << '\n'
           << "base_fingerprint_fnv1a64\t" << base_fingerprint << '\n'
           << "query_source_fingerprint_fnv1a64\t"
           << query_source_fingerprint << '\n'
           << "normalization\t" << (normalize ? "l2" : "none") << '\n'
           << "distance\tl2\n";
    if (!output) {
        throw std::runtime_error("Failed while writing bundle metadata: " + path.string());
    }
}

void verify_cached_bundle(const fs::path& output_dir,
                          const fs::path& base_path,
                          const fs::path& query_source_path,
                          VectorFormat base_format,
                          VectorFormat query_format,
                          const FbinHeader& base_header,
                          const FbinHeader& query_source_header,
                          uint64_t query_count,
                          uint64_t seed,
                          const std::vector<uint64_t>& gt_k_values,
                          bool normalize) {
    const fs::path query_path =
        query_output_path(output_dir, query_count, seed);
    const FbinHeader query_header = read_fbin_header(query_path);
    if (query_header.rows != query_count ||
        query_header.dim != base_header.dim) {
        throw std::runtime_error("Cached query shape mismatch: " + query_path.string());
    }
    const std::vector<float> queries =
        read_fbin_rows(query_path, query_header, 0, query_header.rows);
    if (normalize) {
        verify_unit_vectors(
            queries, query_header.rows, query_header.dim, 2.0e-5,
            "Cached query");
    }

    const fs::path indices_path =
        query_indices_output_path(output_dir, query_count, seed);
    std::ifstream indices_input(indices_path, std::ios::binary);
    if (!indices_input) {
        throw std::runtime_error(
            "Cannot open cached query indices: " + indices_path.string());
    }
    int32_t index_count = 0;
    read_pod(indices_input, index_count, "cached query-index count");
    if (index_count <= 0 || static_cast<uint64_t>(index_count) != query_count ||
        fs::file_size(indices_path) !=
            sizeof(int32_t) +
                checked_product(query_count, sizeof(int32_t), "query-index bytes")) {
        throw std::runtime_error(
            "Cached query-index size mismatch: " + indices_path.string());
    }
    for (uint64_t row = 0; row < query_count; ++row) {
        int32_t index = -1;
        read_pod(indices_input, index, "cached query-source index");
        if (index < 0 ||
            static_cast<uint64_t>(index) >= query_source_header.rows) {
            throw std::runtime_error(
                "Cached query-source index is out of range: " +
                indices_path.string());
        }
    }

    std::vector<uint32_t> widest;
    const uint64_t widest_k = gt_k_values.back();
    for (uint64_t k : gt_k_values) {
        const fs::path gt_path =
            ground_truth_output_path(output_dir, k, query_count, seed);
        std::vector<uint32_t> labels =
            read_uint_matrix(gt_path, query_count, k);
        for (uint32_t label : labels) {
            if (label >= base_header.rows) {
                throw std::runtime_error(
                    "Cached ground-truth label is out of range: " +
                    gt_path.string());
            }
        }
        if (k == widest_k) {
            widest = std::move(labels);
        }
    }
    for (uint64_t k : gt_k_values) {
        if (k == widest_k) {
            continue;
        }
        const fs::path gt_path =
            ground_truth_output_path(output_dir, k, query_count, seed);
        const std::vector<uint32_t> labels =
            read_uint_matrix(gt_path, query_count, k);
        for (uint64_t row = 0; row < query_count; ++row) {
            for (uint64_t column = 0; column < k; ++column) {
                if (labels[static_cast<size_t>(row * k + column)] !=
                    widest[static_cast<size_t>(row * widest_k + column)]) {
                    throw std::runtime_error(
                        "Cached ground-truth prefix mismatch: " +
                        gt_path.string());
                }
            }
        }
    }

    const fs::path metadata =
        bundle_metadata_path(output_dir, query_count, seed);
    std::ifstream metadata_input(metadata);
    if (!metadata_input) {
        throw std::runtime_error(
            "Cannot open bundle metadata: " + metadata.string());
    }
    const std::string metadata_text(
        (std::istreambuf_iterator<char>(metadata_input)),
        std::istreambuf_iterator<char>());
    const std::string expected_normalization =
        std::string("normalization\t") + (normalize ? "l2\n" : "none\n");
    const std::vector<float> source_queries =
        read_vector_rows(
            query_source_path, query_source_header, query_format, 0,
            query_source_header.rows);
    uint64_t query_source_fingerprint = FNV1A_OFFSET_BASIS;
    update_fingerprint(query_source_fingerprint, source_queries);
    const std::vector<std::string> expected_fields{
        "base_path\t" + fs::absolute(base_path).string() + "\n",
        "query_source_path\t" + fs::absolute(query_source_path).string() + "\n",
        "base_format\t" + std::string(vector_format_name(base_format)) + "\n",
        "query_source_format\t" +
            std::string(vector_format_name(query_format)) + "\n",
        "base_rows\t" + std::to_string(base_header.rows) + "\n",
        "query_source_rows\t" + std::to_string(query_source_header.rows) + "\n",
        "dimension\t" + std::to_string(base_header.dim) + "\n",
        "query_count\t" + std::to_string(query_count) + "\n",
        "seed\t" + std::to_string(seed) + "\n",
        "gt_k_list\t" + join_integer_list(gt_k_values) + "\n",
        "query_source_fingerprint_fnv1a64\t" +
            std::to_string(query_source_fingerprint) + "\n",
    };
    if (metadata_text.find("format\tmfnns_normalized_query_gt_v1\n") ==
            std::string::npos ||
        metadata_text.find(expected_normalization) == std::string::npos) {
        throw std::runtime_error(
            "Bundle metadata does not match requested policy: " +
            metadata.string());
    }
    for (const std::string& field : expected_fields) {
        if (metadata_text.find(field) == std::string::npos) {
            throw std::runtime_error(
                "Bundle metadata is stale or mismatched at field " + field +
                "in " + metadata.string());
        }
    }
}

void run_prepare(const Options& options) {
    options.reject_unknown(
        {"help", "base", "base-format", "queries", "query-format",
         "output-dir", "query-count", "seed", "gt-k-list", "normalize",
         "threads", "batch-size", "force"});
    if (options.has("help")) {
        print_usage(std::cout);
        return;
    }

    const fs::path base_path = options.require("base");
    const fs::path query_source_path = options.require("queries");
    const fs::path output_dir = options.require("output-dir");
    const VectorFormat base_format =
        parse_vector_format(options.get("base-format", "fbin"));
    const VectorFormat query_format =
        parse_vector_format(options.get("query-format", "fbin"));
    const FbinHeader base_header =
        read_vector_header(base_path, base_format);
    const FbinHeader query_source_header =
        read_vector_header(query_source_path, query_format);
    const uint64_t query_count = options.get_u64("query-count", 1000);
    const uint64_t seed = options.get_u64("seed", 42);
    const std::vector<uint64_t> gt_k_values =
        parse_positive_integer_list(
            options.get("gt-k-list", "5,10,100"), "gt-k-list");
    const bool normalize = options.get_bool("normalize", true);
    const bool force = options.get_bool("force", false);
    const int threads =
        checked_thread_count(options.get_u64("threads", 1));
    const uint64_t batch_size =
        options.get_u64("batch-size", 100000);

    if (query_count == 0 || batch_size == 0 ||
        base_header.dim != query_source_header.dim ||
        gt_k_values.back() > base_header.rows) {
        throw std::runtime_error(
            "Require matching dimensions, nonzero query/batch counts, and "
            "every GT k <= base rows");
    }

    std::vector<fs::path> outputs{
        query_output_path(output_dir, query_count, seed),
        query_indices_output_path(output_dir, query_count, seed),
        bundle_metadata_path(output_dir, query_count, seed),
    };
    for (uint64_t k : gt_k_values) {
        outputs.push_back(
            ground_truth_output_path(output_dir, k, query_count, seed));
    }
    size_t existing = 0;
    for (const fs::path& path : outputs) {
        existing += fs::exists(path);
    }
    if (existing == outputs.size() && !force) {
        verify_cached_bundle(
            output_dir, base_path, query_source_path, base_format, query_format,
            base_header, query_source_header, query_count, seed, gt_k_values,
            normalize);
        std::cout << "RESULT command=prepare status=cached"
                  << " queries=" << query_count
                  << " gt_k_list=" << join_integer_list(gt_k_values)
                  << " normalization=" << (normalize ? "l2" : "none")
                  << " output_dir=" << fs::absolute(output_dir) << std::endl;
        return;
    }
    if (existing != 0 && !force) {
        throw std::runtime_error(
            "Partial query/GT bundle exists; pass --force 1 to replace it: " +
            output_dir.string());
    }

    const std::vector<uint64_t> selected_indices =
        select_query_indices(query_source_header.rows, query_count, seed);
    const std::vector<float> source_queries =
        read_vector_rows(
            query_source_path, query_source_header, query_format, 0,
            query_source_header.rows);
    uint64_t query_source_fingerprint = FNV1A_OFFSET_BASIS;
    update_fingerprint(query_source_fingerprint, source_queries);
    const uint64_t selected_values =
        checked_product(query_count, query_source_header.dim, "selected queries");
    if (selected_values > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Selected query matrix is too large");
    }
    std::vector<float> queries(static_cast<size_t>(selected_values));
    for (uint64_t row = 0; row < query_count; ++row) {
        const uint64_t source_row =
            selected_indices[static_cast<size_t>(row)];
        std::copy_n(
            source_queries.data() + source_row * query_source_header.dim,
            query_source_header.dim,
            queries.data() + row * query_source_header.dim);
    }
    if (normalize) {
        normalize_vectors(
            queries, query_count, query_source_header.dim, threads);
        verify_unit_vectors(
            queries, query_count, query_source_header.dim, 2.0e-5,
            "Generated query");
    }

    std::unordered_map<uint64_t, uint64_t> unique_positions;
    std::vector<uint64_t> query_to_unique(query_count);
    std::vector<float> unique_queries;
    unique_queries.reserve(queries.size());
    for (uint64_t row = 0; row < query_count; ++row) {
        const uint64_t source_row =
            selected_indices[static_cast<size_t>(row)];
        const auto inserted = unique_positions.emplace(
            source_row, unique_positions.size());
        const uint64_t unique_row = inserted.first->second;
        query_to_unique[static_cast<size_t>(row)] = unique_row;
        if (inserted.second) {
            const float* begin =
                queries.data() + row * query_source_header.dim;
            unique_queries.insert(
                unique_queries.end(), begin,
                begin + query_source_header.dim);
        }
    }

    omp_set_dynamic(0);
    const uint64_t unique_count = unique_positions.size();
    uint64_t base_fingerprint = FNV1A_OFFSET_BASIS;
    const std::vector<std::vector<Label>> unique_ground_truth =
        compute_exact_ground_truth(
            base_path, base_header, base_format, base_header.rows,
            unique_queries, unique_count, gt_k_values.back(), batch_size,
            normalize, threads, &base_fingerprint);
    std::vector<std::vector<Label>> ground_truth(query_count);
    for (uint64_t row = 0; row < query_count; ++row) {
        ground_truth[static_cast<size_t>(row)] =
            unique_ground_truth[
                static_cast<size_t>(query_to_unique[static_cast<size_t>(row)])];
    }

    fs::create_directories(output_dir);
    write_float_matrix(
        query_output_path(output_dir, query_count, seed), queries,
        query_count, query_source_header.dim);
    write_query_indices(
        query_indices_output_path(output_dir, query_count, seed),
        selected_indices);
    for (uint64_t k : gt_k_values) {
        write_ground_truth(
            ground_truth_output_path(output_dir, k, query_count, seed),
            ground_truth, k);
    }
    write_bundle_metadata(
        bundle_metadata_path(output_dir, query_count, seed), base_path,
        query_source_path, base_format, query_format, base_header,
        query_source_header, query_count, unique_count, seed, gt_k_values,
        normalize, base_fingerprint, query_source_fingerprint);
    verify_cached_bundle(
        output_dir, base_path, query_source_path, base_format, query_format,
        base_header, query_source_header, query_count, seed, gt_k_values,
        normalize);

    std::cout << "RESULT command=prepare status=generated"
              << " queries=" << query_count
              << " unique_queries=" << unique_count
              << " gt_k_list=" << join_integer_list(gt_k_values)
              << " normalization=" << (normalize ? "l2" : "none")
              << " output_dir=" << fs::absolute(output_dir) << std::endl;
}

struct PrecisionConfig {
    std::string name;
    hnswlib::PrecisionType type;
};

PrecisionConfig precision_from_name(const std::string& raw_name) {
    const std::string name = Options::lowercase(raw_name);
    static const std::map<std::string, hnswlib::PrecisionType> modes = {
        {"fp32", hnswlib::PrecisionType::FP32},
        {"fp32_fpma", hnswlib::PrecisionType::FP32_FPMA},
        {"fp16_true", hnswlib::PrecisionType::FP16_TRUE},
        {"fp16_fpma", hnswlib::PrecisionType::FP16_FPMA},
        {"fp8_e4m3", hnswlib::PrecisionType::FP8_E4M3},
        {"fp8_fpma", hnswlib::PrecisionType::FP8_FPMA},
        {"fp8_e2m5", hnswlib::PrecisionType::FP8_E2M5},
        {"fp8_fpma_e2m5", hnswlib::PrecisionType::FP8_FPMA_E2M5},
        {"fp8_e3m4", hnswlib::PrecisionType::FP8_E3M4},
        {"fp8_fpma_e3m4", hnswlib::PrecisionType::FP8_FPMA_E3M4},
        {"int16_diff_fp16_fpma", hnswlib::PrecisionType::INT16_DIFF_FP16_FPMA},
        {"int16", hnswlib::PrecisionType::INT16},
        {"int8", hnswlib::PrecisionType::INT8},
    };
    const auto it = modes.find(name);
    if (it == modes.end()) {
        throw std::runtime_error("Unknown precision mode: " + raw_name);
    }
    return {name, it->second};
}

std::vector<PrecisionConfig> parse_precisions(const std::string& text) {
    std::vector<PrecisionConfig> result;
    result.push_back(precision_from_name("fp32"));
    std::unordered_set<std::string> seen{"fp32"};
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            throw std::runtime_error("Empty precision name in --precisions");
        }
        PrecisionConfig config = precision_from_name(token);
        if (seen.insert(config.name).second) {
            result.push_back(config);
        }
    }
    return result;
}

std::vector<std::vector<Label>> search_index(
    hnswlib::HierarchicalNSW<float>& index,
    const std::vector<float>& queries,
    uint64_t query_count,
    uint64_t dim,
    uint64_t k,
    int threads) {
    std::vector<std::vector<Label>> predictions(query_count);
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (int64_t query_id = 0; query_id < static_cast<int64_t>(query_count); ++query_id) {
        auto result =
            index.searchKnn(queries.data() + static_cast<size_t>(query_id) * dim, k);
        std::vector<Label> labels;
        labels.reserve(k);
        while (!result.empty()) {
            labels.push_back(result.top().second);
            result.pop();
        }
        std::reverse(labels.begin(), labels.end());
        predictions[static_cast<size_t>(query_id)] = std::move(labels);
    }
    return predictions;
}

double average_overlap(const std::vector<std::vector<Label>>& lhs,
                       const std::vector<std::vector<Label>>& rhs,
                       uint64_t k) {
    if (lhs.size() != rhs.size() || lhs.empty()) {
        throw std::runtime_error("Cannot compare result sets with different/empty query counts");
    }
    uint64_t hits = 0;
    for (size_t query_id = 0; query_id < lhs.size(); ++query_id) {
        std::unordered_set<Label> expected(rhs[query_id].begin(), rhs[query_id].end());
        for (Label label : lhs[query_id]) {
            hits += expected.count(label);
        }
    }
    return static_cast<double>(hits) / static_cast<double>(lhs.size() * k);
}

struct EvalResult {
    std::string precision;
    double recall = 0.0;
    double recall_loss_vs_fp32 = 0.0;
    double overlap_vs_fp32 = 0.0;
    double seconds = 0.0;
    double average_ms = 0.0;
    double qps = 0.0;
};

void write_results_csv(const fs::path& output_path,
                       const std::vector<EvalResult>& results,
                       uint64_t k) {
    ensure_parent_directory(output_path);
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("Cannot create result CSV: " + output_path.string());
    }
    output << "precision,recall_at_" << k
           << ",recall_loss_vs_fp32,overlap_vs_fp32,total_query_s,avg_query_ms,qps\n";
    output << std::setprecision(10);
    for (const auto& result : results) {
        output << result.precision << ',' << result.recall << ','
               << result.recall_loss_vs_fp32 << ',' << result.overlap_vs_fp32 << ','
               << result.seconds << ',' << result.average_ms << ',' << result.qps << '\n';
    }
    if (!output) {
        throw std::runtime_error("Failed while writing result CSV: " + output_path.string());
    }
}

void run_evaluate(const Options& options) {
    options.reject_unknown({"help", "base", "queries", "index", "normalize",
                            "query-limit", "query-offset", "k", "ef", "threads",
                            "batch-size", "precisions", "output"});
    if (options.has("help")) {
        print_usage(std::cout);
        return;
    }

    const fs::path base_path = options.require("base");
    const fs::path query_path = options.require("queries");
    const fs::path index_path = options.require("index");
    const FbinHeader base_header = read_fbin_header(base_path);
    const FbinHeader query_header = read_fbin_header(query_path);
    const IndexHeader index_header = read_index_header(index_path);
    const uint64_t index_dim = infer_index_dim(index_header);
    const bool normalize = options.get_bool("normalize", true);
    const uint64_t query_limit = options.get_u64("query-limit", 10);
    const uint64_t query_offset = options.get_u64("query-offset", 0);
    const uint64_t k = options.get_u64("k", 10);
    const uint64_t ef = options.get_u64("ef", 100);
    const uint64_t batch_size = options.get_u64("batch-size", 100000);
    const int threads = checked_thread_count(options.get_u64("threads", 1));
    const std::vector<PrecisionConfig> precisions = parse_precisions(options.get(
        "precisions", "fp32,fp16_true,fp16_fpma,fp8_e4m3,int16,int8"));

    if (base_header.dim != query_header.dim || base_header.dim != index_dim) {
        throw std::runtime_error("Base, query, and index dimensions do not match");
    }
    if (index_header.current_elements > base_header.rows) {
        throw std::runtime_error("Index contains more elements than the provided base FBIN");
    }
    if (query_limit == 0 || query_offset > query_header.rows ||
        query_limit > query_header.rows - query_offset) {
        throw std::runtime_error("Invalid query range");
    }
    if (k == 0 || k > index_header.current_elements || ef < k || batch_size == 0) {
        throw std::runtime_error("Require 1 <= k <= index size, ef >= k, and batch-size > 0");
    }

    omp_set_dynamic(0);
    std::vector<float> queries =
        read_fbin_rows(query_path, query_header, query_offset, query_limit);
    if (normalize) {
        normalize_vectors(queries, query_limit, query_header.dim, threads);
    }

    std::cout << "[evaluate] base=" << fs::absolute(base_path) << '\n'
              << "[evaluate] queries=" << fs::absolute(query_path) << '\n'
              << "[evaluate] index=" << fs::absolute(index_path) << '\n'
              << "[evaluate] indexed_rows=" << index_header.current_elements
              << " dim=" << index_dim << " queries=" << query_limit
              << " query_offset=" << query_offset << '\n'
              << "[evaluate] normalize=" << (normalize ? "true" : "false")
              << " k=" << k << " ef=" << ef << " threads=" << threads
              << " batch_size=" << batch_size << std::endl;

    const auto gt_started = Clock::now();
    const auto ground_truth = compute_exact_ground_truth(
        base_path, base_header, VectorFormat::FBIN, index_header.current_elements,
        queries, query_limit, k, batch_size, normalize, threads, nullptr);
    const double gt_seconds =
        std::chrono::duration<double>(Clock::now() - gt_started).count();
    std::cout << "[evaluate] exact_gt_s=" << std::fixed << std::setprecision(6)
              << gt_seconds << std::endl;

    hnswlib::L2SpaceDynamicPrecision space(index_dim, hnswlib::PrecisionType::FP32);
    hnswlib::HierarchicalNSW<float> index(&space, index_path.string());
    index.setEf(ef);

    std::vector<EvalResult> results;
    std::vector<std::vector<Label>> fp32_predictions;
    double fp32_recall = 0.0;
    for (const auto& precision : precisions) {
        space.set_precision(precision.type);
        index.fstdistfunc_ = space.get_dist_func();
        index.dist_func_param_ = space.get_dist_func_param();

        const auto query_started = Clock::now();
        auto predictions =
            search_index(index, queries, query_limit, index_dim, k, threads);
        const double seconds =
            std::chrono::duration<double>(Clock::now() - query_started).count();
        const double recall = average_overlap(predictions, ground_truth, k);

        if (precision.name == "fp32") {
            fp32_predictions = predictions;
            fp32_recall = recall;
        }
        const double fp32_overlap =
            precision.name == "fp32" ? 1.0 : average_overlap(predictions, fp32_predictions, k);
        EvalResult result{
            precision.name,
            recall,
            fp32_recall - recall,
            fp32_overlap,
            seconds,
            seconds * 1000.0 / static_cast<double>(query_limit),
            static_cast<double>(query_limit) / seconds,
        };
        results.push_back(result);
        std::cout << "RESULT command=evaluate precision=" << result.precision
                  << " recall@" << k << '=' << std::fixed << std::setprecision(6)
                  << result.recall
                  << " recall_loss_vs_fp32=" << result.recall_loss_vs_fp32
                  << " overlap_vs_fp32=" << result.overlap_vs_fp32
                  << " total_query_s=" << result.seconds
                  << " avg_query_ms=" << result.average_ms
                  << " qps=" << result.qps << std::endl;
    }

    if (options.has("output")) {
        const fs::path output_path = options.require("output");
        write_results_csv(output_path, results, k);
        std::cout << "[evaluate] result_csv=" << fs::absolute(output_path) << std::endl;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(std::cerr);
            return 2;
        }
        const std::string command = argv[1];
        if (command == "--help" || command == "-h" || command == "help") {
            print_usage(std::cout);
            return 0;
        }

        const Options options(argc, argv, 2);
        if (command == "build") {
            run_build(options);
        } else if (command == "prepare") {
            run_prepare(options);
        } else if (command == "inspect") {
            run_inspect(options);
        } else if (command == "evaluate") {
            run_evaluate(options);
        } else {
            throw std::runtime_error("Unknown command: " + command);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
