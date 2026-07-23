#include <hnswlib/space_l2_dynamic_precision_et.h>

#include <algorithm>
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
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
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

void normalize_vectors(std::vector<float>& data, uint64_t rows, uint64_t dim, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
        float norm_squared = 0.0f;
        const size_t offset = static_cast<size_t>(row) * dim;
        for (uint64_t col = 0; col < dim; ++col) {
            const float value = data[offset + col];
            norm_squared += value * value;
        }
        const float norm = std::sqrt(std::max(norm_squared, 1.0e-24f));
        for (uint64_t col = 0; col < dim; ++col) {
            data[offset + col] /= norm;
        }
    }
}

void ensure_parent_directory(const fs::path& path) {
    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
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
        << "mfnns_hnsw_tool: build, inspect, and evaluate MFNNS HNSW indexes\n\n"
        << "Usage:\n"
        << "  mfnns_hnsw_tool build --base BASE.fbin --index OUT.bin [options]\n"
        << "  mfnns_hnsw_tool inspect --index INDEX.bin\n"
        << "  mfnns_hnsw_tool evaluate --base BASE.fbin --queries QUERY.fbin "
           "--index INDEX.bin [options]\n\n"
        << "Build options:\n"
        << "  --normalize 0|1          L2-normalize each vector (default: 1)\n"
        << "  --limit N                Build the first N rows (default: all)\n"
        << "  --m N                    HNSW M (default: 16)\n"
        << "  --ef-construction N      HNSW ef_construction (default: 500)\n"
        << "  --threads N              OpenMP normalization threads (default: 1)\n"
        << "                           HNSW insertion is intentionally serialized\n"
        << "  --batch-size N           Streaming batch rows (default: 100000)\n"
        << "  --seed N                 HNSW random seed (default: 100)\n"
        << "  --force 0|1              Replace an existing output index (default: 0)\n\n"
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
    options.reject_unknown({"help", "base", "index", "normalize", "limit", "m",
                            "ef-construction", "threads", "batch-size", "seed", "force"});
    if (options.has("help")) {
        print_usage(std::cout);
        return;
    }

    const fs::path base_path = options.require("base");
    const fs::path index_path = options.require("index");
    const FbinHeader base_header = read_fbin_header(base_path);
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
              << "[build] shape=" << rows << "x" << base_header.dim << '\n'
              << "[build] normalize=" << (normalize ? "true" : "false") << '\n'
              << "[build] M=" << m << " ef_construction=" << ef_construction
              << " seed=" << seed << " preprocess_threads=" << threads
              << " insertion_threads=1"
              << " batch_size=" << batch_size << '\n'
              << "[build] output=" << fs::absolute(index_path) << std::endl;

    omp_set_dynamic(0);
    hnswlib::L2SpaceDynamicPrecision space(base_header.dim, hnswlib::PrecisionType::FP32);
    hnswlib::HierarchicalNSW<float> index(
        &space, rows, m, ef_construction, static_cast<size_t>(seed));

    const auto started = Clock::now();
    uint64_t inserted = 0;
    while (inserted < rows) {
        const uint64_t count = std::min(batch_size, rows - inserted);
        std::vector<float> batch = read_fbin_rows(base_path, base_header, inserted, count);
        if (normalize) {
            normalize_vectors(batch, count, base_header.dim, threads);
        }

        for (uint64_t row = 0; row < count; ++row) {
            index.addPoint(batch.data() + row * base_header.dim,
                           static_cast<Label>(inserted + row));
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

    const double total_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    std::cout << "RESULT command=build rows=" << rows
              << " dim=" << base_header.dim
              << " M=" << m
              << " ef_construction=" << ef_construction
              << " preprocess_threads=" << threads
              << " insertion_threads=1"
              << " total_s=" << std::fixed << std::setprecision(6) << total_seconds
              << " save_s=" << save_seconds
              << " index_bytes=" << fs::file_size(index_path) << std::endl;
}

float squared_l2(const float* lhs, const float* rhs, uint64_t dim) {
    float distance = 0.0f;
#pragma omp simd reduction(+ : distance)
    for (uint64_t col = 0; col < dim; ++col) {
        const float delta = lhs[col] - rhs[col];
        distance += delta * delta;
    }
    return distance;
}

using DistanceLabel = std::pair<float, Label>;
using MaxHeap = std::priority_queue<DistanceLabel>;

std::vector<std::vector<Label>> compute_exact_ground_truth(
    const fs::path& base_path,
    const FbinHeader& base_header,
    uint64_t rows,
    const std::vector<float>& queries,
    uint64_t query_count,
    uint64_t k,
    uint64_t batch_size,
    bool normalize,
    int threads) {
    std::vector<MaxHeap> heaps(query_count);
    uint64_t processed = 0;
    const auto started = Clock::now();
    while (processed < rows) {
        const uint64_t count = std::min(batch_size, rows - processed);
        std::vector<float> batch = read_fbin_rows(base_path, base_header, processed, count);
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
                    squared_l2(query, point, base_header.dim),
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
        base_path, base_header, index_header.current_elements, queries, query_limit, k,
        batch_size, normalize, threads);
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
