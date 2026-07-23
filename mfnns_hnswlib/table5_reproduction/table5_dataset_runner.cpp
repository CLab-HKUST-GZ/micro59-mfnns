#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <omp.h>

#include "hnswlib/hnswalg.h"
#include "hnswlib/space_l2_dynamic_precision_et.h"

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path cache_root;
    fs::path index_root;
    std::string dataset;
    std::string variant;
    fs::path output;
    size_t query_limit = 1000;
    size_t k = 10;
    size_t ef = 500;
    int threads = 8;
    double risk_ratio_upper = 1.0073;
};

template <typename T>
struct Matrix {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<T> values;
};

struct SearchResult {
    double recall = 0.0;
    double et_ratio_pct = 0.0;
    long et_count = 0;
    long accepted_count = 0;
    long rejected_count = 0;
    double elapsed_seconds = 0.0;
};

struct BoundaryResult {
    uint64_t comparisons = 0;
    uint64_t risk_updates = 0;
    uint64_t flips = 0;
    double risk_update_pct = 0.0;
    double flip_ratio_pct = 0.0;
    double elapsed_seconds = 0.0;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

size_t parse_size(const std::string& value, const std::string& name) {
    size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &consumed);
    } catch (const std::exception&) {
        fail("Invalid " + name + ": " + value);
    }
    if (consumed != value.size() || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        fail("Invalid " + name + ": " + value);
    }
    return static_cast<size_t>(parsed);
}

double parse_double(const std::string& value, const std::string& name) {
    size_t consumed = 0;
    double parsed = 0.0;
    try {
        parsed = std::stod(value, &consumed);
    } catch (const std::exception&) {
        fail("Invalid " + name + ": " + value);
    }
    if (consumed != value.size()) {
        fail("Invalid " + name + ": " + value);
    }
    return parsed;
}

void print_usage(std::ostream& out) {
    out
        << "Usage: table5_dataset_runner [options]\n\n"
        << "Required:\n"
        << "  --cache-root PATH   Root containing cached queries and ground truth\n"
        << "  --index-root PATH   Root containing dataset/variant HNSW indexes\n"
        << "  --dataset NAME      Dataset cache name\n"
        << "  --variant NAME      Dataset variant\n"
        << "  --output PATH       One-row detail CSV output\n\n"
        << "Table 5 defaults:\n"
        << "  --query-limit N     Number of cached queries (default: 1000)\n"
        << "  --k N               Recall K (default: 10)\n"
        << "  --ef N              efSearch (default: 500)\n"
        << "  --threads N         Recall-search threads (default: 8)\n"
        << "  --risk-ratio X      Risk upper ratio (default: 1.0073)\n";
}

Options parse_options(int argc, char** argv) {
    std::map<std::string, std::string> values;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage(std::cout);
            std::exit(0);
        }
        if (arg.rfind("--", 0) != 0 || i + 1 >= argc) {
            fail("Expected --option value, got: " + arg);
        }
        values[arg.substr(2)] = argv[++i];
    }

    for (const auto& required :
         {"cache-root", "index-root", "dataset", "variant", "output"}) {
        if (values.find(required) == values.end()) {
            fail("Missing required option --" + std::string(required));
        }
    }

    Options options;
    options.cache_root = values.at("cache-root");
    options.index_root = values.at("index-root");
    options.dataset = values.at("dataset");
    options.variant = values.at("variant");
    options.output = values.at("output");
    if (values.count("query-limit")) {
        options.query_limit = parse_size(values.at("query-limit"), "query-limit");
    }
    if (values.count("k")) {
        options.k = parse_size(values.at("k"), "k");
    }
    if (values.count("ef")) {
        options.ef = parse_size(values.at("ef"), "ef");
    }
    if (values.count("threads")) {
        const size_t parsed = parse_size(values.at("threads"), "threads");
        if (parsed > static_cast<size_t>(std::numeric_limits<int>::max())) {
            fail("threads is too large");
        }
        options.threads = static_cast<int>(parsed);
    }
    if (values.count("risk-ratio")) {
        options.risk_ratio_upper = parse_double(values.at("risk-ratio"), "risk-ratio");
    }
    if (options.k > options.ef) {
        fail("k must not exceed ef");
    }
    if (!(options.risk_ratio_upper > 1.0)) {
        fail("risk-ratio must be greater than 1");
    }
    return options;
}

template <typename T>
Matrix<T> read_matrix(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail("Cannot open matrix: " + path.string());
    }

    Matrix<T> matrix;
    input.read(reinterpret_cast<char*>(&matrix.rows), sizeof(matrix.rows));
    input.read(reinterpret_cast<char*>(&matrix.cols), sizeof(matrix.cols));
    if (!input || matrix.rows <= 0 || matrix.cols <= 0) {
        fail("Invalid matrix header: " + path.string());
    }

    const uint64_t count =
        static_cast<uint64_t>(matrix.rows) * static_cast<uint64_t>(matrix.cols);
    const uint64_t expected =
        2 * sizeof(int32_t) + count * static_cast<uint64_t>(sizeof(T));
    if (fs::file_size(path) != expected) {
        fail("Matrix size does not match header: " + path.string());
    }
    if (count > std::numeric_limits<size_t>::max()) {
        fail("Matrix is too large: " + path.string());
    }

    matrix.values.resize(static_cast<size_t>(count));
    input.read(
        reinterpret_cast<char*>(matrix.values.data()),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) {
        fail("Cannot read matrix payload: " + path.string());
    }
    return matrix;
}

void set_precision(
    hnswlib::L2SpaceDynamicPrecisionET& space,
    hnswlib::HierarchicalNSW<float>& index,
    hnswlib::PrecisionType precision) {
    space.set_precision(precision);
    index.fstdistfunc_ = space.get_dist_func();
    index.dist_func_param_ = space.get_dist_func_param();
}

SearchResult run_search_mode(
    const Matrix<float>& queries,
    const Matrix<int32_t>& ground_truth,
    size_t query_count,
    size_t k,
    int threads,
    hnswlib::PrecisionType precision,
    bool dual_queue_et,
    hnswlib::L2SpaceDynamicPrecisionET& space,
    hnswlib::HierarchicalNSW<float>& index) {
    set_precision(space, index, precision);
    space.set_et_enabled(false);
    index.use_dual_queue_et_ = dual_queue_et;
    index.set_fpma_error_logging(false);
    index.set_breakdown_timing(true);
    index.reset_breakdown_timing();

    long long hits = 0;
    const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static) num_threads(threads) reduction(+ : hits)
    for (long long query_id = 0;
         query_id < static_cast<long long>(query_count);
         ++query_id) {
        const float* query =
            queries.values.data() + static_cast<size_t>(query_id) * queries.cols;
        auto result = index.searchKnn(query, k);
        while (!result.empty()) {
            const auto label = result.top().second;
            result.pop();
            const int32_t* gt =
                ground_truth.values.data() +
                static_cast<size_t>(query_id) * ground_truth.cols;
            for (size_t rank = 0; rank < k; ++rank) {
                if (label == static_cast<hnswlib::labeltype>(gt[rank])) {
                    ++hits;
                    break;
                }
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();

    const auto stats = index.get_breakdown_stats();
    SearchResult result;
    result.recall =
        static_cast<double>(hits) / static_cast<double>(query_count * k);
    result.accepted_count = std::get<4>(stats);
    result.rejected_count = std::get<5>(stats);
    result.et_count = std::get<6>(stats);
    const long total = result.accepted_count + result.rejected_count;
    result.et_ratio_pct =
        total > 0 ? 100.0 * static_cast<double>(result.et_count) / total : 0.0;
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    index.set_breakdown_timing(false);
    return result;
}

BoundaryResult run_boundary_analysis(
    const Matrix<float>& queries,
    size_t query_count,
    size_t k,
    double risk_ratio_upper,
    hnswlib::L2SpaceDynamicPrecisionET& space,
    hnswlib::HierarchicalNSW<float>& index) {
    set_precision(space, index, hnswlib::PrecisionType::FP16_FPMA);
    space.set_et_enabled(false);
    index.use_dual_queue_et_ = false;
    index.set_breakdown_timing(false);
    index.set_fpma_error_logging(true);

    const auto start = std::chrono::steady_clock::now();
    for (size_t query_id = 0; query_id < query_count; ++query_id) {
        const float* query = queries.values.data() + query_id * queries.cols;
        index.searchKnnWithQueryId(
            query, k, nullptr, static_cast<long long>(query_id));
    }
    const auto end = std::chrono::steady_clock::now();

    BoundaryResult result;
    result.comparisons = index.fpma_error_log_.size();
    for (const auto& entry : index.fpma_error_log_) {
        if (entry.exact_accept != entry.fpma_accept) {
            ++result.flips;
        }
        const double boundary = entry.boundary_exact;
        const double candidate = entry.candidate_exact;
        if (boundary > 0.0 && candidate > 0.0 &&
            boundary > candidate &&
            boundary / candidate < risk_ratio_upper) {
            ++result.risk_updates;
        }
    }
    if (result.comparisons > 0) {
        result.risk_update_pct =
            100.0 * static_cast<double>(result.risk_updates) /
            static_cast<double>(result.comparisons);
        result.flip_ratio_pct =
            100.0 * static_cast<double>(result.flips) /
            static_cast<double>(result.comparisons);
    }
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    index.set_fpma_error_logging(false);
    return result;
}

void write_result(
    const Options& options,
    size_t query_count,
    size_t dimension,
    size_t index_count,
    const SearchResult& fp16,
    const SearchResult& fpma,
    const SearchResult& fp16_et,
    const SearchResult& fpma_et,
    const BoundaryResult& boundary) {
    if (!options.output.parent_path().empty()) {
        fs::create_directories(options.output.parent_path());
    }
    const fs::path temporary = options.output.string() + ".tmp";
    std::ofstream output(temporary);
    if (!output) {
        fail("Cannot create output: " + temporary.string());
    }
    output
        << "dataset,variant,index_count,dimension,query_count,k,ef_search,"
        << "recall_threads,risk_ratio_upper,"
        << "fp16_recall,fpma_recall,fp16_et_recall,fp16_et_ratio_pct,"
        << "fpma_et_recall,fpma_et_ratio_pct,"
        << "risk_update_pct,flip_ratio_pct,boundary_comparisons,"
        << "risk_update_count,flip_count,"
        << "fp16_seconds,fpma_seconds,fp16_et_seconds,fpma_et_seconds,"
        << "boundary_seconds\n";
    output << std::setprecision(12)
           << options.dataset << ','
           << options.variant << ','
           << index_count << ','
           << dimension << ','
           << query_count << ','
           << options.k << ','
           << options.ef << ','
           << options.threads << ','
           << options.risk_ratio_upper << ','
           << fp16.recall << ','
           << fpma.recall << ','
           << fp16_et.recall << ','
           << fp16_et.et_ratio_pct << ','
           << fpma_et.recall << ','
           << fpma_et.et_ratio_pct << ','
           << boundary.risk_update_pct << ','
           << boundary.flip_ratio_pct << ','
           << boundary.comparisons << ','
           << boundary.risk_updates << ','
           << boundary.flips << ','
           << fp16.elapsed_seconds << ','
           << fpma.elapsed_seconds << ','
           << fp16_et.elapsed_seconds << ','
           << fpma_et.elapsed_seconds << ','
           << boundary.elapsed_seconds << '\n';
    output.close();
    if (!output) {
        fail("Failed while writing output: " + temporary.string());
    }
    fs::rename(temporary, options.output);
}

void print_mode(const std::string& name, const SearchResult& result) {
    std::cout << "  " << std::left << std::setw(10) << name
              << " recall=" << std::fixed << std::setprecision(4)
              << result.recall
              << " et_ratio=" << std::setprecision(2)
              << result.et_ratio_pct << "%"
              << " seconds=" << std::setprecision(3)
              << result.elapsed_seconds << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const fs::path cache_dir =
            options.cache_root / options.dataset / options.variant;
        const fs::path index_dir =
            options.index_root / options.dataset / options.variant;
        const fs::path index_path = index_dir / "hnsw_index_M32_ef100.bin";
        const fs::path query_path =
            cache_dir / "query_vectors_n1000_seed42.bin";
        const fs::path gt_path =
            cache_dir /
            ("gt_labels_topk" + std::to_string(options.k) +
             "_n1000_seed42.bin");

        if (!fs::exists(index_path)) {
            fail("Missing index: " + index_path.string());
        }
        const Matrix<float> queries = read_matrix<float>(query_path);
        const Matrix<int32_t> ground_truth = read_matrix<int32_t>(gt_path);
        if (queries.rows != ground_truth.rows) {
            fail("Query and ground-truth row counts differ");
        }
        if (ground_truth.cols < static_cast<int32_t>(options.k)) {
            fail("Ground truth has fewer than k columns");
        }
        if (options.query_limit > static_cast<size_t>(queries.rows)) {
            fail("query-limit exceeds cached query count");
        }

        hnswlib::L2SpaceDynamicPrecisionET space(
            static_cast<size_t>(queries.cols), hnswlib::PrecisionType::FP32);
        hnswlib::HierarchicalNSW<float> index(&space, index_path.string());
        index.setEf(options.ef);
        if (index.data_size_ !=
            static_cast<size_t>(queries.cols) * sizeof(float)) {
            fail("Index dimension differs from cached queries");
        }

        std::cout << "[table5] dataset=" << options.dataset << '/'
                  << options.variant
                  << " index_count=" << index.cur_element_count
                  << " dimension=" << queries.cols
                  << " queries=" << options.query_limit
                  << " k=" << options.k
                  << " ef=" << options.ef << '\n';

        const SearchResult fp16 = run_search_mode(
            queries, ground_truth, options.query_limit, options.k,
            options.threads, hnswlib::PrecisionType::FP16_TRUE, false,
            space, index);
        print_mode("FP16", fp16);
        const SearchResult fpma = run_search_mode(
            queries, ground_truth, options.query_limit, options.k,
            options.threads, hnswlib::PrecisionType::FP16_FPMA, false,
            space, index);
        print_mode("FPMA", fpma);
        const SearchResult fp16_et = run_search_mode(
            queries, ground_truth, options.query_limit, options.k,
            options.threads, hnswlib::PrecisionType::FP16_TRUE, true,
            space, index);
        print_mode("FP16+ET", fp16_et);
        const SearchResult fpma_et = run_search_mode(
            queries, ground_truth, options.query_limit, options.k,
            options.threads, hnswlib::PrecisionType::FP16_FPMA, true,
            space, index);
        print_mode("FPMA+ET", fpma_et);

        const BoundaryResult boundary = run_boundary_analysis(
            queries, options.query_limit, options.k,
            options.risk_ratio_upper, space, index);
        std::cout << "  boundary comparisons=" << boundary.comparisons
                  << " risk_update=" << std::fixed << std::setprecision(4)
                  << boundary.risk_update_pct << "%"
                  << " flip_ratio=" << std::setprecision(6)
                  << boundary.flip_ratio_pct << "%"
                  << " seconds=" << std::setprecision(3)
                  << boundary.elapsed_seconds << '\n';

        write_result(
            options, options.query_limit, static_cast<size_t>(queries.cols),
            index.cur_element_count, fp16, fpma, fp16_et, fpma_et, boundary);
        std::cout << "[table5] output=" << fs::absolute(options.output) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
