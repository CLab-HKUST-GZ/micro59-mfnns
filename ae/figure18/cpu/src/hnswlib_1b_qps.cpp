#include "hnswlib/hnswlib.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

#include <omp.h>

namespace {

struct Config {
    std::string dataset = "unknown";
    std::string index_path;
    std::string query_path;
    std::string gt_path;
    std::string output_path;
    int k = 10;
    int limit_queries = 0;
    int warmup_queries = 0;
    int query_repeats = 1;
    int trials_per_ef = 1;
    std::string omp_schedule = "static";
    int omp_chunk = 64;
    std::vector<int> ef_list{10, 20, 40, 60, 80, 100, 150, 200, 300, 500};
    std::vector<int> thread_list{std::max(1, omp_get_max_threads())};
};

struct FloatMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;
};

struct IntMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> data;
};

struct Result {
    std::string dataset;
    int nq = 0;
    int query_repeats = 1;
    long long query_ops = 0;
    int dim = 0;
    int k = 0;
    int ef = 0;
    int trial = 1;
    int threads = 0;
    int warmup = 0;
    std::string omp_schedule;
    int omp_chunk = 0;
    double load_s = 0.0;
    double search_s = 0.0;
    double qps = 0.0;
    double avg_ms = 0.0;
    double recall = 0.0;
    double avg_results = 0.0;
    double rss_gb = 0.0;
    double peak_rss_gb = 0.0;
};

[[noreturn]] void die_usage(const std::string& msg) {
    std::cerr << "ERROR: " << msg << "\n\n"
              << "Usage: hnswlib_1b_qps --dataset NAME --index INDEX --query QUERY_CACHE --gt GT_CACHE "
              << "[--k 10] [--ef-list 10,20,50] [--threads-list 96] [--limit-queries N] "
              << "[--warmup-queries N] [--query-repeats N] [--trials-per-ef N] [--omp-schedule static] "
              << "[--omp-chunk 64] [--output OUT.tsv]\n";
    std::exit(2);
}

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    size_t start = 0;
    while (start <= text.size()) {
        size_t comma = text.find(',', start);
        std::string token = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            values.push_back(std::stoi(token));
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    if (values.empty()) {
        throw std::runtime_error("empty integer list: " + text);
    }
    return values;
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) die_usage("missing value for " + name);
            return argv[++i];
        };

        if (arg == "--dataset") {
            cfg.dataset = need_value(arg);
        } else if (arg == "--index") {
            cfg.index_path = need_value(arg);
        } else if (arg == "--query") {
            cfg.query_path = need_value(arg);
        } else if (arg == "--gt") {
            cfg.gt_path = need_value(arg);
        } else if (arg == "--output") {
            cfg.output_path = need_value(arg);
        } else if (arg == "--k") {
            cfg.k = std::stoi(need_value(arg));
        } else if (arg == "--limit-queries") {
            cfg.limit_queries = std::stoi(need_value(arg));
        } else if (arg == "--warmup-queries") {
            cfg.warmup_queries = std::stoi(need_value(arg));
        } else if (arg == "--query-repeats") {
            cfg.query_repeats = std::stoi(need_value(arg));
        } else if (arg == "--trials-per-ef") {
            cfg.trials_per_ef = std::stoi(need_value(arg));
        } else if (arg == "--omp-schedule") {
            cfg.omp_schedule = need_value(arg);
        } else if (arg == "--omp-chunk") {
            cfg.omp_chunk = std::stoi(need_value(arg));
        } else if (arg == "--ef-list") {
            cfg.ef_list = parse_int_list(need_value(arg));
        } else if (arg == "--threads-list") {
            cfg.thread_list = parse_int_list(need_value(arg));
        } else if (arg == "--help" || arg == "-h") {
            die_usage("help requested");
        } else {
            die_usage("unknown argument: " + arg);
        }
    }

    if (cfg.index_path.empty()) die_usage("--index is required");
    if (cfg.query_path.empty()) die_usage("--query is required");
    if (cfg.gt_path.empty()) die_usage("--gt is required");
    if (cfg.k <= 0) die_usage("--k must be positive");
    if (cfg.limit_queries < 0) die_usage("--limit-queries must be non-negative");
    if (cfg.warmup_queries < 0) die_usage("--warmup-queries must be non-negative");
    if (cfg.query_repeats <= 0) die_usage("--query-repeats must be positive");
    if (cfg.trials_per_ef <= 0) die_usage("--trials-per-ef must be positive");
    if (cfg.omp_schedule != "static" && cfg.omp_schedule != "dynamic" && cfg.omp_schedule != "guided") {
        die_usage("--omp-schedule must be static, dynamic, or guided");
    }
    if (cfg.omp_chunk <= 0) die_usage("--omp-chunk must be positive");
    for (int ef : cfg.ef_list) {
        if (ef <= 0) die_usage("every ef must be positive");
    }
    for (int threads : cfg.thread_list) {
        if (threads <= 0) die_usage("thread counts must be positive");
    }
    return cfg;
}

omp_sched_t schedule_kind(const std::string& name) {
    if (name == "static") return omp_sched_static;
    if (name == "guided") return omp_sched_guided;
    return omp_sched_dynamic;
}

void require_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        throw std::runtime_error("cannot open file: " + path);
    }
}

FloatMatrix read_float_matrix(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open query cache: " + path);
    }

    FloatMatrix matrix;
    in.read(reinterpret_cast<char*>(&matrix.rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&matrix.cols), sizeof(int));
    if (!in || matrix.rows <= 0 || matrix.cols <= 0) {
        throw std::runtime_error("invalid float matrix header: " + path);
    }
    const size_t count = static_cast<size_t>(matrix.rows) * static_cast<size_t>(matrix.cols);
    matrix.data.resize(count);
    in.read(reinterpret_cast<char*>(matrix.data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in) {
        throw std::runtime_error("failed to read float matrix data: " + path);
    }
    return matrix;
}

IntMatrix read_int_matrix(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open GT cache: " + path);
    }

    IntMatrix matrix;
    in.read(reinterpret_cast<char*>(&matrix.rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&matrix.cols), sizeof(int));
    if (!in || matrix.rows <= 0 || matrix.cols <= 0) {
        throw std::runtime_error("invalid int matrix header: " + path);
    }
    const size_t count = static_cast<size_t>(matrix.rows) * static_cast<size_t>(matrix.cols);
    matrix.data.resize(count);
    in.read(reinterpret_cast<char*>(matrix.data.data()), static_cast<std::streamsize>(count * sizeof(int)));
    if (!in) {
        throw std::runtime_error("failed to read int matrix data: " + path);
    }
    return matrix;
}

double current_rss_gb() {
    std::ifstream statm("/proc/self/statm");
    long pages_total = 0;
    long pages_resident = 0;
    statm >> pages_total >> pages_resident;
    const long page_size = sysconf(_SC_PAGESIZE);
    return static_cast<double>(pages_resident) * static_cast<double>(page_size) /
           (1024.0 * 1024.0 * 1024.0);
}

double peak_rss_gb() {
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
}

bool label_in_gt(hnswlib::labeltype label, const int* gt_row, int k) {
    for (int i = 0; i < k; i++) {
        if (label == static_cast<hnswlib::labeltype>(gt_row[i])) {
            return true;
        }
    }
    return false;
}

std::priority_queue<std::pair<float, hnswlib::labeltype>>
search_knn_decoupled_ef(
    const hnswlib::HierarchicalNSW<float>& index,
    const void* query_data,
    size_t result_k,
    size_t ef_search) {
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
    if (index.cur_element_count == 0 || result_k == 0) return result;
    ef_search = std::max<size_t>(1, ef_search);

    using Table = hnswlib::tableint;
    using InternalQueue = std::priority_queue<
        std::pair<float, Table>,
        std::vector<std::pair<float, Table>>,
        hnswlib::HierarchicalNSW<float>::CompareByFirst>;

    Table curr_obj = index.enterpoint_node_;
    float curdist = index.fstdistfunc_(query_data, index.getDataByInternalId(curr_obj), index.dist_func_param_);

    for (int level = index.maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int* data = reinterpret_cast<unsigned int*>(index.get_linklist(curr_obj, level));
            int size = index.getListCount(data);
            Table* datal = reinterpret_cast<Table*>(data + 1);
            for (int i = 0; i < size; i++) {
                Table cand = datal[i];
                if (cand > index.max_elements_) {
                    throw std::runtime_error("cand error");
                }
                float d = index.fstdistfunc_(query_data, index.getDataByInternalId(cand), index.dist_func_param_);
                if (d < curdist) {
                    curdist = d;
                    curr_obj = cand;
                    changed = true;
                }
            }
        }
    }

    hnswlib::VisitedList* vl = index.visited_list_pool_->getFreeVisitedList();
    hnswlib::vl_type* visited_array = vl->mass;
    hnswlib::vl_type visited_array_tag = vl->curV;

    InternalQueue search_top;
    InternalQueue candidate_set;
    InternalQueue result_top;

    auto add_result = [&](float dist, Table id) {
        if (index.num_deleted_ && index.isMarkedDeleted(id)) {
            return;
        }
        result_top.emplace(dist, id);
        if (result_top.size() > result_k) {
            result_top.pop();
        }
    };

    float lower_bound = curdist;
    search_top.emplace(curdist, curr_obj);
    candidate_set.emplace(-curdist, curr_obj);
    add_result(curdist, curr_obj);
    visited_array[curr_obj] = visited_array_tag;

    while (!candidate_set.empty()) {
        const auto current_node_pair = candidate_set.top();
        const float candidate_dist = -current_node_pair.first;
        if (candidate_dist > lower_bound && search_top.size() >= ef_search) {
            break;
        }
        candidate_set.pop();

        Table current_node_id = current_node_pair.second;
        int* data = reinterpret_cast<int*>(index.get_linklist0(current_node_id));
        size_t size = index.getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));

#ifdef USE_SSE
        _mm_prefetch(reinterpret_cast<char*>(visited_array + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<char*>(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(index.data_level0_memory_ + (*(data + 1)) * index.size_data_per_element_ + index.offsetData_, _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<char*>(data + 2), _MM_HINT_T0);
#endif

        for (size_t j = 1; j <= size; j++) {
            Table candidate_id = static_cast<Table>(*(data + j));
#ifdef USE_SSE
            _mm_prefetch(reinterpret_cast<char*>(visited_array + *(data + j + 1)), _MM_HINT_T0);
            _mm_prefetch(index.data_level0_memory_ + (*(data + j + 1)) * index.size_data_per_element_ + index.offsetData_, _MM_HINT_T0);
#endif
            if (visited_array[candidate_id] == visited_array_tag) {
                continue;
            }
            visited_array[candidate_id] = visited_array_tag;

            char* curr_obj_data = index.getDataByInternalId(candidate_id);
            float dist = index.fstdistfunc_(query_data, curr_obj_data, index.dist_func_param_);
            add_result(dist, candidate_id);

            if (search_top.size() < ef_search || lower_bound > dist) {
                candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                _mm_prefetch(index.data_level0_memory_ + candidate_set.top().second * index.size_data_per_element_ +
                             index.offsetLevel0_, _MM_HINT_T0);
#endif
                search_top.emplace(dist, candidate_id);
                if (search_top.size() > ef_search) {
                    search_top.pop();
                }
                if (!search_top.empty()) {
                    lower_bound = search_top.top().first;
                }
            }
        }
    }

    index.visited_list_pool_->releaseVisitedList(vl);
    while (!result_top.empty()) {
        const auto item = result_top.top();
        result.emplace(item.first, index.getExternalLabel(item.second));
        result_top.pop();
    }
    return result;
}

std::priority_queue<std::pair<float, hnswlib::labeltype>>
search_index(
    const hnswlib::HierarchicalNSW<float>& index,
    const void* query_data,
    size_t result_k,
    size_t ef_search) {
    if (ef_search < result_k) {
        return search_knn_decoupled_ef(index, query_data, result_k, ef_search);
    }
    return index.searchKnn(query_data, result_k);
}

Result run_one(
    const Config& cfg,
    hnswlib::HierarchicalNSW<float>& index,
    const FloatMatrix& queries,
    const IntMatrix& gt,
    int nq,
    double load_s,
    int ef,
    int trial,
    int threads) {
    index.setEf(static_cast<size_t>(ef));
    omp_set_num_threads(threads);
    omp_set_schedule(schedule_kind(cfg.omp_schedule), cfg.omp_chunk);

    const int dim = queries.cols;
    const int k = cfg.k;
    const int warmup = cfg.warmup_queries;
    const long long query_ops = static_cast<long long>(nq) * static_cast<long long>(cfg.query_repeats);

    #pragma omp parallel num_threads(threads)
    {
    }

    if (warmup > 0) {
        #pragma omp parallel for schedule(runtime) num_threads(threads)
        for (int op = 0; op < warmup; op++) {
            const int qid = op % nq;
            auto result = search_index(index, queries.data.data() + static_cast<size_t>(qid) * dim, k, ef);
            while (!result.empty()) result.pop();
        }
    }

    long long correct = 0;
    long long returned = 0;
    const auto t0 = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(runtime) num_threads(threads) reduction(+:correct,returned)
    for (long long op = 0; op < query_ops; op++) {
        const int qid = static_cast<int>(op % nq);
        auto result = search_index(index, queries.data.data() + static_cast<size_t>(qid) * dim, k, ef);
        const int* gt_row = gt.data.data() + static_cast<size_t>(qid) * gt.cols;
        while (!result.empty()) {
            returned++;
            if (label_in_gt(result.top().second, gt_row, k)) {
                correct++;
            }
            result.pop();
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double search_s = std::chrono::duration<double>(t1 - t0).count();

    Result r;
    r.dataset = cfg.dataset;
    r.nq = nq;
    r.query_repeats = cfg.query_repeats;
    r.query_ops = query_ops;
    r.dim = dim;
    r.k = k;
    r.ef = ef;
    r.trial = trial;
    r.threads = threads;
    r.warmup = warmup;
    r.omp_schedule = cfg.omp_schedule;
    r.omp_chunk = cfg.omp_chunk;
    r.load_s = load_s;
    r.search_s = search_s;
    r.qps = search_s > 0.0 ? static_cast<double>(query_ops) / search_s : 0.0;
    r.avg_ms = search_s > 0.0 ? search_s * 1000.0 / static_cast<double>(query_ops) : 0.0;
    r.recall = static_cast<double>(correct) / static_cast<double>(query_ops * k);
    r.avg_results = static_cast<double>(returned) / static_cast<double>(query_ops);
    r.rss_gb = current_rss_gb();
    r.peak_rss_gb = peak_rss_gb();
    return r;
}

void write_header(std::ostream& out) {
    out << "dataset\tnq\tquery_repeats\tquery_ops\tdim\tk\tef\ttrial\tthreads\twarmup_queries\tomp_schedule\tomp_chunk\tindex_load_s\tsearch_s\tqps\tavg_ms_per_query\trecall\tavg_results_per_query\trss_gb\tpeak_rss_gb\n";
}

void write_result(std::ostream& out, const Result& r) {
    out << r.dataset << '\t'
        << r.nq << '\t'
        << r.query_repeats << '\t'
        << r.query_ops << '\t'
        << r.dim << '\t'
        << r.k << '\t'
        << r.ef << '\t'
        << r.trial << '\t'
        << r.threads << '\t'
        << r.warmup << '\t'
        << r.omp_schedule << '\t'
        << r.omp_chunk << '\t'
        << std::fixed << std::setprecision(3) << r.load_s << '\t'
        << std::fixed << std::setprecision(6) << r.search_s << '\t'
        << std::fixed << std::setprecision(3) << r.qps << '\t'
        << std::fixed << std::setprecision(6) << r.avg_ms << '\t'
        << std::fixed << std::setprecision(6) << r.recall << '\t'
        << std::fixed << std::setprecision(3) << r.avg_results << '\t'
        << std::fixed << std::setprecision(3) << r.rss_gb << '\t'
        << std::fixed << std::setprecision(3) << r.peak_rss_gb << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        require_file(cfg.index_path);
        require_file(cfg.query_path);
        require_file(cfg.gt_path);

        std::cerr << "[INFO] dataset=" << cfg.dataset << "\n";
        std::cerr << "[INFO] index=" << cfg.index_path << "\n";
        std::cerr << "[INFO] query=" << cfg.query_path << "\n";
        std::cerr << "[INFO] gt=" << cfg.gt_path << "\n";

        auto queries = read_float_matrix(cfg.query_path);
        auto gt = read_int_matrix(cfg.gt_path);
        if (queries.rows != gt.rows) {
            throw std::runtime_error("query rows and GT rows differ");
        }
        if (gt.cols < cfg.k) {
            throw std::runtime_error("GT topk is smaller than requested k");
        }
        const int nq = cfg.limit_queries > 0 ? std::min(cfg.limit_queries, queries.rows) : queries.rows;
        if (nq <= 0) {
            throw std::runtime_error("no queries selected");
        }

        std::cerr << "[INFO] loaded queries=(" << queries.rows << "," << queries.cols << ")"
                  << " gt=(" << gt.rows << "," << gt.cols << ")"
                  << " selected_nq=" << nq
                  << " query_repeats=" << cfg.query_repeats
                  << " trials_per_ef=" << cfg.trials_per_ef
                  << " query_ops=" << static_cast<long long>(nq) * cfg.query_repeats
                  << " omp_schedule=" << cfg.omp_schedule
                  << " omp_chunk=" << cfg.omp_chunk
                  << " k=" << cfg.k << "\n";

        hnswlib::L2Space space(static_cast<size_t>(queries.cols));
        const auto load_t0 = std::chrono::steady_clock::now();
        hnswlib::HierarchicalNSW<float> index(&space, cfg.index_path, false);
        const auto load_t1 = std::chrono::steady_clock::now();
        const double load_s = std::chrono::duration<double>(load_t1 - load_t0).count();
        std::cerr << "[INFO] index loaded in " << std::fixed << std::setprecision(3) << load_s
                  << " s, rss=" << current_rss_gb() << " GiB\n";

        std::ofstream out_file;
        std::ostream* out = &std::cout;
        if (!cfg.output_path.empty()) {
            out_file.open(cfg.output_path);
            if (!out_file.is_open()) {
                throw std::runtime_error("cannot open output file: " + cfg.output_path);
            }
            out = &out_file;
        }
        write_header(*out);
        std::cout << "# hnswlib_1b_qps results\n";
        write_header(std::cout);

        for (int threads : cfg.thread_list) {
            for (int ef : cfg.ef_list) {
                for (int trial = 1; trial <= cfg.trials_per_ef; trial++) {
                    std::cerr << "[RUN] dataset=" << cfg.dataset
                              << " nq=" << nq
                              << " k=" << cfg.k
                              << " ef=" << ef
                              << " trial=" << trial
                              << " threads=" << threads << "\n";
                    Result r = run_one(cfg, index, queries, gt, nq, load_s, ef, trial, threads);
                    write_result(*out, r);
                    if (out_file.is_open()) out_file.flush();
                    write_result(std::cout, r);
                    std::cout.flush();
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
