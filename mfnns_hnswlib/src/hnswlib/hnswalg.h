#pragma once

#include "early_termination_interface.h"
#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <tuple>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};
    SpaceInterface<dist_t> *space_{nullptr};  // ET support: keep space pointer

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;
    mutable std::mutex level_generator_lock_;
    mutable std::mutex update_probability_generator_lock_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    // Performance breakdown control and statistics
    bool enable_breakdown_timing_ = false;  // flag to enable fine-grained timing breakdown

    // Timing statistics (in nanoseconds, using atomic for thread safety)
    mutable std::atomic<long long> time_dist_accepted_ns{0};    // Time for distance computations that enter top candidates
    mutable std::atomic<long long> time_dist_rejected_ns{0};    // Time for distance computations that are rejected
    mutable std::atomic<long long> time_priority_queue_ns{0};   // Time for priority queue operations (push/pop/emplace)
    mutable std::atomic<long long> time_visited_check_ns{0};    // Time for visited array check operations

    // Counters for breakdown analysis
    mutable std::atomic<long> count_dist_accepted{0};           // Count of accepted distance computations
    mutable std::atomic<long> count_dist_rejected{0};           // Count of rejected distance computations
    mutable std::atomic<long> count_dual_queue_et{0};           // Count of candidates filtered by dual-queue ET

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    // Dual-queue ET mode (may affect recall but potentially improve performance)
    bool use_dual_queue_et_ = false;  // flag to enable dual-queue ET mode

    // Optional FPMA error logging for search boundary comparisons. Disabled by
    // default so normal search does not compute extra distances or write logs.
    struct FpmaBoundaryLogEntry {
        long long query_id;
        size_t step_id;
        size_t ef_search;
        size_t k;
        labeltype boundary_id;
        labeltype candidate_id;
        float boundary_exact;
        float candidate_exact;
        float boundary_fpma;
        float candidate_fpma;
        bool exact_accept;
        bool fpma_accept;
    };

    bool enable_fpma_error_logging_ = false;
    mutable std::mutex fpma_error_log_mutex_;
    mutable std::vector<FpmaBoundaryLogEntry> fpma_error_log_;

    static inline uint16_t fpma_log_fp32_to_fp16_bits(float val) {
        uint32_t f32;
        std::memcpy(&f32, &val, sizeof(float));

        uint32_t sign = (f32 >> 16) & 0x8000;
        int32_t f32_expo = (f32 >> 23) & 0xFF;
        uint32_t f32_mant = f32 & 0x7FFFFF;

        if (f32_expo == 0 && f32_mant == 0) {
            return static_cast<uint16_t>(sign);
        }
        if (f32_expo == 255) {
            if (f32_mant == 0) {
                return static_cast<uint16_t>(sign | 0x7C00);
            }
            return static_cast<uint16_t>(sign | 0x7C00 | (f32_mant >> 13));
        }

        int32_t fp16_expo = f32_expo - 127 + 15;
        if (fp16_expo <= 0) {
            if (fp16_expo < -10) {
                return static_cast<uint16_t>(sign);
            }
            uint32_t mant = (f32_mant | 0x800000) >> (1 - fp16_expo + 13);
            if (((f32_mant | 0x800000) >> (1 - fp16_expo + 12)) & 1) {
                mant++;
            }
            return static_cast<uint16_t>(sign | (mant & 0x3FF));
        }
        if (fp16_expo >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00);
        }

        uint32_t fp16_mant = f32_mant >> 13;
        uint32_t round_bit = (f32_mant >> 12) & 1;
        uint32_t sticky_bits = f32_mant & 0xFFF;
        if (round_bit && (sticky_bits || (fp16_mant & 1))) {
            fp16_mant++;
            if (fp16_mant >= 1024) {
                fp16_expo++;
                fp16_mant = 0;
                if (fp16_expo >= 31) {
                    return static_cast<uint16_t>(sign | 0x7C00);
                }
            }
        }
        return static_cast<uint16_t>(sign | (fp16_expo << 10) | fp16_mant);
    }

    static inline float fpma_log_fp16_bits_to_fp32(uint16_t fp16) {
        uint32_t sign = (fp16 & 0x8000) << 16;
        int32_t expo = (fp16 >> 10) & 0x1F;
        uint32_t mant = fp16 & 0x3FF;

        if (expo == 0 && mant == 0) {
            uint32_t result = sign;
            float f;
            std::memcpy(&f, &result, sizeof(float));
            return f;
        }
        if (expo == 0) {
            expo = 1;
        }

        int32_t f32_expo = expo - 15 + 127;
        uint32_t f32_mant = mant << 13;
        uint32_t f32 = sign | (f32_expo << 23) | f32_mant;
        float result;
        std::memcpy(&result, &f32, sizeof(float));
        return result;
    }

    static inline float fpma_log_fp16_round(float val) {
        return fpma_log_fp16_bits_to_fp32(fpma_log_fp32_to_fp16_bits(val));
    }

    static inline float fpma_log_fp16_fpma_square(float val) {
        static const int16_t square_approx_table[8] = {4, 36, 100, 158, 98, 50, 18, 2};
        uint16_t fp16 = fpma_log_fp32_to_fp16_bits(val);
        int32_t expo = (fp16 >> 10) & 0x1F;
        int32_t mant = fp16 & 0x3FF;

        if (expo == 0) {
            return 0.0f;
        }
        if (expo == 31) {
            return fpma_log_fp16_bits_to_fp32(static_cast<uint16_t>(0x7C00 | (mant & 0x3FF)));
        }

        int32_t result_expo = expo + expo - 15;
        int32_t mant_sum = mant + mant;
        if (mant_sum >= 1024) {
            result_expo++;
            mant_sum -= 1024;
        }

        int32_t result_mant = mant_sum + square_approx_table[(mant >> 7) & 0x07];
        if (result_mant >= 1024) {
            result_expo++;
            result_mant -= 1024;
        }

        if (result_expo <= 0) {
            return 0.0f;
        }
        if (result_expo >= 31) {
            return fpma_log_fp16_bits_to_fp32(0x7C00);
        }

        uint16_t result_fp16 = static_cast<uint16_t>((result_expo << 10) | (result_mant & 0x3FF));
        return fpma_log_fp16_bits_to_fp32(result_fp16);
    }

    inline std::pair<float, float> compute_fpma_error_distances(const void *query_data, const void *candidate_data) const {
        const float *query = static_cast<const float *>(query_data);
        const float *candidate = static_cast<const float *>(candidate_data);
        size_t dim = data_size_ / sizeof(float);
        float exact = 0.0f;
        float fpma = 0.0f;

        for (size_t i = 0; i < dim; i++) {
            float q = fpma_log_fp16_round(query[i]);
            float c = fpma_log_fp16_round(candidate[i]);
            float diff = fpma_log_fp16_round(q - c);
            exact += diff * diff;
            fpma += fpma_log_fp16_fpma_square(diff);
        }
        return std::make_pair(exact, fpma);
    }

    inline void maybe_log_fpma_boundary_comparison(
        const void *query_data,
        tableint boundary_internal_id,
        tableint candidate_internal_id,
        size_t ef_search,
        size_t k,
        long long query_id,
        size_t step_id) const {
        if (!enable_fpma_error_logging_) {
            return;
        }

        auto boundary_dists = compute_fpma_error_distances(query_data, getDataByInternalId(boundary_internal_id));
        auto candidate_dists = compute_fpma_error_distances(query_data, getDataByInternalId(candidate_internal_id));

        FpmaBoundaryLogEntry entry;
        entry.query_id = query_id;
        entry.step_id = step_id;
        entry.ef_search = ef_search;
        entry.k = k;
        entry.boundary_id = getExternalLabel(boundary_internal_id);
        entry.candidate_id = getExternalLabel(candidate_internal_id);
        entry.boundary_exact = boundary_dists.first;
        entry.boundary_fpma = boundary_dists.second;
        entry.candidate_exact = candidate_dists.first;
        entry.candidate_fpma = candidate_dists.second;
        entry.exact_accept = entry.candidate_exact < entry.boundary_exact;
        entry.fpma_accept = entry.candidate_fpma < entry.boundary_fpma;

        std::lock_guard<std::mutex> lock(fpma_error_log_mutex_);
        fpma_error_log_.push_back(entry);
    }


    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        space_ = s;  // ET support: save space pointer
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        std::lock_guard<std::mutex> lock(level_generator_lock_);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                // ET support: use dynamic threshold only when top_candidates is full
                // Count all candidates (including deleted) for accurate hardware statistics
                dist_t dist1;
                auto* et_space = dynamic_cast<L2EarlyTerminationInterface*>(space_);
                if (et_space != nullptr && et_space->get_et_enabled() && top_candidates.size() >= ef_construction_) {
                    // Get threshold from top_candidates (K-th smallest distance)
                    dist_t dynamic_threshold = top_candidates.top().first;
                    dist1 = et_space->dist_func_et(data_point, currObj1, dynamic_threshold);
                } else {
                    dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                }
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr,
        size_t k_for_et = 0,
        long long query_id = -1) const {  // k_for_et: top-k count for ET threshold (0 means use ef)
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        // Dual-queue ET mode: maintain a lower bound queue
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> lower_bound_candidates;
        bool dual_queue_mode = use_dual_queue_et_ && (k_for_et > 0);
        size_t fpma_error_step_id = 0;

        dist_t lowerBound;
        // Threshold for ET: the k-th distance of the true top-k
        size_t et_threshold_size = (k_for_et > 0 && k_for_et < ef) ? k_for_et : ef;
        dist_t topk_threshold = std::numeric_limits<dist_t>::max();
        if (bare_bone_search ||
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                // Timing: visited array check
                auto t_visited_start = enable_breakdown_timing_ ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point();

                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    if (enable_breakdown_timing_) {
                        auto t_visited_end = std::chrono::high_resolution_clock::now();
                        time_visited_check_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_visited_end - t_visited_start).count();
                    }

                    visited_array[candidate_id] = visited_array_tag;

                    // Skip deleted candidates early if not bare_bone_search
                    if (!bare_bone_search && isMarkedDeleted(candidate_id)) continue;
                    if (!bare_bone_search && isIdAllowed && !(*isIdAllowed)(getExternalLabel(candidate_id))) continue;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist;
                    dist_t lower_bound = 0.0f;
                    bool computed_distance = false;

                    // Timing: distance computation
                    auto t_dist_start = enable_breakdown_timing_ ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point();

                    auto* et_space = dynamic_cast<L2EarlyTerminationInterface*>(space_);

                    // ========== Dual-queue ET mode ==========
                    if (dual_queue_mode && et_space != nullptr && top_candidates.size() >= et_threshold_size) {
                        // First compute lower bound
                        lower_bound = et_space->compute_full_l1_lower_bound(data_point, currObj1);

                        // Check if lower bound already exceeds maximum in lower bound queue
                        // Note: ET check only when lower bound queue is full (size >= k)
                        if (lower_bound_candidates.size() >= et_threshold_size &&
                            lower_bound > lower_bound_candidates.top().first) {
                            // Lower bound is too large, early termination! No need to compute actual distance
                            count_dual_queue_et++;
                            computed_distance = false;
                            // Skip this candidate directly
                            if (enable_breakdown_timing_) {
                                auto t_dist_end = std::chrono::high_resolution_clock::now();
                                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_dist_end - t_dist_start).count();
                                time_dist_rejected_ns += duration;
                                count_dist_rejected++;
                            }
                            continue;  // ET succeeded, skip
                        }

                        // Lower bound passed, compute actual distance
                        dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                        computed_distance = true;
                    }
                    // ========== Single-queue ET mode (L1/L2 statistics) ==========
                    else if (et_space != nullptr && et_space->get_et_enabled() && top_candidates.size() >= et_threshold_size) {
                        dist = et_space->dist_func_et(data_point, currObj1, topk_threshold);
                        computed_distance = true;
                    }
                    // ========== No ET mode ==========
                    else {
                        dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                        computed_distance = true;
                    }

                    auto t_dist_end = enable_breakdown_timing_ ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point();

                    bool has_result_boundary = ef > 0 && top_candidates.size() >= ef && !top_candidates.empty();
                    if (enable_fpma_error_logging_ && computed_distance && has_result_boundary) {
                        tableint boundary_id = top_candidates.top().second;
                        maybe_log_fpma_boundary_comparison(
                            data_point,
                            boundary_id,
                            static_cast<tableint>(candidate_id),
                            ef,
                            k_for_et,
                            query_id,
                            fpma_error_step_id++);
                    }

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        // Distance computation ACCEPTED - record time
                        if (enable_breakdown_timing_) {
                            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_dist_end - t_dist_start).count();
                            time_dist_accepted_ns += duration;
                            count_dist_accepted++;
                        }

                        // Timing: priority queue operations
                        auto t_pq_start = enable_breakdown_timing_ ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point();

                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        // Already checked deleted/filter above, so directly add
                        top_candidates.emplace(dist, candidate_id);

                        // Dual-queue mode: add to lower bound queue simultaneously
                        if (dual_queue_mode && et_space != nullptr) {
                            // If lower bound not computed yet (e.g., direct distance computation when queue not full), compute now
                            if (lower_bound == 0.0f) {
                                lower_bound = et_space->compute_full_l1_lower_bound(data_point, currObj1);
                            }
                            lower_bound_candidates.emplace(lower_bound, candidate_id);
                        }

                        if (enable_breakdown_timing_) {
                            auto t_pq_end = std::chrono::high_resolution_clock::now();
                            time_priority_queue_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_pq_end - t_pq_start).count();
                        }
                        if (!bare_bone_search && stop_condition) {
                            stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            // Timing: priority queue pop operation
                            auto t_pop_start = enable_breakdown_timing_ ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point();

                            tableint id = top_candidates.top().second;
                            top_candidates.pop();

                            // Dual-queue mode: synchronously pop from lower bound queue
                            if (dual_queue_mode && !lower_bound_candidates.empty()) {
                                lower_bound_candidates.pop();
                            }

                            if (enable_breakdown_timing_) {
                                auto t_pop_end = std::chrono::high_resolution_clock::now();
                                time_priority_queue_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_pop_end - t_pop_start).count();
                            }
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;

                        // Update top-k threshold for ET
                        if (k_for_et > 0) {
                            size_t curr_size = top_candidates.size();
                            if (curr_size == et_threshold_size) {
                                // When exactly k candidates, top of heap is the k-th smallest (distance of k-th result)
                                topk_threshold = top_candidates.top().first;
                            } else if (curr_size > et_threshold_size) {
                                // When candidate count > k, need to find k-th smallest distance from heap
                                // Update strategy: smaller k, more frequent updates (threshold critical for ET)
                                bool should_update = false;
                                if (et_threshold_size <= 5) {
                                    // Very small k (1-5), update every time (most critical)
                                    should_update = true;
                                } else if (et_threshold_size <= 20) {
                                    // Medium k (6-20), update every 5 times
                                    should_update = (curr_size % 5 == 0);
                                } else {
                                    // Large k (>20), update when close to k or every 20 times
                                    should_update = (curr_size <= et_threshold_size + 10) || (curr_size % 20 == 0);
                                }

                                if (should_update) {
                                    // top_candidates is max-heap, we need to find k-th smallest distance
                                    std::vector<std::pair<dist_t, tableint>> temp_vec;
                                    auto temp_queue = top_candidates;
                                    while (!temp_queue.empty()) {
                                        temp_vec.push_back(temp_queue.top());
                                        temp_queue.pop();
                                    }
                                    // Partial sort in ascending order: smaller first, position k-1 is k-th smallest distance
                                    std::nth_element(temp_vec.begin(), temp_vec.begin() + et_threshold_size - 1, temp_vec.end(),
                                        [](const std::pair<dist_t, tableint>& a, const std::pair<dist_t, tableint>& b) {
                                            return a.first < b.first;  // Ascending: smaller distances first
                                        });
                                    topk_threshold = temp_vec[et_threshold_size - 1].first;
                                }
                            }
                        }
                    } else {
                        // Distance computation REJECTED - record time
                        if (enable_breakdown_timing_) {
                            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_dist_end - t_dist_start).count();
                            time_dist_rejected_ns += duration;
                            count_dist_rejected++;
                        }
                    }
                } else {
                    // Visited check: already visited
                    if (enable_breakdown_timing_) {
                        auto t_visited_end = std::chrono::high_resolution_clock::now();
                        time_visited_check_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_visited_end - t_visited_start).count();
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        space_ = s;  // ET support: save space pointer

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    *
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                {
                    std::lock_guard<std::mutex> lock(update_probability_generator_lock_);
                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;
                }

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnnWithQueryId(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr, long long query_id = -1) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        size_t ef_search = std::max(ef_, k);
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, ef_search, isIdAllowed, nullptr, k, query_id);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, ef_search, isIdAllowed, nullptr, k, query_id);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const override {
        return searchKnnWithQueryId(query_data, k, isIdAllowed, -1);
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }


    // ====================== Performance Breakdown Methods ======================

    /**
     * Enable or disable fine-grained performance breakdown timing
     */
    void set_breakdown_timing(bool enable) {
        enable_breakdown_timing_ = enable;
        if (enable) {
            // Reset all counters when enabling
            reset_breakdown_timing();
        }
    }

    /**
     * Check if breakdown timing is enabled
     */
    bool get_breakdown_timing() const {
        return enable_breakdown_timing_;
    }

    /**
     * Reset all performance breakdown statistics
     */
    void reset_breakdown_timing() {
        time_dist_accepted_ns = 0;
        time_dist_rejected_ns = 0;
        time_priority_queue_ns = 0;
        time_visited_check_ns = 0;
        count_dist_accepted = 0;
        count_dist_rejected = 0;
        count_dual_queue_et = 0;
    }

    /**
     * Get performance breakdown statistics
     * Returns: (time_dist_accepted_ns, time_dist_rejected_ns, time_priority_queue_ns, time_visited_check_ns,
     *           count_dist_accepted, count_dist_rejected, count_dual_queue_et)
     */
    std::tuple<long long, long long, long long, long long, long, long, long> get_breakdown_stats() const {
        return std::make_tuple(
            time_dist_accepted_ns.load(),
            time_dist_rejected_ns.load(),
            time_priority_queue_ns.load(),
            time_visited_check_ns.load(),
            count_dist_accepted.load(),
            count_dist_rejected.load(),
            count_dual_queue_et.load()
        );
    }

    // ====================== FPMA Boundary Error Log Methods ======================

    void set_fpma_error_logging(bool enable) {
        enable_fpma_error_logging_ = enable;
        if (enable) {
            reset_fpma_error_log();
        }
    }

    bool get_fpma_error_logging() const {
        return enable_fpma_error_logging_;
    }

    void reset_fpma_error_log() const {
        std::lock_guard<std::mutex> lock(fpma_error_log_mutex_);
        fpma_error_log_.clear();
    }

    size_t get_fpma_error_log_size() const {
        std::lock_guard<std::mutex> lock(fpma_error_log_mutex_);
        return fpma_error_log_.size();
    }

    void reserve_fpma_error_log(size_t capacity) const {
        std::lock_guard<std::mutex> lock(fpma_error_log_mutex_);
        fpma_error_log_.reserve(capacity);
    }

    void write_fpma_error_log_csv(const std::string &path, const std::string &dataset_name, bool append) const {
        std::lock_guard<std::mutex> lock(fpma_error_log_mutex_);
        std::ofstream out;
        if (append) {
            out.open(path, std::ios::out | std::ios::app);
        } else {
            out.open(path, std::ios::out | std::ios::trunc);
        }
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open FPMA error log CSV: " + path);
        }

        if (!append) {
            out << "dataset,query_id,step_id,queue_type,efSearch,k,"
                << "V_b_id,V_c_id,"
                << "D_b_exact,D_c_exact,D_b_fpma,D_c_fpma,"
                << "exact_accept,fpma_accept,"
                << "rel_gap,normalized_margin,"
                << "rel_err_b,rel_err_c,"
                << "signed_err_b,signed_err_c\n";
        }

        out << std::setprecision(9);
        for (const auto &entry : fpma_error_log_) {
            float d_high = std::max(entry.boundary_exact, entry.candidate_exact);
            float d_low = std::min(entry.boundary_exact, entry.candidate_exact);
            float denom_mu = d_high + d_low;
            float rel_gap = d_low > 0.0f ? (d_high - d_low) / d_low : 0.0f;
            float normalized_margin = denom_mu > 0.0f ? (d_high - d_low) / denom_mu : 0.0f;
            float rel_err_b = entry.boundary_exact > 0.0f ?
                std::fabs(entry.boundary_fpma - entry.boundary_exact) / entry.boundary_exact : 0.0f;
            float rel_err_c = entry.candidate_exact > 0.0f ?
                std::fabs(entry.candidate_fpma - entry.candidate_exact) / entry.candidate_exact : 0.0f;
            float signed_err_b = entry.boundary_exact > 0.0f ?
                (entry.boundary_fpma - entry.boundary_exact) / entry.boundary_exact : 0.0f;
            float signed_err_c = entry.candidate_exact > 0.0f ?
                (entry.candidate_fpma - entry.candidate_exact) / entry.candidate_exact : 0.0f;

            out << dataset_name << ','
                << entry.query_id << ','
                << entry.step_id << ','
                << "result_boundary_update" << ','
                << entry.ef_search << ','
                << entry.k << ','
                << entry.boundary_id << ','
                << entry.candidate_id << ','
                << entry.boundary_exact << ','
                << entry.candidate_exact << ','
                << entry.boundary_fpma << ','
                << entry.candidate_fpma << ','
                << (entry.exact_accept ? 1 : 0) << ','
                << (entry.fpma_accept ? 1 : 0) << ','
                << rel_gap << ','
                << normalized_margin << ','
                << rel_err_b << ','
                << rel_err_c << ','
                << signed_err_b << ','
                << signed_err_c << '\n';
        }
    }
};
}  // namespace hnswlib
