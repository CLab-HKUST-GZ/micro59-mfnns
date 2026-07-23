#include <unordered_set>
#include <type_traits>
#include <random>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cmath>
#include <numeric>
#include "frontend/frontend.h"
#include "hnsw.h"
#include "base/anns.h"
#include <chrono>

namespace Ramulator {

namespace {

inline uint16_t fp32_to_fp16_bits_dualq(float val) {
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &val, sizeof(float));
    uint32_t sign = (f32_bits >> 16) & 0x8000;
    int32_t f32_expo = (f32_bits >> 23) & 0xFF;
    uint32_t f32_mant = f32_bits & 0x7FFFFF;
    if (f32_expo == 0xFF) {
        if (f32_mant == 0) {
            return static_cast<uint16_t>(sign | 0x7C00);
        }
        uint16_t nan_mant = static_cast<uint16_t>(f32_mant >> 13);
        if (nan_mant == 0) nan_mant = 1;
        return static_cast<uint16_t>(sign | 0x7C00 | nan_mant);
    }
    int32_t fp16_expo = f32_expo - 127 + 15;
    if (fp16_expo <= 0) {
        if (fp16_expo < -10) return static_cast<uint16_t>(sign);
        uint32_t mant = (f32_mant | 0x800000) >> (1 - fp16_expo + 13);
        if (((f32_mant | 0x800000) >> (1 - fp16_expo + 12)) & 1) {
            mant += 1;
        }
        return static_cast<uint16_t>(sign | mant);
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

inline float compute_l1_lower_bound_dualq(float q_val, uint16_t v_fp16_bits) {
    uint32_t v_sign = (v_fp16_bits >> 15) & 0x1;
    int32_t v_expo = (v_fp16_bits >> 10) & 0x1F;
    if (v_expo == 0) return 0.0f;
    uint32_t q_bits;
    std::memcpy(&q_bits, &q_val, sizeof(float));
    uint32_t q_sign = (q_bits >> 31) & 0x1;
    int32_t target_expo = v_expo - 15;
    float scale = std::ldexp(1.0f, target_expo);
    if (q_sign == 0 && v_sign == 1) return q_val * scale;
    if (q_sign == 1 && v_sign == 0) return -q_val * scale;
    if (q_sign == 0 && v_sign == 0) return -q_val * (2.0f * scale);
    return q_val * (2.0f * scale);
}

inline float compute_dualq_coarse_score_contribution(float cand_value, float query_value, uint16_t v_fp16_bits) {
    return 0.5f * cand_value * cand_value +
           compute_l1_lower_bound_dualq(query_value, v_fp16_bits);
}

inline uint32_t get_float_stored_exponent_bits(float val, uint32_t stored_bit_width) {
    if (stored_bit_width == 16U) {
        const uint16_t fp16_bits = fp32_to_fp16_bits_dualq(val);
        return static_cast<uint32_t>((fp16_bits >> 10) & 0x1F);
    }
    uint32_t fp32_bits = 0;
    std::memcpy(&fp32_bits, &val, sizeof(float));
    return (fp32_bits >> 23) & 0xFF;
}

inline uint32_t get_float_exponent_bit_count(uint32_t stored_bit_width) {
    return stored_bit_width == 16U ? 5U : 8U;
}

inline uint32_t get_float_signexp_bit_count(uint32_t stored_bit_width) {
    return 1U + get_float_exponent_bit_count(stored_bit_width);
}

constexpr Addr_t kGraphSyntheticAddrBase = static_cast<Addr_t>(0x4000000000ULL);
constexpr uint32_t kGraphCacheLineBytes = 64U;
constexpr size_t kAnsmetBitStepArrayLength = 32U;

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

template <typename ValueT>
inline void copy_point_to_center(const ValueT* point,
                                 std::vector<float>& centers,
                                 uint32_t center_idx,
                                 uint32_t dim) {
    float* center = centers.data() + static_cast<size_t>(center_idx) * static_cast<size_t>(dim);
    for (uint32_t d = 0; d < dim; ++d) {
        center[d] = static_cast<float>(point[d]);
    }
}

template <typename ValueT>
inline float compute_sq_l2_distance_to_center(const ValueT* point,
                                              const std::vector<float>& centers,
                                              uint32_t center_idx,
                                              uint32_t dim) {
    const float* center = centers.data() + static_cast<size_t>(center_idx) * static_cast<size_t>(dim);
    float dist = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        const float delta = static_cast<float>(point[d]) - center[d];
        dist += delta * delta;
    }
    return dist;
}

template <typename ValueT>
inline void accumulate_point_to_sum(const ValueT* point,
                                    std::vector<double>& sums,
                                    uint32_t center_idx,
                                    uint32_t dim) {
    double* sum = sums.data() + static_cast<size_t>(center_idx) * static_cast<size_t>(dim);
    for (uint32_t d = 0; d < dim; ++d) {
        sum[d] += static_cast<double>(point[d]);
    }
}

inline void normalize_center_from_sum(const std::vector<double>& sums,
                                      std::vector<float>& centers,
                                      uint32_t center_idx,
                                      uint32_t dim,
                                      uint32_t count) {
    const double inv_count = 1.0 / static_cast<double>(count);
    const double* sum = sums.data() + static_cast<size_t>(center_idx) * static_cast<size_t>(dim);
    float* center = centers.data() + static_cast<size_t>(center_idx) * static_cast<size_t>(dim);
    for (uint32_t d = 0; d < dim; ++d) {
        center[d] = static_cast<float>(sum[d] * inv_count);
    }
}

inline uint64_t choose_coprime_stride(uint64_t modulo, uint64_t seed) {
    if (modulo <= 1ULL) {
        return 1ULL;
    }
    uint64_t stride = (splitmix64(seed) % (modulo - 1ULL)) + 1ULL;
    while (std::gcd(stride, modulo) != 1ULL) {
        stride += 1ULL;
        if (stride >= modulo) {
            stride = 1ULL;
        }
    }
    return stride;
}

inline uint32_t normalize_dim_step_from_bit_step(uint32_t bit_step) {
    return std::max<uint32_t>(1U, 512U / std::max<uint32_t>(1U, bit_step));
}

inline bool uses_ansmet_ndp_dual_granularity(bool early_exit_enable,
                                             bool allow_sample,
                                             bool dual_queue_enable,
                                             bool mfnns_enable) {
    return early_exit_enable && allow_sample && !dual_queue_enable && !mfnns_enable;
}

inline std::vector<int> build_ansmet_dual_granularity_schedule(uint32_t coarse_bit_step,
                                                               uint32_t fine_bit_step,
                                                               int coarse_fetch_count) {
    std::vector<int> schedule(kAnsmetBitStepArrayLength, static_cast<int>(std::max<uint32_t>(1U, fine_bit_step)));
    const size_t coarse_steps = std::min<size_t>(
        schedule.size(),
        static_cast<size_t>(std::max(0, coarse_fetch_count)));
    for (size_t idx = 0; idx < coarse_steps; ++idx) {
        schedule[idx] = static_cast<int>(std::max<uint32_t>(1U, coarse_bit_step));
    }
    return schedule;
}

template <typename T>
T compute_p95_metric(std::vector<T> values) {
    if (values.empty()) {
        return T{};
    }
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::ceil(static_cast<double>(values.size()) * 0.95)) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

} // anonymous namespace

uint32_t HNSWTraversalUnit::getGraphRankId(Addr_t addr) {
    uint32_t rankId = (addr / sizeof(PointId)) % nEmbUnit;
    assert(rankId < nEmbUnit);
    return rankId;
}

uint32_t HNSWTraversalUnit::getGraphLevel0Degree(PointId candId) const {
    unsigned int* data = (unsigned int *) hnsw->get_linklist_at_level(candId, 0);
    return hnsw->getListCount(data);
}

std::vector<PointId> HNSWTraversalUnit::buildGraphLevel0Order() {
    std::vector<PointId> order;
    order.reserve(nData);
    if (!usesGraphDegreeBfsReorder()) {
        for (PointId candId = 0; candId < static_cast<PointId>(nData); ++candId) {
            order.push_back(candId);
        }
        return order;
    }

    std::vector<uint8_t> visited(nData, 0);
    std::deque<PointId> frontier;
    auto push_vertex = [&](PointId candId) {
        if (candId >= static_cast<PointId>(nData) || visited[candId]) {
            return;
        }
        visited[candId] = 1;
        frontier.push_back(candId);
    };

    push_vertex(hnsw->enterpoint_node_);
    PointId seedCursor = 0;
    while (order.size() < nData) {
        if (frontier.empty()) {
            while (seedCursor < static_cast<PointId>(nData) && visited[seedCursor]) {
                seedCursor += 1;
            }
            if (seedCursor >= static_cast<PointId>(nData)) {
                break;
            }
            push_vertex(seedCursor);
        }

        PointId candId = frontier.front();
        frontier.pop_front();
        order.push_back(candId);

        unsigned int* data = (unsigned int *) hnsw->get_linklist_at_level(candId, 0);
        uint32_t size = hnsw->getListCount(data);
        PointId* datal = (PointId*) (data + 1);
        std::vector<std::pair<uint32_t, PointId>> neighbors;
        neighbors.reserve(size);
        for (uint32_t i = 0; i < size; ++i) {
            PointId nbr = datal[i];
            if (nbr >= static_cast<PointId>(nData) || visited[nbr]) {
                continue;
            }
            neighbors.emplace_back(getGraphLevel0Degree(nbr), nbr);
        }
        std::sort(neighbors.begin(), neighbors.end(),
                  [](const std::pair<uint32_t, PointId>& lhs, const std::pair<uint32_t, PointId>& rhs) {
                      if (lhs.first != rhs.first) {
                          return lhs.first < rhs.first;
                      }
                      return lhs.second < rhs.second;
                  });
        s_graph_level0_degree_bfs_neighbor_visits += neighbors.size();
        for (const auto& entry : neighbors) {
            push_vertex(entry.second);
        }
    }

    if (order.size() < nData) {
        for (PointId candId = 0; candId < static_cast<PointId>(nData); ++candId) {
            if (!visited[candId]) {
                order.push_back(candId);
            }
        }
    }
    return order;
}

std::vector<PointId> HNSWTraversalUnit::buildVectorPlacementOrder() {
    std::vector<PointId> order;
    order.reserve(nData);
    if (hnsw == nullptr || nData == 0) {
        return order;
    }

    std::vector<uint8_t> visited(nData, 0);
    std::deque<PointId> frontier;
    auto push_vertex = [&](PointId candId) {
        if (candId >= static_cast<PointId>(nData) || visited[candId]) {
            return;
        }
        visited[candId] = 1;
        frontier.push_back(candId);
    };

    push_vertex(hnsw->enterpoint_node_);
    PointId seedCursor = 0;
    while (order.size() < nData) {
        if (frontier.empty()) {
            while (seedCursor < static_cast<PointId>(nData) && visited[seedCursor]) {
                seedCursor += 1;
            }
            if (seedCursor >= static_cast<PointId>(nData)) {
                break;
            }
            push_vertex(seedCursor);
        }

        PointId candId = frontier.front();
        frontier.pop_front();
        order.push_back(candId);

        unsigned int* data = (unsigned int *) hnsw->get_linklist_at_level(candId, 0);
        uint32_t size = hnsw->getListCount(data);
        PointId* datal = (PointId*) (data + 1);
        std::vector<std::pair<uint32_t, PointId>> neighbors;
        neighbors.reserve(size);
        for (uint32_t i = 0; i < size; ++i) {
            PointId nbr = datal[i];
            if (nbr >= static_cast<PointId>(nData) || visited[nbr]) {
                continue;
            }
            neighbors.emplace_back(getGraphLevel0Degree(nbr), nbr);
        }
        std::sort(neighbors.begin(), neighbors.end(),
                  [](const std::pair<uint32_t, PointId>& lhs, const std::pair<uint32_t, PointId>& rhs) {
                      if (lhs.first != rhs.first) {
                          return lhs.first < rhs.first;
                      }
                      return lhs.second < rhs.second;
                  });
        for (const auto& entry : neighbors) {
            push_vertex(entry.second);
        }
    }

    if (order.size() < nData) {
        for (PointId candId = 0; candId < static_cast<PointId>(nData); ++candId) {
            if (!visited[candId]) {
                order.push_back(candId);
            }
        }
    }
    return order;
}

void HNSWTraversalUnit::buildKMeansBalancedPartition(
    uint32_t n_v_emb_unit,
    std::vector<uint16_t>& outRankLocalId,
    std::vector<uint32_t>& outLocalSlot,
    std::vector<uint32_t>& outLocalRankCount)
{
    outRankLocalId.assign(nData, 0);
    outLocalSlot.assign(nData, 0);
    outLocalRankCount.assign(n_v_emb_unit, 0);

    if (nData == 0 || n_v_emb_unit == 0) {
        return;
    }

    if (hnsw == nullptr || nData <= n_v_emb_unit) {
        for (PointId candId = 0; candId < static_cast<PointId>(nData); ++candId) {
            const uint32_t local_rank = static_cast<uint32_t>(candId) % n_v_emb_unit;
            outRankLocalId[candId] = static_cast<uint16_t>(local_rank);
            outLocalSlot[candId] = outLocalRankCount[local_rank];
            outLocalRankCount[local_rank] += 1U;
        }
        print("[Config] kmeans_balanced fallback_to_modulo: nData=%u nRanks=%u hnsw=%d",
              nData,
              n_v_emb_unit,
              hnsw != nullptr ? 1 : 0);
        return;
    }

    const uint32_t sample_per_cluster = 4096U;
    const uint32_t min_sample_size = 32768U;
    const uint32_t max_sample_size = 131072U;
    const uint32_t desired_sample_size =
        std::max<uint32_t>(min_sample_size, n_v_emb_unit * sample_per_cluster);
    const uint32_t sample_size =
        std::min<uint32_t>(nData, std::min<uint32_t>(max_sample_size, desired_sample_size));
    const uint32_t max_iters = 6U;
    const uint64_t seed = 42ULL;

    std::vector<PointId> sample_ids;
    sample_ids.reserve(sample_size);
    const uint64_t sample_offset = splitmix64(seed ^ static_cast<uint64_t>(nData)) % std::max<uint32_t>(1U, nData);
    for (uint32_t i = 0; i < sample_size; ++i) {
        const uint64_t stepped = (static_cast<uint64_t>(i) * static_cast<uint64_t>(nData)) /
                                 static_cast<uint64_t>(sample_size);
        sample_ids.push_back(static_cast<PointId>((sample_offset + stepped) % static_cast<uint64_t>(nData)));
    }

    std::vector<float> centers(static_cast<size_t>(n_v_emb_unit) * static_cast<size_t>(nDim), 0.0f);
    std::vector<float> sample_min_dist(sample_size, std::numeric_limits<float>::max());
    std::vector<PointId> center_seeds(n_v_emb_unit, sample_ids.front());

    const PointId first_seed =
        (hnsw->enterpoint_node_ < static_cast<PointId>(nData)) ? hnsw->enterpoint_node_ : sample_ids.front();
    center_seeds[0] = first_seed;
    copy_point_to_center(getEmbData(first_seed), centers, 0, nDim);
    for (uint32_t sample_idx = 0; sample_idx < sample_size; ++sample_idx) {
        sample_min_dist[sample_idx] =
            compute_sq_l2_distance_to_center(getEmbData(sample_ids[sample_idx]), centers, 0, nDim);
    }
    for (uint32_t center_idx = 1; center_idx < n_v_emb_unit; ++center_idx) {
        uint32_t farthest_sample_idx = 0;
        for (uint32_t sample_idx = 1; sample_idx < sample_size; ++sample_idx) {
            if (sample_min_dist[sample_idx] > sample_min_dist[farthest_sample_idx]) {
                farthest_sample_idx = sample_idx;
            }
        }
        center_seeds[center_idx] = sample_ids[farthest_sample_idx];
        copy_point_to_center(getEmbData(center_seeds[center_idx]), centers, center_idx, nDim);
        for (uint32_t sample_idx = 0; sample_idx < sample_size; ++sample_idx) {
            const float dist =
                compute_sq_l2_distance_to_center(getEmbData(sample_ids[sample_idx]), centers, center_idx, nDim);
            if (dist < sample_min_dist[sample_idx]) {
                sample_min_dist[sample_idx] = dist;
            }
        }
    }

    std::vector<double> sums(static_cast<size_t>(n_v_emb_unit) * static_cast<size_t>(nDim), 0.0);
    std::vector<uint32_t> sample_counts(n_v_emb_unit, 0U);
    for (uint32_t iter = 0; iter < max_iters; ++iter) {
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(sample_counts.begin(), sample_counts.end(), 0U);
        std::fill(sample_min_dist.begin(), sample_min_dist.end(), std::numeric_limits<float>::max());

        for (uint32_t sample_idx = 0; sample_idx < sample_size; ++sample_idx) {
            const Type* point = getEmbData(sample_ids[sample_idx]);
            uint32_t best_center = 0;
            float best_dist = compute_sq_l2_distance_to_center(point, centers, 0, nDim);
            for (uint32_t center_idx = 1; center_idx < n_v_emb_unit; ++center_idx) {
                const float dist = compute_sq_l2_distance_to_center(point, centers, center_idx, nDim);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_center = center_idx;
                }
            }
            sample_min_dist[sample_idx] = best_dist;
            sample_counts[best_center] += 1U;
            accumulate_point_to_sum(point, sums, best_center, nDim);
        }

        for (uint32_t center_idx = 0; center_idx < n_v_emb_unit; ++center_idx) {
            if (sample_counts[center_idx] == 0U) {
                uint32_t farthest_sample_idx = 0;
                for (uint32_t sample_idx = 1; sample_idx < sample_size; ++sample_idx) {
                    if (sample_min_dist[sample_idx] > sample_min_dist[farthest_sample_idx]) {
                        farthest_sample_idx = sample_idx;
                    }
                }
                center_seeds[center_idx] = sample_ids[farthest_sample_idx];
                copy_point_to_center(getEmbData(center_seeds[center_idx]), centers, center_idx, nDim);
                sample_min_dist[farthest_sample_idx] = 0.0f;
                continue;
            }
            normalize_center_from_sum(sums, centers, center_idx, nDim, sample_counts[center_idx]);
        }
    }

    const uint32_t base_capacity = nData / n_v_emb_unit;
    const uint32_t remainder = nData % n_v_emb_unit;
    std::vector<uint32_t> rank_order(n_v_emb_unit, 0U);
    std::iota(rank_order.begin(), rank_order.end(), 0U);
    std::sort(rank_order.begin(), rank_order.end(), [&](uint32_t lhs, uint32_t rhs) {
        if (sample_counts[lhs] != sample_counts[rhs]) {
            return sample_counts[lhs] > sample_counts[rhs];
        }
        return lhs < rhs;
    });

    std::vector<uint32_t> remaining_capacity(n_v_emb_unit, base_capacity);
    for (uint32_t idx = 0; idx < remainder; ++idx) {
        remaining_capacity[rank_order[idx]] += 1U;
    }

    const uint64_t assign_start =
        splitmix64(seed ^ static_cast<uint64_t>(nData) ^ static_cast<uint64_t>(nDim)) % static_cast<uint64_t>(nData);
    const uint64_t assign_stride = choose_coprime_stride(static_cast<uint64_t>(nData), seed ^ 0x6a09e667f3bcc909ULL);
    for (uint32_t order_idx = 0; order_idx < nData; ++order_idx) {
        const PointId candId = static_cast<PointId>(
            (assign_start + static_cast<uint64_t>(order_idx) * assign_stride) % static_cast<uint64_t>(nData));
        const Type* point = getEmbData(candId);
        uint32_t best_center = std::numeric_limits<uint32_t>::max();
        float best_dist = std::numeric_limits<float>::max();
        for (uint32_t center_idx = 0; center_idx < n_v_emb_unit; ++center_idx) {
            if (remaining_capacity[center_idx] == 0U) {
                continue;
            }
            const float dist = compute_sq_l2_distance_to_center(point, centers, center_idx, nDim);
            if (dist < best_dist) {
                best_dist = dist;
                best_center = center_idx;
            }
        }
        if (best_center == std::numeric_limits<uint32_t>::max()) {
            for (uint32_t center_idx = 0; center_idx < n_v_emb_unit; ++center_idx) {
                if (remaining_capacity[center_idx] > 0U) {
                    best_center = center_idx;
                    break;
                }
            }
        }
        assert(best_center != std::numeric_limits<uint32_t>::max());
        outRankLocalId[candId] = static_cast<uint16_t>(best_center);
        remaining_capacity[best_center] -= 1U;
    }

    std::fill(outLocalRankCount.begin(), outLocalRankCount.end(), 0U);
    for (PointId candId = 0; candId < static_cast<PointId>(nData); ++candId) {
        const uint32_t local_rank = outRankLocalId[candId];
        outLocalSlot[candId] = outLocalRankCount[local_rank];
        outLocalRankCount[local_rank] += 1U;
    }

    uint32_t min_count = std::numeric_limits<uint32_t>::max();
    uint32_t max_count = 0U;
    for (uint32_t count : outLocalRankCount) {
        min_count = std::min<uint32_t>(min_count, count);
        max_count = std::max<uint32_t>(max_count, count);
    }
    print("[Config] kmeans_balanced: nData=%u nRanks=%u sampleSize=%u maxIter=%u seed=%lu minRankCount=%u maxRankCount=%u",
          nData,
          n_v_emb_unit,
          sample_size,
          max_iters,
          seed,
          min_count == std::numeric_limits<uint32_t>::max() ? 0U : min_count,
          max_count);
}

void HNSWTraversalUnit::buildKMeansClusterRoundRobinPartition(
    uint32_t n_v_emb_unit,
    std::vector<uint16_t>& outRankLocalId,
    std::vector<uint32_t>& outLocalSlot,
    std::vector<uint32_t>& outLocalRankCount)
{
    outRankLocalId.assign(nData, 0);
    outLocalSlot.assign(nData, 0);
    outLocalRankCount.assign(n_v_emb_unit, 0);

    if (nData == 0 || n_v_emb_unit == 0) {
        return;
    }

    if (hnsw == nullptr || nData <= n_v_emb_unit) {
        for (PointId candId = 0; candId < static_cast<PointId>(nData); ++candId) {
            const uint32_t local_rank = static_cast<uint32_t>(candId) % n_v_emb_unit;
            outRankLocalId[candId] = static_cast<uint16_t>(local_rank);
            outLocalSlot[candId] = outLocalRankCount[local_rank];
            outLocalRankCount[local_rank] += 1U;
        }
        print("[Config] kmeans_cluster_rr fallback_to_modulo: nData=%u nRanks=%u hnsw=%d",
              nData,
              n_v_emb_unit,
              hnsw != nullptr ? 1 : 0);
        return;
    }

    const uint32_t sample_per_cluster = 4096U;
    const uint32_t min_sample_size = 32768U;
    const uint32_t max_sample_size = 131072U;
    const uint32_t desired_sample_size =
        std::max<uint32_t>(min_sample_size, n_v_emb_unit * sample_per_cluster);
    const uint32_t sample_size =
        std::min<uint32_t>(nData, std::min<uint32_t>(max_sample_size, desired_sample_size));
    const uint32_t max_iters = 6U;
    const uint64_t seed = 42ULL;

    std::vector<PointId> sample_ids;
    sample_ids.reserve(sample_size);
    const uint64_t sample_offset = splitmix64(seed ^ static_cast<uint64_t>(nData)) % std::max<uint32_t>(1U, nData);
    for (uint32_t i = 0; i < sample_size; ++i) {
        const uint64_t stepped = (static_cast<uint64_t>(i) * static_cast<uint64_t>(nData)) /
                                 static_cast<uint64_t>(sample_size);
        sample_ids.push_back(static_cast<PointId>((sample_offset + stepped) % static_cast<uint64_t>(nData)));
    }

    std::vector<float> centers(static_cast<size_t>(n_v_emb_unit) * static_cast<size_t>(nDim), 0.0f);
    std::vector<float> sample_min_dist(sample_size, std::numeric_limits<float>::max());

    const PointId first_seed =
        (hnsw->enterpoint_node_ < static_cast<PointId>(nData)) ? hnsw->enterpoint_node_ : sample_ids.front();
    copy_point_to_center(getEmbData(first_seed), centers, 0, nDim);
    for (uint32_t sample_idx = 0; sample_idx < sample_size; ++sample_idx) {
        sample_min_dist[sample_idx] =
            compute_sq_l2_distance_to_center(getEmbData(sample_ids[sample_idx]), centers, 0, nDim);
    }
    for (uint32_t center_idx = 1; center_idx < n_v_emb_unit; ++center_idx) {
        uint32_t farthest_sample_idx = 0;
        for (uint32_t sample_idx = 1; sample_idx < sample_size; ++sample_idx) {
            if (sample_min_dist[sample_idx] > sample_min_dist[farthest_sample_idx]) {
                farthest_sample_idx = sample_idx;
            }
        }
        copy_point_to_center(getEmbData(sample_ids[farthest_sample_idx]), centers, center_idx, nDim);
        for (uint32_t sample_idx = 0; sample_idx < sample_size; ++sample_idx) {
            const float dist =
                compute_sq_l2_distance_to_center(getEmbData(sample_ids[sample_idx]), centers, center_idx, nDim);
            if (dist < sample_min_dist[sample_idx]) {
                sample_min_dist[sample_idx] = dist;
            }
        }
    }

    std::vector<double> sums(static_cast<size_t>(n_v_emb_unit) * static_cast<size_t>(nDim), 0.0);
    std::vector<uint32_t> sample_counts(n_v_emb_unit, 0U);
    for (uint32_t iter = 0; iter < max_iters; ++iter) {
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(sample_counts.begin(), sample_counts.end(), 0U);
        std::fill(sample_min_dist.begin(), sample_min_dist.end(), std::numeric_limits<float>::max());

        for (uint32_t sample_idx = 0; sample_idx < sample_size; ++sample_idx) {
            const Type* point = getEmbData(sample_ids[sample_idx]);
            uint32_t best_center = 0;
            float best_dist = compute_sq_l2_distance_to_center(point, centers, 0, nDim);
            for (uint32_t center_idx = 1; center_idx < n_v_emb_unit; ++center_idx) {
                const float dist = compute_sq_l2_distance_to_center(point, centers, center_idx, nDim);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_center = center_idx;
                }
            }
            sample_min_dist[sample_idx] = best_dist;
            sample_counts[best_center] += 1U;
            accumulate_point_to_sum(point, sums, best_center, nDim);
        }

        for (uint32_t center_idx = 0; center_idx < n_v_emb_unit; ++center_idx) {
            if (sample_counts[center_idx] == 0U) {
                uint32_t farthest_sample_idx = 0;
                for (uint32_t sample_idx = 1; sample_idx < sample_size; ++sample_idx) {
                    if (sample_min_dist[sample_idx] > sample_min_dist[farthest_sample_idx]) {
                        farthest_sample_idx = sample_idx;
                    }
                }
                copy_point_to_center(getEmbData(sample_ids[farthest_sample_idx]), centers, center_idx, nDim);
                sample_min_dist[farthest_sample_idx] = 0.0f;
                continue;
            }
            normalize_center_from_sum(sums, centers, center_idx, nDim, sample_counts[center_idx]);
        }
    }

    std::vector<std::vector<PointId>> cluster_members(n_v_emb_unit);
    cluster_members.reserve(n_v_emb_unit);
    for (PointId candId = 0; candId < static_cast<PointId>(nData); ++candId) {
        const Type* point = getEmbData(candId);
        uint32_t best_center = 0;
        float best_dist = compute_sq_l2_distance_to_center(point, centers, 0, nDim);
        for (uint32_t center_idx = 1; center_idx < n_v_emb_unit; ++center_idx) {
            const float dist = compute_sq_l2_distance_to_center(point, centers, center_idx, nDim);
            if (dist < best_dist) {
                best_dist = dist;
                best_center = center_idx;
            }
        }
        cluster_members[best_center].push_back(candId);
    }

    std::vector<uint32_t> cluster_order(n_v_emb_unit, 0U);
    std::iota(cluster_order.begin(), cluster_order.end(), 0U);
    std::sort(cluster_order.begin(), cluster_order.end(), [&](uint32_t lhs, uint32_t rhs) {
        if (cluster_members[lhs].size() != cluster_members[rhs].size()) {
            return cluster_members[lhs].size() > cluster_members[rhs].size();
        }
        return lhs < rhs;
    });

    std::vector<uint32_t> rank_loads(n_v_emb_unit, 0U);
    for (uint32_t cluster_id : cluster_order) {
        const auto& members = cluster_members[cluster_id];
        if (members.empty()) {
            continue;
        }
        uint32_t start_rank = 0U;
        for (uint32_t rank = 1; rank < n_v_emb_unit; ++rank) {
            if (rank_loads[rank] < rank_loads[start_rank]) {
                start_rank = rank;
            }
        }
        for (uint32_t idx = 0; idx < members.size(); ++idx) {
            const uint32_t local_rank = (start_rank + idx) % n_v_emb_unit;
            const PointId candId = members[idx];
            outRankLocalId[candId] = static_cast<uint16_t>(local_rank);
            outLocalSlot[candId] = outLocalRankCount[local_rank];
            outLocalRankCount[local_rank] += 1U;
            rank_loads[local_rank] += 1U;
        }
    }

    uint32_t min_rank_count = std::numeric_limits<uint32_t>::max();
    uint32_t max_rank_count = 0U;
    uint32_t min_cluster_size = std::numeric_limits<uint32_t>::max();
    uint32_t max_cluster_size = 0U;
    for (uint32_t rank = 0; rank < n_v_emb_unit; ++rank) {
        min_rank_count = std::min<uint32_t>(min_rank_count, outLocalRankCount[rank]);
        max_rank_count = std::max<uint32_t>(max_rank_count, outLocalRankCount[rank]);
        min_cluster_size = std::min<uint32_t>(min_cluster_size, static_cast<uint32_t>(cluster_members[rank].size()));
        max_cluster_size = std::max<uint32_t>(max_cluster_size, static_cast<uint32_t>(cluster_members[rank].size()));
    }
    if (min_rank_count == std::numeric_limits<uint32_t>::max()) {
        min_rank_count = 0U;
    }
    if (min_cluster_size == std::numeric_limits<uint32_t>::max()) {
        min_cluster_size = 0U;
    }
    print("[Config] kmeans_cluster_rr: nData=%u nRanks=%u sampleSize=%u maxIter=%u seed=%lu clusterMin=%u clusterMax=%u rankMin=%u rankMax=%u",
          nData,
          n_v_emb_unit,
          sample_size,
          max_iters,
          seed,
          min_cluster_size,
          max_cluster_size,
          min_rank_count,
          max_rank_count);
}

void HNSWTraversalUnit::buildMultiStartBalancedPartition(
    uint32_t n_v_emb_unit,
    std::vector<uint16_t>& outRankLocalId,
    std::vector<uint32_t>& outLocalSlot,
    std::vector<uint32_t>& outLocalRankCount)
{
    outRankLocalId.assign(nData, 0);
    outLocalSlot.assign(nData, 0);
    outLocalRankCount.assign(n_v_emb_unit, 0);

    if (nData == 0 || n_v_emb_unit == 0 || hnsw == nullptr) return;

    // Step 1: Collect seed candidates from upper HNSW levels (high-degree hubs)
    std::vector<PointId> seeds;
    seeds.reserve(n_v_emb_unit);
    {
        std::vector<uint8_t> seed_visited(nData, 0);
        // Collect all nodes at level >= 1
        for (PointId v = 0; v < static_cast<PointId>(nData); ++v) {
            if (v < static_cast<PointId>(hnsw->element_levels_.size()) &&
                hnsw->element_levels_[v] >= 1) {
                seeds.push_back(v);
                seed_visited[v] = 1;
            }
        }
        // If too many seeds, pick well-distributed ones using farthest-first
        if (seeds.size() > n_v_emb_unit) {
            // Sort by level descending, then by degree descending — pick top n_v_emb_unit
            std::sort(seeds.begin(), seeds.end(), [&](PointId a, PointId b) {
                int la = (a < static_cast<PointId>(hnsw->element_levels_.size())) ? hnsw->element_levels_[a] : 0;
                int lb = (b < static_cast<PointId>(hnsw->element_levels_.size())) ? hnsw->element_levels_[b] : 0;
                if (la != lb) return la > lb;
                return getGraphLevel0Degree(a) > getGraphLevel0Degree(b);
            });
            seeds.resize(n_v_emb_unit);
        }
        // If too few seeds, supplement with sequential unvisited nodes
        if (seeds.size() < n_v_emb_unit) {
            // Start from enterpoint's level-0 neighbors
            PointId cursor = 0;
            while (seeds.size() < n_v_emb_unit && cursor < static_cast<PointId>(nData)) {
                if (!seed_visited[cursor]) {
                    seeds.push_back(cursor);
                    seed_visited[cursor] = 1;
                }
                cursor++;
            }
        }
    }

    // Step 2: Round-robin BFS expansion
    std::vector<uint8_t> assigned(nData, 0);
    std::vector<std::deque<PointId>> frontiers(n_v_emb_unit);
    const uint32_t target_per_rank = nData / n_v_emb_unit;
    const uint32_t remainder = nData % n_v_emb_unit;

    // Initialize frontiers with seeds
    for (uint32_t r = 0; r < std::min<uint32_t>(n_v_emb_unit, static_cast<uint32_t>(seeds.size())); ++r) {
        PointId seed = seeds[r];
        if (!assigned[seed]) {
            frontiers[r].push_back(seed);
            assigned[seed] = 1;
        }
    }

    // Round-robin expansion
    uint32_t total_assigned = 0;
    bool progress = true;
    while (total_assigned < nData && progress) {
        progress = false;
        for (uint32_t rank = 0; rank < n_v_emb_unit; ++rank) {
            const uint32_t rank_target = target_per_rank + (rank < remainder ? 1U : 0U);
            if (outLocalRankCount[rank] >= rank_target) continue;

            // Try to pop from frontier (nodes are marked assigned on push, so they belong to this rank)
            PointId node = std::numeric_limits<PointId>::max();
            if (!frontiers[rank].empty()) {
                node = frontiers[rank].front();
                frontiers[rank].pop_front();
            }

            if (node == std::numeric_limits<PointId>::max()) {
                // Frontier empty — find nearest unvisited node
                for (PointId v = 0; v < static_cast<PointId>(nData); ++v) {
                    if (!assigned[v]) {
                        node = v;
                        assigned[v] = 1;
                        break;
                    }
                }
                if (node == std::numeric_limits<PointId>::max()) break;
            }

            // Assign node to this rank
            outRankLocalId[node] = static_cast<uint16_t>(rank);
            outLocalSlot[node] = outLocalRankCount[rank];
            outLocalRankCount[rank]++;
            total_assigned++;
            progress = true;

            // Push unvisited neighbors to this rank's frontier
            unsigned int* data = (unsigned int *) hnsw->get_linklist_at_level(node, 0);
            uint32_t size = hnsw->getListCount(data);
            PointId* datal = (PointId*) (data + 1);
            for (uint32_t i = 0; i < size; ++i) {
                PointId nbr = datal[i];
                if (nbr < static_cast<PointId>(nData) && !assigned[nbr]) {
                    assigned[nbr] = 1;
                    frontiers[rank].push_back(nbr);
                }
            }
        }
    }

    // Assign any remaining unassigned nodes (shouldn't happen normally)
    for (PointId v = 0; v < static_cast<PointId>(nData); ++v) {
        if (!assigned[v]) {
            // Find rank with fewest assignments
            uint32_t min_rank = 0;
            for (uint32_t r = 1; r < n_v_emb_unit; ++r) {
                if (outLocalRankCount[r] < outLocalRankCount[min_rank]) {
                    min_rank = r;
                }
            }
            outRankLocalId[v] = static_cast<uint16_t>(min_rank);
            outLocalSlot[v] = outLocalRankCount[min_rank];
            outLocalRankCount[min_rank]++;
            assigned[v] = 1;
        }
    }

    print("[Config] graph_balanced_multistart: nData=%u nRanks=%u targetPerRank=%u seeds=%zu",
          nData, n_v_emb_unit, target_per_rank, seeds.size());
}

void HNSWTraversalUnit::buildModuloBfsSlotReorder(
    uint32_t n_v_emb_unit,
    std::vector<uint16_t>& outRankLocalId,
    std::vector<uint32_t>& outLocalSlot,
    std::vector<uint32_t>& outLocalRankCount)
{
    outRankLocalId.assign(nData, 0);
    outLocalSlot.assign(nData, 0);
    outLocalRankCount.assign(n_v_emb_unit, 0);

    if (nData == 0 || n_v_emb_unit == 0) return;

    // Step 1: Assign ranks via modulo (perfect balance)
    std::vector<std::vector<PointId>> rank_members(n_v_emb_unit);
    for (PointId v = 0; v < static_cast<PointId>(nData); ++v) {
        uint32_t rank = static_cast<uint32_t>(v) % n_v_emb_unit;
        outRankLocalId[v] = static_cast<uint16_t>(rank);
        rank_members[rank].push_back(v);
    }

    if (hnsw == nullptr) {
        // No graph available, just assign slots sequentially
        for (uint32_t rank = 0; rank < n_v_emb_unit; ++rank) {
            for (uint32_t i = 0; i < rank_members[rank].size(); ++i) {
                outLocalSlot[rank_members[rank][i]] = i;
            }
            outLocalRankCount[rank] = static_cast<uint32_t>(rank_members[rank].size());
        }
        return;
    }

    // Step 2: For each rank, BFS-reorder local slots on the induced subgraph
    for (uint32_t rank = 0; rank < n_v_emb_unit; ++rank) {
        const auto& members = rank_members[rank];
        const uint32_t count = static_cast<uint32_t>(members.size());
        if (count == 0) continue;

        // Build membership set for fast lookup
        std::unordered_set<PointId> member_set(members.begin(), members.end());

        // BFS on induced subgraph
        std::vector<uint8_t> visited(nData, 0);
        std::deque<PointId> frontier;
        std::vector<PointId> bfs_order;
        bfs_order.reserve(count);

        // Start BFS from the member that is the enterpoint or has highest degree
        PointId best_start = members[0];
        if (member_set.count(hnsw->enterpoint_node_)) {
            best_start = hnsw->enterpoint_node_;
        } else {
            uint32_t best_deg = 0;
            for (PointId v : members) {
                uint32_t deg = getGraphLevel0Degree(v);
                if (deg > best_deg) {
                    best_deg = deg;
                    best_start = v;
                }
            }
        }

        visited[best_start] = 1;
        frontier.push_back(best_start);
        uint32_t member_cursor = 0;

        while (bfs_order.size() < count) {
            if (frontier.empty()) {
                // Find next unvisited member
                while (member_cursor < count && visited[members[member_cursor]]) {
                    member_cursor++;
                }
                if (member_cursor >= count) break;
                visited[members[member_cursor]] = 1;
                frontier.push_back(members[member_cursor]);
            }

            PointId node = frontier.front();
            frontier.pop_front();
            bfs_order.push_back(node);

            // Push same-rank neighbors
            unsigned int* data = (unsigned int *) hnsw->get_linklist_at_level(node, 0);
            uint32_t size = hnsw->getListCount(data);
            PointId* datal = (PointId*) (data + 1);
            for (uint32_t i = 0; i < size; ++i) {
                PointId nbr = datal[i];
                if (nbr < static_cast<PointId>(nData) && !visited[nbr] && member_set.count(nbr)) {
                    visited[nbr] = 1;
                    frontier.push_back(nbr);
                }
            }
        }

        // Assign slots in BFS order
        for (uint32_t i = 0; i < bfs_order.size(); ++i) {
            outLocalSlot[bfs_order[i]] = i;
        }
        outLocalRankCount[rank] = static_cast<uint32_t>(bfs_order.size());
    }

    print("[Config] modulo_bfs_slot: nData=%u nRanks=%u intra-rank BFS reorder complete", nData, n_v_emb_unit);
}

void HNSWTraversalUnit::prepareVectorRankPlacement() {
    const uint32_t n_v_emb_unit = getVectorRankLocalEmbUnitCount();
    vectorRankLocalId.assign(nData, 0);
    vectorLocalSlot.assign(nData, 0);
    vectorLocalRankCount.assign(n_v_emb_unit, 0);
    vectorReplicaSlot.assign(nData, std::numeric_limits<uint32_t>::max());
    vectorReplicaCount = 0;
    if (nData == 0 || n_v_emb_unit == 0) {
        print("[Config] vector rank placement mode=%s nData=%u nLocalRanks=%u",
              vectorRankPlacementMode.c_str(),
              nData,
              n_v_emb_unit);
        return;
    }

    // Handle new placement modes that build their own partition
    if (usesKMeansBalancedPlacement()) {
        buildKMeansBalancedPartition(n_v_emb_unit,
                                     vectorRankLocalId, vectorLocalSlot, vectorLocalRankCount);
        goto placement_done;
    }
    if (usesKMeansClusterRoundRobinPlacement()) {
        buildKMeansClusterRoundRobinPartition(n_v_emb_unit,
                                              vectorRankLocalId, vectorLocalSlot, vectorLocalRankCount);
        goto placement_done;
    }
    if (usesGraphBalancedMultistartPlacement()) {
        buildMultiStartBalancedPartition(n_v_emb_unit,
                                         vectorRankLocalId, vectorLocalSlot, vectorLocalRankCount);
        goto placement_done;
    }
    if (usesModuloBfsSlotPlacement()) {
        buildModuloBfsSlotReorder(n_v_emb_unit,
                                  vectorRankLocalId, vectorLocalSlot, vectorLocalRankCount);
        goto placement_done;
    }

    {
    std::vector<PointId> placement_order;
    if (usesGraphClusteredVectorRankPlacement()) {
        placement_order = buildVectorPlacementOrder();
    } else {
        placement_order.reserve(nData);
        for (PointId candId = 0; candId < static_cast<PointId>(nData); ++candId) {
            placement_order.push_back(candId);
        }
    }

    if (usesTopLayerReplicaPlacement()) {
        for (PointId candId : placement_order) {
            if (!isHotReplicaNode(candId)) {
                continue;
            }
            vectorReplicaSlot[candId] = vectorReplicaCount;
            vectorRankLocalId[candId] = static_cast<uint16_t>(vectorReplicaCount % n_v_emb_unit);
            vectorLocalSlot[candId] = vectorReplicaCount;
            vectorReplicaCount += 1U;
        }
    }

    if (!usesGraphClusteredVectorRankPlacement()) {
        for (PointId candId : placement_order) {
            if (usesTopLayerReplicaPlacement() && isHotReplicaNode(candId)) {
                continue;
            }
            const uint32_t local_rank = static_cast<uint32_t>(candId) % n_v_emb_unit;
            const uint32_t local_slot = vectorLocalRankCount[local_rank];
            vectorRankLocalId[candId] = static_cast<uint16_t>(local_rank);
            vectorLocalSlot[candId] = local_slot;
            vectorLocalRankCount[local_rank] += 1U;
        }
    } else {
        const uint32_t chunk_size = getVectorRankPlacementEffectiveChunkSize();
        const uint32_t stripe_group_size =
            std::min<uint32_t>(n_v_emb_unit, getVectorRankPlacementStripeGroupSize());
        const uint32_t stripe_block_size = getVectorRankPlacementStripeBlockSize();
        const bool level_aware = usesVectorRankPlacementLevelAwareStripe() && hnsw != nullptr;
        const uint32_t top_stripe_group_size =
            std::min<uint32_t>(n_v_emb_unit, getVectorRankPlacementTopStripeGroupSize());
        const uint32_t top_stripe_block_size = getVectorRankPlacementTopStripeBlockSize();
        const int max_level = (level_aware && hnsw != nullptr) ? std::max(0, hnsw->maxlevel_) : 0;
        const int top_level_cutoff =
            level_aware ? std::max(0, max_level - static_cast<int>(getVectorRankPlacementTopLevelCount()) + 1) : 0;
        auto select_group_start = [&](uint32_t tie_break_rank, uint32_t selected_group_size) {
            uint32_t group_start = tie_break_rank;
            uint64_t best_group_count = std::numeric_limits<uint64_t>::max();
            for (uint32_t offset = 0; offset < n_v_emb_unit; ++offset) {
                const uint32_t probe_group_start = (tie_break_rank + offset) % n_v_emb_unit;
                uint64_t probe_group_count = 0;
                for (uint32_t stripe_idx = 0; stripe_idx < selected_group_size; ++stripe_idx) {
                    const uint32_t probe_rank = (probe_group_start + stripe_idx) % n_v_emb_unit;
                    probe_group_count += vectorLocalRankCount[probe_rank];
                }
                if (probe_group_count < best_group_count) {
                    best_group_count = probe_group_count;
                    group_start = probe_group_start;
                }
            }
            return group_start;
        };
        uint32_t tie_break_rank = 0;
        uint32_t tie_break_rank_top = 0;
        uint32_t tie_break_rank_rest = 0;
        for (uint32_t chunk_begin = 0; chunk_begin < placement_order.size(); chunk_begin += chunk_size) {
            uint32_t group_start = select_group_start(tie_break_rank, stripe_group_size);
            const uint32_t top_group_start = level_aware
                ? select_group_start(tie_break_rank_top, top_stripe_group_size)
                : group_start;
            const uint32_t rest_group_start = level_aware
                ? select_group_start(tie_break_rank_rest, stripe_group_size)
                : group_start;
            const uint32_t chunk_end = std::min<uint32_t>(chunk_begin + chunk_size, static_cast<uint32_t>(placement_order.size()));
            uint32_t top_cursor = 0;
            uint32_t rest_cursor = 0;
            for (uint32_t cursor = chunk_begin; cursor < chunk_end; ++cursor) {
                const PointId candId = placement_order[cursor];
                if (usesTopLayerReplicaPlacement() && isHotReplicaNode(candId)) {
                    continue;
                }
                bool use_top_profile = false;
                if (level_aware && candId < static_cast<PointId>(hnsw->element_levels_.size())) {
                    use_top_profile = static_cast<int>(hnsw->element_levels_[candId]) >= top_level_cutoff;
                }
                const uint32_t selected_group_size = use_top_profile ? top_stripe_group_size : stripe_group_size;
                const uint32_t selected_block_size = use_top_profile ? top_stripe_block_size : stripe_block_size;
                const uint32_t selected_group_start = use_top_profile ? top_group_start : rest_group_start;
                const uint32_t selected_cursor = use_top_profile ? top_cursor : rest_cursor;
                const uint32_t stripe_index = (selected_cursor / selected_block_size) % selected_group_size;
                const uint32_t local_rank = (selected_group_start + stripe_index) % n_v_emb_unit;
                vectorRankLocalId[candId] = static_cast<uint16_t>(local_rank);
                vectorLocalSlot[candId] = vectorLocalRankCount[local_rank];
                vectorLocalRankCount[local_rank] += 1U;
                if (use_top_profile) {
                    top_cursor += 1U;
                } else {
                    rest_cursor += 1U;
                }
            }
            tie_break_rank = (group_start + stripe_group_size) % n_v_emb_unit;
            if (level_aware && top_cursor > 0) {
                tie_break_rank_top = (top_group_start + top_stripe_group_size) % n_v_emb_unit;
            }
            if (level_aware && rest_cursor > 0) {
                tie_break_rank_rest = (rest_group_start + stripe_group_size) % n_v_emb_unit;
            }
        }
    }
    } // end of legacy placement scope

    placement_done:
    uint32_t min_rank_count = std::numeric_limits<uint32_t>::max();
    uint32_t max_rank_count = 0;
    for (uint32_t count : vectorLocalRankCount) {
        min_rank_count = std::min<uint32_t>(min_rank_count, count);
        max_rank_count = std::max<uint32_t>(max_rank_count, count);
    }
    if (min_rank_count == std::numeric_limits<uint32_t>::max()) {
        min_rank_count = 0;
    }
    print("[Config] vector rank placement mode=%s nLocalRanks=%u chunkSize=%u minRankCount=%u maxRankCount=%u",
          vectorRankPlacementMode.c_str(),
          n_v_emb_unit,
          getVectorRankPlacementEffectiveChunkSize(),
          min_rank_count,
          max_rank_count);
    print("[Config] vector rank placement replicaPrefixCount=%u topLayerReplicaPlacement=%d",
          vectorReplicaCount,
          usesTopLayerReplicaPlacement() ? 1 : 0);
    print("[Config] vector rank placement baseChunkSize=%u superChunkFactor=%u effectiveChunkSize=%u",
          getVectorRankPlacementBaseChunkSize(),
          getVectorRankPlacementSuperChunkFactor(),
          getVectorRankPlacementEffectiveChunkSize());
    print("[Config] vector rank placement stripeGroupSize=%u stripeBlockSize=%u",
          std::min<uint32_t>(n_v_emb_unit, getVectorRankPlacementStripeGroupSize()),
          getVectorRankPlacementStripeBlockSize());
    print("[Config] vector rank placement levelAware=%d topLevels=%u topStripeGroupSize=%u topStripeBlockSize=%u",
          usesVectorRankPlacementLevelAwareStripe() ? 1 : 0,
          getVectorRankPlacementTopLevelCount(),
          std::min<uint32_t>(n_v_emb_unit, getVectorRankPlacementTopStripeGroupSize()),
          getVectorRankPlacementTopStripeBlockSize());
}

void HNSWTraversalUnit::prepareGraphStaticSchedule() {
    graphLevel0RankId.clear();
    graphLevel0NodeLineBase.clear();
    graphLevel0EdgeLineBase.clear();
    s_graph_level0_layout_total_lines = 0;
    s_graph_level0_layout_padding_lines = 0;
    s_graph_level0_degree_bfs_neighbor_visits = 0;
    if (!usesGraphStaticPagePlacement()) {
        return;
    }

    graphLevel0RankId.resize(nData, 0);
    graphLevel0NodeLineBase.resize(nData, 0);
    graphLevel0EdgeLineBase.resize(nData, 0);

    const uint32_t pageLines = std::max<uint32_t>(1, graphPlacementPageLines);
    const std::vector<PointId> order = buildGraphLevel0Order();
    uint64_t globalLine = 0;
    for (PointId candId : order) {
        const uint32_t edgeLines = getTravNodeSegment(candId, 0);
        const uint32_t vertexLines = 1 + edgeLines;
        uint64_t pageIndex = globalLine / pageLines;
        uint32_t lineInPage = static_cast<uint32_t>(globalLine % pageLines);
        if (lineInPage != 0 && lineInPage + vertexLines > pageLines) {
            const uint32_t padding = pageLines - lineInPage;
            globalLine += padding;
            s_graph_level0_layout_padding_lines += padding;
            pageIndex = globalLine / pageLines;
            lineInPage = static_cast<uint32_t>(globalLine % pageLines);
        }
        const uint32_t rankId = static_cast<uint32_t>(pageIndex % nEmbUnit);
        const uint64_t localPageIndex = pageIndex / nEmbUnit;
        const uint64_t localLineBase = localPageIndex * pageLines + lineInPage;
        graphLevel0RankId[candId] = static_cast<uint8_t>(rankId);
        graphLevel0NodeLineBase[candId] = static_cast<uint32_t>(localLineBase);
        graphLevel0EdgeLineBase[candId] = static_cast<uint32_t>(localLineBase + 1);
        globalLine += vertexLines;
        s_graph_level0_layout_total_lines += vertexLines;
    }

    print("[Config] graph static schedule mode=%s reorder=%s pageLines=%u totalLines=%lu paddingLines=%lu",
          graphStaticScheduleMode.c_str(),
          graphVertexReorderMode.c_str(),
          pageLines,
          s_graph_level0_layout_total_lines,
          s_graph_level0_layout_padding_lines);
}

HNSWTraversalUnit::GraphMemLoc HNSWTraversalUnit::getGraphNodeMemLoc(TravReq& travReq) {
    const Addr_t addr = getGraphNodeAddress(travReq);
    if (!usesGraphStaticPagePlacement() || travReq.queryLevel != 0 ||
        travReq.candId >= static_cast<PointId>(graphLevel0RankId.size())) {
        return GraphMemLoc{getGraphRankId(addr), addr};
    }
    return GraphMemLoc{
        static_cast<uint32_t>(graphLevel0RankId[travReq.candId]),
        static_cast<Addr_t>(kGraphSyntheticAddrBase +
                            static_cast<uint64_t>(graphLevel0NodeLineBase[travReq.candId]) * kGraphCacheLineBytes)
    };
}

HNSWTraversalUnit::GraphMemLoc HNSWTraversalUnit::getGraphEdgeMemLoc(TravReq& travReq) {
    const Addr_t addr = getGraphEdgeAddress(travReq);
    if (!usesGraphStaticPagePlacement() || travReq.queryLevel != 0 ||
        travReq.candId >= static_cast<PointId>(graphLevel0RankId.size())) {
        return GraphMemLoc{getGraphRankId(addr), addr};
    }
    return GraphMemLoc{
        static_cast<uint32_t>(graphLevel0RankId[travReq.candId]),
        static_cast<Addr_t>(kGraphSyntheticAddrBase +
                            static_cast<uint64_t>(graphLevel0EdgeLineBase[travReq.candId] + travReq.segmentId) * kGraphCacheLineBytes)
    };
}

HNSWTraversalUnit::GraphTravSchedInfo
HNSWTraversalUnit::getGraphTravSchedInfo(const TravReq& travReq, bool isEdgeReq) {
    TravReq localReq = travReq;
    GraphMemLoc memLoc = isEdgeReq ? getGraphEdgeMemLoc(localReq) : getGraphNodeMemLoc(localReq);
    GraphTravSchedInfo info;
    info.rankId = memLoc.rankId;
    if (graphPlacementPageLines == 0) {
        info.pageId = 0;
        info.valid = true;
        return info;
    }

    uint64_t lineId = 0;
    if (usesGraphStaticPagePlacement() && travReq.queryLevel == 0) {
        if (travReq.candId >= static_cast<PointId>(graphLevel0RankId.size())) {
            return info;
        }
        const uint64_t lineBase = isEdgeReq
            ? static_cast<uint64_t>(graphLevel0EdgeLineBase[travReq.candId] + travReq.segmentId)
            : static_cast<uint64_t>(graphLevel0NodeLineBase[travReq.candId]);
        lineId = lineBase;
    } else {
        lineId = static_cast<uint64_t>(memLoc.addr / kGraphCacheLineBytes);
    }
    info.pageId = lineId / std::max<uint32_t>(1, graphPlacementPageLines);
    info.valid = true;
    return info;
}

size_t HNSWTraversalUnit::selectTravReqIndex(const std::deque<TravReq>& queue, bool isEdgeReq) {
    if (queue.empty() || !usesGraphTraversalBatching()) {
        return 0;
    }
    const size_t window = std::min<size_t>(std::max<uint32_t>(1, graphTraversalBatchingWindow), queue.size());
    const uint64_t curCycle = m_memory_system->get_clk();

    size_t oldestIndex = 0;
    uint64_t oldestAge = 0;
    for (size_t idx = 0; idx < window; ++idx) {
        const uint64_t age = curCycle - queue[idx].reqSendCycle;
        if (age > oldestAge) {
            oldestAge = age;
            oldestIndex = idx;
        }
    }
    if (graphTraversalBatchingAgeThreshold > 0 && oldestAge >= graphTraversalBatchingAgeThreshold) {
        return oldestIndex;
    }

    std::unordered_map<uint32_t, uint32_t> rankFreq;
    std::unordered_map<uint64_t, uint32_t> pageFreq;
    std::vector<GraphTravSchedInfo> infos(window);
    for (size_t idx = 0; idx < window; ++idx) {
        infos[idx] = getGraphTravSchedInfo(queue[idx], isEdgeReq);
        if (!infos[idx].valid) {
            continue;
        }
        rankFreq[infos[idx].rankId] += 1;
        const uint64_t pageKey = (static_cast<uint64_t>(infos[idx].rankId) << 56) ^ infos[idx].pageId;
        pageFreq[pageKey] += 1;
    }

    int64_t bestScore = std::numeric_limits<int64_t>::min();
    size_t bestIndex = 0;
    for (size_t idx = 0; idx < window; ++idx) {
        const auto& info = infos[idx];
        int64_t score = -static_cast<int64_t>(idx);
        if (info.valid) {
            const uint64_t pageKey = (static_cast<uint64_t>(info.rankId) << 56) ^ info.pageId;
            score += static_cast<int64_t>(graphTraversalBatchingPageWeight) * pageFreq[pageKey];
            score += static_cast<int64_t>(graphTraversalBatchingRankWeight) * rankFreq[info.rankId];
            if (graphTravLastIssueValid && info.pageId == graphTravLastIssuePageId) {
                score += graphTraversalBatchingLastPageBonus;
            }
            if (graphTravLastIssueValid && info.rankId == graphTravLastIssueRankId) {
                score += graphTraversalBatchingLastRankBonus;
            }
        }
        if (score > bestScore) {
            bestScore = score;
            bestIndex = idx;
        }
    }
    return bestIndex;
}

bool HNSWTraversalUnit::trySendTravReq(std::deque<TravReq>& queue, bool isEdgeReq) {
    if (queue.empty()) {
        return false;
    }
    size_t oldestIndex = 0;
    uint64_t oldestAge = 0;
    if (usesGraphTraversalBatching()) {
        const size_t window = std::min<size_t>(std::max<uint32_t>(1, graphTraversalBatchingWindow), queue.size());
        const uint64_t curCycle = m_memory_system->get_clk();
        for (size_t idx = 0; idx < window; ++idx) {
            const uint64_t age = curCycle - queue[idx].reqSendCycle;
            if (age > oldestAge) {
                oldestAge = age;
                oldestIndex = idx;
            }
        }
    }
    const size_t selectedIndex = selectTravReqIndex(queue, isEdgeReq);
    TravReq travReq = queue[selectedIndex];
    const GraphTravSchedInfo info = usesGraphTraversalBatching()
        ? getGraphTravSchedInfo(travReq, isEdgeReq)
        : GraphTravSchedInfo{};
    const bool ok = isEdgeReq ? sendTravEdgeReq(travReq) : sendTravNodeReq(travReq);
    if (!ok) {
        return false;
    }
    if (usesGraphTraversalBatching()) {
        if (selectedIndex != 0) {
            s_graph_trav_batch_reordered += 1;
        }
        if (graphTraversalBatchingAgeThreshold > 0 &&
            oldestAge >= graphTraversalBatchingAgeThreshold &&
            selectedIndex == oldestIndex) {
            s_graph_trav_batch_age_forced += 1;
        }
        if (info.valid) {
            if (graphTravLastIssueValid && info.pageId == graphTravLastIssuePageId) {
                s_graph_trav_batch_page_match_selected += 1;
            }
            if (graphTravLastIssueValid && info.rankId == graphTravLastIssueRankId) {
                s_graph_trav_batch_rank_match_selected += 1;
            }
        }
    }
    if (selectedIndex == 0) {
        queue.pop_front();
    } else {
        queue.erase(queue.begin() + static_cast<std::deque<TravReq>::difference_type>(selectedIndex));
    }

    if (usesGraphTraversalBatching()) {
        if (info.valid) {
            graphTravLastIssueValid = true;
            graphTravLastIssueRankId = info.rankId;
            graphTravLastIssuePageId = info.pageId;
        }
    }
    return true;
}

Addr_t HNSWTraversalUnit::getGraphNodeAddress(TravReq& travReq) {
    return (hnsw->max_elements_*(travReq.queryLevel)+travReq.candId)*sizeof(int);//+offset
}

Addr_t HNSWTraversalUnit::getGraphEdgeAddress(TravReq& travReq) {
    return (hnsw->max_elements_*(travReq.queryLevel)+hnsw->node_array[travReq.queryLevel][travReq.candId])*sizeof(PointId)+travReq.segmentId*64;
}

uint32_t HNSWTraversalUnit::getTravNodeSegment(PointId candId, uint32_t queryLevel) {
    unsigned int* data = (unsigned int *) hnsw->get_linklist_at_level(candId, queryLevel);
    uint32_t size = hnsw->getListCount(data);
    if (size == 0) return 0;
    uint32_t nSegment = (size * sizeof(PointId) - 1) / 64 + 1;
    return nSegment;
}

Type HNSWTraversalUnit::computeDualQueueLowerBound(const Query* query, PointId candId) {
    assert(query);
    const Type* cand = getEmbData(candId);
    float coarse_score_accum = 0.0f;
    for (uint32_t i = 0; i < nDim; i++) {
        uint16_t v_fp16_bits = fp32_to_fp16_bits_dualq(cand[i]);
        coarse_score_accum +=
            (spacetype == "L2")
                ? compute_dualq_coarse_score_contribution(cand[i], query->query[i], v_fp16_bits)
                : compute_l1_lower_bound_dualq(query->query[i], v_fp16_bits);
    }
    if (spacetype == "L2") {
        const float l2_lower_bound = query->querySquaredNorm + 2.0f * coarse_score_accum;
        return std::max(0.0f, l2_lower_bound);
    }
    return coarse_score_accum;
}

void HNSWTraversalUnit::updateFinalResultQueue(Query* query, PointDistId result) {
    assert(query);
    if (query->type != "search" || query->level != 0) {
        return;
    }
    query->finalResult.emplace(result);
    while (query->finalResult.size() > k_neighbors) {
        query->finalResult.pop();
    }
}

Type HNSWTraversalUnit::getDualQueueLowerBoundThreshold(const Query* query) {
    assert(query);
    if (query->type == "search" && query->level == 0 &&
        query->finalResult.size() < k_neighbors) {
        return std::numeric_limits<Type>::max();
    }
    if (!(query->type == "search" && query->level == 0) &&
        query->result.size() < k_neighbors) {
        return std::numeric_limits<Type>::max();
    }
    const uint32_t warmup = (dualQueueLowerBoundWarmupSize > 0)
                                ? dualQueueLowerBoundWarmupSize
                                : k_neighbors;
    if (query->lowerBoundResult.size() < warmup) {
        return std::numeric_limits<Type>::max();
    }
    return query->lowerBoundResult.top().first;
}

Type HNSWTraversalUnit::getTopKThreshold(const Query* query) {
    assert(query);
    if (query->type == "search" && query->level == 0) {
        if (query->finalResult.size() < k_neighbors) {
            return std::numeric_limits<Type>::max();
        }
        return query->finalResult.top().first;
    }
    if (query->result.size() < k_neighbors) {
        return std::numeric_limits<Type>::max();
    }
    auto topk_candidates = query->result;
    std::vector<PointDistId> distances;
    distances.reserve(topk_candidates.size());
    while (!topk_candidates.empty()) {
        distances.emplace_back(topk_candidates.top());
        topk_candidates.pop();
    }
    const size_t kth = static_cast<size_t>(k_neighbors - 1);
    std::nth_element(
        distances.begin(),
        distances.begin() + kth,
        distances.end(),
        [](const PointDistId& a, const PointDistId& b) {
            return a.first < b.first;
        });
    return distances[kth].first;
}

Type HNSWTraversalUnit::getExactDistanceThreshold(const Query* query) {
    assert(query);
    if (query->result.empty()) {
        return std::numeric_limits<Type>::max();
    }
    Type candidate_threshold = query->result.top().first;
    if (query->type != "search" || query->level > 0) return candidate_threshold;

    if (query->result.size() < ef_search) {
        candidate_threshold = std::numeric_limits<Type>::max();
    }
    Type final_threshold = std::numeric_limits<Type>::max();
    if (query->finalResult.size() >= k_neighbors) {
        final_threshold = query->finalResult.top().first;
    }
    return std::max(candidate_threshold, final_threshold);
}

Type HNSWTraversalUnit::getEarlyTerminationThreshold(const Query* query) {
    assert(query);
    if (dualQueueLowerBoundETEnable) {
        return getDualQueueLowerBoundThreshold(query);
    }
    if (topkThresholdETEnable) {
        return getTopKThreshold(query);
    }
    return getExactDistanceThreshold(query);
}

void HNSWTraversalUnit::handleTravNodeReq(TravReq& travReq) {
    uint32_t nSegment = getTravNodeSegment(travReq.candId, travReq.queryLevel);
    m_logger->info("[handleTravNodeReq] annsId {} candId {} nSegment {}", travReq.annsId, travReq.candId, nSegment);
    uint64_t curCycle = m_memory_system->get_clk();
    uint64_t travCycles = curCycle - travReq.reqSendCycle;
    if (nSegment == 0) {
        handleTravEdgeReq(travReq);
        return;
    }
    for (uint32_t i = 0; i < nSegment; i++) {
        pendTravEdgeReqs.emplace_back(travReq.annsId, travReq.candId, travReq.queryLevel, /* segmentId */ i, curCycle);
    }
}

void HNSWTraversalUnit::handleTravEdgeReq(TravReq& travReq) {
    uint32_t nSegment = getTravNodeSegment(travReq.candId, travReq.queryLevel);
    if (nSegment > 0 && travReq.segmentId < nSegment - 1) return;
    uint64_t curCycle = m_memory_system->get_clk();
    uint64_t travCycles = curCycle - travReq.reqSendCycle;
    assert(queries.count(travReq.annsId));
    if (queries[travReq.annsId]->type == "construct") {
        handleAddPoint(travReq);
    } else {
        assert(queries[travReq.annsId]->type == "search");
        handleSearchLayer(travReq);
    }
}

void HNSWTraversalUnit::handlePollReq(Request& req) {
    Addr_t pollingAddr = 0xdeadbeef;
    Request pollReq(pollingAddr,
        Request::Type::Read,
        req.source_id,
        pollCallback);
    s_num_polling += 1;
    s_bd_polling_cycles += m_memory_system->get_clk() - req.issue_time;
    pendPollReqs.push(pollReq);
}

bool HNSWTraversalUnit::sendTravEdgeReq(TravReq& travReq) {
    if (!traversalMemAcc) {
        handleTravEdgeReq(travReq);
        return true;
    }
    GraphMemLoc edgeLoc = getGraphEdgeMemLoc(travReq);
    queries[travReq.annsId]->indexCycles ++;
    Request req(edgeLoc.addr,
                Request::Type::Read,
                edgeLoc.rankId,
                travEdgeCallback,
                travReq);
    req.is_anns_graph = true;
    bool ok = m_memory_system->send(req);
    if (ok) {
        m_logger->info("[TravUnit] sendTravEdgeReq addr {} rankId {} annsId {} candId {} segmentId {}",
                       edgeLoc.addr,
                       edgeLoc.rankId,
                       travReq.annsId,
                       travReq.candId,
                       travReq.segmentId);
        auto query_it = queries.find(travReq.annsId);
        if (query_it != queries.end() && query_it->second != nullptr) {
            query_it->second->perQueryTravEdgeReq += 1;
        }
    }
    if (ok) s_num_travedge_req += 1;
    return ok;
}

bool HNSWTraversalUnit::sendTravNodeReq(TravReq& travReq) {
    if (!traversalMemAcc) {
        handleTravNodeReq(travReq);
        return true;
    }
    GraphMemLoc nodeLoc = getGraphNodeMemLoc(travReq);
    Request req(nodeLoc.addr,
                Request::Type::Read,
                nodeLoc.rankId,
                travNodeCallback,
                travReq);
    req.is_anns_graph = true;
    queries[travReq.annsId]->indexCycles ++;
    bool ok = m_memory_system->send(req);
    if (ok) {
        m_logger->info("[TravUnit] sendTravNodeReq addr {} rankId {} annsId {} candId {}", nodeLoc.addr, nodeLoc.rankId, travReq.annsId, travReq.candId);
        auto query_it = queries.find(travReq.annsId);
        if (query_it != queries.end() && query_it->second != nullptr) {
            query_it->second->perQueryTravNodeReq += 1;
        }
    }
    if (ok) s_num_travnode_req += 1;
    return ok;
}

uint32_t HNSWTraversalUnit::getVectorPackingGroup(PointId candId) {
    auto it = vecPackingMap.find(candId);
    if (it != vecPackingMap.end()) {
        return it->second;
    }
    if (nPackingVec == 0) {
        return candId;
    }
    return candId / nPackingVec;
}

bool HNSWTraversalUnit::shouldBufferLevel0DualQueueResults(const Query* query) const {
    return dualQueueLowerBoundETEnable &&
           query != nullptr &&
           query->type == "search" &&
           query->level == 0;
}

void HNSWTraversalUnit::beginLevel0DualQueueBatch(Query* query) {
    if (!shouldBufferLevel0DualQueueResults(query)) {
        return;
    }
    query->bufferedLevel0DualQueueResults.clear();
    query->level0DualQueueBatchOrder.clear();
    query->nextLevel0DualQueueBatchOrder = 0;
}

void HNSWTraversalUnit::recordLevel0DualQueueCandidateOrder(Query* query, PointId candId) {
    if (!shouldBufferLevel0DualQueueResults(query)) {
        return;
    }
    query->level0DualQueueBatchOrder[candId] = query->nextLevel0DualQueueBatchOrder++;
}

bool HNSWTraversalUnit::shouldDebugIssueTraceForQuery(uint64_t annsId) const {
    return debugIssueTraceEnable && annsId == static_cast<uint64_t>(debugIssueTraceQueryId);
}

bool HNSWTraversalUnit::shouldDebugDuplicateAcceptForQuery(uint64_t annsId) const {
    return debugDuplicateAcceptEnable && annsId == static_cast<uint64_t>(debugDuplicateAcceptQueryId);
}

void HNSWTraversalUnit::recordIssueTraceSeen(Query* query,
                                             uint32_t level,
                                             PointId parentCandId,
                                             PointId candId,
                                             bool baseVisitBefore,
                                             bool skippedByBaseVisit,
                                             bool markedBaseVisit) {
    if (query == nullptr || !query->debugIssueTraceActive) {
        return;
    }
    const uint64_t key =
        (static_cast<uint64_t>(level) << 32U) |
        static_cast<uint64_t>(static_cast<uint32_t>(candId));
    Query::CandidateIssueDebugStat& stat = query->debugIssueStatsByLevelKey[key];
    if (stat.seenCount == 0) {
        stat.level = level;
        stat.candId = candId;
    }
    stat.seenCount += 1;
    query->debugIssueTraceSeenTotal += 1;
    if (!stat.hasParent) {
        stat.firstParentCandId = parentCandId;
        stat.hasParent = true;
    }
    stat.lastParentCandId = parentCandId;
    if (skippedByBaseVisit) {
        stat.baseVisitSkipCount += 1;
        query->debugIssueTraceBaseVisitSkipTotal += 1;
    }
    if (markedBaseVisit) {
        stat.baseVisitMarkCount += 1;
        query->debugIssueTraceBaseVisitMarkTotal += 1;
    }
    if (skippedByBaseVisit || stat.seenCount > 1) {
        query->debugIssueTraceSeq += 1;
        query->debugIssueEvents.push_back(Query::CandidateIssueDebugEvent{
            query->debugIssueTraceSeq,
            skippedByBaseVisit ? "basevisit_skip" : "repeat_seen",
            level,
            parentCandId,
            candId,
            baseVisitBefore,
            stat.seenCount,
            stat.candidateIssueCount,
            stat.disreqIssueCount,
            0,
            0,
            0,
        });
    }
}

void HNSWTraversalUnit::recordIssueTraceDisreqIssue(Query* query,
                                                    uint32_t level,
                                                    PointId parentCandId,
                                                    PointId candId,
                                                    uint32_t embUnitId,
                                                    uint32_t vDimBase,
                                                    Type upperbound) {
    if (query == nullptr || !query->debugIssueTraceActive) {
        return;
    }
    const uint64_t key =
        (static_cast<uint64_t>(level) << 32U) |
        static_cast<uint64_t>(static_cast<uint32_t>(candId));
    Query::CandidateIssueDebugStat& stat = query->debugIssueStatsByLevelKey[key];
    if (stat.seenCount == 0) {
        stat.level = level;
        stat.candId = candId;
    }
    if (stat.disreqIssueCount == 0) {
        stat.firstParentCandId = parentCandId;
        stat.hasParent = true;
    }
    stat.lastParentCandId = parentCandId;
    stat.disreqIssueCount += 1;
    query->debugIssueTraceDisreqIssueTotal += 1;
    if (vDimBase == 0) {
        stat.candidateIssueCount += 1;
        query->debugIssueTraceCandidateIssueTotal += 1;
    }
    if (stat.candidateIssueCount > 1 || stat.disreqIssueCount > 1) {
        query->debugIssueTraceSeq += 1;
        query->debugIssueEvents.push_back(Query::CandidateIssueDebugEvent{
            query->debugIssueTraceSeq,
            (stat.candidateIssueCount > 1) ? "repeat_candidate_issue" : "repeat_disreq_issue",
            level,
            parentCandId,
            candId,
            false,
            stat.seenCount,
            stat.candidateIssueCount,
            stat.disreqIssueCount,
            embUnitId,
            vDimBase,
            upperbound,
        });
    }
}

void HNSWTraversalUnit::dumpIssueTraceDebug(Query* query, const char* reason) {
    if (query == nullptr || !query->debugIssueTraceActive || query->debugIssueTraceDumped) {
        return;
    }
    query->debugIssueTraceDumped = true;

    std::string output_path = debugIssueTracePath;
    if (output_path.empty()) {
        output_path = statFile + ".issue_trace_q" + std::to_string(query->annsId) + ".log";
    }
    std::ofstream out(output_path, std::ios::app);
    if (!out.is_open()) {
        m_logger->warn("Failed to open issue-trace debug file {}", output_path.c_str());
        return;
    }

    std::vector<Query::CandidateIssueDebugStat> suspicious;
    suspicious.reserve(query->debugIssueStatsByLevelKey.size());
    for (const auto& it : query->debugIssueStatsByLevelKey) {
        const auto& stat = it.second;
        if (stat.seenCount > 1 || stat.baseVisitSkipCount > 0 || stat.candidateIssueCount > 1 || stat.disreqIssueCount > 1) {
            suspicious.push_back(stat);
        }
    }
    std::sort(suspicious.begin(), suspicious.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.level != rhs.level) return lhs.level < rhs.level;
        if (lhs.candidateIssueCount != rhs.candidateIssueCount) return lhs.candidateIssueCount > rhs.candidateIssueCount;
        if (lhs.baseVisitSkipCount != rhs.baseVisitSkipCount) return lhs.baseVisitSkipCount > rhs.baseVisitSkipCount;
        if (lhs.seenCount != rhs.seenCount) return lhs.seenCount > rhs.seenCount;
        return lhs.candId < rhs.candId;
    });

    out << "query_id: " << query->annsId << "\n";
    out << "reason: " << (reason ? reason : "unknown") << "\n";
    out << "type: " << query->type << "\n";
    out << "tracked_scope: handleSearchLayer baseVisitArray + DisReq issue\n";
    out << "tracked_key: (level, candId)\n";
    out << "total_seen_events: " << query->debugIssueTraceSeenTotal << "\n";
    out << "total_basevisit_skip_events: " << query->debugIssueTraceBaseVisitSkipTotal << "\n";
    out << "total_basevisit_mark_events: " << query->debugIssueTraceBaseVisitMarkTotal << "\n";
    out << "total_candidate_issue_events: " << query->debugIssueTraceCandidateIssueTotal << "\n";
    out << "total_disreq_issue_events: " << query->debugIssueTraceDisreqIssueTotal << "\n";
    out << "tracked_level_cand_keys: " << query->debugIssueStatsByLevelKey.size() << "\n";
    out << "suspicious_level_cand_keys: " << suspicious.size() << "\n";
    out << "suspicious_candidates:\n";
    if (suspicious.empty()) {
        out << "  - none\n";
    } else {
        for (const auto& stat : suspicious) {
            out << "  - level: " << stat.level
                << ", cand_id: " << stat.candId
                << ", seen_count: " << stat.seenCount
                << ", basevisit_skip_count: " << stat.baseVisitSkipCount
                << ", basevisit_mark_count: " << stat.baseVisitMarkCount
                << ", candidate_issue_count: " << stat.candidateIssueCount
                << ", disreq_issue_count: " << stat.disreqIssueCount
                << ", first_parent: " << (stat.hasParent ? std::to_string(stat.firstParentCandId) : "NA")
                << ", last_parent: " << (stat.hasParent ? std::to_string(stat.lastParentCandId) : "NA")
                << "\n";
        }
    }
    out << "events:\n";
    if (query->debugIssueEvents.empty()) {
        out << "  - none\n";
    } else {
        for (const auto& event : query->debugIssueEvents) {
            out << "  - seq: " << event.seq
                << ", kind: " << event.kind
                << ", level: " << event.level
                << ", parent_cand: " << event.parentCandId
                << ", cand_id: " << event.candId
                << ", basevisit_before: " << (event.baseVisitBefore ? 1 : 0)
                << ", seen_count_after: " << event.seenCountAfter
                << ", candidate_issue_count_after: " << event.candidateIssueCountAfter
                << ", disreq_issue_count_after: " << event.disreqIssueCountAfter
                << ", embUnitId: " << event.embUnitId
                << ", vDimBase: " << event.vDimBase
                << ", upperbound: " << event.upperbound
                << "\n";
        }
    }
    out << "---\n";
}

void HNSWTraversalUnit::dumpPerQuerySummary(Query* query, const char* reason) {
    if (query == nullptr || !perQuerySummaryEnable || query->type != "search") {
        return;
    }

    std::string output_path = perQuerySummaryPath;
    if (output_path.empty()) {
        output_path = statFile + ".per_query.tsv";
    }

    bool needs_header = true;
    {
        std::ifstream check(output_path);
        needs_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream out(output_path, std::ios::app);
    if (!out.is_open()) {
        m_logger->warn("Failed to open per-query summary file {}", output_path.c_str());
        return;
    }

    if (needs_header) {
        out << "query_id\treason\tlatency_cycles\ttravnode_req\ttravedge_req\ttotal_disreq\tcandidate_issue\tlevel0_basevisit_mark\tlevel0_basevisit_skip\tindex_cycles\tdiscomp_cycles\tdisoffload_cycles\tdisgather_cycles\trecall_hits\trecall_result_count\trecall_compare_k\n";
    }
    out << query->annsId
        << '\t' << (reason ? reason : "unknown")
        << '\t' << query->perQueryLatencyCycles
        << '\t' << query->perQueryTravNodeReq
        << '\t' << query->perQueryTravEdgeReq
        << '\t' << query->perQueryDisreqCompleted
        << '\t' << query->perQueryCandidateIssue
        << '\t' << query->perQueryLevel0BaseVisitMark
        << '\t' << query->perQueryLevel0BaseVisitSkip
        << '\t' << query->indexCycles
        << '\t' << query->disCompCycles
        << '\t' << query->disOffloadCycles
        << '\t' << query->disGatherCycles
        << '\t' << query->perQueryRecallHits
        << '\t' << query->perQueryRecallResultCount
        << '\t' << query->perQueryRecallCompareK
        << '\n';
}

void HNSWTraversalUnit::recordAcceptedCandidateDebugEvent(Query* query,
                                                          PointId candId,
                                                          Type pDistance,
                                                          Type upperbound) {
    if (query == nullptr || !query->debugDuplicateAcceptActive) {
        return;
    }
    const uint64_t level_key =
        (static_cast<uint64_t>(query->level) << 32U) |
        static_cast<uint64_t>(static_cast<uint32_t>(candId));
    const uint32_t result_size_before = static_cast<uint32_t>(query->result.size());
    const uint32_t search_size_before = static_cast<uint32_t>(query->search.size());
    const uint32_t inflight_size_before = static_cast<uint32_t>(query->inflightCands.size());
    uint32_t& accept_count = query->debugAcceptedCountsByLevelKey[level_key];
    accept_count += 1;
    query->debugDuplicateAcceptSeq += 1;
    query->debugDuplicateAcceptTotal += 1;
    query->debugDuplicateAcceptEvents.push_back(Query::DuplicateAcceptDebugEvent{
        query->debugDuplicateAcceptSeq,
        candId,
        pDistance,
        upperbound,
        query->level,
        result_size_before,
        search_size_before,
        inflight_size_before,
        accept_count,
    });
    if (accept_count <= 1) {
        return;
    }
    query->debugDuplicateAcceptDuplicateEvents += 1;
}

void HNSWTraversalUnit::dumpAcceptedCandidateDebug(Query* query, const char* reason) {
    if (query == nullptr || !query->debugDuplicateAcceptActive || query->debugDuplicateAcceptDumped) {
        return;
    }
    query->debugDuplicateAcceptDumped = true;

    std::string output_path = debugDuplicateAcceptPath;
    if (output_path.empty()) {
        output_path = statFile + ".duplicate_accept_q" + std::to_string(query->annsId) + ".log";
    }

    std::ofstream out(output_path, std::ios::app);
    if (!out.is_open()) {
        m_logger->warn("Failed to open duplicate-accept debug file {}", output_path.c_str());
        return;
    }

    out << "query_id: " << query->annsId << "\n";
    out << "reason: " << (reason ? reason : "unknown") << "\n";
    out << "type: " << query->type << "\n";
    out << "final_level: " << query->level << "\n";
    out << "tracked_scope: applyCompletedCandidateResult accepted inserts only\n";
    out << "tracked_key: (level, candId)\n";
    out << "total_accept_events: " << query->debugDuplicateAcceptTotal << "\n";
    out << "unique_level_cand_keys: " << query->debugAcceptedCountsByLevelKey.size() << "\n";
    out << "duplicate_accept_events: " << query->debugDuplicateAcceptDuplicateEvents << "\n";
    out << "duplicate_exists: " << (query->debugDuplicateAcceptDuplicateEvents > 0 ? "true" : "false") << "\n";
    out << "event_stream: all_accepted_inserts\n";
    out << "events:\n";
    if (query->debugDuplicateAcceptEvents.empty()) {
        out << "  - none\n";
    } else {
        for (const auto& event : query->debugDuplicateAcceptEvents) {
            out << "  - accept_seq: " << event.acceptSeq
                << ", level: " << event.level
                << ", cand_id: " << event.candId
                << ", pDistance: " << event.pDistance
                << ", upperbound: " << event.upperbound
                << ", result_size_before: " << event.resultSizeBefore
                << ", search_size_before: " << event.searchSizeBefore
                << ", inflight_size_before: " << event.inflightSizeBefore
                << ", accept_count_after: " << event.acceptCountAfter
                << "\n";
        }
    }
    out << "---\n";
}

void HNSWTraversalUnit::applyCompletedCandidateResult(Query* query,
                                                      PointId candId,
                                                      Type pDistance,
                                                      Type candidateLowerBound,
                                                      bool dualQueuePruned,
                                                      uint64_t num_access) {
    Type upperbound = query->result.top().first;
    uint32_t limit = (query->type == "search") ? ef_search : hnsw->ef_construction_;
    if (!dualQueuePruned && query->type == "search" && query->level == 0) {
        updateFinalResultQueue(query, PointDistId(pDistance, candId));
        if (dualQueueLowerBoundETEnable) {
            query->lowerBoundResult.emplace(candidateLowerBound, candId);
            const uint32_t lb_queue_cap = (dualQueueLowerBoundQueueSize > 0)
                                              ? dualQueueLowerBoundQueueSize
                                              : k_neighbors;
            while (query->lowerBoundResult.size() > lb_queue_cap) {
                query->lowerBoundResult.pop();
            }
        }
    }
    bool positive;
    if (query->level > 0) {
        positive = pDistance < upperbound;
    } else {
        positive = (query->result.size() < limit) || (pDistance < upperbound);
    }
    if (dualQueuePruned) {
        positive = false;
    }

    if (positive) {
        recordAcceptedCandidateDebugEvent(query, candId, pDistance, upperbound);
        query->search.emplace(pDistance, candId);
        query->result.emplace(pDistance, candId);
        if (query->level > 0) {
            while (query->result.size() > 1) query->result.pop();
            while (query->search.size() > 1) query->search.pop();
        } else {
            if (query->result.size() > limit) query->result.pop();
        }
        query->topChanged = true;
        query->disCompAccCycles += num_access;
        m_logger->info("[handleDisReq] annsId {} candId {} result {} upperbound {} num_access {} accept. queryInflightCands {}",
                       query->annsId, candId, pDistance, upperbound, num_access, query->inflightCands.size());
    } else {
        query->disCompRejCycles += num_access;
        m_logger->info("[handleDisReq] query {} candId {} result {} upperbound {} num_access {} reject. queryInflightCands {}",
                       query->annsId, candId, pDistance, upperbound, num_access, query->inflightCands.size());
    }
}

void HNSWTraversalUnit::flushBufferedLevel0DualQueueResults(Query* query) {
    if (!shouldBufferLevel0DualQueueResults(query) || query->bufferedLevel0DualQueueResults.empty()) {
        return;
    }
    std::sort(query->bufferedLevel0DualQueueResults.begin(),
              query->bufferedLevel0DualQueueResults.end(),
              [](const Query::BufferedCandidateResult& lhs, const Query::BufferedCandidateResult& rhs) {
                  if (lhs.batchOrder != rhs.batchOrder) {
                      return lhs.batchOrder < rhs.batchOrder;
                  }
                  return lhs.candId < rhs.candId;
              });
    for (const Query::BufferedCandidateResult& buffered : query->bufferedLevel0DualQueueResults) {
        applyCompletedCandidateResult(query,
                                      buffered.candId,
                                      buffered.pDistance,
                                      buffered.candidateLowerBound,
                                      buffered.dualQueuePruned,
                                      buffered.num_access);
    }
    query->bufferedLevel0DualQueueResults.clear();
    query->level0DualQueueBatchOrder.clear();
    query->nextLevel0DualQueueBatchOrder = 0;
}

void HNSWTraversalUnit::handleSearchLayer(TravReq& travReq) {
    uint32_t annsId = travReq.annsId;
    PointId candId = travReq.candId;
    unsigned int* data = (unsigned int *) hnsw->get_linklist_at_level(candId, travReq.queryLevel);
    uint32_t size = hnsw->getListCount(data);
    PointId* datal = (PointId*) (data + 1);
    Query* query = queries[annsId];
    assert(query->type == "search");
    // Dual-queue ET passes the current lower-bound threshold to EmbUnit for
    // two-phase (sign+exp -> mantissa) memory access modeling.
    Type exact_upperbound = getExactDistanceThreshold(query);
    Type dualq_phase1_upperbound = (dualQueueLowerBoundETEnable && travReq.queryLevel == 0)
                                       ? getDualQueueLowerBoundThreshold(query)
                                       : exact_upperbound;
    Type disreq_upperbound = exact_upperbound;
    if (earlyExitEnable && !mfnnsEnable && !dualQueueLowerBoundETEnable) {
        disreq_upperbound = getEarlyTerminationThreshold(query);
    }
    if (travReq.queryLevel == 0) {
        beginLevel0DualQueueBatch(query);
    }
    m_logger->info("[handleSearchLayer] annsId {} candId {} level {} neighbor num {}", annsId, candId, travReq.queryLevel, size);
    prof_candAcc[candId];
    prof_candAcc[candId] += 1;
    uint64_t curCycle = m_memory_system->get_clk();
    if (travReq.queryLevel > 0 && size == 0) {
        query->inflightCands.insert(candId);
        query->inflightDisReqs[candId];
        query->pDistance[candId] = 0;
        for (uint32_t i = 0; i < nDim / vDimSize; i++) {
            uint32_t vDimBase = i * vDimSize;
            query->inflightDisReqs[candId].insert(disreqId);
            const bool dualQueueTwoPhase = dualQueueLowerBoundETEnable && travReq.queryLevel == 0;
            uint32_t embUnitId = mapDisReq(candId, i, vDimBase, vDimBase + vDimSize, dualQueueTwoPhase);
            recordIssueTraceDisreqIssue(query, travReq.queryLevel, travReq.candId, candId, embUnitId, vDimBase, disreq_upperbound);
            DisReq req(disreqId++, annsId, candId, query->query, disreq_upperbound, vDimBase, vDimBase + vDimSize, nDim, curCycle, travReq.queryLevel);
            req.dualQueueTwoPhase = dualQueueTwoPhase;
            req.dualQueuePhase1Upperbound = dualq_phase1_upperbound;
            pendDisReqs[embUnitId].push(req);
            if (i == 0) {
                recordSameRankDispatch(query, embUnitId % getVectorRankLocalEmbUnitCount());
            }
            m_logger->info("[handleSearchLayer] annsId {} candId {} vDimBase {} no neighbor send request to candId", annsId, candId, vDimBase);
        }
    } else {
        std::unordered_map<uint32_t, uint32_t> vecPackId;
        for (uint32_t i = 0; i < size; i++) {
            PointId candId = datal[i];
            if (candId < 0 || candId > hnsw->max_elements_)
                throw std::runtime_error("candId error");
            const bool basevisit_before =
                (travReq.queryLevel == 0) ? query->baseVisitArray[candId] : false;
            if (travReq.queryLevel == 0) {
                if (query->baseVisitArray[candId]) {
                    query->perQueryLevel0BaseVisitSkip += 1;
                    recordIssueTraceSeen(query, travReq.queryLevel, travReq.candId, candId, basevisit_before, true, false);
                    continue;
                }
                query->baseVisitArray[candId] = true;
                query->perQueryLevel0BaseVisitMark += 1;
                recordLevel0DualQueueCandidateOrder(query, candId);
            }
            recordIssueTraceSeen(query,
                                 travReq.queryLevel,
                                 travReq.candId,
                                 candId,
                                 basevisit_before,
                                 false,
                                 travReq.queryLevel == 0);
            query->inflightCands.insert(candId);
            query->inflightDisReqs[candId];
            query->pDistance[candId] = 0;
            for (uint32_t i = 0; i < nDim / vDimSize; i++) {
                uint32_t vDimBase = i * vDimSize;
                query->inflightDisReqs[candId].insert(disreqId);
                const bool dualQueueTwoPhase = dualQueueLowerBoundETEnable && travReq.queryLevel == 0;
                uint32_t embUnitId = mapDisReq(candId, i, vDimBase, vDimBase + vDimSize, dualQueueTwoPhase);
                if (i == 0) {
                    query->perQueryCandidateIssue += 1;
                }
                recordIssueTraceDisreqIssue(query, travReq.queryLevel, travReq.candId, candId, embUnitId, vDimBase, disreq_upperbound);
                m_logger->info("[handleSearchLayer] send embId {}: annsId {} candId {} vDimBase {} upperbound {}",
                               embUnitId,
                               annsId,
                               candId,
                               vDimBase,
                               disreq_upperbound);
                DisReq req(disreqId++, annsId, candId, query->query, disreq_upperbound, vDimBase, vDimBase + vDimSize, nDim, curCycle, travReq.queryLevel);
                req.dualQueueTwoPhase = dualQueueTwoPhase;
                req.dualQueuePhase1Upperbound = dualq_phase1_upperbound;
                pendDisReqs[embUnitId].push(req);
                if (i == 0) {
                    recordSameRankDispatch(query, embUnitId % getVectorRankLocalEmbUnitCount());
                }
                uint32_t packId = getVectorPackingGroup(candId);
                if (!vecPackId.count(packId)) vecPackId[packId] = 0;
                vecPackId[packId] += 1;
            }
        }
        for (auto& [packId, num] : vecPackId) {
            assert(num <= nPackingVec);
            uint32_t layer = hnsw->element_levels_[travReq.candId];
            s_num_total_disreq[layer] += num;
            if (num > 1) {
                s_num_packed_disreq[layer] += nPackingVec;
                s_num_packed_saving[layer] += num;
            }
        }
        m_logger->info("[handleSearchLayer] annsId {} candId {} send total {} requests", annsId, candId, query->inflightCands.size());
        if (query->inflightCands.empty()) finishQueryCheck(annsId);
    }
}

void HNSWTraversalUnit::searchLayer(uint32_t annsId) {
    Query* query = queries[annsId];
    assert(query->search.size() > 0);
    assert(query->result.size() > 0);
    uint64_t curCycle = m_memory_system->get_clk();
    if (query->startCycle == 0) query->startCycle = curCycle;
    PointDistId curNode = query->search.top();
    PointId& candId = curNode.second;
    query->search.pop();
    uint32_t beamSearchBound = (query->type == "search")? 0 : query->consLevel;
    if (query->level > beamSearchBound) query->topChanged = false;
    if (query->level == beamSearchBound) assert(query->baseVisitArray);
    if (dualQueueLowerBoundETEnable && query->type == "search" && query->level == 0 &&
        query->lowerBoundResult.empty()) {
        Type ep_lower_bound = computeDualQueueLowerBound(query, candId);
        query->lowerBoundResult.emplace(ep_lower_bound, candId);
        query->candidateLowerBounds[candId] = ep_lower_bound;
    }
    pendTravNodeReqs.emplace_back(annsId, candId, query->level, /* segmentId */ 0, curCycle);
}

void HNSWTraversalUnit::prepareHotReplicaNodes() {
    hotReplicaNodes.clear();
    if (!needsHotReplicaNodeSelection() || !hnsw || nData == 0) {
        return;
    }
    const size_t target = std::min<size_t>(hotNodeReplicationCount, nData);
    if (target == 0 && hotNodeReplicationTopLevelCount == 0) {
        return;
    }
    const int max_level = std::max(0, hnsw->maxlevel_);
    int top_level_cutoff = 0;
    if (hotNodeReplicationTopLevelCount > 0 && max_level > 0) {
        const uint32_t clamped_top_levels = std::min<uint32_t>(hotNodeReplicationTopLevelCount, static_cast<uint32_t>(max_level));
        top_level_cutoff = std::max(1, max_level - static_cast<int>(clamped_top_levels) + 1);
        for (PointId candId = 0; candId < static_cast<PointId>(nData); ++candId) {
            if (static_cast<int>(hnsw->element_levels_[candId]) >= top_level_cutoff) {
                hotReplicaNodes.insert(candId);
            }
        }
    }
    if (hotReplicaNodes.empty() && target > 0 && hnsw->enterpoint_node_ < nData) {
        hotReplicaNodes.insert(hnsw->enterpoint_node_);
    }
    auto cmp = [](const std::pair<uint32_t, PointId>& lhs, const std::pair<uint32_t, PointId>& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first > rhs.first;
        }
        return lhs.second < rhs.second;
    };
    for (int level = max_level; level >= 1 && hotReplicaNodes.size() < target; --level) {
        std::vector<std::pair<uint32_t, PointId>> bucket;
        for (PointId candId = 0; candId < static_cast<PointId>(nData); ++candId) {
            if (static_cast<int>(hnsw->element_levels_[candId]) != level) {
                continue;
            }
            if (hotReplicaNodes.count(candId)) {
                continue;
            }
            hnswlib::linklistsizeint* ll_cur = hnsw->get_linklist_at_level(candId, level);
            uint32_t degree = ll_cur ? hnsw->getListCount(ll_cur) : 0;
            bucket.emplace_back(degree, candId);
        }
        if (bucket.empty()) {
            continue;
        }
        const size_t remaining = target - hotReplicaNodes.size();
        if (bucket.size() > remaining) {
            std::partial_sort(bucket.begin(), bucket.begin() + remaining, bucket.end(), cmp);
            bucket.resize(remaining);
        } else {
            std::sort(bucket.begin(), bucket.end(), cmp);
        }
        for (const auto& entry : bucket) {
            hotReplicaNodes.insert(entry.second);
        }
    }
    print("[Config] hot-node replication selected %zu nodes (target=%zu topLevels=%u cutoffLevel=%d partialPlacement=%d replicaDispatch=%d)",
          hotReplicaNodes.size(),
          target,
          hotNodeReplicationTopLevelCount,
          top_level_cutoff,
          usesTopLayerReplicaPlacement() ? 1 : 0,
          usesHotNodeReplicaDispatch() ? 1 : 0);
}

uint32_t HNSWTraversalUnit::selectLeastLoadedEmbUnitInVerticalGroup(uint32_t verticalId) {
    uint32_t nVEmbUnit = nEmbUnit / (nDim / vDimSize);
    uint32_t vEmbBase = verticalId * nVEmbUnit;
    uint32_t best_id = vEmbBase;
    uint64_t best_load = std::numeric_limits<uint64_t>::max();
    for (uint32_t offset = 0; offset < nVEmbUnit; ++offset) {
        uint32_t embUnitId = vEmbBase + offset;
        uint64_t inflight = embUnits[embUnitId]->get_inflight_disreq();
        if (inflight < best_load) {
            best_load = inflight;
            best_id = embUnitId;
        }
    }
    return best_id;
}

uint32_t HNSWTraversalUnit::selectReplicaEmbUnitInVerticalGroup(PointId candId,
                                                                uint32_t verticalId,
                                                                uint32_t vDimBase,
                                                                uint32_t vDimEnd,
                                                                bool dualQueueTwoPhase) {
    if (!isHotReplicaRowAwareEnabled() || !dualQueueTwoPhase) {
        return selectLeastLoadedEmbUnitInVerticalGroup(verticalId);
    }

    uint32_t nVEmbUnit = nEmbUnit / (nDim / vDimSize);
    uint32_t vEmbBase = verticalId * nVEmbUnit;
    uint32_t best_id = vEmbBase;
    bool best_valid = false;
    int64_t best_score = std::numeric_limits<int64_t>::max();
    HNSWEmbUnit::ReplicaDispatchProbe best_probe;

    for (uint32_t offset = 0; offset < nVEmbUnit; ++offset) {
        uint32_t embUnitId = vEmbBase + offset;
        HNSWEmbUnit::ReplicaDispatchProbe probe =
            embUnits[embUnitId]->predictReplicaDispatchProbe(candId, vDimBase, vDimEnd, dualQueueTwoPhase);
        int64_t score = static_cast<int64_t>(getHotReplicaLoadWeight()) * static_cast<int64_t>(probe.inflightLoad)
                      + static_cast<int64_t>(getHotReplicaBgPenaltyWeight()) * static_cast<int64_t>(probe.sumBgOccupancy)
                      + static_cast<int64_t>(getHotReplicaBankPenaltyWeight()) * static_cast<int64_t>(probe.sumBankOccupancy)
                      + static_cast<int64_t>(probe.lineCount)
                      - static_cast<int64_t>(getHotReplicaRowHitBonus()) * static_cast<int64_t>(probe.rowHitPred)
                      - static_cast<int64_t>(getHotReplicaBgRowHitBonus()) * static_cast<int64_t>(probe.bgRowHitPred);
        if (!best_valid ||
            score < best_score ||
            (score == best_score && probe.rowHitPred + probe.bgRowHitPred > best_probe.rowHitPred + best_probe.bgRowHitPred) ||
            (score == best_score && probe.inflightLoad < best_probe.inflightLoad) ||
            (score == best_score && embUnitId < best_id)) {
            best_valid = true;
            best_score = score;
            best_id = embUnitId;
            best_probe = probe;
        }
    }

    if (best_valid) {
        s_hot_replica_rowaware_selected += 1;
        if (best_probe.rowHitPred > 0) {
            s_hot_replica_rowhit_selected += 1;
        }
        if (best_probe.bgRowHitPred > 0) {
            s_hot_replica_bg_rowhit_selected += 1;
        }
        return best_id;
    }
    return selectLeastLoadedEmbUnitInVerticalGroup(verticalId);
}

uint32_t HNSWTraversalUnit::mapDisReq(PointId candId,
                                      uint32_t verticalId,
                                      uint32_t vDimBase,
                                      uint32_t vDimEnd,
                                      bool dualQueueTwoPhase) {
    if (debug_minDisReqMap) {
        if (!disreqMap.count(candId)) {
            uint32_t minId;
            uint64_t minAcc = -1U;
            for (uint32_t i = 0; i < nEmbUnit; i++) {
                if (embUnits[i]->get_inflight_disreq() < minAcc) {
                    minAcc = embUnits[i]->get_inflight_disreq();
                    minId = i;
                }
            }
            disreqMap[candId] = minId;
        }
        return disreqMap[candId];
    } else if (debug_epCopy) {
        if (candId == hnsw->enterpoint_node_) {
            uint32_t minId;
            uint64_t minAcc = -1U;
            for (uint32_t i = 0; i < nEmbUnit; i++) {
                if (embUnits[i]->get_inflight_disreq() < minAcc) {
                    minAcc = embUnits[i]->get_inflight_disreq();
                    minId = i;
                }
            }
            return minId;
        }
        return candId % nEmbUnit;
        // return (uint32_t)(candId * nEmbUnit / 1000000);
    }
    uint32_t nVEmbUnit = nEmbUnit / (nDim / vDimSize);
    uint32_t vEmbBase = verticalId * nVEmbUnit;
    uint32_t default_sel = vEmbBase + getVectorRankLocalId(candId);
    if (usesHotNodeReplicaDispatch() && isHotReplicaNode(candId)) {
        uint32_t replica_sel = selectReplicaEmbUnitInVerticalGroup(candId, verticalId, vDimBase, vDimEnd, dualQueueTwoPhase);
        assert(replica_sel < nEmbUnit);
        if (replica_sel != default_sel) {
            s_hot_replica_remap += 1;
        }
        return replica_sel;
    }
    assert(default_sel < nEmbUnit);
    return default_sel;
    // return (uint32_t)(candId * nVEmbUnit / 1000000);
}

void HNSWTraversalUnit::recordSameRankDispatch(Query* query, uint32_t rankId) {
    if (query == nullptr) {
        return;
    }
    if (!query->sameRankRunValid) {
        query->sameRankRunValid = true;
        query->sameRankLast = rankId;
        query->sameRankCurrentLength = 1;
        return;
    }
    if (query->sameRankLast == rankId) {
        query->sameRankCurrentLength += 1;
        return;
    }
    if (query->sameRankCurrentLength > 0) {
        s_same_rank_run_lengths.push_back(query->sameRankCurrentLength);
    }
    query->sameRankLast = rankId;
    query->sameRankCurrentLength = 1;
}

void HNSWTraversalUnit::flushSameRankRun(Query* query) {
    if (query == nullptr || !query->sameRankRunValid) {
        return;
    }
    if (query->sameRankCurrentLength > 0) {
        s_same_rank_run_lengths.push_back(query->sameRankCurrentLength);
    }
    query->sameRankRunValid = false;
    query->sameRankCurrentLength = 0;
}

void HNSWTraversalUnit::checkRecall(Query* query) {
    assert((query->annsId + 1) * gt_k <= gt.size());
    const uint32_t compare_k = std::min(gt_k, k_neighbors);
    uint32_t n_find = 0, n_result = 0;
    ANNSResultQueue recall_results =
        (query->type == "search" && query->level == 0) ? query->finalResult : query->result;
    while (!recall_results.empty()) {
        PointId res_internal = recall_results.top().second;
        uint32_t res = static_cast<uint32_t>(hnsw->getExternalLabel(res_internal));
        recall_results.pop();
        n_result += 1;
        bool found = false;
        for (uint32_t i = query->annsId * gt_k; i < query->annsId * gt_k + compare_k; i++) {
            if (gt[i] == res) {
                found = true;
                break;
            }
        }
        if (found) n_find += 1;
    }
    n_result = std::min(n_result, compare_k);
    query->perQueryRecallHits = n_find;
    query->perQueryRecallResultCount = n_result;
    query->perQueryRecallCompareK = compare_k;
    s_num_recall += n_find;
    s_num_recall_1 += compare_k;
}

void HNSWTraversalUnit::finishQueryCheck(uint32_t annsId) {
    Query* query = queries[annsId];
    assert(query->type == "search");
    if (query->level > 0 && query->topChanged)
        searchLayer(annsId);
    else if (query->level == 0 && !query->search.empty())
        searchLayer(annsId);
    else if (query->level > 0 && !query->topChanged) {
        PointDistId nxEntryPoint = query->result.top();
        query->result.pop();
        query->search = ANNSSearchQueue();
        query->result = ANNSResultQueue();
        query->search.emplace(nxEntryPoint);
        query->result.emplace(nxEntryPoint);
        query->level -= 1;
        if (query->level == 0) {
            query->finalResult = ANNSResultQueue();
            if (nxEntryPoint.first != static_cast<Type>(-1U)) {
                updateFinalResultQueue(query, nxEntryPoint);
            }
            query->baseVisitArray = new bool[nData];
            std::fill(query->baseVisitArray, query->baseVisitArray + nData, false);
            query->baseVisitArray[nxEntryPoint.second] = true;
            query->lowerBoundResult = ANNSResultQueue();
            query->candidateLowerBounds.clear();
            if (dualQueueLowerBoundETEnable) {
                Type ep_lower_bound = computeDualQueueLowerBound(query, nxEntryPoint.second);
                query->lowerBoundResult.emplace(ep_lower_bound, nxEntryPoint.second);
                query->candidateLowerBounds[nxEntryPoint.second] = ep_lower_bound;
            }
        }
        m_logger->info("[handleQuery] finish level of query {} level. Next level {} Next EP {}", annsId, query->level, nxEntryPoint.second);
        searchLayer(annsId);
    } else if (query->level == 0 && query->search.empty()) {
        assert(queries.count(annsId));
        uint32_t latency = m_memory_system->get_clk() - query->startCycle;
        query->perQueryLatencyCycles = latency;
        s_total_latency += latency;
        s_query_latency_bin[std::min(nLatBin - 1, (uint32_t) latency / latBinSize)] += 1;
        s_bd_index_cycles += query->indexCycles;
        s_bd_disOffload_cycles += query->disOffloadCycles;
        s_bd_disComp_cycles += query->disCompCycles;
        s_bd_disCompAcc_cycles += query->disCompAccCycles;
        s_bd_disCompRej_cycles += query->disCompRejCycles;
        s_bd_disGather_cycles += query->disGatherCycles;
        // Check recall BEFORE clearing the result queue
        if (gt.size() > 0) checkRecall(query);
        dumpPerQuerySummary(query, "finish_query");
        dumpIssueTraceDebug(query, "finish_query");
        dumpAcceptedCandidateDebug(query, "finish_query");
        while (!query->result.empty()) {
            PointDistId pd = query->result.top();
            query->result.pop();
        }
        flushSameRankRun(query);
        delete queries[annsId];
        queries.erase(annsId);
        s_num_query += 1;
        m_logger->info("[handleQuery] finish query {}", annsId);
        return;
    }
}

bool HNSWTraversalUnit::mergeDisReq(Query* query, DisResp& resp) {
    uint32_t& annsId = resp.annsId;
    PointId& candId = resp.candId;
    if (!query->inflightCands.count(candId)) return false; // TODO: here are two return falses. They can be eliminated using a global unique candId, similar to disreqId used in embunits.
    assert(query->inflightCands.count(candId));
    assert(query->inflightDisReqs.count(candId));
    if (!query->inflightDisReqs[candId].count(resp.disreqId)) return false;
    assert(query->inflightDisReqs[candId].count(resp.disreqId));
    query->inflightDisReqs[candId].erase(resp.disreqId);
    query->pDistance[candId] += resp.distance;
    m_logger->info("[mergeDisReq] annsId {} candId {} disreqId {} result {} inflightDisReq {}", annsId, candId, resp.disreqId, query->pDistance[candId], query->inflightDisReqs[candId].size());
    // Dual-queue ET is a separate policy from ANSMET early-exit.
    if (earlyExitEnable && !dualQueueLowerBoundETEnable) {
        Type et_upperbound = getEarlyTerminationThreshold(query);
        if (query->pDistance[candId] >= et_upperbound) {
            if (!query->inflightDisReqs[candId].empty()) s_trav_earlyexit += 1;
            return true;
        }
    }
    return query->inflightDisReqs[candId].empty();
}

void HNSWTraversalUnit::handleDisReq(DisResp& resp) {
    uint32_t& annsId = resp.annsId;
    if (!queries.count(annsId)) {
        assert(earlyExitEnable || mfnnsEnable); return;
    }
    Query* query = queries[annsId];
    PointId& candId = resp.candId;
    if (!query->inflightCands.count(candId)) {
        assert(earlyExitEnable || mfnnsEnable); return;
    }
    if (!mergeDisReq(query, resp)) return;
    Type pDistance = query->pDistance[candId];
    Type candidateLowerBound = pDistance;
    if (resp.dualQueueLowerBoundValid) {
        candidateLowerBound = resp.dualQueueLowerBound;
    }
    auto lb_it = query->candidateLowerBounds.find(candId);
    if (lb_it != query->candidateLowerBounds.end()) {
        candidateLowerBound = lb_it->second;
    }
    if (resp.dualQueuePruned) {
        s_dual_queue_lb_et += 1;
    }
    query->inflightCands.erase(candId);
    query->inflightDisReqs.erase(candId);
    query->pDistance.erase(candId);
    query->candidateLowerBounds.erase(candId);
    s_disreq_compute_time += resp.respSendCycle - resp.reqRecvCycle;
    s_disreq_req_idle_time += resp.reqRecvCycle - resp.reqSendCycle;
    s_disreq_resp_idle_time += resp.respRecvCycle - resp.respSendCycle;
    s_disreq_mem_service_cycles += resp.memServiceCycles;
    s_disreq_raw_compute_cycles += resp.rawComputeCycles;
    s_disreq_hidden_compute_cycles += resp.hiddenComputeCycles;
    s_disreq_compute_queue_cycles += resp.computeQueueCycles;
    s_disreq_exposed_compute_cycles += resp.exposedComputeCycles;
    s_disreq_time += resp.respRecvCycle - resp.reqSendCycle;
    s_total_disreq += 1;
    query->perQueryDisreqCompleted += 1;
    query->disCompCycles += resp.respSendCycle - resp.reqRecvCycle;
    query->disOffloadCycles += resp.reqRecvCycle - resp.reqSendCycle;
    query->disGatherCycles += resp.respRecvCycle - resp.respSendCycle;

    if (shouldBufferLevel0DualQueueResults(query)) {
        uint64_t batch_order = std::numeric_limits<uint64_t>::max();
        auto order_it = query->level0DualQueueBatchOrder.find(candId);
        if (order_it != query->level0DualQueueBatchOrder.end()) {
            batch_order = order_it->second;
            query->level0DualQueueBatchOrder.erase(order_it);
        }
        query->bufferedLevel0DualQueueResults.push_back(
            Query::BufferedCandidateResult{candId, pDistance, candidateLowerBound, resp.dualQueuePruned, resp.num_access, batch_order});
    } else {
        applyCompletedCandidateResult(query, candId, pDistance, candidateLowerBound, resp.dualQueuePruned, resp.num_access);
    }
    if (query->inflightCands.empty()) {
        flushBufferedLevel0DualQueueResults(query);
        if (query->type == "search")
            finishQueryCheck(annsId);
        else {
            assert(query->type == "construct");
            finishConstructCheck(annsId);
        }
    }
}

std::vector<std::pair<Type, hnswlib::tableint>> HNSWTraversalUnit::PrioQueueToVector(ANNSResultQueue &pq) {
    std::vector<std::pair<Type, hnswlib::tableint>> vec;
    while (!pq.empty()) {
        vec.emplace_back(pq.top().first, pq.top().second);
        pq.pop();
    }
    return vec;
}

std::string DisplayResult(std::vector<std::pair<Type, hnswlib::tableint>> result) {
    std::string res = "";
    for (auto& r : result) {
        res += std::to_string(r.second) + ", ";
    }
    return res;
}

void HNSWTraversalUnit::finishConstructCheck(uint32_t annsId) {
    Query* query = queries[annsId];
    assert(query->type == "construct");
    m_logger->info("[handleConstruct] vectorId {} level {} topChanged {}", query->vectorId, query->level, query->topChanged);
    if (query->vectorId == hnsw->enterpoint_node_) {
        // first construct, exit
        delete queries[annsId];
        queries.erase(annsId);
        s_num_construct += 1;
        m_logger->info("[handleConstruct] finish query {}", annsId);
        print("[handleConstruct] finish query %d", annsId);
    } else if (query->level > query->consLevel && query->topChanged) {
        searchLayer(annsId);
    } else if (query->level > query->consLevel && !query->topChanged) {
        PointDistId nxEntryPoint = query->result.top();
        query->result.pop();
        query->search = ANNSSearchQueue();
        query->result = ANNSResultQueue();
        query->search.emplace(nxEntryPoint);
        query->result.emplace(nxEntryPoint);
        query->level -= 1;
        m_logger->info("[handleConstruct] finish level of query {} level. consLevel {}. Next level {} Next EP {}", annsId, query->level, query->consLevel, nxEntryPoint.second);
        searchLayer(annsId);
    } else if (query->level <= query->consLevel && !query->search.empty()) {
        m_logger->info("[handleConstruct] search not empty");
        searchLayer(annsId);
    } else if (query->level <= query->consLevel && query->search.empty()) {
        assert(queries.count(annsId));
        std::vector<std::pair<Type, hnswlib::tableint>> result = PrioQueueToVector(query->result);
        m_logger->info("[mutualConnect] level {} vectorId {} results ({})", query->level, query->vectorId, DisplayResult(result));
        PointId nxEntryPoint = hnsw->mutualConnect(query->vectorId, result, result.size(), query->level); // graph mem acc already issued during search phase
        hnswlib::linklistsizeint* ll_cur = hnsw->get_linklist_at_level(query->vectorId, query->level);
        uint32_t size = hnsw->getListCount(ll_cur);
        m_logger->info("[mutualConnect] neighbor of {}. size {}", query->vectorId, size);
        hnswlib::tableint *data = (hnswlib::tableint *) (ll_cur + 1);
        for (uint32_t i = 0; i < size; i++) {
            m_logger->info("[mutualConnect] {}: {}", i, data[i]);
        }
        if (query->level == 0) {
            delete queries[annsId];
            queries.erase(annsId);
            s_num_construct += 1;
            m_logger->info("[handleConstruct] finish query {}", annsId);
            print("[handleConstruct] finish query %d", annsId);
            return;
        }
        m_logger->info("[handleConstruct] finish level of query {}. consLevel {}. Next level {} Next EP {}", annsId, query->level, query->consLevel, nxEntryPoint);
        Type dis = hnsw->calcDistance(query->vectorId, nxEntryPoint);
        query->search = ANNSSearchQueue();
        query->result = ANNSResultQueue();
        query->search.emplace(dis, nxEntryPoint);
        query->result.emplace(dis, nxEntryPoint);
        std::fill(query->baseVisitArray, query->baseVisitArray + nData, false);
        query->baseVisitArray[nxEntryPoint] = true;
        searchLayer(annsId);
    }
}

void HNSWTraversalUnit::handleAddPoint(TravReq& travReq) {
    uint32_t annsId = travReq.annsId;
    PointId candId = travReq.candId;
    unsigned int* data = (unsigned int *) hnsw->get_linklist_at_level(candId, travReq.queryLevel);
    uint32_t size = hnsw->getListCount(data);
    PointId* datal = (PointId*) (data + 1);
    Query* query = queries[annsId];
    assert(query->type == "construct");
    Type upperbound = query->result.top().first;
    m_logger->info("[handleAddPoint] annsId {} candId {} level {} neighbor num {}", annsId, candId, travReq.queryLevel, size);
    uint64_t curCycle = m_memory_system->get_clk();
    if (travReq.queryLevel > query->consLevel && size == 0) {
        query->inflightCands.insert(candId);
        query->inflightDisReqs[candId];
        query->pDistance[candId] = 0;
        for (uint32_t i = 0; i < nDim / vDimSize; i++) {
            uint32_t vDimBase = i * vDimSize;
            query->inflightDisReqs[candId].insert(disreqId);
            uint32_t embUnitId = mapDisReq(candId, i, vDimBase, vDimBase + vDimSize, false);
            pendDisReqs[embUnitId].push(DisReq(disreqId++, annsId, candId, query->query, upperbound, vDimBase, vDimBase + vDimSize, nDim, curCycle, travReq.queryLevel));
            m_logger->info("[handleAddPoint] annsId {} candId {} no neighbor send request to candId", annsId, candId);
        }
    } else {
        for (uint32_t i = 0; i < size; i++) {
            PointId candId = datal[i];
            if (candId < 0 || candId > hnsw->max_elements_)
                throw std::runtime_error("candId error");
            if (query->baseVisitArray[candId]) continue;
            query->baseVisitArray[candId] = true;
            query->inflightCands.insert(candId);
            query->inflightDisReqs[candId];
            query->pDistance[candId] = 0;
            for (uint32_t i = 0; i < nDim / vDimSize; i++) {
                uint32_t vDimBase = i * vDimSize;
                query->inflightDisReqs[candId].insert(disreqId);
                m_logger->info("[handleAddPoint] send embId {}: annsId {} candId {} upperbound {}",
                               mapDisReq(candId, i, vDimBase, vDimBase + vDimSize, false),
                               annsId,
                               candId,
                               upperbound);
                uint32_t embUnitId = mapDisReq(candId, i, vDimBase, vDimBase + vDimSize, false);
                pendDisReqs[embUnitId].push(DisReq(disreqId++, annsId, candId, query->query, upperbound, vDimBase, vDimBase + vDimSize, nDim, curCycle, travReq.queryLevel));
            }
        }
        m_logger->info("[handleAddPoint] annsId {} candId {} send total {} requests", annsId, candId, query->inflightCands.size());
        if (query->inflightCands.empty()) finishConstructCheck(annsId);
    }
}

void HNSWTraversalUnit::addPoint(uint32_t annsId) {
    Query* query = queries[annsId];
    assert(query->type == "construct");
    uint32_t consLevel = hnsw->getRandomLevel(hnsw->mult_);
    uint32_t vectorId = nData++;
    query->consLevel = consLevel;
    query->vectorId = vectorId;
    hnsw->element_levels_[vectorId] = consLevel;
    memset(hnsw->data_level0_memory_ + vectorId * hnsw->size_data_per_element_ + hnsw->offsetLevel0_, 0, hnsw->size_data_per_element_);
    //get data in query
    memcpy(hnsw->getDataByInternalId(vectorId), query->query.data(), hnsw->data_size_);
    if (consLevel > 0) {
        hnsw->linkLists_[vectorId] = (char *) malloc(hnsw->size_links_per_element_ * consLevel + 1);
        assert(hnsw->linkLists_[vectorId]);
        memset(hnsw->linkLists_[vectorId], 0, hnsw->size_links_per_element_ * consLevel + 1);
    }
    if (hnsw->enterpoint_node_ == -1U) {
        assert(vectorId == 0);
        hnsw->enterpoint_node_ = vectorId;
        hnsw->maxlevel_ = consLevel;
        finishConstructCheck(annsId);
        return;
    }
    if (consLevel > hnsw->maxlevel_) {
        hnsw->enterpoint_node_ = vectorId;
        hnsw->maxlevel_ = consLevel;
        return;
    }
    query->level = hnsw->maxlevel_;
    query->search.push(std::make_pair(-1U, hnsw->enterpoint_node_)); // dis
    query->result.push(std::make_pair(-1U, hnsw->enterpoint_node_)); // dis
    query->baseVisitArray = new bool[nData];
    std::fill(query->baseVisitArray, query->baseVisitArray + nData, false);
    query->baseVisitArray[hnsw->enterpoint_node_] = true;
    m_logger->info("[addPoint] annsId {} vectorId {} consLevel {}", annsId, vectorId, query->consLevel);
    searchLayer(annsId);
}

Type* HNSWTraversalUnit::getEmbData(PointId candId) {
    return (Type*)hnsw->getDataByInternalId(candId);
}

uint32_t HNSWTraversalUnit::getdimstep(){
    return dimStep_trav;
}

uint32_t HNSWTraversalUnit::getbitstep(){
    return bitStep_trav;
}

uint32_t HNSWTraversalUnit::getdimstep1(){
    return dimStep_trav1;
}

uint32_t HNSWTraversalUnit::getbitstep1(){
    return bitStep_trav1;
}

uint32_t HNSWTraversalUnit::getdimstep2(){
    return dimStep_trav2;
}

uint32_t HNSWTraversalUnit::getbitstep2(){
    return bitStep_trav2;
}
void HNSWTraversalUnit::updatebitSteparray(){
    std::vector<int> arrayValues = param<std::vector<int>>("arrayValues").desc("arrayValues").required();
    if (arrayValues.size() > 32) {
        arrayValues.resize(32);
    } else if (arrayValues.size() < 32) {
        if (!arrayValues.empty()) {
            int lastValue = arrayValues.back();  
            arrayValues.resize(32, lastValue); 
        } else {
            arrayValues.resize(32, 0);
        }
    }
    bitStep_array.resize(32);
    for (size_t i = 0; i < arrayValues.size(); ++i) {
        bitStep_array[i] = arrayValues[i];
    }
}

void HNSWTraversalUnit::init() {
    m_id = 0;
    m_clk = 0;
    graphTravLastIssueValid = false;
    graphTravLastIssueRankId = 0;
    graphTravLastIssuePageId = 0;
    travNodeCallback = [this](Request& req) { return this->handleTravNodeReq(req.trav); };
    travEdgeCallback = [this](Request& req) { return this->handleTravEdgeReq(req.trav); };
    pollCallback = [this](Request& req) { return this->handlePollReq(req); };
    m_logger = Logging::create_logger("HNSWTraversalUnit");
    enableLogging = param<bool>("enableLogging").desc("enableLogging").default_val(false);
    if (!enableLogging) m_logger->set_level(spdlog::level::off);
    datatype=param<std::string>("datatype").desc("datatype").required();
    dataunsigned=param<bool>("dataunsigned").desc("dataunsigned").required();
    spacetype=param<std::string>("spacetype").desc("spacetype").required();
    preprocessVectorDataBitWidth = param<uint32_t>("preprocessVectorDataBitWidth")
                                       .desc("Stored vector bit width used by ANSMET preprocessing/sampling")
                                       .default_val(32 * (datatype == "isFloat") + 8 * (datatype == "isInt"));
    if (datatype == "isFloat" && preprocessVectorDataBitWidth != 16U && preprocessVectorDataBitWidth != 32U) {
        print("[Config] preprocessVectorDataBitWidth=%u invalid for datatype=isFloat. Fallback to 32.", preprocessVectorDataBitWidth);
        preprocessVectorDataBitWidth = 32U;
    }
    if (datatype == "isInt" && preprocessVectorDataBitWidth != 8U &&
        preprocessVectorDataBitWidth != 16U && preprocessVectorDataBitWidth != 32U) {
        print("[Config] preprocessVectorDataBitWidth=%u invalid for datatype=isInt. Fallback to 8.", preprocessVectorDataBitWidth);
        preprocessVectorDataBitWidth = 8U;
    }
    // initilize anns config
    nParallelQuery = param<uint32_t>("nParallelQuery").desc("nParallelQuery").required();
    k_neighbors = param<uint32_t>("k_neighbors").desc("k_neighbors").required();
    ef_search = param<uint32_t>("ef_search").desc("ef_search").default_val(k_neighbors);
    // nDim: number of dimensions
    nDim = param<uint32_t>("nDim").desc("nDim").required();
    // vDimSize: vertical partition size for dimension
    vDimSize = param<uint32_t>("vDimSize").desc("vDimSize").default_val(nDim);
    assert(nDim % vDimSize == 0);
    earlyExitEnable = param<bool>("earlyExitEnable").desc("earlyExitEnable").default_val(false);
    topkThresholdETEnable = param<bool>("topkThresholdETEnable")
                                .desc("Use hnswlib-style top-k threshold for level-0 early termination")
                                .default_val(false);
    dualQueueLowerBoundETEnable = param<bool>("dualQueueLowerBoundETEnable")
                                      .desc("Use aggressive dual-queue lower-bound top-k ET at level-0")
                                      .default_val(false);
    mfnnsEnable = param<bool>("mfnnsEnable")
                      .desc("Enable MFNNS execution semantics: reusable incremental pdis, optional dual-queue sign+exp -> mantissa flow")
                      .default_val(dualQueueLowerBoundETEnable);
    debugIssueTraceEnable = param<bool>("debugIssueTraceEnable")
                                .desc("Debug baseVisitArray decisions and DisReq issue for a target query")
                                .default_val(false);
    debugIssueTraceQueryId = param<uint32_t>("debugIssueTraceQueryId")
                                 .desc("Target query id for issue/baseVisit debug")
                                 .default_val(0);
    debugIssueTracePath = param<std::string>("debugIssueTracePath")
                              .desc("Optional output path for issue/baseVisit debug dump")
                              .default_val("");
    perQuerySummaryEnable = param<bool>("perQuerySummaryEnable")
                                .desc("Dump lightweight per-query traversal/disreq summary TSV")
                                .default_val(false);
    perQuerySummaryPath = param<std::string>("perQuerySummaryPath")
                              .desc("Optional output path for per-query summary TSV")
                              .default_val("");
    debugDuplicateAcceptEnable = param<bool>("debugDuplicateAcceptEnable")
                                     .desc("Debug whether a target query accepts the same (level,candId) more than once")
                                     .default_val(false);
    debugDuplicateAcceptQueryId = param<uint32_t>("debugDuplicateAcceptQueryId")
                                      .desc("Target query id for duplicate-accept debug")
                                      .default_val(0);
    debugDuplicateAcceptPath = param<std::string>("debugDuplicateAcceptPath")
                                   .desc("Optional output path for duplicate-accept debug dump")
                                   .default_val("");
    dualQueueLowerBoundQueueSize = param<uint32_t>("dualQueueLowerBoundQueueSize")
                                       .desc("Queue depth used by dual-queue lower-bound pruning")
                                       .default_val(k_neighbors);
    dualQueueLowerBoundWarmupSize = param<uint32_t>("dualQueueLowerBoundWarmupSize")
                                        .desc("Minimum lower-bound queue occupancy before dual-queue pruning activates")
                                        .default_val(dualQueueLowerBoundQueueSize);
    vectorLayoutMode = param<std::string>("vectorLayoutMode")
                           .desc("Legacy vector layout mode: auto|naive_linear|bank_striped|hot_node_replication|coarse_fine_split")
                           .default_val("auto");
    vectorLogicalLayoutMode = param<std::string>("vectorLogicalLayout")
                                  .desc("Logical vector layout: auto|linear|coarse_fine_split|row_aligned_coarse_fine_split|global_coarse_fine_split")
                                  .default_val("auto");
    layout2GlobalCoarseThenFineEnable = param<bool>("layout2GlobalCoarseThenFineEnable")
                                            .desc("Shortcut switch to force layout-2 global coarse-then-fine vector layout")
                                            .default_val(false);
    dualQueueAllowLinearFullReadLayout = param<bool>("dualQueueAllowLinearFullReadLayout")
                                             .desc("Allow dual-queue ET to keep linear layout by making phase-1 read full-width vector data")
                                             .default_val(false);
    vectorPhysicalPlacementMode = param<std::string>("vectorPhysicalPlacement")
                                      .desc("Physical vector placement: auto|naive|striped|replicated|toplayer_replicated")
                                      .default_val("auto");
    vectorRankPlacementMode = param<std::string>("vectorRankPlacement")
                                  .desc("Vector rank placement: auto|modulo|kmeans_balanced|kmeans_cluster_rr|graph_clustered_balanced|graph_balanced_multistart|modulo_bfs_slot")
                                  .default_val("auto");
    vectorRankPlacementChunkSize = param<uint32_t>("vectorRankPlacementChunkSize")
                                       .desc("Base graph-clustered placement chunk size in vectors before super-chunk expansion")
                                       .default_val(128);
    vectorRankPlacementSuperChunkFactor = param<uint32_t>("vectorRankPlacementSuperChunkFactor")
                                              .desc("Multiplier applied to vectorRankPlacementChunkSize to form graph-clustered super-chunk windows")
                                              .default_val(1);
    vectorRankPlacementStripeGroupSize = param<uint32_t>("vectorRankPlacementStripeGroupSize")
                                             .desc("Contiguous local-rank group size used for intra-chunk striping inside each graph super-chunk")
                                             .default_val(1);
    vectorRankPlacementStripeBlockSize = param<uint32_t>("vectorRankPlacementStripeBlockSize")
                                             .desc("Number of consecutive vectors assigned before striping to the next rank in the selected rank-group")
                                             .default_val(1);
    vectorRankPlacementLevelAwareStripeEnable = param<bool>("vectorRankPlacementLevelAwareStripeEnable")
                                                    .desc("Enable separate top-level stripe parameters for graph-clustered vector placement")
                                                    .default_val(false);
    vectorRankPlacementTopLevelCount = param<uint32_t>("vectorRankPlacementTopLevelCount")
                                           .desc("Number of highest HNSW levels that use the top-level stripe parameters when level-aware striping is enabled")
                                           .default_val(0);
    vectorRankPlacementTopStripeGroupSize = param<uint32_t>("vectorRankPlacementTopStripeGroupSize")
                                                .desc("Stripe group size used for the selected top HNSW levels when level-aware striping is enabled")
                                                .default_val(1);
    vectorRankPlacementTopStripeBlockSize = param<uint32_t>("vectorRankPlacementTopStripeBlockSize")
                                                .desc("Stripe block size used for the selected top HNSW levels when level-aware striping is enabled")
                                                .default_val(1);
    vectorAddressMappingMode = param<std::string>("vectorAddressMapping")
                                   .desc("Vector address mapping override: auto|naive|striped")
                                   .default_val("auto");
    vectorAddressMappingPhase1Mode = param<std::string>("vectorAddressMappingPhase1")
                                         .desc("Phase-1 vector address mapping override: auto|naive|striped")
                                         .default_val("auto");
    vectorAddressMappingPhase2Mode = param<std::string>("vectorAddressMappingPhase2")
                                         .desc("Phase-2 vector address mapping override: auto|naive|striped")
                                         .default_val("auto");
    vectorPhaseBankPartitionEnable = param<bool>("vectorPhaseBankPartitionEnable")
                                         .desc("Force phase-1 and phase-2 vector requests onto disjoint bank-group subsets")
                                         .default_val(false);
    vectorPhase1DedicatedBGCount = param<uint32_t>("vectorPhase1DedicatedBGCount")
                                       .desc("Number of bank-groups reserved for phase-1 when vectorPhaseBankPartitionEnable=true")
                                       .default_val(2);
    vectorReplicaDispatchMode = param<std::string>("vectorReplicaDispatch")
                                    .desc("Vector replica dispatch override: auto|none|hot_node")
                                    .default_val("auto");
    hotNodeReplicationCount = param<uint32_t>("hotNodeReplicationCount")
                                  .desc("Number of hot nodes eligible for replica-aware dispatch")
                                  .default_val(10000);
    hotNodeReplicationTopLevelCount = param<uint32_t>("hotNodeReplicationTopLevelCount")
                                          .desc("Replicate/select all nodes in the highest N HNSW levels before applying count-based hot-node fill")
                                          .default_val(0);
    hotReplicaRowAwareEnable = param<bool>("hotReplicaRowAwareEnable")
                                   .desc("Use coarse-first row-aware hot-replica selection")
                                   .default_val(false);
    hotReplicaLoadWeight = param<uint32_t>("hotReplicaLoadWeight")
                               .desc("Load penalty weight for hot-replica selection")
                               .default_val(1);
    hotReplicaBgPenaltyWeight = param<uint32_t>("hotReplicaBgPenaltyWeight")
                                    .desc("BG occupancy penalty weight for hot-replica selection")
                                    .default_val(4);
    hotReplicaBankPenaltyWeight = param<uint32_t>("hotReplicaBankPenaltyWeight")
                                      .desc("Bank occupancy penalty weight for hot-replica selection")
                                      .default_val(8);
    hotReplicaRowHitBonus = param<uint32_t>("hotReplicaRowHitBonus")
                                .desc("Bank row-hit bonus for hot-replica selection")
                                .default_val(32);
    hotReplicaBgRowHitBonus = param<uint32_t>("hotReplicaBgRowHitBonus")
                                  .desc("BG row-hit bonus for hot-replica selection")
                                  .default_val(16);
    graphStaticScheduleMode = param<std::string>("graphStaticScheduleMode")
                                  .desc("Graph static schedule mode: legacy_word_interleave|level0_page_blocked")
                                  .default_val("legacy_word_interleave");
    graphVertexReorderMode = param<std::string>("graphVertexReorderMode")
                                 .desc("Graph vertex reorder mode: none|degree_bfs")
                                 .default_val("none");
    graphPlacementPageLines = param<uint32_t>("graphPlacementPageLines")
                                  .desc("Synthetic graph page size in cache lines for static placement")
                                  .default_val(64);
    graphTraversalBatchingEnable = param<bool>("graphTraversalBatchingEnable")
                                       .desc("Enable graph traversal page/rank-aware batching scheduler")
                                       .default_val(false);
    graphTraversalBatchingWindow = param<uint32_t>("graphTraversalBatchingWindow")
                                       .desc("Lookahead window size for graph traversal batching")
                                       .default_val(32);
    graphTraversalBatchingAgeThreshold = param<uint64_t>("graphTraversalBatchingAgeThreshold")
                                             .desc("Age threshold before graph traversal batching forces oldest request")
                                             .default_val(256);
    graphTraversalBatchingPageWeight = param<uint32_t>("graphTraversalBatchingPageWeight")
                                           .desc("Page-frequency weight for graph traversal batching")
                                           .default_val(8);
    graphTraversalBatchingRankWeight = param<uint32_t>("graphTraversalBatchingRankWeight")
                                           .desc("Rank-frequency weight for graph traversal batching")
                                           .default_val(2);
    graphTraversalBatchingLastPageBonus = param<uint32_t>("graphTraversalBatchingLastPageBonus")
                                              .desc("Last-issued page bonus for graph traversal batching")
                                              .default_val(4);
    graphTraversalBatchingLastRankBonus = param<uint32_t>("graphTraversalBatchingLastRankBonus")
                                              .desc("Last-issued rank bonus for graph traversal batching")
                                              .default_val(1);
    if (dualQueueLowerBoundETEnable && earlyExitEnable) {
        print("[Config] dualQueueLowerBoundETEnable=true, force earlyExitEnable=false to decouple from ANSMET ET.");
        earlyExitEnable = false;
    }
    if (mfnnsEnable && !dualQueueLowerBoundETEnable && earlyExitEnable) {
        print("[Config] mfnnsEnable=true with dualQueueLowerBoundETEnable=false: force earlyExitEnable=false and use MFNNS linear incremental ET.");
        earlyExitEnable = false;
    }
    if (dualQueueLowerBoundETEnable && (nDim / vDimSize) != 1) {
        print("[Config] dualQueueLowerBoundETEnable currently requires vDimSize==nDim. Disable dual-queue ET.");
        dualQueueLowerBoundETEnable = false;
    }
    if (topkThresholdETEnable && !earlyExitEnable) {
        print("[Config] topkThresholdETEnable=true but earlyExitEnable=false, this option is inactive.");
    }
    if (dualQueueLowerBoundETEnable && !earlyExitEnable) {
        print("[Config] dualQueueLowerBoundETEnable=true with earlyExitEnable=false: dual-queue ET runs independently.");
        print("[Config] dualQueueLowerBoundQueueSize=%d warmup=%d", dualQueueLowerBoundQueueSize, dualQueueLowerBoundWarmupSize);
    }
    if (dualQueueLowerBoundETEnable && topkThresholdETEnable) {
        print("[Config] dualQueueLowerBoundETEnable=true overrides topkThresholdETEnable at level-0 ET threshold.");
    }
    if (mfnnsEnable) {
        print("[Config] mfnnsEnable=true dualQueueLowerBoundETEnable=%d", dualQueueLowerBoundETEnable ? 1 : 0);
    }
    auto normalizeLogicalLayout = [](const std::string& raw) {
        if (raw == "linear" || raw == "naive_linear") return std::string("linear");
        if (raw == "coarse_fine_split") return std::string("coarse_fine_split");
        if (raw == "row_aligned_coarse_fine_split") return std::string("row_aligned_coarse_fine_split");
        if (raw == "global_coarse_fine_split") return std::string("global_coarse_fine_split");
        if (raw == "auto") return std::string("auto");
        return std::string();
    };
    auto normalizePhysicalPlacement = [](const std::string& raw) {
        if (raw == "naive" || raw == "naive_linear") return std::string("naive");
        if (raw == "striped" || raw == "bank_striped") return std::string("striped");
        if (raw == "replicated" || raw == "hot_node_replication") return std::string("replicated");
        if (raw == "toplayer_replicated") return std::string("toplayer_replicated");
        if (raw == "auto") return std::string("auto");
        return std::string();
    };
    auto normalizeAddressMapping = [](const std::string& raw) {
        if (raw == "naive") return std::string("naive");
        if (raw == "striped" || raw == "bank_striped") return std::string("striped");
        if (raw == "auto") return std::string("auto");
        return std::string();
    };
    auto normalizeRankPlacement = [](const std::string& raw) {
        if (raw == "modulo") return std::string("modulo");
        if (raw == "kmeans_balanced") return std::string("kmeans_balanced");
        if (raw == "kmeans_cluster_rr") return std::string("kmeans_cluster_rr");
        if (raw == "graph_clustered_balanced") return std::string("graph_clustered_balanced");
        if (raw == "graph_balanced_multistart") return std::string("graph_balanced_multistart");
        if (raw == "modulo_bfs_slot") return std::string("modulo_bfs_slot");
        if (raw == "auto") return std::string("auto");
        return std::string();
    };
    auto normalizeReplicaDispatch = [](const std::string& raw) {
        if (raw == "none") return std::string("none");
        if (raw == "hot_node" || raw == "hot_node_replication") return std::string("hot_node");
        if (raw == "auto") return std::string("auto");
        return std::string();
    };
    auto normalizeGraphStaticSchedule = [](const std::string& raw) {
        if (raw == "legacy_word_interleave") return std::string("legacy_word_interleave");
        if (raw == "level0_page_blocked") return std::string("level0_page_blocked");
        return std::string();
    };
    auto normalizeGraphVertexReorder = [](const std::string& raw) {
        if (raw == "none") return std::string("none");
        if (raw == "degree_bfs") return std::string("degree_bfs");
        return std::string();
    };

    std::string legacyLogical = dualQueueLowerBoundETEnable ? "coarse_fine_split" : "linear";
    std::string legacyPhysical = "naive";
    std::string legacyAddressMapping = "naive";
    std::string legacyReplicaDispatch = "none";
    if (vectorLayoutMode != "auto") {
        if (vectorLayoutMode == "naive_linear") {
            legacyLogical = "linear";
            legacyPhysical = "naive";
            legacyAddressMapping = "naive";
            legacyReplicaDispatch = "none";
        } else if (vectorLayoutMode == "bank_striped") {
            legacyLogical = "linear";
            legacyPhysical = "striped";
            legacyAddressMapping = "striped";
            legacyReplicaDispatch = "none";
        } else if (vectorLayoutMode == "hot_node_replication") {
            legacyLogical = "linear";
            legacyPhysical = "replicated";
            legacyAddressMapping = "striped";
            legacyReplicaDispatch = "hot_node";
        } else if (vectorLayoutMode == "coarse_fine_split") {
            legacyLogical = "coarse_fine_split";
            legacyPhysical = "naive";
            legacyAddressMapping = "naive";
            legacyReplicaDispatch = "none";
        } else {
            print("[Config] unsupported vectorLayoutMode=%s, fallback to auto policy.", vectorLayoutMode.c_str());
            vectorLayoutMode = "auto";
        }
    }
    if (vectorPhysicalPlacementMode != "auto") {
        if (vectorPhysicalPlacementMode == "naive") {
            legacyAddressMapping = "naive";
            legacyReplicaDispatch = "none";
        } else if (vectorPhysicalPlacementMode == "striped") {
            legacyAddressMapping = "striped";
            legacyReplicaDispatch = "none";
        } else if (vectorPhysicalPlacementMode == "replicated") {
            legacyAddressMapping = "striped";
            legacyReplicaDispatch = "hot_node";
        }
    }

    std::string requestedLogical = normalizeLogicalLayout(vectorLogicalLayoutMode);
    if (requestedLogical.empty()) {
        print("[Config] unsupported vectorLogicalLayout=%s, fallback to auto.", vectorLogicalLayoutMode.c_str());
        requestedLogical = "auto";
    }
    std::string requestedPhysical = normalizePhysicalPlacement(vectorPhysicalPlacementMode);
    if (requestedPhysical.empty()) {
        print("[Config] unsupported vectorPhysicalPlacement=%s, fallback to auto.", vectorPhysicalPlacementMode.c_str());
        requestedPhysical = "auto";
    }
    std::string requestedAddressMapping = normalizeAddressMapping(vectorAddressMappingMode);
    if (requestedAddressMapping.empty()) {
        print("[Config] unsupported vectorAddressMapping=%s, fallback to auto.", vectorAddressMappingMode.c_str());
        requestedAddressMapping = "auto";
    }
    std::string requestedRankPlacement = normalizeRankPlacement(vectorRankPlacementMode);
    if (requestedRankPlacement.empty()) {
        print("[Config] unsupported vectorRankPlacement=%s, fallback to auto.", vectorRankPlacementMode.c_str());
        requestedRankPlacement = "auto";
    }
    std::string requestedPhase1AddressMapping = normalizeAddressMapping(vectorAddressMappingPhase1Mode);
    if (requestedPhase1AddressMapping.empty()) {
        print("[Config] unsupported vectorAddressMappingPhase1=%s, fallback to auto.", vectorAddressMappingPhase1Mode.c_str());
        requestedPhase1AddressMapping = "auto";
    }
    std::string requestedPhase2AddressMapping = normalizeAddressMapping(vectorAddressMappingPhase2Mode);
    if (requestedPhase2AddressMapping.empty()) {
        print("[Config] unsupported vectorAddressMappingPhase2=%s, fallback to auto.", vectorAddressMappingPhase2Mode.c_str());
        requestedPhase2AddressMapping = "auto";
    }
    std::string requestedReplicaDispatch = normalizeReplicaDispatch(vectorReplicaDispatchMode);
    if (requestedReplicaDispatch.empty()) {
        print("[Config] unsupported vectorReplicaDispatch=%s, fallback to auto.", vectorReplicaDispatchMode.c_str());
        requestedReplicaDispatch = "auto";
    }
    std::string requestedGraphStaticSchedule = normalizeGraphStaticSchedule(graphStaticScheduleMode);
    if (requestedGraphStaticSchedule.empty()) {
        print("[Config] unsupported graphStaticScheduleMode=%s, fallback to legacy_word_interleave.", graphStaticScheduleMode.c_str());
        requestedGraphStaticSchedule = "legacy_word_interleave";
    }
    std::string requestedGraphVertexReorder = normalizeGraphVertexReorder(graphVertexReorderMode);
    if (requestedGraphVertexReorder.empty()) {
        print("[Config] unsupported graphVertexReorderMode=%s, fallback to none.", graphVertexReorderMode.c_str());
        requestedGraphVertexReorder = "none";
    }

    vectorLogicalLayoutMode = (requestedLogical == "auto") ? legacyLogical : requestedLogical;
    if (layout2GlobalCoarseThenFineEnable) {
        if (vectorLogicalLayoutMode != "global_coarse_fine_split") {
            print("[Config] layout2GlobalCoarseThenFineEnable=true overrides vectorLogicalLayout=%s -> global_coarse_fine_split.",
                  vectorLogicalLayoutMode.c_str());
        }
        vectorLogicalLayoutMode = "global_coarse_fine_split";
    }
    if (dualQueueLowerBoundETEnable &&
        vectorLogicalLayoutMode != "coarse_fine_split" &&
        vectorLogicalLayoutMode != "row_aligned_coarse_fine_split" &&
        vectorLogicalLayoutMode != "global_coarse_fine_split") {
        if (dualQueueAllowLinearFullReadLayout && vectorLogicalLayoutMode == "linear") {
            print("[Config] dualQueueAllowLinearFullReadLayout=true keeps vectorLogicalLayout=linear for phase-1 full-read baseline.");
        } else {
            print("[Config] dualQueueLowerBoundETEnable requires coarse/fine logical layout. Override vectorLogicalLayout=%s -> coarse_fine_split.",
                  vectorLogicalLayoutMode.c_str());
            vectorLogicalLayoutMode = "coarse_fine_split";
        }
    }
    vectorPhysicalPlacementMode = (requestedPhysical == "auto") ? legacyPhysical : requestedPhysical;
    vectorRankPlacementMode = (requestedRankPlacement == "auto") ? "modulo" : requestedRankPlacement;
    if (vectorRankPlacementSuperChunkFactor == 0) {
        print("[Config] vectorRankPlacementSuperChunkFactor=0 is invalid. Clamp to 1.");
        vectorRankPlacementSuperChunkFactor = 1;
    }
    if (vectorRankPlacementStripeGroupSize == 0) {
        print("[Config] vectorRankPlacementStripeGroupSize=0 is invalid. Clamp to 1.");
        vectorRankPlacementStripeGroupSize = 1;
    }
    if (vectorRankPlacementStripeBlockSize == 0) {
        print("[Config] vectorRankPlacementStripeBlockSize=0 is invalid. Clamp to 1.");
        vectorRankPlacementStripeBlockSize = 1;
    }
    if (vectorRankPlacementTopStripeGroupSize == 0) {
        print("[Config] vectorRankPlacementTopStripeGroupSize=0 is invalid. Clamp to 1.");
        vectorRankPlacementTopStripeGroupSize = 1;
    }
    if (vectorRankPlacementTopStripeBlockSize == 0) {
        print("[Config] vectorRankPlacementTopStripeBlockSize=0 is invalid. Clamp to 1.");
        vectorRankPlacementTopStripeBlockSize = 1;
    }
    resolvedVectorAddressMappingMode = (requestedAddressMapping == "auto") ? legacyAddressMapping : requestedAddressMapping;
    resolvedVectorAddressMappingPhase1Mode = (requestedPhase1AddressMapping == "auto")
        ? resolvedVectorAddressMappingMode
        : requestedPhase1AddressMapping;
    resolvedVectorAddressMappingPhase2Mode = (requestedPhase2AddressMapping == "auto")
        ? resolvedVectorAddressMappingMode
        : requestedPhase2AddressMapping;
    resolvedVectorReplicaDispatchMode = (requestedReplicaDispatch == "auto") ? legacyReplicaDispatch : requestedReplicaDispatch;
    if (vectorPhaseBankPartitionEnable) {
        const uint32_t clampedPhase1BG = std::min<uint32_t>(3U, std::max<uint32_t>(1U, vectorPhase1DedicatedBGCount));
        if (clampedPhase1BG != vectorPhase1DedicatedBGCount) {
            print("[Config] vectorPhase1DedicatedBGCount=%u is invalid for partitioning. Clamp to %u.",
                  vectorPhase1DedicatedBGCount,
                  clampedPhase1BG);
            vectorPhase1DedicatedBGCount = clampedPhase1BG;
        }
    }
    graphStaticScheduleMode = requestedGraphStaticSchedule;
    graphVertexReorderMode = requestedGraphVertexReorder;
    resolvedVectorLayoutMode = vectorLogicalLayoutMode + "+" + vectorPhysicalPlacementMode +
                               "[addr=" + resolvedVectorAddressMappingMode +
                               ",replica=" + resolvedVectorReplicaDispatchMode + "]";
    print("[Config] vector layout logical=%s physical=%s rankPlacement=%s rankBaseChunk=%u rankSuperChunkFactor=%u rankEffectiveChunk=%u rankStripeGroup=%u rankStripeBlock=%u resolved=%s addressMapping=%s phase1Addr=%s phase2Addr=%s phaseBankPartition=%d phase1DedicatedBG=%u replicaDispatch=%s hotNodeReplicationCount=%u hotNodeReplicationTopLevelCount=%u hotReplicaRowAware=%d layout2GlobalCoarseThenFineEnable=%d",
          vectorLogicalLayoutMode.c_str(),
          vectorPhysicalPlacementMode.c_str(),
          vectorRankPlacementMode.c_str(),
          getVectorRankPlacementBaseChunkSize(),
          getVectorRankPlacementSuperChunkFactor(),
          getVectorRankPlacementEffectiveChunkSize(),
          getVectorRankPlacementStripeGroupSize(),
          getVectorRankPlacementStripeBlockSize(),
          resolvedVectorLayoutMode.c_str(),
          resolvedVectorAddressMappingMode.c_str(),
          resolvedVectorAddressMappingPhase1Mode.c_str(),
          resolvedVectorAddressMappingPhase2Mode.c_str(),
          vectorPhaseBankPartitionEnable ? 1 : 0,
          vectorPhase1DedicatedBGCount,
          resolvedVectorReplicaDispatchMode.c_str(),
          hotNodeReplicationCount,
          hotNodeReplicationTopLevelCount,
          hotReplicaRowAwareEnable ? 1 : 0,
          layout2GlobalCoarseThenFineEnable ? 1 : 0);
    print("[Config] vector rank placement levelAware=%d topLevels=%u topStripeGroup=%u topStripeBlock=%u",
          vectorRankPlacementLevelAwareStripeEnable ? 1 : 0,
          vectorRankPlacementTopLevelCount,
          vectorRankPlacementTopStripeGroupSize,
          vectorRankPlacementTopStripeBlockSize);
    print("[Config] graph static schedule mode=%s reorder=%s pageLines=%u",
          graphStaticScheduleMode.c_str(),
          graphVertexReorderMode.c_str(),
          graphPlacementPageLines);
    print("[Config] graph traversal batching enable=%d window=%u ageThreshold=%lu pageWeight=%u rankWeight=%u lastPageBonus=%u lastRankBonus=%u",
          graphTraversalBatchingEnable ? 1 : 0,
          graphTraversalBatchingWindow,
          graphTraversalBatchingAgeThreshold,
          graphTraversalBatchingPageWeight,
          graphTraversalBatchingRankWeight,
          graphTraversalBatchingLastPageBonus,
          graphTraversalBatchingLastRankBonus);
    nEmbUnit = param<uint32_t>("nEmbUnit").desc("nEmbUnit").required();
    assert(nEmbUnit % (nDim / vDimSize) == 0);
    pendDisReqs = std::vector<std::queue<DisReq>>(nEmbUnit);
    traversalMemAcc = param<bool>("traversalMemAcc").desc("traversalMemAcc").default_val(true);
    debug_minDisReqMap = param<bool>("debug_minDisReqMap").desc("debug_minDisReqMap").default_val(false);
    debug_epCopy = param<bool>("debug_epCopy").desc("debug_epCopy").default_val(false);
    if (debugIssueTraceEnable) {
        print("[Config] debugIssueTraceEnable=true queryId=%u output=%s",
              debugIssueTraceQueryId,
              debugIssueTracePath.empty() ? "<stat_path-derived>" : debugIssueTracePath.c_str());
    }
    if (perQuerySummaryEnable) {
        print("[Config] perQuerySummaryEnable=true output=%s",
              perQuerySummaryPath.empty() ? "<stat_path-derived>" : perQuerySummaryPath.c_str());
    }
    if (debugDuplicateAcceptEnable) {
        print("[Config] debugDuplicateAcceptEnable=true queryId=%u output=%s",
              debugDuplicateAcceptQueryId,
              debugDuplicateAcceptPath.empty() ? "<stat_path-derived>" : debugDuplicateAcceptPath.c_str());
    }
    nPackingVec = param<uint32_t>("nPackingVec").desc("nPackingVec").default_val(1);
    resProbeEpoch = param<uint32_t>("resProbeEpoch").desc("resProbeEpoch").default_val(25);
    pollingSimulation = param<bool>("pollingSimulation").desc("pollingSimulation").default_val(false);
    adaptiveResultProbe = param<bool>("adaptiveResultProbe").desc("adaptiveResultProbe").default_val(false);

    // initialize stats
    register_stat(s_num_travedge_req).name("s_num_travedge_req");
    register_stat(s_num_travnode_req).name("s_num_travnode_req");
    register_stat(s_num_query).name("s_num_query");
    register_stat(s_num_construct).name("s_num_construct");
    register_stat(s_trav_earlyexit).name("s_trav_earlyexit");
    register_stat(s_dual_queue_lb_et).name("s_dual_queue_lb_et");
    register_stat(s_hot_replica_remap).name("s_hot_replica_remap");
    register_stat(s_hot_replica_rowaware_selected).name("s_hot_replica_rowaware_selected");
    register_stat(s_hot_replica_rowhit_selected).name("s_hot_replica_rowhit_selected");
    register_stat(s_hot_replica_bg_rowhit_selected).name("s_hot_replica_bg_rowhit_selected");
    register_stat(s_graph_level0_layout_total_lines).name("s_graph_level0_layout_total_lines");
    register_stat(s_graph_level0_layout_padding_lines).name("s_graph_level0_layout_padding_lines");
    register_stat(s_graph_level0_degree_bfs_neighbor_visits).name("s_graph_level0_degree_bfs_neighbor_visits");
    register_stat(s_graph_trav_batch_reordered).name("s_graph_trav_batch_reordered");
    register_stat(s_graph_trav_batch_page_match_selected).name("s_graph_trav_batch_page_match_selected");
    register_stat(s_graph_trav_batch_rank_match_selected).name("s_graph_trav_batch_rank_match_selected");
    register_stat(s_graph_trav_batch_age_forced).name("s_graph_trav_batch_age_forced");
    register_stat(s_num_total_disreq).name("s_num_total_disreq");
    register_stat(s_num_packed_disreq).name("s_num_packed_disreq");
    register_stat(s_num_packed_saving).name("s_num_packed_saving");
    register_stat(s_total_disreq).name("s_total_disreq");
    register_stat(s_disreq_time).name("s_disreq_time");
    register_stat(s_disreq_compute_time).name("s_disreq_compute_time");
    register_stat(s_disreq_req_idle_time).name("s_disreq_req_idle_time");
    register_stat(s_disreq_resp_idle_time).name("s_disreq_resp_idle_time");
    register_stat(s_disreq_mem_service_cycles).name("s_disreq_mem_service_cycles");
    register_stat(s_disreq_raw_compute_cycles).name("s_disreq_raw_compute_cycles");
    register_stat(s_disreq_hidden_compute_cycles).name("s_disreq_hidden_compute_cycles");
    register_stat(s_disreq_compute_queue_cycles).name("s_disreq_compute_queue_cycles");
    register_stat(s_disreq_exposed_compute_cycles).name("s_disreq_exposed_compute_cycles");
    register_stat(s_num_result_probe).name("s_num_result_probe");
    register_stat(s_mem_cycle).name("s_mem_cycle");
    register_stat(s_mem_read_req).name("s_mem_read_req");
    register_stat(s_mem_write_req).name("s_mem_write_req");
    register_stat(s_avg_total_latency).name("s_avg_total_latency");
    register_stat(s_recall_rate).name("s_recall_rate");
    register_stat(s_same_rank_run_length_mean).name("s_same_rank_run_length_mean");
    register_stat(s_same_rank_run_length_p95).name("s_same_rank_run_length_p95");
    s_query_latency_bin = std::vector<uint64_t>(nLatBin, 0);
    // vector packing related counters
    s_num_total_disreq = std::vector<uint64_t>(10, 0);
    s_num_packed_disreq = std::vector<uint64_t>(10, 0);
    s_num_packed_saving = std::vector<uint64_t>(10, 0);
    s_top_layer_access_per_embunit = std::vector<uint64_t>(nEmbUnit, 0);
    s_num_total_pdisreq = std::vector<uint64_t>(nEmbUnit, 0);

    // load model and query
    bool loadModel = param<bool>("loadModel").desc("loadModel").default_val(true);
    if (loadModel) {
        std::string model_path = param<std::string>("model_path").desc("Path to the load HNSW graph.").required();
        hnswlib::L2Space space(nDim);
        hnsw = new hnswlib::HierarchicalNSW<Type>(&space, model_path);
        nData = hnsw->cur_element_count;
        print("Load index from %s, with %d elements and %d dims", model_path.c_str(), nData, nDim);
        prepareHotReplicaNodes();
        prepareGraphStaticSchedule();
        prepareVectorRankPlacement();
    } else {
        hnswlib::L2Space space(nDim);
        uint32_t hnsw_cap = param<uint32_t>("hnsw_cap").desc("hnsw_cap").required();
        uint32_t hnsw_M = param<uint32_t>("hnsw_M").desc("hnsw_M").default_val(16);
        uint32_t hnsw_ef_cons = param<uint32_t>("hnsw_ef_cons").desc("hnsw_ef_cons").default_val(200);
        hnsw = new hnswlib::HierarchicalNSW<Type>(&space, hnsw_cap, hnsw_M, hnsw_ef_cons);
        nData = 0;
        print("Create an empty hnsw graph, with %d capacity and %d dims", hnsw_cap, nDim);
        prepareHotReplicaNodes();
        prepareGraphStaticSchedule();
        prepareVectorRankPlacement();
    }
    // Read query file first to get correct nQuery
    std::string queryType = param<std::string>("queryType").desc("queryType").default_val("search");
    query_path = param<std::string>("query_path").desc("Path to the load HNSW graph.").required();
    std::vector<Type> all_queries;
    read_vec<Type>(query_path, all_queries, nQuery, nDim);
    uint32_t nQueryLimit = param<uint32_t>("nQueryLimit").desc("nQueryLimit").default_val(-1U);
    nQuery = std::min(nQuery, nQueryLimit);
    print("Load %d queries from %s. Type: %s", nQuery, query_path.c_str(), queryType.c_str());
    
    // Then read ground truth file using the correct nQuery
    std::string gt_path = param<std::string>("gt_path").desc("Path to ground truth vector.").default_val("");
    if (gt_path != "") {
        gt_k = param<uint32_t>("gt_k").desc("gt_k").required();
        uint32_t gt_nQuery;
        read_vec<uint32_t>(gt_path, gt, gt_nQuery, gt_k);
        assert(gt_nQuery >= nQuery);  // Ground truth should have at least as many queries
        print("Load ground truth from %s. nQuery %d gt_k %d", gt_path.c_str(), nQuery, gt_k);
    }
    if (queryType == "search") {
        for (uint32_t i = 0; i < nQuery; i++) {
            Point q;
            for (uint32_t d = 0; d < nDim; d++) {
                q.emplace_back(all_queries[i * nDim + d]);
            }
            pendQueries.push(new Query(i, "search", q, hnsw->enterpoint_node_, hnsw->maxlevel_));
        }
        print("Search entrypoint %d", hnsw->enterpoint_node_);
    } else {
        assert(queryType == "construct");
        for (uint32_t i = 0; i < nQuery; i++) {
            Point q;
            for (uint32_t d = 0; d < nDim; d++) {
                q.emplace_back(all_queries[i * nDim + d]);
            }
            pendQueries.push(new Query(i, "construct", q));
        }
    }

    int level_count[10] = {0}; 

    for (int candId = 0; candId < nData; candId++) {
        uint32_t level = hnsw->element_levels_[candId];
        if (level < 10) { 
            level_count[level]++;
        }
    }
    statFile = param<std::string>("stat_path").desc("Path for results").required();
    disreqId = 0;
    graphEdgeAddrBase = graphNodeAddrBase + nData * sizeof(PointId);
    allowSample=param<bool>("allowSample").desc("allowSample").required();
    const uint32_t configured_main_dim_step =
        std::max<uint32_t>(1U, param<uint32_t>("dimStep").desc("dimStep").required());
    const uint32_t configured_main_bit_step =
        std::max<uint32_t>(1U, param<uint32_t>("bitStep").desc("bitStep").required());
    const uint32_t configured_stage1_bit_step =
        std::max<uint32_t>(1U, param<uint32_t>("bitStep_trav1").desc("bitStep_trav1").required());
    const uint32_t configured_stage2_bit_step =
        std::max<uint32_t>(1U, param<uint32_t>("bitStep_trav2").desc("bitStep_trav2").required());
    const int configured_coarse_fetch_count =
        param<int>("ansmetDualGranularityCoarseFetchCount")
            .desc("ANSMET dual-granularity coarse-bit fetch count (T_C); default 1 keeps two-stage [coarse,fine]")
            .default_val(1);
    limit_bs_num = param<bool>("limit_bs_num").desc("limit_bs_num").required();
    bs_num = param<int>("bs_num").desc("bs_num").required();
    print("preprocessing...")
    
    if(allowSample){
        auto start = std::chrono::high_resolution_clock::now();
        preprocess();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

    }else{
        dimbitMode=param<std::string>("dimbitMode").desc("dimbitMode").required();
        bitStep_trav1 = configured_stage1_bit_step;
        bitStep_trav2 = configured_stage2_bit_step;
        hnsw->col_cut = 0;
        hnsw->num_outlier_ = 0;
        hnsw->outlier_list.resize(hnsw->max_elements_);
        for(int ii = 0; ii < hnsw->max_elements_; ii++){
            hnsw->outlier_list[ii] = 0;
        }
        hnsw->commonXor_ = 0;
        hnsw->col_c0_list.resize(0);
        hnsw->allocformatmemory();
        updatebitSteparray();
    }
    if (uses_ansmet_ndp_dual_granularity(earlyExitEnable, allowSample,
                                         dualQueueLowerBoundETEnable, mfnnsEnable)) {
        const bool dual_granularity_requested = bs_num >= 2;
        uint32_t coarse_bit_step = configured_main_bit_step;
        uint32_t fine_bit_step = dual_granularity_requested
            ? std::min(configured_stage1_bit_step, configured_stage2_bit_step)
            : configured_main_bit_step;
        if (fine_bit_step > coarse_bit_step) {
            print("[Config] ANSMET dual-granularity requires fineBitStep<=coarseBitStep. Swap coarse=%u fine=%u.",
                  coarse_bit_step, fine_bit_step);
            std::swap(coarse_bit_step, fine_bit_step);
        }

        const uint32_t coarse_dim_step = normalize_dim_step_from_bit_step(coarse_bit_step);
        const uint32_t fine_dim_step = normalize_dim_step_from_bit_step(fine_bit_step);
        if (configured_main_dim_step != coarse_dim_step) {
            print("[Config] ANSMET coarse dimStep=%u normalized to floor(512/%u)=%u.",
                  configured_main_dim_step, coarse_bit_step, coarse_dim_step);
        }
        if (configured_stage1_bit_step != configured_stage2_bit_step) {
            print("[Config] ANSMET manual fine steps differ: bitStep_trav1=%u bitStep_trav2=%u. Use fineBitStep=%u.",
                  configured_stage1_bit_step, configured_stage2_bit_step, fine_bit_step);
        }

        bitStep_trav1 = coarse_bit_step;
        dimStep_trav1 = coarse_dim_step;
        bitStep_trav2 = fine_bit_step;
        dimStep_trav2 = fine_dim_step;
        const int coarse_fetch_count = dual_granularity_requested
            ? std::max(1, configured_coarse_fetch_count)
            : 1;
        bitStep_array = build_ansmet_dual_granularity_schedule(coarse_bit_step, fine_bit_step, coarse_fetch_count);
        dimbitMode = "dimFirst";
        print("[Config] ANSMET main schedule mode=%s coarseBitStep=%u coarseDimStep=%u coarseFetchCount=%d fineBitStep=%u fineDimStep=%u",
              dual_granularity_requested ? "dual_granularity" : "uniform",
              coarse_bit_step, coarse_dim_step, coarse_fetch_count, fine_bit_step, fine_dim_step);
    }
    dimStep_trav1=floor(64*8/bitStep_trav1);
    dimStep_trav2=floor(64*8/bitStep_trav2);
    // printf("[DEBUG] Creating %d EmbUnits...\n", nEmbUnit);
    // fflush(stdout);
    for (uint32_t i = 0; i < nEmbUnit; i++) {
        // printf("[DEBUG] Creating EmbUnit %d/%d\n", i+1, nEmbUnit);
        // fflush(stdout);
        HNSWEmbUnit* embUnit = dynamic_cast<HNSWEmbUnit*>(create_child_ifce<IFrontEnd>());
        assert(embUnit);
        embUnits.emplace_back(embUnit);
        // printf("[DEBUG] EmbUnit %d created successfully\n", i+1);
        // fflush(stdout);
    }
    // printf("[DEBUG] All EmbUnits created\n");
    // fflush(stdout);

    if (pollingSimulation) {
        for (uint32_t rankId = 0; rankId < nEmbUnit; rankId++) {
            Addr_t pollingAddr = 0xdeadbeef;
            Request pollReq(pollingAddr,
                Request::Type::Read,
                rankId,
                pollCallback);
            pendPollReqs.push(pollReq);
        }
    }
};

void HNSWTraversalUnit::preprocess_vecPacking() {
    std::unordered_map<uint32_t, uint32_t> unusedPackId;
    for (uint32_t packId = 0; packId < nData / nPackingVec; packId++) {
        unusedPackId[packId] = nPackingVec;
    }
    auto rand_between = [](uint32_t min, uint32_t max) {
        return min + rand() % (max - min + 1);
    };
    auto getAvailPackId = [&](uint32_t& prevPackId) -> uint32_t {
        if (!unusedPackId.count(prevPackId)) {
            assert(unusedPackId.size() > 0);
            auto random = std::next(std::begin(unusedPackId), rand_between(0, unusedPackId.size() - 1));
            random->second -= 1;
            uint32_t packId = random->first;
            if (random->second == 0) unusedPackId.erase(random);
            return packId;
        }
        unusedPackId[prevPackId] -= 1;
        if (unusedPackId[prevPackId]== 0) {
            unusedPackId.erase(prevPackId);
        }
        return prevPackId;
    };
    // Step 1
    // printf("Step1\n");
    // fflush(stdout);
    int level0 = 0;
    int leveln = 0;
    for (PointId candId = 0; candId < nData; candId++) {
        uint32_t level = hnsw->element_levels_[candId];
        if (level == 0){
            level0++;
            continue;
        } 
        leveln++;
        hnswlib::linklistsizeint* ll_cur = hnsw->get_linklist_at_level(candId, level);
        uint32_t size = hnsw->getListCount(ll_cur);
        hnswlib::tableint *data = (hnswlib::tableint *) (ll_cur + 1);
        uint32_t prevPackId = rand() % (nData / nPackingVec); // randomize the initial pack id
        for (uint32_t i = 0; i < size; i++) {
            PointId neighbor = data[i];
            if (!vecPackingMap.count(neighbor)) {
                vecPackingMap[neighbor] = getAvailPackId(prevPackId);
            }
        }
    }
    // printf("level0:%d, leveln:%d\n",level0,leveln);
    // fflush(stdout);
    // // Step 2
    // printf("Step2\n");
    // fflush(stdout);
    for (PointId candId = 0; candId < nData; candId++) {
        uint32_t level = hnsw->element_levels_[candId];
        if (vecPackingMap.count(candId)) continue;
        hnswlib::linklistsizeint* ll_cur = hnsw->get_linklist_at_level(candId, level);
        uint32_t size = hnsw->getListCount(ll_cur);
        hnswlib::tableint *data = (hnswlib::tableint *) (ll_cur + 1);
        for (uint32_t i = 0; i < size; i++) {
            PointId neighbor = data[i];
            if (vecPackingMap.count(neighbor)) {
                vecPackingMap[candId] = getAvailPackId(vecPackingMap[neighbor]);
                break;
            }
        }
    }
    // Step 3
    // printf("Step3\n");
    // fflush(stdout);
    for (PointId candId = 0; candId < nData; candId++) {
        if (vecPackingMap.count(candId)) continue;
        uint32_t random = rand() % (nData / nPackingVec);
        vecPackingMap[candId] = getAvailPackId(random);
    }
}

int highestBitPosition(int num) {
    if (num == 0) return -1;
    unsigned int unum = num & INT_MAX; 

    int pos = 0;
    while (unum != 0) {
        pos++;
        unum >>= 1;
    }
    return pos;
}

int compareShiftedBits(int a, int b, int n) {
    unsigned int ua = static_cast<unsigned int>(a);
    unsigned int ub = static_cast<unsigned int>(b);
    ua >>= n;
    ub >>= n;
    unsigned int mask = 0x7FFFFFFF;
    ua &= mask;
    ub &= mask;
    return (ua == ub) ? 1 : 0;
}


int findPattern(const std::vector<int>& nums, double outlierPercentage, int& m, int& n) {
    int size = nums.size();
    // printf("size: %d",size);
    std::vector<int> sortedNums = nums;
    std::sort(sortedNums.begin(), sortedNums.end());

    int outliersToRemove = static_cast<int>(size * outlierPercentage);

    m = 0;
    n = 0;
    int firstNonZeroIndex = 0;
    while (firstNonZeroIndex < size && sortedNums[firstNonZeroIndex] == 0) {
        firstNonZeroIndex++;
    }
    if (firstNonZeroIndex == size) {
        std::cerr << "All elements are zero." << std::endl;
        return 0;  
    }
    //printf("size:%d, firstNonZeroIndex:%d\n",size,firstNonZeroIndex);
    for (int i = std::max(firstNonZeroIndex+1,1); i < size ; ++i) {
        int prefixLength = highestBitPosition(sortedNums[i] ^ sortedNums[i - 1]);
        m = std::max(m, prefixLength);
        if(prefixLength == 7){
            printf("\n nums_i:%d,nums_i-1:%d\n",sortedNums[i],sortedNums[i - 1]);
        }
    }
    int commonXor = sortedNums[outliersToRemove+firstNonZeroIndex+1];

    for (int i = std::max(outliersToRemove+firstNonZeroIndex+1,outliersToRemove+1); i < size - outliersToRemove; ++i) {
        int prefixLength = highestBitPosition(sortedNums[i] ^ sortedNums[i - 1]);
        n = std::max(n, prefixLength);
    }
    return commonXor;
}

float labelEnd(float data_dim, bool dimOl) { 
    auto* ptr = reinterpret_cast<std::uint32_t*>(&data_dim);
    if (dimOl) {
        *ptr |= 1; 
    } else {
        *ptr &= ~std::uint32_t(1);  
    }

    return data_dim;
}

int getLastBit(float num) {
    uint32_t* p = reinterpret_cast<uint32_t*>(&num);
    return *p & 1;
}

void select_random_vectors(const std::vector<float>& all_queries, std::vector<float>& queries_prprcs, size_t nq, size_t nDim) {
    size_t total_vectors = all_queries.size() / nDim;
    std::vector<size_t> indices(total_vectors);
    for (size_t i = 0; i < total_vectors; ++i) {
        indices[i] = i;
    }
    std::random_shuffle(indices.begin(), indices.end());
    indices.resize(nq);

    queries_prprcs.resize(nq * nDim);
    for (size_t i = 0; i < nq; ++i) {
        size_t idx = indices[i];
        for (size_t d = 0; d < nDim; ++d) {
            queries_prprcs[i * nDim + d] = all_queries[idx * nDim + d];
        }
    }
}

void generateVectors(int bitstep_max, int stepsum_max, std::vector<int>& current_vector, std::vector<std::vector<int>>& result, int current_sum = 0, bool limit_num = false, int bitstep_num = 0) {
    if (current_sum > stepsum_max || result.size() >= 2000) {
        return;
    }

    if (!current_vector.empty()) {
        result.push_back(current_vector);
    }

    for (int i = bitstep_max; i >= 1; --i) {
        if (limit_num && current_vector.size() >= bitstep_num) {
            break;
        }
        if (current_sum + i <= stepsum_max) {
            current_vector.push_back(i);
            generateVectors(i, stepsum_max, current_vector, result, current_sum + i, limit_num, bitstep_num);
            current_vector.pop_back();
        }
    }
}
void removeEquivalentVectors(std::vector<std::vector<int>>& vectors) {
    std::vector<std::vector<int>> filtered_vectors;

    for (const auto& vec : vectors) {
        bool is_equivalent = false;

        for (const auto& filtered_vec : filtered_vectors) {
            size_t min_size = std::min(vec.size(), filtered_vec.size());
            bool prefix_equal = std::equal(vec.begin(), vec.begin() + min_size, filtered_vec.begin());

            if (prefix_equal) {
                bool remaining_equal = true;
                if (vec.size() > filtered_vec.size()) {
                    for (size_t j = min_size; j < vec.size(); ++j) {
                        if (vec[j] != vec[min_size - 1]) {
                            remaining_equal = false;
                            break;
                        }
                    }
                }
                else {
                    for (size_t j = min_size; j < filtered_vec.size(); ++j) {
                        if (filtered_vec[j] != filtered_vec[min_size - 1]) {
                            remaining_equal = false;
                            break;
                        }
                    }
                }

                if (remaining_equal) {
                    is_equivalent = true;
                    break;
                }
            }
        }

        if (!is_equivalent) {
            filtered_vectors.push_back(vec);
        }
    }

    vectors = std::move(filtered_vectors);
}
int HNSWTraversalUnit::preprocess(){
    // ⚠️ 临时减小以测试：对于 1024 维，先用小规模验证
    int n = (nDim > 512) ? 20 : 100;  // 大维度时减少候选数
    int nq = (nDim > 512) ? 2 : 5;     // 大维度时减少查询数
    int n2=2;
    float aver=0;
    float sum1=0;
    float sum2=0;//square sum
    float stdev=0;
    int col_a0=1;
    int col_cut0 = 0;
    int col_b0=4;
    const int effectiveDataBitWidth = static_cast<int>(preprocessVectorDataBitWidth);
    const int effectiveFloatExponentBits = (datatype == "isFloat")
        ? static_cast<int>(get_float_exponent_bit_count(preprocessVectorDataBitWidth))
        : 8;
    const int effectiveFloatSignExpBits = (datatype == "isFloat")
        ? static_cast<int>(get_float_signexp_bit_count(preprocessVectorDataBitWidth))
        : 9;
    float x_outlier = 0.1;
    HNSWEmbUnit* embUnit_temp = dynamic_cast<HNSWEmbUnit*>(create_child_ifce<IFrontEnd>());
    assert(embUnit_temp);
    std::vector<Type> all_queries;
    uint32_t nQ;
    read_vec<Type>(query_path, all_queries, nQ, nDim);
    std::vector<float> queries_prprcs;
    select_random_vectors(all_queries, queries_prprcs, nq, nDim);
    //sample n vecs
    std::vector<int> v(hnsw->max_elements_);
    std::iota(v.begin(), v.end(), 0); // Fill with 0, 1, ..., hnsw->max_elements_ - 1
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);
    if (n > hnsw->max_elements_) {
        throw std::runtime_error("n cannot be greater than hnsw->max_elements_ for unique values.");
    }
    v.resize(n);
    std::vector<int> exit_list(n * (n - 1) / 2);
    
    outlier_burden = param<bool>("outlier_burden").desc("outlier_burden").required();
    allowoutlier=param<bool>("allowoutlier").desc("allowoutlier").required();
    if(datatype == "isInt" && allowoutlier){
       std::vector<int> intv_list(n * nDim);
    
        for(int i = 0; i<n; ++i) {
            int candId1 = v[i];
            Type* candData1 = getEmbData(candId1);
            for(int dim = 0; dim < nDim; dim++){
                float f = *(candData1 + dim);
                int intv = std::abs((int)f);                
                intv_list[i * nDim + dim] = intv;
            }
        }
        outlier_percentage = param<float>("outlier_percentage").desc("outlier_percentage").required();
        outlier_percentage = outlier_percentage /(nDim*1.0);
        int m, n;
        hnsw->commonXor_ = findPattern(intv_list, outlier_percentage, m, n);
        bool enable_outlier = 1;
        bool cut_enable = 1;
        col_cut0 = cut_enable? (enable_outlier? (8 - n) : (8 - m)) : 0;
        hnsw->col_cut = col_cut0;
        printf("col_cut0:%d, m:%d, n:%d\n",col_cut0,m,n);
        printf("2\n");
        fflush(stdout);
        hnsw->outlier_list.resize(hnsw->max_elements_);
        for(int outlierid_ingraph = 0; outlierid_ingraph < hnsw->max_elements_; outlierid_ingraph++){
        
            bool VecisOl = 0;
            Type* candData1 = getEmbData(outlierid_ingraph);
            for(int dim = 0; dim < nDim; dim++){
                float f = *(candData1 + dim);
                int intv = std::abs((int)f); 
                if(!compareShiftedBits(hnsw->commonXor_,intv,n) && intv!= 0){
                    VecisOl = 1;
                    hnsw->num_outlier_++;
                    break;
                }
            }
            if(VecisOl){
                hnsw->outlier_list[outlierid_ingraph] = hnsw->num_outlier_;
            }else{
                hnsw->outlier_list[outlierid_ingraph] = 0;
            }       
        }
    // printf("num_outlier:%d, max_elements:%d\n",hnsw->num_outlier_,hnsw->max_elements_);
    // fflush(stdout);
    // printf("col_cut0,outlierlist finished\n");
    // fflush(stdout);
    // hnsw->allocformatmemory();
    // printf("alloc format mem finished\n");
    // fflush(stdout);
    // hnsw->col_c0_list.resize(hnsw->num_outlier_);
    // printf("col_c0 resize finished\n");
    // fflush(stdout);
    // printf("num:%d\n",hnsw->num_outlier_);
    // fflush(stdout);
    
        for(int outlierid_ingraph = 0; outlierid_ingraph < hnsw->max_elements_; outlierid_ingraph++){ //in this loop, we fill up the format memory
            
            Type* candFormat = (Type*)hnsw->getFormatByInternalId(outlierid_ingraph);
            if(candFormat == nullptr){
                throw std::runtime_error("candId error");
            }
            int col_c0 = 0;
            Type* candData1 = getEmbData(outlierid_ingraph);
            if(hnsw->outlier_list[outlierid_ingraph]!=0){
                for(int dim = 0; dim < nDim; dim++){
                    float f = *(candData1 + dim);
                    int intv = std::abs((int)f);    
                    if(compareShiftedBits(hnsw->commonXor_,intv,n) || intv == 0){ 
                        //not outlier on this dim
                        *(candFormat + dim) = labelEnd(*(candData1 + dim),0);
                        int temp = highestBitPosition((hnsw->commonXor_^intv)>>n);
                        col_c0 = std::max(col_c0, temp);
                        *(candFormat + dim) = labelEnd(*(candData1 + dim),1);
                    }
                }   
                hnsw->col_c0_list[hnsw->outlier_list[outlierid_ingraph] - 1] = col_c0;  // outlier_list is 1-indexed
            }
        }
        printf("finish outlier processing\n");
}else if((datatype == "isFloat") && allowoutlier){
    
    std::vector<int> exp_list(n * nDim);
    
    for(int i = 0; i<n; ++i) {
        int candId1 = v[i];
        Type* candData1 = getEmbData(candId1);
        for(int dim = 0; dim < nDim; dim++){
            float f = *(candData1 + dim);
            int exp = static_cast<int>(get_float_stored_exponent_bits(f, preprocessVectorDataBitWidth));
            
            exp_list[i * nDim + dim] = exp;
        }
    }
    outlier_percentage = param<float>("outlier_percentage").desc("outlier_percentage").required();
    int m, n;
    hnsw->commonXor_ = findPattern(exp_list, outlier_percentage, m, n);
    bool enable_outlier = 1;
    bool cut_enable = 1;
    col_cut0 = cut_enable? (enable_outlier? (effectiveFloatExponentBits - n) : (effectiveFloatExponentBits - m)) : 0;
    hnsw->col_cut = col_cut0;
    // printf("col_cut0:%d, m:%d, n:%d\n",col_cut0,m,n);
    hnsw->outlier_list.resize(hnsw->max_elements_);
        for(int outlierid_ingraph = 0; outlierid_ingraph < hnsw->max_elements_; outlierid_ingraph++){
            
        bool VecisOl = 0;
        Type* candData1 = getEmbData(outlierid_ingraph);
        for(int dim = 0; dim < nDim; dim++){
            float f = *(candData1 + dim);
            int exp = static_cast<int>(get_float_stored_exponent_bits(f, preprocessVectorDataBitWidth));
            if(!compareShiftedBits(hnsw->commonXor_,exp,n) && exp!= 0){
                VecisOl = 1;
                hnsw->num_outlier_++;
                    break;
                }
            }
            if(VecisOl){
                hnsw->outlier_list[outlierid_ingraph] = hnsw->num_outlier_; 
            }else{
                hnsw->outlier_list[outlierid_ingraph] = 0;
            }       
        }
        // printf("col_cut0,outlierlist finished\n");
        // fflush(stdout);
        hnsw->allocformatmemory();
        // printf("alloc format mem finished\n");
        // fflush(stdout);
        hnsw->col_c0_list.resize(hnsw->num_outlier_);
        // printf("col_c0 resize finished\n");
        // fflush(stdout);
        // printf("num:%d\n",hnsw->num_outlier_);
        // fflush(stdout);
        for(int outlierid_ingraph = 0; outlierid_ingraph < hnsw->max_elements_; outlierid_ingraph++){ //in this loop, we fill up the format memory
            Type* candFormat = (Type*)hnsw->getFormatByInternalId(outlierid_ingraph);//(Type*)
            if(candFormat == nullptr){
                throw std::runtime_error("candId error");
                printf("fatal");
                fflush(stdout);
            }
            int col_c0 = 0;
            Type* candData1 = getEmbData(outlierid_ingraph);
        if(hnsw->outlier_list[outlierid_ingraph]!=0){
            
            for(int dim = 0; dim < nDim; dim++){
                float f = *(candData1 + dim);
                int exp = static_cast<int>(get_float_stored_exponent_bits(f, preprocessVectorDataBitWidth));
                if(compareShiftedBits(hnsw->commonXor_,exp,n) || exp == 0){ 
                    *(candFormat + dim) = labelEnd(*(candData1 + dim),0);
                }else{ //如果这个dim是outlier，就标记这个尾号为1
                        int temp = highestBitPosition((hnsw->commonXor_^exp)>>n);
                        col_c0 = std::max(col_c0, temp);
                        *(candFormat + dim) = labelEnd(*(candData1 + dim),1);
                    }
                }   
                hnsw->col_c0_list[hnsw->outlier_list[outlierid_ingraph] - 1] = col_c0;  // outlier_list is 1-indexed
            }
        }
    
    // printf("finish outlier processing\n"); 
    // fflush(stdout);
    }else{
        // printf("skip outlier processing\n");
        // fflush(stdout);
        col_cut0 = 0;
        hnsw->num_outlier_ = 0;
        hnsw->outlier_list.resize(hnsw->max_elements_);
        for(int ii = 0; ii < hnsw->max_elements_; ii++){
            hnsw->outlier_list[ii] = 0;
        }
        hnsw->commonXor_ = 0;
        hnsw->col_c0_list.resize(0);
        hnsw->allocformatmemory();
    }
    
    hnsw->col_cut = col_cut0;
    // printf("[DEBUG] Starting early-exit analysis, nDim=%d\n", nDim);
    // fflush(stdout);
    std::vector<int> col_b0_list;
    std::vector<int> dim_exit_list;
    col_a0 = (datatype == "isFloat") ? effectiveFloatExponentBits : 8;
    int id = 0;
    for(int iq = 0; iq < nq; iq++){
        // printf("[DEBUG] Processing query %d/%d\n", iq+1, nq);
        // fflush(stdout);
        std::vector<Type> dist_list(n);
        Type* query_prprcs = &queries_prprcs[iq * nDim];
        for(int ic = 0; ic < n ; ic++){
            Type* candData = getEmbData(v[ic]);
            dist_list[ic] = embUnit_temp->getDistance(candData, query_prprcs, nDim,spacetype);
            
        }
        std::vector<Type> dist_list_copy = dist_list;
        std::sort(dist_list_copy.begin(), dist_list_copy.end());
        size_t pos = static_cast<size_t>(0.1 * n);
        if (pos >= n) {
            pos = n - 1;
        }
        Type up_bnd = dist_list_copy[pos];
        
        int num_above_bound = 0;
        for(int ic = 0; ic < n; ic++){
            if(dist_list[ic]>up_bnd){
                num_above_bound++;
                int col_b0;
                int dim_exit;
                Type* candData = getEmbData(v[ic]);
                // ✅ 将 vector 提到外层，避免每次 dim_exit 循环都重新分配
                std::vector<Type> candPartialData(nDim);
                for(int bit = col_a0; bit < 32; bit++){
    
                    for (size_t dim = 0; dim < nDim; dim++) {
                        if(spacetype == "L2"){
                            candPartialData[dim] = *(candData + dim);
                        }else{
                            candPartialData[dim] = 0;
                        }
                    } 
                    int flag = 0;
                    for(int dim_exit = 0; dim_exit < nDim; dim_exit+=16){
                        for (size_t dim = 0; dim < dim_exit; dim++) {
                            candPartialData[dim] = embUnit_temp->adjustCandPartialData0(query_prprcs, dim, bit, candData, bit+1, (datatype=="isFloat"), spacetype, dataunsigned);
                        }
                        for (size_t dim = dim_exit+1; dim < nDim; dim++) {
                            candPartialData[dim] = embUnit_temp->adjustCandPartialData0(query_prprcs, dim, bit, candData, bit, (datatype=="isFloat"), spacetype, dataunsigned);
                        }
                        Type dist = embUnit_temp->getDistance(query_prprcs, candPartialData.data(), nDim, spacetype);
                        Type fdist = embUnit_temp->getDistance(query_prprcs, candData, nDim,spacetype);
                        if(dist>up_bnd){
                            id++;
                            col_b0_list.push_back(bit);
                            dim_exit_list.push_back(dim_exit);
                            flag = 1;
                            break;
                        }
                    }
                        
                    if(bit >= effectiveDataBitWidth - 1 || flag ==1){
                        
                        break;
                    }
                }
            }
        }
        // printf("[DEBUG] Query %d: %d candidates above bound, total exits found: %d\n", iq+1, num_above_bound, id);
        // fflush(stdout);
    }
    // printf("[DEBUG] Early-exit analysis complete, total exits: %d\n", id);
    // fflush(stdout);
    size_t cbl_size = col_b0_list.size();
    // printf("[DEBUG] Found %zu early-exit points\n", cbl_size);
    // fflush(stdout);
    
    if (cbl_size == 0) {
        printf("[WARNING] No early-exit points found, using default parameters\n");
        updatebitSteparray();
        return 0;
    }
    
    size_t rank_position = static_cast<size_t>(cbl_size/3);
    std::vector<int> sorted_list = col_b0_list;
    std::sort(sorted_list.begin(), sorted_list.end(), std::greater<int>()); // 从大到小排序
    int stepsum_max;
    if(datatype == "isInt" ){
        stepsum_max = 8-col_cut0;
    }else{
        stepsum_max = *std::max_element(col_b0_list.begin(), col_b0_list.end())-col_cut0- 0*(datatype == "isFloat")- 24 * (datatype == "isInt");
    }
    int bitstep_max = stepsum_max;
    std::vector<std::vector<int>> arrayValues_array;
    std::vector<int> current_vector;
    // printf("[DEBUG] Generating vector combinations...\n");
    // fflush(stdout);
    generateVectors(bitstep_max, stepsum_max, current_vector, arrayValues_array,0,limit_bs_num,bs_num);

    removeEquivalentVectors(arrayValues_array);
    // printf("[DEBUG] Generated %zu vector combinations\n", arrayValues_array.size());
    // fflush(stdout);
    float min_fetch_uti = 32 * nDim * 512;
    std::vector<int> best_arrayValues;
    // printf("[DEBUG] Evaluating %zu combinations across %d exit points...\n", arrayValues_array.size(), id);
    // fflush(stdout);
    int combo_idx = 0;
    for (auto it = arrayValues_array.rbegin(); it != arrayValues_array.rend(); ++it) {
        // if (combo_idx % 100 == 0) {
        //     printf("[DEBUG] Processing combination %d/%zu\n", combo_idx, arrayValues_array.size());
        //     fflush(stdout);
        // }
        // combo_idx++;
        std::vector<int> arrayValues_temp = *it;
        std::vector<int> arrayValues = arrayValues_temp;
        // for (int value : arrayValues) {
        //     std::cout << value << " ";
        // }
        // std::cout << std::endl;

        const size_t targetLength = 32;
        size_t currentLength = arrayValues.size();
        int fillValue = arrayValues.back(); 
        arrayValues.resize(targetLength, fillValue);
        
        int fetch_times = 0;
        for(int ii = 0; ii < id; ii++){
            int col_exit = col_b0_list[ii] - 0*(datatype == "isFloat") - 24 * (datatype == "isInt")+1;
            int dim_exit = dim_exit_list[ii];
            int ncurbit = 0;
            int dimend = 0;
            int bitend = hnsw->col_cut;
            int bitstep = arrayValues[0];
            int dimstep = std::floor(512/bitstep);
            bitend = bitstep + hnsw->col_cut;
            int bitend_last = hnsw->col_cut;
            while(1){
                if((bitend > col_exit && bitend_last >= col_exit) ||(bitend_last == col_exit && dimend >= dim_exit)){
                    break;//exit
                }else{
                    fetch_times += 1;
                    dimend += dimstep;
                    if(dimend>=nDim){
                        dimend = 0;
                        ncurbit++;
                        bitstep = arrayValues[ncurbit];
                        dimstep = std::floor(512/bitstep);
                        bitend_last = bitend;
                        bitend += bitstep;
                    }
                }

            }
        
        }
        float uti_rate = 0;
        int lgth = 0;
        for (const auto& value : arrayValues_temp) {
            int dimstep = std::floor(512/value);
            uti_rate += std::ceil(nDim*1.0/dimstep) * dimstep / nDim * value;
            lgth += value;
        }
        uti_rate = uti_rate*1.0 / lgth;
        int bitmax = 0;
        for (const auto& value : arrayValues) {
            bitmax += value;
            if(bitmax >= effectiveFloatExponentBits){
                break;
            }
        }
        
        float fetch_uti = fetch_times * uti_rate;
        if (fetch_uti < min_fetch_uti) {
            min_fetch_uti = fetch_uti;
            best_arrayValues.resize(arrayValues_temp.size());
            best_arrayValues = arrayValues_temp;
        }
    }
    // std::cout << "best_arrayValues"<< std::endl;
    // for (int value : best_arrayValues) {
    //         printf("%d ",value);
    //     }
    // printf("\n");
    // fflush(stdout);
    if (best_arrayValues.size() > kAnsmetBitStepArrayLength) {
        best_arrayValues.resize(kAnsmetBitStepArrayLength);
    } else if (best_arrayValues.size() < kAnsmetBitStepArrayLength) {
        if (!best_arrayValues.empty()) {
            int lastValue = best_arrayValues.back();  
            best_arrayValues.resize(kAnsmetBitStepArrayLength, lastValue);  
        } else {
            best_arrayValues.resize(kAnsmetBitStepArrayLength, 0);
        }
    }
    bitStep_array.resize(kAnsmetBitStepArrayLength);
    for (size_t i = 0; i < best_arrayValues.size(); ++i) {
        bitStep_array[i] = best_arrayValues[i];
    }

    int num = (datatype == "isFloat") ? effectiveDataBitWidth : 32;
    col_a0 = (datatype == "isFloat") ? effectiveFloatSignExpBits : 9;
    col_b0 = (datatype == "isFloat") ? std::min(effectiveDataBitWidth, effectiveFloatSignExpBits + 1) : 10;
    int col_b=col_b0;
    int col_a=col_a0;
    int base=(datatype == "isInt")*0+(datatype == "isFloat")*effectiveFloatExponentBits;//
    bool flag=(datatype == "isInt")*1+(datatype == "isFloat")*0;
    if ((col_b0 - col_a0) >= col_a0 && flag) {
        col_a = 0;
    }
    int n1 = std::min(num, col_a + 1) - std::max(base + 1, col_a - 1) + 1;
    int total_times_2[num] = { 0 };
    int actual_dim[num] = { 0 };
    int times[num] = { 0 };
    double final1[n1] = {0.0};
    int ii=0;
    for (int x = std::max(base + 1, col_a - 1); x <= std::min(32, col_a + 1); ++x) {
        ii++;
        int dimStep = round(512.0 / x);

        times[ii - 1] = ceil(static_cast<double>(nDim) / dimStep);
        actual_dim[ii - 1] = times[ii - 1] * dimStep;
        total_times_2[ii - 1] = times[ii - 1] * ceil(static_cast<double>(col_a) / x);
        final1[ii - 1] = total_times_2[ii - 1] * std::sqrt(static_cast<double>(actual_dim[ii - 1]) / nDim);
        
    }
    double min_value = final1[0];
    bitStep_trav1=0;
    for(int i=n1;i>0;i--){
        fflush(stdout);
        if(final1[i-1]<=min_value){
            min_value=final1[i-1];
            bitStep_trav1=i-1+std::max(base + 1, col_a - 1);
        }
    }
    double final[num] = {0.0};
    bitStep_trav2 = 0;
    std::vector<int> bitStep2_list(n * (n - 1) / 2);
    std::vector<int> col_b_list(n * (n - 1) / 2);

    for(int jj = 0; jj < col_b_list.size(); jj++){
        if(datatype == "isInt"){
            col_b_list[jj] = exit_list[jj] + 1 - 24;
        }else{ 
            col_b_list[jj] = exit_list[jj] + 1;
        }
    }
    if ((col_b0 - col_a0) >= col_a0 && flag) {
        for(size_t ii = 0; ii < bitStep2_list.size();ii++){
            int col_b = std::max(col_a0 , col_b_list[ii]);
        for(int x=num;x>0;x--){
            
                int dimStep = std::round(512.0 / x);
                times[x - 1] = std::ceil(static_cast<double>(nDim) / dimStep);
                actual_dim[x - 1] = times[x - 1] * dimStep;
                total_times_2[x - 1] = times[x - 1] * std::ceil(static_cast<double>(col_b) / x);

                final[x - 1] = total_times_2[x - 1] * std::sqrt(static_cast<double>(actual_dim[x - 1]) / nDim);
            }
            double min_value = final[0];
            bitStep2_list[ii] = 0;
            for(int i=num;i>0;i--){
                if(final[i-1]<=min_value){
                    min_value=final[i-1];
                    bitStep2_list[ii] = i;
                }
            }
        }
        int index = std::floor(bitStep2_list.size() / 3);
        std::sort(exit_list.begin(), exit_list.end());
        bitStep_trav2 = bitStep2_list[index];
        bitStep_trav1 = bitStep_trav2;
    } else {
        for(size_t ii = 0; ii < bitStep2_list.size();ii++){
            int col_b = std::max(col_a0 , col_b_list[ii]);
        for(int x = num ; x > 0 ; x-- ){
                int dimStep = std::round(512.0 / x);
                times[x - 1] = std::ceil(static_cast<double>(nDim) / dimStep);
                actual_dim[x - 1] = times[x - 1] * dimStep;
                int temp=1;
                if(col_b - bitStep_trav1 >= 1){
                    temp=col_b-bitStep_trav1;
                }
                total_times_2[x - 1] = times[x - 1] * std::ceil(static_cast<double>(temp) / x);

                final[x - 1] = total_times_2[x - 1] * std::sqrt(static_cast<double>(actual_dim[x - 1]) / nDim);
            }

            double min_value = final[0];
            bitStep2_list[ii] = 0;
            for(int i=num;i>0;i--){
                if(final[i-1]<=min_value){
                    min_value=final[i-1];
                    bitStep2_list[ii] = i;
                }
            }
        }
        int index = std::floor(bitStep2_list.size() / 3);
        std::sort(exit_list.begin(), exit_list.end());
        bitStep_trav2 = bitStep2_list[index];
    }
    dimbitMode="dimFirst";
    if(bitStep_trav1 > static_cast<uint32_t>(num) || bitStep_trav2 > static_cast<uint32_t>(num)){ 
        dimbitMode="bitFirst";
    }

    // printf("[DEBUG] Preprocessing complete!\n");
    // fflush(stdout);
    return 0;
};

void HNSWTraversalUnit::finalize() {
    print("--- Final stats ---");
    s_avg_total_latency = s_num_query == 0 ? 0.0 : 1.0 * s_total_latency / s_num_query;
    s_recall_rate = (gt.size() > 0 && s_num_recall_1 > 0) ? (1.0 * s_num_recall / s_num_recall_1) : 0.0;
    if (!s_same_rank_run_lengths.empty()) {
        const double run_sum = std::accumulate(s_same_rank_run_lengths.begin(), s_same_rank_run_lengths.end(), 0.0);
        s_same_rank_run_length_mean = run_sum / static_cast<double>(s_same_rank_run_lengths.size());
        s_same_rank_run_length_p95 = compute_p95_metric(s_same_rank_run_lengths);
    }

    using CountKeyPair = std::pair<uint32_t, uint64_t>;
    auto comp = [](const CountKeyPair& a, const CountKeyPair& b) {
        return a.second < b.second;
    };
    std::priority_queue<CountKeyPair, std::vector<CountKeyPair>, decltype(comp)> minHeap(comp);
    for (const auto& pair : prof_candAcc) {
        minHeap.push(pair);
    }
    uint32_t nTopCandAcc = std::min(10UL, minHeap.size());
    print("Top %d vector accesses:\n", nTopCandAcc);
    for (uint32_t i = 0; i < nTopCandAcc; i++) {
        auto pair = minHeap.top();
        minHeap.pop();
        print("candId %d rankId %d (first vertical group) acc %ld",
              pair.first,
              mapDisReq(pair.first, 0, 0, vDimSize, false),
              pair.second);
    }
    print("\n");

    // Latency breakdown
    print("Total queries: %ld disreqs: %ld pdisreqs: %ld", s_num_query, s_total_disreq, s_total_pdisreq);
    print("Dual-queue lower-bound ET prunes: %ld", s_dual_queue_lb_et);
    print("Same-rank run length: mean %.4f p95 %ld count %zu",
          s_same_rank_run_length_mean,
          s_same_rank_run_length_p95,
          s_same_rank_run_lengths.size());
    print("Query latency breakdown:")
    print("\tTotal average latency: %.2f", s_avg_total_latency);
    print("\tAverage index latency: %.2f", 1. * s_bd_index_cycles / s_num_query);
    print("\tAverage disOffload latency: %.2f", 1. * s_bd_disOffload_cycles / s_num_query);
    print("\tAverage disComp latency: %.2f", 1. * s_bd_disComp_cycles / s_num_query);
    print("\tAverage disComp Accept latency: %.2f", 1. * s_bd_disCompAcc_cycles / s_num_query);
    print("\tAverage disComp Reject latency: %.2f", 1. * s_bd_disCompRej_cycles / s_num_query);
    print("\tAverage disGather latency: %.2f", 1. * s_bd_disGather_cycles / s_num_query);
    print("\tAverage polling latency: %.2f", 1. * s_bd_polling_cycles / s_num_query);
    // print("\tPercentage disComp Acc: %.2f", 1. * s_bd_disCompAcc_cycles / s_bd_disComp_cycles);
    // print("\tPercentage disComp Rej: %.2f", 1. * s_bd_disCompRej_cycles / s_bd_disComp_cycles);
    // print("\tPercentage disGather: %.2f", 1. * s_bd_disGather_cycles / s_bd_disComp_cycles);
    s_mem_cycle = m_memory_system->get_clk();
    s_mem_read_req = m_memory_system->get_num_read();
    s_mem_write_req = m_memory_system->get_num_write();
    print("Compute/Memory overlap profile:");
    print("	DisReq memory-service cycles (sum): %ld", s_disreq_mem_service_cycles);
    print("	FMAC raw cycles (sum): %ld", s_disreq_raw_compute_cycles);
    print("	FMAC hidden-by-memory cycles (sum): %ld", s_disreq_hidden_compute_cycles);
    print("	FMAC queue cycles (sum): %ld", s_disreq_compute_queue_cycles);
    print("	FMAC exposed cycles (critical-path sum): %ld", s_disreq_exposed_compute_cycles);
    if (s_disreq_raw_compute_cycles > 0) {
        print("	FMAC critical exposed/raw ratio: %.4f", 1.0 * s_disreq_exposed_compute_cycles / s_disreq_raw_compute_cycles);
    } else {
        print("	FMAC critical exposed/raw ratio: 0.0000");
    }
    // Polling
    print("Polling req: %ld", s_num_polling);
    // recall rate
    if (gt.size() > 0) {
        print("Recall rate: %.2f", s_recall_rate);
    }
    print("------CPU-side caching------");
    print("\tTop-layer vector data access: %ld / %ld = %.4f", s_top_layer_access, s_total_pdisreq, 1. * s_top_layer_access / s_total_pdisreq);
    print("\tTop-layer vector data size: %ld * 64B", top_layer_addr.size());
    print("\tIf using a general cache: hits %ld total %ld", s_cache_hit, s_total_pdisreq);
    print("------NDP-side load balancing------");
    print("\tTop-layer real vector size %ld * 64B access %ld", unlimited_top_layer_addr.size(), s_unlimited_top_layer_access);
    uint64_t sum_pdis = 0, max_pdis = 0, min_pdis = -1UL;
    uint64_t sum_loadbalance_pdis = 0, max_loadbalance_pdis = 0, min_loadbalance_pdis = -1UL;
    for (uint32_t i = 0; i < nEmbUnit; i++) {
        print("\t Rank %d, PDisReq %ld, top_layer_access %ld, after loadbalance %ld", i, s_num_total_pdisreq[i], s_top_layer_access_per_embunit[i], s_num_total_pdisreq[i] - s_top_layer_access_per_embunit[i] + s_unlimited_top_layer_access / nEmbUnit);
        sum_pdis += s_num_total_pdisreq[i];
        sum_loadbalance_pdis += s_num_total_pdisreq[i] - s_top_layer_access_per_embunit[i] + s_unlimited_top_layer_access / nEmbUnit;
        max_pdis = std::max(max_pdis, s_num_total_pdisreq[i]);
        min_pdis = std::min(min_pdis, s_num_total_pdisreq[i]);
        max_loadbalance_pdis = std::max(max_loadbalance_pdis, s_num_total_pdisreq[i] - s_top_layer_access_per_embunit[i] + s_unlimited_top_layer_access / nEmbUnit);
        min_loadbalance_pdis = std::min(min_loadbalance_pdis, s_num_total_pdisreq[i] - s_top_layer_access_per_embunit[i] + s_unlimited_top_layer_access / nEmbUnit);
    }
    uint64_t avg_pdis = sum_pdis / nEmbUnit;
    uint64_t avg_loadbalance_pdis = sum_loadbalance_pdis / nEmbUnit;
    print("\t before loadbalance: max %ld (%.4f) min %ld (%.4f) avg %ld", max_pdis, 1. * max_pdis / avg_pdis, min_pdis, 1. * min_pdis / avg_pdis, avg_pdis);
    print("\t after loadbalance: max %ld (%.4f) min %ld (%.4f) avg %ld", max_loadbalance_pdis, 1. * max_loadbalance_pdis / avg_loadbalance_pdis, min_loadbalance_pdis, 1. * min_loadbalance_pdis / avg_loadbalance_pdis, avg_loadbalance_pdis);
    
    // Per-rank vector LRU cache statistics
    print("------Per-Rank Vector LRU Cache------");
    uint64_t total_cache_hit = 0, total_cache_miss = 0, total_cache_access = 0;
    for (uint32_t i = 0; i < nEmbUnit; i++) {
        uint64_t hit = embUnits[i]->get_vector_cache_hit();
        uint64_t miss = embUnits[i]->get_vector_cache_miss();
        uint64_t access = embUnits[i]->get_vector_cache_access();
        total_cache_hit += hit;
        total_cache_miss += miss;
        total_cache_access += access;
        if (access > 0) {
            print("\t Rank %d: hit %ld, miss %ld, access %ld, hit_rate %.4f", i, hit, miss, access, 1. * hit / access);
        }
    }
    if (total_cache_access > 0) {
        print("\t Total: hit %ld, miss %ld, access %ld, hit_rate %.4f", total_cache_hit, total_cache_miss, total_cache_access, 1. * total_cache_hit / total_cache_access);
    } else {
        print("\t Vector cache disabled or no accesses");
    }

    print("------Per-Rank Phase-1 SignExp Cache------");
    uint64_t total_phase1_cache_hit = 0, total_phase1_cache_miss = 0, total_phase1_cache_access = 0;
    for (uint32_t i = 0; i < nEmbUnit; i++) {
        uint64_t hit = embUnits[i]->get_phase1_signexp_cache_hit();
        uint64_t miss = embUnits[i]->get_phase1_signexp_cache_miss();
        uint64_t access = embUnits[i]->get_phase1_signexp_cache_access();
        total_phase1_cache_hit += hit;
        total_phase1_cache_miss += miss;
        total_phase1_cache_access += access;
        if (access > 0) {
            print("\t Rank %d: hit %ld, miss %ld, access %ld, hit_rate %.4f", i, hit, miss, access, 1. * hit / access);
        }
    }
    if (total_phase1_cache_access > 0) {
        print("\t Total: hit %ld, miss %ld, access %ld, hit_rate %.4f",
              total_phase1_cache_hit,
              total_phase1_cache_miss,
              total_phase1_cache_access,
              1. * total_phase1_cache_hit / total_phase1_cache_access);
    } else {
        print("\t Phase-1 signexp cache disabled or no accesses");
    }

    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    m_impl->print_stats(emitter);
    if (auto* multi_dram = dynamic_cast<MultiGenericDRAMSystem*>(m_memory_system)) {
        multi_dram->emit_rank_dashboard(emitter);
    }
    emitter << YAML::EndMap;
    emitter << YAML::EndMap;
    // std::cout << emitter.c_str() << std::endl;
    std::ofstream file(statFile);
    if (file.is_open()) {
        file << emitter.c_str();
        file.close();
    } else {
        m_logger->warn("Failed to write stats to file {}", statFile.c_str());
    }
};

void HNSWTraversalUnit::tick() {
    if (is_finished()) return;
    m_clk += 1;
    // send pending search query
    if (queries.size() < nParallelQuery) {
        if (!pendQueries.empty()) {
            Query* query = pendQueries.front();
            pendQueries.pop();
            uint64_t annsId = query->annsId;
            if (query->type == "search") {
                query->debugIssueTraceActive = shouldDebugIssueTraceForQuery(annsId);
                query->debugDuplicateAcceptActive = shouldDebugDuplicateAcceptForQuery(annsId);
                queries[annsId] = query;
                m_logger->info("[tick] find ready search query {}", annsId);
                searchLayer(annsId);
            } else {
                assert(query->type == "construct");
                queries[annsId] = query;
                m_logger->info("[tick] find ready construct query {}", annsId);
                addPoint(annsId);
            }
        }
    }
    // tick embunits
    for (uint32_t i = 0; i < nEmbUnit; i++) {
        embUnits[i]->tick();
    }
    // send pending DisReqs
    for (uint32_t i = 0; i < nEmbUnit; i++) {
        if (pendDisReqs[i].empty()) continue;
        DisReq dreq = pendDisReqs[i].front();
        if (embUnits[i]->sendDisReq(dreq)) {
            pendDisReqs[i].pop();
        }
    }
    // send pending TravReqs
    trySendTravReq(pendTravNodeReqs, /* isEdgeReq */ false);
    trySendTravReq(pendTravEdgeReqs, /* isEdgeReq */ true);
    // polling
    if (!pendPollReqs.empty()) {
        Request pollReq = pendPollReqs.front();
        pollReq.issue_time = m_memory_system->get_clk();
        if (m_memory_system->send(pollReq)) {
            pendPollReqs.pop();
        }
    }
    // result probing
    if (adaptiveResultProbe) {
        for (uint32_t i = 0; i < nEmbUnit; i++) {
            uint32_t nFinish = embUnits[i]->getFinishDisReqs();
            double prob = 1;
            for (uint32_t i = 0; i < nFinish; i++)
                prob = prob * 0.5; // a pre-trained prob value from preprocessing. Lower means more accurate
            uint32_t random = rand() % 100;
            if (random < prob * 100) continue;
            s_num_result_probe += 1;
            uint32_t nFinishDisReq = embUnits[i]->sendResultProbe();
            m_logger->info("[travUnit] cycle %ld receive %d disreqs from embUnit %d", m_clk, nFinishDisReq, i);
        }
    } else {
        if (m_clk % resProbeEpoch == 0) {
            s_num_result_probe += 1;
            for (uint32_t i = 0; i < nEmbUnit; i++) {
                uint32_t nFinishDisReq = embUnits[i]->sendResultProbe();
                m_logger->info("[travUnit] cycle %ld receive %d disreqs from embUnit %d", m_clk, nFinishDisReq, i);
            }
        }
    }
};

} // namespace Ramulator
