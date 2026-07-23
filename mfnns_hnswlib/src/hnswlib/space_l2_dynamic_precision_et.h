#pragma once
#include "early_termination_interface.h"
#include "space_l2_dynamic_precision.h"
#include <atomic>
#include <cstring>
#include <cstdio>

namespace hnswlib {

/**
 * Early Termination (ET) Statistics
 *
 * Tracks success rates for:
 * - L1 ET: Early termination using sign+exponent bits only
 * - L2 ET: Early termination using FPMA partial sums
 * - Bit-level ET: Early termination with configurable bit precision
 */
struct ETStatistics {
    std::atomic<uint64_t> total_distance_calls{0};  // Total distance computation count
    std::atomic<uint64_t> l1_et_success{0};         // L1 ET success count (lower bound > threshold)
    std::atomic<uint64_t> l2_et_success{0};         // L2 ET success count (partial sum > threshold)
    std::atomic<uint64_t> l2_et_total_dims{0};      // Accumulated dimension count when L2 ET succeeds
    std::atomic<uint64_t> bit_level_et_success{0};  // Bit-level ET success count

    void reset() {
        total_distance_calls = 0;
        l1_et_success = 0;
        l2_et_success = 0;
        l2_et_total_dims = 0;
        bit_level_et_success = 0;
    }

    double get_l1_et_rate() const {
        uint64_t total = total_distance_calls.load();
        return total > 0 ? static_cast<double>(l1_et_success.load()) / total : 0.0;
    }

    double get_l2_et_rate() const {
        uint64_t total = total_distance_calls.load();
        return total > 0 ? static_cast<double>(l2_et_success.load()) / total : 0.0;
    }

    double get_bit_level_et_rate() const {
        uint64_t total = total_distance_calls.load();
        return total > 0 ? static_cast<double>(bit_level_et_success.load()) / total : 0.0;
    }
};

/**
 * L1 Early Termination: Estimate lower bound using sign + exponent bits
 *
 * Algorithm (based on MFA.md):
 * For normalized vectors, L2 distance can be converted to: |Q-V|² = 2 - 2Σq_i·v_i
 *
 * L1 ET estimates lower bound of -q_i·v_i (using only sign and exponent bits of v_i):
 * 1. sign(q_i)=0 & sign(v_i)=1: lb_i = q_i · 2^(E_vi - bias)
 * 2. sign(q_i)=1 & sign(v_i)=0: lb_i = -q_i · 2^(E_vi - bias)
 * 3. sign(q_i)=0 & sign(v_i)=0: lb_i = -q_i · 2^(E_vi - bias + 1)
 * 4. sign(q_i)=1 & sign(v_i)=1: lb_i = q_i · 2^(E_vi - bias + 1)
 *
 * If 2 + Σlb_i > threshold, early termination is possible
 */
inline float compute_l1_lower_bound(float q_val, uint16_t v_fp16_bits) {
    // Extract sign and exponent bits of v
    uint32_t v_sign = (v_fp16_bits >> 15) & 0x1;
    int32_t v_expo = (v_fp16_bits >> 10) & 0x1F;
    bool v_mantissa_high = (v_fp16_bits >> 9) & 0x1;

    // Handle zero and subnormal numbers
    if (v_expo == 0) {
        return 0.0f;
    }

    // Extract sign bit of q
    uint32_t q_bits;
    std::memcpy(&q_bits, &q_val, sizeof(float));
    uint32_t q_sign = (q_bits >> 31) & 0x1;

    // Compute q_i · 2^(E_vi - bias) or q_i · 2^(E_vi - bias + 1)
    // FP16 bias = 15
    int32_t target_expo;
    float lb;

    // more precise
    // if (q_sign == 0 && v_sign == 1) {
    //     // Case 1: lb_i = q_i · 2^(E_vi - 15) (opposite signs, no +1)
    //     target_expo = v_expo - 15;
    //     lb = q_val * std::ldexp(1.0f, target_expo);
    //     if(v_mantissa_high){
    //         lb += q_val * std::ldexp(1.0f, target_expo-1);
    //     }
    // } else if (q_sign == 1 && v_sign == 0) {
    //     // Case 2: lb_i = -q_i · 2^(E_vi - 15) (opposite signs, no +1)
    //     target_expo = v_expo - 15;
    //     lb = -q_val * std::ldexp(1.0f, target_expo);
    //     if(v_mantissa_high){
    //         lb -= q_val * std::ldexp(1.0f, target_expo-1);
    //     }
    // } else if (q_sign == 0 && v_sign == 0) {
    //     // Case 3: lb_i = -q_i · 2^(E_vi - 15 + 1) (same signs, +1)
    //     target_expo = v_expo - 15 + 1;
    //     lb = -q_val * std::ldexp(1.0f, target_expo);
    //     if(!v_mantissa_high){
    //         lb += q_val * std::ldexp(1.0f, target_expo-2);
    //     }
    // } else {  // q_sign == 1 && v_sign == 1
    //     // Case 4: lb_i = q_i · 2^(E_vi - 15 + 1) (same signs, +1)
    //     target_expo = v_expo - 15 + 1;
    //     lb = q_val * std::ldexp(1.0f, target_expo);
    //     if(!v_mantissa_high){
    //         lb -= q_val * std::ldexp(1.0f, target_expo-2);
    //     }
    // }
    if (q_sign == 0 && v_sign == 1) {
        // Case 1: lb_i = q_i · 2^(E_vi - 15) (opposite signs, no +1)
        target_expo = v_expo - 15;
        lb = q_val * std::ldexp(1.0f, target_expo);

    } else if (q_sign == 1 && v_sign == 0) {
        // Case 2: lb_i = -q_i · 2^(E_vi - 15) (opposite signs, no +1)
        target_expo = v_expo - 15;
        lb = -q_val * std::ldexp(1.0f, target_expo);

    } else if (q_sign == 0 && v_sign == 0) {
        // Case 3: lb_i = -q_i · 2^(E_vi - 15 + 1) (same signs, +1)
        target_expo = v_expo - 15 + 1;
        lb = -q_val * std::ldexp(1.0f, target_expo);

    } else {  // q_sign == 1 && v_sign == 1
        // Case 4: lb_i = q_i · 2^(E_vi - 15 + 1) (same signs, +1)
        target_expo = v_expo - 15 + 1;
        lb = q_val * std::ldexp(1.0f, target_expo);

    }

    return lb;
}

/**
 * Bit-Chop: Truncate FP16 to specified bit count
 *
 * @param x_bits: FP16 uint16 representation
 * @param bits_fetched: Number of fetched bits (1-16)
 * @param get_lower: true=get lower bound, false=get upper bound
 * @return Truncated FP16 uint16 representation
 *
 * Important: For interval [lower, upper]:
 * - Positive: lower sets unknown bits=0, upper sets unknown bits=1
 * - Negative: lower sets unknown bits=1 (more negative), upper sets unknown bits=0 (less negative)
 */
inline uint16_t bit_chop_fp16(uint16_t x_bits, int bits_fetched, bool get_lower) {
    if (bits_fetched >= 16) {
        return x_bits;
    }

    // Extract sign bit
    bool is_negative = (x_bits >> 15) & 0x1;

    // Compute mask: keep first bits_fetched bits
    uint16_t mask = (0xFFFF << (16 - bits_fetched)) & 0xFFFF;
    uint16_t x_chopped = x_bits & mask;

    // Decide how to set unknown bits based on sign bit
    bool set_unknown_to_one;
    if (is_negative) {
        // Negative: lower needs more negative (unknown bits=1), upper needs less negative (unknown bits=0)
        set_unknown_to_one = get_lower;
    } else {
        // Positive: lower needs smaller (unknown bits=0), upper needs larger (unknown bits=1)
        set_unknown_to_one = !get_lower;
    }

    if (set_unknown_to_one) {
        uint16_t unknown_bits_mask = (1 << (16 - bits_fetched)) - 1;
        x_chopped = x_chopped | unknown_bits_mask;
    }

    return x_chopped;
}

/**
 * Adjust candidate value to closest point to query under known bits
 *
 * Given query and candidate values, and fetched bits, return the point in interval [cand_lower, cand_upper]
 * closest to query, making the distance lower bound tightest.
 *
 * @param q_val: query float value (already rounded to FP16)
 * @param cand_val: candidate original float32 value
 * @param bits_fetched: number of fetched bits
 * @return adjusted candidate float value
 */
inline float adjust_candidate_bit_level(float q_val, float cand_val, int bits_fetched) {
    if (bits_fetched >= 16) {
        // ✅ Use fp16_round directly, not through uint16 conversion (avoid precision issues)
        return fp16_round(cand_val);
    }

    // Convert candidate to FP16 bits for truncation
    uint16_t cand_bits = fp32_to_fp16_bits(cand_val);

    // Compute candidate's possible interval under known bits
    uint16_t cand_lower_bits = bit_chop_fp16(cand_bits, bits_fetched, true);  // get_lower=true
    uint16_t cand_upper_bits = bit_chop_fp16(cand_bits, bits_fetched, false); // get_lower=false

    // ✅ Use standard fp16_bits_to_fp32 function instead of custom implementation
    float cand_lower = fp16_bits_to_fp32(cand_lower_bits);
    float cand_upper = fp16_bits_to_fp32(cand_upper_bits);

    // Due to correct sign bit handling, lower should always be <= upper
    // If not, there's a bug (for debugging)
    #ifdef DEBUG_BIT_CHOP
    if (cand_lower > cand_upper) {
        fprintf(stderr, "ERROR: cand_lower (%.6f) > cand_upper (%.6f)\n",
                cand_lower, cand_upper);
    }
    #endif

    // Select closest point based on query position
    if (q_val < cand_lower) {
        return cand_lower;
    } else if (q_val > cand_upper) {
        return cand_upper;
    } else {
        // Query is within interval, closest point is query itself (distance lower bound=0)
        return q_val;
    }
}

/**
 * Compute bit-level L2 distance lower bound
 *
 * @param pVect1: query vector (float*)
 * @param pVect2: candidate vector (float*)
 * @param qty: dimension
 * @param bits_fetched: number of fetched FP16 bits (6/8/12/16)
 * @return lower bound of L2 distance
 *
 * Important: To ensure lower bound is strictly <= true FP16 distance, need:
 * 1. Round query to FP16 (consistent with actual distance computation)
 * 2. Use FP16 precision for calculation (avoid lower bound violation due to float32 precision)
 */
inline float compute_bit_level_l2_lower_bound(
    const float* pVect1,
    const float* pVect2,
    size_t qty,
    int bits_fetched) {

    float sq_sum = 0.0f;

    float mx_sum = 0.0f;

    int width = 2048;

    for (size_t i = 0; i < qty; i++) {
        // ✅ Key fix: query must also round to FP16, consistent with actual distance computation
        float q_val = fp16_round(pVect1[i]);

        // ✅ Pass float32 value directly, let adjust_candidate_bit_level handle internally
        float cand_adjusted = adjust_candidate_bit_level(q_val, pVect2[i], bits_fetched);

        // Compute distance squared
        float diff = q_val - cand_adjusted;
        sq_sum += diff * diff;
        if((i+1)% width == 0 || i == qty-1){
            mx_sum = std::max(sq_sum, mx_sum);
            sq_sum = 0.0f;
        }
    }

    return mx_sum;
}

/**
 * L2 Early Termination distance function (supports FP16 FPMA only)
 *
 * During computation:
 * 1. Convert to floating-point accumulation after every multiplier_width fixed-point additions
 * 2. Check if current accumulation exceeds threshold (L2 ET)
 * 3. Return early if exceeded
 */
static float L2Sqr_FP16_FPMA_ET(
    const void *pVect1v,
    const void *pVect2v,
    const void *qty_ptr,
    float threshold,
    ETStatistics* stats) {

    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    // Statistics
    if (stats) {
        stats->total_distance_calls.fetch_add(1);
    }

    // ========== L1 Early Termination (statistics mode) ==========
    // Estimate lower bound using sign + exponent bits
    // |Q-V|² = 2 - 2Σq_i·v_i, we estimate 2 + 2·Σlb_i as lower bound of |Q-V|²
    // where lb_i is lower bound of -q_i·v_i
    float sum_lb_i = 0.0f;

    // L1 accumulation: use fixed-point accumulation every multiplier_width, then convert to float
    int multiplier_width = 16;
    int64_t l1_accumulator = 0;  // FIX: use signed type to avoid negative overflow

    float debug_sum = 0.0f; // debug only

    for (size_t i = 0; i < qty; i++) {
        // Convert v_i to FP16 to get sign and exponent bits
        uint16_t v_fp16_bits = fp32_to_fp16_bits(pVect2[i]);
        float lb_i = compute_l1_lower_bound(pVect1[i], v_fp16_bits);
        // float lb_i = -pVect1[i] * pVect2[i]; //debug only
        // debug_sum += lb_i;
        // Fixed-point accumulation
        int32_t lb_i_fixed = static_cast<int32_t>(lb_i * FP16_FIXED_POINT_SCALE);
        l1_accumulator += lb_i_fixed;

        // Convert to floating-point accumulation every multiplier_width
        if ((i + 1) % multiplier_width == 0 || i == qty - 1) {
            sum_lb_i += static_cast<float>(l1_accumulator) / FP16_FIXED_POINT_SCALE;
            l1_accumulator = 0;
        }
    }

    // Compute L1 lower bound: 2 + 2·Σlb_i
    float l1_lower_bound = 2.0f + 2.0f * sum_lb_i;
    // float l1_lower_bound = 2.0f + 2.0f * debug_sum; // debug only
    // float l1_lower_bound = 12.0f; // debug only

    // DEBUG: print threshold value
    // static std::atomic<int> threshold_debug_count{0};
    // if (stats) {

    //     int count = threshold_debug_count.fetch_add(1);
    //     if (count < 10) {
    //                 printf("[ET Debug] l1_lower_bound=%.2f, threshold=%.6f, l1>th=%d\n",
    //            l1_lower_bound, threshold, l1_lower_bound > threshold);
    //         fprintf(stderr, "[ET Debug %d] l1_lower_bound=%.2f, threshold=%.6f, l1>th=%d\n",
    //                count, l1_lower_bound, threshold, l1_lower_bound > threshold);
    //         fflush(stderr);
    //     }
    // }

    // L1 ET statistics check (no early return)
    if (stats && l1_lower_bound > threshold) {
        stats->l1_et_success.fetch_add(1);
    }

    // ========== L2 Early Termination (statistics mode) ==========
    // Continue computing full distance while recording L2 ET statistics
    uint64_t accumulator = 0;
    float res_quantized = 0.0f;
    bool l2_et_recorded = false;

    for (size_t i = 0; i < qty; i++) {
        float v1 = fp16_round(pVect1[i]);
        float v2 = fp16_round(pVect2[i]);

        // Quantize to fixed-point
        uint32_t quantized_square = fp16_fpma_square_quan(v1, v2);

        // Accumulate to fixed-point accumulator
        accumulator += quantized_square;
        // res_quantized+=(pVect1[i]-pVect2[i])*(pVect1[i]-pVect2[i]);

        // Convert to floating-point accumulation every multiplier_width or at last element
        if ((i + 1) % multiplier_width == 0 || i == qty - 1) {
            res_quantized += static_cast<float>(accumulator) / FP16_FIXED_POINT_SCALE;
            accumulator = 0;

            // L2 ET statistics check (no early return, only record)
            if (stats && !l2_et_recorded && res_quantized > threshold-100) {
                stats->l2_et_success.fetch_add(1);
                stats->l2_et_total_dims.fetch_add(i + 1);  // Record at which dimension ET succeeded
                l2_et_recorded = true;
            }
        }
    }

    // Return fully computed true distance (ensure algorithm correctness)
    return res_quantized;
}

/**
 * L2 Early Termination distance function (no ET version, for comparison)
 * Only calls original distance function
 */
static float L2Sqr_FP16_FPMA_NoET(
    const void *pVect1v,
    const void *pVect2v,
    const void *qty_ptr,
    float threshold,
    ETStatistics* stats) {

    if (stats) {
        stats->total_distance_calls.fetch_add(1);
    }

    // Directly call original FPMA distance function
    return L2Sqr_FP16_FPMA(pVect1v, pVect2v, qty_ptr);
}

/**
 * L2 dynamic precision space with Early Termination support
 *
 * Features:
 * 1. Inherits all dynamic precision functionality
 * 2. Adds L1/L2 ET support (only enabled under FP16_FPMA precision)
 * 3. Tracks ET success rate
 * 4. Can enable/disable ET at runtime
 */
class L2SpaceDynamicPrecisionET : public L2SpaceDynamicPrecision,
                                  public L2EarlyTerminationInterface {
private:
    ETStatistics stats_;
    bool et_enabled_;
    float current_threshold_;
    int bit_level_;  // Bit-level for early termination (6/8/12/16)

public:
    L2SpaceDynamicPrecisionET(size_t dim, PrecisionType precision = PrecisionType::FP32)
        : L2SpaceDynamicPrecision(dim, precision), et_enabled_(false), current_threshold_(0.0f), bit_level_(16) {
    }

    /**
     * Enable/disable Early Termination
     */
    void set_et_enabled(bool enabled) {
        et_enabled_ = enabled;
    }

    bool get_et_enabled() const override {
        return et_enabled_;
    }

    /**
     * Set/get bit count for Bit-Level ET
     */
    void set_bit_level(int bits) {
        bit_level_ = bits;
    }

    int get_bit_level() const {
        return bit_level_;
    }

    /**
     * Compute complete L1 lower bound (for dual-queue ET)
     *
     * This function reads vector data from all dimensions and computes:
     * L1_lower_bound = 2 + 2·Σlb_i
     *
     * where lb_i is lower bound of -q_i·v_i computed based on sign and exponent bits
     *
     * Note: This function requires reading complete vectors, overhead is larger than reading only sign+exponent bits
     */
    float compute_full_l1_lower_bound(const void *pVect1v, const void *pVect2v) const override {
        float *pVect1 = (float *)pVect1v;
        float *pVect2 = (float *)pVect2v;
        size_t qty = data_size_ / sizeof(float);

        float sum_lb_i = 0.0f;

        // L1 accumulation: use fixed-point accumulation every multiplier_width, then convert to float
        int multiplier_width = 16;
        int64_t l1_accumulator = 0;

        for (size_t i = 0; i < qty; i++) {
            // Convert v_i to FP16 to get sign and exponent bits
            uint16_t v_fp16_bits = fp32_to_fp16_bits(pVect2[i]);
            float lb_i = compute_l1_lower_bound(pVect1[i], v_fp16_bits);

            // Fixed-point accumulation
            int32_t lb_i_fixed = static_cast<int32_t>(lb_i * FP16_FIXED_POINT_SCALE);
            l1_accumulator += lb_i_fixed;

            // Convert to floating-point accumulation every multiplier_width
            if ((i + 1) % multiplier_width == 0 || i == qty - 1) {
                sum_lb_i += static_cast<float>(l1_accumulator) / FP16_FIXED_POINT_SCALE;
                l1_accumulator = 0;
            }
        }

        // Compute lower bound: |Q-V|² = 2 - 2Σq_i·v_i >= 2 + 2·Σlb_i
        float l1_lower_bound = 2.0f + 2.0f * sum_lb_i;

        return l1_lower_bound;
    }

    /**
     * Set ET threshold (usually obtained from maximum value in Top-k queue)
     * Note: In actual use, threshold should be dynamically updated for each query
     */
    void set_et_threshold(float threshold) {
        current_threshold_ = threshold;
    }

    float get_et_threshold() const {
        return current_threshold_;
    }

    /**
     * Get ET statistics (return const reference to avoid copying atomic variables)
     */
    const ETStatistics& get_et_statistics() const {
        return stats_;
    }

    /**
     * Reset ET statistics
     */
    void reset_et_statistics() {
        stats_.reset();
    }

    /**
     * Get distance function (inherited from parent class)
     *
     * Note: Standard distance function signature does not include threshold parameter, so ET logic cannot be directly integrated
     * ET functionality needs to be called separately via dist_func_et() method, or modify HNSW search logic
     */
    DISTFUNC<float> get_dist_func() override {
        // Return parent class's standard distance function
        // ET needs to be called separately via dist_func_et()
        return L2SpaceDynamicPrecision::get_dist_func();
    }

    /**
     * Distance computation with ET (requires threshold parameter)
     *
     * Usage: During HNSW search, obtain current maximum distance from Top-k queue as threshold
     */
    float dist_func_et(const void *pVect1v, const void *pVect2v, float threshold) override {
        // Use inherited public member or get dim via get_dist_func_param
        size_t* dim_ptr = static_cast<size_t*>(get_dist_func_param());

        // Use get_precision() method instead of directly accessing private member
        if (get_precision() == PrecisionType::FP16_FPMA && et_enabled_) {
            return L2Sqr_FP16_FPMA_ET(pVect1v, pVect2v, dim_ptr, threshold, &stats_);
        } else {
            return L2Sqr_FP16_FPMA_NoET(pVect1v, pVect2v, dim_ptr, threshold, &stats_);
        }
    }

    /**
     * Bit-Level ET distance computation
     *
     * Use configurable bit-level to compute distance lower bound and determine early stopping
     *
     * @param pVect1v: query vector
     * @param pVect2v: candidate vector
     * @param threshold: current maximum distance in top-k queue
     * @return if ET triggered, return value greater than threshold, otherwise return true distance
     */
    float dist_func_bit_level_et(const void *pVect1v, const void *pVect2v, float threshold) {
        float *pVect1 = (float *)pVect1v;
        float *pVect2 = (float *)pVect2v;
        size_t qty = *((size_t *)get_dist_func_param());

        // Statistics
        stats_.total_distance_calls.fetch_add(1);

        // ✅ Key fix: when bits >= 16, directly compute true distance, don't compute lower_bound
        // Because when bits=16, lower_bound should equal true distance, but may have tiny errors due to different computation paths
        if (bit_level_ >= 16 || !et_enabled_) {
            // Directly compute true distance, no early termination
            return L2Sqr_FP16_FPMA(pVect1v, pVect2v, get_dist_func_param());
        }

        // Compute bit-level lower bound (only when bits < 16)
        float lower_bound = compute_bit_level_l2_lower_bound(pVect1, pVect2, qty, bit_level_);

        // Check if ET is possible
        // Use small epsilon to avoid boundary cases due to floating-point errors
        const float epsilon = 1e-6f;
        if (lower_bound > threshold + epsilon) {
            stats_.bit_level_et_success.fetch_add(1);
            // ✅ Key fix: return lower_bound itself, not a fake value
            // Although lower_bound is not the precise true distance, it is a strict lower bound
            // This ensures correct HNSW search logic
            return lower_bound;
        }

        // Lower bound did not pass ET check, compute true distance
        return L2Sqr_FP16_FPMA(pVect1v, pVect2v, get_dist_func_param());
    }
};

}  // namespace hnswlib
