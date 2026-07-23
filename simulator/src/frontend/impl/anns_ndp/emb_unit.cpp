#include <unordered_map>
#include <type_traits>
#include <bitset>
#include <algorithm>
#include <unordered_set>
#include <cstring>
#include <cmath>
#include <cctype>
#include <limits>
#include "hnsw.h"


namespace Ramulator {

namespace {

inline uint16_t fp32_to_fp16_bits_dual_mem(float val) {
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &val, sizeof(float));
    uint32_t sign = (f32_bits >> 16) & 0x8000;
    int32_t f32_expo = (f32_bits >> 23) & 0xFF;
    uint32_t f32_mant = f32_bits & 0x7FFFFF;
    if (f32_expo == 0xFF) {
        if (f32_mant == 0) return static_cast<uint16_t>(sign | 0x7C00);
        uint16_t nan_mant = static_cast<uint16_t>(f32_mant >> 13);
        if (nan_mant == 0) nan_mant = 1;
        return static_cast<uint16_t>(sign | 0x7C00 | nan_mant);
    }
    int32_t fp16_expo = f32_expo - 127 + 15;
    if (fp16_expo <= 0) {
        if (fp16_expo < -10) return static_cast<uint16_t>(sign);
        uint32_t mant = (f32_mant | 0x800000) >> (1 - fp16_expo + 13);
        if (((f32_mant | 0x800000) >> (1 - fp16_expo + 12)) & 1) mant += 1;
        return static_cast<uint16_t>(sign | mant);
    }
    if (fp16_expo >= 31) return static_cast<uint16_t>(sign | 0x7C00);
    uint32_t fp16_mant = f32_mant >> 13;
    uint32_t round_bit = (f32_mant >> 12) & 1;
    uint32_t sticky_bits = f32_mant & 0xFFF;
    if (round_bit && (sticky_bits || (fp16_mant & 1))) {
        fp16_mant++;
        if (fp16_mant >= 1024) {
            fp16_expo++;
            fp16_mant = 0;
            if (fp16_expo >= 31) return static_cast<uint16_t>(sign | 0x7C00);
        }
    }
    return static_cast<uint16_t>(sign | (fp16_expo << 10) | fp16_mant);
}

inline float fp16_bits_to_fp32_dual_mem(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits >> 15) & 0x1U;
    const uint32_t exponent = static_cast<uint32_t>(bits >> 10) & 0x1FU;
    const uint32_t mantissa = static_cast<uint32_t>(bits & 0x3FFU);

    uint32_t fp32_bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            fp32_bits = sign << 31;
        } else {
            int32_t exp = -14;
            uint32_t mant = mantissa;
            while ((mant & 0x400U) == 0U) {
                mant <<= 1U;
                --exp;
            }
            mant &= 0x3FFU;
            fp32_bits = (sign << 31) | (static_cast<uint32_t>(exp + 127) << 23) | (mant << 13);
        }
    } else if (exponent == 0x1FU) {
        fp32_bits = (sign << 31) | 0x7F800000U | (mantissa << 13);
    } else {
        const uint32_t fp32_exponent = exponent - 15U + 127U;
        fp32_bits = (sign << 31) | (fp32_exponent << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &fp32_bits, sizeof(float));
    return result;
}

inline float sanitize_fp16_interval_endpoint(float value, bool negative_sign) {
    if (std::isnan(value)) {
        return negative_sign ? -std::numeric_limits<float>::infinity()
                             : std::numeric_limits<float>::infinity();
    }
    return value;
}

struct Fp16PrefixInterval {
    float low = 0.0f;
    float high = 0.0f;
    float stored = 0.0f;
};

inline Fp16PrefixInterval get_fp16_prefix_interval_dual_mem(float value, uint32_t bit_end) {
    const uint16_t stored_bits = fp32_to_fp16_bits_dual_mem(value);
    const float stored = fp16_bits_to_fp32_dual_mem(stored_bits);
    const uint32_t clamped_bit_end = std::min<uint32_t>(bit_end, 16U);
    if (clamped_bit_end >= 16U) {
        return Fp16PrefixInterval{stored, stored, stored};
    }

    const uint32_t tail_bits = 16U - clamped_bit_end;
    const uint16_t prefix_mask = (clamped_bit_end == 0U)
        ? static_cast<uint16_t>(0U)
        : static_cast<uint16_t>(0xFFFFU << tail_bits);
    const uint16_t tail_mask = (tail_bits == 16U)
        ? static_cast<uint16_t>(0xFFFFU)
        : static_cast<uint16_t>((1U << tail_bits) - 1U);
    const uint16_t floor_bits = static_cast<uint16_t>(stored_bits & prefix_mask);
    const uint16_t ceil_bits = static_cast<uint16_t>(floor_bits | tail_mask);

    float endpoint_a = sanitize_fp16_interval_endpoint(
        fp16_bits_to_fp32_dual_mem(floor_bits), (floor_bits & 0x8000U) != 0U);
    float endpoint_b = sanitize_fp16_interval_endpoint(
        fp16_bits_to_fp32_dual_mem(ceil_bits), (ceil_bits & 0x8000U) != 0U);
    return Fp16PrefixInterval{std::min(endpoint_a, endpoint_b), std::max(endpoint_a, endpoint_b), stored};
}

inline float quantize_to_fp16_stored_value_dual_mem(float value) {
    return fp16_bits_to_fp32_dual_mem(fp32_to_fp16_bits_dual_mem(value));
}

inline float project_l2_to_fp16_prefix_interval_dual_mem(float cand_value, float query_value, uint32_t bit_end) {
    const Fp16PrefixInterval interval = get_fp16_prefix_interval_dual_mem(cand_value, bit_end);
    if (query_value < interval.low) {
        return interval.low;
    }
    if (query_value > interval.high) {
        return interval.high;
    }
    return query_value;
}

inline float select_ip_lower_bound_fp16_prefix_interval_dual_mem(float cand_value, float query_value, uint32_t bit_end) {
    const Fp16PrefixInterval interval = get_fp16_prefix_interval_dual_mem(cand_value, bit_end);
    return (query_value >= 0.0f) ? interval.high : interval.low;
}

inline float compute_l1_lower_bound_dual_mem(float q_val, uint16_t v_fp16_bits) {
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
           compute_l1_lower_bound_dual_mem(query_value, v_fp16_bits);
}

constexpr uint32_t kCacheLineBytes = 64;
constexpr uint32_t kVectorLayoutRowBytes = 16U * kCacheLineBytes;
constexpr uint32_t kVectorLayoutColLineBits = 7U;
constexpr uint64_t kVectorLayoutColsPerRow = 1ULL << kVectorLayoutColLineBits;
constexpr uint32_t kVectorLayoutNumBankGroups = 4U;
constexpr uint32_t kVectorLayoutNumBanksPerGroup = 4U;
constexpr uint32_t kStandardFpDistanceOpsPerDim = 2U;

inline hnswlib::FP16L2SquareMethod parse_fp16_l2_square_method(const std::string& raw, bool* valid = nullptr) {
    std::string normalized = raw;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (valid != nullptr) {
        *valid = true;
    }
    if (normalized == "standard") {
        return hnswlib::FP16L2SquareMethod::Standard;
    }
    if (normalized == "fp16_fpma") {
        return hnswlib::FP16L2SquareMethod::FP16_FPMA;
    }
    if (normalized == "fp16_fpma_quantized") {
        return hnswlib::FP16L2SquareMethod::FP16_FPMA_Quantized;
    }
    if (valid != nullptr) {
        *valid = false;
    }
    return hnswlib::FP16L2SquareMethod::Standard;
}

inline uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    if (alignment == 0) {
        return value;
    }
    return ((value + alignment - 1ULL) / alignment) * alignment;
}
inline Addr_t compose_vector_striped_line_addr(uint64_t logicalLine, uint32_t bg_base, uint32_t bg_count) {
    if (bg_count == 0) {
        return static_cast<Addr_t>(logicalLine * kCacheLineBytes);
    }
    const uint64_t slot_count = static_cast<uint64_t>(bg_count) * static_cast<uint64_t>(kVectorLayoutNumBanksPerGroup);
    const uint64_t slot = logicalLine % slot_count;
    const uint64_t rowcol = logicalLine / slot_count;
    const uint64_t col = rowcol % kVectorLayoutColsPerRow;
    const uint64_t row = rowcol / kVectorLayoutColsPerRow;
    const uint64_t bg_local = slot % static_cast<uint64_t>(bg_count);
    const uint64_t bank = slot / static_cast<uint64_t>(bg_count);
    const uint64_t bg = static_cast<uint64_t>(bg_base) + bg_local;
    const uint64_t physicalLine =
        (((row * static_cast<uint64_t>(kVectorLayoutNumBanksPerGroup)) + bank) * static_cast<uint64_t>(kVectorLayoutNumBankGroups) + bg) *
            kVectorLayoutColsPerRow +
        col;
    return static_cast<Addr_t>(physicalLine * kCacheLineBytes);
}
inline Addr_t compose_vector_partitioned_row_chunk_addr(uint64_t logicalLine, uint32_t bg_base, uint32_t bg_count) {
    if (bg_count == 0) {
        return static_cast<Addr_t>(logicalLine * kCacheLineBytes);
    }
    const uint64_t slot_count = static_cast<uint64_t>(bg_count) * static_cast<uint64_t>(kVectorLayoutNumBanksPerGroup);
    const uint64_t logical_row = logicalLine / kVectorLayoutColsPerRow;
    const uint64_t col = logicalLine % kVectorLayoutColsPerRow;
    const uint64_t slot = logical_row % slot_count;
    const uint64_t physical_row = logical_row / slot_count;
    const uint64_t bg_local = slot % static_cast<uint64_t>(bg_count);
    const uint64_t bank = slot / static_cast<uint64_t>(bg_count);
    const uint64_t bg = static_cast<uint64_t>(bg_base) + bg_local;
    const uint64_t physicalLine =
        (((physical_row * static_cast<uint64_t>(kVectorLayoutNumBanksPerGroup)) + bank) *
             static_cast<uint64_t>(kVectorLayoutNumBankGroups) +
         bg) *
            kVectorLayoutColsPerRow +
        col;
    return static_cast<Addr_t>(physicalLine * kCacheLineBytes);
}
inline uint32_t get_dual_phase1_bits(uint32_t native_data_bit_width, bool is_float) {
    if (is_float) {
        if (native_data_bit_width <= 16) return 6U;
        return 9U;
    }
    return std::max<uint32_t>(1, native_data_bit_width / 2);
}

inline uint32_t get_dual_phase2_bits(uint32_t native_data_bit_width, bool is_float) {
    const uint32_t phase1_bits = get_dual_phase1_bits(native_data_bit_width, is_float);
    return (native_data_bit_width > phase1_bits) ? (native_data_bit_width - phase1_bits) : 1U;
}

inline uint32_t get_dual_phase_dim_step(uint32_t phase, uint32_t native_data_bit_width, bool is_float) {
    const uint32_t fetch_bits = (phase == 1)
        ? get_dual_phase1_bits(native_data_bit_width, is_float)
        : get_dual_phase2_bits(native_data_bit_width, is_float);
    uint32_t dim_step = 512U / fetch_bits;
    return dim_step > 0 ? dim_step : 1U;
}

} // anonymous namespace

void HNSWEmbUnit::recordMultiplications(uint32_t nDim, DisReq* source) {
    uint64_t count = nDim;
    if (source) {
        source->mulCount += count;
    }
}

hnswlib::FP16L2SquareMethod HNSWEmbUnit::getEffectiveL2SquareMethod(const std::string& spacetype, bool allowApprox) const {
    if (!allowApprox || fp16L2SquareMethod == hnswlib::FP16L2SquareMethod::Standard) {
        return hnswlib::FP16L2SquareMethod::Standard;
    }
    if (spacetype != "L2") {
        return hnswlib::FP16L2SquareMethod::Standard;
    }
    if (travUnit != nullptr && travUnit->datatype != "isFloat") {
        return hnswlib::FP16L2SquareMethod::Standard;
    }
    return fp16L2SquareMethod;
}

Type HNSWEmbUnit::computeDistanceValue(
    const Type* x,
    const Type* y,
    uint32_t nDim,
    const std::string& spacetype,
    bool allowApprox) const
{
    if (spacetype == "L2") {
        const auto method = getEffectiveL2SquareMethod(spacetype, allowApprox);
        return hnswlib::compute_l2_distance(x, y, static_cast<size_t>(nDim), method);
    }

    Type res = 0;
    for (uint32_t i = 0; i < nDim; i++) {
        res += -x[i] * y[i];
    }
    return res;
}

HNSWEmbUnit::KernelTimingCfg HNSWEmbUnit::getKernelTimingCfg(ComputeTaskKind kind) const {
    KernelTimingCfg cfg;
    switch (kind) {
        case ComputeTaskKind::DualQueueCoarseLowerBound:
            cfg.lanes = std::max<uint32_t>(1, dualQueueCoarseLanes);
            cfg.issueInterval = std::max<uint32_t>(1, dualQueueCoarseCyclesPerOp);
            cfg.pipelineLatency = std::max<uint32_t>(1, dualQueueCoarseLatencyCycles);
            cfg.setupCycles = dualQueueCoarseSetupCycles;
            cfg.finalizeCycles = dualQueueCoarseFinalizeCycles;
            cfg.pipelined = dualQueueCoarsePipelined;
            break;
        case ComputeTaskKind::DualQueueFineApproxDistance:
            cfg.lanes = std::max<uint32_t>(1, dualQueueFineLanes);
            cfg.issueInterval = std::max<uint32_t>(1, dualQueueFineCyclesPerOp);
            cfg.pipelineLatency = std::max<uint32_t>(1, dualQueueFineLatencyCycles);
            cfg.setupCycles = dualQueueFineSetupCycles;
            cfg.finalizeCycles = dualQueueFineFinalizeCycles;
            cfg.pipelined = dualQueueFinePipelined;
            break;
        case ComputeTaskKind::StandardFpDistance:
            cfg.lanes = std::max<uint32_t>(1, stdFpLanes);
            cfg.issueInterval = std::max<uint32_t>(1, cyclesPerFMAC);
            cfg.pipelineLatency = std::max<uint32_t>(1, multiplierLatencyCycles);
            cfg.setupCycles = stdFpSetupCycles;
            cfg.finalizeCycles = stdFpFinalizeCycles;
            cfg.pipelined = fmacPipelined;
            break;
        case ComputeTaskKind::ZeroMemFinalize:
            cfg.lanes = std::max<uint32_t>(1, zeroMemFinalizeLanes);
            cfg.issueInterval = std::max<uint32_t>(1, zeroMemFinalizeIssueInterval);
            cfg.pipelineLatency = std::max<uint32_t>(1, zeroMemFinalizeLatencyCycles);
            cfg.setupCycles = zeroMemFinalizeSetupCycles;
            cfg.finalizeCycles = zeroMemFinalizeFinalizeCycles;
            cfg.pipelined = zeroMemFinalizePipelined;
            break;
        case ComputeTaskKind::None:
        default:
            break;
    }
    return cfg;
}

size_t HNSWEmbUnit::getComputePoolIndex(ComputeTaskKind kind) const {
    switch (kind) {
        case ComputeTaskKind::DualQueueCoarseLowerBound:
            return 0;
        case ComputeTaskKind::DualQueueFineApproxDistance:
            return 1;
        case ComputeTaskKind::ZeroMemFinalize:
            return 2;
        case ComputeTaskKind::StandardFpDistance:
        case ComputeTaskKind::None:
        default:
            return 3;
    }
}

const char* HNSWEmbUnit::getComputeTaskKindName(ComputeTaskKind kind) const {
    switch (kind) {
        case ComputeTaskKind::DualQueueCoarseLowerBound:
            return "dual_coarse";
        case ComputeTaskKind::DualQueueFineApproxDistance:
            return "dual_fine";
        case ComputeTaskKind::StandardFpDistance:
            return "std_fp";
        case ComputeTaskKind::ZeroMemFinalize:
            return "zero_mem_finalize";
        case ComputeTaskKind::None:
        default:
            return "none";
    }
}

uint32_t HNSWEmbUnit::getComputeResourceCount(size_t poolIndex) const {
    if (!dualQueueCrossLevelNMPEnable) {
        return 1;
    }
    switch (poolIndex) {
        case 0:
            return dualQueueCoarseNMPLevel == "bankgroup" ? kBGAwareNumBankGroups : 1U;
        case 1:
            if (dualQueueFineNMPLevel == "bank") {
                return kBGAwareNumBanks;
            }
            if (dualQueueFineNMPLevel == "bank_subarray") {
                return kBGAwareNumBanks * std::max<uint32_t>(1, dualQueueFineSubarrayWays);
            }
            return 1U;
        case 3:
            return dualQueueCoarseNMPLevel == "bankgroup" ? kBGAwareNumBankGroups : 1U;
        case 2:
        default:
            return 1U;
    }
}

uint32_t HNSWEmbUnit::predictSubarrayId(uint64_t row) const {
    const uint32_t subarray_count = std::max<uint32_t>(1, dualQueueFineSubarrayCount);
    const uint32_t interleave_rows = std::max<uint32_t>(1, dualQueueFineSubarrayInterleaveRows);
    return static_cast<uint32_t>((row / static_cast<uint64_t>(interleave_rows)) % static_cast<uint64_t>(subarray_count));
}

void HNSWEmbUnit::assignPredictedComputeTarget(PDisReq& pDisReq, const std::vector<Addr_t>& requestAddrs) const {
    pDisReq.computeTargetValid = false;
    pDisReq.targetBankGroup = 0;
    pDisReq.targetFlatBank = 0;
    pDisReq.targetRow = 0;
    pDisReq.targetSubarray = 0;
    if (requestAddrs.empty()) {
        return;
    }

    std::array<uint32_t, kBGAwareNumBankGroups> bgVotes = {0, 0, 0, 0};
    std::array<uint32_t, kBGAwareNumBanks> bankVotes = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::unordered_map<uint64_t, uint32_t> rowVotes;
    std::unordered_map<uint64_t, uint32_t> subarrayVotes;

    for (Addr_t addr : requestAddrs) {
        const PredictedBGTgt tgt = predictBGTgt(addr);
        if (tgt.bankgroup >= kBGAwareNumBankGroups || tgt.bank >= kBGAwareNumBanksPerGroup) {
            continue;
        }
        const uint32_t flatBank = flattenBank(tgt);
        const uint32_t subarray = predictSubarrayId(tgt.row);
        bgVotes[tgt.bankgroup] += 1;
        bankVotes[flatBank] += 1;
        rowVotes[(static_cast<uint64_t>(flatBank) << 56) ^ (tgt.row & 0x00FFFFFFFFFFFFFFULL)] += 1;
        subarrayVotes[(static_cast<uint64_t>(flatBank) << 32) | static_cast<uint64_t>(subarray)] += 1;
        pDisReq.computeTargetValid = true;
    }
    if (!pDisReq.computeTargetValid) {
        return;
    }

    uint32_t bestBG = 0;
    for (uint32_t bg = 1; bg < kBGAwareNumBankGroups; ++bg) {
        if (bgVotes[bg] > bgVotes[bestBG]) {
            bestBG = bg;
        }
    }
    uint32_t bestBank = 0;
    for (uint32_t bank = 1; bank < kBGAwareNumBanks; ++bank) {
        if (bankVotes[bank] > bankVotes[bestBank]) {
            bestBank = bank;
        }
    }
    uint64_t bestRow = 0;
    uint32_t bestRowVotes = 0;
    for (const auto& it : rowVotes) {
        const uint32_t flatBank = static_cast<uint32_t>(it.first >> 56);
        if (flatBank != bestBank) {
            continue;
        }
        if (it.second > bestRowVotes) {
            bestRowVotes = it.second;
            bestRow = it.first & 0x00FFFFFFFFFFFFFFULL;
        }
    }
    uint32_t bestSubarray = 0;
    uint32_t bestSubarrayVotes = 0;
    for (const auto& it : subarrayVotes) {
        const uint32_t flatBank = static_cast<uint32_t>(it.first >> 32);
        if (flatBank != bestBank) {
            continue;
        }
        if (it.second > bestSubarrayVotes) {
            bestSubarrayVotes = it.second;
            bestSubarray = static_cast<uint32_t>(it.first & 0xFFFFFFFFULL);
        }
    }

    pDisReq.targetBankGroup = bestBG;
    pDisReq.targetFlatBank = bestBank;
    pDisReq.targetRow = bestRow;
    pDisReq.targetSubarray = bestSubarray;
}

uint32_t HNSWEmbUnit::selectComputeResourceUnit(const DisReq& disReq, const PendingComputeTask& task) const {
    const uint32_t pool_units = getComputeResourceCount(task.poolIndex);
    if (pool_units <= 1) {
        return 0;
    }
    if (!dualQueueCrossLevelNMPEnable || !disReq.dualQueueTwoPhase || !task.req.computeTargetValid) {
        return 0;
    }
    switch (task.kind) {
        case ComputeTaskKind::DualQueueCoarseLowerBound:
            if (dualQueueCoarseNMPLevel == "bankgroup") {
                return std::min<uint32_t>(task.req.targetBankGroup, pool_units - 1);
            }
            return 0;
        case ComputeTaskKind::DualQueueFineApproxDistance:
            if (dualQueueFineNMPLevel == "bank") {
                return std::min<uint32_t>(task.req.targetFlatBank, pool_units - 1);
            }
            if (dualQueueFineNMPLevel == "bank_subarray") {
                const uint32_t ways = std::max<uint32_t>(1, dualQueueFineSubarrayWays);
                const uint32_t bank = std::min<uint32_t>(task.req.targetFlatBank, kBGAwareNumBanks - 1);
                return std::min<uint32_t>(bank * ways + (task.req.targetSubarray % ways), pool_units - 1);
            }
            return 0;
        case ComputeTaskKind::StandardFpDistance:
            if (disReq.dualQueuePhase == 1 && dualQueueCoarseNMPLevel == "bankgroup") {
                return std::min<uint32_t>(task.req.targetBankGroup, pool_units - 1);
            }
            return 0;
        case ComputeTaskKind::ZeroMemFinalize:
        case ComputeTaskKind::None:
        default:
            return 0;
    }
}

uint64_t HNSWEmbUnit::calculateComputeLatency(const ComputeTaskModel& task) const {
    if (task.kind == ComputeTaskKind::None || task.workItems == 0) {
        return 0;
    }

    const KernelTimingCfg cfg = getKernelTimingCfg(task.kind);
    const uint64_t lanes = std::max<uint32_t>(1, cfg.lanes);
    const uint64_t batches = (task.workItems + lanes - 1) / lanes;
    uint64_t latency = cfg.setupCycles;

    if (cfg.pipelined) {
        latency += cfg.pipelineLatency;
        if (batches > 1) {
            latency += (batches - 1) * std::max<uint32_t>(1, cfg.issueInterval);
        }
    } else {
        latency += batches * std::max<uint32_t>(cfg.pipelineLatency, cfg.issueInterval);
    }

    latency += cfg.finalizeCycles;
    const double ratio = (std::isfinite(embComputeCycleRatio) && embComputeCycleRatio > 0.0)
                             ? embComputeCycleRatio
                             : 1.0;
    const double scaled = static_cast<double>(latency) * ratio;
    return static_cast<uint64_t>(std::max<double>(1.0, std::ceil(scaled)));
}

Type HNSWEmbUnit::computeDualQueueDistanceFromScore(Type scoreAccum, Type querySquaredNorm) const {
    if (travUnit && travUnit->spacetype == "L2") {
        return std::max<Type>(0, querySquaredNorm + 2.0f * scoreAccum);
    }
    return scoreAccum;
}

Type HNSWEmbUnit::computeDualQueueExactScoreContribution(Type cand, Type query, DisReq* source) {
    (void)source;
    if (travUnit && travUnit->spacetype == "L2") {
        return 0.5f * cand * cand - query * cand;
    }
    return -query * cand;
}

Type HNSWEmbUnit::getDistance(const Type* x, const Type* y, uint32_t nDim, DisReq* source) {
    recordMultiplications(nDim, source);
    const std::string spacetype = travUnit ? travUnit->spacetype : disMethod;
    return computeDistanceValue(x, y, nDim, spacetype, true);
}
Type HNSWEmbUnit::getDistance(const Type* x, const Type* y, uint32_t nDim, std::string spacetype, DisReq* source) {//for travUnit that as not been initialized
    recordMultiplications(nDim, source);
    return computeDistanceValue(x, y, nDim, spacetype, true);
}

bool HNSWEmbUnit::usesAnsmetRuntimeTrueFp16BitChop(bool dtype) const {
    if (!ansmetRuntimeTrueFp16BitChopEnable || !dtype) {
        return false;
    }
    if (!earlyExitEnable || nativeDataBitWidth != 16) {
        return false;
    }
    if (travUnit == nullptr || travUnit->datatype != "isFloat") {
        return false;
    }
    return !travUnit->isDualQueueLowerBoundETEnabled() && !travUnit->isMFNNSEnabled();
}

Type HNSWEmbUnit::bitChop(Type x,Type q, uint32_t curBit, uint32_t bitEnd,int up, bool dtype,bool dataunsigned) {
    if (usesAnsmetRuntimeTrueFp16BitChop(dtype)) {
        const Fp16PrefixInterval interval = get_fp16_prefix_interval_dual_mem(x, bitEnd);
        if (bitEnd >= nativeDataBitWidth) {
            return interval.stored;
        }
        return up ? interval.high : interval.low;
    }
    typedef union {
        float f;
        struct {
            unsigned int mantisa : 23;
            unsigned int exponent : 8;
            unsigned int sign : 1;
        } parts;
    } float32_cast;
    typedef union {
        float f;
        struct {
            unsigned int exp_man : 31;
            unsigned int sign : 1;
        } parts;
    } float32_cast2;

    typedef union {
        int i;
        struct {
            unsigned int value : 31;
            unsigned int sign : 1;
        } parts;
    } int32_cast;
    int x2=(int)x;
    if (dtype && nativeDataBitWidth < 32 && bitEnd >= nativeDataBitWidth) {
        return x;
    }
    if(x < 0 && dataunsigned==false){
        x2 = -(int)x;
    }
    int q2=(int)q;
    if(dtype){
        if (bitEnd <=9 && up==1) {
            float32_cast2 d1 = { .f = x };
            int i2;
            std::memcpy(&i2, &d1.f, sizeof(float));
            int32_cast i3 = {.i=i2 };
            int32_cast increment={.i=0};
            increment.parts.sign=i3.parts.sign;
            increment.parts.value=1;
            i3.i=i3.i+(increment.i<<(32-bitEnd));
            std::memcpy(&d1.f, &i3.i, sizeof(int));
            return d1.f;
        }else if(bitEnd<32 && up==1){
            float32_cast d1 = { .f = x };
            float32_cast increment={.f=0};
            float32_cast decrease={.f=0};
            increment.parts.sign=d1.parts.sign;
            increment.parts.exponent=d1.parts.exponent;
            increment.parts.mantisa=1<<(32-bitEnd);
            decrease.parts.sign=d1.parts.sign;
            decrease.parts.exponent=d1.parts.exponent;
            decrease.parts.mantisa=0;
            d1.f=d1.f+increment.f-decrease.f;
            uint32_t newMantisa=((d1.parts.mantisa>>(32-bitEnd))<<(32-bitEnd));//| (d2.parts.mantisa & 0xFFFF);
            d1.parts.mantisa=newMantisa;
            return d1.f;
        }else if(bitEnd<=9){
            float32_cast2 d1 = { .f = x };
            uint32_t newMantisa=((d1.parts.exp_man>>(32-bitEnd))<<(32-bitEnd));//| (d2.parts.mantisa & 0xFFFF);
            d1.parts.exp_man=newMantisa;
            return d1.f;
        }else if(bitEnd<32){
            float32_cast d1 = { .f = x };
            uint32_t newMantisa=((d1.parts.mantisa>>(32-bitEnd))<<(32-bitEnd));//| (d2.parts.mantisa & 0xFFFF);
            d1.parts.mantisa=newMantisa;
            return d1.f;
        }else if (bitEnd == 32) {
            return x;
        }
    }else if(!dtype){
        int32_cast d1 = {.i = x2};
        int32_cast d2 = {.i=q2};
        if(bitEnd < 32 && up == 1){
            int32_cast temp_x={.i=x2};
            int32_cast temp_q={.i=q2};
            uint32_t qvalue_temp=((temp_q.parts.value>>(32-bitEnd))<<(32-bitEnd));
            temp_q.parts.value=qvalue_temp;
            uint32_t xvalue_temp=((temp_x.parts.value>>(32-bitEnd))<<(32-bitEnd));
            temp_x.parts.value=xvalue_temp;
            if(temp_x.i==0 && temp_q.i==0){
                d1.i=d2.i;
                return (float)d1.i;
            }else{
            int32_cast increment = {.i=0 };
            increment.parts.sign=d1.parts.sign;
            increment.parts.value=1;
            d1.i=d1.i+(increment.i<<(32-bitEnd));
            uint32_t newValue=((d1.parts.value>>(32-bitEnd))<<(32-bitEnd));
            d1.parts.value=newValue;
            if(x < 0 && dataunsigned==false){
                return -(float)d1.i;
            }
            return (float)d1.i;
            }
        }else if(bitEnd < 32){
            int32_cast temp_x={.i=x2};
            int32_cast temp_q={.i=q2};
            uint32_t qvalue_temp=((temp_q.parts.value>>(32-bitEnd))<<(32-bitEnd));
            temp_q.parts.value=qvalue_temp;
            uint32_t xvalue_temp=((temp_x.parts.value>>(32-bitEnd))<<(32-bitEnd));
            temp_x.parts.value=xvalue_temp;
            if(temp_x.i==0 && temp_q.i==0){
                d1.i=d2.i;
                return (float)d1.i;
            }else{
            uint32_t newValue=((d1.parts.value>>(32-bitEnd))<<(32-bitEnd));//| (d2.parts.value & (1<<(32-bitEnd)-1));
            d1.parts.value=newValue;
            if(x < 0 && dataunsigned==false){
                return -(float)d1.i;
            }
            return (float)d1.i;
            }
        }else if(bitEnd == 32){
            return x;
        }

    }
    return 0;
}

Type HNSWEmbUnit::bitChop(Type x,Type q, uint32_t curBit, uint32_t bitEnd,int up) {
    if (usesAnsmetRuntimeTrueFp16BitChop(travUnit && travUnit->datatype == "isFloat")) {
        const Fp16PrefixInterval interval = get_fp16_prefix_interval_dual_mem(x, bitEnd);
        if (bitEnd >= nativeDataBitWidth) {
            return interval.stored;
        }
        return up ? interval.high : interval.low;
    }
    typedef union {
        float f;
        struct {
            unsigned int mantisa : 23;
            unsigned int exponent : 8;
            unsigned int sign : 1;
        } parts;
    } float32_cast;
    typedef union {
        float f;
        struct {
            unsigned int exp_man : 31;
            unsigned int sign : 1;
        } parts;
    } float32_cast2;

    typedef union {
        int i;
        struct {
            unsigned int value : 31;
            unsigned int sign : 1;
        } parts;
    } int32_cast;
    int x2=(int)x;
    if (travUnit->datatype == "isFloat" && nativeDataBitWidth < 32 && bitEnd >= nativeDataBitWidth) {
        return x;
    }
    if(x < 0 && travUnit->dataunsigned == false){
        x2 = -(int)x;
    }
    int q2=(int)q;
    if(travUnit->datatype == "isFloat"){ //
        if (bitEnd <=9 && up==1) {
            float32_cast2 d1 = { .f = x };
            int i2;
            std::memcpy(&i2, &d1.f, sizeof(float));
            int32_cast i3 = {.i=i2 };
            int32_cast increment={.i=0};
            increment.parts.sign=i3.parts.sign;
            increment.parts.value=1;
            i3.i=i3.i+(increment.i<<(32-bitEnd));
            std::memcpy(&d1.f, &i3.i, sizeof(int));
            return d1.f;
        }else if(bitEnd<32 && up==1){
            float32_cast d1 = { .f = x };
            float32_cast increment={.f=0};
            float32_cast decrease={.f=0};
            increment.parts.sign=d1.parts.sign;
            increment.parts.exponent=d1.parts.exponent;
            increment.parts.mantisa=1<<(32-bitEnd);
            decrease.parts.sign=d1.parts.sign;
            decrease.parts.exponent=d1.parts.exponent;
            decrease.parts.mantisa=0;
            d1.f=d1.f+increment.f-decrease.f;
            uint32_t newMantisa=((d1.parts.mantisa>>(32-bitEnd))<<(32-bitEnd));//| (d2.parts.mantisa & 0xFFFF);
            d1.parts.mantisa=newMantisa;
            return d1.f;
        }else if(bitEnd<=9){
            float32_cast2 d1 = { .f = x };
            uint32_t newMantisa=((d1.parts.exp_man>>(32-bitEnd))<<(32-bitEnd));//| (d2.parts.mantisa & 0xFFFF);
            d1.parts.exp_man=newMantisa;
            return d1.f;
        }else if(bitEnd<32){
            float32_cast d1 = { .f = x };
            uint32_t newMantisa=((d1.parts.mantisa>>(32-bitEnd))<<(32-bitEnd));//| (d2.parts.mantisa & 0xFFFF);
            d1.parts.mantisa=newMantisa;
            return d1.f;
        }else if (bitEnd == 32) {
            return x;
        }
    }else if(travUnit->datatype == "isInt"){
        int32_cast d1 = {.i = x2};
        int32_cast d2 = {.i=q2};
        if(bitEnd < 32 && up == 1){
            int32_cast temp_x={.i=x2};
            int32_cast temp_q={.i=q2};
            uint32_t qvalue_temp=((temp_q.parts.value>>(32-bitEnd))<<(32-bitEnd));
            temp_q.parts.value=qvalue_temp;
            uint32_t xvalue_temp=((temp_x.parts.value>>(32-bitEnd))<<(32-bitEnd));
            temp_x.parts.value=xvalue_temp;
            if(temp_x.i==0 && temp_q.i==0){
                d1.i=d2.i;
                return (float)d1.i;
            }else{
            int32_cast increment = {.i=0 };
            increment.parts.sign=d1.parts.sign;
            increment.parts.value=1;
            d1.i=d1.i+(increment.i<<(32-bitEnd));
            uint32_t newValue=((d1.parts.value>>(32-bitEnd))<<(32-bitEnd));//| (d2.parts.value & (1<<(32-bitEnd)-1));
            d1.parts.value=newValue;
            if(x < 0 &&  travUnit->dataunsigned == false){
                return -(float)d1.i;
            }
            return (float)d1.i;
            }
        }else if(bitEnd < 32){
            int32_cast temp_x={.i=x2};
            int32_cast temp_q={.i=q2};
            uint32_t qvalue_temp=((temp_q.parts.value>>(32-bitEnd))<<(32-bitEnd));
            temp_q.parts.value=qvalue_temp;
            uint32_t xvalue_temp=((temp_x.parts.value>>(32-bitEnd))<<(32-bitEnd));
            temp_x.parts.value=xvalue_temp;
            if(temp_x.i==0 && temp_q.i==0){
                d1.i=d2.i;
                return (float)d1.i;
            }else{
            uint32_t newValue=((d1.parts.value>>(32-bitEnd))<<(32-bitEnd));//| (d2.parts.value & (1<<(32-bitEnd)-1));
            d1.parts.value=newValue;
            if(x < 0 &&  travUnit->dataunsigned == false){
                return -(float)d1.i;
            }
            return (float)d1.i;
            }
        }else if(bitEnd == 32){
            return x;
        }

    }
    return 0;
}

void HNSWEmbUnit::printBinary(Type value,int isint_tmp) {
    if(travUnit->datatype == "isInt" || isint_tmp){
        int ux;
        ux=(int)value;
        for (int i = sizeof(Type) * 8 - 1; i >= 0; i--) {
            putchar((ux & (1 << i)) ? '1' : '0');
        }
    }else if(travUnit->datatype == "isFloat"){
        unsigned int ux;
        float ux2 = static_cast<float>(std::abs(value));
        std::memcpy(&ux, &ux2, sizeof(float));
        for (int i = sizeof(Type) * 8 - 1; i >= 0; i--) {
            putchar((ux & (1 << i)) ? '1' : '0');
        }
    }
    printf("\n");
    fflush(stdout);

}

Type HNSWEmbUnit::mergeDistance(DisReq& disReq, PDisReq& resp) {
    const bool true_fp16_runtime =
        usesAnsmetRuntimeTrueFp16BitChop(travUnit && travUnit->datatype == "isFloat");
    if(earlyExitEnable){
        bitStep = travUnit->bitStep_array[disReq.last.ncurstep_last];
        dimStep = floor(512/bitStep);
    }else{
        // Keep full-precision distance computation, while modeledReadBitWidth only
        // changes how many dims are fetched per 64B request for cycle modeling.
        bitStep = nativeDataBitWidth;
        dimStep = std::max(1U, 512U / modeledReadBitWidth);
    }
    Type* candData = travUnit->getEmbData(disReq.candId);
    Type* candFormatData = (Type*)travUnit->hnsw->getFormatByInternalId(disReq.candId);
    uint32_t dimEnd = std::min(resp.curDim + dimStep, disReq.vDimEnd);
    uint32_t bitEnd = std::min(resp.curBit + bitStep, nBit);
    uint32_t bitEnd_last=resp.curBit;
    uint32_t dimEnd_last=resp.curDim;
    const bool full_step = (bitEnd == nBit);
    const auto full_distance_method = getEffectiveL2SquareMethod(travUnit->spacetype, true);
    const bool use_fp16_fpma_per_dim =
        full_step && full_distance_method == hnswlib::FP16L2SquareMethod::FP16_FPMA;
    const bool use_fp16_fpma_quantized_recompute =
        full_step && full_distance_method == hnswlib::FP16L2SquareMethod::FP16_FPMA_Quantized;
    auto getDistanceNoCount = [&](const Type* x, const Type* y, uint32_t nDimLocal, bool allowApprox) -> Type {
        return computeDistanceValue(x, y, nDimLocal, travUnit->spacetype, allowApprox);
    };
    std::vector<Type> candPartialData(nDim, 0);
    for (size_t dim = 0; dim < nDim; dim++) {
        if(travUnit->spacetype == "L2"){
            candPartialData[dim] = *(resp.query.data() + dim);
        }else{
            candPartialData[dim] = 0;
        }

    }
    int Olid = travUnit->hnsw->outlier_list[disReq.candId];
    int col_c0 = getOutlierColC0(disReq.candId);
    for (size_t dim = resp.curDim; dim < dimEnd; dim++) {
        bool accurarcy_loss = false;
        bool dimOl = false;
        Type candFormatData1 = *(candFormatData+dim);
        union {
            Type input;
            unsigned int output;
        } data;
        data.input = candFormatData1;
        std::bitset<sizeof(Type) * 8> binaryRepresentation(data.output);
        if(binaryRepresentation[0]==1){
            dimOl = true;
        }
        if(accurarcy_loss && Olid){//outlier vec
            if(dimOl && bitEnd > nBit - col_c0 - 1){
                bitEnd = nBit - col_c0 - 1;
            }else if(!dimOl && bitEnd > nBit - 1){
                bitEnd = nBit - 1;
            }
        }

        if(bitEnd <= nBit - 1){
            candPartialData[dim] = adjustCandPartialData(resp.query.data(), dim, resp.curBit, candData, bitEnd,candFormatData,disReq.candId);
        }else{
            candPartialData[dim] = true_fp16_runtime
                ? quantize_to_fp16_stored_value_dual_mem(*(candData + dim))
                : *(candFormatData + dim);
        }
    }
    Type deltaDistance = 0, deltaDistance1 = 0;
    for (uint32_t dim = resp.curDim; dim < dimEnd; dim++) {
        Type cand = *(candData + dim);
        if (true_fp16_runtime) {
            cand = quantize_to_fp16_stored_value_dual_mem(cand);
        }
        Type pCand = candPartialData[dim];
        Type query = *(resp.query.data() + dim);
        Type pQuery = query;//jyx TODO
        recordMultiplications(1, &disReq);
        Type pDistance = computeDistanceValue(&pCand, &pQuery, 1, travUnit->spacetype, use_fp16_fpma_per_dim);
        Type prevDistance = disReq.curDis[dim];
        deltaDistance += prevDistance;
        disReq.curDis[dim] = pDistance;// redundant multiplication
        disReq.pDistanceAccum += pDistance - prevDistance;
        deltaDistance1 += disReq.curDis[dim];
        Type exactDistance = getDistanceNoCount(&cand, &query, 1, false);
        Type selectedDistance = getDistanceNoCount(&cand, &query, 1, use_fp16_fpma_per_dim);
        if(dim == 10|| dim == 40||dim == 70 || dim == 100){
            m_logger->info("[{}] mergeDistance annsId {} candId {} dim {} bit {}-{} ncurstep_last {} pDistance {:.2f} fullResult {:.2f}", name.c_str(), disReq.annsId, disReq.candId, dim, resp.curBit, bitEnd,disReq.last.ncurstep_last, pDistance, exactDistance);
        }
        if (!use_fp16_fpma_per_dim) {
            assert(disReq.curDis[dim] <= exactDistance);
        }
        if (full_step) {
            assert(pDistance == selectedDistance);
            assert(disReq.curDis[dim] == selectedDistance);
        }
    }

    Type pDistance = disReq.pDistanceAccum;
    Type fDistance = 0;
    if (true_fp16_runtime) {
        for (uint32_t dim = 0; dim < disReq.vDimEnd; ++dim) {
            const Type exactCand = quantize_to_fp16_stored_value_dual_mem(candData[dim]);
            const Type query = resp.query[dim];
            if (travUnit->spacetype == "L2") {
                fDistance += (exactCand - query) * (exactCand - query);
            } else {
                fDistance += -exactCand * query;
            }
        }
    } else {
        fDistance = getDistanceNoCount(resp.query.data(), candData, disReq.vDimEnd, false);
    }
    if (use_fp16_fpma_quantized_recompute) {
        pDistance = getDistanceNoCount(resp.query.data(), candData, disReq.vDimEnd, true);
        disReq.pDistanceAccum = pDistance;
    }
    // if(pDistance>fDistance && bitEnd>4){
    //     printf("calculation ERROR: pRes %.2f>fRes %.2f\n",pDistance,fDistance);
    // }
    m_logger->info("[{}] mergeDistance annsId {} candId {} curDim {}-{} delta {:.2f} -> {:.2f} pResult {:.2f} fullResult {:.2f} upperbound {:.2f}", name.c_str(), disReq.annsId, disReq.candId, resp.curDim, dimEnd, deltaDistance, deltaDistance1, pDistance, fDistance,  disReq.upperbound);
    return pDistance;
}

// Apply hybrid bit/dim early-exit strategy, enabling calculation partial distance
Type HNSWEmbUnit::adjustCandPartialData0(Type* queryData, size_t dim, uint32_t curBit, Type* candData, uint32_t bitEnd, bool dtype,std::string spacetype,bool dataunsigned) {
    if (usesAnsmetRuntimeTrueFp16BitChop(dtype)) {
        const Type cand = *(candData + dim);
        const Type query = *(queryData + dim);
        if (spacetype == "L2") {
            return project_l2_to_fp16_prefix_interval_dual_mem(cand, query, bitEnd);
        }
        return select_ip_lower_bound_fp16_prefix_interval_dual_mem(cand, query, bitEnd);
    }
    Type candPartialData = 0;
    if(spacetype == "L2"){// L2
        if (*(candData + dim) * *(queryData + dim) > 0 && abs(*(queryData + dim)) > abs(*(candData + dim))) {
            candPartialData = bitChop(*(candData + dim), *(queryData + dim), curBit, bitEnd, 1,dtype,dataunsigned);
            if (abs(*(queryData + dim)) < abs(candPartialData)) {
                candPartialData = *(queryData + dim);
            }
        } else {
            candPartialData = bitChop(*(candData + dim), *(queryData + dim), curBit, bitEnd, 0,dtype,dataunsigned);
            if (abs(*(candData + dim) - *(queryData + dim)) < abs(candPartialData - *(queryData + dim))) {
                candPartialData = *(queryData + dim);
            }
        }
        return candPartialData;
    } else {// IP
        if(*(candData + dim) * *(queryData + dim) > 0){
            candPartialData = bitChop(*(candData + dim), *(queryData + dim), curBit, bitEnd, 1,dtype,dataunsigned);
        }else{
            candPartialData = bitChop(*(candData + dim), *(queryData + dim), curBit, bitEnd, 0,dtype,dataunsigned);
        }
        return candPartialData;//in 'getDistance', to continue using the 'upperbound' (actually we should use lowerbound for IP),we multiply '-1'
    }

}

Type HNSWEmbUnit::adjustCandPartialData(Type* queryData, size_t dim, uint32_t curBit, Type* candData, uint32_t bitEnd,Type* candFormatData, uint32_t candId) {
    Type candPartialData = 0;
    uint32_t bitEnd_temp = bitEnd;
    if(travUnit->hnsw->outlier_list[candId]!=0){
        int col_c0;
        Type candFormatData1 = *(candFormatData+dim);
        union {
            Type input;
            unsigned int output;
        } data;
        data.input = candFormatData1;
        bitEnd_temp = bitEnd_temp -1;

        std::bitset<sizeof(Type) * 8> binaryRepresentation(data.output);
        if(binaryRepresentation[0]==1){
            col_c0 = getOutlierColC0(candId);
            bitEnd_temp = bitEnd_temp - col_c0;
            //printf("adjust outlier vec candId:%d dim:%ld bitEnd_temp:%d col_c0:%d Olid:%d\n",candId,dim,bitEnd_temp,col_c0,Olid);

        }
    }
    bitEnd_temp = bitEnd_temp>0?bitEnd_temp:0;
    if (usesAnsmetRuntimeTrueFp16BitChop(travUnit && travUnit->datatype == "isFloat")) {
        const Type cand = *(candData + dim);
        const Type query = *(queryData + dim);
        if (travUnit->spacetype == "L2") {
            return project_l2_to_fp16_prefix_interval_dual_mem(cand, query, bitEnd_temp);
        }
        return select_ip_lower_bound_fp16_prefix_interval_dual_mem(cand, query, bitEnd_temp);
    }
    if(travUnit->spacetype == "L2"){// L2
        if (*(candData + dim) * *(queryData + dim) > 0 && abs(*(queryData + dim)) > abs(*(candData + dim))) {
            candPartialData = bitChop(*(candData + dim), *(queryData + dim), curBit, bitEnd_temp, 1);
            if (abs(*(queryData + dim)) < abs(candPartialData)) {
                candPartialData = *(queryData + dim);
            }
        } else {
            candPartialData = bitChop(*(candData + dim), *(queryData + dim), curBit, bitEnd_temp, 0);
            if (abs(*(candData + dim) - *(queryData + dim)) < abs(candPartialData - *(queryData + dim))) {
                candPartialData = *(queryData + dim);
            }
        }
        return candPartialData;
    } else {// IP
        if(*(candData + dim) * *(queryData + dim) > 0){
            candPartialData = bitChop(*(candData + dim), *(queryData + dim), curBit, bitEnd_temp, 1);
        }else{
            candPartialData = bitChop(*(candData + dim), *(queryData + dim), curBit, bitEnd_temp, 0);
        }
        return candPartialData;
    }

}

int HNSWEmbUnit::getOutlierColC0(uint32_t candId) const {
    if (travUnit == nullptr || travUnit->hnsw == nullptr) {
        return 0;
    }
    if (candId >= travUnit->hnsw->outlier_list.size()) {
        return 0;
    }
    int outlierId = travUnit->hnsw->outlier_list[candId];
    if (outlierId <= 0) {
        return 0;
    }
    size_t index = static_cast<size_t>(outlierId - 1);
    if (index >= travUnit->hnsw->col_c0_list.size()) {
        return 0;
    }
    return travUnit->hnsw->col_c0_list[index];
}


void HNSWEmbUnit::getNextDimBit(DisReq& disReq) {
    DimBit& db = disReq.last;
    if (disReq.dualQueueTwoPhase) {
        db.dim += get_dual_phase_dim_step(disReq.dualQueuePhase, nativeDataBitWidth, travUnit->datatype == "isFloat");
        return;
    }
    int n_bitstep_begin = preprocess(disReq.candId);
    if(!earlyExitEnable){
        db.dim+=dimStep;
    }else if (dimbitMode == "bitFirst") {
        db.bit += bitStep2;
        if (db.bit >= nBit) {
            db.bit=n_bitstep_begin;
            db.dim += dimStep2;
        }
    } else if (dimbitMode == "dimFirst") {
        int ncurstep;
        int ncurstep_last;
        if(db.ncurstep<32){
            ncurstep = db.ncurstep;
        }else{
            ncurstep = 32;
        }
        ncurstep_last = ncurstep;
        db.ncurstep_last = ncurstep_last;
        bitStep = travUnit->bitStep_array[ncurstep];
        dimStep = floor(512/bitStep);
        db.dim += dimStep;
        if (db.dim >= disReq.vDimEnd) {
            db.dim = disReq.vDimBase;
            db.bit += bitStep;
            db.ncurstep++;
        }
    } else {
        assert(false);
    }
}


bool HNSWEmbUnit::isLastDimBit(DisReq& disReq) {
    DimBit& db = disReq.last;
    if (disReq.dualQueueTwoPhase) {
        return db.dim >= disReq.vDimEnd;
    }
    if(!earlyExitEnable){
        return db.dim >= disReq.vDimEnd;
    }else if (dimbitMode == "bitFirst") {
        return db.dim >= disReq.vDimEnd;
    } else if (dimbitMode == "dimFirst") {
        return db.bit >= nBit;
        printf("db.bit:%d\n",db.bit);
    } else {
        assert(false);
        return false;
    }
}

Addr_t HNSWEmbUnit::mapVectorLineAddr(Addr_t logicalLineAddr, uint32_t phase) const {
    if (!travUnit) {
        return logicalLineAddr;
    }
    if (travUnit->usesVectorPhaseBankPartitioning() && (phase == 1 || phase == 2)) {
        const uint32_t bg_count = travUnit->getVectorPhasePartitionBGCount(phase);
        if (bg_count > 0 && bg_count < 4U) {
            const uint32_t bg_base = travUnit->getVectorPhasePartitionBGBase(phase);
            const uint64_t logicalLine = static_cast<uint64_t>(logicalLineAddr) / kCacheLineBytes;
            return compose_vector_partitioned_row_chunk_addr(logicalLine, bg_base, bg_count);
        }
    }
    if (!travUnit->usesStripedAddressMapping(phase)) {
        return logicalLineAddr;
    }
    const uint64_t logicalLine = static_cast<uint64_t>(logicalLineAddr) / kCacheLineBytes;
    return compose_vector_striped_line_addr(logicalLine, /* bg_base */ 0U, /* bg_count */ kVectorLayoutNumBankGroups);
}

uint32_t HNSWEmbUnit::getDualQueueModeledFetchBits(uint32_t phase) const {
    if (travUnit && travUnit->usesCoarseFineSplitVectorLayout()) {
        return (phase == 1)
            ? get_dual_phase1_bits(nativeDataBitWidth, travUnit->datatype == "isFloat")
            : get_dual_phase2_bits(nativeDataBitWidth, travUnit->datatype == "isFloat");
    }
    if (phase == 1) {
        return modeledReadBitWidth;
    }
    return 0;
}

bool HNSWEmbUnit::usesAnsmetFp16FullDimFetch(const DisReq& disReq) const {
    if (disReq.dualQueueTwoPhase || !earlyExitEnable || nativeDataBitWidth != 16) {
        return false;
    }
    if (travUnit == nullptr || travUnit->datatype != "isFloat") {
        return false;
    }
    return !travUnit->isDualQueueLowerBoundETEnabled() && !travUnit->isMFNNSEnabled();
}

uint32_t HNSWEmbUnit::getLinearStageBitStep(const DisReq& disReq, uint32_t curBit) {
    if (!earlyExitEnable) {
        return nativeDataBitWidth;
    }
    if (dimbitMode == "bitFirst") {
        return std::max<uint32_t>(1U, bitStep2);
    }
    if (dimbitMode != "dimFirst" || travUnit == nullptr || travUnit->bitStep_array.empty()) {
        return std::max<uint32_t>(1U, nativeDataBitWidth);
    }

    const int bit_begin = preprocess(disReq.candId);
    int remaining_bits = static_cast<int>(curBit) - bit_begin;
    uint32_t step_index = 0;
    while (remaining_bits > 0 && step_index + 1 < travUnit->bitStep_array.size()) {
        const int stage_bits = travUnit->bitStep_array[step_index];
        if (remaining_bits < stage_bits) {
            break;
        }
        remaining_bits -= stage_bits;
        step_index += 1;
    }
    return std::max<uint32_t>(1U, static_cast<uint32_t>(travUnit->bitStep_array[step_index]));
}

uint32_t HNSWEmbUnit::getLinearRequestDimStep(const DisReq& disReq, uint32_t curBit) {
    if (!earlyExitEnable) {
        return std::max<uint32_t>(1U, dimStep);
    }
    if (dimbitMode == "bitFirst") {
        return std::max<uint32_t>(1U, dimStep2);
    }
    const uint32_t stage_bit_step = getLinearStageBitStep(disReq, curBit);
    return std::max<uint32_t>(1U, 512U / stage_bit_step);
}

std::vector<Addr_t> HNSWEmbUnit::buildRequestAddrs(DisReq& disReq, uint32_t curDim, uint32_t curBit, bool commitChargedState) {
    const uint32_t phase = disReq.dualQueueTwoPhase ? disReq.dualQueuePhase : 0;
    uint64_t vector_slot_index = static_cast<uint64_t>(disReq.candId);
    uint64_t vectors_on_current_chip = travUnit ? static_cast<uint64_t>(travUnit->getDataCount()) : 0ULL;
    if (travUnit) {
        const uint32_t vertical_width = std::max<uint32_t>(1, disReq.vDimEnd - disReq.vDimBase);
        const uint32_t n_vertical_group = (vertical_width > 0 && (nDim % vertical_width) == 0)
            ? std::max<uint32_t>(1, nDim / vertical_width)
            : 1U;
        const uint32_t emb_units = std::max<uint32_t>(1, travUnit->getEmbUnitCount());
        const uint32_t n_v_emb_unit = std::max<uint32_t>(1, emb_units / n_vertical_group);
        const uint32_t local_rank = std::min<uint32_t>(n_v_emb_unit - 1U, travUnit->getVectorRankLocalId(disReq.candId));
        const bool partial_replica_node =
            travUnit->usesTopLayerReplicaPlacement() && travUnit->hasVectorReplicaSlot(disReq.candId);
        if (partial_replica_node) {
            vector_slot_index = static_cast<uint64_t>(travUnit->getVectorReplicaSlot(disReq.candId));
            vectors_on_current_chip = static_cast<uint64_t>(travUnit->getVectorReplicaCount());
        } else if (travUnit->getVectorPhysicalPlacementMode() == "naive" ||
                   travUnit->usesTopLayerReplicaPlacement()) {
            const uint64_t replica_prefix = travUnit->usesTopLayerReplicaPlacement()
                ? static_cast<uint64_t>(travUnit->getVectorReplicaCount())
                : 0ULL;
            vector_slot_index = replica_prefix + static_cast<uint64_t>(travUnit->getVectorLocalSlot(disReq.candId));
            vectors_on_current_chip = replica_prefix + static_cast<uint64_t>(travUnit->getVectorLocalRankCount(local_rank));
        }
    }
    if (!disReq.dualQueueTwoPhase) {
        if (!usesAnsmetFp16FullDimFetch(disReq)) {
            return {mapVectorLineAddr(static_cast<Addr_t>((vector_slot_index * nDim + curDim) * dimSize), phase)};
        }

        const uint32_t fetch_dim_step = getLinearRequestDimStep(disReq, curBit);
        const uint32_t dim_end = std::min(curDim + fetch_dim_step, disReq.vDimEnd);
        const uint64_t dims_requested = static_cast<uint64_t>(dim_end - curDim);
        if (dims_requested == 0) {
            return {};
        }

        const uint64_t full_dim_bits = static_cast<uint64_t>(nativeDataBitWidth);
        const uint64_t full_bits_per_vector = static_cast<uint64_t>(nDim) * full_dim_bits;
        const uint64_t vector_base_bits = vector_slot_index * full_bits_per_vector;
        const uint64_t start_bit = vector_base_bits + static_cast<uint64_t>(curDim) * full_dim_bits;
        const uint64_t end_bit = start_bit + dims_requested * full_dim_bits - 1;
        const Addr_t first_line_addr = static_cast<Addr_t>(((start_bit / 8ULL) / kCacheLineBytes) * kCacheLineBytes);
        const Addr_t last_line_addr = static_cast<Addr_t>(((end_bit / 8ULL) / kCacheLineBytes) * kCacheLineBytes);

        Addr_t new_first_line_addr = first_line_addr;
        if (disReq.ansmetLinearChargedEndBit >= 0) {
            const Addr_t charged_last_line_addr = static_cast<Addr_t>((((static_cast<uint64_t>(disReq.ansmetLinearChargedEndBit) / 8ULL) / kCacheLineBytes) * kCacheLineBytes));
            if (charged_last_line_addr >= last_line_addr) {
                return {};
            }
            new_first_line_addr = std::max(first_line_addr, static_cast<Addr_t>(charged_last_line_addr + kCacheLineBytes));
        }
        if (commitChargedState) {
            disReq.ansmetLinearChargedEndBit = std::max<int64_t>(disReq.ansmetLinearChargedEndBit, static_cast<int64_t>(end_bit));
        }

        std::vector<Addr_t> addrs;
        for (Addr_t addr = new_first_line_addr; addr <= last_line_addr; addr += kCacheLineBytes) {
            addrs.push_back(mapVectorLineAddr(addr, phase));
        }
        return addrs;
    }

    const bool coarseFineSplit = travUnit && travUnit->usesCoarseFineSplitVectorLayout();
    const bool rowAlignedCoarseFineSplit = travUnit && travUnit->usesRowAlignedCoarseFineVectorLayout();
    const bool globalCoarseFineSplit = travUnit && travUnit->usesGlobalCoarseFineVectorLayout();
    const uint32_t fetch_dim_step = get_dual_phase_dim_step(phase, nativeDataBitWidth, travUnit->datatype == "isFloat");
    const uint32_t dim_end = std::min(curDim + fetch_dim_step, disReq.vDimEnd);
    const uint64_t dims_requested = static_cast<uint64_t>(dim_end - curDim);
    if (dims_requested == 0) {
        return {static_cast<Addr_t>((disReq.candId * nDim + curDim) * dimSize)};
    }
    if (!coarseFineSplit && phase == 2) {
        return {};
    }

    const uint32_t phase1_bits = get_dual_phase1_bits(nativeDataBitWidth, travUnit->datatype == "isFloat");
    const uint32_t phase2_bits = get_dual_phase2_bits(nativeDataBitWidth, travUnit->datatype == "isFloat");
    const uint64_t coarse_bits_per_vector = static_cast<uint64_t>(nDim) * static_cast<uint64_t>(phase1_bits);
    const uint64_t fine_bits_per_vector = static_cast<uint64_t>(nDim) * static_cast<uint64_t>(phase2_bits);
    const uint64_t full_bits_per_vector = static_cast<uint64_t>(nDim) * static_cast<uint64_t>(nativeDataBitWidth);

    uint64_t start_bit = 0;
    uint64_t end_bit = 0;
    if (coarseFineSplit) {
        const uint32_t bits_per_dim = (phase == 1) ? phase1_bits : phase2_bits;
        if (rowAlignedCoarseFineSplit) {
            const uint64_t vector_slot_bits = align_up_u64(full_bits_per_vector, static_cast<uint64_t>(kVectorLayoutRowBytes) * 8ULL);
            const uint64_t vector_base_bits = vector_slot_index * vector_slot_bits;
            const uint64_t phase_offset_bits = (phase == 1)
                ? static_cast<uint64_t>(curDim) * static_cast<uint64_t>(phase1_bits)
                : coarse_bits_per_vector + static_cast<uint64_t>(curDim) * static_cast<uint64_t>(phase2_bits);
            const uint64_t request_bits = dims_requested * bits_per_dim;
            start_bit = vector_base_bits + phase_offset_bits;
            end_bit = start_bit + request_bits - 1;
        } else if (globalCoarseFineSplit) {
            const uint64_t phase_offset_bits = (phase == 1)
                ? vector_slot_index * coarse_bits_per_vector + static_cast<uint64_t>(curDim) * static_cast<uint64_t>(phase1_bits)
                : vectors_on_current_chip * coarse_bits_per_vector +
                      vector_slot_index * fine_bits_per_vector +
                      static_cast<uint64_t>(curDim) * static_cast<uint64_t>(phase2_bits);
            const uint64_t request_bits = dims_requested * bits_per_dim;
            start_bit = phase_offset_bits;
            end_bit = start_bit + request_bits - 1;
        } else {
            const uint64_t vector_base_bits = vector_slot_index * full_bits_per_vector;
            const uint64_t phase_offset_bits = (phase == 1)
                ? static_cast<uint64_t>(curDim) * static_cast<uint64_t>(phase1_bits)
                : coarse_bits_per_vector + static_cast<uint64_t>(curDim) * static_cast<uint64_t>(phase2_bits);
            const uint64_t request_bits = dims_requested * bits_per_dim;
            start_bit = vector_base_bits + phase_offset_bits;
            end_bit = start_bit + request_bits - 1;
        }
    } else {
        const uint64_t full_dim_bits = static_cast<uint64_t>(nativeDataBitWidth);
        const uint64_t request_bits = dims_requested * full_dim_bits;
        const uint64_t vector_base_bits = vector_slot_index * full_bits_per_vector;
        start_bit = vector_base_bits + static_cast<uint64_t>(curDim) * full_dim_bits;
        end_bit = start_bit + request_bits - 1;
    }
    const Addr_t first_line_addr = static_cast<Addr_t>(((start_bit / 8ULL) / kCacheLineBytes) * kCacheLineBytes);
    const Addr_t last_line_addr = static_cast<Addr_t>(((end_bit / 8ULL) / kCacheLineBytes) * kCacheLineBytes);

    const int64_t charged_end_bit = (phase == 1)
        ? disReq.dualQueuePhase1ChargedEndBit
        : disReq.dualQueuePhase2ChargedEndBit;
    Addr_t new_first_line_addr = first_line_addr;
    if (charged_end_bit >= 0) {
        const Addr_t charged_last_line_addr = static_cast<Addr_t>((((static_cast<uint64_t>(charged_end_bit) / 8ULL) / kCacheLineBytes) * kCacheLineBytes));
        if (charged_last_line_addr >= last_line_addr) {
            return {};
        }
        new_first_line_addr = std::max(first_line_addr, static_cast<Addr_t>(charged_last_line_addr + kCacheLineBytes));
    }
    if (commitChargedState) {
        int64_t& commit_end_bit = (phase == 1)
            ? disReq.dualQueuePhase1ChargedEndBit
            : disReq.dualQueuePhase2ChargedEndBit;
        commit_end_bit = std::max<int64_t>(commit_end_bit, static_cast<int64_t>(end_bit));
    }

    std::vector<Addr_t> addrs;
    for (Addr_t addr = new_first_line_addr; addr <= last_line_addr; addr += kCacheLineBytes) {
        addrs.push_back(mapVectorLineAddr(addr, phase));
    }
    return addrs;
}

std::vector<Addr_t> HNSWEmbUnit::getRequestAddrs(DisReq& disReq, uint32_t curDim, uint32_t curBit) {
    return buildRequestAddrs(disReq, curDim, curBit, true);
}

bool HNSWEmbUnit::isDualQueueScheduleEnabled() const {
    return dualQueueBGAwareScheduleEnable || dualQueueFineBankRowAwareScheduleEnable;
}

bool HNSWEmbUnit::needsDualQueueMemoryTracking() const {
    if (isDualQueueScheduleEnabled()) {
        return true;
    }
    return travUnit &&
           travUnit->usesHotNodeReplicaDispatch() &&
           travUnit->isHotReplicaRowAwareEnabled();
}

bool HNSWEmbUnit::isDualQueueStagedAdmissionEnabled() const {
    return dualQueueStagedAdmissionEnable;
}

uint32_t HNSWEmbUnit::getDualQueueFineAdmissionOccupancy() const {
    return dualQueueActiveFineDisReqs + static_cast<uint32_t>(dualQueuePendingFineAdmissions.size());
}

bool HNSWEmbUnit::canAdmitDualQueueCoarseDisReq() const {
    if (!isDualQueueStagedAdmissionEnabled()) {
        return true;
    }
    return dualQueueActiveCoarseDisReqs < dualQueueCoarseAdmissionWindow &&
           getDualQueueFineAdmissionOccupancy() < dualQueueFineAdmissionWindow;
}

void HNSWEmbUnit::markDualQueueCoarseAdmitted(DisReq& disReq) {
    if (!isDualQueueStagedAdmissionEnabled() || !disReq.dualQueueTwoPhase || disReq.dualQueueAdmissionStage != 0) {
        return;
    }
    disReq.dualQueueAdmissionStage = 1;
    disReq.dualQueueWaitingFineAdmission = false;
    disReq.dualQueueFineReadyCycle = 0;
    dualQueueActiveCoarseDisReqs += 1;
}

void HNSWEmbUnit::releaseDualQueueCoarseAdmission(DisReq& disReq) {
    if (!isDualQueueStagedAdmissionEnabled() || disReq.dualQueueAdmissionStage != 1) {
        return;
    }
    if (dualQueueActiveCoarseDisReqs > 0) {
        dualQueueActiveCoarseDisReqs -= 1;
    }
    disReq.dualQueueAdmissionStage = 0;
}

bool HNSWEmbUnit::tryActivateDualQueueFineAdmission(DisReq& disReq, uint64_t curCycle) {
    if (!isDualQueueStagedAdmissionEnabled() || !disReq.dualQueueTwoPhase) {
        return true;
    }
    if (dualQueueActiveFineDisReqs >= dualQueueFineAdmissionWindow) {
        return false;
    }
    releaseDualQueueCoarseAdmission(disReq);
    if (disReq.dualQueueWaitingFineAdmission && disReq.dualQueueFineReadyCycle > 0 && curCycle >= disReq.dualQueueFineReadyCycle) {
        s_dualq_stage_admission_fine_wait_cycles += curCycle - disReq.dualQueueFineReadyCycle;
    }
    disReq.dualQueueWaitingFineAdmission = false;
    disReq.dualQueueFineReadyCycle = 0;
    disReq.dualQueueAdmissionStage = 2;
    dualQueueActiveFineDisReqs += 1;
    s_dualq_stage_admission_fine_promoted += 1;
    return true;
}

void HNSWEmbUnit::deferDualQueueFineAdmission(DisReq& disReq, uint64_t curCycle) {
    if (!isDualQueueStagedAdmissionEnabled() || !disReq.dualQueueTwoPhase) {
        return;
    }
    releaseDualQueueCoarseAdmission(disReq);
    if (disReq.dualQueueWaitingFineAdmission) {
        return;
    }
    disReq.dualQueueWaitingFineAdmission = true;
    disReq.dualQueueFineReadyCycle = curCycle;
    dualQueuePendingFineAdmissions.push_back(disReq.disreqId);
    s_dualq_stage_admission_fine_deferred += 1;
}

void HNSWEmbUnit::releaseDualQueueFineAdmission(DisReq& disReq) {
    if (!isDualQueueStagedAdmissionEnabled() || disReq.dualQueueAdmissionStage != 2) {
        return;
    }
    if (dualQueueActiveFineDisReqs > 0) {
        dualQueueActiveFineDisReqs -= 1;
    }
    disReq.dualQueueAdmissionStage = 0;
    disReq.dualQueueWaitingFineAdmission = false;
    disReq.dualQueueFineReadyCycle = 0;
}

void HNSWEmbUnit::startDualQueueFinePhase(DisReq& disReq) {
    disReq.dualQueuePhase = 2;
    disReq.last = DimBit{disReq.vDimBase, 0, 0, 0};
    for (uint32_t i = 0; i < nParallelPDisReq; i++) {
        disReq.pends.push_back(disReq.last);
        trySendPDisReq(disReq.disreqId, disReq.last.dim, disReq.last.bit, 0);
        getNextDimBit(disReq);
        if (isLastDimBit(disReq)) {
            break;
        }
    }
}

void HNSWEmbUnit::tryPromoteDeferredDualQueueFineAdmissions(uint64_t curCycle) {
    if (!isDualQueueStagedAdmissionEnabled()) {
        return;
    }
    while (dualQueueActiveFineDisReqs < dualQueueFineAdmissionWindow && !dualQueuePendingFineAdmissions.empty()) {
        const uint64_t disreqId = dualQueuePendingFineAdmissions.front();
        dualQueuePendingFineAdmissions.pop_front();
        auto inflightIt = inflightDisReqs.find(disreqId);
        if (inflightIt == inflightDisReqs.end()) {
            continue;
        }
        DisReq& disReq = inflightIt->second;
        if (!disReq.dualQueueTwoPhase || disReq.dualQueuePhase != 2 || !disReq.dualQueueWaitingFineAdmission) {
            continue;
        }
        if (!tryActivateDualQueueFineAdmission(disReq, curCycle)) {
            dualQueuePendingFineAdmissions.push_front(disreqId);
            break;
        }
        startDualQueueFinePhase(disReq);
    }
}

uint32_t HNSWEmbUnit::flattenBank(const PredictedBGTgt& tgt) const {
    return tgt.bankgroup * kBGAwareNumBanksPerGroup + tgt.bank;
}

HNSWEmbUnit::PredictedBGTgt HNSWEmbUnit::predictBGTgt(Addr_t addr) const {
    PredictedBGTgt tgt;
    if (auto* multi_mem = dynamic_cast<MultiGenericDRAMSystem*>(m_memory_system)) {
        AddrVec_t addr_vec = multi_mem->decode_address_vec(addr);
        if (addr_vec.size() >= 5) {
            tgt.bankgroup = static_cast<uint32_t>(std::max(0, addr_vec[2]));
            tgt.bank = static_cast<uint32_t>(std::max(0, addr_vec[3]));
            tgt.row = static_cast<uint64_t>(std::max(0, addr_vec[4]));
            return tgt;
        }
    }
    const uint64_t line = static_cast<uint64_t>(addr) / kCacheLineBytes;
    tgt.bankgroup = static_cast<uint32_t>((line >> kVectorLayoutColLineBits) & 0x3ULL);
    tgt.bank = static_cast<uint32_t>((line >> (kVectorLayoutColLineBits + 2U)) & 0x3ULL);
    tgt.row = static_cast<uint64_t>(line >> (kVectorLayoutColLineBits + 4U));
    return tgt;
}

HNSWEmbUnit::ReplicaDispatchProbe HNSWEmbUnit::predictReplicaDispatchProbe(PointId candId,
                                                                           uint32_t vDimBase,
                                                                           uint32_t vDimEnd,
                                                                           bool dualQueueTwoPhase) {
    ReplicaDispatchProbe probe;
    probe.inflightLoad = static_cast<uint32_t>(inflightDisReqs.size());

    if (!dualQueueTwoPhase) {
        return probe;
    }

    DisReq disReq;
    disReq.candId = candId;
    disReq.vDimBase = vDimBase;
    disReq.vDimEnd = vDimEnd;
    disReq.nDim = nDim;
    disReq.dualQueueTwoPhase = dualQueueTwoPhase;
    disReq.dualQueuePhase = 1;
    disReq.dualQueuePhase1ChargedEndBit = -1;
    disReq.dualQueuePhase2ChargedEndBit = -1;

    const std::vector<Addr_t> addrs = buildRequestAddrs(disReq, vDimBase, 0, false);
    probe.lineCount = static_cast<uint32_t>(addrs.size());

    std::array<uint32_t, kBGAwareNumBankGroups> linesPerBG = {0, 0, 0, 0};
    std::array<uint32_t, kBGAwareNumBanks> linesPerBank = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (Addr_t addr : addrs) {
        const PredictedBGTgt tgt = predictBGTgt(addr);
        if (tgt.bankgroup >= kBGAwareNumBankGroups || tgt.bank >= kBGAwareNumBanksPerGroup) {
            continue;
        }
        const uint32_t flatBank = flattenBank(tgt);
        linesPerBG[tgt.bankgroup] += 1;
        linesPerBank[flatBank] += 1;
        probe.sumBgOccupancy += m_bgOutstanding[tgt.bankgroup] + linesPerBG[tgt.bankgroup];
        probe.sumBankOccupancy += m_bankOutstanding[flatBank] + linesPerBank[flatBank];
        if (m_bankOutstanding[flatBank] > 0 && m_bankRecentRow[flatBank] == static_cast<int64_t>(tgt.row)) {
            probe.rowHitPred += 1;
        }
        if (m_bgOutstanding[tgt.bankgroup] > 0 &&
            m_bgRecentBank[tgt.bankgroup] == static_cast<int32_t>(tgt.bank) &&
            m_bgRecentRow[tgt.bankgroup] == static_cast<int64_t>(tgt.row)) {
            probe.bgRowHitPred += 1;
        }
    }
    return probe;
}

void HNSWEmbUnit::updateBGOutstandingOnIssue(const std::vector<Addr_t>& addrs, uint32_t phase) {
    if (!needsDualQueueMemoryTracking()) {
        return;
    }
    for (Addr_t addr : addrs) {
        const PredictedBGTgt tgt = predictBGTgt(addr);
        if (tgt.bankgroup >= kBGAwareNumBankGroups || tgt.bank >= kBGAwareNumBanksPerGroup) {
            continue;
        }
        const uint32_t flatBank = flattenBank(tgt);
        m_bgOutstanding[tgt.bankgroup] += 1;
        m_bankOutstanding[flatBank] += 1;
        m_bgCredits[tgt.bankgroup] = (m_bgOutstanding[tgt.bankgroup] >= dualQueueBGAwareCreditLimit)
            ? 0
            : (dualQueueBGAwareCreditLimit - m_bgOutstanding[tgt.bankgroup]);
        m_bankCredits[flatBank] = (m_bankOutstanding[flatBank] >= dualQueueFineBankCreditLimit)
            ? 0
            : (dualQueueFineBankCreditLimit - m_bankOutstanding[flatBank]);
        m_bgRecentRow[tgt.bankgroup] = static_cast<int64_t>(tgt.row);
        m_bgRecentBank[tgt.bankgroup] = static_cast<int32_t>(tgt.bank);
        m_bankRecentRow[flatBank] = static_cast<int64_t>(tgt.row);
        if (dualQueueFineBankRowAwareScheduleEnable && phase == 2) {
            m_bankFineTokens[flatBank] += 1;
            m_bgFineTokens[tgt.bankgroup] += 1;
        }
    }
}

void HNSWEmbUnit::updateBGOutstandingOnReturn(Addr_t addr, uint32_t phase) {
    if (!needsDualQueueMemoryTracking()) {
        return;
    }
    const PredictedBGTgt tgt = predictBGTgt(addr);
    if (tgt.bankgroup >= kBGAwareNumBankGroups || tgt.bank >= kBGAwareNumBanksPerGroup) {
        return;
    }
    const uint32_t flatBank = flattenBank(tgt);
    if (m_bgOutstanding[tgt.bankgroup] > 0) {
        m_bgOutstanding[tgt.bankgroup] -= 1;
    }
    if (m_bankOutstanding[flatBank] > 0) {
        m_bankOutstanding[flatBank] -= 1;
    }
    m_bgCredits[tgt.bankgroup] = (m_bgOutstanding[tgt.bankgroup] >= dualQueueBGAwareCreditLimit)
        ? 0
        : (dualQueueBGAwareCreditLimit - m_bgOutstanding[tgt.bankgroup]);
    m_bankCredits[flatBank] = (m_bankOutstanding[flatBank] >= dualQueueFineBankCreditLimit)
        ? 0
        : (dualQueueFineBankCreditLimit - m_bankOutstanding[flatBank]);
    if (m_bgOutstanding[tgt.bankgroup] == 0) {
        m_bgRecentRow[tgt.bankgroup] = -1;
        m_bgRecentBank[tgt.bankgroup] = -1;
    }
    if (m_bankOutstanding[flatBank] == 0) {
        m_bankRecentRow[flatBank] = -1;
    }
    if (dualQueueFineBankRowAwareScheduleEnable && phase == 2) {
        if (m_bankFineTokens[flatBank] > 0) {
            m_bankFineTokens[flatBank] -= 1;
        }
        if (m_bgFineTokens[tgt.bankgroup] > 0) {
            m_bgFineTokens[tgt.bankgroup] -= 1;
        }
    }
}

HNSWEmbUnit::PendingPDisReqSelection HNSWEmbUnit::analyzePendingPDisReq(const PDisReq& pDisReq, uint64_t curCycle, bool anyCoarseReady) {
    PendingPDisReqSelection info;
    auto inflightIt = inflightDisReqs.find(pDisReq.disreqId);
    if (inflightIt == inflightDisReqs.end()) {
        info.zeroMem = true;
        return info;
    }
    DisReq& disReq = inflightIt->second;
    info.phase = disReq.dualQueueTwoPhase ? disReq.dualQueuePhase : 0;
    info.enqueueAge = curCycle >= pDisReq.enqueueCycle ? (curCycle - pDisReq.enqueueCycle) : 0;
    const std::vector<Addr_t> requestAddrs = buildRequestAddrs(disReq, pDisReq.curDim, pDisReq.curBit, false);
    info.zeroMem = requestAddrs.empty();
    info.lineCount = static_cast<uint32_t>(requestAddrs.size());
    info.minCredit = dualQueueBGAwareCreditLimit;
    info.minBankCredit = dualQueueFineBankCreditLimit;
    const bool collectFineRowKeys = dualQueueFineBankRowAwareScheduleEnable && info.phase == 2;
    if (collectFineRowKeys) {
        info.uniqueRowKeys.reserve(info.lineCount);
    }
    std::array<uint32_t, kBGAwareNumBankGroups> linesPerBG = {0, 0, 0, 0};
    std::array<uint32_t, kBGAwareNumBanks> linesPerBank = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (Addr_t addr : requestAddrs) {
        PredictedBGTgt tgt = predictBGTgt(addr);
        if (tgt.bankgroup >= kBGAwareNumBankGroups || tgt.bank >= kBGAwareNumBanksPerGroup) {
            continue;
        }
        const uint32_t flatBank = flattenBank(tgt);
        if (collectFineRowKeys) {
            const uint64_t key = (static_cast<uint64_t>(flatBank) << 56) ^ (tgt.row & 0x00FFFFFFFFFFFFFFULL);
            info.uniqueRowKeys.push_back(key);
        }
        linesPerBG[tgt.bankgroup] += 1;
        linesPerBank[flatBank] += 1;
        const uint32_t predictedBGOcc = m_bgOutstanding[tgt.bankgroup] + linesPerBG[tgt.bankgroup];
        const uint32_t predictedBankOcc = m_bankOutstanding[flatBank] + linesPerBank[flatBank];
        info.maxOccupancy = std::max(info.maxOccupancy, predictedBGOcc);
        info.sumOccupancy += predictedBGOcc;
        info.maxBankOccupancy = std::max(info.maxBankOccupancy, predictedBankOcc);
        info.sumBankOccupancy += predictedBankOcc;
        info.minCredit = std::min(info.minCredit, m_bgCredits[tgt.bankgroup]);
        info.minBankCredit = std::min(info.minBankCredit, m_bankCredits[flatBank]);
        if (predictedBGOcc >= dualQueueBGAwareCongestionThreshold) {
            info.congestedBGs += 1;
        }
        if (m_bankOutstanding[flatBank] > 0 && m_bankRecentRow[flatBank] == static_cast<int64_t>(tgt.row)) {
            info.rowHitPred += 1;
        }
        if (info.phase == 2 && dualQueueFineBankRowAwareScheduleEnable) {
            const uint32_t predictedBankTokens = m_bankFineTokens[flatBank] + linesPerBank[flatBank];
            const uint32_t predictedBGTokens = m_bgFineTokens[tgt.bankgroup] + linesPerBG[tgt.bankgroup];
            info.maxBankToken = std::max(info.maxBankToken, predictedBankTokens);
            info.maxBGToken = std::max(info.maxBGToken, predictedBGTokens);
            if (predictedBankTokens > dualQueueFineBankTokenLimit) {
                info.throttledBanks += 1;
            }
            if (predictedBGTokens > dualQueueFineBGTokenLimit) {
                info.throttledBGs += 1;
            }
        }
    }
    if (collectFineRowKeys && !info.uniqueRowKeys.empty()) {
        std::sort(info.uniqueRowKeys.begin(), info.uniqueRowKeys.end());
        info.uniqueRowKeys.erase(std::unique(info.uniqueRowKeys.begin(), info.uniqueRowKeys.end()), info.uniqueRowKeys.end());
    }
    if (dualQueueFineBankRowAwareScheduleEnable && info.phase == 2 && travUnit) {
        info.thresholdQueueFull = travUnit->isResultQueueFull(disReq.annsId);
        if (info.thresholdQueueFull) {
            const Type liveUpperbound = travUnit->getCurrentUpperbound(disReq.annsId);
            if (std::isfinite(liveUpperbound)) {
                const Type lowerBound = std::max<Type>(0, disReq.dualQueueLowerBound);
                const Type gap = std::max<Type>(0, liveUpperbound - lowerBound);
                const Type clampedGap = std::min<Type>(4.0f, gap);
                info.thresholdGainScaled = static_cast<int64_t>(std::llround(clampedGap * 4.0f));
            }
        }
        if (info.enqueueAge >= dualQueueFineAgeThreshold) {
            info.ageForced = true;
        }
    }
    if (info.phase == 2 && anyCoarseReady &&
        (info.maxOccupancy >= dualQueueBGAwareCongestionThreshold ||
         (dualQueueFineBankRowAwareScheduleEnable &&
          (info.maxBankToken > dualQueueFineBankTokenLimit || info.maxBGToken > dualQueueFineBGTokenLimit)))) {
        info.fineDeferredByCongestion = true;
    }
    return info;
}

size_t HNSWEmbUnit::selectNextPendingPDisReqIndex(uint64_t curCycle) {
    if (!isDualQueueScheduleEnabled() || pendPDisReqs.size() <= 1) {
        return 0;
    }
    bool anyCoarseReady = false;
    for (const PDisReq& pDisReq : pendPDisReqs) {
        auto inflightIt = inflightDisReqs.find(pDisReq.disreqId);
        if (inflightIt == inflightDisReqs.end()) {
            continue;
        }
        if (inflightIt->second.dualQueueTwoPhase && inflightIt->second.dualQueuePhase == 1) {
            anyCoarseReady = true;
            break;
        }
    }

    bool penalizedFinePresent = false;
    std::vector<PendingPDisReqSelection> infos(pendPDisReqs.size());
    std::unordered_map<uint64_t, uint32_t> rowBucketCounts;
    rowBucketCounts.reserve(pendPDisReqs.size() * 2);
    for (size_t idx = 0; idx < pendPDisReqs.size(); ++idx) {
        const PDisReq& pDisReq = pendPDisReqs[idx];
        auto inflightIt = inflightDisReqs.find(pDisReq.disreqId);
        if (inflightIt == inflightDisReqs.end()) {
            return idx;
        }
        if (!inflightIt->second.dualQueueTwoPhase) {
            return 0;
        }
        infos[idx] = analyzePendingPDisReq(pDisReq, curCycle, anyCoarseReady);
        if (infos[idx].fineDeferredByCongestion) {
            penalizedFinePresent = true;
        }
        if (dualQueueFineBankRowAwareScheduleEnable && infos[idx].phase == 2 && !infos[idx].zeroMem) {
            for (uint64_t key : infos[idx].uniqueRowKeys) {
                rowBucketCounts[key] += 1;
            }
        }
    }

    if (dualQueueFineBankRowAwareScheduleEnable) {
        for (size_t idx = 0; idx < infos.size(); ++idx) {
            PendingPDisReqSelection& info = infos[idx];
            if (info.phase != 2 || info.zeroMem) {
                continue;
            }
            uint32_t bestBurst = 0;
            for (uint64_t key : info.uniqueRowKeys) {
                auto it = rowBucketCounts.find(key);
                if (it != rowBucketCounts.end()) {
                    bestBurst = std::max(bestBurst, it->second);
                }
            }
            info.rowBurstPred = bestBurst;
        }
    }

    bool bestValid = false;
    size_t bestIdx = 0;
    int64_t bestScore = std::numeric_limits<int64_t>::max();
    uint64_t bestAge = 0;
    uint32_t bestPhase = std::numeric_limits<uint32_t>::max();
    PendingPDisReqSelection bestInfo;

    for (size_t idx = 0; idx < infos.size(); ++idx) {
        const PendingPDisReqSelection& info = infos[idx];
        int64_t score = 0;
        if (info.zeroMem) {
            score = -1000000;
        } else if (info.phase == 1 || !dualQueueFineBankRowAwareScheduleEnable) {
            score = static_cast<int64_t>(dualQueueBGAwareBalanceWeight) * static_cast<int64_t>(info.sumOccupancy)
                  + static_cast<int64_t>(dualQueueFineBalanceWeight) * static_cast<int64_t>(info.sumBankOccupancy)
                  - static_cast<int64_t>(info.minCredit)
                  - static_cast<int64_t>(info.minBankCredit)
                  + static_cast<int64_t>(info.lineCount);
            if (info.phase == 2) {
                score -= static_cast<int64_t>(dualQueueBGAwareLocalityWeight) * static_cast<int64_t>(info.rowHitPred);
                if (info.fineDeferredByCongestion) {
                    score += static_cast<int64_t>(dualQueueBGAwareFineDeferPenalty);
                }
            }
        } else {
            const uint32_t rowBurst = std::min(info.rowBurstPred, dualQueueFineRowBurstCap);
            const uint32_t bankTokenOver = (info.maxBankToken > dualQueueFineBankTokenLimit)
                ? (info.maxBankToken - dualQueueFineBankTokenLimit)
                : 0;
            const uint32_t bgTokenOver = (info.maxBGToken > dualQueueFineBGTokenLimit)
                ? (info.maxBGToken - dualQueueFineBGTokenLimit)
                : 0;
            const int64_t tokenPenalty = static_cast<int64_t>(dualQueueFineTokenPenalty) * static_cast<int64_t>(bankTokenOver + bgTokenOver);
            const int64_t localityBonus = static_cast<int64_t>(dualQueueFineLocalityWeight) * static_cast<int64_t>(info.rowHitPred + rowBurst);
            const int64_t thresholdBonus = static_cast<int64_t>(dualQueueFineThresholdGainWeight) * info.thresholdGainScaled;
            score = static_cast<int64_t>(dualQueueFineBalanceWeight) * static_cast<int64_t>(info.sumOccupancy + info.sumBankOccupancy)
                  + static_cast<int64_t>(info.lineCount)
                  - static_cast<int64_t>(info.minCredit)
                  - static_cast<int64_t>(info.minBankCredit)
                  - localityBonus
                  - thresholdBonus
                  + tokenPenalty;
            if (info.fineDeferredByCongestion) {
                score += static_cast<int64_t>(dualQueueBGAwareFineDeferPenalty);
            }
            if (info.ageForced) {
                score -= static_cast<int64_t>(dualQueueFineAgeWeight);
            }
        }
        if (!bestValid || score < bestScore ||
            (score == bestScore && info.phase < bestPhase) ||
            (score == bestScore && info.phase == bestPhase && info.enqueueAge > bestAge)) {
            bestValid = true;
            bestIdx = idx;
            bestScore = score;
            bestAge = info.enqueueAge;
            bestPhase = info.phase;
            bestInfo = info;
        }
    }

    if (bestValid) {
        if (dualQueueFineBankRowAwareScheduleEnable && bestInfo.phase == 2) {
            if (bestIdx > 0) {
                s_dualq_rowaware_reordered += 1;
            }
            if (bestInfo.rowHitPred > 0) {
                s_dualq_rowaware_rowhit_selected += 1;
            }
            if (bestInfo.rowBurstPred > 1) {
                s_dualq_rowaware_rowburst_selected += 1;
            }
            if (bestInfo.thresholdGainScaled > 0) {
                s_dualq_rowaware_threshold_selected += 1;
            }
            if (bestInfo.ageForced) {
                s_dualq_rowaware_age_forced += 1;
            }
            if (bestInfo.maxBankToken > dualQueueFineBankTokenLimit || bestInfo.maxBGToken > dualQueueFineBGTokenLimit) {
                s_dualq_rowaware_token_throttled += 1;
            }
        } else {
            if (bestIdx > 0) {
                s_dualq_bgaware_reordered += 1;
            }
            if (bestInfo.rowHitPred > 0) {
                s_dualq_bgaware_rowhit_selected += 1;
            }
        }
        if (bestInfo.phase == 1 && penalizedFinePresent) {
            s_dualq_bgaware_fine_deferred += 1;
        }
    }
    return bestIdx;
}

bool CountAndErase(std::vector<DimBit>& vec, DimBit db) {
    for (size_t i = 0; i < vec.size(); i++) {
        if (vec[i].dim == db.dim && vec[i].bit == db.bit) {
            vec.erase(vec.begin() + i);
            return true;
        }
    }
    return false;
}

void HNSWEmbUnit::handlePDisReqLineReturn(Request& req) {
    PDisReq& pDisReq = req.anns;
    updateBGOutstandingOnReturn(req.addr, pDisReq.phase);
    auto it = inflightPDisReqCallbacks.find(pDisReq.token);
    if (it == inflightPDisReqCallbacks.end()) {
        return;
    }
    PendingPDisReqState& state = it->second;
    state.latestReturnCycle = std::max<uint64_t>(state.latestReturnCycle, m_memory_system->get_clk());
    if (state.remainingCallbacks > 0) {
        state.remainingCallbacks -= 1;
    }
    if (state.remainingCallbacks == 0) {
        PDisReq completed = state.req;
        completed.memServiceCycles = state.latestReturnCycle - state.sendCycle;
        inflightPDisReqCallbacks.erase(it);
        schedulePDisReqCompute(completed);
    }
}

HNSWEmbUnit::ComputeTaskModel HNSWEmbUnit::describeBlockCompute(DisReq& inflightDisReq, const PDisReq& pDisReq) const {
    ComputeTaskModel model;
    uint32_t dimEnd = pDisReq.curDim;

    if (inflightDisReq.dualQueueTwoPhase) {
        const uint32_t fetch_dim_step = get_dual_phase_dim_step(
            inflightDisReq.dualQueuePhase,
            nativeDataBitWidth,
            travUnit->datatype == "isFloat");
        dimEnd = std::min(pDisReq.curDim + fetch_dim_step, inflightDisReq.vDimEnd);
        model.workItems = dimEnd - pDisReq.curDim;
        if (model.workItems == 0) {
            return model;
        }
        if (inflightDisReq.dualQueuePhase == 1) {
            if (dualQueueCoarseComputeModel == "std_fp") {
                model.kind = ComputeTaskKind::StandardFpDistance;
                model.modeledOps = model.workItems * std::max<uint32_t>(1, dualQueueCoarseStdFpOpsPerDim);
                model.fmacOps = model.modeledOps;
            } else {
                model.kind = ComputeTaskKind::DualQueueCoarseLowerBound;
                const uint32_t coarse_ops_per_dim =
                    (travUnit && travUnit->spacetype == "L2")
                        ? std::max<uint32_t>(2U, dualQueueCoarseOpsPerDim)
                        : std::max<uint32_t>(1U, dualQueueCoarseOpsPerDim);
                model.modeledOps = model.workItems * coarse_ops_per_dim;
            }
            return model;
        }
        model.kind = ComputeTaskKind::DualQueueFineApproxDistance;
        const uint32_t fine_ops_per_dim =
            (travUnit && travUnit->spacetype == "L2")
                ? std::max<uint32_t>(2U, dualQueueFineOpsPerDim)
                : std::max<uint32_t>(1U, dualQueueFineOpsPerDim);
        model.modeledOps = model.workItems * fine_ops_per_dim;
        model.fmacOps = model.modeledOps;
        return model;
    }

    const uint32_t req_dim_step = std::max<uint32_t>(1U, pDisReq.dimStep);
    dimEnd = std::min(pDisReq.curDim + req_dim_step, inflightDisReq.vDimEnd);
    model.workItems = dimEnd - pDisReq.curDim;
    if (model.workItems == 0) {
        return model;
    }
    model.kind = ComputeTaskKind::StandardFpDistance;
    model.modeledOps = model.workItems * kStandardFpDistanceOpsPerDim;
    model.fmacOps = model.modeledOps;
    return model;
}

void HNSWEmbUnit::tryStartNextComputeTask(uint64_t curCycle, size_t poolIndex) {
    (void)curCycle;
    (void)poolIndex;
}

void HNSWEmbUnit::tryStartNextComputeTasks(uint64_t curCycle) {
    (void)curCycle;
}

bool HNSWEmbUnit::hasPendingComputeWork() const {
    return !pendingComputeCompletions.empty();
}

bool HNSWEmbUnit::canLookupVectorCache(const DisReq& disReq) const {
    if (!vectorCacheEnabled) {
        return false;
    }
    if (vectorCacheMode == "phase2_tail_only") {
        return disReq.dualQueueTwoPhase && disReq.dualQueuePhase == 2;
    }
    return !disReq.dualQueueTwoPhase || disReq.dualQueuePhase == 2;
}

bool HNSWEmbUnit::canFillVectorCache(const DisReq& disReq) const {
    if (!vectorCacheEnabled) {
        return false;
    }
    if (vectorCacheMode == "phase2_tail_only") {
        return disReq.dualQueueTwoPhase && disReq.dualQueuePhase == 2;
    }
    return !disReq.dualQueueTwoPhase || disReq.dualQueuePhase == 2;
}

void HNSWEmbUnit::schedulePDisReqCompute(PDisReq& pDisReq) {
    if (!inflightDisReqs.count(pDisReq.disreqId)) {
        assert(earlyExitEnable || (travUnit && (travUnit->isDualQueueLowerBoundETEnabled() || travUnit->isMFNNSEnabled())));
        return;
    }
    DisReq& inflightDisReq = inflightDisReqs[pDisReq.disreqId];
    ComputeTaskModel compute_model = describeBlockCompute(inflightDisReq, pDisReq);
    if (zeroMemFinalizeSeparatePoolEnable &&
        pDisReq.memServiceCycles == 0 &&
        compute_model.kind == ComputeTaskKind::DualQueueFineApproxDistance) {
        compute_model.kind = ComputeTaskKind::ZeroMemFinalize;
        compute_model.workItems = std::max<uint32_t>(1, zeroMemFinalizeWorkItemsPerTask);
        compute_model.modeledOps = std::max<uint32_t>(1, zeroMemFinalizeModeledOpsPerTask);
        compute_model.fmacOps = 0;
    }
    PendingComputeTask task;
    task.taskId = nextComputeTaskId++;
    task.req = pDisReq;
    task.kind = compute_model.kind;
    task.workItems = compute_model.workItems;
    task.modeledOps = compute_model.modeledOps;
    task.fmacOps = compute_model.fmacOps;
    task.readyCycle = m_memory_system->get_clk();
    task.rawComputeCycles = calculateComputeLatency(compute_model);
    task.hiddenComputeCycles = std::min<uint64_t>(task.rawComputeCycles, pDisReq.memServiceCycles);
    task.serviceExposedComputeCycles = task.rawComputeCycles - task.hiddenComputeCycles;
    task.exposedComputeCycles = task.serviceExposedComputeCycles;
    if (task.serviceExposedComputeCycles == 0) {
        task.startCycle = task.readyCycle;
        task.completionCycle = task.readyCycle;
        handleCompletedPDisReq(task);
        return;
    }
    task.poolIndex = getComputePoolIndex(task.kind);
    task.resourceUnit = selectComputeResourceUnit(inflightDisReq, task);
    auto& resourceReady = computeResourceReadyCycles[task.poolIndex];
    if (resourceReady.empty()) {
        resourceReady.assign(1, 0);
    }
    if (task.resourceUnit >= resourceReady.size()) {
        task.resourceUnit = 0;
    }
    task.startCycle = std::max<uint64_t>(task.readyCycle, resourceReady[task.resourceUnit]);
    task.queueCycles = task.startCycle - task.readyCycle;
    task.exposedComputeCycles = task.queueCycles + task.serviceExposedComputeCycles;
    task.completionCycle = task.startCycle + task.serviceExposedComputeCycles;
    resourceReady[task.resourceUnit] = task.completionCycle;
    if (dualQueueCrossLevelNMPEnable && inflightDisReq.dualQueueTwoPhase) {
        if (task.kind == ComputeTaskKind::DualQueueCoarseLowerBound && dualQueueCoarseNMPLevel == "bankgroup") {
            s_dualq_crosslevel_coarse_bg_tasks += 1;
        } else if (task.kind == ComputeTaskKind::DualQueueFineApproxDistance && dualQueueFineNMPLevel == "bank") {
            s_dualq_crosslevel_fine_bank_tasks += 1;
        } else if (task.kind == ComputeTaskKind::DualQueueFineApproxDistance && dualQueueFineNMPLevel == "bank_subarray") {
            s_dualq_crosslevel_fine_subarray_tasks += 1;
        }
    }
    pendingComputeCompletions.push(task);
    m_logger->info(
        "[{}] cycle {} reserve compute task kind {} pool {} unit {} disreqId {} annsId {} candId {} ready {} start {} queue {} service {} ops {} work {} complete@{}",
        name.c_str(),
        task.readyCycle,
        getComputeTaskKindName(task.kind),
        task.poolIndex,
        task.resourceUnit,
        task.req.disreqId,
        inflightDisReq.annsId,
        inflightDisReq.candId,
        task.readyCycle,
        task.startCycle,
        task.queueCycles,
        task.serviceExposedComputeCycles,
        task.modeledOps,
        task.workItems,
        task.completionCycle);
}

void HNSWEmbUnit::handleCompletedPDisReq(const PendingComputeTask& task) {
    PDisReq pDisReq = task.req;
    if (!inflightDisReqs.count(pDisReq.disreqId)) {
        assert(earlyExitEnable || (travUnit && (travUnit->isDualQueueLowerBoundETEnabled() || travUnit->isMFNNSEnabled())));
        return;
    }
    DisReq& inflightDisReq = inflightDisReqs[pDisReq.disreqId];
    uint32_t annsId = inflightDisReq.annsId;
    PointId candId = inflightDisReq.candId;
    bool count = CountAndErase(inflightDisReq.pends, DimBit{pDisReq.curDim, pDisReq.curBit});
    assert(count);

    inflightDisReq.memServiceCyclesAccum += pDisReq.memServiceCycles;
    s_pdis_mem_service_cycles += pDisReq.memServiceCycles;
    if (pDisReq.phase == 1) {
        s_dualq_phase1_pdisreq += 1;
        s_dualq_phase1_pdis_mem_service_cycles += pDisReq.memServiceCycles;
    } else if (pDisReq.phase == 2) {
        s_dualq_phase2_pdisreq += 1;
        s_dualq_phase2_pdis_mem_service_cycles += pDisReq.memServiceCycles;
    }
    if (pDisReq.memServiceCycles == 0) {
        s_num_zero_mem_pdisreq += 1;
    }
    if (task.kind == ComputeTaskKind::ZeroMemFinalize) {
        s_num_zero_mem_finalize_tasks += 1;
        s_zero_mem_finalize_cycles += task.rawComputeCycles;
    }

    inflightDisReq.fmacOpsAccum += task.fmacOps;
    inflightDisReq.rawComputeCyclesAccum += task.rawComputeCycles;
    inflightDisReq.hiddenComputeCyclesAccum += task.hiddenComputeCycles;
    inflightDisReq.computeQueueCyclesAccum += task.queueCycles;
    inflightDisReq.exposedComputeCyclesAccum += task.exposedComputeCycles;
    s_fmac_raw_cycles += task.rawComputeCycles;
    s_fmac_hidden_cycles += task.hiddenComputeCycles;
    s_fmac_queue_cycles += task.queueCycles;
    s_fmac_compute_cycles += task.exposedComputeCycles;
    if (pDisReq.phase == 1) {
        s_dualq_phase1_exposed_compute_cycles += task.exposedComputeCycles;
        s_dualq_phase1_fmac_ops += task.fmacOps;
        s_dualq_phase1_modeled_ops += task.modeledOps;
    } else if (pDisReq.phase == 2) {
        s_dualq_phase2_exposed_compute_cycles += task.exposedComputeCycles;
        s_dualq_phase2_fmac_ops += task.fmacOps;
        s_dualq_phase2_modeled_ops += task.modeledOps;
    }

    auto fillCompletedCachesIfEligible = [&](const DisReq& completedReq, uint32_t phase, bool fullPhaseDone) {
        if (!fullPhaseDone || !completedReq.pends.empty()) {
            return;
        }
        if (phase == 1 &&
            phase1SignExpCacheEnabled &&
            completedReq.dualQueueTwoPhase) {
            m_phase1_signexp_cache.fill(completedReq.candId);
        }
        if (canFillVectorCache(completedReq)) {
            m_vector_cache.fill(completedReq.candId);
        }
    };

    if (inflightDisReq.dualQueueTwoPhase) {
        const uint32_t phase = pDisReq.phase ? pDisReq.phase : inflightDisReq.dualQueuePhase;
        const uint32_t fetch_dim_step = get_dual_phase_dim_step(phase, nativeDataBitWidth, travUnit->datatype == "isFloat");
        const uint32_t dimEnd = std::min(pDisReq.curDim + fetch_dim_step, inflightDisReq.vDimEnd);
        const uint32_t phase_fetch_bits = getDualQueueModeledFetchBits(phase);
        const uint64_t modeled_bits = static_cast<uint64_t>(dimEnd - pDisReq.curDim) * phase_fetch_bits;
        s_modeled_read_bits += modeled_bits;
        if (phase == 1) {
            s_dualq_phase1_modeled_read_bits += modeled_bits;
        } else if (phase == 2) {
            s_dualq_phase2_modeled_read_bits += modeled_bits;
        }
        Type* candData = travUnit->getEmbData(candId);

        if (phase == 1) {
            for (uint32_t dim = pDisReq.curDim; dim < dimEnd; dim++) {
                uint16_t v_fp16_bits = fp32_to_fp16_bits_dual_mem(candData[dim]);
                const Type coarseScore =
                    (travUnit && travUnit->spacetype == "L2")
                        ? compute_dualq_coarse_score_contribution(candData[dim], pDisReq.query[dim], v_fp16_bits)
                        : compute_l1_lower_bound_dual_mem(pDisReq.query[dim], v_fp16_bits);
                const Type prevScore = inflightDisReq.curDis[dim];
                inflightDisReq.curDis[dim] = coarseScore;
                inflightDisReq.dualQueueCoarseScoreAccum += coarseScore - prevScore;
            }
        } else {
            for (uint32_t dim = pDisReq.curDim; dim < dimEnd; dim++) {
                Type cand = candData[dim];
                Type query = pDisReq.query[dim];
                recordMultiplications((travUnit && travUnit->spacetype == "L2") ? 2U : 1U, &inflightDisReq);
                const Type prevScore = inflightDisReq.curDis[dim];
                const Type exactScore = computeDualQueueExactScoreContribution(cand, query, &inflightDisReq);
                inflightDisReq.curDis[dim] = exactScore;
                inflightDisReq.dualQueueScoreAccum += exactScore - prevScore;
            }
        }

        bool full = isLastDimBit(inflightDisReq);
        const Type currentDualQueueDistance =
            (phase == 1)
                ? computeDualQueueDistanceFromScore(
                      inflightDisReq.dualQueueCoarseScoreAccum,
                      inflightDisReq.querySquaredNorm)
                : computeDualQueueDistanceFromScore(
                      inflightDisReq.dualQueueScoreAccum,
                      inflightDisReq.querySquaredNorm);
        if (full && inflightDisReq.pends.empty()) {
            uint64_t curCycle = task.completionCycle;
            fillCompletedCachesIfEligible(inflightDisReq, phase, full);

            if (phase == 1) {
                inflightDisReq.dualQueueScoreAccum = inflightDisReq.dualQueueCoarseScoreAccum;
                inflightDisReq.dualQueueLowerBound = computeDualQueueDistanceFromScore(
                    inflightDisReq.dualQueueCoarseScoreAccum,
                    inflightDisReq.querySquaredNorm);
                if (inflightDisReq.dualQueueLowerBound > inflightDisReq.dualQueuePhase1Upperbound) {
                    s_num_early_exit += 1;
                    s_num_similarity_mul += inflightDisReq.mulCount;
                    s_fmac_ops += inflightDisReq.fmacOpsAccum;
                    releaseDualQueueCoarseAdmission(inflightDisReq);

                    Type rejectDistance = std::nextafter(
                        inflightDisReq.upperbound, std::numeric_limits<Type>::infinity());
                    DisResp disResp(annsId, candId, false, rejectDistance, inflightDisReq.disreqId, curCycle);
                    disResp.reqSendCycle = inflightDisReq.sendCycle;
                    disResp.reqRecvCycle = inflightDisReq.recvCycle;
                    disResp.num_access = inflightDisReq.num_access;
                    disResp.memServiceCycles = inflightDisReq.memServiceCyclesAccum;
                    disResp.rawComputeCycles = inflightDisReq.rawComputeCyclesAccum;
                    disResp.hiddenComputeCycles = inflightDisReq.hiddenComputeCyclesAccum;
                    disResp.computeQueueCycles = inflightDisReq.computeQueueCyclesAccum;
                    disResp.exposedComputeCycles = inflightDisReq.exposedComputeCyclesAccum;
                    disResp.dualQueuePruned = true;
                    disResp.dualQueueLowerBoundValid = true;
                    disResp.dualQueueLowerBound = inflightDisReq.dualQueueLowerBound;

                    inflightDisReqs.erase(pDisReq.disreqId);
                    assert(!finishDisReqs.count(pDisReq.disreqId));
                    finishDisReqs[pDisReq.disreqId] = disResp;
                } else {
                    inflightDisReq.dualQueuePhase = 2;
                    if (isDualQueueStagedAdmissionEnabled()) {
                        if (!tryActivateDualQueueFineAdmission(inflightDisReq, curCycle)) {
                            deferDualQueueFineAdmission(inflightDisReq, curCycle);
                        } else {
                            startDualQueueFinePhase(inflightDisReq);
                        }
                    } else {
                        startDualQueueFinePhase(inflightDisReq);
                    }
                }
            } else {
                if (currentDualQueueDistance > inflightDisReq.upperbound) {
                    s_num_early_exit += 1;
                    s_dualq_phase2_early_exit += 1;
                    s_num_similarity_mul += inflightDisReq.mulCount;
                    s_fmac_ops += inflightDisReq.fmacOpsAccum;

                    DisResp disResp(annsId, candId, false, currentDualQueueDistance, inflightDisReq.disreqId, curCycle);
                    disResp.reqSendCycle = inflightDisReq.sendCycle;
                    disResp.reqRecvCycle = inflightDisReq.recvCycle;
                    disResp.num_access = inflightDisReq.num_access;
                    disResp.memServiceCycles = inflightDisReq.memServiceCyclesAccum;
                    disResp.rawComputeCycles = inflightDisReq.rawComputeCyclesAccum;
                    disResp.hiddenComputeCycles = inflightDisReq.hiddenComputeCyclesAccum;
                    disResp.computeQueueCycles = inflightDisReq.computeQueueCyclesAccum;
                    disResp.exposedComputeCycles = inflightDisReq.exposedComputeCyclesAccum;
                    disResp.dualQueuePruned = true;
                    disResp.dualQueueLowerBoundValid = true;
                    disResp.dualQueueLowerBound = inflightDisReq.dualQueueLowerBound;

                    releaseDualQueueFineAdmission(inflightDisReq);
                    inflightDisReqs.erase(pDisReq.disreqId);
                    assert(!finishDisReqs.count(pDisReq.disreqId));
                    finishDisReqs[pDisReq.disreqId] = disResp;
                    tryPromoteDeferredDualQueueFineAdmissions(curCycle);
                } else {
                    const Type pDistance = currentDualQueueDistance;
                    s_num_similarity_mul += inflightDisReq.mulCount;
                    s_fmac_ops += inflightDisReq.fmacOpsAccum;

                    DisResp disResp(annsId, candId, true, pDistance, inflightDisReq.disreqId, curCycle);
                    disResp.reqSendCycle = inflightDisReq.sendCycle;
                    disResp.reqRecvCycle = inflightDisReq.recvCycle;
                    disResp.num_access = inflightDisReq.num_access;
                    disResp.memServiceCycles = inflightDisReq.memServiceCyclesAccum;
                    disResp.rawComputeCycles = inflightDisReq.rawComputeCyclesAccum;
                    disResp.hiddenComputeCycles = inflightDisReq.hiddenComputeCyclesAccum;
                    disResp.computeQueueCycles = inflightDisReq.computeQueueCyclesAccum;
                    disResp.exposedComputeCycles = inflightDisReq.exposedComputeCyclesAccum;
                    disResp.dualQueuePruned = false;
                    disResp.dualQueueLowerBoundValid = true;
                    disResp.dualQueueLowerBound = inflightDisReq.dualQueueLowerBound;

                    releaseDualQueueFineAdmission(inflightDisReq);
                    inflightDisReqs.erase(pDisReq.disreqId);
                    assert(!finishDisReqs.count(pDisReq.disreqId));
                    finishDisReqs[pDisReq.disreqId] = disResp;
                    tryPromoteDeferredDualQueueFineAdmissions(curCycle);
                }
            }
        } else if (phase == 2 && currentDualQueueDistance > inflightDisReq.upperbound) {
            const uint64_t curCycle = task.completionCycle;
            s_num_early_exit += 1;
            s_dualq_phase2_early_exit += 1;
            s_num_similarity_mul += inflightDisReq.mulCount;
            s_fmac_ops += inflightDisReq.fmacOpsAccum;

            DisResp disResp(annsId, candId, false, currentDualQueueDistance, inflightDisReq.disreqId, curCycle);
            disResp.reqSendCycle = inflightDisReq.sendCycle;
            disResp.reqRecvCycle = inflightDisReq.recvCycle;
            disResp.num_access = inflightDisReq.num_access;
            disResp.memServiceCycles = inflightDisReq.memServiceCyclesAccum;
            disResp.rawComputeCycles = inflightDisReq.rawComputeCyclesAccum;
            disResp.hiddenComputeCycles = inflightDisReq.hiddenComputeCyclesAccum;
            disResp.computeQueueCycles = inflightDisReq.computeQueueCyclesAccum;
            disResp.exposedComputeCycles = inflightDisReq.exposedComputeCyclesAccum;
            disResp.dualQueuePruned = true;
            disResp.dualQueueLowerBoundValid = true;
            disResp.dualQueueLowerBound = inflightDisReq.dualQueueLowerBound;

            releaseDualQueueFineAdmission(inflightDisReq);
            inflightDisReqs.erase(pDisReq.disreqId);
            assert(!finishDisReqs.count(pDisReq.disreqId));
            finishDisReqs[pDisReq.disreqId] = disResp;
            tryPromoteDeferredDualQueueFineAdmissions(curCycle);
        } else if (!full) {
            inflightDisReq.pends.push_back(inflightDisReq.last);
            trySendPDisReq(pDisReq.disreqId, inflightDisReq.last.dim, inflightDisReq.last.bit, 0);
            getNextDimBit(inflightDisReq);
        }
        return;
    }

    bool accept = false, reject = false;
    bool earlyExitTriggered = false;
    int ncurstep_ = std::min(inflightDisReq.last.ncurstep, static_cast<uint32_t>(31));
    int bitend = travUnit->bitStep_array[ncurstep_] + pDisReq.curBit;
    bool full = isLastDimBit(inflightDisReq);// if this is outlier vector, bitend = 32 still not full
    Type pDistance = mergeDistance(inflightDisReq, pDisReq);
    const bool mfnnsLinearET =
        travUnit &&
        travUnit->isMFNNSEnabled() &&
        !travUnit->isDualQueueLowerBoundETEnabled() &&
        !inflightDisReq.dualQueueTwoPhase;
    if (full && inflightDisReq.pends.empty()) {
        fillCompletedCachesIfEligible(inflightDisReq, 0, full);
    }
    if (mfnnsLinearET) {
        if (!full && pDistance >= inflightDisReq.upperbound) {
            s_num_early_exit += 1;
            s_mfnns_linear_early_exit += 1;
            reject = true;
            earlyExitTriggered = true;
        } else if (full && inflightDisReq.pends.empty()) {
            accept = pDistance < inflightDisReq.upperbound;
            reject = pDistance >= inflightDisReq.upperbound;
        }
    } else if(bitend > (travUnit->datatype == "isFloat")*4 + (travUnit->datatype == "isInt")*1 &&(travUnit->spacetype == "L2" ||(travUnit->spacetype == "IP" && pDisReq.curBit!= preprocess(candId)||(travUnit->spacetype == "IP" && !earlyExitEnable)))){
        if (!full && earlyExitEnable && pDistance >= inflightDisReq.upperbound) {
            s_num_early_exit += 1;
            reject = true;
            earlyExitTriggered = true;
        } else if (full && inflightDisReq.pends.empty()) {
            accept = pDistance < inflightDisReq.upperbound;
            reject = pDistance >= inflightDisReq.upperbound;
        }
    }else{

    }
    int num_extra_send = 0;
    Type* candFormatData = (Type*)travUnit->hnsw->getFormatByInternalId(candId);
    int col_c0=0;
    if(travUnit->hnsw->outlier_list[candId] && travUnit -> outlier_burden){
        col_c0 = getOutlierColC0(candId);

            int num_Ol = 0;
            for(int dim = 0; dim < nDim; dim++){
                Type candFormatData1 = *(candFormatData+dim);
            union {
                Type input;
                unsigned int output;
            } data;
            data.input = candFormatData1;
            std::bitset<sizeof(Type) * 8> binaryRepresentation(data.output);
                if(binaryRepresentation[0]==1){
                    num_Ol ++;
                }
            }
            int bitend = travUnit->bitStep_array[inflightDisReq.last.ncurstep] + pDisReq.curBit;
            if(pDisReq.curBit <= nBit - 1 && bitend >nBit - 1){
                num_extra_send += std::ceil((col_c0 * num_Ol + nDim)*1.0/512.0);
            }

    }

    m_logger->info("[{}] handlePDisReq annsId {} candId {} curDim {} curBit {}num_extra_send {}  acc {} rej {} pDistance {} earlyExitTriggered {} computeQ {} computeExposed {}", name.c_str(), annsId, candId, pDisReq.curDim, pDisReq.curBit, num_extra_send, accept, reject, pDistance, earlyExitTriggered, task.queueCycles, task.exposedComputeCycles);
    uint64_t curCycle = task.completionCycle;
    const uint32_t req_dim_step = std::max<uint32_t>(1U, pDisReq.dimStep);
    uint32_t dimEnd = std::min(pDisReq.curDim + req_dim_step, inflightDisReq.vDimEnd);
    uint32_t num_dims_computed = dimEnd - pDisReq.curDim;
    if (pDisReq.modeledReadBits > 0) {
        s_modeled_read_bits += pDisReq.modeledReadBits;
    } else {
        const uint32_t req_bit_step = std::max<uint32_t>(1U, pDisReq.bitStep);
        uint32_t modeled_bits_per_dim = earlyExitEnable ? std::min(req_bit_step, nBit) : modeledReadBitWidth;
        s_modeled_read_bits += static_cast<uint64_t>(num_dims_computed) * modeled_bits_per_dim;
    }

    if (accept || reject) {
        s_num_similarity_mul += inflightDisReq.mulCount;
        s_fmac_ops += inflightDisReq.fmacOpsAccum;

        // Track ANSMET-path ops (non-MFNNS-linear, non-dual-queue)
        if (!mfnnsLinearET && !inflightDisReq.dualQueueTwoPhase) {
            s_ansmet_total_fmac_ops += inflightDisReq.fmacOpsAccum;
            if (earlyExitTriggered) {
                s_ansmet_et_fmac_ops += inflightDisReq.fmacOpsAccum;
            }
        }

        DisResp disResp(annsId, candId, accept, pDistance, inflightDisReq.disreqId, curCycle);
        disResp.reqSendCycle = inflightDisReqs[pDisReq.disreqId].sendCycle;
        disResp.reqRecvCycle = inflightDisReqs[pDisReq.disreqId].recvCycle;
        disResp.num_access = inflightDisReqs[pDisReq.disreqId].num_access;
        disResp.memServiceCycles = inflightDisReqs[pDisReq.disreqId].memServiceCyclesAccum;
        disResp.rawComputeCycles = inflightDisReqs[pDisReq.disreqId].rawComputeCyclesAccum;
        disResp.hiddenComputeCycles = inflightDisReqs[pDisReq.disreqId].hiddenComputeCyclesAccum;
        disResp.computeQueueCycles = inflightDisReqs[pDisReq.disreqId].computeQueueCyclesAccum;
        disResp.exposedComputeCycles = inflightDisReqs[pDisReq.disreqId].exposedComputeCyclesAccum;

        inflightDisReqs.erase(pDisReq.disreqId);
        assert(!finishDisReqs.count(pDisReq.disreqId));
        finishDisReqs[pDisReq.disreqId] = disResp;

        m_logger->info("[{}] cycle {} finish critical-path FMAC DisReq annsId {} candId {} {} (dims={}, ops={}, exposed_delay={})",
                      name.c_str(), curCycle, annsId, candId, accept ? "accept" : "reject",
                      num_dims_computed, task.fmacOps, task.exposedComputeCycles);
    } else if (!full) {
        inflightDisReq.pends.push_back(inflightDisReq.last);
        trySendPDisReq(pDisReq.disreqId, inflightDisReq.last.dim, inflightDisReq.last.bit,num_extra_send);
        getNextDimBit(inflightDisReq);
    }
    if (finishDisReqs.size() >= 16) {
        m_logger->warn("[{}] cycle {} finishDisReqs.size() >= 16", name.c_str(), curCycle);
    }
}

uint32_t HNSWEmbUnit::sendResultProbe() {

    if (finishDisReqs.empty()) return 0;
    uint64_t nFinishDisReq = finishDisReqs.size();
    for (auto it = finishDisReqs.begin(); it != finishDisReqs.end(); ++it) {
        DisResp& disResp = it->second;
        disResp.respRecvCycle = m_memory_system->get_clk();
        travUnit->handleDisReq(disResp);
    }
    finishDisReqs.clear();
    return nFinishDisReq;
}

bool HNSWEmbUnit::sendPDisReq(PDisReq &pDisReq) {
    if (!inflightDisReqs.count(pDisReq.disreqId)) {
        assert(earlyExitEnable || (travUnit && (travUnit->isDualQueueLowerBoundETEnabled() || travUnit->isMFNNSEnabled())));
        return true;
    }
    uint32_t annsId = inflightDisReqs[pDisReq.disreqId].annsId;
    PointId candId = inflightDisReqs[pDisReq.disreqId].candId;
    //Validates inflightDisReqs still holds the referenced disreqId;
    //otherwise (early exit already triggered) it returns true to drop the orphan request.
    DisReq& disReq = inflightDisReqs[pDisReq.disreqId];
    pDisReq.annsId = disReq.annsId;
    pDisReq.candId = disReq.candId;
    pDisReq.phase = disReq.dualQueueTwoPhase ? disReq.dualQueuePhase : 0;
    pDisReq.bitStep = getLinearStageBitStep(disReq, pDisReq.curBit);
    pDisReq.dimStep = getLinearRequestDimStep(disReq, pDisReq.curBit);
    std::vector<Addr_t> request_addrs = getRequestAddrs(disReq, pDisReq.curDim, pDisReq.curBit);
    if (usesAnsmetFp16FullDimFetch(disReq) && !request_addrs.empty()) {
        const uint32_t dim_end = std::min(pDisReq.curDim + pDisReq.dimStep, disReq.vDimEnd);
        pDisReq.modeledReadBits = static_cast<uint64_t>(dim_end - pDisReq.curDim) * static_cast<uint64_t>(nativeDataBitWidth);
    } else {
        pDisReq.modeledReadBits = 0;
    }
    assignPredictedComputeTarget(pDisReq, request_addrs);
    if (request_addrs.empty()) {
        pDisReq.issueCycle = m_memory_system->get_clk();
        pDisReq.memServiceCycles = 0;
        pendCacheHitPDisReqs.push(pDisReq);
        return true;
    }
    inflightDisReqs[pDisReq.disreqId].num_access++;
    Addr_t addr = request_addrs.front();

    // Object-level vector cache lookup is phase-aware.
    const bool vector_cache_eligible = canLookupVectorCache(disReq);
    bool vector_cache_hit = false;
    if (vector_cache_eligible) {
        vector_cache_hit = m_vector_cache.contains(candId);
    }
    const bool phase1_signexp_cache_eligible =
        phase1SignExpCacheEnabled &&
        disReq.dualQueueTwoPhase &&
        disReq.dualQueuePhase == 1;
    bool phase1_signexp_cache_hit = false;
    if (phase1_signexp_cache_eligible) {
        phase1_signexp_cache_hit = m_phase1_signexp_cache.contains(candId);
    }
    const bool any_vector_cache_hit = vector_cache_hit || phase1_signexp_cache_hit;

    bool ok = true;
    if (any_vector_cache_hit) {
        pDisReq.modeledReadBits = 0;
        // Cache hit: add to async queue to avoid recursive deadlock
        if (phase1_signexp_cache_hit) {
            m_logger->info("[{}] PHASE1 SIGNEXP CACHE HIT candId {} annsId {} curDim {} curBit {}",
                           name.c_str(), candId, annsId, pDisReq.curDim, pDisReq.curBit);
        } else {
            m_logger->info("[{}] VECTOR CACHE HIT candId {} annsId {} curDim {} curBit {}",
                           name.c_str(), candId, annsId, pDisReq.curDim, pDisReq.curBit);
        }
        pDisReq.issueCycle = m_memory_system->get_clk();
        pDisReq.memServiceCycles = 0;
        pendCacheHitPDisReqs.push(pDisReq);
    } else {
        const uint64_t issue_cycle = m_memory_system->get_clk();
        pDisReq.token = nextPDisReqToken++;
        pDisReq.issueCycle = issue_cycle;
        pDisReq.callbackLineCount = static_cast<uint32_t>(request_addrs.size());
        inflightPDisReqCallbacks[pDisReq.token] = PendingPDisReqState{pDisReq, static_cast<uint32_t>(request_addrs.size()), issue_cycle, issue_cycle};
        for (size_t i = 0; i < request_addrs.size(); i++) {
            Request req(request_addrs[i],
                        Request::Type::Read,
                        pDisReq.embUnitId,
                        pdis_callback,
                        pDisReq);
            ok = m_memory_system->send(req) && ok;
        }
        if (ok) {
            if (disReq.dualQueueTwoPhase) {
                updateBGOutstandingOnIssue(request_addrs, pDisReq.phase);
            }
            m_logger->info("[{}] sendPDisReq addr {} annsId {} candId {} curDim {} curBit {} lines {}",
                             name.c_str(), addr, annsId, candId, pDisReq.curDim, pDisReq.curBit, request_addrs.size());
        }
    }

    if (ok) {
        bool cache_hit = g_cache.access(addr);
        if (cache_hit) {
            travUnit->s_cache_hit += 1;
        }

        s_num_pdisreq += 1;
        travUnit->s_total_pdisreq += 1;
        travUnit->s_num_total_pdisreq[m_id] += 1;
        uint32_t layer = inflightDisReqs[pDisReq.disreqId].layer;
        if (layer > 0) {
            travUnit->unlimited_top_layer_addr.insert(addr);
            travUnit->s_unlimited_top_layer_access += 1;
            travUnit->s_top_layer_access_per_embunit[m_id] += 1;
        }
        if (layer > 0 && travUnit->top_layer_addr.size() < travUnit->top_layer_size_limit) {
            // top-layer vector data
            travUnit->top_layer_addr.insert(addr);
        }
        if (travUnit->top_layer_addr.count(addr)) {
            travUnit->s_top_layer_access += 1;
        }
    }

    // Handle extra sends only if not a vector cache hit
    if (!any_vector_cache_hit) {
        for(int i = 0; i < pDisReq.num_extra_send; i++){
            Request req(addr,
                    Request::Type::Read,
                    pDisReq.embUnitId,
                    pdis_callback,
                    pDisReq,
                    true);
            ok = (m_memory_system->send(req))*ok;
        }
    }
    if (!ok && pDisReq.token != 0) {
        inflightPDisReqCallbacks.erase(pDisReq.token);
    }
    return ok;
}

void HNSWEmbUnit::trySendPDisReq(uint64_t disreqId, uint32_t curDim, uint32_t curBit,int num_extra_send) {
    assert(inflightDisReqs.count(disreqId));
    PDisReq pDisReq;
    pDisReq.embUnitId = m_id;
    pDisReq.disreqId = disreqId;
    pDisReq.query = inflightDisReqs[disreqId].query;
    pDisReq.curDim = curDim;
    pDisReq.curBit = curBit;
    pDisReq.pDistance = 0;
    pDisReq.num_extra_send = num_extra_send;
    pDisReq.dimStep = 0;
    pDisReq.bitStep = 0;
    pDisReq.modeledReadBits = 0;
    pDisReq.phase = inflightDisReqs[disreqId].dualQueueTwoPhase ? inflightDisReqs[disreqId].dualQueuePhase : 0;
    pDisReq.enqueueCycle = m_memory_system->get_clk();
    pendPDisReqs.push_back(pDisReq);
}

bool HNSWEmbUnit::sendDisReq(DisReq& disreq) {
    const bool stagedDualQueue = disreq.dualQueueTwoPhase && isDualQueueStagedAdmissionEnabled();
    if (stagedDualQueue) {
        if (!canAdmitDualQueueCoarseDisReq()) {
            if (dualQueueActiveCoarseDisReqs >= dualQueueCoarseAdmissionWindow) {
                s_dualq_stage_admission_coarse_blocked += 1;
            }
            if (getDualQueueFineAdmissionOccupancy() >= dualQueueFineAdmissionWindow) {
                s_dualq_stage_admission_fine_blocked += 1;
            }
            return false;
        }
    } else if (inflightDisReqs.size() == nParallelDisReq) {
        return false;
    }
    uint64_t disreqId = disreq.disreqId;
    assert(!inflightDisReqs.count(disreqId));
    m_logger->info("[{}] receive DisReq annsId {}", name.c_str(), disreq.annsId);
    s_num_disreq += 1;
    inflightDisReqs[disreqId] = disreq;
    inflightDisReqs[disreqId].recvCycle = m_memory_system->get_clk();
    inflightDisReqs[disreqId].fmacOpsAccum = 0;  // Initialize FMAC ops accumulator
    inflightDisReqs[disreqId].pDistanceAccum = 0;
    inflightDisReqs[disreqId].dualQueueScoreAccum = 0;
    inflightDisReqs[disreqId].dualQueueExactDistanceAccum = 0;
    inflightDisReqs[disreqId].dualQueueCoarseScoreAccum = 0;
    inflightDisReqs[disreqId].dualQueueLowerBound = 0;
    inflightDisReqs[disreqId].ansmetLinearChargedEndBit = -1;
    inflightDisReqs[disreqId].dualQueueAdmissionStage = 0;
    inflightDisReqs[disreqId].dualQueueWaitingFineAdmission = false;
    inflightDisReqs[disreqId].dualQueueFineReadyCycle = 0;
    PointId candId = inflightDisReqs[disreqId].candId;
    if (inflightDisReqs[disreqId].dualQueueTwoPhase) {
        markDualQueueCoarseAdmitted(inflightDisReqs[disreqId]);
        inflightDisReqs[disreqId].dualQueuePhase = 1;
        inflightDisReqs[disreqId].last = DimBit{inflightDisReqs[disreqId].vDimBase, 0, 0, 0};
    } else {
        int n_bitstep_begin = (travUnit && travUnit->isMFNNSEnabled()) ? 0 : preprocess(candId);
        inflightDisReqs[disreqId].last.bit = n_bitstep_begin;
        inflightDisReqs[disreqId].last.ncurstep = 0;
    }
    for (uint32_t i = 0; i < nParallelPDisReq; i++) {
        inflightDisReqs[disreqId].pends.push_back(inflightDisReqs[disreqId].last);
        trySendPDisReq(disreqId, inflightDisReqs[disreqId].last.dim, inflightDisReqs[disreqId].last.bit,0);
        getNextDimBit(inflightDisReqs[disreqId]);
        if (isLastDimBit(inflightDisReqs[disreqId])) break;
    }
    return true;
}

int HNSWEmbUnit::preprocess( PointId candId){
    int temp;
    int isOl = (travUnit->hnsw->outlier_list[candId]>0);
    temp = 0*(travUnit->datatype == "isFloat") + 24*(travUnit->datatype == "isInt") + travUnit->hnsw->col_cut - 1*(isOl);// - 1*( travUnit->dataunsigned == false && travUnit->datatype == "isInt");//temp
    return temp;
}

void HNSWEmbUnit::setup(uint32_t id, HNSWTraversalUnit* frontend, MultiGenericDRAMSystem* memory_system) {
    m_memory_system = memory_system;
    travUnit = frontend;
    m_id = id;
    name = "HNSWEmbUnit-" + std::to_string(m_id);
    assert(m_id < memory_system->get_num_memory());

    m_logger = Logging::create_logger(name);
    enableLogging = param<bool>("enableLogging").desc("enableLogging").default_val(false);
    m_logger->info("setup {}", name.c_str());
    if (!enableLogging) m_logger->set_level(spdlog::level::off);
    const uint32_t default_data_bit_width = 32 * (travUnit->datatype == "isFloat") + 8 * (travUnit->datatype == "isInt");
    nativeDataBitWidth = configuredVectorDataBitWidth > 0 ? configuredVectorDataBitWidth : default_data_bit_width;
    if (travUnit->datatype == "isFloat" && nativeDataBitWidth != 16 && nativeDataBitWidth != 32) {
        print("[Config] vectorDataBitWidth=%u invalid for datatype=isFloat. Fallback to %u.", nativeDataBitWidth, default_data_bit_width);
        nativeDataBitWidth = default_data_bit_width;
    }
    if (travUnit->datatype == "isInt" && nativeDataBitWidth != 8 && nativeDataBitWidth != 16 && nativeDataBitWidth != 32) {
        print("[Config] vectorDataBitWidth=%u invalid for datatype=isInt. Fallback to %u.", nativeDataBitWidth, default_data_bit_width);
        nativeDataBitWidth = default_data_bit_width;
    }
    if (fp16L2SquareMethod != hnswlib::FP16L2SquareMethod::Standard) {
        const bool fp16_fpma_supported =
            travUnit->spacetype == "L2" && travUnit->datatype == "isFloat";
        if (!fp16_fpma_supported) {
            print("[Config] fp16L2SquareMethod=%s requires spacetype=L2 and datatype=isFloat. Fallback to standard.",
                  hnswlib::fp16_l2_square_method_name(fp16L2SquareMethod));
            fp16L2SquareMethod = hnswlib::FP16L2SquareMethod::Standard;
        } else {
            print("[Config] fp16L2SquareMethod=%s enabled for L2 full-distance evaluation.",
                  hnswlib::fp16_l2_square_method_name(fp16L2SquareMethod));
            if (earlyExitEnable) {
                print("[Config] partial ANSMET lower-bound updates keep standard L2 square; selected FP16 FPMA kernel is applied on the full-distance step.");
            }
        }
    }
    dimSize = std::max<uint32_t>(1, (nativeDataBitWidth + 7) / 8);
    modeledReadBitWidth = nativeDataBitWidth;
    if (simulateFp16DataRead) {
        if (travUnit->datatype != "isFloat") {
            print("[Config] simulateFp16DataRead=true only supports datatype=isFloat. Ignore this option.");
            simulateFp16DataRead = false;
        } else {
            modeledReadBitWidth = 16;
            print("[Config] simulateFp16DataRead=true: model float vector fetch width as 16-bit for bandwidth/cycle estimation.");
        }
    }
    if (ansmetRuntimeTrueFp16BitChopEnable) {
        print("[Config] ansmetRuntimeTrueFp16BitChopEnable=true: ANSMET runtime partial distance uses FP16 stored-bit interval semantics.");
    }
    nBit = nativeDataBitWidth;
    if (earlyExitEnable) {
        bitStep = nativeDataBitWidth;
        dimStep = std::max(1U, 512U / nativeDataBitWidth);
        bitStep1=travUnit->getbitstep1();
        dimStep1=travUnit->getdimstep1();
        bitStep2=travUnit->getbitstep2();
        dimStep2=travUnit->getdimstep2();
        if(!earlyExitEnable){
            bitStep1=bitStep;
            dimStep1=dimStep;
            bitStep2=bitStep;
            dimStep2=dimStep;
        }
    } else {
        bitStep = nativeDataBitWidth;
        dimStep = std::max(1U, 512U / modeledReadBitWidth);
        bitStep1=travUnit->getbitstep1();
        dimStep1=travUnit->getdimstep1();
        bitStep2=travUnit->getbitstep2();
        dimStep2=travUnit->getdimstep2();
    }
    dimbitMode = travUnit->getDimbitMode();
    assert(nativeDataBitWidth > 0);
    assert(modeledReadBitWidth > 0);
    assert(nBit >= nativeDataBitWidth);
}

void HNSWEmbUnit::init() {
    m_clk = 0;
    pdis_callback = [this](Request& req) { return this->handlePDisReqLineReturn(req); };
    nParallelDisReq = param<uint32_t>("nParallelDisReq").desc("nParallelDisReq").required();
    nParallelPDisReq = param<uint32_t>("nParallelPDisReq").desc("nParallelPDisReq").required();
    nDim = param<uint32_t>("nDim").desc("nDim").required();
    configuredVectorDataBitWidth = param<uint32_t>("vectorDataBitWidth")
                                       .desc("Physical stored bit width per vector dimension for memory modeling")
                                       .default_val(0);
    const uint32_t default_data_bit_width = sizeof(Type) * 8;
    nativeDataBitWidth = configuredVectorDataBitWidth > 0 ? configuredVectorDataBitWidth : default_data_bit_width;
    nBit = nativeDataBitWidth;
    dimSize = std::max<uint32_t>(1, (nBit + 7) / 8);
    earlyExitEnable = param<bool>("earlyExitEnable").desc("earlyExitEnable").required();
    simulateFp16DataRead = param<bool>("simulateFp16DataRead")
                               .desc("Model vector fetch width as FP16 (16b) while keeping FP32 compute values")
                               .default_val(false);
    ansmetRuntimeTrueFp16BitChopEnable = param<bool>("ansmetRuntimeTrueFp16BitChopEnable")
                                             .desc("Use true FP16 stored-bit semantics for ANSMET runtime bitChop/partial distance")
                                             .default_val(false);
    disMethod = param<std::string>("disMethod").desc("disMethod").default_val("L2");
    {
        const std::string fp16_l2_square_method_raw = param<std::string>("fp16L2SquareMethod")
                                                          .desc("L2 square kernel: standard | fp16_fpma | fp16_fpma_quantized")
                                                          .default_val("standard");
        bool fp16_l2_square_method_valid = false;
        fp16L2SquareMethod = parse_fp16_l2_square_method(fp16_l2_square_method_raw, &fp16_l2_square_method_valid);
        if (!fp16_l2_square_method_valid) {
            print("[Config] invalid fp16L2SquareMethod=%s, fallback to standard", fp16_l2_square_method_raw.c_str());
            fp16L2SquareMethod = hnswlib::FP16L2SquareMethod::Standard;
        }
    }
    dualQueueBGAwareScheduleEnable = param<bool>("dualQueueBGAwareScheduleEnable")
                                         .desc("Enable bank-group aware scheduling for dual-queue ET requests")
                                         .default_val(false);
    dualQueueFineBankRowAwareScheduleEnable = param<bool>("dualQueueFineBankRowAwareScheduleEnable")
                                                  .desc("Enable bank-row-aware threshold-aware fine-stage scheduling for dual-queue ET")
                                                  .default_val(false);
    dualQueueCrossLevelNMPEnable = param<bool>("dualQueueCrossLevelNMPEnable")
                                       .desc("Enable ReCross-inspired cross-level NMP compute resource mapping for dual-queue ET")
                                       .default_val(false);
    dualQueueCoarseNMPLevel = param<std::string>("dualQueueCoarseNMPLevel")
                                  .desc("Phase-1 compute placement: rank | bankgroup")
                                  .default_val("rank");
    dualQueueFineNMPLevel = param<std::string>("dualQueueFineNMPLevel")
                                .desc("Phase-2 compute placement: rank | bank | bank_subarray")
                                .default_val("rank");
    dualQueueFineSubarrayWays = param<uint32_t>("dualQueueFineSubarrayWays")
                                    .desc("Effective subarray-parallel compute slots per bank for phase-2 bank-level NMP")
                                    .default_val(1);
    dualQueueFineSubarrayCount = param<uint32_t>("dualQueueFineSubarrayCount")
                                     .desc("Modeled number of subarrays per bank when assigning phase-2 bank-subarray targets")
                                     .default_val(256);
    dualQueueFineSubarrayInterleaveRows = param<uint32_t>("dualQueueFineSubarrayInterleaveRows")
                                              .desc("Number of rows per subarray interleave bucket for phase-2 bank-subarray target prediction")
                                              .default_val(1);
    dualQueueBGAwareCreditLimit = param<uint32_t>("dualQueueBGAwareCreditLimit")
                                      .desc("Per-bank-group soft credit budget used by dual-queue BG-aware scheduling")
                                      .default_val(32);
    dualQueueBGAwareCongestionThreshold = param<uint32_t>("dualQueueBGAwareCongestionThreshold")
                                              .desc("Predicted BG occupancy threshold to defer fine fetches")
                                              .default_val(24);
    dualQueueBGAwareBalanceWeight = param<uint32_t>("dualQueueBGAwareBalanceWeight")
                                        .desc("Balance-first weight for BG-aware scheduling")
                                        .default_val(4);
    dualQueueBGAwareLocalityWeight = param<uint32_t>("dualQueueBGAwareLocalityWeight")
                                         .desc("Row-hit bonus weight for fine-phase BG-aware scheduling")
                                         .default_val(8);
    dualQueueBGAwareFineDeferPenalty = param<uint32_t>("dualQueueBGAwareFineDeferPenalty")
                                           .desc("Penalty added to fine-phase requests targeting congested BGs")
                                           .default_val(128);
    dualQueueFineBankCreditLimit = param<uint32_t>("dualQueueFineBankCreditLimit")
                                       .desc("Per-bank soft credit budget for bank-row-aware fine-stage scheduling")
                                       .default_val(16);
    dualQueueFineBankTokenLimit = param<uint32_t>("dualQueueFineBankTokenLimit")
                                      .desc("Per-bank fine-stage token cap for bank-row-aware scheduling")
                                      .default_val(8);
    dualQueueFineBGTokenLimit = param<uint32_t>("dualQueueFineBGTokenLimit")
                                    .desc("Per-bank-group fine-stage token cap for bank-row-aware scheduling")
                                    .default_val(16);
    dualQueueFineBalanceWeight = param<uint32_t>("dualQueueFineBalanceWeight")
                                     .desc("Balance weight for bank-row-aware fine-stage scheduling")
                                     .default_val(4);
    dualQueueFineLocalityWeight = param<uint32_t>("dualQueueFineLocalityWeight")
                                      .desc("Row locality weight for bank-row-aware fine-stage scheduling")
                                      .default_val(16);
    dualQueueFineThresholdGainWeight = param<uint32_t>("dualQueueFineThresholdGainWeight")
                                           .desc("Threshold-gain weight for bank-row-aware fine-stage scheduling")
                                           .default_val(16);
    dualQueueFineAgeWeight = param<uint32_t>("dualQueueFineAgeWeight")
                                 .desc("Age bonus for long-waiting fine-stage requests")
                                 .default_val(256);
    dualQueueFineAgeThreshold = param<uint32_t>("dualQueueFineAgeThreshold")
                                    .desc("Age threshold in cycles before forcing fine-stage requests")
                                    .default_val(256);
    dualQueueFineTokenPenalty = param<uint32_t>("dualQueueFineTokenPenalty")
                                    .desc("Penalty per excess fine-stage bank/BG token")
                                    .default_val(64);
    dualQueueFineRowBurstCap = param<uint32_t>("dualQueueFineRowBurstCap")
                                   .desc("Cap for row-hit burst bonus in bank-row-aware scheduling")
                                   .default_val(4);
    maxPDisReqDispatchPerCycle = param<uint32_t>("maxPDisReqDispatchPerCycle")
                                     .desc("Max PDisReqs dispatched per tick (1 = legacy single-dispatch)")
                                     .default_val(1);
    rowSweepEnable = param<bool>("rowSweepEnable")
                         .desc("Enable row-sweep coalescing: after dispatching best PDisReq, sweep for same-row requests")
                         .default_val(false);
    maxRowSweepBatch = param<uint32_t>("maxRowSweepBatch")
                            .desc("Max additional same-row requests dispatched per row-sweep pass")
                            .default_val(4);
    dualQueueStagedAdmissionEnable = param<bool>("dualQueueStagedAdmissionEnable")
                                         .desc("Enable dual-window staged admission for dual-queue ET large-batch execution")
                                         .default_val(false);
    dualQueueCoarseAdmissionWindow = param<uint32_t>("dualQueueCoarseAdmissionWindow")
                                         .desc("Maximum number of active dual-queue coarse-stage disreqs per EmbUnit")
                                         .default_val(0);
    dualQueueFineAdmissionWindow = param<uint32_t>("dualQueueFineAdmissionWindow")
                                       .desc("Maximum active+queued dual-queue fine-stage admissions per EmbUnit")
                                       .default_val(0);
    std::fill(m_bgOutstanding.begin(), m_bgOutstanding.end(), 0);
    std::fill(m_bgRecentRow.begin(), m_bgRecentRow.end(), -1);
    std::fill(m_bgRecentBank.begin(), m_bgRecentBank.end(), -1);
    std::fill(m_bgCredits.begin(), m_bgCredits.end(), dualQueueBGAwareCreditLimit);
    std::fill(m_bankOutstanding.begin(), m_bankOutstanding.end(), 0);
    std::fill(m_bankRecentRow.begin(), m_bankRecentRow.end(), -1);
    std::fill(m_bankCredits.begin(), m_bankCredits.end(), dualQueueFineBankCreditLimit);
    std::fill(m_bankFineTokens.begin(), m_bankFineTokens.end(), 0);
    std::fill(m_bgFineTokens.begin(), m_bgFineTokens.end(), 0);
    modeledReadBitWidth = nativeDataBitWidth;
    if (simulateFp16DataRead) {
        if (travUnit->datatype != "isFloat") {
            print("[Config] simulateFp16DataRead=true only supports datatype=isFloat. Ignore this option.");
            simulateFp16DataRead = false;
        } else {
            modeledReadBitWidth = 16;
            print("[Config] simulateFp16DataRead=true: model float vector fetch width as 16-bit for bandwidth/cycle estimation.");
        }
    }

    // EmbUnit compute configuration parameters
    nFMAC = param<uint32_t>("nFMAC").desc("Legacy/default compute lane count per NDP").default_val(4);
    cyclesPerFMAC = param<uint32_t>("cyclesPerFMAC")
                        .desc("Standard FP distance issue interval in EmbUnit cycles")
                        .default_val(1);
    multiplierLatencyCycles = param<uint32_t>("multiplierLatencyCycles")
                                  .desc("Standard FP multiplier/FMAC pipeline latency in EmbUnit cycles")
                                  .default_val(1);
    fmacPipelined = param<bool>("fmacPipelined")
                        .desc("Whether the standard FP distance pipeline is pipelined")
                        .default_val(true);
    embComputeCycleRatio = param<double>("embComputeCycleRatio")
                               .desc("EmbUnit-cycle to memory-cycle conversion ratio")
                               .default_val(2.0);
    dualQueueCoarseOpsPerDim = param<uint32_t>("dualQueueCoarseOpsPerDim")
                                   .desc("Modeled exponent-only lower-bound ops per dimension")
                                   .default_val(1);
    dualQueueCoarseComputeModel = param<std::string>("dualQueueCoarseComputeModel")
                                      .desc("Phase-1 compute model: coarse_lb | std_fp")
                                      .default_val("coarse_lb");
    dualQueueCoarseStdFpOpsPerDim = param<uint32_t>("dualQueueCoarseStdFpOpsPerDim")
                                        .desc("Modeled std-FP ops per dim when phase-1 uses std_fp")
                                        .default_val(1);
    dualQueueFineOpsPerDim = param<uint32_t>("dualQueueFineOpsPerDim")
                                 .desc("Modeled dual fine-distance ops per dimension")
                                 .default_val(kStandardFpDistanceOpsPerDim);
    if (dualQueueCoarseComputeModel != "coarse_lb" && dualQueueCoarseComputeModel != "std_fp") {
        print("[Config] invalid dualQueueCoarseComputeModel=%s, fallback to coarse_lb", dualQueueCoarseComputeModel.c_str());
        dualQueueCoarseComputeModel = "coarse_lb";
    }
    dualQueueCoarseLanes = param<uint32_t>("dualQueueCoarseLanes")
                               .desc("Lane count for dual coarse lower-bound compute pool (0 => fallback to nFMAC)")
                               .default_val(0);
    dualQueueCoarseCyclesPerOp = param<uint32_t>("dualQueueCoarseCyclesPerOp")
                                     .desc("Coarse lower-bound issue interval in EmbUnit cycles")
                                     .default_val(1);
    dualQueueCoarseLatencyCycles = param<uint32_t>("dualQueueCoarseLatencyCycles")
                                       .desc("Coarse lower-bound pipeline latency in EmbUnit cycles")
                                       .default_val(1);
    dualQueueCoarseSetupCycles = param<uint32_t>("dualQueueCoarseSetupCycles")
                                     .desc("Coarse lower-bound setup cycles before pipelined work issues")
                                     .default_val(0);
    dualQueueCoarseFinalizeCycles = param<uint32_t>("dualQueueCoarseFinalizeCycles")
                                        .desc("Coarse lower-bound finalize cycles after work drains")
                                        .default_val(0);
    dualQueueCoarsePipelined = param<bool>("dualQueueCoarsePipelined")
                                   .desc("Whether the coarse lower-bound pipeline is pipelined")
                                   .default_val(true);
    dualQueueFineLanes = param<uint32_t>("dualQueueFineLanes")
                             .desc("Lane count for dual fine-distance compute pool (0 => fallback to nFMAC)")
                             .default_val(0);
    dualQueueFineCyclesPerOp = param<uint32_t>("dualQueueFineCyclesPerOp")
                                   .desc("Dual fine-distance issue interval in EmbUnit cycles")
                                   .default_val(1);
    dualQueueFineLatencyCycles = param<uint32_t>("dualQueueFineLatencyCycles")
                                     .desc("Dual fine-distance pipeline latency in EmbUnit cycles")
                                     .default_val(1);
    dualQueueFineSetupCycles = param<uint32_t>("dualQueueFineSetupCycles")
                                   .desc("Dual fine-distance setup cycles before pipelined work issues")
                                   .default_val(0);
    dualQueueFineFinalizeCycles = param<uint32_t>("dualQueueFineFinalizeCycles")
                                      .desc("Dual fine-distance finalize cycles after work drains")
                                      .default_val(0);
    dualQueueFinePipelined = param<bool>("dualQueueFinePipelined")
                                 .desc("Whether the dual fine-distance pipeline is pipelined")
                                 .default_val(true);
    stdFpLanes = param<uint32_t>("stdFpLanes")
                     .desc("Lane count for standard FP compute pool (0 => fallback to nFMAC)")
                     .default_val(0);
    stdFpSetupCycles = param<uint32_t>("stdFpSetupCycles")
                           .desc("Standard FP setup cycles before pipelined work issues")
                           .default_val(0);
    stdFpFinalizeCycles = param<uint32_t>("stdFpFinalizeCycles")
                              .desc("Standard FP finalize cycles after work drains")
                              .default_val(0);
    zeroMemFinalizeSeparatePoolEnable = param<bool>("zeroMemFinalizeSeparatePoolEnable")
                                            .desc("Route dual fine zero-mem tasks to a separate finalize pool")
                                            .default_val(false);
    zeroMemFinalizeWorkItemsPerTask = param<uint32_t>("zeroMemFinalizeWorkItemsPerTask")
                                          .desc("Work items assigned to each zero-mem finalize task")
                                          .default_val(1);
    zeroMemFinalizeModeledOpsPerTask = param<uint32_t>("zeroMemFinalizeModeledOpsPerTask")
                                           .desc("Modeled ops assigned to each zero-mem finalize task")
                                           .default_val(1);
    zeroMemFinalizeLanes = param<uint32_t>("zeroMemFinalizeLanes")
                               .desc("Lane count for zero-mem finalize pool (0 => fallback to 1)")
                               .default_val(0);
    zeroMemFinalizeIssueInterval = param<uint32_t>("zeroMemFinalizeIssueInterval")
                                       .desc("Zero-mem finalize issue interval in EmbUnit cycles")
                                       .default_val(1);
    zeroMemFinalizeLatencyCycles = param<uint32_t>("zeroMemFinalizeLatencyCycles")
                                       .desc("Zero-mem finalize pipeline latency in EmbUnit cycles")
                                       .default_val(1);
    zeroMemFinalizeSetupCycles = param<uint32_t>("zeroMemFinalizeSetupCycles")
                                     .desc("Zero-mem finalize setup cycles before work issues")
                                     .default_val(0);
    zeroMemFinalizeFinalizeCycles = param<uint32_t>("zeroMemFinalizeFinalizeCycles")
                                        .desc("Zero-mem finalize drain/finalize cycles after work issues")
                                        .default_val(0);
    zeroMemFinalizePipelined = param<bool>("zeroMemFinalizePipelined")
                                   .desc("Whether the zero-mem finalize pool is pipelined")
                                   .default_val(true);
    if (dualQueueCoarseNMPLevel != "rank" && dualQueueCoarseNMPLevel != "bankgroup") {
        print("[Config] invalid dualQueueCoarseNMPLevel=%s, fallback to rank", dualQueueCoarseNMPLevel.c_str());
        dualQueueCoarseNMPLevel = "rank";
    }
    if (dualQueueFineNMPLevel != "rank" && dualQueueFineNMPLevel != "bank" && dualQueueFineNMPLevel != "bank_subarray") {
        print("[Config] invalid dualQueueFineNMPLevel=%s, fallback to rank", dualQueueFineNMPLevel.c_str());
        dualQueueFineNMPLevel = "rank";
    }
    if (dualQueueFineSubarrayWays == 0) {
        dualQueueFineSubarrayWays = 1;
    }
    if (dualQueueFineSubarrayCount == 0) {
        dualQueueFineSubarrayCount = 1;
    }
    if (dualQueueFineSubarrayInterleaveRows == 0) {
        dualQueueFineSubarrayInterleaveRows = 1;
    }
    if (dualQueueCoarseLanes == 0) dualQueueCoarseLanes = nFMAC;
    if (dualQueueFineLanes == 0) dualQueueFineLanes = nFMAC;
    if (stdFpLanes == 0) stdFpLanes = nFMAC;
    if (zeroMemFinalizeLanes == 0) zeroMemFinalizeLanes = 1;
    if (dualQueueCoarseAdmissionWindow == 0) dualQueueCoarseAdmissionWindow = nParallelDisReq;
    if (dualQueueFineAdmissionWindow == 0) dualQueueFineAdmissionWindow = nParallelDisReq;
    if (dualQueueStagedAdmissionEnable) {
        print("[Config] dualQueueStagedAdmissionEnable=true coarseWindow=%u fineWindow=%u", dualQueueCoarseAdmissionWindow, dualQueueFineAdmissionWindow);
    }
    for (size_t poolIndex = 0; poolIndex < kNumComputePools; ++poolIndex) {
        computeResourceReadyCycles[poolIndex].assign(getComputeResourceCount(poolIndex), 0);
    }
    if (dualQueueCrossLevelNMPEnable) {
        print("[Config] dualQueueCrossLevelNMPEnable=true coarseLevel=%s fineLevel=%s fineSubarrayWays=%u fineSubarrayCount=%u fineSubarrayInterleaveRows=%u",
              dualQueueCoarseNMPLevel.c_str(),
              dualQueueFineNMPLevel.c_str(),
              dualQueueFineSubarrayWays,
              dualQueueFineSubarrayCount,
              dualQueueFineSubarrayInterleaveRows);
    }

    // Per-rank vector LRU cache configuration
    vectorCacheSize = param<uint32_t>("vectorCacheSize").desc("Number of vectors to cache per rank (0=disabled)").default_val(0);
    vectorCacheMode = param<std::string>("vectorCacheMode")
                          .desc("Object cache mode: full_object | phase2_tail_only")
                          .default_val("full_object");
    if (vectorCacheMode != "full_object" && vectorCacheMode != "phase2_tail_only") {
        print("[Config] invalid vectorCacheMode=%s, fallback to full_object", vectorCacheMode.c_str());
        vectorCacheMode = "full_object";
    }
    vectorCacheEnabled = (vectorCacheSize > 0);
    m_vector_cache.init(vectorCacheSize);
    print("[Config] vector cache size=%u mode=%s enabled=%d",
          vectorCacheSize,
          vectorCacheMode.c_str(),
          vectorCacheEnabled ? 1 : 0);
    phase1SignExpCacheSize = param<uint32_t>("phase1SignExpCacheSize")
                                 .desc("Number of phase-1 sign+exp vectors to cache per rank (0=disabled)")
                                 .default_val(0);
    phase1SignExpCacheEnabled = (phase1SignExpCacheSize > 0);
    m_phase1_signexp_cache.init(phase1SignExpCacheSize);
    print("[Config] phase1 signexp cache size=%u enabled=%d",
          phase1SignExpCacheSize,
          phase1SignExpCacheEnabled ? 1 : 0);
    if (travUnit && travUnit->isMFNNSEnabled()) {
        print("[Config] MFNNS semantics enabled in EmbUnit (dualQueue=%d)",
              travUnit->isDualQueueLowerBoundETEnabled() ? 1 : 0);
    }

    register_stat(s_num_pdisreq).name("s_num_pdisreq");
    register_stat(s_num_disreq).name("s_num_disreq");
    register_stat(s_num_early_exit).name("s_num_early_exit");
    register_stat(s_num_similarity_mul).name("s_num_similarity_mul");
    register_stat(s_num_redundant_mul).name("s_num_redundant_mul");
    register_stat(s_modeled_read_bits).name("s_modeled_read_bits");
    register_stat(s_dualq_phase1_pdisreq).name("s_dualq_phase1_pdisreq");
    register_stat(s_dualq_phase2_pdisreq).name("s_dualq_phase2_pdisreq");
    register_stat(s_dualq_phase1_modeled_read_bits).name("s_dualq_phase1_modeled_read_bits");
    register_stat(s_dualq_phase2_modeled_read_bits).name("s_dualq_phase2_modeled_read_bits");
    register_stat(s_pdis_mem_service_cycles).name("s_pdis_mem_service_cycles");
    register_stat(s_dualq_phase1_pdis_mem_service_cycles).name("s_dualq_phase1_pdis_mem_service_cycles");
    register_stat(s_dualq_phase2_pdis_mem_service_cycles).name("s_dualq_phase2_pdis_mem_service_cycles");
    register_stat(s_fmac_raw_cycles).name("s_fmac_raw_cycles");
    register_stat(s_fmac_hidden_cycles).name("s_fmac_hidden_cycles");
    register_stat(s_fmac_queue_cycles).name("s_fmac_queue_cycles");
    register_stat(s_fmac_compute_cycles).name("s_fmac_compute_cycles");
    register_stat(s_dualq_phase1_exposed_compute_cycles).name("s_dualq_phase1_exposed_compute_cycles");
    register_stat(s_dualq_phase2_exposed_compute_cycles).name("s_dualq_phase2_exposed_compute_cycles");
    register_stat(s_dualq_phase2_early_exit).name("s_dualq_phase2_early_exit");
    register_stat(s_mfnns_linear_early_exit).name("s_mfnns_linear_early_exit");
    register_stat(s_fmac_ops).name("s_fmac_ops");
    register_stat(s_dualq_phase1_fmac_ops).name("s_dualq_phase1_fmac_ops");
    register_stat(s_dualq_phase2_fmac_ops).name("s_dualq_phase2_fmac_ops");
    register_stat(s_dualq_phase1_modeled_ops).name("s_dualq_phase1_modeled_ops");
    register_stat(s_dualq_phase2_modeled_ops).name("s_dualq_phase2_modeled_ops");
    register_stat(s_ansmet_et_fmac_ops).name("s_ansmet_et_fmac_ops");
    register_stat(s_ansmet_total_fmac_ops).name("s_ansmet_total_fmac_ops");
    register_stat(s_num_zero_mem_pdisreq).name("s_num_zero_mem_pdisreq");
    register_stat(s_num_zero_mem_finalize_tasks).name("s_num_zero_mem_finalize_tasks");
    register_stat(s_zero_mem_finalize_cycles).name("s_zero_mem_finalize_cycles");
    register_stat(s_dualq_bgaware_reordered).name("s_dualq_bgaware_reordered");
    register_stat(s_dualq_bgaware_fine_deferred).name("s_dualq_bgaware_fine_deferred");
    register_stat(s_dualq_bgaware_rowhit_selected).name("s_dualq_bgaware_rowhit_selected");
    register_stat(s_dualq_rowaware_reordered).name("s_dualq_rowaware_reordered");
    register_stat(s_dualq_rowaware_rowhit_selected).name("s_dualq_rowaware_rowhit_selected");
    register_stat(s_dualq_rowaware_rowburst_selected).name("s_dualq_rowaware_rowburst_selected");
    register_stat(s_dualq_rowaware_threshold_selected).name("s_dualq_rowaware_threshold_selected");
    register_stat(s_dualq_rowaware_age_forced).name("s_dualq_rowaware_age_forced");
    register_stat(s_dualq_rowaware_token_throttled).name("s_dualq_rowaware_token_throttled");
    register_stat(s_dualq_crosslevel_coarse_bg_tasks).name("s_dualq_crosslevel_coarse_bg_tasks");
    register_stat(s_dualq_crosslevel_fine_bank_tasks).name("s_dualq_crosslevel_fine_bank_tasks");
    register_stat(s_dualq_crosslevel_fine_subarray_tasks).name("s_dualq_crosslevel_fine_subarray_tasks");
    register_stat(s_dualq_stage_admission_coarse_blocked).name("s_dualq_stage_admission_coarse_blocked");
    register_stat(s_dualq_stage_admission_fine_blocked).name("s_dualq_stage_admission_fine_blocked");
    register_stat(s_dualq_stage_admission_fine_deferred).name("s_dualq_stage_admission_fine_deferred");
    register_stat(s_dualq_stage_admission_fine_promoted).name("s_dualq_stage_admission_fine_promoted");
    register_stat(s_dualq_stage_admission_fine_wait_cycles).name("s_dualq_stage_admission_fine_wait_cycles");
    register_stat(s_batch_dispatch_total).name("s_batch_dispatch_total");
    register_stat(s_batch_dispatch_extra).name("s_batch_dispatch_extra");
    register_stat(s_row_sweep_dispatched).name("s_row_sweep_dispatched");
    register_stat(s_row_sweep_opportunities).name("s_row_sweep_opportunities");
}

void HNSWEmbUnit::tick() {
    m_clk += 1;
    uint64_t curCycle = m_memory_system->get_clk();
    tryPromoteDeferredDualQueueFineAdmissions(curCycle);

    std::vector<PendingComputeTask> completedTasks;
    while (!pendingComputeCompletions.empty() && pendingComputeCompletions.top().completionCycle <= curCycle) {
        completedTasks.push_back(pendingComputeCompletions.top());
        pendingComputeCompletions.pop();
    }
    for (const PendingComputeTask& completedTask : completedTasks) {
        handleCompletedPDisReq(completedTask);
    }

    if (!pendCacheHitPDisReqs.empty()) {
        PDisReq pDisReq = pendCacheHitPDisReqs.front();
        pendCacheHitPDisReqs.pop();
        schedulePDisReqCompute(pDisReq);
    }

    dispatchPendingPDisReqs(curCycle);
}

void HNSWEmbUnit::dispatchPendingPDisReqs(uint64_t curCycle) {
    if (pendPDisReqs.empty()) {
        return;
    }
    const uint32_t dispatchBudget = maxPDisReqDispatchPerCycle;
    uint32_t dispatched = 0;

    // --- Primary dispatch: select best PDisReq and send ---
    const size_t selectedIdx = selectNextPendingPDisReqIndex(curCycle);
    PDisReq bestReq = pendPDisReqs[selectedIdx];
    if (!sendPDisReq(bestReq)) {
        return;  // Memory system backpressure
    }
    // Erase via swap-and-pop for O(1) removal from deque-converted-to-vector section
    if (selectedIdx + 1 < pendPDisReqs.size()) {
        pendPDisReqs[selectedIdx] = std::move(pendPDisReqs.back());
    }
    pendPDisReqs.pop_back();
    dispatched = 1;
    s_batch_dispatch_total += 1;

    // --- Row-sweep: find and dispatch same-row requests ---
    // Row sweep has its own budget (maxRowSweepBatch), independent of maxPDisReqDispatchPerCycle.
    // This allows row-sweep to fire even with maxPDisReqDispatchPerCycle=1.
    if (rowSweepEnable && !pendPDisReqs.empty()) {
        // Collect open (bank, row) keys from the just-dispatched request
        auto disreqIt = inflightDisReqs.find(bestReq.disreqId);
        if (disreqIt != inflightDisReqs.end()) {
            std::vector<Addr_t> bestAddrs = buildRequestAddrs(disreqIt->second, bestReq.curDim, bestReq.curBit, false);
            std::unordered_set<uint64_t> openRowKeys;
            for (Addr_t addr : bestAddrs) {
                PredictedBGTgt tgt = predictBGTgt(addr);
                if (tgt.bankgroup < kBGAwareNumBankGroups && tgt.bank < kBGAwareNumBanksPerGroup) {
                    uint64_t key = (static_cast<uint64_t>(flattenBank(tgt)) << 56) ^ (tgt.row & 0x00FFFFFFFFFFFFFFULL);
                    openRowKeys.insert(key);
                }
            }

            if (!openRowKeys.empty()) {
                uint32_t sweepDispatched = 0;
                // Scan pending queue for same-row candidates
                for (size_t i = 0; i < pendPDisReqs.size() && sweepDispatched < maxRowSweepBatch; ) {
                    auto candIt = inflightDisReqs.find(pendPDisReqs[i].disreqId);
                    if (candIt == inflightDisReqs.end()) {
                        // Orphaned request, send to drop it
                        PDisReq orphan = pendPDisReqs[i];
                        sendPDisReq(orphan);
                        if (i + 1 < pendPDisReqs.size()) {
                            pendPDisReqs[i] = std::move(pendPDisReqs.back());
                        }
                        pendPDisReqs.pop_back();
                        dispatched++;
                        s_batch_dispatch_total += 1;
                        s_batch_dispatch_extra += 1;
                        continue;
                    }
                    std::vector<Addr_t> candAddrs = buildRequestAddrs(candIt->second, pendPDisReqs[i].curDim, pendPDisReqs[i].curBit, false);
                    bool hasRowHit = false;
                    for (Addr_t addr : candAddrs) {
                        PredictedBGTgt tgt = predictBGTgt(addr);
                        if (tgt.bankgroup < kBGAwareNumBankGroups && tgt.bank < kBGAwareNumBanksPerGroup) {
                            uint64_t key = (static_cast<uint64_t>(flattenBank(tgt)) << 56) ^ (tgt.row & 0x00FFFFFFFFFFFFFFULL);
                            if (openRowKeys.count(key)) {
                                hasRowHit = true;
                                break;
                            }
                        }
                    }
                    if (hasRowHit) {
                        s_row_sweep_opportunities += 1;
                        PDisReq sweepReq = pendPDisReqs[i];
                        if (sendPDisReq(sweepReq)) {
                            if (i + 1 < pendPDisReqs.size()) {
                                pendPDisReqs[i] = std::move(pendPDisReqs.back());
                            }
                            pendPDisReqs.pop_back();
                            dispatched++;
                            sweepDispatched++;
                            s_batch_dispatch_total += 1;
                            s_batch_dispatch_extra += 1;
                            s_row_sweep_dispatched += 1;
                            // Also add swept request's rows to open set for chain effect
                            for (Addr_t addr : candAddrs) {
                                PredictedBGTgt tgt = predictBGTgt(addr);
                                if (tgt.bankgroup < kBGAwareNumBankGroups && tgt.bank < kBGAwareNumBanksPerGroup) {
                                    uint64_t key = (static_cast<uint64_t>(flattenBank(tgt)) << 56) ^ (tgt.row & 0x00FFFFFFFFFFFFFFULL);
                                    openRowKeys.insert(key);
                                }
                            }
                        } else {
                            break;  // Memory backpressure, stop sweep
                        }
                    } else {
                        i++;
                    }
                }
            }
        }
    }

    // --- Additional batch dispatch (non-sweep): dispatch more best-scored requests ---
    while (dispatched < dispatchBudget && !pendPDisReqs.empty()) {
        const size_t nextIdx = selectNextPendingPDisReqIndex(curCycle);
        PDisReq nextReq = pendPDisReqs[nextIdx];
        if (!sendPDisReq(nextReq)) {
            break;
        }
        if (nextIdx + 1 < pendPDisReqs.size()) {
            pendPDisReqs[nextIdx] = std::move(pendPDisReqs.back());
        }
        pendPDisReqs.pop_back();
        dispatched++;
        s_batch_dispatch_total += 1;
        s_batch_dispatch_extra += 1;
    }
}


} // namespace Ramulator
