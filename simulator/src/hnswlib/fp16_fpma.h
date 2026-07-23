#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace hnswlib {

enum class FP16L2SquareMethod {
    Standard = 0,
    FP16_FPMA,
    FP16_FPMA_Quantized,
};

inline const char* fp16_l2_square_method_name(FP16L2SquareMethod method) {
    switch (method) {
        case FP16L2SquareMethod::FP16_FPMA:
            return "fp16_fpma";
        case FP16L2SquareMethod::FP16_FPMA_Quantized:
            return "fp16_fpma_quantized";
        case FP16L2SquareMethod::Standard:
        default:
            return "standard";
    }
}

inline uint16_t fp32_to_fp16_bits(float val) {
    uint32_t f32 = 0;
    std::memcpy(&f32, &val, sizeof(float));

    const uint32_t sign = (f32 >> 16) & 0x8000U;
    const int32_t f32_expo = static_cast<int32_t>((f32 >> 23) & 0xFFU);
    const uint32_t f32_mant = f32 & 0x7FFFFFU;

    if (f32_expo == 0 && f32_mant == 0) {
        return static_cast<uint16_t>(sign);
    }

    if (f32_expo == 255) {
        if (f32_mant == 0) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
        uint16_t nan_mant = static_cast<uint16_t>(f32_mant >> 13);
        if (nan_mant == 0) {
            nan_mant = 1;
        }
        return static_cast<uint16_t>(sign | 0x7C00U | nan_mant);
    }

    int32_t fp16_expo = f32_expo - 127 + 15;
    if (fp16_expo <= 0) {
        if (fp16_expo < -10) {
            return static_cast<uint16_t>(sign);
        }
        uint32_t mant = (f32_mant | 0x800000U) >> (1 - fp16_expo + 13);
        if (((f32_mant | 0x800000U) >> (1 - fp16_expo + 12)) & 1U) {
            mant += 1U;
        }
        return static_cast<uint16_t>(sign | (mant & 0x03FFU));
    }

    if (fp16_expo >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    uint32_t fp16_mant = f32_mant >> 13;
    const uint32_t round_bit = (f32_mant >> 12) & 1U;
    const uint32_t sticky_bits = f32_mant & 0xFFFU;
    if (round_bit && (sticky_bits || (fp16_mant & 1U))) {
        fp16_mant += 1U;
        if (fp16_mant >= 1024U) {
            fp16_expo += 1;
            fp16_mant = 0;
            if (fp16_expo >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00U);
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(fp16_expo) << 10) | fp16_mant);
}

inline float fp16_bits_to_fp32(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits & 0x8000U)) << 16;
    const uint32_t expo = static_cast<uint32_t>((bits >> 10) & 0x1FU);
    const uint32_t mant = static_cast<uint32_t>(bits & 0x03FFU);

    uint32_t f32 = 0;
    if (expo == 0) {
        if (mant == 0) {
            f32 = sign;
        } else {
            int32_t e = -14;
            uint32_t m = mant;
            while ((m & 0x400U) == 0U) {
                m <<= 1U;
                --e;
            }
            m &= 0x03FFU;
            f32 = sign | (static_cast<uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else if (expo == 0x1FU) {
        f32 = sign | 0x7F800000U | (mant << 13);
    } else {
        f32 = sign | ((expo - 15U + 127U) << 23) | (mant << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &f32, sizeof(float));
    return result;
}

inline float fp16_round(float val) {
    return fp16_bits_to_fp32(fp32_to_fp16_bits(val));
}

constexpr int kFP16FixedPointIntBits = 3;
constexpr int kFP16FixedPointFracBits = 14;
constexpr int kFP16FixedPointTotalBits = kFP16FixedPointIntBits + kFP16FixedPointFracBits;
constexpr uint64_t kFP16FixedPointScale = 1ULL << kFP16FixedPointFracBits;
constexpr uint64_t kFP16FixedPointMax = (1ULL << kFP16FixedPointTotalBits) - 1ULL;
constexpr uint32_t kFP16FpmaAccumulatorFlushWidth = 1600U;

static_assert(kFP16FixedPointTotalBits <= 64, "FP16 FPMA fixed-point width must fit in 64 bits");

inline float fp16_fpma_square(float val) {
    // Diagonal of recall_analysis/hnswlib/space_l2_dynamic_precision.h:E5_M10_16x16_comp_table.
    static constexpr int16_t kSquareApproxTable[16] = {
        1, 9, 25, 49, 81, 121, 166, 145, 113, 85, 61, 41, 25, 13, 5, 0,
    };

    const uint16_t fp16 = fp32_to_fp16_bits(val);
    const int32_t expo = (fp16 >> 10) & 0x1F;
    const int32_t mant = fp16 & 0x03FF;

    if (expo == 0) {
        return 0.0f;
    }

    if (expo == 31) {
        if (mant == 0) {
            return fp16_bits_to_fp32(0x7C00U);
        }
        return fp16_bits_to_fp32(static_cast<uint16_t>(0x7C00U | mant));
    }

    int32_t result_expo = expo + expo - 15;
    int32_t mant_sum = mant + mant;
    if (mant_sum >= 1024) {
        result_expo += 1;
        mant_sum -= 1024;
    }

    int32_t result_mant = mant_sum + kSquareApproxTable[(mant >> 6) & 0xF];
    if (result_mant >= 1024) {
        result_expo += 1;
        result_mant -= 1024;
    }

    if (result_expo <= 0) {
        return 0.0f;
    }

    if (result_expo >= 31) {
        return fp16_bits_to_fp32(0x7C00U);
    }

    const uint16_t result_bits =
        static_cast<uint16_t>((static_cast<uint32_t>(result_expo) << 10) |
                              (static_cast<uint32_t>(result_mant) & 0x03FFU));
    return fp16_bits_to_fp32(result_bits);
}

inline uint32_t fp16_fpma_square_quantized(float lhs, float rhs) {
    const float lhs_fp16 = fp16_bits_to_fp32(fp32_to_fp16_bits(lhs));
    const float rhs_fp16 = fp16_bits_to_fp32(fp32_to_fp16_bits(rhs));
    const float diff = fp16_round(lhs_fp16 - rhs_fp16);
    const float square = fp16_fpma_square(diff);
    const float scaled = square * static_cast<float>(kFP16FixedPointScale);

    uint64_t rounded = 0;
    if (scaled >= static_cast<float>(kFP16FixedPointMax)) {
        rounded = kFP16FixedPointMax;
    } else if (scaled > 0.0f) {
        rounded = static_cast<uint64_t>(scaled + 0.5f);
    }
    return static_cast<uint32_t>(rounded);
}

inline float dequantize_fp16_fpma(uint64_t quantized) {
    return static_cast<float>(quantized) / static_cast<float>(kFP16FixedPointScale);
}

inline float compute_l2_square(float lhs, float rhs, FP16L2SquareMethod method) {
    if (method == FP16L2SquareMethod::Standard) {
        const float diff = lhs - rhs;
        return diff * diff;
    }
    if (method == FP16L2SquareMethod::FP16_FPMA) {
        const float lhs_fp16 = fp16_round(lhs);
        const float rhs_fp16 = fp16_round(rhs);
        const float diff = fp16_round(lhs_fp16 - rhs_fp16);
        return fp16_fpma_square(diff);
    }
    return dequantize_fp16_fpma(fp16_fpma_square_quantized(lhs, rhs));
}

inline float compute_l2_distance(const float* lhs, const float* rhs, size_t qty, FP16L2SquareMethod method) {
    if (method == FP16L2SquareMethod::Standard) {
        float result = 0.0f;
        for (size_t i = 0; i < qty; ++i) {
            const float diff = lhs[i] - rhs[i];
            result += diff * diff;
        }
        return result;
    }

    if (method == FP16L2SquareMethod::FP16_FPMA) {
        float result = 0.0f;
        for (size_t i = 0; i < qty; ++i) {
            result += compute_l2_square(lhs[i], rhs[i], method);
        }
        return result;
    }

    uint64_t accumulator = 0;
    float result = 0.0f;
    for (size_t i = 0; i < qty; ++i) {
        accumulator += fp16_fpma_square_quantized(lhs[i], rhs[i]);
        if (((i + 1U) % kFP16FpmaAccumulatorFlushWidth) == 0U || (i + 1U) == qty) {
            result += dequantize_fp16_fpma(accumulator);
            accumulator = 0;
        }
    }
    return result;
}

}  // namespace hnswlib
