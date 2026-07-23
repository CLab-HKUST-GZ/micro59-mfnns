#pragma once
#include "../hnswlib/hnswlib.h"
#include <algorithm>
#include <cmath>
#include <cstring>  // for memcpy
#include <cstdint>  // for uint32_t, uint16_t

namespace hnswlib {

/**
 * Dynamic precision distance computation wrapper
 *
 * Core idea:
 * 1. Vectors always stored as FP32 (graph structure based on this)
 * 2. Dynamically convert to target precision during distance computation
 * 3. Switch precision via set_precision() without rebuilding graph
 */

// ====================== FP16 FPMA Support ======================

/**
 * FP16 square operation compensation table (complete 1024-entry table)
 *
 * Principle:
 * - FP16 format: E5M10 (1 sign + 5 exponent + 10 mantissa)
 * - FPMA square: Use full mantissa (0-1023) to directly index compensation table
 * - Compensation table pre-computed via Python script: generate_fp16_square_comp_table.py
 *
 * Generation date: 2025-10-23
 * Purpose: Test if complete compensation table can achieve same precision as standard FP16 for FPMA
 */


// ====================== Fixed-Point Configuration ======================

/**
 * Quantization version switch
 *
 * Set to 1: FP16_FPMA/FP8_FPMA use quantized version (Q-format fixed-point)
 * Set to 0: FP16_FPMA/FP8_FPMA use floating-point version (recommended, faster and more accurate)
 */
#define USE_FPMA_QUANTIZED 0

/**
 * FP16 fixed-point configuration constants (bit width freely configurable, not limited to 16 bits)
 *
 * Q format: Q_m.n represents m integer bits + n fractional bits
 * Total bit width = m + n (can be arbitrarily configured, recommended ≤32 bits)
 *
 * Common configuration examples:
 * - Q4.12 (16-bit): integer[0,15],     precision 1/4096   ≈ 0.00024  ✓ Current config
 * - Q5.11 (16-bit): integer[0,31],     precision 1/2048   ≈ 0.0005
 * - Q6.10 (16-bit): integer[0,63],     precision 1/1024   ≈ 0.001
 * - Q7.9  (16-bit): integer[0,127],    precision 1/512    ≈ 0.002
 * - Q8.16 (24-bit): integer[0,255],    precision 1/65536  ≈ 0.000015
 * - Q10.14(24-bit): integer[0,1023],   precision 1/16384  ≈ 0.00006
 * - Q16.16(32-bit): integer[0,65535],  precision 1/65536  ≈ 0.000015
 *
 * Recommended configuration (based on data distribution):
 * - Single (a-b)^2 range: [0, 0.36]
 * - After 300-dim accumulation: [0, ~60]
 * - Current: Q4.12 (16-bit, Recall 0.9066)
 * - High precision: Q8.16 (24-bit) or Q16.16 (32-bit)
 */
#define FP16_FIXED_POINT_INT_BITS  3    // Integer bits (freely adjustable)
#define FP16_FIXED_POINT_FRAC_BITS 14   // Fractional bits (freely adjustable)
#define FP16_FIXED_POINT_TOTAL_BITS (FP16_FIXED_POINT_INT_BITS + FP16_FIXED_POINT_FRAC_BITS)
#define FP16_FIXED_POINT_SCALE (1ULL << FP16_FIXED_POINT_FRAC_BITS)
#define FP16_FIXED_POINT_MAX ((1ULL << FP16_FIXED_POINT_TOTAL_BITS) - 1)

// 3-13

// Bit width validation: ensure total bit width does not exceed 64 bits (actually recommend ≤32 bits)
static_assert(FP16_FIXED_POINT_TOTAL_BITS <= 64,
              "FP16 total bits must not exceed 64");

#define QUERY_SCALE_M 0

typedef union{
    float fp32;
    uint32_t u32;
} fp32_union;

uint32_t fp32_to_u32(float fp32){
    fp32_union u;
    u.fp32 = fp32;
    return u.u32;
}

float u32_to_fp32(uint32_t u32){
    fp32_union u;
    u.u32 = u32;
    return u.fp32;
}


static const int16_t SQUARE_COMP_TABLE[1024] = {
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,
2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,
4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,
6,6,7,7,7,7,7,7,8,8,8,8,8,8,9,9,
9,9,9,10,10,10,10,10,11,11,11,11,11,12,12,12,
12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,
16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,
20,21,21,21,21,22,22,22,23,23,23,23,24,24,24,25,
25,25,26,26,26,27,27,27,28,28,28,29,29,29,30,30,
30,31,31,31,32,32,32,33,33,33,34,34,35,35,35,36,
36,36,37,37,38,38,38,39,39,39,40,40,41,41,41,42,
42,43,43,43,44,44,45,45,46,46,46,47,47,48,48,49,
49,49,50,50,51,51,52,52,53,53,53,54,54,55,55,56,
56,57,57,58,58,59,59,60,60,61,61,62,62,63,63,64,
64,65,65,66,66,67,67,68,68,69,69,70,70,71,71,72,
72,73,73,74,74,75,75,76,77,77,78,78,79,79,80,80,
81,82,82,83,83,84,84,85,86,86,87,87,88,88,89,90,
90,91,91,92,93,93,94,94,95,96,96,97,98,98,99,99,
100,101,101,102,103,103,104,104,105,106,106,107,108,108,109,110,
110,111,112,112,113,114,114,115,116,116,117,118,118,119,120,120,
121,122,122,123,124,124,125,126,127,127,128,129,129,130,131,132,
132,133,134,134,135,136,137,137,138,139,140,140,141,142,143,143,
144,145,146,146,147,148,149,149,150,151,152,152,153,154,155,155,
156,157,158,159,159,160,161,162,163,163,164,165,166,167,167,168,
169,170,171,171,172,173,174,175,176,175,175,174,173,173,172,172,
171,171,170,169,169,168,168,167,167,166,165,165,164,164,163,163,
162,161,161,160,160,159,159,158,158,157,156,156,155,155,154,154,
153,153,152,151,151,150,150,149,149,148,148,147,147,146,146,145,
144,144,143,143,142,142,141,141,140,140,139,139,138,138,137,137,
136,136,135,135,134,134,133,133,132,132,131,131,130,130,129,129,
128,128,127,127,126,126,125,125,124,124,123,123,122,122,121,121,
120,120,119,119,118,118,117,117,116,116,115,115,114,114,113,113,
112,112,112,111,111,110,110,109,109,108,108,107,107,106,106,106,
105,105,104,104,103,103,102,102,102,101,101,100,100,99,99,98,
98,98,97,97,96,96,95,95,95,94,94,93,93,92,92,92,
91,91,90,90,89,89,89,88,88,87,87,87,86,86,85,85,
84,84,84,83,83,82,82,82,81,81,80,80,80,79,79,79,
78,78,77,77,77,76,76,75,75,75,74,74,74,73,73,72,
72,72,71,71,71,70,70,69,69,69,68,68,68,67,67,66,
66,66,65,65,65,64,64,64,63,63,63,62,62,62,61,61,
60,60,60,59,59,59,58,58,58,57,57,57,56,56,56,55,
55,55,54,54,54,53,53,53,53,52,52,52,51,51,51,50,
50,50,49,49,49,48,48,48,48,47,47,47,46,46,46,45,
45,45,45,44,44,44,43,43,43,42,42,42,42,41,41,41,
40,40,40,40,39,39,39,39,38,38,38,37,37,37,37,36,
36,36,36,35,35,35,35,34,34,34,34,33,33,33,33,32,
32,32,32,31,31,31,31,30,30,30,30,29,29,29,29,28,
28,28,28,27,27,27,27,27,26,26,26,26,25,25,25,25,
24,24,24,24,24,23,23,23,23,23,22,22,22,22,22,21,
21,21,21,21,20,20,20,20,20,19,19,19,19,19,18,18,
18,18,18,17,17,17,17,17,17,16,16,16,16,16,15,15,
15,15,15,15,14,14,14,14,14,14,13,13,13,13,13,13,
12,12,12,12,12,12,12,11,11,11,11,11,11,11,10,10,
10,10,10,10,10,9,9,9,9,9,9,9,9,8,8,8,
8,8,8,8,8,7,7,7,7,7,7,7,7,6,6,6,
6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,
4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,
3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,
2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};


/*
    Method: mean     | Values: [5, 37, 101, 154, 99, 51, 19, 3]
    Method: median   | Values: [4, 36, 100, 154, 98, 50, 18, 2]
    Downsample 2D    | Values: [4, 36, 100, 158, 98, 50, 18, 2]
*/
// 5, 37, 101, 154, 99, 51, 19, 3
static const int16_t SQUARE_APPROX_TABLE[8] = {
    4, 36, 100, 158, 98, 50, 18, 2
    // 20, 20, 129, 129, 74, 74, 10, 10
    // 75, 75, 75, 75, 42, 42, 42, 42
    // 59, 59, 59, 59, 59, 59, 59, 59
    // 0, 0, 0, 0, 0, 0, 0, 0
};
// static const int16_t SQUARE_APPROX_TABLE2[8] = {
//     // 5, 37, 101, 154, 99, 51, 19, 3
//     // 0, 0, 0, 0, 0, 0, 0, 0
//     20, 20, 229, 229, 74, 74, 10, 10
// };
// static const int16_t SQUARE_APPROX_TABLE3[8] = {
//     // 5, 37, 101, 154, 99, 51, 19, 3
//     // 0, 0, 0, 0, 0, 0, 0, 0
//     125, 125, 125, 125, 42, 42, 42, 42
// };
// static const int16_t SQUARE_APPROX_TABLE4[8] = {
//     // 5, 37, 101, 154, 99, 51, 19, 3
//     // 0, 0, 0, 0, 0, 0, 0, 0
//     84, 84, 84, 84, 84, 84, 84, 84
// };
static const int16_t SQUARE_FP8_TABLE[8] = {
    0, 0, 0, 1, 1, 1, 0,
};

static const uint32_t SQUARE_FP32_APPROX_TABLE[8] = {
    32768, 294912, 819200, 1296579, 802816, 409600, 147456, 16384
};


// E5_M10_16x16_comp_table = np.array([
//     [ 1,  3,   5,   7,   9,  11,  13,  15,  17, 19, 21, 23, 25, 27, 27, 13],
//     [ 3,  9,  15,  21,  27,  33,  39,  45,  51, 57, 63, 69, 74, 68, 44, 15],
//     [ 5, 15,  25,  35,  45,  55,  65,  75,  85, 95,105,110, 94, 68, 41, 14],
//     [ 7, 21,  35,  49,  63,  77,  91, 105, 119,132,134,113, 88, 63, 38, 13],
//     [ 9, 27,  45,  63,  81,  99, 117, 135, 151,149,127,104, 81, 58, 35, 12],
//     [11, 33,  55,  77,  99, 121, 143, 162, 157,137,116, 95, 74, 53, 32, 11],
//     [13, 39,  65,  91, 117, 143, 166, 162, 143,124,105, 86, 67, 48, 29, 10],
//     [15, 45,  75, 105, 135, 162, 162, 145, 128,111, 94, 77, 60, 43, 26,  9],
//     [17, 51,  85, 119, 151, 157, 143, 128, 113, 98, 83, 68, 53, 38, 23,  8],
//     [19, 57,  95, 132, 149, 137, 124, 111,  98, 85, 72, 59, 46, 33, 20,  7],
//     [21, 63, 105, 134, 127, 116, 105,  94,  83, 72, 61, 50, 39, 28, 17,  6],
//     [23, 69, 110, 113, 104,  95,  86,  77,  68, 59, 50, 41, 32, 23, 14,  5],
//     [25, 74,  94,  88,  81,  74,  67,  60,  53, 46, 39, 32, 25, 18, 11,  4],
//     [27, 68,  68,  63,  58,  53,  48,  43,  38, 33, 28, 23, 18, 13,  8,  3],
//     [27, 44,  41,  38,  35,  32,  29,  26,  23, 20, 17, 14, 11,  8,  5,  2],
//     [13, 15,  14,  13,  12,  11,  10,   9,   8,  7,  6,  5,  4,  3,  2,  0],
// ])

/*
E4_M3_comp_table = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
])
*/

/**
 * FP16 helper function: FP32 → FP16 bits
 * Using round-to-nearest-even (standard IEEE 754 rounding)
 */
inline uint16_t fp32_to_fp16_bits(float val) {
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(float));

    uint32_t sign = (f32 >> 16) & 0x8000;
    int32_t f32_expo = (f32 >> 23) & 0xFF;
    uint32_t f32_mant = f32 & 0x7FFFFF;

    // Special value: zero
    if (f32_expo == 0 && f32_mant == 0) {
        return static_cast<uint16_t>(sign);
    }

    // Special value: Inf/NaN
    if (f32_expo == 255) {
        if (f32_mant == 0) {
            return static_cast<uint16_t>(sign | 0x7C00); // Inf
        } else {
            return static_cast<uint16_t>(sign | 0x7C00 | (f32_mant >> 13)); // NaN
        }
    }

    // Convert exponent: FP32 bias=127, FP16 bias=15
    int32_t fp16_expo = f32_expo - 127 + 15;

    // Handle underflow
    if (fp16_expo <= 0) {
        // Subnormal or underflow to zero
        if (fp16_expo < -10) {
            return static_cast<uint16_t>(sign); // Too small, becomes zero
        }
        // Subnormal: right shift mantissa
        uint32_t mant = (f32_mant | 0x800000) >> (1 - fp16_expo + 13);
        // Rounding
        if (((f32_mant | 0x800000) >> (1 - fp16_expo + 12)) & 1) {
            mant++;
        }
        return static_cast<uint16_t>(sign | (mant & 0x3FF));
    }

    // Handle overflow
    if (fp16_expo >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00); // Inf
    }

    // Normal number: extract mantissa (high 10 bits of FP32)
    uint32_t fp16_mant = f32_mant >> 13;

    // Round to nearest even
    uint32_t round_bit = (f32_mant >> 12) & 1;
    uint32_t sticky_bits = f32_mant & 0xFFF;

    if (round_bit && (sticky_bits || (fp16_mant & 1))) {
        fp16_mant++;
        if (fp16_mant >= 1024) {
            fp16_expo++;
            fp16_mant = 0;
            if (fp16_expo >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00); // Overflow to Inf
            }
        }
    }

    return static_cast<uint16_t>(sign | (fp16_expo << 10) | fp16_mant);
}

/**
 * FP16 helper function: FP16 bits → FP32
 */
inline float fp16_bits_to_fp32(uint16_t fp16) {
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
        // Subnormal number
        expo = 1;
    }

    int32_t f32_expo = expo - 15 + 127;
    uint32_t f32_mant = mant << 13;

    uint32_t f32 = sign | (f32_expo << 23) | f32_mant;
    float result;
    std::memcpy(&result, &f32, sizeof(float));
    return result;
}

/**
 * FP8 helper function: FP32 → FP8 E4M3 bits
 * FP8 E4M3 format: 1 sign + 4 exponent + 3 mantissa
 * Exponent bias = 7
 */
inline uint8_t fp32_to_fp8_e4m3_bits(float val) {
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(float));

    uint32_t sign = (f32 >> 24) & 0x80;  // Sign bit to bit 7
    int32_t f32_expo = (f32 >> 23) & 0xFF;
    uint32_t f32_mant = f32 & 0x7FFFFF;

    // Special value: zero
    if (f32_expo == 0 && f32_mant == 0) {
        return static_cast<uint8_t>(sign);
    }

    // Special value: Inf/NaN → map to FP8 maximum value
    if (f32_expo == 255) {
        return static_cast<uint8_t>(sign | 0x7F);  // exp=15, mant=7
    }

    // Convert exponent: FP32 bias=127, FP8 E4M3 bias=7
    int32_t fp8_expo = f32_expo - 127 + 7;

    // Handle underflow
    if (fp8_expo <= 0) {
        // Subnormal or underflow to zero
        if (fp8_expo < -3) {
            return static_cast<uint8_t>(sign);  // Too small, becomes zero
        }
        // Subnormal: right shift mantissa (simplified to zero)
        return static_cast<uint8_t>(sign);
    }

    // Handle overflow
    if (fp8_expo >= 15) {
        return static_cast<uint8_t>(sign | 0x7F);  // Maximum value
    }

    // Normal number: extract mantissa (high 3 bits of FP32)
    uint32_t fp8_mant = f32_mant >> 20;  // 23-3=20

    // Round to nearest even
    uint32_t round_bit = (f32_mant >> 19) & 1;
    uint32_t sticky_bits = f32_mant & 0x7FFFF;  // Low 19 bits

    if (round_bit && (sticky_bits || (fp8_mant & 1))) {
        fp8_mant++;
        if (fp8_mant >= 8) {
            fp8_expo++;
            fp8_mant = 0;
            if (fp8_expo >= 15) {
                return static_cast<uint8_t>(sign | 0x7F);  // Overflow to maximum value
            }
        }
    }

    return static_cast<uint8_t>(sign | (fp8_expo << 3) | fp8_mant);
}

/**
 * FP8 helper function: FP8 E4M3 bits → FP32
 */
inline float fp8_e4m3_bits_to_fp32(uint8_t fp8) {
    uint32_t sign = (fp8 & 0x80) << 24;  // Sign bit from bit 7 to bit 31
    int32_t expo = (fp8 >> 3) & 0x0F;    // 4-bit exponent
    uint32_t mant = fp8 & 0x07;          // 3-bit mantissa

    // Zero value
    if (expo == 0 && mant == 0) {
        uint32_t f32 = sign;
        float result;
        std::memcpy(&result, &f32, sizeof(float));
        return result;
    }

    // Subnormal number (simplified to zero)
    if (expo == 0) {
        uint32_t f32 = sign;
        float result;
        std::memcpy(&result, &f32, sizeof(float));
        return result;
    }

    // Normal number: convert exponent and mantissa
    int32_t f32_expo = expo - 7 + 127;  // FP8 bias=7, FP32 bias=127
    uint32_t f32_mant = mant << 20;     // 3 bits expand to 23 bits

    uint32_t f32 = sign | (f32_expo << 23) | f32_mant;
    float result;
    std::memcpy(&result, &f32, sizeof(float));
    return result;
}

/**
 * FP8 helper function: FP32 → FP8 E2M5 bits
 * FP8 E2M5 format: 1 sign + 2 exponent + 5 mantissa
 * Exponent bias = 1
 */
inline uint8_t fp32_to_fp8_e2m5_bits(float val) {
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(float));

    uint32_t sign = (f32 >> 24) & 0x80;  // Sign bit to bit 7
    int32_t f32_expo = (f32 >> 23) & 0xFF;
    uint32_t f32_mant = f32 & 0x7FFFFF;

    // Special value: zero
    if (f32_expo == 0 && f32_mant == 0) {
        return static_cast<uint8_t>(sign);
    }

    // Special value: Inf/NaN → map to FP8 maximum value
    if (f32_expo == 255) {
        return static_cast<uint8_t>(sign | 0x7F);  // exp=3, mant=31
    }

    // Convert exponent: FP32 bias=127, FP8 E2M5 bias=1
    int32_t fp8_expo = f32_expo - 127 + 1;

    // Handle underflow
    if (fp8_expo <= 0) {
        if (fp8_expo < -5) {
            return static_cast<uint8_t>(sign);
        }
        return static_cast<uint8_t>(sign);
    }

    // Handle overflow
    if (fp8_expo >= 3) {
        return static_cast<uint8_t>(sign | 0x7F);  // Maximum value
    }

    // Normal number: extract mantissa (high 5 bits of FP32)
    uint32_t fp8_mant = f32_mant >> 18;  // 23-5=18

    // Round to nearest even
    uint32_t round_bit = (f32_mant >> 17) & 1;
    uint32_t sticky_bits = f32_mant & 0x1FFFF;  // Low 17 bits

    if (round_bit && (sticky_bits || (fp8_mant & 1))) {
        fp8_mant++;
        if (fp8_mant >= 32) {
            fp8_expo++;
            fp8_mant = 0;
            if (fp8_expo >= 3) {
                return static_cast<uint8_t>(sign | 0x7F);
            }
        }
    }

    return static_cast<uint8_t>(sign | (fp8_expo << 5) | fp8_mant);
}

/**
 * FP8 helper function: FP8 E2M5 bits → FP32
 */
inline float fp8_e2m5_bits_to_fp32(uint8_t fp8) {
    uint32_t sign = (fp8 & 0x80) << 24;  // Sign bit from bit 7 to bit 31
    int32_t expo = (fp8 >> 5) & 0x03;    // 2-bit exponent
    uint32_t mant = fp8 & 0x1F;          // 5-bit mantissa

    // Zero value
    if (expo == 0 && mant == 0) {
        uint32_t f32 = sign;
        float result;
        std::memcpy(&result, &f32, sizeof(float));
        return result;
    }

    // Subnormal number (simplified to zero)
    if (expo == 0) {
        uint32_t f32 = sign;
        float result;
        std::memcpy(&result, &f32, sizeof(float));
        return result;
    }

    // Normal number: convert exponent and mantissa
    int32_t f32_expo = expo - 1 + 127;  // FP8 bias=1, FP32 bias=127
    uint32_t f32_mant = mant << 18;     // 5 bits expand to 23 bits

    uint32_t f32 = sign | (f32_expo << 23) | f32_mant;
    float result;
    std::memcpy(&result, &f32, sizeof(float));
    return result;
}

/**
 * FP8 helper function: FP32 → FP8 E3M4 bits
 * FP8 E3M4 format: 1 sign + 3 exponent + 4 mantissa
 * Exponent bias = 3
 */
inline uint8_t fp32_to_fp8_e3m4_bits(float val) {
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(float));

    uint32_t sign = (f32 >> 24) & 0x80;  // Sign bit to bit 7
    int32_t f32_expo = (f32 >> 23) & 0xFF;
    uint32_t f32_mant = f32 & 0x7FFFFF;

    // Special value: zero
    if (f32_expo == 0 && f32_mant == 0) {
        return static_cast<uint8_t>(sign);
    }

    // Special value: Inf/NaN → map to FP8 maximum value
    if (f32_expo == 255) {
        return static_cast<uint8_t>(sign | 0x7F);  // exp=7, mant=15
    }

    // Convert exponent: FP32 bias=127, FP8 E3M4 bias=3
    int32_t fp8_expo = f32_expo - 127 + 3;

    // Handle underflow
    if (fp8_expo <= 0) {
        if (fp8_expo < -4) {
            return static_cast<uint8_t>(sign);
        }
        return static_cast<uint8_t>(sign);
    }

    // Handle overflow
    if (fp8_expo >= 7) {
        return static_cast<uint8_t>(sign | 0x7F);  // Maximum value
    }

    // Normal number: extract mantissa (high 4 bits of FP32)
    uint32_t fp8_mant = f32_mant >> 19;  // 23-4=19

    // Round to nearest even
    uint32_t round_bit = (f32_mant >> 18) & 1;
    uint32_t sticky_bits = f32_mant & 0x3FFFF;  // Low 18 bits

    if (round_bit && (sticky_bits || (fp8_mant & 1))) {
        fp8_mant++;
        if (fp8_mant >= 16) {
            fp8_expo++;
            fp8_mant = 0;
            if (fp8_expo >= 7) {
                return static_cast<uint8_t>(sign | 0x7F);
            }
        }
    }

    return static_cast<uint8_t>(sign | (fp8_expo << 4) | fp8_mant);
}

/**
 * FP8 helper function: FP8 E3M4 bits → FP32
 */
inline float fp8_e3m4_bits_to_fp32(uint8_t fp8) {
    uint32_t sign = (fp8 & 0x80) << 24;  // Sign bit from bit 7 to bit 31
    int32_t expo = (fp8 >> 4) & 0x07;    // 3-bit exponent
    uint32_t mant = fp8 & 0x0F;          // 4-bit mantissa

    // Zero value
    if (expo == 0 && mant == 0) {
        uint32_t f32 = sign;
        float result;
        std::memcpy(&result, &f32, sizeof(float));
        return result;
    }

    // Subnormal number (simplified to zero)
    if (expo == 0) {
        uint32_t f32 = sign;
        float result;
        std::memcpy(&result, &f32, sizeof(float));
        return result;
    }

    // Normal number: convert exponent and mantissa
    int32_t f32_expo = expo - 3 + 127;  // FP8 bias=3, FP32 bias=127
    uint32_t f32_mant = mant << 19;     // 4 bits expand to 23 bits

    uint32_t f32 = sign | (f32_expo << 23) | f32_mant;
    float result;
    std::memcpy(&result, &f32, sizeof(float));
    return result;
}

/**
 * FP16 FPMA square operation (with compensation table)
 *
 * Algorithm:
 * 1. Extract sign, expo, mant from FP16
 * 2. Handle subnormal numbers and special values
 * 3. FPMA computation: expo' = 2*expo - 15, mant' = 2*mant
 * 4. Add compensation: mant' += SQUARE_COMP_TABLE[mant]
 * 5. Handle carry and overflow
 */
inline float fp16_fpma_square(float val) {
    // Input is already FP16 precision, directly extract bits
    // Note: assume val has been converted by fp16_round()
    uint16_t fp16 = fp32_to_fp16_bits(val);

    // Extract fields
    int32_t expo = (fp16 >> 10) & 0x1F;
    int32_t mant = fp16 & 0x3FF;

    // Handle zero and subnormal numbers
    if (expo == 0) {
        if (mant == 0) {
            // Zero value
            return 0.0f;
        } else {
            // Subnormal number: very small value, underflows to zero after squaring
            // FP16 subnormal range: 2^-24 to 2^-14
            // After squaring: 2^-48 to 2^-28, much smaller than FP16 minimum normal 2^-14
            return 0.0f;
        }
    }

    // Handle Inf/NaN
    if (expo == 31) {
        if (mant == 0) {
            // Inf² = Inf
            return fp16_bits_to_fp32(0x7C00);
        } else {
            // NaN² = NaN
            return fp16_bits_to_fp32(0x7C00 | mant);
        }
    }

    // Normal number FPMA square (using complete 1024-entry compensation table)
    int32_t result_expo = expo + expo - 15;  // bias = 15
    int32_t mant_sum = mant + mant;

    // Step 1: Handle 2*mant overflow first
    if (mant_sum >= 1024) {
        result_expo++;
        mant_sum -= 1024;
        // mant_sum = mant_sum >> 1;
    }

    // Step 2: Use full mantissa to directly index 1024-entry compensation table (highest precision)
    // int32_t result_mant = mant_sum;
    // int32_t result_mant = mant_sum + SQUARE_COMP_TABLE[mant];
    int32_t result_mant = mant_sum + SQUARE_APPROX_TABLE[((mant>>7)&7)];

    // Step 3: Handle overflow caused by compensation
    if (result_mant >= 1024) {
        result_expo++;
        result_mant -= 1024;
        // result_mant = result_mant >> 1;
    }

    // Handle exponent underflow
    if (result_expo <= 0) {
        if (result_expo < -10) {
            // Too small, underflows to zero
            return 0.0f;
        }
        // Subnormal number: temporarily simplified to zero
        // Full implementation requires right-shifting mantissa, but impact on retrieval is minimal
        return 0.0f;
    }

    // Handle exponent overflow
    if (result_expo >= 31) {
        // Overflow to Inf
        return fp16_bits_to_fp32(0x7C00);
    }

    // Reconstruct FP16 (square result is always positive)
    uint16_t result_fp16 = static_cast<uint16_t>((result_expo << 10) | (result_mant & 0x3FF));

    // Convert back to FP32
    return fp16_bits_to_fp32(result_fp16);
}

// ====================== Fixed-Point Configuration ======================


/**
 * FP8 fixed-point configuration constants (bit width freely configurable, not limited to 8 bits)
 *
 * Q format: Q_m.n represents m integer bits + n fractional bits
 * Total bit width = m + n (can be arbitrarily configured, recommended ≤32 bits)
 *
 * Common configuration examples:
 * - Q2.6  (8-bit):  integer[0,3],      precision 1/64     ≈ 0.015625  ✓ Current config
 * - Q3.5  (8-bit):  integer[0,7],      precision 1/32     ≈ 0.03125
 * - Q4.8  (12-bit): integer[0,15],     precision 1/256    ≈ 0.0039
 * - Q6.10 (16-bit): integer[0,63],     precision 1/1024   ≈ 0.001
 * - Q8.12 (20-bit): integer[0,255],    precision 1/4096   ≈ 0.00024
 * - Q10.14(24-bit): integer[0,1023],   precision 1/16384  ≈ 0.00006
 *
 * Recommended configuration:
 * - Low width: Q2.6 (8-bit) or Q3.5 (8-bit)
 * - Medium width: Q6.10 (16-bit)
 * - High width: Q10.14 (24-bit)
 */
#define FP8_FIXED_POINT_INT_BITS 3      // Integer bits (freely adjustable)
#define FP8_FIXED_POINT_FRAC_BITS 13     // Fractional bits (freely adjustable)
#define FP8_FIXED_POINT_TOTAL_BITS (FP8_FIXED_POINT_INT_BITS + FP8_FIXED_POINT_FRAC_BITS)
#define FP8_FIXED_POINT_SCALE (1ULL << FP8_FIXED_POINT_FRAC_BITS)
#define FP8_FIXED_POINT_MAX ((1ULL << FP8_FIXED_POINT_TOTAL_BITS) - 1)

// Bit width validation: ensure total bit width does not exceed 64 bits (actually recommend ≤32 bits)
static_assert(FP8_FIXED_POINT_TOTAL_BITS <= 64,
              "FP8 total bits must not exceed 64");

// Compatibility macro (pointing to FP16 configuration)
#define FIXED_POINT_INT_BITS FP16_FIXED_POINT_INT_BITS
#define FIXED_POINT_FRAC_BITS FP16_FIXED_POINT_FRAC_BITS
#define FP16_FIXED_POINT_SCALE (1ULL << FP16_FIXED_POINT_FRAC_BITS)
#define FP16_FIXED_POINT_MAX ((1ULL << FP16_FIXED_POINT_TOTAL_BITS) - 1)

/**
 * FP16 FPMA square operation, quantized to fixed-point (Q-format, configurable bit width)
 *
 * Fixed-point format: Q_m.n (m integer bits + n fractional bits)
 * - Total bit width = m + n (freely configurable, recommended ≤32 bits)
 * - Value range: [0, 2^(m+n) - 1] / 2^n
 * - Precision: 1/2^n
 *
 * Algorithm:
 * 1. Compute (a-b)^2 using FPMA approximation
 * 2. Quantize to fixed-point: quantized = round(value * 2^n)
 * 3. Saturate: clamp to [0, 2^(m+n)-1]
 *
 * Return: uint32_t generic type (supports ≤32-bit fixed-point)
 */
inline uint32_t fp16_fpma_square_quan(float a, float b) {
    // Inline fp16_round: FP32 -> FP16 -> FP32
    auto inline_fp16_round = [](float val) {
        uint16_t fp16_bits = fp32_to_fp16_bits(val);
        return fp16_bits_to_fp32(fp16_bits);
    };

    //scale query first
    // a *= (1<<QUERY_SCALE_M);
    // a *= powf(2.0f, QUERY_SCALE_M);

    //quant
    uint16_t a_16bits = fp32_to_fp16_bits(a);
    uint16_t b_16bits = fp32_to_fp16_bits(b);

    // experiment
    // b_16bits = b_16bits & 0xFFFE;
    // b_16bits = b_16bits & 0xFFFC;
    // b_16bits = b_16bits & 0xFFF8;
    // b_16bits = b_16bits & 0xFFF0;
    // b_16bits = b_16bits & 0xFFE0;
    // b_16bits = b_16bits & 0xFFC0;
    // b_16bits = b_16bits & 0xFF80;
    // b_16bits = b_16bits & 0xFF00;
    // b_16bits = b_16bits & 0xFE00;
    // b_16bits = b_16bits & 0xFC00;


    // if((a_16bits & 0x8000) == (b_16bits & 0x8000)){
    //     // a_16bits = a_16bits & 0xFFF0;
    //     // b_16bits = b_16bits & 0xFFF0;

    //     // mean compensation
    //     // b_16bits = (b_16bits & 0xFFF0) + 0x0008;
    // }
    // else{
    //     // b_16bits = (b_16bits & 0xFFF0) + (a_16bits & 0x000F);

    //     // mean compensation
    // }

    // b_16bits = (b_16bits & 0xFFF0) + 0x0008;
    // b_16bits = (b_16bits & 0xFFC0) + 0x0020;

    float a_quan_fp32 = fp16_bits_to_fp32(a_16bits);
    float b_quan_fp32 = fp16_bits_to_fp32(b_16bits);

    // Method 1: compute (a-b)^2 using FPMA approximation
    float diff = inline_fp16_round(a_quan_fp32 - b_quan_fp32);
    float square = fp16_fpma_square(diff);
    // float square = diff*diff;
    // float square = (a_quan_fp32 - b_quan_fp32)*(a_quan_fp32 - b_quan_fp32);

    // Method 2: quantize to fixed-point Q_m.n
    // quantized = round(square * 2^n)
    float scaled = square * FP16_FIXED_POINT_SCALE;

    // Round and saturate
    uint64_t rounded;
    if (scaled >= FP16_FIXED_POINT_MAX) {
        rounded = FP16_FIXED_POINT_MAX;
    } else if (scaled <= 0.0f) {
        rounded = 0;
    } else {
        rounded = static_cast<uint64_t>(scaled + 0.5f);
    }

    return static_cast<uint32_t>(rounded);
}


// ====================== Precision Type Enum ======================
enum class PrecisionType {
    FP32,          // Standard FP32
    FP32_FPMA,     // FP32 FPMA square optimization (with compensation table)
    FP16_TRUE,     // True FP16 (round each step)
    FP16_FPMA,     // FP16 FPMA square optimization (with compensation table)
    FP8_E4M3,      // FP8-E4M3 (standard rounding)
    FP8_FPMA,      // FP8 E4M3 FPMA square optimization (fixed-point quantization)
    FP8_E2M5,      // FP8-E2M5 (standard rounding)
    FP8_FPMA_E2M5, // FP8 E2M5 FPMA square optimization
    FP8_E3M4,      // FP8-E3M4 (standard rounding)
    FP8_FPMA_E3M4, // FP8 E3M4 FPMA square optimization
    INT16_DIFF_FP16_FPMA, // Integer-source diff widened to int16, squared via fp16-style compensation
    INT16,         // INT16 quantization
    INT8           // INT8 quantization
};

// ====================== Helper Functions: Precision Conversion ======================

// FP16 simulated rounding - using unified bits conversion
inline float fp16_round(float val) {
    // Use same conversion function as FPMA to ensure consistency
    uint16_t fp16_bits = fp32_to_fp16_bits(val);
    return fp16_bits_to_fp32(fp16_bits);
}

// FP32 -> INT16
inline int16_t quantize_int16(float val, float scale) {
    int v = static_cast<int>(std::round(val / scale));
    return static_cast<int16_t>(std::max(-32767, std::min(32767, v)));
}

// FP32 -> INT8
inline int8_t quantize_int8(float val, float scale) {
    int v = static_cast<int>(std::round(val / scale));
    return static_cast<int8_t>(std::max(-127, std::min(127, v)));
}

/**
 * Dequantize fixed-point value to float (FP16 FPMA, supports arbitrary bit width)
 *
 * Fixed-point format: Q_m.n (m integer bits + n fractional bits)
 * Dequantization formula: value = quantized / 2^n
 *
 * Parameters:
 * - quantized: fixed-point value (uint32_t, supports ≤32-bit)
 *
 * Return: FP32 floating-point value
 */
inline float dequantize_fpma(uint32_t quantized) {
    // Simple division: value = quantized / 2^n
    return static_cast<float>(quantized) / FP16_FIXED_POINT_SCALE;
}

/**
 * Dequantize fixed-point value to float (FP8 FPMA, supports arbitrary bit width)
 *
 * Fixed-point format: Q_m.n (m integer bits + n fractional bits)
 * Dequantization formula: value = quantized / 2^n
 *
 * Parameters:
 * - quantized: fixed-point value (uint32_t, supports ≤32-bit)
 *
 * Return: FP32 floating-point value
 */
inline float dequantize_fpma_fp8(uint32_t quantized) {
    // Simple division: value = quantized / 2^n
    return static_cast<float>(quantized) / FP8_FIXED_POINT_SCALE;
}

/**
 * Integer diff square approximation for raw integer-source datasets.
 *
 * Workflow:
 * 1. Compute integer diff c = a - b
 * 2. Widen c to int16 range
 * 3. Use leading-1 to form an fp16-style exponent/mantissa pair
 * 4. Reuse the current 8-entry fp16 square compensation table
 */
inline float int16_diff_fp16_fpma_square_from_diff(int diff) {
    int clipped_diff = std::max(-32768, std::min(32767, diff));
    int magnitude = clipped_diff == -32768 ? 32768 : std::abs(clipped_diff);

    if (magnitude == 0) {
        return 0.0f;
    }

    int leading_pos = 31 - __builtin_clz(static_cast<unsigned int>(magnitude));
    int input_expo = leading_pos + 15;
    int leading_value = 1 << leading_pos;
    int remainder = magnitude - leading_value;
    int input_mant = 0;

    if (leading_pos <= 10) {
        input_mant = remainder << (10 - leading_pos);
    } else {
        input_mant = remainder >> (leading_pos - 10);
    }
    input_mant &= 0x03FF;

    int result_expo = input_expo + input_expo - 15;
    int mant_sum = input_mant + input_mant;
    if (mant_sum >= 1024) {
        result_expo++;
        mant_sum -= 1024;
    }

    int result_mant = mant_sum + SQUARE_APPROX_TABLE[(input_mant >> 7) & 0x07];
    if (result_mant >= 1024) {
        result_expo++;
        result_mant -= 1024;
    }

    if (result_expo <= 0) {
        return 0.0f;
    }
    if (result_expo >= 31) {
        return fp16_bits_to_fp32(0x7C00);
    }

    uint16_t result_fp16 = static_cast<uint16_t>((result_expo << 10) | (result_mant & 0x03FF));
    return fp16_bits_to_fp32(result_fp16);
}

// ====================== FP8 FPMA related functions ======================

/**
 * FP8 FPMA square operation (in FP32 domain, using high 3 bits to index FP8 compensation table)
 *
 * Algorithm: similar to FP16 FPMA, but:
 * 1. Extract FP32 expo and mant (full 23-bit mantissa)
 * 2. Use high 3 bits of mant (corresponding to FP8 precision) to index compensation table
 * 3. Perform FPMA computation in FP32 domain (preserve precision)
 * 4. Reconstruct FP32 result
 *
 * Note: avoid quantizing to FP8 too early, otherwise precision loss is large
 */
inline float fp8_fpma_square(float val) {
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(float));

    // Extract FP32 fields
    int32_t expo = (f32 >> 23) & 0xFF;
    uint32_t mant = f32 & 0x7FFFFF;

    // Handle zero and subnormal numbers
    if (expo == 0) {
        return 0.0f;
    }

    // Handle Inf/NaN
    if (expo == 255) {
        if (mant == 0) {
            // Inf² = Inf
            uint32_t inf = 0x7F800000;
            float result;
            std::memcpy(&result, &inf, sizeof(float));
            return result;
        } else {
            // NaN² = NaN
            return val;
        }
    }

    // Normal FPMA square (using FP8-precision compensation table)
    int32_t result_expo = expo + expo - 127;  // bias = 127
    uint32_t mant_sum = mant + mant;

    // Step 1: handle overflow of 2*mant
    if (mant_sum >= 0x800000) {  // FP32: 23-bit mantissa, threshold 2^23
        result_expo++;
        mant_sum -= 0x800000;
    }

    // Step 2: use FP8 compensation table (index by high 3 bits of mant)
    // FP32 mant is 23 bits, take high 3 bits corresponding to FP8 3-bit mantissa
    int mant_index = (mant >> 20) & 0x07;

    // Compensation scaling: FP8 table is for 3-bit mantissa and needs scaling to 23 bits
    // SQUARE_FP8_TABLE[i] is compensation under 3-bit mantissa, scaled to 23 bits:
    // compensation_fp32 = SQUARE_FP8_TABLE[i] << (23 - 3) = SQUARE_FP8_TABLE[i] << 20
    // uint32_t result_mant = mant_sum + (static_cast<uint32_t>(SQUARE_FP8_TABLE[mant_index]) << 20) +(static_cast<uint32_t>(SQUARE_APPROX_TABLE[mant_index]) << 13);
    uint32_t result_mant = mant_sum + (static_cast<uint32_t>(SQUARE_FP8_TABLE[mant_index]) << 20);

    // Step 3: handle overflow caused by compensation
    while (result_mant >= 0x800000) {  // FP32: 23-bit mantissa, threshold 2^23
        result_expo++;
        result_mant -= 0x800000;
    }

    // Handle exponent underflow
    if (result_expo <= 0) {
        return 0.0f;
    }

    // Handle exponent overflow
    if (result_expo >= 255) {
        uint32_t inf = 0x7F800000;
        float result;
        std::memcpy(&result, &inf, sizeof(float));
        return result;
    }

    // Reconstruct FP32 (square result is always non-negative)
    uint32_t result_f32 = (result_expo << 23) | (result_mant & 0x7FFFFF);
    float result;
    std::memcpy(&result, &result_f32, sizeof(float));
    return result;
}

/**
 * FP8 FPMA square operation, quantized to fixed-point (Q-format, configurable bit width)
 *
 * Fixed-point format: Q_m.n (m integer bits + n fractional bits)
 * - Total bit width = m+n (freely configurable, recommended ≤32 bits)
 * - Value range: [0, 2^(m+n) - 1] / 2^n
 * - Precision: 1/2^n
 *
 * Algorithm:
 * 1. Compute (a-b)^2 using FP8 FPMA approximation
 * 2. Quantize to fixed-point: quantized = round(value * 2^n)
 * 3. Saturate: clamp to [0, 2^(m+n)-1]
 *
 * Return: uint32_t generic type (supports ≤32-bit fixed-point)
 */
inline uint32_t fp8_fpma_square_quan(float a, float b);  // Forward declaration

// FP32 -> FP8 E4M3 rounding
inline float fp8_e4m3_round(float val) {
    // FP8 E4M3 format: 1 sign + 4 exponent + 3 mantissa
    // Exponent bias = 7
    // Normalized form: ±1.mmm × 2^(exp - 7), where mmm is a 3-bit fraction
    // Range: max = 448 (1.111 × 2^8), min_normal = 0.015625 (1.000 × 2^-6)

    if (val == 0.0f) return 0.0f;

    const float MAX_FP8 = 448.0f;
    const float MIN_NORMAL_FP8 = 0.015625f;  // 2^-6

    // Handle sign
    float sign = (val < 0) ? -1.0f : 1.0f;
    float abs_val = std::fabs(val);

    // Clip to FP8 range
    if (abs_val > MAX_FP8) {
        return sign * MAX_FP8;
    }

    if (abs_val < MIN_NORMAL_FP8) {
        // Subnormal numbers - simplified handling: directly map to 0
        return 0.0f;
    }

    // Extract exponent and mantissa
    int exp_frexp;
    float mant_frexp = std::frexp(abs_val, &exp_frexp);
    // frexp returns: abs_val = mant_frexp × 2^exp_frexp, where mant_frexp ∈ [0.5, 1)

    // Convert to normalized form: 1.xxx × 2^exp_normalized
    // mant_frexp ∈ [0.5, 1) → mant_normalized ∈ [1, 2)
    float mant_normalized = mant_frexp * 2.0f;  // [1.0, 2.0)
    int exp_normalized = exp_frexp - 1;

    // Convert to FP8 E4M3 biased exponent
    // FP8 exponent range: -6 to 8 (biased: 1 to 15, 0 reserved)
    int fp8_exp_biased = exp_normalized + 7;

    if (fp8_exp_biased <= 0) {
        // Underflow
        return 0.0f;
    }
    if (fp8_exp_biased > 15) {
        // Overflow
        return sign * MAX_FP8;
    }

    // Round mantissa to 3 bits
    // mant_normalized ∈ [1.0, 2.0) represented as 1.mmm, where mmm is a 3-bit fraction
    // Extract fractional part: mant_normalized - 1.0 ∈ [0, 1)
    float fraction = mant_normalized - 1.0f;  // [0.0, 1.0)

    // 3 bits give 8 levels: 0/8, 1/8, 2/8, ..., 7/8
    float scaled_fraction = fraction * 8.0f;  // [0.0, 8.0)
    int rounded_fraction = static_cast<int>(std::round(scaled_fraction));

    // Handle carry into integer part
    if (rounded_fraction >= 8) {
        // Carry: 1.111 → 10.000 = 2.0
        rounded_fraction = 0;
        fp8_exp_biased++;
        if (fp8_exp_biased > 15) {
            return sign * MAX_FP8;
        }
    }

    // Reconstruct mantissa: 1.mmm
    float final_mantissa = 1.0f + static_cast<float>(rounded_fraction) / 8.0f;

    // Reconstruct float: mantissa × 2^(exp_biased - 7)
    int final_exp = fp8_exp_biased - 7;
    float result = std::ldexp(final_mantissa, final_exp);

    return sign * result;
}

// FP32 -> FP8 E2M5 rounding
inline float fp8_e2m5_round(float val) {
    // FP8 E2M5 format: 1 sign + 2 exponent + 5 mantissa
    // Exponent bias = 1
    // Normalized form: ±1.mmmmm × 2^(exp - 1), where mmmmm is a 5-bit fraction
    // Range: max = 6.0 (1.11111 × 2^2), min_normal = 0.5 (1.00000 × 2^-1)

    if (val == 0.0f) return 0.0f;

    const float MAX_FP8 = 6.0f;
    const float MIN_NORMAL_FP8 = 0.5f;  // 2^-1

    // Handle sign
    float sign = (val < 0) ? -1.0f : 1.0f;
    float abs_val = std::fabs(val);

    // Clip to FP8 range
    if (abs_val > MAX_FP8) {
        return sign * MAX_FP8;
    }

    if (abs_val < MIN_NORMAL_FP8) {
        return 0.0f;
    }

    // Extract exponent and mantissa
    int exp_frexp;
    float mant_frexp = std::frexp(abs_val, &exp_frexp);
    float mant_normalized = mant_frexp * 2.0f;
    int exp_normalized = exp_frexp - 1;

    // Convert to FP8 E2M5 biased exponent
    // FP8 exponent range: -1 to 2 (biased: 1 to 3, 0 reserved)
    int fp8_exp_biased = exp_normalized + 1;

    if (fp8_exp_biased <= 0) {
        return 0.0f;
    }
    if (fp8_exp_biased > 3) {
        return sign * MAX_FP8;
    }

    // Round mantissa to 5 bits
    float fraction = mant_normalized - 1.0f;
    float scaled_fraction = fraction * 32.0f;  // [0.0, 32.0)
    int rounded_fraction = static_cast<int>(std::round(scaled_fraction));

    // Handle carry
    if (rounded_fraction >= 32) {
        rounded_fraction = 0;
        fp8_exp_biased++;
        if (fp8_exp_biased > 3) {
            return sign * MAX_FP8;
        }
    }

    // Reconstruct mantissa and result
    float final_mantissa = 1.0f + static_cast<float>(rounded_fraction) / 32.0f;
    int final_exp = fp8_exp_biased - 1;
    float result = std::ldexp(final_mantissa, final_exp);

    return sign * result;
}

// FP32 -> FP8 E3M4 rounding
inline float fp8_e3m4_round(float val) {
    // FP8 E3M4 format: 1 sign + 3 exponent + 4 mantissa
    // Exponent bias = 3
    // Normalized form: ±1.mmmm × 2^(exp - 3), where mmmm is a 4-bit fraction
    // Range: max = 30.0 (1.1111 × 2^4), min_normal = 0.125 (1.0000 × 2^-3)

    if (val == 0.0f) return 0.0f;

    const float MAX_FP8 = 30.0f;
    const float MIN_NORMAL_FP8 = 0.125f;  // 2^-3

    // Handle sign
    float sign = (val < 0) ? -1.0f : 1.0f;
    float abs_val = std::fabs(val);

    // Clip to FP8 range
    if (abs_val > MAX_FP8) {
        return sign * MAX_FP8;
    }

    if (abs_val < MIN_NORMAL_FP8) {
        // Subnormal numbers - simplified handling: directly map to 0
        return 0.0f;
    }

    // Extract exponent and mantissa
    int exp_frexp;
    float mant_frexp = std::frexp(abs_val, &exp_frexp);
    float mant_normalized = mant_frexp * 2.0f;
    int exp_normalized = exp_frexp - 1;

    // Convert to FP8 E3M4 biased exponent
    // FP8 exponent range: -3 to 4 (biased: 1 to 7, 0 reserved)
    int fp8_exp_biased = exp_normalized + 3;

    if (fp8_exp_biased <= 0) {
        return 0.0f;
    }
    if (fp8_exp_biased > 7) {
        return sign * MAX_FP8;
    }

    // Round mantissa to 4 bits
    float fraction = mant_normalized - 1.0f;
    float scaled_fraction = fraction * 16.0f;  // [0.0, 16.0)
    int rounded_fraction = static_cast<int>(std::round(scaled_fraction));

    // Handle carry
    if (rounded_fraction >= 16) {
        rounded_fraction = 0;
        fp8_exp_biased++;
        if (fp8_exp_biased > 7) {
            return sign * MAX_FP8;
        }
    }

    // Reconstruct mantissa and result
    float final_mantissa = 1.0f + static_cast<float>(rounded_fraction) / 16.0f;
    int final_exp = fp8_exp_biased - 3;
    float result = std::ldexp(final_mantissa, final_exp);

    return sign * result;
}

/**
 * FP8 FPMA square operation, quantized to fixed-point (actual implementation)
 */
inline uint32_t fp8_fpma_square_quan(float a, float b) {
    // Method 1: compute (a-b)^2 using FP8 FPMA approximation
    float diff = fp8_e4m3_round(a - b);
    float square = fp8_fpma_square(diff);

    // Method 2: quantize to fixed-point Q_m.n
    // quantized = round(square * 2^n)
    float scaled = square * FP8_FIXED_POINT_SCALE;

    // Round and saturate
    uint64_t rounded;
    if (scaled >= FP8_FIXED_POINT_MAX) {
        rounded = FP8_FIXED_POINT_MAX;
    } else if (scaled <= 0.0f) {
        rounded = 0;
    } else {
        rounded = static_cast<uint64_t>(scaled + 0.5f);
    }

    return static_cast<uint32_t>(rounded);
}

/**
 * FP8 E2M5 FPMA square operation, quantized to fixed-point
 */
inline uint32_t fp8_fpma_square_quan_e2m5(float a, float b) {
    // Method 1: compute (a-b)^2 using FP8 E2M5 FPMA approximation
    float diff = fp8_e2m5_round(a - b);
    float square = fp8_fpma_square(diff);

    // Method 2: quantize to fixed-point Q_m.n
    float scaled = square * FP8_FIXED_POINT_SCALE;

    // Round and saturate
    uint64_t rounded;
    if (scaled >= FP8_FIXED_POINT_MAX) {
        rounded = FP8_FIXED_POINT_MAX;
    } else if (scaled <= 0.0f) {
        rounded = 0;
    } else {
        rounded = static_cast<uint64_t>(scaled + 0.5f);
    }

    return static_cast<uint32_t>(rounded);
}

/**
 * FP8 E3M4 FPMA square operation, quantized to fixed-point
 */
inline uint32_t fp8_fpma_square_quan_e3m4(float a, float b) {
    // Method 1: compute (a-b)^2 using FP8 E3M4 FPMA approximation
    float diff = fp8_e3m4_round(a - b);
    float square = fp8_fpma_square(diff);

    // Method 2: quantize to fixed-point Q_m.n
    float scaled = square * FP8_FIXED_POINT_SCALE;

    // Round and saturate
    uint64_t rounded;
    if (scaled >= FP8_FIXED_POINT_MAX) {
        rounded = FP8_FIXED_POINT_MAX;
    } else if (scaled <= 0.0f) {
        rounded = 0;
    } else {
        rounded = static_cast<uint64_t>(scaled + 0.5f);
    }

    return static_cast<uint32_t>(rounded);
}

// ====================== Dynamic-precision distance functions ======================

/**
 * FP32 reference L2 distance
 */
static float L2Sqr_FP32(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = pVect1[i] - pVect2[i];
        res += t * t;
    }
    return res;
}

/**
 * FP32 FPMA square operation (with 8-entry compensation table).
 *
 * This mirrors fp16_fpma_square in FP32 bit fields:
 * result_exp = exp + exp - bias, result_mant = 2 * mant + table[high3(mant)].
 */
inline float fp32_fpma_square(float val) {
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(float));

    int32_t expo = (f32 >> 23) & 0xFF;
    uint32_t mant = f32 & 0x7FFFFF;

    if (expo == 0) {
        return 0.0f;
    }

    if (expo == 255) {
        if (mant == 0) {
            uint32_t inf = 0x7F800000;
            float result;
            std::memcpy(&result, &inf, sizeof(float));
            return result;
        }
        return val;
    }

    int32_t result_expo = expo + expo - 127;
    uint32_t result_mant = mant + mant;

    if (result_mant >= 0x800000) {
        result_expo++;
        result_mant -= 0x800000;
    }

    result_mant += SQUARE_FP32_APPROX_TABLE[(mant >> 20) & 0x07];
    while (result_mant >= 0x800000) {
        result_expo++;
        result_mant -= 0x800000;
    }

    if (result_expo <= 0) {
        return 0.0f;
    }

    if (result_expo >= 255) {
        uint32_t inf = 0x7F800000;
        float result;
        std::memcpy(&result, &inf, sizeof(float));
        return result;
    }

    uint32_t result_f32 = (static_cast<uint32_t>(result_expo) << 23) | (result_mant & 0x7FFFFF);
    float result;
    std::memcpy(&result, &result_f32, sizeof(float));
    return result;
}

static float L2Sqr_FP32_FPMA(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float res = 0.0f;
    for (size_t i = 0; i < qty; i++) {
        float diff = pVect1[i] - pVect2[i];
        res += fp32_fpma_square(diff);
    }
    return res;
}

/**
 * FP16 true-computation L2 distance
 * Note: storage is still FP32; conversion happens only during computation
 */
static float L2Sqr_FP16_Dynamic(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float res = 0.0f;

    for (size_t i = 0; i < qty; i++) {
        // Read FP32 values and dynamically convert to FP16
        float v1_fp32 = pVect1[i];
        float v2_fp32 = pVect2[i];

        // Simulate FP16 rounding
        float v1 = fp16_round(v1_fp32);
        float v2 = fp16_round(v2_fp32);

        // FP16 subtraction
        float diff = fp16_round(v1 - v2);

        // FP16 square
        float square = fp16_round(diff * diff);

        // Accumulate
        res += square;
    }
    return res;
}

/**
 * FP16 FPMA-optimized L2 distance (with square compensation table)
 *
 * Advantages:
 * - Uses table-based compensation, faster than standard FP16 multiplication
 * - Square operation accuracy close to standard FP16
 * - Suitable for hardware acceleration (FPMA units)
 *
 * Note: storage is still FP32; FPMA is used only during computation
 */
// static float L2Sqr_FP16_FPMA(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     float *pVect2 = (float *)pVect2v;
//     size_t qty = *((size_t *)qty_ptr);

//     float res = 0.0f;
//
//     for (size_t i = 0; i < qty; i++) {
//         // Read FP32 values and convert to FP16
//         float v1 = fp16_round(pVect1[i]);
//         float v2 = fp16_round(pVect2[i]);
//
//         // FP16 subtraction (with standard rounding)
//         float diff = fp16_round(v1 - v2);
//
//         // FP16 FPMA square (with compensation table)
//         float square = fp16_fpma_square(diff);

//         // Accumulate
//         res += square;
//     }
//     return res;
// }
// Forward declaration
static float L2Sqr_FP16_FPMA_Quantized(const void *pVect1v, const void *pVect2v, const void *qty_ptr);

/**
 * FP16 FPMA-optimized L2 distance
 *
 * Choose implementation via USE_FPMA_QUANTIZED:
 * - 1: quantized version (Q-format fixed-point with 32-bit accumulator)
 * - 0: floating-point version (faster, higher precision)
 */
static float L2Sqr_FP16_FPMA(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
#if USE_FPMA_QUANTIZED
    // Quantized version: use Q-format fixed-point
    return L2Sqr_FP16_FPMA_Quantized(pVect1v, pVect2v, qty_ptr);
#else
    // Floating-point version
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float res = 0.0f;

    for (size_t i = 0; i < qty; i++) {
        // Read FP32 values and convert to FP16
        float v1 = fp16_round(pVect1[i]);
        float v2 = fp16_round(pVect2[i]);

        // FP16 subtraction
        float diff = fp16_round(v1 - v2);

        // FP16 FPMA square (direct floating-point computation, no quantization)
        float square = fp16_fpma_square(diff);

        // Accumulate
        res += square;
    }

    return res;
#endif
}


/**
 * FP16 FPMA 16-bit fixed-point quantized version (Q-format)
 *
 * Strategy:
 * 1. For each dimension, compute (a-b)^2 and quantize to 16-bit fixed-point
 * 2. Accumulate fixed-point values in a 32-bit accumulator (avoid overflow)
 * 3. Dequantize the accumulated result at the end
 *
 * Advantages:
 * - True fixed-point computation
 * - Configurable fractional bit position
 * - Accumulation without precision loss
 */
static float L2Sqr_FP16_FPMA_Quantized(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    // Use 64-bit accumulator (supports accumulation of arbitrary-width fixed-point values)
    uint64_t accumulator = 0;

    // multiplier width
    int multiplier_width = 1600;
    // int multiplier_width = 1;

    float res_quantized = 0.0f;

    for (size_t i = 0; i < qty; i++) {
        float v1 = fp16_round(pVect1[i]);
        float v2 = fp16_round(pVect2[i]);

        // Quantize to fixed-point Q_m.n (configurable bit width)
        uint32_t quantized_square = fp16_fpma_square_quan(v1, v2);

        // Accumulate into 64-bit accumulator
        accumulator += quantized_square;
        if((i+1) % multiplier_width == 0 || i == qty-1){
            res_quantized += static_cast<float>(accumulator) / FP16_FIXED_POINT_SCALE;
            accumulator = 0;
        }
    }

    return res_quantized;
    // return (res_quantized - 1)/powf(2.0f, QUERY_SCALE_M) + 2 - powf(2.0f, QUERY_SCALE_M);
    // return 0;
}

// a *= (1<<QUERY_SCALE_M);
    // a *= powf(2.0f, QUERY_SCALE_M);

/**
 * INT16-diff + fp16-style compensation-table square approximation.
 *
 * This path is intended for raw integer-source datasets stored as FP32.
 * Each input is rounded to the nearest integer, diff is widened to int16,
 * and the square is approximated with the existing 8-entry fp16 table.
 */
static float L2Sqr_INT16_DIFF_FP16_FPMA(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float res = 0.0f;
    for (size_t i = 0; i < qty; i++) {
        int left_value = static_cast<int>(std::lrint(pVect1[i]));
        int right_value = static_cast<int>(std::lrint(pVect2[i]));
        int diff = left_value - right_value;
        res += int16_diff_fp16_fpma_square_from_diff(diff);
    }
    return res;
}

/**
 * INT16 quantized L2 distance
 * Dynamic quantization: quantize first for each computation, then use integer arithmetic
 */
static float L2Sqr_INT16_Dynamic(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    // Optionally compute quantization scale dynamically
    // float max_val = 0.0f;
    // for (size_t i = 0; i < qty; i++) {
    //     max_val = std::max(max_val, std::fabs(pVect1[i]));
    //     max_val = std::max(max_val, std::fabs(pVect2[i]));
    // }

    // Static scale
    float max_val = 1.0f;
    float scale = (max_val > 0) ? (max_val / 32767.0f) : 1.0f;

    float res = 0.0f;
    for (size_t i = 0; i < qty; i++) {
        // Dynamically quantize to INT16
        int16_t v1_int16 = quantize_int16(pVect1[i], scale);
        int16_t v2_int16 = quantize_int16(pVect2[i], scale);

        // INT16 arithmetic
        int diff = static_cast<int>(v1_int16) - static_cast<int>(v2_int16);
        int square = diff * diff;

        // Dequantize
        res += static_cast<float>(square) * scale * scale;
    }
    return res;
}

/**
 * INT8 quantized L2 distance
 */
static float L2Sqr_INT8_Dynamic(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    // Optionally compute quantization scale dynamically
    // float max_val = 0.0f;
    // for (size_t i = 0; i < qty; i++) {
    //     max_val = std::max(max_val, std::fabs(pVect1[i]));
    //     max_val = std::max(max_val, std::fabs(pVect2[i]));
    // }

    // Static scale
    float max_val = 1.0f;
    float scale = (max_val > 0) ? (max_val / 127.0f) : 1.0f;

    float res = 0.0f;
    for (size_t i = 0; i < qty; i++) {
        // Dynamically quantize to INT8
        int8_t v1_int8 = quantize_int8(pVect1[i], scale);
        int8_t v2_int8 = quantize_int8(pVect2[i], scale);

        // INT8 arithmetic
        int diff = static_cast<int>(v1_int8) - static_cast<int>(v2_int8);
        int square = diff * diff;

        // Dequantize
        res += static_cast<float>(square) * scale * scale;
    }
    return res;
}

/**
 * FP8 E4M3 rounded L2 distance (with dynamic scaling)
 * Note: storage is still FP32; values are converted to FP8 E4M3 only during computation
 *
 * Strategy:
 * 1. Dynamically compute a scaling factor to map values into the best FP8 range
 * 2. Round only the input vectors; intermediate computations use FP32 precision
 * 3. Rescale the final result back
 *
 * Motivation: normalized vectors ([-0.3, 0.3]) have insufficient precision in FP8; scaling improves this
 */
static float L2Sqr_FP8_E4M3_Dynamic(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    // Optionally compute scaling factor dynamically
    // Goal: scale values into the best FP8 E4M3 range (around 1-100)
    // float max_val = 0.0f;
    // for (size_t i = 0; i < qty; i++) {
    //     max_val = std::max(max_val, std::fabs(pVect1[i]));
    //     max_val = std::max(max_val, std::fabs(pVect2[i]));
    // }

    // static scale
    float max_val = 1.0f;

    // Scaling factor: map max_val to 100 (FP8 "sweet spot")
    // For normalized vectors (max ~0.3), scale ~0.003 and scaled values ~100
    float scale = (max_val > 1e-6f) ? (max_val / 100.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    float res = 0.0f;

    for (size_t i = 0; i < qty; i++) {
        // Scale and convert to FP8 E4M3
        float v1_scaled = pVect1[i] * inv_scale;
        float v2_scaled = pVect2[i] * inv_scale;

        float v1 = fp8_e4m3_round(v1_scaled);
        float v2 = fp8_e4m3_round(v2_scaled);

        // Compute difference and square in FP32 precision
        float diff = v1 - v2;
        float square = diff * diff;

        // Accumulate
        res += square;
    }

    // Rescale back: distance_original = distance_scaled × scale²
    return res * scale * scale;
}

/**
 * FP8 E2M5 rounded L2 distance (with dynamic scaling)
 */
static float L2Sqr_FP8_E2M5_Dynamic(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float max_val = 1.0f;
    float scale = (max_val > 1e-6f) ? (max_val / 3.0f) : 1.0f;  // E2M5 range: [0.5, 6.0]
    float inv_scale = 1.0f / scale;

    float res = 0.0f;

    for (size_t i = 0; i < qty; i++) {
        float v1_scaled = pVect1[i] * inv_scale;
        float v2_scaled = pVect2[i] * inv_scale;

        float v1 = fp8_e2m5_round(v1_scaled);
        float v2 = fp8_e2m5_round(v2_scaled);

        float diff = v1 - v2;
        float square = diff * diff;

        res += square;
    }

    return res * scale * scale;
}

/**
 * FP8 E3M4 rounded L2 distance (with dynamic scaling)
 */
static float L2Sqr_FP8_E3M4_Dynamic(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float max_val = 1.0f;
    float scale = (max_val > 1e-6f) ? (max_val / 15.0f) : 1.0f;  // E3M4 range: [0.125, 30.0]
    float inv_scale = 1.0f / scale;

    float res = 0.0f;

    for (size_t i = 0; i < qty; i++) {
        float v1_scaled = pVect1[i] * inv_scale;
        float v2_scaled = pVect2[i] * inv_scale;

        float v1 = fp8_e3m4_round(v1_scaled);
        float v2 = fp8_e3m4_round(v2_scaled);

        float diff = v1 - v2;
        float square = diff * diff;

        res += square;
    }

    return res * scale * scale;
}

/**
 * FP8 FPMA 8-bit fixed-point quantized version (Q-format)
 *
 * Strategy:
 * 1. For each dimension, compute (a-b)^2 and quantize to 8-bit fixed-point
 * 2. Accumulate fixed-point values using a 32-bit accumulator (avoid overflow)
 * 3. Dequantize the accumulated result at the end
 *
 * Advantages:
 * - True FP8 fixed-point computation
 * - Configurable fractional bit position
 * - Accumulation without precision loss
 */
static float L2Sqr_FP8_FPMA_Quantized(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    // Use 64-bit accumulator (supports accumulation of arbitrary-width fixed-point values)
    uint64_t accumulator = 0;

    //multiplier width
    int multiplier_width = 16;

    float res_quantized = 0.0f;

    for (size_t i = 0; i < qty; i++) {
        float v1 = fp8_e4m3_round(pVect1[i]);
        float v2 = fp8_e4m3_round(pVect2[i]);

        // Quantize to fixed-point Q_m.n (configurable bit width)
        uint32_t quantized_square = fp8_fpma_square_quan(v1, v2);

        // Accumulate into 64-bit accumulator
        accumulator += quantized_square;
        if((i+1) % multiplier_width == 0 || i == qty-1){
            res_quantized += static_cast<float>(accumulator) / FP8_FIXED_POINT_SCALE;
            accumulator = 0;
        }
    }

    // Global dequantization: result = accumulator / 2^n
    // return static_cast<float>(accumulator) / FP8_FIXED_POINT_SCALE;
    return res_quantized;
}

/**
 * FP8 FPMA-optimized L2 distance
 *
 * Choose implementation via USE_FPMA_QUANTIZED:
 * - 1: quantized version (Q-format fixed-point with 32-bit accumulator)
 * - 0: floating-point version (faster)
 */
static float L2Sqr_FP8_FPMA(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
// #if USE_FPMA_QUANTIZED
//     // Quantized version: use Q-format fixed-point
//     return L2Sqr_FP8_FPMA_Quantized(pVect1v, pVect2v, qty_ptr);
// #else
    // Floating-point version
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    // Static scale
    float max_val = 1.0f;

    // Scaling factor: map max_val to 100 (FP8 "sweet spot")
    // For normalized vectors (max ~0.3), scale ~0.003 and scaled values ~100
    float scale = (max_val > 1e-6f) ? (max_val / 100.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    float res = 0.0f;

    for (size_t i = 0; i < qty; i++) {
        // Read FP32 values and convert to FP8
        float v1_scaled = pVect1[i] * inv_scale;
        float v2_scaled = pVect2[i] * inv_scale;

        float v1 = fp8_e4m3_round(v1_scaled);
        float v2 = fp8_e4m3_round(v2_scaled);

        // FP8 subtraction
        // float diff = fp8_e4m3_round(v1 - v2);
        float diff = v1 - v2;

        // FP8 FPMA square (direct floating-point computation, no quantization)
        float square = fp8_fpma_square(diff);
        // float square = diff * diff;

        // Accumulate
        res += square;
    }

    return res * scale * scale;
// #endif
}

/**
 * FP8 E2M5 FPMA-optimized L2 distance
 */
static float L2Sqr_FP8_FPMA_E2M5(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float max_val = 1.0f;
    float scale = (max_val > 1e-6f) ? (max_val / 3.0f) : 1.0f;  // E2M5 range: [0.5, 6.0]
    float inv_scale = 1.0f / scale;

    float res = 0.0f;

    for (size_t i = 0; i < qty; i++) {
        float v1_scaled = pVect1[i] * inv_scale;
        float v2_scaled = pVect2[i] * inv_scale;

        float v1 = fp8_e2m5_round(v1_scaled);
        float v2 = fp8_e2m5_round(v2_scaled);

        float diff = v1 - v2;
        float square = fp8_fpma_square(diff);

        res += square;
    }

    return res * scale * scale;
}

/**
 * FP8 E3M4 FPMA-optimized L2 distance
 */
static float L2Sqr_FP8_FPMA_E3M4(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float max_val = 1.0f;
    float scale = (max_val > 1e-6f) ? (max_val / 15.0f) : 1.0f;  // E3M4 range: [0.125, 30.0]
    float inv_scale = 1.0f / scale;

    float res = 0.0f;

    for (size_t i = 0; i < qty; i++) {
        float v1_scaled = pVect1[i] * inv_scale;
        float v2_scaled = pVect2[i] * inv_scale;

        float v1 = fp8_e3m4_round(v1_scaled);
        float v2 = fp8_e3m4_round(v2_scaled);

        float diff = v1 - v2;
        float square = fp8_fpma_square(diff);

        res += square;
    }

    return res * scale * scale;
}

    // float *pVect1 = (float *)pVect1v;
    // float *pVect2 = (float *)pVect2v;
    // size_t qty = *((size_t *)qty_ptr);

    // // Optionally compute scaling factor dynamically
    // // Goal: scale values into the best FP8 E4M3 range (around 1-100)
    // // float max_val = 0.0f;
    // // for (size_t i = 0; i < qty; i++) {
    // //     max_val = std::max(max_val, std::fabs(pVect1[i]));
    // //     max_val = std::max(max_val, std::fabs(pVect2[i]));
    // // }

    // // Static scale
    // float max_val = 1.0f;

    // // Scaling factor: map max_val to 100 (FP8 "sweet spot")
    // // For normalized vectors (max ~0.3), scale ~0.003 and scaled values ~100
    // float scale = (max_val > 1e-6f) ? (max_val / 100.0f) : 1.0f;
    // float inv_scale = 1.0f / scale;

    // float res = 0.0f;

    // for (size_t i = 0; i < qty; i++) {
    //     // Scale and convert to FP8 E4M3
    //     float v1_scaled = pVect1[i] * inv_scale;
    //     float v2_scaled = pVect2[i] * inv_scale;

    //     float v1 = fp8_e4m3_round(v1_scaled);
    //     float v2 = fp8_e4m3_round(v2_scaled);

    //     // Compute difference and square in FP32 precision
    //     float diff = v1 - v2;
    //     float square = diff * diff;

    //     // Accumulate
    //     res += square;
    // }

    // // Rescale back: distance_original = distance_scaled × scale²
    // return res * scale * scale;

// ====================== Dynamic-precision L2 space class ======================

/**
 * L2 space with dynamic precision support
 *
 * Usage:
 * 1. Build the graph with FP32 vectors: add_items(fp32_vectors)
 * 2. Switch precision: set_precision(PrecisionType::INT8)
 * 3. Query: knn_query(query) - distances are computed automatically with INT8 precision
 * 4. Switch to another precision: set_precision(PrecisionType::FP16_TRUE)
 * 5. Query again: distance computation automatically uses FP16
 */
class L2SpaceDynamicPrecision : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    PrecisionType precision_type_;

 protected:
    size_t data_size_;
    size_t dim_;

 public:
    L2SpaceDynamicPrecision(size_t dim, PrecisionType precision = PrecisionType::FP32)
        : precision_type_(precision), dim_(dim) {
        data_size_ = dim * sizeof(float);
        set_precision(precision);
    }

    size_t get_data_size() override {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() override {
        return fstdistfunc_;
    }

    void *get_dist_func_param() override {
        return &dim_;
    }

    ~L2SpaceDynamicPrecision() {}

    /**
     * Switch precision at runtime
     *
     * Note:
     * - Vector storage remains FP32
     * - Graph structure remains unchanged
     * - Only the distance computation changes
     */
    void set_precision(PrecisionType precision) {
        precision_type_ = precision;

        switch (precision_type_) {
            case PrecisionType::FP32_FPMA:
                fstdistfunc_ = L2Sqr_FP32_FPMA;
                break;
            case PrecisionType::FP16_TRUE:
                fstdistfunc_ = L2Sqr_FP16_Dynamic;
                break;
            case PrecisionType::FP16_FPMA:
                fstdistfunc_ = L2Sqr_FP16_FPMA;
                break;
            case PrecisionType::INT16_DIFF_FP16_FPMA:
                fstdistfunc_ = L2Sqr_INT16_DIFF_FP16_FPMA;
                break;
            case PrecisionType::FP8_E4M3:
                fstdistfunc_ = L2Sqr_FP8_E4M3_Dynamic;
                break;
            case PrecisionType::FP8_FPMA:
                fstdistfunc_ = L2Sqr_FP8_FPMA;
                break;
            case PrecisionType::FP8_E2M5:
                fstdistfunc_ = L2Sqr_FP8_E2M5_Dynamic;
                break;
            case PrecisionType::FP8_FPMA_E2M5:
                fstdistfunc_ = L2Sqr_FP8_FPMA_E2M5;
                break;
            case PrecisionType::FP8_E3M4:
                fstdistfunc_ = L2Sqr_FP8_E3M4_Dynamic;
                break;
            case PrecisionType::FP8_FPMA_E3M4:
                fstdistfunc_ = L2Sqr_FP8_FPMA_E3M4;
                break;
            case PrecisionType::INT16:
                fstdistfunc_ = L2Sqr_INT16_Dynamic;
                break;
            case PrecisionType::INT8:
                fstdistfunc_ = L2Sqr_INT8_Dynamic;
                break;
            case PrecisionType::FP32:
            default:
                fstdistfunc_ = L2Sqr_FP32;
                break;
        }
    }

    PrecisionType get_precision() const {
        return precision_type_;
    }
};

}  // namespace hnswlib
