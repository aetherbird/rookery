#pragma once

#include <cstdint>

namespace rookery {

// 16-byte aligned 48-byte activation block for Q8_0, Q6_K, Q2_K, Q3_K, IQ2_S, IQ3_S, IQ4_XS dp4a GEMV.
// - qs[0..31]: quantized int8 activations (2 x int4 for 100% LDS.128 loads)
// - d:         amax / 127.0f
// - s0, s1:    exact int32 sums of qs[0..15] and qs[16..31] (for Q6_K -32 offset & Q2_K/Q3_K)
// - s_all:     d * (s0 + s1) (for Q2_K/Q4_K/Q5_K dmin subtraction)
struct alignas(16) BlockQ8_1_Act {
    int8_t  qs[32];
    float   d;
    int32_t s0;
    int32_t s1;
    float   s_all;
};
static_assert(sizeof(BlockQ8_1_Act) == 48, "BlockQ8_1_Act must be 48 bytes (16B aligned)");

__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#elif defined(__CUDA_ARCH__)
    // Pre-sm_61 (Maxwell sm_50/sm_52, Pascal GP100 sm_60):
    // Unpack 4 signed int8 lanes into two packed int16x2 registers via PTX prmt.b32
    // (selectors 0x9180 / 0xB3A2 sign-extend each byte into 16 bits), then accumulate
    // with 4 single-cycle XMAD.S16.S16 (.H0/.H1) instructions (8 SASS ops vs 12 for bfe+shr).
    int r = c;
    asm("{\n\t"
        ".reg .b32 al, ah, bl, bh;\n\t"
        ".reg .s16 a0, a1, a2, a3, b0, b1, b2, b3;\n\t"
        "prmt.b32 al, %1, 0, 0x9180;\n\t"
        "prmt.b32 bl, %2, 0, 0x9180;\n\t"
        "prmt.b32 ah, %1, 0, 0xB3A2;\n\t"
        "prmt.b32 bh, %2, 0, 0xB3A2;\n\t"
        "mov.b32 {a0, a1}, al;\n\t"
        "mov.b32 {b0, b1}, bl;\n\t"
        "mov.b32 {a2, a3}, ah;\n\t"
        "mov.b32 {b2, b3}, bh;\n\t"
        "mad.wide.s16 %0, a0, b0, %0;\n\t"
        "mad.wide.s16 %0, a1, b1, %0;\n\t"
        "mad.wide.s16 %0, a2, b2, %0;\n\t"
        "mad.wide.s16 %0, a3, b3, %0;\n\t"
        "}" : "+r"(r) : "r"(a), "r"(b));
    return r;
#else
    const int8_t * pa = reinterpret_cast<const int8_t *>(&a);
    const int8_t * pb = reinterpret_cast<const int8_t *>(&b);
    return c + pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
#endif
}

} // namespace rookery
