#include "core/device.h"
#include "kernels/kernels.h"
#include "kernels/q8_1_act.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace rookery {

namespace {

// 16-byte aligned 80-byte FP16 activation block for sm_60 native half2 GEMV / MoE.
// Works uniformly for Q3_K, Q4_K, Q5_K, Q6_K, Q8_0!
struct alignas(16) ActF16Block {
    __half2 x[16];  // 32 normalized elements: x_raw[0..31] / amax (64 bytes = 4 x int4)
    float   d;      // amax
    float   s0;     // exact unnormalized sum(x_raw[0..15])
    float   s1;     // exact unnormalized sum(x_raw[16..31])
    float   s_all;  // s0 + s1
};
static_assert(sizeof(ActF16Block) == 80, "ActF16Block must be 80 bytes");

struct BlockQ4_K_M {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ4_K_M) == 144, "BlockQ4_K_M size");

struct BlockQ5_K_M {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ5_K_M) == 176, "BlockQ5_K_M size");

struct MaxwTileDesc {
    int32_t eid;
    int32_t start;
    int32_t count;
};

__device__ __forceinline__ float mw_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float mw_warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ float mw_half_bits_to_f32(uint16_t bits) {
    __half h;
    *reinterpret_cast<uint16_t *>(&h) = bits;
    return __half2float(h);
}

__device__ __forceinline__ __half2 mw_hfma2(__half2 a, __half2 b, __half2 c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 600 || __CUDA_ARCH__ == 620 || __CUDA_ARCH__ >= 700)
    return __hfma2(a, b, c);
#else
    float2 fa = __half22float2(a), fb = __half22float2(b), fc = __half22float2(c);
    return __floats2half2_rn(fa.x * fb.x + fc.x, fa.y * fb.y + fc.y);
#endif
}

__device__ __forceinline__ __half2 mw_hadd2(__half2 a, __half2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hadd2(a, b);
#else
    float2 fa = __half22float2(a), fb = __half22float2(b);
    return __floats2half2_rn(fa.x + fb.x, fa.y + fb.y);
#endif
}

__device__ __forceinline__ __half2 mw_hmul2(__half2 a, __half2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hmul2(a, b);
#else
    float2 fa = __half22float2(a), fb = __half22float2(b);
    return __floats2half2_rn(fa.x * fb.x, fa.y * fb.y);
#endif
}

__device__ __forceinline__ __half2 mw_bits_to_h2(uint32_t bits) {
    return *reinterpret_cast<const __half2 *>(&bits);
}

// IEEE-754 FP16 1024.0 (0x6400) exact byte-to-half2 injection (for unsigned byte u in 0..255):
// Returns exact half2(1024.0 + u_lo, 1024.0 + u_hi).
__device__ __forceinline__ __half2 mw_u8x2_to_h2_lo(uint32_t u) {
    uint32_t p = __byte_perm(u, 0x00006400u, 0x5150);
    return *reinterpret_cast<const __half2 *>(&p);
}

__device__ __forceinline__ __half2 mw_u8x2_to_h2_hi(uint32_t u) {
    uint32_t p = __byte_perm(u, 0x00006400u, 0x5352);
    return *reinterpret_cast<const __half2 *>(&p);
}

__device__ __forceinline__ void mw_unpack_q4k_sc(
    int idx, uint32_t s0, uint32_t s1, uint32_t s2, float & sc, float & mn) {
    if (idx < 4) {
        sc = static_cast<float>((s0 >> (8 * idx)) & 63u);
        mn = static_cast<float>((s1 >> (8 * idx)) & 63u);
    } else {
        const int j = idx - 4;
        const uint32_t hi_s = (s0 >> (8 * j + 6)) & 3u;
        const uint32_t hi_m = (s1 >> (8 * j + 6)) & 3u;
        const uint32_t b    = (s2 >> (8 * j)) & 0xFFu;
        sc = static_cast<float>((b & 0xFu) | (hi_s << 4));
        mn = static_cast<float>((b >> 4)   | (hi_m << 4));
    }
}

__device__ __forceinline__ void mw_unpack_q3k_scales(
    const uint32_t * __restrict__ W_sc, int64_t bi, uint32_t aux[4]) {
    const uint32_t a0  = __ldg(&W_sc[bi * 3 + 0]);
    const uint32_t a1  = __ldg(&W_sc[bi * 3 + 1]);
    const uint32_t tmp = __ldg(&W_sc[bi * 3 + 2]);
    constexpr uint32_t kmask1 = 0x03030303u;
    constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    aux[0] = (a0 & kmask2)        | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (a1 & kmask2)        | (((tmp >> 2) & kmask1) << 4);
    aux[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
}

// ============================================================================
// 1. ActF16Block Quantization (1 warp per 32-element block)
// ============================================================================
__global__ void mw_quantize_act_f16_kernel(
    const float * __restrict__ x,
    ActF16Block * __restrict__ out,
    int total_blocks) {

    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= total_blocks) return;
    const int lane = threadIdx.x;
    const float v = x[static_cast<size_t>(b) * 32 + lane];
    const float amax = __shfl_sync(0xffffffffu, mw_warp_max(fabsf(v)), 0);
    const float inv  = (amax > 0.0f) ? (1.0f / amax) : 0.0f;
    const float vn   = v * inv;
    const float vn_hi = __shfl_down_sync(0xffffffffu, vn, 1);
    if ((lane & 1) == 0) {
        out[b].x[lane >> 1] = __halves2half2(__float2half_rn(vn), __float2half_rn(vn_hi));
    }

    float hs = v;
    hs += __shfl_xor_sync(0xffffffffu, hs, 1);
    hs += __shfl_xor_sync(0xffffffffu, hs, 2);
    hs += __shfl_xor_sync(0xffffffffu, hs, 4);
    hs += __shfl_xor_sync(0xffffffffu, hs, 8);
    const float s0 = __shfl_sync(0xffffffffu, hs, 0);
    const float s1 = __shfl_sync(0xffffffffu, hs, 16);
    if (lane == 0) {
        out[b].d     = amax;
        out[b].s0    = s0;
        out[b].s1    = s1;
        out[b].s_all = s0 + s1;
    }
}

// Even/Odd interleaved ActF16Block quantization:
// For each 4-element group (x0, x1, x2, x3), packs (x0, x2) and (x1, x3) as __half2 pairs.
// Enables zero-__byte_perm, zero-__hsub2 Q4_K sm_60 decoding (SHL/SHR + LOP3 0x3C003C00)!
__global__ void mw_quantize_act_f16_eo_kernel(
    const float * __restrict__ x,
    ActF16Block * __restrict__ out,
    int total_blocks) {

    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= total_blocks) return;
    const int lane = threadIdx.x;
    const float v = x[static_cast<size_t>(b) * 32 + lane];
    const float amax = __shfl_sync(0xffffffffu, mw_warp_max(fabsf(v)), 0);
    const float inv  = (amax > 0.0f) ? (1.0f / amax) : 0.0f;
    const float vn   = v * inv;
    const float vn_p2 = __shfl_down_sync(0xffffffffu, vn, 2);
    if ((lane & 2) == 0) {
        out[b].x[((lane >> 2) << 1) | (lane & 1)] = __halves2half2(__float2half_rn(vn), __float2half_rn(vn_p2));
    }

    float hs = v;
    hs += __shfl_xor_sync(0xffffffffu, hs, 1);
    hs += __shfl_xor_sync(0xffffffffu, hs, 2);
    hs += __shfl_xor_sync(0xffffffffu, hs, 4);
    hs += __shfl_xor_sync(0xffffffffu, hs, 8);
    const float s0 = __shfl_sync(0xffffffffu, hs, 0);
    const float s1 = __shfl_sync(0xffffffffu, hs, 16);
    if (lane == 0) {
        out[b].d     = amax;
        out[b].s0    = s0;
        out[b].s1    = s1;
        out[b].s_all = s0 + s1;
    }
}

__global__ void mw_f16_to_f32_vec_kernel(const __half * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t i2 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t idx = i2 * 2;
    if (idx + 1 < n) {
        const __half2 h = reinterpret_cast<const __half2 *>(src)[i2];
        const float2 f = __half22float2(h);
        reinterpret_cast<float2 *>(dst)[i2] = f;
    } else if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

// ============================================================================
// 2. Q3_K SoA Embedding Lookup & Fast Dequant (F16 / F32)
// ============================================================================
__global__ void mw_embedding_q3_k_soa_kernel(
    const uint8_t *  __restrict__ W_qs,
    const uint8_t *  __restrict__ W_hm,
    const uint32_t * __restrict__ W_sc,
    const uint16_t * __restrict__ W_d,
    const int32_t *  __restrict__ tokens,
    float *          __restrict__ out,
    int n_embd, int hc_mult) {

    const int bi   = blockIdx.x;
    const int t    = blockIdx.y;
    const int lane = threadIdx.x; // 0..31 -> 8 elems per lane
    const int nb   = n_embd >> 8;
    const int tok  = tokens[t];
    const int64_t bidx = static_cast<int64_t>(tok) * nb + bi;

    const float d_all = mw_half_bits_to_f32(__ldg(&W_d[bidx]));
    uint32_t aux[4];
    mw_unpack_q3k_scales(W_sc, bidx, aux);

    const int grp   = lane >> 1;      // 0..15
    const int half  = (lane & 1) * 8; // 0 or 8
    const int n128  = grp >> 3;
    const int j     = (grp & 7) >> 1;
    const int sub   = grp & 1;
    const int shift = 2 * j;
    const uint8_t m = static_cast<uint8_t>(1u << (n128 * 4 + j));
    const int sc_val = static_cast<int>((aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu) - 32;
    const float dl = d_all * static_cast<float>(sc_val);

    const uint8_t * qs = W_qs + bidx * 64 + n128 * 32 + sub * 16 + half;
    const uint8_t * hm = W_hm + bidx * 32 + sub * 16 + half;
    const int y_base = bi * 256 + grp * 16 + half;

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int ql = (static_cast<int>(__ldg(&qs[k])) >> shift) & 3;
        const int qh = (__ldg(&hm[k]) & m) ? 0 : 4;
        const float v = dl * static_cast<float>(ql - qh);
        for (int h = 0; h < hc_mult; ++h) {
            out[(static_cast<size_t>(t) * hc_mult + h) * n_embd + y_base + k] = v;
        }
    }
}

template <typename TDst>
__global__ void mw_dequant_q3_k_soa_kernel(
    const uint8_t *  __restrict__ W_qs,
    const uint8_t *  __restrict__ W_hm,
    const uint32_t * __restrict__ W_sc,
    const uint16_t * __restrict__ W_d,
    TDst *           __restrict__ dst,
    int64_t n_blocks) {

    const int64_t bi = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (bi >= n_blocks) return;
    const int lane = threadIdx.x;

    const float d_all = mw_half_bits_to_f32(__ldg(&W_d[bi]));
    uint32_t aux[4];
    mw_unpack_q3k_scales(W_sc, bi, aux);

    const int grp   = lane >> 1;      // 0..15
    const int half  = (lane & 1) * 8; // 0 or 8
    const int n128  = grp >> 3;
    const int j     = (grp & 7) >> 1;
    const int sub   = grp & 1;
    const int shift = 2 * j;
    const int hbit  = n128 * 4 + j;
    const int sc_val = static_cast<int>((aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu) - 32;
    const float dl = d_all * static_cast<float>(sc_val);

    const uint2 qv = __ldg(reinterpret_cast<const uint2 *>(W_qs + bi * 64 + n128 * 32 + sub * 16 + half));
    const uint2 hv = __ldg(reinterpret_cast<const uint2 *>(W_hm + bi * 32 + sub * 16 + half));

    const uint32_t u0 = ((qv.x >> shift) & 0x03030303u) | (((hv.x >> hbit) & 0x01010101u) << 2);
    const uint32_t u1 = ((qv.y >> shift) & 0x03030303u) | (((hv.y >> hbit) & 0x01010101u) << 2);

    TDst * out = dst + bi * 256 + grp * 16 + half;
    if constexpr (std::is_same_v<TDst, __half>) {
        const __half2 neg1028 = __floats2half2_rn(-1028.0f, -1028.0f);
        const __half2 dl2     = __floats2half2_rn(dl, dl);
        auto * o2 = reinterpret_cast<__half2 *>(out);
        o2[0] = mw_hmul2(dl2, mw_hadd2(mw_u8x2_to_h2_lo(u0), neg1028));
        o2[1] = mw_hmul2(dl2, mw_hadd2(mw_u8x2_to_h2_hi(u0), neg1028));
        o2[2] = mw_hmul2(dl2, mw_hadd2(mw_u8x2_to_h2_lo(u1), neg1028));
        o2[3] = mw_hmul2(dl2, mw_hadd2(mw_u8x2_to_h2_hi(u1), neg1028));
    } else {
        auto * o4 = reinterpret_cast<float4 *>(out);
        o4[0] = make_float4(
            dl * (static_cast<int>((u0 >>  0) & 7u) - 4),
            dl * (static_cast<int>((u0 >>  8) & 7u) - 4),
            dl * (static_cast<int>((u0 >> 16) & 7u) - 4),
            dl * (static_cast<int>((u0 >> 24) & 7u) - 4));
        o4[1] = make_float4(
            dl * (static_cast<int>((u1 >>  0) & 7u) - 4),
            dl * (static_cast<int>((u1 >>  8) & 7u) - 4),
            dl * (static_cast<int>((u1 >> 16) & 7u) - 4),
            dl * (static_cast<int>((u1 >> 24) & 7u) - 4));
    }
}

__global__ void mw_dequant_q4_k_f16_kernel(
    const BlockQ4_K_M * __restrict__ src, __half * __restrict__ dst, int64_t n_blocks) {
    const int64_t bi = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (bi >= n_blocks) return;
    const int lane = threadIdx.x;
    const BlockQ4_K_M & b = src[bi];
    const int4 hdr = __ldg(reinterpret_cast<const int4 *>(&b));
    const float d    = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
    const float dmin = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
    const int j   = lane >> 3;
    const int sub = (lane & 7) * 4;
    float sc1, m1, sc2, m2;
    mw_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
    mw_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
    const float d1 = d * sc1, mn1 = dmin * m1;
    const float d2 = d * sc2, mn2 = dmin * m2;
    const uint32_t ql4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qs)[j * 8 + (lane & 7)]);
    const uint32_t u_lo = ql4 & 0x0F0F0F0Fu;
    const uint32_t u_hi = (ql4 >> 4) & 0x0F0F0F0Fu;

    const __half2 neg1024 = __floats2half2_rn(-1024.0f, -1024.0f);
    const __half2 d1_2 = __floats2half2_rn(d1, d1), mn1_2 = __floats2half2_rn(-mn1, -mn1);
    const __half2 d2_2 = __floats2half2_rn(d2, d2), mn2_2 = __floats2half2_rn(-mn2, -mn2);

    auto * o_lo = reinterpret_cast<__half2 *>(dst + bi * 256 + j * 64 + sub);
    auto * o_hi = reinterpret_cast<__half2 *>(dst + bi * 256 + j * 64 + 32 + sub);
    o_lo[0] = mw_hfma2(d1_2, mw_hadd2(mw_u8x2_to_h2_lo(u_lo), neg1024), mn1_2);
    o_lo[1] = mw_hfma2(d1_2, mw_hadd2(mw_u8x2_to_h2_hi(u_lo), neg1024), mn1_2);
    o_hi[0] = mw_hfma2(d2_2, mw_hadd2(mw_u8x2_to_h2_lo(u_hi), neg1024), mn2_2);
    o_hi[1] = mw_hfma2(d2_2, mw_hadd2(mw_u8x2_to_h2_hi(u_hi), neg1024), mn2_2);
}

__global__ void mw_dequant_q5_k_f16_kernel(
    const BlockQ5_K_M * __restrict__ src, __half * __restrict__ dst, int64_t n_blocks) {
    const int64_t bi = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (bi >= n_blocks) return;
    const int lane = threadIdx.x;
    const BlockQ5_K_M & b = src[bi];
    const int4 hdr = __ldg(reinterpret_cast<const int4 *>(&b));
    const float d    = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
    const float dmin = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
    const int j   = lane >> 3;
    const int sub = (lane & 7) * 4;
    float sc1, m1, sc2, m2;
    mw_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
    mw_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
    const float d1 = d * sc1, mn1 = dmin * m1;
    const float d2 = d * sc2, mn2 = dmin * m2;
    const uint32_t ql4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qs)[j * 8 + (lane & 7)]);
    const uint32_t qh4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qh)[lane & 7]);
    const uint32_t u_lo = (ql4 & 0x0F0F0F0Fu) | (((qh4 >> (2 * j + 0)) & 0x01010101u) << 4);
    const uint32_t u_hi = ((ql4 >> 4) & 0x0F0F0F0Fu) | (((qh4 >> (2 * j + 1)) & 0x01010101u) << 4);

    const __half2 neg1024 = __floats2half2_rn(-1024.0f, -1024.0f);
    const __half2 d1_2 = __floats2half2_rn(d1, d1), mn1_2 = __floats2half2_rn(-mn1, -mn1);
    const __half2 d2_2 = __floats2half2_rn(d2, d2), mn2_2 = __floats2half2_rn(-mn2, -mn2);

    auto * o_lo = reinterpret_cast<__half2 *>(dst + bi * 256 + j * 64 + sub);
    auto * o_hi = reinterpret_cast<__half2 *>(dst + bi * 256 + j * 64 + 32 + sub);
    o_lo[0] = mw_hfma2(d1_2, mw_hadd2(mw_u8x2_to_h2_lo(u_lo), neg1024), mn1_2);
    o_lo[1] = mw_hfma2(d1_2, mw_hadd2(mw_u8x2_to_h2_hi(u_lo), neg1024), mn1_2);
    o_hi[0] = mw_hfma2(d2_2, mw_hadd2(mw_u8x2_to_h2_lo(u_hi), neg1024), mn2_2);
    o_hi[1] = mw_hfma2(d2_2, mw_hadd2(mw_u8x2_to_h2_hi(u_hi), neg1024), mn2_2);
}

// ============================================================================
// 3. Q3_K SoA Row Dot Helpers (dp4a & sm_60 half2)
// ============================================================================
__device__ __forceinline__ float dot_q3_k_soa_row_dp4a(
    const int4 *          __restrict__ W_qs,
    const int4 *          __restrict__ W_hm,
    const uint32_t *      __restrict__ W_sc,
    const uint16_t *      __restrict__ W_d,
    const BlockQ8_1_Act * __restrict__ s_x,
    int nb, int lane) {

    const int blk_in_w = lane >> 2; // 0..7
    const int sub_idx  = lane & 3;  // 0..3
    const int n128     = sub_idx >> 1;
    const int sub      = sub_idx & 1;

    float acc = 0.0f;
    for (int bi = blk_in_w; bi < nb; bi += 8) {
        const int4 qs_v = __ldg(&W_qs[bi * 4 + sub_idx]);
        const int4 hm_v = __ldg(&W_hm[bi * 2 + sub]);
        const float d_all = mw_half_bits_to_f32(__ldg(&W_d[bi]));
        uint32_t aux[4];
        mw_unpack_q3k_scales(W_sc, bi, aux);

        const uint32_t qw[4] = {
            static_cast<uint32_t>(qs_v.x), static_cast<uint32_t>(qs_v.y),
            static_cast<uint32_t>(qs_v.z), static_cast<uint32_t>(qs_v.w)
        };
        const uint32_t hw[4] = {
            static_cast<uint32_t>(hm_v.x), static_cast<uint32_t>(hm_v.y),
            static_cast<uint32_t>(hm_v.z), static_cast<uint32_t>(hm_v.w)
        };

        float bsum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int grp   = n128 * 8 + j * 2 + sub;
            const int sc_v  = static_cast<int>((aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu) - 32;
            const int xb    = bi * 8 + n128 * 4 + j;
            const int4 x4   = reinterpret_cast<const int4 *>(s_x[xb].qs)[sub];
            const int sx    = sub ? s_x[xb].s1 : s_x[xb].s0;
            const int sh_q  = 2 * j;
            const int sh_h  = n128 * 4 + j;

            const int u0 = static_cast<int>(((qw[0] >> sh_q) & 0x03030303u) | (((hw[0] >> sh_h) & 0x01010101u) << 2));
            const int u1 = static_cast<int>(((qw[1] >> sh_q) & 0x03030303u) | (((hw[1] >> sh_h) & 0x01010101u) << 2));
            const int u2 = static_cast<int>(((qw[2] >> sh_q) & 0x03030303u) | (((hw[2] >> sh_h) & 0x01010101u) << 2));
            const int u3 = static_cast<int>(((qw[3] >> sh_q) & 0x03030303u) | (((hw[3] >> sh_h) & 0x01010101u) << 2));

            int s = dp4a_s32(u0, x4.x, 0);
            s     = dp4a_s32(u1, x4.y, s);
            s     = dp4a_s32(u2, x4.z, s);
            s     = dp4a_s32(u3, x4.w, s);
            bsum += (static_cast<float>(sc_v) * s_x[xb].d) * static_cast<float>(s - 4 * sx);
        }
        acc += d_all * bsum;
    }
    return mw_warp_sum(acc);
}

__device__ __forceinline__ void dot_q3_k_soa_row_dp4a_t2(
    const int4 *          __restrict__ W_qs,
    const int4 *          __restrict__ W_hm,
    const uint32_t *      __restrict__ W_sc,
    const uint16_t *      __restrict__ W_d,
    const BlockQ8_1_Act * __restrict__ s_x0,
    const BlockQ8_1_Act * __restrict__ s_x1,
    int nb, int lane, float & out0, float & out1) {

    const int blk_in_w = lane >> 2;
    const int sub_idx  = lane & 3;
    const int n128     = sub_idx >> 1;
    const int sub      = sub_idx & 1;

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int bi = blk_in_w; bi < nb; bi += 8) {
        const int4 qs_v = __ldg(&W_qs[bi * 4 + sub_idx]);
        const int4 hm_v = __ldg(&W_hm[bi * 2 + sub]);
        const float d_all = mw_half_bits_to_f32(__ldg(&W_d[bi]));
        uint32_t aux[4];
        mw_unpack_q3k_scales(W_sc, bi, aux);

        const uint32_t qw[4] = {
            static_cast<uint32_t>(qs_v.x), static_cast<uint32_t>(qs_v.y),
            static_cast<uint32_t>(qs_v.z), static_cast<uint32_t>(qs_v.w)
        };
        const uint32_t hw[4] = {
            static_cast<uint32_t>(hm_v.x), static_cast<uint32_t>(hm_v.y),
            static_cast<uint32_t>(hm_v.z), static_cast<uint32_t>(hm_v.w)
        };

        float bsum0 = 0.0f, bsum1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int grp  = n128 * 8 + j * 2 + sub;
            const int sc_v = static_cast<int>((aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu) - 32;
            const int xb   = bi * 8 + n128 * 4 + j;
            const int sh_q = 2 * j;
            const int sh_h = n128 * 4 + j;

            const int u0 = static_cast<int>(((qw[0] >> sh_q) & 0x03030303u) | (((hw[0] >> sh_h) & 0x01010101u) << 2));
            const int u1 = static_cast<int>(((qw[1] >> sh_q) & 0x03030303u) | (((hw[1] >> sh_h) & 0x01010101u) << 2));
            const int u2 = static_cast<int>(((qw[2] >> sh_q) & 0x03030303u) | (((hw[2] >> sh_h) & 0x01010101u) << 2));
            const int u3 = static_cast<int>(((qw[3] >> sh_q) & 0x03030303u) | (((hw[3] >> sh_h) & 0x01010101u) << 2));

            const int4 x0 = reinterpret_cast<const int4 *>(s_x0[xb].qs)[sub];
            const int4 x1 = reinterpret_cast<const int4 *>(s_x1[xb].qs)[sub];
            const int s0 = dp4a_s32(u0, x0.x, dp4a_s32(u1, x0.y, dp4a_s32(u2, x0.z, dp4a_s32(u3, x0.w, 0))));
            const int s1 = dp4a_s32(u0, x1.x, dp4a_s32(u1, x1.y, dp4a_s32(u2, x1.z, dp4a_s32(u3, x1.w, 0))));
            const int sx0 = sub ? s_x0[xb].s1 : s_x0[xb].s0;
            const int sx1 = sub ? s_x1[xb].s1 : s_x1[xb].s0;
            bsum0 += (static_cast<float>(sc_v) * s_x0[xb].d) * static_cast<float>(s0 - 4 * sx0);
            bsum1 += (static_cast<float>(sc_v) * s_x1[xb].d) * static_cast<float>(s1 - 4 * sx1);
        }
        acc0 += d_all * bsum0;
        acc1 += d_all * bsum1;
    }
    out0 = mw_warp_sum(acc0);
    out1 = mw_warp_sum(acc1);
}

template <bool T2>
__device__ __forceinline__ void dot_q3_k_soa_row_h2(
    const int4 *        __restrict__ W_qs,
    const int4 *        __restrict__ W_hm,
    const uint32_t *    __restrict__ W_sc,
    const uint16_t *    __restrict__ W_d,
    const ActF16Block * __restrict__ s_x0,
    const ActF16Block * __restrict__ s_x1,
    int nb, int lane, float & out0, float & out1) {

    const int blk_in_w = lane >> 2;
    const int sub_idx  = lane & 3;
    const int n128     = sub_idx >> 1;
    const int sub      = sub_idx & 1;
    const __half2 neg1028 = __floats2half2_rn(-1028.0f, -1028.0f);

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int bi = blk_in_w; bi < nb; bi += 8) {
        const int4 qs_v = __ldg(&W_qs[bi * 4 + sub_idx]);
        const int4 hm_v = __ldg(&W_hm[bi * 2 + sub]);
        const float d_all = mw_half_bits_to_f32(__ldg(&W_d[bi]));
        uint32_t aux[4];
        mw_unpack_q3k_scales(W_sc, bi, aux);

        const uint32_t qw[4] = {
            static_cast<uint32_t>(qs_v.x), static_cast<uint32_t>(qs_v.y),
            static_cast<uint32_t>(qs_v.z), static_cast<uint32_t>(qs_v.w)
        };
        const uint32_t hw[4] = {
            static_cast<uint32_t>(hm_v.x), static_cast<uint32_t>(hm_v.y),
            static_cast<uint32_t>(hm_v.z), static_cast<uint32_t>(hm_v.w)
        };

        float bsum0 = 0.0f, bsum1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int grp  = n128 * 8 + j * 2 + sub;
            const int sc_v = static_cast<int>((aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu) - 32;
            const int xb   = bi * 8 + n128 * 4 + j;
            const int sh_q = 2 * j;
            const int sh_h = n128 * 4 + j;

            const __half2 * x0h = s_x0[xb].x + sub * 8;
            const __half2 * x1h = T2 ? (s_x1[xb].x + sub * 8) : nullptr;
            __half2 h0 = __floats2half2_rn(0.0f, 0.0f);
            __half2 h1 = __floats2half2_rn(0.0f, 0.0f);

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint32_t u = ((qw[c] >> sh_q) & 0x03030303u) | (((hw[c] >> sh_h) & 0x01010101u) << 2);
                const __half2 w01 = mw_hadd2(mw_u8x2_to_h2_lo(u), neg1028);
                const __half2 w23 = mw_hadd2(mw_u8x2_to_h2_hi(u), neg1028);
                h0 = mw_hfma2(w01, x0h[c * 2 + 0], h0);
                h0 = mw_hfma2(w23, x0h[c * 2 + 1], h0);
                if constexpr (T2) {
                    h1 = mw_hfma2(w01, x1h[c * 2 + 0], h1);
                    h1 = mw_hfma2(w23, x1h[c * 2 + 1], h1);
                }
            }
            const float2 f0 = __half22float2(h0);
            bsum0 += (static_cast<float>(sc_v) * s_x0[xb].d) * (f0.x + f0.y);
            if constexpr (T2) {
                const float2 f1 = __half22float2(h1);
                bsum1 += (static_cast<float>(sc_v) * s_x1[xb].d) * (f1.x + f1.y);
            }
        }
        acc0 += d_all * bsum0;
        if constexpr (T2) acc1 += d_all * bsum1;
    }
    out0 = mw_warp_sum(acc0);
    if constexpr (T2) out1 = mw_warp_sum(acc1);
}

template <bool T2>
__device__ __forceinline__ void dot_q4_k_row_h2(
    const BlockQ4_K_M * __restrict__ w_row,
    const ActF16Block * __restrict__ s_x0,
    const ActF16Block * __restrict__ s_x1,
    int nb, int lane, float & out0, float & out1) {

    const int blk_in_w = lane >> 3;
    const int sub      = lane & 7;
    const int j        = sub >> 1;
    const int is       = sub & 1;
    const __half2 neg1024 = __floats2half2_rn(-1024.0f, -1024.0f);

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int bi = blk_in_w; bi < nb; bi += 4) {
        const BlockQ4_K_M & b = w_row[bi];
        const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
        const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
        const float d    = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
        const float dmin = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
        float sc1, m1, sc2, m2;
        mw_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
        mw_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
        const float d1 = d * sc1, mn1 = dmin * m1;
        const float d2 = d * sc2, mn2 = dmin * m2;

        const uint32_t qws[4] = {
            static_cast<uint32_t>(ql_v.x), static_cast<uint32_t>(ql_v.y),
            static_cast<uint32_t>(ql_v.z), static_cast<uint32_t>(ql_v.w)
        };
        const int xb = bi * 8 + 2 * j;
        const __half2 * x10 = s_x0[xb + 0].x + is * 8;
        const __half2 * x20 = s_x0[xb + 1].x + is * 8;
        const __half2 * x11 = T2 ? (s_x1[xb + 0].x + is * 8) : nullptr;
        const __half2 * x21 = T2 ? (s_x1[xb + 1].x + is * 8) : nullptr;

        __half2 h10 = __floats2half2_rn(0.0f, 0.0f), h20 = __floats2half2_rn(0.0f, 0.0f);
        __half2 h11 = __floats2half2_rn(0.0f, 0.0f), h21 = __floats2half2_rn(0.0f, 0.0f);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t u_lo = qws[c] & 0x0F0F0F0Fu;
            const uint32_t u_hi = (qws[c] >> 4) & 0x0F0F0F0Fu;
            const __half2 w1a = mw_hadd2(mw_u8x2_to_h2_lo(u_lo), neg1024);
            const __half2 w1b = mw_hadd2(mw_u8x2_to_h2_hi(u_lo), neg1024);
            const __half2 w2a = mw_hadd2(mw_u8x2_to_h2_lo(u_hi), neg1024);
            const __half2 w2b = mw_hadd2(mw_u8x2_to_h2_hi(u_hi), neg1024);
            h10 = mw_hfma2(w1a, x10[c * 2 + 0], h10); h10 = mw_hfma2(w1b, x10[c * 2 + 1], h10);
            h20 = mw_hfma2(w2a, x20[c * 2 + 0], h20); h20 = mw_hfma2(w2b, x20[c * 2 + 1], h20);
            if constexpr (T2) {
                h11 = mw_hfma2(w1a, x11[c * 2 + 0], h11); h11 = mw_hfma2(w1b, x11[c * 2 + 1], h11);
                h21 = mw_hfma2(w2a, x21[c * 2 + 0], h21); h21 = mw_hfma2(w2b, x21[c * 2 + 1], h21);
            }
        }
        const float2 f10 = __half22float2(h10), f20 = __half22float2(h20);
        const float s10 = is ? s_x0[xb + 0].s1 : s_x0[xb + 0].s0;
        const float s20 = is ? s_x0[xb + 1].s1 : s_x0[xb + 1].s0;
        acc0 += (d1 * s_x0[xb + 0].d) * (f10.x + f10.y) - mn1 * s10
              + (d2 * s_x0[xb + 1].d) * (f20.x + f20.y) - mn2 * s20;
        if constexpr (T2) {
            const float2 f11 = __half22float2(h11), f21 = __half22float2(h21);
            const float s11 = is ? s_x1[xb + 0].s1 : s_x1[xb + 0].s0;
            const float s21 = is ? s_x1[xb + 1].s1 : s_x1[xb + 1].s0;
            acc1 += (d1 * s_x1[xb + 0].d) * (f11.x + f11.y) - mn1 * s11
                  + (d2 * s_x1[xb + 1].d) * (f21.x + f21.y) - mn2 * s21;
        }
    }
    out0 = mw_warp_sum(acc0);
    if constexpr (T2) out1 = mw_warp_sum(acc1);
}

template <bool T2>
__device__ __forceinline__ void dot_q5_k_row_h2(
    const BlockQ5_K_M * __restrict__ w_row,
    const ActF16Block * __restrict__ s_x0,
    const ActF16Block * __restrict__ s_x1,
    int nb, int lane, float & out0, float & out1) {

    const int blk_in_w = lane >> 3;
    const int sub      = lane & 7;
    const int j        = sub >> 1;
    const int is       = sub & 1;
    const __half2 neg1024 = __floats2half2_rn(-1024.0f, -1024.0f);

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int bi = blk_in_w; bi < nb; bi += 4) {
        const BlockQ5_K_M & b = w_row[bi];
        const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
        const int4 qh_v = __ldg(&reinterpret_cast<const int4 *>(b.qh)[is]);
        const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
        const float d    = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
        const float dmin = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
        float sc1, m1, sc2, m2;
        mw_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
        mw_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
        const float d1 = d * sc1, mn1 = dmin * m1;
        const float d2 = d * sc2, mn2 = dmin * m2;

        const uint32_t qws[4] = {
            static_cast<uint32_t>(ql_v.x), static_cast<uint32_t>(ql_v.y),
            static_cast<uint32_t>(ql_v.z), static_cast<uint32_t>(ql_v.w)
        };
        const uint32_t hws[4] = {
            static_cast<uint32_t>(qh_v.x), static_cast<uint32_t>(qh_v.y),
            static_cast<uint32_t>(qh_v.z), static_cast<uint32_t>(qh_v.w)
        };
        const int xb = bi * 8 + 2 * j;
        const __half2 * x10 = s_x0[xb + 0].x + is * 8;
        const __half2 * x20 = s_x0[xb + 1].x + is * 8;
        const __half2 * x11 = T2 ? (s_x1[xb + 0].x + is * 8) : nullptr;
        const __half2 * x21 = T2 ? (s_x1[xb + 1].x + is * 8) : nullptr;

        __half2 h10 = __floats2half2_rn(0.0f, 0.0f), h20 = __floats2half2_rn(0.0f, 0.0f);
        __half2 h11 = __floats2half2_rn(0.0f, 0.0f), h21 = __floats2half2_rn(0.0f, 0.0f);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t u_lo = (qws[c] & 0x0F0F0F0Fu) | (((hws[c] >> (2 * j + 0)) & 0x01010101u) << 4);
            const uint32_t u_hi = ((qws[c] >> 4) & 0x0F0F0F0Fu) | (((hws[c] >> (2 * j + 1)) & 0x01010101u) << 4);
            const __half2 w1a = mw_hadd2(mw_u8x2_to_h2_lo(u_lo), neg1024);
            const __half2 w1b = mw_hadd2(mw_u8x2_to_h2_hi(u_lo), neg1024);
            const __half2 w2a = mw_hadd2(mw_u8x2_to_h2_lo(u_hi), neg1024);
            const __half2 w2b = mw_hadd2(mw_u8x2_to_h2_hi(u_hi), neg1024);
            h10 = mw_hfma2(w1a, x10[c * 2 + 0], h10); h10 = mw_hfma2(w1b, x10[c * 2 + 1], h10);
            h20 = mw_hfma2(w2a, x20[c * 2 + 0], h20); h20 = mw_hfma2(w2b, x20[c * 2 + 1], h20);
            if constexpr (T2) {
                h11 = mw_hfma2(w1a, x11[c * 2 + 0], h11); h11 = mw_hfma2(w1b, x11[c * 2 + 1], h11);
                h21 = mw_hfma2(w2a, x21[c * 2 + 0], h21); h21 = mw_hfma2(w2b, x21[c * 2 + 1], h21);
            }
        }
        const float2 f10 = __half22float2(h10), f20 = __half22float2(h20);
        const float s10 = is ? s_x0[xb + 0].s1 : s_x0[xb + 0].s0;
        const float s20 = is ? s_x0[xb + 1].s1 : s_x0[xb + 1].s0;
        acc0 += (d1 * s_x0[xb + 0].d) * (f10.x + f10.y) - mn1 * s10
              + (d2 * s_x0[xb + 1].d) * (f20.x + f20.y) - mn2 * s20;
        if constexpr (T2) {
            const float2 f11 = __half22float2(h11), f21 = __half22float2(h21);
            const float s11 = is ? s_x1[xb + 0].s1 : s_x1[xb + 0].s0;
            const float s21 = is ? s_x1[xb + 1].s1 : s_x1[xb + 1].s0;
            acc1 += (d1 * s_x1[xb + 0].d) * (f11.x + f11.y) - mn1 * s11
                  + (d2 * s_x1[xb + 1].d) * (f21.x + f21.y) - mn2 * s21;
        }
    }
    out0 = mw_warp_sum(acc0);
    if constexpr (T2) out1 = mw_warp_sum(acc1);
}

template <bool T2>
__device__ __forceinline__ void dot_q6_k_soa_row_h2(
    const int4 *        __restrict__ W_ql,
    const int4 *        __restrict__ W_qh,
    const int8_t *      __restrict__ W_sc,
    const uint16_t *    __restrict__ W_d,
    const ActF16Block * __restrict__ s_x0,
    const ActF16Block * __restrict__ s_x1,
    int nb, int lane, float & out0, float & out1) {

    const int blk_in_w = lane >> 2;
    const int sub      = lane & 3;
    const int h        = sub >> 1;
    const int is       = sub & 1;
    const __half2 neg1056 = __floats2half2_rn(-1056.0f, -1056.0f);

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int bi = blk_in_w; bi < nb; bi += 8) {
        const int4 qh_v  = __ldg(&W_qh[bi * 4 + sub]);
        const int4 ql0_v = __ldg(&W_ql[bi * 8 + h * 4 + is]);
        const int4 ql1_v = __ldg(&W_ql[bi * 8 + h * 4 + 2 + is]);
        const int8_t * sc = W_sc + bi * 16 + h * 8 + is;
        const float dw  = mw_half_bits_to_f32(__ldg(&W_d[bi]));
        const float sc0 = dw * static_cast<float>(__ldg(&sc[0]));
        const float sc2 = dw * static_cast<float>(__ldg(&sc[2]));
        const float sc4 = dw * static_cast<float>(__ldg(&sc[4]));
        const float sc6 = dw * static_cast<float>(__ldg(&sc[6]));

        const uint32_t ql0w[4] = { static_cast<uint32_t>(ql0_v.x), static_cast<uint32_t>(ql0_v.y), static_cast<uint32_t>(ql0_v.z), static_cast<uint32_t>(ql0_v.w) };
        const uint32_t ql1w[4] = { static_cast<uint32_t>(ql1_v.x), static_cast<uint32_t>(ql1_v.y), static_cast<uint32_t>(ql1_v.z), static_cast<uint32_t>(ql1_v.w) };
        const uint32_t qhw[4]  = { static_cast<uint32_t>(qh_v.x),  static_cast<uint32_t>(qh_v.y),  static_cast<uint32_t>(qh_v.z),  static_cast<uint32_t>(qh_v.w)  };

        const int xb = bi * 8 + h * 4;
        const __half2 * x0_0 = s_x0[xb + 0].x + is * 8;
        const __half2 * x0_1 = s_x0[xb + 1].x + is * 8;
        const __half2 * x0_2 = s_x0[xb + 2].x + is * 8;
        const __half2 * x0_3 = s_x0[xb + 3].x + is * 8;

        __half2 a0 = __floats2half2_rn(0.0f, 0.0f), a1 = __floats2half2_rn(0.0f, 0.0f);
        __half2 a2 = __floats2half2_rn(0.0f, 0.0f), a3 = __floats2half2_rn(0.0f, 0.0f);
        __half2 b0 = __floats2half2_rn(0.0f, 0.0f), b1 = __floats2half2_rn(0.0f, 0.0f);
        __half2 b2 = __floats2half2_rn(0.0f, 0.0f), b3 = __floats2half2_rn(0.0f, 0.0f);

        const __half2 * x1_0 = T2 ? (s_x1[xb + 0].x + is * 8) : nullptr;
        const __half2 * x1_1 = T2 ? (s_x1[xb + 1].x + is * 8) : nullptr;
        const __half2 * x1_2 = T2 ? (s_x1[xb + 2].x + is * 8) : nullptr;
        const __half2 * x1_3 = T2 ? (s_x1[xb + 3].x + is * 8) : nullptr;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t q1 = (ql0w[c] & 0x0F0F0F0Fu) | ((qhw[c] & 0x03030303u) << 4);
            const uint32_t q2 = (ql1w[c] & 0x0F0F0F0Fu) | ((qhw[c] & 0x0C0C0C0Cu) << 2);
            const uint32_t q3 = ((ql0w[c] >> 4) & 0x0F0F0F0Fu) | (qhw[c] & 0x30303030u);
            const uint32_t q4 = ((ql1w[c] >> 4) & 0x0F0F0F0Fu) | ((qhw[c] >> 2) & 0x30303030u);

            const __half2 w1a = mw_hadd2(mw_u8x2_to_h2_lo(q1), neg1056), w1b = mw_hadd2(mw_u8x2_to_h2_hi(q1), neg1056);
            const __half2 w2a = mw_hadd2(mw_u8x2_to_h2_lo(q2), neg1056), w2b = mw_hadd2(mw_u8x2_to_h2_hi(q2), neg1056);
            const __half2 w3a = mw_hadd2(mw_u8x2_to_h2_lo(q3), neg1056), w3b = mw_hadd2(mw_u8x2_to_h2_hi(q3), neg1056);
            const __half2 w4a = mw_hadd2(mw_u8x2_to_h2_lo(q4), neg1056), w4b = mw_hadd2(mw_u8x2_to_h2_hi(q4), neg1056);

            a0 = mw_hfma2(w1a, x0_0[c * 2 + 0], a0); a0 = mw_hfma2(w1b, x0_0[c * 2 + 1], a0);
            a1 = mw_hfma2(w2a, x0_1[c * 2 + 0], a1); a1 = mw_hfma2(w2b, x0_1[c * 2 + 1], a1);
            a2 = mw_hfma2(w3a, x0_2[c * 2 + 0], a2); a2 = mw_hfma2(w3b, x0_2[c * 2 + 1], a2);
            a3 = mw_hfma2(w4a, x0_3[c * 2 + 0], a3); a3 = mw_hfma2(w4b, x0_3[c * 2 + 1], a3);

            if constexpr (T2) {
                b0 = mw_hfma2(w1a, x1_0[c * 2 + 0], b0); b0 = mw_hfma2(w1b, x1_0[c * 2 + 1], b0);
                b1 = mw_hfma2(w2a, x1_1[c * 2 + 0], b1); b1 = mw_hfma2(w2b, x1_1[c * 2 + 1], b1);
                b2 = mw_hfma2(w3a, x1_2[c * 2 + 0], b2); b2 = mw_hfma2(w3b, x1_2[c * 2 + 1], b2);
                b3 = mw_hfma2(w4a, x1_3[c * 2 + 0], b3); b3 = mw_hfma2(w4b, x1_3[c * 2 + 1], b3);
            }
        }
        const float2 f0 = __half22float2(a0), f1 = __half22float2(a1), f2 = __half22float2(a2), f3 = __half22float2(a3);
        acc0 += (sc0 * s_x0[xb + 0].d) * (f0.x + f0.y) + (sc2 * s_x0[xb + 1].d) * (f1.x + f1.y)
              + (sc4 * s_x0[xb + 2].d) * (f2.x + f2.y) + (sc6 * s_x0[xb + 3].d) * (f3.x + f3.y);
        if constexpr (T2) {
            const float2 g0 = __half22float2(b0), g1 = __half22float2(b1), g2 = __half22float2(b2), g3 = __half22float2(b3);
            acc1 += (sc0 * s_x1[xb + 0].d) * (g0.x + g0.y) + (sc2 * s_x1[xb + 1].d) * (g1.x + g1.y)
                  + (sc4 * s_x1[xb + 2].d) * (g2.x + g2.y) + (sc6 * s_x1[xb + 3].d) * (g3.x + g3.y);
        }
    }
    out0 = mw_warp_sum(acc0);
    if constexpr (T2) out1 = mw_warp_sum(acc1);
}

// ============================================================================
// 4. Linear GEMV Kernels (dp4a Q3_K SoA + sm_60 half2 Q3_K/Q4_K/Q5_K/Q6_K!)
// ============================================================================
template <bool T2>
__global__ void mw_gemv_q3_k_soa_dp4a_kernel(
    const int4 *          __restrict__ W_qs,
    const int4 *          __restrict__ W_hm,
    const uint32_t *      __restrict__ W_sc,
    const uint16_t *      __restrict__ W_d,
    const BlockQ8_1_Act * __restrict__ X,
    float *               __restrict__ Y,
    int M, int K) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<BlockQ8_1_Act *>(smem_raw);
    const int t        = T2 ? 0 : blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int n_i4     = ((T2 ? 2 : 1) * K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(T2 ? X : (X + static_cast<size_t>(t) * K_blocks));
    int4 *       sx_ou = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_ou[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const size_t rblk = static_cast<size_t>(row) * nb;
    if constexpr (!T2) {
        const float acc = dot_q3_k_soa_row_dp4a(
            W_qs + rblk * 4, W_hm + rblk * 2, W_sc + rblk * 3, W_d + rblk, s_x, nb, threadIdx.x);
        if (threadIdx.x == 0) Y[static_cast<size_t>(t) * M + row] = acc;
    } else {
        float a0 = 0.0f, a1 = 0.0f;
        dot_q3_k_soa_row_dp4a_t2(
            W_qs + rblk * 4, W_hm + rblk * 2, W_sc + rblk * 3, W_d + rblk, s_x, s_x + K_blocks, nb, threadIdx.x, a0, a1);
        if (threadIdx.x == 0) { Y[row] = a0; Y[M + row] = a1; }
    }
}

struct MwMultiProjDesc {
    int          n_mat;
    int          dt[6]; // 0=Q3_K_soa, 1=Q4_K, 2=Q5_K, 3=Q6_K_soa
    const void * p0[6];
    const void * p1[6];
    const void * p2[6];
    const void * p3[6];
    float *      y[6];
    int          m[6];
    int          row_end[6];
};

template <bool T2>
__global__ void mw_gemv_multi_h2_kernel(
    MwMultiProjDesc desc, const ActF16Block * __restrict__ X, int K) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<ActF16Block *>(smem_raw);
    const int t        = T2 ? 0 : blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int n_i4     = ((T2 ? 2 : 1) * K_blocks * static_cast<int>(sizeof(ActF16Block))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(T2 ? X : (X + static_cast<size_t>(t) * K_blocks));
    int4 *       sx_ou = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_ou[i] = sx_in[i];
    __syncthreads();

    const int grow  = blockIdx.x * blockDim.y + threadIdx.y;
    const int tot_m = desc.row_end[desc.n_mat - 1];
    if (grow >= tot_m) return;

    int mi = 0;
    while (mi + 1 < desc.n_mat && grow >= desc.row_end[mi]) ++mi;
    const int row = grow - ((mi > 0) ? desc.row_end[mi - 1] : 0);
    const int M   = desc.m[mi];
    const size_t rblk = static_cast<size_t>(row) * nb;
    const ActF16Block * s_x0 = s_x;
    const ActF16Block * s_x1 = T2 ? (s_x + K_blocks) : nullptr;

    float a0 = 0.0f, a1 = 0.0f;
    const int dt = desc.dt[mi];
    if (dt == 0) {
        dot_q3_k_soa_row_h2<T2>(
            static_cast<const int4 *>(desc.p0[mi]) + rblk * 4,
            static_cast<const int4 *>(desc.p1[mi]) + rblk * 2,
            static_cast<const uint32_t *>(desc.p2[mi]) + rblk * 3,
            static_cast<const uint16_t *>(desc.p3[mi]) + rblk,
            s_x0, s_x1, nb, threadIdx.x, a0, a1);
    } else if (dt == 1) {
        dot_q4_k_row_h2<T2>(static_cast<const BlockQ4_K_M *>(desc.p0[mi]) + rblk, s_x0, s_x1, nb, threadIdx.x, a0, a1);
    } else if (dt == 2) {
        dot_q5_k_row_h2<T2>(static_cast<const BlockQ5_K_M *>(desc.p0[mi]) + rblk, s_x0, s_x1, nb, threadIdx.x, a0, a1);
    } else {
        dot_q6_k_soa_row_h2<T2>(
            static_cast<const int4 *>(desc.p0[mi]) + rblk * 8,
            static_cast<const int4 *>(desc.p1[mi]) + rblk * 4,
            static_cast<const int8_t *>(desc.p2[mi]) + rblk * 16,
            static_cast<const uint16_t *>(desc.p3[mi]) + rblk,
            s_x0, s_x1, nb, threadIdx.x, a0, a1);
    }

    if (threadIdx.x == 0) {
        desc.y[mi][static_cast<size_t>(t) * M + row] = a0;
        if constexpr (T2) desc.y[mi][M + row] = a1;
    }
}

template <bool T2>
__global__ __launch_bounds__(256) void mw_gemv_q4_k_h2_kernel(
    const BlockQ4_K_M * __restrict__ W,
    const ActF16Block * __restrict__ X,
    float *             __restrict__ Y,
    int M, int K) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<ActF16Block *>(smem_raw);
    const int t        = T2 ? 0 : blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int n_i4     = ((T2 ? 2 : 1) * K_blocks * static_cast<int>(sizeof(ActF16Block))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(T2 ? X : (X + static_cast<size_t>(t) * K_blocks));
    int4 *       sx_ou = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_ou[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    float a0 = 0.0f, a1 = 0.0f;
    dot_q4_k_row_h2<T2>(
        W + static_cast<size_t>(row) * nb, s_x, T2 ? (s_x + K_blocks) : nullptr, nb, threadIdx.x, a0, a1);
    if (threadIdx.x == 0) {
        Y[static_cast<size_t>(t) * M + row] = a0;
        if constexpr (T2) Y[M + row] = a1;
    }
}

template <bool T2, int R_PER_WARP = 1>
__global__ __launch_bounds__(256) void mw_gemv_q6_k_soa_h2_kernel(
    const int4 *        __restrict__ W_ql,
    const int4 *        __restrict__ W_qh,
    const int8_t *      __restrict__ W_sc,
    const uint16_t *    __restrict__ W_d,
    const ActF16Block * __restrict__ X,
    float *             __restrict__ Y,
    int M, int K) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<ActF16Block *>(smem_raw);
    const int t        = T2 ? 0 : blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int n_i4     = ((T2 ? 2 : 1) * K_blocks * static_cast<int>(sizeof(ActF16Block))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(T2 ? X : (X + static_cast<size_t>(t) * K_blocks));
    int4 *       sx_ou = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_ou[i] = sx_in[i];
    __syncthreads();

    const int base_row = (blockIdx.x * blockDim.y + threadIdx.y) * R_PER_WARP;
    #pragma unroll
    for (int r = 0; r < R_PER_WARP; ++r) {
        const int row = base_row + r;
        if (row >= M) return;
        const size_t rblk = static_cast<size_t>(row) * nb;
        float a0 = 0.0f, a1 = 0.0f;
        dot_q6_k_soa_row_h2<T2>(
            W_ql + rblk * 8, W_qh + rblk * 4, W_sc + rblk * 16, W_d + rblk,
            s_x, T2 ? (s_x + K_blocks) : nullptr, nb, threadIdx.x, a0, a1);
        if (threadIdx.x == 0) {
            Y[static_cast<size_t>(t) * M + row] = a0;
            if constexpr (T2) Y[M + row] = a1;
        }
    }
}

// ============================================================================
// 4B. Zero-Bank-Conflict Sub-Warp Multicast Q4_K (8 thr/row) & Q6_K SoA (4 thr/row)
// ============================================================================
// All rows within a warp iterate over super-blocks bi in lockstep so every SMEM
// LDS.128 load is hardware-multicast across 4 (Q4_K) or 8 (Q6_K) rows with ZERO
// bank conflicts, while tiling K at TILE_NB=16 keeps SMEM <= 10 KB even at K=12288!
static constexpr int MW_SW_TILE_NB = 16;

template <bool T2>
__device__ __forceinline__ void dot_q4_k_sw8_eo_tile_h2(
    const BlockQ4_K_M * __restrict__ w_tile,
    const ActF16Block * __restrict__ s_x0,
    const ActF16Block * __restrict__ s_x1,
    int bc, int sub, float & acc0, float & acc1) {

    const int j  = sub >> 1;
    const int is = sub & 1;

    for (int bi = 0; bi < bc; ++bi) {
        const BlockQ4_K_M & b = w_tile[bi];
        const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
        const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
        const float d    = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu)) * 16.0f;
        const float dmin = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
        float sc1, m1, sc2, m2;
        mw_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
        mw_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
        const float d1 = d * sc1, mn1_eff = dmin * m1 + d1;
        const float d2 = d * sc2, mn2_eff = dmin * m2 + d2;

        const uint32_t qws[4] = {
            static_cast<uint32_t>(ql_v.x), static_cast<uint32_t>(ql_v.y),
            static_cast<uint32_t>(ql_v.z), static_cast<uint32_t>(ql_v.w)
        };
        const int xb = bi * 8 + 2 * j;
        const int4 * x10_4 = reinterpret_cast<const int4 *>(s_x0[xb + 0].x + is * 8);
        const int4 * x20_4 = reinterpret_cast<const int4 *>(s_x0[xb + 1].x + is * 8);
        const int4 a10 = x10_4[0], b10 = x10_4[1];
        const int4 a20 = x20_4[0], b20 = x20_4[1];
        const __half2 x10[8] = {
            *reinterpret_cast<const __half2 *>(&a10.x), *reinterpret_cast<const __half2 *>(&a10.y),
            *reinterpret_cast<const __half2 *>(&a10.z), *reinterpret_cast<const __half2 *>(&a10.w),
            *reinterpret_cast<const __half2 *>(&b10.x), *reinterpret_cast<const __half2 *>(&b10.y),
            *reinterpret_cast<const __half2 *>(&b10.z), *reinterpret_cast<const __half2 *>(&b10.w)
        };
        const __half2 x20[8] = {
            *reinterpret_cast<const __half2 *>(&a20.x), *reinterpret_cast<const __half2 *>(&a20.y),
            *reinterpret_cast<const __half2 *>(&a20.z), *reinterpret_cast<const __half2 *>(&a20.w),
            *reinterpret_cast<const __half2 *>(&b20.x), *reinterpret_cast<const __half2 *>(&b20.y),
            *reinterpret_cast<const __half2 *>(&b20.z), *reinterpret_cast<const __half2 *>(&b20.w)
        };

        __half2 h10 = __floats2half2_rn(0.0f, 0.0f), h20 = __floats2half2_rn(0.0f, 0.0f);
        __half2 h11 = __floats2half2_rn(0.0f, 0.0f), h21 = __floats2half2_rn(0.0f, 0.0f);
        const __half2 * x11 = T2 ? (s_x1[xb + 0].x + is * 8) : nullptr;
        const __half2 * x21 = T2 ? (s_x1[xb + 1].x + is * 8) : nullptr;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t q = qws[c];
            const __half2 w1a = mw_bits_to_h2(((q << 6) & 0x03C003C0u) | 0x3C003C00u);
            const __half2 w1b = mw_bits_to_h2(((q >> 2) & 0x03C003C0u) | 0x3C003C00u);
            const __half2 w2a = mw_bits_to_h2(((q << 2) & 0x03C003C0u) | 0x3C003C00u);
            const __half2 w2b = mw_bits_to_h2(((q >> 6) & 0x03C003C0u) | 0x3C003C00u);
            h10 = mw_hfma2(w1a, x10[c * 2 + 0], h10); h10 = mw_hfma2(w1b, x10[c * 2 + 1], h10);
            h20 = mw_hfma2(w2a, x20[c * 2 + 0], h20); h20 = mw_hfma2(w2b, x20[c * 2 + 1], h20);
            if constexpr (T2) {
                h11 = mw_hfma2(w1a, x11[c * 2 + 0], h11); h11 = mw_hfma2(w1b, x11[c * 2 + 1], h11);
                h21 = mw_hfma2(w2a, x21[c * 2 + 0], h21); h21 = mw_hfma2(w2b, x21[c * 2 + 1], h21);
            }
        }
        const float2 f10 = __half22float2(h10), f20 = __half22float2(h20);
        const float s10 = is ? s_x0[xb + 0].s1 : s_x0[xb + 0].s0;
        const float s20 = is ? s_x0[xb + 1].s1 : s_x0[xb + 1].s0;
        acc0 += (d1 * s_x0[xb + 0].d) * (f10.x + f10.y) - mn1_eff * s10
              + (d2 * s_x0[xb + 1].d) * (f20.x + f20.y) - mn2_eff * s20;
        if constexpr (T2) {
            const float2 f11 = __half22float2(h11), f21 = __half22float2(h21);
            const float s11 = is ? s_x1[xb + 0].s1 : s_x1[xb + 0].s0;
            const float s21 = is ? s_x1[xb + 1].s1 : s_x1[xb + 1].s0;
            acc1 += (d1 * s_x1[xb + 0].d) * (f11.x + f11.y) - mn1_eff * s11
                  + (d2 * s_x1[xb + 1].d) * (f21.x + f21.y) - mn2_eff * s21;
        }
    }
}

template <bool T2>
__global__ void mw_gemv_q4_k_sw8_h2_kernel(
    const BlockQ4_K_M * __restrict__ W,
    const ActF16Block * __restrict__ X,
    float *             __restrict__ Y,
    int M, int K, int tile_nb) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x0 = reinterpret_cast<ActF16Block *>(smem_raw);
    auto * s_x1 = T2 ? (s_x0 + tile_nb * 8) : nullptr;
    const int t        = T2 ? 0 : blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int sub      = threadIdx.x & 7;
    const int r_in_w   = threadIdx.x >> 3;
    const int row      = (blockIdx.x * blockDim.y + threadIdx.y) * 4 + r_in_w;

    const ActF16Block * x_t0 = X + static_cast<size_t>(t) * K_blocks;
    const ActF16Block * x_t1 = T2 ? (X + K_blocks) : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += tile_nb) {
        const int bc   = (b0 + tile_nb < nb) ? tile_nb : (nb - b0);
        const int n_i4 = (bc * 8 * static_cast<int>(sizeof(ActF16Block))) / 16;
        const int4 * src0 = reinterpret_cast<const int4 *>(x_t0 + b0 * 8);
        int4 *       dst0 = reinterpret_cast<int4 *>(s_x0);
        for (int i = tid; i < n_i4; i += nthr) dst0[i] = __ldg(&src0[i]);
        if constexpr (T2) {
            const int4 * src1 = reinterpret_cast<const int4 *>(x_t1 + b0 * 8);
            int4 *       dst1 = reinterpret_cast<int4 *>(s_x1);
            for (int i = tid; i < n_i4; i += nthr) dst1[i] = __ldg(&src1[i]);
        }
        __syncthreads();

        if (row < M) {
            dot_q4_k_sw8_eo_tile_h2<T2>(W + static_cast<size_t>(row) * nb + b0, s_x0, s_x1, bc, sub, acc0, acc1);
        }
        if (b0 + tile_nb < nb) __syncthreads();
    }

    if (row < M) {
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, 4, 8);
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2, 8);
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1, 8);
        if constexpr (T2) {
            acc1 += __shfl_xor_sync(0xffffffffu, acc1, 4, 8);
            acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2, 8);
            acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1, 8);
        }
        if (sub == 0) {
            Y[static_cast<size_t>(t) * M + row] = acc0;
            if constexpr (T2) Y[M + row] = acc1;
        }
    }
}

template <bool T2>
__global__ void mw_gemv_q4_k_multi_sw8_h2_kernel(
    const BlockQ4_K_M * __restrict__ w0, float * __restrict__ y0, int m0, int e0,
    const BlockQ4_K_M * __restrict__ w1, float * __restrict__ y1, int m1, int e1,
    const BlockQ4_K_M * __restrict__ w2, float * __restrict__ y2, int m2, int e2,
    const BlockQ4_K_M * __restrict__ w3, float * __restrict__ y3, int m3, int e3,
    const BlockQ4_K_M * __restrict__ w4, float * __restrict__ y4, int m4, int e4,
    const BlockQ4_K_M * __restrict__ w5, float * __restrict__ y5, int m5, int e5,
    int tot_m, const ActF16Block * __restrict__ X, int K, int tile_nb) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x0 = reinterpret_cast<ActF16Block *>(smem_raw);
    auto * s_x1 = T2 ? (s_x0 + tile_nb * 8) : nullptr;
    const int t        = T2 ? 0 : blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int sub      = threadIdx.x & 7;
    const int r_in_w   = threadIdx.x >> 3;
    const int grow     = (blockIdx.x * blockDim.y + threadIdx.y) * 4 + r_in_w;

    const BlockQ4_K_M * w_row = nullptr;
    float * y_dst = nullptr;
    int m_val = 0, l_row = 0;
    if (grow < tot_m) {
        if (grow < e0)      { l_row = grow;      w_row = w0 + static_cast<size_t>(l_row) * nb; y_dst = y0; m_val = m0; }
        else if (grow < e1) { l_row = grow - e0; w_row = w1 + static_cast<size_t>(l_row) * nb; y_dst = y1; m_val = m1; }
        else if (grow < e2) { l_row = grow - e1; w_row = w2 + static_cast<size_t>(l_row) * nb; y_dst = y2; m_val = m2; }
        else if (grow < e3) { l_row = grow - e2; w_row = w3 + static_cast<size_t>(l_row) * nb; y_dst = y3; m_val = m3; }
        else if (grow < e4) { l_row = grow - e3; w_row = w4 + static_cast<size_t>(l_row) * nb; y_dst = y4; m_val = m4; }
        else                { l_row = grow - e4; w_row = w5 + static_cast<size_t>(l_row) * nb; y_dst = y5; m_val = m5; }
    }

    const ActF16Block * x_t0 = X + static_cast<size_t>(t) * K_blocks;
    const ActF16Block * x_t1 = T2 ? (X + K_blocks) : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += tile_nb) {
        const int bc   = (b0 + tile_nb < nb) ? tile_nb : (nb - b0);
        const int n_i4 = (bc * 8 * static_cast<int>(sizeof(ActF16Block))) / 16;
        const int4 * src0 = reinterpret_cast<const int4 *>(x_t0 + b0 * 8);
        int4 *       dst0 = reinterpret_cast<int4 *>(s_x0);
        for (int i = tid; i < n_i4; i += nthr) dst0[i] = __ldg(&src0[i]);
        if constexpr (T2) {
            const int4 * src1 = reinterpret_cast<const int4 *>(x_t1 + b0 * 8);
            int4 *       dst1 = reinterpret_cast<int4 *>(s_x1);
            for (int i = tid; i < n_i4; i += nthr) dst1[i] = __ldg(&src1[i]);
        }
        __syncthreads();

        if (w_row != nullptr) {
            dot_q4_k_sw8_eo_tile_h2<T2>(w_row + b0, s_x0, s_x1, bc, sub, acc0, acc1);
        }
        if (b0 + tile_nb < nb) __syncthreads();
    }

    if (w_row != nullptr) {
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, 4, 8);
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2, 8);
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1, 8);
        if constexpr (T2) {
            acc1 += __shfl_xor_sync(0xffffffffu, acc1, 4, 8);
            acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2, 8);
            acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1, 8);
        }
        if (sub == 0) {
            y_dst[static_cast<size_t>(t) * m_val + l_row] = acc0;
            if constexpr (T2) y_dst[m_val + l_row] = acc1;
        }
    }
}

template <bool T2>
__device__ __forceinline__ void dot_q4_k_eo32_tile_h2(
    const BlockQ4_K_M * __restrict__ w_tile,
    const ActF16Block * __restrict__ s_x0,
    const ActF16Block * __restrict__ s_x1,
    int bc, int lane, float & acc0, float & acc1) {

    const int blk_in_w = lane >> 3;
    const int sub      = lane & 7;
    const int j        = sub >> 1;
    const int is       = sub & 1;

    for (int bi = blk_in_w; bi < bc; bi += 4) {
        const BlockQ4_K_M & b = w_tile[bi];
        const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
        const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
        const float d    = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu)) * 16.0f;
        const float dmin = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
        float sc1, m1, sc2, m2;
        mw_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
        mw_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
        const float d1 = d * sc1, mn1_eff = dmin * m1 + d1;
        const float d2 = d * sc2, mn2_eff = dmin * m2 + d2;

        const uint32_t qws[4] = {
            static_cast<uint32_t>(ql_v.x), static_cast<uint32_t>(ql_v.y),
            static_cast<uint32_t>(ql_v.z), static_cast<uint32_t>(ql_v.w)
        };
        const int xb = bi * 8 + 2 * j;
        const int4 * x10_4 = reinterpret_cast<const int4 *>(s_x0[xb + 0].x + is * 8);
        const int4 * x20_4 = reinterpret_cast<const int4 *>(s_x0[xb + 1].x + is * 8);
        const int4 a10 = x10_4[0], b10 = x10_4[1];
        const int4 a20 = x20_4[0], b20 = x20_4[1];
        const __half2 x10[8] = {
            *reinterpret_cast<const __half2 *>(&a10.x), *reinterpret_cast<const __half2 *>(&a10.y),
            *reinterpret_cast<const __half2 *>(&a10.z), *reinterpret_cast<const __half2 *>(&a10.w),
            *reinterpret_cast<const __half2 *>(&b10.x), *reinterpret_cast<const __half2 *>(&b10.y),
            *reinterpret_cast<const __half2 *>(&b10.z), *reinterpret_cast<const __half2 *>(&b10.w)
        };
        const __half2 x20[8] = {
            *reinterpret_cast<const __half2 *>(&a20.x), *reinterpret_cast<const __half2 *>(&a20.y),
            *reinterpret_cast<const __half2 *>(&a20.z), *reinterpret_cast<const __half2 *>(&a20.w),
            *reinterpret_cast<const __half2 *>(&b20.x), *reinterpret_cast<const __half2 *>(&b20.y),
            *reinterpret_cast<const __half2 *>(&b20.z), *reinterpret_cast<const __half2 *>(&b20.w)
        };

        __half2 h10 = __floats2half2_rn(0.0f, 0.0f), h20 = __floats2half2_rn(0.0f, 0.0f);
        __half2 h11 = __floats2half2_rn(0.0f, 0.0f), h21 = __floats2half2_rn(0.0f, 0.0f);
        const __half2 * x11 = T2 ? (s_x1[xb + 0].x + is * 8) : nullptr;
        const __half2 * x21 = T2 ? (s_x1[xb + 1].x + is * 8) : nullptr;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t q = qws[c];
            const __half2 w1a = mw_bits_to_h2(((q << 6) & 0x03C003C0u) | 0x3C003C00u);
            const __half2 w1b = mw_bits_to_h2(((q >> 2) & 0x03C003C0u) | 0x3C003C00u);
            const __half2 w2a = mw_bits_to_h2(((q << 2) & 0x03C003C0u) | 0x3C003C00u);
            const __half2 w2b = mw_bits_to_h2(((q >> 6) & 0x03C003C0u) | 0x3C003C00u);
            h10 = mw_hfma2(w1a, x10[c * 2 + 0], h10); h10 = mw_hfma2(w1b, x10[c * 2 + 1], h10);
            h20 = mw_hfma2(w2a, x20[c * 2 + 0], h20); h20 = mw_hfma2(w2b, x20[c * 2 + 1], h20);
            if constexpr (T2) {
                h11 = mw_hfma2(w1a, x11[c * 2 + 0], h11); h11 = mw_hfma2(w1b, x11[c * 2 + 1], h11);
                h21 = mw_hfma2(w2a, x21[c * 2 + 0], h21); h21 = mw_hfma2(w2b, x21[c * 2 + 1], h21);
            }
        }
        const float2 f10 = __half22float2(h10), f20 = __half22float2(h20);
        const float s10 = is ? s_x0[xb + 0].s1 : s_x0[xb + 0].s0;
        const float s20 = is ? s_x0[xb + 1].s1 : s_x0[xb + 1].s0;
        acc0 += (d1 * s_x0[xb + 0].d) * (f10.x + f10.y) - mn1_eff * s10
              + (d2 * s_x0[xb + 1].d) * (f20.x + f20.y) - mn2_eff * s20;
        if constexpr (T2) {
            const float2 f11 = __half22float2(h11), f21 = __half22float2(h21);
            const float s11 = is ? s_x1[xb + 0].s1 : s_x1[xb + 0].s0;
            const float s21 = is ? s_x1[xb + 1].s1 : s_x1[xb + 1].s0;
            acc1 += (d1 * s_x1[xb + 0].d) * (f11.x + f11.y) - mn1_eff * s11
                  + (d2 * s_x1[xb + 1].d) * (f21.x + f21.y) - mn2_eff * s21;
        }
    }
}

template <bool T2>
__global__ __launch_bounds__(256) void mw_gemv_q4_k_eo32_h2_kernel(
    const BlockQ4_K_M * __restrict__ W,
    const ActF16Block * __restrict__ X,
    float *             __restrict__ Y,
    int M, int K, int tile_nb) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x0 = reinterpret_cast<ActF16Block *>(smem_raw);
    auto * s_x1 = T2 ? (s_x0 + tile_nb * 8) : nullptr;
    const int t        = T2 ? 0 : blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int row      = blockIdx.x * blockDim.y + threadIdx.y;

    const ActF16Block * x_t0 = X + static_cast<size_t>(t) * K_blocks;
    const ActF16Block * x_t1 = T2 ? (X + K_blocks) : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += tile_nb) {
        const int bc   = (b0 + tile_nb < nb) ? tile_nb : (nb - b0);
        const int n_i4 = (bc * 8 * static_cast<int>(sizeof(ActF16Block))) / 16;
        const int4 * src0 = reinterpret_cast<const int4 *>(x_t0 + b0 * 8);
        int4 *       dst0 = reinterpret_cast<int4 *>(s_x0);
        for (int i = tid; i < n_i4; i += nthr) dst0[i] = __ldg(&src0[i]);
        if constexpr (T2) {
            const int4 * src1 = reinterpret_cast<const int4 *>(x_t1 + b0 * 8);
            int4 *       dst1 = reinterpret_cast<int4 *>(s_x1);
            for (int i = tid; i < n_i4; i += nthr) dst1[i] = __ldg(&src1[i]);
        }
        __syncthreads();

        if (row < M) {
            dot_q4_k_eo32_tile_h2<T2>(W + static_cast<size_t>(row) * nb + b0, s_x0, s_x1, bc, threadIdx.x, acc0, acc1);
        }
        if (b0 + tile_nb < nb) __syncthreads();
    }

    if (row < M) {
        const float v0 = mw_warp_sum(acc0);
        const float v1 = T2 ? mw_warp_sum(acc1) : 0.0f;
        if (threadIdx.x == 0) {
            Y[static_cast<size_t>(t) * M + row] = v0;
            if constexpr (T2) Y[M + row] = v1;
        }
    }
}

template <bool T2>
__global__ __launch_bounds__(256) void mw_gemv_q4_k_multi_eo32_h2_kernel(
    const BlockQ4_K_M * __restrict__ w0, float * __restrict__ y0, int m0, int e0,
    const BlockQ4_K_M * __restrict__ w1, float * __restrict__ y1, int m1, int e1,
    const BlockQ4_K_M * __restrict__ w2, float * __restrict__ y2, int m2, int e2,
    const BlockQ4_K_M * __restrict__ w3, float * __restrict__ y3, int m3, int e3,
    const BlockQ4_K_M * __restrict__ w4, float * __restrict__ y4, int m4, int e4,
    const BlockQ4_K_M * __restrict__ w5, float * __restrict__ y5, int m5, int e5,
    int tot_m, const ActF16Block * __restrict__ X, int K, int tile_nb) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x0 = reinterpret_cast<ActF16Block *>(smem_raw);
    auto * s_x1 = T2 ? (s_x0 + tile_nb * 8) : nullptr;
    const int t        = T2 ? 0 : blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int grow     = blockIdx.x * blockDim.y + threadIdx.y;

    const BlockQ4_K_M * w_row = nullptr;
    float * y_dst = nullptr;
    int m_val = 0, l_row = 0;
    if (grow < tot_m) {
        if (grow < e0)      { l_row = grow;      w_row = w0 + static_cast<size_t>(l_row) * nb; y_dst = y0; m_val = m0; }
        else if (grow < e1) { l_row = grow - e0; w_row = w1 + static_cast<size_t>(l_row) * nb; y_dst = y1; m_val = m1; }
        else if (grow < e2) { l_row = grow - e1; w_row = w2 + static_cast<size_t>(l_row) * nb; y_dst = y2; m_val = m2; }
        else if (grow < e3) { l_row = grow - e2; w_row = w3 + static_cast<size_t>(l_row) * nb; y_dst = y3; m_val = m3; }
        else if (grow < e4) { l_row = grow - e3; w_row = w4 + static_cast<size_t>(l_row) * nb; y_dst = y4; m_val = m4; }
        else                { l_row = grow - e4; w_row = w5 + static_cast<size_t>(l_row) * nb; y_dst = y5; m_val = m5; }
    }

    const ActF16Block * x_t0 = X + static_cast<size_t>(t) * K_blocks;
    const ActF16Block * x_t1 = T2 ? (X + K_blocks) : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += tile_nb) {
        const int bc   = (b0 + tile_nb < nb) ? tile_nb : (nb - b0);
        const int n_i4 = (bc * 8 * static_cast<int>(sizeof(ActF16Block))) / 16;
        const int4 * src0 = reinterpret_cast<const int4 *>(x_t0 + b0 * 8);
        int4 *       dst0 = reinterpret_cast<int4 *>(s_x0);
        for (int i = tid; i < n_i4; i += nthr) dst0[i] = __ldg(&src0[i]);
        if constexpr (T2) {
            const int4 * src1 = reinterpret_cast<const int4 *>(x_t1 + b0 * 8);
            int4 *       dst1 = reinterpret_cast<int4 *>(s_x1);
            for (int i = tid; i < n_i4; i += nthr) dst1[i] = __ldg(&src1[i]);
        }
        __syncthreads();

        if (w_row != nullptr) {
            dot_q4_k_eo32_tile_h2<T2>(w_row + b0, s_x0, s_x1, bc, threadIdx.x, acc0, acc1);
        }
        if (b0 + tile_nb < nb) __syncthreads();
    }

    if (w_row != nullptr) {
        const float v0 = mw_warp_sum(acc0);
        const float v1 = T2 ? mw_warp_sum(acc1) : 0.0f;
        if (threadIdx.x == 0) {
            y_dst[static_cast<size_t>(t) * m_val + l_row] = v0;
            if constexpr (T2) y_dst[m_val + l_row] = v1;
        }
    }
}

template <bool T2>
__global__ __launch_bounds__(256) void mw_gemv_q6_k_soa_tiled_h2_kernel(
    const int4 *        __restrict__ W_ql,
    const int4 *        __restrict__ W_qh,
    const int8_t *      __restrict__ W_sc,
    const uint16_t *    __restrict__ W_d,
    const ActF16Block * __restrict__ X,
    float *             __restrict__ Y,
    int M, int K, int tile_nb) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x0 = reinterpret_cast<ActF16Block *>(smem_raw);
    auto * s_x1 = T2 ? (s_x0 + tile_nb * 8) : nullptr;
    const int t        = T2 ? 0 : blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int row      = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane     = threadIdx.x;
    const int blk_in_w = lane >> 2;
    const int sub      = lane & 3;
    const int h        = sub >> 1;
    const int is       = sub & 1;
    const __half2 neg1056 = __floats2half2_rn(-1056.0f, -1056.0f);

    const ActF16Block * x_t0 = X + static_cast<size_t>(t) * K_blocks;
    const ActF16Block * x_t1 = T2 ? (X + K_blocks) : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += tile_nb) {
        const int bc   = (b0 + tile_nb < nb) ? tile_nb : (nb - b0);
        const int n_i4 = (bc * 8 * static_cast<int>(sizeof(ActF16Block))) / 16;
        const int4 * src0 = reinterpret_cast<const int4 *>(x_t0 + b0 * 8);
        int4 *       dst0 = reinterpret_cast<int4 *>(s_x0);
        for (int i = tid; i < n_i4; i += nthr) dst0[i] = __ldg(&src0[i]);
        if constexpr (T2) {
            const int4 * src1 = reinterpret_cast<const int4 *>(x_t1 + b0 * 8);
            int4 *       dst1 = reinterpret_cast<int4 *>(s_x1);
            for (int i = tid; i < n_i4; i += nthr) dst1[i] = __ldg(&src1[i]);
        }
        __syncthreads();

        if (row < M) {
            const size_t rblk = static_cast<size_t>(row) * nb + b0;
            const int4 *     W_ql_t = W_ql + rblk * 8;
            const int4 *     W_qh_t = W_qh + rblk * 4;
            const int8_t *   W_sc_t = W_sc + rblk * 16;
            const uint16_t * W_d_t  = W_d  + rblk;
            for (int bi = blk_in_w; bi < bc; bi += 8) {
                const int4 qh_v  = __ldg(&W_qh_t[bi * 4 + sub]);
                const int4 ql0_v = __ldg(&W_ql_t[bi * 8 + h * 4 + is]);
                const int4 ql1_v = __ldg(&W_ql_t[bi * 8 + h * 4 + 2 + is]);
                const int8_t * sc = W_sc_t + bi * 16 + h * 8 + is;
                const float dw  = mw_half_bits_to_f32(__ldg(&W_d_t[bi]));
                const float sc0 = dw * static_cast<float>(__ldg(&sc[0]));
                const float sc2 = dw * static_cast<float>(__ldg(&sc[2]));
                const float sc4 = dw * static_cast<float>(__ldg(&sc[4]));
                const float sc6 = dw * static_cast<float>(__ldg(&sc[6]));

                const uint32_t ql0w[4] = { static_cast<uint32_t>(ql0_v.x), static_cast<uint32_t>(ql0_v.y), static_cast<uint32_t>(ql0_v.z), static_cast<uint32_t>(ql0_v.w) };
                const uint32_t ql1w[4] = { static_cast<uint32_t>(ql1_v.x), static_cast<uint32_t>(ql1_v.y), static_cast<uint32_t>(ql1_v.z), static_cast<uint32_t>(ql1_v.w) };
                const uint32_t qhw[4]  = { static_cast<uint32_t>(qh_v.x),  static_cast<uint32_t>(qh_v.y),  static_cast<uint32_t>(qh_v.z),  static_cast<uint32_t>(qh_v.w)  };

                const int xb = bi * 8 + h * 4;
                const __half2 * x0_0 = s_x0[xb + 0].x + is * 8;
                const __half2 * x0_1 = s_x0[xb + 1].x + is * 8;
                const __half2 * x0_2 = s_x0[xb + 2].x + is * 8;
                const __half2 * x0_3 = s_x0[xb + 3].x + is * 8;
                const __half2 * x1_0 = T2 ? (s_x1[xb + 0].x + is * 8) : nullptr;
                const __half2 * x1_1 = T2 ? (s_x1[xb + 1].x + is * 8) : nullptr;
                const __half2 * x1_2 = T2 ? (s_x1[xb + 2].x + is * 8) : nullptr;
                const __half2 * x1_3 = T2 ? (s_x1[xb + 3].x + is * 8) : nullptr;

                __half2 a0 = __floats2half2_rn(0.0f, 0.0f), a1 = __floats2half2_rn(0.0f, 0.0f);
                __half2 a2 = __floats2half2_rn(0.0f, 0.0f), a3 = __floats2half2_rn(0.0f, 0.0f);
                __half2 b0h = __floats2half2_rn(0.0f, 0.0f), b1h = __floats2half2_rn(0.0f, 0.0f);
                __half2 b2h = __floats2half2_rn(0.0f, 0.0f), b3h = __floats2half2_rn(0.0f, 0.0f);

                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint32_t q1 = (ql0w[c] & 0x0F0F0F0Fu) | ((qhw[c] & 0x03030303u) << 4);
                    const uint32_t q2 = (ql1w[c] & 0x0F0F0F0Fu) | ((qhw[c] & 0x0C0C0C0Cu) << 2);
                    const uint32_t q3 = ((ql0w[c] >> 4) & 0x0F0F0F0Fu) | (qhw[c] & 0x30303030u);
                    const uint32_t q4 = ((ql1w[c] >> 4) & 0x0F0F0F0Fu) | ((qhw[c] >> 2) & 0x30303030u);

                    const __half2 w1a = mw_hadd2(mw_u8x2_to_h2_lo(q1), neg1056), w1b = mw_hadd2(mw_u8x2_to_h2_hi(q1), neg1056);
                    const __half2 w2a = mw_hadd2(mw_u8x2_to_h2_lo(q2), neg1056), w2b = mw_hadd2(mw_u8x2_to_h2_hi(q2), neg1056);
                    const __half2 w3a = mw_hadd2(mw_u8x2_to_h2_lo(q3), neg1056), w3b = mw_hadd2(mw_u8x2_to_h2_hi(q3), neg1056);
                    const __half2 w4a = mw_hadd2(mw_u8x2_to_h2_lo(q4), neg1056), w4b = mw_hadd2(mw_u8x2_to_h2_hi(q4), neg1056);

                    a0 = mw_hfma2(w1a, x0_0[c * 2 + 0], a0); a0 = mw_hfma2(w1b, x0_0[c * 2 + 1], a0);
                    a1 = mw_hfma2(w2a, x0_1[c * 2 + 0], a1); a1 = mw_hfma2(w2b, x0_1[c * 2 + 1], a1);
                    a2 = mw_hfma2(w3a, x0_2[c * 2 + 0], a2); a2 = mw_hfma2(w3b, x0_2[c * 2 + 1], a2);
                    a3 = mw_hfma2(w4a, x0_3[c * 2 + 0], a3); a3 = mw_hfma2(w4b, x0_3[c * 2 + 1], a3);

                    if constexpr (T2) {
                        b0h = mw_hfma2(w1a, x1_0[c * 2 + 0], b0h); b0h = mw_hfma2(w1b, x1_0[c * 2 + 1], b0h);
                        b1h = mw_hfma2(w2a, x1_1[c * 2 + 0], b1h); b1h = mw_hfma2(w2b, x1_1[c * 2 + 1], b1h);
                        b2h = mw_hfma2(w3a, x1_2[c * 2 + 0], b2h); b2h = mw_hfma2(w3b, x1_2[c * 2 + 1], b2h);
                        b3h = mw_hfma2(w4a, x1_3[c * 2 + 0], b3h); b3h = mw_hfma2(w4b, x1_3[c * 2 + 1], b3h);
                    }
                }
                const float2 f0 = __half22float2(a0), f1 = __half22float2(a1), f2 = __half22float2(a2), f3 = __half22float2(a3);
                acc0 += (sc0 * s_x0[xb + 0].d) * (f0.x + f0.y) + (sc2 * s_x0[xb + 1].d) * (f1.x + f1.y)
                      + (sc4 * s_x0[xb + 2].d) * (f2.x + f2.y) + (sc6 * s_x0[xb + 3].d) * (f3.x + f3.y);
                if constexpr (T2) {
                    const float2 g0 = __half22float2(b0h), g1 = __half22float2(b1h), g2 = __half22float2(b2h), g3 = __half22float2(b3h);
                    acc1 += (sc0 * s_x1[xb + 0].d) * (g0.x + g0.y) + (sc2 * s_x1[xb + 1].d) * (g1.x + g1.y)
                          + (sc4 * s_x1[xb + 2].d) * (g2.x + g2.y) + (sc6 * s_x1[xb + 3].d) * (g3.x + g3.y);
                }
            }
        }
        if (b0 + tile_nb < nb) __syncthreads();
    }

    if (row < M) {
        const float v0 = mw_warp_sum(acc0);
        const float v1 = T2 ? mw_warp_sum(acc1) : 0.0f;
        if (lane == 0) {
            Y[static_cast<size_t>(t) * M + row] = v0;
            if constexpr (T2) Y[M + row] = v1;
        }
    }
}

// ============================================================================
// 5. MoE Token Grouping & Flat Tile Worklist (for Prefill Grouped MoE!)
// ============================================================================
__global__ void mw_group_moe_tokens_kernel(
    const int32_t * __restrict__ top_ids,
    int32_t *       __restrict__ exp_offsets,
    int32_t *       __restrict__ sorted_pairs,
    int total_pairs, int n_expert, int k_used) {

    extern __shared__ int32_t s_cnt[];
    const int tid = threadIdx.x;
    for (int e = tid; e < n_expert; e += blockDim.x) s_cnt[e] = 0;
    __syncthreads();
    for (int i = tid; i < total_pairs; i += blockDim.x) {
        int eid = top_ids[i];
        if (eid >= 0 && eid < n_expert) atomicAdd(&s_cnt[eid], 1);
    }
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
        for (int e = 0; e < n_expert; ++e) {
            int c = s_cnt[e];
            exp_offsets[e] = sum;
            s_cnt[e] = sum;
            sum += c;
        }
        exp_offsets[n_expert] = sum;
    }
    __syncthreads();
    for (int i = tid; i < total_pairs; i += blockDim.x) {
        int eid = top_ids[i];
        if (eid >= 0 && eid < n_expert) {
            int dst = atomicAdd(&s_cnt[eid], 1);
            sorted_pairs[dst] = ((i / k_used) << 4) | (i % k_used);
        }
    }
}

__global__ void mw_build_tile_worklist_kernel(
    const int32_t * __restrict__ exp_offsets,
    MaxwTileDesc *  __restrict__ tiles,
    int32_t *       __restrict__ n_tiles_out,
    int n_expert, int BT) {
    if (threadIdx.x != 0) return;
    int nt = 0;
    for (int e = 0; e < n_expert; ++e) {
        const int s  = exp_offsets[e];
        const int en = exp_offsets[e + 1];
        for (int p = s; p < en; p += BT) {
            const int cnt = (p + BT < en) ? BT : (en - p);
            tiles[nt++] = {e, p, cnt};
        }
    }
    *n_tiles_out = nt;
}

__global__ void mw_reduce_slots_add_kernel(
    const float * __restrict__ slots,
    float *       __restrict__ out,
    int K, int n_slots, bool add_to_existing) {
    const int K4 = K >> 2;
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
    const int t  = blockIdx.y;
    if (i >= K4) return;
    const float4 * base = reinterpret_cast<const float4 *>(slots + static_cast<size_t>(t) * n_slots * K) + i;
    float4 sum = add_to_existing
        ? reinterpret_cast<const float4 *>(out + static_cast<size_t>(t) * K)[i]
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < n_slots; ++s) {
        const float4 v = base[static_cast<size_t>(s) * K4];
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    reinterpret_cast<float4 *>(out + static_cast<size_t>(t) * K)[i] = sum;
}

// ============================================================================
// 6. ULTRA-FAST DECODE MoE (T < 16): Routed (k_used) + Shared (1) FUSED!
// ============================================================================
__global__ __launch_bounds__(256) void mw_moe_gate_up_q3_k_soa_h2_dec_kernel(
    const int4 *        __restrict__ g_qs,  const int4 * __restrict__ g_hm,
    const uint32_t *    __restrict__ g_sc,  const uint16_t * __restrict__ g_d,
    const int4 *        __restrict__ u_qs,  const int4 * __restrict__ u_hm,
    const uint32_t *    __restrict__ u_sc,  const uint16_t * __restrict__ u_d,
    const int4 *        __restrict__ sg_qs, const int4 * __restrict__ sg_hm,
    const uint32_t *    __restrict__ sg_sc, const uint16_t * __restrict__ sg_d,
    const int4 *        __restrict__ su_qs, const int4 * __restrict__ su_hm,
    const uint32_t *    __restrict__ su_sc, const uint16_t * __restrict__ su_d,
    const ActF16Block * __restrict__ x_f16,
    const int32_t *     __restrict__ top_ids,
    ActF16Block *       __restrict__ act_r_f16,
    ActF16Block *       __restrict__ act_s_f16,
    int K, int ff, int ff_sh, int k_used, float clamp_limit) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<ActF16Block *>(smem_raw);
    __shared__ float s_act[32];

    const int routed_tiles_per_slot = ff >> 5;
    const int total_routed_tiles    = k_used * routed_tiles_per_slot;
    const int flat_tile = blockIdx.x;
    const int t         = blockIdx.y;
    const bool is_sh    = (flat_tile >= total_routed_tiles);
    const int slot      = is_sh ? k_used : (flat_tile / routed_tiles_per_slot);
    const int tile_x    = is_sh ? (flat_tile - total_routed_tiles) : (flat_tile % routed_tiles_per_slot);

    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;

    const int n_i4 = (K_blocks * static_cast<int>(sizeof(ActF16Block))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(x_f16 + static_cast<size_t>(t) * K_blocks);
    int4 *       sx_ou = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_ou[i] = sx_in[i];
    __syncthreads();

    const int lane     = threadIdx.x;
    const int sub_r    = lane >> 3;
    const int sub_l    = lane & 7;
    const int r_in_cta = threadIdx.y * 4 + sub_r;
    const int row      = tile_x * 32 + r_in_cta;

    const int b_init  = sub_l >> 2;
    const int sub_idx = sub_l & 3;
    const int n128    = sub_idx >> 1;
    const int sub     = sub_idx & 1;

    size_t rblk = 0;
    const int4 *     gq, * gh, * uq, * uh;
    const uint32_t * gs, * us;
    const uint16_t * gd, * ud;
    if (!is_sh) {
        const int eid = top_ids[static_cast<size_t>(t) * k_used + slot];
        rblk = (static_cast<size_t>(eid) * ff + row) * nb;
        gq = g_qs + rblk * 4; gh = g_hm + rblk * 2; gs = g_sc + rblk * 3; gd = g_d + rblk;
        uq = u_qs + rblk * 4; uh = u_hm + rblk * 2; us = u_sc + rblk * 3; ud = u_d + rblk;
    } else {
        rblk = static_cast<size_t>(row) * nb;
        gq = sg_qs + rblk * 4; gh = sg_hm + rblk * 2; gs = sg_sc + rblk * 3; gd = sg_d + rblk;
        uq = su_qs + rblk * 4; uh = su_hm + rblk * 2; us = su_sc + rblk * 3; ud = su_d + rblk;
    }

    const __half2 neg1028 = __floats2half2_rn(-1028.0f, -1028.0f);
    float ag = 0.0f, au = 0.0f;

    for (int bi = b_init; bi < nb; bi += 2) {
        const int4 g_qv = __ldg(&gq[bi * 4 + sub_idx]);
        const int4 g_hv = __ldg(&gh[bi * 2 + sub]);
        const float g_da = mw_half_bits_to_f32(__ldg(&gd[bi]));
        uint32_t g_aux[4];
        mw_unpack_q3k_scales(gs, bi, g_aux);

        const int4 u_qv = __ldg(&uq[bi * 4 + sub_idx]);
        const int4 u_hv = __ldg(&uh[bi * 2 + sub]);
        const float u_da = mw_half_bits_to_f32(__ldg(&ud[bi]));
        uint32_t u_aux[4];
        mw_unpack_q3k_scales(us, bi, u_aux);

        const uint32_t gqw[4] = { static_cast<uint32_t>(g_qv.x), static_cast<uint32_t>(g_qv.y), static_cast<uint32_t>(g_qv.z), static_cast<uint32_t>(g_qv.w) };
        const uint32_t ghw[4] = { static_cast<uint32_t>(g_hv.x), static_cast<uint32_t>(g_hv.y), static_cast<uint32_t>(g_hv.z), static_cast<uint32_t>(g_hv.w) };
        const uint32_t uqw[4] = { static_cast<uint32_t>(u_qv.x), static_cast<uint32_t>(u_qv.y), static_cast<uint32_t>(u_qv.z), static_cast<uint32_t>(u_qv.w) };
        const uint32_t uhw[4] = { static_cast<uint32_t>(u_hv.x), static_cast<uint32_t>(u_hv.y), static_cast<uint32_t>(u_hv.z), static_cast<uint32_t>(u_hv.w) };

        float bg = 0.0f, bu = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int grp  = n128 * 8 + j * 2 + sub;
            const int gsc  = static_cast<int>((g_aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu) - 32;
            const int usc  = static_cast<int>((u_aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu) - 32;
            const int xb   = bi * 8 + n128 * 4 + j;
            const int sh_q = 2 * j;
            const int sh_h = n128 * 4 + j;
            const int4 * x4 = reinterpret_cast<const int4 *>(s_x[xb].x + sub * 8);
            const int4 xa = x4[0], xb4 = x4[1];
            const __half2 xh[8] = {
                *reinterpret_cast<const __half2 *>(&xa.x),  *reinterpret_cast<const __half2 *>(&xa.y),
                *reinterpret_cast<const __half2 *>(&xa.z),  *reinterpret_cast<const __half2 *>(&xa.w),
                *reinterpret_cast<const __half2 *>(&xb4.x), *reinterpret_cast<const __half2 *>(&xb4.y),
                *reinterpret_cast<const __half2 *>(&xb4.z), *reinterpret_cast<const __half2 *>(&xb4.w)
            };
            const float xd = s_x[xb].d;

            __half2 hg = __floats2half2_rn(0.0f, 0.0f);
            __half2 hu = __floats2half2_rn(0.0f, 0.0f);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint32_t gu = ((gqw[c] >> sh_q) & 0x03030303u) | (((ghw[c] >> sh_h) & 0x01010101u) << 2);
                const uint32_t uu = ((uqw[c] >> sh_q) & 0x03030303u) | (((uhw[c] >> sh_h) & 0x01010101u) << 2);
                hg = mw_hfma2(mw_hadd2(mw_u8x2_to_h2_lo(gu), neg1028), xh[c * 2 + 0], hg);
                hg = mw_hfma2(mw_hadd2(mw_u8x2_to_h2_hi(gu), neg1028), xh[c * 2 + 1], hg);
                hu = mw_hfma2(mw_hadd2(mw_u8x2_to_h2_lo(uu), neg1028), xh[c * 2 + 0], hu);
                hu = mw_hfma2(mw_hadd2(mw_u8x2_to_h2_hi(uu), neg1028), xh[c * 2 + 1], hu);
            }
            const float2 fg = __half22float2(hg), fu = __half22float2(hu);
            bg += (static_cast<float>(gsc) * xd) * (fg.x + fg.y);
            bu += (static_cast<float>(usc) * xd) * (fu.x + fu.y);
        }
        ag += g_da * bg;
        au += u_da * bu;
    }
    ag += __shfl_down_sync(0xffffffffu, ag, 4, 8); au += __shfl_down_sync(0xffffffffu, au, 4, 8);
    ag += __shfl_down_sync(0xffffffffu, ag, 2, 8); au += __shfl_down_sync(0xffffffffu, au, 2, 8);
    ag += __shfl_down_sync(0xffffffffu, ag, 1, 8); au += __shfl_down_sync(0xffffffffu, au, 1, 8);

    if (sub_l == 0) {
        if (clamp_limit > 0.0f) {
            ag = fminf(fmaxf(ag, -clamp_limit), clamp_limit);
            au = fminf(fmaxf(au, -clamp_limit), clamp_limit);
        }
        s_act[r_in_cta] = (ag / (1.0f + __expf(-ag))) * au;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        const float v = s_act[lane];
        const float amax = __shfl_sync(0xffffffffu, mw_warp_max(fabsf(v)), 0);
        const float inv  = (amax > 0.0f) ? (1.0f / amax) : 0.0f;
        const float vn   = v * inv;
        const float vn_hi = __shfl_down_sync(0xffffffffu, vn, 1);
        float hs = v;
        hs += __shfl_xor_sync(0xffffffffu, hs, 1);
        hs += __shfl_xor_sync(0xffffffffu, hs, 2);
        hs += __shfl_xor_sync(0xffffffffu, hs, 4);
        hs += __shfl_xor_sync(0xffffffffu, hs, 8);
        const float s0 = __shfl_sync(0xffffffffu, hs, 0);
        const float s1 = __shfl_sync(0xffffffffu, hs, 16);

        ActF16Block * dst = is_sh
            ? (act_s_f16 + static_cast<size_t>(t) * (ff_sh >> 5) + tile_x)
            : (act_r_f16 + (static_cast<size_t>(t) * k_used + slot) * (ff >> 5) + tile_x);
        if ((lane & 1) == 0) {
            dst->x[lane >> 1] = __halves2half2(__float2half_rn(vn), __float2half_rn(vn_hi));
        }
        if (lane == 0) {
            dst->d     = amax;
            dst->s0    = s0;
            dst->s1    = s1;
            dst->s_all = s0 + s1;
        }
    }
}

__global__ void mw_moe_down_q4_k_h2_split_dec_kernel(
    const BlockQ4_K_M * __restrict__ dn_exps,
    const BlockQ4_K_M * __restrict__ dn_shexp,
    const ActF16Block * __restrict__ act_r_f16,
    const ActF16Block * __restrict__ act_s_f16,
    const int32_t *     __restrict__ top_ids,
    const float *       __restrict__ top_w,
    float *             __restrict__ down_slots,
    int K, int ff, int ff_sh, int k_used) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_a = reinterpret_cast<ActF16Block *>(smem_raw);
    const int slot   = blockIdx.y;
    const int t      = blockIdx.z;
    const bool is_sh = (slot == k_used);
    const int cur_ff = is_sh ? ff_sh : ff;
    const int cur_fb = cur_ff >> 5;
    const int nb     = cur_ff >> 8;

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const ActF16Block * a_src = is_sh
        ? (act_s_f16 + static_cast<size_t>(t) * cur_fb)
        : (act_r_f16 + (static_cast<size_t>(t) * k_used + slot) * cur_fb);
    const int n_i4 = (cur_fb * static_cast<int>(sizeof(ActF16Block))) / 16;
    for (int i = tid; i < n_i4; i += nthr) {
        reinterpret_cast<int4 *>(s_a)[i] = reinterpret_cast<const int4 *>(a_src)[i];
    }
    __syncthreads();

    const int lane  = threadIdx.x;
    const int sub_r = lane >> 4;
    const int sub_l = lane & 15;
    const int b_init = sub_l >> 3;
    const int sub    = sub_l & 7;
    const int j      = sub >> 1;
    const int is     = sub & 1;
    const __half2 neg1024 = __floats2half2_rn(-1024.0f, -1024.0f);

    const size_t slot_idx = static_cast<size_t>(t) * k_used + slot;
    const int eid   = is_sh ? 0 : top_ids[slot_idx];
    const float ew  = is_sh ? 1.0f : top_w[slot_idx];
    const int n_slots = k_used + 1;
    const int row_base = (blockIdx.x * blockDim.y + threadIdx.y) * 4 + sub_r * 2;

    #pragma unroll
    for (int r_it = 0; r_it < 2; ++r_it) {
        const int row = row_base + r_it;
        if (row >= K) return;
        const BlockQ4_K_M * w_row = is_sh
            ? (dn_shexp + static_cast<size_t>(row) * nb)
            : (dn_exps  + (static_cast<size_t>(eid) * K + row) * nb);

        float acc = 0.0f;
        for (int bi = b_init; bi < nb; bi += 2) {
            const BlockQ4_K_M & b = w_row[bi];
            const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
            const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
            const float d    = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
            const float dmin = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
            float sc1, m1, sc2, m2;
            mw_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
            mw_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
            const float d1 = d * sc1, mn1 = dmin * m1;
            const float d2 = d * sc2, mn2 = dmin * m2;

            const uint32_t qws[4] = {
                static_cast<uint32_t>(ql_v.x), static_cast<uint32_t>(ql_v.y),
                static_cast<uint32_t>(ql_v.z), static_cast<uint32_t>(ql_v.w)
            };
            const int xb = bi * 8 + 2 * j;
            const int4 * x1_4 = reinterpret_cast<const int4 *>(s_a[xb + 0].x + is * 8);
            const int4 * x2_4 = reinterpret_cast<const int4 *>(s_a[xb + 1].x + is * 8);
            const int4 x1a = x1_4[0], x1b = x1_4[1];
            const int4 x2a = x2_4[0], x2b = x2_4[1];
            const __half2 x1[8] = {
                *reinterpret_cast<const __half2 *>(&x1a.x), *reinterpret_cast<const __half2 *>(&x1a.y),
                *reinterpret_cast<const __half2 *>(&x1a.z), *reinterpret_cast<const __half2 *>(&x1a.w),
                *reinterpret_cast<const __half2 *>(&x1b.x), *reinterpret_cast<const __half2 *>(&x1b.y),
                *reinterpret_cast<const __half2 *>(&x1b.z), *reinterpret_cast<const __half2 *>(&x1b.w)
            };
            const __half2 x2[8] = {
                *reinterpret_cast<const __half2 *>(&x2a.x), *reinterpret_cast<const __half2 *>(&x2a.y),
                *reinterpret_cast<const __half2 *>(&x2a.z), *reinterpret_cast<const __half2 *>(&x2a.w),
                *reinterpret_cast<const __half2 *>(&x2b.x), *reinterpret_cast<const __half2 *>(&x2b.y),
                *reinterpret_cast<const __half2 *>(&x2b.z), *reinterpret_cast<const __half2 *>(&x2b.w)
            };
            __half2 h1 = __floats2half2_rn(0.0f, 0.0f), h2 = __floats2half2_rn(0.0f, 0.0f);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint32_t u_lo = qws[c] & 0x0F0F0F0Fu;
                const uint32_t u_hi = (qws[c] >> 4) & 0x0F0F0F0Fu;
                h1 = mw_hfma2(mw_hadd2(mw_u8x2_to_h2_lo(u_lo), neg1024), x1[c * 2 + 0], h1);
                h1 = mw_hfma2(mw_hadd2(mw_u8x2_to_h2_hi(u_lo), neg1024), x1[c * 2 + 1], h1);
                h2 = mw_hfma2(mw_hadd2(mw_u8x2_to_h2_lo(u_hi), neg1024), x2[c * 2 + 0], h2);
                h2 = mw_hfma2(mw_hadd2(mw_u8x2_to_h2_hi(u_hi), neg1024), x2[c * 2 + 1], h2);
            }
            const float2 f1 = __half22float2(h1), f2 = __half22float2(h2);
            const float s1 = is ? s_a[xb + 0].s1 : s_a[xb + 0].s0;
            const float s2 = is ? s_a[xb + 1].s1 : s_a[xb + 1].s0;
            acc += (d1 * s_a[xb + 0].d) * (f1.x + f1.y) - mn1 * s1
                 + (d2 * s_a[xb + 1].d) * (f2.x + f2.y) - mn2 * s2;
        }

        acc += __shfl_down_sync(0xffffffffu, acc, 8, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 4, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 2, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 1, 16);
        if (sub_l == 0) {
            down_slots[(static_cast<size_t>(t) * n_slots + slot) * K + row] = ew * acc;
        }
    }
}

__global__ void mw_moe_gate_up_q3_k_soa_dp4a_dec_kernel(
    const int4 *          __restrict__ g_qs,  const int4 * __restrict__ g_hm,
    const uint32_t *      __restrict__ g_sc,  const uint16_t * __restrict__ g_d,
    const int4 *          __restrict__ u_qs,  const int4 * __restrict__ u_hm,
    const uint32_t *      __restrict__ u_sc,  const uint16_t * __restrict__ u_d,
    const int4 *          __restrict__ sg_qs, const int4 * __restrict__ sg_hm,
    const uint32_t *      __restrict__ sg_sc, const uint16_t * __restrict__ sg_d,
    const int4 *          __restrict__ su_qs, const int4 * __restrict__ su_hm,
    const uint32_t *      __restrict__ su_sc, const uint16_t * __restrict__ su_d,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ top_ids,
    float *               __restrict__ act_r,
    float *               __restrict__ act_s,
    int K, int ff, int ff_sh, int k_used, float clamp_limit) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<BlockQ8_1_Act *>(smem_raw);
    const int slot     = blockIdx.y;
    const int t        = blockIdx.z;
    const bool is_sh   = (slot == k_used);
    const int lim_rows = is_sh ? ff_sh : ff;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;

    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(t) * K_blocks);
    for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_x)[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= lim_rows) return;

    float ag, au;
    if (!is_sh) {
        const int eid = top_ids[static_cast<size_t>(t) * k_used + slot];
        const size_t rblk = (static_cast<size_t>(eid) * ff + row) * nb;
        ag = dot_q3_k_soa_row_dp4a(g_qs + rblk * 4, g_hm + rblk * 2, g_sc + rblk * 3, g_d + rblk, s_x, nb, threadIdx.x);
        au = dot_q3_k_soa_row_dp4a(u_qs + rblk * 4, u_hm + rblk * 2, u_sc + rblk * 3, u_d + rblk, s_x, nb, threadIdx.x);
    } else {
        const size_t rblk = static_cast<size_t>(row) * nb;
        ag = dot_q3_k_soa_row_dp4a(sg_qs + rblk * 4, sg_hm + rblk * 2, sg_sc + rblk * 3, sg_d + rblk, s_x, nb, threadIdx.x);
        au = dot_q3_k_soa_row_dp4a(su_qs + rblk * 4, su_hm + rblk * 2, su_sc + rblk * 3, su_d + rblk, s_x, nb, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        if (clamp_limit > 0.0f) {
            ag = fminf(fmaxf(ag, -clamp_limit), clamp_limit);
            au = fminf(fmaxf(au, -clamp_limit), clamp_limit);
        }
        const float y = (ag / (1.0f + expf(-ag))) * au;
        if (!is_sh) act_r[(static_cast<size_t>(t) * k_used + slot) * ff + row] = y;
        else        act_s[static_cast<size_t>(t) * ff_sh + row] = y;
    }
}

__global__ void mw_moe_down_q4_k_dp4a_split_dec_kernel(
    const BlockQ4_K_M *   __restrict__ dn_exps,
    const BlockQ4_K_M *   __restrict__ dn_shexp,
    const BlockQ8_1_Act * __restrict__ aq8,
    const BlockQ8_1_Act * __restrict__ sq8,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ down_slots,
    int K, int ff, int ff_sh, int k_used) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_a = reinterpret_cast<BlockQ8_1_Act *>(smem_raw);
    const int slot   = blockIdx.y;
    const int t      = blockIdx.z;
    const bool is_sh = (slot == k_used);
    const int cur_ff = is_sh ? ff_sh : ff;
    const int cur_fb = cur_ff >> 5;
    const int nb     = cur_ff >> 8;

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const BlockQ8_1_Act * src = is_sh
        ? (sq8 + static_cast<size_t>(t) * cur_fb)
        : (aq8 + (static_cast<size_t>(t) * k_used + slot) * cur_fb);
    const int n_i4 = (cur_fb * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    for (int i = tid; i < n_i4; i += nthr) {
        reinterpret_cast<int4 *>(s_a)[i] = reinterpret_cast<const int4 *>(src)[i];
    }
    __syncthreads();

    const int lane  = threadIdx.x;
    const int sub_r = lane >> 4;
    const int sub_l = lane & 15;
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 2 + sub_r;
    if (row >= K) return;

    const int b_init = sub_l >> 3;
    const int sub    = sub_l & 7;
    const int j      = sub >> 1;
    const int is     = sub & 1;

    const BlockQ4_K_M * w_row = nullptr;
    float ew = 1.0f;
    if (!is_sh) {
        const size_t slot_idx = static_cast<size_t>(t) * k_used + slot;
        const int eid = top_ids[slot_idx];
        ew    = top_w[slot_idx];
        w_row = dn_exps + (static_cast<size_t>(eid) * K + row) * nb;
    } else {
        w_row = dn_shexp + static_cast<size_t>(row) * nb;
    }

    float acc = 0.0f;
    for (int bi = b_init; bi < nb; bi += 2) {
        const BlockQ4_K_M & b = w_row[bi];
        const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
        const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
        const float d    = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
        const float dmin = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
        float sc1, m1, sc2, m2;
        mw_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
        mw_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
        const float d1 = d * sc1, mn1 = dmin * m1;
        const float d2 = d * sc2, mn2 = dmin * m2;

        const int q1x = static_cast<int>(static_cast<uint32_t>(ql_v.x) & 0x0F0F0F0Fu);
        const int q2x = static_cast<int>((static_cast<uint32_t>(ql_v.x) >> 4) & 0x0F0F0F0Fu);
        const int q1y = static_cast<int>(static_cast<uint32_t>(ql_v.y) & 0x0F0F0F0Fu);
        const int q2y = static_cast<int>((static_cast<uint32_t>(ql_v.y) >> 4) & 0x0F0F0F0Fu);
        const int q1z = static_cast<int>(static_cast<uint32_t>(ql_v.z) & 0x0F0F0F0Fu);
        const int q2z = static_cast<int>((static_cast<uint32_t>(ql_v.z) >> 4) & 0x0F0F0F0Fu);
        const int q1w = static_cast<int>(static_cast<uint32_t>(ql_v.w) & 0x0F0F0F0Fu);
        const int q2w = static_cast<int>((static_cast<uint32_t>(ql_v.w) >> 4) & 0x0F0F0F0Fu);

        const int xb = bi * 8 + 2 * j;
        const int4 x1 = reinterpret_cast<const int4 *>(s_a[xb + 0].qs)[is];
        const int4 x2 = reinterpret_cast<const int4 *>(s_a[xb + 1].qs)[is];
        const int sum1 = dp4a_s32(q1x, x1.x, dp4a_s32(q1y, x1.y, dp4a_s32(q1z, x1.z, dp4a_s32(q1w, x1.w, 0))));
        const int sum2 = dp4a_s32(q2x, x2.x, dp4a_s32(q2y, x2.y, dp4a_s32(q2z, x2.z, dp4a_s32(q2w, x2.w, 0))));
        const int sx1 = is ? s_a[xb + 0].s1 : s_a[xb + 0].s0;
        const int sx2 = is ? s_a[xb + 1].s1 : s_a[xb + 1].s0;
        acc += s_a[xb + 0].d * (d1 * static_cast<float>(sum1) - mn1 * static_cast<float>(sx1))
             + s_a[xb + 1].d * (d2 * static_cast<float>(sum2) - mn2 * static_cast<float>(sx2));
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 8, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 4, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 2, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 1, 16);
    if (sub_l == 0) {
        const int n_slots = k_used + 1;
        down_slots[(static_cast<size_t>(t) * n_slots + slot) * K + row] = ew * acc;
    }
}

// ============================================================================
// 7. ULTRA-FAST GROUPED MoE PREFILL (T >= 16) — FLAT TILE WORKLIST (BT = 16)!
// ============================================================================
template <int BT, int KT_BLK>
__global__ __launch_bounds__(128, 6) void mw_moe_gate_up_grouped_q3_k_soa_h2_kernel(
    const int4 *         __restrict__ g_qs, const int4 * __restrict__ g_hm,
    const uint32_t *     __restrict__ g_sc, const uint16_t * __restrict__ g_d,
    const int4 *         __restrict__ u_qs, const int4 * __restrict__ u_hm,
    const uint32_t *     __restrict__ u_sc, const uint16_t * __restrict__ u_d,
    const ActF16Block *  __restrict__ x_f16,
    const MaxwTileDesc * __restrict__ tiles,
    const int32_t *      __restrict__ n_tiles_ptr,
    const int32_t *      __restrict__ sorted_pairs,
    float *              __restrict__ act,
    int ff, int K, int k_used, float clamp_limit) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const MaxwTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    constexpr int KT32 = KT_BLK * 8;
    auto * s_xh2 = reinterpret_cast<__half2 *>(smem_raw);
    auto * s_xd  = reinterpret_cast<float *>(s_xh2 + KT32 * BT * 16);

    const int tid     = threadIdx.x;
    const int nthr    = blockDim.x;
    const int row     = blockIdx.x * 32 + (tid >> 2);
    const int sub_idx = tid & 3;
    const int n128    = sub_idx >> 1;
    const int sub     = sub_idx & 1;

    if (tid < n_tile) s_pairs[tid] = sorted_pairs[start + tid];
    __syncthreads();

    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const size_t rblk  = (static_cast<size_t>(eid) * ff + row) * nb;
    const __half2 neg1028 = __floats2half2_rn(-1028.0f, -1028.0f);

    float ag[BT], au[BT];
    #pragma unroll
    for (int j = 0; j < BT; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

    for (int b0 = 0; b0 < nb; b0 += KT_BLK) {
        const int bc   = (b0 + KT_BLK < nb) ? KT_BLK : (nb - b0);
        const int bc32 = bc * 8;
        const int tot_i4 = bc32 * BT * 4;
        for (int idx = tid; idx < tot_i4; idx += nthr) {
            const int c   = idx & 3;
            const int rem = idx >> 2;
            const int tj  = rem % BT;
            const int bk  = rem / BT;
            if (tj < n_tile) {
                const int tok = s_pairs[tj] >> 4;
                reinterpret_cast<int4 *>(s_xh2)[idx] =
                    reinterpret_cast<const int4 *>(x_f16[static_cast<size_t>(tok) * K_blocks + b0 * 8 + bk].x)[c];
            } else {
                reinterpret_cast<int4 *>(s_xh2)[idx] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = bc32 * BT;
        for (int idx = tid; idx < tot_d; idx += nthr) {
            const int tj = idx % BT;
            const int bk = idx / BT;
            s_xd[idx] = (tj < n_tile)
                ? x_f16[static_cast<size_t>(s_pairs[tj] >> 4) * K_blocks + b0 * 8 + bk].d : 0.0f;
        }
        __syncthreads();

        if (row < ff) {
            for (int db = 0; db < bc; ++db) {
                const int64_t bi = static_cast<int64_t>(rblk) + b0 + db;
                // Keep gate & up weight registers in separate scopes -> ~56 regs & 6-8 CTAs/SM on P100!
                {
                    const int4 g_qv = __ldg(&g_qs[bi * 4 + sub_idx]);
                    const int4 g_hv = __ldg(&g_hm[bi * 2 + sub]);
                    const float g_da = mw_half_bits_to_f32(__ldg(&g_d[bi]));
                    uint32_t g_aux[4];
                    mw_unpack_q3k_scales(g_sc, bi, g_aux);
                    const uint32_t gqw[4] = { static_cast<uint32_t>(g_qv.x), static_cast<uint32_t>(g_qv.y), static_cast<uint32_t>(g_qv.z), static_cast<uint32_t>(g_qv.w) };
                    const uint32_t ghw[4] = { static_cast<uint32_t>(g_hv.x), static_cast<uint32_t>(g_hv.y), static_cast<uint32_t>(g_hv.z), static_cast<uint32_t>(g_hv.w) };
                    #pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        const int grp   = n128 * 8 + g * 2 + sub;
                        const float gdl = g_da * static_cast<float>(static_cast<int>((g_aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu) - 32);
                        const int bk    = db * 8 + n128 * 4 + g;
                        const int sh_q  = 2 * g;
                        const int sh_h  = n128 * 4 + g;
                        const uint32_t gu0 = ((gqw[0] >> sh_q) & 0x03030303u) | (((ghw[0] >> sh_h) & 0x01010101u) << 2);
                        const uint32_t gu1 = ((gqw[1] >> sh_q) & 0x03030303u) | (((ghw[1] >> sh_h) & 0x01010101u) << 2);
                        const uint32_t gu2 = ((gqw[2] >> sh_q) & 0x03030303u) | (((ghw[2] >> sh_h) & 0x01010101u) << 2);
                        const uint32_t gu3 = ((gqw[3] >> sh_q) & 0x03030303u) | (((ghw[3] >> sh_h) & 0x01010101u) << 2);
                        const __half2 gw0 = mw_hadd2(mw_u8x2_to_h2_lo(gu0), neg1028), gw1 = mw_hadd2(mw_u8x2_to_h2_hi(gu0), neg1028);
                        const __half2 gw2 = mw_hadd2(mw_u8x2_to_h2_lo(gu1), neg1028), gw3 = mw_hadd2(mw_u8x2_to_h2_hi(gu1), neg1028);
                        const __half2 gw4 = mw_hadd2(mw_u8x2_to_h2_lo(gu2), neg1028), gw5 = mw_hadd2(mw_u8x2_to_h2_hi(gu2), neg1028);
                        const __half2 gw6 = mw_hadd2(mw_u8x2_to_h2_lo(gu3), neg1028), gw7 = mw_hadd2(mw_u8x2_to_h2_hi(gu3), neg1028);
                        const int4 *  xk4 = reinterpret_cast<const int4 *>(s_xh2 + bk * BT * 16 + sub * 8);
                        const float * dk  = s_xd + bk * BT;
                        #pragma unroll
                        for (int tj = 0; tj < BT; ++tj) {
                            const int4 xa = xk4[tj * 4 + 0];
                            const int4 xb = xk4[tj * 4 + 1];
                            __half2 rg = mw_hmul2(gw0, *reinterpret_cast<const __half2 *>(&xa.x));
                            rg = mw_hfma2(gw1, *reinterpret_cast<const __half2 *>(&xa.y), rg);
                            rg = mw_hfma2(gw2, *reinterpret_cast<const __half2 *>(&xa.z), rg);
                            rg = mw_hfma2(gw3, *reinterpret_cast<const __half2 *>(&xa.w), rg);
                            rg = mw_hfma2(gw4, *reinterpret_cast<const __half2 *>(&xb.x), rg);
                            rg = mw_hfma2(gw5, *reinterpret_cast<const __half2 *>(&xb.y), rg);
                            rg = mw_hfma2(gw6, *reinterpret_cast<const __half2 *>(&xb.z), rg);
                            rg = mw_hfma2(gw7, *reinterpret_cast<const __half2 *>(&xb.w), rg);
                            const float2 fg = __half22float2(rg);
                            ag[tj] += (gdl * dk[tj]) * (fg.x + fg.y);
                        }
                    }
                }
                {
                    const int4 u_qv = __ldg(&u_qs[bi * 4 + sub_idx]);
                    const int4 u_hv = __ldg(&u_hm[bi * 2 + sub]);
                    const float u_da = mw_half_bits_to_f32(__ldg(&u_d[bi]));
                    uint32_t u_aux[4];
                    mw_unpack_q3k_scales(u_sc, bi, u_aux);
                    const uint32_t uqw[4] = { static_cast<uint32_t>(u_qv.x), static_cast<uint32_t>(u_qv.y), static_cast<uint32_t>(u_qv.z), static_cast<uint32_t>(u_qv.w) };
                    const uint32_t uhw[4] = { static_cast<uint32_t>(u_hv.x), static_cast<uint32_t>(u_hv.y), static_cast<uint32_t>(u_hv.z), static_cast<uint32_t>(u_hv.w) };
                    #pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        const int grp   = n128 * 8 + g * 2 + sub;
                        const float udl = u_da * static_cast<float>(static_cast<int>((u_aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu) - 32);
                        const int bk    = db * 8 + n128 * 4 + g;
                        const int sh_q  = 2 * g;
                        const int sh_h  = n128 * 4 + g;
                        const uint32_t uu0 = ((uqw[0] >> sh_q) & 0x03030303u) | (((uhw[0] >> sh_h) & 0x01010101u) << 2);
                        const uint32_t uu1 = ((uqw[1] >> sh_q) & 0x03030303u) | (((uhw[1] >> sh_h) & 0x01010101u) << 2);
                        const uint32_t uu2 = ((uqw[2] >> sh_q) & 0x03030303u) | (((uhw[2] >> sh_h) & 0x01010101u) << 2);
                        const uint32_t uu3 = ((uqw[3] >> sh_q) & 0x03030303u) | (((uhw[3] >> sh_h) & 0x01010101u) << 2);
                        const __half2 uw0 = mw_hadd2(mw_u8x2_to_h2_lo(uu0), neg1028), uw1 = mw_hadd2(mw_u8x2_to_h2_hi(uu0), neg1028);
                        const __half2 uw2 = mw_hadd2(mw_u8x2_to_h2_lo(uu1), neg1028), uw3 = mw_hadd2(mw_u8x2_to_h2_hi(uu1), neg1028);
                        const __half2 uw4 = mw_hadd2(mw_u8x2_to_h2_lo(uu2), neg1028), uw5 = mw_hadd2(mw_u8x2_to_h2_hi(uu2), neg1028);
                        const __half2 uw6 = mw_hadd2(mw_u8x2_to_h2_lo(uu3), neg1028), uw7 = mw_hadd2(mw_u8x2_to_h2_hi(uu3), neg1028);
                        const int4 *  xk4 = reinterpret_cast<const int4 *>(s_xh2 + bk * BT * 16 + sub * 8);
                        const float * dk  = s_xd + bk * BT;
                        #pragma unroll
                        for (int tj = 0; tj < BT; ++tj) {
                            const int4 xa = xk4[tj * 4 + 0];
                            const int4 xb = xk4[tj * 4 + 1];
                            __half2 ru = mw_hmul2(uw0, *reinterpret_cast<const __half2 *>(&xa.x));
                            ru = mw_hfma2(uw1, *reinterpret_cast<const __half2 *>(&xa.y), ru);
                            ru = mw_hfma2(uw2, *reinterpret_cast<const __half2 *>(&xa.z), ru);
                            ru = mw_hfma2(uw3, *reinterpret_cast<const __half2 *>(&xa.w), ru);
                            ru = mw_hfma2(uw4, *reinterpret_cast<const __half2 *>(&xb.x), ru);
                            ru = mw_hfma2(uw5, *reinterpret_cast<const __half2 *>(&xb.y), ru);
                            ru = mw_hfma2(uw6, *reinterpret_cast<const __half2 *>(&xb.z), ru);
                            ru = mw_hfma2(uw7, *reinterpret_cast<const __half2 *>(&xb.w), ru);
                            const float2 fu = __half22float2(ru);
                            au[tj] += (udl * dk[tj]) * (fu.x + fu.y);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (row < ff) {
        #pragma unroll
        for (int tj = 0; tj < BT; ++tj) {
            float gv = ag[tj], uv = au[tj];
            gv += __shfl_xor_sync(0xffffffffu, gv, 2, 4);
            uv += __shfl_xor_sync(0xffffffffu, uv, 2, 4);
            gv += __shfl_xor_sync(0xffffffffu, gv, 1, 4);
            uv += __shfl_xor_sync(0xffffffffu, uv, 1, 4);
            if (sub_idx == 0 && tj < n_tile) {
                if (clamp_limit > 0.0f) {
                    gv = fminf(fmaxf(gv, -clamp_limit), clamp_limit);
                    uv = fminf(fmaxf(uv, -clamp_limit), clamp_limit);
                }
                const int pair = s_pairs[tj];
                act[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * ff + row] =
                    (gv / (1.0f + expf(-gv))) * uv;
            }
        }
    }
}

template <int BT, int KT_BLK>
__global__ __launch_bounds__(128, 6) void mw_moe_down_grouped_q4_k_h2_kernel(
    const BlockQ4_K_M *  __restrict__ dn_exps,
    const ActF16Block *  __restrict__ a_f16,
    const float *        __restrict__ top_w,
    const MaxwTileDesc * __restrict__ tiles,
    const int32_t *      __restrict__ n_tiles_ptr,
    const int32_t *      __restrict__ sorted_pairs,
    float *              __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const MaxwTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    __shared__ float   s_tw[BT];
    constexpr int KT32 = KT_BLK * 8;
    auto * s_ah2 = reinterpret_cast<__half2 *>(smem_raw);
    auto * s_ad  = reinterpret_cast<float *>(s_ah2 + KT32 * BT * 16);
    auto * s_as0 = s_ad  + KT32 * BT;
    auto * s_as1 = s_as0 + KT32 * BT;

    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;
    const int row  = blockIdx.x * 16 + (tid >> 3);
    const int sub  = tid & 7;
    const int j    = sub >> 1;
    const int is   = sub & 1;

    if (tid < n_tile) {
        const int pair = sorted_pairs[start + tid];
        s_pairs[tid] = pair;
        s_tw[tid] = top_w[static_cast<size_t>(pair >> 4) * k_used + (pair & 15)];
    }
    __syncthreads();

    const int ff_blks = ff >> 5;
    const int nb      = ff >> 8;
    const BlockQ4_K_M * w_row = dn_exps + (static_cast<size_t>(eid) * K + row) * nb;
    const __half2 neg1024 = __floats2half2_rn(-1024.0f, -1024.0f);

    float acc[BT];
    #pragma unroll
    for (int tj = 0; tj < BT; ++tj) acc[tj] = 0.0f;

    for (int b0 = 0; b0 < nb; b0 += KT_BLK) {
        const int bc   = (b0 + KT_BLK < nb) ? KT_BLK : (nb - b0);
        const int bc32 = bc * 8;
        const int tot_i4 = bc32 * BT * 4;
        for (int idx = tid; idx < tot_i4; idx += nthr) {
            const int c   = idx & 3;
            const int rem = idx >> 2;
            const int tj  = rem % BT;
            const int bk  = rem / BT;
            if (tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                reinterpret_cast<int4 *>(s_ah2)[idx] =
                    reinterpret_cast<const int4 *>(a_f16[slot_idx * ff_blks + b0 * 8 + bk].x)[c];
            } else {
                reinterpret_cast<int4 *>(s_ah2)[idx] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = bc32 * BT;
        for (int idx = tid; idx < tot_d; idx += nthr) {
            const int tj = idx % BT;
            const int bk = idx / BT;
            if (tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                const ActF16Block & ab = a_f16[slot_idx * ff_blks + b0 * 8 + bk];
                s_ad[idx]  = ab.d;
                s_as0[idx] = ab.s0;
                s_as1[idx] = ab.s1;
            } else {
                s_ad[idx]  = 0.0f;
                s_as0[idx] = 0.0f;
                s_as1[idx] = 0.0f;
            }
        }
        __syncthreads();

        if (row < K) {
            for (int db = 0; db < bc; ++db) {
                const BlockQ4_K_M & b = w_row[b0 + db];
                const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
                const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
                const float d    = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
                const float dmin = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
                float sc1, m1, sc2, m2;
                mw_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
                mw_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
                const float d1 = d * sc1, mn1 = dmin * m1;
                const float d2 = d * sc2, mn2 = dmin * m2;

                const uint32_t qws[4] = {
                    static_cast<uint32_t>(ql_v.x), static_cast<uint32_t>(ql_v.y),
                    static_cast<uint32_t>(ql_v.z), static_cast<uint32_t>(ql_v.w)
                };
                const int bk0 = db * 8 + 2 * j;
                const float * s_as_cur = is ? s_as1 : s_as0;

                {
                    const uint32_t u0 = qws[0] & 0x0F0F0F0Fu, u1 = qws[1] & 0x0F0F0F0Fu;
                    const uint32_t u2 = qws[2] & 0x0F0F0F0Fu, u3 = qws[3] & 0x0F0F0F0Fu;
                    const __half2 w0 = mw_hadd2(mw_u8x2_to_h2_lo(u0), neg1024), w1 = mw_hadd2(mw_u8x2_to_h2_hi(u0), neg1024);
                    const __half2 w2 = mw_hadd2(mw_u8x2_to_h2_lo(u1), neg1024), w3 = mw_hadd2(mw_u8x2_to_h2_hi(u1), neg1024);
                    const __half2 w4 = mw_hadd2(mw_u8x2_to_h2_lo(u2), neg1024), w5 = mw_hadd2(mw_u8x2_to_h2_hi(u2), neg1024);
                    const __half2 w6 = mw_hadd2(mw_u8x2_to_h2_lo(u3), neg1024), w7 = mw_hadd2(mw_u8x2_to_h2_hi(u3), neg1024);
                    const int4 *  ak4 = reinterpret_cast<const int4 *>(s_ah2 + bk0 * BT * 16 + is * 8);
                    const float * dk  = s_ad     + bk0 * BT;
                    const float * sk  = s_as_cur + bk0 * BT;
                    #pragma unroll
                    for (int tj = 0; tj < BT; ++tj) {
                        const int4 aa = ak4[tj * 4 + 0];
                        const int4 ab = ak4[tj * 4 + 1];
                        __half2 r = mw_hmul2(w0, *reinterpret_cast<const __half2 *>(&aa.x));
                        r = mw_hfma2(w1, *reinterpret_cast<const __half2 *>(&aa.y), r);
                        r = mw_hfma2(w2, *reinterpret_cast<const __half2 *>(&aa.z), r);
                        r = mw_hfma2(w3, *reinterpret_cast<const __half2 *>(&aa.w), r);
                        r = mw_hfma2(w4, *reinterpret_cast<const __half2 *>(&ab.x), r);
                        r = mw_hfma2(w5, *reinterpret_cast<const __half2 *>(&ab.y), r);
                        r = mw_hfma2(w6, *reinterpret_cast<const __half2 *>(&ab.z), r);
                        r = mw_hfma2(w7, *reinterpret_cast<const __half2 *>(&ab.w), r);
                        const float2 rf = __half22float2(r);
                        acc[tj] += (d1 * dk[tj]) * (rf.x + rf.y) - mn1 * sk[tj];
                    }
                }
                {
                    const uint32_t u0 = (qws[0] >> 4) & 0x0F0F0F0Fu, u1 = (qws[1] >> 4) & 0x0F0F0F0Fu;
                    const uint32_t u2 = (qws[2] >> 4) & 0x0F0F0F0Fu, u3 = (qws[3] >> 4) & 0x0F0F0F0Fu;
                    const __half2 w0 = mw_hadd2(mw_u8x2_to_h2_lo(u0), neg1024), w1 = mw_hadd2(mw_u8x2_to_h2_hi(u0), neg1024);
                    const __half2 w2 = mw_hadd2(mw_u8x2_to_h2_lo(u1), neg1024), w3 = mw_hadd2(mw_u8x2_to_h2_hi(u1), neg1024);
                    const __half2 w4 = mw_hadd2(mw_u8x2_to_h2_lo(u2), neg1024), w5 = mw_hadd2(mw_u8x2_to_h2_hi(u2), neg1024);
                    const __half2 w6 = mw_hadd2(mw_u8x2_to_h2_lo(u3), neg1024), w7 = mw_hadd2(mw_u8x2_to_h2_hi(u3), neg1024);
                    const int4 *  ak4 = reinterpret_cast<const int4 *>(s_ah2 + (bk0 + 1) * BT * 16 + is * 8);
                    const float * dk  = s_ad     + (bk0 + 1) * BT;
                    const float * sk  = s_as_cur + (bk0 + 1) * BT;
                    #pragma unroll
                    for (int tj = 0; tj < BT; ++tj) {
                        const int4 aa = ak4[tj * 4 + 0];
                        const int4 ab = ak4[tj * 4 + 1];
                        __half2 r = mw_hmul2(w0, *reinterpret_cast<const __half2 *>(&aa.x));
                        r = mw_hfma2(w1, *reinterpret_cast<const __half2 *>(&aa.y), r);
                        r = mw_hfma2(w2, *reinterpret_cast<const __half2 *>(&aa.z), r);
                        r = mw_hfma2(w3, *reinterpret_cast<const __half2 *>(&aa.w), r);
                        r = mw_hfma2(w4, *reinterpret_cast<const __half2 *>(&ab.x), r);
                        r = mw_hfma2(w5, *reinterpret_cast<const __half2 *>(&ab.y), r);
                        r = mw_hfma2(w6, *reinterpret_cast<const __half2 *>(&ab.z), r);
                        r = mw_hfma2(w7, *reinterpret_cast<const __half2 *>(&ab.w), r);
                        const float2 rf = __half22float2(r);
                        acc[tj] += (d2 * dk[tj]) * (rf.x + rf.y) - mn2 * sk[tj];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (row < K) {
        #pragma unroll
        for (int tj = 0; tj < BT; ++tj) {
            float v = acc[tj];
            v += __shfl_xor_sync(0xffffffffu, v, 4, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 2, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 8);
            if (sub == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row] = s_tw[tj] * v;
            }
        }
    }
}

template <int BT, int KT_BLK>
__global__ __launch_bounds__(128, 6) void mw_moe_gate_up_grouped_q3_k_soa_dp4a_kernel(
    const int4 *          __restrict__ g_qs, const int4 * __restrict__ g_hm,
    const uint32_t *      __restrict__ g_sc, const uint16_t * __restrict__ g_d,
    const int4 *          __restrict__ u_qs, const int4 * __restrict__ u_hm,
    const uint32_t *      __restrict__ u_sc, const uint16_t * __restrict__ u_d,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const MaxwTileDesc *  __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ act,
    int ff, int K, int k_used, float clamp_limit) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const MaxwTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    constexpr int KT32 = KT_BLK * 8;
    auto * s_xq  = reinterpret_cast<int4 *>(smem_raw);
    auto * s_xd  = reinterpret_cast<float *>(s_xq + KT32 * BT * 2);
    auto * s_xs0 = reinterpret_cast<int32_t *>(s_xd + KT32 * BT);
    auto * s_xs1 = s_xs0 + KT32 * BT;

    const int tid     = threadIdx.x;
    const int nthr    = blockDim.x;
    const int row     = blockIdx.x * 32 + (tid >> 2);
    const int sub_idx = tid & 3;
    const int n128    = sub_idx >> 1;
    const int sub     = sub_idx & 1;

    if (tid < n_tile) s_pairs[tid] = sorted_pairs[start + tid];
    __syncthreads();

    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const size_t rblk  = (static_cast<size_t>(eid) * ff + row) * nb;

    float ag[BT], au[BT];
    #pragma unroll
    for (int tj = 0; tj < BT; ++tj) { ag[tj] = 0.0f; au[tj] = 0.0f; }

    for (int b0 = 0; b0 < nb; b0 += KT_BLK) {
        const int bc   = (b0 + KT_BLK < nb) ? KT_BLK : (nb - b0);
        const int bc32 = bc * 8;
        const int tot_i4 = bc32 * BT * 2;
        for (int idx = tid; idx < tot_i4; idx += nthr) {
            const int c   = idx & 1;
            const int rem = idx >> 1;
            const int tj  = rem % BT;
            const int bk  = rem / BT;
            if (tj < n_tile) {
                const int tok = s_pairs[tj] >> 4;
                s_xq[idx] = reinterpret_cast<const int4 *>(x_q8[static_cast<size_t>(tok) * K_blocks + b0 * 8 + bk].qs)[c];
            } else {
                s_xq[idx] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = bc32 * BT;
        for (int idx = tid; idx < tot_d; idx += nthr) {
            const int tj = idx % BT;
            const int bk = idx / BT;
            if (tj < n_tile) {
                const BlockQ8_1_Act & xb = x_q8[static_cast<size_t>(s_pairs[tj] >> 4) * K_blocks + b0 * 8 + bk];
                s_xd[idx]  = xb.d;
                s_xs0[idx] = xb.s0;
                s_xs1[idx] = xb.s1;
            } else {
                s_xd[idx]  = 0.0f;
                s_xs0[idx] = 0;
                s_xs1[idx] = 0;
            }
        }
        __syncthreads();

        if (row < ff) {
            const int32_t * s_xs_cur = sub ? s_xs1 : s_xs0;
            for (int db = 0; db < bc; ++db) {
                const int64_t bi = static_cast<int64_t>(rblk) + b0 + db;
                {
                    const int4 g_qv = __ldg(&g_qs[bi * 4 + sub_idx]);
                    const int4 g_hv = __ldg(&g_hm[bi * 2 + sub]);
                    const float g_da = mw_half_bits_to_f32(__ldg(&g_d[bi]));
                    uint32_t g_aux[4];
                    mw_unpack_q3k_scales(g_sc, bi, g_aux);
                    const uint32_t gqw[4] = { static_cast<uint32_t>(g_qv.x), static_cast<uint32_t>(g_qv.y), static_cast<uint32_t>(g_qv.z), static_cast<uint32_t>(g_qv.w) };
                    const uint32_t ghw[4] = { static_cast<uint32_t>(g_hv.x), static_cast<uint32_t>(g_hv.y), static_cast<uint32_t>(g_hv.z), static_cast<uint32_t>(g_hv.w) };
                    #pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        const int grp   = n128 * 8 + g * 2 + sub;
                        const float gdl = g_da * static_cast<float>(static_cast<int>((g_aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu) - 32);
                        const int bk    = db * 8 + n128 * 4 + g;
                        const int sh_q  = 2 * g;
                        const int sh_h  = n128 * 4 + g;
                        const int gu0 = static_cast<int>(((gqw[0] >> sh_q) & 0x03030303u) | (((ghw[0] >> sh_h) & 0x01010101u) << 2));
                        const int gu1 = static_cast<int>(((gqw[1] >> sh_q) & 0x03030303u) | (((ghw[1] >> sh_h) & 0x01010101u) << 2));
                        const int gu2 = static_cast<int>(((gqw[2] >> sh_q) & 0x03030303u) | (((ghw[2] >> sh_h) & 0x01010101u) << 2));
                        const int gu3 = static_cast<int>(((gqw[3] >> sh_q) & 0x03030303u) | (((ghw[3] >> sh_h) & 0x01010101u) << 2));
                        const int4 *    xk = s_xq     + bk * BT * 2 + sub;
                        const float *   dk = s_xd     + bk * BT;
                        const int32_t * sk = s_xs_cur + bk * BT;
                        #pragma unroll
                        for (int tj = 0; tj < BT; ++tj) {
                            const int4 x4 = xk[tj * 2];
                            const int sg = dp4a_s32(gu0, x4.x, dp4a_s32(gu1, x4.y, dp4a_s32(gu2, x4.z, dp4a_s32(gu3, x4.w, 0))));
                            ag[tj] += (gdl * dk[tj]) * static_cast<float>(sg - 4 * sk[tj]);
                        }
                    }
                }
                {
                    const int4 u_qv = __ldg(&u_qs[bi * 4 + sub_idx]);
                    const int4 u_hv = __ldg(&u_hm[bi * 2 + sub]);
                    const float u_da = mw_half_bits_to_f32(__ldg(&u_d[bi]));
                    uint32_t u_aux[4];
                    mw_unpack_q3k_scales(u_sc, bi, u_aux);
                    const uint32_t uqw[4] = { static_cast<uint32_t>(u_qv.x), static_cast<uint32_t>(u_qv.y), static_cast<uint32_t>(u_qv.z), static_cast<uint32_t>(u_qv.w) };
                    const uint32_t uhw[4] = { static_cast<uint32_t>(u_hv.x), static_cast<uint32_t>(u_hv.y), static_cast<uint32_t>(u_hv.z), static_cast<uint32_t>(u_hv.w) };
                    #pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        const int grp   = n128 * 8 + g * 2 + sub;
                        const float udl = u_da * static_cast<float>(static_cast<int>((u_aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu) - 32);
                        const int bk    = db * 8 + n128 * 4 + g;
                        const int sh_q  = 2 * g;
                        const int sh_h  = n128 * 4 + g;
                        const int uu0 = static_cast<int>(((uqw[0] >> sh_q) & 0x03030303u) | (((uhw[0] >> sh_h) & 0x01010101u) << 2));
                        const int uu1 = static_cast<int>(((uqw[1] >> sh_q) & 0x03030303u) | (((uhw[1] >> sh_h) & 0x01010101u) << 2));
                        const int uu2 = static_cast<int>(((uqw[2] >> sh_q) & 0x03030303u) | (((uhw[2] >> sh_h) & 0x01010101u) << 2));
                        const int uu3 = static_cast<int>(((uqw[3] >> sh_q) & 0x03030303u) | (((uhw[3] >> sh_h) & 0x01010101u) << 2));
                        const int4 *    xk = s_xq     + bk * BT * 2 + sub;
                        const float *   dk = s_xd     + bk * BT;
                        const int32_t * sk = s_xs_cur + bk * BT;
                        #pragma unroll
                        for (int tj = 0; tj < BT; ++tj) {
                            const int4 x4 = xk[tj * 2];
                            const int su = dp4a_s32(uu0, x4.x, dp4a_s32(uu1, x4.y, dp4a_s32(uu2, x4.z, dp4a_s32(uu3, x4.w, 0))));
                            au[tj] += (udl * dk[tj]) * static_cast<float>(su - 4 * sk[tj]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (row < ff) {
        #pragma unroll
        for (int tj = 0; tj < BT; ++tj) {
            float gv = ag[tj], uv = au[tj];
            gv += __shfl_xor_sync(0xffffffffu, gv, 2, 4);
            uv += __shfl_xor_sync(0xffffffffu, uv, 2, 4);
            gv += __shfl_xor_sync(0xffffffffu, gv, 1, 4);
            uv += __shfl_xor_sync(0xffffffffu, uv, 1, 4);
            if (sub_idx == 0 && tj < n_tile) {
                if (clamp_limit > 0.0f) {
                    gv = fminf(fmaxf(gv, -clamp_limit), clamp_limit);
                    uv = fminf(fmaxf(uv, -clamp_limit), clamp_limit);
                }
                const int pair = s_pairs[tj];
                act[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * ff + row] =
                    (gv / (1.0f + expf(-gv))) * uv;
            }
        }
    }
}

template <int BT, int KT_BLK>
__global__ __launch_bounds__(128) void mw_moe_down_grouped_q4_k_dp4a_kernel(
    const BlockQ4_K_M *   __restrict__ dn_exps,
    const BlockQ8_1_Act * __restrict__ a_q8,
    const float *         __restrict__ top_w,
    const MaxwTileDesc *  __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const MaxwTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    __shared__ float   s_tw[BT];
    constexpr int KT32 = KT_BLK * 8;
    auto * s_aq  = reinterpret_cast<int4 *>(smem_raw);
    auto * s_ad  = reinterpret_cast<float *>(s_aq + KT32 * BT * 2);
    auto * s_as0 = reinterpret_cast<int32_t *>(s_ad + KT32 * BT);
    auto * s_as1 = s_as0 + KT32 * BT;

    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;
    const int row  = blockIdx.x * 16 + (tid >> 3);
    const int sub  = tid & 7;
    const int j    = sub >> 1;
    const int is   = sub & 1;

    if (tid < n_tile) {
        const int pair = sorted_pairs[start + tid];
        s_pairs[tid] = pair;
        s_tw[tid] = top_w[static_cast<size_t>(pair >> 4) * k_used + (pair & 15)];
    }
    __syncthreads();

    const int ff_blks = ff >> 5;
    const int nb      = ff >> 8;
    const BlockQ4_K_M * w_row = dn_exps + (static_cast<size_t>(eid) * K + row) * nb;

    float acc[BT];
    #pragma unroll
    for (int tj = 0; tj < BT; ++tj) acc[tj] = 0.0f;

    for (int b0 = 0; b0 < nb; b0 += KT_BLK) {
        const int bc   = (b0 + KT_BLK < nb) ? KT_BLK : (nb - b0);
        const int bc32 = bc * 8;
        const int tot_i4 = bc32 * BT * 2;
        for (int idx = tid; idx < tot_i4; idx += nthr) {
            const int c   = idx & 1;
            const int rem = idx >> 1;
            const int tj  = rem % BT;
            const int bk  = rem / BT;
            if (tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                s_aq[idx] = reinterpret_cast<const int4 *>(a_q8[slot_idx * ff_blks + b0 * 8 + bk].qs)[c];
            } else {
                s_aq[idx] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = bc32 * BT;
        for (int idx = tid; idx < tot_d; idx += nthr) {
            const int tj = idx % BT;
            const int bk = idx / BT;
            if (tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                const BlockQ8_1_Act & ab = a_q8[slot_idx * ff_blks + b0 * 8 + bk];
                s_ad[idx]  = ab.d;
                s_as0[idx] = ab.s0;
                s_as1[idx] = ab.s1;
            } else {
                s_ad[idx]  = 0.0f;
                s_as0[idx] = 0;
                s_as1[idx] = 0;
            }
        }
        __syncthreads();

        if (row < K) {
            const int32_t * s_as_cur = is ? s_as1 : s_as0;
            for (int db = 0; db < bc; ++db) {
                const BlockQ4_K_M & b = w_row[b0 + db];
                const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
                const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
                const float d    = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
                const float dmin = mw_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
                float sc1, m1, sc2, m2;
                mw_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
                mw_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
                const float d1 = d * sc1, mn1 = dmin * m1;
                const float d2 = d * sc2, mn2 = dmin * m2;

                const int q1x = static_cast<int>(static_cast<uint32_t>(ql_v.x) & 0x0F0F0F0Fu);
                const int q2x = static_cast<int>((static_cast<uint32_t>(ql_v.x) >> 4) & 0x0F0F0F0Fu);
                const int q1y = static_cast<int>(static_cast<uint32_t>(ql_v.y) & 0x0F0F0F0Fu);
                const int q2y = static_cast<int>((static_cast<uint32_t>(ql_v.y) >> 4) & 0x0F0F0F0Fu);
                const int q1z = static_cast<int>(static_cast<uint32_t>(ql_v.z) & 0x0F0F0F0Fu);
                const int q2z = static_cast<int>((static_cast<uint32_t>(ql_v.z) >> 4) & 0x0F0F0F0Fu);
                const int q1w = static_cast<int>(static_cast<uint32_t>(ql_v.w) & 0x0F0F0F0Fu);
                const int q2w = static_cast<int>((static_cast<uint32_t>(ql_v.w) >> 4) & 0x0F0F0F0Fu);

                const int bk0 = db * 8 + 2 * j;
                const int4 *    ak0 = s_aq     + bk0 * BT * 2 + is;
                const int4 *    ak1 = s_aq     + (bk0 + 1) * BT * 2 + is;
                const float *   dk0 = s_ad     + bk0 * BT;
                const float *   dk1 = s_ad     + (bk0 + 1) * BT;
                const int32_t * sk0 = s_as_cur + bk0 * BT;
                const int32_t * sk1 = s_as_cur + (bk0 + 1) * BT;
                #pragma unroll
                for (int tj = 0; tj < BT; ++tj) {
                    const int4 x1 = ak0[tj * 2];
                    const int4 x2 = ak1[tj * 2];
                    const int sum1 = dp4a_s32(q1x, x1.x, dp4a_s32(q1y, x1.y, dp4a_s32(q1z, x1.z, dp4a_s32(q1w, x1.w, 0))));
                    const int sum2 = dp4a_s32(q2x, x2.x, dp4a_s32(q2y, x2.y, dp4a_s32(q2z, x2.z, dp4a_s32(q2w, x2.w, 0))));
                    acc[tj] += dk0[tj] * (d1 * static_cast<float>(sum1) - mn1 * static_cast<float>(sk0[tj]))
                             + dk1[tj] * (d2 * static_cast<float>(sum2) - mn2 * static_cast<float>(sk1[tj]));
                }
            }
        }
        __syncthreads();
    }

    if (row < K) {
        #pragma unroll
        for (int tj = 0; tj < BT; ++tj) {
            float v = acc[tj];
            v += __shfl_xor_sync(0xffffffffu, v, 4, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 2, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 8);
            if (sub == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row] = s_tw[tj] * v;
            }
        }
    }
}

// Generic Q3_K SoA MoE fallback (e.g. if Q3_K SoA is paired with non-Q4_K down, or Inkling2 Q3_K)
__global__ void mw_moe_gate_up_q3_k_soa_only_kernel(
    const int4 *          __restrict__ g_qs, const int4 * __restrict__ g_hm,
    const uint32_t *      __restrict__ g_sc, const uint16_t * __restrict__ g_d,
    const int4 *          __restrict__ u_qs, const int4 * __restrict__ u_hm,
    const uint32_t *      __restrict__ u_sc, const uint16_t * __restrict__ u_d,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ top_ids,
    float *               __restrict__ act,
    int ff, int K, int k_used, float clamp_limit) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<BlockQ8_1_Act *>(smem_raw);
    const int slot     = blockIdx.y;
    const int t        = blockIdx.z;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int n_i4     = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(t) * K_blocks);
    for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_x)[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= ff) return;
    const int eid = top_ids[static_cast<size_t>(t) * k_used + slot];
    const size_t rblk = (static_cast<size_t>(eid) * ff + row) * nb;
    float ag = dot_q3_k_soa_row_dp4a(g_qs + rblk * 4, g_hm + rblk * 2, g_sc + rblk * 3, g_d + rblk, s_x, nb, threadIdx.x);
    float au = dot_q3_k_soa_row_dp4a(u_qs + rblk * 4, u_hm + rblk * 2, u_sc + rblk * 3, u_d + rblk, s_x, nb, threadIdx.x);
    if (threadIdx.x == 0) {
        if (clamp_limit > 0.0f) {
            ag = fminf(fmaxf(ag, -clamp_limit), clamp_limit);
            au = fminf(fmaxf(au, -clamp_limit), clamp_limit);
        }
        act[(static_cast<size_t>(t) * k_used + slot) * ff + row] = (ag / (1.0f + expf(-ag))) * au;
    }
}

__global__ void mw_moe_down_q3_k_soa_only_kernel(
    const int4 *          __restrict__ d_qs, const int4 * __restrict__ d_hm,
    const uint32_t *      __restrict__ d_sc, const uint16_t * __restrict__ d_d,
    const BlockQ8_1_Act * __restrict__ a_q8,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ out,
    int K, int ff, int k_used) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_a = reinterpret_cast<BlockQ8_1_Act *>(smem_raw);
    const int t       = blockIdx.y;
    const int row     = blockIdx.x * blockDim.y + threadIdx.y;
    const int ff_blks = ff >> 5;
    const int nb      = ff >> 8;
    const int tid     = threadIdx.y * 32 + threadIdx.x;
    const int nthr    = blockDim.y * 32;
    const int n_i4    = (ff_blks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;

    float total = 0.0f;
    for (int slot = 0; slot < k_used; ++slot) {
        const int4 * src = reinterpret_cast<const int4 *>(a_q8 + (static_cast<size_t>(t) * k_used + slot) * ff_blks);
        for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_a)[i] = src[i];
        __syncthreads();
        if (row < K) {
            const int   eid = top_ids[static_cast<size_t>(t) * k_used + slot];
            const float ew  = top_w  [static_cast<size_t>(t) * k_used + slot];
            const size_t rblk = (static_cast<size_t>(eid) * K + row) * nb;
            total += ew * dot_q3_k_soa_row_dp4a(d_qs + rblk * 4, d_hm + rblk * 2, d_sc + rblk * 3, d_d + rblk, s_a, nb, threadIdx.x);
        }
        __syncthreads();
    }
    if (row < K && threadIdx.x == 0) out[static_cast<size_t>(t) * K + row] = total;
}

} // namespace

// ============================================================================
// Public Launchers
// ============================================================================

void launch_embedding_q3_k_soa(
    const DeviceTensor & tok_embd, const int32_t * d_tokens, float * d_out,
    int n_embd, int hc_mult, int n_tokens, cudaStream_t stream) {
    dim3 grid(n_embd >> 8, n_tokens);
    mw_embedding_q3_k_soa_kernel<<<grid, 32, 0, stream>>>(
        static_cast<const uint8_t *>(tok_embd.ptr),
        static_cast<const uint8_t *>(tok_embd.aux_ptr),
        static_cast<const uint32_t *>(tok_embd.scales_ptr),
        static_cast<const uint16_t *>(tok_embd.d_ptr),
        d_tokens, d_out, n_embd, hc_mult);
}

void launch_dequant_q3_k_soa(
    const DeviceTensor & W, int64_t b0, int64_t nblk, void * d_dst, bool dst_f16, cudaStream_t stream) {
    constexpr int wpb = 8;
    const auto * qs = static_cast<const uint8_t *>(W.ptr) + b0 * 64;
    const auto * hm = static_cast<const uint8_t *>(W.aux_ptr) + b0 * 32;
    const auto * sc = static_cast<const uint32_t *>(W.scales_ptr) + b0 * 3;
    const auto * d  = static_cast<const uint16_t *>(W.d_ptr) + b0;
    if (dst_f16) {
        mw_dequant_q3_k_soa_kernel<__half><<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
            qs, hm, sc, d, static_cast<__half *>(d_dst), nblk);
    } else {
        mw_dequant_q3_k_soa_kernel<float><<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
            qs, hm, sc, d, static_cast<float *>(d_dst), nblk);
    }
}

void launch_linear_q3_k_soa_prequant(
    const DeviceTensor & W, const void * d_xq8, float * d_y, int T, cudaStream_t stream) {
    const int K        = static_cast<int>(W.ne[0]);
    const int M        = static_cast<int>(W.ne[1]);
    const int K_blocks = K >> 5;
    constexpr int wpb  = 8;
    const auto * xq = static_cast<const BlockQ8_1_Act *>(d_xq8);
    const auto * qs = static_cast<const int4 *>(W.ptr);
    const auto * hm = static_cast<const int4 *>(W.aux_ptr);
    const auto * sc = static_cast<const uint32_t *>(W.scales_ptr);
    const auto * d  = static_cast<const uint16_t *>(W.d_ptr);
    if (T == 2 && K_blocks <= 384) {
        const size_t smem = static_cast<size_t>(2 * K_blocks) * sizeof(BlockQ8_1_Act);
        mw_gemv_q3_k_soa_dp4a_kernel<true><<<(M + wpb - 1) / wpb, dim3(32, wpb), smem, stream>>>(qs, hm, sc, d, xq, d_y, M, K);
    } else {
        const size_t smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
        mw_gemv_q3_k_soa_dp4a_kernel<false><<<dim3((M + wpb - 1) / wpb, T), dim3(32, wpb), smem, stream>>>(qs, hm, sc, d, xq, d_y, M, K);
    }
}

bool try_launch_linear_kquant_prefill_sm60_f16(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    if (cublas == nullptr || T < 16 || device_sm_version(W.device) != 60) return false;
    if (!(W.dtype == DType::Q4_K || W.dtype == DType::Q5_K || (W.dtype == DType::Q3_K && W.is_soa))) {
        return false;
    }

    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    const size_t xf16_b = align256(static_cast<size_t>(T) * K * sizeof(__half));
    const size_t yf16_b = align256(static_cast<size_t>(T) * M * sizeof(__half));
    if (xf16_b + yf16_b + static_cast<size_t>(K) * sizeof(__half) > scratch_bytes) return false;

    const size_t rem = scratch_bytes - xf16_b - yf16_b;
    const int chunk_m = std::max(1, std::min(M, static_cast<int>(rem / (static_cast<size_t>(K) * sizeof(__half)))));
    auto * base   = static_cast<uint8_t *>(d_scratch);
    auto * d_x16  = reinterpret_cast<__half *>(base);
    auto * d_y16  = reinterpret_cast<__half *>(base + xf16_b);
    auto * d_w16  = reinterpret_cast<__half *>(base + xf16_b + yf16_b);

    launch_cast_f32_to_f16(d_x, d_x16, static_cast<int64_t>(T) * K, stream);
    cublasSetStream(cublas, stream);
    const __half h_one = __float2half_rn(1.0f), h_zero = __float2half_rn(0.0f);
    const int nb_row = K >> 8;
    constexpr int wpb = 8;

    for (int m0 = 0; m0 < M; m0 += chunk_m) {
        const int cm = std::min(chunk_m, M - m0);
        const int64_t b0   = static_cast<int64_t>(m0) * nb_row;
        const int64_t nblk = static_cast<int64_t>(cm) * nb_row;
        if (W.dtype == DType::Q3_K) {
            launch_dequant_q3_k_soa(W, b0, nblk, d_w16, true, stream);
        } else if (W.dtype == DType::Q4_K) {
            mw_dequant_q4_k_f16_kernel<<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
                static_cast<const BlockQ4_K_M *>(W.ptr) + b0, d_w16, nblk);
        } else {
            mw_dequant_q5_k_f16_kernel<<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
                static_cast<const BlockQ5_K_M *>(W.ptr) + b0, d_w16, nblk);
        }
        cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    cm, T, K,
                    &h_one,  d_w16, K,
                    d_x16, K,
                    &h_zero, d_y16 + m0, M);
    }

    const int64_t ny = static_cast<int64_t>(T) * M;
    mw_f16_to_f32_vec_kernel<<<((ny + 1) / 2 + 255) / 256, 256, 0, stream>>>(d_y16, d_y, ny);
    return true;
}

bool try_launch_linear_kquant_sm60_h2(
    const DeviceTensor & W, const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    if (T >= 16 || device_sm_version(W.device) != 60) return false;
    int dt_code = -1;
    if (W.dtype == DType::Q3_K && W.is_soa)      dt_code = 0;
    else if (W.dtype == DType::Q4_K)             dt_code = 1;
    else if (W.dtype == DType::Q5_K)             dt_code = 2;
    else if (W.dtype == DType::Q6_K && W.is_soa) dt_code = 3;
    else return false;

    const int K        = static_cast<int>(W.ne[0]);
    const int M        = static_cast<int>(W.ne[1]);
    const int K_blocks = K >> 5;
    const size_t x_bytes = static_cast<size_t>(T) * K_blocks * sizeof(ActF16Block);
    if (x_bytes > scratch_bytes) return false;

    auto * x_f16 = static_cast<ActF16Block *>(d_scratch);
    const int tot_b = T * K_blocks;
    const int q_wpb = (tot_b <= 256) ? 1 : 8;
    const int nb    = K >> 8;

    if (dt_code == 1) {
        mw_quantize_act_f16_eo_kernel<<<(tot_b + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(d_x, x_f16, tot_b);
        const auto * wp4 = static_cast<const BlockQ4_K_M *>(W.ptr);
        const bool use_t2 = (T == 2);
        const int tile_nb = std::min(nb, MW_SW_TILE_NB);
        const size_t sw_smem = static_cast<size_t>((use_t2 ? 2 : 1) * tile_nb * 8) * sizeof(ActF16Block);
        if (nb < 16) {
            const int sw_wpb = (M >= 4096) ? 4 : 2;
            const int rpc    = sw_wpb * 4;
            if (use_t2) mw_gemv_q4_k_sw8_h2_kernel<true> <<<(M + rpc - 1) / rpc, dim3(32, sw_wpb), sw_smem, stream>>>(wp4, x_f16, d_y, M, K, tile_nb);
            else        mw_gemv_q4_k_sw8_h2_kernel<false><<<dim3((M + rpc - 1) / rpc, T), dim3(32, sw_wpb), sw_smem, stream>>>(wp4, x_f16, d_y, M, K, tile_nb);
        } else {
            constexpr int wpb = 8;
            if (use_t2) mw_gemv_q4_k_eo32_h2_kernel<true> <<<(M + wpb - 1) / wpb, dim3(32, wpb), sw_smem, stream>>>(wp4, x_f16, d_y, M, K, tile_nb);
            else        mw_gemv_q4_k_eo32_h2_kernel<false><<<dim3((M + wpb - 1) / wpb, T), dim3(32, wpb), sw_smem, stream>>>(wp4, x_f16, d_y, M, K, tile_nb);
        }
        return true;
    }

    if (dt_code == 3) {
        mw_quantize_act_f16_kernel<<<(tot_b + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(d_x, x_f16, tot_b);
        const auto * ql = static_cast<const int4 *>(W.ptr);
        const auto * qh = static_cast<const int4 *>(W.aux_ptr);
        const auto * sc = static_cast<const int8_t *>(W.scales_ptr);
        const auto * d  = static_cast<const uint16_t *>(W.d_ptr);
        constexpr int wpb = 8;
        if (nb > MW_SW_TILE_NB) {
            // Large K (e.g. K=12288 ffn_down): tile smem at 10 KB for 3x higher occupancy!
            const bool use_t2 = (T == 2);
            const size_t t_smem = static_cast<size_t>((use_t2 ? 2 : 1) * MW_SW_TILE_NB * 8) * sizeof(ActF16Block);
            if (use_t2) {
                mw_gemv_q6_k_soa_tiled_h2_kernel<true><<<(M + wpb - 1) / wpb, dim3(32, wpb), t_smem, stream>>>(ql, qh, sc, d, x_f16, d_y, M, K, MW_SW_TILE_NB);
            } else {
                mw_gemv_q6_k_soa_tiled_h2_kernel<false><<<dim3((M + wpb - 1) / wpb, T), dim3(32, wpb), t_smem, stream>>>(ql, qh, sc, d, x_f16, d_y, M, K, MW_SW_TILE_NB);
            }
            return true;
        }
        const bool use_t2 = (T == 2 && K_blocks <= 256);
        const size_t smem = static_cast<size_t>((use_t2 ? 2 : 1) * K_blocks) * sizeof(ActF16Block);
        if (M >= 16384) {
            constexpr int R = 4;
            if (use_t2) mw_gemv_q6_k_soa_h2_kernel<true, R> <<<(M + wpb * R - 1) / (wpb * R), dim3(32, wpb), smem, stream>>>(ql, qh, sc, d, x_f16, d_y, M, K);
            else        mw_gemv_q6_k_soa_h2_kernel<false, R><<<dim3((M + wpb * R - 1) / (wpb * R), T), dim3(32, wpb), smem, stream>>>(ql, qh, sc, d, x_f16, d_y, M, K);
        } else {
            if (use_t2) mw_gemv_q6_k_soa_h2_kernel<true, 1> <<<(M + wpb - 1) / wpb, dim3(32, wpb), smem, stream>>>(ql, qh, sc, d, x_f16, d_y, M, K);
            else        mw_gemv_q6_k_soa_h2_kernel<false, 1><<<dim3((M + wpb - 1) / wpb, T), dim3(32, wpb), smem, stream>>>(ql, qh, sc, d, x_f16, d_y, M, K);
        }
        return true;
    }

    mw_quantize_act_f16_kernel<<<(tot_b + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(d_x, x_f16, tot_b);
    constexpr int wpb = 8;
    const bool use_t2 = (T == 2 && K_blocks <= 256);
    const size_t smem = static_cast<size_t>((use_t2 ? 2 : 1) * K_blocks) * sizeof(ActF16Block);
    if (smem > 48000) return false;

    MwMultiProjDesc desc{};
    desc.n_mat = 1;
    desc.dt[0] = dt_code;
    desc.p0[0] = W.ptr;
    desc.p1[0] = W.aux_ptr;
    desc.p2[0] = W.scales_ptr;
    desc.p3[0] = W.d_ptr;
    desc.y[0]  = d_y;
    desc.m[0]  = M;
    desc.row_end[0] = M;

    if (use_t2) {
        mw_gemv_multi_h2_kernel<true><<<(M + wpb - 1) / wpb, dim3(32, wpb), smem, stream>>>(desc, x_f16, K);
    } else {
        mw_gemv_multi_h2_kernel<false><<<dim3((M + wpb - 1) / wpb, T), dim3(32, wpb), smem, stream>>>(desc, x_f16, K);
    }
    return true;
}

bool try_launch_linear_multi_sm60_h2(
    const DeviceTensor * const * Ws, float * const * Ys, int n_mat,
    const float * d_x, int T, void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    if (n_mat <= 0 || n_mat > 6 || T >= 16 || device_sm_version(Ws[0]->device) != 60) return false;
    const int K        = static_cast<int>(Ws[0]->ne[0]);
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    if (static_cast<size_t>(T) * K_blocks * sizeof(ActF16Block) > scratch_bytes) return false;

    bool all_q4 = true;
    for (int i = 0; i < n_mat; ++i) {
        if (Ws[i]->dtype != DType::Q4_K) { all_q4 = false; break; }
    }

    auto * x_f16 = static_cast<ActF16Block *>(d_scratch);
    const int tot_b = T * K_blocks;
    const int q_wpb = (tot_b <= 256) ? 1 : 8;

    if (all_q4) {
        const bool use_t2 = (T == 2);
        const int tile_nb = std::min(nb, MW_SW_TILE_NB);
        const size_t sw_smem = static_cast<size_t>((use_t2 ? 2 : 1) * tile_nb * 8) * sizeof(ActF16Block);

        mw_quantize_act_f16_eo_kernel<<<(tot_b + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(d_x, x_f16, tot_b);
        const BlockQ4_K_M * q4_w[6] = {};
        float *             q4_y[6] = {};
        int                 q4_m[6] = {};
        int                 q4_e[6] = {};
        int tot_q4 = 0;
        for (int i = 0; i < n_mat; ++i) {
            q4_w[i] = static_cast<const BlockQ4_K_M *>(Ws[i]->ptr);
            q4_y[i] = Ys[i];
            q4_m[i] = static_cast<int>(Ws[i]->ne[1]);
            tot_q4 += q4_m[i];
            q4_e[i] = tot_q4;
        }
        if (nb < 16) {
            const int sw_wpb = (tot_q4 >= 4096) ? 4 : 2;
            const int rpc    = sw_wpb * 4;
            if (n_mat == 1) {
                if (use_t2) mw_gemv_q4_k_sw8_h2_kernel<true> <<<(tot_q4 + rpc - 1) / rpc, dim3(32, sw_wpb), sw_smem, stream>>>(q4_w[0], x_f16, q4_y[0], tot_q4, K, tile_nb);
                else        mw_gemv_q4_k_sw8_h2_kernel<false><<<dim3((tot_q4 + rpc - 1) / rpc, T), dim3(32, sw_wpb), sw_smem, stream>>>(q4_w[0], x_f16, q4_y[0], tot_q4, K, tile_nb);
            } else {
                for (int i = n_mat; i < 6; ++i) q4_e[i] = tot_q4;
                dim3 grid = use_t2 ? dim3((tot_q4 + rpc - 1) / rpc, 1) : dim3((tot_q4 + rpc - 1) / rpc, T);
                if (use_t2) {
                    mw_gemv_q4_k_multi_sw8_h2_kernel<true><<<grid, dim3(32, sw_wpb), sw_smem, stream>>>(
                        q4_w[0], q4_y[0], q4_m[0], q4_e[0], q4_w[1], q4_y[1], q4_m[1], q4_e[1],
                        q4_w[2], q4_y[2], q4_m[2], q4_e[2], q4_w[3], q4_y[3], q4_m[3], q4_e[3],
                        q4_w[4], q4_y[4], q4_m[4], q4_e[4], q4_w[5], q4_y[5], q4_m[5], q4_e[5],
                        tot_q4, x_f16, K, tile_nb);
                } else {
                    mw_gemv_q4_k_multi_sw8_h2_kernel<false><<<grid, dim3(32, sw_wpb), sw_smem, stream>>>(
                        q4_w[0], q4_y[0], q4_m[0], q4_e[0], q4_w[1], q4_y[1], q4_m[1], q4_e[1],
                        q4_w[2], q4_y[2], q4_m[2], q4_e[2], q4_w[3], q4_y[3], q4_m[3], q4_e[3],
                        q4_w[4], q4_y[4], q4_m[4], q4_e[4], q4_w[5], q4_y[5], q4_m[5], q4_e[5],
                        tot_q4, x_f16, K, tile_nb);
                }
            }
        } else {
            constexpr int wpb = 8;
            if (n_mat == 1) {
                if (use_t2) mw_gemv_q4_k_eo32_h2_kernel<true> <<<(tot_q4 + wpb - 1) / wpb, dim3(32, wpb), sw_smem, stream>>>(q4_w[0], x_f16, q4_y[0], tot_q4, K, tile_nb);
                else        mw_gemv_q4_k_eo32_h2_kernel<false><<<dim3((tot_q4 + wpb - 1) / wpb, T), dim3(32, wpb), sw_smem, stream>>>(q4_w[0], x_f16, q4_y[0], tot_q4, K, tile_nb);
            } else {
                for (int i = n_mat; i < 6; ++i) q4_e[i] = tot_q4;
                dim3 grid = use_t2 ? dim3((tot_q4 + wpb - 1) / wpb, 1) : dim3((tot_q4 + wpb - 1) / wpb, T);
                if (use_t2) {
                    mw_gemv_q4_k_multi_eo32_h2_kernel<true><<<grid, dim3(32, wpb), sw_smem, stream>>>(
                        q4_w[0], q4_y[0], q4_m[0], q4_e[0], q4_w[1], q4_y[1], q4_m[1], q4_e[1],
                        q4_w[2], q4_y[2], q4_m[2], q4_e[2], q4_w[3], q4_y[3], q4_m[3], q4_e[3],
                        q4_w[4], q4_y[4], q4_m[4], q4_e[4], q4_w[5], q4_y[5], q4_m[5], q4_e[5],
                        tot_q4, x_f16, K, tile_nb);
                } else {
                    mw_gemv_q4_k_multi_eo32_h2_kernel<false><<<grid, dim3(32, wpb), sw_smem, stream>>>(
                        q4_w[0], q4_y[0], q4_m[0], q4_e[0], q4_w[1], q4_y[1], q4_m[1], q4_e[1],
                        q4_w[2], q4_y[2], q4_m[2], q4_e[2], q4_w[3], q4_y[3], q4_m[3], q4_e[3],
                        q4_w[4], q4_y[4], q4_m[4], q4_e[4], q4_w[5], q4_y[5], q4_m[5], q4_e[5],
                        tot_q4, x_f16, K, tile_nb);
                }
            }
        }
        return true;
    }

    const bool use_t2  = (T == 2 && K_blocks <= 256);
    const size_t smem  = static_cast<size_t>((use_t2 ? 2 : 1) * K_blocks) * sizeof(ActF16Block);
    if (smem > 48000) return false;

    MwMultiProjDesc desc{};
    int tot = 0;
    for (int i = 0; i < n_mat; ++i) {
        const DeviceTensor & W = *Ws[i];
        int dt = -1;
        if (W.dtype == DType::Q3_K && W.is_soa)      dt = 0;
        else if (W.dtype == DType::Q4_K)             dt = 1;
        else if (W.dtype == DType::Q5_K)             dt = 2;
        else if (W.dtype == DType::Q6_K && W.is_soa) dt = 3;
        else return false;
        desc.dt[i] = dt;
        desc.p0[i] = W.ptr;
        desc.p1[i] = W.aux_ptr;
        desc.p2[i] = W.scales_ptr;
        desc.p3[i] = W.d_ptr;
        desc.y[i]  = Ys[i];
        desc.m[i]  = static_cast<int>(W.ne[1]);
        tot += desc.m[i];
        desc.row_end[i] = tot;
    }
    desc.n_mat = n_mat;

    mw_quantize_act_f16_kernel<<<(tot_b + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(d_x, x_f16, tot_b);

    constexpr int wpb = 8;
    if (use_t2) {
        mw_gemv_multi_h2_kernel<true><<<(tot + wpb - 1) / wpb, dim3(32, wpb), smem, stream>>>(desc, x_f16, K);
    } else {
        mw_gemv_multi_h2_kernel<false><<<dim3((tot + wpb - 1) / wpb, T), dim3(32, wpb), smem, stream>>>(desc, x_f16, K);
    }
    return true;
}

void launch_moe_gate_up_q3_k_soa_only(
    const DeviceTensor & gate_exps, const DeviceTensor & up_exps,
    const void * d_xq8, const int32_t * d_top_ids, float * d_act,
    int T, int k_used, float clamp_exp, cudaStream_t stream) {
    const int K  = static_cast<int>(gate_exps.ne[0]);
    const int ff = static_cast<int>(gate_exps.ne[1]);
    constexpr int wpb = 8;
    const size_t smem = static_cast<size_t>(K >> 5) * sizeof(BlockQ8_1_Act);
    mw_moe_gate_up_q3_k_soa_only_kernel<<<dim3((ff + wpb - 1) / wpb, k_used, T), dim3(32, wpb), smem, stream>>>(
        static_cast<const int4 *>(gate_exps.ptr), static_cast<const int4 *>(gate_exps.aux_ptr),
        static_cast<const uint32_t *>(gate_exps.scales_ptr), static_cast<const uint16_t *>(gate_exps.d_ptr),
        static_cast<const int4 *>(up_exps.ptr), static_cast<const int4 *>(up_exps.aux_ptr),
        static_cast<const uint32_t *>(up_exps.scales_ptr), static_cast<const uint16_t *>(up_exps.d_ptr),
        static_cast<const BlockQ8_1_Act *>(d_xq8), d_top_ids, d_act, ff, K, k_used, clamp_exp);
}

void launch_moe_down_q3_k_soa_only(
    const DeviceTensor & down_exps, const void * d_aq8,
    const int32_t * d_top_ids, const float * d_top_w, float * d_out,
    int T, int k_used, cudaStream_t stream) {
    const int ff = static_cast<int>(down_exps.ne[0]);
    const int K  = static_cast<int>(down_exps.ne[1]);
    constexpr int wpb = 8;
    const size_t smem = static_cast<size_t>(ff >> 5) * sizeof(BlockQ8_1_Act);
    mw_moe_down_q3_k_soa_only_kernel<<<dim3((K + wpb - 1) / wpb, T), dim3(32, wpb), smem, stream>>>(
        static_cast<const int4 *>(down_exps.ptr), static_cast<const int4 *>(down_exps.aux_ptr),
        static_cast<const uint32_t *>(down_exps.scales_ptr), static_cast<const uint16_t *>(down_exps.d_ptr),
        static_cast<const BlockQ8_1_Act *>(d_aq8), d_top_ids, d_top_w, d_out, K, ff, k_used);
}

bool try_launch_moe_q3k_q4k_fast(
    cublasHandle_t cublas,
    const DeviceTensor & gate_exps, const DeviceTensor & up_exps, const DeviceTensor & down_exps,
    const DeviceTensor & gate_shexp, const DeviceTensor & up_shexp, const DeviceTensor & down_shexp,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, float clamp_exp, float clamp_shexp, cudaStream_t stream) {

    if (!(gate_exps.dtype == DType::Q3_K && gate_exps.is_soa &&
          up_exps.dtype   == DType::Q3_K && up_exps.is_soa &&
          down_exps.dtype == DType::Q4_K)) {
        return false;
    }

    const int K        = static_cast<int>(gate_exps.ne[0]);
    const int ff       = static_cast<int>(gate_exps.ne[1]);
    const int n_expert = static_cast<int>(gate_exps.ne[2]);
    const int ff_sh    = gate_shexp.valid() ? static_cast<int>(gate_shexp.ne[1]) : 0;
    const int K_blocks = K >> 5;
    const int ff_blks  = ff >> 5;
    const int sh_blks  = ff_sh >> 5;
    const bool has_sh_fast = (ff_sh > 0 &&
                              gate_shexp.dtype == DType::Q3_K && gate_shexp.is_soa &&
                              up_shexp.dtype   == DType::Q3_K && up_shexp.is_soa &&
                              down_shexp.dtype == DType::Q4_K);
    const int sm = device_sm_version(gate_exps.device);
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    auto * base = static_cast<uint8_t *>(d_scratch);

    // A. PREFILL (T >= 16): Flat-worklist BT=8 KT=1 High-Occupancy K-Tiled Grouped MoE!
    if (T >= 16 && k_used <= 16 && n_expert > 0 && n_expert <= 512) {
        constexpr int BT     = 8;
        constexpr int KT_GU  = 1;
        constexpr int KT_DN  = 1;
        const int max_tiles  = (T * k_used + BT - 1) / BT + n_expert;

        const size_t act_b  = align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
        const size_t shg_b  = align256(static_cast<size_t>(T) * std::max(1, ff_sh) * sizeof(float));
        const size_t shu_b  = shg_b;
        const size_t off_b  = align256(static_cast<size_t>(n_expert + 1) * sizeof(int32_t));
        const size_t sp_b   = align256(static_cast<size_t>(T) * k_used * sizeof(int32_t));
        const size_t tl_b   = align256(static_cast<size_t>(max_tiles) * sizeof(MaxwTileDesc));
        const size_t nt_b   = 256;
        const size_t sl_b   = align256(static_cast<size_t>(T) * k_used * K * sizeof(float));
        const size_t meta_b = off_b + sp_b + tl_b + nt_b;

        if (sm == 60) {
            const size_t xf_b = align256(static_cast<size_t>(T) * K_blocks * sizeof(ActF16Block));
            const size_t af_b = align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(ActF16Block));
            const size_t persist_b = meta_b + af_b;
            const size_t peak_ovl  = std::max(shg_b + shu_b + (static_cast<size_t>(4) << 20), std::max(xf_b + act_b, sl_b));
            if (persist_b + peak_ovl <= scratch_bytes) {
                auto * exp_offsets  = reinterpret_cast<int32_t *>(base);
                auto * sorted_pairs = reinterpret_cast<int32_t *>(base + off_b);
                auto * tiles        = reinterpret_cast<MaxwTileDesc *>(base + off_b + sp_b);
                auto * n_tiles_ptr  = reinterpret_cast<int32_t *>(base + off_b + sp_b + tl_b);
                auto * a_f16        = reinterpret_cast<ActF16Block *>(base + meta_b);
                uint8_t * ovl       = base + persist_b;
                const size_t ovl_b  = scratch_bytes - persist_b;

                // Phase 0: Shared expert (overlays sh_g + sh_u + cuBLAS scratch inside ovl)
                if (ff_sh > 0 && gate_shexp.valid() && up_shexp.valid() && down_shexp.valid()) {
                    auto * sh_g = reinterpret_cast<float *>(ovl);
                    auto * sh_u = reinterpret_cast<float *>(ovl + shg_b);
                    void * sh_rem = ovl + shg_b + shu_b;
                    const size_t sh_rem_b = ovl_b - shg_b - shu_b;
                    launch_linear(cublas, gate_shexp, d_x, sh_g, T, sh_rem, sh_rem_b, stream);
                    launch_linear(cublas, up_shexp,   d_x, sh_u, T, sh_rem, sh_rem_b, stream);
                    launch_silu_mul_clamp(sh_g, sh_u, sh_g, static_cast<int64_t>(T) * ff_sh, clamp_shexp, stream);
                    launch_linear(cublas, down_shexp, sh_g, d_out, T, sh_rem, sh_rem_b, stream);
                } else {
                    cudaMemsetAsync(d_out, 0, static_cast<size_t>(T) * K * sizeof(float), stream);
                }

                // Phase 1: Gate+Up (overlays x_f16 + act inside ovl, quantizes act -> persistent a_f16)
                auto * x_f16 = reinterpret_cast<ActF16Block *>(ovl);
                auto * act   = reinterpret_cast<float *>(ovl + xf_b);
                const int xb_tot = T * K_blocks;
                const int q_wpb  = (xb_tot <= 256) ? 1 : 8;
                mw_quantize_act_f16_kernel<<<(xb_tot + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(d_x, x_f16, xb_tot);
                mw_group_moe_tokens_kernel<<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
                    d_top_ids, exp_offsets, sorted_pairs, T * k_used, n_expert, k_used);
                mw_build_tile_worklist_kernel<<<1, 1, 0, stream>>>(exp_offsets, tiles, n_tiles_ptr, n_expert, BT);

                const size_t gu_smem = static_cast<size_t>(KT_GU * 8) * BT * (16 * sizeof(__half2) + sizeof(float));
                mw_moe_gate_up_grouped_q3_k_soa_h2_kernel<BT, KT_GU>
                    <<<dim3((ff + 31) / 32, max_tiles), 128, gu_smem, stream>>>(
                        static_cast<const int4 *>(gate_exps.ptr), static_cast<const int4 *>(gate_exps.aux_ptr),
                        static_cast<const uint32_t *>(gate_exps.scales_ptr), static_cast<const uint16_t *>(gate_exps.d_ptr),
                        static_cast<const int4 *>(up_exps.ptr), static_cast<const int4 *>(up_exps.aux_ptr),
                        static_cast<const uint32_t *>(up_exps.scales_ptr), static_cast<const uint16_t *>(up_exps.d_ptr),
                        x_f16, tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, clamp_exp);

                const int ab_tot = T * k_used * ff_blks;
                mw_quantize_act_f16_kernel<<<(ab_tot + 7) / 8, dim3(32, 8), 0, stream>>>(act, a_f16, ab_tot);

                // Phase 2: Down (overlays down_slots inside ovl, reads persistent a_f16)
                auto * down_slots = reinterpret_cast<float *>(ovl);
                const size_t dn_smem = static_cast<size_t>(KT_DN * 8) * BT * (16 * sizeof(__half2) + 3 * sizeof(float));
                mw_moe_down_grouped_q4_k_h2_kernel<BT, KT_DN>
                    <<<dim3((K + 15) / 16, max_tiles), 128, dn_smem, stream>>>(
                        static_cast<const BlockQ4_K_M *>(down_exps.ptr),
                        a_f16, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);

                mw_reduce_slots_add_kernel<<<dim3(((K >> 2) + 255) / 256, T), 256, 0, stream>>>(
                    down_slots, d_out, K, k_used, true);
                return true;
            }
        } else {
            // sm_61+ (and sm_52 / sm_70+ when Q3_K has no WMMA): flat-worklist BT=8 KT=1 K-tiled dp4a!
            const size_t xq_b = align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
            const size_t aq_b = align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BlockQ8_1_Act));
            const size_t persist_b = meta_b + aq_b;
            const size_t peak_ovl  = std::max(shg_b + shu_b + (static_cast<size_t>(4) << 20), std::max(xq_b + act_b, sl_b));
            if (persist_b + peak_ovl <= scratch_bytes) {
                auto * exp_offsets  = reinterpret_cast<int32_t *>(base);
                auto * sorted_pairs = reinterpret_cast<int32_t *>(base + off_b);
                auto * tiles        = reinterpret_cast<MaxwTileDesc *>(base + off_b + sp_b);
                auto * n_tiles_ptr  = reinterpret_cast<int32_t *>(base + off_b + sp_b + tl_b);
                auto * a_q8         = reinterpret_cast<BlockQ8_1_Act *>(base + meta_b);
                uint8_t * ovl       = base + persist_b;
                const size_t ovl_b  = scratch_bytes - persist_b;

                if (ff_sh > 0 && gate_shexp.valid() && up_shexp.valid() && down_shexp.valid()) {
                    auto * sh_g = reinterpret_cast<float *>(ovl);
                    auto * sh_u = reinterpret_cast<float *>(ovl + shg_b);
                    void * sh_rem = ovl + shg_b + shu_b;
                    const size_t sh_rem_b = ovl_b - shg_b - shu_b;
                    launch_linear(cublas, gate_shexp, d_x, sh_g, T, sh_rem, sh_rem_b, stream);
                    launch_linear(cublas, up_shexp,   d_x, sh_u, T, sh_rem, sh_rem_b, stream);
                    launch_silu_mul_clamp(sh_g, sh_u, sh_g, static_cast<int64_t>(T) * ff_sh, clamp_shexp, stream);
                    launch_linear(cublas, down_shexp, sh_g, d_out, T, sh_rem, sh_rem_b, stream);
                } else {
                    cudaMemsetAsync(d_out, 0, static_cast<size_t>(T) * K * sizeof(float), stream);
                }

                auto * x_q8 = reinterpret_cast<BlockQ8_1_Act *>(ovl);
                auto * act  = reinterpret_cast<float *>(ovl + xq_b);
                launch_quantize_q8_1(d_x, x_q8, K, T, stream);
                mw_group_moe_tokens_kernel<<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
                    d_top_ids, exp_offsets, sorted_pairs, T * k_used, n_expert, k_used);
                mw_build_tile_worklist_kernel<<<1, 1, 0, stream>>>(exp_offsets, tiles, n_tiles_ptr, n_expert, BT);

                const size_t gu_smem = static_cast<size_t>(KT_GU * 8) * BT * (2 * sizeof(int4) + sizeof(float) + 2 * sizeof(int32_t));
                mw_moe_gate_up_grouped_q3_k_soa_dp4a_kernel<BT, KT_GU>
                    <<<dim3((ff + 31) / 32, max_tiles), 128, gu_smem, stream>>>(
                        static_cast<const int4 *>(gate_exps.ptr), static_cast<const int4 *>(gate_exps.aux_ptr),
                        static_cast<const uint32_t *>(gate_exps.scales_ptr), static_cast<const uint16_t *>(gate_exps.d_ptr),
                        static_cast<const int4 *>(up_exps.ptr), static_cast<const int4 *>(up_exps.aux_ptr),
                        static_cast<const uint32_t *>(up_exps.scales_ptr), static_cast<const uint16_t *>(up_exps.d_ptr),
                        x_q8, tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, clamp_exp);

                launch_quantize_q8_1(act, a_q8, k_used * ff, T, stream);

                auto * down_slots = reinterpret_cast<float *>(ovl);
                const size_t dn_smem = static_cast<size_t>(KT_DN * 8) * BT * (2 * sizeof(int4) + sizeof(float) + 2 * sizeof(int32_t));
                mw_moe_down_grouped_q4_k_dp4a_kernel<BT, KT_DN>
                    <<<dim3((K + 15) / 16, max_tiles), 128, dn_smem, stream>>>(
                        static_cast<const BlockQ4_K_M *>(down_exps.ptr),
                        a_q8, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);

                mw_reduce_slots_add_kernel<<<dim3(((K >> 2) + 255) / 256, T), 256, 0, stream>>>(
                    down_slots, d_out, K, k_used, true);
                return true;
            }
        }
        return false;
    }

    // B. DECODE (T < 16): Fused (k_used + 1) Routed + Shared MoE!
    if (T < 16 && has_sh_fast && (ff & 31) == 0 && (ff_sh & 31) == 0) {
        const int n_slots = k_used + 1;
        const int max_ff  = std::max(ff, ff_sh);

        if (sm == 60) {
            size_t off = 0;
            auto * x_f16  = reinterpret_cast<ActF16Block *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * sizeof(ActF16Block));
            auto * ar_f16 = reinterpret_cast<ActF16Block *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(ActF16Block));
            auto * as_f16 = reinterpret_cast<ActF16Block *>(base + off); off += align256(static_cast<size_t>(T) * sh_blks * sizeof(ActF16Block));
            auto * slots  = reinterpret_cast<float *>      (base + off); off += align256(static_cast<size_t>(T) * n_slots * K * sizeof(float));
            if (off <= scratch_bytes) {
                const int xb_tot = T * K_blocks;
                const int q_wpb  = (xb_tot <= 256) ? 2 : 8;
                mw_quantize_act_f16_kernel<<<(xb_tot + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(d_x, x_f16, xb_tot);

                const size_t gu_smem = static_cast<size_t>(K_blocks) * sizeof(ActF16Block);
                const int n_gu_tiles = k_used * (ff >> 5) + (ff_sh >> 5);
                mw_moe_gate_up_q3_k_soa_h2_dec_kernel
                    <<<dim3(n_gu_tiles, T), dim3(32, 8), gu_smem, stream>>>(
                        static_cast<const int4 *>(gate_exps.ptr), static_cast<const int4 *>(gate_exps.aux_ptr),
                        static_cast<const uint32_t *>(gate_exps.scales_ptr), static_cast<const uint16_t *>(gate_exps.d_ptr),
                        static_cast<const int4 *>(up_exps.ptr), static_cast<const int4 *>(up_exps.aux_ptr),
                        static_cast<const uint32_t *>(up_exps.scales_ptr), static_cast<const uint16_t *>(up_exps.d_ptr),
                        static_cast<const int4 *>(gate_shexp.ptr), static_cast<const int4 *>(gate_shexp.aux_ptr),
                        static_cast<const uint32_t *>(gate_shexp.scales_ptr), static_cast<const uint16_t *>(gate_shexp.d_ptr),
                        static_cast<const int4 *>(up_shexp.ptr), static_cast<const int4 *>(up_shexp.aux_ptr),
                        static_cast<const uint32_t *>(up_shexp.scales_ptr), static_cast<const uint16_t *>(up_shexp.d_ptr),
                        x_f16, d_top_ids, ar_f16, as_f16, K, ff, ff_sh, k_used, clamp_exp);

                constexpr int wpb_dn = 8;
                const size_t dn_smem = static_cast<size_t>(std::max(ff_blks, sh_blks)) * sizeof(ActF16Block);
                mw_moe_down_q4_k_h2_split_dec_kernel
                    <<<dim3((K + wpb_dn * 4 - 1) / (wpb_dn * 4), n_slots, T), dim3(32, wpb_dn), dn_smem, stream>>>(
                        static_cast<const BlockQ4_K_M *>(down_exps.ptr),
                        static_cast<const BlockQ4_K_M *>(down_shexp.ptr),
                        ar_f16, as_f16, d_top_ids, d_top_w, slots, K, ff, ff_sh, k_used);

                mw_reduce_slots_add_kernel<<<dim3(((K >> 2) + 63) / 64, T), 64, 0, stream>>>(
                    slots, d_out, K, n_slots, false);
                return true;
            }
        } else {
            size_t off = 0;
            auto * x_q8  = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
            auto * act_r = reinterpret_cast<float *>        (base + off); off += align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
            auto * act_s = reinterpret_cast<float *>        (base + off); off += align256(static_cast<size_t>(T) * ff_sh * sizeof(float));
            auto * aq8   = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BlockQ8_1_Act));
            auto * sq8   = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * sh_blks * sizeof(BlockQ8_1_Act));
            auto * slots = reinterpret_cast<float *>        (base + off); off += align256(static_cast<size_t>(T) * n_slots * K * sizeof(float));
            if (off <= scratch_bytes) {
                launch_quantize_q8_1(d_x, x_q8, K, T, stream);

                constexpr int wpb_gu = 8;
                const size_t gu_smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
                mw_moe_gate_up_q3_k_soa_dp4a_dec_kernel
                    <<<dim3((max_ff + wpb_gu - 1) / wpb_gu, n_slots, T), dim3(32, wpb_gu), gu_smem, stream>>>(
                        static_cast<const int4 *>(gate_exps.ptr), static_cast<const int4 *>(gate_exps.aux_ptr),
                        static_cast<const uint32_t *>(gate_exps.scales_ptr), static_cast<const uint16_t *>(gate_exps.d_ptr),
                        static_cast<const int4 *>(up_exps.ptr), static_cast<const int4 *>(up_exps.aux_ptr),
                        static_cast<const uint32_t *>(up_exps.scales_ptr), static_cast<const uint16_t *>(up_exps.d_ptr),
                        static_cast<const int4 *>(gate_shexp.ptr), static_cast<const int4 *>(gate_shexp.aux_ptr),
                        static_cast<const uint32_t *>(gate_shexp.scales_ptr), static_cast<const uint16_t *>(gate_shexp.d_ptr),
                        static_cast<const int4 *>(up_shexp.ptr), static_cast<const int4 *>(up_shexp.aux_ptr),
                        static_cast<const uint32_t *>(up_shexp.scales_ptr), static_cast<const uint16_t *>(up_shexp.d_ptr),
                        x_q8, d_top_ids, act_r, act_s, K, ff, ff_sh, k_used, clamp_exp);

                launch_quantize_q8_1(act_r, aq8, k_used * ff, T, stream);
                launch_quantize_q8_1(act_s, sq8, ff_sh, T, stream);

                constexpr int wpb_dn = 8;
                const size_t dn_smem = static_cast<size_t>(std::max(ff_blks, sh_blks)) * sizeof(BlockQ8_1_Act);
                mw_moe_down_q4_k_dp4a_split_dec_kernel
                    <<<dim3((K + wpb_dn * 2 - 1) / (wpb_dn * 2), n_slots, T), dim3(32, wpb_dn), dn_smem, stream>>>(
                        static_cast<const BlockQ4_K_M *>(down_exps.ptr),
                        static_cast<const BlockQ4_K_M *>(down_shexp.ptr),
                        aq8, sq8, d_top_ids, d_top_w, slots, K, ff, ff_sh, k_used);

                mw_reduce_slots_add_kernel<<<dim3(((K >> 2) + 63) / 64, T), 64, 0, stream>>>(
                    slots, d_out, K, n_slots, false);
                return true;
            }
        }
    }

    return false;
}

} // namespace rookery
