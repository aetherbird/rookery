#include "core/device.h"
#include "kernels/kernels.h"
#include "kernels/q8_1_act.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace rookery {

namespace {

struct alignas(16) BmActF16Block {
    __half2 x[16];  // 32 normalized elements: x_raw[0..31] / amax
    float   d;      // amax
    float   s0;     // exact unnormalized sum(x_raw[0..15])
    float   s1;     // exact unnormalized sum(x_raw[16..31])
    float   s_all;  // s0 + s1
};
static_assert(sizeof(BmActF16Block) == 80, "BmActF16Block must be 80 bytes");

struct BlockQ4_K_BM {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ4_K_BM) == 144, "BlockQ4_K_BM size");

struct BmTileDesc {
    int32_t eid;
    int32_t start;
    int32_t count;
};

__device__ __forceinline__ float bm_warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ float bm_half_bits_to_f32(uint16_t bits) {
    __half h;
    *reinterpret_cast<uint16_t *>(&h) = bits;
    return __half2float(h);
}

__device__ __forceinline__ __half2 bm_hfma2(__half2 a, __half2 b, __half2 c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 600 || __CUDA_ARCH__ == 620 || __CUDA_ARCH__ >= 700)
    return __hfma2(a, b, c);
#else
    float2 fa = __half22float2(a), fb = __half22float2(b), fc = __half22float2(c);
    return __floats2half2_rn(fa.x * fb.x + fc.x, fa.y * fb.y + fc.y);
#endif
}

__device__ __forceinline__ __half2 bm_hadd2(__half2 a, __half2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hadd2(a, b);
#else
    float2 fa = __half22float2(a), fb = __half22float2(b);
    return __floats2half2_rn(fa.x + fb.x, fa.y + fb.y);
#endif
}

__device__ __forceinline__ __half2 bm_hmul2(__half2 a, __half2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hmul2(a, b);
#else
    float2 fa = __half22float2(a), fb = __half22float2(b);
    return __floats2half2_rn(fa.x * fb.x, fa.y * fb.y);
#endif
}

__device__ __forceinline__ __half2 bm_u8x2_to_h2_lo(uint32_t u) {
    uint32_t p = __byte_perm(u, 0x00006400u, 0x5150);
    return *reinterpret_cast<const __half2 *>(&p);
}

__device__ __forceinline__ __half2 bm_u8x2_to_h2_hi(uint32_t u) {
    uint32_t p = __byte_perm(u, 0x00006400u, 0x5352);
    return *reinterpret_cast<const __half2 *>(&p);
}

__device__ __forceinline__ void bm_unpack_q4k_sc(
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

// ============================================================================
// 1. FP16->FP32 Copy + Batch Q4_K / Q6_K SoA Dequant for cuBLAS Prefill
// ============================================================================
__global__ void bm_f16_to_f32_vec_kernel(
    const __half * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t i2 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t i  = i2 * 2;
    if (i + 1 < n) {
        const __half2 h = reinterpret_cast<const __half2 *>(src)[i2];
        const float2  f = __half22float2(h);
        reinterpret_cast<float2 *>(dst)[i2] = f;
    } else if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

template <typename OutT>
__global__ void bm_dequant_q6_k_soa_batch_kernel(
    const uint8_t *  __restrict__ ql,
    const uint8_t *  __restrict__ qh,
    const int8_t *   __restrict__ scales,
    const uint16_t * __restrict__ d_arr,
    OutT *           __restrict__ dst,
    int64_t n_blocks) {

    const int64_t bi = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (bi >= n_blocks) return;
    const int l  = threadIdx.x;
    const int is = l >> 4;
    const float dw = bm_half_bits_to_f32(__ldg(&d_arr[bi]));
    const int8_t * sc_b = scales + bi * 16;

    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t ql0 = __ldg(&ql[bi * 128 + h * 64 + l]);
        const uint32_t ql1 = __ldg(&ql[bi * 128 + h * 64 + 32 + l]);
        const uint32_t qhv = __ldg(&qh[bi * 64  + h * 32 + l]);

        const int q1 = static_cast<int>((ql0 & 0xFu) | ((qhv & 0x03u) << 4)) - 32;
        const int q2 = static_cast<int>((ql1 & 0xFu) | ((qhv & 0x0Cu) << 2)) - 32;
        const int q3 = static_cast<int>((ql0 >> 4)   |  (qhv & 0x30u))       - 32;
        const int q4 = static_cast<int>((ql1 >> 4)   | ((qhv >> 2) & 0x30u)) - 32;

        const float sc0 = dw * static_cast<float>(__ldg(&sc_b[h * 8 + is + 0]));
        const float sc2 = dw * static_cast<float>(__ldg(&sc_b[h * 8 + is + 2]));
        const float sc4 = dw * static_cast<float>(__ldg(&sc_b[h * 8 + is + 4]));
        const float sc6 = dw * static_cast<float>(__ldg(&sc_b[h * 8 + is + 6]));

        OutT * out = dst + bi * 256 + h * 128;
        if constexpr (sizeof(OutT) == 2) {
            out[0  + l] = __float2half_rn(sc0 * static_cast<float>(q1));
            out[32 + l] = __float2half_rn(sc2 * static_cast<float>(q2));
            out[64 + l] = __float2half_rn(sc4 * static_cast<float>(q3));
            out[96 + l] = __float2half_rn(sc6 * static_cast<float>(q4));
        } else {
            out[0  + l] = sc0 * static_cast<float>(q1);
            out[32 + l] = sc2 * static_cast<float>(q2);
            out[64 + l] = sc4 * static_cast<float>(q3);
            out[96 + l] = sc6 * static_cast<float>(q4);
        }
    }
}

// 1B. sm_61 Native dp4a Dense Linear Prefill (Bank-Conflict-Free BT=2*BT_TH, RM=2)
template <int BT_TH>
__global__ __launch_bounds__(256, 3) void bm_linear_q4_k_dp4a_kernel(
    const BlockQ4_K_BM *  __restrict__ W,
    const BlockQ8_1_Act * __restrict__ x_q8,
    float *               __restrict__ y,
    int M, int K, int T) {

    constexpr int BT = 2 * BT_TH;
    const int t0     = blockIdx.y * BT;
    const int n_tile = (t0 + BT < T) ? BT : (T - t0);
    if (n_tile <= 0) return;

    extern __shared__ uint8_t smem_raw[];
    auto * s_xq    = reinterpret_cast<int4 *>(smem_raw);
    auto * s_xd    = reinterpret_cast<float *>(s_xq + BT * 16);
    auto * s_xs0_f = s_xd + BT * 8;
    auto * s_xs1_f = s_xs0_f + BT * 8;

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int r_loc  = ltid >> 3;
    const int sub    = ltid & 7;
    const int j      = sub >> 1;
    const int is     = sub & 1;
    const int t_base = t_half * BT_TH;

    const int row0 = blockIdx.x * 32 + r_loc * 2;
    const int row1 = row0 + 1;
    const bool v0  = (row0 < M);
    const bool v1  = (row1 < M);

    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const BlockQ4_K_BM * w_row0 = v0 ? (W + static_cast<size_t>(row0) * nb) : nullptr;
    const BlockQ4_K_BM * w_row1 = v1 ? (W + static_cast<size_t>(row1) * nb) : nullptr;

    float acc0[BT_TH], acc1[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { acc0[it] = 0.0f; acc1[it] = 0.0f; }

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_i4 = BT * 16;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 1;
            const int bk = (idx >> 1) & 7;
            const int tj = idx >> 4;
            const int dst = tj * 16 + (bk & 1) * 8 + (bk >> 1) * 2 + c;
            if (tj < n_tile) {
                s_xq[dst] = reinterpret_cast<const int4 *>(
                    x_q8[static_cast<size_t>(t0 + tj) * K_blocks + bi * 8 + bk].qs)[c];
            } else {
                s_xq[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            const int dst = tj * 8 + (bk & 1) * 4 + (bk >> 1);
            if (tj < n_tile) {
                const BlockQ8_1_Act & xb = x_q8[static_cast<size_t>(t0 + tj) * K_blocks + bi * 8 + bk];
                s_xd[dst]    = xb.d;
                s_xs0_f[dst] = xb.d * static_cast<float>(xb.s0);
                s_xs1_f[dst] = xb.d * static_cast<float>(xb.s1);
            } else {
                s_xd[dst]    = 0.0f;
                s_xs0_f[dst] = 0.0f;
                s_xs1_f[dst] = 0.0f;
            }
        }
        __syncthreads();

        if (v0) {
            const float * s_xs_cur = is ? s_xs1_f : s_xs0_f;
            float a_d1 = 0.0f, a_mn1 = 0.0f, a_d2 = 0.0f, a_mn2 = 0.0f;
            int a_q1x = 0, a_q2x = 0, a_q1y = 0, a_q2y = 0, a_q1z = 0, a_q2z = 0, a_q1w = 0, a_q2w = 0;
            {
                const BlockQ4_K_BM & b = w_row0[bi];
                const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
                const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
                const float d    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
                const float dmin = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
                float sc1, m1, sc2, m2;
                bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
                bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
                a_d1 = d * sc1; a_mn1 = dmin * m1; a_d2 = d * sc2; a_mn2 = dmin * m2;
                a_q1x = static_cast<int>(static_cast<uint32_t>(ql_v.x) & 0x0F0F0F0Fu);
                a_q2x = static_cast<int>((static_cast<uint32_t>(ql_v.x) >> 4) & 0x0F0F0F0Fu);
                a_q1y = static_cast<int>(static_cast<uint32_t>(ql_v.y) & 0x0F0F0F0Fu);
                a_q2y = static_cast<int>((static_cast<uint32_t>(ql_v.y) >> 4) & 0x0F0F0F0Fu);
                a_q1z = static_cast<int>(static_cast<uint32_t>(ql_v.z) & 0x0F0F0F0Fu);
                a_q2z = static_cast<int>((static_cast<uint32_t>(ql_v.z) >> 4) & 0x0F0F0F0Fu);
                a_q1w = static_cast<int>(static_cast<uint32_t>(ql_v.w) & 0x0F0F0F0Fu);
                a_q2w = static_cast<int>((static_cast<uint32_t>(ql_v.w) >> 4) & 0x0F0F0F0Fu);
            }
            float b_d1 = 0.0f, b_mn1 = 0.0f, b_d2 = 0.0f, b_mn2 = 0.0f;
            int b_q1x = 0, b_q2x = 0, b_q1y = 0, b_q2y = 0, b_q1z = 0, b_q2z = 0, b_q1w = 0, b_q2w = 0;
            if (v1) {
                const BlockQ4_K_BM & b = w_row1[bi];
                const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
                const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
                const float d    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
                const float dmin = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
                float sc1, m1, sc2, m2;
                bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
                bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
                b_d1 = d * sc1; b_mn1 = dmin * m1; b_d2 = d * sc2; b_mn2 = dmin * m2;
                b_q1x = static_cast<int>(static_cast<uint32_t>(ql_v.x) & 0x0F0F0F0Fu);
                b_q2x = static_cast<int>((static_cast<uint32_t>(ql_v.x) >> 4) & 0x0F0F0F0Fu);
                b_q1y = static_cast<int>(static_cast<uint32_t>(ql_v.y) & 0x0F0F0F0Fu);
                b_q2y = static_cast<int>((static_cast<uint32_t>(ql_v.y) >> 4) & 0x0F0F0F0Fu);
                b_q1z = static_cast<int>(static_cast<uint32_t>(ql_v.z) & 0x0F0F0F0Fu);
                b_q2z = static_cast<int>((static_cast<uint32_t>(ql_v.z) >> 4) & 0x0F0F0F0Fu);
                b_q1w = static_cast<int>(static_cast<uint32_t>(ql_v.w) & 0x0F0F0F0Fu);
                b_q2w = static_cast<int>((static_cast<uint32_t>(ql_v.w) >> 4) & 0x0F0F0F0Fu);
            }

            #pragma unroll
            for (int it = 0; it < BT_TH; ++it) {
                const int tj = t_base + it;
                const int4 x1 = s_xq[tj * 16 + sub];
                const int4 x2 = s_xq[tj * 16 + 8 + sub];
                const float d0v = s_xd[tj * 8 + j];
                const float d1v = s_xd[tj * 8 + 4 + j];
                const float s0f = s_xs_cur[tj * 8 + j];
                const float s1f = s_xs_cur[tj * 8 + 4 + j];

                const int s0a = dp4a_s32(a_q1x, x1.x, dp4a_s32(a_q1y, x1.y, dp4a_s32(a_q1z, x1.z, dp4a_s32(a_q1w, x1.w, 0))));
                const int s1a = dp4a_s32(a_q2x, x2.x, dp4a_s32(a_q2y, x2.y, dp4a_s32(a_q2z, x2.z, dp4a_s32(a_q2w, x2.w, 0))));
                acc0[it] = fmaf(d0v * a_d1, static_cast<float>(s0a), fmaf(-a_mn1, s0f, acc0[it]));
                acc0[it] = fmaf(d1v * a_d2, static_cast<float>(s1a), fmaf(-a_mn2, s1f, acc0[it]));

                if (v1) {
                    const int s0b = dp4a_s32(b_q1x, x1.x, dp4a_s32(b_q1y, x1.y, dp4a_s32(b_q1z, x1.z, dp4a_s32(b_q1w, x1.w, 0))));
                    const int s1b = dp4a_s32(b_q2x, x2.x, dp4a_s32(b_q2y, x2.y, dp4a_s32(b_q2z, x2.z, dp4a_s32(b_q2w, x2.w, 0))));
                    acc1[it] = fmaf(d0v * b_d1, static_cast<float>(s0b), fmaf(-b_mn1, s0f, acc1[it]));
                    acc1[it] = fmaf(d1v * b_d2, static_cast<float>(s1b), fmaf(-b_mn2, s1f, acc1[it]));
                }
            }
        }
        __syncthreads();
    }

    if (v0) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float v = acc0[it];
            v += __shfl_xor_sync(0xffffffffu, v, 4, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 2, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 8);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                y[static_cast<size_t>(t0 + tj) * M + row0] = v;
            }
        }
    }
    if (v1) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float v = acc1[it];
            v += __shfl_xor_sync(0xffffffffu, v, 4, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 2, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 8);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                y[static_cast<size_t>(t0 + tj) * M + row1] = v;
            }
        }
    }
}

template <int BT_TH>
__global__ __launch_bounds__(256, 3) void bm_linear_q6_k_soa_dp4a_kernel(
    const int4 *          __restrict__ W_ql,
    const int4 *          __restrict__ W_qh,
    const int8_t *        __restrict__ W_sc,
    const uint16_t *      __restrict__ W_d,
    const BlockQ8_1_Act * __restrict__ x_q8,
    float *               __restrict__ y,
    int M, int K, int T) {

    constexpr int BT = 2 * BT_TH;
    const int t0     = blockIdx.y * BT;
    const int n_tile = (t0 + BT < T) ? BT : (T - t0);
    if (n_tile <= 0) return;

    extern __shared__ uint8_t smem_raw[];
    auto * s_xq    = reinterpret_cast<int4 *>(smem_raw);
    auto * s_xd    = reinterpret_cast<float *>(s_xq + BT * 16);
    auto * s_xs0_f = s_xd + BT * 8;
    auto * s_xs1_f = s_xs0_f + BT * 8;

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int r_loc  = ltid >> 2;
    const int sub    = ltid & 3;
    const int h      = sub >> 1;
    const int is     = sub & 1;
    const int t_base = t_half * BT_TH;

    const int row0 = blockIdx.x * 64 + r_loc * 2;
    const int row1 = row0 + 1;
    const bool v0  = (row0 < M);
    const bool v1  = (row1 < M);

    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const size_t rblk0 = static_cast<size_t>(row0) * nb;
    const size_t rblk1 = static_cast<size_t>(row1) * nb;

    float acc0[BT_TH], acc1[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { acc0[it] = 0.0f; acc1[it] = 0.0f; }

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_i4 = BT * 16;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 1;
            const int bk = (idx >> 1) & 7;
            const int tj = idx >> 4;
            const int dst = tj * 16 + (bk & 3) * 4 + (bk >> 2) * 2 + c;
            if (tj < n_tile) {
                s_xq[dst] = reinterpret_cast<const int4 *>(
                    x_q8[static_cast<size_t>(t0 + tj) * K_blocks + bi * 8 + bk].qs)[c];
            } else {
                s_xq[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            const int dst = tj * 8 + (bk & 3) * 2 + (bk >> 2);
            if (tj < n_tile) {
                const BlockQ8_1_Act & xb = x_q8[static_cast<size_t>(t0 + tj) * K_blocks + bi * 8 + bk];
                s_xd[dst]    = xb.d;
                s_xs0_f[dst] = xb.d * static_cast<float>(32 * xb.s0);
                s_xs1_f[dst] = xb.d * static_cast<float>(32 * xb.s1);
            } else {
                s_xd[dst]    = 0.0f;
                s_xs0_f[dst] = 0.0f;
                s_xs1_f[dst] = 0.0f;
            }
        }
        __syncthreads();

        if (v0) {
            const float * s_xs_cur = is ? s_xs1_f : s_xs0_f;
            const int64_t b0 = static_cast<int64_t>(rblk0) + bi;
            const int4 qh0_v  = __ldg(&W_qh[b0 * 4 + sub]);
            const int4 ql00_v = __ldg(&W_ql[b0 * 8 + h * 4 + is]);
            const int4 ql01_v = __ldg(&W_ql[b0 * 8 + h * 4 + 2 + is]);
            const int8_t * sc0p = W_sc + b0 * 16 + h * 8 + is;
            const float dw0 = bm_half_bits_to_f32(__ldg(&W_d[b0]));
            const float sc0s[4] = {
                dw0 * static_cast<float>(__ldg(&sc0p[0])), dw0 * static_cast<float>(__ldg(&sc0p[2])),
                dw0 * static_cast<float>(__ldg(&sc0p[4])), dw0 * static_cast<float>(__ldg(&sc0p[6]))
            };
            const uint32_t a_ql0[4] = { static_cast<uint32_t>(ql00_v.x), static_cast<uint32_t>(ql00_v.y), static_cast<uint32_t>(ql00_v.z), static_cast<uint32_t>(ql00_v.w) };
            const uint32_t a_ql1[4] = { static_cast<uint32_t>(ql01_v.x), static_cast<uint32_t>(ql01_v.y), static_cast<uint32_t>(ql01_v.z), static_cast<uint32_t>(ql01_v.w) };
            const uint32_t a_qh[4]  = { static_cast<uint32_t>(qh0_v.x),  static_cast<uint32_t>(qh0_v.y),  static_cast<uint32_t>(qh0_v.z),  static_cast<uint32_t>(qh0_v.w)  };

            float sc1s[4] = {0, 0, 0, 0};
            uint32_t b_ql0[4] = {0, 0, 0, 0}, b_ql1[4] = {0, 0, 0, 0}, b_qh[4] = {0, 0, 0, 0};
            if (v1) {
                const int64_t b1 = static_cast<int64_t>(rblk1) + bi;
                const int4 qh1_v  = __ldg(&W_qh[b1 * 4 + sub]);
                const int4 ql10_v = __ldg(&W_ql[b1 * 8 + h * 4 + is]);
                const int4 ql11_v = __ldg(&W_ql[b1 * 8 + h * 4 + 2 + is]);
                const int8_t * sc1p = W_sc + b1 * 16 + h * 8 + is;
                const float dw1 = bm_half_bits_to_f32(__ldg(&W_d[b1]));
                sc1s[0] = dw1 * static_cast<float>(__ldg(&sc1p[0]));
                sc1s[1] = dw1 * static_cast<float>(__ldg(&sc1p[2]));
                sc1s[2] = dw1 * static_cast<float>(__ldg(&sc1p[4]));
                sc1s[3] = dw1 * static_cast<float>(__ldg(&sc1p[6]));
                b_ql0[0] = static_cast<uint32_t>(ql10_v.x); b_ql0[1] = static_cast<uint32_t>(ql10_v.y); b_ql0[2] = static_cast<uint32_t>(ql10_v.z); b_ql0[3] = static_cast<uint32_t>(ql10_v.w);
                b_ql1[0] = static_cast<uint32_t>(ql11_v.x); b_ql1[1] = static_cast<uint32_t>(ql11_v.y); b_ql1[2] = static_cast<uint32_t>(ql11_v.z); b_ql1[3] = static_cast<uint32_t>(ql11_v.w);
                b_qh[0]  = static_cast<uint32_t>(qh1_v.x);  b_qh[1]  = static_cast<uint32_t>(qh1_v.y);  b_qh[2]  = static_cast<uint32_t>(qh1_v.z);  b_qh[3]  = static_cast<uint32_t>(qh1_v.w);
            }

            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                int qa[4], qb[4];
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    uint32_t ua, ub;
                    if (q == 0)      { ua = (a_ql0[c] & 0x0F0F0F0Fu) | ((a_qh[c] & 0x03030303u) << 4); ub = (b_ql0[c] & 0x0F0F0F0Fu) | ((b_qh[c] & 0x03030303u) << 4); }
                    else if (q == 1) { ua = (a_ql1[c] & 0x0F0F0F0Fu) | ((a_qh[c] & 0x0C0C0C0Cu) << 2); ub = (b_ql1[c] & 0x0F0F0F0Fu) | ((b_qh[c] & 0x0C0C0C0Cu) << 2); }
                    else if (q == 2) { ua = ((a_ql0[c] >> 4) & 0x0F0F0F0Fu) | (a_qh[c] & 0x30303030u); ub = ((b_ql0[c] >> 4) & 0x0F0F0F0Fu) | (b_qh[c] & 0x30303030u); }
                    else             { ua = ((a_ql1[c] >> 4) & 0x0F0F0F0Fu) | ((a_qh[c] >> 2) & 0x30303030u); ub = ((b_ql1[c] >> 4) & 0x0F0F0F0Fu) | ((b_qh[c] >> 2) & 0x30303030u); }
                    qa[c] = static_cast<int>(ua);
                    qb[c] = static_cast<int>(ub);
                }
                const float sca = sc0s[q], scb = sc1s[q];
                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 x4 = s_xq[tj * 16 + q * 4 + sub];
                    const float dv = s_xd[tj * 8 + q * 2 + h];
                    const float sf = s_xs_cur[tj * 8 + q * 2 + h];
                    const int sa = dp4a_s32(qa[0], x4.x, dp4a_s32(qa[1], x4.y, dp4a_s32(qa[2], x4.z, dp4a_s32(qa[3], x4.w, 0))));
                    acc0[it] = fmaf(sca * dv, static_cast<float>(sa), fmaf(-sca, sf, acc0[it]));
                    if (v1) {
                        const int sb = dp4a_s32(qb[0], x4.x, dp4a_s32(qb[1], x4.y, dp4a_s32(qb[2], x4.z, dp4a_s32(qb[3], x4.w, 0))));
                        acc1[it] = fmaf(scb * dv, static_cast<float>(sb), fmaf(-scb, sf, acc1[it]));
                    }
                }
            }
        }
        __syncthreads();
    }

    if (v0) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float v = acc0[it];
            v += __shfl_xor_sync(0xffffffffu, v, 2, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 4);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                y[static_cast<size_t>(t0 + tj) * M + row0] = v;
            }
        }
    }
    if (v1) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float v = acc1[it];
            v += __shfl_xor_sync(0xffffffffu, v, 2, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 4);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                y[static_cast<size_t>(t0 + tj) * M + row1] = v;
            }
        }
    }
}

// ============================================================================
// 2. Activation Quantizer & MoE Grouping / Worklist / Slot Reduction Helpers
// ============================================================================
__global__ void bm_quantize_act_f16_kernel(
    const float * __restrict__ src, BmActF16Block * __restrict__ dst, int total_blocks) {
    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= total_blocks) return;
    const int lane = threadIdx.x;
    const float v = src[static_cast<size_t>(b) * 32 + lane];
    const float amax = __shfl_sync(0xffffffffu, bm_warp_max(fabsf(v)), 0);
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

    if ((lane & 1) == 0) {
        dst[b].x[lane >> 1] = __halves2half2(__float2half_rn(vn), __float2half_rn(vn_hi));
    }
    if (lane == 0) {
        dst[b].d     = amax;
        dst[b].s0    = s0;
        dst[b].s1    = s1;
        dst[b].s_all = s0 + s1;
    }
}

__global__ void bm_group_moe_tokens_kernel(
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

__global__ void bm_build_tile_worklist_kernel(
    const int32_t * __restrict__ exp_offsets,
    BmTileDesc *    __restrict__ tiles,
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

__global__ void bm_reduce_slots_add_kernel(
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
// 3. PREFILL GROUPED MoE (T >= 16) — CONFLICT-FREE BT=2*BT_TH, FUSED GU & RM=2 DOWN
// ============================================================================

// 3A. sm_60 Native half2 Q4_K Fused Gate+Up+SiLU Grouped Prefill (256 thr/CTA, 16 rows x BT)
template <int BT_TH>
__global__ __launch_bounds__(256, 3) void bm_moe_gate_up_grouped_q4_k_h2_kernel(
    const BlockQ4_K_BM *  __restrict__ g_exps,
    const BlockQ4_K_BM *  __restrict__ u_exps,
    const BmActF16Block * __restrict__ x_f16,
    const BmTileDesc *    __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ act,
    int ff, int K, int k_used, float clamp_limit) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const BmTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT = 2 * BT_TH;
    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    auto * s_xh2 = reinterpret_cast<int4 *>(smem_raw);
    auto * s_xd  = reinterpret_cast<float *>(s_xh2 + BT * 32);
    auto * s_xs0 = s_xd  + BT * 8;
    auto * s_xs1 = s_xs0 + BT * 8;

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int row    = blockIdx.x * 16 + (ltid >> 3);
    const int sub    = ltid & 7;
    const int j      = sub >> 1;
    const int is     = sub & 1;
    const int t_base = t_half * BT_TH;

    if (tid < n_tile) s_pairs[tid] = sorted_pairs[start + tid];
    __syncthreads();

    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const BlockQ4_K_BM * g_row = g_exps + (static_cast<size_t>(eid) * ff + row) * nb;
    const BlockQ4_K_BM * u_row = u_exps + (static_cast<size_t>(eid) * ff + row) * nb;
    const __half2 neg1024 = __floats2half2_rn(-1024.0f, -1024.0f);

    float ag[BT_TH], au[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { ag[it] = 0.0f; au[it] = 0.0f; }

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_i4 = BT * 32;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 3;
            const int bk = (idx >> 2) & 7;
            const int tj = idx >> 5;
            const int dst = tj * 32 + (bk & 1) * 16 + (c & 1) * 8 + (bk >> 1) * 2 + (c >> 1);
            if (tj < n_tile) {
                const int tok = s_pairs[tj] >> 4;
                s_xh2[dst] = reinterpret_cast<const int4 *>(
                    x_f16[static_cast<size_t>(tok) * K_blocks + bi * 8 + bk].x)[c];
            } else {
                s_xh2[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            const int dst = tj * 8 + (bk & 1) * 4 + (bk >> 1);
            if (tj < n_tile) {
                const int tok = s_pairs[tj] >> 4;
                const BmActF16Block & xb = x_f16[static_cast<size_t>(tok) * K_blocks + bi * 8 + bk];
                s_xd[dst]  = xb.d;
                s_xs0[dst] = xb.s0;
                s_xs1[dst] = xb.s1;
            } else {
                s_xd[dst]  = 0.0f;
                s_xs0[dst] = 0.0f;
                s_xs1[dst] = 0.0f;
            }
        }
        __syncthreads();

        if (row < ff) {
            const float * s_xs_cur = is ? s_xs1 : s_xs0;

            const BlockQ4_K_BM & gb = g_row[bi];
            const int4 ghdr  = __ldg(reinterpret_cast<const int4 *>(&gb));
            const int4 gql_v = __ldg(&reinterpret_cast<const int4 *>(gb.qs)[sub]);
            const float gd    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(ghdr.x) & 0xFFFFu));
            const float gdmin = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(ghdr.x) >> 16));
            float gsc1, gm1, gsc2, gm2;
            bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(ghdr.y), static_cast<uint32_t>(ghdr.z), static_cast<uint32_t>(ghdr.w), gsc1, gm1);
            bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(ghdr.y), static_cast<uint32_t>(ghdr.z), static_cast<uint32_t>(ghdr.w), gsc2, gm2);
            const float gd1 = gd * gsc1, gmn1 = gdmin * gm1;
            const float gd2 = gd * gsc2, gmn2 = gdmin * gm2;

            const BlockQ4_K_BM & ub = u_row[bi];
            const int4 uhdr  = __ldg(reinterpret_cast<const int4 *>(&ub));
            const int4 uql_v = __ldg(&reinterpret_cast<const int4 *>(ub.qs)[sub]);
            const float ud    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(uhdr.x) & 0xFFFFu));
            const float udmin = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(uhdr.x) >> 16));
            float usc1, um1, usc2, um2;
            bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(uhdr.y), static_cast<uint32_t>(uhdr.z), static_cast<uint32_t>(uhdr.w), usc1, um1);
            bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(uhdr.y), static_cast<uint32_t>(uhdr.z), static_cast<uint32_t>(uhdr.w), usc2, um2);
            const float ud1 = ud * usc1, umn1 = udmin * um1;
            const float ud2 = ud * usc2, umn2 = udmin * um2;

            // Low nibbles (bk = 2*j) — fused Gate + Up share smem loads!
            {
                const uint32_t g0 = static_cast<uint32_t>(gql_v.x) & 0x0F0F0F0Fu, g1 = static_cast<uint32_t>(gql_v.y) & 0x0F0F0F0Fu;
                const uint32_t g2 = static_cast<uint32_t>(gql_v.z) & 0x0F0F0F0Fu, g3 = static_cast<uint32_t>(gql_v.w) & 0x0F0F0F0Fu;
                const __half2 gw0 = bm_hadd2(bm_u8x2_to_h2_lo(g0), neg1024), gw1 = bm_hadd2(bm_u8x2_to_h2_hi(g0), neg1024);
                const __half2 gw2 = bm_hadd2(bm_u8x2_to_h2_lo(g1), neg1024), gw3 = bm_hadd2(bm_u8x2_to_h2_hi(g1), neg1024);
                const __half2 gw4 = bm_hadd2(bm_u8x2_to_h2_lo(g2), neg1024), gw5 = bm_hadd2(bm_u8x2_to_h2_hi(g2), neg1024);
                const __half2 gw6 = bm_hadd2(bm_u8x2_to_h2_lo(g3), neg1024), gw7 = bm_hadd2(bm_u8x2_to_h2_hi(g3), neg1024);

                const uint32_t u0 = static_cast<uint32_t>(uql_v.x) & 0x0F0F0F0Fu, u1 = static_cast<uint32_t>(uql_v.y) & 0x0F0F0F0Fu;
                const uint32_t u2 = static_cast<uint32_t>(uql_v.z) & 0x0F0F0F0Fu, u3 = static_cast<uint32_t>(uql_v.w) & 0x0F0F0F0Fu;
                const __half2 uw0 = bm_hadd2(bm_u8x2_to_h2_lo(u0), neg1024), uw1 = bm_hadd2(bm_u8x2_to_h2_hi(u0), neg1024);
                const __half2 uw2 = bm_hadd2(bm_u8x2_to_h2_lo(u1), neg1024), uw3 = bm_hadd2(bm_u8x2_to_h2_hi(u1), neg1024);
                const __half2 uw4 = bm_hadd2(bm_u8x2_to_h2_lo(u2), neg1024), uw5 = bm_hadd2(bm_u8x2_to_h2_hi(u2), neg1024);
                const __half2 uw6 = bm_hadd2(bm_u8x2_to_h2_lo(u3), neg1024), uw7 = bm_hadd2(bm_u8x2_to_h2_hi(u3), neg1024);

                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_xh2[tj * 32 + 0 + sub];
                    const int4 ab = s_xh2[tj * 32 + 8 + sub];
                    const float dv = s_xd[tj * 8 + j];
                    const float sv = s_xs_cur[tj * 8 + j];

                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    __half2 rg = bm_hmul2(gw0, x0);
                    rg = bm_hfma2(gw1, x1, rg); rg = bm_hfma2(gw2, x2, rg); rg = bm_hfma2(gw3, x3, rg);
                    rg = bm_hfma2(gw4, x4, rg); rg = bm_hfma2(gw5, x5, rg); rg = bm_hfma2(gw6, x6, rg); rg = bm_hfma2(gw7, x7, rg);
                    const float2 fg = __half22float2(rg);
                    ag[it] = fmaf(gd1 * dv, fg.x + fg.y, fmaf(-gmn1, sv, ag[it]));

                    __half2 ru = bm_hmul2(uw0, x0);
                    ru = bm_hfma2(uw1, x1, ru); ru = bm_hfma2(uw2, x2, ru); ru = bm_hfma2(uw3, x3, ru);
                    ru = bm_hfma2(uw4, x4, ru); ru = bm_hfma2(uw5, x5, ru); ru = bm_hfma2(uw6, x6, ru); ru = bm_hfma2(uw7, x7, ru);
                    const float2 fu = __half22float2(ru);
                    au[it] = fmaf(ud1 * dv, fu.x + fu.y, fmaf(-umn1, sv, au[it]));
                }
            }
            // High nibbles (bk = 2*j + 1) — fused Gate + Up share smem loads!
            {
                const uint32_t g0 = (static_cast<uint32_t>(gql_v.x) >> 4) & 0x0F0F0F0Fu, g1 = (static_cast<uint32_t>(gql_v.y) >> 4) & 0x0F0F0F0Fu;
                const uint32_t g2 = (static_cast<uint32_t>(gql_v.z) >> 4) & 0x0F0F0F0Fu, g3 = (static_cast<uint32_t>(gql_v.w) >> 4) & 0x0F0F0F0Fu;
                const __half2 gw0 = bm_hadd2(bm_u8x2_to_h2_lo(g0), neg1024), gw1 = bm_hadd2(bm_u8x2_to_h2_hi(g0), neg1024);
                const __half2 gw2 = bm_hadd2(bm_u8x2_to_h2_lo(g1), neg1024), gw3 = bm_hadd2(bm_u8x2_to_h2_hi(g1), neg1024);
                const __half2 gw4 = bm_hadd2(bm_u8x2_to_h2_lo(g2), neg1024), gw5 = bm_hadd2(bm_u8x2_to_h2_hi(g2), neg1024);
                const __half2 gw6 = bm_hadd2(bm_u8x2_to_h2_lo(g3), neg1024), gw7 = bm_hadd2(bm_u8x2_to_h2_hi(g3), neg1024);

                const uint32_t u0 = (static_cast<uint32_t>(uql_v.x) >> 4) & 0x0F0F0F0Fu, u1 = (static_cast<uint32_t>(uql_v.y) >> 4) & 0x0F0F0F0Fu;
                const uint32_t u2 = (static_cast<uint32_t>(uql_v.z) >> 4) & 0x0F0F0F0Fu, u3 = (static_cast<uint32_t>(uql_v.w) >> 4) & 0x0F0F0F0Fu;
                const __half2 uw0 = bm_hadd2(bm_u8x2_to_h2_lo(u0), neg1024), uw1 = bm_hadd2(bm_u8x2_to_h2_hi(u0), neg1024);
                const __half2 uw2 = bm_hadd2(bm_u8x2_to_h2_lo(u1), neg1024), uw3 = bm_hadd2(bm_u8x2_to_h2_hi(u1), neg1024);
                const __half2 uw4 = bm_hadd2(bm_u8x2_to_h2_lo(u2), neg1024), uw5 = bm_hadd2(bm_u8x2_to_h2_hi(u2), neg1024);
                const __half2 uw6 = bm_hadd2(bm_u8x2_to_h2_lo(u3), neg1024), uw7 = bm_hadd2(bm_u8x2_to_h2_hi(u3), neg1024);

                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_xh2[tj * 32 + 16 + sub];
                    const int4 ab = s_xh2[tj * 32 + 24 + sub];
                    const float dv = s_xd[tj * 8 + 4 + j];
                    const float sv = s_xs_cur[tj * 8 + 4 + j];

                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    __half2 rg = bm_hmul2(gw0, x0);
                    rg = bm_hfma2(gw1, x1, rg); rg = bm_hfma2(gw2, x2, rg); rg = bm_hfma2(gw3, x3, rg);
                    rg = bm_hfma2(gw4, x4, rg); rg = bm_hfma2(gw5, x5, rg); rg = bm_hfma2(gw6, x6, rg); rg = bm_hfma2(gw7, x7, rg);
                    const float2 fg = __half22float2(rg);
                    ag[it] = fmaf(gd2 * dv, fg.x + fg.y, fmaf(-gmn2, sv, ag[it]));

                    __half2 ru = bm_hmul2(uw0, x0);
                    ru = bm_hfma2(uw1, x1, ru); ru = bm_hfma2(uw2, x2, ru); ru = bm_hfma2(uw3, x3, ru);
                    ru = bm_hfma2(uw4, x4, ru); ru = bm_hfma2(uw5, x5, ru); ru = bm_hfma2(uw6, x6, ru); ru = bm_hfma2(uw7, x7, ru);
                    const float2 fu = __half22float2(ru);
                    au[it] = fmaf(ud2 * dv, fu.x + fu.y, fmaf(-umn2, sv, au[it]));
                }
            }
        }
        __syncthreads();
    }

    if (row < ff) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float gv = ag[it], uv = au[it];
            gv += __shfl_xor_sync(0xffffffffu, gv, 4, 8);
            uv += __shfl_xor_sync(0xffffffffu, uv, 4, 8);
            gv += __shfl_xor_sync(0xffffffffu, gv, 2, 8);
            uv += __shfl_xor_sync(0xffffffffu, uv, 2, 8);
            gv += __shfl_xor_sync(0xffffffffu, gv, 1, 8);
            uv += __shfl_xor_sync(0xffffffffu, uv, 1, 8);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                if (clamp_limit > 0.0f) {
                    gv = fminf(fmaxf(gv, -clamp_limit), clamp_limit);
                    uv = fminf(fmaxf(uv, -clamp_limit), clamp_limit);
                }
                const int pair = s_pairs[tj];
                act[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * ff + row] =
                    (gv / (1.0f + __expf(-gv))) * uv;
            }
        }
    }
}

// 3B. sm_61/sm_52 dp4a Q4_K Fused Gate+Up+SiLU Grouped Prefill (256 thr/CTA, 16 rows x BT)
template <int BT_TH>
__global__ __launch_bounds__(256, 3) void bm_moe_gate_up_grouped_q4_k_dp4a_kernel(
    const BlockQ4_K_BM *  __restrict__ g_exps,
    const BlockQ4_K_BM *  __restrict__ u_exps,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const BmTileDesc *    __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ act,
    int ff, int K, int k_used, float clamp_limit) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const BmTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT = 2 * BT_TH;
    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    auto * s_xq    = reinterpret_cast<int4 *>(smem_raw);
    auto * s_xd    = reinterpret_cast<float *>(s_xq + BT * 16);
    auto * s_xs0_f = s_xd + BT * 8;
    auto * s_xs1_f = s_xs0_f + BT * 8;

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int row    = blockIdx.x * 16 + (ltid >> 3);
    const int sub    = ltid & 7;
    const int j      = sub >> 1;
    const int is     = sub & 1;
    const int t_base = t_half * BT_TH;

    if (tid < n_tile) s_pairs[tid] = sorted_pairs[start + tid];
    __syncthreads();

    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const BlockQ4_K_BM * g_row = g_exps + (static_cast<size_t>(eid) * ff + row) * nb;
    const BlockQ4_K_BM * u_row = u_exps + (static_cast<size_t>(eid) * ff + row) * nb;

    float ag[BT_TH], au[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { ag[it] = 0.0f; au[it] = 0.0f; }

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_i4 = BT * 16;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 1;
            const int bk = (idx >> 1) & 7;
            const int tj = idx >> 4;
            const int dst = tj * 16 + (bk & 1) * 8 + (bk >> 1) * 2 + c;
            if (tj < n_tile) {
                const int tok = s_pairs[tj] >> 4;
                s_xq[dst] = reinterpret_cast<const int4 *>(
                    x_q8[static_cast<size_t>(tok) * K_blocks + bi * 8 + bk].qs)[c];
            } else {
                s_xq[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            const int dst = tj * 8 + (bk & 1) * 4 + (bk >> 1);
            if (tj < n_tile) {
                const int tok = s_pairs[tj] >> 4;
                const BlockQ8_1_Act & xb = x_q8[static_cast<size_t>(tok) * K_blocks + bi * 8 + bk];
                s_xd[dst]    = xb.d;
                s_xs0_f[dst] = xb.d * static_cast<float>(xb.s0);
                s_xs1_f[dst] = xb.d * static_cast<float>(xb.s1);
            } else {
                s_xd[dst]    = 0.0f;
                s_xs0_f[dst] = 0.0f;
                s_xs1_f[dst] = 0.0f;
            }
        }
        __syncthreads();

        if (row < ff) {
            const float * s_xs_cur = is ? s_xs1_f : s_xs0_f;

            const BlockQ4_K_BM & gb = g_row[bi];
            const int4 ghdr  = __ldg(reinterpret_cast<const int4 *>(&gb));
            const int4 gql_v = __ldg(&reinterpret_cast<const int4 *>(gb.qs)[sub]);
            const float gd    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(ghdr.x) & 0xFFFFu));
            const float gdmin = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(ghdr.x) >> 16));
            float gsc1, gm1, gsc2, gm2;
            bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(ghdr.y), static_cast<uint32_t>(ghdr.z), static_cast<uint32_t>(ghdr.w), gsc1, gm1);
            bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(ghdr.y), static_cast<uint32_t>(ghdr.z), static_cast<uint32_t>(ghdr.w), gsc2, gm2);
            const float gd1 = gd * gsc1, gmn1 = gdmin * gm1;
            const float gd2 = gd * gsc2, gmn2 = gdmin * gm2;
            const int g_q1x = static_cast<int>(static_cast<uint32_t>(gql_v.x) & 0x0F0F0F0Fu);
            const int g_q2x = static_cast<int>((static_cast<uint32_t>(gql_v.x) >> 4) & 0x0F0F0F0Fu);
            const int g_q1y = static_cast<int>(static_cast<uint32_t>(gql_v.y) & 0x0F0F0F0Fu);
            const int g_q2y = static_cast<int>((static_cast<uint32_t>(gql_v.y) >> 4) & 0x0F0F0F0Fu);
            const int g_q1z = static_cast<int>(static_cast<uint32_t>(gql_v.z) & 0x0F0F0F0Fu);
            const int g_q2z = static_cast<int>((static_cast<uint32_t>(gql_v.z) >> 4) & 0x0F0F0F0Fu);
            const int g_q1w = static_cast<int>(static_cast<uint32_t>(gql_v.w) & 0x0F0F0F0Fu);
            const int g_q2w = static_cast<int>((static_cast<uint32_t>(gql_v.w) >> 4) & 0x0F0F0F0Fu);

            const BlockQ4_K_BM & ub = u_row[bi];
            const int4 uhdr  = __ldg(reinterpret_cast<const int4 *>(&ub));
            const int4 uql_v = __ldg(&reinterpret_cast<const int4 *>(ub.qs)[sub]);
            const float ud    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(uhdr.x) & 0xFFFFu));
            const float udmin = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(uhdr.x) >> 16));
            float usc1, um1, usc2, um2;
            bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(uhdr.y), static_cast<uint32_t>(uhdr.z), static_cast<uint32_t>(uhdr.w), usc1, um1);
            bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(uhdr.y), static_cast<uint32_t>(uhdr.z), static_cast<uint32_t>(uhdr.w), usc2, um2);
            const float ud1 = ud * usc1, umn1 = udmin * um1;
            const float ud2 = ud * usc2, umn2 = udmin * um2;
            const int u_q1x = static_cast<int>(static_cast<uint32_t>(uql_v.x) & 0x0F0F0F0Fu);
            const int u_q2x = static_cast<int>((static_cast<uint32_t>(uql_v.x) >> 4) & 0x0F0F0F0Fu);
            const int u_q1y = static_cast<int>(static_cast<uint32_t>(uql_v.y) & 0x0F0F0F0Fu);
            const int u_q2y = static_cast<int>((static_cast<uint32_t>(uql_v.y) >> 4) & 0x0F0F0F0Fu);
            const int u_q1z = static_cast<int>(static_cast<uint32_t>(uql_v.z) & 0x0F0F0F0Fu);
            const int u_q2z = static_cast<int>((static_cast<uint32_t>(uql_v.z) >> 4) & 0x0F0F0F0Fu);
            const int u_q1w = static_cast<int>(static_cast<uint32_t>(uql_v.w) & 0x0F0F0F0Fu);
            const int u_q2w = static_cast<int>((static_cast<uint32_t>(uql_v.w) >> 4) & 0x0F0F0F0Fu);

            #pragma unroll
            for (int it = 0; it < BT_TH; ++it) {
                const int tj = t_base + it;
                const int4 x1 = s_xq[tj * 16 + sub];
                const int4 x2 = s_xq[tj * 16 + 8 + sub];
                const float d0v = s_xd[tj * 8 + j];
                const float d1v = s_xd[tj * 8 + 4 + j];
                const float s0f = s_xs_cur[tj * 8 + j];
                const float s1f = s_xs_cur[tj * 8 + 4 + j];

                const int gs1 = dp4a_s32(g_q1x, x1.x, dp4a_s32(g_q1y, x1.y, dp4a_s32(g_q1z, x1.z, dp4a_s32(g_q1w, x1.w, 0))));
                const int gs2 = dp4a_s32(g_q2x, x2.x, dp4a_s32(g_q2y, x2.y, dp4a_s32(g_q2z, x2.z, dp4a_s32(g_q2w, x2.w, 0))));
                ag[it] = fmaf(d0v * gd1, static_cast<float>(gs1), fmaf(-gmn1, s0f, ag[it]));
                ag[it] = fmaf(d1v * gd2, static_cast<float>(gs2), fmaf(-gmn2, s1f, ag[it]));

                const int us1 = dp4a_s32(u_q1x, x1.x, dp4a_s32(u_q1y, x1.y, dp4a_s32(u_q1z, x1.z, dp4a_s32(u_q1w, x1.w, 0))));
                const int us2 = dp4a_s32(u_q2x, x2.x, dp4a_s32(u_q2y, x2.y, dp4a_s32(u_q2z, x2.z, dp4a_s32(u_q2w, x2.w, 0))));
                au[it] = fmaf(d0v * ud1, static_cast<float>(us1), fmaf(-umn1, s0f, au[it]));
                au[it] = fmaf(d1v * ud2, static_cast<float>(us2), fmaf(-umn2, s1f, au[it]));
            }
        }
        __syncthreads();
    }

    if (row < ff) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float gv = ag[it], uv = au[it];
            gv += __shfl_xor_sync(0xffffffffu, gv, 4, 8);
            uv += __shfl_xor_sync(0xffffffffu, uv, 4, 8);
            gv += __shfl_xor_sync(0xffffffffu, gv, 2, 8);
            uv += __shfl_xor_sync(0xffffffffu, uv, 2, 8);
            gv += __shfl_xor_sync(0xffffffffu, gv, 1, 8);
            uv += __shfl_xor_sync(0xffffffffu, uv, 1, 8);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                if (clamp_limit > 0.0f) {
                    gv = fminf(fmaxf(gv, -clamp_limit), clamp_limit);
                    uv = fminf(fmaxf(uv, -clamp_limit), clamp_limit);
                }
                const int pair = s_pairs[tj];
                act[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * ff + row] =
                    (gv / (1.0f + __expf(-gv))) * uv;
            }
        }
    }
}

// 3C. sm_60 Native half2 Q4_K RM=2 Down Grouped Prefill (256 thr/CTA, 32 rows x BT)
template <int BT_TH>
__global__ __launch_bounds__(256, 3) void bm_moe_down_grouped_q4_k_h2_kernel(
    const BlockQ4_K_BM *  __restrict__ dn_exps,
    const BmActF16Block * __restrict__ a_f16,
    const float *         __restrict__ top_w,
    const BmTileDesc *    __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const BmTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT = 2 * BT_TH;
    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    __shared__ float   s_tw[BT];
    auto * s_ah2 = reinterpret_cast<int4 *>(smem_raw);
    auto * s_ad  = reinterpret_cast<float *>(s_ah2 + BT * 32);
    auto * s_as0 = s_ad  + BT * 8;
    auto * s_as1 = s_as0 + BT * 8;

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int r_loc  = ltid >> 3;
    const int sub    = ltid & 7;
    const int j      = sub >> 1;
    const int is     = sub & 1;
    const int t_base = t_half * BT_TH;

    const int row0 = blockIdx.x * 32 + r_loc * 2;
    const int row1 = row0 + 1;
    const bool v0  = (row0 < K);
    const bool v1  = (row1 < K);

    if (tid < n_tile) {
        const int pair = sorted_pairs[start + tid];
        s_pairs[tid] = pair;
        s_tw[tid] = top_w[static_cast<size_t>(pair >> 4) * k_used + (pair & 15)];
    }
    __syncthreads();

    const int ff_blks = ff >> 5;
    const int nb      = ff >> 8;
    const BlockQ4_K_BM * w_row0 = v0 ? (dn_exps + (static_cast<size_t>(eid) * K + row0) * nb) : nullptr;
    const BlockQ4_K_BM * w_row1 = v1 ? (dn_exps + (static_cast<size_t>(eid) * K + row1) * nb) : nullptr;
    const __half2 neg1024 = __floats2half2_rn(-1024.0f, -1024.0f);

    float acc0[BT_TH], acc1[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { acc0[it] = 0.0f; acc1[it] = 0.0f; }

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_i4 = BT * 32;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 3;
            const int bk = (idx >> 2) & 7;
            const int tj = idx >> 5;
            const int dst = tj * 32 + (bk & 1) * 16 + (c & 1) * 8 + (bk >> 1) * 2 + (c >> 1);
            if (tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                s_ah2[dst] = reinterpret_cast<const int4 *>(
                    a_f16[slot_idx * ff_blks + bi * 8 + bk].x)[c];
            } else {
                s_ah2[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            const int dst = tj * 8 + (bk & 1) * 4 + (bk >> 1);
            if (tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                const BmActF16Block & ab = a_f16[slot_idx * ff_blks + bi * 8 + bk];
                s_ad[dst]  = ab.d;
                s_as0[dst] = ab.s0;
                s_as1[dst] = ab.s1;
            } else {
                s_ad[dst]  = 0.0f;
                s_as0[dst] = 0.0f;
                s_as1[dst] = 0.0f;
            }
        }
        __syncthreads();

        if (v0) {
            const float * s_as_cur = is ? s_as1 : s_as0;

            const BlockQ4_K_BM & b0 = w_row0[bi];
            const int4 h0  = __ldg(reinterpret_cast<const int4 *>(&b0));
            const int4 q0v = __ldg(&reinterpret_cast<const int4 *>(b0.qs)[sub]);
            const float d0_b    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(h0.x) & 0xFFFFu));
            const float dmin0_b = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(h0.x) >> 16));
            float a_sc1, a_m1, a_sc2, a_m2;
            bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(h0.y), static_cast<uint32_t>(h0.z), static_cast<uint32_t>(h0.w), a_sc1, a_m1);
            bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(h0.y), static_cast<uint32_t>(h0.z), static_cast<uint32_t>(h0.w), a_sc2, a_m2);
            const float a_d1 = d0_b * a_sc1, a_mn1 = dmin0_b * a_m1;
            const float a_d2 = d0_b * a_sc2, a_mn2 = dmin0_b * a_m2;

            int4 q1v = make_int4(0, 0, 0, 0);
            float b_d1 = 0.0f, b_mn1 = 0.0f, b_d2 = 0.0f, b_mn2 = 0.0f;
            if (v1) {
                const BlockQ4_K_BM & b1 = w_row1[bi];
                const int4 h1 = __ldg(reinterpret_cast<const int4 *>(&b1));
                q1v = __ldg(&reinterpret_cast<const int4 *>(b1.qs)[sub]);
                const float d1_b    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(h1.x) & 0xFFFFu));
                const float dmin1_b = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(h1.x) >> 16));
                float b_sc1, b_m1, b_sc2, b_m2;
                bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(h1.y), static_cast<uint32_t>(h1.z), static_cast<uint32_t>(h1.w), b_sc1, b_m1);
                bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(h1.y), static_cast<uint32_t>(h1.z), static_cast<uint32_t>(h1.w), b_sc2, b_m2);
                b_d1 = d1_b * b_sc1; b_mn1 = dmin1_b * b_m1;
                b_d2 = d1_b * b_sc2; b_mn2 = dmin1_b * b_m2;
            }

            // Low nibbles (bk = 2*j)
            {
                const uint32_t a0 = static_cast<uint32_t>(q0v.x) & 0x0F0F0F0Fu, a1 = static_cast<uint32_t>(q0v.y) & 0x0F0F0F0Fu;
                const uint32_t a2 = static_cast<uint32_t>(q0v.z) & 0x0F0F0F0Fu, a3 = static_cast<uint32_t>(q0v.w) & 0x0F0F0F0Fu;
                const __half2 aw0 = bm_hadd2(bm_u8x2_to_h2_lo(a0), neg1024), aw1 = bm_hadd2(bm_u8x2_to_h2_hi(a0), neg1024);
                const __half2 aw2 = bm_hadd2(bm_u8x2_to_h2_lo(a1), neg1024), aw3 = bm_hadd2(bm_u8x2_to_h2_hi(a1), neg1024);
                const __half2 aw4 = bm_hadd2(bm_u8x2_to_h2_lo(a2), neg1024), aw5 = bm_hadd2(bm_u8x2_to_h2_hi(a2), neg1024);
                const __half2 aw6 = bm_hadd2(bm_u8x2_to_h2_lo(a3), neg1024), aw7 = bm_hadd2(bm_u8x2_to_h2_hi(a3), neg1024);

                const uint32_t b0u = static_cast<uint32_t>(q1v.x) & 0x0F0F0F0Fu, b1u = static_cast<uint32_t>(q1v.y) & 0x0F0F0F0Fu;
                const uint32_t b2u = static_cast<uint32_t>(q1v.z) & 0x0F0F0F0Fu, b3u = static_cast<uint32_t>(q1v.w) & 0x0F0F0F0Fu;
                const __half2 bw0 = bm_hadd2(bm_u8x2_to_h2_lo(b0u), neg1024), bw1 = bm_hadd2(bm_u8x2_to_h2_hi(b0u), neg1024);
                const __half2 bw2 = bm_hadd2(bm_u8x2_to_h2_lo(b1u), neg1024), bw3 = bm_hadd2(bm_u8x2_to_h2_hi(b1u), neg1024);
                const __half2 bw4 = bm_hadd2(bm_u8x2_to_h2_lo(b2u), neg1024), bw5 = bm_hadd2(bm_u8x2_to_h2_hi(b2u), neg1024);
                const __half2 bw6 = bm_hadd2(bm_u8x2_to_h2_lo(b3u), neg1024), bw7 = bm_hadd2(bm_u8x2_to_h2_hi(b3u), neg1024);

                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_ah2[tj * 32 + 0 + sub];
                    const int4 ab = s_ah2[tj * 32 + 8 + sub];
                    const float dv = s_ad[tj * 8 + j];
                    const float sv = s_as_cur[tj * 8 + j];

                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    __half2 r0 = bm_hmul2(aw0, x0);
                    r0 = bm_hfma2(aw1, x1, r0); r0 = bm_hfma2(aw2, x2, r0); r0 = bm_hfma2(aw3, x3, r0);
                    r0 = bm_hfma2(aw4, x4, r0); r0 = bm_hfma2(aw5, x5, r0); r0 = bm_hfma2(aw6, x6, r0); r0 = bm_hfma2(aw7, x7, r0);
                    const float2 f0 = __half22float2(r0);
                    acc0[it] = fmaf(a_d1 * dv, f0.x + f0.y, fmaf(-a_mn1, sv, acc0[it]));

                    if (v1) {
                        __half2 r1 = bm_hmul2(bw0, x0);
                        r1 = bm_hfma2(bw1, x1, r1); r1 = bm_hfma2(bw2, x2, r1); r1 = bm_hfma2(bw3, x3, r1);
                        r1 = bm_hfma2(bw4, x4, r1); r1 = bm_hfma2(bw5, x5, r1); r1 = bm_hfma2(bw6, x6, r1); r1 = bm_hfma2(bw7, x7, r1);
                        const float2 f1 = __half22float2(r1);
                        acc1[it] = fmaf(b_d1 * dv, f1.x + f1.y, fmaf(-b_mn1, sv, acc1[it]));
                    }
                }
            }
            // High nibbles (bk = 2*j + 1)
            {
                const uint32_t a0 = (static_cast<uint32_t>(q0v.x) >> 4) & 0x0F0F0F0Fu, a1 = (static_cast<uint32_t>(q0v.y) >> 4) & 0x0F0F0F0Fu;
                const uint32_t a2 = (static_cast<uint32_t>(q0v.z) >> 4) & 0x0F0F0F0Fu, a3 = (static_cast<uint32_t>(q0v.w) >> 4) & 0x0F0F0F0Fu;
                const __half2 aw0 = bm_hadd2(bm_u8x2_to_h2_lo(a0), neg1024), aw1 = bm_hadd2(bm_u8x2_to_h2_hi(a0), neg1024);
                const __half2 aw2 = bm_hadd2(bm_u8x2_to_h2_lo(a1), neg1024), aw3 = bm_hadd2(bm_u8x2_to_h2_hi(a1), neg1024);
                const __half2 aw4 = bm_hadd2(bm_u8x2_to_h2_lo(a2), neg1024), aw5 = bm_hadd2(bm_u8x2_to_h2_hi(a2), neg1024);
                const __half2 aw6 = bm_hadd2(bm_u8x2_to_h2_lo(a3), neg1024), aw7 = bm_hadd2(bm_u8x2_to_h2_hi(a3), neg1024);

                const uint32_t b0u = (static_cast<uint32_t>(q1v.x) >> 4) & 0x0F0F0F0Fu, b1u = (static_cast<uint32_t>(q1v.y) >> 4) & 0x0F0F0F0Fu;
                const uint32_t b2u = (static_cast<uint32_t>(q1v.z) >> 4) & 0x0F0F0F0Fu, b3u = (static_cast<uint32_t>(q1v.w) >> 4) & 0x0F0F0F0Fu;
                const __half2 bw0 = bm_hadd2(bm_u8x2_to_h2_lo(b0u), neg1024), bw1 = bm_hadd2(bm_u8x2_to_h2_hi(b0u), neg1024);
                const __half2 bw2 = bm_hadd2(bm_u8x2_to_h2_lo(b1u), neg1024), bw3 = bm_hadd2(bm_u8x2_to_h2_hi(b1u), neg1024);
                const __half2 bw4 = bm_hadd2(bm_u8x2_to_h2_lo(b2u), neg1024), bw5 = bm_hadd2(bm_u8x2_to_h2_hi(b2u), neg1024);
                const __half2 bw6 = bm_hadd2(bm_u8x2_to_h2_lo(b3u), neg1024), bw7 = bm_hadd2(bm_u8x2_to_h2_hi(b3u), neg1024);

                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_ah2[tj * 32 + 16 + sub];
                    const int4 ab = s_ah2[tj * 32 + 24 + sub];
                    const float dv = s_ad[tj * 8 + 4 + j];
                    const float sv = s_as_cur[tj * 8 + 4 + j];

                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    __half2 r0 = bm_hmul2(aw0, x0);
                    r0 = bm_hfma2(aw1, x1, r0); r0 = bm_hfma2(aw2, x2, r0); r0 = bm_hfma2(aw3, x3, r0);
                    r0 = bm_hfma2(aw4, x4, r0); r0 = bm_hfma2(aw5, x5, r0); r0 = bm_hfma2(aw6, x6, r0); r0 = bm_hfma2(aw7, x7, r0);
                    const float2 f0 = __half22float2(r0);
                    acc0[it] = fmaf(a_d2 * dv, f0.x + f0.y, fmaf(-a_mn2, sv, acc0[it]));

                    if (v1) {
                        __half2 r1 = bm_hmul2(bw0, x0);
                        r1 = bm_hfma2(bw1, x1, r1); r1 = bm_hfma2(bw2, x2, r1); r1 = bm_hfma2(bw3, x3, r1);
                        r1 = bm_hfma2(bw4, x4, r1); r1 = bm_hfma2(bw5, x5, r1); r1 = bm_hfma2(bw6, x6, r1); r1 = bm_hfma2(bw7, x7, r1);
                        const float2 f1 = __half22float2(r1);
                        acc1[it] = fmaf(b_d2 * dv, f1.x + f1.y, fmaf(-b_mn2, sv, acc1[it]));
                    }
                }
            }
        }
        __syncthreads();
    }

    if (v0) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float v = acc0[it];
            v += __shfl_xor_sync(0xffffffffu, v, 4, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 2, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 8);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row0] = s_tw[tj] * v;
            }
        }
    }
    if (v1) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float v = acc1[it];
            v += __shfl_xor_sync(0xffffffffu, v, 4, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 2, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 8);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row1] = s_tw[tj] * v;
            }
        }
    }
}

// 3D. sm_61/sm_52 dp4a Q4_K RM=2 Down Grouped Prefill (256 thr/CTA, 32 rows x BT)
template <int BT_TH>
__global__ __launch_bounds__(256, 3) void bm_moe_down_grouped_q4_k_dp4a_kernel(
    const BlockQ4_K_BM *  __restrict__ dn_exps,
    const BlockQ8_1_Act * __restrict__ a_q8,
    const float *         __restrict__ top_w,
    const BmTileDesc *    __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const BmTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT = 2 * BT_TH;
    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    __shared__ float   s_tw[BT];
    auto * s_aq    = reinterpret_cast<int4 *>(smem_raw);
    auto * s_ad    = reinterpret_cast<float *>(s_aq + BT * 16);
    auto * s_as0_f = s_ad + BT * 8;
    auto * s_as1_f = s_as0_f + BT * 8;

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int r_loc  = ltid >> 3;
    const int sub    = ltid & 7;
    const int j      = sub >> 1;
    const int is     = sub & 1;
    const int t_base = t_half * BT_TH;

    const int row0 = blockIdx.x * 32 + r_loc * 2;
    const int row1 = row0 + 1;
    const bool v0  = (row0 < K);
    const bool v1  = (row1 < K);

    if (tid < n_tile) {
        const int pair = sorted_pairs[start + tid];
        s_pairs[tid] = pair;
        s_tw[tid] = top_w[static_cast<size_t>(pair >> 4) * k_used + (pair & 15)];
    }
    __syncthreads();

    const int ff_blks = ff >> 5;
    const int nb      = ff >> 8;
    const BlockQ4_K_BM * w_row0 = v0 ? (dn_exps + (static_cast<size_t>(eid) * K + row0) * nb) : nullptr;
    const BlockQ4_K_BM * w_row1 = v1 ? (dn_exps + (static_cast<size_t>(eid) * K + row1) * nb) : nullptr;

    float acc0[BT_TH], acc1[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { acc0[it] = 0.0f; acc1[it] = 0.0f; }

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_i4 = BT * 16;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 1;
            const int bk = (idx >> 1) & 7;
            const int tj = idx >> 4;
            const int dst = tj * 16 + (bk & 1) * 8 + (bk >> 1) * 2 + c;
            if (tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                s_aq[dst] = reinterpret_cast<const int4 *>(
                    a_q8[slot_idx * ff_blks + bi * 8 + bk].qs)[c];
            } else {
                s_aq[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            const int dst = tj * 8 + (bk & 1) * 4 + (bk >> 1);
            if (tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                const BlockQ8_1_Act & ab = a_q8[slot_idx * ff_blks + bi * 8 + bk];
                s_ad[dst]    = ab.d;
                s_as0_f[dst] = ab.d * static_cast<float>(ab.s0);
                s_as1_f[dst] = ab.d * static_cast<float>(ab.s1);
            } else {
                s_ad[dst]    = 0.0f;
                s_as0_f[dst] = 0.0f;
                s_as1_f[dst] = 0.0f;
            }
        }
        __syncthreads();

        if (v0) {
            const float * s_as_cur = is ? s_as1_f : s_as0_f;
            float a_d1 = 0.0f, a_mn1 = 0.0f, a_d2 = 0.0f, a_mn2 = 0.0f;
            int a_q1x = 0, a_q2x = 0, a_q1y = 0, a_q2y = 0, a_q1z = 0, a_q2z = 0, a_q1w = 0, a_q2w = 0;
            {
                const BlockQ4_K_BM & b = w_row0[bi];
                const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
                const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
                const float d    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
                const float dmin = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
                float sc1, m1, sc2, m2;
                bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
                bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
                a_d1 = d * sc1; a_mn1 = dmin * m1; a_d2 = d * sc2; a_mn2 = dmin * m2;
                a_q1x = static_cast<int>(static_cast<uint32_t>(ql_v.x) & 0x0F0F0F0Fu);
                a_q2x = static_cast<int>((static_cast<uint32_t>(ql_v.x) >> 4) & 0x0F0F0F0Fu);
                a_q1y = static_cast<int>(static_cast<uint32_t>(ql_v.y) & 0x0F0F0F0Fu);
                a_q2y = static_cast<int>((static_cast<uint32_t>(ql_v.y) >> 4) & 0x0F0F0F0Fu);
                a_q1z = static_cast<int>(static_cast<uint32_t>(ql_v.z) & 0x0F0F0F0Fu);
                a_q2z = static_cast<int>((static_cast<uint32_t>(ql_v.z) >> 4) & 0x0F0F0F0Fu);
                a_q1w = static_cast<int>(static_cast<uint32_t>(ql_v.w) & 0x0F0F0F0Fu);
                a_q2w = static_cast<int>((static_cast<uint32_t>(ql_v.w) >> 4) & 0x0F0F0F0Fu);
            }
            float b_d1 = 0.0f, b_mn1 = 0.0f, b_d2 = 0.0f, b_mn2 = 0.0f;
            int b_q1x = 0, b_q2x = 0, b_q1y = 0, b_q2y = 0, b_q1z = 0, b_q2z = 0, b_q1w = 0, b_q2w = 0;
            if (v1) {
                const BlockQ4_K_BM & b = w_row1[bi];
                const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
                const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
                const float d    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
                const float dmin = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
                float sc1, m1, sc2, m2;
                bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
                bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
                b_d1 = d * sc1; b_mn1 = dmin * m1; b_d2 = d * sc2; b_mn2 = dmin * m2;
                b_q1x = static_cast<int>(static_cast<uint32_t>(ql_v.x) & 0x0F0F0F0Fu);
                b_q2x = static_cast<int>((static_cast<uint32_t>(ql_v.x) >> 4) & 0x0F0F0F0Fu);
                b_q1y = static_cast<int>(static_cast<uint32_t>(ql_v.y) & 0x0F0F0F0Fu);
                b_q2y = static_cast<int>((static_cast<uint32_t>(ql_v.y) >> 4) & 0x0F0F0F0Fu);
                b_q1z = static_cast<int>(static_cast<uint32_t>(ql_v.z) & 0x0F0F0F0Fu);
                b_q2z = static_cast<int>((static_cast<uint32_t>(ql_v.z) >> 4) & 0x0F0F0F0Fu);
                b_q1w = static_cast<int>(static_cast<uint32_t>(ql_v.w) & 0x0F0F0F0Fu);
                b_q2w = static_cast<int>((static_cast<uint32_t>(ql_v.w) >> 4) & 0x0F0F0F0Fu);
            }

            #pragma unroll
            for (int it = 0; it < BT_TH; ++it) {
                const int tj = t_base + it;
                const int4 x1 = s_aq[tj * 16 + sub];
                const int4 x2 = s_aq[tj * 16 + 8 + sub];
                const float d0v = s_ad[tj * 8 + j];
                const float d1v = s_ad[tj * 8 + 4 + j];
                const float s0f = s_as_cur[tj * 8 + j];
                const float s1f = s_as_cur[tj * 8 + 4 + j];

                const int s0a = dp4a_s32(a_q1x, x1.x, dp4a_s32(a_q1y, x1.y, dp4a_s32(a_q1z, x1.z, dp4a_s32(a_q1w, x1.w, 0))));
                const int s1a = dp4a_s32(a_q2x, x2.x, dp4a_s32(a_q2y, x2.y, dp4a_s32(a_q2z, x2.z, dp4a_s32(a_q2w, x2.w, 0))));
                acc0[it] = fmaf(d0v * a_d1, static_cast<float>(s0a), fmaf(-a_mn1, s0f, acc0[it]));
                acc0[it] = fmaf(d1v * a_d2, static_cast<float>(s1a), fmaf(-a_mn2, s1f, acc0[it]));

                if (v1) {
                    const int s0b = dp4a_s32(b_q1x, x1.x, dp4a_s32(b_q1y, x1.y, dp4a_s32(b_q1z, x1.z, dp4a_s32(b_q1w, x1.w, 0))));
                    const int s1b = dp4a_s32(b_q2x, x2.x, dp4a_s32(b_q2y, x2.y, dp4a_s32(b_q2z, x2.z, dp4a_s32(b_q2w, x2.w, 0))));
                    acc1[it] = fmaf(d0v * b_d1, static_cast<float>(s0b), fmaf(-b_mn1, s0f, acc1[it]));
                    acc1[it] = fmaf(d1v * b_d2, static_cast<float>(s1b), fmaf(-b_mn2, s1f, acc1[it]));
                }
            }
        }
        __syncthreads();
    }

    if (v0) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float v = acc0[it];
            v += __shfl_xor_sync(0xffffffffu, v, 4, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 2, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 8);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row0] = s_tw[tj] * v;
            }
        }
    }
    if (v1) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float v = acc1[it];
            v += __shfl_xor_sync(0xffffffffu, v, 4, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 2, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 8);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row1] = s_tw[tj] * v;
            }
        }
    }
}

// 3E. sm_60 Native half2 Q6_K SoA RM=2 Down Grouped Prefill (256 thr/CTA, 64 rows x BT)
template <int BT_TH>
__global__ __launch_bounds__(256, 3) void bm_moe_down_grouped_q6_k_soa_h2_kernel(
    const int4 *          __restrict__ dn_ql,
    const int4 *          __restrict__ dn_qh,
    const int8_t *        __restrict__ dn_sc,
    const uint16_t *      __restrict__ dn_d,
    const BmActF16Block * __restrict__ a_f16,
    const float *         __restrict__ top_w,
    const BmTileDesc *    __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const BmTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT = 2 * BT_TH;
    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    __shared__ float   s_tw[BT];
    auto * s_ah2 = reinterpret_cast<int4 *>(smem_raw);
    auto * s_ad  = reinterpret_cast<float *>(s_ah2 + BT * 32);

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int r_loc  = ltid >> 2;
    const int sub    = ltid & 3;
    const int h      = sub >> 1;
    const int is     = sub & 1;
    const int t_base = t_half * BT_TH;

    const int row0 = blockIdx.x * 64 + r_loc * 2;
    const int row1 = row0 + 1;
    const bool v0  = (row0 < K);
    const bool v1  = (row1 < K);

    if (tid < n_tile) {
        const int pair = sorted_pairs[start + tid];
        s_pairs[tid] = pair;
        s_tw[tid] = top_w[static_cast<size_t>(pair >> 4) * k_used + (pair & 15)];
    }
    __syncthreads();

    const int ff_blks = ff >> 5;
    const int nb      = ff >> 8;
    const size_t rblk0 = (static_cast<size_t>(eid) * K + row0) * nb;
    const size_t rblk1 = (static_cast<size_t>(eid) * K + row1) * nb;
    const __half2 neg1056 = __floats2half2_rn(-1056.0f, -1056.0f);

    float acc0[BT_TH], acc1[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { acc0[it] = 0.0f; acc1[it] = 0.0f; }

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_i4 = BT * 32;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 3;
            const int bk = (idx >> 2) & 7;
            const int tj = idx >> 5;
            const int dst = tj * 32 + (bk & 3) * 8 + (c & 1) * 4 + (bk >> 2) * 2 + (c >> 1);
            if (tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                s_ah2[dst] = reinterpret_cast<const int4 *>(
                    a_f16[slot_idx * ff_blks + bi * 8 + bk].x)[c];
            } else {
                s_ah2[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            const int dst = tj * 8 + (bk & 3) * 2 + (bk >> 2);
            if (tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                s_ad[dst] = a_f16[slot_idx * ff_blks + bi * 8 + bk].d;
            } else {
                s_ad[dst] = 0.0f;
            }
        }
        __syncthreads();

        if (v0) {
            const int64_t b0 = static_cast<int64_t>(rblk0) + bi;
            const int4 qh0_v  = __ldg(&dn_qh[b0 * 4 + sub]);
            const int4 ql00_v = __ldg(&dn_ql[b0 * 8 + h * 4 + is]);
            const int4 ql01_v = __ldg(&dn_ql[b0 * 8 + h * 4 + 2 + is]);
            const int8_t * sc0p = dn_sc + b0 * 16 + h * 8 + is;
            const float dw0 = bm_half_bits_to_f32(__ldg(&dn_d[b0]));
            const float sc0s[4] = {
                dw0 * static_cast<float>(__ldg(&sc0p[0])), dw0 * static_cast<float>(__ldg(&sc0p[2])),
                dw0 * static_cast<float>(__ldg(&sc0p[4])), dw0 * static_cast<float>(__ldg(&sc0p[6]))
            };
            const uint32_t a_ql0[4] = { static_cast<uint32_t>(ql00_v.x), static_cast<uint32_t>(ql00_v.y), static_cast<uint32_t>(ql00_v.z), static_cast<uint32_t>(ql00_v.w) };
            const uint32_t a_ql1[4] = { static_cast<uint32_t>(ql01_v.x), static_cast<uint32_t>(ql01_v.y), static_cast<uint32_t>(ql01_v.z), static_cast<uint32_t>(ql01_v.w) };
            const uint32_t a_qh[4]  = { static_cast<uint32_t>(qh0_v.x),  static_cast<uint32_t>(qh0_v.y),  static_cast<uint32_t>(qh0_v.z),  static_cast<uint32_t>(qh0_v.w)  };

            float sc1s[4] = {0, 0, 0, 0};
            uint32_t b_ql0[4] = {0, 0, 0, 0}, b_ql1[4] = {0, 0, 0, 0}, b_qh[4] = {0, 0, 0, 0};
            if (v1) {
                const int64_t b1 = static_cast<int64_t>(rblk1) + bi;
                const int4 qh1_v  = __ldg(&dn_qh[b1 * 4 + sub]);
                const int4 ql10_v = __ldg(&dn_ql[b1 * 8 + h * 4 + is]);
                const int4 ql11_v = __ldg(&dn_ql[b1 * 8 + h * 4 + 2 + is]);
                const int8_t * sc1p = dn_sc + b1 * 16 + h * 8 + is;
                const float dw1 = bm_half_bits_to_f32(__ldg(&dn_d[b1]));
                sc1s[0] = dw1 * static_cast<float>(__ldg(&sc1p[0]));
                sc1s[1] = dw1 * static_cast<float>(__ldg(&sc1p[2]));
                sc1s[2] = dw1 * static_cast<float>(__ldg(&sc1p[4]));
                sc1s[3] = dw1 * static_cast<float>(__ldg(&sc1p[6]));
                b_ql0[0] = static_cast<uint32_t>(ql10_v.x); b_ql0[1] = static_cast<uint32_t>(ql10_v.y); b_ql0[2] = static_cast<uint32_t>(ql10_v.z); b_ql0[3] = static_cast<uint32_t>(ql10_v.w);
                b_ql1[0] = static_cast<uint32_t>(ql11_v.x); b_ql1[1] = static_cast<uint32_t>(ql11_v.y); b_ql1[2] = static_cast<uint32_t>(ql11_v.z); b_ql1[3] = static_cast<uint32_t>(ql11_v.w);
                b_qh[0]  = static_cast<uint32_t>(qh1_v.x);  b_qh[1]  = static_cast<uint32_t>(qh1_v.y);  b_qh[2]  = static_cast<uint32_t>(qh1_v.z);  b_qh[3]  = static_cast<uint32_t>(qh1_v.w);
            }

            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                uint32_t ua[4], ub[4];
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    if (q == 0)      { ua[c] = (a_ql0[c] & 0x0F0F0F0Fu) | ((a_qh[c] & 0x03030303u) << 4); ub[c] = (b_ql0[c] & 0x0F0F0F0Fu) | ((b_qh[c] & 0x03030303u) << 4); }
                    else if (q == 1) { ua[c] = (a_ql1[c] & 0x0F0F0F0Fu) | ((a_qh[c] & 0x0C0C0C0Cu) << 2); ub[c] = (b_ql1[c] & 0x0F0F0F0Fu) | ((b_qh[c] & 0x0C0C0C0Cu) << 2); }
                    else if (q == 2) { ua[c] = ((a_ql0[c] >> 4) & 0x0F0F0F0Fu) | (a_qh[c] & 0x30303030u); ub[c] = ((b_ql0[c] >> 4) & 0x0F0F0F0Fu) | (b_qh[c] & 0x30303030u); }
                    else             { ua[c] = ((a_ql1[c] >> 4) & 0x0F0F0F0Fu) | ((a_qh[c] >> 2) & 0x30303030u); ub[c] = ((b_ql1[c] >> 4) & 0x0F0F0F0Fu) | ((b_qh[c] >> 2) & 0x30303030u); }
                }
                const __half2 aw0 = bm_hadd2(bm_u8x2_to_h2_lo(ua[0]), neg1056), aw1 = bm_hadd2(bm_u8x2_to_h2_hi(ua[0]), neg1056);
                const __half2 aw2 = bm_hadd2(bm_u8x2_to_h2_lo(ua[1]), neg1056), aw3 = bm_hadd2(bm_u8x2_to_h2_hi(ua[1]), neg1056);
                const __half2 aw4 = bm_hadd2(bm_u8x2_to_h2_lo(ua[2]), neg1056), aw5 = bm_hadd2(bm_u8x2_to_h2_hi(ua[2]), neg1056);
                const __half2 aw6 = bm_hadd2(bm_u8x2_to_h2_lo(ua[3]), neg1056), aw7 = bm_hadd2(bm_u8x2_to_h2_hi(ua[3]), neg1056);

                const __half2 bw0 = bm_hadd2(bm_u8x2_to_h2_lo(ub[0]), neg1056), bw1 = bm_hadd2(bm_u8x2_to_h2_hi(ub[0]), neg1056);
                const __half2 bw2 = bm_hadd2(bm_u8x2_to_h2_lo(ub[1]), neg1056), bw3 = bm_hadd2(bm_u8x2_to_h2_hi(ub[1]), neg1056);
                const __half2 bw4 = bm_hadd2(bm_u8x2_to_h2_lo(ub[2]), neg1056), bw5 = bm_hadd2(bm_u8x2_to_h2_hi(ub[2]), neg1056);
                const __half2 bw6 = bm_hadd2(bm_u8x2_to_h2_lo(ub[3]), neg1056), bw7 = bm_hadd2(bm_u8x2_to_h2_hi(ub[3]), neg1056);

                const float sca = sc0s[q], scb = sc1s[q];
                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_ah2[tj * 32 + q * 8 + 0 + sub];
                    const int4 ab = s_ah2[tj * 32 + q * 8 + 4 + sub];
                    const float dv = s_ad[tj * 8 + q * 2 + h];

                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    __half2 r0 = bm_hmul2(aw0, x0);
                    r0 = bm_hfma2(aw1, x1, r0); r0 = bm_hfma2(aw2, x2, r0); r0 = bm_hfma2(aw3, x3, r0);
                    r0 = bm_hfma2(aw4, x4, r0); r0 = bm_hfma2(aw5, x5, r0); r0 = bm_hfma2(aw6, x6, r0); r0 = bm_hfma2(aw7, x7, r0);
                    const float2 f0 = __half22float2(r0);
                    acc0[it] = fmaf(sca * dv, f0.x + f0.y, acc0[it]);

                    if (v1) {
                        __half2 r1 = bm_hmul2(bw0, x0);
                        r1 = bm_hfma2(bw1, x1, r1); r1 = bm_hfma2(bw2, x2, r1); r1 = bm_hfma2(bw3, x3, r1);
                        r1 = bm_hfma2(bw4, x4, r1); r1 = bm_hfma2(bw5, x5, r1); r1 = bm_hfma2(bw6, x6, r1); r1 = bm_hfma2(bw7, x7, r1);
                        const float2 f1 = __half22float2(r1);
                        acc1[it] = fmaf(scb * dv, f1.x + f1.y, acc1[it]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (v0) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float v = acc0[it];
            v += __shfl_xor_sync(0xffffffffu, v, 2, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 4);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row0] = s_tw[tj] * v;
            }
        }
    }
    if (v1) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float v = acc1[it];
            v += __shfl_xor_sync(0xffffffffu, v, 2, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 4);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row1] = s_tw[tj] * v;
            }
        }
    }
}

// 3F. sm_61/sm_52 dp4a Q6_K SoA RM=2 Down Grouped Prefill (256 thr/CTA, 64 rows x BT)
template <int BT_TH>
__global__ __launch_bounds__(256, 3) void bm_moe_down_grouped_q6_k_soa_dp4a_kernel(
    const int4 *          __restrict__ dn_ql,
    const int4 *          __restrict__ dn_qh,
    const int8_t *        __restrict__ dn_sc,
    const uint16_t *      __restrict__ dn_d,
    const BlockQ8_1_Act * __restrict__ a_q8,
    const float *         __restrict__ top_w,
    const BmTileDesc *    __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const BmTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT = 2 * BT_TH;
    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    __shared__ float   s_tw[BT];
    auto * s_aq    = reinterpret_cast<int4 *>(smem_raw);
    auto * s_ad    = reinterpret_cast<float *>(s_aq + BT * 16);
    auto * s_as0_f = s_ad + BT * 8;
    auto * s_as1_f = s_as0_f + BT * 8;

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int r_loc  = ltid >> 2;
    const int sub    = ltid & 3;
    const int h      = sub >> 1;
    const int is     = sub & 1;
    const int t_base = t_half * BT_TH;

    const int row0 = blockIdx.x * 64 + r_loc * 2;
    const int row1 = row0 + 1;
    const bool v0  = (row0 < K);
    const bool v1  = (row1 < K);

    if (tid < n_tile) {
        const int pair = sorted_pairs[start + tid];
        s_pairs[tid] = pair;
        s_tw[tid] = top_w[static_cast<size_t>(pair >> 4) * k_used + (pair & 15)];
    }
    __syncthreads();

    const int ff_blks = ff >> 5;
    const int nb      = ff >> 8;
    const size_t rblk0 = (static_cast<size_t>(eid) * K + row0) * nb;
    const size_t rblk1 = (static_cast<size_t>(eid) * K + row1) * nb;

    float acc0[BT_TH], acc1[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { acc0[it] = 0.0f; acc1[it] = 0.0f; }

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_i4 = BT * 16;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 1;
            const int bk = (idx >> 1) & 7;
            const int tj = idx >> 4;
            const int dst = tj * 16 + (bk & 3) * 4 + (bk >> 2) * 2 + c;
            if (tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                s_aq[dst] = reinterpret_cast<const int4 *>(
                    a_q8[slot_idx * ff_blks + bi * 8 + bk].qs)[c];
            } else {
                s_aq[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            const int dst = tj * 8 + (bk & 3) * 2 + (bk >> 2);
            if (tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                const BlockQ8_1_Act & ab = a_q8[slot_idx * ff_blks + bi * 8 + bk];
                s_ad[dst]    = ab.d;
                s_as0_f[dst] = ab.d * static_cast<float>(32 * ab.s0);
                s_as1_f[dst] = ab.d * static_cast<float>(32 * ab.s1);
            } else {
                s_ad[dst]    = 0.0f;
                s_as0_f[dst] = 0.0f;
                s_as1_f[dst] = 0.0f;
            }
        }
        __syncthreads();

        if (v0) {
            const float * s_as_cur = is ? s_as1_f : s_as0_f;
            const int64_t b0 = static_cast<int64_t>(rblk0) + bi;
            const int4 qh0_v  = __ldg(&dn_qh[b0 * 4 + sub]);
            const int4 ql00_v = __ldg(&dn_ql[b0 * 8 + h * 4 + is]);
            const int4 ql01_v = __ldg(&dn_ql[b0 * 8 + h * 4 + 2 + is]);
            const int8_t * sc0p = dn_sc + b0 * 16 + h * 8 + is;
            const float dw0 = bm_half_bits_to_f32(__ldg(&dn_d[b0]));
            const float sc0s[4] = {
                dw0 * static_cast<float>(__ldg(&sc0p[0])), dw0 * static_cast<float>(__ldg(&sc0p[2])),
                dw0 * static_cast<float>(__ldg(&sc0p[4])), dw0 * static_cast<float>(__ldg(&sc0p[6]))
            };
            const uint32_t a_ql0[4] = { static_cast<uint32_t>(ql00_v.x), static_cast<uint32_t>(ql00_v.y), static_cast<uint32_t>(ql00_v.z), static_cast<uint32_t>(ql00_v.w) };
            const uint32_t a_ql1[4] = { static_cast<uint32_t>(ql01_v.x), static_cast<uint32_t>(ql01_v.y), static_cast<uint32_t>(ql01_v.z), static_cast<uint32_t>(ql01_v.w) };
            const uint32_t a_qh[4]  = { static_cast<uint32_t>(qh0_v.x),  static_cast<uint32_t>(qh0_v.y),  static_cast<uint32_t>(qh0_v.z),  static_cast<uint32_t>(qh0_v.w)  };

            float sc1s[4] = {0, 0, 0, 0};
            uint32_t b_ql0[4] = {0, 0, 0, 0}, b_ql1[4] = {0, 0, 0, 0}, b_qh[4] = {0, 0, 0, 0};
            if (v1) {
                const int64_t b1 = static_cast<int64_t>(rblk1) + bi;
                const int4 qh1_v  = __ldg(&dn_qh[b1 * 4 + sub]);
                const int4 ql10_v = __ldg(&dn_ql[b1 * 8 + h * 4 + is]);
                const int4 ql11_v = __ldg(&dn_ql[b1 * 8 + h * 4 + 2 + is]);
                const int8_t * sc1p = dn_sc + b1 * 16 + h * 8 + is;
                const float dw1 = bm_half_bits_to_f32(__ldg(&dn_d[b1]));
                sc1s[0] = dw1 * static_cast<float>(__ldg(&sc1p[0]));
                sc1s[1] = dw1 * static_cast<float>(__ldg(&sc1p[2]));
                sc1s[2] = dw1 * static_cast<float>(__ldg(&sc1p[4]));
                sc1s[3] = dw1 * static_cast<float>(__ldg(&sc1p[6]));
                b_ql0[0] = static_cast<uint32_t>(ql10_v.x); b_ql0[1] = static_cast<uint32_t>(ql10_v.y); b_ql0[2] = static_cast<uint32_t>(ql10_v.z); b_ql0[3] = static_cast<uint32_t>(ql10_v.w);
                b_ql1[0] = static_cast<uint32_t>(ql11_v.x); b_ql1[1] = static_cast<uint32_t>(ql11_v.y); b_ql1[2] = static_cast<uint32_t>(ql11_v.z); b_ql1[3] = static_cast<uint32_t>(ql11_v.w);
                b_qh[0]  = static_cast<uint32_t>(qh1_v.x);  b_qh[1]  = static_cast<uint32_t>(qh1_v.y);  b_qh[2]  = static_cast<uint32_t>(qh1_v.z);  b_qh[3]  = static_cast<uint32_t>(qh1_v.w);
            }

            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                int qa[4], qb[4];
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    uint32_t ua, ub;
                    if (q == 0)      { ua = (a_ql0[c] & 0x0F0F0F0Fu) | ((a_qh[c] & 0x03030303u) << 4); ub = (b_ql0[c] & 0x0F0F0F0Fu) | ((b_qh[c] & 0x03030303u) << 4); }
                    else if (q == 1) { ua = (a_ql1[c] & 0x0F0F0F0Fu) | ((a_qh[c] & 0x0C0C0C0Cu) << 2); ub = (b_ql1[c] & 0x0F0F0F0Fu) | ((b_qh[c] & 0x0C0C0C0Cu) << 2); }
                    else if (q == 2) { ua = ((a_ql0[c] >> 4) & 0x0F0F0F0Fu) | (a_qh[c] & 0x30303030u); ub = ((b_ql0[c] >> 4) & 0x0F0F0F0Fu) | (b_qh[c] & 0x30303030u); }
                    else             { ua = ((a_ql1[c] >> 4) & 0x0F0F0F0Fu) | ((a_qh[c] >> 2) & 0x30303030u); ub = ((b_ql1[c] >> 4) & 0x0F0F0F0Fu) | ((b_qh[c] >> 2) & 0x30303030u); }
                    qa[c] = static_cast<int>(ua);
                    qb[c] = static_cast<int>(ub);
                }
                const float sca = sc0s[q], scb = sc1s[q];
                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 a4 = s_aq[tj * 16 + q * 4 + sub];
                    const float dv = s_ad[tj * 8 + q * 2 + h];
                    const float sf = s_as_cur[tj * 8 + q * 2 + h];
                    const int sa = dp4a_s32(qa[0], a4.x, dp4a_s32(qa[1], a4.y, dp4a_s32(qa[2], a4.z, dp4a_s32(qa[3], a4.w, 0))));
                    acc0[it] = fmaf(sca * dv, static_cast<float>(sa), fmaf(-sca, sf, acc0[it]));
                    if (v1) {
                        const int sb = dp4a_s32(qb[0], a4.x, dp4a_s32(qb[1], a4.y, dp4a_s32(qb[2], a4.z, dp4a_s32(qb[3], a4.w, 0))));
                        acc1[it] = fmaf(scb * dv, static_cast<float>(sb), fmaf(-scb, sf, acc1[it]));
                    }
                }
            }
        }
        __syncthreads();
    }

    if (v0) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float v = acc0[it];
            v += __shfl_xor_sync(0xffffffffu, v, 2, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 4);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row0] = s_tw[tj] * v;
            }
        }
    }
    if (v1) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float v = acc1[it];
            v += __shfl_xor_sync(0xffffffffu, v, 2, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 1, 4);
            const int tj = t_base + it;
            if (sub == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row1] = s_tw[tj] * v;
            }
        }
    }
}

// ============================================================================
// 4. DECODE FUSED (k_used + 1) ROUTED + SHARED MoE (T < 16)
// ============================================================================

// 4A. sm_60 Native half2 Q4_K Gate+Up+SiLU -> Direct ActF16Block Quantize!
__global__ __launch_bounds__(256) void bm_moe_gate_up_q4_k_h2_dec_kernel(
    const BlockQ4_K_BM *  __restrict__ g_exps,
    const BlockQ4_K_BM *  __restrict__ u_exps,
    const BlockQ4_K_BM *  __restrict__ g_shexp,
    const BlockQ4_K_BM *  __restrict__ u_shexp,
    const BmActF16Block * __restrict__ x_f16,
    const int32_t *       __restrict__ top_ids,
    BmActF16Block *       __restrict__ act_r_f16,
    BmActF16Block *       __restrict__ act_s_f16,
    int K, int ff, int ff_sh, int k_used, float clamp_exp, float clamp_shexp) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<BmActF16Block *>(smem_raw);
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

    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BmActF16Block))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(x_f16 + static_cast<size_t>(t) * K_blocks);
    for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_x)[i] = sx_in[i];
    __syncthreads();

    const int lane     = threadIdx.x;
    const int sub_r    = lane >> 3; // 0..3 (4 rows per warp -> 32 rows per 8-warp CTA!)
    const int sub      = lane & 7;  // 0..7
    const int j        = sub >> 1;  // 0..3
    const int is       = sub & 1;   // 0..1
    const int r_in_cta = threadIdx.y * 4 + sub_r;
    const int row      = tile_x * 32 + r_in_cta;

    const BlockQ4_K_BM * g_row;
    const BlockQ4_K_BM * u_row;
    if (!is_sh) {
        const int eid = top_ids[static_cast<size_t>(t) * k_used + slot];
        g_row = g_exps + (static_cast<size_t>(eid) * ff + row) * nb;
        u_row = u_exps + (static_cast<size_t>(eid) * ff + row) * nb;
    } else {
        g_row = g_shexp + static_cast<size_t>(row) * nb;
        u_row = u_shexp + static_cast<size_t>(row) * nb;
    }

    const __half2 neg1024 = __floats2half2_rn(-1024.0f, -1024.0f);
    float ag = 0.0f, au = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const int xb = bi * 8 + 2 * j;
        const int4 * x1_4 = reinterpret_cast<const int4 *>(s_x[xb + 0].x + is * 8);
        const int4 * x2_4 = reinterpret_cast<const int4 *>(s_x[xb + 1].x + is * 8);
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
        const float xd0 = s_x[xb + 0].d, xd1 = s_x[xb + 1].d;
        const float s1  = is ? s_x[xb + 0].s1 : s_x[xb + 0].s0;
        const float s2  = is ? s_x[xb + 1].s1 : s_x[xb + 1].s0;

        auto dot_blk = [&](const BlockQ4_K_BM & b) -> float {
            const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
            const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
            const float d    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
            const float dmin = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
            float sc1, m1, sc2, m2;
            bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
            bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
            const uint32_t qws[4] = {
                static_cast<uint32_t>(ql_v.x), static_cast<uint32_t>(ql_v.y),
                static_cast<uint32_t>(ql_v.z), static_cast<uint32_t>(ql_v.w)
            };
            __half2 h1 = __floats2half2_rn(0.0f, 0.0f), h2 = __floats2half2_rn(0.0f, 0.0f);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint32_t u_lo = qws[c] & 0x0F0F0F0Fu;
                const uint32_t u_hi = (qws[c] >> 4) & 0x0F0F0F0Fu;
                h1 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_lo(u_lo), neg1024), x1[c * 2 + 0], h1);
                h1 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_hi(u_lo), neg1024), x1[c * 2 + 1], h1);
                h2 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_lo(u_hi), neg1024), x2[c * 2 + 0], h2);
                h2 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_hi(u_hi), neg1024), x2[c * 2 + 1], h2);
            }
            const float2 f1 = __half22float2(h1), f2 = __half22float2(h2);
            return (d * sc1 * xd0) * (f1.x + f1.y) - (dmin * m1) * s1
                 + (d * sc2 * xd1) * (f2.x + f2.y) - (dmin * m2) * s2;
        };
        ag += dot_blk(g_row[bi]);
        au += dot_blk(u_row[bi]);
    }

    ag += __shfl_down_sync(0xffffffffu, ag, 4, 8); au += __shfl_down_sync(0xffffffffu, au, 4, 8);
    ag += __shfl_down_sync(0xffffffffu, ag, 2, 8); au += __shfl_down_sync(0xffffffffu, au, 2, 8);
    ag += __shfl_down_sync(0xffffffffu, ag, 1, 8); au += __shfl_down_sync(0xffffffffu, au, 1, 8);

    if (sub == 0) {
        const float clamp_lim = is_sh ? clamp_shexp : clamp_exp;
        if (clamp_lim > 0.0f) {
            ag = fminf(fmaxf(ag, -clamp_lim), clamp_lim);
            au = fminf(fmaxf(au, -clamp_lim), clamp_lim);
        }
        s_act[r_in_cta] = (ag / (1.0f + __expf(-ag))) * au;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        const float v = s_act[lane];
        const float amax = __shfl_sync(0xffffffffu, bm_warp_max(fabsf(v)), 0);
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

        BmActF16Block * dst = is_sh
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

// 4B. sm_61/sm_52 dp4a Q4_K Gate+Up+SiLU -> Direct BlockQ8_1_Act Quantize!
__global__ __launch_bounds__(256) void bm_moe_gate_up_q4_k_dp4a_dec_kernel(
    const BlockQ4_K_BM *  __restrict__ g_exps,
    const BlockQ4_K_BM *  __restrict__ u_exps,
    const BlockQ4_K_BM *  __restrict__ g_shexp,
    const BlockQ4_K_BM *  __restrict__ u_shexp,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ top_ids,
    BlockQ8_1_Act *       __restrict__ act_r_q8,
    BlockQ8_1_Act *       __restrict__ act_s_q8,
    int K, int ff, int ff_sh, int k_used, float clamp_exp, float clamp_shexp) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<BlockQ8_1_Act *>(smem_raw);
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

    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(t) * K_blocks);
    for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_x)[i] = sx_in[i];
    __syncthreads();

    const int lane     = threadIdx.x;
    const int sub_r    = lane >> 3;
    const int sub      = lane & 7;
    const int j        = sub >> 1;
    const int is       = sub & 1;
    const int r_in_cta = threadIdx.y * 4 + sub_r;
    const int row      = tile_x * 32 + r_in_cta;

    const BlockQ4_K_BM * g_row;
    const BlockQ4_K_BM * u_row;
    if (!is_sh) {
        const int eid = top_ids[static_cast<size_t>(t) * k_used + slot];
        g_row = g_exps + (static_cast<size_t>(eid) * ff + row) * nb;
        u_row = u_exps + (static_cast<size_t>(eid) * ff + row) * nb;
    } else {
        g_row = g_shexp + static_cast<size_t>(row) * nb;
        u_row = u_shexp + static_cast<size_t>(row) * nb;
    }

    float ag = 0.0f, au = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        const int xb = bi * 8 + 2 * j;
        const int4 x1 = reinterpret_cast<const int4 *>(s_x[xb + 0].qs)[is];
        const int4 x2 = reinterpret_cast<const int4 *>(s_x[xb + 1].qs)[is];
        const int sx1 = is ? s_x[xb + 0].s1 : s_x[xb + 0].s0;
        const int sx2 = is ? s_x[xb + 1].s1 : s_x[xb + 1].s0;
        const float xd0 = s_x[xb + 0].d, xd1 = s_x[xb + 1].d;

        auto dot_blk = [&](const BlockQ4_K_BM & b) -> float {
            const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
            const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
            const float d    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
            const float dmin = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
            float sc1, m1, sc2, m2;
            bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
            bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
            const int q1x = static_cast<int>(static_cast<uint32_t>(ql_v.x) & 0x0F0F0F0Fu);
            const int q2x = static_cast<int>((static_cast<uint32_t>(ql_v.x) >> 4) & 0x0F0F0F0Fu);
            const int q1y = static_cast<int>(static_cast<uint32_t>(ql_v.y) & 0x0F0F0F0Fu);
            const int q2y = static_cast<int>((static_cast<uint32_t>(ql_v.y) >> 4) & 0x0F0F0F0Fu);
            const int q1z = static_cast<int>(static_cast<uint32_t>(ql_v.z) & 0x0F0F0F0Fu);
            const int q2z = static_cast<int>((static_cast<uint32_t>(ql_v.z) >> 4) & 0x0F0F0F0Fu);
            const int q1w = static_cast<int>(static_cast<uint32_t>(ql_v.w) & 0x0F0F0F0Fu);
            const int q2w = static_cast<int>((static_cast<uint32_t>(ql_v.w) >> 4) & 0x0F0F0F0Fu);

            const int sum1 = dp4a_s32(q1x, x1.x, dp4a_s32(q1y, x1.y, dp4a_s32(q1z, x1.z, dp4a_s32(q1w, x1.w, 0))));
            const int sum2 = dp4a_s32(q2x, x2.x, dp4a_s32(q2y, x2.y, dp4a_s32(q2z, x2.z, dp4a_s32(q2w, x2.w, 0))));
            return xd0 * (d * sc1 * static_cast<float>(sum1) - dmin * m1 * static_cast<float>(sx1))
                 + xd1 * (d * sc2 * static_cast<float>(sum2) - dmin * m2 * static_cast<float>(sx2));
        };
        ag += dot_blk(g_row[bi]);
        au += dot_blk(u_row[bi]);
    }

    ag += __shfl_down_sync(0xffffffffu, ag, 4, 8); au += __shfl_down_sync(0xffffffffu, au, 4, 8);
    ag += __shfl_down_sync(0xffffffffu, ag, 2, 8); au += __shfl_down_sync(0xffffffffu, au, 2, 8);
    ag += __shfl_down_sync(0xffffffffu, ag, 1, 8); au += __shfl_down_sync(0xffffffffu, au, 1, 8);

    if (sub == 0) {
        const float clamp_lim = is_sh ? clamp_shexp : clamp_exp;
        if (clamp_lim > 0.0f) {
            ag = fminf(fmaxf(ag, -clamp_lim), clamp_lim);
            au = fminf(fmaxf(au, -clamp_lim), clamp_lim);
        }
        s_act[r_in_cta] = (ag / (1.0f + expf(-ag))) * au;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        const float v = s_act[lane];
        const float amax = __shfl_sync(0xffffffffu, bm_warp_max(fabsf(v)), 0);
        const float d = amax / 127.0f;
        const float inv = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        const int q = __float2int_rn(v * inv);

        int hs = q;
        hs += __shfl_xor_sync(0xffffffffu, hs, 1);
        hs += __shfl_xor_sync(0xffffffffu, hs, 2);
        hs += __shfl_xor_sync(0xffffffffu, hs, 4);
        hs += __shfl_xor_sync(0xffffffffu, hs, 8);
        const int s0 = __shfl_sync(0xffffffffu, hs, 0);
        const int s1 = __shfl_sync(0xffffffffu, hs, 16);

        BlockQ8_1_Act * dst = is_sh
            ? (act_s_q8 + static_cast<size_t>(t) * (ff_sh >> 5) + tile_x)
            : (act_r_q8 + (static_cast<size_t>(t) * k_used + slot) * (ff >> 5) + tile_x);
        dst->qs[lane] = static_cast<int8_t>(q);
        if (lane == 0) {
            dst->d     = d;
            dst->s0    = s0;
            dst->s1    = s1;
            dst->s_all = d * static_cast<float>(s0 + s1);
        }
    }
}

// 4C. sm_60 Native half2 Q4_K Split-Slot Down Decode
__global__ void bm_moe_down_q4_k_h2_split_dec_kernel(
    const BlockQ4_K_BM *  __restrict__ dn_exps,
    const BlockQ4_K_BM *  __restrict__ dn_shexp,
    const BmActF16Block * __restrict__ act_r_f16,
    const BmActF16Block * __restrict__ act_s_f16,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ down_slots,
    int K, int ff, int ff_sh, int k_used) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_a = reinterpret_cast<BmActF16Block *>(smem_raw);
    const int slot   = blockIdx.y;
    const int t      = blockIdx.z;
    const bool is_sh = (slot == k_used);
    const int cur_ff = is_sh ? ff_sh : ff;
    const int cur_fb = cur_ff >> 5;
    const int nb     = cur_ff >> 8;

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const BmActF16Block * a_src = is_sh
        ? (act_s_f16 + static_cast<size_t>(t) * cur_fb)
        : (act_r_f16 + (static_cast<size_t>(t) * k_used + slot) * cur_fb);
    const int n_i4 = (cur_fb * static_cast<int>(sizeof(BmActF16Block))) / 16;
    for (int i = tid; i < n_i4; i += nthr) {
        reinterpret_cast<int4 *>(s_a)[i] = reinterpret_cast<const int4 *>(a_src)[i];
    }
    __syncthreads();

    const int lane   = threadIdx.x;
    const int sub_r  = lane >> 4;
    const int sub_l  = lane & 15;
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
        const BlockQ4_K_BM * w_row = is_sh
            ? (dn_shexp + static_cast<size_t>(row) * nb)
            : (dn_exps  + (static_cast<size_t>(eid) * K + row) * nb);

        float acc = 0.0f;
        for (int bi = b_init; bi < nb; bi += 2) {
            const BlockQ4_K_BM & b = w_row[bi];
            const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
            const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
            const float d    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
            const float dmin = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
            float sc1, m1, sc2, m2;
            bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
            bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
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
                h1 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_lo(u_lo), neg1024), x1[c * 2 + 0], h1);
                h1 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_hi(u_lo), neg1024), x1[c * 2 + 1], h1);
                h2 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_lo(u_hi), neg1024), x2[c * 2 + 0], h2);
                h2 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_hi(u_hi), neg1024), x2[c * 2 + 1], h2);
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

// 4D. sm_61/sm_52 dp4a Q4_K Split-Slot Down Decode
__global__ void bm_moe_down_q4_k_dp4a_split_dec_kernel(
    const BlockQ4_K_BM *  __restrict__ dn_exps,
    const BlockQ4_K_BM *  __restrict__ dn_shexp,
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

    const int lane   = threadIdx.x;
    const int sub_r  = lane >> 4;
    const int sub_l  = lane & 15;
    const int row    = (blockIdx.x * blockDim.y + threadIdx.y) * 2 + sub_r;
    if (row >= K) return;

    const int b_init = sub_l >> 3;
    const int sub    = sub_l & 7;
    const int j      = sub >> 1;
    const int is     = sub & 1;

    const BlockQ4_K_BM * w_row;
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
        const BlockQ4_K_BM & b = w_row[bi];
        const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
        const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);
        const float d    = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
        const float dmin = bm_half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
        float sc1, m1, sc2, m2;
        bm_unpack_q4k_sc(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
        bm_unpack_q4k_sc(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
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

// 4E. sm_60 Native half2 Q6_K SoA Split-Slot Down Decode (4 rows/warp = 32 rows/CTA)
__global__ void bm_moe_down_q6_k_soa_h2_split_dec_kernel(
    const int4 *          __restrict__ dn_ql,  const int4 * __restrict__ dn_qh,
    const int8_t *        __restrict__ dn_sc,  const uint16_t * __restrict__ dn_d,
    const int4 *          __restrict__ sh_ql,  const int4 * __restrict__ sh_qh,
    const int8_t *        __restrict__ sh_sc,  const uint16_t * __restrict__ sh_d,
    const BmActF16Block * __restrict__ act_r_f16,
    const BmActF16Block * __restrict__ act_s_f16,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ down_slots,
    int K, int ff, int ff_sh, int k_used) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_a = reinterpret_cast<BmActF16Block *>(smem_raw);
    const int slot   = blockIdx.y;
    const int t      = blockIdx.z;
    const bool is_sh = (slot == k_used);
    const int cur_ff = is_sh ? ff_sh : ff;
    const int cur_fb = cur_ff >> 5;
    const int nb     = cur_ff >> 8;

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const BmActF16Block * a_src = is_sh
        ? (act_s_f16 + static_cast<size_t>(t) * cur_fb)
        : (act_r_f16 + (static_cast<size_t>(t) * k_used + slot) * cur_fb);
    const int n_i4 = (cur_fb * static_cast<int>(sizeof(BmActF16Block))) / 16;
    for (int i = tid; i < n_i4; i += nthr) {
        reinterpret_cast<int4 *>(s_a)[i] = reinterpret_cast<const int4 *>(a_src)[i];
    }
    __syncthreads();

    const int lane   = threadIdx.x;
    const int sub_r  = lane >> 3; // 0..3 (4 rows per warp -> 32 rows/CTA)
    const int sub_l  = lane & 7;  // 0..7
    const int b_init = sub_l >> 2;
    const int sub    = sub_l & 3;
    const int h      = sub >> 1;
    const int is     = sub & 1;

    const int row = (blockIdx.x * blockDim.y + threadIdx.y) * 4 + sub_r;
    if (row >= K) return;

    const int4 * ql_p, * qh_p;
    const int8_t * sc_p;
    const uint16_t * d_p;
    float ew = 1.0f;
    if (!is_sh) {
        const size_t slot_idx = static_cast<size_t>(t) * k_used + slot;
        const int eid = top_ids[slot_idx];
        ew = top_w[slot_idx];
        const size_t rblk = (static_cast<size_t>(eid) * K + row) * nb;
        ql_p = dn_ql + rblk * 8; qh_p = dn_qh + rblk * 4; sc_p = dn_sc + rblk * 16; d_p = dn_d + rblk;
    } else {
        const size_t rblk = static_cast<size_t>(row) * nb;
        ql_p = sh_ql + rblk * 8; qh_p = sh_qh + rblk * 4; sc_p = sh_sc + rblk * 16; d_p = sh_d + rblk;
    }

    const __half2 neg1056 = __floats2half2_rn(-1056.0f, -1056.0f);
    float acc = 0.0f;
    for (int bi = b_init; bi < nb; bi += 2) {
        const int4 qh_v  = __ldg(&qh_p[bi * 4 + sub]);
        const int4 ql0_v = __ldg(&ql_p[bi * 8 + h * 4 + is]);
        const int4 ql1_v = __ldg(&ql_p[bi * 8 + h * 4 + 2 + is]);
        const int8_t * sc = sc_p + bi * 16 + h * 8 + is;
        const float dw  = bm_half_bits_to_f32(__ldg(&d_p[bi]));
        const float sc0 = dw * static_cast<float>(__ldg(&sc[0]));
        const float sc2 = dw * static_cast<float>(__ldg(&sc[2]));
        const float sc4 = dw * static_cast<float>(__ldg(&sc[4]));
        const float sc6 = dw * static_cast<float>(__ldg(&sc[6]));
        const uint32_t ql0w[4] = { static_cast<uint32_t>(ql0_v.x), static_cast<uint32_t>(ql0_v.y), static_cast<uint32_t>(ql0_v.z), static_cast<uint32_t>(ql0_v.w) };
        const uint32_t ql1w[4] = { static_cast<uint32_t>(ql1_v.x), static_cast<uint32_t>(ql1_v.y), static_cast<uint32_t>(ql1_v.z), static_cast<uint32_t>(ql1_v.w) };
        const uint32_t qhw[4]  = { static_cast<uint32_t>(qh_v.x),  static_cast<uint32_t>(qh_v.y),  static_cast<uint32_t>(qh_v.z),  static_cast<uint32_t>(qh_v.w)  };

        const int xb = bi * 8 + h * 4;
        const __half2 * x0 = s_a[xb + 0].x + is * 8;
        const __half2 * x1 = s_a[xb + 1].x + is * 8;
        const __half2 * x2 = s_a[xb + 2].x + is * 8;
        const __half2 * x3 = s_a[xb + 3].x + is * 8;

        __half2 a0 = __floats2half2_rn(0.0f, 0.0f), a1 = __floats2half2_rn(0.0f, 0.0f);
        __half2 a2 = __floats2half2_rn(0.0f, 0.0f), a3 = __floats2half2_rn(0.0f, 0.0f);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t q1 = (ql0w[c] & 0x0F0F0F0Fu) | ((qhw[c] & 0x03030303u) << 4);
            const uint32_t q2 = (ql1w[c] & 0x0F0F0F0Fu) | ((qhw[c] & 0x0C0C0C0Cu) << 2);
            const uint32_t q3 = ((ql0w[c] >> 4) & 0x0F0F0F0Fu) | (qhw[c] & 0x30303030u);
            const uint32_t q4 = ((ql1w[c] >> 4) & 0x0F0F0F0Fu) | ((qhw[c] >> 2) & 0x30303030u);
            a0 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_lo(q1), neg1056), x0[c * 2 + 0], a0);
            a0 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_hi(q1), neg1056), x0[c * 2 + 1], a0);
            a1 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_lo(q2), neg1056), x1[c * 2 + 0], a1);
            a1 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_hi(q2), neg1056), x1[c * 2 + 1], a1);
            a2 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_lo(q3), neg1056), x2[c * 2 + 0], a2);
            a2 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_hi(q3), neg1056), x2[c * 2 + 1], a2);
            a3 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_lo(q4), neg1056), x3[c * 2 + 0], a3);
            a3 = bm_hfma2(bm_hadd2(bm_u8x2_to_h2_hi(q4), neg1056), x3[c * 2 + 1], a3);
        }
        const float2 f0 = __half22float2(a0), f1 = __half22float2(a1), f2 = __half22float2(a2), f3 = __half22float2(a3);
        acc += (sc0 * s_a[xb + 0].d) * (f0.x + f0.y) + (sc2 * s_a[xb + 1].d) * (f1.x + f1.y)
             + (sc4 * s_a[xb + 2].d) * (f2.x + f2.y) + (sc6 * s_a[xb + 3].d) * (f3.x + f3.y);
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 4, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 2, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 1, 8);
    if (sub_l == 0) {
        const int n_slots = k_used + 1;
        down_slots[(static_cast<size_t>(t) * n_slots + slot) * K + row] = ew * acc;
    }
}

// 4F. sm_61/sm_52 dp4a Q6_K SoA Split-Slot Down Decode (4 rows/warp = 32 rows/CTA)
__global__ void bm_moe_down_q6_k_soa_dp4a_split_dec_kernel(
    const int4 *          __restrict__ dn_ql,  const int4 * __restrict__ dn_qh,
    const int8_t *        __restrict__ dn_sc,  const uint16_t * __restrict__ dn_d,
    const int4 *          __restrict__ sh_ql,  const int4 * __restrict__ sh_qh,
    const int8_t *        __restrict__ sh_sc,  const uint16_t * __restrict__ sh_d,
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

    const int lane   = threadIdx.x;
    const int sub_r  = lane >> 3;
    const int sub_l  = lane & 7;
    const int b_init = sub_l >> 2;
    const int sub    = sub_l & 3;
    const int h      = sub >> 1;
    const int is     = sub & 1;

    const int row = (blockIdx.x * blockDim.y + threadIdx.y) * 4 + sub_r;
    if (row >= K) return;

    const int4 * ql_p, * qh_p;
    const int8_t * sc_p;
    const uint16_t * d_p;
    float ew = 1.0f;
    if (!is_sh) {
        const size_t slot_idx = static_cast<size_t>(t) * k_used + slot;
        const int eid = top_ids[slot_idx];
        ew = top_w[slot_idx];
        const size_t rblk = (static_cast<size_t>(eid) * K + row) * nb;
        ql_p = dn_ql + rblk * 8; qh_p = dn_qh + rblk * 4; sc_p = dn_sc + rblk * 16; d_p = dn_d + rblk;
    } else {
        const size_t rblk = static_cast<size_t>(row) * nb;
        ql_p = sh_ql + rblk * 8; qh_p = sh_qh + rblk * 4; sc_p = sh_sc + rblk * 16; d_p = sh_d + rblk;
    }

    float acc = 0.0f;
    for (int bi = b_init; bi < nb; bi += 2) {
        const int4 qh_v  = __ldg(&qh_p[bi * 4 + sub]);
        const int4 ql0_v = __ldg(&ql_p[bi * 8 + h * 4 + is]);
        const int4 ql1_v = __ldg(&ql_p[bi * 8 + h * 4 + 2 + is]);
        const int8_t * sc = sc_p + bi * 16 + h * 8 + is;
        const float dw  = bm_half_bits_to_f32(__ldg(&d_p[bi]));
        const float sc0 = dw * static_cast<float>(__ldg(&sc[0]));
        const float sc2 = dw * static_cast<float>(__ldg(&sc[2]));
        const float sc4 = dw * static_cast<float>(__ldg(&sc[4]));
        const float sc6 = dw * static_cast<float>(__ldg(&sc[6]));
        const uint32_t ql0w[4] = { static_cast<uint32_t>(ql0_v.x), static_cast<uint32_t>(ql0_v.y), static_cast<uint32_t>(ql0_v.z), static_cast<uint32_t>(ql0_v.w) };
        const uint32_t ql1w[4] = { static_cast<uint32_t>(ql1_v.x), static_cast<uint32_t>(ql1_v.y), static_cast<uint32_t>(ql1_v.z), static_cast<uint32_t>(ql1_v.w) };
        const uint32_t qhw[4]  = { static_cast<uint32_t>(qh_v.x),  static_cast<uint32_t>(qh_v.y),  static_cast<uint32_t>(qh_v.z),  static_cast<uint32_t>(qh_v.w)  };

        const int xb = bi * 8 + h * 4;
        const int4 x0 = reinterpret_cast<const int4 *>(s_a[xb + 0].qs)[is];
        const int4 x1 = reinterpret_cast<const int4 *>(s_a[xb + 1].qs)[is];
        const int4 x2 = reinterpret_cast<const int4 *>(s_a[xb + 2].qs)[is];
        const int4 x3 = reinterpret_cast<const int4 *>(s_a[xb + 3].qs)[is];
        const int sx0 = is ? s_a[xb + 0].s1 : s_a[xb + 0].s0;
        const int sx1 = is ? s_a[xb + 1].s1 : s_a[xb + 1].s0;
        const int sx2 = is ? s_a[xb + 2].s1 : s_a[xb + 2].s0;
        const int sx3 = is ? s_a[xb + 3].s1 : s_a[xb + 3].s0;

        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        const int x0a[4] = {x0.x, x0.y, x0.z, x0.w};
        const int x1a[4] = {x1.x, x1.y, x1.z, x1.w};
        const int x2a[4] = {x2.x, x2.y, x2.z, x2.w};
        const int x3a[4] = {x3.x, x3.y, x3.z, x3.w};
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int q1 = static_cast<int>((ql0w[c] & 0x0F0F0F0Fu) | ((qhw[c] & 0x03030303u) << 4));
            const int q2 = static_cast<int>((ql1w[c] & 0x0F0F0F0Fu) | ((qhw[c] & 0x0C0C0C0Cu) << 2));
            const int q3 = static_cast<int>(((ql0w[c] >> 4) & 0x0F0F0F0Fu) | (qhw[c] & 0x30303030u));
            const int q4 = static_cast<int>(((ql1w[c] >> 4) & 0x0F0F0F0Fu) | ((qhw[c] >> 2) & 0x30303030u));
            s0 = dp4a_s32(q1, x0a[c], s0);
            s1 = dp4a_s32(q2, x1a[c], s1);
            s2 = dp4a_s32(q3, x2a[c], s2);
            s3 = dp4a_s32(q4, x3a[c], s3);
        }
        acc += (sc0 * s_a[xb + 0].d) * static_cast<float>(s0 - 32 * sx0)
             + (sc2 * s_a[xb + 1].d) * static_cast<float>(s1 - 32 * sx1)
             + (sc4 * s_a[xb + 2].d) * static_cast<float>(s2 - 32 * sx2)
             + (sc6 * s_a[xb + 3].d) * static_cast<float>(s3 - 32 * sx3);
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 4, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 2, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 1, 8);
    if (sub_l == 0) {
        const int n_slots = k_used + 1;
        down_slots[(static_cast<size_t>(t) * n_slots + slot) * K + row] = ew * acc;
    }
}

} // namespace

// ============================================================================
// Public Launchers
// ============================================================================

bool try_launch_linear_q6k_soa_prefill_sm60_f16(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    if (cublas == nullptr || T < 16 || device_sm_version(W.device) != 60) return false;
    if (!(W.dtype == DType::Q6_K && W.is_soa)) return false;

    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    const size_t xf16_b = align256(static_cast<size_t>(T) * K * sizeof(__half));
    const size_t yf16_b = align256(static_cast<size_t>(T) * M * sizeof(__half));
    if (xf16_b + yf16_b + static_cast<size_t>(K) * sizeof(__half) > scratch_bytes) return false;

    const size_t rem = scratch_bytes - xf16_b - yf16_b;
    const int chunk_m = std::max(1, std::min(M, static_cast<int>(rem / (static_cast<size_t>(K) * sizeof(__half)))));
    auto * base  = static_cast<uint8_t *>(d_scratch);
    auto * d_x16 = reinterpret_cast<__half *>(base);
    auto * d_y16 = reinterpret_cast<__half *>(base + xf16_b);
    auto * d_w16 = reinterpret_cast<__half *>(base + xf16_b + yf16_b);

    launch_cast_f32_to_f16(d_x, d_x16, static_cast<int64_t>(T) * K, stream);
    cublasSetStream(cublas, stream);
    const __half h_one = __float2half_rn(1.0f), h_zero = __float2half_rn(0.0f);
    const int nb_row = K >> 8;
    constexpr int wpb = 8;

    const auto * ql = static_cast<const uint8_t *>(W.ptr);
    const auto * qh = static_cast<const uint8_t *>(W.aux_ptr);
    const auto * sc = static_cast<const int8_t *>(W.scales_ptr);
    const auto * d  = static_cast<const uint16_t *>(W.d_ptr);

    for (int m0 = 0; m0 < M; m0 += chunk_m) {
        const int cm = std::min(chunk_m, M - m0);
        const int64_t b0   = static_cast<int64_t>(m0) * nb_row;
        const int64_t nblk = static_cast<int64_t>(cm) * nb_row;
        bm_dequant_q6_k_soa_batch_kernel<__half><<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
            ql + b0 * 128, qh + b0 * 64, sc + b0 * 16, d + b0, d_w16, nblk);
        cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    cm, T, K,
                    &h_one,  d_w16, K,
                    d_x16, K,
                    &h_zero, d_y16 + m0, M);
    }

    const int64_t ny = static_cast<int64_t>(T) * M;
    bm_f16_to_f32_vec_kernel<<<((ny + 1) / 2 + 255) / 256, 256, 0, stream>>>(d_y16, d_y, ny);
    return true;
}

bool try_launch_linear_kquant_prefill_sm61_dp4a(
    const DeviceTensor & W, const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    // At T=16..127 dp4a beats FP32 dequant+cublasSgemm by ~2x on sm_61; at T>=128 cublasSgemm wins.
    if (T < 16 || T >= 128 || device_sm_version(W.device) != 61) return false;
    const bool is_q4k = (W.dtype == DType::Q4_K);
    const bool is_q6k = (W.dtype == DType::Q6_K && W.is_soa);
    if (!is_q4k && !is_q6k) return false;

    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);
    if ((K & 255) != 0) return false;

    const size_t xq8_b = (static_cast<size_t>(T) * (K >> 5) * sizeof(BlockQ8_1_Act) + 255) & ~size_t(255);
    if (xq8_b > scratch_bytes) return false;

    auto * x_q8 = static_cast<BlockQ8_1_Act *>(d_scratch);
    launch_quantize_q8_1(d_x, x_q8, K, T, stream);

    if (T >= 32) {
        constexpr int BT_TH = 16;
        constexpr int BT    = 2 * BT_TH;
        const size_t smem   = static_cast<size_t>(BT) * (16 * sizeof(int4) + 24 * sizeof(float));
        if (is_q4k) {
            bm_linear_q4_k_dp4a_kernel<BT_TH><<<dim3((M + 31) / 32, (T + BT - 1) / BT), 256, smem, stream>>>(
                static_cast<const BlockQ4_K_BM *>(W.ptr), x_q8, d_y, M, K, T);
        } else {
            bm_linear_q6_k_soa_dp4a_kernel<BT_TH><<<dim3((M + 63) / 64, (T + BT - 1) / BT), 256, smem, stream>>>(
                static_cast<const int4 *>(W.ptr), static_cast<const int4 *>(W.aux_ptr),
                static_cast<const int8_t *>(W.scales_ptr), static_cast<const uint16_t *>(W.d_ptr),
                x_q8, d_y, M, K, T);
        }
    } else {
        constexpr int BT_TH = 8;
        constexpr int BT    = 2 * BT_TH;
        const size_t smem   = static_cast<size_t>(BT) * (16 * sizeof(int4) + 24 * sizeof(float));
        if (is_q4k) {
            bm_linear_q4_k_dp4a_kernel<BT_TH><<<dim3((M + 31) / 32, (T + BT - 1) / BT), 256, smem, stream>>>(
                static_cast<const BlockQ4_K_BM *>(W.ptr), x_q8, d_y, M, K, T);
        } else {
            bm_linear_q6_k_soa_dp4a_kernel<BT_TH><<<dim3((M + 63) / 64, (T + BT - 1) / BT), 256, smem, stream>>>(
                static_cast<const int4 *>(W.ptr), static_cast<const int4 *>(W.aux_ptr),
                static_cast<const int8_t *>(W.scales_ptr), static_cast<const uint16_t *>(W.d_ptr),
                x_q8, d_y, M, K, T);
        }
    }
    return true;
}

bool try_launch_moe_q4k_q6k_pascal(
    cublasHandle_t cublas,
    const DeviceTensor & gate_exps, const DeviceTensor & up_exps, const DeviceTensor & down_exps,
    const DeviceTensor & gate_shexp, const DeviceTensor & up_shexp, const DeviceTensor & down_shexp,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, float clamp_exp, float clamp_shexp, cudaStream_t stream) {

    const int sm = device_sm_version(gate_exps.device);
    if (sm >= 70) return false; // Volta/Ampere+ use WMMA / Tensor Core paths

    if (!(gate_exps.dtype == DType::Q4_K && up_exps.dtype == DType::Q4_K)) return false;
    const bool dn_is_q4k = (down_exps.dtype == DType::Q4_K);
    const bool dn_is_q6k = (down_exps.dtype == DType::Q6_K && down_exps.is_soa);
    if (!dn_is_q4k && !dn_is_q6k) return false;

    const int K        = static_cast<int>(gate_exps.ne[0]);
    const int ff       = static_cast<int>(gate_exps.ne[1]);
    const int n_expert = static_cast<int>(gate_exps.ne[2]);
    const int ff_sh    = gate_shexp.valid() ? static_cast<int>(gate_shexp.ne[1]) : 0;
    if ((K & 255) != 0 || (ff & 255) != 0) return false;

    const int K_blocks = K >> 5;
    const int ff_blks  = ff >> 5;
    const int sh_blks  = ff_sh >> 5;
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    auto * base = static_cast<uint8_t *>(d_scratch);

    // A. PREFILL (T >= 16): Bank-Conflict-Free BT=32 (T>=128) / BT=16 (T<128) Fused Grouped MoE!
    if (T >= 16 && k_used <= 16 && n_expert > 0 && n_expert <= 512) {
        const bool use_bt32 = (T >= 128);
        const int BT        = use_bt32 ? 32 : 16;
        const int max_tiles = (T * k_used + BT - 1) / BT + n_expert;

        const size_t act_b  = align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
        const size_t shg_b  = align256(static_cast<size_t>(T) * std::max(1, ff_sh) * sizeof(float));
        const size_t shu_b  = shg_b;
        const size_t off_b  = align256(static_cast<size_t>(n_expert + 1) * sizeof(int32_t));
        const size_t sp_b   = align256(static_cast<size_t>(T) * k_used * sizeof(int32_t));
        const size_t tl_b   = align256(static_cast<size_t>(max_tiles) * sizeof(BmTileDesc));
        const size_t nt_b   = 256;
        const size_t sl_b   = align256(static_cast<size_t>(T) * k_used * K * sizeof(float));
        const size_t meta_b = off_b + sp_b + tl_b + nt_b;

        if (sm == 60) {
            const size_t xf_b = align256(static_cast<size_t>(T) * K_blocks * sizeof(BmActF16Block));
            const size_t af_b = align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BmActF16Block));
            const size_t persist_b = meta_b + af_b;
            const size_t peak_ovl  = std::max(shg_b + shu_b + (static_cast<size_t>(4) << 20), std::max(xf_b + act_b, sl_b));
            if (persist_b + peak_ovl <= scratch_bytes) {
                auto * exp_offsets  = reinterpret_cast<int32_t *>(base);
                auto * sorted_pairs = reinterpret_cast<int32_t *>(base + off_b);
                auto * tiles        = reinterpret_cast<BmTileDesc *>(base + off_b + sp_b);
                auto * n_tiles_ptr  = reinterpret_cast<int32_t *>(base + off_b + sp_b + tl_b);
                auto * a_f16        = reinterpret_cast<BmActF16Block *>(base + meta_b);
                uint8_t * ovl       = base + persist_b;
                const size_t ovl_b  = scratch_bytes - persist_b;

                // Phase 0: Shared expert via cublasHgemm
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

                // Phase 1: Routed Gate+Up (native sm_60 half2, conflict-free, fused)
                auto * x_f16 = reinterpret_cast<BmActF16Block *>(ovl);
                auto * act   = reinterpret_cast<float *>(ovl + xf_b);
                const int xb_tot = T * K_blocks;
                const int q_wpb  = (xb_tot <= 256) ? 1 : 8;
                bm_quantize_act_f16_kernel<<<(xb_tot + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(d_x, x_f16, xb_tot);
                bm_group_moe_tokens_kernel<<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
                    d_top_ids, exp_offsets, sorted_pairs, T * k_used, n_expert, k_used);
                bm_build_tile_worklist_kernel<<<1, 1, 0, stream>>>(exp_offsets, tiles, n_tiles_ptr, n_expert, BT);

                const size_t q4_smem = static_cast<size_t>(BT) * (32 * sizeof(int4) + 24 * sizeof(float));
                const size_t q6_smem = static_cast<size_t>(BT) * (32 * sizeof(int4) +  8 * sizeof(float));
                if (use_bt32) {
                    bm_moe_gate_up_grouped_q4_k_h2_kernel<16>
                        <<<dim3((ff + 15) / 16, max_tiles), 256, q4_smem, stream>>>(
                            static_cast<const BlockQ4_K_BM *>(gate_exps.ptr),
                            static_cast<const BlockQ4_K_BM *>(up_exps.ptr),
                            x_f16, tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, clamp_exp);
                } else {
                    bm_moe_gate_up_grouped_q4_k_h2_kernel<8>
                        <<<dim3((ff + 15) / 16, max_tiles), 256, q4_smem, stream>>>(
                            static_cast<const BlockQ4_K_BM *>(gate_exps.ptr),
                            static_cast<const BlockQ4_K_BM *>(up_exps.ptr),
                            x_f16, tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, clamp_exp);
                }

                const int ab_tot = T * k_used * ff_blks;
                bm_quantize_act_f16_kernel<<<(ab_tot + 7) / 8, dim3(32, 8), 0, stream>>>(act, a_f16, ab_tot);

                // Phase 2: Routed Down (native sm_60 half2 RM=2 Q4_K or Q6_K SoA)
                auto * down_slots = reinterpret_cast<float *>(ovl);
                if (dn_is_q4k) {
                    if (use_bt32) {
                        bm_moe_down_grouped_q4_k_h2_kernel<16>
                            <<<dim3((K + 31) / 32, max_tiles), 256, q4_smem, stream>>>(
                                static_cast<const BlockQ4_K_BM *>(down_exps.ptr),
                                a_f16, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                    } else {
                        bm_moe_down_grouped_q4_k_h2_kernel<8>
                            <<<dim3((K + 31) / 32, max_tiles), 256, q4_smem, stream>>>(
                                static_cast<const BlockQ4_K_BM *>(down_exps.ptr),
                                a_f16, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                    }
                } else {
                    if (use_bt32) {
                        bm_moe_down_grouped_q6_k_soa_h2_kernel<16>
                            <<<dim3((K + 63) / 64, max_tiles), 256, q6_smem, stream>>>(
                                static_cast<const int4 *>(down_exps.ptr), static_cast<const int4 *>(down_exps.aux_ptr),
                                static_cast<const int8_t *>(down_exps.scales_ptr), static_cast<const uint16_t *>(down_exps.d_ptr),
                                a_f16, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                    } else {
                        bm_moe_down_grouped_q6_k_soa_h2_kernel<8>
                            <<<dim3((K + 63) / 64, max_tiles), 256, q6_smem, stream>>>(
                                static_cast<const int4 *>(down_exps.ptr), static_cast<const int4 *>(down_exps.aux_ptr),
                                static_cast<const int8_t *>(down_exps.scales_ptr), static_cast<const uint16_t *>(down_exps.d_ptr),
                                a_f16, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                    }
                }

                bm_reduce_slots_add_kernel<<<dim3(((K >> 2) + 255) / 256, T), 256, 0, stream>>>(
                    down_slots, d_out, K, k_used, true);
                return true;
            }
        } else {
            // sm_61 (native __dp4a) & sm_50/52 (Maxwell fallback): conflict-free BT=32/16 fused grouped MoE!
            const size_t xq_b = align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
            const size_t aq_b = align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BlockQ8_1_Act));
            const size_t persist_b = meta_b + aq_b;
            const size_t peak_ovl  = std::max(shg_b + shu_b + (static_cast<size_t>(4) << 20), std::max(xq_b + act_b, sl_b));
            if (persist_b + peak_ovl <= scratch_bytes) {
                auto * exp_offsets  = reinterpret_cast<int32_t *>(base);
                auto * sorted_pairs = reinterpret_cast<int32_t *>(base + off_b);
                auto * tiles        = reinterpret_cast<BmTileDesc *>(base + off_b + sp_b);
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
                bm_group_moe_tokens_kernel<<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
                    d_top_ids, exp_offsets, sorted_pairs, T * k_used, n_expert, k_used);
                bm_build_tile_worklist_kernel<<<1, 1, 0, stream>>>(exp_offsets, tiles, n_tiles_ptr, n_expert, BT);

                const size_t dp_smem = static_cast<size_t>(BT) * (16 * sizeof(int4) + 24 * sizeof(float));
                if (use_bt32) {
                    bm_moe_gate_up_grouped_q4_k_dp4a_kernel<16>
                        <<<dim3((ff + 15) / 16, max_tiles), 256, dp_smem, stream>>>(
                            static_cast<const BlockQ4_K_BM *>(gate_exps.ptr),
                            static_cast<const BlockQ4_K_BM *>(up_exps.ptr),
                            x_q8, tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, clamp_exp);
                } else {
                    bm_moe_gate_up_grouped_q4_k_dp4a_kernel<8>
                        <<<dim3((ff + 15) / 16, max_tiles), 256, dp_smem, stream>>>(
                            static_cast<const BlockQ4_K_BM *>(gate_exps.ptr),
                            static_cast<const BlockQ4_K_BM *>(up_exps.ptr),
                            x_q8, tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, clamp_exp);
                }

                launch_quantize_q8_1(act, a_q8, k_used * ff, T, stream);

                auto * down_slots = reinterpret_cast<float *>(ovl);
                if (dn_is_q4k) {
                    if (use_bt32) {
                        bm_moe_down_grouped_q4_k_dp4a_kernel<16>
                            <<<dim3((K + 31) / 32, max_tiles), 256, dp_smem, stream>>>(
                                static_cast<const BlockQ4_K_BM *>(down_exps.ptr),
                                a_q8, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                    } else {
                        bm_moe_down_grouped_q4_k_dp4a_kernel<8>
                            <<<dim3((K + 31) / 32, max_tiles), 256, dp_smem, stream>>>(
                                static_cast<const BlockQ4_K_BM *>(down_exps.ptr),
                                a_q8, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                    }
                } else {
                    if (use_bt32) {
                        bm_moe_down_grouped_q6_k_soa_dp4a_kernel<16>
                            <<<dim3((K + 63) / 64, max_tiles), 256, dp_smem, stream>>>(
                                static_cast<const int4 *>(down_exps.ptr), static_cast<const int4 *>(down_exps.aux_ptr),
                                static_cast<const int8_t *>(down_exps.scales_ptr), static_cast<const uint16_t *>(down_exps.d_ptr),
                                a_q8, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                    } else {
                        bm_moe_down_grouped_q6_k_soa_dp4a_kernel<8>
                            <<<dim3((K + 63) / 64, max_tiles), 256, dp_smem, stream>>>(
                                static_cast<const int4 *>(down_exps.ptr), static_cast<const int4 *>(down_exps.aux_ptr),
                                static_cast<const int8_t *>(down_exps.scales_ptr), static_cast<const uint16_t *>(down_exps.d_ptr),
                                a_q8, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                    }
                }

                bm_reduce_slots_add_kernel<<<dim3(((K >> 2) + 255) / 256, T), 256, 0, stream>>>(
                    down_slots, d_out, K, k_used, true);
                return true;
            }
        }
        return false;
    }

    // B. DECODE (T < 16): Fused (k_used + 1) Routed + Shared MoE with Direct In-Smem Activation Quant!
    const bool has_sh_fast = (ff_sh > 0 && (ff_sh & 255) == 0 &&
                              gate_shexp.dtype == DType::Q4_K &&
                              up_shexp.dtype   == DType::Q4_K &&
                              down_shexp.dtype == down_exps.dtype &&
                              down_shexp.is_soa == down_exps.is_soa);
    if (T < 16 && has_sh_fast) {
        const int n_slots = k_used + 1;
        if (sm == 60) {
            const size_t gu_smem = static_cast<size_t>(K_blocks) * sizeof(BmActF16Block);
            const size_t dn_smem = static_cast<size_t>(std::max(ff_blks, sh_blks)) * sizeof(BmActF16Block);
            if (gu_smem > 48000 || dn_smem > 48000) return false;

            size_t off = 0;
            auto * x_f16  = reinterpret_cast<BmActF16Block *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * sizeof(BmActF16Block));
            auto * ar_f16 = reinterpret_cast<BmActF16Block *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BmActF16Block));
            auto * as_f16 = reinterpret_cast<BmActF16Block *>(base + off); off += align256(static_cast<size_t>(T) * sh_blks * sizeof(BmActF16Block));
            auto * slots  = reinterpret_cast<float *>        (base + off); off += align256(static_cast<size_t>(T) * n_slots * K * sizeof(float));
            if (off <= scratch_bytes) {
                const int xb_tot = T * K_blocks;
                const int q_wpb  = (xb_tot <= 256) ? 2 : 8;
                bm_quantize_act_f16_kernel<<<(xb_tot + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(d_x, x_f16, xb_tot);

                const int n_gu_tiles = k_used * (ff >> 5) + (ff_sh >> 5);
                bm_moe_gate_up_q4_k_h2_dec_kernel
                    <<<dim3(n_gu_tiles, T), dim3(32, 8), gu_smem, stream>>>(
                        static_cast<const BlockQ4_K_BM *>(gate_exps.ptr),
                        static_cast<const BlockQ4_K_BM *>(up_exps.ptr),
                        static_cast<const BlockQ4_K_BM *>(gate_shexp.ptr),
                        static_cast<const BlockQ4_K_BM *>(up_shexp.ptr),
                        x_f16, d_top_ids, ar_f16, as_f16, K, ff, ff_sh, k_used, clamp_exp, clamp_shexp);

                constexpr int wpb_dn = 8;
                if (dn_is_q4k) {
                    bm_moe_down_q4_k_h2_split_dec_kernel
                        <<<dim3((K + wpb_dn * 4 - 1) / (wpb_dn * 4), n_slots, T), dim3(32, wpb_dn), dn_smem, stream>>>(
                            static_cast<const BlockQ4_K_BM *>(down_exps.ptr),
                            static_cast<const BlockQ4_K_BM *>(down_shexp.ptr),
                            ar_f16, as_f16, d_top_ids, d_top_w, slots, K, ff, ff_sh, k_used);
                } else {
                    bm_moe_down_q6_k_soa_h2_split_dec_kernel
                        <<<dim3((K + wpb_dn * 4 - 1) / (wpb_dn * 4), n_slots, T), dim3(32, wpb_dn), dn_smem, stream>>>(
                            static_cast<const int4 *>(down_exps.ptr), static_cast<const int4 *>(down_exps.aux_ptr),
                            static_cast<const int8_t *>(down_exps.scales_ptr), static_cast<const uint16_t *>(down_exps.d_ptr),
                            static_cast<const int4 *>(down_shexp.ptr), static_cast<const int4 *>(down_shexp.aux_ptr),
                            static_cast<const int8_t *>(down_shexp.scales_ptr), static_cast<const uint16_t *>(down_shexp.d_ptr),
                            ar_f16, as_f16, d_top_ids, d_top_w, slots, K, ff, ff_sh, k_used);
                }

                bm_reduce_slots_add_kernel<<<dim3(((K >> 2) + 63) / 64, T), 64, 0, stream>>>(
                    slots, d_out, K, n_slots, false);
                return true;
            }
        } else {
            const size_t gu_smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
            const size_t dn_smem = static_cast<size_t>(std::max(ff_blks, sh_blks)) * sizeof(BlockQ8_1_Act);
            if (gu_smem > 48000 || dn_smem > 48000) return false;

            size_t off = 0;
            auto * x_q8  = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
            auto * aq8   = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BlockQ8_1_Act));
            auto * sq8   = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * sh_blks * sizeof(BlockQ8_1_Act));
            auto * slots = reinterpret_cast<float *>        (base + off); off += align256(static_cast<size_t>(T) * n_slots * K * sizeof(float));
            if (off <= scratch_bytes) {
                launch_quantize_q8_1(d_x, x_q8, K, T, stream);

                const int n_gu_tiles = k_used * (ff >> 5) + (ff_sh >> 5);
                bm_moe_gate_up_q4_k_dp4a_dec_kernel
                    <<<dim3(n_gu_tiles, T), dim3(32, 8), gu_smem, stream>>>(
                        static_cast<const BlockQ4_K_BM *>(gate_exps.ptr),
                        static_cast<const BlockQ4_K_BM *>(up_exps.ptr),
                        static_cast<const BlockQ4_K_BM *>(gate_shexp.ptr),
                        static_cast<const BlockQ4_K_BM *>(up_shexp.ptr),
                        x_q8, d_top_ids, aq8, sq8, K, ff, ff_sh, k_used, clamp_exp, clamp_shexp);

                constexpr int wpb_dn = 8;
                if (dn_is_q4k) {
                    bm_moe_down_q4_k_dp4a_split_dec_kernel
                        <<<dim3((K + wpb_dn * 2 - 1) / (wpb_dn * 2), n_slots, T), dim3(32, wpb_dn), dn_smem, stream>>>(
                            static_cast<const BlockQ4_K_BM *>(down_exps.ptr),
                            static_cast<const BlockQ4_K_BM *>(down_shexp.ptr),
                            aq8, sq8, d_top_ids, d_top_w, slots, K, ff, ff_sh, k_used);
                } else {
                    bm_moe_down_q6_k_soa_dp4a_split_dec_kernel
                        <<<dim3((K + wpb_dn * 4 - 1) / (wpb_dn * 4), n_slots, T), dim3(32, wpb_dn), dn_smem, stream>>>(
                            static_cast<const int4 *>(down_exps.ptr), static_cast<const int4 *>(down_exps.aux_ptr),
                            static_cast<const int8_t *>(down_exps.scales_ptr), static_cast<const uint16_t *>(down_exps.d_ptr),
                            static_cast<const int4 *>(down_shexp.ptr), static_cast<const int4 *>(down_shexp.aux_ptr),
                            static_cast<const int8_t *>(down_shexp.scales_ptr), static_cast<const uint16_t *>(down_shexp.d_ptr),
                            aq8, sq8, d_top_ids, d_top_w, slots, K, ff, ff_sh, k_used);
                }

                bm_reduce_slots_add_kernel<<<dim3(((K >> 2) + 63) / 64, T), 64, 0, stream>>>(
                    slots, d_out, K, n_slots, false);
                return true;
            }
        }
    }

    return false;
}

} // namespace rookery
