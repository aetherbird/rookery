#include "core/device.h"
#include "kernels/kernels.h"
#include "kernels/iq_tables.cuh"
#include "kernels/q8_1_act.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace rookery {

namespace {

// Exact ggml K-quant & IQ-quant on-disk block structs (QK_K = 256)
struct BlockQ2_K {
    uint8_t  scales[16];
    uint8_t  qs[64];
    uint16_t d;
    uint16_t dmin;
};
static_assert(sizeof(BlockQ2_K) == 84, "BlockQ2_K size");

struct BlockQ3_K {
    uint8_t  hmask[32];
    uint8_t  qs[64];
    uint8_t  scales[12];
    uint16_t d;
};
static_assert(sizeof(BlockQ3_K) == 110, "BlockQ3_K size");

struct BlockQ4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K size");

struct BlockQ5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ5_K) == 176, "BlockQ5_K size");

struct BlockQ6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K size");

struct BlockIQ2_XXS {
    uint16_t d;
    uint16_t qs[32];
};
static_assert(sizeof(BlockIQ2_XXS) == 66, "BlockIQ2_XXS size");

struct BlockIQ3_XXS {
    uint16_t d;
    uint8_t  qs[96];
};
static_assert(sizeof(BlockIQ3_XXS) == 98, "BlockIQ3_XXS size");

struct BlockIQ2_S {
    uint16_t d;
    uint8_t  qs[64];
    uint8_t  qh[8];
    uint8_t  scales[8];
};
static_assert(sizeof(BlockIQ2_S) == 82, "BlockIQ2_S size");

struct BlockIQ3_S {
    uint16_t d;
    uint8_t  qs[64];
    uint8_t  qh[8];
    uint8_t  signs[32];
    uint8_t  scales[4];
};
static_assert(sizeof(BlockIQ3_S) == 110, "BlockIQ3_S size");

struct BlockIQ4_XS {
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[4];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockIQ4_XS) == 136, "BlockIQ4_XS size");

// Branchless, ZERO-memory-lookup 4-byte conditional 2's-complement negation!
// Expands 4 sign bits s4 (b3,b2,b1,b0) into 4 bytes (0x00 or 0x01) in 2 ALU ops:
//   add4 = (s4 * 0x00204081u) & 0x01010101u
//   xor4 = add4 * 0xFFu
// Valid for all positive bytes in [1..127] (guaranteed for iq2s_grid_d, iq3s_grid_d, iq3xxs_grid_d!).
__device__ __forceinline__ int apply_signs4(uint32_t g4, uint32_t s4) {
    const uint32_t add4 = (s4 * 0x00204081u) & 0x01010101u;
    const uint32_t xor4 = add4 * 0xFFu;
    return static_cast<int>((g4 ^ xor4) + add4);
}

__device__ __forceinline__ float half_bits_to_f32(uint16_t bits) {
    __half h;
    *reinterpret_cast<uint16_t *>(&h) = bits;
    return __half2float(h);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

template <DType DT>
__host__ __device__ constexpr size_t block_bytes_of() {
    if constexpr (DT == DType::Q6_K)    return sizeof(BlockQ6_K);
    if constexpr (DT == DType::Q5_K)    return sizeof(BlockQ5_K);
    if constexpr (DT == DType::Q4_K)    return sizeof(BlockQ4_K);
    if constexpr (DT == DType::IQ4_XS)  return sizeof(BlockIQ4_XS);
    if constexpr (DT == DType::IQ3_S)   return sizeof(BlockIQ3_S);
    if constexpr (DT == DType::IQ3_XXS) return sizeof(BlockIQ3_XXS);
    if constexpr (DT == DType::IQ2_S)   return sizeof(BlockIQ2_S);
    if constexpr (DT == DType::IQ2_XXS) return sizeof(BlockIQ2_XXS);
    if constexpr (DT == DType::Q3_K)    return sizeof(BlockQ3_K);
    if constexpr (DT == DType::Q2_K)    return sizeof(BlockQ2_K);
    return 0;
}

__device__ __forceinline__ void get_scale_min_k4(
    int j, const uint8_t * __restrict__ q, uint8_t & d_out, uint8_t & m_out) {
    if (j < 4) {
        d_out = __ldg(&q[j]) & 63;
        m_out = __ldg(&q[j + 4]) & 63;
    } else {
        d_out = (__ldg(&q[j + 4]) & 0xF) | ((__ldg(&q[j - 4]) >> 6) << 4);
        m_out = (__ldg(&q[j + 4]) >> 4)  | ((__ldg(&q[j])     >> 6) << 4);
    }
}

// ============================================================================
// PART 1: ULTRA-FAST Q6_K SoA LDG.128 + LDS.128 + dp4a.s32 KERNELS
// 4 lanes per 256-elem Q6_K block -> 8 blocks/warp -> 100% coalesced LDG.128!
// ============================================================================

__device__ __forceinline__ float dot_q6_k_soa_row_q8_1(
    const int4 *          __restrict__ W_ql,   // [nb * 8] int4 for this row
    const int4 *          __restrict__ W_qh,   // [nb * 4] int4 for this row
    const int8_t *        __restrict__ W_sc,   // [nb * 16] int8 for this row
    const uint16_t *      __restrict__ W_d,    // [nb] __half for this row
    const BlockQ8_1_Act * __restrict__ s_x,    // [nb * 8] in shared memory
    int nb, int lane) {

    const int blk_in_warp = lane >> 2; // 0..7
    const int sub         = lane & 3;  // 0..3
    const int h           = sub >> 1;  // 0 or 1 (which 128-elem half)
    const int is          = sub & 1;   // 0 or 1 (l=0..15 or l=16..31)

    float acc = 0.0f;
    for (int bi = blk_in_warp; bi < nb; bi += 8) {
        const int4 qh_v  = __ldg(&W_qh[bi * 4 + sub]);
        const int4 ql0_v = __ldg(&W_ql[bi * 8 + h * 4 + is]);
        const int4 ql1_v = __ldg(&W_ql[bi * 8 + h * 4 + 2 + is]);

        const int8_t * sc = W_sc + bi * 16 + h * 8 + is;
        const float sc0 = static_cast<float>(__ldg(&sc[0]));
        const float sc2 = static_cast<float>(__ldg(&sc[2]));
        const float sc4 = static_cast<float>(__ldg(&sc[4]));
        const float sc6 = static_cast<float>(__ldg(&sc[6]));
        const float dw  = half_bits_to_f32(__ldg(&W_d[bi]));

        const int xb = bi * 8 + h * 4;
        const int4 x1 = reinterpret_cast<const int4 *>(s_x[xb + 0].qs)[is];
        const int4 x2 = reinterpret_cast<const int4 *>(s_x[xb + 1].qs)[is];
        const int4 x3 = reinterpret_cast<const int4 *>(s_x[xb + 2].qs)[is];
        const int4 x4 = reinterpret_cast<const int4 *>(s_x[xb + 3].qs)[is];

        auto step4 = [](uint32_t ql0, uint32_t ql1, uint32_t qh,
                        int xv1, int xv2, int xv3, int xv4,
                        int & s1, int & s2, int & s3, int & s4) {
            const int q1 = static_cast<int>((ql0 & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4));
            const int q2 = static_cast<int>((ql1 & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2));
            const int q3 = static_cast<int>(((ql0 >> 4) & 0x0F0F0F0Fu) | (qh & 0x30303030u));
            const int q4 = static_cast<int>(((ql1 >> 4) & 0x0F0F0F0Fu) | ((qh >> 2) & 0x30303030u));
            s1 = dp4a_s32(q1, xv1, s1);
            s2 = dp4a_s32(q2, xv2, s2);
            s3 = dp4a_s32(q3, xv3, s3);
            s4 = dp4a_s32(q4, xv4, s4);
        };

        int sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
        step4(ql0_v.x, ql1_v.x, qh_v.x, x1.x, x2.x, x3.x, x4.x, sum1, sum2, sum3, sum4);
        step4(ql0_v.y, ql1_v.y, qh_v.y, x1.y, x2.y, x3.y, x4.y, sum1, sum2, sum3, sum4);
        step4(ql0_v.z, ql1_v.z, qh_v.z, x1.z, x2.z, x3.z, x4.z, sum1, sum2, sum3, sum4);
        step4(ql0_v.w, ql1_v.w, qh_v.w, x1.w, x2.w, x3.w, x4.w, sum1, sum2, sum3, sum4);

        const int sx0 = is ? s_x[xb + 0].s1 : s_x[xb + 0].s0;
        const int sx1 = is ? s_x[xb + 1].s1 : s_x[xb + 1].s0;
        const int sx2 = is ? s_x[xb + 2].s1 : s_x[xb + 2].s0;
        const int sx3 = is ? s_x[xb + 3].s1 : s_x[xb + 3].s0;

        acc += dw * (
            (sc0 * s_x[xb + 0].d) * static_cast<float>(sum1 - 32 * sx0) +
            (sc2 * s_x[xb + 1].d) * static_cast<float>(sum2 - 32 * sx1) +
            (sc4 * s_x[xb + 2].d) * static_cast<float>(sum3 - 32 * sx2) +
            (sc6 * s_x[xb + 3].d) * static_cast<float>(sum4 - 32 * sx3));
    }
    return warp_sum(acc);
}

__global__ void gemv_q6_k_soa_dp4a_kernel(
    const int4 *          __restrict__ W_ql,
    const int4 *          __restrict__ W_qh,
    const int8_t *        __restrict__ W_sc,
    const uint16_t *      __restrict__ W_d,
    const BlockQ8_1_Act * __restrict__ X,
    float *               __restrict__ Y,
    int M, int K) {

    extern __shared__ BlockQ8_1_Act s_x[];

    const int t        = blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;

    const BlockQ8_1_Act * x_row = X + static_cast<size_t>(t) * K_blocks;
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(x_row);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const size_t rblk = static_cast<size_t>(row) * nb;
    const float acc = dot_q6_k_soa_row_q8_1(
        W_ql + rblk * 8, W_qh + rblk * 4, W_sc + rblk * 16, W_d + rblk,
        s_x, nb, threadIdx.x);
    if (threadIdx.x == 0) {
        Y[static_cast<size_t>(t) * M + row] = acc;
    }
}

// Fused Q6_K SoA Gate + Up + SiLU (+ optional clamp) sharing s_x in smem!
__global__ void gemv_q6_k_soa_gate_up_silu_clamp_kernel(
    const int4 *          __restrict__ G_ql,
    const int4 *          __restrict__ G_qh,
    const int8_t *        __restrict__ G_sc,
    const uint16_t *      __restrict__ G_d,
    const int4 *          __restrict__ U_ql,
    const int4 *          __restrict__ U_qh,
    const int8_t *        __restrict__ U_sc,
    const uint16_t *      __restrict__ U_d,
    const BlockQ8_1_Act * __restrict__ X,
    float *               __restrict__ Y,
    int M, int K, float clamp_limit) {

    extern __shared__ BlockQ8_1_Act s_x[];
    const int t        = blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;

    const BlockQ8_1_Act * x_row = X + static_cast<size_t>(t) * K_blocks;
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(x_row);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const size_t rblk = static_cast<size_t>(row) * nb;
    float g = dot_q6_k_soa_row_q8_1(G_ql + rblk * 8, G_qh + rblk * 4, G_sc + rblk * 16, G_d + rblk, s_x, nb, threadIdx.x);
    float u = dot_q6_k_soa_row_q8_1(U_ql + rblk * 8, U_qh + rblk * 4, U_sc + rblk * 16, U_d + rblk, s_x, nb, threadIdx.x);
    if (threadIdx.x == 0) {
        if (clamp_limit > 0.0f) {
            g = fminf(fmaxf(g, -clamp_limit), clamp_limit);
            u = fminf(fmaxf(u, -clamp_limit), clamp_limit);
        }
        Y[static_cast<size_t>(t) * M + row] = (g / (1.0f + expf(-g))) * u;
    }
}

// Dual-token (T == 2) weight-reuse Q6_K SoA GEMV for MTP verification!
__global__ void gemv_q6_k_soa_dp4a_t2_kernel(
    const int4 *          __restrict__ W_ql,
    const int4 *          __restrict__ W_qh,
    const int8_t *        __restrict__ W_sc,
    const uint16_t *      __restrict__ W_d,
    const BlockQ8_1_Act * __restrict__ X, // [2, K_blocks]
    float *               __restrict__ Y, // [2, M]
    int M, int K) {

    extern __shared__ BlockQ8_1_Act s_x[];
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    BlockQ8_1_Act * s_x0 = s_x;
    BlockQ8_1_Act * s_x1 = s_x + K_blocks;

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const int n_i4 = (2 * K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(X);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const int lane        = threadIdx.x;
    const int blk_in_warp = lane >> 2;
    const int sub         = lane & 3;
    const int h           = sub >> 1;
    const int is          = sub & 1;

    const size_t rblk = static_cast<size_t>(row) * nb;
    const int4 *     r_ql = W_ql + rblk * 8;
    const int4 *     r_qh = W_qh + rblk * 4;
    const int8_t *   r_sc = W_sc + rblk * 16;
    const uint16_t * r_d  = W_d  + rblk;

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int bi = blk_in_warp; bi < nb; bi += 8) {
        const int4 qh_v  = __ldg(&r_qh[bi * 4 + sub]);
        const int4 ql0_v = __ldg(&r_ql[bi * 8 + h * 4 + is]);
        const int4 ql1_v = __ldg(&r_ql[bi * 8 + h * 4 + 2 + is]);

        const int8_t * sc = r_sc + bi * 16 + h * 8 + is;
        const float sc0 = static_cast<float>(__ldg(&sc[0]));
        const float sc2 = static_cast<float>(__ldg(&sc[2]));
        const float sc4 = static_cast<float>(__ldg(&sc[4]));
        const float sc6 = static_cast<float>(__ldg(&sc[6]));
        const float dw  = half_bits_to_f32(__ldg(&r_d[bi]));

        const int xb = bi * 8 + h * 4;
        const int4 x10 = reinterpret_cast<const int4 *>(s_x0[xb + 0].qs)[is];
        const int4 x20 = reinterpret_cast<const int4 *>(s_x0[xb + 1].qs)[is];
        const int4 x30 = reinterpret_cast<const int4 *>(s_x0[xb + 2].qs)[is];
        const int4 x40 = reinterpret_cast<const int4 *>(s_x0[xb + 3].qs)[is];

        const int4 x11 = reinterpret_cast<const int4 *>(s_x1[xb + 0].qs)[is];
        const int4 x21 = reinterpret_cast<const int4 *>(s_x1[xb + 1].qs)[is];
        const int4 x31 = reinterpret_cast<const int4 *>(s_x1[xb + 2].qs)[is];
        const int4 x41 = reinterpret_cast<const int4 *>(s_x1[xb + 3].qs)[is];

        auto step4_t2 = [](uint32_t ql0, uint32_t ql1, uint32_t qh,
                           int a1, int a2, int a3, int a4,
                           int b1, int b2, int b3, int b4,
                           int & sa1, int & sa2, int & sa3, int & sa4,
                           int & sb1, int & sb2, int & sb3, int & sb4) {
            const int q1 = static_cast<int>((ql0 & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4));
            const int q2 = static_cast<int>((ql1 & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2));
            const int q3 = static_cast<int>(((ql0 >> 4) & 0x0F0F0F0Fu) | (qh & 0x30303030u));
            const int q4 = static_cast<int>(((ql1 >> 4) & 0x0F0F0F0Fu) | ((qh >> 2) & 0x30303030u));
            sa1 = dp4a_s32(q1, a1, sa1); sb1 = dp4a_s32(q1, b1, sb1);
            sa2 = dp4a_s32(q2, a2, sa2); sb2 = dp4a_s32(q2, b2, sb2);
            sa3 = dp4a_s32(q3, a3, sa3); sb3 = dp4a_s32(q3, b3, sb3);
            sa4 = dp4a_s32(q4, a4, sa4); sb4 = dp4a_s32(q4, b4, sb4);
        };

        int sa1 = 0, sa2 = 0, sa3 = 0, sa4 = 0;
        int sb1 = 0, sb2 = 0, sb3 = 0, sb4 = 0;
        step4_t2(ql0_v.x, ql1_v.x, qh_v.x, x10.x, x20.x, x30.x, x40.x, x11.x, x21.x, x31.x, x41.x, sa1, sa2, sa3, sa4, sb1, sb2, sb3, sb4);
        step4_t2(ql0_v.y, ql1_v.y, qh_v.y, x10.y, x20.y, x30.y, x40.y, x11.y, x21.y, x31.y, x41.y, sa1, sa2, sa3, sa4, sb1, sb2, sb3, sb4);
        step4_t2(ql0_v.z, ql1_v.z, qh_v.z, x10.z, x20.z, x30.z, x40.z, x11.z, x21.z, x31.z, x41.z, sa1, sa2, sa3, sa4, sb1, sb2, sb3, sb4);
        step4_t2(ql0_v.w, ql1_v.w, qh_v.w, x10.w, x20.w, x30.w, x40.w, x11.w, x21.w, x31.w, x41.w, sa1, sa2, sa3, sa4, sb1, sb2, sb3, sb4);

        const int sxa0 = is ? s_x0[xb + 0].s1 : s_x0[xb + 0].s0;
        const int sxa1 = is ? s_x0[xb + 1].s1 : s_x0[xb + 1].s0;
        const int sxa2 = is ? s_x0[xb + 2].s1 : s_x0[xb + 2].s0;
        const int sxa3 = is ? s_x0[xb + 3].s1 : s_x0[xb + 3].s0;
        acc0 += dw * (
            (sc0 * s_x0[xb + 0].d) * static_cast<float>(sa1 - 32 * sxa0) +
            (sc2 * s_x0[xb + 1].d) * static_cast<float>(sa2 - 32 * sxa1) +
            (sc4 * s_x0[xb + 2].d) * static_cast<float>(sa3 - 32 * sxa2) +
            (sc6 * s_x0[xb + 3].d) * static_cast<float>(sa4 - 32 * sxa3));

        const int sxb0 = is ? s_x1[xb + 0].s1 : s_x1[xb + 0].s0;
        const int sxb1 = is ? s_x1[xb + 1].s1 : s_x1[xb + 1].s0;
        const int sxb2 = is ? s_x1[xb + 2].s1 : s_x1[xb + 2].s0;
        const int sxb3 = is ? s_x1[xb + 3].s1 : s_x1[xb + 3].s0;
        acc1 += dw * (
            (sc0 * s_x1[xb + 0].d) * static_cast<float>(sb1 - 32 * sxb0) +
            (sc2 * s_x1[xb + 1].d) * static_cast<float>(sb2 - 32 * sxb1) +
            (sc4 * s_x1[xb + 2].d) * static_cast<float>(sb3 - 32 * sxb2) +
            (sc6 * s_x1[xb + 3].d) * static_cast<float>(sb4 - 32 * sxb3));
    }

    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if (lane == 0) {
        Y[row]     = acc0;
        Y[M + row] = acc1;
    }
}

// Q6_K SoA dequant -> F16 / F32 for prefill cuBLAS HGEMM / SGEMM!
template <typename TDst>
__global__ void dequant_q6_k_soa_kernel(
    const uint8_t *  __restrict__ W_ql,
    const uint8_t *  __restrict__ W_qh,
    const int8_t *   __restrict__ W_sc,
    const uint16_t * __restrict__ W_d,
    TDst *           __restrict__ dst,
    int64_t n_blocks) {

    const int64_t bi = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (bi >= n_blocks) return;
    const int l  = threadIdx.x;
    const int is = l >> 4;
    const float d = half_bits_to_f32(__ldg(&W_d[bi]));
    const uint8_t * ql = W_ql + bi * 128;
    const uint8_t * qh = W_qh + bi * 64;
    const int8_t *  sc = W_sc + bi * 16;
    TDst * out = dst + bi * 256;

    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint8_t ql0 = __ldg(&ql[h * 64 + l]);
        const uint8_t ql1 = __ldg(&ql[h * 64 + l + 32]);
        const uint8_t qhv = __ldg(&qh[h * 32 + l]);
        const int8_t q1 = static_cast<int8_t>((ql0 & 0xF) | (((qhv >> 0) & 3) << 4)) - 32;
        const int8_t q2 = static_cast<int8_t>((ql1 & 0xF) | (((qhv >> 2) & 3) << 4)) - 32;
        const int8_t q3 = static_cast<int8_t>((ql0 >> 4)  | (((qhv >> 4) & 3) << 4)) - 32;
        const int8_t q4 = static_cast<int8_t>((ql1 >> 4)  | (((qhv >> 6) & 3) << 4)) - 32;
        const int base = h * 128 + l;
        const float v1 = d * __ldg(&sc[h * 8 + is + 0]) * q1;
        const float v2 = d * __ldg(&sc[h * 8 + is + 2]) * q2;
        const float v3 = d * __ldg(&sc[h * 8 + is + 4]) * q3;
        const float v4 = d * __ldg(&sc[h * 8 + is + 6]) * q4;
        if constexpr (std::is_same_v<TDst, __half>) {
            out[base +  0] = __float2half_rn(v1);
            out[base + 32] = __float2half_rn(v2);
            out[base + 64] = __float2half_rn(v3);
            out[base + 96] = __float2half_rn(v4);
        } else {
            out[base +  0] = v1;
            out[base + 32] = v2;
            out[base + 64] = v3;
            out[base + 96] = v4;
        }
    }
}

// Q6_K SoA embedding lookup -> broadcasts across hc_mult streams!
__global__ void embedding_q6_k_soa_kernel(
    const uint8_t *  __restrict__ W_ql,
    const uint8_t *  __restrict__ W_qh,
    const int8_t *   __restrict__ W_sc,
    const uint16_t * __restrict__ W_d,
    const int32_t *  __restrict__ tokens,
    float *          __restrict__ out,
    int n_embd, int hc_mult) {

    const int bi = blockIdx.x;
    const int t  = blockIdx.y;
    const int l  = threadIdx.x;
    const int nb = n_embd >> 8;
    if (bi >= nb) return;

    const int tok = tokens[t];
    const int64_t b = static_cast<int64_t>(tok) * nb + bi;
    const int is = l >> 4;
    const float d = half_bits_to_f32(__ldg(&W_d[b]));
    const uint8_t * ql = W_ql + b * 128;
    const uint8_t * qh = W_qh + b * 64;
    const int8_t *  sc = W_sc + b * 16;

    float w[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint8_t ql0 = __ldg(&ql[h * 64 + l]);
        const uint8_t ql1 = __ldg(&ql[h * 64 + l + 32]);
        const uint8_t qhv = __ldg(&qh[h * 32 + l]);
        const int8_t q1 = static_cast<int8_t>((ql0 & 0xF) | (((qhv >> 0) & 3) << 4)) - 32;
        const int8_t q2 = static_cast<int8_t>((ql1 & 0xF) | (((qhv >> 2) & 3) << 4)) - 32;
        const int8_t q3 = static_cast<int8_t>((ql0 >> 4)  | (((qhv >> 4) & 3) << 4)) - 32;
        const int8_t q4 = static_cast<int8_t>((ql1 >> 4)  | (((qhv >> 6) & 3) << 4)) - 32;
        w[h * 4 + 0] = d * __ldg(&sc[h * 8 + is + 0]) * q1;
        w[h * 4 + 1] = d * __ldg(&sc[h * 8 + is + 2]) * q2;
        w[h * 4 + 2] = d * __ldg(&sc[h * 8 + is + 4]) * q3;
        w[h * 4 + 3] = d * __ldg(&sc[h * 8 + is + 6]) * q4;
    }

    const int base = bi * 256 + l;
    for (int s = 0; s < hc_mult; ++s) {
        float * dst = out + (static_cast<size_t>(t) * hc_mult + s) * n_embd;
        dst[base +   0] = w[0]; dst[base +  32] = w[1];
        dst[base +  64] = w[2]; dst[base +  96] = w[3];
        dst[base + 128] = w[4]; dst[base + 160] = w[5];
        dst[base + 192] = w[6]; dst[base + 224] = w[7];
    }
}

// ============================================================================
// PART 1B: ULTRA-FAST Q5_K LDG.128 + LDS.128 + dp4a.s32 KERNELS (T=1 & T=2!)
// BlockQ5_K is 176B = 11x16B (already 16B aligned on disk!).
// 8 lanes per 256-elem block -> 4 blocks/warp -> 100% LDG.128 on hdr, qh, qs!
// ============================================================================

__device__ __forceinline__ void unpack_q5k_sc_reg(
    int idx, uint32_t s0, uint32_t s1, uint32_t s2, float & sc_out, float & m_out) {
    if (idx < 4) {
        const int sh = 8 * idx;
        sc_out = static_cast<float>((s0 >> sh) & 63u);
        m_out  = static_cast<float>((s1 >> sh) & 63u);
    } else {
        const int sh = 8 * (idx - 4);
        const uint32_t b8 = (s2 >> sh) & 0xFFu;
        const uint32_t b0 = (s0 >> sh) & 0xFFu;
        const uint32_t b4 = (s1 >> sh) & 0xFFu;
        sc_out = static_cast<float>((b8 & 0xFu) | ((b0 >> 6) << 4));
        m_out  = static_cast<float>((b8 >> 4)   | ((b4 >> 6) << 4));
    }
}

template <bool T2>
__global__ void gemv_q5_k_ldg128_kernel(
    const BlockQ5_K *     __restrict__ W,
    const BlockQ8_1_Act * __restrict__ X,
    float *               __restrict__ Y,
    int M, int K) {

    extern __shared__ BlockQ8_1_Act s_x[];
    const int t        = T2 ? 0 : blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;

    const BlockQ8_1_Act * x_in = T2 ? X : (X + static_cast<size_t>(t) * K_blocks);
    const int n_i4 = ((T2 ? 2 : 1) * K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(x_in);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const BlockQ8_1_Act * s_x0 = s_x;
    const BlockQ8_1_Act * s_x1 = T2 ? (s_x + K_blocks) : nullptr;

    const int lane        = threadIdx.x;
    const int blk_in_warp = lane >> 3; // 0..3 (4 blocks/warp!)
    const int sub         = lane & 7;  // 0..7 (1 int4 of qs = 16 bytes)
    const int j           = sub >> 1;  // 0..3 (which 64-elem group)
    const int is          = sub & 1;   // 0 or 1 (which 16-elem half)
    const int sh1         = 2 * j;
    const int sh2         = 2 * j + 1;

    const BlockQ5_K * w_row = W + static_cast<size_t>(row) * nb;
    float acc0 = 0.0f, acc1 = 0.0f;

    for (int bi = blk_in_warp; bi < nb; bi += 4) {
        const BlockQ5_K & b = w_row[bi];
        const int4 hdr  = __ldg(reinterpret_cast<const int4 *>(&b));
        const int4 qh_v = __ldg(&reinterpret_cast<const int4 *>(b.qh)[is]);
        const int4 ql_v = __ldg(&reinterpret_cast<const int4 *>(b.qs)[sub]);

        const float d    = half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) & 0xFFFFu));
        const float dmin = half_bits_to_f32(static_cast<uint16_t>(static_cast<uint32_t>(hdr.x) >> 16));
        float sc1, m1, sc2, m2;
        unpack_q5k_sc_reg(2 * j + 0, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc1, m1);
        unpack_q5k_sc_reg(2 * j + 1, static_cast<uint32_t>(hdr.y), static_cast<uint32_t>(hdr.z), static_cast<uint32_t>(hdr.w), sc2, m2);
        const float d1 = d * sc1, mn1 = dmin * m1;
        const float d2 = d * sc2, mn2 = dmin * m2;

        const int q1x = static_cast<int>((static_cast<uint32_t>(ql_v.x) & 0x0F0F0F0Fu) | (((static_cast<uint32_t>(qh_v.x) >> sh1) & 0x01010101u) << 4));
        const int q2x = static_cast<int>(((static_cast<uint32_t>(ql_v.x) >> 4) & 0x0F0F0F0Fu) | (((static_cast<uint32_t>(qh_v.x) >> sh2) & 0x01010101u) << 4));
        const int q1y = static_cast<int>((static_cast<uint32_t>(ql_v.y) & 0x0F0F0F0Fu) | (((static_cast<uint32_t>(qh_v.y) >> sh1) & 0x01010101u) << 4));
        const int q2y = static_cast<int>(((static_cast<uint32_t>(ql_v.y) >> 4) & 0x0F0F0F0Fu) | (((static_cast<uint32_t>(qh_v.y) >> sh2) & 0x01010101u) << 4));
        const int q1z = static_cast<int>((static_cast<uint32_t>(ql_v.z) & 0x0F0F0F0Fu) | (((static_cast<uint32_t>(qh_v.z) >> sh1) & 0x01010101u) << 4));
        const int q2z = static_cast<int>(((static_cast<uint32_t>(ql_v.z) >> 4) & 0x0F0F0F0Fu) | (((static_cast<uint32_t>(qh_v.z) >> sh2) & 0x01010101u) << 4));
        const int q1w = static_cast<int>((static_cast<uint32_t>(ql_v.w) & 0x0F0F0F0Fu) | (((static_cast<uint32_t>(qh_v.w) >> sh1) & 0x01010101u) << 4));
        const int q2w = static_cast<int>(((static_cast<uint32_t>(ql_v.w) >> 4) & 0x0F0F0F0Fu) | (((static_cast<uint32_t>(qh_v.w) >> sh2) & 0x01010101u) << 4));

        const int xb = bi * 8 + 2 * j;
        {
            const int4 x1 = reinterpret_cast<const int4 *>(s_x0[xb + 0].qs)[is];
            const int4 x2 = reinterpret_cast<const int4 *>(s_x0[xb + 1].qs)[is];
            int sum1 = dp4a_s32(q1x, x1.x, dp4a_s32(q1y, x1.y, dp4a_s32(q1z, x1.z, dp4a_s32(q1w, x1.w, 0))));
            int sum2 = dp4a_s32(q2x, x2.x, dp4a_s32(q2y, x2.y, dp4a_s32(q2z, x2.z, dp4a_s32(q2w, x2.w, 0))));
            const int sx1 = is ? s_x0[xb + 0].s1 : s_x0[xb + 0].s0;
            const int sx2 = is ? s_x0[xb + 1].s1 : s_x0[xb + 1].s0;
            acc0 += s_x0[xb + 0].d * (d1 * static_cast<float>(sum1) - mn1 * static_cast<float>(sx1))
                  + s_x0[xb + 1].d * (d2 * static_cast<float>(sum2) - mn2 * static_cast<float>(sx2));
        }
        if constexpr (T2) {
            const int4 x1 = reinterpret_cast<const int4 *>(s_x1[xb + 0].qs)[is];
            const int4 x2 = reinterpret_cast<const int4 *>(s_x1[xb + 1].qs)[is];
            int sum1 = dp4a_s32(q1x, x1.x, dp4a_s32(q1y, x1.y, dp4a_s32(q1z, x1.z, dp4a_s32(q1w, x1.w, 0))));
            int sum2 = dp4a_s32(q2x, x2.x, dp4a_s32(q2y, x2.y, dp4a_s32(q2z, x2.z, dp4a_s32(q2w, x2.w, 0))));
            const int sx1 = is ? s_x1[xb + 0].s1 : s_x1[xb + 0].s0;
            const int sx2 = is ? s_x1[xb + 1].s1 : s_x1[xb + 1].s0;
            acc1 += s_x1[xb + 0].d * (d1 * static_cast<float>(sum1) - mn1 * static_cast<float>(sx1))
                  + s_x1[xb + 1].d * (d2 * static_cast<float>(sum2) - mn2 * static_cast<float>(sx2));
        }
    }

    acc0 = warp_sum(acc0);
    if constexpr (T2) acc1 = warp_sum(acc1);
    if (lane == 0) {
        Y[static_cast<size_t>(t) * M + row] = acc0;
        if constexpr (T2) Y[M + row] = acc1;
    }
}

// ============================================================================
// PART 2: FAST dp4a.s32 + SHARED-MEMORY BlockQ8_1_Act FOR IQ2_S / IQ3_S / IQ4_XS / Q2_K / Q3_K
// ============================================================================

template <DType DT>
__device__ __forceinline__ float dot_kblock_q8_1(
    const void *          __restrict__ blk_ptr,
    const BlockQ8_1_Act * __restrict__ x_blks, // 8 BlockQ8_1_Act (256 elems) in smem/global
    int lane);

template <>
__device__ __forceinline__ float dot_kblock_q8_1<DType::IQ2_S>(
    const void * __restrict__ blk_ptr, const BlockQ8_1_Act * __restrict__ x_blks, int lane) {
    const auto & b = *static_cast<const BlockIQ2_S *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2; // 0..7
    const int l    = lane & 3;  // 0..3
    const uint8_t scb = __ldg(&b.scales[ib32]);
    const int sc4 = (l >= 2) ? (scb >> 4) : (scb & 0xF);
    const float dl = d * (0.5f + static_cast<float>(sc4)) * 0.25f;

    const uint8_t qs_b = __ldg(&b.qs[lane]);
    const uint8_t sg_b = __ldg(&b.qs[32 + lane]);
    const uint8_t qh_b = __ldg(&b.qh[ib32]);
    const uint32_t gidx = static_cast<uint32_t>(qs_b) | ((static_cast<uint32_t>(qh_b) << (8 - 2 * l)) & 0x300u);
    const uint64_t g64  = __ldg(&iq2s_grid_d[gidx]);

    const int w0 = apply_signs4(static_cast<uint32_t>(g64),       sg_b & 0xFu);
    const int w1 = apply_signs4(static_cast<uint32_t>(g64 >> 32), sg_b >> 4);
    const int32_t * x32 = reinterpret_cast<const int32_t *>(x_blks[ib32].qs) + (l << 1);
    const int sumi = dp4a_s32(w0, x32[0], dp4a_s32(w1, x32[1], 0));
    return (dl * x_blks[ib32].d) * static_cast<float>(sumi);
}

template <>
__device__ __forceinline__ float dot_kblock_q8_1<DType::IQ3_S>(
    const void * __restrict__ blk_ptr, const BlockQ8_1_Act * __restrict__ x_blks, int lane) {
    const auto & b = *static_cast<const BlockIQ3_S *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint8_t scb = __ldg(&b.scales[ib32 >> 1]);
    const int sc4 = (ib32 & 1) ? (scb >> 4) : (scb & 0xF);
    const float db = d * static_cast<float>(1 + 2 * sc4);

    const uint8_t  qh  = __ldg(&b.qh[ib32]);
    // b.qs is at offset 2 in 110-byte struct -> always 2-byte aligned -> 1 coalesced 16-bit load!
    const uint16_t q01 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs)[lane]);
    const uint32_t q0  = q01 & 0xFFu;
    const uint32_t q1  = q01 >> 8;
    const uint8_t  sg  = __ldg(&b.signs[lane]);

    const uint32_t g1 = __ldg(&iq3s_grid_d[q0 | ((static_cast<uint32_t>(qh) << (8 - 2 * l)) & 256u)]);
    const uint32_t g2 = __ldg(&iq3s_grid_d[q1 | ((static_cast<uint32_t>(qh) << (7 - 2 * l)) & 256u)]);
    const int w0 = apply_signs4(g1, sg & 0xFu);
    const int w1 = apply_signs4(g2, sg >> 4);

    const int32_t * x32 = reinterpret_cast<const int32_t *>(x_blks[ib32].qs) + (l << 1);
    const int sumi = dp4a_s32(w0, x32[0], dp4a_s32(w1, x32[1], 0));
    return (db * x_blks[ib32].d) * static_cast<float>(sumi);
}

template <>
__device__ __forceinline__ float dot_kblock_q8_1<DType::IQ4_XS>(
    const void * __restrict__ blk_ptr, const BlockQ8_1_Act * __restrict__ x_blks, int lane) {
    const auto & b = *static_cast<const BlockIQ4_XS *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const uint16_t sh = __ldg(&b.scales_h);
    const int ib   = lane >> 2;
    const int sub4 = lane & 3;
    const uint8_t sl = __ldg(&b.scales_l[ib >> 1]);
    const int ls = ((sl >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = d * static_cast<float>(ls - 32);

    // b.qs is at offset 8 in 136-byte struct -> always 8-byte aligned -> 1 coalesced 32-bit load!
    const uint32_t q4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qs)[ib * 4 + sub4]);
    const uint32_t w_lo =
          static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >>  0) & 0xFu])))
        | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >>  8) & 0xFu]))) << 8)
        | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 16) & 0xFu]))) << 16)
        | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 24) & 0xFu]))) << 24);
    const uint32_t w_hi =
          static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >>  4) & 0xFu])))
        | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 12) & 0xFu]))) << 8)
        | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 20) & 0xFu]))) << 16)
        | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 28) & 0xFu]))) << 24);

    const int32_t * x32 = reinterpret_cast<const int32_t *>(x_blks[ib].qs);
    const int sumi = dp4a_s32(static_cast<int>(w_lo), x32[sub4], dp4a_s32(static_cast<int>(w_hi), x32[4 + sub4], 0));
    return (dl * x_blks[ib].d) * static_cast<float>(sumi);
}

template <>
__device__ __forceinline__ float dot_kblock_q8_1<DType::Q2_K>(
    const void * __restrict__ blk_ptr, const BlockQ8_1_Act * __restrict__ x_blks, int lane) {
    const auto & b = *static_cast<const BlockQ2_K *>(blk_ptr);
    const float d    = half_bits_to_f32(__ldg(&b.d));
    const float dmin = half_bits_to_f32(__ldg(&b.dmin));
    const int grp   = lane >> 1;
    const int half  = (lane & 1) * 8;
    const int n128  = grp >> 3;
    const int j     = (grp & 7) >> 1;
    const int sub   = grp & 1;
    const int shift = 2 * j;
    const uint8_t sc = __ldg(&b.scales[grp]);
    const float dl = d    * static_cast<float>(sc & 0xF);
    const float ml = dmin * static_cast<float>(sc >> 4);
    const int q_base = n128 * 32 + sub * 16 + half;

    uint32_t w0 = 0, w1 = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        w0 |= static_cast<uint32_t>((__ldg(&b.qs[q_base + k])     >> shift) & 3u) << (8 * k);
        w1 |= static_cast<uint32_t>((__ldg(&b.qs[q_base + 4 + k]) >> shift) & 3u) << (8 * k);
    }
    const int xb_idx = grp >> 1;
    const int32_t * x32 = reinterpret_cast<const int32_t *>(x_blks[xb_idx].qs) + ((grp & 1) * 4 + (lane & 1) * 2);
    const int sum_q = dp4a_s32(static_cast<int>(w0), x32[0], dp4a_s32(static_cast<int>(w1), x32[1], 0));
    const int sum_x = dp4a_s32(0x01010101,           x32[0], dp4a_s32(0x01010101,           x32[1], 0));
    return x_blks[xb_idx].d * (dl * static_cast<float>(sum_q) - ml * static_cast<float>(sum_x));
}

template <>
__device__ __forceinline__ float dot_kblock_q8_1<DType::Q3_K>(
    const void * __restrict__ blk_ptr, const BlockQ8_1_Act * __restrict__ x_blks, int lane) {
    const auto & b = *static_cast<const BlockQ3_K *>(blk_ptr);
    const float d_all = half_bits_to_f32(__ldg(&b.d));

    uint32_t aux[4];
    const uint8_t * sc = b.scales;
    aux[0] = static_cast<uint32_t>(__ldg(&sc[0])) | (static_cast<uint32_t>(__ldg(&sc[1])) << 8) |
             (static_cast<uint32_t>(__ldg(&sc[2])) << 16) | (static_cast<uint32_t>(__ldg(&sc[3])) << 24);
    aux[1] = static_cast<uint32_t>(__ldg(&sc[4])) | (static_cast<uint32_t>(__ldg(&sc[5])) << 8) |
             (static_cast<uint32_t>(__ldg(&sc[6])) << 16) | (static_cast<uint32_t>(__ldg(&sc[7])) << 24);
    uint32_t tmp = static_cast<uint32_t>(__ldg(&sc[8])) | (static_cast<uint32_t>(__ldg(&sc[9])) << 8) |
                   (static_cast<uint32_t>(__ldg(&sc[10])) << 16) | (static_cast<uint32_t>(__ldg(&sc[11])) << 24);
    constexpr uint32_t kmask1 = 0x03030303u;
    constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    const int grp   = lane >> 1;
    const int half  = (lane & 1) * 8;
    const int n128  = grp >> 3;
    const int j     = (grp & 7) >> 1;
    const int sub   = grp & 1;
    const int shift = 2 * j;
    const uint8_t m = static_cast<uint8_t>(1u << (n128 * 4 + j));
    const int8_t sc_val = static_cast<int8_t>((aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu);
    const float dl = d_all * static_cast<float>(sc_val - 32);

    const int q_base  = n128 * 32 + sub * 16 + half;
    const int hm_base = sub * 16 + half;
    uint32_t w0 = 0, w1 = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int8_t v0 = static_cast<int8_t>((__ldg(&b.qs[q_base + k]) >> shift) & 3) -
                          ((__ldg(&b.hmask[hm_base + k]) & m) ? 0 : 4);
        const int8_t v1 = static_cast<int8_t>((__ldg(&b.qs[q_base + 4 + k]) >> shift) & 3) -
                          ((__ldg(&b.hmask[hm_base + 4 + k]) & m) ? 0 : 4);
        w0 |= static_cast<uint32_t>(static_cast<uint8_t>(v0)) << (8 * k);
        w1 |= static_cast<uint32_t>(static_cast<uint8_t>(v1)) << (8 * k);
    }
    const int xb_idx = grp >> 1;
    const int32_t * x32 = reinterpret_cast<const int32_t *>(x_blks[xb_idx].qs) + ((grp & 1) * 4 + (lane & 1) * 2);
    const int sumi = dp4a_s32(static_cast<int>(w0), x32[0], dp4a_s32(static_cast<int>(w1), x32[1], 0));
    return (dl * x_blks[xb_idx].d) * static_cast<float>(sumi);
}

template <>
__device__ __forceinline__ float dot_kblock_q8_1<DType::Q4_K>(
    const void * __restrict__ blk_ptr, const BlockQ8_1_Act * __restrict__ x_blks, int lane) {
    const auto & b = *static_cast<const BlockQ4_K *>(blk_ptr);
    const float d    = half_bits_to_f32(__ldg(&b.d));
    const float dmin = half_bits_to_f32(__ldg(&b.dmin));
    const int j   = lane >> 3; // 0..3 (four 64-elem groups)
    const int sub = lane & 7;  // 0..7 (4 bytes of qs -> 4 low + 4 high nibbles)
    uint8_t sc1, m1, sc2, m2;
    get_scale_min_k4(2 * j + 0, b.scales, sc1, m1);
    get_scale_min_k4(2 * j + 1, b.scales, sc2, m2);
    const float d1 = d * static_cast<float>(sc1), mn1 = dmin * static_cast<float>(m1);
    const float d2 = d * static_cast<float>(sc2), mn2 = dmin * static_cast<float>(m2);

    const uint32_t ql4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qs)[j * 8 + sub]);
    const int q1 = static_cast<int>(ql4 & 0x0F0F0F0Fu);
    const int q2 = static_cast<int>((ql4 >> 4) & 0x0F0F0F0Fu);
    const int x1 = reinterpret_cast<const int32_t *>(x_blks[2 * j + 0].qs)[sub];
    const int x2 = reinterpret_cast<const int32_t *>(x_blks[2 * j + 1].qs)[sub];
    float acc = (d1 * x_blks[2 * j + 0].d) * static_cast<float>(dp4a_s32(q1, x1, 0))
              + (d2 * x_blks[2 * j + 1].d) * static_cast<float>(dp4a_s32(q2, x2, 0));
    if (sub == 0) {
        acc -= mn1 * x_blks[2 * j + 0].s_all + mn2 * x_blks[2 * j + 1].s_all;
    }
    return acc;
}

template <>
__device__ __forceinline__ float dot_kblock_q8_1<DType::Q5_K>(
    const void * __restrict__ blk_ptr, const BlockQ8_1_Act * __restrict__ x_blks, int lane) {
    const auto & b = *static_cast<const BlockQ5_K *>(blk_ptr);
    const float d    = half_bits_to_f32(__ldg(&b.d));
    const float dmin = half_bits_to_f32(__ldg(&b.dmin));
    const int j   = lane >> 3;
    const int sub = lane & 7;
    uint8_t sc1, m1, sc2, m2;
    get_scale_min_k4(2 * j + 0, b.scales, sc1, m1);
    get_scale_min_k4(2 * j + 1, b.scales, sc2, m2);
    const float d1 = d * static_cast<float>(sc1), mn1 = dmin * static_cast<float>(m1);
    const float d2 = d * static_cast<float>(sc2), mn2 = dmin * static_cast<float>(m2);

    const uint32_t ql4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qs)[j * 8 + sub]);
    const uint32_t qh4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qh)[sub]);
    const uint32_t hi1 = ((qh4 >> (2 * j + 0)) & 0x01010101u) << 4;
    const uint32_t hi2 = ((qh4 >> (2 * j + 1)) & 0x01010101u) << 4;
    const int q1 = static_cast<int>((ql4 & 0x0F0F0F0Fu) | hi1);
    const int q2 = static_cast<int>(((ql4 >> 4) & 0x0F0F0F0Fu) | hi2);
    const int x1 = reinterpret_cast<const int32_t *>(x_blks[2 * j + 0].qs)[sub];
    const int x2 = reinterpret_cast<const int32_t *>(x_blks[2 * j + 1].qs)[sub];
    float acc = (d1 * x_blks[2 * j + 0].d) * static_cast<float>(dp4a_s32(q1, x1, 0))
              + (d2 * x_blks[2 * j + 1].d) * static_cast<float>(dp4a_s32(q2, x2, 0));
    if (sub == 0) {
        acc -= mn1 * x_blks[2 * j + 0].s_all + mn2 * x_blks[2 * j + 1].s_all;
    }
    return acc;
}

template <>
__device__ __forceinline__ float dot_kblock_q8_1<DType::IQ2_XXS>(
    const void * __restrict__ blk_ptr, const BlockQ8_1_Act * __restrict__ x_blks, int lane) {
    const auto & b = *static_cast<const BlockIQ2_XXS *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint16_t * q2 = b.qs + 4 * ib32;
    const uint16_t w_idx = __ldg(&q2[l >> 1]);
    const uint32_t gidx  = (l & 1) ? (w_idx >> 8) : (w_idx & 0xFFu);
    const uint32_t aux32 = static_cast<uint32_t>(__ldg(&q2[2])) | (static_cast<uint32_t>(__ldg(&q2[3])) << 16);
    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.25f;
    const uint8_t  sg  = __ldg(&ksigns_iq2xs_d[(aux32 >> (7 * l)) & 127u]);
    const uint64_t g64 = __ldg(&iq2xxs_grid_d[gidx]);
    const int w0 = apply_signs4(static_cast<uint32_t>(g64),       sg & 0xFu);
    const int w1 = apply_signs4(static_cast<uint32_t>(g64 >> 32), sg >> 4);
    const int32_t * x32 = reinterpret_cast<const int32_t *>(x_blks[ib32].qs) + (l << 1);
    const int sumi = dp4a_s32(w0, x32[0], dp4a_s32(w1, x32[1], 0));
    return (db * x_blks[ib32].d) * static_cast<float>(sumi);
}

template <>
__device__ __forceinline__ float dot_kblock_q8_1<DType::IQ3_XXS>(
    const void * __restrict__ blk_ptr, const BlockQ8_1_Act * __restrict__ x_blks, int lane) {
    const auto & b = *static_cast<const BlockIQ3_XXS *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint16_t q01 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs)[lane]);
    const uint16_t ss0 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs + 64)[ib32 * 2 + 0]);
    const uint16_t ss1 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs + 64)[ib32 * 2 + 1]);
    const uint32_t aux32 = static_cast<uint32_t>(ss0) | (static_cast<uint32_t>(ss1) << 16);
    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
    const uint8_t  sg = __ldg(&ksigns_iq2xs_d[(aux32 >> (7 * l)) & 127u]);
    const uint32_t g1 = __ldg(&iq3xxs_grid_d[q01 & 0xFFu]);
    const uint32_t g2 = __ldg(&iq3xxs_grid_d[q01 >> 8]);
    const int w0 = apply_signs4(g1, sg & 0xFu);
    const int w1 = apply_signs4(g2, sg >> 4);
    const int32_t * x32 = reinterpret_cast<const int32_t *>(x_blks[ib32].qs) + (l << 1);
    const int sumi = dp4a_s32(w0, x32[0], dp4a_s32(w1, x32[1], 0));
    return (db * x_blks[ib32].d) * static_cast<float>(sumi);
}

// ============================================================================
// PART 3: Reference float decoder (for prefill F16 dequant -> cuBLAS HGEMM)
// ============================================================================

template <DType DT>
__device__ __forceinline__ void decode_8_elems(
    const void * __restrict__ blk_ptr, int lane, float w[8], int idx[8]);

template <>
__device__ __forceinline__ void decode_8_elems<DType::IQ4_XS>(
    const void * __restrict__ blk_ptr, int lane, float w[8], int idx[8]) {
    const auto & b = *static_cast<const BlockIQ4_XS *>(blk_ptr);
    const float d   = half_bits_to_f32(__ldg(&b.d));
    const uint16_t sh = __ldg(&b.scales_h);
    const int ib  = lane >> 2;
    const int sub = (lane & 3) << 2;
    const uint8_t sl = __ldg(&b.scales_l[ib >> 1]);
    const int ls = ((sl >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = d * static_cast<float>(ls - 32);
    const int base_q = ib * 16 + sub;
    const int base_y = ib * 32 + sub;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint8_t qb = __ldg(&b.qs[base_q + j]);
        w[j]     = dl * static_cast<float>(kvalues_iq4nl_d[qb & 0xF]);
        idx[j]   = base_y + j;
        w[j + 4] = dl * static_cast<float>(kvalues_iq4nl_d[qb >> 4]);
        idx[j+4] = base_y + 16 + j;
    }
}

template <>
__device__ __forceinline__ void decode_8_elems<DType::IQ3_S>(
    const void * __restrict__ blk_ptr, int lane, float w[8], int idx[8]) {
    const auto & b = *static_cast<const BlockIQ3_S *>(blk_ptr);
    const float d = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint8_t scb = __ldg(&b.scales[ib32 >> 1]);
    const int sc4 = (ib32 & 1) ? (scb >> 4) : (scb & 0xF);
    const float db = d * static_cast<float>(1 + 2 * sc4);
    const uint8_t qh = __ldg(&b.qh[ib32]);
    const uint8_t q0 = __ldg(&b.qs[lane * 2 + 0]);
    const uint8_t q1 = __ldg(&b.qs[lane * 2 + 1]);
    const uint8_t sg = __ldg(&b.signs[lane]);
    const uint32_t g1 = __ldg(&iq3s_grid_d[q0 | ((static_cast<uint32_t>(qh) << (8 - 2 * l)) & 256u)]);
    const uint32_t g2 = __ldg(&iq3s_grid_d[q1 | ((static_cast<uint32_t>(qh) << (7 - 2 * l)) & 256u)]);
    const int base_y = lane * 8;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j]     = db * static_cast<float>((g1 >> (8 * j)) & 0xFFu) * ((sg & (1u << j))       ? -1.0f : 1.0f);
        idx[j]   = base_y + j;
        w[j + 4] = db * static_cast<float>((g2 >> (8 * j)) & 0xFFu) * ((sg & (1u << (j + 4))) ? -1.0f : 1.0f);
        idx[j+4] = base_y + 4 + j;
    }
}

template <>
__device__ __forceinline__ void decode_8_elems<DType::IQ2_S>(
    const void * __restrict__ blk_ptr, int lane, float w[8], int idx[8]) {
    const auto & b = *static_cast<const BlockIQ2_S *>(blk_ptr);
    const float d = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint8_t scb = __ldg(&b.scales[ib32]);
    const int sc4 = (l >= 2) ? (scb >> 4) : (scb & 0xF);
    const float dl = d * (0.5f + static_cast<float>(sc4)) * 0.25f;
    const uint8_t qs_b = __ldg(&b.qs[lane]);
    const uint8_t sg_b = __ldg(&b.qs[32 + lane]);
    const uint8_t qh_b = __ldg(&b.qh[ib32]);
    const uint32_t gidx = static_cast<uint32_t>(qs_b) | ((static_cast<uint32_t>(qh_b) << (8 - 2 * l)) & 0x300u);
    const uint64_t g64  = __ldg(&iq2s_grid_d[gidx]);
    const int base_y = lane * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        w[j]   = dl * static_cast<float>((g64 >> (8 * j)) & 0xFFu) * ((sg_b & (1u << j)) ? -1.0f : 1.0f);
        idx[j] = base_y + j;
    }
}

template <>
__device__ __forceinline__ void decode_8_elems<DType::Q2_K>(
    const void * __restrict__ blk_ptr, int lane, float w[8], int idx[8]) {
    const auto & b = *static_cast<const BlockQ2_K *>(blk_ptr);
    const float d    = half_bits_to_f32(__ldg(&b.d));
    const float dmin = half_bits_to_f32(__ldg(&b.dmin));
    const int grp = lane >> 1; const int half = (lane & 1) * 8;
    const int n128 = grp >> 3; const int j = (grp & 7) >> 1; const int sub = grp & 1;
    const int shift = 2 * j;
    const uint8_t sc = __ldg(&b.scales[grp]);
    const float dl = d * static_cast<float>(sc & 0xF);
    const float ml = dmin * static_cast<float>(sc >> 4);
    const int q_base = n128 * 32 + sub * 16 + half;
    const int y_base = grp * 16 + half;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        w[k]   = dl * static_cast<float>((__ldg(&b.qs[q_base + k]) >> shift) & 3) - ml;
        idx[k] = y_base + k;
    }
}

template <>
__device__ __forceinline__ void decode_8_elems<DType::Q3_K>(
    const void * __restrict__ blk_ptr, int lane, float w[8], int idx[8]) {
    const auto & b = *static_cast<const BlockQ3_K *>(blk_ptr);
    const float d_all = half_bits_to_f32(__ldg(&b.d));
    uint32_t aux[4];
    const uint8_t * sc = b.scales;
    aux[0] = static_cast<uint32_t>(__ldg(&sc[0])) | (static_cast<uint32_t>(__ldg(&sc[1])) << 8) |
             (static_cast<uint32_t>(__ldg(&sc[2])) << 16) | (static_cast<uint32_t>(__ldg(&sc[3])) << 24);
    aux[1] = static_cast<uint32_t>(__ldg(&sc[4])) | (static_cast<uint32_t>(__ldg(&sc[5])) << 8) |
             (static_cast<uint32_t>(__ldg(&sc[6])) << 16) | (static_cast<uint32_t>(__ldg(&sc[7])) << 24);
    uint32_t tmp = static_cast<uint32_t>(__ldg(&sc[8])) | (static_cast<uint32_t>(__ldg(&sc[9])) << 8) |
                   (static_cast<uint32_t>(__ldg(&sc[10])) << 16) | (static_cast<uint32_t>(__ldg(&sc[11])) << 24);
    constexpr uint32_t kmask1 = 0x03030303u;
    constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int grp = lane >> 1; const int half = (lane & 1) * 8;
    const int n128 = grp >> 3; const int j = (grp & 7) >> 1; const int sub = grp & 1;
    const int shift = 2 * j;
    const uint8_t m = static_cast<uint8_t>(1u << (n128 * 4 + j));
    const int8_t sc_val = static_cast<int8_t>((aux[grp >> 2] >> (8 * (grp & 3))) & 0xFFu);
    const float dl = d_all * static_cast<float>(sc_val - 32);
    const int q_base  = n128 * 32 + sub * 16 + half;
    const int hm_base = sub * 16 + half;
    const int y_base  = grp * 16 + half;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int8_t ql = static_cast<int8_t>((__ldg(&b.qs[q_base + k]) >> shift) & 3);
        const int8_t qh = (__ldg(&b.hmask[hm_base + k]) & m) ? 0 : 4;
        w[k]   = dl * static_cast<float>(ql - qh);
        idx[k] = y_base + k;
    }
}

template <>
__device__ __forceinline__ void decode_8_elems<DType::Q4_K>(
    const void * __restrict__ blk_ptr, int lane, float w[8], int idx[8]) {
    const auto & b = *static_cast<const BlockQ4_K *>(blk_ptr);
    const float d    = half_bits_to_f32(__ldg(&b.d));
    const float dmin = half_bits_to_f32(__ldg(&b.dmin));
    const int j   = lane >> 3;
    const int sub = (lane & 7) * 4;
    uint8_t sc1, m1, sc2, m2;
    get_scale_min_k4(2 * j + 0, b.scales, sc1, m1);
    get_scale_min_k4(2 * j + 1, b.scales, sc2, m2);
    const float d1 = d * sc1, mn1 = dmin * m1;
    const float d2 = d * sc2, mn2 = dmin * m2;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint8_t qb = __ldg(&b.qs[j * 32 + sub + k]);
        w[k]     = d1 * static_cast<float>(qb & 0xF) - mn1; idx[k]     = j * 64 + sub + k;
        w[k + 4] = d2 * static_cast<float>(qb >> 4)  - mn2; idx[k + 4] = j * 64 + 32 + sub + k;
    }
}

template <>
__device__ __forceinline__ void decode_8_elems<DType::Q5_K>(
    const void * __restrict__ blk_ptr, int lane, float w[8], int idx[8]) {
    const auto & b = *static_cast<const BlockQ5_K *>(blk_ptr);
    const float d    = half_bits_to_f32(__ldg(&b.d));
    const float dmin = half_bits_to_f32(__ldg(&b.dmin));
    const int j   = lane >> 3;
    const int sub = (lane & 7) * 4;
    uint8_t sc1, m1, sc2, m2;
    get_scale_min_k4(2 * j + 0, b.scales, sc1, m1);
    get_scale_min_k4(2 * j + 1, b.scales, sc2, m2);
    const float d1 = d * sc1, mn1 = dmin * m1;
    const float d2 = d * sc2, mn2 = dmin * m2;
    const uint8_t u1 = static_cast<uint8_t>(1u << (2 * j));
    const uint8_t u2 = static_cast<uint8_t>(2u << (2 * j));
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint8_t qb = __ldg(&b.qs[j * 32 + sub + k]);
        const uint8_t hb = __ldg(&b.qh[sub + k]);
        w[k]     = d1 * static_cast<float>((qb & 0xF) + ((hb & u1) ? 16 : 0)) - mn1; idx[k]     = j * 64 + sub + k;
        w[k + 4] = d2 * static_cast<float>((qb >> 4)  + ((hb & u2) ? 16 : 0)) - mn2; idx[k + 4] = j * 64 + 32 + sub + k;
    }
}

template <>
__device__ __forceinline__ void decode_8_elems<DType::IQ2_XXS>(
    const void * __restrict__ blk_ptr, int lane, float w[8], int idx[8]) {
    const auto & b = *static_cast<const BlockIQ2_XXS *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint16_t * q2 = b.qs + 4 * ib32;
    const uint16_t w_idx = __ldg(&q2[l >> 1]);
    const uint32_t gidx  = (l & 1) ? (w_idx >> 8) : (w_idx & 0xFFu);
    const uint32_t aux32 = static_cast<uint32_t>(__ldg(&q2[2])) | (static_cast<uint32_t>(__ldg(&q2[3])) << 16);
    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.25f;
    const uint8_t  sg  = __ldg(&ksigns_iq2xs_d[(aux32 >> (7 * l)) & 127u]);
    const uint64_t g64 = __ldg(&iq2xxs_grid_d[gidx]);
    const int base_y = lane * 8;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        w[k]   = db * static_cast<float>((g64 >> (8 * k)) & 0xFFu) * ((sg & (1u << k)) ? -1.0f : 1.0f);
        idx[k] = base_y + k;
    }
}

template <>
__device__ __forceinline__ void decode_8_elems<DType::IQ3_XXS>(
    const void * __restrict__ blk_ptr, int lane, float w[8], int idx[8]) {
    const auto & b = *static_cast<const BlockIQ3_XXS *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint16_t q01 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs)[lane]);
    const uint16_t ss0 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs + 64)[ib32 * 2 + 0]);
    const uint16_t ss1 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs + 64)[ib32 * 2 + 1]);
    const uint32_t aux32 = static_cast<uint32_t>(ss0) | (static_cast<uint32_t>(ss1) << 16);
    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
    const uint8_t  sg = __ldg(&ksigns_iq2xs_d[(aux32 >> (7 * l)) & 127u]);
    const uint32_t g1 = __ldg(&iq3xxs_grid_d[q01 & 0xFFu]);
    const uint32_t g2 = __ldg(&iq3xxs_grid_d[q01 >> 8]);
    const int base_y = lane * 8;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        w[k]     = db * static_cast<float>((g1 >> (8 * k)) & 0xFFu) * ((sg & (1u << k))       ? -1.0f : 1.0f);
        idx[k]   = base_y + k;
        w[k + 4] = db * static_cast<float>((g2 >> (8 * k)) & 0xFFu) * ((sg & (1u << (k + 4))) ? -1.0f : 1.0f);
        idx[k+4] = base_y + 4 + k;
    }
}

template <DType DT, typename TDst>
__global__ void dequant_kquant_kernel(
    const uint8_t * __restrict__ src, TDst * __restrict__ dst, int64_t n_blocks) {
    const int64_t bi = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (bi >= n_blocks) return;
    const int lane = threadIdx.x;
    constexpr size_t B = block_bytes_of<DT>();
    float w[8]; int idx[8];
    decode_8_elems<DT>(src + bi * B, lane, w, idx);
    TDst * out = dst + bi * 256;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if constexpr (std::is_same_v<TDst, __half>) {
            out[idx[k]] = __float2half_rn(w[k]);
        } else {
            out[idx[k]] = w[k];
        }
    }
}

// Generic shared-mem BlockQ8_1 + dp4a GEMV for Q5_K / Q4_K / IQ2_S / IQ2_XXS / IQ3_S / IQ3_XXS / IQ4_XS / Q2_K / Q3_K
template <DType DT>
__global__ void gemv_kquant_q8_1_kernel(
    const uint8_t *       __restrict__ W,
    const BlockQ8_1_Act * __restrict__ X,
    float *               __restrict__ Y,
    int M, int K) {

    extern __shared__ BlockQ8_1_Act s_x[];
    const int t        = blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;

    const BlockQ8_1_Act * x_row = X + static_cast<size_t>(t) * K_blocks;
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(x_row);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const int lane = threadIdx.x;
    constexpr size_t B = block_bytes_of<DT>();
    const uint8_t * w_row = W + static_cast<size_t>(row) * nb * B;

    float acc = 0.0f;
    #pragma unroll 2
    for (int bi = 0; bi < nb; ++bi) {
        acc += dot_kblock_q8_1<DT>(w_row + static_cast<size_t>(bi) * B, s_x + bi * 8, lane);
    }
    acc = warp_sum(acc);
    if (lane == 0) Y[static_cast<size_t>(t) * M + row] = acc;
}

// Dual-token (T == 2) weight-reuse GEMV for Q5_K / Q4_K / IQ* (MTP verification!)
template <DType DT>
__global__ void gemv_kquant_q8_1_t2_kernel(
    const uint8_t *       __restrict__ W,
    const BlockQ8_1_Act * __restrict__ X, // [2, K/32]
    float *               __restrict__ Y, // [2, M]
    int M, int K) {

    extern __shared__ BlockQ8_1_Act s_x[];
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    BlockQ8_1_Act * s_x0 = s_x;
    BlockQ8_1_Act * s_x1 = s_x + K_blocks;

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const int n_i4 = (2 * K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(X);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const int lane = threadIdx.x;
    constexpr size_t B = block_bytes_of<DT>();
    const uint8_t * w_row = W + static_cast<size_t>(row) * nb * B;

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t * bp = w_row + static_cast<size_t>(bi) * B;
        acc0 += dot_kblock_q8_1<DT>(bp, s_x0 + bi * 8, lane);
        acc1 += dot_kblock_q8_1<DT>(bp, s_x1 + bi * 8, lane);
    }
    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if (lane == 0) {
        Y[row]     = acc0;
        Y[M + row] = acc1;
    }
}

template <DType DT>
__global__ void embedding_kquant_generic_kernel(
    const uint8_t * __restrict__ table,
    const int32_t * __restrict__ tokens,
    float *         __restrict__ out,
    int n_embd, int hc_mult) {

    const int bi   = blockIdx.x;
    const int t    = blockIdx.y;
    const int lane = threadIdx.x;
    const int nb   = n_embd >> 8;
    if (bi >= nb) return;

    const int tok = tokens[t];
    constexpr size_t B = block_bytes_of<DT>();
    float w[8]; int idx[8];
    decode_8_elems<DT>(table + (static_cast<size_t>(tok) * nb + bi) * B, lane, w, idx);
    const int base = bi << 8;
    for (int s = 0; s < hc_mult; ++s) {
        float * dst = out + (static_cast<size_t>(t) * hc_mult + s) * n_embd;
        #pragma unroll
        for (int k = 0; k < 8; ++k) dst[base + idx[k]] = w[k];
    }
}

// ============================================================================
// PART 4: FAST SHARED-MEM BlockQ8_1 + dp4a MoE GATE+UP AND FUSED DOWN+SHARED
// ============================================================================

// Fast shared-memory-grid IQ2_S & IQ3_S variants (ZERO L1 cache contention with weight streams!)
__device__ __forceinline__ float dot_iq2_s_sgrid(
    const void *          __restrict__ blk_ptr,
    const BlockQ8_1_Act * __restrict__ x_blks,
    const uint64_t *      __restrict__ s_grid,
    int lane) {
    const auto & b = *static_cast<const BlockIQ2_S *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint8_t scb = __ldg(&b.scales[ib32]);
    const int sc4 = (l >= 2) ? (scb >> 4) : (scb & 0xF);
    const float dl = d * (0.5f + static_cast<float>(sc4)) * 0.25f;

    const uint8_t qs_b = __ldg(&b.qs[lane]);
    const uint8_t sg_b = __ldg(&b.qs[32 + lane]);
    const uint8_t qh_b = __ldg(&b.qh[ib32]);
    const uint32_t gidx = static_cast<uint32_t>(qs_b) | ((static_cast<uint32_t>(qh_b) << (8 - 2 * l)) & 0x300u);
    const uint64_t g64  = s_grid[gidx];

    const int w0 = apply_signs4(static_cast<uint32_t>(g64),       sg_b & 0xFu);
    const int w1 = apply_signs4(static_cast<uint32_t>(g64 >> 32), sg_b >> 4);
    const int32_t * x32 = reinterpret_cast<const int32_t *>(x_blks[ib32].qs) + (l << 1);
    const int sumi = dp4a_s32(w0, x32[0], dp4a_s32(w1, x32[1], 0));
    return (dl * x_blks[ib32].d) * static_cast<float>(sumi);
}

__device__ __forceinline__ float dot_iq3_s_sgrid(
    const void *          __restrict__ blk_ptr,
    const BlockQ8_1_Act * __restrict__ x_blks,
    const uint32_t *      __restrict__ s_grid,
    int lane) {
    const auto & b = *static_cast<const BlockIQ3_S *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint8_t scb = __ldg(&b.scales[ib32 >> 1]);
    const int sc4 = (ib32 & 1) ? (scb >> 4) : (scb & 0xF);
    const float db = d * static_cast<float>(1 + 2 * sc4);

    const uint8_t  qh  = __ldg(&b.qh[ib32]);
    const uint16_t q01 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs)[lane]);
    const uint32_t q0  = q01 & 0xFFu;
    const uint32_t q1  = q01 >> 8;
    const uint8_t  sg  = __ldg(&b.signs[lane]);

    const uint32_t g1 = s_grid[q0 | ((static_cast<uint32_t>(qh) << (8 - 2 * l)) & 256u)];
    const uint32_t g2 = s_grid[q1 | ((static_cast<uint32_t>(qh) << (7 - 2 * l)) & 256u)];
    const int w0 = apply_signs4(g1, sg & 0xFu);
    const int w1 = apply_signs4(g2, sg >> 4);

    const int32_t * x32 = reinterpret_cast<const int32_t *>(x_blks[ib32].qs) + (l << 1);
    const int sumi = dp4a_s32(w0, x32[0], dp4a_s32(w1, x32[1], 0));
    return (db * x_blks[ib32].d) * static_cast<float>(sumi);
}

__device__ __forceinline__ float dot_iq2_xxs_sgrid(
    const void *          __restrict__ blk_ptr,
    const BlockQ8_1_Act * __restrict__ x_blks,
    const uint64_t *      __restrict__ s_grid,
    int lane) {
    const auto & b = *static_cast<const BlockIQ2_XXS *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint16_t * q2 = b.qs + 4 * ib32;
    const uint16_t w_idx = __ldg(&q2[l >> 1]);
    const uint32_t gidx  = (l & 1) ? (w_idx >> 8) : (w_idx & 0xFFu);
    const uint32_t aux32 = static_cast<uint32_t>(__ldg(&q2[2])) | (static_cast<uint32_t>(__ldg(&q2[3])) << 16);
    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.25f;
    const uint8_t  sg  = __ldg(&ksigns_iq2xs_d[(aux32 >> (7 * l)) & 127u]);
    const uint64_t g64 = s_grid[gidx];
    const int w0 = apply_signs4(static_cast<uint32_t>(g64),       sg & 0xFu);
    const int w1 = apply_signs4(static_cast<uint32_t>(g64 >> 32), sg >> 4);
    const int32_t * x32 = reinterpret_cast<const int32_t *>(x_blks[ib32].qs) + (l << 1);
    const int sumi = dp4a_s32(w0, x32[0], dp4a_s32(w1, x32[1], 0));
    return (db * x_blks[ib32].d) * static_cast<float>(sumi);
}

__device__ __forceinline__ float dot_iq3_xxs_sgrid(
    const void *          __restrict__ blk_ptr,
    const BlockQ8_1_Act * __restrict__ x_blks,
    const uint32_t *      __restrict__ s_grid,
    int lane) {
    const auto & b = *static_cast<const BlockIQ3_XXS *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint16_t q01 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs)[lane]);
    const uint16_t ss0 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs + 64)[ib32 * 2 + 0]);
    const uint16_t ss1 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs + 64)[ib32 * 2 + 1]);
    const uint32_t aux32 = static_cast<uint32_t>(ss0) | (static_cast<uint32_t>(ss1) << 16);
    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
    const uint8_t  sg = __ldg(&ksigns_iq2xs_d[(aux32 >> (7 * l)) & 127u]);
    const uint32_t g1 = s_grid[q01 & 0xFFu];
    const uint32_t g2 = s_grid[q01 >> 8];
    const int w0 = apply_signs4(g1, sg & 0xFu);
    const int w1 = apply_signs4(g2, sg >> 4);
    const int32_t * x32 = reinterpret_cast<const int32_t *>(x_blks[ib32].qs) + (l << 1);
    const int sumi = dp4a_s32(w0, x32[0], dp4a_s32(w1, x32[1], 0));
    return (db * x_blks[ib32].d) * static_cast<float>(sumi);
}

template <DType DT_G, DType DT_U>
__global__ void moe_gate_up_kquant_q8_1_kernel(
    const uint8_t *       __restrict__ gate_exps,
    const uint8_t *       __restrict__ up_exps,
    const BlockQ8_1_Act * __restrict__ x_q8,    // [T, K/32]
    const int32_t *       __restrict__ top_ids, // [T, k_used]
    float *               __restrict__ act,     // [T, k_used, ff]
    int ff, int K, int k_used, float clamp_limit) {

    extern __shared__ BlockQ8_1_Act s_x[];
    __shared__ uint64_t s_iq2_grid[DT_G == DType::IQ2_S ? 1024 : (DT_G == DType::IQ2_XXS ? 256 : 1)];

    const int slot     = blockIdx.y;
    const int t        = blockIdx.z;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;

    const BlockQ8_1_Act * x_row = x_q8 + static_cast<size_t>(t) * K_blocks;
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(x_row);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];

    if constexpr (DT_G == DType::IQ2_S && DT_U == DType::IQ2_S) {
        for (int i = tid; i < 1024; i += nthr) s_iq2_grid[i] = __ldg(&iq2s_grid_d[i]);
    } else if constexpr (DT_G == DType::IQ2_XXS && DT_U == DType::IQ2_XXS) {
        for (int i = tid; i < 256; i += nthr) s_iq2_grid[i] = __ldg(&iq2xxs_grid_d[i]);
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= ff) return;

    const int lane = threadIdx.x;
    const int eid  = top_ids[static_cast<size_t>(t) * k_used + slot];
    constexpr size_t BG = block_bytes_of<DT_G>();
    constexpr size_t BU = block_bytes_of<DT_U>();
    const uint8_t * g_row = gate_exps + (static_cast<size_t>(eid) * ff + row) * nb * BG;
    const uint8_t * u_row = up_exps   + (static_cast<size_t>(eid) * ff + row) * nb * BU;

    float ag = 0.0f, au = 0.0f;
    #pragma unroll 2
    for (int bi = 0; bi < nb; ++bi) {
        const BlockQ8_1_Act * xb = s_x + bi * 8;
        if constexpr (DT_G == DType::IQ2_S && DT_U == DType::IQ2_S) {
            ag += dot_iq2_s_sgrid(g_row + static_cast<size_t>(bi) * BG, xb, s_iq2_grid, lane);
            au += dot_iq2_s_sgrid(u_row + static_cast<size_t>(bi) * BU, xb, s_iq2_grid, lane);
        } else if constexpr (DT_G == DType::IQ2_XXS && DT_U == DType::IQ2_XXS) {
            ag += dot_iq2_xxs_sgrid(g_row + static_cast<size_t>(bi) * BG, xb, s_iq2_grid, lane);
            au += dot_iq2_xxs_sgrid(u_row + static_cast<size_t>(bi) * BU, xb, s_iq2_grid, lane);
        } else {
            ag += dot_kblock_q8_1<DT_G>(g_row + static_cast<size_t>(bi) * BG, xb, lane);
            au += dot_kblock_q8_1<DT_U>(u_row + static_cast<size_t>(bi) * BU, xb, lane);
        }
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) {
        if (clamp_limit > 0.0f) {
            ag = fminf(fmaxf(ag, -clamp_limit), clamp_limit);
            au = fminf(fmaxf(au, -clamp_limit), clamp_limit);
        }
        const float silu = ag / (1.0f + expf(-ag));
        act[(static_cast<size_t>(t) * k_used + slot) * ff + row] = silu * au;
    }
}

// Fused MoE Down + Shared Expert Down (Q6_K SoA LDG.128!) with STREAMING 3 KB shared memory!
template <DType DT_D>
__global__ void moe_down_kquant_q8_1_kernel(
    const uint8_t *       __restrict__ down_exps,
    const int4 *          __restrict__ sh_ql,     // Q6_K SoA down_shexp (optional)
    const int4 *          __restrict__ sh_qh,
    const int8_t *        __restrict__ sh_sc,
    const uint16_t *      __restrict__ sh_d,
    const BlockQ8_1_Act * __restrict__ act_q8,    // [T, k_used, ff/32]
    const BlockQ8_1_Act * __restrict__ sh_act_q8, // [T, ff_sh/32]
    const int32_t *       __restrict__ top_ids,   // [T, k_used]
    const float *         __restrict__ top_w,     // [T, k_used]
    float *               __restrict__ out,       // [T, n_embd]
    int n_embd, int ff, int ff_sh, int k_used) {

    extern __shared__ BlockQ8_1_Act s_buf[];
    __shared__ uint32_t s_iq3_grid[DT_D == DType::IQ3_S ? 512 : (DT_D == DType::IQ3_XXS ? 256 : 1)];

    const int t        = blockIdx.y;
    const int ff_blk   = ff >> 5;
    const int sh_blk   = ff_sh >> 5;
    const int exp_blks = k_used * ff_blk;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int lane     = threadIdx.x;
    const int row      = blockIdx.x * blockDim.y + threadIdx.y;
    const int nb       = ff >> 8;
    constexpr size_t BD = block_bytes_of<DT_D>();

    if constexpr (DT_D == DType::IQ3_S) {
        for (int i = tid; i < 512; i += nthr) s_iq3_grid[i] = __ldg(&iq3s_grid_d[i]);
    } else if constexpr (DT_D == DType::IQ3_XXS) {
        for (int i = tid; i < 256; i += nthr) s_iq3_grid[i] = __ldg(&iq3xxs_grid_d[i]);
    }

    const BlockQ8_1_Act * g_act = act_q8    + static_cast<size_t>(t) * exp_blks;
    const BlockQ8_1_Act * g_sh  = sh_act_q8 + static_cast<size_t>(t) * sh_blk;
    const int n_i4_e = (ff_blk * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;

    float total = 0.0f;
    for (int slot = 0; slot < k_used; ++slot) {
        const int4 * in_e = reinterpret_cast<const int4 *>(g_act + slot * ff_blk);
        int4 *       ou_e = reinterpret_cast<int4 *>(s_buf);
        for (int i = tid; i < n_i4_e; i += nthr) ou_e[i] = in_e[i];
        __syncthreads();

        if (row < n_embd) {
            const int   eid = top_ids[static_cast<size_t>(t) * k_used + slot];
            const float ew  = top_w  [static_cast<size_t>(t) * k_used + slot];
            const uint8_t * d_row = down_exps + (static_cast<size_t>(eid) * n_embd + row) * nb * BD;

            float acc = 0.0f;
            #pragma unroll 2
            for (int bi = 0; bi < nb; ++bi) {
                if constexpr (DT_D == DType::IQ3_S) {
                    acc += dot_iq3_s_sgrid(d_row + static_cast<size_t>(bi) * BD, s_buf + bi * 8, s_iq3_grid, lane);
                } else if constexpr (DT_D == DType::IQ3_XXS) {
                    acc += dot_iq3_xxs_sgrid(d_row + static_cast<size_t>(bi) * BD, s_buf + bi * 8, s_iq3_grid, lane);
                } else {
                    acc += dot_kblock_q8_1<DT_D>(d_row + static_cast<size_t>(bi) * BD, s_buf + bi * 8, lane);
                }
            }
            total += ew * warp_sum(acc);
        }
        __syncthreads();
    }

    if (sh_ql != nullptr) {
        const int n_i4_s = (sh_blk * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
        const int4 * in_s = reinterpret_cast<const int4 *>(g_sh);
        int4 *       ou_s = reinterpret_cast<int4 *>(s_buf);
        for (int i = tid; i < n_i4_s; i += nthr) ou_s[i] = in_s[i];
        __syncthreads();

        if (row < n_embd) {
            const int nb_sh = ff_sh >> 8;
            const size_t rblk = static_cast<size_t>(row) * nb_sh;
            total += dot_q6_k_soa_row_q8_1(
                sh_ql + rblk * 8, sh_qh + rblk * 4, sh_sc + rblk * 16, sh_d + rblk,
                s_buf, nb_sh, lane);
        }
    }

    if (row < n_embd && lane == 0) out[static_cast<size_t>(t) * n_embd + row] = total;
}

__global__ void moe_down_q6_k_soa_kernel(
    const int4 *          __restrict__ dn_ql,
    const int4 *          __restrict__ dn_qh,
    const int8_t *        __restrict__ dn_sc,
    const uint16_t *      __restrict__ dn_d,
    const int4 *          __restrict__ sh_ql,
    const int4 *          __restrict__ sh_qh,
    const int8_t *        __restrict__ sh_sc,
    const uint16_t *      __restrict__ sh_d,
    const BlockQ8_1_Act * __restrict__ act_q8,
    const BlockQ8_1_Act * __restrict__ sh_act_q8,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ out,
    int n_embd, int ff, int ff_sh, int k_used) {

    extern __shared__ BlockQ8_1_Act s_buf[];
    const int t        = blockIdx.y;
    const int ff_blk   = ff >> 5;
    const int sh_blk   = ff_sh >> 5;
    const int exp_blks = k_used * ff_blk;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int lane     = threadIdx.x;
    const int row      = blockIdx.x * blockDim.y + threadIdx.y;
    const int nb       = ff >> 8;

    const BlockQ8_1_Act * g_act = act_q8    + static_cast<size_t>(t) * exp_blks;
    const BlockQ8_1_Act * g_sh  = sh_act_q8 + static_cast<size_t>(t) * sh_blk;
    const int n_i4_e = (ff_blk * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;

    float total = 0.0f;
    for (int slot = 0; slot < k_used; ++slot) {
        const int4 * in_e = reinterpret_cast<const int4 *>(g_act + slot * ff_blk);
        int4 *       ou_e = reinterpret_cast<int4 *>(s_buf);
        for (int i = tid; i < n_i4_e; i += nthr) ou_e[i] = in_e[i];
        __syncthreads();

        if (row < n_embd) {
            const int   eid = top_ids[static_cast<size_t>(t) * k_used + slot];
            const float ew  = top_w  [static_cast<size_t>(t) * k_used + slot];
            const size_t rblk = (static_cast<size_t>(eid) * n_embd + row) * nb;
            total += ew * dot_q6_k_soa_row_q8_1(
                dn_ql + rblk * 8, dn_qh + rblk * 4, dn_sc + rblk * 16, dn_d + rblk,
                s_buf, nb, lane);
        }
        __syncthreads();
    }

    if (sh_ql != nullptr) {
        const int n_i4_s = (sh_blk * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
        const int4 * in_s = reinterpret_cast<const int4 *>(g_sh);
        int4 *       ou_s = reinterpret_cast<int4 *>(s_buf);
        for (int i = tid; i < n_i4_s; i += nthr) ou_s[i] = in_s[i];
        __syncthreads();

        if (row < n_embd) {
            const int nb_sh = ff_sh >> 8;
            const size_t rblk = static_cast<size_t>(row) * nb_sh;
            total += dot_q6_k_soa_row_q8_1(
                sh_ql + rblk * 8, sh_qh + rblk * 4, sh_sc + rblk * 16, sh_d + rblk,
                s_buf, nb_sh, lane);
        }
    }

    if (row < n_embd && lane == 0) out[static_cast<size_t>(t) * n_embd + row] = total;
}

// ============================================================================
// PART 4B: FAST GROUPED-EXPERT MoE PREFILL (T >= 16) — ~6-7x WEIGHT REUSE!
// 1. 1-block GPU counting sort groups (t, slot) pairs by expert `eid` (zero host sync!).
// 2. Each expert decodes each 256-elem weight block ONCE and multiplies up to BT tokens via dp4a!
// ============================================================================

__global__ void group_moe_tokens_kernel(
    const int32_t * __restrict__ top_ids,      // [T * k_used]
    int32_t *       __restrict__ exp_offsets,  // [n_expert + 1]
    int32_t *       __restrict__ sorted_pairs, // [T * k_used] packed as (t << 4) | slot
    int total_pairs, int n_expert, int k_used) {

    extern __shared__ int32_t s_cnt[]; // [n_expert]
    const int tid = threadIdx.x;
    for (int e = tid; e < n_expert; e += blockDim.x) s_cnt[e] = 0;
    __syncthreads();

    for (int i = tid; i < total_pairs; i += blockDim.x) {
        const int eid = top_ids[i];
        if (eid >= 0 && eid < n_expert) atomicAdd(&s_cnt[eid], 1);
    }
    __syncthreads();

    // Single-thread prefix sum over n_expert (<= 512 -> ~1 microsecond!)
    if (tid == 0) {
        int sum = 0;
        for (int e = 0; e < n_expert; ++e) {
            int c = s_cnt[e];
            exp_offsets[e] = sum;
            s_cnt[e] = sum; // reuse s_cnt as write cursor!
            sum += c;
        }
        exp_offsets[n_expert] = sum;
    }
    __syncthreads();

    for (int i = tid; i < total_pairs; i += blockDim.x) {
        const int eid = top_ids[i];
        if (eid >= 0 && eid < n_expert) {
            const int dst = atomicAdd(&s_cnt[eid], 1);
            const int t    = i / k_used;
            const int slot = i % k_used;
            sorted_pairs[dst] = (t << 4) | slot;
        }
    }
}

// Decode ONE 256-elem block ONCE into (w0, w1, scale, xb_idx, x_off0, x_off1) using smem LUTs!
template <DType DT>
__device__ __forceinline__ void decode_kblock_dp4a_pair(
    const void * __restrict__ blk_ptr,
    const void * __restrict__ s_grid,
    const uint8_t * __restrict__ s_ksigns,
    int lane,
    int & w0, int & w1, float & sc,
    int & xb_idx, int & x_off0, int & x_off1);

template <>
__device__ __forceinline__ void decode_kblock_dp4a_pair<DType::IQ2_S>(
    const void * __restrict__ blk_ptr, const void * __restrict__ s_grid, const uint8_t * __restrict__, int lane,
    int & w0, int & w1, float & sc, int & xb_idx, int & x_off0, int & x_off1) {
    const auto & b = *static_cast<const BlockIQ2_S *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint8_t scb = __ldg(&b.scales[ib32]);
    const int sc4 = (l >= 2) ? (scb >> 4) : (scb & 0xF);
    sc = d * (0.5f + static_cast<float>(sc4)) * 0.25f;
    const uint8_t qs_b = __ldg(&b.qs[lane]);
    const uint8_t sg_b = __ldg(&b.qs[32 + lane]);
    const uint8_t qh_b = __ldg(&b.qh[ib32]);
    const uint32_t gidx = static_cast<uint32_t>(qs_b) | ((static_cast<uint32_t>(qh_b) << (8 - 2 * l)) & 0x300u);
    const uint64_t g64  = static_cast<const uint64_t *>(s_grid)[gidx];
    w0 = apply_signs4(static_cast<uint32_t>(g64),       sg_b & 0xFu);
    w1 = apply_signs4(static_cast<uint32_t>(g64 >> 32), sg_b >> 4);
    xb_idx = ib32; x_off0 = (l << 1); x_off1 = (l << 1) + 1;
}

template <>
__device__ __forceinline__ void decode_kblock_dp4a_pair<DType::IQ2_XXS>(
    const void * __restrict__ blk_ptr, const void * __restrict__ s_grid, const uint8_t * __restrict__ s_ksigns, int lane,
    int & w0, int & w1, float & sc, int & xb_idx, int & x_off0, int & x_off1) {
    const auto & b = *static_cast<const BlockIQ2_XXS *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint16_t * q2 = b.qs + 4 * ib32;
    const uint16_t w_idx = __ldg(&q2[l >> 1]);
    const uint32_t gidx  = (l & 1) ? (w_idx >> 8) : (w_idx & 0xFFu);
    const uint32_t aux32 = static_cast<uint32_t>(__ldg(&q2[2])) | (static_cast<uint32_t>(__ldg(&q2[3])) << 16);
    sc = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.25f;
    const uint8_t  sg  = s_ksigns[(aux32 >> (7 * l)) & 127u];
    const uint64_t g64 = static_cast<const uint64_t *>(s_grid)[gidx];
    w0 = apply_signs4(static_cast<uint32_t>(g64),       sg & 0xFu);
    w1 = apply_signs4(static_cast<uint32_t>(g64 >> 32), sg >> 4);
    xb_idx = ib32; x_off0 = (l << 1); x_off1 = (l << 1) + 1;
}

template <>
__device__ __forceinline__ void decode_kblock_dp4a_pair<DType::IQ3_S>(
    const void * __restrict__ blk_ptr, const void * __restrict__ s_grid, const uint8_t * __restrict__, int lane,
    int & w0, int & w1, float & sc, int & xb_idx, int & x_off0, int & x_off1) {
    const auto & b = *static_cast<const BlockIQ3_S *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint8_t scb = __ldg(&b.scales[ib32 >> 1]);
    const int sc4 = (ib32 & 1) ? (scb >> 4) : (scb & 0xF);
    sc = d * static_cast<float>(1 + 2 * sc4);
    const uint8_t  qh  = __ldg(&b.qh[ib32]);
    const uint16_t q01 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs)[lane]);
    const uint8_t  sg  = __ldg(&b.signs[lane]);
    const auto * g32   = static_cast<const uint32_t *>(s_grid);
    const uint32_t g1  = g32[(q01 & 0xFFu) | ((static_cast<uint32_t>(qh) << (8 - 2 * l)) & 256u)];
    const uint32_t g2  = g32[(q01 >> 8)    | ((static_cast<uint32_t>(qh) << (7 - 2 * l)) & 256u)];
    w0 = apply_signs4(g1, sg & 0xFu);
    w1 = apply_signs4(g2, sg >> 4);
    xb_idx = ib32; x_off0 = (l << 1); x_off1 = (l << 1) + 1;
}

template <>
__device__ __forceinline__ void decode_kblock_dp4a_pair<DType::IQ3_XXS>(
    const void * __restrict__ blk_ptr, const void * __restrict__ s_grid, const uint8_t * __restrict__ s_ksigns, int lane,
    int & w0, int & w1, float & sc, int & xb_idx, int & x_off0, int & x_off1) {
    const auto & b = *static_cast<const BlockIQ3_XXS *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const int ib32 = lane >> 2;
    const int l    = lane & 3;
    const uint16_t q01 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs)[lane]);
    const uint16_t ss0 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs + 64)[ib32 * 2 + 0]);
    const uint16_t ss1 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs + 64)[ib32 * 2 + 1]);
    const uint32_t aux32 = static_cast<uint32_t>(ss0) | (static_cast<uint32_t>(ss1) << 16);
    sc = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
    const uint8_t  sg  = s_ksigns[(aux32 >> (7 * l)) & 127u];
    const auto * g32   = static_cast<const uint32_t *>(s_grid);
    const uint32_t g1  = g32[q01 & 0xFFu];
    const uint32_t g2  = g32[q01 >> 8];
    w0 = apply_signs4(g1, sg & 0xFu);
    w1 = apply_signs4(g2, sg >> 4);
    xb_idx = ib32; x_off0 = (l << 1); x_off1 = (l << 1) + 1;
}

template <>
__device__ __forceinline__ void decode_kblock_dp4a_pair<DType::IQ4_XS>(
    const void * __restrict__ blk_ptr, const void * __restrict__, const uint8_t * __restrict__, int lane,
    int & w0, int & w1, float & sc, int & xb_idx, int & x_off0, int & x_off1) {
    const auto & b = *static_cast<const BlockIQ4_XS *>(blk_ptr);
    const float d  = half_bits_to_f32(__ldg(&b.d));
    const uint16_t sh = __ldg(&b.scales_h);
    const int ib   = lane >> 2;
    const int sub4 = lane & 3;
    const uint8_t sl = __ldg(&b.scales_l[ib >> 1]);
    const int ls = ((sl >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    sc = d * static_cast<float>(ls - 32);
    const uint32_t q4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qs)[ib * 4 + sub4]);
    w0 = static_cast<int>(
          static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >>  0) & 0xFu])))
        | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >>  8) & 0xFu]))) << 8)
        | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 16) & 0xFu]))) << 16)
        | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 24) & 0xFu]))) << 24));
    w1 = static_cast<int>(
          static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >>  4) & 0xFu])))
        | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 12) & 0xFu]))) << 8)
        | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 20) & 0xFu]))) << 16)
        | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 28) & 0xFu]))) << 24));
    xb_idx = ib; x_off0 = sub4; x_off1 = 4 + sub4;
}

// Decode ONE 256-elem Q4_K block ONCE for multi-token dp4a reuse!
__device__ __forceinline__ void decode_q4_k_dp4a_pair(
    const void * __restrict__ blk_ptr, int lane,
    int & qlo, int & qhi, float & d1, float & mn1, float & d2, float & mn2,
    int & xb0, int & x_off) {
    const auto & b = *static_cast<const BlockQ4_K *>(blk_ptr);
    const float d    = half_bits_to_f32(__ldg(&b.d));
    const float dmin = half_bits_to_f32(__ldg(&b.dmin));
    const int j    = lane >> 3; // 0..3 (which 64-elem pair)
    const int sub4 = lane & 7;  // 0..7 (4 bytes = 4 low + 4 high nibbles)
    uint8_t sc1, m1, sc2, m2;
    get_scale_min_k4(2 * j + 0, b.scales, sc1, m1);
    get_scale_min_k4(2 * j + 1, b.scales, sc2, m2);
    d1 = d * static_cast<float>(sc1); mn1 = dmin * static_cast<float>(m1);
    d2 = d * static_cast<float>(sc2); mn2 = dmin * static_cast<float>(m2);
    const uint32_t q4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qs)[j * 8 + sub4]);
    qlo = static_cast<int>(q4 & 0x0F0F0F0Fu);
    qhi = static_cast<int>((q4 >> 4) & 0x0F0F0F0Fu);
    xb0 = 2 * j;
    x_off = sub4;
}

// Multi-row-per-block Grouped MoE Gate+Up+SwiGLU for prefill:
// Loads BT tokens of x_q8 (entire K) + LUT ONCE per block, then computes ROWS_PER_BLK=64 rows
// (wpb=8 warps x 8 rows/warp) with ZERO __syncthreads() inside the row/K loops -> 16x less x_q8 traffic!
template <DType DT_G, DType DT_U, int BT, int R_PER_WARP>
__global__ void moe_gate_up_grouped_kquant_kernel(
    const uint8_t *       __restrict__ gate_exps,
    const uint8_t *       __restrict__ up_exps,
    const BlockQ8_1_Act * __restrict__ x_q8,         // [T, K/32]
    const int32_t *       __restrict__ exp_offsets,  // [n_expert + 1]
    const int32_t *       __restrict__ sorted_pairs, // [T * k_used]
    float *               __restrict__ act,          // [T, k_used, ff]
    int ff, int K, int k_used, float clamp_limit) {

    const int eid   = blockIdx.y;
    const int start = exp_offsets[eid];
    const int end   = exp_offsets[eid + 1];
    if (start >= end) return;

    extern __shared__ BlockQ8_1_Act s_x[]; // [BT, K/32] (6 * 6 KB = 36 KB for K=4096)
    __shared__ uint64_t s_grid64[(DT_G == DType::IQ2_S) ? 1024 : ((DT_G == DType::IQ2_XXS) ? 256 : 1)];
    __shared__ uint32_t s_grid32[(DT_G == DType::IQ3_S) ? 512  : ((DT_G == DType::IQ3_XXS) ? 256 : 1)];
    __shared__ uint8_t  s_ksigns[(DT_G == DType::IQ2_XXS || DT_G == DType::IQ3_XXS) ? 128 : 1];
    __shared__ int32_t  s_pairs[BT];

    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int lane     = threadIdx.x;
    const int row_base = blockIdx.x * (blockDim.y * R_PER_WARP) + threadIdx.y;

    if constexpr (DT_G == DType::IQ2_S) {
        for (int i = tid; i < 1024; i += nthr) s_grid64[i] = __ldg(&iq2s_grid_d[i]);
    } else if constexpr (DT_G == DType::IQ2_XXS) {
        for (int i = tid; i < 256; i += nthr) s_grid64[i] = __ldg(&iq2xxs_grid_d[i]);
        if (tid < 128) s_ksigns[tid] = __ldg(&ksigns_iq2xs_d[tid]);
    } else if constexpr (DT_G == DType::IQ3_S) {
        for (int i = tid; i < 512; i += nthr) s_grid32[i] = __ldg(&iq3s_grid_d[i]);
    } else if constexpr (DT_G == DType::IQ3_XXS) {
        for (int i = tid; i < 256; i += nthr) s_grid32[i] = __ldg(&iq3xxs_grid_d[i]);
        if (tid < 128) s_ksigns[tid] = __ldg(&ksigns_iq2xs_d[tid]);
    }

    const void * s_grid_ptr = (DT_G == DType::IQ2_S || DT_G == DType::IQ2_XXS)
        ? static_cast<const void *>(s_grid64) : static_cast<const void *>(s_grid32);

    constexpr size_t BG = block_bytes_of<DT_G>();
    constexpr size_t BU = block_bytes_of<DT_U>();
    const int n_i4_tok = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) s_pairs[tid] = sorted_pairs[base_i + tid];
        __syncthreads();

        // Load full x_q8[0..K-1] for all n_tile tokens ONCE -> reused across all 64 rows!
        for (int j = 0; j < n_tile; ++j) {
            const int tok = s_pairs[j] >> 4;
            const int4 * src = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(tok) * K_blocks);
            int4 *       dst = reinterpret_cast<int4 *>(s_x + j * K_blocks);
            for (int i = tid; i < n_i4_tok; i += nthr) dst[i] = src[i];
        }
        __syncthreads();

        for (int rr = 0; rr < R_PER_WARP; ++rr) {
            const int row = row_base + rr * blockDim.y;
            if (row >= ff) break;

            const uint8_t * g_row = gate_exps + (static_cast<size_t>(eid) * ff + row) * nb * BG;
            const uint8_t * u_row = up_exps   + (static_cast<size_t>(eid) * ff + row) * nb * BU;

            float ag[BT], au[BT];
            #pragma unroll
            for (int j = 0; j < BT; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

            for (int bi = 0; bi < nb; ++bi) {
                if constexpr ((DT_G == DType::IQ2_S || DT_G == DType::IQ2_XXS || DT_G == DType::IQ3_S ||
                               DT_G == DType::IQ3_XXS || DT_G == DType::IQ4_XS) && DT_G == DT_U) {
                    int gw0, gw1, uw0, uw1, xbi, xo0, xo1;
                    float gsc, usc;
                    decode_kblock_dp4a_pair<DT_G>(g_row + static_cast<size_t>(bi) * BG, s_grid_ptr, s_ksigns, lane, gw0, gw1, gsc, xbi, xo0, xo1);
                    decode_kblock_dp4a_pair<DT_U>(u_row + static_cast<size_t>(bi) * BU, s_grid_ptr, s_ksigns, lane, uw0, uw1, usc, xbi, xo0, xo1);
                    const int boff = bi * 8 + xbi;
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) {
                        if (j < n_tile) {
                            const BlockQ8_1_Act & xb = s_x[j * K_blocks + boff];
                            const int32_t * x32 = reinterpret_cast<const int32_t *>(xb.qs);
                            const int x0 = x32[xo0], x1 = x32[xo1];
                            ag[j] += (gsc * xb.d) * static_cast<float>(dp4a_s32(gw0, x0, dp4a_s32(gw1, x1, 0)));
                            au[j] += (usc * xb.d) * static_cast<float>(dp4a_s32(uw0, x0, dp4a_s32(uw1, x1, 0)));
                        }
                    }
                } else if constexpr (DT_G == DType::Q4_K && DT_U == DType::Q4_K) {
                    int gq0, gq1, uq0, uq1, xb0, xo;
                    float gd1, gmn1, gd2, gmn2, ud1, umn1, ud2, umn2;
                    decode_q4_k_dp4a_pair(g_row + static_cast<size_t>(bi) * BG, lane, gq0, gq1, gd1, gmn1, gd2, gmn2, xb0, xo);
                    decode_q4_k_dp4a_pair(u_row + static_cast<size_t>(bi) * BU, lane, uq0, uq1, ud1, umn1, ud2, umn2, xb0, xo);
                    const bool is_zero_sub = (xo == 0);
                    const int boff = bi * 8 + xb0;
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) {
                        if (j < n_tile) {
                            const BlockQ8_1_Act & xba = s_x[j * K_blocks + boff + 0];
                            const BlockQ8_1_Act & xbb = s_x[j * K_blocks + boff + 1];
                            const int x0 = reinterpret_cast<const int32_t *>(xba.qs)[xo];
                            const int x1 = reinterpret_cast<const int32_t *>(xbb.qs)[xo];
                            ag[j] += (gd1 * xba.d) * static_cast<float>(dp4a_s32(gq0, x0, 0))
                                   + (gd2 * xbb.d) * static_cast<float>(dp4a_s32(gq1, x1, 0))
                                   - (is_zero_sub ? (gmn1 * xba.s_all + gmn2 * xbb.s_all) : 0.0f);
                            au[j] += (ud1 * xba.d) * static_cast<float>(dp4a_s32(uq0, x0, 0))
                                   + (ud2 * xbb.d) * static_cast<float>(dp4a_s32(uq1, x1, 0))
                                   - (is_zero_sub ? (umn1 * xba.s_all + umn2 * xbb.s_all) : 0.0f);
                        }
                    }
                } else {
                    const uint8_t * gb = g_row + static_cast<size_t>(bi) * BG;
                    const uint8_t * ub = u_row + static_cast<size_t>(bi) * BU;
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) {
                        if (j < n_tile) {
                            const BlockQ8_1_Act * xb = s_x + j * K_blocks + bi * 8;
                            ag[j] += dot_kblock_q8_1<DT_G>(gb, xb, lane);
                            au[j] += dot_kblock_q8_1<DT_U>(ub, xb, lane);
                        }
                    }
                }
            }

            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                if (j < n_tile) {
                    float g = warp_sum(ag[j]);
                    float u = warp_sum(au[j]);
                    if (lane == 0) {
                        if (clamp_limit > 0.0f) {
                            g = fminf(fmaxf(g, -clamp_limit), clamp_limit);
                            u = fminf(fmaxf(u, -clamp_limit), clamp_limit);
                        }
                        const int pair = s_pairs[j];
                        const int tok  = pair >> 4;
                        const int slot = pair & 15;
                        act[(static_cast<size_t>(tok) * k_used + slot) * ff + row] = (g / (1.0f + expf(-g))) * u;
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Multi-row-per-block Grouped MoE Down for prefill:
// Loads BT=8 tokens of act_q8 (entire ff=2048 -> 24 KB -> 2 blocks/SM!) + LUT ONCE per block,
// then computes ROWS_PER_BLK=64 rows with ZERO __syncthreads() inside the row/K loops -> 16x less act_q8 traffic!
template <DType DT_D, int BT, int R_PER_WARP, bool USE_SLOTS>
__global__ void moe_down_grouped_kquant_kernel(
    const uint8_t *       __restrict__ down_exps,
    const BlockQ8_1_Act * __restrict__ act_q8,       // [T, k_used, ff/32]
    const float *         __restrict__ top_w,        // [T, k_used]
    const int32_t *       __restrict__ exp_offsets,  // [n_expert + 1]
    const int32_t *       __restrict__ sorted_pairs, // [T * k_used]
    float *               __restrict__ out,          // USE_SLOTS: [T, k_used, n_embd]; else [T, n_embd]
    int n_embd, int ff, int k_used) {

    const int eid   = blockIdx.y;
    const int start = exp_offsets[eid];
    const int end   = exp_offsets[eid + 1];
    if (start >= end) return;

    extern __shared__ BlockQ8_1_Act s_a[]; // [BT, ff/32] (8 * 3 KB = 24 KB for ff=2048!)
    __shared__ uint64_t s_grid64[(DT_D == DType::IQ2_S) ? 1024 : ((DT_D == DType::IQ2_XXS) ? 256 : 1)];
    __shared__ uint32_t s_grid32[(DT_D == DType::IQ3_S) ? 512  : ((DT_D == DType::IQ3_XXS) ? 256 : 1)];
    __shared__ uint8_t  s_ksigns[(DT_D == DType::IQ2_XXS || DT_D == DType::IQ3_XXS) ? 128 : 1];
    __shared__ int32_t  s_pairs[BT];

    const int ff_blk   = ff >> 5;
    const int nb       = ff >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int lane     = threadIdx.x;
    const int row_base = blockIdx.x * (blockDim.y * R_PER_WARP) + threadIdx.y;

    if constexpr (DT_D == DType::IQ2_S) {
        for (int i = tid; i < 1024; i += nthr) s_grid64[i] = __ldg(&iq2s_grid_d[i]);
    } else if constexpr (DT_D == DType::IQ2_XXS) {
        for (int i = tid; i < 256; i += nthr) s_grid64[i] = __ldg(&iq2xxs_grid_d[i]);
        if (tid < 128) s_ksigns[tid] = __ldg(&ksigns_iq2xs_d[tid]);
    } else if constexpr (DT_D == DType::IQ3_S) {
        for (int i = tid; i < 512; i += nthr) s_grid32[i] = __ldg(&iq3s_grid_d[i]);
    } else if constexpr (DT_D == DType::IQ3_XXS) {
        for (int i = tid; i < 256; i += nthr) s_grid32[i] = __ldg(&iq3xxs_grid_d[i]);
        if (tid < 128) s_ksigns[tid] = __ldg(&ksigns_iq2xs_d[tid]);
    }

    const void * s_grid_ptr = (DT_D == DType::IQ2_S || DT_D == DType::IQ2_XXS)
        ? static_cast<const void *>(s_grid64) : static_cast<const void *>(s_grid32);

    constexpr size_t BD = block_bytes_of<DT_D>();
    const int n_i4_tok = (ff_blk * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) s_pairs[tid] = sorted_pairs[base_i + tid];
        __syncthreads();

        for (int j = 0; j < n_tile; ++j) {
            const int pair = s_pairs[j];
            const int tok  = pair >> 4;
            const int slot = pair & 15;
            const int4 * src = reinterpret_cast<const int4 *>(act_q8 + (static_cast<size_t>(tok) * k_used + slot) * ff_blk);
            int4 *       dst = reinterpret_cast<int4 *>(s_a + j * ff_blk);
            for (int i = tid; i < n_i4_tok; i += nthr) dst[i] = src[i];
        }
        __syncthreads();

        for (int rr = 0; rr < R_PER_WARP; ++rr) {
            const int row = row_base + rr * blockDim.y;
            if (row >= n_embd) break;

            const uint8_t * d_row = down_exps + (static_cast<size_t>(eid) * n_embd + row) * nb * BD;

            float acc[BT];
            #pragma unroll
            for (int j = 0; j < BT; ++j) acc[j] = 0.0f;

            for (int bi = 0; bi < nb; ++bi) {
                if constexpr (DT_D == DType::IQ2_S || DT_D == DType::IQ2_XXS || DT_D == DType::IQ3_S ||
                              DT_D == DType::IQ3_XXS || DT_D == DType::IQ4_XS) {
                    int dw0, dw1, xbi, xo0, xo1;
                    float dsc;
                    decode_kblock_dp4a_pair<DT_D>(d_row + static_cast<size_t>(bi) * BD, s_grid_ptr, s_ksigns, lane, dw0, dw1, dsc, xbi, xo0, xo1);
                    const int boff = bi * 8 + xbi;
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) {
                        if (j < n_tile) {
                            const BlockQ8_1_Act & xb = s_a[j * ff_blk + boff];
                            const int32_t * x32 = reinterpret_cast<const int32_t *>(xb.qs);
                            acc[j] += (dsc * xb.d) * static_cast<float>(dp4a_s32(dw0, x32[xo0], dp4a_s32(dw1, x32[xo1], 0)));
                        }
                    }
                } else if constexpr (DT_D == DType::Q4_K) {
                    int dq0, dq1, xb0, xo;
                    float dd1, dmn1, dd2, dmn2;
                    decode_q4_k_dp4a_pair(d_row + static_cast<size_t>(bi) * BD, lane, dq0, dq1, dd1, dmn1, dd2, dmn2, xb0, xo);
                    const bool is_zero_sub = (xo == 0);
                    const int boff = bi * 8 + xb0;
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) {
                        if (j < n_tile) {
                            const BlockQ8_1_Act & xba = s_a[j * ff_blk + boff + 0];
                            const BlockQ8_1_Act & xbb = s_a[j * ff_blk + boff + 1];
                            const int x0 = reinterpret_cast<const int32_t *>(xba.qs)[xo];
                            const int x1 = reinterpret_cast<const int32_t *>(xbb.qs)[xo];
                            acc[j] += (dd1 * xba.d) * static_cast<float>(dp4a_s32(dq0, x0, 0))
                                    + (dd2 * xbb.d) * static_cast<float>(dp4a_s32(dq1, x1, 0))
                                    - (is_zero_sub ? (dmn1 * xba.s_all + dmn2 * xbb.s_all) : 0.0f);
                        }
                    }
                } else {
                    const uint8_t * db = d_row + static_cast<size_t>(bi) * BD;
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) {
                        if (j < n_tile) {
                            acc[j] += dot_kblock_q8_1<DT_D>(db, s_a + j * ff_blk + bi * 8, lane);
                        }
                    }
                }
            }

            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                if (j < n_tile) {
                    const float v = warp_sum(acc[j]);
                    if (lane == 0) {
                        const int pair = s_pairs[j];
                        const int tok  = pair >> 4;
                        const int slot = pair & 15;
                        const float ew = top_w[static_cast<size_t>(tok) * k_used + slot];
                        if constexpr (USE_SLOTS) {
                            out[(static_cast<size_t>(tok) * k_used + slot) * n_embd + row] = ew * v;
                        } else {
                            atomicAdd(&out[static_cast<size_t>(tok) * n_embd + row], ew * v);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Deterministic reduction of down_slots[T, k_used, n_embd] into out[T, n_embd] (adds to down_shexp!)
template <int BT, int R_PER_WARP, bool USE_SLOTS>
__global__ void moe_down_grouped_q6_k_soa_kernel(
    const int4 *          __restrict__ dn_ql,
    const int4 *          __restrict__ dn_qh,
    const int8_t *        __restrict__ dn_sc,
    const uint16_t *      __restrict__ dn_d,
    const BlockQ8_1_Act * __restrict__ act_q8,
    const float *         __restrict__ top_w,
    const int32_t *       __restrict__ exp_offsets,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ out,
    int n_embd, int ff, int k_used) {

    const int eid   = blockIdx.y;
    const int start = exp_offsets[eid];
    const int end   = exp_offsets[eid + 1];
    if (start >= end) return;

    extern __shared__ BlockQ8_1_Act s_a[];
    __shared__ int32_t s_pairs[BT];

    const int ff_blk   = ff >> 5;
    const int nb       = ff >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int lane     = threadIdx.x;
    const int row_base = blockIdx.x * (blockDim.y * R_PER_WARP) + threadIdx.y;
    const int n_i4_tok = (ff_blk * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) s_pairs[tid] = sorted_pairs[base_i + tid];
        __syncthreads();

        for (int j = 0; j < n_tile; ++j) {
            const int pair = s_pairs[j];
            const int tok  = pair >> 4;
            const int slot = pair & 15;
            const int4 * src = reinterpret_cast<const int4 *>(act_q8 + (static_cast<size_t>(tok) * k_used + slot) * ff_blk);
            int4 *       dst = reinterpret_cast<int4 *>(s_a + j * ff_blk);
            for (int i = tid; i < n_i4_tok; i += nthr) dst[i] = src[i];
        }
        __syncthreads();

        for (int rr = 0; rr < R_PER_WARP; ++rr) {
            const int row = row_base + rr * blockDim.y;
            if (row >= n_embd) break;

            const size_t rblk = (static_cast<size_t>(eid) * n_embd + row) * nb;
            for (int j = 0; j < n_tile; ++j) {
                const float v = dot_q6_k_soa_row_q8_1(
                    dn_ql + rblk * 8, dn_qh + rblk * 4, dn_sc + rblk * 16, dn_d + rblk,
                    s_a + j * ff_blk, nb, lane);
                if (lane == 0) {
                    const int pair = s_pairs[j];
                    const int tok  = pair >> 4;
                    const int slot = pair & 15;
                    const float ew = top_w[static_cast<size_t>(tok) * k_used + slot];
                    if constexpr (USE_SLOTS) {
                        out[(static_cast<size_t>(tok) * k_used + slot) * n_embd + row] = ew * v;
                    } else {
                        atomicAdd(&out[static_cast<size_t>(tok) * n_embd + row], ew * v);
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ void moe_reduce_slots_add_kernel(
    const float * __restrict__ slots,
    float *       __restrict__ out,
    int n_embd, int k_used) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= n_embd) return;
    float sum = out[static_cast<size_t>(t) * n_embd + d];
    const float * base = slots + static_cast<size_t>(t) * k_used * n_embd + d;
    for (int s = 0; s < k_used; ++s) sum += base[static_cast<size_t>(s) * n_embd];
    out[static_cast<size_t>(t) * n_embd + d] = sum;
}

template <DType DT>
void dispatch_linear_kquant_prefill_f16(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);
    const size_t x_f16_bytes = static_cast<size_t>(T) * K * sizeof(__half);
    const size_t align_x = (x_f16_bytes + 255) & ~size_t(255);
    const size_t rem = scratch_bytes - align_x;
    const int chunk_m = std::max(1, std::min(M, static_cast<int>(rem / (static_cast<size_t>(K) * sizeof(__half)))));
    __half * d_x16 = static_cast<__half *>(d_scratch);
    __half * d_w16 = reinterpret_cast<__half *>(static_cast<uint8_t *>(d_scratch) + align_x);

    launch_cast_f32_to_f16(d_x, d_x16, static_cast<int64_t>(T) * K, stream);
    cublasSetStream(cublas, stream);
    const float alpha = 1.0f, beta = 0.0f;
    const int nb_row = K >> 8;

    for (int m0 = 0; m0 < M; m0 += chunk_m) {
        const int cm = std::min(chunk_m, M - m0);
        const int64_t nblk = static_cast<int64_t>(cm) * nb_row;
        constexpr int wpb = 8;
        if constexpr (DT == DType::Q6_K) {
            const int64_t b0 = static_cast<int64_t>(m0) * nb_row;
            dequant_q6_k_soa_kernel<__half><<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
                static_cast<const uint8_t *>(W.ptr) + b0 * 128,
                static_cast<const uint8_t *>(W.aux_ptr) + b0 * 64,
                static_cast<const int8_t *>(W.scales_ptr) + b0 * 16,
                static_cast<const uint16_t *>(W.d_ptr) + b0,
                d_w16, nblk);
        } else {
            constexpr size_t B = block_bytes_of<DT>();
            dequant_kquant_kernel<DT, __half><<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
                static_cast<const uint8_t *>(W.ptr) + static_cast<size_t>(m0) * nb_row * B, d_w16, nblk);
        }
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                     cm, T, K,
                     &alpha,
                     d_w16, CUDA_R_16F, K,
                     d_x16, CUDA_R_16F, K,
                     &beta,
                     d_y + m0, CUDA_R_32F, M,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

template <DType DT>
void dispatch_linear_kquant_prefill_f32(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);
    const int chunk_m = std::max(1, std::min(M, static_cast<int>(scratch_bytes / (static_cast<size_t>(K) * sizeof(float)))));
    float * d_w32 = static_cast<float *>(d_scratch);

    cublasSetStream(cublas, stream);
    const float alpha = 1.0f, beta = 0.0f;
    const int nb_row = K >> 8;

    for (int m0 = 0; m0 < M; m0 += chunk_m) {
        const int cm = std::min(chunk_m, M - m0);
        const int64_t nblk = static_cast<int64_t>(cm) * nb_row;
        constexpr int wpb = 8;
        if constexpr (DT == DType::Q6_K) {
            const int64_t b0 = static_cast<int64_t>(m0) * nb_row;
            dequant_q6_k_soa_kernel<float><<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
                static_cast<const uint8_t *>(W.ptr) + b0 * 128,
                static_cast<const uint8_t *>(W.aux_ptr) + b0 * 64,
                static_cast<const int8_t *>(W.scales_ptr) + b0 * 16,
                static_cast<const uint16_t *>(W.d_ptr) + b0,
                d_w32, nblk);
        } else {
            constexpr size_t B = block_bytes_of<DT>();
            dequant_kquant_kernel<DT, float><<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
                static_cast<const uint8_t *>(W.ptr) + static_cast<size_t>(m0) * nb_row * B, d_w32, nblk);
        }
        cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    cm, T, K,
                    &alpha,
                    d_w32, K,
                    d_x, K,
                    &beta,
                    d_y + m0, M);
    }
}

__global__ void cast_f32_to_f16_k_kernel(const float * __restrict__ src, __half * __restrict__ dst, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

__global__ void silu_mul_clamp_kernel(
    const float * __restrict__ gate, const float * __restrict__ up, float * __restrict__ out,
    int64_t n, float clamp_limit) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp_limit > 0.0f) {
        g = fminf(fmaxf(g, -clamp_limit), clamp_limit);
        u = fminf(fmaxf(u, -clamp_limit), clamp_limit);
    }
    out[i] = (g / (1.0f + expf(-g))) * u;
}

} // namespace

bool is_kquant_dtype(DType t) {
    return t == DType::Q6_K || t == DType::Q5_K || t == DType::Q4_K ||
           t == DType::IQ4_XS || t == DType::IQ3_S || t == DType::IQ3_XXS ||
           t == DType::IQ2_S || t == DType::IQ2_XXS || t == DType::Q3_K || t == DType::Q2_K;
}

void launch_cast_f32_to_f16(const float * d_src, void * d_dst, int64_t n, cudaStream_t stream) {
    cast_f32_to_f16_k_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_src, static_cast<__half *>(d_dst), n);
}

void launch_silu_mul_clamp(const float * d_gate, const float * d_up, float * d_out, int64_t n, float clamp_limit, cudaStream_t stream) {
    silu_mul_clamp_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_gate, d_up, d_out, n, clamp_limit);
}

void launch_linear_prequant(
    const DeviceTensor & W, const void * d_xq8, float * d_y, int T, cudaStream_t stream) {

    if (W.dtype == DType::Q8_0) {
        launch_linear_q8_0_prequant(W, d_xq8, d_y, T, stream);
        return;
    }

    const int K        = static_cast<int>(W.ne[0]);
    const int M        = static_cast<int>(W.ne[1]);
    const int K_blocks = K >> 5;
    constexpr int wpb  = 8;
    const auto * xq = static_cast<const BlockQ8_1_Act *>(d_xq8);

    if (W.dtype == DType::Q6_K && W.is_soa) {
        const auto * ql = static_cast<const int4 *>(W.ptr);
        const auto * qh = static_cast<const int4 *>(W.aux_ptr);
        const auto * sc = static_cast<const int8_t *>(W.scales_ptr);
        const auto * d  = static_cast<const uint16_t *>(W.d_ptr);
        if (T == 2 && K_blocks <= 384) {
            const size_t smem = static_cast<size_t>(2 * K_blocks) * sizeof(BlockQ8_1_Act);
            gemv_q6_k_soa_dp4a_t2_kernel<<<(M + wpb - 1) / wpb, dim3(32, wpb), smem, stream>>>(ql, qh, sc, d, xq, d_y, M, K);
            return;
        }
        const size_t smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
        gemv_q6_k_soa_dp4a_kernel<<<dim3((M + wpb - 1) / wpb, T), dim3(32, wpb), smem, stream>>>(ql, qh, sc, d, xq, d_y, M, K);
        return;
    }

    if (W.dtype == DType::Q5_K) {
        const auto * wp5 = static_cast<const BlockQ5_K *>(W.ptr);
        if (T == 2 && K_blocks <= 384) {
            const size_t smem = static_cast<size_t>(2 * K_blocks) * sizeof(BlockQ8_1_Act);
            gemv_q5_k_ldg128_kernel<true><<<(M + wpb - 1) / wpb, dim3(32, wpb), smem, stream>>>(wp5, xq, d_y, M, K);
            return;
        }
        const size_t smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
        gemv_q5_k_ldg128_kernel<false><<<dim3((M + wpb - 1) / wpb, T), dim3(32, wpb), smem, stream>>>(wp5, xq, d_y, M, K);
        return;
    }

    const auto * wp = static_cast<const uint8_t *>(W.ptr);
    auto run_kq = [&](auto dt_tag) {
        constexpr DType DT = decltype(dt_tag)::value;
        if (T == 2 && K_blocks <= 384) {
            const size_t smem = static_cast<size_t>(2 * K_blocks) * sizeof(BlockQ8_1_Act);
            gemv_kquant_q8_1_t2_kernel<DT><<<(M + wpb - 1) / wpb, dim3(32, wpb), smem, stream>>>(wp, xq, d_y, M, K);
        } else {
            const size_t smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
            gemv_kquant_q8_1_kernel<DT><<<dim3((M + wpb - 1) / wpb, T), dim3(32, wpb), smem, stream>>>(wp, xq, d_y, M, K);
        }
    };

    switch (W.dtype) {
        case DType::Q5_K:    run_kq(std::integral_constant<DType, DType::Q5_K>{});    break;
        case DType::Q4_K:    run_kq(std::integral_constant<DType, DType::Q4_K>{});    break;
        case DType::IQ4_XS:  run_kq(std::integral_constant<DType, DType::IQ4_XS>{});  break;
        case DType::IQ3_S:   run_kq(std::integral_constant<DType, DType::IQ3_S>{});   break;
        case DType::IQ3_XXS: run_kq(std::integral_constant<DType, DType::IQ3_XXS>{}); break;
        case DType::IQ2_S:   run_kq(std::integral_constant<DType, DType::IQ2_S>{});   break;
        case DType::IQ2_XXS: run_kq(std::integral_constant<DType, DType::IQ2_XXS>{}); break;
        case DType::Q3_K:    run_kq(std::integral_constant<DType, DType::Q3_K>{});    break;
        case DType::Q2_K:    run_kq(std::integral_constant<DType, DType::Q2_K>{});    break;
        default:
            throw std::runtime_error("launch_linear_prequant: unsupported dtype");
    }
}

void launch_linear_kquant(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    const int K = static_cast<int>(W.ne[0]);
    const size_t x_f16_bytes = static_cast<size_t>(T) * K * sizeof(__half);
    const size_t align_x = (x_f16_bytes + 255) & ~size_t(255);

    if (cublas != nullptr && T >= 16) {
        const bool use_tc = device_has_tensor_cores(W.device);
        if (use_tc && align_x + static_cast<size_t>(K) * sizeof(__half) <= scratch_bytes) {
            switch (W.dtype) {
                case DType::Q6_K:    dispatch_linear_kquant_prefill_f16<DType::Q6_K>   (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::Q5_K:    dispatch_linear_kquant_prefill_f16<DType::Q5_K>   (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::Q4_K:    dispatch_linear_kquant_prefill_f16<DType::Q4_K>   (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::IQ4_XS:  dispatch_linear_kquant_prefill_f16<DType::IQ4_XS> (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::IQ3_S:   dispatch_linear_kquant_prefill_f16<DType::IQ3_S>  (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::IQ3_XXS: dispatch_linear_kquant_prefill_f16<DType::IQ3_XXS>(cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::IQ2_S:   dispatch_linear_kquant_prefill_f16<DType::IQ2_S>  (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::IQ2_XXS: dispatch_linear_kquant_prefill_f16<DType::IQ2_XXS>(cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::Q3_K:    dispatch_linear_kquant_prefill_f16<DType::Q3_K>   (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::Q2_K:    dispatch_linear_kquant_prefill_f16<DType::Q2_K>   (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                default: break;
            }
        } else if (!use_tc && static_cast<size_t>(K) * sizeof(float) <= scratch_bytes) {
            switch (W.dtype) {
                case DType::Q6_K:    dispatch_linear_kquant_prefill_f32<DType::Q6_K>   (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::Q5_K:    dispatch_linear_kquant_prefill_f32<DType::Q5_K>   (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::Q4_K:    dispatch_linear_kquant_prefill_f32<DType::Q4_K>   (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::IQ4_XS:  dispatch_linear_kquant_prefill_f32<DType::IQ4_XS> (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::IQ3_S:   dispatch_linear_kquant_prefill_f32<DType::IQ3_S>  (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::IQ3_XXS: dispatch_linear_kquant_prefill_f32<DType::IQ3_XXS>(cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::IQ2_S:   dispatch_linear_kquant_prefill_f32<DType::IQ2_S>  (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::IQ2_XXS: dispatch_linear_kquant_prefill_f32<DType::IQ2_XXS>(cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::Q3_K:    dispatch_linear_kquant_prefill_f32<DType::Q3_K>   (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                case DType::Q2_K:    dispatch_linear_kquant_prefill_f32<DType::Q2_K>   (cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream); return;
                default: break;
            }
        }
    }

    launch_quantize_q8_1(d_x, d_scratch, K, T, stream);
    launch_linear_prequant(W, d_scratch, d_y, T, stream);
}

void launch_embedding_kquant(
    const DeviceTensor & tok_embd, const int32_t * d_tokens, float * d_out,
    int n_embd, int hc_mult, int n_tokens, cudaStream_t stream) {
    dim3 grid(n_embd >> 8, n_tokens);
    if (tok_embd.dtype == DType::Q6_K && tok_embd.is_soa) {
        embedding_q6_k_soa_kernel<<<grid, 32, 0, stream>>>(
            static_cast<const uint8_t *>(tok_embd.ptr),
            static_cast<const uint8_t *>(tok_embd.aux_ptr),
            static_cast<const int8_t *>(tok_embd.scales_ptr),
            static_cast<const uint16_t *>(tok_embd.d_ptr),
            d_tokens, d_out, n_embd, hc_mult);
        return;
    }
    const auto * tbl = static_cast<const uint8_t *>(tok_embd.ptr);
    if (tok_embd.dtype == DType::Q5_K) {
        embedding_kquant_generic_kernel<DType::Q5_K><<<grid, 32, 0, stream>>>(tbl, d_tokens, d_out, n_embd, hc_mult);
        return;
    }
    if (tok_embd.dtype == DType::Q4_K) {
        embedding_kquant_generic_kernel<DType::Q4_K><<<grid, 32, 0, stream>>>(tbl, d_tokens, d_out, n_embd, hc_mult);
        return;
    }
    if (tok_embd.dtype == DType::Q3_K) {
        embedding_kquant_generic_kernel<DType::Q3_K><<<grid, 32, 0, stream>>>(tbl, d_tokens, d_out, n_embd, hc_mult);
        return;
    }
    if (tok_embd.dtype == DType::IQ4_XS) {
        embedding_kquant_generic_kernel<DType::IQ4_XS><<<grid, 32, 0, stream>>>(tbl, d_tokens, d_out, n_embd, hc_mult);
        return;
    }
    throw std::runtime_error("launch_embedding_kquant: unsupported tok_embd dtype");
}

void launch_moe_ffn_kquant_with_shared(
    cublasHandle_t cublas,
    const DeviceTensor & gate_exps, const DeviceTensor & up_exps, const DeviceTensor & down_exps,
    const DeviceTensor & gate_shexp, const DeviceTensor & up_shexp, const DeviceTensor & down_shexp,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, float clamp_exp, float clamp_shexp, cudaStream_t stream) {

    const int K     = static_cast<int>(gate_exps.ne[0]);
    const int ff    = static_cast<int>(gate_exps.ne[1]);
    const int ff_sh = static_cast<int>(gate_shexp.ne[1]);

    // Partition scratch (256B aligned):
    // [act: T*k_used*ff float] [sh_g: T*ff_sh float] [sh_u: T*ff_sh float] [xq8] [act_q8] [sh_q8] [rest]
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    auto * base = static_cast<uint8_t *>(d_scratch);
    size_t off = 0;

    auto * act  = reinterpret_cast<float *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
    auto * sh_g = reinterpret_cast<float *>(base + off); off += align256(static_cast<size_t>(T) * ff_sh * sizeof(float));
    auto * sh_u = reinterpret_cast<float *>(base + off); off += align256(static_cast<size_t>(T) * ff_sh * sizeof(float));
    auto * xq8  = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * (K >> 5) * sizeof(BlockQ8_1_Act));
    auto * aq8  = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * k_used * (ff >> 5) * sizeof(BlockQ8_1_Act));
    auto * sq8  = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * (ff_sh >> 5) * sizeof(BlockQ8_1_Act));
    void * rest = base + off;
    const size_t rem_b = (off < scratch_bytes) ? (scratch_bytes - off) : 0;

    // 1. Pre-quantize d_x ONCE -> feeds fused shared gate+up+SiLU AND all k_used routed gate+up!
    launch_quantize_q8_1(d_x, xq8, K, T, stream);

    constexpr int wpb = 8;
    const size_t gu_smem = static_cast<size_t>(K >> 5) * sizeof(BlockQ8_1_Act);
    if (ff_sh > 0 && gate_shexp.valid() && up_shexp.valid()) {
        if (T < 16 && gate_shexp.dtype == DType::Q6_K && gate_shexp.is_soa &&
            up_shexp.dtype == DType::Q6_K && up_shexp.is_soa) {
            gemv_q6_k_soa_gate_up_silu_clamp_kernel<<<dim3((ff_sh + wpb - 1) / wpb, T), dim3(32, wpb), gu_smem, stream>>>(
                static_cast<const int4 *>(gate_shexp.ptr), static_cast<const int4 *>(gate_shexp.aux_ptr),
                static_cast<const int8_t *>(gate_shexp.scales_ptr), static_cast<const uint16_t *>(gate_shexp.d_ptr),
                static_cast<const int4 *>(up_shexp.ptr), static_cast<const int4 *>(up_shexp.aux_ptr),
                static_cast<const int8_t *>(up_shexp.scales_ptr), static_cast<const uint16_t *>(up_shexp.d_ptr),
                xq8, sh_g, ff_sh, K, clamp_shexp);
        } else {
            if (T < 16 && (gate_shexp.dtype == DType::Q8_0 || is_kquant_dtype(gate_shexp.dtype))) {
                launch_linear_prequant(gate_shexp, xq8, sh_g, T, stream);
            } else {
                launch_linear(cublas, gate_shexp, d_x, sh_g, T, rest, rem_b, stream);
            }
            if (T < 16 && (up_shexp.dtype == DType::Q8_0 || is_kquant_dtype(up_shexp.dtype))) {
                launch_linear_prequant(up_shexp, xq8, sh_u, T, stream);
            } else {
                launch_linear(cublas, up_shexp, d_x, sh_u, T, rest, rem_b, stream);
            }
            launch_silu_mul_clamp(sh_g, sh_u, sh_g, static_cast<int64_t>(T) * ff_sh, clamp_shexp, stream);
        }
    }

    // 2. Routed experts: fast grouped-expert multi-token weight-reuse path for prefill (T >= 16)!
    const int n_expert = static_cast<int>(gate_exps.ne[2]);
    const auto * gp = static_cast<const uint8_t *>(gate_exps.ptr);
    const auto * up = static_cast<const uint8_t *>(up_exps.ptr);
    const auto * dp = static_cast<const uint8_t *>(down_exps.ptr);

    constexpr int BT_GU      = 6;
    constexpr int BT_DN      = 8;
    constexpr int R_PER_WARP = 8; // 8 warps * 8 rows/warp = 64 rows/block -> 16x less x_q8/act_q8 traffic!
    const size_t gu_grp_smem = static_cast<size_t>(BT_GU) * (K >> 5)  * sizeof(BlockQ8_1_Act);
    const size_t dn_grp_smem = static_cast<size_t>(BT_DN) * (ff >> 5) * sizeof(BlockQ8_1_Act);

    if (T >= 16 && k_used <= 16 && n_expert > 0 && n_expert <= 512 &&
        gu_grp_smem + 8400 <= 49152 && dn_grp_smem + 8400 <= 49152) {
        auto * exp_offsets  = reinterpret_cast<int32_t *>(rest);
        const size_t off_b  = align256(static_cast<size_t>(n_expert + 1) * sizeof(int32_t));
        auto * sorted_pairs = reinterpret_cast<int32_t *>(static_cast<uint8_t *>(rest) + off_b);
        const size_t sp_b   = align256(static_cast<size_t>(T) * k_used * sizeof(int32_t));
        void * sh_rest      = static_cast<uint8_t *>(rest) + off_b + sp_b;
        const size_t sh_rem = (rem_b > off_b + sp_b) ? (rem_b - off_b - sp_b) : 0;

        group_moe_tokens_kernel<<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
            d_top_ids, exp_offsets, sorted_pairs, T * k_used, n_expert, k_used);

        constexpr int wpb_grp  = 8;
        constexpr int rows_blk = wpb_grp * R_PER_WARP;
        dim3 gu_grp_grid((ff + rows_blk - 1) / rows_blk, n_expert);
        dim3 dn_grp_grid((K  + rows_blk - 1) / rows_blk, n_expert);

        auto run_gu_grp = [&](auto dt_tag) {
            constexpr DType DTG = decltype(dt_tag)::value;
            moe_gate_up_grouped_kquant_kernel<DTG, DTG, BT_GU, R_PER_WARP><<<gu_grp_grid, dim3(32, wpb_grp), gu_grp_smem, stream>>>(
                gp, up, xq8, exp_offsets, sorted_pairs, act, ff, K, k_used, clamp_exp);
        };
        if (gate_exps.dtype == DType::IQ2_S && up_exps.dtype == DType::IQ2_S)          run_gu_grp(std::integral_constant<DType, DType::IQ2_S>{});
        else if (gate_exps.dtype == DType::IQ2_XXS && up_exps.dtype == DType::IQ2_XXS) run_gu_grp(std::integral_constant<DType, DType::IQ2_XXS>{});
        else if (gate_exps.dtype == DType::IQ4_XS && up_exps.dtype == DType::IQ4_XS)   run_gu_grp(std::integral_constant<DType, DType::IQ4_XS>{});
        else if (gate_exps.dtype == DType::Q2_K && up_exps.dtype == DType::Q2_K)       run_gu_grp(std::integral_constant<DType, DType::Q2_K>{});
        else if (gate_exps.dtype == DType::Q3_K && up_exps.dtype == DType::Q3_K)       run_gu_grp(std::integral_constant<DType, DType::Q3_K>{});
        else if (gate_exps.dtype == DType::Q4_K && up_exps.dtype == DType::Q4_K)       run_gu_grp(std::integral_constant<DType, DType::Q4_K>{});
        else if (gate_exps.dtype == DType::Q5_K && up_exps.dtype == DType::Q5_K)       run_gu_grp(std::integral_constant<DType, DType::Q5_K>{});
        else if (gate_exps.dtype == DType::IQ3_S && up_exps.dtype == DType::IQ3_S)     run_gu_grp(std::integral_constant<DType, DType::IQ3_S>{});
        else if (gate_exps.dtype == DType::IQ3_XXS && up_exps.dtype == DType::IQ3_XXS) run_gu_grp(std::integral_constant<DType, DType::IQ3_XXS>{});
        else throw std::runtime_error("launch_moe_ffn_kquant (grouped): unsupported gate/up dtype combo");

        // Quantize routed `act` for grouped down; compute down_shexp into d_out via cuBLAS HGEMM first!
        launch_quantize_q8_1(act, aq8, k_used * ff, T, stream);
        if (down_shexp.valid()) {
            launch_linear(cublas, down_shexp, sh_g, d_out, T, sh_rest, sh_rem, stream);
        } else {
            cudaMemsetAsync(d_out, 0, static_cast<size_t>(T) * K * sizeof(float), stream);
        }

        // Use deterministic down_slots[T, k_used, K] when it fits in sh_rem (zero atomics + bit-exact!)
        const size_t slots_bytes = static_cast<size_t>(T) * k_used * K * sizeof(float);
        const bool use_slots = (slots_bytes <= sh_rem);
        float * dn_target = use_slots ? reinterpret_cast<float *>(sh_rest) : d_out;

        auto run_dn_grp = [&](auto dt_tag) {
            constexpr DType DTD = decltype(dt_tag)::value;
            if (use_slots) {
                moe_down_grouped_kquant_kernel<DTD, BT_DN, R_PER_WARP, true><<<dn_grp_grid, dim3(32, wpb_grp), dn_grp_smem, stream>>>(
                    dp, aq8, d_top_w, exp_offsets, sorted_pairs, dn_target, K, ff, k_used);
            } else {
                moe_down_grouped_kquant_kernel<DTD, BT_DN, R_PER_WARP, false><<<dn_grp_grid, dim3(32, wpb_grp), dn_grp_smem, stream>>>(
                    dp, aq8, d_top_w, exp_offsets, sorted_pairs, dn_target, K, ff, k_used);
            }
        };
        if (down_exps.dtype == DType::Q6_K && down_exps.is_soa) {
            if (use_slots) {
                moe_down_grouped_q6_k_soa_kernel<BT_DN, R_PER_WARP, true><<<dn_grp_grid, dim3(32, wpb_grp), dn_grp_smem, stream>>>(
                    static_cast<const int4 *>(down_exps.ptr), static_cast<const int4 *>(down_exps.aux_ptr),
                    static_cast<const int8_t *>(down_exps.scales_ptr), static_cast<const uint16_t *>(down_exps.d_ptr),
                    aq8, d_top_w, exp_offsets, sorted_pairs, dn_target, K, ff, k_used);
            } else {
                moe_down_grouped_q6_k_soa_kernel<BT_DN, R_PER_WARP, false><<<dn_grp_grid, dim3(32, wpb_grp), dn_grp_smem, stream>>>(
                    static_cast<const int4 *>(down_exps.ptr), static_cast<const int4 *>(down_exps.aux_ptr),
                    static_cast<const int8_t *>(down_exps.scales_ptr), static_cast<const uint16_t *>(down_exps.d_ptr),
                    aq8, d_top_w, exp_offsets, sorted_pairs, dn_target, K, ff, k_used);
            }
        }
        else if (down_exps.dtype == DType::IQ3_S)   run_dn_grp(std::integral_constant<DType, DType::IQ3_S>{});
        else if (down_exps.dtype == DType::IQ3_XXS) run_dn_grp(std::integral_constant<DType, DType::IQ3_XXS>{});
        else if (down_exps.dtype == DType::IQ4_XS)  run_dn_grp(std::integral_constant<DType, DType::IQ4_XS>{});
        else if (down_exps.dtype == DType::Q3_K)    run_dn_grp(std::integral_constant<DType, DType::Q3_K>{});
        else if (down_exps.dtype == DType::Q4_K)    run_dn_grp(std::integral_constant<DType, DType::Q4_K>{});
        else if (down_exps.dtype == DType::Q5_K)    run_dn_grp(std::integral_constant<DType, DType::Q5_K>{});
        else if (down_exps.dtype == DType::IQ2_S)   run_dn_grp(std::integral_constant<DType, DType::IQ2_S>{});
        else if (down_exps.dtype == DType::IQ2_XXS) run_dn_grp(std::integral_constant<DType, DType::IQ2_XXS>{});
        else throw std::runtime_error("launch_moe_ffn_kquant (grouped): unsupported down dtype");

        if (use_slots) {
            moe_reduce_slots_add_kernel<<<dim3((K + 255) / 256, T), 256, 0, stream>>>(dn_target, d_out, K, k_used);
        }
        return;
    }

    // 2. Decode / low-batch (T < 16): Routed experts gate + up + SwiGLU (shared-memory BlockQ8_1 + LUT + dp4a!)
    dim3 gu_grid((ff + wpb - 1) / wpb, k_used, T);

    if (gate_exps.dtype == DType::IQ2_S && up_exps.dtype == DType::IQ2_S) {
        moe_gate_up_kquant_q8_1_kernel<DType::IQ2_S, DType::IQ2_S><<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(gp, up, xq8, d_top_ids, act, ff, K, k_used, clamp_exp);
    } else if (gate_exps.dtype == DType::IQ2_XXS && up_exps.dtype == DType::IQ2_XXS) {
        moe_gate_up_kquant_q8_1_kernel<DType::IQ2_XXS, DType::IQ2_XXS><<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(gp, up, xq8, d_top_ids, act, ff, K, k_used, clamp_exp);
    } else if (gate_exps.dtype == DType::IQ4_XS && up_exps.dtype == DType::IQ4_XS) {
        moe_gate_up_kquant_q8_1_kernel<DType::IQ4_XS, DType::IQ4_XS><<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(gp, up, xq8, d_top_ids, act, ff, K, k_used, clamp_exp);
    } else if (gate_exps.dtype == DType::Q2_K && up_exps.dtype == DType::Q2_K) {
        moe_gate_up_kquant_q8_1_kernel<DType::Q2_K, DType::Q2_K><<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(gp, up, xq8, d_top_ids, act, ff, K, k_used, clamp_exp);
    } else if (gate_exps.dtype == DType::Q3_K && up_exps.dtype == DType::Q3_K) {
        moe_gate_up_kquant_q8_1_kernel<DType::Q3_K, DType::Q3_K><<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(gp, up, xq8, d_top_ids, act, ff, K, k_used, clamp_exp);
    } else if (gate_exps.dtype == DType::Q4_K && up_exps.dtype == DType::Q4_K) {
        moe_gate_up_kquant_q8_1_kernel<DType::Q4_K, DType::Q4_K><<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(gp, up, xq8, d_top_ids, act, ff, K, k_used, clamp_exp);
    } else if (gate_exps.dtype == DType::Q5_K && up_exps.dtype == DType::Q5_K) {
        moe_gate_up_kquant_q8_1_kernel<DType::Q5_K, DType::Q5_K><<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(gp, up, xq8, d_top_ids, act, ff, K, k_used, clamp_exp);
    } else if (gate_exps.dtype == DType::IQ3_S && up_exps.dtype == DType::IQ3_S) {
        moe_gate_up_kquant_q8_1_kernel<DType::IQ3_S, DType::IQ3_S><<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(gp, up, xq8, d_top_ids, act, ff, K, k_used, clamp_exp);
    } else if (gate_exps.dtype == DType::IQ3_XXS && up_exps.dtype == DType::IQ3_XXS) {
        moe_gate_up_kquant_q8_1_kernel<DType::IQ3_XXS, DType::IQ3_XXS><<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(gp, up, xq8, d_top_ids, act, ff, K, k_used, clamp_exp);
    } else {
        throw std::runtime_error("launch_moe_ffn_kquant: unsupported gate/up dtype combo");
    }

    // 3. Quantize routed `act` + shared `sh_g` -> Fused Down + Shared Q6_K SoA Down in 3 KB streaming smem!
    launch_quantize_q8_1(act,  aq8, k_used * ff, T, stream);
    if (ff_sh > 0 && down_shexp.valid()) {
        launch_quantize_q8_1(sh_g, sq8, ff_sh, T, stream);
    }

    dim3 dn_grid((K + wpb - 1) / wpb, T);
    const size_t dn_smem = static_cast<size_t>(std::max(ff >> 5, ff_sh >> 5)) * sizeof(BlockQ8_1_Act);
    const bool has_sh = (down_shexp.dtype == DType::Q6_K && down_shexp.is_soa);
    const auto * sh_ql = has_sh ? static_cast<const int4 *>(down_shexp.ptr)           : nullptr;
    const auto * sh_qh = has_sh ? static_cast<const int4 *>(down_shexp.aux_ptr)       : nullptr;
    const auto * sh_sc = has_sh ? static_cast<const int8_t *>(down_shexp.scales_ptr)  : nullptr;
    const auto * sh_d  = has_sh ? static_cast<const uint16_t *>(down_shexp.d_ptr)     : nullptr;

    auto run_dn = [&](auto dt_tag) {
        constexpr DType DTD = decltype(dt_tag)::value;
        moe_down_kquant_q8_1_kernel<DTD><<<dn_grid, dim3(32, wpb), dn_smem, stream>>>(
            dp, sh_ql, sh_qh, sh_sc, sh_d, aq8, sq8, d_top_ids, d_top_w, d_out, K, ff, ff_sh, k_used);
    };

    if (down_exps.dtype == DType::Q6_K && down_exps.is_soa) {
        moe_down_q6_k_soa_kernel<<<dn_grid, dim3(32, wpb), dn_smem, stream>>>(
            static_cast<const int4 *>(down_exps.ptr), static_cast<const int4 *>(down_exps.aux_ptr),
            static_cast<const int8_t *>(down_exps.scales_ptr), static_cast<const uint16_t *>(down_exps.d_ptr),
            sh_ql, sh_qh, sh_sc, sh_d, aq8, sq8, d_top_ids, d_top_w, d_out, K, ff, ff_sh, k_used);
    }
    else if (down_exps.dtype == DType::IQ3_S)   run_dn(std::integral_constant<DType, DType::IQ3_S>{});
    else if (down_exps.dtype == DType::IQ3_XXS) run_dn(std::integral_constant<DType, DType::IQ3_XXS>{});
    else if (down_exps.dtype == DType::IQ4_XS)  run_dn(std::integral_constant<DType, DType::IQ4_XS>{});
    else if (down_exps.dtype == DType::Q3_K)    run_dn(std::integral_constant<DType, DType::Q3_K>{});
    else if (down_exps.dtype == DType::Q4_K)    run_dn(std::integral_constant<DType, DType::Q4_K>{});
    else if (down_exps.dtype == DType::Q5_K)    run_dn(std::integral_constant<DType, DType::Q5_K>{});
    else if (down_exps.dtype == DType::IQ2_S)   run_dn(std::integral_constant<DType, DType::IQ2_S>{});
    else if (down_exps.dtype == DType::IQ2_XXS) run_dn(std::integral_constant<DType, DType::IQ2_XXS>{});
    else throw std::runtime_error("launch_moe_ffn_kquant: unsupported down dtype");

    if (!has_sh && down_shexp.valid()) {
        auto * sh_out = reinterpret_cast<float *>(rest);
        void * sh_rem = static_cast<uint8_t *>(rest) + align256(static_cast<size_t>(T) * K * sizeof(float));
        const size_t sh_rem_b = (rem_b > align256(static_cast<size_t>(T) * K * sizeof(float)))
                                    ? (rem_b - align256(static_cast<size_t>(T) * K * sizeof(float))) : 0;
        launch_linear(cublas, down_shexp, sh_g, sh_out, T, sh_rem, sh_rem_b, stream);
        launch_add_inplace(d_out, sh_out, static_cast<int64_t>(T) * K, stream);
    }
}

namespace {

__device__ __forceinline__ float inkling_situ_act(float g, float u, float beta, float lb) {
    const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g)));
    const float b = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
    return a * b;
}

template <DType DT>
__global__ void inkling2_moe_gate_up_situ_kquant_kernel(
    const uint8_t *       __restrict__ gu_exps, // [n_expert, 2*ff, K]
    const BlockQ8_1_Act * __restrict__ x_q8,    // [T, K/32]
    const int32_t *       __restrict__ top_ids, // [T, k_used]
    float *               __restrict__ act,     // [T, k_used, ff]
    int ff, int K, int k_used, float beta, float lb) {

    extern __shared__ BlockQ8_1_Act s_x[];
    __shared__ uint64_t s_iq2_grid[DT == DType::IQ2_S ? 1024 : (DT == DType::IQ2_XXS ? 256 : 1)];

    const int slot     = blockIdx.y;
    const int t        = blockIdx.z;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;

    const BlockQ8_1_Act * x_row = x_q8 + static_cast<size_t>(t) * K_blocks;
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(x_row);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];

    if constexpr (DT == DType::IQ2_S) {
        for (int i = tid; i < 1024; i += nthr) s_iq2_grid[i] = __ldg(&iq2s_grid_d[i]);
    } else if constexpr (DT == DType::IQ2_XXS) {
        for (int i = tid; i < 256; i += nthr) s_iq2_grid[i] = __ldg(&iq2xxs_grid_d[i]);
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= ff) return;

    const int lane = threadIdx.x;
    const int eid  = top_ids[static_cast<size_t>(t) * k_used + slot];
    constexpr size_t B = block_bytes_of<DT>();
    const size_t row_bytes = static_cast<size_t>(nb) * B;
    const uint8_t * g_row = gu_exps + (static_cast<size_t>(eid) * (2 * ff) + row)      * row_bytes;
    const uint8_t * u_row = gu_exps + (static_cast<size_t>(eid) * (2 * ff) + ff + row) * row_bytes;

    float ag = 0.0f, au = 0.0f;
    #pragma unroll 2
    for (int bi = 0; bi < nb; ++bi) {
        const BlockQ8_1_Act * xb = s_x + bi * 8;
        if constexpr (DT == DType::IQ2_S) {
            ag += dot_iq2_s_sgrid(g_row + static_cast<size_t>(bi) * B, xb, s_iq2_grid, lane);
            au += dot_iq2_s_sgrid(u_row + static_cast<size_t>(bi) * B, xb, s_iq2_grid, lane);
        } else if constexpr (DT == DType::IQ2_XXS) {
            ag += dot_iq2_xxs_sgrid(g_row + static_cast<size_t>(bi) * B, xb, s_iq2_grid, lane);
            au += dot_iq2_xxs_sgrid(u_row + static_cast<size_t>(bi) * B, xb, s_iq2_grid, lane);
        } else {
            ag += dot_kblock_q8_1<DT>(g_row + static_cast<size_t>(bi) * B, xb, lane);
            au += dot_kblock_q8_1<DT>(u_row + static_cast<size_t>(bi) * B, xb, lane);
        }
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) {
        act[(static_cast<size_t>(t) * k_used + slot) * ff + row] = inkling_situ_act(ag, au, beta, lb);
    }
}

__global__ void inkling2_moe_gate_up_situ_q6_k_soa_kernel(
    const int4 *          __restrict__ W_ql,
    const int4 *          __restrict__ W_qh,
    const int8_t *        __restrict__ W_sc,
    const uint16_t *      __restrict__ W_d,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ top_ids,
    float *               __restrict__ act,
    int ff, int K, int k_used, float beta, float lb) {

    extern __shared__ BlockQ8_1_Act s_x[];
    const int slot     = blockIdx.y;
    const int t        = blockIdx.z;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;

    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(t) * K_blocks);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= ff) return;

    const int lane = threadIdx.x;
    const int eid  = top_ids[static_cast<size_t>(t) * k_used + slot];
    const size_t g_blk = (static_cast<size_t>(eid) * (2 * ff) + row)      * nb;
    const size_t u_blk = (static_cast<size_t>(eid) * (2 * ff) + ff + row) * nb;
    const float ag = dot_q6_k_soa_row_q8_1(W_ql + g_blk * 8, W_qh + g_blk * 4, W_sc + g_blk * 16, W_d + g_blk, s_x, nb, lane);
    const float au = dot_q6_k_soa_row_q8_1(W_ql + u_blk * 8, W_qh + u_blk * 4, W_sc + u_blk * 16, W_d + u_blk, s_x, nb, lane);
    if (lane == 0) {
        act[(static_cast<size_t>(t) * k_used + slot) * ff + row] = inkling_situ_act(ag, au, beta, lb);
    }
}

__device__ __forceinline__ float dot_iq4_nl_row_q8_1(
    const uint8_t *       __restrict__ row_ptr,
    const BlockQ8_1_Act * __restrict__ s_x,
    int K_blocks, int lane) {

    float acc = 0.0f;
    for (int b = lane; b < K_blocks; b += 32) {
        const uint8_t * blk = row_ptr + static_cast<size_t>(b) * 18;
        const uint16_t * p16 = reinterpret_cast<const uint16_t *>(blk);
        const float d = half_bits_to_f32(__ldg(&p16[0]));
        const int * x32 = reinterpret_cast<const int *>(s_x[b].qs);

        int sum = 0;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t q4 = static_cast<uint32_t>(__ldg(&p16[1 + g * 2]))
                              | (static_cast<uint32_t>(__ldg(&p16[2 + g * 2])) << 16);
            const int w0 = static_cast<int>(
                  static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >>  0) & 0xFu])))
                | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >>  8) & 0xFu]))) << 8)
                | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 16) & 0xFu]))) << 16)
                | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 24) & 0xFu]))) << 24));
            const int w1 = static_cast<int>(
                  static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >>  4) & 0xFu])))
                | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 12) & 0xFu]))) << 8)
                | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 20) & 0xFu]))) << 16)
                | (static_cast<uint32_t>(static_cast<uint8_t>(__ldg(&kvalues_iq4nl_d[(q4 >> 28) & 0xFu]))) << 24));
            sum = dp4a_s32(w0, x32[g], sum);
            sum = dp4a_s32(w1, x32[4 + g], sum);
        }
        acc += (d * s_x[b].d) * static_cast<float>(sum);
    }
    return warp_sum(acc);
}

__global__ void inkling2_moe_gate_up_situ_iq4_nl_kernel(
    const uint8_t *       __restrict__ gu_exps,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ top_ids,
    float *               __restrict__ act,
    int ff, int K, int k_used, float beta, float lb) {

    extern __shared__ BlockQ8_1_Act s_x[];
    const int slot     = blockIdx.y;
    const int t        = blockIdx.z;
    const int K_blocks = K >> 5;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;

    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(t) * K_blocks);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= ff) return;

    const int lane = threadIdx.x;
    const int eid  = top_ids[static_cast<size_t>(t) * k_used + slot];
    const size_t row_bytes = static_cast<size_t>(K_blocks) * 18;
    const uint8_t * g_row = gu_exps + (static_cast<size_t>(eid) * (2 * ff) + row)      * row_bytes;
    const uint8_t * u_row = gu_exps + (static_cast<size_t>(eid) * (2 * ff) + ff + row) * row_bytes;

    const float ag = dot_iq4_nl_row_q8_1(g_row, s_x, K_blocks, lane);
    const float au = dot_iq4_nl_row_q8_1(u_row, s_x, K_blocks, lane);
    if (lane == 0) {
        act[(static_cast<size_t>(t) * k_used + slot) * ff + row] = inkling_situ_act(ag, au, beta, lb);
    }
}

__global__ void inkling2_moe_down_iq4_nl_kernel(
    const uint8_t *       __restrict__ dn_exps,
    const BlockQ8_1_Act * __restrict__ act_q8,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ out,
    int K, int ff, int k_used) {

    extern __shared__ BlockQ8_1_Act s_a[];
    const int t       = blockIdx.y;
    const int row     = blockIdx.x * blockDim.y + threadIdx.y;
    const int ff_blks = ff >> 5;
    const int tid     = threadIdx.y * 32 + threadIdx.x;
    const int nthr    = blockDim.y * 32;
    const int lane    = threadIdx.x;
    const int n_i4    = (ff_blks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const size_t row_bytes = static_cast<size_t>(ff_blks) * 18;

    float total = 0.0f;
    for (int slot = 0; slot < k_used; ++slot) {
        const int4 * src = reinterpret_cast<const int4 *>(act_q8 + (static_cast<size_t>(t) * k_used + slot) * ff_blks);
        int4 *       dst = reinterpret_cast<int4 *>(s_a);
        for (int i = tid; i < n_i4; i += nthr) dst[i] = src[i];
        __syncthreads();

        if (row < K) {
            const int   eid = top_ids[static_cast<size_t>(t) * k_used + slot];
            const float ew  = top_w  [static_cast<size_t>(t) * k_used + slot];
            const uint8_t * d_row = dn_exps + (static_cast<size_t>(eid) * K + row) * row_bytes;
            total += ew * dot_iq4_nl_row_q8_1(d_row, s_a, ff_blks, lane);
        }
        __syncthreads();
    }
    if (row < K && lane == 0) out[static_cast<size_t>(t) * K + row] = total;
}

__global__ void inkling2_moe_down_q8_0_soa_kernel(
    const int4 *          __restrict__ dn_q,
    const uint16_t *      __restrict__ dn_s,
    const BlockQ8_1_Act * __restrict__ act_q8,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ out,
    int K, int ff, int k_used) {

    extern __shared__ BlockQ8_1_Act s_a[];
    const int t       = blockIdx.y;
    const int row     = blockIdx.x * blockDim.y + threadIdx.y;
    const int ff_blks = ff >> 5;
    const int tid     = threadIdx.y * 32 + threadIdx.x;
    const int nthr    = blockDim.y * 32;
    const int lane    = threadIdx.x;
    const int blk_w   = lane >> 1;
    const int half    = lane & 1;
    const int n_i4    = (ff_blks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;

    float total = 0.0f;
    for (int slot = 0; slot < k_used; ++slot) {
        const int4 * src = reinterpret_cast<const int4 *>(act_q8 + (static_cast<size_t>(t) * k_used + slot) * ff_blks);
        int4 *       dst = reinterpret_cast<int4 *>(s_a);
        for (int i = tid; i < n_i4; i += nthr) dst[i] = src[i];
        __syncthreads();

        if (row < K) {
            const int   eid = top_ids[static_cast<size_t>(t) * k_used + slot];
            const float ew  = top_w  [static_cast<size_t>(t) * k_used + slot];
            const size_t roff = (static_cast<size_t>(eid) * K + row) * ff_blks;
            const int4 *     dq = dn_q + roff * 2;
            const uint16_t * ds = dn_s + roff;
            float acc = 0.0f;
            for (int b = blk_w; b < ff_blks; b += 16) {
                const int4 w4 = __ldg(&dq[b * 2 + half]);
                const int4 a4 = reinterpret_cast<const int4 *>(s_a[b].qs)[half];
                int si = dp4a_s32(w4.x, a4.x, dp4a_s32(w4.y, a4.y, dp4a_s32(w4.z, a4.z, dp4a_s32(w4.w, a4.w, 0))));
                __half dh; *reinterpret_cast<uint16_t *>(&dh) = __ldg(&ds[b]);
                acc += (__half2float(dh) * s_a[b].d) * static_cast<float>(si);
            }
            total += ew * warp_sum(acc);
        }
        __syncthreads();
    }
    if (row < K && lane == 0) out[static_cast<size_t>(t) * K + row] = total;
}

} // namespace

void launch_inkling2_moe_ffn_kquant(
    const DeviceTensor & gate_up_exps, const DeviceTensor & down_exps,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, float situ_beta, float situ_linear_beta, cudaStream_t stream) {

    const int K  = static_cast<int>(gate_up_exps.ne[0]);
    const int ff = static_cast<int>(gate_up_exps.ne[1]) / 2;

    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    auto * base = static_cast<uint8_t *>(d_scratch);
    size_t off = 0;
    auto * xq8 = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * (K >> 5) * sizeof(BlockQ8_1_Act));
    auto * act = reinterpret_cast<float *>        (base + off); off += align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
    auto * aq8 = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * k_used * (ff >> 5) * sizeof(BlockQ8_1_Act));
    (void) scratch_bytes;

    launch_quantize_q8_1(d_x, xq8, K, T, stream);

    constexpr int wpb = 8;
    const size_t gu_smem = static_cast<size_t>(K >> 5) * sizeof(BlockQ8_1_Act);
    dim3 gu_grid((ff + wpb - 1) / wpb, k_used, T);
    const auto * gup = static_cast<const uint8_t *>(gate_up_exps.ptr);

    auto run_gu = [&](auto dt_tag) {
        constexpr DType DTG = decltype(dt_tag)::value;
        inkling2_moe_gate_up_situ_kquant_kernel<DTG><<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(
            gup, xq8, d_top_ids, act, ff, K, k_used, situ_beta, situ_linear_beta);
    };

    if (gate_up_exps.dtype == DType::IQ4_NL) {
        inkling2_moe_gate_up_situ_iq4_nl_kernel<<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(
            gup, xq8, d_top_ids, act, ff, K, k_used, situ_beta, situ_linear_beta);
    } else if (gate_up_exps.dtype == DType::Q6_K && gate_up_exps.is_soa) {
        inkling2_moe_gate_up_situ_q6_k_soa_kernel<<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(
            static_cast<const int4 *>(gate_up_exps.ptr), static_cast<const int4 *>(gate_up_exps.aux_ptr),
            static_cast<const int8_t *>(gate_up_exps.scales_ptr), static_cast<const uint16_t *>(gate_up_exps.d_ptr),
            xq8, d_top_ids, act, ff, K, k_used, situ_beta, situ_linear_beta);
    }
    else if (gate_up_exps.dtype == DType::Q2_K)    run_gu(std::integral_constant<DType, DType::Q2_K>{});
    else if (gate_up_exps.dtype == DType::Q3_K)    run_gu(std::integral_constant<DType, DType::Q3_K>{});
    else if (gate_up_exps.dtype == DType::Q4_K)    run_gu(std::integral_constant<DType, DType::Q4_K>{});
    else if (gate_up_exps.dtype == DType::Q5_K)    run_gu(std::integral_constant<DType, DType::Q5_K>{});
    else if (gate_up_exps.dtype == DType::IQ4_XS)  run_gu(std::integral_constant<DType, DType::IQ4_XS>{});
    else if (gate_up_exps.dtype == DType::IQ3_S)   run_gu(std::integral_constant<DType, DType::IQ3_S>{});
    else if (gate_up_exps.dtype == DType::IQ3_XXS) run_gu(std::integral_constant<DType, DType::IQ3_XXS>{});
    else if (gate_up_exps.dtype == DType::IQ2_S)   run_gu(std::integral_constant<DType, DType::IQ2_S>{});
    else if (gate_up_exps.dtype == DType::IQ2_XXS) run_gu(std::integral_constant<DType, DType::IQ2_XXS>{});
    else throw std::runtime_error("launch_inkling2_moe_ffn_kquant: unsupported gate_up_exps dtype");

    launch_quantize_q8_1(act, aq8, k_used * ff, T, stream);

    dim3 dn_grid((K + wpb - 1) / wpb, T);
    const size_t dn_smem = static_cast<size_t>(ff >> 5) * sizeof(BlockQ8_1_Act);
    const auto * dp = static_cast<const uint8_t *>(down_exps.ptr);

    auto run_dn = [&](auto dt_tag) {
        constexpr DType DTD = decltype(dt_tag)::value;
        moe_down_kquant_q8_1_kernel<DTD><<<dn_grid, dim3(32, wpb), dn_smem, stream>>>(
            dp, nullptr, nullptr, nullptr, nullptr, aq8, nullptr, d_top_ids, d_top_w, d_out, K, ff, 0, k_used);
    };

    if (down_exps.dtype == DType::IQ4_NL) {
        inkling2_moe_down_iq4_nl_kernel<<<dn_grid, dim3(32, wpb), dn_smem, stream>>>(
            dp, aq8, d_top_ids, d_top_w, d_out, K, ff, k_used);
    } else if (down_exps.dtype == DType::Q8_0 && down_exps.is_soa) {
        inkling2_moe_down_q8_0_soa_kernel<<<dn_grid, dim3(32, wpb), dn_smem, stream>>>(
            static_cast<const int4 *>(down_exps.ptr), static_cast<const uint16_t *>(down_exps.scales_ptr),
            aq8, d_top_ids, d_top_w, d_out, K, ff, k_used);
    } else if (down_exps.dtype == DType::Q6_K && down_exps.is_soa) {
        moe_down_q6_k_soa_kernel<<<dn_grid, dim3(32, wpb), dn_smem, stream>>>(
            static_cast<const int4 *>(down_exps.ptr), static_cast<const int4 *>(down_exps.aux_ptr),
            static_cast<const int8_t *>(down_exps.scales_ptr), static_cast<const uint16_t *>(down_exps.d_ptr),
            nullptr, nullptr, nullptr, nullptr, aq8, nullptr, d_top_ids, d_top_w, d_out, K, ff, 0, k_used);
    }
    else if (down_exps.dtype == DType::Q2_K)    run_dn(std::integral_constant<DType, DType::Q2_K>{});
    else if (down_exps.dtype == DType::Q3_K)    run_dn(std::integral_constant<DType, DType::Q3_K>{});
    else if (down_exps.dtype == DType::Q4_K)    run_dn(std::integral_constant<DType, DType::Q4_K>{});
    else if (down_exps.dtype == DType::Q5_K)    run_dn(std::integral_constant<DType, DType::Q5_K>{});
    else if (down_exps.dtype == DType::IQ4_XS)  run_dn(std::integral_constant<DType, DType::IQ4_XS>{});
    else if (down_exps.dtype == DType::IQ3_S)   run_dn(std::integral_constant<DType, DType::IQ3_S>{});
    else if (down_exps.dtype == DType::IQ3_XXS) run_dn(std::integral_constant<DType, DType::IQ3_XXS>{});
    else if (down_exps.dtype == DType::IQ2_S)   run_dn(std::integral_constant<DType, DType::IQ2_S>{});
    else if (down_exps.dtype == DType::IQ2_XXS) run_dn(std::integral_constant<DType, DType::IQ2_XXS>{});
    else throw std::runtime_error("launch_inkling2_moe_ffn_kquant: unsupported down_exps dtype");
}

} // namespace rookery
