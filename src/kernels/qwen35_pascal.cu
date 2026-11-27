#include "core/device.h"
#include "kernels/iq_tables.cuh"
#include "kernels/kernels.h"
#include "kernels/q8_1_act.cuh"

#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace rookery {

namespace {

struct BlockIQ3_XXS_P {
    uint16_t d;
    uint8_t  qs[96];
};
static_assert(sizeof(BlockIQ3_XXS_P) == 98, "BlockIQ3_XXS_P size");

struct BlockIQ4_XS_P {
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[4];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockIQ4_XS_P) == 136, "BlockIQ4_XS_P size");

struct Q35TileDesc {
    int32_t eid;
    int32_t start;
    int32_t count;
};

__device__ __forceinline__ float q35_half_bits_to_f32(uint16_t bits) {
    __half h;
    *reinterpret_cast<uint16_t *>(&h) = bits;
    return __half2float(h);
}

__device__ __forceinline__ float q35_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float q35_warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ int q35_apply_signs4(uint32_t g4, uint32_t s4) {
    const uint32_t add4 = (s4 * 0x00204081u) & 0x01010101u;
    const uint32_t xor4 = add4 * 0xFFu;
    return static_cast<int>((g4 ^ xor4) + add4);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
__device__ __forceinline__ void q35_i8x4_to_h2x2(int w_i32, __half2 & h_lo, __half2 & h_hi) {
    const uint32_t u  = static_cast<uint32_t>(w_i32) ^ 0x80808080u;
    const uint32_t p0 = __byte_perm(u, 0x64u, 0x4140u);
    const uint32_t p1 = __byte_perm(u, 0x64u, 0x4342u);
    constexpr uint32_t k1152 = 0x64806480u;
    const __half2 c1152 = *reinterpret_cast<const __half2 *>(&k1152);
    h_lo = __hsub2(*reinterpret_cast<const __half2 *>(&p0), c1152);
    h_hi = __hsub2(*reinterpret_cast<const __half2 *>(&p1), c1152);
}

// Direct FP16 LUT IQ3_XXS decoder: 4 XORs replace 22 integer/PRMT/HSUB2 ALU ops per 8 weights!
__device__ __forceinline__ void q35_decode_iq3xxs_lane_h2(
    const BlockIQ3_XXS_P & b,
    const uint2 *   __restrict__ s_grid_h2,
    const uint2 *   __restrict__ s_sign_mask,
    const uint8_t * __restrict__ s_ksigns,
    int lane,
    __half2 & h0, __half2 & h1, __half2 & h2, __half2 & h3, float & sc) {

    const float d    = q35_half_bits_to_f32(__ldg(&b.d));
    const int   ib32 = lane >> 2;
    const int   l    = lane & 3;
    const uint16_t q01 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs)[lane]);
    const uint16_t ss0 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs + 64)[ib32 * 2 + 0]);
    const uint16_t ss1 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs + 64)[ib32 * 2 + 1]);
    const uint32_t aux32 = static_cast<uint32_t>(ss0) | (static_cast<uint32_t>(ss1) << 16);
    sc = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
    const uint8_t sg = s_ksigns[(aux32 >> (7 * l)) & 127u];
    const uint2 g1 = s_grid_h2[q01 & 0xFFu];
    const uint2 g2 = s_grid_h2[q01 >> 8];
    const uint2 m1 = s_sign_mask[sg & 0xFu];
    const uint2 m2 = s_sign_mask[sg >> 4];
    const uint32_t u0 = g1.x ^ m1.x;
    const uint32_t u1 = g1.y ^ m1.y;
    const uint32_t u2 = g2.x ^ m2.x;
    const uint32_t u3 = g2.y ^ m2.y;
    h0 = *reinterpret_cast<const __half2 *>(&u0);
    h1 = *reinterpret_cast<const __half2 *>(&u1);
    h2 = *reinterpret_cast<const __half2 *>(&u2);
    h3 = *reinterpret_cast<const __half2 *>(&u3);
}

// Direct FP16 LUT IQ4_XS decoder: packs 16-bit half LUT values straight into __half2 (zero PRMT/HSUB2!)
__device__ __forceinline__ void q35_decode_iq4xs_lane_h2(
    const BlockIQ4_XS_P & b,
    const uint16_t * __restrict__ s_iq4_h,
    int lane,
    __half2 & h0, __half2 & h1, __half2 & h2, __half2 & h3, float & sc) {

    const float    d    = q35_half_bits_to_f32(__ldg(&b.d));
    const uint16_t sh   = __ldg(&b.scales_h);
    const int      ib   = lane >> 2;
    const int      sub4 = lane & 3;
    const uint8_t  sl   = __ldg(&b.scales_l[ib >> 1]);
    const int      ls   = ((sl >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    sc = d * static_cast<float>(ls - 32);
    const uint32_t q4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qs)[ib * 4 + sub4]);
    const uint32_t u0 = static_cast<uint32_t>(s_iq4_h[(q4 >>  0) & 0xFu]) | (static_cast<uint32_t>(s_iq4_h[(q4 >>  8) & 0xFu]) << 16);
    const uint32_t u1 = static_cast<uint32_t>(s_iq4_h[(q4 >> 16) & 0xFu]) | (static_cast<uint32_t>(s_iq4_h[(q4 >> 24) & 0xFu]) << 16);
    const uint32_t u2 = static_cast<uint32_t>(s_iq4_h[(q4 >>  4) & 0xFu]) | (static_cast<uint32_t>(s_iq4_h[(q4 >> 12) & 0xFu]) << 16);
    const uint32_t u3 = static_cast<uint32_t>(s_iq4_h[(q4 >> 20) & 0xFu]) | (static_cast<uint32_t>(s_iq4_h[(q4 >> 28) & 0xFu]) << 16);
    h0 = *reinterpret_cast<const __half2 *>(&u0);
    h1 = *reinterpret_cast<const __half2 *>(&u1);
    h2 = *reinterpret_cast<const __half2 *>(&u2);
    h3 = *reinterpret_cast<const __half2 *>(&u3);
}

// Direct contiguous-lane Q6_K SoA decoder (lane 0..31 -> elements lane*8 .. lane*8+7, 1 scale/lane!)
__device__ __forceinline__ void q35_decode_q6k_soa_lane_h2(
    const uint2 *    __restrict__ ql_blk, // 16 uint2s (128 bytes)
    const uint2 *    __restrict__ qh_blk, // 8 uint2s (64 bytes)
    const int8_t *   __restrict__ sc_blk, // 16 int8_t
    uint16_t d_bits,
    int lane,
    __half2 & h0, __half2 & h1, __half2 & h2, __half2 & h3, float & sc) {

    const int h        = lane >> 4;
    const int grp      = (lane >> 2) & 3;
    const int sub8     = lane & 3;
    const int ql_shift = (grp >= 2) ? 4 : 0;
    const int qh_shift = 2 * grp;
    const uint2 ql_v   = __ldg(&ql_blk[h * 8 + (grp & 1) * 4 + sub8]);
    const uint2 qh_v   = __ldg(&qh_blk[h * 4 + sub8]);
    const uint32_t q_lo = ((ql_v.x >> ql_shift) & 0x0F0F0F0Fu) | (((qh_v.x >> qh_shift) & 0x03030303u) << 4);
    const uint32_t q_hi = ((ql_v.y >> ql_shift) & 0x0F0F0F0Fu) | (((qh_v.y >> qh_shift) & 0x03030303u) << 4);

    constexpr uint32_t k1056 = 0x64206420u; // 1056.0 = 1024.0 + 32.0
    const __half2 c1056 = *reinterpret_cast<const __half2 *>(&k1056);
    const uint32_t p0 = __byte_perm(q_lo, 0x00006400u, 0x5150u);
    const uint32_t p1 = __byte_perm(q_lo, 0x00006400u, 0x5352u);
    const uint32_t p2 = __byte_perm(q_hi, 0x00006400u, 0x5150u);
    const uint32_t p3 = __byte_perm(q_hi, 0x00006400u, 0x5352u);
    h0 = __hsub2(*reinterpret_cast<const __half2 *>(&p0), c1056);
    h1 = __hsub2(*reinterpret_cast<const __half2 *>(&p1), c1056);
    h2 = __hsub2(*reinterpret_cast<const __half2 *>(&p2), c1056);
    h3 = __hsub2(*reinterpret_cast<const __half2 *>(&p3), c1056);
    sc = q35_half_bits_to_f32(d_bits) * static_cast<float>(__ldg(&sc_blk[lane >> 1]));
}
#endif

__device__ __forceinline__ void q35_decode_iq3xxs_lane(
    const BlockIQ3_XXS_P & b,
    const uint32_t * __restrict__ s_grid32,
    const uint8_t *  __restrict__ s_ksigns,
    int lane,
    int & w0, int & w1, float & sc) {

    const float d    = q35_half_bits_to_f32(__ldg(&b.d));
    const int   ib32 = lane >> 2;
    const int   l    = lane & 3;
    const uint16_t q01 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs)[lane]);
    const uint16_t ss0 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs + 64)[ib32 * 2 + 0]);
    const uint16_t ss1 = __ldg(&reinterpret_cast<const uint16_t *>(b.qs + 64)[ib32 * 2 + 1]);
    const uint32_t aux32 = static_cast<uint32_t>(ss0) | (static_cast<uint32_t>(ss1) << 16);
    sc = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
    const uint8_t  sg = s_ksigns[(aux32 >> (7 * l)) & 127u];
    const uint32_t g1 = s_grid32[q01 & 0xFFu];
    const uint32_t g2 = s_grid32[q01 >> 8];
    w0 = q35_apply_signs4(g1, sg & 0xFu);
    w1 = q35_apply_signs4(g2, sg >> 4);
}

__device__ __forceinline__ void q35_decode_iq4xs_lane(
    const BlockIQ4_XS_P & b,
    const int8_t * __restrict__ s_iq4,
    int lane,
    int & w0, int & w1, float & sc) {

    const float    d    = q35_half_bits_to_f32(__ldg(&b.d));
    const uint16_t sh   = __ldg(&b.scales_h);
    const int      ib   = lane >> 2;
    const int      sub4 = lane & 3;
    const uint8_t  sl   = __ldg(&b.scales_l[ib >> 1]);
    const int      ls   = ((sl >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    sc = d * static_cast<float>(ls - 32);
    const uint32_t q4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qs)[ib * 4 + sub4]);
    w0 = static_cast<int>(
          static_cast<uint32_t>(static_cast<uint8_t>(s_iq4[(q4 >>  0) & 0xFu]))
        | (static_cast<uint32_t>(static_cast<uint8_t>(s_iq4[(q4 >>  8) & 0xFu])) << 8)
        | (static_cast<uint32_t>(static_cast<uint8_t>(s_iq4[(q4 >> 16) & 0xFu])) << 16)
        | (static_cast<uint32_t>(static_cast<uint8_t>(s_iq4[(q4 >> 24) & 0xFu])) << 24));
    w1 = static_cast<int>(
          static_cast<uint32_t>(static_cast<uint8_t>(s_iq4[(q4 >>  4) & 0xFu]))
        | (static_cast<uint32_t>(static_cast<uint8_t>(s_iq4[(q4 >> 12) & 0xFu])) << 8)
        | (static_cast<uint32_t>(static_cast<uint8_t>(s_iq4[(q4 >> 20) & 0xFu])) << 16)
        | (static_cast<uint32_t>(static_cast<uint8_t>(s_iq4[(q4 >> 28) & 0xFu])) << 24));
}

// Per-row FP16 activation prep for sm_60 pure-HFMA2 MoE prefill:
// Scales each row of K floats into [-4.0, 4.0] as __half2 + stores row_scale = amax / 64.0f
// (compensating for the 16.0x scale folded into weight scales for zero FP16 subnormal loss).
__global__ void q35_prep_row_h2_kernel(
    const float * __restrict__ x,        // [n_rows, K]
    __half2 *     __restrict__ x_h2,     // [n_rows, K / 2]
    float *       __restrict__ row_sc,   // [n_rows]
    int K) {

    const int r    = blockIdx.x;
    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float * xr = x + static_cast<size_t>(r) * K;
    float amax = 0.0f;
    for (int i = tid; i < K; i += 256) {
        amax = fmaxf(amax, fabsf(xr[i]));
    }
    amax = q35_warp_max(amax);
    __shared__ float s_m[8];
    if (lane == 0) s_m[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? s_m[lane] : 0.0f;
        v = q35_warp_max(v);
        if (lane == 0) s_m[0] = v;
    }
    __syncthreads();
    amax = s_m[0];
    if (tid == 0) row_sc[r] = amax * (1.0f / 64.0f);

    const float inv = (amax > 0.0f) ? (4.0f / amax) : 0.0f;
    const int K2 = K >> 1;
    const float2 * xr2 = reinterpret_cast<const float2 *>(xr);
    __half2 *      yh2 = x_h2 + static_cast<size_t>(r) * K2;
    for (int i = tid; i < K2; i += 256) {
        const float2 v = xr2[i];
        yh2[i] = __float22half2_rn(make_float2(v.x * inv, v.y * inv));
    }
}

template <int BT>
__global__ void q35_group_and_tile_kernel(
    const int32_t * __restrict__ top_ids,
    int32_t *       __restrict__ sorted_pairs,
    Q35TileDesc *   __restrict__ tiles,
    int32_t *       __restrict__ n_tiles_out,
    int total_pairs, int n_expert, int k_used) {

    extern __shared__ int32_t s_cnt[];
    const int tid = threadIdx.x;
    for (int e = tid; e < n_expert; e += blockDim.x) s_cnt[e] = 0;
    __syncthreads();

    for (int i = tid; i < total_pairs; i += blockDim.x) {
        const int eid = top_ids[i];
        if (eid >= 0 && eid < n_expert) atomicAdd(&s_cnt[eid], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        int nt  = 0;
        for (int e = 0; e < n_expert; ++e) {
            const int c = s_cnt[e];
            s_cnt[e] = sum;
            for (int off = 0; off < c; off += BT) {
                const int cnt = (off + BT < c) ? BT : (c - off);
                tiles[nt++] = Q35TileDesc{e, sum + off, cnt};
            }
            sum += c;
        }
        *n_tiles_out = nt;
    }
    __syncthreads();

    for (int i = tid; i < total_pairs; i += blockDim.x) {
        const int eid = top_ids[i];
        if (eid >= 0 && eid < n_expert) {
            const int dst  = atomicAdd(&s_cnt[eid], 1);
            const int t    = i / k_used;
            const int slot = i % k_used;
            sorted_pairs[dst] = (t << 4) | slot;
        }
    }
}

__global__ void q35_reduce_slots_kernel(
    const float * __restrict__ slots,
    float *       __restrict__ out,
    int K, int k_used) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= K) return;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(t) * k_used * K + c;
    for (int s = 0; s < k_used; ++s) {
        sum += slots[base + static_cast<size_t>(s) * K];
    }
    out[static_cast<size_t>(t) * K + c] = sum;
}

// ============================================================================
// PREFILL KERNELS — sm_60 (Tesla P100):
// BT=16, 1 row/warp (8 rows/CTA), pure branchless __half2 HFMA2 inner loop!
// ============================================================================
template <int BT>
__global__ __launch_bounds__(256, 4) void q35_moe_gu_iq3xxs_sm60_h2_kernel(
    const BlockIQ3_XXS_P * __restrict__ gate_exps,
    const BlockIQ3_XXS_P * __restrict__ up_exps,
    const int4 *           __restrict__ X_h2,   // [T, K/8] int4
    const float *          __restrict__ X_rsc,  // [T]
    const Q35TileDesc *    __restrict__ tiles,
    const int32_t *        __restrict__ n_tiles_ptr,
    const int32_t *        __restrict__ sorted_pairs,
    float *                __restrict__ act,    // [T, k_used, ff]
    int ff, int K, int k_used, float clamp_limit) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Q35TileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    __shared__ uint4    s_x4[BT * 32]; // 4 KB (BT=8) / 8 KB (BT=16)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    __shared__ uint2    s_grid_h2[256]; // 2 KB direct FP16x2 LUT
    __shared__ uint2    s_sign_mask[16]; // 128 B
#else
    __shared__ uint32_t s_grid32[256]; // fallback for sm_52 compile
#endif
    __shared__ uint8_t  s_ksigns[128]; // 128 B
    __shared__ int32_t  s_pairs[BT];
    __shared__ float    s_rsc[BT];

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int lane = threadIdx.x;
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;

    {
        const uint32_t gv = __ldg(&iq3xxs_grid_d[tid]);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
        const __half2 h_lo = __floats2half2_rn(static_cast<float>(gv & 0xFFu), static_cast<float>((gv >> 8) & 0xFFu));
        const __half2 h_hi = __floats2half2_rn(static_cast<float>((gv >> 16) & 0xFFu), static_cast<float>((gv >> 24) & 0xFFu));
        s_grid_h2[tid] = make_uint2(*reinterpret_cast<const uint32_t *>(&h_lo), *reinterpret_cast<const uint32_t *>(&h_hi));
#else
        s_grid32[tid] = gv;
#endif
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    if (tid < 16) {
        s_sign_mask[tid] = make_uint2(
            ((tid & 1) ? 0x00008000u : 0u) | ((tid & 2) ? 0x80000000u : 0u),
            ((tid & 4) ? 0x00008000u : 0u) | ((tid & 8) ? 0x80000000u : 0u));
    }
#endif
    if (tid < 128) s_ksigns[tid] = __ldg(&ksigns_iq2xs_d[tid]);
    if (tid < BT) {
        if (tid < n_tile) {
            const int pair = sorted_pairs[start + tid];
            s_pairs[tid] = pair;
            s_rsc[tid]   = __ldg(&X_rsc[pair >> 4]);
        } else {
            s_pairs[tid] = 0;
            s_rsc[tid]   = 0.0f;
        }
    }
    __syncthreads();

    const int nb   = K >> 8;
    const int K_i4 = K >> 3;
    const BlockIQ3_XXS_P * g_row = (row < ff) ? (gate_exps + (static_cast<size_t>(eid) * ff + row) * nb) : nullptr;
    const BlockIQ3_XXS_P * u_row = (row < ff) ? (up_exps   + (static_cast<size_t>(eid) * ff + row) * nb) : nullptr;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    const __half2 hzero = __float2half2_rn(0.0f);
    __half2 ag[BT], au[BT];
    #pragma unroll
    for (int j = 0; j < BT; ++j) { ag[j] = hzero; au[j] = hzero; }
#else
    float ag[BT], au[BT];
    #pragma unroll
    for (int j = 0; j < BT; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }
#endif

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_v = BT * 32;
        for (int idx = tid; idx < tot_v; idx += 256) {
            const int j = idx >> 5;
            const int l = idx & 31;
            if (j < n_tile) {
                const int tok = s_pairs[j] >> 4;
                const int4 v  = __ldg(&X_h2[static_cast<size_t>(tok) * K_i4 + bi * 32 + l]);
                s_x4[j * 32 + l] = make_uint4(static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y),
                                              static_cast<uint32_t>(v.z), static_cast<uint32_t>(v.w));
            } else {
                s_x4[j * 32 + l] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        if (g_row != nullptr) {
            auto run_mat = [&](const BlockIQ3_XXS_P * wr, auto & acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
                __half2 h0, h1, h2, h3;
                float sc;
                q35_decode_iq3xxs_lane_h2(wr[bi], s_grid_h2, s_sign_mask, s_ksigns, lane, h0, h1, h2, h3, sc);
                const __half2 sc2 = __float2half2_rn(sc * 16.0f);
                h0 = __hmul2(h0, sc2); h1 = __hmul2(h1, sc2);
                h2 = __hmul2(h2, sc2); h3 = __hmul2(h3, sc2);
                #pragma unroll
                for (int j = 0; j < BT; ++j) {
                    const uint4 xv = s_x4[j * 32 + lane];
                    acc[j] = __hfma2(h0, *reinterpret_cast<const __half2 *>(&xv.x), acc[j]);
                    acc[j] = __hfma2(h1, *reinterpret_cast<const __half2 *>(&xv.y), acc[j]);
                    acc[j] = __hfma2(h2, *reinterpret_cast<const __half2 *>(&xv.z), acc[j]);
                    acc[j] = __hfma2(h3, *reinterpret_cast<const __half2 *>(&xv.w), acc[j]);
                }
#else
                int w0, w1; float sc;
                q35_decode_iq3xxs_lane(wr[bi], s_grid32, s_ksigns, lane, w0, w1, sc);
                auto dot4 = [](int wi, uint32_t a, uint32_t b) {
                    const float2 fa = __half22float2(*reinterpret_cast<const __half2 *>(&a));
                    const float2 fb = __half22float2(*reinterpret_cast<const __half2 *>(&b));
                    return static_cast<float>(static_cast<int8_t>((wi >>  0) & 0xFF)) * fa.x
                         + static_cast<float>(static_cast<int8_t>((wi >>  8) & 0xFF)) * fa.y
                         + static_cast<float>(static_cast<int8_t>((wi >> 16) & 0xFF)) * fb.x
                         + static_cast<float>(static_cast<int8_t>((wi >> 24) & 0xFF)) * fb.y;
                };
                #pragma unroll
                for (int j = 0; j < BT; ++j) {
                    const uint4 xv = s_x4[j * 32 + lane];
                    acc[j] += (sc * 16.0f) * (dot4(w0, xv.x, xv.y) + dot4(w1, xv.z, xv.w));
                }
#endif
            };
            run_mat(g_row, ag);
            run_mat(u_row, au);
        }
        __syncthreads();
    }

    if (row < ff) {
        #pragma unroll
        for (int j = 0; j < BT; ++j) {
            if (j < n_tile) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
                const float2 fg = __half22float2(ag[j]);
                const float2 fu = __half22float2(au[j]);
                float g = q35_warp_sum(fg.x + fg.y);
                float u = q35_warp_sum(fu.x + fu.y);
#else
                float g = q35_warp_sum(ag[j]);
                float u = q35_warp_sum(au[j]);
#endif
                if (lane == 0) {
                    const float rsc = s_rsc[j];
                    g *= rsc; u *= rsc;
                    if (clamp_limit > 0.0f) {
                        g = fminf(fmaxf(g, -clamp_limit), clamp_limit);
                        u = fminf(fmaxf(u, -clamp_limit), clamp_limit);
                    }
                    const int pair = s_pairs[j];
                    act[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * ff + row] = (g / (1.0f + expf(-g))) * u;
                }
            }
        }
    }
}

template <int BT>
__global__ __launch_bounds__(256, 4) void q35_moe_dn_iq4xs_sm60_h2_kernel(
    const BlockIQ4_XS_P * __restrict__ dn_exps,
    const int4 *          __restrict__ A_h2,   // [T * k_used, ff/8] int4
    const float *         __restrict__ A_rsc,  // [T * k_used]
    const float *         __restrict__ top_w,
    const Q35TileDesc *   __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Q35TileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    __shared__ uint2    s_a2[BT * 64]; // 4 KB (BT=8) / 8 KB (BT=16)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    __shared__ uint16_t s_iq4_h[16];   // direct FP16 LUT
#else
    __shared__ int8_t   s_iq4[16];     // fallback for sm_52
#endif
    __shared__ int32_t  s_pairs[BT];
    __shared__ float    s_scale[BT];

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int lane = threadIdx.x;
    const int ib   = lane >> 2;
    const int sub4 = lane & 3;
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    const int row1 = row0 + 1;

    if (tid < 16) {
        const int8_t iv = __ldg(&kvalues_iq4nl_d[tid]);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
        const __half hv = __float2half_rn(static_cast<float>(iv));
        s_iq4_h[tid] = *reinterpret_cast<const uint16_t *>(&hv);
#else
        s_iq4[tid] = iv;
#endif
    }
    if (tid < BT) {
        if (tid < n_tile) {
            const int pair = sorted_pairs[start + tid];
            const size_t sidx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
            s_pairs[tid] = pair;
            s_scale[tid] = top_w[sidx] * __ldg(&A_rsc[sidx]);
        } else {
            s_pairs[tid] = 0;
            s_scale[tid] = 0.0f;
        }
    }
    __syncthreads();

    const int nb    = ff >> 8;
    const int ff_i4 = ff >> 3;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    const __half2 hzero = __float2half2_rn(0.0f);
    __half2 acc0[BT], acc1[BT];
    #pragma unroll
    for (int j = 0; j < BT; ++j) { acc0[j] = hzero; acc1[j] = hzero; }
#else
    float acc0[BT], acc1[BT];
    #pragma unroll
    for (int j = 0; j < BT; ++j) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
#endif

    const int a_off0 = ib * 8 + sub4;
    const int a_off1 = ib * 8 + 4 + sub4;

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_v = BT * 32;
        for (int idx = tid; idx < tot_v; idx += 256) {
            const int j = idx >> 5;
            const int l = idx & 31;
            if (j < n_tile) {
                const int pair = s_pairs[j];
                const size_t sidx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                const int4 v = __ldg(&A_h2[sidx * ff_i4 + bi * 32 + l]);
                s_a2[j * 64 + l * 2 + 0] = make_uint2(static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y));
                s_a2[j * 64 + l * 2 + 1] = make_uint2(static_cast<uint32_t>(v.z), static_cast<uint32_t>(v.w));
            } else {
                s_a2[j * 64 + l * 2 + 0] = make_uint2(0u, 0u);
                s_a2[j * 64 + l * 2 + 1] = make_uint2(0u, 0u);
            }
        }
        __syncthreads();

        auto accum_row = [&](int row, auto & acc) {
            if (row >= K) return;
            const BlockIQ4_XS_P & blk = dn_exps[(static_cast<size_t>(eid) * K + row) * nb + bi];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
            __half2 h0, h1, h2, h3;
            float sc;
            q35_decode_iq4xs_lane_h2(blk, s_iq4_h, lane, h0, h1, h2, h3, sc);
            const __half2 sc2 = __float2half2_rn(sc * 16.0f);
            h0 = __hmul2(h0, sc2); h1 = __hmul2(h1, sc2);
            h2 = __hmul2(h2, sc2); h3 = __hmul2(h3, sc2);
            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                const uint2 xa = s_a2[j * 64 + a_off0];
                const uint2 xb = s_a2[j * 64 + a_off1];
                acc[j] = __hfma2(h0, *reinterpret_cast<const __half2 *>(&xa.x), acc[j]);
                acc[j] = __hfma2(h1, *reinterpret_cast<const __half2 *>(&xa.y), acc[j]);
                acc[j] = __hfma2(h2, *reinterpret_cast<const __half2 *>(&xb.x), acc[j]);
                acc[j] = __hfma2(h3, *reinterpret_cast<const __half2 *>(&xb.y), acc[j]);
            }
#else
            int w0, w1; float sc;
            q35_decode_iq4xs_lane(blk, s_iq4, lane, w0, w1, sc);
            auto dot4 = [](int wi, uint32_t a, uint32_t b) {
                const float2 fa = __half22float2(*reinterpret_cast<const __half2 *>(&a));
                const float2 fb = __half22float2(*reinterpret_cast<const __half2 *>(&b));
                return static_cast<float>(static_cast<int8_t>((wi >>  0) & 0xFF)) * fa.x
                     + static_cast<float>(static_cast<int8_t>((wi >>  8) & 0xFF)) * fa.y
                     + static_cast<float>(static_cast<int8_t>((wi >> 16) & 0xFF)) * fb.x
                     + static_cast<float>(static_cast<int8_t>((wi >> 24) & 0xFF)) * fb.y;
            };
            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                const uint2 xa = s_a2[j * 64 + a_off0];
                const uint2 xb = s_a2[j * 64 + a_off1];
                acc[j] += (sc * 16.0f) * (dot4(w0, xa.x, xa.y) + dot4(w1, xb.x, xb.y));
            }
#endif
        };

        accum_row(row0, acc0);
        accum_row(row1, acc1);
        __syncthreads();
    }

    auto emit_row = [&](int row, auto & acc) {
        if (row >= K) return;
        #pragma unroll
        for (int j = 0; j < BT; ++j) {
            if (j < n_tile) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
                const float2 fv = __half22float2(acc[j]);
                const float v = q35_warp_sum(fv.x + fv.y);
#else
                const float v = q35_warp_sum(acc[j]);
#endif
                if (lane == 0) {
                    const int pair = s_pairs[j];
                    down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row] = s_scale[j] * v;
                }
            }
        }
    };
    emit_row(row0, acc0);
    emit_row(row1, acc1);
}

// Native sm_60 FP16x2 Q6_K SoA Down Prefill (eliminates slow software-dp4a on Q6_K down layers!)
template <int BT>
__global__ __launch_bounds__(256, 4) void q35_moe_dn_q6k_soa_sm60_h2_kernel(
    const uint2 *       __restrict__ dn_ql,
    const uint2 *       __restrict__ dn_qh,
    const int8_t *      __restrict__ dn_sc,
    const uint16_t *    __restrict__ dn_d,
    const int4 *        __restrict__ A_h2,
    const float *       __restrict__ A_rsc,
    const float *       __restrict__ top_w,
    const Q35TileDesc * __restrict__ tiles,
    const int32_t *     __restrict__ n_tiles_ptr,
    const int32_t *     __restrict__ sorted_pairs,
    float *             __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Q35TileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    __shared__ uint4   s_a4[BT * 32]; // 4 KB (BT=8)
    __shared__ int32_t s_pairs[BT];
    __shared__ float   s_scale[BT];

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int lane = threadIdx.x;
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    const int row1 = row0 + 1;

    if (tid < BT) {
        if (tid < n_tile) {
            const int pair = sorted_pairs[start + tid];
            const size_t sidx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
            s_pairs[tid] = pair;
            s_scale[tid] = top_w[sidx] * __ldg(&A_rsc[sidx]);
        } else {
            s_pairs[tid] = 0;
            s_scale[tid] = 0.0f;
        }
    }
    __syncthreads();

    const int nb    = ff >> 8;
    const int ff_i4 = ff >> 3;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    const __half2 hzero = __float2half2_rn(0.0f);
    __half2 acc0[BT], acc1[BT];
    #pragma unroll
    for (int j = 0; j < BT; ++j) { acc0[j] = hzero; acc1[j] = hzero; }
#else
    float acc0[BT], acc1[BT];
    #pragma unroll
    for (int j = 0; j < BT; ++j) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
#endif

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_v = BT * 32;
        for (int idx = tid; idx < tot_v; idx += 256) {
            const int j = idx >> 5;
            const int l = idx & 31;
            if (j < n_tile) {
                const int pair = s_pairs[j];
                const size_t sidx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                const int4 v = __ldg(&A_h2[sidx * ff_i4 + bi * 32 + l]);
                s_a4[j * 32 + l] = make_uint4(static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y),
                                              static_cast<uint32_t>(v.z), static_cast<uint32_t>(v.w));
            } else {
                s_a4[j * 32 + l] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        auto accum_row = [&](int row, auto & acc) {
            if (row >= K) return;
            const size_t blk_i = (static_cast<size_t>(eid) * K + row) * nb + bi;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
            __half2 h0, h1, h2, h3;
            float sc;
            q35_decode_q6k_soa_lane_h2(
                dn_ql + blk_i * 16, dn_qh + blk_i * 8, dn_sc + blk_i * 16, __ldg(&dn_d[blk_i]),
                lane, h0, h1, h2, h3, sc);
            const __half2 sc2 = __float2half2_rn(sc * 16.0f);
            h0 = __hmul2(h0, sc2); h1 = __hmul2(h1, sc2);
            h2 = __hmul2(h2, sc2); h3 = __hmul2(h3, sc2);
            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                const uint4 av = s_a4[j * 32 + lane];
                acc[j] = __hfma2(h0, *reinterpret_cast<const __half2 *>(&av.x), acc[j]);
                acc[j] = __hfma2(h1, *reinterpret_cast<const __half2 *>(&av.y), acc[j]);
                acc[j] = __hfma2(h2, *reinterpret_cast<const __half2 *>(&av.z), acc[j]);
                acc[j] = __hfma2(h3, *reinterpret_cast<const __half2 *>(&av.w), acc[j]);
            }
#else
            (void)blk_i; (void)acc;
#endif
        };

        accum_row(row0, acc0);
        accum_row(row1, acc1);
        __syncthreads();
    }

    auto emit_row = [&](int row, auto & acc) {
        if (row >= K) return;
        #pragma unroll
        for (int j = 0; j < BT; ++j) {
            if (j < n_tile) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
                const float2 fv = __half22float2(acc[j]);
                const float v = q35_warp_sum(fv.x + fv.y);
#else
                const float v = q35_warp_sum(acc[j]);
#endif
                if (lane == 0) {
                    const int pair = s_pairs[j];
                    down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row] = s_scale[j] * v;
                }
            }
        }
    };
    emit_row(row0, acc0);
    emit_row(row1, acc1);
}

// ============================================================================
// PREFILL KERNELS — sm_61 (GTX 1060):
// BT=16, 1 row/warp (8 rows/CTA), branchless native hardware __dp4a!
// ============================================================================
template <int BT>
__global__ __launch_bounds__(256, 4) void q35_moe_gu_iq3xxs_sm61_dp4a_kernel(
    const BlockIQ3_XXS_P * __restrict__ gate_exps,
    const BlockIQ3_XXS_P * __restrict__ up_exps,
    const BlockQ8_1_Act *  __restrict__ x_q8,
    const Q35TileDesc *    __restrict__ tiles,
    const int32_t *        __restrict__ n_tiles_ptr,
    const int32_t *        __restrict__ sorted_pairs,
    float *                __restrict__ act,
    int ff, int K, int k_used, float clamp_limit) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Q35TileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    __shared__ uint2    s_xq[BT * 32]; // 4 KB
    __shared__ float    s_xd[BT * 8];  // 512 B
    __shared__ uint32_t s_grid32[256];
    __shared__ uint8_t  s_ksigns[128];
    __shared__ int32_t  s_pairs[BT];

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int lane = threadIdx.x;
    const int ib32 = lane >> 2;
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;

    s_grid32[tid] = __ldg(&iq3xxs_grid_d[tid]);
    if (tid < 128) s_ksigns[tid] = __ldg(&ksigns_iq2xs_d[tid]);
    if (tid < n_tile) s_pairs[tid] = sorted_pairs[start + tid];
    __syncthreads();

    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const BlockIQ3_XXS_P * g_row = (row < ff) ? (gate_exps + (static_cast<size_t>(eid) * ff + row) * nb) : nullptr;
    const BlockIQ3_XXS_P * u_row = (row < ff) ? (up_exps   + (static_cast<size_t>(eid) * ff + row) * nb) : nullptr;

    float ag[BT], au[BT];
    #pragma unroll
    for (int j = 0; j < BT; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

    for (int bi = 0; bi < nb; ++bi) {
        const int b_idx = tid >> 1;
        const int half  = tid & 1;
        const int j     = b_idx >> 3;
        const int b8    = b_idx & 7;
        if (j < BT) {
            if (j < n_tile) {
                const int tok = s_pairs[j] >> 4;
                const BlockQ8_1_Act & xb = x_q8[static_cast<size_t>(tok) * K_blocks + bi * 8 + b8];
                const int4 q4 = __ldg(&reinterpret_cast<const int4 *>(xb.qs)[half]);
                s_xq[j * 32 + b8 * 4 + half * 2 + 0] = make_uint2(static_cast<uint32_t>(q4.x), static_cast<uint32_t>(q4.y));
                s_xq[j * 32 + b8 * 4 + half * 2 + 1] = make_uint2(static_cast<uint32_t>(q4.z), static_cast<uint32_t>(q4.w));
                if (half == 0) s_xd[j * 8 + b8] = xb.d;
            } else {
                s_xq[j * 32 + b8 * 4 + half * 2 + 0] = make_uint2(0u, 0u);
                s_xq[j * 32 + b8 * 4 + half * 2 + 1] = make_uint2(0u, 0u);
                if (half == 0) s_xd[j * 8 + b8] = 0.0f;
            }
        }
        __syncthreads();

        if (g_row != nullptr) {
            int gw0, gw1, uw0, uw1;
            float gsc, usc;
            q35_decode_iq3xxs_lane(g_row[bi], s_grid32, s_ksigns, lane, gw0, gw1, gsc);
            q35_decode_iq3xxs_lane(u_row[bi], s_grid32, s_ksigns, lane, uw0, uw1, usc);
            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                const uint2 xv = s_xq[j * 32 + lane];
                const float xd = s_xd[j * 8 + ib32];
                ag[j] += (gsc * xd) * static_cast<float>(dp4a_s32(gw0, static_cast<int>(xv.x), dp4a_s32(gw1, static_cast<int>(xv.y), 0)));
                au[j] += (usc * xd) * static_cast<float>(dp4a_s32(uw0, static_cast<int>(xv.x), dp4a_s32(uw1, static_cast<int>(xv.y), 0)));
            }
        }
        __syncthreads();
    }

    if (row < ff) {
        #pragma unroll
        for (int j = 0; j < BT; ++j) {
            if (j < n_tile) {
                float g = q35_warp_sum(ag[j]);
                float u = q35_warp_sum(au[j]);
                if (lane == 0) {
                    if (clamp_limit > 0.0f) {
                        g = fminf(fmaxf(g, -clamp_limit), clamp_limit);
                        u = fminf(fmaxf(u, -clamp_limit), clamp_limit);
                    }
                    const int pair = s_pairs[j];
                    act[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * ff + row] = (g / (1.0f + expf(-g))) * u;
                }
            }
        }
    }
}

template <int BT>
__global__ __launch_bounds__(256, 4) void q35_moe_dn_iq4xs_sm61_dp4a_kernel(
    const BlockIQ4_XS_P * __restrict__ dn_exps,
    const BlockQ8_1_Act * __restrict__ act_q8,
    const float *         __restrict__ top_w,
    const Q35TileDesc *   __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Q35TileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    __shared__ int32_t s_aq[BT * 64]; // 4 KB
    __shared__ float   s_ad[BT * 8];
    __shared__ int8_t  s_iq4[16];
    __shared__ int32_t s_pairs[BT];
    __shared__ float   s_tw[BT];

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int lane = threadIdx.x;
    const int ib   = lane >> 2;
    const int sub4 = lane & 3;
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;

    if (tid < 16) s_iq4[tid] = __ldg(&kvalues_iq4nl_d[tid]);
    if (tid < n_tile) {
        const int pair = sorted_pairs[start + tid];
        s_pairs[tid] = pair;
        s_tw[tid] = top_w[static_cast<size_t>(pair >> 4) * k_used + (pair & 15)];
    }
    __syncthreads();

    const int ff_blks = ff >> 5;
    const int nb      = ff >> 8;

    float acc[BT];
    #pragma unroll
    for (int j = 0; j < BT; ++j) acc[j] = 0.0f;

    const int a_off0 = ib * 8 + sub4;
    const int a_off1 = ib * 8 + 4 + sub4;

    for (int bi = 0; bi < nb; ++bi) {
        const int b_idx = tid >> 1;
        const int half  = tid & 1;
        const int j     = b_idx >> 3;
        const int b8    = b_idx & 7;
        if (j < BT) {
            int32_t * dst = s_aq + j * 64 + b8 * 8 + half * 4;
            if (j < n_tile) {
                const int pair = s_pairs[j];
                const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                const BlockQ8_1_Act & ab = act_q8[slot_idx * ff_blks + bi * 8 + b8];
                const int4 q4 = __ldg(&reinterpret_cast<const int4 *>(ab.qs)[half]);
                dst[0] = q4.x; dst[1] = q4.y; dst[2] = q4.z; dst[3] = q4.w;
                if (half == 0) s_ad[j * 8 + b8] = ab.d;
            } else {
                dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0;
                if (half == 0) s_ad[j * 8 + b8] = 0.0f;
            }
        }
        __syncthreads();

        if (row < K) {
            int w0, w1; float sc;
            q35_decode_iq4xs_lane(dn_exps[(static_cast<size_t>(eid) * K + row) * nb + bi], s_iq4, lane, w0, w1, sc);
            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                const int a0 = s_aq[j * 64 + a_off0];
                const int a1 = s_aq[j * 64 + a_off1];
                acc[j] += (sc * s_ad[j * 8 + ib]) * static_cast<float>(dp4a_s32(w0, a0, dp4a_s32(w1, a1, 0)));
            }
        }
        __syncthreads();
    }

    if (row < K) {
        #pragma unroll
        for (int j = 0; j < BT; ++j) {
            if (j < n_tile) {
                const float v = q35_warp_sum(acc[j]);
                if (lane == 0) {
                    const int pair = s_pairs[j];
                    down_slots[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * K + row] = s_tw[j] * v;
                }
            }
        }
    }
}

// ============================================================================
// DECODE KERNELS — sm_60 (Tesla P100, T < 16): Smem-Cached FP16x2 HFMA2 + Split-Slot!
// ============================================================================
// 16 warps (512 threads) x 2 rows/warp = 32 outputs = 1 exact A_h2/A_sc block!
// Caches X_h2 + X_sc in dynamic smem and writes A_h2/A_sc directly (eliminating act DRAM + prep launch!).
__global__ void q35_moe_gu_iq3xxs_decode_sm60_h2_fused_prep_kernel(
    const BlockIQ3_XXS_P * __restrict__ gate_exps,
    const BlockIQ3_XXS_P * __restrict__ up_exps,
    const int4 *           __restrict__ X_h2,
    const float *          __restrict__ X_sc,
    const int32_t *        __restrict__ top_ids,
    __half2 *              __restrict__ A_h2,
    float *                __restrict__ A_sc,
    int ff, int K, int k_used, float clamp_limit) {

    extern __shared__ uint8_t s_raw[];
    const int K_blocks = K >> 5;
    int4 *  s_xh2 = reinterpret_cast<int4 *>(s_raw);
    float * s_xsc = reinterpret_cast<float *>(s_raw + static_cast<size_t>(K_blocks) * 64);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    __shared__ uint2    s_grid_h2[256];
    __shared__ uint2    s_sign_mask[16];
#else
    __shared__ uint32_t s_grid32[256];
#endif
    __shared__ uint8_t  s_ksigns[128];
    __shared__ float    s_act[32];

    const int warp = threadIdx.y; // 0..15
    const int lane = threadIdx.x; // 0..31
    const int tid  = warp * 32 + lane;
    const int slot = blockIdx.y;
    const int t    = blockIdx.z;

    if (tid < 256) {
        const uint32_t gv = __ldg(&iq3xxs_grid_d[tid]);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
        const __half2 h_lo = __floats2half2_rn(static_cast<float>(gv & 0xFFu), static_cast<float>((gv >> 8) & 0xFFu));
        const __half2 h_hi = __floats2half2_rn(static_cast<float>((gv >> 16) & 0xFFu), static_cast<float>((gv >> 24) & 0xFFu));
        s_grid_h2[tid] = make_uint2(*reinterpret_cast<const uint32_t *>(&h_lo), *reinterpret_cast<const uint32_t *>(&h_hi));
#else
        s_grid32[tid] = gv;
#endif
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    if (tid < 16) {
        s_sign_mask[tid] = make_uint2(
            ((tid & 1) ? 0x00008000u : 0u) | ((tid & 2) ? 0x80000000u : 0u),
            ((tid & 4) ? 0x00008000u : 0u) | ((tid & 8) ? 0x80000000u : 0u));
    }
#endif
    if (tid < 128) s_ksigns[tid] = __ldg(&ksigns_iq2xs_d[tid]);
    const int4 *  xh_g = X_h2 + static_cast<size_t>(t) * (K_blocks * 4);
    const float * xs_g = X_sc + static_cast<size_t>(t) * K_blocks;
    const int tot4 = K_blocks * 4;
    for (int i = tid; i < tot4; i += 512) s_xh2[i] = __ldg(&xh_g[i]);
    for (int i = tid; i < K_blocks; i += 512) s_xsc[i] = __ldg(&xs_g[i]);
    __syncthreads();

    const int blk_ff = blockIdx.x;
    const int row0   = blk_ff * 32 + warp * 2;
    const int ib32   = lane >> 2;
    const int nb     = K >> 8;
    const int eid    = top_ids[static_cast<size_t>(t) * k_used + slot];

    float ag[2] = {0.0f, 0.0f}, au[2] = {0.0f, 0.0f};
    for (int bi = 0; bi < nb; ++bi) {
        const int4  xv  = s_xh2[bi * 32 + lane];
        const float xsc = s_xsc[bi * 8 + ib32];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row0 + r;
            const BlockIQ3_XXS_P * g_row = gate_exps + (static_cast<size_t>(eid) * ff + row) * nb;
            const BlockIQ3_XXS_P * u_row = up_exps   + (static_cast<size_t>(eid) * ff + row) * nb;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
            const __half2 x0 = *reinterpret_cast<const __half2 *>(&xv.x);
            const __half2 x1 = *reinterpret_cast<const __half2 *>(&xv.y);
            const __half2 x2 = *reinterpret_cast<const __half2 *>(&xv.z);
            const __half2 x3 = *reinterpret_cast<const __half2 *>(&xv.w);
            __half2 h0, h1, h2, h3;
            float gsc, usc;
            q35_decode_iq3xxs_lane_h2(g_row[bi], s_grid_h2, s_sign_mask, s_ksigns, lane, h0, h1, h2, h3, gsc);
            __half2 sg = __hfma2(h3, x3, __hfma2(h2, x2, __hfma2(h1, x1, __hmul2(h0, x0))));
            const float2 fg = __half22float2(sg);
            ag[r] += (gsc * xsc) * (fg.x + fg.y);

            q35_decode_iq3xxs_lane_h2(u_row[bi], s_grid_h2, s_sign_mask, s_ksigns, lane, h0, h1, h2, h3, usc);
            __half2 su = __hfma2(h3, x3, __hfma2(h2, x2, __hfma2(h1, x1, __hmul2(h0, x0))));
            const float2 fu = __half22float2(su);
            au[r] += (usc * xsc) * (fu.x + fu.y);
#else
            int gw0, gw1, uw0, uw1;
            float gsc, usc;
            q35_decode_iq3xxs_lane(g_row[bi], s_grid32, s_ksigns, lane, gw0, gw1, gsc);
            q35_decode_iq3xxs_lane(u_row[bi], s_grid32, s_ksigns, lane, uw0, uw1, usc);
            auto dot4 = [](int wi, int a, int b) {
                const float2 fa = __half22float2(*reinterpret_cast<const __half2 *>(&a));
                const float2 fb = __half22float2(*reinterpret_cast<const __half2 *>(&b));
                return static_cast<float>(static_cast<int8_t>((wi >>  0) & 0xFF)) * fa.x
                     + static_cast<float>(static_cast<int8_t>((wi >>  8) & 0xFF)) * fa.y
                     + static_cast<float>(static_cast<int8_t>((wi >> 16) & 0xFF)) * fb.x
                     + static_cast<float>(static_cast<int8_t>((wi >> 24) & 0xFF)) * fb.y;
            };
            ag[r] += (gsc * xsc) * (dot4(gw0, xv.x, xv.y) + dot4(gw1, xv.z, xv.w));
            au[r] += (usc * xsc) * (dot4(uw0, xv.x, xv.y) + dot4(uw1, xv.z, xv.w));
#endif
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        float g = q35_warp_sum(ag[r]);
        float u = q35_warp_sum(au[r]);
        if (lane == 0) {
            if (clamp_limit > 0.0f) {
                g = fminf(fmaxf(g, -clamp_limit), clamp_limit);
                u = fminf(fmaxf(u, -clamp_limit), clamp_limit);
            }
            s_act[warp * 2 + r] = (g / (1.0f + expf(-g))) * u;
        }
    }
    __syncthreads();

    if (warp == 0) {
        const int ff_blks = ff >> 5;
        const size_t b_out = (static_cast<size_t>(t) * k_used + slot) * ff_blks + blk_ff;
        const float v = s_act[lane];
        float am = fabsf(v);
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_down_sync(0xffffffffu, am, off));
        am = __shfl_sync(0xffffffffu, am, 0);
        const float inv = (am > 0.0f) ? (8.0f / am) : 0.0f;
        const float vn  = v * inv;
        const float vp  = __shfl_xor_sync(0xffffffffu, vn, 1);
        if ((lane & 1) == 0) {
            A_h2[b_out * 16 + (lane >> 1)] = __float22half2_rn(make_float2(vn, vp));
        }
        if (lane == 0) A_sc[b_out] = am * 0.125f;
    }
}

// Split-Slot Smem-Cached IQ4_XS Down Decode (2 rows/warp x 8 warps, grid(K/16, k_used, T) -> 8x parallelism!)
__global__ void q35_moe_dn_iq4xs_split_decode_sm60_h2_kernel(
    const BlockIQ4_XS_P * __restrict__ dn_exps,
    const uint2 *         __restrict__ A_h2,
    const float *         __restrict__ A_sc,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ down_slots,
    int K, int ff, int k_used) {

    extern __shared__ uint8_t s_raw[];
    const int ff_blks = ff >> 5;
    uint2 * s_ah2 = reinterpret_cast<uint2 *>(s_raw);
    float * s_asc = reinterpret_cast<float *>(s_raw + static_cast<size_t>(ff_blks) * 64);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    __shared__ uint16_t s_iq4_h[16];
#else
    __shared__ int8_t   s_iq4[16];
#endif
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int slot = blockIdx.y;
    const int t    = blockIdx.z;
    const size_t sidx = static_cast<size_t>(t) * k_used + slot;

    if (tid < 16) {
        const int8_t iv = __ldg(&kvalues_iq4nl_d[tid]);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
        const __half hv = __float2half_rn(static_cast<float>(iv));
        s_iq4_h[tid] = *reinterpret_cast<const uint16_t *>(&hv);
#else
        s_iq4[tid] = iv;
#endif
    }
    const uint2 * ah_g = A_h2 + sidx * (ff_blks * 8);
    const float * as_g = A_sc + sidx * ff_blks;
    const int tot_u2 = ff_blks * 8;
    for (int i = tid; i < tot_u2; i += 256) s_ah2[i] = __ldg(&ah_g[i]);
    for (int i = tid; i < ff_blks; i += 256) s_asc[i] = __ldg(&as_g[i]);
    __syncthreads();

    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= K) return;
    const int lane = threadIdx.x;
    const int ib   = lane >> 2;
    const int sub4 = lane & 3;
    const int nb   = ff >> 8;
    const int eid  = top_ids[sidx];
    const float ew = top_w[sidx];

    float acc[2] = {0.0f, 0.0f};
    for (int bi = 0; bi < nb; ++bi) {
        const uint2 xa  = s_ah2[bi * 64 + ib * 8 + sub4];
        const uint2 xb  = s_ah2[bi * 64 + ib * 8 + 4 + sub4];
        const float asc = s_asc[bi * 8 + ib];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row0 + r;
            if (row < K) {
                const BlockIQ4_XS_P * wr = dn_exps + (static_cast<size_t>(eid) * K + row) * nb;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
                __half2 h0, h1, h2, h3;
                float sc;
                q35_decode_iq4xs_lane_h2(wr[bi], s_iq4_h, lane, h0, h1, h2, h3, sc);
                __half2 s = __hmul2(h0, *reinterpret_cast<const __half2 *>(&xa.x));
                s = __hfma2(h1, *reinterpret_cast<const __half2 *>(&xa.y), s);
                s = __hfma2(h2, *reinterpret_cast<const __half2 *>(&xb.x), s);
                s = __hfma2(h3, *reinterpret_cast<const __half2 *>(&xb.y), s);
                const float2 sf = __half22float2(s);
                acc[r] += (sc * asc) * (sf.x + sf.y);
#else
                int w0, w1; float sc;
                q35_decode_iq4xs_lane(wr[bi], s_iq4, lane, w0, w1, sc);
                auto dot4 = [](int wi, uint32_t a, uint32_t b) {
                    const float2 fa = __half22float2(*reinterpret_cast<const __half2 *>(&a));
                    const float2 fb = __half22float2(*reinterpret_cast<const __half2 *>(&b));
                    return static_cast<float>(static_cast<int8_t>((wi >>  0) & 0xFF)) * fa.x
                         + static_cast<float>(static_cast<int8_t>((wi >>  8) & 0xFF)) * fa.y
                         + static_cast<float>(static_cast<int8_t>((wi >> 16) & 0xFF)) * fb.x
                         + static_cast<float>(static_cast<int8_t>((wi >> 24) & 0xFF)) * fb.y;
                };
                acc[r] += (sc * asc) * (dot4(w0, xa.x, xa.y) + dot4(w1, xb.x, xb.y));
#endif
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float res = q35_warp_sum(acc[r]);
        if (lane == 0 && row0 + r < K) {
            down_slots[sidx * K + row0 + r] = ew * res;
        }
    }
}

// Split-Slot Smem-Cached Q6_K SoA Down Decode on sm_60 (eliminates software-dp4a on blk.34/38/39!)
__global__ void q35_moe_dn_q6k_soa_split_decode_sm60_h2_kernel(
    const uint2 *    __restrict__ dn_ql,
    const uint2 *    __restrict__ dn_qh,
    const int8_t *   __restrict__ dn_sc,
    const uint16_t * __restrict__ dn_d,
    const int4 *     __restrict__ A_h2,
    const float *    __restrict__ A_sc,
    const int32_t *  __restrict__ top_ids,
    const float *    __restrict__ top_w,
    float *          __restrict__ down_slots,
    int K, int ff, int k_used) {

    extern __shared__ uint8_t s_raw[];
    const int ff_blks = ff >> 5;
    int4 *  s_ah2 = reinterpret_cast<int4 *>(s_raw);
    float * s_asc = reinterpret_cast<float *>(s_raw + static_cast<size_t>(ff_blks) * 64);

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int slot = blockIdx.y;
    const int t    = blockIdx.z;
    const size_t sidx = static_cast<size_t>(t) * k_used + slot;

    const int4 *  ah_g = A_h2 + sidx * (ff_blks * 4);
    const float * as_g = A_sc + sidx * ff_blks;
    const int tot_i4 = ff_blks * 4;
    for (int i = tid; i < tot_i4; i += 256) s_ah2[i] = __ldg(&ah_g[i]);
    for (int i = tid; i < ff_blks; i += 256) s_asc[i] = __ldg(&as_g[i]);
    __syncthreads();

    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= K) return;
    const int lane = threadIdx.x;
    const int ib32 = lane >> 2;
    const int nb   = ff >> 8;
    const int eid  = top_ids[sidx];
    const float ew = top_w[sidx];

    float acc[2] = {0.0f, 0.0f};
    for (int bi = 0; bi < nb; ++bi) {
        const int4  av  = s_ah2[bi * 32 + lane];
        const float asc = s_asc[bi * 8 + ib32];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row0 + r;
            if (row < K) {
                const size_t blk_i = (static_cast<size_t>(eid) * K + row) * nb + bi;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
                __half2 h0, h1, h2, h3;
                float sc;
                q35_decode_q6k_soa_lane_h2(
                    dn_ql + blk_i * 16, dn_qh + blk_i * 8, dn_sc + blk_i * 16, __ldg(&dn_d[blk_i]),
                    lane, h0, h1, h2, h3, sc);
                __half2 s = __hmul2(h0, *reinterpret_cast<const __half2 *>(&av.x));
                s = __hfma2(h1, *reinterpret_cast<const __half2 *>(&av.y), s);
                s = __hfma2(h2, *reinterpret_cast<const __half2 *>(&av.z), s);
                s = __hfma2(h3, *reinterpret_cast<const __half2 *>(&av.w), s);
                const float2 sf = __half22float2(s);
                acc[r] += (sc * asc) * (sf.x + sf.y);
#else
                (void)blk_i; (void)av; (void)asc;
#endif
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float res = q35_warp_sum(acc[r]);
        if (lane == 0 && row0 + r < K) {
            down_slots[sidx * K + row0 + r] = ew * res;
        }
    }
}

// ============================================================================
// DECODE KERNELS — sm_61 (GTX 1060, T < 16): Split-Slot Hardware __dp4a!
// ============================================================================
__global__ void q35_moe_gu_iq3xxs_decode_sm61_dp4a_kernel(
    const BlockIQ3_XXS_P * __restrict__ gate_exps,
    const BlockIQ3_XXS_P * __restrict__ up_exps,
    const BlockQ8_1_Act *  __restrict__ x_q8,
    const int32_t *        __restrict__ top_ids,
    float *                __restrict__ act,
    int ff, int K, int k_used, float clamp_limit) {

    extern __shared__ BlockQ8_1_Act s_x[];
    __shared__ uint32_t s_grid32[256];
    __shared__ uint8_t  s_ksigns[128];

    const int K_blocks = K >> 5;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int slot = blockIdx.y;
    const int t    = blockIdx.z;

    s_grid32[tid] = __ldg(&iq3xxs_grid_d[tid]);
    if (tid < 128) s_ksigns[tid] = __ldg(&ksigns_iq2xs_d[tid]);
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * src = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(t) * K_blocks);
    int4 *       dst = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += 256) dst[i] = __ldg(&src[i]);
    __syncthreads();

    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= ff) return;
    const int lane = threadIdx.x;
    const int ib32 = lane >> 2;
    const int sub  = lane & 3;
    const int nb   = K >> 8;
    const int eid  = top_ids[static_cast<size_t>(t) * k_used + slot];
    const BlockIQ3_XXS_P * g_row = gate_exps + (static_cast<size_t>(eid) * ff + row) * nb;
    const BlockIQ3_XXS_P * u_row = up_exps   + (static_cast<size_t>(eid) * ff + row) * nb;

    float ag = 0.0f, au = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        const BlockQ8_1_Act & xb = s_x[bi * 8 + ib32];
        const int * x32 = reinterpret_cast<const int *>(xb.qs);
        const int xv0 = x32[sub * 2 + 0];
        const int xv1 = x32[sub * 2 + 1];
        int gw0, gw1, uw0, uw1;
        float gsc, usc;
        q35_decode_iq3xxs_lane(g_row[bi], s_grid32, s_ksigns, lane, gw0, gw1, gsc);
        q35_decode_iq3xxs_lane(u_row[bi], s_grid32, s_ksigns, lane, uw0, uw1, usc);
        ag += (gsc * xb.d) * static_cast<float>(dp4a_s32(gw0, xv0, dp4a_s32(gw1, xv1, 0)));
        au += (usc * xb.d) * static_cast<float>(dp4a_s32(uw0, xv0, dp4a_s32(uw1, xv1, 0)));
    }
    float g = q35_warp_sum(ag);
    float u = q35_warp_sum(au);
    if (lane == 0) {
        if (clamp_limit > 0.0f) {
            g = fminf(fmaxf(g, -clamp_limit), clamp_limit);
            u = fminf(fmaxf(u, -clamp_limit), clamp_limit);
        }
        act[(static_cast<size_t>(t) * k_used + slot) * ff + row] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ void q35_moe_dn_iq4xs_split_decode_sm61_dp4a_kernel(
    const BlockIQ4_XS_P * __restrict__ dn_exps,
    const BlockQ8_1_Act * __restrict__ act_q8,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ down_slots,
    int K, int ff, int k_used) {

    extern __shared__ BlockQ8_1_Act s_a[];
    __shared__ int8_t s_iq4[16];

    const int ff_blks = ff >> 5;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int slot = blockIdx.y;
    const int t    = blockIdx.z;
    const size_t sidx = static_cast<size_t>(t) * k_used + slot;

    if (tid < 16) s_iq4[tid] = __ldg(&kvalues_iq4nl_d[tid]);
    const int n_i4 = (ff_blks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * src = reinterpret_cast<const int4 *>(act_q8 + sidx * ff_blks);
    int4 *       dst = reinterpret_cast<int4 *>(s_a);
    for (int i = tid; i < n_i4; i += 256) dst[i] = __ldg(&src[i]);
    __syncthreads();

    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= K) return;
    const int lane = threadIdx.x;
    const int ib   = lane >> 2;
    const int sub4 = lane & 3;
    const int nb   = ff >> 8;
    const int eid  = top_ids[sidx];
    const float ew = top_w[sidx];

    float acc[2] = {0.0f, 0.0f};
    for (int bi = 0; bi < nb; ++bi) {
        const BlockQ8_1_Act & ab = s_a[bi * 8 + ib];
        const int * a32 = reinterpret_cast<const int *>(ab.qs);
        const int a0 = a32[sub4];
        const int a1 = a32[4 + sub4];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row0 + r;
            if (row < K) {
                int w0, w1; float sc;
                q35_decode_iq4xs_lane(dn_exps[(static_cast<size_t>(eid) * K + row) * nb + bi], s_iq4, lane, w0, w1, sc);
                acc[r] += (sc * ab.d) * static_cast<float>(dp4a_s32(w0, a0, dp4a_s32(w1, a1, 0)));
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float res = q35_warp_sum(acc[r]);
        if (lane == 0 && row0 + r < K) {
            down_slots[sidx * K + row0 + r] = ew * res;
        }
    }
}

// ============================================================================
// FUSED Q8_0 SoA SHARED EXPERT DECODE — sm_60 (2 launches instead of 7!)
// ============================================================================
__global__ void q35_shexp_gu_q8_0_sm60_h2_fused_prep_kernel(
    const int4 *     __restrict__ g_q,
    const uint16_t * __restrict__ g_s,
    const int4 *     __restrict__ u_q,
    const uint16_t * __restrict__ u_s,
    const int4 *     __restrict__ X_h2,
    const float *    __restrict__ X_sc,
    __half2 *        __restrict__ A_h2,
    float *          __restrict__ A_sc,
    int ff_sh, int K) {

    extern __shared__ uint8_t s_raw[];
    const int K_blocks = K >> 5;
    int4 *  s_xh2 = reinterpret_cast<int4 *>(s_raw);
    float * s_xsc = reinterpret_cast<float *>(s_raw + static_cast<size_t>(K_blocks) * 64);
    __shared__ float s_act[32];

    const int warp = threadIdx.y; // 0..15
    const int lane = threadIdx.x; // 0..31
    const int tid  = warp * 32 + lane;
    const int t    = blockIdx.y;

    const int4 *  xh_g = X_h2 + static_cast<size_t>(t) * (K_blocks * 4);
    const float * xs_g = X_sc + static_cast<size_t>(t) * K_blocks;
    const int tot4 = K_blocks * 4;
    for (int i = tid; i < tot4; i += 512) s_xh2[i] = __ldg(&xh_g[i]);
    for (int i = tid; i < K_blocks; i += 512) s_xsc[i] = __ldg(&xs_g[i]);
    __syncthreads();

    const int blk_ff = blockIdx.x;
    const int row0   = blk_ff * 32 + warp * 2;
    const int blk_w  = lane >> 1;
    const int half   = lane & 1;

    float ag[2] = {0.0f, 0.0f}, au[2] = {0.0f, 0.0f};
    for (int b = blk_w; b < K_blocks; b += 16) {
        const int4  x0  = s_xh2[b * 4 + half * 2 + 0];
        const int4  x1  = s_xh2[b * 4 + half * 2 + 1];
        const float xsc = s_xsc[b];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row0 + r;
            const size_t roff = static_cast<size_t>(row) * K_blocks + b;
            const int4 gw = __ldg(&g_q[roff * 2 + half]);
            const int4 uw = __ldg(&u_q[roff * 2 + half]);
            const float gsc = q35_half_bits_to_f32(__ldg(&g_s[roff])) * xsc;
            const float usc = q35_half_bits_to_f32(__ldg(&u_s[roff])) * xsc;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
            const __half2 xh0 = *reinterpret_cast<const __half2 *>(&x0.x);
            const __half2 xh1 = *reinterpret_cast<const __half2 *>(&x0.y);
            const __half2 xh2 = *reinterpret_cast<const __half2 *>(&x0.z);
            const __half2 xh3 = *reinterpret_cast<const __half2 *>(&x0.w);
            const __half2 xh4 = *reinterpret_cast<const __half2 *>(&x1.x);
            const __half2 xh5 = *reinterpret_cast<const __half2 *>(&x1.y);
            const __half2 xh6 = *reinterpret_cast<const __half2 *>(&x1.z);
            const __half2 xh7 = *reinterpret_cast<const __half2 *>(&x1.w);
            __half2 h0, h1, h2, h3;
            q35_i8x4_to_h2x2(gw.x, h0, h1); q35_i8x4_to_h2x2(gw.y, h2, h3);
            __half2 sg = __hfma2(h3, xh3, __hfma2(h2, xh2, __hfma2(h1, xh1, __hmul2(h0, xh0))));
            q35_i8x4_to_h2x2(gw.z, h0, h1); q35_i8x4_to_h2x2(gw.w, h2, h3);
            sg = __hfma2(h3, xh7, __hfma2(h2, xh6, __hfma2(h1, xh5, __hfma2(h0, xh4, sg))));
            const float2 fg = __half22float2(sg);
            ag[r] += gsc * (fg.x + fg.y);

            q35_i8x4_to_h2x2(uw.x, h0, h1); q35_i8x4_to_h2x2(uw.y, h2, h3);
            __half2 su = __hfma2(h3, xh3, __hfma2(h2, xh2, __hfma2(h1, xh1, __hmul2(h0, xh0))));
            q35_i8x4_to_h2x2(uw.z, h0, h1); q35_i8x4_to_h2x2(uw.w, h2, h3);
            su = __hfma2(h3, xh7, __hfma2(h2, xh6, __hfma2(h1, xh5, __hfma2(h0, xh4, su))));
            const float2 fu = __half22float2(su);
            au[r] += usc * (fu.x + fu.y);
#else
            (void)gw; (void)uw; (void)gsc; (void)usc; (void)x0; (void)x1;
#endif
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float g = q35_warp_sum(ag[r]);
        const float u = q35_warp_sum(au[r]);
        if (lane == 0) {
            s_act[warp * 2 + r] = (g / (1.0f + expf(-g))) * u;
        }
    }
    __syncthreads();

    if (warp == 0) {
        const int ff_blks = ff_sh >> 5;
        const size_t b_out = static_cast<size_t>(t) * ff_blks + blk_ff;
        const float v = s_act[lane];
        float am = fabsf(v);
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_down_sync(0xffffffffu, am, off));
        am = __shfl_sync(0xffffffffu, am, 0);
        const float inv = (am > 0.0f) ? (8.0f / am) : 0.0f;
        const float vn  = v * inv;
        const float vp  = __shfl_xor_sync(0xffffffffu, vn, 1);
        if ((lane & 1) == 0) {
            A_h2[b_out * 16 + (lane >> 1)] = __float22half2_rn(make_float2(vn, vp));
        }
        if (lane == 0) A_sc[b_out] = am * 0.125f;
    }
}

__global__ void q35_shexp_dn_q8_0_sm60_h2_add_kernel(
    const int4 *     __restrict__ dn_q,
    const uint16_t * __restrict__ dn_s,
    const int4 *     __restrict__ A_h2,
    const float *    __restrict__ A_sc,
    const float *    __restrict__ gate_logit,
    float *          __restrict__ d_out,
    int K, int ff_sh) {

    extern __shared__ uint8_t s_raw[];
    const int ff_blks = ff_sh >> 5;
    int4 *  s_ah2 = reinterpret_cast<int4 *>(s_raw);
    float * s_asc = reinterpret_cast<float *>(s_raw + static_cast<size_t>(ff_blks) * 64);

    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int t   = blockIdx.y;
    const int4 *  ah_g = A_h2 + static_cast<size_t>(t) * (ff_blks * 4);
    const float * as_g = A_sc + static_cast<size_t>(t) * ff_blks;
    const int tot4 = ff_blks * 4;
    for (int i = tid; i < tot4; i += 256) s_ah2[i] = __ldg(&ah_g[i]);
    for (int i = tid; i < ff_blks; i += 256) s_asc[i] = __ldg(&as_g[i]);
    __syncthreads();

    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    if (row0 >= K) return;
    const int lane  = threadIdx.x;
    const int blk_w = lane >> 1;
    const int half  = lane & 1;

    float acc[2] = {0.0f, 0.0f};
    for (int b = blk_w; b < ff_blks; b += 16) {
        const int4  a0  = s_ah2[b * 4 + half * 2 + 0];
        const int4  a1  = s_ah2[b * 4 + half * 2 + 1];
        const float asc = s_asc[b];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row0 + r;
            if (row < K) {
                const size_t roff = static_cast<size_t>(row) * ff_blks + b;
                const int4 wv = __ldg(&dn_q[roff * 2 + half]);
                const float sc = q35_half_bits_to_f32(__ldg(&dn_s[roff])) * asc;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
                const __half2 ah0 = *reinterpret_cast<const __half2 *>(&a0.x);
                const __half2 ah1 = *reinterpret_cast<const __half2 *>(&a0.y);
                const __half2 ah2 = *reinterpret_cast<const __half2 *>(&a0.z);
                const __half2 ah3 = *reinterpret_cast<const __half2 *>(&a0.w);
                const __half2 ah4 = *reinterpret_cast<const __half2 *>(&a1.x);
                const __half2 ah5 = *reinterpret_cast<const __half2 *>(&a1.y);
                const __half2 ah6 = *reinterpret_cast<const __half2 *>(&a1.z);
                const __half2 ah7 = *reinterpret_cast<const __half2 *>(&a1.w);
                __half2 h0, h1, h2, h3;
                q35_i8x4_to_h2x2(wv.x, h0, h1); q35_i8x4_to_h2x2(wv.y, h2, h3);
                __half2 s = __hfma2(h3, ah3, __hfma2(h2, ah2, __hfma2(h1, ah1, __hmul2(h0, ah0))));
                q35_i8x4_to_h2x2(wv.z, h0, h1); q35_i8x4_to_h2x2(wv.w, h2, h3);
                s = __hfma2(h3, ah7, __hfma2(h2, ah6, __hfma2(h1, ah5, __hfma2(h0, ah4, s))));
                const float2 sf = __half22float2(s);
                acc[r] += sc * (sf.x + sf.y);
#else
                (void)wv; (void)sc; (void)a0; (void)a1;
#endif
            }
        }
    }
    const float sh_g = 1.0f / (1.0f + expf(-__ldg(&gate_logit[t])));
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float res = q35_warp_sum(acc[r]);
        if (lane == 0 && row0 + r < K) {
            d_out[static_cast<size_t>(t) * K + row0 + r] += sh_g * res;
        }
    }
}

} // namespace

bool try_launch_qwen_shexp_q8_0_sm60_decode(
    const DeviceTensor & gate_shexp, const DeviceTensor & up_shexp, const DeviceTensor & down_shexp,
    const float * d_x, const float * d_shexp_gate_logit,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, cudaStream_t stream, bool x_already_prepped) {

    if (T >= 16 || device_sm_version(gate_shexp.device) != 60) return false;
    if (!(gate_shexp.dtype == DType::Q8_0 && gate_shexp.is_soa &&
          up_shexp.dtype   == DType::Q8_0 && up_shexp.is_soa &&
          down_shexp.dtype == DType::Q8_0 && down_shexp.is_soa)) {
        return false;
    }
    const int K     = static_cast<int>(gate_shexp.ne[0]);
    const int ff_sh = static_cast<int>(gate_shexp.ne[1]);
    if ((K & 31) != 0 || (ff_sh & 31) != 0) return false;

    const int K_blocks = K >> 5;
    const int ff_blks  = ff_sh >> 5;
    const size_t gu_smem = static_cast<size_t>(K_blocks) * 68;
    const size_t dn_smem = static_cast<size_t>(ff_blks)  * 68;
    if (gu_smem > 32768 || dn_smem > 32768) return false;

    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    const size_t xh2_b = align256(static_cast<size_t>(T) * K * sizeof(__half));
    const size_t xsc_b = align256(static_cast<size_t>(T) * K_blocks * sizeof(float));
    const size_t ah2_b = align256(static_cast<size_t>(T) * ff_sh * sizeof(__half));
    const size_t asc_b = align256(static_cast<size_t>(T) * ff_blks * sizeof(float));
    if (xh2_b + xsc_b + ah2_b + asc_b > scratch_bytes) return false;

    auto * base = static_cast<uint8_t *>(d_scratch);
    void * x_prep = base;
    auto * a_h2   = reinterpret_cast<__half2 *>(base + xh2_b + xsc_b);
    auto * a_sc   = reinterpret_cast<float *>(base + xh2_b + xsc_b + ah2_b);

    if (!x_already_prepped) {
        launch_q8_0_prep_x_sm60_h2(d_x, x_prep, K, T, stream);
    }
    const auto * x_h2 = static_cast<const int4 *>(x_prep);
    const auto * x_sc = reinterpret_cast<const float *>(static_cast<const uint8_t *>(x_prep) + xh2_b);

    q35_shexp_gu_q8_0_sm60_h2_fused_prep_kernel<<<dim3(ff_blks, T), dim3(32, 16), gu_smem, stream>>>(
        static_cast<const int4 *>(gate_shexp.ptr), static_cast<const uint16_t *>(gate_shexp.scales_ptr),
        static_cast<const int4 *>(up_shexp.ptr),   static_cast<const uint16_t *>(up_shexp.scales_ptr),
        x_h2, x_sc, a_h2, a_sc, ff_sh, K);

    constexpr int wpb = 8;
    q35_shexp_dn_q8_0_sm60_h2_add_kernel<<<dim3((K + wpb * 2 - 1) / (wpb * 2), T), dim3(32, wpb), dn_smem, stream>>>(
        static_cast<const int4 *>(down_shexp.ptr), static_cast<const uint16_t *>(down_shexp.scales_ptr),
        reinterpret_cast<const int4 *>(a_h2), a_sc, d_shexp_gate_logit, d_out, K, ff_sh);
    return true;
}

bool try_launch_moe_iq_pascal(
    const DeviceTensor & gate_exps, const DeviceTensor & up_exps, const DeviceTensor & down_exps,
    const DeviceTensor & gate_shexp,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, float clamp_exp, cudaStream_t stream) {

    const int sm = device_sm_version(gate_exps.device);
    if (sm != 60 && sm != 61) return false;
    if (gate_shexp.valid()) return false;
    if (gate_exps.dtype != DType::IQ3_XXS || up_exps.dtype != DType::IQ3_XXS) {
        return false;
    }
    const bool dn_iq4xs = (down_exps.dtype == DType::IQ4_XS);
    const bool dn_q6k   = (down_exps.dtype == DType::Q6_K && down_exps.is_soa && sm == 60);
    if (!dn_iq4xs && !dn_q6k) {
        return false;
    }

    const int K        = static_cast<int>(gate_exps.ne[0]);
    const int ff       = static_cast<int>(gate_exps.ne[1]);
    const int n_expert = static_cast<int>(gate_exps.ne[2]);
    if ((K & 255) != 0 || (ff & 255) != 0 || n_expert <= 0 || n_expert > 512 || k_used <= 0 || k_used > 16) {
        return false;
    }

    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    auto * base = static_cast<uint8_t *>(d_scratch);

    // 1. Decode path (T < 16) on sm_60 (Direct FP16 LUT + fused A-prep + Split-Slot Down) & sm_61 (Split-Slot dp4a!)
    if (T < 16) {
        if (sm == 60) {
            const int K_blocks  = K >> 5;
            const int ff_blks   = ff >> 5;
            const size_t gu_smem = static_cast<size_t>(K_blocks) * 68;
            const size_t dn_smem = static_cast<size_t>(ff_blks)  * 68;
            if (gu_smem > 32768 || dn_smem > 32768) return false;

            const size_t xh2_b   = align256(static_cast<size_t>(T) * K * sizeof(__half));
            const size_t xsc_b   = align256(static_cast<size_t>(T) * K_blocks * sizeof(float));
            const size_t ah2_b   = align256(static_cast<size_t>(T) * k_used * ff * sizeof(__half));
            const size_t asc_b   = align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(float));
            const size_t slots_b = align256(static_cast<size_t>(T) * k_used * K * sizeof(float));
            if (xh2_b + xsc_b + ah2_b + asc_b + slots_b > scratch_bytes) return false;

            size_t off = 0;
            void *  x_prep     = base + off; off += xh2_b + xsc_b;
            auto *  a_h2       = reinterpret_cast<__half2 *>(base + off); off += ah2_b;
            auto *  a_sc       = reinterpret_cast<float *>(base + off);   off += asc_b;
            auto *  down_slots = reinterpret_cast<float *>(base + off);

            launch_q8_0_prep_x_sm60_h2(d_x, x_prep, K, T, stream);
            const auto * x_h2 = static_cast<const int4 *>(x_prep);
            const auto * x_sc = reinterpret_cast<const float *>(static_cast<const uint8_t *>(x_prep) + xh2_b);

            q35_moe_gu_iq3xxs_decode_sm60_h2_fused_prep_kernel
                <<<dim3(ff_blks, k_used, T), dim3(32, 16), gu_smem, stream>>>(
                    static_cast<const BlockIQ3_XXS_P *>(gate_exps.ptr),
                    static_cast<const BlockIQ3_XXS_P *>(up_exps.ptr),
                    x_h2, x_sc, d_top_ids, a_h2, a_sc, ff, K, k_used, clamp_exp);

            constexpr int wpb = 8;
            if (dn_iq4xs) {
                q35_moe_dn_iq4xs_split_decode_sm60_h2_kernel
                    <<<dim3((K + wpb * 2 - 1) / (wpb * 2), k_used, T), dim3(32, wpb), dn_smem, stream>>>(
                        static_cast<const BlockIQ4_XS_P *>(down_exps.ptr),
                        reinterpret_cast<const uint2 *>(a_h2), a_sc,
                        d_top_ids, d_top_w, down_slots, K, ff, k_used);
            } else {
                q35_moe_dn_q6k_soa_split_decode_sm60_h2_kernel
                    <<<dim3((K + wpb * 2 - 1) / (wpb * 2), k_used, T), dim3(32, wpb), dn_smem, stream>>>(
                        static_cast<const uint2 *>(down_exps.ptr),
                        static_cast<const uint2 *>(down_exps.aux_ptr),
                        static_cast<const int8_t *>(down_exps.scales_ptr),
                        static_cast<const uint16_t *>(down_exps.d_ptr),
                        reinterpret_cast<const int4 *>(a_h2), a_sc,
                        d_top_ids, d_top_w, down_slots, K, ff, k_used);
            }

            q35_reduce_slots_kernel<<<dim3((K + 255) / 256, T), 256, 0, stream>>>(down_slots, d_out, K, k_used);
            return true;
        } else {
            // sm_61 (GTX 1060): Split-slot hardware __dp4a decode!
            const int K_blocks  = K >> 5;
            const int ff_blks   = ff >> 5;
            const size_t gu_smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
            const size_t dn_smem = static_cast<size_t>(ff_blks)  * sizeof(BlockQ8_1_Act);
            if (gu_smem > 32768 || dn_smem > 32768) return false;

            const size_t xq8_b   = align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
            const size_t act_b   = align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
            const size_t aq8_b   = align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BlockQ8_1_Act));
            const size_t slots_b = align256(static_cast<size_t>(T) * k_used * K * sizeof(float));
            if (xq8_b + act_b + aq8_b + slots_b > scratch_bytes) return false;

            size_t off = 0;
            auto * xq8        = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += xq8_b;
            auto * act        = reinterpret_cast<float *>(base + off);         off += act_b;
            auto * aq8        = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += aq8_b;
            auto * down_slots = reinterpret_cast<float *>(base + off);

            launch_quantize_q8_1(d_x, xq8, K, T, stream);
            constexpr int wpb = 8;
            q35_moe_gu_iq3xxs_decode_sm61_dp4a_kernel
                <<<dim3((ff + wpb - 1) / wpb, k_used, T), dim3(32, wpb), gu_smem, stream>>>(
                    static_cast<const BlockIQ3_XXS_P *>(gate_exps.ptr),
                    static_cast<const BlockIQ3_XXS_P *>(up_exps.ptr),
                    xq8, d_top_ids, act, ff, K, k_used, clamp_exp);

            launch_quantize_q8_1(act, aq8, k_used * ff, T, stream);
            q35_moe_dn_iq4xs_split_decode_sm61_dp4a_kernel
                <<<dim3((K + wpb * 2 - 1) / (wpb * 2), k_used, T), dim3(32, wpb), dn_smem, stream>>>(
                    static_cast<const BlockIQ4_XS_P *>(down_exps.ptr),
                    aq8, d_top_ids, d_top_w, down_slots, K, ff, k_used);

            q35_reduce_slots_kernel<<<dim3((K + 255) / 256, T), 256, 0, stream>>>(down_slots, d_out, K, k_used);
            return true;
        }
    }

    // 2. Prefill path (T >= 16): adaptive BT=8 (T < 256) / BT=16 (T >= 256) streaming MoE!
    auto run_prefill = [&](auto bt_tag) -> bool {
        constexpr int BT = decltype(bt_tag)::value;
        const int total_pairs = T * k_used;
        const int max_tiles   = (total_pairs / BT) + n_expert;

        const size_t act_b   = align256(static_cast<size_t>(total_pairs) * ff * sizeof(float));
        const size_t sp_b    = align256(static_cast<size_t>(total_pairs) * sizeof(int32_t));
        const size_t tile_b  = align256(static_cast<size_t>(max_tiles)   * sizeof(Q35TileDesc));
        const size_t nt_b    = align256(sizeof(int32_t));
        const size_t slots_b = align256(static_cast<size_t>(total_pairs) * K * sizeof(float));

        if (sm == 60) {
            const size_t xh2_b = align256(static_cast<size_t>(T) * K * sizeof(__half));
            const size_t xsc_b = align256(static_cast<size_t>(T) * sizeof(float));
            const size_t ah2_b = align256(static_cast<size_t>(total_pairs) * ff * sizeof(__half));
            const size_t asc_b = align256(static_cast<size_t>(total_pairs) * sizeof(float));
            const size_t need  = act_b + sp_b + tile_b + nt_b + slots_b + xh2_b + xsc_b + ah2_b + asc_b;
            if (need > scratch_bytes) return false;

            size_t off = 0;
            auto * act          = reinterpret_cast<float *>(base + off);       off += act_b;
            auto * sorted_pairs = reinterpret_cast<int32_t *>(base + off);     off += sp_b;
            auto * tiles        = reinterpret_cast<Q35TileDesc *>(base + off); off += tile_b;
            auto * n_tiles_ptr  = reinterpret_cast<int32_t *>(base + off);     off += nt_b;
            auto * down_slots   = reinterpret_cast<float *>(base + off);       off += slots_b;
            auto * x_h2         = reinterpret_cast<__half2 *>(base + off);     off += xh2_b;
            auto * x_rsc        = reinterpret_cast<float *>(base + off);       off += xsc_b;
            auto * a_h2         = reinterpret_cast<__half2 *>(base + off);     off += ah2_b;
            auto * a_rsc        = reinterpret_cast<float *>(base + off);

            q35_prep_row_h2_kernel<<<T, 256, 0, stream>>>(d_x, x_h2, x_rsc, K);

            q35_group_and_tile_kernel<BT><<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
                d_top_ids, sorted_pairs, tiles, n_tiles_ptr, total_pairs, n_expert, k_used);

            q35_moe_gu_iq3xxs_sm60_h2_kernel<BT><<<dim3((ff + 7) / 8, max_tiles), dim3(32, 8), 0, stream>>>(
                static_cast<const BlockIQ3_XXS_P *>(gate_exps.ptr),
                static_cast<const BlockIQ3_XXS_P *>(up_exps.ptr),
                reinterpret_cast<const int4 *>(x_h2), x_rsc,
                tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, clamp_exp);

            q35_prep_row_h2_kernel<<<total_pairs, 256, 0, stream>>>(act, a_h2, a_rsc, ff);

            if (dn_iq4xs) {
                q35_moe_dn_iq4xs_sm60_h2_kernel<BT><<<dim3((K + 15) / 16, max_tiles), dim3(32, 8), 0, stream>>>(
                    static_cast<const BlockIQ4_XS_P *>(down_exps.ptr),
                    reinterpret_cast<const int4 *>(a_h2), a_rsc, d_top_w,
                    tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
            } else {
                q35_moe_dn_q6k_soa_sm60_h2_kernel<BT><<<dim3((K + 15) / 16, max_tiles), dim3(32, 8), 0, stream>>>(
                    static_cast<const uint2 *>(down_exps.ptr),
                    static_cast<const uint2 *>(down_exps.aux_ptr),
                    static_cast<const int8_t *>(down_exps.scales_ptr),
                    static_cast<const uint16_t *>(down_exps.d_ptr),
                    reinterpret_cast<const int4 *>(a_h2), a_rsc, d_top_w,
                    tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
            }

            q35_reduce_slots_kernel<<<dim3((K + 255) / 256, T), 256, 0, stream>>>(down_slots, d_out, K, k_used);
            return true;
        } else {
            const size_t xq8_b = align256(static_cast<size_t>(T) * (K >> 5) * sizeof(BlockQ8_1_Act));
            const size_t aq8_b = align256(static_cast<size_t>(total_pairs) * (ff >> 5) * sizeof(BlockQ8_1_Act));
            const size_t need  = act_b + sp_b + tile_b + nt_b + slots_b + xq8_b + aq8_b;
            if (need > scratch_bytes) return false;

            size_t off = 0;
            auto * act          = reinterpret_cast<float *>(base + off);         off += act_b;
            auto * sorted_pairs = reinterpret_cast<int32_t *>(base + off);       off += sp_b;
            auto * tiles        = reinterpret_cast<Q35TileDesc *>(base + off);   off += tile_b;
            auto * n_tiles_ptr  = reinterpret_cast<int32_t *>(base + off);       off += nt_b;
            auto * down_slots   = reinterpret_cast<float *>(base + off);         off += slots_b;
            auto * xq8          = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += xq8_b;
            auto * aq8          = reinterpret_cast<BlockQ8_1_Act *>(base + off);

            launch_quantize_q8_1(d_x, xq8, K, T, stream);
            q35_group_and_tile_kernel<BT><<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
                d_top_ids, sorted_pairs, tiles, n_tiles_ptr, total_pairs, n_expert, k_used);

            q35_moe_gu_iq3xxs_sm61_dp4a_kernel<BT><<<dim3((ff + 7) / 8, max_tiles), dim3(32, 8), 0, stream>>>(
                static_cast<const BlockIQ3_XXS_P *>(gate_exps.ptr),
                static_cast<const BlockIQ3_XXS_P *>(up_exps.ptr),
                xq8, tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, clamp_exp);

            launch_quantize_q8_1(act, aq8, k_used * ff, T, stream);

            q35_moe_dn_iq4xs_sm61_dp4a_kernel<BT><<<dim3((K + 7) / 8, max_tiles), dim3(32, 8), 0, stream>>>(
                static_cast<const BlockIQ4_XS_P *>(down_exps.ptr),
                aq8, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);

            q35_reduce_slots_kernel<<<dim3((K + 255) / 256, T), 256, 0, stream>>>(down_slots, d_out, K, k_used);
            return true;
        }
    };

    if (T >= 256) return run_prefill(std::integral_constant<int, 16>{});
    return run_prefill(std::integral_constant<int, 8>{});
}

} // namespace rookery
