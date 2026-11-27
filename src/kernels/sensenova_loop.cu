#include "core/device.h"
#include "kernels/kernels.h"
#include "kernels/q8_1_act.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

namespace rookery {

namespace {

__device__ __forceinline__ float sn_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float sn_warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ float sn_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sn_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
__device__ __forceinline__ void sn_i8x4_to_h2x2(int w_i32, __half2 & h_lo, __half2 & h_hi) {
    const uint32_t u  = static_cast<uint32_t>(w_i32) ^ 0x80808080u;
    const uint32_t p0 = __byte_perm(u, 0x64u, 0x4140u);
    const uint32_t p1 = __byte_perm(u, 0x64u, 0x4342u);
    constexpr uint32_t k1152 = 0x64806480u;
    const __half2 c1152 = *reinterpret_cast<const __half2 *>(&k1152);
    h_lo = __hsub2(*reinterpret_cast<const __half2 *>(&p0), c1152);
    h_hi = __hsub2(*reinterpret_cast<const __half2 *>(&p1), c1152);
}
#endif

// Exact 2x2 log-space Sinkhorn-Knopp normalization matching modeling_sensenova_loop.py
__device__ __forceinline__ void sn_sinkhorn_2x2(
    float a, float b, float c, float d,
    int iterations, bool fixed_identity,
    float & m00, float & m01, float & m10, float & m11) {

    if (fixed_identity) {
        m00 = 1.0f; m01 = 0.0f; m10 = 0.0f; m11 = 1.0f;
        return;
    }
    for (int it = 0; it < iterations; ++it) {
        float mr0 = fmaxf(a, b);
        float lr0 = mr0 + logf(expf(a - mr0) + expf(b - mr0));
        a -= lr0; b -= lr0;

        float mr1 = fmaxf(c, d);
        float lr1 = mr1 + logf(expf(c - mr1) + expf(d - mr1));
        c -= lr1; d -= lr1;

        float mc0 = fmaxf(a, c);
        float lc0 = mc0 + logf(expf(a - mc0) + expf(c - mc0));
        a -= lc0; c -= lc0;

        float mc1 = fmaxf(b, d);
        float lc1 = mc1 + logf(expf(b - mc1) + expf(d - mc1));
        b -= lc1; d -= lc1;
    }
    m00 = expf(a); m01 = expf(b); m10 = expf(c); m11 = expf(d);
}

// In-warp helper: quantize 32 contiguous floats `v` across lanes 0..31 into either
// sm_60 FP16x2 prep (x_h2 + x_sc) OR sm_61+ BlockQ8_1_Act!
template <int PREP_MODE>
__device__ __forceinline__ void sn_store_prepped_block32(
    float v, void * d_xprep, int block_idx, int total_h2_bytes) {

    const int lane = threadIdx.x & 31;
    if constexpr (PREP_MODE == 1) {
        // sm_60 FP16x2 prep
        auto * x_h2 = static_cast<__half2 *>(d_xprep);
        auto * x_sc = reinterpret_cast<float *>(static_cast<uint8_t *>(d_xprep) + total_h2_bytes);
        const float amax = __shfl_sync(0xffffffffu, sn_warp_max(fabsf(v)), 0);
        const float inv  = (amax > 0.0f) ? (8.0f / amax) : 0.0f;
        const float vn   = v * inv;
        const float vp   = __shfl_xor_sync(0xffffffffu, vn, 1);
        if ((lane & 1) == 0) {
            x_h2[static_cast<size_t>(block_idx) * 16 + (lane >> 1)] = __float22half2_rn(make_float2(vn, vp));
        }
        if (lane == 0) x_sc[block_idx] = amax * 0.125f;
    } else if constexpr (PREP_MODE == 2) {
        // sm_61+ BlockQ8_1_Act
        auto * qx = static_cast<BlockQ8_1_Act *>(d_xprep);
        const float amax = __shfl_sync(0xffffffffu, sn_warp_max(fabsf(v)), 0);
        const float d    = amax / 127.0f;
        const float inv  = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        const int   q    = __float2int_rn(v * inv);
        qx[block_idx].qs[lane] = static_cast<int8_t>(q);
        int hsum = q;
        hsum += __shfl_xor_sync(0xffffffffu, hsum, 1);
        hsum += __shfl_xor_sync(0xffffffffu, hsum, 2);
        hsum += __shfl_xor_sync(0xffffffffu, hsum, 4);
        hsum += __shfl_xor_sync(0xffffffffu, hsum, 8);
        const int s0 = __shfl_sync(0xffffffffu, hsum, 0);
        const int s1 = __shfl_sync(0xffffffffu, hsum, 16);
        if (lane == 0) {
            qx[block_idx].d     = d;
            qx[block_idx].s0    = s0;
            qx[block_idx].s1    = s1;
            qx[block_idx].s_all = d * static_cast<float>(s0 + s1);
        }
    }
}

// ============================================================================
// 1. 2-Stream mHC Enter / Router / Aggregate+RMSNorm(+Prep!) / Merge / Leave
// ============================================================================

__global__ void sn_mhc_enter_kernel(
    const float * __restrict__ src,
    float *       __restrict__ dst,
    int H) {

    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int t  = blockIdx.y;
    const int H4 = H >> 2;
    if (i4 >= H4) return;
    const float4 v = reinterpret_cast<const float4 *>(src + static_cast<size_t>(t) * H)[i4];
    auto * d0 = reinterpret_cast<float4 *>(dst + static_cast<size_t>(t) * 2 * H);
    auto * d1 = reinterpret_cast<float4 *>(dst + (static_cast<size_t>(t) * 2 + 1) * H);
    d0[i4] = v;
    d1[i4] = v;
}

// Decode (T <= 4) Split-K mHC Router Partials (S=16 blocks x 256 threads — optimal on P100!)
template <int S>
__global__ void sn_mhc_router_part_kernel(
    const float * __restrict__ res,
    const float * __restrict__ w_hc,
    float *       __restrict__ d_part,
    int H2) {

    const int s    = blockIdx.x;
    const int t    = blockIdx.y;
    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;    // 0..7

    const int H2_4  = H2 >> 2;
    const int chunk = (H2_4 + S - 1) / S;
    const int i0    = s * chunk;
    const int i1    = (i0 + chunk < H2_4) ? (i0 + chunk) : H2_4;

    const auto * x4 = reinterpret_cast<const float4 *>(res + static_cast<size_t>(t) * H2);
    float ss = 0.0f;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    float d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;

    for (int i = i0 + tid; i < i1; i += 256) {
        const float4 xv = x4[i];
        ss += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
        #define ACC_W(ROW, ACC) { \
            const float4 wv = __ldg(&reinterpret_cast<const float4 *>(w_hc + static_cast<size_t>(ROW) * H2)[i]); \
            ACC += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w; \
        }
        ACC_W(0, d0); ACC_W(1, d1); ACC_W(2, d2); ACC_W(3, d3);
        ACC_W(4, d4); ACC_W(5, d5); ACC_W(6, d6); ACC_W(7, d7);
        #undef ACC_W
    }

    __shared__ float s_red[9][8];
    ss = sn_warp_sum(ss);
    d0 = sn_warp_sum(d0); d1 = sn_warp_sum(d1); d2 = sn_warp_sum(d2); d3 = sn_warp_sum(d3);
    d4 = sn_warp_sum(d4); d5 = sn_warp_sum(d5); d6 = sn_warp_sum(d6); d7 = sn_warp_sum(d7);
    if (lane == 0) {
        s_red[0][warp] = ss;
        s_red[1][warp] = d0; s_red[2][warp] = d1; s_red[3][warp] = d2; s_red[4][warp] = d3;
        s_red[5][warp] = d4; s_red[6][warp] = d5; s_red[7][warp] = d6; s_red[8][warp] = d7;
    }
    __syncthreads();

    if (tid < 9) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) sum += s_red[tid][w];
        d_part[(static_cast<size_t>(t) * S + s) * 9 + tid] = sum;
    }
}

// Decode (T <= 4) FUSED mHC Merge + Next mHC Router Partials (S=16 blocks x 256 threads!)
template <int S>
__global__ void sn_mhc_merge_router_part_kernel(
    const float * __restrict__ res_in,
    const float * __restrict__ branch,
    const float * __restrict__ ctrl,
    float *       __restrict__ res_out,
    const float * __restrict__ next_w_hc,
    float *       __restrict__ d_part,
    int H) {

    const int s    = blockIdx.x;
    const int t    = blockIdx.y;
    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;    // 0..7

    const float * ct = ctrl + static_cast<size_t>(t) * 6;
    const float p0 = ct[0], p1 = ct[1];
    const float m00 = ct[2], m01 = ct[3], m10 = ct[4], m11 = ct[5];

    const int H4    = H >> 2;
    const int H2    = H << 1;
    const int chunk = (H4 + S - 1) / S;
    const int i0    = s * chunk;
    const int i1    = (i0 + chunk < H4) ? (i0 + chunk) : H4;

    const auto * r0_in  = reinterpret_cast<const float4 *>(res_in  + static_cast<size_t>(t) * H2);
    const auto * r1_in  = reinterpret_cast<const float4 *>(res_in  + static_cast<size_t>(t) * H2 + H);
    const auto * br_in  = reinterpret_cast<const float4 *>(branch  + static_cast<size_t>(t) * H);
    auto *       r0_out = reinterpret_cast<float4 *>(res_out + static_cast<size_t>(t) * H2);
    auto *       r1_out = reinterpret_cast<float4 *>(res_out + static_cast<size_t>(t) * H2 + H);

    float ss = 0.0f;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    float d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;

    for (int i = i0 + tid; i < i1; i += 256) {
        const float4 a  = r0_in[i];
        const float4 b  = r1_in[i];
        const float4 br = br_in[i];

        const float4 n0 = make_float4(
            m00 * a.x + m01 * b.x + p0 * br.x,
            m00 * a.y + m01 * b.y + p0 * br.y,
            m00 * a.z + m01 * b.z + p0 * br.z,
            m00 * a.w + m01 * b.w + p0 * br.w);
        const float4 n1 = make_float4(
            m10 * a.x + m11 * b.x + p1 * br.x,
            m10 * a.y + m11 * b.y + p1 * br.y,
            m10 * a.z + m11 * b.z + p1 * br.z,
            m10 * a.w + m11 * b.w + p1 * br.w);

        r0_out[i] = n0;
        r1_out[i] = n1;

        ss += n0.x * n0.x + n0.y * n0.y + n0.z * n0.z + n0.w * n0.w
            + n1.x * n1.x + n1.y * n1.y + n1.z * n1.z + n1.w * n1.w;

        #define ACC_W2(ROW, ACC) { \
            const float * wr = next_w_hc + static_cast<size_t>(ROW) * H2; \
            const float4 w0 = __ldg(&reinterpret_cast<const float4 *>(wr)[i]); \
            const float4 w1 = __ldg(&reinterpret_cast<const float4 *>(wr + H)[i]); \
            ACC += w0.x * n0.x + w0.y * n0.y + w0.z * n0.z + w0.w * n0.w \
                 + w1.x * n1.x + w1.y * n1.y + w1.z * n1.z + w1.w * n1.w; \
        }
        ACC_W2(0, d0); ACC_W2(1, d1); ACC_W2(2, d2); ACC_W2(3, d3);
        ACC_W2(4, d4); ACC_W2(5, d5); ACC_W2(6, d6); ACC_W2(7, d7);
        #undef ACC_W2
    }

    __shared__ float s_red[9][8];
    ss = sn_warp_sum(ss);
    d0 = sn_warp_sum(d0); d1 = sn_warp_sum(d1); d2 = sn_warp_sum(d2); d3 = sn_warp_sum(d3);
    d4 = sn_warp_sum(d4); d5 = sn_warp_sum(d5); d6 = sn_warp_sum(d6); d7 = sn_warp_sum(d7);
    if (lane == 0) {
        s_red[0][warp] = ss;
        s_red[1][warp] = d0; s_red[2][warp] = d1; s_red[3][warp] = d2; s_red[4][warp] = d3;
        s_red[5][warp] = d4; s_red[6][warp] = d5; s_red[7][warp] = d6; s_red[8][warp] = d7;
    }
    __syncthreads();

    if (tid < 9) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) sum += s_red[tid][w];
        d_part[(static_cast<size_t>(t) * S + s) * 9 + tid] = sum;
    }
}

// Decode (T <= 4): Reduce S=32 router partials -> 2x2 Sinkhorn -> Aggregate -> RMSNorm ->
// DIRECTLY emit prepped activation (PREP_MODE: 0=x_norm F32 only, 1=sm_60 h2+sc, 2=sm_61+ Q8_1)!
// 8 warps/CTA, each warp owns 16 contiguous 32-elem blocks (for H=4096) -> ZERO extra prep launch!
template <int S, int PREP_MODE>
__global__ void sn_mhc_reduce_agg_rmsnorm_prep_kernel(
    const float * __restrict__ res,     // [T, 2, H]
    const float * __restrict__ d_part,  // [T, S, 9]
    const float * __restrict__ b_hc,    // [8]
    const float * __restrict__ norm_w,  // [H]
    float *       __restrict__ ctrl,    // [T, 6]
    float *       __restrict__ x_norm,  // [T, H] (only written if PREP_MODE == 0)
    void *        __restrict__ d_xprep, // prepped activation (when PREP_MODE > 0)
    int H, int T, int sinkhorn_iters, bool fixed_identity, float pre_scale, float rms_eps) {

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;    // 0..7

    __shared__ float s_val[9];
    __shared__ float s_pre[2];
    __shared__ float s_red[8];

    // S=32 -> warp 0..8? Use tid < 9 summing 32 partials (or 9 warps? Only 8 warps!)
    if (tid < 9) {
        float sum = 0.0f;
        #pragma unroll
        for (int s = 0; s < S; ++s) {
            sum += d_part[(static_cast<size_t>(t) * S + s) * 9 + tid];
        }
        s_val[tid] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        const float rms = rsqrtf(s_val[0] / static_cast<float>(2 * H) + 1e-6f);
        float lg[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) lg[j] = s_val[1 + j] * rms + b_hc[j];

        const float pre0  = sn_sigmoid(lg[0]) * pre_scale;
        const float pre1  = sn_sigmoid(lg[1]) * pre_scale;
        const float post0 = 2.0f * sn_sigmoid(lg[2]);
        const float post1 = 2.0f * sn_sigmoid(lg[3]);
        float m00, m01, m10, m11;
        sn_sinkhorn_2x2(lg[4], lg[5], lg[6], lg[7], sinkhorn_iters, fixed_identity, m00, m01, m10, m11);

        s_pre[0] = pre0;
        s_pre[1] = pre1;
        float * ct = ctrl + static_cast<size_t>(t) * 6;
        ct[0] = post0; ct[1] = post1;
        ct[2] = m00;   ct[3] = m01;
        ct[4] = m10;   ct[5] = m11;
    }
    __syncthreads();

    const float pre0 = s_pre[0];
    const float pre1 = s_pre[1];
    const int K_blocks = H >> 5; // 128 for H=4096
    const float * r0 = res + static_cast<size_t>(t) * 2 * H;
    const float * r1 = r0 + H;

    // Warp-contiguous 32-element blocks so each warp can immediately quantize/prep in registers!
    float br_reg[16];
    float ss = 0.0f;
    if (K_blocks == 128) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int d = (warp * 16 + j) * 32 + lane;
            const float v = pre0 * r0[d] + pre1 * r1[d];
            br_reg[j] = v;
            ss += v * v;
        }
    } else {
        for (int b = warp; b < K_blocks; b += 8) {
            const int d = b * 32 + lane;
            const float v = pre0 * r0[d] + pre1 * r1[d];
            ss += v * v;
        }
    }

    ss = sn_warp_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? s_red[lane] : 0.0f;
        v = sn_warp_sum(v);
        if (lane == 0) s_red[0] = rsqrtf(v / static_cast<float>(H) + rms_eps);
    }
    __syncthreads();
    const float inv = s_red[0];
    const int total_h2_bytes = static_cast<int>((static_cast<size_t>(T) * H * sizeof(__half) + 255) & ~size_t(255));

    if (K_blocks == 128) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int b  = warp * 16 + j;
            const int d  = b * 32 + lane;
            const float xn = br_reg[j] * inv * __ldg(&norm_w[d]);
            if constexpr (PREP_MODE == 0) {
                x_norm[static_cast<size_t>(t) * H + d] = xn;
            } else {
                sn_store_prepped_block32<PREP_MODE>(xn, d_xprep, t * K_blocks + b, total_h2_bytes);
            }
        }
    } else {
        for (int b = warp; b < K_blocks; b += 8) {
            const int d = b * 32 + lane;
            const float xn = (pre0 * r0[d] + pre1 * r1[d]) * inv * __ldg(&norm_w[d]);
            if constexpr (PREP_MODE == 0) {
                x_norm[static_cast<size_t>(t) * H + d] = xn;
            } else {
                sn_store_prepped_block32<PREP_MODE>(xn, d_xprep, t * K_blocks + b, total_h2_bytes);
            }
        }
    }
}

// Fused RMSNorm (or Add + RMSNorm) + Direct Activation Prep (T <= 2) for non-mHC layers & output_norm!
template <bool DO_ADD, int PREP_MODE>
__global__ void sn_rmsnorm_prep_kernel(
    float *       __restrict__ cur,     // [T, H] (updated in-place if DO_ADD)
    const float * __restrict__ sub,     // [T, H] (only used if DO_ADD)
    const float * __restrict__ norm_w,  // [H]
    void *        __restrict__ d_xprep, // prepped activation
    float *       __restrict__ x_out,   // optional [T, H] F32 copy (e.g. d_h_nextn)
    int H, int T, float rms_eps) {

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int K_blocks = H >> 5;

    float *       cr = cur + static_cast<size_t>(t) * H;
    const float * sr = DO_ADD ? (sub + static_cast<size_t>(t) * H) : nullptr;

    float reg[16];
    float ss = 0.0f;
    if (K_blocks == 128) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int d = (warp * 16 + j) * 32 + lane;
            float v = cr[d];
            if constexpr (DO_ADD) { v += sr[d]; cr[d] = v; }
            reg[j] = v;
            ss += v * v;
        }
    } else {
        for (int b = warp; b < K_blocks; b += 8) {
            const int d = b * 32 + lane;
            float v = cr[d];
            if constexpr (DO_ADD) { v += sr[d]; cr[d] = v; }
            ss += v * v;
        }
    }

    __shared__ float s_red[8];
    ss = sn_warp_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? s_red[lane] : 0.0f;
        v = sn_warp_sum(v);
        if (lane == 0) s_red[0] = rsqrtf(v / static_cast<float>(H) + rms_eps);
    }
    __syncthreads();
    const float inv = s_red[0];
    const int total_h2_bytes = static_cast<int>((static_cast<size_t>(T) * H * sizeof(__half) + 255) & ~size_t(255));

    if (K_blocks == 128) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int b = warp * 16 + j;
            const int d = b * 32 + lane;
            const float xn = reg[j] * inv * __ldg(&norm_w[d]);
            if (x_out != nullptr) x_out[static_cast<size_t>(t) * H + d] = xn;
            sn_store_prepped_block32<PREP_MODE>(xn, d_xprep, t * K_blocks + b, total_h2_bytes);
        }
    } else {
        for (int b = warp; b < K_blocks; b += 8) {
            const int d = b * 32 + lane;
            const float xn = cr[d] * inv * __ldg(&norm_w[d]);
            if (x_out != nullptr) x_out[static_cast<size_t>(t) * H + d] = xn;
            sn_store_prepped_block32<PREP_MODE>(xn, d_xprep, t * K_blocks + b, total_h2_bytes);
        }
    }
}

// Prefill (T >= 5): Flat RMSNorm of res [T, 2*H] -> d_flat_norm [T, 2*H]
__global__ void sn_flat_rmsnorm_kernel(
    const float * __restrict__ src,
    float *       __restrict__ dst,
    int H2) {

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int H2_4 = H2 >> 2;
    const auto * x4 = reinterpret_cast<const float4 *>(src + static_cast<size_t>(t) * H2);
    auto *       y4 = reinterpret_cast<float4 *>(dst + static_cast<size_t>(t) * H2);

    float ss = 0.0f;
    for (int i = tid; i < H2_4; i += 256) {
        const float4 v = x4[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    __shared__ float s_red[8];
    ss = sn_warp_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? s_red[lane] : 0.0f;
        v = sn_warp_sum(v);
        if (lane == 0) s_red[0] = rsqrtf(v / static_cast<float>(H2) + 1e-6f);
    }
    __syncthreads();
    const float inv = s_red[0];
    for (int i = tid; i < H2_4; i += 256) {
        const float4 v = x4[i];
        y4[i] = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
    }
}

__global__ void sn_mhc_logits_agg_rmsnorm_kernel(
    const float * __restrict__ res,
    const float * __restrict__ logits,
    const float * __restrict__ b_hc,
    const float * __restrict__ norm_w,
    float *       __restrict__ ctrl,
    float *       __restrict__ x_norm,
    int H, int sinkhorn_iters, bool fixed_identity, float pre_scale, float rms_eps) {

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    __shared__ float s_pre[2];
    __shared__ float s_red[8];
    if (tid == 0) {
        const float * lg_in = logits + static_cast<size_t>(t) * 8;
        float lg[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) lg[j] = lg_in[j] + b_hc[j];

        s_pre[0] = sn_sigmoid(lg[0]) * pre_scale;
        s_pre[1] = sn_sigmoid(lg[1]) * pre_scale;
        float m00, m01, m10, m11;
        sn_sinkhorn_2x2(lg[4], lg[5], lg[6], lg[7], sinkhorn_iters, fixed_identity, m00, m01, m10, m11);
        float * ct = ctrl + static_cast<size_t>(t) * 6;
        ct[0] = 2.0f * sn_sigmoid(lg[2]);
        ct[1] = 2.0f * sn_sigmoid(lg[3]);
        ct[2] = m00; ct[3] = m01; ct[4] = m10; ct[5] = m11;
    }
    __syncthreads();

    const float pre0 = s_pre[0];
    const float pre1 = s_pre[1];
    const int H4 = H >> 2;
    const auto * r0 = reinterpret_cast<const float4 *>(res + static_cast<size_t>(t) * 2 * H);
    const auto * r1 = reinterpret_cast<const float4 *>(res + (static_cast<size_t>(t) * 2 + 1) * H);
    const auto * nw = reinterpret_cast<const float4 *>(norm_w);
    auto *       yn = reinterpret_cast<float4 *>(x_norm + static_cast<size_t>(t) * H);

    float ss = 0.0f;
    for (int i = tid; i < H4; i += 256) {
        const float4 a = r0[i]; const float4 b = r1[i];
        const float4 br = make_float4(pre0 * a.x + pre1 * b.x, pre0 * a.y + pre1 * b.y,
                                      pre0 * a.z + pre1 * b.z, pre0 * a.w + pre1 * b.w);
        ss += br.x * br.x + br.y * br.y + br.z * br.z + br.w * br.w;
    }
    ss = sn_warp_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? s_red[lane] : 0.0f;
        v = sn_warp_sum(v);
        if (lane == 0) s_red[0] = rsqrtf(v / static_cast<float>(H) + rms_eps);
    }
    __syncthreads();
    const float inv = s_red[0];
    for (int i = tid; i < H4; i += 256) {
        const float4 a = r0[i]; const float4 b = r1[i];
        const float4 w = __ldg(&nw[i]);
        yn[i] = make_float4(
            (pre0 * a.x + pre1 * b.x) * inv * w.x,
            (pre0 * a.y + pre1 * b.y) * inv * w.y,
            (pre0 * a.z + pre1 * b.z) * inv * w.z,
            (pre0 * a.w + pre1 * b.w) * inv * w.w);
    }
}

__global__ void sn_mhc_merge_kernel(
    const float * __restrict__ res_in,
    const float * __restrict__ branch,
    const float * __restrict__ ctrl,
    float *       __restrict__ res_out,
    int H) {

    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int t  = blockIdx.y;
    const int H4 = H >> 2;
    if (i4 >= H4) return;

    const float * ct = ctrl + static_cast<size_t>(t) * 6;
    const float p0 = ct[0], p1 = ct[1];
    const float m00 = ct[2], m01 = ct[3], m10 = ct[4], m11 = ct[5];

    const auto * r0 = reinterpret_cast<const float4 *>(res_in + static_cast<size_t>(t) * 2 * H);
    const auto * r1 = reinterpret_cast<const float4 *>(res_in + (static_cast<size_t>(t) * 2 + 1) * H);
    const auto * br = reinterpret_cast<const float4 *>(branch + static_cast<size_t>(t) * H);
    auto *       o0 = reinterpret_cast<float4 *>(res_out + static_cast<size_t>(t) * 2 * H);
    auto *       o1 = reinterpret_cast<float4 *>(res_out + (static_cast<size_t>(t) * 2 + 1) * H);

    const float4 a = r0[i4];
    const float4 b = r1[i4];
    const float4 c = br[i4];
    o0[i4] = make_float4(m00 * a.x + m01 * b.x + p0 * c.x,
                         m00 * a.y + m01 * b.y + p0 * c.y,
                         m00 * a.z + m01 * b.z + p0 * c.z,
                         m00 * a.w + m01 * b.w + p0 * c.w);
    o1[i4] = make_float4(m10 * a.x + m11 * b.x + p1 * c.x,
                         m10 * a.y + m11 * b.y + p1 * c.y,
                         m10 * a.z + m11 * b.z + p1 * c.z,
                         m10 * a.w + m11 * b.w + p1 * c.w);
}

__global__ void sn_mhc_merge_leave_kernel(
    const float * __restrict__ res_in,
    const float * __restrict__ branch,
    const float * __restrict__ ctrl,
    float *       __restrict__ cur_out,
    int H) {

    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int t  = blockIdx.y;
    const int H4 = H >> 2;
    if (i4 >= H4) return;

    const float * ct = ctrl + static_cast<size_t>(t) * 6;
    const float c0 = 0.5f * (ct[2] + ct[4]);
    const float c1 = 0.5f * (ct[3] + ct[5]);
    const float cb = 0.5f * (ct[0] + ct[1]);

    const auto * r0 = reinterpret_cast<const float4 *>(res_in + static_cast<size_t>(t) * 2 * H);
    const auto * r1 = reinterpret_cast<const float4 *>(res_in + (static_cast<size_t>(t) * 2 + 1) * H);
    const auto * br = reinterpret_cast<const float4 *>(branch + static_cast<size_t>(t) * H);
    auto *       yo = reinterpret_cast<float4 *>(cur_out + static_cast<size_t>(t) * H);

    const float4 a = r0[i4];
    const float4 b = r1[i4];
    const float4 c = br[i4];
    yo[i4] = make_float4(c0 * a.x + c1 * b.x + cb * c.x,
                         c0 * a.y + c1 * b.y + cb * c.y,
                         c0 * a.z + c1 * b.z + cb * c.z,
                         c0 * a.w + c1 * b.w + cb * c.w);
}

// ============================================================================
// 2. Strided & Chunked-Smem GDN Kernels (Prefill + Decode Fused L2/Prep!)
// ============================================================================

__global__ void sn_gdn_gate_beta_packed_kernel(
    const float * __restrict__ ba,
    const float * __restrict__ dt_bias,
    const float * __restrict__ ssm_a,
    float *       __restrict__ g_out,
    float *       __restrict__ b_out,
    int n_v_heads) {

    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (h >= n_v_heads) return;

    const float * row = ba + static_cast<size_t>(t) * (2 * n_v_heads);
    const float b_raw = row[h];
    const float a_raw = row[n_v_heads + h] + dt_bias[h];
    const float sp    = (a_raw > 20.0f) ? a_raw : log1pf(expf(a_raw));
    const size_t off  = static_cast<size_t>(t) * n_v_heads + h;
    g_out[off] = sp * ssm_a[h];
    b_out[off] = sn_sigmoid(b_raw);
}

// Decode (T <= 4) FUSED Strided Conv1D + SiLU + per-head L2-Norm(Q, K) in ONE launch!
// Grid: (n_total_heads = 2*n_k_heads + n_v_heads = 64), Block: 128 threads (1 head of 128 dims!).
__global__ void sn_conv1d_silu_l2_strided_dec_kernel(
    const float * __restrict__ x,          // [T, x_stride]
    const float * __restrict__ w,          // [C, 4]
    float *       __restrict__ conv_state, // [3, C]
    float *       __restrict__ out,        // [T, C]
    float *       __restrict__ conv_snap,  // optional [3, C]
    int T, int C, int x_stride, int n_qk_heads, float eps) {

    const int h = blockIdx.x;
    const int d = threadIdx.x; // 0..127
    const int c = h * 128 + d;
    if (c >= C) return;

    float s0 = conv_state[0 * C + c];
    float s1 = conv_state[1 * C + c];
    float s2 = conv_state[2 * C + c];
    const float w0 = __ldg(&w[c * 4 + 0]);
    const float w1 = __ldg(&w[c * 4 + 1]);
    const float w2 = __ldg(&w[c * 4 + 2]);
    const float w3 = __ldg(&w[c * 4 + 3]);

    __shared__ float s_red[4];
    const int lane = d & 31;
    const int warp = d >> 5;
    const bool do_l2 = (h < n_qk_heads);

    for (int t = 0; t < T; ++t) {
        const float xt  = x[static_cast<size_t>(t) * x_stride + c];
        const float acc = s0 * w0 + s1 * w1 + s2 * w2 + xt * w3;
        float val = sn_silu(acc);
        s0 = s1; s1 = s2; s2 = xt;
        if (t == 0 && conv_snap != nullptr) {
            conv_snap[0 * C + c] = s0;
            conv_snap[1 * C + c] = s1;
            conv_snap[2 * C + c] = s2;
        }
        if (do_l2) {
            float ss = sn_warp_sum(val * val);
            if (lane == 0) s_red[warp] = ss;
            __syncthreads();
            if (warp == 0) {
                float tot = (lane < 4) ? s_red[lane] : 0.0f;
                tot = sn_warp_sum(tot);
                if (lane == 0) s_red[0] = rsqrtf(fmaxf(tot, eps * eps));
            }
            __syncthreads();
            val *= s_red[0];
        }
        out[static_cast<size_t>(t) * C + c] = val;
    }

    conv_state[0 * C + c] = s0;
    conv_state[1 * C + c] = s1;
    conv_state[2 * C + c] = s2;
}

__global__ void sn_conv1d_silu_strided_prefill_kernel(
    const float * __restrict__ x,
    const float * __restrict__ w,
    const float * __restrict__ conv_state,
    float *       __restrict__ out,
    int T, int C, int x_stride) {

    const int c  = blockIdx.x * blockDim.x + threadIdx.x;
    const int t0 = blockIdx.y * 8;
    if (c >= C || t0 >= T) return;

    const float w0 = __ldg(&w[c * 4 + 0]);
    const float w1 = __ldg(&w[c * 4 + 1]);
    const float w2 = __ldg(&w[c * 4 + 2]);
    const float w3 = __ldg(&w[c * 4 + 3]);

    float s0 = (t0 >= 3) ? x[static_cast<size_t>(t0 - 3) * x_stride + c] : conv_state[static_cast<size_t>(t0) * C + c];
    float s1 = (t0 >= 2) ? x[static_cast<size_t>(t0 - 2) * x_stride + c] : conv_state[static_cast<size_t>(t0 + 1) * C + c];
    float s2 = (t0 >= 1) ? x[static_cast<size_t>(t0 - 1) * x_stride + c] : conv_state[static_cast<size_t>(t0 + 2) * C + c];

    const int t1 = (t0 + 8 < T) ? (t0 + 8) : T;
    for (int t = t0; t < t1; ++t) {
        const float xt  = x[static_cast<size_t>(t) * x_stride + c];
        const float acc = s0 * w0 + s1 * w1 + s2 * w2 + xt * w3;
        out[static_cast<size_t>(t) * C + c] = sn_silu(acc);
        s0 = s1; s1 = s2; s2 = xt;
    }
}

__global__ void sn_conv1d_update_strided_kernel(
    const float * __restrict__ x,
    float *       __restrict__ conv_state,
    int T, int C, int x_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    conv_state[0 * C + c] = x[static_cast<size_t>(T - 3) * x_stride + c];
    conv_state[1 * C + c] = x[static_cast<size_t>(T - 2) * x_stride + c];
    conv_state[2 * C + c] = x[static_cast<size_t>(T - 1) * x_stride + c];
}

__global__ void sn_qkv_l2_norm_d128_kernel(
    float * __restrict__ qkv_conv, int n_k_heads, int row_stride, float eps) {

    const int qk_h = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    if (qk_h >= 2 * n_k_heads) return;
    const int lane = threadIdx.x;

    auto * x4 = reinterpret_cast<float4 *>(qkv_conv + static_cast<size_t>(t) * row_stride + qk_h * 128);
    float4 v  = x4[lane];
    float ss  = sn_warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    float inv = rsqrtf(fmaxf(__shfl_sync(0xffffffffu, ss, 0), eps * eps));
    x4[lane]  = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
}

// High-throughput Chunked-Smem GDN Recurrence for Prefill (T >= 8)!
// 8 warps/CTA (8 cols of head h), cooperatively stages BC=16 tokens of Q, K, V, exp(g), beta
// in shared memory -> cuts global Q/K traffic by 8x and expf() calls by 256x!
// Grid: (n_v_heads=32, 128/8 = 16) = 512 CTAs -> 100% P100 SM saturation!
__global__ void sn_gdn_packed_d128_prefill_smem_kernel(
    const float * __restrict__ qkv_conv,
    const float * __restrict__ g,
    const float * __restrict__ beta,
    float *       __restrict__ state,
    float *       __restrict__ out,
    int T, int n_k_heads, int n_v_heads, int row_stride, float scale) {

    constexpr int BC = 16;
    const int h    = blockIdx.x;
    const int col0 = blockIdx.y * 8;
    const int warp = threadIdx.y; // 0..7
    const int lane = threadIdx.x; // 0..31
    const int tid  = warp * 32 + lane; // 0..255
    const int col  = col0 + warp;

    const int kh     = h % n_k_heads;
    const int q_base = kh * 128;
    const int k_base = (n_k_heads + kh) * 128;
    const int v_base = (2 * n_k_heads + h) * 128 + col0;

    __shared__ float s_q[BC][128];
    __shared__ float s_k[BC][128];
    __shared__ float s_v[BC][8];
    __shared__ float s_d[BC];
    __shared__ float s_b[BC];

    const size_t col_off = (static_cast<size_t>(h) * 128 + col) * 128;
    float * s_col = state + col_off;
    float s0 = s_col[lane +  0];
    float s1 = s_col[lane + 32];
    float s2 = s_col[lane + 64];
    float s3 = s_col[lane + 96];

    for (int t0 = 0; t0 < T; t0 += BC) {
        const int tc = (t0 + BC < T) ? BC : (T - t0);

        // Cooperatively load Q[tc][128] and K[tc][128] using coalesced float4 loads (512 float4s / 256 threads = 2 iters!)
        for (int idx = tid; idx < tc * 32; idx += 256) {
            const int c  = idx >> 5;
            const int l4 = idx & 31;
            const float * row = qkv_conv + static_cast<size_t>(t0 + c) * row_stride;
            reinterpret_cast<float4 *>(s_q[c])[l4] = reinterpret_cast<const float4 *>(row + q_base)[l4];
            reinterpret_cast<float4 *>(s_k[c])[l4] = reinterpret_cast<const float4 *>(row + k_base)[l4];
        }
        if (tid < tc * 8) {
            const int c = tid >> 3;
            const int j = tid & 7;
            s_v[c][j] = qkv_conv[static_cast<size_t>(t0 + c) * row_stride + v_base + j];
        }
        if (tid < tc) {
            const size_t off = static_cast<size_t>(t0 + tid) * n_v_heads + h;
            s_d[tid] = expf(g[off]);
            s_b[tid] = beta[off];
        }
        __syncthreads();

        for (int c = 0; c < tc; ++c) {
            const float d    = s_d[c];
            const float bval = s_b[c];
            const float k0 = s_k[c][lane +  0];
            const float k1 = s_k[c][lane + 32];
            const float k2 = s_k[c][lane + 64];
            const float k3 = s_k[c][lane + 96];
            const float q0 = s_q[c][lane +  0];
            const float q1 = s_q[c][lane + 32];
            const float q2 = s_q[c][lane + 64];
            const float q3 = s_q[c][lane + 96];

            s0 *= d; s1 *= d; s2 *= d; s3 *= d;
            float kv = sn_warp_sum(s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3);
            kv = __shfl_sync(0xffffffffu, kv, 0);

            const float delta = (s_v[c][warp] - kv) * bval;
            s0 += k0 * delta; s1 += k1 * delta; s2 += k2 * delta; s3 += k3 * delta;

            float attn = sn_warp_sum(s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3);
            if (lane == 0) out[(static_cast<size_t>(t0 + c) * n_v_heads + h) * 128 + col] = attn * scale;
        }
        __syncthreads();
    }

    s_col[lane +  0] = s0;
    s_col[lane + 32] = s1;
    s_col[lane + 64] = s2;
    s_col[lane + 96] = s3;
}

// Decode (T <= 4) GDN Recurrence (assumes Q, K already L2-normalized by fused Conv1D+SiLU+L2!)
__global__ void sn_gdn_packed_d128_dec_kernel(
    const float * __restrict__ qkv_conv,
    const float * __restrict__ g,
    const float * __restrict__ beta,
    float *       __restrict__ state,
    float *       __restrict__ out,
    float *       __restrict__ state_snap,
    int T, int n_k_heads, int n_v_heads, int row_stride, float scale) {

    const int h    = blockIdx.x;
    const int col  = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (col >= 128) return;

    const int kh     = h % n_k_heads;
    const int q_base = kh * 128;
    const int k_base = (n_k_heads + kh) * 128;
    const int v_base = (2 * n_k_heads + h) * 128;

    const size_t col_off = (static_cast<size_t>(h) * 128 + col) * 128;
    float * s_col = state + col_off;
    float s0 = s_col[lane +  0];
    float s1 = s_col[lane + 32];
    float s2 = s_col[lane + 64];
    float s3 = s_col[lane + 96];

    for (int t = 0; t < T; ++t) {
        const float * row = qkv_conv + static_cast<size_t>(t) * row_stride;
        const float * qt  = row + q_base;
        const float * kt  = row + k_base;
        const float * vt  = row + v_base;

        const float k0 = __ldg(&kt[lane +  0]);
        const float k1 = __ldg(&kt[lane + 32]);
        const float k2 = __ldg(&kt[lane + 64]);
        const float k3 = __ldg(&kt[lane + 96]);
        const float q0 = __ldg(&qt[lane +  0]);
        const float q1 = __ldg(&qt[lane + 32]);
        const float q2 = __ldg(&qt[lane + 64]);
        const float q3 = __ldg(&qt[lane + 96]);
        const float bval = __ldg(&beta[static_cast<size_t>(t) * n_v_heads + h]);
        const float d    = expf(__ldg(&g[static_cast<size_t>(t) * n_v_heads + h]));

        s0 *= d; s1 *= d; s2 *= d; s3 *= d;
        float kv = sn_warp_sum(s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3);
        kv = __shfl_sync(0xffffffffu, kv, 0);

        const float delta = (__ldg(&vt[col]) - kv) * bval;
        s0 += k0 * delta; s1 += k1 * delta; s2 += k2 * delta; s3 += k3 * delta;

        float attn = sn_warp_sum(s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3);
        if (lane == 0) out[(static_cast<size_t>(t) * n_v_heads + h) * 128 + col] = attn * scale;

        if (t == 0 && state_snap != nullptr) {
            float * sn = state_snap + col_off;
            sn[lane +  0] = s0; sn[lane + 32] = s1; sn[lane + 64] = s2; sn[lane + 96] = s3;
        }
    }

    s_col[lane +  0] = s0;
    s_col[lane + 32] = s1;
    s_col[lane + 64] = s2;
    s_col[lane + 96] = s3;
}

// Decode (T <= 4): Strided RMSNorm(o) * SiLU(z) + DIRECT Activation Prep into d_xprep!
// Each warp owns 1 head = 128 dims = 4 contiguous 32-elem blocks -> ZERO extra prep launch!
template <int PREP_MODE>
__global__ void sn_rms_norm_silu_gate_prep_kernel(
    float *       __restrict__ o,
    const float * __restrict__ norm_w,
    const float * __restrict__ z_base,
    void *        __restrict__ d_xprep,
    int T, int n_v_heads, int z_stride, int z_offset, float eps) {

    const int h = blockIdx.x * blockDim.y + threadIdx.y;
    const int t = blockIdx.y;
    if (h >= n_v_heads) return;

    const int lane = threadIdx.x;
    const size_t o_off = (static_cast<size_t>(t) * n_v_heads + h) * 128;
    const size_t z_off = static_cast<size_t>(t) * z_stride + z_offset + h * 128;

    float v[4];
    float ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d = k * 32 + lane;
        const float ov = o[o_off + d];
        v[k] = ov;
        ss += ov * ov;
    }
    const float inv = rsqrtf(__shfl_sync(0xffffffffu, sn_warp_sum(ss), 0) * (1.0f / 128.0f) + eps);
    const int H = n_v_heads * 128;
    const int K_blocks = H >> 5;
    const int total_h2_bytes = static_cast<int>((static_cast<size_t>(T) * H * sizeof(__half) + 255) & ~size_t(255));

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d = k * 32 + lane;
        const float val = v[k] * inv * __ldg(&norm_w[d]) * sn_silu(z_base[z_off + d]);
        if constexpr (PREP_MODE == 0) {
            o[o_off + d] = val;
        } else {
            sn_store_prepped_block32<PREP_MODE>(val, d_xprep, t * K_blocks + h * 4 + k, total_h2_bytes);
        }
    }
}

// Decode (T <= 2): Fused Sigmoid Gate Mul (attn_out * sigmoid(gate)) + DIRECT Activation Prep!
template <int PREP_MODE>
__global__ void sn_sigmoid_gate_mul_prep_kernel(
    const float * __restrict__ a,
    const float * __restrict__ g,
    void *        __restrict__ d_xprep,
    int H, int T) {

    const int K_blocks = H >> 5;
    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    const int t = blockIdx.y;
    if (b >= K_blocks) return;

    const int lane = threadIdx.x;
    const size_t off = static_cast<size_t>(t) * H + b * 32 + lane;
    const float val  = a[off] * sn_sigmoid(g[off]);
    const int total_h2_bytes = static_cast<int>((static_cast<size_t>(T) * H * sizeof(__half) + 255) & ~size_t(255));
    sn_store_prepped_block32<PREP_MODE>(val, d_xprep, t * K_blocks + b, total_h2_bytes);
}

// ============================================================================
// 3. Exact YaRN Partial NeoX RoPE (rope_dim = 64, head_dim = 256)
// ============================================================================

__global__ void sn_qgate_split_norm_yarn_rope_d256_kernel(
    const float *   __restrict__ q_full,
    const float *   __restrict__ q_norm_w,
    const int32_t * __restrict__ pos,
    const float *   __restrict__ inv_freq,
    float *         __restrict__ q_out,
    float *         __restrict__ gate_out,
    int n_head, int rope_dim, float rms_eps, float yarn_mscale) {

    const int h = blockIdx.x * blockDim.y + threadIdx.y;
    const int t = blockIdx.y;
    if (h >= n_head) return;
    const int lane = threadIdx.x;

    const float * src_q = q_full + (static_cast<size_t>(t) * n_head + h) * 512;
    const float * src_g = src_q + 256;
    float * dst_q = q_out    + (static_cast<size_t>(t) * n_head + h) * 256;
    float * dst_g = gate_out + (static_cast<size_t>(t) * n_head + h) * 256;

    float qv[8];
    float ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int d = k * 32 + lane;
        qv[k] = src_q[d];
        ss += qv[k] * qv[k];
        dst_g[d] = src_g[d];
    }
    ss = __shfl_sync(0xffffffffu, sn_warp_sum(ss), 0);
    const float inv = rsqrtf(ss * (1.0f / 256.0f) + rms_eps);

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int d = k * 32 + lane;
        qv[k] *= inv * __ldg(&q_norm_w[d]);
    }

    const int half_r = rope_dim >> 1;
    if (lane < half_r) {
        const float angle = static_cast<float>(pos[t]) * __ldg(&inv_freq[lane]);
        float s, c;
        __sincosf(angle, &s, &c);
        c *= yarn_mscale;
        s *= yarn_mscale;
        const float x0 = qv[0];
        const float x1 = qv[1];
        qv[0] = x0 * c - x1 * s;
        qv[1] = x0 * s + x1 * c;
    }

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        dst_q[k * 32 + lane] = qv[k];
    }
}

__global__ void sn_k_norm_yarn_rope_d256_kernel(
    float *         __restrict__ k_buf,
    const float *   __restrict__ k_norm_w,
    const int32_t * __restrict__ pos,
    const float *   __restrict__ inv_freq,
    int n_kv, int rope_dim, float rms_eps, float yarn_mscale) {

    const int h = blockIdx.x * blockDim.y + threadIdx.y;
    const int t = blockIdx.y;
    if (h >= n_kv) return;
    const int lane = threadIdx.x;

    float * row = k_buf + (static_cast<size_t>(t) * n_kv + h) * 256;
    float kv[8];
    float ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int d = k * 32 + lane;
        kv[k] = row[d];
        ss += kv[k] * kv[k];
    }
    ss = __shfl_sync(0xffffffffu, sn_warp_sum(ss), 0);
    const float inv = rsqrtf(ss * (1.0f / 256.0f) + rms_eps);
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int d = k * 32 + lane;
        kv[k] *= inv * __ldg(&k_norm_w[d]);
    }

    const int half_r = rope_dim >> 1;
    if (lane < half_r) {
        const float angle = static_cast<float>(pos[t]) * __ldg(&inv_freq[lane]);
        float s, c;
        __sincosf(angle, &s, &c);
        c *= yarn_mscale;
        s *= yarn_mscale;
        const float x0 = kv[0];
        const float x1 = kv[1];
        kv[0] = x0 * c - x1 * s;
        kv[1] = x0 * s + x1 * c;
    }

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        row[k * 32 + lane] = kv[k];
    }
}

// ============================================================================
// 4. Pascal-Optimized sm_60 (Smem-Cached FP16x2!) & sm_61+ (dp4a) Q8_0 Kernels
//    Final tuned:
//    - Smem-cached Q8_0 SoA GEMV with rpw=2, wpb=8 (T=1 and T=2 weight-reuse!)
//      Beats zero-smem by -2.7 ms on T=2 MTP and -0.26 ms on gdn_o/gqa_o!
//    - Smem-cached 8-warp SwiGLU Gate+Up in 2 sequential passes of 2 rows -> ZERO register spill!
// ============================================================================

template <int ROWS_PER_WARP, bool IS_T2>
__global__ void sn_gemv_q8_0_sm60_smem_kernel(
    const int4 *     __restrict__ W_qs,
    const uint16_t * __restrict__ W_sc,
    const int4 *     __restrict__ X_h2,
    const float *    __restrict__ X_sc,
    float *          __restrict__ Y,
    int K_blocks, int N) {

    extern __shared__ uint8_t s_raw[];
    constexpr int T = IS_T2 ? 2 : 1;
    int4 *  s_xh2 = reinterpret_cast<int4 *>(s_raw);
    float * s_xsc = reinterpret_cast<float *>(s_raw + static_cast<size_t>(T) * K_blocks * 64);

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const int tot4 = T * K_blocks * 4;
    for (int i = tid; i < tot4; i += nthr) s_xh2[i] = X_h2[i];
    const int tots = T * K_blocks;
    for (int i = tid; i < tots; i += nthr) s_xsc[i] = X_sc[i];
    __syncthreads();

    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * ROWS_PER_WARP;
    if (row0 >= N) return;

    const int lane        = threadIdx.x;
    const int blk_in_warp = lane >> 1;
    const int half_in_blk = lane & 1;

    const int4 *  s_x0 = s_xh2;
    const float * s_c0 = s_xsc;
    const int4 *  s_x1 = s_xh2 + K_blocks * 4;
    const float * s_c1 = s_xsc + K_blocks;

    float acc0[ROWS_PER_WARP];
    float acc1[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }

    for (int b = blk_in_warp; b < K_blocks; b += 16) {
        const int4 x0a = s_x0[b * 4 + half_in_blk * 2 + 0];
        const int4 x0b = s_x0[b * 4 + half_in_blk * 2 + 1];
        const float c0 = s_c0[b];

        int4 x1a{}, x1b{};
        float c1 = 0.0f;
        if constexpr (IS_T2) {
            x1a = s_x1[b * 4 + half_in_blk * 2 + 0];
            x1b = s_x1[b * 4 + half_in_blk * 2 + 1];
            c1  = s_c1[b];
        }

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int row = row0 + r;
            if (row < N) {
                const int4 w4 = __ldg(&W_qs[(static_cast<size_t>(row) * K_blocks + b) * 2 + half_in_blk]);
                __half wd_h;
                *reinterpret_cast<uint16_t *>(&wd_h) = __ldg(&W_sc[static_cast<size_t>(row) * K_blocks + b]);
                const float wd = __half2float(wd_h);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
                __half2 h0, h1, h2, h3, h4, h5, h6, h7;
                sn_i8x4_to_h2x2(w4.x, h0, h1);
                sn_i8x4_to_h2x2(w4.y, h2, h3);
                sn_i8x4_to_h2x2(w4.z, h4, h5);
                sn_i8x4_to_h2x2(w4.w, h6, h7);

                __half2 p0 = __hmul2(h0, *reinterpret_cast<const __half2 *>(&x0a.x));
                p0 = __hfma2(h1, *reinterpret_cast<const __half2 *>(&x0a.y), p0);
                p0 = __hfma2(h2, *reinterpret_cast<const __half2 *>(&x0a.z), p0);
                p0 = __hfma2(h3, *reinterpret_cast<const __half2 *>(&x0a.w), p0);
                p0 = __hfma2(h4, *reinterpret_cast<const __half2 *>(&x0b.x), p0);
                p0 = __hfma2(h5, *reinterpret_cast<const __half2 *>(&x0b.y), p0);
                p0 = __hfma2(h6, *reinterpret_cast<const __half2 *>(&x0b.z), p0);
                p0 = __hfma2(h7, *reinterpret_cast<const __half2 *>(&x0b.w), p0);
                const float2 f0 = __half22float2(p0);
                acc0[r] += (wd * c0) * (f0.x + f0.y);

                if constexpr (IS_T2) {
                    __half2 p1 = __hmul2(h0, *reinterpret_cast<const __half2 *>(&x1a.x));
                    p1 = __hfma2(h1, *reinterpret_cast<const __half2 *>(&x1a.y), p1);
                    p1 = __hfma2(h2, *reinterpret_cast<const __half2 *>(&x1a.z), p1);
                    p1 = __hfma2(h3, *reinterpret_cast<const __half2 *>(&x1a.w), p1);
                    p1 = __hfma2(h4, *reinterpret_cast<const __half2 *>(&x1b.x), p1);
                    p1 = __hfma2(h5, *reinterpret_cast<const __half2 *>(&x1b.y), p1);
                    p1 = __hfma2(h6, *reinterpret_cast<const __half2 *>(&x1b.z), p1);
                    p1 = __hfma2(h7, *reinterpret_cast<const __half2 *>(&x1b.w), p1);
                    const float2 f1 = __half22float2(p1);
                    acc1[r] += (wd * c1) * (f1.x + f1.y);
                }
#else
                auto dot_blk = [&w4](const int4 & xa, const int4 & xb) {
                    auto dot4 = [](int wi, uint32_t a, uint32_t b) {
                        const float2 fa = __half22float2(*reinterpret_cast<const __half2 *>(&a));
                        const float2 fb = __half22float2(*reinterpret_cast<const __half2 *>(&b));
                        return static_cast<float>(static_cast<int8_t>((wi >>  0) & 0xFF)) * fa.x
                             + static_cast<float>(static_cast<int8_t>((wi >>  8) & 0xFF)) * fa.y
                             + static_cast<float>(static_cast<int8_t>((wi >> 16) & 0xFF)) * fb.x
                             + static_cast<float>(static_cast<int8_t>((wi >> 24) & 0xFF)) * fb.y;
                    };
                    return dot4(w4.x, xa.x, xa.y) + dot4(w4.y, xa.z, xa.w)
                         + dot4(w4.z, xb.x, xb.y) + dot4(w4.w, xb.z, xb.w);
                };
                acc0[r] += (wd * c0) * dot_blk(x0a, x0b);
                if constexpr (IS_T2) acc1[r] += (wd * c1) * dot_blk(x1a, x1b);
#endif
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const float s0 = sn_warp_sum(acc0[r]);
        if (lane == 0 && row0 + r < N) Y[row0 + r] = s0;
        if constexpr (IS_T2) {
            const float s1 = sn_warp_sum(acc1[r]);
            if (lane == 0 && row0 + r < N) Y[N + row0 + r] = s1;
        }
    }
}

// FUSED Dense SwiGLU Gate + Up + SiLU + In-Kernel Prep (sm_60 HFMA2, 16-warp 1-pass smem X_h2!)
template <bool IS_T2>
__global__ void sn_swiglu_q8_0_sm60_fused_prep_kernel(
    const int4 *     __restrict__ G_qs,
    const uint16_t * __restrict__ G_sc,
    const int4 *     __restrict__ U_qs,
    const uint16_t * __restrict__ U_sc,
    const int4 *     __restrict__ X_h2,
    const float *    __restrict__ X_sc,
    __half2 *        __restrict__ out_h2,
    float *          __restrict__ out_sc,
    int K_blocks, int ff) {

    extern __shared__ uint8_t s_raw[];
    constexpr int T = IS_T2 ? 2 : 1;
    int4 *  s_xh2 = reinterpret_cast<int4 *>(s_raw);
    float * s_xsc = reinterpret_cast<float *>(s_raw + static_cast<size_t>(T) * K_blocks * 64);

    __shared__ float s_act[T][32];

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int warp = threadIdx.y; // 0..15
    const int lane = threadIdx.x; // 0..31
    const int tot4 = T * K_blocks * 4;
    for (int i = tid; i < tot4; i += 512) s_xh2[i] = X_h2[i];
    const int tots = T * K_blocks;
    for (int i = tid; i < tots; i += 512) s_xsc[i] = X_sc[i];
    __syncthreads();

    const int blk_ff = blockIdx.x;
    const int row0   = blk_ff * 32 + warp * 2;
    const int blk_in_warp = lane >> 1;
    const int half_in_blk = lane & 1;

    const int4 *  s_x0 = s_xh2;
    const float * s_c0 = s_xsc;
    const int4 *  s_x1 = s_xh2 + K_blocks * 4;
    const float * s_c1 = s_xsc + K_blocks;

    float g0[2] = {0, 0}, u0[2] = {0, 0};
    float g1[2] = {0, 0}, u1[2] = {0, 0};

    for (int b = blk_in_warp; b < K_blocks; b += 16) {
        const int4 x0a = s_x0[b * 4 + half_in_blk * 2 + 0];
        const int4 x0b = s_x0[b * 4 + half_in_blk * 2 + 1];
        const float c0 = s_c0[b];

        int4 x1a{}, x1b{};
        float c1 = 0.0f;
        if constexpr (IS_T2) {
            x1a = s_x1[b * 4 + half_in_blk * 2 + 0];
            x1b = s_x1[b * 4 + half_in_blk * 2 + 1];
            c1  = s_c1[b];
        }

        auto accum_mat = [&](const int4 * Wq, const uint16_t * Ws, float a0[2], float a1[2]) {
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = row0 + r;
                const int4 w4 = __ldg(&Wq[(static_cast<size_t>(row) * K_blocks + b) * 2 + half_in_blk]);
                __half wd_h;
                *reinterpret_cast<uint16_t *>(&wd_h) = __ldg(&Ws[static_cast<size_t>(row) * K_blocks + b]);
                const float wd = __half2float(wd_h);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
                __half2 h0, h1, h2, h3, h4, h5, h6, h7;
                sn_i8x4_to_h2x2(w4.x, h0, h1);
                sn_i8x4_to_h2x2(w4.y, h2, h3);
                sn_i8x4_to_h2x2(w4.z, h4, h5);
                sn_i8x4_to_h2x2(w4.w, h6, h7);

                __half2 p0 = __hmul2(h0, *reinterpret_cast<const __half2 *>(&x0a.x));
                p0 = __hfma2(h1, *reinterpret_cast<const __half2 *>(&x0a.y), p0);
                p0 = __hfma2(h2, *reinterpret_cast<const __half2 *>(&x0a.z), p0);
                p0 = __hfma2(h3, *reinterpret_cast<const __half2 *>(&x0a.w), p0);
                p0 = __hfma2(h4, *reinterpret_cast<const __half2 *>(&x0b.x), p0);
                p0 = __hfma2(h5, *reinterpret_cast<const __half2 *>(&x0b.y), p0);
                p0 = __hfma2(h6, *reinterpret_cast<const __half2 *>(&x0b.z), p0);
                p0 = __hfma2(h7, *reinterpret_cast<const __half2 *>(&x0b.w), p0);
                const float2 f0 = __half22float2(p0);
                a0[r] += (wd * c0) * (f0.x + f0.y);

                if constexpr (IS_T2) {
                    __half2 p1 = __hmul2(h0, *reinterpret_cast<const __half2 *>(&x1a.x));
                    p1 = __hfma2(h1, *reinterpret_cast<const __half2 *>(&x1a.y), p1);
                    p1 = __hfma2(h2, *reinterpret_cast<const __half2 *>(&x1a.z), p1);
                    p1 = __hfma2(h3, *reinterpret_cast<const __half2 *>(&x1a.w), p1);
                    p1 = __hfma2(h4, *reinterpret_cast<const __half2 *>(&x1b.x), p1);
                    p1 = __hfma2(h5, *reinterpret_cast<const __half2 *>(&x1b.y), p1);
                    p1 = __hfma2(h6, *reinterpret_cast<const __half2 *>(&x1b.z), p1);
                    p1 = __hfma2(h7, *reinterpret_cast<const __half2 *>(&x1b.w), p1);
                    const float2 f1 = __half22float2(p1);
                    a1[r] += (wd * c1) * (f1.x + f1.y);
                }
#else
                auto dot_blk = [&w4](const int4 & xa, const int4 & xb) {
                    auto dot4 = [](int wi, uint32_t a, uint32_t b) {
                        const float2 fa = __half22float2(*reinterpret_cast<const __half2 *>(&a));
                        const float2 fb = __half22float2(*reinterpret_cast<const __half2 *>(&b));
                        return static_cast<float>(static_cast<int8_t>((wi >>  0) & 0xFF)) * fa.x
                             + static_cast<float>(static_cast<int8_t>((wi >>  8) & 0xFF)) * fa.y
                             + static_cast<float>(static_cast<int8_t>((wi >> 16) & 0xFF)) * fb.x
                             + static_cast<float>(static_cast<int8_t>((wi >> 24) & 0xFF)) * fb.y;
                    };
                    return dot4(w4.x, xa.x, xa.y) + dot4(w4.y, xa.z, xa.w)
                         + dot4(w4.z, xb.x, xb.y) + dot4(w4.w, xb.z, xb.w);
                };
                a0[r] += (wd * c0) * dot_blk(x0a, x0b);
                if constexpr (IS_T2) a1[r] += (wd * c1) * dot_blk(x1a, x1b);
#endif
            }
        };
        accum_mat(G_qs, G_sc, g0, g1);
        accum_mat(U_qs, U_sc, u0, u1);
    }

    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float vg0 = sn_warp_sum(g0[r]);
        const float vu0 = sn_warp_sum(u0[r]);
        if (lane == 0) s_act[0][warp * 2 + r] = sn_silu(vg0) * vu0;
        if constexpr (IS_T2) {
            const float vg1 = sn_warp_sum(g1[r]);
            const float vu1 = sn_warp_sum(u1[r]);
            if (lane == 0) s_act[1][warp * 2 + r] = sn_silu(vg1) * vu1;
        }
    }
    __syncthreads();

    if (warp < T) {
        const int ff_blocks = ff >> 5;
        const int b_out     = warp * ff_blocks + blk_ff;
        const float v    = s_act[warp][lane];
        const float amax = __shfl_sync(0xffffffffu, sn_warp_max(fabsf(v)), 0);
        const float inv  = (amax > 0.0f) ? (8.0f / amax) : 0.0f;
        const float vn   = v * inv;
        const float vp   = __shfl_xor_sync(0xffffffffu, vn, 1);
        if ((lane & 1) == 0) {
            out_h2[static_cast<size_t>(b_out) * 16 + (lane >> 1)] = __float22half2_rn(make_float2(vn, vp));
        }
        if (lane == 0) out_sc[b_out] = amax * 0.125f;
    }
}

template <bool IS_T2>
__global__ void sn_swiglu_q8_0_dp4a_fused_q81_kernel(
    const int4 *          __restrict__ G_qs,
    const uint16_t *      __restrict__ G_sc,
    const int4 *          __restrict__ U_qs,
    const uint16_t *      __restrict__ U_sc,
    const BlockQ8_1_Act * __restrict__ X,
    BlockQ8_1_Act *       __restrict__ out_q81,
    int K_blocks, int ff) {

    extern __shared__ BlockQ8_1_Act s_x[];
    constexpr int T = IS_T2 ? 2 : 1;
    __shared__ float s_act[T][32];

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;

    const int n_i4 = (T * K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const auto * sx_in  = reinterpret_cast<const int4 *>(X);
    auto *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += 256) sx_out[i] = sx_in[i];
    __syncthreads();

    const int blk_ff = blockIdx.x;
    const int row0   = blk_ff * 32 + warp * 4;
    const int blk_in_warp = lane >> 1;
    const int half_in_blk = lane & 1;

    const BlockQ8_1_Act * s_x0 = s_x;
    const BlockQ8_1_Act * s_x1 = s_x + K_blocks;

    float g0[4] = {0, 0, 0, 0}, u0[4] = {0, 0, 0, 0};
    float g1[4] = {0, 0, 0, 0}, u1[4] = {0, 0, 0, 0};

    for (int b = blk_in_warp; b < K_blocks; b += 16) {
        const int4 x0 = reinterpret_cast<const int4 *>(s_x0[b].qs)[half_in_blk];
        const float c0 = s_x0[b].d;
        int4 x1{}; float c1 = 0.0f;
        if constexpr (IS_T2) {
            x1 = reinterpret_cast<const int4 *>(s_x1[b].qs)[half_in_blk];
            c1 = s_x1[b].d;
        }
        auto accum_mat = [&](const int4 * Wq, const uint16_t * Ws, float a0[4], float a1[4]) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row0 + r;
                const int4 w4 = __ldg(&Wq[(static_cast<size_t>(row) * K_blocks + b) * 2 + half_in_blk]);
                __half wd_h;
                *reinterpret_cast<uint16_t *>(&wd_h) = __ldg(&Ws[static_cast<size_t>(row) * K_blocks + b]);
                const float wd = __half2float(wd_h);
                int s0 = dp4a_s32(w4.x, x0.x, 0);
                s0     = dp4a_s32(w4.y, x0.y, s0);
                s0     = dp4a_s32(w4.z, x0.z, s0);
                s0     = dp4a_s32(w4.w, x0.w, s0);
                a0[r] += (wd * c0) * static_cast<float>(s0);
                if constexpr (IS_T2) {
                    int s1 = dp4a_s32(w4.x, x1.x, 0);
                    s1     = dp4a_s32(w4.y, x1.y, s1);
                    s1     = dp4a_s32(w4.z, x1.z, s1);
                    s1     = dp4a_s32(w4.w, x1.w, s1);
                    a1[r] += (wd * c1) * static_cast<float>(s1);
                }
            }
        };
        accum_mat(G_qs, G_sc, g0, g1);
        accum_mat(U_qs, U_sc, u0, u1);
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float vg0 = sn_warp_sum(g0[r]);
        const float vu0 = sn_warp_sum(u0[r]);
        if (lane == 0) s_act[0][warp * 4 + r] = sn_silu(vg0) * vu0;
        if constexpr (IS_T2) {
            const float vg1 = sn_warp_sum(g1[r]);
            const float vu1 = sn_warp_sum(u1[r]);
            if (lane == 0) s_act[1][warp * 4 + r] = sn_silu(vg1) * vu1;
        }
    }
    __syncthreads();

    if (warp < T) {
        const int ff_blocks = ff >> 5;
        const int b_out     = warp * ff_blocks + blk_ff;
        const float v    = s_act[warp][lane];
        const float amax = __shfl_sync(0xffffffffu, sn_warp_max(fabsf(v)), 0);
        const float d    = amax / 127.0f;
        const float inv  = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        const int   q    = __float2int_rn(v * inv);
        out_q81[b_out].qs[lane] = static_cast<int8_t>(q);
        int hsum = q;
        hsum += __shfl_xor_sync(0xffffffffu, hsum, 1);
        hsum += __shfl_xor_sync(0xffffffffu, hsum, 2);
        hsum += __shfl_xor_sync(0xffffffffu, hsum, 4);
        hsum += __shfl_xor_sync(0xffffffffu, hsum, 8);
        const int s0 = __shfl_sync(0xffffffffu, hsum, 0);
        const int s1 = __shfl_sync(0xffffffffu, hsum, 16);
        if (lane == 0) {
            out_q81[b_out].d     = d;
            out_q81[b_out].s0    = s0;
            out_q81[b_out].s1    = s1;
            out_q81[b_out].s_all = d * static_cast<float>(s0 + s1);
        }
    }
}

// ============================================================================
// 5. Pure-FP16 Prefill Pipeline on sm_60 (Q8_0 -> F16 + cublasHgemm + F16 SwiGLU!)
// ============================================================================

__global__ void sn_dequant_q8_0_to_f16_kernel(
    const int4 *     __restrict__ src_qs,
    const uint16_t * __restrict__ src_sc,
    __half *         __restrict__ dst,
    int64_t n_blocks) {

    const int lane = threadIdx.x;
    const int64_t b = (static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y) * 16 + (lane >> 1);
    if (b >= n_blocks) return;
    const int half = lane & 1;

    const int4 q4 = __ldg(&src_qs[b * 2 + half]);
    __half dh;
    *reinterpret_cast<uint16_t *>(&dh) = __ldg(&src_sc[b]);

    uint4 out0, out1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    const __half2 dh2 = __half2half2(dh);
    __half2 h0, h1, h2, h3, h4, h5, h6, h7;
    sn_i8x4_to_h2x2(q4.x, h0, h1);
    sn_i8x4_to_h2x2(q4.y, h2, h3);
    sn_i8x4_to_h2x2(q4.z, h4, h5);
    sn_i8x4_to_h2x2(q4.w, h6, h7);
    h0 = __hmul2(h0, dh2); h1 = __hmul2(h1, dh2);
    h2 = __hmul2(h2, dh2); h3 = __hmul2(h3, dh2);
    h4 = __hmul2(h4, dh2); h5 = __hmul2(h5, dh2);
    h6 = __hmul2(h6, dh2); h7 = __hmul2(h7, dh2);
    out0.x = *reinterpret_cast<const uint32_t *>(&h0);
    out0.y = *reinterpret_cast<const uint32_t *>(&h1);
    out0.z = *reinterpret_cast<const uint32_t *>(&h2);
    out0.w = *reinterpret_cast<const uint32_t *>(&h3);
    out1.x = *reinterpret_cast<const uint32_t *>(&h4);
    out1.y = *reinterpret_cast<const uint32_t *>(&h5);
    out1.z = *reinterpret_cast<const uint32_t *>(&h6);
    out1.w = *reinterpret_cast<const uint32_t *>(&h7);
#else
    const float d = __half2float(dh);
    auto pack2 = [d](int w, uint32_t & lo, uint32_t & hi) {
        const float f0 = d * static_cast<float>(static_cast<int8_t>((w >>  0) & 0xFF));
        const float f1 = d * static_cast<float>(static_cast<int8_t>((w >>  8) & 0xFF));
        const float f2 = d * static_cast<float>(static_cast<int8_t>((w >> 16) & 0xFF));
        const float f3 = d * static_cast<float>(static_cast<int8_t>((w >> 24) & 0xFF));
        const __half2 h0 = __float22half2_rn(make_float2(f0, f1));
        const __half2 h1 = __float22half2_rn(make_float2(f2, f3));
        lo = *reinterpret_cast<const uint32_t *>(&h0);
        hi = *reinterpret_cast<const uint32_t *>(&h1);
    };
    pack2(q4.x, out0.x, out0.y);
    pack2(q4.y, out0.z, out0.w);
    pack2(q4.z, out1.x, out1.y);
    pack2(q4.w, out1.z, out1.w);
#endif
    uint4 * dst4 = reinterpret_cast<uint4 *>(dst) + (b * 4 + half * 2);
    dst4[0] = out0;
    dst4[1] = out1;
}

__global__ void sn_f32_to_f16_vec_kernel(const float * __restrict__ src, __half * __restrict__ dst, int64_t n) {
    const int64_t i4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t idx = i4 * 4;
    if (idx + 3 < n) {
        const float4 v = reinterpret_cast<const float4 *>(src)[i4];
        const __half2 h0 = __float22half2_rn(make_float2(v.x, v.y));
        const __half2 h1 = __float22half2_rn(make_float2(v.z, v.w));
        uint2 pk;
        pk.x = *reinterpret_cast<const uint32_t *>(&h0);
        pk.y = *reinterpret_cast<const uint32_t *>(&h1);
        reinterpret_cast<uint2 *>(dst)[i4] = pk;
    } else {
        for (int64_t i = idx; i < n; ++i) dst[i] = __float2half_rn(src[i]);
    }
}

__global__ void sn_f16_to_f32_vec_kernel(const __half * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t i4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t idx = i4 * 4;
    if (idx + 3 < n) {
        const uint2 pk = reinterpret_cast<const uint2 *>(src)[i4];
        const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.x));
        const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.y));
        reinterpret_cast<float4 *>(dst)[i4] = make_float4(f0.x, f0.y, f1.x, f1.y);
    } else {
        for (int64_t i = idx; i < n; ++i) dst[i] = __half2float(src[i]);
    }
}

__global__ void sn_f16_to_f32_strided_kernel(
    const __half * __restrict__ src, float * __restrict__ dst, int cn, int N, int n0, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c < cn && t < T) {
        dst[static_cast<size_t>(t) * N + n0 + c] = __half2float(__ldg(&src[static_cast<size_t>(t) * cn + c]));
    }
}

// Vectorized FP16 SwiGLU: gate[i] = half(silu(float(gate[i])) * float(up[i]))
__global__ void sn_silu_mul_f16_inplace_kernel(
    __half * __restrict__ gate, const __half * __restrict__ up, int64_t n) {
    const int64_t i4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t idx = i4 * 4;
    if (idx + 3 < n) {
        const uint2 gk = reinterpret_cast<const uint2 *>(gate)[i4];
        const uint2 uk = reinterpret_cast<const uint2 *>(up)[i4];
        const float2 g0 = __half22float2(*reinterpret_cast<const __half2 *>(&gk.x));
        const float2 g1 = __half22float2(*reinterpret_cast<const __half2 *>(&gk.y));
        const float2 u0 = __half22float2(*reinterpret_cast<const __half2 *>(&uk.x));
        const float2 u1 = __half22float2(*reinterpret_cast<const __half2 *>(&uk.y));
        const __half2 r0 = __float22half2_rn(make_float2(sn_silu(g0.x) * u0.x, sn_silu(g0.y) * u0.y));
        const __half2 r1 = __float22half2_rn(make_float2(sn_silu(g1.x) * u1.x, sn_silu(g1.y) * u1.y));
        uint2 out;
        out.x = *reinterpret_cast<const uint32_t *>(&r0);
        out.y = *reinterpret_cast<const uint32_t *>(&r1);
        reinterpret_cast<uint2 *>(gate)[i4] = out;
    } else {
        for (int64_t i = idx; i < n; ++i) {
            gate[i] = __float2half_rn(sn_silu(__half2float(gate[i])) * __half2float(up[i]));
        }
    }
}

} // namespace

void launch_sensenova_mhc_enter(
    const float * d_src, float * d_dst, int H, int T, cudaStream_t stream) {
    const int H4 = H >> 2;
    sn_mhc_enter_kernel<<<dim3((H4 + 255) / 256, T), 256, 0, stream>>>(d_src, d_dst, H);
}

void launch_sensenova_mhc_pre_rmsnorm(
    cublasHandle_t cublas,
    const float *  d_res,
    const float *  d_w_hc,
    const float *  d_b_hc,
    const float *  d_norm_w,
    float *        d_ctrl,
    float *        d_x_norm,
    void *         d_scratch,
    int            H,
    int            T,
    int            sinkhorn_iters,
    bool           fixed_identity,
    float          pre_scale,
    float          rms_eps,
    bool           has_precomputed_part,
    cudaStream_t   stream,
    void *         d_xprep_out,
    int            prep_mode) {

    const int H2 = 2 * H;
    if (T <= 4) {
        constexpr int S = 16;
        auto * d_part = static_cast<float *>(d_scratch);
        if (!has_precomputed_part) {
            sn_mhc_router_part_kernel<S><<<dim3(S, T), 256, 0, stream>>>(d_res, d_w_hc, d_part, H2);
        }
        if (d_xprep_out != nullptr && prep_mode == 1) {
            sn_mhc_reduce_agg_rmsnorm_prep_kernel<S, 1><<<T, 256, 0, stream>>>(
                d_res, d_part, d_b_hc, d_norm_w, d_ctrl, d_x_norm, d_xprep_out,
                H, T, sinkhorn_iters, fixed_identity, pre_scale, rms_eps);
        } else if (d_xprep_out != nullptr && prep_mode == 2) {
            sn_mhc_reduce_agg_rmsnorm_prep_kernel<S, 2><<<T, 256, 0, stream>>>(
                d_res, d_part, d_b_hc, d_norm_w, d_ctrl, d_x_norm, d_xprep_out,
                H, T, sinkhorn_iters, fixed_identity, pre_scale, rms_eps);
        } else {
            sn_mhc_reduce_agg_rmsnorm_prep_kernel<S, 0><<<T, 256, 0, stream>>>(
                d_res, d_part, d_b_hc, d_norm_w, d_ctrl, d_x_norm, nullptr,
                H, T, sinkhorn_iters, fixed_identity, pre_scale, rms_eps);
        }
        return;
    }

    auto * d_flat = static_cast<float *>(d_scratch);
    const size_t flat_b = (static_cast<size_t>(T) * H2 * sizeof(float) + 255) & ~size_t(255);
    auto * d_logits = reinterpret_cast<float *>(static_cast<uint8_t *>(d_scratch) + flat_b);

    sn_flat_rmsnorm_kernel<<<T, 256, 0, stream>>>(d_res, d_flat, H2);
    const float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(cublas, stream);
    cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                8, T, H2,
                &alpha,
                d_w_hc, H2,
                d_flat, H2,
                &beta,
                d_logits, 8);
    sn_mhc_logits_agg_rmsnorm_kernel<<<T, 256, 0, stream>>>(
        d_res, d_logits, d_b_hc, d_norm_w, d_ctrl, d_x_norm,
        H, sinkhorn_iters, fixed_identity, pre_scale, rms_eps);
}

void launch_sensenova_rmsnorm_prep(
    float * d_cur, const float * d_sub, const float * d_norm_w,
    void * d_xprep, float * d_x_out, int H, int T, float rms_eps,
    bool do_add, int prep_mode, cudaStream_t stream) {

    if (do_add) {
        if (prep_mode == 1) {
            sn_rmsnorm_prep_kernel<true, 1><<<T, 256, 0, stream>>>(d_cur, d_sub, d_norm_w, d_xprep, d_x_out, H, T, rms_eps);
        } else {
            sn_rmsnorm_prep_kernel<true, 2><<<T, 256, 0, stream>>>(d_cur, d_sub, d_norm_w, d_xprep, d_x_out, H, T, rms_eps);
        }
    } else {
        if (prep_mode == 1) {
            sn_rmsnorm_prep_kernel<false, 1><<<T, 256, 0, stream>>>(d_cur, nullptr, d_norm_w, d_xprep, d_x_out, H, T, rms_eps);
        } else {
            sn_rmsnorm_prep_kernel<false, 2><<<T, 256, 0, stream>>>(d_cur, nullptr, d_norm_w, d_xprep, d_x_out, H, T, rms_eps);
        }
    }
}

void launch_sensenova_mhc_merge(
    const float * d_res_in, const float * d_branch, const float * d_ctrl,
    float * d_res_out, int H, int T, cudaStream_t stream) {
    const int H4 = H >> 2;
    sn_mhc_merge_kernel<<<dim3((H4 + 255) / 256, T), 256, 0, stream>>>(
        d_res_in, d_branch, d_ctrl, d_res_out, H);
}

void launch_sensenova_mhc_merge_router_part(
    const float * d_res_in, const float * d_branch, const float * d_ctrl,
    float * d_res_out, const float * d_next_w_hc, void * d_part,
    int H, int T, cudaStream_t stream) {
    constexpr int S = 16;
    sn_mhc_merge_router_part_kernel<S><<<dim3(S, T), 256, 0, stream>>>(
        d_res_in, d_branch, d_ctrl, d_res_out, d_next_w_hc,
        static_cast<float *>(d_part), H);
}

void launch_sensenova_mhc_merge_leave(
    const float * d_res_in, const float * d_branch, const float * d_ctrl,
    float * d_cur_out, int H, int T, cudaStream_t stream) {
    const int H4 = H >> 2;
    sn_mhc_merge_leave_kernel<<<dim3((H4 + 255) / 256, T), 256, 0, stream>>>(
        d_res_in, d_branch, d_ctrl, d_cur_out, H);
}

void launch_sensenova_gdn_gate_beta_packed(
    const float * d_ba, const float * d_dt_bias, const float * d_ssm_a,
    float * d_g_out, float * d_beta_out, int T, int n_v_heads, cudaStream_t stream) {
    sn_gdn_gate_beta_packed_kernel<<<dim3((n_v_heads + 63) / 64, T), 64, 0, stream>>>(
        d_ba, d_dt_bias, d_ssm_a, d_g_out, d_beta_out, n_v_heads);
}

void launch_sensenova_gdn_conv_and_recurrence(
    const float * d_qkvz,
    const float * d_conv_w,
    float *       d_conv_state,
    float *       d_qkv_conv,
    const float * d_g,
    const float * d_beta,
    float *       d_ssm_state,
    float *       d_gdn_out,
    int           T,
    int           n_k_heads,
    int           n_v_heads,
    int           qkvz_stride,
    float         rms_eps,
    float         gdn_scale,
    cudaStream_t  stream,
    float *       d_conv_snap,
    float *       d_ssm_snap) {

    const int conv_dim = (2 * n_k_heads + n_v_heads) * 128;
    if (T >= 8 && d_conv_snap == nullptr && d_ssm_snap == nullptr) {
        dim3 grid_c((conv_dim + 255) / 256, (T + 7) / 8);
        sn_conv1d_silu_strided_prefill_kernel<<<grid_c, 256, 0, stream>>>(
            d_qkvz, d_conv_w, d_conv_state, d_qkv_conv, T, conv_dim, qkvz_stride);
        sn_conv1d_update_strided_kernel<<<(conv_dim + 255) / 256, 256, 0, stream>>>(
            d_qkvz, d_conv_state, T, conv_dim, qkvz_stride);
        constexpr int l2_wpb = 8;
        sn_qkv_l2_norm_d128_kernel<<<dim3((2 * n_k_heads + l2_wpb - 1) / l2_wpb, T), dim3(32, l2_wpb), 0, stream>>>(
            d_qkv_conv, n_k_heads, conv_dim, rms_eps);
        sn_gdn_packed_d128_prefill_smem_kernel<<<dim3(n_v_heads, 16), dim3(32, 8), 0, stream>>>(
            d_qkv_conv, d_g, d_beta, d_ssm_state, d_gdn_out,
            T, n_k_heads, n_v_heads, conv_dim, gdn_scale);
    } else {
        // Fused Conv1D + SiLU + per-head L2-Norm(Q, K) in ONE launch!
        const int total_h = 2 * n_k_heads + n_v_heads;
        sn_conv1d_silu_l2_strided_dec_kernel<<<total_h, 128, 0, stream>>>(
            d_qkvz, d_conv_w, d_conv_state, d_qkv_conv, d_conv_snap,
            T, conv_dim, qkvz_stride, 2 * n_k_heads, rms_eps);
        constexpr int wpb = 4;
        sn_gdn_packed_d128_dec_kernel<<<dim3(n_v_heads, 128 / wpb), dim3(32, wpb), 0, stream>>>(
            d_qkv_conv, d_g, d_beta, d_ssm_state, d_gdn_out, d_ssm_snap,
            T, n_k_heads, n_v_heads, conv_dim, gdn_scale);
    }
}

void launch_sensenova_rms_norm_silu_gate_d128_strided(
    float * d_o, const float * d_norm_w, const float * d_z_base,
    int T, int n_v_heads, int z_stride, int z_offset, float eps,
    cudaStream_t stream, void * d_xprep_out, int prep_mode) {
    constexpr int wpb = 8;
    dim3 grid((n_v_heads + wpb - 1) / wpb, T);
    if (d_xprep_out != nullptr && prep_mode == 1) {
        sn_rms_norm_silu_gate_prep_kernel<1><<<grid, dim3(32, wpb), 0, stream>>>(
            d_o, d_norm_w, d_z_base, d_xprep_out, T, n_v_heads, z_stride, z_offset, eps);
    } else if (d_xprep_out != nullptr && prep_mode == 2) {
        sn_rms_norm_silu_gate_prep_kernel<2><<<grid, dim3(32, wpb), 0, stream>>>(
            d_o, d_norm_w, d_z_base, d_xprep_out, T, n_v_heads, z_stride, z_offset, eps);
    } else {
        sn_rms_norm_silu_gate_prep_kernel<0><<<grid, dim3(32, wpb), 0, stream>>>(
            d_o, d_norm_w, d_z_base, nullptr, T, n_v_heads, z_stride, z_offset, eps);
    }
}

void launch_sensenova_sigmoid_gate_mul_prep(
    const float * d_a, const float * d_g, void * d_xprep,
    int H, int T, int prep_mode, cudaStream_t stream) {
    const int K_blocks = H >> 5;
    constexpr int wpb = 8;
    dim3 grid((K_blocks + wpb - 1) / wpb, T);
    if (prep_mode == 1) {
        sn_sigmoid_gate_mul_prep_kernel<1><<<grid, dim3(32, wpb), 0, stream>>>(d_a, d_g, d_xprep, H, T);
    } else {
        sn_sigmoid_gate_mul_prep_kernel<2><<<grid, dim3(32, wpb), 0, stream>>>(d_a, d_g, d_xprep, H, T);
    }
}

void launch_sensenova_qgate_split_norm_yarn_rope_d256(
    const float * d_q_full, const float * d_q_norm_w, const int32_t * d_pos,
    const float * d_inv_freq, float * d_q_out, float * d_gate_out,
    int T, int n_head, int rope_dim, float rms_eps, float yarn_mscale, cudaStream_t stream) {
    constexpr int wpb = 4;
    sn_qgate_split_norm_yarn_rope_d256_kernel<<<dim3((n_head + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
        d_q_full, d_q_norm_w, d_pos, d_inv_freq, d_q_out, d_gate_out,
        n_head, rope_dim, rms_eps, yarn_mscale);
}

void launch_sensenova_k_norm_yarn_rope_d256(
    float * d_k, const float * d_k_norm_w, const int32_t * d_pos,
    const float * d_inv_freq, int T, int n_kv, int rope_dim,
    float rms_eps, float yarn_mscale, cudaStream_t stream) {
    constexpr int wpb = 4;
    sn_k_norm_yarn_rope_d256_kernel<<<dim3((n_kv + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
        d_k, d_k_norm_w, d_pos, d_inv_freq, n_kv, rope_dim, rms_eps, yarn_mscale);
}

bool try_launch_sensenova_q8_0_sm60_smem_prepped(
    const DeviceTensor & W, const void * d_xprep, float * d_y, int T, cudaStream_t stream) {
    if ((T != 1 && T != 2) || W.dtype != DType::Q8_0 || !W.is_soa) return false;
    const int K        = static_cast<int>(W.ne[0]);
    const int K_blocks = K >> 5;
    const int N        = static_cast<int>(W.ne[1]);
    const size_t smem  = static_cast<size_t>(T) * K_blocks * (64 + sizeof(float));
    if (smem > 32768) return false;

    const size_t h2_bytes = (static_cast<size_t>(T) * K * sizeof(__half) + 255) & ~size_t(255);
    const auto * x_h2 = static_cast<const int4 *>(d_xprep);
    const auto * x_sc = reinterpret_cast<const float *>(static_cast<const uint8_t *>(d_xprep) + h2_bytes);

    if (T == 1) {
        if (N >= 1024) {
            constexpr int wpb = 8, rpw = 2;
            sn_gemv_q8_0_sm60_smem_kernel<rpw, false><<<(N + wpb * rpw - 1) / (wpb * rpw), dim3(32, wpb), smem, stream>>>(
                static_cast<const int4 *>(W.ptr), static_cast<const uint16_t *>(W.scales_ptr),
                x_h2, x_sc, d_y, K_blocks, N);
        } else {
            const int wpb = (N <= 128) ? 2 : 4;
            sn_gemv_q8_0_sm60_smem_kernel<1, false><<<(N + wpb - 1) / wpb, dim3(32, wpb), smem, stream>>>(
                static_cast<const int4 *>(W.ptr), static_cast<const uint16_t *>(W.scales_ptr),
                x_h2, x_sc, d_y, K_blocks, N);
        }
    } else {
        if (N >= 1024) {
            constexpr int wpb = 8, rpw = 2;
            sn_gemv_q8_0_sm60_smem_kernel<rpw, true><<<(N + wpb * rpw - 1) / (wpb * rpw), dim3(32, wpb), smem, stream>>>(
                static_cast<const int4 *>(W.ptr), static_cast<const uint16_t *>(W.scales_ptr),
                x_h2, x_sc, d_y, K_blocks, N);
        } else {
            const int wpb = (N <= 128) ? 2 : 4;
            sn_gemv_q8_0_sm60_smem_kernel<1, true><<<(N + wpb - 1) / wpb, dim3(32, wpb), smem, stream>>>(
                static_cast<const int4 *>(W.ptr), static_cast<const uint16_t *>(W.scales_ptr),
                x_h2, x_sc, d_y, K_blocks, N);
        }
    }
    return true;
}

bool try_launch_sensenova_swiglu_q8_0_fused(
    const DeviceTensor & W_gate,
    const DeviceTensor & W_up,
    const DeviceTensor & W_down,
    const void *         d_xprep,
    void *               d_act_prep,
    float *              d_out,
    int                  T,
    int                  sm,
    cudaStream_t         stream) {

    if ((T != 1 && T != 2) || W_gate.dtype != DType::Q8_0 || !W_gate.is_soa ||
        W_up.dtype != DType::Q8_0 || !W_up.is_soa || W_down.dtype != DType::Q8_0 || !W_down.is_soa) {
        return false;
    }
    const int K        = static_cast<int>(W_gate.ne[0]);
    const int ff       = static_cast<int>(W_gate.ne[1]);
    if ((K & 31) != 0 || (ff & 31) != 0) return false;
    const int K_blocks = K >> 5;
    const int ff_blks  = ff >> 5;

    if (sm == 60) {
        const size_t smem = static_cast<size_t>(T) * K_blocks * (64 + sizeof(float));
        if (smem > 32768) return false;
        const size_t x_h2_b = (static_cast<size_t>(T) * K * sizeof(__half) + 255) & ~size_t(255);
        const auto * x_h2   = static_cast<const int4 *>(d_xprep);
        const auto * x_sc   = reinterpret_cast<const float *>(static_cast<const uint8_t *>(d_xprep) + x_h2_b);

        const size_t a_h2_b = (static_cast<size_t>(T) * ff * sizeof(__half) + 255) & ~size_t(255);
        auto * a_h2 = static_cast<__half2 *>(d_act_prep);
        auto * a_sc = reinterpret_cast<float *>(static_cast<uint8_t *>(d_act_prep) + a_h2_b);

        if (T == 1) {
            sn_swiglu_q8_0_sm60_fused_prep_kernel<false><<<ff_blks, dim3(32, 16), smem, stream>>>(
                static_cast<const int4 *>(W_gate.ptr), static_cast<const uint16_t *>(W_gate.scales_ptr),
                static_cast<const int4 *>(W_up.ptr),   static_cast<const uint16_t *>(W_up.scales_ptr),
                x_h2, x_sc, a_h2, a_sc, K_blocks, ff);
        } else {
            sn_swiglu_q8_0_sm60_fused_prep_kernel<true><<<ff_blks, dim3(32, 16), smem, stream>>>(
                static_cast<const int4 *>(W_gate.ptr), static_cast<const uint16_t *>(W_gate.scales_ptr),
                static_cast<const int4 *>(W_up.ptr),   static_cast<const uint16_t *>(W_up.scales_ptr),
                x_h2, x_sc, a_h2, a_sc, K_blocks, ff);
        }
        if (!try_launch_sensenova_q8_0_sm60_smem_prepped(W_down, d_act_prep, d_out, T, stream)) {
            launch_linear_q8_0_sm60_h2_prepped(W_down, d_act_prep, d_out, T, stream);
        }
        return true;
    } else {
        const size_t smem = static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act);
        if (smem > 32768) return false;
        if (T == 1) {
            sn_swiglu_q8_0_dp4a_fused_q81_kernel<false><<<ff_blks, dim3(32, 8), smem, stream>>>(
                static_cast<const int4 *>(W_gate.ptr), static_cast<const uint16_t *>(W_gate.scales_ptr),
                static_cast<const int4 *>(W_up.ptr),   static_cast<const uint16_t *>(W_up.scales_ptr),
                static_cast<const BlockQ8_1_Act *>(d_xprep),
                static_cast<BlockQ8_1_Act *>(d_act_prep), K_blocks, ff);
        } else {
            sn_swiglu_q8_0_dp4a_fused_q81_kernel<true><<<ff_blks, dim3(32, 8), smem, stream>>>(
                static_cast<const int4 *>(W_gate.ptr), static_cast<const uint16_t *>(W_gate.scales_ptr),
                static_cast<const int4 *>(W_up.ptr),   static_cast<const uint16_t *>(W_up.scales_ptr),
                static_cast<const BlockQ8_1_Act *>(d_xprep),
                static_cast<BlockQ8_1_Act *>(d_act_prep), K_blocks, ff);
        }
        launch_linear_q8_0_prequant(W_down, d_act_prep, d_out, T, stream);
        return true;
    }
}

// Pure-FP16 Prefill SwiGLU FFN on sm_60:
// Keeps Gate, Up, and SiLU(Gate)*Up in FP16 -> eliminates 175 MB/layer of F16<->F32 & F32 silu_mul!
bool try_launch_sensenova_ffn_prefill_sm60_f16(
    cublasHandle_t       cublas,
    const DeviceTensor & W_gate,
    const DeviceTensor & W_up,
    const DeviceTensor & W_down,
    const float *        d_x,
    float *              d_out,
    int                  T,
    void *               d_scratch,
    size_t               scratch_bytes,
    cudaStream_t         stream) {

    if (cublas == nullptr || T < 16 || W_gate.dtype != DType::Q8_0 || !W_gate.is_soa ||
        W_up.dtype != DType::Q8_0 || !W_up.is_soa || W_down.dtype != DType::Q8_0 || !W_down.is_soa) {
        return false;
    }
    const int H  = static_cast<int>(W_gate.ne[0]);
    const int ff = static_cast<int>(W_gate.ne[1]);
    const size_t x_b  = (static_cast<size_t>(T) * H * sizeof(__half) + 255) & ~size_t(255);
    const size_t g_b  = (static_cast<size_t>(T) * ff * sizeof(__half) + 255) & ~size_t(255);
    const size_t u_b  = g_b;
    const size_t base = x_b + g_b + u_b;
    // Need room for at least 1024 rows of W_f16 (1024 * ff * 2 = 24 MB)
    const size_t min_w = static_cast<size_t>(ff) * 512 * sizeof(__half);
    if (base + min_w > scratch_bytes) return false;

    auto * d_xf16 = static_cast<__half *>(d_scratch);
    auto * d_gf16 = reinterpret_cast<__half *>(static_cast<uint8_t *>(d_scratch) + x_b);
    auto * d_uf16 = reinterpret_cast<__half *>(static_cast<uint8_t *>(d_scratch) + x_b + g_b);
    auto * d_wf16 = reinterpret_cast<__half *>(static_cast<uint8_t *>(d_scratch) + base);
    const size_t w_cap = scratch_bytes - base;

    const int64_t nx  = static_cast<int64_t>(T) * H;
    sn_f32_to_f16_vec_kernel<<<((nx + 3) / 4 + 255) / 256, 256, 0, stream>>>(d_x, d_xf16, nx);

    const __half h_one  = __float2half_rn(1.0f);
    const __half h_zero = __float2half_rn(0.0f);
    cublasSetStream(cublas, stream);
    constexpr int wpb = 8;

    auto run_hgemm_q8 = [&](const DeviceTensor & W, const __half * Xf16, __half * Yf16) {
        const int K = static_cast<int>(W.ne[0]);
        const int N = static_cast<int>(W.ne[1]);
        const int nb_row = K / 32;
        int chunk_n = std::max(1, std::min(N, static_cast<int>(w_cap / (static_cast<size_t>(K) * sizeof(__half)))));
        if (chunk_n < N && chunk_n >= 128) chunk_n &= ~127;
        for (int n0 = 0; n0 < N; n0 += chunk_n) {
            const int cn = std::min(chunk_n, N - n0);
            const int64_t b0 = static_cast<int64_t>(n0) * nb_row;
            const int64_t n_blocks = static_cast<int64_t>(cn) * nb_row;
            sn_dequant_q8_0_to_f16_kernel<<<(n_blocks + 127) / 128, dim3(32, wpb), 0, stream>>>(
                static_cast<const int4 *>(W.ptr) + b0 * 2,
                static_cast<const uint16_t *>(W.scales_ptr) + b0,
                d_wf16, n_blocks);
            cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        cn, T, K, &h_one, d_wf16, K, Xf16, K, &h_zero, Yf16 + n0, N);
        }
    };

    // 1. Gate & Up -> FP16 (single X conversion, zero F16->F32!)
    run_hgemm_q8(W_gate, d_xf16, d_gf16);
    run_hgemm_q8(W_up,   d_xf16, d_uf16);

    // 2. Vectorized FP16 SiLU(Gate) * Up in-place into d_gf16!
    const int64_t nf = static_cast<int64_t>(T) * ff;
    sn_silu_mul_f16_inplace_kernel<<<((nf + 3) / 4 + 255) / 256, 256, 0, stream>>>(d_gf16, d_uf16, nf);

    // 3. Down: reuse d_xf16 as output FP16 buffer [T, H], and d_uf16..end as W_down FP16 buffer!
    auto * d_wdown_f16 = d_uf16;
    const size_t wd_cap = scratch_bytes - (x_b + g_b);
    const int nb_ff = ff / 32;
    int chunk_h = std::max(1, std::min(H, static_cast<int>(wd_cap / (static_cast<size_t>(ff) * sizeof(__half)))));
    if (chunk_h < H && chunk_h >= 128) chunk_h &= ~127;
    for (int n0 = 0; n0 < H; n0 += chunk_h) {
        const int cn = std::min(chunk_h, H - n0);
        const int64_t b0 = static_cast<int64_t>(n0) * nb_ff;
        const int64_t n_blocks = static_cast<int64_t>(cn) * nb_ff;
        sn_dequant_q8_0_to_f16_kernel<<<(n_blocks + 127) / 128, dim3(32, wpb), 0, stream>>>(
            static_cast<const int4 *>(W_down.ptr) + b0 * 2,
            static_cast<const uint16_t *>(W_down.scales_ptr) + b0,
            d_wdown_f16, n_blocks);
        cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    cn, T, ff, &h_one, d_wdown_f16, ff, d_gf16, ff, &h_zero, d_xf16 + n0, H);
    }
    sn_f16_to_f32_vec_kernel<<<((nx + 3) / 4 + 255) / 256, 256, 0, stream>>>(d_xf16, d_out, nx);
    return true;
}

// Multi-matrix shared-X FP16 Prefill on sm_60 (for GDN qkv_gate+ba and GQA q+k+v!)
bool try_launch_sensenova_multi_prefill_sm60_f16(
    cublasHandle_t               cublas,
    const DeviceTensor * const * Ws,
    float * const *              Ys,
    int                          n_mat,
    const float *                d_x,
    int                          T,
    void *                       d_scratch,
    size_t                       scratch_bytes,
    cudaStream_t                 stream) {

    if (cublas == nullptr || T < 16 || n_mat <= 0) return false;
    const int K = static_cast<int>(Ws[0]->ne[0]);
    for (int m = 0; m < n_mat; ++m) {
        if (Ws[m]->dtype != DType::Q8_0 || !Ws[m]->is_soa || static_cast<int>(Ws[m]->ne[0]) != K) return false;
    }
    const size_t align_x = (static_cast<size_t>(T) * K * sizeof(__half) + 255) & ~size_t(255);
    if (align_x + static_cast<size_t>(K + T) * sizeof(__half) + 256 > scratch_bytes) return false;

    auto * d_xf16 = static_cast<__half *>(d_scratch);
    const int64_t nx = static_cast<int64_t>(T) * K;
    sn_f32_to_f16_vec_kernel<<<((nx + 3) / 4 + 255) / 256, 256, 0, stream>>>(d_x, d_xf16, nx);

    const __half h_one  = __float2half_rn(1.0f);
    const __half h_zero = __float2half_rn(0.0f);
    cublasSetStream(cublas, stream);
    const int nb_row = K / 32;
    constexpr int wpb = 8;

    for (int m = 0; m < n_mat; ++m) {
        const DeviceTensor & W = *Ws[m];
        float * d_y = Ys[m];
        const int N = static_cast<int>(W.ne[1]);
        const size_t rem = scratch_bytes - align_x - 256;
        int chunk_n = std::max(1, std::min(N, static_cast<int>(rem / (static_cast<size_t>(K + T) * sizeof(__half)))));
        if (chunk_n < N && chunk_n >= 128) chunk_n &= ~127;

        auto * d_wf16 = reinterpret_cast<__half *>(static_cast<uint8_t *>(d_scratch) + align_x);
        const size_t w_bytes = (static_cast<size_t>(chunk_n) * K * sizeof(__half) + 255) & ~size_t(255);
        auto * d_yf16 = reinterpret_cast<__half *>(static_cast<uint8_t *>(d_scratch) + align_x + w_bytes);

        for (int n0 = 0; n0 < N; n0 += chunk_n) {
            const int cn = std::min(chunk_n, N - n0);
            const int64_t b0 = static_cast<int64_t>(n0) * nb_row;
            const int64_t n_blocks = static_cast<int64_t>(cn) * nb_row;
            sn_dequant_q8_0_to_f16_kernel<<<(n_blocks + 127) / 128, dim3(32, wpb), 0, stream>>>(
                static_cast<const int4 *>(W.ptr) + b0 * 2,
                static_cast<const uint16_t *>(W.scales_ptr) + b0,
                d_wf16, n_blocks);
            cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        cn, T, K, &h_one, d_wf16, K, d_xf16, K, &h_zero, d_yf16, cn);
            if (cn == N && n0 == 0) {
                const int64_t ny = static_cast<int64_t>(T) * N;
                sn_f16_to_f32_vec_kernel<<<((ny + 3) / 4 + 255) / 256, 256, 0, stream>>>(d_yf16, d_y, ny);
            } else {
                sn_f16_to_f32_strided_kernel<<<dim3((cn + 255) / 256, T), 256, 0, stream>>>(d_yf16, d_y, cn, N, n0, T);
            }
        }
    }
    return true;
}

} // namespace rookery
