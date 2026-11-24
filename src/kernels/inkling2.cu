#include "core/device.h"
#include "kernels/kernels.h"
#include "kernels/iq_tables.cuh"
#include "kernels/q8_1_act.cuh"

#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace rookery {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

// ============================================================================
// 1. Block Attention Residuals: mix(blocks, cur, query[n_q, n_embd])
//    Exact match to llama_model_inkling2::graph::mix():
//      states = RMSNorm([blocks[0..n_blocks-1], cur], eps=1e-6)
//      scores[q, s] = dot(states[s], query[q]) / sqrt(n_embd)
//      w[q, :] = softmax(scores[q, :])
//      out[q, t, :] = sum_s w[q, s] * states[s, t, :]
// ============================================================================
constexpr int MAX_DEPTH_STATES = 8; // max blocks (<=5) + cur <= 6

__global__ void inkling2_depth_mix_kernel(
    const float * const * __restrict__ block_ptrs,
    int n_blocks,
    const float * __restrict__ cur,
    const float * __restrict__ query, // [n_q, n_embd]
    float *       __restrict__ out,   // [n_q, T, n_embd]
    int n_embd, int n_q, int T, float eps) {

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_st = n_blocks + 1;
    const int n4   = n_embd >> 2;

    __shared__ float s_inv_rms[MAX_DEPTH_STATES];
    __shared__ float s_red[8];
    __shared__ float s_w[4][MAX_DEPTH_STATES]; // n_q <= 4

    // 1. Compute inv_rms for each state s = 0..n_st-1
    for (int s = 0; s < n_st; ++s) {
        const float * sp = (s < n_blocks) ? block_ptrs[s] : cur;
        const auto * x4 = reinterpret_cast<const float4 *>(sp + static_cast<size_t>(t) * n_embd);
        float ss = 0.0f;
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 v = x4[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        ss = warp_sum(ss);
        if (lane == 0) s_red[warp] = ss;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 8) ? s_red[lane] : 0.0f;
            tot = warp_sum(tot);
            if (lane == 0) s_inv_rms[s] = rsqrtf(tot / static_cast<float>(n_embd) + eps);
        }
        __syncthreads();
    }

    const float scale = rsqrtf(static_cast<float>(n_embd));

    // 2. For each query q = 0..n_q-1, compute softmax weights over states
    for (int q = 0; q < n_q; ++q) {
        const auto * q4 = reinterpret_cast<const float4 *>(query + static_cast<size_t>(q) * n_embd);
        for (int s = 0; s < n_st; ++s) {
            const float * sp = (s < n_blocks) ? block_ptrs[s] : cur;
            const auto * x4 = reinterpret_cast<const float4 *>(sp + static_cast<size_t>(t) * n_embd);
            float dot = 0.0f;
            for (int i = tid; i < n4; i += blockDim.x) {
                float4 xv = x4[i], qv = q4[i];
                dot += xv.x * qv.x + xv.y * qv.y + xv.z * qv.z + xv.w * qv.w;
            }
            dot = warp_sum(dot);
            if (lane == 0) s_red[warp] = dot;
            __syncthreads();
            if (warp == 0) {
                float tot = (lane < 8) ? s_red[lane] : 0.0f;
                tot = warp_sum(tot);
                if (lane == 0) s_w[q][s] = tot * s_inv_rms[s] * scale;
            }
            __syncthreads();
        }
        if (tid == 0) {
            float mx = -FLT_MAX;
            for (int s = 0; s < n_st; ++s) mx = fmaxf(mx, s_w[q][s]);
            float sum = 0.0f;
            for (int s = 0; s < n_st; ++s) {
                s_w[q][s] = expf(s_w[q][s] - mx);
                sum += s_w[q][s];
            }
            const float inv = 1.0f / sum;
            for (int s = 0; s < n_st; ++s) s_w[q][s] *= inv * s_inv_rms[s];
        }
        __syncthreads();
    }

    // 3. Weighted sum of RMSNorm'd states -> out[q, t, :]
    for (int q = 0; q < n_q; ++q) {
        auto * o4 = reinterpret_cast<float4 *>(out + (static_cast<size_t>(q) * T + t) * n_embd);
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int s = 0; s < n_st; ++s) {
                const float * sp = (s < n_blocks) ? block_ptrs[s] : cur;
                float4 xv = reinterpret_cast<const float4 *>(sp + static_cast<size_t>(t) * n_embd)[i];
                const float ws = s_w[q][s];
                acc.x += ws * xv.x; acc.y += ws * xv.y;
                acc.z += ws * xv.z; acc.w += ws * xv.w;
            }
            o4[i] = acc;
        }
    }
}

// ============================================================================
// 2. Loop Controller Helpers: RMSNorm concat [cur, msg] + (1 - tanh(f))*cur + tanh(g)*msg
// ============================================================================
__global__ void inkling2_rmsnorm_concat2_kernel(
    const float * __restrict__ a,
    const float * __restrict__ b,
    float *       __restrict__ out, // [T, 2 * n_embd]
    int n_embd, float eps) {

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n4   = n_embd >> 2;
    __shared__ float s_red[8];

    auto do_half = [&](const float * src, float * dst) {
        const auto * x4 = reinterpret_cast<const float4 *>(src + static_cast<size_t>(t) * n_embd);
        float ss = 0.0f;
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 v = x4[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        ss = warp_sum(ss);
        if (lane == 0) s_red[warp] = ss;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 8) ? s_red[lane] : 0.0f;
            tot = warp_sum(tot);
            if (lane == 0) s_red[0] = rsqrtf(tot / static_cast<float>(n_embd) + eps);
        }
        __syncthreads();
        const float inv = s_red[0];
        auto * o4 = reinterpret_cast<float4 *>(dst);
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 v = x4[i];
            o4[i] = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
        }
        __syncthreads();
    };

    float * row = out + static_cast<size_t>(t) * (2 * n_embd);
    do_half(a, row);
    do_half(b, row + n_embd);
}

__global__ void inkling2_loop_combine_kernel(
    const float * __restrict__ cur,
    const float * __restrict__ msg,
    const float * __restrict__ gain_pre,
    const float * __restrict__ forget_pre,
    float *       __restrict__ out,
    int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float c = cur[i];
        const float g = tanhf(gain_pre[i]);
        const float f = tanhf(forget_pre[i]);
        out[i] = (c - f * c) + g * msg[i];
    }
}

// ============================================================================
// 3. Causal ShortConv (l_cache=4) + optional Dynamic Per-Head Taps + Residual + Scale
//    Exact match to llama_model_inkling2::graph::convolve():
//      static:  sum_{k=0..3} w[c*4 + k] * x[t - (3 - k), c]
//      dynamic: sum_{lag=0..3} dyn[t, h, lag] * x[t - lag, c]
//      out = (x[t] + static + dynamic) / sqrt((dyn ? 2 : 1) * 4 + 1)
// ============================================================================
__global__ void inkling2_shortconv_kernel(
    const float * __restrict__ x,      // [T, C]
    const float * __restrict__ w,      // [C, 4]
    const float * __restrict__ dyn,    // optional [T, n_head_kv, 4]
    float *       __restrict__ state,  // [3, C] (oldest=0 .. newest=2)
    float *       __restrict__ out,    // [T, C]
    const float * __restrict__ res,    // optional [T, C] residual add
    int T, int C, int head_dim, int n_head_kv) {

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float s0 = state[0 * C + c]; // x[t-3]
    float s1 = state[1 * C + c]; // x[t-2]
    float s2 = state[2 * C + c]; // x[t-1]

    const float w0 = w[c * 4 + 0];
    const float w1 = w[c * 4 + 1];
    const float w2 = w[c * 4 + 2];
    const float w3 = w[c * 4 + 3];
    const float inv_scale = dyn ? (1.0f / 3.0f) : rsqrtf(5.0f); // 1/sqrt(9) or 1/sqrt(5)
    const int h = dyn ? (c / head_dim) : 0;

    for (int t = 0; t < T; ++t) {
        const float xt = x[static_cast<size_t>(t) * C + c];
        float acc = s0 * w0 + s1 * w1 + s2 * w2 + xt * w3;
        if (dyn) {
            const float * dt = dyn + (static_cast<size_t>(t) * n_head_kv + h) * 4;
            // lag=0 -> xt (newest), lag=1 -> s2 (t-1), lag=2 -> s1 (t-2), lag=3 -> s0 (t-3)
            acc += dt[0] * xt + dt[1] * s2 + dt[2] * s1 + dt[3] * s0;
        }
        const float yv = (xt + acc) * inv_scale;
        const size_t idx = static_cast<size_t>(t) * C + c;
        out[idx] = res ? (res[idx] + yv) : yv;
        s0 = s1;
        s1 = s2;
        s2 = xt;
    }

    state[0 * C + c] = s0;
    state[1 * C + c] = s1;
    state[2 * C + c] = s2;
}

// ============================================================================
// 4. Per-Head L2 Clip (+ optional per-channel scale + optional log-scale tau) & Head RMSNorm
// ============================================================================
__global__ void inkling2_l2_clip_kernel(
    float *         __restrict__ x,      // [T, n_head, head_dim]
    const float *   __restrict__ scale,  // optional [n_head, head_dim]
    const int32_t * __restrict__ pos,    // optional [T]
    float threshold, float log_floor, float log_alpha,
    int n_head, int head_dim) {

    const int h    = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    if (h >= n_head) return;
    const int lane = threadIdx.x;

    float * row = x + (static_cast<size_t>(t) * n_head + h) * head_dim;
    const float * sr = scale ? (scale + static_cast<size_t>(h) * head_dim) : nullptr;

    float ss = 0.0f;
    for (int d = lane; d < head_dim; d += 32) {
        float v = row[d];
        if (sr) v *= sr[d];
        row[d] = v;
        ss += v * v;
    }
    ss = __shfl_sync(0xffffffffu, warp_sum(ss), 0);
    const float norm = sqrtf(ss) + 1e-6f;
    float mul = fminf(1.0f, threshold / norm);
    if (pos != nullptr) {
        const float ratio = fmaxf(1.0f, static_cast<float>(pos[t] + 1) / log_floor);
        const float tau   = 1.0f + log_alpha * logf(ratio);
        mul *= tau;
    }
    for (int d = lane; d < head_dim; d += 32) {
        row[d] *= mul;
    }
}

__global__ void inkling2_head_rmsnorm_kernel(
    float * __restrict__ x, int n_head, int head_dim, float eps) {
    const int h    = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    if (h >= n_head) return;
    const int lane = threadIdx.x;
    float * row = x + (static_cast<size_t>(t) * n_head + h) * head_dim;
    float ss = 0.0f;
    for (int d = lane; d < head_dim; d += 32) {
        float v = row[d];
        ss += v * v;
    }
    ss = __shfl_sync(0xffffffffu, warp_sum(ss), 0);
    const float inv = rsqrtf(ss / static_cast<float>(head_dim) + eps);
    for (int d = lane; d < head_dim; d += 32) row[d] *= inv;
}

// ============================================================================
// 5. Relative Position Bias Table: table[t, h, d] = dot(R[t,h], rel_bias[d]) - sinks[t,h]
//    for d = 0..n_extent-1, and table[t, h, n_extent] = -sinks[t, h] (padded zero row!).
// ============================================================================
__global__ void inkling2_rel_bias_table_kernel(
    const float * __restrict__ r,          // [T, n_head, n_rel]
    const float * __restrict__ rel_bias,   // [n_extent, n_rel]
    const float * __restrict__ sinks,      // [T, n_head]
    float *       __restrict__ bias_table, // [T, n_head, n_extent + 1]
    int n_head, int n_rel, int n_extent) {

    const int h = blockIdx.y;
    const int t = blockIdx.z;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d > n_extent) return;

    const float sink = sinks[static_cast<size_t>(t) * n_head + h];
    float val = 0.0f;
    if (d < n_extent) {
        const float * rp = r        + (static_cast<size_t>(t) * n_head + h) * n_rel;
        const float * bp = rel_bias + static_cast<size_t>(d) * n_rel;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            if (k < n_rel) val += rp[k] * bp[k];
        }
    }
    bias_table[(static_cast<size_t>(t) * n_head + h) * (n_extent + 1) + d] = val - sink;
}

// Fast shared-memory tiled rel_bias_table for n_rel == 16 (Inkling-2-Mini):
// - Decode (T <= 4): loads rel_bias[d, 0..15] ONCE into registers (4 x float4) and computes ALL n_head heads!
// - Prefill (T > 4): 2D tile [32 d-cols x 32 (t,h)-rows] in smem via float4 -> 25x faster than scalar!
__global__ void inkling2_rel_bias_table_r16_dec_kernel(
    const float * __restrict__ r,          // [T, n_head, 16]
    const float * __restrict__ rel_bias,   // [n_extent, 16]
    const float * __restrict__ sinks,      // [T, n_head]
    float *       __restrict__ bias_table, // [T, n_head, n_extent + 1]
    int n_head, int n_extent) {

    extern __shared__ float smem_r[]; // [n_head * 16] + [n_head]
    float * s_r    = smem_r;
    float * s_sink = smem_r + n_head * 16;

    const int t   = blockIdx.y;
    const int tid = threadIdx.x;
    const int d   = blockIdx.x * blockDim.x + tid;

    const int n_r4 = n_head * 4;
    const auto * r4_in  = reinterpret_cast<const float4 *>(r + static_cast<size_t>(t) * n_head * 16);
    auto *       r4_out = reinterpret_cast<float4 *>(s_r);
    for (int i = tid; i < n_r4; i += blockDim.x) r4_out[i] = r4_in[i];
    if (tid < n_head) s_sink[tid] = sinks[static_cast<size_t>(t) * n_head + tid];
    __syncthreads();

    if (d > n_extent) return;
    if (d == n_extent) {
        for (int h = 0; h < n_head; ++h) {
            bias_table[(static_cast<size_t>(t) * n_head + h) * (n_extent + 1) + d] = -s_sink[h];
        }
        return;
    }

    const auto * b4 = reinterpret_cast<const float4 *>(rel_bias + static_cast<size_t>(d) * 16);
    const float4 bv0 = __ldg(&b4[0]);
    const float4 bv1 = __ldg(&b4[1]);
    const float4 bv2 = __ldg(&b4[2]);
    const float4 bv3 = __ldg(&b4[3]);

    const auto * sr4 = reinterpret_cast<const float4 *>(s_r);
    for (int h = 0; h < n_head; ++h) {
        const float4 rv0 = sr4[h * 4 + 0];
        const float4 rv1 = sr4[h * 4 + 1];
        const float4 rv2 = sr4[h * 4 + 2];
        const float4 rv3 = sr4[h * 4 + 3];
        float dot = (bv0.x * rv0.x + bv0.y * rv0.y + bv0.z * rv0.z + bv0.w * rv0.w)
                  + (bv1.x * rv1.x + bv1.y * rv1.y + bv1.z * rv1.z + bv1.w * rv1.w)
                  + (bv2.x * rv2.x + bv2.y * rv2.y + bv2.z * rv2.z + bv2.w * rv2.w)
                  + (bv3.x * rv3.x + bv3.y * rv3.y + bv3.z * rv3.z + bv3.w * rv3.w);
        bias_table[(static_cast<size_t>(t) * n_head + h) * (n_extent + 1) + d] = dot - s_sink[h];
    }
}

__global__ void inkling2_rel_bias_table_r16_tiled_kernel(
    const float * __restrict__ r,          // [M, 16] where M = T * n_head
    const float * __restrict__ rel_bias,   // [n_extent, 16]
    const float * __restrict__ sinks,      // [M]
    float *       __restrict__ bias_table, // [M, n_extent + 1]
    int M, int n_extent) {

    // Tile: 32 d-cols x 32 M-rows per CTA (blockDim = 32 x 8 = 256 threads -> each thread computes 4 M-rows!)
    __shared__ float s_b[32][16];
    __shared__ float s_r[32][16];
    __shared__ float s_sink[32];

    const int tx  = threadIdx.x; // 0..31 (d col in tile)
    const int ty  = threadIdx.y; // 0..7
    const int tid = ty * 32 + tx;
    const int d0  = blockIdx.x * 32;
    const int m0  = blockIdx.y * 32;

    // Load 32 x 16 = 512 floats (128 float4) of rel_bias into s_b
    if (tid < 128) {
        const int row = tid >> 2;
        const int col4 = tid & 3;
        const int dg = d0 + row;
        float4 v = (dg < n_extent) ? __ldg(&reinterpret_cast<const float4 *>(rel_bias + static_cast<size_t>(dg) * 16)[col4])
                                   : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        reinterpret_cast<float4 *>(s_b[row])[col4] = v;
    } else {
        const int id = tid - 128; // 0..127
        const int row = id >> 2;
        const int col4 = id & 3;
        const int mg = m0 + row;
        float4 v = (mg < M) ? __ldg(&reinterpret_cast<const float4 *>(r + static_cast<size_t>(mg) * 16)[col4])
                            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        reinterpret_cast<float4 *>(s_r[row])[col4] = v;
    }
    if (tid < 32) {
        const int mg = m0 + tid;
        s_sink[tid] = (mg < M) ? __ldg(&sinks[mg]) : 0.0f;
    }
    __syncthreads();

    const int dg = d0 + tx;
    if (dg > n_extent) return;

    const bool is_pad = (dg == n_extent);
    float4 b0 = make_float4(0,0,0,0), b1 = b0, b2 = b0, b3 = b0;
    if (!is_pad) {
        b0 = reinterpret_cast<const float4 *>(s_b[tx])[0];
        b1 = reinterpret_cast<const float4 *>(s_b[tx])[1];
        b2 = reinterpret_cast<const float4 *>(s_b[tx])[2];
        b3 = reinterpret_cast<const float4 *>(s_b[tx])[3];
    }

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int mr = ty + k * 8;
        const int mg = m0 + mr;
        if (mg < M) {
            float val = -s_sink[mr];
            if (!is_pad) {
                const float4 * r4 = reinterpret_cast<const float4 *>(s_r[mr]);
                const float4 r0 = r4[0], r1 = r4[1], r2 = r4[2], r3 = r4[3];
                val += (b0.x * r0.x + b0.y * r0.y + b0.z * r0.z + b0.w * r0.w)
                     + (b1.x * r1.x + b1.y * r1.y + b1.z * r1.z + b1.w * r1.w)
                     + (b2.x * r2.x + b2.y * r2.y + b2.z * r2.z + b2.w * r2.w)
                     + (b3.x * r3.x + b3.y * r3.y + b3.z * r3.z + b3.w * r3.w);
            }
            bias_table[static_cast<size_t>(mg) * (n_extent + 1) + dg] = val;
        }
    }
}

// ============================================================================
// 6. KV Cache Write (Q8_0 or F16) with Ring Buffer Modulo `cache_cap` for SWA!
// ============================================================================
// SoA Q8_0 KV cache layout inside a single allocation of size cache_cap * n_kv * (head_dim/32) * 34:
//   [0 .. qs_bytes): int8_t qs[cache_cap, n_kv, head_dim] (100% 128-byte aligned per head!)
//   [qs_bytes ..):   uint16_t scales[cache_cap, n_kv, head_dim/32]
__global__ void inkling2_kv_write_q8_0_kernel(
    const float *   __restrict__ k,
    const float *   __restrict__ v,
    void *          __restrict__ k_cache,
    void *          __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    int T, int n_kv, int head_dim, int cache_cap) {

    const int n_blk_h = head_dim >> 5; // 4 for d=128
    const int total_b = n_kv * n_blk_h;
    const int b    = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    if (t >= T || b >= total_b) return;

    const int lane = threadIdx.x;
    const int slot = pos[t] % cache_cap;
    const size_t src_off  = static_cast<size_t>(t) * (n_kv * head_dim) + b * 32 + lane;
    const size_t dst_b    = static_cast<size_t>(slot) * total_b + b;
    const size_t qs_bytes = static_cast<size_t>(cache_cap) * n_kv * head_dim;

    auto quant_store = [&](const float * src, void * dst_raw) {
        auto * dst_qs = static_cast<int8_t *>(dst_raw);
        auto * dst_sc = reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(dst_raw) + qs_bytes);
        const float val  = src[src_off];
        const float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(val)), 0);
        const float d    = amax / 127.0f;
        const float inv  = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        if (lane == 0) {
            __half dh = __float2half_rn(d);
            dst_sc[dst_b] = *reinterpret_cast<uint16_t *>(&dh);
        }
        dst_qs[dst_b * 32 + lane] = static_cast<int8_t>(__float2int_rn(val * inv));
    };
    quant_store(k, k_cache);
    quant_store(v, v_cache);
}

__global__ void inkling2_kv_write_f16_kernel(
    const float *   __restrict__ k,
    const float *   __restrict__ v,
    __half *        __restrict__ k_cache,
    __half *        __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    int T, int kv_dim, int cache_cap) {

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (t >= T || d >= kv_dim) return;
    const int slot = pos[t] % cache_cap;
    const size_t src_off = static_cast<size_t>(t) * kv_dim + d;
    const size_t dst_off = static_cast<size_t>(slot) * kv_dim + d;
    k_cache[dst_off] = __float2half_rn(k[src_off]);
    v_cache[dst_off] = __float2half_rn(v[src_off]);
}

// ============================================================================
// 7. Split-K GQA FlashAttention with On-The-Fly Relative Bias + Zero-Value Sink!
//    head_dim = 128 (4 floats/lane per warp). Supports both SWA (window=1024) and Global!
// ============================================================================
// High-Occupancy 4-Warp/CTA GQA-grouped Split-K Decode Attention (GQA=3, 100% coalesced SoA KV):
// - 4 warps/CTA * 32 splits * 8 kv_heads = 1024 active warps across 56 SMs (~18 warps/SM)!
// - Each warp loads 128 contiguous bytes (32 x uint32_t) per K & V vector in ONE cache line transaction!
// - Shares each K & V load across all GQA=3 query heads (3x less HBM traffic)!
template <bool KV_Q8_0, int SPLITS, int GQA, int WARPS>
__global__ void inkling2_attn_splitk_gqa_kernel(
    const float *   __restrict__ q,          // [T, n_head, 128]
    const void *    __restrict__ k_cache,    // SoA Q8_0 or F16 [cache_cap, n_kv, 128]
    const void *    __restrict__ v_cache,    // SoA Q8_0 or F16 [cache_cap, n_kv, 128]
    const float *   __restrict__ bias_table, // [T, n_head, n_extent + 1]
    const int32_t * __restrict__ pos,        // [T]
    float *         __restrict__ part_m,     // [T, n_head, SPLITS]
    float *         __restrict__ part_l,     // [T, n_head, SPLITS]
    float *         __restrict__ part_o,     // [T, n_head, SPLITS, 128]
    int n_head, int n_kv, int swa_window, int n_extent, int cache_cap, float scale) {

    const int kvh  = blockIdx.x;
    const int s    = blockIdx.y;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x; // 0..31 -> 4 dims/lane
    const int warp = threadIdx.y; // 0..WARPS-1
    const int h0   = kvh * GQA;

    const int maxp  = pos[t];
    const int p_min = (swa_window > 0 && maxp >= swa_window) ? (maxp - swa_window + 1) : 0;
    const int total = maxp - p_min + 1;
    const int chunk = (total + SPLITS - 1) / SPLITS;
    const int p0    = p_min + s * chunk;
    const int p1    = (p0 + chunk < maxp + 1) ? (p0 + chunk) : (maxp + 1);

    float4 qv[GQA];
    const float * b_row[GQA];
    float  m_i[GQA];
    float  l_i[GQA];
    float4 ov[GQA];

    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        const int h = h0 + g;
        float4 qtmp = reinterpret_cast<const float4 *>(q + (static_cast<size_t>(t) * n_head + h) * 128)[lane];
        qv[g]    = make_float4(qtmp.x * scale, qtmp.y * scale, qtmp.z * scale, qtmp.w * scale);
        b_row[g] = bias_table + (static_cast<size_t>(t) * n_head + h) * (n_extent + 1);
        m_i[g]   = -FLT_MAX;
        l_i[g]   = 0.0f;
        ov[g]    = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const int blk_in_h    = lane >> 3; // 0..3
    const bool is_swa     = (swa_window > 0);
    const bool pow2_cap   = ((cache_cap & (cache_cap - 1)) == 0);
    const int  cap_mask   = cache_cap - 1;
    const size_t qs_bytes = static_cast<size_t>(cache_cap) * n_kv * 128;
    const auto * k_u32    = static_cast<const uint32_t *>(k_cache);
    const auto * v_u32    = static_cast<const uint32_t *>(v_cache);
    const auto * k_sc     = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(k_cache) + qs_bytes);
    const auto * v_sc     = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(v_cache) + qs_bytes);

    for (int p = p0 + warp; p < p1; p += WARPS) {
        const int slot = is_swa ? (pow2_cap ? (p & cap_mask) : (p % cache_cap)) : p;
        float4 kv, vv;
        if constexpr (KV_Q8_0) {
            const size_t head_base = static_cast<size_t>(slot) * n_kv + kvh;
            const uint32_t kq = __ldg(&k_u32[head_base * 32 + lane]);
            const uint32_t vq = __ldg(&v_u32[head_base * 32 + lane]);
            __half kdh, vdh;
            *reinterpret_cast<uint16_t *>(&kdh) = __ldg(&k_sc[head_base * 4 + blk_in_h]);
            *reinterpret_cast<uint16_t *>(&vdh) = __ldg(&v_sc[head_base * 4 + blk_in_h]);
            const float kd = __half2float(kdh);
            const float vd = __half2float(vdh);
            kv = make_float4(kd * static_cast<int8_t>(kq & 0xFFu),
                             kd * static_cast<int8_t>((kq >> 8) & 0xFFu),
                             kd * static_cast<int8_t>((kq >> 16) & 0xFFu),
                             kd * static_cast<int8_t>(kq >> 24));
            vv = make_float4(vd * static_cast<int8_t>(vq & 0xFFu),
                             vd * static_cast<int8_t>((vq >> 8) & 0xFFu),
                             vd * static_cast<int8_t>((vq >> 16) & 0xFFu),
                             vd * static_cast<int8_t>(vq >> 24));
        } else {
            const size_t off = (static_cast<size_t>(slot) * n_kv + kvh) * 32 + lane;
            uint2 pk = __ldg(&static_cast<const uint2 *>(k_cache)[off]);
            uint2 pv = __ldg(&static_cast<const uint2 *>(v_cache)[off]);
            float2 k0 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.x));
            float2 k1 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.y));
            float2 v0 = __half22float2(*reinterpret_cast<const __half2 *>(&pv.x));
            float2 v1 = __half22float2(*reinterpret_cast<const __half2 *>(&pv.y));
            kv = make_float4(k0.x, k0.y, k1.x, k1.y);
            vv = make_float4(v0.x, v0.y, v1.x, v1.y);
        }

        const int dist = maxp - p;
        const int bidx = (dist >= 0 && dist < n_extent) ? dist : n_extent;

        #pragma unroll
        for (int g = 0; g < GQA; ++g) {
            float dot = warp_sum(qv[g].x * kv.x + qv[g].y * kv.y + qv[g].z * kv.z + qv[g].w * kv.w);
            const float score = __shfl_sync(0xffffffffu, dot, 0) + __ldg(&b_row[g][bidx]);
            const float m_new = fmaxf(m_i[g], score);
            const float alpha = (m_i[g] == -FLT_MAX) ? 0.0f : __expf(m_i[g] - m_new);
            const float beta  = __expf(score - m_new);
            l_i[g]  = l_i[g] * alpha + beta;
            ov[g].x = ov[g].x * alpha + beta * vv.x;
            ov[g].y = ov[g].y * alpha + beta * vv.y;
            ov[g].z = ov[g].z * alpha + beta * vv.z;
            ov[g].w = ov[g].w * alpha + beta * vv.w;
            m_i[g]  = m_new;
        }
    }

    // Reduce WARPS warps in shared memory for each head g in 0..GQA-1
    __shared__ float  s_m[GQA][WARPS];
    __shared__ float  s_l[GQA][WARPS];
    __shared__ float4 s_o[GQA][WARPS][32];

    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        if (lane == 0) { s_m[g][warp] = m_i[g]; s_l[g][warp] = l_i[g]; }
        s_o[g][warp][lane] = ov[g];
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int g = 0; g < GQA; ++g) {
            float mm = -FLT_MAX;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) if (s_l[g][w] > 0.0f) mm = fmaxf(mm, s_m[g][w]);
            float ll = 0.0f;
            float4 oo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (mm > -FLT_MAX) {
                #pragma unroll
                for (int w = 0; w < WARPS; ++w) {
                    const float lw = s_l[g][w];
                    if (lw > 0.0f) {
                        const float sc = __expf(s_m[g][w] - mm);
                        ll += lw * sc;
                        float4 ow = s_o[g][w][lane];
                        oo.x += ow.x * sc; oo.y += ow.y * sc; oo.z += ow.z * sc; oo.w += ow.w * sc;
                    }
                }
            }
            const size_t slot = (static_cast<size_t>(t) * n_head + (h0 + g)) * SPLITS + s;
            if (lane == 0) { part_m[slot] = mm; part_l[slot] = ll; }
            reinterpret_cast<float4 *>(part_o + slot * 128)[lane] = oo;
        }
    }
}

template <bool KV_Q8_0, int SPLITS>
__global__ void inkling2_attn_splitk_kernel(
    const float *   __restrict__ q,          // [T, n_head, 128]
    const void *    __restrict__ k_cache,    // [cache_cap, n_kv, 128]
    const void *    __restrict__ v_cache,    // [cache_cap, n_kv, 128]
    const float *   __restrict__ bias_table, // [T, n_head, n_extent + 1]
    const int32_t * __restrict__ pos,        // [T]
    float *         __restrict__ part_m,     // [T, n_head, SPLITS]
    float *         __restrict__ part_l,     // [T, n_head, SPLITS]
    float *         __restrict__ part_o,     // [T, n_head, SPLITS, 128]
    int n_head, int n_kv, int swa_window, int n_extent, int cache_cap, float scale) {

    const int h    = blockIdx.x;
    const int s    = blockIdx.y;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x; // 0..31 -> 4 dims/lane (lane*4 .. lane*4+3)
    const int kvh  = h / (n_head / n_kv);

    const int maxp   = pos[t];
    const int p_min  = (swa_window > 0 && maxp >= swa_window) ? (maxp - swa_window + 1) : 0;
    const int total  = maxp - p_min + 1;
    const int chunk  = (total + SPLITS - 1) / SPLITS;
    const int p0     = p_min + s * chunk;
    const int p1     = (p0 + chunk < maxp + 1) ? (p0 + chunk) : (maxp + 1);

    const auto * q4 = reinterpret_cast<const float4 *>(q + (static_cast<size_t>(t) * n_head + h) * 128);
    float4 qv = q4[lane];
    qv.x *= scale; qv.y *= scale; qv.z *= scale; qv.w *= scale;

    const float * b_row = bias_table + (static_cast<size_t>(t) * n_head + h) * (n_extent + 1);

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float4 ov = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const int blk_in_h = lane >> 3;        // 0..3 (which of the 4 Q8_0 blocks in 128 dims)

    for (int p = p0; p < p1; ++p) {
        const int slot = p % cache_cap;
        float4 kv, vv;
        if constexpr (KV_Q8_0) {
            const size_t qs_bytes = static_cast<size_t>(cache_cap) * n_kv * 128;
            const size_t bidx     = (static_cast<size_t>(slot) * n_kv + kvh) * 4 + blk_in_h;
            const size_t w_off    = (static_cast<size_t>(slot) * n_kv + kvh) * 32 + lane;
            const uint32_t kq = __ldg(&static_cast<const uint32_t *>(k_cache)[w_off]);
            const uint32_t vq = __ldg(&static_cast<const uint32_t *>(v_cache)[w_off]);
            const auto * k_sc = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(k_cache) + qs_bytes);
            const auto * v_sc = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(v_cache) + qs_bytes);
            __half kdh, vdh;
            *reinterpret_cast<uint16_t *>(&kdh) = __ldg(&k_sc[bidx]);
            *reinterpret_cast<uint16_t *>(&vdh) = __ldg(&v_sc[bidx]);
            const float kd = __half2float(kdh);
            const float vd = __half2float(vdh);
            kv = make_float4(kd * static_cast<int8_t>(kq & 0xFFu),
                             kd * static_cast<int8_t>((kq >> 8) & 0xFFu),
                             kd * static_cast<int8_t>((kq >> 16) & 0xFFu),
                             kd * static_cast<int8_t>(kq >> 24));
            vv = make_float4(vd * static_cast<int8_t>(vq & 0xFFu),
                             vd * static_cast<int8_t>((vq >> 8) & 0xFFu),
                             vd * static_cast<int8_t>((vq >> 16) & 0xFFu),
                             vd * static_cast<int8_t>(vq >> 24));
        } else {
            const size_t off = (static_cast<size_t>(slot) * n_kv + kvh) * 32 + lane;
            uint2 pk = __ldg(&static_cast<const uint2 *>(k_cache)[off]);
            uint2 pv = __ldg(&static_cast<const uint2 *>(v_cache)[off]);
            float2 k0 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.x));
            float2 k1 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.y));
            float2 v0 = __half22float2(*reinterpret_cast<const __half2 *>(&pv.x));
            float2 v1 = __half22float2(*reinterpret_cast<const __half2 *>(&pv.y));
            kv = make_float4(k0.x, k0.y, k1.x, k1.y);
            vv = make_float4(v0.x, v0.y, v1.x, v1.y);
        }

        float dot = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w);
        const int dist = maxp - p;
        const int bidx = (dist >= 0 && dist < n_extent) ? dist : n_extent;
        const float score = __shfl_sync(0xffffffffu, dot, 0) + __ldg(&b_row[bidx]);

        const float m_new = fmaxf(m_i, score);
        const float alpha = (m_i == -FLT_MAX) ? 0.0f : expf(m_i - m_new);
        const float beta  = expf(score - m_new);
        l_i = l_i * alpha + beta;
        ov.x = ov.x * alpha + beta * vv.x;
        ov.y = ov.y * alpha + beta * vv.y;
        ov.z = ov.z * alpha + beta * vv.z;
        ov.w = ov.w * alpha + beta * vv.w;
        m_i = m_new;
    }

    const size_t slot = (static_cast<size_t>(t) * n_head + h) * SPLITS + s;
    if (lane == 0) {
        part_m[slot] = m_i;
        part_l[slot] = l_i;
    }
    reinterpret_cast<float4 *>(part_o + slot * 128)[lane] = ov;
}

// Split-K Reduce WITH zero-value attention sink (sink logit = 0.0f, sink value = 0.0f)!
template <int SPLITS>
__global__ void inkling2_attn_reduce_sink_kernel(
    const float * __restrict__ part_m,
    const float * __restrict__ part_l,
    const float * __restrict__ part_o,
    float *       __restrict__ out,
    int n_head) {

    const int h    = blockIdx.x;
    const int t    = blockIdx.y;
    const int lane = threadIdx.x;
    const size_t base = (static_cast<size_t>(t) * n_head + h) * SPLITS;

    // Virtual sink token has logit 0.0f -> initialize m_max = 0.0f!
    float m_max = 0.0f;
    for (int s = 0; s < SPLITS; ++s) {
        if (part_l[base + s] > 0.0f) m_max = fmaxf(m_max, part_m[base + s]);
    }

    float l_tot = expf(0.0f - m_max); // sink weight!
    float4 ov = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < SPLITS; ++s) {
        const float ls = part_l[base + s];
        if (ls <= 0.0f) continue;
        const float sc = expf(part_m[base + s] - m_max);
        l_tot += ls * sc;
        float4 po = reinterpret_cast<const float4 *>(part_o + (base + s) * 128)[lane];
        ov.x += po.x * sc; ov.y += po.y * sc; ov.z += po.z * sc; ov.w += po.w * sc;
    }

    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * n_head + h) * 128)[lane] =
        make_float4(ov.x * inv_l, ov.y * inv_l, ov.z * inv_l, ov.w * inv_l);
}

// ============================================================================
// 7B. Tiled FlashAttention Prefill (T >= 8): F16 Tensor Core (sm>=70) or F32 SGEMM (<sm_70)
// ============================================================================
template <typename TMat>
__device__ __forceinline__ TMat ink2_f32_to_mat(float v) {
    if constexpr (std::is_same_v<TMat, __half>) return __float2half_rn(v);
    else return v;
}

template <typename TMat>
__device__ __forceinline__ TMat ink2_f16_to_mat(__half h) {
    if constexpr (std::is_same_v<TMat, __half>) return h;
    else return __half2float(h);
}

template <typename TMat>
__global__ void inkling2_q_to_kvh_kernel(
    const float * __restrict__ q,  // [T, n_head, 128]
    TMat *        __restrict__ qm, // [n_kv, T * gqa, 128]
    int T, int n_head, int n_kv, float scale) {
    const int gqa = n_head / n_kv;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = T * n_head * 128;
    if (idx >= tot) return;
    const int d   = idx & 127;
    const int h   = (idx >> 7) % n_head;
    const int t   = idx / (n_head * 128);
    const int kvh = h / gqa;
    const int gh  = h % gqa;
    const size_t dst = ((static_cast<size_t>(kvh) * (T * gqa) + t * gqa + gh) * 128) + d;
    qm[dst] = ink2_f32_to_mat<TMat>(q[idx] * scale);
}

template <bool KV_Q8_0, typename TMat>
__global__ void inkling2_dequant_kv_tile_kernel(
    const void * __restrict__ k_cache,
    const void * __restrict__ v_cache,
    TMat *       __restrict__ km, // [n_kv, kc, 128]
    TMat *       __restrict__ vm, // [n_kv, kc, 128]
    int k0, int kc, int n_kv, int cache_cap) {
    const int kvh = blockIdx.x;
    const int c   = blockIdx.y;
    if (c >= kc) return;
    const int slot = (k0 + c) % cache_cap;
    const size_t dst_base = (static_cast<size_t>(kvh) * kc + c) * 128;
    if constexpr (KV_Q8_0) {
        const int b    = threadIdx.y; // 0..3
        const int lane = threadIdx.x; // 0..31
        const size_t qs_bytes = static_cast<size_t>(cache_cap) * n_kv * 128;
        const size_t bidx     = (static_cast<size_t>(slot) * n_kv + kvh) * 4 + b;
        const auto * k_qs = static_cast<const int8_t *>(k_cache);
        const auto * k_sc = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(k_cache) + qs_bytes);
        const auto * v_qs = static_cast<const int8_t *>(v_cache);
        const auto * v_sc = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(v_cache) + qs_bytes);
        __half kdh, vdh;
        *reinterpret_cast<uint16_t *>(&kdh) = __ldg(&k_sc[bidx]);
        *reinterpret_cast<uint16_t *>(&vdh) = __ldg(&v_sc[bidx]);
        km[dst_base + b * 32 + lane] = ink2_f32_to_mat<TMat>(__half2float(kdh) * static_cast<float>(__ldg(&k_qs[bidx * 32 + lane])));
        vm[dst_base + b * 32 + lane] = ink2_f32_to_mat<TMat>(__half2float(vdh) * static_cast<float>(__ldg(&v_qs[bidx * 32 + lane])));
    } else {
        const int d = threadIdx.x; // 0..127
        const size_t src = (static_cast<size_t>(slot) * n_kv + kvh) * 128 + d;
        km[dst_base + d] = ink2_f16_to_mat<TMat>(static_cast<const __half *>(k_cache)[src]);
        vm[dst_base + d] = ink2_f16_to_mat<TMat>(static_cast<const __half *>(v_cache)[src]);
    }
}

__global__ void inkling2_init_flash_sink_kernel(float * m, float * l, int M) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) { m[i] = 0.0f; l[i] = 1.0f; } // virtual sink logit=0 -> m=0, l=exp(0)=1!
}

// 1 block of 128 threads per row r in [0 .. n_kv * T * gqa)
template <typename TMat>
__global__ void inkling2_flash_softmax_relbias_rescale_kernel(
    const float *   __restrict__ s_tile,     // [n_kv, T*gqa, kc]
    TMat *          __restrict__ p_tile,     // [n_kv, T*gqa, kc]
    const float *   __restrict__ bias_table, // [T, n_head, n_extent+1]
    const int32_t * __restrict__ pos,        // [T]
    float *         __restrict__ m_buf,      // [n_kv * T * gqa]
    float *         __restrict__ l_buf,      // [n_kv * T * gqa]
    float *         __restrict__ o_buf,      // [n_kv, T*gqa, 128]
    int T, int n_head, int n_kv, int k0, int kc, int swa_window, int n_extent, bool is_first_tile) {

    const int gqa  = n_head / n_kv;
    const int r    = blockIdx.x;
    const int kvh  = r / (T * gqa);
    const int rem  = r - kvh * (T * gqa);
    const int t    = rem / gqa;
    const int gh   = rem - t * gqa;
    const int h    = kvh * gqa + gh;
    const int maxp = pos[t];

    const int tid  = threadIdx.x; // 0..127
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float * s_row = s_tile + static_cast<size_t>(r) * kc;
    TMat *        p_row = p_tile + static_cast<size_t>(r) * kc;
    const float * b_row = bias_table + (static_cast<size_t>(t) * n_head + h) * (n_extent + 1);
    float *       o_row = o_buf + static_cast<size_t>(r) * 128;

    const int k_lo    = (swa_window > 0 && maxp >= swa_window) ? (maxp - swa_window + 1) : 0;
    const int c_start = (k_lo > k0) ? (k_lo - k0) : 0;
    const int c_end   = (k0 + kc - 1 <= maxp) ? kc : (maxp - k0 + 1);

    if (c_start >= c_end) {
        if (is_first_tile) o_row[tid] = 0.0f;
        for (int c = tid; c < kc; c += blockDim.x) p_row[c] = ink2_f32_to_mat<TMat>(0.0f);
        return;
    }

    float local_m = -FLT_MAX;
    for (int c = c_start + tid; c < c_end; c += blockDim.x) {
        const int dist = maxp - (k0 + c);
        const int bi   = (dist < n_extent) ? dist : n_extent;
        local_m = fmaxf(local_m, s_row[c] + __ldg(&b_row[bi]));
    }
    local_m = warp_max(local_m);
    __shared__ float s_red[4];
    if (lane == 0) s_red[warp] = local_m;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 4) ? s_red[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    const float m_tile = s_red[0];

    const float m_old = m_buf[r];
    const float m_new = fmaxf(m_old, m_tile);
    const float alpha = expf(m_old - m_new);

    float local_l = 0.0f;
    for (int c = tid; c < kc; c += blockDim.x) {
        if (c >= c_start && c < c_end) {
            const int dist = maxp - (k0 + c);
            const int bi   = (dist < n_extent) ? dist : n_extent;
            const float e = expf(s_row[c] + __ldg(&b_row[bi]) - m_new);
            p_row[c] = ink2_f32_to_mat<TMat>(e);
            local_l += e;
        } else {
            p_row[c] = ink2_f32_to_mat<TMat>(0.0f);
        }
    }
    local_l = warp_sum(local_l);
    if (lane == 0) s_red[warp] = local_l;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 4) ? s_red[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) {
            m_buf[r] = m_new;
            l_buf[r] = l_buf[r] * alpha + v;
        }
    }

    if (is_first_tile) {
        o_row[tid] = 0.0f;
    } else if (alpha != 1.0f) {
        o_row[tid] *= alpha;
    }
}

__global__ void inkling2_o_from_kvh_finalize_kernel(
    const float * __restrict__ o_buf, // [n_kv, T*gqa, 128]
    const float * __restrict__ l_buf, // [n_kv * T * gqa]
    float *       __restrict__ d_out, // [T, n_head, 128]
    int T, int n_head, int n_kv) {
    const int gqa = n_head / n_kv;
    const int r   = blockIdx.x;
    const int d   = threadIdx.x; // 0..127
    const int kvh = r / (T * gqa);
    const int rem = r - kvh * (T * gqa);
    const int t   = rem / gqa;
    const int gh  = rem - t * gqa;
    const int h   = kvh * gqa + gh;
    const float inv_l = (l_buf[r] > 0.0f) ? (1.0f / l_buf[r]) : 0.0f;
    d_out[(static_cast<size_t>(t) * n_head + h) * 128 + d] = o_buf[static_cast<size_t>(r) * 128 + d] * inv_l;
}

// ============================================================================
// 8. Sigmoid Gate Mul & Logit Softcap
// ============================================================================
__global__ void inkling2_sigmoid_gate_mul_kernel(float * __restrict__ x, const float * __restrict__ g, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-g[i]));
}

__global__ void inkling2_logit_softcap_kernel(float * __restrict__ x, int64_t n, float cap, float inv_cap) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] * inv_cap);
}

// ============================================================================
// 9. Ultra-Fast Q8_0 SoA LDG.128 + dp4a MoE with Fused Gate+Up+SiTU (Decode + Grouped Prefill!)
// ============================================================================
__device__ __forceinline__ float situ_act(float g, float u, float beta, float lb) {
    const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g)));
    const float b = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
    return a * b;
}

// Decode (T < 16): fused gate+up+SiTU on packed gate_up_exps[n_embd, 2*ff, n_expert]
__global__ void inkling2_moe_gate_up_situ_decode_kernel(
    const int4 *          __restrict__ gu_q,     // [n_expert, 2*ff, K/16]
    const uint16_t *      __restrict__ gu_s,     // [n_expert, 2*ff, K/32]
    const BlockQ8_1_Act * __restrict__ x_q8,     // [T, K/32]
    const int32_t *       __restrict__ top_ids,  // [T, k_used]
    float *               __restrict__ act,      // [T, k_used, ff]
    int K_blocks, int ff, int k_used, float beta, float lb) {

    extern __shared__ BlockQ8_1_Act s_x[];
    const int t    = blockIdx.z;
    const int slot = blockIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;

    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * src = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(t) * K_blocks);
    int4 *       dst = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) dst[i] = src[i];
    __syncthreads();

    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= ff) return;
    const int lane = threadIdx.x;
    const int eid  = top_ids[t * k_used + slot];

    const size_t exp_stride = static_cast<size_t>(2 * ff) * K_blocks;
    const size_t g_off      = static_cast<size_t>(eid) * exp_stride + static_cast<size_t>(row) * K_blocks;
    const size_t u_off      = g_off + static_cast<size_t>(ff) * K_blocks;

    const int4 *     gq_row = gu_q + g_off * 2;
    const uint16_t * gs_row = gu_s + g_off;
    const int4 *     uq_row = gu_q + u_off * 2;
    const uint16_t * us_row = gu_s + u_off;

    const int blk_in_warp = lane >> 1;
    const int half_in_blk = lane & 1;

    float ag = 0.0f, au = 0.0f;
    for (int b = blk_in_warp; b < K_blocks; b += 16) {
        const int4  x4 = reinterpret_cast<const int4 *>(s_x[b].qs)[half_in_blk];
        const float xd = s_x[b].d;
        {
            const int4 gw = __ldg(&gq_row[b * 2 + half_in_blk]);
            int sg = dp4a_s32(gw.x, x4.x, dp4a_s32(gw.y, x4.y, dp4a_s32(gw.z, x4.z, dp4a_s32(gw.w, x4.w, 0))));
            __half dh; *reinterpret_cast<uint16_t *>(&dh) = __ldg(&gs_row[b]);
            ag += (__half2float(dh) * xd) * static_cast<float>(sg);
        }
        {
            const int4 uw = __ldg(&uq_row[b * 2 + half_in_blk]);
            int su = dp4a_s32(uw.x, x4.x, dp4a_s32(uw.y, x4.y, dp4a_s32(uw.z, x4.z, dp4a_s32(uw.w, x4.w, 0))));
            __half dh; *reinterpret_cast<uint16_t *>(&dh) = __ldg(&us_row[b]);
            au += (__half2float(dh) * xd) * static_cast<float>(su);
        }
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) {
        act[(static_cast<size_t>(t) * k_used + slot) * ff + row] = situ_act(ag, au, beta, lb);
    }
}

// Decode (T < 16): streaming 1-expert-at-a-time down projection (1.7 KB smem -> 100% occupancy!)
__global__ void inkling2_moe_down_decode_kernel(
    const int4 *          __restrict__ dn_q,     // [n_expert, n_embd, ff/16]
    const uint16_t *      __restrict__ dn_s,     // [n_expert, n_embd, ff/32]
    const BlockQ8_1_Act * __restrict__ act_q8,   // [T, k_used, ff/32]
    const int32_t *       __restrict__ top_ids,  // [T, k_used]
    const float *         __restrict__ top_w,    // [T, k_used]
    float *               __restrict__ out,      // [T, n_embd]
    int ff_blocks, int n_embd, int k_used) {

    extern __shared__ BlockQ8_1_Act s_a[];
    const int t    = blockIdx.y;
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const int lane = threadIdx.x;
    const int blk_in_warp = lane >> 1;
    const int half_in_blk = lane & 1;
    const int n_i4 = (ff_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;

    float total = 0.0f;
    for (int slot = 0; slot < k_used; ++slot) {
        const int4 * src = reinterpret_cast<const int4 *>(act_q8 + (static_cast<size_t>(t) * k_used + slot) * ff_blocks);
        int4 *       dst = reinterpret_cast<int4 *>(s_a);
        for (int i = tid; i < n_i4; i += nthr) dst[i] = src[i];
        __syncthreads();

        if (row < n_embd) {
            const int   eid = top_ids[t * k_used + slot];
            const float ew  = top_w  [t * k_used + slot];
            const size_t row_off = (static_cast<size_t>(eid) * n_embd + row) * ff_blocks;
            const int4 *     dq = dn_q + row_off * 2;
            const uint16_t * ds = dn_s + row_off;

            float acc = 0.0f;
            for (int b = blk_in_warp; b < ff_blocks; b += 16) {
                const int4 w4 = __ldg(&dq[b * 2 + half_in_blk]);
                const int4 a4 = reinterpret_cast<const int4 *>(s_a[b].qs)[half_in_blk];
                int si = dp4a_s32(w4.x, a4.x, dp4a_s32(w4.y, a4.y, dp4a_s32(w4.z, a4.z, dp4a_s32(w4.w, a4.w, 0))));
                __half dh; *reinterpret_cast<uint16_t *>(&dh) = __ldg(&ds[b]);
                acc += (__half2float(dh) * s_a[b].d) * static_cast<float>(si);
            }
            total += ew * warp_sum(acc);
        }
        __syncthreads();
    }
    if (row < n_embd && lane == 0) out[static_cast<size_t>(t) * n_embd + row] = total;
}

// Prefill (T >= 16): GPU counting sort + Grouped-Expert Q8_0 SoA LDG.128+dp4a (BT=8, 8x weight reuse!)
struct Ink2TileDesc {
    int32_t eid;
    int32_t start;
    int32_t count;
};

__device__ __forceinline__ float ink2_half_bits_to_f32(uint16_t bits) {
    __half h;
    *reinterpret_cast<uint16_t *>(&h) = bits;
    return __half2float(h);
}

__device__ __forceinline__ __half2 ink2_hfma2(__half2 a, __half2 b, __half2 c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 600 || __CUDA_ARCH__ == 620 || __CUDA_ARCH__ >= 700)
    return __hfma2(a, b, c);
#else
    float2 fa = __half22float2(a);
    float2 fb = __half22float2(b);
    float2 fc = __half22float2(c);
    return __floats2half2_rn(fa.x * fb.x + fc.x, fa.y * fb.y + fc.y);
#endif
}

__device__ __forceinline__ __half2 ink2_hadd2(__half2 a, __half2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hadd2(a, b);
#else
    const float2 fa = __half22float2(a), fb = __half22float2(b);
    return __floats2half2_rn(fa.x + fb.x, fa.y + fb.y);
#endif
}

__device__ __forceinline__ __half2 ink2_hmul2(__half2 a, __half2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hmul2(a, b);
#else
    const float2 fa = __half22float2(a), fb = __half22float2(b);
    return __floats2half2_rn(fa.x * fb.x, fa.y * fb.y);
#endif
}

// Fast exact int8x2 -> half2 via IEEE-754 FP16 1024.0f (0x6400) exponent bit-injection:
// Given u = w_i8x4 ^ 0x80808080u (unsigned 0..255 = w + 128),
// (0x6400 | u_i) is exact FP16 (1024.0 + u_i) = (1152.0 + w_i).
__device__ __forceinline__ __half2 ink2_i8x2_to_h2_lo(uint32_t u) {
    uint32_t p = __byte_perm(u, 0x00006400u, 0x5150);
    return *reinterpret_cast<const __half2 *>(&p);
}

__device__ __forceinline__ __half2 ink2_i8x2_to_h2_hi(uint32_t u) {
    uint32_t p = __byte_perm(u, 0x00006400u, 0x5352);
    return *reinterpret_cast<const __half2 *>(&p);
}

// Contiguous half2 quantization for Q8_0 on sm_60:
// Packs 32 floats -> 16 contiguous __half2 (x[2i], x[2i+1]) / amax + float d = amax.
__global__ void inkling2_quantize_f16_contig_kernel(
    const float * __restrict__ x,
    __half2 *     __restrict__ out_h2,
    float *       __restrict__ out_d,
    int total_blocks) {
    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= total_blocks) return;
    const int lane = threadIdx.x;
    const float v = x[static_cast<size_t>(b) * 32 + lane];
    const float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(v)), 0);
    const float inv  = (amax > 0.0f) ? (1.0f / amax) : 0.0f;
    const float vn   = v * inv;
    const float vn_hi = __shfl_down_sync(0xffffffffu, vn, 1);
    if ((lane & 1) == 0) {
        out_h2[static_cast<size_t>(b) * 16 + (lane >> 1)] = __halves2half2(__float2half_rn(vn), __float2half_rn(vn_hi));
    }
    if (lane == 0) out_d[b] = amax;
}

__global__ void inkling2_group_moe_tokens_kernel(
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

template <int BT, int R_PER_WARP>
__global__ void inkling2_moe_gate_up_grouped_kernel(
    const int4 *          __restrict__ gu_q,
    const uint16_t *      __restrict__ gu_s,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const Ink2TileDesc *  __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ act,
    int K_blocks, int ff, int k_used, float beta, float lb) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Ink2TileDesc td = tiles[tile_idx];
    const int eid   = td.eid;
    const int start = td.start;
    const int end   = td.start + td.count;

    extern __shared__ BlockQ8_1_Act s_x[]; // [BT, K_blocks] (8 * 3.45 KB = 27.6 KB!)
    __shared__ int32_t s_pairs[BT];
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int lane     = threadIdx.x;
    const int row_base = blockIdx.x * (blockDim.y * R_PER_WARP) + threadIdx.y;
    const int blk_in_w = lane >> 1;
    const int half_in  = lane & 1;
    const int n_i4_tok = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) s_pairs[tid] = sorted_pairs[base_i + tid];
        __syncthreads();

        for (int j = 0; j < n_tile; ++j) {
            const int tok = s_pairs[j] >> 4;
            const int4 * src = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(tok) * K_blocks);
            int4 *       dst = reinterpret_cast<int4 *>(s_x + j * K_blocks);
            for (int i = tid; i < n_i4_tok; i += nthr) dst[i] = src[i];
        }
        __syncthreads();

        const size_t exp_stride = static_cast<size_t>(2 * ff) * K_blocks;
        for (int rr = 0; rr < R_PER_WARP; ++rr) {
            const int row = row_base + rr * blockDim.y;
            if (row >= ff) break;
            const size_t g_off = static_cast<size_t>(eid) * exp_stride + static_cast<size_t>(row) * K_blocks;
            const size_t u_off = g_off + static_cast<size_t>(ff) * K_blocks;
            const int4 *     gq = gu_q + g_off * 2;
            const uint16_t * gs = gu_s + g_off;
            const int4 *     uq = gu_q + u_off * 2;
            const uint16_t * us = gu_s + u_off;

            float ag[BT], au[BT];
            #pragma unroll
            for (int j = 0; j < BT; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

            for (int b = blk_in_w; b < K_blocks; b += 16) {
                const int4 gw = __ldg(&gq[b * 2 + half_in]);
                const int4 uw = __ldg(&uq[b * 2 + half_in]);
                __half gdh, udh;
                *reinterpret_cast<uint16_t *>(&gdh) = __ldg(&gs[b]);
                *reinterpret_cast<uint16_t *>(&udh) = __ldg(&us[b]);
                const float gd = __half2float(gdh), ud = __half2float(udh);

                #pragma unroll
                for (int j = 0; j < BT; ++j) {
                    if (j < n_tile) {
                        const BlockQ8_1_Act & xb = s_x[j * K_blocks + b];
                        const int4 x4 = reinterpret_cast<const int4 *>(xb.qs)[half_in];
                        int sg = dp4a_s32(gw.x, x4.x, dp4a_s32(gw.y, x4.y, dp4a_s32(gw.z, x4.z, dp4a_s32(gw.w, x4.w, 0))));
                        int su = dp4a_s32(uw.x, x4.x, dp4a_s32(uw.y, x4.y, dp4a_s32(uw.z, x4.z, dp4a_s32(uw.w, x4.w, 0))));
                        ag[j] += (gd * xb.d) * static_cast<float>(sg);
                        au[j] += (ud * xb.d) * static_cast<float>(su);
                    }
                }
            }

            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                if (j < n_tile) {
                    float g = warp_sum(ag[j]);
                    float u = warp_sum(au[j]);
                    if (lane == 0) {
                        const int pair = s_pairs[j];
                        act[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * ff + row] = situ_act(g, u, beta, lb);
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <int BT, int R_PER_WARP>
__global__ void inkling2_moe_down_grouped_kernel(
    const int4 *          __restrict__ dn_q,
    const uint16_t *      __restrict__ dn_s,
    const BlockQ8_1_Act * __restrict__ act_q8,
    const float *         __restrict__ top_w,
    const Ink2TileDesc *  __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots, // [T, k_used, n_embd] (deterministic!)
    int ff_blocks, int n_embd, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Ink2TileDesc td = tiles[tile_idx];
    const int eid   = td.eid;
    const int start = td.start;
    const int end   = td.start + td.count;

    extern __shared__ BlockQ8_1_Act s_a[]; // [BT, ff_blocks] (12 * 1.72 KB = 20.7 KB!)
    __shared__ int32_t s_pairs[BT];
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int lane     = threadIdx.x;
    const int row_base = blockIdx.x * (blockDim.y * R_PER_WARP) + threadIdx.y;
    const int blk_in_w = lane >> 1;
    const int half_in  = lane & 1;
    const int n_i4_tok = (ff_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) s_pairs[tid] = sorted_pairs[base_i + tid];
        __syncthreads();

        for (int j = 0; j < n_tile; ++j) {
            const int pair = s_pairs[j];
            const int4 * src = reinterpret_cast<const int4 *>(
                act_q8 + (static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * ff_blocks);
            int4 * dst = reinterpret_cast<int4 *>(s_a + j * ff_blocks);
            for (int i = tid; i < n_i4_tok; i += nthr) dst[i] = src[i];
        }
        __syncthreads();

        for (int rr = 0; rr < R_PER_WARP; ++rr) {
            const int row = row_base + rr * blockDim.y;
            if (row >= n_embd) break;
            const size_t row_off = (static_cast<size_t>(eid) * n_embd + row) * ff_blocks;
            const int4 *     dq = dn_q + row_off * 2;
            const uint16_t * ds = dn_s + row_off;

            float acc[BT];
            #pragma unroll
            for (int j = 0; j < BT; ++j) acc[j] = 0.0f;

            for (int b = blk_in_w; b < ff_blocks; b += 16) {
                const int4 w4 = __ldg(&dq[b * 2 + half_in]);
                __half dh; *reinterpret_cast<uint16_t *>(&dh) = __ldg(&ds[b]);
                const float wd = __half2float(dh);
                #pragma unroll
                for (int j = 0; j < BT; ++j) {
                    if (j < n_tile) {
                        const BlockQ8_1_Act & xb = s_a[j * ff_blocks + b];
                        const int4 a4 = reinterpret_cast<const int4 *>(xb.qs)[half_in];
                        int si = dp4a_s32(w4.x, a4.x, dp4a_s32(w4.y, a4.y, dp4a_s32(w4.z, a4.z, dp4a_s32(w4.w, a4.w, 0))));
                        acc[j] += (wd * xb.d) * static_cast<float>(si);
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
                        down_slots[(static_cast<size_t>(tok) * k_used + slot) * n_embd + row] = ew * v;
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ void inkling2_reduce_slots_kernel(
    const float * __restrict__ slots, float * __restrict__ out, int n_embd, int k_used) {
    if ((n_embd & 3) == 0) {
        const int K4 = n_embd >> 2;
        const int i  = blockIdx.x * blockDim.x + threadIdx.x;
        const int t  = blockIdx.y;
        if (i >= K4) return;
        const float4 * base = reinterpret_cast<const float4 *>(slots + static_cast<size_t>(t) * k_used * n_embd) + i;
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int s = 0; s < k_used; ++s) {
            const float4 v = base[static_cast<size_t>(s) * K4];
            sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
        reinterpret_cast<float4 *>(out + static_cast<size_t>(t) * n_embd)[i] = sum;
        return;
    }
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= n_embd) return;
    const float * base = slots + static_cast<size_t>(t) * k_used * n_embd + d;
    float sum = 0.0f;
    for (int s = 0; s < k_used; ++s) sum += base[static_cast<size_t>(s) * n_embd];
    out[static_cast<size_t>(t) * n_embd + d] = sum;
}



// Flat tile worklist builder: turns exp_offsets[n_expert+1] into flat array of
// packed tiles ((eid << 20) | (tile_start << 6) | n_tile) so ALL hot-expert BT-token
// tiles run in parallel across all SMs with ZERO serial stragglers!

__global__ void inkling2_build_tile_worklist_kernel(
    const int32_t * __restrict__ exp_offsets,
    Ink2TileDesc *  __restrict__ tiles,
    int32_t *       __restrict__ n_tiles_out,
    int n_expert, int BT) {
    if (threadIdx.x != 0) return;
    int nt = 0;
    for (int e = 0; e < n_expert; ++e) {
        const int s = exp_offsets[e];
        const int en = exp_offsets[e + 1];
        for (int p = s; p < en; p += BT) {
            const int cnt = (p + BT < en) ? BT : (en - p);
            tiles[nt++] = {e, p, cnt};
        }
    }
    *n_tiles_out = nt;
}

// ============================================================================
// 9B. Ultra-Fast IQ4_NL SoA MoE (Decode + Grouped Prefill):
//     - sm_60 (GP100): native 2x-rate FP16 half2 HFMA2 + 1 KB s_lut256 (ZERO dp4a emulation!)
//     - sm_61+ / other: coalesced LDG.128 + shared 16B LUT + hardware dp4a!
// ============================================================================

__global__ void inkling2_quantize_f16_pair_kernel(
    const float * __restrict__ x,      // [total_blocks * 32]
    __half2 *     __restrict__ out_h2, // [total_blocks * 16] interleaved (x[i], x[i+16]) / amax
    float *       __restrict__ out_d,  // [total_blocks] amax
    int total_blocks) {

    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= total_blocks) return;
    const int lane = threadIdx.x;
    const float v = x[static_cast<size_t>(b) * 32 + lane];
    const float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(v)), 0);
    const float inv  = (amax > 0.0f) ? (1.0f / amax) : 0.0f;
    const float vn   = v * inv;
    const float vn_hi = __shfl_down_sync(0xffffffffu, vn, 16);
    if (lane < 16) {
        out_h2[static_cast<size_t>(b) * 16 + lane] = __halves2half2(__float2half_rn(vn), __float2half_rn(vn_hi));
    }
    if (lane == 0) {
        out_d[b] = amax;
    }
}


__device__ __forceinline__ float ink2_dot_iq4_nl_blk_h2(
    int4 w4, const __half2 * __restrict__ s_lut256, const int4 * __restrict__ x_i4) {
    __half2 acc2 = __floats2half2_rn(0.0f, 0.0f);
    auto step4 = [&](uint32_t word, int4 xv) {
        const __half2 * xh = reinterpret_cast<const __half2 *>(&xv);
        acc2 = ink2_hfma2(s_lut256[(word >>  0) & 0xFFu], xh[0], acc2);
        acc2 = ink2_hfma2(s_lut256[(word >>  8) & 0xFFu], xh[1], acc2);
        acc2 = ink2_hfma2(s_lut256[(word >> 16) & 0xFFu], xh[2], acc2);
        acc2 = ink2_hfma2(s_lut256[(word >> 24) & 0xFFu], xh[3], acc2);
    };
    step4(static_cast<uint32_t>(w4.x), x_i4[0]);
    step4(static_cast<uint32_t>(w4.y), x_i4[1]);
    step4(static_cast<uint32_t>(w4.z), x_i4[2]);
    step4(static_cast<uint32_t>(w4.w), x_i4[3]);
    const float2 f2 = __half22float2(acc2);
    return f2.x + f2.y;
}

// Zero-bank-conflict warp-shuffle 16-entry FP32 LUT IQ4_NL block dot product:
// Avoids random-index shared-memory bank conflicts on real weights (382 GB/s vs 248 GB/s on P100!).
__device__ __forceinline__ float ink2_dot_iq4_nl_blk_shfl_f32(
    int4 w4, float my_kv, const int4 * __restrict__ x_i4) {
    float a0 = 0.0f, a1 = 0.0f;
    auto step4 = [&](uint32_t word, int4 xv) {
        const __half2 * xh = reinterpret_cast<const __half2 *>(&xv);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t b = (word >> (k * 8)) & 0xFFu;
            const float2 xf = __half22float2(xh[k]);
            a0 = fmaf(__shfl_sync(0xffffffffu, my_kv, b & 15), xf.x, a0);
            a1 = fmaf(__shfl_sync(0xffffffffu, my_kv, b >> 4), xf.y, a1);
        }
    };
    step4(static_cast<uint32_t>(w4.x), x_i4[0]);
    step4(static_cast<uint32_t>(w4.y), x_i4[1]);
    step4(static_cast<uint32_t>(w4.z), x_i4[2]);
    step4(static_cast<uint32_t>(w4.w), x_i4[3]);
    return a0 + a1;
}

__device__ __forceinline__ void ink2_init_lut256(__half2 * s_lut256, int tid, int nthr) {
    for (int i = tid; i < 256; i += nthr) {
        const float lo = static_cast<float>(kvalues_iq4nl_d[i & 0xF]);
        const float hi = static_cast<float>(kvalues_iq4nl_d[i >> 4]);
        s_lut256[i] = __floats2half2_rn(lo, hi);
    }
}

// Decode (T < 16) sm_60: 16 lanes/row (2 rows/warp, 16 rows/CTA -> 256B coalesced half-warp loads!)
// Decode (T < 16) sm_60: half2 s_lut256 (wpb=8, LPR=8 -> 32 rows/CTA = 1 block!)
// Fuses SiTU activation + inline quantize_f16_pair directly into (act_h2, act_d) -> ZERO intermediate act traffic!
__global__ void inkling2_moe_gate_up_situ_iq4_nl_soa_f16_kernel(
    const int4 *     __restrict__ gu_q,
    const uint16_t * __restrict__ gu_s,
    const __half2 *  __restrict__ x_h2,
    const float *    __restrict__ x_d,
    const int32_t *  __restrict__ top_ids,
    __half2 *        __restrict__ act_h2,
    float *          __restrict__ act_d,
    int K_blocks, int ff, int k_used, float beta, float lb) {

    extern __shared__ uint8_t smem_raw[];
    __shared__ __half2 s_lut256[256];
    __shared__ float s_act[32];
    auto * s_xh2 = reinterpret_cast<int4 *>(smem_raw);
    auto * s_xd  = reinterpret_cast<float *>(s_xh2 + K_blocks * 4);

    const int slot = blockIdx.y;
    const int t    = blockIdx.z;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;

    ink2_init_lut256(s_lut256, tid, nthr);
    const int4 * src_h2 = reinterpret_cast<const int4 *>(x_h2 + static_cast<size_t>(t) * K_blocks * 16);
    const int n_i4 = K_blocks * 4;
    for (int i = tid; i < n_i4; i += nthr) s_xh2[i] = src_h2[i];
    const float * src_d = x_d + static_cast<size_t>(t) * K_blocks;
    for (int i = tid; i < K_blocks; i += nthr) s_xd[i] = src_d[i];
    __syncthreads();

    const int lane     = threadIdx.x;
    const int sub_r    = lane >> 3;
    const int sub_l    = lane & 7;
    const int r_in_cta = threadIdx.y * 4 + sub_r;
    const int row      = blockIdx.x * 32 + r_in_cta;

    if (row < ff) {
        const int eid = top_ids[static_cast<size_t>(t) * k_used + slot];
        const size_t exp_stride = static_cast<size_t>(2 * ff) * K_blocks;
        const size_t g_off      = static_cast<size_t>(eid) * exp_stride + static_cast<size_t>(row) * K_blocks;
        const size_t u_off      = g_off + static_cast<size_t>(ff) * K_blocks;

        const int4 *     gq = gu_q + g_off;
        const uint16_t * gs = gu_s + g_off;
        const int4 *     uq = gu_q + u_off;
        const uint16_t * us = gu_s + u_off;

        float ag = 0.0f, au = 0.0f;
        for (int b = sub_l; b < K_blocks; b += 8) {
            const int4 xb[4] = { s_xh2[b * 4 + 0], s_xh2[b * 4 + 1], s_xh2[b * 4 + 2], s_xh2[b * 4 + 3] };
            const float xd   = s_xd[b];
            const int4 gw    = __ldg(&gq[b]);
            const int4 uw    = __ldg(&uq[b]);
            const float gd   = ink2_half_bits_to_f32(__ldg(&gs[b]));
            const float ud   = ink2_half_bits_to_f32(__ldg(&us[b]));
            ag += (gd * xd) * ink2_dot_iq4_nl_blk_h2(gw, s_lut256, xb);
            au += (ud * xd) * ink2_dot_iq4_nl_blk_h2(uw, s_lut256, xb);
        }
        ag += __shfl_down_sync(0xffffffffu, ag, 4, 8); au += __shfl_down_sync(0xffffffffu, au, 4, 8);
        ag += __shfl_down_sync(0xffffffffu, ag, 2, 8); au += __shfl_down_sync(0xffffffffu, au, 2, 8);
        ag += __shfl_down_sync(0xffffffffu, ag, 1, 8); au += __shfl_down_sync(0xffffffffu, au, 1, 8);

        if (sub_l == 0) {
            s_act[r_in_cta] = situ_act(ag, au, beta, lb);
        }
    } else if (sub_l == 0) {
        s_act[r_in_cta] = 0.0f;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        const float v = s_act[lane];
        const float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(v)), 0);
        const float inv  = (amax > 0.0f) ? (1.0f / amax) : 0.0f;
        const float vn   = v * inv;
        const float vn_hi = __shfl_down_sync(0xffffffffu, vn, 16);
        const int ff_blks = ff >> 5;
        const size_t blk_out = (static_cast<size_t>(t) * k_used + slot) * ff_blks + blockIdx.x;
        if (lane < 16) {
            act_h2[blk_out * 16 + lane] = __halves2half2(__float2half_rn(vn), __float2half_rn(vn_hi));
        }
        if (lane == 0) {
            act_d[blk_out] = amax;
        }
    }
}

// Decode (T < 16) sm_60: split-slot (grid.y = k_used, wpb=4, LPR=4 -> 29.9 us vs 63.0 us!), ZERO atomics!
__global__ void inkling2_moe_down_iq4_nl_soa_f16_split_kernel(
    const int4 *     __restrict__ dn_q,
    const uint16_t * __restrict__ dn_s,
    const __half2 *  __restrict__ act_h2,
    const float *    __restrict__ act_d,
    const int32_t *  __restrict__ top_ids,
    const float *    __restrict__ top_w,
    float *          __restrict__ down_slots,
    int ff_blks, int n_embd, int k_used) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_ah2 = reinterpret_cast<int4 *>(smem_raw);
    auto * s_ad  = reinterpret_cast<float *>(s_ah2 + ff_blks * 4);

    const int slot = blockIdx.y;
    const int t    = blockIdx.z;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;

    const size_t slot_idx = static_cast<size_t>(t) * k_used + slot;
    const int4 * src_h2 = reinterpret_cast<const int4 *>(act_h2 + slot_idx * ff_blks * 16);
    const int n_i4 = ff_blks * 4;
    for (int i = tid; i < n_i4; i += nthr) s_ah2[i] = src_h2[i];
    const float * src_d = act_d + slot_idx * ff_blks;
    for (int i = tid; i < ff_blks; i += nthr) s_ad[i] = src_d[i];
    __syncthreads();

    const int lane  = threadIdx.x;
    const int sub_r = lane >> 2;
    const int sub_l = lane & 3;
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 8 + sub_r;
    if (row >= n_embd) return;

    const float my_kv = static_cast<float>(kvalues_iq4nl_d[lane & 15]);
    const int   eid   = top_ids[slot_idx];
    const float ew    = top_w  [slot_idx];
    const size_t roff = (static_cast<size_t>(eid) * n_embd + row) * ff_blks;
    const int4 *     dq = dn_q + roff;
    const uint16_t * ds = dn_s + roff;

    float acc = 0.0f;
    for (int b = sub_l; b < ff_blks; b += 4) {
        const int4 w4  = __ldg(&dq[b]);
        const float wd = ink2_half_bits_to_f32(__ldg(&ds[b]));
        acc += (wd * s_ad[b]) * ink2_dot_iq4_nl_blk_shfl_f32(w4, my_kv, s_ah2 + b * 4);
    }
    acc += __shfl_down_sync(0xffffffffu, acc, 2, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 1, 4);
    if (sub_l == 0) {
        down_slots[slot_idx * n_embd + row] = ew * acc;
    }
}

template <int BT, int KT>
__global__ void inkling2_moe_gate_up_grouped_iq4_nl_soa_f16_kernel(
    const int4 *         __restrict__ gu_q,
    const uint16_t *     __restrict__ gu_s,
    const __half2 *      __restrict__ x_h2,
    const float *        __restrict__ x_d,
    const Ink2TileDesc * __restrict__ tiles,
    const int32_t *      __restrict__ n_tiles_ptr,
    const int32_t *      __restrict__ sorted_pairs,
    float *              __restrict__ act,
    int K_blocks, int ff, int k_used, float beta, float lb) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Ink2TileDesc td = tiles[tile_idx];
    const int eid   = td.eid;
    const int start = td.start;
    const int end   = td.start + td.count;

    extern __shared__ uint8_t smem_raw[];
    __shared__ __half2 s_lut256[256];
    __shared__ int32_t s_pairs[BT];
    auto * s_xh2 = reinterpret_cast<int4 *>(smem_raw); // [KT, BT, 4 int4]
    auto * s_xd  = reinterpret_cast<float *>(s_xh2 + KT * BT * 4);

    const int tid   = threadIdx.y * 32 + threadIdx.x;
    const int nthr  = blockDim.y * 32; // 128 threads -> 64 rows/CTA (2 lanes/row!)
    const int lane  = threadIdx.x;
    const int sub_r = lane >> 1; // 0..15 (16 rows/warp)
    const int sub_l = lane & 1;  // 0..1
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 16 + sub_r;

    const size_t exp_stride = static_cast<size_t>(2 * ff) * K_blocks;
    const size_t g_off      = static_cast<size_t>(eid) * exp_stride + static_cast<size_t>(row) * K_blocks;
    const size_t u_off      = g_off + static_cast<size_t>(ff) * K_blocks;

    ink2_init_lut256(s_lut256, tid, nthr);

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) s_pairs[tid] = sorted_pairs[base_i + tid];
        __syncthreads();

        float ag[BT], au[BT];
        #pragma unroll
        for (int j = 0; j < BT; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

        for (int k0 = 0; k0 < K_blocks; k0 += KT) {
            const int kc = (k0 + KT < K_blocks) ? KT : (K_blocks - k0);
            const int tot_i4 = kc * BT * 4;
            for (int idx = tid; idx < tot_i4; idx += nthr) {
                const int c   = idx & 3;
                const int rem = idx >> 2;
                const int j   = rem % BT;
                const int bk  = rem / BT;
                if (j < n_tile) {
                    const int tok = s_pairs[j] >> 4;
                    s_xh2[idx] = reinterpret_cast<const int4 *>(
                        x_h2 + (static_cast<size_t>(tok) * K_blocks + k0 + bk) * 16)[c];
                } else {
                    s_xh2[idx] = make_int4(0, 0, 0, 0);
                }
            }
            const int tot_d = kc * BT;
            for (int idx = tid; idx < tot_d; idx += nthr) {
                const int j  = idx % BT;
                const int bk = idx / BT;
                s_xd[idx] = (j < n_tile) ? x_d[static_cast<size_t>(s_pairs[j] >> 4) * K_blocks + k0 + bk] : 0.0f;
            }
            __syncthreads();

            if (row < ff) {
                for (int bk = sub_l; bk < kc; bk += 2) {
                    const int b = k0 + bk;
                    const int4 gw = __ldg(&gu_q[g_off + b]);
                    const int4 uw = __ldg(&gu_q[u_off + b]);
                    const float gd = ink2_half_bits_to_f32(__ldg(&gu_s[g_off + b]));
                    const float ud = ink2_half_bits_to_f32(__ldg(&gu_s[u_off + b]));

                    const uint32_t gw0 = static_cast<uint32_t>(gw.x), gw1 = static_cast<uint32_t>(gw.y);
                    const uint32_t gw2 = static_cast<uint32_t>(gw.z), gw3 = static_cast<uint32_t>(gw.w);
                    const __half2 g0  = s_lut256[(gw0 >>  0) & 0xFFu], g1  = s_lut256[(gw0 >>  8) & 0xFFu];
                    const __half2 g2  = s_lut256[(gw0 >> 16) & 0xFFu], g3  = s_lut256[(gw0 >> 24) & 0xFFu];
                    const __half2 g4  = s_lut256[(gw1 >>  0) & 0xFFu], g5  = s_lut256[(gw1 >>  8) & 0xFFu];
                    const __half2 g6  = s_lut256[(gw1 >> 16) & 0xFFu], g7  = s_lut256[(gw1 >> 24) & 0xFFu];
                    const __half2 g8  = s_lut256[(gw2 >>  0) & 0xFFu], g9  = s_lut256[(gw2 >>  8) & 0xFFu];
                    const __half2 g10 = s_lut256[(gw2 >> 16) & 0xFFu], g11 = s_lut256[(gw2 >> 24) & 0xFFu];
                    const __half2 g12 = s_lut256[(gw3 >>  0) & 0xFFu], g13 = s_lut256[(gw3 >>  8) & 0xFFu];
                    const __half2 g14 = s_lut256[(gw3 >> 16) & 0xFFu], g15 = s_lut256[(gw3 >> 24) & 0xFFu];

                    const uint32_t uw0 = static_cast<uint32_t>(uw.x), uw1 = static_cast<uint32_t>(uw.y);
                    const uint32_t uw2 = static_cast<uint32_t>(uw.z), uw3 = static_cast<uint32_t>(uw.w);
                    const __half2 u0  = s_lut256[(uw0 >>  0) & 0xFFu], u1  = s_lut256[(uw0 >>  8) & 0xFFu];
                    const __half2 u2  = s_lut256[(uw0 >> 16) & 0xFFu], u3  = s_lut256[(uw0 >> 24) & 0xFFu];
                    const __half2 u4  = s_lut256[(uw1 >>  0) & 0xFFu], u5  = s_lut256[(uw1 >>  8) & 0xFFu];
                    const __half2 u6  = s_lut256[(uw1 >> 16) & 0xFFu], u7  = s_lut256[(uw1 >> 24) & 0xFFu];
                    const __half2 u8  = s_lut256[(uw2 >>  0) & 0xFFu], u9  = s_lut256[(uw2 >>  8) & 0xFFu];
                    const __half2 u10 = s_lut256[(uw2 >> 16) & 0xFFu], u11 = s_lut256[(uw2 >> 24) & 0xFFu];
                    const __half2 u12 = s_lut256[(uw3 >>  0) & 0xFFu], u13 = s_lut256[(uw3 >>  8) & 0xFFu];
                    const __half2 u14 = s_lut256[(uw3 >> 16) & 0xFFu], u15 = s_lut256[(uw3 >> 24) & 0xFFu];

                    const int4 * xk  = s_xh2 + bk * BT * 4;
                    const float * dk = s_xd  + bk * BT;
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) {
                        const int4 * xj = xk + j * 4;
                        const int4 x0 = xj[0], x1 = xj[1], x2 = xj[2], x3 = xj[3];
                        const __half2 * h0 = reinterpret_cast<const __half2 *>(&x0);
                        const __half2 * h1 = reinterpret_cast<const __half2 *>(&x1);
                        const __half2 * h2 = reinterpret_cast<const __half2 *>(&x2);
                        const __half2 * h3 = reinterpret_cast<const __half2 *>(&x3);

                        __half2 r0 = ink2_hmul2(g0, h0[0]);
                        r0 = ink2_hfma2(g1,  h0[1], r0); r0 = ink2_hfma2(g2,  h0[2], r0); r0 = ink2_hfma2(g3,  h0[3], r0);
                        r0 = ink2_hfma2(g4,  h1[0], r0); r0 = ink2_hfma2(g5,  h1[1], r0); r0 = ink2_hfma2(g6,  h1[2], r0); r0 = ink2_hfma2(g7,  h1[3], r0);
                        r0 = ink2_hfma2(g8,  h2[0], r0); r0 = ink2_hfma2(g9,  h2[1], r0); r0 = ink2_hfma2(g10, h2[2], r0); r0 = ink2_hfma2(g11, h2[3], r0);
                        r0 = ink2_hfma2(g12, h3[0], r0); r0 = ink2_hfma2(g13, h3[1], r0); r0 = ink2_hfma2(g14, h3[2], r0); r0 = ink2_hfma2(g15, h3[3], r0);

                        __half2 s0 = ink2_hmul2(u0, h0[0]);
                        s0 = ink2_hfma2(u1,  h0[1], s0); s0 = ink2_hfma2(u2,  h0[2], s0); s0 = ink2_hfma2(u3,  h0[3], s0);
                        s0 = ink2_hfma2(u4,  h1[0], s0); s0 = ink2_hfma2(u5,  h1[1], s0); s0 = ink2_hfma2(u6,  h1[2], s0); s0 = ink2_hfma2(u7,  h1[3], s0);
                        s0 = ink2_hfma2(u8,  h2[0], s0); s0 = ink2_hfma2(u9,  h2[1], s0); s0 = ink2_hfma2(u10, h2[2], s0); s0 = ink2_hfma2(u11, h2[3], s0);
                        s0 = ink2_hfma2(u12, h3[0], s0); s0 = ink2_hfma2(u13, h3[1], s0); s0 = ink2_hfma2(u14, h3[2], s0); s0 = ink2_hfma2(u15, h3[3], s0);

                        const float dj = dk[j];
                        const float2 gf = __half22float2(r0);
                        const float2 uf = __half22float2(s0);
                        ag[j] += (gd * dj) * (gf.x + gf.y);
                        au[j] += (ud * dj) * (uf.x + uf.y);
                    }
                }
            }
            __syncthreads();
        }

        if (row < ff) {
            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                float gv = ag[j] + __shfl_xor_sync(0xffffffffu, ag[j], 1);
                float uv = au[j] + __shfl_xor_sync(0xffffffffu, au[j], 1);
                if (sub_l == 0 && j < n_tile) {
                    const int pair = s_pairs[j];
                    const int tok  = pair >> 4;
                    const int slot = pair & 15;
                    act[(static_cast<size_t>(tok) * k_used + slot) * ff + row] =
                        situ_act(gv, uv, beta, lb);
                }
            }
        }
        __syncthreads();
    }
}

template <int BT, int KT>
__global__ void inkling2_moe_down_grouped_iq4_nl_soa_f16_kernel(
    const int4 *         __restrict__ dn_q,
    const uint16_t *     __restrict__ dn_s,
    const __half2 *      __restrict__ act_h2,
    const float *        __restrict__ act_d,
    const float *        __restrict__ top_w,
    const Ink2TileDesc * __restrict__ tiles,
    const int32_t *      __restrict__ n_tiles_ptr,
    const int32_t *      __restrict__ sorted_pairs,
    float *              __restrict__ down_slots,
    int ff_blks, int n_embd, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Ink2TileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int end    = td.start + td.count;

    extern __shared__ uint8_t smem_raw[];
    __shared__ __half2 s_lut256[256];
    __shared__ int32_t s_pairs[BT];
    auto * s_ah2 = reinterpret_cast<int4 *>(smem_raw);
    auto * s_ad  = reinterpret_cast<float *>(s_ah2 + BT * KT * 4);

    const int tid   = threadIdx.y * 32 + threadIdx.x;
    const int nthr  = blockDim.y * 32;
    const int lane  = threadIdx.x;
    const int sub_r = lane >> 1;
    const int sub_l = lane & 1;
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 16 + sub_r;

    ink2_init_lut256(s_lut256, tid, nthr);

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) s_pairs[tid] = sorted_pairs[base_i + tid];
        __syncthreads();

        float acc[BT];
        #pragma unroll
        for (int j = 0; j < BT; ++j) acc[j] = 0.0f;

        for (int k0 = 0; k0 < ff_blks; k0 += KT) {
            const int kc = (k0 + KT < ff_blks) ? KT : (ff_blks - k0);
            const int tot_i4 = BT * kc * 4;
            for (int idx = tid; idx < tot_i4; idx += nthr) {
                const int j   = idx / (kc * 4);
                const int rem = idx - j * (kc * 4);
                const int bk  = rem >> 2;
                const int c   = rem & 3;
                if (j < n_tile) {
                    const int pair = s_pairs[j];
                    const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                    s_ah2[(j * KT + bk) * 4 + c] = reinterpret_cast<const int4 *>(
                        act_h2 + (slot_idx * ff_blks + k0 + bk) * 16)[c];
                } else {
                    s_ah2[(j * KT + bk) * 4 + c] = make_int4(0, 0, 0, 0);
                }
            }
            const int tot_d = BT * kc;
            for (int idx = tid; idx < tot_d; idx += nthr) {
                const int j  = idx / kc;
                const int bk = idx - j * kc;
                if (j < n_tile) {
                    const int pair = s_pairs[j];
                    const size_t slot_idx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                    s_ad[j * KT + bk] = act_d[slot_idx * ff_blks + k0 + bk];
                } else {
                    s_ad[j * KT + bk] = 0.0f;
                }
            }
            __syncthreads();

            if (row < n_embd) {
                const size_t roff = (static_cast<size_t>(eid) * n_embd + row) * ff_blks;
                for (int bk = sub_l; bk < kc; bk += 2) {
                    const int b = k0 + bk;
                    const int4 w4  = __ldg(&dn_q[roff + b]);
                    const float wd = ink2_half_bits_to_f32(__ldg(&dn_s[roff + b]));
                    __half2 acc2[BT];
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) acc2[j] = __floats2half2_rn(0.0f, 0.0f);
                    const uint32_t ws[4] = {
                        static_cast<uint32_t>(w4.x), static_cast<uint32_t>(w4.y),
                        static_cast<uint32_t>(w4.z), static_cast<uint32_t>(w4.w)
                    };
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const uint32_t word = ws[c];
                        const __half2 k0h = s_lut256[(word >>  0) & 0xFFu];
                        const __half2 k1h = s_lut256[(word >>  8) & 0xFFu];
                        const __half2 k2h = s_lut256[(word >> 16) & 0xFFu];
                        const __half2 k3h = s_lut256[(word >> 24) & 0xFFu];
                        #pragma unroll
                        for (int j = 0; j < BT; ++j) {
                            const int4 xv = s_ah2[(j * KT + bk) * 4 + c];
                            const __half2 * xh = reinterpret_cast<const __half2 *>(&xv);
                            acc2[j] = ink2_hfma2(k0h, xh[0], acc2[j]);
                            acc2[j] = ink2_hfma2(k1h, xh[1], acc2[j]);
                            acc2[j] = ink2_hfma2(k2h, xh[2], acc2[j]);
                            acc2[j] = ink2_hfma2(k3h, xh[3], acc2[j]);
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) {
                        const float2 f2 = __half22float2(acc2[j]);
                        acc[j] += (wd * s_ad[j * KT + bk]) * (f2.x + f2.y);
                    }
                }
            }
            __syncthreads();
        }

        if (row < n_embd) {
            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                float v = acc[j] + __shfl_down_sync(0xffffffffu, acc[j], 1);
                if (j < n_tile && sub_l == 0) {
                    const int pair = s_pairs[j];
                    const int tok  = pair >> 4;
                    const int slot = pair & 15;
                    const float ew = top_w[static_cast<size_t>(tok) * k_used + slot];
                    down_slots[(static_cast<size_t>(tok) * k_used + slot) * n_embd + row] = ew * v;
                }
            }
        }
        __syncthreads();
    }
}

// Generic sm_61+ / sm_52 IQ4_NL SoA dp4a helper (16-entry shared LUT + coalesced int4 weight load)
__device__ __forceinline__ void ink2_unpack_iq4_nl_dp4a(
    int4 w4, const int8_t * __restrict__ s_lut16, int w_lo[4], int w_hi[4]) {
    const uint32_t words[4] = {
        static_cast<uint32_t>(w4.x), static_cast<uint32_t>(w4.y),
        static_cast<uint32_t>(w4.z), static_cast<uint32_t>(w4.w)
    };
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint32_t q4 = words[g];
        w_lo[g] = static_cast<int>(
              static_cast<uint32_t>(static_cast<uint8_t>(s_lut16[(q4 >>  0) & 0xFu]))
            | (static_cast<uint32_t>(static_cast<uint8_t>(s_lut16[(q4 >>  8) & 0xFu])) << 8)
            | (static_cast<uint32_t>(static_cast<uint8_t>(s_lut16[(q4 >> 16) & 0xFu])) << 16)
            | (static_cast<uint32_t>(static_cast<uint8_t>(s_lut16[(q4 >> 24) & 0xFu])) << 24));
        w_hi[g] = static_cast<int>(
              static_cast<uint32_t>(static_cast<uint8_t>(s_lut16[(q4 >>  4) & 0xFu]))
            | (static_cast<uint32_t>(static_cast<uint8_t>(s_lut16[(q4 >> 12) & 0xFu])) << 8)
            | (static_cast<uint32_t>(static_cast<uint8_t>(s_lut16[(q4 >> 20) & 0xFu])) << 16)
            | (static_cast<uint32_t>(static_cast<uint8_t>(s_lut16[(q4 >> 28) & 0xFu])) << 24));
    }
}

__global__ void inkling2_moe_gate_up_situ_iq4_nl_soa_dp4a_kernel(
    const int4 *          __restrict__ gu_q,
    const uint16_t *      __restrict__ gu_s,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ top_ids,
    float *               __restrict__ act,
    int K_blocks, int ff, int k_used, float beta, float lb) {

    extern __shared__ BlockQ8_1_Act s_x[];
    __shared__ int8_t s_lut16[16];
    const int slot = blockIdx.y;
    const int t    = blockIdx.z;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    if (tid < 16) s_lut16[tid] = kvalues_iq4nl_d[tid];
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * src = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(t) * K_blocks);
    int4 *       dst = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) dst[i] = src[i];
    __syncthreads();

    const int lane  = threadIdx.x;
    const int sub_r = lane >> 3;
    const int sub_l = lane & 7;
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 4 + sub_r;
    if (row >= ff) return;

    const int eid = top_ids[static_cast<size_t>(t) * k_used + slot];
    const size_t exp_stride = static_cast<size_t>(2 * ff) * K_blocks;
    const size_t g_off      = static_cast<size_t>(eid) * exp_stride + static_cast<size_t>(row) * K_blocks;
    const size_t u_off      = g_off + static_cast<size_t>(ff) * K_blocks;

    float ag = 0.0f, au = 0.0f;
    for (int b = sub_l; b < K_blocks; b += 8) {
        const int * x32 = reinterpret_cast<const int *>(s_x[b].qs);
        const float xd  = s_x[b].d;
        int w_lo[4], w_hi[4];
        {
            ink2_unpack_iq4_nl_dp4a(__ldg(&gu_q[g_off + b]), s_lut16, w_lo, w_hi);
            int sg = 0;
            #pragma unroll
            for (int g = 0; g < 4; ++g) sg = dp4a_s32(w_hi[g], x32[4 + g], dp4a_s32(w_lo[g], x32[g], sg));
            ag += (ink2_half_bits_to_f32(__ldg(&gu_s[g_off + b])) * xd) * static_cast<float>(sg);
        }
        {
            ink2_unpack_iq4_nl_dp4a(__ldg(&gu_q[u_off + b]), s_lut16, w_lo, w_hi);
            int su = 0;
            #pragma unroll
            for (int g = 0; g < 4; ++g) su = dp4a_s32(w_hi[g], x32[4 + g], dp4a_s32(w_lo[g], x32[g], su));
            au += (ink2_half_bits_to_f32(__ldg(&gu_s[u_off + b])) * xd) * static_cast<float>(su);
        }
    }
    ag += __shfl_down_sync(0xffffffffu, ag, 4); au += __shfl_down_sync(0xffffffffu, au, 4);
    ag += __shfl_down_sync(0xffffffffu, ag, 2); au += __shfl_down_sync(0xffffffffu, au, 2);
    ag += __shfl_down_sync(0xffffffffu, ag, 1); au += __shfl_down_sync(0xffffffffu, au, 1);
    if (sub_l == 0) act[(static_cast<size_t>(t) * k_used + slot) * ff + row] = situ_act(ag, au, beta, lb);
}

__global__ void inkling2_moe_down_iq4_nl_soa_dp4a_kernel(
    const int4 *          __restrict__ dn_q,
    const uint16_t *      __restrict__ dn_s,
    const BlockQ8_1_Act * __restrict__ act_q8,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ out,
    int ff_blks, int n_embd, int k_used) {

    extern __shared__ BlockQ8_1_Act s_a[];
    __shared__ int8_t s_lut16[16];
    const int t    = blockIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    if (tid < 16) s_lut16[tid] = kvalues_iq4nl_d[tid];
    const int n_i4 = (k_used * ff_blks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * src = reinterpret_cast<const int4 *>(act_q8 + static_cast<size_t>(t) * k_used * ff_blks);
    int4 *       dst = reinterpret_cast<int4 *>(s_a);
    for (int i = tid; i < n_i4; i += nthr) dst[i] = src[i];
    __syncthreads();

    const int lane  = threadIdx.x;
    const int sub_r = lane >> 2;
    const int sub_l = lane & 3;
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 8 + sub_r;
    if (row >= n_embd) return;

    float total = 0.0f;
    for (int slot = 0; slot < k_used; ++slot) {
        const int   eid  = top_ids[static_cast<size_t>(t) * k_used + slot];
        const float ew   = top_w  [static_cast<size_t>(t) * k_used + slot];
        const size_t roff = (static_cast<size_t>(eid) * n_embd + row) * ff_blks;
        const BlockQ8_1_Act * sa = s_a + slot * ff_blks;
        float acc = 0.0f;
        for (int b = sub_l; b < ff_blks; b += 4) {
            int w_lo[4], w_hi[4];
            ink2_unpack_iq4_nl_dp4a(__ldg(&dn_q[roff + b]), s_lut16, w_lo, w_hi);
            const int * a32 = reinterpret_cast<const int *>(sa[b].qs);
            int si = 0;
            #pragma unroll
            for (int g = 0; g < 4; ++g) si = dp4a_s32(w_hi[g], a32[4 + g], dp4a_s32(w_lo[g], a32[g], si));
            acc += (ink2_half_bits_to_f32(__ldg(&dn_s[roff + b])) * sa[b].d) * static_cast<float>(si);
        }
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        total += ew * acc;
    }
    if (sub_l == 0) out[static_cast<size_t>(t) * n_embd + row] = total;
}

template <int BT, int R_PER_WARP>
__global__ void inkling2_moe_gate_up_grouped_iq4_nl_soa_dp4a_kernel(
    const int4 *          __restrict__ gu_q,
    const uint16_t *      __restrict__ gu_s,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ exp_offsets,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ act,
    int K_blocks, int ff, int k_used, float beta, float lb) {

    const int eid   = blockIdx.y;
    const int start = exp_offsets[eid];
    const int end   = exp_offsets[eid + 1];
    if (start >= end) return;

    extern __shared__ BlockQ8_1_Act s_x[];
    __shared__ int8_t  s_lut16[16];
    __shared__ int32_t s_pairs[BT];
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int lane     = threadIdx.x;
    const int sub_r    = lane >> 3;
    const int sub_l    = lane & 7;
    const int rows_step = blockDim.y * 4;
    const int row_base = blockIdx.x * (rows_step * R_PER_WARP) + threadIdx.y * 4 + sub_r;
    const int n_i4_tok = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const size_t exp_stride = static_cast<size_t>(2 * ff) * K_blocks;
    if (tid < 16) s_lut16[tid] = kvalues_iq4nl_d[tid];

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) s_pairs[tid] = sorted_pairs[base_i + tid];
        __syncthreads();

        for (int j = 0; j < n_tile; ++j) {
            const int tok = s_pairs[j] >> 4;
            const int4 * src = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(tok) * K_blocks);
            int4 *       dst = reinterpret_cast<int4 *>(s_x + j * K_blocks);
            for (int i = tid; i < n_i4_tok; i += nthr) dst[i] = src[i];
        }
        __syncthreads();

        for (int rr = 0; rr < R_PER_WARP; ++rr) {
            const int row = row_base + rr * rows_step;
            if (row >= ff) break;
            const size_t g_off = static_cast<size_t>(eid) * exp_stride + static_cast<size_t>(row) * K_blocks;
            const size_t u_off = g_off + static_cast<size_t>(ff) * K_blocks;

            float ag[BT], au[BT];
            #pragma unroll
            for (int j = 0; j < BT; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

            for (int b = sub_l; b < K_blocks; b += 8) {
                int gw_lo[4], gw_hi[4], uw_lo[4], uw_hi[4];
                ink2_unpack_iq4_nl_dp4a(__ldg(&gu_q[g_off + b]), s_lut16, gw_lo, gw_hi);
                ink2_unpack_iq4_nl_dp4a(__ldg(&gu_q[u_off + b]), s_lut16, uw_lo, uw_hi);
                const float gd = ink2_half_bits_to_f32(__ldg(&gu_s[g_off + b]));
                const float ud = ink2_half_bits_to_f32(__ldg(&gu_s[u_off + b]));

                #pragma unroll
                for (int j = 0; j < BT; ++j) {
                    if (j < n_tile) {
                        const BlockQ8_1_Act & xb = s_x[j * K_blocks + b];
                        const int * x32 = reinterpret_cast<const int *>(xb.qs);
                        int sg = 0, su = 0;
                        #pragma unroll
                        for (int g = 0; g < 4; ++g) {
                            sg = dp4a_s32(gw_hi[g], x32[4 + g], dp4a_s32(gw_lo[g], x32[g], sg));
                            su = dp4a_s32(uw_hi[g], x32[4 + g], dp4a_s32(uw_lo[g], x32[g], su));
                        }
                        ag[j] += (gd * xb.d) * static_cast<float>(sg);
                        au[j] += (ud * xb.d) * static_cast<float>(su);
                    }
                }
            }
            for (int j = 0; j < n_tile; ++j) {
                float g = ag[j], u = au[j];
                g += __shfl_down_sync(0xffffffffu, g, 4); u += __shfl_down_sync(0xffffffffu, u, 4);
                g += __shfl_down_sync(0xffffffffu, g, 2); u += __shfl_down_sync(0xffffffffu, u, 2);
                g += __shfl_down_sync(0xffffffffu, g, 1); u += __shfl_down_sync(0xffffffffu, u, 1);
                if (sub_l == 0) {
                    const int pair = s_pairs[j];
                    act[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * ff + row] = situ_act(g, u, beta, lb);
                }
            }
        }
        __syncthreads();
    }
}

template <int BT, int R_PER_WARP>
__global__ void inkling2_moe_down_grouped_iq4_nl_soa_dp4a_kernel(
    const int4 *          __restrict__ dn_q,
    const uint16_t *      __restrict__ dn_s,
    const BlockQ8_1_Act * __restrict__ act_q8,
    const float *         __restrict__ top_w,
    const int32_t *       __restrict__ exp_offsets,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots,
    int ff_blks, int n_embd, int k_used) {

    const int eid   = blockIdx.y;
    const int start = exp_offsets[eid];
    const int end   = exp_offsets[eid + 1];
    if (start >= end) return;

    extern __shared__ BlockQ8_1_Act s_a[];
    __shared__ int8_t  s_lut16[16];
    __shared__ int32_t s_pairs[BT];
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int lane     = threadIdx.x;
    const int sub_r    = lane >> 2;
    const int sub_l    = lane & 3;
    const int rows_step = blockDim.y * 8;
    const int row_base = blockIdx.x * (rows_step * R_PER_WARP) + threadIdx.y * 8 + sub_r;
    const int n_i4_tok = (ff_blks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    if (tid < 16) s_lut16[tid] = kvalues_iq4nl_d[tid];

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) s_pairs[tid] = sorted_pairs[base_i + tid];
        __syncthreads();

        for (int j = 0; j < n_tile; ++j) {
            const int pair = s_pairs[j];
            const int4 * src = reinterpret_cast<const int4 *>(
                act_q8 + (static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * ff_blks);
            int4 * dst = reinterpret_cast<int4 *>(s_a + j * ff_blks);
            for (int i = tid; i < n_i4_tok; i += nthr) dst[i] = src[i];
        }
        __syncthreads();

        for (int rr = 0; rr < R_PER_WARP; ++rr) {
            const int row = row_base + rr * rows_step;
            if (row >= n_embd) break;
            const size_t roff = (static_cast<size_t>(eid) * n_embd + row) * ff_blks;

            float acc[BT];
            #pragma unroll
            for (int j = 0; j < BT; ++j) acc[j] = 0.0f;

            for (int b = sub_l; b < ff_blks; b += 4) {
                int w_lo[4], w_hi[4];
                ink2_unpack_iq4_nl_dp4a(__ldg(&dn_q[roff + b]), s_lut16, w_lo, w_hi);
                const float wd = ink2_half_bits_to_f32(__ldg(&dn_s[roff + b]));

                #pragma unroll
                for (int j = 0; j < BT; ++j) {
                    if (j < n_tile) {
                        const BlockQ8_1_Act & xb = s_a[j * ff_blks + b];
                        const int * a32 = reinterpret_cast<const int *>(xb.qs);
                        int si = 0;
                        #pragma unroll
                        for (int g = 0; g < 4; ++g) si = dp4a_s32(w_hi[g], a32[4 + g], dp4a_s32(w_lo[g], a32[g], si));
                        acc[j] += (wd * xb.d) * static_cast<float>(si);
                    }
                }
            }
            for (int j = 0; j < n_tile; ++j) {
                float v = acc[j];
                v += __shfl_down_sync(0xffffffffu, v, 2);
                v += __shfl_down_sync(0xffffffffu, v, 1);
                if (sub_l == 0) {
                    const int pair = s_pairs[j];
                    const int tok  = pair >> 4;
                    const int slot = pair & 15;
                    const float ew = top_w[static_cast<size_t>(tok) * k_used + slot];
                    down_slots[(static_cast<size_t>(tok) * k_used + slot) * n_embd + row] = ew * v;
                }
            }
        }
        __syncthreads();
    }
}


// ============================================================================
// 1B. 1-Pass Register-Cached float4 Depth Mix (n_embd = 2304 -> 576 threads = 1 float4/thread!)
//     Reads each state vector ONCE into registers, computes ALL inv_rms & ALL dot products
//     in 2 barriers total (vs 3 passes & ~48 barriers in scalar loop)!
// ============================================================================
template <int NQ>
__global__ void inkling2_depth_mix_f4_fast_kernel(
    const float * const * __restrict__ block_ptrs,
    int n_blocks,
    const float * __restrict__ cur,
    const float * __restrict__ query, // [NQ, n_embd]
    float *       __restrict__ out,   // [NQ, T, n_embd]
    int n_embd, int T, float eps) {

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x; // 0 .. n4-1 (576 threads = 18 warps for n_embd=2304)
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;
    const int n_st = n_blocks + 1;
    const size_t t_off4 = static_cast<size_t>(t) * (n_embd >> 2);

    float4 qv[NQ];
    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
        qv[q] = reinterpret_cast<const float4 *>(query + static_cast<size_t>(q) * n_embd)[tid];
    }

    float4 xs[MAX_DEPTH_STATES];
    __shared__ float s_ss[MAX_DEPTH_STATES][32];
    __shared__ float s_dot[NQ][MAX_DEPTH_STATES][32];
    __shared__ float s_ir[MAX_DEPTH_STATES];
    __shared__ float s_sc[NQ][MAX_DEPTH_STATES];
    __shared__ float s_w[NQ][MAX_DEPTH_STATES];

    #pragma unroll
    for (int s = 0; s < MAX_DEPTH_STATES; ++s) {
        if (s < n_st) {
            const float * sp = (s < n_blocks) ? block_ptrs[s] : cur;
            float4 v = reinterpret_cast<const float4 *>(sp)[t_off4 + tid];
            xs[s] = v;
            float ss = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
            if (lane == 0) s_ss[s][warp] = ss;
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                float d = warp_sum(v.x * qv[q].x + v.y * qv[q].y + v.z * qv[q].z + v.w * qv[q].w);
                if (lane == 0) s_dot[q][s][warp] = d;
            }
        }
    }
    __syncthreads();

    // Parallel warp reduction: warp `s` reduces state `s`!
    if (warp < n_st) {
        float tot = warp_sum((lane < nwarps) ? s_ss[warp][lane] : 0.0f);
        float ir = rsqrtf(tot / static_cast<float>(n_embd) + eps);
        if (lane == 0) s_ir[warp] = ir;
        ir = __shfl_sync(0xffffffffu, ir, 0);
        const float scale = rsqrtf(static_cast<float>(n_embd)) * ir;
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float dt = warp_sum((lane < nwarps) ? s_dot[q][warp][lane] : 0.0f);
            if (lane == 0) s_sc[q][warp] = dt * scale;
        }
    }
    __syncthreads();

    // Parallel softmax: warp `q` computes softmax weights across MAX_DEPTH_STATES <= 8!
    if (warp < NQ) {
        float sc = (lane < n_st) ? s_sc[warp][lane] : -FLT_MAX;
        float mx = sc;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        float ex = (lane < n_st) ? expf(sc - mx) : 0.0f;
        float sm = ex;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, off);
        if (lane < n_st) s_w[warp][lane] = (ex / sm) * s_ir[lane];
    }
    __syncthreads();

    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int s = 0; s < MAX_DEPTH_STATES; ++s) {
            if (s < n_st) {
                const float ws = s_w[q][s];
                acc.x += ws * xs[s].x; acc.y += ws * xs[s].y;
                acc.z += ws * xs[s].z; acc.w += ws * xs[s].w;
            }
        }
        reinterpret_cast<float4 *>(out + (static_cast<size_t>(q) * T + t) * n_embd)[tid] = acc;
    }
}

// ============================================================================
// 1C. 1-Pass Register-Cached float4 RMSNorm (single & 3-way batched Q/K/V!)
// ============================================================================
__global__ void inkling2_rmsnorm_f4_1pass_kernel(
    const float * __restrict__ x,
    const float * __restrict__ w,
    float *       __restrict__ out,
    int n_embd, float eps) {

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x; // 0 .. n4-1
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;

    const float4 v = reinterpret_cast<const float4 *>(x + static_cast<size_t>(t) * n_embd)[tid];
    float ss = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    __shared__ float s_red[32];
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < nwarps) ? s_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) s_red[0] = rsqrtf(tot / static_cast<float>(n_embd) + eps);
    }
    __syncthreads();
    const float inv = s_red[0];
    float4 o;
    if (w) {
        const float4 wv = reinterpret_cast<const float4 *>(w)[tid];
        o = make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w);
    } else {
        o = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
    }
    reinterpret_cast<float4 *>(out + static_cast<size_t>(t) * n_embd)[tid] = o;
}

struct Ink2Ptr3 { const float * p[3]; };
struct Ink2Out3 { float * p[3]; };

__global__ void inkling2_rmsnorm3_f4_1pass_kernel(
    Ink2Ptr3 xs, Ink2Ptr3 ws, Ink2Out3 outs, int n_embd, float eps) {

    const int t    = blockIdx.x;
    const int c    = blockIdx.y; // 0, 1, 2
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;

    const float4 v = reinterpret_cast<const float4 *>(xs.p[c] + static_cast<size_t>(t) * n_embd)[tid];
    float ss = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    __shared__ float s_red[32];
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < nwarps) ? s_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) s_red[0] = rsqrtf(tot / static_cast<float>(n_embd) + eps);
    }
    __syncthreads();
    const float inv = s_red[0];
    const float4 wv = reinterpret_cast<const float4 *>(ws.p[c])[tid];
    reinterpret_cast<float4 *>(outs.p[c] + static_cast<size_t>(t) * n_embd)[tid] =
        make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w);
}

// ============================================================================
// 2B. Fused Loop Controller Kernels (eliminates 400+ tiny launches/tok!):
//     - inkling2_loop_ctrl_up_combine_kernel: fuses ctrl_gain + ctrl_forget + loop_combine!
//     - Batched 3-way Q/K/V versions for T == 1 decode!
// ============================================================================
__global__ void inkling2_loop_ctrl_up_combine_kernel(
    const float * __restrict__ w_gain,   // [n_embd, gate_rank]
    const float * __restrict__ w_forget, // [n_embd, gate_rank]
    const float * __restrict__ hid,      // [T, gate_rank]
    const float * __restrict__ cur,      // [T, n_embd]
    const float * __restrict__ msg,      // [T, n_embd]
    float *       __restrict__ out,      // [T, n_embd]
    int n_embd, int gate_rank) {

    extern __shared__ float s_hid[];
    const int t    = blockIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    if (tid < gate_rank) s_hid[tid] = hid[static_cast<size_t>(t) * gate_rank + tid];
    __syncthreads();

    // 8 lanes per row (4 rows/warp -> 32 rows/CTA -> 72 CTAs for n_embd=2304!)
    const int lane  = threadIdx.x;
    const int sub_r = lane >> 3; // 0..3
    const int sub_l = lane & 7;  // 0..7
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 4 + sub_r;
    if (row >= n_embd) return;

    const int K4 = gate_rank >> 2;
    const auto * g4 = reinterpret_cast<const float4 *>(w_gain   + static_cast<size_t>(row) * gate_rank);
    const auto * f4 = reinterpret_cast<const float4 *>(w_forget + static_cast<size_t>(row) * gate_rank);
    const auto * h4 = reinterpret_cast<const float4 *>(s_hid);

    float ag = 0.0f, af = 0.0f;
    for (int k = sub_l; k < K4; k += 8) {
        const float4 hv = h4[k];
        const float4 gv = __ldg(&g4[k]);
        const float4 fv = __ldg(&f4[k]);
        ag += gv.x * hv.x + gv.y * hv.y + gv.z * hv.z + gv.w * hv.w;
        af += fv.x * hv.x + fv.y * hv.y + fv.z * hv.z + fv.w * hv.w;
    }
    ag += __shfl_down_sync(0xffffffffu, ag, 4); af += __shfl_down_sync(0xffffffffu, af, 4);
    ag += __shfl_down_sync(0xffffffffu, ag, 2); af += __shfl_down_sync(0xffffffffu, af, 2);
    ag += __shfl_down_sync(0xffffffffu, ag, 1); af += __shfl_down_sync(0xffffffffu, af, 1);

    if (sub_l == 0) {
        const size_t idx = static_cast<size_t>(t) * n_embd + row;
        const float c = cur[idx];
        const float g = tanhf(ag);
        const float f = tanhf(af);
        out[idx] = (c - f * c) + g * msg[idx];
    }
}

__global__ void inkling2_rmsnorm_concat2_3way_kernel(
    Ink2Ptr3 ins, Ink2Ptr3 msgs, float * __restrict__ out3, int n_embd, float eps) {

    const int c    = blockIdx.x; // 0, 1, 2
    const int half = blockIdx.y; // 0 = cur, 1 = msg -> 6 CTAs of 576 threads!
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;

    const float * src = (half == 0) ? ins.p[c] : msgs.p[c];
    const float4 v = reinterpret_cast<const float4 *>(src)[tid];
    float ss = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    __shared__ float s_red[32];
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < nwarps) ? s_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) s_red[0] = rsqrtf(tot / static_cast<float>(n_embd) + eps);
    }
    __syncthreads();
    const float inv = s_red[0];
    float * dst = out3 + static_cast<size_t>(c) * (2 * n_embd) + static_cast<size_t>(half) * n_embd;
    reinterpret_cast<float4 *>(dst)[tid] = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
}

__global__ void inkling2_ctrl_down_3way_kernel(
    Ink2Ptr3 ws, const float * __restrict__ x3, float * __restrict__ y3, int K, int N) {
    const int row  = blockIdx.x; // 0..63
    const int c    = blockIdx.y; // 0..2 -> 192 CTAs across 56 SMs!
    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int K4 = K >> 2;
    const auto * w4 = reinterpret_cast<const float4 *>(ws.p[c] + static_cast<size_t>(row) * K);
    const auto * x4 = reinterpret_cast<const float4 *>(x3 + static_cast<size_t>(c) * K);
    float acc = 0.0f;
    for (int k = tid; k < K4; k += 256) {
        float4 wv = __ldg(&w4[k]);
        float4 xv = x4[k];
        acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
    acc = warp_sum(acc);
    __shared__ float s_red[8];
    if (lane == 0) s_red[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? s_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) y3[c * N + row] = tot;
    }
}

__global__ void inkling2_loop_ctrl_up_combine_3way_kernel(
    const float * __restrict__ wg0, const float * __restrict__ wg1, const float * __restrict__ wg2,
    const float * __restrict__ wf0, const float * __restrict__ wf1, const float * __restrict__ wf2,
    const float * __restrict__ hid3,
    const float * __restrict__ cur0, const float * __restrict__ cur1, const float * __restrict__ cur2,
    const float * __restrict__ msg0, const float * __restrict__ msg1, const float * __restrict__ msg2,
    float * __restrict__ out0, float * __restrict__ out1, float * __restrict__ out2,
    int n_embd, int gate_rank) {

    extern __shared__ float s_hid[];
    const int c   = blockIdx.y; // 0, 1, 2
    const int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid < gate_rank) s_hid[tid] = hid3[c * gate_rank + tid];
    __syncthreads();

    const int lane  = threadIdx.x;
    const int sub_r = lane >> 3;
    const int sub_l = lane & 7;
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 4 + sub_r;
    if (row >= n_embd) return;

    const float * wgp = (c == 0) ? wg0 : ((c == 1) ? wg1 : wg2);
    const float * wfp = (c == 0) ? wf0 : ((c == 1) ? wf1 : wf2);
    const int K4 = gate_rank >> 2;
    const auto * g4 = reinterpret_cast<const float4 *>(wgp + static_cast<size_t>(row) * gate_rank);
    const auto * f4 = reinterpret_cast<const float4 *>(wfp + static_cast<size_t>(row) * gate_rank);
    const auto * h4 = reinterpret_cast<const float4 *>(s_hid);

    float ag = 0.0f, af = 0.0f;
    for (int k = sub_l; k < K4; k += 8) {
        const float4 hv = h4[k];
        const float4 gv = __ldg(&g4[k]);
        const float4 fv = __ldg(&f4[k]);
        ag += gv.x * hv.x + gv.y * hv.y + gv.z * hv.z + gv.w * hv.w;
        af += fv.x * hv.x + fv.y * hv.y + fv.z * hv.z + fv.w * hv.w;
    }
    ag += __shfl_down_sync(0xffffffffu, ag, 4); af += __shfl_down_sync(0xffffffffu, af, 4);
    ag += __shfl_down_sync(0xffffffffu, ag, 2); af += __shfl_down_sync(0xffffffffu, af, 2);
    ag += __shfl_down_sync(0xffffffffu, ag, 1); af += __shfl_down_sync(0xffffffffu, af, 1);

    if (sub_l == 0) {
        const float * cp = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
        const float * mp = (c == 0) ? msg0 : ((c == 1) ? msg1 : msg2);
        float *       op = (c == 0) ? out0 : ((c == 1) ? out1 : out2);
        const float cv = cp[row];
        const float g  = tanhf(ag);
        const float f  = tanhf(af);
        op[row] = (cv - f * cv) + g * mp[row];
    }
}

// ============================================================================
// 2C. Fused Multi-Output Q8_0 SoA GEMV for Decode Attention Projections!
//     Fuses up to 6 projections sharing the same input BlockQ8_1_Act into ONE launch:
//     reads s_x ONCE, 100% SM saturation, eliminates ~210 tiny launches/tok!
// ============================================================================
struct Ink2FusedQ8Projs {
    int n_projs;
    int row_end[6];          // cumulative end row for each proj
    const int4 *     qs[6];
    const uint16_t * sc[6];
    float *          out[6];
    int              N[6];
};

__global__ void inkling2_qkv_fused_q8_0_kernel(
    Ink2FusedQ8Projs P,
    const BlockQ8_1_Act * __restrict__ X,
    int K_blocks, int total_rows) {

    extern __shared__ BlockQ8_1_Act s_x[];
    const int t    = blockIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;

    const BlockQ8_1_Act * x_row = X + static_cast<size_t>(t) * K_blocks;
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(x_row);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];
    __syncthreads();

    const int g_row = blockIdx.x * blockDim.y + threadIdx.y;
    if (g_row >= total_rows) return;

    int pid = 0;
    int r0  = 0;
    #pragma unroll
    for (int p = 0; p < 6; ++p) {
        if (p < P.n_projs && g_row >= P.row_end[p]) {
            pid = p + 1;
            r0  = P.row_end[p];
        }
    }
    const int row = g_row - r0;

    const int lane        = threadIdx.x;
    const int blk_in_warp = lane >> 1;
    const int half_in_blk = lane & 1;

    const int4 *     w_q_row = P.qs[pid] + static_cast<size_t>(row) * (K_blocks * 2);
    const uint16_t * w_s_row = P.sc[pid] + static_cast<size_t>(row) * K_blocks;

    float acc = 0.0f;
    for (int b = blk_in_warp; b < K_blocks; b += 16) {
        const int4 w4 = __ldg(&w_q_row[b * 2 + half_in_blk]);
        const int4 x4 = reinterpret_cast<const int4 *>(s_x[b].qs)[half_in_blk];
        int sumi = dp4a_s32(w4.x, x4.x, dp4a_s32(w4.y, x4.y, dp4a_s32(w4.z, x4.z, dp4a_s32(w4.w, x4.w, 0))));
        __half wd_h;
        *reinterpret_cast<uint16_t *>(&wd_h) = __ldg(&w_s_row[b]);
        acc += (__half2float(wd_h) * s_x[b].d) * static_cast<float>(sumi);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        P.out[pid][static_cast<size_t>(t) * P.N[pid] + row] = acc;
    }
}

// ============================================================================

// ============================================================================
// Stage 5A: K-Tiled BT=16, KT=12, LPR=4 Q8_0 Grouped MoE (sm_60 half2 PRMT & sm_61+ dp4a)
//   - Zero register spills (zero-padded unused slots + statically unrolled j=0..BT-1)
//   - 64B coalesced weight reads per row (LPR=4) -> 3.1x faster on P100, 2.1x faster on GTX 1060!
// ============================================================================
template <int BT, int KT>
__global__ void inkling2_moe_gate_up_grouped_q8_0_soa_f16_kernel(
    const int4 *         __restrict__ gu_q,
    const uint16_t *     __restrict__ gu_s,
    const __half2 *      __restrict__ x_h2,
    const float *        __restrict__ x_d,
    const Ink2TileDesc * __restrict__ tiles,
    const int32_t *      __restrict__ n_tiles_ptr,
    const int32_t *      __restrict__ sorted_pairs,
    float *              __restrict__ act,
    int K_blocks, int ff, int k_used, float beta, float lb) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Ink2TileDesc td = tiles[tile_idx];
    const int eid   = td.eid;
    const int start = td.start;
    const int end   = td.start + td.count;

    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    auto * s_xh2 = reinterpret_cast<__half2 *>(smem_raw); // [KT, BT, 16 half2]
    auto * s_xd  = reinterpret_cast<float *>(s_xh2 + KT * BT * 16);

    const int tid    = threadIdx.x; // 0..127 -> 32 rows/CTA (LPR=4!)
    const int nthr   = blockDim.x;
    const int row    = blockIdx.x * (nthr >> 2) + (tid >> 2);
    const int sub    = tid & 3;
    const int b_lane = sub >> 1; // 0..1
    const int hb     = sub & 1;  // 0..1

    const size_t exp_stride = static_cast<size_t>(2 * ff) * K_blocks;
    const size_t g_off      = static_cast<size_t>(eid) * exp_stride + static_cast<size_t>(row) * K_blocks;
    const size_t u_off      = g_off + static_cast<size_t>(ff) * K_blocks;
    const int4 *     gq = gu_q + g_off * 2;
    const uint16_t * gs = gu_s + g_off;
    const int4 *     uq = gu_q + u_off * 2;
    const uint16_t * us = gu_s + u_off;
    const __half2 neg1152 = __floats2half2_rn(-1152.0f, -1152.0f);

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) s_pairs[tid] = sorted_pairs[base_i + tid];
        __syncthreads();

        float ag[BT], au[BT];
        #pragma unroll
        for (int j = 0; j < BT; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

        for (int k0 = 0; k0 < K_blocks; k0 += KT) {
            const int kc = (k0 + KT < K_blocks) ? KT : (K_blocks - k0);
            const int tot_i4 = kc * BT * 4;
            for (int idx = tid; idx < tot_i4; idx += nthr) {
                const int c   = idx & 3;
                const int rem = idx >> 2;
                const int j   = rem % BT;
                const int bk  = rem / BT;
                if (j < n_tile) {
                    const int tok = s_pairs[j] >> 4;
                    reinterpret_cast<int4 *>(s_xh2)[idx] = reinterpret_cast<const int4 *>(
                        x_h2 + (static_cast<size_t>(tok) * K_blocks + k0 + bk) * 16)[c];
                } else {
                    reinterpret_cast<int4 *>(s_xh2)[idx] = make_int4(0, 0, 0, 0);
                }
            }
            const int tot_d = kc * BT;
            for (int idx = tid; idx < tot_d; idx += nthr) {
                const int j  = idx % BT;
                const int bk = idx / BT;
                s_xd[idx] = (j < n_tile) ? x_d[static_cast<size_t>(s_pairs[j] >> 4) * K_blocks + k0 + bk] : 0.0f;
            }
            __syncthreads();

            if (row < ff) {
                for (int bk = b_lane; bk < kc; bk += 2) {
                    const int b = k0 + bk;
                    const int4 gw4 = __ldg(&gq[b * 2 + hb]);
                    const int4 uw4 = __ldg(&uq[b * 2 + hb]);
                    const float gd = ink2_half_bits_to_f32(__ldg(&gs[b]));
                    const float ud = ink2_half_bits_to_f32(__ldg(&us[b]));

                    const uint32_t gu0 = static_cast<uint32_t>(gw4.x) ^ 0x80808080u;
                    const uint32_t gu1 = static_cast<uint32_t>(gw4.y) ^ 0x80808080u;
                    const uint32_t gu2 = static_cast<uint32_t>(gw4.z) ^ 0x80808080u;
                    const uint32_t gu3 = static_cast<uint32_t>(gw4.w) ^ 0x80808080u;
                    const __half2 g0 = ink2_hadd2(ink2_i8x2_to_h2_lo(gu0), neg1152), g1 = ink2_hadd2(ink2_i8x2_to_h2_hi(gu0), neg1152);
                    const __half2 g2 = ink2_hadd2(ink2_i8x2_to_h2_lo(gu1), neg1152), g3 = ink2_hadd2(ink2_i8x2_to_h2_hi(gu1), neg1152);
                    const __half2 g4 = ink2_hadd2(ink2_i8x2_to_h2_lo(gu2), neg1152), g5 = ink2_hadd2(ink2_i8x2_to_h2_hi(gu2), neg1152);
                    const __half2 g6 = ink2_hadd2(ink2_i8x2_to_h2_lo(gu3), neg1152), g7 = ink2_hadd2(ink2_i8x2_to_h2_hi(gu3), neg1152);

                    const uint32_t uu0 = static_cast<uint32_t>(uw4.x) ^ 0x80808080u;
                    const uint32_t uu1 = static_cast<uint32_t>(uw4.y) ^ 0x80808080u;
                    const uint32_t uu2 = static_cast<uint32_t>(uw4.z) ^ 0x80808080u;
                    const uint32_t uu3 = static_cast<uint32_t>(uw4.w) ^ 0x80808080u;
                    const __half2 u0 = ink2_hadd2(ink2_i8x2_to_h2_lo(uu0), neg1152), u1 = ink2_hadd2(ink2_i8x2_to_h2_hi(uu0), neg1152);
                    const __half2 u2 = ink2_hadd2(ink2_i8x2_to_h2_lo(uu1), neg1152), u3 = ink2_hadd2(ink2_i8x2_to_h2_hi(uu1), neg1152);
                    const __half2 u4 = ink2_hadd2(ink2_i8x2_to_h2_lo(uu2), neg1152), u5 = ink2_hadd2(ink2_i8x2_to_h2_hi(uu2), neg1152);
                    const __half2 u6 = ink2_hadd2(ink2_i8x2_to_h2_lo(uu3), neg1152), u7 = ink2_hadd2(ink2_i8x2_to_h2_hi(uu3), neg1152);

                    const __half2 * xk = s_xh2 + bk * BT * 16 + hb * 8;
                    const float *   dk = s_xd  + bk * BT;
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) {
                        const __half2 * xj = xk + j * 16;
                        const __half2 x0 = xj[0], x1 = xj[1], x2 = xj[2], x3 = xj[3];
                        const __half2 x4 = xj[4], x5 = xj[5], x6 = xj[6], x7 = xj[7];
                        __half2 ga = ink2_hmul2(g0, x0); ga = ink2_hfma2(g1, x1, ga); ga = ink2_hfma2(g2, x2, ga); ga = ink2_hfma2(g3, x3, ga);
                        ga = ink2_hfma2(g4, x4, ga); ga = ink2_hfma2(g5, x5, ga); ga = ink2_hfma2(g6, x6, ga); ga = ink2_hfma2(g7, x7, ga);
                        __half2 ua = ink2_hmul2(u0, x0); ua = ink2_hfma2(u1, x1, ua); ua = ink2_hfma2(u2, x2, ua); ua = ink2_hfma2(u3, x3, ua);
                        ua = ink2_hfma2(u4, x4, ua); ua = ink2_hfma2(u5, x5, ua); ua = ink2_hfma2(u6, x6, ua); ua = ink2_hfma2(u7, x7, ua);
                        const float dj = dk[j];
                        const float2 gf = __half22float2(ga), uf = __half22float2(ua);
                        ag[j] += (gd * dj) * (gf.x + gf.y);
                        au[j] += (ud * dj) * (uf.x + uf.y);
                    }
                }
            }
            __syncthreads();
        }

        if (row < ff) {
            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                float gv = ag[j], uv = au[j];
                gv += __shfl_xor_sync(0xffffffffu, gv, 2, 4);
                uv += __shfl_xor_sync(0xffffffffu, uv, 2, 4);
                gv += __shfl_xor_sync(0xffffffffu, gv, 1, 4);
                uv += __shfl_xor_sync(0xffffffffu, uv, 1, 4);
                if (sub == 0 && j < n_tile) {
                    const int pair = s_pairs[j];
                    const int tok  = pair >> 4;
                    const int slot = pair & 15;
                    act[(static_cast<size_t>(tok) * k_used + slot) * ff + row] =
                        situ_act(gv, uv, beta, lb);
                }
            }
        }
        __syncthreads();
    }
}

template <int BT, int KT>
__global__ void inkling2_moe_down_grouped_q8_0_soa_f16_kernel(
    const int4 *         __restrict__ dn_q,
    const uint16_t *     __restrict__ dn_s,
    const __half2 *      __restrict__ a_h2,
    const float *        __restrict__ a_d,
    const float *        __restrict__ top_w,
    const Ink2TileDesc * __restrict__ tiles,
    const int32_t *      __restrict__ n_tiles_ptr,
    const int32_t *      __restrict__ sorted_pairs,
    float *              __restrict__ down_slots,
    int ff_blks, int K, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Ink2TileDesc td = tiles[tile_idx];
    const int eid   = td.eid;
    const int start = td.start;
    const int end   = td.start + td.count;

    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    __shared__ float   s_tw[BT];
    auto * s_ah2 = reinterpret_cast<__half2 *>(smem_raw); // [KT, BT, 16 half2]
    auto * s_ad  = reinterpret_cast<float *>(s_ah2 + KT * BT * 16);

    const int tid    = threadIdx.x; // 0..127 -> 32 rows/CTA (LPR=4!)
    const int nthr   = blockDim.x;
    const int row    = blockIdx.x * (nthr >> 2) + (tid >> 2);
    const int sub    = tid & 3;
    const int b_lane = sub >> 1;
    const int hb     = sub & 1;

    const size_t exp_stride = static_cast<size_t>(K) * ff_blks;
    const size_t d_off      = static_cast<size_t>(eid) * exp_stride + static_cast<size_t>(row) * ff_blks;
    const int4 *     dq = dn_q + d_off * 2;
    const uint16_t * ds = dn_s + d_off;
    const __half2 neg1152 = __floats2half2_rn(-1152.0f, -1152.0f);

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) {
            const int pair = sorted_pairs[base_i + tid];
            s_pairs[tid] = pair;
            const int tok = pair >> 4, slot = pair & 15;
            s_tw[tid] = top_w[static_cast<size_t>(tok) * k_used + slot];
        }
        __syncthreads();

        float acc[BT];
        #pragma unroll
        for (int j = 0; j < BT; ++j) acc[j] = 0.0f;

        for (int k0 = 0; k0 < ff_blks; k0 += KT) {
            const int kc = (k0 + KT < ff_blks) ? KT : (ff_blks - k0);
            const int tot_i4 = kc * BT * 4;
            for (int idx = tid; idx < tot_i4; idx += nthr) {
                const int c   = idx & 3;
                const int rem = idx >> 2;
                const int j   = rem % BT;
                const int bk  = rem / BT;
                if (j < n_tile) {
                    const int pair = s_pairs[j];
                    const int tok = pair >> 4, slot = pair & 15;
                    const size_t a_row = (static_cast<size_t>(tok) * k_used + slot) * ff_blks + k0 + bk;
                    reinterpret_cast<int4 *>(s_ah2)[idx] = reinterpret_cast<const int4 *>(a_h2 + a_row * 16)[c];
                } else {
                    reinterpret_cast<int4 *>(s_ah2)[idx] = make_int4(0, 0, 0, 0);
                }
            }
            const int tot_d = kc * BT;
            for (int idx = tid; idx < tot_d; idx += nthr) {
                const int j  = idx % BT;
                const int bk = idx / BT;
                if (j < n_tile) {
                    const int pair = s_pairs[j];
                    const int tok = pair >> 4, slot = pair & 15;
                    s_ad[idx] = a_d[(static_cast<size_t>(tok) * k_used + slot) * ff_blks + k0 + bk];
                } else {
                    s_ad[idx] = 0.0f;
                }
            }
            __syncthreads();

            if (row < K) {
                for (int bk = b_lane; bk < kc; bk += 2) {
                    const int b = k0 + bk;
                    const int4 dw4 = __ldg(&dq[b * 2 + hb]);
                    const float dd = ink2_half_bits_to_f32(__ldg(&ds[b]));
                    const uint32_t du0 = static_cast<uint32_t>(dw4.x) ^ 0x80808080u;
                    const uint32_t du1 = static_cast<uint32_t>(dw4.y) ^ 0x80808080u;
                    const uint32_t du2 = static_cast<uint32_t>(dw4.z) ^ 0x80808080u;
                    const uint32_t du3 = static_cast<uint32_t>(dw4.w) ^ 0x80808080u;
                    const __half2 w0 = ink2_hadd2(ink2_i8x2_to_h2_lo(du0), neg1152), w1 = ink2_hadd2(ink2_i8x2_to_h2_hi(du0), neg1152);
                    const __half2 w2 = ink2_hadd2(ink2_i8x2_to_h2_lo(du1), neg1152), w3 = ink2_hadd2(ink2_i8x2_to_h2_hi(du1), neg1152);
                    const __half2 w4 = ink2_hadd2(ink2_i8x2_to_h2_lo(du2), neg1152), w5 = ink2_hadd2(ink2_i8x2_to_h2_hi(du2), neg1152);
                    const __half2 w6 = ink2_hadd2(ink2_i8x2_to_h2_lo(du3), neg1152), w7 = ink2_hadd2(ink2_i8x2_to_h2_hi(du3), neg1152);

                    const __half2 * ak = s_ah2 + bk * BT * 16 + hb * 8;
                    const float *   dk = s_ad  + bk * BT;
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) {
                        const __half2 * aj = ak + j * 16;
                        __half2 r = ink2_hmul2(w0, aj[0]); r = ink2_hfma2(w1, aj[1], r);
                        r = ink2_hfma2(w2, aj[2], r); r = ink2_hfma2(w3, aj[3], r);
                        r = ink2_hfma2(w4, aj[4], r); r = ink2_hfma2(w5, aj[5], r);
                        r = ink2_hfma2(w6, aj[6], r); r = ink2_hfma2(w7, aj[7], r);
                        const float2 rf = __half22float2(r);
                        acc[j] += (dd * dk[j]) * (rf.x + rf.y);
                    }
                }
            }
            __syncthreads();
        }

        if (row < K) {
            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                float v = acc[j];
                v += __shfl_xor_sync(0xffffffffu, v, 2, 4);
                v += __shfl_xor_sync(0xffffffffu, v, 1, 4);
                if (sub == 0 && j < n_tile) {
                    const int pair = s_pairs[j];
                    const int tok = pair >> 4, slot = pair & 15;
                    down_slots[(static_cast<size_t>(tok) * k_used + slot) * K + row] = s_tw[j] * v;
                }
            }
        }
        __syncthreads();
    }
}

// sm_61+ / sm_52 K-tiled BT=16, KT=12, LPR=4 dp4a Q8_0 grouped MoE (ZERO register spills!)
template <int BT, int KT>
__global__ void inkling2_moe_gate_up_grouped_q8_0_ktiled_dp4a_kernel(
    const int4 *          __restrict__ gu_q,
    const uint16_t *      __restrict__ gu_s,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const Ink2TileDesc *  __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ act,
    int K_blocks, int ff, int k_used, float beta, float lb) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Ink2TileDesc td = tiles[tile_idx];
    const int eid   = td.eid;
    const int start = td.start;
    const int end   = td.start + td.count;

    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    auto * s_xq = reinterpret_cast<int4 *>(smem_raw); // [KT, BT, 2 int4]
    auto * s_xd = reinterpret_cast<float *>(s_xq + KT * BT * 2);

    const int tid    = threadIdx.x; // 0..127 -> 32 rows/CTA (LPR=4!)
    const int nthr   = blockDim.x;
    const int row    = blockIdx.x * (nthr >> 2) + (tid >> 2);
    const int sub    = tid & 3;
    const int b_lane = sub >> 1;
    const int hb     = sub & 1;

    const size_t exp_stride = static_cast<size_t>(2 * ff) * K_blocks;
    const size_t g_off      = static_cast<size_t>(eid) * exp_stride + static_cast<size_t>(row) * K_blocks;
    const size_t u_off      = g_off + static_cast<size_t>(ff) * K_blocks;
    const int4 *     gq = gu_q + g_off * 2;
    const uint16_t * gs = gu_s + g_off;
    const int4 *     uq = gu_q + u_off * 2;
    const uint16_t * us = gu_s + u_off;

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) s_pairs[tid] = sorted_pairs[base_i + tid];
        __syncthreads();

        float ag[BT], au[BT];
        #pragma unroll
        for (int j = 0; j < BT; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

        for (int k0 = 0; k0 < K_blocks; k0 += KT) {
            const int kc = (k0 + KT < K_blocks) ? KT : (K_blocks - k0);
            const int tot_i4 = kc * BT * 2;
            for (int idx = tid; idx < tot_i4; idx += nthr) {
                const int c   = idx & 1;
                const int rem = idx >> 1;
                const int j   = rem % BT;
                const int bk  = rem / BT;
                if (j < n_tile) {
                    const int tok = s_pairs[j] >> 4;
                    s_xq[idx] = reinterpret_cast<const int4 *>(x_q8[static_cast<size_t>(tok) * K_blocks + k0 + bk].qs)[c];
                } else {
                    s_xq[idx] = make_int4(0, 0, 0, 0);
                }
            }
            const int tot_d = kc * BT;
            for (int idx = tid; idx < tot_d; idx += nthr) {
                const int j  = idx % BT;
                const int bk = idx / BT;
                s_xd[idx] = (j < n_tile) ? x_q8[static_cast<size_t>(s_pairs[j] >> 4) * K_blocks + k0 + bk].d : 0.0f;
            }
            __syncthreads();

            if (row < ff) {
                for (int bk = b_lane; bk < kc; bk += 2) {
                    const int b = k0 + bk;
                    const int4 gw = __ldg(&gq[b * 2 + hb]);
                    const int4 uw = __ldg(&uq[b * 2 + hb]);
                    const float gd = ink2_half_bits_to_f32(__ldg(&gs[b]));
                    const float ud = ink2_half_bits_to_f32(__ldg(&us[b]));
                    const int4 *  xk = s_xq + bk * BT * 2 + hb;
                    const float * dk = s_xd + bk * BT;
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) {
                        const int4 x4 = xk[j * 2];
                        int sg = dp4a_s32(gw.x, x4.x, dp4a_s32(gw.y, x4.y, dp4a_s32(gw.z, x4.z, dp4a_s32(gw.w, x4.w, 0))));
                        int su = dp4a_s32(uw.x, x4.x, dp4a_s32(uw.y, x4.y, dp4a_s32(uw.z, x4.z, dp4a_s32(uw.w, x4.w, 0))));
                        const float dj = dk[j];
                        ag[j] += (gd * dj) * static_cast<float>(sg);
                        au[j] += (ud * dj) * static_cast<float>(su);
                    }
                }
            }
            __syncthreads();
        }

        if (row < ff) {
            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                float gv = ag[j], uv = au[j];
                gv += __shfl_xor_sync(0xffffffffu, gv, 2, 4);
                uv += __shfl_xor_sync(0xffffffffu, uv, 2, 4);
                gv += __shfl_xor_sync(0xffffffffu, gv, 1, 4);
                uv += __shfl_xor_sync(0xffffffffu, uv, 1, 4);
                if (sub == 0 && j < n_tile) {
                    const int pair = s_pairs[j];
                    const int tok  = pair >> 4;
                    const int slot = pair & 15;
                    act[(static_cast<size_t>(tok) * k_used + slot) * ff + row] =
                        situ_act(gv, uv, beta, lb);
                }
            }
        }
        __syncthreads();
    }
}

template <int BT, int KT>
__global__ void inkling2_moe_down_grouped_q8_0_ktiled_dp4a_kernel(
    const int4 *          __restrict__ dn_q,
    const uint16_t *      __restrict__ dn_s,
    const BlockQ8_1_Act * __restrict__ a_q8,
    const float *         __restrict__ top_w,
    const Ink2TileDesc *  __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots,
    int ff_blks, int K, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const Ink2TileDesc td = tiles[tile_idx];
    const int eid   = td.eid;
    const int start = td.start;
    const int end   = td.start + td.count;

    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    __shared__ float   s_tw[BT];
    auto * s_aq = reinterpret_cast<int4 *>(smem_raw); // [KT, BT, 2 int4]
    auto * s_ad = reinterpret_cast<float *>(s_aq + KT * BT * 2);

    const int tid    = threadIdx.x;
    const int nthr   = blockDim.x;
    const int row    = blockIdx.x * (nthr >> 2) + (tid >> 2);
    const int sub    = tid & 3;
    const int b_lane = sub >> 1;
    const int hb     = sub & 1;

    const size_t exp_stride = static_cast<size_t>(K) * ff_blks;
    const size_t d_off      = static_cast<size_t>(eid) * exp_stride + static_cast<size_t>(row) * ff_blks;
    const int4 *     dq = dn_q + d_off * 2;
    const uint16_t * ds = dn_s + d_off;

    for (int base_i = start; base_i < end; base_i += BT) {
        const int n_tile = (base_i + BT < end) ? BT : (end - base_i);
        if (tid < n_tile) {
            const int pair = sorted_pairs[base_i + tid];
            s_pairs[tid] = pair;
            const int tok = pair >> 4, slot = pair & 15;
            s_tw[tid] = top_w[static_cast<size_t>(tok) * k_used + slot];
        }
        __syncthreads();

        float acc[BT];
        #pragma unroll
        for (int j = 0; j < BT; ++j) acc[j] = 0.0f;

        for (int k0 = 0; k0 < ff_blks; k0 += KT) {
            const int kc = (k0 + KT < ff_blks) ? KT : (ff_blks - k0);
            const int tot_i4 = kc * BT * 2;
            for (int idx = tid; idx < tot_i4; idx += nthr) {
                const int c   = idx & 1;
                const int rem = idx >> 1;
                const int j   = rem % BT;
                const int bk  = rem / BT;
                if (j < n_tile) {
                    const int pair = s_pairs[j];
                    const int tok = pair >> 4, slot = pair & 15;
                    s_aq[idx] = reinterpret_cast<const int4 *>(
                        a_q8[(static_cast<size_t>(tok) * k_used + slot) * ff_blks + k0 + bk].qs)[c];
                } else {
                    s_aq[idx] = make_int4(0, 0, 0, 0);
                }
            }
            const int tot_d = kc * BT;
            for (int idx = tid; idx < tot_d; idx += nthr) {
                const int j  = idx % BT;
                const int bk = idx / BT;
                if (j < n_tile) {
                    const int pair = s_pairs[j];
                    const int tok = pair >> 4, slot = pair & 15;
                    s_ad[idx] = a_q8[(static_cast<size_t>(tok) * k_used + slot) * ff_blks + k0 + bk].d;
                } else {
                    s_ad[idx] = 0.0f;
                }
            }
            __syncthreads();

            if (row < K) {
                for (int bk = b_lane; bk < kc; bk += 2) {
                    const int b = k0 + bk;
                    const int4 dw = __ldg(&dq[b * 2 + hb]);
                    const float dd = ink2_half_bits_to_f32(__ldg(&ds[b]));
                    const int4 *  ak = s_aq + bk * BT * 2 + hb;
                    const float * dk = s_ad + bk * BT;
                    #pragma unroll
                    for (int j = 0; j < BT; ++j) {
                        const int4 a4 = ak[j * 2];
                        int s = dp4a_s32(dw.x, a4.x, dp4a_s32(dw.y, a4.y, dp4a_s32(dw.z, a4.z, dp4a_s32(dw.w, a4.w, 0))));
                        acc[j] += (dd * dk[j]) * static_cast<float>(s);
                    }
                }
            }
            __syncthreads();
        }

        if (row < K) {
            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                float v = acc[j];
                v += __shfl_xor_sync(0xffffffffu, v, 2, 4);
                v += __shfl_xor_sync(0xffffffffu, v, 1, 4);
                if (sub == 0 && j < n_tile) {
                    const int pair = s_pairs[j];
                    const int tok = pair >> 4, slot = pair & 15;
                    down_slots[(static_cast<size_t>(tok) * k_used + slot) * K + row] = s_tw[j] * v;
                }
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Stage 5B: Fused Decode Attention Pre-Ops (T == 1)
// ============================================================================
__global__ void inkling2_kv_conv_norm_write_q8_0_t1_kernel(
    const float * __restrict__ k_raw, const float * __restrict__ k_w, const float * __restrict__ k_dyn, float * __restrict__ k_state,
    const float * __restrict__ v_raw, const float * __restrict__ v_w, const float * __restrict__ v_dyn, float * __restrict__ v_state,
    void * __restrict__ k_cache, void * __restrict__ v_cache, const int32_t * __restrict__ pos,
    int n_head_kv, int cache_cap, float qk_clip, float rms_eps) {

    const int is_v = (blockIdx.x >= n_head_kv) ? 1 : 0;
    const int h    = is_v ? (blockIdx.x - n_head_kv) : blockIdx.x;
    const int d    = threadIdx.x; // 0..127 (head_dim == 128)
    const int lane = d & 31;
    const int warp = d >> 5;      // 0..3 (4 Q8_0 blocks per head!)
    const int C    = n_head_kv * 128;
    const int c    = h * 128 + d;

    const float * x_raw = is_v ? v_raw : k_raw;
    const float * w_ptr = is_v ? v_w   : k_w;
    const float * dyn   = is_v ? v_dyn : k_dyn;
    float *       state = is_v ? v_state : k_state;

    const float s0 = state[0 * C + c];
    const float s1 = state[1 * C + c];
    const float s2 = state[2 * C + c];
    const float xt = x_raw[c];
    const float w0 = w_ptr[c * 4 + 0];
    const float w1 = w_ptr[c * 4 + 1];
    const float w2 = w_ptr[c * 4 + 2];
    const float w3 = w_ptr[c * 4 + 3];
    const float inv_scale = dyn ? (1.0f / 3.0f) : rsqrtf(5.0f);

    float acc = s0 * w0 + s1 * w1 + s2 * w2 + xt * w3;
    if (dyn) {
        const float * dt = dyn + h * 4;
        acc += dt[0] * xt + dt[1] * s2 + dt[2] * s1 + dt[3] * s0;
    }
    float val = (xt + acc) * inv_scale;
    state[0 * C + c] = s1;
    state[1 * C + c] = s2;
    state[2 * C + c] = xt;

    __shared__ float s_ss[4];
    float ws = warp_sum(val * val);
    if (lane == 0) s_ss[warp] = ws;
    __syncthreads();
    const float ss = s_ss[0] + s_ss[1] + s_ss[2] + s_ss[3];
    if (!is_v) {
        const float norm = sqrtf(ss) + 1e-6f;
        val *= fminf(1.0f, qk_clip / norm);
    } else {
        val *= rsqrtf(ss / 128.0f + rms_eps);
    }

    const int slot = pos[0] % cache_cap;
    const int total_b = n_head_kv * 4;
    const size_t dst_b = static_cast<size_t>(slot) * total_b + (h * 4 + warp);
    const size_t qs_bytes = static_cast<size_t>(cache_cap) * n_head_kv * 128;
    void * dst_raw = is_v ? v_cache : k_cache;
    auto * dst_qs = static_cast<int8_t *>(dst_raw);
    auto * dst_sc = reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(dst_raw) + qs_bytes);

    const float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(val)), 0);
    const float sc   = amax / 127.0f;
    const float inv  = (sc > 0.0f) ? (127.0f / amax) : 0.0f;
    if (lane == 0) {
        __half dh = __float2half_rn(sc);
        dst_sc[dst_b] = *reinterpret_cast<uint16_t *>(&dh);
    }
    dst_qs[dst_b * 32 + lane] = static_cast<int8_t>(__float2int_rn(val * inv));
}

__global__ void inkling2_qr_l2_clip_fused_t1_kernel(
    float * __restrict__ q, const float * __restrict__ q_scale,
    float * __restrict__ r, const float * __restrict__ r_scale,
    const int32_t * __restrict__ pos,
    float qk_clip, float rel_clip, float log_floor, float log_alpha,
    int n_head, int head_dim, int n_rel) {

    const int lane = threadIdx.x;
    const int is_r = (blockIdx.x >= n_head) ? 1 : 0;
    const int h    = is_r ? (blockIdx.x - n_head) : blockIdx.x;
    const int dim  = is_r ? n_rel : head_dim;
    float *       row = (is_r ? r : q) + static_cast<size_t>(h) * dim;
    const float * sr  = (is_r ? r_scale : q_scale);
    if (sr) sr += static_cast<size_t>(h) * dim;
    const float thr = is_r ? rel_clip : qk_clip;

    float ss = 0.0f;
    for (int d = lane; d < dim; d += 32) {
        float v = row[d];
        if (sr) v *= sr[d];
        row[d] = v;
        ss += v * v;
    }
    ss = __shfl_sync(0xffffffffu, warp_sum(ss), 0);
    const float norm = sqrtf(ss) + 1e-6f;
    float mul = fminf(1.0f, thr / norm);
    if (pos != nullptr) {
        const float ratio = fmaxf(1.0f, static_cast<float>(pos[0] + 1) / log_floor);
        mul *= (1.0f + log_alpha * logf(ratio));
    }
    for (int d = lane; d < dim; d += 32) row[d] *= mul;
}

// ============================================================================
// Stage 5C: Native sm_60 half2 0x6400-PRMT Q8_0 Decode GEMV + 3-Input 8-Proj Fused QKV
// ============================================================================
struct Ink2FusedQ8Projs8 {
    const int4 *     qs[8];
    const uint16_t * sc[8];
    float *          out[8];
    int              row_end[8];
    int              N[8];
    int              in_idx[8];
    int              n_projs;
};

__global__ void inkling2_rmsnorm_quant_f16_contig_kernel(
    Ink2Ptr3 xs, Ink2Ptr3 ws,
    __half2 * __restrict__ out_h2,
    float *   __restrict__ out_d,
    int n_embd, int n_inputs, int T, float eps) {

    const int inp = blockIdx.y;
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;

    const float4 v = reinterpret_cast<const float4 *>(xs.p[inp] + static_cast<size_t>(t) * n_embd)[tid];
    float ss = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    __shared__ float s_red[32];
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < nwarps) ? s_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) s_red[0] = rsqrtf(tot / static_cast<float>(n_embd) + eps);
    }
    __syncthreads();
    const float inv = s_red[0];
    const float4 wv = reinterpret_cast<const float4 *>(ws.p[inp])[tid];
    const float x0 = v.x * inv * wv.x;
    const float x1 = v.y * inv * wv.y;
    const float x2 = v.z * inv * wv.z;
    const float x3 = v.w * inv * wv.w;

    float am = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4, 8));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2, 8));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1, 8));
    const float isc = (am > 0.0f) ? (1.0f / am) : 0.0f;

    const int K_blocks = n_embd >> 5;
    const int b = tid >> 3;
    const int s = tid & 7;
    const size_t blk_idx = (static_cast<size_t>(inp) * T + t) * K_blocks + b;
    out_h2[blk_idx * 16 + s * 2 + 0] = __halves2half2(__float2half_rn(x0 * isc), __float2half_rn(x1 * isc));
    out_h2[blk_idx * 16 + s * 2 + 1] = __halves2half2(__float2half_rn(x2 * isc), __float2half_rn(x3 * isc));
    if (s == 0) out_d[blk_idx] = am;
}

__global__ void inkling2_sigmoid_gate_quant_f16_contig_kernel(
    const float * __restrict__ x,
    const float * __restrict__ gate,
    __half2 *     __restrict__ out_h2,
    float *       __restrict__ out_d,
    int total_blocks) {
    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= total_blocks) return;
    const int lane = threadIdx.x;
    const size_t idx = static_cast<size_t>(b) * 32 + lane;
    const float g = gate[idx];
    const float v = x[idx] * (1.0f / (1.0f + __expf(-g)));
    const float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(v)), 0);
    const float inv  = (amax > 0.0f) ? (1.0f / amax) : 0.0f;
    const float vn   = v * inv;
    const float vn_hi = __shfl_down_sync(0xffffffffu, vn, 1);
    if ((lane & 1) == 0) {
        out_h2[static_cast<size_t>(b) * 16 + (lane >> 1)] = __halves2half2(__float2half_rn(vn), __float2half_rn(vn_hi));
    }
    if (lane == 0) out_d[b] = amax;
}

template <int LPR>
__global__ void inkling2_gemv_q8_0_h2_kernel(
    const int4 *     __restrict__ W_qs,
    const uint16_t * __restrict__ W_sc,
    const __half2 *  __restrict__ X_h2,
    const float *    __restrict__ X_d,
    float *          __restrict__ Y,
    int K_blocks, int N, int T) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x  = reinterpret_cast<__half2 *>(smem_raw);
    auto * s_xd = reinterpret_cast<float *>(s_x + K_blocks * 16);

    const int t    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;
    const int4 * xi = reinterpret_cast<const int4 *>(X_h2 + static_cast<size_t>(t) * K_blocks * 16);
    int4 *       xo = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < K_blocks * 4; i += nthr) xo[i] = xi[i];
    for (int i = tid; i < K_blocks; i += nthr) s_xd[i] = X_d[static_cast<size_t>(t) * K_blocks + i];
    __syncthreads();

    const int row = blockIdx.x * (nthr / LPR) + (tid / LPR);
    if (row >= N || t >= T) return;
    const int sub = tid & (LPR - 1);
    const int bw  = sub >> 1;
    const int hb  = sub & 1;

    const int4 *     wr = W_qs + static_cast<size_t>(row) * (K_blocks * 2);
    const uint16_t * sr = W_sc + static_cast<size_t>(row) * K_blocks;
    const __half2 neg1152 = __floats2half2_rn(-1152.0f, -1152.0f);

    float acc = 0.0f;
    for (int b = bw; b < K_blocks; b += (LPR >> 1)) {
        const int4 w4 = __ldg(&wr[b * 2 + hb]);
        const __half2 * xp = s_x + b * 16 + hb * 8;
        const uint32_t u0 = static_cast<uint32_t>(w4.x) ^ 0x80808080u;
        const uint32_t u1 = static_cast<uint32_t>(w4.y) ^ 0x80808080u;
        const uint32_t u2 = static_cast<uint32_t>(w4.z) ^ 0x80808080u;
        const uint32_t u3 = static_cast<uint32_t>(w4.w) ^ 0x80808080u;
        __half2 a0 = ink2_hmul2(ink2_hadd2(ink2_i8x2_to_h2_lo(u0), neg1152), xp[0]);
        __half2 a1 = ink2_hmul2(ink2_hadd2(ink2_i8x2_to_h2_hi(u0), neg1152), xp[1]);
        a0 = ink2_hfma2(ink2_hadd2(ink2_i8x2_to_h2_lo(u1), neg1152), xp[2], a0);
        a1 = ink2_hfma2(ink2_hadd2(ink2_i8x2_to_h2_hi(u1), neg1152), xp[3], a1);
        a0 = ink2_hfma2(ink2_hadd2(ink2_i8x2_to_h2_lo(u2), neg1152), xp[4], a0);
        a1 = ink2_hfma2(ink2_hadd2(ink2_i8x2_to_h2_hi(u2), neg1152), xp[5], a1);
        a0 = ink2_hfma2(ink2_hadd2(ink2_i8x2_to_h2_lo(u3), neg1152), xp[6], a0);
        a1 = ink2_hfma2(ink2_hadd2(ink2_i8x2_to_h2_hi(u3), neg1152), xp[7], a1);
        const float2 sf = __half22float2(ink2_hadd2(a0, a1));
        acc += (ink2_half_bits_to_f32(__ldg(&sr[b])) * s_xd[b]) * (sf.x + sf.y);
    }
    #pragma unroll
    for (int off = LPR >> 1; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off, LPR);
    if (sub == 0) Y[static_cast<size_t>(t) * N + row] = acc;
}

__global__ void inkling2_qkv_fused_q8_0_h2_kernel(
    Ink2FusedQ8Projs8 P,
    const __half2 * __restrict__ X_h2,
    const float *   __restrict__ X_d,
    int K_blocks, int n_inputs, int T, int total_rows) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x  = reinterpret_cast<__half2 *>(smem_raw);
    auto * s_xd = reinterpret_cast<float *>(s_x + n_inputs * K_blocks * 16);

    const int t    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;
    for (int inp = 0; inp < n_inputs; ++inp) {
        const size_t base_blk = (static_cast<size_t>(inp) * T + t) * K_blocks;
        const int4 * xi = reinterpret_cast<const int4 *>(X_h2 + base_blk * 16);
        int4 *       xo = reinterpret_cast<int4 *>(s_x + inp * K_blocks * 16);
        for (int i = tid; i < K_blocks * 4; i += nthr) xo[i] = xi[i];
        for (int i = tid; i < K_blocks; i += nthr) s_xd[inp * K_blocks + i] = X_d[base_blk + i];
    }
    __syncthreads();

    constexpr int LPR = 16;
    const int g_row = blockIdx.x * (nthr / LPR) + (tid / LPR);
    if (g_row >= total_rows) return;

    int pid = 0, r0 = 0;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        if (p < P.n_projs && g_row >= P.row_end[p]) {
            pid = p + 1;
            r0  = P.row_end[p];
        }
    }
    const int row = g_row - r0;
    const int inp = P.in_idx[pid];
    const int sub = tid & (LPR - 1);
    const int bw  = sub >> 1;
    const int hb  = sub & 1;

    const int4 *     wr  = P.qs[pid] + static_cast<size_t>(row) * (K_blocks * 2);
    const uint16_t * sr  = P.sc[pid] + static_cast<size_t>(row) * K_blocks;
    const __half2 *  sxi = s_x  + inp * K_blocks * 16;
    const float *    sdi = s_xd + inp * K_blocks;
    const __half2 neg1152 = __floats2half2_rn(-1152.0f, -1152.0f);

    float acc = 0.0f;
    for (int b = bw; b < K_blocks; b += (LPR >> 1)) {
        const int4 w4 = __ldg(&wr[b * 2 + hb]);
        const __half2 * xp = sxi + b * 16 + hb * 8;
        const uint32_t u0 = static_cast<uint32_t>(w4.x) ^ 0x80808080u;
        const uint32_t u1 = static_cast<uint32_t>(w4.y) ^ 0x80808080u;
        const uint32_t u2 = static_cast<uint32_t>(w4.z) ^ 0x80808080u;
        const uint32_t u3 = static_cast<uint32_t>(w4.w) ^ 0x80808080u;
        __half2 a0 = ink2_hmul2(ink2_hadd2(ink2_i8x2_to_h2_lo(u0), neg1152), xp[0]);
        __half2 a1 = ink2_hmul2(ink2_hadd2(ink2_i8x2_to_h2_hi(u0), neg1152), xp[1]);
        a0 = ink2_hfma2(ink2_hadd2(ink2_i8x2_to_h2_lo(u1), neg1152), xp[2], a0);
        a1 = ink2_hfma2(ink2_hadd2(ink2_i8x2_to_h2_hi(u1), neg1152), xp[3], a1);
        a0 = ink2_hfma2(ink2_hadd2(ink2_i8x2_to_h2_lo(u2), neg1152), xp[4], a0);
        a1 = ink2_hfma2(ink2_hadd2(ink2_i8x2_to_h2_hi(u2), neg1152), xp[5], a1);
        a0 = ink2_hfma2(ink2_hadd2(ink2_i8x2_to_h2_lo(u3), neg1152), xp[6], a0);
        a1 = ink2_hfma2(ink2_hadd2(ink2_i8x2_to_h2_hi(u3), neg1152), xp[7], a1);
        const float2 sf = __half22float2(ink2_hadd2(a0, a1));
        acc += (ink2_half_bits_to_f32(__ldg(&sr[b])) * sdi[b]) * (sf.x + sf.y);
    }
    #pragma unroll
    for (int off = LPR >> 1; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off, LPR);
    if (sub == 0) P.out[pid][static_cast<size_t>(t) * P.N[pid] + row] = acc;
}

// 2D. Warp-Only Sigmoid Top-K Router for n_expert <= 128 (ZERO __syncthreads!)
// ============================================================================
__global__ void inkling2_moe_route_sigmoid_w32_kernel(
    const float * __restrict__ logits,
    const float * __restrict__ bias,
    int32_t *     __restrict__ top_ids,
    float *       __restrict__ top_w,
    int n_expert, int k, bool renorm, float scale) {

    const int t    = blockIdx.x;
    const int lane = threadIdx.x; // 0..31 -> 4 experts per lane!
    const float * row = logits + static_cast<size_t>(t) * n_expert;

    float p[4], sv[4];
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int e = lane + m * 32;
        if (e < n_expert) {
            p[m]  = 1.0f / (1.0f + expf(-row[e]));
            sv[m] = bias ? (p[m] + bias[e]) : p[m];
        } else {
            p[m]  = 0.0f;
            sv[m] = -FLT_MAX;
        }
    }

    int32_t out_i[16];
    float   out_w[16];
    float   w_sum = 0.0f;

    for (int j = 0; j < k; ++j) {
        float   bv = sv[0];
        int32_t bi = lane;
        float   bp = p[0];
        #pragma unroll
        for (int m = 1; m < 4; ++m) {
            if (sv[m] > bv) { bv = sv[m]; bi = lane + m * 32; bp = p[m]; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float   ov = __shfl_down_sync(0xffffffffu, bv, off);
            int32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
            float   op = __shfl_down_sync(0xffffffffu, bp, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bp = op; }
        }
        const int32_t chosen = __shfl_sync(0xffffffffu, bi, 0);
        const float   chosen_p = __shfl_sync(0xffffffffu, bp, 0);
        if (lane == 0) {
            out_i[j] = chosen;
            out_w[j] = chosen_p;
            w_sum   += chosen_p;
        }
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (lane + m * 32 == chosen) sv[m] = -FLT_MAX;
        }
    }

    if (lane == 0) {
        const float mul = (renorm && w_sum > 0.0f) ? (scale / w_sum) : scale;
        int32_t * oi = top_ids + static_cast<size_t>(t) * k;
        float *   ow = top_w   + static_cast<size_t>(t) * k;
        for (int j = 0; j < k; ++j) {
            oi[j] = out_i[j];
            ow[j] = out_w[j] * mul;
        }
    }
}


// ============================================================================
// 2E. Fused Sigmoid Gate Mul + Q8_1 Activation Quantization (for Wo decode!)
//     Eliminates 48 kernel launches/tok + 48 d_attn_out global round-trips!
// ============================================================================
__global__ void inkling2_sigmoid_gate_quant_q8_1_kernel(
    const float *  __restrict__ x,
    const float *  __restrict__ gate,
    BlockQ8_1_Act * __restrict__ qx,
    int total_blocks) {

    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= total_blocks) return;
    const int lane = threadIdx.x;
    const size_t idx = static_cast<size_t>(b) * 32 + lane;
    const float v = x[idx] * (1.0f / (1.0f + __expf(-gate[idx])));
    const float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(v)), 0);
    const float d     = amax / 127.0f;
    const float inv_d = (d > 0.0f) ? (127.0f / amax) : 0.0f;
    const int   q     = __float2int_rn(v * inv_d);
    qx[b].qs[lane]    = static_cast<int8_t>(q);
    if (lane == 0) {
        qx[b].d     = d;
        qx[b].s0    = 0;
        qx[b].s1    = 0;
        qx[b].s_all = 0.0f;
    }
}

// High-occupancy F32 GEMV for MoE router (N=128, K=2304 -> 64 CTAs of 2 warps on 56 SMs!)
__global__ void inkling2_router_gemv_f32_kernel(
    const float * __restrict__ W,
    const float * __restrict__ X,
    float *       __restrict__ Y,
    int K, int N) {
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    if (row >= N) return;
    const int lane = threadIdx.x;
    const int K4   = K >> 2;
    const auto * w4 = reinterpret_cast<const float4 *>(W + static_cast<size_t>(row) * K);
    const auto * x4 = reinterpret_cast<const float4 *>(X + static_cast<size_t>(t) * K);
    float acc = 0.0f;
    for (int k = lane; k < K4; k += 32) {
        const float4 wv = __ldg(&w4[k]);
        const float4 xv = x4[k];
        acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) Y[static_cast<size_t>(t) * N + row] = acc;
}

// Fast sm_60 FP16 Q8_0 dequant + F32<->F16 cast helpers for P100 18.7 TFLOPS cublasHgemm prefill!
__global__ void inkling2_f32_to_f16_vec_kernel(const float * __restrict__ src, __half * __restrict__ dst, int64_t n) {
    const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < n) {
        const float2 v = reinterpret_cast<const float2 *>(src)[i >> 1];
        reinterpret_cast<__half2 *>(dst)[i >> 1] = __floats2half2_rn(v.x, v.y);
    } else if (i < n) {
        dst[i] = __float2half_rn(src[i]);
    }
}

__global__ void inkling2_f16_to_f32_vec_kernel(const __half * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < n) {
        const __half2 v = reinterpret_cast<const __half2 *>(src)[i >> 1];
        reinterpret_cast<float2 *>(dst)[i >> 1] = __half22float2(v);
    } else if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

__global__ void inkling2_dequant_q8_0_soa_f16_kernel(
    const int8_t *   __restrict__ src_qs,
    const uint16_t * __restrict__ src_sc,
    __half *         __restrict__ dst,
    int64_t n_blocks) {
    // 4 threads per 32-element block: each thread reads 8 int8 (int2) and writes 8 __half (int4)!
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t b   = idx >> 2;
    if (b >= n_blocks) return;
    const int q = static_cast<int>(idx & 3);
    const int2 v = __ldg(&reinterpret_cast<const int2 *>(src_qs)[idx]);
    __half dh;
    *reinterpret_cast<uint16_t *>(&dh) = __ldg(&src_sc[b]);
    const __half2 dh2 = __half2half2(dh);
    const __half2 neg1152 = __floats2half2_rn(-1152.0f, -1152.0f);
    const uint32_t u0 = static_cast<uint32_t>(v.x) ^ 0x80808080u;
    const uint32_t u1 = static_cast<uint32_t>(v.y) ^ 0x80808080u;
    __half2 out4[4];
    out4[0] = ink2_hmul2(ink2_hadd2(ink2_i8x2_to_h2_lo(u0), neg1152), dh2);
    out4[1] = ink2_hmul2(ink2_hadd2(ink2_i8x2_to_h2_hi(u0), neg1152), dh2);
    out4[2] = ink2_hmul2(ink2_hadd2(ink2_i8x2_to_h2_lo(u1), neg1152), dh2);
    out4[3] = ink2_hmul2(ink2_hadd2(ink2_i8x2_to_h2_hi(u1), neg1152), dh2);
    reinterpret_cast<int4 *>(dst)[b * 4 + q] = *reinterpret_cast<const int4 *>(out4);
}

// Fast FlashAttention softmax+relbias+rescale taking F16 s_tile (for sm_60 native FP16 HGEMM!)
__global__ void inkling2_flash_softmax_relbias_f16s_kernel(
    const __half *  __restrict__ s_tile,     // [n_kv, T*gqa, kc] F16
    __half *        __restrict__ p_tile,     // [n_kv, T*gqa, kc] F16
    const float *   __restrict__ bias_table, // [T, n_head, n_extent+1]
    const int32_t * __restrict__ pos,        // [T]
    float *         __restrict__ m_buf,
    float *         __restrict__ l_buf,
    float *         __restrict__ o_buf,      // [n_kv, T*gqa, 128] FP32 accumulator
    int T, int n_head, int n_kv, int k0, int kc, int swa_window, int n_extent, bool is_first_tile) {

    const int gqa  = n_head / n_kv;
    const int r    = blockIdx.x;
    const int kvh  = r / (T * gqa);
    const int rem  = r - kvh * (T * gqa);
    const int t    = rem / gqa;
    const int gh   = rem - t * gqa;
    const int h    = kvh * gqa + gh;
    const int maxp = pos[t];

    const int tid  = threadIdx.x; // 0..127
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const __half * s_row = s_tile + static_cast<size_t>(r) * kc;
    __half *       p_row = p_tile + static_cast<size_t>(r) * kc;
    const float *  b_row = bias_table + (static_cast<size_t>(t) * n_head + h) * (n_extent + 1);
    float *        o_row = o_buf + static_cast<size_t>(r) * 128;

    const int k_lo    = (swa_window > 0 && maxp >= swa_window) ? (maxp - swa_window + 1) : 0;
    const int c_start = (k_lo > k0) ? (k_lo - k0) : 0;
    const int c_end   = (k0 + kc - 1 <= maxp) ? kc : (maxp - k0 + 1);

    if (c_start >= c_end) {
        if (is_first_tile) o_row[tid] = 0.0f;
        for (int c = tid; c < kc; c += blockDim.x) p_row[c] = __float2half_rn(0.0f);
        return;
    }

    float local_m = -FLT_MAX;
    for (int c = c_start + tid; c < c_end; c += blockDim.x) {
        const int dist = maxp - (k0 + c);
        const int bi   = (dist < n_extent) ? dist : n_extent;
        local_m = fmaxf(local_m, __half2float(s_row[c]) + __ldg(&b_row[bi]));
    }
    local_m = warp_max(local_m);
    __shared__ float s_red[4];
    if (lane == 0) s_red[warp] = local_m;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 4) ? s_red[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    const float m_tile = s_red[0];

    const float m_old = m_buf[r];
    const float m_new = fmaxf(m_old, m_tile);
    const float alpha = __expf(m_old - m_new);

    float local_l = 0.0f;
    for (int c = tid; c < kc; c += blockDim.x) {
        if (c >= c_start && c < c_end) {
            const int dist = maxp - (k0 + c);
            const int bi   = (dist < n_extent) ? dist : n_extent;
            const float e = __expf(__half2float(s_row[c]) + __ldg(&b_row[bi]) - m_new);
            p_row[c] = __float2half_rn(e);
            local_l += e;
        } else {
            p_row[c] = __float2half_rn(0.0f);
        }
    }
    local_l = warp_sum(local_l);
    if (lane == 0) s_red[warp] = local_l;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 4) ? s_red[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) {
            m_buf[r] = m_new;
            l_buf[r] = l_buf[r] * alpha + v;
        }
    }

    if (is_first_tile) {
        o_row[tid] = 0.0f;
    } else if (alpha != 1.0f) {
        o_row[tid] *= alpha;
    }
}

__global__ void inkling2_f16_add_to_f32_kernel(const __half * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < n) {
        const float2 a = __half22float2(reinterpret_cast<const __half2 *>(src)[i >> 1]);
        float2 d = reinterpret_cast<float2 *>(dst)[i >> 1];
        d.x += a.x; d.y += a.y;
        reinterpret_cast<float2 *>(dst)[i >> 1] = d;
    } else if (i < n) {
        dst[i] += __half2float(src[i]);
    }
}

} // namespace

void launch_inkling2_depth_mix(
    const float * const * d_blocks_dev, int n_blocks,
    const float * d_cur, const float * d_query, float * d_out,
    int n_embd, int n_q, int T, float eps, cudaStream_t stream) {
    const int n4 = n_embd >> 2;
    if ((n_embd & 127) == 0 && n4 >= 128 && n4 <= 1024 && n_blocks + 1 <= MAX_DEPTH_STATES && n_q >= 1 && n_q <= 4) {
        switch (n_q) {
            case 1: inkling2_depth_mix_f4_fast_kernel<1><<<T, n4, 0, stream>>>(d_blocks_dev, n_blocks, d_cur, d_query, d_out, n_embd, T, eps); return;
            case 2: inkling2_depth_mix_f4_fast_kernel<2><<<T, n4, 0, stream>>>(d_blocks_dev, n_blocks, d_cur, d_query, d_out, n_embd, T, eps); return;
            case 3: inkling2_depth_mix_f4_fast_kernel<3><<<T, n4, 0, stream>>>(d_blocks_dev, n_blocks, d_cur, d_query, d_out, n_embd, T, eps); return;
            case 4: inkling2_depth_mix_f4_fast_kernel<4><<<T, n4, 0, stream>>>(d_blocks_dev, n_blocks, d_cur, d_query, d_out, n_embd, T, eps); return;
        }
    }
    inkling2_depth_mix_kernel<<<T, 256, 0, stream>>>(
        d_blocks_dev, n_blocks, d_cur, d_query, d_out, n_embd, n_q, T, eps);
}

void launch_inkling2_rmsnorm_concat2(
    const float * d_a, const float * d_b, float * d_out,
    int n_embd, int T, float eps, cudaStream_t stream) {
    inkling2_rmsnorm_concat2_kernel<<<T, 256, 0, stream>>>(d_a, d_b, d_out, n_embd, eps);
}

void launch_inkling2_loop_combine(
    const float * d_cur, const float * d_msg,
    const float * d_gain_pre, const float * d_forget_pre,
    float * d_out, int64_t n, cudaStream_t stream) {
    inkling2_loop_combine_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        d_cur, d_msg, d_gain_pre, d_forget_pre, d_out, n);
}

void launch_inkling2_shortconv(
    const float * d_x, const float * d_w, const float * d_dyn,
    float * d_state, float * d_out,
    int T, int C, int n_head_kv, cudaStream_t stream,
    const float * d_res) {
    const int head_dim = (n_head_kv > 0) ? (C / n_head_kv) : C;
    const int blk = (T == 1) ? 64 : 256;
    inkling2_shortconv_kernel<<<(C + blk - 1) / blk, blk, 0, stream>>>(
        d_x, d_w, d_dyn, d_state, d_out, d_res, T, C, head_dim, n_head_kv);
}

void launch_inkling2_rmsnorm_fast(
    const float * d_x, const float * d_w, float * d_out,
    int n_embd, int T, float eps, cudaStream_t stream) {
    const int n4 = n_embd >> 2;
    if ((n_embd & 127) == 0 && n4 >= 32 && n4 <= 1024) {
        inkling2_rmsnorm_f4_1pass_kernel<<<T, n4, 0, stream>>>(d_x, d_w, d_out, n_embd, eps);
    } else {
        launch_rmsnorm(d_x, d_w, d_out, n_embd, T, eps, stream);
    }
}

void launch_inkling2_rmsnorm3_fast(
    const float * d_x0, const float * d_x1, const float * d_x2,
    const float * d_w0, const float * d_w1, const float * d_w2,
    float * d_o0, float * d_o1, float * d_o2,
    int n_embd, int T, float eps, cudaStream_t stream) {
    const int n4 = n_embd >> 2;
    if ((n_embd & 127) == 0 && n4 >= 32 && n4 <= 1024) {
        Ink2Ptr3 xs = {{d_x0, d_x1, d_x2}};
        Ink2Ptr3 ws = {{d_w0, d_w1, d_w2}};
        Ink2Out3 os = {{d_o0, d_o1, d_o2}};
        inkling2_rmsnorm3_f4_1pass_kernel<<<dim3(T, 3), n4, 0, stream>>>(xs, ws, os, n_embd, eps);
    } else {
        launch_rmsnorm(d_x0, d_w0, d_o0, n_embd, T, eps, stream);
        launch_rmsnorm(d_x1, d_w1, d_o1, n_embd, T, eps, stream);
        launch_rmsnorm(d_x2, d_w2, d_o2, n_embd, T, eps, stream);
    }
}

void launch_inkling2_loop_ctrl_up_combine(
    const DeviceTensor & W_gain, const DeviceTensor & W_forget,
    const float * d_hid, const float * d_cur, const float * d_msg, float * d_out,
    int T, cudaStream_t stream) {
    const int gate_rank = static_cast<int>(W_gain.ne[0]);
    const int n_embd    = static_cast<int>(W_gain.ne[1]);
    constexpr int wpb = 8;
    const size_t smem = static_cast<size_t>(gate_rank) * sizeof(float);
    inkling2_loop_ctrl_up_combine_kernel<<<dim3((n_embd + wpb * 4 - 1) / (wpb * 4), T), dim3(32, wpb), smem, stream>>>(
        static_cast<const float *>(W_gain.ptr), static_cast<const float *>(W_forget.ptr),
        d_hid, d_cur, d_msg, d_out, n_embd, gate_rank);
}

void launch_inkling2_loop_ctrl_qkv_decode(
    const float * d_in0, const float * d_in1, const float * d_in2,
    const float * d_msg0, const float * d_msg1, const float * d_msg2,
    const DeviceTensor * ctrl_down, const DeviceTensor * ctrl_gain, const DeviceTensor * ctrl_forget,
    float * d_ctrl_in3, float * d_ctrl_hid3,
    float * d_out0, float * d_out1, float * d_out2,
    int n_embd, int gate_rank, float eps, cudaStream_t stream) {
    Ink2Ptr3 ins  = {{d_in0, d_in1, d_in2}};
    Ink2Ptr3 msgs = {{d_msg0, d_msg1, d_msg2}};
    Ink2Ptr3 wd   = {{static_cast<const float *>(ctrl_down[0].ptr),
                      static_cast<const float *>(ctrl_down[1].ptr),
                      static_cast<const float *>(ctrl_down[2].ptr)}};
    Ink2Ptr3 wg   = {{static_cast<const float *>(ctrl_gain[0].ptr),
                      static_cast<const float *>(ctrl_gain[1].ptr),
                      static_cast<const float *>(ctrl_gain[2].ptr)}};
    Ink2Ptr3 wf   = {{static_cast<const float *>(ctrl_forget[0].ptr),
                      static_cast<const float *>(ctrl_forget[1].ptr),
                      static_cast<const float *>(ctrl_forget[2].ptr)}};
    const int n4 = n_embd >> 2;
    inkling2_rmsnorm_concat2_3way_kernel<<<dim3(3, 2), n4, 0, stream>>>(ins, msgs, d_ctrl_in3, n_embd, eps);
    inkling2_ctrl_down_3way_kernel<<<dim3(gate_rank, 3), 256, 0, stream>>>(wd, d_ctrl_in3, d_ctrl_hid3, 2 * n_embd, gate_rank);
    constexpr int wpb = 8;
    inkling2_loop_ctrl_up_combine_3way_kernel<<<dim3((n_embd + wpb * 4 - 1) / (wpb * 4), 3), dim3(32, wpb), static_cast<size_t>(gate_rank) * sizeof(float), stream>>>(
        wg.p[0], wg.p[1], wg.p[2], wf.p[0], wf.p[1], wf.p[2], d_ctrl_hid3,
        d_in0, d_in1, d_in2, d_msg0, d_msg1, d_msg2, d_out0, d_out1, d_out2, n_embd, gate_rank);
}

void launch_inkling2_qkv_fused_q8_0(
    int n_projs,
    const DeviceTensor * const * Ws,
    float * const * Ys,
    const void * d_xq8,
    int T, cudaStream_t stream) {
    Ink2FusedQ8Projs P{};
    P.n_projs = n_projs;
    int tot = 0;
    for (int i = 0; i < n_projs; ++i) {
        const int ni = static_cast<int>(Ws[i]->ne[1]);
        tot += ni;
        P.row_end[i] = tot;
        P.qs[i]      = static_cast<const int4 *>(Ws[i]->ptr);
        P.sc[i]      = static_cast<const uint16_t *>(Ws[i]->scales_ptr);
        P.out[i]     = Ys[i];
        P.N[i]       = ni;
    }
    const int K_blocks = static_cast<int>(Ws[0]->ne[0]) / 32;
    constexpr int wpb = 8;
    const size_t smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
    inkling2_qkv_fused_q8_0_kernel<<<dim3((tot + wpb - 1) / wpb, T), dim3(32, wpb), smem, stream>>>(
        P, static_cast<const BlockQ8_1_Act *>(d_xq8), K_blocks, tot);
}

void launch_inkling2_sigmoid_gate_quant_q8_1(
    const float * d_x, const float * d_gate, void * d_xq8,
    int K, int T, cudaStream_t stream) {
    const int total_b = T * (K / 32);
    if (total_b <= 0) return;
    constexpr int wpb = 8;
    inkling2_sigmoid_gate_quant_q8_1_kernel<<<(total_b + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
        d_x, d_gate, static_cast<BlockQ8_1_Act *>(d_xq8), total_b);
}

void launch_inkling2_rmsnorm_quant_f16_contig(
    int n_inputs,
    const float * d_x0, const float * d_x1, const float * d_x2,
    const float * d_w0, const float * d_w1, const float * d_w2,
    void * d_out_h2_d, int n_embd, int T, float eps, cudaStream_t stream) {
    const int K_blocks = n_embd >> 5;
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    auto * base = static_cast<uint8_t *>(d_out_h2_d);
    auto * out_h2 = reinterpret_cast<__half2 *>(base);
    auto * out_d  = reinterpret_cast<float *>(base + align256(static_cast<size_t>(n_inputs) * T * K_blocks * 16 * sizeof(__half2)));
    Ink2Ptr3 xs = {{d_x0, d_x1, d_x2}};
    Ink2Ptr3 ws = {{d_w0, d_w1, d_w2}};
    inkling2_rmsnorm_quant_f16_contig_kernel<<<dim3(T, n_inputs), n_embd >> 2, 0, stream>>>(
        xs, ws, out_h2, out_d, n_embd, n_inputs, T, eps);
}

void launch_inkling2_qkv_fused_q8_0_h2(
    int n_projs, int n_inputs,
    const DeviceTensor * const * Ws,
    const int * in_idx,
    float * const * Ys,
    const void * d_in_h2_d, int T, cudaStream_t stream) {
    Ink2FusedQ8Projs8 P{};
    P.n_projs = n_projs;
    const int K = static_cast<int>(Ws[0]->ne[0]);
    const int K_blocks = K >> 5;
    int total_rows = 0;
    for (int i = 0; i < n_projs; ++i) {
        const int Ni = static_cast<int>(Ws[i]->ne[1]);
        P.qs[i]     = static_cast<const int4 *>(Ws[i]->ptr);
        P.sc[i]     = static_cast<const uint16_t *>(Ws[i]->scales_ptr);
        P.out[i]    = Ys[i];
        P.N[i]      = Ni;
        P.in_idx[i] = in_idx[i];
        total_rows += Ni;
        P.row_end[i] = total_rows;
    }
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    const auto * base = static_cast<const uint8_t *>(d_in_h2_d);
    const auto * x_h2 = reinterpret_cast<const __half2 *>(base);
    const auto * x_d  = reinterpret_cast<const float *>(base + align256(static_cast<size_t>(n_inputs) * T * K_blocks * 16 * sizeof(__half2)));
    const size_t smem = static_cast<size_t>(n_inputs) * K_blocks * (16 * sizeof(__half2) + sizeof(float));
    constexpr int rows_cta = 256 / 16;
    inkling2_qkv_fused_q8_0_h2_kernel<<<dim3((total_rows + rows_cta - 1) / rows_cta, T), 256, smem, stream>>>(
        P, x_h2, x_d, K_blocks, n_inputs, T, total_rows);
}

void launch_inkling2_sigmoid_gate_quant_f16_contig(
    const float * d_x, const float * d_gate, void * d_out_h2_d,
    int K, int T, cudaStream_t stream) {
    const int K_blocks = K >> 5;
    const int total_b = T * K_blocks;
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    auto * base = static_cast<uint8_t *>(d_out_h2_d);
    auto * out_h2 = reinterpret_cast<__half2 *>(base);
    auto * out_d  = reinterpret_cast<float *>(base + align256(static_cast<size_t>(total_b) * 16 * sizeof(__half2)));
    constexpr int wpb = 8;
    inkling2_sigmoid_gate_quant_f16_contig_kernel<<<(total_b + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
        d_x, d_gate, out_h2, out_d, total_b);
}

void launch_inkling2_quant_f16_contig(
    const float * d_x, void * d_out_h2_d, int K, int T, cudaStream_t stream) {
    const int K_blocks = K >> 5;
    const int total_b = T * K_blocks;
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    auto * base = static_cast<uint8_t *>(d_out_h2_d);
    auto * out_h2 = reinterpret_cast<__half2 *>(base);
    auto * out_d  = reinterpret_cast<float *>(base + align256(static_cast<size_t>(total_b) * 16 * sizeof(__half2)));
    constexpr int wpb = 8;
    inkling2_quantize_f16_contig_kernel<<<(total_b + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
        d_x, out_h2, out_d, total_b);
}

void launch_inkling2_gemv_q8_0_h2(
    const DeviceTensor & W, const void * d_in_h2_d, float * d_y,
    int T, cudaStream_t stream) {
    const int K = static_cast<int>(W.ne[0]);
    const int N = static_cast<int>(W.ne[1]);
    const int K_blocks = K >> 5;
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    const auto * base = static_cast<const uint8_t *>(d_in_h2_d);
    const auto * x_h2 = reinterpret_cast<const __half2 *>(base);
    const auto * x_d  = reinterpret_cast<const float *>(base + align256(static_cast<size_t>(T) * K_blocks * 16 * sizeof(__half2)));
    const size_t smem = static_cast<size_t>(K_blocks) * (16 * sizeof(__half2) + sizeof(float));
    const auto * wq = static_cast<const int4 *>(W.ptr);
    const auto * ws = static_cast<const uint16_t *>(W.scales_ptr);
    if (N >= 32768) {
        constexpr int LPR = 32;
        constexpr int rows_cta = 256 / LPR;
        inkling2_gemv_q8_0_h2_kernel<LPR><<<dim3((N + rows_cta - 1) / rows_cta, T), 256, smem, stream>>>(
            wq, ws, x_h2, x_d, d_y, K_blocks, N, T);
    } else {
        constexpr int LPR = 16;
        constexpr int rows_cta = 256 / LPR;
        inkling2_gemv_q8_0_h2_kernel<LPR><<<dim3((N + rows_cta - 1) / rows_cta, T), 256, smem, stream>>>(
            wq, ws, x_h2, x_d, d_y, K_blocks, N, T);
    }
}

void launch_inkling2_kv_conv_norm_write_q8_0_t1(
    const float * d_k_raw, const float * d_k_w, const float * d_k_dyn, float * d_k_state,
    const float * d_v_raw, const float * d_v_w, const float * d_v_dyn, float * d_v_state,
    void * d_k_cache, void * d_v_cache, const int32_t * d_pos,
    int n_head_kv, int head_dim, int cache_cap, float qk_clip, float rms_eps,
    cudaStream_t stream) {
    (void) head_dim;
    inkling2_kv_conv_norm_write_q8_0_t1_kernel<<<n_head_kv * 2, 128, 0, stream>>>(
        d_k_raw, d_k_w, d_k_dyn, d_k_state,
        d_v_raw, d_v_w, d_v_dyn, d_v_state,
        d_k_cache, d_v_cache, d_pos,
        n_head_kv, cache_cap, qk_clip, rms_eps);
}

void launch_inkling2_qr_l2_clip_fused_t1(
    float * d_q, const float * d_q_scale,
    float * d_r, const float * d_r_scale,
    const int32_t * d_pos,
    float qk_clip, float rel_clip, float log_floor, float log_alpha,
    int n_head, int head_dim, int n_rel, cudaStream_t stream) {
    inkling2_qr_l2_clip_fused_t1_kernel<<<n_head * 2, 32, 0, stream>>>(
        d_q, d_q_scale, d_r, d_r_scale, d_pos,
        qk_clip, rel_clip, log_floor, log_alpha, n_head, head_dim, n_rel);
}

void launch_inkling2_linear_sm60_or_default(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    if (cublas != nullptr && T >= 16 && W.dtype == DType::Q8_0 && W.is_soa &&
        device_sm_version(W.device) == 60) {
        const int K = static_cast<int>(W.ne[0]);
        const int N = static_cast<int>(W.ne[1]);
        auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
        const size_t xf16_b = align256(static_cast<size_t>(T) * K * sizeof(__half));
        const size_t yf16_b = align256(static_cast<size_t>(T) * N * sizeof(__half));
        const size_t wf16_b = align256(static_cast<size_t>(N) * K * sizeof(__half));
        if (xf16_b + yf16_b + wf16_b <= scratch_bytes) {
            auto * base   = static_cast<uint8_t *>(d_scratch);
            auto * d_xf16 = reinterpret_cast<__half *>(base);
            auto * d_yf16 = reinterpret_cast<__half *>(base + xf16_b);
            auto * d_wf16 = reinterpret_cast<__half *>(base + xf16_b + yf16_b);

            const int64_t nx = static_cast<int64_t>(T) * K;
            inkling2_f32_to_f16_vec_kernel<<<((nx + 1) / 2 + 255) / 256, 256, 0, stream>>>(d_x, d_xf16, nx);

            const int64_t n_blocks = static_cast<int64_t>(N) * (K / 32);
            const int64_t n_q4 = n_blocks * 4;
            inkling2_dequant_q8_0_soa_f16_kernel<<<(n_q4 + 255) / 256, 256, 0, stream>>>(
                static_cast<const int8_t *>(W.ptr), static_cast<const uint16_t *>(W.scales_ptr), d_wf16, n_blocks);

            const __half h_one = __float2half_rn(1.0f), h_zero = __float2half_rn(0.0f);
            cublasSetStream(cublas, stream);
            cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        N, T, K,
                        &h_one, d_wf16, K,
                        d_xf16, K,
                        &h_zero, d_yf16, N);

            const int64_t ny = static_cast<int64_t>(T) * N;
            inkling2_f16_to_f32_vec_kernel<<<((ny + 1) / 2 + 255) / 256, 256, 0, stream>>>(d_yf16, d_y, ny);
            return;
        }
    }
    launch_linear(cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream);
}

void launch_inkling2_router_gemv_f32(
    const DeviceTensor & W, const float * d_x, float * d_y, int T, cudaStream_t stream) {
    const int K = static_cast<int>(W.ne[0]);
    const int N = static_cast<int>(W.ne[1]);
    constexpr int wpb = 2; // 64 CTAs for N=128 -> saturates all 56 SMs on P100!
    inkling2_router_gemv_f32_kernel<<<dim3((N + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
        static_cast<const float *>(W.ptr), d_x, d_y, K, N);
}

void launch_inkling2_moe_route_sigmoid_fast(
    const float * d_logits, const float * d_bias,
    int32_t * d_top_ids, float * d_top_w,
    int T, int n_expert, int k_used, bool renorm, float scale, cudaStream_t stream) {
    if (n_expert <= 128 && k_used <= 16) {
        inkling2_moe_route_sigmoid_w32_kernel<<<T, 32, 0, stream>>>(
            d_logits, d_bias, d_top_ids, d_top_w, n_expert, k_used, renorm, scale);
    } else {
        launch_moe_route_sigmoid(d_logits, d_bias, d_top_ids, d_top_w, T, n_expert, k_used, renorm, scale, stream);
    }
}

void launch_inkling2_l2_clip(
    float * d_x, const float * d_scale, const int32_t * d_pos,
    float threshold, float log_floor, float log_alpha,
    int T, int n_head, int head_dim, cudaStream_t stream) {
    constexpr int wpb = 4;
    inkling2_l2_clip_kernel<<<dim3((n_head + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
        d_x, d_scale, d_pos, threshold, log_floor, log_alpha, n_head, head_dim);
}

void launch_inkling2_head_rmsnorm(
    float * d_x, int T, int n_head, int head_dim, float eps, cudaStream_t stream) {
    constexpr int wpb = 4;
    inkling2_head_rmsnorm_kernel<<<dim3((n_head + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
        d_x, n_head, head_dim, eps);
}

void launch_inkling2_rel_bias_table(
    const float * d_r, const float * d_rel_bias, const float * d_sinks,
    float * d_bias_table, int T, int n_head, int n_rel, int n_extent, cudaStream_t stream) {
    if (n_rel == 16) {
        if (T <= 4) {
            constexpr int thr = 32;
            const size_t smem = static_cast<size_t>(n_head) * 17 * sizeof(float);
            inkling2_rel_bias_table_r16_dec_kernel<<<dim3((n_extent + 1 + thr - 1) / thr, T), thr, smem, stream>>>(
                d_r, d_rel_bias, d_sinks, d_bias_table, n_head, n_extent);
        } else {
            const int M = T * n_head;
            inkling2_rel_bias_table_r16_tiled_kernel<<<dim3((n_extent + 1 + 31) / 32, (M + 31) / 32), dim3(32, 8), 0, stream>>>(
                d_r, d_rel_bias, d_sinks, d_bias_table, M, n_extent);
        }
        return;
    }
    dim3 grid((n_extent + 1 + 255) / 256, n_head, T);
    inkling2_rel_bias_table_kernel<<<grid, 256, 0, stream>>>(
        d_r, d_rel_bias, d_sinks, d_bias_table, n_head, n_rel, n_extent);
}

void launch_inkling2_kv_write(
    const float * d_k, const float * d_v,
    void * d_k_cache, void * d_v_cache, bool kv_is_q8_0,
    const int32_t * d_pos, int T, int n_kv, int head_dim, int cache_cap, cudaStream_t stream) {
    if (kv_is_q8_0) {
        const int total_b = n_kv * (head_dim >> 5);
        constexpr int wpb = 8;
        inkling2_kv_write_q8_0_kernel<<<dim3((total_b + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
            d_k, d_v, d_k_cache, d_v_cache,
            d_pos, T, n_kv, head_dim, cache_cap);
    } else {
        const int kv_dim = n_kv * head_dim;
        inkling2_kv_write_f16_kernel<<<dim3((kv_dim + 255) / 256, T), 256, 0, stream>>>(
            d_k, d_v, static_cast<__half *>(d_k_cache), static_cast<__half *>(d_v_cache),
            d_pos, T, kv_dim, cache_cap);
    }
}

void launch_inkling2_attn(
    const float * d_q, const void * d_k_cache, const void * d_v_cache, bool kv_is_q8_0,
    const float * d_bias_table, const int32_t * d_pos, float * d_out,
    void * d_scratch, size_t scratch_bytes,
    int T, int n_head, int n_kv, int head_dim,
    int swa_window, int n_extent, int cache_cap, float scale,
    cublasHandle_t cublas, int kv_len_max, cudaStream_t stream) {

    // Fast Tiled FlashAttention for BOTH Global and SWA prefill (T >= 8):
    // FP16 Tensor Core on sm>=70, FP32 SGEMM on <sm_70!
    if (T >= 8 && head_dim == 128 && cublas != nullptr && kv_len_max > 0) {
        const int  sm_ver = device_sm_version(-1);
        const bool use_tc = (sm_ver >= 70);
        const bool use_sm60_f16 = (sm_ver == 60);
        const int gqa = n_head / n_kv;
        const int M_per_kv = T * gqa;
        const int M_tot = n_kv * M_per_kv;
        auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
        const int k_start = (swa_window > 0 && kv_len_max > T + swa_window)
                                ? (kv_len_max - T - swa_window) : 0;

        if (use_tc) {
            constexpr int K_TILE = 512;
            const size_t q16_b = align256(static_cast<size_t>(M_tot) * 128 * sizeof(__half));
            const size_t k16_b = align256(static_cast<size_t>(n_kv) * K_TILE * 128 * sizeof(__half));
            const size_t v16_b = k16_b;
            const size_t s32_b = align256(static_cast<size_t>(M_tot) * K_TILE * sizeof(float));
            const size_t p16_b = align256(static_cast<size_t>(M_tot) * K_TILE * sizeof(__half));
            const size_t o32_b = align256(static_cast<size_t>(M_tot) * 128 * sizeof(float));
            const size_t ml_b  = align256(static_cast<size_t>(M_tot) * sizeof(float));
            const size_t need  = q16_b + k16_b + v16_b + s32_b + p16_b + o32_b + 2 * ml_b;

            if (need <= scratch_bytes) {
                auto * base = static_cast<uint8_t *>(d_scratch);
                size_t off = 0;
                auto * d_q16 = reinterpret_cast<__half *>(base + off); off += q16_b;
                auto * d_k16 = reinterpret_cast<__half *>(base + off); off += k16_b;
                auto * d_v16 = reinterpret_cast<__half *>(base + off); off += v16_b;
                auto * d_s32 = reinterpret_cast<float  *>(base + off); off += s32_b;
                auto * d_p16 = reinterpret_cast<__half *>(base + off); off += p16_b;
                auto * d_ob  = reinterpret_cast<float  *>(base + off); off += o32_b;
                auto * d_m   = reinterpret_cast<float  *>(base + off); off += ml_b;
                auto * d_l   = reinterpret_cast<float  *>(base + off);

                const int tot_q = T * n_head * 128;
                inkling2_q_to_kvh_kernel<__half><<<(tot_q + 255) / 256, 256, 0, stream>>>(d_q, d_q16, T, n_head, n_kv, scale);
                inkling2_init_flash_sink_kernel<<<(M_tot + 255) / 256, 256, 0, stream>>>(d_m, d_l, M_tot);

                const float one = 1.0f, zero = 0.0f;
                cublasSetStream(cublas, stream);

                for (int k0 = k_start; k0 < kv_len_max; k0 += K_TILE) {
                    const int kc = (k0 + K_TILE < kv_len_max) ? K_TILE : (kv_len_max - k0);
                    if (kv_is_q8_0) {
                        inkling2_dequant_kv_tile_kernel<true, __half><<<dim3(n_kv, kc), dim3(32, 4), 0, stream>>>(
                            d_k_cache, d_v_cache, d_k16, d_v16, k0, kc, n_kv, cache_cap);
                    } else {
                        inkling2_dequant_kv_tile_kernel<false, __half><<<dim3(n_kv, kc), 128, 0, stream>>>(
                            d_k_cache, d_v_cache, d_k16, d_v16, k0, kc, n_kv, cache_cap);
                    }

                    cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        kc, M_per_kv, 128,
                        &one,
                        d_k16, CUDA_R_16F, 128, static_cast<long long>(kc) * 128,
                        d_q16, CUDA_R_16F, 128, static_cast<long long>(M_per_kv) * 128,
                        &zero,
                        d_s32, CUDA_R_32F, kc,  static_cast<long long>(M_per_kv) * kc,
                        n_kv, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                    inkling2_flash_softmax_relbias_rescale_kernel<__half><<<M_tot, 128, 0, stream>>>(
                        d_s32, d_p16, d_bias_table, d_pos, d_m, d_l, d_ob, T, n_head, n_kv,
                        k0, kc, swa_window, n_extent, k0 == k_start);

                    cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        128, M_per_kv, kc,
                        &one,
                        d_v16, CUDA_R_16F, 128, static_cast<long long>(kc) * 128,
                        d_p16, CUDA_R_16F, kc,  static_cast<long long>(M_per_kv) * kc,
                        &one,
                        d_ob,  CUDA_R_32F, 128, static_cast<long long>(M_per_kv) * 128,
                        n_kv, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                }

                inkling2_o_from_kvh_finalize_kernel<<<M_tot, 128, 0, stream>>>(d_ob, d_l, d_out, T, n_head, n_kv);
                return;
            }
        } else if (use_sm60_f16) {
            constexpr int K_TILE = 512;
            const size_t q16_b = align256(static_cast<size_t>(M_tot) * 128 * sizeof(__half));
            const size_t k16_b = align256(static_cast<size_t>(n_kv) * K_TILE * 128 * sizeof(__half));
            const size_t v16_b = k16_b;
            const size_t s16_b = align256(static_cast<size_t>(M_tot) * K_TILE * sizeof(__half));
            const size_t p16_b = s16_b;
            const size_t o16_b = q16_b;
            const size_t o32_b = align256(static_cast<size_t>(M_tot) * 128 * sizeof(float));
            const size_t ml_b  = align256(static_cast<size_t>(M_tot) * sizeof(float));
            const size_t need  = q16_b + k16_b + v16_b + s16_b + p16_b + o16_b + o32_b + 2 * ml_b;

            if (need <= scratch_bytes) {
                auto * base = static_cast<uint8_t *>(d_scratch);
                size_t off = 0;
                auto * d_q16 = reinterpret_cast<__half *>(base + off); off += q16_b;
                auto * d_k16 = reinterpret_cast<__half *>(base + off); off += k16_b;
                auto * d_v16 = reinterpret_cast<__half *>(base + off); off += v16_b;
                auto * d_s16 = reinterpret_cast<__half *>(base + off); off += s16_b;
                auto * d_p16 = reinterpret_cast<__half *>(base + off); off += p16_b;
                auto * d_o16 = reinterpret_cast<__half *>(base + off); off += o16_b;
                auto * d_ob  = reinterpret_cast<float  *>(base + off); off += o32_b;
                auto * d_m   = reinterpret_cast<float  *>(base + off); off += ml_b;
                auto * d_l   = reinterpret_cast<float  *>(base + off);

                const int tot_q = T * n_head * 128;
                inkling2_q_to_kvh_kernel<__half><<<(tot_q + 255) / 256, 256, 0, stream>>>(d_q, d_q16, T, n_head, n_kv, scale);
                inkling2_init_flash_sink_kernel<<<(M_tot + 255) / 256, 256, 0, stream>>>(d_m, d_l, M_tot);

                const __half h_one = __float2half_rn(1.0f), h_zero = __float2half_rn(0.0f);
                cublasSetStream(cublas, stream);

                for (int k0 = k_start; k0 < kv_len_max; k0 += K_TILE) {
                    const int kc = (k0 + K_TILE < kv_len_max) ? K_TILE : (kv_len_max - k0);
                    if (kv_is_q8_0) {
                        inkling2_dequant_kv_tile_kernel<true, __half><<<dim3(n_kv, kc), dim3(32, 4), 0, stream>>>(
                            d_k_cache, d_v_cache, d_k16, d_v16, k0, kc, n_kv, cache_cap);
                    } else {
                        inkling2_dequant_kv_tile_kernel<false, __half><<<dim3(n_kv, kc), 128, 0, stream>>>(
                            d_k_cache, d_v_cache, d_k16, d_v16, k0, kc, n_kv, cache_cap);
                    }

                    cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        kc, M_per_kv, 128,
                        &h_one,
                        d_k16, CUDA_R_16F, 128, static_cast<long long>(kc) * 128,
                        d_q16, CUDA_R_16F, 128, static_cast<long long>(M_per_kv) * 128,
                        &h_zero,
                        d_s16, CUDA_R_16F, kc,  static_cast<long long>(M_per_kv) * kc,
                        n_kv, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);

                    inkling2_flash_softmax_relbias_f16s_kernel<<<M_tot, 128, 0, stream>>>(
                        d_s16, d_p16, d_bias_table, d_pos, d_m, d_l, d_ob, T, n_head, n_kv,
                        k0, kc, swa_window, n_extent, k0 == k_start);

                    cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        128, M_per_kv, kc,
                        &h_one,
                        d_v16, CUDA_R_16F, 128, static_cast<long long>(kc) * 128,
                        d_p16, CUDA_R_16F, kc,  static_cast<long long>(M_per_kv) * kc,
                        &h_zero,
                        d_o16, CUDA_R_16F, 128, static_cast<long long>(M_per_kv) * 128,
                        n_kv, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);

                    const int64_t no = static_cast<int64_t>(M_tot) * 128;
                    inkling2_f16_add_to_f32_kernel<<<((no + 1) / 2 + 255) / 256, 256, 0, stream>>>(d_o16, d_ob, no);
                }

                inkling2_o_from_kvh_finalize_kernel<<<M_tot, 128, 0, stream>>>(d_ob, d_l, d_out, T, n_head, n_kv);
                return;
            }
        } else {
            constexpr int K_TILE = 256;
            const size_t q32_b = align256(static_cast<size_t>(M_tot) * 128 * sizeof(float));
            const size_t k32_b = align256(static_cast<size_t>(n_kv) * K_TILE * 128 * sizeof(float));
            const size_t v32_b = k32_b;
            const size_t s32_b = align256(static_cast<size_t>(M_tot) * K_TILE * sizeof(float));
            const size_t p32_b = s32_b;
            const size_t o32_b = q32_b;
            const size_t ml_b  = align256(static_cast<size_t>(M_tot) * sizeof(float));
            const size_t need  = q32_b + k32_b + v32_b + s32_b + p32_b + o32_b + 2 * ml_b;

            if (need <= scratch_bytes) {
                auto * base = static_cast<uint8_t *>(d_scratch);
                size_t off = 0;
                auto * d_q32 = reinterpret_cast<float *>(base + off); off += q32_b;
                auto * d_k32 = reinterpret_cast<float *>(base + off); off += k32_b;
                auto * d_v32 = reinterpret_cast<float *>(base + off); off += v32_b;
                auto * d_s32 = reinterpret_cast<float *>(base + off); off += s32_b;
                auto * d_p32 = reinterpret_cast<float *>(base + off); off += p32_b;
                auto * d_ob  = reinterpret_cast<float *>(base + off); off += o32_b;
                auto * d_m   = reinterpret_cast<float *>(base + off); off += ml_b;
                auto * d_l   = reinterpret_cast<float *>(base + off);

                const int tot_q = T * n_head * 128;
                inkling2_q_to_kvh_kernel<float><<<(tot_q + 255) / 256, 256, 0, stream>>>(d_q, d_q32, T, n_head, n_kv, scale);
                inkling2_init_flash_sink_kernel<<<(M_tot + 255) / 256, 256, 0, stream>>>(d_m, d_l, M_tot);

                const float one = 1.0f, zero = 0.0f;
                cublasSetStream(cublas, stream);

                for (int k0 = k_start; k0 < kv_len_max; k0 += K_TILE) {
                    const int kc = (k0 + K_TILE < kv_len_max) ? K_TILE : (kv_len_max - k0);
                    if (kv_is_q8_0) {
                        inkling2_dequant_kv_tile_kernel<true, float><<<dim3(n_kv, kc), dim3(32, 4), 0, stream>>>(
                            d_k_cache, d_v_cache, d_k32, d_v32, k0, kc, n_kv, cache_cap);
                    } else {
                        inkling2_dequant_kv_tile_kernel<false, float><<<dim3(n_kv, kc), 128, 0, stream>>>(
                            d_k_cache, d_v_cache, d_k32, d_v32, k0, kc, n_kv, cache_cap);
                    }

                    cublasSgemmStridedBatched(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        kc, M_per_kv, 128,
                        &one,
                        d_k32, 128, static_cast<long long>(kc) * 128,
                        d_q32, 128, static_cast<long long>(M_per_kv) * 128,
                        &zero,
                        d_s32, kc,  static_cast<long long>(M_per_kv) * kc,
                        n_kv);

                    inkling2_flash_softmax_relbias_rescale_kernel<float><<<M_tot, 128, 0, stream>>>(
                        d_s32, d_p32, d_bias_table, d_pos, d_m, d_l, d_ob, T, n_head, n_kv,
                        k0, kc, swa_window, n_extent, k0 == k_start);

                    cublasSgemmStridedBatched(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        128, M_per_kv, kc,
                        &one,
                        d_v32, 128, static_cast<long long>(kc) * 128,
                        d_p32, kc,  static_cast<long long>(M_per_kv) * kc,
                        &one,
                        d_ob,  128, static_cast<long long>(M_per_kv) * 128,
                        n_kv);
                }

                inkling2_o_from_kvh_finalize_kernel<<<M_tot, 128, 0, stream>>>(d_ob, d_l, d_out, T, n_head, n_kv);
                return;
            }
        }
    }

    if (T <= 4) {
        if (head_dim == 128 && n_kv > 0 && n_head == n_kv * 3) {
            constexpr int S = 32;
            const size_t n_slots = static_cast<size_t>(T) * n_head * S;
            float * pm = static_cast<float *>(d_scratch);
            float * pl = pm + n_slots;
            float * po = pl + n_slots;
            dim3 grid(n_kv, S, T);
            if (kv_is_q8_0) {
                inkling2_attn_splitk_gqa_kernel<true, S, 3, 4><<<grid, dim3(32, 4), 0, stream>>>(
                    d_q, d_k_cache, d_v_cache, d_bias_table, d_pos, pm, pl, po,
                    n_head, n_kv, swa_window, n_extent, cache_cap, scale);
            } else {
                inkling2_attn_splitk_gqa_kernel<false, S, 3, 4><<<grid, dim3(32, 4), 0, stream>>>(
                    d_q, d_k_cache, d_v_cache, d_bias_table, d_pos, pm, pl, po,
                    n_head, n_kv, swa_window, n_extent, cache_cap, scale);
            }
            inkling2_attn_reduce_sink_kernel<S><<<dim3(n_head, T), 32, 0, stream>>>(pm, pl, po, d_out, n_head);
            return;
        }
        constexpr int S = 16;
        const size_t n_slots = static_cast<size_t>(T) * n_head * S;
        float * pm = static_cast<float *>(d_scratch);
        float * pl = pm + n_slots;
        float * po = pl + n_slots;
        dim3 grid(n_head, S, T);
        if (kv_is_q8_0) {
            inkling2_attn_splitk_kernel<true, S><<<grid, 32, 0, stream>>>(
                d_q, d_k_cache, d_v_cache, d_bias_table, d_pos, pm, pl, po,
                n_head, n_kv, swa_window, n_extent, cache_cap, scale);
        } else {
            inkling2_attn_splitk_kernel<false, S><<<grid, 32, 0, stream>>>(
                d_q, d_k_cache, d_v_cache, d_bias_table, d_pos, pm, pl, po,
                n_head, n_kv, swa_window, n_extent, cache_cap, scale);
        }
        inkling2_attn_reduce_sink_kernel<S><<<dim3(n_head, T), 32, 0, stream>>>(pm, pl, po, d_out, n_head);
    } else {
        constexpr int S = 1;
        const size_t n_slots = static_cast<size_t>(T) * n_head;
        float * pm = static_cast<float *>(d_scratch);
        float * pl = pm + n_slots;
        float * po = pl + n_slots;
        dim3 grid(n_head, 1, T);
        if (kv_is_q8_0) {
            inkling2_attn_splitk_kernel<true, S><<<grid, 32, 0, stream>>>(
                d_q, d_k_cache, d_v_cache, d_bias_table, d_pos, pm, pl, po,
                n_head, n_kv, swa_window, n_extent, cache_cap, scale);
        } else {
            inkling2_attn_splitk_kernel<false, S><<<grid, 32, 0, stream>>>(
                d_q, d_k_cache, d_v_cache, d_bias_table, d_pos, pm, pl, po,
                n_head, n_kv, swa_window, n_extent, cache_cap, scale);
        }
        inkling2_attn_reduce_sink_kernel<S><<<dim3(n_head, T), 32, 0, stream>>>(pm, pl, po, d_out, n_head);
    }
}

void launch_inkling2_sigmoid_gate_mul(float * d_x, const float * d_gate, int64_t n, cudaStream_t stream) {
    inkling2_sigmoid_gate_mul_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_x, d_gate, n);
}

void launch_inkling2_moe_ffn(
    cublasHandle_t,
    const DeviceTensor & gate_up_exps, const DeviceTensor & down_exps,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, float situ_beta, float situ_linear_beta, cudaStream_t stream) {

    if (gate_up_exps.dtype == DType::IQ4_NL && gate_up_exps.is_soa &&
        down_exps.dtype == DType::IQ4_NL && down_exps.is_soa) {
        const int K        = static_cast<int>(gate_up_exps.ne[0]);
        const int ff       = static_cast<int>(gate_up_exps.ne[1]) / 2;
        const int n_expert = static_cast<int>(gate_up_exps.ne[2]);
        const int K_blocks = K >> 5;
        const int ff_blks  = ff >> 5;

        auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
        auto * base = static_cast<uint8_t *>(d_scratch);
        const auto * gu_q = static_cast<const int4 *>(gate_up_exps.ptr);
        const auto * gu_s = static_cast<const uint16_t *>(gate_up_exps.scales_ptr);
        const auto * dn_q = static_cast<const int4 *>(down_exps.ptr);
        const auto * dn_s = static_cast<const uint16_t *>(down_exps.scales_ptr);

        const bool use_sm60_h2 = (device_sm_version(gate_up_exps.device) == 60);

        if (use_sm60_h2) {
            size_t off = 0;
            auto * x_h2 = reinterpret_cast<__half2 *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * 16 * sizeof(__half2));
            auto * x_d  = reinterpret_cast<float   *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * sizeof(float));
            auto * act  = reinterpret_cast<float   *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
            auto * a_h2 = reinterpret_cast<__half2 *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * 16 * sizeof(__half2));
            auto * a_d  = reinterpret_cast<float   *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(float));
            void * rest = base + off;
            const size_t rem_b = (off < scratch_bytes) ? (scratch_bytes - off) : 0;

            constexpr int q_wpb = 8;
            const int x_blks = T * K_blocks;
            inkling2_quantize_f16_pair_kernel<<<(x_blks + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(
                d_x, x_h2, x_d, x_blks);

            constexpr int BT_GU = 16, KT_GU = 12;
            constexpr int BT_DN = 16, KT_DN = 18;
            const size_t gu_smem = static_cast<size_t>(BT_GU) * KT_GU * (16 * sizeof(__half2) + sizeof(float));
            const size_t dn_smem = static_cast<size_t>(BT_DN) * KT_DN * (16 * sizeof(__half2) + sizeof(float));
            const int max_tiles = (T * k_used + BT_GU - 1) / BT_GU + n_expert;
            const size_t off_b  = align256(static_cast<size_t>(n_expert + 1) * sizeof(int32_t));
            const size_t sp_b   = align256(static_cast<size_t>(T) * k_used * sizeof(int32_t));
            const size_t tl_b   = align256(static_cast<size_t>(max_tiles) * sizeof(Ink2TileDesc));
            const size_t nt_b   = 256;
            const size_t sl_b   = align256(static_cast<size_t>(T) * k_used * K * sizeof(float));

            if (T >= 16 && n_expert <= 512 && k_used <= 16 &&
                gu_smem <= 45056 && dn_smem <= 45056 && (off_b + sp_b + tl_b + nt_b + sl_b <= rem_b)) {
                auto * rbase        = static_cast<uint8_t *>(rest);
                auto * exp_offsets  = reinterpret_cast<int32_t *>(rbase);
                auto * sorted_pairs = reinterpret_cast<int32_t *>(rbase + off_b);
                auto * tiles        = reinterpret_cast<Ink2TileDesc *>(rbase + off_b + sp_b);
                auto * n_tiles_ptr  = reinterpret_cast<int32_t *>(rbase + off_b + sp_b + tl_b);
                auto * down_slots   = reinterpret_cast<float   *>(rbase + off_b + sp_b + tl_b + nt_b);

                inkling2_group_moe_tokens_kernel<<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
                    d_top_ids, exp_offsets, sorted_pairs, T * k_used, n_expert, k_used);
                inkling2_build_tile_worklist_kernel<<<1, 1, 0, stream>>>(
                    exp_offsets, tiles, n_tiles_ptr, n_expert, BT_GU);

                constexpr int wpb_gu = 4;
                constexpr int rows_gu = wpb_gu * 16;
                inkling2_moe_gate_up_grouped_iq4_nl_soa_f16_kernel<BT_GU, KT_GU>
                    <<<dim3((ff + rows_gu - 1) / rows_gu, max_tiles), dim3(32, wpb_gu), gu_smem, stream>>>(
                        gu_q, gu_s, x_h2, x_d, tiles, n_tiles_ptr, sorted_pairs, act,
                        K_blocks, ff, k_used, situ_beta, situ_linear_beta);

                const int a_blks = T * k_used * ff_blks;
                inkling2_quantize_f16_pair_kernel<<<(a_blks + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(
                    act, a_h2, a_d, a_blks);

                constexpr int wpb_dn = 8;
                constexpr int rows_dn = wpb_dn * 16;
                inkling2_moe_down_grouped_iq4_nl_soa_f16_kernel<BT_DN, KT_DN>
                    <<<dim3((K + rows_dn - 1) / rows_dn, max_tiles), dim3(32, wpb_dn), dn_smem, stream>>>(
                        dn_q, dn_s, a_h2, a_d, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots,
                        ff_blks, K, k_used);

                const int red_n = ((K & 3) == 0) ? (K >> 2) : K;
                inkling2_reduce_slots_kernel<<<dim3((red_n + 255) / 256, T), 256, 0, stream>>>(down_slots, d_out, K, k_used);
                return;
            }

            // Decode (T < 16) sm_60: zero-bank-conflict shfl_f32 LUT + fused quantize_f16_pair + split8 down!
            constexpr int wpb_gu = 8;
            const size_t dec_gu_smem = static_cast<size_t>(K_blocks) * (16 * sizeof(__half2) + sizeof(float));
            inkling2_moe_gate_up_situ_iq4_nl_soa_f16_kernel
                <<<dim3((ff + 31) / 32, k_used, T), dim3(32, wpb_gu), dec_gu_smem, stream>>>(
                    gu_q, gu_s, x_h2, x_d, d_top_ids, a_h2, a_d, K_blocks, ff, k_used, situ_beta, situ_linear_beta);

            auto * down_slots = reinterpret_cast<float *>(rest);
            constexpr int wpb_dn = 4;
            const size_t dec_dn_smem = static_cast<size_t>(ff_blks) * (16 * sizeof(__half2) + sizeof(float));
            inkling2_moe_down_iq4_nl_soa_f16_split_kernel
                <<<dim3((K + wpb_dn * 8 - 1) / (wpb_dn * 8), k_used, T), dim3(32, wpb_dn), dec_dn_smem, stream>>>(
                    dn_q, dn_s, a_h2, a_d, d_top_ids, d_top_w, down_slots, ff_blks, K, k_used);

            const int red_n = ((K & 3) == 0) ? (K >> 2) : K;
            inkling2_reduce_slots_kernel<<<dim3((red_n + 63) / 64, T), 64, 0, stream>>>(down_slots, d_out, K, k_used);
            return;
        }

        // Non-sm_60 (sm_61+ hardware dp4a / sm_52): coalesced IQ4_NL SoA + grouped prefill!
        size_t off = 0;
        auto * xq8 = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
        auto * act = reinterpret_cast<float *>        (base + off); off += align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
        auto * aq8 = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BlockQ8_1_Act));
        void * rest = base + off;
        const size_t rem_b = (off < scratch_bytes) ? (scratch_bytes - off) : 0;

        launch_quantize_q8_1(d_x, xq8, K, T, stream);

        constexpr int BT_GU = 8, R_GU = 2;
        constexpr int BT_DN = 12, R_DN = 2;
        const size_t gu_smem = static_cast<size_t>(BT_GU) * K_blocks * sizeof(BlockQ8_1_Act);
        const size_t dn_smem = static_cast<size_t>(BT_DN) * ff_blks  * sizeof(BlockQ8_1_Act);
        const size_t off_b   = align256(static_cast<size_t>(n_expert + 1) * sizeof(int32_t));
        const size_t sp_b    = align256(static_cast<size_t>(T) * k_used * sizeof(int32_t));
        const size_t sl_b    = align256(static_cast<size_t>(T) * k_used * K * sizeof(float));

        if (T >= 16 && n_expert <= 512 && k_used <= 16 &&
            gu_smem <= 48000 && dn_smem <= 48000 && (off_b + sp_b + sl_b <= rem_b)) {
            auto * exp_offsets  = reinterpret_cast<int32_t *>(rest);
            auto * sorted_pairs = reinterpret_cast<int32_t *>(static_cast<uint8_t *>(rest) + off_b);
            auto * down_slots   = reinterpret_cast<float   *>(static_cast<uint8_t *>(rest) + off_b + sp_b);

            inkling2_group_moe_tokens_kernel<<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
                d_top_ids, exp_offsets, sorted_pairs, T * k_used, n_expert, k_used);

            constexpr int wpb_gu = 8;
            constexpr int rows_gu = wpb_gu * 4 * R_GU;
            inkling2_moe_gate_up_grouped_iq4_nl_soa_dp4a_kernel<BT_GU, R_GU>
                <<<dim3((ff + rows_gu - 1) / rows_gu, n_expert), dim3(32, wpb_gu), gu_smem, stream>>>(
                    gu_q, gu_s, xq8, exp_offsets, sorted_pairs, act, K_blocks, ff, k_used, situ_beta, situ_linear_beta);

            launch_quantize_q8_1(act, aq8, k_used * ff, T, stream);

            constexpr int wpb_dn = 8;
            constexpr int rows_dn = wpb_dn * 8 * R_DN;
            inkling2_moe_down_grouped_iq4_nl_soa_dp4a_kernel<BT_DN, R_DN>
                <<<dim3((K + rows_dn - 1) / rows_dn, n_expert), dim3(32, wpb_dn), dn_smem, stream>>>(
                    dn_q, dn_s, aq8, d_top_w, exp_offsets, sorted_pairs, down_slots, ff_blks, K, k_used);

            inkling2_reduce_slots_kernel<<<dim3((((K & 3) == 0 ? (K >> 2) : K) + 255) / 256, T), 256, 0, stream>>>(down_slots, d_out, K, k_used);
            return;
        }

        constexpr int wpb_gu = 8;
        const size_t dec_gu_smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
        inkling2_moe_gate_up_situ_iq4_nl_soa_dp4a_kernel
            <<<dim3((ff + wpb_gu * 4 - 1) / (wpb_gu * 4), k_used, T), dim3(32, wpb_gu), dec_gu_smem, stream>>>(
                gu_q, gu_s, xq8, d_top_ids, act, K_blocks, ff, k_used, situ_beta, situ_linear_beta);

        launch_quantize_q8_1(act, aq8, k_used * ff, T, stream);

        constexpr int wpb_dn = 2;
        const size_t dec_dn_smem = static_cast<size_t>(k_used) * ff_blks * sizeof(BlockQ8_1_Act);
        inkling2_moe_down_iq4_nl_soa_dp4a_kernel
            <<<dim3((K + wpb_dn * 8 - 1) / (wpb_dn * 8), T), dim3(32, wpb_dn), dec_dn_smem, stream>>>(
                dn_q, dn_s, aq8, d_top_ids, d_top_w, d_out, ff_blks, K, k_used);
        return;
    }

    if (gate_up_exps.dtype != DType::Q8_0 || !gate_up_exps.is_soa ||
        down_exps.dtype != DType::Q8_0 || !down_exps.is_soa) {
        launch_inkling2_moe_ffn_kquant(
            gate_up_exps, down_exps, d_x, d_top_ids, d_top_w,
            d_out, d_scratch, scratch_bytes, T, k_used, situ_beta, situ_linear_beta, stream);
        return;
    }

    const int K        = static_cast<int>(gate_up_exps.ne[0]);
    const int ff       = static_cast<int>(gate_up_exps.ne[1]) / 2;
    const int n_expert = static_cast<int>(gate_up_exps.ne[2]);
    const int K_blocks = K >> 5;
    const int ff_blks  = ff >> 5;

    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    const auto * gu_q = static_cast<const int4 *>(gate_up_exps.ptr);
    const auto * gu_s = static_cast<const uint16_t *>(gate_up_exps.scales_ptr);
    const auto * dn_q = static_cast<const int4 *>(down_exps.ptr);
    const auto * dn_s = static_cast<const uint16_t *>(down_exps.scales_ptr);

    // Native sm_60 (no dp4a, 2x-rate half2): K-tiled BT=16, KT=12, LPR=4 0x6400-PRMT half2 grouped MoE!
    if (T >= 16 && n_expert <= 512 && k_used <= 16 && device_sm_version(gate_up_exps.device) == 60) {
        auto * base = static_cast<uint8_t *>(d_scratch);
        size_t off = 0;
        auto * x_h2 = reinterpret_cast<__half2 *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * 16 * sizeof(__half2));
        auto * x_d  = reinterpret_cast<float   *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * sizeof(float));
        auto * act  = reinterpret_cast<float   *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
        auto * a_h2 = reinterpret_cast<__half2 *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * 16 * sizeof(__half2));
        auto * a_d  = reinterpret_cast<float   *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(float));
        const size_t rem_b = (off < scratch_bytes) ? (scratch_bytes - off) : 0;

        constexpr int BT = 16, KT = 12;
        const size_t smem_b = static_cast<size_t>(KT) * BT * (16 * sizeof(__half2) + sizeof(float));
        const int max_tiles = (T * k_used + BT - 1) / BT + n_expert;
        const size_t off_b  = align256(static_cast<size_t>(n_expert + 1) * sizeof(int32_t));
        const size_t sp_b   = align256(static_cast<size_t>(T) * k_used * sizeof(int32_t));
        const size_t tl_b   = align256(static_cast<size_t>(max_tiles) * sizeof(Ink2TileDesc));
        const size_t nt_b   = 256;
        const size_t sl_b   = align256(static_cast<size_t>(T) * k_used * K * sizeof(float));

        if (smem_b <= 45056 && (off_b + sp_b + tl_b + nt_b + sl_b <= rem_b)) {
            auto * rbase        = base + off;
            auto * exp_offsets  = reinterpret_cast<int32_t *>(rbase);
            auto * sorted_pairs = reinterpret_cast<int32_t *>(rbase + off_b);
            auto * tiles        = reinterpret_cast<Ink2TileDesc *>(rbase + off_b + sp_b);
            auto * n_tiles_ptr  = reinterpret_cast<int32_t *>(rbase + off_b + sp_b + tl_b);
            auto * down_slots   = reinterpret_cast<float   *>(rbase + off_b + sp_b + tl_b + nt_b);

            constexpr int q_wpb = 8;
            const int x_blks = T * K_blocks;
            inkling2_quantize_f16_contig_kernel<<<(x_blks + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(
                d_x, x_h2, x_d, x_blks);
            inkling2_group_moe_tokens_kernel<<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
                d_top_ids, exp_offsets, sorted_pairs, T * k_used, n_expert, k_used);
            inkling2_build_tile_worklist_kernel<<<1, 1, 0, stream>>>(
                exp_offsets, tiles, n_tiles_ptr, n_expert, BT);

            inkling2_moe_gate_up_grouped_q8_0_soa_f16_kernel<BT, KT>
                <<<dim3((ff + 31) / 32, max_tiles), 128, smem_b, stream>>>(
                    gu_q, gu_s, x_h2, x_d, tiles, n_tiles_ptr, sorted_pairs, act,
                    K_blocks, ff, k_used, situ_beta, situ_linear_beta);

            const int a_blks = T * k_used * ff_blks;
            inkling2_quantize_f16_contig_kernel<<<(a_blks + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(
                act, a_h2, a_d, a_blks);

            inkling2_moe_down_grouped_q8_0_soa_f16_kernel<BT, KT>
                <<<dim3((K + 31) / 32, max_tiles), 128, smem_b, stream>>>(
                    dn_q, dn_s, a_h2, a_d, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots,
                    ff_blks, K, k_used);

            const int red_n = ((K & 3) == 0) ? (K >> 2) : K;
            inkling2_reduce_slots_kernel<<<dim3((red_n + 255) / 256, T), 256, 0, stream>>>(down_slots, d_out, K, k_used);
            return;
        }
    }

    auto * base = static_cast<uint8_t *>(d_scratch);
    size_t off = 0;
    auto * xq8 = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
    auto * act = reinterpret_cast<float *>        (base + off); off += align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
    auto * aq8 = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BlockQ8_1_Act));
    void * rest = base + off;
    const size_t rem_b = (off < scratch_bytes) ? (scratch_bytes - off) : 0;

    launch_quantize_q8_1(d_x, xq8, K, T, stream);

    constexpr int wpb = 8;
    constexpr int BT = 16, KT = 12;
    const size_t smem_kt = static_cast<size_t>(KT) * BT * (2 * sizeof(int4) + sizeof(float));
    const int max_tiles = (T * k_used + BT - 1) / BT + n_expert;
    const size_t off_b   = align256(static_cast<size_t>(n_expert + 1) * sizeof(int32_t));
    const size_t sp_b    = align256(static_cast<size_t>(T) * k_used * sizeof(int32_t));
    const size_t tl_b    = align256(static_cast<size_t>(max_tiles) * sizeof(Ink2TileDesc));
    const size_t nt_b    = 256;
    const size_t sl_b    = align256(static_cast<size_t>(T) * k_used * K * sizeof(float));

    if (T >= 16 && n_expert <= 512 && k_used <= 16 &&
        smem_kt <= 45056 && (off_b + sp_b + tl_b + nt_b + sl_b <= rem_b)) {
        auto * rbase        = static_cast<uint8_t *>(rest);
        auto * exp_offsets  = reinterpret_cast<int32_t *>(rbase);
        auto * sorted_pairs = reinterpret_cast<int32_t *>(rbase + off_b);
        auto * tiles        = reinterpret_cast<Ink2TileDesc *>(rbase + off_b + sp_b);
        auto * n_tiles_ptr  = reinterpret_cast<int32_t *>(rbase + off_b + sp_b + tl_b);
        auto * down_slots   = reinterpret_cast<float   *>(rbase + off_b + sp_b + tl_b + nt_b);

        inkling2_group_moe_tokens_kernel<<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
            d_top_ids, exp_offsets, sorted_pairs, T * k_used, n_expert, k_used);
        inkling2_build_tile_worklist_kernel<<<1, 1, 0, stream>>>(
            exp_offsets, tiles, n_tiles_ptr, n_expert, BT);

        inkling2_moe_gate_up_grouped_q8_0_ktiled_dp4a_kernel<BT, KT>
            <<<dim3((ff + 31) / 32, max_tiles), 128, smem_kt, stream>>>(
                gu_q, gu_s, xq8, tiles, n_tiles_ptr, sorted_pairs, act, K_blocks, ff, k_used, situ_beta, situ_linear_beta);

        launch_quantize_q8_1(act, aq8, k_used * ff, T, stream);

        inkling2_moe_down_grouped_q8_0_ktiled_dp4a_kernel<BT, KT>
            <<<dim3((K + 31) / 32, max_tiles), 128, smem_kt, stream>>>(
                dn_q, dn_s, aq8, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, ff_blks, K, k_used);

        inkling2_reduce_slots_kernel<<<dim3((((K & 3) == 0 ? (K >> 2) : K) + 255) / 256, T), 256, 0, stream>>>(down_slots, d_out, K, k_used);
        return;
    }

    // Decode (T < 16): ultra-low-latency LDG.128 + dp4a!
    const size_t dec_gu_smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
    inkling2_moe_gate_up_situ_decode_kernel<<<dim3((ff + wpb - 1) / wpb, k_used, T), dim3(32, wpb), dec_gu_smem, stream>>>(
        gu_q, gu_s, xq8, d_top_ids, act, K_blocks, ff, k_used, situ_beta, situ_linear_beta);

    launch_quantize_q8_1(act, aq8, k_used * ff, T, stream);

    const size_t dec_dn_smem = static_cast<size_t>(ff_blks) * sizeof(BlockQ8_1_Act);
    inkling2_moe_down_decode_kernel<<<dim3((K + wpb - 1) / wpb, T), dim3(32, wpb), dec_dn_smem, stream>>>(
        dn_q, dn_s, aq8, d_top_ids, d_top_w, d_out, ff_blks, K, k_used);
}

void launch_inkling2_logit_softcap(float * d_logits, int64_t n, float cap, cudaStream_t stream) {
    if (cap <= 0.0f) return;
    inkling2_logit_softcap_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_logits, n, cap, 1.0f / cap);
}

} // namespace rookery
