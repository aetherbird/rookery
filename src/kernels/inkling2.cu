#include "core/device.h"
#include "kernels/kernels.h"
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
        out[static_cast<size_t>(t) * C + c] = (xt + acc) * inv_scale;
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
    const float * __restrict__ r,          // [T, n_head, 16]
    const float * __restrict__ rel_bias,   // [n_extent, 16]
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

// ============================================================================
// 6. KV Cache Write (Q8_0 or F16) with Ring Buffer Modulo `cache_cap` for SWA!
// ============================================================================
__global__ void inkling2_kv_write_q8_0_kernel(
    const float *   __restrict__ k,
    const float *   __restrict__ v,
    BlockQ8_0 *     __restrict__ k_cache,
    BlockQ8_0 *     __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    int T, int n_kv, int head_dim, int cache_cap) {

    const int n_blk_h = head_dim >> 5; // 4 for d=128
    const int total_b = n_kv * n_blk_h;
    const int b    = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    if (t >= T || b >= total_b) return;

    const int lane = threadIdx.x;
    const int slot = pos[t] % cache_cap;
    const size_t src_off = static_cast<size_t>(t) * (n_kv * head_dim) + b * 32 + lane;
    const size_t dst_b   = static_cast<size_t>(slot) * total_b + b;

    auto quant_store = [&](const float * src, BlockQ8_0 * dst) {
        const float val  = src[src_off];
        const float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(val)), 0);
        const float d    = amax / 127.0f;
        const float inv  = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        if (lane == 0) {
            __half dh = __float2half_rn(d);
            dst[dst_b].d = *reinterpret_cast<uint16_t *>(&dh);
        }
        dst[dst_b].qs[lane] = static_cast<int8_t>(__float2int_rn(val * inv));
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
    const int off_in_b = (lane & 7) * 4;   // 0,4,8,...,28

    for (int p = p0; p < p1; ++p) {
        const int slot = p % cache_cap;
        float4 kv, vv;
        if constexpr (KV_Q8_0) {
            const size_t bidx = (static_cast<size_t>(slot) * n_kv + kvh) * 4 + blk_in_h;
            const BlockQ8_0 & kb = static_cast<const BlockQ8_0 *>(k_cache)[bidx];
            const BlockQ8_0 & vb = static_cast<const BlockQ8_0 *>(v_cache)[bidx];
            __half kdh, vdh;
            *reinterpret_cast<uint16_t *>(&kdh) = __ldg(&kb.d);
            *reinterpret_cast<uint16_t *>(&vdh) = __ldg(&vb.d);
            const float kd = __half2float(kdh);
            const float vd = __half2float(vdh);
            // BlockQ8_0 is 34 bytes (d is 2B -> qs[off_in_b] is 2-byte aligned, NOT 4-byte aligned!).
            const uint16_t kq0 = __ldg(reinterpret_cast<const uint16_t *>(&kb.qs[off_in_b + 0]));
            const uint16_t kq1 = __ldg(reinterpret_cast<const uint16_t *>(&kb.qs[off_in_b + 2]));
            const uint16_t vq0 = __ldg(reinterpret_cast<const uint16_t *>(&vb.qs[off_in_b + 0]));
            const uint16_t vq1 = __ldg(reinterpret_cast<const uint16_t *>(&vb.qs[off_in_b + 2]));
            kv = make_float4(kd * static_cast<int8_t>(kq0 & 0xFF),
                             kd * static_cast<int8_t>(kq0 >> 8),
                             kd * static_cast<int8_t>(kq1 & 0xFF),
                             kd * static_cast<int8_t>(kq1 >> 8));
            vv = make_float4(vd * static_cast<int8_t>(vq0 & 0xFF),
                             vd * static_cast<int8_t>(vq0 >> 8),
                             vd * static_cast<int8_t>(vq1 & 0xFF),
                             vd * static_cast<int8_t>(vq1 >> 8));
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
        const size_t bidx = (static_cast<size_t>(slot) * n_kv + kvh) * 4 + b;
        const BlockQ8_0 & kb = static_cast<const BlockQ8_0 *>(k_cache)[bidx];
        const BlockQ8_0 & vb = static_cast<const BlockQ8_0 *>(v_cache)[bidx];
        __half kdh, vdh;
        *reinterpret_cast<uint16_t *>(&kdh) = __ldg(&kb.d);
        *reinterpret_cast<uint16_t *>(&vdh) = __ldg(&vb.d);
        km[dst_base + b * 32 + lane] = ink2_f32_to_mat<TMat>(__half2float(kdh) * static_cast<float>(__ldg(&kb.qs[lane])));
        vm[dst_base + b * 32 + lane] = ink2_f32_to_mat<TMat>(__half2float(vdh) * static_cast<float>(__ldg(&vb.qs[lane])));
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
    const int32_t *       __restrict__ exp_offsets,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ act,
    int K_blocks, int ff, int k_used, float beta, float lb) {

    const int eid   = blockIdx.y;
    const int start = exp_offsets[eid];
    const int end   = exp_offsets[eid + 1];
    if (start >= end) return;

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
    const int32_t *       __restrict__ exp_offsets,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots, // [T, k_used, n_embd] (deterministic!)
    int ff_blocks, int n_embd, int k_used) {

    const int eid   = blockIdx.y;
    const int start = exp_offsets[eid];
    const int end   = exp_offsets[eid + 1];
    if (start >= end) return;

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
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= n_embd) return;
    const float * base = slots + static_cast<size_t>(t) * k_used * n_embd + d;
    float sum = 0.0f;
    for (int s = 0; s < k_used; ++s) sum += base[static_cast<size_t>(s) * n_embd];
    out[static_cast<size_t>(t) * n_embd + d] = sum;
}

} // namespace

void launch_inkling2_depth_mix(
    const float * const * d_blocks_dev, int n_blocks,
    const float * d_cur, const float * d_query, float * d_out,
    int n_embd, int n_q, int T, float eps, cudaStream_t stream) {
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
    int T, int C, int n_head_kv, cudaStream_t stream) {
    const int head_dim = (n_head_kv > 0) ? (C / n_head_kv) : C;
    inkling2_shortconv_kernel<<<(C + 255) / 256, 256, 0, stream>>>(
        d_x, d_w, d_dyn, d_state, d_out, T, C, head_dim, n_head_kv);
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
            d_k, d_v, static_cast<BlockQ8_0 *>(d_k_cache), static_cast<BlockQ8_0 *>(d_v_cache),
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
        const bool use_tc = device_has_tensor_cores(-1);
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
    auto * base = static_cast<uint8_t *>(d_scratch);
    size_t off = 0;
    auto * xq8 = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
    auto * act = reinterpret_cast<float *>        (base + off); off += align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
    auto * aq8 = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BlockQ8_1_Act));
    void * rest = base + off;
    const size_t rem_b = (off < scratch_bytes) ? (scratch_bytes - off) : 0;

    launch_quantize_q8_1(d_x, xq8, K, T, stream);

    const auto * gu_q = static_cast<const int4 *>(gate_up_exps.ptr);
    const auto * gu_s = static_cast<const uint16_t *>(gate_up_exps.scales_ptr);
    const auto * dn_q = static_cast<const int4 *>(down_exps.ptr);
    const auto * dn_s = static_cast<const uint16_t *>(down_exps.scales_ptr);

    constexpr int wpb = 8;
    constexpr int BT_GU = 8, BT_DN = 12, R_PER_WARP = 4;
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

        constexpr int rows_blk = wpb * R_PER_WARP;
        inkling2_moe_gate_up_grouped_kernel<BT_GU, R_PER_WARP>
            <<<dim3((ff + rows_blk - 1) / rows_blk, n_expert), dim3(32, wpb), gu_smem, stream>>>(
                gu_q, gu_s, xq8, exp_offsets, sorted_pairs, act, K_blocks, ff, k_used, situ_beta, situ_linear_beta);

        launch_quantize_q8_1(act, aq8, k_used * ff, T, stream);

        inkling2_moe_down_grouped_kernel<BT_DN, R_PER_WARP>
            <<<dim3((K + rows_blk - 1) / rows_blk, n_expert), dim3(32, wpb), dn_smem, stream>>>(
                dn_q, dn_s, aq8, d_top_w, exp_offsets, sorted_pairs, down_slots, ff_blks, K, k_used);

        inkling2_reduce_slots_kernel<<<dim3((K + 255) / 256, T), 256, 0, stream>>>(down_slots, d_out, K, k_used);
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
