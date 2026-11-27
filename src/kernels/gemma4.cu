#include "core/device.h"
#include "kernels/kernels.h"
#include "kernels/q8_1_act.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace rookery {

namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoefA   = 0.044715f;

__device__ __forceinline__ float g4_gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * x * (1.0f + kGeluCoefA * x * x)));
}

__device__ __forceinline__ float g4_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float g4_warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

// ============================================================================
// 1. Elementwise / Normalization / PLE / Activation / Softcap Kernels
// ============================================================================

__global__ void g4_scale_inplace_kernel(float * __restrict__ x, int64_t n, float scale) {
    const int64_t i4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t i  = i4 * 4;
    if (i + 3 < n) {
        float4 v = reinterpret_cast<float4 *>(x)[i4];
        v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
        reinterpret_cast<float4 *>(x)[i4] = v;
    } else {
        for (int64_t j = i; j < n && j < i + 4; ++j) x[j] *= scale;
    }
}

__global__ void g4_scale_dev_inplace_kernel(float * __restrict__ x, const float * __restrict__ d_scale, int64_t n) {
    const float scale = __ldg(d_scale);
    const int64_t i4  = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t i   = i4 * 4;
    if (i + 3 < n) {
        float4 v = reinterpret_cast<float4 *>(x)[i4];
        v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
        reinterpret_cast<float4 *>(x)[i4] = v;
    } else {
        for (int64_t j = i; j < n && j < i + 4; ++j) x[j] *= scale;
    }
}

// Fused PLE combine:
//   proj[t, il, c] -> rmsnorm(eps = rms_eps * n_embd) * proj_norm_w[c] * (1/sqrt(2))
//                   + raw_ple[t, il, c] * (sqrt(ple_dim) / sqrt(2))
__global__ void g4_ple_combine_kernel(
    const float * __restrict__ proj,
    const float * __restrict__ raw_ple,
    const float * __restrict__ proj_norm_w,
    float *       __restrict__ ple_out,
    int n_layers, int ple_dim, float proj_eps,
    float proj_norm_scale, float tok_ple_scale) {

    const int il = blockIdx.x;
    const int t  = blockIdx.y;
    const size_t row_off = (static_cast<size_t>(t) * n_layers + il) * ple_dim;
    const float * pr = proj    + row_off;
    const float * rr = raw_ple + row_off;
    float *       yr = ple_out + row_off;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < ple_dim; i += blockDim.x) {
        const float v = pr[i];
        ss += v * v;
    }
    __shared__ float smem[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    ss = g4_warp_sum(ss);
    if (lane == 0) smem[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        tot = g4_warp_sum(tot);
        if (lane == 0) smem[0] = rsqrtf(tot / static_cast<float>(ple_dim) + proj_eps);
    }
    __syncthreads();
    const float inv_rms = smem[0] * proj_norm_scale;

    for (int i = threadIdx.x; i < ple_dim; i += blockDim.x) {
        yr[i] = pr[i] * inv_rms * __ldg(&proj_norm_w[i]) + rr[i] * tok_ple_scale;
    }
}

__global__ void g4_geglu_kernel(
    const float * __restrict__ gate,
    const float * __restrict__ up,
    float *       __restrict__ out,
    int64_t n) {

    const int64_t i4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t i  = i4 * 4;
    if (i + 3 < n) {
        const float4 g = reinterpret_cast<const float4 *>(gate)[i4];
        const float4 u = reinterpret_cast<const float4 *>(up)[i4];
        reinterpret_cast<float4 *>(out)[i4] = make_float4(
            g4_gelu_tanh(g.x) * u.x,
            g4_gelu_tanh(g.y) * u.y,
            g4_gelu_tanh(g.z) * u.z,
            g4_gelu_tanh(g.w) * u.w);
    } else {
        for (int64_t j = i; j < n && j < i + 4; ++j) {
            out[j] = g4_gelu_tanh(gate[j]) * up[j];
        }
    }
}

__global__ void g4_ple_gate_gelu_mul_kernel(
    float *       __restrict__ gate,    // [T, ple_dim] in-place
    const float * __restrict__ ple,     // [T_full, n_layers, ple_dim]
    int T, int n_layers, int il, int ple_dim, int ple_tok_off) {

    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || c >= ple_dim) return;
    const size_t g_idx = static_cast<size_t>(t) * ple_dim + c;
    const size_t p_idx = (static_cast<size_t>(ple_tok_off + t) * n_layers + il) * ple_dim + c;
    gate[g_idx] = g4_gelu_tanh(gate[g_idx]) * __ldg(&ple[p_idx]);
}

// Fast T=1 F16 PLE Gate GEMV + GeGLU * ple[il]:
//   gate[r] = gelu_tanh(dot(W_gate[r, :], x[:])) * ple[ple_tok_off, il, r]
// Requires H % 8 == 0. One warp per output row, 8 warps per block.
__global__ void g4_ple_f16_gate_mul_kernel(
    const __half * __restrict__ w_gate,
    const float *  __restrict__ x,
    const float *  __restrict__ ple,
    float *        __restrict__ gate_out,
    int H, int ple_dim, int n_layers, int il, int ple_tok_off) {

    const int row = blockIdx.x * 8 + threadIdx.y;
    if (row >= ple_dim) return;
    const int lane = threadIdx.x;
    const int H8   = H >> 3;
    const float4 * w4 = reinterpret_cast<const float4 *>(w_gate + static_cast<size_t>(row) * H);
    const float4 * x4 = reinterpret_cast<const float4 *>(x);

    float acc = 0.0f;
    for (int i = lane; i < H8; i += 32) {
        const float4 wv = __ldg(&w4[i]);
        const __half2 * h2 = reinterpret_cast<const __half2 *>(&wv);
        const float4 xv0 = __ldg(&x4[i * 2]);
        const float4 xv1 = __ldg(&x4[i * 2 + 1]);
        const float2 w0 = __half22float2(h2[0]);
        const float2 w1 = __half22float2(h2[1]);
        const float2 w2 = __half22float2(h2[2]);
        const float2 w3 = __half22float2(h2[3]);
        acc = fmaf(w0.x, xv0.x, fmaf(w0.y, xv0.y, acc));
        acc = fmaf(w1.x, xv0.z, fmaf(w1.y, xv0.w, acc));
        acc = fmaf(w2.x, xv1.x, fmaf(w2.y, xv1.y, acc));
        acc = fmaf(w3.x, xv1.z, fmaf(w3.y, xv1.w, acc));
    }
    acc = g4_warp_sum(acc);
    if (lane == 0) {
        const size_t p_idx = (static_cast<size_t>(ple_tok_off) * n_layers + il) * ple_dim + row;
        gate_out[row] = g4_gelu_tanh(acc) * __ldg(&ple[p_idx]);
    }
}

// Fast T=1 F16 PLE Proj GEMV: out[r] = dot(W_proj[r, :], gate[:])
// Requires ple_dim % 8 == 0 and ple_dim <= 1024. One warp per output row, 8 warps per block.
__global__ void g4_ple_f16_proj_kernel(
    const __half * __restrict__ w_proj,
    const float *  __restrict__ gate,
    float *        __restrict__ out,
    int H, int ple_dim) {

    __shared__ float sg[1024];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = tid; i < ple_dim; i += 256) {
        sg[i] = gate[i];
    }
    __syncthreads();

    const int row = blockIdx.x * 8 + threadIdx.y;
    if (row >= H) return;
    const int lane = threadIdx.x;
    const int P8   = ple_dim >> 3;
    const float4 * w4 = reinterpret_cast<const float4 *>(w_proj + static_cast<size_t>(row) * ple_dim);

    float acc = 0.0f;
    for (int i = lane; i < P8; i += 32) {
        const float4 wv = __ldg(&w4[i]);
        const __half2 * h2 = reinterpret_cast<const __half2 *>(&wv);
        const int base = i * 8;
        const float2 w0 = __half22float2(h2[0]);
        const float2 w1 = __half22float2(h2[1]);
        const float2 w2 = __half22float2(h2[2]);
        const float2 w3 = __half22float2(h2[3]);
        acc = fmaf(w0.x, sg[base + 0], fmaf(w0.y, sg[base + 1], acc));
        acc = fmaf(w1.x, sg[base + 2], fmaf(w1.y, sg[base + 3], acc));
        acc = fmaf(w2.x, sg[base + 4], fmaf(w2.y, sg[base + 5], acc));
        acc = fmaf(w3.x, sg[base + 6], fmaf(w3.y, sg[base + 7], acc));
    }
    acc = g4_warp_sum(acc);
    if (lane == 0) {
        out[row] = acc;
    }
}

// Fused Post-RMSNorm + Residual Add + Optional Layer Scalar:
//   out[r, i] = (res[r, i] + rmsnorm(sub[r], norm_w, eps)[i]) * (d_layer_scale ? d_layer_scale[0] : 1.0f)
__global__ void g4_post_norm_add_scale_kernel(
    const float * __restrict__ res,
    const float * __restrict__ sub,
    const float * __restrict__ norm_w,
    const float * __restrict__ d_layer_scale,
    float *       __restrict__ out,
    int dim, float eps) {

    const int row = blockIdx.x;
    const float * rr = res + static_cast<size_t>(row) * dim;
    const float * sr = sub + static_cast<size_t>(row) * dim;
    float *       yr = out + static_cast<size_t>(row) * dim;

    float ss = 0.0f;
    if ((dim & 3) == 0) {
        const int dim4 = dim >> 2;
        const float4 * s4 = reinterpret_cast<const float4 *>(sr);
        for (int i = threadIdx.x; i < dim4; i += blockDim.x) {
            const float4 v = s4[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            const float v = sr[i];
            ss += v * v;
        }
    }

    __shared__ float smem[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    ss = g4_warp_sum(ss);
    if (lane == 0) smem[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        tot = g4_warp_sum(tot);
        if (lane == 0) smem[0] = rsqrtf(tot / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    const float inv_rms = smem[0];
    const float lscale  = d_layer_scale ? __ldg(d_layer_scale) : 1.0f;

    if ((dim & 3) == 0) {
        const int dim4 = dim >> 2;
        const float4 * r4 = reinterpret_cast<const float4 *>(rr);
        const float4 * s4 = reinterpret_cast<const float4 *>(sr);
        const float4 * w4 = reinterpret_cast<const float4 *>(norm_w);
        float4 *       y4 = reinterpret_cast<float4 *>(yr);
        for (int i = threadIdx.x; i < dim4; i += blockDim.x) {
            const float4 rv = r4[i];
            const float4 sv = s4[i];
            const float4 wv = __ldg(&w4[i]);
            y4[i] = make_float4(
                (rv.x + sv.x * inv_rms * wv.x) * lscale,
                (rv.y + sv.y * inv_rms * wv.y) * lscale,
                (rv.z + sv.z * inv_rms * wv.z) * lscale,
                (rv.w + sv.w * inv_rms * wv.w) * lscale);
        }
    } else {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            yr[i] = (rr[i] + sr[i] * inv_rms * __ldg(&norm_w[i])) * lscale;
        }
    }
}

__global__ void g4_logit_softcap_kernel(float * __restrict__ x, int64_t n, float cap) {
    const float inv_cap = 1.0f / cap;
    const int64_t i4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t i  = i4 * 4;
    if (i + 3 < n) {
        float4 v = reinterpret_cast<float4 *>(x)[i4];
        v.x = cap * tanhf(v.x * inv_cap);
        v.y = cap * tanhf(v.y * inv_cap);
        v.z = cap * tanhf(v.z * inv_cap);
        v.w = cap * tanhf(v.w * inv_cap);
        reinterpret_cast<float4 *>(x)[i4] = v;
    } else {
        for (int64_t j = i; j < n && j < i + 4; ++j) {
            x[j] = cap * tanhf(x[j] * inv_cap);
        }
    }
}

// ============================================================================
// 2. Per-Head RMSNorm + Proportional NeoX RoPE + Unweighted V-Norm + KV Write
// ============================================================================

// 1 CTA (256 threads) per (t, head): supports head_dim = 256 (SWA) and 512 (Full)!
__global__ void g4_qk_norm_rope_neox_kernel(
    float *         __restrict__ x,          // [T, n_heads, head_dim]
    const float *   __restrict__ norm_w,     // [head_dim] or nullptr
    const float *   __restrict__ rope_freqs, // [head_dim/2] divisor or nullptr
    const int32_t * __restrict__ pos,        // [T]
    int n_heads, int head_dim, int rope_dim,
    float rms_eps, float freq_base, float rope_scale) {

    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const float p_scaled = static_cast<float>(pos[t]) / rope_scale;
    float * hptr = x + (static_cast<size_t>(t) * n_heads + h) * head_dim;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = hptr[i];
        ss += v * v;
    }
    __shared__ float smem[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    ss = g4_warp_sum(ss);
    if (lane == 0) smem[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        tot = g4_warp_sum(tot);
        if (lane == 0) smem[0] = rsqrtf(tot / static_cast<float>(head_dim) + rms_eps);
    }
    __syncthreads();
    const float inv_rms = smem[0];

    const int half_d = head_dim >> 1;
    const int half_r = rope_dim >> 1;
    for (int i = threadIdx.x; i < half_d; i += blockDim.x) {
        const float x0 = hptr[i]          * inv_rms * (norm_w ? __ldg(&norm_w[i])          : 1.0f);
        const float x1 = hptr[i + half_d] * inv_rms * (norm_w ? __ldg(&norm_w[i + half_d]) : 1.0f);
        if (i < half_r) {
            const float ffac = rope_freqs ? __ldg(&rope_freqs[i]) : 1.0f;
            if (ffac > 1e20f) {
                hptr[i]          = x0;
                hptr[i + half_d] = x1;
            } else {
                const float theta = p_scaled * powf(freq_base, -2.0f * static_cast<float>(i) / static_cast<float>(rope_dim)) / ffac;
                float s, c;
                sincosf(theta, &s, &c);
                hptr[i]          = x0 * c - x1 * s;
                hptr[i + half_d] = x0 * s + x1 * c;
            }
        } else {
            hptr[i]          = x0;
            hptr[i + half_d] = x1;
        }
    }
}

// Unweighted per-head V RMSNorm + K/V cache write (16B-aligned SoA Q8_0, SoA Q4_0, or F16!)
// Ring-buffer indexing: slot = pos % kv_cap (enables SWA ring buffer for 128K-1M context!)
// Q8_0 SoA layout: [0 .. kv_cap * n_kv * head_dim) int8_t qs + uint16_t d[kv_cap * n_kv * (head_dim/32)]
// Q4_0 SoA layout: [0 .. kv_cap * n_kv * (head_dim/2)) uint8_t qs + uint16_t d[kv_cap * n_kv * (head_dim/32)]
template <KvCacheType KV_TYPE>
__global__ void g4_vnorm_kv_write_kernel(
    const float *   __restrict__ k_in,     // [T, n_kv, head_dim] (already normed + RoPE'd)
    const float *   __restrict__ v_in,     // [T, n_kv, head_dim] (raw)
    void *          __restrict__ k_cache,
    void *          __restrict__ v_cache,
    const int32_t * __restrict__ pos,      // [T]
    int n_kv, int head_dim, int kv_cap, float rms_eps) {

    const int kvh  = blockIdx.x;
    const int t    = blockIdx.y;
    const int p    = pos[t];
    const int slot = p % kv_cap;
    const size_t src_off = (static_cast<size_t>(t) * n_kv + kvh) * head_dim;
    const float * kr = k_in + src_off;
    const float * vr = v_in + src_off;

    // 1. Compute V RMS across head_dim
    float ss = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = vr[i];
        ss += v * v;
    }
    __shared__ float smem[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    ss = g4_warp_sum(ss);
    if (lane == 0) smem[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        tot = g4_warp_sum(tot);
        if (lane == 0) smem[0] = rsqrtf(tot / static_cast<float>(head_dim) + rms_eps);
    }
    __syncthreads();
    const float v_inv_rms = smem[0];

    if constexpr (KV_TYPE == KvCacheType::Q8_0) {
        const int n_blks  = head_dim >> 5;
        const int n_warps = blockDim.x >> 5;
        const size_t qs_total = static_cast<size_t>(kv_cap) * n_kv * head_dim;
        auto * k_qs = static_cast<int8_t *>(k_cache);
        auto * k_d  = reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(k_cache) + qs_total);
        auto * v_qs = static_cast<int8_t *>(v_cache);
        auto * v_d  = reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(v_cache) + qs_total);
        const size_t dst_blk_base = (static_cast<size_t>(slot) * n_kv + kvh) * n_blks;

        for (int b = warp; b < n_blks; b += n_warps) {
            const int d = b * 32 + lane;
            const size_t q_off = (dst_blk_base + b) * 32 + lane;
            {
                const float kv = kr[d];
                const float am = __shfl_sync(0xffffffffu, g4_warp_max(fabsf(kv)), 0);
                const float sc = am * (1.0f / 127.0f);
                const float isc = (sc > 0.0f) ? (127.0f / am) : 0.0f;
                if (lane == 0) {
                    __half dh = __float2half_rn(sc);
                    k_d[dst_blk_base + b] = *reinterpret_cast<uint16_t *>(&dh);
                }
                k_qs[q_off] = static_cast<int8_t>(__float2int_rn(kv * isc));
            }
            {
                const float vv = vr[d] * v_inv_rms;
                const float am = __shfl_sync(0xffffffffu, g4_warp_max(fabsf(vv)), 0);
                const float sc = am * (1.0f / 127.0f);
                const float isc = (sc > 0.0f) ? (127.0f / am) : 0.0f;
                if (lane == 0) {
                    __half dh = __float2half_rn(sc);
                    v_d[dst_blk_base + b] = *reinterpret_cast<uint16_t *>(&dh);
                }
                v_qs[q_off] = static_cast<int8_t>(__float2int_rn(vv * isc));
            }
        }
    } else if constexpr (KV_TYPE == KvCacheType::Q4_0) {
        const int n_blks  = head_dim >> 5;
        const int n_warps = blockDim.x >> 5;
        const size_t qs_total = static_cast<size_t>(kv_cap) * n_kv * (head_dim >> 1);
        auto * k_qs = static_cast<uint8_t *>(k_cache);
        auto * k_d  = reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(k_cache) + qs_total);
        auto * v_qs = static_cast<uint8_t *>(v_cache);
        auto * v_d  = reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(v_cache) + qs_total);
        const size_t dst_blk_base = (static_cast<size_t>(slot) * n_kv + kvh) * n_blks;

        for (int b = warp; b < n_blks; b += n_warps) {
            const int d = b * 32 + lane;
            {
                const float kv = kr[d];
                const float am = __shfl_sync(0xffffffffu, g4_warp_max(fabsf(kv)), 0);
                const float sc = am * (1.0f / 7.0f);
                const float isc = (am > 0.0f) ? (7.0f / am) : 0.0f;
                int q = __float2int_rn(kv * isc) + 8;
                q = (q < 0) ? 0 : ((q > 15) ? 15 : q);
                const int q_hi = __shfl_down_sync(0xffffffffu, q, 16);
                if (lane < 16) {
                    k_qs[(dst_blk_base + b) * 16 + lane] = static_cast<uint8_t>(q | (q_hi << 4));
                }
                if (lane == 0) {
                    __half dh = __float2half_rn(sc);
                    k_d[dst_blk_base + b] = *reinterpret_cast<uint16_t *>(&dh);
                }
            }
            {
                const float vv = vr[d] * v_inv_rms;
                const float am = __shfl_sync(0xffffffffu, g4_warp_max(fabsf(vv)), 0);
                const float sc = am * (1.0f / 7.0f);
                const float isc = (am > 0.0f) ? (7.0f / am) : 0.0f;
                int q = __float2int_rn(vv * isc) + 8;
                q = (q < 0) ? 0 : ((q > 15) ? 15 : q);
                const int q_hi = __shfl_down_sync(0xffffffffu, q, 16);
                if (lane < 16) {
                    v_qs[(dst_blk_base + b) * 16 + lane] = static_cast<uint8_t>(q | (q_hi << 4));
                }
                if (lane == 0) {
                    __half dh = __float2half_rn(sc);
                    v_d[dst_blk_base + b] = *reinterpret_cast<uint16_t *>(&dh);
                }
            }
        }
    } else {
        auto * kc = static_cast<__half *>(k_cache);
        auto * vc = static_cast<__half *>(v_cache);
        const size_t dst_off = (static_cast<size_t>(slot) * n_kv + kvh) * head_dim;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            kc[dst_off + i] = __float2half_rn(kr[i]);
            vc[dst_off + i] = __float2half_rn(vr[i] * v_inv_rms);
        }
    }
}

// ============================================================================
// 3. Tiled FlashAttention Prefill (head_dim = 256 SWA or 512 Full, scale = 1.0)
// ============================================================================

template <typename TDst>
__global__ void g4_q_pack_kernel(
    const float * __restrict__ q,
    TDst *        __restrict__ q_out,
    int T, int n_head, int n_kv, int head_dim) {

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t tot = static_cast<int64_t>(T) * n_head * head_dim;
    if (idx >= tot) return;
    const int d   = static_cast<int>(idx % head_dim);
    const int rem = static_cast<int>(idx / head_dim);
    const int h   = rem % n_head;
    const int t   = rem / n_head;
    const int gqa = n_head / n_kv;
    const int kvh = h / gqa;
    const int g   = h % gqa;
    const size_t dst = ((static_cast<size_t>(kvh) * T + t) * gqa + g) * head_dim + d;
    if constexpr (std::is_same_v<TDst, __half>) {
        q_out[dst] = __float2half_rn(q[idx]);
    } else {
        q_out[dst] = q[idx];
    }
}

__global__ void g4_init_flash_ml_kernel(float * __restrict__ d_m, float * __restrict__ d_l, int M_tot) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M_tot) {
        d_m[i] = -FLT_MAX;
        d_l[i] = 0.0f;
    }
}

template <KvCacheType KV_TYPE, typename TDst>
__global__ void g4_dequant_kv_tile_kernel(
    const void * __restrict__ k_cache,
    const void * __restrict__ v_cache,
    TDst *       __restrict__ k_out,
    TDst *       __restrict__ v_out,
    int k0, int kc, int n_kv, int head_dim, int kv_cap) {

    const int kvh  = blockIdx.x;
    const int kp   = blockIdx.y;
    const int slot = (k0 + kp) % kv_cap;
    const size_t dst_base = (static_cast<size_t>(kvh) * kc + kp) * head_dim;

    if constexpr (KV_TYPE == KvCacheType::Q8_0) {
        const int n_blks  = head_dim >> 5;
        const int lane    = threadIdx.x;
        const int warp    = threadIdx.y;
        const int n_warps = blockDim.y;
        const size_t qs_total = static_cast<size_t>(kv_cap) * n_kv * head_dim;
        const auto * k_qs = static_cast<const int8_t *>(k_cache);
        const auto * k_d  = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(k_cache) + qs_total);
        const auto * v_qs = static_cast<const int8_t *>(v_cache);
        const auto * v_d  = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(v_cache) + qs_total);
        const size_t b0   = (static_cast<size_t>(slot) * n_kv + kvh) * n_blks;

        for (int b = warp; b < n_blks; b += n_warps) {
            __half kd_h, vd_h;
            *reinterpret_cast<uint16_t *>(&kd_h) = __ldg(&k_d[b0 + b]);
            *reinterpret_cast<uint16_t *>(&vd_h) = __ldg(&v_d[b0 + b]);
            const size_t q_off = (b0 + b) * 32 + lane;
            const float kv = __half2float(kd_h) * static_cast<float>(__ldg(&k_qs[q_off]));
            const float vv = __half2float(vd_h) * static_cast<float>(__ldg(&v_qs[q_off]));
            const int d = b * 32 + lane;
            if constexpr (std::is_same_v<TDst, __half>) {
                k_out[dst_base + d] = __float2half_rn(kv);
                v_out[dst_base + d] = __float2half_rn(vv);
            } else {
                k_out[dst_base + d] = kv;
                v_out[dst_base + d] = vv;
            }
        }
    } else if constexpr (KV_TYPE == KvCacheType::Q4_0) {
        const int n_blks  = head_dim >> 5;
        const int lane    = threadIdx.x;
        const int warp    = threadIdx.y;
        const int n_warps = blockDim.y;
        const int j       = lane & 15;
        const int hi      = lane >> 4;
        const size_t qs_total = static_cast<size_t>(kv_cap) * n_kv * (head_dim >> 1);
        const auto * k_qs = static_cast<const uint8_t *>(k_cache);
        const auto * k_d  = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(k_cache) + qs_total);
        const auto * v_qs = static_cast<const uint8_t *>(v_cache);
        const auto * v_d  = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(v_cache) + qs_total);
        const size_t b0   = (static_cast<size_t>(slot) * n_kv + kvh) * n_blks;

        for (int b = warp; b < n_blks; b += n_warps) {
            __half kd_h, vd_h;
            *reinterpret_cast<uint16_t *>(&kd_h) = __ldg(&k_d[b0 + b]);
            *reinterpret_cast<uint16_t *>(&vd_h) = __ldg(&v_d[b0 + b]);
            const uint8_t kb = __ldg(&k_qs[(b0 + b) * 16 + j]);
            const uint8_t vb = __ldg(&v_qs[(b0 + b) * 16 + j]);
            const int kq = static_cast<int>(hi ? (kb >> 4) : (kb & 0x0Fu)) - 8;
            const int vq = static_cast<int>(hi ? (vb >> 4) : (vb & 0x0Fu)) - 8;
            const float kv = __half2float(kd_h) * static_cast<float>(kq);
            const float vv = __half2float(vd_h) * static_cast<float>(vq);
            const int d = b * 32 + lane;
            if constexpr (std::is_same_v<TDst, __half>) {
                k_out[dst_base + d] = __float2half_rn(kv);
                v_out[dst_base + d] = __float2half_rn(vv);
            } else {
                k_out[dst_base + d] = kv;
                v_out[dst_base + d] = vv;
            }
        }
    } else {
        const auto * kc_h = static_cast<const __half *>(k_cache);
        const auto * vc_h = static_cast<const __half *>(v_cache);
        const size_t src  = (static_cast<size_t>(slot) * n_kv + kvh) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const __half kh = __ldg(&kc_h[src + d]);
            const __half vh = __ldg(&vc_h[src + d]);
            if constexpr (std::is_same_v<TDst, __half>) {
                k_out[dst_base + d] = kh;
                v_out[dst_base + d] = vh;
            } else {
                k_out[dst_base + d] = __half2float(kh);
                v_out[dst_base + d] = __half2float(vh);
            }
        }
    }
}

template <typename TS, typename TP>
__global__ void g4_flash_softmax_swa_rescale_kernel(
    const TS *      __restrict__ s_tile, // [n_kv * tc * gqa, kc]
    TP *            __restrict__ p_tile, // [n_kv * tc * gqa, kc]
    const int32_t * __restrict__ pos,    // [T]
    float *         __restrict__ d_m,    // [n_kv * T * gqa]
    float *         __restrict__ d_l,    // [n_kv * T * gqa]
    float *         __restrict__ d_oacc, // [n_kv * T * gqa, head_dim]
    int T, int t_start, int tc, int n_head, int n_kv, int head_dim,
    int k0, int kc, int swa_window) {

    const int srow = blockIdx.x; // 0 .. n_kv * tc * gqa - 1
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int gqa  = n_head / n_kv;
    const int M_cur = tc * gqa;
    const int kvh  = srow / M_cur;
    const int rem  = srow - kvh * M_cur;
    const int t    = t_start + rem / gqa;
    const int g    = rem % gqa;
    const int row  = (kvh * T + t) * gqa + g;
    const int q_pos = pos[t];
    const int k_min = (swa_window > 0) ? (q_pos - swa_window + 1) : 0;

    const TS * s_row = s_tile + static_cast<size_t>(srow) * kc;
    TP *       p_row = p_tile + static_cast<size_t>(srow) * kc;

    float tmax = -FLT_MAX;
    for (int j = tid; j < kc; j += blockDim.x) {
        const int kp = k0 + j;
        if (kp >= k_min && kp <= q_pos) {
            const float v = std::is_same_v<TS, __half> ? __half2float(s_row[j]) : static_cast<float>(s_row[j]);
            tmax = fmaxf(tmax, v);
        }
    }
    __shared__ float s_red[8];
    tmax = g4_warp_max(tmax);
    if (lane == 0) s_red[warp] = tmax;
    __syncthreads();
    if (warp == 0) {
        float mv = (lane < 8) ? s_red[lane] : -FLT_MAX;
        mv = g4_warp_max(mv);
        if (lane == 0) s_red[0] = mv;
    }
    __syncthreads();
    const float tile_max = s_red[0];

    const float m_old = d_m[row];
    const float l_old = d_l[row];

    if (tile_max == -FLT_MAX) {
        for (int j = tid; j < kc; j += blockDim.x) {
            if constexpr (std::is_same_v<TP, __half>) p_row[j] = __float2half_rn(0.0f);
            else                                       p_row[j] = 0.0f;
        }
        return;
    }

    const float m_new = fmaxf(m_old, tile_max);
    const float alpha = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);

    float lsum = 0.0f;
    for (int j = tid; j < kc; j += blockDim.x) {
        const int kp = k0 + j;
        float e = 0.0f;
        if (kp >= k_min && kp <= q_pos) {
            const float v = std::is_same_v<TS, __half> ? __half2float(s_row[j]) : static_cast<float>(s_row[j]);
            e = expf(v - m_new);
        }
        lsum += e;
        if constexpr (std::is_same_v<TP, __half>) p_row[j] = __float2half_rn(e);
        else                                       p_row[j] = e;
    }
    lsum = g4_warp_sum(lsum);
    if (lane == 0) s_red[warp] = lsum;
    __syncthreads();
    if (warp == 0) {
        float lv = (lane < 8) ? s_red[lane] : 0.0f;
        lv = g4_warp_sum(lv);
        if (lane == 0) s_red[0] = lv;
    }
    __syncthreads();
    if (tid == 0) {
        d_m[row] = m_new;
        d_l[row] = l_old * alpha + s_red[0];
    }

    if (m_old != -FLT_MAX && alpha != 1.0f) {
        float * orow = d_oacc + static_cast<size_t>(row) * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) orow[d] *= alpha;
    }
}

__global__ void g4_accum_f16_slice_to_f32_kernel(
    const __half * __restrict__ src,   // [n_kv, tc * gqa, head_dim]
    float *        __restrict__ dst,   // [n_kv, T  * gqa, head_dim]
    int T, int t_start, int tc, int gqa, int n_kv, int head_dim) {

    const int M_cur = tc * gqa;
    const int64_t tot2 = (static_cast<int64_t>(n_kv) * M_cur * head_dim) >> 1;
    const int64_t i2 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i2 >= tot2) return;
    const int hd2 = head_dim >> 1;
    const int d2  = static_cast<int>(i2 % hd2);
    const int srow = static_cast<int>(i2 / hd2);
    const int kvh = srow / M_cur;
    const int rem = srow - kvh * M_cur;
    const int drow = kvh * (T * gqa) + t_start * gqa + rem;

    const float2 a = __half22float2(reinterpret_cast<const __half2 *>(src)[i2]);
    float2 b = reinterpret_cast<float2 *>(dst + static_cast<size_t>(drow) * head_dim)[d2];
    b.x += a.x; b.y += a.y;
    reinterpret_cast<float2 *>(dst + static_cast<size_t>(drow) * head_dim)[d2] = b;
}

__global__ void g4_flash_finalize_unpack_kernel(
    const float * __restrict__ d_oacc, // [n_kv, T, gqa, head_dim]
    const float * __restrict__ d_l,    // [n_kv * T * gqa]
    float *       __restrict__ out,    // [T, n_head, head_dim]
    int T, int n_head, int n_kv, int head_dim) {

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t tot = static_cast<int64_t>(T) * n_head * head_dim;
    if (idx >= tot) return;
    const int d   = static_cast<int>(idx % head_dim);
    const int rem = static_cast<int>(idx / head_dim);
    const int h   = rem % n_head;
    const int t   = rem / n_head;
    const int gqa = n_head / n_kv;
    const int kvh = h / gqa;
    const int g   = h % gqa;
    const size_t row = (static_cast<size_t>(kvh) * T + t) * gqa + g;
    const float lval = d_l[row];
    const float inv_l = (lval > 0.0f) ? (1.0f / lval) : 0.0f;
    out[idx] = d_oacc[row * head_dim + d] * inv_l;
}

// ============================================================================
// 4. Coalesced Shared-KV Split-K MQA/GQA FlashDecode (head_dim = 256 or 512!)
//    1 CTA loads P_TILE=8 KV positions ONCE into smem for ALL gqa (8) Q heads!
// ============================================================================

template <int HEAD_DIM, int SPLITS>
__global__ __launch_bounds__(256) void g4_mqa_q8_decode_kernel(
    const float *   __restrict__ q_in,     // [T, n_head, HEAD_DIM]
    const void *    __restrict__ k_cache,  // SoA Q8_0: [max_ctx, n_kv, HEAD_DIM] int8 + [max_ctx, n_kv, NB] fp16
    const void *    __restrict__ v_cache,
    const int32_t * __restrict__ pos,      // [T]
    float *         __restrict__ part_m,   // [T, n_head, SPLITS]
    float *         __restrict__ part_l,   // [T, n_head, SPLITS]
    float *         __restrict__ part_o,   // [T, n_head, SPLITS, HEAD_DIM]
    int n_head, int n_kv, int kv_cap, int swa_window) {

    constexpr int NB        = HEAD_DIM / 32; // 8 (D=256) or 16 (D=512)
    constexpr int K_ITER    = NB / 4;        // 2 (D=256) or 4 (D=512)
    constexpr int OV_ELEMS  = HEAD_DIM / 32; // 8 (D=256) or 16 (D=512)
    constexpr int H_PER_BLK = 8;
    constexpr int P_TILE    = 8;

    __shared__ int32_t s_k_qs[P_TILE][NB][8];
    __shared__ float   s_k_d [P_TILE][NB];
    __shared__ int32_t s_v_qs[P_TILE][NB][8];
    __shared__ float   s_v_d [P_TILE][NB];

    const int gqa   = n_head / n_kv;
    const int kvh   = blockIdx.x / ((gqa + H_PER_BLK - 1) / H_PER_BLK);
    const int g_blk = blockIdx.x % ((gqa + H_PER_BLK - 1) / H_PER_BLK);
    const int s     = blockIdx.y;
    const int t     = blockIdx.z;

    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..7
    const int tid  = warp * 32 + lane;
    const int g    = g_blk * H_PER_BLK + warp;
    const bool valid_h = (kvh < n_kv) && (g < gqa);
    const int h    = kvh * gqa + g;

    const int q_pos  = pos[t];
    const int p_min  = (swa_window > 0 && q_pos - swa_window + 1 > 0) ? (q_pos - swa_window + 1) : 0;
    const int p_end  = q_pos + 1;
    const int span   = p_end - p_min;
    const int chunk  = (span + SPLITS - 1) / SPLITS;
    const int p0     = p_min + s * chunk;
    const int p1     = (p0 + chunk < p_end) ? (p0 + chunk) : p_end;

    const int grp = lane >> 3; // 0..3
    const int sub = lane & 7;  // 0..7

    int32_t q_i32[K_ITER];
    float   q_d  [K_ITER];
    #pragma unroll
    for (int k = 0; k < K_ITER; ++k) { q_i32[k] = 0; q_d[k] = 0.0f; }

    if (valid_h && p0 < p1) {
        const float * qp = q_in + (static_cast<size_t>(t) * n_head + h) * HEAD_DIM;
        #pragma unroll
        for (int k = 0; k < K_ITER; ++k) {
            const int b = grp + 4 * k;
            const float4 qv = __ldg(reinterpret_cast<const float4 *>(qp + b * 32 + sub * 4));
            float am = fmaxf(fmaxf(fabsf(qv.x), fabsf(qv.y)), fmaxf(fabsf(qv.z), fabsf(qv.w)));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
            const float d     = am * (1.0f / 127.0f);
            const float inv_d = (d > 0.0f) ? (127.0f / am) : 0.0f;
            q_d[k] = d;
            const uint32_t b0 = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(qv.x * inv_d)));
            const uint32_t b1 = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(qv.y * inv_d)));
            const uint32_t b2 = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(qv.z * inv_d)));
            const uint32_t b3 = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(qv.w * inv_d)));
            q_i32[k] = static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
        }
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float ov[OV_ELEMS];
    #pragma unroll
    for (int i = 0; i < OV_ELEMS; ++i) ov[i] = 0.0f;

    const size_t qs_total = static_cast<size_t>(kv_cap) * n_kv * HEAD_DIM;
    const auto * k_qs4 = static_cast<const int4 *>(k_cache);
    const auto * k_d   = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(k_cache) + qs_total);
    const auto * v_qs4 = static_cast<const int4 *>(v_cache);
    const auto * v_d   = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(v_cache) + qs_total);
    int4 * sk_i4 = reinterpret_cast<int4 *>(s_k_qs);
    int4 * sv_i4 = reinterpret_cast<int4 *>(s_v_qs);

    for (int p = p0; p < p1; p += P_TILE) {
        const int n_p    = (p + P_TILE < p1) ? P_TILE : (p1 - p);
        const int tot_i4 = n_p * (NB * 2); // 100% coalesced 128-bit LDG.128!
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c    = idx & (NB * 2 - 1);
            const int pi   = idx / (NB * 2);
            const int slot = (p + pi) % kv_cap;
            const size_t row_i4 = (static_cast<size_t>(slot) * n_kv + kvh) * (NB * 2) + c;
            sk_i4[idx] = __ldg(&k_qs4[row_i4]);
            sv_i4[idx] = __ldg(&v_qs4[row_i4]);
        }
        const int tot_d = n_p * NB;
        if (tid < tot_d) {
            const int b    = tid & (NB - 1);
            const int pi   = tid / NB;
            const int slot = (p + pi) % kv_cap;
            const size_t bidx = (static_cast<size_t>(slot) * n_kv + kvh) * NB + b;
            __half kdh, vdh;
            *reinterpret_cast<uint16_t *>(&kdh) = __ldg(&k_d[bidx]);
            *reinterpret_cast<uint16_t *>(&vdh) = __ldg(&v_d[bidx]);
            s_k_d[pi][b] = __half2float(kdh);
            s_v_d[pi][b] = __half2float(vdh);
        }
        __syncthreads();

        if (valid_h) {
            float scores[P_TILE];
            float m_tile = -FLT_MAX;
            #pragma unroll
            for (int pi = 0; pi < P_TILE; ++pi) {
                if (pi < n_p) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < K_ITER; ++k) {
                        const int b = grp + 4 * k;
                        const int dp = dp4a_s32(q_i32[k], s_k_qs[pi][b][sub], 0);
                        dot += (q_d[k] * s_k_d[pi][b]) * static_cast<float>(dp);
                    }
                    dot = g4_warp_sum(dot);
                    const float sc = __shfl_sync(0xffffffffu, dot, 0);
                    scores[pi] = sc;
                    m_tile = fmaxf(m_tile, sc);
                } else {
                    scores[pi] = -FLT_MAX;
                }
            }

            if (m_tile > m_i - 20.0f) {
                const float m_new = fmaxf(m_i, m_tile);
                const float alpha = (m_i == -FLT_MAX) ? 0.0f : expf(m_i - m_new);
                l_i *= alpha;
                #pragma unroll
                for (int i = 0; i < OV_ELEMS; ++i) ov[i] *= alpha;

                #pragma unroll
                for (int pi = 0; pi < P_TILE; ++pi) {
                    const float sc = scores[pi];
                    if (sc > m_new - 18.0f) {
                        const float beta = expf(sc - m_new);
                        l_i += beta;
                        #pragma unroll
                        for (int k = 0; k < K_ITER; ++k) {
                            const int b = grp + 4 * k;
                            const int32_t v4 = s_v_qs[pi][b][sub];
                            const float c = beta * s_v_d[pi][b];
                            ov[k * 4 + 0] += c * static_cast<float>(static_cast<int8_t>(v4));
                            ov[k * 4 + 1] += c * static_cast<float>(static_cast<int8_t>(v4 >> 8));
                            ov[k * 4 + 2] += c * static_cast<float>(static_cast<int8_t>(v4 >> 16));
                            ov[k * 4 + 3] += c * static_cast<float>(static_cast<int8_t>(v4 >> 24));
                        }
                    }
                }
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (valid_h) {
        const size_t slot = (static_cast<size_t>(t) * n_head + h) * SPLITS + s;
        if (lane == 0) {
            part_m[slot] = m_i;
            part_l[slot] = l_i;
        }
        float * op = part_o + slot * HEAD_DIM;
        #pragma unroll
        for (int k = 0; k < K_ITER; ++k) {
            const int b = grp + 4 * k;
            *reinterpret_cast<float4 *>(op + b * 32 + sub * 4) =
                make_float4(ov[k * 4 + 0], ov[k * 4 + 1], ov[k * 4 + 2], ov[k * 4 + 3]);
        }
    }
}

template <int HEAD_DIM, int SPLITS>
__global__ __launch_bounds__(256) void g4_mqa_q4_decode_kernel(
    const float *   __restrict__ q_in,
    const void *    __restrict__ k_cache, // SoA Q4_0: [kv_cap, n_kv, HEAD_DIM/2] u8 + [kv_cap, n_kv, NB] fp16
    const void *    __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    float *         __restrict__ part_m,
    float *         __restrict__ part_l,
    float *         __restrict__ part_o,
    int n_head, int n_kv, int kv_cap, int swa_window) {

    constexpr int NB        = HEAD_DIM / 32; // 8 (D=256) or 16 (D=512)
    constexpr int K_ITER    = NB / 4;        // 2 (D=256) or 4 (D=512)
    constexpr int OV_ELEMS  = HEAD_DIM / 32; // 8 (D=256) or 16 (D=512)
    constexpr int H_PER_BLK = 8;
    constexpr int P_TILE    = 8;

    __shared__ uint32_t s_k_qs[P_TILE][NB][4];
    __shared__ float    s_k_d [P_TILE][NB];
    __shared__ uint32_t s_v_qs[P_TILE][NB][4];
    __shared__ float    s_v_d [P_TILE][NB];

    const int gqa   = n_head / n_kv;
    const int kvh   = blockIdx.x / ((gqa + H_PER_BLK - 1) / H_PER_BLK);
    const int g_blk = blockIdx.x % ((gqa + H_PER_BLK - 1) / H_PER_BLK);
    const int s     = blockIdx.y;
    const int t     = blockIdx.z;

    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..7
    const int tid  = warp * 32 + lane;
    const int g    = g_blk * H_PER_BLK + warp;
    const bool valid_h = (kvh < n_kv) && (g < gqa);
    const int h    = kvh * gqa + g;

    const int q_pos  = pos[t];
    const int p_min  = (swa_window > 0 && q_pos - swa_window + 1 > 0) ? (q_pos - swa_window + 1) : 0;
    const int p_end  = q_pos + 1;
    const int span   = p_end - p_min;
    const int chunk  = (span + SPLITS - 1) / SPLITS;
    const int p0     = p_min + s * chunk;
    const int p1     = (p0 + chunk < p_end) ? (p0 + chunk) : p_end;

    const int grp = lane >> 3; // 0..3
    const int sub = lane & 7;  // 0..7 (0..3 = low nibbles 0..15, 4..7 = high nibbles 16..31)
    const int w_idx = sub & 3;
    const int shift = (sub >> 2) * 4;

    float q_f[OV_ELEMS];
    #pragma unroll
    for (int i = 0; i < OV_ELEMS; ++i) q_f[i] = 0.0f;
    if (valid_h && p0 < p1) {
        const float * qp = q_in + (static_cast<size_t>(t) * n_head + h) * HEAD_DIM;
        #pragma unroll
        for (int k = 0; k < K_ITER; ++k) {
            const int b = grp + 4 * k;
            const float4 qv = __ldg(reinterpret_cast<const float4 *>(qp + b * 32 + sub * 4));
            q_f[k * 4 + 0] = qv.x;
            q_f[k * 4 + 1] = qv.y;
            q_f[k * 4 + 2] = qv.z;
            q_f[k * 4 + 3] = qv.w;
        }
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float ov[OV_ELEMS];
    #pragma unroll
    for (int i = 0; i < OV_ELEMS; ++i) ov[i] = 0.0f;

    const size_t qs_total = static_cast<size_t>(kv_cap) * n_kv * (HEAD_DIM >> 1);
    const auto * k_qs4 = static_cast<const int4 *>(k_cache);
    const auto * k_d   = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(k_cache) + qs_total);
    const auto * v_qs4 = static_cast<const int4 *>(v_cache);
    const auto * v_d   = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(v_cache) + qs_total);
    int4 * sk_i4 = reinterpret_cast<int4 *>(s_k_qs);
    int4 * sv_i4 = reinterpret_cast<int4 *>(s_v_qs);

    for (int p = p0; p < p1; p += P_TILE) {
        const int n_p    = (p + P_TILE < p1) ? P_TILE : (p1 - p);
        const int tot_i4 = n_p * NB; // 1 int4 (16B) per Q4_0 block!
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int b    = idx & (NB - 1);
            const int pi   = idx / NB;
            const int slot = (p + pi) % kv_cap;
            const size_t bidx = (static_cast<size_t>(slot) * n_kv + kvh) * NB + b;
            sk_i4[idx] = __ldg(&k_qs4[bidx]);
            sv_i4[idx] = __ldg(&v_qs4[bidx]);
            __half kdh, vdh;
            *reinterpret_cast<uint16_t *>(&kdh) = __ldg(&k_d[bidx]);
            *reinterpret_cast<uint16_t *>(&vdh) = __ldg(&v_d[bidx]);
            s_k_d[pi][b] = __half2float(kdh);
            s_v_d[pi][b] = __half2float(vdh);
        }
        __syncthreads();

        if (valid_h) {
            float scores[P_TILE];
            float m_tile = -FLT_MAX;
            #pragma unroll
            for (int pi = 0; pi < P_TILE; ++pi) {
                if (pi < n_p) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < K_ITER; ++k) {
                        const int b = grp + 4 * k;
                        const uint32_t w = s_k_qs[pi][b][w_idx] >> shift;
                        const float d0 = static_cast<float>(static_cast<int>(w & 0xFu) - 8);
                        const float d1 = static_cast<float>(static_cast<int>((w >> 8) & 0xFu) - 8);
                        const float d2 = static_cast<float>(static_cast<int>((w >> 16) & 0xFu) - 8);
                        const float d3 = static_cast<float>(static_cast<int>((w >> 24) & 0xFu) - 8);
                        dot += s_k_d[pi][b] * (d0 * q_f[k * 4 + 0] + d1 * q_f[k * 4 + 1] +
                                               d2 * q_f[k * 4 + 2] + d3 * q_f[k * 4 + 3]);
                    }
                    dot = g4_warp_sum(dot);
                    const float sc = __shfl_sync(0xffffffffu, dot, 0);
                    scores[pi] = sc;
                    m_tile = fmaxf(m_tile, sc);
                } else {
                    scores[pi] = -FLT_MAX;
                }
            }

            if (m_tile > m_i - 20.0f) {
                const float m_new = fmaxf(m_i, m_tile);
                const float alpha = (m_i == -FLT_MAX) ? 0.0f : expf(m_i - m_new);
                l_i *= alpha;
                #pragma unroll
                for (int i = 0; i < OV_ELEMS; ++i) ov[i] *= alpha;

                #pragma unroll
                for (int pi = 0; pi < P_TILE; ++pi) {
                    const float sc = scores[pi];
                    if (sc > m_new - 18.0f) {
                        const float beta = expf(sc - m_new);
                        l_i += beta;
                        #pragma unroll
                        for (int k = 0; k < K_ITER; ++k) {
                            const int b = grp + 4 * k;
                            const uint32_t w = s_v_qs[pi][b][w_idx] >> shift;
                            const float c = beta * s_v_d[pi][b];
                            ov[k * 4 + 0] += c * static_cast<float>(static_cast<int>(w & 0xFu) - 8);
                            ov[k * 4 + 1] += c * static_cast<float>(static_cast<int>((w >> 8) & 0xFu) - 8);
                            ov[k * 4 + 2] += c * static_cast<float>(static_cast<int>((w >> 16) & 0xFu) - 8);
                            ov[k * 4 + 3] += c * static_cast<float>(static_cast<int>((w >> 24) & 0xFu) - 8);
                        }
                    }
                }
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (valid_h) {
        const size_t slot = (static_cast<size_t>(t) * n_head + h) * SPLITS + s;
        if (lane == 0) {
            part_m[slot] = m_i;
            part_l[slot] = l_i;
        }
        float * op = part_o + slot * HEAD_DIM;
        #pragma unroll
        for (int k = 0; k < K_ITER; ++k) {
            const int b = grp + 4 * k;
            *reinterpret_cast<float4 *>(op + b * 32 + sub * 4) =
                make_float4(ov[k * 4 + 0], ov[k * 4 + 1], ov[k * 4 + 2], ov[k * 4 + 3]);
        }
    }
}

template <int HEAD_DIM, int SPLITS>
__global__ __launch_bounds__(256) void g4_mqa_f16_decode_kernel(
    const float *   __restrict__ q_in,
    const __half *  __restrict__ k_cache,
    const __half *  __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    float *         __restrict__ part_m,
    float *         __restrict__ part_l,
    float *         __restrict__ part_o,
    int n_head, int n_kv, int kv_cap, int swa_window) {

    constexpr int H_PER_BLK = 8;
    constexpr int P_TILE    = 8;
    constexpr int ELEMS_PER_LANE = HEAD_DIM / 32; // 8 or 16

    __shared__ __half s_k[P_TILE][HEAD_DIM];
    __shared__ __half s_v[P_TILE][HEAD_DIM];

    const int gqa   = n_head / n_kv;
    const int kvh   = blockIdx.x / ((gqa + H_PER_BLK - 1) / H_PER_BLK);
    const int g_blk = blockIdx.x % ((gqa + H_PER_BLK - 1) / H_PER_BLK);
    const int s     = blockIdx.y;
    const int t     = blockIdx.z;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int g    = g_blk * H_PER_BLK + warp;
    const bool valid_h = (kvh < n_kv) && (g < gqa);
    const int h    = kvh * gqa + g;

    const int q_pos = pos[t];
    const int p_min = (swa_window > 0 && q_pos - swa_window + 1 > 0) ? (q_pos - swa_window + 1) : 0;
    const int p_end = q_pos + 1;
    const int span  = p_end - p_min;
    const int chunk = (span + SPLITS - 1) / SPLITS;
    const int p0    = p_min + s * chunk;
    const int p1    = (p0 + chunk < p_end) ? (p0 + chunk) : p_end;

    float qv[ELEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) qv[i] = 0.0f;
    if (valid_h && p0 < p1) {
        const float * qp = q_in + (static_cast<size_t>(t) * n_head + h) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; ++i) {
            qv[i] = __ldg(&qp[i * 32 + lane]);
        }
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float ov[ELEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) ov[i] = 0.0f;

    for (int p = p0; p < p1; p += P_TILE) {
        const int n_p    = (p + P_TILE < p1) ? P_TILE : (p1 - p);
        const int tot_i4 = n_p * (HEAD_DIM / 8); // 8 halfs per int4
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c    = idx % (HEAD_DIM / 8);
            const int pi   = idx / (HEAD_DIM / 8);
            const int slot = (p + pi) % kv_cap;
            const size_t base = (static_cast<size_t>(slot) * n_kv + kvh) * HEAD_DIM;
            reinterpret_cast<int4 *>(s_k[pi])[c] = __ldg(&reinterpret_cast<const int4 *>(k_cache + base)[c]);
            reinterpret_cast<int4 *>(s_v[pi])[c] = __ldg(&reinterpret_cast<const int4 *>(v_cache + base)[c]);
        }
        __syncthreads();

        if (valid_h) {
            for (int pi = 0; pi < n_p; ++pi) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < ELEMS_PER_LANE; ++i) {
                    dot += qv[i] * __half2float(s_k[pi][i * 32 + lane]);
                }
                dot = g4_warp_sum(dot);
                const float sc = __shfl_sync(0xffffffffu, dot, 0);
                const float m_new = fmaxf(m_i, sc);
                const float alpha = (m_i == -FLT_MAX) ? 0.0f : expf(m_i - m_new);
                const float beta  = expf(sc - m_new);
                l_i = l_i * alpha + beta;
                #pragma unroll
                for (int i = 0; i < ELEMS_PER_LANE; ++i) {
                    ov[i] = ov[i] * alpha + beta * __half2float(s_v[pi][i * 32 + lane]);
                }
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (valid_h) {
        const size_t slot = (static_cast<size_t>(t) * n_head + h) * SPLITS + s;
        if (lane == 0) {
            part_m[slot] = m_i;
            part_l[slot] = l_i;
        }
        float * op = part_o + slot * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; ++i) {
            op[i * 32 + lane] = ov[i];
        }
    }
}

template <int SPLITS>
__global__ void g4_splitk_reduce_kernel(
    const float * __restrict__ part_m, // [T, n_head, SPLITS]
    const float * __restrict__ part_l, // [T, n_head, SPLITS]
    const float * __restrict__ part_o, // [T, n_head, SPLITS, head_dim]
    float *       __restrict__ out,    // [T, n_head, head_dim]
    int n_head, int head_dim) {

    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const size_t base = (static_cast<size_t>(t) * n_head + h) * SPLITS;

    __shared__ float s_w[SPLITS];
    __shared__ float s_inv_l;

    if (threadIdx.x == 0) {
        float m_max = -FLT_MAX;
        for (int s = 0; s < SPLITS; ++s) {
            const float ms = part_m[base + s];
            if (ms > m_max) m_max = ms;
        }
        float l_tot = 0.0f;
        for (int s = 0; s < SPLITS; ++s) {
            const float ms = part_m[base + s];
            const float ls = part_l[base + s];
            const float w  = (ms > -FLT_MAX * 0.5f && ls > 0.0f) ? expf(ms - m_max) : 0.0f;
            s_w[s] = w;
            l_tot += w * ls;
        }
        s_inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    }
    __syncthreads();

    const float inv_l = s_inv_l;
    float * y = out + (static_cast<size_t>(t) * n_head + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < SPLITS; ++s) {
            const float w = s_w[s];
            if (w > 0.0f) {
                acc += w * part_o[(base + s) * head_dim + d];
            }
        }
        y[d] = acc * inv_l;
    }
}

// ============================================================================
// 5. Fast sm_60 FP16 cublasHgemm Prefill for Q6_K SoA & F16 Weights
// ============================================================================

__global__ void g4_dequant_q6_k_soa_f16_kernel(
    const uint8_t *  __restrict__ ql,
    const uint8_t *  __restrict__ qh,
    const int8_t *   __restrict__ scales,
    const uint16_t * __restrict__ d_arr,
    __half *         __restrict__ dst,
    int64_t n_blocks) {

    const int64_t bi = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (bi >= n_blocks) return;
    const int l  = threadIdx.x;
    const int is = l >> 4;
    __half dh;
    *reinterpret_cast<uint16_t *>(&dh) = __ldg(&d_arr[bi]);
    const float dw = __half2float(dh);
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

        __half * out = dst + bi * 256 + h * 128;
        out[0  + l] = __float2half_rn(sc0 * static_cast<float>(q1));
        out[32 + l] = __float2half_rn(sc2 * static_cast<float>(q2));
        out[64 + l] = __float2half_rn(sc4 * static_cast<float>(q3));
        out[96 + l] = __float2half_rn(sc6 * static_cast<float>(q4));
    }
}

__global__ void g4_f16_to_f32_vec_kernel(const __half * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t i2 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t i  = i2 * 2;
    if (i + 1 < n) {
        reinterpret_cast<float2 *>(dst)[i2] = __half22float2(reinterpret_cast<const __half2 *>(src)[i2]);
    } else if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

// ============================================================================
// 6. High-Throughput SoA Q4_0 Weight Kernels (Embedding, Decode GEMV, Prefill GEMM)
//    - sm_60 (P100): IEEE-754 0x6400 nibble->half2 + block-normalized __hfma2 GEMV & cublasHgemm
//    - sm_61+ (GTX 1060 / Volta / Ampere): hardware dp4a_s32 Q4_0 x Q8_1 GEMV & Hgemm/Sgemm/TC
//    - sm_50/52 (Maxwell): clean FP32 GEMV & cublasSgemm
// ============================================================================

__global__ void g4_embedding_q4_0_soa_kernel(
    const uint8_t *  __restrict__ W_qs,
    const uint16_t * __restrict__ W_sc,
    const int32_t *  __restrict__ tokens,
    float *          __restrict__ out,
    int n_embd) {

    const int t = blockIdx.x;
    const int tok = tokens[t];
    const int nb_row = n_embd >> 5;
    const size_t row_b0 = static_cast<size_t>(tok) * nb_row;
    float * dst = out + static_cast<size_t>(t) * n_embd;

    const int lane    = threadIdx.x & 31;
    const int warp    = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;
    const int j       = lane & 15;
    const int hi      = lane >> 4;

    for (int b = warp; b < nb_row; b += n_warps) {
        __half dh;
        *reinterpret_cast<uint16_t *>(&dh) = __ldg(&W_sc[row_b0 + b]);
        const uint8_t qbyte = __ldg(&W_qs[(row_b0 + b) * 16 + j]);
        const int q = static_cast<int>(hi ? (qbyte >> 4) : (qbyte & 0x0Fu)) - 8;
        dst[b * 32 + lane] = __half2float(dh) * static_cast<float>(q);
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
__device__ __forceinline__ __half2 g4_u32_to_h2(uint32_t u) {
    __half2 h;
    *reinterpret_cast<uint32_t *>(&h) = u;
    return h;
}
#endif

// Vectorized 128-bit LDG.128 / STG.128 SoA Q4_0 -> F16/F32 dequantizer (1 block = 32 elems per thread)
template <typename TDst>
__global__ void g4_dequant_q4_0_soa_kernel(
    const int4 *     __restrict__ W_qs4,
    const uint16_t * __restrict__ W_sc,
    TDst *           __restrict__ dst,
    int64_t n_blocks) {

    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    const int4 w4 = __ldg(&W_qs4[b]);
    __half dh;
    *reinterpret_cast<uint16_t *>(&dh) = __ldg(&W_sc[b]);

    if constexpr (std::is_same_v<TDst, __half>) {
#if __CUDA_ARCH__ >= 600
        const __half2 d2    = __half2half2(dh);
        const __half2 c1032 = __floats2half2_rn(1032.0f, 1032.0f);
        auto unpack4 = [&](uint32_t w, __half2 & lo01, __half2 & lo23, __half2 & hi01, __half2 & hi23) {
            const uint32_t lo = w & 0x0F0F0F0Fu;
            const uint32_t hi = (w >> 4) & 0x0F0F0F0Fu;
            lo01 = __hmul2(__hsub2(g4_u32_to_h2(__byte_perm(lo, 0x64u, 0x4140u)), c1032), d2);
            lo23 = __hmul2(__hsub2(g4_u32_to_h2(__byte_perm(lo, 0x64u, 0x4342u)), c1032), d2);
            hi01 = __hmul2(__hsub2(g4_u32_to_h2(__byte_perm(hi, 0x64u, 0x4140u)), c1032), d2);
            hi23 = __hmul2(__hsub2(g4_u32_to_h2(__byte_perm(hi, 0x64u, 0x4342u)), c1032), d2);
        };
        __half2 l0, l1, l2, l3, l4, l5, l6, l7;
        __half2 h0, h1, h2, h3, h4, h5, h6, h7;
        unpack4(static_cast<uint32_t>(w4.x), l0, l1, h0, h1);
        unpack4(static_cast<uint32_t>(w4.y), l2, l3, h2, h3);
        unpack4(static_cast<uint32_t>(w4.z), l4, l5, h4, h5);
        unpack4(static_cast<uint32_t>(w4.w), l6, l7, h6, h7);

        int4 o0, o1, o2, o3;
        auto * p0 = reinterpret_cast<__half2 *>(&o0); p0[0] = l0; p0[1] = l1; p0[2] = l2; p0[3] = l3;
        auto * p1 = reinterpret_cast<__half2 *>(&o1); p1[0] = l4; p1[1] = l5; p1[2] = l6; p1[3] = l7;
        auto * p2 = reinterpret_cast<__half2 *>(&o2); p2[0] = h0; p2[1] = h1; p2[2] = h2; p2[3] = h3;
        auto * p3 = reinterpret_cast<__half2 *>(&o3); p3[0] = h4; p3[1] = h5; p3[2] = h6; p3[3] = h7;
        auto * dst4 = reinterpret_cast<int4 *>(dst);
        dst4[b * 4 + 0] = o0;
        dst4[b * 4 + 1] = o1;
        dst4[b * 4 + 2] = o2;
        dst4[b * 4 + 3] = o3;
#else
        const float d = __half2float(dh);
        const uint8_t * qb = reinterpret_cast<const uint8_t *>(&w4);
        for (int j = 0; j < 16; ++j) {
            dst[b * 32 + j]      = __float2half_rn(d * static_cast<float>(static_cast<int>(qb[j] & 0x0Fu) - 8));
            dst[b * 32 + j + 16] = __float2half_rn(d * static_cast<float>(static_cast<int>(qb[j] >> 4) - 8));
        }
#endif
    } else {
        const float d = __half2float(dh);
        const uint32_t ws[4] = {
            static_cast<uint32_t>(w4.x), static_cast<uint32_t>(w4.y),
            static_cast<uint32_t>(w4.z), static_cast<uint32_t>(w4.w)
        };
        auto * dst4 = reinterpret_cast<float4 *>(dst + b * 32);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t w = ws[k];
            dst4[k] = make_float4(
                d * static_cast<float>(static_cast<int>(w & 0xFu) - 8),
                d * static_cast<float>(static_cast<int>((w >> 8) & 0xFu) - 8),
                d * static_cast<float>(static_cast<int>((w >> 16) & 0xFu) - 8),
                d * static_cast<float>(static_cast<int>((w >> 24) & 0xFu) - 8));
            dst4[k + 4] = make_float4(
                d * static_cast<float>(static_cast<int>((w >> 4) & 0xFu) - 8),
                d * static_cast<float>(static_cast<int>((w >> 12) & 0xFu) - 8),
                d * static_cast<float>(static_cast<int>((w >> 20) & 0xFu) - 8),
                d * static_cast<float>(static_cast<int>((w >> 28) & 0xFu) - 8));
        }
    }
}

// Pre-normalize activation X[T, K] ONCE into block-normalized __half2 [-1, 1] in stride-2 pairs
// (x[0],x[2]), (x[1],x[3]), ... + per-block float scale (16*am).
// Pairs with the exact zero-PRMT 0x3C003C00 nibble-to-FP16x2 unpack ((1.0 + q/16.0) - 1.5 = (q-8)/16.0).
__global__ void g4_quantize_x_h2_kernel(
    const float * __restrict__ X,
    __half2 *     __restrict__ X_h2, // [total_blocks, 16]
    float *       __restrict__ X_d,  // [total_blocks] = 16 * am
    int total_blocks) {

    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= total_blocks) return;
    const int lane = threadIdx.x;
    const float xv = __ldg(&X[static_cast<size_t>(b) * 32 + lane]);
    const float am = __shfl_sync(0xffffffffu, g4_warp_max(fabsf(xv)), 0);
    const float inv_am = (am > 0.0f) ? (1.0f / am) : 0.0f;
    const float xn = xv * inv_am;
    const float xn_p2 = __shfl_down_sync(0xffffffffu, xn, 2);
    if ((lane & 2) == 0) {
        const int slot = ((lane >> 2) << 1) | (lane & 1);
        X_h2[static_cast<size_t>(b) * 16 + slot] = __floats2half2_rn(xn, xn_p2);
    }
    if (lane == 0) {
        X_d[b] = am * 16.0f;
    }
}

// sm_60 (P100) Tiled-SMEM LDG.128/LDS.128 Zero-PRMT SoA Q4_0 GEMV (2 rows/warp!)
// - Tiles K_blocks in KB_TILE=128 chunks into 8.5 KB smem (int4 s_x[128][4], float s_d[128]).
// - 4x fewer smem load/store instructions via 128-bit int4 transfers; low register count (~40 regs).
// - Zero __byte_perm: unpacks nibbles directly to (q-8)/16.0 via SHL/SHR + LOP3 + exact __hsub2(1.5).
__global__ void g4_gemv_q4_0_soa_sm60_h2_kernel(
    const int4 *     __restrict__ W_qs, // [N, K_blocks] int4 (16B/block)
    const uint16_t * __restrict__ W_sc, // [N, K_blocks] fp16
    const int4 *     __restrict__ X_h2, // [T, K_blocks * 4] int4 (= 16 half2/block)
    const float *    __restrict__ X_d,  // [T, K_blocks] = 16 * am
    float *          __restrict__ Y,    // [T, N]
    int K_blocks, int N, int T) {

    constexpr int KB_TILE = 128;
    __shared__ int4  s_x[KB_TILE * 4];
    __shared__ float s_d[KB_TILE];

    const int t = blockIdx.y;
    if (t >= T) return;

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const int lane = threadIdx.x;
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    const int row1 = row0 + 1;
    const bool has_r0 = (row0 < N);
    const bool has_r1 = (row1 < N);

    const int4 *     w_q_r0 = W_qs + static_cast<size_t>(has_r0 ? row0 : 0) * K_blocks;
    const uint16_t * w_s_r0 = W_sc + static_cast<size_t>(has_r0 ? row0 : 0) * K_blocks;
    const int4 *     w_q_r1 = W_qs + static_cast<size_t>(has_r1 ? row1 : 0) * K_blocks;
    const uint16_t * w_s_r1 = W_sc + static_cast<size_t>(has_r1 ? row1 : 0) * K_blocks;
    const int4 *     x_h2_t = X_h2 + static_cast<size_t>(t) * K_blocks * 4;
    const float *    x_d_t  = X_d  + static_cast<size_t>(t) * K_blocks;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
#if __CUDA_ARCH__ >= 600
    const __half2 c1p5 = __floats2half2_rn(1.5f, 1.5f);
#endif

    for (int kb0 = 0; kb0 < K_blocks; kb0 += KB_TILE) {
        const int kb_n   = (kb0 + KB_TILE <= K_blocks) ? KB_TILE : (K_blocks - kb0);
        const int tot_i4 = kb_n * 4;
        const int4 * src_x = x_h2_t + kb0 * 4;
        for (int i = tid; i < tot_i4; i += nthr) {
            s_x[i] = __ldg(&src_x[i]);
        }
        for (int i = tid; i < kb_n; i += nthr) {
            s_d[i] = __ldg(&x_d_t[kb0 + i]);
        }
        __syncthreads();

        if (has_r0) {
            for (int tb = lane; tb < kb_n; tb += 32) {
                const int b = kb0 + tb;
                const float xd = s_d[tb];
                const int4 * sxb = s_x + tb * 4;

#if __CUDA_ARCH__ >= 600
                const int4 xl0 = sxb[0], xh0 = sxb[2];
                const int4 xl1 = sxb[1], xh1 = sxb[3];
                const auto * p_xl0 = reinterpret_cast<const __half2 *>(&xl0);
                const auto * p_xh0 = reinterpret_cast<const __half2 *>(&xh0);
                const auto * p_xl1 = reinterpret_cast<const __half2 *>(&xl1);
                const auto * p_xh1 = reinterpret_cast<const __half2 *>(&xh1);

                auto dot_blk = [&](const int4 & w4, uint16_t wd_bits) -> float {
                    __half wd_h;
                    *reinterpret_cast<uint16_t *>(&wd_h) = wd_bits;
                    __half2 hsum = __floats2half2_rn(0.0f, 0.0f);
                    auto mac4 = [&](uint32_t w, __half2 xa0, __half2 xa1, __half2 xb0, __half2 xb1) {
                        const uint32_t p0 = ((w << 6) & 0x03C003C0u) | 0x3C003C00u;
                        const uint32_t p1 = ((w >> 2) & 0x03C003C0u) | 0x3C003C00u;
                        const uint32_t p2 = ((w << 2) & 0x03C003C0u) | 0x3C003C00u;
                        const uint32_t p3 = ((w >> 6) & 0x03C003C0u) | 0x3C003C00u;
                        hsum = __hfma2(__hsub2(g4_u32_to_h2(p0), c1p5), xa0, hsum);
                        hsum = __hfma2(__hsub2(g4_u32_to_h2(p1), c1p5), xa1, hsum);
                        hsum = __hfma2(__hsub2(g4_u32_to_h2(p2), c1p5), xb0, hsum);
                        hsum = __hfma2(__hsub2(g4_u32_to_h2(p3), c1p5), xb1, hsum);
                    };
                    mac4(static_cast<uint32_t>(w4.x), p_xl0[0], p_xl0[1], p_xh0[0], p_xh0[1]);
                    mac4(static_cast<uint32_t>(w4.y), p_xl0[2], p_xl0[3], p_xh0[2], p_xh0[3]);
                    mac4(static_cast<uint32_t>(w4.z), p_xl1[0], p_xl1[1], p_xh1[0], p_xh1[1]);
                    mac4(static_cast<uint32_t>(w4.w), p_xl1[2], p_xl1[3], p_xh1[2], p_xh1[3]);
                    return (__half2float(wd_h) * xd) * (__low2float(hsum) + __high2float(hsum));
                };
#else
                const __half2 * sx = reinterpret_cast<const __half2 *>(sxb);
                auto dot_blk = [&](const int4 & w4, uint16_t wd_bits) -> float {
                    __half wd_h;
                    *reinterpret_cast<uint16_t *>(&wd_h) = wd_bits;
                    const uint32_t ws[4] = {
                        static_cast<uint32_t>(w4.x), static_cast<uint32_t>(w4.y),
                        static_cast<uint32_t>(w4.z), static_cast<uint32_t>(w4.w)
                    };
                    float bsum = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t w = ws[k];
                        const float2 xl0 = __half22float2(sx[k * 2 + 0]);
                        const float2 xl1 = __half22float2(sx[k * 2 + 1]);
                        const float2 xh0 = __half22float2(sx[k * 2 + 8]);
                        const float2 xh1 = __half22float2(sx[k * 2 + 9]);
                        bsum += (static_cast<float>((w >>  0) & 0xFu) - 8.0f) * 0.0625f * xl0.x;
                        bsum += (static_cast<float>((w >> 16) & 0xFu) - 8.0f) * 0.0625f * xl0.y;
                        bsum += (static_cast<float>((w >>  8) & 0xFu) - 8.0f) * 0.0625f * xl1.x;
                        bsum += (static_cast<float>((w >> 24) & 0xFu) - 8.0f) * 0.0625f * xl1.y;
                        bsum += (static_cast<float>((w >>  4) & 0xFu) - 8.0f) * 0.0625f * xh0.x;
                        bsum += (static_cast<float>((w >> 20) & 0xFu) - 8.0f) * 0.0625f * xh0.y;
                        bsum += (static_cast<float>((w >> 12) & 0xFu) - 8.0f) * 0.0625f * xh1.x;
                        bsum += (static_cast<float>((w >> 28) & 0xFu) - 8.0f) * 0.0625f * xh1.y;
                    }
                    return (__half2float(wd_h) * xd) * bsum;
                };
#endif
                acc0 += dot_blk(__ldg(&w_q_r0[b]), __ldg(&w_s_r0[b]));
                if (has_r1) acc1 += dot_blk(__ldg(&w_q_r1[b]), __ldg(&w_s_r1[b]));
            }
        }
        if (kb0 + KB_TILE < K_blocks) __syncthreads();
    }

    if (has_r0) {
        acc0 = g4_warp_sum(acc0);
        if (has_r1) acc1 = g4_warp_sum(acc1);
        if (lane == 0) {
            Y[static_cast<size_t>(t) * N + row0] = acc0;
            if (has_r1) Y[static_cast<size_t>(t) * N + row1] = acc1;
        }
    }
}

// Single-launch Multi-Matrix Q4_0 GEMV on sm_60 (Q+K+V in ONE launch, ZERO local-memory stack!)
__global__ void g4_gemv_multi_q4_0_soa_sm60_h2_kernel(
    const int4 *     __restrict__ W0_qs, const uint16_t * __restrict__ W0_sc, float * __restrict__ Y0, int M0,
    const int4 *     __restrict__ W1_qs, const uint16_t * __restrict__ W1_sc, float * __restrict__ Y1, int M1,
    const int4 *     __restrict__ W2_qs, const uint16_t * __restrict__ W2_sc, float * __restrict__ Y2, int M2,
    const int4 *     __restrict__ X_h2,
    const float *    __restrict__ X_d,
    int K_blocks, int T) {

    constexpr int KB_TILE = 128;
    __shared__ int4  s_x[KB_TILE * 4];
    __shared__ float s_d[KB_TILE];

    const int t = blockIdx.y;
    if (t >= T) return;

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const int lane = threadIdx.x;
    const int g_row0   = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    const int m01      = M0 + M1;
    const int tot_rows = m01 + M2;

    // Resolve matrix pointer via pure register ternary (0 bytes local stack!)
    const bool has_r0 = (g_row0 < tot_rows);
    const int4 *     w_q_r0 = W0_qs;
    const uint16_t * w_s_r0 = W0_sc;
    float *          y_r0   = Y0;
    int              m_r0   = M0;
    int              r0     = g_row0;
    if (g_row0 >= m01) {
        w_q_r0 = W2_qs; w_s_r0 = W2_sc; y_r0 = Y2; m_r0 = M2; r0 = g_row0 - m01;
    } else if (g_row0 >= M0) {
        w_q_r0 = W1_qs; w_s_r0 = W1_sc; y_r0 = Y1; m_r0 = M1; r0 = g_row0 - M0;
    }

    const int g_row1  = g_row0 + 1;
    const bool has_r1 = (g_row1 < tot_rows);
    const int4 *     w_q_r1 = W0_qs;
    const uint16_t * w_s_r1 = W0_sc;
    float *          y_r1   = Y0;
    int              m_r1   = M0;
    int              r1     = g_row1;
    if (g_row1 >= m01) {
        w_q_r1 = W2_qs; w_s_r1 = W2_sc; y_r1 = Y2; m_r1 = M2; r1 = g_row1 - m01;
    } else if (g_row1 >= M0) {
        w_q_r1 = W1_qs; w_s_r1 = W1_sc; y_r1 = Y1; m_r1 = M1; r1 = g_row1 - M0;
    }

    if (has_r0) { w_q_r0 += static_cast<size_t>(r0) * K_blocks; w_s_r0 += static_cast<size_t>(r0) * K_blocks; }
    if (has_r1) { w_q_r1 += static_cast<size_t>(r1) * K_blocks; w_s_r1 += static_cast<size_t>(r1) * K_blocks; }

    const int4 *  x_h2_t = X_h2 + static_cast<size_t>(t) * K_blocks * 4;
    const float * x_d_t  = X_d  + static_cast<size_t>(t) * K_blocks;

    float acc0 = 0.0f, acc1 = 0.0f;
#if __CUDA_ARCH__ >= 600
    const __half2 c1p5 = __floats2half2_rn(1.5f, 1.5f);
#endif

    for (int kb0 = 0; kb0 < K_blocks; kb0 += KB_TILE) {
        const int kb_n   = (kb0 + KB_TILE <= K_blocks) ? KB_TILE : (K_blocks - kb0);
        const int tot_i4 = kb_n * 4;
        const int4 * src_x = x_h2_t + kb0 * 4;
        for (int i = tid; i < tot_i4; i += nthr) {
            s_x[i] = __ldg(&src_x[i]);
        }
        for (int i = tid; i < kb_n; i += nthr) {
            s_d[i] = __ldg(&x_d_t[kb0 + i]);
        }
        __syncthreads();

        if (has_r0) {
            for (int tb = lane; tb < kb_n; tb += 32) {
                const int b = kb0 + tb;
                const float xd = s_d[tb];
                const int4 * sxb = s_x + tb * 4;

#if __CUDA_ARCH__ >= 600
                const int4 xl0 = sxb[0], xh0 = sxb[2];
                const int4 xl1 = sxb[1], xh1 = sxb[3];
                const auto * p_xl0 = reinterpret_cast<const __half2 *>(&xl0);
                const auto * p_xh0 = reinterpret_cast<const __half2 *>(&xh0);
                const auto * p_xl1 = reinterpret_cast<const __half2 *>(&xl1);
                const auto * p_xh1 = reinterpret_cast<const __half2 *>(&xh1);

                auto dot_blk = [&](const int4 & w4, uint16_t wd_bits) -> float {
                    __half wd_h;
                    *reinterpret_cast<uint16_t *>(&wd_h) = wd_bits;
                    __half2 hsum = __floats2half2_rn(0.0f, 0.0f);
                    auto mac4 = [&](uint32_t w, __half2 xa0, __half2 xa1, __half2 xb0, __half2 xb1) {
                        const uint32_t p0 = ((w << 6) & 0x03C003C0u) | 0x3C003C00u;
                        const uint32_t p1 = ((w >> 2) & 0x03C003C0u) | 0x3C003C00u;
                        const uint32_t p2 = ((w << 2) & 0x03C003C0u) | 0x3C003C00u;
                        const uint32_t p3 = ((w >> 6) & 0x03C003C0u) | 0x3C003C00u;
                        hsum = __hfma2(__hsub2(g4_u32_to_h2(p0), c1p5), xa0, hsum);
                        hsum = __hfma2(__hsub2(g4_u32_to_h2(p1), c1p5), xa1, hsum);
                        hsum = __hfma2(__hsub2(g4_u32_to_h2(p2), c1p5), xb0, hsum);
                        hsum = __hfma2(__hsub2(g4_u32_to_h2(p3), c1p5), xb1, hsum);
                    };
                    mac4(static_cast<uint32_t>(w4.x), p_xl0[0], p_xl0[1], p_xh0[0], p_xh0[1]);
                    mac4(static_cast<uint32_t>(w4.y), p_xl0[2], p_xl0[3], p_xh0[2], p_xh0[3]);
                    mac4(static_cast<uint32_t>(w4.z), p_xl1[0], p_xl1[1], p_xh1[0], p_xh1[1]);
                    mac4(static_cast<uint32_t>(w4.w), p_xl1[2], p_xl1[3], p_xh1[2], p_xh1[3]);
                    return (__half2float(wd_h) * xd) * (__low2float(hsum) + __high2float(hsum));
                };
#else
                const __half2 * sx = reinterpret_cast<const __half2 *>(sxb);
                auto dot_blk = [&](const int4 & w4, uint16_t wd_bits) -> float {
                    __half wd_h;
                    *reinterpret_cast<uint16_t *>(&wd_h) = wd_bits;
                    const uint32_t ws[4] = {
                        static_cast<uint32_t>(w4.x), static_cast<uint32_t>(w4.y),
                        static_cast<uint32_t>(w4.z), static_cast<uint32_t>(w4.w)
                    };
                    float bsum = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t w = ws[k];
                        const float2 xl0 = __half22float2(sx[k * 2 + 0]);
                        const float2 xl1 = __half22float2(sx[k * 2 + 1]);
                        const float2 xh0 = __half22float2(sx[k * 2 + 8]);
                        const float2 xh1 = __half22float2(sx[k * 2 + 9]);
                        bsum += (static_cast<float>((w >>  0) & 0xFu) - 8.0f) * 0.0625f * xl0.x;
                        bsum += (static_cast<float>((w >> 16) & 0xFu) - 8.0f) * 0.0625f * xl0.y;
                        bsum += (static_cast<float>((w >>  8) & 0xFu) - 8.0f) * 0.0625f * xl1.x;
                        bsum += (static_cast<float>((w >> 24) & 0xFu) - 8.0f) * 0.0625f * xl1.y;
                        bsum += (static_cast<float>((w >>  4) & 0xFu) - 8.0f) * 0.0625f * xh0.x;
                        bsum += (static_cast<float>((w >> 20) & 0xFu) - 8.0f) * 0.0625f * xh0.y;
                        bsum += (static_cast<float>((w >> 12) & 0xFu) - 8.0f) * 0.0625f * xh1.x;
                        bsum += (static_cast<float>((w >> 28) & 0xFu) - 8.0f) * 0.0625f * xh1.y;
                    }
                    return (__half2float(wd_h) * xd) * bsum;
                };
#endif
                acc0 += dot_blk(__ldg(&w_q_r0[b]), __ldg(&w_s_r0[b]));
                if (has_r1) acc1 += dot_blk(__ldg(&w_q_r1[b]), __ldg(&w_s_r1[b]));
            }
        }
        if (kb0 + KB_TILE < K_blocks) __syncthreads();
    }

    if (has_r0) {
        acc0 = g4_warp_sum(acc0);
        if (has_r1) acc1 = g4_warp_sum(acc1);
        if (lane == 0) {
            y_r0[static_cast<size_t>(t) * m_r0 + r0] = acc0;
            if (has_r1) y_r1[static_cast<size_t>(t) * m_r1 + r1] = acc1;
        }
    }
}

// FUSED sm_60 Q4_0 Gate + Up + GeGLU + Direct Q4_0 Input-Prep (16 warps = 32 outputs = 1 down block!)
// Streams 8 half2s at a time from smem (<= 56 regs -> 2 CTAs/SM = 32 warps/SM on P100!).
__global__ void g4_ffn_gate_up_geglu_prep_q4_0_sm60_kernel(
    const int4 *     __restrict__ G_qs,
    const uint16_t * __restrict__ G_sc,
    const int4 *     __restrict__ U_qs,
    const uint16_t * __restrict__ U_sc,
    const int4 *     __restrict__ X_h2,
    const float *    __restrict__ X_d,
    __half2 *        __restrict__ out_h2,
    float *          __restrict__ out_d,
    int K_blocks, int ff, int T) {

    constexpr int KB_TILE = 128;
    __shared__ int4  s_x[KB_TILE * 4];
    __shared__ float s_d[KB_TILE];
    __shared__ float s_act[32];

    const int t = blockIdx.y;
    if (t >= T) return;

    const int warp   = threadIdx.y; // 0..15
    const int lane   = threadIdx.x; // 0..31
    const int tid    = warp * 32 + lane;
    const int blk_ff = blockIdx.x;
    const int row0   = blk_ff * 32 + warp * 2;
    const int row1   = row0 + 1;

    const int4 *  x_h2_t = X_h2 + static_cast<size_t>(t) * K_blocks * 4;
    const float * x_d_t  = X_d  + static_cast<size_t>(t) * K_blocks;

    float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
#if __CUDA_ARCH__ >= 600
    const __half2 c1p5 = __floats2half2_rn(1.5f, 1.5f);
#endif

    for (int kb0 = 0; kb0 < K_blocks; kb0 += KB_TILE) {
        const int kb_n   = (kb0 + KB_TILE <= K_blocks) ? KB_TILE : (K_blocks - kb0);
        const int tot_i4 = kb_n * 4;
        const int4 * src_x = x_h2_t + kb0 * 4;
        if (tid < tot_i4) {
            s_x[tid] = __ldg(&src_x[tid]);
        }
        if (tid < kb_n) {
            s_d[tid] = __ldg(&x_d_t[kb0 + tid]);
        }
        __syncthreads();

        for (int tb = lane; tb < kb_n; tb += 32) {
            const int b = kb0 + tb;
            const float xd = s_d[tb];
            const int4 * sxb = s_x + tb * 4;

#if __CUDA_ARCH__ >= 600
            auto dot_blk = [&](const int4 & w4, uint16_t wd_bits) -> float {
                __half wd_h;
                *reinterpret_cast<uint16_t *>(&wd_h) = wd_bits;
                __half2 hsum = __floats2half2_rn(0.0f, 0.0f);
                auto mac4 = [&](uint32_t w, __half2 xa0, __half2 xa1, __half2 xb0, __half2 xb1) {
                    const uint32_t p0 = ((w << 6) & 0x03C003C0u) | 0x3C003C00u;
                    const uint32_t p1 = ((w >> 2) & 0x03C003C0u) | 0x3C003C00u;
                    const uint32_t p2 = ((w << 2) & 0x03C003C0u) | 0x3C003C00u;
                    const uint32_t p3 = ((w >> 6) & 0x03C003C0u) | 0x3C003C00u;
                    hsum = __hfma2(__hsub2(g4_u32_to_h2(p0), c1p5), xa0, hsum);
                    hsum = __hfma2(__hsub2(g4_u32_to_h2(p1), c1p5), xa1, hsum);
                    hsum = __hfma2(__hsub2(g4_u32_to_h2(p2), c1p5), xb0, hsum);
                    hsum = __hfma2(__hsub2(g4_u32_to_h2(p3), c1p5), xb1, hsum);
                };
                {
                    const int4 xl0 = sxb[0], xh0 = sxb[2];
                    const auto * p_xl0 = reinterpret_cast<const __half2 *>(&xl0);
                    const auto * p_xh0 = reinterpret_cast<const __half2 *>(&xh0);
                    mac4(static_cast<uint32_t>(w4.x), p_xl0[0], p_xl0[1], p_xh0[0], p_xh0[1]);
                    mac4(static_cast<uint32_t>(w4.y), p_xl0[2], p_xl0[3], p_xh0[2], p_xh0[3]);
                }
                {
                    const int4 xl1 = sxb[1], xh1 = sxb[3];
                    const auto * p_xl1 = reinterpret_cast<const __half2 *>(&xl1);
                    const auto * p_xh1 = reinterpret_cast<const __half2 *>(&xh1);
                    mac4(static_cast<uint32_t>(w4.z), p_xl1[0], p_xl1[1], p_xh1[0], p_xh1[1]);
                    mac4(static_cast<uint32_t>(w4.w), p_xl1[2], p_xl1[3], p_xh1[2], p_xh1[3]);
                }
                return (__half2float(wd_h) * xd) * (__low2float(hsum) + __high2float(hsum));
            };
#else
            const __half2 * sx = reinterpret_cast<const __half2 *>(sxb);
            auto dot_blk = [&](const int4 & w4, uint16_t wd_bits) -> float {
                __half wd_h;
                *reinterpret_cast<uint16_t *>(&wd_h) = wd_bits;
                const uint32_t ws[4] = {
                    static_cast<uint32_t>(w4.x), static_cast<uint32_t>(w4.y),
                    static_cast<uint32_t>(w4.z), static_cast<uint32_t>(w4.w)
                };
                float bsum = 0.0f;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t w = ws[k];
                    const float2 xl0 = __half22float2(sx[k * 2 + 0]);
                    const float2 xl1 = __half22float2(sx[k * 2 + 1]);
                    const float2 xh0 = __half22float2(sx[k * 2 + 8]);
                    const float2 xh1 = __half22float2(sx[k * 2 + 9]);
                    bsum += (static_cast<float>((w >>  0) & 0xFu) - 8.0f) * 0.0625f * xl0.x;
                    bsum += (static_cast<float>((w >> 16) & 0xFu) - 8.0f) * 0.0625f * xl0.y;
                    bsum += (static_cast<float>((w >>  8) & 0xFu) - 8.0f) * 0.0625f * xl1.x;
                    bsum += (static_cast<float>((w >> 24) & 0xFu) - 8.0f) * 0.0625f * xl1.y;
                    bsum += (static_cast<float>((w >>  4) & 0xFu) - 8.0f) * 0.0625f * xh0.x;
                    bsum += (static_cast<float>((w >> 20) & 0xFu) - 8.0f) * 0.0625f * xh0.y;
                    bsum += (static_cast<float>((w >> 12) & 0xFu) - 8.0f) * 0.0625f * xh1.x;
                    bsum += (static_cast<float>((w >> 28) & 0xFu) - 8.0f) * 0.0625f * xh1.y;
                }
                return (__half2float(wd_h) * xd) * bsum;
            };
#endif
            const size_t idx0 = static_cast<size_t>(row0) * K_blocks + b;
            const size_t idx1 = static_cast<size_t>(row1) * K_blocks + b;
            g0 += dot_blk(__ldg(&G_qs[idx0]), __ldg(&G_sc[idx0]));
            g1 += dot_blk(__ldg(&G_qs[idx1]), __ldg(&G_sc[idx1]));
            u0 += dot_blk(__ldg(&U_qs[idx0]), __ldg(&U_sc[idx0]));
            u1 += dot_blk(__ldg(&U_qs[idx1]), __ldg(&U_sc[idx1]));
        }
        if (kb0 + KB_TILE < K_blocks) __syncthreads();
    }

    g0 = g4_warp_sum(g0); g1 = g4_warp_sum(g1);
    u0 = g4_warp_sum(u0); u1 = g4_warp_sum(u1);
    if (lane == 0) {
        s_act[warp * 2 + 0] = g4_gelu_tanh(g0) * u0;
        s_act[warp * 2 + 1] = g4_gelu_tanh(g1) * u1;
    }
    __syncthreads();

    if (warp == 0) {
        const int ff_blks = ff >> 5;
        const int b_out   = t * ff_blks + blk_ff;
        const float xv    = s_act[lane];
        const float am    = __shfl_sync(0xffffffffu, g4_warp_max(fabsf(xv)), 0);
        const float inv   = (am > 0.0f) ? (1.0f / am) : 0.0f;
        const float xn    = xv * inv;
        const float xn_p2 = __shfl_down_sync(0xffffffffu, xn, 2);
        if ((lane & 2) == 0) {
            const int slot = ((lane >> 2) << 1) | (lane & 1);
            out_h2[static_cast<size_t>(b_out) * 16 + slot] = __floats2half2_rn(xn, xn_p2);
        }
        if (lane == 0) {
            out_d[b_out] = am * 16.0f;
        }
    }
}

__global__ void g4_geglu_f16_inplace_kernel(
    __half *       __restrict__ gate,
    const __half * __restrict__ up,
    int64_t n) {
    const int64_t i2 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t i  = i2 * 2;
    if (i + 1 < n) {
        const float2 g = __half22float2(reinterpret_cast<const __half2 *>(gate)[i2]);
        const float2 u = __half22float2(reinterpret_cast<const __half2 *>(up)[i2]);
        reinterpret_cast<__half2 *>(gate)[i2] = __floats2half2_rn(g4_gelu_tanh(g.x) * u.x, g4_gelu_tanh(g.y) * u.y);
    } else if (i < n) {
        gate[i] = __float2half_rn(g4_gelu_tanh(__half2float(gate[i])) * __half2float(up[i]));
    }
}

// sm_61+ (GTX 1060 / Volta / Ampere) hardware dp4a_s32 SoA Q4_0 x Q8_1 GEMV (100% coalesced LDG.128!)
__global__ void g4_gemv_q4_0_soa_dp4a_kernel(
    const int4 *          __restrict__ W_qs, // [N, K_blocks] int4
    const uint16_t *      __restrict__ W_sc, // [N, K_blocks] fp16
    const BlockQ8_1_Act * __restrict__ X,    // [T, K_blocks]
    float *               __restrict__ Y,    // [T, N]
    int K_blocks, int N, int T) {

    extern __shared__ BlockQ8_1_Act s_x[];
    const int t    = blockIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * src = reinterpret_cast<const int4 *>(X + static_cast<size_t>(t) * K_blocks);
    int4 *       dst = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) dst[i] = src[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N || t >= T) return;

    const int lane = threadIdx.x;
    const int4 *     w_q_row = W_qs + static_cast<size_t>(row) * K_blocks;
    const uint16_t * w_s_row = W_sc + static_cast<size_t>(row) * K_blocks;

    float acc = 0.0f;
    for (int b = lane; b < K_blocks; b += 32) {
        const int4 w4 = __ldg(&w_q_row[b]);
        const int4 xlo = reinterpret_cast<const int4 *>(s_x[b].qs)[0];
        const int4 xhi = reinterpret_cast<const int4 *>(s_x[b].qs)[1];

        int sumi = dp4a_s32(static_cast<int>(static_cast<uint32_t>(w4.x) & 0x0F0F0F0Fu), xlo.x, 0);
        sumi     = dp4a_s32(static_cast<int>(static_cast<uint32_t>(w4.y) & 0x0F0F0F0Fu), xlo.y, sumi);
        sumi     = dp4a_s32(static_cast<int>(static_cast<uint32_t>(w4.z) & 0x0F0F0F0Fu), xlo.z, sumi);
        sumi     = dp4a_s32(static_cast<int>(static_cast<uint32_t>(w4.w) & 0x0F0F0F0Fu), xlo.w, sumi);
        sumi     = dp4a_s32(static_cast<int>((static_cast<uint32_t>(w4.x) >> 4) & 0x0F0F0F0Fu), xhi.x, sumi);
        sumi     = dp4a_s32(static_cast<int>((static_cast<uint32_t>(w4.y) >> 4) & 0x0F0F0F0Fu), xhi.y, sumi);
        sumi     = dp4a_s32(static_cast<int>((static_cast<uint32_t>(w4.z) >> 4) & 0x0F0F0F0Fu), xhi.z, sumi);
        sumi     = dp4a_s32(static_cast<int>((static_cast<uint32_t>(w4.w) >> 4) & 0x0F0F0F0Fu), xhi.w, sumi);
        sumi    -= 8 * (s_x[b].s0 + s_x[b].s1);

        __half wd_h;
        *reinterpret_cast<uint16_t *>(&wd_h) = __ldg(&w_s_row[b]);
        acc += (__half2float(wd_h) * s_x[b].d) * static_cast<float>(sumi);
    }

    acc = g4_warp_sum(acc);
    if (lane == 0) Y[static_cast<size_t>(t) * N + row] = acc;
}

// Generic FP32 SoA Q4_0 GEMV fallback (Maxwell sm_50/52 or very large K > smem limit)
__global__ void g4_gemv_q4_0_soa_f32_kernel(
    const int4 *     __restrict__ W_qs,
    const uint16_t * __restrict__ W_sc,
    const float *    __restrict__ X,
    float *          __restrict__ Y,
    int K_blocks, int N, int T) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int t   = blockIdx.y;
    if (row >= N || t >= T) return;

    const int lane = threadIdx.x;
    const int4 *     w_q_row = W_qs + static_cast<size_t>(row) * K_blocks;
    const uint16_t * w_s_row = W_sc + static_cast<size_t>(row) * K_blocks;
    const float *    x_row   = X + static_cast<size_t>(t) * (K_blocks * 32);

    float acc = 0.0f;
    for (int b = lane; b < K_blocks; b += 32) {
        const int4 w4 = __ldg(&w_q_row[b]);
        __half wd_h;
        *reinterpret_cast<uint16_t *>(&wd_h) = __ldg(&w_s_row[b]);
        const uint32_t ws[4] = {
            static_cast<uint32_t>(w4.x), static_cast<uint32_t>(w4.y),
            static_cast<uint32_t>(w4.z), static_cast<uint32_t>(w4.w)
        };
        const float4 * x4 = reinterpret_cast<const float4 *>(x_row + b * 32);
        float bsum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t w = ws[k];
            const float4 xl = __ldg(&x4[k]);
            const float4 xh = __ldg(&x4[k + 4]);
            bsum += static_cast<float>(static_cast<int>(w & 0xFu) - 8) * xl.x;
            bsum += static_cast<float>(static_cast<int>((w >> 8) & 0xFu) - 8) * xl.y;
            bsum += static_cast<float>(static_cast<int>((w >> 16) & 0xFu) - 8) * xl.z;
            bsum += static_cast<float>(static_cast<int>((w >> 24) & 0xFu) - 8) * xl.w;
            bsum += static_cast<float>(static_cast<int>((w >> 4) & 0xFu) - 8) * xh.x;
            bsum += static_cast<float>(static_cast<int>((w >> 12) & 0xFu) - 8) * xh.y;
            bsum += static_cast<float>(static_cast<int>((w >> 20) & 0xFu) - 8) * xh.z;
            bsum += static_cast<float>(static_cast<int>((w >> 28) & 0xFu) - 8) * xh.w;
        }
        acc += __half2float(wd_h) * bsum;
    }

    acc = g4_warp_sum(acc);
    if (lane == 0) Y[static_cast<size_t>(t) * N + row] = acc;
}

} // namespace

// ============================================================================
// Public Launchers
// ============================================================================

void launch_gemma4_embedding_q4_0(
    const DeviceTensor & tok_embd, const int32_t * d_tokens, float * d_out,
    int n_embd, int n_tokens, cudaStream_t stream) {

    if (tok_embd.dtype != DType::Q4_0 || !tok_embd.is_soa) {
        throw std::runtime_error("launch_gemma4_embedding_q4_0: expected SoA Q4_0 tensor");
    }
    g4_embedding_q4_0_soa_kernel<<<n_tokens, 256, 0, stream>>>(
        static_cast<const uint8_t *>(tok_embd.ptr),
        static_cast<const uint16_t *>(tok_embd.scales_ptr),
        d_tokens, d_out, n_embd);
}

bool try_launch_gemma4_linear_q4_0(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    if (W.dtype != DType::Q4_0 || !W.is_soa) return false;
    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);
    const int K_blocks = K >> 5;
    const int sm = device_sm_version(W.device);
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };

    // Prefill GEMM (T >= 16)
    if (T >= 16 && cublas != nullptr) {
        cublasSetStream(cublas, stream);
        if (sm == 60 || sm >= 70) {
            const size_t xf16_b = align256(static_cast<size_t>(T) * K * sizeof(__half));
            const size_t yf16_b = (sm == 60) ? align256(static_cast<size_t>(T) * M * sizeof(__half)) : 0;
            if (xf16_b + yf16_b + static_cast<size_t>(K) * sizeof(__half) <= scratch_bytes) {
                auto * base  = static_cast<uint8_t *>(d_scratch);
                auto * d_x16 = reinterpret_cast<__half *>(base);
                auto * d_y16 = reinterpret_cast<__half *>(base + xf16_b);
                auto * d_w16 = reinterpret_cast<__half *>(base + xf16_b + yf16_b);
                const size_t rem = scratch_bytes - xf16_b - yf16_b;
                int chunk_m = std::max(1, std::min(M, static_cast<int>(rem / (static_cast<size_t>(K) * sizeof(__half)))));
                if (chunk_m < M && chunk_m >= 128) chunk_m &= ~127; // Keep cublasHgemm M and d_y16+m0 128-aligned!

                launch_cast_f32_to_f16(d_x, d_x16, static_cast<int64_t>(T) * K, stream);
                const __half h_one = __float2half_rn(1.0f), h_zero = __float2half_rn(0.0f);
                const float  f_one = 1.0f, f_zero = 0.0f;
                const auto * qs4 = static_cast<const int4 *>(W.ptr);
                const auto * sc  = static_cast<const uint16_t *>(W.scales_ptr);

                for (int m0 = 0; m0 < M; m0 += chunk_m) {
                    const int cm = std::min(chunk_m, M - m0);
                    const int64_t b0   = static_cast<int64_t>(m0) * K_blocks;
                    const int64_t nblk = static_cast<int64_t>(cm) * K_blocks;
                    const int64_t grid = (nblk + 255) / 256;
                    g4_dequant_q4_0_soa_kernel<__half><<<static_cast<unsigned int>(grid), 256, 0, stream>>>(
                        qs4 + b0, sc + b0, d_w16, nblk);
                    if (sm == 60) {
                        cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                    cm, T, K, &h_one, d_w16, K, d_x16, K, &h_zero, d_y16 + m0, M);
                    } else {
                        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                     cm, T, K, &f_one,
                                     d_w16, CUDA_R_16F, K, d_x16, CUDA_R_16F, K,
                                     &f_zero, d_y + m0, CUDA_R_32F, M,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    }
                }
                if (sm == 60) {
                    const int64_t ny = static_cast<int64_t>(T) * M;
                    g4_f16_to_f32_vec_kernel<<<static_cast<int>(((ny + 1) / 2 + 255) / 256), 256, 0, stream>>>(d_y16, d_y, ny);
                }
                return true;
            }
        } else if (static_cast<size_t>(K) * sizeof(float) <= scratch_bytes) {
            auto * d_wf32 = static_cast<float *>(d_scratch);
            int chunk_m = std::max(1, std::min(M, static_cast<int>(scratch_bytes / (static_cast<size_t>(K) * sizeof(float)))));
            if (chunk_m < M && chunk_m >= 128) chunk_m &= ~127;
            const float f_one = 1.0f, f_zero = 0.0f;
            const auto * qs4 = static_cast<const int4 *>(W.ptr);
            const auto * sc  = static_cast<const uint16_t *>(W.scales_ptr);

            for (int m0 = 0; m0 < M; m0 += chunk_m) {
                const int cm = std::min(chunk_m, M - m0);
                const int64_t b0   = static_cast<int64_t>(m0) * K_blocks;
                const int64_t nblk = static_cast<int64_t>(cm) * K_blocks;
                const int64_t grid = (nblk + 255) / 256;
                g4_dequant_q4_0_soa_kernel<float><<<static_cast<unsigned int>(grid), 256, 0, stream>>>(
                    qs4 + b0, sc + b0, d_wf32, nblk);
                cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            cm, T, K, &f_one, d_wf32, K, d_x, K, &f_zero, d_y + m0, M);
            }
            return true;
        }
    }

    // Decode GEMV (T < 16)
    if (sm == 60 && d_scratch != nullptr) {
        const size_t xh2_b = align256(static_cast<size_t>(T) * K_blocks * 16 * sizeof(__half2));
        const size_t xd_b  = align256(static_cast<size_t>(T) * K_blocks * sizeof(float));
        if (xh2_b + xd_b <= scratch_bytes) {
            auto * d_xh2 = reinterpret_cast<__half2 *>(d_scratch);
            auto * d_xd  = reinterpret_cast<float *>(static_cast<uint8_t *>(d_scratch) + xh2_b);
            const int tot_b = T * K_blocks;
            g4_quantize_x_h2_kernel<<<(tot_b + 7) / 8, dim3(32, 8), 0, stream>>>(d_x, d_xh2, d_xd, tot_b);
            const int wpb = (M <= 1024) ? 4 : 8;
            const int rows_per_blk = wpb * 2;
            g4_gemv_q4_0_soa_sm60_h2_kernel<<<dim3((M + rows_per_blk - 1) / rows_per_blk, T), dim3(32, wpb), 0, stream>>>(
                static_cast<const int4 *>(W.ptr), static_cast<const uint16_t *>(W.scales_ptr),
                reinterpret_cast<const int4 *>(d_xh2), d_xd, d_y, K_blocks, M, T);
            return true;
        }
    } else if (sm >= 61 && d_scratch != nullptr) {
        const int wpb = (M <= 640 || K_blocks > 256) ? 4 : 8;
        const size_t q8_b = static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act);
        const size_t smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
        if (q8_b <= scratch_bytes && smem <= 48 * 1024) {
            launch_quantize_q8_1(d_x, d_scratch, K, T, stream);
            g4_gemv_q4_0_soa_dp4a_kernel<<<dim3((M + wpb - 1) / wpb, T), dim3(32, wpb), smem, stream>>>(
                static_cast<const int4 *>(W.ptr), static_cast<const uint16_t *>(W.scales_ptr),
                static_cast<const BlockQ8_1_Act *>(d_scratch), d_y, K_blocks, M, T);
            return true;
        }
    }

    const int wpb = (M <= 640) ? 4 : 8;
    g4_gemv_q4_0_soa_f32_kernel<<<dim3((M + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
        static_cast<const int4 *>(W.ptr), static_cast<const uint16_t *>(W.scales_ptr),
        d_x, d_y, K_blocks, M, T);
    return true;
}

bool try_launch_gemma4_linear_multi_q4_0(
    const DeviceTensor * const * Ws, float * const * Ys, int n_mat,
    const float * d_x, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    if (n_mat <= 0 || n_mat > 3 || T >= 16) return false;
    const int K = static_cast<int>(Ws[0]->ne[0]);
    const int K_blocks = K >> 5;
    const int sm = device_sm_version(Ws[0]->device);
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    for (int i = 0; i < n_mat; ++i) {
        if (!Ws[i] || Ws[i]->dtype != DType::Q4_0 || !Ws[i]->is_soa || static_cast<int>(Ws[i]->ne[0]) != K) {
            return false;
        }
    }

    if (sm == 60 && d_scratch != nullptr) {
        const size_t xh2_b = align256(static_cast<size_t>(T) * K_blocks * 16 * sizeof(__half2));
        const size_t xd_b  = align256(static_cast<size_t>(T) * K_blocks * sizeof(float));
        if (xh2_b + xd_b <= scratch_bytes) {
            auto * d_xh2 = reinterpret_cast<__half2 *>(d_scratch);
            auto * d_xd  = reinterpret_cast<float *>(static_cast<uint8_t *>(d_scratch) + xh2_b);
            const int tot_b = T * K_blocks;
            g4_quantize_x_h2_kernel<<<(tot_b + 7) / 8, dim3(32, 8), 0, stream>>>(d_x, d_xh2, d_xd, tot_b);
            const auto * w0_q = static_cast<const int4 *>(Ws[0]->ptr);
            const auto * w0_s = static_cast<const uint16_t *>(Ws[0]->scales_ptr);
            float *      y0   = Ys[0];
            const int    m0   = static_cast<int>(Ws[0]->ne[1]);
            const auto * w1_q = (n_mat > 1) ? static_cast<const int4 *>(Ws[1]->ptr) : w0_q;
            const auto * w1_s = (n_mat > 1) ? static_cast<const uint16_t *>(Ws[1]->scales_ptr) : w0_s;
            float *      y1   = (n_mat > 1) ? Ys[1] : y0;
            const int    m1   = (n_mat > 1) ? static_cast<int>(Ws[1]->ne[1]) : 0;
            const auto * w2_q = (n_mat > 2) ? static_cast<const int4 *>(Ws[2]->ptr) : w0_q;
            const auto * w2_s = (n_mat > 2) ? static_cast<const uint16_t *>(Ws[2]->scales_ptr) : w0_s;
            float *      y2   = (n_mat > 2) ? Ys[2] : y0;
            const int    m2   = (n_mat > 2) ? static_cast<int>(Ws[2]->ne[1]) : 0;
            const int tot_rows = m0 + m1 + m2;
            constexpr int wpb = 8;
            constexpr int rows_per_blk = wpb * 2;
            g4_gemv_multi_q4_0_soa_sm60_h2_kernel<<<dim3((tot_rows + rows_per_blk - 1) / rows_per_blk, T), dim3(32, wpb), 0, stream>>>(
                w0_q, w0_s, y0, m0, w1_q, w1_s, y1, m1, w2_q, w2_s, y2, m2,
                reinterpret_cast<const int4 *>(d_xh2), d_xd, K_blocks, T);
            return true;
        }
    } else if (sm >= 61 && d_scratch != nullptr) {
        const size_t q8_b = static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act);
        const size_t smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
        if (q8_b <= scratch_bytes && smem <= 48 * 1024) {
            launch_quantize_q8_1(d_x, d_scratch, K, T, stream);
            for (int i = 0; i < n_mat; ++i) {
                const int M = static_cast<int>(Ws[i]->ne[1]);
                const int wpb = (M <= 640 || K_blocks > 256) ? 4 : 8;
                g4_gemv_q4_0_soa_dp4a_kernel<<<dim3((M + wpb - 1) / wpb, T), dim3(32, wpb), smem, stream>>>(
                    static_cast<const int4 *>(Ws[i]->ptr), static_cast<const uint16_t *>(Ws[i]->scales_ptr),
                    static_cast<const BlockQ8_1_Act *>(d_scratch), Ys[i], K_blocks, M, T);
            }
            return true;
        }
    }

    for (int i = 0; i < n_mat; ++i) {
        try_launch_gemma4_linear_q4_0(nullptr, *Ws[i], d_x, Ys[i], T, d_scratch, scratch_bytes, stream);
    }
    return true;
}

bool try_launch_gemma4_ffn_q4_0_sm60(
    cublasHandle_t cublas,
    const DeviceTensor & W_gate, const DeviceTensor & W_up, const DeviceTensor & W_down,
    const float * d_x, float * d_out, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    if (d_scratch == nullptr || device_sm_version(W_gate.device) != 60) return false;
    if (W_gate.dtype != DType::Q4_0 || !W_gate.is_soa ||
        W_up.dtype   != DType::Q4_0 || !W_up.is_soa   ||
        W_down.dtype != DType::Q4_0 || !W_down.is_soa) {
        return false;
    }
    const int K  = static_cast<int>(W_gate.ne[0]);
    const int ff = static_cast<int>(W_gate.ne[1]);
    if ((K & 31) != 0 || (ff & 31) != 0) return false;
    const int K_blocks = K >> 5;
    const int ff_blks  = ff >> 5;
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    auto * base = static_cast<uint8_t *>(d_scratch);

    if (T < 16) {
        const size_t xh2_b = align256(static_cast<size_t>(T) * K_blocks * 16 * sizeof(__half2));
        const size_t xd_b  = align256(static_cast<size_t>(T) * K_blocks * sizeof(float));
        const size_t ah2_b = align256(static_cast<size_t>(T) * ff_blks  * 16 * sizeof(__half2));
        const size_t ad_b  = align256(static_cast<size_t>(T) * ff_blks  * sizeof(float));
        if (xh2_b + xd_b + ah2_b + ad_b > scratch_bytes) return false;

        size_t off = 0;
        auto * d_xh2 = reinterpret_cast<__half2 *>(base + off); off += xh2_b;
        auto * d_xd  = reinterpret_cast<float *>  (base + off); off += xd_b;
        auto * d_ah2 = reinterpret_cast<__half2 *>(base + off); off += ah2_b;
        auto * d_ad  = reinterpret_cast<float *>  (base + off);

        const int tot_xb = T * K_blocks;
        g4_quantize_x_h2_kernel<<<(tot_xb + 7) / 8, dim3(32, 8), 0, stream>>>(d_x, d_xh2, d_xd, tot_xb);
        g4_ffn_gate_up_geglu_prep_q4_0_sm60_kernel<<<dim3(ff_blks, T), dim3(32, 16), 0, stream>>>(
            static_cast<const int4 *>(W_gate.ptr), static_cast<const uint16_t *>(W_gate.scales_ptr),
            static_cast<const int4 *>(W_up.ptr),   static_cast<const uint16_t *>(W_up.scales_ptr),
            reinterpret_cast<const int4 *>(d_xh2), d_xd, d_ah2, d_ad, K_blocks, ff, T);
        constexpr int wpb = 8;
        constexpr int rpb = wpb * 2;
        g4_gemv_q4_0_soa_sm60_h2_kernel<<<dim3((K + rpb - 1) / rpb, T), dim3(32, wpb), 0, stream>>>(
            static_cast<const int4 *>(W_down.ptr), static_cast<const uint16_t *>(W_down.scales_ptr),
            reinterpret_cast<const int4 *>(d_ah2), d_ad, d_out, ff_blks, K, T);
        return true;
    }

    // Pure-FP16 cublasHgemm FFN Prefill (T >= 16) on sm_60: eliminates intermediate F16<->F32 & F32 GeGLU!
    if (cublas == nullptr) return false;
    const size_t x_b = align256(static_cast<size_t>(T) * K  * sizeof(__half));
    const size_t g_b = align256(static_cast<size_t>(T) * ff * sizeof(__half));
    const size_t u_b = g_b;
    const size_t min_w = static_cast<size_t>(std::max(K, ff)) * 256 * sizeof(__half);
    if (x_b + g_b + u_b + min_w > scratch_bytes) return false;

    auto * d_x16 = reinterpret_cast<__half *>(base);
    auto * d_g16 = reinterpret_cast<__half *>(base + x_b);
    auto * d_u16 = reinterpret_cast<__half *>(base + x_b + g_b);
    auto * d_w16 = reinterpret_cast<__half *>(base + x_b + g_b + u_b);
    const size_t w_cap = scratch_bytes - (x_b + g_b + u_b);

    launch_cast_f32_to_f16(d_x, d_x16, static_cast<int64_t>(T) * K, stream);
    cublasSetStream(cublas, stream);
    const __half h_one = __float2half_rn(1.0f), h_zero = __float2half_rn(0.0f);

    auto run_q4_hgemm = [&](const DeviceTensor & W, const __half * X16, __half * Y16, __half * W16, size_t cap) {
        const int Kin = static_cast<int>(W.ne[0]);
        const int M   = static_cast<int>(W.ne[1]);
        const int kb  = Kin >> 5;
        int chunk_m = std::max(1, std::min(M, static_cast<int>(cap / (static_cast<size_t>(Kin) * sizeof(__half)))));
        if (chunk_m < M && chunk_m >= 128) chunk_m &= ~127;
        const auto * qs4 = static_cast<const int4 *>(W.ptr);
        const auto * sc  = static_cast<const uint16_t *>(W.scales_ptr);
        for (int m0 = 0; m0 < M; m0 += chunk_m) {
            const int cm = std::min(chunk_m, M - m0);
            const int64_t b0   = static_cast<int64_t>(m0) * kb;
            const int64_t nblk = static_cast<int64_t>(cm) * kb;
            g4_dequant_q4_0_soa_kernel<__half><<<static_cast<unsigned int>((nblk + 255) / 256), 256, 0, stream>>>(
                qs4 + b0, sc + b0, W16, nblk);
            cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        cm, T, Kin, &h_one, W16, Kin, X16, Kin, &h_zero, Y16 + m0, M);
        }
    };

    run_q4_hgemm(W_gate, d_x16, d_g16, d_w16, w_cap);
    run_q4_hgemm(W_up,   d_x16, d_u16, d_w16, w_cap);
    const int64_t nf = static_cast<int64_t>(T) * ff;
    g4_geglu_f16_inplace_kernel<<<static_cast<int>(((nf + 1) / 2 + 255) / 256), 256, 0, stream>>>(d_g16, d_u16, nf);
    // Reuse d_x16 as output FP16 buffer [T, K] and d_u16..end as W_down FP16 buffer!
    run_q4_hgemm(W_down, d_g16, d_x16, d_u16, scratch_bytes - (x_b + g_b));
    const int64_t ny = static_cast<int64_t>(T) * K;
    g4_f16_to_f32_vec_kernel<<<static_cast<int>(((ny + 1) / 2 + 255) / 256), 256, 0, stream>>>(d_x16, d_out, ny);
    return true;
}

bool try_launch_gemma4_linear_sm60_f16(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    if (cublas == nullptr || T < 16 || device_sm_version(W.device) != 60) return false;
    const bool is_q6k = (W.dtype == DType::Q6_K && W.is_soa);
    const bool is_f16 = (W.dtype == DType::F16);
    if (!is_q6k && !is_f16) return false;

    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    const size_t xf16_b = align256(static_cast<size_t>(T) * K * sizeof(__half));
    const size_t yf16_b = align256(static_cast<size_t>(T) * M * sizeof(__half));
    const size_t need_w = is_q6k ? static_cast<size_t>(K) * sizeof(__half) : 0;
    if (xf16_b + yf16_b + need_w > scratch_bytes) return false;

    auto * base  = static_cast<uint8_t *>(d_scratch);
    auto * d_x16 = reinterpret_cast<__half *>(base);
    auto * d_y16 = reinterpret_cast<__half *>(base + xf16_b);

    launch_cast_f32_to_f16(d_x, d_x16, static_cast<int64_t>(T) * K, stream);
    cublasSetStream(cublas, stream);
    const __half h_one = __float2half_rn(1.0f), h_zero = __float2half_rn(0.0f);

    if (is_f16) {
        cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    M, T, K,
                    &h_one,  static_cast<const __half *>(W.ptr), K,
                    d_x16, K,
                    &h_zero, d_y16, M);
    } else {
        auto * d_w16 = reinterpret_cast<__half *>(base + xf16_b + yf16_b);
        const size_t rem = scratch_bytes - xf16_b - yf16_b;
        const int chunk_m = std::max(1, std::min(M, static_cast<int>(rem / (static_cast<size_t>(K) * sizeof(__half)))));
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
            g4_dequant_q6_k_soa_f16_kernel<<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
                ql + b0 * 128, qh + b0 * 64, sc + b0 * 16, d + b0, d_w16, nblk);
            cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        cm, T, K,
                        &h_one,  d_w16, K,
                        d_x16, K,
                        &h_zero, d_y16 + m0, M);
        }
    }

    const int64_t ny = static_cast<int64_t>(T) * M;
    g4_f16_to_f32_vec_kernel<<<static_cast<int>(((ny + 1) / 2 + 255) / 256), 256, 0, stream>>>(d_y16, d_y, ny);
    return true;
}

void launch_gemma4_scale(float * d_x, int64_t n, float scale, cudaStream_t stream) {
    const int64_t n4 = (n + 3) >> 2;
    g4_scale_inplace_kernel<<<static_cast<int>((n4 + 255) / 256), 256, 0, stream>>>(d_x, n, scale);
}

void launch_gemma4_scale_dev(float * d_x, const float * d_scale, int64_t n, cudaStream_t stream) {
    const int64_t n4 = (n + 3) >> 2;
    g4_scale_dev_inplace_kernel<<<static_cast<int>((n4 + 255) / 256), 256, 0, stream>>>(d_x, d_scale, n);
}

void launch_gemma4_ple_combine(
    const float * d_proj, const float * d_raw_ple, const float * d_proj_norm_w,
    float * d_ple_out, int T, int n_layers, int n_embd, int ple_dim,
    float rms_eps, cudaStream_t stream) {

    const float proj_eps        = rms_eps * static_cast<float>(n_embd);
    const float proj_norm_scale = 1.0f / std::sqrt(2.0f);
    const float tok_ple_scale   = std::sqrt(static_cast<float>(ple_dim)) / std::sqrt(2.0f);
    g4_ple_combine_kernel<<<dim3(n_layers, T), 256, 0, stream>>>(
        d_proj, d_raw_ple, d_proj_norm_w, d_ple_out,
        n_layers, ple_dim, proj_eps, proj_norm_scale, tok_ple_scale);
}

void launch_gemma4_geglu(const float * d_gate, const float * d_up, float * d_out, int64_t n, cudaStream_t stream) {
    const int64_t n4 = (n + 3) >> 2;
    g4_geglu_kernel<<<static_cast<int>((n4 + 255) / 256), 256, 0, stream>>>(d_gate, d_up, d_out, n);
}

void launch_gemma4_ple_gate_gelu_mul(
    float * d_gate, const float * d_ple,
    int T, int n_layers, int il, int ple_dim, int ple_tok_off, cudaStream_t stream) {
    g4_ple_gate_gelu_mul_kernel<<<dim3((ple_dim + 255) / 256, T), 256, 0, stream>>>(
        d_gate, d_ple, T, n_layers, il, ple_dim, ple_tok_off);
}

void launch_gemma4_ple_f16_t1(
    const void * d_w_gate_f16, const void * d_w_proj_f16,
    const float * d_x, const float * d_ple,
    float * d_gate_tmp, float * d_out,
    int H, int ple_dim, int n_layers, int il, int ple_tok_off, cudaStream_t stream) {

    const dim3 blk(32, 8);
    g4_ple_f16_gate_mul_kernel<<<(ple_dim + 7) / 8, blk, 0, stream>>>(
        static_cast<const __half *>(d_w_gate_f16), d_x, d_ple, d_gate_tmp,
        H, ple_dim, n_layers, il, ple_tok_off);
    g4_ple_f16_proj_kernel<<<(H + 7) / 8, blk, 0, stream>>>(
        static_cast<const __half *>(d_w_proj_f16), d_gate_tmp, d_out,
        H, ple_dim);
}

void launch_gemma4_post_norm_add_scale(
    const float * d_res, const float * d_sub, const float * d_norm_w,
    const float * d_layer_scale, float * d_out,
    int dim, int n_rows, float eps, cudaStream_t stream) {
    g4_post_norm_add_scale_kernel<<<n_rows, 256, 0, stream>>>(
        d_res, d_sub, d_norm_w, d_layer_scale, d_out, dim, eps);
}

void launch_gemma4_logit_softcap(float * d_logits, int64_t n, float cap, cudaStream_t stream) {
    if (cap <= 0.0f) return;
    const int64_t n4 = (n + 3) >> 2;
    g4_logit_softcap_kernel<<<static_cast<int>((n4 + 255) / 256), 256, 0, stream>>>(d_logits, n, cap);
}

void launch_gemma4_qkv_norm_rope_kv_write(
    float * d_q, float * d_k, const float * d_v,
    const float * d_q_norm_w, const float * d_k_norm_w, const float * d_rope_freqs,
    void * d_k_cache, void * d_v_cache, KvCacheType kv_type,
    const int32_t * d_pos, int T, int n_head, int n_kv,
    int head_dim, int rope_dim, int kv_cap, float rms_eps, float freq_base,
    float rope_scale, bool has_kv, cudaStream_t stream) {

    g4_qk_norm_rope_neox_kernel<<<dim3(n_head, T), 256, 0, stream>>>(
        d_q, d_q_norm_w, d_rope_freqs, d_pos, n_head, head_dim, rope_dim, rms_eps, freq_base, rope_scale);

    if (!has_kv) return;

    g4_qk_norm_rope_neox_kernel<<<dim3(n_kv, T), 256, 0, stream>>>(
        d_k, d_k_norm_w, d_rope_freqs, d_pos, n_kv, head_dim, rope_dim, rms_eps, freq_base, rope_scale);

    if (kv_type == KvCacheType::Q8_0) {
        g4_vnorm_kv_write_kernel<KvCacheType::Q8_0><<<dim3(n_kv, T), 256, 0, stream>>>(
            d_k, d_v, d_k_cache, d_v_cache, d_pos, n_kv, head_dim, kv_cap, rms_eps);
    } else if (kv_type == KvCacheType::Q4_0) {
        g4_vnorm_kv_write_kernel<KvCacheType::Q4_0><<<dim3(n_kv, T), 256, 0, stream>>>(
            d_k, d_v, d_k_cache, d_v_cache, d_pos, n_kv, head_dim, kv_cap, rms_eps);
    } else {
        g4_vnorm_kv_write_kernel<KvCacheType::F16><<<dim3(n_kv, T), 256, 0, stream>>>(
            d_k, d_v, d_k_cache, d_v_cache, d_pos, n_kv, head_dim, kv_cap, rms_eps);
    }
}

void launch_gemma4_attn(
    cublasHandle_t cublas,
    const float * d_q, const void * d_k_cache, const void * d_v_cache, KvCacheType kv_type,
    const int32_t * d_pos, float * d_out,
    void * d_scratch, size_t scratch_bytes,
    int T, int n_head, int n_kv, int head_dim, int kv_cap,
    int swa_window, int pos_min, int pos_max, cudaStream_t stream) {

    int dev = 0;
    cudaGetDevice(&dev);
    const int sm = device_sm_version(dev);
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };

    // A. Tiled FlashAttention Prefill (T >= 8) with Causal + SWA Active-Query Tile Pruning!
    if (T >= 8 && cublas != nullptr && pos_max >= 0) {
        const int gqa      = n_head / n_kv;
        const int M_per_kv = T * gqa;
        const int M_tot    = n_kv * M_per_kv;
        const int k_start  = (swa_window > 0 && pos_min - swa_window + 1 > 0) ? (pos_min - swa_window + 1) : 0;
        const int k_end    = pos_max + 1;
        const int BK       = 128;
        const bool contig_pos = (pos_max - pos_min == T - 1);

        const bool use_fp16 = (sm == 60 || sm >= 70);
        const size_t elem_b = use_fp16 ? sizeof(__half) : sizeof(float);

        const size_t q_b   = align256(static_cast<size_t>(M_tot) * head_dim * elem_b);
        const size_t kv_b  = align256(static_cast<size_t>(n_kv) * BK * head_dim * elem_b);
        const size_t s_b   = align256(static_cast<size_t>(M_tot) * BK * elem_b);
        const size_t p_b   = s_b;
        const size_t oa_b  = align256(static_cast<size_t>(M_tot) * head_dim * sizeof(float));
        const size_t ot_b  = (sm == 60) ? align256(static_cast<size_t>(M_tot) * head_dim * sizeof(__half)) : 0;
        const size_t ml_b  = align256(static_cast<size_t>(M_tot) * sizeof(float));

        if (q_b + 2 * kv_b + s_b + p_b + oa_b + ot_b + 2 * ml_b <= scratch_bytes) {
            auto * base = static_cast<uint8_t *>(d_scratch);
            size_t off = 0;
            uint8_t * q_raw  = base + off; off += q_b;
            uint8_t * k_raw  = base + off; off += kv_b;
            uint8_t * v_raw  = base + off; off += kv_b;
            uint8_t * s_raw  = base + off; off += s_b;
            uint8_t * p_raw  = base + off; off += p_b;
            auto *    d_oacc = reinterpret_cast<float *>(base + off); off += oa_b;
            auto *    d_ot16 = reinterpret_cast<__half *>(base + off); off += ot_b;
            auto *    d_m    = reinterpret_cast<float *>(base + off); off += ml_b;
            auto *    d_l    = reinterpret_cast<float *>(base + off); off += ml_b;

            const int64_t q_tot = static_cast<int64_t>(T) * n_head * head_dim;
            g4_init_flash_ml_kernel<<<(M_tot + 255) / 256, 256, 0, stream>>>(d_m, d_l, M_tot);
            cudaMemsetAsync(d_oacc, 0, static_cast<size_t>(M_tot) * head_dim * sizeof(float), stream);
            cublasSetStream(cublas, stream);

            if (use_fp16) {
                auto * q16 = reinterpret_cast<__half *>(q_raw);
                auto * k16 = reinterpret_cast<__half *>(k_raw);
                auto * v16 = reinterpret_cast<__half *>(v_raw);
                auto * s16 = reinterpret_cast<__half *>(s_raw);
                auto * p16 = reinterpret_cast<__half *>(p_raw);
                g4_q_pack_kernel<__half><<<static_cast<int>((q_tot + 255) / 256), 256, 0, stream>>>(
                    d_q, q16, T, n_head, n_kv, head_dim);

                const __half h_one = __float2half_rn(1.0f), h_zero = __float2half_rn(0.0f);
                const float  f_one = 1.0f, f_zero = 0.0f;

                for (int k0 = k_start; k0 < k_end; k0 += BK) {
                    const int kc = std::min(BK, k_end - k0);
                    const int t_start = contig_pos ? std::max(0, k0 - pos_min) : 0;
                    const int t_end   = (contig_pos && swa_window > 0)
                        ? std::min(T, k0 + kc - 1 + swa_window - pos_min) : T;
                    const int tc = t_end - t_start;
                    if (tc <= 0) continue;
                    const int M_cur = tc * gqa;

                    if (kv_type == KvCacheType::Q8_0) {
                        g4_dequant_kv_tile_kernel<KvCacheType::Q8_0, __half><<<dim3(n_kv, kc), dim3(32, 8), 0, stream>>>(
                            d_k_cache, d_v_cache, k16, v16, k0, kc, n_kv, head_dim, kv_cap);
                    } else if (kv_type == KvCacheType::Q4_0) {
                        g4_dequant_kv_tile_kernel<KvCacheType::Q4_0, __half><<<dim3(n_kv, kc), dim3(32, 8), 0, stream>>>(
                            d_k_cache, d_v_cache, k16, v16, k0, kc, n_kv, head_dim, kv_cap);
                    } else {
                        g4_dequant_kv_tile_kernel<KvCacheType::F16, __half><<<dim3(n_kv, kc), 256, 0, stream>>>(
                            d_k_cache, d_v_cache, k16, v16, k0, kc, n_kv, head_dim, kv_cap);
                    }

                    const __half * q_slice = q16 + static_cast<size_t>(t_start) * gqa * head_dim;
                    if (sm == 60) {
                        cublasHgemmStridedBatched(
                            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            kc, M_cur, head_dim,
                            &h_one,
                            k16, head_dim, static_cast<long long>(kc) * head_dim,
                            q_slice, head_dim, static_cast<long long>(M_per_kv) * head_dim,
                            &h_zero,
                            s16, kc, static_cast<long long>(M_cur) * kc,
                            n_kv);
                    } else {
                        cublasGemmStridedBatchedEx(
                            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            kc, M_cur, head_dim,
                            &f_one,
                            k16, CUDA_R_16F, head_dim, static_cast<long long>(kc) * head_dim,
                            q_slice, CUDA_R_16F, head_dim, static_cast<long long>(M_per_kv) * head_dim,
                            &f_zero,
                            s16, CUDA_R_16F, kc, static_cast<long long>(M_cur) * kc,
                            n_kv, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    }

                    g4_flash_softmax_swa_rescale_kernel<__half, __half><<<n_kv * M_cur, 256, 0, stream>>>(
                        s16, p16, d_pos, d_m, d_l, d_oacc, T, t_start, tc, n_head, n_kv, head_dim, k0, kc, swa_window);

                    if (sm == 60) {
                        cublasHgemmStridedBatched(
                            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                            head_dim, M_cur, kc,
                            &h_one,
                            v16, head_dim, static_cast<long long>(kc) * head_dim,
                            p16, kc, static_cast<long long>(M_cur) * kc,
                            &h_zero,
                            d_ot16, head_dim, static_cast<long long>(M_cur) * head_dim,
                            n_kv);
                        const int64_t o_n2 = (static_cast<int64_t>(n_kv) * M_cur * head_dim) >> 1;
                        g4_accum_f16_slice_to_f32_kernel<<<static_cast<int>((o_n2 + 255) / 256), 256, 0, stream>>>(
                            d_ot16, d_oacc, T, t_start, tc, gqa, n_kv, head_dim);
                    } else {
                        float * o_slice = d_oacc + static_cast<size_t>(t_start) * gqa * head_dim;
                        cublasGemmStridedBatchedEx(
                            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                            head_dim, M_cur, kc,
                            &f_one,
                            v16, CUDA_R_16F, head_dim, static_cast<long long>(kc) * head_dim,
                            p16, CUDA_R_16F, kc, static_cast<long long>(M_cur) * kc,
                            &f_one,
                            o_slice, CUDA_R_32F, head_dim, static_cast<long long>(M_per_kv) * head_dim,
                            n_kv, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    }
                }
            } else {
                // Maxwell sm_50/52 & Pascal sm_61: native FP32 cublasSgemmStridedBatched
                auto * q32 = reinterpret_cast<float *>(q_raw);
                auto * k32 = reinterpret_cast<float *>(k_raw);
                auto * v32 = reinterpret_cast<float *>(v_raw);
                auto * s32 = reinterpret_cast<float *>(s_raw);
                auto * p32 = reinterpret_cast<float *>(p_raw);
                g4_q_pack_kernel<float><<<static_cast<int>((q_tot + 255) / 256), 256, 0, stream>>>(
                    d_q, q32, T, n_head, n_kv, head_dim);

                const float f_one = 1.0f, f_zero = 0.0f;
                for (int k0 = k_start; k0 < k_end; k0 += BK) {
                    const int kc = std::min(BK, k_end - k0);
                    const int t_start = contig_pos ? std::max(0, k0 - pos_min) : 0;
                    const int t_end   = (contig_pos && swa_window > 0)
                        ? std::min(T, k0 + kc - 1 + swa_window - pos_min) : T;
                    const int tc = t_end - t_start;
                    if (tc <= 0) continue;
                    const int M_cur = tc * gqa;

                    if (kv_type == KvCacheType::Q8_0) {
                        g4_dequant_kv_tile_kernel<KvCacheType::Q8_0, float><<<dim3(n_kv, kc), dim3(32, 8), 0, stream>>>(
                            d_k_cache, d_v_cache, k32, v32, k0, kc, n_kv, head_dim, kv_cap);
                    } else if (kv_type == KvCacheType::Q4_0) {
                        g4_dequant_kv_tile_kernel<KvCacheType::Q4_0, float><<<dim3(n_kv, kc), dim3(32, 8), 0, stream>>>(
                            d_k_cache, d_v_cache, k32, v32, k0, kc, n_kv, head_dim, kv_cap);
                    } else {
                        g4_dequant_kv_tile_kernel<KvCacheType::F16, float><<<dim3(n_kv, kc), 256, 0, stream>>>(
                            d_k_cache, d_v_cache, k32, v32, k0, kc, n_kv, head_dim, kv_cap);
                    }

                    const float * q_slice = q32 + static_cast<size_t>(t_start) * gqa * head_dim;
                    cublasSgemmStridedBatched(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        kc, M_cur, head_dim,
                        &f_one,
                        k32, head_dim, static_cast<long long>(kc) * head_dim,
                        q_slice, head_dim, static_cast<long long>(M_per_kv) * head_dim,
                        &f_zero,
                        s32, kc, static_cast<long long>(M_cur) * kc,
                        n_kv);

                    g4_flash_softmax_swa_rescale_kernel<float, float><<<n_kv * M_cur, 256, 0, stream>>>(
                        s32, p32, d_pos, d_m, d_l, d_oacc, T, t_start, tc, n_head, n_kv, head_dim, k0, kc, swa_window);

                    float * o_slice = d_oacc + static_cast<size_t>(t_start) * gqa * head_dim;
                    cublasSgemmStridedBatched(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        head_dim, M_cur, kc,
                        &f_one,
                        v32, head_dim, static_cast<long long>(kc) * head_dim,
                        p32, kc, static_cast<long long>(M_cur) * kc,
                        &f_one,
                        o_slice, head_dim, static_cast<long long>(M_per_kv) * head_dim,
                        n_kv);
                }
            }

            g4_flash_finalize_unpack_kernel<<<static_cast<int>((q_tot + 255) / 256), 256, 0, stream>>>(
                d_oacc, d_l, d_out, T, n_head, n_kv, head_dim);
            return;
        }
    }

    // B. Coalesced Shared-KV Split-K MQA/GQA Decode (CUDA-Graph-safe, fixed SPLITS!)
    auto run_decode = [&](auto d_tag, auto s_tag) {
        constexpr int HD = decltype(d_tag)::value;
        constexpr int SP = decltype(s_tag)::value;
        const size_t pm_b = align256(static_cast<size_t>(T) * n_head * SP * sizeof(float));
        const size_t pl_b = pm_b;
        const size_t po_b = align256(static_cast<size_t>(T) * n_head * SP * HD * sizeof(float));
        if (pm_b + pl_b + po_b > scratch_bytes) {
            throw std::runtime_error("launch_gemma4_attn: scratch_bytes too small for split-K decode");
        }
        auto * base = static_cast<uint8_t *>(d_scratch);
        auto * pm   = reinterpret_cast<float *>(base);
        auto * pl   = reinterpret_cast<float *>(base + pm_b);
        auto * po   = reinterpret_cast<float *>(base + pm_b + pl_b);

        const int gqa = n_head / n_kv;
        const int g_blks = (gqa + 7) / 8;
        dim3 grid(n_kv * g_blks, SP, T);
        if (kv_type == KvCacheType::Q8_0) {
            g4_mqa_q8_decode_kernel<HD, SP><<<grid, dim3(32, 8), 0, stream>>>(
                d_q, d_k_cache, d_v_cache,
                d_pos, pm, pl, po, n_head, n_kv, kv_cap, swa_window);
        } else if (kv_type == KvCacheType::Q4_0) {
            g4_mqa_q4_decode_kernel<HD, SP><<<grid, dim3(32, 8), 0, stream>>>(
                d_q, d_k_cache, d_v_cache,
                d_pos, pm, pl, po, n_head, n_kv, kv_cap, swa_window);
        } else {
            g4_mqa_f16_decode_kernel<HD, SP><<<grid, dim3(32, 8), 0, stream>>>(
                d_q, static_cast<const __half *>(d_k_cache), static_cast<const __half *>(d_v_cache),
                d_pos, pm, pl, po, n_head, n_kv, kv_cap, swa_window);
        }
        g4_splitk_reduce_kernel<SP><<<dim3(n_head, T), 256, 0, stream>>>(pm, pl, po, d_out, n_head, HD);
    };

    if (head_dim == 256) {
        if (swa_window > 0) run_decode(std::integral_constant<int, 256>{}, std::integral_constant<int, 16>{});
        else                run_decode(std::integral_constant<int, 256>{}, std::integral_constant<int, 64>{});
    } else if (head_dim == 512) {
        if (swa_window > 0) run_decode(std::integral_constant<int, 512>{}, std::integral_constant<int, 16>{});
        else                run_decode(std::integral_constant<int, 512>{}, std::integral_constant<int, 64>{});
    } else {
        throw std::runtime_error("launch_gemma4_attn: unsupported head_dim (expected 256 or 512)");
    }
}

} // namespace rookery
