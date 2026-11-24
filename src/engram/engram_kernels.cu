#include "engram/engram_table.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace rookery {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// Step 1 (low-rank): z[T, rank] = e_raw[T, d_engram] @ A^T  where A is [rank, d_engram]
// If d_params != nullptr and d_params[t].scale == 0.0f, returns immediately (zero-cost per-token disable inside CUDA Graphs!).
__global__ void engram_matvec_f32_kernel(
    const float *               __restrict__ x,   // [T, K]
    const float *               __restrict__ W,   // [M, K]
    float *                     __restrict__ y,   // [T, M]
    const EngramRuntimeParams * __restrict__ d_params, // [T] or nullptr
    int M, int K) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int t   = blockIdx.y;
    if (row >= M) return;
    if (d_params != nullptr && d_params[t].scale == 0.0f) return;

    const int lane = threadIdx.x;
    const float * xp = x + static_cast<size_t>(t) * K;
    const float * wp = W + static_cast<size_t>(row) * K;

    float acc = 0.0f;
    if ((K & 3) == 0) {
        const float4 * xp4 = reinterpret_cast<const float4 *>(xp);
        const float4 * wp4 = reinterpret_cast<const float4 *>(wp);
        const int K4 = K >> 2;
        for (int k = lane; k < K4; k += 32) {
            const float4 xv = xp4[k];
            const float4 wv = __ldg(&wp4[k]);
            acc += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
        }
    } else {
        for (int k = lane; k < K; k += 32) {
            acc += xp[k] * __ldg(&wp[k]);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[static_cast<size_t>(t) * M + row] = acc;
}

// Step 2: Gated or direct residual injection into mHC streams d_inpL[T, hc, d_host].
// Reads per-token CUDA-Graph-safe EngramRuntimeParams d_params[t]:
//   - p.scale == 0.0f: exact bit-for-bit no-op (no CUDA Graph recapture needed!)
//   - p.gate_bias: additive logit bias inside sigmoid gate
//   - p.delta_clip_ratio > 0: clamps |g| so ||g * v||_rms <= ratio * ||h||_rms
__global__ void engram_gated_inject_kernel(
    float *                     __restrict__ d_inpL,   // [T, hc, d_host]
    const float *               __restrict__ d_val,    // [T, d_host]
    const float *               __restrict__ d_key,    // [T, d_host] or nullptr
    const float *               __restrict__ d_gate_w, // [hc, d_host] or nullptr
    const EngramRuntimeParams * __restrict__ d_params, // [T]
    int d_host,
    int hc,
    float rms_eps) {

    const int s   = blockIdx.x; // 0..hc-1
    const int t   = blockIdx.y;
    const EngramRuntimeParams p = d_params[t];
    if (p.scale == 0.0f) return;

    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float * h = d_inpL + (static_cast<size_t>(t) * hc + s) * d_host;
    const float * v = d_val + static_cast<size_t>(t) * d_host;

    __shared__ float s_h[8], s_k[8], s_v[8], s_dot[8];
    const bool need_gate = (d_key != nullptr);
    const bool need_clip = (p.delta_clip_ratio > 0.0f);

    float h_ss_tot = 0.0f;
    float v_ss_tot = 0.0f;
    float g = p.scale;

    if (need_gate || need_clip) {
        const float * k  = need_gate ? (d_key + static_cast<size_t>(t) * d_host) : nullptr;
        float h_ss = 0.0f, k_ss = 0.0f, v_ss = 0.0f;
        for (int d = tid; d < d_host; d += 256) {
            const float hv = h[d];
            h_ss += hv * hv;
            if (need_gate) { const float kv = k[d]; k_ss += kv * kv; }
            if (need_clip) { const float vv = v[d]; v_ss += vv * vv; }
        }
        h_ss = warp_sum(h_ss);
        if (need_gate) k_ss = warp_sum(k_ss);
        if (need_clip) v_ss = warp_sum(v_ss);

        if (lane == 0) {
            s_h[warp] = h_ss;
            s_k[warp] = k_ss;
            s_v[warp] = v_ss;
        }
        __syncthreads();

        if (warp == 0) {
            float hs = warp_sum((lane < 8) ? s_h[lane] : 0.0f);
            float ks = need_gate ? warp_sum((lane < 8) ? s_k[lane] : 0.0f) : 0.0f;
            float vs = need_clip ? warp_sum((lane < 8) ? s_v[lane] : 0.0f) : 0.0f;
            if (lane == 0) {
                s_h[0] = hs;
                s_k[0] = need_gate ? rsqrtf(ks / static_cast<float>(d_host) + rms_eps) : 0.0f;
                s_v[0] = vs;
            }
        }
        __syncthreads();
        h_ss_tot = s_h[0];
        v_ss_tot = s_v[0];

        if (need_gate) {
            const float inv_h = rsqrtf(h_ss_tot / static_cast<float>(d_host) + rms_eps);
            const float inv_k = s_k[0];
            const float * gw  = d_gate_w ? (d_gate_w + static_cast<size_t>(s) * d_host) : nullptr;

            float dot = 0.0f;
            for (int d = tid; d < d_host; d += 256) {
                const float w = gw ? gw[d] : 1.0f;
                dot += (h[d] * inv_h) * (k[d] * inv_k) * w;
            }
            dot = warp_sum(dot);
            if (lane == 0) s_dot[warp] = dot;
            __syncthreads();

            if (warp == 0) {
                float dt = warp_sum((lane < 8) ? s_dot[lane] : 0.0f);
                if (lane == 0) {
                    const float scaled = dt * rsqrtf(static_cast<float>(d_host));
                    const float mag    = sqrtf(fmaxf(fabsf(scaled), 1e-6f));
                    const float sgn    = (scaled >= 0.0f) ? 1.0f : -1.0f;
                    const float sig    = 1.0f / (1.0f + expf(-(sgn * mag + p.gate_bias)));
                    s_dot[0] = p.scale * sig;
                }
            }
            __syncthreads();
            g = s_dot[0];
        }

        if (need_clip) {
            const float h_rms = sqrtf(h_ss_tot / static_cast<float>(d_host) + rms_eps);
            const float v_rms = sqrtf(v_ss_tot / static_cast<float>(d_host) + rms_eps);
            const float max_g = p.delta_clip_ratio * h_rms / fmaxf(v_rms, 1e-6f);
            if (fabsf(g) > max_g) g = copysignf(max_g, g);
        }
    }

    for (int d = tid; d < d_host; d += 256) {
        h[d] += g * v[d];
    }
}

// Compute mean across hc streams: out[T, d_host] = mean_s d_inpL[T, s, d_host] (for calibration dumps)
__global__ void engram_mean_streams_kernel(
    const float * __restrict__ d_inpL,
    float *       __restrict__ d_out,
    int d_host, int hc) {

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= d_host) return;
    float sum = 0.0f;
    for (int s = 0; s < hc; ++s) {
        sum += d_inpL[(static_cast<size_t>(t) * hc + s) * d_host + d];
    }
    d_out[static_cast<size_t>(t) * d_host + d] = sum / static_cast<float>(hc);
}

} // namespace

void launch_engram_matvec_f32(
    const float * d_x, const float * d_W, float * d_y,
    const EngramRuntimeParams * d_params,
    int T, int M, int K, cudaStream_t stream) {
    constexpr int wpb = 8;
    dim3 grid((M + wpb - 1) / wpb, T);
    dim3 block(32, wpb);
    engram_matvec_f32_kernel<<<grid, block, 0, stream>>>(d_x, d_W, d_y, d_params, M, K);
}

void launch_engram_gated_inject(
    float * d_inpL, const float * d_val, const float * d_key, const float * d_gate_w,
    const EngramRuntimeParams * d_params, int T, int d_host, int hc, float rms_eps, cudaStream_t stream) {
    dim3 grid(hc, T);
    engram_gated_inject_kernel<<<grid, 256, 0, stream>>>(
        d_inpL, d_val, d_key, d_gate_w, d_params, d_host, hc, rms_eps);
}

void launch_engram_mean_streams(
    const float * d_inpL, float * d_out, int T, int d_host, int hc, cudaStream_t stream) {
    dim3 grid((d_host + 255) / 256, T);
    engram_mean_streams_kernel<<<grid, 256, 0, stream>>>(d_inpL, d_out, d_host, hc);
}

} // namespace rookery
