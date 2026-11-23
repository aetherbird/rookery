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
__global__ void engram_matvec_f32_kernel(
    const float * __restrict__ x,   // [T, K]
    const float * __restrict__ W,   // [M, K]
    float *       __restrict__ y,   // [T, M]
    int M, int K) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int t   = blockIdx.y;
    if (row >= M) return;

    const int lane = threadIdx.x;
    const float * xp = x + static_cast<size_t>(t) * K;
    const float * wp = W + static_cast<size_t>(row) * K;

    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) {
        acc += xp[k] * wp[k];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[static_cast<size_t>(t) * M + row] = acc;
}

// Step 2: Gated or direct residual injection into mHC streams d_inpL[T, hc, d_host].
// When `d_key == nullptr`: direct scaled residual `h[t, s, d] += scale * v[t, d]`.
// When `d_key != nullptr`: exact DeepSeek-V4.1 / HY4 context-dependent gate per stream s:
//   dot = sum_d (RMSNorm(h[t,s])[d] * RMSNorm(k[t])[d] * gate_w[s, d]) / sqrt(d_host)
//   g   = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
//   h[t, s, d] += scale * g * v[t, d]
__global__ void engram_gated_inject_kernel(
    float *       __restrict__ d_inpL,   // [T, hc, d_host]
    const float * __restrict__ d_val,    // [T, d_host]
    const float * __restrict__ d_key,    // [T, d_host] or nullptr
    const float * __restrict__ d_gate_w, // [hc, d_host] or nullptr
    float scale,
    int d_host,
    int hc,
    float rms_eps) {

    const int s   = blockIdx.x; // 0..hc-1
    const int t   = blockIdx.y;
    const int tid = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float * h = d_inpL + (static_cast<size_t>(t) * hc + s) * d_host;
    const float * v = d_val + static_cast<size_t>(t) * d_host;

    float g = scale;
    if (d_key != nullptr) {
        const float * k  = d_key + static_cast<size_t>(t) * d_host;
        const float * gw = d_gate_w ? (d_gate_w + static_cast<size_t>(s) * d_host) : nullptr;

        float h_ss = 0.0f, k_ss = 0.0f;
        for (int d = tid; d < d_host; d += 256) {
            const float hv = h[d];
            const float kv = k[d];
            h_ss += hv * hv;
            k_ss += kv * kv;
        }
        h_ss = warp_sum(h_ss);
        k_ss = warp_sum(k_ss);

        __shared__ float s_h[8], s_k[8], s_dot[8];
        if (lane == 0) { s_h[warp] = h_ss; s_k[warp] = k_ss; }
        __syncthreads();

        float inv_h = 0.0f, inv_k = 0.0f;
        if (warp == 0) {
            float hs = (lane < 8) ? s_h[lane] : 0.0f;
            float ks = (lane < 8) ? s_k[lane] : 0.0f;
            hs = warp_sum(hs);
            ks = warp_sum(ks);
            if (lane == 0) {
                s_h[0] = rsqrtf(hs / static_cast<float>(d_host) + rms_eps);
                s_k[0] = rsqrtf(ks / static_cast<float>(d_host) + rms_eps);
            }
        }
        __syncthreads();
        inv_h = s_h[0];
        inv_k = s_k[0];

        float dot = 0.0f;
        for (int d = tid; d < d_host; d += 256) {
            float w = gw ? gw[d] : 1.0f;
            dot += (h[d] * inv_h) * (k[d] * inv_k) * w;
        }
        dot = warp_sum(dot);
        if (lane == 0) s_dot[warp] = dot;
        __syncthreads();

        if (warp == 0) {
            float dt = (lane < 8) ? s_dot[lane] : 0.0f;
            dt = warp_sum(dt);
            if (lane == 0) {
                const float scaled = dt * rsqrtf(static_cast<float>(d_host));
                const float mag    = sqrtf(fmaxf(fabsf(scaled), 1e-6f));
                const float sgn    = (scaled >= 0.0f) ? 1.0f : -1.0f;
                const float sig    = 1.0f / (1.0f + expf(-(sgn * mag)));
                s_dot[0] = scale * sig;
            }
        }
        __syncthreads();
        g = s_dot[0];
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
    int T, int M, int K, cudaStream_t stream) {
    constexpr int wpb = 8;
    dim3 grid((M + wpb - 1) / wpb, T);
    dim3 block(32, wpb);
    engram_matvec_f32_kernel<<<grid, block, 0, stream>>>(d_x, d_W, d_y, M, K);
}

void launch_engram_gated_inject(
    float * d_inpL, const float * d_val, const float * d_key, const float * d_gate_w,
    float scale, int T, int d_host, int hc, float rms_eps, cudaStream_t stream) {
    dim3 grid(hc, T);
    engram_gated_inject_kernel<<<grid, 256, 0, stream>>>(
        d_inpL, d_val, d_key, d_gate_w, scale, d_host, hc, rms_eps);
}

void launch_engram_mean_streams(
    const float * d_inpL, float * d_out, int T, int d_host, int hc, cudaStream_t stream) {
    dim3 grid((d_host + 255) / 256, T);
    engram_mean_streams_kernel<<<grid, 256, 0, stream>>>(d_inpL, d_out, d_host, hc);
}

} // namespace rookery
