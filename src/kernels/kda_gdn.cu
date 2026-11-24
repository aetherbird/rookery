#include "kernels/kernels.h"

#include <cmath>
#include <cstdint>

namespace rookery {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// Causal depthwise Conv1D (d_conv = 4) + SiLU + in-place state update.
// `conv_state` holds the last (d_conv - 1) = 3 input vectors: [3, C].
// Processes all T tokens sequentially per channel `c` (C = 8192 -> 32 blocks of 256 threads!).
// Works identically for prefill T >= 1 and decode T = 1!
__global__ void ssm_conv1d_silu_kernel(
    const float * __restrict__ x,          // [T, C]
    const float * __restrict__ w,          // [C, 4] (ggml layout: [d_conv=4, 1, C] -> w[c*4 + k])
    float *       __restrict__ conv_state, // [3, C] (oldest=0, ..., newest=2)
    float *       __restrict__ out,        // [T, C]
    float *       __restrict__ conv_snap,  // optional [3, C]: saves post-t=0 state when T==2!
    int T, int C) {

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float s0 = conv_state[0 * C + c];
    float s1 = conv_state[1 * C + c];
    float s2 = conv_state[2 * C + c];

    const float w0 = w[c * 4 + 0];
    const float w1 = w[c * 4 + 1];
    const float w2 = w[c * 4 + 2];
    const float w3 = w[c * 4 + 3];

    for (int t = 0; t < T; ++t) {
        const float xt = x[static_cast<size_t>(t) * C + c];
        const float acc = s0 * w0 + s1 * w1 + s2 * w2 + xt * w3;
        out[static_cast<size_t>(t) * C + c] = acc / (1.0f + expf(-acc)); // SiLU
        s0 = s1;
        s1 = s2;
        s2 = xt;
        if (t == 0 && conv_snap != nullptr) {
            conv_snap[0 * C + c] = s0;
            conv_snap[1 * C + c] = s1;
            conv_snap[2 * C + c] = s2;
        }
    }

    conv_state[0 * C + c] = s0;
    conv_state[1 * C + c] = s1;
    conv_state[2 * C + c] = s2;
}

// Per-head L2 normalization in-place: x[t, h, d] /= sqrt(sum_d x^2 + eps)
// Matches ggml_l2_norm(..., 1e-6f) for head_dim = 128 (1 warp per head, 4 dims/lane!).
__global__ void l2_norm_d128_kernel(float * __restrict__ x, int n_heads, float eps) {
    const int h    = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    if (h >= n_heads) return;

    const int lane = threadIdx.x;
    auto * x4 = reinterpret_cast<float4 *>(x + (static_cast<size_t>(t) * n_heads + h) * 128);
    float4 v  = x4[lane];
    float ss  = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    float inv = rsqrtf(fmaxf(__shfl_sync(0xffffffffu, ss, 0), eps * eps));
    x4[lane]  = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
}

// Fused KDA gate + beta:
// g[t, h, d] = lower_bound * sigmoid(-(f[t, h, d] + dt_bias[h, d]) * ssm_a[h])
// (Note: ssm_a in GLM-5.3 holds -exp(A_log), so -(x * ssm_a) = +exp(A_log) * x!)
template <bool NEGATE_A>
__global__ void kda_gate_kernel(
    float *       __restrict__ f,       // [T, n_heads * head_dim] in-place -> g
    const float * __restrict__ dt_bias, // [n_heads * head_dim]
    const float * __restrict__ ssm_a,   // [n_heads]
    float lower_bound,
    int n_heads, int head_dim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int t   = blockIdx.y;
    const int D   = n_heads * head_dim;
    if (idx >= D) return;

    const int h = idx / head_dim;
    const size_t off = static_cast<size_t>(t) * D + idx;
    const float val = (f[off] + dt_bias[idx]) * ssm_a[h];
    if constexpr (NEGATE_A) {
        f[off] = lower_bound / (1.0f + expf(val));  // lower_bound * sigmoid(-val) (GLM-5.3)
    } else {
        f[off] = lower_bound / (1.0f + expf(-val)); // lower_bound * sigmoid(+val) (BailingMoE3)
    }
}

__global__ void sigmoid_inplace_kernel(float * __restrict__ x, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (1.0f + expf(-x[i]));
}

// Exact warp-parallel KDA / GDN Delta-Rule Recurrence (head_dim = 128)!
// State layout: S[h, col, row] -> [n_heads, 128, 128] (col contiguous across 128 rows).
// Each warp owns 1 column `col` (0..127) of head `h` (0..n_heads-1), 4 rows per lane in registers!
// Grid: (n_heads, 128 / wpb), Block: (32, wpb) -> 64 * 32 = 2,048 blocks -> 100% SM saturation!
template <bool KDA>
__global__ void gated_delta_net_d128_kernel(
    const float * __restrict__ q,          // [T, n_heads, 128]
    const float * __restrict__ k,          // [T, n_heads, 128]
    const float * __restrict__ v,          // [T, n_heads, 128]
    const float * __restrict__ g,          // KDA: [T, n_heads, 128]; !KDA: [T, n_heads]
    const float * __restrict__ beta,       // [T, n_heads]
    float *       __restrict__ state,      // [n_heads, 128, 128]
    float *       __restrict__ out,        // [T, n_heads, 128]
    float *       __restrict__ state_snap, // optional [n_heads, 128, 128]: saves post-t=0 state when T==2!
    int T, int n_heads, float scale) {

    const int h    = blockIdx.x;
    const int col  = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x; // 0..31 -> rows lane, lane+32, lane+64, lane+96
    if (col >= 128) return;

    const size_t col_off = (static_cast<size_t>(h) * 128 + col) * 128;
    float * s_col = state + col_off;
    float s0 = s_col[lane +  0];
    float s1 = s_col[lane + 32];
    float s2 = s_col[lane + 64];
    float s3 = s_col[lane + 96];

    for (int t = 0; t < T; ++t) {
        const size_t h_off = (static_cast<size_t>(t) * n_heads + h) * 128;
        const float * qt = q + h_off;
        const float * kt = k + h_off;
        const float * vt = v + h_off;

        const float k0 = kt[lane + 0], k1 = kt[lane + 32], k2 = kt[lane + 64], k3 = kt[lane + 96];
        const float q0 = qt[lane + 0], q1 = qt[lane + 32], q2 = qt[lane + 64], q3 = qt[lane + 96];
        const float bval = beta[static_cast<size_t>(t) * n_heads + h];

        if constexpr (KDA) {
            const float * gt = g + h_off;
            const float d0 = expf(gt[lane +  0]);
            const float d1 = expf(gt[lane + 32]);
            const float d2 = expf(gt[lane + 64]);
            const float d3 = expf(gt[lane + 96]);

            s0 *= d0; s1 *= d1; s2 *= d2; s3 *= d3;
            float kv = warp_sum(s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3);
            kv = __shfl_sync(0xffffffffu, kv, 0);

            const float delta = (vt[col] - kv) * bval;
            s0 += k0 * delta; s1 += k1 * delta; s2 += k2 * delta; s3 += k3 * delta;

            float attn = warp_sum(s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3);
            if (lane == 0) out[h_off + col] = attn * scale;
        } else {
            const float d = expf(g[static_cast<size_t>(t) * n_heads + h]);
            s0 *= d; s1 *= d; s2 *= d; s3 *= d;
            float kv = warp_sum(s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3);
            kv = __shfl_sync(0xffffffffu, kv, 0);

            const float delta = (vt[col] - kv) * bval;
            s0 += k0 * delta; s1 += k1 * delta; s2 += k2 * delta; s3 += k3 * delta;

            float attn = warp_sum(s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3);
            if (lane == 0) out[h_off + col] = attn * scale;
        }

        if (t == 0 && state_snap != nullptr) {
            float * sn_col = state_snap + col_off;
            sn_col[lane +  0] = s0;
            sn_col[lane + 32] = s1;
            sn_col[lane + 64] = s2;
            sn_col[lane + 96] = s3;
        }
    }

    s_col[lane +  0] = s0;
    s_col[lane + 32] = s1;
    s_col[lane + 64] = s2;
    s_col[lane + 96] = s3;
}

// Fused per-head RMSNorm(o, norm_w, eps) * sigmoid(gate) (in-place on o, head_dim = 128)
__global__ void rms_norm_sigmoid_gate_d128_kernel(
    float *       __restrict__ o,      // [T, n_heads, 128]
    const float * __restrict__ norm_w, // [128]
    const float * __restrict__ gate,   // [T, n_heads, 128]
    int n_heads, float eps) {

    const int h    = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    if (h >= n_heads) return;

    const int lane = threadIdx.x;
    const size_t off = (static_cast<size_t>(t) * n_heads + h) * 128;
    auto * o4 = reinterpret_cast<float4 *>(o + off);
    const auto * g4 = reinterpret_cast<const float4 *>(gate + off);
    const auto * w4 = reinterpret_cast<const float4 *>(norm_w);

    float4 ov = o4[lane];
    float ss  = warp_sum(ov.x * ov.x + ov.y * ov.y + ov.z * ov.z + ov.w * ov.w);
    float inv = rsqrtf(__shfl_sync(0xffffffffu, ss, 0) * (1.0f / 128.0f) + eps);
    float4 wv = w4[lane];
    float4 gv = g4[lane];

    o4[lane] = make_float4(
        ov.x * inv * wv.x / (1.0f + expf(-gv.x)),
        ov.y * inv * wv.y / (1.0f + expf(-gv.y)),
        ov.z * inv * wv.z / (1.0f + expf(-gv.z)),
        ov.w * inv * wv.w / (1.0f + expf(-gv.w)));
}

} // namespace

void launch_ssm_conv1d_silu(
    const float * d_x, const float * d_w, float * d_state, float * d_out,
    int T, int C, cudaStream_t stream, float * d_state_snap) {
    ssm_conv1d_silu_kernel<<<(C + 255) / 256, 256, 0, stream>>>(d_x, d_w, d_state, d_out, d_state_snap, T, C);
}

void launch_l2_norm_d128(float * d_x, int T, int n_heads, float eps, cudaStream_t stream) {
    constexpr int wpb = 4;
    l2_norm_d128_kernel<<<dim3((n_heads + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(d_x, n_heads, eps);
}

void launch_kda_gate_and_beta(
    float * d_f, const float * d_dt_bias, const float * d_ssm_a,
    float * d_beta, float lower_bound, int T, int n_heads, int head_dim, cudaStream_t stream,
    bool negate_a) {
    const int D = n_heads * head_dim;
    if (negate_a) {
        kda_gate_kernel<true><<<dim3((D + 255) / 256, T), 256, 0, stream>>>(d_f, d_dt_bias, d_ssm_a, lower_bound, n_heads, head_dim);
    } else {
        kda_gate_kernel<false><<<dim3((D + 255) / 256, T), 256, 0, stream>>>(d_f, d_dt_bias, d_ssm_a, lower_bound, n_heads, head_dim);
    }
    const int64_t nb = static_cast<int64_t>(T) * n_heads;
    sigmoid_inplace_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(d_beta, nb);
}

void launch_gated_delta_net_d128(
    const float * d_q, const float * d_k, const float * d_v,
    const float * d_g, const float * d_beta,
    float * d_state, float * d_out,
    int T, int n_heads, float scale, bool is_kda, cudaStream_t stream,
    float * d_state_snap) {
    constexpr int wpb = 4;
    dim3 grid(n_heads, 128 / wpb);
    dim3 block(32, wpb);
    if (is_kda) {
        gated_delta_net_d128_kernel<true><<<grid, block, 0, stream>>>(d_q, d_k, d_v, d_g, d_beta, d_state, d_out, d_state_snap, T, n_heads, scale);
    } else {
        gated_delta_net_d128_kernel<false><<<grid, block, 0, stream>>>(d_q, d_k, d_v, d_g, d_beta, d_state, d_out, d_state_snap, T, n_heads, scale);
    }
}

void launch_rms_norm_sigmoid_gate_d128(
    float * d_o, const float * d_norm_w, const float * d_gate,
    int T, int n_heads, float eps, cudaStream_t stream) {
    constexpr int wpb = 4;
    rms_norm_sigmoid_gate_d128_kernel<<<dim3((n_heads + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(d_o, d_norm_w, d_gate, n_heads, eps);
}

} // namespace rookery
