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

// Parallel 2D Causal Depthwise Conv1D (d_conv = 4) + SiLU for prefill (T >= 8 -> 100% SM saturation!)
__global__ void ssm_conv1d_silu_prefill_kernel(
    const float * __restrict__ x,          // [T, C]
    const float * __restrict__ w,          // [C, 4]
    const float * __restrict__ conv_state, // [3, C] (read-only during conv pass!)
    float *       __restrict__ out,        // [T, C]
    int T, int C) {

    const int c  = blockIdx.x * blockDim.x + threadIdx.x;
    const int t0 = blockIdx.y * 8;
    if (c >= C || t0 >= T) return;

    const float w0 = __ldg(&w[c * 4 + 0]);
    const float w1 = __ldg(&w[c * 4 + 1]);
    const float w2 = __ldg(&w[c * 4 + 2]);
    const float w3 = __ldg(&w[c * 4 + 3]);

    float s0 = (t0 >= 3) ? x[static_cast<size_t>(t0 - 3) * C + c] : conv_state[static_cast<size_t>(t0) * C + c];
    float s1 = (t0 >= 2) ? x[static_cast<size_t>(t0 - 2) * C + c] : conv_state[static_cast<size_t>(t0 + 1) * C + c];
    float s2 = (t0 >= 1) ? x[static_cast<size_t>(t0 - 1) * C + c] : conv_state[static_cast<size_t>(t0 + 2) * C + c];

    const int t1 = (t0 + 8 < T) ? (t0 + 8) : T;
    for (int t = t0; t < t1; ++t) {
        const float xt  = x[static_cast<size_t>(t) * C + c];
        const float acc = s0 * w0 + s1 * w1 + s2 * w2 + xt * w3;
        out[static_cast<size_t>(t) * C + c] = acc / (1.0f + expf(-acc));
        s0 = s1;
        s1 = s2;
        s2 = xt;
    }
}

__global__ void ssm_conv1d_update_state_kernel(
    const float * __restrict__ x,          // [T, C], T >= 3
    float *       __restrict__ conv_state, // [3, C]
    int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    conv_state[0 * C + c] = x[static_cast<size_t>(T - 3) * C + c];
    conv_state[1 * C + c] = x[static_cast<size_t>(T - 2) * C + c];
    conv_state[2 * C + c] = x[static_cast<size_t>(T - 1) * C + c];
}

// Fused Q/K/V Conv1D + SiLU + per-head L2-Norm(Q, K) in ONE launch for decode (T <= 4)!
// Grid: (n_heads, 3) [which = 0:Q, 1:K, 2:V], Block: 128 threads (1 thread/dim in head_dim=128)
__global__ void ssm_conv1d_silu_qkv_l2_decode_kernel(
    const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
    const float * __restrict__ wq, const float * __restrict__ wk, const float * __restrict__ wv,
    float * __restrict__ sq, float * __restrict__ sk, float * __restrict__ sv,
    float * __restrict__ qc, float * __restrict__ kc, float * __restrict__ vc,
    float * __restrict__ sq_snap, float * __restrict__ sk_snap, float * __restrict__ sv_snap,
    int T, int n_heads, float eps) {

    const int h     = blockIdx.x;
    const int which = blockIdx.y; // 0:Q, 1:K, 2:V
    const int d     = threadIdx.x; // 0..127
    const int C     = n_heads * 128;
    const int c     = h * 128 + d;

    const float * x     = (which == 0) ? q  : ((which == 1) ? k  : v);
    const float * w     = (which == 0) ? wq : ((which == 1) ? wk : wv);
    float *       state = (which == 0) ? sq : ((which == 1) ? sk : sv);
    float *       out   = (which == 0) ? qc : ((which == 1) ? kc : vc);
    float *       snap  = (which == 0) ? sq_snap : ((which == 1) ? sk_snap : sv_snap);

    float s0 = state[0 * C + c];
    float s1 = state[1 * C + c];
    float s2 = state[2 * C + c];

    const float w0 = __ldg(&w[c * 4 + 0]);
    const float w1 = __ldg(&w[c * 4 + 1]);
    const float w2 = __ldg(&w[c * 4 + 2]);
    const float w3 = __ldg(&w[c * 4 + 3]);

    __shared__ float s_red[4];
    const int lane = d & 31;
    const int warp = d >> 5;

    for (int t = 0; t < T; ++t) {
        const float xt  = x[static_cast<size_t>(t) * C + c];
        const float acc = s0 * w0 + s1 * w1 + s2 * w2 + xt * w3;
        float val = acc / (1.0f + expf(-acc));
        s0 = s1;
        s1 = s2;
        s2 = xt;
        if (t == 0 && snap != nullptr) {
            snap[0 * C + c] = s0;
            snap[1 * C + c] = s1;
            snap[2 * C + c] = s2;
        }
        if (which < 2) {
            float ss = warp_sum(val * val);
            if (lane == 0) s_red[warp] = ss;
            __syncthreads();
            if (warp == 0) {
                float tot = (lane < 4) ? s_red[lane] : 0.0f;
                tot = warp_sum(tot);
                if (lane == 0) s_red[0] = rsqrtf(fmaxf(tot, eps * eps));
            }
            __syncthreads();
            val *= s_red[0];
        }
        out[static_cast<size_t>(t) * C + c] = val;
    }

    state[0 * C + c] = s0;
    state[1 * C + c] = s1;
    state[2 * C + c] = s2;
}

// Parallel 2D Q/K/V Prefill Conv1D + SiLU in ONE launch (grid.z = 3)!
__global__ void ssm_conv1d_silu_qkv_prefill_kernel(
    const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
    const float * __restrict__ wq, const float * __restrict__ wk, const float * __restrict__ wv,
    const float * __restrict__ sq, const float * __restrict__ sk, const float * __restrict__ sv,
    float * __restrict__ qc, float * __restrict__ kc, float * __restrict__ vc,
    int T, int C) {

    const int c     = blockIdx.x * blockDim.x + threadIdx.x;
    const int t0    = blockIdx.y * 8;
    const int which = blockIdx.z;
    if (c >= C || t0 >= T) return;

    const float * x     = (which == 0) ? q  : ((which == 1) ? k  : v);
    const float * w     = (which == 0) ? wq : ((which == 1) ? wk : wv);
    const float * state = (which == 0) ? sq : ((which == 1) ? sk : sv);
    float *       out   = (which == 0) ? qc : ((which == 1) ? kc : vc);

    const float w0 = __ldg(&w[c * 4 + 0]);
    const float w1 = __ldg(&w[c * 4 + 1]);
    const float w2 = __ldg(&w[c * 4 + 2]);
    const float w3 = __ldg(&w[c * 4 + 3]);

    float s0 = (t0 >= 3) ? x[static_cast<size_t>(t0 - 3) * C + c] : state[static_cast<size_t>(t0) * C + c];
    float s1 = (t0 >= 2) ? x[static_cast<size_t>(t0 - 2) * C + c] : state[static_cast<size_t>(t0 + 1) * C + c];
    float s2 = (t0 >= 1) ? x[static_cast<size_t>(t0 - 1) * C + c] : state[static_cast<size_t>(t0 + 2) * C + c];

    const int t1 = (t0 + 8 < T) ? (t0 + 8) : T;
    for (int t = t0; t < t1; ++t) {
        const float xt  = x[static_cast<size_t>(t) * C + c];
        const float acc = s0 * w0 + s1 * w1 + s2 * w2 + xt * w3;
        out[static_cast<size_t>(t) * C + c] = acc / (1.0f + expf(-acc));
        s0 = s1;
        s1 = s2;
        s2 = xt;
    }
}

__global__ void ssm_conv1d_qkv_update_state_kernel(
    const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
    float * __restrict__ sq, float * __restrict__ sk, float * __restrict__ sv,
    int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const size_t i0 = static_cast<size_t>(T - 3) * C + c;
    const size_t i1 = static_cast<size_t>(T - 2) * C + c;
    const size_t i2 = static_cast<size_t>(T - 1) * C + c;
    sq[0 * C + c] = q[i0]; sq[1 * C + c] = q[i1]; sq[2 * C + c] = q[i2];
    sk[0 * C + c] = k[i0]; sk[1 * C + c] = k[i1]; sk[2 * C + c] = k[i2];
    sv[0 * C + c] = v[i0]; sv[1 * C + c] = v[i1]; sv[2 * C + c] = v[i2];
}

__global__ void l2_norm_d128_qk_kernel(float * __restrict__ q, float * __restrict__ k, int n_heads, float eps) {
    const int h     = blockIdx.x * blockDim.y + threadIdx.y;
    const int t     = blockIdx.y;
    const int which = blockIdx.z;
    if (h >= n_heads) return;

    const int lane = threadIdx.x;
    float * base = (which == 0) ? q : k;
    auto * x4 = reinterpret_cast<float4 *>(base + (static_cast<size_t>(t) * n_heads + h) * 128);
    float4 v  = x4[lane];
    float ss  = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    float inv = rsqrtf(fmaxf(__shfl_sync(0xffffffffu, ss, 0), eps * eps));
    x4[lane]  = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
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

// Fused KDA gate + beta in ONE launch:
// g[t, h, d] = lower_bound * sigmoid(+- (f[t, h, d] + dt_bias[h, d]) * ssm_a[h])
// beta[t, h] = sigmoid(beta[t, h])
template <bool NEGATE_A>
__global__ void kda_gate_and_beta_kernel(
    float *       __restrict__ f,       // [T, n_heads * head_dim] in-place -> g
    const float * __restrict__ dt_bias, // [n_heads * head_dim]
    const float * __restrict__ ssm_a,   // [n_heads]
    float *       __restrict__ beta,    // [T, n_heads] in-place -> sigmoid(beta)
    float lower_bound,
    int n_heads, int head_dim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int t   = blockIdx.y;
    const int D   = n_heads * head_dim;
    if (idx >= D) return;

    const int h = idx / head_dim;
    const size_t off = static_cast<size_t>(t) * D + idx;
    const float val = (f[off] + __ldg(&dt_bias[idx])) * __ldg(&ssm_a[h]);
    if constexpr (NEGATE_A) {
        f[off] = lower_bound / (1.0f + expf(val));  // lower_bound * sigmoid(-val) (GLM-5.3)
    } else {
        f[off] = lower_bound / (1.0f + expf(-val)); // lower_bound * sigmoid(+val) (BailingMoE3)
    }
    if (idx < n_heads) {
        const size_t boff = static_cast<size_t>(t) * n_heads + idx;
        beta[boff] = 1.0f / (1.0f + expf(-beta[boff]));
    }
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

// Shared-Memory Tile-Prefetched 2-Column-per-Warp KDA / GDN Prefill Kernel (T >= 8)!
// - Grid: (n_heads, 8), Block: (32, 8) = 256 thr -> 16 columns/CTA (2 cols/warp)!
// - Caches T_TILE=8 tokens of Q, K, exp(g), V, beta in 12.8 KB smem via coalesced float4 loads!
// - 16x fewer global loads, 16x fewer expf() calls, 2x ILP hiding shuffle latency!
template <bool KDA>
__global__ __launch_bounds__(256) void gated_delta_net_d128_prefill_kernel(
    const float * __restrict__ q,     // [T, n_heads, 128]
    const float * __restrict__ k,     // [T, n_heads, 128]
    const float * __restrict__ v,     // [T, n_heads, 128]
    const float * __restrict__ g,     // KDA: [T, n_heads, 128]; !KDA: [T, n_heads]
    const float * __restrict__ beta,  // [T, n_heads]
    float *       __restrict__ state, // [n_heads, 128, 128]
    float *       __restrict__ out,   // [T, n_heads, 128]
    int T, int n_heads, float scale) {

    constexpr int T_TILE = 8;
    __shared__ float s_q[T_TILE][128];
    __shared__ float s_k[T_TILE][128];
    __shared__ float s_d[T_TILE][KDA ? 128 : 4];
    __shared__ float s_v[T_TILE][16];
    __shared__ float s_beta[T_TILE];

    const int h        = blockIdx.x;
    const int blk_col0 = blockIdx.y * 16;
    const int lane     = threadIdx.x; // 0..31
    const int warp     = threadIdx.y; // 0..7
    const int tid      = warp * 32 + lane;
    const int col0     = blk_col0 + warp * 2;
    const int col1     = col0 + 1;

    float * sc0 = state + (static_cast<size_t>(h) * 128 + col0) * 128;
    float * sc1 = state + (static_cast<size_t>(h) * 128 + col1) * 128;
    float a0 = sc0[lane +  0], a1 = sc0[lane + 32], a2 = sc0[lane + 64], a3 = sc0[lane + 96];
    float b0 = sc1[lane +  0], b1 = sc1[lane + 32], b2 = sc1[lane + 64], b3 = sc1[lane + 96];

    const int load_ti = tid >> 5; // 0..7
    const int load_c4 = tid & 31; // 0..31 (32 float4 = 128 floats)

    for (int t0 = 0; t0 < T; t0 += T_TILE) {
        const int n_t = (t0 + T_TILE < T) ? T_TILE : (T - t0);
        if (load_ti < n_t) {
            const int t = t0 + load_ti;
            const size_t h_off = (static_cast<size_t>(t) * n_heads + h) * 128;
            reinterpret_cast<float4 *>(s_q[load_ti])[load_c4] = __ldg(&reinterpret_cast<const float4 *>(q + h_off)[load_c4]);
            reinterpret_cast<float4 *>(s_k[load_ti])[load_c4] = __ldg(&reinterpret_cast<const float4 *>(k + h_off)[load_c4]);
            if constexpr (KDA) {
                const float4 gv = __ldg(&reinterpret_cast<const float4 *>(g + h_off)[load_c4]);
                reinterpret_cast<float4 *>(s_d[load_ti])[load_c4] = make_float4(expf(gv.x), expf(gv.y), expf(gv.z), expf(gv.w));
            } else {
                if (load_c4 == 0) s_d[load_ti][0] = expf(g[static_cast<size_t>(t) * n_heads + h]);
            }
            if (load_c4 < 16) s_v[load_ti][load_c4] = v[h_off + blk_col0 + load_c4];
            if (load_c4 == 0) s_beta[load_ti] = beta[static_cast<size_t>(t) * n_heads + h];
        }
        __syncthreads();

        for (int ti = 0; ti < n_t; ++ti) {
            const float k0 = s_k[ti][lane +  0], k1 = s_k[ti][lane + 32], k2 = s_k[ti][lane + 64], k3 = s_k[ti][lane + 96];
            const float q0 = s_q[ti][lane +  0], q1 = s_q[ti][lane + 32], q2 = s_q[ti][lane + 64], q3 = s_q[ti][lane + 96];
            const float bval = s_beta[ti];

            if constexpr (KDA) {
                const float d0 = s_d[ti][lane +  0], d1 = s_d[ti][lane + 32], d2 = s_d[ti][lane + 64], d3 = s_d[ti][lane + 96];
                a0 *= d0; a1 *= d1; a2 *= d2; a3 *= d3;
                b0 *= d0; b1 *= d1; b2 *= d2; b3 *= d3;
            } else {
                const float d = s_d[ti][0];
                a0 *= d; a1 *= d; a2 *= d; a3 *= d;
                b0 *= d; b1 *= d; b2 *= d; b3 *= d;
            }

            float kva = a0 * k0 + a1 * k1 + a2 * k2 + a3 * k3;
            float kvb = b0 * k0 + b1 * k1 + b2 * k2 + b3 * k3;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                kva += __shfl_down_sync(0xffffffffu, kva, off);
                kvb += __shfl_down_sync(0xffffffffu, kvb, off);
            }
            kva = __shfl_sync(0xffffffffu, kva, 0);
            kvb = __shfl_sync(0xffffffffu, kvb, 0);

            const float da = (s_v[ti][warp * 2 + 0] - kva) * bval;
            const float db = (s_v[ti][warp * 2 + 1] - kvb) * bval;
            a0 += k0 * da; a1 += k1 * da; a2 += k2 * da; a3 += k3 * da;
            b0 += k0 * db; b1 += k1 * db; b2 += k2 * db; b3 += k3 * db;

            float oa = a0 * q0 + a1 * q1 + a2 * q2 + a3 * q3;
            float ob = b0 * q0 + b1 * q1 + b2 * q2 + b3 * q3;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                oa += __shfl_down_sync(0xffffffffu, oa, off);
                ob += __shfl_down_sync(0xffffffffu, ob, off);
            }
            if (lane == 0) {
                const size_t h_off = (static_cast<size_t>(t0 + ti) * n_heads + h) * 128;
                out[h_off + col0] = oa * scale;
                out[h_off + col1] = ob * scale;
            }
        }
        __syncthreads();
    }

    sc0[lane +  0] = a0; sc0[lane + 32] = a1; sc0[lane + 64] = a2; sc0[lane + 96] = a3;
    sc1[lane +  0] = b0; sc1[lane + 32] = b1; sc1[lane + 64] = b2; sc1[lane + 96] = b3;
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
    if (T >= 8 && d_state_snap == nullptr) {
        ssm_conv1d_silu_prefill_kernel<<<dim3((C + 255) / 256, (T + 7) / 8), 256, 0, stream>>>(
            d_x, d_w, d_state, d_out, T, C);
        ssm_conv1d_update_state_kernel<<<(C + 255) / 256, 256, 0, stream>>>(d_x, d_state, T, C);
        return;
    }
    ssm_conv1d_silu_kernel<<<(C + 255) / 256, 256, 0, stream>>>(d_x, d_w, d_state, d_out, d_state_snap, T, C);
}

void launch_ssm_conv1d_silu_qkv_l2(
    const float * d_q, const float * d_k, const float * d_v,
    const float * d_wq, const float * d_wk, const float * d_wv,
    float * d_sq, float * d_sk, float * d_sv,
    float * d_qc, float * d_kc, float * d_vc,
    int T, int n_heads, float eps, cudaStream_t stream,
    float * d_sq_snap, float * d_sk_snap, float * d_sv_snap) {
    const int C = n_heads * 128;
    if (T <= 4) {
        ssm_conv1d_silu_qkv_l2_decode_kernel<<<dim3(n_heads, 3), 128, 0, stream>>>(
            d_q, d_k, d_v, d_wq, d_wk, d_wv, d_sq, d_sk, d_sv,
            d_qc, d_kc, d_vc, d_sq_snap, d_sk_snap, d_sv_snap, T, n_heads, eps);
        return;
    }
    ssm_conv1d_silu_qkv_prefill_kernel<<<dim3((C + 255) / 256, (T + 7) / 8, 3), 256, 0, stream>>>(
        d_q, d_k, d_v, d_wq, d_wk, d_wv, d_sq, d_sk, d_sv, d_qc, d_kc, d_vc, T, C);
    ssm_conv1d_qkv_update_state_kernel<<<(C + 255) / 256, 256, 0, stream>>>(
        d_q, d_k, d_v, d_sq, d_sk, d_sv, T, C);
    constexpr int wpb = 4;
    l2_norm_d128_qk_kernel<<<dim3((n_heads + wpb - 1) / wpb, T, 2), dim3(32, wpb), 0, stream>>>(
        d_qc, d_kc, n_heads, eps);
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
        kda_gate_and_beta_kernel<true><<<dim3((D + 255) / 256, T), 256, 0, stream>>>(
            d_f, d_dt_bias, d_ssm_a, d_beta, lower_bound, n_heads, head_dim);
    } else {
        kda_gate_and_beta_kernel<false><<<dim3((D + 255) / 256, T), 256, 0, stream>>>(
            d_f, d_dt_bias, d_ssm_a, d_beta, lower_bound, n_heads, head_dim);
    }
}

void launch_gated_delta_net_d128(
    const float * d_q, const float * d_k, const float * d_v,
    const float * d_g, const float * d_beta,
    float * d_state, float * d_out,
    int T, int n_heads, float scale, bool is_kda, cudaStream_t stream,
    float * d_state_snap) {
    if (T >= 8 && d_state_snap == nullptr) {
        dim3 grid(n_heads, 8);
        dim3 block(32, 8);
        if (is_kda) {
            gated_delta_net_d128_prefill_kernel<true><<<grid, block, 0, stream>>>(
                d_q, d_k, d_v, d_g, d_beta, d_state, d_out, T, n_heads, scale);
        } else {
            gated_delta_net_d128_prefill_kernel<false><<<grid, block, 0, stream>>>(
                d_q, d_k, d_v, d_g, d_beta, d_state, d_out, T, n_heads, scale);
        }
        return;
    }
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
