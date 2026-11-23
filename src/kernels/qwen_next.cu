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
// 1. Qwen GDN Gate (ssm_a * softplus(alpha + dt_bias)) + Beta (sigmoid(beta))
// ============================================================================
__global__ void qwen_gdn_gate_beta_kernel(
    float *       __restrict__ alpha,
    const float * __restrict__ dt_bias,
    const float * __restrict__ ssm_a,
    float *       __restrict__ beta,
    int n_v_heads) {

    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (h >= n_v_heads) return;

    const size_t off = static_cast<size_t>(t) * n_v_heads + h;
    const float x = alpha[off] + dt_bias[h];
    const float sp = (x > 20.0f) ? x : log1pf(expf(x));
    alpha[off] = sp * ssm_a[h];
    beta[off]  = 1.0f / (1.0f + expf(-beta[off]));
}

// In-place L2 norm on Q and K slices of packed qkv_conv[T, 2*n_k*128 + n_v*128]
__global__ void qwen_qkv_l2_norm_d128_kernel(
    float * __restrict__ qkv_conv,
    int n_k_heads, int row_stride, float eps) {

    const int qk_h = blockIdx.x * blockDim.y + threadIdx.y; // 0 .. 2*n_k_heads - 1
    const int t    = blockIdx.y;
    if (qk_h >= 2 * n_k_heads) return;
    const int lane = threadIdx.x;

    auto * x4 = reinterpret_cast<float4 *>(qkv_conv + static_cast<size_t>(t) * row_stride + qk_h * 128);
    float4 v  = x4[lane];
    float ss  = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    float inv = rsqrtf(fmaxf(__shfl_sync(0xffffffffu, ss, 0), eps * eps));
    x4[lane]  = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
}

// Warp-parallel GDN Recurrence reading packed qkv_conv[T, 2*n_k*128 + n_v*128]!
// Matches ggml_gated_delta_net: k_head = h % n_k_heads (exact modulo!).
__global__ void qwen_gdn_packed_d128_kernel(
    const float * __restrict__ qkv_conv,   // [T, row_stride]
    const float * __restrict__ g,          // [T, n_v_heads]
    const float * __restrict__ beta,       // [T, n_v_heads]
    float *       __restrict__ state,      // [n_v_heads, 128, 128]
    float *       __restrict__ out,        // [T, n_v_heads, 128]
    float *       __restrict__ state_snap, // optional [n_v_heads, 128, 128]
    int T, int n_k_heads, int n_v_heads, int row_stride, float scale) {

    const int h    = blockIdx.x;
    const int col  = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (col >= 128) return;

    const int kh = h % n_k_heads;
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

        const float k0 = kt[lane + 0], k1 = kt[lane + 32], k2 = kt[lane + 64], k3 = kt[lane + 96];
        const float q0 = qt[lane + 0], q1 = qt[lane + 32], q2 = qt[lane + 64], q3 = qt[lane + 96];
        const float bval = beta[static_cast<size_t>(t) * n_v_heads + h];
        const float d    = expf(g[static_cast<size_t>(t) * n_v_heads + h]);

        s0 *= d; s1 *= d; s2 *= d; s3 *= d;
        float kv = warp_sum(s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3);
        kv = __shfl_sync(0xffffffffu, kv, 0);

        const float delta = (vt[col] - kv) * bval;
        s0 += k0 * delta; s1 += k1 * delta; s2 += k2 * delta; s3 += k3 * delta;

        float attn = warp_sum(s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3);
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

// Fused per-head RMSNorm(o, norm_w, eps) * SiLU(z) in-place on o [T, n_v_heads, 128]
__global__ void rms_norm_silu_gate_d128_kernel(
    float *       __restrict__ o,
    const float * __restrict__ norm_w,
    const float * __restrict__ gate,
    int n_v_heads, float eps) {

    const int h = blockIdx.x * blockDim.y + threadIdx.y;
    const int t = blockIdx.y;
    if (h >= n_v_heads) return;

    const int lane = threadIdx.x;
    const size_t off = (static_cast<size_t>(t) * n_v_heads + h) * 128;
    auto * o4 = reinterpret_cast<float4 *>(o + off);
    const auto * g4 = reinterpret_cast<const float4 *>(gate + off);
    const auto * w4 = reinterpret_cast<const float4 *>(norm_w);

    float4 ov = o4[lane];
    float ss  = warp_sum(ov.x * ov.x + ov.y * ov.y + ov.z * ov.z + ov.w * ov.w);
    float inv = rsqrtf(__shfl_sync(0xffffffffu, ss, 0) * (1.0f / 128.0f) + eps);
    float4 wv = w4[lane];
    float4 gv = g4[lane];

    auto silu = [](float x) { return x / (1.0f + expf(-x)); };
    o4[lane] = make_float4(
        ov.x * inv * wv.x * silu(gv.x),
        ov.y * inv * wv.y * silu(gv.y),
        ov.z * inv * wv.z * silu(gv.z),
        ov.w * inv * wv.w * silu(gv.w));
}

// ============================================================================
// 2. Qwen d256 Interleaved [Q, gate] Split + RMSNorm + Partial NeoX RoPE (rope_dim=64)
// ============================================================================
__global__ void qwen_qgate_split_norm_rope_d256_kernel(
    const float *   __restrict__ q_full,   // [T, n_head, 512]
    const float *   __restrict__ q_norm_w, // [256]
    const int32_t * __restrict__ pos,      // [T]
    float *         __restrict__ q_out,    // [T, n_head, 256]
    float *         __restrict__ gate_out, // [T, n_head, 256]
    int n_head, int rope_dim, float rms_eps, float rope_freq_base) {

    const int h    = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    if (h >= n_head) return;
    const int lane = threadIdx.x; // 0..31 -> 8 dims/lane

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
    ss = __shfl_sync(0xffffffffu, warp_sum(ss), 0);
    const float inv = rsqrtf(ss * (1.0f / 256.0f) + rms_eps);

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int d = k * 32 + lane;
        qv[k] *= inv * q_norm_w[d];
    }

    // Partial NeoX RoPE on first rope_dim=64 dims: k=0 (d=0..31) paired with k=1 (d=32..63)!
    const int half_r = rope_dim >> 1; // 32
    if (lane < half_r) {
        const float p     = static_cast<float>(pos[t]);
        const float freq  = powf(rope_freq_base, -2.0f * static_cast<float>(lane) / static_cast<float>(rope_dim));
        const float angle = p * freq;
        float s, c;
        __sincosf(angle, &s, &c);
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

__global__ void qwen_k_norm_rope_d256_kernel(
    float *         __restrict__ k_buf,    // [T, n_kv, 256]
    const float *   __restrict__ k_norm_w, // [256]
    const int32_t * __restrict__ pos,      // [T]
    int n_kv, int rope_dim, float rms_eps, float rope_freq_base) {

    const int h    = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
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
    ss = __shfl_sync(0xffffffffu, warp_sum(ss), 0);
    const float inv = rsqrtf(ss * (1.0f / 256.0f) + rms_eps);
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int d = k * 32 + lane;
        kv[k] *= inv * k_norm_w[d];
    }

    const int half_r = rope_dim >> 1; // 32
    if (lane < half_r) {
        const float p     = static_cast<float>(pos[t]);
        const float freq  = powf(rope_freq_base, -2.0f * static_cast<float>(lane) / static_cast<float>(rope_dim));
        const float angle = p * freq;
        float s, c;
        __sincosf(angle, &s, &c);
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
// 3. GQA d256 KV Cache Write + Split-K Decode + Tiled Tensor Core FlashAttention Prefill
// ============================================================================
__global__ void gqa_kv_write_q8_0_d256_kernel(
    const float *   __restrict__ k,
    const float *   __restrict__ v,
    BlockQ8_0 *     __restrict__ k_cache,
    BlockQ8_0 *     __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    int T, int n_kv) {

    const int total_b = n_kv * 8; // 8 Q8_0 blocks per 256-dim head
    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    const int t = blockIdx.y;
    if (t >= T || b >= total_b) return;

    const int lane = threadIdx.x;
    const size_t src_off = static_cast<size_t>(t) * (n_kv * 256) + b * 32 + lane;
    const size_t dst_b   = static_cast<size_t>(pos[t]) * total_b + b;

    auto qstore = [&](const float * src, BlockQ8_0 * dst) {
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
    qstore(k, k_cache);
    qstore(v, v_cache);
}

__global__ void gqa_kv_write_f16_d256_kernel(
    const float *   __restrict__ k,
    const float *   __restrict__ v,
    __half *        __restrict__ k_cache,
    __half *        __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    int T, int kv_dim) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (t >= T || d >= kv_dim) return;
    const size_t src = static_cast<size_t>(t) * kv_dim + d;
    const size_t dst = static_cast<size_t>(pos[t]) * kv_dim + d;
    k_cache[dst] = __float2half_rn(k[src]);
    v_cache[dst] = __float2half_rn(v[src]);
}

// Split-K GQA Decode (head_dim=256, 8 dims/lane per warp; safe 2-byte aligned BlockQ8_0 reads!)
template <bool KV_Q8_0, int SPLITS>
__global__ void gqa_attn_d256_splitk_kernel(
    const float *   __restrict__ q,       // [T, n_head, 256]
    const void *    __restrict__ k_cache, // [max_ctx, n_kv, 256]
    const void *    __restrict__ v_cache,
    const int32_t * __restrict__ pos,     // [T]
    float *         __restrict__ part_m,  // [T, n_head, SPLITS]
    float *         __restrict__ part_l,  // [T, n_head, SPLITS]
    float *         __restrict__ part_o,  // [T, n_head, SPLITS, 256]
    int n_head, int n_kv, float scale) {

    const int h    = blockIdx.x;
    const int s    = blockIdx.y;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x; // 0..31 -> 8 dims: lane*8 .. lane*8+7 (2 float4s)
    const int kvh  = h / (n_head / n_kv);

    const int kv_len = pos[t] + 1;
    const int chunk  = (kv_len + SPLITS - 1) / SPLITS;
    const int p0     = s * chunk;
    const int p1     = (p0 + chunk < kv_len) ? (p0 + chunk) : kv_len;

    const auto * q4 = reinterpret_cast<const float4 *>(q + (static_cast<size_t>(t) * n_head + h) * 256);
    float4 q0 = q4[lane * 2 + 0];
    float4 q1 = q4[lane * 2 + 1];
    q0.x *= scale; q0.y *= scale; q0.z *= scale; q0.w *= scale;
    q1.x *= scale; q1.y *= scale; q1.z *= scale; q1.w *= scale;

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float4 o0 = make_float4(0, 0, 0, 0);
    float4 o1 = make_float4(0, 0, 0, 0);

    const int blk_in_h = lane >> 2;      // 0..7 (8 Q8_0 blocks in 256 dims)
    const int off_in_b = (lane & 3) * 8; // 0, 8, 16, 24

    for (int p = p0; p < p1; ++p) {
        float4 k0, k1, v0, v1;
        if constexpr (KV_Q8_0) {
            const size_t bidx = (static_cast<size_t>(p) * n_kv + kvh) * 8 + blk_in_h;
            const BlockQ8_0 & kb = static_cast<const BlockQ8_0 *>(k_cache)[bidx];
            const BlockQ8_0 & vb = static_cast<const BlockQ8_0 *>(v_cache)[bidx];
            __half kdh, vdh;
            *reinterpret_cast<uint16_t *>(&kdh) = __ldg(&kb.d);
            *reinterpret_cast<uint16_t *>(&vdh) = __ldg(&vb.d);
            const float kd = __half2float(kdh), vd = __half2float(vdh);
            const uint16_t ka = __ldg(reinterpret_cast<const uint16_t *>(&kb.qs[off_in_b + 0]));
            const uint16_t kb2= __ldg(reinterpret_cast<const uint16_t *>(&kb.qs[off_in_b + 2]));
            const uint16_t kc = __ldg(reinterpret_cast<const uint16_t *>(&kb.qs[off_in_b + 4]));
            const uint16_t kd2= __ldg(reinterpret_cast<const uint16_t *>(&kb.qs[off_in_b + 6]));
            const uint16_t va = __ldg(reinterpret_cast<const uint16_t *>(&vb.qs[off_in_b + 0]));
            const uint16_t vb2= __ldg(reinterpret_cast<const uint16_t *>(&vb.qs[off_in_b + 2]));
            const uint16_t vc = __ldg(reinterpret_cast<const uint16_t *>(&vb.qs[off_in_b + 4]));
            const uint16_t vd2= __ldg(reinterpret_cast<const uint16_t *>(&vb.qs[off_in_b + 6]));
            k0 = make_float4(kd * static_cast<int8_t>(ka & 0xFF),  kd * static_cast<int8_t>(ka >> 8),
                             kd * static_cast<int8_t>(kb2 & 0xFF), kd * static_cast<int8_t>(kb2 >> 8));
            k1 = make_float4(kd * static_cast<int8_t>(kc & 0xFF),  kd * static_cast<int8_t>(kc >> 8),
                             kd * static_cast<int8_t>(kd2 & 0xFF), kd * static_cast<int8_t>(kd2 >> 8));
            v0 = make_float4(vd * static_cast<int8_t>(va & 0xFF),  vd * static_cast<int8_t>(va >> 8),
                             vd * static_cast<int8_t>(vb2 & 0xFF), vd * static_cast<int8_t>(vb2 >> 8));
            v1 = make_float4(vd * static_cast<int8_t>(vc & 0xFF),  vd * static_cast<int8_t>(vc >> 8),
                             vd * static_cast<int8_t>(vd2 & 0xFF), vd * static_cast<int8_t>(vd2 >> 8));
        } else {
            const size_t off = (static_cast<size_t>(p) * n_kv + kvh) * 64 + lane * 2;
            uint2 pk0 = __ldg(&static_cast<const uint2 *>(k_cache)[off + 0]);
            uint2 pk1 = __ldg(&static_cast<const uint2 *>(k_cache)[off + 1]);
            uint2 pv0 = __ldg(&static_cast<const uint2 *>(v_cache)[off + 0]);
            uint2 pv1 = __ldg(&static_cast<const uint2 *>(v_cache)[off + 1]);
            float2 f0 = __half22float2(*reinterpret_cast<const __half2 *>(&pk0.x));
            float2 f1 = __half22float2(*reinterpret_cast<const __half2 *>(&pk0.y));
            float2 f2 = __half22float2(*reinterpret_cast<const __half2 *>(&pk1.x));
            float2 f3 = __half22float2(*reinterpret_cast<const __half2 *>(&pk1.y));
            k0 = make_float4(f0.x, f0.y, f1.x, f1.y);
            k1 = make_float4(f2.x, f2.y, f3.x, f3.y);
            f0 = __half22float2(*reinterpret_cast<const __half2 *>(&pv0.x));
            f1 = __half22float2(*reinterpret_cast<const __half2 *>(&pv0.y));
            f2 = __half22float2(*reinterpret_cast<const __half2 *>(&pv1.x));
            f3 = __half22float2(*reinterpret_cast<const __half2 *>(&pv1.y));
            v0 = make_float4(f0.x, f0.y, f1.x, f1.y);
            v1 = make_float4(f2.x, f2.y, f3.x, f3.y);
        }

        float dot = warp_sum(q0.x*k0.x + q0.y*k0.y + q0.z*k0.z + q0.w*k0.w +
                             q1.x*k1.x + q1.y*k1.y + q1.z*k1.z + q1.w*k1.w);
        const float score = __shfl_sync(0xffffffffu, dot, 0);
        const float m_new = fmaxf(m_i, score);
        const float alpha = (m_i == -FLT_MAX) ? 0.0f : expf(m_i - m_new);
        const float beta  = expf(score - m_new);
        l_i = l_i * alpha + beta;
        o0.x = o0.x * alpha + beta * v0.x; o0.y = o0.y * alpha + beta * v0.y;
        o0.z = o0.z * alpha + beta * v0.z; o0.w = o0.w * alpha + beta * v0.w;
        o1.x = o1.x * alpha + beta * v1.x; o1.y = o1.y * alpha + beta * v1.y;
        o1.z = o1.z * alpha + beta * v1.z; o1.w = o1.w * alpha + beta * v1.w;
        m_i = m_new;
    }

    const size_t slot = (static_cast<size_t>(t) * n_head + h) * SPLITS + s;
    if (lane == 0) {
        part_m[slot] = m_i;
        part_l[slot] = l_i;
    }
    auto * po4 = reinterpret_cast<float4 *>(part_o + slot * 256);
    po4[lane * 2 + 0] = o0;
    po4[lane * 2 + 1] = o1;
}

template <int SPLITS>
__global__ void gqa_attn_d256_reduce_kernel(
    const float * __restrict__ part_m,
    const float * __restrict__ part_l,
    const float * __restrict__ part_o,
    float *       __restrict__ out,
    int n_head) {

    const int h    = blockIdx.x;
    const int t    = blockIdx.y;
    const int lane = threadIdx.x;
    const size_t base = (static_cast<size_t>(t) * n_head + h) * SPLITS;

    float m_max = -FLT_MAX;
    for (int s = 0; s < SPLITS; ++s) {
        if (part_l[base + s] > 0.0f) m_max = fmaxf(m_max, part_m[base + s]);
    }
    float l_tot = 0.0f;
    float4 o0 = make_float4(0, 0, 0, 0);
    float4 o1 = make_float4(0, 0, 0, 0);
    for (int s = 0; s < SPLITS; ++s) {
        const float ls = part_l[base + s];
        if (ls <= 0.0f) continue;
        const float sc = expf(part_m[base + s] - m_max);
        l_tot += ls * sc;
        const auto * po4 = reinterpret_cast<const float4 *>(part_o + (base + s) * 256);
        float4 a = po4[lane * 2 + 0], b = po4[lane * 2 + 1];
        o0.x += a.x * sc; o0.y += a.y * sc; o0.z += a.z * sc; o0.w += a.w * sc;
        o1.x += b.x * sc; o1.y += b.y * sc; o1.z += b.z * sc; o1.w += b.w * sc;
    }
    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    auto * dst4 = reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * n_head + h) * 256);
    dst4[lane * 2 + 0] = make_float4(o0.x * inv_l, o0.y * inv_l, o0.z * inv_l, o0.w * inv_l);
    dst4[lane * 2 + 1] = make_float4(o1.x * inv_l, o1.y * inv_l, o1.z * inv_l, o1.w * inv_l);
}

// Tiled FlashAttention Prefill (head_dim=256, T >= 8): F16 Tensor Core (sm>=70) or F32 SGEMM (<sm_70)
template <typename TMat>
__device__ __forceinline__ TMat qwen_f32_to_mat(float v) {
    if constexpr (std::is_same_v<TMat, __half>) return __float2half_rn(v);
    else return v;
}

template <typename TMat>
__device__ __forceinline__ TMat qwen_f16_to_mat(__half h) {
    if constexpr (std::is_same_v<TMat, __half>) return h;
    else return __half2float(h);
}

template <typename TMat>
__global__ void gqa_d256_q_to_kvh_kernel(
    const float * __restrict__ q, TMat * __restrict__ qm,
    int T, int n_head, int n_kv, float scale) {
    const int gqa = n_head / n_kv;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = T * n_head * 256;
    if (idx >= tot) return;
    const int d   = idx & 255;
    const int h   = (idx >> 8) % n_head;
    const int t   = idx / (n_head * 256);
    const int kvh = h / gqa;
    const int gh  = h % gqa;
    qm[((static_cast<size_t>(kvh) * (T * gqa) + t * gqa + gh) * 256) + d] = qwen_f32_to_mat<TMat>(q[idx] * scale);
}

template <bool KV_Q8_0, typename TMat>
__global__ void gqa_d256_dequant_kv_tile_kernel(
    const void * __restrict__ k_cache, const void * __restrict__ v_cache,
    TMat * __restrict__ km, TMat * __restrict__ vm,
    int k0, int kc, int n_kv) {
    const int kvh = blockIdx.x;
    const int c   = blockIdx.y;
    if (c >= kc) return;
    const int gp = k0 + c;
    const size_t dst_base = (static_cast<size_t>(kvh) * kc + c) * 256;
    if constexpr (KV_Q8_0) {
        const int b    = threadIdx.y; // 0..7
        const int lane = threadIdx.x; // 0..31
        const size_t bidx = (static_cast<size_t>(gp) * n_kv + kvh) * 8 + b;
        const BlockQ8_0 & kb = static_cast<const BlockQ8_0 *>(k_cache)[bidx];
        const BlockQ8_0 & vb = static_cast<const BlockQ8_0 *>(v_cache)[bidx];
        __half kdh, vdh;
        *reinterpret_cast<uint16_t *>(&kdh) = __ldg(&kb.d);
        *reinterpret_cast<uint16_t *>(&vdh) = __ldg(&vb.d);
        km[dst_base + b * 32 + lane] = qwen_f32_to_mat<TMat>(__half2float(kdh) * static_cast<float>(__ldg(&kb.qs[lane])));
        vm[dst_base + b * 32 + lane] = qwen_f32_to_mat<TMat>(__half2float(vdh) * static_cast<float>(__ldg(&vb.qs[lane])));
    } else {
        const int d = threadIdx.x; // 0..255
        const size_t src = (static_cast<size_t>(gp) * n_kv + kvh) * 256 + d;
        km[dst_base + d] = qwen_f16_to_mat<TMat>(static_cast<const __half *>(k_cache)[src]);
        vm[dst_base + d] = qwen_f16_to_mat<TMat>(static_cast<const __half *>(v_cache)[src]);
    }
}

__global__ void gqa_d256_init_flash_kernel(float * m, float * l, int M) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) { m[i] = -FLT_MAX; l[i] = 0.0f; }
}

template <typename TMat>
__global__ void gqa_d256_flash_softmax_rescale_kernel(
    const float *   __restrict__ s_tile,
    TMat *          __restrict__ p_tile,
    const int32_t * __restrict__ pos,
    float *         __restrict__ m_buf,
    float *         __restrict__ l_buf,
    float *         __restrict__ o_buf,
    int T, int n_head, int n_kv, int k0, int kc) {

    const int gqa  = n_head / n_kv;
    const int r    = blockIdx.x;
    const int kvh  = r / (T * gqa);
    const int rem  = r - kvh * (T * gqa);
    const int t    = rem / gqa;
    const int maxp = pos[t];

    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float * s_row = s_tile + static_cast<size_t>(r) * kc;
    TMat *        p_row = p_tile + static_cast<size_t>(r) * kc;
    float *       o_row = o_buf  + static_cast<size_t>(r) * 256;

    if (k0 > maxp) {
        if (k0 == 0) o_row[tid] = 0.0f;
        for (int c = tid; c < kc; c += blockDim.x) p_row[c] = qwen_f32_to_mat<TMat>(0.0f);
        return;
    }

    const int valid_c = (k0 + kc - 1 <= maxp) ? kc : (maxp - k0 + 1);
    float local_m = -FLT_MAX;
    for (int c = tid; c < valid_c; c += blockDim.x) local_m = fmaxf(local_m, s_row[c]);
    local_m = warp_max(local_m);
    __shared__ float s_red[8];
    if (lane == 0) s_red[warp] = local_m;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? s_red[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    const float m_tile = s_red[0];

    const float m_old = m_buf[r];
    const float m_new = fmaxf(m_old, m_tile);
    const float alpha = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);

    float local_l = 0.0f;
    for (int c = tid; c < kc; c += blockDim.x) {
        if (c < valid_c) {
            const float e = expf(s_row[c] - m_new);
            p_row[c] = qwen_f32_to_mat<TMat>(e);
            local_l += e;
        } else {
            p_row[c] = qwen_f32_to_mat<TMat>(0.0f);
        }
    }
    local_l = warp_sum(local_l);
    if (lane == 0) s_red[warp] = local_l;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? s_red[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) {
            m_buf[r] = m_new;
            l_buf[r] = l_buf[r] * alpha + v;
        }
    }

    if (k0 == 0) o_row[tid] = 0.0f;
    else if (alpha != 1.0f) o_row[tid] *= alpha;
}

__global__ void gqa_d256_flash_finalize_kernel(
    const float * __restrict__ o_buf, const float * __restrict__ l_buf,
    float * __restrict__ d_out, int T, int n_head, int n_kv) {
    const int gqa = n_head / n_kv;
    const int r   = blockIdx.x;
    const int d   = threadIdx.x; // 0..255
    const int kvh = r / (T * gqa);
    const int rem = r - kvh * (T * gqa);
    const int t   = rem / gqa;
    const int gh  = rem - t * gqa;
    const int h   = kvh * gqa + gh;
    const float inv_l = (l_buf[r] > 0.0f) ? (1.0f / l_buf[r]) : 0.0f;
    d_out[(static_cast<size_t>(t) * n_head + h) * 256 + d] = o_buf[static_cast<size_t>(r) * 256 + d] * inv_l;
}

// ============================================================================
// 4. Softmax MoE Router (supports up to 512 experts!) + Gated Shared Expert Add
// ============================================================================
__global__ void moe_route_softmax_kernel(
    const float * __restrict__ logits,
    int32_t *     __restrict__ top_ids,
    float *       __restrict__ top_w,
    int n_expert, int k, bool renorm, float scale) {

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float * row = logits + static_cast<size_t>(t) * n_expert;

    float v0 = (tid < n_expert)       ? row[tid]       : -FLT_MAX;
    float v1 = (tid + 256 < n_expert) ? row[tid + 256] : -FLT_MAX;
    float lm = fmaxf(v0, v1);
    lm = warp_max(lm);
    __shared__ float s_red[8];
    if (lane == 0) s_red[warp] = lm;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < 8) ? s_red[lane] : -FLT_MAX;
        m = warp_max(m);
        if (lane == 0) s_red[0] = m;
    }
    __syncthreads();
    const float row_max = s_red[0];

    float e0 = (tid < n_expert)       ? expf(v0 - row_max) : 0.0f;
    float e1 = (tid + 256 < n_expert) ? expf(v1 - row_max) : 0.0f;
    float ls = warp_sum(e0 + e1);
    if (lane == 0) s_red[warp] = ls;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < 8) ? s_red[lane] : 0.0f;
        s = warp_sum(s);
        if (lane == 0) s_red[0] = 1.0f / s;
    }
    __syncthreads();
    const float inv_s = s_red[0];
    float p0 = (tid < n_expert)       ? (e0 * inv_s) : -FLT_MAX;
    float p1 = (tid + 256 < n_expert) ? (e1 * inv_s) : -FLT_MAX;

    __shared__ float   s_wval[8];
    __shared__ int32_t s_widx[8];
    __shared__ int32_t s_chosen;
    __shared__ int32_t s_out_i[16];
    __shared__ float   s_out_w[16];
    float w_sum = 0.0f;

    for (int j = 0; j < k; ++j) {
        float   bv = p0;
        int32_t bi = tid;
        if (p1 > bv) { bv = p1; bi = tid + 256; }
        for (int off = 16; off > 0; off >>= 1) {
            float   ov = __shfl_down_sync(0xffffffffu, bv, off);
            int32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { s_wval[warp] = bv; s_widx[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            float   wv = (lane < 8) ? s_wval[lane] : -FLT_MAX;
            int32_t wi = (lane < 8) ? s_widx[lane] : 999999;
            for (int off = 4; off > 0; off >>= 1) {
                float   ov = __shfl_down_sync(0xffffffffu, wv, off);
                int32_t oi = __shfl_down_sync(0xffffffffu, wi, off);
                if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
            }
            if (lane == 0) {
                s_chosen   = wi;
                s_out_i[j] = wi;
                s_out_w[j] = wv;
                w_sum     += wv;
            }
        }
        __syncthreads();
        const int32_t picked = s_chosen;
        if (tid == picked)       p0 = -FLT_MAX;
        if (tid + 256 == picked) p1 = -FLT_MAX;
    }

    if (tid == 0) {
        const float mul = (renorm && w_sum > 0.0f) ? (scale / w_sum) : scale;
        int32_t * oi = top_ids + static_cast<size_t>(t) * k;
        float *   ow = top_w   + static_cast<size_t>(t) * k;
        for (int j = 0; j < k; ++j) {
            oi[j] = s_out_i[j];
            ow[j] = s_out_w[j] * mul;
        }
    }
}

__global__ void add_sigmoid_gated_vec_kernel(
    float *       __restrict__ dst,        // [T, H]
    const float * __restrict__ src,        // [T, H]
    const float * __restrict__ gate_logit, // [T]
    int H) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= H) return;
    const float g = 1.0f / (1.0f + expf(-gate_logit[t]));
    const size_t idx = static_cast<size_t>(t) * H + d;
    dst[idx] += g * src[idx];
}

// ============================================================================
// 5. Q8_0 SoA Routed-Only MoE (Decode + Grouped Prefill!) for Qwen3.6 (256 experts, ff=512)
// ============================================================================
__global__ void qwen_moe_gate_up_silu_decode_q8_0_kernel(
    const int4 *          __restrict__ g_q,
    const uint16_t *      __restrict__ g_s,
    const int4 *          __restrict__ u_q,
    const uint16_t *      __restrict__ u_s,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ top_ids,
    float *               __restrict__ act, // [T, k_used, ff]
    int K_blocks, int ff, int k_used) {

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
    const size_t row_off = (static_cast<size_t>(eid) * ff + row) * K_blocks;

    const int4 *     gq = g_q + row_off * 2;
    const uint16_t * gs = g_s + row_off;
    const int4 *     uq = u_q + row_off * 2;
    const uint16_t * us = u_s + row_off;
    const int blk_w = lane >> 1;
    const int half  = lane & 1;

    float ag = 0.0f, au = 0.0f;
    for (int b = blk_w; b < K_blocks; b += 16) {
        const int4  x4 = reinterpret_cast<const int4 *>(s_x[b].qs)[half];
        const float xd = s_x[b].d;
        {
            const int4 gw = __ldg(&gq[b * 2 + half]);
            int sg = dp4a_s32(gw.x, x4.x, dp4a_s32(gw.y, x4.y, dp4a_s32(gw.z, x4.z, dp4a_s32(gw.w, x4.w, 0))));
            __half dh; *reinterpret_cast<uint16_t *>(&dh) = __ldg(&gs[b]);
            ag += (__half2float(dh) * xd) * static_cast<float>(sg);
        }
        {
            const int4 uw = __ldg(&uq[b * 2 + half]);
            int su = dp4a_s32(uw.x, x4.x, dp4a_s32(uw.y, x4.y, dp4a_s32(uw.z, x4.z, dp4a_s32(uw.w, x4.w, 0))));
            __half dh; *reinterpret_cast<uint16_t *>(&dh) = __ldg(&us[b]);
            au += (__half2float(dh) * xd) * static_cast<float>(su);
        }
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) {
        act[(static_cast<size_t>(t) * k_used + slot) * ff + row] = (ag / (1.0f + expf(-ag))) * au;
    }
}

// Fused Q8_0 SoA Shared Expert Gate+Up+SiLU reusing already-quantized x_q8 in smem!
__global__ void qwen_shexp_gate_up_silu_decode_q8_0_kernel(
    const int4 *          __restrict__ g_q,
    const uint16_t *      __restrict__ g_s,
    const int4 *          __restrict__ u_q,
    const uint16_t *      __restrict__ u_s,
    const BlockQ8_1_Act * __restrict__ x_q8,
    float *               __restrict__ sh_act, // [T, ff_sh]
    int K_blocks, int ff_sh) {

    extern __shared__ BlockQ8_1_Act s_x[];
    const int t    = blockIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * src = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(t) * K_blocks);
    int4 *       dst = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) dst[i] = src[i];
    __syncthreads();

    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= ff_sh) return;
    const int lane  = threadIdx.x;
    const int blk_w = lane >> 1;
    const int half  = lane & 1;

    const size_t row_off = static_cast<size_t>(row) * K_blocks;
    const int4 *     gq = g_q + row_off * 2;
    const uint16_t * gs = g_s + row_off;
    const int4 *     uq = u_q + row_off * 2;
    const uint16_t * us = u_s + row_off;

    float ag = 0.0f, au = 0.0f;
    for (int b = blk_w; b < K_blocks; b += 16) {
        const int4  x4 = reinterpret_cast<const int4 *>(s_x[b].qs)[half];
        const float xd = s_x[b].d;
        {
            const int4 gw = __ldg(&gq[b * 2 + half]);
            int sg = dp4a_s32(gw.x, x4.x, dp4a_s32(gw.y, x4.y, dp4a_s32(gw.z, x4.z, dp4a_s32(gw.w, x4.w, 0))));
            __half dh; *reinterpret_cast<uint16_t *>(&dh) = __ldg(&gs[b]);
            ag += (__half2float(dh) * xd) * static_cast<float>(sg);
        }
        {
            const int4 uw = __ldg(&uq[b * 2 + half]);
            int su = dp4a_s32(uw.x, x4.x, dp4a_s32(uw.y, x4.y, dp4a_s32(uw.z, x4.z, dp4a_s32(uw.w, x4.w, 0))));
            __half dh; *reinterpret_cast<uint16_t *>(&dh) = __ldg(&us[b]);
            au += (__half2float(dh) * xd) * static_cast<float>(su);
        }
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) {
        sh_act[static_cast<size_t>(t) * ff_sh + row] = (ag / (1.0f + expf(-ag))) * au;
    }
}

// Single-barrier Fused Routed Down + Gated Shared Expert Down (Q8_0 SoA)!
// Loads all k_used routed act_q8 + sh_q8 into smem ONCE -> zero inner __syncthreads() & zero extra HBM pass!
__global__ void qwen_moe_down_decode_q8_0_kernel(
    const int4 *          __restrict__ dn_q,
    const uint16_t *      __restrict__ dn_s,
    const int4 *          __restrict__ sh_dn_q,    // optional Q8_0 SoA shared down
    const uint16_t *      __restrict__ sh_dn_s,
    const BlockQ8_1_Act * __restrict__ act_q8,     // [T, k_used, ff_blocks]
    const BlockQ8_1_Act * __restrict__ sh_q8,      // [T, ff_sh_blocks] (optional)
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    const float *         __restrict__ shexp_logit,// [T] (optional)
    float *               __restrict__ out,
    int ff_blocks, int ff_sh_blocks, int n_embd, int k_used) {

    extern __shared__ BlockQ8_1_Act s_all[];
    const int t    = blockIdx.y;
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const int lane = threadIdx.x;
    const int blk_w = lane >> 1;
    const int half  = lane & 1;

    const int routed_blks = k_used * ff_blocks;
    const int n_i4_r = (routed_blks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * src_r = reinterpret_cast<const int4 *>(act_q8 + static_cast<size_t>(t) * routed_blks);
    int4 *       dst_r = reinterpret_cast<int4 *>(s_all);
    for (int i = tid; i < n_i4_r; i += nthr) dst_r[i] = src_r[i];

    BlockQ8_1_Act * s_sh = s_all + routed_blks;
    if (sh_dn_q != nullptr && sh_q8 != nullptr) {
        const int n_i4_s = (ff_sh_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
        const int4 * src_s = reinterpret_cast<const int4 *>(sh_q8 + static_cast<size_t>(t) * ff_sh_blocks);
        int4 *       dst_s = reinterpret_cast<int4 *>(s_sh);
        for (int i = tid; i < n_i4_s; i += nthr) dst_s[i] = src_s[i];
    }
    __syncthreads();

    if (row >= n_embd) return;

    float total = 0.0f;
    for (int slot = 0; slot < k_used; ++slot) {
        const BlockQ8_1_Act * s_a = s_all + slot * ff_blocks;
        const int   eid = top_ids[t * k_used + slot];
        const float ew  = top_w  [t * k_used + slot];
        const size_t row_off = (static_cast<size_t>(eid) * n_embd + row) * ff_blocks;
        const int4 *     dq = dn_q + row_off * 2;
        const uint16_t * ds = dn_s + row_off;

        float acc = 0.0f;
        for (int b = blk_w; b < ff_blocks; b += 16) {
            const int4 w4 = __ldg(&dq[b * 2 + half]);
            const int4 a4 = reinterpret_cast<const int4 *>(s_a[b].qs)[half];
            int si = dp4a_s32(w4.x, a4.x, dp4a_s32(w4.y, a4.y, dp4a_s32(w4.z, a4.z, dp4a_s32(w4.w, a4.w, 0))));
            __half dh; *reinterpret_cast<uint16_t *>(&dh) = __ldg(&ds[b]);
            acc += (__half2float(dh) * s_a[b].d) * static_cast<float>(si);
        }
        total += ew * warp_sum(acc);
    }

    if (sh_dn_q != nullptr && sh_q8 != nullptr) {
        const size_t row_off = static_cast<size_t>(row) * ff_sh_blocks;
        const int4 *     sq = sh_dn_q + row_off * 2;
        const uint16_t * ss = sh_dn_s + row_off;
        float acc = 0.0f;
        for (int b = blk_w; b < ff_sh_blocks; b += 16) {
            const int4 w4 = __ldg(&sq[b * 2 + half]);
            const int4 a4 = reinterpret_cast<const int4 *>(s_sh[b].qs)[half];
            int si = dp4a_s32(w4.x, a4.x, dp4a_s32(w4.y, a4.y, dp4a_s32(w4.z, a4.z, dp4a_s32(w4.w, a4.w, 0))));
            __half dh; *reinterpret_cast<uint16_t *>(&dh) = __ldg(&ss[b]);
            acc += (__half2float(dh) * s_sh[b].d) * static_cast<float>(si);
        }
        const float sh_g = 1.0f / (1.0f + expf(-shexp_logit[t]));
        total += sh_g * warp_sum(acc);
    }

    if (lane == 0) out[static_cast<size_t>(t) * n_embd + row] = total;
}

__global__ void qwen_group_moe_tokens_kernel(
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
__global__ void qwen_moe_gate_up_grouped_q8_0_kernel(
    const int4 *          __restrict__ g_q,
    const uint16_t *      __restrict__ g_s,
    const int4 *          __restrict__ u_q,
    const uint16_t *      __restrict__ u_s,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ exp_offsets,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ act,
    int K_blocks, int ff, int k_used) {

    const int eid   = blockIdx.y;
    const int start = exp_offsets[eid];
    const int end   = exp_offsets[eid + 1];
    if (start >= end) return;

    extern __shared__ BlockQ8_1_Act s_x[];
    __shared__ int32_t s_pairs[BT];
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int lane     = threadIdx.x;
    const int row_base = blockIdx.x * (blockDim.y * R_PER_WARP) + threadIdx.y;
    const int blk_w    = lane >> 1;
    const int half     = lane & 1;
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

        for (int rr = 0; rr < R_PER_WARP; ++rr) {
            const int row = row_base + rr * blockDim.y;
            if (row >= ff) break;
            const size_t row_off = (static_cast<size_t>(eid) * ff + row) * K_blocks;
            const int4 *     gq = g_q + row_off * 2;
            const uint16_t * gs = g_s + row_off;
            const int4 *     uq = u_q + row_off * 2;
            const uint16_t * us = u_s + row_off;

            float ag[BT], au[BT];
            #pragma unroll
            for (int j = 0; j < BT; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

            for (int b = blk_w; b < K_blocks; b += 16) {
                const int4 gw = __ldg(&gq[b * 2 + half]);
                const int4 uw = __ldg(&uq[b * 2 + half]);
                __half gdh, udh;
                *reinterpret_cast<uint16_t *>(&gdh) = __ldg(&gs[b]);
                *reinterpret_cast<uint16_t *>(&udh) = __ldg(&us[b]);
                const float gd = __half2float(gdh), ud = __half2float(udh);

                #pragma unroll
                for (int j = 0; j < BT; ++j) {
                    if (j < n_tile) {
                        const BlockQ8_1_Act & xb = s_x[j * K_blocks + b];
                        const int4 x4 = reinterpret_cast<const int4 *>(xb.qs)[half];
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
                        act[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * ff + row] = (g / (1.0f + expf(-g))) * u;
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <int BT, int R_PER_WARP>
__global__ void qwen_moe_down_grouped_q8_0_kernel(
    const int4 *          __restrict__ dn_q,
    const uint16_t *      __restrict__ dn_s,
    const BlockQ8_1_Act * __restrict__ act_q8,
    const float *         __restrict__ top_w,
    const int32_t *       __restrict__ exp_offsets,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots,
    int ff_blocks, int n_embd, int k_used) {

    const int eid   = blockIdx.y;
    const int start = exp_offsets[eid];
    const int end   = exp_offsets[eid + 1];
    if (start >= end) return;

    extern __shared__ BlockQ8_1_Act s_a[];
    __shared__ int32_t s_pairs[BT];
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int lane     = threadIdx.x;
    const int row_base = blockIdx.x * (blockDim.y * R_PER_WARP) + threadIdx.y;
    const int blk_w    = lane >> 1;
    const int half     = lane & 1;
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

            for (int b = blk_w; b < ff_blocks; b += 16) {
                const int4 w4 = __ldg(&dq[b * 2 + half]);
                __half dh; *reinterpret_cast<uint16_t *>(&dh) = __ldg(&ds[b]);
                const float wd = __half2float(dh);
                #pragma unroll
                for (int j = 0; j < BT; ++j) {
                    if (j < n_tile) {
                        const BlockQ8_1_Act & xb = s_a[j * ff_blocks + b];
                        const int4 a4 = reinterpret_cast<const int4 *>(xb.qs)[half];
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

__global__ void qwen_reduce_slots_kernel(
    const float * __restrict__ slots, float * __restrict__ out, int n_embd, int k_used) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= n_embd) return;
    const float * base = slots + static_cast<size_t>(t) * k_used * n_embd + d;
    float sum = 0.0f;
    for (int s = 0; s < k_used; ++s) sum += base[static_cast<size_t>(s) * n_embd];
    out[static_cast<size_t>(t) * n_embd + d] = sum;
}

// ============================================================================
// 5B. Q6_K SoA Gate+Up+SiLU MoE Kernels (Decode + Grouped Prefill) for Qwen3.8!
// ============================================================================
__device__ __forceinline__ float qwen_half_bits_to_f32(uint16_t b) {
    __half h;
    *reinterpret_cast<uint16_t *>(&h) = b;
    return __half2float(h);
}

__device__ __forceinline__ float qwen_dot_q6_k_soa_row_q8_1(
    const int4 *          __restrict__ W_ql,
    const int4 *          __restrict__ W_qh,
    const int8_t *        __restrict__ W_sc,
    const uint16_t *      __restrict__ W_d,
    const BlockQ8_1_Act * __restrict__ s_x,
    int nb, int lane) {

    const int blk_in_warp = lane >> 2;
    const int sub         = lane & 3;
    const int h           = sub >> 1;
    const int is          = sub & 1;

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
        const float dw  = qwen_half_bits_to_f32(__ldg(&W_d[bi]));

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

__global__ void qwen_moe_gate_up_silu_decode_q6_k_kernel(
    const int4 *          __restrict__ G_ql,
    const int4 *          __restrict__ G_qh,
    const int8_t *        __restrict__ G_sc,
    const uint16_t *      __restrict__ G_d,
    const int4 *          __restrict__ U_ql,
    const int4 *          __restrict__ U_qh,
    const int8_t *        __restrict__ U_sc,
    const uint16_t *      __restrict__ U_d,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ top_ids,
    float *               __restrict__ act,
    int K_blocks, int ff, int k_used) {

    const int t    = blockIdx.z;
    const int slot = blockIdx.y;
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    const int nb   = K_blocks >> 3;

    extern __shared__ BlockQ8_1_Act s_x[];
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * src = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(t) * K_blocks);
    int4 *       dst = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) dst[i] = src[i];
    __syncthreads();

    if (row >= ff) return;
    const int eid = top_ids[static_cast<size_t>(t) * k_used + slot];
    const size_t rblk = (static_cast<size_t>(eid) * ff + row) * nb;
    const float g = qwen_dot_q6_k_soa_row_q8_1(G_ql + rblk * 8, G_qh + rblk * 4, G_sc + rblk * 16, G_d + rblk, s_x, nb, threadIdx.x);
    const float u = qwen_dot_q6_k_soa_row_q8_1(U_ql + rblk * 8, U_qh + rblk * 4, U_sc + rblk * 16, U_d + rblk, s_x, nb, threadIdx.x);
    if (threadIdx.x == 0) {
        act[(static_cast<size_t>(t) * k_used + slot) * ff + row] = (g / (1.0f + expf(-g))) * u;
    }
}

template <int BT, int R_PER_WARP>
__global__ void qwen_moe_gate_up_grouped_q6_k_kernel(
    const int4 *          __restrict__ G_ql,
    const int4 *          __restrict__ G_qh,
    const int8_t *        __restrict__ G_sc,
    const uint16_t *      __restrict__ G_d,
    const int4 *          __restrict__ U_ql,
    const int4 *          __restrict__ U_qh,
    const int8_t *        __restrict__ U_sc,
    const uint16_t *      __restrict__ U_d,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ exp_offsets,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ act,
    int K_blocks, int ff, int k_used) {

    const int eid   = blockIdx.y;
    const int start = exp_offsets[eid];
    const int end   = exp_offsets[eid + 1];
    if (start >= end) return;

    extern __shared__ BlockQ8_1_Act s_x[];
    __shared__ int32_t s_pairs[BT];
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;
    const int lane     = threadIdx.x;
    const int nb       = K_blocks >> 3;
    const int row_base = blockIdx.x * (blockDim.y * R_PER_WARP) + threadIdx.y;
    const int blk_in_warp = lane >> 2;
    const int sub      = lane & 3;
    const int h        = sub >> 1;
    const int is       = sub & 1;
    const int n_i4_tok = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;

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
            const int row = row_base + rr * blockDim.y;
            if (row >= ff) break;
            const size_t rblk = (static_cast<size_t>(eid) * ff + row) * nb;

            float ag[BT], au[BT];
            #pragma unroll
            for (int j = 0; j < BT; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

            // Compute Gate then Up sequentially to halve live register pressure (zero spills on sm_70 & sm_86!)
            for (int bi = blk_in_warp; bi < nb; bi += 8) {
                const int4 g_qh  = __ldg(&G_qh[(rblk + bi) * 4 + sub]);
                const int4 g_ql0 = __ldg(&G_ql[(rblk + bi) * 8 + h * 4 + is]);
                const int4 g_ql1 = __ldg(&G_ql[(rblk + bi) * 8 + h * 4 + 2 + is]);
                const int8_t * gsc = G_sc + (rblk + bi) * 16 + h * 8 + is;
                const float gsc0 = static_cast<float>(__ldg(&gsc[0]));
                const float gsc2 = static_cast<float>(__ldg(&gsc[2]));
                const float gsc4 = static_cast<float>(__ldg(&gsc[4]));
                const float gsc6 = static_cast<float>(__ldg(&gsc[6]));
                const float gdw  = qwen_half_bits_to_f32(__ldg(&G_d[rblk + bi]));
                const int xb_rel = bi * 8 + h * 4;
                #pragma unroll
                for (int j = 0; j < BT; ++j) {
                    if (j < n_tile) {
                        const BlockQ8_1_Act * xj = s_x + j * K_blocks + xb_rel;
                        const int4 x1 = reinterpret_cast<const int4 *>(xj[0].qs)[is];
                        const int4 x2 = reinterpret_cast<const int4 *>(xj[1].qs)[is];
                        const int4 x3 = reinterpret_cast<const int4 *>(xj[2].qs)[is];
                        const int4 x4 = reinterpret_cast<const int4 *>(xj[3].qs)[is];
                        const int sx0 = is ? xj[0].s1 : xj[0].s0;
                        const int sx1 = is ? xj[1].s1 : xj[1].s0;
                        const int sx2 = is ? xj[2].s1 : xj[2].s0;
                        const int sx3 = is ? xj[3].s1 : xj[3].s0;
                        int gs1 = 0, gs2 = 0, gs3 = 0, gs4 = 0;
                        step4(g_ql0.x, g_ql1.x, g_qh.x, x1.x, x2.x, x3.x, x4.x, gs1, gs2, gs3, gs4);
                        step4(g_ql0.y, g_ql1.y, g_qh.y, x1.y, x2.y, x3.y, x4.y, gs1, gs2, gs3, gs4);
                        step4(g_ql0.z, g_ql1.z, g_qh.z, x1.z, x2.z, x3.z, x4.z, gs1, gs2, gs3, gs4);
                        step4(g_ql0.w, g_ql1.w, g_qh.w, x1.w, x2.w, x3.w, x4.w, gs1, gs2, gs3, gs4);
                        ag[j] += gdw * (
                            (gsc0 * xj[0].d) * static_cast<float>(gs1 - 32 * sx0) +
                            (gsc2 * xj[1].d) * static_cast<float>(gs2 - 32 * sx1) +
                            (gsc4 * xj[2].d) * static_cast<float>(gs3 - 32 * sx2) +
                            (gsc6 * xj[3].d) * static_cast<float>(gs4 - 32 * sx3));
                    }
                }
            }
            for (int bi = blk_in_warp; bi < nb; bi += 8) {
                const int4 u_qh  = __ldg(&U_qh[(rblk + bi) * 4 + sub]);
                const int4 u_ql0 = __ldg(&U_ql[(rblk + bi) * 8 + h * 4 + is]);
                const int4 u_ql1 = __ldg(&U_ql[(rblk + bi) * 8 + h * 4 + 2 + is]);
                const int8_t * usc = U_sc + (rblk + bi) * 16 + h * 8 + is;
                const float usc0 = static_cast<float>(__ldg(&usc[0]));
                const float usc2 = static_cast<float>(__ldg(&usc[2]));
                const float usc4 = static_cast<float>(__ldg(&usc[4]));
                const float usc6 = static_cast<float>(__ldg(&usc[6]));
                const float udw  = qwen_half_bits_to_f32(__ldg(&U_d[rblk + bi]));
                const int xb_rel = bi * 8 + h * 4;
                #pragma unroll
                for (int j = 0; j < BT; ++j) {
                    if (j < n_tile) {
                        const BlockQ8_1_Act * xj = s_x + j * K_blocks + xb_rel;
                        const int4 x1 = reinterpret_cast<const int4 *>(xj[0].qs)[is];
                        const int4 x2 = reinterpret_cast<const int4 *>(xj[1].qs)[is];
                        const int4 x3 = reinterpret_cast<const int4 *>(xj[2].qs)[is];
                        const int4 x4 = reinterpret_cast<const int4 *>(xj[3].qs)[is];
                        const int sx0 = is ? xj[0].s1 : xj[0].s0;
                        const int sx1 = is ? xj[1].s1 : xj[1].s0;
                        const int sx2 = is ? xj[2].s1 : xj[2].s0;
                        const int sx3 = is ? xj[3].s1 : xj[3].s0;
                        int us1 = 0, us2 = 0, us3 = 0, us4 = 0;
                        step4(u_ql0.x, u_ql1.x, u_qh.x, x1.x, x2.x, x3.x, x4.x, us1, us2, us3, us4);
                        step4(u_ql0.y, u_ql1.y, u_qh.y, x1.y, x2.y, x3.y, x4.y, us1, us2, us3, us4);
                        step4(u_ql0.z, u_ql1.z, u_qh.z, x1.z, x2.z, x3.z, x4.z, us1, us2, us3, us4);
                        step4(u_ql0.w, u_ql1.w, u_qh.w, x1.w, x2.w, x3.w, x4.w, us1, us2, us3, us4);
                        au[j] += udw * (
                            (usc0 * xj[0].d) * static_cast<float>(us1 - 32 * sx0) +
                            (usc2 * xj[1].d) * static_cast<float>(us2 - 32 * sx1) +
                            (usc4 * xj[2].d) * static_cast<float>(us3 - 32 * sx2) +
                            (usc6 * xj[3].d) * static_cast<float>(us4 - 32 * sx3));
                    }
                }
            }
            #pragma unroll
            for (int j = 0; j < BT; ++j) {
                if (j < n_tile) {
                    const float g = warp_sum(ag[j]);
                    const float u = warp_sum(au[j]);
                    if (lane == 0) {
                        const int pair = s_pairs[j];
                        act[(static_cast<size_t>(pair >> 4) * k_used + (pair & 15)) * ff + row] = (g / (1.0f + expf(-g))) * u;
                    }
                }
            }
        }
        __syncthreads();
    }
}

} // namespace

void launch_qwen_gdn_gate_and_beta(
    float * d_alpha, const float * d_dt_bias, const float * d_ssm_a,
    float * d_beta, int T, int n_v_heads, cudaStream_t stream) {
    qwen_gdn_gate_beta_kernel<<<dim3((n_v_heads + 63) / 64, T), 64, 0, stream>>>(
        d_alpha, d_dt_bias, d_ssm_a, d_beta, n_v_heads);
}

void launch_qwen_gdn_packed_d128(
    float * d_qkv_conv, const float * d_g, const float * d_beta,
    float * d_state, float * d_out,
    int T, int n_k_heads, int n_v_heads, float eps, float scale,
    cudaStream_t stream, float * d_state_snap) {

    const int row_stride = (2 * n_k_heads + n_v_heads) * 128;
    constexpr int wpb_n = 4;
    qwen_qkv_l2_norm_d128_kernel<<<dim3((2 * n_k_heads + wpb_n - 1) / wpb_n, T), dim3(32, wpb_n), 0, stream>>>(
        d_qkv_conv, n_k_heads, row_stride, eps);

    constexpr int wpb_g = 4;
    qwen_gdn_packed_d128_kernel<<<dim3(n_v_heads, 128 / wpb_g), dim3(32, wpb_g), 0, stream>>>(
        d_qkv_conv, d_g, d_beta, d_state, d_out, d_state_snap,
        T, n_k_heads, n_v_heads, row_stride, scale);
}

void launch_rms_norm_silu_gate_d128(
    float * d_o, const float * d_norm_w, const float * d_gate,
    int T, int n_v_heads, float eps, cudaStream_t stream) {
    constexpr int wpb = 4;
    rms_norm_silu_gate_d128_kernel<<<dim3((n_v_heads + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
        d_o, d_norm_w, d_gate, n_v_heads, eps);
}

void launch_qwen_qgate_split_norm_rope_d256(
    const float * d_q_full, const float * d_q_norm_w, const int32_t * d_pos,
    float * d_q_out, float * d_gate_out,
    int T, int n_head, int rope_dim, float rms_eps, float rope_freq_base, cudaStream_t stream) {
    constexpr int wpb = 4;
    qwen_qgate_split_norm_rope_d256_kernel<<<dim3((n_head + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
        d_q_full, d_q_norm_w, d_pos, d_q_out, d_gate_out, n_head, rope_dim, rms_eps, rope_freq_base);
}

void launch_qwen_k_norm_rope_d256(
    float * d_k, const float * d_k_norm_w, const int32_t * d_pos,
    int T, int n_kv, int rope_dim, float rms_eps, float rope_freq_base, cudaStream_t stream) {
    constexpr int wpb = 2;
    qwen_k_norm_rope_d256_kernel<<<dim3((n_kv + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
        d_k, d_k_norm_w, d_pos, n_kv, rope_dim, rms_eps, rope_freq_base);
}

void launch_gqa_kv_write_d256(
    const float * d_k, const float * d_v,
    void * d_k_cache, void * d_v_cache, bool kv_is_q8_0,
    const int32_t * d_pos, int T, int n_kv, cudaStream_t stream) {
    if (kv_is_q8_0) {
        const int total_b = n_kv * 8;
        constexpr int wpb = 8;
        gqa_kv_write_q8_0_d256_kernel<<<dim3((total_b + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
            d_k, d_v, static_cast<BlockQ8_0 *>(d_k_cache), static_cast<BlockQ8_0 *>(d_v_cache), d_pos, T, n_kv);
    } else {
        const int kv_dim = n_kv * 256;
        gqa_kv_write_f16_d256_kernel<<<dim3((kv_dim + 255) / 256, T), 256, 0, stream>>>(
            d_k, d_v, static_cast<__half *>(d_k_cache), static_cast<__half *>(d_v_cache), d_pos, T, kv_dim);
    }
}

void launch_gqa_attn_d256(
    const float * d_q, const void * d_k_cache, const void * d_v_cache, bool kv_is_q8_0,
    const int32_t * d_pos, float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int n_head, int n_kv, float scale,
    cublasHandle_t cublas, int kv_len_max, cudaStream_t stream) {

    // Fast Tiled FlashAttention for prefill (T >= 8): FP16 Tensor Core (sm>=70) or FP32 SGEMM (<sm_70)!
    if (T >= 8 && cublas != nullptr && kv_len_max > 0) {
        const bool use_tc = device_has_tensor_cores(-1);
        const int gqa = n_head / n_kv;
        const int M_per_kv = T * gqa;
        const int M_tot = n_kv * M_per_kv;
        auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };

        if (use_tc) {
            constexpr int K_TILE = 512;
            const size_t q16_b = align256(static_cast<size_t>(M_tot) * 256 * sizeof(__half));
            const size_t k16_b = align256(static_cast<size_t>(n_kv) * K_TILE * 256 * sizeof(__half));
            const size_t v16_b = k16_b;
            const size_t s32_b = align256(static_cast<size_t>(M_tot) * K_TILE * sizeof(float));
            const size_t p16_b = align256(static_cast<size_t>(M_tot) * K_TILE * sizeof(__half));
            const size_t o32_b = align256(static_cast<size_t>(M_tot) * 256 * sizeof(float));
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

                const int tot_q = T * n_head * 256;
                gqa_d256_q_to_kvh_kernel<__half><<<(tot_q + 255) / 256, 256, 0, stream>>>(d_q, d_q16, T, n_head, n_kv, scale);
                gqa_d256_init_flash_kernel<<<(M_tot + 255) / 256, 256, 0, stream>>>(d_m, d_l, M_tot);

                const float one = 1.0f, zero = 0.0f;
                cublasSetStream(cublas, stream);

                for (int k0 = 0; k0 < kv_len_max; k0 += K_TILE) {
                    const int kc = (k0 + K_TILE < kv_len_max) ? K_TILE : (kv_len_max - k0);
                    if (kv_is_q8_0) {
                        gqa_d256_dequant_kv_tile_kernel<true, __half><<<dim3(n_kv, kc), dim3(32, 8), 0, stream>>>(
                            d_k_cache, d_v_cache, d_k16, d_v16, k0, kc, n_kv);
                    } else {
                        gqa_d256_dequant_kv_tile_kernel<false, __half><<<dim3(n_kv, kc), 256, 0, stream>>>(
                            d_k_cache, d_v_cache, d_k16, d_v16, k0, kc, n_kv);
                    }
                    cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        kc, M_per_kv, 256,
                        &one,
                        d_k16, CUDA_R_16F, 256, static_cast<long long>(kc) * 256,
                        d_q16, CUDA_R_16F, 256, static_cast<long long>(M_per_kv) * 256,
                        &zero,
                        d_s32, CUDA_R_32F, kc,  static_cast<long long>(M_per_kv) * kc,
                        n_kv, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                    gqa_d256_flash_softmax_rescale_kernel<__half><<<M_tot, 256, 0, stream>>>(
                        d_s32, d_p16, d_pos, d_m, d_l, d_ob, T, n_head, n_kv, k0, kc);

                    cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        256, M_per_kv, kc,
                        &one,
                        d_v16, CUDA_R_16F, 256, static_cast<long long>(kc) * 256,
                        d_p16, CUDA_R_16F, kc,  static_cast<long long>(M_per_kv) * kc,
                        &one,
                        d_ob,  CUDA_R_32F, 256, static_cast<long long>(M_per_kv) * 256,
                        n_kv, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                }
                gqa_d256_flash_finalize_kernel<<<M_tot, 256, 0, stream>>>(d_ob, d_l, d_out, T, n_head, n_kv);
                return;
            }
        } else {
            constexpr int K_TILE = 256;
            const size_t q32_b = align256(static_cast<size_t>(M_tot) * 256 * sizeof(float));
            const size_t k32_b = align256(static_cast<size_t>(n_kv) * K_TILE * 256 * sizeof(float));
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

                const int tot_q = T * n_head * 256;
                gqa_d256_q_to_kvh_kernel<float><<<(tot_q + 255) / 256, 256, 0, stream>>>(d_q, d_q32, T, n_head, n_kv, scale);
                gqa_d256_init_flash_kernel<<<(M_tot + 255) / 256, 256, 0, stream>>>(d_m, d_l, M_tot);

                const float one = 1.0f, zero = 0.0f;
                cublasSetStream(cublas, stream);

                for (int k0 = 0; k0 < kv_len_max; k0 += K_TILE) {
                    const int kc = (k0 + K_TILE < kv_len_max) ? K_TILE : (kv_len_max - k0);
                    if (kv_is_q8_0) {
                        gqa_d256_dequant_kv_tile_kernel<true, float><<<dim3(n_kv, kc), dim3(32, 8), 0, stream>>>(
                            d_k_cache, d_v_cache, d_k32, d_v32, k0, kc, n_kv);
                    } else {
                        gqa_d256_dequant_kv_tile_kernel<false, float><<<dim3(n_kv, kc), 256, 0, stream>>>(
                            d_k_cache, d_v_cache, d_k32, d_v32, k0, kc, n_kv);
                    }
                    cublasSgemmStridedBatched(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        kc, M_per_kv, 256,
                        &one,
                        d_k32, 256, static_cast<long long>(kc) * 256,
                        d_q32, 256, static_cast<long long>(M_per_kv) * 256,
                        &zero,
                        d_s32, kc,  static_cast<long long>(M_per_kv) * kc,
                        n_kv);

                    gqa_d256_flash_softmax_rescale_kernel<float><<<M_tot, 256, 0, stream>>>(
                        d_s32, d_p32, d_pos, d_m, d_l, d_ob, T, n_head, n_kv, k0, kc);

                    cublasSgemmStridedBatched(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        256, M_per_kv, kc,
                        &one,
                        d_v32, 256, static_cast<long long>(kc) * 256,
                        d_p32, kc,  static_cast<long long>(M_per_kv) * kc,
                        &one,
                        d_ob,  256, static_cast<long long>(M_per_kv) * 256,
                        n_kv);
                }
                gqa_d256_flash_finalize_kernel<<<M_tot, 256, 0, stream>>>(d_ob, d_l, d_out, T, n_head, n_kv);
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
        if (kv_is_q8_0) gqa_attn_d256_splitk_kernel<true,  S><<<grid, 32, 0, stream>>>(d_q, d_k_cache, d_v_cache, d_pos, pm, pl, po, n_head, n_kv, scale);
        else            gqa_attn_d256_splitk_kernel<false, S><<<grid, 32, 0, stream>>>(d_q, d_k_cache, d_v_cache, d_pos, pm, pl, po, n_head, n_kv, scale);
        gqa_attn_d256_reduce_kernel<S><<<dim3(n_head, T), 32, 0, stream>>>(pm, pl, po, d_out, n_head);
    } else {
        constexpr int S = 1;
        const size_t n_slots = static_cast<size_t>(T) * n_head;
        float * pm = static_cast<float *>(d_scratch);
        float * pl = pm + n_slots;
        float * po = pl + n_slots;
        dim3 grid(n_head, 1, T);
        if (kv_is_q8_0) gqa_attn_d256_splitk_kernel<true,  S><<<grid, 32, 0, stream>>>(d_q, d_k_cache, d_v_cache, d_pos, pm, pl, po, n_head, n_kv, scale);
        else            gqa_attn_d256_splitk_kernel<false, S><<<grid, 32, 0, stream>>>(d_q, d_k_cache, d_v_cache, d_pos, pm, pl, po, n_head, n_kv, scale);
        gqa_attn_d256_reduce_kernel<S><<<dim3(n_head, T), 32, 0, stream>>>(pm, pl, po, d_out, n_head);
    }
}

void launch_moe_route_softmax(
    const float * d_logits, int32_t * d_top_ids, float * d_top_w,
    int n_tokens, int n_expert, int k, bool renorm, float scale, cudaStream_t stream) {
    moe_route_softmax_kernel<<<n_tokens, 256, 0, stream>>>(d_logits, d_top_ids, d_top_w, n_expert, k, renorm, scale);
}

void launch_qwen_moe_ffn_gated_shared(
    cublasHandle_t cublas,
    const DeviceTensor & gate_exps, const DeviceTensor & up_exps, const DeviceTensor & down_exps,
    const DeviceTensor & gate_shexp, const DeviceTensor & up_shexp, const DeviceTensor & down_shexp,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    const float * d_shexp_gate_logit,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, cudaStream_t stream) {

    const int K        = static_cast<int>(gate_exps.ne[0]);
    const int ff       = static_cast<int>(gate_exps.ne[1]);
    const int n_expert = static_cast<int>(gate_exps.ne[2]);
    const int ff_sh    = static_cast<int>(gate_shexp.ne[1]);

    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    auto * base = static_cast<uint8_t *>(d_scratch);

    // 1. Routed experts -> d_out
    const bool gu_is_q8 = (gate_exps.dtype == DType::Q8_0 && gate_exps.is_soa &&
                           up_exps.dtype   == DType::Q8_0 && up_exps.is_soa);
    const bool gu_is_q6 = (gate_exps.dtype == DType::Q6_K && gate_exps.is_soa &&
                           up_exps.dtype   == DType::Q6_K && up_exps.is_soa);
    const bool dn_is_q8 = (down_exps.dtype == DType::Q8_0 && down_exps.is_soa);

    if ((gu_is_q8 || gu_is_q6) && dn_is_q8) {
        const int K_blocks = K >> 5;
        const int ff_blks  = ff >> 5;
        const auto * dq = static_cast<const int4 *>(down_exps.ptr);
        const auto * ds = static_cast<const uint16_t *>(down_exps.scales_ptr);

        // Fast Grouped MoE Prefill (sm_70 + sm_86 native hardware __dp4a, exact 0.000000 checkpoint parity!)
        const size_t xq8_b = align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
        const size_t act_b = align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
        const size_t aq8_b = align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BlockQ8_1_Act));
        const size_t off_b = align256(static_cast<size_t>(n_expert + 1) * sizeof(int32_t));
        const size_t sp_b  = align256(static_cast<size_t>(T) * k_used * sizeof(int32_t));
        const size_t sl_b  = align256(static_cast<size_t>(T) * k_used * K * sizeof(float));
        constexpr int BT = 8;
        const size_t gu_smem = static_cast<size_t>(BT) * K_blocks * sizeof(BlockQ8_1_Act);
        const size_t dn_smem = static_cast<size_t>(BT) * ff_blks  * sizeof(BlockQ8_1_Act);

        if (T >= 8 && n_expert <= 512 && k_used <= 16 && gu_smem <= 48000 && dn_smem <= 48000 &&
            (xq8_b + act_b + aq8_b + off_b + sp_b + sl_b <= scratch_bytes)) {
            size_t off = 0;
            auto * xq8          = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += xq8_b;
            auto * act          = reinterpret_cast<float *>        (base + off); off += act_b;
            auto * aq8          = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += aq8_b;
            auto * exp_offsets  = reinterpret_cast<int32_t *>      (base + off); off += off_b;
            auto * sorted_pairs = reinterpret_cast<int32_t *>      (base + off); off += sp_b;
            auto * down_slots   = reinterpret_cast<float *>        (base + off);

            launch_quantize_q8_1(d_x, xq8, K, T, stream);
            qwen_group_moe_tokens_kernel<<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
                d_top_ids, exp_offsets, sorted_pairs, T * k_used, n_expert, k_used);

            if (gu_is_q8) {
                constexpr int wpb = 8, rpw = 2;
                qwen_moe_gate_up_grouped_q8_0_kernel<BT, rpw><<<dim3((ff + wpb * rpw - 1) / (wpb * rpw), n_expert), dim3(32, wpb), gu_smem, stream>>>(
                    static_cast<const int4 *>(gate_exps.ptr), static_cast<const uint16_t *>(gate_exps.scales_ptr),
                    static_cast<const int4 *>(up_exps.ptr),   static_cast<const uint16_t *>(up_exps.scales_ptr),
                    xq8, exp_offsets, sorted_pairs, act, K_blocks, ff, k_used);
            } else {
                constexpr int wpb = 4, rpw = 2;
                qwen_moe_gate_up_grouped_q6_k_kernel<BT, rpw><<<dim3((ff + wpb * rpw - 1) / (wpb * rpw), n_expert), dim3(32, wpb), gu_smem, stream>>>(
                    static_cast<const int4 *>(gate_exps.ptr), static_cast<const int4 *>(gate_exps.aux_ptr),
                    static_cast<const int8_t *>(gate_exps.scales_ptr), static_cast<const uint16_t *>(gate_exps.d_ptr),
                    static_cast<const int4 *>(up_exps.ptr),   static_cast<const int4 *>(up_exps.aux_ptr),
                    static_cast<const int8_t *>(up_exps.scales_ptr), static_cast<const uint16_t *>(up_exps.d_ptr),
                    xq8, exp_offsets, sorted_pairs, act, K_blocks, ff, k_used);
            }

            launch_quantize_q8_1(act, aq8, k_used * ff, T, stream);
            {
                constexpr int wpb = 8, rpw = 2;
                qwen_moe_down_grouped_q8_0_kernel<BT, rpw><<<dim3((K + wpb * rpw - 1) / (wpb * rpw), n_expert), dim3(32, wpb), dn_smem, stream>>>(
                    dq, ds, aq8, d_top_w, exp_offsets, sorted_pairs, down_slots, ff_blks, K, k_used);
            }
            qwen_reduce_slots_kernel<<<dim3((K + 255) / 256, T), 256, 0, stream>>>(down_slots, d_out, K, k_used);
        } else {
            size_t off = 0;
            auto * xq8 = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
            auto * act = reinterpret_cast<float *>        (base + off); off += align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
            auto * aq8 = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BlockQ8_1_Act));
            void * rest = base + off;
            const size_t rem_b = (off < scratch_bytes) ? (scratch_bytes - off) : 0;

            launch_quantize_q8_1(d_x, xq8, K, T, stream);

            constexpr int wpb = 8;
            const bool sh_is_q8 = (gate_shexp.dtype == DType::Q8_0 && gate_shexp.is_soa &&
                                   up_shexp.dtype   == DType::Q8_0 && up_shexp.is_soa &&
                                   down_shexp.dtype == DType::Q8_0 && down_shexp.is_soa);
            const int ff_sh_blks = ff_sh >> 5;
            const size_t dec_gu_smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
            if (gu_is_q8) {
                qwen_moe_gate_up_silu_decode_q8_0_kernel<<<dim3((ff + wpb - 1) / wpb, k_used, T), dim3(32, wpb), dec_gu_smem, stream>>>(
                    static_cast<const int4 *>(gate_exps.ptr), static_cast<const uint16_t *>(gate_exps.scales_ptr),
                    static_cast<const int4 *>(up_exps.ptr),   static_cast<const uint16_t *>(up_exps.scales_ptr),
                    xq8, d_top_ids, act, K_blocks, ff, k_used);
            } else {
                qwen_moe_gate_up_silu_decode_q6_k_kernel<<<dim3((ff + wpb - 1) / wpb, k_used, T), dim3(32, wpb), dec_gu_smem, stream>>>(
                    static_cast<const int4 *>(gate_exps.ptr), static_cast<const int4 *>(gate_exps.aux_ptr),
                    static_cast<const int8_t *>(gate_exps.scales_ptr), static_cast<const uint16_t *>(gate_exps.d_ptr),
                    static_cast<const int4 *>(up_exps.ptr),   static_cast<const int4 *>(up_exps.aux_ptr),
                    static_cast<const int8_t *>(up_exps.scales_ptr), static_cast<const uint16_t *>(up_exps.d_ptr),
                    xq8, d_top_ids, act, K_blocks, ff, k_used);
            }

            launch_quantize_q8_1(act, aq8, k_used * ff, T, stream);

            const size_t sh_act_b = align256(static_cast<size_t>(T) * ff_sh * sizeof(float));
            const size_t sh_q8_b  = align256(static_cast<size_t>(T) * ff_sh_blks * sizeof(BlockQ8_1_Act));
            const size_t dec_dn_smem = static_cast<size_t>(k_used * ff_blks + (sh_is_q8 ? ff_sh_blks : 0)) * sizeof(BlockQ8_1_Act);
            if (sh_is_q8 && dec_dn_smem <= 48000 && sh_act_b + sh_q8_b <= rem_b) {
                auto * sh_act = reinterpret_cast<float *>(rest);
                auto * sh_q8  = reinterpret_cast<BlockQ8_1_Act *>(static_cast<uint8_t *>(rest) + sh_act_b);
                qwen_shexp_gate_up_silu_decode_q8_0_kernel<<<dim3((ff_sh + wpb - 1) / wpb, T), dim3(32, wpb), dec_gu_smem, stream>>>(
                    static_cast<const int4 *>(gate_shexp.ptr), static_cast<const uint16_t *>(gate_shexp.scales_ptr),
                    static_cast<const int4 *>(up_shexp.ptr),   static_cast<const uint16_t *>(up_shexp.scales_ptr),
                    xq8, sh_act, K_blocks, ff_sh);
                launch_quantize_q8_1(sh_act, sh_q8, ff_sh, T, stream);
                qwen_moe_down_decode_q8_0_kernel<<<dim3((K + wpb - 1) / wpb, T), dim3(32, wpb), dec_dn_smem, stream>>>(
                    dq, ds,
                    static_cast<const int4 *>(down_shexp.ptr), static_cast<const uint16_t *>(down_shexp.scales_ptr),
                    aq8, sh_q8, d_top_ids, d_top_w, d_shexp_gate_logit, d_out, ff_blks, ff_sh_blks, K, k_used);
                return;
            }

            qwen_moe_down_decode_q8_0_kernel<<<dim3((K + wpb - 1) / wpb, T), dim3(32, wpb), static_cast<size_t>(k_used * ff_blks) * sizeof(BlockQ8_1_Act), stream>>>(
                dq, ds, nullptr, nullptr, aq8, nullptr, d_top_ids, d_top_w, nullptr, d_out, ff_blks, 0, K, k_used);
        }
    } else {
        DeviceTensor empty_sh{};
        launch_moe_ffn_kquant_with_shared(
            cublas, gate_exps, up_exps, down_exps, empty_sh, empty_sh, empty_sh,
            d_x, d_top_ids, d_top_w, d_out, d_scratch, scratch_bytes, T, k_used, 0.0f, 0.0f, stream);
    }

    // 2. Gated Shared Expert: d_out += sigmoid(d_shexp_gate_logit[t]) * down_shexp(SiLU(gate_shexp(x)) * up_shexp(x))
    if (gate_shexp.valid() && up_shexp.valid() && down_shexp.valid()) {
        size_t off = 0;
        auto * sh_g   = reinterpret_cast<float *>(base + off); off += align256(static_cast<size_t>(T) * ff_sh * sizeof(float));
        auto * sh_u   = reinterpret_cast<float *>(base + off); off += align256(static_cast<size_t>(T) * ff_sh * sizeof(float));
        auto * sh_out = reinterpret_cast<float *>(base + off); off += align256(static_cast<size_t>(T) * K     * sizeof(float));
        void * sh_scr = base + off;
        const size_t sh_rem = (off < scratch_bytes) ? (scratch_bytes - off) : 0;

        launch_linear(cublas, gate_shexp, d_x, sh_g, T, sh_scr, sh_rem, stream);
        launch_linear(cublas, up_shexp,   d_x, sh_u, T, sh_scr, sh_rem, stream);
        launch_silu_mul(sh_g, sh_u, sh_g, static_cast<int64_t>(T) * ff_sh, stream);
        launch_linear(cublas, down_shexp, sh_g, sh_out, T, sh_scr, sh_rem, stream);

        add_sigmoid_gated_vec_kernel<<<dim3((K + 255) / 256, T), 256, 0, stream>>>(
            d_out, sh_out, d_shexp_gate_logit, K);
    }
}


} // namespace rookery
