#include "kernels/kernels.h"

#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

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

__global__ void kv_write_f16_kernel(
    const float *   __restrict__ k,
    const float *   __restrict__ v,
    __half *        __restrict__ k_cache,
    __half *        __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    int T, int n_kv_heads, int head_dim) {

    const int t = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = n_kv_heads * head_dim;
    if (t >= T || idx >= stride) return;

    const int p = pos[t];
    const size_t src_off = static_cast<size_t>(t) * stride + idx;
    const size_t dst_off = static_cast<size_t>(p) * stride + idx;
    k_cache[dst_off] = __float2half_rn(k[src_off]);
    v_cache[dst_off] = __float2half_rn(v[src_off]);
}

// 1 warp per 32-element block in (T, n_kv_heads, head_dim/32)
__global__ void kv_write_q8_0_kernel(
    const float *   __restrict__ k,
    const float *   __restrict__ v,
    BlockQ8_0 *     __restrict__ k_cache,
    BlockQ8_0 *     __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    int T, int n_kv_heads, int head_dim) {

    const int blocks_per_tok = (n_kv_heads * head_dim) / 32;
    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    const int t = blockIdx.y;
    if (t >= T || b >= blocks_per_tok) return;

    const int lane = threadIdx.x;
    const int p    = pos[t];
    const size_t src_base = (static_cast<size_t>(t) * blocks_per_tok + b) * 32;
    const size_t dst_blk  = static_cast<size_t>(p) * blocks_per_tok + b;

    // Quantize K block
    {
        float kv = k[src_base + lane];
        float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(kv)), 0);
        float d = amax / 127.0f;
        float inv_d = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        if (lane == 0) {
            __half dh = __float2half_rn(d);
            k_cache[dst_blk].d = *reinterpret_cast<uint16_t *>(&dh);
        }
        k_cache[dst_blk].qs[lane] = static_cast<int8_t>(__float2int_rn(kv * inv_d));
    }
    // Quantize V block
    {
        float vv = v[src_base + lane];
        float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(vv)), 0);
        float d = amax / 127.0f;
        float inv_d = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        if (lane == 0) {
            __half dh = __float2half_rn(d);
            v_cache[dst_blk].d = *reinterpret_cast<uint16_t *>(&dh);
        }
        v_cache[dst_blk].qs[lane] = static_cast<int8_t>(__float2int_rn(vv * inv_d));
    }
}

// Zero-syncthreads 1-warp-per-head online-softmax FlashAttention for GQA (head_dim = 128).
// Each of the 32 lanes owns 4 contiguous elements of head_dim=128 (d = lane*4 + {0,1,2,3}) in registers!
// Eliminates ALL __syncthreads() per KV position -> ~5x faster than block-synced version!
template <bool KV_Q8_0>
__global__ void gqa_flash_attn_d128_warp_kernel(
    const float *   __restrict__ q,
    const void *    __restrict__ k_cache,
    const void *    __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    float *         __restrict__ out,
    int n_heads, int n_kv_heads, float scale) {

    const int h    = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    if (h >= n_heads) return;

    const int lane      = threadIdx.x; // 0..31 -> covers 4 dims: lane*4 .. lane*4+3
    const int gqa_ratio = n_heads / n_kv_heads;
    const int kvh       = h / gqa_ratio;
    const int kv_len    = pos[t] + 1;

    const float4 q4_raw = reinterpret_cast<const float4 *>(q + (static_cast<size_t>(t) * n_heads + h) * 128)[lane];
    const float q0 = q4_raw.x * scale;
    const float q1 = q4_raw.y * scale;
    const float q2 = q4_raw.z * scale;
    const float q3 = q4_raw.w * scale;

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const int blk_idx = lane >> 3;        // 0..3 (which of the 4 Q8_0 blocks in head_dim=128)
    const int blk_off = (lane & 7) << 2;  // 0,4,8,...,28

    for (int p = 0; p < kv_len; ++p) {
        float k0, k1, k2, k3;
        float v0, v1, v2, v3;

        if constexpr (KV_Q8_0) {
            const auto * kc = static_cast<const BlockQ8_0 *>(k_cache);
            const auto * vc = static_cast<const BlockQ8_0 *>(v_cache);
            const size_t blk = (static_cast<size_t>(p) * n_kv_heads + kvh) * 4 + blk_idx;
            const BlockQ8_0 & kb = kc[blk];
            const BlockQ8_0 & vb = vc[blk];

            __half kd_h, vd_h;
            *reinterpret_cast<uint16_t *>(&kd_h) = __ldg(&kb.d);
            *reinterpret_cast<uint16_t *>(&vd_h) = __ldg(&vb.d);
            const float kd = __half2float(kd_h);
            const float vd = __half2float(vd_h);

            k0 = kd * static_cast<float>(kb.qs[blk_off + 0]);
            k1 = kd * static_cast<float>(kb.qs[blk_off + 1]);
            k2 = kd * static_cast<float>(kb.qs[blk_off + 2]);
            k3 = kd * static_cast<float>(kb.qs[blk_off + 3]);

            v0 = vd * static_cast<float>(vb.qs[blk_off + 0]);
            v1 = vd * static_cast<float>(vb.qs[blk_off + 1]);
            v2 = vd * static_cast<float>(vb.qs[blk_off + 2]);
            v3 = vd * static_cast<float>(vb.qs[blk_off + 3]);
        } else {
            const auto * kc = static_cast<const uint2 *>(k_cache);
            const auto * vc = static_cast<const uint2 *>(v_cache);
            const size_t off4 = (static_cast<size_t>(p) * n_kv_heads + kvh) * 32 + lane;
            uint2 kp = __ldg(&kc[off4]);
            uint2 vp = __ldg(&vc[off4]);
            __half2 kh0 = *reinterpret_cast<const __half2 *>(&kp.x);
            __half2 kh1 = *reinterpret_cast<const __half2 *>(&kp.y);
            __half2 vh0 = *reinterpret_cast<const __half2 *>(&vp.x);
            __half2 vh1 = *reinterpret_cast<const __half2 *>(&vp.y);
            float2 kf0 = __half22float2(kh0), kf1 = __half22float2(kh1);
            float2 vf0 = __half22float2(vh0), vf1 = __half22float2(vh1);
            k0 = kf0.x; k1 = kf0.y; k2 = kf1.x; k3 = kf1.y;
            v0 = vf0.x; v1 = vf0.y; v2 = vf1.x; v3 = vf1.y;
        }

        float dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffffu, dot, 0);

        const float m_new = fmaxf(m_i, score);
        const float alpha = expf(m_i - m_new);
        const float beta  = expf(score - m_new);

        l_i = l_i * alpha + beta;
        o0  = o0  * alpha + beta * v0;
        o1  = o1  * alpha + beta * v1;
        o2  = o2  * alpha + beta * v2;
        o3  = o3  * alpha + beta * v3;
        m_i = m_new;
    }

    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * n_heads + h) * 128)[lane] =
        make_float4(o0 * inv_l, o1 * inv_l, o2 * inv_l, o3 * inv_l);
}

} // namespace

void launch_kv_write_f16(
    const float * d_k, const float * d_v,
    void * d_k_cache, void * d_v_cache,
    const int32_t * d_pos, int T, int n_kv_heads, int head_dim,
    cudaStream_t stream) {

    const int stride = n_kv_heads * head_dim;
    constexpr int threads = 256;
    dim3 grid((stride + threads - 1) / threads, T);
    kv_write_f16_kernel<<<grid, threads, 0, stream>>>(
        d_k, d_v,
        static_cast<__half *>(d_k_cache), static_cast<__half *>(d_v_cache),
        d_pos, T, n_kv_heads, head_dim);
}

void launch_kv_write_q8_0(
    const float * d_k, const float * d_v,
    void * d_k_cache, void * d_v_cache,
    const int32_t * d_pos, int T, int n_kv_heads, int head_dim,
    cudaStream_t stream) {

    const int blocks_per_tok = (n_kv_heads * head_dim) / 32;
    constexpr int wpb = 8;
    dim3 grid((blocks_per_tok + wpb - 1) / wpb, T);
    kv_write_q8_0_kernel<<<grid, dim3(32, wpb), 0, stream>>>(
        d_k, d_v,
        static_cast<BlockQ8_0 *>(d_k_cache), static_cast<BlockQ8_0 *>(d_v_cache),
        d_pos, T, n_kv_heads, head_dim);
}

void launch_gqa_attn(
    const float * d_q, const void * d_k_cache, const void * d_v_cache,
    bool kv_is_q8_0, const int32_t * d_pos, float * d_out, void * d_scratch,
    int T, int n_heads, int n_kv_heads, int head_dim, float scale,
    cudaStream_t stream) {

    // Reuse Split-K=16 zero-copy FlashAttention for decode/MTP (T <= 4) so dense attention
    // at long contexts (e.g. blk.45 MTP at 20K ctx) saturates all SMs!
    if (T <= 4 && d_scratch != nullptr) {
        launch_gqa_attn_msa(d_q, d_k_cache, d_v_cache, kv_is_q8_0,
                            d_pos, nullptr, nullptr, d_out, d_scratch,
                            T, n_heads, n_kv_heads, head_dim, 128, 0, scale, stream);
        return;
    }

    constexpr int wpb = 4; // 4 heads per block = 8 blocks for 32 heads
    dim3 grid((n_heads + wpb - 1) / wpb, T);
    dim3 block(32, wpb);
    if (kv_is_q8_0) {
        gqa_flash_attn_d128_warp_kernel<true><<<grid, block, 0, stream>>>(
            d_q, d_k_cache, d_v_cache, d_pos, d_out, n_heads, n_kv_heads, scale);
    } else {
        gqa_flash_attn_d128_warp_kernel<false><<<grid, block, 0, stream>>>(
            d_q, d_k_cache, d_v_cache, d_pos, d_out, n_heads, n_kv_heads, scale);
    }
}

} // namespace rookery
