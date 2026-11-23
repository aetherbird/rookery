#include "kernels/kernels.h"

#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

namespace rookery {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void idx_k_write_f16_kernel(
    const float *   __restrict__ ik,
    __half *        __restrict__ ik_cache,
    const int32_t * __restrict__ pos,
    int T, int idx_dim) {

    const int t = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || d >= idx_dim) return;
    const int p = pos[t];
    ik_cache[static_cast<size_t>(p) * idx_dim + d] = __float2half_rn(ik[static_cast<size_t>(t) * idx_dim + d]);
}

// ============================================================================
// Split-Block MSA Indexer: 1 block per (block_b, idx_head, token)
// Saturates ALL 80-82 SMs (e.g. 256 * 4 = 1024 blocks vs 4 blocks previously -> ~80x faster!)
// ============================================================================
__global__ void msa_indexer_score_blocks_d128_kernel(
    const float *   __restrict__ iq,         // [T, n_idx_heads, 128]
    const __half *  __restrict__ ik_cache,   // [max_ctx, 128] F16
    const int32_t * __restrict__ pos,        // [T]
    float *         __restrict__ bscores,    // [T, n_idx_heads, max_nblk]
    int max_nblk, int n_idx_heads, int blk_size, int prefix_blks, int local_blks, int dense_prefix_tokens) {

    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;

    const int p = pos[t];
    if (p < dense_prefix_tokens) return;

    const int kv_len = p + 1;
    const int nblk   = (kv_len + blk_size - 1) / blk_size;
    if (b >= nblk) return;

    const int cur_b = p / blk_size;
    const size_t out_idx = (static_cast<size_t>(t) * n_idx_heads + h) * max_nblk + b;

    if (b < prefix_blks || (b <= cur_b && b > cur_b - local_blks)) {
        if (threadIdx.x == 0) bscores[out_idx] = 1e30f;
        return;
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5; // 0..3 (4 warps per 128-token block)

    const float4 q4 = reinterpret_cast<const float4 *>(iq + (static_cast<size_t>(t) * n_idx_heads + h) * 128)[lane];

    const int p_start = b * blk_size;
    const int p_end   = (p_start + blk_size < kv_len) ? (p_start + blk_size) : kv_len;

    float wmax = -FLT_MAX;
    for (int kp = p_start + warp; kp < p_end; kp += 4) {
        const uint2 kpack = __ldg(reinterpret_cast<const uint2 *>(ik_cache + static_cast<size_t>(kp) * 128) + lane);
        const float2 kf0 = __half22float2(*reinterpret_cast<const __half2 *>(&kpack.x));
        const float2 kf1 = __half22float2(*reinterpret_cast<const __half2 *>(&kpack.y));
        float dot = warp_sum(q4.x * kf0.x + q4.y * kf0.y + q4.z * kf1.x + q4.w * kf1.y);
        if (lane == 0) wmax = fmaxf(wmax, dot);
    }

    __shared__ float s_wmax[4];
    if (lane == 0) s_wmax[warp] = wmax;
    __syncthreads();

    if (threadIdx.x == 0) {
        float m01 = fmaxf(s_wmax[0], s_wmax[1]);
        float m23 = fmaxf(s_wmax[2], s_wmax[3]);
        bscores[out_idx] = fmaxf(m01, m23);
    }
}

__global__ void msa_indexer_select_topk_kernel(
    const float *   __restrict__ bscores,     // [T, n_idx_heads, max_nblk]
    const int32_t * __restrict__ pos,         // [T]
    int32_t *       __restrict__ out_blocks,  // [T, n_idx_heads, k_max]
    int32_t *       __restrict__ out_k_count, // [T]
    int max_nblk, int n_idx_heads, int blk_size, int k_max, int dense_prefix_tokens) {

    const int h    = blockIdx.x;
    const int t    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int p = pos[t];
    if (p < dense_prefix_tokens) {
        if (h == 0 && tid == 0) out_k_count[t] = 0;
        return;
    }

    const int kv_len = p + 1;
    const int nblk   = (kv_len + blk_size - 1) / blk_size;
    const int K      = (k_max < nblk) ? k_max : nblk;
    if (h == 0 && tid == 0) out_k_count[t] = K;

    __shared__ float   s_bscore[4096];
    __shared__ float   s_wval[8];
    __shared__ int32_t s_widx[8];

    const float * src = bscores + (static_cast<size_t>(t) * n_idx_heads + h) * max_nblk;
    for (int b = tid; b < nblk; b += 256) {
        s_bscore[b] = src[b];
    }
    __syncthreads();

    int32_t * dst = out_blocks + (static_cast<size_t>(t) * n_idx_heads + h) * k_max;
    for (int ki = 0; ki < K; ++ki) {
        float   bv = -FLT_MAX;
        int32_t bi = 0;
        for (int b = tid; b < nblk; b += 256) {
            float v = s_bscore[b];
            if (v > bv) { bv = v; bi = b; }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            float   ov = __shfl_down_sync(0xffffffffu, bv, offset);
            int32_t oi = __shfl_down_sync(0xffffffffu, bi, offset);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { s_wval[warp] = bv; s_widx[warp] = bi; }
        __syncthreads();

        if (warp == 0) {
            float   wv = (lane < 8) ? s_wval[lane] : -FLT_MAX;
            int32_t wi = (lane < 8) ? s_widx[lane] : 999999;
            for (int offset = 4; offset > 0; offset >>= 1) {
                float   ov = __shfl_down_sync(0xffffffffu, wv, offset);
                int32_t oi = __shfl_down_sync(0xffffffffu, wi, offset);
                if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
            }
            if (lane == 0) {
                dst[ki]      = wi;
                s_bscore[wi] = -FLT_MAX;
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Split-K Zero-Copy Block-Sparse + Dense FlashAttention over Q8_0 / F16 KV Cache
// For decode (T <= 4): splits KV across S=16 blocks per head -> 32*16 = 512 blocks!
// Saturates ALL SMs on RTX 3090 (82 SMs) and V100 (80 SMs) with ZERO KV copies!
// ============================================================================
template <bool KV_Q8_0>
__device__ __forceinline__ void attend_one_pos(
    int p, int n_kv_heads, int kvh, int lane, int blk_idx, int blk_off,
    float q0, float q1, float q2, float q3,
    const void * k_cache, const void * v_cache,
    float & m_i, float & l_i, float & o0, float & o1, float & o2, float & o3) {

    float k0, k1, k2, k3, v0, v1, v2, v3;
    if constexpr (KV_Q8_0) {
        const auto * kc = static_cast<const BlockQ8_0 *>(k_cache);
        const auto * vc = static_cast<const BlockQ8_0 *>(v_cache);
        const size_t blk = (static_cast<size_t>(p) * n_kv_heads + kvh) * 4 + blk_idx;
        const BlockQ8_0 & kb = kc[blk];
        const BlockQ8_0 & vb = vc[blk];
        __half kd_h, vd_h;
        *reinterpret_cast<uint16_t *>(&kd_h) = __ldg(&kb.d);
        *reinterpret_cast<uint16_t *>(&vd_h) = __ldg(&vb.d);
        const float kd = __half2float(kd_h), vd = __half2float(vd_h);
        // Load 4 int8s with two 16-bit __ldg loads (2-byte aligned inside BlockQ8_0!)
        const uint16_t kq01 = __ldg(reinterpret_cast<const uint16_t *>(kb.qs + blk_off));
        const uint16_t kq23 = __ldg(reinterpret_cast<const uint16_t *>(kb.qs + blk_off + 2));
        const uint16_t vq01 = __ldg(reinterpret_cast<const uint16_t *>(vb.qs + blk_off));
        const uint16_t vq23 = __ldg(reinterpret_cast<const uint16_t *>(vb.qs + blk_off + 2));
        k0 = kd * static_cast<float>(static_cast<int8_t>(kq01 & 0xFF));
        k1 = kd * static_cast<float>(static_cast<int8_t>(kq01 >> 8));
        k2 = kd * static_cast<float>(static_cast<int8_t>(kq23 & 0xFF));
        k3 = kd * static_cast<float>(static_cast<int8_t>(kq23 >> 8));
        v0 = vd * static_cast<float>(static_cast<int8_t>(vq01 & 0xFF));
        v1 = vd * static_cast<float>(static_cast<int8_t>(vq01 >> 8));
        v2 = vd * static_cast<float>(static_cast<int8_t>(vq23 & 0xFF));
        v3 = vd * static_cast<float>(static_cast<int8_t>(vq23 >> 8));
    } else {
        const auto * kc = static_cast<const uint2 *>(k_cache);
        const auto * vc = static_cast<const uint2 *>(v_cache);
        const size_t off4 = (static_cast<size_t>(p) * n_kv_heads + kvh) * 32 + lane;
        uint2 kp = __ldg(&kc[off4]), vp = __ldg(&vc[off4]);
        float2 kf0 = __half22float2(*reinterpret_cast<const __half2 *>(&kp.x));
        float2 kf1 = __half22float2(*reinterpret_cast<const __half2 *>(&kp.y));
        float2 vf0 = __half22float2(*reinterpret_cast<const __half2 *>(&vp.x));
        float2 vf1 = __half22float2(*reinterpret_cast<const __half2 *>(&vp.y));
        k0 = kf0.x; k1 = kf0.y; k2 = kf1.x; k3 = kf1.y;
        v0 = vf0.x; v1 = vf0.y; v2 = vf1.x; v3 = vf1.y;
    }
    float dot = warp_sum(q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3);
    const float score = __shfl_sync(0xffffffffu, dot, 0);
    const float m_new = fmaxf(m_i, score);
    const float alpha = expf(m_i - m_new);
    const float beta  = expf(score - m_new);
    l_i = l_i * alpha + beta;
    o0 = o0 * alpha + beta * v0;
    o1 = o1 * alpha + beta * v1;
    o2 = o2 * alpha + beta * v2;
    o3 = o3 * alpha + beta * v3;
    m_i = m_new;
}

template <bool KV_Q8_0, int SPLITS>
__global__ void gqa_flash_attn_msa_splitk_d128_kernel(
    const float *   __restrict__ q,
    const void *    __restrict__ k_cache,
    const void *    __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    const int32_t * __restrict__ msa_blocks,
    const int32_t * __restrict__ msa_k_count,
    float *         __restrict__ part_m,   // [T, n_heads, SPLITS]
    float *         __restrict__ part_l,   // [T, n_heads, SPLITS]
    float *         __restrict__ part_o,   // [T, n_heads, SPLITS, 128]
    int n_heads, int n_kv_heads, int blk_size, int k_max, float scale) {

    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int t = blockIdx.z;
    const int lane = threadIdx.x;

    const int gqa_ratio = n_heads / n_kv_heads;
    const int kvh       = h / gqa_ratio;
    const int kv_len    = pos[t] + 1;
    const int k_sel     = msa_k_count ? msa_k_count[t] : 0;

    const float4 q4 = reinterpret_cast<const float4 *>(q + (static_cast<size_t>(t) * n_heads + h) * 128)[lane];
    const float q0 = q4.x * scale, q1 = q4.y * scale, q2 = q4.z * scale, q3 = q4.w * scale;

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const int blk_idx = lane >> 3;
    const int blk_off = (lane & 7) << 2;

    if (k_sel == 0) {
        const int chunk = (kv_len + SPLITS - 1) / SPLITS;
        const int p0    = s * chunk;
        const int p1    = (p0 + chunk < kv_len) ? (p0 + chunk) : kv_len;
        for (int p = p0; p < p1; ++p) {
            attend_one_pos<KV_Q8_0>(p, n_kv_heads, kvh, lane, blk_idx, blk_off,
                                    q0, q1, q2, q3, k_cache, v_cache, m_i, l_i, o0, o1, o2, o3);
        }
    } else {
        const int32_t * blks = msa_blocks + (static_cast<size_t>(t) * n_kv_heads + kvh) * k_max;
        const int bchunk = (k_sel + SPLITS - 1) / SPLITS;
        const int b0     = s * bchunk;
        const int b1     = (b0 + bchunk < k_sel) ? (b0 + bchunk) : k_sel;
        for (int bi = b0; bi < b1; ++bi) {
            const int base_p = blks[bi] * blk_size;
            const int end_p  = (base_p + blk_size < kv_len) ? (base_p + blk_size) : kv_len;
            for (int p = base_p; p < end_p; ++p) {
                attend_one_pos<KV_Q8_0>(p, n_kv_heads, kvh, lane, blk_idx, blk_off,
                                        q0, q1, q2, q3, k_cache, v_cache, m_i, l_i, o0, o1, o2, o3);
            }
        }
    }

    const size_t slot = (static_cast<size_t>(t) * n_heads + h) * SPLITS + s;
    if (lane == 0) {
        part_m[slot] = m_i;
        part_l[slot] = l_i;
    }
    reinterpret_cast<float4 *>(part_o + slot * 128)[lane] = make_float4(o0, o1, o2, o3);
}

template <int SPLITS>
__global__ void gqa_flash_attn_splitk_reduce_d128_kernel(
    const float * __restrict__ part_m,
    const float * __restrict__ part_l,
    const float * __restrict__ part_o,
    float *       __restrict__ out,
    int n_heads) {

    const int h    = blockIdx.x;
    const int t    = blockIdx.y;
    const int lane = threadIdx.x;

    const size_t base = (static_cast<size_t>(t) * n_heads + h) * SPLITS;

    float m_max = -FLT_MAX;
    for (int s = 0; s < SPLITS; ++s) {
        if (part_l[base + s] > 0.0f) {
            m_max = fmaxf(m_max, part_m[base + s]);
        }
    }

    float l_tot = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    for (int s = 0; s < SPLITS; ++s) {
        const float ls = part_l[base + s];
        if (ls <= 0.0f) continue;
        const float scale = expf(part_m[base + s] - m_max);
        l_tot += ls * scale;
        const float4 po = reinterpret_cast<const float4 *>(part_o + (base + s) * 128)[lane];
        o0 += po.x * scale;
        o1 += po.y * scale;
        o2 += po.z * scale;
        o3 += po.w * scale;
    }

    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * n_heads + h) * 128)[lane] =
        make_float4(o0 * inv_l, o1 * inv_l, o2 * inv_l, o3 * inv_l);
}

// Single-pass warp kernel for large prefill batches (T > 4), where T * n_heads already saturates all SMs.
template <bool KV_Q8_0>
__global__ void gqa_flash_attn_msa_d128_warp_kernel(
    const float *   __restrict__ q,
    const void *    __restrict__ k_cache,
    const void *    __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    const int32_t * __restrict__ msa_blocks,
    const int32_t * __restrict__ msa_k_count,
    float *         __restrict__ out,
    int n_heads, int n_kv_heads, int blk_size, int k_max, float scale) {

    const int h = blockIdx.x * blockDim.y + threadIdx.y;
    const int t = blockIdx.y;
    if (h >= n_heads) return;

    const int lane      = threadIdx.x;
    const int gqa_ratio = n_heads / n_kv_heads;
    const int kvh       = h / gqa_ratio;
    const int kv_len    = pos[t] + 1;
    const int k_sel     = msa_k_count ? msa_k_count[t] : 0;

    const float4 q4 = reinterpret_cast<const float4 *>(q + (static_cast<size_t>(t) * n_heads + h) * 128)[lane];
    const float q0 = q4.x * scale, q1 = q4.y * scale, q2 = q4.z * scale, q3 = q4.w * scale;

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const int blk_idx = lane >> 3;
    const int blk_off = (lane & 7) << 2;

    if (k_sel == 0) {
        for (int p = 0; p < kv_len; ++p) {
            attend_one_pos<KV_Q8_0>(p, n_kv_heads, kvh, lane, blk_idx, blk_off,
                                    q0, q1, q2, q3, k_cache, v_cache, m_i, l_i, o0, o1, o2, o3);
        }
    } else {
        const int32_t * blks = msa_blocks + (static_cast<size_t>(t) * n_kv_heads + kvh) * k_max;
        for (int bi = 0; bi < k_sel; ++bi) {
            const int base_p = blks[bi] * blk_size;
            const int end_p  = (base_p + blk_size < kv_len) ? (base_p + blk_size) : kv_len;
            for (int p = base_p; p < end_p; ++p) {
                attend_one_pos<KV_Q8_0>(p, n_kv_heads, kvh, lane, blk_idx, blk_off,
                                        q0, q1, q2, q3, k_cache, v_cache, m_i, l_i, o0, o1, o2, o3);
            }
        }
    }

    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * n_heads + h) * 128)[lane] =
        make_float4(o0 * inv_l, o1 * inv_l, o2 * inv_l, o3 * inv_l);
}

} // namespace

void launch_idx_k_write_f16(
    const float * d_ik, void * d_ik_cache, const int32_t * d_pos,
    int T, int idx_dim, cudaStream_t stream) {

    dim3 grid((idx_dim + 127) / 128, T);
    idx_k_write_f16_kernel<<<grid, 128, 0, stream>>>(
        d_ik, static_cast<__half *>(d_ik_cache), d_pos, T, idx_dim);
}

void launch_msa_indexer_topk(
    const float * d_iq, const void * d_ik_cache, const int32_t * d_pos,
    int32_t * d_out_blocks, int32_t * d_out_k_count, void * d_scratch,
    int T, int max_ctx, int n_idx_heads, int blk_size, int prefix_blks, int local_blks, int k_max, int dense_prefix_tokens,
    cudaStream_t stream) {

    const int max_nblk = (max_ctx + blk_size - 1) / blk_size;
    float * d_bscores = static_cast<float *>(d_scratch);

    dim3 score_grid(max_nblk, n_idx_heads, T);
    msa_indexer_score_blocks_d128_kernel<<<score_grid, 128, 0, stream>>>(
        d_iq, static_cast<const __half *>(d_ik_cache), d_pos, d_bscores,
        max_nblk, n_idx_heads, blk_size, prefix_blks, local_blks, dense_prefix_tokens);

    dim3 topk_grid(n_idx_heads, T);
    msa_indexer_select_topk_kernel<<<topk_grid, 256, 0, stream>>>(
        d_bscores, d_pos, d_out_blocks, d_out_k_count,
        max_nblk, n_idx_heads, blk_size, k_max, dense_prefix_tokens);
}

void launch_gqa_attn_msa(
    const float * d_q, const void * d_k_cache, const void * d_v_cache,
    bool kv_is_q8_0, const int32_t * d_pos,
    const int32_t * d_msa_blocks, const int32_t * d_msa_k_count,
    float * d_out, void * d_scratch, int T, int n_heads, int n_kv_heads, int head_dim, int blk_size, int k_max, float scale,
    cudaStream_t stream) {

    if (T <= 4 && d_scratch != nullptr) {
        constexpr int S = 16;
        const size_t n_slots = static_cast<size_t>(T) * n_heads * S;
        float * part_m = static_cast<float *>(d_scratch);
        float * part_l = part_m + n_slots;
        float * part_o = part_l + n_slots;

        dim3 grid(n_heads, S, T);
        if (kv_is_q8_0) {
            gqa_flash_attn_msa_splitk_d128_kernel<true, S><<<grid, 32, 0, stream>>>(
                d_q, d_k_cache, d_v_cache, d_pos, d_msa_blocks, d_msa_k_count,
                part_m, part_l, part_o, n_heads, n_kv_heads, blk_size, k_max, scale);
        } else {
            gqa_flash_attn_msa_splitk_d128_kernel<false, S><<<grid, 32, 0, stream>>>(
                d_q, d_k_cache, d_v_cache, d_pos, d_msa_blocks, d_msa_k_count,
                part_m, part_l, part_o, n_heads, n_kv_heads, blk_size, k_max, scale);
        }
        gqa_flash_attn_splitk_reduce_d128_kernel<S><<<dim3(n_heads, T), 32, 0, stream>>>(
            part_m, part_l, part_o, d_out, n_heads);
        return;
    }

    constexpr int wpb = 4;
    dim3 grid((n_heads + wpb - 1) / wpb, T);
    dim3 block(32, wpb);
    if (kv_is_q8_0) {
        gqa_flash_attn_msa_d128_warp_kernel<true><<<grid, block, 0, stream>>>(
            d_q, d_k_cache, d_v_cache, d_pos, d_msa_blocks, d_msa_k_count,
            d_out, n_heads, n_kv_heads, blk_size, k_max, scale);
    } else {
        gqa_flash_attn_msa_d128_warp_kernel<false><<<grid, block, 0, stream>>>(
            d_q, d_k_cache, d_v_cache, d_pos, d_msa_blocks, d_msa_k_count,
            d_out, n_heads, n_kv_heads, blk_size, k_max, scale);
    }
}

} // namespace rookery
