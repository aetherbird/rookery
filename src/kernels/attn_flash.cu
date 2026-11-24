#include "core/device.h"
#include "kernels/kernels.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cfloat>
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

// ============================================================================
// Tiled FlashAttention Prefill (T >= 8, head_dim == 128)
// Uses cublasGemmStridedBatchedEx:
//   - sm_60:  CUBLAS_COMPUTE_16F (18.7 TFLOPS native FP16 on P100!)
//   - sm_70+: CUBLAS_COMPUTE_32F Tensor Cores
//   - sm_61:  cublasSgemmStridedBatched (fast FP32 on GTX 1060)
// ============================================================================
template <typename TDst>
__global__ void gqa_q_to_kvh_d128_kernel(
    const float * __restrict__ q,
    TDst *        __restrict__ q_out,
    int T, int n_head, int n_kv, float scale) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = T * n_head * 128;
    if (idx >= tot) return;

    const int d   = idx & 127;
    const int rem = idx >> 7;
    const int h   = rem % n_head;
    const int t   = rem / n_head;
    const int gqa = n_head / n_kv;
    const int kvh = h / gqa;
    const int g   = h % gqa;

    const float val = q[idx] * scale;
    const size_t dst = ((static_cast<size_t>(kvh) * T + t) * gqa + g) * 128 + d;
    if constexpr (std::is_same_v<TDst, __half>) {
        q_out[dst] = __float2half_rn(val);
    } else {
        q_out[dst] = val;
    }
}

__global__ void gqa_init_flash_ml_kernel(float * __restrict__ d_m, float * __restrict__ d_l, int M_tot) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M_tot) {
        d_m[i] = -FLT_MAX;
        d_l[i] = 0.0f;
    }
}

template <bool KV_Q8_0, typename TDst>
__global__ void gqa_dequant_kv_tile_d128_kernel(
    const void * __restrict__ k_cache,
    const void * __restrict__ v_cache,
    TDst *       __restrict__ k_out,
    TDst *       __restrict__ v_out,
    int k0, int kc, int n_kv) {

    const int kvh = blockIdx.x;
    const int kp  = blockIdx.y; // 0..kc-1
    const int p   = k0 + kp;
    const size_t dst_base = (static_cast<size_t>(kvh) * kc + kp) * 128;

    if constexpr (KV_Q8_0) {
        // blockDim = (32, 4): 4 warps = 4 Q8_0 blocks of 32 elems
        const int lane = threadIdx.x;
        const int blk  = threadIdx.y;
        const auto * kc_b = static_cast<const BlockQ8_0 *>(k_cache);
        const auto * vc_b = static_cast<const BlockQ8_0 *>(v_cache);
        const size_t bidx = (static_cast<size_t>(p) * n_kv + kvh) * 4 + blk;
        const BlockQ8_0 & kb = kc_b[bidx];
        const BlockQ8_0 & vb = vc_b[bidx];

        __half kd_h, vd_h;
        *reinterpret_cast<uint16_t *>(&kd_h) = __ldg(&kb.d);
        *reinterpret_cast<uint16_t *>(&vd_h) = __ldg(&vb.d);
        const float kv = __half2float(kd_h) * static_cast<float>(__ldg(&kb.qs[lane]));
        const float vv = __half2float(vd_h) * static_cast<float>(__ldg(&vb.qs[lane]));
        const int d = blk * 32 + lane;
        if constexpr (std::is_same_v<TDst, __half>) {
            k_out[dst_base + d] = __float2half_rn(kv);
            v_out[dst_base + d] = __float2half_rn(vv);
        } else {
            k_out[dst_base + d] = kv;
            v_out[dst_base + d] = vv;
        }
    } else {
        // blockDim = 128
        const int d = threadIdx.x;
        const auto * kc_h = static_cast<const __half *>(k_cache);
        const auto * vc_h = static_cast<const __half *>(v_cache);
        const size_t src = (static_cast<size_t>(p) * n_kv + kvh) * 128 + d;
        const __half kh = __ldg(&kc_h[src]);
        const __half vh = __ldg(&vc_h[src]);
        if constexpr (std::is_same_v<TDst, __half>) {
            k_out[dst_base + d] = kh;
            v_out[dst_base + d] = vh;
        } else {
            k_out[dst_base + d] = __half2float(kh);
            v_out[dst_base + d] = __half2float(vh);
        }
    }
}

template <typename TS, typename TP>
__global__ void gqa_flash_softmax_causal_rescale_kernel(
    const TS *      __restrict__ s_tile, // [M_tot, kc]
    TP *            __restrict__ p_tile, // [M_tot, kc]
    const int32_t * __restrict__ pos,    // [T]
    float *         __restrict__ d_m,    // [M_tot]
    float *         __restrict__ d_l,    // [M_tot]
    float *         __restrict__ d_ob,   // [M_tot, 128]
    int T, int n_head, int n_kv, int k0, int kc, bool is_first_tile) {

    const int row = blockIdx.x; // 0..M_tot-1
    const int tid = threadIdx.x; // 0..127
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int gqa = n_head / n_kv;
    const int M_per_kv = T * gqa;
    const int rem = row % M_per_kv;
    const int t   = rem / gqa;
    const int q_pos = pos[t];

    const TS * s_row = s_tile + static_cast<size_t>(row) * kc;
    TP *       p_row = p_tile + static_cast<size_t>(row) * kc;

    // 1. Tile row max (masked by causal q_pos)
    float tmax = -FLT_MAX;
    for (int j = tid; j < kc; j += 128) {
        if (k0 + j <= q_pos) {
            float v = std::is_same_v<TS, __half> ? __half2float(s_row[j]) : static_cast<float>(s_row[j]);
            tmax = fmaxf(tmax, v);
        }
    }
    __shared__ float s_red[4];
    tmax = warp_max(tmax);
    if (lane == 0) s_red[warp] = tmax;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 4) ? s_red[lane] : -FLT_MAX;
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, 2));
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, 1));
        if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    const float tile_max = s_red[0];

    const float m_old = d_m[row];
    const float m_new = fmaxf(m_old, tile_max);
    const float alpha = (m_old > -1e30f && m_new > -1e30f) ? expf(m_old - m_new) : 0.0f;

    // 2. Compute exp(s - m_new) -> p_tile and sum
    float tsum = 0.0f;
    for (int j = tid; j < kc; j += 128) {
        float pv = 0.0f;
        if (k0 + j <= q_pos && m_new > -1e30f) {
            float v = std::is_same_v<TS, __half> ? __half2float(s_row[j]) : static_cast<float>(s_row[j]);
            pv = expf(v - m_new);
            tsum += pv;
        }
        if constexpr (std::is_same_v<TP, __half>) {
            p_row[j] = __float2half_rn(pv);
        } else {
            p_row[j] = pv;
        }
    }
    tsum = warp_sum(tsum);
    if (lane == 0) s_red[warp] = tsum;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 4) ? s_red[lane] : 0.0f;
        v += __shfl_down_sync(0xffffffffu, v, 2);
        v += __shfl_down_sync(0xffffffffu, v, 1);
        if (lane == 0) {
            d_m[row] = m_new;
            d_l[row] = d_l[row] * alpha + v;
        }
    }

    // 3. Rescale existing d_ob[row, 0..127] by alpha (or zero on first tile)
    float * ob = d_ob + static_cast<size_t>(row) * 128;
    if (is_first_tile) {
        ob[tid] = 0.0f;
    } else {
        ob[tid] *= alpha;
    }
}

__global__ void gqa_f16_add_to_f32_kernel(const __half * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t i2 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t idx = i2 * 2;
    if (idx + 1 < n) {
        const float2 sf = __half22float2(reinterpret_cast<const __half2 *>(src)[i2]);
        float2 df = reinterpret_cast<const float2 *>(dst)[i2];
        df.x += sf.x;
        df.y += sf.y;
        reinterpret_cast<float2 *>(dst)[i2] = df;
    } else if (idx < n) {
        dst[idx] += __half2float(src[idx]);
    }
}

__global__ void gqa_o_from_kvh_finalize_d128_kernel(
    const float * __restrict__ d_ob,
    const float * __restrict__ d_l,
    float *       __restrict__ out,
    int T, int n_head, int n_kv) {

    const int row = blockIdx.x;
    const int d   = threadIdx.x; // 0..127
    const int gqa = n_head / n_kv;
    const int M_per_kv = T * gqa;
    const int kvh = row / M_per_kv;
    const int rem = row % M_per_kv;
    const int t   = rem / gqa;
    const int g   = rem % gqa;
    const int h   = kvh * gqa + g;

    const float l = d_l[row];
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    out[(static_cast<size_t>(t) * n_head + h) * 128 + d] =
        d_ob[static_cast<size_t>(row) * 128 + d] * inv_l;
}

// Zero-syncthreads 1-warp-per-head online-softmax FlashAttention for GQA (head_dim = 128).
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
    cudaStream_t stream,
    cublasHandle_t cublas, size_t scratch_bytes, int kv_len_max) {

    // 1. Tiled FlashAttention Prefill (T >= 8, head_dim == 128) via strided batched cuBLAS!
    if (T >= 8 && head_dim == 128 && cublas != nullptr && d_scratch != nullptr && kv_len_max > 0) {
        const int  sm_ver       = device_sm_version(-1);
        const bool use_tc       = (sm_ver >= 70);
        const bool use_sm60_f16 = (sm_ver == 60);
        const int  gqa          = n_heads / n_kv_heads;
        const int  M_per_kv     = T * gqa;
        const int  M_tot        = n_kv_heads * M_per_kv;
        auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };

        if (use_tc) {
            constexpr int K_TILE = 512;
            const size_t q16_b = align256(static_cast<size_t>(M_tot) * 128 * sizeof(__half));
            const size_t k16_b = align256(static_cast<size_t>(n_kv_heads) * K_TILE * 128 * sizeof(__half));
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

                const int tot_q = T * n_heads * 128;
                gqa_q_to_kvh_d128_kernel<__half><<<(tot_q + 255) / 256, 256, 0, stream>>>(
                    d_q, d_q16, T, n_heads, n_kv_heads, scale);
                gqa_init_flash_ml_kernel<<<(M_tot + 255) / 256, 256, 0, stream>>>(d_m, d_l, M_tot);

                const float one = 1.0f, zero = 0.0f;
                cublasSetStream(cublas, stream);

                for (int k0 = 0; k0 < kv_len_max; k0 += K_TILE) {
                    const int kc = (k0 + K_TILE < kv_len_max) ? K_TILE : (kv_len_max - k0);
                    if (kv_is_q8_0) {
                        gqa_dequant_kv_tile_d128_kernel<true, __half><<<dim3(n_kv_heads, kc), dim3(32, 4), 0, stream>>>(
                            d_k_cache, d_v_cache, d_k16, d_v16, k0, kc, n_kv_heads);
                    } else {
                        gqa_dequant_kv_tile_d128_kernel<false, __half><<<dim3(n_kv_heads, kc), 128, 0, stream>>>(
                            d_k_cache, d_v_cache, d_k16, d_v16, k0, kc, n_kv_heads);
                    }

                    cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        kc, M_per_kv, 128,
                        &one,
                        d_k16, CUDA_R_16F, 128, static_cast<long long>(kc) * 128,
                        d_q16, CUDA_R_16F, 128, static_cast<long long>(M_per_kv) * 128,
                        &zero,
                        d_s32, CUDA_R_32F, kc,  static_cast<long long>(M_per_kv) * kc,
                        n_kv_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                    gqa_flash_softmax_causal_rescale_kernel<float, __half><<<M_tot, 128, 0, stream>>>(
                        d_s32, d_p16, d_pos, d_m, d_l, d_ob, T, n_heads, n_kv_heads, k0, kc, k0 == 0);

                    cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        128, M_per_kv, kc,
                        &one,
                        d_v16, CUDA_R_16F, 128, static_cast<long long>(kc) * 128,
                        d_p16, CUDA_R_16F, kc,  static_cast<long long>(M_per_kv) * kc,
                        &one,
                        d_ob,  CUDA_R_32F, 128, static_cast<long long>(M_per_kv) * 128,
                        n_kv_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                }

                gqa_o_from_kvh_finalize_d128_kernel<<<M_tot, 128, 0, stream>>>(d_ob, d_l, d_out, T, n_heads, n_kv_heads);
                return;
            }
        } else if (use_sm60_f16) {
            constexpr int K_TILE = 512;
            const size_t q16_b = align256(static_cast<size_t>(M_tot) * 128 * sizeof(__half));
            const size_t k16_b = align256(static_cast<size_t>(n_kv_heads) * K_TILE * 128 * sizeof(__half));
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

                const int tot_q = T * n_heads * 128;
                gqa_q_to_kvh_d128_kernel<__half><<<(tot_q + 255) / 256, 256, 0, stream>>>(
                    d_q, d_q16, T, n_heads, n_kv_heads, scale);
                gqa_init_flash_ml_kernel<<<(M_tot + 255) / 256, 256, 0, stream>>>(d_m, d_l, M_tot);

                const __half h_one = __float2half_rn(1.0f), h_zero = __float2half_rn(0.0f);
                cublasSetStream(cublas, stream);

                for (int k0 = 0; k0 < kv_len_max; k0 += K_TILE) {
                    const int kc = (k0 + K_TILE < kv_len_max) ? K_TILE : (kv_len_max - k0);
                    if (kv_is_q8_0) {
                        gqa_dequant_kv_tile_d128_kernel<true, __half><<<dim3(n_kv_heads, kc), dim3(32, 4), 0, stream>>>(
                            d_k_cache, d_v_cache, d_k16, d_v16, k0, kc, n_kv_heads);
                    } else {
                        gqa_dequant_kv_tile_d128_kernel<false, __half><<<dim3(n_kv_heads, kc), 128, 0, stream>>>(
                            d_k_cache, d_v_cache, d_k16, d_v16, k0, kc, n_kv_heads);
                    }

                    cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        kc, M_per_kv, 128,
                        &h_one,
                        d_k16, CUDA_R_16F, 128, static_cast<long long>(kc) * 128,
                        d_q16, CUDA_R_16F, 128, static_cast<long long>(M_per_kv) * 128,
                        &h_zero,
                        d_s16, CUDA_R_16F, kc,  static_cast<long long>(M_per_kv) * kc,
                        n_kv_heads, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);

                    gqa_flash_softmax_causal_rescale_kernel<__half, __half><<<M_tot, 128, 0, stream>>>(
                        d_s16, d_p16, d_pos, d_m, d_l, d_ob, T, n_heads, n_kv_heads, k0, kc, k0 == 0);

                    cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        128, M_per_kv, kc,
                        &h_one,
                        d_v16, CUDA_R_16F, 128, static_cast<long long>(kc) * 128,
                        d_p16, CUDA_R_16F, kc,  static_cast<long long>(M_per_kv) * kc,
                        &h_zero,
                        d_o16, CUDA_R_16F, 128, static_cast<long long>(M_per_kv) * 128,
                        n_kv_heads, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);

                    const int64_t no = static_cast<int64_t>(M_tot) * 128;
                    gqa_f16_add_to_f32_kernel<<<((no + 1) / 2 + 255) / 256, 256, 0, stream>>>(d_o16, d_ob, no);
                }

                gqa_o_from_kvh_finalize_d128_kernel<<<M_tot, 128, 0, stream>>>(d_ob, d_l, d_out, T, n_heads, n_kv_heads);
                return;
            }
        } else {
            constexpr int K_TILE = 256;
            const size_t q32_b = align256(static_cast<size_t>(M_tot) * 128 * sizeof(float));
            const size_t k32_b = align256(static_cast<size_t>(n_kv_heads) * K_TILE * 128 * sizeof(float));
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

                const int tot_q = T * n_heads * 128;
                gqa_q_to_kvh_d128_kernel<float><<<(tot_q + 255) / 256, 256, 0, stream>>>(
                    d_q, d_q32, T, n_heads, n_kv_heads, scale);
                gqa_init_flash_ml_kernel<<<(M_tot + 255) / 256, 256, 0, stream>>>(d_m, d_l, M_tot);

                const float one = 1.0f, zero = 0.0f;
                cublasSetStream(cublas, stream);

                for (int k0 = 0; k0 < kv_len_max; k0 += K_TILE) {
                    const int kc = (k0 + K_TILE < kv_len_max) ? K_TILE : (kv_len_max - k0);
                    if (kv_is_q8_0) {
                        gqa_dequant_kv_tile_d128_kernel<true, float><<<dim3(n_kv_heads, kc), dim3(32, 4), 0, stream>>>(
                            d_k_cache, d_v_cache, d_k32, d_v32, k0, kc, n_kv_heads);
                    } else {
                        gqa_dequant_kv_tile_d128_kernel<false, float><<<dim3(n_kv_heads, kc), 128, 0, stream>>>(
                            d_k_cache, d_v_cache, d_k32, d_v32, k0, kc, n_kv_heads);
                    }

                    cublasSgemmStridedBatched(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        kc, M_per_kv, 128,
                        &one,
                        d_k32, 128, static_cast<long long>(kc) * 128,
                        d_q32, 128, static_cast<long long>(M_per_kv) * 128,
                        &zero,
                        d_s32, kc,  static_cast<long long>(M_per_kv) * kc,
                        n_kv_heads);

                    gqa_flash_softmax_causal_rescale_kernel<float, float><<<M_tot, 128, 0, stream>>>(
                        d_s32, d_p32, d_pos, d_m, d_l, d_ob, T, n_heads, n_kv_heads, k0, kc, k0 == 0);

                    cublasSgemmStridedBatched(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        128, M_per_kv, kc,
                        &one,
                        d_v32, 128, static_cast<long long>(kc) * 128,
                        d_p32, kc,  static_cast<long long>(M_per_kv) * kc,
                        &one,
                        d_ob,  128, static_cast<long long>(M_per_kv) * 128,
                        n_kv_heads);
                }

                gqa_o_from_kvh_finalize_d128_kernel<<<M_tot, 128, 0, stream>>>(d_ob, d_l, d_out, T, n_heads, n_kv_heads);
                return;
            }
        }
    }

    // 2. Reuse Split-K GQA-8 Shared-KV FlashAttention for decode/MTP (T <= 4)!
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
