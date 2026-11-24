#include "core/device.h"
#include "kernels/kernels.h"
#include "kernels/q8_1_act.cuh"
#include "kernels/q8_1_act.cuh"

#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
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

// Batched per-head Q8_0 SoA matrix-vector multiply with FUSED on-the-fly Q8_1 shared-memory
// activation quantization + 100% coalesced LDG.128 + dp4a.s32!
// W has shape [K, M, n_heads] (ne[0]=K, ne[1]=M, ne[2]=n_heads), K <= 1024 (256 or 512 for MLA!).
// X has shape [T, n_heads, K], Y has shape [T, n_heads, M].
__global__ void batched_head_proj_q8_0_soa_kernel(
    const int4 *     __restrict__ qs_i4,  // [n_heads, M, K/16]
    const uint16_t * __restrict__ scales, // [n_heads, M, K/32]
    const float *    __restrict__ x,      // [T, n_heads, K]
    float *          __restrict__ y,      // [T, n_heads, M]
    int M, int K, int n_heads) {

    __shared__ BlockQ8_1_Act s_x[32]; // up to K = 1024 (32 * 48 = 1536 bytes!)

    const int h    = blockIdx.y;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int nb   = K >> 5; // 8 (K=256) or 16 (K=512)

    // 1. All 8 warps cooperatively quantize x[t, h, 0..K-1] into shared memory s_x[0..nb-1]!
    const float * xp = x + (static_cast<size_t>(t) * n_heads + h) * K;
    for (int b = warp; b < nb; b += blockDim.y) {
        const float v = xp[b * 32 + lane];
        const float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(v)), 0);
        const float d = amax / 127.0f;
        const float inv_d = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        if (lane == 0) s_x[b].d = d;
        s_x[b].qs[lane] = static_cast<int8_t>(__float2int_rn(v * inv_d));
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + warp;
    if (row >= M) return;

    const int blk_in_warp = lane >> 1; // 0..15 (processes up to 16 Q8_0 blocks in 1 iteration!)
    const int half_in_blk = lane & 1;  // 0 or 1 (LDG.128 = 16 bytes)
    const int4 *     w_q_row = qs_i4  + (static_cast<size_t>(h) * M + row) * (nb * 2);
    const uint16_t * w_s_row = scales + (static_cast<size_t>(h) * M + row) * nb;

    float acc = 0.0f;
    for (int b = blk_in_warp; b < nb; b += 16) {
        const int4 w4 = __ldg(&w_q_row[b * 2 + half_in_blk]);
        const int4 x4 = reinterpret_cast<const int4 *>(s_x[b].qs)[half_in_blk];
        int sumi = dp4a_s32(w4.x, x4.x, 0);
        sumi     = dp4a_s32(w4.y, x4.y, sumi);
        sumi     = dp4a_s32(w4.z, x4.z, sumi);
        sumi     = dp4a_s32(w4.w, x4.w, sumi);
        __half dh;
        *reinterpret_cast<uint16_t *>(&dh) = __ldg(&w_s_row[b]);
        acc += (__half2float(dh) * s_x[b].d) * static_cast<float>(sumi);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        y[(static_cast<size_t>(t) * n_heads + h) * M + row] = acc;
    }
}

// Write single-head 512-dim latent KV vector into Q8_0 or F16 KV cache
__global__ void mla_kv_write_q8_0_kernel(
    const float *   __restrict__ kv,       // [T, 512]
    BlockQ8_0 *     __restrict__ kv_cache, // [max_ctx, 16] (512 / 32 = 16 Q8_0 blocks per pos)
    const int32_t * __restrict__ pos,
    int T) {

    const int b    = blockIdx.x * blockDim.y + threadIdx.y; // 0..15
    const int t    = blockIdx.y;
    if (t >= T || b >= 16) return;

    const int lane = threadIdx.x;
    const int p    = pos[t];
    const float v  = kv[static_cast<size_t>(t) * 512 + b * 32 + lane];
    const float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(v)), 0);
    const float d = amax / 127.0f;
    const float inv_d = (d > 0.0f) ? (127.0f / amax) : 0.0f;

    BlockQ8_0 & dst = kv_cache[static_cast<size_t>(p) * 16 + b];
    if (lane == 0) {
        __half dh = __float2half_rn(d);
        dst.d = *reinterpret_cast<uint16_t *>(&dh);
    }
    dst.qs[lane] = static_cast<int8_t>(__float2int_rn(v * inv_d));
}

__global__ void mla_kv_write_f16_kernel(
    const float *   __restrict__ kv,
    __half *        __restrict__ kv_cache,
    const int32_t * __restrict__ pos,
    int T, int kv_dim) {

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (t >= T || d >= kv_dim) return;
    kv_cache[static_cast<size_t>(pos[t]) * kv_dim + d] = __float2half_rn(kv[static_cast<size_t>(t) * kv_dim + d]);
}

// ============================================================================
// Split-K Absorbed MLA MQA FlashAttention over single 512-dim latent KV cache!
// K == V == kv_cache[p, 512] -> reads each latent vector ONCE for BOTH score & value!
// Each warp (32 lanes) owns 1 head `h` (0..63), 16 dims per lane (512 / 32 = 16 floats)!
// ============================================================================
template <bool KV_Q8_0, int SPLITS>
__global__ void mla_mqa_d512_splitk_kernel(
    const float *   __restrict__ q_abs,    // [T, n_heads, 512]
    const void *    __restrict__ kv_cache, // [max_ctx, 512] (Q8_0 or F16)
    const int32_t * __restrict__ pos,      // [T]
    float *         __restrict__ part_m,   // [T, n_heads, SPLITS]
    float *         __restrict__ part_l,   // [T, n_heads, SPLITS]
    float *         __restrict__ part_o,   // [T, n_heads, SPLITS, 512]
    int n_heads, float scale) {

    const int h    = blockIdx.x;
    const int s    = blockIdx.y;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x; // 0..31

    const int kv_len = pos[t] + 1;
    const int chunk  = (kv_len + SPLITS - 1) / SPLITS;
    const int p0     = s * chunk;
    const int p1     = (p0 + chunk < kv_len) ? (p0 + chunk) : kv_len;

    // Load 16 dims of q_abs for this lane (lane*16 .. lane*16+15 -> 4 float4s!)
    const auto * q4p = reinterpret_cast<const float4 *>(q_abs + (static_cast<size_t>(t) * n_heads + h) * 512) + lane * 4;
    float qv[16];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        float4 tmp = q4p[g];
        qv[g * 4 + 0] = tmp.x * scale;
        qv[g * 4 + 1] = tmp.y * scale;
        qv[g * 4 + 2] = tmp.z * scale;
        qv[g * 4 + 3] = tmp.w * scale;
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float ov[16] = {0};

    const int blk_idx = lane >> 1;          // 0..15 (which of the 16 Q8_0 blocks in 512 dims)
    const int blk_off = (lane & 1) * 16;    // 0 or 16

    for (int p = p0; p < p1; ++p) {
        float kv[16];
        if constexpr (KV_Q8_0) {
            const BlockQ8_0 & kb = static_cast<const BlockQ8_0 *>(kv_cache)[static_cast<size_t>(p) * 16 + blk_idx];
            __half dh;
            *reinterpret_cast<uint16_t *>(&dh) = __ldg(&kb.d);
            const float d = __half2float(dh);
            #pragma unroll
            for (int i = 0; i < 16; ++i) kv[i] = d * static_cast<float>(kb.qs[blk_off + i]);
        } else {
            const auto * kc = static_cast<const uint2 *>(kv_cache) + static_cast<size_t>(p) * 128 + lane * 4;
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                uint2 pk = __ldg(&kc[g]);
                float2 f0 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.x));
                float2 f1 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.y));
                kv[g * 4 + 0] = f0.x; kv[g * 4 + 1] = f0.y;
                kv[g * 4 + 2] = f1.x; kv[g * 4 + 3] = f1.y;
            }
        }

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) dot += qv[i] * kv[i];
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffffu, dot, 0);

        const float m_new = fmaxf(m_i, score);
        const float alpha = expf(m_i - m_new);
        const float beta  = expf(score - m_new);
        l_i = l_i * alpha + beta;
        #pragma unroll
        for (int i = 0; i < 16; ++i) ov[i] = ov[i] * alpha + beta * kv[i];
        m_i = m_new;
    }

    const size_t slot = (static_cast<size_t>(t) * n_heads + h) * SPLITS + s;
    if (lane == 0) {
        part_m[slot] = m_i;
        part_l[slot] = l_i;
    }
    auto * o4p = reinterpret_cast<float4 *>(part_o + slot * 512) + lane * 4;
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        o4p[g] = make_float4(ov[g * 4 + 0], ov[g * 4 + 1], ov[g * 4 + 2], ov[g * 4 + 3]);
    }
}

template <int SPLITS>
__global__ void mla_mqa_d512_reduce_kernel(
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
        if (part_l[base + s] > 0.0f) m_max = fmaxf(m_max, part_m[base + s]);
    }

    float l_tot = 0.0f;
    float ov[16] = {0};
    for (int s = 0; s < SPLITS; ++s) {
        const float ls = part_l[base + s];
        if (ls <= 0.0f) continue;
        const float sc = expf(part_m[base + s] - m_max);
        l_tot += ls * sc;
        const auto * po4 = reinterpret_cast<const float4 *>(part_o + (base + s) * 512) + lane * 4;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            float4 v = po4[g];
            ov[g * 4 + 0] += v.x * sc; ov[g * 4 + 1] += v.y * sc;
            ov[g * 4 + 2] += v.z * sc; ov[g * 4 + 3] += v.w * sc;
        }
    }

    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    auto * o4p = reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * n_heads + h) * 512) + lane * 4;
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        o4p[g] = make_float4(ov[g * 4 + 0] * inv_l, ov[g * 4 + 1] * inv_l, ov[g * 4 + 2] * inv_l, ov[g * 4 + 3] * inv_l);
    }
}

template <typename TMat>
__device__ __forceinline__ TMat mla_f32_to_mat(float v) {
    if constexpr (std::is_same_v<TMat, __half>) return __float2half_rn(v);
    else return v;
}

template <typename TMat>
__device__ __forceinline__ TMat mla_f16_to_mat(__half h) {
    if constexpr (std::is_same_v<TMat, __half>) return h;
    else return __half2float(h);
}

template <typename TMat>
__global__ void dequant_q8_0_soa_mla_kernel(
    const int8_t *   __restrict__ qs,
    const uint16_t * __restrict__ scales,
    TMat *           __restrict__ dst,
    int64_t n_blocks) {
    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (b >= n_blocks) return;
    const int lane = threadIdx.x;
    __half dh;
    *reinterpret_cast<uint16_t *>(&dh) = __ldg(&scales[b]);
    dst[b * 32 + lane] = mla_f32_to_mat<TMat>(__half2float(dh) * static_cast<float>(__ldg(&qs[b * 32 + lane])));
}

// Transpose [T, H, D] F32 -> [H, T, D] F16/F32 for strided batched cuBLAS HGEMM/SGEMM!
template <typename TMat>
__global__ void transpose_thd_to_htd_kernel(
    const float * __restrict__ src,
    TMat *        __restrict__ dst,
    int T, int H, int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = T * H * D;
    if (idx >= tot) return;
    const int d = idx % D;
    const int h = (idx / D) % H;
    const int t = idx / (H * D);
    dst[(static_cast<size_t>(h) * T + t) * D + d] = mla_f32_to_mat<TMat>(src[idx]);
}

// Transpose [H, T, D] F32 -> [T, H, D] F32
__global__ void transpose_htd_to_thd_f32_kernel(
    const float * __restrict__ src,
    float *       __restrict__ dst,
    int T, int H, int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = T * H * D;
    if (idx >= tot) return;
    const int d = idx % D;
    const int t = (idx / D) % T;
    const int h = idx / (T * D);
    dst[(static_cast<size_t>(t) * H + h) * D + d] = src[idx];
}

// ============================================================================
// Multi-Head-Shared Split-K Absorbed MLA Decode:
//   - T==1: H_PER_BLK=16 heads share 1 KV load (only 4 KV sweeps across all 64 heads!)
//   - T==2 (MTP): 16 heads x 2 tokens = 32 warps/CTA share 1 KV load (4x fewer KV reads vs H=8,T=1!)
// ============================================================================
template <bool KV_Q8_0, int SPLITS, int H_PER_BLK>
__global__ void mla_mqa_d512_decode_shared_kernel(
    const float *   __restrict__ q_abs,    // [T, n_heads, 512]
    const void *    __restrict__ kv_cache, // [max_ctx, 512] (Q8_0 or F16)
    const int32_t * __restrict__ pos,      // [T]
    float *         __restrict__ part_m,   // [T, n_heads, SPLITS]
    float *         __restrict__ part_l,   // [T, n_heads, SPLITS]
    float *         __restrict__ part_o,   // [T, n_heads, SPLITS, 512]
    int n_heads, float scale) {

    constexpr int P_TILE = 8; // 16 KB shared memory, loaded cooperatively by 16 warps (2 warps/pos)
    __shared__ float s_kv[P_TILE][512];

    const int h_base = blockIdx.x * H_PER_BLK;
    const int s      = blockIdx.y;
    const int t      = blockIdx.z;
    const int warp   = threadIdx.y; // 0..H_PER_BLK-1 (16)
    const int lane   = threadIdx.x; // 0..31
    const int h      = h_base + warp;

    const int kv_len = pos[t] + 1;
    const int chunk  = (kv_len + SPLITS - 1) / SPLITS;
    const int p0     = s * chunk;
    const int p1     = (p0 + chunk < kv_len) ? (p0 + chunk) : kv_len;

    float qv[16];
    if (h < n_heads) {
        const auto * q4p = reinterpret_cast<const float4 *>(q_abs + (static_cast<size_t>(t) * n_heads + h) * 512) + lane * 4;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            float4 tmp = q4p[g];
            qv[g * 4 + 0] = tmp.x * scale;
            qv[g * 4 + 1] = tmp.y * scale;
            qv[g * 4 + 2] = tmp.z * scale;
            qv[g * 4 + 3] = tmp.w * scale;
        }
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float ov[16] = {0};

    // 16 warps -> 8 positions x 2 halves (256 elems/warp)
    const int p_sub    = warp >> 1; // 0..7
    const int half_512 = warp & 1;  // 0..1

    for (int p = p0; p < p1; p += P_TILE) {
        const int n_p = (p + P_TILE < p1) ? P_TILE : (p1 - p);

        if (p_sub < n_p) {
            const int gp = p + p_sub;
            if constexpr (KV_Q8_0) {
                const BlockQ8_0 * kb_row = static_cast<const BlockQ8_0 *>(kv_cache) + static_cast<size_t>(gp) * 16 + half_512 * 8;
                float * dst = &s_kv[p_sub][half_512 * 256];
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    __half dh;
                    *reinterpret_cast<uint16_t *>(&dh) = __ldg(&kb_row[b].d);
                    const float d = __half2float(dh);
                    dst[b * 32 + lane] = d * static_cast<float>(__ldg(&kb_row[b].qs[lane]));
                }
            } else {
                const auto * kc = static_cast<const uint2 *>(kv_cache) + static_cast<size_t>(gp) * 128 + half_512 * 64;
                auto * dst4 = reinterpret_cast<float4 *>(&s_kv[p_sub][half_512 * 256]);
                #pragma unroll
                for (int it = 0; it < 2; ++it) {
                    const int i = it * 32 + lane;
                    uint2 pk = __ldg(&kc[i]);
                    float2 f0 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.x));
                    float2 f1 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.y));
                    dst4[i] = make_float4(f0.x, f0.y, f1.x, f1.y);
                }
            }
        }
        __syncthreads();

        if (h < n_heads) {
            #pragma unroll
            for (int pi = 0; pi < P_TILE; ++pi) {
                if (pi >= n_p) break;
                const auto * k4p = reinterpret_cast<const float4 *>(&s_kv[pi][lane * 16]);
                float kv[16];
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    float4 tmp = k4p[g];
                    kv[g * 4 + 0] = tmp.x; kv[g * 4 + 1] = tmp.y;
                    kv[g * 4 + 2] = tmp.z; kv[g * 4 + 3] = tmp.w;
                }
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) dot += qv[i] * kv[i];
                dot = warp_sum(dot);
                const float score = __shfl_sync(0xffffffffu, dot, 0);

                const float m_new = fmaxf(m_i, score);
                const float alpha = expf(m_i - m_new);
                const float beta  = expf(score - m_new);
                l_i = l_i * alpha + beta;
                #pragma unroll
                for (int i = 0; i < 16; ++i) ov[i] = ov[i] * alpha + beta * kv[i];
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (h < n_heads) {
        const size_t slot = (static_cast<size_t>(t) * n_heads + h) * SPLITS + s;
        if (lane == 0) {
            part_m[slot] = m_i;
            part_l[slot] = l_i;
        }
        auto * o4p = reinterpret_cast<float4 *>(part_o + slot * 512) + lane * 4;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            o4p[g] = make_float4(ov[g * 4 + 0], ov[g * 4 + 1], ov[g * 4 + 2], ov[g * 4 + 3]);
        }
    }
}

// Joint T=2 MTP Split-K MLA Decode: 8 heads x 2 tokens = 16 warps (512 threads/CTA) share 1 KV load!
// 16x less KV DRAM traffic vs 1-head-per-CTA and safely within the 65K register/CTA limit on sm_70 & sm_86!
template <bool KV_Q8_0, int SPLITS>
__global__ __launch_bounds__(512) void mla_mqa_d512_decode_shared_t2_kernel(
    const float *   __restrict__ q_abs,    // [2, n_heads, 512]
    const void *    __restrict__ kv_cache, // [max_ctx, 512]
    const int32_t * __restrict__ pos,      // [2]
    float *         __restrict__ part_m,   // [2, n_heads, SPLITS]
    float *         __restrict__ part_l,   // [2, n_heads, SPLITS]
    float *         __restrict__ part_o,   // [2, n_heads, SPLITS, 512]
    int n_heads, float scale) {

    constexpr int H_PER_BLK = 8;
    constexpr int P_TILE    = 8; // 16 KB shared memory, loaded by 16 warps (2 warps/pos = 8 blks/warp!)
    __shared__ float s_kv[P_TILE][512];

    const int h_base = blockIdx.x * H_PER_BLK;
    const int s      = blockIdx.y;
    const int warp   = threadIdx.y; // 0..15: 0..7 -> t=0, 8..15 -> t=1
    const int lane   = threadIdx.x; // 0..31
    const int t      = warp >> 3;   // 0 or 1
    const int h      = h_base + (warp & 7);

    const int max_p  = max(pos[0], pos[1]);
    const int kv_len = max_p + 1;
    const int my_max = pos[t];
    const int chunk  = (kv_len + SPLITS - 1) / SPLITS;
    const int p0     = s * chunk;
    const int p1     = (p0 + chunk < kv_len) ? (p0 + chunk) : kv_len;

    float qv[16];
    if (h < n_heads) {
        const auto * q4p = reinterpret_cast<const float4 *>(q_abs + (static_cast<size_t>(t) * n_heads + h) * 512) + lane * 4;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            float4 tmp = q4p[g];
            qv[g * 4 + 0] = tmp.x * scale;
            qv[g * 4 + 1] = tmp.y * scale;
            qv[g * 4 + 2] = tmp.z * scale;
            qv[g * 4 + 3] = tmp.w * scale;
        }
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float ov[16] = {0};

    // 16 warps -> 8 positions x 2 halves (256 elems = 8 Q8_0 blocks per warp!)
    const int p_sub    = warp >> 1; // 0..7
    const int half_512 = warp & 1;  // 0..1

    for (int p = p0; p < p1; p += P_TILE) {
        const int n_p = (p + P_TILE < p1) ? P_TILE : (p1 - p);

        if (p_sub < n_p) {
            const int gp = p + p_sub;
            if constexpr (KV_Q8_0) {
                const BlockQ8_0 * kb_row = static_cast<const BlockQ8_0 *>(kv_cache) + static_cast<size_t>(gp) * 16 + half_512 * 8;
                float * dst = &s_kv[p_sub][half_512 * 256];
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    __half dh;
                    *reinterpret_cast<uint16_t *>(&dh) = __ldg(&kb_row[b].d);
                    const float d = __half2float(dh);
                    dst[b * 32 + lane] = d * static_cast<float>(__ldg(&kb_row[b].qs[lane]));
                }
            } else {
                const auto * kc = static_cast<const uint2 *>(kv_cache) + static_cast<size_t>(gp) * 128 + half_512 * 64;
                auto * dst4 = reinterpret_cast<float4 *>(&s_kv[p_sub][half_512 * 256]);
                #pragma unroll
                for (int it = 0; it < 2; ++it) {
                    const int i = it * 32 + lane;
                    uint2 pk = __ldg(&kc[i]);
                    float2 f0 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.x));
                    float2 f1 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.y));
                    dst4[i] = make_float4(f0.x, f0.y, f1.x, f1.y);
                }
            }
        }
        __syncthreads();

        if (h < n_heads) {
            #pragma unroll
            for (int pi = 0; pi < P_TILE; ++pi) {
                if (pi >= n_p) break;
                if (p + pi > my_max) break; // causal mask for t=0 vs t=1!
                const auto * k4p = reinterpret_cast<const float4 *>(&s_kv[pi][lane * 16]);
                float kv[16];
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    float4 tmp = k4p[g];
                    kv[g * 4 + 0] = tmp.x; kv[g * 4 + 1] = tmp.y;
                    kv[g * 4 + 2] = tmp.z; kv[g * 4 + 3] = tmp.w;
                }
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) dot += qv[i] * kv[i];
                dot = warp_sum(dot);
                const float score = __shfl_sync(0xffffffffu, dot, 0);

                const float m_new = fmaxf(m_i, score);
                const float alpha = expf(m_i - m_new);
                const float beta  = expf(score - m_new);
                l_i = l_i * alpha + beta;
                #pragma unroll
                for (int i = 0; i < 16; ++i) ov[i] = ov[i] * alpha + beta * kv[i];
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (h < n_heads) {
        const size_t slot = (static_cast<size_t>(t) * n_heads + h) * SPLITS + s;
        if (lane == 0) {
            part_m[slot] = m_i;
            part_l[slot] = l_i;
        }
        auto * o4p = reinterpret_cast<float4 *>(part_o + slot * 512) + lane * 4;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            o4p[g] = make_float4(ov[g * 4 + 0], ov[g * 4 + 1], ov[g * 4 + 2], ov[g * 4 + 3]);
        }
    }
}

// ============================================================================
// Tiled Tensor Core FlashMLA Prefill (T >= 8) — ~20-30x faster at 32K+ context!
// Uses cublasGemmEx FP16 Tensor Cores on shared 512-dim latent KV cache:
//   Tile K_tile=512: S = Q_scaled @ K_tile^T -> online softmax + causal mask -> O = alpha*O + P @ K_tile
// ============================================================================
template <typename TMat>
__global__ void mla_scale_cast_kernel(
    const float * __restrict__ src, TMat * __restrict__ dst, int64_t n, float scale) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = mla_f32_to_mat<TMat>(src[i] * scale);
}

__global__ void mla_init_flash_state_kernel(float * __restrict__ m, float * __restrict__ l, int M) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) { m[i] = -FLT_MAX; l[i] = 0.0f; }
}

template <bool KV_Q8_0, typename TMat>
__global__ void mla_dequant_kv_tile_kernel(
    const void * __restrict__ kv_cache,
    TMat *       __restrict__ dst, // [kc, 512]
    int k0, int kc) {
    const int c = blockIdx.y;
    if (c >= kc) return;
    const int p = k0 + c;
    if constexpr (KV_Q8_0) {
        const int b    = blockIdx.x * blockDim.y + threadIdx.y; // 0..15
        const int lane = threadIdx.x;                           // 0..31
        if (b >= 16) return;
        const BlockQ8_0 & kb = static_cast<const BlockQ8_0 *>(kv_cache)[static_cast<size_t>(p) * 16 + b];
        __half dh;
        *reinterpret_cast<uint16_t *>(&dh) = __ldg(&kb.d);
        const float d = __half2float(dh);
        dst[static_cast<size_t>(c) * 512 + b * 32 + lane] = mla_f32_to_mat<TMat>(d * static_cast<float>(__ldg(&kb.qs[lane])));
    } else {
        const int d = blockIdx.x * blockDim.x + threadIdx.x;
        if (d < 512) {
            dst[static_cast<size_t>(c) * 512 + d] = mla_f16_to_mat<TMat>(static_cast<const __half *>(kv_cache)[static_cast<size_t>(p) * 512 + d]);
        }
    }
}

// 1 block of 128 threads per row r = t * n_heads + h (M = T * n_heads rows):
// Applies causal mask (k0 + c <= pos[t]), updates online (m[r], l[r]), writes P_tile[r, c] in TMat,
// AND rescales d_out[r, 0..511] *= alpha in-place (zero extra memory pass!).
template <typename TMat>
__global__ void mla_flash_online_softmax_rescale_kernel(
    const float *   __restrict__ s_tile, // [M, kc]
    TMat *          __restrict__ p_tile, // [M, kc]
    const int32_t * __restrict__ pos,    // [T]
    float *         __restrict__ m_buf,  // [M]
    float *         __restrict__ l_buf,  // [M]
    float *         __restrict__ d_out,  // [M, 512]
    int n_heads, int k0, int kc) {

    const int r    = blockIdx.x;
    const int t    = r / n_heads;
    const int maxp = pos[t];
    const int tid  = threadIdx.x; // 0..127
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float * s_row = s_tile + static_cast<size_t>(r) * kc;
    TMat *        p_row = p_tile + static_cast<size_t>(r) * kc;

    // Fast path: entire tile is strictly beyond causal limit pos[t]
    if (k0 > maxp) {
        for (int c = tid; c < kc; c += blockDim.x) p_row[c] = mla_f32_to_mat<TMat>(0.0f);
        return;
    }

    const int valid_c = (k0 + kc - 1 <= maxp) ? kc : (maxp - k0 + 1);

    // 1. Tile max across valid_c columns
    float local_m = -FLT_MAX;
    for (int c = tid; c < valid_c; c += blockDim.x) {
        local_m = fmaxf(local_m, s_row[c]);
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
    const float alpha = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);

    // 2. Compute exp(s - m_new), cast to TMat, accumulate tile sum
    float local_l = 0.0f;
    for (int c = tid; c < kc; c += blockDim.x) {
        if (c < valid_c) {
            const float e = expf(s_row[c] - m_new);
            p_row[c] = mla_f32_to_mat<TMat>(e);
            local_l += e;
        } else {
            p_row[c] = mla_f32_to_mat<TMat>(0.0f);
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

    // 3. Rescale existing d_out[r, 0..511] *= alpha in-place (128 threads x float4 = 512 floats in 1 op!)
    auto * o4 = reinterpret_cast<float4 *>(d_out + static_cast<size_t>(r) * 512);
    if (alpha == 0.0f) {
        o4[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else if (alpha != 1.0f) {
        float4 v = o4[tid];
        o4[tid] = make_float4(v.x * alpha, v.y * alpha, v.z * alpha, v.w * alpha);
    }
}

__global__ void mla_flash_finalize_out_kernel(
    float * __restrict__ d_out, const float * __restrict__ l_buf, int M) {
    const int r   = blockIdx.x;
    const int tid = threadIdx.x; // 0..127 -> 512 floats via float4!
    const float l = l_buf[r];
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    auto * o4 = reinterpret_cast<float4 *>(d_out + static_cast<size_t>(r) * 512);
    float4 v = o4[tid];
    o4[tid] = make_float4(v.x * inv_l, v.y * inv_l, v.z * inv_l, v.w * inv_l);
}

} // namespace

void launch_batched_head_proj_q8_0(
    const DeviceTensor & W, const float * d_x, float * d_y,
    int T, cudaStream_t stream,
    cublasHandle_t cublas, void * d_scratch, size_t scratch_bytes) {
    const int K       = static_cast<int>(W.ne[0]);
    const int M       = static_cast<int>(W.ne[1]);
    const int n_heads = static_cast<int>(W.ne[2]);

    // Fast strided batched GEMM for prefill (T >= 8): FP16 Tensor Core (sm>=70) or FP32 SGEMM (<sm_70)!
    if (T >= 8 && W.is_soa && cublas != nullptr && d_scratch != nullptr) {
        const bool use_tc = device_has_tensor_cores(W.device);
        auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
        if (use_tc) {
            const size_t w_bytes = align256(static_cast<size_t>(n_heads) * M * K * sizeof(__half));
            const size_t x_bytes = align256(static_cast<size_t>(n_heads) * T * K * sizeof(__half));
            const size_t y_bytes = align256(static_cast<size_t>(n_heads) * T * M * sizeof(float));
            if (w_bytes + x_bytes + y_bytes <= scratch_bytes) {
                auto * base  = static_cast<uint8_t *>(d_scratch);
                auto * d_w16 = reinterpret_cast<__half *>(base);
                auto * d_x16 = reinterpret_cast<__half *>(base + w_bytes);
                auto * d_yh  = reinterpret_cast<float  *>(base + w_bytes + x_bytes);

                const int64_t nblk = (static_cast<int64_t>(n_heads) * M * K) / 32;
                constexpr int wpb = 8;
                dequant_q8_0_soa_mla_kernel<__half><<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
                    static_cast<const int8_t *>(W.ptr), static_cast<const uint16_t *>(W.scales_ptr), d_w16, nblk);

                const int tot_x = T * n_heads * K;
                transpose_thd_to_htd_kernel<__half><<<(tot_x + 255) / 256, 256, 0, stream>>>(d_x, d_x16, T, n_heads, K);

                const float alpha = 1.0f, beta = 0.0f;
                cublasSetStream(cublas, stream);
                cublasGemmStridedBatchedEx(
                    cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    M, T, K,
                    &alpha,
                    d_w16, CUDA_R_16F, K, static_cast<long long>(M) * K,
                    d_x16, CUDA_R_16F, K, static_cast<long long>(T) * K,
                    &beta,
                    d_yh,  CUDA_R_32F, M, static_cast<long long>(T) * M,
                    n_heads,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                const int tot_y = T * n_heads * M;
                transpose_htd_to_thd_f32_kernel<<<(tot_y + 255) / 256, 256, 0, stream>>>(d_yh, d_y, T, n_heads, M);
                return;
            }
        } else {
            const size_t w_bytes = align256(static_cast<size_t>(n_heads) * M * K * sizeof(float));
            const size_t x_bytes = align256(static_cast<size_t>(n_heads) * T * K * sizeof(float));
            const size_t y_bytes = align256(static_cast<size_t>(n_heads) * T * M * sizeof(float));
            if (w_bytes + x_bytes + y_bytes <= scratch_bytes) {
                auto * base  = static_cast<uint8_t *>(d_scratch);
                auto * d_w32 = reinterpret_cast<float *>(base);
                auto * d_x32 = reinterpret_cast<float *>(base + w_bytes);
                auto * d_yh  = reinterpret_cast<float *>(base + w_bytes + x_bytes);

                const int64_t nblk = (static_cast<int64_t>(n_heads) * M * K) / 32;
                constexpr int wpb = 8;
                dequant_q8_0_soa_mla_kernel<float><<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
                    static_cast<const int8_t *>(W.ptr), static_cast<const uint16_t *>(W.scales_ptr), d_w32, nblk);

                const int tot_x = T * n_heads * K;
                transpose_thd_to_htd_kernel<float><<<(tot_x + 255) / 256, 256, 0, stream>>>(d_x, d_x32, T, n_heads, K);

                const float alpha = 1.0f, beta = 0.0f;
                cublasSetStream(cublas, stream);
                cublasSgemmStridedBatched(
                    cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    M, T, K,
                    &alpha,
                    d_w32, K, static_cast<long long>(M) * K,
                    d_x32, K, static_cast<long long>(T) * K,
                    &beta,
                    d_yh,  M, static_cast<long long>(T) * M,
                    n_heads);

                const int tot_y = T * n_heads * M;
                transpose_htd_to_thd_f32_kernel<<<(tot_y + 255) / 256, 256, 0, stream>>>(d_yh, d_y, T, n_heads, M);
                return;
            }
        }
    }

    constexpr int wpb = 8;
    dim3 grid((M + wpb - 1) / wpb, n_heads, T);
    batched_head_proj_q8_0_soa_kernel<<<grid, dim3(32, wpb), 0, stream>>>(
        static_cast<const int4 *>(W.ptr), static_cast<const uint16_t *>(W.scales_ptr),
        d_x, d_y, M, K, n_heads);
}

void launch_mla_kv_write(
    const float * d_kv, void * d_kv_cache, bool kv_is_q8_0,
    const int32_t * d_pos, int T, int kv_dim, cudaStream_t stream) {
    if (kv_is_q8_0) {
        mla_kv_write_q8_0_kernel<<<dim3(2, T), dim3(32, 8), 0, stream>>>(
            d_kv, static_cast<BlockQ8_0 *>(d_kv_cache), d_pos, T);
    } else {
        mla_kv_write_f16_kernel<<<dim3((kv_dim + 255) / 256, T), 256, 0, stream>>>(
            d_kv, static_cast<__half *>(d_kv_cache), d_pos, T, kv_dim);
    }
}

void launch_mla_attn_d512(
    const float * d_q_abs, const void * d_kv_cache, bool kv_is_q8_0,
    const int32_t * d_pos, float * d_out, void * d_scratch,
    int T, int n_heads, float scale, cudaStream_t stream,
    cublasHandle_t cublas, size_t scratch_bytes, int kv_len_max) {

    // Fast Tiled FlashMLA Prefill (T >= 8): FP16 Tensor Core (sm>=70) or FP32 SGEMM (<sm_70)!
    if (T >= 8 && cublas != nullptr && kv_len_max > 0) {
        const bool use_tc = device_has_tensor_cores(-1);
        const int M = T * n_heads;
        auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };

        if (use_tc) {
            constexpr int K_TILE = 512;
            const size_t q16_b = align256(static_cast<size_t>(M) * 512 * sizeof(__half));
            const size_t k16_b = align256(static_cast<size_t>(K_TILE) * 512 * sizeof(__half));
            const size_t s32_b = align256(static_cast<size_t>(M) * K_TILE * sizeof(float));
            const size_t p16_b = align256(static_cast<size_t>(M) * K_TILE * sizeof(__half));
            const size_t ml_b  = align256(static_cast<size_t>(M) * sizeof(float));
            const size_t need  = q16_b + k16_b + s32_b + p16_b + 2 * ml_b;

            if (need <= scratch_bytes) {
                auto * base = static_cast<uint8_t *>(d_scratch);
                size_t off = 0;
                auto * d_q16 = reinterpret_cast<__half *>(base + off); off += q16_b;
                auto * d_k16 = reinterpret_cast<__half *>(base + off); off += k16_b;
                auto * d_s32 = reinterpret_cast<float  *>(base + off); off += s32_b;
                auto * d_p16 = reinterpret_cast<__half *>(base + off); off += p16_b;
                auto * d_m   = reinterpret_cast<float  *>(base + off); off += ml_b;
                auto * d_l   = reinterpret_cast<float  *>(base + off);

                const int64_t nq = static_cast<int64_t>(M) * 512;
                mla_scale_cast_kernel<__half><<<(nq + 255) / 256, 256, 0, stream>>>(d_q_abs, d_q16, nq, scale);
                mla_init_flash_state_kernel<<<(M + 255) / 256, 256, 0, stream>>>(d_m, d_l, M);

                const float one = 1.0f, zero = 0.0f;
                cublasSetStream(cublas, stream);

                for (int k0 = 0; k0 < kv_len_max; k0 += K_TILE) {
                    const int kc = (k0 + K_TILE < kv_len_max) ? K_TILE : (kv_len_max - k0);
                    if (kv_is_q8_0) {
                        mla_dequant_kv_tile_kernel<true, __half><<<dim3(2, kc), dim3(32, 8), 0, stream>>>(d_kv_cache, d_k16, k0, kc);
                    } else {
                        mla_dequant_kv_tile_kernel<false, __half><<<dim3(2, kc), 256, 0, stream>>>(d_kv_cache, d_k16, k0, kc);
                    }

                    cublasGemmEx(
                        cublas,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        kc, M, 512,
                        &one,
                        d_k16, CUDA_R_16F, 512,
                        d_q16, CUDA_R_16F, 512,
                        &zero,
                        d_s32, CUDA_R_32F, kc,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                    mla_flash_online_softmax_rescale_kernel<__half><<<M, 128, 0, stream>>>(
                        d_s32, d_p16, d_pos, d_m, d_l, d_out, n_heads, k0, kc);

                    cublasGemmEx(
                        cublas,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        512, M, kc,
                        &one,
                        d_k16, CUDA_R_16F, 512,
                        d_p16, CUDA_R_16F, kc,
                        &one,
                        d_out, CUDA_R_32F, 512,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                }

                mla_flash_finalize_out_kernel<<<M, 128, 0, stream>>>(d_out, d_l, M);
                return;
            }
        } else {
            constexpr int K_TILE = 256;
            const size_t q32_b = align256(static_cast<size_t>(M) * 512 * sizeof(float));
            const size_t k32_b = align256(static_cast<size_t>(K_TILE) * 512 * sizeof(float));
            const size_t s32_b = align256(static_cast<size_t>(M) * K_TILE * sizeof(float));
            const size_t p32_b = s32_b;
            const size_t ml_b  = align256(static_cast<size_t>(M) * sizeof(float));
            const size_t need  = q32_b + k32_b + s32_b + p32_b + 2 * ml_b;

            if (need <= scratch_bytes) {
                auto * base = static_cast<uint8_t *>(d_scratch);
                size_t off = 0;
                auto * d_q32 = reinterpret_cast<float *>(base + off); off += q32_b;
                auto * d_k32 = reinterpret_cast<float *>(base + off); off += k32_b;
                auto * d_s32 = reinterpret_cast<float *>(base + off); off += s32_b;
                auto * d_p32 = reinterpret_cast<float *>(base + off); off += p32_b;
                auto * d_m   = reinterpret_cast<float *>(base + off); off += ml_b;
                auto * d_l   = reinterpret_cast<float *>(base + off);

                const int64_t nq = static_cast<int64_t>(M) * 512;
                mla_scale_cast_kernel<float><<<(nq + 255) / 256, 256, 0, stream>>>(d_q_abs, d_q32, nq, scale);
                mla_init_flash_state_kernel<<<(M + 255) / 256, 256, 0, stream>>>(d_m, d_l, M);

                const float one = 1.0f, zero = 0.0f;
                cublasSetStream(cublas, stream);

                for (int k0 = 0; k0 < kv_len_max; k0 += K_TILE) {
                    const int kc = (k0 + K_TILE < kv_len_max) ? K_TILE : (kv_len_max - k0);
                    if (kv_is_q8_0) {
                        mla_dequant_kv_tile_kernel<true, float><<<dim3(2, kc), dim3(32, 8), 0, stream>>>(d_kv_cache, d_k32, k0, kc);
                    } else {
                        mla_dequant_kv_tile_kernel<false, float><<<dim3(2, kc), 256, 0, stream>>>(d_kv_cache, d_k32, k0, kc);
                    }

                    cublasSgemm(
                        cublas,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        kc, M, 512,
                        &one,
                        d_k32, 512,
                        d_q32, 512,
                        &zero,
                        d_s32, kc);

                    mla_flash_online_softmax_rescale_kernel<float><<<M, 128, 0, stream>>>(
                        d_s32, d_p32, d_pos, d_m, d_l, d_out, n_heads, k0, kc);

                    cublasSgemm(
                        cublas,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        512, M, kc,
                        &one,
                        d_k32, 512,
                        d_p32, kc,
                        &one,
                        d_out, 512);
                }

                mla_flash_finalize_out_kernel<<<M, 128, 0, stream>>>(d_out, d_l, M);
                return;
            }
        }
    }

    if (T == 2) {
        // Joint T=2 MTP Split-K Decode: 8 heads x 2 tokens (16 warps = 512 threads/CTA) share 1 KV load!
        constexpr int S = 32;
        constexpr int H_PER_BLK = 8;
        const size_t n_slots = static_cast<size_t>(2) * n_heads * S;
        float * pm = static_cast<float *>(d_scratch);
        float * pl = pm + n_slots;
        float * po = pl + n_slots;
        dim3 grid((n_heads + H_PER_BLK - 1) / H_PER_BLK, S);
        dim3 block(32, 16);
        if (kv_is_q8_0) {
            mla_mqa_d512_decode_shared_t2_kernel<true, S><<<grid, block, 0, stream>>>(d_q_abs, d_kv_cache, d_pos, pm, pl, po, n_heads, scale);
        } else {
            mla_mqa_d512_decode_shared_t2_kernel<false, S><<<grid, block, 0, stream>>>(d_q_abs, d_kv_cache, d_pos, pm, pl, po, n_heads, scale);
        }
        mla_mqa_d512_reduce_kernel<S><<<dim3(n_heads, 2), 32, 0, stream>>>(pm, pl, po, d_out, n_heads);
    } else if (T <= 4) {
        // Multi-Head-Shared Split-K Decode: 16 heads/CTA share 1 KV load (only 4 KV sweeps across 64 heads!)
        constexpr int S = 32;
        constexpr int H_PER_BLK = 16;
        const size_t n_slots = static_cast<size_t>(T) * n_heads * S;
        float * pm = static_cast<float *>(d_scratch);
        float * pl = pm + n_slots;
        float * po = pl + n_slots;
        dim3 grid((n_heads + H_PER_BLK - 1) / H_PER_BLK, S, T);
        dim3 block(32, H_PER_BLK);
        if (kv_is_q8_0) {
            mla_mqa_d512_decode_shared_kernel<true, S, H_PER_BLK><<<grid, block, 0, stream>>>(d_q_abs, d_kv_cache, d_pos, pm, pl, po, n_heads, scale);
        } else {
            mla_mqa_d512_decode_shared_kernel<false, S, H_PER_BLK><<<grid, block, 0, stream>>>(d_q_abs, d_kv_cache, d_pos, pm, pl, po, n_heads, scale);
        }
        mla_mqa_d512_reduce_kernel<S><<<dim3(n_heads, T), 32, 0, stream>>>(pm, pl, po, d_out, n_heads);
    } else {
        constexpr int S = 1;
        const size_t n_slots = static_cast<size_t>(T) * n_heads;
        float * pm = static_cast<float *>(d_scratch);
        float * pl = pm + n_slots;
        float * po = pl + n_slots;
        dim3 grid(n_heads, 1, T);
        if (kv_is_q8_0) {
            mla_mqa_d512_splitk_kernel<true, S><<<grid, 32, 0, stream>>>(d_q_abs, d_kv_cache, d_pos, pm, pl, po, n_heads, scale);
        } else {
            mla_mqa_d512_splitk_kernel<false, S><<<grid, 32, 0, stream>>>(d_q_abs, d_kv_cache, d_pos, pm, pl, po, n_heads, scale);
        }
        mla_mqa_d512_reduce_kernel<S><<<dim3(n_heads, T), 32, 0, stream>>>(pm, pl, po, d_out, n_heads);
    }
}

// ============================================================================
// RoPE-576 Absorbed MLA Kernels (BailingMoE3 / Ling-3.0 / Ling-3.2-Air)
// K = [kv_norm(512), k_pe_rot(64)] (576 dims), V = kv_norm(512) (first 512 dims)
// ============================================================================
namespace {

// 64-byte-aligned SoA-per-position Q8_0 MLA KV entry (640 B/pos = 10 cache lines):
// qs[576] = 36 contiguous int4 (100% coalesced LDG.128!), d[18] = FP16 block scales!
struct alignas(64) MlaKv576Q8Pos {
    int8_t   qs[576];
    uint16_t d[18];
    uint16_t pad[14];
};
static_assert(sizeof(MlaKv576Q8Pos) == 640, "MlaKv576Q8Pos must be 640 bytes");

__global__ void mla_rope_split_q_kernel(
    const float * __restrict__ q_all,   // [T, n_heads, qk_nope + rope_dim]
    const int32_t * __restrict__ pos,   // [T]
    float * __restrict__ q_nope,        // [T, n_heads, qk_nope]
    float * __restrict__ q_pe,          // [T, n_heads, rope_dim]
    int n_heads, int qk_nope, int rope_dim, float rope_freq_base) {

    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    const int qk_tot = qk_nope + rope_dim;
    const float * src = q_all + (static_cast<size_t>(t) * n_heads + h) * qk_tot;
    float * dst_nope  = q_nope + (static_cast<size_t>(t) * n_heads + h) * qk_nope;
    float * dst_pe    = q_pe   + (static_cast<size_t>(t) * n_heads + h) * rope_dim;

    for (int i = d; i < qk_nope; i += blockDim.x) {
        dst_nope[i] = src[i];
    }
    const int half_r = rope_dim >> 1;
    if (d < half_r) {
        const float p = static_cast<float>(pos[t]);
        const float freq = powf(rope_freq_base, -static_cast<float>(2 * d) / static_cast<float>(rope_dim));
        const float theta = p * freq;
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = src[qk_nope + 2 * d + 0];
        const float x1 = src[qk_nope + 2 * d + 1];
        dst_pe[2 * d + 0] = x0 * c - x1 * s;
        dst_pe[2 * d + 1] = x0 * s + x1 * c;
    }
}

template <bool KV_Q8_0>
__global__ void mla_kv_norm_rope_write_576_kernel(
    const float * __restrict__ kv_all,  // [T, 576]
    const float * __restrict__ norm_w,  // [512]
    const int32_t * __restrict__ pos,   // [T]
    void * __restrict__ kv_cache,       // [max_ctx, 576] (18 BlockQ8_0 or 576 __half)
    float rms_eps, float rope_freq_base) {

    __shared__ float s_buf[576];
    __shared__ float s_warp[8];

    const int t = blockIdx.x;
    const int tid = threadIdx.x; // 0..255 (8 warps)
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float * src = kv_all + static_cast<size_t>(t) * 576;

    // 1. RMSNorm first 512 dims
    const float v0 = src[tid];
    const float v1 = src[tid + 256];
    float ss = warp_sum(v0 * v0 + v1 * v1);
    if (lane == 0) s_warp[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? s_warp[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) s_warp[0] = rsqrtf(tot * (1.0f / 512.0f) + rms_eps);
    }
    __syncthreads();
    const float inv = s_warp[0];
    s_buf[tid]       = v0 * inv * norm_w[tid];
    s_buf[tid + 256] = v1 * inv * norm_w[tid + 256];

    // 2. Normal-pair RoPE on last 64 dims
    if (tid < 32) {
        const float p = static_cast<float>(pos[t]);
        const float freq = powf(rope_freq_base, -static_cast<float>(2 * tid) / 64.0f);
        const float theta = p * freq;
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = src[512 + 2 * tid + 0];
        const float x1 = src[512 + 2 * tid + 1];
        s_buf[512 + 2 * tid + 0] = x0 * c - x1 * s;
        s_buf[512 + 2 * tid + 1] = x0 * s + x1 * c;
    }
    __syncthreads();

    const int p = pos[t];
    if constexpr (KV_Q8_0) {
        // 18 Q8_0 blocks (576 / 32 = 18) stored in 64B-aligned MlaKv576Q8Pos!
        MlaKv576Q8Pos & dst = static_cast<MlaKv576Q8Pos *>(kv_cache)[p];
        for (int b = warp; b < 18; b += 8) {
            const float v = s_buf[b * 32 + lane];
            const float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(v)), 0);
            const float d = amax / 127.0f;
            const float inv_d = (d > 0.0f) ? (127.0f / amax) : 0.0f;
            if (lane == 0) {
                __half dh = __float2half_rn(d);
                dst.d[b] = *reinterpret_cast<uint16_t *>(&dh);
            }
            dst.qs[b * 32 + lane] = static_cast<int8_t>(__float2int_rn(v * inv_d));
        }
    } else {
        __half * dst = static_cast<__half *>(kv_cache) + static_cast<size_t>(p) * 576;
        for (int i = tid; i < 576; i += 256) {
            dst[i] = __float2half_rn(s_buf[i]);
        }
    }
}

__global__ void mla_concat_q576_kernel(
    const float * __restrict__ q_abs,   // [T, n_heads, 512]
    const float * __restrict__ q_pe,    // [T, n_heads, 64]
    float * __restrict__ q576,          // [T, n_heads, 576]
    int n_heads) {

    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int i = threadIdx.x; // 0..255
    const float * a = q_abs + (static_cast<size_t>(t) * n_heads + h) * 512;
    const float * p = q_pe  + (static_cast<size_t>(t) * n_heads + h) * 64;
    float * dst     = q576  + (static_cast<size_t>(t) * n_heads + h) * 576;
    dst[i]       = a[i];
    dst[i + 256] = a[i + 256];
    if (i < 64) dst[512 + i] = p[i];
}

// Warp-parallel FP16 batched per-head GEMV (CUDA-Graph safe for T<=4, zero cuBLAS!)
// W: [n_heads, M, K] FP16, x: [T, n_heads, K] FP32, y: [T, n_heads, M] FP32
__global__ void batched_head_proj_f16_gemv_kernel(
    const __half * __restrict__ W,
    const float *  __restrict__ x,
    float *        __restrict__ y,
    int M, int K, int n_heads) {

    __shared__ float s_x[1024]; // K <= 1024 (128 or 512)
    const int h    = blockIdx.y;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int ntid = blockDim.x * blockDim.y;

    const float * xp = x + (static_cast<size_t>(t) * n_heads + h) * K;
    for (int i = tid; i < K; i += ntid) s_x[i] = xp[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + warp;
    if (row >= M) return;

    const __half * wr = W + (static_cast<size_t>(h) * M + row) * K;
    float acc = 0.0f;
    const int k2 = K >> 1;
    const __half2 * wr2 = reinterpret_cast<const __half2 *>(wr);
    for (int i = lane; i < k2; i += 32) {
        float2 wv = __half22float2(__ldg(&wr2[i]));
        acc += wv.x * s_x[2 * i] + wv.y * s_x[2 * i + 1];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(static_cast<size_t>(t) * n_heads + h) * M + row] = acc;
}

template <bool KV_Q8_0, typename OutT>
__global__ void mla_dequant_kv576_tile_kernel(
    const void * __restrict__ kv_cache,
    OutT *       __restrict__ out_tile, // [kc, 576]
    int k0, int kc) {

    const int row = blockIdx.x; // 0..kc-1
    if (row >= kc) return;
    const int p = k0 + row;
    const int tid = threadIdx.x; // 0..255
    OutT * dst = out_tile + static_cast<size_t>(row) * 576;

    if constexpr (KV_Q8_0) {
        const int lane = tid & 31;
        const int warp = tid >> 5;
        const MlaKv576Q8Pos & kv = static_cast<const MlaKv576Q8Pos *>(kv_cache)[p];
        for (int b = warp; b < 18; b += 8) {
            __half dh;
            *reinterpret_cast<uint16_t *>(&dh) = __ldg(&kv.d[b]);
            const float sc = __half2float(dh);
            if constexpr (std::is_same_v<OutT, __half>) {
                if (lane < 16) {
                    const uint16_t q2 = __ldg(&reinterpret_cast<const uint16_t *>(kv.qs + b * 32)[lane]);
                    const float v0 = sc * static_cast<float>(static_cast<int8_t>(q2 & 0xFFu));
                    const float v1 = sc * static_cast<float>(static_cast<int8_t>(q2 >> 8));
                    reinterpret_cast<__half2 *>(dst + b * 32)[lane] = __float22half2_rn(make_float2(v0, v1));
                }
            } else {
                dst[b * 32 + lane] = sc * static_cast<float>(__ldg(&kv.qs[b * 32 + lane]));
            }
        }
    } else {
        const __half * src = static_cast<const __half *>(kv_cache) + static_cast<size_t>(p) * 576;
        for (int i = tid; i < 576; i += 256) {
            if constexpr (std::is_same_v<OutT, __half>) dst[i] = src[i];
            else dst[i] = __half2float(src[i]);
        }
    }
}

// Ultra-fast Coalesced LDG.128 Q8_0 dp4a + Tile-Softmax Split-K RoPE-576 MLA Decode!
// - Loads P_TILE=8 MlaKv576Q8Pos structs via 100% coalesced int4 LDG.128 (zero bank conflicts!)
// - Rescales ov[0..15] ONCE per 8-token tile (8x fewer expf & alpha rescales + tile sparsity skip!)
template <int SPLITS, int H_PER_BLK>
__global__ __launch_bounds__(512) void mla_mqa_d576_q8_dp4a_decode_kernel(
    const float *   __restrict__ q576,     // [T, n_heads, 576]
    const void *    __restrict__ kv_cache, // [max_ctx] MlaKv576Q8Pos
    const int32_t * __restrict__ pos,      // [T]
    float *         __restrict__ part_m,   // [T, n_heads, SPLITS]
    float *         __restrict__ part_l,   // [T, n_heads, SPLITS]
    float *         __restrict__ part_o,   // [T, n_heads, SPLITS, 512]
    int n_heads, float scale) {

    constexpr int P_TILE = 8;
    __shared__ int32_t s_kv_qs[P_TILE][18][8]; // 4,608 B = 288 int4
    __shared__ float   s_kv_d[P_TILE][18];     //   576 B -> 5.2 KB total!

    const int h0   = blockIdx.x * H_PER_BLK;
    const int s    = blockIdx.y;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int nthr = H_PER_BLK * 32;
    const int h    = h0 + warp;
    const bool valid_h = (h < n_heads);

    const int kv_len = pos[t] + 1;
    const int chunk  = (kv_len + SPLITS - 1) / SPLITS;
    const int p0     = s * chunk;
    const int p1     = (p0 + chunk < kv_len) ? (p0 + chunk) : kv_len;

    const int grp = lane >> 3; // 0..3
    const int sub = lane & 7;  // 0..7

    int32_t q_i32[5] = {0, 0, 0, 0, 0};
    float   q_d[5]   = {0, 0, 0, 0, 0};
    if (valid_h) {
        const float * qp = q576 + (static_cast<size_t>(t) * n_heads + h) * 576;
        #pragma unroll
        for (int k = 0; k < 5; ++k) {
            const int b = grp + 4 * k;
            float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
            if (b < 18) {
                const float4 qv = __ldg(reinterpret_cast<const float4 *>(qp + b * 32 + sub * 4));
                x0 = qv.x * scale; x1 = qv.y * scale; x2 = qv.z * scale; x3 = qv.w * scale;
            }
            float am = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
            if (b < 18) {
                const float d = am * (1.0f / 127.0f);
                const float inv_d = (d > 0.0f) ? (127.0f / am) : 0.0f;
                q_d[k] = d;
                const uint32_t b0 = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(x0 * inv_d)));
                const uint32_t b1 = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(x1 * inv_d)));
                const uint32_t b2 = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(x2 * inv_d)));
                const uint32_t b3 = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(x3 * inv_d)));
                q_i32[k] = static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
            }
        }
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float ov[16] = {0};

    const MlaKv576Q8Pos * kv_base = static_cast<const MlaKv576Q8Pos *>(kv_cache);
    int4 * s_qs_i4 = reinterpret_cast<int4 *>(s_kv_qs);

    for (int p = p0; p < p1; p += P_TILE) {
        const int n_p    = (p + P_TILE < p1) ? P_TILE : (p1 - p);
        const int tot_i4 = n_p * 36;
        for (int idx = tid; idx < tot_i4; idx += nthr) {
            const int pi = (idx * 1821) >> 16; // exact idx / 36 for 0 <= idx < 288
            const int c  = idx - pi * 36;
            s_qs_i4[idx] = __ldg(&reinterpret_cast<const int4 *>(kv_base[p + pi].qs)[c]);
        }
        const int tot_d = n_p * 18;
        for (int idx = tid; idx < tot_d; idx += nthr) {
            const int pi = (idx * 3641) >> 16;
            const int b  = idx - pi * 18;
            __half dh;
            *reinterpret_cast<uint16_t *>(&dh) = __ldg(&kv_base[p + pi].d[b]);
            s_kv_d[pi][b] = __half2float(dh);
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
                    for (int k = 0; k < 5; ++k) {
                        const int b = grp + 4 * k;
                        if (b < 18) {
                            const int dp = dp4a_s32(q_i32[k], s_kv_qs[pi][b][sub], 0);
                            dot += (q_d[k] * s_kv_d[pi][b]) * static_cast<float>(dp);
                        }
                    }
                    dot = warp_sum(dot);
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
                for (int i = 0; i < 16; ++i) ov[i] *= alpha;

                #pragma unroll
                for (int pi = 0; pi < P_TILE; ++pi) {
                    const float sc = scores[pi];
                    if (sc > m_new - 18.0f) {
                        const float beta = expf(sc - m_new);
                        l_i += beta;
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const int b = grp + 4 * k;
                            const int32_t k4 = s_kv_qs[pi][b][sub];
                            const float c = beta * s_kv_d[pi][b];
                            ov[k * 4 + 0] += c * static_cast<float>(static_cast<int8_t>(k4));
                            ov[k * 4 + 1] += c * static_cast<float>(static_cast<int8_t>(k4 >> 8));
                            ov[k * 4 + 2] += c * static_cast<float>(static_cast<int8_t>(k4 >> 16));
                            ov[k * 4 + 3] += c * static_cast<float>(static_cast<int8_t>(k4 >> 24));
                        }
                    }
                }
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (valid_h) {
        const size_t slot = (static_cast<size_t>(t) * n_heads + h) * SPLITS + s;
        if (lane == 0) {
            part_m[slot] = m_i;
            part_l[slot] = l_i;
        }
        float * op = part_o + slot * 512;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int b = grp + 4 * k;
            *reinterpret_cast<float4 *>(op + b * 32 + sub * 4) =
                make_float4(ov[k * 4 + 0], ov[k * 4 + 1], ov[k * 4 + 2], ov[k * 4 + 3]);
        }
    }
}

// Joint T=2 MTP Coalesced LDG.128 Q8_0 dp4a + Tile-Softmax Split-K RoPE-576 MLA Decode!
template <int SPLITS>
__global__ __launch_bounds__(512) void mla_mqa_d576_q8_dp4a_decode_t2_kernel(
    const float *   __restrict__ q576,     // [2, n_heads, 576]
    const void *    __restrict__ kv_cache, // [max_ctx] MlaKv576Q8Pos
    const int32_t * __restrict__ pos,      // [2]
    float *         __restrict__ part_m,   // [2, n_heads, SPLITS]
    float *         __restrict__ part_l,   // [2, n_heads, SPLITS]
    float *         __restrict__ part_o,   // [2, n_heads, SPLITS, 512]
    int n_heads, float scale) {

    constexpr int H_PER_BLK = 8;
    constexpr int P_TILE    = 8;
    __shared__ int32_t s_kv_qs[P_TILE][18][8];
    __shared__ float   s_kv_d[P_TILE][18];

    const int h0   = blockIdx.x * H_PER_BLK;
    const int s    = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int t    = warp >> 3;
    const int h    = h0 + (warp & 7);
    const bool valid_h = (h < n_heads);

    const int kv_len_0 = pos[0] + 1;
    const int kv_len_1 = pos[1] + 1;
    const int kv_max   = (kv_len_1 > kv_len_0) ? kv_len_1 : kv_len_0;
    const int my_kv    = (t == 0) ? kv_len_0 : kv_len_1;

    const int chunk = (kv_max + SPLITS - 1) / SPLITS;
    const int p0    = s * chunk;
    const int p1    = (p0 + chunk < kv_max) ? (p0 + chunk) : kv_max;

    const int grp = lane >> 3;
    const int sub = lane & 7;

    int32_t q_i32[5] = {0, 0, 0, 0, 0};
    float   q_d[5]   = {0, 0, 0, 0, 0};
    if (valid_h) {
        const float * qp = q576 + (static_cast<size_t>(t) * n_heads + h) * 576;
        #pragma unroll
        for (int k = 0; k < 5; ++k) {
            const int b = grp + 4 * k;
            float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
            if (b < 18) {
                const float4 qv = __ldg(reinterpret_cast<const float4 *>(qp + b * 32 + sub * 4));
                x0 = qv.x * scale; x1 = qv.y * scale; x2 = qv.z * scale; x3 = qv.w * scale;
            }
            float am = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
            if (b < 18) {
                const float d = am * (1.0f / 127.0f);
                const float inv_d = (d > 0.0f) ? (127.0f / am) : 0.0f;
                q_d[k] = d;
                const uint32_t b0 = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(x0 * inv_d)));
                const uint32_t b1 = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(x1 * inv_d)));
                const uint32_t b2 = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(x2 * inv_d)));
                const uint32_t b3 = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(x3 * inv_d)));
                q_i32[k] = static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
            }
        }
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float ov[16] = {0};

    const MlaKv576Q8Pos * kv_base = static_cast<const MlaKv576Q8Pos *>(kv_cache);
    int4 * s_qs_i4 = reinterpret_cast<int4 *>(s_kv_qs);

    for (int p = p0; p < p1; p += P_TILE) {
        const int n_p    = (p + P_TILE < p1) ? P_TILE : (p1 - p);
        const int tot_i4 = n_p * 36;
        if (tid < tot_i4) {
            const int pi = (tid * 1821) >> 16;
            const int c  = tid - pi * 36;
            s_qs_i4[tid] = __ldg(&reinterpret_cast<const int4 *>(kv_base[p + pi].qs)[c]);
        }
        const int tot_d = n_p * 18;
        if (tid < tot_d) {
            const int pi = (tid * 3641) >> 16;
            const int b  = tid - pi * 18;
            __half dh;
            *reinterpret_cast<uint16_t *>(&dh) = __ldg(&kv_base[p + pi].d[b]);
            s_kv_d[pi][b] = __half2float(dh);
        }
        __syncthreads();

        if (valid_h) {
            float scores[P_TILE];
            float m_tile = -FLT_MAX;
            #pragma unroll
            for (int pi = 0; pi < P_TILE; ++pi) {
                if (pi < n_p && (p + pi) < my_kv) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 5; ++k) {
                        const int b = grp + 4 * k;
                        if (b < 18) {
                            const int dp = dp4a_s32(q_i32[k], s_kv_qs[pi][b][sub], 0);
                            dot += (q_d[k] * s_kv_d[pi][b]) * static_cast<float>(dp);
                        }
                    }
                    dot = warp_sum(dot);
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
                for (int i = 0; i < 16; ++i) ov[i] *= alpha;

                #pragma unroll
                for (int pi = 0; pi < P_TILE; ++pi) {
                    const float sc = scores[pi];
                    if (sc > m_new - 18.0f) {
                        const float beta = expf(sc - m_new);
                        l_i += beta;
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const int b = grp + 4 * k;
                            const int32_t k4 = s_kv_qs[pi][b][sub];
                            const float c = beta * s_kv_d[pi][b];
                            ov[k * 4 + 0] += c * static_cast<float>(static_cast<int8_t>(k4));
                            ov[k * 4 + 1] += c * static_cast<float>(static_cast<int8_t>(k4 >> 8));
                            ov[k * 4 + 2] += c * static_cast<float>(static_cast<int8_t>(k4 >> 16));
                            ov[k * 4 + 3] += c * static_cast<float>(static_cast<int8_t>(k4 >> 24));
                        }
                    }
                }
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (valid_h) {
        const size_t slot = (static_cast<size_t>(t) * n_heads + h) * SPLITS + s;
        if (lane == 0) {
            part_m[slot] = m_i;
            part_l[slot] = l_i;
        }
        float * op = part_o + slot * 512;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int b = grp + 4 * k;
            *reinterpret_cast<float4 *>(op + b * 32 + sub * 4) =
                make_float4(ov[k * 4 + 0], ov[k * 4 + 1], ov[k * 4 + 2], ov[k * 4 + 3]);
        }
    }
}

// Fast 4-warp (128 thr x float4 = 512 dims) coalesced Split-K MLA reduction!
template <int SPLITS>
__global__ void mla_mqa_d512_reduce_fast_kernel(
    const float * __restrict__ part_m,
    const float * __restrict__ part_l,
    const float * __restrict__ part_o,
    float *       __restrict__ out,
    int n_heads) {

    __shared__ float s_w[SPLITS];
    __shared__ float s_inv_l;

    const int h   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x; // 0..127 -> 128 float4 = 512 floats!
    const size_t base = (static_cast<size_t>(t) * n_heads + h) * SPLITS;

    if (tid == 0) {
        float m_max = -FLT_MAX;
        for (int s = 0; s < SPLITS; ++s) {
            if (part_l[base + s] > 0.0f) m_max = fmaxf(m_max, part_m[base + s]);
        }
        float l_tot = 0.0f;
        for (int s = 0; s < SPLITS; ++s) {
            const float ls = part_l[base + s];
            if (ls > 0.0f) {
                const float w = expf(part_m[base + s] - m_max);
                s_w[s] = w;
                l_tot += ls * w;
            } else {
                s_w[s] = 0.0f;
            }
        }
        s_inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    }
    __syncthreads();

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < SPLITS; ++s) {
        const float w = s_w[s];
        if (w > 0.0f) {
            const float4 v = reinterpret_cast<const float4 *>(part_o + (base + s) * 512)[tid];
            acc.x += v.x * w;
            acc.y += v.y * w;
            acc.z += v.z * w;
            acc.w += v.w * w;
        }
    }
    const float inv_l = s_inv_l;
    reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * n_heads + h) * 512)[tid] =
        make_float4(acc.x * inv_l, acc.y * inv_l, acc.z * inv_l, acc.w * inv_l);
}

// Zero-bank-conflict stride-32 Multi-Head-Shared Split-K RoPE-576 MLA Decode (P_TILE=8, H_PER_BLK=16)
template <bool KV_Q8_0, int SPLITS, int H_PER_BLK>
__global__ __launch_bounds__(512) void mla_mqa_d576_decode_shared_kernel(
    const float *   __restrict__ q576,     // [T, n_heads, 576]
    const void *    __restrict__ kv_cache, // [max_ctx, 576]
    const int32_t * __restrict__ pos,      // [T]
    float *         __restrict__ part_m,   // [T, n_heads, SPLITS]
    float *         __restrict__ part_l,   // [T, n_heads, SPLITS]
    float *         __restrict__ part_o,   // [T, n_heads, SPLITS, 512]
    int n_heads, float scale) {

    constexpr int P_TILE = 8; // 8 * 576 * 4 B = 18.4 KB smem; 8 * 18 = 144 Q8_0 blocks / 16 warps = 9 exact!
    __shared__ float s_kv[P_TILE][576];

    const int h0   = blockIdx.x * H_PER_BLK;
    const int s    = blockIdx.y;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..H_PER_BLK-1 (16 warps = 512 threads)
    const int tid  = warp * 32 + lane;
    const int h    = h0 + warp;
    const bool valid_h = (h < n_heads);

    const int kv_len = pos[t] + 1;
    const int chunk  = (kv_len + SPLITS - 1) / SPLITS;
    const int p0     = s * chunk;
    const int p1     = (p0 + chunk < kv_len) ? (p0 + chunk) : kv_len;

    // Stride-32 coalesced Q load: lane owns dims {i*32 + lane} -> ZERO shared-mem bank conflicts!
    float qv[16] = {0};
    float qp0 = 0.0f, qp1 = 0.0f;
    if (valid_h) {
        const float * qp = q576 + (static_cast<size_t>(t) * n_heads + h) * 576;
        #pragma unroll
        for (int i = 0; i < 16; ++i) qv[i] = __ldg(&qp[i * 32 + lane]) * scale;
        qp0 = __ldg(&qp[512 + lane]) * scale;
        qp1 = __ldg(&qp[544 + lane]) * scale;
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float ov[16] = {0};

    for (int p = p0; p < p1; p += P_TILE) {
        const int n_p = (p + P_TILE < p1) ? P_TILE : (p1 - p);
        if constexpr (KV_Q8_0) {
            const int tot_blk = n_p * 18;
            const BlockQ8_0 * kb_base = static_cast<const BlockQ8_0 *>(kv_cache) + static_cast<size_t>(p) * 18;
            for (int idx = warp; idx < tot_blk; idx += H_PER_BLK) {
                const int pi = (idx * 3641) >> 16; // exact idx / 18 for 0 <= idx < 180
                const int b  = idx - pi * 18;
                const BlockQ8_0 & kb = kb_base[idx];
                __half dh;
                *reinterpret_cast<uint16_t *>(&dh) = __ldg(&kb.d);
                s_kv[pi][b * 32 + lane] = __half2float(dh) * static_cast<float>(__ldg(&kb.qs[lane]));
            }
        } else {
            const int tot_u2 = n_p * 144;
            const uint2 * kc_base = static_cast<const uint2 *>(kv_cache) + static_cast<size_t>(p) * 144;
            for (int idx = tid; idx < tot_u2; idx += H_PER_BLK * 32) {
                const int pi = idx / 144;
                const int c4 = idx - pi * 144;
                uint2 pk = __ldg(&kc_base[idx]);
                float2 f0 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.x));
                float2 f1 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.y));
                reinterpret_cast<float4 *>(s_kv[pi])[c4] = make_float4(f0.x, f0.y, f1.x, f1.y);
            }
        }
        __syncthreads();

        if (valid_h) {
            #pragma unroll
            for (int pi = 0; pi < P_TILE; ++pi) {
                if (pi >= n_p) break;
                const float * sk = s_kv[pi];
                float dot = qp0 * sk[512 + lane] + qp1 * sk[544 + lane];
                #pragma unroll
                for (int i = 0; i < 16; ++i) dot += qv[i] * sk[i * 32 + lane];
                dot = warp_sum(dot);
                const float score = __shfl_sync(0xffffffffu, dot, 0);

                const float m_new = fmaxf(m_i, score);
                const float alpha = expf(m_i - m_new);
                const float beta  = expf(score - m_new);
                l_i = l_i * alpha + beta;
                #pragma unroll
                for (int i = 0; i < 16; ++i) ov[i] = ov[i] * alpha + beta * sk[i * 32 + lane];
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (valid_h) {
        const size_t slot = (static_cast<size_t>(t) * n_heads + h) * SPLITS + s;
        if (lane == 0) {
            part_m[slot] = m_i;
            part_l[slot] = l_i;
        }
        float * op = part_o + slot * 512;
        #pragma unroll
        for (int i = 0; i < 16; ++i) op[i * 32 + lane] = ov[i];
    }
}

// Zero-bank-conflict stride-32 Joint T=2 MTP Split-K RoPE-576 MLA Decode (P_TILE=8, 8 heads x 2 tokens = 16 warps!)
template <bool KV_Q8_0, int SPLITS>
__global__ __launch_bounds__(512) void mla_mqa_d576_decode_shared_t2_kernel(
    const float *   __restrict__ q576,     // [2, n_heads, 576]
    const void *    __restrict__ kv_cache, // [max_ctx, 576] (Q8_0 or F16)
    const int32_t * __restrict__ pos,      // [2]
    float *         __restrict__ part_m,   // [2, n_heads, SPLITS]
    float *         __restrict__ part_l,   // [2, n_heads, SPLITS]
    float *         __restrict__ part_o,   // [2, n_heads, SPLITS, 512]
    int n_heads, float scale) {

    constexpr int H_PER_BLK = 8;
    constexpr int P_TILE    = 8;
    __shared__ float s_kv[P_TILE][576];

    const int h0   = blockIdx.x * H_PER_BLK;
    const int s    = blockIdx.y;
    const int lane = threadIdx.x;          // 0..31
    const int warp = threadIdx.y;          // 0..15 (t = warp/8, hi = warp%8)
    const int tid  = warp * 32 + lane;     // 0..511
    const int t    = warp >> 3;            // 0 or 1
    const int h    = h0 + (warp & 7);
    const bool valid_h = (h < n_heads);

    const int kv_len_0 = pos[0] + 1;
    const int kv_len_1 = pos[1] + 1;
    const int kv_max   = (kv_len_1 > kv_len_0) ? kv_len_1 : kv_len_0;
    const int my_kv    = (t == 0) ? kv_len_0 : kv_len_1;

    const int chunk = (kv_max + SPLITS - 1) / SPLITS;
    const int p0    = s * chunk;
    const int p1    = (p0 + chunk < kv_max) ? (p0 + chunk) : kv_max;

    float qv[16] = {0};
    float qp0 = 0.0f, qp1 = 0.0f;
    if (valid_h) {
        const float * qp = q576 + (static_cast<size_t>(t) * n_heads + h) * 576;
        #pragma unroll
        for (int i = 0; i < 16; ++i) qv[i] = __ldg(&qp[i * 32 + lane]) * scale;
        qp0 = __ldg(&qp[512 + lane]) * scale;
        qp1 = __ldg(&qp[544 + lane]) * scale;
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float ov[16] = {0};

    for (int p = p0; p < p1; p += P_TILE) {
        const int n_p = (p + P_TILE < p1) ? P_TILE : (p1 - p);
        if constexpr (KV_Q8_0) {
            const int tot_blk = n_p * 18;
            const BlockQ8_0 * kb_base = static_cast<const BlockQ8_0 *>(kv_cache) + static_cast<size_t>(p) * 18;
            for (int idx = warp; idx < tot_blk; idx += 16) {
                const int pi = (idx * 3641) >> 16; // exact idx / 18 for 0 <= idx < 180
                const int b  = idx - pi * 18;
                const BlockQ8_0 & kb = kb_base[idx];
                __half dh;
                *reinterpret_cast<uint16_t *>(&dh) = __ldg(&kb.d);
                s_kv[pi][b * 32 + lane] = __half2float(dh) * static_cast<float>(__ldg(&kb.qs[lane]));
            }
        } else {
            const int tot_u2 = n_p * 144;
            const uint2 * kc_base = static_cast<const uint2 *>(kv_cache) + static_cast<size_t>(p) * 144;
            for (int idx = tid; idx < tot_u2; idx += 512) {
                const int pi = idx / 144;
                const int c4 = idx - pi * 144;
                uint2 pk = __ldg(&kc_base[idx]);
                float2 f0 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.x));
                float2 f1 = __half22float2(*reinterpret_cast<const __half2 *>(&pk.y));
                reinterpret_cast<float4 *>(s_kv[pi])[c4] = make_float4(f0.x, f0.y, f1.x, f1.y);
            }
        }
        __syncthreads();

        if (valid_h) {
            #pragma unroll
            for (int pi = 0; pi < P_TILE; ++pi) {
                if (pi >= n_p || (p + pi) >= my_kv) break;
                const float * sk = s_kv[pi];
                float dot = qp0 * sk[512 + lane] + qp1 * sk[544 + lane];
                #pragma unroll
                for (int i = 0; i < 16; ++i) dot += qv[i] * sk[i * 32 + lane];
                dot = warp_sum(dot);
                const float score = __shfl_sync(0xffffffffu, dot, 0);

                const float m_new = fmaxf(m_i, score);
                const float alpha = expf(m_i - m_new);
                const float beta  = expf(score - m_new);
                l_i = l_i * alpha + beta;
                #pragma unroll
                for (int i = 0; i < 16; ++i) ov[i] = ov[i] * alpha + beta * sk[i * 32 + lane];
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (valid_h) {
        const size_t slot = (static_cast<size_t>(t) * n_heads + h) * SPLITS + s;
        if (lane == 0) {
            part_m[slot] = m_i;
            part_l[slot] = l_i;
        }
        float * op = part_o + slot * 512;
        #pragma unroll
        for (int i = 0; i < 16; ++i) op[i * 32 + lane] = ov[i];
    }
}

template <int SPLITS>
__global__ void mla_mqa_d512_reduce_strided_kernel(
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
        if (part_l[base + s] > 0.0f) m_max = fmaxf(m_max, part_m[base + s]);
    }

    float l_tot = 0.0f;
    float ov[16] = {0};
    for (int s = 0; s < SPLITS; ++s) {
        const float ls = part_l[base + s];
        if (ls <= 0.0f) continue;
        const float sc = expf(part_m[base + s] - m_max);
        l_tot += ls * sc;
        const float * po = part_o + (base + s) * 512;
        #pragma unroll
        for (int i = 0; i < 16; ++i) ov[i] += po[i * 32 + lane] * sc;
    }

    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    float * op = out + (static_cast<size_t>(t) * n_heads + h) * 512;
    #pragma unroll
    for (int i = 0; i < 16; ++i) op[i * 32 + lane] = ov[i] * inv_l;
}

__global__ void mla_head_sigmoid_gate_kernel(
    float *       __restrict__ o,    // [T, n_heads, head_dim]
    const float * __restrict__ gate, // [T, n_heads]
    int n_heads, int head_dim) {

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    if (d >= head_dim) return;
    const float g = 1.0f / (1.0f + expf(-gate[static_cast<size_t>(t) * n_heads + h]));
    o[(static_cast<size_t>(t) * n_heads + h) * head_dim + d] *= g;
}

} // namespace

void launch_mla_rope_split_q(
    const float * d_q_all, const int32_t * d_pos, float * d_q_nope, float * d_q_pe,
    int T, int n_heads, int qk_nope, int rope_dim, float rope_freq_base, cudaStream_t stream) {
    mla_rope_split_q_kernel<<<dim3(n_heads, T), 128, 0, stream>>>(
        d_q_all, d_pos, d_q_nope, d_q_pe, n_heads, qk_nope, rope_dim, rope_freq_base);
}

void launch_mla_kv_norm_rope_write_576(
    const float * d_kv_all, const float * d_norm_w, const int32_t * d_pos,
    void * d_kv_cache, bool kv_is_q8_0,
    int T, float rms_eps, float rope_freq_base, cudaStream_t stream) {
    if (kv_is_q8_0) {
        mla_kv_norm_rope_write_576_kernel<true><<<T, 256, 0, stream>>>(
            d_kv_all, d_norm_w, d_pos, d_kv_cache, rms_eps, rope_freq_base);
    } else {
        mla_kv_norm_rope_write_576_kernel<false><<<T, 256, 0, stream>>>(
            d_kv_all, d_norm_w, d_pos, d_kv_cache, rms_eps, rope_freq_base);
    }
}

void launch_mla_concat_q576(
    const float * d_q_abs, const float * d_q_pe, float * d_q576,
    int T, int n_heads, cudaStream_t stream) {
    mla_concat_q576_kernel<<<dim3(n_heads, T), 256, 0, stream>>>(d_q_abs, d_q_pe, d_q576, n_heads);
}

void launch_batched_head_proj_f16(
    const DeviceTensor & W, const float * d_x, float * d_y, int T, cudaStream_t stream,
    cublasHandle_t cublas, void * d_scratch, size_t scratch_bytes) {

    const int K       = static_cast<int>(W.ne[0]);
    const int M       = static_cast<int>(W.ne[1]);
    const int n_heads = static_cast<int>(W.ne[2]);

    if (T >= 8 && cublas != nullptr && d_scratch != nullptr) {
        auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
        if (device_has_tensor_cores(-1)) {
            const size_t x_bytes = align256(static_cast<size_t>(n_heads) * T * K * sizeof(__half));
            const size_t y_bytes = align256(static_cast<size_t>(n_heads) * T * M * sizeof(float));
            if (x_bytes + y_bytes <= scratch_bytes) {
                auto * base  = static_cast<uint8_t *>(d_scratch);
                auto * d_x16 = reinterpret_cast<__half *>(base);
                auto * d_yh  = reinterpret_cast<float  *>(base + x_bytes);

                const int tot_x = T * n_heads * K;
                transpose_thd_to_htd_kernel<__half><<<(tot_x + 255) / 256, 256, 0, stream>>>(d_x, d_x16, T, n_heads, K);

                const float alpha = 1.0f, beta = 0.0f;
                cublasSetStream(cublas, stream);
                cublasGemmStridedBatchedEx(
                    cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    M, T, K,
                    &alpha,
                    W.ptr, CUDA_R_16F, K, static_cast<long long>(M) * K,
                    d_x16, CUDA_R_16F, K, static_cast<long long>(T) * K,
                    &beta,
                    d_yh,  CUDA_R_32F, M, static_cast<long long>(T) * M,
                    n_heads,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                const int tot_y = T * n_heads * M;
                transpose_htd_to_thd_f32_kernel<<<(tot_y + 255) / 256, 256, 0, stream>>>(d_yh, d_y, T, n_heads, M);
                return;
            }
        }
    }

    constexpr int wpb = 8;
    dim3 grid((M + wpb - 1) / wpb, n_heads, T);
    batched_head_proj_f16_gemv_kernel<<<grid, dim3(32, wpb), 0, stream>>>(
        static_cast<const __half *>(W.ptr), d_x, d_y, M, K, n_heads);
}

void launch_mla_attn_d576_v512(
    const float * d_q576, const void * d_kv_cache, bool kv_is_q8_0,
    const int32_t * d_pos, float * d_out, void * d_scratch,
    int T, int n_heads, float scale, cudaStream_t stream,
    cublasHandle_t cublas, size_t scratch_bytes, int kv_len_max) {

    // Fast Tiled FlashMLA Prefill (T >= 8): FP16 Tensor Core with large K_TILE (up to 2048 -> 4x less O DRAM!)
    if (T >= 8 && cublas != nullptr && kv_len_max > 0 && device_has_tensor_cores(-1)) {
        auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
        const int K_TILE = (kv_len_max > 2048) ? 2048 : ((kv_len_max > 512) ? 1024 : 512);
        // Pick largest t_slice <= T whose scratch (q16 + k16 + s32 + p16 + m + l) fits in scratch_bytes!
        const size_t k16_b = align256(static_cast<size_t>(K_TILE) * 576 * sizeof(__half));
        const size_t per_m = 576 * sizeof(__half) + static_cast<size_t>(K_TILE) * (sizeof(float) + sizeof(__half)) + 2 * sizeof(float);
        int max_m = (scratch_bytes > k16_b + 4096) ? static_cast<int>((scratch_bytes - k16_b - 4096) / per_m) : 0;
        int t_slice = std::min(T, std::max(0, max_m / n_heads));
        if (t_slice >= 8) {
            const float one = 1.0f, zero = 0.0f;
            cublasSetStream(cublas, stream);
            for (int t0 = 0; t0 < T; t0 += t_slice) {
                const int tc = std::min(t_slice, T - t0);
                const int M  = tc * n_heads;
                const size_t q16_b = align256(static_cast<size_t>(M) * 576 * sizeof(__half));
                const size_t s32_b = align256(static_cast<size_t>(M) * K_TILE * sizeof(float));
                const size_t p16_b = align256(static_cast<size_t>(M) * K_TILE * sizeof(__half));
                const size_t ml_b  = align256(static_cast<size_t>(M) * sizeof(float));

                auto * base = static_cast<uint8_t *>(d_scratch);
                size_t off = 0;
                auto * d_q16 = reinterpret_cast<__half *>(base + off); off += q16_b;
                auto * d_k16 = reinterpret_cast<__half *>(base + off); off += k16_b;
                auto * d_s32 = reinterpret_cast<float  *>(base + off); off += s32_b;
                auto * d_p16 = reinterpret_cast<__half *>(base + off); off += p16_b;
                auto * d_m   = reinterpret_cast<float  *>(base + off); off += ml_b;
                auto * d_l   = reinterpret_cast<float  *>(base + off);

                const float * q_ptr = d_q576 + static_cast<size_t>(t0) * n_heads * 576;
                float *       o_ptr = d_out  + static_cast<size_t>(t0) * n_heads * 512;
                const int32_t * p_ptr = d_pos + t0;

                const int64_t nq = static_cast<int64_t>(M) * 576;
                mla_scale_cast_kernel<__half><<<(nq + 255) / 256, 256, 0, stream>>>(q_ptr, d_q16, nq, scale);
                mla_init_flash_state_kernel<<<(M + 255) / 256, 256, 0, stream>>>(d_m, d_l, M);

                for (int k0 = 0; k0 < kv_len_max; k0 += K_TILE) {
                    const int kc = (k0 + K_TILE < kv_len_max) ? K_TILE : (kv_len_max - k0);
                    if (kv_is_q8_0) {
                        mla_dequant_kv576_tile_kernel<true, __half><<<kc, 256, 0, stream>>>(d_kv_cache, d_k16, k0, kc);
                    } else {
                        mla_dequant_kv576_tile_kernel<false, __half><<<kc, 256, 0, stream>>>(d_kv_cache, d_k16, k0, kc);
                    }

                    cublasGemmEx(
                        cublas,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        kc, M, 576,
                        &one,
                        d_k16, CUDA_R_16F, 576,
                        d_q16, CUDA_R_16F, 576,
                        &zero,
                        d_s32, CUDA_R_32F, kc,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                    mla_flash_online_softmax_rescale_kernel<__half><<<M, 128, 0, stream>>>(
                        d_s32, d_p16, p_ptr, d_m, d_l, o_ptr, n_heads, k0, kc);

                    cublasGemmEx(
                        cublas,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        512, M, kc,
                        &one,
                        d_k16, CUDA_R_16F, 576,
                        d_p16, CUDA_R_16F, kc,
                        &one,
                        o_ptr, CUDA_R_32F, 512,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                }

                mla_flash_finalize_out_kernel<<<M, 128, 0, stream>>>(o_ptr, d_l, M);
            }
            return;
        }
    }

    // Multi-Head-Shared Split-K Decode (Coalesced LDG.128 Q8_0 dp4a + Tile-Softmax!)
    if (kv_is_q8_0) {
        if (n_heads <= 16) {
            constexpr int S = 128;
            const size_t n_slots = static_cast<size_t>(T) * n_heads * S;
            float * pm = static_cast<float *>(d_scratch);
            float * pl = pm + n_slots;
            float * po = pl + n_slots;
            if (T == 2) {
                constexpr int H_PER_BLK = 8;
                mla_mqa_d576_q8_dp4a_decode_t2_kernel<S><<<dim3((n_heads + H_PER_BLK - 1) / H_PER_BLK, S), dim3(32, 16), 0, stream>>>(
                    d_q576, d_kv_cache, d_pos, pm, pl, po, n_heads, scale);
            } else {
                constexpr int H_PER_BLK = 16;
                mla_mqa_d576_q8_dp4a_decode_kernel<S, H_PER_BLK><<<dim3((n_heads + H_PER_BLK - 1) / H_PER_BLK, S, T), dim3(32, H_PER_BLK), 0, stream>>>(
                    d_q576, d_kv_cache, d_pos, pm, pl, po, n_heads, scale);
            }
            mla_mqa_d512_reduce_fast_kernel<S><<<dim3(n_heads, T), 128, 0, stream>>>(pm, pl, po, d_out, n_heads);
        } else {
            constexpr int S = 64;
            const size_t n_slots = static_cast<size_t>(T) * n_heads * S;
            float * pm = static_cast<float *>(d_scratch);
            float * pl = pm + n_slots;
            float * po = pl + n_slots;
            if (T == 2) {
                constexpr int H_PER_BLK = 8;
                mla_mqa_d576_q8_dp4a_decode_t2_kernel<S><<<dim3((n_heads + H_PER_BLK - 1) / H_PER_BLK, S), dim3(32, 16), 0, stream>>>(
                    d_q576, d_kv_cache, d_pos, pm, pl, po, n_heads, scale);
            } else {
                constexpr int H_PER_BLK = 16;
                mla_mqa_d576_q8_dp4a_decode_kernel<S, H_PER_BLK><<<dim3((n_heads + H_PER_BLK - 1) / H_PER_BLK, S, T), dim3(32, H_PER_BLK), 0, stream>>>(
                    d_q576, d_kv_cache, d_pos, pm, pl, po, n_heads, scale);
            }
            mla_mqa_d512_reduce_fast_kernel<S><<<dim3(n_heads, T), 128, 0, stream>>>(pm, pl, po, d_out, n_heads);
        }
        return;
    }
    constexpr int S = 64;
    const size_t n_slots = static_cast<size_t>(T) * n_heads * S;
    float * pm = static_cast<float *>(d_scratch);
    float * pl = pm + n_slots;
    float * po = pl + n_slots;
    if (T == 2) {
        constexpr int H_PER_BLK = 8;
        dim3 grid((n_heads + H_PER_BLK - 1) / H_PER_BLK, S);
        dim3 block(32, 16);
        mla_mqa_d576_decode_shared_t2_kernel<false, S><<<grid, block, 0, stream>>>(
            d_q576, d_kv_cache, d_pos, pm, pl, po, n_heads, scale);
    } else {
        constexpr int H_PER_BLK = 16;
        dim3 grid((n_heads + H_PER_BLK - 1) / H_PER_BLK, S, T);
        dim3 block(32, H_PER_BLK);
        mla_mqa_d576_decode_shared_kernel<false, S, H_PER_BLK><<<grid, block, 0, stream>>>(
            d_q576, d_kv_cache, d_pos, pm, pl, po, n_heads, scale);
    }
    mla_mqa_d512_reduce_strided_kernel<S><<<dim3(n_heads, T), 32, 0, stream>>>(pm, pl, po, d_out, n_heads);
}

void launch_mla_head_sigmoid_gate(
    float * d_o, const float * d_gate, int T, int n_heads, int head_dim, cudaStream_t stream) {
    mla_head_sigmoid_gate_kernel<<<dim3((head_dim + 127) / 128, n_heads, T), 128, 0, stream>>>(
        d_o, d_gate, n_heads, head_dim);
}

} // namespace rookery
