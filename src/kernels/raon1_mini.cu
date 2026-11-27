#include "core/device.h"
#include "kernels/iq_tables.cuh"
#include "kernels/kernels.h"
#include "kernels/q8_1_act.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace rookery {

namespace {

// ============================================================================
// Common Device Helpers & Block Structs
// ============================================================================

struct alignas(16) RaonActF16Block {
    __half2 x[16];  // 32 normalized elements: x_raw[0..31] / amax
    float   d;      // amax
    float   s0;     // sum(x_raw[0..15])
    float   s1;     // sum(x_raw[16..31])
    float   s_all;  // s0 + s1
};
static_assert(sizeof(RaonActF16Block) == 80, "RaonActF16Block must be 80 bytes");

struct BlockIQ4_XS_R {
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[4];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockIQ4_XS_R) == 136, "BlockIQ4_XS_R size");

struct BlockQ5_1_R {
    uint16_t d;
    uint16_t m;
    uint8_t  qh[4];
    uint8_t  qs[16];
};
static_assert(sizeof(BlockQ5_1_R) == 24, "BlockQ5_1_R size");

struct RaonTileDesc {
    int32_t eid;
    int32_t start;
    int32_t count;
};

struct alignas(64) RaonMlaKv576Q8Pos {
    int8_t   qs[576];
    uint16_t d[18];
    uint16_t pad[14];
};
static_assert(sizeof(RaonMlaKv576Q8Pos) == 640, "RaonMlaKv576Q8Pos must match MlaKv576Q8Pos (640B)");

struct RaonMultiIQ4Desc {
    const BlockIQ4_XS_R * w[6];
    float *               y[6];
    int                   m[6];
    int                   row_end[6];
    int                   n_mat;
};

struct RaonMultiF16Desc {
    const __half * w[4];
    float *        y[4];
    int            m[4];
    int            row_end[4];
    int            n_mat;
};

__device__ __forceinline__ float raon_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float raon_warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ float raon_half_bits_to_f32(uint16_t bits) {
    __half h;
    *reinterpret_cast<uint16_t *>(&h) = bits;
    return __half2float(h);
}

__device__ __forceinline__ float raon_sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float raon_logsigmoidf(float x) {
    return (x >= 0.0f) ? -log1pf(expf(-x)) : (x - log1pf(expf(x)));
}

__device__ __forceinline__ float raon_situ_act(float g, float u, float beta, float linear_beta) {
    const float a = beta * tanhf(g / beta) * raon_sigmoidf(g);
    const float b = (linear_beta > 0.0f) ? (linear_beta * tanhf(u / linear_beta)) : u;
    return a * b;
}

__device__ __forceinline__ __half2 raon_hfma2(__half2 a, __half2 b, __half2 c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 600 || __CUDA_ARCH__ == 620 || __CUDA_ARCH__ >= 700)
    return __hfma2(a, b, c);
#else
    float2 fa = __half22float2(a), fb = __half22float2(b), fc = __half22float2(c);
    return __floats2half2_rn(fa.x * fb.x + fc.x, fa.y * fb.y + fc.y);
#endif
}

__device__ __forceinline__ __half2 raon_hmul2(__half2 a, __half2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hmul2(a, b);
#else
    float2 fa = __half22float2(a), fb = __half22float2(b);
    return __floats2half2_rn(fa.x * fb.x, fa.y * fb.y);
#endif
}

__device__ __forceinline__ __half2 raon_hadd2(__half2 a, __half2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hadd2(a, b);
#else
    float2 fa = __half22float2(a), fb = __half22float2(b);
    return __floats2half2_rn(fa.x + fb.x, fa.y + fb.y);
#endif
}

__device__ __forceinline__ __half2 raon_u8x2_to_h2_lo(uint32_t u) {
    uint32_t p = __byte_perm(u, 0x00006400u, 0x5150);
    return *reinterpret_cast<const __half2 *>(&p);
}

__device__ __forceinline__ __half2 raon_u8x2_to_h2_hi(uint32_t u) {
    uint32_t p = __byte_perm(u, 0x00006400u, 0x5352);
    return *reinterpret_cast<const __half2 *>(&p);
}

__device__ __forceinline__ __half2 raon_lut_h2(const __half * __restrict__ lut, uint32_t i0, uint32_t i1) {
    return __halves2half2(lut[i0], lut[i1]);
}

// Single-instruction LDG.64 IQ4_XS header decode (d, scales_h, scales_l[4] packed in 8 bytes!)
__device__ __forceinline__ float raon_iq4_xs_dl(const BlockIQ4_XS_R & b, int ib) {
    const uint2 hdr = __ldg(reinterpret_cast<const uint2 *>(&b));
    const float d     = raon_half_bits_to_f32(static_cast<uint16_t>(hdr.x & 0xFFFFu));
    const uint32_t sh = hdr.x >> 16;
    const uint32_t sl = (hdr.y >> (4 * ib)) & 0xFu;
    const int ls      = static_cast<int>(sl | (((sh >> (2 * ib)) & 3u) << 4));
    return d * static_cast<float>(ls - 32);
}

__device__ __forceinline__ int raon_pack_iq4_lo4(const int8_t * __restrict__ lut, uint32_t q4) {
    return static_cast<int>(
          static_cast<uint32_t>(static_cast<uint8_t>(lut[(q4 >>  0) & 0xFu]))
        | (static_cast<uint32_t>(static_cast<uint8_t>(lut[(q4 >>  8) & 0xFu])) << 8)
        | (static_cast<uint32_t>(static_cast<uint8_t>(lut[(q4 >> 16) & 0xFu])) << 16)
        | (static_cast<uint32_t>(static_cast<uint8_t>(lut[(q4 >> 24) & 0xFu])) << 24));
}

__device__ __forceinline__ int raon_pack_iq4_hi4(const int8_t * __restrict__ lut, uint32_t q4) {
    return static_cast<int>(
          static_cast<uint32_t>(static_cast<uint8_t>(lut[(q4 >>  4) & 0xFu]))
        | (static_cast<uint32_t>(static_cast<uint8_t>(lut[(q4 >> 12) & 0xFu])) << 8)
        | (static_cast<uint32_t>(static_cast<uint8_t>(lut[(q4 >> 20) & 0xFu])) << 16)
        | (static_cast<uint32_t>(static_cast<uint8_t>(lut[(q4 >> 28) & 0xFu])) << 24));
}

// ============================================================================
// 1. RaonMHC Closed-Form 4x3 Helmert Chart + Aggregate + RMSNorm & Mix+Steer
// ============================================================================

__global__ void raon_mhc_pre_rmsnorm_kernel(
    const float * __restrict__ stream,
    const float * __restrict__ mixes,
    const float * __restrict__ hc_scale,
    const float * __restrict__ hc_base,
    const float * __restrict__ norm_w,
    float *       __restrict__ post_out,
    float *       __restrict__ comb_out,
    float *       __restrict__ x_norm_out,
    int n_embd, float post_factor, float rms_eps) {

    const int t = blockIdx.x;
    __shared__ float s_pre[4];
    __shared__ float s_red[8];

    if (threadIdx.x == 0) {
        const float * m = mixes + static_cast<size_t>(t) * 17;
        const float a_pre  = hc_scale[0];
        const float a_post = hc_scale[1];
        const float a_z    = hc_scale[2];
        const float eta    = hc_scale[3];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            s_pre[i] = raon_sigmoidf(a_pre * m[i] + hc_base[i]);
            post_out[static_cast<size_t>(t) * 4 + i] = post_factor * raon_sigmoidf(a_post * m[4 + i] + hc_base[4 + i]);
        }

        float tz[3][3];
        #pragma unroll
        for (int r = 0; r < 3; ++r) {
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                tz[r][c] = a_z * m[8 + r * 3 + c] + hc_base[8 + r * 3 + c];
            }
        }

        constexpr float inv_s2  = 0.7071067811865475f;
        constexpr float inv_s6  = 0.4082482904638630f;
        constexpr float inv_s12 = 0.2886751345948129f;
        const float Q[4][3] = {
            {  inv_s2,   inv_s6,        inv_s12 },
            { -inv_s2,   inv_s6,        inv_s12 },
            {  0.0f,    -2.0f * inv_s6, inv_s12 },
            {  0.0f,     0.0f,         -3.0f * inv_s12 }
        };

        float tmp[4][3];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                tmp[i][k] = Q[i][0] * tz[0][k] + Q[i][1] * tz[1][k] + Q[i][2] * tz[2][k];
            }
        }

        float E[4][4];
        float rho = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float eij = tmp[i][0] * Q[j][0] + tmp[i][1] * Q[j][1] + tmp[i][2] * Q[j][2];
                E[i][j] = eij;
                const float inv_b = (i == j) ? 1.6f : 8.0f;
                rho = fmaxf(rho, -eij * inv_b);
            }
        }

        const float inv_1prho = 1.0f / (1.0f + rho);
        const float gamma     = raon_sigmoidf(eta);
        const float om_gamma  = 1.0f - gamma;
        float * comb = comb_out + static_cast<size_t>(t) * 16;

        #pragma unroll
        for (int dst = 0; dst < 4; ++dst) {
            #pragma unroll
            for (int src = 0; src < 4; ++src) {
                const float bij = (dst == src) ? 0.625f : 0.125f;
                const float phi = bij + E[dst][src] * inv_1prho;
                comb[dst + 4 * src] = ((dst == src) ? om_gamma : 0.0f) + gamma * phi;
            }
        }
    }
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    const int n4 = n_embd >> 2;
    const float4 * r0 = reinterpret_cast<const float4 *>(stream + (static_cast<size_t>(t) * 4 + 0) * n_embd);
    const float4 * r1 = reinterpret_cast<const float4 *>(stream + (static_cast<size_t>(t) * 4 + 1) * n_embd);
    const float4 * r2 = reinterpret_cast<const float4 *>(stream + (static_cast<size_t>(t) * 4 + 2) * n_embd);
    const float4 * r3 = reinterpret_cast<const float4 *>(stream + (static_cast<size_t>(t) * 4 + 3) * n_embd);
    const float4 * nw = norm_w ? reinterpret_cast<const float4 *>(norm_w) : nullptr;
    float4 * yo       = reinterpret_cast<float4 *>(x_norm_out + static_cast<size_t>(t) * n_embd);

    float ss = 0.0f;
    const bool use_reg = (n4 <= 768);
    float4 reg_v[3];
    if (use_reg) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int i = threadIdx.x + k * 256;
            if (i < n4) {
                const float4 a0 = r0[i], a1 = r1[i], a2 = r2[i], a3 = r3[i];
                float4 v;
                v.x = p0 * a0.x + p1 * a1.x + p2 * a2.x + p3 * a3.x;
                v.y = p0 * a0.y + p1 * a1.y + p2 * a2.y + p3 * a3.y;
                v.z = p0 * a0.z + p1 * a1.z + p2 * a2.z + p3 * a3.z;
                v.w = p0 * a0.w + p1 * a1.w + p2 * a2.w + p3 * a3.w;
                reg_v[k] = v;
                ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        }
    } else {
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 a0 = r0[i], a1 = r1[i], a2 = r2[i], a3 = r3[i];
            float4 v;
            v.x = p0 * a0.x + p1 * a1.x + p2 * a2.x + p3 * a3.x;
            v.y = p0 * a0.y + p1 * a1.y + p2 * a2.y + p3 * a3.y;
            v.z = p0 * a0.z + p1 * a1.z + p2 * a2.z + p3 * a3.z;
            v.w = p0 * a0.w + p1 * a1.w + p2 * a2.w + p3 * a3.w;
            yo[i] = v;
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    ss = raon_warp_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();

    if (warp == 0) {
        float tot = (lane < 8) ? s_red[lane] : 0.0f;
        tot = raon_warp_sum(tot);
        if (lane == 0) s_red[0] = rsqrtf(tot / static_cast<float>(n_embd) + rms_eps);
    }
    __syncthreads();

    const float inv_rms = s_red[0];
    if (use_reg) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int i = threadIdx.x + k * 256;
            if (i < n4) {
                float4 v = reg_v[k];
                if (nw) {
                    const float4 w = __ldg(&nw[i]);
                    v.x *= inv_rms * w.x; v.y *= inv_rms * w.y;
                    v.z *= inv_rms * w.z; v.w *= inv_rms * w.w;
                } else {
                    v.x *= inv_rms; v.y *= inv_rms; v.z *= inv_rms; v.w *= inv_rms;
                }
                yo[i] = v;
            }
        }
    } else {
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            float4 v = yo[i];
            if (nw) {
                const float4 w = __ldg(&nw[i]);
                v.x *= inv_rms * w.x; v.y *= inv_rms * w.y;
                v.z *= inv_rms * w.z; v.w *= inv_rms * w.w;
            } else {
                v.x *= inv_rms; v.y *= inv_rms; v.z *= inv_rms; v.w *= inv_rms;
            }
            yo[i] = v;
        }
    }
}

__global__ void raon_mhc_post_steer_kernel(
    const float * __restrict__ sub,
    const float * __restrict__ residual,
    const float * __restrict__ post,
    const float * __restrict__ comb,
    const float * __restrict__ steer,
    float steer_alpha,
    float *       __restrict__ out,
    int n_embd) {

    const int t  = blockIdx.y;
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = n_embd >> 2;
    if (i >= n4) return;

    float4 sv = reinterpret_cast<const float4 *>(sub + static_cast<size_t>(t) * n_embd)[i];
    if (steer && steer_alpha != 0.0f) {
        const float4 st = __ldg(&reinterpret_cast<const float4 *>(steer)[i]);
        sv.x += steer_alpha * st.x;
        sv.y += steer_alpha * st.y;
        sv.z += steer_alpha * st.z;
        sv.w += steer_alpha * st.w;
    }

    const float4 v0 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 0) * n_embd)[i];
    const float4 v1 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 1) * n_embd)[i];
    const float4 v2 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 2) * n_embd)[i];
    const float4 v3 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 3) * n_embd)[i];

    const float * p = post + static_cast<size_t>(t) * 4;
    const float * c = comb + static_cast<size_t>(t) * 16;

    auto mix4 = [&](float pd, float c0, float c1, float c2, float c3) -> float4 {
        return make_float4(
            pd * sv.x + c0 * v0.x + c1 * v1.x + c2 * v2.x + c3 * v3.x,
            pd * sv.y + c0 * v0.y + c1 * v1.y + c2 * v2.y + c3 * v3.y,
            pd * sv.z + c0 * v0.z + c1 * v1.z + c2 * v2.z + c3 * v3.z,
            pd * sv.w + c0 * v0.w + c1 * v1.w + c2 * v2.w + c3 * v3.w);
    };

    reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * 4 + 0) * n_embd)[i] = mix4(p[0], c[0], c[4], c[8],  c[12]);
    reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * 4 + 1) * n_embd)[i] = mix4(p[1], c[1], c[5], c[9],  c[13]);
    reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * 4 + 2) * n_embd)[i] = mix4(p[2], c[2], c[6], c[10], c[14]);
    reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * 4 + 3) * n_embd)[i] = mix4(p[3], c[3], c[7], c[11], c[15]);
}

// Fused FFN write-back (+ optional reasoning-effort steering) + next layer's hc_attn_fn BF16 flat RMSNorm mix!
__global__ void raon_mhc_post_steer_flat_mix_bf16_kernel(
    const float *    __restrict__ sub,
    const float *    __restrict__ residual,
    const float *    __restrict__ post,
    const float *    __restrict__ comb,
    const float *    __restrict__ steer,
    float steer_alpha,
    float *          __restrict__ out,
    const uint16_t * __restrict__ hc_fn,
    float *          __restrict__ mixes,
    int n_embd, int n_out, float rms_eps) {

    const int row = blockIdx.x;
    const int t   = blockIdx.y;
    if (row >= n_out) return;

    const int n4     = n_embd >> 2;
    const int hc_dim = n_embd * 4;
    const float * p  = post + static_cast<size_t>(t) * 4;
    const float * c  = comb + static_cast<size_t>(t) * 16;

    const float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
    const float c00 = c[0], c01 = c[4], c02 = c[8],  c03 = c[12];
    const float c10 = c[1], c11 = c[5], c12 = c[9],  c13 = c[13];
    const float c20 = c[2], c21 = c[6], c22 = c[10], c23 = c[14];
    const float c30 = c[3], c31 = c[7], c32 = c[11], c33 = c[15];

    const float4 * s_ptr  = reinterpret_cast<const float4 *>(sub + static_cast<size_t>(t) * n_embd);
    const float4 * st_ptr = (steer && steer_alpha != 0.0f) ? reinterpret_cast<const float4 *>(steer) : nullptr;
    const float4 * r0 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 0) * n_embd);
    const float4 * r1 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 1) * n_embd);
    const float4 * r2 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 2) * n_embd);
    const float4 * r3 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 3) * n_embd);

    float4 * o0 = reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * 4 + 0) * n_embd);
    float4 * o1 = reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * 4 + 1) * n_embd);
    float4 * o2 = reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * 4 + 2) * n_embd);
    float4 * o3 = reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * 4 + 3) * n_embd);

    const uint2 * w0_p = reinterpret_cast<const uint2 *>(hc_fn + static_cast<size_t>(row) * hc_dim + 0 * n_embd);
    const uint2 * w1_p = reinterpret_cast<const uint2 *>(hc_fn + static_cast<size_t>(row) * hc_dim + 1 * n_embd);
    const uint2 * w2_p = reinterpret_cast<const uint2 *>(hc_fn + static_cast<size_t>(row) * hc_dim + 2 * n_embd);
    const uint2 * w3_p = reinterpret_cast<const uint2 *>(hc_fn + static_cast<size_t>(row) * hc_dim + 3 * n_embd);

    auto dot_bf16_4 = [](const float4 & v, const uint2 & wp) -> float {
        return __uint_as_float((wp.x & 0xffffu) << 16) * v.x
             + __uint_as_float(wp.x & 0xffff0000u)     * v.y
             + __uint_as_float((wp.y & 0xffffu) << 16) * v.z
             + __uint_as_float(wp.y & 0xffff0000u)     * v.w;
    };

    float ss = 0.0f, dot = 0.0f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 sv = s_ptr[i];
        if (st_ptr) {
            const float4 st = __ldg(&st_ptr[i]);
            sv.x += steer_alpha * st.x; sv.y += steer_alpha * st.y;
            sv.z += steer_alpha * st.z; sv.w += steer_alpha * st.w;
        }
        const float4 v0 = r0[i], v1 = r1[i], v2 = r2[i], v3 = r3[i];

        const float4 y0 = make_float4(
            p0 * sv.x + c00 * v0.x + c01 * v1.x + c02 * v2.x + c03 * v3.x,
            p0 * sv.y + c00 * v0.y + c01 * v1.y + c02 * v2.y + c03 * v3.y,
            p0 * sv.z + c00 * v0.z + c01 * v1.z + c02 * v2.z + c03 * v3.z,
            p0 * sv.w + c00 * v0.w + c01 * v1.w + c02 * v2.w + c03 * v3.w);
        const float4 y1 = make_float4(
            p1 * sv.x + c10 * v0.x + c11 * v1.x + c12 * v2.x + c13 * v3.x,
            p1 * sv.y + c10 * v0.y + c11 * v1.y + c12 * v2.y + c13 * v3.y,
            p1 * sv.z + c10 * v0.z + c11 * v1.z + c12 * v2.z + c13 * v3.z,
            p1 * sv.w + c10 * v0.w + c11 * v1.w + c12 * v2.w + c13 * v3.w);
        const float4 y2 = make_float4(
            p2 * sv.x + c20 * v0.x + c21 * v1.x + c22 * v2.x + c23 * v3.x,
            p2 * sv.y + c20 * v0.y + c21 * v1.y + c22 * v2.y + c23 * v3.y,
            p2 * sv.z + c20 * v0.z + c21 * v1.z + c22 * v2.z + c23 * v3.z,
            p2 * sv.w + c20 * v0.w + c21 * v1.w + c22 * v2.w + c23 * v3.w);
        const float4 y3 = make_float4(
            p3 * sv.x + c30 * v0.x + c31 * v1.x + c32 * v2.x + c33 * v3.x,
            p3 * sv.y + c30 * v0.y + c31 * v1.y + c32 * v2.y + c33 * v3.y,
            p3 * sv.z + c30 * v0.z + c31 * v1.z + c32 * v2.z + c33 * v3.z,
            p3 * sv.w + c30 * v0.w + c31 * v1.w + c32 * v2.w + c33 * v3.w);

        if (row == 0) {
            o0[i] = y0; o1[i] = y1; o2[i] = y2; o3[i] = y3;
        }
        ss += y0.x * y0.x + y0.y * y0.y + y0.z * y0.z + y0.w * y0.w
            + y1.x * y1.x + y1.y * y1.y + y1.z * y1.z + y1.w * y1.w
            + y2.x * y2.x + y2.y * y2.y + y2.z * y2.z + y2.w * y2.w
            + y3.x * y3.x + y3.y * y3.y + y3.z * y3.z + y3.w * y3.w;
        dot += dot_bf16_4(y0, __ldg(&w0_p[i])) + dot_bf16_4(y1, __ldg(&w1_p[i]))
             + dot_bf16_4(y2, __ldg(&w2_p[i])) + dot_bf16_4(y3, __ldg(&w3_p[i]));
    }

    __shared__ float s_ss[8];
    __shared__ float s_dot[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    ss  = raon_warp_sum(ss);
    dot = raon_warp_sum(dot);
    if (lane == 0) { s_ss[warp] = ss; s_dot[warp] = dot; }
    __syncthreads();

    if (warp == 0) {
        float t_ss  = raon_warp_sum((lane < 8) ? s_ss[lane]  : 0.0f);
        float t_dot = raon_warp_sum((lane < 8) ? s_dot[lane] : 0.0f);
        if (lane == 0) {
            const float inv_rms = rsqrtf(t_ss / static_cast<float>(hc_dim) + rms_eps);
            mixes[static_cast<size_t>(t) * n_out + row] = t_dot * inv_rms;
        }
    }
}

// Single-pass (or Split-K for decode) fused post+steer+flat-mix BF16 kernel.
// Computes y0..y3 ONCE per chunk (17x fewer residual reads/ALU than per-row grid) and dots all
// n_out <= 18 rows concurrently via 18 half-warps (288 threads) with 128B coalesced BF16 loads.
template <bool SPLIT_K>
__global__ void raon_mhc_post_steer_flat_mix_bf16_fast_kernel(
    const float *    __restrict__ sub,
    const float *    __restrict__ residual,
    const float *    __restrict__ post,
    const float *    __restrict__ comb,
    const float *    __restrict__ steer,
    float steer_alpha,
    float *          __restrict__ out,
    const uint16_t * __restrict__ hc_fn,
    float *          __restrict__ out_buf, // mixes if !SPLIT_K, else part[T, S, n_out+1]
    int n_embd, int n_out, float rms_eps) {

    const int s      = blockIdx.x;
    const int S      = gridDim.x;
    const int t      = blockIdx.y;
    const int hw     = threadIdx.x >> 4; // 0..17
    const int sl     = threadIdx.x & 15; // 0..15
    const int n4     = n_embd >> 2;
    const int hc_dim = n_embd * 4;

    __shared__ float4 s_y[4][16];
    __shared__ float  s_inv_rms;

    const float * p = post + static_cast<size_t>(t) * 4;
    const float * c = comb + static_cast<size_t>(t) * 16;
    const float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
    const float c00 = c[0], c01 = c[4], c02 = c[8],  c03 = c[12];
    const float c10 = c[1], c11 = c[5], c12 = c[9],  c13 = c[13];
    const float c20 = c[2], c21 = c[6], c22 = c[10], c23 = c[14];
    const float c30 = c[3], c31 = c[7], c32 = c[11], c33 = c[15];

    const float4 * s_ptr  = reinterpret_cast<const float4 *>(sub + static_cast<size_t>(t) * n_embd);
    const float4 * st_ptr = (steer && steer_alpha != 0.0f) ? reinterpret_cast<const float4 *>(steer) : nullptr;
    const float4 * r0 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 0) * n_embd);
    const float4 * r1 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 1) * n_embd);
    const float4 * r2 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 2) * n_embd);
    const float4 * r3 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 3) * n_embd);

    float4 * o0 = reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * 4 + 0) * n_embd);
    float4 * o1 = reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * 4 + 1) * n_embd);
    float4 * o2 = reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * 4 + 2) * n_embd);
    float4 * o3 = reinterpret_cast<float4 *>(out + (static_cast<size_t>(t) * 4 + 3) * n_embd);

    const int row = (hw < n_out) ? hw : 0;
    const uint2 * w0_p = reinterpret_cast<const uint2 *>(hc_fn + static_cast<size_t>(row) * hc_dim + 0 * n_embd);
    const uint2 * w1_p = reinterpret_cast<const uint2 *>(hc_fn + static_cast<size_t>(row) * hc_dim + 1 * n_embd);
    const uint2 * w2_p = reinterpret_cast<const uint2 *>(hc_fn + static_cast<size_t>(row) * hc_dim + 2 * n_embd);
    const uint2 * w3_p = reinterpret_cast<const uint2 *>(hc_fn + static_cast<size_t>(row) * hc_dim + 3 * n_embd);

    auto dot_bf16_4 = [](const float4 & v, const uint2 & wp) -> float {
        return __uint_as_float((wp.x & 0xffffu) << 16) * v.x
             + __uint_as_float(wp.x & 0xffff0000u)     * v.y
             + __uint_as_float((wp.y & 0xffffu) << 16) * v.z
             + __uint_as_float(wp.y & 0xffff0000u)     * v.w;
    };

    float ss = 0.0f;
    float dot = 0.0f;
    const int step = S * 16;
    for (int base = s * 16; base < n4; base += step) {
        const int i = base + sl;
        if (hw == 0 && i < n4) {
            float4 sv = s_ptr[i];
            if (st_ptr) {
                const float4 st = __ldg(&st_ptr[i]);
                sv.x += steer_alpha * st.x; sv.y += steer_alpha * st.y;
                sv.z += steer_alpha * st.z; sv.w += steer_alpha * st.w;
            }
            const float4 v0 = r0[i], v1 = r1[i], v2 = r2[i], v3 = r3[i];
            const float4 y0 = make_float4(
                p0 * sv.x + c00 * v0.x + c01 * v1.x + c02 * v2.x + c03 * v3.x,
                p0 * sv.y + c00 * v0.y + c01 * v1.y + c02 * v2.y + c03 * v3.y,
                p0 * sv.z + c00 * v0.z + c01 * v1.z + c02 * v2.z + c03 * v3.z,
                p0 * sv.w + c00 * v0.w + c01 * v1.w + c02 * v2.w + c03 * v3.w);
            const float4 y1 = make_float4(
                p1 * sv.x + c10 * v0.x + c11 * v1.x + c12 * v2.x + c13 * v3.x,
                p1 * sv.y + c10 * v0.y + c11 * v1.y + c12 * v2.y + c13 * v3.y,
                p1 * sv.z + c10 * v0.z + c11 * v1.z + c12 * v2.z + c13 * v3.z,
                p1 * sv.w + c10 * v0.w + c11 * v1.w + c12 * v2.w + c13 * v3.w);
            const float4 y2 = make_float4(
                p2 * sv.x + c20 * v0.x + c21 * v1.x + c22 * v2.x + c23 * v3.x,
                p2 * sv.y + c20 * v0.y + c21 * v1.y + c22 * v2.y + c23 * v3.y,
                p2 * sv.z + c20 * v0.z + c21 * v1.z + c22 * v2.z + c23 * v3.z,
                p2 * sv.w + c20 * v0.w + c21 * v1.w + c22 * v2.w + c23 * v3.w);
            const float4 y3 = make_float4(
                p3 * sv.x + c30 * v0.x + c31 * v1.x + c32 * v2.x + c33 * v3.x,
                p3 * sv.y + c30 * v0.y + c31 * v1.y + c32 * v2.y + c33 * v3.y,
                p3 * sv.z + c30 * v0.z + c31 * v1.z + c32 * v2.z + c33 * v3.z,
                p3 * sv.w + c30 * v0.w + c31 * v1.w + c32 * v2.w + c33 * v3.w);
            o0[i] = y0; o1[i] = y1; o2[i] = y2; o3[i] = y3;
            s_y[0][sl] = y0; s_y[1][sl] = y1; s_y[2][sl] = y2; s_y[3][sl] = y3;
            ss += y0.x * y0.x + y0.y * y0.y + y0.z * y0.z + y0.w * y0.w
                + y1.x * y1.x + y1.y * y1.y + y1.z * y1.z + y1.w * y1.w
                + y2.x * y2.x + y2.y * y2.y + y2.z * y2.z + y2.w * y2.w
                + y3.x * y3.x + y3.y * y3.y + y3.z * y3.z + y3.w * y3.w;
        }
        __syncthreads();
        if (hw < n_out && i < n4) {
            dot += dot_bf16_4(s_y[0][sl], __ldg(&w0_p[i]))
                 + dot_bf16_4(s_y[1][sl], __ldg(&w1_p[i]))
                 + dot_bf16_4(s_y[2][sl], __ldg(&w2_p[i]))
                 + dot_bf16_4(s_y[3][sl], __ldg(&w3_p[i]));
        }
        if (base + step < n4) __syncthreads();
    }

    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, off, 16);
        ss  += __shfl_down_sync(0xffffffffu, ss,  off, 16);
    }

    if constexpr (SPLIT_K) {
        float * p_out = out_buf + (static_cast<size_t>(t) * S + s) * (n_out + 1);
        if (hw < n_out && sl == 0) p_out[hw]    = dot;
        if (hw == 0    && sl == 0) p_out[n_out] = ss;
    } else {
        if (hw == 0 && sl == 0) s_inv_rms = rsqrtf(ss / static_cast<float>(hc_dim) + rms_eps);
        __syncthreads();
        if (hw < n_out && sl == 0) out_buf[static_cast<size_t>(t) * n_out + hw] = dot * s_inv_rms;
    }
}

__global__ void raon_mhc_flat_mix_splitk_finalize_kernel(
    const float * __restrict__ part,
    float *       __restrict__ mixes,
    int S, int hc_dim, int n_out, float rms_eps) {

    const int t    = blockIdx.x;
    const int lane = threadIdx.x;
    const int stride = n_out + 1;
    const float * base = part + static_cast<size_t>(t) * S * stride;

    float sum = 0.0f;
    if (lane <= n_out) {
        for (int s = 0; s < S; ++s) {
            sum += base[s * stride + lane];
        }
    }
    const float tot_ss = __shfl_sync(0xffffffffu, sum, n_out);
    const float inv_rms = rsqrtf(tot_ss / static_cast<float>(hc_dim) + rms_eps);
    if (lane < n_out) {
        mixes[static_cast<size_t>(t) * n_out + lane] = sum * inv_rms;
    }
}

// ============================================================================
// 2. RDA Gate + Decoupled Write/Erase V-Scale + Beta
// ============================================================================

__global__ void raon_rda_gate_beta_vscale_kernel(
    float *       __restrict__ decay,
    const float * __restrict__ dt_bias,
    const float * __restrict__ ssm_a,
    float *       __restrict__ beta,
    const float * __restrict__ rda_delta,
    float *       __restrict__ v,
    float gate_lower_bound,
    int n_heads, int head_dim) {

    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim) return;

    const size_t th  = static_cast<size_t>(t) * n_heads + h;
    const size_t idx = th * head_dim + d;

    const float exp_a = -ssm_a[h];
    const float x     = decay[idx] + dt_bias[h * head_dim + d];
    decay[idx] = gate_lower_bound * raon_sigmoidf(exp_a * x);

    const float b_raw = beta[th];
    const float w_raw = rda_delta[th];
    const float ratio = expf(raon_logsigmoidf(b_raw + w_raw) - raon_logsigmoidf(b_raw));
    v[idx] *= ratio;

    if (d == 0) {
        beta[th] = raon_sigmoidf(b_raw);
    }
}

// ============================================================================
// 3. NoPE Gated MLA Prep Kernels (No RoPE! Uses exact MlaKv576Q8Pos layout!)
// ============================================================================

__global__ void raon_mla_split_q_kernel(
    const float * __restrict__ q_all,
    float *       __restrict__ q_nope,
    float *       __restrict__ q_pe,
    int n_heads, int qk_nope, int qk_pe) {

    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    const int qk_tot = qk_nope + qk_pe;
    const float * src = q_all  + (static_cast<size_t>(t) * n_heads + h) * qk_tot;
    float * dst_nope  = q_nope + (static_cast<size_t>(t) * n_heads + h) * qk_nope;
    float * dst_pe    = q_pe   + (static_cast<size_t>(t) * n_heads + h) * qk_pe;

    if (d < qk_nope) dst_nope[d] = src[d];
    if (d < qk_pe)   dst_pe[d]   = src[qk_nope + d];
}

template <bool KV_Q8_0>
__global__ void raon_mla_kv_norm_write_576_kernel(
    const float *   __restrict__ kv_all,
    const float *   __restrict__ norm_w,
    const int32_t * __restrict__ pos,
    void *          __restrict__ kv_cache,
    float rms_eps) {

    __shared__ float s_buf[576];
    __shared__ float s_warp[8];

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float * src = kv_all + static_cast<size_t>(t) * 576;

    const float v0 = src[tid];
    const float v1 = src[tid + 256];
    float ss = raon_warp_sum(v0 * v0 + v1 * v1);
    if (lane == 0) s_warp[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = raon_warp_sum((lane < 8) ? s_warp[lane] : 0.0f);
        if (lane == 0) s_warp[0] = rsqrtf(tot * (1.0f / 512.0f) + rms_eps);
    }
    __syncthreads();
    const float inv = s_warp[0];
    s_buf[tid]       = v0 * inv * norm_w[tid];
    s_buf[tid + 256] = v1 * inv * norm_w[tid + 256];
    if (tid < 64) s_buf[512 + tid] = src[512 + tid];
    __syncthreads();

    const int p = pos[t];
    if constexpr (KV_Q8_0) {
        RaonMlaKv576Q8Pos & dst = static_cast<RaonMlaKv576Q8Pos *>(kv_cache)[p];
        for (int b = warp; b < 18; b += 8) {
            const float v = s_buf[b * 32 + lane];
            const float amax = __shfl_sync(0xffffffffu, raon_warp_max(fabsf(v)), 0);
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

// ============================================================================
// 4. SiTU-GLU Activation & FP16 Helpers
// ============================================================================

__global__ void raon_situ_mul_kernel(
    const float * __restrict__ gate,
    const float * __restrict__ up,
    float *       __restrict__ out,
    int64_t n, float beta, float linear_beta) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = raon_situ_act(gate[i], up[i], beta, linear_beta);
}

__global__ void raon_f16_to_f32_vec_kernel(const __half * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t i2 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t i  = i2 * 2;
    if (i + 1 < n) {
        const __half2 h = reinterpret_cast<const __half2 *>(src)[i2];
        reinterpret_cast<float2 *>(dst)[i2] = __half22float2(h);
    } else if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

__global__ void raon_quantize_act_f16_kernel(
    const float * __restrict__ src, RaonActF16Block * __restrict__ dst, int total_blocks) {
    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= total_blocks) return;
    const int lane = threadIdx.x;
    const float v = src[static_cast<size_t>(b) * 32 + lane];
    const float amax = __shfl_sync(0xffffffffu, raon_warp_max(fabsf(v)), 0);
    const float inv  = (amax > 0.0f) ? (1.0f / amax) : 0.0f;
    const float vn   = v * inv;
    const float vn_hi = __shfl_down_sync(0xffffffffu, vn, 1);

    float hs = v;
    hs += __shfl_xor_sync(0xffffffffu, hs, 1);
    hs += __shfl_xor_sync(0xffffffffu, hs, 2);
    hs += __shfl_xor_sync(0xffffffffu, hs, 4);
    hs += __shfl_xor_sync(0xffffffffu, hs, 8);
    const float s0 = __shfl_sync(0xffffffffu, hs, 0);
    const float s1 = __shfl_sync(0xffffffffu, hs, 16);

    if ((lane & 1) == 0) {
        dst[b].x[lane >> 1] = __halves2half2(__float2half_rn(vn), __float2half_rn(vn_hi));
    }
    if (lane == 0) {
        dst[b].d     = amax;
        dst[b].s0    = s0;
        dst[b].s1    = s1;
        dst[b].s_all = s0 + s1;
    }
}

// ============================================================================
// 5. Fast Vectorized FP16 Multi-GEMV (for RDA ssm_beta + ssm_rda + ssm_f_a + ssm_f_b)
// ============================================================================

__global__ void raon_gemv_multi_f16_kernel(
    RaonMultiF16Desc desc,
    const float * __restrict__ X,
    int K) {

    extern __shared__ float4 s_x4[];
    const int t    = blockIdx.y;
    const int K4   = K >> 2;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const float4 * gx4 = reinterpret_cast<const float4 *>(X + static_cast<size_t>(t) * K);
    for (int i = tid; i < K4; i += nthr) s_x4[i] = gx4[i];
    __syncthreads();

    const int grow  = blockIdx.x * blockDim.y + threadIdx.y;
    const int tot_m = desc.row_end[desc.n_mat - 1];
    if (grow >= tot_m) return;

    int mi = 0;
    while (mi + 1 < desc.n_mat && grow >= desc.row_end[mi]) ++mi;
    const int row = grow - ((mi > 0) ? desc.row_end[mi - 1] : 0);
    const int M   = desc.m[mi];

    const int lane = threadIdx.x;
    const int K8 = K >> 3;
    const int4 * w8 = reinterpret_cast<const int4 *>(desc.w[mi] + static_cast<size_t>(row) * K);

    float acc = 0.0f;
    for (int i = lane; i < K8; i += 32) {
        const int4 wv  = __ldg(&w8[i]);
        const float4 xa = s_x4[i * 2 + 0];
        const float4 xb = s_x4[i * 2 + 1];
        const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *>(&wv.x));
        const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *>(&wv.y));
        const float2 f2 = __half22float2(*reinterpret_cast<const __half2 *>(&wv.z));
        const float2 f3 = __half22float2(*reinterpret_cast<const __half2 *>(&wv.w));
        acc += f0.x * xa.x + f0.y * xa.y + f1.x * xa.z + f1.y * xa.w
             + f2.x * xb.x + f2.y * xb.y + f3.x * xb.z + f3.y * xb.w;
    }
    acc = raon_warp_sum(acc);
    if (lane == 0) desc.y[mi][static_cast<size_t>(t) * M + row] = acc;
}

// Fast 4-rows/warp F16 GEMV for small-K (K=128 -> ssm_f_b: [128, 5120]): 8 lanes/row, LDG.128!
__global__ void raon_gemv_f16_k128_kernel(
    const __half * __restrict__ W,
    const float *  __restrict__ X,
    float *        __restrict__ Y,
    int M) {

    __shared__ float4 s_x4[32]; // 128 floats
    const int t    = blockIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    if (tid < 32) s_x4[tid] = reinterpret_cast<const float4 *>(X + static_cast<size_t>(t) * 128)[tid];
    __syncthreads();

    const int lane  = threadIdx.x;
    const int sub_r = lane >> 3; // 0..3 (4 rows/warp -> 32 rows/CTA)
    const int sub_l = lane & 7;  // 0..7 (each reads 16 halfs = 2 x int4!)
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 4 + sub_r;
    if (row >= M) return;

    const int4 * w8 = reinterpret_cast<const int4 *>(W + static_cast<size_t>(row) * 128);
    const int4 wa = __ldg(&w8[sub_l]);
    const int4 wb = __ldg(&w8[8 + sub_l]);
    const float4 xa0 = s_x4[sub_l * 2 + 0], xa1 = s_x4[sub_l * 2 + 1];
    const float4 xb0 = s_x4[16 + sub_l * 2 + 0], xb1 = s_x4[16 + sub_l * 2 + 1];

    const float2 a0 = __half22float2(*reinterpret_cast<const __half2 *>(&wa.x));
    const float2 a1 = __half22float2(*reinterpret_cast<const __half2 *>(&wa.y));
    const float2 a2 = __half22float2(*reinterpret_cast<const __half2 *>(&wa.z));
    const float2 a3 = __half22float2(*reinterpret_cast<const __half2 *>(&wa.w));
    const float2 b0 = __half22float2(*reinterpret_cast<const __half2 *>(&wb.x));
    const float2 b1 = __half22float2(*reinterpret_cast<const __half2 *>(&wb.y));
    const float2 b2 = __half22float2(*reinterpret_cast<const __half2 *>(&wb.z));
    const float2 b3 = __half22float2(*reinterpret_cast<const __half2 *>(&wb.w));

    float acc = a0.x * xa0.x + a0.y * xa0.y + a1.x * xa0.z + a1.y * xa0.w
              + a2.x * xa1.x + a2.y * xa1.y + a3.x * xa1.z + a3.y * xa1.w
              + b0.x * xb0.x + b0.y * xb0.y + b1.x * xb0.z + b1.y * xb0.w
              + b2.x * xb1.x + b2.y * xb1.y + b3.x * xb1.z + b3.y * xb1.w;

    acc += __shfl_down_sync(0xffffffffu, acc, 4, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 2, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 1, 8);
    if (sub_l == 0) Y[static_cast<size_t>(t) * M + row] = acc;
}

// ============================================================================
// 6. Fast Pascal IQ4_XS Dense Linear (sm_60 FP16 Prefill + R=2 Fused Multi Decode)
// ============================================================================

__global__ void raon_dequant_iq4_xs_f16_kernel(
    const BlockIQ4_XS_R * __restrict__ src,
    __half *              __restrict__ dst,
    int64_t n_blocks) {

    __shared__ float s_lut[16];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid < 16) s_lut[tid] = static_cast<float>(kvalues_iq4nl_d[tid]);
    __syncthreads();

    const int64_t bi = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (bi >= n_blocks) return;
    const int lane = threadIdx.x;
    const BlockIQ4_XS_R & b = src[bi];
    const int ib   = lane >> 2;
    const int l    = lane & 3;
    const float dl = raon_iq4_xs_dl(b, ib);

    const uint32_t q4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qs)[ib * 4 + l]);
    const __half2 lo0 = __floats2half2_rn(dl * s_lut[(q4 >>  0) & 0xFu], dl * s_lut[(q4 >>  8) & 0xFu]);
    const __half2 lo1 = __floats2half2_rn(dl * s_lut[(q4 >> 16) & 0xFu], dl * s_lut[(q4 >> 24) & 0xFu]);
    const __half2 hi0 = __floats2half2_rn(dl * s_lut[(q4 >>  4) & 0xFu], dl * s_lut[(q4 >> 12) & 0xFu]);
    const __half2 hi1 = __floats2half2_rn(dl * s_lut[(q4 >> 20) & 0xFu], dl * s_lut[(q4 >> 28) & 0xFu]);

    uint2 plo, phi;
    plo.x = *reinterpret_cast<const uint32_t *>(&lo0);
    plo.y = *reinterpret_cast<const uint32_t *>(&lo1);
    phi.x = *reinterpret_cast<const uint32_t *>(&hi0);
    phi.y = *reinterpret_cast<const uint32_t *>(&hi1);

    auto * out4 = reinterpret_cast<uint2 *>(dst + bi * 256 + ib * 32);
    out4[l]     = plo;
    out4[4 + l] = phi;
}

__device__ __forceinline__ float raon_dot_iq4_xs_block_h2(
    const BlockIQ4_XS_R & b,
    const RaonActF16Block * __restrict__ x_blks,
    const __half * __restrict__ s_lut_h,
    int lane) {

    const int ib   = lane >> 2;
    const int l    = lane & 3;
    const float dl = raon_iq4_xs_dl(b, ib);

    const uint32_t q4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qs)[ib * 4 + l]);
    const __half2 * xh = x_blks[ib].x;

    const __half2 w0 = raon_lut_h2(s_lut_h, (q4 >>  0) & 0xFu, (q4 >>  8) & 0xFu);
    const __half2 w1 = raon_lut_h2(s_lut_h, (q4 >> 16) & 0xFu, (q4 >> 24) & 0xFu);
    const __half2 w2 = raon_lut_h2(s_lut_h, (q4 >>  4) & 0xFu, (q4 >> 12) & 0xFu);
    const __half2 w3 = raon_lut_h2(s_lut_h, (q4 >> 20) & 0xFu, (q4 >> 28) & 0xFu);

    __half2 acc = raon_hmul2(w0, xh[l * 2 + 0]);
    acc = raon_hfma2(w1, xh[l * 2 + 1], acc);
    acc = raon_hfma2(w2, xh[8 + l * 2 + 0], acc);
    acc = raon_hfma2(w3, xh[8 + l * 2 + 1], acc);
    const float2 f = __half22float2(acc);
    return (dl * x_blks[ib].d) * (f.x + f.y);
}

__device__ __forceinline__ float raon_dot_iq4_xs_block_dp4a(
    const BlockIQ4_XS_R & b,
    const BlockQ8_1_Act * __restrict__ x_blks,
    const int8_t * __restrict__ s_lut_i8,
    int lane) {

    const int ib   = lane >> 2;
    const int l    = lane & 3;
    const float dl = raon_iq4_xs_dl(b, ib);

    const uint32_t q4 = __ldg(&reinterpret_cast<const uint32_t *>(b.qs)[ib * 4 + l]);
    const int w0 = raon_pack_iq4_lo4(s_lut_i8, q4);
    const int w1 = raon_pack_iq4_hi4(s_lut_i8, q4);
    const int * x32 = reinterpret_cast<const int *>(x_blks[ib].qs);
    const int s = dp4a_s32(w0, x32[l], dp4a_s32(w1, x32[4 + l], 0));
    return (dl * x_blks[ib].d) * static_cast<float>(s);
}

// 2 rows/warp fused multi-IQ4_XS GEMV (sm_60 half2): halves s_x loads & doubles ILP!
__global__ void raon_gemv_multi_iq4_xs_h2_kernel(
    RaonMultiIQ4Desc desc,
    const RaonActF16Block * __restrict__ X,
    int K) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<RaonActF16Block *>(smem_raw);
    __shared__ __half s_lut_h[16];

    const int t        = blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;

    if (tid < 16) s_lut_h[tid] = __float2half_rn(static_cast<float>(kvalues_iq4nl_d[tid]));
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(RaonActF16Block))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(X + static_cast<size_t>(t) * K_blocks);
    for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_x)[i] = sx_in[i];
    __syncthreads();

    const int tot_m = desc.row_end[desc.n_mat - 1];
    const int grow0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    const int grow1 = grow0 + 1;
    if (grow0 >= tot_m) return;

    int mi0 = 0;
    while (mi0 + 1 < desc.n_mat && grow0 >= desc.row_end[mi0]) ++mi0;
    const int row0 = grow0 - ((mi0 > 0) ? desc.row_end[mi0 - 1] : 0);
    const int M0   = desc.m[mi0];
    const BlockIQ4_XS_R * w0 = desc.w[mi0] + static_cast<size_t>(row0) * nb;

    const bool has1 = (grow1 < tot_m);
    int mi1 = mi0;
    int row1 = row0 + 1;
    if (has1 && grow1 >= desc.row_end[mi1]) { ++mi1; row1 = 0; }
    const BlockIQ4_XS_R * w1 = has1 ? (desc.w[mi1] + static_cast<size_t>(row1) * nb) : nullptr;

    const int lane = threadIdx.x;
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll 2
    for (int bi = 0; bi < nb; ++bi) {
        const RaonActF16Block * xb = s_x + bi * 8;
        acc0 += raon_dot_iq4_xs_block_h2(w0[bi], xb, s_lut_h, lane);
        if (has1) acc1 += raon_dot_iq4_xs_block_h2(w1[bi], xb, s_lut_h, lane);
    }
    acc0 = raon_warp_sum(acc0);
    if (has1) acc1 = raon_warp_sum(acc1);
    if (lane == 0) {
        desc.y[mi0][static_cast<size_t>(t) * M0 + row0] = acc0;
        if (has1) desc.y[mi1][static_cast<size_t>(t) * desc.m[mi1] + row1] = acc1;
    }
}

__global__ void raon_gemv_multi_iq4_xs_dp4a_kernel(
    RaonMultiIQ4Desc desc,
    const BlockQ8_1_Act * __restrict__ X,
    int K) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<BlockQ8_1_Act *>(smem_raw);
    __shared__ int8_t s_lut_i8[16];

    const int t        = blockIdx.y;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
    const int nthr     = blockDim.y * 32;

    if (tid < 16) s_lut_i8[tid] = kvalues_iq4nl_d[tid];
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(X + static_cast<size_t>(t) * K_blocks);
    for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_x)[i] = sx_in[i];
    __syncthreads();

    const int tot_m = desc.row_end[desc.n_mat - 1];
    const int grow0 = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
    const int grow1 = grow0 + 1;
    if (grow0 >= tot_m) return;

    int mi0 = 0;
    while (mi0 + 1 < desc.n_mat && grow0 >= desc.row_end[mi0]) ++mi0;
    const int row0 = grow0 - ((mi0 > 0) ? desc.row_end[mi0 - 1] : 0);
    const int M0   = desc.m[mi0];
    const BlockIQ4_XS_R * w0 = desc.w[mi0] + static_cast<size_t>(row0) * nb;

    const bool has1 = (grow1 < tot_m);
    int mi1 = mi0;
    int row1 = row0 + 1;
    if (has1 && grow1 >= desc.row_end[mi1]) { ++mi1; row1 = 0; }
    const BlockIQ4_XS_R * w1 = has1 ? (desc.w[mi1] + static_cast<size_t>(row1) * nb) : nullptr;

    const int lane = threadIdx.x;
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll 2
    for (int bi = 0; bi < nb; ++bi) {
        const BlockQ8_1_Act * xb = s_x + bi * 8;
        acc0 += raon_dot_iq4_xs_block_dp4a(w0[bi], xb, s_lut_i8, lane);
        if (has1) acc1 += raon_dot_iq4_xs_block_dp4a(w1[bi], xb, s_lut_i8, lane);
    }
    acc0 = raon_warp_sum(acc0);
    if (has1) acc1 = raon_warp_sum(acc1);
    if (lane == 0) {
        desc.y[mi0][static_cast<size_t>(t) * M0 + row0] = acc0;
        if (has1) desc.y[mi1][static_cast<size_t>(t) * desc.m[mi1] + row1] = acc1;
    }
}

// ============================================================================
// 7. MoE Grouping, Worklist, and Slot Reduction Helpers
// ============================================================================

__global__ void raon_group_moe_tokens_kernel(
    const int32_t * __restrict__ top_ids,
    int32_t *       __restrict__ exp_offsets,
    int32_t *       __restrict__ sorted_pairs,
    int total_pairs, int n_expert, int k_used) {

    extern __shared__ int32_t s_cnt[];
    const int tid = threadIdx.x;
    for (int e = tid; e < n_expert; e += blockDim.x) s_cnt[e] = 0;
    __syncthreads();
    for (int i = tid; i < total_pairs; i += blockDim.x) {
        const int eid = top_ids[i];
        if (eid >= 0 && eid < n_expert) atomicAdd(&s_cnt[eid], 1);
    }
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
        for (int e = 0; e < n_expert; ++e) {
            const int c = s_cnt[e];
            exp_offsets[e] = sum;
            s_cnt[e] = sum;
            sum += c;
        }
        exp_offsets[n_expert] = sum;
    }
    __syncthreads();
    for (int i = tid; i < total_pairs; i += blockDim.x) {
        const int eid = top_ids[i];
        if (eid >= 0 && eid < n_expert) {
            const int dst = atomicAdd(&s_cnt[eid], 1);
            sorted_pairs[dst] = ((i / k_used) << 4) | (i % k_used);
        }
    }
}

__global__ void raon_build_tile_worklist_kernel(
    const int32_t * __restrict__ exp_offsets,
    RaonTileDesc *  __restrict__ tiles,
    int32_t *       __restrict__ n_tiles_out,
    int n_expert, int BT) {
    if (threadIdx.x != 0) return;
    int nt = 0;
    for (int e = 0; e < n_expert; ++e) {
        const int s  = exp_offsets[e];
        const int en = exp_offsets[e + 1];
        for (int p = s; p < en; p += BT) {
            const int cnt = (p + BT < en) ? BT : (en - p);
            tiles[nt++] = {e, p, cnt};
        }
    }
    *n_tiles_out = nt;
}

__global__ void raon_reduce_slots_add_kernel(
    const float * __restrict__ slots,
    float *       __restrict__ out,
    int K, int n_slots, bool add_to_existing) {
    const int K4 = K >> 2;
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
    const int t  = blockIdx.y;
    if (i >= K4) return;
    const float4 * base = reinterpret_cast<const float4 *>(slots + static_cast<size_t>(t) * n_slots * K) + i;
    float4 sum = add_to_existing
        ? reinterpret_cast<const float4 *>(out + static_cast<size_t>(t) * K)[i]
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < n_slots; ++s) {
        const float4 v = base[static_cast<size_t>(s) * K4];
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    reinterpret_cast<float4 *>(out + static_cast<size_t>(t) * K)[i] = sum;
}

// ============================================================================
// 8. Raon SiTU-GLU MoE DECODE Kernels (T < 16)
//    - Gate+Up: 32 rows/CTA (4 rows/warp) with DIRECT in-smem ActF16 / Q8_1 quant!
//      Eliminates act float DRAM roundtrip AND separate quantize kernel launch!
//    - Down:    IQ4_NL SoA & Q5_1 Split-Slot (sm_60 half2 & sm_61+ dp4a)
// ============================================================================

__global__ void raon_moe_gate_up_iq4_xs_h2_qact_dec_kernel(
    const BlockIQ4_XS_R *   __restrict__ g_exps,
    const BlockIQ4_XS_R *   __restrict__ u_exps,
    const RaonActF16Block * __restrict__ x_f16,
    const int32_t *         __restrict__ top_ids,
    RaonActF16Block *       __restrict__ a_f16, // [T, k_used, ff/32]
    int ff, int K, int k_used, float beta, float linear_beta) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<RaonActF16Block *>(smem_raw);
    __shared__ __half s_lut_h[16];
    __shared__ float  s_act[32];

    const int blk32    = blockIdx.x; // 0..(ff/32 - 1)
    const int slot     = blockIdx.y;
    const int t        = blockIdx.z;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int lane     = threadIdx.x;
    const int warp     = threadIdx.y;
    const int tid      = warp * 32 + lane;
    const int nthr     = blockDim.y * 32;

    if (tid < 16) s_lut_h[tid] = __float2half_rn(static_cast<float>(kvalues_iq4nl_d[tid]));
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(RaonActF16Block))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(x_f16 + static_cast<size_t>(t) * K_blocks);
    for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_x)[i] = sx_in[i];
    __syncthreads();

    const int eid = top_ids[static_cast<size_t>(t) * k_used + slot];
    const int base_row = blk32 * 32 + warp * 4;

    #pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int row = base_row + rr;
        const BlockIQ4_XS_R * g_row = g_exps + (static_cast<size_t>(eid) * ff + row) * nb;
        const BlockIQ4_XS_R * u_row = u_exps + (static_cast<size_t>(eid) * ff + row) * nb;
        float ag = 0.0f, au = 0.0f;
        #pragma unroll 2
        for (int bi = 0; bi < nb; ++bi) {
            const RaonActF16Block * xb = s_x + bi * 8;
            ag += raon_dot_iq4_xs_block_h2(g_row[bi], xb, s_lut_h, lane);
            au += raon_dot_iq4_xs_block_h2(u_row[bi], xb, s_lut_h, lane);
        }
        ag = raon_warp_sum(ag);
        au = raon_warp_sum(au);
        if (lane == 0) s_act[warp * 4 + rr] = raon_situ_act(ag, au, beta, linear_beta);
    }
    __syncthreads();

    // Warp 0 quantizes all 32 activations directly into RaonActF16Block!
    if (warp == 0) {
        const float v     = s_act[lane];
        const float amax  = __shfl_sync(0xffffffffu, raon_warp_max(fabsf(v)), 0);
        const float inv   = (amax > 0.0f) ? (1.0f / amax) : 0.0f;
        const float vn    = v * inv;
        const float vn_hi = __shfl_down_sync(0xffffffffu, vn, 1);

        float hs = v;
        hs += __shfl_xor_sync(0xffffffffu, hs, 1);
        hs += __shfl_xor_sync(0xffffffffu, hs, 2);
        hs += __shfl_xor_sync(0xffffffffu, hs, 4);
        hs += __shfl_xor_sync(0xffffffffu, hs, 8);
        const float s0 = __shfl_sync(0xffffffffu, hs, 0);
        const float s1 = __shfl_sync(0xffffffffu, hs, 16);

        const int ff_blks = ff >> 5;
        RaonActF16Block & dst = a_f16[(static_cast<size_t>(t) * k_used + slot) * ff_blks + blk32];
        if ((lane & 1) == 0) {
            dst.x[lane >> 1] = __halves2half2(__float2half_rn(vn), __float2half_rn(vn_hi));
        }
        if (lane == 0) {
            dst.d     = amax;
            dst.s0    = s0;
            dst.s1    = s1;
            dst.s_all = s0 + s1;
        }
    }
}

__global__ void raon_moe_gate_up_iq4_xs_dp4a_qact_dec_kernel(
    const BlockIQ4_XS_R * __restrict__ g_exps,
    const BlockIQ4_XS_R * __restrict__ u_exps,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ top_ids,
    BlockQ8_1_Act *       __restrict__ a_q8, // [T, k_used, ff/32]
    int ff, int K, int k_used, float beta, float linear_beta) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_x = reinterpret_cast<BlockQ8_1_Act *>(smem_raw);
    __shared__ int8_t s_lut_i8[16];
    __shared__ float  s_act[32];

    const int blk32    = blockIdx.x;
    const int slot     = blockIdx.y;
    const int t        = blockIdx.z;
    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const int lane     = threadIdx.x;
    const int warp     = threadIdx.y;
    const int tid      = warp * 32 + lane;
    const int nthr     = blockDim.y * 32;

    if (tid < 16) s_lut_i8[tid] = kvalues_iq4nl_d[tid];
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in = reinterpret_cast<const int4 *>(x_q8 + static_cast<size_t>(t) * K_blocks);
    for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_x)[i] = sx_in[i];
    __syncthreads();

    const int eid = top_ids[static_cast<size_t>(t) * k_used + slot];
    const int base_row = blk32 * 32 + warp * 4;

    #pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int row = base_row + rr;
        const BlockIQ4_XS_R * g_row = g_exps + (static_cast<size_t>(eid) * ff + row) * nb;
        const BlockIQ4_XS_R * u_row = u_exps + (static_cast<size_t>(eid) * ff + row) * nb;
        float ag = 0.0f, au = 0.0f;
        #pragma unroll 2
        for (int bi = 0; bi < nb; ++bi) {
            const BlockQ8_1_Act * xb = s_x + bi * 8;
            ag += raon_dot_iq4_xs_block_dp4a(g_row[bi], xb, s_lut_i8, lane);
            au += raon_dot_iq4_xs_block_dp4a(u_row[bi], xb, s_lut_i8, lane);
        }
        ag = raon_warp_sum(ag);
        au = raon_warp_sum(au);
        if (lane == 0) s_act[warp * 4 + rr] = raon_situ_act(ag, au, beta, linear_beta);
    }
    __syncthreads();

    if (warp == 0) {
        const float v     = s_act[lane];
        const float amax  = __shfl_sync(0xffffffffu, raon_warp_max(fabsf(v)), 0);
        const float d     = amax / 127.0f;
        const float inv_d = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        const int   q     = __float2int_rn(v * inv_d);

        int hsum = q;
        hsum += __shfl_xor_sync(0xffffffffu, hsum, 1);
        hsum += __shfl_xor_sync(0xffffffffu, hsum, 2);
        hsum += __shfl_xor_sync(0xffffffffu, hsum, 4);
        hsum += __shfl_xor_sync(0xffffffffu, hsum, 8);
        const int s0 = __shfl_sync(0xffffffffu, hsum, 0);
        const int s1 = __shfl_sync(0xffffffffu, hsum, 16);

        const int ff_blks = ff >> 5;
        BlockQ8_1_Act & dst = a_q8[(static_cast<size_t>(t) * k_used + slot) * ff_blks + blk32];
        dst.qs[lane] = static_cast<int8_t>(q);
        if (lane == 0) {
            dst.d     = d;
            dst.s0    = s0;
            dst.s1    = s1;
            dst.s_all = d * static_cast<float>(s0 + s1);
        }
    }
}

__global__ void raon_moe_down_iq4_nl_soa_h2_split_dec_kernel(
    const int4 *            __restrict__ dn_qs,
    const uint16_t *        __restrict__ dn_d,
    const RaonActF16Block * __restrict__ a_f16,
    const int32_t *         __restrict__ top_ids,
    const float *           __restrict__ top_w,
    float *                 __restrict__ slots,
    int K, int ff, int k_used) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_a = reinterpret_cast<RaonActF16Block *>(smem_raw);
    __shared__ __half s_lut_h[16];

    const int slot    = blockIdx.y;
    const int t       = blockIdx.z;
    const int ff_blks = ff >> 5;
    const int tid     = threadIdx.y * 32 + threadIdx.x;
    const int nthr    = blockDim.y * 32;

    if (tid < 16) s_lut_h[tid] = __float2half_rn(static_cast<float>(kvalues_iq4nl_d[tid]));
    const int n_i4 = (ff_blks * static_cast<int>(sizeof(RaonActF16Block))) / 16;
    const int4 * sa_in = reinterpret_cast<const int4 *>(a_f16 + (static_cast<size_t>(t) * k_used + slot) * ff_blks);
    for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_a)[i] = sa_in[i];
    __syncthreads();

    const int lane  = threadIdx.x;
    const int sub_r = lane >> 4;
    const int sub_l = lane & 15;
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 2 + sub_r;
    if (row >= K) return;

    const size_t slot_idx = static_cast<size_t>(t) * k_used + slot;
    const int   eid = top_ids[slot_idx];
    const float ew  = top_w[slot_idx];
    const size_t roff = (static_cast<size_t>(eid) * K + row) * ff_blks;
    const int4 *     w_qs = dn_qs + roff;
    const uint16_t * w_d  = dn_d  + roff;

    float acc = 0.0f;
    for (int b = sub_l; b < ff_blks; b += 16) {
        const int4 qv = __ldg(&w_qs[b]);
        const float d = raon_half_bits_to_f32(__ldg(&w_d[b]));
        const uint32_t qw[4] = {
            static_cast<uint32_t>(qv.x), static_cast<uint32_t>(qv.y),
            static_cast<uint32_t>(qv.z), static_cast<uint32_t>(qv.w)
        };
        const __half2 * xh = s_a[b].x;
        __half2 hsum = __floats2half2_rn(0.0f, 0.0f);
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t q4 = qw[g];
            hsum = raon_hfma2(raon_lut_h2(s_lut_h, (q4 >>  0) & 0xFu, (q4 >>  8) & 0xFu), xh[g * 2 + 0], hsum);
            hsum = raon_hfma2(raon_lut_h2(s_lut_h, (q4 >> 16) & 0xFu, (q4 >> 24) & 0xFu), xh[g * 2 + 1], hsum);
            hsum = raon_hfma2(raon_lut_h2(s_lut_h, (q4 >>  4) & 0xFu, (q4 >> 12) & 0xFu), xh[8 + g * 2 + 0], hsum);
            hsum = raon_hfma2(raon_lut_h2(s_lut_h, (q4 >> 20) & 0xFu, (q4 >> 28) & 0xFu), xh[8 + g * 2 + 1], hsum);
        }
        const float2 f = __half22float2(hsum);
        acc += (d * s_a[b].d) * (f.x + f.y);
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 8, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 4, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 2, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 1, 16);
    if (sub_l == 0) slots[slot_idx * K + row] = ew * acc;
}

__global__ void raon_moe_down_iq4_nl_soa_dp4a_split_dec_kernel(
    const int4 *          __restrict__ dn_qs,
    const uint16_t *      __restrict__ dn_d,
    const BlockQ8_1_Act * __restrict__ a_q8,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ slots,
    int K, int ff, int k_used) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_a = reinterpret_cast<BlockQ8_1_Act *>(smem_raw);
    __shared__ int8_t s_lut_i8[16];

    const int slot    = blockIdx.y;
    const int t       = blockIdx.z;
    const int ff_blks = ff >> 5;
    const int tid     = threadIdx.y * 32 + threadIdx.x;
    const int nthr    = blockDim.y * 32;

    if (tid < 16) s_lut_i8[tid] = kvalues_iq4nl_d[tid];
    const int n_i4 = (ff_blks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sa_in = reinterpret_cast<const int4 *>(a_q8 + (static_cast<size_t>(t) * k_used + slot) * ff_blks);
    for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_a)[i] = sa_in[i];
    __syncthreads();

    const int lane  = threadIdx.x;
    const int sub_r = lane >> 4;
    const int sub_l = lane & 15;
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 2 + sub_r;
    if (row >= K) return;

    const size_t slot_idx = static_cast<size_t>(t) * k_used + slot;
    const int   eid = top_ids[slot_idx];
    const float ew  = top_w[slot_idx];
    const size_t roff = (static_cast<size_t>(eid) * K + row) * ff_blks;
    const int4 *     w_qs = dn_qs + roff;
    const uint16_t * w_d  = dn_d  + roff;

    float acc = 0.0f;
    for (int b = sub_l; b < ff_blks; b += 16) {
        const int4 qv = __ldg(&w_qs[b]);
        const float d = raon_half_bits_to_f32(__ldg(&w_d[b]));
        const uint32_t qw[4] = {
            static_cast<uint32_t>(qv.x), static_cast<uint32_t>(qv.y),
            static_cast<uint32_t>(qv.z), static_cast<uint32_t>(qv.w)
        };
        const int * x32 = reinterpret_cast<const int *>(s_a[b].qs);
        int sumi = 0;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            sumi = dp4a_s32(raon_pack_iq4_lo4(s_lut_i8, qw[g]), x32[g],     sumi);
            sumi = dp4a_s32(raon_pack_iq4_hi4(s_lut_i8, qw[g]), x32[4 + g], sumi);
        }
        acc += (d * s_a[b].d) * static_cast<float>(sumi);
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 8, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 4, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 2, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 1, 16);
    if (sub_l == 0) slots[slot_idx * K + row] = ew * acc;
}

__global__ void raon_moe_down_q5_1_h2_split_dec_kernel(
    const BlockQ5_1_R *     __restrict__ dn_exps,
    const RaonActF16Block * __restrict__ a_f16,
    const int32_t *         __restrict__ top_ids,
    const float *           __restrict__ top_w,
    float *                 __restrict__ slots,
    int K, int ff, int k_used) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_a = reinterpret_cast<RaonActF16Block *>(smem_raw);

    const int slot    = blockIdx.y;
    const int t       = blockIdx.z;
    const int ff_blks = ff >> 5;
    const int tid     = threadIdx.y * 32 + threadIdx.x;
    const int nthr    = blockDim.y * 32;

    const int n_i4 = (ff_blks * static_cast<int>(sizeof(RaonActF16Block))) / 16;
    const int4 * sa_in = reinterpret_cast<const int4 *>(a_f16 + (static_cast<size_t>(t) * k_used + slot) * ff_blks);
    for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_a)[i] = sa_in[i];
    __syncthreads();

    const int lane  = threadIdx.x;
    const int sub_r = lane >> 4;
    const int sub_l = lane & 15;
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 2 + sub_r;
    if (row >= K) return;

    const size_t slot_idx = static_cast<size_t>(t) * k_used + slot;
    const int   eid = top_ids[slot_idx];
    const float ew  = top_w[slot_idx];
    const BlockQ5_1_R * w_row = dn_exps + (static_cast<size_t>(eid) * K + row) * ff_blks;
    const __half2 neg1024 = __floats2half2_rn(-1024.0f, -1024.0f);

    float acc = 0.0f;
    for (int b = sub_l; b < ff_blks; b += 16) {
        const BlockQ5_1_R & blk = w_row[b];
        const uint2 hdr = __ldg(reinterpret_cast<const uint2 *>(&blk));
        const float d   = raon_half_bits_to_f32(static_cast<uint16_t>(hdr.x & 0xFFFFu));
        const float m   = raon_half_bits_to_f32(static_cast<uint16_t>(hdr.x >> 16));
        const uint32_t qh = hdr.y;
        const uint32_t * qs32 = reinterpret_cast<const uint32_t *>(blk.qs);
        const __half2 * xh = s_a[b].x;
        __half2 hsum = __floats2half2_rn(0.0f, 0.0f);
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t q4 = __ldg(&qs32[g]);
            const uint32_t hlo = (qh >> (g * 4)) & 0xFu;
            const uint32_t hhi = (qh >> (16 + g * 4)) & 0xFu;
            const uint32_t mlo = ((hlo * 0x00204081u) & 0x01010101u) << 4;
            const uint32_t mhi = ((hhi * 0x00204081u) & 0x01010101u) << 4;
            const uint32_t u0  = (q4 & 0x0F0F0F0Fu) | mlo;
            const uint32_t u1  = ((q4 >> 4) & 0x0F0F0F0Fu) | mhi;
            hsum = raon_hfma2(raon_hadd2(raon_u8x2_to_h2_lo(u0), neg1024), xh[g * 2 + 0], hsum);
            hsum = raon_hfma2(raon_hadd2(raon_u8x2_to_h2_hi(u0), neg1024), xh[g * 2 + 1], hsum);
            hsum = raon_hfma2(raon_hadd2(raon_u8x2_to_h2_lo(u1), neg1024), xh[8 + g * 2 + 0], hsum);
            hsum = raon_hfma2(raon_hadd2(raon_u8x2_to_h2_hi(u1), neg1024), xh[8 + g * 2 + 1], hsum);
        }
        const float2 f = __half22float2(hsum);
        acc += (d * s_a[b].d) * (f.x + f.y) + m * s_a[b].s_all;
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 8, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 4, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 2, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 1, 16);
    if (sub_l == 0) slots[slot_idx * K + row] = ew * acc;
}

__global__ void raon_moe_down_q5_1_dp4a_split_dec_kernel(
    const BlockQ5_1_R *   __restrict__ dn_exps,
    const BlockQ8_1_Act * __restrict__ a_q8,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ slots,
    int K, int ff, int k_used) {

    extern __shared__ uint8_t smem_raw[];
    auto * s_a = reinterpret_cast<BlockQ8_1_Act *>(smem_raw);

    const int slot    = blockIdx.y;
    const int t       = blockIdx.z;
    const int ff_blks = ff >> 5;
    const int tid     = threadIdx.y * 32 + threadIdx.x;
    const int nthr    = blockDim.y * 32;

    const int n_i4 = (ff_blks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sa_in = reinterpret_cast<const int4 *>(a_q8 + (static_cast<size_t>(t) * k_used + slot) * ff_blks);
    for (int i = tid; i < n_i4; i += nthr) reinterpret_cast<int4 *>(s_a)[i] = sa_in[i];
    __syncthreads();

    const int lane  = threadIdx.x;
    const int sub_r = lane >> 4;
    const int sub_l = lane & 15;
    const int row   = (blockIdx.x * blockDim.y + threadIdx.y) * 2 + sub_r;
    if (row >= K) return;

    const size_t slot_idx = static_cast<size_t>(t) * k_used + slot;
    const int   eid = top_ids[slot_idx];
    const float ew  = top_w[slot_idx];
    const BlockQ5_1_R * w_row = dn_exps + (static_cast<size_t>(eid) * K + row) * ff_blks;

    float acc = 0.0f;
    for (int b = sub_l; b < ff_blks; b += 16) {
        const BlockQ5_1_R & blk = w_row[b];
        const uint2 hdr = __ldg(reinterpret_cast<const uint2 *>(&blk));
        const float d   = raon_half_bits_to_f32(static_cast<uint16_t>(hdr.x & 0xFFFFu));
        const float m   = raon_half_bits_to_f32(static_cast<uint16_t>(hdr.x >> 16));
        const uint32_t qh = hdr.y;
        const uint32_t * qs32 = reinterpret_cast<const uint32_t *>(blk.qs);
        const int * x32 = reinterpret_cast<const int *>(s_a[b].qs);
        int sumi = 0;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t q4  = __ldg(&qs32[g]);
            const uint32_t hlo = (qh >> (g * 4)) & 0xFu;
            const uint32_t hhi = (qh >> (16 + g * 4)) & 0xFu;
            const uint32_t mlo = ((hlo * 0x00204081u) & 0x01010101u) << 4;
            const uint32_t mhi = ((hhi * 0x00204081u) & 0x01010101u) << 4;
            const int u0 = static_cast<int>((q4 & 0x0F0F0F0Fu) | mlo);
            const int u1 = static_cast<int>(((q4 >> 4) & 0x0F0F0F0Fu) | mhi);
            sumi = dp4a_s32(u0, x32[g],     sumi);
            sumi = dp4a_s32(u1, x32[4 + g], sumi);
        }
        acc += (d * s_a[b].d) * static_cast<float>(sumi) + m * s_a[b].s_all;
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 8, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 4, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 2, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 1, 16);
    if (sub_l == 0) slots[slot_idx * K + row] = ew * acc;
}

// ============================================================================
// 9. Raon SiTU-GLU MoE GROUPED PREFILL Kernels (T >= 16)
//    - sm_60 (P100): Pure-__half2 accumulators + per-row X_rsc + pre-scaled weights!
//      Eliminates per-token FP32 conversions & register spilling -> 2x+ faster MoE prefill!
//    - sm_61+ (dp4a): Bank-conflict-free BT=32 (T>=128) / BT=16 (T<128)
// ============================================================================

__global__ void raon_prep_row_h2_kernel(
    const float * __restrict__ x,      // [n_rows, K]
    __half2 *     __restrict__ x_h2,   // [n_rows, K / 2]
    float *       __restrict__ row_sc, // [n_rows]
    int K) {

    const int r    = blockIdx.x;
    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float * xr = x + static_cast<size_t>(r) * K;
    float amax = 0.0f;
    for (int i = tid; i < K; i += 256) amax = fmaxf(amax, fabsf(xr[i]));
    amax = raon_warp_max(amax);
    __shared__ float s_m[8];
    if (lane == 0) s_m[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? s_m[lane] : 0.0f;
        v = raon_warp_max(v);
        if (lane == 0) s_m[0] = v;
    }
    __syncthreads();
    amax = s_m[0];
    if (tid == 0) row_sc[r] = amax * (1.0f / 64.0f);

    const float inv = (amax > 0.0f) ? (4.0f / amax) : 0.0f;
    const int K2 = K >> 1;
    const auto * xr2 = reinterpret_cast<const float2 *>(xr);
    __half2 *    yh2 = x_h2 + static_cast<size_t>(r) * K2;
    for (int i = tid; i < K2; i += 256) {
        const float2 v = xr2[i];
        yh2[i] = __float22half2_rn(make_float2(v.x * inv, v.y * inv));
    }
}

// Pure-HFMA2 sm_60 Grouped Gate+Up IQ4_XS (BT=16, BT_TH=8, 8.2KB smem -> high occupancy, ZERO spill!)
__global__ __launch_bounds__(256, 4) void raon_moe_gate_up_grouped_iq4_xs_sm60_pureh2_kernel(
    const BlockIQ4_XS_R * __restrict__ g_exps,
    const BlockIQ4_XS_R * __restrict__ u_exps,
    const int4 *          __restrict__ X_h2,   // [T, K / 8] int4
    const float *         __restrict__ X_rsc,  // [T]
    const RaonTileDesc *  __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ act,    // [T, k_used, ff]
    int ff, int K, int k_used, float beta, float linear_beta) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const RaonTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT    = 16;
    constexpr int BT_TH = 8;
    __shared__ int4    s_xh2[BT * 32]; // 8 KB (bank-conflict-free: tj*32 + c*8 + bk)
    __shared__ int32_t s_pairs[BT];
    __shared__ float   s_rsc[BT];
    __shared__ __half  s_lut_h[16];

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int row    = blockIdx.x * 16 + (ltid >> 3);
    const int ib     = ltid & 7;
    const int t_base = t_half * BT_TH;

    if (tid < 16) s_lut_h[tid] = __float2half_rn(static_cast<float>(kvalues_iq4nl_d[tid]));
    if (tid < BT) {
        if (tid < n_tile) {
            const int pair = sorted_pairs[start + tid];
            s_pairs[tid] = pair;
            s_rsc[tid]   = __ldg(&X_rsc[pair >> 4]);
        } else {
            s_pairs[tid] = 0;
            s_rsc[tid]   = 0.0f;
        }
    }
    __syncthreads();

    const int nb   = K >> 8;
    const int K_i4 = K >> 3;
    const BlockIQ4_XS_R * g_row = g_exps + (static_cast<size_t>(eid) * ff + row) * nb;
    const BlockIQ4_XS_R * u_row = u_exps + (static_cast<size_t>(eid) * ff + row) * nb;

    const __half2 hzero = __float2half2_rn(0.0f);
    __half2 ag[BT_TH], au[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { ag[it] = hzero; au[it] = hzero; }

    for (int bi = 0; bi < nb; ++bi) {
        for (int idx = tid; idx < BT * 32; idx += 256) {
            const int c   = idx & 3;
            const int bk  = (idx >> 2) & 7;
            const int tj  = idx >> 5;
            const int dst = tj * 32 + c * 8 + bk;
            if (tj < n_tile) {
                const int tok = s_pairs[tj] >> 4;
                s_xh2[dst] = __ldg(&X_h2[static_cast<size_t>(tok) * K_i4 + bi * 32 + bk * 4 + c]);
            } else {
                s_xh2[dst] = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        if (row < ff) {
            const BlockIQ4_XS_R & gb = g_row[bi];
            const __half2 gsc2 = __float2half2_rn(raon_iq4_xs_dl(gb, ib) * 16.0f);
            const uint32_t * gq32 = reinterpret_cast<const uint32_t *>(gb.qs + ib * 16);
            const uint32_t g0 = __ldg(&gq32[0]), g1 = __ldg(&gq32[1]), g2 = __ldg(&gq32[2]), g3 = __ldg(&gq32[3]);

            const BlockIQ4_XS_R & ub = u_row[bi];
            const __half2 usc2 = __float2half2_rn(raon_iq4_xs_dl(ub, ib) * 16.0f);
            const uint32_t * uq32 = reinterpret_cast<const uint32_t *>(ub.qs + ib * 16);
            const uint32_t u0 = __ldg(&uq32[0]), u1 = __ldg(&uq32[1]), u2 = __ldg(&uq32[2]), u3 = __ldg(&uq32[3]);

            // Low nibbles (elems 0..15)
            {
                const __half2 gw0 = raon_hmul2(raon_lut_h2(s_lut_h, (g0 >>  0) & 0xFu, (g0 >>  8) & 0xFu), gsc2);
                const __half2 gw1 = raon_hmul2(raon_lut_h2(s_lut_h, (g0 >> 16) & 0xFu, (g0 >> 24) & 0xFu), gsc2);
                const __half2 gw2 = raon_hmul2(raon_lut_h2(s_lut_h, (g1 >>  0) & 0xFu, (g1 >>  8) & 0xFu), gsc2);
                const __half2 gw3 = raon_hmul2(raon_lut_h2(s_lut_h, (g1 >> 16) & 0xFu, (g1 >> 24) & 0xFu), gsc2);
                const __half2 gw4 = raon_hmul2(raon_lut_h2(s_lut_h, (g2 >>  0) & 0xFu, (g2 >>  8) & 0xFu), gsc2);
                const __half2 gw5 = raon_hmul2(raon_lut_h2(s_lut_h, (g2 >> 16) & 0xFu, (g2 >> 24) & 0xFu), gsc2);
                const __half2 gw6 = raon_hmul2(raon_lut_h2(s_lut_h, (g3 >>  0) & 0xFu, (g3 >>  8) & 0xFu), gsc2);
                const __half2 gw7 = raon_hmul2(raon_lut_h2(s_lut_h, (g3 >> 16) & 0xFu, (g3 >> 24) & 0xFu), gsc2);

                const __half2 uw0 = raon_hmul2(raon_lut_h2(s_lut_h, (u0 >>  0) & 0xFu, (u0 >>  8) & 0xFu), usc2);
                const __half2 uw1 = raon_hmul2(raon_lut_h2(s_lut_h, (u0 >> 16) & 0xFu, (u0 >> 24) & 0xFu), usc2);
                const __half2 uw2 = raon_hmul2(raon_lut_h2(s_lut_h, (u1 >>  0) & 0xFu, (u1 >>  8) & 0xFu), usc2);
                const __half2 uw3 = raon_hmul2(raon_lut_h2(s_lut_h, (u1 >> 16) & 0xFu, (u1 >> 24) & 0xFu), usc2);
                const __half2 uw4 = raon_hmul2(raon_lut_h2(s_lut_h, (u2 >>  0) & 0xFu, (u2 >>  8) & 0xFu), usc2);
                const __half2 uw5 = raon_hmul2(raon_lut_h2(s_lut_h, (u2 >> 16) & 0xFu, (u2 >> 24) & 0xFu), usc2);
                const __half2 uw6 = raon_hmul2(raon_lut_h2(s_lut_h, (u3 >>  0) & 0xFu, (u3 >>  8) & 0xFu), usc2);
                const __half2 uw7 = raon_hmul2(raon_lut_h2(s_lut_h, (u3 >> 16) & 0xFu, (u3 >> 24) & 0xFu), usc2);

                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_xh2[tj * 32 + 0 + ib];
                    const int4 ab = s_xh2[tj * 32 + 8 + ib];
                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    ag[it] = raon_hfma2(gw0, x0, ag[it]); ag[it] = raon_hfma2(gw1, x1, ag[it]);
                    ag[it] = raon_hfma2(gw2, x2, ag[it]); ag[it] = raon_hfma2(gw3, x3, ag[it]);
                    ag[it] = raon_hfma2(gw4, x4, ag[it]); ag[it] = raon_hfma2(gw5, x5, ag[it]);
                    ag[it] = raon_hfma2(gw6, x6, ag[it]); ag[it] = raon_hfma2(gw7, x7, ag[it]);

                    au[it] = raon_hfma2(uw0, x0, au[it]); au[it] = raon_hfma2(uw1, x1, au[it]);
                    au[it] = raon_hfma2(uw2, x2, au[it]); au[it] = raon_hfma2(uw3, x3, au[it]);
                    au[it] = raon_hfma2(uw4, x4, au[it]); au[it] = raon_hfma2(uw5, x5, au[it]);
                    au[it] = raon_hfma2(uw6, x6, au[it]); au[it] = raon_hfma2(uw7, x7, au[it]);
                }
            }
            // High nibbles (elems 16..31)
            {
                const __half2 gw0 = raon_hmul2(raon_lut_h2(s_lut_h, (g0 >>  4) & 0xFu, (g0 >> 12) & 0xFu), gsc2);
                const __half2 gw1 = raon_hmul2(raon_lut_h2(s_lut_h, (g0 >> 20) & 0xFu, (g0 >> 28) & 0xFu), gsc2);
                const __half2 gw2 = raon_hmul2(raon_lut_h2(s_lut_h, (g1 >>  4) & 0xFu, (g1 >> 12) & 0xFu), gsc2);
                const __half2 gw3 = raon_hmul2(raon_lut_h2(s_lut_h, (g1 >> 20) & 0xFu, (g1 >> 28) & 0xFu), gsc2);
                const __half2 gw4 = raon_hmul2(raon_lut_h2(s_lut_h, (g2 >>  4) & 0xFu, (g2 >> 12) & 0xFu), gsc2);
                const __half2 gw5 = raon_hmul2(raon_lut_h2(s_lut_h, (g2 >> 20) & 0xFu, (g2 >> 28) & 0xFu), gsc2);
                const __half2 gw6 = raon_hmul2(raon_lut_h2(s_lut_h, (g3 >>  4) & 0xFu, (g3 >> 12) & 0xFu), gsc2);
                const __half2 gw7 = raon_hmul2(raon_lut_h2(s_lut_h, (g3 >> 20) & 0xFu, (g3 >> 28) & 0xFu), gsc2);

                const __half2 uw0 = raon_hmul2(raon_lut_h2(s_lut_h, (u0 >>  4) & 0xFu, (u0 >> 12) & 0xFu), usc2);
                const __half2 uw1 = raon_hmul2(raon_lut_h2(s_lut_h, (u0 >> 20) & 0xFu, (u0 >> 28) & 0xFu), usc2);
                const __half2 uw2 = raon_hmul2(raon_lut_h2(s_lut_h, (u1 >>  4) & 0xFu, (u1 >> 12) & 0xFu), usc2);
                const __half2 uw3 = raon_hmul2(raon_lut_h2(s_lut_h, (u1 >> 20) & 0xFu, (u1 >> 28) & 0xFu), usc2);
                const __half2 uw4 = raon_hmul2(raon_lut_h2(s_lut_h, (u2 >>  4) & 0xFu, (u2 >> 12) & 0xFu), usc2);
                const __half2 uw5 = raon_hmul2(raon_lut_h2(s_lut_h, (u2 >> 20) & 0xFu, (u2 >> 28) & 0xFu), usc2);
                const __half2 uw6 = raon_hmul2(raon_lut_h2(s_lut_h, (u3 >>  4) & 0xFu, (u3 >> 12) & 0xFu), usc2);
                const __half2 uw7 = raon_hmul2(raon_lut_h2(s_lut_h, (u3 >> 20) & 0xFu, (u3 >> 28) & 0xFu), usc2);

                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_xh2[tj * 32 + 16 + ib];
                    const int4 ab = s_xh2[tj * 32 + 24 + ib];
                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    ag[it] = raon_hfma2(gw0, x0, ag[it]); ag[it] = raon_hfma2(gw1, x1, ag[it]);
                    ag[it] = raon_hfma2(gw2, x2, ag[it]); ag[it] = raon_hfma2(gw3, x3, ag[it]);
                    ag[it] = raon_hfma2(gw4, x4, ag[it]); ag[it] = raon_hfma2(gw5, x5, ag[it]);
                    ag[it] = raon_hfma2(gw6, x6, ag[it]); ag[it] = raon_hfma2(gw7, x7, ag[it]);

                    au[it] = raon_hfma2(uw0, x0, au[it]); au[it] = raon_hfma2(uw1, x1, au[it]);
                    au[it] = raon_hfma2(uw2, x2, au[it]); au[it] = raon_hfma2(uw3, x3, au[it]);
                    au[it] = raon_hfma2(uw4, x4, au[it]); au[it] = raon_hfma2(uw5, x5, au[it]);
                    au[it] = raon_hfma2(uw6, x6, au[it]); au[it] = raon_hfma2(uw7, x7, au[it]);
                }
            }
        }
        __syncthreads();
    }

    if (row < ff) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            const float2 fg = __half22float2(ag[it]);
            const float2 fu = __half22float2(au[it]);
            float vg = fg.x + fg.y;
            float vu = fu.x + fu.y;
            vg += __shfl_xor_sync(0xffffffffu, vg, 4, 8);
            vg += __shfl_xor_sync(0xffffffffu, vg, 2, 8);
            vg += __shfl_xor_sync(0xffffffffu, vg, 1, 8);
            vu += __shfl_xor_sync(0xffffffffu, vu, 4, 8);
            vu += __shfl_xor_sync(0xffffffffu, vu, 2, 8);
            vu += __shfl_xor_sync(0xffffffffu, vu, 1, 8);
            const int tj = t_base + it;
            if (ib == 0 && tj < n_tile) {
                const float rsc = s_rsc[tj];
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                act[(static_cast<size_t>(tok) * k_used + slot) * ff + row] =
                    raon_situ_act(vg * rsc, vu * rsc, beta, linear_beta);
            }
        }
    }
}

// Pure-HFMA2 sm_60 Grouped Down IQ4_NL SoA (BT=16, BT_TH=8, 2 rows/thread = 32 rows/CTA, 8.2KB smem!)
__global__ __launch_bounds__(256, 4) void raon_moe_down_grouped_iq4_nl_soa_sm60_pureh2_kernel(
    const int4 *         __restrict__ dn_qs,
    const uint16_t *     __restrict__ dn_d,
    const int4 *         __restrict__ A_h2,   // [T * k_used, ff / 8] int4
    const float *        __restrict__ A_rsc,  // [T * k_used]
    const float *        __restrict__ top_w,
    const RaonTileDesc * __restrict__ tiles,
    const int32_t *      __restrict__ n_tiles_ptr,
    const int32_t *      __restrict__ sorted_pairs,
    float *              __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const RaonTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT    = 16;
    constexpr int BT_TH = 8;
    __shared__ int4    s_xh2[BT * 32]; // 8 KB
    __shared__ int32_t s_pairs[BT];
    __shared__ float   s_scale[BT];
    __shared__ __half  s_lut_h[16];

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int r_loc  = ltid >> 3;
    const int ib     = ltid & 7;
    const int t_base = t_half * BT_TH;

    const int row0 = blockIdx.x * 32 + r_loc * 2;
    const int row1 = row0 + 1;
    const bool v0  = (row0 < K);
    const bool v1  = (row1 < K);

    if (tid < 16) s_lut_h[tid] = __float2half_rn(static_cast<float>(kvalues_iq4nl_d[tid]));
    if (tid < BT) {
        if (tid < n_tile) {
            const int pair = sorted_pairs[start + tid];
            const size_t sidx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
            s_pairs[tid] = pair;
            s_scale[tid] = __ldg(&A_rsc[sidx]) * __ldg(&top_w[sidx]);
        } else {
            s_pairs[tid] = 0;
            s_scale[tid] = 0.0f;
        }
    }
    __syncthreads();

    const int ff_blks = ff >> 5;
    const int ff_i4   = ff >> 3;
    const size_t roff0 = (static_cast<size_t>(eid) * K + row0) * ff_blks;
    const size_t roff1 = (static_cast<size_t>(eid) * K + row1) * ff_blks;

    const __half2 hzero = __float2half2_rn(0.0f);
    __half2 acc0[BT_TH], acc1[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { acc0[it] = hzero; acc1[it] = hzero; }

    for (int b0 = 0; b0 < ff_blks; b0 += 8) {
        for (int idx = tid; idx < BT * 32; idx += 256) {
            const int c   = idx & 3;
            const int bk  = (idx >> 2) & 7;
            const int tj  = idx >> 5;
            const int b   = b0 + bk;
            const int dst = tj * 32 + c * 8 + bk;
            if (tj < n_tile && b < ff_blks) {
                const int pair = s_pairs[tj];
                const size_t sidx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                s_xh2[dst] = __ldg(&A_h2[sidx * ff_i4 + b * 4 + c]);
            } else {
                s_xh2[dst] = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        const int b = b0 + ib;
        if (v0 && b < ff_blks) {
            const int4 qva = __ldg(&dn_qs[roff0 + b]);
            const __half2 da2 = __float2half2_rn(raon_half_bits_to_f32(__ldg(&dn_d[roff0 + b])) * 16.0f);
            const uint32_t a0 = static_cast<uint32_t>(qva.x), a1 = static_cast<uint32_t>(qva.y);
            const uint32_t a2 = static_cast<uint32_t>(qva.z), a3 = static_cast<uint32_t>(qva.w);

            __half2 db2 = hzero;
            uint32_t b0w = 0, b1w = 0, b2w = 0, b3w = 0;
            if (v1) {
                const int4 qvb = __ldg(&dn_qs[roff1 + b]);
                db2 = __float2half2_rn(raon_half_bits_to_f32(__ldg(&dn_d[roff1 + b])) * 16.0f);
                b0w = static_cast<uint32_t>(qvb.x); b1w = static_cast<uint32_t>(qvb.y);
                b2w = static_cast<uint32_t>(qvb.z); b3w = static_cast<uint32_t>(qvb.w);
            }

            {
                const __half2 aw0 = raon_hmul2(raon_lut_h2(s_lut_h, (a0 >>  0) & 0xFu, (a0 >>  8) & 0xFu), da2);
                const __half2 aw1 = raon_hmul2(raon_lut_h2(s_lut_h, (a0 >> 16) & 0xFu, (a0 >> 24) & 0xFu), da2);
                const __half2 aw2 = raon_hmul2(raon_lut_h2(s_lut_h, (a1 >>  0) & 0xFu, (a1 >>  8) & 0xFu), da2);
                const __half2 aw3 = raon_hmul2(raon_lut_h2(s_lut_h, (a1 >> 16) & 0xFu, (a1 >> 24) & 0xFu), da2);
                const __half2 aw4 = raon_hmul2(raon_lut_h2(s_lut_h, (a2 >>  0) & 0xFu, (a2 >>  8) & 0xFu), da2);
                const __half2 aw5 = raon_hmul2(raon_lut_h2(s_lut_h, (a2 >> 16) & 0xFu, (a2 >> 24) & 0xFu), da2);
                const __half2 aw6 = raon_hmul2(raon_lut_h2(s_lut_h, (a3 >>  0) & 0xFu, (a3 >>  8) & 0xFu), da2);
                const __half2 aw7 = raon_hmul2(raon_lut_h2(s_lut_h, (a3 >> 16) & 0xFu, (a3 >> 24) & 0xFu), da2);

                const __half2 bw0 = raon_hmul2(raon_lut_h2(s_lut_h, (b0w >>  0) & 0xFu, (b0w >>  8) & 0xFu), db2);
                const __half2 bw1 = raon_hmul2(raon_lut_h2(s_lut_h, (b0w >> 16) & 0xFu, (b0w >> 24) & 0xFu), db2);
                const __half2 bw2 = raon_hmul2(raon_lut_h2(s_lut_h, (b1w >>  0) & 0xFu, (b1w >>  8) & 0xFu), db2);
                const __half2 bw3 = raon_hmul2(raon_lut_h2(s_lut_h, (b1w >> 16) & 0xFu, (b1w >> 24) & 0xFu), db2);
                const __half2 bw4 = raon_hmul2(raon_lut_h2(s_lut_h, (b2w >>  0) & 0xFu, (b2w >>  8) & 0xFu), db2);
                const __half2 bw5 = raon_hmul2(raon_lut_h2(s_lut_h, (b2w >> 16) & 0xFu, (b2w >> 24) & 0xFu), db2);
                const __half2 bw6 = raon_hmul2(raon_lut_h2(s_lut_h, (b3w >>  0) & 0xFu, (b3w >>  8) & 0xFu), db2);
                const __half2 bw7 = raon_hmul2(raon_lut_h2(s_lut_h, (b3w >> 16) & 0xFu, (b3w >> 24) & 0xFu), db2);

                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_xh2[tj * 32 + 0 + ib];
                    const int4 ab = s_xh2[tj * 32 + 8 + ib];
                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    acc0[it] = raon_hfma2(aw0, x0, acc0[it]); acc0[it] = raon_hfma2(aw1, x1, acc0[it]);
                    acc0[it] = raon_hfma2(aw2, x2, acc0[it]); acc0[it] = raon_hfma2(aw3, x3, acc0[it]);
                    acc0[it] = raon_hfma2(aw4, x4, acc0[it]); acc0[it] = raon_hfma2(aw5, x5, acc0[it]);
                    acc0[it] = raon_hfma2(aw6, x6, acc0[it]); acc0[it] = raon_hfma2(aw7, x7, acc0[it]);

                    if (v1) {
                        acc1[it] = raon_hfma2(bw0, x0, acc1[it]); acc1[it] = raon_hfma2(bw1, x1, acc1[it]);
                        acc1[it] = raon_hfma2(bw2, x2, acc1[it]); acc1[it] = raon_hfma2(bw3, x3, acc1[it]);
                        acc1[it] = raon_hfma2(bw4, x4, acc1[it]); acc1[it] = raon_hfma2(bw5, x5, acc1[it]);
                        acc1[it] = raon_hfma2(bw6, x6, acc1[it]); acc1[it] = raon_hfma2(bw7, x7, acc1[it]);
                    }
                }
            }
            {
                const __half2 aw0 = raon_hmul2(raon_lut_h2(s_lut_h, (a0 >>  4) & 0xFu, (a0 >> 12) & 0xFu), da2);
                const __half2 aw1 = raon_hmul2(raon_lut_h2(s_lut_h, (a0 >> 20) & 0xFu, (a0 >> 28) & 0xFu), da2);
                const __half2 aw2 = raon_hmul2(raon_lut_h2(s_lut_h, (a1 >>  4) & 0xFu, (a1 >> 12) & 0xFu), da2);
                const __half2 aw3 = raon_hmul2(raon_lut_h2(s_lut_h, (a1 >> 20) & 0xFu, (a1 >> 28) & 0xFu), da2);
                const __half2 aw4 = raon_hmul2(raon_lut_h2(s_lut_h, (a2 >>  4) & 0xFu, (a2 >> 12) & 0xFu), da2);
                const __half2 aw5 = raon_hmul2(raon_lut_h2(s_lut_h, (a2 >> 20) & 0xFu, (a2 >> 28) & 0xFu), da2);
                const __half2 aw6 = raon_hmul2(raon_lut_h2(s_lut_h, (a3 >>  4) & 0xFu, (a3 >> 12) & 0xFu), da2);
                const __half2 aw7 = raon_hmul2(raon_lut_h2(s_lut_h, (a3 >> 20) & 0xFu, (a3 >> 28) & 0xFu), da2);

                const __half2 bw0 = raon_hmul2(raon_lut_h2(s_lut_h, (b0w >>  4) & 0xFu, (b0w >> 12) & 0xFu), db2);
                const __half2 bw1 = raon_hmul2(raon_lut_h2(s_lut_h, (b0w >> 20) & 0xFu, (b0w >> 28) & 0xFu), db2);
                const __half2 bw2 = raon_hmul2(raon_lut_h2(s_lut_h, (b1w >>  4) & 0xFu, (b1w >> 12) & 0xFu), db2);
                const __half2 bw3 = raon_hmul2(raon_lut_h2(s_lut_h, (b1w >> 20) & 0xFu, (b1w >> 28) & 0xFu), db2);
                const __half2 bw4 = raon_hmul2(raon_lut_h2(s_lut_h, (b2w >>  4) & 0xFu, (b2w >> 12) & 0xFu), db2);
                const __half2 bw5 = raon_hmul2(raon_lut_h2(s_lut_h, (b2w >> 20) & 0xFu, (b2w >> 28) & 0xFu), db2);
                const __half2 bw6 = raon_hmul2(raon_lut_h2(s_lut_h, (b3w >>  4) & 0xFu, (b3w >> 12) & 0xFu), db2);
                const __half2 bw7 = raon_hmul2(raon_lut_h2(s_lut_h, (b3w >> 20) & 0xFu, (b3w >> 28) & 0xFu), db2);

                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_xh2[tj * 32 + 16 + ib];
                    const int4 ab = s_xh2[tj * 32 + 24 + ib];
                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    acc0[it] = raon_hfma2(aw0, x0, acc0[it]); acc0[it] = raon_hfma2(aw1, x1, acc0[it]);
                    acc0[it] = raon_hfma2(aw2, x2, acc0[it]); acc0[it] = raon_hfma2(aw3, x3, acc0[it]);
                    acc0[it] = raon_hfma2(aw4, x4, acc0[it]); acc0[it] = raon_hfma2(aw5, x5, acc0[it]);
                    acc0[it] = raon_hfma2(aw6, x6, acc0[it]); acc0[it] = raon_hfma2(aw7, x7, acc0[it]);

                    if (v1) {
                        acc1[it] = raon_hfma2(bw0, x0, acc1[it]); acc1[it] = raon_hfma2(bw1, x1, acc1[it]);
                        acc1[it] = raon_hfma2(bw2, x2, acc1[it]); acc1[it] = raon_hfma2(bw3, x3, acc1[it]);
                        acc1[it] = raon_hfma2(bw4, x4, acc1[it]); acc1[it] = raon_hfma2(bw5, x5, acc1[it]);
                        acc1[it] = raon_hfma2(bw6, x6, acc1[it]); acc1[it] = raon_hfma2(bw7, x7, acc1[it]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (v0) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            const float2 fa = __half22float2(acc0[it]);
            float va = fa.x + fa.y;
            va += __shfl_xor_sync(0xffffffffu, va, 4, 8);
            va += __shfl_xor_sync(0xffffffffu, va, 2, 8);
            va += __shfl_xor_sync(0xffffffffu, va, 1, 8);
            const int tj = t_base + it;
            if (ib == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t sidx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                down_slots[sidx * K + row0] = s_scale[tj] * va;
            }
        }
    }
    if (v1) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            const float2 fb = __half22float2(acc1[it]);
            float vb = fb.x + fb.y;
            vb += __shfl_xor_sync(0xffffffffu, vb, 4, 8);
            vb += __shfl_xor_sync(0xffffffffu, vb, 2, 8);
            vb += __shfl_xor_sync(0xffffffffu, vb, 1, 8);
            const int tj = t_base + it;
            if (ib == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                const size_t sidx = static_cast<size_t>(pair >> 4) * k_used + (pair & 15);
                down_slots[sidx * K + row1] = s_scale[tj] * vb;
            }
        }
    }
}

template <int BT_TH>
__global__ __launch_bounds__(256, 2) void raon_moe_gate_up_grouped_iq4_xs_h2_kernel(
    const BlockIQ4_XS_R *   __restrict__ g_exps,
    const BlockIQ4_XS_R *   __restrict__ u_exps,
    const RaonActF16Block * __restrict__ x_f16,
    const RaonTileDesc *    __restrict__ tiles,
    const int32_t *         __restrict__ n_tiles_ptr,
    const int32_t *         __restrict__ sorted_pairs,
    float *                 __restrict__ act,
    int ff, int K, int k_used, float beta, float linear_beta) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const RaonTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT = 2 * BT_TH;
    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    __shared__ __half  s_lut_h[16];
    auto * s_xh2 = reinterpret_cast<int4 *>(smem_raw);
    auto * s_xd  = reinterpret_cast<float *>(s_xh2 + BT * 32);

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int row    = blockIdx.x * 16 + (ltid >> 3);
    const int ib     = ltid & 7;
    const int t_base = t_half * BT_TH;

    if (tid < 16) s_lut_h[tid] = __float2half_rn(static_cast<float>(kvalues_iq4nl_d[tid]));
    if (tid < n_tile) s_pairs[tid] = sorted_pairs[start + tid];
    __syncthreads();

    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const BlockIQ4_XS_R * g_row = g_exps + (static_cast<size_t>(eid) * ff + row) * nb;
    const BlockIQ4_XS_R * u_row = u_exps + (static_cast<size_t>(eid) * ff + row) * nb;

    float ag[BT_TH], au[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { ag[it] = 0.0f; au[it] = 0.0f; }

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_i4 = BT * 32;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 3;
            const int bk = (idx >> 2) & 7;
            const int tj = idx >> 5;
            // Bank-conflict-free layout: c*8 + bk -> ib=0..7 spans all 32 banks!
            const int dst = tj * 32 + c * 8 + bk;
            if (tj < n_tile) {
                const int tok = s_pairs[tj] >> 4;
                s_xh2[dst] = reinterpret_cast<const int4 *>(
                    x_f16[static_cast<size_t>(tok) * K_blocks + bi * 8 + bk].x)[c];
            } else {
                s_xh2[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            if (tj < n_tile) {
                const int tok = s_pairs[tj] >> 4;
                s_xd[tj * 8 + bk] = x_f16[static_cast<size_t>(tok) * K_blocks + bi * 8 + bk].d;
            } else {
                s_xd[tj * 8 + bk] = 0.0f;
            }
        }
        __syncthreads();

        if (row < ff) {
            const BlockIQ4_XS_R & gb = g_row[bi];
            const float gdl = raon_iq4_xs_dl(gb, ib);
            const uint32_t * gq32 = reinterpret_cast<const uint32_t *>(gb.qs + ib * 16);
            const uint32_t g0 = __ldg(&gq32[0]), g1 = __ldg(&gq32[1]), g2 = __ldg(&gq32[2]), g3 = __ldg(&gq32[3]);

            const BlockIQ4_XS_R & ub = u_row[bi];
            const float udl = raon_iq4_xs_dl(ub, ib);
            const uint32_t * uq32 = reinterpret_cast<const uint32_t *>(ub.qs + ib * 16);
            const uint32_t u0 = __ldg(&uq32[0]), u1 = __ldg(&uq32[1]), u2 = __ldg(&uq32[2]), u3 = __ldg(&uq32[3]);

            {
                const __half2 gw0 = raon_lut_h2(s_lut_h, (g0 >>  0) & 0xFu, (g0 >>  8) & 0xFu);
                const __half2 gw1 = raon_lut_h2(s_lut_h, (g0 >> 16) & 0xFu, (g0 >> 24) & 0xFu);
                const __half2 gw2 = raon_lut_h2(s_lut_h, (g1 >>  0) & 0xFu, (g1 >>  8) & 0xFu);
                const __half2 gw3 = raon_lut_h2(s_lut_h, (g1 >> 16) & 0xFu, (g1 >> 24) & 0xFu);
                const __half2 gw4 = raon_lut_h2(s_lut_h, (g2 >>  0) & 0xFu, (g2 >>  8) & 0xFu);
                const __half2 gw5 = raon_lut_h2(s_lut_h, (g2 >> 16) & 0xFu, (g2 >> 24) & 0xFu);
                const __half2 gw6 = raon_lut_h2(s_lut_h, (g3 >>  0) & 0xFu, (g3 >>  8) & 0xFu);
                const __half2 gw7 = raon_lut_h2(s_lut_h, (g3 >> 16) & 0xFu, (g3 >> 24) & 0xFu);

                const __half2 uw0 = raon_lut_h2(s_lut_h, (u0 >>  0) & 0xFu, (u0 >>  8) & 0xFu);
                const __half2 uw1 = raon_lut_h2(s_lut_h, (u0 >> 16) & 0xFu, (u0 >> 24) & 0xFu);
                const __half2 uw2 = raon_lut_h2(s_lut_h, (u1 >>  0) & 0xFu, (u1 >>  8) & 0xFu);
                const __half2 uw3 = raon_lut_h2(s_lut_h, (u1 >> 16) & 0xFu, (u1 >> 24) & 0xFu);
                const __half2 uw4 = raon_lut_h2(s_lut_h, (u2 >>  0) & 0xFu, (u2 >>  8) & 0xFu);
                const __half2 uw5 = raon_lut_h2(s_lut_h, (u2 >> 16) & 0xFu, (u2 >> 24) & 0xFu);
                const __half2 uw6 = raon_lut_h2(s_lut_h, (u3 >>  0) & 0xFu, (u3 >>  8) & 0xFu);
                const __half2 uw7 = raon_lut_h2(s_lut_h, (u3 >> 16) & 0xFu, (u3 >> 24) & 0xFu);

                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_xh2[tj * 32 + 0 + ib];
                    const int4 ab = s_xh2[tj * 32 + 8 + ib];
                    const float dv = s_xd[tj * 8 + ib];
                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    __half2 rg = raon_hmul2(gw0, x0);
                    rg = raon_hfma2(gw1, x1, rg); rg = raon_hfma2(gw2, x2, rg); rg = raon_hfma2(gw3, x3, rg);
                    rg = raon_hfma2(gw4, x4, rg); rg = raon_hfma2(gw5, x5, rg); rg = raon_hfma2(gw6, x6, rg); rg = raon_hfma2(gw7, x7, rg);
                    const float2 fg = __half22float2(rg);
                    ag[it] = fmaf(gdl * dv, fg.x + fg.y, ag[it]);

                    __half2 ru = raon_hmul2(uw0, x0);
                    ru = raon_hfma2(uw1, x1, ru); ru = raon_hfma2(uw2, x2, ru); ru = raon_hfma2(uw3, x3, ru);
                    ru = raon_hfma2(uw4, x4, ru); ru = raon_hfma2(uw5, x5, ru); ru = raon_hfma2(uw6, x6, ru); ru = raon_hfma2(uw7, x7, ru);
                    const float2 fu = __half22float2(ru);
                    au[it] = fmaf(udl * dv, fu.x + fu.y, au[it]);
                }
            }
            {
                const __half2 gw0 = raon_lut_h2(s_lut_h, (g0 >>  4) & 0xFu, (g0 >> 12) & 0xFu);
                const __half2 gw1 = raon_lut_h2(s_lut_h, (g0 >> 20) & 0xFu, (g0 >> 28) & 0xFu);
                const __half2 gw2 = raon_lut_h2(s_lut_h, (g1 >>  4) & 0xFu, (g1 >> 12) & 0xFu);
                const __half2 gw3 = raon_lut_h2(s_lut_h, (g1 >> 20) & 0xFu, (g1 >> 28) & 0xFu);
                const __half2 gw4 = raon_lut_h2(s_lut_h, (g2 >>  4) & 0xFu, (g2 >> 12) & 0xFu);
                const __half2 gw5 = raon_lut_h2(s_lut_h, (g2 >> 20) & 0xFu, (g2 >> 28) & 0xFu);
                const __half2 gw6 = raon_lut_h2(s_lut_h, (g3 >>  4) & 0xFu, (g3 >> 12) & 0xFu);
                const __half2 gw7 = raon_lut_h2(s_lut_h, (g3 >> 20) & 0xFu, (g3 >> 28) & 0xFu);

                const __half2 uw0 = raon_lut_h2(s_lut_h, (u0 >>  4) & 0xFu, (u0 >> 12) & 0xFu);
                const __half2 uw1 = raon_lut_h2(s_lut_h, (u0 >> 20) & 0xFu, (u0 >> 28) & 0xFu);
                const __half2 uw2 = raon_lut_h2(s_lut_h, (u1 >>  4) & 0xFu, (u1 >> 12) & 0xFu);
                const __half2 uw3 = raon_lut_h2(s_lut_h, (u1 >> 20) & 0xFu, (u1 >> 28) & 0xFu);
                const __half2 uw4 = raon_lut_h2(s_lut_h, (u2 >>  4) & 0xFu, (u2 >> 12) & 0xFu);
                const __half2 uw5 = raon_lut_h2(s_lut_h, (u2 >> 20) & 0xFu, (u2 >> 28) & 0xFu);
                const __half2 uw6 = raon_lut_h2(s_lut_h, (u3 >>  4) & 0xFu, (u3 >> 12) & 0xFu);
                const __half2 uw7 = raon_lut_h2(s_lut_h, (u3 >> 20) & 0xFu, (u3 >> 28) & 0xFu);

                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_xh2[tj * 32 + 16 + ib];
                    const int4 ab = s_xh2[tj * 32 + 24 + ib];
                    const float dv = s_xd[tj * 8 + ib];
                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    __half2 rg = raon_hmul2(gw0, x0);
                    rg = raon_hfma2(gw1, x1, rg); rg = raon_hfma2(gw2, x2, rg); rg = raon_hfma2(gw3, x3, rg);
                    rg = raon_hfma2(gw4, x4, rg); rg = raon_hfma2(gw5, x5, rg); rg = raon_hfma2(gw6, x6, rg); rg = raon_hfma2(gw7, x7, rg);
                    const float2 fg = __half22float2(rg);
                    ag[it] = fmaf(gdl * dv, fg.x + fg.y, ag[it]);

                    __half2 ru = raon_hmul2(uw0, x0);
                    ru = raon_hfma2(uw1, x1, ru); ru = raon_hfma2(uw2, x2, ru); ru = raon_hfma2(uw3, x3, ru);
                    ru = raon_hfma2(uw4, x4, ru); ru = raon_hfma2(uw5, x5, ru); ru = raon_hfma2(uw6, x6, ru); ru = raon_hfma2(uw7, x7, ru);
                    const float2 fu = __half22float2(ru);
                    au[it] = fmaf(udl * dv, fu.x + fu.y, au[it]);
                }
            }
        }
        __syncthreads();
    }

    if (row < ff) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float vg = ag[it], vu = au[it];
            vg += __shfl_xor_sync(0xffffffffu, vg, 4, 8);
            vg += __shfl_xor_sync(0xffffffffu, vg, 2, 8);
            vg += __shfl_xor_sync(0xffffffffu, vg, 1, 8);
            vu += __shfl_xor_sync(0xffffffffu, vu, 4, 8);
            vu += __shfl_xor_sync(0xffffffffu, vu, 2, 8);
            vu += __shfl_xor_sync(0xffffffffu, vu, 1, 8);
            const int tj = t_base + it;
            if (ib == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                act[(static_cast<size_t>(tok) * k_used + slot) * ff + row] = raon_situ_act(vg, vu, beta, linear_beta);
            }
        }
    }
}

template <int BT_TH>
__global__ __launch_bounds__(256, 2) void raon_moe_gate_up_grouped_iq4_xs_dp4a_kernel(
    const BlockIQ4_XS_R * __restrict__ g_exps,
    const BlockIQ4_XS_R * __restrict__ u_exps,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const RaonTileDesc *  __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ act,
    int ff, int K, int k_used, float beta, float linear_beta) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const RaonTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT = 2 * BT_TH;
    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    __shared__ int8_t  s_lut_i8[16];
    auto * s_xq = reinterpret_cast<int4 *>(smem_raw);
    auto * s_xd = reinterpret_cast<float *>(s_xq + BT * 16);

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int row    = blockIdx.x * 16 + (ltid >> 3);
    const int ib     = ltid & 7;
    const int t_base = t_half * BT_TH;

    if (tid < 16) s_lut_i8[tid] = kvalues_iq4nl_d[tid];
    if (tid < n_tile) s_pairs[tid] = sorted_pairs[start + tid];
    __syncthreads();

    const int K_blocks = K >> 5;
    const int nb       = K >> 8;
    const BlockIQ4_XS_R * g_row = g_exps + (static_cast<size_t>(eid) * ff + row) * nb;
    const BlockIQ4_XS_R * u_row = u_exps + (static_cast<size_t>(eid) * ff + row) * nb;

    float ag[BT_TH], au[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { ag[it] = 0.0f; au[it] = 0.0f; }

    for (int bi = 0; bi < nb; ++bi) {
        const int tot_i4 = BT * 16;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 1;
            const int bk = (idx >> 1) & 7;
            const int tj = idx >> 4;
            const int dst = tj * 16 + c * 8 + bk;
            if (tj < n_tile) {
                const int tok = s_pairs[tj] >> 4;
                s_xq[dst] = reinterpret_cast<const int4 *>(
                    x_q8[static_cast<size_t>(tok) * K_blocks + bi * 8 + bk].qs)[c];
            } else {
                s_xq[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            if (tj < n_tile) {
                const int tok = s_pairs[tj] >> 4;
                s_xd[tj * 8 + bk] = x_q8[static_cast<size_t>(tok) * K_blocks + bi * 8 + bk].d;
            } else {
                s_xd[tj * 8 + bk] = 0.0f;
            }
        }
        __syncthreads();

        if (row < ff) {
            const BlockIQ4_XS_R & gb = g_row[bi];
            const float gdl = raon_iq4_xs_dl(gb, ib);
            const uint32_t * gq32 = reinterpret_cast<const uint32_t *>(gb.qs + ib * 16);
            const uint32_t g0 = __ldg(&gq32[0]), g1 = __ldg(&gq32[1]), g2 = __ldg(&gq32[2]), g3 = __ldg(&gq32[3]);

            const int gl0 = raon_pack_iq4_lo4(s_lut_i8, g0), gl1 = raon_pack_iq4_lo4(s_lut_i8, g1);
            const int gl2 = raon_pack_iq4_lo4(s_lut_i8, g2), gl3 = raon_pack_iq4_lo4(s_lut_i8, g3);
            const int gh0 = raon_pack_iq4_hi4(s_lut_i8, g0), gh1 = raon_pack_iq4_hi4(s_lut_i8, g1);
            const int gh2 = raon_pack_iq4_hi4(s_lut_i8, g2), gh3 = raon_pack_iq4_hi4(s_lut_i8, g3);

            const BlockIQ4_XS_R & ub = u_row[bi];
            const float udl = raon_iq4_xs_dl(ub, ib);
            const uint32_t * uq32 = reinterpret_cast<const uint32_t *>(ub.qs + ib * 16);
            const uint32_t u0 = __ldg(&uq32[0]), u1 = __ldg(&uq32[1]), u2 = __ldg(&uq32[2]), u3 = __ldg(&uq32[3]);

            const int ul0 = raon_pack_iq4_lo4(s_lut_i8, u0), ul1 = raon_pack_iq4_lo4(s_lut_i8, u1);
            const int ul2 = raon_pack_iq4_lo4(s_lut_i8, u2), ul3 = raon_pack_iq4_lo4(s_lut_i8, u3);
            const int uh0 = raon_pack_iq4_hi4(s_lut_i8, u0), uh1 = raon_pack_iq4_hi4(s_lut_i8, u1);
            const int uh2 = raon_pack_iq4_hi4(s_lut_i8, u2), uh3 = raon_pack_iq4_hi4(s_lut_i8, u3);

            #pragma unroll
            for (int it = 0; it < BT_TH; ++it) {
                const int tj = t_base + it;
                const int4 xl = s_xq[tj * 16 + 0 + ib];
                const int4 xh = s_xq[tj * 16 + 8 + ib];
                const float dv = s_xd[tj * 8 + ib];

                int sg = dp4a_s32(gl0, xl.x, dp4a_s32(gl1, xl.y, dp4a_s32(gl2, xl.z, dp4a_s32(gl3, xl.w, 0))));
                sg     = dp4a_s32(gh0, xh.x, dp4a_s32(gh1, xh.y, dp4a_s32(gh2, xh.z, dp4a_s32(gh3, xh.w, sg))));
                ag[it] = fmaf(gdl * dv, static_cast<float>(sg), ag[it]);

                int su = dp4a_s32(ul0, xl.x, dp4a_s32(ul1, xl.y, dp4a_s32(ul2, xl.z, dp4a_s32(ul3, xl.w, 0))));
                su     = dp4a_s32(uh0, xh.x, dp4a_s32(uh1, xh.y, dp4a_s32(uh2, xh.z, dp4a_s32(uh3, xh.w, su))));
                au[it] = fmaf(udl * dv, static_cast<float>(su), au[it]);
            }
        }
        __syncthreads();
    }

    if (row < ff) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float vg = ag[it], vu = au[it];
            vg += __shfl_xor_sync(0xffffffffu, vg, 4, 8);
            vg += __shfl_xor_sync(0xffffffffu, vg, 2, 8);
            vg += __shfl_xor_sync(0xffffffffu, vg, 1, 8);
            vu += __shfl_xor_sync(0xffffffffu, vu, 4, 8);
            vu += __shfl_xor_sync(0xffffffffu, vu, 2, 8);
            vu += __shfl_xor_sync(0xffffffffu, vu, 1, 8);
            const int tj = t_base + it;
            if (ib == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                act[(static_cast<size_t>(tok) * k_used + slot) * ff + row] = raon_situ_act(vg, vu, beta, linear_beta);
            }
        }
    }
}

template <int BT_TH>
__global__ __launch_bounds__(256, 2) void raon_moe_down_grouped_iq4_nl_soa_h2_kernel(
    const int4 *            __restrict__ dn_qs,
    const uint16_t *        __restrict__ dn_d,
    const RaonActF16Block * __restrict__ a_f16,
    const float *           __restrict__ top_w,
    const RaonTileDesc *    __restrict__ tiles,
    const int32_t *         __restrict__ n_tiles_ptr,
    const int32_t *         __restrict__ sorted_pairs,
    float *                 __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const RaonTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT = 2 * BT_TH;
    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    __shared__ __half  s_lut_h[16];
    auto * s_xh2 = reinterpret_cast<int4 *>(smem_raw);
    auto * s_xd  = reinterpret_cast<float *>(s_xh2 + BT * 32);

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int r_loc  = ltid >> 3;
    const int ib     = ltid & 7;
    const int t_base = t_half * BT_TH;

    const int row0 = blockIdx.x * 32 + r_loc * 2;
    const int row1 = row0 + 1;
    const bool v0  = (row0 < K);
    const bool v1  = (row1 < K);

    if (tid < 16) s_lut_h[tid] = __float2half_rn(static_cast<float>(kvalues_iq4nl_d[tid]));
    if (tid < n_tile) s_pairs[tid] = sorted_pairs[start + tid];
    __syncthreads();

    const int ff_blks = ff >> 5;
    const size_t roff0 = (static_cast<size_t>(eid) * K + row0) * ff_blks;
    const size_t roff1 = (static_cast<size_t>(eid) * K + row1) * ff_blks;

    float acc0[BT_TH], acc1[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { acc0[it] = 0.0f; acc1[it] = 0.0f; }

    for (int b0 = 0; b0 < ff_blks; b0 += 8) {
        const int tot_i4 = BT * 32;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 3;
            const int bk = (idx >> 2) & 7;
            const int tj = idx >> 5;
            const int b  = b0 + bk;
            const int dst = tj * 32 + c * 8 + bk;
            if (tj < n_tile && b < ff_blks) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                s_xh2[dst] = reinterpret_cast<const int4 *>(
                    a_f16[(static_cast<size_t>(tok) * k_used + slot) * ff_blks + b].x)[c];
            } else {
                s_xh2[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            const int b  = b0 + bk;
            if (tj < n_tile && b < ff_blks) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                s_xd[tj * 8 + bk] = a_f16[(static_cast<size_t>(tok) * k_used + slot) * ff_blks + b].d;
            } else {
                s_xd[tj * 8 + bk] = 0.0f;
            }
        }
        __syncthreads();

        const int b = b0 + ib;
        if (v0 && b < ff_blks) {
            const int4 qva = __ldg(&dn_qs[roff0 + b]);
            const float da = raon_half_bits_to_f32(__ldg(&dn_d[roff0 + b]));
            const uint32_t a0 = static_cast<uint32_t>(qva.x), a1 = static_cast<uint32_t>(qva.y);
            const uint32_t a2 = static_cast<uint32_t>(qva.z), a3 = static_cast<uint32_t>(qva.w);

            float db = 0.0f;
            uint32_t b0w = 0, b1w = 0, b2w = 0, b3w = 0;
            if (v1) {
                const int4 qvb = __ldg(&dn_qs[roff1 + b]);
                db = raon_half_bits_to_f32(__ldg(&dn_d[roff1 + b]));
                b0w = static_cast<uint32_t>(qvb.x); b1w = static_cast<uint32_t>(qvb.y);
                b2w = static_cast<uint32_t>(qvb.z); b3w = static_cast<uint32_t>(qvb.w);
            }

            {
                const __half2 aw0 = raon_lut_h2(s_lut_h, (a0 >>  0) & 0xFu, (a0 >>  8) & 0xFu);
                const __half2 aw1 = raon_lut_h2(s_lut_h, (a0 >> 16) & 0xFu, (a0 >> 24) & 0xFu);
                const __half2 aw2 = raon_lut_h2(s_lut_h, (a1 >>  0) & 0xFu, (a1 >>  8) & 0xFu);
                const __half2 aw3 = raon_lut_h2(s_lut_h, (a1 >> 16) & 0xFu, (a1 >> 24) & 0xFu);
                const __half2 aw4 = raon_lut_h2(s_lut_h, (a2 >>  0) & 0xFu, (a2 >>  8) & 0xFu);
                const __half2 aw5 = raon_lut_h2(s_lut_h, (a2 >> 16) & 0xFu, (a2 >> 24) & 0xFu);
                const __half2 aw6 = raon_lut_h2(s_lut_h, (a3 >>  0) & 0xFu, (a3 >>  8) & 0xFu);
                const __half2 aw7 = raon_lut_h2(s_lut_h, (a3 >> 16) & 0xFu, (a3 >> 24) & 0xFu);

                const __half2 bw0 = raon_lut_h2(s_lut_h, (b0w >>  0) & 0xFu, (b0w >>  8) & 0xFu);
                const __half2 bw1 = raon_lut_h2(s_lut_h, (b0w >> 16) & 0xFu, (b0w >> 24) & 0xFu);
                const __half2 bw2 = raon_lut_h2(s_lut_h, (b1w >>  0) & 0xFu, (b1w >>  8) & 0xFu);
                const __half2 bw3 = raon_lut_h2(s_lut_h, (b1w >> 16) & 0xFu, (b1w >> 24) & 0xFu);
                const __half2 bw4 = raon_lut_h2(s_lut_h, (b2w >>  0) & 0xFu, (b2w >>  8) & 0xFu);
                const __half2 bw5 = raon_lut_h2(s_lut_h, (b2w >> 16) & 0xFu, (b2w >> 24) & 0xFu);
                const __half2 bw6 = raon_lut_h2(s_lut_h, (b3w >>  0) & 0xFu, (b3w >>  8) & 0xFu);
                const __half2 bw7 = raon_lut_h2(s_lut_h, (b3w >> 16) & 0xFu, (b3w >> 24) & 0xFu);

                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_xh2[tj * 32 + 0 + ib];
                    const int4 ab = s_xh2[tj * 32 + 8 + ib];
                    const float dv = s_xd[tj * 8 + ib];
                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    __half2 ra = raon_hmul2(aw0, x0);
                    ra = raon_hfma2(aw1, x1, ra); ra = raon_hfma2(aw2, x2, ra); ra = raon_hfma2(aw3, x3, ra);
                    ra = raon_hfma2(aw4, x4, ra); ra = raon_hfma2(aw5, x5, ra); ra = raon_hfma2(aw6, x6, ra); ra = raon_hfma2(aw7, x7, ra);
                    const float2 fa = __half22float2(ra);
                    acc0[it] = fmaf(da * dv, fa.x + fa.y, acc0[it]);

                    if (v1) {
                        __half2 rb = raon_hmul2(bw0, x0);
                        rb = raon_hfma2(bw1, x1, rb); rb = raon_hfma2(bw2, x2, rb); rb = raon_hfma2(bw3, x3, rb);
                        rb = raon_hfma2(bw4, x4, rb); rb = raon_hfma2(bw5, x5, rb); rb = raon_hfma2(bw6, x6, rb); rb = raon_hfma2(bw7, x7, rb);
                        const float2 fb = __half22float2(rb);
                        acc1[it] = fmaf(db * dv, fb.x + fb.y, acc1[it]);
                    }
                }
            }
            {
                const __half2 aw0 = raon_lut_h2(s_lut_h, (a0 >>  4) & 0xFu, (a0 >> 12) & 0xFu);
                const __half2 aw1 = raon_lut_h2(s_lut_h, (a0 >> 20) & 0xFu, (a0 >> 28) & 0xFu);
                const __half2 aw2 = raon_lut_h2(s_lut_h, (a1 >>  4) & 0xFu, (a1 >> 12) & 0xFu);
                const __half2 aw3 = raon_lut_h2(s_lut_h, (a1 >> 20) & 0xFu, (a1 >> 28) & 0xFu);
                const __half2 aw4 = raon_lut_h2(s_lut_h, (a2 >>  4) & 0xFu, (a2 >> 12) & 0xFu);
                const __half2 aw5 = raon_lut_h2(s_lut_h, (a2 >> 20) & 0xFu, (a2 >> 28) & 0xFu);
                const __half2 aw6 = raon_lut_h2(s_lut_h, (a3 >>  4) & 0xFu, (a3 >> 12) & 0xFu);
                const __half2 aw7 = raon_lut_h2(s_lut_h, (a3 >> 20) & 0xFu, (a3 >> 28) & 0xFu);

                const __half2 bw0 = raon_lut_h2(s_lut_h, (b0w >>  4) & 0xFu, (b0w >> 12) & 0xFu);
                const __half2 bw1 = raon_lut_h2(s_lut_h, (b0w >> 20) & 0xFu, (b0w >> 28) & 0xFu);
                const __half2 bw2 = raon_lut_h2(s_lut_h, (b1w >>  4) & 0xFu, (b1w >> 12) & 0xFu);
                const __half2 bw3 = raon_lut_h2(s_lut_h, (b1w >> 20) & 0xFu, (b1w >> 28) & 0xFu);
                const __half2 bw4 = raon_lut_h2(s_lut_h, (b2w >>  4) & 0xFu, (b2w >> 12) & 0xFu);
                const __half2 bw5 = raon_lut_h2(s_lut_h, (b2w >> 20) & 0xFu, (b2w >> 28) & 0xFu);
                const __half2 bw6 = raon_lut_h2(s_lut_h, (b3w >>  4) & 0xFu, (b3w >> 12) & 0xFu);
                const __half2 bw7 = raon_lut_h2(s_lut_h, (b3w >> 20) & 0xFu, (b3w >> 28) & 0xFu);

                #pragma unroll
                for (int it = 0; it < BT_TH; ++it) {
                    const int tj = t_base + it;
                    const int4 aa = s_xh2[tj * 32 + 16 + ib];
                    const int4 ab = s_xh2[tj * 32 + 24 + ib];
                    const float dv = s_xd[tj * 8 + ib];
                    const __half2 x0 = *reinterpret_cast<const __half2 *>(&aa.x);
                    const __half2 x1 = *reinterpret_cast<const __half2 *>(&aa.y);
                    const __half2 x2 = *reinterpret_cast<const __half2 *>(&aa.z);
                    const __half2 x3 = *reinterpret_cast<const __half2 *>(&aa.w);
                    const __half2 x4 = *reinterpret_cast<const __half2 *>(&ab.x);
                    const __half2 x5 = *reinterpret_cast<const __half2 *>(&ab.y);
                    const __half2 x6 = *reinterpret_cast<const __half2 *>(&ab.z);
                    const __half2 x7 = *reinterpret_cast<const __half2 *>(&ab.w);

                    __half2 ra = raon_hmul2(aw0, x0);
                    ra = raon_hfma2(aw1, x1, ra); ra = raon_hfma2(aw2, x2, ra); ra = raon_hfma2(aw3, x3, ra);
                    ra = raon_hfma2(aw4, x4, ra); ra = raon_hfma2(aw5, x5, ra); ra = raon_hfma2(aw6, x6, ra); ra = raon_hfma2(aw7, x7, ra);
                    const float2 fa = __half22float2(ra);
                    acc0[it] = fmaf(da * dv, fa.x + fa.y, acc0[it]);

                    if (v1) {
                        __half2 rb = raon_hmul2(bw0, x0);
                        rb = raon_hfma2(bw1, x1, rb); rb = raon_hfma2(bw2, x2, rb); rb = raon_hfma2(bw3, x3, rb);
                        rb = raon_hfma2(bw4, x4, rb); rb = raon_hfma2(bw5, x5, rb); rb = raon_hfma2(bw6, x6, rb); rb = raon_hfma2(bw7, x7, rb);
                        const float2 fb = __half22float2(rb);
                        acc1[it] = fmaf(db * dv, fb.x + fb.y, acc1[it]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (v0) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float va = acc0[it];
            va += __shfl_xor_sync(0xffffffffu, va, 4, 8);
            va += __shfl_xor_sync(0xffffffffu, va, 2, 8);
            va += __shfl_xor_sync(0xffffffffu, va, 1, 8);
            const int tj = t_base + it;
            if (ib == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                const size_t sidx = static_cast<size_t>(tok) * k_used + slot;
                down_slots[sidx * K + row0] = top_w[sidx] * va;
            }
        }
    }
    if (v1) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float vb = acc1[it];
            vb += __shfl_xor_sync(0xffffffffu, vb, 4, 8);
            vb += __shfl_xor_sync(0xffffffffu, vb, 2, 8);
            vb += __shfl_xor_sync(0xffffffffu, vb, 1, 8);
            const int tj = t_base + it;
            if (ib == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                const size_t sidx = static_cast<size_t>(tok) * k_used + slot;
                down_slots[sidx * K + row1] = top_w[sidx] * vb;
            }
        }
    }
}

template <int BT_TH>
__global__ __launch_bounds__(256, 2) void raon_moe_down_grouped_iq4_nl_soa_dp4a_kernel(
    const int4 *          __restrict__ dn_qs,
    const uint16_t *      __restrict__ dn_d,
    const BlockQ8_1_Act * __restrict__ a_q8,
    const float *         __restrict__ top_w,
    const RaonTileDesc *  __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const RaonTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT = 2 * BT_TH;
    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    __shared__ int8_t  s_lut_i8[16];
    auto * s_xq = reinterpret_cast<int4 *>(smem_raw);
    auto * s_xd = reinterpret_cast<float *>(s_xq + BT * 16);

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int r_loc  = ltid >> 3;
    const int ib     = ltid & 7;
    const int t_base = t_half * BT_TH;

    const int row0 = blockIdx.x * 32 + r_loc * 2;
    const int row1 = row0 + 1;
    const bool v0  = (row0 < K);
    const bool v1  = (row1 < K);

    if (tid < 16) s_lut_i8[tid] = kvalues_iq4nl_d[tid];
    if (tid < n_tile) s_pairs[tid] = sorted_pairs[start + tid];
    __syncthreads();

    const int ff_blks = ff >> 5;
    const size_t roff0 = (static_cast<size_t>(eid) * K + row0) * ff_blks;
    const size_t roff1 = (static_cast<size_t>(eid) * K + row1) * ff_blks;

    float acc0[BT_TH], acc1[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) { acc0[it] = 0.0f; acc1[it] = 0.0f; }

    for (int b0 = 0; b0 < ff_blks; b0 += 8) {
        const int tot_i4 = BT * 16;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 1;
            const int bk = (idx >> 1) & 7;
            const int tj = idx >> 4;
            const int b  = b0 + bk;
            const int dst = tj * 16 + c * 8 + bk;
            if (tj < n_tile && b < ff_blks) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                s_xq[dst] = reinterpret_cast<const int4 *>(
                    a_q8[(static_cast<size_t>(tok) * k_used + slot) * ff_blks + b].qs)[c];
            } else {
                s_xq[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            const int b  = b0 + bk;
            if (tj < n_tile && b < ff_blks) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                s_xd[tj * 8 + bk] = a_q8[(static_cast<size_t>(tok) * k_used + slot) * ff_blks + b].d;
            } else {
                s_xd[tj * 8 + bk] = 0.0f;
            }
        }
        __syncthreads();

        const int b = b0 + ib;
        if (v0 && b < ff_blks) {
            const int4 qva = __ldg(&dn_qs[roff0 + b]);
            const float da = raon_half_bits_to_f32(__ldg(&dn_d[roff0 + b]));
            const int al0 = raon_pack_iq4_lo4(s_lut_i8, static_cast<uint32_t>(qva.x));
            const int al1 = raon_pack_iq4_lo4(s_lut_i8, static_cast<uint32_t>(qva.y));
            const int al2 = raon_pack_iq4_lo4(s_lut_i8, static_cast<uint32_t>(qva.z));
            const int al3 = raon_pack_iq4_lo4(s_lut_i8, static_cast<uint32_t>(qva.w));
            const int ah0 = raon_pack_iq4_hi4(s_lut_i8, static_cast<uint32_t>(qva.x));
            const int ah1 = raon_pack_iq4_hi4(s_lut_i8, static_cast<uint32_t>(qva.y));
            const int ah2 = raon_pack_iq4_hi4(s_lut_i8, static_cast<uint32_t>(qva.z));
            const int ah3 = raon_pack_iq4_hi4(s_lut_i8, static_cast<uint32_t>(qva.w));

            float db = 0.0f;
            int bl0 = 0, bl1 = 0, bl2 = 0, bl3 = 0, bh0 = 0, bh1 = 0, bh2 = 0, bh3 = 0;
            if (v1) {
                const int4 qvb = __ldg(&dn_qs[roff1 + b]);
                db  = raon_half_bits_to_f32(__ldg(&dn_d[roff1 + b]));
                bl0 = raon_pack_iq4_lo4(s_lut_i8, static_cast<uint32_t>(qvb.x));
                bl1 = raon_pack_iq4_lo4(s_lut_i8, static_cast<uint32_t>(qvb.y));
                bl2 = raon_pack_iq4_lo4(s_lut_i8, static_cast<uint32_t>(qvb.z));
                bl3 = raon_pack_iq4_lo4(s_lut_i8, static_cast<uint32_t>(qvb.w));
                bh0 = raon_pack_iq4_hi4(s_lut_i8, static_cast<uint32_t>(qvb.x));
                bh1 = raon_pack_iq4_hi4(s_lut_i8, static_cast<uint32_t>(qvb.y));
                bh2 = raon_pack_iq4_hi4(s_lut_i8, static_cast<uint32_t>(qvb.z));
                bh3 = raon_pack_iq4_hi4(s_lut_i8, static_cast<uint32_t>(qvb.w));
            }

            #pragma unroll
            for (int it = 0; it < BT_TH; ++it) {
                const int tj = t_base + it;
                const int4 xl = s_xq[tj * 16 + 0 + ib];
                const int4 xh = s_xq[tj * 16 + 8 + ib];
                const float dv = s_xd[tj * 8 + ib];

                int sa = dp4a_s32(al0, xl.x, dp4a_s32(al1, xl.y, dp4a_s32(al2, xl.z, dp4a_s32(al3, xl.w, 0))));
                sa     = dp4a_s32(ah0, xh.x, dp4a_s32(ah1, xh.y, dp4a_s32(ah2, xh.z, dp4a_s32(ah3, xh.w, sa))));
                acc0[it] = fmaf(da * dv, static_cast<float>(sa), acc0[it]);

                if (v1) {
                    int sb = dp4a_s32(bl0, xl.x, dp4a_s32(bl1, xl.y, dp4a_s32(bl2, xl.z, dp4a_s32(bl3, xl.w, 0))));
                    sb     = dp4a_s32(bh0, xh.x, dp4a_s32(bh1, xh.y, dp4a_s32(bh2, xh.z, dp4a_s32(bh3, xh.w, sb))));
                    acc1[it] = fmaf(db * dv, static_cast<float>(sb), acc1[it]);
                }
            }
        }
        __syncthreads();
    }

    if (v0) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float va = acc0[it];
            va += __shfl_xor_sync(0xffffffffu, va, 4, 8);
            va += __shfl_xor_sync(0xffffffffu, va, 2, 8);
            va += __shfl_xor_sync(0xffffffffu, va, 1, 8);
            const int tj = t_base + it;
            if (ib == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                const size_t sidx = static_cast<size_t>(tok) * k_used + slot;
                down_slots[sidx * K + row0] = top_w[sidx] * va;
            }
        }
    }
    if (v1) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float vb = acc1[it];
            vb += __shfl_xor_sync(0xffffffffu, vb, 4, 8);
            vb += __shfl_xor_sync(0xffffffffu, vb, 2, 8);
            vb += __shfl_xor_sync(0xffffffffu, vb, 1, 8);
            const int tj = t_base + it;
            if (ib == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                const size_t sidx = static_cast<size_t>(tok) * k_used + slot;
                down_slots[sidx * K + row1] = top_w[sidx] * vb;
            }
        }
    }
}

template <int BT_TH>
__global__ __launch_bounds__(256, 2) void raon_moe_down_grouped_q5_1_h2_kernel(
    const BlockQ5_1_R *     __restrict__ dn_exps,
    const RaonActF16Block * __restrict__ a_f16,
    const float *           __restrict__ top_w,
    const RaonTileDesc *    __restrict__ tiles,
    const int32_t *         __restrict__ n_tiles_ptr,
    const int32_t *         __restrict__ sorted_pairs,
    float *                 __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const RaonTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT = 2 * BT_TH;
    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    auto * s_xh2 = reinterpret_cast<int4 *>(smem_raw);
    auto * s_xd  = reinterpret_cast<float *>(s_xh2 + BT * 32);
    auto * s_xs  = s_xd + BT * 8;

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int row    = blockIdx.x * 16 + (ltid >> 3);
    const int ib     = ltid & 7;
    const int t_base = t_half * BT_TH;
    const bool valid = (row < K);

    if (tid < n_tile) s_pairs[tid] = sorted_pairs[start + tid];
    __syncthreads();

    const int ff_blks = ff >> 5;
    const BlockQ5_1_R * w_row = dn_exps + (static_cast<size_t>(eid) * K + row) * ff_blks;
    const __half2 neg1024 = __floats2half2_rn(-1024.0f, -1024.0f);

    float acc[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) acc[it] = 0.0f;

    for (int b0 = 0; b0 < ff_blks; b0 += 8) {
        const int tot_i4 = BT * 32;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 3;
            const int bk = (idx >> 2) & 7;
            const int tj = idx >> 5;
            const int b  = b0 + bk;
            const int dst = tj * 32 + c * 8 + bk;
            if (tj < n_tile && b < ff_blks) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                s_xh2[dst] = reinterpret_cast<const int4 *>(
                    a_f16[(static_cast<size_t>(tok) * k_used + slot) * ff_blks + b].x)[c];
            } else {
                s_xh2[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            const int b  = b0 + bk;
            if (tj < n_tile && b < ff_blks) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                const RaonActF16Block & ab = a_f16[(static_cast<size_t>(tok) * k_used + slot) * ff_blks + b];
                s_xd[tj * 8 + bk] = ab.d;
                s_xs[tj * 8 + bk] = ab.s_all;
            } else {
                s_xd[tj * 8 + bk] = 0.0f;
                s_xs[tj * 8 + bk] = 0.0f;
            }
        }
        __syncthreads();

        const int b = b0 + ib;
        if (valid && b < ff_blks) {
            const BlockQ5_1_R & blk = w_row[b];
            const uint2 hdr = __ldg(reinterpret_cast<const uint2 *>(&blk));
            const float d   = raon_half_bits_to_f32(static_cast<uint16_t>(hdr.x & 0xFFFFu));
            const float m   = raon_half_bits_to_f32(static_cast<uint16_t>(hdr.x >> 16));
            const uint32_t qh = hdr.y;
            const uint32_t * qs32 = reinterpret_cast<const uint32_t *>(blk.qs);
            const uint32_t q0 = __ldg(&qs32[0]), q1 = __ldg(&qs32[1]), q2 = __ldg(&qs32[2]), q3 = __ldg(&qs32[3]);

            const uint32_t ul0 = (q0 & 0x0F0F0F0Fu) | ((((qh >>  0) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            const uint32_t ul1 = (q1 & 0x0F0F0F0Fu) | ((((qh >>  4) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            const uint32_t ul2 = (q2 & 0x0F0F0F0Fu) | ((((qh >>  8) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            const uint32_t ul3 = (q3 & 0x0F0F0F0Fu) | ((((qh >> 12) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            const uint32_t uh0 = ((q0 >> 4) & 0x0F0F0F0Fu) | ((((qh >> 16) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            const uint32_t uh1 = ((q1 >> 4) & 0x0F0F0F0Fu) | ((((qh >> 20) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            const uint32_t uh2 = ((q2 >> 4) & 0x0F0F0F0Fu) | ((((qh >> 24) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            const uint32_t uh3 = ((q3 >> 4) & 0x0F0F0F0Fu) | ((((qh >> 28) & 0xFu) * 0x00204081u & 0x01010101u) << 4);

            const __half2 wl0 = raon_hadd2(raon_u8x2_to_h2_lo(ul0), neg1024);
            const __half2 wl1 = raon_hadd2(raon_u8x2_to_h2_hi(ul0), neg1024);
            const __half2 wl2 = raon_hadd2(raon_u8x2_to_h2_lo(ul1), neg1024);
            const __half2 wl3 = raon_hadd2(raon_u8x2_to_h2_hi(ul1), neg1024);
            const __half2 wl4 = raon_hadd2(raon_u8x2_to_h2_lo(ul2), neg1024);
            const __half2 wl5 = raon_hadd2(raon_u8x2_to_h2_hi(ul2), neg1024);
            const __half2 wl6 = raon_hadd2(raon_u8x2_to_h2_lo(ul3), neg1024);
            const __half2 wl7 = raon_hadd2(raon_u8x2_to_h2_hi(ul3), neg1024);

            const __half2 wh0 = raon_hadd2(raon_u8x2_to_h2_lo(uh0), neg1024);
            const __half2 wh1 = raon_hadd2(raon_u8x2_to_h2_hi(uh0), neg1024);
            const __half2 wh2 = raon_hadd2(raon_u8x2_to_h2_lo(uh1), neg1024);
            const __half2 wh3 = raon_hadd2(raon_u8x2_to_h2_hi(uh1), neg1024);
            const __half2 wh4 = raon_hadd2(raon_u8x2_to_h2_lo(uh2), neg1024);
            const __half2 wh5 = raon_hadd2(raon_u8x2_to_h2_hi(uh2), neg1024);
            const __half2 wh6 = raon_hadd2(raon_u8x2_to_h2_lo(uh3), neg1024);
            const __half2 wh7 = raon_hadd2(raon_u8x2_to_h2_hi(uh3), neg1024);

            #pragma unroll
            for (int it = 0; it < BT_TH; ++it) {
                const int tj = t_base + it;
                const int4 a0 = s_xh2[tj * 32 +  0 + ib];
                const int4 a1 = s_xh2[tj * 32 +  8 + ib];
                const int4 a2 = s_xh2[tj * 32 + 16 + ib];
                const int4 a3 = s_xh2[tj * 32 + 24 + ib];
                const float dv = s_xd[tj * 8 + ib];
                const float sv = s_xs[tj * 8 + ib];

                __half2 r = raon_hmul2(wl0, *reinterpret_cast<const __half2 *>(&a0.x));
                r = raon_hfma2(wl1, *reinterpret_cast<const __half2 *>(&a0.y), r);
                r = raon_hfma2(wl2, *reinterpret_cast<const __half2 *>(&a0.z), r);
                r = raon_hfma2(wl3, *reinterpret_cast<const __half2 *>(&a0.w), r);
                r = raon_hfma2(wl4, *reinterpret_cast<const __half2 *>(&a1.x), r);
                r = raon_hfma2(wl5, *reinterpret_cast<const __half2 *>(&a1.y), r);
                r = raon_hfma2(wl6, *reinterpret_cast<const __half2 *>(&a1.z), r);
                r = raon_hfma2(wl7, *reinterpret_cast<const __half2 *>(&a1.w), r);
                r = raon_hfma2(wh0, *reinterpret_cast<const __half2 *>(&a2.x), r);
                r = raon_hfma2(wh1, *reinterpret_cast<const __half2 *>(&a2.y), r);
                r = raon_hfma2(wh2, *reinterpret_cast<const __half2 *>(&a2.z), r);
                r = raon_hfma2(wh3, *reinterpret_cast<const __half2 *>(&a2.w), r);
                r = raon_hfma2(wh4, *reinterpret_cast<const __half2 *>(&a3.x), r);
                r = raon_hfma2(wh5, *reinterpret_cast<const __half2 *>(&a3.y), r);
                r = raon_hfma2(wh6, *reinterpret_cast<const __half2 *>(&a3.z), r);
                r = raon_hfma2(wh7, *reinterpret_cast<const __half2 *>(&a3.w), r);
                const float2 fr = __half22float2(r);
                acc[it] = fmaf(d * dv, fr.x + fr.y, fmaf(m, sv, acc[it]));
            }
        }
        __syncthreads();
    }

    if (valid) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float va = acc[it];
            va += __shfl_xor_sync(0xffffffffu, va, 4, 8);
            va += __shfl_xor_sync(0xffffffffu, va, 2, 8);
            va += __shfl_xor_sync(0xffffffffu, va, 1, 8);
            const int tj = t_base + it;
            if (ib == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                const size_t sidx = static_cast<size_t>(tok) * k_used + slot;
                down_slots[sidx * K + row] = top_w[sidx] * va;
            }
        }
    }
}

template <int BT_TH>
__global__ __launch_bounds__(256, 2) void raon_moe_down_grouped_q5_1_dp4a_kernel(
    const BlockQ5_1_R *   __restrict__ dn_exps,
    const BlockQ8_1_Act * __restrict__ a_q8,
    const float *         __restrict__ top_w,
    const RaonTileDesc *  __restrict__ tiles,
    const int32_t *       __restrict__ n_tiles_ptr,
    const int32_t *       __restrict__ sorted_pairs,
    float *               __restrict__ down_slots,
    int K, int ff, int k_used) {

    const int tile_idx = blockIdx.y;
    if (tile_idx >= *n_tiles_ptr) return;
    const RaonTileDesc td = tiles[tile_idx];
    const int eid    = td.eid;
    const int start  = td.start;
    const int n_tile = td.count;

    constexpr int BT = 2 * BT_TH;
    extern __shared__ uint8_t smem_raw[];
    __shared__ int32_t s_pairs[BT];
    auto * s_xq = reinterpret_cast<int4 *>(smem_raw);
    auto * s_xd = reinterpret_cast<float *>(s_xq + BT * 16);
    auto * s_xs = s_xd + BT * 8;

    const int tid    = threadIdx.x;
    const int t_half = tid >> 7;
    const int ltid   = tid & 127;
    const int row    = blockIdx.x * 16 + (ltid >> 3);
    const int ib     = ltid & 7;
    const int t_base = t_half * BT_TH;
    const bool valid = (row < K);

    if (tid < n_tile) s_pairs[tid] = sorted_pairs[start + tid];
    __syncthreads();

    const int ff_blks = ff >> 5;
    const BlockQ5_1_R * w_row = dn_exps + (static_cast<size_t>(eid) * K + row) * ff_blks;

    float acc[BT_TH];
    #pragma unroll
    for (int it = 0; it < BT_TH; ++it) acc[it] = 0.0f;

    for (int b0 = 0; b0 < ff_blks; b0 += 8) {
        const int tot_i4 = BT * 16;
        for (int idx = tid; idx < tot_i4; idx += 256) {
            const int c  = idx & 1;
            const int bk = (idx >> 1) & 7;
            const int tj = idx >> 4;
            const int b  = b0 + bk;
            const int dst = tj * 16 + c * 8 + bk;
            if (tj < n_tile && b < ff_blks) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                s_xq[dst] = reinterpret_cast<const int4 *>(
                    a_q8[(static_cast<size_t>(tok) * k_used + slot) * ff_blks + b].qs)[c];
            } else {
                s_xq[dst] = make_int4(0, 0, 0, 0);
            }
        }
        const int tot_d = BT * 8;
        for (int idx = tid; idx < tot_d; idx += 256) {
            const int bk = idx & 7;
            const int tj = idx >> 3;
            const int b  = b0 + bk;
            if (tj < n_tile && b < ff_blks) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                const BlockQ8_1_Act & ab = a_q8[(static_cast<size_t>(tok) * k_used + slot) * ff_blks + b];
                s_xd[tj * 8 + bk] = ab.d;
                s_xs[tj * 8 + bk] = ab.s_all;
            } else {
                s_xd[tj * 8 + bk] = 0.0f;
                s_xs[tj * 8 + bk] = 0.0f;
            }
        }
        __syncthreads();

        const int b = b0 + ib;
        if (valid && b < ff_blks) {
            const BlockQ5_1_R & blk = w_row[b];
            const uint2 hdr = __ldg(reinterpret_cast<const uint2 *>(&blk));
            const float d   = raon_half_bits_to_f32(static_cast<uint16_t>(hdr.x & 0xFFFFu));
            const float m   = raon_half_bits_to_f32(static_cast<uint16_t>(hdr.x >> 16));
            const uint32_t qh = hdr.y;
            const uint32_t * qs32 = reinterpret_cast<const uint32_t *>(blk.qs);
            const uint32_t q0 = __ldg(&qs32[0]), q1 = __ldg(&qs32[1]), q2 = __ldg(&qs32[2]), q3 = __ldg(&qs32[3]);

            const int ul0 = static_cast<int>((q0 & 0x0F0F0F0Fu) | ((((qh >>  0) & 0xFu) * 0x00204081u & 0x01010101u) << 4));
            const int ul1 = static_cast<int>((q1 & 0x0F0F0F0Fu) | ((((qh >>  4) & 0xFu) * 0x00204081u & 0x01010101u) << 4));
            const int ul2 = static_cast<int>((q2 & 0x0F0F0F0Fu) | ((((qh >>  8) & 0xFu) * 0x00204081u & 0x01010101u) << 4));
            const int ul3 = static_cast<int>((q3 & 0x0F0F0F0Fu) | ((((qh >> 12) & 0xFu) * 0x00204081u & 0x01010101u) << 4));
            const int uh0 = static_cast<int>(((q0 >> 4) & 0x0F0F0F0Fu) | ((((qh >> 16) & 0xFu) * 0x00204081u & 0x01010101u) << 4));
            const int uh1 = static_cast<int>(((q1 >> 4) & 0x0F0F0F0Fu) | ((((qh >> 20) & 0xFu) * 0x00204081u & 0x01010101u) << 4));
            const int uh2 = static_cast<int>(((q2 >> 4) & 0x0F0F0F0Fu) | ((((qh >> 24) & 0xFu) * 0x00204081u & 0x01010101u) << 4));
            const int uh3 = static_cast<int>(((q3 >> 4) & 0x0F0F0F0Fu) | ((((qh >> 28) & 0xFu) * 0x00204081u & 0x01010101u) << 4));

            #pragma unroll
            for (int it = 0; it < BT_TH; ++it) {
                const int tj = t_base + it;
                const int4 xl = s_xq[tj * 16 + 0 + ib];
                const int4 xh = s_xq[tj * 16 + 8 + ib];
                const float dv = s_xd[tj * 8 + ib];
                const float sv = s_xs[tj * 8 + ib];
                int s = dp4a_s32(ul0, xl.x, dp4a_s32(ul1, xl.y, dp4a_s32(ul2, xl.z, dp4a_s32(ul3, xl.w, 0))));
                s     = dp4a_s32(uh0, xh.x, dp4a_s32(uh1, xh.y, dp4a_s32(uh2, xh.z, dp4a_s32(uh3, xh.w, s))));
                acc[it] = fmaf(d * dv, static_cast<float>(s), fmaf(m, sv, acc[it]));
            }
        }
        __syncthreads();
    }

    if (valid) {
        #pragma unroll
        for (int it = 0; it < BT_TH; ++it) {
            float va = acc[it];
            va += __shfl_xor_sync(0xffffffffu, va, 4, 8);
            va += __shfl_xor_sync(0xffffffffu, va, 2, 8);
            va += __shfl_xor_sync(0xffffffffu, va, 1, 8);
            const int tj = t_base + it;
            if (ib == 0 && tj < n_tile) {
                const int pair = s_pairs[tj];
                const int tok  = pair >> 4;
                const int slot = pair & 15;
                const size_t sidx = static_cast<size_t>(tok) * k_used + slot;
                down_slots[sidx * K + row] = top_w[sidx] * va;
            }
        }
    }
}

} // namespace

// ============================================================================
// Public Launchers
// ============================================================================

void launch_raon_mhc_pre_rmsnorm(
    const float * d_stream, const float * d_mixes,
    const float * d_hc_scale, const float * d_hc_base, const float * d_norm_w,
    float * d_post_out, float * d_comb_out, float * d_x_norm_out,
    int n_embd, int n_tokens, float post_factor, float rms_eps, cudaStream_t stream) {

    raon_mhc_pre_rmsnorm_kernel<<<n_tokens, 256, 0, stream>>>(
        d_stream, d_mixes, d_hc_scale, d_hc_base, d_norm_w,
        d_post_out, d_comb_out, d_x_norm_out, n_embd, post_factor, rms_eps);
}

void launch_raon_mhc_post_steer(
    const float * d_sub, const float * d_residual,
    const float * d_post, const float * d_comb,
    const float * d_steer, float steer_alpha,
    float * d_out, int n_embd, int n_tokens, cudaStream_t stream) {

    const int n4 = n_embd >> 2;
    constexpr int threads = 128;
    dim3 grid((n4 + threads - 1) / threads, n_tokens);
    raon_mhc_post_steer_kernel<<<grid, threads, 0, stream>>>(
        d_sub, d_residual, d_post, d_comb, d_steer, steer_alpha, d_out, n_embd);
}

void launch_raon_mhc_post_steer_flat_mix(
    const float * d_sub, const float * d_residual,
    const float * d_post, const float * d_comb,
    const float * d_steer, float steer_alpha,
    float * d_out, const DeviceTensor & hc_fn, float * d_mixes,
    int n_embd, int n_out, int n_tokens, float rms_eps, cudaStream_t stream,
    void * d_scratch, size_t scratch_bytes) {

    if (hc_fn.dtype == DType::BF16 && (n_embd & 3) == 0) {
        if (n_out > 0 && n_out <= 18) {
            const int n4 = n_embd >> 2;
            const int max_s = (n4 + 15) / 16;
            const int S = (n_tokens < 16 && d_scratch != nullptr) ? std::min(40, max_s) : 1;
            const size_t need = static_cast<size_t>(n_tokens) * S * (n_out + 1) * sizeof(float);
            if (S > 1 && need <= scratch_bytes) {
                auto * part = static_cast<float *>(d_scratch);
                raon_mhc_post_steer_flat_mix_bf16_fast_kernel<true><<<dim3(S, n_tokens), 288, 0, stream>>>(
                    d_sub, d_residual, d_post, d_comb, d_steer, steer_alpha, d_out,
                    static_cast<const uint16_t *>(hc_fn.ptr), part, n_embd, n_out, rms_eps);
                raon_mhc_flat_mix_splitk_finalize_kernel<<<n_tokens, 32, 0, stream>>>(
                    part, d_mixes, S, n_embd * 4, n_out, rms_eps);
                return;
            }
            raon_mhc_post_steer_flat_mix_bf16_fast_kernel<false><<<dim3(1, n_tokens), 288, 0, stream>>>(
                d_sub, d_residual, d_post, d_comb, d_steer, steer_alpha, d_out,
                static_cast<const uint16_t *>(hc_fn.ptr), d_mixes, n_embd, n_out, rms_eps);
            return;
        }
        raon_mhc_post_steer_flat_mix_bf16_kernel<<<dim3(n_out, n_tokens), 256, 0, stream>>>(
            d_sub, d_residual, d_post, d_comb, d_steer, steer_alpha, d_out,
            static_cast<const uint16_t *>(hc_fn.ptr), d_mixes, n_embd, n_out, rms_eps);
        return;
    }
    launch_raon_mhc_post_steer(d_sub, d_residual, d_post, d_comb, d_steer, steer_alpha, d_out, n_embd, n_tokens, stream);
    launch_hc_flat_rmsnorm_mix(d_out, hc_fn, d_mixes, n_embd * 4, n_out, n_tokens, rms_eps, stream);
}

void launch_raon_rda_gate_beta_vscale(
    float * d_decay, const float * d_dt_bias, const float * d_ssm_a,
    float * d_beta, const float * d_rda_delta, float * d_v,
    float gate_lower_bound, int T, int n_heads, int head_dim, cudaStream_t stream) {

    raon_rda_gate_beta_vscale_kernel<<<dim3(n_heads, T), head_dim, 0, stream>>>(
        d_decay, d_dt_bias, d_ssm_a, d_beta, d_rda_delta, d_v, gate_lower_bound, n_heads, head_dim);
}

void launch_raon_mla_split_q(
    const float * d_q_all, float * d_q_nope, float * d_q_pe,
    int T, int n_heads, int qk_nope, int qk_pe, cudaStream_t stream) {

    const int thr = (qk_nope >= qk_pe) ? qk_nope : qk_pe;
    raon_mla_split_q_kernel<<<dim3(n_heads, T), thr, 0, stream>>>(
        d_q_all, d_q_nope, d_q_pe, n_heads, qk_nope, qk_pe);
}

void launch_raon_mla_kv_norm_write_576(
    const float * d_kv_all, const float * d_norm_w, const int32_t * d_pos,
    void * d_kv_cache, bool kv_is_q8_0,
    int T, float rms_eps, cudaStream_t stream) {

    if (kv_is_q8_0) {
        raon_mla_kv_norm_write_576_kernel<true><<<T, 256, 0, stream>>>(
            d_kv_all, d_norm_w, d_pos, d_kv_cache, rms_eps);
    } else {
        raon_mla_kv_norm_write_576_kernel<false><<<T, 256, 0, stream>>>(
            d_kv_all, d_norm_w, d_pos, d_kv_cache, rms_eps);
    }
}

void launch_raon_situ_mul(
    const float * d_gate, const float * d_up, float * d_out,
    int64_t n, float beta, float linear_beta, cudaStream_t stream) {
    raon_situ_mul_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_gate, d_up, d_out, n, beta, linear_beta);
}

bool try_launch_raon_linear_multi_f16(
    const DeviceTensor * const * Ws, float * const * Ys, int n_mat,
    const float * d_x, int T, cudaStream_t stream) {

    if (n_mat <= 0 || n_mat > 4 || T >= 16) return false;
    const int K = static_cast<int>(Ws[0]->ne[0]);
    if ((K & 7) != 0) return false;

    if (n_mat == 1 && K == 128 && Ws[0]->dtype == DType::F16) {
        const int M = static_cast<int>(Ws[0]->ne[1]);
        constexpr int wpb = 8;
        raon_gemv_f16_k128_kernel<<<dim3((M + wpb * 4 - 1) / (wpb * 4), T), dim3(32, wpb), 0, stream>>>(
            static_cast<const __half *>(Ws[0]->ptr), d_x, Ys[0], M);
        return true;
    }

    const size_t smem = static_cast<size_t>(K) * sizeof(float);
    if (smem > 46000) return false;
    RaonMultiF16Desc desc{};
    int tot = 0;
    for (int i = 0; i < n_mat; ++i) {
        if (Ws[i]->dtype != DType::F16 || static_cast<int>(Ws[i]->ne[0]) != K) return false;
        desc.w[i] = static_cast<const __half *>(Ws[i]->ptr);
        desc.y[i] = Ys[i];
        desc.m[i] = static_cast<int>(Ws[i]->ne[1]);
        tot += desc.m[i];
        desc.row_end[i] = tot;
    }
    desc.n_mat = n_mat;
    constexpr int wpb = 8;
    raon_gemv_multi_f16_kernel<<<dim3((tot + wpb - 1) / wpb, T), dim3(32, wpb), smem, stream>>>(desc, d_x, K);
    return true;
}

bool try_launch_raon_linear_iq4_xs_sm60_f16(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    if (cublas == nullptr || T < 16 || W.dtype != DType::IQ4_XS || device_sm_version(W.device) != 60) {
        return false;
    }
    const int K = static_cast<int>(W.ne[0]);
    const int M = static_cast<int>(W.ne[1]);
    if ((K & 255) != 0) return false;

    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    const size_t xf16_b = align256(static_cast<size_t>(T) * K * sizeof(__half));
    const size_t yf16_b = align256(static_cast<size_t>(T) * M * sizeof(__half));
    if (xf16_b + yf16_b + static_cast<size_t>(K) * sizeof(__half) > scratch_bytes) return false;

    const size_t rem = scratch_bytes - xf16_b - yf16_b;
    int chunk_m = std::max(1, std::min(M, static_cast<int>(rem / (static_cast<size_t>(K) * sizeof(__half)))));
    if (chunk_m < M && chunk_m >= 128) chunk_m &= ~127;

    auto * base  = static_cast<uint8_t *>(d_scratch);
    auto * d_x16 = reinterpret_cast<__half *>(base);
    auto * d_y16 = reinterpret_cast<__half *>(base + xf16_b);
    auto * d_w16 = reinterpret_cast<__half *>(base + xf16_b + yf16_b);

    launch_cast_f32_to_f16(d_x, d_x16, static_cast<int64_t>(T) * K, stream);
    cublasSetStream(cublas, stream);
    const __half h_one = __float2half_rn(1.0f), h_zero = __float2half_rn(0.0f);
    const int nb_row = K >> 8;
    constexpr int wpb = 8;

    for (int m0 = 0; m0 < M; m0 += chunk_m) {
        const int cm = std::min(chunk_m, M - m0);
        const int64_t b0   = static_cast<int64_t>(m0) * nb_row;
        const int64_t nblk = static_cast<int64_t>(cm) * nb_row;
        raon_dequant_iq4_xs_f16_kernel<<<(nblk + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
            static_cast<const BlockIQ4_XS_R *>(W.ptr) + b0, d_w16, nblk);
        cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    cm, T, K,
                    &h_one,  d_w16, K,
                    d_x16, K,
                    &h_zero, d_y16 + m0, M);
    }

    const int64_t ny = static_cast<int64_t>(T) * M;
    raon_f16_to_f32_vec_kernel<<<((ny + 1) / 2 + 255) / 256, 256, 0, stream>>>(d_y16, d_y, ny);
    return true;
}

bool try_launch_raon_linear_multi_iq4_xs(
    const DeviceTensor * const * Ws, float * const * Ys, int n_mat,
    const float * d_x, int T, void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    if (n_mat <= 0 || n_mat > 6 || T >= 16) return false;
    const int K = static_cast<int>(Ws[0]->ne[0]);
    if ((K & 255) != 0) return false;

    RaonMultiIQ4Desc desc{};
    int tot = 0;
    for (int i = 0; i < n_mat; ++i) {
        if (Ws[i]->dtype != DType::IQ4_XS || static_cast<int>(Ws[i]->ne[0]) != K) return false;
        desc.w[i] = static_cast<const BlockIQ4_XS_R *>(Ws[i]->ptr);
        desc.y[i] = Ys[i];
        desc.m[i] = static_cast<int>(Ws[i]->ne[1]);
        tot += desc.m[i];
        desc.row_end[i] = tot;
    }
    desc.n_mat = n_mat;

    const int sm = device_sm_version(Ws[0]->device);
    const int K_blocks = K >> 5;
    constexpr int wpb = 8;

    if (sm == 60) {
        const size_t smem = static_cast<size_t>(K_blocks) * sizeof(RaonActF16Block);
        const size_t x_b  = static_cast<size_t>(T) * smem;
        if (smem > 46000 || x_b > scratch_bytes) return false;
        auto * x_f16 = static_cast<RaonActF16Block *>(d_scratch);
        const int tot_b = T * K_blocks;
        const int q_wpb = (tot_b <= 256) ? 2 : 8;
        raon_quantize_act_f16_kernel<<<(tot_b + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(d_x, x_f16, tot_b);
        raon_gemv_multi_iq4_xs_h2_kernel<<<dim3((tot + wpb * 2 - 1) / (wpb * 2), T), dim3(32, wpb), smem, stream>>>(desc, x_f16, K);
        return true;
    } else {
        const size_t smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
        const size_t x_b  = static_cast<size_t>(T) * smem;
        if (smem > 46000 || x_b > scratch_bytes) return false;
        auto * x_q8 = static_cast<BlockQ8_1_Act *>(d_scratch);
        launch_quantize_q8_1(d_x, x_q8, K, T, stream);
        raon_gemv_multi_iq4_xs_dp4a_kernel<<<dim3((tot + wpb * 2 - 1) / (wpb * 2), T), dim3(32, wpb), smem, stream>>>(desc, x_q8, K);
        return true;
    }
}

void launch_raon_moe_ffn_with_shared(
    cublasHandle_t cublas,
    const DeviceTensor & gate_exps, const DeviceTensor & up_exps, const DeviceTensor & down_exps,
    const DeviceTensor & gate_shexp, const DeviceTensor & up_shexp, const DeviceTensor & down_shexp,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, float situ_beta, float situ_linear_beta, cudaStream_t stream) {

    const int K        = static_cast<int>(gate_exps.ne[0]);
    const int ff       = static_cast<int>(gate_exps.ne[1]);
    const int n_expert = static_cast<int>(gate_exps.ne[2]);
    const int ff_sh    = gate_shexp.valid() ? static_cast<int>(gate_shexp.ne[1]) : 0;
    const int K_blocks = K >> 5;
    const int ff_blks  = ff >> 5;
    const int sm       = device_sm_version(gate_exps.device);
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };
    auto * base = static_cast<uint8_t *>(d_scratch);

    if (gate_exps.dtype != DType::IQ4_XS || up_exps.dtype != DType::IQ4_XS) {
        throw std::runtime_error("launch_raon_moe_ffn_with_shared: unsupported gate/up_exps dtype (expected IQ4_XS)");
    }
    const bool dn_iq4nl = (down_exps.dtype == DType::IQ4_NL && down_exps.is_soa);
    const bool dn_q5_1  = (down_exps.dtype == DType::Q5_1);
    if (!dn_iq4nl && !dn_q5_1) {
        throw std::runtime_error("launch_raon_moe_ffn_with_shared: unsupported down_exps dtype (expected SoA IQ4_NL or Q5_1)");
    }

    // 1. Shared Expert (writes directly into d_out)
    if (ff_sh > 0 && gate_shexp.valid() && up_shexp.valid() && down_shexp.valid()) {
        const size_t shg_b = align256(static_cast<size_t>(T) * ff_sh * sizeof(float));
        const size_t shu_b = shg_b;
        auto * sh_g = reinterpret_cast<float *>(base);
        auto * sh_u = reinterpret_cast<float *>(base + shg_b);
        void * sh_rem = base + shg_b + shu_b;
        const size_t sh_rem_b = scratch_bytes - shg_b - shu_b;

        if (T < 16) {
            const DeviceTensor * ws[2] = { &gate_shexp, &up_shexp };
            float * ys[2] = { sh_g, sh_u };
            if (!try_launch_linear_multi_sm60_h2(ws, ys, 2, d_x, T, sh_rem, sh_rem_b, stream)) {
                launch_quantize_q8_1(d_x, sh_rem, K, T, stream);
                launch_linear_multi_prequant(ws, ys, 2, sh_rem, T, stream);
            }
            launch_raon_situ_mul(sh_g, sh_u, sh_g, static_cast<int64_t>(T) * ff_sh, situ_beta, situ_linear_beta, stream);
            launch_linear(cublas, down_shexp, sh_g, d_out, T, sh_rem, sh_rem_b, stream);
        } else {
            launch_linear(cublas, gate_shexp, d_x, sh_g, T, sh_rem, sh_rem_b, stream);
            launch_linear(cublas, up_shexp,   d_x, sh_u, T, sh_rem, sh_rem_b, stream);
            launch_raon_situ_mul(sh_g, sh_u, sh_g, static_cast<int64_t>(T) * ff_sh, situ_beta, situ_linear_beta, stream);
            launch_linear(cublas, down_shexp, sh_g, d_out, T, sh_rem, sh_rem_b, stream);
        }
    } else {
        cudaMemsetAsync(d_out, 0, static_cast<size_t>(T) * K * sizeof(float), stream);
    }

    // 2. Routed Experts — PREFILL (T >= 16)
    if (T >= 16 && k_used <= 16 && n_expert > 0 && n_expert <= 512) {
        const bool sm60_pureh2 = (sm == 60 && dn_iq4nl);
        const bool use_bt32 = (!sm60_pureh2 && T >= 128);
        const int BT        = use_bt32 ? 32 : 16;
        const int max_tiles = (T * k_used + BT - 1) / BT + n_expert;

        const size_t off_b  = align256(static_cast<size_t>(n_expert + 1) * sizeof(int32_t));
        const size_t sp_b   = align256(static_cast<size_t>(T) * k_used * sizeof(int32_t));
        const size_t tl_b   = align256(static_cast<size_t>(max_tiles) * sizeof(RaonTileDesc));
        const size_t nt_b   = 256;
        const size_t meta_b = off_b + sp_b + tl_b + nt_b;
        const size_t act_b  = align256(static_cast<size_t>(T) * k_used * ff * sizeof(float));
        const size_t sl_b   = align256(static_cast<size_t>(T) * k_used * K * sizeof(float));

        auto * exp_offsets  = reinterpret_cast<int32_t *>(base);
        auto * sorted_pairs = reinterpret_cast<int32_t *>(base + off_b);
        auto * tiles        = reinterpret_cast<RaonTileDesc *>(base + off_b + sp_b);
        auto * n_tiles_ptr  = reinterpret_cast<int32_t *>(base + off_b + sp_b + tl_b);

        raon_group_moe_tokens_kernel<<<1, 256, static_cast<size_t>(n_expert) * sizeof(int32_t), stream>>>(
            d_top_ids, exp_offsets, sorted_pairs, T * k_used, n_expert, k_used);
        raon_build_tile_worklist_kernel<<<1, 1, 0, stream>>>(exp_offsets, tiles, n_tiles_ptr, n_expert, BT);

        if (sm60_pureh2) {
            const size_t ah2_b     = align256(static_cast<size_t>(T) * k_used * ff * sizeof(__half));
            const size_t arsc_b    = align256(static_cast<size_t>(T) * k_used * sizeof(float));
            const size_t xh2_b     = align256(static_cast<size_t>(T) * K * sizeof(__half));
            const size_t xrsc_b    = align256(static_cast<size_t>(T) * sizeof(float));
            const size_t persist_b = meta_b + ah2_b + arsc_b;
            const size_t peak_ovl  = std::max(xh2_b + xrsc_b + act_b, sl_b);
            if (persist_b + peak_ovl > scratch_bytes) {
                throw std::runtime_error("launch_raon_moe_ffn_with_shared: scratch overflow (sm_60 pureh2 prefill)");
            }
            auto * a_h2   = reinterpret_cast<__half2 *>(base + meta_b);
            auto * a_rsc  = reinterpret_cast<float *>(base + meta_b + ah2_b);
            uint8_t * ovl = base + persist_b;
            auto * x_h2   = reinterpret_cast<__half2 *>(ovl);
            auto * x_rsc  = reinterpret_cast<float *>(ovl + xh2_b);
            auto * act    = reinterpret_cast<float *>(ovl + xh2_b + xrsc_b);

            raon_prep_row_h2_kernel<<<T, 256, 0, stream>>>(d_x, x_h2, x_rsc, K);
            raon_moe_gate_up_grouped_iq4_xs_sm60_pureh2_kernel
                <<<dim3((ff + 15) / 16, max_tiles), 256, 0, stream>>>(
                    static_cast<const BlockIQ4_XS_R *>(gate_exps.ptr),
                    static_cast<const BlockIQ4_XS_R *>(up_exps.ptr),
                    reinterpret_cast<const int4 *>(x_h2), x_rsc,
                    tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, situ_beta, situ_linear_beta);

            raon_prep_row_h2_kernel<<<T * k_used, 256, 0, stream>>>(act, a_h2, a_rsc, ff);
            auto * down_slots = reinterpret_cast<float *>(ovl);
            raon_moe_down_grouped_iq4_nl_soa_sm60_pureh2_kernel
                <<<dim3((K + 31) / 32, max_tiles), 256, 0, stream>>>(
                    static_cast<const int4 *>(down_exps.ptr),
                    static_cast<const uint16_t *>(down_exps.scales_ptr),
                    reinterpret_cast<const int4 *>(a_h2), a_rsc, d_top_w,
                    tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);

            raon_reduce_slots_add_kernel<<<dim3(((K >> 2) + 255) / 256, T), 256, 0, stream>>>(
                down_slots, d_out, K, k_used, true);
            return;
        }

        if (sm == 60) {
            const size_t af_b      = align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(RaonActF16Block));
            const size_t xf_b      = align256(static_cast<size_t>(T) * K_blocks * sizeof(RaonActF16Block));
            const size_t persist_b = meta_b + af_b;
            const size_t peak_ovl  = std::max(xf_b + act_b, sl_b);
            if (persist_b + peak_ovl > scratch_bytes) {
                throw std::runtime_error("launch_raon_moe_ffn_with_shared: scratch overflow (sm_60 prefill)");
            }
            auto * a_f16  = reinterpret_cast<RaonActF16Block *>(base + meta_b);
            uint8_t * ovl = base + persist_b;
            auto * x_f16  = reinterpret_cast<RaonActF16Block *>(ovl);
            auto * act    = reinterpret_cast<float *>(ovl + xf_b);

            const int xb_tot = T * K_blocks;
            raon_quantize_act_f16_kernel<<<(xb_tot + 7) / 8, dim3(32, 8), 0, stream>>>(d_x, x_f16, xb_tot);

            const size_t gu_smem = static_cast<size_t>(BT) * (32 * sizeof(int4) + 8 * sizeof(float));
            const size_t q5_smem = static_cast<size_t>(BT) * (32 * sizeof(int4) + 16 * sizeof(float));
            if (use_bt32) {
                raon_moe_gate_up_grouped_iq4_xs_h2_kernel<16>
                    <<<dim3((ff + 15) / 16, max_tiles), 256, gu_smem, stream>>>(
                        static_cast<const BlockIQ4_XS_R *>(gate_exps.ptr),
                        static_cast<const BlockIQ4_XS_R *>(up_exps.ptr),
                        x_f16, tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, situ_beta, situ_linear_beta);
            } else {
                raon_moe_gate_up_grouped_iq4_xs_h2_kernel<8>
                    <<<dim3((ff + 15) / 16, max_tiles), 256, gu_smem, stream>>>(
                        static_cast<const BlockIQ4_XS_R *>(gate_exps.ptr),
                        static_cast<const BlockIQ4_XS_R *>(up_exps.ptr),
                        x_f16, tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, situ_beta, situ_linear_beta);
            }

            const int ab_tot = T * k_used * ff_blks;
            raon_quantize_act_f16_kernel<<<(ab_tot + 7) / 8, dim3(32, 8), 0, stream>>>(act, a_f16, ab_tot);

            auto * down_slots = reinterpret_cast<float *>(ovl);
            if (dn_iq4nl) {
                if (use_bt32) {
                    raon_moe_down_grouped_iq4_nl_soa_h2_kernel<16>
                        <<<dim3((K + 31) / 32, max_tiles), 256, gu_smem, stream>>>(
                            static_cast<const int4 *>(down_exps.ptr),
                            static_cast<const uint16_t *>(down_exps.scales_ptr),
                            a_f16, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                } else {
                    raon_moe_down_grouped_iq4_nl_soa_h2_kernel<8>
                        <<<dim3((K + 31) / 32, max_tiles), 256, gu_smem, stream>>>(
                            static_cast<const int4 *>(down_exps.ptr),
                            static_cast<const uint16_t *>(down_exps.scales_ptr),
                            a_f16, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                }
            } else {
                if (use_bt32) {
                    raon_moe_down_grouped_q5_1_h2_kernel<16>
                        <<<dim3((K + 15) / 16, max_tiles), 256, q5_smem, stream>>>(
                            static_cast<const BlockQ5_1_R *>(down_exps.ptr),
                            a_f16, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                } else {
                    raon_moe_down_grouped_q5_1_h2_kernel<8>
                        <<<dim3((K + 15) / 16, max_tiles), 256, q5_smem, stream>>>(
                            static_cast<const BlockQ5_1_R *>(down_exps.ptr),
                            a_f16, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                }
            }
            raon_reduce_slots_add_kernel<<<dim3(((K >> 2) + 255) / 256, T), 256, 0, stream>>>(
                down_slots, d_out, K, k_used, true);
        } else {
            const size_t aq_b      = align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BlockQ8_1_Act));
            const size_t xq_b      = align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
            const size_t persist_b = meta_b + aq_b;
            const size_t peak_ovl  = std::max(xq_b + act_b, sl_b);
            if (persist_b + peak_ovl > scratch_bytes) {
                throw std::runtime_error("launch_raon_moe_ffn_with_shared: scratch overflow (dp4a prefill)");
            }
            auto * a_q8   = reinterpret_cast<BlockQ8_1_Act *>(base + meta_b);
            uint8_t * ovl = base + persist_b;
            auto * x_q8   = reinterpret_cast<BlockQ8_1_Act *>(ovl);
            auto * act    = reinterpret_cast<float *>(ovl + xq_b);

            launch_quantize_q8_1(d_x, x_q8, K, T, stream);

            const size_t gu_smem = static_cast<size_t>(BT) * (16 * sizeof(int4) + 8 * sizeof(float));
            const size_t q5_smem = static_cast<size_t>(BT) * (16 * sizeof(int4) + 16 * sizeof(float));
            if (use_bt32) {
                raon_moe_gate_up_grouped_iq4_xs_dp4a_kernel<16>
                    <<<dim3((ff + 15) / 16, max_tiles), 256, gu_smem, stream>>>(
                        static_cast<const BlockIQ4_XS_R *>(gate_exps.ptr),
                        static_cast<const BlockIQ4_XS_R *>(up_exps.ptr),
                        x_q8, tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, situ_beta, situ_linear_beta);
            } else {
                raon_moe_gate_up_grouped_iq4_xs_dp4a_kernel<8>
                    <<<dim3((ff + 15) / 16, max_tiles), 256, gu_smem, stream>>>(
                        static_cast<const BlockIQ4_XS_R *>(gate_exps.ptr),
                        static_cast<const BlockIQ4_XS_R *>(up_exps.ptr),
                        x_q8, tiles, n_tiles_ptr, sorted_pairs, act, ff, K, k_used, situ_beta, situ_linear_beta);
            }

            launch_quantize_q8_1(act, a_q8, k_used * ff, T, stream);

            auto * down_slots = reinterpret_cast<float *>(ovl);
            if (dn_iq4nl) {
                if (use_bt32) {
                    raon_moe_down_grouped_iq4_nl_soa_dp4a_kernel<16>
                        <<<dim3((K + 31) / 32, max_tiles), 256, gu_smem, stream>>>(
                            static_cast<const int4 *>(down_exps.ptr),
                            static_cast<const uint16_t *>(down_exps.scales_ptr),
                            a_q8, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                } else {
                    raon_moe_down_grouped_iq4_nl_soa_dp4a_kernel<8>
                        <<<dim3((K + 31) / 32, max_tiles), 256, gu_smem, stream>>>(
                            static_cast<const int4 *>(down_exps.ptr),
                            static_cast<const uint16_t *>(down_exps.scales_ptr),
                            a_q8, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                }
            } else {
                if (use_bt32) {
                    raon_moe_down_grouped_q5_1_dp4a_kernel<16>
                        <<<dim3((K + 15) / 16, max_tiles), 256, q5_smem, stream>>>(
                            static_cast<const BlockQ5_1_R *>(down_exps.ptr),
                            a_q8, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                } else {
                    raon_moe_down_grouped_q5_1_dp4a_kernel<8>
                        <<<dim3((K + 15) / 16, max_tiles), 256, q5_smem, stream>>>(
                            static_cast<const BlockQ5_1_R *>(down_exps.ptr),
                            a_q8, d_top_w, tiles, n_tiles_ptr, sorted_pairs, down_slots, K, ff, k_used);
                }
            }
            raon_reduce_slots_add_kernel<<<dim3(((K >> 2) + 255) / 256, T), 256, 0, stream>>>(
                down_slots, d_out, K, k_used, true);
        }
        return;
    }

    // 3. Routed Experts — DECODE (T < 16): Direct In-Smem Activation Quant + Split-Slot Down!
    constexpr int wpb = 8;
    size_t off = 0;
    auto * slots = reinterpret_cast<float *>(base + off); off += align256(static_cast<size_t>(T) * k_used * K * sizeof(float));

    if (sm == 60) {
        auto * x_f16 = reinterpret_cast<RaonActF16Block *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * sizeof(RaonActF16Block));
        auto * a_f16 = reinterpret_cast<RaonActF16Block *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(RaonActF16Block));
        if (off > scratch_bytes) throw std::runtime_error("launch_raon_moe_ffn_with_shared: scratch overflow (sm_60 decode)");

        const int xb_tot = T * K_blocks;
        const int q_wpb  = (xb_tot <= 256) ? 2 : 8;
        raon_quantize_act_f16_kernel<<<(xb_tot + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(d_x, x_f16, xb_tot);

        const size_t gu_smem = static_cast<size_t>(K_blocks) * sizeof(RaonActF16Block);
        raon_moe_gate_up_iq4_xs_h2_qact_dec_kernel<<<dim3(ff_blks, k_used, T), dim3(32, wpb), gu_smem, stream>>>(
            static_cast<const BlockIQ4_XS_R *>(gate_exps.ptr),
            static_cast<const BlockIQ4_XS_R *>(up_exps.ptr),
            x_f16, d_top_ids, a_f16, ff, K, k_used, situ_beta, situ_linear_beta);

        const size_t dn_smem = static_cast<size_t>(ff_blks) * sizeof(RaonActF16Block);
        if (dn_iq4nl) {
            raon_moe_down_iq4_nl_soa_h2_split_dec_kernel
                <<<dim3((K + wpb * 2 - 1) / (wpb * 2), k_used, T), dim3(32, wpb), dn_smem, stream>>>(
                    static_cast<const int4 *>(down_exps.ptr),
                    static_cast<const uint16_t *>(down_exps.scales_ptr),
                    a_f16, d_top_ids, d_top_w, slots, K, ff, k_used);
        } else {
            raon_moe_down_q5_1_h2_split_dec_kernel
                <<<dim3((K + wpb * 2 - 1) / (wpb * 2), k_used, T), dim3(32, wpb), dn_smem, stream>>>(
                    static_cast<const BlockQ5_1_R *>(down_exps.ptr),
                    a_f16, d_top_ids, d_top_w, slots, K, ff, k_used);
        }
    } else {
        auto * x_q8 = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
        auto * a_q8 = reinterpret_cast<BlockQ8_1_Act *>(base + off); off += align256(static_cast<size_t>(T) * k_used * ff_blks * sizeof(BlockQ8_1_Act));
        if (off > scratch_bytes) throw std::runtime_error("launch_raon_moe_ffn_with_shared: scratch overflow (dp4a decode)");

        launch_quantize_q8_1(d_x, x_q8, K, T, stream);

        const size_t gu_smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
        raon_moe_gate_up_iq4_xs_dp4a_qact_dec_kernel<<<dim3(ff_blks, k_used, T), dim3(32, wpb), gu_smem, stream>>>(
            static_cast<const BlockIQ4_XS_R *>(gate_exps.ptr),
            static_cast<const BlockIQ4_XS_R *>(up_exps.ptr),
            x_q8, d_top_ids, a_q8, ff, K, k_used, situ_beta, situ_linear_beta);

        const size_t dn_smem = static_cast<size_t>(ff_blks) * sizeof(BlockQ8_1_Act);
        if (dn_iq4nl) {
            raon_moe_down_iq4_nl_soa_dp4a_split_dec_kernel
                <<<dim3((K + wpb * 2 - 1) / (wpb * 2), k_used, T), dim3(32, wpb), dn_smem, stream>>>(
                    static_cast<const int4 *>(down_exps.ptr),
                    static_cast<const uint16_t *>(down_exps.scales_ptr),
                    a_q8, d_top_ids, d_top_w, slots, K, ff, k_used);
        } else {
            raon_moe_down_q5_1_dp4a_split_dec_kernel
                <<<dim3((K + wpb * 2 - 1) / (wpb * 2), k_used, T), dim3(32, wpb), dn_smem, stream>>>(
                    static_cast<const BlockQ5_1_R *>(down_exps.ptr),
                    a_q8, d_top_ids, d_top_w, slots, K, ff, k_used);
        }
    }

    raon_reduce_slots_add_kernel<<<dim3(((K >> 2) + 63) / 64, T), 64, 0, stream>>>(
        slots, d_out, K, k_used, true);
}

} // namespace rookery
