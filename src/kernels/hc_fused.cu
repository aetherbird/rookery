#include "kernels/kernels.h"

#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>

namespace rookery {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Fused flat RMSNorm(inpL[T, hc_dim]) + BF16 GEMV(hc_fn[n_out, hc_dim]) -> mixes[T, n_out].
// Grid: (n_out, T) = 24 blocks for T=1; Block: 256 threads.
// Each block reads inpL (8192 floats = 32 KB -> L1/L2 cached across all 24 blocks!) via float4,
// computes both sum(x^2) and dot(w_row, x) in a SINGLE pass, then scales dot * inv_rms!
// Eliminates d_flat_norm DRAM roundtrip AND increases parallelism from 1+6 blocks to 24 full blocks!
__global__ void hc_flat_rmsnorm_mix_bf16_kernel(
    const float *    __restrict__ inpL,   // [T, hc_dim]
    const uint16_t * __restrict__ hc_fn,  // [n_out, hc_dim] BF16
    float *          __restrict__ mixes,  // [T, n_out]
    int hc_dim, int n_out, float rms_eps) {

    const int row = blockIdx.x;
    const int t   = blockIdx.y;
    if (row >= n_out) return;

    const float4 * x4 = reinterpret_cast<const float4 *>(inpL + static_cast<size_t>(t) * hc_dim);
    const uint4 *  w8 = reinterpret_cast<const uint4 *>(hc_fn + static_cast<size_t>(row) * hc_dim);
    const int n8 = hc_dim >> 3;

    float ss  = 0.0f;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < n8; i += blockDim.x) {
        const float4 xa = x4[i * 2 + 0];
        const float4 xb = x4[i * 2 + 1];
        const uint4  wp = __ldg(&w8[i]);

        ss  += xa.x * xa.x + xa.y * xa.y + xa.z * xa.z + xa.w * xa.w
             + xb.x * xb.x + xb.y * xb.y + xb.z * xb.z + xb.w * xb.w;
        dot += __uint_as_float((wp.x & 0xffffu) << 16) * xa.x
             + __uint_as_float(wp.x & 0xffff0000u)     * xa.y
             + __uint_as_float((wp.y & 0xffffu) << 16) * xa.z
             + __uint_as_float(wp.y & 0xffff0000u)     * xa.w
             + __uint_as_float((wp.z & 0xffffu) << 16) * xb.x
             + __uint_as_float(wp.z & 0xffff0000u)     * xb.y
             + __uint_as_float((wp.w & 0xffffu) << 16) * xb.z
             + __uint_as_float(wp.w & 0xffff0000u)     * xb.w;
    }

    __shared__ float s_ss[8];
    __shared__ float s_dot[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    ss  = warp_sum(ss);
    dot = warp_sum(dot);
    if (lane == 0) {
        s_ss[warp]  = ss;
        s_dot[warp] = dot;
    }
    __syncthreads();

    if (warp == 0) {
        float t_ss  = (lane < 8) ? s_ss[lane]  : 0.0f;
        float t_dot = (lane < 8) ? s_dot[lane] : 0.0f;
        t_ss  = warp_sum(t_ss);
        t_dot = warp_sum(t_dot);
        if (lane == 0) {
            const float inv_rms = rsqrtf(t_ss / static_cast<float>(hc_dim) + rms_eps);
            mixes[static_cast<size_t>(t) * n_out + row] = t_dot * inv_rms;
        }
    }
}

__global__ void hc_flat_rmsnorm_mix_f32_kernel(
    const float * __restrict__ inpL,   // [T, hc_dim]
    const float * __restrict__ hc_fn,  // [n_out, hc_dim] F32
    float *       __restrict__ mixes,  // [T, n_out]
    int hc_dim, int n_out, float rms_eps) {

    const int row = blockIdx.x;
    const int t   = blockIdx.y;
    if (row >= n_out) return;

    const float4 * x4 = reinterpret_cast<const float4 *>(inpL  + static_cast<size_t>(t)   * hc_dim);
    const float4 * w4 = reinterpret_cast<const float4 *>(hc_fn + static_cast<size_t>(row) * hc_dim);
    const int n4 = hc_dim >> 2;

    float ss  = 0.0f;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 xv = x4[i];
        const float4 wv = __ldg(&w4[i]);
        ss  += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
        dot += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }

    __shared__ float s_ss[8];
    __shared__ float s_dot[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    ss  = warp_sum(ss);
    dot = warp_sum(dot);
    if (lane == 0) { s_ss[warp] = ss; s_dot[warp] = dot; }
    __syncthreads();

    if (warp == 0) {
        float t_ss  = warp_sum((lane < 8) ? s_ss[lane]  : 0.0f);
        float t_dot = warp_sum((lane < 8) ? s_dot[lane] : 0.0f);
        if (lane == 0) {
            const float inv_rms = rsqrtf(t_ss / static_cast<float>(hc_dim) + rms_eps);
            mixes[static_cast<size_t>(t) * n_out + row] = t_dot * inv_rms;
        }
    }
}

__global__ void hc_flat_rmsnorm_mix_f16_kernel(
    const float *  __restrict__ inpL,   // [T, hc_dim]
    const __half * __restrict__ hc_fn,  // [n_out, hc_dim] F16
    float *        __restrict__ mixes,  // [T, n_out]
    int hc_dim, int n_out, float rms_eps) {

    const int row = blockIdx.x;
    const int t   = blockIdx.y;
    if (row >= n_out) return;

    const float4 * x4 = reinterpret_cast<const float4 *>(inpL  + static_cast<size_t>(t)   * hc_dim);
    const uint2 *  w4 = reinterpret_cast<const uint2 *>(hc_fn  + static_cast<size_t>(row) * hc_dim);
    const int n4 = hc_dim >> 2;

    float ss  = 0.0f;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 xv = x4[i];
        const uint2  wp = __ldg(&w4[i]);
        const __half2 h01 = *reinterpret_cast<const __half2 *>(&wp.x);
        const __half2 h23 = *reinterpret_cast<const __half2 *>(&wp.y);
        const float2  f01 = __half22float2(h01);
        const float2  f23 = __half22float2(h23);
        ss  += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
        dot += xv.x * f01.x + xv.y * f01.y + xv.z * f23.x + xv.w * f23.y;
    }

    __shared__ float s_ss[8];
    __shared__ float s_dot[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    ss  = warp_sum(ss);
    dot = warp_sum(dot);
    if (lane == 0) { s_ss[warp] = ss; s_dot[warp] = dot; }
    __syncthreads();

    if (warp == 0) {
        float t_ss  = warp_sum((lane < 8) ? s_ss[lane]  : 0.0f);
        float t_dot = warp_sum((lane < 8) ? s_dot[lane] : 0.0f);
        if (lane == 0) {
            const float inv_rms = rsqrtf(t_ss / static_cast<float>(hc_dim) + rms_eps);
            mixes[static_cast<size_t>(t) * n_out + row] = t_dot * inv_rms;
        }
    }
}

// Parallel 16-lane warp Sinkhorn (20 iterations in registers with zero loops!) + stream collapse + RMSNorm.
__global__ void hc_sinkhorn_pre_rmsnorm_kernel(
    const float * __restrict__ inpL,       // [T, 4, n_embd]
    const float * __restrict__ mixes,      // [T, 24]
    const float * __restrict__ hc_scale,   // [3]
    const float * __restrict__ hc_base,    // [24]
    const float * __restrict__ norm_w,     // [n_embd]
    float *       __restrict__ post_out,   // [T, 4]
    float *       __restrict__ comb_out,   // [T, 16]
    float *       __restrict__ x_norm_out, // [T, n_embd]
    int n_embd, int sinkhorn_iters, float hc_eps, float rms_eps) {

    const int t = blockIdx.x;
    __shared__ float s_pre[4];
    __shared__ float s_reduce[8];

    // Warp 0 computes pre[4], post[4], and 20-iter Sinkhorn comb[4x4] in parallel across lanes 0..15!
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        const float * m = mixes + static_cast<size_t>(t) * 24;

        if (lane < 4) {
            s_pre[lane] = sigmoidf_fast(m[lane] * hc_scale[0] + hc_base[lane]) + hc_eps;
            post_out[t * 4 + lane] = 2.0f * sigmoidf_fast(m[4 + lane] * hc_scale[1] + hc_base[4 + lane]);
        }

        // Lanes 0..15 hold comb[dst + 4*src] where dst = lane & 3, src = lane >> 2
        float c = (lane < 16) ? (m[8 + lane] * hc_scale[2] + hc_base[8 + lane]) : 0.0f;

        // 1. Row softmax over dst (groups of 4: xor 1, xor 2)
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        c = expf(c - mx);
        float rsum = c + __shfl_xor_sync(0xffffffffu, c, 1);
        rsum += __shfl_xor_sync(0xffffffffu, rsum, 2);
        c = c / rsum + hc_eps;

        // 2. First norm_cols (sum over src for fixed dst: xor 4, xor 8)
        {
            float csum = c + __shfl_xor_sync(0xffffffffu, c, 4);
            csum += __shfl_xor_sync(0xffffffffu, csum, 8);
            c /= (csum + hc_eps);
        }

        // 3. Remaining (sinkhorn_iters - 1) alternating norm_rows & norm_cols
        for (int iter = 1; iter < sinkhorn_iters; ++iter) {
            float rs = c + __shfl_xor_sync(0xffffffffu, c, 1);
            rs += __shfl_xor_sync(0xffffffffu, rs, 2);
            c /= (rs + hc_eps);

            float cs = c + __shfl_xor_sync(0xffffffffu, c, 4);
            cs += __shfl_xor_sync(0xffffffffu, cs, 8);
            c /= (cs + hc_eps);
        }

        if (lane < 16) {
            comb_out[t * 16 + lane] = c;
        }
    }
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    const float4 * r0 = reinterpret_cast<const float4 *>(inpL + (static_cast<size_t>(t) * 4 + 0) * n_embd);
    const float4 * r1 = reinterpret_cast<const float4 *>(inpL + (static_cast<size_t>(t) * 4 + 1) * n_embd);
    const float4 * r2 = reinterpret_cast<const float4 *>(inpL + (static_cast<size_t>(t) * 4 + 2) * n_embd);
    const float4 * r3 = reinterpret_cast<const float4 *>(inpL + (static_cast<size_t>(t) * 4 + 3) * n_embd);
    float4 *       yo = reinterpret_cast<float4 *>(x_norm_out + static_cast<size_t>(t) * n_embd);
    const float4 * nw = norm_w ? reinterpret_cast<const float4 *>(norm_w) : nullptr;
    const int n4 = n_embd >> 2;

    // Fast register-resident 1-pass RMSNorm for n_embd <= 2048 (n4 <= 512 -> 2 float4/thread, ZERO DRAM re-read!)
    float4 reg_v[2];
    const bool use_reg = (n4 <= 512);
    float ss = 0.0f;
    if (use_reg) {
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
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
    ss = warp_sum(ss);
    if (lane == 0) s_reduce[warp] = ss;
    __syncthreads();

    if (warp == 0) {
        float tot = (lane < 8) ? s_reduce[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) s_reduce[0] = rsqrtf(tot / static_cast<float>(n_embd) + rms_eps);
    }
    __syncthreads();

    const float inv_rms = s_reduce[0];
    if (use_reg) {
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
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

// Multi-block vectorized float4 hc_post: Grid (ceil((n_embd/4)/256), T) -> saturates GPU!
__global__ void hc_post_kernel(
    const float * __restrict__ sub,
    const float * __restrict__ residual,
    const float * __restrict__ post,
    const float * __restrict__ comb,
    float *       __restrict__ out,
    int n_embd) {

    const int t  = blockIdx.y;
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = n_embd >> 2;
    if (i >= n4) return;

    const float4 sv = reinterpret_cast<const float4 *>(sub + static_cast<size_t>(t) * n_embd)[i];
    const float4 v0 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 0) * n_embd)[i];
    const float4 v1 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 1) * n_embd)[i];
    const float4 v2 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 2) * n_embd)[i];
    const float4 v3 = reinterpret_cast<const float4 *>(residual + (static_cast<size_t>(t) * 4 + 3) * n_embd)[i];

    const float * p = post + t * 4;
    const float * c = comb + t * 16;

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

// Fused hc_post + next sub-block hc_flat_rmsnorm_mix_bf16 (eliminates hc_post launch + 32KB DRAM roundtrip!)
// Grid: (n_out=24, T), Block: 256 threads.
// Block 0 writes `out[t, 0..3, :]`; ALL 24 blocks compute `out` in registers and dot with `hc_fn[row]`!
__global__ void hc_post_flat_rmsnorm_mix_bf16_kernel(
    const float *    __restrict__ sub,      // [T, n_embd]
    const float *    __restrict__ residual, // [T, 4, n_embd]
    const float *    __restrict__ post,     // [T, 4]
    const float *    __restrict__ comb,     // [T, 16]
    float *          __restrict__ out,      // [T, 4, n_embd]
    const uint16_t * __restrict__ hc_fn,    // [n_out, 4 * n_embd] BF16
    float *          __restrict__ mixes,    // [T, n_out]
    int n_embd, int n_out, float rms_eps) {

    const int row = blockIdx.x;
    const int t   = blockIdx.y;
    if (row >= n_out) return;

    const int n4     = n_embd >> 2;
    const int hc_dim = n_embd * 4;
    const float * p  = post + t * 4;
    const float * c  = comb + t * 16;

    const float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
    const float c00 = c[0], c01 = c[4], c02 = c[8],  c03 = c[12];
    const float c10 = c[1], c11 = c[5], c12 = c[9],  c13 = c[13];
    const float c20 = c[2], c21 = c[6], c22 = c[10], c23 = c[14];
    const float c30 = c[3], c31 = c[7], c32 = c[11], c33 = c[15];

    const float4 * s_ptr = reinterpret_cast<const float4 *>(sub + static_cast<size_t>(t) * n_embd);
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
        const float4 sv = s_ptr[i];
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
    ss  = warp_sum(ss);
    dot = warp_sum(dot);
    if (lane == 0) { s_ss[warp] = ss; s_dot[warp] = dot; }
    __syncthreads();

    if (warp == 0) {
        float t_ss  = (lane < 8) ? s_ss[lane]  : 0.0f;
        float t_dot = (lane < 8) ? s_dot[lane] : 0.0f;
        t_ss  = warp_sum(t_ss);
        t_dot = warp_sum(t_dot);
        if (lane == 0) {
            const float inv_rms = rsqrtf(t_ss / static_cast<float>(hc_dim) + rms_eps);
            mixes[static_cast<size_t>(t) * n_out + row] = t_dot * inv_rms;
        }
    }
}

__global__ void hc_head_rmsnorm_kernel(
    const float * __restrict__ inpL,
    const float * __restrict__ mixes,
    const float * __restrict__ hc_scale,
    const float * __restrict__ hc_base,
    const float * __restrict__ out_norm_w,
    float *       __restrict__ out,
    int n_embd, float hc_eps, float rms_eps) {

    const int t = blockIdx.x;
    __shared__ float s_pre[4];
    __shared__ float s_reduce[8];

    if (threadIdx.x < 4) {
        const float sc = hc_scale[0];
        const float * m = mixes + t * 4;
        s_pre[threadIdx.x] = sigmoidf_fast(m[threadIdx.x] * sc + hc_base[threadIdx.x]) + hc_eps;
    }
    __syncthreads();

    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    const float * r0 = inpL + (static_cast<size_t>(t) * 4 + 0) * n_embd;
    const float * r1 = inpL + (static_cast<size_t>(t) * 4 + 1) * n_embd;
    const float * r2 = inpL + (static_cast<size_t>(t) * 4 + 2) * n_embd;
    const float * r3 = inpL + (static_cast<size_t>(t) * 4 + 3) * n_embd;
    float *       yo = out  + static_cast<size_t>(t) * n_embd;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float v = p0 * r0[i] + p1 * r1[i] + p2 * r2[i] + p3 * r3[i];
        yo[i] = v;
        ss += v * v;
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    ss = warp_sum(ss);
    if (lane == 0) s_reduce[warp] = ss;
    __syncthreads();

    if (warp == 0) {
        float tot = (lane < 8) ? s_reduce[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) s_reduce[0] = rsqrtf(tot / static_cast<float>(n_embd) + rms_eps);
    }
    __syncthreads();

    const float inv_rms = s_reduce[0];
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        yo[i] = yo[i] * inv_rms * (out_norm_w ? out_norm_w[i] : 1.0f);
    }
}

// Fused flat RMSNorm(inpL[T, hc_dim]) + Q8_0 SoA GEMV(hc_fn[n_out, hc_dim]) -> mixes[T, n_out].
// Used by GLM-5.3-Flash where hc_attn_fn / hc_ffn_fn are Q8_0 [16384, 24]!
// Vectorized LDG.128 (int4 = 16 bytes of Q8_0) + 4x float4 loads of inpL!
__global__ void hc_flat_rmsnorm_mix_q8_0_soa_kernel(
    const float *    __restrict__ inpL,      // [T, hc_dim]
    const int8_t *   __restrict__ qs,        // [n_out, hc_dim]
    const uint16_t * __restrict__ scales,    // [n_out, hc_dim/32]
    float *          __restrict__ mixes,     // [T, n_out]
    int hc_dim, int n_out, float rms_eps) {

    const int row = blockIdx.x;
    const int t   = blockIdx.y;
    if (row >= n_out) return;

    const int nb  = hc_dim >> 5;
    const int n16 = hc_dim >> 4; // 16 elements per int4 load (half a Q8_0 block!)
    const float4 *   x4  = reinterpret_cast<const float4 *>(inpL + static_cast<size_t>(t) * hc_dim);
    const int4 *     w16 = reinterpret_cast<const int4 *>(qs + static_cast<size_t>(row) * hc_dim);
    const uint16_t * sp  = scales + static_cast<size_t>(row) * nb;

    float ss  = 0.0f;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < n16; i += blockDim.x) {
        const int4 wv = __ldg(&w16[i]);
        __half dh;
        *reinterpret_cast<uint16_t *>(&dh) = __ldg(&sp[i >> 1]);
        const float d = __half2float(dh);

        const int32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
        float sub_dot = 0.0f;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const float4 xv = x4[i * 4 + g];
            const int32_t qw = words[g];
            const int8_t q0 = static_cast<int8_t>(qw & 0xFF);
            const int8_t q1 = static_cast<int8_t>((qw >> 8) & 0xFF);
            const int8_t q2 = static_cast<int8_t>((qw >> 16) & 0xFF);
            const int8_t q3 = static_cast<int8_t>(qw >> 24);
            ss      += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
            sub_dot += xv.x * q0 + xv.y * q1 + xv.z * q2 + xv.w * q3;
        }
        dot += d * sub_dot;
    }

    __shared__ float s_ss[8];
    __shared__ float s_dot[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    ss  = warp_sum(ss);
    dot = warp_sum(dot);
    if (lane == 0) { s_ss[warp] = ss; s_dot[warp] = dot; }
    __syncthreads();

    if (warp == 0) {
        float t_ss  = (lane < 8) ? s_ss[lane]  : 0.0f;
        float t_dot = (lane < 8) ? s_dot[lane] : 0.0f;
        t_ss  = warp_sum(t_ss);
        t_dot = warp_sum(t_dot);
        if (lane == 0) {
            const float inv_rms = rsqrtf(t_ss / static_cast<float>(hc_dim) + rms_eps);
            mixes[static_cast<size_t>(t) * n_out + row] = t_dot * inv_rms;
        }
    }
}

// Unweighted mean across hc=4 streams + RMSNorm (GLM-5.3-Flash head collapse)
__global__ void hc_mean_rmsnorm_kernel(
    const float * __restrict__ inpL,       // [T, 4, n_embd]
    const float * __restrict__ out_norm_w, // [n_embd]
    float *       __restrict__ out,        // [T, n_embd]
    int n_embd, float rms_eps) {

    const int t = blockIdx.x;
    const float * r0 = inpL + (static_cast<size_t>(t) * 4 + 0) * n_embd;
    const float * r1 = inpL + (static_cast<size_t>(t) * 4 + 1) * n_embd;
    const float * r2 = inpL + (static_cast<size_t>(t) * 4 + 2) * n_embd;
    const float * r3 = inpL + (static_cast<size_t>(t) * 4 + 3) * n_embd;
    float *       yo = out  + static_cast<size_t>(t) * n_embd;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float v = 0.25f * (r0[i] + r1[i] + r2[i] + r3[i]);
        yo[i] = v;
        ss += v * v;
    }

    __shared__ float s_red[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    ss = warp_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? s_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) s_red[0] = rsqrtf(tot / static_cast<float>(n_embd) + rms_eps);
    }
    __syncthreads();
    const float inv_rms = s_red[0];
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        yo[i] = yo[i] * inv_rms * (out_norm_w ? out_norm_w[i] : 1.0f);
    }
}

} // namespace

void launch_hc_flat_rmsnorm_mix(
    const float * d_inpL, const DeviceTensor & hc_fn, float * d_mixes,
    int hc_dim, int n_out, int n_tokens, float rms_eps, cudaStream_t stream) {

    dim3 grid(n_out, n_tokens);
    if (hc_fn.dtype == DType::BF16) {
        hc_flat_rmsnorm_mix_bf16_kernel<<<grid, 256, 0, stream>>>(
            d_inpL, static_cast<const uint16_t *>(hc_fn.ptr), d_mixes, hc_dim, n_out, rms_eps);
    } else if (hc_fn.dtype == DType::F32) {
        hc_flat_rmsnorm_mix_f32_kernel<<<grid, 256, 0, stream>>>(
            d_inpL, static_cast<const float *>(hc_fn.ptr), d_mixes, hc_dim, n_out, rms_eps);
    } else if (hc_fn.dtype == DType::F16) {
        hc_flat_rmsnorm_mix_f16_kernel<<<grid, 256, 0, stream>>>(
            d_inpL, static_cast<const __half *>(hc_fn.ptr), d_mixes, hc_dim, n_out, rms_eps);
    } else if (hc_fn.dtype == DType::Q8_0 && hc_fn.is_soa) {
        hc_flat_rmsnorm_mix_q8_0_soa_kernel<<<grid, 256, 0, stream>>>(
            d_inpL, static_cast<const int8_t *>(hc_fn.ptr), static_cast<const uint16_t *>(hc_fn.scales_ptr),
            d_mixes, hc_dim, n_out, rms_eps);
    } else {
        throw std::runtime_error("launch_hc_flat_rmsnorm_mix: expected BF16, F32, F16, or SoA Q8_0 hc_fn");
    }
}

void launch_hc_mean_rmsnorm(
    const float * d_inpL, const float * d_out_norm_w, float * d_out,
    int n_embd, int n_tokens, float rms_eps, cudaStream_t stream) {
    hc_mean_rmsnorm_kernel<<<n_tokens, 256, 0, stream>>>(d_inpL, d_out_norm_w, d_out, n_embd, rms_eps);
}

void launch_hc_sinkhorn_pre_rmsnorm(
    const float * d_inpL, const float * d_mixes,
    const float * d_hc_scale, const float * d_hc_base, const float * d_norm_w,
    float * d_post_out, float * d_comb_out, float * d_x_norm_out,
    int n_embd, int n_tokens, int sinkhorn_iters, float hc_eps, float rms_eps,
    cudaStream_t stream) {

    hc_sinkhorn_pre_rmsnorm_kernel<<<n_tokens, 256, 0, stream>>>(
        d_inpL, d_mixes, d_hc_scale, d_hc_base, d_norm_w,
        d_post_out, d_comb_out, d_x_norm_out,
        n_embd, sinkhorn_iters, hc_eps, rms_eps);
}

void launch_hc_post(
    const float * d_sub, const float * d_residual,
    const float * d_post, const float * d_comb, float * d_out,
    int n_embd, int n_tokens, cudaStream_t stream) {

    const int n4 = n_embd >> 2;
    constexpr int threads = 128; // 4 blocks of 128 for n_embd=2048 -> 4x more SMs than 1 block
    dim3 grid((n4 + threads - 1) / threads, n_tokens);
    hc_post_kernel<<<grid, threads, 0, stream>>>(
        d_sub, d_residual, d_post, d_comb, d_out, n_embd);
}

void launch_hc_post_flat_rmsnorm_mix(
    const float * d_sub, const float * d_residual,
    const float * d_post, const float * d_comb, float * d_out,
    const DeviceTensor & hc_fn, float * d_mixes,
    int n_embd, int n_out, int n_tokens, float rms_eps, cudaStream_t stream) {

    if (hc_fn.dtype == DType::BF16 && (n_embd & 3) == 0) {
        hc_post_flat_rmsnorm_mix_bf16_kernel<<<dim3(n_out, n_tokens), 256, 0, stream>>>(
            d_sub, d_residual, d_post, d_comb, d_out,
            static_cast<const uint16_t *>(hc_fn.ptr), d_mixes, n_embd, n_out, rms_eps);
        return;
    }
    launch_hc_post(d_sub, d_residual, d_post, d_comb, d_out, n_embd, n_tokens, stream);
    launch_hc_flat_rmsnorm_mix(d_out, hc_fn, d_mixes, n_embd * 4, n_out, n_tokens, rms_eps, stream);
}

void launch_hc_head_rmsnorm(
    const float * d_inpL, const float * d_mixes,
    const float * d_hc_scale, const float * d_hc_base, const float * d_out_norm_w,
    float * d_out, int n_embd, int n_tokens, float hc_eps, float rms_eps,
    cudaStream_t stream) {

    hc_head_rmsnorm_kernel<<<n_tokens, 256, 0, stream>>>(
        d_inpL, d_mixes, d_hc_scale, d_hc_base, d_out_norm_w,
        d_out, n_embd, hc_eps, rms_eps);
}

} // namespace rookery
