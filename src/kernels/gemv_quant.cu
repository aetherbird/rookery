#include "core/device.h"
#include "kernels/kernels.h"
#include "kernels/q8_1_act.cuh"

#include <cuda_fp16.h>
#include <cfloat>
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

__device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__global__ void quantize_q8_1_kernel(
    const float *  __restrict__ x,
    BlockQ8_1_Act * __restrict__ qx,
    int total_blocks) {

    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= total_blocks) return;

    const int lane = threadIdx.x;
    const float v = x[static_cast<size_t>(b) * 32 + lane];
    float amax = __shfl_sync(0xffffffffu, warp_max(fabsf(v)), 0);

    const float d     = amax / 127.0f;
    const float inv_d = (d > 0.0f) ? (127.0f / amax) : 0.0f;
    const int   q     = __float2int_rn(v * inv_d);
    qx[b].qs[lane]    = static_cast<int8_t>(q);

    // 16-lane butterfly sums for s0 = sum(qs[0..15]) and s1 = sum(qs[16..31])
    int hsum = q;
    hsum += __shfl_xor_sync(0xffffffffu, hsum, 1);
    hsum += __shfl_xor_sync(0xffffffffu, hsum, 2);
    hsum += __shfl_xor_sync(0xffffffffu, hsum, 4);
    hsum += __shfl_xor_sync(0xffffffffu, hsum, 8);
    const int s0 = __shfl_sync(0xffffffffu, hsum, 0);
    const int s1 = __shfl_sync(0xffffffffu, hsum, 16);
    if (lane == 0) {
        qx[b].d     = d;
        qx[b].s0    = s0;
        qx[b].s1    = s1;
        qx[b].s_all = d * static_cast<float>(s0 + s1);
    }
}

// 100% COALESCED 128-bit LDG.128 SoA Q8_0 x Q8_1 GEMV!
// 2 threads per 32-element block -> 16 blocks processed in parallel per warp -> 512 contiguous bytes per warp load!
// 8 warps/block share s_x in shared memory.
__global__ void gemv_q8_0_soa_ldg128_kernel(
    const int4 *          __restrict__ W_qs,   // [N, K_blocks * 2] int4 (16B per half-block)
    const uint16_t *      __restrict__ W_sc,   // [N, K_blocks]     __half scales
    const BlockQ8_1_Act * __restrict__ X,      // [T, K_blocks]
    float *               __restrict__ Y,      // [T, N]
    int K_blocks, int N, int T) {

    extern __shared__ BlockQ8_1_Act s_x[];

    const int t    = blockIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;

    const BlockQ8_1_Act * x_row = X + static_cast<size_t>(t) * K_blocks;
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(x_row);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) {
        sx_out[i] = sx_in[i];
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N || t >= T) return;

    const int lane        = threadIdx.x;
    const int blk_in_warp = lane >> 1; // 0..15 (16 blocks/warp!)
    const int half_in_blk = lane & 1;  // 0 or 1 (each reads 16B = 4 int32s via ONE LDG.128!)

    const int4 *     w_q_row = W_qs + static_cast<size_t>(row) * (K_blocks * 2);
    const uint16_t * w_s_row = W_sc + static_cast<size_t>(row) * K_blocks;

    float acc = 0.0f;
    for (int b = blk_in_warp; b < K_blocks; b += 16) {
        const int4 w4 = __ldg(&w_q_row[b * 2 + half_in_blk]);
        const int4 x4 = reinterpret_cast<const int4 *>(s_x[b].qs)[half_in_blk];

        int sumi = dp4a_s32(w4.x, x4.x, 0);
        sumi     = dp4a_s32(w4.y, x4.y, sumi);
        sumi     = dp4a_s32(w4.z, x4.z, sumi);
        sumi     = dp4a_s32(w4.w, x4.w, sumi);

        __half wd_h;
        *reinterpret_cast<uint16_t *>(&wd_h) = __ldg(&w_s_row[b]);
        acc += (__half2float(wd_h) * s_x[b].d) * static_cast<float>(sumi);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        Y[static_cast<size_t>(t) * N + row] = acc;
    }
}

// DUAL-TOKEN (T == 2) weight-reuse Q8_0 SoA GEMV: reads W_qs ONCE for BOTH t=0 and t=1!
// Makes T==2 MTP verification pay 1.00x memory bandwidth for QKV, WO, dense FFN, and lm_head (426 MB)!
__global__ void gemv_q8_0_soa_ldg128_t2_kernel(
    const int4 *          __restrict__ W_qs,
    const uint16_t *      __restrict__ W_sc,
    const BlockQ8_1_Act * __restrict__ X, // [2, K_blocks]
    float *               __restrict__ Y, // [2, N]
    int K_blocks, int N) {

    extern __shared__ BlockQ8_1_Act s_x[];
    BlockQ8_1_Act * s_x0 = s_x;
    BlockQ8_1_Act * s_x1 = s_x + K_blocks;

    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const int n_i4 = (2 * K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(X);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    const int lane        = threadIdx.x;
    const int blk_in_warp = lane >> 1;
    const int half_in_blk = lane & 1;

    const int4 *     w_q_row = W_qs + static_cast<size_t>(row) * (K_blocks * 2);
    const uint16_t * w_s_row = W_sc + static_cast<size_t>(row) * K_blocks;

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int b = blk_in_warp; b < K_blocks; b += 16) {
        const int4 w4  = __ldg(&w_q_row[b * 2 + half_in_blk]);
        const int4 x40 = reinterpret_cast<const int4 *>(s_x0[b].qs)[half_in_blk];
        const int4 x41 = reinterpret_cast<const int4 *>(s_x1[b].qs)[half_in_blk];

        int s0 = dp4a_s32(w4.x, x40.x, 0);
        s0     = dp4a_s32(w4.y, x40.y, s0);
        s0     = dp4a_s32(w4.z, x40.z, s0);
        s0     = dp4a_s32(w4.w, x40.w, s0);

        int s1 = dp4a_s32(w4.x, x41.x, 0);
        s1     = dp4a_s32(w4.y, x41.y, s1);
        s1     = dp4a_s32(w4.z, x41.z, s1);
        s1     = dp4a_s32(w4.w, x41.w, s1);

        __half wd_h;
        *reinterpret_cast<uint16_t *>(&wd_h) = __ldg(&w_s_row[b]);
        const float wd = __half2float(wd_h);
        acc0 += (wd * s_x0[b].d) * static_cast<float>(s0);
        acc1 += (wd * s_x1[b].d) * static_cast<float>(s1);
    }

    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if (lane == 0) {
        Y[row]     = acc0;
        Y[N + row] = acc1;
    }
}

__global__ void gemv_f16_kernel(
    const __half * __restrict__ W,
    const float *  __restrict__ X,
    float *        __restrict__ Y,
    int K, int N, int T) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int t   = blockIdx.y;
    if (row >= N || t >= T) return;

    const int lane = threadIdx.x;
    const __half * w_row = W + static_cast<size_t>(row) * K;
    const float *  x_row = X + static_cast<size_t>(t) * K;

    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) {
        acc += __half2float(__ldg(&w_row[k])) * x_row[k];
    }
    acc = warp_sum(acc);
    if (lane == 0) Y[static_cast<size_t>(t) * N + row] = acc;
}

__global__ void gemv_bf16_kernel(
    const uint16_t * __restrict__ W,
    const float *    __restrict__ X,
    float *          __restrict__ Y,
    int K, int N, int T) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int t   = blockIdx.y;
    if (row >= N || t >= T) return;

    const int lane = threadIdx.x;
    const uint32_t * w_row2 = reinterpret_cast<const uint32_t *>(W + static_cast<size_t>(row) * K);
    const float2 *   x_row2 = reinterpret_cast<const float2 *>(X + static_cast<size_t>(t) * K);
    const int K2 = K >> 1;

    float acc = 0.0f;
    for (int k = lane; k < K2; k += 32) {
        uint32_t packed = __ldg(&w_row2[k]);
        float2   xv     = x_row2[k];
        float w0 = __uint_as_float((packed & 0xffffu) << 16);
        float w1 = __uint_as_float(packed & 0xffff0000u);
        acc += w0 * xv.x + w1 * xv.y;
    }
    acc = warp_sum(acc);
    if (lane == 0) Y[static_cast<size_t>(t) * N + row] = acc;
}

__global__ void gemv_f32_kernel(
    const float * __restrict__ W,
    const float * __restrict__ X,
    float *       __restrict__ Y,
    int K, int N, int T) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int t   = blockIdx.y;
    if (row >= N || t >= T) return;

    const int lane = threadIdx.x;
    const float4 * w_row4 = reinterpret_cast<const float4 *>(W + static_cast<size_t>(row) * K);
    const float4 * x_row4 = reinterpret_cast<const float4 *>(X + static_cast<size_t>(t) * K);
    const int K4 = K >> 2;

    float acc = 0.0f;
    for (int k = lane; k < K4; k += 32) {
        float4 wv = __ldg(&w_row4[k]);
        float4 xv = x_row4[k];
        acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) Y[static_cast<size_t>(t) * N + row] = acc;
}

// Fast shared-memory F32 GEMV for T=1 and T=2 (halves router DRAM reads on MTP T=2!)
template <bool T2>
__global__ void gemv_f32_smem_kernel(
    const float * __restrict__ W,
    const float * __restrict__ X,
    float *       __restrict__ Y,
    int K, int N) {

    extern __shared__ float4 s_x4[];
    const int K4   = K >> 2;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;
    const int tot4 = T2 ? (2 * K4) : K4;
    const float4 * gx4 = reinterpret_cast<const float4 *>(X);
    for (int i = tid; i < tot4; i += nthr) s_x4[i] = gx4[i];
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    const int lane = threadIdx.x;
    const float4 * w_row4 = reinterpret_cast<const float4 *>(W + static_cast<size_t>(row) * K);
    const float4 * x0 = s_x4;
    const float4 * x1 = T2 ? (s_x4 + K4) : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int k = lane; k < K4; k += 32) {
        const float4 wv = __ldg(&w_row4[k]);
        const float4 a  = x0[k];
        acc0 += wv.x * a.x + wv.y * a.y + wv.z * a.z + wv.w * a.w;
        if constexpr (T2) {
            const float4 b = x1[k];
            acc1 += wv.x * b.x + wv.y * b.y + wv.z * b.z + wv.w * b.w;
        }
    }
    acc0 = warp_sum(acc0);
    if constexpr (T2) acc1 = warp_sum(acc1);
    if (lane == 0) {
        Y[row] = acc0;
        if constexpr (T2) Y[N + row] = acc1;
    }
}

// High-occupancy F32 GEMV for small-N projections (ssm_alpha/beta N=48, ffn_gate_inp_shexp N=1):
// 1 CTA of 256 threads (8 warps) per (row, t) with 128-bit float4 loads!
__global__ void gemv_f32_wide_kernel(
    const float * __restrict__ W,
    const float * __restrict__ X,
    float *       __restrict__ Y,
    int K, int N) {

    const int row  = blockIdx.x;
    const int t    = blockIdx.y;
    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float4 * w_row4 = reinterpret_cast<const float4 *>(W + static_cast<size_t>(row) * K);
    const float4 * x_row4 = reinterpret_cast<const float4 *>(X + static_cast<size_t>(t) * K);
    const int K4 = K >> 2;

    float acc = 0.0f;
    for (int k = tid; k < K4; k += 256) {
        float4 wv = __ldg(&w_row4[k]);
        float4 xv = x_row4[k];
        acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
    acc = warp_sum(acc);
    __shared__ float s_red[8];
    if (lane == 0) s_red[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? s_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) Y[static_cast<size_t>(t) * N + row] = tot;
    }
}

// Split-K F32 GEMV for ultra-narrow, huge-K projections (hc_*_inject: K=10240, N=4 — 96x/tok on Qwen3.8!):
// Launches N * SPLITS CTAs of 256 threads -> 32x higher SM occupancy!
template <int SPLITS>
__global__ void gemv_f32_splitk_kernel(
    const float * __restrict__ W,
    const float * __restrict__ X,
    float *       __restrict__ part, // [T, N, SPLITS]
    int K, int N) {

    const int row  = blockIdx.x;
    const int s    = blockIdx.y;
    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int K4    = K >> 2;
    const int chunk = (K4 + SPLITS - 1) / SPLITS;
    const int k0    = s * chunk;
    const int k1    = (k0 + chunk < K4) ? (k0 + chunk) : K4;

    const float4 * w_row4 = reinterpret_cast<const float4 *>(W + static_cast<size_t>(row) * K);
    const float4 * x_row4 = reinterpret_cast<const float4 *>(X + static_cast<size_t>(t) * K);

    float acc = 0.0f;
    for (int k = k0 + tid; k < k1; k += 256) {
        float4 wv = __ldg(&w_row4[k]);
        float4 xv = x_row4[k];
        acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
    acc = warp_sum(acc);
    __shared__ float s_red[8];
    if (lane == 0) s_red[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? s_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) part[(static_cast<size_t>(t) * N + row) * SPLITS + s] = tot;
    }
}

template <int SPLITS>
__global__ void gemv_f32_splitk_reduce_kernel(
    const float * __restrict__ part, float * __restrict__ Y, int N, int T) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * T) return;
    const float * p = part + static_cast<size_t>(idx) * SPLITS;
    float sum = 0.0f;
    #pragma unroll
    for (int s = 0; s < SPLITS; ++s) sum += p[s];
    Y[idx] = sum;
}

__global__ void bf16_to_f16_kernel(const uint16_t * __restrict__ src, __half * __restrict__ dst, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float f = __uint_as_float(static_cast<uint32_t>(__ldg(&src[i])) << 16);
        dst[i] = __float2half_rn(f);
    }
}

__global__ void bf16_to_f32_kernel(const uint16_t * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __uint_as_float(static_cast<uint32_t>(__ldg(&src[i])) << 16);
    }
}

__global__ void f16_to_f32_kernel(const __half * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(__ldg(&src[i]));
    }
}

__global__ void dequant_q8_0_soa_to_f16_kernel(
    const int8_t *   __restrict__ src_qs,
    const uint16_t * __restrict__ src_sc,
    __half *         __restrict__ dst,
    int64_t n_blocks) {

    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (b >= n_blocks) return;
    const int lane = threadIdx.x;
    __half dh;
    *reinterpret_cast<uint16_t *>(&dh) = __ldg(&src_sc[b]);
    const float d = __half2float(dh);
    if (lane < 16) {
        const uint16_t q2 = __ldg(&reinterpret_cast<const uint16_t *>(src_qs)[b * 16 + lane]);
        const float v0 = d * static_cast<float>(static_cast<int8_t>(q2 & 0xFFu));
        const float v1 = d * static_cast<float>(static_cast<int8_t>(q2 >> 8));
        reinterpret_cast<__half2 *>(dst)[b * 16 + lane] = __float22half2_rn(make_float2(v0, v1));
    }
}

__global__ void dequant_q8_0_soa_to_f32_kernel(
    const int8_t *   __restrict__ src_qs,
    const uint16_t * __restrict__ src_sc,
    float *          __restrict__ dst,
    int64_t n_blocks) {

    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (b >= n_blocks) return;
    const int lane = threadIdx.x;
    __half dh;
    *reinterpret_cast<uint16_t *>(&dh) = __ldg(&src_sc[b]);
    const float d = __half2float(dh);
    dst[b * 32 + lane] = d * static_cast<float>(__ldg(&src_qs[b * 32 + lane]));
}

__global__ void f32_to_f16_kernel(const float * __restrict__ src, __half * __restrict__ dst, int64_t n) {
    const int64_t i4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t idx = i4 * 4;
    if (idx + 3 < n) {
        const float4 v = reinterpret_cast<const float4 *>(src)[i4];
        const __half2 h0 = __float22half2_rn(make_float2(v.x, v.y));
        const __half2 h1 = __float22half2_rn(make_float2(v.z, v.w));
        uint2 pk;
        pk.x = *reinterpret_cast<const uint32_t *>(&h0);
        pk.y = *reinterpret_cast<const uint32_t *>(&h1);
        reinterpret_cast<uint2 *>(dst)[i4] = pk;
    } else {
        for (int64_t i = idx; i < n; ++i) dst[i] = __float2half_rn(src[i]);
    }
}

__global__ void argmax_f32_batch_kernel(const float * __restrict__ x, int32_t * __restrict__ out_idx, int n) {
    const int t = blockIdx.x;
    const float * xr = x + static_cast<size_t>(t) * n;
    __shared__ float   s_val[32];
    __shared__ int32_t s_idx[32];

    float   best_v = -FLT_MAX;
    int32_t best_i = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float   ov = __shfl_down_sync(0xffffffffu, best_v, offset);
        int32_t oi = __shfl_down_sync(0xffffffffu, best_i, offset);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) { s_val[warp] = best_v; s_idx[warp] = best_i; }
    __syncthreads();

    if (warp == 0) {
        const int nwarps = blockDim.x >> 5;
        best_v = (lane < nwarps) ? s_val[lane] : -FLT_MAX;
        best_i = (lane < nwarps) ? s_idx[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float   ov = __shfl_down_sync(0xffffffffu, best_v, offset);
            int32_t oi = __shfl_down_sync(0xffffffffu, best_i, offset);
            if (ov > best_v) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) out_idx[t] = best_i;
    }
}

} // namespace

void launch_quantize_q8_1(const float * d_x, void * d_xq8, int K, int T, cudaStream_t stream) {
    const int total_xb = T * (K / 32);
    if (total_xb <= 0) return;
    constexpr int q_wpb = 8;
    quantize_q8_1_kernel<<<(total_xb + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(
        d_x, static_cast<BlockQ8_1_Act *>(d_xq8), total_xb);
}

void launch_linear_q8_0_prequant(
    const DeviceTensor & W,
    const void *         d_xq8,
    float *              d_y,
    int                  T,
    cudaStream_t         stream) {

    const int K_blocks = static_cast<int>(W.ne[0]) / 32;
    const int N        = static_cast<int>(W.ne[1]);
    const int wpb      = (N <= 640) ? 4 : 8;
    dim3 block(32, wpb);
    if (T == 2 && K_blocks <= 256) {
        dim3 grid((N + wpb - 1) / wpb, 1);
        const size_t smem = static_cast<size_t>(2 * K_blocks) * sizeof(BlockQ8_1_Act);
        gemv_q8_0_soa_ldg128_t2_kernel<<<grid, block, smem, stream>>>(
            static_cast<const int4 *>(W.ptr),
            static_cast<const uint16_t *>(W.scales_ptr),
            static_cast<const BlockQ8_1_Act *>(d_xq8),
            d_y, K_blocks, N);
        return;
    }
    dim3 grid((N + wpb - 1) / wpb, T);
    const size_t smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
    gemv_q8_0_soa_ldg128_kernel<<<grid, block, smem, stream>>>(
        static_cast<const int4 *>(W.ptr),
        static_cast<const uint16_t *>(W.scales_ptr),
        static_cast<const BlockQ8_1_Act *>(d_xq8),
        d_y, K_blocks, N, T);
}

void launch_linear(
    cublasHandle_t       cublas,
    const DeviceTensor & W,
    const float *        d_x,
    float *              d_y,
    int                  T,
    void *               d_scratch,
    size_t               scratch_bytes,
    cudaStream_t         stream) {

    const int K = static_cast<int>(W.ne[0]);
    const int N = static_cast<int>(W.ne[1]);

    const size_t x_f16_bytes = static_cast<size_t>(T) * K * sizeof(__half);
    const size_t align_x = (x_f16_bytes + 255) & ~size_t(255);
    if (cublas != nullptr && T >= 16 && W.dtype == DType::Q8_0 && W.is_soa) {
        const bool use_tc = device_has_tensor_cores(W.device);
        if (use_tc && align_x + static_cast<size_t>(K) * sizeof(__half) <= scratch_bytes) {
            const size_t rem = scratch_bytes - align_x;
            const int chunk_n = std::max(1, std::min(N, static_cast<int>(rem / (static_cast<size_t>(K) * sizeof(__half)))));
            auto * d_xf16 = static_cast<__half *>(d_scratch);
            auto * d_wf16 = reinterpret_cast<__half *>(static_cast<uint8_t *>(d_scratch) + align_x);

            const int64_t nx = static_cast<int64_t>(T) * K;
            const int64_t nx4 = (nx + 3) >> 2;
            f32_to_f16_kernel<<<(nx4 + 255) / 256, 256, 0, stream>>>(d_x, d_xf16, nx);

            const float alpha = 1.0f, beta = 0.0f;
            cublasSetStream(cublas, stream);
            const int nb_row = K / 32;
            constexpr int wpb = 8;

            for (int n0 = 0; n0 < N; n0 += chunk_n) {
                const int cn = std::min(chunk_n, N - n0);
                const int64_t b0 = static_cast<int64_t>(n0) * nb_row;
                const int64_t n_blocks = static_cast<int64_t>(cn) * nb_row;
                dequant_q8_0_soa_to_f16_kernel<<<(n_blocks + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
                    static_cast<const int8_t *>(W.ptr) + b0 * 32,
                    static_cast<const uint16_t *>(W.scales_ptr) + b0,
                    d_wf16, n_blocks);

                cublasGemmEx(
                    cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    cn, T, K,
                    &alpha,
                    d_wf16, CUDA_R_16F, K,
                    d_xf16, CUDA_R_16F, K,
                    &beta,
                    d_y + n0, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
            return;
        }
        if (!use_tc && static_cast<size_t>(K) * sizeof(float) <= scratch_bytes) {
            const int chunk_n = std::max(1, std::min(N, static_cast<int>(scratch_bytes / (static_cast<size_t>(K) * sizeof(float)))));
            auto * d_wf32 = static_cast<float *>(d_scratch);
            const float alpha = 1.0f, beta = 0.0f;
            cublasSetStream(cublas, stream);
            const int nb_row = K / 32;
            constexpr int wpb = 8;

            for (int n0 = 0; n0 < N; n0 += chunk_n) {
                const int cn = std::min(chunk_n, N - n0);
                const int64_t b0 = static_cast<int64_t>(n0) * nb_row;
                const int64_t n_blocks = static_cast<int64_t>(cn) * nb_row;
                dequant_q8_0_soa_to_f32_kernel<<<(n_blocks + wpb - 1) / wpb, dim3(32, wpb), 0, stream>>>(
                    static_cast<const int8_t *>(W.ptr) + b0 * 32,
                    static_cast<const uint16_t *>(W.scales_ptr) + b0,
                    d_wf32, n_blocks);

                cublasSgemm(
                    cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    cn, T, K,
                    &alpha,
                    d_wf32, K,
                    d_x, K,
                    &beta,
                    d_y + n0, N);
            }
            return;
        }
    }

    if (is_kquant_dtype(W.dtype)) {
        launch_linear_kquant(cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream);
        return;
    }

    if (W.dtype == DType::Q8_0) {
        launch_quantize_q8_1(d_x, d_scratch, K, T, stream);
        launch_linear_q8_0_prequant(W, d_scratch, d_y, T, stream);
        return;
    }

    if (W.dtype == DType::F32 && T >= 16 && cublas != nullptr) {
        const float alpha = 1.0f, beta = 0.0f;
        cublasSetStream(cublas, stream);
        cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, T, K,
                    &alpha,
                    static_cast<const float *>(W.ptr), K,
                    d_x, K,
                    &beta,
                    d_y, N);
        return;
    }

    if (W.dtype == DType::BF16 && T >= 16 && cublas != nullptr) {
        const bool use_tc = device_has_tensor_cores(W.device);
        if (use_tc && align_x + static_cast<size_t>(N) * K * sizeof(__half) <= scratch_bytes) {
            auto * d_xf16 = static_cast<__half *>(d_scratch);
            auto * d_wf16 = reinterpret_cast<__half *>(static_cast<uint8_t *>(d_scratch) + align_x);
            const int64_t nx = static_cast<int64_t>(T) * K;
            const int64_t nw = static_cast<int64_t>(N) * K;
            f32_to_f16_kernel<<<(nx + 255) / 256, 256, 0, stream>>>(d_x, d_xf16, nx);
            bf16_to_f16_kernel<<<(nw + 255) / 256, 256, 0, stream>>>(static_cast<const uint16_t *>(W.ptr), d_wf16, nw);
            const float alpha = 1.0f, beta = 0.0f;
            cublasSetStream(cublas, stream);
            cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K,
                         &alpha, d_wf16, CUDA_R_16F, K, d_xf16, CUDA_R_16F, K,
                         &beta, d_y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            return;
        }
        if (!use_tc && static_cast<size_t>(K) * sizeof(float) <= scratch_bytes) {
            const int chunk_n = std::max(1, std::min(N, static_cast<int>(scratch_bytes / (static_cast<size_t>(K) * sizeof(float)))));
            auto * d_wf32 = static_cast<float *>(d_scratch);
            const float alpha = 1.0f, beta = 0.0f;
            cublasSetStream(cublas, stream);
            for (int n0 = 0; n0 < N; n0 += chunk_n) {
                const int cn = std::min(chunk_n, N - n0);
                const int64_t nw = static_cast<int64_t>(cn) * K;
                bf16_to_f32_kernel<<<(nw + 255) / 256, 256, 0, stream>>>(
                    static_cast<const uint16_t *>(W.ptr) + static_cast<size_t>(n0) * K, d_wf32, nw);
                cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, cn, T, K,
                            &alpha, d_wf32, K, d_x, K, &beta, d_y + n0, N);
            }
            return;
        }
    }

    if (W.dtype == DType::F16 && T >= 16 && cublas != nullptr) {
        const bool use_tc = device_has_tensor_cores(W.device);
        if (use_tc && align_x <= scratch_bytes) {
            auto * d_xf16 = static_cast<__half *>(d_scratch);
            const int64_t nx = static_cast<int64_t>(T) * K;
            const int64_t nx4 = (nx + 3) >> 2;
            f32_to_f16_kernel<<<(nx4 + 255) / 256, 256, 0, stream>>>(d_x, d_xf16, nx);
            const float alpha = 1.0f, beta = 0.0f;
            cublasSetStream(cublas, stream);
            cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K,
                         &alpha, W.ptr, CUDA_R_16F, K, d_xf16, CUDA_R_16F, K,
                         &beta, d_y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            return;
        }
        if (!use_tc && static_cast<size_t>(K) * sizeof(float) <= scratch_bytes) {
            const int chunk_n = std::max(1, std::min(N, static_cast<int>(scratch_bytes / (static_cast<size_t>(K) * sizeof(float)))));
            auto * d_wf32 = static_cast<float *>(d_scratch);
            const float alpha = 1.0f, beta = 0.0f;
            cublasSetStream(cublas, stream);
            for (int n0 = 0; n0 < N; n0 += chunk_n) {
                const int cn = std::min(chunk_n, N - n0);
                const int64_t nw = static_cast<int64_t>(cn) * K;
                f16_to_f32_kernel<<<(nw + 255) / 256, 256, 0, stream>>>(
                    static_cast<const __half *>(W.ptr) + static_cast<size_t>(n0) * K, d_wf32, nw);
                cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, cn, T, K,
                            &alpha, d_wf32, K, d_x, K, &beta, d_y + n0, N);
            }
            return;
        }
    }

    if (W.dtype == DType::F32 && T <= 4 && N <= 64 && K >= 512 && (K & 3) == 0) {
        if (N <= 8 && K >= 4096 && d_scratch != nullptr &&
            scratch_bytes >= static_cast<size_t>(T) * N * 8 * sizeof(float)) {
            constexpr int S = 8;
            auto * part = static_cast<float *>(d_scratch);
            gemv_f32_splitk_kernel<S><<<dim3(N, S, T), 256, 0, stream>>>(
                static_cast<const float *>(W.ptr), d_x, part, K, N);
            gemv_f32_splitk_reduce_kernel<S><<<(N * T + 31) / 32, 32, 0, stream>>>(part, d_y, N, T);
            return;
        }
        gemv_f32_wide_kernel<<<dim3(N, T), 256, 0, stream>>>(
            static_cast<const float *>(W.ptr), d_x, d_y, K, N);
        return;
    }

    if (W.dtype == DType::F32 && (T == 1 || T == 2) && (K & 3) == 0 && K <= 4096) {
        const int wpb_f = (N <= 512) ? 4 : 8;
        const size_t smem = static_cast<size_t>(T) * K * sizeof(float);
        if (T == 2) {
            gemv_f32_smem_kernel<true><<<(N + wpb_f - 1) / wpb_f, dim3(32, wpb_f), smem, stream>>>(
                static_cast<const float *>(W.ptr), d_x, d_y, K, N);
        } else {
            gemv_f32_smem_kernel<false><<<(N + wpb_f - 1) / wpb_f, dim3(32, wpb_f), smem, stream>>>(
                static_cast<const float *>(W.ptr), d_x, d_y, K, N);
        }
        return;
    }

    constexpr int warps_per_block = 4;
    dim3 block(32, warps_per_block);
    dim3 grid((N + warps_per_block - 1) / warps_per_block, T);

    if (W.dtype == DType::F16) {
        gemv_f16_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __half *>(W.ptr), d_x, d_y, K, N, T);
    } else if (W.dtype == DType::BF16) {
        gemv_bf16_kernel<<<grid, block, 0, stream>>>(
            static_cast<const uint16_t *>(W.ptr), d_x, d_y, K, N, T);
    } else if (W.dtype == DType::F32) {
        gemv_f32_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float *>(W.ptr), d_x, d_y, K, N, T);
    } else {
        throw std::runtime_error("launch_linear: unsupported weight dtype");
    }
}

void launch_argmax_f32(const float * d_x, int32_t * d_out_idx, int n, cudaStream_t stream) {
    argmax_f32_batch_kernel<<<1, 512, 0, stream>>>(d_x, d_out_idx, n);
}

void launch_argmax_f32_batch(const float * d_x, int32_t * d_out_idx, int n, int T, cudaStream_t stream) {
    argmax_f32_batch_kernel<<<T, 512, 0, stream>>>(d_x, d_out_idx, n);
}

} // namespace rookery
