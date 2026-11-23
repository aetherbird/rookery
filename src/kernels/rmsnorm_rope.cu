#include "kernels/kernels.h"

#include <cuda_bf16.h>
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

__global__ void embd_lookup_q8_0_soa_kernel(
    const int8_t *   __restrict__ qs,
    const uint16_t * __restrict__ scales,
    const int32_t *  __restrict__ tokens,
    float *          __restrict__ out,
    int n_embd, int hc_mult, int n_tokens) {

    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int tok = tokens[t];
    const int n_blocks = n_embd / 32;
    const int8_t *   q_row = qs     + static_cast<size_t>(tok) * n_embd;
    const uint16_t * s_row = scales + static_cast<size_t>(tok) * n_blocks;

    for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) {
        const int b = idx >> 5;
        __half dh;
        *reinterpret_cast<uint16_t *>(&dh) = __ldg(&s_row[b]);
        const float val = __half2float(dh) * static_cast<float>(__ldg(&q_row[idx]));
        for (int h = 0; h < hc_mult; ++h) {
            out[(static_cast<size_t>(t) * hc_mult + h) * n_embd + idx] = val;
        }
    }
}

__global__ void embd_lookup_q8_0_kernel(
    const BlockQ8_0 * __restrict__ table,
    const int32_t *   __restrict__ tokens,
    float *           __restrict__ out,
    int n_embd, int hc_mult, int n_tokens) {

    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int tok = tokens[t];
    const int n_blocks = n_embd / 32;
    const BlockQ8_0 * row = table + static_cast<size_t>(tok) * n_blocks;

    for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) {
        const int b = idx >> 5;
        const int j = idx & 31;
        __half dh;
        *reinterpret_cast<uint16_t *>(&dh) = row[b].d;
        const float val = __half2float(dh) * static_cast<float>(row[b].qs[j]);
        for (int h = 0; h < hc_mult; ++h) {
            out[(static_cast<size_t>(t) * hc_mult + h) * n_embd + idx] = val;
        }
    }
}

__global__ void embd_lookup_f32_kernel(
    const float *   __restrict__ table,
    const int32_t * __restrict__ tokens,
    float *         __restrict__ out,
    int n_embd, int hc_mult, int n_tokens) {

    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int tok = tokens[t];
    const float * row = table + static_cast<size_t>(tok) * n_embd;

    for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) {
        const float val = row[idx];
        for (int h = 0; h < hc_mult; ++h) {
            out[(static_cast<size_t>(t) * hc_mult + h) * n_embd + idx] = val;
        }
    }
}

__global__ void embd_lookup_f16_kernel(
    const __half *  __restrict__ table,
    const int32_t * __restrict__ tokens,
    float *         __restrict__ out,
    int n_embd, int hc_mult, int n_tokens) {

    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int tok = tokens[t];
    const __half * row = table + static_cast<size_t>(tok) * n_embd;

    for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) {
        const float val = __half2float(row[idx]);
        for (int h = 0; h < hc_mult; ++h) {
            out[(static_cast<size_t>(t) * hc_mult + h) * n_embd + idx] = val;
        }
    }
}

__global__ void rmsnorm_kernel(
    const float * __restrict__ x,
    const float * __restrict__ w,
    float *       __restrict__ out,
    int dim, float eps) {

    const int row = blockIdx.x;
    const float * xr = x + static_cast<size_t>(row) * dim;
    float *       yr = out + static_cast<size_t>(row) * dim;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xr[i];
        ss += v * v;
    }

    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    ss = warp_sum(ss);
    if (lane == 0) smem[warp] = ss;
    __syncthreads();

    float total = 0.0f;
    if (warp == 0) {
        total = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) smem[0] = rsqrtf(total / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    const float inv_rms = smem[0];

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xr[i] * inv_rms;
        yr[i] = w ? (v * w[i]) : v;
    }
}

// Per-head RMSNorm + NeoX (half-split) RoPE in-place.
// One warp (32 threads) per (token, head) when head_dim <= 256 (Maxwell: head_dim = 128).
__global__ void qk_norm_rope_neox_kernel(
    float *         __restrict__ x,
    const float *   __restrict__ norm_w,
    const int32_t * __restrict__ pos,
    int n_tokens, int n_heads, int head_dim,
    float rms_eps, float freq_base) {

    const int global_head = blockIdx.x * blockDim.y + threadIdx.y;
    if (global_head >= n_tokens * n_heads) return;

    const int t = global_head / n_heads;
    const int p = pos[t];
    float * hptr = x + static_cast<size_t>(global_head) * head_dim;

    const int half_d = head_dim >> 1;
    const int lane   = threadIdx.x; // 0..31

    // 1. RMSNorm across head_dim
    float ss = 0.0f;
    for (int i = lane; i < head_dim; i += 32) {
        float v = hptr[i];
        ss += v * v;
    }
    ss = warp_sum(ss);
    const float inv_rms = __shfl_sync(0xffffffffu, rsqrtf(ss / static_cast<float>(head_dim) + rms_eps), 0);

    // 2. Apply norm weight + NeoX half-split RoPE: pairs (i, i + half_d)
    for (int i = lane; i < half_d; i += 32) {
        float x0 = hptr[i]          * inv_rms * (norm_w ? norm_w[i]          : 1.0f);
        float x1 = hptr[i + half_d] * inv_rms * (norm_w ? norm_w[i + half_d] : 1.0f);

        const float exponent = -2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
        const float theta = static_cast<float>(p) * powf(freq_base, exponent);
        float c, s;
        sincosf(theta, &s, &c);

        hptr[i]          = x0 * c - x1 * s;
        hptr[i + half_d] = x0 * s + x1 * c;
    }
}

__global__ void silu_mul_kernel(
    const float * __restrict__ gate,
    const float * __restrict__ up,
    float *       __restrict__ out,
    int64_t n) {

    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void add_inplace_kernel(
    float *       __restrict__ dst,
    const float * __restrict__ src,
    int64_t n) {

    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// MTP input prep: for each (t, s) in [T, hc], writes:
//   eh_in[(t*hc + s) * 2*n_embd + 0..n_embd-1]        = rmsnorm(e_embd[t], enorm_w)
//   eh_in[(t*hc + s) * 2*n_embd + n_embd..2*n_embd-1] = rmsnorm(h_state[t, s], hnorm_w)
__global__ void mtp_eh_concat_kernel(
    const float * __restrict__ e_embd,
    const float * __restrict__ h_state,
    const float * __restrict__ enorm_w,
    const float * __restrict__ hnorm_w,
    float *       __restrict__ eh_in,
    int n_embd, int hc_mult, float rms_eps) {

    const int ts = blockIdx.x; // 0 .. T*hc - 1
    const int t  = ts / hc_mult;
    const int s  = ts % hc_mult;

    const float * er = e_embd  + static_cast<size_t>(t) * n_embd;
    const float * hr = h_state + (static_cast<size_t>(t) * hc_mult + s) * n_embd;
    float *       ye = eh_in   + static_cast<size_t>(ts) * (2 * n_embd);
    float *       yh = ye      + n_embd;

    float sse = 0.0f, ssh = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float ev = er[i], hv = hr[i];
        sse += ev * ev;
        ssh += hv * hv;
    }

    __shared__ float s_e[8], s_h[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    sse = warp_sum(sse);
    ssh = warp_sum(ssh);
    if (lane == 0) { s_e[warp] = sse; s_h[warp] = ssh; }
    __syncthreads();

    if (warp == 0) {
        float te = (lane < 8) ? s_e[lane] : 0.0f;
        float th = (lane < 8) ? s_h[lane] : 0.0f;
        te = warp_sum(te);
        th = warp_sum(th);
        if (lane == 0) {
            s_e[0] = rsqrtf(te / static_cast<float>(n_embd) + rms_eps);
            s_h[0] = rsqrtf(th / static_cast<float>(n_embd) + rms_eps);
        }
    }
    __syncthreads();

    const float inv_e = s_e[0], inv_h = s_h[0];
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        ye[i] = er[i] * inv_e * (enorm_w ? enorm_w[i] : 1.0f);
        yh[i] = hr[i] * inv_h * (hnorm_w ? hnorm_w[i] : 1.0f);
    }
}

} // namespace

void launch_embedding_lookup(
    const DeviceTensor & tok_embd,
    const int32_t * d_tokens,
    float * d_out,
    int n_embd, int hc_mult, int n_tokens,
    cudaStream_t stream) {

    constexpr int threads = 256;
    if (is_kquant_dtype(tok_embd.dtype)) {
        launch_embedding_kquant(tok_embd, d_tokens, d_out, n_embd, hc_mult, n_tokens, stream);
        return;
    }
    if (tok_embd.dtype == DType::Q8_0) {
        if (tok_embd.is_soa) {
            embd_lookup_q8_0_soa_kernel<<<n_tokens, threads, 0, stream>>>(
                static_cast<const int8_t *>(tok_embd.ptr),
                static_cast<const uint16_t *>(tok_embd.scales_ptr),
                d_tokens, d_out, n_embd, hc_mult, n_tokens);
        } else {
            embd_lookup_q8_0_kernel<<<n_tokens, threads, 0, stream>>>(
                static_cast<const BlockQ8_0 *>(tok_embd.ptr), d_tokens, d_out, n_embd, hc_mult, n_tokens);
        }
    } else if (tok_embd.dtype == DType::F32) {
        embd_lookup_f32_kernel<<<n_tokens, threads, 0, stream>>>(
            static_cast<const float *>(tok_embd.ptr), d_tokens, d_out, n_embd, hc_mult, n_tokens);
    } else if (tok_embd.dtype == DType::F16) {
        embd_lookup_f16_kernel<<<n_tokens, threads, 0, stream>>>(
            static_cast<const __half *>(tok_embd.ptr), d_tokens, d_out, n_embd, hc_mult, n_tokens);
    } else {
        throw std::runtime_error("Unsupported token_embd dtype");
    }
}

void launch_rmsnorm(
    const float * d_x, const float * d_w, float * d_out,
    int dim, int n_rows, float eps, cudaStream_t stream) {
    constexpr int threads = 256;
    rmsnorm_kernel<<<n_rows, threads, 0, stream>>>(d_x, d_w, d_out, dim, eps);
}

void launch_qk_norm_rope_neox(
    float * d_x, const float * d_norm_w, const int32_t * d_pos,
    int n_tokens, int n_heads, int head_dim,
    float rms_eps, float rope_freq_base, cudaStream_t stream) {

    const int total_heads = n_tokens * n_heads;
    constexpr int warps_per_block = 4;
    dim3 block(32, warps_per_block);
    dim3 grid((total_heads + warps_per_block - 1) / warps_per_block);
    qk_norm_rope_neox_kernel<<<grid, block, 0, stream>>>(
        d_x, d_norm_w, d_pos, n_tokens, n_heads, head_dim, rms_eps, rope_freq_base);
}

void launch_silu_mul(const float * d_gate, const float * d_up, float * d_out, int64_t n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(d_gate, d_up, d_out, n);
}

void launch_add_inplace(float * d_dst, const float * d_src, int64_t n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    add_inplace_kernel<<<blocks, threads, 0, stream>>>(d_dst, d_src, n);
}

void launch_mtp_eh_concat(
    const float * d_e_embd, const float * d_h_state,
    const float * d_enorm_w, const float * d_hnorm_w,
    float * d_eh_in, int n_embd, int hc_mult, int n_tokens,
    float rms_eps, cudaStream_t stream) {

    mtp_eh_concat_kernel<<<n_tokens * hc_mult, 256, 0, stream>>>(
        d_e_embd, d_h_state, d_enorm_w, d_hnorm_w, d_eh_in, n_embd, hc_mult, rms_eps);
}

} // namespace rookery
