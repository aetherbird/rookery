#include "kernels/kernels.h"

#include <cuda_fp16.h>
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

// ============================================================================
// 1. Low-Rank mHC (Hyper-Connections) Kernels
// ============================================================================
__global__ void hc_init_kernel(const float * __restrict__ embd, float * __restrict__ res_hc, int H, int hc) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= H) return;
    const float v = embd[static_cast<size_t>(t) * H + d];
    float * dst = res_hc + static_cast<size_t>(t) * hc * H + d;
    for (int s = 0; s < hc; ++s) dst[static_cast<size_t>(s) * H] = v;
}

// Grouped RMSNorm over each stream s in [0..hc-1]:
// x: [T, hc, H], w: [hc, H], out: [T, hc, H]
__global__ void hc_grouped_rmsnorm_kernel(
    const float * __restrict__ x,
    const float * __restrict__ w,
    float *       __restrict__ out,
    int H, int hc, float eps) {

    const int s    = blockIdx.x;
    const int t    = blockIdx.y;
    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const size_t off = (static_cast<size_t>(t) * hc + s) * H;
    const float * xr = x + off;
    const float * wr = w + static_cast<size_t>(s) * H;
    float *       orow = out + off;

    float ss = 0.0f;
    for (int d = tid; d < H; d += 256) {
        const float v = xr[d];
        ss += v * v;
    }
    ss = warp_sum(ss);
    __shared__ float s_red[8];
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? s_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) s_red[0] = rsqrtf(tot / static_cast<float>(H) + eps);
    }
    __syncthreads();
    const float inv = s_red[0];

    for (int d = tid; d < H; d += 256) {
        orow[d] = xr[d] * inv * wr[d];
    }
}

__global__ void scale_silu_inplace_kernel(float * __restrict__ x, float scale, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] * scale;
    x[i] = v / (1.0f + expf(-v));
}

// mixed[t, d] = (1 / hc) * sum_{s=0..hc-1} xn[t, s, d] * sigmoid(gate[t, s, d])
__global__ void hc_pre_gated_mean_kernel(
    const float * __restrict__ xn,
    const float * __restrict__ gate,
    float *       __restrict__ mixed,
    int H, int hc) {

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= H) return;

    const size_t base = static_cast<size_t>(t) * hc * H + d;
    float acc = 0.0f;
    for (int s = 0; s < hc; ++s) {
        const size_t idx = base + static_cast<size_t>(s) * H;
        const float g = gate[idx];
        acc += xn[idx] / (1.0f + expf(-g));
    }
    mixed[static_cast<size_t>(t) * H + d] = acc * (1.0f / static_cast<float>(hc));
}

// res_hc[t, s, d] += block_out[t, d] * (2.0f * sigmoid(inject[t, s] * inject_scale))
__global__ void hc_combine_kernel(
    float *       __restrict__ res_hc,
    const float * __restrict__ block_out,
    const float * __restrict__ inject,
    float inject_scale, int H, int hc) {

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= H) return;

    const float b = block_out[static_cast<size_t>(t) * H + d];
    float * dst = res_hc + static_cast<size_t>(t) * hc * H + d;
    const float * inj = inject + static_cast<size_t>(t) * hc;
    for (int s = 0; s < hc; ++s) {
        const float w = 2.0f / (1.0f + expf(-(inj[s] * inject_scale)));
        dst[static_cast<size_t>(s) * H] += b * w;
    }
}

// ============================================================================
// 2. PLE n-gram Engram Kernels (blk.1)
// ============================================================================
// key: [T, hc, H], query: [T, hc, H], value: [T, H] -> gated: [T, hc, H]
__global__ void ple_gate_value_kernel(
    const float * __restrict__ key,
    const float * __restrict__ query,
    const float * __restrict__ value,
    float *       __restrict__ gated,
    int H, int hc) {

    const int s    = blockIdx.x;
    const int t    = blockIdx.y;
    const int tid  = threadIdx.x; // 0..255
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const size_t off = (static_cast<size_t>(t) * hc + s) * H;
    const float * kr = key   + off;
    const float * qr = query + off;
    const float * vr = value + static_cast<size_t>(t) * H;
    float *       gr = gated + off;

    float dot = 0.0f;
    for (int d = tid; d < H; d += 256) {
        dot += kr[d] * qr[d];
    }
    dot = warp_sum(dot);
    __shared__ float s_red[8];
    if (lane == 0) s_red[warp] = dot;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? s_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) {
            const float sc  = tot * rsqrtf(static_cast<float>(H));
            const float sgn = (sc > 0.0f) ? 1.0f : ((sc < 0.0f) ? -1.0f : 0.0f);
            const float mag = sqrtf(fmaxf(fabsf(sc), 1e-6f));
            s_red[0] = 1.0f / (1.0f + expf(-(sgn * mag)));
        }
    }
    __syncthreads();
    const float g = s_red[0];

    for (int d = tid; d < H; d += 256) {
        gr[d] = vr[d] * g;
    }
}

// Causal dilated depthwise conv1d (kern=4, dil=3 -> hist=9) + SiLU + res_hc += gated + conv_out!
// norm_in: [T, hc_dim], conv_w: [hc_dim, kern] (ggml [kern, hc_dim]), conv_state: [hist, hc_dim]
__global__ void ple_dilated_conv1d_silu_add_kernel(
    const float * __restrict__ norm_in,
    const float * __restrict__ conv_w,
    float *       __restrict__ conv_state,
    const float * __restrict__ gated,
    float *       __restrict__ res_hc,
    int hc_dim, int kern, int dil, int T) {

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hc_dim) return;

    const int hist = (kern - 1) * dil; // 9 for kern=4, dil=3
    float w[8];
    for (int k = 0; k < kern; ++k) {
        w[k] = conv_w[static_cast<size_t>(c) * kern + k];
    }

    // Load history into local ring/shift buffer (hist <= 16)
    float hbuf[16];
    for (int i = 0; i < hist; ++i) {
        hbuf[i] = conv_state[static_cast<size_t>(i) * hc_dim + c];
    }

    for (int t = 0; t < T; ++t) {
        const size_t idx = static_cast<size_t>(t) * hc_dim + c;
        const float x_cur = norm_in[idx];

        // Tap k reads (kern - 1 - k) * dil steps back!
        // k == kern - 1 reads lag 0 (x_cur); k < kern - 1 reads hbuf[hist - (kern - 1 - k)*dil]
        float acc = 0.0f;
        for (int k = 0; k < kern - 1; ++k) {
            const int lag = (kern - 1 - k) * dil;
            acc += w[k] * hbuf[hist - lag];
        }
        acc += w[kern - 1] * x_cur;

        const float conv_silu = acc / (1.0f + expf(-acc));
        res_hc[idx] += gated[idx] + conv_silu;

        // Shift history left by 1, append x_cur
        for (int i = 0; i < hist - 1; ++i) hbuf[i] = hbuf[i + 1];
        hbuf[hist - 1] = x_cur;
    }

    for (int i = 0; i < hist; ++i) {
        conv_state[static_cast<size_t>(i) * hc_dim + c] = hbuf[i];
    }
}

// ============================================================================
// 3. QSA Sparse Indexer (d=128, n_idx_h=4, kpool=4, top_k=2048)
// ============================================================================
__global__ void qsa_k_raw_write_kernel(
    const float *   __restrict__ k_raw,       // [T, 128]
    float *         __restrict__ k_raw_cache, // [max_ctx, 128]
    const int32_t * __restrict__ pos,         // [T]
    int T) {

    const int d = threadIdx.x; // 0..127
    const int t = blockIdx.x;
    if (t >= T) return;
    const int p = pos[t];
    k_raw_cache[static_cast<size_t>(p) * 128 + d] = k_raw[static_cast<size_t>(t) * 128 + d];
}

// For every token t where (pos[t] + 1) % kpool == 0, pool block b = pos[t] / kpool:
// pooled = RoPE(RMSNorm(mean(k_raw_cache[4*b..4*b+3]), k_norm_w), pos = 4*b)
__global__ void qsa_k_pool_complete_kernel(
    const float *   __restrict__ k_raw_cache,  // [max_ctx, 128]
    float *         __restrict__ k_pool_cache, // [max_ctx/4, 128]
    const float *   __restrict__ k_norm_w,     // [128]
    const int32_t * __restrict__ pos,          // [T]
    int T, int kpool, int rope_dim, float rms_eps, float rope_freq_base) {

    const int t = blockIdx.x;
    if (t >= T) return;
    const int p = pos[t];
    if (((p + 1) % kpool) != 0) return;

    const int b    = p / kpool;
    const int p0   = b * kpool;
    const int lane = threadIdx.x; // 0..31 -> 4 dims/lane

    float kv[4];
    float ss = 0.0f;
    const float inv_kp = 1.0f / static_cast<float>(kpool);
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int d = r * 32 + lane;
        float acc = 0.0f;
        for (int m = 0; m < kpool; ++m) {
            acc += k_raw_cache[static_cast<size_t>(p0 + m) * 128 + d];
        }
        kv[r] = acc * inv_kp;
        ss += kv[r] * kv[r];
    }
    ss = __shfl_sync(0xffffffffu, warp_sum(ss), 0);
    const float inv = rsqrtf(ss * (1.0f / 128.0f) + rms_eps);
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        kv[r] *= inv * k_norm_w[r * 32 + lane];
    }

    // Partial NeoX RoPE on first rope_dim=64 dims (r=0 paired with r=1) at position p0 = 4*b!
    const int half_r = rope_dim >> 1; // 32
    if (lane < half_r) {
        const float freq  = powf(rope_freq_base, -static_cast<float>(2 * lane) / static_cast<float>(rope_dim));
        const float theta = static_cast<float>(p0) * freq;
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = kv[0], x1 = kv[1];
        kv[0] = x0 * c - x1 * s;
        kv[1] = x0 * s + x1 * c;
    }

    float * dst = k_pool_cache + static_cast<size_t>(b) * 128;
    #pragma unroll
    for (int r = 0; r < 4; ++r) dst[r * 32 + lane] = kv[r];
}

// QSA indexer Q: [T, n_idx_h, 128] -> RMSNorm + partial NeoX RoPE in-place!
__global__ void qsa_q_norm_rope_kernel(
    float *         __restrict__ q,        // [T, n_idx_h, 128]
    const float *   __restrict__ q_norm_w, // [128]
    const int32_t * __restrict__ pos,      // [T]
    int n_idx_h, int rope_dim, float rms_eps, float rope_freq_base) {

    const int h    = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    if (h >= n_idx_h) return;
    const int lane = threadIdx.x; // 0..31

    float * dst = q + (static_cast<size_t>(t) * n_idx_h + h) * 128;
    float qv[4];
    float ss = 0.0f;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        qv[r] = dst[r * 32 + lane];
        ss += qv[r] * qv[r];
    }
    ss = __shfl_sync(0xffffffffu, warp_sum(ss), 0);
    const float inv = rsqrtf(ss * (1.0f / 128.0f) + rms_eps);
    #pragma unroll
    for (int r = 0; r < 4; ++r) qv[r] *= inv * q_norm_w[r * 32 + lane];

    const int half_r = rope_dim >> 1;
    if (lane < half_r) {
        const float freq  = powf(rope_freq_base, -static_cast<float>(2 * lane) / static_cast<float>(rope_dim));
        const float theta = static_cast<float>(pos[t]) * freq;
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = qv[0], x1 = qv[1];
        qv[0] = x0 * c - x1 * s;
        qv[1] = x0 * s + x1 * c;
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) dst[r * 32 + lane] = qv[r];
}

// Persistent-grid QSA pool scoring for T==1 (CUDA-Graph-safe for ALL pos from 0 to 262K!)
__global__ void qsa_score_pools_decode_kernel(
    const float *   __restrict__ q_idx,        // [1, n_idx_h, 128]
    const float *   __restrict__ k_pool_cache, // [max_pools, 128]
    const int32_t * __restrict__ pos,          // [1]
    float *         __restrict__ scores,       // [max_pools]
    int n_idx_h, int kpool, int top_k) {

    const int n_vis = (pos[0] + 1) / kpool;
    if (n_vis <= top_k / kpool) return;

    const int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int n_warps = gridDim.x * blockDim.y;
    const int lane    = threadIdx.x;
    const float inv_s = 0.08838834764831845f;

    // Load all 4 query heads into registers once (4 * 4 = 16 floats/lane)
    float4 qv[4];
    #pragma unroll
    for (int h = 0; h < 4; ++h) {
        if (h < n_idx_h) {
            qv[h] = reinterpret_cast<const float4 *>(q_idx + static_cast<size_t>(h) * 128)[lane];
        }
    }

    for (int b = warp_id; b < n_vis; b += n_warps) {
        const float4 kv = reinterpret_cast<const float4 *>(k_pool_cache + static_cast<size_t>(b) * 128)[lane];
        float total = 0.0f;
        #pragma unroll
        for (int h = 0; h < 4; ++h) {
            if (h < n_idx_h) {
                float d = warp_sum(qv[h].x * kv.x + qv[h].y * kv.y + qv[h].z * kv.z + qv[h].w * kv.w);
                if (lane == 0 && d > 0.0f) total += d * inv_s;
            }
        }
        if (lane == 0) scores[b] = total;
    }
}

// Compute lightning indexer scores: score[t, b] = sum_{h=0..n_idx_h-1} (1/sqrt(128)) * ReLU(dot(q[t, h], k_pool[b]))
__global__ void qsa_score_pools_kernel(
    const float *   __restrict__ q_idx,        // [T, n_idx_h, 128]
    const float *   __restrict__ k_pool_cache, // [max_pools, 128]
    const int32_t * __restrict__ pos,          // [T]
    float *         __restrict__ scores,       // [T, score_stride]
    int T, int n_idx_h, int kpool, int score_stride) {

    const int b    = blockIdx.x * blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    const int lane = threadIdx.x; // 0..31
    if (t >= T) return;

    const int n_vis = (pos[t] + 1) / kpool;
    if (b >= n_vis) return;

    const auto * k4 = reinterpret_cast<const float4 *>(k_pool_cache + static_cast<size_t>(b) * 128);
    const float4 kv = k4[lane];

    float total = 0.0f;
    const float inv_s = 0.08838834764831845f; // 1 / sqrt(128)
    for (int h = 0; h < n_idx_h; ++h) {
        const auto * q4 = reinterpret_cast<const float4 *>(q_idx + (static_cast<size_t>(t) * n_idx_h + h) * 128);
        const float4 qv = q4[lane];
        float d = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w);
        if (lane == 0 && d > 0.0f) total += d * inv_s;
    }
    if (lane == 0) {
        scores[static_cast<size_t>(t) * score_stride + b] = total;
    }
}

// Exact Radix-256 Top-K selection on non-negative IEEE-754 scores[t, 0..n_vis-1]!
// 1 CTA (256 threads) per token t.
__global__ void qsa_radix_select_topk_kernel(
    const float *   __restrict__ scores,       // [T, score_stride]
    const int32_t * __restrict__ pos,          // [T]
    int32_t *       __restrict__ sel_pos,      // [T, max_sel]
    int32_t *       __restrict__ sel_len,      // [T] (-1 if dense)
    int T, int kpool, int top_k, int score_stride, int max_sel) {

    const int t   = blockIdx.x;
    const int tid = threadIdx.x; // 0..255
    if (t >= T) return;

    const int p       = pos[t];
    const int n_vis   = (p + 1) / kpool;
    const int n_top   = top_k / kpool; // 512

    if (n_vis <= n_top) {
        if (tid == 0) sel_len[t] = -1; // Dense causal attention!
        return;
    }

    const float * row = scores + static_cast<size_t>(t) * score_stride;

    __shared__ uint32_t s_hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ uint32_t s_rem;
    __shared__ uint32_t s_out_cnt;

    if (tid == 0) {
        s_prefix = 0u;
        s_rem    = static_cast<uint32_t>(n_top);
    }
    __syncthreads();

    // 4 passes of 8-bit MSB-first radix selection over order-preserving uint32_t bits (score >= 0)
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - pass * 8;
        const uint32_t mask = (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
        const uint32_t pref = s_prefix;

        s_hist[tid] = 0u;
        __syncthreads();

        for (int b = tid; b < n_vis; b += 256) {
            const uint32_t bits = __float_as_uint(row[b]);
            if ((bits & mask) == pref) {
                const uint32_t bucket = (bits >> shift) & 0xFFu;
                atomicAdd(&s_hist[bucket], 1u);
            }
        }
        __syncthreads();

        if (tid == 0) {
            uint32_t acc = 0u;
            uint32_t rem = s_rem;
            int chosen = 0;
            for (int k = 255; k >= 0; --k) {
                const uint32_t c = s_hist[k];
                if (acc + c >= rem) {
                    chosen = k;
                    s_rem = rem - acc;
                    break;
                }
                acc += c;
            }
            s_prefix = pref | (static_cast<uint32_t>(chosen) << shift);
        }
        __syncthreads();
    }

    const uint32_t thresh = s_prefix;
    if (tid == 0) {
        s_out_cnt = 0u;
    }
    __syncthreads();

    int32_t * dst = sel_pos + static_cast<size_t>(t) * max_sel;

    // Pass 1: emit all strictly-greater pools (bits > thresh)
    for (int b = tid; b < n_vis; b += 256) {
        const uint32_t bits = __float_as_uint(row[b]);
        if (bits > thresh) {
            const uint32_t slot = atomicAdd(&s_out_cnt, 1u);
            if (slot < static_cast<uint32_t>(n_top)) {
                const int p0 = b * kpool;
                const size_t base = static_cast<size_t>(slot) * kpool;
                for (int m = 0; m < kpool; ++m) dst[base + m] = p0 + m;
            }
        }
    }
    __syncthreads();

    // Pass 2: deterministic tie-breaking on (bits == thresh) up to n_top (bit-exact across runs!)
    if (tid == 0) {
        int cnt = static_cast<int>(s_out_cnt);
        if (cnt < n_top) {
            for (int b = 0; b < n_vis && cnt < n_top; ++b) {
                if (__float_as_uint(row[b]) == thresh) {
                    const int p0 = b * kpool;
                    const size_t base = static_cast<size_t>(cnt++) * kpool;
                    for (int m = 0; m < kpool; ++m) dst[base + m] = p0 + m;
                }
            }
        }
        int out_len = n_top * kpool;
        for (int tp = n_vis * kpool; tp <= p; ++tp) {
            dst[out_len++] = tp;
        }
        sel_len[t] = out_len;
    }
}

// ============================================================================
// 4. GQA-Group-Shared Sparse-Index Split-K Attention (d=256, Q8_0 or F16 KV)
//    All `gqa = n_head / n_kv` Q heads (e.g. 12 warps/CTA) share ONE KV load -> 12x less KV bandwidth!
//    Attends to sel_pos[t, 0..sel_len[t]-1] if sel_len[t] >= 0, else dense 0..pos[t]!
// ============================================================================
template <bool KV_Q8_0, int SPLITS>
__global__ void gqa_attn_d256_sparse_splitk_kernel(
    const float *   __restrict__ q,
    const void *    __restrict__ k_cache,
    const void *    __restrict__ v_cache,
    const int32_t * __restrict__ pos,
    const int32_t * __restrict__ sel_pos, // [T, max_sel]
    const int32_t * __restrict__ sel_len, // [T] (-1 => dense 0..pos[t])
    float *         __restrict__ part_m,
    float *         __restrict__ part_l,
    float *         __restrict__ part_o,
    int n_head, int n_kv, int max_sel, float scale) {

    const int kvh  = blockIdx.x;
    const int s    = blockIdx.y;
    const int t    = blockIdx.z;
    const int gh   = threadIdx.y; // 0 .. gqa-1
    const int lane = threadIdx.x; // 0 .. 31
    const int gqa  = n_head / n_kv;
    const int h    = kvh * gqa + gh;

    const int slen = sel_len[t];
    const bool is_sparse = (slen >= 0);
    const int total_k = is_sparse ? slen : (pos[t] + 1);
    const int32_t * t_sel = is_sparse ? (sel_pos + static_cast<size_t>(t) * max_sel) : nullptr;

    const float * qr = q + (static_cast<size_t>(t) * n_head + h) * 256;
    float qv[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) qv[r] = qr[r * 32 + lane] * scale;

    // Shared memory tile of 4 KV positions (4 * 256 * 2 * 4 = 8 KB) loaded cooperatively by all gqa warps!
    constexpr int P_TILE = 4;
    __shared__ float s_k[P_TILE][256];
    __shared__ float s_v[P_TILE][256];

    float m_run = -FLT_MAX;
    float l_run = 0.0f;
    float o[8]  = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int chunk = (total_k + SPLITS - 1) / SPLITS;
    const int idx0  = s * chunk;
    const int idx1  = (idx0 + chunk < total_k) ? (idx0 + chunk) : total_k;

    for (int base_i = idx0; base_i < idx1; base_i += P_TILE) {
        const int n_p = (base_i + P_TILE < idx1) ? P_TILE : (idx1 - base_i);

        // First P_TILE warps cooperatively load K and V into shared memory (1 warp per position!)
        if (gh < n_p) {
            const int kp = is_sparse ? t_sel[base_i + gh] : (base_i + gh);
            if constexpr (KV_Q8_0) {
                const BlockQ8_0 * kb = static_cast<const BlockQ8_0 *>(k_cache) + (static_cast<size_t>(kp) * n_kv + kvh) * 8;
                const BlockQ8_0 * vb = static_cast<const BlockQ8_0 *>(v_cache) + (static_cast<size_t>(kp) * n_kv + kvh) * 8;
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    __half kdh, vdh;
                    *reinterpret_cast<uint16_t *>(&kdh) = __ldg(&kb[r].d);
                    *reinterpret_cast<uint16_t *>(&vdh) = __ldg(&vb[r].d);
                    s_k[gh][r * 32 + lane] = __half2float(kdh) * static_cast<float>(__ldg(&kb[r].qs[lane]));
                    s_v[gh][r * 32 + lane] = __half2float(vdh) * static_cast<float>(__ldg(&vb[r].qs[lane]));
                }
            } else {
                const __half * kr = static_cast<const __half *>(k_cache) + (static_cast<size_t>(kp) * n_kv + kvh) * 256;
                const __half * vr = static_cast<const __half *>(v_cache) + (static_cast<size_t>(kp) * n_kv + kvh) * 256;
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    s_k[gh][r * 32 + lane] = __half2float(__ldg(&kr[r * 32 + lane]));
                    s_v[gh][r * 32 + lane] = __half2float(__ldg(&vr[r * 32 + lane]));
                }
            }
        }
        __syncthreads();

        for (int pi = 0; pi < n_p; ++pi) {
            float dot = 0.0f;
            #pragma unroll
            for (int r = 0; r < 8; ++r) dot += qv[r] * s_k[pi][r * 32 + lane];
            dot = __shfl_sync(0xffffffffu, warp_sum(dot), 0);

            const float m_new = fmaxf(m_run, dot);
            const float alpha = (m_run == -FLT_MAX) ? 0.0f : expf(m_run - m_new);
            const float p_val = expf(dot - m_new);
            l_run = l_run * alpha + p_val;
            m_run = m_new;

            #pragma unroll
            for (int r = 0; r < 8; ++r) o[r] = o[r] * alpha + p_val * s_v[pi][r * 32 + lane];
        }
        __syncthreads();
    }

    const size_t slot_idx = (static_cast<size_t>(t) * n_head + h) * SPLITS + s;
    if (lane == 0) {
        part_m[slot_idx] = m_run;
        part_l[slot_idx] = l_run;
    }
    float * po = part_o + slot_idx * 256;
    #pragma unroll
    for (int r = 0; r < 8; ++r) po[r * 32 + lane] = o[r];
}

template <int SPLITS>
__global__ void gqa_attn_d256_sparse_reduce_kernel(
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
    for (int s = 0; s < SPLITS; ++s) m_max = fmaxf(m_max, part_m[base + s]);

    float l_sum = 0.0f;
    float o[8]  = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < SPLITS; ++s) {
        const float ls = part_l[base + s];
        if (ls > 0.0f) {
            const float w = expf(part_m[base + s] - m_max);
            l_sum += ls * w;
            const float * po = part_o + (base + s) * 256;
            #pragma unroll
            for (int r = 0; r < 8; ++r) o[r] += po[r * 32 + lane] * w;
        }
    }
    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    float * dst = out + (static_cast<size_t>(t) * n_head + h) * 256;
    #pragma unroll
    for (int r = 0; r < 8; ++r) dst[r * 32 + lane] = o[r] * inv_l;
}

} // namespace

void launch_hc_init(const float * d_embd, float * d_res_hc, int H, int hc, int T, cudaStream_t stream) {
    hc_init_kernel<<<dim3((H + 255) / 256, T), 256, 0, stream>>>(d_embd, d_res_hc, H, hc);
}

void launch_hc_grouped_rmsnorm(
    const float * d_x, const float * d_w, float * d_out,
    int H, int hc, int T, float eps, cudaStream_t stream) {
    hc_grouped_rmsnorm_kernel<<<dim3(hc, T), 256, 0, stream>>>(d_x, d_w, d_out, H, hc, eps);
}

void launch_scale_silu_inplace(float * d_x, float scale, int64_t n, cudaStream_t stream) {
    scale_silu_inplace_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_x, scale, n);
}

void launch_hc_pre_gated_mean(
    const float * d_xn, const float * d_gate, float * d_mixed,
    int H, int hc, int T, cudaStream_t stream) {
    hc_pre_gated_mean_kernel<<<dim3((H + 255) / 256, T), 256, 0, stream>>>(d_xn, d_gate, d_mixed, H, hc);
}

void launch_hc_combine(
    float * d_res_hc, const float * d_block_out, const float * d_inject,
    float inject_scale, int H, int hc, int T, cudaStream_t stream) {
    hc_combine_kernel<<<dim3((H + 255) / 256, T), 256, 0, stream>>>(d_res_hc, d_block_out, d_inject, inject_scale, H, hc);
}

void launch_ple_gate_value(
    const float * d_key, const float * d_query, const float * d_value,
    float * d_gated, int H, int hc, int T, cudaStream_t stream) {
    ple_gate_value_kernel<<<dim3(hc, T), 256, 0, stream>>>(d_key, d_query, d_value, d_gated, H, hc);
}

void launch_ple_dilated_conv1d_silu_add(
    const float * d_norm, const float * d_conv_w, float * d_conv_state,
    const float * d_gated, float * d_res_hc,
    int hc_dim, int kern, int dil, int T, cudaStream_t stream) {
    ple_dilated_conv1d_silu_add_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(
        d_norm, d_conv_w, d_conv_state, d_gated, d_res_hc, hc_dim, kern, dil, T);
}

void launch_qsa_k_raw_write_and_pool(
    const float * d_k_raw, float * d_k_raw_cache, float * d_k_pool_cache,
    const float * d_k_norm_w, const int32_t * d_pos,
    int T, int kpool, int rope_dim, float rms_eps, float rope_freq_base, cudaStream_t stream) {
    qsa_k_raw_write_kernel<<<T, 128, 0, stream>>>(d_k_raw, d_k_raw_cache, d_pos, T);
    qsa_k_pool_complete_kernel<<<T, 32, 0, stream>>>(
        d_k_raw_cache, d_k_pool_cache, d_k_norm_w, d_pos, T, kpool, rope_dim, rms_eps, rope_freq_base);
}

void launch_qsa_q_norm_rope(
    float * d_q, const float * d_q_norm_w, const int32_t * d_pos,
    int T, int n_idx_h, int rope_dim, float rms_eps, float rope_freq_base, cudaStream_t stream) {
    constexpr int wpb = 4;
    qsa_q_norm_rope_kernel<<<dim3((n_idx_h + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
        d_q, d_q_norm_w, d_pos, n_idx_h, rope_dim, rms_eps, rope_freq_base);
}

void launch_qsa_select_topk(
    const float * d_q_idx, const float * d_k_pool_cache, const int32_t * d_pos,
    int32_t * d_sel_pos, int32_t * d_sel_len,
    void * d_scratch, size_t scratch_bytes,
    int T, int n_idx_h, int kpool, int top_k, int kv_len_max, cudaStream_t stream) {

    const int max_sel = top_k + kpool;
    // Graph-safe T==1 persistent path (works for ANY pos in [0..max_ctx-1]!)
    if (T == 1 && kv_len_max <= 0) {
        float * d_scores = static_cast<float *>(d_scratch);
        constexpr int wpb = 4;
        qsa_score_pools_decode_kernel<<<256, dim3(32, wpb), 0, stream>>>(
            d_q_idx, d_k_pool_cache, d_pos, d_scores, n_idx_h, kpool, top_k);
        qsa_radix_select_topk_kernel<<<1, 256, 0, stream>>>(
            d_scores, d_pos, d_sel_pos, d_sel_len, 1, kpool, top_k, 1, max_sel);
        return;
    }

    const int n_vis_max = kv_len_max / kpool;
    if (n_vis_max <= top_k / kpool) {
        cudaMemsetAsync(d_sel_len, 0xFF, static_cast<size_t>(T) * sizeof(int32_t), stream);
        return;
    }

    // Score visible pools into scratch buffer [T, n_vis_max]
    float * d_scores = static_cast<float *>(d_scratch);
    const size_t need = static_cast<size_t>(T) * n_vis_max * sizeof(float);
    if (need > scratch_bytes) {
        // Fallback if somehow giant: run in chunks of tokens
        const int t_chunk = std::max(1, static_cast<int>(scratch_bytes / (static_cast<size_t>(n_vis_max) * sizeof(float))));
        for (int t0 = 0; t0 < T; t0 += t_chunk) {
            const int tc = (t0 + t_chunk < T) ? t_chunk : (T - t0);
            constexpr int wpb = 4;
            qsa_score_pools_kernel<<<dim3((n_vis_max + wpb - 1) / wpb, tc), dim3(32, wpb), 0, stream>>>(
                d_q_idx + static_cast<size_t>(t0) * n_idx_h * 128, d_k_pool_cache, d_pos + t0,
                d_scores, tc, n_idx_h, kpool, n_vis_max);
            qsa_radix_select_topk_kernel<<<tc, 256, 0, stream>>>(
                d_scores, d_pos + t0, d_sel_pos + static_cast<size_t>(t0) * max_sel, d_sel_len + t0,
                tc, kpool, top_k, n_vis_max, max_sel);
        }
        return;
    }

    constexpr int wpb = 4;
    qsa_score_pools_kernel<<<dim3((n_vis_max + wpb - 1) / wpb, T), dim3(32, wpb), 0, stream>>>(
        d_q_idx, d_k_pool_cache, d_pos, d_scores, T, n_idx_h, kpool, n_vis_max);
    qsa_radix_select_topk_kernel<<<T, 256, 0, stream>>>(
        d_scores, d_pos, d_sel_pos, d_sel_len, T, kpool, top_k, n_vis_max, max_sel);
}

void launch_gqa_attn_d256_qsa(
    const float * d_q, const void * d_k_cache, const void * d_v_cache, bool kv_is_q8_0,
    const int32_t * d_pos, const int32_t * d_sel_pos, const int32_t * d_sel_len,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int n_head, int n_kv, int max_sel, float scale,
    cublasHandle_t cublas, int kv_len_max, int top_k, int kpool, cudaStream_t stream) {

    // When kv_len_max is within the dense QSA window, use our Tiled Tensor Core FlashAttention / dense Split-K!
    if (kv_len_max > 0 && kv_len_max <= top_k + kpool - 1) {
        launch_gqa_attn_d256(d_q, d_k_cache, d_v_cache, kv_is_q8_0,
                             d_pos, d_out, d_scratch, scratch_bytes,
                             T, n_head, n_kv, scale, cublas, kv_len_max, stream);
        return;
    }

    const int gqa = n_head / n_kv;
    if (T <= 4) {
        constexpr int S = 16;
        const size_t n_slots = static_cast<size_t>(T) * n_head * S;
        float * pm = static_cast<float *>(d_scratch);
        float * pl = pm + n_slots;
        float * po = pl + n_slots;
        dim3 grid(n_kv, S, T);
        dim3 block(32, gqa);
        if (kv_is_q8_0) {
            gqa_attn_d256_sparse_splitk_kernel<true, S><<<grid, block, 0, stream>>>(
                d_q, d_k_cache, d_v_cache, d_pos, d_sel_pos, d_sel_len, pm, pl, po, n_head, n_kv, max_sel, scale);
        } else {
            gqa_attn_d256_sparse_splitk_kernel<false, S><<<grid, block, 0, stream>>>(
                d_q, d_k_cache, d_v_cache, d_pos, d_sel_pos, d_sel_len, pm, pl, po, n_head, n_kv, max_sel, scale);
        }
        gqa_attn_d256_sparse_reduce_kernel<S><<<dim3(n_head, T), 32, 0, stream>>>(pm, pl, po, d_out, n_head);
    } else {
        constexpr int S = 4;
        const size_t n_slots = static_cast<size_t>(T) * n_head * S;
        float * pm = static_cast<float *>(d_scratch);
        float * pl = pm + n_slots;
        float * po = pl + n_slots;
        dim3 grid(n_kv, S, T);
        dim3 block(32, gqa);
        if (kv_is_q8_0) {
            gqa_attn_d256_sparse_splitk_kernel<true, S><<<grid, block, 0, stream>>>(
                d_q, d_k_cache, d_v_cache, d_pos, d_sel_pos, d_sel_len, pm, pl, po, n_head, n_kv, max_sel, scale);
        } else {
            gqa_attn_d256_sparse_splitk_kernel<false, S><<<grid, block, 0, stream>>>(
                d_q, d_k_cache, d_v_cache, d_pos, d_sel_pos, d_sel_len, pm, pl, po, n_head, n_kv, max_sel, scale);
        }
        gqa_attn_d256_sparse_reduce_kernel<S><<<dim3(n_head, T), 32, 0, stream>>>(pm, pl, po, d_out, n_head);
    }
}

} // namespace rookery
