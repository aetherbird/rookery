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

// Ultra-fast single-warp register-resident Top-K sigmoid router for n_expert <= 256 (ZERO __syncthreads, ZERO smem!)
__global__ void moe_route_sigmoid_warp_256_kernel(
    const float * __restrict__ logits,
    const float * __restrict__ bias,
    int32_t *     __restrict__ top_ids,
    float *       __restrict__ top_w,
    int n_expert, int k, bool renorm, float scale) {

    const int t    = blockIdx.x;
    const int lane = threadIdx.x; // 0..31 -> owns 8 experts: lane + i*32 (i=0..7)
    const float * row = logits + static_cast<size_t>(t) * n_expert;

    float p_reg[8];
    float s_reg[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int idx = lane + i * 32;
        if (idx < n_expert) {
            const float p = 1.0f / (1.0f + __expf(-row[idx]));
            p_reg[i] = p;
            s_reg[i] = bias ? (p + __ldg(&bias[idx])) : p;
        } else {
            p_reg[i] = 0.0f;
            s_reg[i] = -FLT_MAX;
        }
    }

    int32_t out_i[16];
    float   out_w[16];
    float   w_sum = 0.0f;

    for (int j = 0; j < k; ++j) {
        float   bv = s_reg[0];
        int32_t bi = lane;
        float   bp = p_reg[0];
        #pragma unroll
        for (int i = 1; i < 8; ++i) {
            const float sv = s_reg[i];
            if (sv > bv) {
                bv = sv;
                bi = lane + i * 32;
                bp = p_reg[i];
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float   ov = __shfl_down_sync(0xffffffffu, bv, offset);
            const int32_t oi = __shfl_down_sync(0xffffffffu, bi, offset);
            const float   op = __shfl_down_sync(0xffffffffu, bp, offset);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov; bi = oi; bp = op;
            }
        }
        const int32_t picked = __shfl_sync(0xffffffffu, bi, 0);
        if (lane == 0) {
            out_i[j] = bi;
            out_w[j] = bp;
            w_sum   += bp;
        }
        if ((picked & 31) == lane) {
            const int slot_i = picked >> 5;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i == slot_i) s_reg[i] = -FLT_MAX;
            }
        }
    }

    if (lane == 0) {
        const float mul = (renorm && w_sum > 0.0f) ? (scale / w_sum) : scale;
        int32_t * dst_i = top_ids + static_cast<size_t>(t) * k;
        float *   dst_w = top_w   + static_cast<size_t>(t) * k;
        for (int j = 0; j < k; ++j) {
            dst_i[j] = out_i[j];
            dst_w[j] = out_w[j] * mul;
        }
    }
}

__global__ void moe_route_sigmoid_kernel(
    const float * __restrict__ logits,
    const float * __restrict__ bias,
    int32_t *     __restrict__ top_ids,
    float *       __restrict__ top_w,
    int n_expert, int k, bool renorm, float scale,
    int n_expert_groups, int n_group_used) {

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float * row = logits + static_cast<size_t>(t) * n_expert;

    float my_p0 = 0.0f, my_s0 = -FLT_MAX;
    float my_p1 = 0.0f, my_s1 = -FLT_MAX;
    if (tid < n_expert) {
        my_p0 = 1.0f / (1.0f + expf(-row[tid]));
        my_s0 = bias ? (my_p0 + bias[tid]) : my_p0;
    }
    if (tid + 256 < n_expert) {
        my_p1 = 1.0f / (1.0f + expf(-row[tid + 256]));
        my_s1 = bias ? (my_p1 + bias[tid + 256]) : my_p1;
    }

    __shared__ float   s_sel[512];
    __shared__ float   s_gscore[32];
    __shared__ uint8_t s_gmask[32];

    // Optional DeepSeek-V3 / BailingMoE3 grouped top-k routing (top-2 sum per group -> keep top n_group_used groups)
    if (n_expert_groups > 1 && n_group_used > 0 && n_group_used < n_expert_groups) {
        const int n_per_grp = n_expert / n_expert_groups;
        s_sel[tid]       = my_s0;
        s_sel[tid + 256] = my_s1;
        __syncthreads();

        if (tid < n_expert_groups) {
            const int g0 = tid * n_per_grp;
            float t1 = -FLT_MAX, t2 = -FLT_MAX;
            for (int i = 0; i < n_per_grp; ++i) {
                float v = s_sel[g0 + i];
                if (v > t1) { t2 = t1; t1 = v; }
                else if (v > t2) { t2 = v; }
            }
            s_gscore[tid] = t1 + t2;
            s_gmask[tid]  = 0;
        }
        __syncthreads();

        if (tid == 0) {
            for (int j = 0; j < n_group_used; ++j) {
                float best_v = -FLT_MAX;
                int   best_g = 0;
                for (int g = 0; g < n_expert_groups; ++g) {
                    if (!s_gmask[g] && s_gscore[g] > best_v) {
                        best_v = s_gscore[g];
                        best_g = g;
                    }
                }
                s_gmask[best_g] = 1;
            }
        }
        __syncthreads();

        if (tid < n_expert && !s_gmask[tid / n_per_grp]) my_s0 = -FLT_MAX;
        if (tid + 256 < n_expert && !s_gmask[(tid + 256) / n_per_grp]) my_s1 = -FLT_MAX;
    }

    __shared__ float   s_wval[8];
    __shared__ int32_t s_widx[8];
    __shared__ float   s_wp[8];
    __shared__ int32_t s_chosen_id;
    __shared__ float   s_out_w[16];
    __shared__ int32_t s_out_i[16];

    float w_sum = 0.0f;

    for (int j = 0; j < k; ++j) {
        float   bv = my_s0;
        int32_t bi = tid;
        float   bp = my_p0;
        if (my_s1 > bv) {
            bv = my_s1;
            bi = tid + 256;
            bp = my_p1;
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            float   ov = __shfl_down_sync(0xffffffffu, bv, offset);
            int32_t oi = __shfl_down_sync(0xffffffffu, bi, offset);
            float   op = __shfl_down_sync(0xffffffffu, bp, offset);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov; bi = oi; bp = op;
            }
        }
        if (lane == 0) {
            s_wval[warp] = bv;
            s_widx[warp] = bi;
            s_wp[warp]   = bp;
        }
        __syncthreads();

        if (warp == 0) {
            float   wv = (lane < 8) ? s_wval[lane] : -FLT_MAX;
            int32_t wi = (lane < 8) ? s_widx[lane] : 999999;
            float   wp = (lane < 8) ? s_wp[lane]   : 0.0f;
            for (int offset = 4; offset > 0; offset >>= 1) {
                float   ov = __shfl_down_sync(0xffffffffu, wv, offset);
                int32_t oi = __shfl_down_sync(0xffffffffu, wi, offset);
                float   op = __shfl_down_sync(0xffffffffu, wp, offset);
                if (ov > wv || (ov == wv && oi < wi)) {
                    wv = ov; wi = oi; wp = op;
                }
            }
            if (lane == 0) {
                s_chosen_id = wi;
                s_out_i[j]  = wi;
                s_out_w[j]  = wp;
                w_sum      += wp;
            }
        }
        __syncthreads();

        const int32_t picked = s_chosen_id;
        if (tid == picked)       my_s0 = -FLT_MAX;
        if (tid + 256 == picked) my_s1 = -FLT_MAX;
    }

    if (tid == 0) {
        const float mul = (renorm && w_sum > 0.0f) ? (scale / w_sum) : scale;
        int32_t * out_i = top_ids + static_cast<size_t>(t) * k;
        float *   out_w = top_w   + static_cast<size_t>(t) * k;
        for (int j = 0; j < k; ++j) {
            out_i[j] = s_out_i[j];
            out_w[j] = s_out_w[j] * mul;
        }
    }
}

__global__ void quantize_q8_1_moe_kernel(
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
    if (lane == 0) qx[b].d = d;
    qx[b].qs[lane] = static_cast<int8_t>(__float2int_rn(v * inv_d));
}

// 100% COALESCED 128-bit LDG.128 Fused Gate + Up + SiLU for routed (k) + shared (1) experts!
// 2 threads per 32-element block -> 16 blocks/warp -> 4 iterations for K=2048 (K_blocks=64)!
__global__ void moe_gate_up_silu_soa_ldg128_kernel(
    const int4 *          __restrict__ gate_exps_q,
    const uint16_t *      __restrict__ gate_exps_s,
    const int4 *          __restrict__ up_exps_q,
    const uint16_t *      __restrict__ up_exps_s,
    const int4 *          __restrict__ gate_shexp_q,
    const uint16_t *      __restrict__ gate_shexp_s,
    const int4 *          __restrict__ up_shexp_q,
    const uint16_t *      __restrict__ up_shexp_s,
    const BlockQ8_1_Act * __restrict__ x_q8,
    const int32_t *       __restrict__ top_ids,
    float *               __restrict__ hid_routed,
    float *               __restrict__ hid_shared,
    int K_blocks, int ff_exp, int ff_shexp, int k_active) {

    extern __shared__ BlockQ8_1_Act s_x[];

    const int t    = blockIdx.z;
    const int ki   = blockIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;

    const BlockQ8_1_Act * xr = x_q8 + static_cast<size_t>(t) * K_blocks;
    const int n_i4 = (K_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
    const int4 * sx_in  = reinterpret_cast<const int4 *>(xr);
    int4 *       sx_out = reinterpret_cast<int4 *>(s_x);
    for (int i = tid; i < n_i4; i += nthr) sx_out[i] = sx_in[i];
    __syncthreads();

    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;

    const int4 *     gq_row = nullptr;
    const uint16_t * gs_row = nullptr;
    const int4 *     uq_row = nullptr;
    const uint16_t * us_row = nullptr;
    float *          dst    = nullptr;

    if (ki < k_active) {
        if (row >= ff_exp) return;
        const int exp_id = top_ids[t * k_active + ki];
        const size_t exp_sc_stride = static_cast<size_t>(ff_exp) * K_blocks;
        const size_t row_sc_off    = static_cast<size_t>(exp_id) * exp_sc_stride + static_cast<size_t>(row) * K_blocks;
        gq_row = gate_exps_q + row_sc_off * 2;
        gs_row = gate_exps_s + row_sc_off;
        uq_row = up_exps_q   + row_sc_off * 2;
        us_row = up_exps_s   + row_sc_off;
        dst    = &hid_routed[(static_cast<size_t>(t) * k_active + ki) * ff_exp + row];
    } else {
        if (row >= ff_shexp) return;
        const size_t row_sc_off = static_cast<size_t>(row) * K_blocks;
        gq_row = gate_shexp_q + row_sc_off * 2;
        gs_row = gate_shexp_s + row_sc_off;
        uq_row = up_shexp_q   + row_sc_off * 2;
        us_row = up_shexp_s   + row_sc_off;
        dst    = &hid_shared[static_cast<size_t>(t) * ff_shexp + row];
    }

    const int blk_in_warp = lane >> 1; // 0..15
    const int half_in_blk = lane & 1;  // 0..1

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int b = blk_in_warp; b < K_blocks; b += 16) {
        const int4  x4 = reinterpret_cast<const int4 *>(s_x[b].qs)[half_in_blk];
        const float xd = s_x[b].d;

        {
            const int4 gw = __ldg(&gq_row[b * 2 + half_in_blk]);
            int sg = dp4a_s32(gw.x, x4.x, 0);
            sg     = dp4a_s32(gw.y, x4.y, sg);
            sg     = dp4a_s32(gw.z, x4.z, sg);
            sg     = dp4a_s32(gw.w, x4.w, sg);
            __half gd_h;
            *reinterpret_cast<uint16_t *>(&gd_h) = __ldg(&gs_row[b]);
            acc_g += (__half2float(gd_h) * xd) * static_cast<float>(sg);
        }
        {
            const int4 uw = __ldg(&uq_row[b * 2 + half_in_blk]);
            int su = dp4a_s32(uw.x, x4.x, 0);
            su     = dp4a_s32(uw.y, x4.y, su);
            su     = dp4a_s32(uw.z, x4.z, su);
            su     = dp4a_s32(uw.w, x4.w, su);
            __half ud_h;
            *reinterpret_cast<uint16_t *>(&ud_h) = __ldg(&us_row[b]);
            acc_u += (__half2float(ud_h) * xd) * static_cast<float>(su);
        }
    }

    acc_g = warp_sum(acc_g);
    acc_u = warp_sum(acc_u);
    if (lane == 0) {
        *dst = (acc_g / (1.0f + expf(-acc_g))) * acc_u;
    }
}

// 100% COALESCED 128-bit LDG.128 Fused Weighted Down-Projection for ALL k routed + shared experts!
// For Maxwell (ff_blocks=16, ff_sh_blocks=32):
//   - Each routed expert takes EXACTLY ONE single pass (16 blocks = 32 threads x LDG.128 = 512 contiguous bytes!)
//   - Shared expert takes TWO passes (32 blocks = 2 x 512 contiguous bytes!)
__global__ void moe_down_all_soa_ldg128_kernel(
    const int4 *          __restrict__ down_exps_q,
    const uint16_t *      __restrict__ down_exps_s,
    const int4 *          __restrict__ down_shexp_q,
    const uint16_t *      __restrict__ down_shexp_s,
    const BlockQ8_1_Act * __restrict__ hid_rq8,
    const BlockQ8_1_Act * __restrict__ hid_sq8,
    const int32_t *       __restrict__ top_ids,
    const float *         __restrict__ top_w,
    float *               __restrict__ out,
    int ff_blocks, int ff_sh_blocks, int n_embd, int k_active) {

    extern __shared__ BlockQ8_1_Act s_hid[];
    BlockQ8_1_Act * s_rq8 = s_hid;
    BlockQ8_1_Act * s_sq8 = s_hid + k_active * ff_blocks;

    const int t    = blockIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int nthr = blockDim.y * 32;

    {
        const int r_i4 = (k_active * ff_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
        const int4 * src = reinterpret_cast<const int4 *>(hid_rq8 + static_cast<size_t>(t) * k_active * ff_blocks);
        int4 *       dst = reinterpret_cast<int4 *>(s_rq8);
        for (int i = tid; i < r_i4; i += nthr) dst[i] = src[i];

        const int s_i4 = (ff_sh_blocks * static_cast<int>(sizeof(BlockQ8_1_Act))) / 16;
        const int4 * ssrc = reinterpret_cast<const int4 *>(hid_sq8 + static_cast<size_t>(t) * ff_sh_blocks);
        int4 *       sdst = reinterpret_cast<int4 *>(s_sq8);
        for (int i = tid; i < s_i4; i += nthr) sdst[i] = ssrc[i];
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_embd) return;

    const int lane        = threadIdx.x;
    const int blk_in_warp = lane >> 1; // 0..15
    const int half_in_blk = lane & 1;  // 0..1
    const size_t exp_sc_stride = static_cast<size_t>(n_embd) * ff_blocks;

    float total = 0.0f;

    for (int ki = 0; ki < k_active; ++ki) {
        const int   exp_id = top_ids[t * k_active + ki];
        const float w      = top_w[t * k_active + ki];
        const size_t row_sc_off = static_cast<size_t>(exp_id) * exp_sc_stride + static_cast<size_t>(row) * ff_blocks;
        const int4 *     dq = down_exps_q + row_sc_off * 2;
        const uint16_t * ds = down_exps_s + row_sc_off;
        const BlockQ8_1_Act * hr = s_rq8 + ki * ff_blocks;

        float acc = 0.0f;
        for (int b = blk_in_warp; b < ff_blocks; b += 16) {
            const int4 dw = __ldg(&dq[b * 2 + half_in_blk]);
            const int4 hx = reinterpret_cast<const int4 *>(hr[b].qs)[half_in_blk];
            int s = dp4a_s32(dw.x, hx.x, 0);
            s     = dp4a_s32(dw.y, hx.y, s);
            s     = dp4a_s32(dw.z, hx.z, s);
            s     = dp4a_s32(dw.w, hx.w, s);
            __half dd_h;
            *reinterpret_cast<uint16_t *>(&dd_h) = __ldg(&ds[b]);
            acc += (__half2float(dd_h) * hr[b].d) * static_cast<float>(s);
        }
        total += w * acc;
    }

    {
        const size_t row_sc_off = static_cast<size_t>(row) * ff_sh_blocks;
        const int4 *     sdq = down_shexp_q + row_sc_off * 2;
        const uint16_t * sds = down_shexp_s + row_sc_off;
        for (int b = blk_in_warp; b < ff_sh_blocks; b += 16) {
            const int4 dw = __ldg(&sdq[b * 2 + half_in_blk]);
            const int4 hx = reinterpret_cast<const int4 *>(s_sq8[b].qs)[half_in_blk];
            int s = dp4a_s32(dw.x, hx.x, 0);
            s     = dp4a_s32(dw.y, hx.y, s);
            s     = dp4a_s32(dw.z, hx.z, s);
            s     = dp4a_s32(dw.w, hx.w, s);
            __half dd_h;
            *reinterpret_cast<uint16_t *>(&dd_h) = __ldg(&sds[b]);
            total += (__half2float(dd_h) * s_sq8[b].d) * static_cast<float>(s);
        }
    }

    total = warp_sum(total);
    if (lane == 0) {
        out[static_cast<size_t>(t) * n_embd + row] = total;
    }
}

} // namespace

void launch_moe_route_sigmoid(
    const float * d_logits, const float * d_bias,
    int32_t * d_top_ids, float * d_top_w,
    int n_tokens, int n_expert, int k, bool renorm, float scale,
    cudaStream_t stream,
    int n_expert_groups, int n_group_used) {

    if (n_expert > 512) {
        throw std::runtime_error("moe_route_sigmoid: n_expert > 512 not yet supported");
    }
    if (n_expert_groups > 32) {
        throw std::runtime_error("moe_route_sigmoid: n_expert_groups > 32 not yet supported");
    }
    if (n_expert <= 256 && k <= 16 && n_expert_groups <= 1) {
        moe_route_sigmoid_warp_256_kernel<<<n_tokens, 32, 0, stream>>>(
            d_logits, d_bias, d_top_ids, d_top_w, n_expert, k, renorm, scale);
        return;
    }
    moe_route_sigmoid_kernel<<<n_tokens, 256, 0, stream>>>(
        d_logits, d_bias, d_top_ids, d_top_w, n_expert, k, renorm, scale, n_expert_groups, n_group_used);
}

void launch_moe_ffn_q8_0_with_shared(
    const DeviceTensor & gate_exps,
    const DeviceTensor & up_exps,
    const DeviceTensor & down_exps,
    const DeviceTensor & gate_shexp,
    const DeviceTensor & up_shexp,
    const DeviceTensor & down_shexp,
    const float *        d_x,
    const int32_t *      d_top_ids,
    const float *        d_top_w,
    float *              d_out,
    void *               d_scratch,
    int                  T,
    int                  k,
    cudaStream_t         stream) {

    const int n_embd       = static_cast<int>(gate_exps.ne[0]);
    const int ff_exp       = static_cast<int>(gate_exps.ne[1]);
    const int ff_shexp     = static_cast<int>(gate_shexp.ne[1]);
    const int K_blocks     = n_embd / 32;
    const int ff_blocks    = ff_exp / 32;
    const int ff_sh_blocks = ff_shexp / 32;

    auto * ptr = static_cast<uint8_t *>(d_scratch);
    auto align256 = [](size_t n) { return (n + 255) & ~size_t(255); };

    const size_t x_q8_bytes   = align256(static_cast<size_t>(T) * K_blocks * sizeof(BlockQ8_1_Act));
    const size_t hid_r_bytes  = align256(static_cast<size_t>(T) * k * ff_exp * sizeof(float));
    const size_t hid_s_bytes  = align256(static_cast<size_t>(T) * ff_shexp * sizeof(float));
    const size_t hid_rq_bytes = align256(static_cast<size_t>(T) * k * ff_blocks * sizeof(BlockQ8_1_Act));

    auto * d_xq    = reinterpret_cast<BlockQ8_1_Act *>(ptr);
    auto * d_hidr  = reinterpret_cast<float *>(ptr + x_q8_bytes);
    auto * d_hids  = reinterpret_cast<float *>(ptr + x_q8_bytes + hid_r_bytes);
    auto * d_hidrq = reinterpret_cast<BlockQ8_1_Act *>(ptr + x_q8_bytes + hid_r_bytes + hid_s_bytes);
    auto * d_hidsq = reinterpret_cast<BlockQ8_1_Act *>(ptr + x_q8_bytes + hid_r_bytes + hid_s_bytes + hid_rq_bytes);

    constexpr int q_wpb = 8;
    const int total_xb = T * K_blocks;
    quantize_q8_1_moe_kernel<<<(total_xb + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(
        d_x, d_xq, total_xb);

    constexpr int wpb = 8;
    const int max_ff = (ff_shexp > ff_exp) ? ff_shexp : ff_exp;
    dim3 gu_grid((max_ff + wpb - 1) / wpb, k + 1, T);
    const size_t gu_smem = static_cast<size_t>(K_blocks) * sizeof(BlockQ8_1_Act);
    moe_gate_up_silu_soa_ldg128_kernel<<<gu_grid, dim3(32, wpb), gu_smem, stream>>>(
        static_cast<const int4 *>(gate_exps.ptr),    static_cast<const uint16_t *>(gate_exps.scales_ptr),
        static_cast<const int4 *>(up_exps.ptr),      static_cast<const uint16_t *>(up_exps.scales_ptr),
        static_cast<const int4 *>(gate_shexp.ptr),   static_cast<const uint16_t *>(gate_shexp.scales_ptr),
        static_cast<const int4 *>(up_shexp.ptr),     static_cast<const uint16_t *>(up_shexp.scales_ptr),
        d_xq, d_top_ids, d_hidr, d_hids, K_blocks, ff_exp, ff_shexp, k);

    const int total_rb = T * k * ff_blocks;
    quantize_q8_1_moe_kernel<<<(total_rb + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(
        d_hidr, d_hidrq, total_rb);
    const int total_sb = T * ff_sh_blocks;
    quantize_q8_1_moe_kernel<<<(total_sb + q_wpb - 1) / q_wpb, dim3(32, q_wpb), 0, stream>>>(
        d_hids, d_hidsq, total_sb);

    dim3 dn_grid((n_embd + wpb - 1) / wpb, T);
    const size_t dn_smem = static_cast<size_t>(k * ff_blocks + ff_sh_blocks) * sizeof(BlockQ8_1_Act);
    moe_down_all_soa_ldg128_kernel<<<dn_grid, dim3(32, wpb), dn_smem, stream>>>(
        static_cast<const int4 *>(down_exps.ptr),    static_cast<const uint16_t *>(down_exps.scales_ptr),
        static_cast<const int4 *>(down_shexp.ptr),   static_cast<const uint16_t *>(down_shexp.scales_ptr),
        d_hidrq, d_hidsq, d_top_ids, d_top_w, d_out, ff_blocks, ff_sh_blocks, n_embd, k);
}

} // namespace rookery
