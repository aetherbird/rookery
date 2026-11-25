#pragma once

#include "core/types.h"
#include "gguf/loader.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace rookery {

void launch_embedding_lookup(
    const DeviceTensor & tok_embd,
    const int32_t *      d_tokens,
    float *              d_out,
    int                  n_embd,
    int                  hc_mult,
    int                  n_tokens,
    cudaStream_t         stream);

void launch_rmsnorm(
    const float * d_x,
    const float * d_w,
    float *       d_out,
    int           dim,
    int           n_rows,
    float         eps,
    cudaStream_t  stream);

void launch_add_rmsnorm(
    float *       d_cur,
    const float * d_sub,
    const float * d_w,
    float *       d_out,
    int           dim,
    int           n_rows,
    float         eps,
    cudaStream_t  stream);

void launch_qk_norm_rope_neox(
    float *         d_x,
    const float *   d_norm_w,
    const int32_t * d_pos,
    int             n_tokens,
    int             n_heads,
    int             head_dim,
    float           rms_eps,
    float           rope_freq_base,
    cudaStream_t    stream);

void launch_maxwell_qk_norm_rope_kv_write(
    float *         d_q,
    const float *   d_k,
    const float *   d_v,
    const float *   d_q_norm_w,
    const float *   d_k_norm_w,
    void *          d_k_cache,
    void *          d_v_cache,
    KvCacheType     kv_type,
    const int32_t * d_pos,
    int             n_tokens,
    int             n_heads,
    int             n_kv_heads,
    int             head_dim,
    float           rms_eps,
    float           rope_freq_base,
    cudaStream_t    stream);

void launch_silu_mul(
    const float * d_gate,
    const float * d_up,
    float *       d_out,
    int64_t       n,
    cudaStream_t  stream);

void launch_add_inplace(
    float *       d_dst,
    const float * d_src,
    int64_t       n,
    cudaStream_t  stream);

// --- Manifold-Constrained Hyper-Connections (mHC, hc=4) ---
// Fused flat RMSNorm(inpL[T, hc_dim]) + BF16/F16/F32 GEMV(hc_fn[n_out, hc_dim]) -> d_mixes[T, n_out]
void launch_hc_flat_rmsnorm_mix(
    const float *        d_inpL,
    const DeviceTensor & hc_fn,
    float *              d_mixes,
    int                  hc_dim,
    int                  n_out,
    int                  n_tokens,
    float                rms_eps,
    cudaStream_t         stream);

void launch_hc_sinkhorn_pre_rmsnorm(
    const float * d_inpL,
    const float * d_mixes,
    const float * d_hc_scale,
    const float * d_hc_base,
    const float * d_norm_w,
    float *       d_post_out,
    float *       d_comb_out,
    float *       d_x_norm_out,
    int           n_embd,
    int           n_tokens,
    int           sinkhorn_iters,
    float         hc_eps,
    float         rms_eps,
    cudaStream_t  stream);

void launch_hc_post(
    const float * d_sub,
    const float * d_residual,
    const float * d_post,
    const float * d_comb,
    float *       d_out,
    int           n_embd,
    int           n_tokens,
    cudaStream_t  stream);

void launch_hc_post_flat_rmsnorm_mix(
    const float *        d_sub,
    const float *        d_residual,
    const float *        d_post,
    const float *        d_comb,
    float *              d_out,
    const DeviceTensor & hc_fn,
    float *              d_mixes,
    int                  n_embd,
    int                  n_out,
    int                  n_tokens,
    float                rms_eps,
    cudaStream_t         stream);

void launch_hc_head_rmsnorm(
    const float * d_inpL,
    const float * d_mixes,
    const float * d_hc_scale,
    const float * d_hc_base,
    const float * d_out_norm_w,
    float *       d_out,
    int           n_embd,
    int           n_tokens,
    float         hc_eps,
    float         rms_eps,
    cudaStream_t  stream);

void launch_hc_mean_rmsnorm(
    const float * d_inpL,
    const float * d_out_norm_w,
    float *       d_out,
    int           n_embd,
    int           n_tokens,
    float         rms_eps,
    cudaStream_t  stream);

// --- Quantized Linear / GEMV / GEMM ---
void launch_quantize_q8_1(
    const float * d_x,
    void *        d_xq8,
    int           K,
    int           T,
    cudaStream_t  stream);

void launch_linear_q8_0_prequant(
    const DeviceTensor & W,
    const void *         d_xq8,
    float *              d_y,
    int                  T,
    cudaStream_t         stream);

void launch_linear(
    cublasHandle_t       cublas,
    const DeviceTensor & W,
    const float *        d_x,
    float *              d_y,
    int                  T,
    void *               d_scratch,
    size_t               scratch_bytes,
    cudaStream_t         stream);

// MTP input prep: concatenates [RMSNorm(tok_embd[tok], enorm) ; RMSNorm(h[s], hnorm)] for each stream s in 0..3 -> d_eh_in[T, 4, 2*n_embd]
void launch_mtp_eh_concat(
    const float * d_e_embd,   // [T, n_embd]
    const float * d_h_state,  // [T, 4, n_embd]
    const float * d_enorm_w,  // [n_embd]
    const float * d_hnorm_w,  // [n_embd]
    float *       d_eh_in,    // [T * 4, 2 * n_embd]
    int           n_embd,
    int           hc_mult,
    int           n_tokens,
    float         rms_eps,
    cudaStream_t  stream);

void launch_argmax_f32(
    const float * d_x,
    int32_t *     d_out_idx,
    int           n,
    cudaStream_t  stream);

// Batched argmax: finds argmax for each row t in 0..T-1 of d_x[T, n] -> d_out_idx[T]
void launch_argmax_f32_batch(
    const float * d_x,
    int32_t *     d_out_idx,
    int           n,
    int           T,
    cudaStream_t  stream);

// --- MoE (Maxwell sigmoid top-K + shared expert) ---
void launch_moe_route_sigmoid(
    const float * d_logits,
    const float * d_bias,
    int32_t *     d_top_ids,
    float *       d_top_w,
    int           n_tokens,
    int           n_expert,
    int           k,
    bool          renorm,
    float         scale,
    cudaStream_t  stream,
    int           n_expert_groups = 0,
    int           n_group_used = 0);

// Fused MoE FFN (routed top-K + shared expert) using pre-quantized x_q8 for T==1 or general T.
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
    cudaStream_t         stream);

// --- Attention & KV Cache (F16, Q8_0, and Q4_0 KV) ---
void launch_kv_write_f16(
    const float *   d_k,
    const float *   d_v,
    void *          d_k_cache,
    void *          d_v_cache,
    const int32_t * d_pos,
    int             T,
    int             n_kv_heads,
    int             head_dim,
    cudaStream_t    stream);

void launch_kv_write_q8_0(
    const float *   d_k,
    const float *   d_v,
    void *          d_k_cache,
    void *          d_v_cache,
    const int32_t * d_pos,
    int             T,
    int             n_kv_heads,
    int             head_dim,
    cudaStream_t    stream);

void launch_kv_write_q4_0(
    const float *   d_k,
    const float *   d_v,
    void *          d_k_cache,
    void *          d_v_cache,
    const int32_t * d_pos,
    int             T,
    int             n_kv_heads,
    int             head_dim,
    cudaStream_t    stream);

void launch_gqa_attn(
    const float *   d_q,
    const void *    d_k_cache,
    const void *    d_v_cache,
    KvCacheType     kv_type,
    const int32_t * d_pos,
    float *         d_out,
    void *          d_scratch,
    int             T,
    int             n_heads,
    int             n_kv_heads,
    int             head_dim,
    float           scale,
    cudaStream_t    stream,
    cublasHandle_t  cublas = nullptr,
    size_t          scratch_bytes = 0,
    int             kv_len_max = 0);

// --- Maxwell Block-Sparse Attention (MSA) ---
void launch_idx_k_write_f16(
    const float *   d_ik,
    void *          d_ik_cache,
    const int32_t * d_pos,
    int             T,
    int             idx_dim,
    cudaStream_t    stream);

void launch_idx_k_write_q4_0(
    const float *   d_ik,
    void *          d_ik_cache,
    const int32_t * d_pos,
    int             T,
    int             idx_dim,
    cudaStream_t    stream);

void launch_msa_indexer_topk(
    const float *   d_iq,
    const void *    d_ik_cache,
    bool            ik_is_q4_0,
    const int32_t * d_pos,
    int32_t *       d_out_blocks,
    int32_t *       d_out_k_count,
    void *          d_scratch,
    int             T,
    int             max_ctx,
    int             n_idx_heads,
    int             blk_size,
    int             prefix_blks,
    int             local_blks,
    int             k_max,
    int             dense_prefix_tokens,
    cudaStream_t    stream);

void launch_gqa_attn_msa(
    const float *   d_q,
    const void *    d_k_cache,
    const void *    d_v_cache,
    KvCacheType     kv_type,
    const int32_t * d_pos,
    const int32_t * d_msa_blocks,
    const int32_t * d_msa_k_count,
    float *         d_out,
    void *          d_scratch,
    int             T,
    int             n_heads,
    int             n_kv_heads,
    int             head_dim,
    int             blk_size,
    int             k_max,
    float           scale,
    cudaStream_t    stream);

// --- K-Quant / IQ-Quant Linear, Embedding & MoE (Q6_K, IQ4_XS, IQ3_S, IQ2_S, Q3_K, Q2_K) ---
bool is_kquant_dtype(DType t);

void launch_cast_f32_to_f16(const float * d_src, void * d_dst, int64_t n, cudaStream_t stream);

void launch_silu_mul_clamp(
    const float * d_gate, const float * d_up, float * d_out,
    int64_t n, float clamp_limit, cudaStream_t stream);

void launch_linear_kquant(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream);

void launch_linear_prequant(
    const DeviceTensor & W, const void * d_xq8, float * d_y,
    int T, cudaStream_t stream);

// Fused multi-projection prequant GEMV: computes Ys[i] = Ws[i] @ x_q8 in minimal launches!
void launch_linear_multi_prequant(
    const DeviceTensor * const * Ws, float * const * Ys, int n_mat,
    const void * d_xq8, int T, cudaStream_t stream);

void launch_embedding_kquant(
    const DeviceTensor & tok_embd, const int32_t * d_tokens, float * d_out,
    int n_embd, int hc_mult, int n_tokens, cudaStream_t stream);

void launch_moe_ffn_kquant_with_shared(
    cublasHandle_t cublas,
    const DeviceTensor & gate_exps, const DeviceTensor & up_exps, const DeviceTensor & down_exps,
    const DeviceTensor & gate_shexp, const DeviceTensor & up_shexp, const DeviceTensor & down_shexp,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, float clamp_exp, float clamp_shexp, cudaStream_t stream);

// --- KDA / GDN Linear Attention (GLM-5.3, Ling-3.0, Qwen3.6, Qwen3.8) ---
// Optional d_state_snap: when T==2 and non-null, saves post-t=0 state for lossless MTP rollback!
void launch_ssm_conv1d_silu(
    const float * d_x, const float * d_w, float * d_state, float * d_out,
    int T, int C, cudaStream_t stream, float * d_state_snap = nullptr);

void launch_ssm_conv1d_silu_qkv_l2(
    const float * d_q, const float * d_k, const float * d_v,
    const float * d_wq, const float * d_wk, const float * d_wv,
    float * d_sq, float * d_sk, float * d_sv,
    float * d_qc, float * d_kc, float * d_vc,
    int T, int n_heads, float eps, cudaStream_t stream,
    float * d_sq_snap = nullptr, float * d_sk_snap = nullptr, float * d_sv_snap = nullptr);

void launch_l2_norm_d128(float * d_x, int T, int n_heads, float eps, cudaStream_t stream);

void launch_kda_gate_and_beta(
    float * d_f, const float * d_dt_bias, const float * d_ssm_a,
    float * d_beta, float lower_bound, int T, int n_heads, int head_dim, cudaStream_t stream,
    bool negate_a = true);

void launch_gated_delta_net_d128(
    const float * d_q, const float * d_k, const float * d_v,
    const float * d_g, const float * d_beta,
    float * d_state, float * d_out,
    int T, int n_heads, float scale, bool is_kda, cudaStream_t stream,
    float * d_state_snap = nullptr);

void launch_rms_norm_sigmoid_gate_d128(
    float * d_o, const float * d_norm_w, const float * d_gate,
    int T, int n_heads, float eps, cudaStream_t stream);

// --- Absorbed MLA Multi-Head Latent Attention (GLM-5.3, DeepSeek-V4.1, Ling-3.0) ---
void launch_batched_head_proj_q8_0(
    const DeviceTensor & W, const float * d_x, float * d_y,
    int T, cudaStream_t stream,
    cublasHandle_t cublas = nullptr, void * d_scratch = nullptr, size_t scratch_bytes = 0);

void launch_mla_kv_write(
    const float * d_kv, void * d_kv_cache, bool kv_is_q8_0,
    const int32_t * d_pos, int T, int kv_dim, cudaStream_t stream);

void launch_mla_attn_d512(
    const float * d_q_abs, const void * d_kv_cache, bool kv_is_q8_0,
    const int32_t * d_pos, float * d_out, void * d_scratch,
    int T, int n_heads, float scale, cudaStream_t stream,
    cublasHandle_t cublas = nullptr, size_t scratch_bytes = 0, int kv_len_max = 0);

// --- RoPE-576 Absorbed MLA (BailingMoE3 / Ling-3.0 / Ling-3.2-Air / DeepSeek MLA) ---
void launch_mla_rope_split_q(
    const float * d_q_all,    // [T, n_heads, qk_nope + rope_dim]
    const int32_t * d_pos,    // [T]
    float * d_q_nope,         // [T, n_heads, qk_nope]
    float * d_q_pe,           // [T, n_heads, rope_dim]
    int T, int n_heads, int qk_nope, int rope_dim, float rope_freq_base, cudaStream_t stream);

void launch_mla_kv_norm_rope_write_576(
    const float * d_kv_all,   // [T, 576] = [kv_latent(512), k_pe(64)]
    const float * d_norm_w,   // [512]
    const int32_t * d_pos,    // [T]
    void * d_kv_cache,        // [max_ctx, 576] (18 BlockQ8_0 or 576 F16 per pos)
    bool kv_is_q8_0,
    int T, float rms_eps, float rope_freq_base, cudaStream_t stream);

void launch_mla_concat_q576(
    const float * d_q_abs,    // [T, n_heads, 512]
    const float * d_q_pe,     // [T, n_heads, 64]
    float * d_q576,           // [T, n_heads, 576]
    int T, int n_heads, cudaStream_t stream);

void launch_batched_head_proj_f16(
    const DeviceTensor & W,   // F16 [K, M, n_heads] (ne[0]=K, ne[1]=M, ne[2]=n_heads)
    const float * d_x,        // [T, n_heads, K]
    float * d_y,              // [T, n_heads, M]
    int T, cudaStream_t stream,
    cublasHandle_t cublas = nullptr, void * d_scratch = nullptr, size_t scratch_bytes = 0);

void launch_mla_attn_d576_v512(
    const float * d_q576, const void * d_kv_cache, bool kv_is_q8_0,
    const int32_t * d_pos, float * d_out, void * d_scratch,
    int T, int n_heads, float scale, cudaStream_t stream,
    cublasHandle_t cublas = nullptr, size_t scratch_bytes = 0, int kv_len_max = 0);

void launch_mla_head_sigmoid_gate(
    float * d_o,              // [T, n_heads, head_dim] in-place
    const float * d_gate,     // [T, n_heads]
    int T, int n_heads, int head_dim, cudaStream_t stream);

// --- Inkling 2 Mini (inkling2_loop: Looped MoE + Depth AttnRes + ShortConv + RelBias Sink Attn + SiTU) ---
void launch_inkling2_depth_mix(
    const float * const * d_blocks_dev, // device array of [n_blocks] pointers to [T, n_embd]
    int n_blocks,
    const float * d_cur,                // [T, n_embd]
    const float * d_query,              // [n_q, n_embd]
    float * d_out,                      // [n_q, T, n_embd]
    int n_embd, int n_q, int T, float eps, cudaStream_t stream);

void launch_inkling2_rmsnorm_concat2(
    const float * d_a, const float * d_b, float * d_out,
    int n_embd, int T, float eps, cudaStream_t stream);

void launch_inkling2_loop_combine(
    const float * d_cur, const float * d_msg,
    const float * d_gain_pre, const float * d_forget_pre,
    float * d_out, int64_t n, cudaStream_t stream);

void launch_inkling2_shortconv(
    const float * d_x,          // [T, C]
    const float * d_w,          // [C, 4] (ggml [n_conv=4, C])
    const float * d_dyn,        // optional [T, n_head_kv, 4] (nullptr for static conv)
    float *       d_state,      // [3, C] (oldest=0 .. newest=2)
    float *       d_out,        // [T, C]
    int T, int C, int n_head_kv, cudaStream_t stream,
    const float * d_res = nullptr);

void launch_inkling2_rmsnorm_fast(
    const float * d_x, const float * d_w, float * d_out,
    int n_embd, int T, float eps, cudaStream_t stream);

void launch_inkling2_rmsnorm3_fast(
    const float * d_x0, const float * d_x1, const float * d_x2,
    const float * d_w0, const float * d_w1, const float * d_w2,
    float * d_o0, float * d_o1, float * d_o2,
    int n_embd, int T, float eps, cudaStream_t stream);

void launch_inkling2_loop_ctrl_up_combine(
    const DeviceTensor & W_gain,   // F32 [gate_rank, n_embd]
    const DeviceTensor & W_forget, // F32 [gate_rank, n_embd]
    const float * d_hid,           // [T, gate_rank]
    const float * d_cur,           // [T, n_embd]
    const float * d_msg,           // [T, n_embd]
    float *       d_out,           // [T, n_embd]
    int T, cudaStream_t stream);

void launch_inkling2_loop_ctrl_qkv_decode(
    const float * d_in0, const float * d_in1, const float * d_in2,
    const float * d_msg0, const float * d_msg1, const float * d_msg2,
    const DeviceTensor * ctrl_down,   // [3] F32 [2*n_embd, gate_rank]
    const DeviceTensor * ctrl_gain,   // [3] F32 [gate_rank, n_embd]
    const DeviceTensor * ctrl_forget, // [3] F32 [gate_rank, n_embd]
    float * d_ctrl_in3,               // scratch [3, 2*n_embd]
    float * d_ctrl_hid3,              // scratch [3, gate_rank]
    float * d_out0, float * d_out1, float * d_out2,
    int n_embd, int gate_rank, float eps, cudaStream_t stream);

void launch_inkling2_qkv_fused_q8_0(
    int n_projs,
    const DeviceTensor * const * Ws, // up to 6 Q8_0 SoA weights sharing the same d_xq8
    float * const * Ys,              // corresponding output buffers [T, N_i]
    const void * d_xq8,              // pre-quantized BlockQ8_1_Act [T, K/32]
    int T, cudaStream_t stream);

void launch_inkling2_sigmoid_gate_quant_q8_1(
    const float * d_x, const float * d_gate, void * d_xq8,
    int K, int T, cudaStream_t stream);

void launch_inkling2_rmsnorm_quant_f16_contig(
    int n_inputs,
    const float * d_x0, const float * d_x1, const float * d_x2,
    const float * d_w0, const float * d_w1, const float * d_w2,
    void * d_out_h2_d, int n_embd, int T, float eps, cudaStream_t stream);

void launch_inkling2_qkv_fused_q8_0_h2(
    int n_projs, int n_inputs,
    const DeviceTensor * const * Ws,
    const int * in_idx,
    float * const * Ys,
    const void * d_in_h2_d, int T, cudaStream_t stream);

void launch_inkling2_sigmoid_gate_quant_f16_contig(
    const float * d_x, const float * d_gate, void * d_out_h2_d,
    int K, int T, cudaStream_t stream);

void launch_inkling2_gemv_q8_0_h2(
    const DeviceTensor & W, const void * d_in_h2_d, float * d_y,
    int T, cudaStream_t stream);

void launch_inkling2_quant_f16_contig(
    const float * d_x, void * d_out_h2_d, int K, int T, cudaStream_t stream);

void launch_inkling2_kv_conv_norm_write_t1(
    const float * d_k_raw, const float * d_k_w, const float * d_k_dyn, float * d_k_state,
    const float * d_v_raw, const float * d_v_w, const float * d_v_dyn, float * d_v_state,
    void * d_k_cache, void * d_v_cache, KvCacheType kv_type, const int32_t * d_pos,
    int n_head_kv, int head_dim, int cache_cap, float qk_clip, float rms_eps,
    cudaStream_t stream);

void launch_inkling2_qr_l2_clip_fused_t1(
    float * d_q, const float * d_q_scale,
    float * d_r, const float * d_r_scale,
    const int32_t * d_pos,
    float qk_clip, float rel_clip, float log_floor, float log_alpha,
    int n_head, int head_dim, int n_rel, cudaStream_t stream);

void launch_inkling2_linear_sm60_or_default(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream);

void launch_inkling2_router_gemv_f32(
    const DeviceTensor & W, const float * d_x, float * d_y,
    int T, cudaStream_t stream);

void launch_inkling2_moe_route_sigmoid_fast(
    const float * d_logits, const float * d_bias,
    int32_t * d_top_ids, float * d_top_w,
    int T, int n_expert, int k_used, bool renorm, float scale, cudaStream_t stream);

void launch_inkling2_l2_clip(
    float * d_x,                // [T, n_head, head_dim] in-place
    const float * d_scale,      // optional [n_head, head_dim] (nullptr if none)
    const int32_t * d_pos,      // optional [T] for log-scale tau (nullptr if SWA)
    float threshold, float log_floor, float log_alpha,
    int T, int n_head, int head_dim, cudaStream_t stream);

void launch_inkling2_head_rmsnorm(
    float * d_x, int T, int n_head, int head_dim, float eps, cudaStream_t stream);

void launch_inkling2_rel_bias_table(
    const float * d_r,          // [T, n_head, n_rel]
    const float * d_rel_bias,   // [n_extent, n_rel] (ggml [n_rel, n_extent])
    const float * d_sinks,      // [T, n_head]
    float * d_bias_table,       // [T, n_head, n_extent + 1]
    int T, int n_head, int n_rel, int n_extent, cudaStream_t stream);

void launch_inkling2_kv_write(
    const float * d_k, const float * d_v,
    void * d_k_cache, void * d_v_cache, KvCacheType kv_type,
    const int32_t * d_pos, int T, int n_kv, int head_dim, int cache_cap, cudaStream_t stream);

void launch_inkling2_attn(
    const float * d_q,              // [T, n_head, head_dim]
    const void *  d_k_cache,        // [cache_cap, n_kv, head_dim]
    const void *  d_v_cache,        // [cache_cap, n_kv, head_dim]
    KvCacheType   kv_type,
    const float * d_bias_table,     // [T, n_head, n_extent + 1]
    const int32_t * d_pos,          // [T]
    float *       d_out,            // [T, n_head, head_dim]
    void *        d_scratch,
    size_t        scratch_bytes,
    int T, int n_head, int n_kv, int head_dim,
    int swa_window, int n_extent, int cache_cap, float scale,
    cublasHandle_t cublas, int kv_len_max, cudaStream_t stream);

void launch_inkling2_sigmoid_gate_mul(
    float * d_x, const float * d_gate, int64_t n, cudaStream_t stream);

void launch_inkling2_moe_ffn(
    cublasHandle_t cublas,
    const DeviceTensor & gate_up_exps, // Q8_0 SoA [n_embd, 2*ff, n_expert]
    const DeviceTensor & down_exps,    // Q8_0 SoA [ff, n_embd, n_expert]
    const float * d_x,                 // [T, n_embd]
    const int32_t * d_top_ids,         // [T, k_used]
    const float * d_top_w,             // [T, k_used]
    float * d_out,                     // [T, n_embd]
    void * d_scratch, size_t scratch_bytes,
    int T, int k_used, float situ_beta, float situ_linear_beta, cudaStream_t stream);

void launch_inkling2_moe_ffn_kquant(
    const DeviceTensor & gate_up_exps, const DeviceTensor & down_exps,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, float situ_beta, float situ_linear_beta, cudaStream_t stream);

void launch_inkling2_logit_softcap(
    float * d_logits, int64_t n, float cap, cudaStream_t stream);

// --- Qwen3.5 / Qwen3.6 / Qwen3.8 Next (GDN + d256 Partial-RoPE GQA + Softmax MoE + Gated Shared Expert) ---
void launch_qwen_gdn_gate_and_beta(
    float * d_alpha,          // [T, n_v_heads] in-place -> g = ssm_a * softplus(alpha + dt_bias)
    const float * d_dt_bias,  // [n_v_heads]
    const float * d_ssm_a,    // [n_v_heads]
    float * d_beta,           // [T, n_v_heads] in-place -> sigmoid(beta)
    int T, int n_v_heads, cudaStream_t stream);

void launch_qwen_gdn_packed_d128(
    float * d_qkv_conv,       // [T, 2*n_k*128 + n_v*128] (Q & K normalized in-place!)
    const float * d_g,        // [T, n_v_heads]
    const float * d_beta,     // [T, n_v_heads]
    float * d_state,          // [n_v_heads, 128, 128]
    float * d_out,            // [T, n_v_heads, 128]
    int T, int n_k_heads, int n_v_heads, float eps, float scale,
    cudaStream_t stream, float * d_state_snap = nullptr);

void launch_rms_norm_silu_gate_d128(
    float * d_o,              // [T, n_v_heads, 128] in-place
    const float * d_norm_w,   // [128]
    const float * d_gate,     // [T, n_v_heads, 128]
    int T, int n_v_heads, float eps, cudaStream_t stream);

void launch_qwen_qgate_split_norm_rope_d256(
    const float * d_q_full,   // [T, n_head, 512] interleaved [Q(256), gate(256)]
    const float * d_q_norm_w, // [256]
    const int32_t * d_pos,    // [T]
    float * d_q_out,          // [T, n_head, 256]
    float * d_gate_out,       // [T, n_head, 256]
    int T, int n_head, int rope_dim, float rms_eps, float rope_freq_base, cudaStream_t stream);

void launch_qwen_k_norm_rope_d256(
    float * d_k,              // [T, n_kv, 256] in-place
    const float * d_k_norm_w, // [256]
    const int32_t * d_pos,    // [T]
    int T, int n_kv, int rope_dim, float rms_eps, float rope_freq_base, cudaStream_t stream);

void launch_gqa_kv_write_d256(
    const float * d_k, const float * d_v,
    void * d_k_cache, void * d_v_cache, bool kv_is_q8_0,
    const int32_t * d_pos, int T, int n_kv, cudaStream_t stream);

void launch_gqa_attn_d256(
    const float * d_q, const void * d_k_cache, const void * d_v_cache, bool kv_is_q8_0,
    const int32_t * d_pos, float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int n_head, int n_kv, float scale,
    cublasHandle_t cublas, int kv_len_max, cudaStream_t stream);

void launch_moe_route_softmax(
    const float * d_logits, int32_t * d_top_ids, float * d_top_w,
    int n_tokens, int n_expert, int k, bool renorm, float scale, cudaStream_t stream);

void launch_qwen_moe_ffn_gated_shared(
    cublasHandle_t cublas,
    const DeviceTensor & gate_exps, const DeviceTensor & up_exps, const DeviceTensor & down_exps,
    const DeviceTensor & gate_shexp, const DeviceTensor & up_shexp, const DeviceTensor & down_shexp,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    const float * d_shexp_gate_logit, // [T] raw scalar logit before sigmoid!
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, cudaStream_t stream);

// --- Qwen3.8-Flash-Next (qwen4exp: low-rank mHC + PLE Engram + QSA Sparse Indexer) ---
void launch_hc_init(const float * d_embd, float * d_res_hc, int H, int hc, int T, cudaStream_t stream);
void launch_hc_grouped_rmsnorm(
    const float * d_x, const float * d_w, float * d_out,
    int H, int hc, int T, float eps, cudaStream_t stream);
void launch_scale_silu_inplace(float * d_x, float scale, int64_t n, cudaStream_t stream);
void launch_hc_pre_gated_mean(
    const float * d_xn, const float * d_gate, float * d_mixed,
    int H, int hc, int T, cudaStream_t stream);
void launch_hc_combine(
    float * d_res_hc, const float * d_block_out, const float * d_inject,
    float inject_scale, int H, int hc, int T, cudaStream_t stream);

void launch_ple_gate_value(
    const float * d_key, const float * d_query, const float * d_value,
    float * d_gated, int H, int hc, int T, cudaStream_t stream);
void launch_ple_dilated_conv1d_silu_add(
    const float * d_norm, const float * d_conv_w, float * d_conv_state,
    const float * d_gated, float * d_res_hc,
    int hc_dim, int kern, int dil, int T, cudaStream_t stream);

void launch_qsa_k_raw_write_and_pool(
    const float * d_k_raw, float * d_k_raw_cache, float * d_k_pool_cache,
    const float * d_k_norm_w, const int32_t * d_pos,
    int T, int kpool, int rope_dim, float rms_eps, float rope_freq_base, cudaStream_t stream);
void launch_qsa_q_norm_rope(
    float * d_q, const float * d_q_norm_w, const int32_t * d_pos,
    int T, int n_idx_h, int rope_dim, float rms_eps, float rope_freq_base, cudaStream_t stream);
void launch_qsa_select_topk(
    const float * d_q_idx, const float * d_k_pool_cache, const int32_t * d_pos,
    int32_t * d_sel_pos, int32_t * d_sel_len,
    void * d_scratch, size_t scratch_bytes,
    int T, int n_idx_h, int kpool, int top_k, int kv_len_max, cudaStream_t stream);
void launch_gqa_attn_d256_qsa(
    const float * d_q, const void * d_k_cache, const void * d_v_cache, bool kv_is_q8_0,
    const int32_t * d_pos, const int32_t * d_sel_pos, const int32_t * d_sel_len,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int n_head, int n_kv, int max_sel, float scale,
    cublasHandle_t cublas, int kv_len_max, int top_k, int kpool, cudaStream_t stream);

// --- Fast Q3_K SoA & sm_60 Native FP16 (half2) / sm_61+ dp4a K-Quant & MoE Kernels ---
void launch_embedding_q3_k_soa(
    const DeviceTensor & tok_embd, const int32_t * d_tokens, float * d_out,
    int n_embd, int hc_mult, int n_tokens, cudaStream_t stream);

void launch_dequant_q3_k_soa(
    const DeviceTensor & W, int64_t b0, int64_t nblk, void * d_dst, bool dst_f16, cudaStream_t stream);

void launch_linear_q3_k_soa_prequant(
    const DeviceTensor & W, const void * d_xq8, float * d_y, int T, cudaStream_t stream);

bool try_launch_linear_kquant_prefill_sm60_f16(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream);

bool try_launch_linear_kquant_sm60_h2(
    const DeviceTensor & W, const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream);

bool try_launch_linear_multi_sm60_h2(
    const DeviceTensor * const * Ws, float * const * Ys, int n_mat,
    const float * d_x, int T, void * d_scratch, size_t scratch_bytes, cudaStream_t stream);

void launch_moe_gate_up_q3_k_soa_only(
    const DeviceTensor & gate_exps, const DeviceTensor & up_exps,
    const void * d_xq8, const int32_t * d_top_ids, float * d_act,
    int T, int k_used, float clamp_exp, cudaStream_t stream);

void launch_moe_down_q3_k_soa_only(
    const DeviceTensor & down_exps, const void * d_aq8,
    const int32_t * d_top_ids, const float * d_top_w, float * d_out,
    int T, int k_used, cudaStream_t stream);

bool try_launch_moe_q3k_q4k_fast(
    cublasHandle_t cublas,
    const DeviceTensor & gate_exps, const DeviceTensor & up_exps, const DeviceTensor & down_exps,
    const DeviceTensor & gate_shexp, const DeviceTensor & up_shexp, const DeviceTensor & down_shexp,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, float clamp_exp, float clamp_shexp, cudaStream_t stream);

bool try_launch_linear_q6k_soa_prefill_sm60_f16(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream);

bool try_launch_linear_kquant_prefill_sm61_dp4a(
    const DeviceTensor & W, const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream);

bool try_launch_moe_q4k_q6k_pascal(
    cublasHandle_t cublas,
    const DeviceTensor & gate_exps, const DeviceTensor & up_exps, const DeviceTensor & down_exps,
    const DeviceTensor & gate_shexp, const DeviceTensor & up_shexp, const DeviceTensor & down_shexp,
    const float * d_x, const int32_t * d_top_ids, const float * d_top_w,
    float * d_out, void * d_scratch, size_t scratch_bytes,
    int T, int k_used, float clamp_exp, float clamp_shexp, cudaStream_t stream);

// --- Gemma 4 (PLE, Proportional NeoX RoPE, SWA + Shared-KV MQA, GeGLU, Softcap) ---
void launch_gemma4_scale(float * d_x, int64_t n, float scale, cudaStream_t stream);
void launch_gemma4_scale_dev(float * d_x, const float * d_scale, int64_t n, cudaStream_t stream);

void launch_gemma4_ple_combine(
    const float * d_proj, const float * d_raw_ple, const float * d_proj_norm_w,
    float * d_ple_out, int T, int n_layers, int n_embd, int ple_dim,
    float rms_eps, cudaStream_t stream);

void launch_gemma4_geglu(const float * d_gate, const float * d_up, float * d_out, int64_t n, cudaStream_t stream);

void launch_gemma4_ple_gate_gelu_mul(
    float * d_gate, const float * d_ple,
    int T, int n_layers, int il, int ple_dim, int ple_tok_off, cudaStream_t stream);

void launch_gemma4_ple_f16_t1(
    const void * d_w_gate_f16, const void * d_w_proj_f16,
    const float * d_x, const float * d_ple,
    float * d_gate_tmp, float * d_out,
    int H, int ple_dim, int n_layers, int il, int ple_tok_off, cudaStream_t stream);

void launch_gemma4_post_norm_add_scale(
    const float * d_res, const float * d_sub, const float * d_norm_w,
    const float * d_layer_scale, float * d_out,
    int dim, int n_rows, float eps, cudaStream_t stream);

void launch_gemma4_logit_softcap(float * d_logits, int64_t n, float cap, cudaStream_t stream);

void launch_gemma4_qkv_norm_rope_kv_write(
    float * d_q, float * d_k, const float * d_v,
    const float * d_q_norm_w, const float * d_k_norm_w, const float * d_rope_freqs,
    void * d_k_cache, void * d_v_cache, bool kv_q8_0,
    const int32_t * d_pos, int T, int n_head, int n_kv,
    int head_dim, int rope_dim, int max_ctx, float rms_eps, float freq_base,
    bool has_kv, cudaStream_t stream);

void launch_gemma4_attn(
    cublasHandle_t cublas,
    const float * d_q, const void * d_k_cache, const void * d_v_cache, bool kv_q8_0,
    const int32_t * d_pos, float * d_out,
    void * d_scratch, size_t scratch_bytes,
    int T, int n_head, int n_kv, int head_dim, int max_ctx,
    int swa_window, int pos_min, int pos_max, cudaStream_t stream);

bool try_launch_gemma4_linear_sm60_f16(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream);

} // namespace rookery
