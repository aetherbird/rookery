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
    cudaStream_t  stream);

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

// --- Attention & KV Cache (F16 and Q8_0 KV) ---
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

void launch_gqa_attn(
    const float *   d_q,
    const void *    d_k_cache,
    const void *    d_v_cache,
    bool            kv_is_q8_0,
    const int32_t * d_pos,
    float *         d_out,
    void *          d_scratch,
    int             T,
    int             n_heads,
    int             n_kv_heads,
    int             head_dim,
    float           scale,
    cudaStream_t    stream);

// --- Maxwell Block-Sparse Attention (MSA) ---
void launch_idx_k_write_f16(
    const float *   d_ik,
    void *          d_ik_cache,
    const int32_t * d_pos,
    int             T,
    int             idx_dim,
    cudaStream_t    stream);

void launch_msa_indexer_topk(
    const float *   d_iq,
    const void *    d_ik_cache,
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
    bool            kv_is_q8_0,
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

void launch_l2_norm_d128(float * d_x, int T, int n_heads, float eps, cudaStream_t stream);

void launch_kda_gate_and_beta(
    float * d_f, const float * d_dt_bias, const float * d_ssm_a,
    float * d_beta, float lower_bound, int T, int n_heads, int head_dim, cudaStream_t stream);

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
    int T, int C, int n_head_kv, cudaStream_t stream);

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
    void * d_k_cache, void * d_v_cache, bool kv_is_q8_0,
    const int32_t * d_pos, int T, int n_kv, int head_dim, int cache_cap, cudaStream_t stream);

void launch_inkling2_attn(
    const float * d_q,              // [T, n_head, head_dim]
    const void *  d_k_cache,        // [cache_cap, n_kv, head_dim]
    const void *  d_v_cache,        // [cache_cap, n_kv, head_dim]
    bool          kv_is_q8_0,
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

} // namespace rookery
