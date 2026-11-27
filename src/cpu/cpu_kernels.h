#pragma once

#include "core/types.h"
#include "cpu/cpu_pool.h"

#include <cstddef>
#include <cstdint>

namespace rookery::cpu {

// Zero-copy view of a host/mmap tensor from ShardedGguf (or aligned owned buffer).
struct CpuTensor {
    const void * ptr   = nullptr;
    DType        dtype = DType::F32;
    uint64_t     ne[4] = {1, 1, 1, 1};
    size_t       nbytes = 0;

    bool valid() const { return ptr != nullptr; }
};

// CPU activation quant block for Q8_0 / Q4_0 / Q5_0 / IQ4_NL weights (32 elems, aligned for AVX2!)
struct alignas(32) CpuActQ8_0 {
    int8_t qs[32];
    float  d;
    float  s;      // d * sum(qs[0..31]) for zero-overhead Q4_0 / Q5_0 bias terms
    float  pad[6]; // 64 bytes total = 1 cache line, qs is 32-byte aligned!
};
static_assert(sizeof(CpuActQ8_0) == 64, "CpuActQ8_0 must be 64 bytes");

// CPU activation quant block for Q6_K / K-quant weights (256 elems, aligned for AVX2!)
struct alignas(32) CpuActQ8_K {
    int8_t  qs[256];   // offset 0   (32-byte aligned!)
    int16_t bsums[16]; // offset 256 (32-byte aligned!)
    float   d;         // offset 288
    uint8_t pad[28];   // 320 bytes = 5 cache lines, 32-byte aligned!
};
static_assert(sizeof(CpuActQ8_K) == 320, "CpuActQ8_K must be 320 bytes");
static_assert(offsetof(CpuActQ8_K, qs) % 32 == 0, "CpuActQ8_K::qs must be 32-byte aligned");
static_assert(offsetof(CpuActQ8_K, bsums) % 32 == 0, "CpuActQ8_K::bsums must be 32-byte aligned");

// 64-byte-aligned SoA-per-position Q8_0 MLA KV entry (identical layout to GPU MlaKv576Q8Pos!)
struct alignas(64) CpuMlaKv576Q8Pos {
    int8_t   qs[576];
    uint16_t d[18];
    uint16_t pad[14];
};
static_assert(sizeof(CpuMlaKv576Q8Pos) == 640, "CpuMlaKv576Q8Pos must be 640 bytes");

// 64-byte-aligned SoA-per-position Q4_0 MLA KV entry (18 x 32 nibble-packed elems + 18 fp16 scales = 384B)
struct alignas(64) CpuMlaKv576Q4Pos {
    uint8_t  qs[288];
    uint16_t d[18];
    uint8_t  pad[60];
};
static_assert(sizeof(CpuMlaKv576Q4Pos) == 384, "CpuMlaKv576Q4Pos must be 384 bytes");

// 16-byte-aligned per-head-position Q8_0 GQA KV entry (head_dim=128 = 4 x 32 blocks + 4 FP32 scales)
struct alignas(16) CpuGqaKv128Q8Pos {
    int8_t qs[128];
    float  d[4];
};
static_assert(sizeof(CpuGqaKv128Q8Pos) == 144, "CpuGqaKv128Q8Pos must be 144 bytes");

// 16-byte-aligned per-head-position Q4_0 GQA KV entry (head_dim=128 = 4 x 32 4-bit blocks + 4 FP32 scales)
struct alignas(16) CpuGqaKv128Q4Pos {
    uint8_t qs[64];
    float   d[4];
};
static_assert(sizeof(CpuGqaKv128Q4Pos) == 80, "CpuGqaKv128Q4Pos must be 80 bytes");

// Fast FP16 / BF16 conversion helpers
float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);
float bf16_to_f32(uint16_t b);

// Quantize FP32 rows [T, K] to CpuActQ8_0 [T, K/32] or CpuActQ8_K [T, K/256]
void quantize_row_q8_0(const float * x, CpuActQ8_0 * dst, int K);
void quantize_rows_q8_0(ThreadPool & pool, const float * x, CpuActQ8_0 * dst, int T, int K);
void quantize_row_q8_k(const float * x, CpuActQ8_K * dst, int K);
void quantize_rows_q8_k(ThreadPool & pool, const float * x, CpuActQ8_K * dst, int T, int K);

// Embedding lookup: dst[t, 0..n_embd) = W[tokens[t], :]
void embedding_lookup(const CpuTensor & W, const int32_t * tokens, float * dst, int n_embd, int T);

// Dequantize one full row of K elements to FP32 (supports all CPU linear dtypes).
void dequant_row_to_f32(DType dt, const void * row_ptr, int K, float * dst);

// RMSNorm & fused residual add + RMSNorm
void rmsnorm(const float * x, const float * w, float * y, int n_embd, int T, float eps);
void rmsnorm_unweighted(const float * x, float * y, int n_embd, int T, float eps);
void add_rmsnorm(float * x_inout, const float * sub_out, const float * w, float * y, int n_embd, int T, float eps);
void add_inplace(float * a, const float * b, size_t n);
void scale_inplace(float * a, size_t n, float scale);

// Linear projection Y[T, M] = X[T, K] * W^T where W has shape [K, M] (ne[0]=K, ne[1]=M).
// Supports Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, Q4_0, Q5_0, IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S, BF16, F16, F32.
// Optional pre-quantized activations `xq8` (for 32-elem quants) or `xq8k` (for 256-elem quants).
void linear(ThreadPool & pool,
            const CpuTensor & W,
            const float * x,
            const CpuActQ8_0 * xq8,
            const CpuActQ8_K * xq8k,
            float * y,
            int T);

// Fused multi-projection sharing ONE pre-quantized input X (1 thread-pool barrier for all N projections!).
// All W[i] must share the same K = W[i].ne[0].
struct LinearSpec {
    const CpuTensor * W;
    float *           y;
};
void linear_multi(ThreadPool & pool,
                  const LinearSpec * specs,
                  int n_specs,
                  const float * x,
                  const CpuActQ8_0 * xq8,
                  const CpuActQ8_K * xq8k,
                  int T);

// Dense FFN activations: SwiGLU & GeGLU (tanh-approx)
void silu_mul(const float * gate, const float * up, float * out, size_t n);
void silu_mul_mt(ThreadPool & pool, const float * gate, const float * up, float * out, size_t n);
void geglu_tanh_mul(float * gate_inout, const float * up, size_t n);
void geglu_tanh_mul_mt(ThreadPool & pool, float * gate_inout, const float * up, size_t n);
void softcap_tanh(float * logits, int n, float cap);

// Gemma 4 kernels (Dense+PLE E2B/E4B & MoE 26B-A4B)
void gemma4_ple_combine(
    const float * proj, const float * raw_ple, const float * proj_norm_w,
    float * ple_out, int T, int n_layers, int ple_dim, float rms_eps, int n_embd);

void gemma4_qk_norm_rope(
    float * qk, const float * norm_w, const float * rope_freqs, const int32_t * pos,
    int T, int n_head, int head_dim, int rope_dim, float freq_base, float eps);

void gemma4_vnorm_kv_write(
    const float * k_in, const float * v_in,
    void * k_cache, void * v_cache, KvCacheType kv_type,
    const int32_t * pos, int T, int n_kv, int head_dim, int max_slots, bool is_swa, float eps);

void gemma4_gqa_attn(
    ThreadPool & pool,
    const float * q, const void * k_cache, const void * v_cache, KvCacheType kv_type,
    const int32_t * pos, float * out,
    int T, int n_head, int n_kv, int head_dim, int max_slots, int swa_window);

// Softmax top-k MoE router (Gemma 4 26B-A4B)
void moe_route_softmax(
    const float * logits, int32_t * top_ids, float * top_w,
    int T, int n_expert, int k, bool renorm, float scale);

// Gemma 4 MoE expert FFN (supports fused gate_up_exps or separate gate/up, GeGLU-tanh, per-expert down scale)
void gemma4_moe_ffn(
    ThreadPool & pool,
    const CpuTensor & gate_up_exps,
    const CpuTensor & gate_exps, const CpuTensor & up_exps,
    const CpuTensor & down_exps, const float * down_exps_scale,
    const float * x, const CpuActQ8_0 * xq8, const CpuActQ8_K * xq8k,
    const int32_t * top_ids, const float * top_w,
    float * out, int T, int k);

// Fused T=1 dense SwiGLU FFN: streams 256-row blocks of gate/up -> L1d SiLU + quant -> down
void dense_ffn_swiglu_t1(
    ThreadPool & pool,
    const CpuTensor & w_gate,
    const CpuTensor & w_up,
    const CpuTensor & w_down,
    const float * x,
    const CpuActQ8_0 * xq8,
    const CpuActQ8_K * xq8k,
    float * out);

// Fused T=1 dense GeGLU-tanh FFN (Gemma 4): streams 256-row blocks of gate/up -> L1d GeGLU + quant -> down
void dense_ffn_geglu_t1(
    ThreadPool & pool,
    const CpuTensor & w_gate,
    const CpuTensor & w_up,
    const CpuTensor & w_down,
    const float * x,
    const CpuActQ8_0 * xq8,
    const CpuActQ8_K * xq8k,
    float * out);

// Row-wise bias add: y[t, j] += bias[j] for t in [0, T), j in [0, M)
void add_bias_rows(float * y, const float * bias, int T, int M);

// Standard GQA Transformer kernels (Qwen2 / Qwen2.5, head_dim=128, NeoX split-half RoPE)
void neox_rope_qk(
    float * q, float * k, const int32_t * pos,
    int T, int n_head, int n_head_kv, int head_dim, float rope_freq_base);

void gqa_kv_write_d128(
    const float * k, const float * v, const int32_t * pos,
    void * k_cache, void * v_cache, KvCacheType kv_type,
    int T, int n_head_kv, int max_ctx);

void gqa_attn_d128(
    ThreadPool & pool,
    const float * q, const void * k_cache, const void * v_cache, KvCacheType kv_type,
    const int32_t * pos, float * out,
    int T, int n_head, int n_head_kv, int max_ctx, float scale);

// KDA linear-attention kernels
void ssm_conv1d_silu_qkv_l2(
    ThreadPool & pool,
    const float * q, const float * k, const float * v,
    const float * wq, const float * wk, const float * wv,
    float * sq, float * sk, float * sv,
    float * qc, float * kc, float * vc,
    int T, int n_heads, float eps,
    float * sq_snap = nullptr, float * sk_snap = nullptr, float * sv_snap = nullptr);

void kda_gate_and_beta(
    ThreadPool & pool,
    float * f_inout, const float * dt_bias, const float * ssm_a,
    float * beta_inout, float lower_bound, int T, int n_heads, int head_dim,
    bool negate_a = false);

void gated_delta_net_d128(
    ThreadPool & pool,
    const float * q, const float * k, const float * v,
    const float * g, const float * beta,
    float * state, float * out,
    int T, int n_heads, float scale,
    float * state_snap = nullptr);

void rms_norm_sigmoid_gate_d128(
    ThreadPool & pool,
    float * o_inout, const float * norm_w, const float * gate,
    int T, int n_heads, float eps);

// RoPE-576 Gated MLA kernels
void mla_rope_split_q(
    const float * q_all, const int32_t * pos,
    float * q_nope, float * q_pe,
    int T, int n_heads, int qk_nope, int rope_dim, float rope_freq_base);

void mla_kv_norm_rope_write_576(
    const float * kv_all, const float * norm_w, const int32_t * pos,
    void * kv_cache, KvCacheType kv_type,
    int T, float rms_eps, float rope_freq_base);

// Zero-copy batched per-head linear: W[K, M, n_heads] (Q8_0, Q6_K, F16, BF16, F32), X[T, n_heads, K] -> Y[T, n_heads, M]
void batched_head_proj(
    ThreadPool & pool,
    const CpuTensor & W, const float * x, float * y,
    int T);

void mla_concat_q576(
    const float * q_abs, const float * q_pe, float * q576,
    int T, int n_heads);

void mla_attn_d576_v512(
    ThreadPool & pool,
    const float * q576, const void * kv_cache, KvCacheType kv_type,
    const int32_t * pos, float * out,
    int T, int n_heads, float scale);

void mla_head_sigmoid_gate(
    float * o_inout, const float * gate,
    int T, int n_heads, int head_dim);

// MoE routing (grouped top-k sigmoid router matching GPU moe_route_sigmoid_kernel verbatim!)
void moe_route_sigmoid(
    const float * logits, const float * bias,
    int32_t * top_ids, float * top_w,
    int T, int n_expert, int k, bool renorm, float scale,
    int n_expert_groups, int n_group_used);

// Fused MoE + shared expert FFN (supports all quant dtypes, SwiGLU clamp, optional sigmoid shared-expert gate, expert-grouped prefill!)
void moe_ffn_with_shared(
    ThreadPool & pool,
    const CpuTensor & gate_exps, const CpuTensor & up_exps, const CpuTensor & down_exps,
    const CpuTensor & gate_shexp, const CpuTensor & up_shexp, const CpuTensor & down_shexp,
    const float * x, const CpuActQ8_0 * xq8, const CpuActQ8_K * xq8k,
    const int32_t * top_ids, const float * top_w,
    float * out,
    int T, int k, float clamp_exp, float clamp_shexp,
    const float * shexp_gate = nullptr);

// Parallel argmax over FP32 logits
int32_t argmax_f32(const float * logits, int n_vocab);

} // namespace rookery::cpu
