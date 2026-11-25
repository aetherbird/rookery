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

// CPU activation quant block for Q8_0 weights (32 elems, aligned for AVX2!)
struct alignas(32) CpuActQ8_0 {
    int8_t qs[32];
    float  d;
    float  pad[7]; // 64 bytes total = 1 cache line, qs is 32-byte aligned!
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

// RMSNorm & fused residual add + RMSNorm
void rmsnorm(const float * x, const float * w, float * y, int n_embd, int T, float eps);
void add_rmsnorm(float * x_inout, const float * sub_out, const float * w, float * y, int n_embd, int T, float eps);
void add_inplace(float * a, const float * b, size_t n);

// Linear projection Y[T, M] = X[T, K] * W^T where W has shape [K, M] (ne[0]=K, ne[1]=M).
// Supports Q8_0, Q6_K, Q4_K, Q5_K, Q4_0, BF16, F16, F32.
// Optional pre-quantized activations `xq8` (for Q8_0) or `xq8k` (for Q6_K) avoid redundant quantization.
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

// Dense FFN: gate/up + SiLU + down
void silu_mul(const float * gate, const float * up, float * out, size_t n);

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
    void * kv_cache, bool kv_q8_0,
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
    const float * q576, const void * kv_cache, bool kv_q8_0,
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

// Fused MoE + shared expert FFN (supports Q6_K and Q8_0 experts, SwiGLU clamp, expert-grouped prefill!)
void moe_ffn_with_shared(
    ThreadPool & pool,
    const CpuTensor & gate_exps, const CpuTensor & up_exps, const CpuTensor & down_exps,
    const CpuTensor & gate_shexp, const CpuTensor & up_shexp, const CpuTensor & down_shexp,
    const float * x, const CpuActQ8_0 * xq8, const CpuActQ8_K * xq8k,
    const int32_t * top_ids, const float * top_w,
    float * out,
    int T, int k, float clamp_exp, float clamp_shexp);

// Parallel argmax over FP32 logits
int32_t argmax_f32(const float * logits, int n_vocab);

} // namespace rookery::cpu
