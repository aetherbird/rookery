#pragma once

#include "cpu/cpu_kernels.h"
#include "cpu/cpu_pool.h"
#include "gguf/sharded.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rookery {

struct LayerHook;

namespace cpu {

struct Qwen35CpuConfig {
    int   n_blocks             = 32;
    int   n_layers             = 32;
    int   n_nextn              = 0;
    int   n_embd               = 2048;
    int   n_vocab              = 248320;
    int   n_head               = 16;
    int   n_head_kv            = 2;
    int   head_dim             = 256;
    int   rope_dim             = 64;
    float rope_freq_base       = 10000000.0f;
    float rms_eps              = 1e-6f;

    // GDN linear-attention dimensions
    int   ssm_d_conv           = 4;
    int   ssm_d_state          = 128;
    int   ssm_n_group          = 16;   // n_k_heads
    int   ssm_dt_rank          = 32;   // n_v_heads
    int   ssm_d_inner          = 4096; // n_v_heads * ssm_d_state
    int   full_attn_interval   = 4;

    // Dense or MoE FFN
    bool  is_dense             = true;
    int   n_ff                 = 12288;
    int   n_expert             = 0;
    int   n_expert_used        = 0;
    int   n_ff_exp             = 0;
    int   n_ff_shexp           = 0;
    float expert_weights_scale = 1.0f;

    int   max_ctx              = 32768;
    bool  kv_q8_0              = true;

    std::vector<bool> is_recr; // [n_layers]: true = GDN, false = Full GQA
};

struct Qwen35CpuLayer {
    bool is_recr = false;

    CpuTensor attn_norm, post_attn_norm;

    // GDN linear attention
    CpuTensor wqkv, wqkv_gate;
    CpuTensor ssm_conv1d, ssm_dt, ssm_a, ssm_alpha, ssm_beta, ssm_norm, ssm_out;

    // Full GQA attention
    CpuTensor wq, wk, wv, wo;
    CpuTensor attn_q_norm, attn_k_norm;

    // Dense SwiGLU FFN (qwen35)
    CpuTensor ffn_gate, ffn_up, ffn_down;

    // MoE + Sigmoid-Gated Shared Expert (qwen35moe)
    CpuTensor ffn_gate_inp;
    CpuTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    CpuTensor ffn_gate_inp_shexp;
    CpuTensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

class Qwen35CpuModel {
public:
    Qwen35CpuModel() = default;
    ~Qwen35CpuModel() = default;

    Qwen35CpuModel(const Qwen35CpuModel &) = delete;
    Qwen35CpuModel & operator=(const Qwen35CpuModel &) = delete;

    void load(const std::string & gguf_path,
              int max_ctx = 32768,
              bool kv_q8_0 = true,
              int n_threads = 0);

    void reset_state();

    struct GdnCheckpoint {
        int pos = 0;
        std::vector<std::vector<float>> conv;
        std::vector<std::vector<float>> ssm;
    };
    void save_gdn_checkpoint(GdnCheckpoint & out, int pos) const;
    void restore_gdn_checkpoint(const GdnCheckpoint & cp);

    bool has_mtp() const { return false; }
    void set_layer_hook(LayerHook * /*hook*/) {}
    void warmup_decode_graphs() {}

    void forward(const std::vector<int32_t> & tokens,
                 const std::vector<int32_t> & positions,
                 std::vector<float> & out_logits);

    int32_t forward_decode_greedy(int32_t token, int32_t pos);
    void    forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits);

    const Qwen35CpuConfig & config() const { return cfg_; }
    std::vector<int> layer_devices() const { return std::vector<int>(layers_.size(), -1); }
    int num_threads() const { return pool_.num_threads(); }

private:
    CpuTensor bind_tensor(const std::string & name, bool required = true) const;
    void ensure_batch_buffers(int T);
    void ensure_kv_capacity(int needed_pos);
    void run_forward(const int32_t * tokens, const int32_t * positions, int T, float * out_logits);

    ShardedGguf                 sg_; // owns zero-copy mmap backing all CpuTensors
    Qwen35CpuConfig             cfg_;
    ThreadPool                  pool_;

    CpuTensor                   tok_embd_;
    CpuTensor                   output_norm_;
    CpuTensor                   output_;
    std::vector<Qwen35CpuLayer> layers_;

    // Per-layer GDN recurrent states (empty for full-attention layers)
    std::vector<std::vector<float>> gdn_conv_;  // [(d_conv-1) * conv_dim]
    std::vector<std::vector<float>> gdn_state_; // [n_v_heads * d_state * d_state]

    // Per-layer Full GQA KV caches (empty for GDN layers); grows lazily up to cfg_.max_ctx
    int kv_cap_ = 0;
    std::vector<std::vector<uint8_t>> k_cache_;
    std::vector<std::vector<uint8_t>> v_cache_;

    // Reusable activation scratch buffers
    int max_alloc_T_ = 0;
    std::vector<float>      cur_, x_norm_, sub_out_;
    std::vector<CpuActQ8_0> x_q8_;
    std::vector<CpuActQ8_K> x_q8k_;

    std::vector<float> qkv_raw_, qkv_conv_, z_buf_, alpha_buf_, beta_buf_, gdn_out_;
    std::vector<float> q_full_, q_buf_, gate_buf_, k_buf_, v_buf_, attn_out_;
    std::vector<float> ffn_g_, ffn_u_;
    std::vector<float> router_logits_, shexp_gate_, top_w_, shexp_out_;
    std::vector<int32_t> top_ids_;
    std::vector<float> logits_buf_;
};

} // namespace cpu
} // namespace rookery
