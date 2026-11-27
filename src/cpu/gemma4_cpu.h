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

struct Gemma4CpuConfig {
    int   n_layers              = 42;
    int   n_embd                = 2560;
    int   n_vocab               = 262144;
    int   n_head                = 8;
    int   head_dim_full         = 512;
    int   head_dim_swa          = 256;
    int   rope_dim_full         = 512;
    int   rope_dim_swa          = 256;
    float rope_freq_base_full   = 1000000.0f;
    float rope_freq_base_swa    = 10000.0f;
    float rms_eps               = 1e-6f;
    int   sliding_window        = 512;
    int   shared_kv_layers      = 0;
    int   n_layer_kv_from_start = 42;
    int   n_embd_per_layer      = 0;
    int   n_expert              = 0;
    int   n_expert_used         = 0;
    int   n_ff_exp              = 0;
    float final_logit_softcap   = 30.0f;
    int         max_ctx         = 8192;
    int         swa_slots       = 2048;
    KvCacheType kv_type         = KvCacheType::Q8_0;
    bool        kv_q8_0         = true;
    bool        tied_embeddings = true;

    std::vector<int>  n_head_kv;  // [n_layers]
    std::vector<int>  n_ff;       // [n_layers]
    std::vector<bool> is_swa;     // [n_layers]
    std::vector<int>  kv_source;  // [n_layers]
};

struct Gemma4CpuLayer {
    bool is_swa    = true;
    bool has_kv    = true;
    bool is_moe    = false;
    int  kv_source = 0;
    int  n_head_kv = 2;
    int  n_ff      = 10240;

    // Attention
    CpuTensor attn_norm;
    CpuTensor wqkv;
    CpuTensor wq, wk, wv, wo;
    CpuTensor attn_q_norm, attn_k_norm, attn_post_norm;
    CpuTensor rope_freqs;

    // Dense FFN (or parallel dense MLP branch in MoE layers)
    CpuTensor ffn_norm;
    CpuTensor ffn_gate, ffn_up, ffn_down;
    CpuTensor ffn_post_norm;

    // MoE FFN (Gemma 4 26B-A4B)
    CpuTensor ffn_gate_inp, ffn_gate_inp_scale;
    CpuTensor ffn_pre_norm_2, ffn_post_norm_1, ffn_post_norm_2;
    CpuTensor ffn_gate_up_exps;
    CpuTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    CpuTensor ffn_down_exps_scale;

    // Per-Layer Embedding (PLE) gating block
    CpuTensor per_layer_inp_gate;
    CpuTensor per_layer_proj;
    CpuTensor per_layer_post_norm;

    // Optional layer scalar
    float layer_output_scale = 1.0f;
    bool  has_output_scale   = false;
};

class Gemma4CpuModel {
public:
    Gemma4CpuModel() = default;
    ~Gemma4CpuModel() = default;

    Gemma4CpuModel(const Gemma4CpuModel &) = delete;
    Gemma4CpuModel & operator=(const Gemma4CpuModel &) = delete;

    void load(const std::string & gguf_path,
              int max_ctx = 8192,
              KvCacheType kv_type = KvCacheType::Q8_0,
              int n_threads = 0);

    void load(const std::string & gguf_path,
              int max_ctx,
              bool kv_q8_0,
              int n_threads = 0) {
        load(gguf_path, max_ctx, kv_q8_0 ? KvCacheType::Q8_0 : KvCacheType::F16, n_threads);
    }

    void reset_state();

    bool has_mtp() const { return false; }
    void set_layer_hook(LayerHook * /*hook*/) {}
    void warmup_decode_graphs() {}

    // Safe prefix rollback limit due to SWA ring buffer capacity.
    int swa_safe_rollback_limit() const {
        if (cfg_.swa_slots >= cfg_.max_ctx) return cfg_.max_ctx;
        return std::max(0, cfg_.swa_slots - cfg_.sliding_window);
    }

    void forward(const std::vector<int32_t> & tokens,
                 const std::vector<int32_t> & positions,
                 std::vector<float> & out_logits,
                 bool compute_logits = true);

    int32_t forward_decode_greedy(int32_t token, int32_t pos);
    void    forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits);

    const Gemma4CpuConfig & config() const { return cfg_; }
    std::vector<int> layer_devices() const { return std::vector<int>(layers_.size(), -1); }
    int num_threads() const { return pool_.num_threads(); }

private:
    CpuTensor bind_tensor(const std::string & name, bool required = true) const;
    void ensure_batch_buffers(int T);
    void ensure_full_kv_capacity(int needed_pos);
    void run_forward(const int32_t * tokens, const int32_t * positions, int T, float * out_logits, bool compute_logits, bool apply_softcap = true);

    ShardedGguf                 sg_; // owns mmap backing all zero-copy CpuTensors!
    Gemma4CpuConfig             cfg_;
    ThreadPool                  pool_;

    CpuTensor                   tok_embd_;
    CpuTensor                   per_layer_tok_embd_;
    CpuTensor                   per_layer_model_proj_;
    CpuTensor                   per_layer_proj_norm_;
    CpuTensor                   output_norm_;
    CpuTensor                   output_;
    std::vector<Gemma4CpuLayer> layers_;

    // KV caches (only allocated for layers [0 .. n_layer_kv_from_start))
    int                              full_kv_cap_ = 0;
    std::vector<std::vector<uint8_t>> k_cache_;
    std::vector<std::vector<uint8_t>> v_cache_;

    // Reusable activation scratch buffers
    int max_alloc_T_ = 0;
    std::vector<float> cur_, x_norm_, sub_out_, attn_res_;
    std::vector<float> q_buf_, k_buf_, v_buf_, qkv_buf_, attn_out_;
    std::vector<float> ffn_g_, ffn_u_, ffn_mlp_, ffn_moe_, router_in_, router_logits_;
    std::vector<CpuActQ8_0> ffn_q8_;
    std::vector<int32_t> top_ids_;
    std::vector<float>   top_w_;

    // PLE scratch
    std::vector<float> ple_raw_, ple_proj_, ple_buf_, ple_gate_;

    // Output logits buffer for greedy decode
    std::vector<float> logits_buf_;
};

} // namespace cpu
} // namespace rookery
