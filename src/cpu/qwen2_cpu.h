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

struct Qwen2CpuConfig {
    int   n_layers       = 28;
    int   n_embd         = 1536;
    int   n_ff           = 8960;
    int   n_vocab        = 151936;
    int   n_head         = 12;
    int   n_head_kv      = 2;
    int   head_dim       = 128;
    float rope_freq_base = 1000000.0f;
    float rms_eps        = 1e-6f;
    int         max_ctx        = 32768;
    KvCacheType kv_type        = KvCacheType::Q8_0;
};

struct Qwen2CpuLayer {
    CpuTensor attn_norm, ffn_norm;
    CpuTensor wq, wk, wv, wo;
    CpuTensor bq, bk, bv; // optional QKV biases (present on Qwen2/Qwen2.5)
    CpuTensor ffn_gate, ffn_up, ffn_down;
};

class Qwen2CpuModel {
public:
    Qwen2CpuModel() = default;
    ~Qwen2CpuModel() = default;

    Qwen2CpuModel(const Qwen2CpuModel &) = delete;
    Qwen2CpuModel & operator=(const Qwen2CpuModel &) = delete;

    void load(const std::string & gguf_path,
              int max_ctx = 32768,
              KvCacheType kv_type = KvCacheType::Q8_0,
              int n_threads = 0);

    void reset_state() {} // Pure KV-cache transformer; no recurrent state to clear

    bool has_mtp() const { return false; }
    void set_layer_hook(LayerHook * /*hook*/) {}
    void warmup_decode_graphs() {}

    void forward(const std::vector<int32_t> & tokens,
                 const std::vector<int32_t> & positions,
                 std::vector<float> & out_logits);

    int32_t forward_decode_greedy(int32_t token, int32_t pos);
    void    forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits);

    const Qwen2CpuConfig & config() const { return cfg_; }
    std::vector<int> layer_devices() const { return std::vector<int>(layers_.size(), -1); }
    int num_threads() const { return pool_.num_threads(); }

private:
    CpuTensor bind_tensor(const std::string & name, bool required = true) const;
    void ensure_batch_buffers(int T);
    void run_forward(const int32_t * tokens, const int32_t * positions, int T, float * out_logits);

    ShardedGguf                sg_; // owns zero-copy mmap backing all CpuTensors
    Qwen2CpuConfig             cfg_;
    ThreadPool                 pool_;

    CpuTensor                  tok_embd_;
    CpuTensor                  output_norm_;
    CpuTensor                  output_;
    std::vector<Qwen2CpuLayer> layers_;

    // Per-layer KV cache: [n_head_kv][max_ctx]
    std::vector<std::vector<CpuGqaKv128Q8Pos>> k_cache_q8_;
    std::vector<std::vector<CpuGqaKv128Q8Pos>> v_cache_q8_;
    std::vector<std::vector<CpuGqaKv128Q4Pos>> k_cache_q4_;
    std::vector<std::vector<CpuGqaKv128Q4Pos>> v_cache_q4_;
    std::vector<std::vector<uint16_t>>         k_cache_f16_;
    std::vector<std::vector<uint16_t>>         v_cache_f16_;

    // Reusable activation scratch buffers
    int max_alloc_T_ = 0;
    std::vector<float>      cur_, x_norm_, sub_out_;
    std::vector<CpuActQ8_0> x_q8_;
    std::vector<CpuActQ8_K> x_q8k_;

    std::vector<float> q_buf_, k_buf_, v_buf_, attn_out_;
    std::vector<float> ffn_g_, ffn_u_;
    std::vector<float> logits_buf_;
};

} // namespace cpu
} // namespace rookery
