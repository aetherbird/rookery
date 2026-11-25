#pragma once

#include "cpu/cpu_kernels.h"
#include "cpu/cpu_pool.h"
#include "gguf/sharded.h"
#include "sampler/sampler.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rookery {

struct LayerHook;
struct MtpStats;

namespace cpu {

struct BailingMoe3CpuConfig {
    int   n_blocks             = 24;
    int   n_layers             = 24;
    int   n_dense_lead         = 1;
    int   n_nextn              = 0;
    int   n_embd               = 1536;
    int   n_ff_dense           = 4608;
    int   n_vocab              = 157184;
    int   n_head               = 16;
    int   kda_head_dim         = 128;
    float kda_gate_lower_bound = -5.0f;
    int   q_lora_rank          = 0;
    int   kv_lora_rank         = 512;
    int   rope_dim             = 64;
    int   qk_head_dim_mla      = 192;
    int   v_head_dim_mla       = 128;
    float rope_freq_base       = 6000000.0f;
    int   n_expert             = 128;
    int   n_expert_used        = 8;
    int   n_expert_groups      = 0;
    int   n_group_used         = 0;
    int   n_ff_exp             = 512;
    int   n_ff_shexp           = 512;
    bool  expert_weights_norm  = true;
    float expert_weights_scale = 2.5f;
    float rms_eps              = 1e-6f;
    int   max_ctx              = 131072;
    bool  kv_q8_0              = true;

    std::vector<int>   head_count_kv;
    std::vector<float> swiglu_clamp_exp;
    std::vector<float> swiglu_clamp_shexp;
};

struct BailingMoe3CpuLayer {
    bool is_kda = false;

    CpuTensor attn_norm, ffn_norm;

    // KDA tensors
    CpuTensor wq, wk, wv, wo;
    CpuTensor ssm_conv_q, ssm_conv_k, ssm_conv_v;
    CpuTensor ssm_f_a, ssm_dt_bias, ssm_a;
    CpuTensor ssm_beta;
    CpuTensor ssm_g_a, ssm_norm;

    // Gated RoPE-576 MLA tensors
    CpuTensor wq_a, attn_q_a_norm, wq_b;
    CpuTensor wkv_a_mqa, attn_kv_a_norm;
    CpuTensor wk_b, wv_b;
    CpuTensor attn_gate;

    // Dense FFN (il < n_dense_lead)
    CpuTensor ffn_gate, ffn_up, ffn_down;

    // MoE FFN (il >= n_dense_lead)
    CpuTensor ffn_gate_inp, ffn_exp_probs_b;
    CpuTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    CpuTensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

class BailingMoe3CpuModel {
public:
    BailingMoe3CpuModel() = default;
    ~BailingMoe3CpuModel() = default;

    BailingMoe3CpuModel(const BailingMoe3CpuModel &) = delete;
    BailingMoe3CpuModel & operator=(const BailingMoe3CpuModel &) = delete;

    void load(const std::string & gguf_path,
              int max_ctx = 131072,
              bool kv_q8_0 = true,
              int n_threads = 0);

    void reset_state();

    struct KdaCheckpoint {
        int pos = 0;
        std::vector<std::vector<float>> conv_q;
        std::vector<std::vector<float>> conv_k;
        std::vector<std::vector<float>> conv_v;
        std::vector<std::vector<float>> ssm;
    };
    void save_kda_checkpoint(KdaCheckpoint & out, int pos) const;
    void restore_kda_checkpoint(const KdaCheckpoint & cp);

    bool has_mtp() const { return false; }
    void set_layer_hook(LayerHook * /*hook*/) {}
    void warmup_decode_graphs() {}

    void forward(const std::vector<int32_t> & tokens,
                 const std::vector<int32_t> & positions,
                 std::vector<float> & out_logits);

    int32_t forward_decode_greedy(int32_t token, int32_t pos);
    void    forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits);

    const BailingMoe3CpuConfig & config() const { return cfg_; }
    std::vector<int> layer_devices() const { return std::vector<int>(layers_.size(), -1); }
    int num_threads() const { return pool_.num_threads(); }

private:
    CpuTensor bind_tensor(const std::string & name, bool required = true) const;
    void ensure_batch_buffers(int T);
    void run_forward(const int32_t * tokens, const int32_t * positions, int T, float * out_logits);

    ShardedGguf                    sg_; // owns mmap backing all zero-copy CpuTensors!
    BailingMoe3CpuConfig           cfg_;
    ThreadPool                     pool_;

    CpuTensor                      tok_embd_;
    CpuTensor                      output_norm_;
    CpuTensor                      output_;
    std::vector<BailingMoe3CpuLayer> layers_;

    // KDA recurrent states
    std::vector<std::vector<float>> kda_conv_q_;
    std::vector<std::vector<float>> kda_conv_k_;
    std::vector<std::vector<float>> kda_conv_v_;
    std::vector<std::vector<float>> kda_state_;

    // MLA RoPE-576 KV caches (only allocated where is_kda == false)
    std::vector<std::vector<CpuMlaKv576Q8Pos>> mla_kv_q8_;
    std::vector<std::vector<uint16_t>>         mla_kv_f16_;

    // Reusable activation scratch buffers
    int max_alloc_T_ = 0;
    std::vector<float>      cur_, x_norm_, sub_out_;
    std::vector<CpuActQ8_0> x_q8_;
    std::vector<CpuActQ8_K> x_q8k_;

    // KDA scratch
    std::vector<float> kda_q_, kda_k_, kda_v_, kda_qc_, kda_kc_, kda_vc_;
    std::vector<float> kda_g_, kda_beta_, kda_out_, kda_gate_;

    // MLA scratch
    std::vector<float> mla_qr_, mla_qall_, mla_qnope_, mla_qpe_, mla_qabs_, mla_q576_;
    std::vector<float> mla_kvall_, mla_olat_, mla_o_, mla_gate_;

    // FFN / MoE scratch
    std::vector<float>   ffn_g_, ffn_u_, router_;
    std::vector<int32_t> top_ids_;
    std::vector<float>   top_w_;

    // Output logits buffer for greedy decode
    std::vector<float> logits_buf_;
};

} // namespace cpu
} // namespace rookery
