#pragma once

#include "core/device.h"
#include "gguf/loader.h"
#include "gguf/sharded.h"
#include "models/maxwell_moe.h" // for LayerHook

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rookery {

struct Raon1MiniConfig {
    int   n_layers             = 25;
    int   n_dense_lead         = 1;
    int   n_embd               = 2560;
    int   n_ff_dense           = 12032;
    int   n_vocab              = 259840;
    int   n_head               = 40;
    int   rda_head_dim         = 128;
    int   rda_decay_rank       = 128;
    int   rda_conv_kernel      = 4;
    float rda_gate_lower_bound = -5.0f;
    int   q_lora_rank          = 576;
    int   kv_lora_rank         = 512;
    int   qk_pe_dim            = 64;    // NoPE shared PE channels (no rotation!)
    int   qk_head_dim_mla      = 192;   // 128 nope + 64 pe
    int   v_head_dim_mla       = 128;
    int   n_expert             = 128;
    int   n_expert_used        = 6;
    int   n_ff_exp             = 1088;
    int   n_ff_shexp           = 1088;
    bool  expert_weights_norm  = true;
    float expert_weights_scale = 1.0f;
    float situ_beta            = 4.0f;
    float situ_linear_beta     = 25.0f;
    int   hc_mult              = 4;
    float hc_eps               = 1e-5f;
    float hc_post_factor       = 2.0f;
    float rms_eps              = 1e-5f;
    float mla_latent_rms_eps   = 1e-6f;
    int   max_ctx              = 262144;
    bool  kv_q8_0              = true;

    std::vector<int> head_count_kv; // 0 = RDA, >0 = NoPE Gated MLA
    std::string default_reasoning_effort = "high";
    std::unordered_map<std::string, float> reasoning_effort_alpha;
};

struct Raon1MiniLayerWeights {
    int  device = 0;
    bool is_rda = false;

    DeviceTensor attn_norm, ffn_norm;

    // 4-stream RaonMHC weights (attention site and FFN site)
    DeviceTensor hc_attn_fn, hc_attn_scale, hc_attn_base;
    DeviceTensor hc_ffn_fn,  hc_ffn_scale,  hc_ffn_base;

    // RDA linear-attention tensors (when is_rda == true)
    DeviceTensor wq, wk, wv, wo;
    DeviceTensor ssm_conv_q, ssm_conv_k, ssm_conv_v;
    DeviceTensor ssm_f_a, ssm_f_b, ssm_dt_bias, ssm_a;
    DeviceTensor ssm_beta, ssm_rda, ssm_g, ssm_norm;

    // NoPE Gated MLA tensors (when is_rda == false)
    DeviceTensor wq_a, attn_q_a_norm, wq_b;
    DeviceTensor wkv_a_mqa, attn_kv_a_norm;
    DeviceTensor wk_b, wv_b;   // uploaded as FP16 for cublasHgemm / Tensor Core speed
    DeviceTensor attn_gate;    // [n_embd, n_head * v_head_dim] per-element sigmoid gate

    // Dense FFN (il < n_dense_lead)
    DeviceTensor ffn_gate, ffn_up, ffn_down;

    // MoE FFN (il >= n_dense_lead)
    DeviceTensor ffn_gate_inp, ffn_exp_probs_b;
    DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    DeviceTensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;

    // Optional reasoning-effort steering vector [n_embd] F32 (layers 12,14,16,18,20)
    DeviceTensor ffn_effort_steer;
};

class Raon1MiniModel {
public:
    Raon1MiniModel() = default;
    ~Raon1MiniModel();

    Raon1MiniModel(const Raon1MiniModel &) = delete;
    Raon1MiniModel & operator=(const Raon1MiniModel &) = delete;

    void load(const std::string & gguf_path,
              const std::vector<int> & gpu_ids,
              const std::vector<float> & split_weights,
              int max_ctx = 262144,
              bool kv_q8_0 = true);

    void reset_state();

    struct RdaCheckpoint {
        int pos = 0;
        std::vector<std::vector<float>> conv_q;
        std::vector<std::vector<float>> conv_k;
        std::vector<std::vector<float>> conv_v;
        std::vector<std::vector<float>> ssm;
    };
    void save_rda_checkpoint(RdaCheckpoint & out, int pos) const;
    void restore_rda_checkpoint(const RdaCheckpoint & cp);

    void set_reasoning_effort(const std::string & effort);
    const std::string & reasoning_effort() const { return cur_effort_; }
    float reasoning_effort_alpha() const { return effort_alpha_; }

    void set_layer_hook(LayerHook * hook) { hook_ = hook; invalidate_graphs(); }
    void set_use_cuda_graphs(bool enable) { use_cuda_graphs_ = enable; invalidate_graphs(); }
    void warmup_decode_graphs();

    void forward(const std::vector<int32_t> & tokens,
                 const std::vector<int32_t> & positions,
                 std::vector<float> & out_logits,
                 bool compute_logits = true);

    int32_t forward_decode_greedy(int32_t token, int32_t pos);
    void    forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits);

    const Raon1MiniConfig & config() const { return cfg_; }
    std::vector<int> layer_devices() const {
        std::vector<int> out;
        out.reserve(layers_.size());
        for (const auto & L : layers_) out.push_back(L.device);
        return out;
    }

private:
    struct DevRuntime {
        int             device          = 0;
        int             layer_start     = 0;
        int             layer_end       = 0;
        cudaStream_t    stream          = nullptr;
        cudaEvent_t     done_event      = nullptr;
        cublasHandle_t  cublas          = nullptr;
        cudaGraphExec_t decode_graph_t1 = nullptr;

        // 4-stream RaonMHC state (ping-pong) + branch buffers
        float *   d_stream    = nullptr; // [T, 4, n_embd]
        float *   d_stream_tmp= nullptr; // [T, 4, n_embd]
        float *   d_hc_mixes  = nullptr; // [T, 17]
        float *   d_hc_post   = nullptr; // [T, 4]
        float *   d_hc_comb   = nullptr; // [T, 16]
        float *   d_x_norm    = nullptr; // [T, n_embd]
        float *   d_sub_out   = nullptr; // [T, n_embd]

        // RDA scratch (d_inner = 5120)
        float *   d_rda_q     = nullptr;
        float *   d_rda_k     = nullptr;
        float *   d_rda_v     = nullptr;
        float *   d_rda_qc    = nullptr;
        float *   d_rda_kc    = nullptr;
        float *   d_rda_vc    = nullptr;
        float *   d_rda_fa    = nullptr; // [T, decay_rank]
        float *   d_rda_decay = nullptr; // [T, d_inner]
        float *   d_rda_beta  = nullptr; // [T, n_head]
        float *   d_rda_delta = nullptr; // [T, n_head]
        float *   d_rda_out   = nullptr; // [T, d_inner]
        float *   d_rda_gate  = nullptr; // [T, d_inner]

        // NoPE Gated MLA scratch
        float *   d_mla_qr    = nullptr; // [T, q_lora_rank]
        float *   d_mla_qall  = nullptr; // [T, n_head * 192]
        float *   d_mla_qnope = nullptr; // [T, n_head * 128]
        float *   d_mla_qpe   = nullptr; // [T, n_head * 64]
        float *   d_mla_qabs  = nullptr; // [T, n_head * 512]
        float *   d_mla_q576  = nullptr; // [T, n_head * 576]
        float *   d_mla_kvall = nullptr; // [T, 576]
        float *   d_mla_olat  = nullptr; // [T, n_head * 512]
        float *   d_mla_o     = nullptr; // [T, n_head * 128]
        float *   d_mla_gate  = nullptr; // [T, n_head * 128]

        // FFN / MoE scratch
        float *   d_ffn_g     = nullptr; // [T, n_ff_dense]
        float *   d_ffn_u     = nullptr; // [T, n_ff_dense]
        float *   d_router    = nullptr; // [T, n_expert]
        int32_t * d_top_ids   = nullptr; // [T, n_expert_used]
        float *   d_top_w     = nullptr; // [T, n_expert_used]

        // IO buffers
        int32_t * d_tokens    = nullptr; // [T]
        int32_t * d_pos       = nullptr; // [T]
        float *   d_logits    = nullptr; // [n_vocab] (last GPU)
        int32_t * d_argmax    = nullptr; // [1]       (last GPU)

        void *    d_scratch     = nullptr;
        size_t    scratch_bytes = 0;
    };

    void run_gpu_segment(size_t gi, int T, bool run_argmax, bool compute_logits, int kv_len_max = 0);
    void run_pipeline(int T, bool run_argmax, bool compute_logits, int kv_len_max, bool prefill_sync = false);
    void capture_decode_graphs();
    void invalidate_graphs() noexcept;

    Raon1MiniConfig                    cfg_;
    WeightPool                         pool_;
    DeviceTensor                       tok_embd_;
    DeviceTensor                       output_norm_;
    DeviceTensor                       output_;
    std::vector<Raon1MiniLayerWeights> layers_;
    std::vector<DevRuntime>            runtimes_;

    // RDA recurrent states per layer (only allocated where is_rda == true)
    std::vector<float *> rda_conv_q_;
    std::vector<float *> rda_conv_k_;
    std::vector<float *> rda_conv_v_;
    std::vector<float *> rda_state_;

    // NoPE MLA 576-dim KV cache per layer (only allocated where is_rda == false)
    std::vector<void *>  mla_kv_cache_;

    int         max_batch_       = 512;
    std::string cur_effort_      = "high";
    float       effort_alpha_    = 0.05f;
    LayerHook * hook_            = nullptr;
    bool        use_cuda_graphs_ = true;
    bool        graphs_ready_    = false;
};

} // namespace rookery
