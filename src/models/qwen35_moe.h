#pragma once

#include "core/device.h"
#include "gguf/loader.h"
#include "gguf/sharded.h"
#include "models/maxwell_moe.h" // for LayerHook, MtpStats
#include "sampler/sampler.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace rookery {

struct Qwen35Config {
    int   n_blocks             = 41;
    int   n_layers             = 40;
    int   n_nextn              = 1;
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
    int   ssm_d_inner          = 4096; // n_v_heads * head_v_dim
    int   full_attn_interval   = 4;

    // MoE / Dense FFN
    bool  is_dense             = false;
    int   n_ff                 = 0;
    int   n_expert             = 256;
    int   n_expert_used        = 8;
    int   n_ff_exp             = 512;
    int   n_ff_shexp           = 512;
    float expert_weights_scale = 1.0f;

    int   max_ctx              = 262144;
    bool  kv_q8_0              = true;

    std::vector<bool> is_recr; // [n_layers]: true = GDN, false = Full GQA
};

struct Qwen35LayerWeights {
    int  device  = 0;
    bool is_recr = false;

    DeviceTensor attn_norm, post_attn_norm;

    // GDN linear attention
    DeviceTensor wqkv, wqkv_gate;
    DeviceTensor ssm_conv1d, ssm_dt, ssm_a, ssm_alpha, ssm_beta, ssm_norm, ssm_out;

    // Full GQA attention
    DeviceTensor wq, wk, wv, wo;
    DeviceTensor attn_q_norm, attn_k_norm;

    // Dense SwiGLU FFN (qwen35)
    DeviceTensor ffn_gate, ffn_up, ffn_down;

    // MoE + Gated Shared Expert (qwen35moe)
    DeviceTensor ffn_gate_inp;
    DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    DeviceTensor ffn_gate_inp_shexp;
    DeviceTensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

struct Qwen35MtpWeights {
    int device = 0;
    DeviceTensor eh_proj, enorm, hnorm, shared_head_norm;
    DeviceTensor attn_norm, post_attn_norm;
    DeviceTensor wq, wk, wv, wo;
    DeviceTensor attn_q_norm, attn_k_norm;
    DeviceTensor ffn_gate, ffn_up, ffn_down;
    DeviceTensor ffn_gate_inp;
    DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    DeviceTensor ffn_gate_inp_shexp;
    DeviceTensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

class Qwen35MoeModel {
public:
    Qwen35MoeModel() = default;
    ~Qwen35MoeModel();

    Qwen35MoeModel(const Qwen35MoeModel &) = delete;
    Qwen35MoeModel & operator=(const Qwen35MoeModel &) = delete;

    void load(const std::string & gguf_path,
              const std::vector<int> & gpu_ids,
              const std::vector<float> & split_weights,
              int max_ctx = 65536,
              bool kv_q8_0 = true,
              bool load_mtp = true);

    void reset_state();

    struct GdnCheckpoint {
        int pos = 0;
        std::vector<std::vector<float>> conv;
        std::vector<std::vector<float>> ssm;
    };
    void save_gdn_checkpoint(GdnCheckpoint & out, int pos) const;
    void restore_gdn_checkpoint(const GdnCheckpoint & cp);

    bool has_mtp() const { return has_mtp_; }

    void set_layer_hook(LayerHook * hook) { hook_ = hook; invalidate_graphs(); }
    void set_use_cuda_graphs(bool enable) { use_cuda_graphs_ = enable; invalidate_graphs(); }
    void warmup_decode_graphs();

    void forward(const std::vector<int32_t> & tokens,
                 const std::vector<int32_t> & positions,
                 std::vector<float> & out_logits,
                 bool compute_logits = true);

    int32_t forward_decode_greedy(int32_t token, int32_t pos);
    void    forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits);

    // MTP speculative decoding API (matching Glm5NextModel!)
    int32_t mtp_prefill_chunk(const std::vector<int32_t> & next_tokens,
                              const std::vector<int32_t> & positions);
    void    mtp_prefill_chunk_logits(const std::vector<int32_t> & next_tokens,
                                     const std::vector<int32_t> & positions,
                                     std::vector<float> & out_logits);

    void decode_mtp_greedy_step(int32_t & cur_tok,
                                int32_t & draft_tok,
                                int32_t & cur_pos,
                                std::vector<int32_t> & out_tokens,
                                MtpStats & stats);

    void decode_mtp_sample_step(int32_t & cur_tok,
                                int32_t & draft_tok,
                                Sampler::SparseDist & draft_q,
                                int32_t & cur_pos,
                                Sampler & sampler,
                                std::vector<int32_t> & out_tokens,
                                MtpStats & stats);

    void rollback_gdn_states();

    const Qwen35Config & config() const { return cfg_; }
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
        cudaGraphExec_t decode_graph_t2 = nullptr;

        float *   d_cur        = nullptr; // [T, H]
        float *   d_x_norm     = nullptr; // [T, H]
        float *   d_sub_out    = nullptr; // [T, H]

        // GDN scratch
        float *   d_qkv_raw    = nullptr; // [T, conv_dim]
        float *   d_qkv_conv   = nullptr; // [T, conv_dim]
        float *   d_z          = nullptr; // [T, d_inner]
        float *   d_alpha      = nullptr; // [T, n_v_heads]
        float *   d_beta       = nullptr; // [T, n_v_heads]
        float *   d_gdn_out    = nullptr; // [T, d_inner]

        // Full GQA scratch
        float *   d_q_full     = nullptr; // [T, n_head * 512]
        float *   d_q          = nullptr; // [T, n_head * 256]
        float *   d_gate       = nullptr; // [T, n_head * 256]
        float *   d_k          = nullptr; // [T, n_kv * 256]
        float *   d_v          = nullptr; // [T, n_kv * 256]
        float *   d_attn_out   = nullptr; // [T, n_head * 256]

        // Dense FFN scratch (qwen35)
        float *   d_ff_gate    = nullptr; // [T, n_ff]
        float *   d_ff_up      = nullptr; // [T, n_ff]

        // MoE scratch (qwen35moe)
        float *   d_router     = nullptr; // [T, n_expert]
        float *   d_shexp_gate = nullptr; // [T]
        int32_t * d_top_ids    = nullptr; // [T, k_used]
        float *   d_top_w      = nullptr; // [T, k_used]

        int32_t * d_tokens     = nullptr;
        int32_t * d_pos        = nullptr;
        float *   d_logits     = nullptr; // [2, n_vocab] on last GPU
        int32_t * d_argmax     = nullptr; // [2] on last GPU

        // MTP buffers (on last GPU only)
        float *   d_h_nextn    = nullptr; // [T, H]
        float *   d_mtp_e      = nullptr; // [T, H]
        float *   d_mtp_eh     = nullptr; // [T, 2*H]
        float *   d_mtp_cur    = nullptr; // [T, H]
        int32_t * d_mtp_pos    = nullptr; // [T]
        int32_t * d_mtp_argmax = nullptr; // [1]

        void *    d_scratch     = nullptr;
        size_t    scratch_bytes = 0;
    };

    void run_gpu_segment(size_t gi, int T, bool run_argmax, bool save_gdn_snap,
                         int kv_len_max = 0, bool compute_logits = true);
    void run_mtp_layer(int T, bool run_argmax, int kv_len_max = 0);
    void embed_tokens_to_last_dev(const int32_t * h_tokens, int T);
    void capture_decode_graphs();
    void invalidate_graphs() noexcept;

    Qwen35Config                    cfg_;
    WeightPool                      pool_;
    DeviceTensor                    tok_embd_;
    DeviceTensor                    output_norm_, output_;
    std::vector<Qwen35LayerWeights> layers_;
    bool                            has_mtp_ = false;
    Qwen35MtpWeights                mtp_;
    void *                          mtp_k_cache_  = nullptr;
    void *                          mtp_v_cache_  = nullptr;
    cudaGraphExec_t                 mtp_graph_t1_ = nullptr;
    cudaGraphExec_t                 mtp_graph_t2_ = nullptr;

    std::vector<float *> gdn_conv_;
    std::vector<float *> gdn_state_;
    std::vector<float *> gdn_conv_snap_;
    std::vector<float *> gdn_state_snap_;
    std::vector<void *>  k_cache_;
    std::vector<void *>  v_cache_;

    std::vector<DevRuntime> runtimes_;
    int                     max_batch_       = 512;
    bool                    use_cuda_graphs_ = true;
    bool                    graphs_ready_    = false;
    LayerHook *             hook_            = nullptr;
};

} // namespace rookery
