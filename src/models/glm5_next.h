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

struct Glm5Config {
    int   n_blocks             = 46;
    int   n_layers             = 45;
    int   n_dense_lead         = 3;
    int   n_nextn              = 1;
    int   n_embd               = 4096;
    int   n_ff_dense           = 12288;
    int   n_vocab              = 154880;
    int   n_head               = 64;
    int   kda_head_dim         = 128;
    float kda_gate_lower_bound = -5.0f;
    int   q_lora_rank          = 1536;
    int   kv_lora_rank         = 512;
    int   qk_head_dim_mla      = 256;
    int   v_head_dim_mla       = 256;
    int   n_expert             = 288;
    int   n_expert_used        = 8;
    int   n_ff_exp             = 2048;
    int   n_ff_shexp           = 2048;
    bool  expert_weights_norm  = true;
    float expert_weights_scale = 2.5f;
    int   hc_mult              = 4;
    int   hc_sinkhorn_iters    = 20;
    float hc_eps               = 1e-6f;
    float rms_eps              = 1e-5f;
    int   max_ctx              = 8192;
    bool  kv_q8_0              = true;

    std::vector<int>   head_count_kv;       // [n_blocks]: 0 = KDA, 1 = MLA
    std::vector<float> swiglu_clamp_exp;    // [n_blocks]
    std::vector<float> swiglu_clamp_shexp;  // [n_blocks]
};

struct Glm5LayerWeights {
    int  device = 0;
    bool is_kda = false;

    DeviceTensor attn_norm, ffn_norm;
    DeviceTensor hc_attn_fn, hc_attn_base, hc_attn_scale;
    DeviceTensor hc_ffn_fn,  hc_ffn_base,  hc_ffn_scale;

    // KDA linear-attention tensors (when is_kda == true)
    DeviceTensor wq, wk, wv, wo;
    DeviceTensor ssm_conv_q, ssm_conv_k, ssm_conv_v;
    DeviceTensor ssm_f_a, ssm_f_b, ssm_dt_bias, ssm_a;
    DeviceTensor ssm_beta;
    DeviceTensor ssm_g_a, ssm_g_b, ssm_norm;

    // MLA attention tensors (when is_kda == false)
    DeviceTensor wq_a, attn_q_a_norm, wq_b;
    DeviceTensor wkv_a_mqa, attn_kv_a_norm, wk_b, wv_b;

    // Dense FFN (blk.0..2)
    DeviceTensor ffn_gate, ffn_up, ffn_down;

    // MoE FFN (blk.3..45)
    DeviceTensor ffn_gate_inp, ffn_exp_probs_b;
    DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    DeviceTensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

struct Glm5MtpWeights {
    int device = 0;
    DeviceTensor eh_proj, enorm, hnorm, shared_head_norm;
    DeviceTensor attn_norm, ffn_norm;
    DeviceTensor wq_a, attn_q_a_norm, wq_b;
    DeviceTensor wkv_a_mqa, attn_kv_a_norm, wk_b, wv_b, wo;
    DeviceTensor ffn_gate_inp, ffn_exp_probs_b;
    DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    DeviceTensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

class Glm5NextModel {
public:
    Glm5NextModel() = default;
    ~Glm5NextModel();

    Glm5NextModel(const Glm5NextModel &) = delete;
    Glm5NextModel & operator=(const Glm5NextModel &) = delete;

    void load(const std::string & gguf_path,
              const std::vector<int> & gpu_ids,
              const std::vector<float> & split_weights,
              int max_ctx = 8192,
              bool kv_q8_0 = true,
              bool load_mtp = true);

    void reset_state(); // Zeroes KDA conv & recurrent states before a new conversation!

    struct KdaCheckpoint {
        int pos = 0; // number of tokens processed when this KDA state was captured
        std::vector<std::vector<float>> conv_q;
        std::vector<std::vector<float>> conv_k;
        std::vector<std::vector<float>> conv_v;
        std::vector<std::vector<float>> ssm;
    };
    void save_kda_checkpoint(KdaCheckpoint & out, int pos) const;
    void restore_kda_checkpoint(const KdaCheckpoint & cp);

    bool has_mtp() const { return has_mtp_; }

    void set_layer_hook(LayerHook * hook) { hook_ = hook; invalidate_graphs(); }
    void set_use_cuda_graphs(bool enable) { use_cuda_graphs_ = enable; invalidate_graphs(); }
    void warmup_decode_graphs();

    void forward(const std::vector<int32_t> & tokens,
                 const std::vector<int32_t> & positions,
                 std::vector<float> & out_logits);

    int32_t forward_decode_greedy(int32_t token, int32_t pos);
    void    forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits);

    // MTP speculative decoding API (matching MaxwellMoeModel for unified rookery-server!)
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

    const Glm5Config & config() const { return cfg_; }
    void rollback_kda_states();
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

        float *   d_inpL     = nullptr; // [T, 4, 4096]
        float *   d_inpL_alt = nullptr; // [T, 4, 4096]
        float *   d_mixes    = nullptr; // [T, 24]
        float *   d_post     = nullptr; // [T, 4]
        float *   d_comb     = nullptr; // [T, 16]
        float *   d_x_norm   = nullptr; // [T, 4096]
        float *   d_sub_out  = nullptr; // [T, 4096]

        // KDA scratch (d_inner = 8192)
        float *   d_kda_q    = nullptr;
        float *   d_kda_k    = nullptr;
        float *   d_kda_v    = nullptr;
        float *   d_kda_qc   = nullptr;
        float *   d_kda_kc   = nullptr;
        float *   d_kda_vc   = nullptr;
        float *   d_kda_fa   = nullptr;
        float *   d_kda_g    = nullptr;
        float *   d_kda_beta = nullptr;
        float *   d_kda_out  = nullptr;
        float *   d_kda_gate = nullptr;

        // MLA scratch
        float *   d_mla_qr   = nullptr;
        float *   d_mla_q    = nullptr;
        float *   d_mla_qabs = nullptr;
        float *   d_mla_kv   = nullptr;
        float *   d_mla_olat = nullptr;
        float *   d_mla_o    = nullptr;

        // FFN / MoE scratch
        float *   d_ffn_g    = nullptr;
        float *   d_ffn_u    = nullptr;
        float *   d_router   = nullptr;
        int32_t * d_top_ids  = nullptr;
        float *   d_top_w    = nullptr;

        int32_t * d_tokens   = nullptr;
        int32_t * d_pos      = nullptr;
        float *   d_logits   = nullptr; // [2, n_vocab] on last GPU
        int32_t * d_argmax   = nullptr; // [2] on last GPU

        // MTP buffers (allocated on last GPU only)
        float *   d_h_nextn    = nullptr; // [Tmax, 4096]
        float *   d_mtp_e      = nullptr; // [Tmax, 4096]
        float *   d_mtp_eh     = nullptr; // [Tmax, 8192]
        float *   d_mtp_cur    = nullptr; // [Tmax, 4096]
        int32_t * d_mtp_pos    = nullptr; // [Tmax]
        int32_t * d_mtp_argmax = nullptr; // [1]

        void *    d_scratch     = nullptr;
        size_t    scratch_bytes = 0;
    };

    void run_gpu_segment(size_t gi, int T, bool run_argmax, bool save_kda_snap, int kv_len_max = 0);
    void run_mtp_layer(int T, bool run_argmax, int kv_len_max = 0);
    void embed_tokens_to_last_dev(const int32_t * h_tokens, int T);
    void capture_decode_graphs();
    void invalidate_graphs() noexcept;

    Glm5Config                    cfg_;
    WeightPool                    pool_;
    DeviceTensor                  tok_embd_;
    DeviceTensor                  output_norm_, output_;
    std::vector<Glm5LayerWeights> layers_;
    bool                          has_mtp_ = false;
    Glm5MtpWeights                mtp_;
    void *                        mtp_kv_cache_   = nullptr;
    cudaGraphExec_t               mtp_graph_t1_   = nullptr;
    cudaGraphExec_t               mtp_graph_t2_   = nullptr;

    // Per-layer persistent state (on each layer's owning GPU)
    std::vector<float *> kda_conv_q_;
    std::vector<float *> kda_conv_k_;
    std::vector<float *> kda_conv_v_;
    std::vector<float *> kda_state_;
    std::vector<float *> kda_conv_q_snap_;
    std::vector<float *> kda_conv_k_snap_;
    std::vector<float *> kda_conv_v_snap_;
    std::vector<float *> kda_state_snap_;
    std::vector<void *>  mla_kv_cache_;

    std::vector<DevRuntime> runtimes_;
    int                     max_batch_       = 256;
    bool                    use_cuda_graphs_ = true;
    bool                    graphs_ready_    = false;
    LayerHook *             hook_            = nullptr;
};

} // namespace rookery
