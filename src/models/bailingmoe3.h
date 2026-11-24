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

struct BailingMoe3Config {
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
    int   q_lora_rank          = 0;     // 0 => direct attn_q (Ling-3.0-Flash / Ling-3.2-Air); >0 => LoRA (Ling-3.0-tiny)
    int   kv_lora_rank         = 512;
    int   rope_dim             = 64;
    int   qk_head_dim_mla      = 192;   // qk_nope (128) + rope_dim (64)
    int   v_head_dim_mla       = 128;
    float rope_freq_base       = 6000000.0f;
    int   n_expert             = 128;
    int   n_expert_used        = 8;
    int   n_expert_groups      = 0;     // >1 => DeepSeek-V3/BailingMoE3 grouped top-k routing (e.g. 8 groups on Ling-3.2-Air)
    int   n_group_used         = 0;     // e.g. 4 groups on Ling-3.2-Air
    int   n_ff_exp             = 512;
    int   n_ff_shexp           = 512;
    bool  expert_weights_norm  = true;
    float expert_weights_scale = 2.5f;
    float rms_eps              = 1e-6f;
    int   max_ctx              = 131072;
    bool  kv_q8_0              = true;

    std::vector<int>   head_count_kv;       // [n_blocks]: 0 = KDA, >0 = Gated MLA
    std::vector<float> swiglu_clamp_exp;    // [n_blocks]
    std::vector<float> swiglu_clamp_shexp;  // [n_blocks]
};

struct BailingMoe3LayerWeights {
    int  device = 0;
    bool is_kda = false;

    DeviceTensor attn_norm, ffn_norm;

    // KDA linear-attention tensors (when is_kda == true)
    DeviceTensor wq, wk, wv, wo;
    DeviceTensor ssm_conv_q, ssm_conv_k, ssm_conv_v;
    DeviceTensor ssm_f_a, ssm_dt_bias, ssm_a;
    DeviceTensor ssm_beta;
    DeviceTensor ssm_g_a, ssm_norm;

    // Gated RoPE-576 MLA tensors (when is_kda == false)
    DeviceTensor wq_a, attn_q_a_norm, wq_b; // used when q_lora_rank > 0 (else wq is used!)
    DeviceTensor wkv_a_mqa, attn_kv_a_norm;
    DeviceTensor wk_b, wv_b;                // uploaded as FP16 for any-quant Tensor Core speed
    DeviceTensor attn_gate;                 // [n_embd, n_head] per-head sigmoid gate

    // Dense FFN (il < n_dense_lead)
    DeviceTensor ffn_gate, ffn_up, ffn_down;

    // MoE FFN (il >= n_dense_lead)
    DeviceTensor ffn_gate_inp, ffn_exp_probs_b;
    DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    DeviceTensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

struct BailingMoe3MtpWeights {
    int device = 0;
    DeviceTensor eh_proj, enorm, hnorm, shared_head_norm;
    DeviceTensor attn_norm, ffn_norm;
    DeviceTensor wq, wq_a, attn_q_a_norm, wq_b;
    DeviceTensor wkv_a_mqa, attn_kv_a_norm, wk_b, wv_b, attn_gate, wo;
    DeviceTensor ffn_gate_inp, ffn_exp_probs_b;
    DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    DeviceTensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

class BailingMoe3Model {
public:
    BailingMoe3Model() = default;
    ~BailingMoe3Model();

    BailingMoe3Model(const BailingMoe3Model &) = delete;
    BailingMoe3Model & operator=(const BailingMoe3Model &) = delete;

    void load(const std::string & gguf_path,
              const std::vector<int> & gpu_ids,
              const std::vector<float> & split_weights,
              int max_ctx = 131072,
              bool kv_q8_0 = true,
              bool load_mtp = true);

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

    bool has_mtp() const { return has_mtp_; }

    void set_layer_hook(LayerHook * hook) { hook_ = hook; invalidate_graphs(); }
    void set_use_cuda_graphs(bool enable) { use_cuda_graphs_ = enable; invalidate_graphs(); }
    void warmup_decode_graphs();

    void forward(const std::vector<int32_t> & tokens,
                 const std::vector<int32_t> & positions,
                 std::vector<float> & out_logits);

    int32_t forward_decode_greedy(int32_t token, int32_t pos);
    void    forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits);

    // NextN MTP speculative decoding API (matching Glm5NextModel for unified rookery-server!)
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

    void rollback_kda_states();

    const BailingMoe3Config & config() const { return cfg_; }
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

        float *   d_cur       = nullptr; // [T, n_embd]
        float *   d_x_norm    = nullptr; // [T, n_embd]
        float *   d_sub_out   = nullptr; // [T, n_embd]

        // KDA scratch (d_inner = n_head * 128)
        float *   d_kda_q     = nullptr;
        float *   d_kda_k     = nullptr;
        float *   d_kda_v     = nullptr;
        float *   d_kda_qc    = nullptr;
        float *   d_kda_kc    = nullptr;
        float *   d_kda_vc    = nullptr;
        float *   d_kda_g     = nullptr;
        float *   d_kda_beta  = nullptr;
        float *   d_kda_out   = nullptr;
        float *   d_kda_gate  = nullptr;

        // Gated RoPE-576 MLA scratch
        float *   d_mla_qr    = nullptr; // [T, max(1, q_lora_rank)]
        float *   d_mla_qall  = nullptr; // [T, n_head * qk_head_dim_mla]
        float *   d_mla_qnope = nullptr; // [T, n_head * (qk_head_dim_mla - rope_dim)]
        float *   d_mla_qpe   = nullptr; // [T, n_head * rope_dim]
        float *   d_mla_qabs  = nullptr; // [T, n_head * kv_lora_rank]
        float *   d_mla_q576  = nullptr; // [T, n_head * (kv_lora_rank + rope_dim)]
        float *   d_mla_kvall = nullptr; // [T, kv_lora_rank + rope_dim]
        float *   d_mla_olat  = nullptr; // [T, n_head * kv_lora_rank]
        float *   d_mla_o     = nullptr; // [T, n_head * v_head_dim_mla]
        float *   d_mla_gate  = nullptr; // [T, n_head]

        // FFN / MoE scratch
        float *   d_ffn_g     = nullptr; // [T, n_ff_dense]
        float *   d_ffn_u     = nullptr; // [T, n_ff_dense]
        float *   d_router    = nullptr; // [T, n_expert]
        int32_t * d_top_ids   = nullptr; // [T, n_expert_used]
        float *   d_top_w     = nullptr; // [T, n_expert_used]

        // IO buffers
        int32_t * d_tokens    = nullptr; // [T]
        int32_t * d_pos       = nullptr; // [T]
        float *   d_logits    = nullptr; // [2 * n_vocab] (last GPU)
        int32_t * d_argmax    = nullptr; // [2]           (last GPU)

        // NextN MTP buffers (last GPU only)
        float *   d_h_nextn    = nullptr; // [T, n_embd]
        float *   d_mtp_e      = nullptr; // [T, n_embd]
        float *   d_mtp_eh     = nullptr; // [T, 2 * n_embd]
        float *   d_mtp_cur    = nullptr; // [T, n_embd]
        int32_t * d_mtp_pos    = nullptr; // [T]
        int32_t * d_mtp_argmax = nullptr; // [1]

        void *    d_scratch     = nullptr;
        size_t    scratch_bytes = 0;
    };

    void run_gpu_segment(size_t gi, int T, bool run_argmax, bool save_kda_snap, int kv_len_max = 0);
    void run_mtp_layer(int T, bool run_argmax, int kv_len_max = 0);
    void embed_tokens_to_last_dev(const int32_t * h_tokens, int T);
    void run_pipeline(int T, bool run_argmax, bool save_kda_snap, int kv_len_max, bool prefill_sync = false);
    void capture_decode_graphs();
    void invalidate_graphs() noexcept;

    BailingMoe3Config                     cfg_;
    WeightPool                            pool_;
    DeviceTensor                          tok_embd_;
    DeviceTensor                          output_norm_;
    DeviceTensor                          output_;
    std::vector<BailingMoe3LayerWeights>  layers_;
    BailingMoe3MtpWeights                 mtp_;
    bool                                  has_mtp_ = false;
    std::vector<DevRuntime>               runtimes_;

    // KDA recurrent states per layer (only allocated where is_kda == true)
    std::vector<float *> kda_conv_q_;
    std::vector<float *> kda_conv_k_;
    std::vector<float *> kda_conv_v_;
    std::vector<float *> kda_state_;

    // KDA 1-token snapshot buffers for MTP reject rollback
    std::vector<float *> kda_conv_q_snap_;
    std::vector<float *> kda_conv_k_snap_;
    std::vector<float *> kda_conv_v_snap_;
    std::vector<float *> kda_state_snap_;

    // MLA RoPE-576 KV cache per layer (only allocated where is_kda == false)
    std::vector<void *>  mla_kv_cache_;
    void *               mtp_kv_cache_  = nullptr;
    cudaGraphExec_t      mtp_graph_t1_  = nullptr;
    cudaGraphExec_t      mtp_graph_t2_  = nullptr;

    int         max_batch_       = 512;
    LayerHook * hook_            = nullptr;
    bool        use_cuda_graphs_ = true;
    bool        graphs_ready_    = false;
};

} // namespace rookery
