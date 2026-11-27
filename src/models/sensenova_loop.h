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

struct SenseNovaLoopConfig {
    int   n_phys               = 32;
    int   n_visits             = 56;
    int   n_nextn              = 1;

    int   loop_start           = 4;
    int   loop_end             = 28;
    int   loop_count           = 2;

    int   mhc_start            = 4;
    int   mhc_end              = 28;
    int   mhc_streams          = 2;
    int   mhc_iterations       = 20;
    bool  mhc_fixed_identity   = false;
    float mhc_pre_scale        = 1.0f;

    int   n_embd               = 4096;
    int   n_ff                 = 12288;
    int   n_vocab              = 248320;
    int   n_head               = 16;
    int   n_head_kv            = 4;
    int   head_dim             = 256;
    int   rope_dim             = 64;
    float rope_freq_base       = 10000000.0f;
    float rope_yarn_factor     = 2.0f;
    int   rope_orig_ctx        = 262144;
    float yarn_mscale          = 1.0f;
    float rms_eps              = 1e-6f;

    // GDN linear-attention dimensions
    int   ssm_d_conv           = 4;
    int   ssm_d_state          = 128;
    int   ssm_n_group          = 16;   // n_k_heads
    int   ssm_dt_rank          = 32;   // n_v_heads
    int   ssm_d_inner          = 4096; // n_v_heads * head_v_dim
    int   full_attn_interval   = 4;

    int   max_ctx              = 131072;
    bool  kv_q8_0              = true;

    std::vector<int>  plan;         // [n_visits]: physical layer index per visit
    std::vector<bool> phys_is_recr; // [n_phys]
};

struct SenseNovaLoopLayerWeights {
    int  device  = 0;
    bool is_recr = false;
    bool has_mhc = false;

    DeviceTensor attn_norm, post_attn_norm;

    // 2-stream mHC routers (FP32, present for physical layers in [mhc_start, mhc_end))
    DeviceTensor hc_attn_w, hc_attn_b;
    DeviceTensor hc_ffn_w, hc_ffn_b;

    // GDN linear attention (fused attn_qkv_gate [H, 12288] & ssm_beta_alpha [H, 64])
    DeviceTensor wqkv_gate;      // fused [Q, K, V, Z]
    DeviceTensor ssm_beta_alpha; // fused [beta, alpha]
    DeviceTensor wqkv, wgate;    // fallback if unfused GGUF
    DeviceTensor ssm_alpha, ssm_beta;
    DeviceTensor ssm_conv1d, ssm_dt, ssm_a, ssm_norm, ssm_out;

    // Full GQA attention
    DeviceTensor wq, wk, wv, wo;
    DeviceTensor attn_q_norm, attn_k_norm;

    // Dense SwiGLU FFN
    DeviceTensor ffn_gate, ffn_up, ffn_down;
};

struct SenseNovaLoopMtpWeights {
    int device = 0;
    DeviceTensor eh_proj, enorm, hnorm, shared_head_norm;
    DeviceTensor attn_norm, post_attn_norm;
    DeviceTensor wq, wk, wv, wo;
    DeviceTensor attn_q_norm, attn_k_norm;
    DeviceTensor ffn_gate, ffn_up, ffn_down;
};

class SenseNovaLoopModel {
public:
    SenseNovaLoopModel() = default;
    ~SenseNovaLoopModel();

    SenseNovaLoopModel(const SenseNovaLoopModel &) = delete;
    SenseNovaLoopModel & operator=(const SenseNovaLoopModel &) = delete;

    void load(const std::string & gguf_path,
              const std::vector<int> & gpu_ids,
              const std::vector<float> & split_weights = {},
              int max_ctx = 131072,
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
    void rollback_gdn_states();

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

    const SenseNovaLoopConfig & config() const { return cfg_; }
    std::vector<int> layer_devices() const {
        std::vector<int> devs(cfg_.n_visits);
        for (int v = 0; v < cfg_.n_visits; ++v) devs[v] = layers_[cfg_.plan[v]].device;
        return devs;
    }

private:
    struct GpuRuntime {
        int            device        = 0;
        int            phys_start    = 0;
        int            phys_end      = 0;
        cudaStream_t   stream        = nullptr;
        cudaEvent_t    done_event    = nullptr;
        cublasHandle_t cublas        = nullptr;

        float *   d_inv_freq         = nullptr; // [rope_dim / 2] YaRN inv_freq
        float *   d_cur              = nullptr; // [B, H]
        float *   d_res              = nullptr; // [B, 2, H]
        float *   d_mhc_ctrl         = nullptr; // [B, 6]
        float *   d_x_norm           = nullptr; // [B, H]
        float *   d_sub_out          = nullptr; // [B, H]

        // GDN buffers
        float *   d_qkvz             = nullptr; // [B, 12288]
        float *   d_qkv_conv         = nullptr; // [B, 8192]
        float *   d_ba               = nullptr; // [B, 64]
        float *   d_alpha            = nullptr; // [B, 32]
        float *   d_beta             = nullptr; // [B, 32]
        float *   d_gdn_out          = nullptr; // [B, 4096]

        // Full GQA buffers
        float *   d_q_full           = nullptr; // [B, n_head * 2 * head_dim]
        float *   d_q                = nullptr; // [B, n_head * head_dim]
        float *   d_gate             = nullptr; // [B, n_head * head_dim]
        float *   d_k                = nullptr; // [B, n_kv * head_dim]
        float *   d_v                = nullptr; // [B, n_kv * head_dim]
        float *   d_attn_out         = nullptr; // [B, n_head * head_dim]

        // Dense FFN buffers
        float *   d_ff_gate          = nullptr; // [B, n_ff]
        float *   d_ff_up            = nullptr; // [B, n_ff]

        // Token / pos / logits / MTP buffers
        int32_t * d_tokens           = nullptr;
        int32_t * d_pos              = nullptr;
        float *   d_logits           = nullptr; // [2, n_vocab]
        int32_t * d_argmax           = nullptr; // [2]

        float *   d_h_nextn          = nullptr; // [B, H]
        float *   d_mtp_e            = nullptr; // [B, H]
        float *   d_mtp_eh           = nullptr; // [B, 2*H]
        float *   d_mtp_cur          = nullptr; // [B, H]
        int32_t * d_mtp_pos          = nullptr; // [B]
        int32_t * d_mtp_argmax       = nullptr; // [1]

        void *    d_scratch          = nullptr;
        size_t    scratch_bytes      = 0;

        cudaGraphExec_t decode_graph_t1 = nullptr;
        cudaGraphExec_t decode_graph_t2 = nullptr;
    };

    struct VisitSegment {
        size_t gpu_idx     = 0;
        int    visit_start = 0;
        int    visit_end   = 0;
    };

    void run_visit_segment(size_t seg_idx, int T, bool run_argmax, bool save_gdn_snap,
                           int kv_len_max, bool compute_logits = true);
    void run_mtp_layer(int T, bool run_argmax, int kv_len_max);
    void embed_tokens_to_last_dev(const int32_t * h_tokens, int T);
    void capture_decode_graphs();
    void invalidate_graphs() noexcept;

    SenseNovaLoopConfig                    cfg_;
    WeightPool                             pool_;
    DeviceTensor                           tok_embd_, output_norm_, output_;
    std::vector<SenseNovaLoopLayerWeights> layers_;       // [n_phys]
    SenseNovaLoopMtpWeights                mtp_;
    bool                                   has_mtp_ = false;

    // Per-logical-visit state slots [n_visits]
    std::vector<float *> gdn_conv_;
    std::vector<float *> gdn_state_;
    std::vector<float *> gdn_conv_snap_;
    std::vector<float *> gdn_state_snap_;
    std::vector<void *>  k_cache_;
    std::vector<void *>  v_cache_;
    void *               mtp_k_cache_ = nullptr;
    void *               mtp_v_cache_ = nullptr;

    std::vector<GpuRuntime>    runtimes_;
    std::vector<VisitSegment>  segments_;
    std::vector<cudaGraphExec_t> seg_graphs_t1_;
    std::vector<cudaGraphExec_t> seg_graphs_t2_;
    cudaGraphExec_t            mtp_graph_t1_ = nullptr;
    cudaGraphExec_t            mtp_graph_t2_ = nullptr;
    bool                       use_cuda_graphs_ = true;
    bool                       graphs_ready_    = false;
    LayerHook *                hook_            = nullptr;
    int                        max_batch_       = 512;
};

} // namespace rookery
