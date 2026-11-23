#pragma once

#include "core/device.h"
#include "gguf/loader.h"
#include "gguf/sharded.h"
#include "models/maxwell_moe.h" // LayerHook

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace rookery {

struct Qwen4ExpConfig {
    int   n_layers             = 48;
    int   n_embd               = 2560;
    int   n_vocab              = 248320;
    int   n_head               = 24;
    int   n_head_kv            = 2;
    int   head_dim             = 256;
    int   rope_dim             = 64;
    float rope_freq_base       = 10000000.0f;
    float rms_eps              = 1e-6f;

    // GDN linear-attention
    int   ssm_d_conv           = 4;
    int   ssm_d_state          = 128;
    int   ssm_n_group          = 16;   // n_k_heads
    int   ssm_dt_rank          = 48;   // n_v_heads
    int   ssm_d_inner          = 6144; // n_v_heads * 128
    int   full_attn_interval   = 4;

    // MoE
    int   n_expert             = 512;
    int   n_expert_used        = 10;
    int   n_ff_exp             = 640;
    int   n_ff_shexp           = 640;
    float expert_weights_scale = 1.0f;

    // Low-rank mHC
    int   hc_mult              = 4;
    int   hc_low_rank          = 320;

    // QSA sparse indexer
    int   idx_n_head           = 4;
    int   idx_head_dim         = 128;
    int   idx_top_k            = 2048;
    int   idx_kpool            = 4;

    // PLE n-gram Engram (host RAM mmap'd Q8_0 table!)
    int   ple_ngram_size       = 3;
    int   ple_heads_per_ngram  = 8;
    int   ple_n_heads          = 16;
    int   ple_head_dim         = 160;
    int   ple_conv_kernel      = 4;
    int32_t ple_eos_token_id   = 248044;
    std::vector<uint64_t> ple_multipliers;
    std::vector<uint64_t> ple_head_offsets;
    std::vector<uint64_t> ple_head_vocab_sizes;

    int   max_ctx              = 262144;
    bool  kv_q8_0              = true;

    std::vector<bool> is_recr; // [n_layers]: true = GDN, false = Full QSA
    std::vector<bool> is_ple;  // [n_layers]: true on blk.1
};

struct Qwen4LayerWeights {
    int  device  = 0;
    bool is_recr = false;
    bool is_ple  = false;

    // Low-rank mHC pre-attn & pre-ffn
    DeviceTensor hc_attn_norm, hc_attn_down, hc_attn_up, hc_attn_inject;
    DeviceTensor hc_ffn_norm,  hc_ffn_down,  hc_ffn_up,  hc_ffn_inject;

    // PLE Engram (blk.1 only)
    DeviceTensor ple_key, ple_value;
    DeviceTensor ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv1d;

    // GDN linear attention
    DeviceTensor wqkv, wqkv_gate;
    DeviceTensor ssm_conv1d, ssm_dt, ssm_a, ssm_alpha, ssm_beta, ssm_norm, ssm_out;

    // Full QSA attention
    DeviceTensor wq, wk, wv, wo;
    DeviceTensor attn_q_norm, attn_k_norm;
    DeviceTensor idx_q_proj, idx_k_proj, idx_q_norm, idx_k_norm;

    // MoE + Gated Shared Expert
    DeviceTensor ffn_gate_inp;
    DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    DeviceTensor ffn_gate_inp_shexp;
    DeviceTensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

class Qwen4ExpModel {
public:
    Qwen4ExpModel() = default;
    ~Qwen4ExpModel();

    Qwen4ExpModel(const Qwen4ExpModel &) = delete;
    Qwen4ExpModel & operator=(const Qwen4ExpModel &) = delete;

    void load(const std::string & gguf_path,
              const std::vector<int> & gpu_ids,
              const std::vector<float> & split_weights,
              int max_ctx = 65536,
              bool kv_q8_0 = true);

    void reset_state();

    struct StateCheckpoint {
        int pos = 0;
        std::vector<std::vector<float>> gdn_conv;
        std::vector<std::vector<float>> gdn_ssm;
        std::vector<std::vector<float>> ple_conv;
        std::vector<int32_t>            prev_tokens; // token history [0..pos-1] for PLE n-gram hashing
    };
    void save_checkpoint(StateCheckpoint & out, int pos) const;
    void restore_checkpoint(const StateCheckpoint & cp);

    void set_layer_hook(LayerHook * hook) { hook_ = hook; invalidate_graphs(); }
    void set_use_cuda_graphs(bool enable) { use_cuda_graphs_ = enable; invalidate_graphs(); }
    void warmup_decode_graphs();

    void forward(const std::vector<int32_t> & tokens,
                 const std::vector<int32_t> & positions,
                 std::vector<float> & out_logits);

    int32_t forward_decode_greedy(int32_t token, int32_t pos);
    void    forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits);

    const Qwen4ExpConfig & config() const { return cfg_; }
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
        bool            has_ple_layer   = false;
        cudaStream_t    stream          = nullptr;
        cudaEvent_t     done_event      = nullptr;
        cublasHandle_t  cublas          = nullptr;
        cudaGraphExec_t decode_graph_t1 = nullptr;

        // Wide mHC residual & mix scratch
        float *   d_res_hc     = nullptr; // [T, hc * H]
        float *   d_hc_xn      = nullptr; // [T, hc * H]
        float *   d_hc_lo      = nullptr; // [T, hc_lr]
        float *   d_hc_gate    = nullptr; // [T, hc * H]
        float *   d_hc_inject  = nullptr; // [T, hc]
        float *   d_cur        = nullptr; // [T, H]
        float *   d_sub_out    = nullptr; // [T, H]

        // PLE Engram scratch (allocated only on GPU owning a PLE layer)
        float *   d_ple_emb    = nullptr; // [T, n_heads * head_dim]
        float *   d_ple_key    = nullptr; // [T, hc * H]
        float *   d_ple_query  = nullptr; // [T, hc * H]
        float *   d_ple_val    = nullptr; // [T, H]
        float *   d_ple_gated  = nullptr; // [T, hc * H]
        float *   d_ple_norm   = nullptr; // [T, hc * H]

        // GDN scratch
        float *   d_qkv_raw    = nullptr; // [T, conv_dim]
        float *   d_qkv_conv   = nullptr; // [T, conv_dim]
        float *   d_z          = nullptr; // [T, d_inner]
        float *   d_alpha      = nullptr; // [T, n_v_heads]
        float *   d_beta       = nullptr; // [T, n_v_heads]
        float *   d_gdn_out    = nullptr; // [T, d_inner]

        // Full QSA attention scratch
        float *   d_q_full     = nullptr; // [T, n_head * 512]
        float *   d_q          = nullptr; // [T, n_head * 256]
        float *   d_gate       = nullptr; // [T, n_head * 256]
        float *   d_k          = nullptr; // [T, n_kv * 256]
        float *   d_v          = nullptr; // [T, n_kv * 256]
        float *   d_attn_out   = nullptr; // [T, n_head * 256]
        float *   d_idx_q      = nullptr; // [T, idx_n_head * 128]
        float *   d_idx_k      = nullptr; // [T, 128]
        int32_t * d_sel_pos    = nullptr; // [T, top_k + kpool]
        int32_t * d_sel_len    = nullptr; // [T]

        // MoE scratch
        float *   d_router     = nullptr; // [T, n_expert]
        float *   d_shexp_gate = nullptr; // [T]
        int32_t * d_top_ids    = nullptr; // [T, k_used]
        float *   d_top_w      = nullptr; // [T, k_used]

        int32_t * d_tokens     = nullptr;
        int32_t * d_pos        = nullptr;
        float *   d_logits     = nullptr; // [n_vocab] on last GPU
        int32_t * d_argmax     = nullptr; // [1] on last GPU

        void *    d_scratch     = nullptr;
        size_t    scratch_bytes = 0;
    };

    void compute_ple_host(const int32_t * tokens, const int32_t * positions, int T, std::vector<float> & out_emb);
    void run_hc_mix(DevRuntime & rt, const DeviceTensor & w_norm, const DeviceTensor & w_down,
                    const DeviceTensor & w_up, const DeviceTensor & w_inject, int T);
    void run_gpu_segment(size_t gi, int T, bool run_argmax, int kv_len_max = 0);
    void capture_decode_graphs();
    void invalidate_graphs() noexcept;

    Qwen4ExpConfig                   cfg_;
    ShardedGguf                      sg_;            // Kept open for mmap'd host-RAM PLE table!
    const uint8_t *                  ple_tbl_bytes_ = nullptr;
    size_t                           ple_row_bytes_ = 0;
    float                            ple_scale_     = 1.0f;
    std::vector<int32_t>             token_history_; // [max_ctx] for PLE predecessor lookup
    std::vector<float>               h_ple_buf_;

    WeightPool                       pool_;
    DeviceTensor                     tok_embd_;
    DeviceTensor                     output_hc_norm_, output_hc_down_, output_hc_up_, output_;
    std::vector<Qwen4LayerWeights>   layers_;

    std::vector<float *>             gdn_conv_;
    std::vector<float *>             gdn_state_;
    std::vector<float *>             ple_conv_;
    std::vector<void *>              k_cache_;
    std::vector<void *>              v_cache_;
    std::vector<float *>             idx_k_raw_cache_;
    std::vector<float *>             idx_k_pool_cache_;

    std::vector<DevRuntime>          runtimes_;
    int                              max_batch_       = 512;
    bool                             use_cuda_graphs_ = true;
    bool                             graphs_ready_    = false;
    LayerHook *                      hook_            = nullptr;
};

} // namespace rookery
