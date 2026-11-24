#pragma once

#include "core/device.h"
#include "gguf/loader.h"
#include "gguf/reader.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace rookery {

struct LayerHook {
    virtual ~LayerHook() = default;
    virtual bool hooks_layer(int /*layer_idx*/) const { return true; }
    virtual void on_post_layer(int layer_idx, int device, float * d_inpL, int n_tokens, int n_embd, int hc, cudaStream_t stream) = 0;
};

struct MaxwellConfig {
    int   n_blocks             = 46;
    int   n_layers             = 45;
    int   n_dense_lead         = 3;
    int   n_nextn              = 1;
    int   n_embd               = 2048;
    int   n_ff_dense           = 6144;
    int   n_vocab              = 200704;
    int   n_head               = 32;
    int   n_head_kv            = 4;
    int   head_dim             = 128;
    int   n_expert             = 256;
    int   n_expert_used        = 12;
    int   n_expert_shared      = 1;
    int   n_ff_exp             = 512;
    int   n_ff_shexp           = 1024;
    bool  expert_weights_norm  = true;
    float expert_weights_scale = 1.0f;
    int   hc_mult              = 4;
    int   hc_sinkhorn_iters    = 20;
    float hc_eps               = 1e-6f;
    float rms_eps              = 1e-6f;
    float rope_freq_base       = 20000000.0f;
    int         max_ctx        = 4096;
    KvCacheType kv_type        = KvCacheType::Q8_0;
    // MSA block-sparse attention params
    int   idx_n_head           = 4;
    int   idx_head_dim         = 128;
    int   idx_top_k            = 16;
    int   idx_blk_size         = 128;
    int   idx_local_blks       = 8;
    int   idx_prefix_blks      = 8;
    int   idx_dense_prefix     = 16384;
};

struct MaxwellLayerWeights {
    int device = 0;

    DeviceTensor attn_norm, ffn_norm;
    DeviceTensor wq, wk, wv, wo;
    DeviceTensor attn_q_norm, attn_k_norm;
    DeviceTensor idx_q_proj, idx_k_proj, idx_q_norm, idx_k_norm;

    DeviceTensor hc_attn_fn, hc_attn_base, hc_attn_scale;
    DeviceTensor hc_ffn_fn,  hc_ffn_base,  hc_ffn_scale;

    DeviceTensor ffn_gate, ffn_up, ffn_down;

    DeviceTensor ffn_gate_inp, ffn_exp_probs_b;
    DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    DeviceTensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

struct MaxwellMtpWeights {
    bool                loaded = false;
    MaxwellLayerWeights layer;
    DeviceTensor        eh_proj, enorm, hnorm;
    DeviceTensor        shared_head_norm;
    DeviceTensor        hc_head_fn, hc_head_base, hc_head_scale;
    DeviceTensor        tok_embd; // shared with tok_embd_ when on same GPU, else copy on last_dev
};

struct MtpStats {
    int steps    = 0;
    int accepted = 0;
    int emitted  = 0;
};

class MaxwellMoeModel {
public:
    MaxwellMoeModel() = default;
    ~MaxwellMoeModel();

    MaxwellMoeModel(const MaxwellMoeModel &) = delete;
    MaxwellMoeModel & operator=(const MaxwellMoeModel &) = delete;

    void load(const std::string & gguf_path,
              const std::vector<int> & gpu_ids,
              const std::vector<float> & split_weights,
              int max_ctx = 4096,
              KvCacheType kv_type = KvCacheType::Q8_0,
              bool load_mtp = true);

    void set_layer_hook(LayerHook * hook) { hook_ = hook; invalidate_graphs(); }
    void set_use_cuda_graphs(bool enable) { use_cuda_graphs_ = enable; invalidate_graphs(); }
    void set_mtp_pos_offset(int off) { mtp_pos_offset_ = off; }
    bool has_mtp() const { return mtp_.loaded; }

    // General forward (prefill T >= 1 or decode with full logits D2H copy).
    void forward(const std::vector<int32_t> & tokens,
                 const std::vector<int32_t> & positions,
                 std::vector<float> & out_logits);

    // Ultra-fast T==1 greedy decode step (CUDA Graph + GPU-side argmax, 4-byte D2H copy).
    int32_t forward_decode_greedy(int32_t token, int32_t pos);

    // Ultra-fast T==1 decode returning full logits (reuses T=1 CUDA Graph for sampled temperature > 0).
    void forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits);

    // Pre-capture T=1 and T=2 CUDA Graphs so the very first request hits steady-state speed immediately.
    void warmup_decode_graphs();

    // Chunked MTP prefill: call immediately after each forward() prefill chunk while d_inpL still holds
    // that chunk's backbone hidden states. Returns the MTP draft token for the last position in the chunk.
    int32_t mtp_prefill_chunk(const std::vector<int32_t> & next_tokens, const std::vector<int32_t> & positions);

    // Warm up MTP KV cache after single-chunk prompt prefill and produce first speculative draft token.
    int32_t mtp_prefill_and_draft(const std::vector<int32_t> & prompt, int32_t tok0);

    // One speculative MTP verify+draft step:
    // Verifies [cur_tok, draft_tok] at [cur_pos, cur_pos+1] in a single T=2 CUDA Graph pass,
    // emits 1 or 2 exact greedy tokens into `out_emitted`, updates `cur_pos` and `next_draft`,
    // and updates `stats`.
    void decode_mtp_greedy_step(int32_t & cur_tok,
                                int32_t & draft_tok,
                                int32_t & cur_pos,
                                std::vector<int32_t> & out_emitted,
                                MtpStats & stats);

    const MaxwellConfig & config() const { return cfg_; }
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

        float *   d_inpL      = nullptr;
        float *   d_inpL_alt  = nullptr;
        float *   d_mixes     = nullptr;
        float *   d_post      = nullptr;
        float *   d_comb      = nullptr;
        float *   d_x_norm    = nullptr;
        float *   d_q         = nullptr;
        float *   d_k         = nullptr;
        float *   d_v         = nullptr;
        float *   d_attn_out  = nullptr;
        float *   d_sub_out   = nullptr;
        float *   d_ffn_g     = nullptr;
        float *   d_ffn_u     = nullptr;
        float *   d_router    = nullptr;
        int32_t * d_top_ids   = nullptr;
        float *   d_top_w     = nullptr;
        int32_t * d_tokens    = nullptr;
        int32_t * d_pos       = nullptr;
        float *   d_logits    = nullptr; // [2, n_vocab] on head GPU
        int32_t * d_argmax    = nullptr; // [2] on head GPU
        void *    d_scratch   = nullptr;
        size_t    scratch_bytes = 0;

        // MSA scratch
        float *   d_iq         = nullptr; // [max_batch, idx_n_head * idx_head_dim]
        float *   d_ik         = nullptr; // [max_batch, idx_head_dim]
        int32_t * d_msa_blocks = nullptr; // [max_batch, idx_n_head, K_max]
        int32_t * d_msa_kcnt   = nullptr; // [max_batch]

        // MTP scratch on last GPU
        float *   d_mtp_e      = nullptr; // [max_batch, n_embd]
        float *   d_mtp_eh_in  = nullptr; // [max_batch * 4, 2 * n_embd]
        float *   d_mtp_inpL   = nullptr; // [max_batch, 4, n_embd]
        float *   d_mtp_alt    = nullptr; // [max_batch, 4, n_embd]
        int32_t * d_mtp_tok    = nullptr; // [max_batch]
        int32_t * d_mtp_pos    = nullptr; // [max_batch]
    };

    void run_gpu_segment(size_t gi, int T, bool run_argmax, bool logits_all_t = false, int kv_len_max = 0);
    void run_head_on_head_gpu(int head_T, bool run_argmax);
    void run_mtp_body_on_last_gpu(const float * d_h_states, int T_mtp, int kv_len_max = 0);
    int32_t run_mtp_on_last_gpu(const int32_t * h_tokens, const float * d_h_states, const int32_t * h_positions, int T_mtp);
    void capture_decode_graphs(int T);
    void invalidate_graphs() noexcept;

    MaxwellConfig                    cfg_;
    WeightPool                       pool_;
    DeviceTensor                     tok_embd_;
    DeviceTensor                     hc_head_fn_, hc_head_base_, hc_head_scale_;
    DeviceTensor                     output_norm_, output_;
    std::vector<MaxwellLayerWeights> layers_;
    std::vector<void *>              k_caches_;
    std::vector<void *>              v_caches_;
    std::vector<void *>              ik_caches_;
    bool                             msa_enabled_ = false;
    MaxwellMtpWeights                mtp_;
    void *                           k_cache_mtp_ = nullptr;
    void *                           v_cache_mtp_ = nullptr;
    std::vector<DevRuntime>          runtimes_;
    // Asymmetric dual-GPU mode: cold token_embd on last GPU, hot lm_head on GPU 0, max layers on GPU 0!
    bool                             asym_mode_       = false;
    size_t                           embd_gi_         = 0;
    size_t                           head_gi_         = 0;
    cudaGraphExec_t                  graph_embd_t1_   = nullptr;
    cudaGraphExec_t                  graph_embd_t2_   = nullptr;
    cudaGraphExec_t                  graph_head_t1_   = nullptr;
    cudaGraphExec_t                  graph_head_t2_   = nullptr;
    cudaGraphExec_t                  graph_mtp_t1_    = nullptr;
    cudaGraphExec_t                  graph_mtp_t2_    = nullptr;
    int                              max_batch_       = 512;
    bool                             use_cuda_graphs_ = true;
    bool                             graphs_t1_ready_ = false;
    bool                             graphs_t2_ready_ = false;
    int                              mtp_pos_offset_  = 0; // 0 = pair (h_p, x_{p+1}) at pos p; 1 = at pos p+1
    LayerHook *                      hook_            = nullptr;
};

} // namespace rookery
