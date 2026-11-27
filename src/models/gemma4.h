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

struct Gemma4Config {
    int   n_layers              = 35;
    int   n_embd                = 1536;
    int   n_vocab               = 262144;
    int   n_head                = 8;
    int   max_head_kv           = 1;
    int   head_dim_full         = 512;
    int   head_dim_swa          = 256;
    int   rope_dim_full         = 512;
    int   rope_dim_swa          = 256;
    float rope_freq_base_full   = 1000000.0f;
    float rope_freq_base_swa    = 10000.0f;
    float rope_scale_full       = 1.0f;
    float rms_eps               = 1e-6f;
    int   sliding_window        = 512;
    int   shared_kv_layers      = 20;
    int   n_layer_kv_from_start = 15; // layers [0..n_layer_kv_from_start) own KV caches
    int   n_embd_per_layer      = 256;
    float final_logit_softcap   = 30.0f;
    int   max_ctx               = 8192;
    KvCacheType kv_type         = KvCacheType::Q8_0;
    bool  tied_embeddings       = true;

    std::vector<int>  n_head_kv;  // [n_layers] (can vary per layer, e.g. 8 SWA / 1 Full on 12B)
    std::vector<int>  n_ff;       // [n_layers]
    std::vector<bool> is_swa;     // [n_layers]
    std::vector<int>  kv_source;  // [n_layers]: own layer index if < n_layer_kv_from_start, else shared source layer
    std::vector<int>  kv_cap;     // [n_layers]: ring capacity (swa+max_batch for SWA, max_ctx for Full)
};

struct Gemma4LayerWeights {
    int  device    = 0;
    bool is_swa    = true;
    bool has_kv    = true;
    int  kv_source = 0;
    int  kv_cap    = 0;
    int  n_head_kv = 1;
    int  n_ff      = 6144;

    // Attention
    DeviceTensor attn_norm;
    DeviceTensor wq, wk, wv, wo;
    DeviceTensor attn_q_norm, attn_k_norm, attn_post_norm;
    DeviceTensor rope_freqs; // full-attn layers only

    // Dense FFN (GeGLU)
    DeviceTensor ffn_norm;
    DeviceTensor ffn_gate, ffn_up, ffn_down;
    DeviceTensor ffn_post_norm;

    // Per-Layer Embedding (PLE) gating block
    DeviceTensor per_layer_inp_gate;
    DeviceTensor per_layer_proj;
    DeviceTensor per_layer_post_norm;

    // Layer output scalar
    DeviceTensor layer_output_scale;
};

class Gemma4Model {
public:
    Gemma4Model() = default;
    ~Gemma4Model();

    Gemma4Model(const Gemma4Model &) = delete;
    Gemma4Model & operator=(const Gemma4Model &) = delete;

    void load(const std::string & gguf_path,
              const std::vector<int> & gpu_ids,
              const std::vector<float> & split_weights,
              int max_ctx = 8192,
              KvCacheType kv_type = KvCacheType::Q8_0,
              float rope_scale_full = 1.0f);

    void load(const std::string & gguf_path,
              const std::vector<int> & gpu_ids,
              const std::vector<float> & split_weights,
              int max_ctx,
              bool kv_q8_0) {
        load(gguf_path, gpu_ids, split_weights, max_ctx, kv_q8_0 ? KvCacheType::Q8_0 : KvCacheType::F16, 1.0f);
    }

    void reset_state();

    void set_layer_hook(LayerHook * hook) { hook_ = hook; invalidate_graphs(); }
    void set_use_cuda_graphs(bool enable) { use_cuda_graphs_ = enable; invalidate_graphs(); }
    void warmup_decode_graphs();

    void forward(const std::vector<int32_t> & tokens,
                 const std::vector<int32_t> & positions,
                 std::vector<float> & out_logits,
                 bool compute_logits = true);

    int32_t forward_decode_greedy(int32_t token, int32_t pos);
    void    forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits);

    const Gemma4Config & config() const { return cfg_; }
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

        float *   d_cur          = nullptr; // [T, H]
        float *   d_norm         = nullptr; // [T, H]
        float *   d_sub          = nullptr; // [T, H]
        float *   d_attn_res     = nullptr; // [T, H]
        float *   d_q            = nullptr; // [T, n_head * max_head_dim]
        float *   d_k            = nullptr; // [T, n_kv * max_head_dim]
        float *   d_v            = nullptr; // [T, n_kv * max_head_dim]
        float *   d_attn_out     = nullptr; // [T, n_head * max_head_dim]
        float *   d_ff_gate      = nullptr; // [T, max_ff]
        float *   d_ff_up        = nullptr; // [T, max_ff]
        float *   d_ple_raw      = nullptr; // [T, n_layers * ple_dim] (GPU 0 only)
        float *   d_ple_proj     = nullptr; // [T, n_layers * ple_dim] (GPU 0 only)
        float *   d_ple          = nullptr; // [T, n_layers * ple_dim] (replicated / transferred per GPU if multi-GPU)
        float *   d_ple_gate     = nullptr; // [T, ple_dim]

        int32_t * d_tokens       = nullptr; // [T]
        int32_t * d_pos          = nullptr; // [T]
        float *   d_logits       = nullptr; // [n_vocab] on last GPU
        int32_t * d_argmax       = nullptr; // [1] on last GPU

        void *    d_scratch      = nullptr;
        size_t    scratch_bytes  = 0;
    };

    void run_gpu_segment(size_t gi, int T, bool run_argmax, int pos_min, int pos_max,
                         bool prune_prefill, bool compute_logits,
                         int ple_tok_off_in, int & T_out, int & ple_tok_off_out);
    void capture_decode_graphs();
    void invalidate_graphs() noexcept;

    Gemma4Config                    cfg_;
    WeightPool                      pool_;
    DeviceTensor                    tok_embd_;
    DeviceTensor                    per_layer_tok_embd_;
    DeviceTensor                    per_layer_model_proj_;
    DeviceTensor                    per_layer_proj_norm_;
    DeviceTensor                    output_norm_;
    DeviceTensor                    output_; // may point to tok_embd_ if tied on same GPU
    std::vector<Gemma4LayerWeights> layers_;

    // KV caches (indexed by layer il; only allocated for il < n_layer_kv_from_start)
    std::vector<void *>             k_cache_;
    std::vector<void *>             v_cache_;
    // If a shared-KV layer sits on a different GPU than its kv_source, mirror the source KV cache on that GPU
    struct KvMirror {
        int    src_layer = -1;
        int    dst_dev   = -1;
        void * k_buf     = nullptr;
        void * v_buf     = nullptr;
        size_t bytes     = 0;
    };
    std::vector<KvMirror>           kv_mirrors_;

    std::vector<DevRuntime>         runtimes_;
    int                             max_batch_       = 512;
    bool                            use_cuda_graphs_ = true;
    bool                            graphs_ready_    = false;
    LayerHook *                     hook_            = nullptr;
};

} // namespace rookery
