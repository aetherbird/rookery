#pragma once

#include "core/device.h"
#include "gguf/loader.h"
#include "gguf/sharded.h"
#include "models/maxwell_moe.h" // for LayerHook

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace rookery {

struct Inkling2Config {
    int   n_phys               = 24;
    int   n_slots              = 48; // logical KV/conv slots = n_phys + (n_passes - 1)*(core_end - core_start)
    int   core_start           = 6;
    int   core_end             = 18;
    int   n_passes             = 3;
    int   gate_rank            = 64;
    int   attn_res_block_size  = 6;

    int   n_embd               = 2304;
    int   n_head               = 24;
    int   n_head_kv            = 8;
    int   head_dim             = 128;
    int   n_expert             = 128;
    int   n_expert_used        = 8;
    int   n_ff_exp             = 1152;
    int   n_vocab_padded       = 201024;
    int   n_vocab_unpadded     = 200058;

    int   swa_window           = 1024;
    int   swa_cap              = 4096; // ring buffer size for SWA KV slots (allows up to 3072-tok prefix rollback!)
    int   n_conv               = 4;
    int   n_rel                = 16;
    int   n_rel_extent         = 1024;
    float qk_clip              = 22.627417f;
    float rel_clip             = 8.0f;
    int   log_scale_floor      = 8192;
    float log_scale_alpha      = 0.1f;
    float situ_beta            = 4.0f;
    float situ_linear_beta     = 25.0f;
    float expert_weights_scale = 8.0f;
    float logit_softcap        = 30.0f;
    float rms_eps              = 1e-6f;

    int         max_ctx        = 192000;
    KvCacheType kv_type        = KvCacheType::Q8_0;

    std::vector<bool> is_swa; // [n_phys]
};

struct Inkling2LoopCtrlWeights {
    DeviceTensor msg_query;        // [n_embd, 4] F32
    DeviceTensor ctrl_down[4];     // [2*n_embd, gate_rank] F32 (Q, K, V, FFN)
    DeviceTensor ctrl_gain[4];     // [gate_rank, n_embd] F32
    DeviceTensor ctrl_forget[4];   // [gate_rank, n_embd] F32
};

struct Inkling2LayerWeights {
    int  device = 0;
    bool is_swa = true;

    // Attention projections
    DeviceTensor wq, wk, wv, wo, wqkv_gate;
    DeviceTensor attn_q_clip, attn_sink_proj;
    DeviceTensor attn_rel, attn_rel_bias, attn_rel_clip;

    // Short convolutions
    DeviceTensor attn_k_conv, attn_v_conv, attn_post_conv, ffn_post_conv;
    DeviceTensor attn_k_conv_proj, attn_v_conv_proj; // global layers only

    // Norms & Block Attention Residuals
    DeviceTensor attn_norm; // SWA only
    DeviceTensor attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm; // global only
    DeviceTensor attn_res_query; // optional (absent on ip == 0)
    DeviceTensor ffn_res_query;

    // MoE
    DeviceTensor ffn_norm;
    DeviceTensor ffn_gate_inp;
    DeviceTensor ffn_exp_probs_b;
    DeviceTensor ffn_gate_up_exps;
    DeviceTensor ffn_down_exps;

    // Loop controller (only populated for ip in [core_start, core_end))
    bool has_loop_ctrl = false;
    Inkling2LoopCtrlWeights loop;
};

class Inkling2LoopModel {
public:
    Inkling2LoopModel() = default;
    ~Inkling2LoopModel();

    Inkling2LoopModel(const Inkling2LoopModel &) = delete;
    Inkling2LoopModel & operator=(const Inkling2LoopModel &) = delete;

    void load(const std::string & gguf_path,
              const std::vector<int> & gpu_ids,
              int max_ctx = 65536,
              KvCacheType kv_type = KvCacheType::Q8_0);

    void reset_state();

    struct ConvCheckpoint {
        int pos = 0;
        std::vector<std::vector<float>> conv_k;
        std::vector<std::vector<float>> conv_v;
        std::vector<std::vector<float>> conv_attn;
        std::vector<std::vector<float>> conv_ffn;
    };
    void save_conv_checkpoint(ConvCheckpoint & out, int pos) const;
    void restore_conv_checkpoint(const ConvCheckpoint & cp);

    // Maximum tokens we can safely roll back in prefix cache without SWA ring-buffer wraparound
    int swa_safe_rollback_limit() const {
        return std::max(0, cfg_.swa_cap - cfg_.swa_window);
    }

    void set_layer_hook(LayerHook * hook) { hook_ = hook; invalidate_graphs(); }
    void set_use_cuda_graphs(bool enable) { use_cuda_graphs_ = enable; invalidate_graphs(); }
    void warmup_decode_graphs();

    void forward(const std::vector<int32_t> & tokens,
                 const std::vector<int32_t> & positions,
                 std::vector<float> & out_logits);

    int32_t forward_decode_greedy(int32_t token, int32_t pos);
    void    forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits);

    const Inkling2Config & config() const { return cfg_; }
    std::vector<int> layer_devices() const {
        std::vector<int> out;
        out.reserve(layers_.size());
        for (const auto & L : layers_) out.push_back(L.device);
        return out;
    }

private:
    static constexpr int MAX_BLOCKS = 8;
    static constexpr int MAX_BANK   = 4;

    struct DevRuntime {
        int            device     = 0;
        cudaStream_t   stream     = nullptr;
        cudaEvent_t    done_event = nullptr;
        cublasHandle_t cublas     = nullptr;

        // Activation / residual stream buffers ([max_batch_, n_embd])
        float * d_cur        = nullptr;
        float * d_inp_sa     = nullptr;
        float * d_ffn_inp    = nullptr;
        float * d_prelude    = nullptr;
        float * d_bank_out   = nullptr;
        float * d_blocks[MAX_BLOCKS] = {};
        float * d_bank[MAX_BANK]     = {};
        const float ** d_block_ptrs  = nullptr; // device array [MAX_BLOCKS]
        const float ** d_bank_ptrs   = nullptr; // device array [MAX_BANK]

        // Depth mix & loop controller scratch
        float * d_mixed3     = nullptr; // [3, max_batch_, n_embd]
        float * d_messages   = nullptr; // [4, max_batch_, n_embd]
        float * d_inputs[3]  = {};      // [max_batch_, n_embd]
        float * d_ctrl_in    = nullptr; // [max_batch_, 2*n_embd]
        float * d_ctrl_hid   = nullptr; // [max_batch_, gate_rank]
        float * d_ctrl_gain  = nullptr; // [max_batch_, n_embd]
        float * d_ctrl_forg  = nullptr; // [max_batch_, n_embd]

        // Attention scratch
        float * d_q          = nullptr; // [max_batch_, d_q]
        float * d_k_raw      = nullptr; // [max_batch_, d_kv]
        float * d_v_raw      = nullptr; // [max_batch_, d_kv]
        float * d_k          = nullptr; // [max_batch_, d_kv]
        float * d_v          = nullptr; // [max_batch_, d_kv]
        float * d_k_dyn      = nullptr; // [max_batch_, n_kv * 4]
        float * d_v_dyn      = nullptr; // [max_batch_, n_kv * 4]
        float * d_r          = nullptr; // [max_batch_, n_head * n_rel]
        float * d_sinks      = nullptr; // [max_batch_, n_head]
        float * d_bias_table = nullptr; // [max_batch_, n_head, n_extent + 1]
        float * d_attn_out   = nullptr; // [max_batch_, d_q]
        float * d_attn_gate  = nullptr; // [max_batch_, d_q]
        float * d_wo_raw     = nullptr; // [max_batch_, n_embd]
        float * d_sub_out    = nullptr; // [max_batch_, n_embd]

        // MoE scratch
        float *   d_x_norm   = nullptr; // [max_batch_, n_embd]
        float *   d_router   = nullptr; // [max_batch_, n_expert]
        int32_t * d_top_ids  = nullptr; // [max_batch_, n_expert_used]
        float *   d_top_w    = nullptr; // [max_batch_, n_expert_used]
        float *   d_moe_raw  = nullptr; // [max_batch_, n_embd]

        // IO buffers (on GPU 0)
        int32_t * d_tokens   = nullptr; // [max_batch_]
        int32_t * d_pos      = nullptr; // [max_batch_]
        float *   d_logits   = nullptr; // [n_vocab_padded]
        int32_t * d_argmax   = nullptr; // [1]

        void * d_scratch     = nullptr;
        size_t scratch_bytes = 0;
    };

    void run_layer(DevRuntime & rt, int il, int ip, int T, int & n_blocks, bool has_messages, int kv_len_max);
    void run_prelude(int T, int kv_len_max);
    void run_core(int T, int kv_len_max);
    void run_postlude_prefix(int T, int kv_len_max);
    void run_postlude_and_head(int T, bool run_argmax, int kv_len_max);
    void run_head_only(bool run_argmax);
    void run_single_gpu_all(int T, bool run_argmax, int kv_len_max);
    void capture_decode_graphs();
    void invalidate_graphs() noexcept;

    Inkling2Config                    cfg_;
    WeightPool                        pool_;
    DeviceTensor                      tok_embd_, tok_norm_;
    DeviceTensor                      output_res_query_, output_norm_, output_;
    std::vector<Inkling2LayerWeights> layers_; // [n_phys]

    // Per-logical-slot persistent state ([n_slots], allocated on slot's owning GPU)
    std::vector<int>     slot_device_;
    std::vector<void *>  k_cache_;
    std::vector<void *>  v_cache_;
    std::vector<float *> conv_k_;
    std::vector<float *> conv_v_;
    std::vector<float *> conv_attn_;
    std::vector<float *> conv_ffn_;

    // 1-GPU: runtimes_.size() == 1 (all layers on rt[0], 1 single CUDA Graph)
    // 2-GPU symmetric (postlude_split_ip_ < 0): rt[0] = prelude + postlude + embd/out, rt[1] = looped core (3 CUDA Graphs)
    // 2-GPU asymmetric (postlude_split_ip_ >= core_end): rt[0] = embd + [0, postlude_split_ip_), rt[1] = [postlude_split_ip_, n_phys) + out (2 CUDA Graphs, 1 hop)
    std::vector<DevRuntime> runtimes_;
    int                     postlude_split_ip_ = -1;
    cudaGraphExec_t         graph_all_         = nullptr; // 1-GPU
    cudaGraphExec_t         graph_prelude_     = nullptr; // 2-GPU GPU0 (prelude, or prelude+core+postlude_prefix when asymmetric)
    cudaGraphExec_t         graph_core_        = nullptr; // 2-GPU GPU1 (symmetric mode only)
    cudaGraphExec_t         graph_postlude_    = nullptr; // 2-GPU postlude (GPU0 symmetric, GPU1 asymmetric)
    cudaGraphExec_t         graph_head_        = nullptr; // 2-GPU asymmetric lm_head on GPU0

    int         max_batch_       = 256;
    bool        use_cuda_graphs_ = true;
    bool        graphs_ready_    = false;
    LayerHook * hook_            = nullptr;
};

} // namespace rookery
