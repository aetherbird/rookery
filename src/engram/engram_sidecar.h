#pragma once

#include "engram/engram_table.h"
#include "models/maxwell_moe.h"
#include "tokenizer/bpe.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace rookery {

// Kernel launchers in engram_kernels.cu
void launch_engram_matvec_f32(
    const float * d_x, const float * d_W, float * d_y,
    int T, int M, int K, cudaStream_t stream);

void launch_engram_gated_inject(
    float * d_inpL, const float * d_val, const float * d_key, const float * d_gate_w,
    float scale, int T, int d_host, int hc, float rms_eps, cudaStream_t stream);

void launch_engram_mean_streams(
    const float * d_inpL, float * d_out, int T, int d_host, int hc, cudaStream_t stream);

struct EngramProfileStats {
    uint64_t steps             = 0;
    uint64_t tokens            = 0;
    double   tokenize_align_us = 0.0;
    double   hash_prefetch_us  = 0.0;
    double   gather_dequant_us = 0.0;
    double   h2d_submit_us     = 0.0;
    double   gpu_hook_us       = 0.0;
    uint64_t gpu_hook_calls    = 0;
};

struct EngramHookSpec {
    int   target_host_layer = 0;  // backbone layer index on host model (e.g. Maxwell 1, 11, 21, 31)
    int   src_engram_layer  = 0;  // index into EngramTable's layers (0..n_layers-1)
    int   device            = 0;  // GPU owning target_host_layer
    float scale             = 0.0f;

    // Optional RKAD v2 mean-centering vector mu_E[d_engram] subtracted on CPU staging buffer before H2D
    std::vector<float> h_mu_e;

    // Projection weights on `device`:
    // If rank == 0: dense W_v[d_host, d_engram]
    // If rank >  0: low-rank A_v[rank, d_engram], B_v[d_host, rank]
    int     rank    = 0;
    float * d_Av    = nullptr; // W_v (if rank==0) or A_v (if rank>0)
    float * d_Bv    = nullptr; // nullptr (if rank==0) or B_v (if rank>0)
    float * d_Ak    = nullptr; // optional key projection for context-dependent gate
    float * d_Bk    = nullptr;
    float * d_gatew = nullptr; // optional [hc, d_host]

    // Fixed-address pinned host + device staging buffers (100% CUDA-Graph compatible!)
    float * h_engram_pin = nullptr; // [max_batch, d_engram]
    float * d_engram_in  = nullptr; // [max_batch, d_engram]
    float * d_mid        = nullptr; // [max_batch, max(rank, 1)]
    float * d_val        = nullptr; // [max_batch, d_host]
    float * d_key        = nullptr; // [max_batch, d_host] (if gated)
    float * d_hmean      = nullptr; // [max_batch, d_host] (for calibration dumps)

    // Optional CUDA events for stage-by-stage GPU hook profiling (only when gpu_profile_ == true, non-graph)
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop  = nullptr;
};

// Decoupled, model-agnostic Engram Sidecar implementing `LayerHook`.
// Supports cross-model grafting (e.g. HY4-Flash 59 GiB or DeepSeek-V4.1-Flash 103 GiB Engram -> Maxwell-V1-Flash),
// dual-tokenizer rolling text-window bridging, pinned-RAM async H2D prefetch, CUDA-Graph compatibility,
// and paired (engram_vec, host_hidden) calibration dumps for closed-form Ridge/SVD adapter fitting!
class EngramSidecar : public LayerHook {
public:
    EngramSidecar() = default;
    ~EngramSidecar() override;

    EngramSidecar(const EngramSidecar &) = delete;
    EngramSidecar & operator=(const EngramSidecar &) = delete;

    // Initialize sidecar attached to a host model (`n_host_layers`, `d_host`, `hc`, per-layer GPU mapping).
    // `adapter_path`: optional `.rkad` adapter file; if empty, initializes deterministic norm-preserving low-rank projection.
    // `override_host_layers`: if non-empty, maps Engram layers onto these host layers instead of default proportional mapping.
    void init(const std::string & engram_path,
              const BpeTokenizer * host_tokenizer,
              const std::vector<int> & layer_to_device,
              int d_host,
              int hc,
              int max_batch = 512,
              const std::string & adapter_path = "",
              float global_scale = 0.0f,
              const std::vector<int> & override_host_layers = {});

    // Optional: enable binary dump of paired (hook_idx, engram_raw[T, d_engram], host_hidden_mean[T, d_host])
    // If `dump_delta == true`, dumps (e_raw_L, h_{L+1} - h_L) — the actual additive residual update of layer L+1!
    void enable_calibration_dump(const std::string & out_path, bool dump_delta = false);
    void close_calibration_dump();

    void set_shuffle_salt(uint64_t salt) { table_.set_shuffle_salt(salt); }
    void set_global_scale(float s) { global_scale_ = s; }
    float global_scale() const { return global_scale_; }
    const EngramTable & table() const { return table_; }
    size_t num_hooks() const { return hooks_.size(); }

    void set_gpu_profiling(bool enable) { gpu_profile_ = enable; }
    void reset_profile() { prof_ = EngramProfileStats{}; }
    const EngramProfileStats & profile() const { return prof_; }

    // Verifies that incremental token-by-token rolling-window tokenization + hashing produces
    // identical source token IDs and N-gram hash row IDs as full-sequence source tokenization.
    // Returns true iff 100% of tokens and hash columns match exactly.
    bool verify_incremental_addressing(const std::vector<int32_t> & host_tokens,
                                       int & out_tokens_checked,
                                       int & out_src_tok_matches,
                                       int & out_hash_col_matches,
                                       int & out_hash_col_total,
                                       bool verbose = false);

    // Call BEFORE each host `forward` / `forward_decode_*` / `decode_mtp_greedy_step`!
    // Translates host tokens -> source Engram tokens (via rolling UTF-8 text bridge),
    // computes exact N-gram hashes, gathers+dequantizes from mmap'd table into pinned RAM,
    // and issues async H2D copies into each hook's fixed GPU buffer.
    void prepare_step(const std::vector<int32_t> & host_tokens, int start_pos);

    // LayerHook callback invoked inside host model's layer loop (including inside CUDA Graphs!).
    void on_post_layer(int layer_idx, int device, float * d_inpL, int n_tokens, int n_embd, int hc, cudaStream_t stream) override;

private:
    void free_gpu_buffers() noexcept;
    void load_rkad_adapter(const std::string & adapter_path, const std::vector<int> & layer_to_device);
    void init_zero_shot_projection(const std::vector<int> & layer_to_device,
                                   const std::vector<int> & override_host_layers);

    EngramTable             table_;
    const BpeTokenizer *    host_tok_     = nullptr;
    int                     d_host_       = 0;
    int                     hc_           = 4;
    int                     max_batch_    = 512;
    float                   global_scale_ = 0.0f;
    bool                    gpu_profile_  = false;
    EngramProfileStats      prof_{};

    std::vector<EngramHookSpec> hooks_;
    std::vector<int>            layer_to_hook_; // [n_host_layers] -> index into hooks_ or -1

    // Rolling sequence state for dual-tokenizer text-window bridging
    std::string          rolling_text_;
    std::vector<int32_t> src_seq_cache_;
    std::vector<size_t>  host_byte_ends_; // byte offset in rolling_text_ after each host token

    // Optional calibration dump file ("RKPR" v1)
    FILE *               dump_fp_       = nullptr;
    bool                 dump_delta_    = false;
    uint64_t             dump_records_  = 0;
    std::vector<float>   dump_h_tmp_;
    std::vector<int>     delta_next_to_hook_; // [n_host_layers] -> hook idx waiting for L+1 delta, or -1
    std::vector<std::vector<float>> dump_h_prev_; // [n_hooks] cached h_L for L+1 delta
};

} // namespace rookery
