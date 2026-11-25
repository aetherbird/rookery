#pragma once

#include "engram/engram_table.h"
#include "models/maxwell_moe.h"
#include "tokenizer/bpe.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace rookery {

// Kernel launchers in engram_kernels.cu
void launch_engram_matvec_f32(
    const float * d_x, const float * d_W, float * d_y,
    const EngramRuntimeParams * d_params,
    int T, int M, int K, cudaStream_t stream);

void launch_engram_gated_inject(
    float * d_inpL, const float * d_val, const float * d_key, const float * d_gate_w,
    const EngramRuntimeParams * d_params,
    int T, int d_host, int hc, float rms_eps, cudaStream_t stream);

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

struct EngramAdapterManifest {
    uint32_t    format_version       = 0; // 1, 2, or 3
    std::string schema;
    std::string provider_arch;
    std::string provider_mode;
    int         provider_ngram_size  = 0;
    int         provider_n_cols      = 0;
    int         provider_head_dim    = 0;
    std::string provider_fingerprint;
    std::string host_arch;
    int         host_n_layers        = 0;
    int         host_d_embd          = 0;
    int         host_hc_mult         = 0;
    std::string calibration;
};

// Pure-CPU contract validator for RKAD v3 JSON manifests (also unit-tested without GPU weights).
// Throws std::runtime_error with an actionable message on any schema/provider/host mismatch.
EngramAdapterManifest validate_rkad_v3_manifest(
    const std::string & manifest_json,
    const EngramConfig & table_cfg,
    const std::string & host_arch,
    int host_n_layers,
    int d_host,
    int hc);

struct EngramHookSpec {
    int   target_host_layer = 0;    // backbone layer index on host model (e.g. Maxwell 1, 44)
    int   src_engram_layer  = 0;    // index into EngramTable's layers (0..n_layers-1)
    int   device            = 0;    // GPU owning target_host_layer
    float scale             = 0.0f; // adapter-calibrated base scale (0.0f = permanently disabled hook)
    float default_delta_clip = 0.0f;// optional RKAD v3 per-hook default delta clip ratio (0 = disabled)

    // Optional RKAD v2/v3 mean-centering vector mu_E[d_engram] subtracted on CPU staging buffer before H2D
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
    float *               h_engram_pin = nullptr; // [max_batch, d_engram]
    float *               d_engram_in  = nullptr; // [max_batch, d_engram]
    EngramRuntimeParams * h_params_pin = nullptr; // [max_batch] per-token runtime controls
    EngramRuntimeParams * d_params     = nullptr; // [max_batch] per-token runtime controls on GPU
    float *               d_mid        = nullptr; // [max_batch, max(rank, 1)]
    float *               d_val        = nullptr; // [max_batch, d_host]
    float *               d_key        = nullptr; // [max_batch, d_host] (if gated)
    float *               d_hmean      = nullptr; // [max_batch, d_host] (for calibration dumps)

    // Dedicated non-blocking H2D stream (explicit sync in prepare_step/prepare_batch_step — zero reliance on legacy default stream)
    cudaStream_t h2d_stream = nullptr;

    // Optional CUDA events for stage-by-stage GPU hook profiling (only when gpu_profile_ == true, non-graph)
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop  = nullptr;
};

// Per-sequence isolated rolling tokenizer & boundary-anchored incremental commit cache.
// Supports exact O(1) MTP backtrack and concurrent multi-sequence batching without clobbering.
struct EngramSeqState {
    std::string          rolling_text;
    std::vector<size_t>  host_byte_ends;       // byte offset in rolling_text after each host token
    std::vector<int32_t> committed_src_tokens; // finalized source-tokenizer prefix tokens
    std::vector<size_t>  committed_src_ends;   // absolute byte end in rolling_text of each committed token
    std::vector<int32_t> src_seq_cache;        // direct token-ID passthrough mode (when no dual tokenizer)

    void clear() {
        rolling_text.clear();
        host_byte_ends.clear();
        committed_src_tokens.clear();
        committed_src_ends.clear();
        src_seq_cache.clear();
    }

    void rollback_to(int start_pos) {
        if (start_pos <= 0) {
            clear();
            return;
        }
        if (start_pos < static_cast<int>(host_byte_ends.size())) {
            const size_t keep_bytes = host_byte_ends[start_pos - 1];
            rolling_text.resize(keep_bytes);
            host_byte_ends.resize(start_pos);
            // Roll back committed source tokens strictly past keep_bytes (with safety margin so boundary re-merges cleanly)
            constexpr size_t kRollbackMargin = 64;
            const size_t safe_keep = (keep_bytes > kRollbackMargin) ? (keep_bytes - kRollbackMargin) : 0;
            while (!committed_src_ends.empty() && committed_src_ends.back() > safe_keep) {
                committed_src_ends.pop_back();
                committed_src_tokens.pop_back();
            }
        }
        if (start_pos < static_cast<int>(src_seq_cache.size())) {
            src_seq_cache.resize(start_pos);
        }
    }
};

// Decoupled, model-agnostic Engram Sidecar implementing `LayerHook`.
// Supports cross-model grafting (e.g. HY4-Flash 59 GiB or DeepSeek-V4.1-Flash 103 GiB Engram -> Maxwell-V1-Flash),
// boundary-anchored incremental dual-tokenizer bridging, RKAD v1/v2/v3 strict manifest validation,
// per-sequence CUDA-Graph-safe runtime controls (scale / gate_bias / delta_clip without graph recapture),
// pinned-RAM async H2D prefetch, and paired calibration / RKCE tail-CE dumps.
class EngramSidecar : public LayerHook {
public:
    EngramSidecar() = default;
    ~EngramSidecar() override;

    EngramSidecar(const EngramSidecar &) = delete;
    EngramSidecar & operator=(const EngramSidecar &) = delete;

    // Initialize sidecar attached to a host model (`n_host_layers`, `d_host`, `hc`, per-layer GPU mapping).
    // `adapter_path`: optional `.rkad` (v1/v2/v3) adapter file; if empty, initializes deterministic low-rank projection.
    // `override_host_layers`: if non-empty, maps Engram layers onto these host layers instead of default mapping.
    // `host_arch`: optional host architecture string (e.g. "maxwell-moe") validated strictly against RKAD v3 manifest.
    void init(const std::string & engram_path,
              const BpeTokenizer * host_tokenizer,
              const std::vector<int> & layer_to_device,
              int d_host,
              int hc,
              int max_batch = 512,
              const std::string & adapter_path = "",
              float global_scale = 0.0f,
              const std::vector<int> & override_host_layers = {},
              const std::string & host_arch = "");

    // Optional: enable binary dump of paired (hook_idx, engram_raw[T, d_engram], host_hidden_mean[T, d_host])
    void enable_calibration_dump(const std::string & out_path, bool dump_delta = false);
    void close_calibration_dump();

    // Optional: enable "RKCE" v1 tail-CE dump at a target host layer (e.g. L44 right before hc_head+output_norm+lm_head).
    // If src_engram_layer >= 0, dumps only that single Engram layer (2x faster & smaller for single-layer training).
    void enable_tail_ce_dump(const std::string & out_path, int target_host_layer, int src_engram_layer = -1);
    void set_tail_ce_targets(const std::vector<int32_t> & targets,
                             const std::vector<int32_t> & host_tokens,
                             int start_pos);
    void close_tail_ce_dump();

    void set_shuffle_salt(uint64_t salt) { table_.set_shuffle_salt(salt); }
    void set_mask_special_tokens(bool enable) { mask_special_tokens_ = enable; }
    bool mask_special_tokens() const { return mask_special_tokens_; }
    bool should_mask_token(int32_t host_tok) const {
        return mask_special_tokens_ && host_tok_ != nullptr && host_tok_->is_special_token(host_tok);
    }
    float resolve_token_scale(int32_t host_tok, float eff_scale) const {
        return should_mask_token(host_tok) ? 0.0f : eff_scale;
    }
    void set_global_scale(float s) { default_params_.scale = s; }
    float global_scale() const { return default_params_.scale; }
    void set_default_params(const EngramRuntimeParams & p) { default_params_ = p; }
    const EngramRuntimeParams & default_params() const { return default_params_; }
    const EngramTable & table() const { return table_; }
    const EngramAdapterManifest & manifest() const { return manifest_; }
    size_t num_hooks() const { return hooks_.size(); }

    void release_sequence(int seq_id) { seq_states_.erase(seq_id); }
    void clear_all_sequences() { seq_states_.clear(); }
    size_t active_sequences() const { return seq_states_.size(); }

    // Boundary-anchored BPE cut helper (exposed for property-based unit tests).
    // Falls back to a valid UTF-8 codepoint boundary when space-less runs exceed `max_uncommitted` bytes.
    static size_t find_safe_bpe_cut(const std::string & text, size_t min_cut, size_t max_cut, size_t max_uncommitted = 1024);

    void set_gpu_profiling(bool enable) { gpu_profile_ = enable; }
    void reset_profile() { prof_ = EngramProfileStats{}; }
    const EngramProfileStats & profile() const { return prof_; }

    // Verifies that the production incremental boundary-anchored commit cache produces
    // 100.00% identical source token IDs and N-gram hash row IDs as full-prefix causal tokenization.
    bool verify_incremental_addressing(const std::vector<int32_t> & host_tokens,
                                       int & out_tokens_checked,
                                       int & out_src_tok_matches,
                                       int & out_hash_col_matches,
                                       int & out_hash_col_total,
                                       bool verbose = false);

    // Prepare a contiguous token slice for sequence `seq_id` using `default_params_` (backward-compatible).
    void prepare_step(const std::vector<int32_t> & host_tokens, int start_pos);

    // Prepare a contiguous token slice for sequence `seq_id` with explicit per-request CUDA-Graph-safe `params`.
    void prepare_step(const std::vector<int32_t> & host_tokens,
                      int start_pos,
                      const EngramRuntimeParams & params,
                      int seq_id = 0);

    // Prepare a multi-sequence continuous-batching step where each slot t in [0..T) has its own
    // `seq_ids[t]`, `host_tokens[t]`, `positions[t]`, and `per_token_params[t]`.
    void prepare_batch_step(const std::vector<int> & seq_ids,
                            const std::vector<int32_t> & host_tokens,
                            const std::vector<int32_t> & positions,
                            const std::vector<EngramRuntimeParams> & per_token_params);

    // LayerHook callbacks invoked inside host model's layer loop (including inside CUDA Graphs!).
    bool hooks_layer(int layer_idx) const override {
        if (layer_idx < 0) return false;
        if (ce_fp_ != nullptr && layer_idx == ce_target_layer_) return true;
        if (dump_fp_ != nullptr && dump_delta_ &&
            static_cast<size_t>(layer_idx) < delta_next_to_hook_.size() &&
            delta_next_to_hook_[layer_idx] >= 0) return true;
        if (static_cast<size_t>(layer_idx) < layer_to_hook_.size()) {
            const int hi = layer_to_hook_[layer_idx];
            if (hi >= 0 && (hooks_[hi].scale != 0.0f || dump_fp_ != nullptr)) return true;
        }
        return false;
    }
    void on_post_layer(int layer_idx, int device, float * d_inpL, int n_tokens, int n_embd, int hc, cudaStream_t stream) override;

private:
    void free_gpu_buffers() noexcept;
    void load_rkad_adapter(const std::string & adapter_path, const std::vector<int> & layer_to_device);
    void init_zero_shot_projection(const std::vector<int> & layer_to_device,
                                   const std::vector<int> & override_host_layers);

    // Advance `st` by `host_tokens` starting at `start_pos` using the incremental commit cache,
    // outputting a compact source token window and per-host-token index inside that window.
    void advance_seq_state(EngramSeqState & st,
                           const std::vector<int32_t> & host_tokens,
                           int start_pos,
                           std::vector<int32_t> & out_src_window,
                           std::vector<int32_t> & out_aligned_pos) const;

    EngramTable             table_;
    const BpeTokenizer *    host_tok_       = nullptr;
    std::string             host_arch_;
    int                     d_host_         = 0;
    int                     hc_             = 4;
    int                     max_batch_      = 512;
    EngramRuntimeParams     default_params_{};
    EngramAdapterManifest   manifest_{};
    bool                    mask_special_tokens_ = true;
    bool                    gpu_profile_    = false;
    EngramProfileStats      prof_{};

    std::vector<EngramHookSpec> hooks_;
    std::vector<int>            layer_to_hook_; // [n_host_layers] -> index into hooks_ or -1

    // Per-sequence rolling tokenizer & incremental commit cache states (isolated by seq_id)
    std::unordered_map<int, EngramSeqState> seq_states_;

    // Reusable per-step CPU scratch buffers (avoid heap reallocations inside token loop)
    std::vector<int32_t>              scratch_src_win_;
    std::vector<int32_t>              scratch_aligned_pos_;
    std::vector<int32_t>              scratch_single_ids_;
    std::vector<float>                scratch_single_valid_;
    std::vector<std::vector<int32_t>> scratch_hook_rows_;
    std::vector<std::vector<float>>   scratch_hook_valid_;

    // Optional calibration dump file ("RKPR" v1)
    FILE *               dump_fp_       = nullptr;
    bool                 dump_delta_    = false;
    uint64_t             dump_records_  = 0;
    std::vector<float>   dump_h_tmp_;
    std::vector<int>     delta_next_to_hook_; // [n_host_layers] -> hook idx waiting for L+1 delta, or -1
    std::vector<std::vector<float>> dump_h_prev_; // [n_hooks] cached h_L for L+1 delta

    // Optional tail-CE dump file ("RKCE" v1)
    FILE *               ce_fp_            = nullptr;
    int                  ce_target_layer_  = -1;
    int                  ce_src_layer_     = -1; // -1 = all Engram layers, >=0 = single Engram layer
    uint64_t             ce_tokens_        = 0;
    std::vector<int32_t> ce_cur_targets_;
    std::vector<std::vector<uint16_t>> ce_cur_e_f16_; // [n_eng_layers][T * d_engram]
    std::vector<float>   ce_h_tmp_;
    EngramSeqState       ce_seq_state_;
};

} // namespace rookery
