#pragma once

#include "core/types.h"
#include "gguf/reader.h"
#include "tokenizer/bpe.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rookery {

enum class EngramMode {
    NGRAM_HASH,   // DeepSeek-V4.1-Flash / HY4-Flash multi-head N-gram rolling XOR hash
    UNIGRAM_PLE,  // Gemma 3n/4 E2B/E4B or direct token-ID per-layer embedding lookup
};

// Per-token / per-sequence CUDA-Graph-safe runtime controls for Engram injection.
// Stored in fixed-address pinned + device buffers so requests can independently scale,
// disable (scale=0.0 -> exact no-op), shift gate selectivity, or clip delta norm WITHOUT
// invalidating or recapturing CUDA Graphs!
struct EngramRuntimeParams {
    float scale            = 1.0f; // multiplier on adapter hook scale (0.0f = exact no-op)
    float gate_bias        = 0.0f; // additive logit bias inside sigmoid gate
    float delta_clip_ratio = 0.0f; // max ||g*v||_rms / ||h||_rms cap (0.0f = disabled)
    float _pad             = 0.0f;
};

struct EngramConfig {
    EngramMode  mode            = EngramMode::NGRAM_HASH;
    std::string arch;
    std::string table_path;
    DType       table_dtype     = DType::F8_E4M3;

    int         n_layers        = 0;   // number of Engram layers in table (e.g. 4 for HY4, 2 for DeepSeek)
    int         ngram_size      = 3;   // max N-gram order (3 = 2,3-gram for HY4; 4 = 2,3,4-gram for DeepSeek)
    int         heads_per_ngram = 8;
    int         n_cols          = 16;  // (ngram_size - 1) * heads_per_ngram (1 for UNIGRAM_PLE)
    int         head_dim        = 96;  // per-head embedding dim (96 for HY4, 256 for DeepSeek)
    uint64_t    total_rows      = 0;
    size_t      row_bytes       = 0;
    uint64_t    fingerprint     = 0;   // FNV-1a 64-bit contract fingerprint over addressing spec + token_map

    int32_t     pad_token_id    = 0;
    std::vector<int32_t>  source_layers;     // e.g. [1, 11, 21, 31] or [1, 14]
    std::vector<uint64_t> layer_multipliers; // [n_layers * ngram_size]
    std::vector<uint32_t> head_offsets;      // [n_layers * n_cols]
    std::vector<uint32_t> head_vocab_sizes;  // [n_layers * n_cols]
    std::vector<int64_t>  token_map;         // [vocab_size] -> compressed token ID (or -1 if dead)

    int engram_dim() const { return n_cols * head_dim; }
    std::string fingerprint_hex() const;
};

// Zero-copy mmap loader for standalone or multi-shard Engram / PLE tables (59 GiB HY4, 103 GiB DeepSeek, Gemma PLE).
// Keeps table purely in OS page cache / NVMe (ZERO GPU VRAM, ZERO startup copy).
class EngramTable {
public:
    EngramTable() = default;

    // Opens a multi-shard GGUF (shard 1 or any shard) or standalone Engram/PLE GGUF.
    void open(const std::string & path);

    const EngramConfig & cfg() const { return cfg_; }
    uint64_t fingerprint() const { return cfg_.fingerprint; }
    std::string fingerprint_hex() const { return cfg_.fingerprint_hex(); }
    bool has_tokenizer() const { return has_tokenizer_; }
    const BpeTokenizer & tokenizer() const { return tokenizer_; }
    void set_shuffle_salt(uint64_t salt) { shuffle_salt_ = salt; }
    uint64_t shuffle_salt() const { return shuffle_salt_; }

    // Exact N-gram hash matching hy_v4_flash / deepseek41 (or direct row lookup for UNIGRAM_PLE).
    // `seq_tokens` is the full (or window) source-tokenizer sequence; hashes positions [start_idx .. start_idx + n_toks).
    // Writes `n_toks * cfg_.n_cols` row indices into `out_row_ids`, and `n_toks * cfg_.n_cols` validity flags into `out_valid`.
    void hash_tokens(int engram_layer_idx,
                     const std::vector<int32_t> & seq_tokens,
                     int start_idx,
                     int n_toks,
                     std::vector<int32_t> & out_row_ids,
                     std::vector<float> & out_valid) const;

    // Non-blocking OS/CPU prefetch of table rows (overlaps NVMe/RAM latency with early GPU layers).
    void prefetch_rows(const int32_t * row_ids, int n_rows) const;

    // Gathers `n_toks * cfg_.n_cols` rows from mmap'd table and dequantizes into `out_f32[n_toks, cfg_.engram_dim()]`.
    // Supports F8_E4M3 (HY4), IQ4_NL (DeepSeek), Q8_0, F16, BF16, F32 natively.
    void gather_dequant_f32(const int32_t * row_ids,
                            const float * valid_mask,
                            int n_toks,
                            float * out_f32) const;

private:
    EngramConfig     cfg_;
    GgufFile         table_gguf_;
    const uint8_t *  table_base_    = nullptr;
    size_t           table_nbytes_  = 0;
    BpeTokenizer     tokenizer_;
    bool             has_tokenizer_ = false;
    uint64_t         shuffle_salt_  = 0;
};

} // namespace rookery
