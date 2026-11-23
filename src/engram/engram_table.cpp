#include "engram/engram_table.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <sys/mman.h>

namespace rookery {

namespace {

// Exact IEEE-754 FP8 E4M3 (1 sign, 4 exp bias=7, 3 mantissa) -> F32 LUT matching ggml_fp8_e4m3_to_fp32.
std::array<float, 256> build_f8_e4m3_lut() {
    std::array<float, 256> lut{};
    for (int i = 0; i < 256; ++i) {
        const uint32_t s = (i >> 7) & 1u;
        const uint32_t e = (i >> 3) & 0xFu;
        const uint32_t m = i & 0x7u;
        float val = 0.0f;
        if (e == 0) {
            // Subnormal: (-1)^s * 2^(-6) * (m / 8)
            val = std::ldexp(static_cast<float>(m), -9);
        } else if (e == 15 && m == 7) {
            val = 0.0f; // NaN -> safe 0 for lookup stability
        } else {
            // Normal (including e=15, m=0..6 in E4M3FN): (-1)^s * 2^(e-7) * (1 + m/8)
            val = std::ldexp(1.0f + static_cast<float>(m) * 0.125f, static_cast<int>(e) - 7);
        }
        lut[i] = s ? -val : val;
    }
    return lut;
}

const std::array<float, 256> kF8E4M3Lut = build_f8_e4m3_lut();

// Exact ggml IQ4_NL non-linear 4-bit quantization table.
constexpr int8_t kIq4NlValues[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            float v = std::ldexp(static_cast<float>(mant), -24);
            return sign ? -v : v;
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline float bf16_to_fp32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

template <typename T>
std::vector<T> read_int_array(const GgufFile & g, const std::string & key) {
    std::vector<T> out;
    if (!g.has_kv(key)) return out;
    const auto & kv = g.kv(key);
    if (auto * v = std::get_if<std::vector<uint64_t>>(&kv.data)) {
        out.reserve(v->size());
        for (uint64_t x : *v) out.push_back(static_cast<T>(x));
    } else if (auto * v = std::get_if<std::vector<int64_t>>(&kv.data)) {
        out.reserve(v->size());
        for (int64_t x : *v) out.push_back(static_cast<T>(x));
    }
    return out;
}

// Given any shard path `foo-0000k-of-0000N.gguf`, return all sibling shard paths in sorted order.
std::vector<std::string> discover_shards(const std::string & path) {
    namespace fs = std::filesystem;
    const std::string fname = fs::path(path).filename().string();
    const size_t of_pos = fname.rfind("-of-");
    if (of_pos == std::string::npos || of_pos < 6 || fname[of_pos - 6] != '-') {
        return {path};
    }
    const std::string prefix = fname.substr(0, of_pos - 6);
    const std::string suffix = fname.substr(of_pos); // "-of-0000N.gguf"
    const fs::path dir = fs::path(path).parent_path();

    std::vector<std::string> shards;
    for (const auto & ent : fs::directory_iterator(dir.empty() ? fs::path(".") : dir)) {
        if (!ent.is_regular_file()) continue;
        const std::string n = ent.path().filename().string();
        if (n.size() == fname.size() &&
            n.compare(0, prefix.size(), prefix) == 0 &&
            n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0) {
            shards.push_back(ent.path().string());
        }
    }
    std::sort(shards.begin(), shards.end());
    if (shards.empty()) shards.push_back(path);
    return shards;
}

} // namespace

void EngramTable::open(const std::string & path) {
    const auto shards = discover_shards(path);

    // Shard 0 always holds model KV metadata & tokenizer
    GgufFile meta = GgufFile::open(shards.front());
    cfg_.arch = meta.get_str("general.architecture", "engram");
    const std::string pfx = cfg_.arch + ".";

    // Load source tokenizer from metadata shard if present
    if (meta.has_kv("tokenizer.ggml.tokens")) {
        tokenizer_.load(meta);
        has_tokenizer_ = true;
    }

    // Parse Engram / PLE metadata
    cfg_.source_layers     = read_int_array<int32_t>(meta, pfx + "ple.layers");
    cfg_.layer_multipliers = read_int_array<uint64_t>(meta, pfx + "ple.layer_multipliers");
    cfg_.head_offsets      = read_int_array<uint32_t>(meta, pfx + "ple.head_offsets");
    cfg_.head_vocab_sizes  = read_int_array<uint32_t>(meta, pfx + "ple.head_vocab_sizes");
    cfg_.token_map         = read_int_array<int64_t>(meta, pfx + "engram.token_map");

    if (!cfg_.source_layers.empty() && !cfg_.layer_multipliers.empty()) {
        cfg_.mode            = EngramMode::NGRAM_HASH;
        cfg_.n_layers        = static_cast<int>(cfg_.source_layers.size());
        cfg_.ngram_size      = static_cast<int>(meta.get_u64(pfx + "ple.ngram_size", 3));
        cfg_.heads_per_ngram = static_cast<int>(meta.get_u64(pfx + "ple.heads_per_ngram", 8));
        cfg_.n_cols          = (cfg_.ngram_size - 1) * cfg_.heads_per_ngram;
        cfg_.head_dim        = static_cast<int>(meta.get_u64(pfx + "embedding_length_per_layer_input", 96));
        cfg_.pad_token_id    = static_cast<int32_t>(meta.get_u64(pfx + "engram.pad_token_id", 0));
    } else {
        // Fallback: Unigram PLE (e.g. Gemma 3n/4 E2B/E4B or harvested standalone table)
        cfg_.mode            = EngramMode::UNIGRAM_PLE;
        cfg_.n_layers        = static_cast<int>(meta.get_u64(pfx + "block_count", 1));
        cfg_.ngram_size      = 1;
        cfg_.heads_per_ngram = 1;
        cfg_.n_cols          = 1;
        cfg_.head_dim        = static_cast<int>(meta.get_u64(pfx + "embedding_length_per_layer_input", 256));
    }

    // Locate which shard holds `per_layer_token_embd.weight` (check last shard first since HY4 & DeepSeek put it there!)
    const GgufTensorInfo * tbl_info = nullptr;
    std::string found_shard;
    for (auto it = shards.rbegin(); it != shards.rend(); ++it) {
        GgufFile cand = (*it == shards.front()) ? std::move(meta) : GgufFile::open(*it);
        const GgufTensorInfo * ti = cand.find_tensor("per_layer_token_embd.weight");
        if (!ti) ti = cand.find_tensor("engram_table.weight");
        if (ti) {
            table_gguf_   = std::move(cand);
            tbl_info      = table_gguf_.find_tensor(ti->name);
            found_shard   = *it;
            break;
        }
    }
    if (!tbl_info) {
        throw std::runtime_error("EngramTable::open: no per_layer_token_embd.weight tensor found in " + path);
    }

    cfg_.table_path  = found_shard;
    cfg_.table_dtype = tbl_info->dtype;
    cfg_.head_dim    = static_cast<int>(tbl_info->ne[0]);
    cfg_.total_rows  = tbl_info->ne[1] * tbl_info->ne[2] * tbl_info->ne[3];
    cfg_.row_bytes   = tensor_nbytes(cfg_.table_dtype, cfg_.head_dim, 1);

    table_base_   = static_cast<const uint8_t *>(table_gguf_.tensor_data(*tbl_info));
    table_nbytes_ = tbl_info->nbytes;

    // Advise OS kernel for random row access over huge (59-103 GiB) mmap'd tables
    const size_t page_sz = 4096;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(table_base_) & ~(page_sz - 1);
    const size_t len = table_nbytes_ + (reinterpret_cast<uintptr_t>(table_base_) - addr);
    madvise(reinterpret_cast<void *>(addr), len, MADV_RANDOM);

    std::printf("Loaded EngramTable [%s, mode=%s]: %s\n"
                "  layers=%d, ngram=%d, heads=%d, head_dim=%d (engram_dim=%d), rows=%llu, dtype=%s (%.2f GiB mmap)\n",
                cfg_.arch.c_str(),
                cfg_.mode == EngramMode::NGRAM_HASH ? "ngram_hash" : "unigram_ple",
                cfg_.table_path.c_str(),
                cfg_.n_layers, cfg_.ngram_size, cfg_.n_cols, cfg_.head_dim, cfg_.engram_dim(),
                static_cast<unsigned long long>(cfg_.total_rows),
                dtype_traits(cfg_.table_dtype).name,
                static_cast<double>(table_nbytes_) / (1024.0 * 1024.0 * 1024.0));
}

void EngramTable::hash_tokens(int engram_layer_idx,
                              const std::vector<int32_t> & seq_tokens,
                              int start_idx,
                              int n_toks,
                              std::vector<int32_t> & out_row_ids,
                              std::vector<float> & out_valid) const {
    const int n_cols = cfg_.n_cols;
    out_row_ids.resize(static_cast<size_t>(n_toks) * n_cols);
    out_valid.resize(static_cast<size_t>(n_toks) * n_cols);

    if (cfg_.mode == EngramMode::UNIGRAM_PLE) {
        for (int i = 0; i < n_toks; ++i) {
            const int idx = start_idx + i;
            int32_t tok = (idx >= 0 && idx < static_cast<int>(seq_tokens.size())) ? seq_tokens[idx] : 0;
            uint64_t row = static_cast<uint64_t>(std::max<int32_t>(0, tok)) * cfg_.n_layers + engram_layer_idx;
            if (row >= cfg_.total_rows) row %= cfg_.total_rows;
            out_row_ids[i] = static_cast<int32_t>(row);
            out_valid[i]   = 1.0f;
        }
        return;
    }

    const int n_gram = cfg_.ngram_size;
    const int n_head = cfg_.heads_per_ngram;
    const int e = std::clamp(engram_layer_idx, 0, cfg_.n_layers - 1);

    const uint64_t * mult   = cfg_.layer_multipliers.data() + static_cast<size_t>(e) * n_gram;
    const uint32_t * primes = cfg_.head_vocab_sizes.data()  + static_cast<size_t>(e) * n_cols;
    const uint32_t * offs   = cfg_.head_offsets.data()      + static_cast<size_t>(e) * n_cols;
    const int64_t    pad    = (cfg_.pad_token_id >= 0 && static_cast<size_t>(cfg_.pad_token_id) < cfg_.token_map.size())
                                  ? cfg_.token_map[cfg_.pad_token_id] : 0;

    auto cid = [&](int32_t t) -> int64_t {
        if (t < 0 || static_cast<size_t>(t) >= cfg_.token_map.size()) return -1;
        return cfg_.token_map[t];
    };

    const bool cascade_block = (cfg_.arch == "deepseek41");
    std::vector<uint64_t> toks(n_gram);

    for (int i = 0; i < n_toks; ++i) {
        const int pos = start_idx + i;
        int64_t c0 = (pos >= 0 && pos < static_cast<int>(seq_tokens.size())) ? cid(seq_tokens[pos]) : -1;
        bool blocked = (c0 < 0);
        const bool dead = blocked;
        toks[0] = blocked ? static_cast<uint64_t>(pad) : static_cast<uint64_t>(c0);

        for (int s = 1; s < n_gram; ++s) {
            const int p = pos - s;
            int64_t cs = (p >= 0 && p < static_cast<int>(seq_tokens.size())) ? cid(seq_tokens[p]) : -1;
            if (cascade_block) {
                blocked = blocked || (cs < 0);
                toks[s] = blocked ? static_cast<uint64_t>(pad) : static_cast<uint64_t>(cs);
            } else {
                toks[s] = (cs < 0) ? static_cast<uint64_t>(pad) : static_cast<uint64_t>(cs);
            }
        }

        uint64_t rolling = (toks[0] * mult[0]) ^ shuffle_salt_;
        for (int o = 1; o < n_gram; ++o) {
            rolling ^= toks[o] * mult[o];
            for (int h = 0; h < n_head; ++h) {
                const int col = (o - 1) * n_head + h;
                const uint64_t row = (rolling % primes[col]) + offs[col];
                out_row_ids[static_cast<size_t>(i) * n_cols + col] = static_cast<int32_t>(row);
                out_valid[static_cast<size_t>(i) * n_cols + col]   = dead ? 0.0f : 1.0f;
            }
        }
    }
}

void EngramTable::prefetch_rows(const int32_t * row_ids, int n_rows) const {
    if (!table_base_) return;
    for (int i = 0; i < n_rows; ++i) {
        const uint64_t r = static_cast<uint64_t>(row_ids[i]);
        if (r >= cfg_.total_rows) continue;
        const uint8_t * ptr = table_base_ + r * cfg_.row_bytes;
        __builtin_prefetch(ptr, 0, 1);
    }
}

void EngramTable::gather_dequant_f32(const int32_t * row_ids,
                                     const float * valid_mask,
                                     int n_toks,
                                     float * out_f32) const {
    const int n_cols   = cfg_.n_cols;
    const int head_dim = cfg_.head_dim;
    const int n_rows   = n_toks * n_cols;

    for (int r_idx = 0; r_idx < n_rows; ++r_idx) {
        float * dst = out_f32 + static_cast<size_t>(r_idx) * head_dim;
        const float v_scale = valid_mask ? valid_mask[r_idx] : 1.0f;
        const uint64_t row = static_cast<uint64_t>(row_ids[r_idx]);

        if (v_scale == 0.0f || row >= cfg_.total_rows) {
            std::memset(dst, 0, static_cast<size_t>(head_dim) * sizeof(float));
            continue;
        }

        const uint8_t * src = table_base_ + row * cfg_.row_bytes;
        switch (cfg_.table_dtype) {
            case DType::F8_E4M3: {
                for (int d = 0; d < head_dim; ++d) {
                    dst[d] = kF8E4M3Lut[src[d]];
                }
                break;
            }
            case DType::IQ4_NL: {
                // 32 elems per block: 2-byte fp16 scale `d` + 16 bytes packed 4-bit nibbles
                // Low nibbles -> elems 0..15, High nibbles -> elems 16..31 (exact ggml block_iq4_nl layout)
                const int nb = head_dim / 32;
                for (int b = 0; b < nb; ++b) {
                    const uint8_t * blk = src + b * 18;
                    uint16_t dh;
                    std::memcpy(&dh, blk, 2);
                    const float d = fp16_to_fp32(dh);
                    const uint8_t * qs = blk + 2;
                    float * out = dst + b * 32;
                    for (int j = 0; j < 16; ++j) {
                        out[j]      = d * static_cast<float>(kIq4NlValues[qs[j] & 0x0F]);
                        out[j + 16] = d * static_cast<float>(kIq4NlValues[qs[j] >> 4]);
                    }
                }
                break;
            }
            case DType::Q8_0: {
                const int nb = head_dim / 32;
                for (int b = 0; b < nb; ++b) {
                    const auto * blk = reinterpret_cast<const BlockQ8_0 *>(src + b * sizeof(BlockQ8_0));
                    const float d = fp16_to_fp32(blk->d);
                    float * out = dst + b * 32;
                    for (int j = 0; j < 32; ++j) out[j] = d * static_cast<float>(blk->qs[j]);
                }
                break;
            }
            case DType::F16: {
                const auto * s16 = reinterpret_cast<const uint16_t *>(src);
                for (int d = 0; d < head_dim; ++d) dst[d] = fp16_to_fp32(s16[d]);
                break;
            }
            case DType::BF16: {
                const auto * s16 = reinterpret_cast<const uint16_t *>(src);
                for (int d = 0; d < head_dim; ++d) dst[d] = bf16_to_fp32(s16[d]);
                break;
            }
            case DType::F32: {
                std::memcpy(dst, src, static_cast<size_t>(head_dim) * sizeof(float));
                break;
            }
            default:
                throw std::runtime_error("EngramTable::gather_dequant_f32: unsupported table dtype " +
                                         std::string(dtype_traits(cfg_.table_dtype).name));
        }

        // Per-head RMSNorm (matches HY4's engram_lookup_norm and puts F8_E4M3 [RMS~165], IQ4_NL [RMS~0.8],
        // and Gemma PLE all on the exact same unit-RMS manifold before adapter projection!)
        float ss = 0.0f;
        for (int d = 0; d < head_dim; ++d) ss += dst[d] * dst[d];
        const float inv_rms = 1.0f / std::sqrt(ss / static_cast<float>(head_dim) + 1e-6f);
        for (int d = 0; d < head_dim; ++d) dst[d] *= inv_rms;
    }
}

} // namespace rookery
