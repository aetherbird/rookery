#include "engram/engram_sidecar.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

namespace rookery {

namespace {

inline void cuda_ok(cudaError_t err, const char * msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

template <typename T>
T * dev_alloc(int dev, size_t n) {
    cuda_ok(cudaSetDevice(dev), "engram dev_alloc setDevice");
    T * p = nullptr;
    cuda_ok(cudaMalloc(&p, n * sizeof(T)), "engram cudaMalloc");
    return p;
}

template <typename T>
T * pin_alloc(size_t n) {
    T * p = nullptr;
    cuda_ok(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocDefault), "engram cudaHostAlloc");
    return p;
}

inline uint16_t f32_to_f16_bits(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        uint32_t t = 14 - exp;
        uint32_t half_m = (mant + (1u << (t - 1))) >> t;
        return static_cast<uint16_t>(sign | half_m);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7BFFu);
    uint32_t rounded = (mant + 0x00000FFFu + ((mant >> 13) & 1u));
    if (rounded & 0x00800000u) { rounded = 0; ++exp; if (exp >= 31) return static_cast<uint16_t>(sign | 0x7BFFu); }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (rounded >> 13));
}

inline bool is_ws_char(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

} // namespace

// Find the LATEST BPE pre-tokenizer-safe cut offset in [min_cut, max_cut] so encoding text[cut:]
// produces exact suffix token IDs identical to encoding full text[0:].
// Scanning backwards keeps the active tail as short as possible for incremental caching!
// If no whitespace/newline boundary exists and the uncommitted tail exceeds `max_uncommitted` bytes
// (e.g. long CJK runs, base64, minified JSON), falls back to the latest valid UTF-8 codepoint boundary
// so memory & retokenization stay strictly O(1) bounded.
size_t EngramSidecar::find_safe_bpe_cut(const std::string & text, size_t min_cut, size_t max_cut, size_t max_uncommitted) {
    if (min_cut == 0 || min_cut > max_cut || max_cut >= text.size()) return 0;

    // Pass 1: clean start-of-line right after '\n' (not followed by whitespace)
    for (size_t p = max_cut; p >= min_cut; --p) {
        if (text[p - 1] == '\n' && !is_ws_char(text[p])) {
            return p;
        }
        if (p == min_cut) break;
    }
    // Pass 2: single leading space before a non-whitespace token (GPT-2/Llama3/DeepSeek BPE attaches ' ' to next word)
    for (size_t p = max_cut; p >= min_cut; --p) {
        if (text[p] == ' ' && !is_ws_char(text[p - 1]) && (p + 1 < text.size()) && !is_ws_char(text[p + 1])) {
            return p;
        }
        if (p == min_cut) break;
    }
    // Pass 3 (bounded-tail fallback for space-less text): if uncommitted suffix exceeds max_uncommitted bytes,
    // cut at the latest valid UTF-8 codepoint boundary (not a continuation byte 10xxxxxx).
    const size_t committed_bytes = min_cut - 1;
    if (max_uncommitted > 0 && text.size() > committed_bytes + max_uncommitted) {
        for (size_t p = max_cut; p >= min_cut; --p) {
            const unsigned char b = static_cast<unsigned char>(text[p]);
            if ((b & 0xC0u) != 0x80u) {
                return p;
            }
            if (p == min_cut) break;
        }
    }
    return 0;
}

EngramAdapterManifest validate_rkad_v3_manifest(
    const std::string & manifest_json,
    const EngramConfig & table_cfg,
    const std::string & host_arch,
    int host_n_layers,
    int d_host,
    int hc) {

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(manifest_json);
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("RKAD v3 invalid manifest JSON: ") + e.what());
    }
    if (!j.is_object()) {
        throw std::runtime_error("RKAD v3 manifest must be a JSON object");
    }

    EngramAdapterManifest m;
    m.format_version       = 3;
    m.schema               = j.value("schema", "");
    m.provider_arch        = j.value("provider_arch", "");
    m.provider_mode        = j.value("provider_mode", "");
    m.provider_ngram_size  = j.value("provider_ngram_size", 0);
    m.provider_n_cols      = j.value("provider_n_cols", 0);
    m.provider_head_dim    = j.value("provider_head_dim", 0);
    m.provider_fingerprint = j.value("provider_fingerprint", "");
    m.host_arch            = j.value("host_arch", "");
    m.host_n_layers        = j.value("host_n_layers", 0);
    m.host_d_embd          = j.value("host_d_embd", 0);
    m.host_hc_mult         = j.value("host_hc_mult", 0);
    m.calibration          = j.value("calibration", "");

    if (m.schema.rfind("rookery.engram.adapter.v3", 0) != 0) {
        throw std::runtime_error("RKAD v3 unsupported manifest schema: '" + m.schema + "'");
    }
    if (!m.provider_arch.empty() && !table_cfg.arch.empty() && m.provider_arch != table_cfg.arch) {
        throw std::runtime_error("RKAD v3 provider_arch mismatch: adapter expects '" + m.provider_arch +
                                 "', loaded Engram table is '" + table_cfg.arch + "'");
    }
    const std::string expected_mode = (table_cfg.mode == EngramMode::NGRAM_HASH) ? "ngram_hash" : "unigram_ple";
    if (!m.provider_mode.empty() && m.provider_mode != expected_mode) {
        throw std::runtime_error("RKAD v3 provider_mode mismatch: adapter expects '" + m.provider_mode +
                                 "', loaded Engram table is '" + expected_mode + "'");
    }
    if (m.provider_ngram_size > 0 && m.provider_ngram_size != table_cfg.ngram_size) {
        throw std::runtime_error("RKAD v3 provider_ngram_size mismatch: adapter=" + std::to_string(m.provider_ngram_size) +
                                 ", table=" + std::to_string(table_cfg.ngram_size));
    }
    if (m.provider_n_cols > 0 && m.provider_n_cols != table_cfg.n_cols) {
        throw std::runtime_error("RKAD v3 provider_n_cols mismatch: adapter=" + std::to_string(m.provider_n_cols) +
                                 ", table=" + std::to_string(table_cfg.n_cols));
    }
    if (m.provider_head_dim > 0 && m.provider_head_dim != table_cfg.head_dim) {
        throw std::runtime_error("RKAD v3 provider_head_dim mismatch: adapter=" + std::to_string(m.provider_head_dim) +
                                 ", table=" + std::to_string(table_cfg.head_dim));
    }
    if (!m.provider_fingerprint.empty() && table_cfg.fingerprint != 0) {
        const std::string actual_fp = table_cfg.fingerprint_hex();
        if (m.provider_fingerprint != actual_fp) {
            throw std::runtime_error("RKAD v3 provider_fingerprint mismatch: adapter expects " + m.provider_fingerprint +
                                     ", loaded Engram table has " + actual_fp +
                                     " (incompatible hash spec, tokenizer, or table dimensions)");
        }
    }
    if (!m.host_arch.empty() && !host_arch.empty() && m.host_arch != host_arch) {
        throw std::runtime_error("RKAD v3 host_arch mismatch: adapter expects '" + m.host_arch +
                                 "', host model is '" + host_arch + "'");
    }
    if (m.host_n_layers > 0 && host_n_layers > 0 && m.host_n_layers != host_n_layers) {
        throw std::runtime_error("RKAD v3 host_n_layers mismatch: adapter expects " + std::to_string(m.host_n_layers) +
                                 ", host model has " + std::to_string(host_n_layers));
    }
    if (m.host_d_embd > 0 && m.host_d_embd != d_host) {
        throw std::runtime_error("RKAD v3 host_d_embd mismatch: adapter expects " + std::to_string(m.host_d_embd) +
                                 ", host model has " + std::to_string(d_host));
    }
    if (m.host_hc_mult > 0 && m.host_hc_mult != hc) {
        throw std::runtime_error("RKAD v3 host_hc_mult mismatch: adapter expects " + std::to_string(m.host_hc_mult) +
                                 ", host model has " + std::to_string(hc));
    }
    return m;
}

EngramSidecar::~EngramSidecar() {
    close_calibration_dump();
    close_tail_ce_dump();
    free_gpu_buffers();
}

void EngramSidecar::free_gpu_buffers() noexcept {
    for (auto & hk : hooks_) {
        cudaSetDevice(hk.device);
        auto freep = [](void * p) { if (p) cudaFree(p); };
        freep(hk.d_Av);
        freep(hk.d_Bv);
        freep(hk.d_Ak);
        freep(hk.d_Bk);
        freep(hk.d_gatew);
        freep(hk.d_engram_in);
        freep(hk.d_params);
        freep(hk.d_mid);
        freep(hk.d_val);
        freep(hk.d_key);
        freep(hk.d_hmean);
        if (hk.h_engram_pin) cudaFreeHost(hk.h_engram_pin);
        if (hk.h_params_pin) cudaFreeHost(hk.h_params_pin);
        if (hk.h2d_stream)   cudaStreamDestroy(hk.h2d_stream);
        if (hk.ev_start) cudaEventDestroy(hk.ev_start);
        if (hk.ev_stop)  cudaEventDestroy(hk.ev_stop);
    }
    hooks_.clear();
    layer_to_hook_.clear();
}

void EngramSidecar::init(const std::string & engram_path,
                         const BpeTokenizer * host_tokenizer,
                         const std::vector<int> & layer_to_device,
                         int d_host,
                         int hc,
                         int max_batch,
                         const std::string & adapter_path,
                         float global_scale,
                         const std::vector<int> & override_host_layers,
                         const std::string & host_arch) {
    free_gpu_buffers();

    host_tok_             = host_tokenizer;
    host_arch_            = host_arch;
    d_host_               = d_host;
    hc_                   = hc;
    max_batch_            = max_batch;
    default_params_       = EngramRuntimeParams{};
    default_params_.scale = global_scale;
    manifest_             = EngramAdapterManifest{};
    seq_states_.clear();

    table_.open(engram_path);
    layer_to_hook_.assign(layer_to_device.size(), -1);

    if (!adapter_path.empty()) {
        load_rkad_adapter(adapter_path, layer_to_device);
    } else {
        init_zero_shot_projection(layer_to_device, override_host_layers);
    }

    // Allocate fixed-address pinned host + device staging buffers (100% CUDA-Graph compatible!)
    const int d_eng = table_.cfg().engram_dim();
    for (size_t i = 0; i < hooks_.size(); ++i) {
        auto & hk = hooks_[i];
        hk.h_engram_pin = pin_alloc<float>(static_cast<size_t>(max_batch_) * d_eng);
        hk.d_engram_in  = dev_alloc<float>(hk.device, static_cast<size_t>(max_batch_) * d_eng);
        hk.h_params_pin = pin_alloc<EngramRuntimeParams>(static_cast<size_t>(max_batch_));
        hk.d_params     = dev_alloc<EngramRuntimeParams>(hk.device, static_cast<size_t>(max_batch_));
        hk.d_mid        = dev_alloc<float>(hk.device, static_cast<size_t>(max_batch_) * std::max(hk.rank, 1));
        hk.d_val        = dev_alloc<float>(hk.device, static_cast<size_t>(max_batch_) * d_host_);
        if (hk.d_Ak != nullptr) {
            hk.d_key = dev_alloc<float>(hk.device, static_cast<size_t>(max_batch_) * d_host_);
        }
        hk.d_hmean = dev_alloc<float>(hk.device, static_cast<size_t>(max_batch_) * d_host_);
        cuda_ok(cudaSetDevice(hk.device), "engram memset setDevice");
        cuda_ok(cudaStreamCreateWithFlags(&hk.h2d_stream, cudaStreamNonBlocking), "engram h2d stream");
        cuda_ok(cudaEventCreate(&hk.ev_start), "engram event start");
        cuda_ok(cudaEventCreate(&hk.ev_stop),  "engram event stop");
        cuda_ok(cudaMemset(hk.d_engram_in, 0, static_cast<size_t>(max_batch_) * d_eng * sizeof(float)), "engram memset");
        for (int t = 0; t < max_batch_; ++t) {
            hk.h_params_pin[t] = EngramRuntimeParams{0.0f, 0.0f, 0.0f, 0.0f};
        }
        cuda_ok(cudaMemcpy(hk.d_params, hk.h_params_pin,
                           static_cast<size_t>(max_batch_) * sizeof(EngramRuntimeParams),
                           cudaMemcpyHostToDevice), "engram params init");
        layer_to_hook_[hk.target_host_layer] = static_cast<int>(i);
    }

    scratch_hook_rows_.resize(hooks_.size());
    scratch_hook_valid_.resize(hooks_.size());

    std::printf("EngramSidecar attached: %zu hook(s), d_engram=%d -> d_host=%d, global_scale=%.4f, mode=%s (RKAD v%u%s)\n",
                hooks_.size(), d_eng, d_host_, default_params_.scale,
                adapter_path.empty() ? "zero-shot-ortho" : adapter_path.c_str(),
                manifest_.format_version,
                manifest_.calibration.empty() ? "" : (", " + manifest_.calibration).c_str());
    for (size_t i = 0; i < hooks_.size(); ++i) {
        std::printf("  hook[%zu]: engram_layer=%d -> host_layer=%d (GPU%d, rank=%d, gated=%s, centered=%s, scale=%.4f, clip=%.3f)\n",
                    i, hooks_[i].src_engram_layer, hooks_[i].target_host_layer, hooks_[i].device,
                    hooks_[i].rank, hooks_[i].d_Ak ? "yes" : "no",
                    hooks_[i].h_mu_e.empty() ? "no" : "yes",
                    hooks_[i].scale, hooks_[i].default_delta_clip);
    }
}

void EngramSidecar::init_zero_shot_projection(const std::vector<int> & layer_to_device,
                                              const std::vector<int> & override_host_layers) {
    const int n_host = static_cast<int>(layer_to_device.size());
    const int n_eng  = std::max(1, table_.cfg().n_layers);
    const int d_eng  = table_.cfg().engram_dim();
    constexpr int kDefaultRank = 128;

    std::vector<int> targets = override_host_layers;
    if (targets.empty()) {
        if (!table_.cfg().source_layers.empty()) {
            for (int sl : table_.cfg().source_layers) {
                int hl = std::clamp(sl, 0, n_host - 1);
                targets.push_back(hl);
            }
        } else {
            for (int e = 0; e < std::min(n_eng, 4); ++e) {
                targets.push_back(std::clamp((e + 1) * n_host / (std::min(n_eng, 4) + 1), 0, n_host - 1));
            }
        }
    }

    hooks_.resize(targets.size());
    std::mt19937 rng(20261123u);
    const float std_a = 1.0f / std::sqrt(static_cast<float>(d_eng));
    const float std_b = 1.0f / std::sqrt(static_cast<float>(kDefaultRank));
    std::normal_distribution<float> dist_a(0.0f, std_a);
    std::normal_distribution<float> dist_b(0.0f, std_b);

    std::vector<float> h_A(static_cast<size_t>(kDefaultRank) * d_eng);
    std::vector<float> h_B(static_cast<size_t>(d_host_) * kDefaultRank);

    for (size_t i = 0; i < targets.size(); ++i) {
        auto & hk = hooks_[i];
        hk.target_host_layer = std::clamp(targets[i], 0, n_host - 1);
        hk.src_engram_layer  = static_cast<int>(i) % n_eng;
        hk.device            = layer_to_device[hk.target_host_layer];
        hk.scale             = 1.0f;
        hk.rank              = kDefaultRank;

        for (float & v : h_A) v = dist_a(rng);
        for (float & v : h_B) v = dist_b(rng);

        hk.d_Av = dev_alloc<float>(hk.device, h_A.size());
        hk.d_Bv = dev_alloc<float>(hk.device, h_B.size());
        cuda_ok(cudaMemcpy(hk.d_Av, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice), "copy zero-shot A");
        cuda_ok(cudaMemcpy(hk.d_Bv, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice), "copy zero-shot B");
    }
}

void EngramSidecar::load_rkad_adapter(const std::string & adapter_path,
                                      const std::vector<int> & layer_to_device) {
    FILE * fp = std::fopen(adapter_path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Failed to open Engram adapter: " + adapter_path);

    char magic[4]{};
    uint32_t hdr[6]{}; // version, n_hooks, d_engram, d_host, default_rank, has_gate_or_flags
    if (std::fread(magic, 1, 4, fp) != 4 || std::memcmp(magic, "RKAD", 4) != 0 ||
        std::fread(hdr, sizeof(uint32_t), 6, fp) != 6) {
        std::fclose(fp);
        throw std::runtime_error("Invalid RKAD adapter header in: " + adapter_path);
    }

    const uint32_t version  = hdr[0];
    const uint32_t n_hooks  = hdr[1];
    const uint32_t d_engram = hdr[2];
    const uint32_t d_host   = hdr[3];
    const uint32_t rank_hdr = hdr[4];
    const uint32_t has_gate_hdr = hdr[5];

    if ((version != 1 && version != 2 && version != 3) ||
        static_cast<int>(d_engram) != table_.cfg().engram_dim() ||
        static_cast<int>(d_host) != d_host_) {
        std::fclose(fp);
        throw std::runtime_error("RKAD adapter version/dimension mismatch (v" + std::to_string(version) +
                                 ", d_engram=" + std::to_string(d_engram) +
                                 " vs table " + std::to_string(table_.cfg().engram_dim()) +
                                 ", d_host=" + std::to_string(d_host) +
                                 " vs host " + std::to_string(d_host_) + ")");
    }

    const int n_host = static_cast<int>(layer_to_device.size());
    const int n_eng  = std::max(1, table_.cfg().n_layers);

    manifest_.format_version = version;
    if (version >= 3) {
        uint32_t manifest_len = 0;
        if (std::fread(&manifest_len, sizeof(uint32_t), 1, fp) != 1 || manifest_len > 1024u * 1024u) {
            std::fclose(fp);
            throw std::runtime_error("Invalid RKAD v3 manifest length in: " + adapter_path);
        }
        std::string manifest_str(manifest_len, '\0');
        if (manifest_len > 0 && std::fread(manifest_str.data(), 1, manifest_len, fp) != manifest_len) {
            std::fclose(fp);
            throw std::runtime_error("Unexpected EOF reading RKAD v3 manifest JSON");
        }
        try {
            manifest_ = validate_rkad_v3_manifest(manifest_str, table_.cfg(), host_arch_, n_host, d_host_, hc_);
        } catch (...) {
            std::fclose(fp);
            throw;
        }
    }

    hooks_.resize(n_hooks);

    auto read_upload = [&](int dev, size_t n_elems) -> float * {
        std::vector<float> tmp(n_elems);
        if (std::fread(tmp.data(), sizeof(float), n_elems, fp) != n_elems) {
            std::fclose(fp);
            throw std::runtime_error("Unexpected EOF reading RKAD adapter weights");
        }
        float * d_ptr = dev_alloc<float>(dev, n_elems);
        cuda_ok(cudaMemcpy(d_ptr, tmp.data(), n_elems * sizeof(float), cudaMemcpyHostToDevice), "upload RKAD weight");
        return d_ptr;
    };

    for (uint32_t i = 0; i < n_hooks; ++i) {
        int32_t  meta_i[2]{};
        float    scale      = 1.0f;
        uint32_t hook_rank  = rank_hdr;
        bool     has_mu_e   = (version >= 2);
        bool     has_gate   = (has_gate_hdr != 0);
        float    delta_clip = 0.0f;

        if (std::fread(meta_i, sizeof(int32_t), 2, fp) != 2 || std::fread(&scale, sizeof(float), 1, fp) != 1) {
            std::fclose(fp);
            throw std::runtime_error("Unexpected EOF reading RKAD hook metadata");
        }
        if (version >= 3) {
            uint32_t h_extra[2]{}; // hook_rank, hook_flags (bit0=mu_e, bit1=gate, bit2=delta_clip)
            if (std::fread(h_extra, sizeof(uint32_t), 2, fp) != 2 ||
                std::fread(&delta_clip, sizeof(float), 1, fp) != 1) {
                std::fclose(fp);
                throw std::runtime_error("Unexpected EOF reading RKAD v3 hook extension metadata");
            }
            hook_rank = h_extra[0];
            has_mu_e  = (h_extra[1] & 1u) != 0;
            has_gate  = (h_extra[1] & 2u) != 0;
            if ((h_extra[1] & 4u) == 0) delta_clip = 0.0f;

            // Strict bounds checking in v3 (no silent clamping!)
            if (meta_i[0] < 0 || meta_i[0] >= n_host) {
                std::fclose(fp);
                throw std::runtime_error("RKAD v3 hook[" + std::to_string(i) + "] target_host_layer=" +
                                         std::to_string(meta_i[0]) + " out of range [0, " + std::to_string(n_host) + ")");
            }
            if (meta_i[1] < 0 || meta_i[1] >= n_eng) {
                std::fclose(fp);
                throw std::runtime_error("RKAD v3 hook[" + std::to_string(i) + "] src_engram_layer=" +
                                         std::to_string(meta_i[1]) + " out of range [0, " + std::to_string(n_eng) + ")");
            }
        }

        auto & hk = hooks_[i];
        hk.target_host_layer  = std::clamp(meta_i[0], 0, n_host - 1);
        hk.src_engram_layer   = std::clamp(meta_i[1], 0, n_eng - 1);
        hk.device             = layer_to_device[hk.target_host_layer];
        hk.scale              = scale;
        hk.default_delta_clip = delta_clip;
        hk.rank               = static_cast<int>(hook_rank);
        hk.h_mu_e.clear();

        if (has_mu_e) {
            hk.h_mu_e.resize(d_engram);
            if (std::fread(hk.h_mu_e.data(), sizeof(float), d_engram, fp) != d_engram) {
                std::fclose(fp);
                throw std::runtime_error("Unexpected EOF reading RKAD mu_E vector");
            }
        }

        if (hook_rank == 0) {
            hk.d_Av = read_upload(hk.device, static_cast<size_t>(d_host) * d_engram);
        } else {
            hk.d_Av = read_upload(hk.device, static_cast<size_t>(hook_rank) * d_engram);
            hk.d_Bv = read_upload(hk.device, static_cast<size_t>(d_host) * hook_rank);
        }
        if (has_gate) {
            if (hook_rank == 0) {
                hk.d_Ak = read_upload(hk.device, static_cast<size_t>(d_host) * d_engram);
            } else {
                hk.d_Ak = read_upload(hk.device, static_cast<size_t>(hook_rank) * d_engram);
                hk.d_Bk = read_upload(hk.device, static_cast<size_t>(d_host) * hook_rank);
            }
            hk.d_gatew = read_upload(hk.device, static_cast<size_t>(hc_) * d_host);
        }
    }
    std::fclose(fp);
}

void EngramSidecar::enable_calibration_dump(const std::string & out_path, bool dump_delta) {
    close_calibration_dump();
    dump_delta_ = dump_delta;
    delta_next_to_hook_.assign(layer_to_hook_.size(), -1);
    dump_h_prev_.assign(hooks_.size(), {});
    if (dump_delta_) {
        for (size_t i = 0; i < hooks_.size(); ++i) {
            const int next_l = hooks_[i].target_host_layer + 1;
            if (next_l < static_cast<int>(delta_next_to_hook_.size())) {
                delta_next_to_hook_[next_l] = static_cast<int>(i);
            }
        }
    }

    dump_fp_ = std::fopen(out_path.c_str(), "wb");
    if (!dump_fp_) throw std::runtime_error("Failed to open calibration dump path: " + out_path);

    const char magic[4] = {'R', 'K', 'P', 'R'};
    uint32_t hdr[4] = {1u, static_cast<uint32_t>(hooks_.size()),
                       static_cast<uint32_t>(table_.cfg().engram_dim()), static_cast<uint32_t>(d_host_)};
    uint64_t zero_cnt = 0;
    std::fwrite(magic, 1, 4, dump_fp_);
    std::fwrite(hdr, sizeof(uint32_t), 4, dump_fp_);
    std::fwrite(&zero_cnt, sizeof(uint64_t), 1, dump_fp_);
    for (const auto & hk : hooks_) {
        int32_t pair[2] = {hk.target_host_layer, hk.src_engram_layer};
        std::fwrite(pair, sizeof(int32_t), 2, dump_fp_);
    }
    dump_records_ = 0;
    std::printf("EngramSidecar calibration pair dump enabled (mode=%s) -> %s\n",
                dump_delta_ ? "delta(H_{L+1}-H_L)" : "state(H_L)", out_path.c_str());
}

void EngramSidecar::close_calibration_dump() {
    if (!dump_fp_) return;
    std::fseek(dump_fp_, 20, SEEK_SET);
    std::fwrite(&dump_records_, sizeof(uint64_t), 1, dump_fp_);
    std::fclose(dump_fp_);
    dump_fp_ = nullptr;
    std::printf("EngramSidecar calibration dump closed (%llu paired token records written)\n",
                static_cast<unsigned long long>(dump_records_));
}

void EngramSidecar::enable_tail_ce_dump(const std::string & out_path, int target_host_layer, int src_engram_layer) {
    close_tail_ce_dump();
    ce_target_layer_ = target_host_layer;
    ce_src_layer_    = src_engram_layer;
    ce_tokens_       = 0;
    ce_seq_state_.clear();

    ce_fp_ = std::fopen(out_path.c_str(), "wb");
    if (!ce_fp_) throw std::runtime_error("Failed to open tail-CE dump path: " + out_path);

    const int n_dump_layers = (ce_src_layer_ >= 0) ? 1 : std::max(1, table_.cfg().n_layers);
    const char magic[4] = {'R', 'K', 'C', 'E'};
    uint32_t hdr[6] = {
        1u,
        static_cast<uint32_t>(n_dump_layers),
        static_cast<uint32_t>(table_.cfg().engram_dim()),
        static_cast<uint32_t>(d_host_),
        static_cast<uint32_t>(hc_),
        static_cast<uint32_t>(target_host_layer)
    };
    uint64_t zero_tok = 0;
    std::fwrite(magic, 1, 4, ce_fp_);
    std::fwrite(hdr, sizeof(uint32_t), 6, ce_fp_);
    std::fwrite(&zero_tok, sizeof(uint64_t), 1, ce_fp_);
    std::printf("EngramSidecar tail-CE dump enabled (layer=%d, src_layer=%d, n_eng_layers=%u, d_engram=%u, hc=%u, d_host=%u) -> %s\n",
                target_host_layer, ce_src_layer_, hdr[1], hdr[2], hdr[4], hdr[3], out_path.c_str());
}

void EngramSidecar::close_tail_ce_dump() {
    if (!ce_fp_) return;
    std::fseek(ce_fp_, 28, SEEK_SET);
    std::fwrite(&ce_tokens_, sizeof(uint64_t), 1, ce_fp_);
    std::fclose(ce_fp_);
    ce_fp_ = nullptr;
    std::printf("EngramSidecar tail-CE dump closed (%llu tokens written)\n",
                static_cast<unsigned long long>(ce_tokens_));
}

void EngramSidecar::advance_seq_state(EngramSeqState & st,
                                      const std::vector<int32_t> & host_tokens,
                                      int start_pos,
                                      std::vector<int32_t> & out_src_window,
                                      std::vector<int32_t> & out_aligned_pos) const {
    const int T = static_cast<int>(host_tokens.size());
    out_aligned_pos.assign(T, 0);
    out_src_window.clear();

    if (start_pos == 0) {
        st.clear();
    } else if (start_pos < static_cast<int>(st.host_byte_ends.size()) ||
               start_pos < static_cast<int>(st.src_seq_cache.size())) {
        st.rollback_to(start_pos);
    }

    if (host_tok_ == nullptr || !table_.has_tokenizer()) {
        // Direct token-ID passthrough
        const int base = static_cast<int>(st.src_seq_cache.size());
        for (int i = 0; i < T; ++i) {
            st.src_seq_cache.push_back(host_tokens[i]);
            out_aligned_pos[i] = base + i;
        }
        out_src_window = st.src_seq_cache;
        return;
    }

    const int n_ctx_need = std::max(1, table_.cfg().ngram_size - 1);
    for (int i = 0; i < T; ++i) {
        st.rolling_text.append(host_tok_->decode_token(host_tokens[i]));
        st.host_byte_ends.push_back(st.rolling_text.size());
    }

    const size_t committed_bytes = st.committed_src_ends.empty() ? 0 : st.committed_src_ends.back();
    const std::string tail_text  = (committed_bytes > 0) ? st.rolling_text.substr(committed_bytes) : st.rolling_text;
    const auto tail_src = table_.tokenizer().encode(tail_text, false, false);
    const int n_tail = static_cast<int>(tail_src.size());

    // Build absolute byte ends for tail tokens (O(1) per token via cached token_byte_len!)
    std::vector<size_t> tail_ends(n_tail);
    {
        size_t cb = committed_bytes;
        for (int j = 0; j < n_tail; ++j) {
            cb += table_.tokenizer().token_byte_len(tail_src[j]);
            tail_ends[j] = cb;
        }
    }

    // Build compact source window = [last (ngram_size - 1) committed tokens] + [tail_src]
    const int n_comm = static_cast<int>(st.committed_src_tokens.size());
    const int prefix_take = std::min(n_comm, n_ctx_need);
    out_src_window.reserve(static_cast<size_t>(prefix_take + n_tail));
    for (int k = n_comm - prefix_take; k < n_comm; ++k) {
        out_src_window.push_back(st.committed_src_tokens[k]);
    }
    for (int j = 0; j < n_tail; ++j) {
        out_src_window.push_back(tail_src[j]);
    }

    // Align each incoming host token's byte end to its owning source token in `out_src_window`
    int j = 0;
    for (int i = 0; i < T; ++i) {
        const size_t target_b = st.host_byte_ends[start_pos + i];
        while (j + 1 < n_tail && tail_ends[j] < target_b) ++j;
        out_aligned_pos[i] = (n_tail > 0) ? (prefix_take + j) : std::max(0, prefix_take - 1);
    }

    // Advance boundary-anchored commit cursor: keep a 48-byte safety guard behind rolling_text.size(),
    // and commit everything strictly up to the LATEST pre-tokenizer-safe BPE cut <= end - 48!
    // If space-less text exceeds kMaxUncommittedTail bytes without a whitespace cut, commit up to the
    // latest provider token boundary <= max_c so memory & retokenization remain strictly O(1) bounded.
    constexpr size_t kCommitGuard = 48;
    constexpr size_t kMaxUncommittedTail = 1024;
    if (st.rolling_text.size() > committed_bytes + kCommitGuard + 16) {
        const size_t min_c = committed_bytes + 1;
        const size_t max_c = st.rolling_text.size() - kCommitGuard;
        const size_t cut = find_safe_bpe_cut(st.rolling_text, min_c, max_c, kMaxUncommittedTail);
        int commit_k = -1;
        if (cut > committed_bytes) {
            for (int k = 0; k < n_tail; ++k) {
                if (tail_ends[k] == cut) { commit_k = k; break; }
                if (tail_ends[k] > cut) break;
            }
            // Fallback when bounded-tail UTF-8 cut landed inside a multi-char space-less BPE token:
            // commit up to the latest complete source token ending <= cut.
            if (commit_k < 0 && st.rolling_text.size() > committed_bytes + kMaxUncommittedTail) {
                for (int k = 0; k < n_tail; ++k) {
                    if (tail_ends[k] <= cut) commit_k = k;
                    else break;
                }
            }
        }
        if (commit_k >= 0) {
            for (int k = 0; k <= commit_k; ++k) {
                st.committed_src_tokens.push_back(tail_src[k]);
                st.committed_src_ends.push_back(tail_ends[k]);
            }
        }
    }
}

void EngramSidecar::set_tail_ce_targets(const std::vector<int32_t> & targets,
                                        const std::vector<int32_t> & host_tokens,
                                        int start_pos) {
    if (!ce_fp_) return;
    const int T      = static_cast<int>(host_tokens.size());
    const int n_eng  = std::max(1, table_.cfg().n_layers);
    const int e_beg  = (ce_src_layer_ >= 0 && ce_src_layer_ < n_eng) ? ce_src_layer_ : 0;
    const int e_end  = (ce_src_layer_ >= 0 && ce_src_layer_ < n_eng) ? (ce_src_layer_ + 1) : n_eng;
    const int n_dump = e_end - e_beg;
    const int d_eng  = table_.cfg().engram_dim();
    const int n_cols = table_.cfg().n_cols;

    ce_cur_targets_ = targets;
    if (mask_special_tokens_ && host_tok_ != nullptr) {
        for (int i = 0; i < T && i < static_cast<int>(ce_cur_targets_.size()); ++i) {
            if (host_tok_->is_special_token(host_tokens[i])) ce_cur_targets_[i] = -1;
        }
    }
    if (start_pos == 0) ce_seq_state_.clear();

    // Strictly causal per-token source addressing using the production incremental commit cache!
    std::vector<std::vector<int32_t>> layer_rows(n_dump, std::vector<int32_t>(static_cast<size_t>(T) * n_cols));
    std::vector<std::vector<float>>   layer_val(n_dump,  std::vector<float>(static_cast<size_t>(T) * n_cols));
    std::vector<int32_t> s_win, s_pos, s_ids;
    std::vector<float>   s_val;

    for (int i = 0; i < T; ++i) {
        advance_seq_state(ce_seq_state_, {host_tokens[i]}, start_pos + i, s_win, s_pos);
        for (int idx = 0; idx < n_dump; ++idx) {
            const int e = e_beg + idx;
            table_.hash_tokens(e, s_win, s_pos[0], 1, s_ids, s_val);
            std::memcpy(layer_rows[idx].data() + static_cast<size_t>(i) * n_cols, s_ids.data(), n_cols * sizeof(int32_t));
            std::memcpy(layer_val[idx].data()  + static_cast<size_t>(i) * n_cols, s_val.data(), n_cols * sizeof(float));
        }
    }

    ce_cur_e_f16_.assign(n_dump, std::vector<uint16_t>(static_cast<size_t>(T) * d_eng));
    std::vector<float> tmp_e(static_cast<size_t>(T) * d_eng);
    for (int idx = 0; idx < n_dump; ++idx) {
        table_.prefetch_rows(layer_rows[idx].data(), static_cast<int>(layer_rows[idx].size()));
        table_.gather_dequant_f32(layer_rows[idx].data(), layer_val[idx].data(), T, tmp_e.data());
        for (size_t k = 0; k < tmp_e.size(); ++k) ce_cur_e_f16_[idx][k] = f32_to_f16_bits(tmp_e[k]);
    }
}

bool EngramSidecar::verify_incremental_addressing(const std::vector<int32_t> & host_tokens,
                                                  int & out_tokens_checked,
                                                  int & out_src_tok_matches,
                                                  int & out_hash_col_matches,
                                                  int & out_hash_col_total,
                                                  bool verbose) {
    out_tokens_checked   = 0;
    out_src_tok_matches  = 0;
    out_hash_col_matches = 0;
    out_hash_col_total   = 0;
    if (!host_tok_ || !table_.has_tokenizer() || host_tokens.empty()) return false;

    const int n_tok  = static_cast<int>(host_tokens.size());
    const int n_cols = table_.cfg().n_cols;
    const int n_eng  = std::max(1, table_.cfg().n_layers);

    // 1. Non-causal full-sequence reference (just to report lookahead skew)
    std::string full_text;
    std::vector<size_t> host_ends(n_tok);
    for (int i = 0; i < n_tok; ++i) {
        full_text.append(host_tok_->decode_token(host_tokens[i]));
        host_ends[i] = full_text.size();
    }
    const auto full_src = table_.tokenizer().encode(full_text, false, false);
    std::vector<size_t> full_src_ends(full_src.size());
    {
        size_t cb = 0;
        for (size_t j = 0; j < full_src.size(); ++j) {
            cb += table_.tokenizer().token_byte_len(full_src[j]);
            full_src_ends[j] = cb;
        }
    }
    std::vector<int> full_aligned(n_tok, 0);
    {
        int j = 0;
        for (int i = 0; i < n_tok; ++i) {
            while (j + 1 < static_cast<int>(full_src.size()) && full_src_ends[j] < host_ends[i]) ++j;
            full_aligned[i] = full_src.empty() ? 0 : j;
        }
    }

    // 2. Compare PRODUCTION incremental commit cache (`advance_seq_state`) token-by-token
    //    against the exact CAUSAL full-prefix oracle!
    EngramSeqState inc_st;
    std::string oracle_text;
    std::vector<int32_t> inc_win, inc_pos, ref_ids, inc_ids;
    std::vector<float>   ref_val, inc_val;
    int shown_mismatches = 0;
    int committed_steps  = 0;
    int noncausal_skew   = 0;
    size_t sum_tail_bytes = 0;

    for (int i = 0; i < n_tok; ++i) {
        oracle_text.append(host_tok_->decode_token(host_tokens[i]));
        const auto causal_full_src = table_.tokenizer().encode(oracle_text, false, false);
        const int  causal_idx      = std::max(0, static_cast<int>(causal_full_src.size()) - 1);

        const int32_t full_nc_tok = full_src.empty()        ? -1 : full_src[full_aligned[i]];
        const int32_t causal_tok  = causal_full_src.empty() ? -1 : causal_full_src[causal_idx];
        if (full_nc_tok != causal_tok) ++noncausal_skew;

        const size_t comm_before = inc_st.committed_src_ends.empty() ? 0 : inc_st.committed_src_ends.back();
        if (comm_before > 0) ++committed_steps;
        sum_tail_bytes += (oracle_text.size() - comm_before);

        advance_seq_state(inc_st, {host_tokens[i]}, i, inc_win, inc_pos);
        const int inc_idx = inc_pos[0];

        ++out_tokens_checked;
        const int32_t inc_tok = inc_win.empty() ? -1 : inc_win[inc_idx];
        if (causal_tok == inc_tok) ++out_src_tok_matches;

        for (int e = 0; e < n_eng; ++e) {
            table_.hash_tokens(e, causal_full_src, causal_idx, 1, ref_ids, ref_val);
            table_.hash_tokens(e, inc_win,         inc_idx,    1, inc_ids, inc_val);
            for (int c = 0; c < n_cols; ++c) {
                ++out_hash_col_total;
                if (ref_ids[c] == inc_ids[c] && ref_val[c] == inc_val[c]) {
                    ++out_hash_col_matches;
                } else if (verbose && shown_mismatches < 10) {
                    ++shown_mismatches;
                    std::printf("  [MISMATCH] pos=%d layer=%d col=%d comm_bytes=%zu causal_tok=%d('%s') inc_tok=%d('%s') ref_row=%d inc_row=%d\n",
                                i, e, c, comm_before, causal_tok, table_.tokenizer().decode_token(causal_tok).c_str(),
                                inc_tok, table_.tokenizer().decode_token(inc_tok).c_str(), ref_ids[c], inc_ids[c]);
                }
            }
        }
    }

    if (verbose) {
        std::printf("  (incremental commit cache active on %d / %d tokens, avg tail=%.1f bytes; non-causal prefill skew: %d / %d = %.2f%%)\n",
                    committed_steps, n_tok, static_cast<double>(sum_tail_bytes) / std::max(1, n_tok),
                    noncausal_skew, n_tok, 100.0 * noncausal_skew / std::max(1, n_tok));
    }

    return (out_src_tok_matches == out_tokens_checked) && (out_hash_col_matches == out_hash_col_total);
}

void EngramSidecar::prepare_step(const std::vector<int32_t> & host_tokens, int start_pos) {
    prepare_step(host_tokens, start_pos, default_params_, 0);
}

void EngramSidecar::prepare_step(const std::vector<int32_t> & host_tokens,
                                 int start_pos,
                                 const EngramRuntimeParams & params,
                                 int seq_id) {
    const int T = static_cast<int>(host_tokens.size());
    if (T <= 0 || hooks_.empty()) return;

    const auto tp0 = std::chrono::steady_clock::now();

    auto & st = seq_states_[seq_id];
    advance_seq_state(st, host_tokens, start_pos, scratch_src_win_, scratch_aligned_pos_);

    const auto tp1 = std::chrono::steady_clock::now();

    const int d_eng  = table_.cfg().engram_dim();
    const int n_cols = table_.cfg().n_cols;

    // 1. Hash all hooked Engram layers & issue non-blocking prefetches up front
    for (size_t hi = 0; hi < hooks_.size(); ++hi) {
        if (hooks_[hi].scale == 0.0f && dump_fp_ == nullptr) continue;
        scratch_hook_rows_[hi].resize(static_cast<size_t>(T) * n_cols);
        scratch_hook_valid_[hi].resize(static_cast<size_t>(T) * n_cols);
        for (int i = 0; i < T; ++i) {
            table_.hash_tokens(hooks_[hi].src_engram_layer, scratch_src_win_, scratch_aligned_pos_[i], 1,
                               scratch_single_ids_, scratch_single_valid_);
            std::memcpy(scratch_hook_rows_[hi].data() + static_cast<size_t>(i) * n_cols,
                        scratch_single_ids_.data(), n_cols * sizeof(int32_t));
            std::memcpy(scratch_hook_valid_[hi].data() + static_cast<size_t>(i) * n_cols,
                        scratch_single_valid_.data(), n_cols * sizeof(float));
        }
        table_.prefetch_rows(scratch_hook_rows_[hi].data(), static_cast<int>(scratch_hook_rows_[hi].size()));
    }

    const auto tp2 = std::chrono::steady_clock::now();

    // 2. Gather + dequant (+ optional RKAD v2/v3 mean-centering) & populate per-token CUDA-Graph-safe params
    for (size_t hi = 0; hi < hooks_.size(); ++hi) {
        auto & hk = hooks_[hi];
        if (hk.scale == 0.0f && dump_fp_ == nullptr) continue;

        const float eff_scale = params.scale * hk.scale;
        const float eff_clip  = (params.delta_clip_ratio > 0.0f) ? params.delta_clip_ratio : hk.default_delta_clip;
        bool any_active = false;
        for (int t = 0; t < T; ++t) {
            const float tok_scale = resolve_token_scale(host_tokens[t], eff_scale);
            hk.h_params_pin[t] = EngramRuntimeParams{tok_scale, params.gate_bias, eff_clip, 0.0f};
            if (tok_scale != 0.0f) any_active = true;
        }

        // Skip table gather if all tokens have tok_scale == 0 and no dump is active (still upload zeroed params so GPU early-exits!)
        if (any_active || dump_fp_ != nullptr) {
            table_.gather_dequant_f32(scratch_hook_rows_[hi].data(), scratch_hook_valid_[hi].data(), T, hk.h_engram_pin);
            // Never subtract mu_E when dumping calibration vectors (prevents double-centering!)
            if (!hk.h_mu_e.empty() && dump_fp_ == nullptr && ce_fp_ == nullptr) {
                const float * mu = hk.h_mu_e.data();
                for (int t = 0; t < T; ++t) {
                    float * dst = hk.h_engram_pin + static_cast<size_t>(t) * d_eng;
                    for (int d = 0; d < d_eng; ++d) dst[d] -= mu[d];
                }
            }
        }
    }

    const auto tp3 = std::chrono::steady_clock::now();

    // 3. Upload fixed-address params & Engram inputs on each hook's dedicated non-blocking H2D stream,
    //    then explicitly synchronize H2D completion before returning so ANY host compute stream
    //    (blocking, cudaStreamNonBlocking, or CUDA Graph replay) sees coherent buffers!
    for (size_t hi = 0; hi < hooks_.size(); ++hi) {
        auto & hk = hooks_[hi];
        if (hk.scale == 0.0f && dump_fp_ == nullptr) continue;
        bool any_active = false;
        for (int t = 0; t < T; ++t) {
            if (hk.h_params_pin[t].scale != 0.0f) { any_active = true; break; }
        }
        cuda_ok(cudaSetDevice(hk.device), "engram upload setDevice");
        cuda_ok(cudaMemcpyAsync(hk.d_params, hk.h_params_pin,
                                static_cast<size_t>(T) * sizeof(EngramRuntimeParams),
                                cudaMemcpyHostToDevice, hk.h2d_stream),
                "engram params H2D");
        if (any_active || dump_fp_ != nullptr) {
            cuda_ok(cudaMemcpyAsync(hk.d_engram_in, hk.h_engram_pin,
                                    static_cast<size_t>(T) * d_eng * sizeof(float),
                                    cudaMemcpyHostToDevice, hk.h2d_stream),
                    "engram H2D copy");
        }
    }
    for (size_t hi = 0; hi < hooks_.size(); ++hi) {
        auto & hk = hooks_[hi];
        if (hk.scale == 0.0f && dump_fp_ == nullptr) continue;
        cuda_ok(cudaSetDevice(hk.device), "engram sync setDevice");
        cuda_ok(cudaStreamSynchronize(hk.h2d_stream), "engram h2d sync");
    }

    const auto tp4 = std::chrono::steady_clock::now();
    prof_.steps += 1;
    prof_.tokens += static_cast<uint64_t>(T);
    prof_.tokenize_align_us += std::chrono::duration<double, std::micro>(tp1 - tp0).count();
    prof_.hash_prefetch_us  += std::chrono::duration<double, std::micro>(tp2 - tp1).count();
    prof_.gather_dequant_us += std::chrono::duration<double, std::micro>(tp3 - tp2).count();
    prof_.h2d_submit_us     += std::chrono::duration<double, std::micro>(tp4 - tp3).count();
}

void EngramSidecar::prepare_batch_step(const std::vector<int> & seq_ids,
                                       const std::vector<int32_t> & host_tokens,
                                       const std::vector<int32_t> & positions,
                                       const std::vector<EngramRuntimeParams> & per_token_params) {
    const int T = static_cast<int>(host_tokens.size());
    if (T <= 0 || hooks_.empty()) return;
    if (static_cast<int>(seq_ids.size()) != T ||
        static_cast<int>(positions.size()) != T ||
        static_cast<int>(per_token_params.size()) != T) {
        throw std::runtime_error("EngramSidecar::prepare_batch_step size mismatch");
    }

    const auto tp0 = std::chrono::steady_clock::now();
    const int d_eng  = table_.cfg().engram_dim();
    const int n_cols = table_.cfg().n_cols;

    for (size_t hi = 0; hi < hooks_.size(); ++hi) {
        scratch_hook_rows_[hi].resize(static_cast<size_t>(T) * n_cols);
        scratch_hook_valid_[hi].resize(static_cast<size_t>(T) * n_cols);
    }

    std::vector<int32_t> s_win, s_pos;
    for (int t = 0; t < T; ++t) {
        auto & st = seq_states_[seq_ids[t]];
        advance_seq_state(st, {host_tokens[t]}, positions[t], s_win, s_pos);
        for (size_t hi = 0; hi < hooks_.size(); ++hi) {
            if (hooks_[hi].scale == 0.0f) continue;
            table_.hash_tokens(hooks_[hi].src_engram_layer, s_win, s_pos[0], 1,
                               scratch_single_ids_, scratch_single_valid_);
            std::memcpy(scratch_hook_rows_[hi].data() + static_cast<size_t>(t) * n_cols,
                        scratch_single_ids_.data(), n_cols * sizeof(int32_t));
            std::memcpy(scratch_hook_valid_[hi].data() + static_cast<size_t>(t) * n_cols,
                        scratch_single_valid_.data(), n_cols * sizeof(float));
        }
    }

    const auto tp1 = std::chrono::steady_clock::now();

    for (size_t hi = 0; hi < hooks_.size(); ++hi) {
        if (hooks_[hi].scale == 0.0f) continue;
        table_.prefetch_rows(scratch_hook_rows_[hi].data(), static_cast<int>(scratch_hook_rows_[hi].size()));
    }

    const auto tp2 = std::chrono::steady_clock::now();

    for (size_t hi = 0; hi < hooks_.size(); ++hi) {
        auto & hk = hooks_[hi];
        if (hk.scale == 0.0f) continue;
        bool any_active = false;
        for (int t = 0; t < T; ++t) {
            const float eff_scale = resolve_token_scale(host_tokens[t], per_token_params[t].scale * hk.scale);
            const float eff_clip  = (per_token_params[t].delta_clip_ratio > 0.0f)
                ? per_token_params[t].delta_clip_ratio : hk.default_delta_clip;
            hk.h_params_pin[t] = EngramRuntimeParams{eff_scale, per_token_params[t].gate_bias, eff_clip, 0.0f};
            if (eff_scale != 0.0f) any_active = true;
        }
        if (any_active) {
            table_.gather_dequant_f32(scratch_hook_rows_[hi].data(), scratch_hook_valid_[hi].data(), T, hk.h_engram_pin);
            if (!hk.h_mu_e.empty()) {
                const float * mu = hk.h_mu_e.data();
                for (int t = 0; t < T; ++t) {
                    float * dst = hk.h_engram_pin + static_cast<size_t>(t) * d_eng;
                    for (int d = 0; d < d_eng; ++d) dst[d] -= mu[d];
                }
            }
        }
        cuda_ok(cudaSetDevice(hk.device), "engram batch upload setDevice");
        cuda_ok(cudaMemcpyAsync(hk.d_params, hk.h_params_pin,
                                static_cast<size_t>(T) * sizeof(EngramRuntimeParams),
                                cudaMemcpyHostToDevice, hk.h2d_stream),
                "engram batch params H2D");
        if (any_active) {
            cuda_ok(cudaMemcpyAsync(hk.d_engram_in, hk.h_engram_pin,
                                    static_cast<size_t>(T) * d_eng * sizeof(float),
                                    cudaMemcpyHostToDevice, hk.h2d_stream),
                    "engram batch H2D");
        }
    }
    for (size_t hi = 0; hi < hooks_.size(); ++hi) {
        auto & hk = hooks_[hi];
        if (hk.scale == 0.0f) continue;
        cuda_ok(cudaSetDevice(hk.device), "engram batch sync setDevice");
        cuda_ok(cudaStreamSynchronize(hk.h2d_stream), "engram batch h2d sync");
    }

    const auto tp3 = std::chrono::steady_clock::now();
    prof_.steps += 1;
    prof_.tokens += static_cast<uint64_t>(T);
    prof_.tokenize_align_us += std::chrono::duration<double, std::micro>(tp1 - tp0).count();
    prof_.hash_prefetch_us  += std::chrono::duration<double, std::micro>(tp2 - tp1).count();
    prof_.gather_dequant_us += std::chrono::duration<double, std::micro>(tp3 - tp2).count();
}

void EngramSidecar::on_post_layer(int layer_idx, int device, float * d_inpL, int n_tokens, int n_embd, int hc, cudaStream_t stream) {
    if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_to_hook_.size()) return;

    const int d_eng = table_.cfg().engram_dim();

    // Optional RKCE tail-CE dump at target layer
    if (ce_fp_ != nullptr && layer_idx == ce_target_layer_) {
        const size_t n_h = static_cast<size_t>(n_tokens) * hc * n_embd;
        ce_h_tmp_.resize(n_h);
        cuda_ok(cudaMemcpyAsync(ce_h_tmp_.data(), d_inpL, n_h * sizeof(float), cudaMemcpyDeviceToHost, stream), "engram ce D2H");
        cuda_ok(cudaStreamSynchronize(stream), "engram ce sync");

        std::vector<uint16_t> h_f16(n_h);
        for (size_t k = 0; k < n_h; ++k) h_f16[k] = f32_to_f16_bits(ce_h_tmp_[k]);

        const int32_t nt = n_tokens;
        std::fwrite(&nt, sizeof(int32_t), 1, ce_fp_);
        std::fwrite(ce_cur_targets_.data(), sizeof(int32_t), n_tokens, ce_fp_);
        std::fwrite(h_f16.data(), sizeof(uint16_t), n_h, ce_fp_);
        for (const auto & ef : ce_cur_e_f16_) {
            std::fwrite(ef.data(), sizeof(uint16_t), ef.size(), ce_fp_);
        }
        ce_tokens_ += static_cast<uint64_t>(n_tokens);
    }

    // If dump_delta_ is active and this layer is L+1 for some hook, compute H_{L+1} - H_L and write pair!
    if (dump_fp_ != nullptr && dump_delta_ &&
        layer_idx < static_cast<int>(delta_next_to_hook_.size()) && delta_next_to_hook_[layer_idx] >= 0) {
        const int hi_prev = delta_next_to_hook_[layer_idx];
        auto & hk_prev = hooks_[hi_prev];
        float * d_tmp = (device == hk_prev.device) ? hk_prev.d_hmean : dev_alloc<float>(device, static_cast<size_t>(n_tokens) * n_embd);
        launch_engram_mean_streams(d_inpL, d_tmp, n_tokens, n_embd, hc, stream);
        dump_h_tmp_.resize(static_cast<size_t>(n_tokens) * n_embd);
        cuda_ok(cudaMemcpyAsync(dump_h_tmp_.data(), d_tmp,
                                static_cast<size_t>(n_tokens) * n_embd * sizeof(float),
                                cudaMemcpyDeviceToHost, stream), "engram delta D2H");
        cuda_ok(cudaStreamSynchronize(stream), "engram delta sync");
        if (device != hk_prev.device) cudaFree(d_tmp);

        const auto & h0 = dump_h_prev_[hi_prev];
        if (h0.size() == dump_h_tmp_.size()) {
            for (size_t k = 0; k < dump_h_tmp_.size(); ++k) dump_h_tmp_[k] -= h0[k];
            const int32_t rec_hdr[2] = {hi_prev, n_tokens};
            std::fwrite(rec_hdr, sizeof(int32_t), 2, dump_fp_);
            std::fwrite(hk_prev.h_engram_pin, sizeof(float), static_cast<size_t>(n_tokens) * d_eng, dump_fp_);
            std::fwrite(dump_h_tmp_.data(), sizeof(float), static_cast<size_t>(n_tokens) * n_embd, dump_fp_);
            dump_records_ += static_cast<uint64_t>(n_tokens);
        }
    }

    const int hi = layer_to_hook_[layer_idx];
    if (hi < 0) return;

    auto & hk = hooks_[hi];

    // Optional calibration pair dump (before injection, so target hidden state is unperturbed!)
    if (dump_fp_ != nullptr) {
        launch_engram_mean_streams(d_inpL, hk.d_hmean, n_tokens, n_embd, hc, stream);
        dump_h_tmp_.resize(static_cast<size_t>(n_tokens) * n_embd);
        cuda_ok(cudaMemcpyAsync(dump_h_tmp_.data(), hk.d_hmean,
                                static_cast<size_t>(n_tokens) * n_embd * sizeof(float),
                                cudaMemcpyDeviceToHost, stream), "engram dump D2H");
        cuda_ok(cudaStreamSynchronize(stream), "engram dump sync");

        if (dump_delta_) {
            dump_h_prev_[hi] = dump_h_tmp_;
        } else {
            const int32_t rec_hdr[2] = {hi, n_tokens};
            std::fwrite(rec_hdr, sizeof(int32_t), 2, dump_fp_);
            std::fwrite(hk.h_engram_pin, sizeof(float), static_cast<size_t>(n_tokens) * d_eng, dump_fp_);
            std::fwrite(dump_h_tmp_.data(), sizeof(float), static_cast<size_t>(n_tokens) * n_embd, dump_fp_);
            dump_records_ += static_cast<uint64_t>(n_tokens);
        }
    }

    // Permanently disabled hook (e.g. auto-gated off in adapter file): skip graph capture completely
    if (hk.scale == 0.0f) return;

    if (gpu_profile_) cudaEventRecord(hk.ev_start, stream);

    // Always capture fixed-address kernels reading hk.d_params[t] so per-request enable/disable/scale/clip
    // works dynamically WITHOUT ever invalidating or recapturing CUDA Graphs!
    if (hk.rank == 0) {
        launch_engram_matvec_f32(hk.d_engram_in, hk.d_Av, hk.d_val, hk.d_params, n_tokens, n_embd, d_eng, stream);
    } else {
        launch_engram_matvec_f32(hk.d_engram_in, hk.d_Av, hk.d_mid, hk.d_params, n_tokens, hk.rank, d_eng, stream);
        launch_engram_matvec_f32(hk.d_mid,       hk.d_Bv, hk.d_val, hk.d_params, n_tokens, n_embd, hk.rank, stream);
    }

    if (hk.d_Ak != nullptr && hk.d_key != nullptr) {
        if (hk.rank == 0) {
            launch_engram_matvec_f32(hk.d_engram_in, hk.d_Ak, hk.d_key, hk.d_params, n_tokens, n_embd, d_eng, stream);
        } else {
            launch_engram_matvec_f32(hk.d_engram_in, hk.d_Ak, hk.d_mid, hk.d_params, n_tokens, hk.rank, d_eng, stream);
            launch_engram_matvec_f32(hk.d_mid,       hk.d_Bk, hk.d_key, hk.d_params, n_tokens, n_embd, hk.rank, stream);
        }
    }

    launch_engram_gated_inject(d_inpL, hk.d_val, hk.d_key, hk.d_gatew,
                               hk.d_params, n_tokens, n_embd, hc, 1e-6f, stream);

    if (gpu_profile_) {
        cudaEventRecord(hk.ev_stop, stream);
        cudaEventSynchronize(hk.ev_stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, hk.ev_start, hk.ev_stop);
        prof_.gpu_hook_us += static_cast<double>(ms) * 1000.0;
        prof_.gpu_hook_calls += 1;
    }
}

} // namespace rookery
