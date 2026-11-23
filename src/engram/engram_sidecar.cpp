#include "engram/engram_sidecar.h"

#include <algorithm>
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

} // namespace

EngramSidecar::~EngramSidecar() {
    close_calibration_dump();
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
        freep(hk.d_mid);
        freep(hk.d_val);
        freep(hk.d_key);
        freep(hk.d_hmean);
        if (hk.h_engram_pin) cudaFreeHost(hk.h_engram_pin);
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
                         const std::vector<int> & override_host_layers) {
    free_gpu_buffers();

    host_tok_     = host_tokenizer;
    d_host_       = d_host;
    hc_           = hc;
    max_batch_    = max_batch;
    global_scale_ = global_scale;

    table_.open(engram_path);
    layer_to_hook_.assign(layer_to_device.size(), -1);

    if (!adapter_path.empty()) {
        load_rkad_adapter(adapter_path, layer_to_device);
    } else {
        init_zero_shot_projection(layer_to_device, override_host_layers);
    }

    // Allocate fixed-address pinned host + device staging buffers (CUDA-Graph safe!)
    const int d_eng = table_.cfg().engram_dim();
    for (size_t i = 0; i < hooks_.size(); ++i) {
        auto & hk = hooks_[i];
        hk.h_engram_pin = pin_alloc<float>(static_cast<size_t>(max_batch_) * d_eng);
        hk.d_engram_in  = dev_alloc<float>(hk.device, static_cast<size_t>(max_batch_) * d_eng);
        hk.d_mid        = dev_alloc<float>(hk.device, static_cast<size_t>(max_batch_) * std::max(hk.rank, 1));
        hk.d_val        = dev_alloc<float>(hk.device, static_cast<size_t>(max_batch_) * d_host_);
        if (hk.d_Ak != nullptr) {
            hk.d_key = dev_alloc<float>(hk.device, static_cast<size_t>(max_batch_) * d_host_);
        }
        hk.d_hmean = dev_alloc<float>(hk.device, static_cast<size_t>(max_batch_) * d_host_);
        cuda_ok(cudaSetDevice(hk.device), "engram memset setDevice");
        cuda_ok(cudaMemset(hk.d_engram_in, 0, static_cast<size_t>(max_batch_) * d_eng * sizeof(float)), "engram memset");
        layer_to_hook_[hk.target_host_layer] = static_cast<int>(i);
    }

    std::printf("EngramSidecar attached: %zu hook(s), d_engram=%d -> d_host=%d, global_scale=%.4f, mode=%s\n",
                hooks_.size(), d_eng, d_host_, global_scale_,
                adapter_path.empty() ? "zero-shot-ortho" : adapter_path.c_str());
    for (size_t i = 0; i < hooks_.size(); ++i) {
        std::printf("  hook[%zu]: engram_layer=%d -> host_layer=%d (GPU%d, rank=%d, gated=%s, scale=%.4f)\n",
                    i, hooks_[i].src_engram_layer, hooks_[i].target_host_layer, hooks_[i].device,
                    hooks_[i].rank, hooks_[i].d_Ak ? "yes" : "no", hooks_[i].scale);
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
            // Map proportionally onto host layers (clamped to valid range)
            for (int sl : table_.cfg().source_layers) {
                int hl = std::clamp(sl, 0, n_host - 1);
                targets.push_back(hl);
            }
        } else {
            // Uniformly space across host layers
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
    uint32_t hdr[6]{}; // version, n_hooks, d_engram, d_host, rank, has_gate
    if (std::fread(magic, 1, 4, fp) != 4 || std::memcmp(magic, "RKAD", 4) != 0 ||
        std::fread(hdr, sizeof(uint32_t), 6, fp) != 6) {
        std::fclose(fp);
        throw std::runtime_error("Invalid RKAD adapter header in: " + adapter_path);
    }

    const uint32_t version  = hdr[0];
    const uint32_t n_hooks  = hdr[1];
    const uint32_t d_engram = hdr[2];
    const uint32_t d_host   = hdr[3];
    const uint32_t rank     = hdr[4];
    const uint32_t has_gate = hdr[5];

    if (version != 1 || static_cast<int>(d_engram) != table_.cfg().engram_dim() || static_cast<int>(d_host) != d_host_) {
        std::fclose(fp);
        throw std::runtime_error("RKAD adapter dimension mismatch (file d_engram=" + std::to_string(d_engram) +
                                 ", d_host=" + std::to_string(d_host) + ")");
    }

    const int n_host = static_cast<int>(layer_to_device.size());
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
        int32_t meta_i[2]{};
        float   scale = 1.0f;
        if (std::fread(meta_i, sizeof(int32_t), 2, fp) != 2 || std::fread(&scale, sizeof(float), 1, fp) != 1) {
            std::fclose(fp);
            throw std::runtime_error("Unexpected EOF reading RKAD hook metadata");
        }
        auto & hk = hooks_[i];
        hk.target_host_layer = std::clamp(meta_i[0], 0, n_host - 1);
        hk.src_engram_layer  = meta_i[1];
        hk.device            = layer_to_device[hk.target_host_layer];
        hk.scale             = scale;
        hk.rank              = static_cast<int>(rank);

        if (rank == 0) {
            hk.d_Av = read_upload(hk.device, static_cast<size_t>(d_host) * d_engram);
        } else {
            hk.d_Av = read_upload(hk.device, static_cast<size_t>(rank) * d_engram);
            hk.d_Bv = read_upload(hk.device, static_cast<size_t>(d_host) * rank);
        }
        if (has_gate) {
            if (rank == 0) {
                hk.d_Ak = read_upload(hk.device, static_cast<size_t>(d_host) * d_engram);
            } else {
                hk.d_Ak = read_upload(hk.device, static_cast<size_t>(rank) * d_engram);
                hk.d_Bk = read_upload(hk.device, static_cast<size_t>(d_host) * rank);
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

    // "RKPR" v1 header: magic[4], version(u32), n_hooks(u32), d_engram(u32), d_host(u32), record_count(u64 placeholder)
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
    // Patch record_count at byte offset 20
    std::fseek(dump_fp_, 20, SEEK_SET);
    std::fwrite(&dump_records_, sizeof(uint64_t), 1, dump_fp_);
    std::fclose(dump_fp_);
    dump_fp_ = nullptr;
    std::printf("EngramSidecar calibration dump closed (%llu paired token records written)\n",
                static_cast<unsigned long long>(dump_records_));
}

void EngramSidecar::prepare_step(const std::vector<int32_t> & host_tokens, int start_pos) {
    const int T = static_cast<int>(host_tokens.size());
    if (T <= 0 || hooks_.empty()) return;

    // Reset rolling state on new sequence (start_pos == 0) or backtrack (e.g. MTP rejection!)
    if (start_pos == 0) {
        rolling_text_.clear();
        host_byte_ends_.clear();
        src_seq_cache_.clear();
    } else if (start_pos < static_cast<int>(host_byte_ends_.size())) {
        // MTP rejected draft token(s): trim rolling text back to exact byte end of start_pos - 1!
        const size_t keep_bytes = (start_pos > 0) ? host_byte_ends_[start_pos - 1] : 0;
        rolling_text_.resize(keep_bytes);
        host_byte_ends_.resize(start_pos);
    }

    // Dual-tokenizer text-window bridging:
    // Decode incoming host tokens into UTF-8 bytes and align each host token to the Engram source tokenizer.
    std::vector<int32_t> aligned_src_pos(T, 0);
    std::vector<int32_t> src_window_tokens;

    if (host_tok_ != nullptr && table_.has_tokenizer()) {
        for (int i = 0; i < T; ++i) {
            rolling_text_.append(host_tok_->decode_token(host_tokens[i]));
            host_byte_ends_.push_back(rolling_text_.size());
        }

        // For decode (T <= 4) on long contexts, only re-tokenize a bounded UTF-8 suffix (~128 bytes)
        // since N-gram hash only needs the last `ngram_size` (~3-4) tokens!
        constexpr size_t kMaxWindowBytes = 128;
        if (T <= 4 && rolling_text_.size() > kMaxWindowBytes) {
            size_t cut = rolling_text_.size() - kMaxWindowBytes;
            // Advance cut to a UTF-8 codepoint boundary
            while (cut < rolling_text_.size() && (static_cast<uint8_t>(rolling_text_[cut]) & 0xC0) == 0x80) {
                ++cut;
            }
            const std::string suffix = rolling_text_.substr(cut);
            src_window_tokens = table_.tokenizer().encode(suffix, false, false);
            const int n_src = static_cast<int>(src_window_tokens.size());
            // Map the T new host tokens to the tail of src_window_tokens by byte progress
            size_t cur_bytes = 0;
            std::vector<size_t> src_byte_ends;
            src_byte_ends.reserve(n_src);
            for (int32_t st : src_window_tokens) {
                cur_bytes += table_.tokenizer().decode_token(st).size();
                src_byte_ends.push_back(cur_bytes);
            }
            for (int i = 0; i < T; ++i) {
                const size_t rel_end = host_byte_ends_[start_pos + i] - cut;
                int s_idx = std::max(0, n_src - 1);
                for (int j = 0; j < n_src; ++j) {
                    if (src_byte_ends[j] >= rel_end) { s_idx = j; break; }
                }
                aligned_src_pos[i] = s_idx;
            }
        } else {
            // Prefill / short context: tokenize rolling_text_ once and align each host token by byte end
            src_window_tokens = table_.tokenizer().encode(rolling_text_, false, false);
            const int n_src = static_cast<int>(src_window_tokens.size());
            std::vector<size_t> src_byte_ends;
            src_byte_ends.reserve(n_src);
            size_t cur_bytes = 0;
            for (int32_t st : src_window_tokens) {
                cur_bytes += table_.tokenizer().decode_token(st).size();
                src_byte_ends.push_back(cur_bytes);
            }
            int j = 0;
            for (int i = 0; i < T; ++i) {
                const size_t target_b = host_byte_ends_[start_pos + i];
                while (j + 1 < n_src && src_byte_ends[j] < target_b) ++j;
                aligned_src_pos[i] = (n_src > 0) ? j : 0;
            }
        }
    } else {
        // Direct token-ID passthrough (same tokenizer or unit test without host tokenizer)
        if (start_pos < static_cast<int>(src_seq_cache_.size())) {
            src_seq_cache_.resize(start_pos);
        }
        for (int i = 0; i < T; ++i) {
            src_seq_cache_.push_back(host_tokens[i]);
            aligned_src_pos[i] = start_pos + i;
        }
        src_window_tokens = src_seq_cache_;
    }

    const int d_eng  = table_.cfg().engram_dim();
    const int n_cols = table_.cfg().n_cols;
    std::vector<int32_t> row_ids;
    std::vector<float>   valid_mask;
    std::vector<int32_t> single_ids;
    std::vector<float>   single_valid;

    // 1. Hash all hooked Engram layers & issue non-blocking prefetches up front
    std::vector<std::vector<int32_t>> hook_rows(hooks_.size());
    std::vector<std::vector<float>>   hook_valid(hooks_.size());

    for (size_t hi = 0; hi < hooks_.size(); ++hi) {
        hook_rows[hi].resize(static_cast<size_t>(T) * n_cols);
        hook_valid[hi].resize(static_cast<size_t>(T) * n_cols);
        for (int i = 0; i < T; ++i) {
            table_.hash_tokens(hooks_[hi].src_engram_layer, src_window_tokens, aligned_src_pos[i], 1, single_ids, single_valid);
            std::memcpy(hook_rows[hi].data() + static_cast<size_t>(i) * n_cols, single_ids.data(), n_cols * sizeof(int32_t));
            std::memcpy(hook_valid[hi].data() + static_cast<size_t>(i) * n_cols, single_valid.data(), n_cols * sizeof(float));
        }
        table_.prefetch_rows(hook_rows[hi].data(), static_cast<int>(hook_rows[hi].size()));
    }

    // 2. Gather + dequant into pinned host memory and upload to each hook's fixed GPU address
    for (size_t hi = 0; hi < hooks_.size(); ++hi) {
        auto & hk = hooks_[hi];
        table_.gather_dequant_f32(hook_rows[hi].data(), hook_valid[hi].data(), T, hk.h_engram_pin);

        cuda_ok(cudaSetDevice(hk.device), "engram upload setDevice");
        cuda_ok(cudaMemcpyAsync(hk.d_engram_in, hk.h_engram_pin,
                                static_cast<size_t>(T) * d_eng * sizeof(float),
                                cudaMemcpyHostToDevice, nullptr),
                "engram H2D copy");
    }
}

void EngramSidecar::on_post_layer(int layer_idx, int device, float * d_inpL, int n_tokens, int n_embd, int hc, cudaStream_t stream) {
    if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_to_hook_.size()) return;

    const int d_eng = table_.cfg().engram_dim();

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

    const float eff_scale = global_scale_ * hk.scale;
    if (eff_scale == 0.0f) return;

    // Project Engram vector -> host value vector d_val[T, n_embd]
    if (hk.rank == 0) {
        launch_engram_matvec_f32(hk.d_engram_in, hk.d_Av, hk.d_val, n_tokens, n_embd, d_eng, stream);
    } else {
        launch_engram_matvec_f32(hk.d_engram_in, hk.d_Av, hk.d_mid, n_tokens, hk.rank, d_eng, stream);
        launch_engram_matvec_f32(hk.d_mid,       hk.d_Bv, hk.d_val, n_tokens, n_embd, hk.rank, stream);
    }

    // Optional context-dependent key projection for DeepSeek/HY4 RMSNorm gate
    if (hk.d_Ak != nullptr && hk.d_key != nullptr) {
        if (hk.rank == 0) {
            launch_engram_matvec_f32(hk.d_engram_in, hk.d_Ak, hk.d_key, n_tokens, n_embd, d_eng, stream);
        } else {
            launch_engram_matvec_f32(hk.d_engram_in, hk.d_Ak, hk.d_mid, n_tokens, hk.rank, d_eng, stream);
            launch_engram_matvec_f32(hk.d_mid,       hk.d_Bk, hk.d_key, n_tokens, n_embd, hk.rank, stream);
        }
    }

    launch_engram_gated_inject(d_inpL, hk.d_val, hk.d_key, hk.d_gatew,
                               eff_scale, n_tokens, n_embd, hc, 1e-6f, stream);
}

} // namespace rookery
