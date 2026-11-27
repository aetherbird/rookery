#include "cpu/gemma4_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace rookery::cpu {

namespace {

inline size_t kv_head_stride_bytes(int head_dim, KvCacheType kv_type) {
    const size_t d_bytes = static_cast<size_t>(head_dim >> 5) * sizeof(uint16_t);
    const size_t d_pad   = (d_bytes + 31u) & ~size_t(31);
    if (kv_type == KvCacheType::Q8_0) {
        return static_cast<size_t>(head_dim) + d_pad;
    }
    if (kv_type == KvCacheType::Q4_0) {
        return static_cast<size_t>(head_dim >> 1) + d_pad;
    }
    return static_cast<size_t>(head_dim) * sizeof(uint16_t);
}

inline std::vector<int> read_int_or_array(const GgufFile & meta, const std::string & key, int n_layers, int fallback) {
    std::vector<int> out(static_cast<size_t>(n_layers), fallback);
    if (!meta.has_kv(key)) return out;
    const auto & v = meta.kv(key);
    if (auto * vu = std::get_if<std::vector<uint64_t>>(&v.data)) {
        for (int i = 0; i < n_layers && i < static_cast<int>(vu->size()); ++i) out[i] = static_cast<int>((*vu)[i]);
    } else if (auto * vi = std::get_if<std::vector<int64_t>>(&v.data)) {
        for (int i = 0; i < n_layers && i < static_cast<int>(vi->size()); ++i) out[i] = static_cast<int>((*vi)[i]);
    } else {
        out.assign(static_cast<size_t>(n_layers), static_cast<int>(v.as_u64()));
    }
    return out;
}

} // namespace

CpuTensor Gemma4CpuModel::bind_tensor(const std::string & name, bool required) const {
    const auto loc = sg_.find_tensor(name);
    if (!loc.info) {
        if (required) throw std::runtime_error("Gemma4CpuModel: missing tensor '" + name + "'");
        return CpuTensor{};
    }
    CpuTensor t;
    t.ptr    = loc.data();
    t.dtype  = loc.info->dtype;
    t.nbytes = loc.info->nbytes;
    for (int d = 0; d < 4; ++d) t.ne[d] = loc.info->ne[d];
    return t;
}

void Gemma4CpuModel::load(const std::string & gguf_path,
                          int max_ctx,
                          KvCacheType kv_type,
                          int n_threads) {
    const auto t0 = std::chrono::steady_clock::now();
    sg_ = ShardedGguf::open(gguf_path);
    const std::string arch = sg_.get_str("general.architecture");
    if (arch != "gemma4") {
        throw std::runtime_error("Gemma4CpuModel::load: expected general.architecture='gemma4', got '" + arch + "'");
    }

    pool_.init(n_threads);

    const GgufFile & meta = sg_.meta();
    cfg_.n_layers              = static_cast<int>(sg_.get_u64("gemma4.block_count", 42));
    cfg_.n_embd                = static_cast<int>(sg_.get_u64("gemma4.embedding_length", 2560));
    cfg_.n_head                = static_cast<int>(sg_.get_u64("gemma4.attention.head_count", 8));
    cfg_.head_dim_full         = static_cast<int>(sg_.get_u64("gemma4.attention.key_length", 512));
    cfg_.head_dim_swa          = static_cast<int>(sg_.get_u64("gemma4.attention.key_length_swa", 256));
    cfg_.rope_dim_full         = static_cast<int>(sg_.get_u64("gemma4.rope.dimension_count", cfg_.head_dim_full));
    cfg_.rope_dim_swa          = static_cast<int>(sg_.get_u64("gemma4.rope.dimension_count_swa", cfg_.head_dim_swa));
    cfg_.rope_freq_base_full   = static_cast<float>(sg_.get_f64("gemma4.rope.freq_base", 1000000.0));
    cfg_.rope_freq_base_swa    = static_cast<float>(sg_.get_f64("gemma4.rope.freq_base_swa", 10000.0));
    cfg_.rms_eps               = static_cast<float>(sg_.get_f64("gemma4.attention.layer_norm_rms_epsilon", 1e-6));
    cfg_.sliding_window        = static_cast<int>(sg_.get_u64("gemma4.attention.sliding_window", 512));
    cfg_.shared_kv_layers      = static_cast<int>(sg_.get_u64("gemma4.attention.shared_kv_layers", 0));
    cfg_.n_layer_kv_from_start = cfg_.n_layers - cfg_.shared_kv_layers;
    cfg_.n_embd_per_layer      = static_cast<int>(sg_.get_u64("gemma4.embedding_length_per_layer_input", 0));
    cfg_.n_expert              = static_cast<int>(sg_.get_u64("gemma4.expert_count", 0));
    cfg_.n_expert_used         = static_cast<int>(sg_.get_u64("gemma4.expert_used_count", 0));
    cfg_.final_logit_softcap   = static_cast<float>(sg_.get_f64("gemma4.final_logit_softcapping", 30.0));
    cfg_.max_ctx               = max_ctx;
    cfg_.kv_type               = kv_type;
    cfg_.kv_q8_0               = (kv_type != KvCacheType::F16);
    cfg_.swa_slots             = std::min(max_ctx, std::max(2048, cfg_.sliding_window * 2));

    if (meta.has_kv("gemma4.expert_feed_forward_length")) {
        auto ff_exp_arr = read_int_or_array(meta, "gemma4.expert_feed_forward_length", cfg_.n_layers, 0);
        cfg_.n_ff_exp = ff_exp_arr.empty() ? 0 : ff_exp_arr[0];
    } else {
        cfg_.n_ff_exp = 0;
    }

    cfg_.n_head_kv = read_int_or_array(meta, "gemma4.attention.head_count_kv", cfg_.n_layers, 1);
    cfg_.n_ff      = read_int_or_array(meta, "gemma4.feed_forward_length", cfg_.n_layers, 6144);

    cfg_.is_swa.assign(static_cast<size_t>(cfg_.n_layers), true);
    if (meta.has_kv("gemma4.attention.sliding_window_pattern")) {
        const auto & pv = meta.kv("gemma4.attention.sliding_window_pattern");
        if (auto * vb = std::get_if<std::vector<bool>>(&pv.data)) {
            for (int il = 0; il < cfg_.n_layers && il < static_cast<int>(vb->size()); ++il) cfg_.is_swa[il] = (*vb)[il];
        } else if (auto * vu = std::get_if<std::vector<uint64_t>>(&pv.data)) {
            for (int il = 0; il < cfg_.n_layers && il < static_cast<int>(vu->size()); ++il) cfg_.is_swa[il] = ((*vu)[il] != 0);
        } else if (auto * vi = std::get_if<std::vector<int64_t>>(&pv.data)) {
            for (int il = 0; il < cfg_.n_layers && il < static_cast<int>(vi->size()); ++il) cfg_.is_swa[il] = ((*vi)[il] != 0);
        }
    }

    // KV sharing map: layers >= n_layer_kv_from_start reuse the last own-KV layer with the same SWA type
    cfg_.kv_source.resize(static_cast<size_t>(cfg_.n_layers));
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (il < cfg_.n_layer_kv_from_start) {
            cfg_.kv_source[il] = il;
        } else {
            int src = 0;
            for (int j = cfg_.n_layer_kv_from_start - 1; j >= 0; --j) {
                if (cfg_.is_swa[j] == cfg_.is_swa[il]) { src = j; break; }
            }
            cfg_.kv_source[il] = src;
        }
    }

    tok_embd_ = bind_tensor("token_embd.weight");
    cfg_.n_vocab = static_cast<int>(tok_embd_.ne[1]);

    if (cfg_.n_embd_per_layer > 0) {
        per_layer_tok_embd_   = bind_tensor("per_layer_token_embd.weight");
        per_layer_model_proj_ = bind_tensor("per_layer_model_proj.weight");
        per_layer_proj_norm_  = bind_tensor("per_layer_proj_norm.weight");
    }

    output_norm_ = bind_tensor("output_norm.weight");
    output_ = bind_tensor("output.weight", false);
    if (!output_.valid()) {
        output_ = tok_embd_;
        cfg_.tied_embeddings = true;
    } else {
        cfg_.tied_embeddings = false;
    }

    const CpuTensor global_rope_freqs = bind_tensor("rope_freqs.weight", false);

    layers_.resize(static_cast<size_t>(cfg_.n_layers));
    int n_moe_layers = 0;
    for (int il = 0; il < cfg_.n_layers; ++il) {
        auto & L = layers_[il];
        L.is_swa    = cfg_.is_swa[il];
        L.has_kv    = (il < cfg_.n_layer_kv_from_start);
        L.kv_source = cfg_.kv_source[il];
        L.n_head_kv = cfg_.n_head_kv[il];
        L.n_ff      = cfg_.n_ff[il];

        const std::string pfx = "blk." + std::to_string(il) + ".";
        L.attn_norm = bind_tensor(pfx + "attn_norm.weight");
        L.wqkv      = bind_tensor(pfx + "attn_qkv.weight", false);
        if (!L.wqkv.valid()) {
            L.wq = bind_tensor(pfx + "attn_q.weight");
            if (L.has_kv) {
                L.wk = bind_tensor(pfx + "attn_k.weight");
                L.wv = bind_tensor(pfx + "attn_v.weight", false); // optional (k_eq_v full-attn layers)
            }
        }
        L.wo             = bind_tensor(pfx + "attn_output.weight");
        L.attn_q_norm    = bind_tensor(pfx + "attn_q_norm.weight");
        if (L.has_kv) L.attn_k_norm = bind_tensor(pfx + "attn_k_norm.weight");
        L.attn_post_norm = bind_tensor(pfx + "post_attention_norm.weight");

        if (!L.is_swa) {
            L.rope_freqs = bind_tensor(pfx + "rope_freqs.weight", false);
            if (!L.rope_freqs.valid()) L.rope_freqs = global_rope_freqs;
        }

        L.ffn_norm      = bind_tensor(pfx + "ffn_norm.weight");
        L.ffn_gate      = bind_tensor(pfx + "ffn_gate.weight");
        L.ffn_up        = bind_tensor(pfx + "ffn_up.weight");
        L.ffn_down      = bind_tensor(pfx + "ffn_down.weight");
        L.ffn_post_norm = bind_tensor(pfx + "post_ffw_norm.weight");

        L.ffn_gate_inp = bind_tensor(pfx + "ffn_gate_inp.weight", false);
        L.is_moe = L.ffn_gate_inp.valid();
        if (L.is_moe) {
            ++n_moe_layers;
            L.ffn_gate_inp_scale = bind_tensor(pfx + "ffn_gate_inp.scale");
            L.ffn_pre_norm_2     = bind_tensor(pfx + "pre_ffw_norm_2.weight");
            L.ffn_post_norm_1    = bind_tensor(pfx + "post_ffw_norm_1.weight");
            L.ffn_post_norm_2    = bind_tensor(pfx + "post_ffw_norm_2.weight");
            L.ffn_gate_up_exps   = bind_tensor(pfx + "ffn_gate_up_exps.weight", false);
            if (!L.ffn_gate_up_exps.valid()) {
                L.ffn_gate_exps = bind_tensor(pfx + "ffn_gate_exps.weight");
                L.ffn_up_exps   = bind_tensor(pfx + "ffn_up_exps.weight");
            }
            L.ffn_down_exps       = bind_tensor(pfx + "ffn_down_exps.weight");
            L.ffn_down_exps_scale = bind_tensor(pfx + "ffn_down_exps.scale", false);
        }

        if (cfg_.n_embd_per_layer > 0) {
            L.per_layer_inp_gate  = bind_tensor(pfx + "inp_gate.weight");
            L.per_layer_proj      = bind_tensor(pfx + "proj.weight");
            L.per_layer_post_norm = bind_tensor(pfx + "post_norm.weight");
        }

        const CpuTensor sc_t = bind_tensor(pfx + "layer_output_scale.weight", false);
        if (sc_t.valid()) {
            L.has_output_scale = true;
            if (sc_t.dtype == DType::F32) {
                L.layer_output_scale = *static_cast<const float *>(sc_t.ptr);
            } else if (sc_t.dtype == DType::F16) {
                L.layer_output_scale = f16_to_f32(*static_cast<const uint16_t *>(sc_t.ptr));
            } else if (sc_t.dtype == DType::BF16) {
                L.layer_output_scale = bf16_to_f32(*static_cast<const uint16_t *>(sc_t.ptr));
            }
        }
    }

    // Allocate KV caches for own-KV layers [0 .. n_layer_kv_from_start)
    // SWA layers use fixed ring buffer of cfg_.swa_slots; Full layers grow lazily up to cfg_.max_ctx.
    full_kv_cap_ = std::min(cfg_.max_ctx, 2048);
    k_cache_.resize(static_cast<size_t>(cfg_.n_layers));
    v_cache_.resize(static_cast<size_t>(cfg_.n_layers));
    size_t kv_bytes_total = 0;
    for (int il = 0; il < cfg_.n_layer_kv_from_start; ++il) {
        const int hd    = cfg_.is_swa[il] ? cfg_.head_dim_swa : cfg_.head_dim_full;
        const int slots = cfg_.is_swa[il] ? cfg_.swa_slots : full_kv_cap_;
        const size_t bytes = static_cast<size_t>(slots) * cfg_.n_head_kv[il] * kv_head_stride_bytes(hd, cfg_.kv_type);
        k_cache_[il].assign(bytes, 0);
        v_cache_[il].assign(bytes, 0);
        kv_bytes_total += bytes * 2;
    }

    logits_buf_.resize(static_cast<size_t>(cfg_.n_vocab));
    ensure_batch_buffers(1);

    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[gemma4-cpu] Loaded in %.2f s (zero-copy mmap): L=%d (own_kv=%d, moe=%d, ple=%d) H=%d V=%d "
                "head=%d hd_swa=%d hd_full=%d swa=%d ctx=%d (swa_ring=%d, kv=%s %.1f MiB) threads=%d\n",
                load_s, cfg_.n_layers, cfg_.n_layer_kv_from_start, n_moe_layers, cfg_.n_embd_per_layer,
                cfg_.n_embd, cfg_.n_vocab, cfg_.n_head, cfg_.head_dim_swa, cfg_.head_dim_full,
                cfg_.sliding_window, cfg_.max_ctx, cfg_.swa_slots,
                kv_cache_type_name(cfg_.kv_type), kv_bytes_total / (1024.0 * 1024.0), pool_.num_threads());
    std::fflush(stdout);
}

void Gemma4CpuModel::reset_state() {
    for (int il = 0; il < cfg_.n_layer_kv_from_start; ++il) {
        std::memset(k_cache_[il].data(), 0, k_cache_[il].size());
        std::memset(v_cache_[il].data(), 0, v_cache_[il].size());
    }
}

void Gemma4CpuModel::ensure_full_kv_capacity(int needed_pos) {
    if (needed_pos < full_kv_cap_) return;
    if (needed_pos >= cfg_.max_ctx) {
        throw std::runtime_error("Gemma4CpuModel: position " + std::to_string(needed_pos) +
                                 " >= max_ctx " + std::to_string(cfg_.max_ctx));
    }
    int new_cap = full_kv_cap_;
    while (new_cap <= needed_pos) new_cap = std::min(cfg_.max_ctx, new_cap * 2);
    for (int il = 0; il < cfg_.n_layer_kv_from_start; ++il) {
        if (cfg_.is_swa[il]) continue;
        const size_t bytes = static_cast<size_t>(new_cap) * cfg_.n_head_kv[il] * kv_head_stride_bytes(cfg_.head_dim_full, cfg_.kv_type);
        k_cache_[il].resize(bytes, 0);
        v_cache_[il].resize(bytes, 0);
    }
    full_kv_cap_ = new_cap;
}

void Gemma4CpuModel::ensure_batch_buffers(int T) {
    if (T <= max_alloc_T_) return;
    max_alloc_T_ = T;

    const int H      = cfg_.n_embd;
    const int max_hd = std::max(cfg_.head_dim_full, cfg_.head_dim_swa);
    int max_kv = 1, max_ff = 1;
    for (int kv : cfg_.n_head_kv) max_kv = std::max(max_kv, kv);
    for (int ff : cfg_.n_ff)      max_ff = std::max(max_ff, ff);

    cur_.resize(static_cast<size_t>(T) * H);
    x_norm_.resize(static_cast<size_t>(T) * H);
    sub_out_.resize(static_cast<size_t>(T) * H);
    attn_res_.resize(static_cast<size_t>(T) * H);

    q_buf_.resize(static_cast<size_t>(T) * cfg_.n_head * max_hd);
    k_buf_.resize(static_cast<size_t>(T) * max_kv * max_hd);
    v_buf_.resize(static_cast<size_t>(T) * max_kv * max_hd);
    qkv_buf_.resize(static_cast<size_t>(T) * (cfg_.n_head + 2 * max_kv) * max_hd);
    attn_out_.resize(static_cast<size_t>(T) * cfg_.n_head * max_hd);

    ffn_g_.resize(static_cast<size_t>(T) * max_ff);
    ffn_u_.resize(static_cast<size_t>(T) * max_ff);

    if (cfg_.n_expert > 0) {
        ffn_mlp_.resize(static_cast<size_t>(T) * H);
        ffn_moe_.resize(static_cast<size_t>(T) * H);
        router_in_.resize(static_cast<size_t>(T) * H);
        router_logits_.resize(static_cast<size_t>(T) * cfg_.n_expert);
        top_ids_.resize(static_cast<size_t>(T) * cfg_.n_expert_used);
        top_w_.resize(static_cast<size_t>(T) * cfg_.n_expert_used);
    }

    if (cfg_.n_embd_per_layer > 0) {
        const size_t ple_tot = static_cast<size_t>(T) * cfg_.n_layers * cfg_.n_embd_per_layer;
        ple_raw_.resize(ple_tot);
        ple_proj_.resize(ple_tot);
        ple_buf_.resize(ple_tot);
        ple_gate_.resize(static_cast<size_t>(T) * cfg_.n_embd_per_layer);
    }
}

void Gemma4CpuModel::run_forward(const int32_t * tokens,
                                 const int32_t * positions,
                                 int T,
                                 float * out_logits,
                                 bool compute_logits,
                                 bool apply_softcap) {
    if (T <= 0) return;
    int max_pos = positions[0];
    for (int t = 1; t < T; ++t) max_pos = std::max(max_pos, positions[t]);
    ensure_full_kv_capacity(max_pos);
    ensure_batch_buffers(T);

    const int H = cfg_.n_embd;
    const bool prof = (std::getenv("ROOKERY_PROFILE") != nullptr);
    double ms_emb = 0, ms_qkv = 0, ms_attn = 0, ms_wo = 0, ms_dense = 0, ms_rtr = 0, ms_moe = 0, ms_ple = 0, ms_norm = 0, ms_head = 0;
    auto mark = [&](double & acc, auto fn) {
        if (!prof) { fn(); return; }
        const auto a = std::chrono::steady_clock::now();
        fn();
        const auto b = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(b - a).count();
    };

    // 1. Token embedding * sqrt(n_embd)
    mark(ms_emb, [&]() {
        embedding_lookup(tok_embd_, tokens, cur_.data(), H, T);
        scale_inplace(cur_.data(), static_cast<size_t>(T) * H, std::sqrt(static_cast<float>(H)));

        // 2. Per-Layer Embeddings (PLE, E2B/E4B)
        if (cfg_.n_embd_per_layer > 0) {
            const int ple_total_dim = cfg_.n_layers * cfg_.n_embd_per_layer;
            embedding_lookup(per_layer_tok_embd_, tokens, ple_raw_.data(), ple_total_dim, T);
            linear(pool_, per_layer_model_proj_, cur_.data(), nullptr, nullptr, ple_proj_.data(), T);
            gemma4_ple_combine(ple_proj_.data(), ple_raw_.data(),
                               static_cast<const float *>(per_layer_proj_norm_.ptr),
                               ple_buf_.data(), T, cfg_.n_layers, cfg_.n_embd_per_layer,
                               cfg_.rms_eps, H);
        }
    });

    int T_cur = T;
    int ple_tok_off = 0;
    const int32_t * pos_cur = positions;
    int32_t last_pos = positions[T - 1];

    // Parallel rmsnorm + residual helpers for prefill (T >= 8) with zero barrier overhead on decode (T == 1)
    auto rmsnorm_par = [&](const float * src, const float * w, float * dst, int rows) {
        if (rows >= 8 && pool_.num_threads() > 1) {
            pool_.parallel_for(rows, [&](int t0, int t1, int) {
                rmsnorm(src + static_cast<size_t>(t0) * H, w, dst + static_cast<size_t>(t0) * H, H, t1 - t0, cfg_.rms_eps);
            });
        } else {
            rmsnorm(src, w, dst, H, rows, cfg_.rms_eps);
        }
    };
    auto post_norm_add_par = [&](float * sub, const float * w, const float * base, float * dst, int rows) {
        if (rows >= 8 && pool_.num_threads() > 1) {
            pool_.parallel_for(rows, [&](int t0, int t1, int) {
                const int n = t1 - t0;
                const size_t off = static_cast<size_t>(t0) * H;
                rmsnorm(sub + off, w, sub + off, H, n, cfg_.rms_eps);
                const size_t cnt = static_cast<size_t>(n) * H;
                for (size_t i = 0; i < cnt; ++i) dst[off + i] = base[off + i] + sub[off + i];
            });
        } else {
            rmsnorm(sub, w, sub, H, rows, cfg_.rms_eps);
            const size_t cnt = static_cast<size_t>(rows) * H;
            for (size_t i = 0; i < cnt; ++i) dst[i] = base[i] + sub[i];
        }
    };

    // 3. Transformer blocks
    for (int il = 0; il < cfg_.n_layers; ++il) {
        // Prefill pruning for shared-KV models: layers >= n_layer_kv_from_start do not write KV!
        if (il == cfg_.n_layer_kv_from_start && cfg_.n_layer_kv_from_start < cfg_.n_layers && T_cur > 1) {
            if (!compute_logits) break;
            std::memcpy(cur_.data(), cur_.data() + static_cast<size_t>(T_cur - 1) * H, static_cast<size_t>(H) * sizeof(float));
            ple_tok_off = T_cur - 1;
            T_cur = 1;
            pos_cur = &last_pos;
        }

        const auto & L = layers_[il];
        const int hd       = L.is_swa ? cfg_.head_dim_swa : cfg_.head_dim_full;
        const int rope_dim = L.is_swa ? cfg_.rope_dim_swa : cfg_.rope_dim_full;
        const float fbase  = L.is_swa ? cfg_.rope_freq_base_swa : cfg_.rope_freq_base_full;
        const int n_kv     = L.n_head_kv;
        const int kv_src   = L.kv_source;
        const int kv_slots = L.is_swa ? cfg_.swa_slots : full_kv_cap_;
        const int swa_win  = L.is_swa ? cfg_.sliding_window : 0;

        // ---- Attention ----
        mark(ms_norm, [&]() {
            rmsnorm_par(cur_.data(), static_cast<const float *>(L.attn_norm.ptr), x_norm_.data(), T_cur);
        });

        mark(ms_qkv, [&]() {
            if (L.wqkv.valid()) {
                const int q_dim = cfg_.n_head * hd;
                const int k_dim = n_kv * hd;
                const int v_dim = n_kv * hd;
                const int tot_d = q_dim + k_dim + v_dim;
                linear(pool_, L.wqkv, x_norm_.data(), nullptr, nullptr, qkv_buf_.data(), T_cur);
                for (int t = 0; t < T_cur; ++t) {
                    const float * row = qkv_buf_.data() + static_cast<size_t>(t) * tot_d;
                    std::memcpy(q_buf_.data() + static_cast<size_t>(t) * q_dim, row, static_cast<size_t>(q_dim) * sizeof(float));
                    if (L.has_kv) {
                        std::memcpy(k_buf_.data() + static_cast<size_t>(t) * k_dim, row + q_dim, static_cast<size_t>(k_dim) * sizeof(float));
                        std::memcpy(v_buf_.data() + static_cast<size_t>(t) * v_dim, row + q_dim + k_dim, static_cast<size_t>(v_dim) * sizeof(float));
                    }
                }
            } else if (L.has_kv) {
                if (L.wv.valid()) {
                    const LinearSpec specs[3] = {
                        {&L.wq, q_buf_.data()},
                        {&L.wk, k_buf_.data()},
                        {&L.wv, v_buf_.data()},
                    };
                    linear_multi(pool_, specs, 3, x_norm_.data(), nullptr, nullptr, T_cur);
                } else {
                    // k_eq_v alternative attention (full-attn layers in Gemma 4 26B-A4B): V = raw K before K-norm/RoPE
                    const LinearSpec specs[2] = {
                        {&L.wq, q_buf_.data()},
                        {&L.wk, k_buf_.data()},
                    };
                    linear_multi(pool_, specs, 2, x_norm_.data(), nullptr, nullptr, T_cur);
                    std::memcpy(v_buf_.data(), k_buf_.data(), static_cast<size_t>(T_cur) * n_kv * hd * sizeof(float));
                }
            } else {
                linear(pool_, L.wq, x_norm_.data(), nullptr, nullptr, q_buf_.data(), T_cur);
            }
        });

        mark(ms_attn, [&]() {
            const float * rope_f = L.rope_freqs.valid() ? static_cast<const float *>(L.rope_freqs.ptr) : nullptr;
            if (T_cur >= 8 && pool_.num_threads() > 1) {
                pool_.parallel_for(T_cur, [&](int t0, int t1, int) {
                    const int nt = t1 - t0;
                    gemma4_qk_norm_rope(q_buf_.data() + static_cast<size_t>(t0) * cfg_.n_head * hd,
                                        static_cast<const float *>(L.attn_q_norm.ptr),
                                        rope_f, pos_cur + t0, nt, cfg_.n_head, hd, rope_dim, fbase, cfg_.rms_eps);
                    if (L.has_kv) {
                        gemma4_qk_norm_rope(k_buf_.data() + static_cast<size_t>(t0) * n_kv * hd,
                                            static_cast<const float *>(L.attn_k_norm.ptr),
                                            rope_f, pos_cur + t0, nt, n_kv, hd, rope_dim, fbase, cfg_.rms_eps);
                        gemma4_vnorm_kv_write(k_buf_.data() + static_cast<size_t>(t0) * n_kv * hd,
                                              v_buf_.data() + static_cast<size_t>(t0) * n_kv * hd,
                                              k_cache_[il].data(), v_cache_[il].data(), cfg_.kv_type,
                                              pos_cur + t0, nt, n_kv, hd, kv_slots, L.is_swa, cfg_.rms_eps);
                    }
                });
            } else {
                gemma4_qk_norm_rope(q_buf_.data(), static_cast<const float *>(L.attn_q_norm.ptr),
                                    rope_f, pos_cur, T_cur, cfg_.n_head, hd, rope_dim, fbase, cfg_.rms_eps);

                if (L.has_kv) {
                    gemma4_qk_norm_rope(k_buf_.data(), static_cast<const float *>(L.attn_k_norm.ptr),
                                        rope_f, pos_cur, T_cur, n_kv, hd, rope_dim, fbase, cfg_.rms_eps);
                    gemma4_vnorm_kv_write(k_buf_.data(), v_buf_.data(),
                                          k_cache_[il].data(), v_cache_[il].data(), cfg_.kv_type,
                                          pos_cur, T_cur, n_kv, hd, kv_slots, L.is_swa, cfg_.rms_eps);
                }
            }

            gemma4_gqa_attn(pool_, q_buf_.data(),
                            k_cache_[kv_src].data(), v_cache_[kv_src].data(), cfg_.kv_type,
                            pos_cur, attn_out_.data(),
                            T_cur, cfg_.n_head, n_kv, hd, kv_slots, swa_win);
        });

        mark(ms_wo, [&]() {
            linear(pool_, L.wo, attn_out_.data(), nullptr, nullptr, sub_out_.data(), T_cur);
        });
        mark(ms_norm, [&]() {
            post_norm_add_par(sub_out_.data(), static_cast<const float *>(L.attn_post_norm.ptr), cur_.data(), attn_res_.data(), T_cur);
        });

        // ---- Feed-Forward Network ----
        if (L.is_moe) {
            // 1. Dense MLP branch
            mark(ms_dense, [&]() {
                rmsnorm_par(attn_res_.data(), static_cast<const float *>(L.ffn_norm.ptr), x_norm_.data(), T_cur);
                if (T_cur == 1 && (L.n_ff & 31) == 0 && (L.ffn_down.dtype != DType::Q6_K || (L.n_ff & 255) == 0)) {
                    dense_ffn_geglu_t1(pool_, L.ffn_gate, L.ffn_up, L.ffn_down,
                                       x_norm_.data(), nullptr, nullptr, ffn_mlp_.data());
                } else {
                    const LinearSpec specs[2] = {{&L.ffn_gate, ffn_g_.data()}, {&L.ffn_up, ffn_u_.data()}};
                    linear_multi(pool_, specs, 2, x_norm_.data(), nullptr, nullptr, T_cur);
                    geglu_tanh_mul_mt(pool_, ffn_g_.data(), ffn_u_.data(), static_cast<size_t>(T_cur) * L.n_ff);
                    linear(pool_, L.ffn_down, ffn_g_.data(), nullptr, nullptr, ffn_mlp_.data(), T_cur);
                }
                rmsnorm_par(ffn_mlp_.data(), static_cast<const float *>(L.ffn_post_norm_1.ptr), ffn_mlp_.data(), T_cur);
            });

            // 2. Router (operates on attn_res)
            mark(ms_rtr, [&]() {
                rmsnorm_unweighted(attn_res_.data(), router_in_.data(), H, T_cur, cfg_.rms_eps);
                const float inv_sqrt_h = 1.0f / std::sqrt(static_cast<float>(H));
                const float * gsc = static_cast<const float *>(L.ffn_gate_inp_scale.ptr);
                for (int t = 0; t < T_cur; ++t) {
                    float * r = router_in_.data() + static_cast<size_t>(t) * H;
                    for (int d = 0; d < H; ++d) r[d] *= inv_sqrt_h * gsc[d];
                }
                linear(pool_, L.ffn_gate_inp, router_in_.data(), nullptr, nullptr, router_logits_.data(), T_cur);
                moe_route_softmax(router_logits_.data(), top_ids_.data(), top_w_.data(),
                                  T_cur, cfg_.n_expert, cfg_.n_expert_used, true, 1.0f);
            });

            // 3. Expert FFN branch
            mark(ms_moe, [&]() {
                rmsnorm_par(attn_res_.data(), static_cast<const float *>(L.ffn_pre_norm_2.ptr), x_norm_.data(), T_cur);
                const float * down_sc = L.ffn_down_exps_scale.valid()
                                      ? static_cast<const float *>(L.ffn_down_exps_scale.ptr)
                                      : nullptr;
                gemma4_moe_ffn(pool_, L.ffn_gate_up_exps, L.ffn_gate_exps, L.ffn_up_exps,
                               L.ffn_down_exps, down_sc,
                               x_norm_.data(), nullptr, nullptr,
                               top_ids_.data(), top_w_.data(),
                               ffn_moe_.data(), T_cur, cfg_.n_expert_used);
            });

            // 4. Combine + post_ffw_norm + residual (fused single parallel sweep for prefill!)
            mark(ms_norm, [&]() {
                if (T_cur >= 8 && pool_.num_threads() > 1) {
                    const float * pn2 = static_cast<const float *>(L.ffn_post_norm_2.ptr);
                    const float * pn  = static_cast<const float *>(L.ffn_post_norm.ptr);
                    pool_.parallel_for(T_cur, [&](int t0, int t1, int) {
                        const int n = t1 - t0;
                        const size_t off = static_cast<size_t>(t0) * H;
                        const size_t cnt = static_cast<size_t>(n) * H;
                        rmsnorm(ffn_moe_.data() + off, pn2, ffn_moe_.data() + off, H, n, cfg_.rms_eps);
                        add_inplace(ffn_mlp_.data() + off, ffn_moe_.data() + off, cnt);
                        rmsnorm(ffn_mlp_.data() + off, pn, sub_out_.data() + off, H, n, cfg_.rms_eps);
                        for (size_t i = 0; i < cnt; ++i) cur_[off + i] = attn_res_[off + i] + sub_out_[off + i];
                    });
                } else {
                    rmsnorm(ffn_moe_.data(), static_cast<const float *>(L.ffn_post_norm_2.ptr), ffn_moe_.data(), H, T_cur, cfg_.rms_eps);
                    add_inplace(ffn_mlp_.data(), ffn_moe_.data(), static_cast<size_t>(T_cur) * H);
                    rmsnorm(ffn_mlp_.data(), static_cast<const float *>(L.ffn_post_norm.ptr), sub_out_.data(), H, T_cur, cfg_.rms_eps);
                    for (size_t i = 0; i < static_cast<size_t>(T_cur) * H; ++i) cur_[i] = attn_res_[i] + sub_out_[i];
                }
            });
        } else {
            mark(ms_dense, [&]() {
                rmsnorm_par(attn_res_.data(), static_cast<const float *>(L.ffn_norm.ptr), x_norm_.data(), T_cur);
                if (T_cur == 1 && (L.n_ff & 31) == 0 && (L.ffn_down.dtype != DType::Q6_K || (L.n_ff & 255) == 0)) {
                    dense_ffn_geglu_t1(pool_, L.ffn_gate, L.ffn_up, L.ffn_down,
                                       x_norm_.data(), nullptr, nullptr, sub_out_.data());
                } else {
                    const LinearSpec specs[2] = {{&L.ffn_gate, ffn_g_.data()}, {&L.ffn_up, ffn_u_.data()}};
                    linear_multi(pool_, specs, 2, x_norm_.data(), nullptr, nullptr, T_cur);
                    geglu_tanh_mul_mt(pool_, ffn_g_.data(), ffn_u_.data(), static_cast<size_t>(T_cur) * L.n_ff);
                    linear(pool_, L.ffn_down, ffn_g_.data(), nullptr, nullptr, sub_out_.data(), T_cur);
                }
            });
            mark(ms_norm, [&]() {
                post_norm_add_par(sub_out_.data(), static_cast<const float *>(L.ffn_post_norm.ptr), attn_res_.data(), cur_.data(), T_cur);
            });
        }

        // ---- Per-Layer Embedding (PLE) Gating Block ----
        if (cfg_.n_embd_per_layer > 0) {
            mark(ms_ple, [&]() {
                linear(pool_, L.per_layer_inp_gate, cur_.data(), nullptr, nullptr, ple_gate_.data(), T_cur);
                for (int ti = 0; ti < T_cur; ++ti) {
                    const float * ple_il = ple_buf_.data() +
                        (static_cast<size_t>(ple_tok_off + ti) * cfg_.n_layers + il) * cfg_.n_embd_per_layer;
                    geglu_tanh_mul(ple_gate_.data() + static_cast<size_t>(ti) * cfg_.n_embd_per_layer,
                                   ple_il, static_cast<size_t>(cfg_.n_embd_per_layer));
                }
                linear(pool_, L.per_layer_proj, ple_gate_.data(), nullptr, nullptr, sub_out_.data(), T_cur);
                post_norm_add_par(sub_out_.data(), static_cast<const float *>(L.per_layer_post_norm.ptr), cur_.data(), cur_.data(), T_cur);
            });
        }

        // ---- Layer Output Scale ----
        if (L.has_output_scale) {
            mark(ms_norm, [&]() {
                if (T_cur >= 8 && pool_.num_threads() > 1) {
                    pool_.parallel_for(T_cur, [&](int t0, int t1, int) {
                        scale_inplace(cur_.data() + static_cast<size_t>(t0) * H, static_cast<size_t>(t1 - t0) * H, L.layer_output_scale);
                    });
                } else {
                    scale_inplace(cur_.data(), static_cast<size_t>(T_cur) * H, L.layer_output_scale);
                }
            });
        }
    }

    if (compute_logits && out_logits != nullptr) {
        // 4. Final norm + LM head + final logit softcapping (last token only)
        mark(ms_head, [&]() {
            const float * last_h = cur_.data() + static_cast<size_t>(T_cur - 1) * H;
            rmsnorm(last_h, static_cast<const float *>(output_norm_.ptr), x_norm_.data(), H, 1, cfg_.rms_eps);
            linear(pool_, output_, x_norm_.data(), nullptr, nullptr, out_logits, 1);
            if (apply_softcap && cfg_.final_logit_softcap > 0.0f) {
                if (cfg_.n_vocab >= 32768 && pool_.num_threads() > 1) {
                    pool_.parallel_for_blocked(cfg_.n_vocab, 1024, [&](int i0, int i1, int) {
                        softcap_tanh(out_logits + i0, i1 - i0, cfg_.final_logit_softcap);
                    });
                } else {
                    softcap_tanh(out_logits, cfg_.n_vocab, cfg_.final_logit_softcap);
                }
            }
        });
    }

    if (prof) {
        const double tot = ms_emb + ms_qkv + ms_attn + ms_wo + ms_dense + ms_rtr + ms_moe + ms_ple + ms_norm + ms_head;
        std::printf("  [PROF G4 T=%d] emb=%.2f qkv=%.2f attn=%.2f wo=%.2f dense=%.2f rtr=%.2f moe=%.2f ple=%.2f norm=%.2f head=%.2f | tot=%.2f ms (%.1f tok/s)\n",
                    T, ms_emb, ms_qkv, ms_attn, ms_wo, ms_dense, ms_rtr, ms_moe, ms_ple, ms_norm, ms_head,
                    tot, (tot > 0.0) ? (T * 1000.0 / tot) : 0.0);
    }
}

void Gemma4CpuModel::forward(const std::vector<int32_t> & tokens,
                             const std::vector<int32_t> & positions,
                             std::vector<float> & out_logits,
                             bool compute_logits) {
    if (tokens.size() != positions.size()) {
        throw std::runtime_error("Gemma4CpuModel::forward: tokens.size() != positions.size()");
    }
    if (compute_logits) {
        out_logits.resize(static_cast<size_t>(cfg_.n_vocab));
    }
    run_forward(tokens.data(), positions.data(), static_cast<int>(tokens.size()),
                compute_logits ? out_logits.data() : nullptr, compute_logits, true);
}

int32_t Gemma4CpuModel::forward_decode_greedy(int32_t token, int32_t pos) {
    // softcap_tanh is strictly monotonically increasing, so argmax is identical without it.
    run_forward(&token, &pos, 1, logits_buf_.data(), true, false);
    return argmax_f32(logits_buf_.data(), cfg_.n_vocab);
}

void Gemma4CpuModel::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    out_logits.resize(static_cast<size_t>(cfg_.n_vocab));
    run_forward(&token, &pos, 1, out_logits.data(), true, true);
}

} // namespace rookery::cpu
