#include "cpu/qwen2_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace rookery::cpu {

CpuTensor Qwen2CpuModel::bind_tensor(const std::string & name, bool required) const {
    auto loc = sg_.find_tensor(name);
    if (!loc.info) {
        if (required) throw std::runtime_error("Qwen2CpuModel: missing tensor " + name);
        return CpuTensor{};
    }
    CpuTensor t;
    t.ptr   = loc.data();
    t.dtype = loc.info->dtype;
    for (int i = 0; i < 4; ++i) t.ne[i] = loc.info->ne[i];
    t.nbytes = tensor_nbytes(t.dtype, t.ne[0], t.ne[1], t.ne[2], t.ne[3]);
    return t;
}

void Qwen2CpuModel::load(const std::string & gguf_path,
                         int max_ctx,
                         KvCacheType kv_type,
                         int n_threads) {
    const auto t0 = std::chrono::steady_clock::now();
    pool_.init(n_threads);
    print_cpu_caps(pool_.num_threads());

    sg_ = ShardedGguf::open(gguf_path);
    const std::string arch = sg_.get_str("general.architecture");
    if (arch != "qwen2") {
        throw std::runtime_error("Expected qwen2 GGUF, got: " + arch);
    }
    const std::string pfx = arch + ".";

    cfg_.n_layers       = static_cast<int>(sg_.get_u64(pfx + "block_count", 28));
    cfg_.n_embd         = static_cast<int>(sg_.get_u64(pfx + "embedding_length", 1536));
    cfg_.n_ff           = static_cast<int>(sg_.get_u64(pfx + "feed_forward_length", 8960));
    cfg_.n_head         = static_cast<int>(sg_.get_u64(pfx + "attention.head_count", 12));
    cfg_.n_head_kv      = static_cast<int>(sg_.get_u64(pfx + "attention.head_count_kv", 2));
    cfg_.head_dim       = static_cast<int>(sg_.get_u64(pfx + "attention.key_length", cfg_.n_embd / cfg_.n_head));
    cfg_.rope_freq_base = static_cast<float>(sg_.get_f64(pfx + "rope.freq_base", 1000000.0));
    cfg_.rms_eps        = static_cast<float>(sg_.get_f64(pfx + "attention.layer_norm_rms_epsilon", 1e-6));
    cfg_.max_ctx        = max_ctx;
    cfg_.kv_type        = kv_type;

    if (cfg_.head_dim != 128) {
        throw std::runtime_error("Qwen2CpuModel currently expects head_dim=128, got: " + std::to_string(cfg_.head_dim));
    }

    tok_embd_    = bind_tensor("token_embd.weight");
    output_norm_ = bind_tensor("output_norm.weight");
    output_      = bind_tensor("output.weight", false);
    if (!output_.valid()) output_ = tok_embd_;
    cfg_.n_vocab = static_cast<int>(output_.ne[1]);

    layers_.clear();
    layers_.resize(static_cast<size_t>(cfg_.n_layers));
    k_cache_q8_.resize(static_cast<size_t>(cfg_.n_layers));
    v_cache_q8_.resize(static_cast<size_t>(cfg_.n_layers));
    k_cache_q4_.resize(static_cast<size_t>(cfg_.n_layers));
    v_cache_q4_.resize(static_cast<size_t>(cfg_.n_layers));
    k_cache_f16_.resize(static_cast<size_t>(cfg_.n_layers));
    v_cache_f16_.resize(static_cast<size_t>(cfg_.n_layers));

    size_t kv_bytes_total = 0;
    const size_t kv_slots = static_cast<size_t>(cfg_.n_head_kv) * static_cast<size_t>(cfg_.max_ctx);

    for (int il = 0; il < cfg_.n_layers; ++il) {
        auto & L = layers_[il];
        const std::string p = "blk." + std::to_string(il) + ".";

        L.attn_norm = bind_tensor(p + "attn_norm.weight");
        L.ffn_norm  = bind_tensor(p + "ffn_norm.weight");
        L.wq        = bind_tensor(p + "attn_q.weight");
        L.wk        = bind_tensor(p + "attn_k.weight");
        L.wv        = bind_tensor(p + "attn_v.weight");
        L.wo        = bind_tensor(p + "attn_output.weight");
        L.bq        = bind_tensor(p + "attn_q.bias", false);
        L.bk        = bind_tensor(p + "attn_k.bias", false);
        L.bv        = bind_tensor(p + "attn_v.bias", false);
        L.ffn_gate  = bind_tensor(p + "ffn_gate.weight");
        L.ffn_up    = bind_tensor(p + "ffn_up.weight");
        L.ffn_down  = bind_tensor(p + "ffn_down.weight");

        if (kv_type == KvCacheType::Q4_0) {
            k_cache_q4_[il].resize(kv_slots);
            v_cache_q4_[il].resize(kv_slots);
            kv_bytes_total += 2 * kv_slots * sizeof(CpuGqaKv128Q4Pos);
        } else if (kv_type == KvCacheType::Q8_0) {
            k_cache_q8_[il].resize(kv_slots);
            v_cache_q8_[il].resize(kv_slots);
            kv_bytes_total += 2 * kv_slots * sizeof(CpuGqaKv128Q8Pos);
        } else {
            k_cache_f16_[il].resize(kv_slots * static_cast<size_t>(cfg_.head_dim));
            v_cache_f16_[il].resize(kv_slots * static_cast<size_t>(cfg_.head_dim));
            kv_bytes_total += 2 * kv_slots * static_cast<size_t>(cfg_.head_dim) * sizeof(uint16_t);
        }
    }

    logits_buf_.resize(static_cast<size_t>(cfg_.n_vocab));
    ensure_batch_buffers(128);

    const auto t1 = std::chrono::steady_clock::now();
    std::printf("[Qwen2-CPU] Loaded %s (%d layers, n_embd=%d, n_ff=%d, heads=%d/%d, d=%d, vocab=%d, ctx=%d, kv=%s %.1f MiB, mmap=%.2f GiB, threads=%d) in %.2f s\n",
                sg_.get_str("general.name", "qwen2").c_str(),
                cfg_.n_layers, cfg_.n_embd, cfg_.n_ff,
                cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim, cfg_.n_vocab,
                cfg_.max_ctx, kv_cache_type_name(kv_type),
                static_cast<double>(kv_bytes_total) / (1024.0 * 1024.0),
                static_cast<double>(sg_.total_bytes()) / (1024.0 * 1024.0 * 1024.0),
                pool_.num_threads(),
                std::chrono::duration<double>(t1 - t0).count());
}

void Qwen2CpuModel::ensure_batch_buffers(int T) {
    if (T <= max_alloc_T_) return;
    max_alloc_T_ = T;

    const int q_dim  = cfg_.n_head * cfg_.head_dim;
    const int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
    const int max_k  = std::max({cfg_.n_embd, q_dim, cfg_.n_ff});

    cur_.resize(static_cast<size_t>(T) * cfg_.n_embd);
    x_norm_.resize(static_cast<size_t>(T) * cfg_.n_embd);
    sub_out_.resize(static_cast<size_t>(T) * cfg_.n_embd);
    x_q8_.resize(static_cast<size_t>(T) * (max_k >> 5));
    x_q8k_.resize(static_cast<size_t>(T) * (max_k >> 8));

    q_buf_.resize(static_cast<size_t>(T) * q_dim);
    k_buf_.resize(static_cast<size_t>(T) * kv_dim);
    v_buf_.resize(static_cast<size_t>(T) * kv_dim);
    attn_out_.resize(static_cast<size_t>(T) * q_dim);

    ffn_g_.resize(static_cast<size_t>(T) * cfg_.n_ff);
    ffn_u_.resize(static_cast<size_t>(T) * cfg_.n_ff);
}

void Qwen2CpuModel::run_forward(const int32_t * tokens, const int32_t * positions, int T, float * out_logits) {
    ensure_batch_buffers(T);

    const int   q_dim      = cfg_.n_head * cfg_.head_dim;
    const int   kv_dim     = cfg_.n_head_kv * cfg_.head_dim;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));

    auto dt_needs_q8 = [](DType dt) -> bool {
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
        return dt == DType::Q8_0 || dt == DType::Q5_0 || dt == DType::Q4_0;
#else
        return dt == DType::Q8_0;
#endif
    };
    auto dt_needs_q8k = [](DType dt) -> bool {
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
        return dt == DType::Q6_K || dt == DType::Q4_K || dt == DType::Q5_K;
#else
        return dt == DType::Q6_K;
#endif
    };
#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512VNNI__)
    const bool use_direct = (T < 12);
#else
    const bool use_direct = true;
#endif

    const bool prof = (std::getenv("ROOKERY_PROFILE") != nullptr);
    double ms_norm = 0, ms_qkv = 0, ms_attn = 0, ms_wo = 0, ms_ffn = 0, ms_head = 0;
    auto mark = [&](double & acc, auto fn) {
        if (!prof) { fn(); return; }
        const auto a = std::chrono::steady_clock::now();
        fn();
        const auto b = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(b - a).count();
    };

    embedding_lookup(tok_embd_, tokens, cur_.data(), cfg_.n_embd, T);

    bool attn_norm_ready = false;
    for (int il = 0; il < cfg_.n_layers; ++il) {
        const auto & L = layers_[il];

        if (!attn_norm_ready) {
            mark(ms_norm, [&]() {
                rmsnorm(cur_.data(), static_cast<const float *>(L.attn_norm.ptr), x_norm_.data(), cfg_.n_embd, T, cfg_.rms_eps);
            });
        }
        attn_norm_ready = false;

        mark(ms_qkv, [&]() {
            const bool need_q8  = use_direct && (dt_needs_q8(L.wq.dtype)  || dt_needs_q8(L.wk.dtype)  || dt_needs_q8(L.wv.dtype));
            const bool need_q8k = use_direct && (dt_needs_q8k(L.wq.dtype) || dt_needs_q8k(L.wk.dtype) || dt_needs_q8k(L.wv.dtype));
            if (need_q8)  quantize_rows_q8_0(pool_, x_norm_.data(), x_q8_.data(),  T, cfg_.n_embd);
            if (need_q8k) quantize_rows_q8_k(pool_, x_norm_.data(), x_q8k_.data(), T, cfg_.n_embd);

            const LinearSpec specs[3] = {
                {&L.wq, q_buf_.data()},
                {&L.wk, k_buf_.data()},
                {&L.wv, v_buf_.data()},
            };
            linear_multi(pool_, specs, 3, x_norm_.data(),
                         need_q8 ? x_q8_.data() : nullptr,
                         need_q8k ? x_q8k_.data() : nullptr, T);
            if (L.bq.valid()) add_bias_rows(q_buf_.data(), static_cast<const float *>(L.bq.ptr), T, q_dim);
            if (L.bk.valid()) add_bias_rows(k_buf_.data(), static_cast<const float *>(L.bk.ptr), T, kv_dim);
            if (L.bv.valid()) add_bias_rows(v_buf_.data(), static_cast<const float *>(L.bv.ptr), T, kv_dim);
        });

        mark(ms_attn, [&]() {
            neox_rope_qk(q_buf_.data(), k_buf_.data(), positions,
                         T, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim, cfg_.rope_freq_base);
            void * k_ptr = (cfg_.kv_type == KvCacheType::Q4_0) ? static_cast<void *>(k_cache_q4_[il].data())
                         : (cfg_.kv_type == KvCacheType::Q8_0) ? static_cast<void *>(k_cache_q8_[il].data())
                                                               : static_cast<void *>(k_cache_f16_[il].data());
            void * v_ptr = (cfg_.kv_type == KvCacheType::Q4_0) ? static_cast<void *>(v_cache_q4_[il].data())
                         : (cfg_.kv_type == KvCacheType::Q8_0) ? static_cast<void *>(v_cache_q8_[il].data())
                                                               : static_cast<void *>(v_cache_f16_[il].data());
            gqa_kv_write_d128(k_buf_.data(), v_buf_.data(), positions,
                              k_ptr, v_ptr, cfg_.kv_type, T, cfg_.n_head_kv, cfg_.max_ctx);
            gqa_attn_d128(pool_, q_buf_.data(), k_ptr, v_ptr, cfg_.kv_type,
                          positions, attn_out_.data(),
                          T, cfg_.n_head, cfg_.n_head_kv, cfg_.max_ctx, attn_scale);
        });

        mark(ms_wo, [&]() {
            linear(pool_, L.wo, attn_out_.data(), nullptr, nullptr, sub_out_.data(), T);
        });

        mark(ms_norm, [&]() {
            add_rmsnorm(cur_.data(), sub_out_.data(), static_cast<const float *>(L.ffn_norm.ptr), x_norm_.data(), cfg_.n_embd, T, cfg_.rms_eps);
        });

        mark(ms_ffn, [&]() {
            const bool need_q8  = use_direct && (dt_needs_q8(L.ffn_gate.dtype)  || dt_needs_q8(L.ffn_up.dtype));
            const bool need_q8k = use_direct && (dt_needs_q8k(L.ffn_gate.dtype) || dt_needs_q8k(L.ffn_up.dtype));
            if (need_q8)  quantize_rows_q8_0(pool_, x_norm_.data(), x_q8_.data(),  T, cfg_.n_embd);
            if (need_q8k) quantize_rows_q8_k(pool_, x_norm_.data(), x_q8k_.data(), T, cfg_.n_embd);

            if (T == 1 && (cfg_.n_ff % 256) == 0) {
                dense_ffn_swiglu_t1(pool_, L.ffn_gate, L.ffn_up, L.ffn_down,
                                    x_norm_.data(),
                                    need_q8 ? x_q8_.data() : nullptr,
                                    need_q8k ? x_q8k_.data() : nullptr,
                                    sub_out_.data());
            } else {
                const LinearSpec specs[2] = {{&L.ffn_gate, ffn_g_.data()}, {&L.ffn_up, ffn_u_.data()}};
                linear_multi(pool_, specs, 2, x_norm_.data(),
                             need_q8 ? x_q8_.data() : nullptr,
                             need_q8k ? x_q8k_.data() : nullptr, T);
                silu_mul_mt(pool_, ffn_g_.data(), ffn_u_.data(), ffn_g_.data(), static_cast<size_t>(T) * cfg_.n_ff);
                linear(pool_, L.ffn_down, ffn_g_.data(), nullptr, nullptr, sub_out_.data(), T);
            }
        });

        if (il + 1 < cfg_.n_layers) {
            mark(ms_norm, [&]() {
                add_rmsnorm(cur_.data(), sub_out_.data(), static_cast<const float *>(layers_[il + 1].attn_norm.ptr),
                            x_norm_.data(), cfg_.n_embd, T, cfg_.rms_eps);
            });
            attn_norm_ready = true;
        } else {
            mark(ms_norm, [&]() {
                add_inplace(cur_.data(), sub_out_.data(), static_cast<size_t>(T) * cfg_.n_embd);
            });
        }
    }

    mark(ms_head, [&]() {
        const float * last_h = cur_.data() + static_cast<size_t>(T - 1) * cfg_.n_embd;
        rmsnorm(last_h, static_cast<const float *>(output_norm_.ptr), x_norm_.data(), cfg_.n_embd, 1, cfg_.rms_eps);
        linear(pool_, output_, x_norm_.data(), nullptr, nullptr, out_logits, 1);
    });

    if (prof) {
        const double tot = ms_norm + ms_qkv + ms_attn + ms_wo + ms_ffn + ms_head;
        std::printf("  [PROF Qwen2-CPU T=%d] norm=%.2f qkv=%.2f attn=%.2f wo=%.2f ffn=%.2f head=%.2f | tot=%.2f ms (%.1f tok/s)\n",
                    T, ms_norm, ms_qkv, ms_attn, ms_wo, ms_ffn, ms_head, tot, (tot > 0.0) ? (T * 1000.0 / tot) : 0.0);
    }
}

void Qwen2CpuModel::forward(const std::vector<int32_t> & tokens,
                            const std::vector<int32_t> & positions,
                            std::vector<float> & out_logits) {
    const int T = static_cast<int>(tokens.size());
    if (T <= 0 || positions.size() != tokens.size()) {
        throw std::runtime_error("Qwen2CpuModel::forward invalid T");
    }
    out_logits.resize(static_cast<size_t>(cfg_.n_vocab));
    run_forward(tokens.data(), positions.data(), T, out_logits.data());
}

int32_t Qwen2CpuModel::forward_decode_greedy(int32_t token, int32_t pos) {
    run_forward(&token, &pos, 1, logits_buf_.data());
    return argmax_f32(logits_buf_.data(), cfg_.n_vocab);
}

void Qwen2CpuModel::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    out_logits.resize(static_cast<size_t>(cfg_.n_vocab));
    run_forward(&token, &pos, 1, out_logits.data());
}

} // namespace rookery::cpu
