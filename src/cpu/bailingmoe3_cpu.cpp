#include "cpu/bailingmoe3_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <variant>

namespace rookery::cpu {

namespace {

template <typename T>
std::vector<T> read_num_arr(const ShardedGguf & sg, const std::string & key, size_t n_def, T def_val) {
    std::vector<T> out(n_def, def_val);
    if (!sg.has_kv(key)) return out;
    const auto & kv = sg.kv(key);
    if (auto * v = std::get_if<std::vector<uint64_t>>(&kv.data)) {
        out.resize(v->size());
        for (size_t i = 0; i < v->size(); ++i) out[i] = static_cast<T>((*v)[i]);
    } else if (auto * v = std::get_if<std::vector<int64_t>>(&kv.data)) {
        out.resize(v->size());
        for (size_t i = 0; i < v->size(); ++i) out[i] = static_cast<T>((*v)[i]);
    } else if (auto * v = std::get_if<std::vector<double>>(&kv.data)) {
        out.resize(v->size());
        for (size_t i = 0; i < v->size(); ++i) out[i] = static_cast<T>((*v)[i]);
    } else if (auto * v = std::get_if<double>(&kv.data)) {
        out.assign(n_def, static_cast<T>(*v));
    }
    return out;
}

} // namespace

CpuTensor BailingMoe3CpuModel::bind_tensor(const std::string & name, bool required) const {
    auto loc = sg_.find_tensor(name);
    if (!loc.info) {
        if (required) throw std::runtime_error("BailingMoe3CpuModel: missing tensor " + name);
        return CpuTensor{};
    }
    CpuTensor t;
    t.ptr   = loc.data();
    t.dtype = loc.info->dtype;
    for (int i = 0; i < 4; ++i) t.ne[i] = loc.info->ne[i];
    t.nbytes = tensor_nbytes(t.dtype, t.ne[0], t.ne[1], t.ne[2], t.ne[3]);
    return t;
}

void BailingMoe3CpuModel::load(const std::string & gguf_path,
                               int max_ctx,
                               KvCacheType kv_type,
                               int n_threads) {
    const auto t0 = std::chrono::steady_clock::now();
    pool_.init(n_threads);
    print_cpu_caps(pool_.num_threads());

    sg_ = ShardedGguf::open(gguf_path);
    const std::string arch = sg_.get_str("general.architecture");
    if (arch != "bailingmoe3") {
        throw std::runtime_error("Expected bailingmoe3 GGUF, got: " + arch);
    }
    const std::string pfx = arch + ".";

    cfg_.n_blocks             = static_cast<int>(sg_.get_u64(pfx + "block_count", 24));
    cfg_.n_nextn              = static_cast<int>(sg_.get_u64(pfx + "nextn_predict_layers", 0));
    cfg_.n_layers             = cfg_.n_blocks - cfg_.n_nextn;
    cfg_.n_dense_lead         = static_cast<int>(sg_.get_u64(pfx + "leading_dense_block_count", 1));
    cfg_.n_embd               = static_cast<int>(sg_.get_u64(pfx + "embedding_length", 1536));
    cfg_.n_ff_dense           = static_cast<int>(sg_.get_u64(pfx + "feed_forward_length", 4608));
    cfg_.n_vocab              = static_cast<int>(sg_.get_u64(pfx + "vocab_size", 157184));
    cfg_.n_head               = static_cast<int>(sg_.get_u64(pfx + "attention.head_count", 16));
    cfg_.kda_head_dim         = static_cast<int>(sg_.get_u64(pfx + "kda.head_dim", 128));
    cfg_.kda_gate_lower_bound = static_cast<float>(sg_.get_f64(pfx + "kda.gate_lower_bound", -5.0));
    cfg_.q_lora_rank          = static_cast<int>(sg_.get_u64(pfx + "attention.q_lora_rank", 0));
    cfg_.kv_lora_rank         = static_cast<int>(sg_.get_u64(pfx + "attention.kv_lora_rank", 512));
    cfg_.rope_dim             = static_cast<int>(sg_.get_u64(pfx + "rope.dimension_count", 64));
    cfg_.qk_head_dim_mla      = static_cast<int>(sg_.get_u64(pfx + "attention.key_length_mla", 192));
    cfg_.v_head_dim_mla       = static_cast<int>(sg_.get_u64(pfx + "attention.value_length_mla", 128));
    cfg_.rope_freq_base       = static_cast<float>(sg_.get_f64(pfx + "rope.freq_base", 6000000.0));
    cfg_.n_expert             = static_cast<int>(sg_.get_u64(pfx + "expert_count", 128));
    cfg_.n_expert_used        = static_cast<int>(sg_.get_u64(pfx + "expert_used_count", 8));
    cfg_.n_expert_groups      = static_cast<int>(sg_.get_u64(pfx + "expert_group_count", 0));
    cfg_.n_group_used         = static_cast<int>(sg_.get_u64(pfx + "expert_group_used_count", 0));
    cfg_.n_ff_exp             = static_cast<int>(sg_.get_u64(pfx + "expert_feed_forward_length", 512));
    cfg_.n_ff_shexp           = static_cast<int>(sg_.get_u64(pfx + "expert_shared_feed_forward_length", cfg_.n_ff_exp));
    cfg_.expert_weights_norm  = sg_.get_bool(pfx + "expert_weights_norm", true);
    cfg_.expert_weights_scale = static_cast<float>(sg_.get_f64(pfx + "expert_weights_scale", 2.5));
    if (cfg_.expert_weights_scale == 0.0f) cfg_.expert_weights_scale = 1.0f;
    cfg_.rms_eps              = static_cast<float>(sg_.get_f64(pfx + "attention.layer_norm_rms_epsilon", 1e-6));
    cfg_.max_ctx              = max_ctx;
    cfg_.kv_type              = kv_type;
    cfg_.kv_q8_0              = (kv_type != KvCacheType::F16);

    cfg_.head_count_kv      = read_num_arr<int>(sg_, pfx + "attention.head_count_kv", cfg_.n_blocks, 0);
    cfg_.swiglu_clamp_exp   = read_num_arr<float>(sg_, pfx + "swiglu_clamp_exp", cfg_.n_blocks, 0.0f);
    cfg_.swiglu_clamp_shexp = read_num_arr<float>(sg_, pfx + "swiglu_clamp_shexp", cfg_.n_blocks, 0.0f);

    if (cfg_.kv_lora_rank != 512 || cfg_.rope_dim != 64) {
        throw std::runtime_error("BailingMoe3CpuModel expects kv_lora_rank=512, rope_dim=64");
    }

    tok_embd_    = bind_tensor("token_embd.weight");
    output_norm_ = bind_tensor("output_norm.weight");
    output_      = bind_tensor("output.weight", false);
    if (!output_.valid()) output_ = tok_embd_;

    layers_.clear();
    layers_.resize(static_cast<size_t>(cfg_.n_layers));
    kda_conv_q_.resize(static_cast<size_t>(cfg_.n_layers));
    kda_conv_k_.resize(static_cast<size_t>(cfg_.n_layers));
    kda_conv_v_.resize(static_cast<size_t>(cfg_.n_layers));
    kda_state_.resize(static_cast<size_t>(cfg_.n_layers));
    mla_kv_q8_.resize(static_cast<size_t>(cfg_.n_layers));
    mla_kv_q4_.resize(static_cast<size_t>(cfg_.n_layers));
    mla_kv_f16_.resize(static_cast<size_t>(cfg_.n_layers));

    const int d_inner = cfg_.n_head * cfg_.kda_head_dim;
    const size_t conv_n = static_cast<size_t>(3) * d_inner;
    const size_t ssm_n  = static_cast<size_t>(cfg_.n_head) * cfg_.kda_head_dim * cfg_.kda_head_dim;
    size_t kv_bytes_total = 0;

    for (int il = 0; il < cfg_.n_layers; ++il) {
        auto & L = layers_[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        L.is_kda    = (il < static_cast<int>(cfg_.head_count_kv.size()) && cfg_.head_count_kv[il] == 0);
        L.attn_norm = bind_tensor(p + "attn_norm.weight");
        L.ffn_norm  = bind_tensor(p + "ffn_norm.weight");

        if (L.is_kda) {
            L.wq          = bind_tensor(p + "attn_q.weight");
            L.wk          = bind_tensor(p + "attn_k.weight");
            L.wv          = bind_tensor(p + "attn_v.weight");
            L.wo          = bind_tensor(p + "attn_output.weight");
            L.ssm_conv_q  = bind_tensor(p + "ssm_conv1d_q.weight");
            L.ssm_conv_k  = bind_tensor(p + "ssm_conv1d_k.weight");
            L.ssm_conv_v  = bind_tensor(p + "ssm_conv1d_v.weight");
            L.ssm_f_a     = bind_tensor(p + "ssm_f_a.weight");
            L.ssm_dt_bias = bind_tensor(p + "ssm_dt.bias");
            L.ssm_a       = bind_tensor(p + "ssm_a");
            L.ssm_beta    = bind_tensor(p + "ssm_beta.weight");
            L.ssm_g_a     = bind_tensor(p + "ssm_g_a.weight");
            L.ssm_norm    = bind_tensor(p + "ssm_norm.weight");

            kda_conv_q_[il].assign(conv_n, 0.0f);
            kda_conv_k_[il].assign(conv_n, 0.0f);
            kda_conv_v_[il].assign(conv_n, 0.0f);
            kda_state_[il].assign(ssm_n, 0.0f);
        } else {
            if (cfg_.q_lora_rank > 0) {
                L.wq_a          = bind_tensor(p + "attn_q_a.weight");
                L.attn_q_a_norm = bind_tensor(p + "attn_q_a_norm.weight");
                L.wq_b          = bind_tensor(p + "attn_q_b.weight");
            } else {
                L.wq            = bind_tensor(p + "attn_q.weight");
            }
            L.wkv_a_mqa      = bind_tensor(p + "attn_kv_a_mqa.weight");
            L.attn_kv_a_norm = bind_tensor(p + "attn_kv_a_norm.weight");
            L.wk_b           = bind_tensor(p + "attn_k_b.weight");
            L.wv_b           = bind_tensor(p + "attn_v_b.weight");
            L.attn_gate      = bind_tensor(p + "attn_gate.weight");
            L.wo             = bind_tensor(p + "attn_output.weight");

            if (kv_type == KvCacheType::Q8_0) {
                mla_kv_q8_[il].resize(static_cast<size_t>(cfg_.max_ctx));
                kv_bytes_total += static_cast<size_t>(cfg_.max_ctx) * sizeof(CpuMlaKv576Q8Pos);
            } else if (kv_type == KvCacheType::Q4_0) {
                mla_kv_q4_[il].resize(static_cast<size_t>(cfg_.max_ctx));
                kv_bytes_total += static_cast<size_t>(cfg_.max_ctx) * sizeof(CpuMlaKv576Q4Pos);
            } else {
                mla_kv_f16_[il].resize(static_cast<size_t>(cfg_.max_ctx) * 576);
                kv_bytes_total += static_cast<size_t>(cfg_.max_ctx) * 576 * sizeof(uint16_t);
            }
        }

        if (il < cfg_.n_dense_lead) {
            L.ffn_gate = bind_tensor(p + "ffn_gate.weight");
            L.ffn_up   = bind_tensor(p + "ffn_up.weight");
            L.ffn_down = bind_tensor(p + "ffn_down.weight");
        } else {
            L.ffn_gate_inp    = bind_tensor(p + "ffn_gate_inp.weight");
            L.ffn_exp_probs_b = bind_tensor(p + "exp_probs_b.bias");
            L.ffn_gate_exps   = bind_tensor(p + "ffn_gate_exps.weight");
            L.ffn_up_exps     = bind_tensor(p + "ffn_up_exps.weight");
            L.ffn_down_exps   = bind_tensor(p + "ffn_down_exps.weight");
            L.ffn_gate_shexp  = bind_tensor(p + "ffn_gate_shexp.weight");
            L.ffn_up_shexp    = bind_tensor(p + "ffn_up_shexp.weight");
            L.ffn_down_shexp  = bind_tensor(p + "ffn_down_shexp.weight");
        }
    }

    logits_buf_.resize(static_cast<size_t>(cfg_.n_vocab));
    ensure_batch_buffers(128);
    reset_state();

    const auto t1 = std::chrono::steady_clock::now();
    std::printf("[BailingMoE3-CPU] Loaded %s (%d layers: %d dense + %d MoE[%d/%d grp=%d/%d], n_embd=%d, n_head=%d, q_lora=%d, ctx=%d, kv=%s %.1f MiB, mmap=%.2f GiB, threads=%d) in %.2f s\n",
                sg_.get_str("general.name", "bailingmoe3").c_str(),
                cfg_.n_layers, cfg_.n_dense_lead, cfg_.n_layers - cfg_.n_dense_lead,
                cfg_.n_expert_used, cfg_.n_expert, cfg_.n_group_used, cfg_.n_expert_groups,
                cfg_.n_embd, cfg_.n_head, cfg_.q_lora_rank,
                cfg_.max_ctx, kv_cache_type_name(cfg_.kv_type),
                static_cast<double>(kv_bytes_total) / (1024.0 * 1024.0),
                static_cast<double>(sg_.total_bytes()) / (1024.0 * 1024.0 * 1024.0),
                pool_.num_threads(),
                std::chrono::duration<double>(t1 - t0).count());
}

void BailingMoe3CpuModel::reset_state() {
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (!layers_[il].is_kda) continue;
        std::fill(kda_conv_q_[il].begin(), kda_conv_q_[il].end(), 0.0f);
        std::fill(kda_conv_k_[il].begin(), kda_conv_k_[il].end(), 0.0f);
        std::fill(kda_conv_v_[il].begin(), kda_conv_v_[il].end(), 0.0f);
        std::fill(kda_state_[il].begin(),  kda_state_[il].end(),  0.0f);
    }
}

void BailingMoe3CpuModel::save_kda_checkpoint(KdaCheckpoint & out, int pos) const {
    out.pos = pos;
    out.conv_q.resize(static_cast<size_t>(cfg_.n_layers));
    out.conv_k.resize(static_cast<size_t>(cfg_.n_layers));
    out.conv_v.resize(static_cast<size_t>(cfg_.n_layers));
    out.ssm.resize(static_cast<size_t>(cfg_.n_layers));
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (!layers_[il].is_kda) continue;
        out.conv_q[il] = kda_conv_q_[il];
        out.conv_k[il] = kda_conv_k_[il];
        out.conv_v[il] = kda_conv_v_[il];
        out.ssm[il]    = kda_state_[il];
    }
}

void BailingMoe3CpuModel::restore_kda_checkpoint(const KdaCheckpoint & cp) {
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (!layers_[il].is_kda) continue;
        kda_conv_q_[il] = cp.conv_q[il];
        kda_conv_k_[il] = cp.conv_k[il];
        kda_conv_v_[il] = cp.conv_v[il];
        kda_state_[il]  = cp.ssm[il];
    }
}

void BailingMoe3CpuModel::ensure_batch_buffers(int T) {
    if (T <= max_alloc_T_) return;
    max_alloc_T_ = T;

    const int d_inner  = cfg_.n_head * cfg_.kda_head_dim;
    const int qk_nope  = cfg_.qk_head_dim_mla - cfg_.rope_dim;
    const int kv_tot   = cfg_.kv_lora_rank + cfg_.rope_dim;

    cur_.resize(static_cast<size_t>(T) * cfg_.n_embd);
    x_norm_.resize(static_cast<size_t>(T) * cfg_.n_embd);
    sub_out_.resize(static_cast<size_t>(T) * cfg_.n_embd);
    x_q8_.resize(static_cast<size_t>(T) * std::max({cfg_.n_embd, d_inner, cfg_.n_ff_dense}) / 32);
    x_q8k_.resize(static_cast<size_t>(T) * std::max({cfg_.n_embd, d_inner, cfg_.n_ff_dense}) / 256);

    kda_q_.resize(static_cast<size_t>(T) * d_inner);
    kda_k_.resize(static_cast<size_t>(T) * d_inner);
    kda_v_.resize(static_cast<size_t>(T) * d_inner);
    kda_qc_.resize(static_cast<size_t>(T) * d_inner);
    kda_kc_.resize(static_cast<size_t>(T) * d_inner);
    kda_vc_.resize(static_cast<size_t>(T) * d_inner);
    kda_g_.resize(static_cast<size_t>(T) * d_inner);
    kda_beta_.resize(static_cast<size_t>(T) * cfg_.n_head);
    kda_out_.resize(static_cast<size_t>(T) * d_inner);
    kda_gate_.resize(static_cast<size_t>(T) * d_inner);

    mla_qr_.resize(static_cast<size_t>(T) * std::max(1, cfg_.q_lora_rank));
    mla_qall_.resize(static_cast<size_t>(T) * cfg_.n_head * cfg_.qk_head_dim_mla);
    mla_qnope_.resize(static_cast<size_t>(T) * cfg_.n_head * qk_nope);
    mla_qpe_.resize(static_cast<size_t>(T) * cfg_.n_head * cfg_.rope_dim);
    mla_qabs_.resize(static_cast<size_t>(T) * cfg_.n_head * cfg_.kv_lora_rank);
    mla_q576_.resize(static_cast<size_t>(T) * cfg_.n_head * kv_tot);
    mla_kvall_.resize(static_cast<size_t>(T) * kv_tot);
    mla_olat_.resize(static_cast<size_t>(T) * cfg_.n_head * cfg_.kv_lora_rank);
    mla_o_.resize(static_cast<size_t>(T) * cfg_.n_head * cfg_.v_head_dim_mla);
    mla_gate_.resize(static_cast<size_t>(T) * cfg_.n_head);

    ffn_g_.resize(static_cast<size_t>(T) * cfg_.n_ff_dense);
    ffn_u_.resize(static_cast<size_t>(T) * cfg_.n_ff_dense);
    router_.resize(static_cast<size_t>(T) * cfg_.n_expert);
    top_ids_.resize(static_cast<size_t>(T) * cfg_.n_expert_used);
    top_w_.resize(static_cast<size_t>(T) * cfg_.n_expert_used);
}

void BailingMoe3CpuModel::run_forward(const int32_t * tokens, const int32_t * positions, int T, float * out_logits) {
    ensure_batch_buffers(T);

    const int   qk_nope   = cfg_.qk_head_dim_mla - cfg_.rope_dim;
    const float kda_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.kda_head_dim));
    const float mla_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.qk_head_dim_mla));

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
    double ms_kda_proj = 0, ms_kda_conv = 0, ms_kda_rec = 0, ms_kda_out = 0;
    double ms_mla_proj = 0, ms_mla_prep = 0, ms_mla_attn = 0, ms_mla_out = 0;
    double ms_dense = 0, ms_router = 0, ms_moe = 0, ms_head = 0, ms_norm = 0;
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

        if (L.is_kda) {
            mark(ms_kda_proj, [&]() {
                const bool need_q8  = use_direct && (dt_needs_q8(L.wq.dtype) || dt_needs_q8(L.wk.dtype) || dt_needs_q8(L.wv.dtype) ||
                                                     dt_needs_q8(L.ssm_f_a.dtype) || dt_needs_q8(L.ssm_beta.dtype) || dt_needs_q8(L.ssm_g_a.dtype));
                const bool need_q8k = use_direct && (dt_needs_q8k(L.wq.dtype) || dt_needs_q8k(L.wk.dtype) || dt_needs_q8k(L.wv.dtype) ||
                                                     dt_needs_q8k(L.ssm_f_a.dtype) || dt_needs_q8k(L.ssm_beta.dtype) || dt_needs_q8k(L.ssm_g_a.dtype));
                if (need_q8)  quantize_rows_q8_0(pool_, x_norm_.data(), x_q8_.data(),  T, cfg_.n_embd);
                if (need_q8k) quantize_rows_q8_k(pool_, x_norm_.data(), x_q8k_.data(), T, cfg_.n_embd);

                const LinearSpec specs[6] = {
                    {&L.wq,       kda_q_.data()},
                    {&L.wk,       kda_k_.data()},
                    {&L.wv,       kda_v_.data()},
                    {&L.ssm_f_a,  kda_g_.data()},
                    {&L.ssm_beta, kda_beta_.data()},
                    {&L.ssm_g_a,  kda_gate_.data()},
                };
                linear_multi(pool_, specs, 6, x_norm_.data(), need_q8 ? x_q8_.data() : nullptr, need_q8k ? x_q8k_.data() : nullptr, T);
            });

            mark(ms_kda_conv, [&]() {
                ssm_conv1d_silu_qkv_l2(
                    pool_,
                    kda_q_.data(), kda_k_.data(), kda_v_.data(),
                    static_cast<const float *>(L.ssm_conv_q.ptr),
                    static_cast<const float *>(L.ssm_conv_k.ptr),
                    static_cast<const float *>(L.ssm_conv_v.ptr),
                    kda_conv_q_[il].data(), kda_conv_k_[il].data(), kda_conv_v_[il].data(),
                    kda_qc_.data(), kda_kc_.data(), kda_vc_.data(),
                    T, cfg_.n_head, cfg_.rms_eps);
                kda_gate_and_beta(
                    pool_,
                    kda_g_.data(), static_cast<const float *>(L.ssm_dt_bias.ptr),
                    static_cast<const float *>(L.ssm_a.ptr), kda_beta_.data(),
                    cfg_.kda_gate_lower_bound, T, cfg_.n_head, cfg_.kda_head_dim, /*negate_a=*/false);
            });

            mark(ms_kda_rec, [&]() {
                gated_delta_net_d128(
                    pool_,
                    kda_qc_.data(), kda_kc_.data(), kda_vc_.data(), kda_g_.data(), kda_beta_.data(),
                    kda_state_[il].data(), kda_out_.data(), T, cfg_.n_head, kda_scale);
            });

            mark(ms_kda_out, [&]() {
                rms_norm_sigmoid_gate_d128(pool_, kda_out_.data(), static_cast<const float *>(L.ssm_norm.ptr),
                                           kda_gate_.data(), T, cfg_.n_head, cfg_.rms_eps);
                linear(pool_, L.wo, kda_out_.data(), nullptr, nullptr, sub_out_.data(), T);
            });
        } else {
            mark(ms_mla_proj, [&]() {
                const CpuTensor & wq_first = (cfg_.q_lora_rank > 0) ? L.wq_a : L.wq;
                float * q_first_out        = (cfg_.q_lora_rank > 0) ? mla_qr_.data() : mla_qall_.data();
                const bool need_q8  = use_direct && (dt_needs_q8(wq_first.dtype) || dt_needs_q8(L.wkv_a_mqa.dtype) || dt_needs_q8(L.attn_gate.dtype));
                const bool need_q8k = use_direct && (dt_needs_q8k(wq_first.dtype) || dt_needs_q8k(L.wkv_a_mqa.dtype) || dt_needs_q8k(L.attn_gate.dtype));
                if (need_q8)  quantize_rows_q8_0(pool_, x_norm_.data(), x_q8_.data(),  T, cfg_.n_embd);
                if (need_q8k) quantize_rows_q8_k(pool_, x_norm_.data(), x_q8k_.data(), T, cfg_.n_embd);

                const LinearSpec specs[3] = {
                    {&wq_first,    q_first_out},
                    {&L.wkv_a_mqa, mla_kvall_.data()},
                    {&L.attn_gate, mla_gate_.data()},
                };
                linear_multi(pool_, specs, 3, x_norm_.data(), need_q8 ? x_q8_.data() : nullptr, need_q8k ? x_q8k_.data() : nullptr, T);
                if (cfg_.q_lora_rank > 0) {
                    rmsnorm(mla_qr_.data(), static_cast<const float *>(L.attn_q_a_norm.ptr), mla_qr_.data(), cfg_.q_lora_rank, T, cfg_.rms_eps);
                    linear(pool_, L.wq_b, mla_qr_.data(), nullptr, nullptr, mla_qall_.data(), T);
                }
            });

            void * kv_ptr = (cfg_.kv_type == KvCacheType::Q8_0) ? static_cast<void *>(mla_kv_q8_[il].data())
                          : (cfg_.kv_type == KvCacheType::Q4_0) ? static_cast<void *>(mla_kv_q4_[il].data())
                                                                : static_cast<void *>(mla_kv_f16_[il].data());
            mark(ms_mla_prep, [&]() {
                mla_rope_split_q(mla_qall_.data(), positions, mla_qnope_.data(), mla_qpe_.data(),
                                 T, cfg_.n_head, qk_nope, cfg_.rope_dim, cfg_.rope_freq_base);
                mla_kv_norm_rope_write_576(mla_kvall_.data(), static_cast<const float *>(L.attn_kv_a_norm.ptr),
                                           positions, kv_ptr, cfg_.kv_type,
                                           T, cfg_.rms_eps, cfg_.rope_freq_base);
                batched_head_proj(pool_, L.wk_b, mla_qnope_.data(), mla_qabs_.data(), T);
                mla_concat_q576(mla_qabs_.data(), mla_qpe_.data(), mla_q576_.data(), T, cfg_.n_head);
            });

            mark(ms_mla_attn, [&]() {
                mla_attn_d576_v512(pool_, mla_q576_.data(), kv_ptr, cfg_.kv_type,
                                   positions, mla_olat_.data(), T, cfg_.n_head, mla_scale);
            });

            mark(ms_mla_out, [&]() {
                batched_head_proj(pool_, L.wv_b, mla_olat_.data(), mla_o_.data(), T);
                mla_head_sigmoid_gate(mla_o_.data(), mla_gate_.data(), T, cfg_.n_head, cfg_.v_head_dim_mla);
                linear(pool_, L.wo, mla_o_.data(), nullptr, nullptr, sub_out_.data(), T);
            });
        }

        mark(ms_norm, [&]() {
            add_rmsnorm(cur_.data(), sub_out_.data(), static_cast<const float *>(L.ffn_norm.ptr), x_norm_.data(), cfg_.n_embd, T, cfg_.rms_eps);
        });

        if (il < cfg_.n_dense_lead) {
            mark(ms_dense, [&]() {
                const bool need_q8  = use_direct && (dt_needs_q8(L.ffn_gate.dtype) || dt_needs_q8(L.ffn_up.dtype));
                const bool need_q8k = use_direct && (dt_needs_q8k(L.ffn_gate.dtype) || dt_needs_q8k(L.ffn_up.dtype));
                if (need_q8)  quantize_rows_q8_0(pool_, x_norm_.data(), x_q8_.data(),  T, cfg_.n_embd);
                if (need_q8k) quantize_rows_q8_k(pool_, x_norm_.data(), x_q8k_.data(), T, cfg_.n_embd);
                if (T == 1 && (cfg_.n_ff_dense % 256) == 0) {
                    dense_ffn_swiglu_t1(pool_, L.ffn_gate, L.ffn_up, L.ffn_down,
                                        x_norm_.data(),
                                        need_q8 ? x_q8_.data() : nullptr,
                                        need_q8k ? x_q8k_.data() : nullptr,
                                        sub_out_.data());
                } else {
                    const LinearSpec specs[2] = {{&L.ffn_gate, ffn_g_.data()}, {&L.ffn_up, ffn_u_.data()}};
                    linear_multi(pool_, specs, 2, x_norm_.data(), need_q8 ? x_q8_.data() : nullptr, need_q8k ? x_q8k_.data() : nullptr, T);
                    silu_mul_mt(pool_, ffn_g_.data(), ffn_u_.data(), ffn_g_.data(), static_cast<size_t>(T) * cfg_.n_ff_dense);
                    linear(pool_, L.ffn_down, ffn_g_.data(), nullptr, nullptr, sub_out_.data(), T);
                }
            });
        } else {
            mark(ms_router, [&]() {
                linear(pool_, L.ffn_gate_inp, x_norm_.data(), nullptr, nullptr, router_.data(), T);
                moe_route_sigmoid(
                    router_.data(), static_cast<const float *>(L.ffn_exp_probs_b.ptr),
                    top_ids_.data(), top_w_.data(),
                    T, cfg_.n_expert, cfg_.n_expert_used, cfg_.expert_weights_norm, cfg_.expert_weights_scale,
                    cfg_.n_expert_groups, cfg_.n_group_used);
            });
            mark(ms_moe, [&]() {
                const bool need_q8  = (dt_needs_q8(L.ffn_gate_exps.dtype) || dt_needs_q8(L.ffn_up_exps.dtype) ||
                                       dt_needs_q8(L.ffn_gate_shexp.dtype) || dt_needs_q8(L.ffn_up_shexp.dtype));
                const bool need_q8k = (dt_needs_q8k(L.ffn_gate_exps.dtype) || dt_needs_q8k(L.ffn_up_exps.dtype) ||
                                       dt_needs_q8k(L.ffn_gate_shexp.dtype) || dt_needs_q8k(L.ffn_up_shexp.dtype));
                if (need_q8)  quantize_rows_q8_0(pool_, x_norm_.data(), x_q8_.data(),  T, cfg_.n_embd);
                if (need_q8k) quantize_rows_q8_k(pool_, x_norm_.data(), x_q8k_.data(), T, cfg_.n_embd);
                const float c_exp = (il < static_cast<int>(cfg_.swiglu_clamp_exp.size())) ? cfg_.swiglu_clamp_exp[il] : 0.0f;
                const float c_sh  = (il < static_cast<int>(cfg_.swiglu_clamp_shexp.size())) ? cfg_.swiglu_clamp_shexp[il] : 0.0f;
                moe_ffn_with_shared(
                    pool_,
                    L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps,
                    L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
                    x_norm_.data(), need_q8 ? x_q8_.data() : nullptr, need_q8k ? x_q8k_.data() : nullptr,
                    top_ids_.data(), top_w_.data(), sub_out_.data(),
                    T, cfg_.n_expert_used, c_exp, c_sh);
            });
        }

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
        const double tot = ms_norm + ms_kda_proj + ms_kda_conv + ms_kda_rec + ms_kda_out +
                           ms_mla_proj + ms_mla_prep + ms_mla_attn + ms_mla_out +
                           ms_dense + ms_router + ms_moe + ms_head;
        std::printf("  [PROF CPU T=%d] norm=%.2f kda(p=%.2f c=%.2f r=%.2f o=%.2f) mla(p=%.2f pr=%.2f a=%.2f o=%.2f) dense=%.2f rtr=%.2f moe=%.2f head=%.2f | tot=%.2f ms (%.1f tok/s)\n",
                    T, ms_norm,
                    ms_kda_proj, ms_kda_conv, ms_kda_rec, ms_kda_out,
                    ms_mla_proj, ms_mla_prep, ms_mla_attn, ms_mla_out,
                    ms_dense, ms_router, ms_moe, ms_head, tot, (tot > 0.0) ? (T * 1000.0 / tot) : 0.0);
    }
}

void BailingMoe3CpuModel::forward(const std::vector<int32_t> & tokens,
                                  const std::vector<int32_t> & positions,
                                  std::vector<float> & out_logits) {
    const int T = static_cast<int>(tokens.size());
    if (T <= 0 || positions.size() != tokens.size()) {
        throw std::runtime_error("BailingMoe3CpuModel::forward invalid T");
    }
    out_logits.resize(static_cast<size_t>(cfg_.n_vocab));
    run_forward(tokens.data(), positions.data(), T, out_logits.data());
}

int32_t BailingMoe3CpuModel::forward_decode_greedy(int32_t token, int32_t pos) {
    run_forward(&token, &pos, 1, logits_buf_.data());
    return argmax_f32(logits_buf_.data(), cfg_.n_vocab);
}

void BailingMoe3CpuModel::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    out_logits.resize(static_cast<size_t>(cfg_.n_vocab));
    run_forward(&token, &pos, 1, out_logits.data());
}

} // namespace rookery::cpu
