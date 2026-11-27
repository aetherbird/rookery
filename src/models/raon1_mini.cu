#include "models/raon1_mini.h"
#include "kernels/kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <variant>

namespace rookery {

namespace {

inline void cuda_ok(cudaError_t err, const char * msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

template <typename T>
T * dev_alloc(int dev, size_t count) {
    cuda_ok(cudaSetDevice(dev), "raon1 dev_alloc setDevice");
    T * p = nullptr;
    cuda_ok(cudaMalloc(&p, count * sizeof(T)), "raon1 dev_alloc cudaMalloc");
    return p;
}

template <typename T>
std::vector<T> read_num_arr(const ShardedGguf & sg, const std::string & key, int expected_len, T fallback) {
    std::vector<T> out(static_cast<size_t>(expected_len), fallback);
    if (!sg.meta().has_kv(key)) return out;
    const auto & val = sg.meta().kv(key).data;
    if (auto * v = std::get_if<std::vector<int64_t>>(&val)) {
        for (size_t i = 0; i < v->size() && i < out.size(); ++i) out[i] = static_cast<T>((*v)[i]);
    } else if (auto * v = std::get_if<std::vector<uint64_t>>(&val)) {
        for (size_t i = 0; i < v->size() && i < out.size(); ++i) out[i] = static_cast<T>((*v)[i]);
    } else if (auto * v = std::get_if<std::vector<double>>(&val)) {
        for (size_t i = 0; i < v->size() && i < out.size(); ++i) out[i] = static_cast<T>((*v)[i]);
    }
    return out;
}

std::vector<std::string> read_str_arr(const ShardedGguf & sg, const std::string & key) {
    if (!sg.meta().has_kv(key)) return {};
    const auto & val = sg.meta().kv(key).data;
    if (auto * v = std::get_if<std::vector<std::string>>(&val)) return *v;
    return {};
}

} // namespace

Raon1MiniModel::~Raon1MiniModel() {
    invalidate_graphs();
    for (size_t il = 0; il < rda_state_.size(); ++il) {
        if (il < layers_.size()) cudaSetDevice(layers_[il].device);
        if (rda_conv_q_[il]) cudaFree(rda_conv_q_[il]);
        if (rda_conv_k_[il]) cudaFree(rda_conv_k_[il]);
        if (rda_conv_v_[il]) cudaFree(rda_conv_v_[il]);
        if (rda_state_[il])  cudaFree(rda_state_[il]);
    }
    for (size_t il = 0; il < mla_kv_cache_.size(); ++il) {
        if (il < layers_.size()) cudaSetDevice(layers_[il].device);
        if (mla_kv_cache_[il]) cudaFree(mla_kv_cache_[il]);
    }
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        if (rt.cublas)       cublasDestroy(rt.cublas);
        if (rt.stream)       cudaStreamDestroy(rt.stream);
        if (rt.done_event)   cudaEventDestroy(rt.done_event);
        if (rt.d_stream)     cudaFree(rt.d_stream);
        if (rt.d_stream_tmp) cudaFree(rt.d_stream_tmp);
        if (rt.d_hc_mixes)   cudaFree(rt.d_hc_mixes);
        if (rt.d_hc_post)    cudaFree(rt.d_hc_post);
        if (rt.d_hc_comb)    cudaFree(rt.d_hc_comb);
        if (rt.d_x_norm)     cudaFree(rt.d_x_norm);
        if (rt.d_sub_out)    cudaFree(rt.d_sub_out);
        if (rt.d_rda_q)      cudaFree(rt.d_rda_q);
        if (rt.d_rda_k)      cudaFree(rt.d_rda_k);
        if (rt.d_rda_v)      cudaFree(rt.d_rda_v);
        if (rt.d_rda_qc)     cudaFree(rt.d_rda_qc);
        if (rt.d_rda_kc)     cudaFree(rt.d_rda_kc);
        if (rt.d_rda_vc)     cudaFree(rt.d_rda_vc);
        if (rt.d_rda_fa)     cudaFree(rt.d_rda_fa);
        if (rt.d_rda_decay)  cudaFree(rt.d_rda_decay);
        if (rt.d_rda_beta)   cudaFree(rt.d_rda_beta);
        if (rt.d_rda_delta)  cudaFree(rt.d_rda_delta);
        if (rt.d_rda_out)    cudaFree(rt.d_rda_out);
        if (rt.d_rda_gate)   cudaFree(rt.d_rda_gate);
        if (rt.d_mla_qr)     cudaFree(rt.d_mla_qr);
        if (rt.d_mla_qall)   cudaFree(rt.d_mla_qall);
        if (rt.d_mla_qnope)  cudaFree(rt.d_mla_qnope);
        if (rt.d_mla_qpe)    cudaFree(rt.d_mla_qpe);
        if (rt.d_mla_qabs)   cudaFree(rt.d_mla_qabs);
        if (rt.d_mla_q576)   cudaFree(rt.d_mla_q576);
        if (rt.d_mla_kvall)  cudaFree(rt.d_mla_kvall);
        if (rt.d_mla_olat)   cudaFree(rt.d_mla_olat);
        if (rt.d_mla_o)      cudaFree(rt.d_mla_o);
        if (rt.d_mla_gate)   cudaFree(rt.d_mla_gate);
        if (rt.d_ffn_g)      cudaFree(rt.d_ffn_g);
        if (rt.d_ffn_u)      cudaFree(rt.d_ffn_u);
        if (rt.d_router)     cudaFree(rt.d_router);
        if (rt.d_top_ids)    cudaFree(rt.d_top_ids);
        if (rt.d_top_w)      cudaFree(rt.d_top_w);
        if (rt.d_tokens)     cudaFree(rt.d_tokens);
        if (rt.d_pos)        cudaFree(rt.d_pos);
        if (rt.d_logits)     cudaFree(rt.d_logits);
        if (rt.d_argmax)     cudaFree(rt.d_argmax);
        if (rt.d_scratch)    cudaFree(rt.d_scratch);
    }
}

void Raon1MiniModel::invalidate_graphs() noexcept {
    for (auto & rt : runtimes_) {
        if (rt.decode_graph_t1) {
            cudaSetDevice(rt.device);
            cudaGraphExecDestroy(rt.decode_graph_t1);
            rt.decode_graph_t1 = nullptr;
        }
    }
    graphs_ready_ = false;
}

void Raon1MiniModel::set_reasoning_effort(const std::string & effort) {
    static constexpr float kNoThinkingAlpha = 0.1f;
    std::string eff = effort.empty() ? cfg_.default_reasoning_effort : effort;
    float alpha = 0.0f;
    if (eff == "none") {
        auto it = cfg_.reasoning_effort_alpha.find("none");
        alpha = (it != cfg_.reasoning_effort_alpha.end() && it->second > 0.0f) ? it->second : kNoThinkingAlpha;
    } else {
        auto it = cfg_.reasoning_effort_alpha.find(eff);
        if (it != cfg_.reasoning_effort_alpha.end()) {
            alpha = it->second;
        } else {
            // Fallback to default effort if caller passes an unrecognized level
            eff = cfg_.default_reasoning_effort;
            auto dit = cfg_.reasoning_effort_alpha.find(eff);
            alpha = (dit != cfg_.reasoning_effort_alpha.end()) ? dit->second : 0.05f;
        }
    }
    if (alpha != effort_alpha_) {
        effort_alpha_ = alpha;
        invalidate_graphs();
    }
    cur_effort_ = eff;
}

void Raon1MiniModel::load(const std::string & gguf_path,
                          const std::vector<int> & gpu_ids,
                          const std::vector<float> & split_weights,
                          int max_ctx,
                          bool kv_q8_0) {
    if (gpu_ids.empty()) throw std::runtime_error("Raon1MiniModel::load: gpu_ids is empty");

    const auto t0 = std::chrono::steady_clock::now();
    auto sg = ShardedGguf::open(gguf_path);
    const std::string arch = sg.get_str("general.architecture", "raon1");
    const std::string pfx  = arch + ".";

    cfg_.n_layers             = static_cast<int>(sg.get_u64(pfx + "block_count", 25));
    cfg_.n_dense_lead         = static_cast<int>(sg.get_u64(pfx + "leading_dense_block_count", 1));
    cfg_.n_embd               = static_cast<int>(sg.get_u64(pfx + "embedding_length", 2560));
    cfg_.n_ff_dense           = static_cast<int>(sg.get_u64(pfx + "feed_forward_length", 12032));
    cfg_.n_vocab              = static_cast<int>(sg.get_u64(pfx + "vocab_size", 259840));
    cfg_.n_head               = static_cast<int>(sg.get_u64(pfx + "attention.head_count", 40));
    cfg_.rda_head_dim         = static_cast<int>(sg.get_u64(pfx + "kda.head_dim", 128));
    cfg_.rda_decay_rank       = static_cast<int>(sg.get_u64(pfx + "ssm.time_step_rank", 128));
    cfg_.rda_conv_kernel      = static_cast<int>(sg.get_u64(pfx + "ssm.conv_kernel", 4));
    cfg_.rda_gate_lower_bound = static_cast<float>(sg.get_f64(pfx + "kda.gate_lower_bound", -5.0));
    cfg_.q_lora_rank          = static_cast<int>(sg.get_u64(pfx + "attention.q_lora_rank", 576));
    cfg_.kv_lora_rank         = static_cast<int>(sg.get_u64(pfx + "attention.kv_lora_rank", 512));
    cfg_.qk_pe_dim            = static_cast<int>(sg.get_u64(pfx + "rope.dimension_count", 64));
    cfg_.qk_head_dim_mla      = static_cast<int>(sg.get_u64(pfx + "attention.key_length_mla", 192));
    cfg_.v_head_dim_mla       = static_cast<int>(sg.get_u64(pfx + "attention.value_length_mla", 128));
    cfg_.n_expert             = static_cast<int>(sg.get_u64(pfx + "expert_count", 128));
    cfg_.n_expert_used        = static_cast<int>(sg.get_u64(pfx + "expert_used_count", 6));
    cfg_.n_ff_exp             = static_cast<int>(sg.get_u64(pfx + "expert_feed_forward_length", 1088));
    cfg_.n_ff_shexp           = cfg_.n_ff_exp * static_cast<int>(sg.get_u64(pfx + "expert_shared_count", 1));
    cfg_.expert_weights_norm  = sg.get_bool(pfx + "expert_weights_norm", true);
    cfg_.expert_weights_scale = static_cast<float>(sg.get_f64(pfx + "expert_weights_scale", 1.0));
    cfg_.situ_beta            = static_cast<float>(sg.get_f64(pfx + "activation.situ_beta", 4.0));
    cfg_.situ_linear_beta     = static_cast<float>(sg.get_f64(pfx + "activation.situ_linear_beta", 25.0));
    cfg_.hc_mult              = static_cast<int>(sg.get_u64(pfx + "hyper_connection.count", 4));
    cfg_.hc_eps               = static_cast<float>(sg.get_f64(pfx + "hyper_connection.epsilon", 1e-5));
    cfg_.hc_post_factor       = static_cast<float>(sg.get_f64(pfx + "hyper_connection.magnitude", 2.0));
    cfg_.rms_eps              = static_cast<float>(sg.get_f64(pfx + "attention.layer_norm_rms_epsilon", 1e-5));
    cfg_.mla_latent_rms_eps   = static_cast<float>(sg.get_f64(pfx + "attention.latent_norm_rms_epsilon", 1e-6));
    cfg_.max_ctx              = max_ctx;
    cfg_.kv_q8_0              = kv_q8_0;
    cfg_.head_count_kv        = read_num_arr<int>(sg, pfx + "attention.head_count_kv", cfg_.n_layers, 0);

    cfg_.default_reasoning_effort = sg.get_str(pfx + "reasoning_effort.default", "high");
    const auto eff_names  = read_str_arr(sg, pfx + "reasoning_effort.names");
    const auto eff_alphas = read_num_arr<float>(sg, pfx + "reasoning_effort.alpha", static_cast<int>(eff_names.size()), 0.0f);
    cfg_.reasoning_effort_alpha.clear();
    for (size_t i = 0; i < eff_names.size(); ++i) {
        cfg_.reasoning_effort_alpha[eff_names[i]] = eff_alphas[i];
    }
    if (cfg_.reasoning_effort_alpha.empty()) {
        cfg_.reasoning_effort_alpha = {{"low", 0.3f}, {"medium", 0.1f}, {"high", 0.05f}, {"xhigh", 0.02f}, {"max", 0.0f}};
    }
    set_reasoning_effort(cfg_.default_reasoning_effort);

    if (cfg_.kv_lora_rank != 512 || cfg_.qk_pe_dim != 64 || cfg_.hc_mult != 4) {
        throw std::runtime_error("Raon1MiniModel expects kv_lora_rank=512, qk_pe_dim=64, hc_mult=4");
    }

    // Partition layers across GPUs proportionally to split_weights (or free VRAM minus headroom!)
    std::vector<float> sw = split_weights;
    if (sw.size() != gpu_ids.size()) {
        sw.resize(gpu_ids.size());
        for (size_t g = 0; g < gpu_ids.size(); ++g) {
            cudaSetDevice(gpu_ids[g]);
            size_t free_b = 0, tot_b = 0;
            cudaMemGetInfo(&free_b, &tot_b);
            // Reserve ~2.0 GiB on last GPU (output.weight + KV + runtime), ~1.5 GiB on others
            const float reserve_gb = (g + 1 == gpu_ids.size()) ? 2.0f : 1.5f;
            sw[g] = std::max(1.0f, static_cast<float>(free_b) / (1024.0f * 1024.0f * 1024.0f) - reserve_gb);
        }
    }
    const float sw_sum = std::accumulate(sw.begin(), sw.end(), 0.0f);

    runtimes_.resize(gpu_ids.size());
    int cur_l = 0;
    for (size_t g = 0; g < gpu_ids.size(); ++g) {
        int count = (g + 1 == gpu_ids.size())
                        ? (cfg_.n_layers - cur_l)
                        : static_cast<int>(std::round(cfg_.n_layers * (sw[g] / sw_sum)));
        count = std::max(0, std::min(count, cfg_.n_layers - cur_l));
        runtimes_[g].device      = gpu_ids[g];
        runtimes_[g].layer_start = cur_l;
        runtimes_[g].layer_end   = cur_l + count;
        cur_l += count;
    }

    const int last_dev = gpu_ids.back();
    layers_.resize(cfg_.n_layers);
    rda_conv_q_.resize(cfg_.n_layers, nullptr);
    rda_conv_k_.resize(cfg_.n_layers, nullptr);
    rda_conv_v_.resize(cfg_.n_layers, nullptr);
    rda_state_.resize(cfg_.n_layers, nullptr);
    mla_kv_cache_.resize(cfg_.n_layers, nullptr);

    const int d_inner    = cfg_.n_head * cfg_.rda_head_dim; // 5120
    const int kv_dim_tot = cfg_.kv_lora_rank + cfg_.qk_pe_dim; // 576
    const size_t mla_kv_bytes = kv_q8_0
        ? static_cast<size_t>(cfg_.max_ctx) * 640
        : static_cast<size_t>(cfg_.max_ctx) * kv_dim_tot * sizeof(uint16_t);

    // Parallel multi-GPU weight upload
    std::vector<std::thread> load_threads;
    std::vector<std::exception_ptr> load_errs(runtimes_.size(), nullptr);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        load_threads.emplace_back([&, gi]() {
            try {
                const int dev = runtimes_[gi].device;
                cuda_ok(cudaSetDevice(dev), "raon1 parallel load setDevice");
                if (gi == 0) {
                    tok_embd_ = pool_.upload(sg, "token_embd.weight", dev, false);
                }
                if (gi + 1 == runtimes_.size()) {
                    output_norm_ = pool_.upload(sg, "output_norm.weight", dev, false);
                    output_      = pool_.upload(sg, "output.weight",      dev, false);
                }
                for (int il = runtimes_[gi].layer_start; il < runtimes_[gi].layer_end; ++il) {
                    auto & L = layers_[il];
                    const std::string p = "blk." + std::to_string(il) + ".";
                    L.device = dev;
                    L.is_rda = (il < static_cast<int>(cfg_.head_count_kv.size()) && cfg_.head_count_kv[il] == 0);

                    L.attn_norm = pool_.upload(sg, p + "attn_norm.weight", dev, false);
                    L.ffn_norm  = pool_.upload(sg, p + "ffn_norm.weight",  dev, false);

                    L.hc_attn_fn    = pool_.upload(sg, p + "hc_attn_fn.weight",    dev, false);
                    L.hc_attn_scale = pool_.upload(sg, p + "hc_attn_scale.weight", dev, false);
                    L.hc_attn_base  = pool_.upload(sg, p + "hc_attn_base.weight",  dev, false);
                    L.hc_ffn_fn     = pool_.upload(sg, p + "hc_ffn_fn.weight",     dev, false);
                    L.hc_ffn_scale  = pool_.upload(sg, p + "hc_ffn_scale.weight",  dev, false);
                    L.hc_ffn_base   = pool_.upload(sg, p + "hc_ffn_base.weight",   dev, false);

                    if (L.is_rda) {
                        L.wq          = pool_.upload(sg, p + "attn_q.weight",       dev, false);
                        L.wk          = pool_.upload(sg, p + "attn_k.weight",       dev, false);
                        L.wv          = pool_.upload(sg, p + "attn_v.weight",       dev, false);
                        L.wo          = pool_.upload(sg, p + "attn_output.weight",  dev, false);
                        L.ssm_conv_q  = pool_.upload(sg, p + "ssm_conv1d_q.weight", dev, false);
                        L.ssm_conv_k  = pool_.upload(sg, p + "ssm_conv1d_k.weight", dev, false);
                        L.ssm_conv_v  = pool_.upload(sg, p + "ssm_conv1d_v.weight", dev, false);
                        // Convert tiny BF16 RDA control projections to FP16 -> cublasHgemm on sm_60 + fused LDG.128 decode!
                        L.ssm_f_a     = pool_.upload(sg, p + "ssm_f_a.weight",      dev, true);
                        L.ssm_f_b     = pool_.upload(sg, p + "ssm_f_b.weight",      dev, true);
                        L.ssm_dt_bias = pool_.upload(sg, p + "ssm_dt.bias",         dev, false);
                        L.ssm_a       = pool_.upload(sg, p + "ssm_a",               dev, false);
                        L.ssm_beta    = pool_.upload(sg, p + "ssm_beta.weight",     dev, true);
                        L.ssm_rda     = pool_.upload(sg, p + "ssm_rda.weight",      dev, true);
                        L.ssm_g       = pool_.upload(sg, p + "ssm_g.weight",        dev, false);
                        L.ssm_norm    = pool_.upload(sg, p + "ssm_norm.weight",     dev, false);

                        const size_t conv_n = static_cast<size_t>(cfg_.rda_conv_kernel - 1) * d_inner;
                        const size_t ssm_n  = static_cast<size_t>(cfg_.n_head) * cfg_.rda_head_dim * cfg_.rda_head_dim;
                        rda_conv_q_[il] = dev_alloc<float>(dev, conv_n);
                        rda_conv_k_[il] = dev_alloc<float>(dev, conv_n);
                        rda_conv_v_[il] = dev_alloc<float>(dev, conv_n);
                        rda_state_[il]  = dev_alloc<float>(dev, ssm_n);
                    } else {
                        L.wq_a           = pool_.upload(sg, p + "attn_q_a.weight",      dev, false);
                        L.attn_q_a_norm  = pool_.upload(sg, p + "attn_q_a_norm.weight", dev, false);
                        L.wq_b           = pool_.upload(sg, p + "attn_q_b.weight",      dev, false);
                        L.wkv_a_mqa      = pool_.upload(sg, p + "attn_kv_a_mqa.weight", dev, false);
                        L.attn_kv_a_norm = pool_.upload(sg, p + "attn_kv_a_norm.weight",dev, false);
                        L.wk_b           = pool_.upload_as_f16(sg, p + "attn_k_b.weight", dev);
                        L.wv_b           = pool_.upload_as_f16(sg, p + "attn_v_b.weight", dev);
                        L.attn_gate      = pool_.upload(sg, p + "attn_gate.weight",     dev, false);
                        L.wo             = pool_.upload(sg, p + "attn_output.weight",   dev, false);
                    }

                    if (il < cfg_.n_dense_lead) {
                        L.ffn_gate = pool_.upload(sg, p + "ffn_gate.weight", dev, false);
                        L.ffn_up   = pool_.upload(sg, p + "ffn_up.weight",   dev, false);
                        L.ffn_down = pool_.upload(sg, p + "ffn_down.weight", dev, false);
                    } else {
                        L.ffn_gate_inp     = pool_.upload(sg, p + "ffn_gate_inp.weight",     dev, false);
                        L.ffn_exp_probs_b  = pool_.upload(sg, p + "exp_probs_b.bias",        dev, false);
                        L.ffn_gate_exps    = pool_.upload(sg, p + "ffn_gate_exps.weight",    dev, false);
                        L.ffn_up_exps      = pool_.upload(sg, p + "ffn_up_exps.weight",      dev, false);
                        L.ffn_down_exps    = pool_.upload(sg, p + "ffn_down_exps.weight",    dev, false);
                        L.ffn_gate_shexp   = pool_.upload(sg, p + "ffn_gate_shexp.weight",   dev, false);
                        L.ffn_up_shexp     = pool_.upload(sg, p + "ffn_up_shexp.weight",     dev, false);
                        L.ffn_down_shexp   = pool_.upload(sg, p + "ffn_down_shexp.weight",   dev, false);
                        L.ffn_effort_steer = pool_.upload(sg, p + "ffn_effort_steer.weight", dev, false, false);
                    }
                }
                cuda_ok(cudaDeviceSynchronize(), "raon1 parallel load sync");
            } catch (...) {
                load_errs[gi] = std::current_exception();
            }
        });
    }
    for (auto & th : load_threads) th.join();
    for (auto & ep : load_errs) if (ep) std::rethrow_exception(ep);

    // Release weight staging buffers BEFORE allocating KV caches & runtime scratch!
    pool_.release_staging();

    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (!layers_[il].is_rda) {
            cuda_ok(cudaSetDevice(layers_[il].device), "raon1 kv setDevice");
            cuda_ok(cudaMalloc(&mla_kv_cache_[il], mla_kv_bytes), "raon1 malloc mla_kv_cache");
        }
    }

    const int Tmax    = max_batch_;
    const int qk_nope = cfg_.qk_head_dim_mla - cfg_.qk_pe_dim; // 128
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        const int dev = rt.device;
        cuda_ok(cudaSetDevice(dev), "raon1 rt setDevice");
        cuda_ok(cudaStreamCreate(&rt.stream), "raon1 stream");
        cuda_ok(cudaEventCreateWithFlags(&rt.done_event, cudaEventDisableTiming), "raon1 event");
        if (cublasCreate(&rt.cublas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("raon1 cublasCreate failed");

        rt.d_stream     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
        rt.d_stream_tmp = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
        rt.d_hc_mixes   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * 17);
        rt.d_hc_post    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * 4);
        rt.d_hc_comb    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * 16);
        rt.d_x_norm     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
        rt.d_sub_out    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);

        rt.d_rda_q      = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_rda_k      = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_rda_v      = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_rda_qc     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_rda_kc     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_rda_vc     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_rda_fa     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.rda_decay_rank);
        rt.d_rda_decay  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_rda_beta   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head);
        rt.d_rda_delta  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head);
        rt.d_rda_out    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_rda_gate   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);

        rt.d_mla_qr     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.q_lora_rank);
        rt.d_mla_qall   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.qk_head_dim_mla);
        rt.d_mla_qnope  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * qk_nope);
        rt.d_mla_qpe    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.qk_pe_dim);
        rt.d_mla_qabs   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.kv_lora_rank);
        rt.d_mla_q576   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * kv_dim_tot);
        rt.d_mla_kvall  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * kv_dim_tot);
        rt.d_mla_olat   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.kv_lora_rank);
        rt.d_mla_o      = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.v_head_dim_mla);
        rt.d_mla_gate   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.v_head_dim_mla);

        if (rt.layer_start < cfg_.n_dense_lead) {
            rt.d_ffn_g = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_ff_dense);
            rt.d_ffn_u = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_ff_dense);
        }
        rt.d_router    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert);
        rt.d_top_ids   = dev_alloc<int32_t>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert_used);
        rt.d_top_w     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert_used);

        rt.d_tokens    = dev_alloc<int32_t>(dev, Tmax);
        rt.d_pos       = dev_alloc<int32_t>(dev, Tmax);
        if (dev == last_dev) {
            rt.d_logits = dev_alloc<float>(dev, static_cast<size_t>(cfg_.n_vocab));
            rt.d_argmax = dev_alloc<int32_t>(dev, 1);
        }

        // 128 MiB scratch (or 96 MiB on <=6GB GPUs) — easily fits T=512 grouped MoE & FlashMLA!
        size_t free_b = 0, tot_b = 0;
        cudaMemGetInfo(&free_b, &tot_b);
        rt.scratch_bytes = (free_b > (512ULL << 20)) ? (128ULL << 20) : (96ULL << 20);
        rt.d_scratch     = dev_alloc<uint8_t>(dev, rt.scratch_bytes);
    }

    reset_state();

    const auto t1 = std::chrono::steady_clock::now();
    std::printf("[Raon1Mini] Loaded %s (%d layers: %d dense + %d MoE[%d/%d, ff=%d], n_embd=%d, hc=%d, n_head=%d, ctx=%d, kv=%s, effort=%s[%.2f]) in %.2f s\n",
                sg.get_str("general.name", "Raon-1-mini").c_str(),
                cfg_.n_layers, cfg_.n_dense_lead, cfg_.n_layers - cfg_.n_dense_lead,
                cfg_.n_expert_used, cfg_.n_expert, cfg_.n_ff_exp,
                cfg_.n_embd, cfg_.hc_mult, cfg_.n_head,
                cfg_.max_ctx, kv_q8_0 ? "Q8_0" : "F16",
                cur_effort_.c_str(), effort_alpha_,
                std::chrono::duration<double>(t1 - t0).count());
    for (const auto & rt : runtimes_) {
        size_t free_b = 0, tot_b = 0;
        cudaSetDevice(rt.device);
        cudaMemGetInfo(&free_b, &tot_b);
        std::printf("  GPU%d: layers [%2d..%2d) (%2d layers), %.2f GiB weights, %.2f GiB free\n",
                    rt.device, rt.layer_start, rt.layer_end, rt.layer_end - rt.layer_start,
                    static_cast<double>(pool_.bytes_on_device(rt.device)) / (1024.0 * 1024.0 * 1024.0),
                    static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0));
    }
    std::fflush(stdout);
}

void Raon1MiniModel::reset_state() {
    const int d_inner = cfg_.n_head * cfg_.rda_head_dim;
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (!layers_[il].is_rda) continue;
        cuda_ok(cudaSetDevice(layers_[il].device), "raon1 reset setDevice");
        cuda_ok(cudaMemset(rda_conv_q_[il], 0, static_cast<size_t>(cfg_.rda_conv_kernel - 1) * d_inner * sizeof(float)), "zero cq");
        cuda_ok(cudaMemset(rda_conv_k_[il], 0, static_cast<size_t>(cfg_.rda_conv_kernel - 1) * d_inner * sizeof(float)), "zero ck");
        cuda_ok(cudaMemset(rda_conv_v_[il], 0, static_cast<size_t>(cfg_.rda_conv_kernel - 1) * d_inner * sizeof(float)), "zero cv");
        cuda_ok(cudaMemset(rda_state_[il],  0, static_cast<size_t>(cfg_.n_head) * cfg_.rda_head_dim * cfg_.rda_head_dim * sizeof(float)), "zero ssm");
    }
}

void Raon1MiniModel::save_rda_checkpoint(RdaCheckpoint & out, int pos) const {
    const size_t conv_n = static_cast<size_t>(cfg_.rda_conv_kernel - 1) * cfg_.n_head * cfg_.rda_head_dim;
    const size_t ssm_n  = static_cast<size_t>(cfg_.n_head) * cfg_.rda_head_dim * cfg_.rda_head_dim;
    out.pos = pos;
    out.conv_q.resize(cfg_.n_layers);
    out.conv_k.resize(cfg_.n_layers);
    out.conv_v.resize(cfg_.n_layers);
    out.ssm.resize(cfg_.n_layers);

    for (const auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "raon1 save_cp setDevice");
        for (int il = rt.layer_start; il < rt.layer_end; ++il) {
            if (!layers_[il].is_rda) continue;
            out.conv_q[il].resize(conv_n);
            out.conv_k[il].resize(conv_n);
            out.conv_v[il].resize(conv_n);
            out.ssm[il].resize(ssm_n);
            cuda_ok(cudaMemcpyAsync(out.conv_q[il].data(), rda_conv_q_[il], conv_n * sizeof(float), cudaMemcpyDeviceToHost, rt.stream), "save cq");
            cuda_ok(cudaMemcpyAsync(out.conv_k[il].data(), rda_conv_k_[il], conv_n * sizeof(float), cudaMemcpyDeviceToHost, rt.stream), "save ck");
            cuda_ok(cudaMemcpyAsync(out.conv_v[il].data(), rda_conv_v_[il], conv_n * sizeof(float), cudaMemcpyDeviceToHost, rt.stream), "save cv");
            cuda_ok(cudaMemcpyAsync(out.ssm[il].data(),    rda_state_[il],  ssm_n  * sizeof(float), cudaMemcpyDeviceToHost, rt.stream), "save ssm");
        }
    }
    for (const auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "raon1 save_cp sync dev");
        cuda_ok(cudaStreamSynchronize(rt.stream), "raon1 save_cp sync");
    }
}

void Raon1MiniModel::restore_rda_checkpoint(const RdaCheckpoint & cp) {
    const size_t conv_n = static_cast<size_t>(cfg_.rda_conv_kernel - 1) * cfg_.n_head * cfg_.rda_head_dim;
    const size_t ssm_n  = static_cast<size_t>(cfg_.n_head) * cfg_.rda_head_dim * cfg_.rda_head_dim;
    for (const auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "raon1 restore_cp setDevice");
        for (int il = rt.layer_start; il < rt.layer_end; ++il) {
            if (!layers_[il].is_rda) continue;
            cuda_ok(cudaMemcpyAsync(rda_conv_q_[il], cp.conv_q[il].data(), conv_n * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "rest cq");
            cuda_ok(cudaMemcpyAsync(rda_conv_k_[il], cp.conv_k[il].data(), conv_n * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "rest ck");
            cuda_ok(cudaMemcpyAsync(rda_conv_v_[il], cp.conv_v[il].data(), conv_n * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "rest cv");
            cuda_ok(cudaMemcpyAsync(rda_state_[il],  cp.ssm[il].data(),    ssm_n  * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "rest ssm");
        }
    }
    for (const auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "raon1 restore_cp sync dev");
        cuda_ok(cudaStreamSynchronize(rt.stream), "raon1 restore_cp sync");
    }
}

void Raon1MiniModel::run_gpu_segment(size_t gi, int T, bool run_argmax, bool compute_logits, int kv_len_max) {
    auto & rt = runtimes_[gi];
    cuda_ok(cudaSetDevice(rt.device), "raon1 seg setDevice");

    const int   d_inner   = cfg_.n_head * cfg_.rda_head_dim; // 5120
    const int   qk_nope   = cfg_.qk_head_dim_mla - cfg_.qk_pe_dim; // 128
    const float rda_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.rda_head_dim));
    const float mla_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.qk_head_dim_mla));

    const bool prof = (std::getenv("ROOKERY_PROFILE") != nullptr) && !use_cuda_graphs_;
    double ms_mhc = 0, ms_rda_proj = 0, ms_rda_conv = 0, ms_rda_rec = 0, ms_rda_out = 0;
    double ms_mla_proj = 0, ms_mla_prep = 0, ms_mla_attn = 0, ms_mla_out = 0;
    double ms_dense = 0, ms_router = 0, ms_moe = 0, ms_head = 0;
    auto mark = [&](double & acc, auto fn) {
        if (!prof) { fn(); return; }
        cudaStreamSynchronize(rt.stream);
        const auto a = std::chrono::steady_clock::now();
        fn();
        cudaStreamSynchronize(rt.stream);
        const auto b = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(b - a).count();
    };

    auto linear_fast = [&](const DeviceTensor & W, const float * d_x, float * d_y) {
        if (W.dtype == DType::IQ4_XS) {
            if (T >= 16 && try_launch_raon_linear_iq4_xs_sm60_f16(rt.cublas, W, d_x, d_y, T, rt.d_scratch, rt.scratch_bytes, rt.stream)) {
                return;
            }
            if (T < 16) {
                const DeviceTensor * wp[1] = { &W };
                float * yp[1] = { d_y };
                if (try_launch_raon_linear_multi_iq4_xs(wp, yp, 1, d_x, T, rt.d_scratch, rt.scratch_bytes, rt.stream)) {
                    return;
                }
            }
        }
        launch_linear(rt.cublas, W, d_x, d_y, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
    };

    // Ping-pong pointers so we never alias src/dst in RaonMHC mix
    float * s_cur = rt.d_stream;
    float * s_nxt = rt.d_stream_tmp;

    if (gi == 0) {
        // Expands embedding directly into all 4 streams: [T, 4, n_embd]!
        launch_embedding_lookup(tok_embd_, rt.d_tokens, s_cur, cfg_.n_embd, cfg_.hc_mult, T, rt.stream);
    }

    bool attn_mixes_ready = false;
    for (int il = rt.layer_start; il < rt.layer_end; ++il) {
        const auto & L = layers_[il];

        // --- Site 1: Attention RaonMHC route + aggregate + RMSNorm ---
        mark(ms_mhc, [&] {
            if (!attn_mixes_ready) {
                launch_hc_flat_rmsnorm_mix(s_cur, L.hc_attn_fn, rt.d_hc_mixes,
                                           cfg_.hc_mult * cfg_.n_embd, 17, T, cfg_.hc_eps, rt.stream);
            }
            attn_mixes_ready = false;
            launch_raon_mhc_pre_rmsnorm(s_cur, rt.d_hc_mixes,
                                        static_cast<const float *>(L.hc_attn_scale.ptr),
                                        static_cast<const float *>(L.hc_attn_base.ptr),
                                        static_cast<const float *>(L.attn_norm.ptr),
                                        rt.d_hc_post, rt.d_hc_comb, rt.d_x_norm,
                                        cfg_.n_embd, T, cfg_.hc_post_factor, cfg_.rms_eps, rt.stream);
        });

        // --- Attention branch: RDA or NoPE Gated MLA ---
        if (L.is_rda) {
            mark(ms_rda_proj, [&] {
                if (T < 16) {
                    const DeviceTensor * iq_ws[4] = { &L.wq, &L.wk, &L.wv, &L.ssm_g };
                    float * iq_ys[4] = { rt.d_rda_q, rt.d_rda_k, rt.d_rda_v, rt.d_rda_gate };
                    if (!try_launch_raon_linear_multi_iq4_xs(iq_ws, iq_ys, 4, rt.d_x_norm, T, rt.d_scratch, rt.scratch_bytes, rt.stream)) {
                        linear_fast(L.wq,    rt.d_x_norm, rt.d_rda_q);
                        linear_fast(L.wk,    rt.d_x_norm, rt.d_rda_k);
                        linear_fast(L.wv,    rt.d_x_norm, rt.d_rda_v);
                        linear_fast(L.ssm_g, rt.d_x_norm, rt.d_rda_gate);
                    }
                    const DeviceTensor * f16_ws[3] = { &L.ssm_beta, &L.ssm_rda, &L.ssm_f_a };
                    float * f16_ys[3] = { rt.d_rda_beta, rt.d_rda_delta, rt.d_rda_fa };
                    if (!try_launch_raon_linear_multi_f16(f16_ws, f16_ys, 3, rt.d_x_norm, T, rt.stream)) {
                        launch_linear(rt.cublas, L.ssm_beta, rt.d_x_norm, rt.d_rda_beta,  T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                        launch_linear(rt.cublas, L.ssm_rda,  rt.d_x_norm, rt.d_rda_delta, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                        launch_linear(rt.cublas, L.ssm_f_a,  rt.d_x_norm, rt.d_rda_fa,    T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                    }
                    const DeviceTensor * fb_w[1] = { &L.ssm_f_b };
                    float * fb_y[1] = { rt.d_rda_decay };
                    if (!try_launch_raon_linear_multi_f16(fb_w, fb_y, 1, rt.d_rda_fa, T, rt.stream)) {
                        launch_linear(rt.cublas, L.ssm_f_b, rt.d_rda_fa, rt.d_rda_decay, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                    }
                } else {
                    linear_fast(L.wq,    rt.d_x_norm, rt.d_rda_q);
                    linear_fast(L.wk,    rt.d_x_norm, rt.d_rda_k);
                    linear_fast(L.wv,    rt.d_x_norm, rt.d_rda_v);
                    linear_fast(L.ssm_g, rt.d_x_norm, rt.d_rda_gate);
                    launch_linear(rt.cublas, L.ssm_beta, rt.d_x_norm, rt.d_rda_beta,  T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                    launch_linear(rt.cublas, L.ssm_rda,  rt.d_x_norm, rt.d_rda_delta, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                    launch_linear(rt.cublas, L.ssm_f_a,  rt.d_x_norm, rt.d_rda_fa,    T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                    launch_linear(rt.cublas, L.ssm_f_b,  rt.d_rda_fa, rt.d_rda_decay, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                }
            });

            mark(ms_rda_conv, [&] {
                launch_ssm_conv1d_silu_qkv_l2(
                    rt.d_rda_q, rt.d_rda_k, rt.d_rda_v,
                    static_cast<const float *>(L.ssm_conv_q.ptr),
                    static_cast<const float *>(L.ssm_conv_k.ptr),
                    static_cast<const float *>(L.ssm_conv_v.ptr),
                    rda_conv_q_[il], rda_conv_k_[il], rda_conv_v_[il],
                    rt.d_rda_qc, rt.d_rda_kc, rt.d_rda_vc,
                    T, cfg_.n_head, 1e-6f, rt.stream);

                launch_raon_rda_gate_beta_vscale(
                    rt.d_rda_decay,
                    static_cast<const float *>(L.ssm_dt_bias.ptr),
                    static_cast<const float *>(L.ssm_a.ptr),
                    rt.d_rda_beta, rt.d_rda_delta, rt.d_rda_vc,
                    cfg_.rda_gate_lower_bound, T, cfg_.n_head, cfg_.rda_head_dim, rt.stream);
            });

            mark(ms_rda_rec, [&] {
                launch_gated_delta_net_d128(
                    rt.d_rda_qc, rt.d_rda_kc, rt.d_rda_vc,
                    rt.d_rda_decay, rt.d_rda_beta,
                    rda_state_[il], rt.d_rda_out,
                    T, cfg_.n_head, rda_scale, /*is_kda=*/true, rt.stream);
            });

            mark(ms_rda_out, [&] {
                launch_rms_norm_sigmoid_gate_d128(
                    rt.d_rda_out, static_cast<const float *>(L.ssm_norm.ptr), rt.d_rda_gate,
                    T, cfg_.n_head, cfg_.rms_eps, rt.stream);
                linear_fast(L.wo, rt.d_rda_out, rt.d_sub_out);
            });
        } else {
            mark(ms_mla_proj, [&] {
                if (T < 16) {
                    const DeviceTensor * ws[3] = { &L.wq_a, &L.wkv_a_mqa, &L.attn_gate };
                    float * ys[3] = { rt.d_mla_qr, rt.d_mla_kvall, rt.d_mla_gate };
                    if (!try_launch_linear_multi_sm60_h2(ws, ys, 3, rt.d_x_norm, T, rt.d_scratch, rt.scratch_bytes, rt.stream)) {
                        launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T, rt.stream);
                        launch_linear_multi_prequant(ws, ys, 3, rt.d_scratch, T, rt.stream);
                    }
                } else {
                    linear_fast(L.wq_a,      rt.d_x_norm, rt.d_mla_qr);
                    linear_fast(L.wkv_a_mqa, rt.d_x_norm, rt.d_mla_kvall);
                    linear_fast(L.attn_gate, rt.d_x_norm, rt.d_mla_gate);
                }
                launch_rmsnorm(rt.d_mla_qr, static_cast<const float *>(L.attn_q_a_norm.ptr), rt.d_mla_qr,
                               cfg_.q_lora_rank, T, cfg_.mla_latent_rms_eps, rt.stream);
                linear_fast(L.wq_b, rt.d_mla_qr, rt.d_mla_qall);
            });

            mark(ms_mla_prep, [&] {
                launch_raon_mla_split_q(rt.d_mla_qall, rt.d_mla_qnope, rt.d_mla_qpe,
                                        T, cfg_.n_head, qk_nope, cfg_.qk_pe_dim, rt.stream);
                launch_raon_mla_kv_norm_write_576(
                    rt.d_mla_kvall, static_cast<const float *>(L.attn_kv_a_norm.ptr),
                    rt.d_pos, mla_kv_cache_[il], cfg_.kv_q8_0,
                    T, cfg_.mla_latent_rms_eps, rt.stream);
                launch_batched_head_proj_f16(L.wk_b, rt.d_mla_qnope, rt.d_mla_qabs, T, rt.stream,
                                             rt.cublas, rt.d_scratch, rt.scratch_bytes);
                launch_mla_concat_q576(rt.d_mla_qabs, rt.d_mla_qpe, rt.d_mla_q576, T, cfg_.n_head, rt.stream);
            });

            mark(ms_mla_attn, [&] {
                launch_mla_attn_d576_v512(
                    rt.d_mla_q576, mla_kv_cache_[il], cfg_.kv_q8_0,
                    rt.d_pos, rt.d_mla_olat, rt.d_scratch,
                    T, cfg_.n_head, mla_scale, rt.stream,
                    rt.cublas, rt.scratch_bytes, kv_len_max);
            });

            mark(ms_mla_out, [&] {
                launch_batched_head_proj_f16(L.wv_b, rt.d_mla_olat, rt.d_mla_o, T, rt.stream,
                                             rt.cublas, rt.d_scratch, rt.scratch_bytes);
                launch_inkling2_sigmoid_gate_mul(rt.d_mla_o, rt.d_mla_gate,
                                                 static_cast<int64_t>(T) * d_inner, rt.stream);
                linear_fast(L.wo, rt.d_mla_o, rt.d_sub_out);
            });
        }

        // --- Site 1 write-back + Site 2 FFN RaonMHC route (fused!) + aggregate + RMSNorm ---
        mark(ms_mhc, [&] {
            launch_raon_mhc_post_steer_flat_mix(
                rt.d_sub_out, s_cur, rt.d_hc_post, rt.d_hc_comb,
                nullptr, 0.0f, s_nxt,
                L.hc_ffn_fn, rt.d_hc_mixes,
                cfg_.n_embd, 17, T, cfg_.hc_eps, rt.stream, rt.d_scratch, rt.scratch_bytes);
            std::swap(s_cur, s_nxt);
            launch_raon_mhc_pre_rmsnorm(s_cur, rt.d_hc_mixes,
                                        static_cast<const float *>(L.hc_ffn_scale.ptr),
                                        static_cast<const float *>(L.hc_ffn_base.ptr),
                                        static_cast<const float *>(L.ffn_norm.ptr),
                                        rt.d_hc_post, rt.d_hc_comb, rt.d_x_norm,
                                        cfg_.n_embd, T, cfg_.hc_post_factor, cfg_.rms_eps, rt.stream);
        });

        // --- FFN branch: Dense SiTU-GLU (layer 0) or MoE SiTU-GLU (layers 1..24) ---
        if (il < cfg_.n_dense_lead) {
            mark(ms_dense, [&] {
                if (T < 16) {
                    const DeviceTensor * ws[2] = { &L.ffn_gate, &L.ffn_up };
                    float * ys[2] = { rt.d_ffn_g, rt.d_ffn_u };
                    if (!try_launch_raon_linear_multi_iq4_xs(ws, ys, 2, rt.d_x_norm, T, rt.d_scratch, rt.scratch_bytes, rt.stream)) {
                        linear_fast(L.ffn_gate, rt.d_x_norm, rt.d_ffn_g);
                        linear_fast(L.ffn_up,   rt.d_x_norm, rt.d_ffn_u);
                    }
                } else {
                    linear_fast(L.ffn_gate, rt.d_x_norm, rt.d_ffn_g);
                    linear_fast(L.ffn_up,   rt.d_x_norm, rt.d_ffn_u);
                }
                launch_raon_situ_mul(rt.d_ffn_g, rt.d_ffn_u, rt.d_ffn_g,
                                     static_cast<int64_t>(T) * cfg_.n_ff_dense,
                                     cfg_.situ_beta, cfg_.situ_linear_beta, rt.stream);
                linear_fast(L.ffn_down, rt.d_ffn_g, rt.d_sub_out);
            });
        } else {
            mark(ms_router, [&] {
                launch_linear(rt.cublas, L.ffn_gate_inp, rt.d_x_norm, rt.d_router, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                launch_moe_route_sigmoid(
                    rt.d_router, static_cast<const float *>(L.ffn_exp_probs_b.ptr),
                    rt.d_top_ids, rt.d_top_w,
                    T, cfg_.n_expert, cfg_.n_expert_used,
                    cfg_.expert_weights_norm, cfg_.expert_weights_scale, rt.stream);
            });

            mark(ms_moe, [&] {
                launch_raon_moe_ffn_with_shared(
                    rt.cublas,
                    L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps,
                    L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
                    rt.d_x_norm, rt.d_top_ids, rt.d_top_w,
                    rt.d_sub_out, rt.d_scratch, rt.scratch_bytes,
                    T, cfg_.n_expert_used, cfg_.situ_beta, cfg_.situ_linear_beta, rt.stream);
            });
        }

        // --- Site 2 write-back (+ optional steering, fused with next layer's hc_attn_fn mix when no hook!) ---
        mark(ms_mhc, [&] {
            const float * steer_ptr = L.ffn_effort_steer.valid()
                ? static_cast<const float *>(L.ffn_effort_steer.ptr)
                : nullptr;
            const bool can_fuse_next = (il + 1 < rt.layer_end) && !(hook_ && hook_->hooks_layer(il));
            if (can_fuse_next) {
                launch_raon_mhc_post_steer_flat_mix(
                    rt.d_sub_out, s_cur, rt.d_hc_post, rt.d_hc_comb,
                    steer_ptr, effort_alpha_, s_nxt,
                    layers_[il + 1].hc_attn_fn, rt.d_hc_mixes,
                    cfg_.n_embd, 17, T, cfg_.hc_eps, rt.stream, rt.d_scratch, rt.scratch_bytes);
                attn_mixes_ready = true;
            } else {
                launch_raon_mhc_post_steer(
                    rt.d_sub_out, s_cur, rt.d_hc_post, rt.d_hc_comb,
                    steer_ptr, effort_alpha_, s_nxt, cfg_.n_embd, T, rt.stream);
            }
            std::swap(s_cur, s_nxt);
        });

        if (hook_ && hook_->hooks_layer(il)) {
            hook_->on_post_layer(il, rt.device, s_cur, T, cfg_.n_embd, cfg_.hc_mult, rt.stream);
        }
    }

    // Ensure final stream state lands in rt.d_stream (so peer copy & CUDA graph contracts stay fixed!)
    if (s_cur != rt.d_stream) {
        cuda_ok(cudaMemcpyAsync(rt.d_stream, s_cur,
                                static_cast<size_t>(T) * cfg_.hc_mult * cfg_.n_embd * sizeof(float),
                                cudaMemcpyDeviceToDevice, rt.stream), "raon1 stream copy");
    }

    // Final mean reduction over 4 streams + RMSNorm + untied lm_head on last GPU
    if (gi + 1 == runtimes_.size() && compute_logits) {
        mark(ms_head, [&] {
            const float * last_stream = rt.d_stream + static_cast<size_t>(T - 1) * cfg_.hc_mult * cfg_.n_embd;
            launch_hc_mean_rmsnorm(last_stream, static_cast<const float *>(output_norm_.ptr),
                                   rt.d_x_norm, cfg_.n_embd, 1, cfg_.rms_eps, rt.stream);
            launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, rt.stream);
            if (run_argmax) {
                launch_argmax_f32(rt.d_logits, rt.d_argmax, cfg_.n_vocab, rt.stream);
            }
        });
    }

    if (prof) {
        const double tot = ms_mhc + ms_rda_proj + ms_rda_conv + ms_rda_rec + ms_rda_out +
                           ms_mla_proj + ms_mla_prep + ms_mla_attn + ms_mla_out +
                           ms_dense + ms_router + ms_moe + ms_head;
        std::printf("[PROF GPU%d T=%d] tot=%.2fms | mhc=%.2f rda_p=%.2f rda_c=%.2f rda_r=%.2f rda_o=%.2f | mla_p=%.2f mla_pr=%.2f mla_a=%.2f mla_o=%.2f | dense=%.2f rtr=%.2f moe=%.2f head=%.2f\n",
                    rt.device, T, tot,
                    ms_mhc, ms_rda_proj, ms_rda_conv, ms_rda_rec, ms_rda_out,
                    ms_mla_proj, ms_mla_prep, ms_mla_attn, ms_mla_out,
                    ms_dense, ms_router, ms_moe, ms_head);
        std::fflush(stdout);
    }
}

void Raon1MiniModel::capture_decode_graphs() {
    invalidate_graphs();
    if (!use_cuda_graphs_) return;

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "raon1 cap setDevice");
        cuda_ok(cudaStreamSynchronize(rt.stream), "raon1 cap presync");

        cudaGraph_t g1 = nullptr;
        cuda_ok(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeGlobal), "raon1 cap begin T1");
        run_gpu_segment(gi, 1, true, true, 0);
        cuda_ok(cudaStreamEndCapture(rt.stream, &g1), "raon1 cap end T1");
        cuda_ok(cudaGraphInstantiate(&rt.decode_graph_t1, g1, nullptr, nullptr, 0), "raon1 cap inst T1");
        cudaGraphDestroy(g1);
    }
    graphs_ready_ = true;
}

void Raon1MiniModel::warmup_decode_graphs() {
    if (use_cuda_graphs_ && !graphs_ready_) {
        capture_decode_graphs();
    }
}

void Raon1MiniModel::run_pipeline(int T, bool run_argmax, bool compute_logits, int kv_len_max, bool prefill_sync) {
    const size_t n_gpus = runtimes_.size();
    const bool use_g1 = (T == 1 && run_argmax && compute_logits && use_cuda_graphs_ && graphs_ready_);

    for (size_t gi = 0; gi < n_gpus; ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "raon1 pipe setDevice");
        if (gi > 0) {
            auto & prev = runtimes_[gi - 1];
            cuda_ok(cudaStreamWaitEvent(rt.stream, prev.done_event, 0), "raon1 pipe wait");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_stream, rt.device, prev.d_stream, prev.device,
                                        static_cast<size_t>(T) * cfg_.hc_mult * cfg_.n_embd * sizeof(float),
                                        rt.stream), "raon1 pipe peer");
        }
        if (use_g1) {
            cuda_ok(cudaGraphLaunch(rt.decode_graph_t1, rt.stream), "raon1 launch g1");
        } else {
            run_gpu_segment(gi, T, run_argmax, compute_logits, kv_len_max);
        }
        if (gi + 1 < n_gpus) {
            cuda_ok(cudaEventRecord(rt.done_event, rt.stream), "raon1 pipe rec");
        }
        if (prefill_sync) {
            cuda_ok(cudaStreamSynchronize(rt.stream), "raon1 pipe sync");
        }
    }
}

void Raon1MiniModel::forward(const std::vector<int32_t> & tokens,
                             const std::vector<int32_t> & positions,
                             std::vector<float> & out_logits,
                             bool compute_logits) {
    const int T = static_cast<int>(tokens.size());
    if (T <= 0) return;
    if (T > max_batch_) throw std::runtime_error("Raon1MiniModel::forward: T exceeds max_batch");

    const int kv_len_max = *std::max_element(positions.begin(), positions.end()) + 1;
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "raon1 fwd setDevice");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, tokens.data(), static_cast<size_t>(T) * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, rt.stream), "raon1 fwd tok H2D");
        }
        cuda_ok(cudaMemcpyAsync(rt.d_pos, positions.data(), static_cast<size_t>(T) * sizeof(int32_t),
                                cudaMemcpyHostToDevice, rt.stream), "raon1 fwd pos H2D");
    }

    run_pipeline(T, false, compute_logits, kv_len_max, T >= 16);

    auto & last_rt = runtimes_.back();
    cuda_ok(cudaSetDevice(last_rt.device), "raon1 fwd last dev");
    if (compute_logits) {
        out_logits.resize(static_cast<size_t>(cfg_.n_vocab));
        cuda_ok(cudaMemcpyAsync(out_logits.data(), last_rt.d_logits,
                                static_cast<size_t>(cfg_.n_vocab) * sizeof(float),
                                cudaMemcpyDeviceToHost, last_rt.stream), "raon1 fwd logits D2H");
    }
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "raon1 fwd sync");
}

int32_t Raon1MiniModel::forward_decode_greedy(int32_t token, int32_t pos) {
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "raon1 dec setDevice");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "raon1 dec tok");
        }
        cuda_ok(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "raon1 dec pos");
    }

    run_pipeline(1, true, true, pos + 1, false);

    auto & last_rt = runtimes_.back();
    int32_t out_tok = 0;
    cuda_ok(cudaSetDevice(last_rt.device), "raon1 dec last dev");
    cuda_ok(cudaMemcpyAsync(&out_tok, last_rt.d_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, last_rt.stream), "raon1 dec argmax");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "raon1 dec sync");
    return out_tok;
}

void Raon1MiniModel::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "raon1 declog setDevice");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "raon1 declog tok");
        }
        cuda_ok(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "raon1 declog pos");
    }

    // Reuse T=1 CUDA graph (argmax is tiny <5us) so sampled decode stays graph-accelerated!
    run_pipeline(1, true, true, pos + 1, false);

    auto & last_rt = runtimes_.back();
    out_logits.resize(static_cast<size_t>(cfg_.n_vocab));
    cuda_ok(cudaSetDevice(last_rt.device), "raon1 declog last dev");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), last_rt.d_logits,
                            static_cast<size_t>(cfg_.n_vocab) * sizeof(float),
                            cudaMemcpyDeviceToHost, last_rt.stream), "raon1 declog D2H");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "raon1 declog sync");
}

} // namespace rookery
