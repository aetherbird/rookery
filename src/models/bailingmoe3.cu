#include "models/bailingmoe3.h"
#include "kernels/kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace rookery {

namespace {

inline void cuda_ok(cudaError_t err, const char * msg) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

template <typename T>
T * dev_alloc(int dev, size_t n) {
    cuda_ok(cudaSetDevice(dev), "bailingmoe3 dev_alloc setDevice");
    T * p = nullptr;
    cuda_ok(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)), "bailingmoe3 cudaMalloc");
    return p;
}

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

void BailingMoe3Model::invalidate_graphs() noexcept {
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        if (rt.decode_graph_t1) { cudaGraphExecDestroy(rt.decode_graph_t1); rt.decode_graph_t1 = nullptr; }
    }
    graphs_ready_ = false;
}

BailingMoe3Model::~BailingMoe3Model() {
    invalidate_graphs();
    for (size_t il = 0; il < layers_.size(); ++il) {
        cudaSetDevice(layers_[il].device);
        auto freep = [](void * p) { if (p) cudaFree(p); };
        if (il < kda_conv_q_.size())   freep(kda_conv_q_[il]);
        if (il < kda_conv_k_.size())   freep(kda_conv_k_[il]);
        if (il < kda_conv_v_.size())   freep(kda_conv_v_[il]);
        if (il < kda_state_.size())    freep(kda_state_[il]);
        if (il < mla_kv_cache_.size()) freep(mla_kv_cache_[il]);
    }
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        auto freep = [](void * p) { if (p) cudaFree(p); };
        freep(rt.d_cur); freep(rt.d_x_norm); freep(rt.d_sub_out);
        freep(rt.d_kda_q); freep(rt.d_kda_k); freep(rt.d_kda_v);
        freep(rt.d_kda_qc); freep(rt.d_kda_kc); freep(rt.d_kda_vc);
        freep(rt.d_kda_g); freep(rt.d_kda_beta);
        freep(rt.d_kda_out); freep(rt.d_kda_gate);
        freep(rt.d_mla_qr); freep(rt.d_mla_qall); freep(rt.d_mla_qnope); freep(rt.d_mla_qpe);
        freep(rt.d_mla_qabs); freep(rt.d_mla_q576); freep(rt.d_mla_kvall);
        freep(rt.d_mla_olat); freep(rt.d_mla_o); freep(rt.d_mla_gate);
        freep(rt.d_ffn_g); freep(rt.d_ffn_u); freep(rt.d_router);
        freep(rt.d_top_ids); freep(rt.d_top_w);
        freep(rt.d_tokens); freep(rt.d_pos); freep(rt.d_logits); freep(rt.d_argmax);
        freep(rt.d_scratch);
        if (rt.done_event) cudaEventDestroy(rt.done_event);
        if (rt.cublas) cublasDestroy(rt.cublas);
        if (rt.stream) cudaStreamDestroy(rt.stream);
    }
}

void BailingMoe3Model::load(const std::string & gguf_path,
                            const std::vector<int> & gpu_ids,
                            const std::vector<float> & split_weights,
                            int max_ctx,
                            bool kv_q8_0) {
    if (gpu_ids.empty()) throw std::runtime_error("BailingMoe3Model::load requires >= 1 GPU");

    const auto t0 = std::chrono::steady_clock::now();
    auto sg = ShardedGguf::open(gguf_path);
    const std::string arch = sg.get_str("general.architecture");
    if (arch != "bailingmoe3") {
        throw std::runtime_error("Expected bailingmoe3 GGUF, got: " + arch);
    }
    const std::string pfx = arch + ".";

    cfg_.n_blocks             = static_cast<int>(sg.get_u64(pfx + "block_count", 24));
    cfg_.n_nextn              = static_cast<int>(sg.get_u64(pfx + "nextn_predict_layers", 0));
    cfg_.n_layers             = cfg_.n_blocks - cfg_.n_nextn;
    cfg_.n_dense_lead         = static_cast<int>(sg.get_u64(pfx + "leading_dense_block_count", 1));
    cfg_.n_embd               = static_cast<int>(sg.get_u64(pfx + "embedding_length", 1536));
    cfg_.n_ff_dense           = static_cast<int>(sg.get_u64(pfx + "feed_forward_length", 4608));
    cfg_.n_vocab              = static_cast<int>(sg.get_u64(pfx + "vocab_size", 157184));
    cfg_.n_head               = static_cast<int>(sg.get_u64(pfx + "attention.head_count", 16));
    cfg_.kda_head_dim         = static_cast<int>(sg.get_u64(pfx + "kda.head_dim", 128));
    cfg_.kda_gate_lower_bound = static_cast<float>(sg.get_f64(pfx + "kda.gate_lower_bound", -5.0));
    cfg_.q_lora_rank          = static_cast<int>(sg.get_u64(pfx + "attention.q_lora_rank", 0));
    cfg_.kv_lora_rank         = static_cast<int>(sg.get_u64(pfx + "attention.kv_lora_rank", 512));
    cfg_.rope_dim             = static_cast<int>(sg.get_u64(pfx + "rope.dimension_count", 64));
    cfg_.qk_head_dim_mla      = static_cast<int>(sg.get_u64(pfx + "attention.key_length_mla", 192));
    cfg_.v_head_dim_mla       = static_cast<int>(sg.get_u64(pfx + "attention.value_length_mla", 128));
    cfg_.rope_freq_base       = static_cast<float>(sg.get_f64(pfx + "rope.freq_base", 6000000.0));
    cfg_.n_expert             = static_cast<int>(sg.get_u64(pfx + "expert_count", 128));
    cfg_.n_expert_used        = static_cast<int>(sg.get_u64(pfx + "expert_used_count", 8));
    cfg_.n_ff_exp             = static_cast<int>(sg.get_u64(pfx + "expert_feed_forward_length", 512));
    cfg_.n_ff_shexp           = static_cast<int>(sg.get_u64(pfx + "expert_shared_feed_forward_length", cfg_.n_ff_exp));
    cfg_.expert_weights_norm  = sg.get_bool(pfx + "expert_weights_norm", true);
    cfg_.expert_weights_scale = static_cast<float>(sg.get_f64(pfx + "expert_weights_scale", 2.5));
    if (cfg_.expert_weights_scale == 0.0f) cfg_.expert_weights_scale = 1.0f;
    cfg_.rms_eps              = static_cast<float>(sg.get_f64(pfx + "attention.layer_norm_rms_epsilon", 1e-6));
    cfg_.max_ctx              = max_ctx;
    cfg_.kv_q8_0              = kv_q8_0;

    cfg_.head_count_kv      = read_num_arr<int>(sg, pfx + "attention.head_count_kv", cfg_.n_blocks, 0);
    cfg_.swiglu_clamp_exp   = read_num_arr<float>(sg, pfx + "swiglu_clamp_exp", cfg_.n_blocks, 0.0f);
    cfg_.swiglu_clamp_shexp = read_num_arr<float>(sg, pfx + "swiglu_clamp_shexp", cfg_.n_blocks, 0.0f);

    if (cfg_.kv_lora_rank != 512 || cfg_.rope_dim != 64) {
        throw std::runtime_error("BailingMoe3Model expects kv_lora_rank=512, rope_dim=64");
    }

    // Partition backbone layers across GPUs proportionally to split_weights (or free VRAM!)
    std::vector<float> sw = split_weights;
    if (sw.size() != gpu_ids.size()) {
        sw.resize(gpu_ids.size());
        for (size_t g = 0; g < gpu_ids.size(); ++g) {
            cudaSetDevice(gpu_ids[g]);
            size_t free_b = 0, tot_b = 0;
            cudaMemGetInfo(&free_b, &tot_b);
            sw[g] = std::max(1.0f, static_cast<float>(free_b) / (1024.0f * 1024.0f * 1024.0f) - 1.5f);
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
    kda_conv_q_.resize(cfg_.n_layers, nullptr);
    kda_conv_k_.resize(cfg_.n_layers, nullptr);
    kda_conv_v_.resize(cfg_.n_layers, nullptr);
    kda_state_.resize(cfg_.n_layers, nullptr);
    mla_kv_cache_.resize(cfg_.n_layers, nullptr);

    const int d_inner = cfg_.n_head * cfg_.kda_head_dim;
    const int kv_dim_tot = cfg_.kv_lora_rank + cfg_.rope_dim; // 576
    const size_t mla_kv_bytes = kv_q8_0
        ? static_cast<size_t>(cfg_.max_ctx) * (kv_dim_tot / 32) * sizeof(BlockQ8_0)
        : static_cast<size_t>(cfg_.max_ctx) * kv_dim_tot * sizeof(uint16_t);

    // Parallel multi-GPU weight upload
    std::vector<std::thread> load_threads;
    std::vector<std::exception_ptr> load_errs(runtimes_.size(), nullptr);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        load_threads.emplace_back([&, gi]() {
            try {
                const int dev = runtimes_[gi].device;
                cuda_ok(cudaSetDevice(dev), "bailing3 parallel load setDevice");
                if (gi == 0) {
                    tok_embd_ = pool_.upload(sg, "token_embd.weight", dev, false);
                }
                if (gi + 1 == runtimes_.size()) {
                    output_norm_ = pool_.upload(sg, "output_norm.weight", dev, false);
                    output_      = pool_.upload(sg, "output.weight",      dev, false, false);
                    if (!output_.valid()) output_ = pool_.upload(sg, "token_embd.weight", dev, false);
                }
                for (int il = runtimes_[gi].layer_start; il < runtimes_[gi].layer_end; ++il) {
                    auto & L = layers_[il];
                    const std::string p = "blk." + std::to_string(il) + ".";
                    L.device = dev;
                    L.is_kda = (il < static_cast<int>(cfg_.head_count_kv.size()) && cfg_.head_count_kv[il] == 0);

                    L.attn_norm = pool_.upload(sg, p + "attn_norm.weight", dev, false);
                    L.ffn_norm  = pool_.upload(sg, p + "ffn_norm.weight",  dev, false);

                    if (L.is_kda) {
                        L.wq          = pool_.upload(sg, p + "attn_q.weight",        dev, false);
                        L.wk          = pool_.upload(sg, p + "attn_k.weight",        dev, false);
                        L.wv          = pool_.upload(sg, p + "attn_v.weight",        dev, false);
                        L.wo          = pool_.upload(sg, p + "attn_output.weight",   dev, false);
                        L.ssm_conv_q  = pool_.upload(sg, p + "ssm_conv1d_q.weight",  dev, false);
                        L.ssm_conv_k  = pool_.upload(sg, p + "ssm_conv1d_k.weight",  dev, false);
                        L.ssm_conv_v  = pool_.upload(sg, p + "ssm_conv1d_v.weight",  dev, false);
                        L.ssm_f_a     = pool_.upload(sg, p + "ssm_f_a.weight",       dev, false);
                        L.ssm_dt_bias = pool_.upload(sg, p + "ssm_dt.bias",          dev, false);
                        L.ssm_a       = pool_.upload(sg, p + "ssm_a",                dev, false);
                        L.ssm_beta    = pool_.upload(sg, p + "ssm_beta.weight",      dev, false);
                        L.ssm_g_a     = pool_.upload(sg, p + "ssm_g_a.weight",       dev, false);
                        L.ssm_norm    = pool_.upload(sg, p + "ssm_norm.weight",      dev, false);

                        const size_t conv_n = static_cast<size_t>(3) * d_inner;
                        const size_t ssm_n  = static_cast<size_t>(cfg_.n_head) * cfg_.kda_head_dim * cfg_.kda_head_dim;
                        kda_conv_q_[il] = dev_alloc<float>(dev, conv_n);
                        kda_conv_k_[il] = dev_alloc<float>(dev, conv_n);
                        kda_conv_v_[il] = dev_alloc<float>(dev, conv_n);
                        kda_state_[il]  = dev_alloc<float>(dev, ssm_n);
                    } else {
                        if (cfg_.q_lora_rank > 0) {
                            L.wq_a          = pool_.upload(sg, p + "attn_q_a.weight",      dev, false);
                            L.attn_q_a_norm = pool_.upload(sg, p + "attn_q_a_norm.weight", dev, false);
                            L.wq_b          = pool_.upload(sg, p + "attn_q_b.weight",      dev, false);
                        } else {
                            L.wq            = pool_.upload(sg, p + "attn_q.weight",        dev, false);
                        }
                        L.wkv_a_mqa      = pool_.upload(sg, p + "attn_kv_a_mqa.weight",  dev, false);
                        L.attn_kv_a_norm = pool_.upload(sg, p + "attn_kv_a_norm.weight", dev, false);
                        L.wk_b           = pool_.upload_as_f16(sg, p + "attn_k_b.weight", dev);
                        L.wv_b           = pool_.upload_as_f16(sg, p + "attn_v_b.weight", dev);
                        L.attn_gate      = pool_.upload(sg, p + "attn_gate.weight",      dev, false);
                        L.wo             = pool_.upload(sg, p + "attn_output.weight",    dev, false);

                        cuda_ok(cudaMalloc(&mla_kv_cache_[il], mla_kv_bytes), "malloc bailing3 mla_kv_cache");
                    }

                    if (il < cfg_.n_dense_lead) {
                        L.ffn_gate = pool_.upload(sg, p + "ffn_gate.weight", dev, false);
                        L.ffn_up   = pool_.upload(sg, p + "ffn_up.weight",   dev, false);
                        L.ffn_down = pool_.upload(sg, p + "ffn_down.weight", dev, false);
                    } else {
                        L.ffn_gate_inp    = pool_.upload(sg, p + "ffn_gate_inp.weight",   dev, false);
                        L.ffn_exp_probs_b = pool_.upload(sg, p + "exp_probs_b.bias",      dev, false);
                        L.ffn_gate_exps   = pool_.upload(sg, p + "ffn_gate_exps.weight",  dev, false);
                        L.ffn_up_exps     = pool_.upload(sg, p + "ffn_up_exps.weight",    dev, false);
                        L.ffn_down_exps   = pool_.upload(sg, p + "ffn_down_exps.weight",  dev, false);
                        L.ffn_gate_shexp  = pool_.upload(sg, p + "ffn_gate_shexp.weight", dev, false);
                        L.ffn_up_shexp    = pool_.upload(sg, p + "ffn_up_shexp.weight",   dev, false);
                        L.ffn_down_shexp  = pool_.upload(sg, p + "ffn_down_shexp.weight", dev, false);
                    }
                }
                cuda_ok(cudaDeviceSynchronize(), "bailing3 parallel load sync");
            } catch (...) {
                load_errs[gi] = std::current_exception();
            }
        });
    }
    for (auto & th : load_threads) th.join();
    for (auto & ep : load_errs) if (ep) std::rethrow_exception(ep);

    pool_.release_staging();

    const int Tmax    = max_batch_;
    const int qk_nope = cfg_.qk_head_dim_mla - cfg_.rope_dim;
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        const int dev = rt.device;
        cuda_ok(cudaSetDevice(dev), "bailing3 rt setDevice");
        cuda_ok(cudaStreamCreate(&rt.stream), "stream");
        cuda_ok(cudaEventCreateWithFlags(&rt.done_event, cudaEventDisableTiming), "event");
        if (cublasCreate(&rt.cublas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");

        rt.d_cur      = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
        rt.d_x_norm   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
        rt.d_sub_out  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);

        rt.d_kda_q    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_k    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_v    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_qc   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_kc   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_vc   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_g    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_beta = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head);
        rt.d_kda_out  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_gate = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);

        rt.d_mla_qr    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * std::max(1, cfg_.q_lora_rank));
        rt.d_mla_qall  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.qk_head_dim_mla);
        rt.d_mla_qnope = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * qk_nope);
        rt.d_mla_qpe   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.rope_dim);
        rt.d_mla_qabs  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.kv_lora_rank);
        rt.d_mla_q576  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * kv_dim_tot);
        rt.d_mla_kvall = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * kv_dim_tot);
        rt.d_mla_olat  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.kv_lora_rank);
        rt.d_mla_o     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.v_head_dim_mla);
        rt.d_mla_gate  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head);

        rt.d_ffn_g     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_ff_dense);
        rt.d_ffn_u     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_ff_dense);
        rt.d_router    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert);
        rt.d_top_ids   = dev_alloc<int32_t>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert_used);
        rt.d_top_w     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert_used);

        rt.d_tokens    = dev_alloc<int32_t>(dev, Tmax);
        rt.d_pos       = dev_alloc<int32_t>(dev, Tmax);
        if (dev == last_dev) {
            rt.d_logits = dev_alloc<float>(dev, static_cast<size_t>(cfg_.n_vocab));
            rt.d_argmax = dev_alloc<int32_t>(dev, 1);
        }
        rt.scratch_bytes = 96ULL << 20;
        rt.d_scratch = dev_alloc<uint8_t>(dev, rt.scratch_bytes);
    }

    reset_state();

    const auto t1 = std::chrono::steady_clock::now();
    std::printf("[BailingMoE3] Loaded %s (%d layers: %d dense + %d MoE[%d/%d], n_embd=%d, n_head=%d, q_lora=%d, ctx=%d, kv=%s) in %.2f s\n",
                sg.get_str("general.name", "bailingmoe3").c_str(),
                cfg_.n_layers, cfg_.n_dense_lead, cfg_.n_layers - cfg_.n_dense_lead,
                cfg_.n_expert_used, cfg_.n_expert, cfg_.n_embd, cfg_.n_head, cfg_.q_lora_rank,
                cfg_.max_ctx, kv_q8_0 ? "Q8_0" : "F16",
                std::chrono::duration<double>(t1 - t0).count());
    for (const auto & rt : runtimes_) {
        std::printf("  GPU%d: layers [%2d..%2d) (%2d layers), %.2f GiB weights\n",
                    rt.device, rt.layer_start, rt.layer_end, rt.layer_end - rt.layer_start,
                    static_cast<double>(pool_.bytes_on_device(rt.device)) / (1024.0 * 1024.0 * 1024.0));
    }
}

void BailingMoe3Model::reset_state() {
    const int d_inner = cfg_.n_head * cfg_.kda_head_dim;
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (!layers_[il].is_kda) continue;
        cuda_ok(cudaSetDevice(layers_[il].device), "bailing3 reset setDevice");
        cuda_ok(cudaMemset(kda_conv_q_[il], 0, static_cast<size_t>(3) * d_inner * sizeof(float)), "zero cq");
        cuda_ok(cudaMemset(kda_conv_k_[il], 0, static_cast<size_t>(3) * d_inner * sizeof(float)), "zero ck");
        cuda_ok(cudaMemset(kda_conv_v_[il], 0, static_cast<size_t>(3) * d_inner * sizeof(float)), "zero cv");
        cuda_ok(cudaMemset(kda_state_[il],  0, static_cast<size_t>(cfg_.n_head) * cfg_.kda_head_dim * cfg_.kda_head_dim * sizeof(float)), "zero ssm");
    }
}

void BailingMoe3Model::save_kda_checkpoint(KdaCheckpoint & out, int pos) const {
    const size_t conv_n = static_cast<size_t>(3) * cfg_.n_head * cfg_.kda_head_dim;
    const size_t ssm_n  = static_cast<size_t>(cfg_.n_head) * cfg_.kda_head_dim * cfg_.kda_head_dim;
    out.pos = pos;
    out.conv_q.resize(cfg_.n_layers);
    out.conv_k.resize(cfg_.n_layers);
    out.conv_v.resize(cfg_.n_layers);
    out.ssm.resize(cfg_.n_layers);

    for (const auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "bailing3 save_cp setDevice");
        for (int il = rt.layer_start; il < rt.layer_end; ++il) {
            if (!layers_[il].is_kda) continue;
            out.conv_q[il].resize(conv_n);
            out.conv_k[il].resize(conv_n);
            out.conv_v[il].resize(conv_n);
            out.ssm[il].resize(ssm_n);
            cuda_ok(cudaMemcpyAsync(out.conv_q[il].data(), kda_conv_q_[il], conv_n * sizeof(float), cudaMemcpyDeviceToHost, rt.stream), "save cq");
            cuda_ok(cudaMemcpyAsync(out.conv_k[il].data(), kda_conv_k_[il], conv_n * sizeof(float), cudaMemcpyDeviceToHost, rt.stream), "save ck");
            cuda_ok(cudaMemcpyAsync(out.conv_v[il].data(), kda_conv_v_[il], conv_n * sizeof(float), cudaMemcpyDeviceToHost, rt.stream), "save cv");
            cuda_ok(cudaMemcpyAsync(out.ssm[il].data(),    kda_state_[il],  ssm_n  * sizeof(float), cudaMemcpyDeviceToHost, rt.stream), "save ssm");
        }
    }
    for (const auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "bailing3 save_cp sync dev");
        cuda_ok(cudaStreamSynchronize(rt.stream), "bailing3 save_cp sync");
    }
}

void BailingMoe3Model::restore_kda_checkpoint(const KdaCheckpoint & cp) {
    const size_t conv_n = static_cast<size_t>(3) * cfg_.n_head * cfg_.kda_head_dim;
    const size_t ssm_n  = static_cast<size_t>(cfg_.n_head) * cfg_.kda_head_dim * cfg_.kda_head_dim;
    for (const auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "bailing3 restore_cp setDevice");
        for (int il = rt.layer_start; il < rt.layer_end; ++il) {
            if (!layers_[il].is_kda) continue;
            cuda_ok(cudaMemcpyAsync(kda_conv_q_[il], cp.conv_q[il].data(), conv_n * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "rest cq");
            cuda_ok(cudaMemcpyAsync(kda_conv_k_[il], cp.conv_k[il].data(), conv_n * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "rest ck");
            cuda_ok(cudaMemcpyAsync(kda_conv_v_[il], cp.conv_v[il].data(), conv_n * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "rest cv");
            cuda_ok(cudaMemcpyAsync(kda_state_[il],  cp.ssm[il].data(),    ssm_n  * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "rest ssm");
        }
    }
    for (const auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "bailing3 restore_cp sync dev");
        cuda_ok(cudaStreamSynchronize(rt.stream), "bailing3 restore_cp sync");
    }
}

void BailingMoe3Model::run_gpu_segment(size_t gi, int T, bool run_argmax, int kv_len_max) {
    auto & rt = runtimes_[gi];
    cuda_ok(cudaSetDevice(rt.device), "bailing3 seg setDevice");

    const int   d_inner   = cfg_.n_head * cfg_.kda_head_dim;
    const int   qk_nope   = cfg_.qk_head_dim_mla - cfg_.rope_dim;
    const float kda_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.kda_head_dim));
    const float mla_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.qk_head_dim_mla));

    if (gi == 0) {
        launch_embedding_lookup(tok_embd_, rt.d_tokens, rt.d_cur, cfg_.n_embd, 1, T, rt.stream);
    }

    auto linear_preq = [&](const DeviceTensor & W, const float * d_x, float * d_y) {
        if (T < 16 && (W.dtype == DType::Q8_0 || is_kquant_dtype(W.dtype))) {
            launch_linear_prequant(W, rt.d_scratch, d_y, T, rt.stream);
        } else {
            launch_linear(rt.cublas, W, d_x, d_y, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
        }
    };

    for (int il = rt.layer_start; il < rt.layer_end; ++il) {
        const auto & L = layers_[il];

        launch_rmsnorm(rt.d_cur, static_cast<const float *>(L.attn_norm.ptr), rt.d_x_norm, cfg_.n_embd, T, cfg_.rms_eps, rt.stream);

        if (L.is_kda) {
            if (T < 16) launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T, rt.stream);
            linear_preq(L.wq,       rt.d_x_norm, rt.d_kda_q);
            linear_preq(L.wk,       rt.d_x_norm, rt.d_kda_k);
            linear_preq(L.wv,       rt.d_x_norm, rt.d_kda_v);
            linear_preq(L.ssm_f_a,  rt.d_x_norm, rt.d_kda_g);
            linear_preq(L.ssm_beta, rt.d_x_norm, rt.d_kda_beta);
            linear_preq(L.ssm_g_a,  rt.d_x_norm, rt.d_kda_gate);

            launch_ssm_conv1d_silu(rt.d_kda_q, static_cast<const float *>(L.ssm_conv_q.ptr), kda_conv_q_[il], rt.d_kda_qc, T, d_inner, rt.stream);
            launch_ssm_conv1d_silu(rt.d_kda_k, static_cast<const float *>(L.ssm_conv_k.ptr), kda_conv_k_[il], rt.d_kda_kc, T, d_inner, rt.stream);
            launch_ssm_conv1d_silu(rt.d_kda_v, static_cast<const float *>(L.ssm_conv_v.ptr), kda_conv_v_[il], rt.d_kda_vc, T, d_inner, rt.stream);

            launch_l2_norm_d128(rt.d_kda_qc, T, cfg_.n_head, cfg_.rms_eps, rt.stream);
            launch_l2_norm_d128(rt.d_kda_kc, T, cfg_.n_head, cfg_.rms_eps, rt.stream);

            launch_kda_gate_and_beta(
                rt.d_kda_g, static_cast<const float *>(L.ssm_dt_bias.ptr),
                static_cast<const float *>(L.ssm_a.ptr), rt.d_kda_beta,
                cfg_.kda_gate_lower_bound, T, cfg_.n_head, cfg_.kda_head_dim, rt.stream, /*negate_a=*/false);

            launch_gated_delta_net_d128(
                rt.d_kda_qc, rt.d_kda_kc, rt.d_kda_vc, rt.d_kda_g, rt.d_kda_beta,
                kda_state_[il], rt.d_kda_out, T, cfg_.n_head, kda_scale, true, rt.stream);

            launch_rms_norm_sigmoid_gate_d128(rt.d_kda_out, static_cast<const float *>(L.ssm_norm.ptr),
                                              rt.d_kda_gate, T, cfg_.n_head, cfg_.rms_eps, rt.stream);
            launch_linear(rt.cublas, L.wo, rt.d_kda_out, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
        } else {
            if (T < 16) launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T, rt.stream);
            if (cfg_.q_lora_rank > 0) {
                linear_preq(L.wq_a, rt.d_x_norm, rt.d_mla_qr);
            } else {
                linear_preq(L.wq,   rt.d_x_norm, rt.d_mla_qall);
            }
            linear_preq(L.wkv_a_mqa, rt.d_x_norm, rt.d_mla_kvall);
            linear_preq(L.attn_gate, rt.d_x_norm, rt.d_mla_gate);

            if (cfg_.q_lora_rank > 0) {
                launch_rmsnorm(rt.d_mla_qr, static_cast<const float *>(L.attn_q_a_norm.ptr), rt.d_mla_qr, cfg_.q_lora_rank, T, cfg_.rms_eps, rt.stream);
                launch_linear(rt.cublas, L.wq_b, rt.d_mla_qr, rt.d_mla_qall, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            }

            launch_mla_rope_split_q(rt.d_mla_qall, rt.d_pos, rt.d_mla_qnope, rt.d_mla_qpe,
                                    T, cfg_.n_head, qk_nope, cfg_.rope_dim, cfg_.rope_freq_base, rt.stream);
            launch_mla_kv_norm_rope_write_576(rt.d_mla_kvall, static_cast<const float *>(L.attn_kv_a_norm.ptr),
                                              rt.d_pos, mla_kv_cache_[il], cfg_.kv_q8_0,
                                              T, cfg_.rms_eps, cfg_.rope_freq_base, rt.stream);

            launch_batched_head_proj_f16(L.wk_b, rt.d_mla_qnope, rt.d_mla_qabs, T, rt.stream, rt.cublas, rt.d_scratch, rt.scratch_bytes);
            launch_mla_concat_q576(rt.d_mla_qabs, rt.d_mla_qpe, rt.d_mla_q576, T, cfg_.n_head, rt.stream);

            launch_mla_attn_d576_v512(rt.d_mla_q576, mla_kv_cache_[il], cfg_.kv_q8_0,
                                      rt.d_pos, rt.d_mla_olat, rt.d_scratch, T, cfg_.n_head, mla_scale, rt.stream,
                                      rt.cublas, rt.scratch_bytes, kv_len_max);

            launch_batched_head_proj_f16(L.wv_b, rt.d_mla_olat, rt.d_mla_o, T, rt.stream, rt.cublas, rt.d_scratch, rt.scratch_bytes);
            launch_mla_head_sigmoid_gate(rt.d_mla_o, rt.d_mla_gate, T, cfg_.n_head, cfg_.v_head_dim_mla, rt.stream);
            launch_linear(rt.cublas, L.wo, rt.d_mla_o, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
        }

        launch_add_inplace(rt.d_cur, rt.d_sub_out, static_cast<int64_t>(T) * cfg_.n_embd, rt.stream);
        launch_rmsnorm(rt.d_cur, static_cast<const float *>(L.ffn_norm.ptr), rt.d_x_norm, cfg_.n_embd, T, cfg_.rms_eps, rt.stream);

        if (il < cfg_.n_dense_lead) {
            if (T < 16) launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T, rt.stream);
            linear_preq(L.ffn_gate, rt.d_x_norm, rt.d_ffn_g);
            linear_preq(L.ffn_up,   rt.d_x_norm, rt.d_ffn_u);
            launch_silu_mul(rt.d_ffn_g, rt.d_ffn_u, rt.d_ffn_g, static_cast<int64_t>(T) * cfg_.n_ff_dense, rt.stream);
            launch_linear(rt.cublas, L.ffn_down, rt.d_ffn_g, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
        } else {
            launch_linear(rt.cublas, L.ffn_gate_inp, rt.d_x_norm, rt.d_router, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            launch_moe_route_sigmoid(
                rt.d_router, static_cast<const float *>(L.ffn_exp_probs_b.ptr),
                rt.d_top_ids, rt.d_top_w,
                T, cfg_.n_expert, cfg_.n_expert_used, cfg_.expert_weights_norm, cfg_.expert_weights_scale, rt.stream);
            const float c_exp = (il < static_cast<int>(cfg_.swiglu_clamp_exp.size())) ? cfg_.swiglu_clamp_exp[il] : 0.0f;
            const float c_sh  = (il < static_cast<int>(cfg_.swiglu_clamp_shexp.size())) ? cfg_.swiglu_clamp_shexp[il] : 0.0f;
            launch_moe_ffn_kquant_with_shared(
                rt.cublas,
                L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps,
                L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
                rt.d_x_norm, rt.d_top_ids, rt.d_top_w, rt.d_sub_out,
                rt.d_scratch, rt.scratch_bytes, T, cfg_.n_expert_used, c_exp, c_sh, rt.stream);
        }

        launch_add_inplace(rt.d_cur, rt.d_sub_out, static_cast<int64_t>(T) * cfg_.n_embd, rt.stream);

        if (hook_ != nullptr) {
            hook_->on_post_layer(il, rt.device, rt.d_cur, T, cfg_.n_embd, 1, rt.stream);
        }
    }

    if (gi + 1 == runtimes_.size()) {
        const float * last_h = rt.d_cur + static_cast<size_t>(T - 1) * cfg_.n_embd;
        launch_rmsnorm(last_h, static_cast<const float *>(output_norm_.ptr), rt.d_x_norm, cfg_.n_embd, 1, cfg_.rms_eps, rt.stream);
        launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, rt.stream);
        if (run_argmax) {
            launch_argmax_f32(rt.d_logits, rt.d_argmax, cfg_.n_vocab, rt.stream);
        }
    }
}

void BailingMoe3Model::run_pipeline(int T, bool run_argmax, int kv_len_max) {
    const size_t x_bytes = static_cast<size_t>(T) * cfg_.n_embd * sizeof(float);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        if (T == 1 && graphs_ready_ && runtimes_[gi].decode_graph_t1 != nullptr) {
            cuda_ok(cudaSetDevice(runtimes_[gi].device), "bailing3 graph dev");
            cuda_ok(cudaGraphLaunch(runtimes_[gi].decode_graph_t1, runtimes_[gi].stream), "bailing3 graph launch");
        } else {
            run_gpu_segment(gi, T, run_argmax, kv_len_max);
        }
        if (gi + 1 < runtimes_.size()) {
            auto & src = runtimes_[gi];
            auto & dst = runtimes_[gi + 1];
            cuda_ok(cudaSetDevice(src.device), "bailing3 peer src dev");
            cuda_ok(cudaEventRecord(src.done_event, src.stream), "bailing3 peer rec");
            cuda_ok(cudaSetDevice(dst.device), "bailing3 peer dst dev");
            cuda_ok(cudaStreamWaitEvent(dst.stream, src.done_event, 0), "bailing3 peer wait");
            cuda_ok(cudaMemcpyPeerAsync(dst.d_cur, dst.device, src.d_cur, src.device, x_bytes, dst.stream), "bailing3 peer copy");
        }
    }
}

void BailingMoe3Model::capture_decode_graphs() {
    invalidate_graphs();
    if (!use_cuda_graphs_) return;
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "bailing3 cap dev");
        cuda_ok(cudaStreamSynchronize(rt.stream), "bailing3 cap pre-sync");
        cudaGraph_t g = nullptr;
        cuda_ok(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeRelaxed), "bailing3 cap begin");
        run_gpu_segment(gi, 1, true, 0);
        cuda_ok(cudaStreamEndCapture(rt.stream, &g), "bailing3 cap end");
        cuda_ok(cudaGraphInstantiate(&rt.decode_graph_t1, g, nullptr, nullptr, 0), "bailing3 cap inst");
        cudaGraphDestroy(g);
    }
    graphs_ready_ = true;
}

void BailingMoe3Model::warmup_decode_graphs() {
    if (!use_cuda_graphs_ || graphs_ready_) return;
    capture_decode_graphs();
}

void BailingMoe3Model::forward(const std::vector<int32_t> & tokens,
                               const std::vector<int32_t> & positions,
                               std::vector<float> & out_logits) {
    const int T = static_cast<int>(tokens.size());
    if (T <= 0 || T > max_batch_) throw std::runtime_error("BailingMoe3Model::forward invalid T");

    int kv_len_max = 0;
    for (int32_t p : positions) if (p + 1 > kv_len_max) kv_len_max = p + 1;

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "bailing3 fwd dev");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, tokens.data(), static_cast<size_t>(T) * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, rt.stream), "tok H2D");
        }
        cuda_ok(cudaMemcpyAsync(rt.d_pos, positions.data(), static_cast<size_t>(T) * sizeof(int32_t),
                                cudaMemcpyHostToDevice, rt.stream), "pos H2D");
    }

    run_pipeline(T, false, kv_len_max);

    auto & last = runtimes_.back();
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaSetDevice(last.device), "bailing3 logits dev");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), last.d_logits, static_cast<size_t>(cfg_.n_vocab) * sizeof(float),
                            cudaMemcpyDeviceToHost, last.stream), "logits D2H");
    cuda_ok(cudaStreamSynchronize(last.stream), "bailing3 fwd sync");
}

int32_t BailingMoe3Model::forward_decode_greedy(int32_t token, int32_t pos) {
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "bailing3 dec dev");
        if (gi == 0) cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok H2D");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "pos H2D");
    }
    run_pipeline(1, true, pos + 1);
    auto & last = runtimes_.back();
    int32_t next_tok = 0;
    cuda_ok(cudaSetDevice(last.device), "bailing3 argmax dev");
    cuda_ok(cudaMemcpyAsync(&next_tok, last.d_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, last.stream), "argmax D2H");
    cuda_ok(cudaStreamSynchronize(last.stream), "bailing3 dec sync");
    return next_tok;
}

void BailingMoe3Model::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "bailing3 decL dev");
        if (gi == 0) cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok H2D");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "pos H2D");
    }
    run_pipeline(1, true, pos + 1);
    auto & last = runtimes_.back();
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaSetDevice(last.device), "bailing3 decL out dev");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), last.d_logits, static_cast<size_t>(cfg_.n_vocab) * sizeof(float),
                            cudaMemcpyDeviceToHost, last.stream), "logits D2H");
    cuda_ok(cudaStreamSynchronize(last.stream), "bailing3 decL sync");
}

} // namespace rookery
