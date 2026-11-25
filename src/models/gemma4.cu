#include "models/gemma4.h"
#include "kernels/kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>

namespace rookery {

namespace {

inline void ck(cudaError_t err, const char * msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

inline void ck_blas(cublasStatus_t st, const char * msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " (cublas=" + std::to_string(static_cast<int>(st)) + ")");
    }
}

inline bool prof_enabled() {
    static const bool on = (std::getenv("ROOKERY_PROFILE") != nullptr && std::atoi(std::getenv("ROOKERY_PROFILE")) > 0);
    return on;
}

inline void gemma4_linear(
    cublasHandle_t cublas, const DeviceTensor & W,
    const float * d_x, float * d_y, int T,
    void * d_scratch, size_t scratch_bytes, cudaStream_t stream) {

    if (T >= 16 && try_launch_gemma4_linear_sm60_f16(cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream)) {
        return;
    }
    launch_linear(cublas, W, d_x, d_y, T, d_scratch, scratch_bytes, stream);
}

} // namespace

Gemma4Model::~Gemma4Model() {
    invalidate_graphs();
    for (size_t il = 0; il < k_cache_.size(); ++il) {
        if (il < layers_.size()) cudaSetDevice(layers_[il].device);
        if (k_cache_[il]) cudaFree(k_cache_[il]);
        if (v_cache_[il]) cudaFree(v_cache_[il]);
    }
    for (auto & m : kv_mirrors_) {
        cudaSetDevice(m.dst_dev);
        if (m.k_buf) cudaFree(m.k_buf);
        if (m.v_buf) cudaFree(m.v_buf);
    }
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        if (rt.cublas)     cublasDestroy(rt.cublas);
        if (rt.done_event) cudaEventDestroy(rt.done_event);
        if (rt.stream)     cudaStreamDestroy(rt.stream);
        cudaFree(rt.d_cur);
        cudaFree(rt.d_norm);
        cudaFree(rt.d_sub);
        cudaFree(rt.d_attn_res);
        cudaFree(rt.d_q);
        cudaFree(rt.d_k);
        cudaFree(rt.d_v);
        cudaFree(rt.d_attn_out);
        cudaFree(rt.d_ff_gate);
        cudaFree(rt.d_ff_up);
        cudaFree(rt.d_ple_raw);
        cudaFree(rt.d_ple_proj);
        cudaFree(rt.d_ple);
        cudaFree(rt.d_ple_gate);
        cudaFree(rt.d_tokens);
        cudaFree(rt.d_pos);
        cudaFree(rt.d_logits);
        cudaFree(rt.d_argmax);
        cudaFree(rt.d_scratch);
    }
}

void Gemma4Model::invalidate_graphs() noexcept {
    for (auto & rt : runtimes_) {
        if (rt.decode_graph_t1) {
            cudaSetDevice(rt.device);
            cudaGraphExecDestroy(rt.decode_graph_t1);
            rt.decode_graph_t1 = nullptr;
        }
    }
    graphs_ready_ = false;
}

void Gemma4Model::load(const std::string & gguf_path,
                       const std::vector<int> & gpu_ids,
                       const std::vector<float> & split_weights,
                       int max_ctx,
                       bool kv_q8_0) {
    if (gpu_ids.empty()) throw std::runtime_error("Gemma4Model::load: gpu_ids is empty");

    const auto t0 = std::chrono::steady_clock::now();
    auto sg = ShardedGguf::open(gguf_path);
    const std::string arch = sg.get_str("general.architecture");
    if (arch != "gemma4") {
        throw std::runtime_error("Gemma4Model::load: expected general.architecture='gemma4', got '" + arch + "'");
    }

    const GgufFile & meta = sg.meta();
    cfg_.n_layers              = static_cast<int>(sg.get_u64("gemma4.block_count", 35));
    cfg_.n_embd                = static_cast<int>(sg.get_u64("gemma4.embedding_length", 1536));
    cfg_.n_head                = static_cast<int>(sg.get_u64("gemma4.attention.head_count", 8));
    cfg_.n_head_kv             = static_cast<int>(sg.get_u64("gemma4.attention.head_count_kv", 1));
    cfg_.head_dim_full         = static_cast<int>(sg.get_u64("gemma4.attention.key_length", 512));
    cfg_.head_dim_swa          = static_cast<int>(sg.get_u64("gemma4.attention.key_length_swa", 256));
    cfg_.rope_dim_full         = static_cast<int>(sg.get_u64("gemma4.rope.dimension_count", cfg_.head_dim_full));
    cfg_.rope_dim_swa          = static_cast<int>(sg.get_u64("gemma4.rope.dimension_count_swa", cfg_.head_dim_swa));
    cfg_.rope_freq_base_full   = static_cast<float>(sg.get_f64("gemma4.rope.freq_base", 1000000.0));
    cfg_.rope_freq_base_swa    = static_cast<float>(sg.get_f64("gemma4.rope.freq_base_swa", 10000.0));
    cfg_.rms_eps               = static_cast<float>(sg.get_f64("gemma4.attention.layer_norm_rms_epsilon", 1e-6));
    cfg_.sliding_window        = static_cast<int>(sg.get_u64("gemma4.attention.sliding_window", 512));
    cfg_.shared_kv_layers      = static_cast<int>(sg.get_u64("gemma4.attention.shared_kv_layers", 0));
    cfg_.n_layer_kv_from_start = cfg_.n_layers - cfg_.shared_kv_layers;
    cfg_.n_embd_per_layer      = static_cast<int>(sg.get_u64("gemma4.embedding_length_per_layer_input", 0));
    cfg_.final_logit_softcap   = static_cast<float>(sg.get_f64("gemma4.final_logit_softcapping", 30.0));
    cfg_.max_ctx               = max_ctx;
    cfg_.kv_q8_0               = kv_q8_0;

    const auto tok_loc = sg.find_tensor("token_embd.weight");
    if (!tok_loc.info) throw std::runtime_error("Gemma4Model::load: missing token_embd.weight");
    cfg_.n_vocab = static_cast<int>(tok_loc.info->ne[1]);

    // Per-layer feed_forward_length (array or scalar)
    cfg_.n_ff.assign(cfg_.n_layers, 6144);
    if (meta.has_kv("gemma4.feed_forward_length")) {
        const auto & fv = meta.kv("gemma4.feed_forward_length");
        if (auto * vu = std::get_if<std::vector<uint64_t>>(&fv.data)) {
            for (int il = 0; il < cfg_.n_layers && il < static_cast<int>(vu->size()); ++il) {
                cfg_.n_ff[il] = static_cast<int>((*vu)[il]);
            }
        } else if (auto * vi = std::get_if<std::vector<int64_t>>(&fv.data)) {
            for (int il = 0; il < cfg_.n_layers && il < static_cast<int>(vi->size()); ++il) {
                cfg_.n_ff[il] = static_cast<int>((*vi)[il]);
            }
        } else {
            int scalar_ff = static_cast<int>(fv.as_u64());
            cfg_.n_ff.assign(cfg_.n_layers, scalar_ff);
        }
    }

    // Per-layer sliding_window_pattern
    cfg_.is_swa.assign(cfg_.n_layers, true);
    if (meta.has_kv("gemma4.attention.sliding_window_pattern")) {
        const auto & pv = meta.kv("gemma4.attention.sliding_window_pattern");
        if (auto * vb = std::get_if<std::vector<bool>>(&pv.data)) {
            for (int il = 0; il < cfg_.n_layers && il < static_cast<int>(vb->size()); ++il) {
                cfg_.is_swa[il] = (*vb)[il];
            }
        }
    }

    // KV sharing map: layers >= n_layer_kv_from_start reuse last own-KV layer of same SWA type
    cfg_.kv_source.resize(cfg_.n_layers);
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (il < cfg_.n_layer_kv_from_start) {
            cfg_.kv_source[il] = il;
        } else {
            cfg_.kv_source[il] = cfg_.n_layer_kv_from_start - (cfg_.is_swa[il] ? 2 : 1);
            if (cfg_.kv_source[il] < 0) cfg_.kv_source[il] = 0;
        }
    }

    // Partition layers across GPUs
    const size_t n_gpus = gpu_ids.size();
    std::vector<int> layer_counts(n_gpus, 0);
    if (split_weights.size() == n_gpus) {
        double sum = 0.0;
        for (float w : split_weights) sum += std::max(0.0f, w);
        int assigned = 0;
        for (size_t g = 0; g < n_gpus; ++g) {
            layer_counts[g] = (g + 1 == n_gpus)
                ? (cfg_.n_layers - assigned)
                : static_cast<int>(std::round(cfg_.n_layers * (std::max(0.0f, split_weights[g]) / std::max(sum, 1e-6))));
            assigned += layer_counts[g];
        }
    } else {
        for (size_t g = 0; g < n_gpus; ++g) {
            layer_counts[g] = cfg_.n_layers / static_cast<int>(n_gpus) + (static_cast<int>(g) < (cfg_.n_layers % static_cast<int>(n_gpus)) ? 1 : 0);
        }
    }

    runtimes_.resize(n_gpus);
    int cur_l = 0;
    for (size_t g = 0; g < n_gpus; ++g) {
        runtimes_[g].device      = gpu_ids[g];
        runtimes_[g].layer_start = cur_l;
        runtimes_[g].layer_end   = cur_l + layer_counts[g];
        cur_l += layer_counts[g];
    }

    if (n_gpus > 1) {
        for (size_t i = 0; i < n_gpus; ++i) {
            cudaSetDevice(gpu_ids[i]);
            for (size_t j = 0; j < n_gpus; ++j) {
                if (i == j) continue;
                int can = 0;
                cudaDeviceCanAccessPeer(&can, gpu_ids[i], gpu_ids[j]);
                if (can) {
                    cudaError_t e = cudaDeviceEnablePeerAccess(gpu_ids[j], 0);
                    if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                }
            }
        }
    }

    // Upload global tensors
    const int dev_first = gpu_ids.front();
    const int dev_last  = gpu_ids.back();
    const bool conv_bf16_first = !device_has_bf16(dev_first);

    tok_embd_ = pool_.upload(sg, "token_embd.weight", dev_first, conv_bf16_first, true);
    if (cfg_.n_embd_per_layer > 0) {
        per_layer_tok_embd_   = pool_.upload(sg, "per_layer_token_embd.weight", dev_first, conv_bf16_first, true);
        per_layer_model_proj_ = pool_.upload(sg, "per_layer_model_proj.weight", dev_first, conv_bf16_first, true);
        per_layer_proj_norm_  = pool_.upload(sg, "per_layer_proj_norm.weight",  dev_first, false, true);
    }

    output_norm_ = pool_.upload(sg, "output_norm.weight", dev_last, false, true);
    const bool has_separate_out = (sg.find_tensor("output.weight").info != nullptr);
    cfg_.tied_embeddings = !has_separate_out;
    if (has_separate_out) {
        output_ = pool_.upload(sg, "output.weight", dev_last, !device_has_bf16(dev_last), true);
    } else if (dev_last == dev_first) {
        output_ = tok_embd_; // zero-copy tied weight!
    } else {
        output_ = pool_.upload(sg, "token_embd.weight", dev_last, !device_has_bf16(dev_last), true);
    }

    // Upload per-layer weights
    layers_.resize(cfg_.n_layers);
    // Keep one copy of shared rope_freqs.weight per GPU that needs it
    std::vector<DeviceTensor> dev_rope_freqs(n_gpus);
    const bool has_global_rope_freqs = (sg.find_tensor("rope_freqs.weight").info != nullptr);

    for (size_t g = 0; g < n_gpus; ++g) {
        const int dev = runtimes_[g].device;
        const bool conv_bf16 = !device_has_bf16(dev);

        for (int il = runtimes_[g].layer_start; il < runtimes_[g].layer_end; ++il) {
            auto & L = layers_[il];
            L.device    = dev;
            L.is_swa    = cfg_.is_swa[il];
            L.has_kv    = (il < cfg_.n_layer_kv_from_start);
            L.kv_source = cfg_.kv_source[il];
            L.n_ff      = cfg_.n_ff[il];

            const std::string pfx = "blk." + std::to_string(il) + ".";
            L.attn_norm      = pool_.upload(sg, pfx + "attn_norm.weight",           dev, false, true);
            L.wq             = pool_.upload(sg, pfx + "attn_q.weight",              dev, conv_bf16, true);
            if (L.has_kv) {
                L.wk         = pool_.upload(sg, pfx + "attn_k.weight",              dev, conv_bf16, true);
                L.wv         = pool_.upload(sg, pfx + "attn_v.weight",              dev, conv_bf16, false);
                L.attn_k_norm = pool_.upload(sg, pfx + "attn_k_norm.weight",        dev, false, true);
            }
            L.wo             = pool_.upload(sg, pfx + "attn_output.weight",         dev, conv_bf16, true);
            L.attn_q_norm    = pool_.upload(sg, pfx + "attn_q_norm.weight",         dev, false, true);
            L.attn_post_norm = pool_.upload(sg, pfx + "post_attention_norm.weight", dev, false, true);

            if (!L.is_swa) {
                if (sg.find_tensor(pfx + "rope_freqs.weight").info != nullptr) {
                    L.rope_freqs = pool_.upload(sg, pfx + "rope_freqs.weight", dev, false, true);
                } else if (has_global_rope_freqs) {
                    if (!dev_rope_freqs[g].valid()) {
                        dev_rope_freqs[g] = pool_.upload(sg, "rope_freqs.weight", dev, false, true);
                    }
                    L.rope_freqs = dev_rope_freqs[g];
                }
            }

            L.ffn_norm      = pool_.upload(sg, pfx + "ffn_norm.weight",      dev, false, true);
            L.ffn_gate      = pool_.upload(sg, pfx + "ffn_gate.weight",      dev, conv_bf16, true);
            L.ffn_up        = pool_.upload(sg, pfx + "ffn_up.weight",        dev, conv_bf16, true);
            L.ffn_down      = pool_.upload(sg, pfx + "ffn_down.weight",      dev, conv_bf16, true);
            L.ffn_post_norm = pool_.upload(sg, pfx + "post_ffw_norm.weight", dev, false, true);

            if (cfg_.n_embd_per_layer > 0) {
                L.per_layer_inp_gate  = pool_.upload_as_f16(sg, pfx + "inp_gate.weight", dev, true);
                L.per_layer_proj      = pool_.upload_as_f16(sg, pfx + "proj.weight",     dev, true);
                L.per_layer_post_norm = pool_.upload(sg, pfx + "post_norm.weight", dev, false, true);
            }
            L.layer_output_scale = pool_.upload(sg, pfx + "layer_output_scale.weight", dev, false, false);
        }
    }
    pool_.release_staging();

    // Allocate KV caches only for layers [0 .. n_layer_kv_from_start)
    k_cache_.assign(cfg_.n_layers, nullptr);
    v_cache_.assign(cfg_.n_layers, nullptr);
    auto kv_layer_bytes = [&](int il) -> size_t {
        const int hd = cfg_.is_swa[il] ? cfg_.head_dim_swa : cfg_.head_dim_full;
        if (cfg_.kv_q8_0) {
            return static_cast<size_t>(cfg_.max_ctx) * cfg_.n_head_kv * (hd / 32) * sizeof(BlockQ8_0);
        }
        return static_cast<size_t>(cfg_.max_ctx) * cfg_.n_head_kv * hd * sizeof(__half);
    };

    for (int il = 0; il < cfg_.n_layer_kv_from_start; ++il) {
        cudaSetDevice(layers_[il].device);
        const size_t bytes = kv_layer_bytes(il);
        ck(cudaMalloc(&k_cache_[il], bytes), "gemma4 malloc k_cache");
        ck(cudaMalloc(&v_cache_[il], bytes), "gemma4 malloc v_cache");
        ck(cudaMemset(k_cache_[il], 0, bytes), "gemma4 zero k_cache");
        ck(cudaMemset(v_cache_[il], 0, bytes), "gemma4 zero v_cache");
    }

    // Create cross-GPU KV mirrors if any shared-KV layer sits on a different GPU than its kv_source
    for (int il = cfg_.n_layer_kv_from_start; il < cfg_.n_layers; ++il) {
        const int src = cfg_.kv_source[il];
        const int dst_dev = layers_[il].device;
        if (dst_dev == layers_[src].device) continue;
        bool exists = false;
        for (const auto & m : kv_mirrors_) {
            if (m.src_layer == src && m.dst_dev == dst_dev) { exists = true; break; }
        }
        if (!exists) {
            KvMirror m;
            m.src_layer = src;
            m.dst_dev   = dst_dev;
            m.bytes     = kv_layer_bytes(src);
            cudaSetDevice(dst_dev);
            ck(cudaMalloc(&m.k_buf, m.bytes), "gemma4 malloc k_mirror");
            ck(cudaMalloc(&m.v_buf, m.bytes), "gemma4 malloc v_mirror");
            kv_mirrors_.push_back(m);
        }
    }

    // Allocate per-GPU streams, cuBLAS handles, activations, and scratch arena
    const int max_hd = std::max(cfg_.head_dim_full, cfg_.head_dim_swa);
    int max_ff = 0;
    for (int ff : cfg_.n_ff) max_ff = std::max(max_ff, ff);
    const int H = cfg_.n_embd;

    for (size_t g = 0; g < n_gpus; ++g) {
        auto & rt = runtimes_[g];
        cudaSetDevice(rt.device);
        ck(cudaStreamCreate(&rt.stream), "gemma4 stream create");
        ck(cudaEventCreateWithFlags(&rt.done_event, cudaEventDisableTiming), "gemma4 event create");
        ck_blas(cublasCreate(&rt.cublas), "gemma4 cublasCreate");
        ck_blas(cublasSetStream(rt.cublas, rt.stream), "gemma4 cublasSetStream");

        const size_t Tmax = static_cast<size_t>(max_batch_);
        ck(cudaMalloc(&rt.d_cur,      Tmax * H * sizeof(float)), "g4 d_cur");
        ck(cudaMalloc(&rt.d_norm,     Tmax * H * sizeof(float)), "g4 d_norm");
        ck(cudaMalloc(&rt.d_sub,      Tmax * H * sizeof(float)), "g4 d_sub");
        ck(cudaMalloc(&rt.d_attn_res, Tmax * H * sizeof(float)), "g4 d_attn_res");
        ck(cudaMalloc(&rt.d_q,        Tmax * cfg_.n_head * max_hd * sizeof(float)), "g4 d_q");
        ck(cudaMalloc(&rt.d_k,        Tmax * cfg_.n_head_kv * max_hd * sizeof(float)), "g4 d_k");
        ck(cudaMalloc(&rt.d_v,        Tmax * cfg_.n_head_kv * max_hd * sizeof(float)), "g4 d_v");
        ck(cudaMalloc(&rt.d_attn_out, Tmax * cfg_.n_head * max_hd * sizeof(float)), "g4 d_attn_out");
        ck(cudaMalloc(&rt.d_ff_gate,  Tmax * max_ff * sizeof(float)), "g4 d_ff_gate");
        ck(cudaMalloc(&rt.d_ff_up,    Tmax * max_ff * sizeof(float)), "g4 d_ff_up");

        if (cfg_.n_embd_per_layer > 0) {
            const size_t ple_tot = Tmax * cfg_.n_layers * cfg_.n_embd_per_layer * sizeof(float);
            if (g == 0) {
                ck(cudaMalloc(&rt.d_ple_raw,  ple_tot), "g4 d_ple_raw");
                ck(cudaMalloc(&rt.d_ple_proj, ple_tot), "g4 d_ple_proj");
            }
            ck(cudaMalloc(&rt.d_ple,      ple_tot), "g4 d_ple");
            ck(cudaMalloc(&rt.d_ple_gate, Tmax * cfg_.n_embd_per_layer * sizeof(float)), "g4 d_ple_gate");
        }

        ck(cudaMalloc(&rt.d_tokens, Tmax * sizeof(int32_t)), "g4 d_tokens");
        ck(cudaMalloc(&rt.d_pos,    Tmax * sizeof(int32_t)), "g4 d_pos");

        if (g + 1 == n_gpus) {
            ck(cudaMalloc(&rt.d_logits, static_cast<size_t>(cfg_.n_vocab) * sizeof(float)), "g4 d_logits");
            ck(cudaMalloc(&rt.d_argmax, sizeof(int32_t)), "g4 d_argmax");
        }

        rt.scratch_bytes = static_cast<size_t>(96) << 20; // 96 MiB scratch
        ck(cudaMalloc(&rt.d_scratch, rt.scratch_bytes), "g4 d_scratch");
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double load_s = std::chrono::duration<double>(t1 - t0).count();
    std::printf("[Gemma4] Loaded %s (%d layers: %d own-KV + %d shared-KV, n_embd=%d, n_head=%d/%d, hd=%d/%d, swa=%d, ple=%d, softcap=%.1f, ctx=%d, kv=%s) in %.2f s\n",
                sg.get_str("general.name", "Gemma-4").c_str(),
                cfg_.n_layers, cfg_.n_layer_kv_from_start, cfg_.shared_kv_layers,
                cfg_.n_embd, cfg_.n_head, cfg_.n_head_kv,
                cfg_.head_dim_swa, cfg_.head_dim_full, cfg_.sliding_window,
                cfg_.n_embd_per_layer, cfg_.final_logit_softcap,
                cfg_.max_ctx, cfg_.kv_q8_0 ? "Q8_0" : "F16", load_s);
    for (size_t g = 0; g < n_gpus; ++g) {
        std::printf("  GPU%d: layers [%2d..%2d) (%2d layers), %.2f GiB weights\n",
                    runtimes_[g].device, runtimes_[g].layer_start, runtimes_[g].layer_end,
                    runtimes_[g].layer_end - runtimes_[g].layer_start,
                    static_cast<double>(pool_.bytes_on_device(runtimes_[g].device)) / (1024.0 * 1024.0 * 1024.0));
    }
    std::fflush(stdout);
}

void Gemma4Model::reset_state() {
    // KV cache is overwritten monotonically by pos, nothing recurrent to zero
}

void Gemma4Model::run_gpu_segment(size_t gi, int T, bool run_argmax, int pos_min, int pos_max,
                                  bool prune_prefill, bool compute_logits,
                                  int ple_tok_off_in, int & T_out, int & ple_tok_off_out) {
    auto & rt = runtimes_[gi];
    cudaSetDevice(rt.device);
    cudaStream_t st = rt.stream;

    const bool do_prof = prof_enabled() && !use_cuda_graphs_;
    cudaEvent_t pe[2];
    if (do_prof) {
        cudaEventCreate(&pe[0]);
        cudaEventCreate(&pe[1]);
    }
    auto p_begin = [&]() { if (do_prof) cudaEventRecord(pe[0], st); };
    auto p_end   = [&](float & acc) {
        if (do_prof) {
            cudaEventRecord(pe[1], st);
            cudaEventSynchronize(pe[1]);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, pe[0], pe[1]);
            acc += ms;
        }
    };
    float tm_emb = 0, tm_qkv = 0, tm_rope = 0, tm_attn = 0, tm_wo = 0, tm_ffn = 0, tm_ple = 0, tm_head = 0;

    const int H = cfg_.n_embd;
    const int sm = device_sm_version(rt.device);
    int T_cur = T;
    int ple_tok_off = ple_tok_off_in;

    // 1. Embeddings + PLE projection on GPU 0
    if (gi == 0) {
        p_begin();
        launch_embedding_lookup(tok_embd_, rt.d_tokens, rt.d_cur, H, 1, T_cur, st);
        launch_gemma4_scale(rt.d_cur, static_cast<int64_t>(T_cur) * H, std::sqrt(static_cast<float>(H)), st);

        if (cfg_.n_embd_per_layer > 0) {
            const int ple_total_dim = cfg_.n_layers * cfg_.n_embd_per_layer;
            launch_embedding_lookup(per_layer_tok_embd_, rt.d_tokens, rt.d_ple_raw, ple_total_dim, 1, T_cur, st);
            gemma4_linear(rt.cublas, per_layer_model_proj_, rt.d_cur, rt.d_ple_proj, T_cur, rt.d_scratch, rt.scratch_bytes, st);
            launch_gemma4_ple_combine(
                rt.d_ple_proj, rt.d_ple_raw,
                static_cast<const float *>(per_layer_proj_norm_.ptr),
                rt.d_ple, T_cur, cfg_.n_layers, H, cfg_.n_embd_per_layer, cfg_.rms_eps, st);
        }
        p_end(tm_emb);
    }

    // 2. Transformer layers owned by this GPU
    const int layer_limit = (!compute_logits && hook_ == nullptr)
        ? std::min(rt.layer_end, cfg_.n_layer_kv_from_start)
        : rt.layer_end;
    for (int il = rt.layer_start; il < layer_limit; ++il) {
        // Prefill pruning before the first KV-shared layer (layers >= n_layer_kv_from_start do not write KV!)
        if (prune_prefill && il == cfg_.n_layer_kv_from_start && T_cur > 1) {
            ck(cudaMemcpyAsync(rt.d_cur, rt.d_cur + static_cast<size_t>(T_cur - 1) * H,
                               static_cast<size_t>(H) * sizeof(float), cudaMemcpyDeviceToDevice, st), "g4 prune cur");
            ck(cudaMemcpyAsync(rt.d_pos, rt.d_pos + (T_cur - 1),
                               sizeof(int32_t), cudaMemcpyDeviceToDevice, st), "g4 prune pos");
            ple_tok_off = T_cur - 1;
            T_cur = 1;
        }

        const auto & L = layers_[il];
        const int hd       = L.is_swa ? cfg_.head_dim_swa : cfg_.head_dim_full;
        const int rdim     = L.is_swa ? cfg_.rope_dim_swa : cfg_.rope_dim_full;
        const float fbase  = L.is_swa ? cfg_.rope_freq_base_swa : cfg_.rope_freq_base_full;
        const int swa_win  = L.is_swa ? cfg_.sliding_window : 0;

        // 2.1 Pre-attention RMSNorm + Q/K/V projections
        p_begin();
        launch_rmsnorm(rt.d_cur, static_cast<const float *>(L.attn_norm.ptr), rt.d_norm, H, T_cur, cfg_.rms_eps, st);

        if (L.has_kv) {
            if (T_cur < 16 && L.wq.dtype == DType::Q4_K && L.wk.dtype == DType::Q4_K &&
                (!L.wv.valid() || L.wv.dtype == DType::Q4_K || (L.wv.dtype == DType::Q6_K && L.wv.is_soa))) {
                const DeviceTensor * ws[3] = {&L.wq, &L.wk, &L.wv};
                float * ys[3]              = {rt.d_q, rt.d_k, rt.d_v};
                const int n_mat = L.wv.valid() ? 3 : 2;
                bool done = false;
                if (sm == 60) {
                    done = try_launch_linear_multi_sm60_h2(ws, ys, n_mat, rt.d_norm, T_cur, rt.d_scratch, rt.scratch_bytes, st);
                }
                if (!done) {
                    launch_quantize_q8_1(rt.d_norm, rt.d_scratch, H, T_cur, st);
                    launch_linear_multi_prequant(ws, ys, n_mat, rt.d_scratch, T_cur, st);
                }
                if (!L.wv.valid()) {
                    ck(cudaMemcpyAsync(rt.d_v, rt.d_k, static_cast<size_t>(T_cur) * cfg_.n_head_kv * hd * sizeof(float),
                                       cudaMemcpyDeviceToDevice, st), "g4 copy k->v");
                }
            } else {
                gemma4_linear(rt.cublas, L.wq, rt.d_norm, rt.d_q, T_cur, rt.d_scratch, rt.scratch_bytes, st);
                gemma4_linear(rt.cublas, L.wk, rt.d_norm, rt.d_k, T_cur, rt.d_scratch, rt.scratch_bytes, st);
                if (L.wv.valid()) {
                    gemma4_linear(rt.cublas, L.wv, rt.d_norm, rt.d_v, T_cur, rt.d_scratch, rt.scratch_bytes, st);
                } else {
                    ck(cudaMemcpyAsync(rt.d_v, rt.d_k, static_cast<size_t>(T_cur) * cfg_.n_head_kv * hd * sizeof(float),
                                       cudaMemcpyDeviceToDevice, st), "g4 copy k->v");
                }
            }
        } else {
            gemma4_linear(rt.cublas, L.wq, rt.d_norm, rt.d_q, T_cur, rt.d_scratch, rt.scratch_bytes, st);
        }
        p_end(tm_qkv);

        // 2.2 QK-Norm + Proportional NeoX RoPE + Unweighted V-Norm + KV Write
        p_begin();
        const float * d_rfreqs = L.rope_freqs.valid() ? static_cast<const float *>(L.rope_freqs.ptr) : nullptr;
        launch_gemma4_qkv_norm_rope_kv_write(
            rt.d_q, rt.d_k, rt.d_v,
            static_cast<const float *>(L.attn_q_norm.ptr),
            L.has_kv ? static_cast<const float *>(L.attn_k_norm.ptr) : nullptr,
            d_rfreqs,
            L.has_kv ? k_cache_[il] : nullptr,
            L.has_kv ? v_cache_[il] : nullptr,
            cfg_.kv_q8_0, rt.d_pos, T_cur, cfg_.n_head, cfg_.n_head_kv,
            hd, rdim, cfg_.max_ctx, cfg_.rms_eps, fbase, L.has_kv, st);
        p_end(tm_rope);

        // 2.3 Resolve KV cache pointers (own layer or shared kv_source / cross-GPU mirror)
        const void * kc_ptr = k_cache_[L.kv_source];
        const void * vc_ptr = v_cache_[L.kv_source];
        if (!L.has_kv && layers_[L.kv_source].device != rt.device) {
            for (const auto & m : kv_mirrors_) {
                if (m.src_layer == L.kv_source && m.dst_dev == rt.device) {
                    kc_ptr = m.k_buf;
                    vc_ptr = m.v_buf;
                    break;
                }
            }
        }

        // 2.4 Attention (scale = 1.0, SWA or Full)
        p_begin();
        launch_gemma4_attn(
            rt.cublas, rt.d_q, kc_ptr, vc_ptr, cfg_.kv_q8_0,
            rt.d_pos, rt.d_attn_out, rt.d_scratch, rt.scratch_bytes,
            T_cur, cfg_.n_head, cfg_.n_head_kv, hd, cfg_.max_ctx, swa_win, pos_min, pos_max, st);
        p_end(tm_attn);

        // 2.5 Output projection + Post-Attention RMSNorm + Residual Add
        p_begin();
        gemma4_linear(rt.cublas, L.wo, rt.d_attn_out, rt.d_sub, T_cur, rt.d_scratch, rt.scratch_bytes, st);
        launch_gemma4_post_norm_add_scale(
            rt.d_cur, rt.d_sub, static_cast<const float *>(L.attn_post_norm.ptr),
            nullptr, rt.d_attn_res, H, T_cur, cfg_.rms_eps, st);
        p_end(tm_wo);

        // 2.6 Dense GeGLU FFN + Post-FFN RMSNorm + Residual Add
        p_begin();
        launch_rmsnorm(rt.d_attn_res, static_cast<const float *>(L.ffn_norm.ptr), rt.d_norm, H, T_cur, cfg_.rms_eps, st);
        if (T_cur < 16 && L.ffn_gate.dtype == DType::Q4_K && L.ffn_up.dtype == DType::Q4_K) {
            const DeviceTensor * ws[2] = {&L.ffn_gate, &L.ffn_up};
            float * ys[2]              = {rt.d_ff_gate, rt.d_ff_up};
            bool done = false;
            if (sm == 60) {
                done = try_launch_linear_multi_sm60_h2(ws, ys, 2, rt.d_norm, T_cur, rt.d_scratch, rt.scratch_bytes, st);
            }
            if (!done) {
                launch_quantize_q8_1(rt.d_norm, rt.d_scratch, H, T_cur, st);
                launch_linear_multi_prequant(ws, ys, 2, rt.d_scratch, T_cur, st);
            }
        } else {
            gemma4_linear(rt.cublas, L.ffn_gate, rt.d_norm, rt.d_ff_gate, T_cur, rt.d_scratch, rt.scratch_bytes, st);
            gemma4_linear(rt.cublas, L.ffn_up,   rt.d_norm, rt.d_ff_up,   T_cur, rt.d_scratch, rt.scratch_bytes, st);
        }
        launch_gemma4_geglu(rt.d_ff_gate, rt.d_ff_up, rt.d_ff_gate, static_cast<int64_t>(T_cur) * L.n_ff, st);
        gemma4_linear(rt.cublas, L.ffn_down, rt.d_ff_gate, rt.d_sub, T_cur, rt.d_scratch, rt.scratch_bytes, st);

        const float * d_lscale = L.layer_output_scale.valid() ? static_cast<const float *>(L.layer_output_scale.ptr) : nullptr;
        launch_gemma4_post_norm_add_scale(
            rt.d_attn_res, rt.d_sub, static_cast<const float *>(L.ffn_post_norm.ptr),
            (cfg_.n_embd_per_layer > 0) ? nullptr : d_lscale,
            rt.d_cur, H, T_cur, cfg_.rms_eps, st);
        p_end(tm_ffn);

        // 2.7 Per-Layer Embedding (PLE) gating block + Post-PLE RMSNorm + Residual Add + Layer Output Scale
        if (cfg_.n_embd_per_layer > 0) {
            p_begin();
            if (T_cur == 1 && L.per_layer_inp_gate.dtype == DType::F16 && L.per_layer_proj.dtype == DType::F16 &&
                (H & 7) == 0 && (cfg_.n_embd_per_layer & 7) == 0 && cfg_.n_embd_per_layer <= 1024) {
                launch_gemma4_ple_f16_t1(
                    L.per_layer_inp_gate.ptr, L.per_layer_proj.ptr,
                    rt.d_cur, rt.d_ple, rt.d_ple_gate, rt.d_sub,
                    H, cfg_.n_embd_per_layer, cfg_.n_layers, il, ple_tok_off, st);
            } else {
                gemma4_linear(rt.cublas, L.per_layer_inp_gate, rt.d_cur, rt.d_ple_gate, T_cur, rt.d_scratch, rt.scratch_bytes, st);
                launch_gemma4_ple_gate_gelu_mul(rt.d_ple_gate, rt.d_ple, T_cur, cfg_.n_layers, il, cfg_.n_embd_per_layer, ple_tok_off, st);
                gemma4_linear(rt.cublas, L.per_layer_proj, rt.d_ple_gate, rt.d_sub, T_cur, rt.d_scratch, rt.scratch_bytes, st);
            }
            launch_gemma4_post_norm_add_scale(
                rt.d_cur, rt.d_sub, static_cast<const float *>(L.per_layer_post_norm.ptr),
                d_lscale, rt.d_cur, H, T_cur, cfg_.rms_eps, st);
            p_end(tm_ple);
        }

        // 2.8 Optional LayerHook (Engram sidecar / diagnostics)
        if (hook_ && hook_->hooks_layer(il)) {
            hook_->on_post_layer(il, rt.device, rt.d_cur, T_cur, H, 1, st);
        }
    }

    // 3. Final Output Norm + LM Head + Logit Softcap on last GPU
    if (compute_logits && gi + 1 == runtimes_.size()) {
        p_begin();
        const float * last_h = rt.d_cur + static_cast<size_t>(T_cur - 1) * H;
        launch_rmsnorm(last_h, static_cast<const float *>(output_norm_.ptr), rt.d_norm, H, 1, cfg_.rms_eps, st);
        gemma4_linear(rt.cublas, output_, rt.d_norm, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, st);
        if (cfg_.final_logit_softcap > 0.0f) {
            launch_gemma4_logit_softcap(rt.d_logits, cfg_.n_vocab, cfg_.final_logit_softcap, st);
        }
        if (run_argmax) {
            launch_argmax_f32(rt.d_logits, rt.d_argmax, cfg_.n_vocab, st);
        }
        p_end(tm_head);
    }

    if (do_prof) {
        const float tot = tm_emb + tm_qkv + tm_rope + tm_attn + tm_wo + tm_ffn + tm_ple + tm_head;
        std::printf("  [PROF GPU%d T=%d L%d..%d] emb=%.2f qkv=%.2f rope=%.2f attn=%.2f wo=%.2f ffn=%.2f ple=%.2f head=%.2f | tot=%.2f ms\n",
                    rt.device, T, rt.layer_start, rt.layer_end,
                    tm_emb, tm_qkv, tm_rope, tm_attn, tm_wo, tm_ffn, tm_ple, tm_head, tot);
        std::fflush(stdout);
        cudaEventDestroy(pe[0]);
        cudaEventDestroy(pe[1]);
    }

    T_out = T_cur;
    ple_tok_off_out = ple_tok_off;
}

void Gemma4Model::capture_decode_graphs() {
    invalidate_graphs();
    if (!use_cuda_graphs_ || hook_ != nullptr || !kv_mirrors_.empty()) return;

    // Seed pos=0 / tok=0 so capture has valid device state
    int32_t zero = 0;
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cudaSetDevice(rt.device);
        ck(cudaMemcpyAsync(rt.d_tokens, &zero, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "g4 cap tok");
        ck(cudaMemcpyAsync(rt.d_pos,    &zero, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "g4 cap pos");
        ck(cudaStreamSynchronize(rt.stream), "g4 cap sync");

        cudaGraph_t graph = nullptr;
        ck(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeRelaxed), "g4 begin capture");
        int t_out = 1, p_out = 0;
        run_gpu_segment(gi, 1, /*run_argmax=*/true, 0, 0, /*prune_prefill=*/false, /*compute_logits=*/true, 0, t_out, p_out);
        ck(cudaStreamEndCapture(rt.stream, &graph), "g4 end capture");
        ck(cudaGraphInstantiate(&rt.decode_graph_t1, graph, nullptr, nullptr, 0), "g4 instantiate graph");
        cudaGraphDestroy(graph);
    }
    graphs_ready_ = true;
}

void Gemma4Model::warmup_decode_graphs() {
    if (use_cuda_graphs_ && !graphs_ready_) {
        capture_decode_graphs();
    }
}

void Gemma4Model::forward(const std::vector<int32_t> & tokens,
                          const std::vector<int32_t> & positions,
                          std::vector<float> & out_logits,
                          bool compute_logits) {
    const int T = static_cast<int>(tokens.size());
    if (T <= 0 || T != static_cast<int>(positions.size())) {
        throw std::runtime_error("Gemma4Model::forward: invalid tokens/positions size");
    }
    if (T > max_batch_) {
        throw std::runtime_error("Gemma4Model::forward: T > max_batch_ (" + std::to_string(max_batch_) + ")");
    }

    int pos_min = positions[0], pos_max = positions[0];
    for (int p : positions) { pos_min = std::min(pos_min, p); pos_max = std::max(pos_max, p); }
    if (pos_max >= cfg_.max_ctx) {
        throw std::runtime_error("Gemma4Model::forward: position exceeds max_ctx");
    }

    // Upload tokens to GPU 0, positions to all GPUs
    {
        auto & rt0 = runtimes_[0];
        cudaSetDevice(rt0.device);
        ck(cudaMemcpyAsync(rt0.d_tokens, tokens.data(), static_cast<size_t>(T) * sizeof(int32_t), cudaMemcpyHostToDevice, rt0.stream), "g4 h2d tok");
    }
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        ck(cudaMemcpyAsync(rt.d_pos, positions.data(), static_cast<size_t>(T) * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "g4 h2d pos");
    }

    const bool can_prune = (T > 1 && hook_ == nullptr);
    int T_cur = T;
    int ple_tok_off = 0;
    size_t last_active_gi = 0;

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        if (!compute_logits && hook_ == nullptr && rt.layer_start >= cfg_.n_layer_kv_from_start) {
            break;
        }
        last_active_gi = gi;
        int T_next = T_cur, ple_next = ple_tok_off;
        run_gpu_segment(gi, T_cur, /*run_argmax=*/false, pos_min, pos_max, can_prune, compute_logits, ple_tok_off, T_next, ple_next);

        // Sync cross-GPU KV mirrors sourced from layers in [rt.layer_start, rt.layer_end)
        for (const auto & m : kv_mirrors_) {
            if (m.src_layer >= rt.layer_start && m.src_layer < rt.layer_end) {
                const int hd = cfg_.is_swa[m.src_layer] ? cfg_.head_dim_swa : cfg_.head_dim_full;
                if (cfg_.kv_q8_0) {
                    const size_t qs_total = static_cast<size_t>(cfg_.max_ctx) * cfg_.n_head_kv * hd;
                    const size_t qs_row_b = static_cast<size_t>(cfg_.n_head_kv) * hd;
                    const size_t d_row_b  = static_cast<size_t>(cfg_.n_head_kv) * (hd / 32) * sizeof(uint16_t);
                    const size_t q_off = static_cast<size_t>(pos_min) * qs_row_b;
                    const size_t q_len = static_cast<size_t>(pos_max - pos_min + 1) * qs_row_b;
                    const size_t d_off = qs_total + static_cast<size_t>(pos_min) * d_row_b;
                    const size_t d_len = static_cast<size_t>(pos_max - pos_min + 1) * d_row_b;
                    ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.k_buf) + q_off, m.dst_dev,
                                           static_cast<const uint8_t *>(k_cache_[m.src_layer]) + q_off, rt.device, q_len, rt.stream), "g4 peer k_qs");
                    ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.k_buf) + d_off, m.dst_dev,
                                           static_cast<const uint8_t *>(k_cache_[m.src_layer]) + d_off, rt.device, d_len, rt.stream), "g4 peer k_d");
                    ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.v_buf) + q_off, m.dst_dev,
                                           static_cast<const uint8_t *>(v_cache_[m.src_layer]) + q_off, rt.device, q_len, rt.stream), "g4 peer v_qs");
                    ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.v_buf) + d_off, m.dst_dev,
                                           static_cast<const uint8_t *>(v_cache_[m.src_layer]) + d_off, rt.device, d_len, rt.stream), "g4 peer v_d");
                } else {
                    const size_t row_b = static_cast<size_t>(cfg_.n_head_kv) * hd * sizeof(__half);
                    const size_t off_b = static_cast<size_t>(pos_min) * row_b;
                    const size_t len_b = static_cast<size_t>(pos_max - pos_min + 1) * row_b;
                    ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.k_buf) + off_b, m.dst_dev,
                                           static_cast<const uint8_t *>(k_cache_[m.src_layer]) + off_b, rt.device, len_b, rt.stream), "g4 peer k_mirror");
                    ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.v_buf) + off_b, m.dst_dev,
                                           static_cast<const uint8_t *>(v_cache_[m.src_layer]) + off_b, rt.device, len_b, rt.stream), "g4 peer v_mirror");
                }
            }
        }

        if (gi + 1 < runtimes_.size() && (compute_logits || hook_ != nullptr || runtimes_[gi + 1].layer_start < cfg_.n_layer_kv_from_start)) {
            auto & nxt = runtimes_[gi + 1];
            ck(cudaMemcpyPeerAsync(nxt.d_cur, nxt.device, rt.d_cur, rt.device,
                                   static_cast<size_t>(T_next) * cfg_.n_embd * sizeof(float), rt.stream), "g4 peer d_cur");
            if (T_next != T_cur) {
                ck(cudaMemcpyPeerAsync(nxt.d_pos, nxt.device, rt.d_pos, rt.device,
                                       static_cast<size_t>(T_next) * sizeof(int32_t), rt.stream), "g4 peer d_pos");
            }
            if (cfg_.n_embd_per_layer > 0) {
                const size_t ple_bytes = static_cast<size_t>(T) * cfg_.n_layers * cfg_.n_embd_per_layer * sizeof(float);
                ck(cudaMemcpyPeerAsync(nxt.d_ple, nxt.device, rt.d_ple, rt.device, ple_bytes, rt.stream), "g4 peer d_ple");
            }
            ck(cudaEventRecord(rt.done_event, rt.stream), "g4 record done");
            cudaSetDevice(nxt.device);
            ck(cudaStreamWaitEvent(nxt.stream, rt.done_event, 0), "g4 wait done");
        }
        T_cur = T_next;
        ple_tok_off = ple_next;
    }

    auto & end_rt = runtimes_[last_active_gi];
    cudaSetDevice(end_rt.device);
    if (compute_logits) {
        out_logits.resize(cfg_.n_vocab);
        ck(cudaMemcpyAsync(out_logits.data(), end_rt.d_logits,
                           static_cast<size_t>(cfg_.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost, end_rt.stream), "g4 logits D2H");
    }
    ck(cudaStreamSynchronize(end_rt.stream), "g4 sync");
}

int32_t Gemma4Model::forward_decode_greedy(int32_t token, int32_t pos) {
    if (pos >= cfg_.max_ctx) throw std::runtime_error("Gemma4Model::forward_decode_greedy: pos >= max_ctx");

    if (use_cuda_graphs_ && !graphs_ready_ && hook_ == nullptr && kv_mirrors_.empty()) {
        capture_decode_graphs();
    }

    {
        auto & rt0 = runtimes_[0];
        cudaSetDevice(rt0.device);
        ck(cudaMemcpyAsync(rt0.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt0.stream), "g4 dec tok");
    }
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        ck(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "g4 dec pos");
    }

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cudaSetDevice(rt.device);
        if (graphs_ready_ && rt.decode_graph_t1) {
            ck(cudaGraphLaunch(rt.decode_graph_t1, rt.stream), "g4 graph launch");
        } else {
            int t_out = 1, p_out = 0;
            run_gpu_segment(gi, 1, /*run_argmax=*/true, pos, pos, /*prune_prefill=*/false, /*compute_logits=*/true, 0, t_out, p_out);
            for (const auto & m : kv_mirrors_) {
                if (m.src_layer >= rt.layer_start && m.src_layer < rt.layer_end) {
                    const int hd = cfg_.is_swa[m.src_layer] ? cfg_.head_dim_swa : cfg_.head_dim_full;
                    if (cfg_.kv_q8_0) {
                        const size_t qs_total = static_cast<size_t>(cfg_.max_ctx) * cfg_.n_head_kv * hd;
                        const size_t qs_row_b = static_cast<size_t>(cfg_.n_head_kv) * hd;
                        const size_t d_row_b  = static_cast<size_t>(cfg_.n_head_kv) * (hd / 32) * sizeof(uint16_t);
                        const size_t q_off = static_cast<size_t>(pos) * qs_row_b;
                        const size_t d_off = qs_total + static_cast<size_t>(pos) * d_row_b;
                        ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.k_buf) + q_off, m.dst_dev,
                                               static_cast<const uint8_t *>(k_cache_[m.src_layer]) + q_off, rt.device, qs_row_b, rt.stream), "g4 dec k_qs");
                        ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.k_buf) + d_off, m.dst_dev,
                                               static_cast<const uint8_t *>(k_cache_[m.src_layer]) + d_off, rt.device, d_row_b, rt.stream), "g4 dec k_d");
                        ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.v_buf) + q_off, m.dst_dev,
                                               static_cast<const uint8_t *>(v_cache_[m.src_layer]) + q_off, rt.device, qs_row_b, rt.stream), "g4 dec v_qs");
                        ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.v_buf) + d_off, m.dst_dev,
                                               static_cast<const uint8_t *>(v_cache_[m.src_layer]) + d_off, rt.device, d_row_b, rt.stream), "g4 dec v_d");
                    } else {
                        const size_t row_b = static_cast<size_t>(cfg_.n_head_kv) * hd * sizeof(__half);
                        const size_t off_b = static_cast<size_t>(pos) * row_b;
                        ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.k_buf) + off_b, m.dst_dev,
                                               static_cast<const uint8_t *>(k_cache_[m.src_layer]) + off_b, rt.device, row_b, rt.stream), "g4 dec peer k_mirror");
                        ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.v_buf) + off_b, m.dst_dev,
                                               static_cast<const uint8_t *>(v_cache_[m.src_layer]) + off_b, rt.device, row_b, rt.stream), "g4 dec peer v_mirror");
                    }
                }
            }
        }
        if (gi + 1 < runtimes_.size()) {
            auto & nxt = runtimes_[gi + 1];
            ck(cudaMemcpyPeerAsync(nxt.d_cur, nxt.device, rt.d_cur, rt.device,
                                   static_cast<size_t>(cfg_.n_embd) * sizeof(float), rt.stream), "g4 dec peer cur");
            if (cfg_.n_embd_per_layer > 0) {
                const size_t ple_bytes = static_cast<size_t>(cfg_.n_layers) * cfg_.n_embd_per_layer * sizeof(float);
                ck(cudaMemcpyPeerAsync(nxt.d_ple, nxt.device, rt.d_ple, rt.device, ple_bytes, rt.stream), "g4 dec peer ple");
            }
            ck(cudaEventRecord(rt.done_event, rt.stream), "g4 dec record");
            cudaSetDevice(nxt.device);
            ck(cudaStreamWaitEvent(nxt.stream, rt.done_event, 0), "g4 dec wait");
        }
    }

    auto & last_rt = runtimes_.back();
    cudaSetDevice(last_rt.device);
    int32_t next_tok = 0;
    ck(cudaMemcpyAsync(&next_tok, last_rt.d_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, last_rt.stream), "g4 argmax D2H");
    ck(cudaStreamSynchronize(last_rt.stream), "g4 dec sync");
    return next_tok;
}

void Gemma4Model::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    if (pos >= cfg_.max_ctx) throw std::runtime_error("Gemma4Model::forward_decode_logits: pos >= max_ctx");

    if (use_cuda_graphs_ && !graphs_ready_ && hook_ == nullptr && kv_mirrors_.empty()) {
        capture_decode_graphs();
    }

    {
        auto & rt0 = runtimes_[0];
        cudaSetDevice(rt0.device);
        ck(cudaMemcpyAsync(rt0.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt0.stream), "g4 dec tok");
    }
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        ck(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "g4 dec pos");
    }

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cudaSetDevice(rt.device);
        if (graphs_ready_ && rt.decode_graph_t1) {
            ck(cudaGraphLaunch(rt.decode_graph_t1, rt.stream), "g4 graph launch");
        } else {
            int t_out = 1, p_out = 0;
            run_gpu_segment(gi, 1, /*run_argmax=*/false, pos, pos, /*prune_prefill=*/false, /*compute_logits=*/true, 0, t_out, p_out);
            for (const auto & m : kv_mirrors_) {
                if (m.src_layer >= rt.layer_start && m.src_layer < rt.layer_end) {
                    const int hd = cfg_.is_swa[m.src_layer] ? cfg_.head_dim_swa : cfg_.head_dim_full;
                    if (cfg_.kv_q8_0) {
                        const size_t qs_total = static_cast<size_t>(cfg_.max_ctx) * cfg_.n_head_kv * hd;
                        const size_t qs_row_b = static_cast<size_t>(cfg_.n_head_kv) * hd;
                        const size_t d_row_b  = static_cast<size_t>(cfg_.n_head_kv) * (hd / 32) * sizeof(uint16_t);
                        const size_t q_off = static_cast<size_t>(pos) * qs_row_b;
                        const size_t d_off = qs_total + static_cast<size_t>(pos) * d_row_b;
                        ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.k_buf) + q_off, m.dst_dev,
                                               static_cast<const uint8_t *>(k_cache_[m.src_layer]) + q_off, rt.device, qs_row_b, rt.stream), "g4 dec k_qs");
                        ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.k_buf) + d_off, m.dst_dev,
                                               static_cast<const uint8_t *>(k_cache_[m.src_layer]) + d_off, rt.device, d_row_b, rt.stream), "g4 dec k_d");
                        ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.v_buf) + q_off, m.dst_dev,
                                               static_cast<const uint8_t *>(v_cache_[m.src_layer]) + q_off, rt.device, qs_row_b, rt.stream), "g4 dec v_qs");
                        ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.v_buf) + d_off, m.dst_dev,
                                               static_cast<const uint8_t *>(v_cache_[m.src_layer]) + d_off, rt.device, d_row_b, rt.stream), "g4 dec v_d");
                    } else {
                        const size_t row_b = static_cast<size_t>(cfg_.n_head_kv) * hd * sizeof(__half);
                        const size_t off_b = static_cast<size_t>(pos) * row_b;
                        ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.k_buf) + off_b, m.dst_dev,
                                               static_cast<const uint8_t *>(k_cache_[m.src_layer]) + off_b, rt.device, row_b, rt.stream), "g4 dec peer k_mirror");
                        ck(cudaMemcpyPeerAsync(static_cast<uint8_t *>(m.v_buf) + off_b, m.dst_dev,
                                               static_cast<const uint8_t *>(v_cache_[m.src_layer]) + off_b, rt.device, row_b, rt.stream), "g4 dec peer v_mirror");
                    }
                }
            }
        }
        if (gi + 1 < runtimes_.size()) {
            auto & nxt = runtimes_[gi + 1];
            ck(cudaMemcpyPeerAsync(nxt.d_cur, nxt.device, rt.d_cur, rt.device,
                                   static_cast<size_t>(cfg_.n_embd) * sizeof(float), rt.stream), "g4 dec peer cur");
            if (cfg_.n_embd_per_layer > 0) {
                const size_t ple_bytes = static_cast<size_t>(cfg_.n_layers) * cfg_.n_embd_per_layer * sizeof(float);
                ck(cudaMemcpyPeerAsync(nxt.d_ple, nxt.device, rt.d_ple, rt.device, ple_bytes, rt.stream), "g4 dec peer ple");
            }
            ck(cudaEventRecord(rt.done_event, rt.stream), "g4 dec record");
            cudaSetDevice(nxt.device);
            ck(cudaStreamWaitEvent(nxt.stream, rt.done_event, 0), "g4 dec wait");
        }
    }

    auto & last_rt = runtimes_.back();
    cudaSetDevice(last_rt.device);
    out_logits.resize(cfg_.n_vocab);
    ck(cudaMemcpyAsync(out_logits.data(), last_rt.d_logits,
                       static_cast<size_t>(cfg_.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost, last_rt.stream), "g4 dec logits D2H");
    ck(cudaStreamSynchronize(last_rt.stream), "g4 dec sync");
}

} // namespace rookery
