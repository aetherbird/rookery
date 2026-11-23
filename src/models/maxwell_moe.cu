#include "models/maxwell_moe.h"
#include "kernels/kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>

namespace rookery {

namespace {

inline void cuda_ok(cudaError_t err, const char * msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

template <typename T>
T * dev_alloc(int dev, size_t count) {
    cuda_ok(cudaSetDevice(dev), "dev_alloc cudaSetDevice");
    T * ptr = nullptr;
    cuda_ok(cudaMalloc(&ptr, count * sizeof(T)), "dev_alloc cudaMalloc");
    return ptr;
}

} // namespace

void MaxwellMoeModel::invalidate_graphs() noexcept {
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        if (rt.decode_graph_t1) {
            cudaGraphExecDestroy(rt.decode_graph_t1);
            rt.decode_graph_t1 = nullptr;
        }
        if (rt.decode_graph_t2) {
            cudaGraphExecDestroy(rt.decode_graph_t2);
            rt.decode_graph_t2 = nullptr;
        }
    }
    graphs_t1_ready_ = false;
    graphs_t2_ready_ = false;
}

MaxwellMoeModel::~MaxwellMoeModel() {
    invalidate_graphs();
    for (size_t il = 0; il < k_caches_.size(); ++il) {
        if (il < layers_.size()) {
            cudaSetDevice(layers_[il].device);
            if (k_caches_[il]) cudaFree(k_caches_[il]);
            if (v_caches_[il]) cudaFree(v_caches_[il]);
            if (il < ik_caches_.size() && ik_caches_[il]) cudaFree(ik_caches_[il]);
        }
    }
    if (mtp_.loaded && !runtimes_.empty()) {
        cudaSetDevice(runtimes_.back().device);
        if (k_cache_mtp_) cudaFree(k_cache_mtp_);
        if (v_cache_mtp_) cudaFree(v_cache_mtp_);
    }
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        auto freep = [](void * p) { if (p) cudaFree(p); };
        freep(rt.d_inpL);
        freep(rt.d_inpL_alt);
        freep(rt.d_flat_norm);
        freep(rt.d_mixes);
        freep(rt.d_post);
        freep(rt.d_comb);
        freep(rt.d_x_norm);
        freep(rt.d_q);
        freep(rt.d_k);
        freep(rt.d_v);
        freep(rt.d_attn_out);
        freep(rt.d_sub_out);
        freep(rt.d_ffn_g);
        freep(rt.d_ffn_u);
        freep(rt.d_router);
        freep(rt.d_top_ids);
        freep(rt.d_top_w);
        freep(rt.d_tokens);
        freep(rt.d_pos);
        freep(rt.d_logits);
        freep(rt.d_argmax);
        freep(rt.d_scratch);
        freep(rt.d_iq);
        freep(rt.d_ik);
        freep(rt.d_msa_blocks);
        freep(rt.d_msa_kcnt);
        freep(rt.d_mtp_e);
        freep(rt.d_mtp_eh_in);
        freep(rt.d_mtp_inpL);
        freep(rt.d_mtp_alt);
        freep(rt.d_mtp_tok);
        freep(rt.d_mtp_pos);
        if (rt.done_event) cudaEventDestroy(rt.done_event);
        if (rt.cublas) cublasDestroy(rt.cublas);
        if (rt.stream) cudaStreamDestroy(rt.stream);
    }
}

void MaxwellMoeModel::load(const std::string & gguf_path,
                           const std::vector<int> & gpu_ids,
                           const std::vector<float> & split_weights,
                           int max_ctx,
                           bool kv_q8_0,
                           bool load_mtp) {
    if (gpu_ids.empty()) {
        throw std::runtime_error("MaxwellMoeModel::load requires at least 1 GPU");
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto gguf = GgufFile::open(gguf_path);

    if (gguf.get_str("general.architecture") != "maxwell-moe") {
        throw std::runtime_error("Expected maxwell-moe GGUF, got: " + gguf.get_str("general.architecture"));
    }

    cfg_.n_blocks             = static_cast<int>(gguf.get_u64("maxwell-moe.block_count", 46));
    cfg_.n_nextn              = static_cast<int>(gguf.get_u64("maxwell-moe.nextn_predict_layers", 1));
    cfg_.n_layers             = cfg_.n_blocks - cfg_.n_nextn;
    cfg_.n_dense_lead         = static_cast<int>(gguf.get_u64("maxwell-moe.leading_dense_block_count", 3));
    cfg_.n_embd               = static_cast<int>(gguf.get_u64("maxwell-moe.embedding_length", 2048));
    cfg_.n_ff_dense           = static_cast<int>(gguf.get_u64("maxwell-moe.feed_forward_length", 6144));
    cfg_.n_vocab              = static_cast<int>(gguf.get_u64("maxwell-moe.vocab_size", 200704));
    cfg_.n_head               = static_cast<int>(gguf.get_u64("maxwell-moe.attention.head_count", 32));
    cfg_.n_head_kv            = static_cast<int>(gguf.get_u64("maxwell-moe.attention.head_count_kv", 4));
    cfg_.head_dim             = static_cast<int>(gguf.get_u64("maxwell-moe.attention.key_length", 128));
    cfg_.n_expert             = static_cast<int>(gguf.get_u64("maxwell-moe.expert_count", 256));
    cfg_.n_expert_used        = static_cast<int>(gguf.get_u64("maxwell-moe.expert_used_count", 12));
    cfg_.n_expert_shared      = static_cast<int>(gguf.get_u64("maxwell-moe.expert_shared_count", 1));
    cfg_.n_ff_exp             = static_cast<int>(gguf.get_u64("maxwell-moe.expert_feed_forward_length", 512));
    cfg_.n_ff_shexp           = static_cast<int>(gguf.get_u64("maxwell-moe.expert_shared_feed_forward_length", 1024));
    cfg_.expert_weights_norm  = gguf.get_bool("maxwell-moe.expert_weights_norm", true);
    cfg_.expert_weights_scale = static_cast<float>(gguf.get_f64("maxwell-moe.expert_weights_scale", 1.0));
    cfg_.hc_mult              = static_cast<int>(gguf.get_u64("maxwell-moe.hyper_connection.count", 4));
    cfg_.hc_sinkhorn_iters    = static_cast<int>(gguf.get_u64("maxwell-moe.hyper_connection.sinkhorn_iterations", 20));
    cfg_.hc_eps               = static_cast<float>(gguf.get_f64("maxwell-moe.hyper_connection.epsilon", 1e-6));
    cfg_.rms_eps              = static_cast<float>(gguf.get_f64("maxwell-moe.attention.layer_norm_rms_epsilon", 1e-6));
    cfg_.rope_freq_base       = static_cast<float>(gguf.get_f64("maxwell-moe.rope.freq_base", 20000000.0));
    cfg_.max_ctx              = max_ctx;
    cfg_.kv_q8_0              = kv_q8_0;
    cfg_.idx_n_head           = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.head_count", 4));
    cfg_.idx_head_dim         = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.key_length", 128));
    cfg_.idx_top_k            = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.top_k", 16));
    cfg_.idx_blk_size         = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.block_size", 128));
    cfg_.idx_local_blks       = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.local_blocks", 8));
    cfg_.idx_prefix_blks      = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.prefix_blocks", 8));
    cfg_.idx_dense_prefix     = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.dense_prefix_tokens", 16384));
    msa_enabled_              = (cfg_.max_ctx > cfg_.idx_dense_prefix);

    std::vector<float> sw = split_weights;
    if (sw.size() != gpu_ids.size()) sw.assign(gpu_ids.size(), 1.0f);
    const float sw_sum = std::accumulate(sw.begin(), sw.end(), 0.0f);

    runtimes_.resize(gpu_ids.size());
    std::vector<int> layer_dev(cfg_.n_layers);
    int cur_l = 0;
    for (size_t g = 0; g < gpu_ids.size(); ++g) {
        int count = (g + 1 == gpu_ids.size())
                        ? (cfg_.n_layers - cur_l)
                        : static_cast<int>(std::round(cfg_.n_layers * (sw[g] / sw_sum)));
        count = std::max(0, std::min(count, cfg_.n_layers - cur_l));
        runtimes_[g].device      = gpu_ids[g];
        runtimes_[g].layer_start = cur_l;
        runtimes_[g].layer_end   = cur_l + count;
        for (int k = 0; k < count; ++k) layer_dev[cur_l++] = gpu_ids[g];
    }

    constexpr bool conv_bf16 = false;
    const int first_dev = gpu_ids.front();
    const int last_dev  = gpu_ids.back();

    tok_embd_      = pool_.upload(gguf, "token_embd.weight",     first_dev, conv_bf16);
    hc_head_fn_    = pool_.upload(gguf, "output_hc_fn.weight",   last_dev,  conv_bf16);
    hc_head_base_  = pool_.upload(gguf, "output_hc_base.weight", last_dev,  conv_bf16);
    hc_head_scale_ = pool_.upload(gguf, "output_hc_scale.weight",last_dev,  conv_bf16);
    output_norm_   = pool_.upload(gguf, "output_norm.weight",    last_dev,  conv_bf16);
    output_        = pool_.upload(gguf, "output.weight",         last_dev,  conv_bf16, false);
    if (!output_.valid()) {
        output_ = pool_.upload(gguf, "token_embd.weight", last_dev, conv_bf16);
    }

    auto load_layer_weights = [&](int il, int dev, bool is_dense, MaxwellLayerWeights & L) {
        const std::string p = "blk." + std::to_string(il) + ".";
        L.device        = dev;
        L.attn_norm     = pool_.upload(gguf, p + "attn_norm.weight",   dev, conv_bf16);
        L.ffn_norm      = pool_.upload(gguf, p + "ffn_norm.weight",    dev, conv_bf16);
        L.wq            = pool_.upload(gguf, p + "attn_q.weight",      dev, conv_bf16);
        L.wk            = pool_.upload(gguf, p + "attn_k.weight",      dev, conv_bf16);
        L.wv            = pool_.upload(gguf, p + "attn_v.weight",      dev, conv_bf16);
        L.wo            = pool_.upload(gguf, p + "attn_output.weight", dev, conv_bf16);
        L.attn_q_norm   = pool_.upload(gguf, p + "attn_q_norm.weight", dev, conv_bf16);
        L.attn_k_norm   = pool_.upload(gguf, p + "attn_k_norm.weight", dev, conv_bf16);
        if (msa_enabled_ && il < cfg_.n_layers) {
            L.idx_q_proj = pool_.upload(gguf, p + "indexer.q_proj.weight", dev, conv_bf16, false);
            L.idx_k_proj = pool_.upload(gguf, p + "indexer.k_proj.weight", dev, conv_bf16, false);
            L.idx_q_norm = pool_.upload(gguf, p + "indexer.q_norm.weight", dev, conv_bf16, false);
            L.idx_k_norm = pool_.upload(gguf, p + "indexer.k_norm.weight", dev, conv_bf16, false);
        }

        L.hc_attn_fn    = pool_.upload(gguf, p + "hc_attn_fn.weight",    dev, conv_bf16);
        L.hc_attn_base  = pool_.upload(gguf, p + "hc_attn_base.weight",  dev, conv_bf16);
        L.hc_attn_scale = pool_.upload(gguf, p + "hc_attn_scale.weight", dev, conv_bf16);
        L.hc_ffn_fn     = pool_.upload(gguf, p + "hc_ffn_fn.weight",     dev, conv_bf16);
        L.hc_ffn_base   = pool_.upload(gguf, p + "hc_ffn_base.weight",   dev, conv_bf16);
        L.hc_ffn_scale  = pool_.upload(gguf, p + "hc_ffn_scale.weight",  dev, conv_bf16);

        if (is_dense) {
            L.ffn_gate = pool_.upload(gguf, p + "ffn_gate.weight", dev, conv_bf16);
            L.ffn_up   = pool_.upload(gguf, p + "ffn_up.weight",   dev, conv_bf16);
            L.ffn_down = pool_.upload(gguf, p + "ffn_down.weight", dev, conv_bf16);
        } else {
            L.ffn_gate_inp    = pool_.upload(gguf, p + "ffn_gate_inp.weight",    dev, conv_bf16);
            L.ffn_exp_probs_b = pool_.upload(gguf, p + "exp_probs_b.bias",       dev, conv_bf16);
            L.ffn_gate_exps   = pool_.upload(gguf, p + "ffn_gate_exps.weight",   dev, conv_bf16);
            L.ffn_up_exps     = pool_.upload(gguf, p + "ffn_up_exps.weight",     dev, conv_bf16);
            L.ffn_down_exps   = pool_.upload(gguf, p + "ffn_down_exps.weight",   dev, conv_bf16);
            L.ffn_gate_shexp  = pool_.upload(gguf, p + "ffn_gate_shexp.weight",  dev, conv_bf16);
            L.ffn_up_shexp    = pool_.upload(gguf, p + "ffn_up_shexp.weight",    dev, conv_bf16);
            L.ffn_down_shexp  = pool_.upload(gguf, p + "ffn_down_shexp.weight",  dev, conv_bf16);
        }
    };

    layers_.resize(cfg_.n_layers);
    k_caches_.resize(cfg_.n_layers, nullptr);
    v_caches_.resize(cfg_.n_layers, nullptr);
    if (msa_enabled_) ik_caches_.resize(cfg_.n_layers, nullptr);

    const size_t kv_bytes_per_layer = kv_q8_0
        ? static_cast<size_t>(cfg_.max_ctx) * cfg_.n_head_kv * (cfg_.head_dim / 32) * sizeof(BlockQ8_0)
        : static_cast<size_t>(cfg_.max_ctx) * cfg_.n_head_kv * cfg_.head_dim * sizeof(uint16_t);
    const size_t ik_bytes_per_layer = static_cast<size_t>(cfg_.max_ctx) * cfg_.idx_head_dim * sizeof(uint16_t);

    for (int il = 0; il < cfg_.n_layers; ++il) {
        const int dev = layer_dev[il];
        load_layer_weights(il, dev, il < cfg_.n_dense_lead, layers_[il]);

        cuda_ok(cudaSetDevice(dev), "kv alloc setDevice");
        cuda_ok(cudaMalloc(&k_caches_[il], kv_bytes_per_layer), "cudaMalloc k_cache");
        cuda_ok(cudaMalloc(&v_caches_[il], kv_bytes_per_layer), "cudaMalloc v_cache");
        if (msa_enabled_ && layers_[il].idx_k_proj.valid()) {
            cuda_ok(cudaMalloc(&ik_caches_[il], ik_bytes_per_layer), "cudaMalloc ik_cache");
        }
    }

    // Load optional MTP NextN layer (blk.45) onto last_dev!
    if (load_mtp && cfg_.n_nextn > 0) {
        const int il = cfg_.n_layers;
        const std::string p = "blk." + std::to_string(il) + ".";
        if (gguf.find_tensor(p + "nextn.eh_proj.weight") != nullptr) {
            load_layer_weights(il, last_dev, false, mtp_.layer);
            mtp_.eh_proj          = pool_.upload(gguf, p + "nextn.eh_proj.weight",       last_dev, conv_bf16);
            mtp_.enorm            = pool_.upload(gguf, p + "nextn.enorm.weight",         last_dev, conv_bf16);
            mtp_.hnorm            = pool_.upload(gguf, p + "nextn.hnorm.weight",         last_dev, conv_bf16);
            mtp_.shared_head_norm = pool_.upload(gguf, p + "layer_output_norm.weight",   last_dev, conv_bf16);
            mtp_.hc_head_fn       = pool_.upload(gguf, p + "nextn.hc_head_fn.weight",    last_dev, conv_bf16);
            mtp_.hc_head_base     = pool_.upload(gguf, p + "nextn.hc_head_base.weight",  last_dev, conv_bf16);
            mtp_.hc_head_scale    = pool_.upload(gguf, p + "nextn.hc_head_scale.weight", last_dev, conv_bf16);
            if (last_dev == first_dev) {
                mtp_.tok_embd = tok_embd_;
            } else {
                mtp_.tok_embd = pool_.upload(gguf, "token_embd.weight", last_dev, conv_bf16);
            }
            cuda_ok(cudaSetDevice(last_dev), "mtp kv setDevice");
            cuda_ok(cudaMalloc(&k_cache_mtp_, kv_bytes_per_layer), "cudaMalloc k_cache_mtp");
            cuda_ok(cudaMalloc(&v_cache_mtp_, kv_bytes_per_layer), "cudaMalloc v_cache_mtp");
            cuda_ok(cudaMemset(k_cache_mtp_, 0, kv_bytes_per_layer), "memset k_cache_mtp");
            cuda_ok(cudaMemset(v_cache_mtp_, 0, kv_bytes_per_layer), "memset v_cache_mtp");
            mtp_.loaded = true;
        }
    }

    pool_.release_staging();

    const int Tmax   = max_batch_;
    const int q_dim  = cfg_.n_head * cfg_.head_dim;
    const int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
    const int max_ff = std::max(cfg_.n_ff_dense, cfg_.n_ff_shexp);

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        const int dev = rt.device;
        cuda_ok(cudaSetDevice(dev), "rt setDevice");
        cuda_ok(cudaStreamCreate(&rt.stream), "cudaStreamCreate");
        cuda_ok(cudaEventCreateWithFlags(&rt.done_event, cudaEventDisableTiming), "cudaEventCreate");
        if (cublasCreate(&rt.cublas) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }

        rt.d_inpL      = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
        rt.d_inpL_alt  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
        rt.d_flat_norm = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
        rt.d_mixes     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * 24);
        rt.d_post      = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult);
        rt.d_comb      = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.hc_mult);
        rt.d_x_norm    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
        rt.d_q         = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * q_dim);
        rt.d_k         = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * kv_dim);
        rt.d_v         = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * kv_dim);
        rt.d_attn_out  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * q_dim);
        rt.d_sub_out   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
        rt.d_ffn_g     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * max_ff);
        rt.d_ffn_u     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * max_ff);
        rt.d_router    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert);
        rt.d_top_ids   = dev_alloc<int32_t>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert_used);
        rt.d_top_w     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert_used);
        rt.d_tokens    = dev_alloc<int32_t>(dev, Tmax);
        rt.d_pos       = dev_alloc<int32_t>(dev, Tmax);
        if (dev == last_dev) {
            rt.d_logits = dev_alloc<float>(dev, static_cast<size_t>(2) * cfg_.n_vocab);
            rt.d_argmax = dev_alloc<int32_t>(dev, 2);
            if (mtp_.loaded) {
                rt.d_mtp_e     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
                rt.d_mtp_eh_in = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * 2 * cfg_.n_embd);
                rt.d_mtp_inpL  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
                rt.d_mtp_alt   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
                rt.d_mtp_tok   = dev_alloc<int32_t>(dev, Tmax);
                rt.d_mtp_pos   = dev_alloc<int32_t>(dev, Tmax);
            }
        }
        rt.scratch_bytes = 64ULL << 20;
        rt.d_scratch = dev_alloc<uint8_t>(dev, rt.scratch_bytes);
        if (msa_enabled_) {
            const int k_max = cfg_.idx_prefix_blks + cfg_.idx_local_blks + cfg_.idx_top_k;
            rt.d_iq         = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.idx_n_head * cfg_.idx_head_dim);
            rt.d_ik         = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.idx_head_dim);
            rt.d_msa_blocks = dev_alloc<int32_t>(dev, static_cast<size_t>(Tmax) * cfg_.idx_n_head * k_max);
            rt.d_msa_kcnt   = dev_alloc<int32_t>(dev, Tmax);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    std::printf("Loaded Maxwell-V1-Flash (%d backbone%s, ctx=%d, kv=%s, SoA=LDG.128) in %.2f s\n",
                cfg_.n_layers, mtp_.loaded ? " + blk.45 MTP" : "", cfg_.max_ctx, kv_q8_0 ? "Q8_0" : "F16", sec);
    for (const auto & rt : runtimes_) {
        std::printf("  GPU%d: layers [%d..%d) (%2d layers), %.2f GiB weights\n",
                    rt.device, rt.layer_start, rt.layer_end, rt.layer_end - rt.layer_start,
                    static_cast<double>(pool_.bytes_on_device(rt.device)) / (1024.0 * 1024.0 * 1024.0));
    }
}

void MaxwellMoeModel::run_gpu_segment(size_t gi, int T, bool run_argmax, bool logits_all_t) {
    auto & rt = runtimes_[gi];
    cuda_ok(cudaSetDevice(rt.device), "segment setDevice");

    const int   hc_dim   = cfg_.hc_mult * cfg_.n_embd;
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));

    if (gi == 0) {
        launch_embedding_lookup(tok_embd_, rt.d_tokens, rt.d_inpL, cfg_.n_embd, cfg_.hc_mult, T, rt.stream);
    }

    for (int il = rt.layer_start; il < rt.layer_end; ++il) {
        const auto & L = layers_[il];

        // --- Attention sub-block: rt.d_inpL -> rt.d_inpL_alt ---
        launch_hc_flat_rmsnorm_mix(rt.d_inpL, L.hc_attn_fn, rt.d_mixes, hc_dim, 24, T, cfg_.rms_eps, rt.stream);
        launch_hc_sinkhorn_pre_rmsnorm(
            rt.d_inpL, rt.d_mixes,
            static_cast<const float *>(L.hc_attn_scale.ptr),
            static_cast<const float *>(L.hc_attn_base.ptr),
            static_cast<const float *>(L.attn_norm.ptr),
            rt.d_post, rt.d_comb, rt.d_x_norm,
            cfg_.n_embd, T, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps, rt.stream);

        if (T < 16 && L.wq.dtype == DType::Q8_0 && L.wk.dtype == DType::Q8_0 && L.wv.dtype == DType::Q8_0) {
            launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T, rt.stream);
            launch_linear_q8_0_prequant(L.wq, rt.d_scratch, rt.d_q, T, rt.stream);
            launch_linear_q8_0_prequant(L.wk, rt.d_scratch, rt.d_k, T, rt.stream);
            launch_linear_q8_0_prequant(L.wv, rt.d_scratch, rt.d_v, T, rt.stream);
        } else {
            launch_linear(rt.cublas, L.wq, rt.d_x_norm, rt.d_q, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            launch_linear(rt.cublas, L.wk, rt.d_x_norm, rt.d_k, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            launch_linear(rt.cublas, L.wv, rt.d_x_norm, rt.d_v, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
        }

        launch_qk_norm_rope_neox(rt.d_q, static_cast<const float *>(L.attn_q_norm.ptr),
                                 rt.d_pos, T, cfg_.n_head, cfg_.head_dim, cfg_.rms_eps, cfg_.rope_freq_base, rt.stream);
        launch_qk_norm_rope_neox(rt.d_k, static_cast<const float *>(L.attn_k_norm.ptr),
                                 rt.d_pos, T, cfg_.n_head_kv, cfg_.head_dim, cfg_.rms_eps, cfg_.rope_freq_base, rt.stream);

        if (cfg_.kv_q8_0) {
            launch_kv_write_q8_0(rt.d_k, rt.d_v, k_caches_[il], v_caches_[il], rt.d_pos,
                                 T, cfg_.n_head_kv, cfg_.head_dim, rt.stream);
        } else {
            launch_kv_write_f16(rt.d_k, rt.d_v, k_caches_[il], v_caches_[il], rt.d_pos,
                                T, cfg_.n_head_kv, cfg_.head_dim, rt.stream);
        }

        if (msa_enabled_ && L.idx_k_proj.valid() && ik_caches_[il] != nullptr) {
            // Always update indexer key cache (dense or sparse) so later sparse steps have complete keys
            launch_linear(rt.cublas, L.idx_k_proj, rt.d_x_norm, rt.d_ik, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            launch_qk_norm_rope_neox(rt.d_ik, static_cast<const float *>(L.idx_k_norm.ptr),
                                     rt.d_pos, T, 1, cfg_.idx_head_dim, cfg_.rms_eps, cfg_.rope_freq_base, rt.stream);
            launch_idx_k_write_f16(rt.d_ik, ik_caches_[il], rt.d_pos, T, cfg_.idx_head_dim, rt.stream);

            launch_linear(rt.cublas, L.idx_q_proj, rt.d_x_norm, rt.d_iq, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            launch_qk_norm_rope_neox(rt.d_iq, static_cast<const float *>(L.idx_q_norm.ptr),
                                     rt.d_pos, T, cfg_.idx_n_head, cfg_.idx_head_dim, cfg_.rms_eps, cfg_.rope_freq_base, rt.stream);

            const int k_max = cfg_.idx_prefix_blks + cfg_.idx_local_blks + cfg_.idx_top_k;
            launch_msa_indexer_topk(
                rt.d_iq, ik_caches_[il], rt.d_pos, rt.d_msa_blocks, rt.d_msa_kcnt, rt.d_scratch,
                T, cfg_.max_ctx, cfg_.idx_n_head, cfg_.idx_blk_size, cfg_.idx_prefix_blks, cfg_.idx_local_blks,
                k_max, cfg_.idx_dense_prefix, rt.stream);

            launch_gqa_attn_msa(
                rt.d_q, k_caches_[il], v_caches_[il], cfg_.kv_q8_0,
                rt.d_pos, rt.d_msa_blocks, rt.d_msa_kcnt,
                rt.d_attn_out, rt.d_scratch, T, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim,
                cfg_.idx_blk_size, k_max, kq_scale, rt.stream);
        } else {
            launch_gqa_attn(rt.d_q, k_caches_[il], v_caches_[il], cfg_.kv_q8_0,
                            rt.d_pos, rt.d_attn_out, rt.d_scratch, T, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim, kq_scale, rt.stream);
        }

        launch_linear(rt.cublas, L.wo, rt.d_attn_out, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
        launch_hc_post(rt.d_sub_out, rt.d_inpL, rt.d_post, rt.d_comb, rt.d_inpL_alt, cfg_.n_embd, T, rt.stream);

        // --- FFN sub-block: rt.d_inpL_alt -> rt.d_inpL ---
        launch_hc_flat_rmsnorm_mix(rt.d_inpL_alt, L.hc_ffn_fn, rt.d_mixes, hc_dim, 24, T, cfg_.rms_eps, rt.stream);
        launch_hc_sinkhorn_pre_rmsnorm(
            rt.d_inpL_alt, rt.d_mixes,
            static_cast<const float *>(L.hc_ffn_scale.ptr),
            static_cast<const float *>(L.hc_ffn_base.ptr),
            static_cast<const float *>(L.ffn_norm.ptr),
            rt.d_post, rt.d_comb, rt.d_x_norm,
            cfg_.n_embd, T, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps, rt.stream);

        if (il < cfg_.n_dense_lead) {
            if (T < 16 && L.ffn_gate.dtype == DType::Q8_0 && L.ffn_up.dtype == DType::Q8_0) {
                launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T, rt.stream);
                launch_linear_q8_0_prequant(L.ffn_gate, rt.d_scratch, rt.d_ffn_g, T, rt.stream);
                launch_linear_q8_0_prequant(L.ffn_up,   rt.d_scratch, rt.d_ffn_u, T, rt.stream);
            } else {
                launch_linear(rt.cublas, L.ffn_gate, rt.d_x_norm, rt.d_ffn_g, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                launch_linear(rt.cublas, L.ffn_up,   rt.d_x_norm, rt.d_ffn_u, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            }
            launch_silu_mul(rt.d_ffn_g, rt.d_ffn_u, rt.d_ffn_g, static_cast<int64_t>(T) * cfg_.n_ff_dense, rt.stream);
            launch_linear(rt.cublas, L.ffn_down, rt.d_ffn_g, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
        } else {
            launch_linear(rt.cublas, L.ffn_gate_inp, rt.d_x_norm, rt.d_router, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            launch_moe_route_sigmoid(
                rt.d_router, static_cast<const float *>(L.ffn_exp_probs_b.ptr),
                rt.d_top_ids, rt.d_top_w,
                T, cfg_.n_expert, cfg_.n_expert_used, cfg_.expert_weights_norm, cfg_.expert_weights_scale, rt.stream);
            launch_moe_ffn_q8_0_with_shared(
                L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps,
                L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
                rt.d_x_norm, rt.d_top_ids, rt.d_top_w, rt.d_sub_out,
                rt.d_scratch, T, cfg_.n_expert_used, rt.stream);
        }

        launch_hc_post(rt.d_sub_out, rt.d_inpL_alt, rt.d_post, rt.d_comb, rt.d_inpL, cfg_.n_embd, T, rt.stream);

        if (hook_ != nullptr) {
            hook_->on_post_layer(il, rt.device, rt.d_inpL, T, cfg_.n_embd, cfg_.hc_mult, rt.stream);
        }
    }

    // Final head on last GPU
    if (gi + 1 == runtimes_.size()) {
        const int   head_T    = logits_all_t ? T : 1;
        const float * head_in = logits_all_t ? rt.d_inpL : (rt.d_inpL + static_cast<size_t>(T - 1) * hc_dim);

        launch_hc_flat_rmsnorm_mix(head_in, hc_head_fn_, rt.d_mixes, hc_dim, 4, head_T, cfg_.rms_eps, rt.stream);
        launch_hc_head_rmsnorm(
            head_in, rt.d_mixes,
            static_cast<const float *>(hc_head_scale_.ptr),
            static_cast<const float *>(hc_head_base_.ptr),
            static_cast<const float *>(output_norm_.ptr),
            rt.d_x_norm, cfg_.n_embd, head_T, cfg_.hc_eps, cfg_.rms_eps, rt.stream);

        launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, head_T, rt.d_scratch, rt.scratch_bytes, rt.stream);
        if (run_argmax) {
            launch_argmax_f32_batch(rt.d_logits, rt.d_argmax, cfg_.n_vocab, head_T, rt.stream);
        }
    }
}

void MaxwellMoeModel::capture_decode_graphs(int T) {
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "capture setDevice");
        cuda_ok(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeGlobal), "begin capture");
        run_gpu_segment(gi, T, true, T == 2);
        cudaGraph_t graph = nullptr;
        cuda_ok(cudaStreamEndCapture(rt.stream, &graph), "end capture");
        cudaGraphExec_t exec = nullptr;
        cuda_ok(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate graph");
        cudaGraphDestroy(graph);
        if (T == 1) rt.decode_graph_t1 = exec;
        else        rt.decode_graph_t2 = exec;
    }
    if (T == 1) graphs_t1_ready_ = true;
    else        graphs_t2_ready_ = true;
}

void MaxwellMoeModel::forward(const std::vector<int32_t> & tokens,
                              const std::vector<int32_t> & positions,
                              std::vector<float> & out_logits) {
    const int T = static_cast<int>(tokens.size());
    if (T <= 0 || T > max_batch_ || positions.size() != tokens.size()) {
        throw std::runtime_error("MaxwellMoeModel::forward invalid batch size");
    }

    const int hc_dim = cfg_.hc_mult * cfg_.n_embd;

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "forward setDevice");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, positions.data(), T * sizeof(int32_t),
                                cudaMemcpyHostToDevice, rt.stream), "copy pos");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, tokens.data(), T * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, rt.stream), "copy tokens");
        }
    }

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        if (gi > 0) {
            auto & src_rt = runtimes_[gi - 1];
            const size_t act_bytes = static_cast<size_t>(T) * hc_dim * sizeof(float);
            cuda_ok(cudaSetDevice(src_rt.device), "record src setDevice");
            cuda_ok(cudaEventRecord(src_rt.done_event, src_rt.stream), "eventRecord");
            cuda_ok(cudaSetDevice(rt.device), "wait dst setDevice");
            cuda_ok(cudaStreamWaitEvent(rt.stream, src_rt.done_event, 0), "streamWaitEvent");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_inpL, rt.device, src_rt.d_inpL, src_rt.device,
                                        act_bytes, rt.stream), "peer transfer");
        }
        run_gpu_segment(gi, T, false, false);
    }

    auto & last_rt = runtimes_.back();
    cuda_ok(cudaSetDevice(last_rt.device), "logits setDevice");
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaMemcpyAsync(out_logits.data(), last_rt.d_logits, cfg_.n_vocab * sizeof(float),
                            cudaMemcpyDeviceToHost, last_rt.stream), "copy logits");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "final sync");
}

int32_t MaxwellMoeModel::forward_decode_greedy(int32_t token, int32_t pos) {
    const int hc_dim = cfg_.hc_mult * cfg_.n_embd;

    if (use_cuda_graphs_ && !graphs_t1_ready_) {
        capture_decode_graphs(1);
    }

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "decode setDevice");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "pos");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok");
        }
    }

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        if (gi > 0) {
            auto & src_rt = runtimes_[gi - 1];
            cuda_ok(cudaSetDevice(src_rt.device), "record src");
            cuda_ok(cudaEventRecord(src_rt.done_event, src_rt.stream), "eventRecord");
            cuda_ok(cudaSetDevice(rt.device), "wait dst");
            cuda_ok(cudaStreamWaitEvent(rt.stream, src_rt.done_event, 0), "streamWaitEvent");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_inpL, rt.device, src_rt.d_inpL, src_rt.device,
                                        static_cast<size_t>(hc_dim) * sizeof(float), rt.stream), "peer");
        }
        cuda_ok(cudaSetDevice(rt.device), "launch setDevice");
        if (use_cuda_graphs_ && rt.decode_graph_t1) {
            cuda_ok(cudaGraphLaunch(rt.decode_graph_t1, rt.stream), "cudaGraphLaunch t1");
        } else {
            run_gpu_segment(gi, 1, true, false);
        }
    }

    auto & last_rt = runtimes_.back();
    int32_t next_tok = 0;
    cuda_ok(cudaSetDevice(last_rt.device), "argmax D2H setDevice");
    cuda_ok(cudaMemcpyAsync(&next_tok, last_rt.d_argmax, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, last_rt.stream), "copy argmax");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "decode sync");
    return next_tok;
}

void MaxwellMoeModel::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    const int hc_dim = cfg_.hc_mult * cfg_.n_embd;

    if (use_cuda_graphs_ && !graphs_t1_ready_) {
        capture_decode_graphs(1);
    }

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "decode_l setDevice");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "pos");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok");
        }
    }

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        if (gi > 0) {
            auto & src_rt = runtimes_[gi - 1];
            cuda_ok(cudaSetDevice(src_rt.device), "record src");
            cuda_ok(cudaEventRecord(src_rt.done_event, src_rt.stream), "eventRecord");
            cuda_ok(cudaSetDevice(rt.device), "wait dst");
            cuda_ok(cudaStreamWaitEvent(rt.stream, src_rt.done_event, 0), "streamWaitEvent");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_inpL, rt.device, src_rt.d_inpL, src_rt.device,
                                        static_cast<size_t>(hc_dim) * sizeof(float), rt.stream), "peer");
        }
        cuda_ok(cudaSetDevice(rt.device), "launch setDevice");
        if (use_cuda_graphs_ && rt.decode_graph_t1) {
            cuda_ok(cudaGraphLaunch(rt.decode_graph_t1, rt.stream), "cudaGraphLaunch t1");
        } else {
            run_gpu_segment(gi, 1, false, false);
        }
    }

    auto & last_rt = runtimes_.back();
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaSetDevice(last_rt.device), "logits D2H setDevice");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), last_rt.d_logits, cfg_.n_vocab * sizeof(float),
                            cudaMemcpyDeviceToHost, last_rt.stream), "copy logits");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "decode_l sync");
}

void MaxwellMoeModel::warmup_decode_graphs() {
    if (!use_cuda_graphs_) return;
    if (!graphs_t1_ready_) capture_decode_graphs(1);
    if (mtp_.loaded && !graphs_t2_ready_) capture_decode_graphs(2);
}

// Run blk.45 MTP layer on last_dev for `T_mtp` (token, h_backbone) pairs and return greedy draft token for the last pair.
int32_t MaxwellMoeModel::run_mtp_on_last_gpu(const int32_t * h_tokens,
                                             const float * d_h_states,
                                             const int32_t * h_positions,
                                             int T_mtp) {
    auto & rt = runtimes_.back();
    cuda_ok(cudaSetDevice(rt.device), "mtp setDevice");

    const int   hc_dim   = cfg_.hc_mult * cfg_.n_embd;
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
    const auto & L       = mtp_.layer;

    cuda_ok(cudaMemcpyAsync(rt.d_mtp_tok, h_tokens,    T_mtp * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "mtp tok");
    cuda_ok(cudaMemcpyAsync(rt.d_mtp_pos, h_positions, T_mtp * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "mtp pos");

    // 1. e_norm + h_norm concat -> eh_proj -> d_mtp_inpL[T_mtp, 4, n_embd]
    launch_embedding_lookup(mtp_.tok_embd, rt.d_mtp_tok, rt.d_mtp_e, cfg_.n_embd, 1, T_mtp, rt.stream);
    launch_mtp_eh_concat(
        rt.d_mtp_e, d_h_states,
        static_cast<const float *>(mtp_.enorm.ptr),
        static_cast<const float *>(mtp_.hnorm.ptr),
        rt.d_mtp_eh_in, cfg_.n_embd, cfg_.hc_mult, T_mtp, cfg_.rms_eps, rt.stream);
    launch_linear(rt.cublas, mtp_.eh_proj, rt.d_mtp_eh_in, rt.d_mtp_inpL,
                  T_mtp * cfg_.hc_mult, rt.d_scratch, rt.scratch_bytes, rt.stream);

    // 2. Attention sub-block (dense GQA, blk.45 KV cache): d_mtp_inpL -> d_mtp_alt
    launch_hc_flat_rmsnorm_mix(rt.d_mtp_inpL, L.hc_attn_fn, rt.d_mixes, hc_dim, 24, T_mtp, cfg_.rms_eps, rt.stream);
    launch_hc_sinkhorn_pre_rmsnorm(
        rt.d_mtp_inpL, rt.d_mixes,
        static_cast<const float *>(L.hc_attn_scale.ptr),
        static_cast<const float *>(L.hc_attn_base.ptr),
        static_cast<const float *>(L.attn_norm.ptr),
        rt.d_post, rt.d_comb, rt.d_x_norm,
        cfg_.n_embd, T_mtp, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps, rt.stream);

    if (T_mtp < 16) {
        launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T_mtp, rt.stream);
        launch_linear_q8_0_prequant(L.wq, rt.d_scratch, rt.d_q, T_mtp, rt.stream);
        launch_linear_q8_0_prequant(L.wk, rt.d_scratch, rt.d_k, T_mtp, rt.stream);
        launch_linear_q8_0_prequant(L.wv, rt.d_scratch, rt.d_v, T_mtp, rt.stream);
    } else {
        launch_linear(rt.cublas, L.wq, rt.d_x_norm, rt.d_q, T_mtp, rt.d_scratch, rt.scratch_bytes, rt.stream);
        launch_linear(rt.cublas, L.wk, rt.d_x_norm, rt.d_k, T_mtp, rt.d_scratch, rt.scratch_bytes, rt.stream);
        launch_linear(rt.cublas, L.wv, rt.d_x_norm, rt.d_v, T_mtp, rt.d_scratch, rt.scratch_bytes, rt.stream);
    }

    launch_qk_norm_rope_neox(rt.d_q, static_cast<const float *>(L.attn_q_norm.ptr),
                             rt.d_mtp_pos, T_mtp, cfg_.n_head, cfg_.head_dim, cfg_.rms_eps, cfg_.rope_freq_base, rt.stream);
    launch_qk_norm_rope_neox(rt.d_k, static_cast<const float *>(L.attn_k_norm.ptr),
                             rt.d_mtp_pos, T_mtp, cfg_.n_head_kv, cfg_.head_dim, cfg_.rms_eps, cfg_.rope_freq_base, rt.stream);

    if (cfg_.kv_q8_0) {
        launch_kv_write_q8_0(rt.d_k, rt.d_v, k_cache_mtp_, v_cache_mtp_, rt.d_mtp_pos,
                             T_mtp, cfg_.n_head_kv, cfg_.head_dim, rt.stream);
    } else {
        launch_kv_write_f16(rt.d_k, rt.d_v, k_cache_mtp_, v_cache_mtp_, rt.d_mtp_pos,
                            T_mtp, cfg_.n_head_kv, cfg_.head_dim, rt.stream);
    }

    launch_gqa_attn(rt.d_q, k_cache_mtp_, v_cache_mtp_, cfg_.kv_q8_0,
                    rt.d_mtp_pos, rt.d_attn_out, rt.d_scratch, T_mtp, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim, kq_scale, rt.stream);

    launch_linear(rt.cublas, L.wo, rt.d_attn_out, rt.d_sub_out, T_mtp, rt.d_scratch, rt.scratch_bytes, rt.stream);
    launch_hc_post(rt.d_sub_out, rt.d_mtp_inpL, rt.d_post, rt.d_comb, rt.d_mtp_alt, cfg_.n_embd, T_mtp, rt.stream);

    // 3. FFN MoE sub-block: d_mtp_alt -> d_mtp_inpL
    launch_hc_flat_rmsnorm_mix(rt.d_mtp_alt, L.hc_ffn_fn, rt.d_mixes, hc_dim, 24, T_mtp, cfg_.rms_eps, rt.stream);
    launch_hc_sinkhorn_pre_rmsnorm(
        rt.d_mtp_alt, rt.d_mixes,
        static_cast<const float *>(L.hc_ffn_scale.ptr),
        static_cast<const float *>(L.hc_ffn_base.ptr),
        static_cast<const float *>(L.ffn_norm.ptr),
        rt.d_post, rt.d_comb, rt.d_x_norm,
        cfg_.n_embd, T_mtp, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps, rt.stream);

    launch_linear(rt.cublas, L.ffn_gate_inp, rt.d_x_norm, rt.d_router, T_mtp, rt.d_scratch, rt.scratch_bytes, rt.stream);
    launch_moe_route_sigmoid(
        rt.d_router, static_cast<const float *>(L.ffn_exp_probs_b.ptr),
        rt.d_top_ids, rt.d_top_w,
        T_mtp, cfg_.n_expert, cfg_.n_expert_used, cfg_.expert_weights_norm, cfg_.expert_weights_scale, rt.stream);
    launch_moe_ffn_q8_0_with_shared(
        L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps,
        L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
        rt.d_x_norm, rt.d_top_ids, rt.d_top_w, rt.d_sub_out,
        rt.d_scratch, T_mtp, cfg_.n_expert_used, rt.stream);

    launch_hc_post(rt.d_sub_out, rt.d_mtp_alt, rt.d_post, rt.d_comb, rt.d_mtp_inpL, cfg_.n_embd, T_mtp, rt.stream);

    // 4. MTP head on last token -> output_ -> argmax
    const float * last_h = rt.d_mtp_inpL + static_cast<size_t>(T_mtp - 1) * hc_dim;
    launch_hc_flat_rmsnorm_mix(last_h, mtp_.hc_head_fn, rt.d_mixes, hc_dim, 4, 1, cfg_.rms_eps, rt.stream);
    launch_hc_head_rmsnorm(
        last_h, rt.d_mixes,
        static_cast<const float *>(mtp_.hc_head_scale.ptr),
        static_cast<const float *>(mtp_.hc_head_base.ptr),
        static_cast<const float *>(mtp_.shared_head_norm.ptr),
        rt.d_x_norm, cfg_.n_embd, 1, cfg_.hc_eps, cfg_.rms_eps, rt.stream);

    launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, rt.stream);
    launch_argmax_f32_batch(rt.d_logits, rt.d_argmax, cfg_.n_vocab, 1, rt.stream);

    int32_t draft_tok = 0;
    cuda_ok(cudaMemcpyAsync(&draft_tok, rt.d_argmax, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, rt.stream), "mtp argmax D2H");
    cuda_ok(cudaStreamSynchronize(rt.stream), "mtp sync");
    return draft_tok;
}

int32_t MaxwellMoeModel::mtp_prefill_chunk(const std::vector<int32_t> & next_tokens,
                                           const std::vector<int32_t> & positions) {
    if (!mtp_.loaded) throw std::runtime_error("MTP weights not loaded");
    const int L = static_cast<int>(next_tokens.size());
    std::vector<int32_t> mtp_pos(L);
    for (int i = 0; i < L; ++i) mtp_pos[i] = positions[i] + mtp_pos_offset_;
    return run_mtp_on_last_gpu(next_tokens.data(), runtimes_.back().d_inpL, mtp_pos.data(), L);
}

int32_t MaxwellMoeModel::mtp_prefill_and_draft(const std::vector<int32_t> & prompt, int32_t tok0) {
    if (!mtp_.loaded) throw std::runtime_error("MTP weights not loaded");
    const int L = static_cast<int>(prompt.size());
    std::vector<int32_t> mtp_toks(L);
    std::vector<int32_t> mtp_pos(L);
    for (int i = 0; i < L - 1; ++i) {
        mtp_toks[i] = prompt[i + 1];
        mtp_pos[i]  = i;
    }
    mtp_toks[L - 1] = tok0;
    mtp_pos[L - 1]  = L - 1;
    return mtp_prefill_chunk(mtp_toks, mtp_pos);
}

void MaxwellMoeModel::decode_mtp_greedy_step(int32_t & cur_tok,
                                             int32_t & draft_tok,
                                             int32_t & cur_pos,
                                             std::vector<int32_t> & out_emitted,
                                             MtpStats & stats) {
    const int hc_dim = cfg_.hc_mult * cfg_.n_embd;

    if (use_cuda_graphs_ && !graphs_t2_ready_) {
        capture_decode_graphs(2);
    }

    int32_t toks2[2] = {cur_tok, draft_tok};
    int32_t pos2[2]  = {cur_pos, cur_pos + 1};

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "mtp_v setDevice");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, pos2, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "pos2");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, toks2, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok2");
        }
    }

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        if (gi > 0) {
            auto & src_rt = runtimes_[gi - 1];
            cuda_ok(cudaSetDevice(src_rt.device), "mtp_v rec");
            cuda_ok(cudaEventRecord(src_rt.done_event, src_rt.stream), "mtp_v event");
            cuda_ok(cudaSetDevice(rt.device), "mtp_v wait");
            cuda_ok(cudaStreamWaitEvent(rt.stream, src_rt.done_event, 0), "mtp_v waitEvent");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_inpL, rt.device, src_rt.d_inpL, src_rt.device,
                                        static_cast<size_t>(2) * hc_dim * sizeof(float), rt.stream), "mtp_v peer");
        }
        cuda_ok(cudaSetDevice(rt.device), "mtp_v launch");
        if (use_cuda_graphs_ && rt.decode_graph_t2) {
            cuda_ok(cudaGraphLaunch(rt.decode_graph_t2, rt.stream), "cudaGraphLaunch t2");
        } else {
            run_gpu_segment(gi, 2, true, true);
        }
    }

    auto & last_rt = runtimes_.back();
    int32_t pred[2] = {0, 0};
    cuda_ok(cudaSetDevice(last_rt.device), "mtp_v D2H");
    cuda_ok(cudaMemcpyAsync(pred, last_rt.d_argmax, 2 * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, last_rt.stream), "pred2");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "mtp_v sync");

    stats.steps += 1;

    if (pred[0] == draft_tok) {
        // ACCEPT! Both draft_tok (pos cur_pos+1) and pred[1] (pos cur_pos+2) are exact greedy outputs!
        stats.accepted += 1;
        stats.emitted  += 2;
        out_emitted.push_back(draft_tok);
        out_emitted.push_back(pred[1]);

        int32_t mtp_t[2] = {draft_tok, pred[1]};
        int32_t mtp_p[2] = {cur_pos + mtp_pos_offset_, cur_pos + 1 + mtp_pos_offset_};
        int32_t next_draft = run_mtp_on_last_gpu(mtp_t, last_rt.d_inpL, mtp_p, 2);

        cur_tok   = pred[1];
        draft_tok = next_draft;
        cur_pos  += 2;
    } else {
        // REJECT: pred[0] is exact greedy output for pos cur_pos+1; backbone KV slot cur_pos+1 will be overwritten next step!
        stats.emitted += 1;
        out_emitted.push_back(pred[0]);

        int32_t mtp_t[1] = {pred[0]};
        int32_t mtp_p[1] = {cur_pos + mtp_pos_offset_};
        int32_t next_draft = run_mtp_on_last_gpu(mtp_t, last_rt.d_inpL, mtp_p, 1);

        cur_tok   = pred[0];
        draft_tok = next_draft;
        cur_pos  += 1;
    }
}

} // namespace rookery
