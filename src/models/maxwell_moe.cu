#include "models/maxwell_moe.h"
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

// Replicate [T, n_embd] -> [T, hc_mult, n_embd] via coalesced float4 stores.
__global__ void broadcast_embd_hc_kernel(
    const float * __restrict__ src,
    float *       __restrict__ dst,
    int n_embd, int hc_mult) {
    const int t  = blockIdx.y;
    const int n4 = n_embd >> 2;
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const float4 v = reinterpret_cast<const float4 *>(src + static_cast<size_t>(t) * n_embd)[i];
    for (int h = 0; h < hc_mult; ++h) {
        reinterpret_cast<float4 *>(dst + (static_cast<size_t>(t) * hc_mult + h) * n_embd)[i] = v;
    }
}

} // namespace

void MaxwellMoeModel::invalidate_graphs() noexcept {
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        if (rt.decode_graph_t1) { cudaGraphExecDestroy(rt.decode_graph_t1); rt.decode_graph_t1 = nullptr; }
        if (rt.decode_graph_t2) { cudaGraphExecDestroy(rt.decode_graph_t2); rt.decode_graph_t2 = nullptr; }
    }
    if (!runtimes_.empty()) {
        if (embd_gi_ < runtimes_.size()) {
            cudaSetDevice(runtimes_[embd_gi_].device);
            if (graph_embd_t1_) { cudaGraphExecDestroy(graph_embd_t1_); graph_embd_t1_ = nullptr; }
            if (graph_embd_t2_) { cudaGraphExecDestroy(graph_embd_t2_); graph_embd_t2_ = nullptr; }
        }
        if (head_gi_ < runtimes_.size()) {
            cudaSetDevice(runtimes_[head_gi_].device);
            if (graph_head_t1_) { cudaGraphExecDestroy(graph_head_t1_); graph_head_t1_ = nullptr; }
            if (graph_head_t2_) { cudaGraphExecDestroy(graph_head_t2_); graph_head_t2_ = nullptr; }
        }
        cudaSetDevice(runtimes_.back().device);
        if (graph_mtp_t1_) { cudaGraphExecDestroy(graph_mtp_t1_); graph_mtp_t1_ = nullptr; }
        if (graph_mtp_t2_) { cudaGraphExecDestroy(graph_mtp_t2_); graph_mtp_t2_ = nullptr; }
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
                           KvCacheType kv_type,
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
    cfg_.kv_type              = kv_type;
    cfg_.idx_n_head           = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.head_count", 4));
    cfg_.idx_head_dim         = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.key_length", 128));
    cfg_.idx_top_k            = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.top_k", 16));
    cfg_.idx_blk_size         = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.block_size", 128));
    cfg_.idx_local_blks       = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.local_blocks", 8));
    cfg_.idx_prefix_blks      = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.prefix_blocks", 8));
    cfg_.idx_dense_prefix     = static_cast<int>(gguf.get_u64("maxwell-moe.attention.indexer.dense_prefix_tokens", 16384));
    msa_enabled_              = (cfg_.max_ctx > cfg_.idx_dense_prefix);

    const size_t kv_bytes_per_layer = (kv_type == KvCacheType::Q8_0)
        ? static_cast<size_t>(cfg_.max_ctx) * cfg_.n_head_kv * (cfg_.head_dim / 32) * sizeof(BlockQ8_0)
        : (kv_type == KvCacheType::Q4_0)
            ? static_cast<size_t>(cfg_.max_ctx) * cfg_.n_head_kv * (cfg_.head_dim / 32) * sizeof(BlockQ4_0)
            : static_cast<size_t>(cfg_.max_ctx) * cfg_.n_head_kv * cfg_.head_dim * sizeof(uint16_t);
    const bool ik_q4_0 = (kv_type == KvCacheType::Q4_0);
    const size_t ik_bytes_per_layer = ik_q4_0
        ? static_cast<size_t>(cfg_.max_ctx) * (cfg_.idx_head_dim / 32) * sizeof(BlockQ4_0)
        : static_cast<size_t>(cfg_.max_ctx) * cfg_.idx_head_dim * sizeof(uint16_t);
    const bool will_load_mtp = load_mtp && cfg_.n_nextn > 0 &&
        (gguf.find_tensor("blk." + std::to_string(cfg_.n_layers) + ".nextn.eh_proj.weight") != nullptr);

    // Enable P2P access between adjacent GPUs when supported
    for (size_t i = 0; i < gpu_ids.size(); ++i) {
        for (size_t j = 0; j < gpu_ids.size(); ++j) {
            if (i == j) continue;
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, gpu_ids[i], gpu_ids[j]) == cudaSuccess && can) {
                cudaSetDevice(gpu_ids[i]);
                cudaDeviceEnablePeerAccess(gpu_ids[j], 0);
                cudaGetLastError();
            }
        }
    }

    runtimes_.resize(gpu_ids.size());
    std::vector<int> layer_dev(cfg_.n_layers, gpu_ids[0]);

    // Detect asymmetric dual-GPU topology (e.g. high-BW 16 GiB GPU0 + secondary 6 GiB GPU1):
    // Place cold token_embd on GPU1 (and share it with blk.45 MTP!), keep hot output lm_head on GPU0,
    // and maximize GPU0 layer count dynamically using real cudaMemGetInfo + exact tensor/KV sizes.
    asym_mode_ = false;
    embd_gi_   = 0;
    head_gi_   = gpu_ids.size() - 1;

    if (gpu_ids.size() == 2) {
        size_t free0 = 0, tot0 = 0, free1 = 0, tot1 = 0;
        cudaSetDevice(gpu_ids[0]); cudaMemGetInfo(&free0, &tot0);
        cudaSetDevice(gpu_ids[1]); cudaMemGetInfo(&free1, &tot1);
        const int sm0 = device_sm_version(gpu_ids[0]);
        const int sm1 = device_sm_version(gpu_ids[1]);

        if (free0 > free1 * 14 / 10 || (sm0 == 60 && sm1 == 61)) {
            asym_mode_ = true;
            embd_gi_   = 1;
            head_gi_   = 0;

            auto tbytes = [&](const std::string & name) -> size_t {
                const auto * info = gguf.find_tensor(name);
                return info ? info->nbytes : 0;
            };
            auto layer_bytes = [&](int il, bool is_dense, bool has_idx) -> size_t {
                const std::string p = "blk." + std::to_string(il) + ".";
                size_t b = tbytes(p + "attn_norm.weight") + tbytes(p + "ffn_norm.weight")
                         + tbytes(p + "attn_q.weight") + tbytes(p + "attn_k.weight")
                         + tbytes(p + "attn_v.weight") + tbytes(p + "attn_output.weight")
                         + tbytes(p + "attn_q_norm.weight") + tbytes(p + "attn_k_norm.weight")
                         + tbytes(p + "hc_attn_fn.weight") + tbytes(p + "hc_attn_base.weight") + tbytes(p + "hc_attn_scale.weight")
                         + tbytes(p + "hc_ffn_fn.weight") + tbytes(p + "hc_ffn_base.weight") + tbytes(p + "hc_ffn_scale.weight");
                if (has_idx) {
                    b += tbytes(p + "indexer.q_proj.weight") + tbytes(p + "indexer.k_proj.weight")
                       + tbytes(p + "indexer.q_norm.weight") + tbytes(p + "indexer.k_norm.weight");
                }
                if (is_dense) {
                    b += tbytes(p + "ffn_gate.weight") + tbytes(p + "ffn_up.weight") + tbytes(p + "ffn_down.weight");
                } else {
                    b += tbytes(p + "ffn_gate_inp.weight") + tbytes(p + "exp_probs_b.bias")
                       + tbytes(p + "ffn_gate_exps.weight") + tbytes(p + "ffn_up_exps.weight") + tbytes(p + "ffn_down_exps.weight")
                       + tbytes(p + "ffn_gate_shexp.weight") + tbytes(p + "ffn_up_shexp.weight") + tbytes(p + "ffn_down_shexp.weight");
                }
                return b + (1ULL << 20); // alignment padding
            };

            std::vector<size_t> lb(cfg_.n_layers, 0);
            for (int il = 0; il < cfg_.n_layers; ++il) {
                lb[il] = layer_bytes(il, il < cfg_.n_dense_lead, msa_enabled_);
            }
            const size_t embd_b = tbytes("token_embd.weight");
            const size_t out_b  = tbytes("output.weight") ? tbytes("output.weight") : embd_b;
            const size_t mtp_b  = will_load_mtp
                ? (layer_bytes(cfg_.n_layers, false, false) + tbytes("blk." + std::to_string(cfg_.n_layers) + ".nextn.eh_proj.weight") + (4ULL << 20))
                : 0;

            const size_t kv_per_l_tot = 2 * kv_bytes_per_layer + (msa_enabled_ ? ik_bytes_per_layer : 0);
            const size_t staging_peak0 = std::max<size_t>(out_b, 160ULL << 20);
            const size_t staging_peak1 = std::max<size_t>(embd_b, 160ULL << 20);
            // Runtime buffers (~150 MiB) + cuBLAS/CUDA graphs (~260 MiB) + safety headroom (~160 MiB)
            const size_t rt_margin0 = 560ULL << 20;
            const size_t rt_margin1 = (will_load_mtp ? 520ULL : 440ULL) << 20;
            const size_t load_margin0 = staging_peak0 + (220ULL << 20);
            const size_t load_margin1 = staging_peak1 + (220ULL << 20);

            int best_split = -1;
            for (int s = cfg_.n_layers - 1; s >= 1; --s) {
                size_t w0 = out_b, w1 = embd_b + mtp_b;
                for (int il = 0; il < s; ++il) w0 += lb[il];
                for (int il = s; il < cfg_.n_layers; ++il) w1 += lb[il];
                const size_t kv0 = static_cast<size_t>(s) * kv_per_l_tot;
                const size_t kv1 = static_cast<size_t>(cfg_.n_layers - s) * kv_per_l_tot
                                 + (will_load_mtp ? 2 * kv_bytes_per_layer : 0);
                const bool ok0 = (w0 + load_margin0 <= free0) && (w0 + kv0 + rt_margin0 <= free0);
                const bool ok1 = (w1 + load_margin1 <= free1) && (w1 + kv1 + rt_margin1 <= free1);
                if (ok0 && ok1) {
                    best_split = s;
                    break;
                }
            }
            if (best_split < 0) {
                int64_t best_slack = INT64_MIN;
                const size_t rt_est0 = 380ULL << 20;
                const size_t rt_est1 = (will_load_mtp ? 420ULL : 340ULL) << 20;
                for (int s = cfg_.n_layers - 1; s >= 1; --s) {
                    size_t w0 = out_b, w1 = embd_b + mtp_b;
                    for (int il = 0; il < s; ++il) w0 += lb[il];
                    for (int il = s; il < cfg_.n_layers; ++il) w1 += lb[il];
                    const size_t kv0 = static_cast<size_t>(s) * kv_per_l_tot;
                    const size_t kv1 = static_cast<size_t>(cfg_.n_layers - s) * kv_per_l_tot
                                     + (will_load_mtp ? 2 * kv_bytes_per_layer : 0);
                    const int64_t s0 = static_cast<int64_t>(free0) - static_cast<int64_t>(std::max(w0 + load_margin0, w0 + kv0 + rt_est0));
                    const int64_t s1 = static_cast<int64_t>(free1) - static_cast<int64_t>(std::max(w1 + load_margin1, w1 + kv1 + rt_est1));
                    const int64_t slack = std::min(s0, s1);
                    if (slack > best_slack) { best_slack = slack; best_split = s; }
                }
            }

            runtimes_[0].device      = gpu_ids[0];
            runtimes_[0].layer_start = 0;
            runtimes_[0].layer_end   = best_split;
            runtimes_[1].device      = gpu_ids[1];
            runtimes_[1].layer_start = best_split;
            runtimes_[1].layer_end   = cfg_.n_layers;
            for (int il = 0; il < cfg_.n_layers; ++il) {
                layer_dev[il] = (il < best_split) ? gpu_ids[0] : gpu_ids[1];
            }
        }
    }

    if (!asym_mode_) {
        std::vector<float> sw = split_weights;
        if (sw.size() != gpu_ids.size()) sw.assign(gpu_ids.size(), 1.0f);
        const float sw_sum = std::accumulate(sw.begin(), sw.end(), 0.0f);

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
    }

    constexpr bool conv_bf16 = false;
    const int embd_dev = runtimes_[embd_gi_].device;
    const int head_dev = runtimes_[head_gi_].device;
    const int last_dev = runtimes_.back().device;

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

    // Parallel weight upload across all active GPUs
    std::vector<std::thread> load_threads;
    std::vector<std::exception_ptr> load_errs(runtimes_.size(), nullptr);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        load_threads.emplace_back([&, gi]() {
            try {
                const int dev = runtimes_[gi].device;
                cuda_ok(cudaSetDevice(dev), "maxwell load setDevice");

                if (gi == embd_gi_) {
                    tok_embd_ = pool_.upload(gguf, "token_embd.weight", dev, conv_bf16);
                }
                if (gi == head_gi_) {
                    output_ = pool_.upload(gguf, "output.weight", dev, conv_bf16, false);
                    if (!output_.valid()) {
                        output_ = pool_.upload(gguf, "token_embd.weight", dev, conv_bf16);
                    }
                }
                if (gi + 1 == runtimes_.size()) {
                    hc_head_fn_    = pool_.upload(gguf, "output_hc_fn.weight",   dev, conv_bf16);
                    hc_head_base_  = pool_.upload(gguf, "output_hc_base.weight", dev, conv_bf16);
                    hc_head_scale_ = pool_.upload(gguf, "output_hc_scale.weight",dev, conv_bf16);
                    output_norm_   = pool_.upload(gguf, "output_norm.weight",    dev, conv_bf16);
                }

                for (int il = runtimes_[gi].layer_start; il < runtimes_[gi].layer_end; ++il) {
                    load_layer_weights(il, dev, il < cfg_.n_dense_lead, layers_[il]);
                }

                if (will_load_mtp && gi + 1 == runtimes_.size()) {
                    const int il = cfg_.n_layers;
                    const std::string p = "blk." + std::to_string(il) + ".";
                    load_layer_weights(il, dev, false, mtp_.layer);
                    mtp_.eh_proj          = pool_.upload(gguf, p + "nextn.eh_proj.weight",       dev, conv_bf16);
                    mtp_.enorm            = pool_.upload(gguf, p + "nextn.enorm.weight",         dev, conv_bf16);
                    mtp_.hnorm            = pool_.upload(gguf, p + "nextn.hnorm.weight",         dev, conv_bf16);
                    mtp_.shared_head_norm = pool_.upload(gguf, p + "layer_output_norm.weight",   dev, conv_bf16);
                    mtp_.hc_head_fn       = pool_.upload(gguf, p + "nextn.hc_head_fn.weight",    dev, conv_bf16);
                    mtp_.hc_head_base     = pool_.upload(gguf, p + "nextn.hc_head_base.weight",  dev, conv_bf16);
                    mtp_.hc_head_scale    = pool_.upload(gguf, p + "nextn.hc_head_scale.weight", dev, conv_bf16);
                    if (embd_dev == dev) {
                        // Zero-copy reuse: cold token_embd is ALREADY on last_dev!
                        mtp_.tok_embd = tok_embd_;
                    } else {
                        mtp_.tok_embd = pool_.upload(gguf, "token_embd.weight", dev, conv_bf16);
                    }
                    mtp_.loaded = true;
                }

                cuda_ok(cudaDeviceSynchronize(), "maxwell load sync");
            } catch (...) {
                load_errs[gi] = std::current_exception();
            }
        });
    }
    for (auto & th : load_threads) th.join();
    for (auto & ep : load_errs) if (ep) std::rethrow_exception(ep);

    // Release staging BEFORE allocating KV caches and runtime buffers (saves 320+ MiB peak VRAM!)
    pool_.release_staging();

    for (int il = 0; il < cfg_.n_layers; ++il) {
        const int dev = layer_dev[il];
        cuda_ok(cudaSetDevice(dev), "kv alloc setDevice");
        cuda_ok(cudaMalloc(&k_caches_[il], kv_bytes_per_layer), "cudaMalloc k_cache");
        cuda_ok(cudaMalloc(&v_caches_[il], kv_bytes_per_layer), "cudaMalloc v_cache");
        if (msa_enabled_ && layers_[il].idx_k_proj.valid()) {
            cuda_ok(cudaMalloc(&ik_caches_[il], ik_bytes_per_layer), "cudaMalloc ik_cache");
        }
    }
    if (mtp_.loaded) {
        cuda_ok(cudaSetDevice(last_dev), "mtp kv setDevice");
        cuda_ok(cudaMalloc(&k_cache_mtp_, kv_bytes_per_layer), "cudaMalloc k_cache_mtp");
        cuda_ok(cudaMalloc(&v_cache_mtp_, kv_bytes_per_layer), "cudaMalloc v_cache_mtp");
        cuda_ok(cudaMemset(k_cache_mtp_, 0, kv_bytes_per_layer), "memset k_cache_mtp");
        cuda_ok(cudaMemset(v_cache_mtp_, 0, kv_bytes_per_layer), "memset v_cache_mtp");
    }

    const int Tmax   = max_batch_;
    const int q_dim  = cfg_.n_head * cfg_.head_dim;
    const int kv_dim = cfg_.n_head_kv * cfg_.head_dim;

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        const int dev = rt.device;
        const bool has_dense = (rt.layer_start < cfg_.n_dense_lead && rt.layer_end > 0);
        const int  max_ff    = has_dense ? std::max(cfg_.n_ff_dense, cfg_.n_ff_shexp) : cfg_.n_ff_shexp;

        cuda_ok(cudaSetDevice(dev), "rt setDevice");
        cuda_ok(cudaStreamCreate(&rt.stream), "cudaStreamCreate");
        cuda_ok(cudaEventCreateWithFlags(&rt.done_event, cudaEventDisableTiming), "cudaEventCreate");
        if (cublasCreate(&rt.cublas) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }
        if (device_has_tensor_cores(dev)) cublasSetMathMode(rt.cublas, CUBLAS_TENSOR_OP_MATH);

        rt.d_inpL     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
        rt.d_inpL_alt = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
        rt.d_mixes    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * 24);
        rt.d_post     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult);
        rt.d_comb     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.hc_mult);
        rt.d_x_norm   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
        rt.d_q        = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * q_dim);
        rt.d_k        = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * kv_dim);
        rt.d_v        = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * kv_dim);
        rt.d_attn_out = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * q_dim);
        rt.d_sub_out  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
        rt.d_ffn_g    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * max_ff);
        rt.d_ffn_u    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * max_ff);
        rt.d_router   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert);
        rt.d_top_ids  = dev_alloc<int32_t>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert_used);
        rt.d_top_w    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert_used);
        rt.d_tokens   = dev_alloc<int32_t>(dev, Tmax);
        rt.d_pos      = dev_alloc<int32_t>(dev, Tmax);
        if (gi == head_gi_) {
            rt.d_logits = dev_alloc<float>(dev, static_cast<size_t>(2) * cfg_.n_vocab);
            rt.d_argmax = dev_alloc<int32_t>(dev, 2);
        }
        if (gi + 1 == runtimes_.size() && mtp_.loaded) {
            rt.d_mtp_e     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
            rt.d_mtp_eh_in = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * 2 * cfg_.n_embd);
            rt.d_mtp_inpL  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
            rt.d_mtp_alt   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
            rt.d_mtp_tok   = dev_alloc<int32_t>(dev, Tmax);
            rt.d_mtp_pos   = dev_alloc<int32_t>(dev, Tmax);
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
    std::printf("Loaded Maxwell-V1-Flash (%d backbone%s, ctx=%d, kv=%s%s, SoA=LDG.128%s) in %.2f s\n",
                cfg_.n_layers, mtp_.loaded ? " + blk.45 MTP" : "", cfg_.max_ctx, kv_cache_type_name(cfg_.kv_type),
                (msa_enabled_ ? (ik_q4_0 ? ", ik=Q4_0" : ", ik=F16") : ""),
                asym_mode_ ? ", asym:head@GPU0,embd@GPU1" : "", sec);
    for (const auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        std::printf("  GPU%d: layers [%d..%d) (%2d layers), %.2f GiB weights, VRAM used %.2f / %.2f GiB\n",
                    rt.device, rt.layer_start, rt.layer_end, rt.layer_end - rt.layer_start,
                    static_cast<double>(pool_.bytes_on_device(rt.device)) / (1024.0 * 1024.0 * 1024.0),
                    static_cast<double>(tb - fb) / (1024.0 * 1024.0 * 1024.0),
                    static_cast<double>(tb) / (1024.0 * 1024.0 * 1024.0));
    }
}

void MaxwellMoeModel::run_head_on_head_gpu(int head_T, bool run_argmax) {
    auto & rt = runtimes_[head_gi_];
    cuda_ok(cudaSetDevice(rt.device), "head setDevice");
    launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, head_T, rt.d_scratch, rt.scratch_bytes, rt.stream);
    if (run_argmax) {
        launch_argmax_f32_batch(rt.d_logits, rt.d_argmax, cfg_.n_vocab, head_T, rt.stream);
    }
}

void MaxwellMoeModel::run_gpu_segment(size_t gi, int T, bool run_argmax, bool logits_all_t, int kv_len_max) {
    auto & rt = runtimes_[gi];
    cuda_ok(cudaSetDevice(rt.device), "segment setDevice");
    (void) kv_len_max;

    const int   hc_dim   = cfg_.hc_mult * cfg_.n_embd;
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
    static const bool prof_env = (std::getenv("ROOKERY_PROFILE") != nullptr && std::atoi(std::getenv("ROOKERY_PROFILE")) > 0);
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    const bool do_prof = prof_env && (cudaStreamIsCapturing(rt.stream, &cap_st) == cudaSuccess) && (cap_st == cudaStreamCaptureStatusNone);
    double p_embd = 0, p_mhc = 0, p_qkv = 0, p_qkkv = 0, p_attn = 0, p_wo = 0, p_dense = 0, p_rtr = 0, p_moe = 0, p_head = 0;
    auto p_mark = [&]() -> std::chrono::steady_clock::time_point {
        if (do_prof) cudaStreamSynchronize(rt.stream);
        return std::chrono::steady_clock::now();
    };
    auto p_add = [&](double & acc, std::chrono::steady_clock::time_point & tprev) {
        if (!do_prof) return;
        cudaStreamSynchronize(rt.stream);
        auto now = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(now - tprev).count();
        tprev = now;
    };
    auto pt = p_mark();

    if (gi == 0) {
        if (asym_mode_) {
            // rt.d_x_norm already holds [T, n_embd] copied from GPU1 tok_embd lookup!
            const int n4 = cfg_.n_embd >> 2;
            broadcast_embd_hc_kernel<<<dim3((n4 + 255) / 256, T), 256, 0, rt.stream>>>(
                rt.d_x_norm, rt.d_inpL, cfg_.n_embd, cfg_.hc_mult);
        } else {
            launch_embedding_lookup(tok_embd_, rt.d_tokens, rt.d_inpL, cfg_.n_embd, cfg_.hc_mult, T, rt.stream);
        }
        p_add(p_embd, pt);
    }

    bool attn_mixes_ready = false;
    for (int il = rt.layer_start; il < rt.layer_end; ++il) {
        const auto & L = layers_[il];

        // --- Attention sub-block: rt.d_inpL -> rt.d_inpL_alt ---
        if (!attn_mixes_ready) {
            launch_hc_flat_rmsnorm_mix(rt.d_inpL, L.hc_attn_fn, rt.d_mixes, hc_dim, 24, T, cfg_.rms_eps, rt.stream);
        }
        attn_mixes_ready = false;
        launch_hc_sinkhorn_pre_rmsnorm(
            rt.d_inpL, rt.d_mixes,
            static_cast<const float *>(L.hc_attn_scale.ptr),
            static_cast<const float *>(L.hc_attn_base.ptr),
            static_cast<const float *>(L.attn_norm.ptr),
            rt.d_post, rt.d_comb, rt.d_x_norm,
            cfg_.n_embd, T, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps, rt.stream);
        p_add(p_mhc, pt);

        if (T < 16 &&
            (L.wq.dtype == DType::Q8_0 || is_kquant_dtype(L.wq.dtype)) &&
            (L.wk.dtype == DType::Q8_0 || is_kquant_dtype(L.wk.dtype)) &&
            (L.wv.dtype == DType::Q8_0 || is_kquant_dtype(L.wv.dtype))) {
            const DeviceTensor * ws[3] = {&L.wq, &L.wk, &L.wv};
            float *              ys[3] = {rt.d_q, rt.d_k, rt.d_v};
            if (!try_launch_linear_multi_sm60_h2(ws, ys, 3, rt.d_x_norm, T, rt.d_scratch, rt.scratch_bytes, rt.stream)) {
                launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T, rt.stream);
                launch_linear_multi_prequant(ws, ys, 3, rt.d_scratch, T, rt.stream);
            }
        } else {
            launch_linear(rt.cublas, L.wq, rt.d_x_norm, rt.d_q, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            launch_linear(rt.cublas, L.wk, rt.d_x_norm, rt.d_k, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            launch_linear(rt.cublas, L.wv, rt.d_x_norm, rt.d_v, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
        }
        p_add(p_qkv, pt);

        launch_maxwell_qk_norm_rope_kv_write(
            rt.d_q, rt.d_k, rt.d_v,
            static_cast<const float *>(L.attn_q_norm.ptr),
            static_cast<const float *>(L.attn_k_norm.ptr),
            k_caches_[il], v_caches_[il], cfg_.kv_type,
            rt.d_pos, T, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim,
            cfg_.rms_eps, cfg_.rope_freq_base, rt.stream);
        p_add(p_qkkv, pt);

        if (msa_enabled_ && L.idx_k_proj.valid() && ik_caches_[il] != nullptr) {
            const bool ik_q4_0 = (cfg_.kv_type == KvCacheType::Q4_0);
            launch_linear(rt.cublas, L.idx_k_proj, rt.d_x_norm, rt.d_ik, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            launch_qk_norm_rope_neox(rt.d_ik, static_cast<const float *>(L.idx_k_norm.ptr),
                                     rt.d_pos, T, 1, cfg_.idx_head_dim, cfg_.rms_eps, cfg_.rope_freq_base, rt.stream);
            if (ik_q4_0) {
                launch_idx_k_write_q4_0(rt.d_ik, ik_caches_[il], rt.d_pos, T, cfg_.idx_head_dim, rt.stream);
            } else {
                launch_idx_k_write_f16(rt.d_ik, ik_caches_[il], rt.d_pos, T, cfg_.idx_head_dim, rt.stream);
            }

            if (T >= 8 && kv_len_max > 0 && kv_len_max <= cfg_.idx_dense_prefix) {
                launch_gqa_attn(rt.d_q, k_caches_[il], v_caches_[il], cfg_.kv_type,
                                rt.d_pos, rt.d_attn_out, rt.d_scratch, T, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim,
                                kq_scale, rt.stream, rt.cublas, rt.scratch_bytes, kv_len_max);
            } else {
                launch_linear(rt.cublas, L.idx_q_proj, rt.d_x_norm, rt.d_iq, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                launch_qk_norm_rope_neox(rt.d_iq, static_cast<const float *>(L.idx_q_norm.ptr),
                                         rt.d_pos, T, cfg_.idx_n_head, cfg_.idx_head_dim, cfg_.rms_eps, cfg_.rope_freq_base, rt.stream);

                const int k_max = cfg_.idx_prefix_blks + cfg_.idx_local_blks + cfg_.idx_top_k;
                launch_msa_indexer_topk(
                    rt.d_iq, ik_caches_[il], ik_q4_0, rt.d_pos, rt.d_msa_blocks, rt.d_msa_kcnt, rt.d_scratch,
                    T, cfg_.max_ctx, cfg_.idx_n_head, cfg_.idx_blk_size, cfg_.idx_prefix_blks, cfg_.idx_local_blks,
                    k_max, cfg_.idx_dense_prefix, rt.stream);

                launch_gqa_attn_msa(
                    rt.d_q, k_caches_[il], v_caches_[il], cfg_.kv_type,
                    rt.d_pos, rt.d_msa_blocks, rt.d_msa_kcnt,
                    rt.d_attn_out, rt.d_scratch, T, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim,
                    cfg_.idx_blk_size, k_max, kq_scale, rt.stream);
            }
        } else {
            launch_gqa_attn(rt.d_q, k_caches_[il], v_caches_[il], cfg_.kv_type,
                            rt.d_pos, rt.d_attn_out, rt.d_scratch, T, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim,
                            kq_scale, rt.stream, rt.cublas, rt.scratch_bytes, kv_len_max);
        }
        p_add(p_attn, pt);

        launch_linear(rt.cublas, L.wo, rt.d_attn_out, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
        p_add(p_wo, pt);

        // --- FFN sub-block: rt.d_inpL_alt -> rt.d_inpL ---
        if (T <= 16 && L.hc_ffn_fn.dtype == DType::BF16) {
            launch_hc_post_flat_rmsnorm_mix(
                rt.d_sub_out, rt.d_inpL, rt.d_post, rt.d_comb, rt.d_inpL_alt,
                L.hc_ffn_fn, rt.d_mixes, cfg_.n_embd, 24, T, cfg_.rms_eps, rt.stream);
        } else {
            launch_hc_post(rt.d_sub_out, rt.d_inpL, rt.d_post, rt.d_comb, rt.d_inpL_alt, cfg_.n_embd, T, rt.stream);
            launch_hc_flat_rmsnorm_mix(rt.d_inpL_alt, L.hc_ffn_fn, rt.d_mixes, hc_dim, 24, T, cfg_.rms_eps, rt.stream);
        }
        launch_hc_sinkhorn_pre_rmsnorm(
            rt.d_inpL_alt, rt.d_mixes,
            static_cast<const float *>(L.hc_ffn_scale.ptr),
            static_cast<const float *>(L.hc_ffn_base.ptr),
            static_cast<const float *>(L.ffn_norm.ptr),
            rt.d_post, rt.d_comb, rt.d_x_norm,
            cfg_.n_embd, T, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps, rt.stream);
        p_add(p_mhc, pt);

        if (il < cfg_.n_dense_lead) {
            if (T < 16 &&
                (L.ffn_gate.dtype == DType::Q8_0 || is_kquant_dtype(L.ffn_gate.dtype)) &&
                (L.ffn_up.dtype   == DType::Q8_0 || is_kquant_dtype(L.ffn_up.dtype))) {
                const DeviceTensor * ws[2] = {&L.ffn_gate, &L.ffn_up};
                float *              ys[2] = {rt.d_ffn_g, rt.d_ffn_u};
                if (!try_launch_linear_multi_sm60_h2(ws, ys, 2, rt.d_x_norm, T, rt.d_scratch, rt.scratch_bytes, rt.stream)) {
                    launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T, rt.stream);
                    launch_linear_multi_prequant(ws, ys, 2, rt.d_scratch, T, rt.stream);
                }
            } else {
                launch_linear(rt.cublas, L.ffn_gate, rt.d_x_norm, rt.d_ffn_g, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                launch_linear(rt.cublas, L.ffn_up,   rt.d_x_norm, rt.d_ffn_u, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            }
            launch_silu_mul(rt.d_ffn_g, rt.d_ffn_u, rt.d_ffn_g, static_cast<int64_t>(T) * cfg_.n_ff_dense, rt.stream);
            launch_linear(rt.cublas, L.ffn_down, rt.d_ffn_g, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            p_add(p_dense, pt);
        } else {
            launch_linear(rt.cublas, L.ffn_gate_inp, rt.d_x_norm, rt.d_router, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            launch_moe_route_sigmoid(
                rt.d_router, static_cast<const float *>(L.ffn_exp_probs_b.ptr),
                rt.d_top_ids, rt.d_top_w,
                T, cfg_.n_expert, cfg_.n_expert_used, cfg_.expert_weights_norm, cfg_.expert_weights_scale, rt.stream);
            p_add(p_rtr, pt);
            const bool use_q8_moe =
                (L.ffn_gate_exps.dtype == DType::Q8_0 && L.ffn_up_exps.dtype == DType::Q8_0 &&
                 L.ffn_down_exps.dtype == DType::Q8_0 && L.ffn_gate_shexp.dtype == DType::Q8_0 &&
                 L.ffn_up_shexp.dtype  == DType::Q8_0 && L.ffn_down_shexp.dtype == DType::Q8_0);
            if (use_q8_moe) {
                launch_moe_ffn_q8_0_with_shared(
                    L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps,
                    L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
                    rt.d_x_norm, rt.d_top_ids, rt.d_top_w, rt.d_sub_out,
                    rt.d_scratch, T, cfg_.n_expert_used, rt.stream);
            } else {
                launch_moe_ffn_kquant_with_shared(
                    rt.cublas,
                    L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps,
                    L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
                    rt.d_x_norm, rt.d_top_ids, rt.d_top_w, rt.d_sub_out,
                    rt.d_scratch, rt.scratch_bytes, T, cfg_.n_expert_used, 0.0f, 0.0f, rt.stream);
            }
            p_add(p_moe, pt);
        }

        const bool layer_hooked = (hook_ != nullptr && hook_->hooks_layer(il));
        if (T <= 16 && !layer_hooked && (il + 1 < rt.layer_end) &&
            layers_[il + 1].hc_attn_fn.dtype == DType::BF16) {
            launch_hc_post_flat_rmsnorm_mix(
                rt.d_sub_out, rt.d_inpL_alt, rt.d_post, rt.d_comb, rt.d_inpL,
                layers_[il + 1].hc_attn_fn, rt.d_mixes, cfg_.n_embd, 24, T, cfg_.rms_eps, rt.stream);
            attn_mixes_ready = true;
        } else {
            launch_hc_post(rt.d_sub_out, rt.d_inpL_alt, rt.d_post, rt.d_comb, rt.d_inpL, cfg_.n_embd, T, rt.stream);
        }
        p_add(p_mhc, pt);

        if (layer_hooked) {
            hook_->on_post_layer(il, rt.device, rt.d_inpL, T, cfg_.n_embd, cfg_.hc_mult, rt.stream);
        }
    }

    // Final head collapse on last GPU; if head_gi_ == last GPU, also run output_ + argmax here!
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

        if (head_gi_ == gi) {
            launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, head_T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            if (run_argmax) {
                launch_argmax_f32_batch(rt.d_logits, rt.d_argmax, cfg_.n_vocab, head_T, rt.stream);
            }
        }
        p_add(p_head, pt);
    }

    if (do_prof) {
        const double tot = p_embd + p_mhc + p_qkv + p_qkkv + p_attn + p_wo + p_dense + p_rtr + p_moe + p_head;
        std::printf("[PROF GPU%d T=%d L=%d..%d] tot=%.2fms | embd=%.2f mHC=%.2f qkv=%.2f qkkv=%.2f attn=%.2f wo=%.2f dense=%.2f rtr=%.2f moe=%.2f head=%.2f\n",
                    rt.device, T, rt.layer_start, rt.layer_end, tot,
                    p_embd, p_mhc, p_qkv, p_qkkv, p_attn, p_wo, p_dense, p_rtr, p_moe, p_head);
    }
}

void MaxwellMoeModel::capture_decode_graphs(int T) {
    if (asym_mode_) {
        // 1. Cold token_embd lookup on GPU 1 (outputs [T, n_embd] into rt1.d_x_norm)
        auto & rt_e = runtimes_[embd_gi_];
        cuda_ok(cudaSetDevice(rt_e.device), "cap embd setDevice");
        cuda_ok(cudaStreamBeginCapture(rt_e.stream, cudaStreamCaptureModeGlobal), "begin cap embd");
        launch_embedding_lookup(tok_embd_, rt_e.d_tokens, rt_e.d_x_norm, cfg_.n_embd, 1, T, rt_e.stream);
        cudaGraph_t ge = nullptr;
        cuda_ok(cudaStreamEndCapture(rt_e.stream, &ge), "end cap embd");
        cudaGraphExec_t ee = nullptr;
        cuda_ok(cudaGraphInstantiate(&ee, ge, nullptr, nullptr, 0), "inst embd");
        cudaGraphDestroy(ge);
        if (T == 1) graph_embd_t1_ = ee; else graph_embd_t2_ = ee;
    }

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "capture setDevice");
        cuda_ok(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeGlobal), "begin capture");
        run_gpu_segment(gi, T, true, T == 2, 0);
        cudaGraph_t graph = nullptr;
        cuda_ok(cudaStreamEndCapture(rt.stream, &graph), "end capture");
        cudaGraphExec_t exec = nullptr;
        cuda_ok(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate graph");
        cudaGraphDestroy(graph);
        if (T == 1) rt.decode_graph_t1 = exec;
        else        rt.decode_graph_t2 = exec;
    }

    if (asym_mode_) {
        // Hot output_ + argmax on GPU 0
        auto & rt_h = runtimes_[head_gi_];
        cuda_ok(cudaSetDevice(rt_h.device), "cap head setDevice");
        cuda_ok(cudaStreamBeginCapture(rt_h.stream, cudaStreamCaptureModeGlobal), "begin cap head");
        run_head_on_head_gpu(T, true);
        cudaGraph_t gh = nullptr;
        cuda_ok(cudaStreamEndCapture(rt_h.stream, &gh), "end cap head");
        cudaGraphExec_t eh = nullptr;
        cuda_ok(cudaGraphInstantiate(&eh, gh, nullptr, nullptr, 0), "inst head");
        cudaGraphDestroy(gh);
        if (T == 1) graph_head_t1_ = eh; else graph_head_t2_ = eh;
    }

    if (mtp_.loaded) {
        auto & rt_m = runtimes_.back();
        cuda_ok(cudaSetDevice(rt_m.device), "cap mtp setDevice");
        cuda_ok(cudaStreamBeginCapture(rt_m.stream, cudaStreamCaptureModeGlobal), "begin cap mtp");
        run_mtp_body_on_last_gpu(rt_m.d_inpL, T, 0);
        cudaGraph_t gm = nullptr;
        cuda_ok(cudaStreamEndCapture(rt_m.stream, &gm), "end cap mtp");
        cudaGraphExec_t em = nullptr;
        cuda_ok(cudaGraphInstantiate(&em, gm, nullptr, nullptr, 0), "inst mtp");
        cudaGraphDestroy(gm);
        if (T == 1) graph_mtp_t1_ = em; else graph_mtp_t2_ = em;
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
    int kv_len_max = 0;
    for (int32_t p : positions) if (p + 1 > kv_len_max) kv_len_max = p + 1;

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "forward setDevice");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, positions.data(), T * sizeof(int32_t),
                                cudaMemcpyHostToDevice, rt.stream), "copy pos");
        if (gi == embd_gi_) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, tokens.data(), T * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, rt.stream), "copy tokens");
        }
    }

    if (asym_mode_) {
        auto & rt_e = runtimes_[embd_gi_];
        auto & rt_0 = runtimes_[0];
        cuda_ok(cudaSetDevice(rt_e.device), "fwd embd setDevice");
        launch_embedding_lookup(tok_embd_, rt_e.d_tokens, rt_e.d_x_norm, cfg_.n_embd, 1, T, rt_e.stream);
        cuda_ok(cudaEventRecord(rt_e.done_event, rt_e.stream), "fwd embd event");
        cuda_ok(cudaSetDevice(rt_0.device), "fwd wait embd");
        cuda_ok(cudaStreamWaitEvent(rt_0.stream, rt_e.done_event, 0), "fwd wait embd ev");
        cuda_ok(cudaMemcpyPeerAsync(rt_0.d_x_norm, rt_0.device, rt_e.d_x_norm, rt_e.device,
                                    static_cast<size_t>(T) * cfg_.n_embd * sizeof(float), rt_0.stream), "fwd embd peer");
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
        run_gpu_segment(gi, T, false, false, kv_len_max);
    }

    if (asym_mode_) {
        auto & last_rt = runtimes_.back();
        auto & head_rt = runtimes_[head_gi_];
        cuda_ok(cudaSetDevice(last_rt.device), "fwd head src");
        cuda_ok(cudaEventRecord(last_rt.done_event, last_rt.stream), "fwd head ev");
        cuda_ok(cudaSetDevice(head_rt.device), "fwd head dst");
        cuda_ok(cudaStreamWaitEvent(head_rt.stream, last_rt.done_event, 0), "fwd head wait");
        cuda_ok(cudaMemcpyPeerAsync(head_rt.d_x_norm, head_rt.device, last_rt.d_x_norm, last_rt.device,
                                    static_cast<size_t>(cfg_.n_embd) * sizeof(float), head_rt.stream), "fwd head peer");
        run_head_on_head_gpu(1, false);
    }

    auto & head_rt = runtimes_[head_gi_];
    cuda_ok(cudaSetDevice(head_rt.device), "logits setDevice");
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaMemcpyAsync(out_logits.data(), head_rt.d_logits, cfg_.n_vocab * sizeof(float),
                            cudaMemcpyDeviceToHost, head_rt.stream), "copy logits");
    cuda_ok(cudaStreamSynchronize(head_rt.stream), "final sync");
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
        if (gi == embd_gi_) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok");
        }
    }

    if (asym_mode_) {
        auto & rt_e = runtimes_[embd_gi_];
        auto & rt_0 = runtimes_[0];
        cuda_ok(cudaSetDevice(rt_e.device), "dec embd setDevice");
        if (use_cuda_graphs_ && graph_embd_t1_) {
            cuda_ok(cudaGraphLaunch(graph_embd_t1_, rt_e.stream), "graph embd t1");
        } else {
            launch_embedding_lookup(tok_embd_, rt_e.d_tokens, rt_e.d_x_norm, cfg_.n_embd, 1, 1, rt_e.stream);
        }
        cuda_ok(cudaEventRecord(rt_e.done_event, rt_e.stream), "dec embd ev");
        cuda_ok(cudaSetDevice(rt_0.device), "dec wait embd");
        cuda_ok(cudaStreamWaitEvent(rt_0.stream, rt_e.done_event, 0), "dec wait embd ev");
        cuda_ok(cudaMemcpyPeerAsync(rt_0.d_x_norm, rt_0.device, rt_e.d_x_norm, rt_e.device,
                                    static_cast<size_t>(cfg_.n_embd) * sizeof(float), rt_0.stream), "dec embd peer");
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
            run_gpu_segment(gi, 1, true, false, pos + 1);
        }
    }

    if (asym_mode_) {
        auto & last_rt = runtimes_.back();
        auto & head_rt = runtimes_[head_gi_];
        cuda_ok(cudaSetDevice(last_rt.device), "dec head src");
        cuda_ok(cudaEventRecord(last_rt.done_event, last_rt.stream), "dec head ev");
        cuda_ok(cudaSetDevice(head_rt.device), "dec head dst");
        cuda_ok(cudaStreamWaitEvent(head_rt.stream, last_rt.done_event, 0), "dec head wait");
        cuda_ok(cudaMemcpyPeerAsync(head_rt.d_x_norm, head_rt.device, last_rt.d_x_norm, last_rt.device,
                                    static_cast<size_t>(cfg_.n_embd) * sizeof(float), head_rt.stream), "dec head peer");
        if (use_cuda_graphs_ && graph_head_t1_) {
            cuda_ok(cudaGraphLaunch(graph_head_t1_, head_rt.stream), "graph head t1");
        } else {
            run_head_on_head_gpu(1, true);
        }
    }

    auto & head_rt = runtimes_[head_gi_];
    int32_t next_tok = 0;
    cuda_ok(cudaSetDevice(head_rt.device), "argmax D2H setDevice");
    cuda_ok(cudaMemcpyAsync(&next_tok, head_rt.d_argmax, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, head_rt.stream), "copy argmax");
    cuda_ok(cudaStreamSynchronize(head_rt.stream), "decode sync");
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
        if (gi == embd_gi_) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok");
        }
    }

    if (asym_mode_) {
        auto & rt_e = runtimes_[embd_gi_];
        auto & rt_0 = runtimes_[0];
        cuda_ok(cudaSetDevice(rt_e.device), "decl embd setDevice");
        if (use_cuda_graphs_ && graph_embd_t1_) {
            cuda_ok(cudaGraphLaunch(graph_embd_t1_, rt_e.stream), "graph embd t1");
        } else {
            launch_embedding_lookup(tok_embd_, rt_e.d_tokens, rt_e.d_x_norm, cfg_.n_embd, 1, 1, rt_e.stream);
        }
        cuda_ok(cudaEventRecord(rt_e.done_event, rt_e.stream), "decl embd ev");
        cuda_ok(cudaSetDevice(rt_0.device), "decl wait embd");
        cuda_ok(cudaStreamWaitEvent(rt_0.stream, rt_e.done_event, 0), "decl wait embd ev");
        cuda_ok(cudaMemcpyPeerAsync(rt_0.d_x_norm, rt_0.device, rt_e.d_x_norm, rt_e.device,
                                    static_cast<size_t>(cfg_.n_embd) * sizeof(float), rt_0.stream), "decl embd peer");
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
            run_gpu_segment(gi, 1, false, false, pos + 1);
        }
    }

    if (asym_mode_) {
        auto & last_rt = runtimes_.back();
        auto & head_rt = runtimes_[head_gi_];
        cuda_ok(cudaSetDevice(last_rt.device), "decl head src");
        cuda_ok(cudaEventRecord(last_rt.done_event, last_rt.stream), "decl head ev");
        cuda_ok(cudaSetDevice(head_rt.device), "decl head dst");
        cuda_ok(cudaStreamWaitEvent(head_rt.stream, last_rt.done_event, 0), "decl head wait");
        cuda_ok(cudaMemcpyPeerAsync(head_rt.d_x_norm, head_rt.device, last_rt.d_x_norm, last_rt.device,
                                    static_cast<size_t>(cfg_.n_embd) * sizeof(float), head_rt.stream), "decl head peer");
        if (use_cuda_graphs_ && graph_head_t1_) {
            cuda_ok(cudaGraphLaunch(graph_head_t1_, head_rt.stream), "graph head t1");
        } else {
            run_head_on_head_gpu(1, false);
        }
    }

    auto & head_rt = runtimes_[head_gi_];
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaSetDevice(head_rt.device), "logits D2H setDevice");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), head_rt.d_logits, cfg_.n_vocab * sizeof(float),
                            cudaMemcpyDeviceToHost, head_rt.stream), "copy logits");
    cuda_ok(cudaStreamSynchronize(head_rt.stream), "decode_l sync");
}

void MaxwellMoeModel::warmup_decode_graphs() {
    if (!use_cuda_graphs_) return;
    if (!graphs_t1_ready_) capture_decode_graphs(1);
    if (mtp_.loaded && !graphs_t2_ready_) capture_decode_graphs(2);
}

void MaxwellMoeModel::run_mtp_body_on_last_gpu(const float * d_h_states, int T_mtp, int kv_len_max) {
    auto & rt = runtimes_.back();
    cuda_ok(cudaSetDevice(rt.device), "mtp body setDevice");
    (void) kv_len_max;

    const int   hc_dim   = cfg_.hc_mult * cfg_.n_embd;
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
    const auto & L       = mtp_.layer;

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

    if (T_mtp < 16 &&
        (L.wq.dtype == DType::Q8_0 || is_kquant_dtype(L.wq.dtype)) &&
        (L.wk.dtype == DType::Q8_0 || is_kquant_dtype(L.wk.dtype)) &&
        (L.wv.dtype == DType::Q8_0 || is_kquant_dtype(L.wv.dtype))) {
        const DeviceTensor * ws[3] = {&L.wq, &L.wk, &L.wv};
        float *              ys[3] = {rt.d_q, rt.d_k, rt.d_v};
        if (!try_launch_linear_multi_sm60_h2(ws, ys, 3, rt.d_x_norm, T_mtp, rt.d_scratch, rt.scratch_bytes, rt.stream)) {
            launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T_mtp, rt.stream);
            launch_linear_multi_prequant(ws, ys, 3, rt.d_scratch, T_mtp, rt.stream);
        }
    } else {
        launch_linear(rt.cublas, L.wq, rt.d_x_norm, rt.d_q, T_mtp, rt.d_scratch, rt.scratch_bytes, rt.stream);
        launch_linear(rt.cublas, L.wk, rt.d_x_norm, rt.d_k, T_mtp, rt.d_scratch, rt.scratch_bytes, rt.stream);
        launch_linear(rt.cublas, L.wv, rt.d_x_norm, rt.d_v, T_mtp, rt.d_scratch, rt.scratch_bytes, rt.stream);
    }

    launch_maxwell_qk_norm_rope_kv_write(
        rt.d_q, rt.d_k, rt.d_v,
        static_cast<const float *>(L.attn_q_norm.ptr),
        static_cast<const float *>(L.attn_k_norm.ptr),
        k_cache_mtp_, v_cache_mtp_, cfg_.kv_type,
        rt.d_mtp_pos, T_mtp, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim,
        cfg_.rms_eps, cfg_.rope_freq_base, rt.stream);

    launch_gqa_attn(rt.d_q, k_cache_mtp_, v_cache_mtp_, cfg_.kv_type,
                    rt.d_mtp_pos, rt.d_attn_out, rt.d_scratch, T_mtp, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim,
                    kq_scale, rt.stream, rt.cublas, rt.scratch_bytes, kv_len_max);

    launch_linear(rt.cublas, L.wo, rt.d_attn_out, rt.d_sub_out, T_mtp, rt.d_scratch, rt.scratch_bytes, rt.stream);

    // 3. FFN MoE sub-block: d_mtp_alt -> d_mtp_inpL
    if (T_mtp <= 16 && L.hc_ffn_fn.dtype == DType::BF16) {
        launch_hc_post_flat_rmsnorm_mix(
            rt.d_sub_out, rt.d_mtp_inpL, rt.d_post, rt.d_comb, rt.d_mtp_alt,
            L.hc_ffn_fn, rt.d_mixes, cfg_.n_embd, 24, T_mtp, cfg_.rms_eps, rt.stream);
    } else {
        launch_hc_post(rt.d_sub_out, rt.d_mtp_inpL, rt.d_post, rt.d_comb, rt.d_mtp_alt, cfg_.n_embd, T_mtp, rt.stream);
        launch_hc_flat_rmsnorm_mix(rt.d_mtp_alt, L.hc_ffn_fn, rt.d_mixes, hc_dim, 24, T_mtp, cfg_.rms_eps, rt.stream);
    }
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
    const bool use_q8_mtp_moe =
        (L.ffn_gate_exps.dtype == DType::Q8_0 && L.ffn_up_exps.dtype == DType::Q8_0 &&
         L.ffn_down_exps.dtype == DType::Q8_0 && L.ffn_gate_shexp.dtype == DType::Q8_0 &&
         L.ffn_up_shexp.dtype  == DType::Q8_0 && L.ffn_down_shexp.dtype == DType::Q8_0);
    if (use_q8_mtp_moe) {
        launch_moe_ffn_q8_0_with_shared(
            L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps,
            L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
            rt.d_x_norm, rt.d_top_ids, rt.d_top_w, rt.d_sub_out,
            rt.d_scratch, T_mtp, cfg_.n_expert_used, rt.stream);
    } else {
        launch_moe_ffn_kquant_with_shared(
            rt.cublas,
            L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps,
            L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
            rt.d_x_norm, rt.d_top_ids, rt.d_top_w, rt.d_sub_out,
            rt.d_scratch, rt.scratch_bytes, T_mtp, cfg_.n_expert_used, 0.0f, 0.0f, rt.stream);
    }

    launch_hc_post(rt.d_sub_out, rt.d_mtp_alt, rt.d_post, rt.d_comb, rt.d_mtp_inpL, cfg_.n_embd, T_mtp, rt.stream);

    // 4. MTP head collapse on last token -> rt.d_x_norm[0..n_embd)
    const float * last_h = rt.d_mtp_inpL + static_cast<size_t>(T_mtp - 1) * hc_dim;
    launch_hc_flat_rmsnorm_mix(last_h, mtp_.hc_head_fn, rt.d_mixes, hc_dim, 4, 1, cfg_.rms_eps, rt.stream);
    launch_hc_head_rmsnorm(
        last_h, rt.d_mixes,
        static_cast<const float *>(mtp_.hc_head_scale.ptr),
        static_cast<const float *>(mtp_.hc_head_base.ptr),
        static_cast<const float *>(mtp_.shared_head_norm.ptr),
        rt.d_x_norm, cfg_.n_embd, 1, cfg_.hc_eps, cfg_.rms_eps, rt.stream);

    if (head_gi_ + 1 == runtimes_.size()) {
        launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, rt.stream);
        launch_argmax_f32_batch(rt.d_logits, rt.d_argmax, cfg_.n_vocab, 1, rt.stream);
    }
}

// Run blk.45 MTP layer on last_dev for `T_mtp` (token, h_backbone) pairs and return greedy draft token for the last pair.
int32_t MaxwellMoeModel::run_mtp_on_last_gpu(const int32_t * h_tokens,
                                             const float * d_h_states,
                                             const int32_t * h_positions,
                                             int T_mtp) {
    auto & rt = runtimes_.back();
    cuda_ok(cudaSetDevice(rt.device), "mtp setDevice");

    cuda_ok(cudaMemcpyAsync(rt.d_mtp_tok, h_tokens,    T_mtp * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "mtp tok");
    cuda_ok(cudaMemcpyAsync(rt.d_mtp_pos, h_positions, T_mtp * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "mtp pos");

    if (use_cuda_graphs_ && d_h_states == rt.d_inpL && T_mtp == 1 && graph_mtp_t1_) {
        cuda_ok(cudaGraphLaunch(graph_mtp_t1_, rt.stream), "graph mtp t1");
    } else if (use_cuda_graphs_ && d_h_states == rt.d_inpL && T_mtp == 2 && graph_mtp_t2_) {
        cuda_ok(cudaGraphLaunch(graph_mtp_t2_, rt.stream), "graph mtp t2");
    } else {
        int kv_len_max = 0;
        for (int i = 0; i < T_mtp; ++i) if (h_positions[i] + 1 > kv_len_max) kv_len_max = h_positions[i] + 1;
        run_mtp_body_on_last_gpu(d_h_states, T_mtp, kv_len_max);
    }

    if (asym_mode_) {
        auto & head_rt = runtimes_[head_gi_];
        cuda_ok(cudaSetDevice(rt.device), "mtp head src");
        cuda_ok(cudaEventRecord(rt.done_event, rt.stream), "mtp head ev");
        cuda_ok(cudaSetDevice(head_rt.device), "mtp head dst");
        cuda_ok(cudaStreamWaitEvent(head_rt.stream, rt.done_event, 0), "mtp head wait");
        cuda_ok(cudaMemcpyPeerAsync(head_rt.d_x_norm, head_rt.device, rt.d_x_norm, rt.device,
                                    static_cast<size_t>(cfg_.n_embd) * sizeof(float), head_rt.stream), "mtp head peer");
        if (use_cuda_graphs_ && graph_head_t1_) {
            cuda_ok(cudaGraphLaunch(graph_head_t1_, head_rt.stream), "mtp graph head t1");
        } else {
            run_head_on_head_gpu(1, true);
        }
    }

    auto & head_rt = runtimes_[head_gi_];
    int32_t draft_tok = 0;
    cuda_ok(cudaSetDevice(head_rt.device), "mtp argmax setDevice");
    cuda_ok(cudaMemcpyAsync(&draft_tok, head_rt.d_argmax, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, head_rt.stream), "mtp argmax D2H");
    cuda_ok(cudaStreamSynchronize(head_rt.stream), "mtp sync");
    return draft_tok;
}

int32_t MaxwellMoeModel::mtp_prefill_chunk(const std::vector<int32_t> & next_tokens,
                                           const std::vector<int32_t> & positions) {
    if (!mtp_.loaded) return 0;
    const int T = static_cast<int>(next_tokens.size());
    if (T <= 0 || T > max_batch_ || positions.size() != next_tokens.size()) {
        throw std::runtime_error("mtp_prefill_chunk: invalid batch size");
    }
    std::vector<int32_t> mtp_pos(T);
    for (int i = 0; i < T; ++i) mtp_pos[i] = positions[i] + mtp_pos_offset_;
    return run_mtp_on_last_gpu(next_tokens.data(), runtimes_.back().d_inpL, mtp_pos.data(), T);
}

int32_t MaxwellMoeModel::mtp_prefill_and_draft(const std::vector<int32_t> & prompt, int32_t tok0) {
    if (!mtp_.loaded) return tok0;
    const int T = static_cast<int>(prompt.size());
    std::vector<int32_t> next_toks(T);
    std::vector<int32_t> mtp_pos(T);
    for (int i = 0; i + 1 < T; ++i) next_toks[i] = prompt[i + 1];
    next_toks[T - 1] = tok0;
    for (int i = 0; i < T; ++i) mtp_pos[i] = i + mtp_pos_offset_;
    return run_mtp_on_last_gpu(next_toks.data(), runtimes_.back().d_inpL, mtp_pos.data(), T);
}

void MaxwellMoeModel::decode_mtp_greedy_step(int32_t & cur_tok,
                                             int32_t & draft_tok,
                                             int32_t & cur_pos,
                                             std::vector<int32_t> & out_emitted,
                                             MtpStats & stats) {
    const int hc_dim = cfg_.hc_mult * cfg_.n_embd;

    if (use_cuda_graphs_ && !graphs_t1_ready_) capture_decode_graphs(1);
    if (use_cuda_graphs_ && !graphs_t2_ready_) capture_decode_graphs(2);

    int32_t toks2[2] = {cur_tok, draft_tok};
    int32_t pos2[2]  = {cur_pos, cur_pos + 1};

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "mtp_v setDevice");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, pos2, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "pos2");
        if (gi == embd_gi_) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, toks2, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok2");
        }
    }

    if (asym_mode_) {
        auto & rt_e = runtimes_[embd_gi_];
        auto & rt_0 = runtimes_[0];
        cuda_ok(cudaSetDevice(rt_e.device), "mtp_v embd setDevice");
        if (use_cuda_graphs_ && graph_embd_t2_) {
            cuda_ok(cudaGraphLaunch(graph_embd_t2_, rt_e.stream), "graph embd t2");
        } else {
            launch_embedding_lookup(tok_embd_, rt_e.d_tokens, rt_e.d_x_norm, cfg_.n_embd, 1, 2, rt_e.stream);
        }
        cuda_ok(cudaEventRecord(rt_e.done_event, rt_e.stream), "mtp_v embd ev");
        cuda_ok(cudaSetDevice(rt_0.device), "mtp_v wait embd");
        cuda_ok(cudaStreamWaitEvent(rt_0.stream, rt_e.done_event, 0), "mtp_v wait embd ev");
        cuda_ok(cudaMemcpyPeerAsync(rt_0.d_x_norm, rt_0.device, rt_e.d_x_norm, rt_e.device,
                                    static_cast<size_t>(2) * cfg_.n_embd * sizeof(float), rt_0.stream), "mtp_v embd peer");
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
            run_gpu_segment(gi, 2, true, true, cur_pos + 2);
        }
    }

    if (asym_mode_) {
        auto & last_rt = runtimes_.back();
        auto & head_rt = runtimes_[head_gi_];
        cuda_ok(cudaSetDevice(last_rt.device), "mtp_v head src");
        cuda_ok(cudaEventRecord(last_rt.done_event, last_rt.stream), "mtp_v head ev");
        cuda_ok(cudaSetDevice(head_rt.device), "mtp_v head dst");
        cuda_ok(cudaStreamWaitEvent(head_rt.stream, last_rt.done_event, 0), "mtp_v head wait");
        cuda_ok(cudaMemcpyPeerAsync(head_rt.d_x_norm, head_rt.device, last_rt.d_x_norm, last_rt.device,
                                    static_cast<size_t>(2) * cfg_.n_embd * sizeof(float), head_rt.stream), "mtp_v head peer");
        if (use_cuda_graphs_ && graph_head_t2_) {
            cuda_ok(cudaGraphLaunch(graph_head_t2_, head_rt.stream), "graph head t2");
        } else {
            run_head_on_head_gpu(2, true);
        }
    }

    auto & last_rt = runtimes_.back();
    auto & head_rt = runtimes_[head_gi_];
    int32_t pred[2] = {0, 0};
    cuda_ok(cudaSetDevice(head_rt.device), "mtp_v D2H");
    cuda_ok(cudaMemcpyAsync(pred, head_rt.d_argmax, 2 * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, head_rt.stream), "pred2");
    cuda_ok(cudaStreamSynchronize(head_rt.stream), "mtp_v sync");

    stats.steps += 1;

    if (pred[0] == draft_tok) {
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
