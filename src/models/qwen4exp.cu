#include "models/qwen4exp.h"
#include "kernels/kernels.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
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
    cuda_ok(cudaSetDevice(dev), "qwen4 dev_alloc setDevice");
    T * ptr = nullptr;
    cuda_ok(cudaMalloc(&ptr, count * sizeof(T)), "qwen4 cudaMalloc");
    return ptr;
}

inline float fp16_bits_to_f32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x03FFu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

std::vector<uint64_t> get_u64_vec(const ShardedGguf & sg, const std::string & key) {
    std::vector<uint64_t> out;
    if (!sg.has_kv(key)) return out;
    const auto & v = sg.kv(key);
    if (auto * a = std::get_if<std::vector<uint64_t>>(&v.data)) return *a;
    if (auto * a = std::get_if<std::vector<int64_t>>(&v.data)) {
        out.reserve(a->size());
        for (int64_t x : *a) out.push_back(static_cast<uint64_t>(x));
    }
    return out;
}

} // namespace

Qwen4ExpModel::~Qwen4ExpModel() {
    invalidate_graphs();
    for (size_t il = 0; il < layers_.size(); ++il) {
        cudaSetDevice(layers_[il].device);
        if (il < gdn_conv_.size()         && gdn_conv_[il])         cudaFree(gdn_conv_[il]);
        if (il < gdn_state_.size()        && gdn_state_[il])        cudaFree(gdn_state_[il]);
        if (il < ple_conv_.size()         && ple_conv_[il])         cudaFree(ple_conv_[il]);
        if (il < k_cache_.size()          && k_cache_[il])          cudaFree(k_cache_[il]);
        if (il < v_cache_.size()          && v_cache_[il])          cudaFree(v_cache_[il]);
        if (il < idx_k_raw_cache_.size()  && idx_k_raw_cache_[il])  cudaFree(idx_k_raw_cache_[il]);
        if (il < idx_k_pool_cache_.size() && idx_k_pool_cache_[il]) cudaFree(idx_k_pool_cache_[il]);
    }
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        if (rt.d_res_hc)     cudaFree(rt.d_res_hc);
        if (rt.d_hc_xn)      cudaFree(rt.d_hc_xn);
        if (rt.d_hc_lo)      cudaFree(rt.d_hc_lo);
        if (rt.d_hc_gate)    cudaFree(rt.d_hc_gate);
        if (rt.d_hc_inject)  cudaFree(rt.d_hc_inject);
        if (rt.d_cur)        cudaFree(rt.d_cur);
        if (rt.d_sub_out)    cudaFree(rt.d_sub_out);
        if (rt.d_ple_emb)    cudaFree(rt.d_ple_emb);
        if (rt.d_ple_key)    cudaFree(rt.d_ple_key);
        if (rt.d_ple_query)  cudaFree(rt.d_ple_query);
        if (rt.d_ple_val)    cudaFree(rt.d_ple_val);
        if (rt.d_ple_gated)  cudaFree(rt.d_ple_gated);
        if (rt.d_ple_norm)   cudaFree(rt.d_ple_norm);
        if (rt.d_qkv_raw)    cudaFree(rt.d_qkv_raw);
        if (rt.d_qkv_conv)   cudaFree(rt.d_qkv_conv);
        if (rt.d_z)          cudaFree(rt.d_z);
        if (rt.d_alpha)      cudaFree(rt.d_alpha);
        if (rt.d_beta)       cudaFree(rt.d_beta);
        if (rt.d_gdn_out)    cudaFree(rt.d_gdn_out);
        if (rt.d_q_full)     cudaFree(rt.d_q_full);
        if (rt.d_q)          cudaFree(rt.d_q);
        if (rt.d_gate)       cudaFree(rt.d_gate);
        if (rt.d_k)          cudaFree(rt.d_k);
        if (rt.d_v)          cudaFree(rt.d_v);
        if (rt.d_attn_out)   cudaFree(rt.d_attn_out);
        if (rt.d_idx_q)      cudaFree(rt.d_idx_q);
        if (rt.d_idx_k)      cudaFree(rt.d_idx_k);
        if (rt.d_sel_pos)    cudaFree(rt.d_sel_pos);
        if (rt.d_sel_len)    cudaFree(rt.d_sel_len);
        if (rt.d_router)     cudaFree(rt.d_router);
        if (rt.d_shexp_gate) cudaFree(rt.d_shexp_gate);
        if (rt.d_top_ids)    cudaFree(rt.d_top_ids);
        if (rt.d_top_w)      cudaFree(rt.d_top_w);
        if (rt.d_tokens)     cudaFree(rt.d_tokens);
        if (rt.d_pos)        cudaFree(rt.d_pos);
        if (rt.d_logits)     cudaFree(rt.d_logits);
        if (rt.d_argmax)     cudaFree(rt.d_argmax);
        if (rt.d_scratch)    cudaFree(rt.d_scratch);
        if (rt.cublas)       cublasDestroy(rt.cublas);
        if (rt.done_event)   cudaEventDestroy(rt.done_event);
        if (rt.stream)       cudaStreamDestroy(rt.stream);
    }
    pool_.free_all();
}

void Qwen4ExpModel::invalidate_graphs() noexcept {
    for (auto & rt : runtimes_) {
        if (rt.decode_graph_t1) {
            cudaSetDevice(rt.device);
            cudaGraphExecDestroy(rt.decode_graph_t1);
            rt.decode_graph_t1 = nullptr;
        }
    }
    graphs_ready_ = false;
}

void Qwen4ExpModel::load(const std::string & gguf_path,
                         const std::vector<int> & gpu_ids,
                         const std::vector<float> & split_weights,
                         int max_ctx,
                         bool kv_q8_0) {
    if (gpu_ids.empty()) throw std::runtime_error("Qwen4ExpModel::load requires >= 1 GPU");

    const auto t0 = std::chrono::steady_clock::now();
    sg_ = ShardedGguf::open(gguf_path);
    const std::string arch = sg_.get_str("general.architecture", "qwen4exp");
    const std::string pfx  = arch + ".";

    cfg_.n_layers             = static_cast<int>(sg_.get_u64(pfx + "block_count", 48));
    cfg_.n_embd               = static_cast<int>(sg_.get_u64(pfx + "embedding_length", 2560));
    cfg_.n_head               = static_cast<int>(sg_.get_u64(pfx + "attention.head_count", 24));
    cfg_.n_head_kv            = static_cast<int>(sg_.get_u64(pfx + "attention.head_count_kv", 2));
    cfg_.head_dim             = static_cast<int>(sg_.get_u64(pfx + "attention.key_length", 256));
    cfg_.rope_dim             = static_cast<int>(sg_.get_u64(pfx + "rope.dimension_count", 64));
    cfg_.rope_freq_base       = static_cast<float>(sg_.get_f64(pfx + "rope.freq_base", 10000000.0));
    cfg_.rms_eps              = static_cast<float>(sg_.get_f64(pfx + "attention.layer_norm_rms_epsilon", 1e-6));

    cfg_.ssm_d_conv           = static_cast<int>(sg_.get_u64(pfx + "ssm.conv_kernel", 4));
    cfg_.ssm_d_state          = static_cast<int>(sg_.get_u64(pfx + "ssm.state_size", 128));
    cfg_.ssm_n_group          = static_cast<int>(sg_.get_u64(pfx + "ssm.group_count", 16));
    cfg_.ssm_dt_rank          = static_cast<int>(sg_.get_u64(pfx + "ssm.time_step_rank", 48));
    cfg_.ssm_d_inner          = static_cast<int>(sg_.get_u64(pfx + "ssm.inner_size", 6144));
    cfg_.full_attn_interval   = static_cast<int>(sg_.get_u64(pfx + "full_attention_interval", 4));

    cfg_.n_expert             = static_cast<int>(sg_.get_u64(pfx + "expert_count", 512));
    cfg_.n_expert_used        = static_cast<int>(sg_.get_u64(pfx + "expert_used_count", 10));
    cfg_.n_ff_exp             = static_cast<int>(sg_.get_u64(pfx + "expert_feed_forward_length", 640));
    cfg_.n_ff_shexp           = static_cast<int>(sg_.get_u64(pfx + "expert_shared_feed_forward_length", 640));
    cfg_.expert_weights_scale = static_cast<float>(sg_.get_f64(pfx + "expert_weights_scale", 1.0));

    cfg_.hc_mult              = static_cast<int>(sg_.get_u64(pfx + "hyper_connection.count", 4));
    cfg_.hc_low_rank          = static_cast<int>(sg_.get_u64(pfx + "hyper_connection.low_rank", 320));

    cfg_.idx_n_head           = static_cast<int>(sg_.get_u64(pfx + "attention.indexer.head_count", 4));
    cfg_.idx_head_dim         = static_cast<int>(sg_.get_u64(pfx + "attention.indexer.key_length", 128));
    cfg_.idx_top_k            = static_cast<int>(sg_.get_u64(pfx + "attention.indexer.top_k", 2048));
    cfg_.idx_kpool            = 4;
    const auto cr = get_u64_vec(sg_, pfx + "attention.compress_ratios");
    for (uint64_t r : cr) if (r > 0) { cfg_.idx_kpool = static_cast<int>(r); break; }

    cfg_.ple_ngram_size       = static_cast<int>(sg_.get_u64(pfx + "ple.ngram_size", 3));
    cfg_.ple_heads_per_ngram  = static_cast<int>(sg_.get_u64(pfx + "ple.heads_per_ngram", 8));
    cfg_.ple_n_heads          = (cfg_.ple_ngram_size - 1) * cfg_.ple_heads_per_ngram;
    cfg_.ple_head_dim         = static_cast<int>(sg_.get_u64(pfx + "embedding_length_per_layer_input", 160));
    cfg_.ple_conv_kernel      = static_cast<int>(sg_.get_u64(pfx + "ple.conv_kernel", 4));
    cfg_.ple_eos_token_id     = static_cast<int32_t>(sg_.get_u64(pfx + "ple.eos_token_id", 248044));
    cfg_.ple_multipliers      = get_u64_vec(sg_, pfx + "ple.layer_multipliers");
    cfg_.ple_head_offsets     = get_u64_vec(sg_, pfx + "ple.head_offsets");
    cfg_.ple_head_vocab_sizes = get_u64_vec(sg_, pfx + "ple.head_vocab_sizes");

    cfg_.max_ctx              = max_ctx;
    cfg_.kv_q8_0              = kv_q8_0;

    auto emb_loc = sg_.find_tensor("token_embd.weight");
    if (!emb_loc.info) throw std::runtime_error("token_embd.weight missing");
    cfg_.n_vocab = static_cast<int>(emb_loc.info->ne[1]);

    // Map per_layer_token_embd.weight directly in host RAM (ZERO GPU VRAM!)
    auto ple_loc = sg_.find_tensor("per_layer_token_embd.weight");
    if (ple_loc.info) {
        if (ple_loc.info->dtype != DType::Q8_0) {
            throw std::runtime_error("Qwen4ExpModel: per_layer_token_embd.weight must be Q8_0");
        }
        ple_tbl_bytes_ = static_cast<const uint8_t *>(ple_loc.data());
        ple_row_bytes_ = static_cast<size_t>(cfg_.ple_head_dim / 32) * sizeof(BlockQ8_0);
        auto ple_sc = sg_.find_tensor("per_layer_token_embd.scale");
        if (ple_sc.info) {
            std::memcpy(&ple_scale_, ple_sc.data(), sizeof(float));
        }
    }

    cfg_.is_recr.assign(cfg_.n_layers, false);
    for (int il = 0; il < cfg_.n_layers; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        cfg_.is_recr[il] = (sg_.find_tensor(p + "ssm_conv1d.weight").info != nullptr);
    }

    cfg_.is_ple.assign(cfg_.n_layers, false);
    const auto ple_layers = get_u64_vec(sg_, pfx + "ple.layers");
    for (uint64_t l : ple_layers) {
        if (l < static_cast<uint64_t>(cfg_.n_layers)) cfg_.is_ple[l] = true;
    }

    // Partition layers across GPUs proportional to split_weights
    std::vector<float> sw = split_weights;
    if (sw.size() != gpu_ids.size()) {
        sw.assign(gpu_ids.size(), 1.0f);
        for (size_t g = 0; g < gpu_ids.size(); ++g) {
            size_t free_b = 0, tot_b = 0;
            cudaSetDevice(gpu_ids[g]);
            cudaMemGetInfo(&free_b, &tot_b);
            sw[g] = static_cast<float>(free_b);
        }
    }
    float sw_sum = 0.0f;
    for (float v : sw) sw_sum += v;

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
        for (int il = cur_l; il < cur_l + count; ++il) {
            if (cfg_.is_ple[il]) runtimes_[g].has_ple_layer = true;
        }
        cur_l += count;
    }

    for (size_t g = 0; g + 1 < runtimes_.size(); ++g) {
        int can = 0;
        cudaDeviceCanAccessPeer(&can, runtimes_[g + 1].device, runtimes_[g].device);
        if (can) { cudaSetDevice(runtimes_[g + 1].device); cudaDeviceEnablePeerAccess(runtimes_[g].device, 0); cudaGetLastError(); }
    }

    layers_.resize(cfg_.n_layers);
    gdn_conv_.assign(cfg_.n_layers, nullptr);
    gdn_state_.assign(cfg_.n_layers, nullptr);
    ple_conv_.assign(cfg_.n_layers, nullptr);
    k_cache_.assign(cfg_.n_layers, nullptr);
    v_cache_.assign(cfg_.n_layers, nullptr);
    idx_k_raw_cache_.assign(cfg_.n_layers, nullptr);
    idx_k_pool_cache_.assign(cfg_.n_layers, nullptr);

    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner; // 10240
    const int d_kv     = cfg_.n_head_kv * cfg_.head_dim;                             // 512
    const int hc_dim   = cfg_.hc_mult * cfg_.n_embd;                                 // 10240
    const int ple_hist = (cfg_.ple_conv_kernel - 1) * cfg_.ple_ngram_size;           // 9
    const size_t kv_bytes = kv_q8_0
        ? static_cast<size_t>(cfg_.max_ctx) * (d_kv / 32) * sizeof(BlockQ8_0)
        : static_cast<size_t>(cfg_.max_ctx) * d_kv * sizeof(uint16_t);
    const size_t idx_raw_n  = static_cast<size_t>(cfg_.max_ctx) * cfg_.idx_head_dim;
    const size_t idx_pool_n = static_cast<size_t>((cfg_.max_ctx + cfg_.idx_kpool - 1) / cfg_.idx_kpool) * cfg_.idx_head_dim;

    // Parallel multi-GPU weight upload
    std::vector<std::thread> load_threads;
    std::vector<std::exception_ptr> load_errs(runtimes_.size(), nullptr);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        load_threads.emplace_back([&, gi]() {
            try {
                const int dev = runtimes_[gi].device;
                cuda_ok(cudaSetDevice(dev), "qwen4 load setDevice");
                if (gi == 0) {
                    tok_embd_ = pool_.upload(sg_, "token_embd.weight", dev, false);
                }
                if (gi + 1 == runtimes_.size()) {
                    output_hc_norm_ = pool_.upload(sg_, "output_hc_norm.weight", dev, false);
                    output_hc_down_ = pool_.upload(sg_, "output_hc_down.weight", dev, false);
                    output_hc_up_   = pool_.upload(sg_, "output_hc_up.weight",   dev, false);
                    output_         = pool_.upload(sg_, "output.weight",         dev, false, false);
                    if (!output_.valid()) output_ = pool_.upload(sg_, "token_embd.weight", dev, false);
                }
                for (int il = runtimes_[gi].layer_start; il < runtimes_[gi].layer_end; ++il) {
                    auto & L = layers_[il];
                    const std::string p = "blk." + std::to_string(il) + ".";
                    L.device  = dev;
                    L.is_recr = cfg_.is_recr[il];
                    L.is_ple  = cfg_.is_ple[il];

                    L.hc_attn_norm   = pool_.upload(sg_, p + "hc_attn_norm.weight",   dev, false);
                    L.hc_attn_down   = pool_.upload(sg_, p + "hc_attn_down.weight",   dev, false);
                    L.hc_attn_up     = pool_.upload(sg_, p + "hc_attn_up.weight",     dev, false);
                    L.hc_attn_inject = pool_.upload(sg_, p + "hc_attn_inject.weight", dev, false);

                    L.hc_ffn_norm    = pool_.upload(sg_, p + "hc_ffn_norm.weight",    dev, false);
                    L.hc_ffn_down    = pool_.upload(sg_, p + "hc_ffn_down.weight",    dev, false);
                    L.hc_ffn_up      = pool_.upload(sg_, p + "hc_ffn_up.weight",      dev, false);
                    L.hc_ffn_inject  = pool_.upload(sg_, p + "hc_ffn_inject.weight",  dev, false);

                    if (L.is_ple) {
                        L.ple_key        = pool_.upload(sg_, p + "ple_key.weight",        dev, false);
                        L.ple_value      = pool_.upload(sg_, p + "ple_value.weight",      dev, false);
                        L.ple_norm_key   = pool_.upload(sg_, p + "ple_norm_key.weight",   dev, false);
                        L.ple_norm_query = pool_.upload(sg_, p + "ple_norm_query.weight", dev, false);
                        L.ple_norm_conv  = pool_.upload(sg_, p + "ple_norm_conv.weight",  dev, false);
                        L.ple_conv1d     = pool_.upload(sg_, p + "ple_conv1d.weight",     dev, false);
                        ple_conv_[il]    = dev_alloc<float>(dev, static_cast<size_t>(ple_hist) * hc_dim);
                    }

                    if (L.is_recr) {
                        L.wqkv       = pool_.upload(sg_, p + "attn_qkv.weight",   dev, false);
                        L.wqkv_gate  = pool_.upload(sg_, p + "attn_gate.weight",  dev, false);
                        L.ssm_conv1d = pool_.upload(sg_, p + "ssm_conv1d.weight", dev, false);
                        L.ssm_dt     = pool_.upload(sg_, p + "ssm_dt.bias",       dev, false);
                        L.ssm_a      = pool_.upload(sg_, p + "ssm_a",             dev, false);
                        L.ssm_alpha  = pool_.upload(sg_, p + "ssm_alpha.weight",  dev, false);
                        L.ssm_beta   = pool_.upload(sg_, p + "ssm_beta.weight",   dev, false);
                        L.ssm_norm   = pool_.upload(sg_, p + "ssm_norm.weight",   dev, false);
                        L.ssm_out    = pool_.upload(sg_, p + "ssm_out.weight",    dev, false);

                        gdn_conv_[il]  = dev_alloc<float>(dev, static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim);
                        gdn_state_[il] = dev_alloc<float>(dev, static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state);
                    } else {
                        L.wq          = pool_.upload(sg_, p + "attn_q.weight",        dev, false);
                        L.wk          = pool_.upload(sg_, p + "attn_k.weight",        dev, false);
                        L.wv          = pool_.upload(sg_, p + "attn_v.weight",        dev, false);
                        L.wo          = pool_.upload(sg_, p + "attn_output.weight",   dev, false);
                        L.attn_q_norm = pool_.upload(sg_, p + "attn_q_norm.weight",   dev, false);
                        L.attn_k_norm = pool_.upload(sg_, p + "attn_k_norm.weight",   dev, false);

                        // Keep BF16 exact (gemv_bf16_kernel works on both sm_70 & sm_86!)
                        L.idx_q_proj  = pool_.upload(sg_, p + "indexer.q_proj.weight", dev, false);
                        L.idx_k_proj  = pool_.upload(sg_, p + "indexer.k_proj.weight", dev, false);
                        L.idx_q_norm  = pool_.upload(sg_, p + "indexer.q_norm.weight", dev, false);
                        L.idx_k_norm  = pool_.upload(sg_, p + "indexer.k_norm.weight", dev, false);

                        cuda_ok(cudaMalloc(&k_cache_[il], kv_bytes), "qwen4 malloc k_cache");
                        cuda_ok(cudaMalloc(&v_cache_[il], kv_bytes), "qwen4 malloc v_cache");
                        idx_k_raw_cache_[il]  = dev_alloc<float>(dev, idx_raw_n);
                        idx_k_pool_cache_[il] = dev_alloc<float>(dev, idx_pool_n);
                    }

                    L.ffn_gate_inp       = pool_.upload(sg_, p + "ffn_gate_inp.weight",       dev, false);
                    L.ffn_gate_exps      = pool_.upload(sg_, p + "ffn_gate_exps.weight",      dev, false);
                    L.ffn_up_exps        = pool_.upload(sg_, p + "ffn_up_exps.weight",        dev, false);
                    L.ffn_down_exps      = pool_.upload(sg_, p + "ffn_down_exps.weight",      dev, false);
                    L.ffn_gate_inp_shexp = pool_.upload(sg_, p + "ffn_gate_inp_shexp.weight", dev, false);
                    L.ffn_gate_shexp     = pool_.upload(sg_, p + "ffn_gate_shexp.weight",     dev, false);
                    L.ffn_up_shexp       = pool_.upload(sg_, p + "ffn_up_shexp.weight",       dev, false);
                    L.ffn_down_shexp     = pool_.upload(sg_, p + "ffn_down_shexp.weight",     dev, false);
                }
                cuda_ok(cudaDeviceSynchronize(), "qwen4 load sync");
            } catch (...) {
                load_errs[gi] = std::current_exception();
            }
        });
    }
    for (auto & th : load_threads) th.join();
    for (auto & ep : load_errs) if (ep) std::rethrow_exception(ep);
    pool_.release_staging();

    // Pre-warm mmap'd host-RAM PLE n-gram table sequentially into OS page cache so decode has ZERO disk page faults!
    if (ple_tbl_bytes_ && ple_row_bytes_ > 0) {
        auto ple_loc = sg_.find_tensor("per_layer_token_embd.weight");
        if (ple_loc.info && ple_loc.info->nbytes > 0) {
            const size_t total_b = ple_loc.info->nbytes;
            const uintptr_t page_start = reinterpret_cast<uintptr_t>(ple_tbl_bytes_) & ~uintptr_t(4095);
            const size_t page_len = (reinterpret_cast<uintptr_t>(ple_tbl_bytes_) - page_start) + total_b;
            madvise(reinterpret_cast<void *>(page_start), page_len, MADV_SEQUENTIAL);
            volatile uint64_t sink = 0;
            uint64_t acc = 0;
            for (size_t off = 0; off < total_b; off += 4096) acc += ple_tbl_bytes_[off];
            sink = acc;
            (void)sink;
            madvise(reinterpret_cast<void *>(page_start), page_len, MADV_RANDOM);
        }
    }

    // Per-GPU runtime activation buffers
    const size_t B = static_cast<size_t>(max_batch_);
    const size_t H = static_cast<size_t>(cfg_.n_embd);
    const size_t d_q = static_cast<size_t>(cfg_.n_head) * cfg_.head_dim;
    const size_t max_sel = static_cast<size_t>(cfg_.idx_top_k + cfg_.idx_kpool);
    const size_t scratch_b = 128ULL * 1024 * 1024;

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        const int dev = rt.device;
        cuda_ok(cudaSetDevice(dev), "qwen4 rt setDevice");
        cuda_ok(cudaStreamCreate(&rt.stream), "qwen4 stream");
        cuda_ok(cudaEventCreateWithFlags(&rt.done_event, cudaEventDisableTiming), "qwen4 event");
        if (cublasCreate(&rt.cublas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        if (device_has_tensor_cores(dev)) cublasSetMathMode(rt.cublas, CUBLAS_TENSOR_OP_MATH);

        rt.d_res_hc     = dev_alloc<float>(dev, B * hc_dim);
        rt.d_hc_xn      = dev_alloc<float>(dev, B * hc_dim);
        rt.d_hc_lo      = dev_alloc<float>(dev, B * cfg_.hc_low_rank);
        rt.d_hc_gate    = dev_alloc<float>(dev, B * hc_dim);
        rt.d_hc_inject  = dev_alloc<float>(dev, B * cfg_.hc_mult);
        rt.d_cur        = dev_alloc<float>(dev, B * H);
        rt.d_sub_out    = dev_alloc<float>(dev, B * H);

        if (rt.has_ple_layer) {
            rt.d_ple_emb   = dev_alloc<float>(dev, B * cfg_.ple_n_heads * cfg_.ple_head_dim);
            rt.d_ple_key   = dev_alloc<float>(dev, B * hc_dim);
            rt.d_ple_query = dev_alloc<float>(dev, B * hc_dim);
            rt.d_ple_val   = dev_alloc<float>(dev, B * H);
            rt.d_ple_gated = dev_alloc<float>(dev, B * hc_dim);
            rt.d_ple_norm  = dev_alloc<float>(dev, B * hc_dim);
        }

        rt.d_qkv_raw    = dev_alloc<float>(dev, B * conv_dim);
        rt.d_qkv_conv   = dev_alloc<float>(dev, B * conv_dim);
        rt.d_z          = dev_alloc<float>(dev, B * cfg_.ssm_d_inner);
        rt.d_alpha      = dev_alloc<float>(dev, B * cfg_.ssm_dt_rank);
        rt.d_beta       = dev_alloc<float>(dev, B * cfg_.ssm_dt_rank);
        rt.d_gdn_out    = dev_alloc<float>(dev, B * cfg_.ssm_d_inner);

        rt.d_q_full     = dev_alloc<float>(dev, B * d_q * 2);
        rt.d_q          = dev_alloc<float>(dev, B * d_q);
        rt.d_gate       = dev_alloc<float>(dev, B * d_q);
        rt.d_k          = dev_alloc<float>(dev, B * d_kv);
        rt.d_v          = dev_alloc<float>(dev, B * d_kv);
        rt.d_attn_out   = dev_alloc<float>(dev, B * d_q);
        rt.d_idx_q      = dev_alloc<float>(dev, B * cfg_.idx_n_head * cfg_.idx_head_dim);
        rt.d_idx_k      = dev_alloc<float>(dev, B * cfg_.idx_head_dim);
        rt.d_sel_pos    = dev_alloc<int32_t>(dev, B * max_sel);
        rt.d_sel_len    = dev_alloc<int32_t>(dev, B);

        rt.d_router     = dev_alloc<float>(dev, B * cfg_.n_expert);
        rt.d_shexp_gate = dev_alloc<float>(dev, B);
        rt.d_top_ids    = dev_alloc<int32_t>(dev, B * cfg_.n_expert_used);
        rt.d_top_w      = dev_alloc<float>(dev, B * cfg_.n_expert_used);
        rt.d_pos        = dev_alloc<int32_t>(dev, B);

        if (gi == 0) rt.d_tokens = dev_alloc<int32_t>(dev, B);
        if (gi + 1 == runtimes_.size()) {
            rt.d_logits = dev_alloc<float>(dev, cfg_.n_vocab);
            rt.d_argmax = dev_alloc<int32_t>(dev, 1);
        }
        cuda_ok(cudaMalloc(&rt.d_scratch, scratch_b), "qwen4 scratch");
        rt.scratch_bytes = scratch_b;
    }

    token_history_.assign(cfg_.max_ctx, -1);
    reset_state();

    const auto t1 = std::chrono::steady_clock::now();
    std::printf("[Qwen4Exp] Loaded %s (%d layers, hc=%d, QSA top_k=%d, PLE host-RAM=%s) in %.2f s across %zu GPU(s):",
                arch.c_str(), cfg_.n_layers, cfg_.hc_mult, cfg_.idx_top_k,
                ple_tbl_bytes_ ? "mmap-Q8_0" : "none",
                std::chrono::duration<double>(t1 - t0).count(), runtimes_.size());
    for (const auto & rt : runtimes_) {
        std::printf(" GPU%d[%d..%d](%.2f GiB)", rt.device, rt.layer_start, rt.layer_end - 1,
                    pool_.bytes_on_device(rt.device) / (1024.0 * 1024.0 * 1024.0));
    }
    std::printf(" | ctx=%d (KV=%s)\n", cfg_.max_ctx, kv_q8_0 ? "Q8_0" : "F16");
}

void Qwen4ExpModel::reset_state() {
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const int hc_dim   = cfg_.hc_mult * cfg_.n_embd;
    const int ple_hist = (cfg_.ple_conv_kernel - 1) * cfg_.ple_ngram_size;
    const size_t conv_b = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim * sizeof(float);
    const size_t ssm_b  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state * sizeof(float);
    const size_t ple_b  = static_cast<size_t>(ple_hist) * hc_dim * sizeof(float);

    for (int il = 0; il < cfg_.n_layers; ++il) {
        cuda_ok(cudaSetDevice(layers_[il].device), "qwen4 reset setDevice");
        if (layers_[il].is_recr) {
            cuda_ok(cudaMemset(gdn_conv_[il],  0, conv_b), "qwen4 zero gdn_conv");
            cuda_ok(cudaMemset(gdn_state_[il], 0, ssm_b),  "qwen4 zero gdn_state");
        }
        if (layers_[il].is_ple && ple_conv_[il]) {
            cuda_ok(cudaMemset(ple_conv_[il], 0, ple_b), "qwen4 zero ple_conv");
        }
    }
    std::fill(token_history_.begin(), token_history_.end(), -1);
}

void Qwen4ExpModel::save_checkpoint(StateCheckpoint & out, int pos) const {
    out.pos = pos;
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const int hc_dim   = cfg_.hc_mult * cfg_.n_embd;
    const int ple_hist = (cfg_.ple_conv_kernel - 1) * cfg_.ple_ngram_size;
    const size_t conv_n = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim;
    const size_t ssm_n  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state;
    const size_t ple_n  = static_cast<size_t>(ple_hist) * hc_dim;

    out.gdn_conv.resize(cfg_.n_layers);
    out.gdn_ssm.resize(cfg_.n_layers);
    out.ple_conv.resize(cfg_.n_layers);
    for (int il = 0; il < cfg_.n_layers; ++il) {
        cuda_ok(cudaSetDevice(layers_[il].device), "qwen4 save setDevice");
        if (layers_[il].is_recr) {
            out.gdn_conv[il].resize(conv_n);
            out.gdn_ssm[il].resize(ssm_n);
            cuda_ok(cudaMemcpy(out.gdn_conv[il].data(), gdn_conv_[il],  conv_n * sizeof(float), cudaMemcpyDeviceToHost), "save conv");
            cuda_ok(cudaMemcpy(out.gdn_ssm[il].data(),  gdn_state_[il], ssm_n  * sizeof(float), cudaMemcpyDeviceToHost), "save ssm");
        }
        if (layers_[il].is_ple && ple_conv_[il]) {
            out.ple_conv[il].resize(ple_n);
            cuda_ok(cudaMemcpy(out.ple_conv[il].data(), ple_conv_[il], ple_n * sizeof(float), cudaMemcpyDeviceToHost), "save ple_conv");
        }
    }
    out.prev_tokens.assign(token_history_.begin(),
                           token_history_.begin() + std::min<size_t>(static_cast<size_t>(pos), token_history_.size()));
}

void Qwen4ExpModel::restore_checkpoint(const StateCheckpoint & cp) {
    for (int il = 0; il < cfg_.n_layers; ++il) {
        cuda_ok(cudaSetDevice(layers_[il].device), "qwen4 restore setDevice");
        if (layers_[il].is_recr && !cp.gdn_conv[il].empty()) {
            cuda_ok(cudaMemcpy(gdn_conv_[il],  cp.gdn_conv[il].data(), cp.gdn_conv[il].size() * sizeof(float), cudaMemcpyHostToDevice), "rest conv");
            cuda_ok(cudaMemcpy(gdn_state_[il], cp.gdn_ssm[il].data(),  cp.gdn_ssm[il].size()  * sizeof(float), cudaMemcpyHostToDevice), "rest ssm");
        }
        if (layers_[il].is_ple && ple_conv_[il] && !cp.ple_conv[il].empty()) {
            cuda_ok(cudaMemcpy(ple_conv_[il], cp.ple_conv[il].data(), cp.ple_conv[il].size() * sizeof(float), cudaMemcpyHostToDevice), "rest ple_conv");
        }
    }
    std::fill(token_history_.begin(), token_history_.end(), -1);
    const size_t n_copy = std::min(cp.prev_tokens.size(), token_history_.size());
    std::copy_n(cp.prev_tokens.begin(), n_copy, token_history_.begin());
}

void Qwen4ExpModel::compute_ple_host(const int32_t * tokens, const int32_t * positions, int T, std::vector<float> & out_emb) {
    if (!ple_tbl_bytes_) return;
    const int n_gram   = cfg_.ple_ngram_size;
    const int n_heads  = cfg_.ple_n_heads;
    const int per_gram = cfg_.ple_heads_per_ngram;
    const int hdim     = cfg_.ple_head_dim;
    const int n_blks   = hdim / 32;
    const int32_t eos  = cfg_.ple_eos_token_id;
    const int emb_dim  = n_heads * hdim;

    out_emb.resize(static_cast<size_t>(T) * emb_dim);

    // Record tokens into history first so within-batch predecessors are visible (matches llama.cpp!)
    for (int i = 0; i < T; ++i) {
        const int p = positions[i];
        if (p >= 0 && p < cfg_.max_ctx) token_history_[p] = tokens[i];
    }

    int64_t ctx[8];
    for (int i = 0; i < T; ++i) {
        const int p = positions[i];
        ctx[0] = tokens[i];
        bool cut = false;
        for (int s = 1; s < n_gram; ++s) {
            const int32_t t = (cut || p - s < 0 || p - s >= cfg_.max_ctx) ? -1 : token_history_[p - s];
            cut = cut || (t < 0) || (t == eos);
            ctx[s] = cut ? eos : t;
        }

        float * dst_tok = out_emb.data() + static_cast<size_t>(i) * emb_dim;
        for (int n = 2; n <= n_gram; ++n) {
            uint64_t mixed = static_cast<uint64_t>(ctx[0]) * cfg_.ple_multipliers[0];
            for (int j = 1; j < n; ++j) {
                mixed ^= static_cast<uint64_t>(ctx[j]) * cfg_.ple_multipliers[j];
            }
            const int base = (n - 2) * per_gram;
            for (int g = 0; g < per_gram; ++g) {
                const int h_i = base + g;
                const uint64_t row = (mixed % cfg_.ple_head_vocab_sizes[h_i]) + cfg_.ple_head_offsets[h_i];
                const auto * blks = reinterpret_cast<const BlockQ8_0 *>(ple_tbl_bytes_ + row * ple_row_bytes_);
                float * dst_h = dst_tok + static_cast<size_t>(h_i) * hdim;
                for (int b = 0; b < n_blks; ++b) {
                    const float d = fp16_bits_to_f32(blks[b].d) * ple_scale_;
                    const int8_t * qs = blks[b].qs;
                    float * out_b = dst_h + b * 32;
                    for (int k = 0; k < 32; ++k) out_b[k] = d * static_cast<float>(qs[k]);
                }
            }
        }
    }
}

void Qwen4ExpModel::run_hc_mix(DevRuntime & rt,
                               const DeviceTensor & w_norm,
                               const DeviceTensor & w_down,
                               const DeviceTensor & w_up,
                               const DeviceTensor & w_inject,
                               int T) {
    const int H      = cfg_.n_embd;
    const int hc     = cfg_.hc_mult;
    const float inv_hc = 1.0f / static_cast<float>(hc);
    cudaStream_t st = rt.stream;

    // 1. Grouped RMSNorm per stream -> d_hc_xn [T, hc*H]
    launch_hc_grouped_rmsnorm(rt.d_res_hc, static_cast<const float *>(w_norm.ptr), rt.d_hc_xn, H, hc, T, cfg_.rms_eps, st);

    // 2. Low-rank bottleneck: lo = SiLU((w_down @ xn) * (1/hc)), gate = w_up @ lo
    launch_linear(rt.cublas, w_down, rt.d_hc_xn, rt.d_hc_lo, T, rt.d_scratch, rt.scratch_bytes, st);
    launch_scale_silu_inplace(rt.d_hc_lo, inv_hc, static_cast<int64_t>(T) * cfg_.hc_low_rank, st);
    launch_linear(rt.cublas, w_up, rt.d_hc_lo, rt.d_hc_gate, T, rt.d_scratch, rt.scratch_bytes, st);

    // 3. Mixed input: cur[t, d] = (1/hc) * sum_s xn[t, s, d] * sigmoid(gate[t, s, d])
    launch_hc_pre_gated_mean(rt.d_hc_xn, rt.d_hc_gate, rt.d_cur, H, hc, T, st);

    // 4. Optional scatter injection weights (scaled by 1/hc inside launch_hc_combine!)
    if (w_inject.valid()) {
        launch_linear(rt.cublas, w_inject, rt.d_hc_xn, rt.d_hc_inject, T, rt.d_scratch, rt.scratch_bytes, st);
    }
}

void Qwen4ExpModel::run_gpu_segment(size_t gi, int T, bool run_argmax, int kv_len_max) {
    auto & rt = runtimes_[gi];
    cudaSetDevice(rt.device);
    cudaStream_t st = rt.stream;

    const int H          = cfg_.n_embd;
    const int hc         = cfg_.hc_mult;
    const int hc_dim     = hc * H;
    const float inv_hc   = 1.0f / static_cast<float>(hc);
    const int conv_dim   = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const int d_q        = cfg_.n_head * cfg_.head_dim;
    const int max_sel    = cfg_.idx_top_k + cfg_.idx_kpool;
    const float attn_sc  = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
    const float gdn_sc   = 1.0f / std::sqrt(static_cast<float>(cfg_.ssm_d_state));

    if (gi == 0) {
        launch_embedding_lookup(tok_embd_, rt.d_tokens, rt.d_cur, H, 1, T, st);
        launch_hc_init(rt.d_cur, rt.d_res_hc, H, hc, T, st);
    }

    for (int il = rt.layer_start; il < rt.layer_end; ++il) {
        const auto & L = layers_[il];

        // 0. Optional PLE n-gram Engram (blk.1) BEFORE hc_attn!
        if (L.is_ple && ple_tbl_bytes_) {
            launch_linear(rt.cublas, L.ple_key,   rt.d_ple_emb, rt.d_ple_key, T, rt.d_scratch, rt.scratch_bytes, st);
            launch_linear(rt.cublas, L.ple_value, rt.d_ple_emb, rt.d_ple_val, T, rt.d_scratch, rt.scratch_bytes, st);
            launch_hc_grouped_rmsnorm(rt.d_ple_key, static_cast<const float *>(L.ple_norm_key.ptr),   rt.d_ple_key,   H, hc, T, cfg_.rms_eps, st);
            launch_hc_grouped_rmsnorm(rt.d_res_hc,  static_cast<const float *>(L.ple_norm_query.ptr), rt.d_ple_query, H, hc, T, cfg_.rms_eps, st);
            launch_ple_gate_value(rt.d_ple_key, rt.d_ple_query, rt.d_ple_val, rt.d_ple_gated, H, hc, T, st);
            launch_hc_grouped_rmsnorm(rt.d_ple_gated, static_cast<const float *>(L.ple_norm_conv.ptr), rt.d_ple_norm, H, hc, T, cfg_.rms_eps, st);
            launch_ple_dilated_conv1d_silu_add(rt.d_ple_norm, static_cast<const float *>(L.ple_conv1d.ptr),
                                               ple_conv_[il], rt.d_ple_gated, rt.d_res_hc,
                                               hc_dim, cfg_.ple_conv_kernel, cfg_.ple_ngram_size, T, st);
        }

        // 1. Pre-Attn mHC mix -> rt.d_cur [T, H] + rt.d_hc_inject [T, hc]
        run_hc_mix(rt, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, T);

        // 2. Token Mixer (GDN linear attention OR Full QSA attention)
        if (L.is_recr) {
            if (T < 16 && L.wqkv.dtype == DType::Q8_0 && L.wqkv_gate.dtype == DType::Q8_0) {
                launch_quantize_q8_1(rt.d_cur, rt.d_scratch, H, T, st);
                launch_linear_q8_0_prequant(L.wqkv,      rt.d_scratch, rt.d_qkv_raw, T, st);
                launch_linear_q8_0_prequant(L.wqkv_gate, rt.d_scratch, rt.d_z,       T, st);
            } else {
                launch_linear(rt.cublas, L.wqkv,      rt.d_cur, rt.d_qkv_raw, T, rt.d_scratch, rt.scratch_bytes, st);
                launch_linear(rt.cublas, L.wqkv_gate, rt.d_cur, rt.d_z,       T, rt.d_scratch, rt.scratch_bytes, st);
            }
            launch_linear(rt.cublas, L.ssm_alpha, rt.d_cur, rt.d_alpha, T, rt.d_scratch, rt.scratch_bytes, st);
            launch_linear(rt.cublas, L.ssm_beta,  rt.d_cur, rt.d_beta,  T, rt.d_scratch, rt.scratch_bytes, st);

            launch_ssm_conv1d_silu(rt.d_qkv_raw, static_cast<const float *>(L.ssm_conv1d.ptr),
                                   gdn_conv_[il], rt.d_qkv_conv, T, conv_dim, st, nullptr);
            launch_qwen_gdn_gate_and_beta(rt.d_alpha, static_cast<const float *>(L.ssm_dt.ptr),
                                          static_cast<const float *>(L.ssm_a.ptr),
                                          rt.d_beta, T, cfg_.ssm_dt_rank, st);
            launch_qwen_gdn_packed_d128(rt.d_qkv_conv, rt.d_alpha, rt.d_beta,
                                        gdn_state_[il], rt.d_gdn_out,
                                        T, cfg_.ssm_n_group, cfg_.ssm_dt_rank,
                                        cfg_.rms_eps, gdn_sc, st, nullptr);
            // Qwen3.8 GDN uses SIGMOID output gate (not SiLU)!
            launch_rms_norm_sigmoid_gate_d128(rt.d_gdn_out, static_cast<const float *>(L.ssm_norm.ptr),
                                              rt.d_z, T, cfg_.ssm_dt_rank, cfg_.rms_eps, st);
            launch_linear(rt.cublas, L.ssm_out, rt.d_gdn_out, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, st);
        } else {
            // QSA Indexer K/Q projections + k-pool update + top-k selection!
            launch_linear(rt.cublas, L.idx_k_proj, rt.d_cur, rt.d_idx_k, T, rt.d_scratch, rt.scratch_bytes, st);
            launch_qsa_k_raw_write_and_pool(rt.d_idx_k, idx_k_raw_cache_[il], idx_k_pool_cache_[il],
                                            static_cast<const float *>(L.idx_k_norm.ptr), rt.d_pos,
                                            T, cfg_.idx_kpool, cfg_.rope_dim, cfg_.rms_eps, cfg_.rope_freq_base, st);
            launch_linear(rt.cublas, L.idx_q_proj, rt.d_cur, rt.d_idx_q, T, rt.d_scratch, rt.scratch_bytes, st);
            launch_qsa_q_norm_rope(rt.d_idx_q, static_cast<const float *>(L.idx_q_norm.ptr), rt.d_pos,
                                   T, cfg_.idx_n_head, cfg_.rope_dim, cfg_.rms_eps, cfg_.rope_freq_base, st);
            launch_qsa_select_topk(rt.d_idx_q, idx_k_pool_cache_[il], rt.d_pos,
                                   rt.d_sel_pos, rt.d_sel_len, rt.d_scratch, rt.scratch_bytes,
                                   T, cfg_.idx_n_head, cfg_.idx_kpool, cfg_.idx_top_k, kv_len_max, st);

            // Standard Qwen d256 GQA projections
            if (T < 16 && L.wq.dtype == DType::Q8_0) {
                launch_quantize_q8_1(rt.d_cur, rt.d_scratch, H, T, st);
                launch_linear_q8_0_prequant(L.wq, rt.d_scratch, rt.d_q_full, T, st);
                launch_linear_q8_0_prequant(L.wk, rt.d_scratch, rt.d_k,      T, st);
                launch_linear_q8_0_prequant(L.wv, rt.d_scratch, rt.d_v,      T, st);
            } else {
                launch_linear(rt.cublas, L.wq, rt.d_cur, rt.d_q_full, T, rt.d_scratch, rt.scratch_bytes, st);
                launch_linear(rt.cublas, L.wk, rt.d_cur, rt.d_k,      T, rt.d_scratch, rt.scratch_bytes, st);
                launch_linear(rt.cublas, L.wv, rt.d_cur, rt.d_v,      T, rt.d_scratch, rt.scratch_bytes, st);
            }
            launch_qwen_qgate_split_norm_rope_d256(rt.d_q_full, static_cast<const float *>(L.attn_q_norm.ptr),
                                                   rt.d_pos, rt.d_q, rt.d_gate,
                                                   T, cfg_.n_head, cfg_.rope_dim, cfg_.rms_eps, cfg_.rope_freq_base, st);
            launch_qwen_k_norm_rope_d256(rt.d_k, static_cast<const float *>(L.attn_k_norm.ptr),
                                         rt.d_pos, T, cfg_.n_head_kv, cfg_.rope_dim, cfg_.rms_eps, cfg_.rope_freq_base, st);
            launch_gqa_kv_write_d256(rt.d_k, rt.d_v, k_cache_[il], v_cache_[il], cfg_.kv_q8_0,
                                     rt.d_pos, T, cfg_.n_head_kv, st);
            launch_gqa_attn_d256_qsa(rt.d_q, k_cache_[il], v_cache_[il], cfg_.kv_q8_0,
                                     rt.d_pos, rt.d_sel_pos, rt.d_sel_len,
                                     rt.d_attn_out, rt.d_scratch, rt.scratch_bytes,
                                     T, cfg_.n_head, cfg_.n_head_kv, max_sel, attn_sc,
                                     rt.cublas, kv_len_max, cfg_.idx_top_k, cfg_.idx_kpool, st);
            launch_inkling2_sigmoid_gate_mul(rt.d_attn_out, rt.d_gate, static_cast<int64_t>(T) * d_q, st);
            launch_linear(rt.cublas, L.wo, rt.d_attn_out, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, st);
        }

        // 3. Post-Attn mHC combine -> res_hc += sub_out * 2*sigmoid(inject / hc)
        launch_hc_combine(rt.d_res_hc, rt.d_sub_out, rt.d_hc_inject, inv_hc, H, hc, T, st);

        // 4. Pre-FFN mHC mix -> rt.d_cur [T, H] + rt.d_hc_inject [T, hc]
        run_hc_mix(rt, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, T);

        // 5. Softmax MoE (512 experts, k=10, Q6_K/Q8_0) + Gated Shared Expert
        launch_linear(rt.cublas, L.ffn_gate_inp,       rt.d_cur, rt.d_router,     T, rt.d_scratch, rt.scratch_bytes, st);
        launch_linear(rt.cublas, L.ffn_gate_inp_shexp, rt.d_cur, rt.d_shexp_gate, T, rt.d_scratch, rt.scratch_bytes, st);
        launch_moe_route_softmax(rt.d_router, rt.d_top_ids, rt.d_top_w,
                                 T, cfg_.n_expert, cfg_.n_expert_used, true, cfg_.expert_weights_scale, st);
        launch_qwen_moe_ffn_gated_shared(rt.cublas,
                                         L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps,
                                         L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
                                         rt.d_cur, rt.d_top_ids, rt.d_top_w, rt.d_shexp_gate,
                                         rt.d_sub_out, rt.d_scratch, rt.scratch_bytes,
                                         T, cfg_.n_expert_used, st);

        // 6. Post-FFN mHC combine
        launch_hc_combine(rt.d_res_hc, rt.d_sub_out, rt.d_hc_inject, inv_hc, H, hc, T, st);
        if (hook_ != nullptr) {
            hook_->on_post_layer(il, rt.device, rt.d_res_hc, T, H, hc, st);
        }
    }

    // 7. Final Output mHC Head Mix (no separate output_norm!) -> LM head
    if (gi + 1 == runtimes_.size()) {
        DeviceTensor empty_inj{};
        if (T == 1) {
            run_hc_mix(rt, output_hc_norm_, output_hc_down_, output_hc_up_, empty_inj, 1);
            launch_linear(rt.cublas, output_, rt.d_cur, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, st);
            if (run_argmax) launch_argmax_f32(rt.d_logits, rt.d_argmax, cfg_.n_vocab, st);
        } else {
            // Only need logits on last token (T - 1)
            const size_t last_off = static_cast<size_t>(T - 1) * hc_dim;
            launch_hc_grouped_rmsnorm(rt.d_res_hc + last_off, static_cast<const float *>(output_hc_norm_.ptr), rt.d_hc_xn, H, hc, 1, cfg_.rms_eps, st);
            launch_linear(rt.cublas, output_hc_down_, rt.d_hc_xn, rt.d_hc_lo, 1, rt.d_scratch, rt.scratch_bytes, st);
            launch_scale_silu_inplace(rt.d_hc_lo, inv_hc, cfg_.hc_low_rank, st);
            launch_linear(rt.cublas, output_hc_up_, rt.d_hc_lo, rt.d_hc_gate, 1, rt.d_scratch, rt.scratch_bytes, st);
            launch_hc_pre_gated_mean(rt.d_hc_xn, rt.d_hc_gate, rt.d_cur, H, hc, 1, st);
            launch_linear(rt.cublas, output_, rt.d_cur, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, st);
        }
    }
}

void Qwen4ExpModel::capture_decode_graphs() {
    invalidate_graphs();
    if (!use_cuda_graphs_) return;
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "qwen4 cap setDevice");
        cuda_ok(cudaStreamSynchronize(rt.stream), "qwen4 cap sync");
        cudaGraph_t g = nullptr;
        cuda_ok(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeRelaxed), "qwen4 cap begin");
        run_gpu_segment(gi, 1, true, 0);
        cuda_ok(cudaStreamEndCapture(rt.stream, &g), "qwen4 cap end");
        cuda_ok(cudaGraphInstantiate(&rt.decode_graph_t1, g, nullptr, nullptr, 0), "qwen4 cap inst");
        cudaGraphDestroy(g);
    }
    graphs_ready_ = true;
}

void Qwen4ExpModel::warmup_decode_graphs() {
    if (!use_cuda_graphs_ || graphs_ready_) return;
    capture_decode_graphs();
}

void Qwen4ExpModel::forward(const std::vector<int32_t> & tokens,
                            const std::vector<int32_t> & positions,
                            std::vector<float> & out_logits) {
    const int T = static_cast<int>(tokens.size());
    if (T == 0 || T != static_cast<int>(positions.size()) || T > max_batch_) {
        throw std::runtime_error("Qwen4ExpModel::forward invalid batch size");
    }
    int max_p = 0;
    for (int32_t p : positions) if (p > max_p) max_p = p;
    const int kv_len_max = max_p + 1;

    // Host-RAM PLE n-gram lookup (only 2.7 KB / token!)
    if (ple_tbl_bytes_) {
        compute_ple_host(tokens.data(), positions.data(), T, h_ple_buf_);
    }

    const size_t xfer_b = static_cast<size_t>(T) * cfg_.hc_mult * cfg_.n_embd * sizeof(float);

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "qwen4 fwd setDevice");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, tokens.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "fwd tok");
        } else {
            cuda_ok(cudaStreamSynchronize(runtimes_[gi - 1].stream), "fwd prev sync");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_res_hc, rt.device, runtimes_[gi - 1].d_res_hc, runtimes_[gi - 1].device, xfer_b, rt.stream), "fwd peer");
        }
        if (rt.has_ple_layer && ple_tbl_bytes_) {
            cuda_ok(cudaMemcpyAsync(rt.d_ple_emb, h_ple_buf_.data(), h_ple_buf_.size() * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "fwd ple");
        }
        cuda_ok(cudaMemcpyAsync(rt.d_pos, positions.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "fwd pos");
        run_gpu_segment(gi, T, false, kv_len_max);
    }

    auto & rtl = runtimes_.back();
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaSetDevice(rtl.device), "qwen4 fwd readout");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), rtl.d_logits, static_cast<size_t>(cfg_.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "fwd logits");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "qwen4 fwd sync");
}

int32_t Qwen4ExpModel::forward_decode_greedy(int32_t token, int32_t pos) {
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();

    if (ple_tbl_bytes_) {
        compute_ple_host(&token, &pos, 1, h_ple_buf_);
    }
    const size_t xfer_b = static_cast<size_t>(cfg_.hc_mult) * cfg_.n_embd * sizeof(float);

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "qwen4 dec setDevice");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "dec tok");
        } else {
            cuda_ok(cudaStreamWaitEvent(rt.stream, runtimes_[gi - 1].done_event, 0), "dec wait");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_res_hc, rt.device, runtimes_[gi - 1].d_res_hc, runtimes_[gi - 1].device, xfer_b, rt.stream), "dec peer");
        }
        if (rt.has_ple_layer && ple_tbl_bytes_) {
            cuda_ok(cudaMemcpyAsync(rt.d_ple_emb, h_ple_buf_.data(), h_ple_buf_.size() * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "dec ple");
        }
        cuda_ok(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "dec pos");
        if (graphs_ready_ && rt.decode_graph_t1) cuda_ok(cudaGraphLaunch(rt.decode_graph_t1, rt.stream), "launch t1");
        else run_gpu_segment(gi, 1, true, 0);
        if (gi + 1 < runtimes_.size()) cuda_ok(cudaEventRecord(rt.done_event, rt.stream), "dec rec");
    }

    auto & rtl = runtimes_.back();
    int32_t next_tok = 0;
    cuda_ok(cudaSetDevice(rtl.device), "qwen4 dec readout");
    cuda_ok(cudaMemcpyAsync(&next_tok, rtl.d_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, rtl.stream), "dec argmax");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "qwen4 dec sync");
    return next_tok;
}

void Qwen4ExpModel::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    forward_decode_greedy(token, pos);
    auto & rtl = runtimes_.back();
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaSetDevice(rtl.device), "qwen4 dec logits dev");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), rtl.d_logits, static_cast<size_t>(cfg_.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "dec logits");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "qwen4 dec logits sync");
}

} // namespace rookery
