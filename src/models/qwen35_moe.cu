#include "models/qwen35_moe.h"
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
    cuda_ok(cudaSetDevice(dev), "qwen35 dev_alloc setDevice");
    T * p = nullptr;
    cuda_ok(cudaMalloc(&p, n * sizeof(T)), "qwen35 cudaMalloc");
    return p;
}

} // namespace

void Qwen35MoeModel::invalidate_graphs() noexcept {
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        if (rt.decode_graph_t1) { cudaGraphExecDestroy(rt.decode_graph_t1); rt.decode_graph_t1 = nullptr; }
        if (rt.decode_graph_t2) { cudaGraphExecDestroy(rt.decode_graph_t2); rt.decode_graph_t2 = nullptr; }
    }
    if (!runtimes_.empty()) {
        cudaSetDevice(runtimes_.back().device);
        if (mtp_graph_t1_) { cudaGraphExecDestroy(mtp_graph_t1_); mtp_graph_t1_ = nullptr; }
        if (mtp_graph_t2_) { cudaGraphExecDestroy(mtp_graph_t2_); mtp_graph_t2_ = nullptr; }
    }
    graphs_ready_ = false;
}

Qwen35MoeModel::~Qwen35MoeModel() {
    invalidate_graphs();
    auto freep = [](void * p) { if (p) cudaFree(p); };
    for (size_t il = 0; il < layers_.size(); ++il) {
        cudaSetDevice(layers_[il].device);
        if (il < gdn_conv_.size())       freep(gdn_conv_[il]);
        if (il < gdn_state_.size())      freep(gdn_state_[il]);
        if (il < gdn_conv_snap_.size())  freep(gdn_conv_snap_[il]);
        if (il < gdn_state_snap_.size()) freep(gdn_state_snap_[il]);
        if (il < k_cache_.size())        freep(k_cache_[il]);
        if (il < v_cache_.size())        freep(v_cache_[il]);
    }
    if (!runtimes_.empty()) {
        cudaSetDevice(runtimes_.back().device);
        freep(mtp_k_cache_);
        freep(mtp_v_cache_);
    }
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        freep(rt.d_cur); freep(rt.d_x_norm); freep(rt.d_sub_out);
        freep(rt.d_qkv_raw); freep(rt.d_qkv_conv); freep(rt.d_z);
        freep(rt.d_alpha); freep(rt.d_beta); freep(rt.d_gdn_out);
        freep(rt.d_q_full); freep(rt.d_q); freep(rt.d_gate);
        freep(rt.d_k); freep(rt.d_v); freep(rt.d_attn_out);
        freep(rt.d_router); freep(rt.d_shexp_gate); freep(rt.d_top_ids); freep(rt.d_top_w);
        freep(rt.d_tokens); freep(rt.d_pos); freep(rt.d_logits); freep(rt.d_argmax);
        freep(rt.d_h_nextn); freep(rt.d_mtp_e); freep(rt.d_mtp_eh);
        freep(rt.d_mtp_cur); freep(rt.d_mtp_pos); freep(rt.d_mtp_argmax);
        freep(rt.d_scratch);
        if (rt.done_event) cudaEventDestroy(rt.done_event);
        if (rt.cublas)     cublasDestroy(rt.cublas);
        if (rt.stream)     cudaStreamDestroy(rt.stream);
    }
}

void Qwen35MoeModel::load(const std::string & gguf_path,
                          const std::vector<int> & gpu_ids,
                          const std::vector<float> & split_weights,
                          int max_ctx,
                          bool kv_q8_0,
                          bool load_mtp) {
    if (gpu_ids.empty()) throw std::runtime_error("Qwen35MoeModel::load requires >= 1 GPU");

    const auto t0 = std::chrono::steady_clock::now();
    auto sg = ShardedGguf::open(gguf_path);
    const std::string arch = sg.get_str("general.architecture");
    if (arch != "qwen35moe") {
        throw std::runtime_error("Expected qwen35moe GGUF, got: " + arch);
    }
    const std::string pfx = arch + ".";

    cfg_.n_blocks             = static_cast<int>(sg.get_u64(pfx + "block_count", 41));
    cfg_.n_nextn              = static_cast<int>(sg.get_u64(pfx + "nextn_predict_layers", 0));
    cfg_.n_layers             = cfg_.n_blocks - cfg_.n_nextn;
    cfg_.n_embd               = static_cast<int>(sg.get_u64(pfx + "embedding_length", 2048));
    cfg_.n_vocab              = static_cast<int>(sg.get_u64(pfx + "vocab_size", 0));
    if (cfg_.n_vocab == 0) {
        auto loc = sg.find_tensor("token_embd.weight");
        if (loc.info) cfg_.n_vocab = static_cast<int>(loc.info->ne[1]);
    }
    cfg_.n_head               = static_cast<int>(sg.get_u64(pfx + "attention.head_count", 16));
    cfg_.n_head_kv            = static_cast<int>(sg.get_u64(pfx + "attention.head_count_kv", 2));
    cfg_.head_dim             = static_cast<int>(sg.get_u64(pfx + "attention.key_length", 256));
    cfg_.rope_dim             = static_cast<int>(sg.get_u64(pfx + "rope.dimension_count", 64));
    cfg_.rope_freq_base       = static_cast<float>(sg.get_f64(pfx + "rope.freq_base", 10000000.0));
    cfg_.rms_eps              = static_cast<float>(sg.get_f64(pfx + "attention.layer_norm_rms_epsilon", 1e-6));
    cfg_.ssm_d_conv           = static_cast<int>(sg.get_u64(pfx + "ssm.conv_kernel", 4));
    cfg_.ssm_d_state          = static_cast<int>(sg.get_u64(pfx + "ssm.state_size", 128));
    cfg_.ssm_n_group          = static_cast<int>(sg.get_u64(pfx + "ssm.group_count", 16));
    cfg_.ssm_dt_rank          = static_cast<int>(sg.get_u64(pfx + "ssm.time_step_rank", 32));
    cfg_.ssm_d_inner          = static_cast<int>(sg.get_u64(pfx + "ssm.inner_size", 4096));
    cfg_.full_attn_interval   = static_cast<int>(sg.get_u64(pfx + "full_attention_interval", 4));
    cfg_.n_expert             = static_cast<int>(sg.get_u64(pfx + "expert_count", 256));
    cfg_.n_expert_used        = static_cast<int>(sg.get_u64(pfx + "expert_used_count", 8));
    cfg_.n_ff_exp             = static_cast<int>(sg.get_u64(pfx + "expert_feed_forward_length", 512));
    cfg_.n_ff_shexp           = static_cast<int>(sg.get_u64(pfx + "expert_shared_feed_forward_length", 512));
    cfg_.expert_weights_scale = static_cast<float>(sg.get_f64(pfx + "expert_weights_scale", 1.0));
    cfg_.max_ctx              = max_ctx;
    cfg_.kv_q8_0              = kv_q8_0;

    cfg_.is_recr.resize(cfg_.n_layers);
    for (int i = 0; i < cfg_.n_layers; ++i) {
        cfg_.is_recr[i] = ((i + 1) % cfg_.full_attn_interval != 0);
    }

    const std::string mtp_pfx = "blk." + std::to_string(cfg_.n_layers) + ".";
    has_mtp_ = load_mtp && cfg_.n_nextn >= 1 && (sg.find_tensor(mtp_pfx + "nextn.eh_proj.weight").info != nullptr);

    // Partition backbone layers proportionally across GPUs
    std::vector<float> sw = split_weights;
    if (sw.size() != gpu_ids.size()) {
        // Auto-balance by free VRAM when no explicit split is given!
        sw.resize(gpu_ids.size());
        for (size_t g = 0; g < gpu_ids.size(); ++g) {
            cudaSetDevice(gpu_ids[g]);
            size_t f = 0, tot = 0;
            cudaMemGetInfo(&f, &tot);
            sw[g] = static_cast<float>(f);
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

    for (size_t g = 0; g + 1 < runtimes_.size(); ++g) {
        int can = 0;
        cudaDeviceCanAccessPeer(&can, runtimes_[g + 1].device, runtimes_[g].device);
        if (can) { cudaSetDevice(runtimes_[g + 1].device); cudaDeviceEnablePeerAccess(runtimes_[g].device, 0); cudaGetLastError(); }
    }

    layers_.resize(cfg_.n_layers);
    gdn_conv_.assign(cfg_.n_layers, nullptr);
    gdn_state_.assign(cfg_.n_layers, nullptr);
    gdn_conv_snap_.assign(cfg_.n_layers, nullptr);
    gdn_state_snap_.assign(cfg_.n_layers, nullptr);
    k_cache_.assign(cfg_.n_layers, nullptr);
    v_cache_.assign(cfg_.n_layers, nullptr);

    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner; // 8192
    const int d_kv     = cfg_.n_head_kv * cfg_.head_dim;                             // 512
    const size_t kv_bytes = kv_q8_0
        ? static_cast<size_t>(cfg_.max_ctx) * (d_kv / 32) * sizeof(BlockQ8_0)
        : static_cast<size_t>(cfg_.max_ctx) * d_kv * sizeof(uint16_t);

    // Parallel multi-GPU weight upload
    std::vector<std::thread> load_threads;
    std::vector<std::exception_ptr> load_errs(runtimes_.size(), nullptr);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        load_threads.emplace_back([&, gi]() {
            try {
                const int dev = runtimes_[gi].device;
                cuda_ok(cudaSetDevice(dev), "qwen35 load setDevice");
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
                    L.device  = dev;
                    L.is_recr = cfg_.is_recr[il];

                    L.attn_norm      = pool_.upload(sg, p + "attn_norm.weight",           dev, false);
                    L.post_attn_norm = pool_.upload(sg, p + "post_attention_norm.weight", dev, false);

                    if (L.is_recr) {
                        L.wqkv       = pool_.upload(sg, p + "attn_qkv.weight",   dev, false);
                        L.wqkv_gate  = pool_.upload(sg, p + "attn_gate.weight",  dev, false);
                        L.ssm_conv1d = pool_.upload(sg, p + "ssm_conv1d.weight", dev, false);
                        L.ssm_dt     = pool_.upload(sg, p + "ssm_dt.bias",       dev, false);
                        L.ssm_a      = pool_.upload(sg, p + "ssm_a",             dev, false);
                        L.ssm_alpha  = pool_.upload(sg, p + "ssm_alpha.weight",  dev, false);
                        L.ssm_beta   = pool_.upload(sg, p + "ssm_beta.weight",   dev, false);
                        L.ssm_norm   = pool_.upload(sg, p + "ssm_norm.weight",   dev, false);
                        L.ssm_out    = pool_.upload(sg, p + "ssm_out.weight",    dev, false);

                        const size_t conv_n = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim;
                        const size_t ssm_n  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state;
                        gdn_conv_[il]  = dev_alloc<float>(dev, conv_n);
                        gdn_state_[il] = dev_alloc<float>(dev, ssm_n);
                        if (has_mtp_) {
                            gdn_conv_snap_[il]  = dev_alloc<float>(dev, conv_n);
                            gdn_state_snap_[il] = dev_alloc<float>(dev, ssm_n);
                        }
                    } else {
                        L.wq          = pool_.upload(sg, p + "attn_q.weight",      dev, false);
                        L.wk          = pool_.upload(sg, p + "attn_k.weight",      dev, false);
                        L.wv          = pool_.upload(sg, p + "attn_v.weight",      dev, false);
                        L.wo          = pool_.upload(sg, p + "attn_output.weight", dev, false);
                        L.attn_q_norm = pool_.upload(sg, p + "attn_q_norm.weight", dev, false);
                        L.attn_k_norm = pool_.upload(sg, p + "attn_k_norm.weight", dev, false);

                        cuda_ok(cudaMalloc(&k_cache_[il], kv_bytes), "qwen35 malloc k_cache");
                        cuda_ok(cudaMalloc(&v_cache_[il], kv_bytes), "qwen35 malloc v_cache");
                    }

                    L.ffn_gate_inp       = pool_.upload(sg, p + "ffn_gate_inp.weight",       dev, false);
                    L.ffn_gate_exps      = pool_.upload(sg, p + "ffn_gate_exps.weight",      dev, false);
                    L.ffn_up_exps        = pool_.upload(sg, p + "ffn_up_exps.weight",        dev, false);
                    L.ffn_down_exps      = pool_.upload(sg, p + "ffn_down_exps.weight",      dev, false);
                    L.ffn_gate_inp_shexp = pool_.upload(sg, p + "ffn_gate_inp_shexp.weight", dev, false);
                    L.ffn_gate_shexp     = pool_.upload(sg, p + "ffn_gate_shexp.weight",     dev, false);
                    L.ffn_up_shexp       = pool_.upload(sg, p + "ffn_up_shexp.weight",       dev, false);
                    L.ffn_down_shexp     = pool_.upload(sg, p + "ffn_down_shexp.weight",     dev, false);
                }

                if (gi + 1 == runtimes_.size() && has_mtp_) {
                    mtp_.device           = dev;
                    mtp_.eh_proj          = pool_.upload(sg, mtp_pfx + "nextn.eh_proj.weight",          dev, false);
                    mtp_.enorm            = pool_.upload(sg, mtp_pfx + "nextn.enorm.weight",            dev, false);
                    mtp_.hnorm            = pool_.upload(sg, mtp_pfx + "nextn.hnorm.weight",            dev, false);
                    mtp_.shared_head_norm = pool_.upload(sg, mtp_pfx + "nextn.shared_head_norm.weight", dev, false, false);
                    if (!mtp_.shared_head_norm.valid()) mtp_.shared_head_norm = output_norm_;

                    mtp_.attn_norm        = pool_.upload(sg, mtp_pfx + "attn_norm.weight",           dev, false);
                    mtp_.post_attn_norm   = pool_.upload(sg, mtp_pfx + "post_attention_norm.weight", dev, false);
                    mtp_.wq               = pool_.upload(sg, mtp_pfx + "attn_q.weight",              dev, false);
                    mtp_.wk               = pool_.upload(sg, mtp_pfx + "attn_k.weight",              dev, false);
                    mtp_.wv               = pool_.upload(sg, mtp_pfx + "attn_v.weight",              dev, false);
                    mtp_.wo               = pool_.upload(sg, mtp_pfx + "attn_output.weight",         dev, false);
                    mtp_.attn_q_norm      = pool_.upload(sg, mtp_pfx + "attn_q_norm.weight",         dev, false);
                    mtp_.attn_k_norm      = pool_.upload(sg, mtp_pfx + "attn_k_norm.weight",         dev, false);

                    mtp_.ffn_gate_inp       = pool_.upload(sg, mtp_pfx + "ffn_gate_inp.weight",       dev, false);
                    mtp_.ffn_gate_exps      = pool_.upload(sg, mtp_pfx + "ffn_gate_exps.weight",      dev, false);
                    mtp_.ffn_up_exps        = pool_.upload(sg, mtp_pfx + "ffn_up_exps.weight",        dev, false);
                    mtp_.ffn_down_exps      = pool_.upload(sg, mtp_pfx + "ffn_down_exps.weight",      dev, false);
                    mtp_.ffn_gate_inp_shexp = pool_.upload(sg, mtp_pfx + "ffn_gate_inp_shexp.weight", dev, false);
                    mtp_.ffn_gate_shexp     = pool_.upload(sg, mtp_pfx + "ffn_gate_shexp.weight",     dev, false);
                    mtp_.ffn_up_shexp       = pool_.upload(sg, mtp_pfx + "ffn_up_shexp.weight",       dev, false);
                    mtp_.ffn_down_shexp     = pool_.upload(sg, mtp_pfx + "ffn_down_shexp.weight",     dev, false);

                    cuda_ok(cudaMalloc(&mtp_k_cache_, kv_bytes), "qwen35 malloc mtp_k");
                    cuda_ok(cudaMalloc(&mtp_v_cache_, kv_bytes), "qwen35 malloc mtp_v");
                }
                cuda_ok(cudaDeviceSynchronize(), "qwen35 load sync");
            } catch (...) {
                load_errs[gi] = std::current_exception();
            }
        });
    }
    for (auto & th : load_threads) th.join();
    for (auto & ep : load_errs) if (ep) std::rethrow_exception(ep);
    pool_.release_staging();

    // Allocate per-GPU runtime buffers
    const size_t B = static_cast<size_t>(max_batch_);
    const size_t H = static_cast<size_t>(cfg_.n_embd);
    const size_t d_q = static_cast<size_t>(cfg_.n_head) * cfg_.head_dim;
    const size_t scratch_b = 128ULL * 1024 * 1024;

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        const int dev = rt.device;
        cuda_ok(cudaSetDevice(dev), "qwen35 rt setDevice");
        cuda_ok(cudaStreamCreate(&rt.stream), "qwen35 stream");
        cuda_ok(cudaEventCreateWithFlags(&rt.done_event, cudaEventDisableTiming), "qwen35 event");
        if (cublasCreate(&rt.cublas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        if (device_has_tensor_cores(dev)) cublasSetMathMode(rt.cublas, CUBLAS_TENSOR_OP_MATH);

        rt.d_cur        = dev_alloc<float>(dev, B * H);
        rt.d_x_norm     = dev_alloc<float>(dev, B * H);
        rt.d_sub_out    = dev_alloc<float>(dev, B * H);
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
        rt.d_router     = dev_alloc<float>(dev, B * cfg_.n_expert);
        rt.d_shexp_gate = dev_alloc<float>(dev, B);
        rt.d_top_ids    = dev_alloc<int32_t>(dev, B * cfg_.n_expert_used);
        rt.d_top_w      = dev_alloc<float>(dev, B * cfg_.n_expert_used);
        rt.d_pos        = dev_alloc<int32_t>(dev, B);
        if (gi == 0) rt.d_tokens = dev_alloc<int32_t>(dev, B);
        if (gi + 1 == runtimes_.size()) {
            rt.d_logits = dev_alloc<float>(dev, 2ULL * cfg_.n_vocab);
            rt.d_argmax = dev_alloc<int32_t>(dev, 2);
            if (has_mtp_) {
                rt.d_h_nextn    = dev_alloc<float>(dev, B * H);
                rt.d_mtp_e      = dev_alloc<float>(dev, B * H);
                rt.d_mtp_eh     = dev_alloc<float>(dev, B * 2 * H);
                rt.d_mtp_cur    = dev_alloc<float>(dev, B * H);
                rt.d_mtp_pos    = dev_alloc<int32_t>(dev, B);
                rt.d_mtp_argmax = dev_alloc<int32_t>(dev, 1);
            }
        }
        cuda_ok(cudaMalloc(&rt.d_scratch, scratch_b), "qwen35 scratch");
        rt.scratch_bytes = scratch_b;
    }

    reset_state();

    const auto t1 = std::chrono::steady_clock::now();
    std::printf("[Qwen3.5-MoE] Loaded %s (%d layers + %d MTP) in %.2f s across %zu GPU(s):",
                arch.c_str(), cfg_.n_layers, has_mtp_ ? 1 : 0,
                std::chrono::duration<double>(t1 - t0).count(), runtimes_.size());
    for (const auto & rt : runtimes_) {
        std::printf(" GPU%d[%d..%d](%.2f GiB)", rt.device, rt.layer_start, rt.layer_end - 1,
                    pool_.bytes_on_device(rt.device) / (1024.0 * 1024.0 * 1024.0));
    }
    std::printf(" | ctx=%d (KV=%s)\n", cfg_.max_ctx, kv_q8_0 ? "Q8_0" : "F16");
}

void Qwen35MoeModel::reset_state() {
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const size_t conv_b = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim * sizeof(float);
    const size_t ssm_b  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state * sizeof(float);
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (!layers_[il].is_recr) continue;
        cuda_ok(cudaSetDevice(layers_[il].device), "qwen35 reset setDevice");
        cuda_ok(cudaMemset(gdn_conv_[il],  0, conv_b), "zero gdn_conv");
        cuda_ok(cudaMemset(gdn_state_[il], 0, ssm_b),  "zero gdn_state");
    }
}

void Qwen35MoeModel::save_gdn_checkpoint(GdnCheckpoint & out, int pos) const {
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const size_t conv_n = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim;
    const size_t ssm_n  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state;
    out.pos = pos;
    out.conv.resize(cfg_.n_layers);
    out.ssm.resize(cfg_.n_layers);
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (!layers_[il].is_recr) continue;
        cuda_ok(cudaSetDevice(layers_[il].device), "qwen35 save_cp setDevice");
        out.conv[il].resize(conv_n);
        out.ssm[il].resize(ssm_n);
        cuda_ok(cudaMemcpy(out.conv[il].data(), gdn_conv_[il],  conv_n * sizeof(float), cudaMemcpyDeviceToHost), "save conv");
        cuda_ok(cudaMemcpy(out.ssm[il].data(),  gdn_state_[il], ssm_n  * sizeof(float), cudaMemcpyDeviceToHost), "save ssm");
    }
}

void Qwen35MoeModel::restore_gdn_checkpoint(const GdnCheckpoint & cp) {
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const size_t conv_n = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim;
    const size_t ssm_n  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state;
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (!layers_[il].is_recr) continue;
        cuda_ok(cudaSetDevice(layers_[il].device), "qwen35 restore_cp setDevice");
        cuda_ok(cudaMemcpy(gdn_conv_[il],  cp.conv[il].data(), conv_n * sizeof(float), cudaMemcpyHostToDevice), "restore conv");
        cuda_ok(cudaMemcpy(gdn_state_[il], cp.ssm[il].data(),  ssm_n  * sizeof(float), cudaMemcpyHostToDevice), "restore ssm");
    }
}

void Qwen35MoeModel::rollback_gdn_states() {
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const size_t conv_b = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim * sizeof(float);
    const size_t ssm_b  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state * sizeof(float);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cudaSetDevice(rt.device);
        for (int il = rt.layer_start; il < rt.layer_end; ++il) {
            if (!layers_[il].is_recr) continue;
            cudaMemcpyAsync(gdn_conv_[il],  gdn_conv_snap_[il],  conv_b, cudaMemcpyDeviceToDevice, rt.stream);
            cudaMemcpyAsync(gdn_state_[il], gdn_state_snap_[il], ssm_b,  cudaMemcpyDeviceToDevice, rt.stream);
        }
    }
}

void Qwen35MoeModel::run_gpu_segment(size_t gi, int T, bool run_argmax, bool save_gdn_snap, int kv_len_max) {
    auto & rt = runtimes_[gi];
    cudaSetDevice(rt.device);
    cudaStream_t st = rt.stream;

    const int H        = cfg_.n_embd;
    const int d_q      = cfg_.n_head * cfg_.head_dim;
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
    const float gdn_scale  = 1.0f / std::sqrt(static_cast<float>(cfg_.ssm_d_state));

    if (gi == 0) {
        launch_embedding_lookup(tok_embd_, rt.d_tokens, rt.d_cur, H, 1, T, st);
    }

    for (int il = rt.layer_start; il < rt.layer_end; ++il) {
        const auto & L = layers_[il];

        // 1. Pre-attention RMSNorm
        launch_rmsnorm(rt.d_cur, static_cast<const float *>(L.attn_norm.ptr), rt.d_x_norm, H, T, cfg_.rms_eps, st);

        if (L.is_recr) {
            // 2A. Gated Delta Net linear attention
            if (T < 16 && L.wqkv.dtype == DType::Q8_0) {
                launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, H, T, st);
                launch_linear_q8_0_prequant(L.wqkv,      rt.d_scratch, rt.d_qkv_raw, T, st);
                launch_linear_q8_0_prequant(L.wqkv_gate, rt.d_scratch, rt.d_z,       T, st);
                if (L.ssm_alpha.dtype == DType::Q8_0) launch_linear_q8_0_prequant(L.ssm_alpha, rt.d_scratch, rt.d_alpha, T, st);
                else launch_linear(rt.cublas, L.ssm_alpha, rt.d_x_norm, rt.d_alpha, T, rt.d_scratch, rt.scratch_bytes, st);
                if (L.ssm_beta.dtype == DType::Q8_0)  launch_linear_q8_0_prequant(L.ssm_beta,  rt.d_scratch, rt.d_beta,  T, st);
                else launch_linear(rt.cublas, L.ssm_beta,  rt.d_x_norm, rt.d_beta,  T, rt.d_scratch, rt.scratch_bytes, st);
            } else {
                launch_linear(rt.cublas, L.wqkv,      rt.d_x_norm, rt.d_qkv_raw, T, rt.d_scratch, rt.scratch_bytes, st);
                launch_linear(rt.cublas, L.wqkv_gate, rt.d_x_norm, rt.d_z,       T, rt.d_scratch, rt.scratch_bytes, st);
                launch_linear(rt.cublas, L.ssm_alpha, rt.d_x_norm, rt.d_alpha,   T, rt.d_scratch, rt.scratch_bytes, st);
                launch_linear(rt.cublas, L.ssm_beta,  rt.d_x_norm, rt.d_beta,    T, rt.d_scratch, rt.scratch_bytes, st);
            }

            launch_qwen_gdn_gate_and_beta(rt.d_alpha, static_cast<const float *>(L.ssm_dt.ptr),
                                          static_cast<const float *>(L.ssm_a.ptr), rt.d_beta,
                                          T, cfg_.ssm_dt_rank, st);
            launch_ssm_conv1d_silu(rt.d_qkv_raw, static_cast<const float *>(L.ssm_conv1d.ptr),
                                   gdn_conv_[il], rt.d_qkv_conv, T, conv_dim, st,
                                   save_gdn_snap ? gdn_conv_snap_[il] : nullptr);
            launch_qwen_gdn_packed_d128(rt.d_qkv_conv, rt.d_alpha, rt.d_beta,
                                        gdn_state_[il], rt.d_gdn_out,
                                        T, cfg_.ssm_n_group, cfg_.ssm_dt_rank, cfg_.rms_eps, gdn_scale, st,
                                        save_gdn_snap ? gdn_state_snap_[il] : nullptr);
            launch_rms_norm_silu_gate_d128(rt.d_gdn_out, static_cast<const float *>(L.ssm_norm.ptr),
                                           rt.d_z, T, cfg_.ssm_dt_rank, cfg_.rms_eps, st);
            launch_linear(rt.cublas, L.ssm_out, rt.d_gdn_out, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, st);
        } else {
            // 2B. Full GQA attention (head_dim=256, partial NeoX RoPE=64, gated output)
            if (T < 16 && L.wq.dtype == DType::Q8_0) {
                launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, H, T, st);
                launch_linear_q8_0_prequant(L.wq, rt.d_scratch, rt.d_q_full, T, st);
                launch_linear_q8_0_prequant(L.wk, rt.d_scratch, rt.d_k,      T, st);
                launch_linear_q8_0_prequant(L.wv, rt.d_scratch, rt.d_v,      T, st);
            } else {
                launch_linear(rt.cublas, L.wq, rt.d_x_norm, rt.d_q_full, T, rt.d_scratch, rt.scratch_bytes, st);
                launch_linear(rt.cublas, L.wk, rt.d_x_norm, rt.d_k,      T, rt.d_scratch, rt.scratch_bytes, st);
                launch_linear(rt.cublas, L.wv, rt.d_x_norm, rt.d_v,      T, rt.d_scratch, rt.scratch_bytes, st);
            }
            launch_qwen_qgate_split_norm_rope_d256(rt.d_q_full, static_cast<const float *>(L.attn_q_norm.ptr),
                                                   rt.d_pos, rt.d_q, rt.d_gate,
                                                   T, cfg_.n_head, cfg_.rope_dim, cfg_.rms_eps, cfg_.rope_freq_base, st);
            launch_qwen_k_norm_rope_d256(rt.d_k, static_cast<const float *>(L.attn_k_norm.ptr),
                                         rt.d_pos, T, cfg_.n_head_kv, cfg_.rope_dim, cfg_.rms_eps, cfg_.rope_freq_base, st);
            launch_gqa_kv_write_d256(rt.d_k, rt.d_v, k_cache_[il], v_cache_[il], cfg_.kv_q8_0,
                                     rt.d_pos, T, cfg_.n_head_kv, st);
            launch_gqa_attn_d256(rt.d_q, k_cache_[il], v_cache_[il], cfg_.kv_q8_0,
                                 rt.d_pos, rt.d_attn_out, rt.d_scratch, rt.scratch_bytes,
                                 T, cfg_.n_head, cfg_.n_head_kv, attn_scale, rt.cublas, kv_len_max, st);
            launch_inkling2_sigmoid_gate_mul(rt.d_attn_out, rt.d_gate, static_cast<int64_t>(T) * d_q, st);
            launch_linear(rt.cublas, L.wo, rt.d_attn_out, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, st);
        }

        // 3. Attention residual + Post-attention RMSNorm
        launch_add_inplace(rt.d_cur, rt.d_sub_out, static_cast<int64_t>(T) * H, st);
        launch_rmsnorm(rt.d_cur, static_cast<const float *>(L.post_attn_norm.ptr), rt.d_x_norm, H, T, cfg_.rms_eps, st);

        // 4. MoE FFN (Softmax top-8 + Gated Shared Expert) + Residual
        launch_linear(rt.cublas, L.ffn_gate_inp,       rt.d_x_norm, rt.d_router,     T, rt.d_scratch, rt.scratch_bytes, st);
        launch_linear(rt.cublas, L.ffn_gate_inp_shexp, rt.d_x_norm, rt.d_shexp_gate, T, rt.d_scratch, rt.scratch_bytes, st);
        launch_moe_route_softmax(rt.d_router, rt.d_top_ids, rt.d_top_w,
                                 T, cfg_.n_expert, cfg_.n_expert_used, true, cfg_.expert_weights_scale, st);
        launch_qwen_moe_ffn_gated_shared(rt.cublas,
                                         L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps,
                                         L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
                                         rt.d_x_norm, rt.d_top_ids, rt.d_top_w, rt.d_shexp_gate,
                                         rt.d_sub_out, rt.d_scratch, rt.scratch_bytes,
                                         T, cfg_.n_expert_used, st);
        launch_add_inplace(rt.d_cur, rt.d_sub_out, static_cast<int64_t>(T) * H, st);
        if (hook_ != nullptr && hook_->hooks_layer(il)) {
            hook_->on_post_layer(il, rt.device, rt.d_cur, T, H, 1, st);
        }
    }

    // 5. Final RMSNorm -> save h_nextn (post-norm!) -> LM head
    if (gi + 1 == runtimes_.size()) {
        launch_rmsnorm(rt.d_cur, static_cast<const float *>(output_norm_.ptr), rt.d_x_norm, H, T, cfg_.rms_eps, st);
        if (has_mtp_ && rt.d_h_nextn) {
            cudaMemcpyAsync(rt.d_h_nextn, rt.d_x_norm, static_cast<size_t>(T) * H * sizeof(float), cudaMemcpyDeviceToDevice, st);
        }
        if (run_argmax && T <= 2) {
            launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, T, rt.d_scratch, rt.scratch_bytes, st);
            launch_argmax_f32_batch(rt.d_logits, rt.d_argmax, cfg_.n_vocab, T, st);
        } else {
            launch_linear(rt.cublas, output_, rt.d_x_norm + static_cast<size_t>(T - 1) * H,
                          rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, st);
        }
    }
}

void Qwen35MoeModel::run_mtp_layer(int T, bool run_argmax, int kv_len_max) {
    auto & rt = runtimes_.back();
    cudaSetDevice(rt.device);
    cudaStream_t st = rt.stream;

    const int H = cfg_.n_embd;
    const int d_q = cfg_.n_head * cfg_.head_dim;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));

    // eh_proj([RMSNorm(embd, enorm); RMSNorm(h_nextn, hnorm)]) -> d_mtp_cur
    launch_mtp_eh_concat(rt.d_mtp_e, rt.d_h_nextn,
                         static_cast<const float *>(mtp_.enorm.ptr),
                         static_cast<const float *>(mtp_.hnorm.ptr),
                         rt.d_mtp_eh, H, 1, T, cfg_.rms_eps, st);
    launch_linear(rt.cublas, mtp_.eh_proj, rt.d_mtp_eh, rt.d_mtp_cur, T, rt.d_scratch, rt.scratch_bytes, st);

    // Full GQA block (blk.40)
    launch_rmsnorm(rt.d_mtp_cur, static_cast<const float *>(mtp_.attn_norm.ptr), rt.d_x_norm, H, T, cfg_.rms_eps, st);
    if (T < 16 && mtp_.wq.dtype == DType::Q8_0) {
        launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, H, T, st);
        launch_linear_q8_0_prequant(mtp_.wq, rt.d_scratch, rt.d_q_full, T, st);
        launch_linear_q8_0_prequant(mtp_.wk, rt.d_scratch, rt.d_k,      T, st);
        launch_linear_q8_0_prequant(mtp_.wv, rt.d_scratch, rt.d_v,      T, st);
    } else {
        launch_linear(rt.cublas, mtp_.wq, rt.d_x_norm, rt.d_q_full, T, rt.d_scratch, rt.scratch_bytes, st);
        launch_linear(rt.cublas, mtp_.wk, rt.d_x_norm, rt.d_k,      T, rt.d_scratch, rt.scratch_bytes, st);
        launch_linear(rt.cublas, mtp_.wv, rt.d_x_norm, rt.d_v,      T, rt.d_scratch, rt.scratch_bytes, st);
    }
    launch_qwen_qgate_split_norm_rope_d256(rt.d_q_full, static_cast<const float *>(mtp_.attn_q_norm.ptr),
                                           rt.d_mtp_pos, rt.d_q, rt.d_gate,
                                           T, cfg_.n_head, cfg_.rope_dim, cfg_.rms_eps, cfg_.rope_freq_base, st);
    launch_qwen_k_norm_rope_d256(rt.d_k, static_cast<const float *>(mtp_.attn_k_norm.ptr),
                                 rt.d_mtp_pos, T, cfg_.n_head_kv, cfg_.rope_dim, cfg_.rms_eps, cfg_.rope_freq_base, st);
    launch_gqa_kv_write_d256(rt.d_k, rt.d_v, mtp_k_cache_, mtp_v_cache_, cfg_.kv_q8_0,
                             rt.d_mtp_pos, T, cfg_.n_head_kv, st);
    launch_gqa_attn_d256(rt.d_q, mtp_k_cache_, mtp_v_cache_, cfg_.kv_q8_0,
                         rt.d_mtp_pos, rt.d_attn_out, rt.d_scratch, rt.scratch_bytes,
                         T, cfg_.n_head, cfg_.n_head_kv, attn_scale, rt.cublas, kv_len_max, st);
    launch_inkling2_sigmoid_gate_mul(rt.d_attn_out, rt.d_gate, static_cast<int64_t>(T) * d_q, st);
    launch_linear(rt.cublas, mtp_.wo, rt.d_attn_out, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, st);

    launch_add_inplace(rt.d_mtp_cur, rt.d_sub_out, static_cast<int64_t>(T) * H, st);
    launch_rmsnorm(rt.d_mtp_cur, static_cast<const float *>(mtp_.post_attn_norm.ptr), rt.d_x_norm, H, T, cfg_.rms_eps, st);

    // MoE FFN
    launch_linear(rt.cublas, mtp_.ffn_gate_inp,       rt.d_x_norm, rt.d_router,     T, rt.d_scratch, rt.scratch_bytes, st);
    launch_linear(rt.cublas, mtp_.ffn_gate_inp_shexp, rt.d_x_norm, rt.d_shexp_gate, T, rt.d_scratch, rt.scratch_bytes, st);
    launch_moe_route_softmax(rt.d_router, rt.d_top_ids, rt.d_top_w,
                             T, cfg_.n_expert, cfg_.n_expert_used, true, cfg_.expert_weights_scale, st);
    launch_qwen_moe_ffn_gated_shared(rt.cublas,
                                     mtp_.ffn_gate_exps, mtp_.ffn_up_exps, mtp_.ffn_down_exps,
                                     mtp_.ffn_gate_shexp, mtp_.ffn_up_shexp, mtp_.ffn_down_shexp,
                                     rt.d_x_norm, rt.d_top_ids, rt.d_top_w, rt.d_shexp_gate,
                                     rt.d_sub_out, rt.d_scratch, rt.scratch_bytes,
                                     T, cfg_.n_expert_used, st);
    launch_add_inplace(rt.d_mtp_cur, rt.d_sub_out, static_cast<int64_t>(T) * H, st);

    // shared_head_norm + LM head on last token
    launch_rmsnorm(rt.d_mtp_cur + static_cast<size_t>(T - 1) * H,
                   static_cast<const float *>(mtp_.shared_head_norm.ptr),
                   rt.d_x_norm, H, 1, cfg_.rms_eps, st);
    launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, st);
    if (run_argmax) {
        launch_argmax_f32(rt.d_logits, rt.d_mtp_argmax, cfg_.n_vocab, st);
    }
}

void Qwen35MoeModel::embed_tokens_to_last_dev(const int32_t * h_tokens, int T) {
    auto & rt0 = runtimes_.front();
    auto & rtl = runtimes_.back();
    const size_t bytes = static_cast<size_t>(T) * cfg_.n_embd * sizeof(float);
    cuda_ok(cudaSetDevice(rt0.device), "mtp embd dev0");
    cuda_ok(cudaMemcpyAsync(rt0.d_tokens, h_tokens, T * sizeof(int32_t), cudaMemcpyHostToDevice, rt0.stream), "mtp tok");
    launch_embedding_lookup(tok_embd_, rt0.d_tokens, rt0.d_sub_out, cfg_.n_embd, 1, T, rt0.stream);
    if (runtimes_.size() == 1) {
        cuda_ok(cudaMemcpyAsync(rtl.d_mtp_e, rt0.d_sub_out, bytes, cudaMemcpyDeviceToDevice, rt0.stream), "mtp e d2d");
    } else {
        cuda_ok(cudaEventRecord(rt0.done_event, rt0.stream), "mtp embd rec");
        cuda_ok(cudaSetDevice(rtl.device), "mtp embd last");
        cuda_ok(cudaStreamWaitEvent(rtl.stream, rt0.done_event, 0), "mtp embd wait");
        cuda_ok(cudaMemcpyPeerAsync(rtl.d_mtp_e, rtl.device, rt0.d_sub_out, rt0.device, bytes, rtl.stream), "mtp e peer");
    }
}

void Qwen35MoeModel::capture_decode_graphs() {
    invalidate_graphs();
    if (!use_cuda_graphs_) return;
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "qwen35 cap setDevice");
        cuda_ok(cudaStreamSynchronize(rt.stream), "qwen35 cap sync");
        {
            cudaGraph_t g = nullptr;
            cuda_ok(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeRelaxed), "cap t1");
            run_gpu_segment(gi, 1, true, false, 0);
            cuda_ok(cudaStreamEndCapture(rt.stream, &g), "end t1");
            cuda_ok(cudaGraphInstantiate(&rt.decode_graph_t1, g, nullptr, nullptr, 0), "inst t1");
            cudaGraphDestroy(g);
        }
        if (has_mtp_) {
            cudaGraph_t g = nullptr;
            cuda_ok(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeRelaxed), "cap t2");
            run_gpu_segment(gi, 2, true, true, 0);
            cuda_ok(cudaStreamEndCapture(rt.stream, &g), "end t2");
            cuda_ok(cudaGraphInstantiate(&rt.decode_graph_t2, g, nullptr, nullptr, 0), "inst t2");
            cudaGraphDestroy(g);
        }
    }
    if (has_mtp_) {
        auto & rtl = runtimes_.back();
        cuda_ok(cudaSetDevice(rtl.device), "cap mtp setDevice");
        cuda_ok(cudaStreamSynchronize(rtl.stream), "cap mtp sync");
        {
            cudaGraph_t g = nullptr;
            cuda_ok(cudaStreamBeginCapture(rtl.stream, cudaStreamCaptureModeRelaxed), "cap mtp t1");
            run_mtp_layer(1, true, 0);
            cuda_ok(cudaStreamEndCapture(rtl.stream, &g), "end mtp t1");
            cuda_ok(cudaGraphInstantiate(&mtp_graph_t1_, g, nullptr, nullptr, 0), "inst mtp t1");
            cudaGraphDestroy(g);
        }
        {
            cudaGraph_t g = nullptr;
            cuda_ok(cudaStreamBeginCapture(rtl.stream, cudaStreamCaptureModeRelaxed), "cap mtp t2");
            run_mtp_layer(2, true, 0);
            cuda_ok(cudaStreamEndCapture(rtl.stream, &g), "end mtp t2");
            cuda_ok(cudaGraphInstantiate(&mtp_graph_t2_, g, nullptr, nullptr, 0), "inst mtp t2");
            cudaGraphDestroy(g);
        }
    }
    graphs_ready_ = true;
}

void Qwen35MoeModel::warmup_decode_graphs() {
    if (!use_cuda_graphs_ || graphs_ready_) return;
    capture_decode_graphs();
}

void Qwen35MoeModel::forward(const std::vector<int32_t> & tokens,
                             const std::vector<int32_t> & positions,
                             std::vector<float> & out_logits) {
    const int T = static_cast<int>(tokens.size());
    if (T == 0 || T != static_cast<int>(positions.size()) || T > max_batch_) {
        throw std::runtime_error("Qwen35MoeModel::forward invalid batch size");
    }
    int max_p = 0;
    for (int32_t p : positions) if (p > max_p) max_p = p;
    const int kv_len_max = max_p + 1;
    const size_t xfer_b = static_cast<size_t>(T) * cfg_.n_embd * sizeof(float);

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "fwd setDevice");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, tokens.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "fwd tok");
        } else {
            cuda_ok(cudaStreamSynchronize(runtimes_[gi - 1].stream), "fwd prev sync");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_cur, rt.device, runtimes_[gi - 1].d_cur, runtimes_[gi - 1].device, xfer_b, rt.stream), "fwd peer");
        }
        cuda_ok(cudaMemcpyAsync(rt.d_pos, positions.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "fwd pos");
        run_gpu_segment(gi, T, false, false, kv_len_max);
    }

    auto & rtl = runtimes_.back();
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaSetDevice(rtl.device), "fwd readout");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), rtl.d_logits, static_cast<size_t>(cfg_.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "fwd logits");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "fwd sync");
}

int32_t Qwen35MoeModel::forward_decode_greedy(int32_t token, int32_t pos) {
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();
    const size_t xfer_b = static_cast<size_t>(cfg_.n_embd) * sizeof(float);

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "dec setDevice");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "dec tok");
        } else {
            cuda_ok(cudaStreamWaitEvent(rt.stream, runtimes_[gi - 1].done_event, 0), "dec wait");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_cur, rt.device, runtimes_[gi - 1].d_cur, runtimes_[gi - 1].device, xfer_b, rt.stream), "dec peer");
        }
        cuda_ok(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "dec pos");
        if (graphs_ready_ && rt.decode_graph_t1) cuda_ok(cudaGraphLaunch(rt.decode_graph_t1, rt.stream), "launch t1");
        else run_gpu_segment(gi, 1, true, false, pos + 1);
        if (gi + 1 < runtimes_.size()) cuda_ok(cudaEventRecord(rt.done_event, rt.stream), "dec rec");
    }

    auto & rtl = runtimes_.back();
    int32_t next_tok = 0;
    cuda_ok(cudaSetDevice(rtl.device), "dec readout");
    cuda_ok(cudaMemcpyAsync(&next_tok, rtl.d_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, rtl.stream), "dec argmax");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "dec sync");
    return next_tok;
}

void Qwen35MoeModel::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    forward_decode_greedy(token, pos);
    auto & rtl = runtimes_.back();
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaSetDevice(rtl.device), "dec logits dev");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), rtl.d_logits, static_cast<size_t>(cfg_.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "dec logits");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "dec logits sync");
}

int32_t Qwen35MoeModel::mtp_prefill_chunk(const std::vector<int32_t> & next_tokens,
                                          const std::vector<int32_t> & positions) {
    if (!has_mtp_) return 0;
    const int T = static_cast<int>(next_tokens.size());
    int max_p = 0;
    for (int32_t p : positions) if (p > max_p) max_p = p;

    embed_tokens_to_last_dev(next_tokens.data(), T);
    auto & rtl = runtimes_.back();
    cuda_ok(cudaSetDevice(rtl.device), "mtp pp dev");
    cuda_ok(cudaMemcpyAsync(rtl.d_mtp_pos, positions.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rtl.stream), "mtp pp pos");
    run_mtp_layer(T, true, max_p + 1);
    int32_t draft = 0;
    cuda_ok(cudaMemcpyAsync(&draft, rtl.d_mtp_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, rtl.stream), "mtp pp argmax");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "mtp pp sync");
    return draft;
}

void Qwen35MoeModel::mtp_prefill_chunk_logits(const std::vector<int32_t> & next_tokens,
                                              const std::vector<int32_t> & positions,
                                              std::vector<float> & out_logits) {
    if (!has_mtp_) return;
    const int T = static_cast<int>(next_tokens.size());
    int max_p = 0;
    for (int32_t p : positions) if (p > max_p) max_p = p;

    embed_tokens_to_last_dev(next_tokens.data(), T);
    auto & rtl = runtimes_.back();
    cuda_ok(cudaSetDevice(rtl.device), "mtp ppl dev");
    cuda_ok(cudaMemcpyAsync(rtl.d_mtp_pos, positions.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rtl.stream), "mtp ppl pos");
    run_mtp_layer(T, false, max_p + 1);
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaMemcpyAsync(out_logits.data(), rtl.d_logits, static_cast<size_t>(cfg_.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "mtp ppl logits");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "mtp ppl sync");
}

void Qwen35MoeModel::decode_mtp_greedy_step(int32_t & cur_tok,
                                            int32_t & draft_tok,
                                            int32_t & cur_pos,
                                            std::vector<int32_t> & out_tokens,
                                            MtpStats & stats) {
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();
    const int32_t h_tok[2] = {cur_tok, draft_tok};
    const int32_t h_pos[2] = {cur_pos, cur_pos + 1};
    const size_t xfer_b = 2ULL * cfg_.n_embd * sizeof(float);

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "mtp dec dev");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, h_tok, sizeof(h_tok), cudaMemcpyHostToDevice, rt.stream), "mtp dec tok");
        } else {
            cuda_ok(cudaStreamWaitEvent(rt.stream, runtimes_[gi - 1].done_event, 0), "mtp dec wait");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_cur, rt.device, runtimes_[gi - 1].d_cur, runtimes_[gi - 1].device, xfer_b, rt.stream), "mtp dec peer");
        }
        cuda_ok(cudaMemcpyAsync(rt.d_pos, h_pos, sizeof(h_pos), cudaMemcpyHostToDevice, rt.stream), "mtp dec pos");
        if (graphs_ready_ && rt.decode_graph_t2) cuda_ok(cudaGraphLaunch(rt.decode_graph_t2, rt.stream), "launch t2");
        else run_gpu_segment(gi, 2, true, true, cur_pos + 2);
        if (gi + 1 < runtimes_.size()) cuda_ok(cudaEventRecord(rt.done_event, rt.stream), "mtp dec rec");
    }

    auto & rtl = runtimes_.back();
    int32_t pred[2] = {0, 0};
    cuda_ok(cudaSetDevice(rtl.device), "mtp dec readout");
    cuda_ok(cudaMemcpyAsync(pred, rtl.d_argmax, sizeof(pred), cudaMemcpyDeviceToHost, rtl.stream), "mtp pred");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "mtp dec sync");

    stats.steps++;
    out_tokens.clear();
    if (pred[0] == draft_tok) {
        stats.accepted++;
        out_tokens.push_back(draft_tok);
        out_tokens.push_back(pred[1]);

        const int32_t next_t[2] = {draft_tok, pred[1]};
        embed_tokens_to_last_dev(next_t, 2);
        cuda_ok(cudaSetDevice(rtl.device), "mtp acc dev");
        cuda_ok(cudaMemcpyAsync(rtl.d_mtp_pos, h_pos, sizeof(h_pos), cudaMemcpyHostToDevice, rtl.stream), "mtp acc pos");
        if (graphs_ready_ && mtp_graph_t2_) cuda_ok(cudaGraphLaunch(mtp_graph_t2_, rtl.stream), "launch mtp t2");
        else run_mtp_layer(2, true, cur_pos + 2);

        int32_t next_draft = 0;
        cuda_ok(cudaMemcpyAsync(&next_draft, rtl.d_mtp_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, rtl.stream), "mtp acc draft");
        cuda_ok(cudaStreamSynchronize(rtl.stream), "mtp acc sync");

        cur_tok   = pred[1];
        draft_tok = next_draft;
        cur_pos  += 2;
    } else {
        out_tokens.push_back(pred[0]);
        rollback_gdn_states();

        embed_tokens_to_last_dev(&pred[0], 1);
        cuda_ok(cudaSetDevice(rtl.device), "mtp rej dev");
        cuda_ok(cudaMemcpyAsync(rtl.d_mtp_pos, &h_pos[0], sizeof(int32_t), cudaMemcpyHostToDevice, rtl.stream), "mtp rej pos");
        if (graphs_ready_ && mtp_graph_t1_) cuda_ok(cudaGraphLaunch(mtp_graph_t1_, rtl.stream), "launch mtp t1");
        else run_mtp_layer(1, true, cur_pos + 1);

        int32_t next_draft = 0;
        cuda_ok(cudaMemcpyAsync(&next_draft, rtl.d_mtp_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, rtl.stream), "mtp rej draft");
        cuda_ok(cudaStreamSynchronize(rtl.stream), "mtp rej sync");

        cur_tok   = pred[0];
        draft_tok = next_draft;
        cur_pos  += 1;
    }
}

void Qwen35MoeModel::decode_mtp_sample_step(int32_t & cur_tok,
                                            int32_t & draft_tok,
                                            Sampler::SparseDist & draft_q,
                                            int32_t & cur_pos,
                                            Sampler & sampler,
                                            std::vector<int32_t> & out_tokens,
                                            MtpStats & stats) {
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();
    const int32_t h_tok[2] = {cur_tok, draft_tok};
    const int32_t h_pos[2] = {cur_pos, cur_pos + 1};
    const size_t xfer_b = 2ULL * cfg_.n_embd * sizeof(float);

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "mtp smp dev");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, h_tok, sizeof(h_tok), cudaMemcpyHostToDevice, rt.stream), "mtp smp tok");
        } else {
            cuda_ok(cudaStreamWaitEvent(rt.stream, runtimes_[gi - 1].done_event, 0), "mtp smp wait");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_cur, rt.device, runtimes_[gi - 1].d_cur, runtimes_[gi - 1].device, xfer_b, rt.stream), "mtp smp peer");
        }
        cuda_ok(cudaMemcpyAsync(rt.d_pos, h_pos, sizeof(h_pos), cudaMemcpyHostToDevice, rt.stream), "mtp smp pos");
        if (graphs_ready_ && rt.decode_graph_t2) cuda_ok(cudaGraphLaunch(rt.decode_graph_t2, rt.stream), "launch t2");
        else run_gpu_segment(gi, 2, true, true, cur_pos + 2);
        if (gi + 1 < runtimes_.size()) cuda_ok(cudaEventRecord(rt.done_event, rt.stream), "mtp smp rec");
    }

    auto & rtl = runtimes_.back();
    std::vector<float> h_logits(2ULL * cfg_.n_vocab);
    cuda_ok(cudaSetDevice(rtl.device), "mtp smp readout");
    cuda_ok(cudaMemcpyAsync(h_logits.data(), rtl.d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "mtp smp logits");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "mtp smp sync");

    auto p0 = sampler.filtered_distribution(h_logits.data(), cfg_.n_vocab);
    int32_t resampled = 0;
    const bool accepted = sampler.verify_speculative(draft_tok, p0, draft_q, resampled);

    stats.steps++;
    out_tokens.clear();
    std::vector<float> mtp_logits(cfg_.n_vocab);

    if (accepted) {
        stats.accepted++;
        auto p1 = sampler.filtered_distribution(h_logits.data() + cfg_.n_vocab, cfg_.n_vocab);
        const int32_t bonus_tok = sampler.sample_dist(p1);
        out_tokens.push_back(draft_tok);
        out_tokens.push_back(bonus_tok);

        const int32_t next_t[2] = {draft_tok, bonus_tok};
        embed_tokens_to_last_dev(next_t, 2);
        cuda_ok(cudaSetDevice(rtl.device), "mtp smp acc dev");
        cuda_ok(cudaMemcpyAsync(rtl.d_mtp_pos, h_pos, sizeof(h_pos), cudaMemcpyHostToDevice, rtl.stream), "mtp smp acc pos");
        if (graphs_ready_ && mtp_graph_t2_) cuda_ok(cudaGraphLaunch(mtp_graph_t2_, rtl.stream), "launch mtp t2");
        else run_mtp_layer(2, true, cur_pos + 2);

        cuda_ok(cudaMemcpyAsync(mtp_logits.data(), rtl.d_logits, cfg_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "mtp smp acc lg");
        cuda_ok(cudaStreamSynchronize(rtl.stream), "mtp smp acc sync");

        draft_q   = sampler.filtered_distribution(mtp_logits.data(), cfg_.n_vocab);
        cur_tok   = bonus_tok;
        draft_tok = sampler.sample_dist(draft_q);
        cur_pos  += 2;
    } else {
        out_tokens.push_back(resampled);
        rollback_gdn_states();

        embed_tokens_to_last_dev(&resampled, 1);
        cuda_ok(cudaSetDevice(rtl.device), "mtp smp rej dev");
        cuda_ok(cudaMemcpyAsync(rtl.d_mtp_pos, &h_pos[0], sizeof(int32_t), cudaMemcpyHostToDevice, rtl.stream), "mtp smp rej pos");
        if (graphs_ready_ && mtp_graph_t1_) cuda_ok(cudaGraphLaunch(mtp_graph_t1_, rtl.stream), "launch mtp t1");
        else run_mtp_layer(1, true, cur_pos + 1);

        cuda_ok(cudaMemcpyAsync(mtp_logits.data(), rtl.d_logits, cfg_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "mtp smp rej lg");
        cuda_ok(cudaStreamSynchronize(rtl.stream), "mtp smp rej sync");

        draft_q   = sampler.filtered_distribution(mtp_logits.data(), cfg_.n_vocab);
        cur_tok   = resampled;
        draft_tok = sampler.sample_dist(draft_q);
        cur_pos  += 1;
    }
}

} // namespace rookery
