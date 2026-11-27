#include "models/sensenova_loop.h"
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
    cuda_ok(cudaSetDevice(dev), "sensenova dev_alloc setDevice");
    T * p = nullptr;
    cuda_ok(cudaMalloc(&p, n * sizeof(T)), "sensenova cudaMalloc");
    return p;
}

std::vector<float> compute_yarn_inv_freq(int rope_dim, float base, float factor, int orig_ctx, float & out_mscale) {
    const int half = rope_dim / 2;
    std::vector<float> inv_freq(half);
    if (factor <= 1.0f) {
        out_mscale = 1.0f;
        for (int i = 0; i < half; ++i) {
            inv_freq[i] = static_cast<float>(std::pow(static_cast<double>(base), -2.0 * i / rope_dim));
        }
        return inv_freq;
    }

    constexpr double kPi       = 3.14159265358979323846;
    constexpr double beta_fast = 32.0;
    constexpr double beta_slow = 1.0;

    auto find_corr_dim = [&](double num_rot) -> double {
        return (rope_dim * std::log(static_cast<double>(orig_ctx) / (num_rot * 2.0 * kPi))) /
               (2.0 * std::log(static_cast<double>(base)));
    };

    const double low  = std::max(std::floor(find_corr_dim(beta_fast)), 0.0);
    const double high = std::min(std::ceil(find_corr_dim(beta_slow)), static_cast<double>(rope_dim - 1));
    const double span = (std::abs(high - low) < 1e-9) ? 0.001 : (high - low);

    out_mscale = static_cast<float>(0.1 * std::log(static_cast<double>(factor)) + 1.0);
    for (int i = 0; i < half; ++i) {
        const double freq_extrap = std::pow(static_cast<double>(base), -2.0 * i / rope_dim);
        const double freq_interp = freq_extrap / static_cast<double>(factor);
        double ramp = (static_cast<double>(i) - low) / span;
        ramp = std::max(0.0, std::min(1.0, ramp));
        inv_freq[i] = static_cast<float>(freq_interp * ramp + freq_extrap * (1.0 - ramp));
    }
    return inv_freq;
}

} // namespace

void SenseNovaLoopModel::invalidate_graphs() noexcept {
    for (size_t si = 0; si < segments_.size(); ++si) {
        cudaSetDevice(runtimes_[segments_[si].gpu_idx].device);
        if (si < seg_graphs_t1_.size() && seg_graphs_t1_[si]) {
            cudaGraphExecDestroy(seg_graphs_t1_[si]);
            seg_graphs_t1_[si] = nullptr;
        }
        if (si < seg_graphs_t2_.size() && seg_graphs_t2_[si]) {
            cudaGraphExecDestroy(seg_graphs_t2_[si]);
            seg_graphs_t2_[si] = nullptr;
        }
    }
    if (!runtimes_.empty()) {
        cudaSetDevice(runtimes_.back().device);
        if (mtp_graph_t1_) { cudaGraphExecDestroy(mtp_graph_t1_); mtp_graph_t1_ = nullptr; }
        if (mtp_graph_t2_) { cudaGraphExecDestroy(mtp_graph_t2_); mtp_graph_t2_ = nullptr; }
    }
    graphs_ready_ = false;
}

SenseNovaLoopModel::~SenseNovaLoopModel() {
    invalidate_graphs();
    auto freep = [](void * p) { if (p) cudaFree(p); };
    for (int v = 0; v < cfg_.n_visits; ++v) {
        if (v >= static_cast<int>(cfg_.plan.size())) break;
        const int p = cfg_.plan[v];
        if (p < static_cast<int>(layers_.size())) {
            cudaSetDevice(layers_[p].device);
            if (v < static_cast<int>(gdn_conv_.size()))       freep(gdn_conv_[v]);
            if (v < static_cast<int>(gdn_state_.size()))      freep(gdn_state_[v]);
            if (v < static_cast<int>(gdn_conv_snap_.size()))  freep(gdn_conv_snap_[v]);
            if (v < static_cast<int>(gdn_state_snap_.size())) freep(gdn_state_snap_[v]);
            if (v < static_cast<int>(k_cache_.size()))        freep(k_cache_[v]);
            if (v < static_cast<int>(v_cache_.size()))        freep(v_cache_[v]);
        }
    }
    if (!runtimes_.empty()) {
        cudaSetDevice(runtimes_.back().device);
        freep(mtp_k_cache_);
        freep(mtp_v_cache_);
    }
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        freep(rt.d_inv_freq);
        freep(rt.d_cur); freep(rt.d_res); freep(rt.d_mhc_ctrl);
        freep(rt.d_x_norm); freep(rt.d_sub_out);
        freep(rt.d_qkvz); freep(rt.d_qkv_conv);
        freep(rt.d_ba); freep(rt.d_alpha); freep(rt.d_beta); freep(rt.d_gdn_out);
        freep(rt.d_q_full); freep(rt.d_q); freep(rt.d_gate);
        freep(rt.d_k); freep(rt.d_v); freep(rt.d_attn_out);
        freep(rt.d_ff_gate); freep(rt.d_ff_up);
        freep(rt.d_tokens); freep(rt.d_pos); freep(rt.d_logits); freep(rt.d_argmax);
        freep(rt.d_h_nextn); freep(rt.d_mtp_e); freep(rt.d_mtp_eh);
        freep(rt.d_mtp_cur); freep(rt.d_mtp_pos); freep(rt.d_mtp_argmax);
        freep(rt.d_scratch);
        if (rt.done_event) cudaEventDestroy(rt.done_event);
        if (rt.cublas)     cublasDestroy(rt.cublas);
        if (rt.stream)     cudaStreamDestroy(rt.stream);
    }
}

void SenseNovaLoopModel::load(const std::string & gguf_path,
                              const std::vector<int> & gpu_ids,
                              const std::vector<float> & split_weights,
                              int max_ctx,
                              bool kv_q8_0,
                              bool load_mtp) {
    if (gpu_ids.empty()) throw std::runtime_error("SenseNovaLoopModel::load requires >= 1 GPU");

    const auto t0 = std::chrono::steady_clock::now();
    auto sg = ShardedGguf::open(gguf_path);
    const std::string arch = sg.get_str("general.architecture");
    if (arch != "sensenova_loop") {
        throw std::runtime_error("Expected sensenova_loop GGUF, got: " + arch);
    }
    const std::string pfx = arch + ".";

    const int block_count      = static_cast<int>(sg.get_u64(pfx + "block_count", 57));
    cfg_.n_nextn               = static_cast<int>(sg.get_u64(pfx + "nextn_predict_layers", 0));
    cfg_.loop_start            = static_cast<int>(sg.get_u64(pfx + "loop.start", 4));
    cfg_.loop_end              = static_cast<int>(sg.get_u64(pfx + "loop.end", 28));
    cfg_.loop_count            = static_cast<int>(sg.get_u64(pfx + "loop.count", 2));
    cfg_.n_visits              = block_count - cfg_.n_nextn;
    const int extra_visits     = (cfg_.loop_count - 1) * (cfg_.loop_end - cfg_.loop_start);
    cfg_.n_phys                = cfg_.n_visits - extra_visits;

    cfg_.mhc_start             = static_cast<int>(sg.get_u64(pfx + "mhc.start", cfg_.loop_start));
    cfg_.mhc_end               = static_cast<int>(sg.get_u64(pfx + "mhc.end", cfg_.loop_end));
    cfg_.mhc_streams           = static_cast<int>(sg.get_u64(pfx + "mhc.streams", 2));
    cfg_.mhc_iterations        = static_cast<int>(sg.get_u64(pfx + "mhc.iterations", 20));
    cfg_.mhc_fixed_identity    = sg.get_bool(pfx + "mhc.fixed_identity", false);
    cfg_.mhc_pre_scale         = static_cast<float>(sg.get_f64(pfx + "mhc.pre_scale", 1.0));
    if (cfg_.mhc_streams != 2) {
        throw std::runtime_error("SenseNovaLoopModel currently supports mhc.streams == 2");
    }

    cfg_.n_embd                = static_cast<int>(sg.get_u64(pfx + "embedding_length", 4096));
    cfg_.n_ff                  = static_cast<int>(sg.get_u64(pfx + "feed_forward_length", 12288));
    cfg_.n_vocab               = static_cast<int>(sg.get_u64(pfx + "vocab_size", 0));
    if (cfg_.n_vocab == 0) {
        auto loc = sg.find_tensor("token_embd.weight");
        if (loc.info) cfg_.n_vocab = static_cast<int>(loc.info->ne[1]);
    }
    cfg_.n_head                = static_cast<int>(sg.get_u64(pfx + "attention.head_count", 16));
    cfg_.n_head_kv             = static_cast<int>(sg.get_u64(pfx + "attention.head_count_kv", 4));
    cfg_.head_dim              = static_cast<int>(sg.get_u64(pfx + "attention.key_length", 256));
    cfg_.rope_dim              = static_cast<int>(sg.get_u64(pfx + "rope.dimension_count", 64));
    cfg_.rope_freq_base        = static_cast<float>(sg.get_f64(pfx + "rope.freq_base", 10000000.0));
    const std::string rope_sc  = sg.get_str(pfx + "rope.scaling.type", "none");
    cfg_.rope_yarn_factor      = (rope_sc == "yarn") ? static_cast<float>(sg.get_f64(pfx + "rope.scaling.factor", 1.0)) : 1.0f;
    cfg_.rope_orig_ctx         = static_cast<int>(sg.get_u64(pfx + "rope.scaling.original_context_length", 262144));
    cfg_.rms_eps               = static_cast<float>(sg.get_f64(pfx + "attention.layer_norm_rms_epsilon", 1e-6));

    cfg_.ssm_d_conv            = static_cast<int>(sg.get_u64(pfx + "ssm.conv_kernel", 4));
    cfg_.ssm_d_state           = static_cast<int>(sg.get_u64(pfx + "ssm.state_size", 128));
    cfg_.ssm_n_group           = static_cast<int>(sg.get_u64(pfx + "ssm.group_count", 16));
    cfg_.ssm_dt_rank           = static_cast<int>(sg.get_u64(pfx + "ssm.time_step_rank", 32));
    cfg_.ssm_d_inner           = static_cast<int>(sg.get_u64(pfx + "ssm.inner_size", 4096));
    cfg_.full_attn_interval    = static_cast<int>(sg.get_u64(pfx + "full_attention_interval", 4));
    cfg_.max_ctx               = max_ctx;
    cfg_.kv_q8_0               = kv_q8_0;

    // Build execution plan: [0..start) + [start..end)*count + [end..n_phys)
    cfg_.plan.clear();
    for (int p = 0; p < cfg_.loop_start; ++p) cfg_.plan.push_back(p);
    for (int c = 0; c < cfg_.loop_count; ++c) {
        for (int p = cfg_.loop_start; p < cfg_.loop_end; ++p) cfg_.plan.push_back(p);
    }
    for (int p = cfg_.loop_end; p < cfg_.n_phys; ++p) cfg_.plan.push_back(p);
    cfg_.n_visits = static_cast<int>(cfg_.plan.size());

    cfg_.phys_is_recr.resize(cfg_.n_phys);
    for (int p = 0; p < cfg_.n_phys; ++p) {
        cfg_.phys_is_recr[p] = ((p + 1) % cfg_.full_attn_interval != 0);
    }

    auto gguf_blk_prefix = [&](int phys) -> std::string {
        const int shifted = (phys < cfg_.loop_end) ? phys : (phys + extra_visits);
        const std::string sp = "blk." + std::to_string(shifted) + ".";
        if (sg.find_tensor(sp + "attn_norm.weight").info != nullptr) return sp;
        return "blk." + std::to_string(phys) + ".";
    };

    const std::string mtp_pfx = "blk." + std::to_string(cfg_.n_visits) + ".";
    has_mtp_ = load_mtp && cfg_.n_nextn >= 1 && (sg.find_tensor(mtp_pfx + "nextn.eh_proj.weight").info != nullptr);

    // Compute YaRN inverse frequencies on host
    const std::vector<float> h_inv_freq = compute_yarn_inv_freq(
        cfg_.rope_dim, cfg_.rope_freq_base, cfg_.rope_yarn_factor, cfg_.rope_orig_ctx, cfg_.yarn_mscale);

    // Partition 32 physical layers across GPUs
    std::vector<float> sw = split_weights;
    if (sw.size() != gpu_ids.size()) {
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
    int cur_p = 0;
    for (size_t g = 0; g < gpu_ids.size(); ++g) {
        int count = (g + 1 == gpu_ids.size())
                        ? (cfg_.n_phys - cur_p)
                        : static_cast<int>(std::round(cfg_.n_phys * (sw[g] / sw_sum)));
        count = std::max(0, std::min(count, cfg_.n_phys - cur_p));
        runtimes_[g].device     = gpu_ids[g];
        runtimes_[g].phys_start = cur_p;
        runtimes_[g].phys_end   = cur_p + count;
        cur_p += count;
    }

    for (size_t g = 0; g < runtimes_.size(); ++g) {
        for (size_t h = 0; h < runtimes_.size(); ++h) {
            if (g == h) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, runtimes_[g].device, runtimes_[h].device);
            if (can) {
                cudaSetDevice(runtimes_[g].device);
                cudaDeviceEnablePeerAccess(runtimes_[h].device, 0);
                cudaGetLastError();
            }
        }
    }

    layers_.resize(cfg_.n_phys);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        for (int p = runtimes_[gi].phys_start; p < runtimes_[gi].phys_end; ++p) {
            layers_[p].device = runtimes_[gi].device;
        }
    }

    // Build contiguous visit segments grouped by GPU index!
    // Single GPU or pre-loop split (e.g. [0..4) on GPU1, [4..32) on GPU0) yields 1-2 segments total!
    auto gpu_idx_of_phys = [&](int p) -> size_t {
        for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
            if (p >= runtimes_[gi].phys_start && p < runtimes_[gi].phys_end) return gi;
        }
        return runtimes_.size() - 1;
    };
    segments_.clear();
    for (int v = 0; v < cfg_.n_visits; ) {
        const size_t gi = gpu_idx_of_phys(cfg_.plan[v]);
        int v_end = v + 1;
        while (v_end < cfg_.n_visits && gpu_idx_of_phys(cfg_.plan[v_end]) == gi) ++v_end;
        segments_.push_back(VisitSegment{gi, v, v_end});
        v = v_end;
    }
    seg_graphs_t1_.assign(segments_.size(), nullptr);
    seg_graphs_t2_.assign(segments_.size(), nullptr);

    gdn_conv_.assign(cfg_.n_visits, nullptr);
    gdn_state_.assign(cfg_.n_visits, nullptr);
    gdn_conv_snap_.assign(cfg_.n_visits, nullptr);
    gdn_state_snap_.assign(cfg_.n_visits, nullptr);
    k_cache_.assign(cfg_.n_visits, nullptr);
    v_cache_.assign(cfg_.n_visits, nullptr);

    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner; // 8192
    const int d_kv     = cfg_.n_head_kv * cfg_.head_dim;                             // 1024
    const size_t kv_bytes = kv_q8_0
        ? static_cast<size_t>(cfg_.max_ctx) * (d_kv / 32) * sizeof(BlockQ8_0)
        : static_cast<size_t>(cfg_.max_ctx) * d_kv * sizeof(uint16_t);

    // Parallel weight upload
    std::vector<std::thread> load_threads;
    std::vector<std::exception_ptr> load_errs(runtimes_.size(), nullptr);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        load_threads.emplace_back([&, gi]() {
            try {
                const int dev = runtimes_[gi].device;
                cuda_ok(cudaSetDevice(dev), "sensenova load setDevice");
                if (gi == 0) {
                    tok_embd_ = pool_.upload(sg, "token_embd.weight", dev, false);
                }
                if (gi + 1 == runtimes_.size()) {
                    output_norm_ = pool_.upload(sg, "output_norm.weight", dev, false);
                    output_      = pool_.upload(sg, "output.weight",      dev, false, false);
                    if (!output_.valid()) output_ = pool_.upload(sg, "token_embd.weight", dev, false);
                }

                for (int il = runtimes_[gi].phys_start; il < runtimes_[gi].phys_end; ++il) {
                    auto & L = layers_[il];
                    const std::string p = gguf_blk_prefix(il);
                    L.device  = dev;
                    L.is_recr = cfg_.phys_is_recr[il];
                    L.has_mhc = (il >= cfg_.mhc_start && il < cfg_.mhc_end);

                    L.attn_norm      = pool_.upload(sg, p + "attn_norm.weight",           dev, false);
                    L.post_attn_norm = pool_.upload(sg, p + "post_attention_norm.weight", dev, false);

                    if (L.has_mhc) {
                        L.hc_attn_w = pool_.upload(sg, p + "hc_attn_fn.weight", dev, false);
                        L.hc_attn_b = pool_.upload(sg, p + "hc_attn_fn.bias",   dev, false);
                        L.hc_ffn_w  = pool_.upload(sg, p + "hc_ffn_fn.weight",  dev, false);
                        L.hc_ffn_b  = pool_.upload(sg, p + "hc_ffn_fn.bias",    dev, false);
                    }

                    if (L.is_recr) {
                        L.wqkv_gate      = pool_.upload(sg, p + "attn_qkv_gate.weight",  dev, false, false);
                        L.ssm_beta_alpha = pool_.upload(sg, p + "ssm_beta_alpha.weight", dev, false, false);
                        if (!L.wqkv_gate.valid()) {
                            L.wqkv  = pool_.upload(sg, p + "attn_qkv.weight",  dev, false);
                            L.wgate = pool_.upload(sg, p + "attn_gate.weight", dev, false);
                        }
                        if (!L.ssm_beta_alpha.valid()) {
                            L.ssm_alpha = pool_.upload(sg, p + "ssm_alpha.weight", dev, false);
                            L.ssm_beta  = pool_.upload(sg, p + "ssm_beta.weight",  dev, false);
                        }
                        L.ssm_conv1d = pool_.upload(sg, p + "ssm_conv1d.weight", dev, false);
                        L.ssm_dt     = pool_.upload(sg, p + "ssm_dt.bias",       dev, false);
                        L.ssm_a      = pool_.upload(sg, p + "ssm_a",             dev, false);
                        L.ssm_norm   = pool_.upload(sg, p + "ssm_norm.weight",   dev, false);
                        L.ssm_out    = pool_.upload(sg, p + "ssm_out.weight",    dev, false);
                    } else {
                        L.wq          = pool_.upload(sg, p + "attn_q.weight",      dev, false);
                        L.wk          = pool_.upload(sg, p + "attn_k.weight",      dev, false);
                        L.wv          = pool_.upload(sg, p + "attn_v.weight",      dev, false);
                        L.wo          = pool_.upload(sg, p + "attn_output.weight", dev, false);
                        L.attn_q_norm = pool_.upload(sg, p + "attn_q_norm.weight", dev, false);
                        L.attn_k_norm = pool_.upload(sg, p + "attn_k_norm.weight", dev, false);
                    }

                    L.ffn_gate = pool_.upload(sg, p + "ffn_gate.weight", dev, false);
                    L.ffn_up   = pool_.upload(sg, p + "ffn_up.weight",   dev, false);
                    L.ffn_down = pool_.upload(sg, p + "ffn_down.weight", dev, false);
                }

                // Allocate logical visit KV / GDN cache slots belonging to this GPU
                const size_t conv_n = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim;
                const size_t ssm_n  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state;
                for (int v = 0; v < cfg_.n_visits; ++v) {
                    const int phys = cfg_.plan[v];
                    if (phys < runtimes_[gi].phys_start || phys >= runtimes_[gi].phys_end) continue;
                    if (cfg_.phys_is_recr[phys]) {
                        gdn_conv_[v]  = dev_alloc<float>(dev, conv_n);
                        gdn_state_[v] = dev_alloc<float>(dev, ssm_n);
                        if (has_mtp_) {
                            gdn_conv_snap_[v]  = dev_alloc<float>(dev, conv_n);
                            gdn_state_snap_[v] = dev_alloc<float>(dev, ssm_n);
                        }
                    } else {
                        cuda_ok(cudaMalloc(&k_cache_[v], kv_bytes), "sensenova malloc k_cache");
                        cuda_ok(cudaMalloc(&v_cache_[v], kv_bytes), "sensenova malloc v_cache");
                    }
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
                    mtp_.ffn_gate         = pool_.upload(sg, mtp_pfx + "ffn_gate.weight",            dev, false);
                    mtp_.ffn_up           = pool_.upload(sg, mtp_pfx + "ffn_up.weight",              dev, false);
                    mtp_.ffn_down         = pool_.upload(sg, mtp_pfx + "ffn_down.weight",            dev, false);

                    cuda_ok(cudaMalloc(&mtp_k_cache_, kv_bytes), "sensenova malloc mtp_k");
                    cuda_ok(cudaMalloc(&mtp_v_cache_, kv_bytes), "sensenova malloc mtp_v");
                }
                cuda_ok(cudaDeviceSynchronize(), "sensenova load sync");
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
        cuda_ok(cudaSetDevice(dev), "sensenova rt setDevice");
        cuda_ok(cudaStreamCreate(&rt.stream), "sensenova stream");
        cuda_ok(cudaEventCreateWithFlags(&rt.done_event, cudaEventDisableTiming), "sensenova event");
        if (cublasCreate(&rt.cublas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        if (device_has_tensor_cores(dev)) cublasSetMathMode(rt.cublas, CUBLAS_TENSOR_OP_MATH);

        rt.d_inv_freq = dev_alloc<float>(dev, h_inv_freq.size());
        cuda_ok(cudaMemcpy(rt.d_inv_freq, h_inv_freq.data(), h_inv_freq.size() * sizeof(float), cudaMemcpyHostToDevice), "copy inv_freq");

        rt.d_cur      = dev_alloc<float>(dev, B * H);
        rt.d_res      = dev_alloc<float>(dev, B * 2 * H);
        rt.d_mhc_ctrl = dev_alloc<float>(dev, B * 6);
        rt.d_x_norm   = dev_alloc<float>(dev, B * H);
        rt.d_sub_out  = dev_alloc<float>(dev, B * H);

        rt.d_qkvz     = dev_alloc<float>(dev, B * (conv_dim + cfg_.ssm_d_inner)); // 12288
        rt.d_qkv_conv = dev_alloc<float>(dev, B * conv_dim);                     // 8192
        rt.d_ba       = dev_alloc<float>(dev, B * 2 * cfg_.ssm_dt_rank);         // 64
        rt.d_alpha    = dev_alloc<float>(dev, B * cfg_.ssm_dt_rank);
        rt.d_beta     = dev_alloc<float>(dev, B * cfg_.ssm_dt_rank);
        rt.d_gdn_out  = dev_alloc<float>(dev, B * cfg_.ssm_d_inner);

        rt.d_q_full   = dev_alloc<float>(dev, B * d_q * 2);
        rt.d_q        = dev_alloc<float>(dev, B * d_q);
        rt.d_gate     = dev_alloc<float>(dev, B * d_q);
        rt.d_k        = dev_alloc<float>(dev, B * d_kv);
        rt.d_v        = dev_alloc<float>(dev, B * d_kv);
        rt.d_attn_out = dev_alloc<float>(dev, B * d_q);

        rt.d_ff_gate  = dev_alloc<float>(dev, B * cfg_.n_ff);
        rt.d_ff_up    = dev_alloc<float>(dev, B * cfg_.n_ff);

        rt.d_pos      = dev_alloc<int32_t>(dev, B);
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
        cuda_ok(cudaMalloc(&rt.d_scratch, scratch_b), "sensenova scratch");
        rt.scratch_bytes = scratch_b;
    }

    reset_state();

    const auto t1 = std::chrono::steady_clock::now();
    std::printf("[SenseNova-Loop] Loaded (%d phys -> %d visits, loop=[%d..%d)x%d, mHC=%d-stream, MTP=%d, YaRN=%.1fx[m=%.4f]) in %.2f s across %zu GPU(s):",
                cfg_.n_phys, cfg_.n_visits, cfg_.loop_start, cfg_.loop_end, cfg_.loop_count,
                cfg_.mhc_streams, has_mtp_ ? 1 : 0, cfg_.rope_yarn_factor, cfg_.yarn_mscale,
                std::chrono::duration<double>(t1 - t0).count(), runtimes_.size());
    for (const auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        size_t f = 0, tot = 0;
        cudaMemGetInfo(&f, &tot);
        std::printf(" GPU%d[p%d..%d](%.2f GiB w, %.2f GiB free)",
                    rt.device, rt.phys_start, rt.phys_end - 1,
                    pool_.bytes_on_device(rt.device) / (1024.0 * 1024.0 * 1024.0),
                    f / (1024.0 * 1024.0 * 1024.0));
    }
    std::printf(" | ctx=%d (KV=%s)\n", cfg_.max_ctx, kv_q8_0 ? "Q8_0" : "F16");
}

void SenseNovaLoopModel::reset_state() {
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const size_t conv_b = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim * sizeof(float);
    const size_t ssm_b  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state * sizeof(float);
    for (int v = 0; v < cfg_.n_visits; ++v) {
        const int phys = cfg_.plan[v];
        if (!layers_[phys].is_recr) continue;
        cuda_ok(cudaSetDevice(layers_[phys].device), "sensenova reset setDevice");
        cuda_ok(cudaMemset(gdn_conv_[v],  0, conv_b), "zero gdn_conv");
        cuda_ok(cudaMemset(gdn_state_[v], 0, ssm_b),  "zero gdn_state");
    }
}

void SenseNovaLoopModel::save_gdn_checkpoint(GdnCheckpoint & out, int pos) const {
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const size_t conv_n = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim;
    const size_t ssm_n  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state;
    out.pos = pos;
    out.conv.resize(cfg_.n_visits);
    out.ssm.resize(cfg_.n_visits);
    for (int v = 0; v < cfg_.n_visits; ++v) {
        const int phys = cfg_.plan[v];
        if (!layers_[phys].is_recr) continue;
        cuda_ok(cudaSetDevice(layers_[phys].device), "sensenova save_cp setDevice");
        out.conv[v].resize(conv_n);
        out.ssm[v].resize(ssm_n);
        cuda_ok(cudaMemcpy(out.conv[v].data(), gdn_conv_[v],  conv_n * sizeof(float), cudaMemcpyDeviceToHost), "save conv");
        cuda_ok(cudaMemcpy(out.ssm[v].data(),  gdn_state_[v], ssm_n  * sizeof(float), cudaMemcpyDeviceToHost), "save ssm");
    }
}

void SenseNovaLoopModel::restore_gdn_checkpoint(const GdnCheckpoint & cp) {
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const size_t conv_n = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim;
    const size_t ssm_n  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state;
    for (int v = 0; v < cfg_.n_visits; ++v) {
        const int phys = cfg_.plan[v];
        if (!layers_[phys].is_recr) continue;
        cuda_ok(cudaSetDevice(layers_[phys].device), "sensenova restore_cp setDevice");
        cuda_ok(cudaMemcpy(gdn_conv_[v],  cp.conv[v].data(), conv_n * sizeof(float), cudaMemcpyHostToDevice), "restore conv");
        cuda_ok(cudaMemcpy(gdn_state_[v], cp.ssm[v].data(),  ssm_n  * sizeof(float), cudaMemcpyHostToDevice), "restore ssm");
    }
}

void SenseNovaLoopModel::rollback_gdn_states() {
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner;
    const size_t conv_b = static_cast<size_t>(cfg_.ssm_d_conv - 1) * conv_dim * sizeof(float);
    const size_t ssm_b  = static_cast<size_t>(cfg_.ssm_dt_rank) * cfg_.ssm_d_state * cfg_.ssm_d_state * sizeof(float);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cudaSetDevice(rt.device);
        for (int v = 0; v < cfg_.n_visits; ++v) {
            const int phys = cfg_.plan[v];
            if (layers_[phys].device != rt.device || !layers_[phys].is_recr) continue;
            cudaMemcpyAsync(gdn_conv_[v],  gdn_conv_snap_[v],  conv_b, cudaMemcpyDeviceToDevice, rt.stream);
            cudaMemcpyAsync(gdn_state_[v], gdn_state_snap_[v], ssm_b,  cudaMemcpyDeviceToDevice, rt.stream);
        }
    }
}

void SenseNovaLoopModel::run_visit_segment(size_t seg_idx, int T, bool run_argmax, bool save_gdn_snap,
                                           int kv_len_max, bool compute_logits) {
    const auto & seg = segments_[seg_idx];
    auto & rt = runtimes_[seg.gpu_idx];
    cudaSetDevice(rt.device);
    cudaStream_t st = rt.stream;

    const int sm       = device_sm_version(rt.device);
    const int H        = cfg_.n_embd;
    const int d_q      = cfg_.n_head * cfg_.head_dim;
    const int conv_dim = 2 * cfg_.ssm_n_group * cfg_.ssm_d_state + cfg_.ssm_d_inner; // 8192
    const int qkvz_dim = conv_dim + cfg_.ssm_d_inner;                                // 12288
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
    const float gdn_scale  = 1.0f / std::sqrt(static_cast<float>(cfg_.ssm_d_state));

    const bool do_prof = (!use_cuda_graphs_ && std::getenv("ROOKERY_PROFILE") != nullptr);
    double ms_mhc = 0, ms_gdn_proj = 0, ms_gdn_core = 0, ms_gdn_out = 0;
    double ms_gqa_qkv = 0, ms_gqa_attn = 0, ms_gqa_out = 0;
    double ms_ffn = 0, ms_lm = 0;
    auto prof_mark = [&](double & acc, auto && fn) {
        if (!do_prof) { fn(); return; }
        cudaStreamSynchronize(st);
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        cudaStreamSynchronize(st);
        acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    // Fast Q8_0 single linear helper using sm_60 smem-cached kernel when T <= 2
    auto run_linear_fast = [&](const DeviceTensor & W, const float * d_x, float * d_y) {
        if ((T == 1 || T == 2) && sm == 60 && W.dtype == DType::Q8_0 && W.is_soa) {
            launch_q8_0_prep_x_sm60_h2(d_x, rt.d_scratch, static_cast<int>(W.ne[0]), T, st);
            if (!try_launch_sensenova_q8_0_sm60_smem_prepped(W, rt.d_scratch, d_y, T, st)) {
                launch_linear_q8_0_sm60_h2_prepped(W, rt.d_scratch, d_y, T, st);
            }
            return;
        }
        launch_linear(rt.cublas, W, d_x, d_y, T, rt.d_scratch, rt.scratch_bytes, st);
    };

    void * d_mhc_part = static_cast<uint8_t *>(rt.d_scratch) + (rt.scratch_bytes - 64 * 1024);
    void * d_act_prep = static_cast<uint8_t *>(rt.d_scratch) + 256 * 1024;
    const size_t work_scratch_b = rt.scratch_bytes - 65536;
    const int prep_mode = (sm == 60) ? 1 : 2;
    const bool fast_dec = (T == 1 || T == 2);

    auto run_prepped_q8 = [&](const DeviceTensor & W, float * d_y) {
        if (sm == 60) {
            if (!try_launch_sensenova_q8_0_sm60_smem_prepped(W, rt.d_scratch, d_y, T, st)) {
                launch_linear_q8_0_sm60_h2_prepped(W, rt.d_scratch, d_y, T, st);
            }
        } else {
            launch_linear_q8_0_prequant(W, rt.d_scratch, d_y, T, st);
        }
    };

    if (seg_idx == 0) {
        launch_embedding_lookup(tok_embd_, rt.d_tokens, rt.d_cur, H, 1, T, st);
    }

    bool next_mhc_part_ready = false;

    for (int v = seg.visit_start; v < seg.visit_end; ++v) {
        const int phys = cfg_.plan[v];
        const auto & L = layers_[phys];

        const bool attn_q8_ready = fast_dec && (L.is_recr
            ? (L.wqkv_gate.valid() && L.wqkv_gate.dtype == DType::Q8_0 && L.wqkv_gate.is_soa &&
               L.ssm_beta_alpha.valid() && L.ssm_beta_alpha.dtype == DType::Q8_0 && L.ssm_beta_alpha.is_soa)
            : (L.wq.dtype == DType::Q8_0 && L.wq.is_soa && L.wk.dtype == DType::Q8_0 && L.wk.is_soa &&
               L.wv.dtype == DType::Q8_0 && L.wv.is_soa));
        const bool ffn_q8_ready = fast_dec &&
            L.ffn_gate.dtype == DType::Q8_0 && L.ffn_gate.is_soa &&
            L.ffn_up.dtype == DType::Q8_0 && L.ffn_up.is_soa &&
            L.ffn_down.dtype == DType::Q8_0 && L.ffn_down.is_soa;

        // 1. Pre-attention mHC (layers [4..28)) or standard RMSNorm — fused with direct Q8 activation prep!
        if (L.has_mhc) {
            prof_mark(ms_mhc, [&]() {
                if (phys == cfg_.mhc_start) {
                    launch_sensenova_mhc_enter(rt.d_cur, rt.d_res, H, T, st);
                    next_mhc_part_ready = false;
                }
                launch_sensenova_mhc_pre_rmsnorm(
                    rt.cublas, rt.d_res,
                    static_cast<const float *>(L.hc_attn_w.ptr),
                    static_cast<const float *>(L.hc_attn_b.ptr),
                    static_cast<const float *>(L.attn_norm.ptr),
                    rt.d_mhc_ctrl, rt.d_x_norm,
                    (T <= 4) ? d_mhc_part : rt.d_scratch,
                    H, T, cfg_.mhc_iterations, cfg_.mhc_fixed_identity,
                    cfg_.mhc_pre_scale, cfg_.rms_eps,
                    (T <= 4) && next_mhc_part_ready, st,
                    attn_q8_ready ? rt.d_scratch : nullptr,
                    attn_q8_ready ? prep_mode : 0);
                next_mhc_part_ready = false;
            });
        } else if (attn_q8_ready) {
            launch_sensenova_rmsnorm_prep(
                rt.d_cur, nullptr, static_cast<const float *>(L.attn_norm.ptr),
                rt.d_scratch, nullptr, H, T, cfg_.rms_eps, false, prep_mode, st);
        } else {
            launch_rmsnorm(rt.d_cur, static_cast<const float *>(L.attn_norm.ptr), rt.d_x_norm, H, T, cfg_.rms_eps, st);
        }

        // 2. Attention branch: GDN Linear Attention OR Full GQA
        if (L.is_recr) {
            prof_mark(ms_gdn_proj, [&]() {
                if (attn_q8_ready) {
                    run_prepped_q8(L.wqkv_gate,      rt.d_qkvz);
                    run_prepped_q8(L.ssm_beta_alpha, rt.d_ba);
                } else if (L.wqkv_gate.valid() && L.ssm_beta_alpha.valid()) {
                    const DeviceTensor * ws[2] = {&L.wqkv_gate, &L.ssm_beta_alpha};
                    float * ys[2] = {rt.d_qkvz, rt.d_ba};
                    if (!(sm == 60 && T >= 16 && try_launch_sensenova_multi_prefill_sm60_f16(
                            rt.cublas, ws, ys, 2, rt.d_x_norm, T, rt.d_scratch, work_scratch_b, st))) {
                        launch_linear(rt.cublas, L.wqkv_gate,      rt.d_x_norm, rt.d_qkvz, T, rt.d_scratch, work_scratch_b, st);
                        launch_linear(rt.cublas, L.ssm_beta_alpha, rt.d_x_norm, rt.d_ba,   T, rt.d_scratch, work_scratch_b, st);
                    }
                } else {
                    launch_linear(rt.cublas, L.wqkv,      rt.d_x_norm, rt.d_qkv_conv,      T, rt.d_scratch, work_scratch_b, st);
                    launch_linear(rt.cublas, L.wgate,     rt.d_x_norm, rt.d_qkvz + conv_dim, T, rt.d_scratch, work_scratch_b, st);
                    launch_linear(rt.cublas, L.ssm_alpha, rt.d_x_norm, rt.d_alpha,         T, rt.d_scratch, work_scratch_b, st);
                    launch_linear(rt.cublas, L.ssm_beta,  rt.d_x_norm, rt.d_beta,          T, rt.d_scratch, work_scratch_b, st);
                }
            });

            const bool gdn_out_q8_ready = fast_dec && L.ssm_out.dtype == DType::Q8_0 && L.ssm_out.is_soa;
            prof_mark(ms_gdn_core, [&]() {
                if (L.ssm_beta_alpha.valid()) {
                    launch_sensenova_gdn_gate_beta_packed(
                        rt.d_ba, static_cast<const float *>(L.ssm_dt.ptr),
                        static_cast<const float *>(L.ssm_a.ptr),
                        rt.d_alpha, rt.d_beta, T, cfg_.ssm_dt_rank, st);
                } else {
                    launch_qwen_gdn_gate_and_beta(
                        rt.d_alpha, static_cast<const float *>(L.ssm_dt.ptr),
                        static_cast<const float *>(L.ssm_a.ptr),
                        rt.d_beta, T, cfg_.ssm_dt_rank, st);
                }

                launch_sensenova_gdn_conv_and_recurrence(
                    rt.d_qkvz, static_cast<const float *>(L.ssm_conv1d.ptr),
                    gdn_conv_[v], rt.d_qkv_conv,
                    rt.d_alpha, rt.d_beta,
                    gdn_state_[v], rt.d_gdn_out,
                    T, cfg_.ssm_n_group, cfg_.ssm_dt_rank, qkvz_dim, cfg_.rms_eps, gdn_scale, st,
                    save_gdn_snap ? gdn_conv_snap_[v] : nullptr,
                    save_gdn_snap ? gdn_state_snap_[v] : nullptr);

                launch_sensenova_rms_norm_silu_gate_d128_strided(
                    rt.d_gdn_out, static_cast<const float *>(L.ssm_norm.ptr),
                    rt.d_qkvz, T, cfg_.ssm_dt_rank, qkvz_dim, conv_dim, cfg_.rms_eps, st,
                    gdn_out_q8_ready ? rt.d_scratch : nullptr,
                    gdn_out_q8_ready ? prep_mode : 0);
            });

            prof_mark(ms_gdn_out, [&]() {
                if (gdn_out_q8_ready) run_prepped_q8(L.ssm_out, rt.d_sub_out);
                else run_linear_fast(L.ssm_out, rt.d_gdn_out, rt.d_sub_out);
            });
        } else {
            prof_mark(ms_gqa_qkv, [&]() {
                if (attn_q8_ready) {
                    run_prepped_q8(L.wq, rt.d_q_full);
                    run_prepped_q8(L.wk, rt.d_k);
                    run_prepped_q8(L.wv, rt.d_v);
                } else {
                    const DeviceTensor * ws[3] = {&L.wq, &L.wk, &L.wv};
                    float * ys[3] = {rt.d_q_full, rt.d_k, rt.d_v};
                    if (!(sm == 60 && T >= 16 && try_launch_sensenova_multi_prefill_sm60_f16(
                            rt.cublas, ws, ys, 3, rt.d_x_norm, T, rt.d_scratch, work_scratch_b, st))) {
                        launch_linear(rt.cublas, L.wq, rt.d_x_norm, rt.d_q_full, T, rt.d_scratch, work_scratch_b, st);
                        launch_linear(rt.cublas, L.wk, rt.d_x_norm, rt.d_k,      T, rt.d_scratch, work_scratch_b, st);
                        launch_linear(rt.cublas, L.wv, rt.d_x_norm, rt.d_v,      T, rt.d_scratch, work_scratch_b, st);
                    }
                }
            });

            const bool gqa_out_q8_ready = fast_dec && L.wo.dtype == DType::Q8_0 && L.wo.is_soa;
            prof_mark(ms_gqa_attn, [&]() {
                launch_sensenova_qgate_split_norm_yarn_rope_d256(
                    rt.d_q_full, static_cast<const float *>(L.attn_q_norm.ptr),
                    rt.d_pos, rt.d_inv_freq, rt.d_q, rt.d_gate,
                    T, cfg_.n_head, cfg_.rope_dim, cfg_.rms_eps, cfg_.yarn_mscale, st);
                launch_sensenova_k_norm_yarn_rope_d256(
                    rt.d_k, static_cast<const float *>(L.attn_k_norm.ptr),
                    rt.d_pos, rt.d_inv_freq,
                    T, cfg_.n_head_kv, cfg_.rope_dim, cfg_.rms_eps, cfg_.yarn_mscale, st);
                launch_gqa_kv_write_d256(rt.d_k, rt.d_v, k_cache_[v], v_cache_[v], cfg_.kv_q8_0,
                                         rt.d_pos, T, cfg_.n_head_kv, st);
                launch_gqa_attn_d256(rt.d_q, k_cache_[v], v_cache_[v], cfg_.kv_q8_0,
                                     rt.d_pos, rt.d_attn_out, rt.d_scratch, work_scratch_b,
                                     T, cfg_.n_head, cfg_.n_head_kv, attn_scale, rt.cublas, kv_len_max, st);
                if (gqa_out_q8_ready) {
                    launch_sensenova_sigmoid_gate_mul_prep(rt.d_attn_out, rt.d_gate, rt.d_scratch, d_q, T, prep_mode, st);
                } else {
                    launch_inkling2_sigmoid_gate_mul(rt.d_attn_out, rt.d_gate, static_cast<int64_t>(T) * d_q, st);
                }
            });

            prof_mark(ms_gqa_out, [&]() {
                if (gqa_out_q8_ready) run_prepped_q8(L.wo, rt.d_sub_out);
                else run_linear_fast(L.wo, rt.d_attn_out, rt.d_sub_out);
            });
        }

        // 3. Attention residual merge + Pre-FFN mHC / Post-attention RMSNorm (+ direct Q8 prep!)
        if (L.has_mhc) {
            prof_mark(ms_mhc, [&]() {
                if (T <= 4) {
                    launch_sensenova_mhc_merge_router_part(
                        rt.d_res, rt.d_sub_out, rt.d_mhc_ctrl, rt.d_res,
                        static_cast<const float *>(L.hc_ffn_w.ptr), d_mhc_part, H, T, st);
                    launch_sensenova_mhc_pre_rmsnorm(
                        rt.cublas, rt.d_res,
                        static_cast<const float *>(L.hc_ffn_w.ptr),
                        static_cast<const float *>(L.hc_ffn_b.ptr),
                        static_cast<const float *>(L.post_attn_norm.ptr),
                        rt.d_mhc_ctrl, rt.d_x_norm, d_mhc_part,
                        H, T, cfg_.mhc_iterations, cfg_.mhc_fixed_identity,
                        cfg_.mhc_pre_scale, cfg_.rms_eps, true, st,
                        ffn_q8_ready ? rt.d_scratch : nullptr,
                        ffn_q8_ready ? prep_mode : 0);
                } else {
                    launch_sensenova_mhc_merge(rt.d_res, rt.d_sub_out, rt.d_mhc_ctrl, rt.d_res, H, T, st);
                    launch_sensenova_mhc_pre_rmsnorm(
                        rt.cublas, rt.d_res,
                        static_cast<const float *>(L.hc_ffn_w.ptr),
                        static_cast<const float *>(L.hc_ffn_b.ptr),
                        static_cast<const float *>(L.post_attn_norm.ptr),
                        rt.d_mhc_ctrl, rt.d_x_norm, rt.d_scratch,
                        H, T, cfg_.mhc_iterations, cfg_.mhc_fixed_identity,
                        cfg_.mhc_pre_scale, cfg_.rms_eps, false, st);
                }
            });
        } else if (ffn_q8_ready) {
            launch_sensenova_rmsnorm_prep(
                rt.d_cur, rt.d_sub_out, static_cast<const float *>(L.post_attn_norm.ptr),
                rt.d_scratch, nullptr, H, T, cfg_.rms_eps, true, prep_mode, st);
        } else {
            launch_add_rmsnorm(rt.d_cur, rt.d_sub_out, static_cast<const float *>(L.post_attn_norm.ptr),
                               rt.d_x_norm, H, T, cfg_.rms_eps, st);
        }

        // 4. Dense SwiGLU FFN (fused decode OR pure-FP16 prefill on sm_60!)
        prof_mark(ms_ffn, [&]() {
            if (ffn_q8_ready) {
                if (!try_launch_sensenova_swiglu_q8_0_fused(L.ffn_gate, L.ffn_up, L.ffn_down,
                                                            rt.d_scratch, d_act_prep, rt.d_sub_out, T, sm, st)) {
                    launch_linear(rt.cublas, L.ffn_gate, rt.d_x_norm, rt.d_ff_gate, T, rt.d_scratch, work_scratch_b, st);
                    launch_linear(rt.cublas, L.ffn_up,   rt.d_x_norm, rt.d_ff_up,   T, rt.d_scratch, work_scratch_b, st);
                    launch_silu_mul(rt.d_ff_gate, rt.d_ff_up, rt.d_ff_gate, static_cast<int64_t>(T) * cfg_.n_ff, st);
                    launch_linear(rt.cublas, L.ffn_down, rt.d_ff_gate, rt.d_sub_out, T, rt.d_scratch, work_scratch_b, st);
                }
            } else if (sm == 60 && T >= 16 &&
                       try_launch_sensenova_ffn_prefill_sm60_f16(
                           rt.cublas, L.ffn_gate, L.ffn_up, L.ffn_down,
                           rt.d_x_norm, rt.d_sub_out, T, rt.d_scratch, work_scratch_b, st)) {
                // done in pure FP16!
            } else {
                launch_linear(rt.cublas, L.ffn_gate, rt.d_x_norm, rt.d_ff_gate, T, rt.d_scratch, work_scratch_b, st);
                launch_linear(rt.cublas, L.ffn_up,   rt.d_x_norm, rt.d_ff_up,   T, rt.d_scratch, work_scratch_b, st);
                launch_silu_mul(rt.d_ff_gate, rt.d_ff_up, rt.d_ff_gate, static_cast<int64_t>(T) * cfg_.n_ff, st);
                launch_linear(rt.cublas, L.ffn_down, rt.d_ff_gate, rt.d_sub_out, T, rt.d_scratch, work_scratch_b, st);
            }
        });

        // 5. FFN residual merge (+ leave at mhc_end - 1, OR fuse with next layer's mHC router!)
        const bool has_hook = (hook_ != nullptr && hook_->hooks_layer(v));
        if (L.has_mhc) {
            prof_mark(ms_mhc, [&]() {
                if (phys == cfg_.mhc_end - 1) {
                    launch_sensenova_mhc_merge_leave(rt.d_res, rt.d_sub_out, rt.d_mhc_ctrl, rt.d_cur, H, T, st);
                    next_mhc_part_ready = false;
                } else {
                    const bool can_fuse_next = (T <= 4) && !has_hook && (v + 1 < seg.visit_end) &&
                                               layers_[cfg_.plan[v + 1]].has_mhc &&
                                               (cfg_.plan[v + 1] != cfg_.mhc_start);
                    if (can_fuse_next) {
                        const auto & Lnext = layers_[cfg_.plan[v + 1]];
                        launch_sensenova_mhc_merge_router_part(
                            rt.d_res, rt.d_sub_out, rt.d_mhc_ctrl, rt.d_res,
                            static_cast<const float *>(Lnext.hc_attn_w.ptr), d_mhc_part, H, T, st);
                        next_mhc_part_ready = true;
                    } else {
                        launch_sensenova_mhc_merge(rt.d_res, rt.d_sub_out, rt.d_mhc_ctrl, rt.d_res, H, T, st);
                        next_mhc_part_ready = false;
                    }
                }
            });
            if (has_hook) {
                if (phys == cfg_.mhc_end - 1) {
                    hook_->on_post_layer(v, rt.device, rt.d_cur, T, H, 1, st);
                } else {
                    hook_->on_post_layer(v, rt.device, rt.d_res, T, H, cfg_.mhc_streams, st);
                }
            }
        } else {
            launch_add_inplace(rt.d_cur, rt.d_sub_out, static_cast<int64_t>(T) * H, st);
            if (has_hook) {
                hook_->on_post_layer(v, rt.device, rt.d_cur, T, H, 1, st);
            }
        }
    }

    // 6. Final RMSNorm -> save h_nextn -> LM head
    if (seg_idx + 1 == segments_.size() && (compute_logits || has_mtp_)) {
        prof_mark(ms_lm, [&]() {
            if (compute_logits && run_argmax && fast_dec && output_.dtype == DType::Q8_0 && output_.is_soa) {
                launch_sensenova_rmsnorm_prep(
                    rt.d_cur, nullptr, static_cast<const float *>(output_norm_.ptr),
                    rt.d_scratch, has_mtp_ ? rt.d_h_nextn : nullptr,
                    H, T, cfg_.rms_eps, false, prep_mode, st);
                run_prepped_q8(output_, rt.d_logits);
                launch_argmax_f32_batch(rt.d_logits, rt.d_argmax, cfg_.n_vocab, T, st);
            } else {
                launch_rmsnorm(rt.d_cur, static_cast<const float *>(output_norm_.ptr), rt.d_x_norm, H, T, cfg_.rms_eps, st);
                if (has_mtp_ && rt.d_h_nextn) {
                    cudaMemcpyAsync(rt.d_h_nextn, rt.d_x_norm, static_cast<size_t>(T) * H * sizeof(float), cudaMemcpyDeviceToDevice, st);
                }
                if (compute_logits) {
                    if (run_argmax && T <= 2) {
                        run_linear_fast(output_, rt.d_x_norm, rt.d_logits);
                        launch_argmax_f32_batch(rt.d_logits, rt.d_argmax, cfg_.n_vocab, T, st);
                    } else {
                        launch_linear(rt.cublas, output_, rt.d_x_norm + static_cast<size_t>(T - 1) * H,
                                      rt.d_logits, 1, rt.d_scratch, work_scratch_b, st);
                    }
                }
            }
        });
    }

    if (do_prof) {
        const double tot = ms_mhc + ms_gdn_proj + ms_gdn_core + ms_gdn_out +
                           ms_gqa_qkv + ms_gqa_attn + ms_gqa_out + ms_ffn + ms_lm;
        std::printf("[PROF seg%zu GPU%d T=%d] tot=%.2f | mhc=%.2f gdn_p=%.2f gdn_c=%.2f gdn_o=%.2f | gqa_p=%.2f gqa_a=%.2f gqa_o=%.2f | ffn=%.2f lm=%.2f ms\n",
                    seg_idx, rt.device, T, tot, ms_mhc, ms_gdn_proj, ms_gdn_core, ms_gdn_out,
                    ms_gqa_qkv, ms_gqa_attn, ms_gqa_out, ms_ffn, ms_lm);
    }
}

void SenseNovaLoopModel::run_mtp_layer(int T, bool run_argmax, int kv_len_max) {
    auto & rt = runtimes_.back();
    cudaSetDevice(rt.device);
    cudaStream_t st = rt.stream;

    const int sm = device_sm_version(rt.device);
    const int H  = cfg_.n_embd;
    const int d_q = cfg_.n_head * cfg_.head_dim;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
    void * d_act_prep = static_cast<uint8_t *>(rt.d_scratch) + 256 * 1024;

    auto run_linear_fast = [&](const DeviceTensor & W, const float * d_x, float * d_y) {
        if ((T == 1 || T == 2) && sm == 60 && W.dtype == DType::Q8_0 && W.is_soa) {
            launch_q8_0_prep_x_sm60_h2(d_x, rt.d_scratch, static_cast<int>(W.ne[0]), T, st);
            if (!try_launch_sensenova_q8_0_sm60_smem_prepped(W, rt.d_scratch, d_y, T, st)) {
                launch_linear_q8_0_sm60_h2_prepped(W, rt.d_scratch, d_y, T, st);
            }
            return;
        }
        launch_linear(rt.cublas, W, d_x, d_y, T, rt.d_scratch, rt.scratch_bytes - 65536, st);
    };

    launch_mtp_eh_concat(rt.d_mtp_e, rt.d_h_nextn,
                         static_cast<const float *>(mtp_.enorm.ptr),
                         static_cast<const float *>(mtp_.hnorm.ptr),
                         rt.d_mtp_eh, H, 1, T, cfg_.rms_eps, st);
    run_linear_fast(mtp_.eh_proj, rt.d_mtp_eh, rt.d_mtp_cur);

    launch_rmsnorm(rt.d_mtp_cur, static_cast<const float *>(mtp_.attn_norm.ptr), rt.d_x_norm, H, T, cfg_.rms_eps, st);
    if ((T == 1 || T == 2) && sm == 60 && mtp_.wq.dtype == DType::Q8_0 && mtp_.wq.is_soa &&
        mtp_.wk.dtype == DType::Q8_0 && mtp_.wk.is_soa && mtp_.wv.dtype == DType::Q8_0 && mtp_.wv.is_soa) {
        launch_q8_0_prep_x_sm60_h2(rt.d_x_norm, rt.d_scratch, H, T, st);
        try_launch_sensenova_q8_0_sm60_smem_prepped(mtp_.wq, rt.d_scratch, rt.d_q_full, T, st);
        try_launch_sensenova_q8_0_sm60_smem_prepped(mtp_.wk, rt.d_scratch, rt.d_k,      T, st);
        try_launch_sensenova_q8_0_sm60_smem_prepped(mtp_.wv, rt.d_scratch, rt.d_v,      T, st);
    } else if (T < 16 && sm != 60 && mtp_.wq.dtype == DType::Q8_0 && mtp_.wk.dtype == DType::Q8_0 && mtp_.wv.dtype == DType::Q8_0) {
        launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, H, T, st);
        launch_linear_q8_0_prequant(mtp_.wq, rt.d_scratch, rt.d_q_full, T, st);
        launch_linear_q8_0_prequant(mtp_.wk, rt.d_scratch, rt.d_k,      T, st);
        launch_linear_q8_0_prequant(mtp_.wv, rt.d_scratch, rt.d_v,      T, st);
    } else {
        launch_linear(rt.cublas, mtp_.wq, rt.d_x_norm, rt.d_q_full, T, rt.d_scratch, rt.scratch_bytes - 65536, st);
        launch_linear(rt.cublas, mtp_.wk, rt.d_x_norm, rt.d_k,      T, rt.d_scratch, rt.scratch_bytes - 65536, st);
        launch_linear(rt.cublas, mtp_.wv, rt.d_x_norm, rt.d_v,      T, rt.d_scratch, rt.scratch_bytes - 65536, st);
    }

    launch_sensenova_qgate_split_norm_yarn_rope_d256(
        rt.d_q_full, static_cast<const float *>(mtp_.attn_q_norm.ptr),
        rt.d_mtp_pos, rt.d_inv_freq, rt.d_q, rt.d_gate,
        T, cfg_.n_head, cfg_.rope_dim, cfg_.rms_eps, cfg_.yarn_mscale, st);
    launch_sensenova_k_norm_yarn_rope_d256(
        rt.d_k, static_cast<const float *>(mtp_.attn_k_norm.ptr),
        rt.d_mtp_pos, rt.d_inv_freq,
        T, cfg_.n_head_kv, cfg_.rope_dim, cfg_.rms_eps, cfg_.yarn_mscale, st);
    launch_gqa_kv_write_d256(rt.d_k, rt.d_v, mtp_k_cache_, mtp_v_cache_, cfg_.kv_q8_0,
                             rt.d_mtp_pos, T, cfg_.n_head_kv, st);
    launch_gqa_attn_d256(rt.d_q, mtp_k_cache_, mtp_v_cache_, cfg_.kv_q8_0,
                         rt.d_mtp_pos, rt.d_attn_out, rt.d_scratch, rt.scratch_bytes - 65536,
                         T, cfg_.n_head, cfg_.n_head_kv, attn_scale, rt.cublas, kv_len_max, st);
    launch_inkling2_sigmoid_gate_mul(rt.d_attn_out, rt.d_gate, static_cast<int64_t>(T) * d_q, st);
    run_linear_fast(mtp_.wo, rt.d_attn_out, rt.d_sub_out);

    launch_add_rmsnorm(rt.d_mtp_cur, rt.d_sub_out, static_cast<const float *>(mtp_.post_attn_norm.ptr),
                       rt.d_x_norm, H, T, cfg_.rms_eps, st);

    if ((T == 1 || T == 2) && mtp_.ffn_gate.dtype == DType::Q8_0 && mtp_.ffn_gate.is_soa) {
        if (sm == 60) launch_q8_0_prep_x_sm60_h2(rt.d_x_norm, rt.d_scratch, H, T, st);
        else          launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, H, T, st);
        try_launch_sensenova_swiglu_q8_0_fused(mtp_.ffn_gate, mtp_.ffn_up, mtp_.ffn_down,
                                               rt.d_scratch, d_act_prep, rt.d_sub_out, T, sm, st);
    } else {
        launch_linear(rt.cublas, mtp_.ffn_gate, rt.d_x_norm, rt.d_ff_gate, T, rt.d_scratch, rt.scratch_bytes - 65536, st);
        launch_linear(rt.cublas, mtp_.ffn_up,   rt.d_x_norm, rt.d_ff_up,   T, rt.d_scratch, rt.scratch_bytes - 65536, st);
        launch_silu_mul(rt.d_ff_gate, rt.d_ff_up, rt.d_ff_gate, static_cast<int64_t>(T) * cfg_.n_ff, st);
        launch_linear(rt.cublas, mtp_.ffn_down, rt.d_ff_gate, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes - 65536, st);
    }
    launch_add_inplace(rt.d_mtp_cur, rt.d_sub_out, static_cast<int64_t>(T) * H, st);

    launch_rmsnorm(rt.d_mtp_cur + static_cast<size_t>(T - 1) * H,
                   static_cast<const float *>(mtp_.shared_head_norm.ptr),
                   rt.d_x_norm, H, 1, cfg_.rms_eps, st);
    if (sm == 60 && output_.dtype == DType::Q8_0 && output_.is_soa) {
        launch_q8_0_prep_x_sm60_h2(rt.d_x_norm, rt.d_scratch, H, 1, st);
        if (!try_launch_sensenova_q8_0_sm60_smem_prepped(output_, rt.d_scratch, rt.d_logits, 1, st)) {
            launch_linear_q8_0_sm60_h2_prepped(output_, rt.d_scratch, rt.d_logits, 1, st);
        }
    } else {
        launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes - 65536, st);
    }
    if (run_argmax) {
        launch_argmax_f32(rt.d_logits, rt.d_mtp_argmax, cfg_.n_vocab, st);
    }
}

void SenseNovaLoopModel::embed_tokens_to_last_dev(const int32_t * h_tokens, int T) {
    auto & rt0 = runtimes_.front();
    auto & rtl = runtimes_.back();
    const size_t bytes = static_cast<size_t>(T) * cfg_.n_embd * sizeof(float);
    cuda_ok(cudaSetDevice(rt0.device), "sn mtp embd dev0");
    cuda_ok(cudaMemcpyAsync(rt0.d_tokens, h_tokens, T * sizeof(int32_t), cudaMemcpyHostToDevice, rt0.stream), "sn mtp tok");
    launch_embedding_lookup(tok_embd_, rt0.d_tokens, rt0.d_sub_out, cfg_.n_embd, 1, T, rt0.stream);
    if (runtimes_.size() == 1) {
        cuda_ok(cudaMemcpyAsync(rtl.d_mtp_e, rt0.d_sub_out, bytes, cudaMemcpyDeviceToDevice, rt0.stream), "sn mtp e d2d");
    } else {
        cuda_ok(cudaEventRecord(rt0.done_event, rt0.stream), "sn mtp embd rec");
        cuda_ok(cudaSetDevice(rtl.device), "sn mtp embd last");
        cuda_ok(cudaStreamWaitEvent(rtl.stream, rt0.done_event, 0), "sn mtp embd wait");
        cuda_ok(cudaMemcpyPeerAsync(rtl.d_mtp_e, rtl.device, rt0.d_sub_out, rt0.device, bytes, rtl.stream), "sn mtp e peer");
    }
}

void SenseNovaLoopModel::capture_decode_graphs() {
    invalidate_graphs();
    if (!use_cuda_graphs_) return;
    for (size_t si = 0; si < segments_.size(); ++si) {
        auto & rt = runtimes_[segments_[si].gpu_idx];
        cuda_ok(cudaSetDevice(rt.device), "sn cap setDevice");
        cuda_ok(cudaStreamSynchronize(rt.stream), "sn cap sync");
        {
            cudaGraph_t g = nullptr;
            cuda_ok(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeRelaxed), "sn cap t1");
            run_visit_segment(si, 1, true, false, 0);
            cuda_ok(cudaStreamEndCapture(rt.stream, &g), "sn end t1");
            cuda_ok(cudaGraphInstantiate(&seg_graphs_t1_[si], g, nullptr, nullptr, 0), "sn inst t1");
            cudaGraphDestroy(g);
        }
        if (has_mtp_) {
            cudaGraph_t g = nullptr;
            cuda_ok(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeRelaxed), "sn cap t2");
            run_visit_segment(si, 2, true, true, 0);
            cuda_ok(cudaStreamEndCapture(rt.stream, &g), "sn end t2");
            cuda_ok(cudaGraphInstantiate(&seg_graphs_t2_[si], g, nullptr, nullptr, 0), "sn inst t2");
            cudaGraphDestroy(g);
        }
    }
    if (has_mtp_) {
        auto & rtl = runtimes_.back();
        cuda_ok(cudaSetDevice(rtl.device), "sn cap mtp dev");
        cuda_ok(cudaStreamSynchronize(rtl.stream), "sn cap mtp sync");
        {
            cudaGraph_t g = nullptr;
            cuda_ok(cudaStreamBeginCapture(rtl.stream, cudaStreamCaptureModeRelaxed), "sn cap mtp t1");
            run_mtp_layer(1, true, 0);
            cuda_ok(cudaStreamEndCapture(rtl.stream, &g), "sn end mtp t1");
            cuda_ok(cudaGraphInstantiate(&mtp_graph_t1_, g, nullptr, nullptr, 0), "sn inst mtp t1");
            cudaGraphDestroy(g);
        }
        {
            cudaGraph_t g = nullptr;
            cuda_ok(cudaStreamBeginCapture(rtl.stream, cudaStreamCaptureModeRelaxed), "sn cap mtp t2");
            run_mtp_layer(2, true, 0);
            cuda_ok(cudaStreamEndCapture(rtl.stream, &g), "sn end mtp t2");
            cuda_ok(cudaGraphInstantiate(&mtp_graph_t2_, g, nullptr, nullptr, 0), "sn inst mtp t2");
            cudaGraphDestroy(g);
        }
    }
    graphs_ready_ = true;
}

void SenseNovaLoopModel::warmup_decode_graphs() {
    if (!use_cuda_graphs_ || graphs_ready_) return;
    capture_decode_graphs();
}

void SenseNovaLoopModel::forward(const std::vector<int32_t> & tokens,
                                 const std::vector<int32_t> & positions,
                                 std::vector<float> & out_logits,
                                 bool compute_logits) {
    const int T = static_cast<int>(tokens.size());
    if (T == 0 || T != static_cast<int>(positions.size()) || T > max_batch_) {
        throw std::runtime_error("SenseNovaLoopModel::forward invalid batch size");
    }
    int max_p = 0;
    for (int32_t p : positions) if (p > max_p) max_p = p;
    const int kv_len_max = max_p + 1;

    // Upload positions to every GPU once
    for (auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "sn fwd pos dev");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, positions.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "sn fwd pos");
    }

    for (size_t si = 0; si < segments_.size(); ++si) {
        auto & rt = runtimes_[segments_[si].gpu_idx];
        cuda_ok(cudaSetDevice(rt.device), "sn fwd seg dev");
        if (si == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, tokens.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "sn fwd tok");
        } else {
            auto & prev_rt = runtimes_[segments_[si - 1].gpu_idx];
            cuda_ok(cudaStreamSynchronize(prev_rt.stream), "sn fwd prev sync");
            const int prev_phys = cfg_.plan[segments_[si - 1].visit_end - 1];
            const bool transfer_mhc = layers_[prev_phys].has_mhc && (prev_phys != cfg_.mhc_end - 1);
            if (transfer_mhc) {
                const size_t b = static_cast<size_t>(T) * 2 * cfg_.n_embd * sizeof(float);
                cuda_ok(cudaMemcpyPeerAsync(rt.d_res, rt.device, prev_rt.d_res, prev_rt.device, b, rt.stream), "sn fwd peer res");
            } else {
                const size_t b = static_cast<size_t>(T) * cfg_.n_embd * sizeof(float);
                cuda_ok(cudaMemcpyPeerAsync(rt.d_cur, rt.device, prev_rt.d_cur, prev_rt.device, b, rt.stream), "sn fwd peer cur");
            }
        }
        run_visit_segment(si, T, false, false, kv_len_max, compute_logits);
    }

    auto & rtl = runtimes_.back();
    cuda_ok(cudaSetDevice(rtl.device), "sn fwd readout");
    if (compute_logits) {
        out_logits.resize(cfg_.n_vocab);
        cuda_ok(cudaMemcpyAsync(out_logits.data(), rtl.d_logits, static_cast<size_t>(cfg_.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "sn fwd logits");
    }
    cuda_ok(cudaStreamSynchronize(rtl.stream), "sn fwd sync");
}

int32_t SenseNovaLoopModel::forward_decode_greedy(int32_t token, int32_t pos) {
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();

    for (auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "sn dec pos dev");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "sn dec pos");
    }

    for (size_t si = 0; si < segments_.size(); ++si) {
        auto & rt = runtimes_[segments_[si].gpu_idx];
        cuda_ok(cudaSetDevice(rt.device), "sn dec seg dev");
        if (si == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "sn dec tok");
        } else {
            auto & prev_rt = runtimes_[segments_[si - 1].gpu_idx];
            cuda_ok(cudaStreamWaitEvent(rt.stream, prev_rt.done_event, 0), "sn dec wait");
            const int prev_phys = cfg_.plan[segments_[si - 1].visit_end - 1];
            const bool transfer_mhc = layers_[prev_phys].has_mhc && (prev_phys != cfg_.mhc_end - 1);
            if (transfer_mhc) {
                const size_t b = 2ULL * cfg_.n_embd * sizeof(float);
                cuda_ok(cudaMemcpyPeerAsync(rt.d_res, rt.device, prev_rt.d_res, prev_rt.device, b, rt.stream), "sn dec peer res");
            } else {
                const size_t b = static_cast<size_t>(cfg_.n_embd) * sizeof(float);
                cuda_ok(cudaMemcpyPeerAsync(rt.d_cur, rt.device, prev_rt.d_cur, prev_rt.device, b, rt.stream), "sn dec peer cur");
            }
        }
        if (graphs_ready_ && seg_graphs_t1_[si]) cuda_ok(cudaGraphLaunch(seg_graphs_t1_[si], rt.stream), "sn launch t1");
        else run_visit_segment(si, 1, true, false, pos + 1);
        if (si + 1 < segments_.size()) cuda_ok(cudaEventRecord(rt.done_event, rt.stream), "sn dec rec");
    }

    auto & rtl = runtimes_.back();
    int32_t next_tok = 0;
    cuda_ok(cudaSetDevice(rtl.device), "sn dec readout");
    cuda_ok(cudaMemcpyAsync(&next_tok, rtl.d_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, rtl.stream), "sn dec argmax");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "sn dec sync");
    return next_tok;
}

void SenseNovaLoopModel::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    forward_decode_greedy(token, pos);
    auto & rtl = runtimes_.back();
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaSetDevice(rtl.device), "sn dec logits dev");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), rtl.d_logits, static_cast<size_t>(cfg_.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "sn dec logits");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "sn dec logits sync");
}

int32_t SenseNovaLoopModel::mtp_prefill_chunk(const std::vector<int32_t> & next_tokens,
                                              const std::vector<int32_t> & positions) {
    if (!has_mtp_) return 0;
    const int T = static_cast<int>(next_tokens.size());
    int max_p = 0;
    for (int32_t p : positions) if (p > max_p) max_p = p;

    embed_tokens_to_last_dev(next_tokens.data(), T);
    auto & rtl = runtimes_.back();
    cuda_ok(cudaSetDevice(rtl.device), "sn mtp pp dev");
    cuda_ok(cudaMemcpyAsync(rtl.d_mtp_pos, positions.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rtl.stream), "sn mtp pp pos");
    run_mtp_layer(T, true, max_p + 1);
    int32_t draft = 0;
    cuda_ok(cudaMemcpyAsync(&draft, rtl.d_mtp_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, rtl.stream), "sn mtp pp argmax");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "sn mtp pp sync");
    return draft;
}

void SenseNovaLoopModel::mtp_prefill_chunk_logits(const std::vector<int32_t> & next_tokens,
                                                  const std::vector<int32_t> & positions,
                                                  std::vector<float> & out_logits) {
    if (!has_mtp_) return;
    const int T = static_cast<int>(next_tokens.size());
    int max_p = 0;
    for (int32_t p : positions) if (p > max_p) max_p = p;

    embed_tokens_to_last_dev(next_tokens.data(), T);
    auto & rtl = runtimes_.back();
    cuda_ok(cudaSetDevice(rtl.device), "sn mtp ppl dev");
    cuda_ok(cudaMemcpyAsync(rtl.d_mtp_pos, positions.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rtl.stream), "sn mtp ppl pos");
    run_mtp_layer(T, false, max_p + 1);
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaMemcpyAsync(out_logits.data(), rtl.d_logits, static_cast<size_t>(cfg_.n_vocab) * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "sn mtp ppl logits");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "sn mtp ppl sync");
}

void SenseNovaLoopModel::decode_mtp_greedy_step(int32_t & cur_tok,
                                                int32_t & draft_tok,
                                                int32_t & cur_pos,
                                                std::vector<int32_t> & out_tokens,
                                                MtpStats & stats) {
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();
    const int32_t h_tok[2] = {cur_tok, draft_tok};
    const int32_t h_pos[2] = {cur_pos, cur_pos + 1};

    for (auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "sn mtp dec pos dev");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, h_pos, sizeof(h_pos), cudaMemcpyHostToDevice, rt.stream), "sn mtp dec pos");
    }

    for (size_t si = 0; si < segments_.size(); ++si) {
        auto & rt = runtimes_[segments_[si].gpu_idx];
        cuda_ok(cudaSetDevice(rt.device), "sn mtp dec seg dev");
        if (si == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, h_tok, sizeof(h_tok), cudaMemcpyHostToDevice, rt.stream), "sn mtp dec tok");
        } else {
            auto & prev_rt = runtimes_[segments_[si - 1].gpu_idx];
            cuda_ok(cudaStreamWaitEvent(rt.stream, prev_rt.done_event, 0), "sn mtp dec wait");
            const int prev_phys = cfg_.plan[segments_[si - 1].visit_end - 1];
            const bool transfer_mhc = layers_[prev_phys].has_mhc && (prev_phys != cfg_.mhc_end - 1);
            if (transfer_mhc) {
                const size_t b = 4ULL * cfg_.n_embd * sizeof(float);
                cuda_ok(cudaMemcpyPeerAsync(rt.d_res, rt.device, prev_rt.d_res, prev_rt.device, b, rt.stream), "sn mtp dec peer res");
            } else {
                const size_t b = 2ULL * cfg_.n_embd * sizeof(float);
                cuda_ok(cudaMemcpyPeerAsync(rt.d_cur, rt.device, prev_rt.d_cur, prev_rt.device, b, rt.stream), "sn mtp dec peer cur");
            }
        }
        if (graphs_ready_ && seg_graphs_t2_[si]) cuda_ok(cudaGraphLaunch(seg_graphs_t2_[si], rt.stream), "sn launch t2");
        else run_visit_segment(si, 2, true, true, cur_pos + 2);
        if (si + 1 < segments_.size()) cuda_ok(cudaEventRecord(rt.done_event, rt.stream), "sn mtp dec rec");
    }

    auto & rtl = runtimes_.back();
    int32_t pred[2] = {0, 0};
    cuda_ok(cudaSetDevice(rtl.device), "sn mtp dec readout");
    cuda_ok(cudaMemcpyAsync(pred, rtl.d_argmax, sizeof(pred), cudaMemcpyDeviceToHost, rtl.stream), "sn mtp pred");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "sn mtp dec sync");

    stats.steps++;
    out_tokens.clear();
    if (pred[0] == draft_tok) {
        stats.accepted++;
        out_tokens.push_back(draft_tok);
        out_tokens.push_back(pred[1]);

        const int32_t next_t[2] = {draft_tok, pred[1]};
        embed_tokens_to_last_dev(next_t, 2);
        cuda_ok(cudaSetDevice(rtl.device), "sn mtp acc dev");
        cuda_ok(cudaMemcpyAsync(rtl.d_mtp_pos, h_pos, sizeof(h_pos), cudaMemcpyHostToDevice, rtl.stream), "sn mtp acc pos");
        if (graphs_ready_ && mtp_graph_t2_) cuda_ok(cudaGraphLaunch(mtp_graph_t2_, rtl.stream), "sn launch mtp t2");
        else run_mtp_layer(2, true, cur_pos + 2);

        int32_t next_draft = 0;
        cuda_ok(cudaMemcpyAsync(&next_draft, rtl.d_mtp_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, rtl.stream), "sn mtp acc draft");
        cuda_ok(cudaStreamSynchronize(rtl.stream), "sn mtp acc sync");

        cur_tok   = pred[1];
        draft_tok = next_draft;
        cur_pos  += 2;
    } else {
        out_tokens.push_back(pred[0]);
        rollback_gdn_states();

        embed_tokens_to_last_dev(&pred[0], 1);
        cuda_ok(cudaSetDevice(rtl.device), "sn mtp rej dev");
        cuda_ok(cudaMemcpyAsync(rtl.d_mtp_pos, &h_pos[0], sizeof(int32_t), cudaMemcpyHostToDevice, rtl.stream), "sn mtp rej pos");
        if (graphs_ready_ && mtp_graph_t1_) cuda_ok(cudaGraphLaunch(mtp_graph_t1_, rtl.stream), "sn launch mtp t1");
        else run_mtp_layer(1, true, cur_pos + 1);

        int32_t next_draft = 0;
        cuda_ok(cudaMemcpyAsync(&next_draft, rtl.d_mtp_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, rtl.stream), "sn mtp rej draft");
        cuda_ok(cudaStreamSynchronize(rtl.stream), "sn mtp rej sync");

        cur_tok   = pred[0];
        draft_tok = next_draft;
        cur_pos  += 1;
    }
}

void SenseNovaLoopModel::decode_mtp_sample_step(int32_t & cur_tok,
                                                int32_t & draft_tok,
                                                Sampler::SparseDist & draft_q,
                                                int32_t & cur_pos,
                                                Sampler & sampler,
                                                std::vector<int32_t> & out_tokens,
                                                MtpStats & stats) {
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();
    const int32_t h_tok[2] = {cur_tok, draft_tok};
    const int32_t h_pos[2] = {cur_pos, cur_pos + 1};

    for (auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "sn mtp smp pos dev");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, h_pos, sizeof(h_pos), cudaMemcpyHostToDevice, rt.stream), "sn mtp smp pos");
    }

    for (size_t si = 0; si < segments_.size(); ++si) {
        auto & rt = runtimes_[segments_[si].gpu_idx];
        cuda_ok(cudaSetDevice(rt.device), "sn mtp smp seg dev");
        if (si == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, h_tok, sizeof(h_tok), cudaMemcpyHostToDevice, rt.stream), "sn mtp smp tok");
        } else {
            auto & prev_rt = runtimes_[segments_[si - 1].gpu_idx];
            cuda_ok(cudaStreamWaitEvent(rt.stream, prev_rt.done_event, 0), "sn mtp smp wait");
            const int prev_phys = cfg_.plan[segments_[si - 1].visit_end - 1];
            const bool transfer_mhc = layers_[prev_phys].has_mhc && (prev_phys != cfg_.mhc_end - 1);
            if (transfer_mhc) {
                const size_t b = 4ULL * cfg_.n_embd * sizeof(float);
                cuda_ok(cudaMemcpyPeerAsync(rt.d_res, rt.device, prev_rt.d_res, prev_rt.device, b, rt.stream), "sn mtp smp peer res");
            } else {
                const size_t b = 2ULL * cfg_.n_embd * sizeof(float);
                cuda_ok(cudaMemcpyPeerAsync(rt.d_cur, rt.device, prev_rt.d_cur, prev_rt.device, b, rt.stream), "sn mtp smp peer cur");
            }
        }
        if (graphs_ready_ && seg_graphs_t2_[si]) cuda_ok(cudaGraphLaunch(seg_graphs_t2_[si], rt.stream), "sn launch t2");
        else run_visit_segment(si, 2, true, true, cur_pos + 2);
        if (si + 1 < segments_.size()) cuda_ok(cudaEventRecord(rt.done_event, rt.stream), "sn mtp smp rec");
    }

    auto & rtl = runtimes_.back();
    std::vector<float> h_logits(2ULL * cfg_.n_vocab);
    cuda_ok(cudaSetDevice(rtl.device), "sn mtp smp readout");
    cuda_ok(cudaMemcpyAsync(h_logits.data(), rtl.d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "sn mtp smp logits");
    cuda_ok(cudaStreamSynchronize(rtl.stream), "sn mtp smp sync");

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
        cuda_ok(cudaSetDevice(rtl.device), "sn mtp smp acc dev");
        cuda_ok(cudaMemcpyAsync(rtl.d_mtp_pos, h_pos, sizeof(h_pos), cudaMemcpyHostToDevice, rtl.stream), "sn mtp smp acc pos");
        if (graphs_ready_ && mtp_graph_t2_) cuda_ok(cudaGraphLaunch(mtp_graph_t2_, rtl.stream), "sn launch mtp t2");
        else run_mtp_layer(2, true, cur_pos + 2);

        cuda_ok(cudaMemcpyAsync(mtp_logits.data(), rtl.d_logits, cfg_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "sn mtp smp acc lg");
        cuda_ok(cudaStreamSynchronize(rtl.stream), "sn mtp smp acc sync");

        draft_q   = sampler.filtered_distribution(mtp_logits.data(), cfg_.n_vocab);
        cur_tok   = bonus_tok;
        draft_tok = sampler.sample_dist(draft_q);
        cur_pos  += 2;
    } else {
        out_tokens.push_back(resampled);
        rollback_gdn_states();

        embed_tokens_to_last_dev(&resampled, 1);
        cuda_ok(cudaSetDevice(rtl.device), "sn mtp smp rej dev");
        cuda_ok(cudaMemcpyAsync(rtl.d_mtp_pos, &h_pos[0], sizeof(int32_t), cudaMemcpyHostToDevice, rtl.stream), "sn mtp smp rej pos");
        if (graphs_ready_ && mtp_graph_t1_) cuda_ok(cudaGraphLaunch(mtp_graph_t1_, rtl.stream), "sn launch mtp t1");
        else run_mtp_layer(1, true, cur_pos + 1);

        cuda_ok(cudaMemcpyAsync(mtp_logits.data(), rtl.d_logits, cfg_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, rtl.stream), "sn mtp smp rej lg");
        cuda_ok(cudaStreamSynchronize(rtl.stream), "sn mtp smp rej sync");

        draft_q   = sampler.filtered_distribution(mtp_logits.data(), cfg_.n_vocab);
        cur_tok   = resampled;
        draft_tok = sampler.sample_dist(draft_q);
        cur_pos  += 1;
    }
}

} // namespace rookery
