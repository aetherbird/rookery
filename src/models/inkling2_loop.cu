#include "models/inkling2_loop.h"
#include "kernels/kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <thread>

namespace rookery {

namespace {

inline void cuda_ok(cudaError_t err, const char * msg) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

template <typename T>
T * dev_alloc(int dev, size_t n) {
    cuda_ok(cudaSetDevice(dev), "inkling2 dev_alloc setDevice");
    T * p = nullptr;
    cuda_ok(cudaMalloc(&p, n * sizeof(T)), "inkling2 cudaMalloc");
    return p;
}

std::vector<bool> read_bool_arr(const ShardedGguf & sg, const std::string & key, size_t n_def) {
    std::vector<bool> out(n_def, true);
    if (!sg.has_kv(key)) return out;
    const auto & kv = sg.kv(key);
    if (auto * v = std::get_if<std::vector<bool>>(&kv.data)) {
        out = *v;
    } else if (auto * v = std::get_if<std::vector<uint64_t>>(&kv.data)) {
        out.resize(v->size());
        for (size_t i = 0; i < v->size(); ++i) out[i] = ((*v)[i] != 0);
    } else if (auto * v = std::get_if<std::vector<int64_t>>(&kv.data)) {
        out.resize(v->size());
        for (size_t i = 0; i < v->size(); ++i) out[i] = ((*v)[i] != 0);
    }
    return out;
}

int read_first_int_or_u64(const ShardedGguf & sg, const std::string & key, int def_val) {
    if (!sg.has_kv(key)) return def_val;
    const auto & kv = sg.kv(key);
    if (auto * v = std::get_if<uint64_t>(&kv.data)) return static_cast<int>(*v);
    if (auto * v = std::get_if<int64_t>(&kv.data))  return static_cast<int>(*v);
    if (auto * v = std::get_if<std::vector<uint64_t>>(&kv.data)) return v->empty() ? def_val : static_cast<int>((*v)[0]);
    if (auto * v = std::get_if<std::vector<int64_t>>(&kv.data))  return v->empty() ? def_val : static_cast<int>((*v)[0]);
    return def_val;
}

} // namespace

void Inkling2LoopModel::invalidate_graphs() noexcept {
    if (!runtimes_.empty()) {
        cudaSetDevice(runtimes_[0].device);
        if (graph_all_)      { cudaGraphExecDestroy(graph_all_);      graph_all_      = nullptr; }
        if (graph_prelude_)  { cudaGraphExecDestroy(graph_prelude_);  graph_prelude_  = nullptr; }
        if (graph_postlude_) { cudaGraphExecDestroy(graph_postlude_); graph_postlude_ = nullptr; }
    }
    if (runtimes_.size() >= 2) {
        cudaSetDevice(runtimes_[1].device);
        if (graph_core_)     { cudaGraphExecDestroy(graph_core_);     graph_core_     = nullptr; }
    }
    graphs_ready_ = false;
}

Inkling2LoopModel::~Inkling2LoopModel() {
    invalidate_graphs();
    auto freep = [](void * p) { if (p) cudaFree(p); };
    for (size_t il = 0; il < slot_device_.size(); ++il) {
        cudaSetDevice(slot_device_[il]);
        if (il < k_cache_.size())   freep(k_cache_[il]);
        if (il < v_cache_.size())   freep(v_cache_[il]);
        if (il < conv_k_.size())    freep(conv_k_[il]);
        if (il < conv_v_.size())    freep(conv_v_[il]);
        if (il < conv_attn_.size()) freep(conv_attn_[il]);
        if (il < conv_ffn_.size())  freep(conv_ffn_[il]);
    }
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        freep(rt.d_cur); freep(rt.d_inp_sa); freep(rt.d_ffn_inp);
        freep(rt.d_prelude); freep(rt.d_bank_out);
        for (int i = 0; i < MAX_BLOCKS; ++i) freep(rt.d_blocks[i]);
        for (int i = 0; i < MAX_BANK; ++i)   freep(rt.d_bank[i]);
        freep(rt.d_block_ptrs); freep(rt.d_bank_ptrs);
        freep(rt.d_mixed3); freep(rt.d_messages);
        for (int i = 0; i < 3; ++i) freep(rt.d_inputs[i]);
        freep(rt.d_ctrl_in); freep(rt.d_ctrl_hid); freep(rt.d_ctrl_gain); freep(rt.d_ctrl_forg);
        freep(rt.d_q); freep(rt.d_k_raw); freep(rt.d_v_raw); freep(rt.d_k); freep(rt.d_v);
        freep(rt.d_k_dyn); freep(rt.d_v_dyn); freep(rt.d_r); freep(rt.d_sinks);
        freep(rt.d_bias_table); freep(rt.d_attn_out); freep(rt.d_attn_gate);
        freep(rt.d_wo_raw); freep(rt.d_sub_out);
        freep(rt.d_x_norm); freep(rt.d_router); freep(rt.d_top_ids); freep(rt.d_top_w); freep(rt.d_moe_raw);
        freep(rt.d_tokens); freep(rt.d_pos); freep(rt.d_logits); freep(rt.d_argmax);
        freep(rt.d_scratch);
        if (rt.done_event) cudaEventDestroy(rt.done_event);
        if (rt.cublas)     cublasDestroy(rt.cublas);
        if (rt.stream)     cudaStreamDestroy(rt.stream);
    }
}

void Inkling2LoopModel::load(const std::string & gguf_path,
                             const std::vector<int> & gpu_ids,
                             int max_ctx,
                             bool kv_q8_0) {
    if (gpu_ids.empty()) throw std::runtime_error("Inkling2LoopModel::load requires >= 1 GPU");

    const auto t0 = std::chrono::steady_clock::now();
    auto sg = ShardedGguf::open(gguf_path);
    const std::string arch = sg.get_str("general.architecture");
    if (arch != "inkling2_loop" && arch != "inkling2") {
        throw std::runtime_error("Expected inkling2_loop or inkling2 GGUF, got: " + arch);
    }
    const std::string pfx = arch + ".";

    cfg_.n_phys               = static_cast<int>(sg.get_u64(pfx + "block_count", 24));
    cfg_.n_embd               = static_cast<int>(sg.get_u64(pfx + "embedding_length", 2304));
    cfg_.n_head               = read_first_int_or_u64(sg, pfx + "attention.head_count", 24);
    cfg_.n_head_kv            = read_first_int_or_u64(sg, pfx + "attention.head_count_kv", 8);
    cfg_.head_dim             = static_cast<int>(sg.get_u64(pfx + "attention.key_length", 128));
    cfg_.n_expert             = static_cast<int>(sg.get_u64(pfx + "expert_count", 128));
    cfg_.n_expert_used        = read_first_int_or_u64(sg, pfx + "expert_used_count", 8);
    cfg_.n_ff_exp             = read_first_int_or_u64(sg, pfx + "expert_feed_forward_length", 1152);
    cfg_.n_vocab_padded       = static_cast<int>(sg.get_u64(pfx + "vocab_size", 201024));
    cfg_.n_vocab_unpadded     = static_cast<int>(sg.get_u64(pfx + "vocab_size_unpadded", 200058));
    cfg_.attn_res_block_size  = static_cast<int>(sg.get_u64(pfx + "attn_res.block_size", 6));
    cfg_.swa_window           = static_cast<int>(sg.get_u64(pfx + "attention.sliding_window", 1024));
    cfg_.n_conv               = static_cast<int>(sg.get_u64(pfx + "shortconv.l_cache", 4));
    cfg_.n_rel                = static_cast<int>(sg.get_u64(pfx + "attention.relative_length", 16));
    cfg_.n_rel_extent         = static_cast<int>(sg.get_u64(pfx + "attention.relative_extent", 1024));
    cfg_.qk_clip              = static_cast<float>(sg.get_f64(pfx + "attention.qk_clip", 22.627417));
    cfg_.rel_clip             = static_cast<float>(sg.get_f64(pfx + "attention.relative_clip", 8.0));
    cfg_.log_scale_floor      = static_cast<int>(sg.get_u64(pfx + "attention.log_scale_floor", 8192));
    cfg_.log_scale_alpha      = static_cast<float>(sg.get_f64(pfx + "attention.log_scale_alpha", 0.1));
    cfg_.situ_beta            = static_cast<float>(sg.get_f64(pfx + "activation.situ_beta", 4.0));
    cfg_.situ_linear_beta     = static_cast<float>(sg.get_f64(pfx + "activation.situ_linear_beta", 25.0));
    cfg_.expert_weights_scale = static_cast<float>(sg.get_f64(pfx + "expert_weights_scale", 8.0));
    cfg_.logit_softcap        = static_cast<float>(sg.get_f64(pfx + "final_logit_softcapping", 30.0));
    cfg_.rms_eps              = static_cast<float>(sg.get_f64(pfx + "attention.layer_norm_rms_epsilon", 1e-6));
    cfg_.max_ctx              = max_ctx;
    cfg_.kv_q8_0              = kv_q8_0;
    cfg_.swa_cap              = std::min(max_ctx, std::max(cfg_.swa_window * 4, 4096));
    cfg_.is_swa               = read_bool_arr(sg, pfx + "attention.sliding_window_pattern", cfg_.n_phys);

    if (arch == "inkling2_loop") {
        cfg_.core_start = static_cast<int>(sg.get_u64(pfx + "loop.start", 6));
        cfg_.core_end   = static_cast<int>(sg.get_u64(pfx + "loop.end", 18));
        cfg_.n_passes   = static_cast<int>(sg.get_u64(pfx + "loop.count", 3));
        cfg_.gate_rank  = static_cast<int>(sg.get_u64(pfx + "loop.gate_rank", 64));
        cfg_.n_slots    = cfg_.n_phys + (cfg_.n_passes - 1) * (cfg_.core_end - cfg_.core_start);
    } else {
        cfg_.core_start = cfg_.n_phys;
        cfg_.core_end   = cfg_.n_phys;
        cfg_.n_passes   = 1;
        cfg_.gate_rank  = 0;
        cfg_.n_slots    = cfg_.n_phys;
    }

    // 1-GPU: all on gpu_ids[0]. 2+-GPU: loop-resident 2-GPU split (GPU0 = prelude+postlude+embd/out, GPU1 = entire looped core!).
    const bool two_gpu = (gpu_ids.size() >= 2 && cfg_.n_passes > 1);
    runtimes_.resize(two_gpu ? 2 : 1);
    runtimes_[0].device = gpu_ids[0];
    if (two_gpu) runtimes_[1].device = gpu_ids[1];

    if (two_gpu) {
        int can01 = 0, can10 = 0;
        cudaDeviceCanAccessPeer(&can01, runtimes_[0].device, runtimes_[1].device);
        cudaDeviceCanAccessPeer(&can10, runtimes_[1].device, runtimes_[0].device);
        if (can01) { cudaSetDevice(runtimes_[0].device); cudaDeviceEnablePeerAccess(runtimes_[1].device, 0); cudaGetLastError(); }
        if (can10) { cudaSetDevice(runtimes_[1].device); cudaDeviceEnablePeerAccess(runtimes_[0].device, 0); cudaGetLastError(); }
    }

    auto phys_dev = [&](int ip) -> int {
        if (!two_gpu) return runtimes_[0].device;
        return (ip >= cfg_.core_start && ip < cfg_.core_end) ? runtimes_[1].device : runtimes_[0].device;
    };

    layers_.resize(cfg_.n_phys);
    slot_device_.resize(cfg_.n_slots, runtimes_[0].device);
    k_cache_.assign(cfg_.n_slots, nullptr);
    v_cache_.assign(cfg_.n_slots, nullptr);
    conv_k_.assign(cfg_.n_slots, nullptr);
    conv_v_.assign(cfg_.n_slots, nullptr);
    conv_attn_.assign(cfg_.n_slots, nullptr);
    conv_ffn_.assign(cfg_.n_slots, nullptr);

    const int d_kv = cfg_.n_head_kv * cfg_.head_dim; // 1024
    auto kv_bytes_for = [&](int cap) -> size_t {
        if (kv_q8_0) return static_cast<size_t>(cap) * (d_kv / 32) * sizeof(BlockQ8_0);
        return static_cast<size_t>(cap) * d_kv * sizeof(uint16_t);
    };

    // Parallel weight upload across the 1 or 2 active GPUs
    std::vector<std::thread> load_threads;
    std::vector<std::exception_ptr> load_errs(runtimes_.size(), nullptr);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        load_threads.emplace_back([&, gi]() {
            try {
                const int dev = runtimes_[gi].device;
                cuda_ok(cudaSetDevice(dev), "inkling2 parallel load setDevice");

                if (gi == 0) {
                    tok_embd_         = pool_.upload(sg, "token_embd.weight",       dev, false);
                    tok_norm_         = pool_.upload(sg, "token_embd_norm.weight",  dev, false);
                    output_res_query_ = pool_.upload(sg, "output_res_query.weight", dev, false);
                    output_norm_      = pool_.upload(sg, "output_norm.weight",      dev, false);
                    output_           = pool_.upload(sg, "output.weight",           dev, false);
                }

                for (int ip = 0; ip < cfg_.n_phys; ++ip) {
                    if (phys_dev(ip) != dev) continue;
                    auto & L = layers_[ip];
                    const std::string p = "blk." + std::to_string(ip) + ".";
                    L.device = dev;
                    L.is_swa = cfg_.is_swa[ip];

                    L.wq              = pool_.upload(sg, p + "attn_q.weight",        dev, false);
                    L.wk              = pool_.upload(sg, p + "attn_k.weight",        dev, false);
                    L.wv              = pool_.upload(sg, p + "attn_v.weight",        dev, false);
                    L.wo              = pool_.upload(sg, p + "attn_output.weight",   dev, false);
                    L.wqkv_gate       = pool_.upload(sg, p + "attn_gate.weight",     dev, false);
                    L.attn_q_clip     = pool_.upload(sg, p + "attn_q_clip.weight",   dev, false);
                    L.attn_sink_proj  = pool_.upload(sg, p + "attn_sink_proj.weight",dev, false);
                    L.attn_rel        = pool_.upload(sg, p + "attn_rel.weight",      dev, false);
                    L.attn_rel_bias   = pool_.upload(sg, p + "attn_rel_bias.weight", dev, false);
                    L.attn_rel_clip   = pool_.upload(sg, p + "attn_rel_clip.weight", dev, false);
                    L.attn_k_conv     = pool_.upload(sg, p + "attn_k_conv.weight",   dev, false);
                    L.attn_v_conv     = pool_.upload(sg, p + "attn_v_conv.weight",   dev, false);
                    L.attn_post_conv  = pool_.upload(sg, p + "attn_post_conv.weight",dev, false);
                    L.ffn_post_conv   = pool_.upload(sg, p + "ffn_post_conv.weight", dev, false);

                    if (L.is_swa) {
                        L.attn_norm = pool_.upload(sg, p + "attn_norm.weight", dev, false);
                    } else {
                        L.attn_q_inp_norm  = pool_.upload(sg, p + "attn_q_inp_norm.weight",  dev, false);
                        L.attn_k_inp_norm  = pool_.upload(sg, p + "attn_k_inp_norm.weight",  dev, false);
                        L.attn_v_inp_norm  = pool_.upload(sg, p + "attn_v_inp_norm.weight",  dev, false);
                        L.attn_k_conv_proj = pool_.upload(sg, p + "attn_k_conv_proj.weight", dev, false);
                        L.attn_v_conv_proj = pool_.upload(sg, p + "attn_v_conv_proj.weight", dev, false);
                    }

                    if (ip > 0) {
                        L.attn_res_query = pool_.upload(sg, p + "attn_res_query.weight", dev, false);
                    }
                    L.ffn_res_query    = pool_.upload(sg, p + "ffn_res_query.weight",    dev, false);
                    L.ffn_norm         = pool_.upload(sg, p + "ffn_norm.weight",         dev, false);
                    L.ffn_gate_inp     = pool_.upload(sg, p + "ffn_gate_inp.weight",     dev, false);
                    L.ffn_exp_probs_b  = pool_.upload(sg, p + "exp_probs_b.bias",        dev, false, false);
                    L.ffn_gate_up_exps = pool_.upload(sg, p + "ffn_gate_up_exps.weight", dev, false);
                    L.ffn_down_exps    = pool_.upload(sg, p + "ffn_down_exps.weight",    dev, false);

                    if (ip >= cfg_.core_start && ip < cfg_.core_end && cfg_.n_passes > 1) {
                        L.has_loop_ctrl = true;
                        L.loop.msg_query = pool_.upload(sg, p + "loop_msg_query.weight", dev, false);
                        static const char * cnames[4] = {"q", "k", "v", "ffn"};
                        for (int c = 0; c < 4; ++c) {
                            const std::string cpfx = p + "loop_" + cnames[c] + "_ctrl_";
                            L.loop.ctrl_down[c]   = pool_.upload(sg, cpfx + "down.weight",   dev, false);
                            L.loop.ctrl_gain[c]   = pool_.upload(sg, cpfx + "gain.weight",   dev, false);
                            L.loop.ctrl_forget[c] = pool_.upload(sg, cpfx + "forget.weight", dev, false);
                        }
                    }
                }

                // Allocate logical KV caches & ShortConv states for slots owned by this GPU
                for (int il = 0; il < cfg_.n_slots; ++il) {
                    const int ip = (il < cfg_.n_phys)
                        ? il
                        : (cfg_.core_start + (il - cfg_.n_phys) % (cfg_.core_end - cfg_.core_start));
                    if (phys_dev(ip) != dev) continue;
                    slot_device_[il] = dev;
                    const int cap = cfg_.is_swa[ip] ? cfg_.swa_cap : cfg_.max_ctx;
                    const size_t kb = kv_bytes_for(cap);
                    cuda_ok(cudaMalloc(&k_cache_[il], kb), "inkling2 malloc k_cache");
                    cuda_ok(cudaMalloc(&v_cache_[il], kb), "inkling2 malloc v_cache");
                    conv_k_[il]    = dev_alloc<float>(dev, static_cast<size_t>(cfg_.n_conv - 1) * d_kv);
                    conv_v_[il]    = dev_alloc<float>(dev, static_cast<size_t>(cfg_.n_conv - 1) * d_kv);
                    conv_attn_[il] = dev_alloc<float>(dev, static_cast<size_t>(cfg_.n_conv - 1) * cfg_.n_embd);
                    conv_ffn_[il]  = dev_alloc<float>(dev, static_cast<size_t>(cfg_.n_conv - 1) * cfg_.n_embd);
                }

                cuda_ok(cudaDeviceSynchronize(), "inkling2 load sync");
            } catch (...) {
                load_errs[gi] = std::current_exception();
            }
        });
    }
    for (auto & th : load_threads) th.join();
    for (auto & ep : load_errs) if (ep) std::rethrow_exception(ep);
    pool_.release_staging();

    // Allocate per-GPU runtime buffers & cuBLAS handles
    const int d_q      = cfg_.n_head * cfg_.head_dim; // 3072
    const int n_ext    = std::max(cfg_.swa_window, cfg_.n_rel_extent);
    const size_t B     = static_cast<size_t>(max_batch_);
    const size_t H     = static_cast<size_t>(cfg_.n_embd);
    const size_t scratch_b = 128ULL * 1024 * 1024; // 128 MiB: covers FlashAttn Global prefill + grouped MoE + Q8_0 cuBLAS

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        const int dev = rt.device;
        cuda_ok(cudaSetDevice(dev), "inkling2 rt setDevice");
        cuda_ok(cudaStreamCreate(&rt.stream), "inkling2 stream");
        cuda_ok(cudaEventCreateWithFlags(&rt.done_event, cudaEventDisableTiming), "inkling2 event");
        if (cublasCreate(&rt.cublas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        if (device_has_tensor_cores(dev)) cublasSetMathMode(rt.cublas, CUBLAS_TENSOR_OP_MATH);

        rt.d_cur      = dev_alloc<float>(dev, B * H);
        rt.d_inp_sa   = dev_alloc<float>(dev, B * H);
        rt.d_ffn_inp  = dev_alloc<float>(dev, B * H);
        rt.d_prelude  = dev_alloc<float>(dev, B * H);
        rt.d_bank_out = dev_alloc<float>(dev, B * H);
        for (int i = 0; i < MAX_BLOCKS; ++i) rt.d_blocks[i] = dev_alloc<float>(dev, B * H);
        for (int i = 0; i < MAX_BANK; ++i)   rt.d_bank[i]   = dev_alloc<float>(dev, B * H);

        cuda_ok(cudaMalloc(&rt.d_block_ptrs, MAX_BLOCKS * sizeof(const float *)), "malloc d_block_ptrs");
        cuda_ok(cudaMalloc(&rt.d_bank_ptrs,  MAX_BANK   * sizeof(const float *)), "malloc d_bank_ptrs");
        const float * h_blk_ptrs[MAX_BLOCKS];
        for (int i = 0; i < MAX_BLOCKS; ++i) h_blk_ptrs[i] = rt.d_blocks[i];
        const float * h_bnk_ptrs[MAX_BANK];
        for (int i = 0; i < MAX_BANK; ++i)   h_bnk_ptrs[i] = rt.d_bank[i];
        cuda_ok(cudaMemcpy(rt.d_block_ptrs, h_blk_ptrs, sizeof(h_blk_ptrs), cudaMemcpyHostToDevice), "copy d_block_ptrs");
        cuda_ok(cudaMemcpy(rt.d_bank_ptrs,  h_bnk_ptrs, sizeof(h_bnk_ptrs), cudaMemcpyHostToDevice), "copy d_bank_ptrs");

        rt.d_mixed3    = dev_alloc<float>(dev, 3 * B * H);
        rt.d_messages  = dev_alloc<float>(dev, 4 * B * H);
        for (int i = 0; i < 3; ++i) rt.d_inputs[i] = dev_alloc<float>(dev, B * H);
        rt.d_ctrl_in   = dev_alloc<float>(dev, B * 2 * H);
        rt.d_ctrl_hid  = dev_alloc<float>(dev, B * std::max(1, cfg_.gate_rank));
        rt.d_ctrl_gain = dev_alloc<float>(dev, B * H);
        rt.d_ctrl_forg = dev_alloc<float>(dev, B * H);

        rt.d_q          = dev_alloc<float>(dev, B * d_q);
        rt.d_k_raw      = dev_alloc<float>(dev, B * d_kv);
        rt.d_v_raw      = dev_alloc<float>(dev, B * d_kv);
        rt.d_k          = dev_alloc<float>(dev, B * d_kv);
        rt.d_v          = dev_alloc<float>(dev, B * d_kv);
        rt.d_k_dyn      = dev_alloc<float>(dev, B * cfg_.n_head_kv * cfg_.n_conv);
        rt.d_v_dyn      = dev_alloc<float>(dev, B * cfg_.n_head_kv * cfg_.n_conv);
        rt.d_r          = dev_alloc<float>(dev, B * cfg_.n_head * cfg_.n_rel);
        rt.d_sinks      = dev_alloc<float>(dev, B * cfg_.n_head);
        rt.d_bias_table = dev_alloc<float>(dev, B * cfg_.n_head * (n_ext + 1));
        rt.d_attn_out   = dev_alloc<float>(dev, B * d_q);
        rt.d_attn_gate  = dev_alloc<float>(dev, B * d_q);
        rt.d_wo_raw     = dev_alloc<float>(dev, B * H);
        rt.d_sub_out    = dev_alloc<float>(dev, B * H);

        rt.d_x_norm     = dev_alloc<float>(dev, B * H);
        rt.d_router     = dev_alloc<float>(dev, B * cfg_.n_expert);
        rt.d_top_ids    = dev_alloc<int32_t>(dev, B * cfg_.n_expert_used);
        rt.d_top_w      = dev_alloc<float>(dev, B * cfg_.n_expert_used);
        rt.d_moe_raw    = dev_alloc<float>(dev, B * H);

        rt.d_pos        = dev_alloc<int32_t>(dev, B);
        if (gi == 0) {
            rt.d_tokens = dev_alloc<int32_t>(dev, B);
            rt.d_logits = dev_alloc<float>(dev, cfg_.n_vocab_padded);
            rt.d_argmax = dev_alloc<int32_t>(dev, 1);
        }

        cuda_ok(cudaMalloc(&rt.d_scratch, scratch_b), "inkling2 malloc scratch");
        rt.scratch_bytes = scratch_b;
    }

    reset_state();

    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    std::printf("[Inkling2] Loaded %s (%d phys -> %d logical slots, core=[%d,%d)x%d) in %.2f s across %zu GPU(s):",
                arch.c_str(), cfg_.n_phys, cfg_.n_slots, cfg_.core_start, cfg_.core_end, cfg_.n_passes,
                sec, runtimes_.size());
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        std::printf(" GPU%d(%.2f GiB weights)", runtimes_[gi].device,
                    pool_.bytes_on_device(runtimes_[gi].device) / (1024.0 * 1024.0 * 1024.0));
    }
    std::printf(" | ctx=%d (SWA cap=%d, KV=%s)\n", cfg_.max_ctx, cfg_.swa_cap, kv_q8_0 ? "Q8_0" : "F16");
}

void Inkling2LoopModel::reset_state() {
    const int d_kv = cfg_.n_head_kv * cfg_.head_dim;
    const size_t kv_bytes   = static_cast<size_t>(cfg_.n_conv - 1) * d_kv * sizeof(float);
    const size_t embd_bytes = static_cast<size_t>(cfg_.n_conv - 1) * cfg_.n_embd * sizeof(float);
    for (int il = 0; il < cfg_.n_slots; ++il) {
        cuda_ok(cudaSetDevice(slot_device_[il]), "inkling2 reset setDevice");
        cuda_ok(cudaMemset(conv_k_[il],    0, kv_bytes),   "zero conv_k");
        cuda_ok(cudaMemset(conv_v_[il],    0, kv_bytes),   "zero conv_v");
        cuda_ok(cudaMemset(conv_attn_[il], 0, embd_bytes), "zero conv_attn");
        cuda_ok(cudaMemset(conv_ffn_[il],  0, embd_bytes), "zero conv_ffn");
    }
}

void Inkling2LoopModel::save_conv_checkpoint(ConvCheckpoint & out, int pos) const {
    const int d_kv = cfg_.n_head_kv * cfg_.head_dim;
    const size_t n_kv   = static_cast<size_t>(cfg_.n_conv - 1) * d_kv;
    const size_t n_embd = static_cast<size_t>(cfg_.n_conv - 1) * cfg_.n_embd;
    out.pos = pos;
    out.conv_k.resize(cfg_.n_slots);
    out.conv_v.resize(cfg_.n_slots);
    out.conv_attn.resize(cfg_.n_slots);
    out.conv_ffn.resize(cfg_.n_slots);
    for (int il = 0; il < cfg_.n_slots; ++il) {
        cuda_ok(cudaSetDevice(slot_device_[il]), "inkling2 save_cp setDevice");
        out.conv_k[il].resize(n_kv);
        out.conv_v[il].resize(n_kv);
        out.conv_attn[il].resize(n_embd);
        out.conv_ffn[il].resize(n_embd);
        cuda_ok(cudaMemcpy(out.conv_k[il].data(),    conv_k_[il],    n_kv   * sizeof(float), cudaMemcpyDeviceToHost), "save conv_k");
        cuda_ok(cudaMemcpy(out.conv_v[il].data(),    conv_v_[il],    n_kv   * sizeof(float), cudaMemcpyDeviceToHost), "save conv_v");
        cuda_ok(cudaMemcpy(out.conv_attn[il].data(), conv_attn_[il], n_embd * sizeof(float), cudaMemcpyDeviceToHost), "save conv_attn");
        cuda_ok(cudaMemcpy(out.conv_ffn[il].data(),  conv_ffn_[il],  n_embd * sizeof(float), cudaMemcpyDeviceToHost), "save conv_ffn");
    }
}

void Inkling2LoopModel::restore_conv_checkpoint(const ConvCheckpoint & cp) {
    const int d_kv = cfg_.n_head_kv * cfg_.head_dim;
    const size_t n_kv   = static_cast<size_t>(cfg_.n_conv - 1) * d_kv;
    const size_t n_embd = static_cast<size_t>(cfg_.n_conv - 1) * cfg_.n_embd;
    for (int il = 0; il < cfg_.n_slots; ++il) {
        cuda_ok(cudaSetDevice(slot_device_[il]), "inkling2 restore_cp setDevice");
        cuda_ok(cudaMemcpy(conv_k_[il],    cp.conv_k[il].data(),    n_kv   * sizeof(float), cudaMemcpyHostToDevice), "restore conv_k");
        cuda_ok(cudaMemcpy(conv_v_[il],    cp.conv_v[il].data(),    n_kv   * sizeof(float), cudaMemcpyHostToDevice), "restore conv_v");
        cuda_ok(cudaMemcpy(conv_attn_[il], cp.conv_attn[il].data(), n_embd * sizeof(float), cudaMemcpyHostToDevice), "restore conv_attn");
        cuda_ok(cudaMemcpy(conv_ffn_[il],  cp.conv_ffn[il].data(),  n_embd * sizeof(float), cudaMemcpyHostToDevice), "restore conv_ffn");
    }
}

void Inkling2LoopModel::run_layer(DevRuntime & rt, int il, int ip, int T, int & n_blocks, bool has_messages, int kv_len_max) {
    const auto & L   = layers_[ip];
    const int H      = cfg_.n_embd;
    const int d_q    = cfg_.n_head * cfg_.head_dim;
    const int d_kv   = cfg_.n_head_kv * cfg_.head_dim;
    const int n_ext  = L.is_swa ? cfg_.swa_window : cfg_.n_rel_extent;
    const int cap    = L.is_swa ? cfg_.swa_cap : cfg_.max_ctx;
    const int swa_w  = L.is_swa ? cfg_.swa_window : 0;
    const size_t TH  = static_cast<size_t>(T) * H;
    cudaStream_t st  = rt.stream;

    // Save inpSA = cur
    cudaMemcpyAsync(rt.d_inp_sa, rt.d_cur, TH * sizeof(float), cudaMemcpyDeviceToDevice, st);

    // 1. Cross-pass messages (when pass > 0)
    if (has_messages) {
        const int n_bank = (cfg_.core_end - cfg_.core_start) / cfg_.attn_res_block_size - 1;
        launch_inkling2_depth_mix(rt.d_bank_ptrs, n_bank, rt.d_bank_out,
                                  static_cast<const float *>(L.loop.msg_query.ptr),
                                  rt.d_messages, H, 4, T, cfg_.rms_eps, st);
    }

    auto apply_ctrl = [&](const float * d_in, int ctrl_idx, float * d_out) {
        const float * d_msg = rt.d_messages + static_cast<size_t>(ctrl_idx) * TH;
        launch_inkling2_rmsnorm_concat2(d_in, d_msg, rt.d_ctrl_in, H, T, cfg_.rms_eps, st);
        launch_linear(rt.cublas, L.loop.ctrl_down[ctrl_idx],   rt.d_ctrl_in,  rt.d_ctrl_hid,  T, rt.d_scratch, rt.scratch_bytes, st);
        launch_linear(rt.cublas, L.loop.ctrl_gain[ctrl_idx],   rt.d_ctrl_hid, rt.d_ctrl_gain, T, rt.d_scratch, rt.scratch_bytes, st);
        launch_linear(rt.cublas, L.loop.ctrl_forget[ctrl_idx], rt.d_ctrl_hid, rt.d_ctrl_forg, T, rt.d_scratch, rt.scratch_bytes, st);
        launch_inkling2_loop_combine(d_in, d_msg, rt.d_ctrl_gain, rt.d_ctrl_forg, d_out, static_cast<int64_t>(TH), st);
    };

    // 2. Attention inputs (Block Attention Residuals + Loop Control + RMSNorm)
    const float * inp_q = rt.d_cur;
    const float * inp_k = rt.d_cur;
    const float * inp_v = rt.d_cur;
    if (L.attn_res_query.valid()) {
        const int n_q = L.is_swa ? 1 : 3;
        launch_inkling2_depth_mix(rt.d_block_ptrs, n_blocks, rt.d_cur,
                                  static_cast<const float *>(L.attn_res_query.ptr),
                                  rt.d_mixed3, H, n_q, T, cfg_.rms_eps, st);
        inp_q = rt.d_mixed3;
        inp_k = L.is_swa ? rt.d_mixed3 : (rt.d_mixed3 + TH);
        inp_v = L.is_swa ? rt.d_mixed3 : (rt.d_mixed3 + 2 * TH);
    }

    if (has_messages) {
        apply_ctrl(inp_q, 0, rt.d_inputs[0]);
        apply_ctrl(inp_k, 1, rt.d_inputs[1]);
        apply_ctrl(inp_v, 2, rt.d_inputs[2]);
        inp_q = rt.d_inputs[0];
        inp_k = rt.d_inputs[1];
        inp_v = rt.d_inputs[2];
    }

    float * norm_q = rt.d_inputs[0];
    float * norm_k = rt.d_inputs[1];
    float * norm_v = rt.d_inputs[2];
    if (L.is_swa && !has_messages) {
        launch_rmsnorm(inp_q, static_cast<const float *>(L.attn_norm.ptr), norm_q, H, T, cfg_.rms_eps, st);
        norm_k = norm_q;
        norm_v = norm_q;
    } else if (L.is_swa) {
        launch_rmsnorm(inp_q, static_cast<const float *>(L.attn_norm.ptr), norm_q, H, T, cfg_.rms_eps, st);
        launch_rmsnorm(inp_k, static_cast<const float *>(L.attn_norm.ptr), norm_k, H, T, cfg_.rms_eps, st);
        launch_rmsnorm(inp_v, static_cast<const float *>(L.attn_norm.ptr), norm_v, H, T, cfg_.rms_eps, st);
    } else {
        launch_rmsnorm(inp_q, static_cast<const float *>(L.attn_q_inp_norm.ptr), norm_q, H, T, cfg_.rms_eps, st);
        launch_rmsnorm(inp_k, static_cast<const float *>(L.attn_k_inp_norm.ptr), norm_k, H, T, cfg_.rms_eps, st);
        launch_rmsnorm(inp_v, static_cast<const float *>(L.attn_v_inp_norm.ptr), norm_v, H, T, cfg_.rms_eps, st);
    }

    // 3. Attention Q, K, V, R, Sink, Gate projections (reuse Q8_1 quantization on T < 16!)
    if (T < 16 && L.wq.dtype == DType::Q8_0) {
        launch_quantize_q8_1(norm_q, rt.d_scratch, H, T, st);
        launch_linear_q8_0_prequant(L.wq,             rt.d_scratch, rt.d_q,         T, st);
        launch_linear_q8_0_prequant(L.attn_rel,       rt.d_scratch, rt.d_r,         T, st);
        launch_linear_q8_0_prequant(L.attn_sink_proj, rt.d_scratch, rt.d_sinks,     T, st);
        launch_linear_q8_0_prequant(L.wqkv_gate,      rt.d_scratch, rt.d_attn_gate, T, st);
        if (norm_k == norm_q) {
            launch_linear_q8_0_prequant(L.wk, rt.d_scratch, rt.d_k_raw, T, st);
            launch_linear_q8_0_prequant(L.wv, rt.d_scratch, rt.d_v_raw, T, st);
        } else {
            launch_quantize_q8_1(norm_k, rt.d_scratch, H, T, st);
            launch_linear_q8_0_prequant(L.wk, rt.d_scratch, rt.d_k_raw, T, st);
            if (!L.is_swa) launch_linear_q8_0_prequant(L.attn_k_conv_proj, rt.d_scratch, rt.d_k_dyn, T, st);

            launch_quantize_q8_1(norm_v, rt.d_scratch, H, T, st);
            launch_linear_q8_0_prequant(L.wv, rt.d_scratch, rt.d_v_raw, T, st);
            if (!L.is_swa) launch_linear_q8_0_prequant(L.attn_v_conv_proj, rt.d_scratch, rt.d_v_dyn, T, st);
        }
    } else {
        launch_linear(rt.cublas, L.wq,             norm_q, rt.d_q,         T, rt.d_scratch, rt.scratch_bytes, st);
        launch_linear(rt.cublas, L.attn_rel,       norm_q, rt.d_r,         T, rt.d_scratch, rt.scratch_bytes, st);
        launch_linear(rt.cublas, L.attn_sink_proj, norm_q, rt.d_sinks,     T, rt.d_scratch, rt.scratch_bytes, st);
        launch_linear(rt.cublas, L.wqkv_gate,      norm_q, rt.d_attn_gate, T, rt.d_scratch, rt.scratch_bytes, st);
        launch_linear(rt.cublas, L.wk,             norm_k, rt.d_k_raw,     T, rt.d_scratch, rt.scratch_bytes, st);
        launch_linear(rt.cublas, L.wv,             norm_v, rt.d_v_raw,     T, rt.d_scratch, rt.scratch_bytes, st);
        if (!L.is_swa) {
            launch_linear(rt.cublas, L.attn_k_conv_proj, norm_k, rt.d_k_dyn, T, rt.d_scratch, rt.scratch_bytes, st);
            launch_linear(rt.cublas, L.attn_v_conv_proj, norm_v, rt.d_v_dyn, T, rt.d_scratch, rt.scratch_bytes, st);
        }
    }

    // 4. Causal ShortConvs on K & V (+ dynamic per-head taps on global layers)
    launch_inkling2_shortconv(rt.d_k_raw, static_cast<const float *>(L.attn_k_conv.ptr),
                              L.is_swa ? nullptr : rt.d_k_dyn, conv_k_[il], rt.d_k,
                              T, d_kv, cfg_.n_head_kv, st);
    launch_inkling2_shortconv(rt.d_v_raw, static_cast<const float *>(L.attn_v_conv.ptr),
                              L.is_swa ? nullptr : rt.d_v_dyn, conv_v_[il], rt.d_v,
                              T, d_kv, cfg_.n_head_kv, st);

    // 5. L2 Clip (Q, K, R) + Head RMSNorm (V) + Log-scale tau on global Q, R
    const int32_t * tau_pos = L.is_swa ? nullptr : rt.d_pos;
    launch_inkling2_l2_clip(rt.d_q, static_cast<const float *>(L.attn_q_clip.ptr), tau_pos,
                            cfg_.qk_clip, static_cast<float>(cfg_.log_scale_floor), cfg_.log_scale_alpha,
                            T, cfg_.n_head, cfg_.head_dim, st);
    launch_inkling2_l2_clip(rt.d_k, nullptr, nullptr, cfg_.qk_clip, 1.0f, 0.0f,
                            T, cfg_.n_head_kv, cfg_.head_dim, st);
    launch_inkling2_head_rmsnorm(rt.d_v, T, cfg_.n_head_kv, cfg_.head_dim, cfg_.rms_eps, st);
    launch_inkling2_l2_clip(rt.d_r, static_cast<const float *>(L.attn_rel_clip.ptr), tau_pos,
                            cfg_.rel_clip, static_cast<float>(cfg_.log_scale_floor), cfg_.log_scale_alpha,
                            T, cfg_.n_head, cfg_.n_rel, st);

    // 6. KV Write + Relative Bias Table + Flash/Split-K Attention with Zero-Value Sink
    launch_inkling2_kv_write(rt.d_k, rt.d_v, k_cache_[il], v_cache_[il], cfg_.kv_q8_0,
                             rt.d_pos, T, cfg_.n_head_kv, cfg_.head_dim, cap, st);
    launch_inkling2_rel_bias_table(rt.d_r, static_cast<const float *>(L.attn_rel_bias.ptr),
                                   rt.d_sinks, rt.d_bias_table, T, cfg_.n_head, cfg_.n_rel, n_ext, st);
    launch_inkling2_attn(rt.d_q, k_cache_[il], v_cache_[il], cfg_.kv_q8_0,
                         rt.d_bias_table, rt.d_pos, rt.d_attn_out,
                         rt.d_scratch, rt.scratch_bytes,
                         T, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim,
                         swa_w, n_ext, cap, 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim)),
                         rt.cublas, kv_len_max, st);

    // 7. Sigmoid Gate + Wo + Post-Attention ShortConv
    launch_inkling2_sigmoid_gate_mul(rt.d_attn_out, rt.d_attn_gate, static_cast<int64_t>(T) * d_q, st);
    launch_linear(rt.cublas, L.wo, rt.d_attn_out, rt.d_wo_raw, T, rt.d_scratch, rt.scratch_bytes, st);
    launch_inkling2_shortconv(rt.d_wo_raw, static_cast<const float *>(L.attn_post_conv.ptr),
                              nullptr, conv_attn_[il], rt.d_cur, T, H, 0, st);

    // 8. Block Residual Update: if ip % block_size == 0, push inpSA to blocks (no add); else cur += inpSA
    if (ip % cfg_.attn_res_block_size == 0) {
        cudaMemcpyAsync(rt.d_blocks[n_blocks], rt.d_inp_sa, TH * sizeof(float), cudaMemcpyDeviceToDevice, st);
        ++n_blocks;
    } else {
        launch_add_inplace(rt.d_cur, rt.d_inp_sa, static_cast<int64_t>(TH), st);
    }

    // Save ffn_inp = cur
    cudaMemcpyAsync(rt.d_ffn_inp, rt.d_cur, TH * sizeof(float), cudaMemcpyDeviceToDevice, st);

    // 9. FFN Depth Mix + Loop Control + RMSNorm
    launch_inkling2_depth_mix(rt.d_block_ptrs, n_blocks, rt.d_cur,
                              static_cast<const float *>(L.ffn_res_query.ptr),
                              rt.d_mixed3, H, 1, T, cfg_.rms_eps, st);
    const float * ffn_x = rt.d_mixed3;
    if (has_messages) {
        apply_ctrl(ffn_x, 3, rt.d_inputs[0]);
        ffn_x = rt.d_inputs[0];
    }
    launch_rmsnorm(ffn_x, static_cast<const float *>(L.ffn_norm.ptr), rt.d_x_norm, H, T, cfg_.rms_eps, st);

    // 10. MoE Router (Sigmoid + bias + renorm + scale=8.0) + Fused SiTU MoE + Post-FFN ShortConv + Residual
    launch_linear(rt.cublas, L.ffn_gate_inp, rt.d_x_norm, rt.d_router, T, rt.d_scratch, rt.scratch_bytes, st);
    launch_moe_route_sigmoid(rt.d_router,
                             L.ffn_exp_probs_b.valid() ? static_cast<const float *>(L.ffn_exp_probs_b.ptr) : nullptr,
                             rt.d_top_ids, rt.d_top_w,
                             T, cfg_.n_expert, cfg_.n_expert_used, true, cfg_.expert_weights_scale, st);
    launch_inkling2_moe_ffn(rt.cublas, L.ffn_gate_up_exps, L.ffn_down_exps,
                            rt.d_x_norm, rt.d_top_ids, rt.d_top_w, rt.d_moe_raw,
                            rt.d_scratch, rt.scratch_bytes,
                            T, cfg_.n_expert_used, cfg_.situ_beta, cfg_.situ_linear_beta, st);
    launch_inkling2_shortconv(rt.d_moe_raw, static_cast<const float *>(L.ffn_post_conv.ptr),
                              nullptr, conv_ffn_[il], rt.d_sub_out, T, H, 0, st);

    cudaMemcpyAsync(rt.d_cur, rt.d_ffn_inp, TH * sizeof(float), cudaMemcpyDeviceToDevice, st);
    launch_add_inplace(rt.d_cur, rt.d_sub_out, static_cast<int64_t>(TH), st);
}

void Inkling2LoopModel::run_prelude(int T, int kv_len_max) {
    auto & rt = runtimes_[0];
    cudaSetDevice(rt.device);
    const int H = cfg_.n_embd;
    launch_embedding_lookup(tok_embd_, rt.d_tokens, rt.d_inp_sa, H, 1, T, rt.stream);
    launch_rmsnorm(rt.d_inp_sa, static_cast<const float *>(tok_norm_.ptr), rt.d_cur, H, T, cfg_.rms_eps, rt.stream);

    int n_blocks = 0;
    for (int ip = 0; ip < cfg_.core_start; ++ip) {
        run_layer(rt, ip, ip, T, n_blocks, false, kv_len_max);
    }
    cudaMemcpyAsync(rt.d_prelude, rt.d_cur, static_cast<size_t>(T) * H * sizeof(float), cudaMemcpyDeviceToDevice, rt.stream);
}

void Inkling2LoopModel::run_core(int T, int kv_len_max) {
    auto & rt = (runtimes_.size() >= 2) ? runtimes_[1] : runtimes_[0];
    cudaSetDevice(rt.device);
    const int H = cfg_.n_embd;
    const size_t TH_bytes = static_cast<size_t>(T) * H * sizeof(float);
    const int n_pre_blk   = cfg_.core_start / cfg_.attn_res_block_size;
    const int n_bank      = (cfg_.core_end - cfg_.core_start) / cfg_.attn_res_block_size - 1;

    int n_blocks = n_pre_blk;
    for (int pass = 0; pass < cfg_.n_passes; ++pass) {
        if (pass > 0) {
            for (int k = 0; k < n_bank; ++k) {
                cudaMemcpyAsync(rt.d_bank[k], rt.d_blocks[n_pre_blk + 1 + k], TH_bytes, cudaMemcpyDeviceToDevice, rt.stream);
            }
            cudaMemcpyAsync(rt.d_bank_out, rt.d_cur, TH_bytes, cudaMemcpyDeviceToDevice, rt.stream);
            cudaMemcpyAsync(rt.d_cur, rt.d_prelude, TH_bytes, cudaMemcpyDeviceToDevice, rt.stream);
            n_blocks = n_pre_blk;
        }
        for (int ip = cfg_.core_start; ip < cfg_.core_end; ++ip) {
            const int il = (pass == 0)
                ? ip
                : (cfg_.n_phys + (pass - 1) * (cfg_.core_end - cfg_.core_start) + (ip - cfg_.core_start));
            run_layer(rt, il, ip, T, n_blocks, pass > 0, kv_len_max);
        }
    }
}

void Inkling2LoopModel::run_postlude_and_head(int T, bool run_argmax, int kv_len_max) {
    auto & rt = runtimes_[0];
    cudaSetDevice(rt.device);
    const int H = cfg_.n_embd;
    int n_blocks = cfg_.core_end / cfg_.attn_res_block_size; // 3 for core_end=18

    for (int ip = cfg_.core_end; ip < cfg_.n_phys; ++ip) {
        run_layer(rt, ip, ip, T, n_blocks, false, kv_len_max);
    }

    // Output head: depth mix -> RMSNorm -> lm_head (last token only!) -> softcap(30) -> argmax over n_vocab_unpadded
    launch_inkling2_depth_mix(rt.d_block_ptrs, n_blocks, rt.d_cur,
                              static_cast<const float *>(output_res_query_.ptr),
                              rt.d_mixed3, H, 1, T, cfg_.rms_eps, rt.stream);
    launch_rmsnorm(rt.d_mixed3 + static_cast<size_t>(T - 1) * H,
                   static_cast<const float *>(output_norm_.ptr),
                   rt.d_x_norm, H, 1, cfg_.rms_eps, rt.stream);
    launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, rt.stream);
    launch_inkling2_logit_softcap(rt.d_logits, cfg_.n_vocab_unpadded, cfg_.logit_softcap, rt.stream);
    if (run_argmax) {
        launch_argmax_f32(rt.d_logits, rt.d_argmax, cfg_.n_vocab_unpadded, rt.stream);
    }
}

void Inkling2LoopModel::run_single_gpu_all(int T, bool run_argmax, int kv_len_max) {
    run_prelude(T, kv_len_max);
    run_core(T, kv_len_max);
    run_postlude_and_head(T, run_argmax, kv_len_max);
}

void Inkling2LoopModel::capture_decode_graphs() {
    invalidate_graphs();
    if (!use_cuda_graphs_) return;

    // Warm up cuBLAS/kernels on T=1 outside capture first? Graph capture is safe since T=1 uses pure CUDA kernels (no cuBLAS!).
    if (runtimes_.size() == 1) {
        auto & rt = runtimes_[0];
        cuda_ok(cudaSetDevice(rt.device), "capture 1gpu setDevice");
        cuda_ok(cudaStreamSynchronize(rt.stream), "pre-capture sync");
        cudaGraph_t g = nullptr;
        cuda_ok(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeRelaxed), "begin capture all");
        run_single_gpu_all(1, true, 0);
        cuda_ok(cudaStreamEndCapture(rt.stream, &g), "end capture all");
        cuda_ok(cudaGraphInstantiate(&graph_all_, g, nullptr, nullptr, 0), "instantiate all");
        cudaGraphDestroy(g);
    } else {
        auto & rt0 = runtimes_[0];
        auto & rt1 = runtimes_[1];
        cuda_ok(cudaSetDevice(rt0.device), "capture 2gpu rt0");
        cuda_ok(cudaStreamSynchronize(rt0.stream), "sync rt0");
        {
            cudaGraph_t g = nullptr;
            cuda_ok(cudaStreamBeginCapture(rt0.stream, cudaStreamCaptureModeRelaxed), "begin prelude");
            run_prelude(1, 0);
            cuda_ok(cudaStreamEndCapture(rt0.stream, &g), "end prelude");
            cuda_ok(cudaGraphInstantiate(&graph_prelude_, g, nullptr, nullptr, 0), "inst prelude");
            cudaGraphDestroy(g);
        }
        {
            cudaGraph_t g = nullptr;
            cuda_ok(cudaStreamBeginCapture(rt0.stream, cudaStreamCaptureModeRelaxed), "begin postlude");
            run_postlude_and_head(1, true, 0);
            cuda_ok(cudaStreamEndCapture(rt0.stream, &g), "end postlude");
            cuda_ok(cudaGraphInstantiate(&graph_postlude_, g, nullptr, nullptr, 0), "inst postlude");
            cudaGraphDestroy(g);
        }
        cuda_ok(cudaSetDevice(rt1.device), "capture 2gpu rt1");
        cuda_ok(cudaStreamSynchronize(rt1.stream), "sync rt1");
        {
            cudaGraph_t g = nullptr;
            cuda_ok(cudaStreamBeginCapture(rt1.stream, cudaStreamCaptureModeRelaxed), "begin core");
            run_core(1, 0);
            cuda_ok(cudaStreamEndCapture(rt1.stream, &g), "end core");
            cuda_ok(cudaGraphInstantiate(&graph_core_, g, nullptr, nullptr, 0), "inst core");
            cudaGraphDestroy(g);
        }
    }
    graphs_ready_ = true;
}

void Inkling2LoopModel::warmup_decode_graphs() {
    if (!use_cuda_graphs_ || graphs_ready_) return;
    capture_decode_graphs();
}

void Inkling2LoopModel::forward(const std::vector<int32_t> & tokens,
                                const std::vector<int32_t> & positions,
                                std::vector<float> & out_logits) {
    const int T = static_cast<int>(tokens.size());
    if (T == 0 || T != static_cast<int>(positions.size())) {
        throw std::runtime_error("Inkling2LoopModel::forward invalid token/pos sizes");
    }
    if (T > max_batch_) {
        throw std::runtime_error("Inkling2LoopModel::forward T > max_batch_");
    }

    int max_p = 0;
    for (int32_t p : positions) if (p > max_p) max_p = p;
    const int kv_len_max = max_p + 1;

    auto & rt0 = runtimes_[0];
    cuda_ok(cudaSetDevice(rt0.device), "fwd setDevice0");
    cuda_ok(cudaMemcpyAsync(rt0.d_tokens, tokens.data(),    T * sizeof(int32_t), cudaMemcpyHostToDevice, rt0.stream), "copy tokens");
    cuda_ok(cudaMemcpyAsync(rt0.d_pos,    positions.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rt0.stream), "copy pos0");

    if (runtimes_.size() == 1) {
        run_single_gpu_all(T, false, kv_len_max);
    } else {
        auto & rt1 = runtimes_[1];
        const size_t TH_bytes = static_cast<size_t>(T) * cfg_.n_embd * sizeof(float);
        const int n_pre_blk   = cfg_.core_start / cfg_.attn_res_block_size;
        const int n_core_blk  = (cfg_.core_end - cfg_.core_start) / cfg_.attn_res_block_size;

        cuda_ok(cudaSetDevice(rt1.device), "fwd setDevice1");
        cuda_ok(cudaMemcpyAsync(rt1.d_pos, positions.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rt1.stream), "copy pos1");

        // Stage 0: Prelude on GPU 0
        run_prelude(T, kv_len_max);
        cuda_ok(cudaStreamSynchronize(rt0.stream), "sync prelude");

        // Hop 1: GPU0 -> GPU1 (cur, prelude, blocks[0..n_pre_blk-1])
        cuda_ok(cudaSetDevice(rt1.device), "hop1 setDevice1");
        cuda_ok(cudaMemcpyPeerAsync(rt1.d_cur,     rt1.device, rt0.d_cur,     rt0.device, TH_bytes, rt1.stream), "hop1 cur");
        cuda_ok(cudaMemcpyPeerAsync(rt1.d_prelude, rt1.device, rt0.d_prelude, rt0.device, TH_bytes, rt1.stream), "hop1 prelude");
        for (int b = 0; b < n_pre_blk; ++b) {
            cuda_ok(cudaMemcpyPeerAsync(rt1.d_blocks[b], rt1.device, rt0.d_blocks[b], rt0.device, TH_bytes, rt1.stream), "hop1 blk");
        }

        // Stage 1: Looped core (3 passes!) on GPU 1
        run_core(T, kv_len_max);
        cuda_ok(cudaStreamSynchronize(rt1.stream), "sync core");

        // Hop 2: GPU1 -> GPU0 (cur, blocks[n_pre_blk .. n_pre_blk + n_core_blk - 1])
        cuda_ok(cudaSetDevice(rt0.device), "hop2 setDevice0");
        cuda_ok(cudaMemcpyPeerAsync(rt0.d_cur, rt0.device, rt1.d_cur, rt1.device, TH_bytes, rt0.stream), "hop2 cur");
        for (int b = 0; b < n_core_blk; ++b) {
            const int idx = n_pre_blk + b;
            cuda_ok(cudaMemcpyPeerAsync(rt0.d_blocks[idx], rt0.device, rt1.d_blocks[idx], rt1.device, TH_bytes, rt0.stream), "hop2 blk");
        }

        // Stage 2: Postlude + Output Head on GPU 0
        run_postlude_and_head(T, false, kv_len_max);
    }

    out_logits.resize(cfg_.n_vocab_unpadded);
    cuda_ok(cudaSetDevice(rt0.device), "fwd readout setDevice0");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), rt0.d_logits,
                            static_cast<size_t>(cfg_.n_vocab_unpadded) * sizeof(float),
                            cudaMemcpyDeviceToHost, rt0.stream), "copy logits");
    cuda_ok(cudaStreamSynchronize(rt0.stream), "fwd sync");
}

int32_t Inkling2LoopModel::forward_decode_greedy(int32_t token, int32_t pos) {
    if (use_cuda_graphs_ && !graphs_ready_) {
        capture_decode_graphs();
    }

    auto & rt0 = runtimes_[0];
    cuda_ok(cudaSetDevice(rt0.device), "dec setDevice0");
    cuda_ok(cudaMemcpyAsync(rt0.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt0.stream), "dec tok");
    cuda_ok(cudaMemcpyAsync(rt0.d_pos,    &pos,   sizeof(int32_t), cudaMemcpyHostToDevice, rt0.stream), "dec pos0");

    if (runtimes_.size() == 1) {
        if (graphs_ready_ && graph_all_) {
            cuda_ok(cudaGraphLaunch(graph_all_, rt0.stream), "launch graph_all");
        } else {
            run_single_gpu_all(1, true, pos + 1);
        }
    } else {
        auto & rt1 = runtimes_[1];
        const size_t H_bytes  = static_cast<size_t>(cfg_.n_embd) * sizeof(float);
        const int n_pre_blk   = cfg_.core_start / cfg_.attn_res_block_size;
        const int n_core_blk  = (cfg_.core_end - cfg_.core_start) / cfg_.attn_res_block_size;

        cuda_ok(cudaSetDevice(rt1.device), "dec setDevice1");
        cuda_ok(cudaMemcpyAsync(rt1.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt1.stream), "dec pos1");

        cuda_ok(cudaSetDevice(rt0.device), "dec stage0");
        if (graphs_ready_ && graph_prelude_) cuda_ok(cudaGraphLaunch(graph_prelude_, rt0.stream), "launch prelude");
        else run_prelude(1, pos + 1);
        cuda_ok(cudaEventRecord(rt0.done_event, rt0.stream), "rec prelude");

        cuda_ok(cudaSetDevice(rt1.device), "dec stage1");
        cuda_ok(cudaStreamWaitEvent(rt1.stream, rt0.done_event, 0), "wait prelude");
        cuda_ok(cudaMemcpyPeerAsync(rt1.d_cur,     rt1.device, rt0.d_cur,     rt0.device, H_bytes, rt1.stream), "dec hop1 cur");
        cuda_ok(cudaMemcpyPeerAsync(rt1.d_prelude, rt1.device, rt0.d_prelude, rt0.device, H_bytes, rt1.stream), "dec hop1 pre");
        for (int b = 0; b < n_pre_blk; ++b) {
            cuda_ok(cudaMemcpyPeerAsync(rt1.d_blocks[b], rt1.device, rt0.d_blocks[b], rt0.device, H_bytes, rt1.stream), "dec hop1 blk");
        }
        if (graphs_ready_ && graph_core_) cuda_ok(cudaGraphLaunch(graph_core_, rt1.stream), "launch core");
        else run_core(1, pos + 1);
        cuda_ok(cudaEventRecord(rt1.done_event, rt1.stream), "rec core");

        cuda_ok(cudaSetDevice(rt0.device), "dec stage2");
        cuda_ok(cudaStreamWaitEvent(rt0.stream, rt1.done_event, 0), "wait core");
        cuda_ok(cudaMemcpyPeerAsync(rt0.d_cur, rt0.device, rt1.d_cur, rt1.device, H_bytes, rt0.stream), "dec hop2 cur");
        for (int b = 0; b < n_core_blk; ++b) {
            const int idx = n_pre_blk + b;
            cuda_ok(cudaMemcpyPeerAsync(rt0.d_blocks[idx], rt0.device, rt1.d_blocks[idx], rt1.device, H_bytes, rt0.stream), "dec hop2 blk");
        }
        if (graphs_ready_ && graph_postlude_) cuda_ok(cudaGraphLaunch(graph_postlude_, rt0.stream), "launch postlude");
        else run_postlude_and_head(1, true, pos + 1);
    }

    int32_t next_tok = 0;
    cuda_ok(cudaSetDevice(rt0.device), "dec readout");
    cuda_ok(cudaMemcpyAsync(&next_tok, rt0.d_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, rt0.stream), "copy argmax");
    cuda_ok(cudaStreamSynchronize(rt0.stream), "dec sync");
    return next_tok;
}

void Inkling2LoopModel::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    forward_decode_greedy(token, pos);
    auto & rt0 = runtimes_[0];
    out_logits.resize(cfg_.n_vocab_unpadded);
    cuda_ok(cudaSetDevice(rt0.device), "dec logits setDevice0");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), rt0.d_logits,
                            static_cast<size_t>(cfg_.n_vocab_unpadded) * sizeof(float),
                            cudaMemcpyDeviceToHost, rt0.stream), "dec copy logits");
    cuda_ok(cudaStreamSynchronize(rt0.stream), "dec logits sync");
}

} // namespace rookery
