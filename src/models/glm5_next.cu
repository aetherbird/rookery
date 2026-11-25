#include "models/glm5_next.h"
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
    cuda_ok(cudaSetDevice(dev), "glm5 dev_alloc setDevice");
    T * p = nullptr;
    cuda_ok(cudaMalloc(&p, n * sizeof(T)), "glm5 cudaMalloc");
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

void Glm5NextModel::invalidate_graphs() noexcept {
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

Glm5NextModel::~Glm5NextModel() {
    invalidate_graphs();
    for (size_t il = 0; il < layers_.size(); ++il) {
        cudaSetDevice(layers_[il].device);
        auto freep = [](void * p) { if (p) cudaFree(p); };
        if (il < kda_conv_q_.size())      freep(kda_conv_q_[il]);
        if (il < kda_conv_k_.size())      freep(kda_conv_k_[il]);
        if (il < kda_conv_v_.size())      freep(kda_conv_v_[il]);
        if (il < kda_state_.size())       freep(kda_state_[il]);
        if (il < kda_conv_q_snap_.size()) freep(kda_conv_q_snap_[il]);
        if (il < kda_conv_k_snap_.size()) freep(kda_conv_k_snap_[il]);
        if (il < kda_conv_v_snap_.size()) freep(kda_conv_v_snap_[il]);
        if (il < kda_state_snap_.size())  freep(kda_state_snap_[il]);
        if (il < mla_kv_cache_.size())    freep(mla_kv_cache_[il]);
    }
    if (mtp_kv_cache_ && !runtimes_.empty()) {
        cudaSetDevice(runtimes_.back().device);
        cudaFree(mtp_kv_cache_);
        mtp_kv_cache_ = nullptr;
    }
    for (auto & rt : runtimes_) {
        cudaSetDevice(rt.device);
        auto freep = [](void * p) { if (p) cudaFree(p); };
        freep(rt.d_inpL); freep(rt.d_inpL_alt); freep(rt.d_mixes); freep(rt.d_post); freep(rt.d_comb);
        freep(rt.d_x_norm); freep(rt.d_sub_out);
        freep(rt.d_kda_q); freep(rt.d_kda_k); freep(rt.d_kda_v);
        freep(rt.d_kda_qc); freep(rt.d_kda_kc); freep(rt.d_kda_vc);
        freep(rt.d_kda_fa); freep(rt.d_kda_g); freep(rt.d_kda_beta);
        freep(rt.d_kda_out); freep(rt.d_kda_gate);
        freep(rt.d_mla_qr); freep(rt.d_mla_q); freep(rt.d_mla_qabs);
        freep(rt.d_mla_kv); freep(rt.d_mla_olat); freep(rt.d_mla_o);
        freep(rt.d_ffn_g); freep(rt.d_ffn_u); freep(rt.d_router);
        freep(rt.d_top_ids); freep(rt.d_top_w);
        freep(rt.d_tokens); freep(rt.d_pos); freep(rt.d_logits); freep(rt.d_argmax);
        freep(rt.d_h_nextn); freep(rt.d_mtp_e); freep(rt.d_mtp_eh);
        freep(rt.d_mtp_cur); freep(rt.d_mtp_pos); freep(rt.d_mtp_argmax);
        freep(rt.d_scratch);
        if (rt.done_event) cudaEventDestroy(rt.done_event);
        if (rt.cublas) cublasDestroy(rt.cublas);
        if (rt.stream) cudaStreamDestroy(rt.stream);
    }
}

void Glm5NextModel::load(const std::string & gguf_path,
                         const std::vector<int> & gpu_ids,
                         const std::vector<float> & split_weights,
                         int max_ctx,
                         bool kv_q8_0,
                         bool load_mtp) {
    if (gpu_ids.empty()) throw std::runtime_error("Glm5NextModel::load requires >= 1 GPU");

    const auto t0 = std::chrono::steady_clock::now();
    auto sg = ShardedGguf::open(gguf_path);
    const std::string arch = sg.get_str("general.architecture");
    if (arch != "glm5-next" && arch != "glm5next") {
        throw std::runtime_error("Expected glm5-next or glm5next GGUF, got: " + arch);
    }
    const std::string pfx = arch + ".";

    cfg_.n_blocks             = static_cast<int>(sg.get_u64(pfx + "block_count", 46));
    cfg_.n_nextn              = static_cast<int>(sg.get_u64(pfx + "nextn_predict_layers", 1));
    cfg_.n_layers             = cfg_.n_blocks - cfg_.n_nextn;
    cfg_.n_dense_lead         = static_cast<int>(sg.get_u64(pfx + "leading_dense_block_count", 3));
    cfg_.n_embd               = static_cast<int>(sg.get_u64(pfx + "embedding_length", 4096));
    cfg_.n_ff_dense           = static_cast<int>(sg.get_u64(pfx + "feed_forward_length", 12288));
    cfg_.n_vocab              = static_cast<int>(sg.get_u64(pfx + "vocab_size", 154880));
    cfg_.n_head               = static_cast<int>(sg.get_u64(pfx + "attention.head_count", 64));
    cfg_.kda_head_dim         = static_cast<int>(sg.get_u64(pfx + "kda.head_dim", 128));
    cfg_.kda_gate_lower_bound = static_cast<float>(sg.get_f64(pfx + "kda.gate_lower_bound", -5.0));
    cfg_.q_lora_rank          = static_cast<int>(sg.get_u64(pfx + "attention.q_lora_rank", 1536));
    cfg_.kv_lora_rank         = static_cast<int>(sg.get_u64(pfx + "attention.kv_lora_rank", 512));
    cfg_.qk_head_dim_mla      = static_cast<int>(sg.get_u64(pfx + "attention.key_length_mla", 256));
    cfg_.v_head_dim_mla       = static_cast<int>(sg.get_u64(pfx + "attention.value_length_mla", 256));
    cfg_.n_expert             = static_cast<int>(sg.get_u64(pfx + "expert_count", 288));
    cfg_.n_expert_used        = static_cast<int>(sg.get_u64(pfx + "expert_used_count", 8));
    cfg_.n_ff_exp             = static_cast<int>(sg.get_u64(pfx + "expert_feed_forward_length", 2048));
    cfg_.n_ff_shexp           = static_cast<int>(sg.get_u64(pfx + "expert_shared_feed_forward_length", 2048));
    cfg_.expert_weights_norm  = sg.get_bool(pfx + "expert_weights_norm", true);
    cfg_.expert_weights_scale = static_cast<float>(sg.get_f64(pfx + "expert_weights_scale", 2.5));
    cfg_.hc_mult              = static_cast<int>(sg.get_u64(pfx + "hyper_connection.count", 4));
    cfg_.hc_sinkhorn_iters    = static_cast<int>(sg.get_u64(pfx + "hyper_connection.sinkhorn_iterations", 20));
    cfg_.hc_eps               = static_cast<float>(sg.get_f64(pfx + "hyper_connection.epsilon", 1e-6));
    cfg_.rms_eps              = static_cast<float>(sg.get_f64(pfx + "attention.layer_norm_rms_epsilon", 1e-5));
    cfg_.max_ctx              = max_ctx;
    cfg_.kv_q8_0              = kv_q8_0;

    cfg_.head_count_kv      = read_num_arr<int>(sg, pfx + "attention.head_count_kv", cfg_.n_blocks, 0);
    cfg_.swiglu_clamp_exp   = read_num_arr<float>(sg, pfx + "swiglu_clamp_exp", cfg_.n_blocks, 0.0f);
    cfg_.swiglu_clamp_shexp = read_num_arr<float>(sg, pfx + "swiglu_clamp_shexp", cfg_.n_blocks, 0.0f);

    const std::string mtp_pfx = "blk." + std::to_string(cfg_.n_layers) + ".";
    has_mtp_ = load_mtp && cfg_.n_nextn >= 1 && (sg.find_tensor(mtp_pfx + "nextn.eh_proj.weight").info != nullptr);

    // Partition 45 backbone layers across GPUs proportionally to split_weights
    std::vector<float> sw = split_weights;
    if (sw.size() != gpu_ids.size()) sw.assign(gpu_ids.size(), 1.0f);
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
    kda_conv_q_snap_.resize(cfg_.n_layers, nullptr);
    kda_conv_k_snap_.resize(cfg_.n_layers, nullptr);
    kda_conv_v_snap_.resize(cfg_.n_layers, nullptr);
    kda_state_snap_.resize(cfg_.n_layers, nullptr);
    mla_kv_cache_.resize(cfg_.n_layers, nullptr);

    const int d_inner = cfg_.n_head * cfg_.kda_head_dim; // 8192
    const size_t mla_kv_bytes = kv_q8_0
        ? static_cast<size_t>(cfg_.max_ctx) * (cfg_.kv_lora_rank / 32) * sizeof(BlockQ8_0)
        : static_cast<size_t>(cfg_.max_ctx) * cfg_.kv_lora_rank * sizeof(uint16_t);

    // Parallel multi-GPU weight upload (1 thread per GPU -> saturates NVMe + all PCIe links!)
    std::vector<std::thread> load_threads;
    std::vector<std::exception_ptr> load_errs(runtimes_.size(), nullptr);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        load_threads.emplace_back([&, gi]() {
            try {
                const int dev = runtimes_[gi].device;
                cuda_ok(cudaSetDevice(dev), "parallel load setDevice");
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

                    L.attn_norm     = pool_.upload(sg, p + "attn_norm.weight",     dev, false);
                    L.ffn_norm      = pool_.upload(sg, p + "ffn_norm.weight",      dev, false);
                    L.hc_attn_fn    = pool_.upload(sg, p + "hc_attn_fn.weight",    dev, false);
                    L.hc_attn_base  = pool_.upload(sg, p + "hc_attn_base.weight",  dev, false);
                    L.hc_attn_scale = pool_.upload(sg, p + "hc_attn_scale.weight", dev, false);
                    L.hc_ffn_fn     = pool_.upload(sg, p + "hc_ffn_fn.weight",     dev, false);
                    L.hc_ffn_base   = pool_.upload(sg, p + "hc_ffn_base.weight",   dev, false);
                    L.hc_ffn_scale  = pool_.upload(sg, p + "hc_ffn_scale.weight",  dev, false);

                    if (L.is_kda) {
                        L.wq          = pool_.upload(sg, p + "attn_q.weight",        dev, false);
                        L.wk          = pool_.upload(sg, p + "attn_k.weight",        dev, false);
                        L.wv          = pool_.upload(sg, p + "attn_v.weight",        dev, false);
                        L.wo          = pool_.upload(sg, p + "attn_output.weight",   dev, false);
                        L.ssm_conv_q  = pool_.upload(sg, p + "ssm_conv1d_q.weight",  dev, false);
                        L.ssm_conv_k  = pool_.upload(sg, p + "ssm_conv1d_k.weight",  dev, false);
                        L.ssm_conv_v  = pool_.upload(sg, p + "ssm_conv1d_v.weight",  dev, false);
                        L.ssm_f_a     = pool_.upload(sg, p + "ssm_f_a.weight",       dev, false);
                        L.ssm_f_b     = pool_.upload(sg, p + "ssm_f_b.weight",       dev, false);
                        L.ssm_dt_bias = pool_.upload(sg, p + "ssm_dt.bias",          dev, false);
                        L.ssm_a       = pool_.upload(sg, p + "ssm_a",                dev, false);
                        L.ssm_beta    = pool_.upload(sg, p + "ssm_beta.weight",      dev, false);
                        L.ssm_g_a     = pool_.upload(sg, p + "ssm_g_a.weight",       dev, false);
                        L.ssm_g_b     = pool_.upload(sg, p + "ssm_g_b.weight",       dev, false);
                        L.ssm_norm    = pool_.upload(sg, p + "ssm_norm.weight",      dev, false);

                        const size_t conv_n = static_cast<size_t>(3) * d_inner;
                        const size_t ssm_n  = static_cast<size_t>(cfg_.n_head) * cfg_.kda_head_dim * cfg_.kda_head_dim;
                        kda_conv_q_[il] = dev_alloc<float>(dev, conv_n);
                        kda_conv_k_[il] = dev_alloc<float>(dev, conv_n);
                        kda_conv_v_[il] = dev_alloc<float>(dev, conv_n);
                        kda_state_[il]  = dev_alloc<float>(dev, ssm_n);
                        if (has_mtp_) {
                            kda_conv_q_snap_[il] = dev_alloc<float>(dev, conv_n);
                            kda_conv_k_snap_[il] = dev_alloc<float>(dev, conv_n);
                            kda_conv_v_snap_[il] = dev_alloc<float>(dev, conv_n);
                            kda_state_snap_[il]  = dev_alloc<float>(dev, ssm_n);
                        }
                    } else {
                        L.wq_a           = pool_.upload(sg, p + "attn_q_a.weight",      dev, false);
                        L.attn_q_a_norm  = pool_.upload(sg, p + "attn_q_a_norm.weight", dev, false);
                        L.wq_b           = pool_.upload(sg, p + "attn_q_b.weight",      dev, false);
                        L.wkv_a_mqa      = pool_.upload(sg, p + "attn_kv_a_mqa.weight", dev, false);
                        L.attn_kv_a_norm = pool_.upload(sg, p + "attn_kv_a_norm.weight",dev, false);
                        L.wk_b           = pool_.upload(sg, p + "attn_k_b.weight",      dev, false);
                        L.wv_b           = pool_.upload(sg, p + "attn_v_b.weight",      dev, false);
                        L.wo             = pool_.upload(sg, p + "attn_output.weight",   dev, false);

                        cuda_ok(cudaMalloc(&mla_kv_cache_[il], mla_kv_bytes), "malloc mla_kv_cache");
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
                if (gi + 1 == runtimes_.size() && has_mtp_) {
                    mtp_.device           = dev;
                    mtp_.eh_proj          = pool_.upload(sg, mtp_pfx + "nextn.eh_proj.weight",          dev, false);
                    mtp_.enorm            = pool_.upload(sg, mtp_pfx + "nextn.enorm.weight",            dev, false);
                    mtp_.hnorm            = pool_.upload(sg, mtp_pfx + "nextn.hnorm.weight",            dev, false);
                    mtp_.shared_head_norm = pool_.upload(sg, mtp_pfx + "nextn.shared_head_norm.weight", dev, false, false);
                    if (!mtp_.shared_head_norm.valid()) mtp_.shared_head_norm = output_norm_;

                    mtp_.attn_norm        = pool_.upload(sg, mtp_pfx + "attn_norm.weight",      dev, false);
                    mtp_.ffn_norm         = pool_.upload(sg, mtp_pfx + "ffn_norm.weight",       dev, false);
                    mtp_.wq_a             = pool_.upload(sg, mtp_pfx + "attn_q_a.weight",       dev, false);
                    mtp_.attn_q_a_norm    = pool_.upload(sg, mtp_pfx + "attn_q_a_norm.weight",  dev, false);
                    mtp_.wq_b             = pool_.upload(sg, mtp_pfx + "attn_q_b.weight",       dev, false);
                    mtp_.wkv_a_mqa        = pool_.upload(sg, mtp_pfx + "attn_kv_a_mqa.weight",  dev, false);
                    mtp_.attn_kv_a_norm   = pool_.upload(sg, mtp_pfx + "attn_kv_a_norm.weight", dev, false);
                    mtp_.wk_b             = pool_.upload(sg, mtp_pfx + "attn_k_b.weight",       dev, false);
                    mtp_.wv_b             = pool_.upload(sg, mtp_pfx + "attn_v_b.weight",       dev, false);
                    mtp_.wo               = pool_.upload(sg, mtp_pfx + "attn_output.weight",    dev, false);

                    mtp_.ffn_gate_inp     = pool_.upload(sg, mtp_pfx + "ffn_gate_inp.weight",   dev, false);
                    mtp_.ffn_exp_probs_b  = pool_.upload(sg, mtp_pfx + "exp_probs_b.bias",      dev, false);
                    mtp_.ffn_gate_exps    = pool_.upload(sg, mtp_pfx + "ffn_gate_exps.weight",  dev, false);
                    mtp_.ffn_up_exps      = pool_.upload(sg, mtp_pfx + "ffn_up_exps.weight",    dev, false);
                    mtp_.ffn_down_exps    = pool_.upload(sg, mtp_pfx + "ffn_down_exps.weight",  dev, false);
                    mtp_.ffn_gate_shexp   = pool_.upload(sg, mtp_pfx + "ffn_gate_shexp.weight", dev, false);
                    mtp_.ffn_up_shexp     = pool_.upload(sg, mtp_pfx + "ffn_up_shexp.weight",   dev, false);
                    mtp_.ffn_down_shexp   = pool_.upload(sg, mtp_pfx + "ffn_down_shexp.weight", dev, false);

                    cuda_ok(cudaMalloc(&mtp_kv_cache_, mla_kv_bytes), "malloc mtp_kv_cache");
                }
                cuda_ok(cudaDeviceSynchronize(), "parallel load sync");
            } catch (...) {
                load_errs[gi] = std::current_exception();
            }
        });
    }
    for (auto & th : load_threads) th.join();
    for (auto & ep : load_errs) if (ep) std::rethrow_exception(ep);

    pool_.release_staging();

    const int Tmax = max_batch_;
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        const int dev = rt.device;
        cuda_ok(cudaSetDevice(dev), "glm5 rt setDevice");
        cuda_ok(cudaStreamCreate(&rt.stream), "stream");
        cuda_ok(cudaEventCreateWithFlags(&rt.done_event, cudaEventDisableTiming), "event");
        if (cublasCreate(&rt.cublas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");

        rt.d_inpL     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
        rt.d_inpL_alt = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.n_embd);
        rt.d_mixes    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * 24);
        rt.d_post     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult);
        rt.d_comb     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.hc_mult * cfg_.hc_mult);
        rt.d_x_norm   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
        rt.d_sub_out  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);

        rt.d_kda_q    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_k    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_v    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_qc   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_kc   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_vc   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_fa   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * 128);
        rt.d_kda_g    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_beta = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head);
        rt.d_kda_out  = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);
        rt.d_kda_gate = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * d_inner);

        rt.d_mla_qr   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.q_lora_rank);
        rt.d_mla_q    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.qk_head_dim_mla);
        rt.d_mla_qabs = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.kv_lora_rank);
        rt.d_mla_kv   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.kv_lora_rank);
        rt.d_mla_olat = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.kv_lora_rank);
        rt.d_mla_o    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_head * cfg_.v_head_dim_mla);

        rt.d_ffn_g    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_ff_dense);
        rt.d_ffn_u    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_ff_dense);
        rt.d_router   = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert);
        rt.d_top_ids  = dev_alloc<int32_t>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert_used);
        rt.d_top_w    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_expert_used);

        rt.d_tokens   = dev_alloc<int32_t>(dev, Tmax);
        rt.d_pos      = dev_alloc<int32_t>(dev, Tmax);
        if (dev == last_dev) {
            rt.d_logits = dev_alloc<float>(dev, static_cast<size_t>(2) * cfg_.n_vocab);
            rt.d_argmax = dev_alloc<int32_t>(dev, 2);
            if (has_mtp_) {
                rt.d_h_nextn    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
                rt.d_mtp_e      = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
                rt.d_mtp_eh     = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * 2 * cfg_.n_embd);
                rt.d_mtp_cur    = dev_alloc<float>(dev, static_cast<size_t>(Tmax) * cfg_.n_embd);
                rt.d_mtp_pos    = dev_alloc<int32_t>(dev, Tmax);
                rt.d_mtp_argmax = dev_alloc<int32_t>(dev, 1);
            }
        }
        rt.scratch_bytes = 96ULL << 20;
        rt.d_scratch = dev_alloc<uint8_t>(dev, rt.scratch_bytes);
    }

    reset_state();

    const auto t1 = std::chrono::steady_clock::now();
    std::printf("Loaded GLM-5.3-Flash (%s, %d backbone layers + mtp=%s, ctx=%d, kv=%s) in %.2f s\n",
                arch.c_str(), cfg_.n_layers, has_mtp_ ? "on" : "off", cfg_.max_ctx, kv_q8_0 ? "Q8_0" : "F16",
                std::chrono::duration<double>(t1 - t0).count());
    for (const auto & rt : runtimes_) {
        std::printf("  GPU%d: layers [%2d..%2d) (%2d layers), %.2f GiB weights\n",
                    rt.device, rt.layer_start, rt.layer_end, rt.layer_end - rt.layer_start,
                    static_cast<double>(pool_.bytes_on_device(rt.device)) / (1024.0 * 1024.0 * 1024.0));
    }
}

void Glm5NextModel::reset_state() {
    const int d_inner = cfg_.n_head * cfg_.kda_head_dim;
    for (int il = 0; il < cfg_.n_layers; ++il) {
        if (!layers_[il].is_kda) continue;
        cuda_ok(cudaSetDevice(layers_[il].device), "reset_state setDevice");
        cuda_ok(cudaMemset(kda_conv_q_[il], 0, static_cast<size_t>(3) * d_inner * sizeof(float)), "zero cq");
        cuda_ok(cudaMemset(kda_conv_k_[il], 0, static_cast<size_t>(3) * d_inner * sizeof(float)), "zero ck");
        cuda_ok(cudaMemset(kda_conv_v_[il], 0, static_cast<size_t>(3) * d_inner * sizeof(float)), "zero cv");
        cuda_ok(cudaMemset(kda_state_[il],  0, static_cast<size_t>(cfg_.n_head) * cfg_.kda_head_dim * cfg_.kda_head_dim * sizeof(float)), "zero ssm");
    }
}

void Glm5NextModel::save_kda_checkpoint(KdaCheckpoint & out, int pos) const {
    const size_t conv_n = static_cast<size_t>(3) * cfg_.n_head * cfg_.kda_head_dim;
    const size_t ssm_n  = static_cast<size_t>(cfg_.n_head) * cfg_.kda_head_dim * cfg_.kda_head_dim;
    out.pos = pos;
    out.conv_q.resize(cfg_.n_layers);
    out.conv_k.resize(cfg_.n_layers);
    out.conv_v.resize(cfg_.n_layers);
    out.ssm.resize(cfg_.n_layers);

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        const auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "save_cp setDevice");
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
        cuda_ok(cudaSetDevice(rt.device), "save_cp sync setDevice");
        cuda_ok(cudaStreamSynchronize(rt.stream), "save_cp sync");
    }
}

void Glm5NextModel::restore_kda_checkpoint(const KdaCheckpoint & cp) {
    const size_t conv_n = static_cast<size_t>(3) * cfg_.n_head * cfg_.kda_head_dim;
    const size_t ssm_n  = static_cast<size_t>(cfg_.n_head) * cfg_.kda_head_dim * cfg_.kda_head_dim;
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "restore_cp setDevice");
        for (int il = rt.layer_start; il < rt.layer_end; ++il) {
            if (!layers_[il].is_kda) continue;
            cuda_ok(cudaMemcpyAsync(kda_conv_q_[il], cp.conv_q[il].data(), conv_n * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "rest cq");
            cuda_ok(cudaMemcpyAsync(kda_conv_k_[il], cp.conv_k[il].data(), conv_n * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "rest ck");
            cuda_ok(cudaMemcpyAsync(kda_conv_v_[il], cp.conv_v[il].data(), conv_n * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "rest cv");
            cuda_ok(cudaMemcpyAsync(kda_state_[il],  cp.ssm[il].data(),    ssm_n  * sizeof(float), cudaMemcpyHostToDevice, rt.stream), "rest ssm");
        }
    }
    for (const auto & rt : runtimes_) {
        cuda_ok(cudaSetDevice(rt.device), "restore_cp sync setDevice");
        cuda_ok(cudaStreamSynchronize(rt.stream), "restore_cp sync");
    }
}

void Glm5NextModel::rollback_kda_states() {
    const size_t conv_bytes = static_cast<size_t>(3) * cfg_.n_head * cfg_.kda_head_dim * sizeof(float);
    const size_t ssm_bytes  = static_cast<size_t>(cfg_.n_head) * cfg_.kda_head_dim * cfg_.kda_head_dim * sizeof(float);
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "rollback setDevice");
        for (int il = rt.layer_start; il < rt.layer_end; ++il) {
            if (!layers_[il].is_kda) continue;
            cuda_ok(cudaMemcpyAsync(kda_conv_q_[il], kda_conv_q_snap_[il], conv_bytes, cudaMemcpyDeviceToDevice, rt.stream), "rb cq");
            cuda_ok(cudaMemcpyAsync(kda_conv_k_[il], kda_conv_k_snap_[il], conv_bytes, cudaMemcpyDeviceToDevice, rt.stream), "rb ck");
            cuda_ok(cudaMemcpyAsync(kda_conv_v_[il], kda_conv_v_snap_[il], conv_bytes, cudaMemcpyDeviceToDevice, rt.stream), "rb cv");
            cuda_ok(cudaMemcpyAsync(kda_state_[il],  kda_state_snap_[il],  ssm_bytes,  cudaMemcpyDeviceToDevice, rt.stream), "rb ssm");
        }
    }
}

void Glm5NextModel::embed_tokens_to_last_dev(const int32_t * h_tokens, int T) {
    auto & first_rt = runtimes_.front();
    auto & last_rt  = runtimes_.back();
    cuda_ok(cudaSetDevice(first_rt.device), "mtp_emb setDevice");
    cuda_ok(cudaMemcpyAsync(first_rt.d_tokens, h_tokens, static_cast<size_t>(T) * sizeof(int32_t),
                            cudaMemcpyHostToDevice, first_rt.stream), "mtp_emb tok H2D");
    launch_embedding_lookup(tok_embd_, first_rt.d_tokens, first_rt.d_x_norm, cfg_.n_embd, 1, T, first_rt.stream);
    if (first_rt.device == last_rt.device) {
        cuda_ok(cudaMemcpyAsync(last_rt.d_mtp_e, first_rt.d_x_norm, static_cast<size_t>(T) * cfg_.n_embd * sizeof(float),
                                cudaMemcpyDeviceToDevice, last_rt.stream), "mtp_emb D2D");
    } else {
        cuda_ok(cudaEventRecord(first_rt.done_event, first_rt.stream), "mtp_emb ev");
        cuda_ok(cudaSetDevice(last_rt.device), "mtp_emb wait setDevice");
        cuda_ok(cudaStreamWaitEvent(last_rt.stream, first_rt.done_event, 0), "mtp_emb wait");
        cuda_ok(cudaMemcpyPeerAsync(last_rt.d_mtp_e, last_rt.device, first_rt.d_x_norm, first_rt.device,
                                    static_cast<size_t>(T) * cfg_.n_embd * sizeof(float), last_rt.stream), "mtp_emb peer");
    }
}

void Glm5NextModel::run_gpu_segment(size_t gi, int T, bool run_argmax, bool save_kda_snap, int kv_len_max) {
    auto & rt = runtimes_[gi];
    cuda_ok(cudaSetDevice(rt.device), "glm5 seg setDevice");

    const int   hc_dim    = cfg_.hc_mult * cfg_.n_embd;
    const int   d_inner   = cfg_.n_head * cfg_.kda_head_dim;
    const float kda_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.kda_head_dim));
    const float mla_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.qk_head_dim_mla));

    const bool prof = (std::getenv("ROOKERY_PROFILE") != nullptr) && !use_cuda_graphs_;
    double ms_mhc = 0, ms_kda = 0, ms_mla = 0, ms_ffn = 0, ms_head = 0;
    auto mark = [&](double & acc, auto fn) {
        if (!prof) { fn(); return; }
        cudaStreamSynchronize(rt.stream);
        const auto a = std::chrono::steady_clock::now();
        fn();
        cudaStreamSynchronize(rt.stream);
        const auto b = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(b - a).count();
    };

    if (gi == 0) {
        launch_embedding_lookup(tok_embd_, rt.d_tokens, rt.d_inpL, cfg_.n_embd, cfg_.hc_mult, T, rt.stream);
    }

    for (int il = rt.layer_start; il < rt.layer_end; ++il) {
        const auto & L = layers_[il];

        mark(ms_mhc, [&]() {
            launch_hc_flat_rmsnorm_mix(rt.d_inpL, L.hc_attn_fn, rt.d_mixes, hc_dim, 24, T, cfg_.rms_eps, rt.stream);
            launch_hc_sinkhorn_pre_rmsnorm(
                rt.d_inpL, rt.d_mixes,
                static_cast<const float *>(L.hc_attn_scale.ptr),
                static_cast<const float *>(L.hc_attn_base.ptr),
                static_cast<const float *>(L.attn_norm.ptr),
                rt.d_post, rt.d_comb, rt.d_x_norm,
                cfg_.n_embd, T, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps, rt.stream);
        });

        auto linear_preq = [&](const DeviceTensor & W, const float * d_x, float * d_y) {
            if (T < 16 && (W.dtype == DType::Q8_0 || is_kquant_dtype(W.dtype))) {
                launch_linear_prequant(W, rt.d_scratch, d_y, T, rt.stream);
            } else {
                launch_linear(rt.cublas, W, d_x, d_y, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            }
        };

        if (L.is_kda) {
            mark(ms_kda, [&]() {
                if (T < 16) launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T, rt.stream);
                linear_preq(L.wq,       rt.d_x_norm, rt.d_kda_q);
                linear_preq(L.wk,       rt.d_x_norm, rt.d_kda_k);
                linear_preq(L.wv,       rt.d_x_norm, rt.d_kda_v);
                linear_preq(L.ssm_f_a,  rt.d_x_norm, rt.d_kda_fa);
                linear_preq(L.ssm_beta, rt.d_x_norm, rt.d_kda_beta);
                linear_preq(L.ssm_g_a,  rt.d_x_norm, rt.d_mla_kv);

                float * cq_snap = (save_kda_snap && has_mtp_) ? kda_conv_q_snap_[il] : nullptr;
                float * ck_snap = (save_kda_snap && has_mtp_) ? kda_conv_k_snap_[il] : nullptr;
                float * cv_snap = (save_kda_snap && has_mtp_) ? kda_conv_v_snap_[il] : nullptr;
                float * st_snap = (save_kda_snap && has_mtp_) ? kda_state_snap_[il]  : nullptr;

                launch_ssm_conv1d_silu(rt.d_kda_q, static_cast<const float *>(L.ssm_conv_q.ptr), kda_conv_q_[il], rt.d_kda_qc, T, d_inner, rt.stream, cq_snap);
                launch_ssm_conv1d_silu(rt.d_kda_k, static_cast<const float *>(L.ssm_conv_k.ptr), kda_conv_k_[il], rt.d_kda_kc, T, d_inner, rt.stream, ck_snap);
                launch_ssm_conv1d_silu(rt.d_kda_v, static_cast<const float *>(L.ssm_conv_v.ptr), kda_conv_v_[il], rt.d_kda_vc, T, d_inner, rt.stream, cv_snap);

                launch_l2_norm_d128(rt.d_kda_qc, T, cfg_.n_head, 1e-6f, rt.stream);
                launch_l2_norm_d128(rt.d_kda_kc, T, cfg_.n_head, 1e-6f, rt.stream);

                launch_linear(rt.cublas, L.ssm_f_b, rt.d_kda_fa, rt.d_kda_g, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                launch_kda_gate_and_beta(
                    rt.d_kda_g, static_cast<const float *>(L.ssm_dt_bias.ptr),
                    static_cast<const float *>(L.ssm_a.ptr), rt.d_kda_beta,
                    cfg_.kda_gate_lower_bound, T, cfg_.n_head, cfg_.kda_head_dim, rt.stream);

                launch_gated_delta_net_d128(
                    rt.d_kda_qc, rt.d_kda_kc, rt.d_kda_vc, rt.d_kda_g, rt.d_kda_beta,
                    kda_state_[il], rt.d_kda_out, T, cfg_.n_head, kda_scale, true, rt.stream, st_snap);

                launch_linear(rt.cublas, L.ssm_g_b, rt.d_mla_kv, rt.d_kda_gate, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                launch_rms_norm_sigmoid_gate_d128(rt.d_kda_out, static_cast<const float *>(L.ssm_norm.ptr),
                                                  rt.d_kda_gate, T, cfg_.n_head, cfg_.rms_eps, rt.stream);
                launch_linear(rt.cublas, L.wo, rt.d_kda_out, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            });
        } else {
            mark(ms_mla, [&]() {
                if (T < 16) launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T, rt.stream);
                linear_preq(L.wq_a,      rt.d_x_norm, rt.d_mla_qr);
                linear_preq(L.wkv_a_mqa, rt.d_x_norm, rt.d_mla_kv);
                launch_rmsnorm(rt.d_mla_qr, static_cast<const float *>(L.attn_q_a_norm.ptr), rt.d_mla_qr, cfg_.q_lora_rank, T, cfg_.rms_eps, rt.stream);
                launch_linear(rt.cublas, L.wq_b, rt.d_mla_qr, rt.d_mla_q, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
                launch_rmsnorm(rt.d_mla_kv, static_cast<const float *>(L.attn_kv_a_norm.ptr), rt.d_mla_kv, cfg_.kv_lora_rank, T, cfg_.rms_eps, rt.stream);

                launch_mla_kv_write(rt.d_mla_kv, mla_kv_cache_[il], cfg_.kv_q8_0, rt.d_pos, T, cfg_.kv_lora_rank, rt.stream);

                launch_batched_head_proj_q8_0(L.wk_b, rt.d_mla_q, rt.d_mla_qabs, T, rt.stream, rt.cublas, rt.d_scratch, rt.scratch_bytes);
                launch_mla_attn_d512(rt.d_mla_qabs, mla_kv_cache_[il], cfg_.kv_q8_0,
                                     rt.d_pos, rt.d_mla_olat, rt.d_scratch, T, cfg_.n_head, mla_scale, rt.stream,
                                     rt.cublas, rt.scratch_bytes, kv_len_max);
                launch_batched_head_proj_q8_0(L.wv_b, rt.d_mla_olat, rt.d_mla_o, T, rt.stream, rt.cublas, rt.d_scratch, rt.scratch_bytes);
                launch_linear(rt.cublas, L.wo, rt.d_mla_o, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
            });
        }

        mark(ms_mhc, [&]() {
            launch_hc_post(rt.d_sub_out, rt.d_inpL, rt.d_post, rt.d_comb, rt.d_inpL_alt, cfg_.n_embd, T, rt.stream);
            launch_hc_flat_rmsnorm_mix(rt.d_inpL_alt, L.hc_ffn_fn, rt.d_mixes, hc_dim, 24, T, cfg_.rms_eps, rt.stream);
            launch_hc_sinkhorn_pre_rmsnorm(
                rt.d_inpL_alt, rt.d_mixes,
                static_cast<const float *>(L.hc_ffn_scale.ptr),
                static_cast<const float *>(L.hc_ffn_base.ptr),
                static_cast<const float *>(L.ffn_norm.ptr),
                rt.d_post, rt.d_comb, rt.d_x_norm,
                cfg_.n_embd, T, cfg_.hc_sinkhorn_iters, cfg_.hc_eps, cfg_.rms_eps, rt.stream);
        });

        mark(ms_ffn, [&]() {
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
        });

        mark(ms_mhc, [&]() {
            launch_hc_post(rt.d_sub_out, rt.d_inpL_alt, rt.d_post, rt.d_comb, rt.d_inpL, cfg_.n_embd, T, rt.stream);
        });

        if (hook_ != nullptr && hook_->hooks_layer(il)) {
            hook_->on_post_layer(il, rt.device, rt.d_inpL, T, cfg_.n_embd, cfg_.hc_mult, rt.stream);
        }
    }

    // Final head on last GPU: unweighted mean across hc=4 streams -> RMSNorm -> output.weight
    if (gi + 1 == runtimes_.size()) {
        mark(ms_head, [&]() {
            if (has_mtp_) {
                // Save post-output_norm h_nextn for ALL T tokens for blk.45 MTP!
                launch_hc_mean_rmsnorm(rt.d_inpL, static_cast<const float *>(output_norm_.ptr),
                                       rt.d_h_nextn, cfg_.n_embd, T, cfg_.rms_eps, rt.stream);
                if (T == 2 && run_argmax) {
                    launch_linear(rt.cublas, output_, rt.d_h_nextn, rt.d_logits, 2, rt.d_scratch, rt.scratch_bytes, rt.stream);
                    launch_argmax_f32_batch(rt.d_logits, rt.d_argmax, cfg_.n_vocab, 2, rt.stream);
                } else {
                    const float * last_h = rt.d_h_nextn + static_cast<size_t>(T - 1) * cfg_.n_embd;
                    launch_linear(rt.cublas, output_, last_h, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, rt.stream);
                    if (run_argmax) {
                        launch_argmax_f32_batch(rt.d_logits, rt.d_argmax, cfg_.n_vocab, 1, rt.stream);
                    }
                }
            } else {
                const float * head_in = rt.d_inpL + static_cast<size_t>(T - 1) * hc_dim;
                launch_hc_mean_rmsnorm(head_in, static_cast<const float *>(output_norm_.ptr),
                                       rt.d_x_norm, cfg_.n_embd, 1, cfg_.rms_eps, rt.stream);
                launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, rt.stream);
                if (run_argmax) {
                    launch_argmax_f32_batch(rt.d_logits, rt.d_argmax, cfg_.n_vocab, 1, rt.stream);
                }
            }
        });
    }
    if (prof) {
        std::printf("  [PROF GPU%d L%d..%d] mhc=%.2f kda=%.2f mla=%.2f ffn=%.2f head=%.2f tot=%.2f ms\n",
                    rt.device, rt.layer_start, rt.layer_end, ms_mhc, ms_kda, ms_mla, ms_ffn, ms_head,
                    ms_mhc + ms_kda + ms_mla + ms_ffn + ms_head);
    }
}

void Glm5NextModel::run_mtp_layer(int T, bool run_argmax, int kv_len_max) {
    auto & rt = runtimes_.back();
    cuda_ok(cudaSetDevice(rt.device), "mtp setDevice");

    const float mla_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.qk_head_dim_mla));
    const int mtp_il = cfg_.n_layers;

    // 1. Concat [RMSNorm(e, enorm) ; RMSNorm(h, hnorm)] -> eh_proj -> cur[T, 4096]
    launch_mtp_eh_concat(
        rt.d_mtp_e, rt.d_h_nextn,
        static_cast<const float *>(mtp_.enorm.ptr),
        static_cast<const float *>(mtp_.hnorm.ptr),
        rt.d_mtp_eh, cfg_.n_embd, 1, T, cfg_.rms_eps, rt.stream);
    launch_linear(rt.cublas, mtp_.eh_proj, rt.d_mtp_eh, rt.d_mtp_cur, T, rt.d_scratch, rt.scratch_bytes, rt.stream);

    // 2. Plain single-stream MLA (no mHC!)
    launch_rmsnorm(rt.d_mtp_cur, static_cast<const float *>(mtp_.attn_norm.ptr), rt.d_x_norm, cfg_.n_embd, T, cfg_.rms_eps, rt.stream);
    if (T < 16 && (mtp_.wq_a.dtype == DType::Q8_0 || is_kquant_dtype(mtp_.wq_a.dtype)) &&
        (mtp_.wkv_a_mqa.dtype == DType::Q8_0 || is_kquant_dtype(mtp_.wkv_a_mqa.dtype))) {
        launch_quantize_q8_1(rt.d_x_norm, rt.d_scratch, cfg_.n_embd, T, rt.stream);
        launch_linear_prequant(mtp_.wq_a,      rt.d_scratch, rt.d_mla_qr, T, rt.stream);
        launch_linear_prequant(mtp_.wkv_a_mqa, rt.d_scratch, rt.d_mla_kv, T, rt.stream);
    } else {
        launch_linear(rt.cublas, mtp_.wq_a,      rt.d_x_norm, rt.d_mla_qr, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
        launch_linear(rt.cublas, mtp_.wkv_a_mqa, rt.d_x_norm, rt.d_mla_kv, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
    }
    launch_rmsnorm(rt.d_mla_qr, static_cast<const float *>(mtp_.attn_q_a_norm.ptr), rt.d_mla_qr, cfg_.q_lora_rank, T, cfg_.rms_eps, rt.stream);
    launch_linear(rt.cublas, mtp_.wq_b, rt.d_mla_qr, rt.d_mla_q, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
    launch_rmsnorm(rt.d_mla_kv, static_cast<const float *>(mtp_.attn_kv_a_norm.ptr), rt.d_mla_kv, cfg_.kv_lora_rank, T, cfg_.rms_eps, rt.stream);

    launch_mla_kv_write(rt.d_mla_kv, mtp_kv_cache_, cfg_.kv_q8_0, rt.d_mtp_pos, T, cfg_.kv_lora_rank, rt.stream);
    launch_batched_head_proj_q8_0(mtp_.wk_b, rt.d_mla_q, rt.d_mla_qabs, T, rt.stream, rt.cublas, rt.d_scratch, rt.scratch_bytes);
    launch_mla_attn_d512(rt.d_mla_qabs, mtp_kv_cache_, cfg_.kv_q8_0,
                         rt.d_mtp_pos, rt.d_mla_olat, rt.d_scratch, T, cfg_.n_head, mla_scale, rt.stream,
                         rt.cublas, rt.scratch_bytes, kv_len_max);
    launch_batched_head_proj_q8_0(mtp_.wv_b, rt.d_mla_olat, rt.d_mla_o, T, rt.stream, rt.cublas, rt.d_scratch, rt.scratch_bytes);
    launch_linear(rt.cublas, mtp_.wo, rt.d_mla_o, rt.d_sub_out, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
    launch_add_inplace(rt.d_mtp_cur, rt.d_sub_out, static_cast<int64_t>(T) * cfg_.n_embd, rt.stream);

    // 3. Plain single-stream MoE (no mHC!)
    launch_rmsnorm(rt.d_mtp_cur, static_cast<const float *>(mtp_.ffn_norm.ptr), rt.d_x_norm, cfg_.n_embd, T, cfg_.rms_eps, rt.stream);
    launch_linear(rt.cublas, mtp_.ffn_gate_inp, rt.d_x_norm, rt.d_router, T, rt.d_scratch, rt.scratch_bytes, rt.stream);
    launch_moe_route_sigmoid(
        rt.d_router, static_cast<const float *>(mtp_.ffn_exp_probs_b.ptr),
        rt.d_top_ids, rt.d_top_w,
        T, cfg_.n_expert, cfg_.n_expert_used, cfg_.expert_weights_norm, cfg_.expert_weights_scale, rt.stream);
    const float c_exp = (mtp_il < static_cast<int>(cfg_.swiglu_clamp_exp.size())) ? cfg_.swiglu_clamp_exp[mtp_il] : 0.0f;
    const float c_sh  = (mtp_il < static_cast<int>(cfg_.swiglu_clamp_shexp.size())) ? cfg_.swiglu_clamp_shexp[mtp_il] : 0.0f;
    launch_moe_ffn_kquant_with_shared(
        rt.cublas,
        mtp_.ffn_gate_exps, mtp_.ffn_up_exps, mtp_.ffn_down_exps,
        mtp_.ffn_gate_shexp, mtp_.ffn_up_shexp, mtp_.ffn_down_shexp,
        rt.d_x_norm, rt.d_top_ids, rt.d_top_w, rt.d_sub_out,
        rt.d_scratch, rt.scratch_bytes, T, cfg_.n_expert_used, c_exp, c_sh, rt.stream);
    launch_add_inplace(rt.d_mtp_cur, rt.d_sub_out, static_cast<int64_t>(T) * cfg_.n_embd, rt.stream);

    // 4. Shared head norm + output.weight -> draft argmax (on last row T-1)
    if (run_argmax) {
        const float * last_cur = rt.d_mtp_cur + static_cast<size_t>(T - 1) * cfg_.n_embd;
        launch_rmsnorm(last_cur, static_cast<const float *>(mtp_.shared_head_norm.ptr),
                       rt.d_x_norm, cfg_.n_embd, 1, cfg_.rms_eps, rt.stream);
        launch_linear(rt.cublas, output_, rt.d_x_norm, rt.d_logits, 1, rt.d_scratch, rt.scratch_bytes, rt.stream);
        launch_argmax_f32_batch(rt.d_logits, rt.d_mtp_argmax, cfg_.n_vocab, 1, rt.stream);
    }
}

void Glm5NextModel::capture_decode_graphs() {
    auto capture_one = [](int dev, cudaStream_t stream, auto body) -> cudaGraphExec_t {
        cuda_ok(cudaSetDevice(dev), "glm5 cap setDevice");
        cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "glm5 begin cap");
        body();
        cudaGraph_t graph = nullptr;
        cuda_ok(cudaStreamEndCapture(stream, &graph), "glm5 end cap");
        cudaGraphExec_t exec = nullptr;
        cuda_ok(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "glm5 inst graph");
        cudaGraphDestroy(graph);
        return exec;
    };

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        rt.decode_graph_t1 = capture_one(rt.device, rt.stream, [&]() { run_gpu_segment(gi, 1, true, false); });
        if (has_mtp_) {
            rt.decode_graph_t2 = capture_one(rt.device, rt.stream, [&]() { run_gpu_segment(gi, 2, true, true); });
        }
    }
    if (has_mtp_) {
        auto & last_rt = runtimes_.back();
        mtp_graph_t1_ = capture_one(last_rt.device, last_rt.stream, [&]() { run_mtp_layer(1, true); });
        mtp_graph_t2_ = capture_one(last_rt.device, last_rt.stream, [&]() { run_mtp_layer(2, true); });
    }
    graphs_ready_ = true;
}

void Glm5NextModel::warmup_decode_graphs() {
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();
}

void Glm5NextModel::forward(const std::vector<int32_t> & tokens,
                            const std::vector<int32_t> & positions,
                            std::vector<float> & out_logits) {
    const int T = static_cast<int>(tokens.size());
    if (T <= 0 || T > max_batch_ || positions.size() != tokens.size()) {
        throw std::runtime_error("Glm5NextModel::forward invalid batch size");
    }
    const int hc_dim = cfg_.hc_mult * cfg_.n_embd;
    int kv_len_max = 0;
    for (int32_t p : positions) kv_len_max = std::max(kv_len_max, p + 1);

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "glm5 fwd setDevice");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, positions.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "pos");
        if (gi == 0) {
            cuda_ok(cudaMemcpyAsync(rt.d_tokens, tokens.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok");
        }
    }
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        if (gi > 0) {
            auto & src_rt = runtimes_[gi - 1];
            cuda_ok(cudaSetDevice(src_rt.device), "rec");
            // Sync previous GPU before non-P2P host-staged peer copy during prefill (avoids 2.4x multi-GPU driver stall!)
            cuda_ok(cudaStreamSynchronize(src_rt.stream), "pre-peer sync");
            cuda_ok(cudaSetDevice(rt.device), "wait");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_inpL, rt.device, src_rt.d_inpL, src_rt.device,
                                        static_cast<size_t>(T) * hc_dim * sizeof(float), rt.stream), "peer");
        }
        run_gpu_segment(gi, T, false, false, kv_len_max);
    }

    auto & last_rt = runtimes_.back();
    cuda_ok(cudaSetDevice(last_rt.device), "logits setDevice");
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaMemcpyAsync(out_logits.data(), last_rt.d_logits, cfg_.n_vocab * sizeof(float),
                            cudaMemcpyDeviceToHost, last_rt.stream), "logits D2H");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "sync");
}

int32_t Glm5NextModel::mtp_prefill_chunk(const std::vector<int32_t> & next_tokens,
                                         const std::vector<int32_t> & positions) {
    if (!has_mtp_) throw std::runtime_error("mtp_prefill_chunk called without MTP");
    const int T = static_cast<int>(next_tokens.size());
    if (T <= 0 || T > max_batch_ || positions.size() != next_tokens.size()) {
        throw std::runtime_error("mtp_prefill_chunk invalid batch size");
    }
    int kv_len_max = 0;
    for (int32_t p : positions) kv_len_max = std::max(kv_len_max, p + 1);
    auto & last_rt = runtimes_.back();
    embed_tokens_to_last_dev(next_tokens.data(), T);
    cuda_ok(cudaSetDevice(last_rt.device), "mtp_pf setDevice");
    cuda_ok(cudaMemcpyAsync(last_rt.d_mtp_pos, positions.data(), T * sizeof(int32_t),
                            cudaMemcpyHostToDevice, last_rt.stream), "mtp_pf pos");
    run_mtp_layer(T, true, kv_len_max);

    int32_t draft = 0;
    cuda_ok(cudaMemcpyAsync(&draft, last_rt.d_mtp_argmax, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, last_rt.stream), "mtp_pf argmax");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "mtp_pf sync");
    return draft;
}

void Glm5NextModel::mtp_prefill_chunk_logits(const std::vector<int32_t> & next_tokens,
                                             const std::vector<int32_t> & positions,
                                             std::vector<float> & out_logits) {
    if (!has_mtp_) throw std::runtime_error("mtp_prefill_chunk_logits called without MTP");
    const int T = static_cast<int>(next_tokens.size());
    if (T <= 0 || T > max_batch_ || positions.size() != next_tokens.size()) {
        throw std::runtime_error("mtp_prefill_chunk_logits invalid batch size");
    }
    int kv_len_max = 0;
    for (int32_t p : positions) kv_len_max = std::max(kv_len_max, p + 1);
    auto & last_rt = runtimes_.back();
    embed_tokens_to_last_dev(next_tokens.data(), T);
    cuda_ok(cudaSetDevice(last_rt.device), "mtp_pfl setDevice");
    cuda_ok(cudaMemcpyAsync(last_rt.d_mtp_pos, positions.data(), T * sizeof(int32_t),
                            cudaMemcpyHostToDevice, last_rt.stream), "mtp_pfl pos");
    run_mtp_layer(T, true, kv_len_max);

    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaMemcpyAsync(out_logits.data(), last_rt.d_logits, cfg_.n_vocab * sizeof(float),
                            cudaMemcpyDeviceToHost, last_rt.stream), "mtp_pfl logits");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "mtp_pfl sync");
}

void Glm5NextModel::decode_mtp_sample_step(int32_t & cur_tok,
                                           int32_t & draft_tok,
                                           Sampler::SparseDist & draft_q,
                                           int32_t & cur_pos,
                                           Sampler & sampler,
                                           std::vector<int32_t> & out_tokens,
                                           MtpStats & stats) {
    if (!has_mtp_) throw std::runtime_error("decode_mtp_sample_step called without MTP");
    const int hc_dim = cfg_.hc_mult * cfg_.n_embd;
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();

    const int32_t toks[2] = {cur_tok, draft_tok};
    const int32_t poss[2] = {cur_pos, cur_pos + 1};

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "mtp_sdec setDevice");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, poss, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "pos2");
        if (gi == 0) cuda_ok(cudaMemcpyAsync(rt.d_tokens, toks, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok2");
    }
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        if (gi > 0) {
            auto & src_rt = runtimes_[gi - 1];
            cuda_ok(cudaSetDevice(src_rt.device), "srec2");
            cuda_ok(cudaEventRecord(src_rt.done_event, src_rt.stream), "sev2");
            cuda_ok(cudaSetDevice(rt.device), "swait2");
            cuda_ok(cudaStreamWaitEvent(rt.stream, src_rt.done_event, 0), "swev2");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_inpL, rt.device, src_rt.d_inpL, src_rt.device,
                                        static_cast<size_t>(2) * hc_dim * sizeof(float), rt.stream), "speer2");
        }
        cuda_ok(cudaSetDevice(rt.device), "slaunch2");
        if (use_cuda_graphs_ && rt.decode_graph_t2) {
            cuda_ok(cudaGraphLaunch(rt.decode_graph_t2, rt.stream), "sgraph t2");
        } else {
            run_gpu_segment(gi, 2, true, true);
        }
    }

    auto & last_rt = runtimes_.back();
    const int V = cfg_.n_vocab;
    std::vector<float> logits2(static_cast<size_t>(2) * V);
    cuda_ok(cudaSetDevice(last_rt.device), "slogits2 setDevice");
    cuda_ok(cudaMemcpyAsync(logits2.data(), last_rt.d_logits, static_cast<size_t>(2) * V * sizeof(float),
                            cudaMemcpyDeviceToHost, last_rt.stream), "slogits2 D2H");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "ssync2");

    const auto target_p0 = sampler.filtered_distribution(logits2.data(), V);
    int32_t corrected = 0;
    const bool accepted = sampler.verify_speculative(draft_tok, target_p0, draft_q, corrected);

    stats.steps++;
    if (accepted) {
        stats.accepted++;
        const auto target_p1 = sampler.filtered_distribution(logits2.data() + V, V);
        const int32_t bonus_tok = sampler.sample_dist(target_p1);
        out_tokens = {draft_tok, bonus_tok};
        const int32_t mtp_toks[2] = {draft_tok, bonus_tok};
        embed_tokens_to_last_dev(mtp_toks, 2);
        cuda_ok(cudaSetDevice(last_rt.device), "smtp_acc setDevice");
        cuda_ok(cudaMemcpyAsync(last_rt.d_mtp_pos, poss, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, last_rt.stream), "smtp_acc pos");
        if (use_cuda_graphs_ && mtp_graph_t2_) {
            cuda_ok(cudaGraphLaunch(mtp_graph_t2_, last_rt.stream), "smtp graph t2");
        } else {
            run_mtp_layer(2, true);
        }
        cur_tok = bonus_tok;
        cur_pos += 2;
    } else {
        rollback_kda_states();
        out_tokens = {corrected};
        embed_tokens_to_last_dev(&corrected, 1);
        cuda_ok(cudaSetDevice(last_rt.device), "smtp_rej setDevice");
        cuda_ok(cudaMemcpyAsync(last_rt.d_mtp_pos, &cur_pos, sizeof(int32_t), cudaMemcpyHostToDevice, last_rt.stream), "smtp_rej pos");
        if (use_cuda_graphs_ && mtp_graph_t1_) {
            cuda_ok(cudaGraphLaunch(mtp_graph_t1_, last_rt.stream), "smtp graph t1");
        } else {
            run_mtp_layer(1, true);
        }
        cur_tok = corrected;
        cur_pos += 1;
    }

    std::vector<float> mtp_logits(V);
    cuda_ok(cudaSetDevice(last_rt.device), "smtp_d2h setDevice");
    cuda_ok(cudaMemcpyAsync(mtp_logits.data(), last_rt.d_logits, static_cast<size_t>(V) * sizeof(float),
                            cudaMemcpyDeviceToHost, last_rt.stream), "smtp logits D2H");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "smtp sync");
    draft_q   = sampler.filtered_distribution(mtp_logits.data(), V);
    draft_tok = sampler.sample_dist(draft_q);
}

void Glm5NextModel::decode_mtp_greedy_step(int32_t & cur_tok,
                                           int32_t & draft_tok,
                                           int32_t & cur_pos,
                                           std::vector<int32_t> & out_tokens,
                                           MtpStats & stats) {
    if (!has_mtp_) throw std::runtime_error("decode_mtp_greedy_step called without MTP");
    const int hc_dim = cfg_.hc_mult * cfg_.n_embd;
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();

    const int32_t toks[2] = {cur_tok, draft_tok};
    const int32_t poss[2] = {cur_pos, cur_pos + 1};

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "mtp_dec setDevice");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, poss, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "pos2");
        if (gi == 0) cuda_ok(cudaMemcpyAsync(rt.d_tokens, toks, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok2");
    }
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        if (gi > 0) {
            auto & src_rt = runtimes_[gi - 1];
            cuda_ok(cudaSetDevice(src_rt.device), "rec2");
            cuda_ok(cudaEventRecord(src_rt.done_event, src_rt.stream), "ev2");
            cuda_ok(cudaSetDevice(rt.device), "wait2");
            cuda_ok(cudaStreamWaitEvent(rt.stream, src_rt.done_event, 0), "wev2");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_inpL, rt.device, src_rt.d_inpL, src_rt.device,
                                        static_cast<size_t>(2) * hc_dim * sizeof(float), rt.stream), "peer2");
        }
        cuda_ok(cudaSetDevice(rt.device), "launch2");
        if (use_cuda_graphs_ && rt.decode_graph_t2) {
            cuda_ok(cudaGraphLaunch(rt.decode_graph_t2, rt.stream), "graph t2");
        } else {
            run_gpu_segment(gi, 2, true, true);
        }
    }

    auto & last_rt = runtimes_.back();
    int32_t preds[2] = {0, 0};
    cuda_ok(cudaSetDevice(last_rt.device), "argmax2 setDevice");
    cuda_ok(cudaMemcpyAsync(preds, last_rt.d_argmax, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, last_rt.stream), "argmax2 D2H");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "sync2");

    stats.steps++;
    const bool accepted = (preds[0] == draft_tok);
    if (accepted) {
        stats.accepted++;
        out_tokens = {draft_tok, preds[1]};
        const int32_t mtp_toks[2] = {draft_tok, preds[1]};
        embed_tokens_to_last_dev(mtp_toks, 2);
        cuda_ok(cudaSetDevice(last_rt.device), "mtp_acc setDevice");
        cuda_ok(cudaMemcpyAsync(last_rt.d_mtp_pos, poss, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, last_rt.stream), "mtp_acc pos");
        if (use_cuda_graphs_ && mtp_graph_t2_) {
            cuda_ok(cudaGraphLaunch(mtp_graph_t2_, last_rt.stream), "mtp graph t2");
        } else {
            run_mtp_layer(2, true);
        }
        cur_tok = preds[1];
        cur_pos += 2;
    } else {
        rollback_kda_states();
        out_tokens = {preds[0]};
        embed_tokens_to_last_dev(&preds[0], 1);
        cuda_ok(cudaSetDevice(last_rt.device), "mtp_rej setDevice");
        cuda_ok(cudaMemcpyAsync(last_rt.d_mtp_pos, &cur_pos, sizeof(int32_t), cudaMemcpyHostToDevice, last_rt.stream), "mtp_rej pos");
        if (use_cuda_graphs_ && mtp_graph_t1_) {
            cuda_ok(cudaGraphLaunch(mtp_graph_t1_, last_rt.stream), "mtp graph t1");
        } else {
            run_mtp_layer(1, true);
        }
        cur_tok = preds[0];
        cur_pos += 1;
    }

    cuda_ok(cudaSetDevice(last_rt.device), "mtp_d2h setDevice");
    cuda_ok(cudaMemcpyAsync(&draft_tok, last_rt.d_mtp_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, last_rt.stream), "mtp argmax D2H");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "mtp sync");
}

int32_t Glm5NextModel::forward_decode_greedy(int32_t token, int32_t pos) {
    const int hc_dim = cfg_.hc_mult * cfg_.n_embd;
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "dec setDevice");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "pos");
        if (gi == 0) cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok");
    }
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        if (gi > 0) {
            auto & src_rt = runtimes_[gi - 1];
            cuda_ok(cudaSetDevice(src_rt.device), "rec");
            cuda_ok(cudaEventRecord(src_rt.done_event, src_rt.stream), "ev");
            cuda_ok(cudaSetDevice(rt.device), "wait");
            cuda_ok(cudaStreamWaitEvent(rt.stream, src_rt.done_event, 0), "wev");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_inpL, rt.device, src_rt.d_inpL, src_rt.device,
                                        static_cast<size_t>(hc_dim) * sizeof(float), rt.stream), "peer");
        }
        cuda_ok(cudaSetDevice(rt.device), "launch");
        if (use_cuda_graphs_ && rt.decode_graph_t1) {
            cuda_ok(cudaGraphLaunch(rt.decode_graph_t1, rt.stream), "graph t1");
        } else {
            run_gpu_segment(gi, 1, true, false);
        }
    }

    auto & last_rt = runtimes_.back();
    int32_t next_tok = 0;
    cuda_ok(cudaSetDevice(last_rt.device), "argmax setDevice");
    cuda_ok(cudaMemcpyAsync(&next_tok, last_rt.d_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, last_rt.stream), "argmax D2H");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "sync");
    return next_tok;
}

void Glm5NextModel::forward_decode_logits(int32_t token, int32_t pos, std::vector<float> & out_logits) {
    const int hc_dim = cfg_.hc_mult * cfg_.n_embd;
    if (use_cuda_graphs_ && !graphs_ready_) capture_decode_graphs();

    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        cuda_ok(cudaSetDevice(rt.device), "dec_l setDevice");
        cuda_ok(cudaMemcpyAsync(rt.d_pos, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "pos");
        if (gi == 0) cuda_ok(cudaMemcpyAsync(rt.d_tokens, &token, sizeof(int32_t), cudaMemcpyHostToDevice, rt.stream), "tok");
    }
    for (size_t gi = 0; gi < runtimes_.size(); ++gi) {
        auto & rt = runtimes_[gi];
        if (gi > 0) {
            auto & src_rt = runtimes_[gi - 1];
            cuda_ok(cudaSetDevice(src_rt.device), "rec");
            cuda_ok(cudaEventRecord(src_rt.done_event, src_rt.stream), "ev");
            cuda_ok(cudaSetDevice(rt.device), "wait");
            cuda_ok(cudaStreamWaitEvent(rt.stream, src_rt.done_event, 0), "wev");
            cuda_ok(cudaMemcpyPeerAsync(rt.d_inpL, rt.device, src_rt.d_inpL, src_rt.device,
                                        static_cast<size_t>(hc_dim) * sizeof(float), rt.stream), "peer");
        }
        cuda_ok(cudaSetDevice(rt.device), "launch");
        if (use_cuda_graphs_ && rt.decode_graph_t1) {
            cuda_ok(cudaGraphLaunch(rt.decode_graph_t1, rt.stream), "graph t1");
        } else {
            run_gpu_segment(gi, 1, false, false);
        }
    }

    auto & last_rt = runtimes_.back();
    out_logits.resize(cfg_.n_vocab);
    cuda_ok(cudaSetDevice(last_rt.device), "logits setDevice");
    cuda_ok(cudaMemcpyAsync(out_logits.data(), last_rt.d_logits, cfg_.n_vocab * sizeof(float),
                            cudaMemcpyDeviceToHost, last_rt.stream), "logits D2H");
    cuda_ok(cudaStreamSynchronize(last_rt.stream), "sync");
}

} // namespace rookery
