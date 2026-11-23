#include "gguf/sharded.h"
#include "models/glm5_next.h"
#include "tokenizer/bpe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(std::stoi(item));
    return out;
}

static std::vector<float> parse_floats(const std::string & s) {
    std::vector<float> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(std::stof(item));
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::printf("Usage: %s <glm5-00001-of-00004.gguf> [--gpus 0,1,2,3] [--split 11,12,12,10] [--prompt \"...\"] [--gen 64] [--ctx 8192] [--no-graph] [--no-mtp]\n", argv[0]);
        return 1;
    }
    std::string path = argv[1];
    std::vector<int>   gpus  = {0, 1, 2, 3, 4};
    std::vector<float> split = {11, 1, 11, 11, 11};
    std::string prompt_str   = "The capital of France is";
    int  n_gen     = 64;
    int  max_ctx   = 8192;
    int  pp_test   = 0;
    bool use_graph = true;
    bool use_mtp   = true;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gpus" && i + 1 < argc)        gpus = parse_ints(argv[++i]);
        else if (a == "--split" && i + 1 < argc)  split = parse_floats(argv[++i]);
        else if (a == "--prompt" && i + 1 < argc) prompt_str = argv[++i];
        else if (a == "--gen" && i + 1 < argc)    n_gen = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc)    max_ctx = std::atoi(argv[++i]);
        else if (a == "--pp-test" && i + 1 < argc) pp_test = std::atoi(argv[++i]);
        else if (a == "--no-graph")               use_graph = false;
        else if (a == "--no-mtp")                 use_mtp = false;
        else if (a == "--mtp")                    use_mtp = true;
    }

    auto topo = rookery::discover_topology();
    rookery::print_topology(topo);

    rookery::BpeTokenizer tok;
    {
        auto sg = rookery::ShardedGguf::open(path);
        tok.load(sg.meta());
    }
    auto prompt = tok.encode(prompt_str, false, false);
    std::printf("Prompt '%s' -> %zu tokens:", prompt_str.c_str(), prompt.size());
    for (int32_t t : prompt) std::printf(" %d", t);
    std::printf("\n");

    rookery::Glm5NextModel model;
    model.set_use_cuda_graphs(use_graph);
    model.load(path, gpus, split, max_ctx, true, use_mtp);

    if (pp_test > 0) {
        const std::string para =
            "In modern GPU inference engines, batched prompt prefill groups routed MoE tokens by expert so each "
            "quantized weight matrix is read from HBM once per batch instead of once per token. Combined with "
            "Tensor Core batched projections for multi-head latent attention and host-RAM state checkpoints for "
            "hybrid linear attention layers, multi-turn conversations achieve sub-second time to first token. ";
        auto para_ids = tok.encode(para, false, false);
        auto tail_ids = tok.encode("The capital of France is", false, false);
        std::vector<int32_t> long_p;
        while (static_cast<int>(long_p.size() + tail_ids.size()) < pp_test) {
            for (int32_t id : para_ids) {
                if (static_cast<int>(long_p.size() + tail_ids.size()) >= pp_test) break;
                long_p.push_back(id);
            }
        }
        long_p.insert(long_p.end(), tail_ids.begin(), tail_ids.end());
        const int N = static_cast<int>(long_p.size());
        std::vector<int32_t> lpos(N);
        std::iota(lpos.begin(), lpos.end(), 0);

        // 1. Batched grouped-MoE + Tensor Core MLA prefill (256-token chunks, UNHOOKED for true wallclock!)
        std::vector<float> logits_batched;
        model.reset_state();
        const auto tb0 = std::chrono::steady_clock::now();
        for (int off = 0; off < N; off += 256) {
            int len = std::min(256, N - off);
            model.forward({long_p.begin() + off, long_p.begin() + off + len},
                          {lpos.begin() + off, lpos.begin() + off + len}, logits_batched);
        }
        const auto tb1 = std::chrono::steady_clock::now();
        const double b_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();

        // 1b. Second identical batched run (determinism check!)
        std::vector<float> logits_batched2;
        model.reset_state();
        for (int off = 0; off < N; off += 256) {
            int len = std::min(256, N - off);
            model.forward({long_p.begin() + off, long_p.begin() + off + len},
                          {lpos.begin() + off, lpos.begin() + off + len}, logits_batched2);
        }

        // 2. Reference ungrouped prefill (T=4 chunks -> uses decode kernels, no grouped MoE or batched MLA)
        std::vector<float> logits_ref;
        model.reset_state();
        for (int off = 0; off < N; off += 4) {
            int len = std::min(4, N - off);
            model.forward({long_p.begin() + off, long_p.begin() + off + len},
                          {lpos.begin() + off, lpos.begin() + off + len}, logits_ref);
        }

        // 3. KDA checkpoint save/restore parity test (split at N/2)
        const int half = (N / 2);
        rookery::Glm5NextModel::KdaCheckpoint cp;
        std::vector<float> logits_cp;
        model.reset_state();
        for (int off = 0; off < half; off += 256) {
            int len = std::min(256, half - off);
            model.forward({long_p.begin() + off, long_p.begin() + off + len},
                          {lpos.begin() + off, lpos.begin() + off + len}, logits_cp);
        }
        const auto tcp0 = std::chrono::steady_clock::now();
        model.save_kda_checkpoint(cp, half);
        model.reset_state();
        model.restore_kda_checkpoint(cp);
        const auto tcp1 = std::chrono::steady_clock::now();
        for (int off = half; off < N; off += 256) {
            int len = std::min(256, N - off);
            model.forward({long_p.begin() + off, long_p.begin() + off + len},
                          {lpos.begin() + off, lpos.begin() + off + len}, logits_cp);
        }

        auto top3 = [&](const std::vector<float> & lg, const char * tag) {
            std::vector<int> ix(lg.size());
            std::iota(ix.begin(), ix.end(), 0);
            std::partial_sort(ix.begin(), ix.begin() + 3, ix.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            std::printf("  %-10s:", tag);
            for (int k = 0; k < 3; ++k) std::printf(" [%d '%s']=%.4f", ix[k], tok.decode_token(ix[k]).c_str(), lg[ix[k]]);
            std::printf("\n");
        };
        float max_diff_ref = 0.0f, max_diff_cp = 0.0f, max_diff_det = 0.0f;
        for (size_t i = 0; i < logits_batched.size(); ++i) {
            max_diff_det = std::max(max_diff_det, std::fabs(logits_batched[i] - logits_batched2[i]));
            max_diff_ref = std::max(max_diff_ref, std::fabs(logits_batched[i] - logits_ref[i]));
            max_diff_cp  = std::max(max_diff_cp,  std::fabs(logits_batched[i] - logits_cp[i]));
        }
        std::printf("\n=== PP-TEST (%d tokens) ===\n", N);
        std::printf("Batched prefill: %.2f ms (%.1f tok/s)\n", b_ms, N * 1000.0 / b_ms);
        std::printf("Checkpoint save+restore: %.2f ms\n", std::chrono::duration<double, std::milli>(tcp1 - tcp0).count());
        std::printf("Max |b1 - b2| = %.6f | Max |batched - ref(T=4)| = %.5f | Max |batched - ckpt| = %.6f\n",
                    max_diff_det, max_diff_ref, max_diff_cp);
        top3(logits_batched, "batched");
        top3(logits_ref,     "ref(T=4)");
        top3(logits_cp,      "ckpt");
        return 0;
    }

    std::vector<int32_t> pos(prompt.size());
    std::iota(pos.begin(), pos.end(), 0);
    std::vector<float> logits;

    const auto t_pp0 = std::chrono::steady_clock::now();
    model.forward(prompt, pos, logits);
    const auto t_pp1 = std::chrono::steady_clock::now();
    const double pp_ms = std::chrono::duration<double, std::milli>(t_pp1 - t_pp0).count();

    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("Prefill top-5:");
    for (int k = 0; k < 5; ++k) std::printf("  [%d '%s']=%.4f", idx[k], tok.decode_token(idx[k]).c_str(), logits[idx[k]]);
    std::printf("\nPrefill (%zu tok): %.2f ms (%.1f tok/s)\n", prompt.size(), pp_ms, prompt.size() * 1000.0 / pp_ms);

    int32_t cur_tok = idx[0];
    int32_t cur_pos = static_cast<int32_t>(prompt.size());
    std::vector<int32_t> gen = {cur_tok};

    if (use_mtp && model.has_mtp()) {
        std::vector<int32_t> next_t(prompt.size());
        for (size_t k = 0; k + 1 < prompt.size(); ++k) next_t[k] = prompt[k + 1];
        next_t.back() = cur_tok;
        int32_t draft = model.mtp_prefill_chunk(next_t, pos);

        rookery::MtpStats warm_st{};
        if (n_gen > 2) {
            std::vector<int32_t> wout;
            model.decode_mtp_greedy_step(cur_tok, draft, cur_pos, wout, warm_st);
            for (int32_t t : wout) gen.push_back(t);
        }

        rookery::MtpStats st{};
        const size_t start_n = gen.size();
        const auto t0 = std::chrono::steady_clock::now();
        while (static_cast<int>(gen.size()) < n_gen) {
            std::vector<int32_t> step_out;
            model.decode_mtp_greedy_step(cur_tok, draft, cur_pos, step_out, st);
            for (int32_t t : step_out) {
                gen.push_back(t);
                if (static_cast<int>(gen.size()) >= n_gen) break;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const int timed = static_cast<int>(gen.size() - start_n);
        std::printf("GLM-5.3-Flash MTP decode steady-state (%d tok, %d steps, accept=%.1f%%, graph=%s): %.2f ms (%.2f tok/s)\n",
                    timed, st.steps, st.steps > 0 ? 100.0 * st.accepted / st.steps : 0.0,
                    use_graph ? "on" : "off", ms, timed * 1000.0 / ms);
    } else {
        if (n_gen > 1) {
            cur_tok = model.forward_decode_greedy(cur_tok, cur_pos++);
            gen.push_back(cur_tok);
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int step = 2; step < n_gen; ++step) {
            cur_tok = model.forward_decode_greedy(cur_tok, cur_pos++);
            gen.push_back(cur_tok);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const int timed = std::max(1, n_gen - 2);
        std::printf("GLM-5.3-Flash decode steady-state (%d tok, graph=%s): %.2f ms (%.2f tok/s)\n",
                    timed, use_graph ? "on" : "off", ms, timed * 1000.0 / ms);
    }
    std::printf("Output: %s%s\n", prompt_str.c_str(), tok.decode(gen, false).c_str());
    return 0;
}
