#include "gguf/sharded.h"
#include "models/qwen35_moe.h"
#include "tokenizer/bpe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
        std::printf("Usage: %s <Qwen3.6-35B-A3B-Q8_0.gguf> [--gpus 0,1] [--split 18,22] [--prompt \"...\"] [--gen 64] [--ctx 65536] [--pp-test 2048] [--no-mtp]\n", argv[0]);
        return 1;
    }
    std::string path = argv[1];
    std::vector<int>   gpus  = {0, 1};
    std::vector<float> split;
    std::string prompt_str   = "The capital of France is";
    int  n_gen     = 64;
    int  max_ctx   = 65536;
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
    auto prompt = tok.encode(prompt_str, false, true);
    std::printf("Prompt '%s' -> %zu tokens:", prompt_str.c_str(), prompt.size());
    for (int32_t t : prompt) std::printf(" %d", t);
    std::printf("\n");

    rookery::Qwen35MoeModel model;
    model.set_use_cuda_graphs(use_graph);
    model.load(path, gpus, split, max_ctx, true, use_mtp);

    if (pp_test > 0) {
        const std::string para =
            "In modern GPU inference engines, batched prompt prefill groups routed MoE tokens by expert so each "
            "quantized weight matrix is read from HBM once per batch instead of once per token. Combined with "
            "Tensor Core FlashAttention and instant host-RAM state checkpoints for gated delta-net layers, "
            "multi-turn conversations achieve sub-second time to first token. ";
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

        std::vector<float> logits_b;
        model.reset_state();
        model.forward({long_p.begin(), long_p.begin() + std::min(16, N)}, {lpos.begin(), lpos.begin() + std::min(16, N)}, logits_b);

        model.reset_state();
        const auto tb0 = std::chrono::steady_clock::now();
        for (int off = 0; off < N; off += 512) {
            int len = std::min(512, N - off);
            model.forward({long_p.begin() + off, long_p.begin() + off + len},
                          {lpos.begin() + off, lpos.begin() + off + len}, logits_b);
        }
        const auto tb1 = std::chrono::steady_clock::now();
        const double b_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();

        const int half = N / 2;
        rookery::Qwen35MoeModel::GdnCheckpoint cp;
        std::vector<float> logits_cp;
        model.reset_state();
        for (int off = 0; off < half; off += 512) {
            int len = std::min(512, half - off);
            model.forward({long_p.begin() + off, long_p.begin() + off + len},
                          {lpos.begin() + off, lpos.begin() + off + len}, logits_cp);
        }
        const auto tcp0 = std::chrono::steady_clock::now();
        model.save_gdn_checkpoint(cp, half);
        model.reset_state();
        model.restore_gdn_checkpoint(cp);
        const auto tcp1 = std::chrono::steady_clock::now();
        for (int off = half; off < N; off += 512) {
            int len = std::min(512, N - off);
            model.forward({long_p.begin() + off, long_p.begin() + off + len},
                          {lpos.begin() + off, lpos.begin() + off + len}, logits_cp);
        }

        float max_diff_cp = 0.0f;
        for (size_t i = 0; i < logits_b.size(); ++i) {
            max_diff_cp = std::max(max_diff_cp, std::fabs(logits_b[i] - logits_cp[i]));
        }
        std::vector<int> ix(logits_b.size());
        std::iota(ix.begin(), ix.end(), 0);
        std::partial_sort(ix.begin(), ix.begin() + 3, ix.end(), [&](int a, int b) { return logits_b[a] > logits_b[b]; });
        std::printf("\n=== PP-TEST (%d tokens) ===\n", N);
        std::printf("Batched prefill: %.2f ms (%.1f tok/s)\n", b_ms, N * 1000.0 / b_ms);
        std::printf("GDN checkpoint save+restore: %.2f ms | Max |batched - ckpt| = %.6f\n",
                    std::chrono::duration<double, std::milli>(tcp1 - tcp0).count(), max_diff_cp);
        std::printf("Top-3:");
        for (int k = 0; k < 3; ++k) std::printf(" [%d '%s']=%.4f", ix[k], tok.decode_token(ix[k]).c_str(), logits_b[ix[k]]);
        std::printf("\n");
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
        std::printf("Qwen3.5-MoE MTP decode steady-state (%d tok, %d steps, accept=%.1f%%, graph=%s): %.2f ms (%.2f tok/s)\n",
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
        std::printf("Qwen3.5-MoE decode steady-state (%d tok, graph=%s): %.2f ms (%.2f tok/s)\n",
                    timed, use_graph ? "on" : "off", ms, timed * 1000.0 / ms);
    }
    std::printf("Output: %s%s\n", prompt_str.c_str(), tok.decode(gen, false).c_str());
    return 0;
}
