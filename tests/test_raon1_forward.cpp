#include "gguf/sharded.h"
#include "models/raon1_mini.h"
#include "server/chat_template.h"
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

static std::vector<int> parse_csv_ints(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(std::stoi(item));
    return out;
}

static std::vector<float> parse_csv_floats(const std::string & s) {
    std::vector<float> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(std::stof(item));
    return out;
}

static void run_bench_matrix(rookery::Raon1MiniModel & model,
                             const rookery::BpeTokenizer & tok,
                             const std::vector<int> & pp_lens,
                             const std::vector<int> & depths,
                             int n_gen,
                             size_t chunk) {
    const std::string base_text =
        "In distributed systems and GPU kernel engineering, memory bandwidth utilization, "
        "warp occupancy, shared-memory bank-conflict elimination, and register pressure "
        "determine whether an autoregressive language model achieves roofline throughput. ";
    auto unit_ids = tok.encode(base_text, false, false);
    auto make_ids = [&](int n) {
        std::vector<int32_t> out;
        out.reserve(static_cast<size_t>(n));
        while (static_cast<int>(out.size()) < n) {
            for (int32_t id : unit_ids) {
                out.push_back(id);
                if (static_cast<int>(out.size()) == n) break;
            }
        }
        return out;
    };

    std::vector<float> logits;
    if (!pp_lens.empty()) {
        model.reset_state();
        auto wids = make_ids(32);
        std::vector<int32_t> wpos(32);
        std::iota(wpos.begin(), wpos.end(), 0);
        model.forward(wids, wpos, logits, true);
    }

    for (int pp : pp_lens) {
        model.reset_state();
        auto ids = make_ids(pp);
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t off = 0; off < ids.size(); off += chunk) {
            const size_t n = std::min(chunk, ids.size() - off);
            std::vector<int32_t> ctoks(ids.begin() + static_cast<ptrdiff_t>(off),
                                       ids.begin() + static_cast<ptrdiff_t>(off + n));
            std::vector<int32_t> cpos(n);
            std::iota(cpos.begin(), cpos.end(), static_cast<int32_t>(off));
            model.forward(ctoks, cpos, logits, off + n >= ids.size());
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("[BENCH] pp%-5d: %8.2f ms | %8.2f tok/s\n", pp, ms, pp * 1000.0 / ms);
        std::fflush(stdout);
    }

    for (int d : depths) {
        model.reset_state();
        const int p_len = std::max(16, d);
        auto ids = make_ids(p_len);
        for (size_t off = 0; off < ids.size(); off += chunk) {
            const size_t n = std::min(chunk, ids.size() - off);
            std::vector<int32_t> ctoks(ids.begin() + static_cast<ptrdiff_t>(off),
                                       ids.begin() + static_cast<ptrdiff_t>(off + n));
            std::vector<int32_t> cpos(n);
            std::iota(cpos.begin(), cpos.end(), static_cast<int32_t>(off));
            model.forward(ctoks, cpos, logits, off + n >= ids.size());
        }
        int32_t cur = static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        int32_t cur_pos = p_len;

        for (int w = 0; w < 2; ++w) cur = model.forward_decode_greedy(cur, cur_pos++);

        const auto g0 = std::chrono::steady_clock::now();
        for (int g = 0; g < n_gen; ++g) cur = model.forward_decode_greedy(cur, cur_pos++);
        const auto g1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
        std::printf("[BENCH] tg%-3d @ d=%-6d: %8.2f ms | %8.2f tok/s\n", n_gen, d, ms, n_gen * 1000.0 / ms);
        std::fflush(stdout);
    }
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The capital of France is";
    std::string effort;
    std::vector<int>   gpus = {0};
    std::vector<float> split;
    std::vector<int>   bench_pp;
    std::vector<int>   bench_d;
    int  max_ctx    = 16384;
    int  n_gen      = 64;
    int  chunk      = 512;
    int  pp_test    = 0;
    bool kv_q8_0    = true;
    bool no_graph   = false;
    bool chat_mode  = false;
    bool think_mode = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m" || a == "--model")           model_path = next();
        else if (a == "-p" || a == "--prompt")     prompt = next();
        else if (a == "--gpus")                    gpus = parse_csv_ints(next());
        else if (a == "--split" || a == "-ts")     split = parse_csv_floats(next());
        else if (a == "-c" || a == "--ctx")        max_ctx = std::atoi(next().c_str());
        else if (a == "--gen" || a == "-n")        n_gen = std::atoi(next().c_str());
        else if (a == "--chunk")                   chunk = std::atoi(next().c_str());
        else if (a == "--pp-test")                 pp_test = std::atoi(next().c_str());
        else if (a == "--bench-pp")                bench_pp = parse_csv_ints(next());
        else if (a == "--bench-d")                 bench_d = parse_csv_ints(next());
        else if (a == "--kv-f16")                  kv_q8_0 = false;
        else if (a == "--no-graph")                no_graph = true;
        else if (a == "--chat")                    chat_mode = true;
        else if (a == "--no-think")                think_mode = false;
        else if (a == "--reasoning-effort")        effort = next();
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "Usage: %s -m <raon.gguf> [--gpus 0,1] [--split 19,6] [-c 262144] [--pp-test 64] [--bench-pp 512,2048] [--bench-d 0,4096] [--gen 128] [--chat] [--no-think] [--reasoning-effort high]\n", argv[0]);
        return 1;
    }

    auto sg = rookery::ShardedGguf::open(model_path);
    rookery::BpeTokenizer tok;
    tok.load(sg.meta());
    tok.add_stop_token("<|im_end|>");
    tok.add_stop_token("<|endoftext|>");

    rookery::Raon1MiniModel model;
    model.load(model_path, gpus, split, max_ctx, kv_q8_0);
    if (!effort.empty()) model.set_reasoning_effort(effort);
    else if (chat_mode && !think_mode) model.set_reasoning_effort("none");
    if (no_graph) model.set_use_cuda_graphs(false);

    // Optional batched-vs-sequential + RDA checkpoint parity check
    if (pp_test > 0) {
        const std::string base =
            "KRAFTON AI developed Raon-1-mini with 4-stream hyper-connections, RDA linear attention, "
            "NoPE gated MLA, and SiTU-GLU MoE experts. ";
        auto unit = tok.encode(base, false, false);
        std::vector<int32_t> ids;
        while (static_cast<int>(ids.size()) < pp_test) {
            for (int32_t id : unit) {
                ids.push_back(id);
                if (static_cast<int>(ids.size()) == pp_test) break;
            }
        }
        std::vector<int32_t> pos(ids.size());
        std::iota(pos.begin(), pos.end(), 0);

        model.reset_state();
        std::vector<float> logits_batch;
        model.forward(ids, pos, logits_batch, true);
        int32_t top_b = static_cast<int32_t>(std::distance(logits_batch.begin(), std::max_element(logits_batch.begin(), logits_batch.end())));

        // Split at midpoint with checkpoint save/restore
        const int mid = pp_test / 2;
        model.reset_state();
        std::vector<float> tmp_l;
        std::vector<int32_t> ids1(ids.begin(), ids.begin() + mid);
        std::vector<int32_t> pos1(pos.begin(), pos.begin() + mid);
        model.forward(ids1, pos1, tmp_l, false);
        rookery::Raon1MiniModel::RdaCheckpoint cp;
        model.save_rda_checkpoint(cp, mid);

        // Corrupt state then restore checkpoint and run second half
        model.reset_state();
        model.restore_rda_checkpoint(cp);
        std::vector<int32_t> ids2(ids.begin() + mid, ids.end());
        std::vector<int32_t> pos2(pos.begin() + mid, pos.end());
        std::vector<float> logits_ckpt;
        model.forward(ids2, pos2, logits_ckpt, true);
        int32_t top_c = static_cast<int32_t>(std::distance(logits_ckpt.begin(), std::max_element(logits_ckpt.begin(), logits_ckpt.end())));

        float max_diff_ckpt = 0.0f;
        for (size_t i = 0; i < logits_batch.size(); ++i) {
            max_diff_ckpt = std::max(max_diff_ckpt, std::fabs(logits_batch[i] - logits_ckpt[i]));
        }

        // Also compare sequential token-by-token (T=1) vs batched (T=pp_test) for small pp_test <= 64
        float max_diff_seq = -1.0f;
        int32_t top_s = -1;
        if (pp_test <= 64) {
            model.reset_state();
            std::vector<float> logits_seq;
            for (int i = 0; i < pp_test; ++i) {
                model.forward_decode_logits(ids[i], i, logits_seq);
            }
            top_s = static_cast<int32_t>(std::distance(logits_seq.begin(), std::max_element(logits_seq.begin(), logits_seq.end())));
            max_diff_seq = 0.0f;
            for (size_t i = 0; i < logits_batch.size(); ++i) {
                max_diff_seq = std::max(max_diff_seq, std::fabs(logits_batch[i] - logits_seq[i]));
            }
        }
        std::printf("[PARITY T=%d] top_batch=%d (%s) | top_ckpt=%d max_diff_ckpt=%.6f",
                    pp_test, top_b, tok.decode_token(top_b).c_str(), top_c, max_diff_ckpt);
        if (max_diff_seq >= 0.0f) {
            std::printf(" | top_seq=%d max_diff_seq=%.6f", top_s, max_diff_seq);
        }
        std::printf("\n");
        std::fflush(stdout);
    }

    if (!bench_pp.empty() || !bench_d.empty()) {
        run_bench_matrix(model, tok, bench_pp, bench_d, n_gen, static_cast<size_t>(chunk));
        return 0;
    }

    std::string full_prompt = prompt;
    if (chat_mode) {
        nlohmann::json msgs = nlohmann::json::array({{{"role", "user"}, {"content", prompt}}});
        full_prompt = rookery::render_raon1_chat_prompt(msgs, nlohmann::json::array(), think_mode);
    }

    auto ids = tok.encode(full_prompt, tok.add_bos_default(), true);
    std::printf("Prompt (%zu toks, effort=%s[%.2f]): %s\n",
                ids.size(), model.reasoning_effort().c_str(), model.reasoning_effort_alpha(), full_prompt.c_str());
    std::fflush(stdout);

    model.reset_state();
    std::vector<float> logits;
    const auto p0 = std::chrono::steady_clock::now();
    for (size_t off = 0; off < ids.size(); off += static_cast<size_t>(chunk)) {
        const size_t n = std::min(static_cast<size_t>(chunk), ids.size() - off);
        std::vector<int32_t> ctoks(ids.begin() + static_cast<ptrdiff_t>(off),
                                   ids.begin() + static_cast<ptrdiff_t>(off + n));
        std::vector<int32_t> cpos(n);
        std::iota(cpos.begin(), cpos.end(), static_cast<int32_t>(off));
        model.forward(ctoks, cpos, logits, off + n >= ids.size());
    }
    const auto p1 = std::chrono::steady_clock::now();
    const double pp_ms = std::chrono::duration<double, std::milli>(p1 - p0).count();

    int32_t cur = static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    int32_t cur_pos = static_cast<int32_t>(ids.size());
    std::vector<int32_t> gen_ids;
    std::printf("--- OUTPUT ---\n");
    const auto g0 = std::chrono::steady_clock::now();
    for (int g = 0; g < n_gen; ++g) {
        if (tok.is_stop_token(cur)) break;
        gen_ids.push_back(cur);
        std::printf("%s", tok.decode_token(cur).c_str());
        std::fflush(stdout);
        if (g + 1 < n_gen) {
            cur = model.forward_decode_greedy(cur, cur_pos++);
        }
    }
    const auto g1 = std::chrono::steady_clock::now();
    const double tg_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
    std::printf("\n--------------\n");
    std::printf("Prefill: %zu toks in %.2f ms (%.1f tok/s) | Decode: %zu toks in %.2f ms (%.2f tok/s)\n",
                ids.size(), pp_ms, ids.size() * 1000.0 / pp_ms,
                gen_ids.size(), tg_ms, gen_ids.size() * 1000.0 / std::max(0.001, tg_ms));
    return 0;
}
