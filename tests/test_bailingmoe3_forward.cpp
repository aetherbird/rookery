#include "gguf/sharded.h"
#include "models/bailingmoe3.h"
#include "tokenizer/bpe.h"

#include <algorithm>
#include <chrono>
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

static void run_bench_matrix(rookery::BailingMoe3Model & model,
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
        out.reserve(n);
        while (static_cast<int>(out.size()) < n) {
            for (int32_t id : unit_ids) {
                out.push_back(id);
                if (static_cast<int>(out.size()) == n) break;
            }
        }
        return out;
    };

    std::vector<float> logits;

    // Warmup cuBLAS GEMM handles & WMMA kernels once before timing pp (matches llama-bench!)
    if (!pp_lens.empty()) {
        model.reset_state();
        auto wids = make_ids(32);
        std::vector<int32_t> wpos(32);
        std::iota(wpos.begin(), wpos.end(), 0);
        model.forward(wids, wpos, logits);
    }

    for (int pp : pp_lens) {
        model.reset_state();
        auto ids = make_ids(pp);
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t off = 0; off < ids.size(); off += chunk) {
            const size_t n = std::min(chunk, ids.size() - off);
            std::vector<int32_t> ctoks(ids.begin() + static_cast<ptrdiff_t>(off), ids.begin() + static_cast<ptrdiff_t>(off + n));
            std::vector<int32_t> cpos(n);
            std::iota(cpos.begin(), cpos.end(), static_cast<int32_t>(off));
            model.forward(ctoks, cpos, logits);
            if (model.has_mtp()) {
                std::vector<int32_t> ntoks(n);
                for (size_t i = 0; i + 1 < n; ++i) ntoks[i] = ids[off + i + 1];
                ntoks[n - 1] = (off + n < ids.size())
                    ? ids[off + n]
                    : static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
                model.mtp_prefill_chunk(ntoks, cpos);
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("[BENCH] pp%-5d (mtp=%s): %8.2f ms | %8.2f tok/s\n",
                    pp, model.has_mtp() ? "on" : "off", ms, pp * 1000.0 / ms);
        std::fflush(stdout);
    }

    for (int d : depths) {
        model.reset_state();
        const int p_len = std::max(16, d);
        auto ids = make_ids(p_len);
        int32_t draft_tok = 0;
        for (size_t off = 0; off < ids.size(); off += chunk) {
            const size_t n = std::min(chunk, ids.size() - off);
            std::vector<int32_t> ctoks(ids.begin() + static_cast<ptrdiff_t>(off), ids.begin() + static_cast<ptrdiff_t>(off + n));
            std::vector<int32_t> cpos(n);
            std::iota(cpos.begin(), cpos.end(), static_cast<int32_t>(off));
            model.forward(ctoks, cpos, logits);
            if (model.has_mtp()) {
                std::vector<int32_t> ntoks(n);
                for (size_t i = 0; i + 1 < n; ++i) ntoks[i] = ids[off + i + 1];
                ntoks[n - 1] = (off + n < ids.size())
                    ? ids[off + n]
                    : static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
                draft_tok = model.mtp_prefill_chunk(ntoks, cpos);
            }
        }
        int32_t cur = static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        int32_t cur_pos = p_len;

        // Warmup 2 decode steps before timing
        rookery::MtpStats warm_st;
        for (int w = 0; w < 2; ++w) {
            if (model.has_mtp()) {
                std::vector<int32_t> st;
                model.decode_mtp_greedy_step(cur, draft_tok, cur_pos, st, warm_st);
            } else {
                cur = model.forward_decode_greedy(cur, cur_pos++);
            }
        }

        rookery::MtpStats mtp_st;
        int gen = 0;
        const auto g0 = std::chrono::steady_clock::now();
        while (gen < n_gen) {
            if (model.has_mtp()) {
                std::vector<int32_t> st;
                model.decode_mtp_greedy_step(cur, draft_tok, cur_pos, st, mtp_st);
                gen += static_cast<int>(st.size());
            } else {
                cur = model.forward_decode_greedy(cur, cur_pos++);
                gen += 1;
            }
        }
        const auto g1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
        if (model.has_mtp()) {
            std::printf("[BENCH] tg%-3d @ d=%-6d (mtp=on ): %8.2f ms | %8.2f tok/s (accept=%d/%d = %.1f%%)\n",
                        gen, d, ms, gen * 1000.0 / ms, mtp_st.accepted, mtp_st.steps,
                        mtp_st.steps > 0 ? 100.0 * mtp_st.accepted / mtp_st.steps : 0.0);
        } else {
            std::printf("[BENCH] tg%-3d @ d=%-6d (mtp=off): %8.2f ms | %8.2f tok/s\n",
                        gen, d, ms, gen * 1000.0 / ms);
        }
        std::fflush(stdout);
    }
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The capital of France is";
    std::vector<int>   gpus = {0};
    std::vector<float> split;
    std::vector<int>   bench_pp;
    std::vector<int>   bench_d;
    int  max_ctx  = 8192;
    int  n_gen    = 64;
    int  chunk    = 512;
    bool kv_q8_0  = true;
    bool no_graph = false;
    bool use_mtp  = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m" || a == "--model")      model_path = next();
        else if (a == "-p" || a == "--prompt") prompt = next();
        else if (a == "--gpus")               gpus = parse_csv_ints(next());
        else if (a == "--split" || a == "-ts") split = parse_csv_floats(next());
        else if (a == "-c" || a == "--ctx")   max_ctx = std::atoi(next().c_str());
        else if (a == "-n" || a == "--gen")   n_gen = std::atoi(next().c_str());
        else if (a == "--chunk")              chunk = std::atoi(next().c_str());
        else if (a == "--kv-f16")             kv_q8_0 = false;
        else if (a == "--no-graph")           no_graph = true;
        else if (a == "--no-mtp")             use_mtp = false;
        else if (a == "--bench-pp")           bench_pp = parse_csv_ints(next());
        else if (a == "--bench-d")            bench_d = parse_csv_ints(next());
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "Usage: %s -m <bailingmoe3.gguf> [--gpus 0,...] [--split ...] [-c 8192] [-n 64] [--chunk 512] [--no-mtp] [--bench-pp 512,2048] [--bench-d 0,4096] [-p prompt]\n", argv[0]);
        return 1;
    }

    rookery::BpeTokenizer tok;
    {
        auto sg = rookery::ShardedGguf::open(model_path);
        tok.load(sg.meta());
    }

    rookery::BailingMoe3Model model;
    model.load(model_path, gpus, split, max_ctx, kv_q8_0, use_mtp);
    if (no_graph) model.set_use_cuda_graphs(false);
    model.warmup_decode_graphs();

    if (!bench_pp.empty() || !bench_d.empty()) {
        run_bench_matrix(model, tok, bench_pp, bench_d, n_gen, static_cast<size_t>(chunk));
        return 0;
    }

    auto ids = tok.encode(prompt, tok.add_bos_default(), true);
    std::printf("Prompt (%zu tok):", ids.size());
    for (size_t i = 0; i < std::min<size_t>(ids.size(), 16); ++i) std::printf(" %d", ids[i]);
    if (ids.size() > 16) std::printf(" ...");
    std::printf("\n");

    const size_t p_chunk = static_cast<size_t>(chunk);
    std::vector<float> logits;
    int32_t draft_tok = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t off = 0; off < ids.size(); off += p_chunk) {
        const size_t n = std::min(p_chunk, ids.size() - off);
        std::vector<int32_t> ctoks(ids.begin() + static_cast<ptrdiff_t>(off), ids.begin() + static_cast<ptrdiff_t>(off + n));
        std::vector<int32_t> cpos(n);
        std::iota(cpos.begin(), cpos.end(), static_cast<int32_t>(off));
        model.forward(ctoks, cpos, logits);
        if (model.has_mtp()) {
            std::vector<int32_t> ntoks(n);
            for (size_t i = 0; i + 1 < n; ++i) ntoks[i] = ids[off + i + 1];
            if (off + n < ids.size()) {
                ntoks[n - 1] = ids[off + n];
            } else {
                ntoks[n - 1] = static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
            }
            draft_tok = model.mtp_prefill_chunk(ntoks, cpos);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double pp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Prefill %zu tok: %.2f ms (%.1f tok/s)\n", ids.size(), pp_ms, ids.size() * 1000.0 / std::max(pp_ms, 0.001));

    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("Top-5 logits:\n");
    for (int k = 0; k < 5; ++k) {
        std::printf("  #%d id=%d (%s) logit=%.4f\n", k + 1, idx[k], tok.decode_token(idx[k]).c_str(), logits[idx[k]]);
    }

    int32_t cur = idx[0];
    int32_t cur_pos = static_cast<int32_t>(ids.size());
    std::string out = tok.decode_token(cur);
    std::vector<int32_t> gen_ids = {cur};

    rookery::MtpStats mtp_st;
    const auto g0 = std::chrono::steady_clock::now();
    int timed = 0;
    while (static_cast<int>(gen_ids.size()) < n_gen) {
        if (tok.is_stop_token(cur)) break;
        if (model.has_mtp()) {
            std::vector<int32_t> step_toks;
            model.decode_mtp_greedy_step(cur, draft_tok, cur_pos, step_toks, mtp_st);
            for (int32_t tid : step_toks) {
                gen_ids.push_back(tid);
                out += tok.decode_token(tid);
                ++timed;
                if (tok.is_stop_token(tid) || static_cast<int>(gen_ids.size()) >= n_gen) break;
            }
        } else {
            cur = model.forward_decode_greedy(cur, cur_pos++);
            gen_ids.push_back(cur);
            out += tok.decode_token(cur);
            ++timed;
        }
    }
    const auto g1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
    if (model.has_mtp()) {
        std::printf("Decode (MTP) %d tok: %.2f ms (%.2f tok/s), accept=%d/%d (%.1f%%)\n",
                    timed, ms, timed > 0 ? (timed * 1000.0 / ms) : 0.0,
                    mtp_st.accepted, mtp_st.steps,
                    mtp_st.steps > 0 ? (100.0 * mtp_st.accepted / mtp_st.steps) : 0.0);
    } else {
        std::printf("Decode %d tok: %.2f ms (%.2f tok/s)\n", timed, ms, timed > 0 ? (timed * 1000.0 / ms) : 0.0);
    }
    std::printf("Output: %s%s\n", prompt.c_str(), out.c_str());
    return 0;
}
