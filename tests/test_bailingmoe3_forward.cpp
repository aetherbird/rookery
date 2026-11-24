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

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The capital of France is";
    std::vector<int>   gpus = {0};
    std::vector<float> split;
    int  max_ctx  = 8192;
    int  n_gen    = 64;
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
        else if (a == "--kv-f16")             kv_q8_0 = false;
        else if (a == "--no-graph")           no_graph = true;
        else if (a == "--no-mtp")             use_mtp = false;
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "Usage: %s -m <bailingmoe3.gguf> [--gpus 0,...] [--split ...] [-c 8192] [-n 64] [--no-mtp] [-p prompt]\n", argv[0]);
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

    auto ids = tok.encode(prompt, tok.add_bos_default(), true);
    std::printf("Prompt (%zu tok):", ids.size());
    for (size_t i = 0; i < std::min<size_t>(ids.size(), 16); ++i) std::printf(" %d", ids[i]);
    if (ids.size() > 16) std::printf(" ...");
    std::printf("\n");

    const size_t chunk = 512;
    std::vector<float> logits;
    int32_t draft_tok = 0;
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
