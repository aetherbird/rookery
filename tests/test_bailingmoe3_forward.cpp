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
    int  max_ctx = 8192;
    int  n_gen   = 64;
    bool kv_q8_0 = true;
    bool no_graph = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m" || a == "--model")   model_path = next();
        else if (a == "-p" || a == "--prompt") prompt = next();
        else if (a == "--gpus")            gpus = parse_csv_ints(next());
        else if (a == "--split" || a == "-ts") split = parse_csv_floats(next());
        else if (a == "-c" || a == "--ctx") max_ctx = std::atoi(next().c_str());
        else if (a == "-n" || a == "--gen") n_gen = std::atoi(next().c_str());
        else if (a == "--kv-f16")          kv_q8_0 = false;
        else if (a == "--no-graph")        no_graph = true;
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "Usage: %s -m <bailingmoe3.gguf> [--gpus 0,...] [--split ...] [-c 8192] [-n 64] [-p prompt]\n", argv[0]);
        return 1;
    }

    rookery::BpeTokenizer tok;
    {
        auto sg = rookery::ShardedGguf::open(model_path);
        tok.load(sg.meta());
    }

    rookery::BailingMoe3Model model;
    model.load(model_path, gpus, split, max_ctx, kv_q8_0);
    if (no_graph) model.set_use_cuda_graphs(false);

    auto ids = tok.encode(prompt, tok.add_bos_default(), false);
    std::printf("Prompt (%zu tok):", ids.size());
    for (int32_t id : ids) std::printf(" %d", id);
    std::printf("\n");

    std::vector<int32_t> pos(ids.size());
    std::iota(pos.begin(), pos.end(), 0);
    std::vector<float> logits;
    const auto t0 = std::chrono::steady_clock::now();
    model.forward(ids, pos, logits);
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("Prefill %zu tok: %.2f ms\n", ids.size(), std::chrono::duration<double, std::milli>(t1 - t0).count());

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

    // Warmup graph capture on step 0, then time remaining decode steps
    if (n_gen > 1) {
        cur = model.forward_decode_greedy(cur, cur_pos++);
        out += tok.decode_token(cur);
    }
    const auto g0 = std::chrono::steady_clock::now();
    int timed = 0;
    for (int step = 2; step < n_gen; ++step) {
        if (tok.is_stop_token(cur)) break;
        cur = model.forward_decode_greedy(cur, cur_pos++);
        out += tok.decode_token(cur);
        ++timed;
    }
    const auto g1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
    std::printf("Decode %d tok: %.2f ms (%.2f tok/s)\n", timed, ms, timed > 0 ? (timed * 1000.0 / ms) : 0.0);
    std::printf("Output: %s%s\n", prompt.c_str(), out.c_str());
    return 0;
}
