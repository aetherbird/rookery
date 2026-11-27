#include "cpu/qwen2_cpu.h"
#include "gguf/sharded.h"
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

static void run_bench_matrix_cpu(rookery::cpu::Qwen2CpuModel & model,
                                 const rookery::BpeTokenizer & tok,
                                 const std::vector<int> & pp_lens,
                                 const std::vector<int> & depths,
                                 int n_gen,
                                 size_t chunk) {
    const std::string base_text =
        "In distributed systems and kernel engineering, memory bandwidth utilization, "
        "cache blocking, register pressure, and SIMD throughput determine whether an "
        "autoregressive language model achieves hardware roofline performance. ";
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
        auto wids = make_ids(16);
        std::vector<int32_t> wpos(16);
        std::iota(wpos.begin(), wpos.end(), 0);
        model.forward(wids, wpos, logits);
    }

    for (int pp : pp_lens) {
        auto ids = make_ids(pp);
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t off = 0; off < ids.size(); off += chunk) {
            const size_t n = std::min(chunk, ids.size() - off);
            std::vector<int32_t> ctoks(ids.begin() + static_cast<ptrdiff_t>(off), ids.begin() + static_cast<ptrdiff_t>(off + n));
            std::vector<int32_t> cpos(n);
            std::iota(cpos.begin(), cpos.end(), static_cast<int32_t>(off));
            model.forward(ctoks, cpos, logits);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("[BENCH-QWEN2] pp%-5d (t=%d): %8.2f ms | %8.2f tok/s\n",
                    pp, model.num_threads(), ms, pp * 1000.0 / ms);
        std::fflush(stdout);
    }

    for (int d : depths) {
        const int p_len = std::max(16, d);
        auto ids = make_ids(p_len);
        for (size_t off = 0; off < ids.size(); off += chunk) {
            const size_t n = std::min(chunk, ids.size() - off);
            std::vector<int32_t> ctoks(ids.begin() + static_cast<ptrdiff_t>(off), ids.begin() + static_cast<ptrdiff_t>(off + n));
            std::vector<int32_t> cpos(n);
            std::iota(cpos.begin(), cpos.end(), static_cast<int32_t>(off));
            model.forward(ctoks, cpos, logits);
        }
        int32_t cur = static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        int32_t cur_pos = p_len;

        for (int w = 0; w < 2; ++w) cur = model.forward_decode_greedy(cur, cur_pos++);

        const auto g0 = std::chrono::steady_clock::now();
        for (int g = 0; g < n_gen; ++g) cur = model.forward_decode_greedy(cur, cur_pos++);
        const auto g1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
        std::printf("[BENCH-QWEN2] tg%-3d @ d=%-6d (t=%d): %8.2f ms | %8.2f tok/s\n",
                    n_gen, d, model.num_threads(), ms, n_gen * 1000.0 / ms);
        std::fflush(stdout);
    }
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The capital of France is";
    std::vector<int> bench_pp;
    std::vector<int> bench_d;
    int  max_ctx   = 4096;
    int  n_gen     = 32;
    int  chunk     = 128;
    int  n_threads = 0;
    rookery::KvCacheType kv_type = rookery::KvCacheType::Q8_0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m" || a == "--model")        model_path = next();
        else if (a == "-p" || a == "--prompt")  prompt = next();
        else if (a == "-c" || a == "--ctx")     max_ctx = std::atoi(next().c_str());
        else if (a == "-n" || a == "--gen")     n_gen = std::atoi(next().c_str());
        else if (a == "--chunk")                chunk = std::atoi(next().c_str());
        else if (a == "-t" || a == "--threads") n_threads = std::atoi(next().c_str());
        else if (a == "--kv-f16")               kv_type = rookery::KvCacheType::F16;
        else if (a == "--kv-q4")                kv_type = rookery::KvCacheType::Q4_0;
        else if (a == "--bench-pp")             bench_pp = parse_csv_ints(next());
        else if (a == "--bench-d")              bench_d = parse_csv_ints(next());
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "Usage: %s -m <qwen2.gguf> [-t 4] [-c 4096] [-n 32] [--chunk 128] [--kv-f16|--kv-q4] [--bench-pp 128,512] [--bench-d 0,1024] [-p prompt]\n", argv[0]);
        return 1;
    }

    rookery::BpeTokenizer tok;
    {
        auto sg = rookery::ShardedGguf::open(model_path);
        tok.load(sg.meta());
    }

    rookery::cpu::Qwen2CpuModel model;
    model.load(model_path, max_ctx, kv_type, n_threads);

    if (!bench_pp.empty() || !bench_d.empty()) {
        run_bench_matrix_cpu(model, tok, bench_pp, bench_d, n_gen, static_cast<size_t>(chunk));
        return 0;
    }

    auto ids = tok.encode(prompt, tok.add_bos_default(), true);
    std::printf("Prompt (%zu tok):", ids.size());
    for (size_t i = 0; i < std::min<size_t>(ids.size(), 16); ++i) std::printf(" %d", ids[i]);
    if (ids.size() > 16) std::printf(" ...");
    std::printf("\n");

    const size_t p_chunk = static_cast<size_t>(chunk);
    std::vector<float> logits;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t off = 0; off < ids.size(); off += p_chunk) {
        const size_t n = std::min(p_chunk, ids.size() - off);
        std::vector<int32_t> ctoks(ids.begin() + static_cast<ptrdiff_t>(off), ids.begin() + static_cast<ptrdiff_t>(off + n));
        std::vector<int32_t> cpos(n);
        std::iota(cpos.begin(), cpos.end(), static_cast<int32_t>(off));
        model.forward(ctoks, cpos, logits);
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

    const auto g0 = std::chrono::steady_clock::now();
    int timed = 0;
    while (static_cast<int>(gen_ids.size()) < n_gen) {
        if (tok.is_stop_token(cur)) break;
        cur = model.forward_decode_greedy(cur, cur_pos++);
        gen_ids.push_back(cur);
        out += tok.decode_token(cur);
        ++timed;
    }
    const auto g1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
    std::printf("Decode %d tok: %.2f ms (%.2f tok/s)\n", timed, ms, timed > 0 ? (timed * 1000.0 / ms) : 0.0);
    std::printf("Output: %s%s\n", prompt.c_str(), out.c_str());
    return 0;
}
