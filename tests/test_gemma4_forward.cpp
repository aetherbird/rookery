#include "cpu/gemma4_cpu.h"
#include "gguf/sharded.h"
#ifdef ROOKERY_ENABLE_CUDA
#include "models/gemma4.h"
#endif
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

template <typename ModelT>
static void run_bench_matrix(ModelT & model,
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
            model.forward(ctoks, cpos, logits, /*compute_logits=*/(off + n >= ids.size()));
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
            std::vector<int32_t> ctoks(ids.begin() + static_cast<ptrdiff_t>(off), ids.begin() + static_cast<ptrdiff_t>(off + n));
            std::vector<int32_t> cpos(n);
            std::iota(cpos.begin(), cpos.end(), static_cast<int32_t>(off));
            model.forward(ctoks, cpos, logits, /*compute_logits=*/(off + n >= ids.size()));
        }
        int32_t cur = static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        int32_t cur_pos = p_len;

        // Warmup 2 decode steps
        for (int w = 0; w < 2; ++w) {
            cur = model.forward_decode_greedy(cur, cur_pos++);
        }

        const auto g0 = std::chrono::steady_clock::now();
        for (int g = 0; g < n_gen; ++g) {
            cur = model.forward_decode_greedy(cur, cur_pos++);
        }
        const auto g1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
        std::printf("[BENCH] tg%-3d @ d=%-6d: %8.2f ms | %8.2f tok/s\n", n_gen, d, ms, n_gen * 1000.0 / ms);
        std::fflush(stdout);
    }
}

template <typename ModelT>
static void run_prompt(ModelT & model,
                       const rookery::BpeTokenizer & tok,
                       const std::string & prompt,
                       int n_gen,
                       size_t p_chunk) {
    auto ids = tok.encode(prompt, tok.add_bos_default(), true);
    std::printf("Prompt (%zu tok):", ids.size());
    for (size_t i = 0; i < std::min<size_t>(ids.size(), 16); ++i) std::printf(" %d", ids[i]);
    if (ids.size() > 16) std::printf(" ...");
    std::printf("\n");

    std::vector<float> logits;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t off = 0; off < ids.size(); off += p_chunk) {
        const size_t n = std::min(p_chunk, ids.size() - off);
        std::vector<int32_t> ctoks(ids.begin() + static_cast<ptrdiff_t>(off), ids.begin() + static_cast<ptrdiff_t>(off + n));
        std::vector<int32_t> cpos(n);
        std::iota(cpos.begin(), cpos.end(), static_cast<int32_t>(off));
        model.forward(ctoks, cpos, logits, /*compute_logits=*/(off + n >= ids.size()));
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
    int gen = 1;
    const auto g0 = std::chrono::steady_clock::now();
    while (gen < n_gen && !tok.is_stop_token(cur)) {
        cur = model.forward_decode_greedy(cur, cur_pos++);
        if (tok.is_stop_token(cur)) break;
        out += tok.decode_token(cur);
        ++gen;
    }
    const auto g1 = std::chrono::steady_clock::now();
    const double tg_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
    std::printf("Decode %d tok: %.2f ms (%.2f tok/s)\n", gen - 1, tg_ms, (gen - 1) * 1000.0 / std::max(tg_ms, 0.001));
    std::printf("Output: %s%s\n", prompt.c_str(), out.c_str());
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The capital of France is";
    std::vector<int>   gpus = {0};
    std::vector<float> split;
    std::vector<int>   bench_pp;
    std::vector<int>   bench_d;
    int  max_ctx   = 8192;
    int  n_gen     = 64;
    int  chunk     = 256;
    int  n_threads = 0;
    rookery::KvCacheType kv_type = rookery::KvCacheType::Q8_0;
    float rope_scale = 1.0f;
    bool no_graph  = false;
#ifdef ROOKERY_ENABLE_CUDA
    bool use_cpu   = false;
#else
    bool use_cpu   = true;
#endif

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m" || a == "--model")       model_path = next();
        else if (a == "-p" || a == "--prompt") prompt = next();
        else if (a == "--gpus")                gpus = parse_csv_ints(next());
        else if (a == "--split" || a == "-ts") split = parse_csv_floats(next());
        else if (a == "-c" || a == "--ctx")    max_ctx = std::atoi(next().c_str());
        else if (a == "-n" || a == "--gen")    n_gen = std::atoi(next().c_str());
        else if (a == "-t" || a == "--threads") n_threads = std::atoi(next().c_str());
        else if (a == "--cpu")                 use_cpu = true;
        else if (a == "--chunk")               chunk = std::atoi(next().c_str());
        else if (a == "--kv-f16")              kv_type = rookery::KvCacheType::F16;
        else if (a == "--kv-q4")               kv_type = rookery::KvCacheType::Q4_0;
        else if (a == "--rope-scale")          rope_scale = std::stof(next());
        else if (a == "--no-graph")            no_graph = true;
        else if (a == "--bench-pp")            bench_pp = parse_csv_ints(next());
        else if (a == "--bench-d")             bench_d = parse_csv_ints(next());
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "Usage: %s -m <gemma4.gguf> [--cpu] [-t 4] [--gpus 0,...] [--split ...] [-c 8192] [-n 64] [--chunk 256] [--kv-f16|--kv-q4] [--rope-scale 1.0] [--no-graph] [--bench-pp 128] [--bench-d 0] [-p prompt]\n", argv[0]);
        return 1;
    }

    rookery::BpeTokenizer tok;
    {
        auto sg = rookery::ShardedGguf::open(model_path);
        tok.load(sg.meta());
    }
    tok.add_stop_token("<turn|>");
    tok.add_stop_token("<eos>");

    if (use_cpu) {
        rookery::cpu::Gemma4CpuModel model;
        model.load(model_path, max_ctx, kv_type, n_threads);
        if (!bench_pp.empty() || !bench_d.empty()) {
            run_bench_matrix(model, tok, bench_pp, bench_d, n_gen, static_cast<size_t>(chunk));
            return 0;
        }
        run_prompt(model, tok, prompt, n_gen, static_cast<size_t>(chunk));
        return 0;
    }

#ifdef ROOKERY_ENABLE_CUDA
    rookery::Gemma4Model model;
    model.load(model_path, gpus, split, max_ctx, kv_type, rope_scale);
    if (no_graph) model.set_use_cuda_graphs(false);
    model.warmup_decode_graphs();

    if (!bench_pp.empty() || !bench_d.empty()) {
        run_bench_matrix(model, tok, bench_pp, bench_d, n_gen, static_cast<size_t>(chunk));
        return 0;
    }
    run_prompt(model, tok, prompt, n_gen, static_cast<size_t>(chunk));
#else
    (void) gpus; (void) split; (void) no_graph; (void) rope_scale;
#endif
    return 0;
}
