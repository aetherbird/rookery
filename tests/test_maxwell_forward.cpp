#include "models/maxwell_moe.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int32_t> parse_int_list(const std::string & s) {
    std::vector<int32_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

static std::vector<float> parse_float_list(const std::string & s) {
    std::vector<float> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stof(item));
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::printf("Usage: %s <maxwell.gguf> [--gpus 0,1] [--split 24,32] [--tokens 1,100] [--gen 128] [--kv f16|q8_0|q4_0] [--mtp] [--mtp-pos-off 0|1] [--no-graph]\n", argv[0]);
        return 1;
    }

    std::string path = argv[1];
    std::vector<int> gpus = {0, 1};
    std::vector<float> split = {24.0f, 32.0f};
    std::vector<int32_t> prompt = {1, 820, 9779, 315, 15493, 362};
    int n_gen = 128;
    int max_ctx = 4096;
    int fill_ctx = 0;
    rookery::KvCacheType kv_type = rookery::KvCacheType::Q8_0;
    bool use_graph = true;
    bool use_mtp = false;
    int  mtp_pos_off = 0;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gpus" && i + 1 < argc) {
            gpus.clear();
            for (int v : parse_int_list(argv[++i])) gpus.push_back(v);
        } else if (a == "--split" && i + 1 < argc) {
            split = parse_float_list(argv[++i]);
        } else if (a == "--tokens" && i + 1 < argc) {
            prompt = parse_int_list(argv[++i]);
        } else if (a == "--gen" && i + 1 < argc) {
            n_gen = std::atoi(argv[++i]);
        } else if (a == "--ctx" && i + 1 < argc) {
            max_ctx = std::atoi(argv[++i]);
        } else if (a == "--fill-ctx" && i + 1 < argc) {
            fill_ctx = std::atoi(argv[++i]);
        } else if (a == "--kv" && i + 1 < argc) {
            kv_type = rookery::parse_kv_cache_type(argv[++i]);
        } else if (a == "--no-graph") {
            use_graph = false;
        } else if (a == "--mtp") {
            use_mtp = true;
        } else if (a == "--mtp-pos-off" && i + 1 < argc) {
            mtp_pos_off = std::atoi(argv[++i]);
        }
    }

    if (fill_ctx > 0 && static_cast<int>(prompt.size()) < fill_ctx) {
        // Repeat prompt tokens to reach fill_ctx (e.g. 20000 tokens to exercise >16K MSA block-sparse attention!)
        std::vector<int32_t> base = prompt;
        prompt.reserve(fill_ctx);
        while (static_cast<int>(prompt.size()) < fill_ctx) {
            for (size_t k = 1; k < base.size() && static_cast<int>(prompt.size()) < fill_ctx; ++k) {
                prompt.push_back(base[k]);
            }
        }
        if (max_ctx < fill_ctx + n_gen + 64) max_ctx = fill_ctx + n_gen + 64;
    }

    rookery::MaxwellMoeModel model;
    model.set_use_cuda_graphs(use_graph);
    model.set_mtp_pos_offset(mtp_pos_off);
    model.load(path, gpus, split, max_ctx, kv_type, use_mtp);

    std::vector<int32_t> pos(prompt.size());
    std::iota(pos.begin(), pos.end(), 0);

    std::vector<float> logits;
    const auto t_pp0 = std::chrono::steady_clock::now();
    const size_t chunk = 512;
    for (size_t off = 0; off < prompt.size(); off += chunk) {
        size_t len = std::min(chunk, prompt.size() - off);
        std::vector<int32_t> pt(prompt.begin() + off, prompt.begin() + off + len);
        std::vector<int32_t> pp(pos.begin() + off, pos.begin() + off + len);
        model.forward(pt, pp, logits);
        if (use_mtp && model.has_mtp() && off + len < prompt.size()) {
            std::vector<int32_t> next_t(len);
            for (size_t k = 0; k < len; ++k) next_t[k] = prompt[off + k + 1];
            model.mtp_prefill_chunk(next_t, pp);
        }
    }
    const auto t_pp1 = std::chrono::steady_clock::now();
    const double pp_ms = std::chrono::duration<double, std::milli>(t_pp1 - t_pp0).count();

    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("Prefill top-5:");
    for (int k = 0; k < 5; ++k) std::printf("  [%d]=%.4f", idx[k], logits[idx[k]]);
    std::printf("\nPrefill (%zu tok): %.2f ms (%.1f tok/s)\n",
                prompt.size(), pp_ms, prompt.size() * 1000.0 / pp_ms);

    int32_t cur_tok = idx[0];
    std::vector<int32_t> gen_tokens;
    gen_tokens.push_back(cur_tok);
    int32_t cur_pos = static_cast<int32_t>(prompt.size());

    if (use_mtp && model.has_mtp()) {
        const size_t last_off = ((prompt.size() - 1) / chunk) * chunk;
        const size_t last_len = prompt.size() - last_off;
        std::vector<int32_t> next_t(last_len);
        std::vector<int32_t> last_p(last_len);
        for (size_t k = 0; k + 1 < last_len; ++k) {
            next_t[k] = prompt[last_off + k + 1];
            last_p[k] = static_cast<int32_t>(last_off + k);
        }
        next_t[last_len - 1] = cur_tok;
        last_p[last_len - 1] = static_cast<int32_t>(prompt.size() - 1);
        int32_t draft_tok = model.mtp_prefill_chunk(next_t, last_p);
        rookery::MtpStats warmup_stats{}, stats{};

        // 1 warmup step to capture T=2 CUDA Graph cleanly
        if (static_cast<int>(gen_tokens.size()) < n_gen) {
            model.decode_mtp_greedy_step(cur_tok, draft_tok, cur_pos, gen_tokens, warmup_stats);
        }

        const int before = static_cast<int>(gen_tokens.size());
        const auto t0 = std::chrono::steady_clock::now();
        while (static_cast<int>(gen_tokens.size()) < n_gen) {
            model.decode_mtp_greedy_step(cur_tok, draft_tok, cur_pos, gen_tokens, stats);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const int timed = static_cast<int>(gen_tokens.size()) - before;
        const double acc_pct = stats.steps > 0 ? (100.0 * stats.accepted / stats.steps) : 0.0;
        std::printf("Decode MTP steady-state (%d tok, %d steps, accept=%.1f%%, pos_off=%d, graph=%s): %.2f ms (%.2f tok/s)\n",
                    timed, stats.steps, acc_pct, mtp_pos_off, use_graph ? "on" : "off", ms, timed * 1000.0 / ms);
    } else {
        if (n_gen > 1) {
            cur_tok = model.forward_decode_greedy(cur_tok, cur_pos++);
            gen_tokens.push_back(cur_tok);
        }
        const auto t_tg0 = std::chrono::steady_clock::now();
        for (int step = 2; step < n_gen; ++step) {
            cur_tok = model.forward_decode_greedy(cur_tok, cur_pos++);
            gen_tokens.push_back(cur_tok);
        }
        const auto t_tg1 = std::chrono::steady_clock::now();
        const double tg_ms = std::chrono::duration<double, std::milli>(t_tg1 - t_tg0).count();
        const int timed = std::max(1, n_gen - 2);
        std::printf("Decode steady-state (%d tok, graph=%s): %.2f ms (%.2f tok/s)\n",
                    timed, use_graph ? "on" : "off", tg_ms, timed * 1000.0 / tg_ms);
    }

    std::printf("Generated token IDs:");
    for (size_t i = 0; i < std::min<size_t>(gen_tokens.size(), 48); ++i) std::printf(" %d", gen_tokens[i]);
    std::printf("\n");
    return 0;
}
