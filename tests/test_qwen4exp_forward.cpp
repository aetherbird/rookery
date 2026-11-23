#include "core/device.h"
#include "gguf/sharded.h"
#include "models/qwen4exp.h"
#include "tokenizer/bpe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using namespace rookery;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model.gguf> [--gpus 0,1,...] [--split w0,w1,...] [--ctx 65536] [--tokens 32] [--prompt TEXT] [--pp-test N] [--no-graphs]\n", argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::vector<int> gpu_ids;
    std::vector<float> split_w;
    int max_ctx = 65536;
    int gen_tokens = 32;
    int pp_test = 0;
    bool kv_q8 = true;
    bool use_graphs = true;
    std::string prompt = "The capital of France is";

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gpus" && i + 1 < argc) {
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t c = s.find(',', pos);
                gpu_ids.push_back(std::stoi(s.substr(pos, c - pos)));
                if (c == std::string::npos) break;
                pos = c + 1;
            }
        } else if (a == "--split" && i + 1 < argc) {
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t c = s.find(',', pos);
                split_w.push_back(std::stof(s.substr(pos, c - pos)));
                if (c == std::string::npos) break;
                pos = c + 1;
            }
        } else if (a == "--ctx" && i + 1 < argc) {
            max_ctx = std::stoi(argv[++i]);
        } else if (a == "--tokens" && i + 1 < argc) {
            gen_tokens = std::stoi(argv[++i]);
        } else if (a == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (a == "--pp-test" && i + 1 < argc) {
            pp_test = std::stoi(argv[++i]);
        } else if (a == "--kv-f16") {
            kv_q8 = false;
        } else if (a == "--no-graphs") {
            use_graphs = false;
        }
    }

    auto devs = discover_topology();
    print_topology(devs);
    if (gpu_ids.empty()) {
        for (const auto & d : devs.gpus) gpu_ids.push_back(d.index);
    }

    BpeTokenizer tok;
    {
        ShardedGguf sg = ShardedGguf::open(model_path);
        tok.load(sg.meta());
    }
    auto prompt_ids = tok.encode(prompt, tok.add_bos_default(), true);
    std::printf("Prompt '%s' -> %zu tokens:", prompt.c_str(), prompt_ids.size());
    for (int32_t id : prompt_ids) std::printf(" %d", id);
    std::printf("\n");

    Qwen4ExpModel model;
    model.load(model_path, gpu_ids, split_w, max_ctx, kv_q8);
    model.set_use_cuda_graphs(use_graphs);

    if (pp_test > 0) {
        std::printf("\n=== PP-TEST (%d tokens) ===\n", pp_test);
        std::vector<int32_t> long_ids;
        long_ids.reserve(pp_test);
        while (static_cast<int>(long_ids.size()) + static_cast<int>(prompt_ids.size()) < pp_test) {
            long_ids.insert(long_ids.end(), prompt_ids.begin(), prompt_ids.end());
        }
        while (static_cast<int>(long_ids.size()) < pp_test) {
            long_ids.push_back(prompt_ids[long_ids.size() % prompt_ids.size()]);
        }

        // Batched prefill in 256-token chunks + save checkpoint halfway
        model.reset_state();
        std::vector<float> logits_b;
        const int half = pp_test / 2;
        Qwen4ExpModel::StateCheckpoint cp;
        bool saved = false;

        const auto t0 = std::chrono::steady_clock::now();
        for (int p = 0; p < pp_test; p += 512) {
            int n = std::min(512, pp_test - p);
            if (!saved && p < half && p + n >= half) n = half - p;
            std::vector<int32_t> c_tok(long_ids.begin() + p, long_ids.begin() + p + n);
            std::vector<int32_t> c_pos(n);
            std::iota(c_pos.begin(), c_pos.end(), p);
            model.forward(c_tok, c_pos, logits_b);
            if (!saved && p + n == half) {
                model.save_checkpoint(cp, half);
                saved = true;
            }
        }
        const double pp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("Batched prefill: %.2f ms (%.1f tok/s)\n", pp_ms, pp_test * 1000.0 / pp_ms);

        // Restore checkpoint at half and replay second half -> check bit-exactness!
        const auto tr0 = std::chrono::steady_clock::now();
        model.restore_checkpoint(cp);
        const double rest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();
        std::vector<float> logits_r;
        for (int p = half; p < pp_test; p += 512) {
            int n = std::min(512, pp_test - p);
            std::vector<int32_t> c_tok(long_ids.begin() + p, long_ids.begin() + p + n);
            std::vector<int32_t> c_pos(n);
            std::iota(c_pos.begin(), c_pos.end(), p);
            model.forward(c_tok, c_pos, logits_r);
        }
        float max_diff = 0.0f;
        for (size_t i = 0; i < logits_b.size(); ++i) max_diff = std::max(max_diff, std::fabs(logits_b[i] - logits_r[i]));
        std::printf("Checkpoint restore: %.2f ms | Max |batched - ckpt| = %.6f\n", rest_ms, max_diff);

        // Also test steady-state decode at pp_test context (exercises QSA sparse path when pp_test > 2048!)
        model.warmup_decode_graphs();
        int32_t cur = static_cast<int32_t>(std::max_element(logits_b.begin(), logits_b.end()) - logits_b.begin());
        int32_t pos = pp_test;
        cur = model.forward_decode_greedy(cur, pos++); // warmup
        const auto td0 = std::chrono::steady_clock::now();
        for (int s = 0; s < 32; ++s) cur = model.forward_decode_greedy(cur, pos++);
        const double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
        std::printf("Decode at ctx=%d (32 tok, graph=%s): %.2f ms (%.2f tok/s)\n",
                    pp_test, use_graphs ? "on" : "off", dec_ms, 32 * 1000.0 / dec_ms);
        return 0;
    }

    // Standard greedy parity + speed benchmark
    model.reset_state();
    std::vector<int32_t> pos(prompt_ids.size());
    std::iota(pos.begin(), pos.end(), 0);
    std::vector<float> logits;

    const auto tp0 = std::chrono::steady_clock::now();
    model.forward(prompt_ids, pos, logits);
    const double tp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();

    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("Prefill top-5:");
    for (int k = 0; k < 5; ++k) {
        std::printf("  [%d '%s']=%.4f", idx[k], tok.decode({idx[k]}).c_str(), logits[idx[k]]);
    }
    std::printf("\nPrefill (%zu tok): %.2f ms (%.1f tok/s)\n", prompt_ids.size(), tp_ms, prompt_ids.size() * 1000.0 / tp_ms);

    model.warmup_decode_graphs();
    std::vector<int32_t> generated;
    int32_t cur_tok = idx[0];
    generated.push_back(cur_tok);
    int32_t cur_pos = static_cast<int32_t>(prompt_ids.size());

    // First step warmup
    if (gen_tokens > 1) {
        cur_tok = model.forward_decode_greedy(cur_tok, cur_pos++);
        generated.push_back(cur_tok);
    }

    const int timed = std::max(0, gen_tokens - 2);
    const auto td0 = std::chrono::steady_clock::now();
    for (int s = 0; s < timed; ++s) {
        cur_tok = model.forward_decode_greedy(cur_tok, cur_pos++);
        generated.push_back(cur_tok);
    }
    const double td_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
    if (timed > 0) {
        std::printf("Qwen4Exp decode steady-state (%d tok, graph=%s): %.2f ms (%.2f tok/s)\n",
                    timed, use_graphs ? "on" : "off", td_ms, timed * 1000.0 / td_ms);
    }
    std::printf("Output: %s%s\n", prompt.c_str(), tok.decode(generated).c_str());
    return 0;
}
