#include "gguf/sharded.h"
#include "models/sensenova_loop.h"
#include "server/chat_template.h"
#include "tokenizer/bpe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::stoi(tok));
    return out;
}

static std::vector<float> parse_floats(const std::string & s) {
    std::vector<float> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::stof(tok));
    return out;
}

static int argmax_vec(const std::vector<float> & v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::vector<int> gpus = {0};
    std::vector<float> split;
    int max_ctx = 131072;
    int gen_n = 64;
    int pp_test = 0;
    bool kv_q8 = true;
    bool use_mtp = false;
    bool use_graph = true;
    bool chat = false;
    bool no_think = false;
    std::string prompt = "The capital of France is";
    std::vector<int> bench_pp;
    std::vector<int> bench_d;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc)               model_path = argv[++i];
        else if (a == "--gpus" && i + 1 < argc)      gpus       = parse_ints(argv[++i]);
        else if (a == "--split" && i + 1 < argc)     split      = parse_floats(argv[++i]);
        else if (a == "-c" && i + 1 < argc)          max_ctx    = std::stoi(argv[++i]);
        else if (a == "--gen" && i + 1 < argc)       gen_n      = std::stoi(argv[++i]);
        else if (a == "--pp-test" && i + 1 < argc)   pp_test    = std::stoi(argv[++i]);
        else if (a == "--kv-f16")                    kv_q8      = false;
        else if (a == "--mtp")                       use_mtp    = true;
        else if (a == "--no-graph")                  use_graph  = false;
        else if (a == "--chat")                      chat       = true;
        else if (a == "--no-think")                  no_think   = true;
        else if (a == "--prompt" && i + 1 < argc)    prompt     = argv[++i];
        else if (a == "--bench-pp" && i + 1 < argc)  bench_pp   = parse_ints(argv[++i]);
        else if (a == "--bench-d" && i + 1 < argc)   bench_d    = parse_ints(argv[++i]);
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "Usage: %s -m <sensenova.gguf> [--gpus 0] [--split ...] [-c 131072] [--mtp] [--pp-test 64] [--bench-pp 512,2048] [--bench-d 0,4096] [--gen 64] [--chat] [--no-think]\n", argv[0]);
        return 1;
    }

    rookery::BpeTokenizer tok;
    {
        auto sg = rookery::ShardedGguf::open(model_path);
        tok.load(sg.meta());
        tok.add_stop_token("<|im_end|>");
        tok.add_stop_token("<|endoftext|>");
    }

    rookery::SenseNovaLoopModel model;
    model.load(model_path, gpus, split, max_ctx, kv_q8, true);
    model.set_use_cuda_graphs(use_graph);
    if (use_graph) model.warmup_decode_graphs();

    // 1. Prefill-vs-sequential + checkpoint parity test
    if (pp_test > 0) {
        std::vector<int32_t> ids(pp_test), pos(pp_test);
        for (int i = 0; i < pp_test; ++i) { ids[i] = 1000 + (i * 131) % 30000; pos[i] = i; }

        model.reset_state();
        std::vector<float> lg_batch;
        model.forward(ids, pos, lg_batch);

        // Split + checkpoint restore parity
        const int mid = pp_test / 2;
        model.reset_state();
        std::vector<float> tmp;
        model.forward(std::vector<int32_t>(ids.begin(), ids.begin() + mid),
                      std::vector<int32_t>(pos.begin(), pos.begin() + mid), tmp, false);
        rookery::SenseNovaLoopModel::GdnCheckpoint cp;
        model.save_gdn_checkpoint(cp, mid);
        model.reset_state();
        model.restore_gdn_checkpoint(cp);
        std::vector<float> lg_ckpt;
        model.forward(std::vector<int32_t>(ids.begin() + mid, ids.end()),
                      std::vector<int32_t>(pos.begin() + mid, pos.end()), lg_ckpt);

        // Sequential T=1 parity (if pp_test <= 32)
        float max_diff_ckpt = 0.0f;
        for (size_t i = 0; i < lg_batch.size(); ++i) {
            max_diff_ckpt = std::max(max_diff_ckpt, std::fabs(lg_batch[i] - lg_ckpt[i]));
        }
        int tb = argmax_vec(lg_batch), tc = argmax_vec(lg_ckpt);
        std::printf("[PARITY T=%d] top_batch=%d top_ckpt=%d (match=%d) max_diff_ckpt=%.6f\n",
                    pp_test, tb, tc, tb == tc ? 1 : 0, max_diff_ckpt);
    }

    // 2. Benchmarks
    for (int plen : bench_pp) {
        model.reset_state();
        std::vector<int32_t> ids(plen), pos(plen);
        for (int i = 0; i < plen; ++i) { ids[i] = 500 + (i * 97) % 30000; pos[i] = i; }
        std::vector<float> lg;
        const auto t0 = std::chrono::steady_clock::now();
        for (int off = 0; off < plen; off += 512) {
            const int n = std::min(512, plen - off);
            model.forward(std::vector<int32_t>(ids.begin() + off, ids.begin() + off + n),
                          std::vector<int32_t>(pos.begin() + off, pos.begin() + off + n),
                          lg, off + n == plen);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("[BENCH] pp%-4d : %8.2f ms | %8.2f tok/s\n", plen, ms, plen * 1000.0 / ms);
    }

    for (int dlen : bench_d) {
        model.reset_state();
        std::vector<float> lg;
        int cur_pos = 0;
        int32_t cur_tok = 785;
        if (dlen > 0) {
            std::vector<int32_t> ids(dlen), pos(dlen);
            for (int i = 0; i < dlen; ++i) { ids[i] = 500 + (i * 97) % 30000; pos[i] = i; }
            for (int off = 0; off < dlen; off += 512) {
                const int n = std::min(512, dlen - off);
                model.forward(std::vector<int32_t>(ids.begin() + off, ids.begin() + off + n),
                              std::vector<int32_t>(pos.begin() + off, pos.begin() + off + n),
                              lg, off + n == dlen);
            }
            cur_tok = argmax_vec(lg);
            cur_pos = dlen;
        } else {
            model.forward({cur_tok}, {0}, lg);
            cur_tok = argmax_vec(lg);
            cur_pos = 1;
        }

        if (!use_mtp) {
            // Warmup 2 steps
            for (int w = 0; w < 2; ++w) cur_tok = model.forward_decode_greedy(cur_tok, cur_pos++);
            const auto t0 = std::chrono::steady_clock::now();
            for (int g = 0; g < gen_n; ++g) {
                cur_tok = model.forward_decode_greedy(cur_tok, cur_pos++);
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("[BENCH] tg%-3d @ d=%-5d (no-MTP): %8.2f ms | %8.2f tok/s\n",
                        gen_n, dlen, ms, gen_n * 1000.0 / ms);
        } else {
            int32_t draft = model.mtp_prefill_chunk({cur_tok}, {cur_pos - 1});
            rookery::MtpStats st{};
            std::vector<int32_t> out;
            int produced = 0;
            const auto t0 = std::chrono::steady_clock::now();
            while (produced < gen_n) {
                model.decode_mtp_greedy_step(cur_tok, draft, cur_pos, out, st);
                produced += static_cast<int>(out.size());
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("[BENCH] tg%-3d @ d=%-5d (MTP   ): %8.2f ms | %8.2f tok/s (acc=%d/%d = %.1f%%)\n",
                        produced, dlen, ms, produced * 1000.0 / ms,
                        st.accepted, st.steps, st.steps ? (100.0 * st.accepted / st.steps) : 0.0);
        }
    }

    // 3. Interactive / prompt generation test
    if (bench_pp.empty() && bench_d.empty()) {
        std::string full_prompt = prompt;
        if (chat) {
            nlohmann::json msgs = nlohmann::json::array({{{"role", "user"}, {"content", prompt}}});
            full_prompt = rookery::render_qwen3_chat_prompt(msgs, nlohmann::json::array(), !no_think);
        }
        auto ids = tok.encode(full_prompt, tok.add_bos_default(), true);
        std::printf("Prompt (%zu toks): %s\n", ids.size(), full_prompt.c_str());

        model.reset_state();
        std::vector<float> lg;
        const auto t_pp0 = std::chrono::steady_clock::now();
        std::vector<int32_t> pos(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) pos[i] = static_cast<int32_t>(i);
        model.forward(ids, pos, lg);
        const double pp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pp0).count();

        int32_t cur_tok = argmax_vec(lg);
        int32_t cur_pos = static_cast<int32_t>(ids.size());
        std::vector<int32_t> gen_ids = {cur_tok};
        rookery::MtpStats st{};

        const auto t_tg0 = std::chrono::steady_clock::now();
        if (!use_mtp) {
            for (int g = 1; g < gen_n && !tok.is_stop_token(cur_tok); ++g) {
                cur_tok = model.forward_decode_greedy(cur_tok, cur_pos++);
                if (tok.is_stop_token(cur_tok)) break;
                gen_ids.push_back(cur_tok);
            }
        } else {
            // MTP prefill over prompt (next_tokens = [ids[1..], cur_tok])
            std::vector<int32_t> next_t(ids.begin() + 1, ids.end());
            next_t.push_back(cur_tok);
            int32_t draft = model.mtp_prefill_chunk(next_t, pos);
            std::vector<int32_t> out;
            while (static_cast<int>(gen_ids.size()) < gen_n && !tok.is_stop_token(cur_tok)) {
                model.decode_mtp_greedy_step(cur_tok, draft, cur_pos, out, st);
                for (int32_t tid : out) {
                    if (tok.is_stop_token(tid)) { cur_tok = tid; break; }
                    gen_ids.push_back(tid);
                    if (static_cast<int>(gen_ids.size()) >= gen_n) break;
                }
            }
        }
        const double tg_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_tg0).count();

        std::printf("--- OUTPUT ---\n%s\n--------------\n", tok.decode(gen_ids, false).c_str());
        if (use_mtp) {
            std::printf("Prefill: %zu toks in %.2f ms (%.1f tok/s) | Decode: %zu toks in %.2f ms (%.2f tok/s, MTP acc=%d/%d = %.1f%%)\n",
                        ids.size(), pp_ms, ids.size() * 1000.0 / pp_ms,
                        gen_ids.size(), tg_ms, gen_ids.size() * 1000.0 / tg_ms,
                        st.accepted, st.steps, st.steps ? (100.0 * st.accepted / st.steps) : 0.0);
        } else {
            std::printf("Prefill: %zu toks in %.2f ms (%.1f tok/s) | Decode: %zu toks in %.2f ms (%.2f tok/s)\n",
                        ids.size(), pp_ms, ids.size() * 1000.0 / pp_ms,
                        gen_ids.size(), tg_ms, gen_ids.size() * 1000.0 / tg_ms);
        }
    }

    return 0;
}
