#include "engram/engram_sidecar.h"
#include "gguf/reader.h"
#include "models/maxwell_moe.h"
#include "tokenizer/bpe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int> parse_csv_ints(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

static std::vector<float> parse_csv_floats(const std::string & s) {
    std::vector<float> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stof(item));
    }
    return out;
}

static std::string read_text_file(const std::string & path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open text file: " + path);
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

struct EvalMetrics {
    double nll   = 0.0;
    double ppl   = 0.0;
    double top1  = 0.0;
    double top5  = 0.0;
    int    count = 0;
    double ms    = 0.0;
};

static EvalMetrics eval_token_ppl(
    rookery::MaxwellMoeModel & model,
    rookery::EngramSidecar * sidecar,
    const std::vector<int32_t> & ids) {

    EvalMetrics m;
    if (ids.size() < 2) return m;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits;

    // Step 0: prime position 0
    if (sidecar) sidecar->prepare_step({ids[0]}, 0);
    model.forward_decode_logits(ids[0], 0, logits);

    double sum_nll = 0.0;
    int hit1 = 0, hit5 = 0;
    const int n_eval = static_cast<int>(ids.size()) - 1;

    for (int i = 0; i < n_eval; ++i) {
        const int32_t target = ids[i + 1];
        const float   tl     = logits[target];

        // Exact log-softmax(logits)[target] + top-1/top-5 rank
        float max_l = logits[0];
        for (float v : logits) if (v > max_l) max_l = v;
        double sum_exp = 0.0;
        int better = 0;
        for (float v : logits) {
            sum_exp += std::exp(static_cast<double>(v - max_l));
            if (v > tl) ++better;
        }
        const double log_p = static_cast<double>(tl - max_l) - std::log(sum_exp);
        sum_nll -= log_p;
        if (better == 0) ++hit1;
        if (better < 5)  ++hit5;

        if (i + 1 < n_eval) {
            if (sidecar) sidecar->prepare_step({target}, i + 1);
            model.forward_decode_logits(target, i + 1, logits);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    m.count = n_eval;
    m.nll   = sum_nll / n_eval;
    m.ppl   = std::exp(m.nll);
    m.top1  = 100.0 * hit1 / n_eval;
    m.top5  = 100.0 * hit5 / n_eval;
    m.ms    = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return m;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string engram_path;
    std::string adapter_path;
    std::string dump_path;
    std::string corpus_path;
    std::string eval_path;
    std::vector<int>   gpus = {0, 1};
    std::vector<float> split = {24.0f, 32.0f};
    std::vector<int>   engram_layers;
    std::vector<float> scale_sweep;
    float    scale        = 0.0f;
    int      n_gen        = 64;
    int      max_toks     = 20000;
    int      eval_toks    = 768;
    uint64_t shuffle_salt = 0;
    bool     table_only   = false;
    bool     dump_delta   = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m" || a == "--model")            model_path = next();
        else if (a == "--engram")                   engram_path = next();
        else if (a == "--engram-adapter")           adapter_path = next();
        else if (a == "--engram-scale")             scale = std::strtof(next().c_str(), nullptr);
        else if (a == "--scale-sweep")              scale_sweep = parse_csv_floats(next());
        else if (a == "--engram-layers")            engram_layers = parse_csv_ints(next());
        else if (a == "--dump-pairs")               dump_path = next();
        else if (a == "--dump-delta")               dump_delta = true;
        else if (a == "--corpus")                   corpus_path = next();
        else if (a == "--eval-ppl")                 eval_path = next();
        else if (a == "--max-tokens")               max_toks = std::atoi(next().c_str());
        else if (a == "--eval-tokens")              eval_toks = std::atoi(next().c_str());
        else if (a == "--shuffle-salt")             shuffle_salt = std::strtoull(next().c_str(), nullptr, 0);
        else if (a == "--gpus")                     gpus = parse_csv_ints(next());
        else if (a == "--split")                    split = parse_csv_floats(next());
        else if (a == "--gen")                      n_gen = std::atoi(next().c_str());
        else if (a == "--table-only")               table_only = true;
    }

    if (engram_path.empty()) {
        std::fprintf(stderr, "Usage: %s --engram <table.gguf> [-m <host.gguf>] [--engram-adapter <a.rkad>] [--engram-scale 0.0] [--scale-sweep 0,0.25,0.5,1] [--dump-pairs <p.rkpr>] [--dump-delta] [--eval-ppl <txt>] [--shuffle-salt 0] [--table-only]\n", argv[0]);
        return 1;
    }

    // 1. Fast standalone EngramTable lookup & latency test
    if (table_only || model_path.empty()) {
        rookery::EngramTable tbl;
        const auto t0 = std::chrono::steady_clock::now();
        tbl.open(engram_path);
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("Table mmap open: %.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

        std::string sample = "The capital of France is Paris, and Rayleigh scattering makes the daytime sky appear blue.";
        auto toks = tbl.has_tokenizer() ? tbl.tokenizer().encode(sample, true, false) : std::vector<int32_t>{1, 100, 200, 300, 400, 500};
        std::printf("Tokenized sample with Engram source tokenizer: %zu tokens\n", toks.size());

        std::vector<int32_t> row_ids;
        std::vector<float>   valid;
        std::vector<float>   buf(toks.size() * tbl.cfg().engram_dim());

        const auto g0 = std::chrono::steady_clock::now();
        for (int e = 0; e < tbl.cfg().n_layers; ++e) {
            tbl.hash_tokens(e, toks, 0, static_cast<int>(toks.size()), row_ids, valid);
            tbl.prefetch_rows(row_ids.data(), static_cast<int>(row_ids.size()));
            tbl.gather_dequant_f32(row_ids.data(), valid.data(), static_cast<int>(toks.size()), buf.data());
        }
        const auto g1 = std::chrono::steady_clock::now();
        double cold_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

        const auto w0 = std::chrono::steady_clock::now();
        for (int e = 0; e < tbl.cfg().n_layers; ++e) {
            tbl.hash_tokens(e, toks, 0, static_cast<int>(toks.size()), row_ids, valid);
            tbl.gather_dequant_f32(row_ids.data(), valid.data(), static_cast<int>(toks.size()), buf.data());
        }
        const auto w1 = std::chrono::steady_clock::now();
        double warm_ms = std::chrono::duration<double, std::milli>(w1 - w0).count();

        double l2 = 0.0;
        for (float v : buf) l2 += static_cast<double>(v) * v;
        l2 = std::sqrt(l2 / std::max<size_t>(1, buf.size()));
        std::printf("Lookup %d layers x %zu toks: cold=%.3f ms, warm=%.3f ms (%.2f us/tok), RMS=%.6f\n",
                    tbl.cfg().n_layers, toks.size(), cold_ms, warm_ms, (warm_ms * 1000.0) / toks.size(), l2);
        return 0;
    }

    // 2. Load host model + tokenizer + sidecar
    rookery::BpeTokenizer host_tok;
    {
        auto gguf = rookery::GgufFile::open(model_path);
        host_tok.load(gguf);
    }

    const int ctx_need = std::max(8192, std::max(max_toks, eval_toks) + 128);
    rookery::MaxwellMoeModel model;
    model.load(model_path, gpus, split, ctx_need, true, false);

    rookery::EngramSidecar sidecar;
    sidecar.init(engram_path, &host_tok, model.layer_devices(),
                 model.config().n_embd, model.config().hc_mult,
                 512, adapter_path, scale, engram_layers);
    if (shuffle_salt != 0) {
        sidecar.set_shuffle_salt(shuffle_salt);
        std::printf("CONTROL MODE: Engram hash shuffle_salt = 0x%llx\n", static_cast<unsigned long long>(shuffle_salt));
    }

    // Mode A: Calibration pair dumping (state H_L OR delta H_{L+1} - H_L)
    if (!dump_path.empty()) {
        std::string text = corpus_path.empty() ? "The capital of France is Paris." : read_text_file(corpus_path);
        auto ids = host_tok.encode(text, true, false);
        if (static_cast<int>(ids.size()) > max_toks) ids.resize(max_toks);
        std::printf("Calibration dump corpus: %zu host tokens (mode=%s)\n",
                    ids.size(), dump_delta ? "delta" : "state");

        sidecar.set_global_scale(0.0f);
        sidecar.enable_calibration_dump(dump_path, dump_delta);
        model.set_layer_hook(&sidecar);

        std::vector<int32_t> pos(ids.size());
        std::iota(pos.begin(), pos.end(), 0);
        std::vector<float> logits;
        const size_t chunk = 256;
        for (size_t off = 0; off < ids.size(); off += chunk) {
            size_t len = std::min(chunk, ids.size() - off);
            std::vector<int32_t> pt(ids.begin() + off, ids.begin() + off + len);
            std::vector<int32_t> pp(pos.begin() + off, pos.begin() + off + len);
            sidecar.prepare_step(pt, static_cast<int>(off));
            model.forward(pt, pp, logits);
        }
        sidecar.close_calibration_dump();
        return 0;
    }

    // Mode B: Held-Out Perplexity & Top-1/Top-5 Accuracy Sweep (+ Shuffled-Table Control!)
    if (!eval_path.empty()) {
        std::string text = read_text_file(eval_path);
        auto ids = host_tok.encode(text, true, false);
        if (static_cast<int>(ids.size()) > eval_toks + 1) ids.resize(eval_toks + 1);
        std::printf("Held-out eval file: %s (%zu tokens evaluated)\n", eval_path.c_str(), ids.size() - 1);

        if (scale_sweep.empty()) scale_sweep = {0.0f, 0.25f, 0.5f, 1.0f, 2.0f};

        for (float s : scale_sweep) {
            sidecar.set_global_scale(s);
            sidecar.set_shuffle_salt(0);
            model.set_layer_hook(s == 0.0f ? nullptr : &sidecar);
            model.warmup_decode_graphs();
            auto res = eval_token_ppl(model, s == 0.0f ? nullptr : &sidecar, ids);
            std::printf("  [REAL    ] global_scale=%6.3f | NLL=%.5f | PPL=%7.3f | top1=%5.2f%% | top5=%5.2f%% (%.1f tok/s)\n",
                        s, res.nll, res.ppl, res.top1, res.top5, res.count * 1000.0 / res.ms);

            if (s > 0.0f && shuffle_salt != 0) {
                sidecar.set_shuffle_salt(shuffle_salt);
                auto ctl = eval_token_ppl(model, &sidecar, ids);
                std::printf("  [SHUFFLED] global_scale=%6.3f | NLL=%.5f | PPL=%7.3f | top1=%5.2f%% | top5=%5.2f%% (delta_PPL vs real=%+.4f)\n",
                            s, ctl.nll, ctl.ppl, ctl.top1, ctl.top5, ctl.ppl - res.ppl);
            }
        }
        return 0;
    }

    // Mode C: Quick prefill + decode smoke test
    const std::string prompt_str = "The capital of France is";
    const auto prompt_ids = host_tok.encode(prompt_str, true, true);
    std::vector<int32_t> pos(prompt_ids.size());
    std::iota(pos.begin(), pos.end(), 0);

    std::vector<float> base_logits;
    model.set_layer_hook(nullptr);
    model.forward(prompt_ids, pos, base_logits);

    model.set_layer_hook(&sidecar);
    model.warmup_decode_graphs();

    std::vector<float> eng_logits;
    sidecar.prepare_step(prompt_ids, 0);
    model.forward(prompt_ids, pos, eng_logits);

    float max_diff = 0.0f;
    for (size_t i = 0; i < base_logits.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(base_logits[i] - eng_logits[i]));
    }
    int32_t base_top = static_cast<int32_t>(std::max_element(base_logits.begin(), base_logits.end()) - base_logits.begin());
    int32_t eng_top  = static_cast<int32_t>(std::max_element(eng_logits.begin(),  eng_logits.end())  - eng_logits.begin());
    std::printf("Prefill logit delta (scale=%.4f): max_abs_diff=%.6f, base_top=%d ('%s'), eng_top=%d ('%s')\n",
                scale, max_diff, base_top, host_tok.decode_token(base_top).c_str(),
                eng_top, host_tok.decode_token(eng_top).c_str());

    int32_t cur_tok = eng_top;
    int32_t cur_pos = static_cast<int32_t>(prompt_ids.size());
    std::vector<int32_t> gen = {cur_tok};

    const auto t0 = std::chrono::steady_clock::now();
    for (int step = 1; step < n_gen; ++step) {
        sidecar.prepare_step({cur_tok}, cur_pos);
        cur_tok = model.forward_decode_greedy(cur_tok, cur_pos++);
        gen.push_back(cur_tok);
    }
    const auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Grafted decode (CUDA Graph, %d tok, scale=%.4f): %.2f ms (%.2f tok/s)\n",
                n_gen - 1, scale, ms, (n_gen - 1) * 1000.0 / ms);
    std::printf("Output: %s%s\n", prompt_str.c_str(), host_tok.decode(gen, false).c_str());
    return 0;
}
