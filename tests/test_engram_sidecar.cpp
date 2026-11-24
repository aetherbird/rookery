#include "engram/engram_sidecar.h"
#include "gguf/reader.h"
#include "models/maxwell_moe.h"
#include "models/qwen35_moe.h"
#include "tokenizer/bpe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

struct HostEvalModel {
    std::string arch;
    std::unique_ptr<rookery::MaxwellMoeModel> maxwell;
    std::unique_ptr<rookery::Qwen35MoeModel>  qwen35;

    void load(const std::string & path, const std::vector<int> & gpus, const std::vector<float> & split, int max_ctx) {
        auto gguf = rookery::GgufFile::open(path);
        arch = gguf.get_str("general.architecture", "maxwell-moe");
        if (arch == "qwen35moe") {
            qwen35 = std::make_unique<rookery::Qwen35MoeModel>();
            qwen35->load(path, gpus, split, max_ctx, true, false);
        } else {
            maxwell = std::make_unique<rookery::MaxwellMoeModel>();
            maxwell->load(path, gpus, split, max_ctx, true, false);
        }
    }
    int n_embd() const { return qwen35 ? qwen35->config().n_embd : maxwell->config().n_embd; }
    int hc_mult() const { return qwen35 ? 1 : maxwell->config().hc_mult; }
    std::vector<int> layer_devices() const { return qwen35 ? qwen35->layer_devices() : maxwell->layer_devices(); }
    void set_layer_hook(rookery::LayerHook * h) { if (qwen35) qwen35->set_layer_hook(h); else maxwell->set_layer_hook(h); }
    void set_use_cuda_graphs(bool en) { if (qwen35) qwen35->set_use_cuda_graphs(en); else maxwell->set_use_cuda_graphs(en); }
    void warmup_decode_graphs() { if (qwen35) qwen35->warmup_decode_graphs(); else maxwell->warmup_decode_graphs(); }
    void reset_state() { if (qwen35) qwen35->reset_state(); }
    void forward(const std::vector<int32_t> & t, const std::vector<int32_t> & p, std::vector<float> & l) {
        if (qwen35) qwen35->forward(t, p, l); else maxwell->forward(t, p, l);
    }
    int32_t forward_decode_greedy(int32_t t, int32_t p) {
        return qwen35 ? qwen35->forward_decode_greedy(t, p) : maxwell->forward_decode_greedy(t, p);
    }
    void forward_decode_logits(int32_t t, int32_t p, std::vector<float> & l) {
        if (qwen35) qwen35->forward_decode_logits(t, p, l); else maxwell->forward_decode_logits(t, p, l);
    }
};

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
    HostEvalModel & model,
    rookery::EngramSidecar * sidecar,
    const std::vector<int32_t> & ids) {

    EvalMetrics m;
    if (ids.size() < 2) return m;

    model.reset_state();
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
    std::string dump_ce_path;
    std::string corpus_path;
    std::string eval_path;
    std::string factual_path;
    std::vector<int>   gpus = {0, 1};
    std::vector<float> split = {24.0f, 32.0f};
    std::vector<int>   engram_layers;
    std::vector<float> scale_sweep;
    float    scale        = 0.0f;
    int      n_gen        = 64;
    int      max_toks     = 20000;
    int      eval_toks    = 768;
    int      ce_layer     = 44;
    int      doc_ctx      = 4096;
    uint64_t shuffle_salt = 0;
    bool     table_only   = false;
    bool     dump_delta   = false;
    bool     verify_addr  = false;
    bool     bench_mode   = false;

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
        else if (a == "--dump-tail-ce")             dump_ce_path = next();
        else if (a == "--ce-layer")                 ce_layer = std::atoi(next().c_str());
        else if (a == "--doc-ctx")                  doc_ctx = std::atoi(next().c_str());
        else if (a == "--dump-delta")               dump_delta = true;
        else if (a == "--corpus")                   corpus_path = next();
        else if (a == "--eval-ppl")                 eval_path = next();
        else if (a == "--eval-factual")             factual_path = next();
        else if (a == "--max-tokens")               max_toks = std::atoi(next().c_str());
        else if (a == "--eval-tokens")              eval_toks = std::atoi(next().c_str());
        else if (a == "--shuffle-salt")             shuffle_salt = std::strtoull(next().c_str(), nullptr, 0);
        else if (a == "--gpus")                     gpus = parse_csv_ints(next());
        else if (a == "--split")                    split = parse_csv_floats(next());
        else if (a == "--gen")                      n_gen = std::atoi(next().c_str());
        else if (a == "--table-only")               table_only = true;
        else if (a == "--verify-addressing")        verify_addr = true;
        else if (a == "--bench")                    bench_mode = true;
    }

    if (engram_path.empty()) {
        std::fprintf(stderr, "Usage: %s --engram <table.gguf> [-m <host.gguf>] [--engram-adapter <a.rkad>] [--engram-scale 0.0] [--scale-sweep 0,0.25,0.5,1] [--dump-pairs <p.rkpr>] [--dump-delta] [--eval-ppl <txt>] [--shuffle-salt 0] [--verify-addressing] [--bench] [--table-only]\n", argv[0]);
        return 1;
    }

    // 0. Fast exact incremental-vs-full source addressing verification (no host GPU weights needed!)
    if (verify_addr) {
        if (model_path.empty()) {
            std::fprintf(stderr, "--verify-addressing requires -m <host.gguf> to load the host tokenizer\n");
            return 1;
        }
        rookery::BpeTokenizer host_tok;
        {
            auto gguf = rookery::GgufFile::open(model_path);
            host_tok.load(gguf);
        }
        rookery::EngramSidecar sidecar;
        std::vector<int> dummy_devs(48, 0);
        sidecar.init(engram_path, &host_tok, dummy_devs, 2048, 4, 512, "", 0.0f, {});

        std::string sample;
        if (!corpus_path.empty()) {
            sample = read_text_file(corpus_path);
        } else {
            sample =
                "The capital of France is Paris, and Rayleigh scattering makes the daytime sky appear blue. "
                "Electromagnetic radiation spans wavelengths from gamma rays (1e-12 m) to long radio waves.\n"
                "In C++20, `std::vector<int32_t> aligned_src_pos(T, 0);` avoids dynamic reallocations.\n"
                "JSON payload: {\"model\": \"maxwell-v1\", \"temperature\": 0.7, \"stop\": [\"\\n\\n\", \"<|im_end|>\"]}\n"
                "Multi-byte UTF-8 test — em-dash, curly quote \xE2\x80\x99, CJK \xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C, "
                "rocket \xF0\x9F\x9A\x80, and URL https://example.org/v2/items?id=42&sort=desc.\n"
                "Repeat to exceed 512 bytes so boundary-aware suffix slicing triggers across every single decode step: "
                "1234567890 9876543210 supercalifragilisticexpialidocious internationalization electroencephalography. "
                "The Treaty of Westphalia (1648) ended the Thirty Years' War in Europe. "
                "Marie Curie won Nobel Prizes in Physics (1903) and Chemistry (1911) for her pioneering research on radioactivity.\n"
                "def fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n";
            // Repeat 3x with slight variation to test deep (>1500 byte) rolling context
            sample = sample + "\n" + sample + "\n" + sample;
        }

        auto host_ids = host_tok.encode(sample, false, false);
        if (static_cast<int>(host_ids.size()) > max_toks) host_ids.resize(max_toks);

        int tok_chk = 0, tok_ok = 0, hash_ok = 0, hash_tot = 0;
        const auto t0 = std::chrono::steady_clock::now();
        bool pass = sidecar.verify_incremental_addressing(host_ids, tok_chk, tok_ok, hash_ok, hash_tot, true);
        const auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::printf("\n=== INCREMENTAL ADDRESSING PARITY TEST ===\n");
        std::printf("Checked %d host tokens (%zu bytes) in %.1f ms\n", tok_chk, sample.size(), ms);
        std::printf("Exact source token match: %d / %d (%.2f%%)\n", tok_ok, tok_chk, 100.0 * tok_ok / std::max(1, tok_chk));
        std::printf("Exact N-gram hash match:  %d / %d (%.2f%%)\n", hash_ok, hash_tot, 100.0 * hash_ok / std::max(1, hash_tot));
        std::printf("Result: %s\n", pass ? "PASS (100% exact)" : "FAIL");
        return pass ? 0 : 1;
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

    const int ctx_need = dump_ce_path.empty()
        ? std::max(8192, std::max(max_toks, eval_toks) + 128)
        : std::max(4096, doc_ctx + 128);
    HostEvalModel model;
    model.load(model_path, gpus, split, ctx_need);

    rookery::EngramSidecar sidecar;
    sidecar.init(engram_path, &host_tok, model.layer_devices(),
                 model.n_embd(), model.hc_mult(),
                 512, adapter_path, scale, engram_layers, model.arch);
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

        model.reset_state();
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

    // Mode A2: Tail-CE Supervised Bridge Dataset Dump ("RKCE" v1, strictly causal addressing, doc-windowed!)
    if (!dump_ce_path.empty()) {
        std::string text = corpus_path.empty() ? "The capital of France is Paris." : read_text_file(corpus_path);
        auto ids = host_tok.encode(text, true, false);
        if (static_cast<int>(ids.size()) > max_toks) ids.resize(max_toks);
        std::printf("Tail-CE dump corpus: %zu host tokens (layer=%d, doc_ctx=%d)\n", ids.size(), ce_layer, doc_ctx);

        sidecar.set_global_scale(0.0f);
        sidecar.enable_tail_ce_dump(dump_ce_path, ce_layer);
        model.set_layer_hook(&sidecar);

        std::vector<float> logits;
        const size_t chunk = 256;
        // Process in non-overlapping windows of up to doc_ctx tokens (resets position to 0 each window to keep KV small)
        for (size_t win_start = 0; win_start + 1 < ids.size(); win_start += static_cast<size_t>(doc_ctx)) {
            model.reset_state();
            const size_t win_end = std::min(ids.size(), win_start + static_cast<size_t>(doc_ctx));
            const size_t win_len = win_end - win_start;
            for (size_t off = 0; off < win_len; off += chunk) {
                const size_t len = std::min(chunk, win_len - off);
                std::vector<int32_t> pt(ids.begin() + win_start + off, ids.begin() + win_start + off + len);
                std::vector<int32_t> pp(len);
                std::iota(pp.begin(), pp.end(), static_cast<int32_t>(off));
                std::vector<int32_t> tgts(len, -1);
                for (size_t k = 0; k < len; ++k) {
                    if (off + k + 1 < win_len) tgts[k] = ids[win_start + off + k + 1];
                }
                sidecar.set_tail_ce_targets(tgts, pt, static_cast<int>(off));
                model.forward(pt, pp, logits);
            }
        }
        sidecar.close_tail_ce_dump();
        return 0;
    }

    // Mode B3: Held-Out Factual QA Probe (file format: one "prompt\tanswer" per line; 4-way Base/Disabled/Real/Shuffled!)
    if (!factual_path.empty()) {
        std::ifstream ifs(factual_path);
        if (!ifs) throw std::runtime_error("Cannot open factual QA file: " + factual_path);
        std::vector<std::pair<std::string, std::string>> qa_pairs;
        std::string line;
        while (std::getline(ifs, line)) {
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::string q = line.substr(0, tab);
            std::string a = line.substr(tab + 1);
            if (!q.empty() && !a.empty()) qa_pairs.emplace_back(std::move(q), std::move(a));
        }
        std::printf("Held-out factual QA file: %s (%zu items)\n", factual_path.c_str(), qa_pairs.size());

        auto eval_factual_run = [&](rookery::EngramSidecar * sc) {
            double sum_nll = 0.0;
            int total_ans_toks = 0, hit1 = 0, hit5 = 0, exact_first1 = 0, exact_first5 = 0;
            std::vector<float> logits;
            for (const auto & [q, a] : qa_pairs) {
                model.reset_state();
                auto q_ids = host_tok.encode(q, true, false);
                auto a_ids = host_tok.encode(a, false, false);
                if (q_ids.empty() || a_ids.empty()) continue;

                // Prefill prompt causally
                if (sc) {
                    for (size_t i = 0; i < q_ids.size(); ++i) {
                        sc->prepare_step({q_ids[i]}, static_cast<int>(i));
                        model.forward_decode_logits(q_ids[i], static_cast<int>(i), logits);
                    }
                } else {
                    for (size_t i = 0; i < q_ids.size(); ++i) {
                        model.forward_decode_logits(q_ids[i], static_cast<int>(i), logits);
                    }
                }

                // Score answer tokens
                for (size_t k = 0; k < a_ids.size(); ++k) {
                    const int32_t target = a_ids[k];
                    const float tl = logits[target];
                    float max_l = logits[0];
                    for (float v : logits) if (v > max_l) max_l = v;
                    double sum_exp = 0.0;
                    int better = 0;
                    for (float v : logits) {
                        sum_exp += std::exp(static_cast<double>(v - max_l));
                        if (v > tl) ++better;
                    }
                    sum_nll -= static_cast<double>(tl - max_l) - std::log(sum_exp);
                    ++total_ans_toks;
                    if (better == 0) ++hit1;
                    if (better < 5)  ++hit5;
                    if (k == 0) {
                        if (better == 0) ++exact_first1;
                        if (better < 5)  ++exact_first5;
                    }
                    if (k + 1 < a_ids.size()) {
                        int pos = static_cast<int>(q_ids.size() + k);
                        if (sc) sc->prepare_step({target}, pos);
                        model.forward_decode_logits(target, pos, logits);
                    }
                }
            }
            struct FRes { double nll, ppl, tok1, tok5, q1, q5; int n_tok, n_q; };
            double nll = sum_nll / std::max(1, total_ans_toks);
            return FRes{nll, std::exp(nll),
                        100.0 * hit1 / std::max(1, total_ans_toks),
                        100.0 * hit5 / std::max(1, total_ans_toks),
                        100.0 * exact_first1 / std::max<int>(1, static_cast<int>(qa_pairs.size())),
                        100.0 * exact_first5 / std::max<int>(1, static_cast<int>(qa_pairs.size())),
                        total_ans_toks, static_cast<int>(qa_pairs.size())};
        };

        if (scale_sweep.empty()) scale_sweep = {0.5f, 1.0f, 1.5f};

        sidecar.set_global_scale(0.0f);
        sidecar.set_shuffle_salt(0);
        model.set_layer_hook(nullptr);
        model.warmup_decode_graphs();
        auto base = eval_factual_run(nullptr);
        std::printf("  [BASE    ] (no hook)          | ans_NLL=%.5f | ans_PPL=%7.3f | first_top1=%5.1f%% | tok_top1=%5.1f%% | tok_top5=%5.1f%%\n",
                    base.nll, base.ppl, base.q1, base.tok1, base.tok5);

        model.set_layer_hook(&sidecar);
        model.warmup_decode_graphs();
        auto dis = eval_factual_run(&sidecar);
        std::printf("  [DISABLED] global_scale= 0.000 | ans_NLL=%.5f | ans_PPL=%7.3f | delta vs base=%+.6f\n",
                    dis.nll, dis.ppl, dis.ppl - base.ppl);

        for (float s : scale_sweep) {
            if (s <= 0.0f) continue;
            sidecar.set_global_scale(s);
            sidecar.set_shuffle_salt(0);
            model.set_layer_hook(&sidecar);
            model.warmup_decode_graphs();
            auto res = eval_factual_run(&sidecar);
            std::printf("  [REAL    ] global_scale=%6.3f | ans_NLL=%.5f | ans_PPL=%7.3f | first_top1=%5.1f%% | tok_top1=%5.1f%% | tok_top5=%5.1f%% (vs base=%+.4f)\n",
                        s, res.nll, res.ppl, res.q1, res.tok1, res.tok5, res.ppl - base.ppl);
            if (shuffle_salt != 0) {
                sidecar.set_shuffle_salt(shuffle_salt);
                auto ctl = eval_factual_run(&sidecar);
                std::printf("  [SHUFFLED] global_scale=%6.3f | ans_NLL=%.5f | ans_PPL=%7.3f | first_top1=%5.1f%% | tok_top1=%5.1f%% | tok_top5=%5.1f%% (shuf-real=%+.4f)\n",
                            s, ctl.nll, ctl.ppl, ctl.q1, ctl.tok1, ctl.tok5, ctl.ppl - res.ppl);
            }
        }
        return 0;
    }

    // Mode B: Held-Out Perplexity & Top-1/Top-5 Accuracy Sweep (4-way: Base, Disabled, Real, Shuffled!)
    if (!eval_path.empty()) {
        std::string text = read_text_file(eval_path);
        auto ids = host_tok.encode(text, true, false);
        if (static_cast<int>(ids.size()) > eval_toks + 1) ids.resize(eval_toks + 1);
        std::printf("Held-out eval file: %s (%zu tokens evaluated)\n", eval_path.c_str(), ids.size() - 1);

        if (scale_sweep.empty()) scale_sweep = {0.25f, 0.5f, 1.0f, 2.0f};

        // Control 1: Pure Base (no LayerHook, no prepare_step)
        sidecar.set_global_scale(0.0f);
        sidecar.set_shuffle_salt(0);
        model.set_layer_hook(nullptr);
        model.warmup_decode_graphs();
        auto base_res = eval_token_ppl(model, nullptr, ids);
        std::printf("  [BASE    ] (no hook)          | NLL=%.5f | PPL=%7.3f | top1=%5.2f%% | top5=%5.2f%% (%.1f tok/s)\n",
                    base_res.nll, base_res.ppl, base_res.top1, base_res.top5, base_res.count * 1000.0 / base_res.ms);

        // Control 2: Bridge-Disabled (hook attached & prepare_step runs, scale=0.0)
        sidecar.set_global_scale(0.0f);
        sidecar.set_shuffle_salt(0);
        model.set_layer_hook(&sidecar);
        model.warmup_decode_graphs();
        auto dis_res = eval_token_ppl(model, &sidecar, ids);
        std::printf("  [DISABLED] global_scale= 0.000 | NLL=%.5f | PPL=%7.3f | top1=%5.2f%% | top5=%5.2f%% (delta vs base=%+.6f)\n",
                    dis_res.nll, dis_res.ppl, dis_res.top1, dis_res.top5, dis_res.ppl - base_res.ppl);

        for (float s : scale_sweep) {
            if (s <= 0.0f) continue;
            sidecar.set_global_scale(s);
            sidecar.set_shuffle_salt(0);
            model.set_layer_hook(&sidecar);
            model.warmup_decode_graphs();
            auto res = eval_token_ppl(model, &sidecar, ids);
            std::printf("  [REAL    ] global_scale=%6.3f | NLL=%.5f | PPL=%7.3f | top1=%5.2f%% | top5=%5.2f%% (vs base=%+.4f, %.1f tok/s)\n",
                        s, res.nll, res.ppl, res.top1, res.top5, res.ppl - base_res.ppl, res.count * 1000.0 / res.ms);

            if (shuffle_salt != 0) {
                sidecar.set_shuffle_salt(shuffle_salt);
                auto ctl = eval_token_ppl(model, &sidecar, ids);
                std::printf("  [SHUFFLED] global_scale=%6.3f | NLL=%.5f | PPL=%7.3f | top1=%5.2f%% | top5=%5.2f%% (shuf-real=%+.4f)\n",
                            s, ctl.nll, ctl.ppl, ctl.top1, ctl.top5, ctl.ppl - res.ppl);
            }
        }
        return 0;
    }

    // Mode B2: Stage-by-Stage Latency & Throughput Profiling (Base vs Disabled vs Active, tg128 + pp1024)
    if (bench_mode) {
        const std::string prompt_str =
            "In computer architecture, memory-bandwidth-bound autoregressive decoding benefits from coalesced quantized "
            "loads, static CUDA graphs, and asynchronous host-to-device prefetching. The capital of France is";
        const auto prompt_ids = host_tok.encode(prompt_str, true, false);
        std::vector<int32_t> ppos(prompt_ids.size());
        std::iota(ppos.begin(), ppos.end(), 0);
        std::vector<float> logits;

        auto run_decode_bench = [&](const char * label, bool use_hook, bool call_prep, float sc, bool use_graph, bool gpu_prof) {
            model.reset_state();
            model.set_use_cuda_graphs(use_graph);
            sidecar.set_global_scale(sc);
            sidecar.set_shuffle_salt(0);
            sidecar.set_gpu_profiling(gpu_prof);
            model.set_layer_hook(use_hook ? &sidecar : nullptr);
            if (use_graph) model.warmup_decode_graphs();

            // Prime prompt
            if (call_prep) sidecar.prepare_step(prompt_ids, 0);
            model.forward(prompt_ids, ppos, logits);
            int32_t cur = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            int32_t pos = static_cast<int32_t>(prompt_ids.size());

            // Warmup 8 steps
            for (int w = 0; w < 8; ++w) {
                if (call_prep) sidecar.prepare_step({cur}, pos);
                cur = model.forward_decode_greedy(cur, pos++);
            }

            // Timed N steps
            const int N = std::max(64, n_gen);
            sidecar.reset_profile();
            const auto t0 = std::chrono::steady_clock::now();
            for (int s = 0; s < N; ++s) {
                if (call_prep) sidecar.prepare_step({cur}, pos);
                cur = model.forward_decode_greedy(cur, pos++);
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double tot_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double us_tok = (tot_ms * 1000.0) / N;
            const double tps    = N * 1000.0 / tot_ms;
            const auto & pr = sidecar.profile();
            std::printf("  %-34s | %6.2f tok/s | %8.1f us/tok", label, tps, us_tok);
            if (call_prep && pr.steps > 0) {
                std::printf(" | prep: tok=%.1f hash=%.1f gath=%.1f h2d=%.1f us",
                            pr.tokenize_align_us / pr.steps, pr.hash_prefetch_us / pr.steps,
                            pr.gather_dequant_us / pr.steps, pr.h2d_submit_us / pr.steps);
            }
            if (gpu_prof && pr.gpu_hook_calls > 0) {
                const double gpu_per_tok = pr.gpu_hook_us / N;
                std::printf(" | gpu_hooks=%.1f us/tok", gpu_per_tok);
            }
            std::printf("\n");
            return us_tok;
        };

        std::printf("\n=== ENGRAM STAGE-BY-STAGE DECODE BENCHMARK (tg%d, CUDA Graph ON) ===\n", std::max(64, n_gen));
        const double base_us = run_decode_bench("1. Base (no hook)",                false, false, 0.0f, true, false);
        const double dis_us  = run_decode_bench("2. Disabled (hook+prep, scale=0)", true,  true,  0.0f, true, false);
        const double act_us  = run_decode_bench("3. Active (hook+prep, scale>0)",   true,  true,  scale > 0.0f ? scale : 0.5f, true, false);
        std::printf("  Overhead vs Base: disabled=%+.1f us/tok (%+.2f%%) | active=%+.1f us/tok (%+.2f%%)\n",
                    dis_us - base_us, 100.0 * (dis_us - base_us) / base_us,
                    act_us - base_us, 100.0 * (act_us - base_us) / base_us);

        std::printf("\n=== GPU HOOK ISOLATION (non-graph CUDA events, tg%d) ===\n", std::max(64, n_gen));
        run_decode_bench("4. Active (non-graph + CUDA events)", true, true, scale > 0.0f ? scale : 0.5f, false, true);

        // Prefill pp1024 comparison
        std::vector<int32_t> pp_ids;
        while (pp_ids.size() < 1024) {
            for (int32_t id : prompt_ids) {
                if (pp_ids.size() >= 1024) break;
                pp_ids.push_back(id);
            }
        }
        std::vector<int32_t> pp_pos(pp_ids.size());
        std::iota(pp_pos.begin(), pp_pos.end(), 0);

        auto run_pp_bench = [&](const char * label, bool use_hook, float sc) {
            model.reset_state();
            sidecar.set_global_scale(sc);
            sidecar.set_gpu_profiling(false);
            model.set_layer_hook(use_hook ? &sidecar : nullptr);
            sidecar.reset_profile();
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t off = 0; off < pp_ids.size(); off += 256) {
                size_t len = std::min<size_t>(256, pp_ids.size() - off);
                std::vector<int32_t> pt(pp_ids.begin() + off, pp_ids.begin() + off + len);
                std::vector<int32_t> pp(pp_pos.begin() + off, pp_pos.begin() + off + len);
                if (use_hook) sidecar.prepare_step(pt, static_cast<int>(off));
                model.forward(pt, pp, logits);
            }
            const auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const auto & pr = sidecar.profile();
            std::printf("  %-34s | %7.1f tok/s | %7.2f ms", label, pp_ids.size() * 1000.0 / ms, ms);
            if (use_hook && pr.steps > 0) {
                std::printf(" | prep total: tok=%.2f hash=%.2f gath=%.2f h2d=%.2f ms",
                            pr.tokenize_align_us / 1000.0, pr.hash_prefetch_us / 1000.0,
                            pr.gather_dequant_us / 1000.0, pr.h2d_submit_us / 1000.0);
            }
            std::printf("\n");
            return ms;
        };

        std::printf("\n=== PREFILL BENCHMARK (pp1024, chunk=256) ===\n");
        const double pp_base = run_pp_bench("1. Base (no hook)", false, 0.0f);
        const double pp_act  = run_pp_bench("2. Active (hook+prep, scale>0)", true, scale > 0.0f ? scale : 0.5f);
        std::printf("  Prefill overhead vs Base: %+.2f ms (%+.2f%%)\n", pp_act - pp_base, 100.0 * (pp_act - pp_base) / pp_base);
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
