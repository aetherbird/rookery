#include "engram/engram_sidecar.h"
#include "gguf/reader.h"
#include "models/bailingmoe3.h"
#include "models/gemma4.h"
#include "models/inkling2_loop.h"
#include "models/maxwell_moe.h"
#include "models/qwen35_moe.h"
#include "models/qwen4exp.h"
#include "models/sensenova_loop.h"
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
    std::unique_ptr<rookery::MaxwellMoeModel>   maxwell;
    std::unique_ptr<rookery::Qwen35MoeModel>    qwen35;
    std::unique_ptr<rookery::Qwen4ExpModel>     qwen4exp;
    std::unique_ptr<rookery::Inkling2LoopModel> inkling2;
    std::unique_ptr<rookery::BailingMoe3Model>  bailing3;
    std::unique_ptr<rookery::Gemma4Model>       gemma4;
    std::unique_ptr<rookery::SenseNovaLoopModel> sensenova;

    void load(const std::string & path, const std::vector<int> & gpus, const std::vector<float> & split, int max_ctx) {
        auto gguf = rookery::GgufFile::open(path);
        arch = gguf.get_str("general.architecture", "maxwell-moe");
        if (arch == "qwen35moe") {
            qwen35 = std::make_unique<rookery::Qwen35MoeModel>();
            qwen35->load(path, gpus, split, max_ctx, true, false);
        } else if (arch == "qwen4exp") {
            qwen4exp = std::make_unique<rookery::Qwen4ExpModel>();
            qwen4exp->load(path, gpus, split, max_ctx, true);
        } else if (arch == "inkling2_loop" || arch == "inkling2") {
            inkling2 = std::make_unique<rookery::Inkling2LoopModel>();
            inkling2->load(path, gpus, max_ctx, rookery::KvCacheType::Q8_0);
        } else if (arch == "bailingmoe3") {
            bailing3 = std::make_unique<rookery::BailingMoe3Model>();
            bailing3->load(path, gpus, split, max_ctx, true, false);
        } else if (arch == "gemma4") {
            gemma4 = std::make_unique<rookery::Gemma4Model>();
            gemma4->load(path, gpus, split, max_ctx, rookery::KvCacheType::Q8_0, 1.0f);
        } else if (arch == "sensenova_loop") {
            sensenova = std::make_unique<rookery::SenseNovaLoopModel>();
            sensenova->load(path, gpus, split, max_ctx, true, false);
        } else {
            maxwell = std::make_unique<rookery::MaxwellMoeModel>();
            maxwell->load(path, gpus, split, max_ctx, rookery::KvCacheType::Q8_0, false);
        }
    }
    int n_embd() const {
        if (qwen35) return qwen35->config().n_embd;
        if (qwen4exp) return qwen4exp->config().n_embd;
        if (inkling2) return inkling2->config().n_embd;
        if (bailing3) return bailing3->config().n_embd;
        if (gemma4) return gemma4->config().n_embd;
        if (sensenova) return sensenova->config().n_embd;
        return maxwell->config().n_embd;
    }
    int hc_mult() const {
        if (qwen4exp) return qwen4exp->config().hc_mult;
        if (qwen35 || inkling2 || bailing3 || gemma4 || sensenova) return 1;
        return maxwell->config().hc_mult;
    }
    std::vector<int> layer_devices() const {
        if (qwen35) return qwen35->layer_devices();
        if (qwen4exp) return qwen4exp->layer_devices();
        if (inkling2) return inkling2->layer_devices();
        if (bailing3) return bailing3->layer_devices();
        if (gemma4) return gemma4->layer_devices();
        if (sensenova) return sensenova->layer_devices();
        return maxwell->layer_devices();
    }
    void set_layer_hook(rookery::LayerHook * h) {
        if (qwen35) qwen35->set_layer_hook(h);
        else if (qwen4exp) qwen4exp->set_layer_hook(h);
        else if (inkling2) inkling2->set_layer_hook(h);
        else if (bailing3) bailing3->set_layer_hook(h);
        else if (gemma4) gemma4->set_layer_hook(h);
        else if (sensenova) sensenova->set_layer_hook(h);
        else maxwell->set_layer_hook(h);
    }
    void set_use_cuda_graphs(bool en) {
        if (qwen35) qwen35->set_use_cuda_graphs(en);
        else if (qwen4exp) qwen4exp->set_use_cuda_graphs(en);
        else if (inkling2) inkling2->set_use_cuda_graphs(en);
        else if (bailing3) bailing3->set_use_cuda_graphs(en);
        else if (gemma4) gemma4->set_use_cuda_graphs(en);
        else if (sensenova) sensenova->set_use_cuda_graphs(en);
        else maxwell->set_use_cuda_graphs(en);
    }
    void warmup_decode_graphs() {
        if (qwen35) qwen35->warmup_decode_graphs();
        else if (qwen4exp) qwen4exp->warmup_decode_graphs();
        else if (inkling2) inkling2->warmup_decode_graphs();
        else if (bailing3) bailing3->warmup_decode_graphs();
        else if (gemma4) gemma4->warmup_decode_graphs();
        else if (sensenova) sensenova->warmup_decode_graphs();
        else maxwell->warmup_decode_graphs();
    }
    void reset_state() {
        if (qwen35) qwen35->reset_state();
        else if (qwen4exp) qwen4exp->reset_state();
        else if (inkling2) inkling2->reset_state();
        else if (bailing3) bailing3->reset_state();
        else if (gemma4) gemma4->reset_state();
        else if (sensenova) sensenova->reset_state();
    }
    void forward(const std::vector<int32_t> & t, const std::vector<int32_t> & p, std::vector<float> & l) {
        if (qwen35) qwen35->forward(t, p, l);
        else if (qwen4exp) qwen4exp->forward(t, p, l);
        else if (inkling2) inkling2->forward(t, p, l);
        else if (bailing3) bailing3->forward(t, p, l);
        else if (gemma4) gemma4->forward(t, p, l);
        else if (sensenova) sensenova->forward(t, p, l);
        else maxwell->forward(t, p, l);
    }
    int32_t forward_decode_greedy(int32_t t, int32_t p) {
        if (qwen35) return qwen35->forward_decode_greedy(t, p);
        if (qwen4exp) return qwen4exp->forward_decode_greedy(t, p);
        if (inkling2) return inkling2->forward_decode_greedy(t, p);
        if (bailing3) return bailing3->forward_decode_greedy(t, p);
        if (gemma4) return gemma4->forward_decode_greedy(t, p);
        if (sensenova) return sensenova->forward_decode_greedy(t, p);
        return maxwell->forward_decode_greedy(t, p);
    }
    void forward_decode_logits(int32_t t, int32_t p, std::vector<float> & l) {
        if (qwen35) qwen35->forward_decode_logits(t, p, l);
        else if (qwen4exp) qwen4exp->forward_decode_logits(t, p, l);
        else if (inkling2) inkling2->forward_decode_logits(t, p, l);
        else if (bailing3) bailing3->forward_decode_logits(t, p, l);
        else if (gemma4) gemma4->forward_decode_logits(t, p, l);
        else if (sensenova) sensenova->forward_decode_logits(t, p, l);
        else maxwell->forward_decode_logits(t, p, l);
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
    const std::vector<rookery::EngramSidecar *> & sidecars,
    const std::vector<int32_t> & ids) {

    EvalMetrics m;
    if (ids.size() < 2) return m;

    model.reset_state();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits;

    // Step 0: prime position 0
    for (auto * sc : sidecars) sc->prepare_step({ids[0]}, 0);
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
            for (auto * sc : sidecars) sc->prepare_step({target}, i + 1);
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
    std::string engram2_path;
    std::string adapter2_path;
    std::string inspect_adapter_path;
    std::string dump_path;
    std::string dump_probe_path;
    std::string dump_ce_path;
    std::string corpus_path;
    std::vector<std::string> eval_paths;
    std::string factual_path;
    std::vector<int>   gpus = {0, 1};
    std::vector<float> split = {24.0f, 32.0f};
    std::vector<int>   engram_layers;
    std::vector<int>   engram2_layers;
    std::vector<float> scale_sweep;
    float    scale        = 0.0f;
    float    scale2_ratio = 1.0f;
    float    gate_bias    = 0.0f;
    bool     gate_bias_set = false;
    float    gate_bias2   = 0.0f;
    bool     gate_bias2_set = false;
    int      n_gen        = 64;
    int      max_toks     = 20000;
    int      eval_toks    = 768;
    int      ce_layer     = 44;
    int      ce_src_layer = -1;
    int      doc_ctx      = 4096;
    uint64_t shuffle_salt = 0;
    bool     table_only   = false;
    bool     dump_delta   = false;
    bool     verify_addr  = false;
    bool     bench_mode   = false;
    bool     parse_special = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m" || a == "--model")            model_path = next();
        else if (a == "--engram")                   engram_path = next();
        else if (a == "--engram-adapter")           adapter_path = next();
        else if (a == "--engram2")                  engram2_path = next();
        else if (a == "--engram-adapter2")          adapter2_path = next();
        else if (a == "--engram-scale2-ratio")      scale2_ratio = std::strtof(next().c_str(), nullptr);
        else if (a == "--engram-gate-bias2")        { gate_bias2 = std::strtof(next().c_str(), nullptr); gate_bias2_set = true; }
        else if (a == "--engram-layers2")           engram2_layers = parse_csv_ints(next());
        else if (a == "--inspect-adapter")          inspect_adapter_path = next();
        else if (a == "--engram-scale")             scale = std::strtof(next().c_str(), nullptr);
        else if (a == "--engram-gate-bias")         { gate_bias = std::strtof(next().c_str(), nullptr); gate_bias_set = true; }
        else if (a == "--scale-sweep")              scale_sweep = parse_csv_floats(next());
        else if (a == "--engram-layers")            engram_layers = parse_csv_ints(next());
        else if (a == "--dump-pairs")               dump_path = next();
        else if (a == "--dump-table-probe")         dump_probe_path = next();
        else if (a == "--dump-tail-ce")             dump_ce_path = next();
        else if (a == "--ce-layer")                 ce_layer = std::atoi(next().c_str());
        else if (a == "--ce-src-layer")             ce_src_layer = std::atoi(next().c_str());
        else if (a == "--doc-ctx")                  doc_ctx = std::atoi(next().c_str());
        else if (a == "--dump-delta")               dump_delta = true;
        else if (a == "--corpus")                   corpus_path = next();
        else if (a == "--eval-ppl")                 eval_paths.push_back(next());
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
        else if (a == "--parse-special")            parse_special = true;
    }

    // -1. Fast standalone RKAD v1/v2/v3 adapter inspector (no GPU or GGUF required!)
    if (!inspect_adapter_path.empty()) {
        FILE * fp = std::fopen(inspect_adapter_path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "Cannot open RKAD adapter: %s\n", inspect_adapter_path.c_str());
            return 1;
        }
        char magic[4]{};
        uint32_t hdr[6]{};
        if (std::fread(magic, 1, 4, fp) != 4 || std::memcmp(magic, "RKAD", 4) != 0 ||
            std::fread(hdr, sizeof(uint32_t), 6, fp) != 6) {
            std::fclose(fp);
            std::fprintf(stderr, "Invalid RKAD header in %s\n", inspect_adapter_path.c_str());
            return 1;
        }
        const uint32_t ver = hdr[0], nh = hdr[1], de = hdr[2], dh = hdr[3], r_hdr = hdr[4], hg_hdr = hdr[5];
        std::printf("RKAD Adapter: %s\n", inspect_adapter_path.c_str());
        std::printf("  format_version=%u  n_hooks=%u  d_engram=%u  d_host=%u  default_rank=%u\n", ver, nh, de, dh, r_hdr);
        uint32_t hc_est = 4;
        if (ver >= 3) {
            uint32_t mlen = 0;
            if (std::fread(&mlen, sizeof(uint32_t), 1, fp) != 1) { std::fclose(fp); return 1; }
            std::string mstr(mlen, '\0');
            if (mlen > 0 && std::fread(mstr.data(), 1, mlen, fp) != mlen) { std::fclose(fp); return 1; }
            std::printf("  manifest=%s\n", mstr.c_str());
            auto pos_hc = mstr.find("\"host_hc_mult\":");
            if (pos_hc != std::string::npos) hc_est = static_cast<uint32_t>(std::atoi(mstr.c_str() + pos_hc + 15));
        }
        for (uint32_t i = 0; i < nh; ++i) {
            int32_t meta[2]{};
            float sc = 0.0f, clip = 0.0f;
            uint32_t hr = r_hdr;
            bool has_mu = (ver >= 2), has_g = (hg_hdr != 0);
            if (std::fread(meta, sizeof(int32_t), 2, fp) != 2 || std::fread(&sc, sizeof(float), 1, fp) != 1) break;
            if (ver >= 3) {
                uint32_t ex[2]{};
                if (std::fread(ex, sizeof(uint32_t), 2, fp) != 2 || std::fread(&clip, sizeof(float), 1, fp) != 1) break;
                hr = ex[0];
                has_mu = (ex[1] & 1u) != 0;
                has_g  = (ex[1] & 2u) != 0;
                if ((ex[1] & 4u) == 0) clip = 0.0f;
            }
            size_t skip_floats = (has_mu ? de : 0) + (hr ? (size_t(hr) * de + size_t(dh) * hr) : size_t(dh) * de);
            if (has_g) skip_floats += (hr ? (size_t(hr) * de + size_t(dh) * hr) : size_t(dh) * de) + size_t(hc_est) * dh;
            std::fseek(fp, static_cast<long>(skip_floats * sizeof(float)), SEEK_CUR);
            std::printf("  hook[%u]: src_layer=%d -> host_layer=%d  rank=%u  scale=%.4f  centered=%s  gated=%s  delta_clip=%.3f  params=%.2fM\n",
                        i, meta[1], meta[0], hr, sc, has_mu ? "yes" : "no", has_g ? "yes" : "no", clip,
                        static_cast<double>(skip_floats) / 1e6);
        }
        std::fclose(fp);
        return 0;
    }

    if (engram_path.empty()) {
        std::fprintf(stderr, "Usage: %s --engram <table.gguf> [-m <host.gguf>] [--engram-adapter <a.rkad>] [--inspect-adapter <a.rkad>] [--engram-scale 0.0] [--scale-sweep 0,0.25,0.5,1] [--dump-pairs <p.rkpr>] [--dump-delta] [--eval-ppl <txt>] [--eval-factual <tsv>] [--shuffle-salt 0] [--verify-addressing] [--bench] [--table-only]\n", argv[0]);
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

    // 0b. Fast CPU-only table probe dump for provider-only engram-z-v3 Exporter alignment (0 VRAM!)
    if (!dump_probe_path.empty()) {
        if (model_path.empty()) {
            std::fprintf(stderr, "--dump-table-probe requires -m <ref_tokenizer.gguf>\n");
            return 1;
        }
        rookery::BpeTokenizer host_tok;
        {
            auto gguf = rookery::GgufFile::open(model_path);
            host_tok.load(gguf);
        }
        rookery::EngramTable tbl;
        tbl.open(engram_path);
        std::string text = corpus_path.empty() ? "The capital of France is Paris." : read_text_file(corpus_path);
        auto ids = host_tok.encode(text, true, false);
        if (static_cast<int>(ids.size()) > max_toks) ids.resize(max_toks);

        const int n_tok = static_cast<int>(ids.size());
        const int n_eng = std::max(1, tbl.cfg().n_layers);
        const int d_eng = tbl.cfg().engram_dim();
        const int n_cols = tbl.cfg().n_cols;
        const int n_hooks = engram_layers.empty() ? 1 : static_cast<int>(engram_layers.size());

        FILE * fp = std::fopen(dump_probe_path.c_str(), "wb");
        if (!fp) throw std::runtime_error("Cannot open dump_probe_path: " + dump_probe_path);
        const char magic[4] = {'R', 'K', 'P', 'R'};
        uint32_t hdr[4] = {1u, static_cast<uint32_t>(n_hooks), static_cast<uint32_t>(d_eng), 1u};
        uint64_t total_rec = static_cast<uint64_t>(n_hooks) * n_tok;
        std::fwrite(magic, 1, 4, fp);
        std::fwrite(hdr, sizeof(uint32_t), 4, fp);
        std::fwrite(&total_rec, sizeof(uint64_t), 1, fp);
        for (int hi = 0; hi < n_hooks; ++hi) {
            int32_t pair[2] = {engram_layers.empty() ? 0 : engram_layers[hi], std::min(hi, n_eng - 1)};
            std::fwrite(pair, sizeof(int32_t), 2, fp);
        }

        // Exact causal tokenization + alignment using EngramSidecar's production commit cache
        rookery::EngramSidecar sc_helper;
        std::vector<int> dummy_devs(4, 0);
        sc_helper.init(engram_path, &host_tok, dummy_devs, 1, 1, 512, "", 0.0f, {0});
        // Run exact token-by-token causal lookup via chunked 256-token steps (matching --dump-pairs!)
        rookery::EngramSeqState st;
        std::vector<int32_t> row_ids(static_cast<size_t>(n_tok) * n_cols);
        std::vector<float>   val_mask(static_cast<size_t>(n_tok) * n_cols);
        std::vector<int32_t> s_ids;
        std::vector<float>   s_val;
        std::string rolling;
        std::vector<size_t> host_ends;
        host_ends.reserve(n_tok);
        std::vector<int32_t> comm_toks;
        std::vector<size_t>  comm_ends;
        const int n_ctx_need = std::max(1, tbl.cfg().ngram_size - 1);

        // Compute aligned source tokens in 256-token chunks (identical to Mode A --dump-pairs!)
        std::vector<std::vector<int32_t>> per_hook_rows(n_hooks, std::vector<int32_t>(static_cast<size_t>(n_tok) * n_cols));
        std::vector<std::vector<float>>   per_hook_vals(n_hooks, std::vector<float>(static_cast<size_t>(n_tok) * n_cols));
        const int chunk = 256;
        for (int off = 0; off < n_tok; off += chunk) {
            const int len = std::min(chunk, n_tok - off);
            for (int i = 0; i < len; ++i) {
                rolling.append(host_tok.decode_token(ids[off + i]));
                host_ends.push_back(rolling.size());
            }
            const size_t comm_b = comm_ends.empty() ? 0 : comm_ends.back();
            const std::string tail_text = (comm_b > 0) ? rolling.substr(comm_b) : rolling;
            const auto tail_src = tbl.has_tokenizer() ? tbl.tokenizer().encode(tail_text, false, false)
                                                      : std::vector<int32_t>(ids.begin() + off, ids.begin() + off + len);
            const int n_tail = static_cast<int>(tail_src.size());
            std::vector<size_t> tail_ends(n_tail);
            size_t cb = comm_b;
            for (int j = 0; j < n_tail; ++j) {
                cb += tbl.has_tokenizer() ? tbl.tokenizer().token_byte_len(tail_src[j]) : 1;
                tail_ends[j] = cb;
            }
            const int n_comm = static_cast<int>(comm_toks.size());
            const int pfx = std::min(n_comm, n_ctx_need);
            std::vector<int32_t> src_win;
            src_win.reserve(pfx + n_tail);
            for (int k = n_comm - pfx; k < n_comm; ++k) src_win.push_back(comm_toks[k]);
            for (int j = 0; j < n_tail; ++j) src_win.push_back(tail_src[j]);

            int j = 0;
            for (int i = 0; i < len; ++i) {
                const size_t tb = host_ends[off + i];
                while (j + 1 < n_tail && tail_ends[j] < tb) ++j;
                const int ap = (n_tail > 0) ? (pfx + j) : std::max(0, pfx - 1);
                for (int hi = 0; hi < n_hooks; ++hi) {
                    const int src_l = std::min(hi, n_eng - 1);
                    tbl.hash_tokens(src_l, src_win, ap, 1, s_ids, s_val);
                    std::memcpy(per_hook_rows[hi].data() + static_cast<size_t>(off + i) * n_cols, s_ids.data(), n_cols * sizeof(int32_t));
                    std::memcpy(per_hook_vals[hi].data() + static_cast<size_t>(off + i) * n_cols, s_val.data(), n_cols * sizeof(float));
                }
            }
            if (tbl.has_tokenizer() && rolling.size() > comm_b + 64) {
                const size_t cut = rookery::EngramSidecar::find_safe_bpe_cut(rolling, comm_b + 1, rolling.size() - 48, 1024);
                int ck = -1;
                for (int k = 0; k < n_tail; ++k) {
                    if (tail_ends[k] == cut) { ck = k; break; }
                    if (tail_ends[k] > cut) break;
                }
                if (ck >= 0) {
                    for (int k = 0; k <= ck; ++k) {
                        comm_toks.push_back(tail_src[k]);
                        comm_ends.push_back(tail_ends[k]);
                    }
                }
            }
        }

        std::vector<float> e_buf(static_cast<size_t>(n_tok) * d_eng);
        std::vector<float> h_dummy(n_tok, 0.0f);
        for (int hi = 0; hi < n_hooks; ++hi) {
            tbl.gather_dequant_f32(per_hook_rows[hi].data(), per_hook_vals[hi].data(), n_tok, e_buf.data());
            const int32_t rh[2] = {hi, n_tok};
            std::fwrite(rh, sizeof(int32_t), 2, fp);
            std::fwrite(e_buf.data(), sizeof(float), e_buf.size(), fp);
            std::fwrite(h_dummy.data(), sizeof(float), h_dummy.size(), fp);
        }
        std::fclose(fp);
        std::printf("[CPU Table Probe] Wrote %s (%d hooks x %d toks x d_eng=%d, 0 VRAM)\n",
                    dump_probe_path.c_str(), n_hooks, n_tok, d_eng);
        return 0;
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
    if (gate_bias_set) {
        sidecar.set_gate_bias(gate_bias);
        std::printf("GATE BIAS OVERRIDE: %.4f\n", gate_bias);
    }
    if (shuffle_salt != 0) {
        sidecar.set_shuffle_salt(shuffle_salt);
        std::printf("CONTROL MODE: Engram hash shuffle_salt = 0x%llx\n", static_cast<unsigned long long>(shuffle_salt));
    }

    rookery::EngramSidecar sidecar2;
    rookery::CompositeLayerHook composite_hook;
    std::vector<rookery::EngramSidecar *> active_sidecars = {&sidecar};
    rookery::LayerHook * active_hook = &sidecar;
    if (!engram2_path.empty()) {
        sidecar2.init(engram2_path, &host_tok, model.layer_devices(),
                      model.n_embd(), model.hc_mult(),
                      512, adapter2_path, scale * scale2_ratio, engram2_layers, model.arch);
        if (gate_bias2_set) {
            sidecar2.set_gate_bias(gate_bias2);
            std::printf("GATE BIAS2 OVERRIDE: %.4f\n", gate_bias2);
        }
        if (shuffle_salt != 0) sidecar2.set_shuffle_salt(shuffle_salt);
        active_sidecars.push_back(&sidecar2);
        composite_hook.add(&sidecar);
        composite_hook.add(&sidecar2);
        active_hook = &composite_hook;
        std::printf("DUAL-TABLE FUSION ACTIVE: table1=%s + table2=%s (scale2_ratio=%.3f)\n",
                    sidecar.table().cfg().arch.c_str(), sidecar2.table().cfg().arch.c_str(), scale2_ratio);
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
        auto ids = host_tok.encode(text, true, parse_special);
        if (static_cast<int>(ids.size()) > max_toks) ids.resize(max_toks);
        std::printf("Tail-CE dump corpus: %zu host tokens (layer=%d, doc_ctx=%d, parse_special=%s)\n",
                    ids.size(), ce_layer, doc_ctx, parse_special ? "yes" : "no");

        sidecar.set_global_scale(0.0f);
        sidecar.enable_tail_ce_dump(dump_ce_path, ce_layer, ce_src_layer);
        model.set_layer_hook(&sidecar);

        std::vector<float> logits;
        const size_t chunk = 256;
        // Process in non-overlapping windows of up to doc_ctx tokens (resets position to 0 each window to keep KV small).
        // For models that require BOS at position 0 (e.g. Gemma-4), prepend bos_id to every window so attention sinks stay valid.
        const bool prepend_bos = host_tok.add_bos_default() && host_tok.bos_id() >= 0;
        const size_t stride = prepend_bos ? static_cast<size_t>(std::max(2, doc_ctx - 1)) : static_cast<size_t>(doc_ctx);
        for (size_t win_start = 0; win_start + 1 < ids.size(); win_start += stride) {
            model.reset_state();
            std::vector<int32_t> win;
            win.reserve(static_cast<size_t>(doc_ctx));
            if (prepend_bos && ids[win_start] != host_tok.bos_id()) {
                win.push_back(host_tok.bos_id());
            }
            const size_t take = std::min(ids.size() - win_start, static_cast<size_t>(doc_ctx) - win.size());
            win.insert(win.end(), ids.begin() + win_start, ids.begin() + win_start + take);
            const size_t win_len = win.size();
            if (win_len < 2) break;
            for (size_t off = 0; off < win_len; off += chunk) {
                const size_t len = std::min(chunk, win_len - off);
                std::vector<int32_t> pt(win.begin() + off, win.begin() + off + len);
                std::vector<int32_t> pp(len);
                std::iota(pp.begin(), pp.end(), static_cast<int32_t>(off));
                std::vector<int32_t> tgts(len, -1);
                for (size_t k = 0; k < len; ++k) {
                    if (off + k + 1 < win_len) tgts[k] = win[off + k + 1];
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

        auto eval_factual_run = [&](const std::vector<rookery::EngramSidecar *> & scs) {
            double sum_nll = 0.0;
            int total_ans_toks = 0, hit1 = 0, hit5 = 0, exact_first1 = 0, exact_first5 = 0;
            std::vector<float> logits;
            for (const auto & [q, a] : qa_pairs) {
                model.reset_state();
                auto q_ids = host_tok.encode(q, true, false);
                auto a_ids = host_tok.encode(a, false, false);
                if (q_ids.empty() || a_ids.empty()) continue;

                // Prefill prompt causally
                if (!scs.empty()) {
                    for (size_t i = 0; i < q_ids.size(); ++i) {
                        for (auto * sc : scs) sc->prepare_step({q_ids[i]}, static_cast<int>(i));
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
                        for (auto * sc : scs) sc->prepare_step({target}, pos);
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

        for (auto * sc : active_sidecars) { sc->set_global_scale(0.0f); sc->set_shuffle_salt(0); }
        model.set_layer_hook(nullptr);
        model.warmup_decode_graphs();
        auto base = eval_factual_run({});
        std::printf("  [BASE    ] (no hook)          | ans_NLL=%.5f | ans_PPL=%7.3f | first_top1=%5.1f%% | tok_top1=%5.1f%% | tok_top5=%5.1f%%\n",
                    base.nll, base.ppl, base.q1, base.tok1, base.tok5);

        model.set_layer_hook(active_hook);
        model.warmup_decode_graphs();
        auto dis = eval_factual_run(active_sidecars);
        std::printf("  [DISABLED] global_scale= 0.000 | ans_NLL=%.5f | ans_PPL=%7.3f | delta vs base=%+.6f\n",
                    dis.nll, dis.ppl, dis.ppl - base.ppl);

        for (float s : scale_sweep) {
            if (s <= 0.0f) continue;
            sidecar.set_global_scale(s);
            if (!engram2_path.empty()) sidecar2.set_global_scale(s * scale2_ratio);
            for (auto * sc : active_sidecars) sc->set_shuffle_salt(0);
            model.set_layer_hook(active_hook);
            model.warmup_decode_graphs();
            auto res = eval_factual_run(active_sidecars);
            std::printf("  [REAL    ] global_scale=%6.3f | ans_NLL=%.5f | ans_PPL=%7.3f | first_top1=%5.1f%% | tok_top1=%5.1f%% | tok_top5=%5.1f%% (vs base=%+.4f)\n",
                        s, res.nll, res.ppl, res.q1, res.tok1, res.tok5, res.ppl - base.ppl);
            if (shuffle_salt != 0) {
                for (auto * sc : active_sidecars) sc->set_shuffle_salt(shuffle_salt);
                auto ctl = eval_factual_run(active_sidecars);
                std::printf("  [SHUFFLED] global_scale=%6.3f | ans_NLL=%.5f | ans_PPL=%7.3f | first_top1=%5.1f%% | tok_top1=%5.1f%% | tok_top5=%5.1f%% (shuf-real=%+.4f)\n",
                            s, ctl.nll, ctl.ppl, ctl.q1, ctl.tok1, ctl.tok5, ctl.ppl - res.ppl);
            }
        }
        if (eval_paths.empty()) return 0;
    }

    // Mode B: Held-Out Perplexity & Top-1/Top-5 Accuracy Sweep (4-way: Base, Disabled, Real, Shuffled!)
    if (!eval_paths.empty()) {
        for (const std::string & eval_path : eval_paths) {
        std::string text = read_text_file(eval_path);
        auto ids = host_tok.encode(text, true, false);
        if (static_cast<int>(ids.size()) > eval_toks + 1) ids.resize(eval_toks + 1);
        std::printf("Held-out eval file: %s (%zu tokens evaluated)\n", eval_path.c_str(), ids.size() - 1);

        if (scale_sweep.empty()) scale_sweep = {0.25f, 0.5f, 1.0f, 2.0f};

        // Control 1: Pure Base (no LayerHook, no prepare_step)
        for (auto * sc : active_sidecars) { sc->set_global_scale(0.0f); sc->set_shuffle_salt(0); }
        model.set_layer_hook(nullptr);
        model.warmup_decode_graphs();
        auto base_res = eval_token_ppl(model, {}, ids);
        std::printf("  [BASE    ] (no hook)          | NLL=%.5f | PPL=%7.3f | top1=%5.2f%% | top5=%5.2f%% (%.1f tok/s)\n",
                    base_res.nll, base_res.ppl, base_res.top1, base_res.top5, base_res.count * 1000.0 / base_res.ms);

        // Control 2: Bridge-Disabled (hook attached & prepare_step runs, scale=0.0)
        for (auto * sc : active_sidecars) { sc->set_global_scale(0.0f); sc->set_shuffle_salt(0); }
        model.set_layer_hook(active_hook);
        model.warmup_decode_graphs();
        auto dis_res = eval_token_ppl(model, active_sidecars, ids);
        std::printf("  [DISABLED] global_scale= 0.000 | NLL=%.5f | PPL=%7.3f | top1=%5.2f%% | top5=%5.2f%% (delta vs base=%+.6f)\n",
                    dis_res.nll, dis_res.ppl, dis_res.top1, dis_res.top5, dis_res.ppl - base_res.ppl);

        for (float s : scale_sweep) {
            if (s <= 0.0f) continue;
            sidecar.set_global_scale(s);
            if (!engram2_path.empty()) sidecar2.set_global_scale(s * scale2_ratio);
            for (auto * sc : active_sidecars) sc->set_shuffle_salt(0);
            model.set_layer_hook(active_hook);
            model.warmup_decode_graphs();
            auto res = eval_token_ppl(model, active_sidecars, ids);
            std::printf("  [REAL    ] global_scale=%6.3f | NLL=%.5f | PPL=%7.3f | top1=%5.2f%% | top5=%5.2f%% (vs base=%+.4f, %.1f tok/s)\n",
                        s, res.nll, res.ppl, res.top1, res.top5, res.ppl - base_res.ppl, res.count * 1000.0 / res.ms);

            if (shuffle_salt != 0) {
                for (auto * sc : active_sidecars) sc->set_shuffle_salt(shuffle_salt);
                auto ctl = eval_token_ppl(model, active_sidecars, ids);
                std::printf("  [SHUFFLED] global_scale=%6.3f | NLL=%.5f | PPL=%7.3f | top1=%5.2f%% | top5=%5.2f%% (shuf-real=%+.4f)\n",
                            s, ctl.nll, ctl.ppl, ctl.top1, ctl.top5, ctl.ppl - res.ppl);
            }
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
