#include "cpu/cpu_kernels.h"
#include "engram/engram_sidecar.h"
#include "gguf/reader.h"
#include "server/chat_template.h"
#include "tokenizer/bpe.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void test_chat_parser() {
    // 1. Normal thinking + answer
    {
        auto p = rookery::parse_maxwell_output("Let me think.\n</think>\n\nThe sky is blue.", true);
        assert(p.reasoning_content == "Let me think.");
        assert(p.content == "The sky is blue.");
        assert(p.tool_calls.empty());
    }
    // 2. Truncated mid-think (hit max_tokens before </think>)
    {
        auto p = rookery::parse_maxwell_output("Still thinking about physics...", true);
        assert(p.reasoning_content == "Still thinking about physics...");
        assert(p.content.empty());
        assert(p.tool_calls.empty());
    }
    // 3. Omitted </parameter> regression test (upstream llama.cpp qwen3-coder XML leak bug!)
    {
        const std::string raw =
            "Plan it.\n</think>\n\n<tool_call>\n<function=todo>\n"
            "<parameter=action>\ncreate\n"
            "<parameter=items>\n[{\"id\": 1, \"title\": \"fix\"}]\n</parameter>\n"
            "</function>\n</tool_call>";
        auto p = rookery::parse_maxwell_output(raw, true);
        assert(p.reasoning_content == "Plan it.");
        assert(p.content.empty());
        assert(p.tool_calls.size() == 1);
        assert(p.tool_calls[0]["function"]["name"] == "todo");
        auto args = nlohmann::json::parse(p.tool_calls[0]["function"]["arguments"].get<std::string>());
        assert(args["action"] == "create");
        assert(args["items"].is_array() && args["items"].size() == 1);
    }
    // 4. Qwen2 / Qwen2.5 Instruct ChatML rendering & JSON tool-call parsing
    {
        nlohmann::json msgs = nlohmann::json::array({
            {{"role", "user"}, {"content", "What is 2+2?"}}
        });
        std::string prompt = rookery::render_qwen2_chat_prompt(msgs, nlohmann::json());
        assert(prompt.find("<|im_start|>system\nYou are Qwen") == 0);
        assert(prompt.find("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n") != std::string::npos);
        assert(prompt.find("<think>") == std::string::npos);

        nlohmann::json tools = nlohmann::json::array({
            {{"type", "function"}, {"function", {{"name", "get_weather"}, {"parameters", {{"type", "object"}}}}}}
        });
        std::string tprompt = rookery::render_qwen2_chat_prompt(msgs, tools);
        assert(tprompt.find("<tools>") != std::string::npos);
        assert(tprompt.find("get_weather") != std::string::npos);

        auto p_plain = rookery::parse_qwen2_output("4.");
        assert(p_plain.content == "4.");
        assert(p_plain.reasoning_content.empty());
        assert(p_plain.tool_calls.empty());

        auto p_tool = rookery::parse_qwen2_output(
            "Checking now.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>");
        assert(p_tool.content == "Checking now.");
        assert(p_tool.tool_calls.size() == 1);
        assert(p_tool.tool_calls[0]["function"]["name"] == "get_weather");
        auto targs = nlohmann::json::parse(p_tool.tool_calls[0]["function"]["arguments"].get<std::string>());
        assert(targs["city"] == "Paris");
    }
    std::printf("[PASS] chat_template & XML-leak-safe tool parser unit tests\n");
}

static void test_engram_rkad_v3_contract() {
    rookery::EngramConfig cfg;
    cfg.arch        = "deepseek41";
    cfg.mode        = rookery::EngramMode::NGRAM_HASH;
    cfg.n_layers    = 2;
    cfg.ngram_size  = 4;
    cfg.n_cols      = 24;
    cfg.head_dim    = 256;
    cfg.fingerprint = 0x123456789abcdef0ULL;

    const std::string valid_json = R"({
        "schema": "rookery.engram.adapter.v3",
        "provider_arch": "deepseek41",
        "provider_mode": "ngram_hash",
        "provider_ngram_size": 4,
        "provider_n_cols": 24,
        "provider_head_dim": 256,
        "provider_fingerprint": "0x123456789abcdef0",
        "host_arch": "maxwell-moe",
        "host_n_layers": 45,
        "host_d_embd": 2048,
        "host_hc_mult": 4,
        "calibration": "level3_ce_kl"
    })";

    // 1. Valid manifest passes and parses all fields
    auto m = rookery::validate_rkad_v3_manifest(valid_json, cfg, "maxwell-moe", 45, 2048, 4);
    assert(m.format_version == 3);
    assert(m.provider_arch == "deepseek41");
    assert(m.host_arch == "maxwell-moe");
    assert(m.calibration == "level3_ce_kl");

    // 2. Every contract violation must throw std::runtime_error
    auto expect_throw = [&](auto fn, const char * label) {
        bool threw = false;
        try { fn(); } catch (const std::runtime_error &) { threw = true; }
        if (!threw) {
            std::fprintf(stderr, "[FAIL] Expected contract rejection for: %s\n", label);
            std::abort();
        }
    };

    expect_throw([&]() { rookery::validate_rkad_v3_manifest(valid_json, cfg, "qwen35moe", 45, 2048, 4); }, "host_arch");
    expect_throw([&]() { rookery::validate_rkad_v3_manifest(valid_json, cfg, "maxwell-moe", 40, 2048, 4); }, "host_n_layers");
    expect_throw([&]() { rookery::validate_rkad_v3_manifest(valid_json, cfg, "maxwell-moe", 45, 4096, 4); }, "host_d_embd");
    expect_throw([&]() { rookery::validate_rkad_v3_manifest(valid_json, cfg, "maxwell-moe", 45, 2048, 1); }, "host_hc_mult");

    auto bad_fp_cfg = cfg;
    bad_fp_cfg.fingerprint = 0xdeadbeefcafebabeULL;
    expect_throw([&]() { rookery::validate_rkad_v3_manifest(valid_json, bad_fp_cfg, "maxwell-moe", 45, 2048, 4); }, "provider_fingerprint");

    auto bad_arch_cfg = cfg;
    bad_arch_cfg.arch = "hy_v4_flash";
    expect_throw([&]() { rookery::validate_rkad_v3_manifest(valid_json, bad_arch_cfg, "maxwell-moe", 45, 2048, 4); }, "provider_arch");

    // 3. Per-sequence rollback_to across every position + divergent continuation
    rookery::EngramSeqState st;
    st.rolling_text = "hello world foo bar";
    st.host_byte_ends = {5, 11, 15, 19};
    st.committed_src_tokens = {10, 20};
    st.committed_src_ends   = {5, 11};
    st.rollback_to(2);
    assert(st.rolling_text == "hello world");
    assert(st.host_byte_ends.size() == 2);
    assert(st.committed_src_tokens.empty()); // rolled back safely due to 64B rollback margin
    st.rollback_to(0);
    assert(st.rolling_text.empty());

    // 4. Property-based tests for find_safe_bpe_cut:
    //    (a) Normal whitespace & newline boundaries
    {
        const std::string s = "first line\nsecond line third";
        const size_t cut = rookery::EngramSidecar::find_safe_bpe_cut(s, 1, s.size() - 5, 1024);
        assert(cut > 0 && (s[cut] == ' ' || s[cut - 1] == '\n'));
    }
    //    (b) Space-less base64 / minified JSON bounded-tail fallback
    {
        const std::string b64(2500, 'A');
        const size_t cut_short = rookery::EngramSidecar::find_safe_bpe_cut(b64.substr(0, 500), 1, 450, 1024);
        assert(cut_short == 0); // below max_uncommitted: waits for real boundary
        const size_t cut_long = rookery::EngramSidecar::find_safe_bpe_cut(b64, 1, 2400, 1024);
        assert(cut_long == 2400); // exceeds max_uncommitted: bounds uncommitted tail!
    }
    //    (c) Multi-byte UTF-8 space-less CJK & emoji: cut must NEVER land on a continuation byte (10xxxxxx)
    {
        std::string cjk;
        for (int i = 0; i < 600; ++i) cjk += "\xe4\xbd\xa0\xe5\xa5\xbd\xf0\x9f\x9a\x80"; // 10 bytes per repeat = 6000 B
        for (size_t max_c = 1500; max_c < 1530; ++max_c) {
            const size_t cut = rookery::EngramSidecar::find_safe_bpe_cut(cjk, 1, max_c, 1024);
            assert(cut >= 1 && cut <= max_c);
            const unsigned char b = static_cast<unsigned char>(cjk[cut]);
            assert((b & 0xC0u) != 0x80u); // guaranteed valid UTF-8 codepoint boundary!
        }
    }
    //    (d) Arbitrary-position rollback on long multi-byte stream
    {
        rookery::EngramSeqState s2;
        std::string txt;
        std::vector<size_t> ends;
        for (int i = 0; i < 200; ++i) {
            txt += "tok" + std::to_string(i) + " ";
            ends.push_back(txt.size());
        }
        for (int rp = 0; rp <= 200; rp += 17) {
            s2.rolling_text = txt;
            s2.host_byte_ends = ends;
            s2.committed_src_tokens = {1, 2, 3, 4, 5};
            s2.committed_src_ends   = {10, 50, 100, 300, 600};
            s2.rollback_to(rp);
            assert(static_cast<int>(s2.host_byte_ends.size()) == rp);
            if (rp == 0) {
                assert(s2.rolling_text.empty());
                assert(s2.committed_src_tokens.empty());
            } else {
                assert(s2.rolling_text.size() == ends[rp - 1]);
                if (!s2.committed_src_ends.empty()) {
                    assert(s2.committed_src_ends.back() + 64 <= s2.rolling_text.size());
                }
            }
        }
    }
    //    (e) release_sequence / clear_all_sequences lifecycle
    {
        rookery::EngramSidecar sc;
        assert(sc.active_sequences() == 0);
        sc.release_sequence(42);
        sc.clear_all_sequences();
        assert(sc.active_sequences() == 0);
    }

    std::printf("[PASS] RKAD v3 manifest contract, BPE cut bounds & EngramSeqState unit tests\n");
}

static void test_kv_cache_types() {
    static_assert(sizeof(rookery::BlockQ8_0) == 34, "BlockQ8_0 size");
    static_assert(sizeof(rookery::BlockQ4_0) == 18, "BlockQ4_0 size");
    assert(rookery::parse_kv_cache_type("f16")  == rookery::KvCacheType::F16);
    assert(rookery::parse_kv_cache_type("F16")  == rookery::KvCacheType::F16);
    assert(rookery::parse_kv_cache_type("q8_0") == rookery::KvCacheType::Q8_0);
    assert(rookery::parse_kv_cache_type("q4_0") == rookery::KvCacheType::Q4_0);
    assert(rookery::parse_kv_cache_type("Q4_0") == rookery::KvCacheType::Q4_0);
    assert(std::string(rookery::kv_cache_type_name(rookery::KvCacheType::Q4_0)) == "Q4_0");
    std::printf("[PASS] KvCacheType & BlockQ4_0 unit tests\n");
}

static void test_cpu_quant_kernels() {
    rookery::cpu::ThreadPool pool;
    pool.init(2);

    const int K = 256;
    const int M = 10;
    const int T_list[2] = {1, 16}; // exercises both direct decode and FMA prefill paths!

    std::vector<float> x(16 * K);
    for (int i = 0; i < 16 * K; ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.137f) * 1.5f;
    }

    const rookery::DType dtypes[] = {
        rookery::DType::F32,  rookery::DType::F16,  rookery::DType::BF16,
        rookery::DType::Q8_0, rookery::DType::Q4_0, rookery::DType::Q5_0,
        rookery::DType::Q2_K, rookery::DType::Q3_K, rookery::DType::Q4_K,
        rookery::DType::Q5_K, rookery::DType::Q6_K,
        rookery::DType::IQ4_NL, rookery::DType::IQ4_XS,
        rookery::DType::IQ2_S, rookery::DType::IQ3_S, rookery::DType::IQ3_XXS,
    };

    uint32_t rng = 0x12345678u;
    auto next_byte = [&]() -> uint8_t {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<uint8_t>((rng >> 16) & 0xFFu);
    };

    for (rookery::DType dt : dtypes) {
        const auto tr = rookery::dtype_traits(dt);
        const size_t row_bytes = (static_cast<size_t>(K) / tr.block_elems) * tr.block_bytes;
        std::vector<uint8_t> raw(static_cast<size_t>(M) * row_bytes);
        for (size_t i = 0; i < raw.size(); ++i) raw[i] = next_byte();

        // Populate finite FP16/BF16/F32 scales so no NaN/Inf blocks occur
        for (int r = 0; r < M; ++r) {
            uint8_t * rp = raw.data() + static_cast<size_t>(r) * row_bytes;
            if (dt == rookery::DType::F32) {
                auto * fp = reinterpret_cast<float *>(rp);
                for (int k = 0; k < K; ++k) fp[k] = std::cos(static_cast<float>(r * K + k) * 0.07f);
            } else if (dt == rookery::DType::F16) {
                auto * hp = reinterpret_cast<uint16_t *>(rp);
                for (int k = 0; k < K; ++k) hp[k] = rookery::cpu::f32_to_f16(std::cos(static_cast<float>(r * K + k) * 0.07f));
            } else if (dt == rookery::DType::BF16) {
                auto * bp = reinterpret_cast<uint16_t *>(rp);
                for (int k = 0; k < K; ++k) {
                    float v = std::cos(static_cast<float>(r * K + k) * 0.07f);
                    uint32_t bits; std::memcpy(&bits, &v, 4);
                    bp[k] = static_cast<uint16_t>(bits >> 16);
                }
            } else {
                const int nb = K / static_cast<int>(tr.block_elems);
                for (int b = 0; b < nb; ++b) {
                    uint8_t * bp = rp + static_cast<size_t>(b) * tr.block_bytes;
                    const uint16_t sc = rookery::cpu::f32_to_f16(0.05f + 0.01f * ((r + b) % 7));
                    if (dt == rookery::DType::Q6_K) {
                        std::memcpy(bp + 208, &sc, 2);
                    } else if (dt == rookery::DType::Q3_K) {
                        std::memcpy(bp + 108, &sc, 2);
                    } else if (dt == rookery::DType::Q2_K) {
                        const uint16_t sm = rookery::cpu::f32_to_f16(0.03f + 0.005f * ((r + b) % 5));
                        std::memcpy(bp + 80, &sc, 2);
                        std::memcpy(bp + 82, &sm, 2);
                    } else if (dt == rookery::DType::Q4_K || dt == rookery::DType::Q5_K) {
                        const uint16_t sm = rookery::cpu::f32_to_f16(0.03f + 0.005f * ((r + b) % 5));
                        std::memcpy(bp + 0, &sc, 2);
                        std::memcpy(bp + 2, &sm, 2);
                    } else {
                        std::memcpy(bp + 0, &sc, 2);
                    }
                }
            }
        }

        // Reference FP32 weights via dequant_row_to_f32
        std::vector<float> W_f32(static_cast<size_t>(M) * K);
        for (int r = 0; r < M; ++r) {
            rookery::cpu::dequant_row_to_f32(dt, raw.data() + static_cast<size_t>(r) * row_bytes, K, W_f32.data() + static_cast<size_t>(r) * K);
        }

        rookery::cpu::CpuTensor W{raw.data(), dt, {static_cast<uint64_t>(K), static_cast<uint64_t>(M), 1, 1}, raw.size()};

        for (int T : T_list) {
            std::vector<float> y(static_cast<size_t>(T) * M, 0.0f);
            rookery::cpu::linear(pool, W, x.data(), nullptr, nullptr, y.data(), T);

            float max_rel = 0.0f;
            for (int t = 0; t < T; ++t) {
                for (int r = 0; r < M; ++r) {
                    double ref = 0.0;
                    double norm = 0.0;
                    for (int k = 0; k < K; ++k) {
                        const double wv = W_f32[static_cast<size_t>(r) * K + k];
                        const double xv = x[static_cast<size_t>(t) * K + k];
                        ref  += wv * xv;
                        norm += std::fabs(wv * xv);
                    }
                    const float rel = static_cast<float>(std::fabs(y[static_cast<size_t>(t) * M + r] - ref) / std::max(norm, 1e-6));
                    if (rel > max_rel) max_rel = rel;
                }
            }
            // Float weights and AVX2 FP32-FMA prefill (T>=12) match FP32 ref to ~1e-6;
            // Q8_0/Q8_K activation-quantized paths (decode T=1, and ARM64 sdot / portable prefill) match within < 0.02 relative to L1 norm.
            const bool is_float_dt = (dt == rookery::DType::F32 || dt == rookery::DType::F16 || dt == rookery::DType::BF16);
#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512VNNI__)
            const bool fma_prefill = (T >= 12);
#else
            const bool fma_prefill = false;
#endif
            const float tol = (is_float_dt || fma_prefill) ? 1e-4f : 0.02f;
            if (!(max_rel < tol)) {
                std::fprintf(stderr, "[FAIL] CPU quant kernel %s T=%d max_rel=%g >= %g\n", tr.name, T, max_rel, tol);
                std::abort();
            }
        }
    }

    // Verify GeGLU-tanh and softcap
    std::vector<float> g = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 3.0f, -3.0f, 1.25f, 0.75f};
    std::vector<float> u = { 1.0f,  2.0f, 3.0f, 4.0f, 5.0f, 6.0f,  7.0f, 8.0f,  9.0f};
    std::vector<float> g_ref(g.size());
    for (size_t i = 0; i < g.size(); ++i) {
        g_ref[i] = 0.5f * g[i] * (1.0f + std::tanh(0.7978845608028654f * g[i] * (1.0f + 0.044715f * g[i] * g[i]))) * u[i];
    }
    rookery::cpu::geglu_tanh_mul(g.data(), u.data(), g.size());
    for (size_t i = 0; i < g.size(); ++i) {
        assert(std::fabs(g[i] - g_ref[i]) < 1e-4f);
    }

    std::vector<float> lg = {-100.0f, -15.0f, 0.0f, 15.0f, 100.0f, 30.0f, -30.0f, 5.0f, -5.0f};
    std::vector<float> lg_ref(lg.size());
    for (size_t i = 0; i < lg.size(); ++i) lg_ref[i] = 30.0f * std::tanh(lg[i] / 30.0f);
    rookery::cpu::softcap_tanh(lg.data(), static_cast<int>(lg.size()), 30.0f);
    for (size_t i = 0; i < lg.size(); ++i) {
        assert(std::fabs(lg[i] - lg_ref[i]) < 1e-4f);
    }

    // Verify GQA d128 attention across F16, Q8_0, and Q4_0 KV caches
    {
        const int n_head = 12, n_kv = 2, max_ctx = 32, seq_len = 16;
        std::vector<float> q(seq_len * n_head * 128);
        std::vector<float> k(seq_len * n_kv * 128);
        std::vector<float> v(seq_len * n_kv * 128);
        std::vector<int32_t> pos(seq_len);
        for (int i = 0; i < seq_len; ++i) pos[i] = i;
        for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(static_cast<float>(i) * 0.031f);
        for (size_t i = 0; i < k.size(); ++i) k[i] = std::cos(static_cast<float>(i) * 0.027f);
        for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin(static_cast<float>(i) * 0.019f + 0.5f);

        std::vector<uint16_t> k_f16(n_kv * max_ctx * 128), v_f16(n_kv * max_ctx * 128);
        std::vector<rookery::cpu::CpuGqaKv128Q8Pos> k_q8(n_kv * max_ctx), v_q8(n_kv * max_ctx);
        std::vector<rookery::cpu::CpuGqaKv128Q4Pos> k_q4(n_kv * max_ctx), v_q4(n_kv * max_ctx);
        const float sc = 1.0f / std::sqrt(128.0f);

        std::vector<float> out_f16(seq_len * n_head * 128), out_q8(out_f16.size()), out_q4(out_f16.size());
        rookery::cpu::gqa_kv_write_d128(k.data(), v.data(), pos.data(), k_f16.data(), v_f16.data(), rookery::KvCacheType::F16, seq_len, n_kv, max_ctx);
        rookery::cpu::gqa_attn_d128(pool, q.data(), k_f16.data(), v_f16.data(), rookery::KvCacheType::F16, pos.data(), out_f16.data(), seq_len, n_head, n_kv, max_ctx, sc);

        rookery::cpu::gqa_kv_write_d128(k.data(), v.data(), pos.data(), k_q8.data(), v_q8.data(), rookery::KvCacheType::Q8_0, seq_len, n_kv, max_ctx);
        rookery::cpu::gqa_attn_d128(pool, q.data(), k_q8.data(), v_q8.data(), rookery::KvCacheType::Q8_0, pos.data(), out_q8.data(), seq_len, n_head, n_kv, max_ctx, sc);

        rookery::cpu::gqa_kv_write_d128(k.data(), v.data(), pos.data(), k_q4.data(), v_q4.data(), rookery::KvCacheType::Q4_0, seq_len, n_kv, max_ctx);
        rookery::cpu::gqa_attn_d128(pool, q.data(), k_q4.data(), v_q4.data(), rookery::KvCacheType::Q4_0, pos.data(), out_q4.data(), seq_len, n_head, n_kv, max_ctx, sc);

        float max_e_q8 = 0.0f, max_e_q4 = 0.0f;
        for (size_t i = 0; i < out_f16.size(); ++i) {
            max_e_q8 = std::max(max_e_q8, std::fabs(out_f16[i] - out_q8[i]));
            max_e_q4 = std::max(max_e_q4, std::fabs(out_f16[i] - out_q4[i]));
        }
        assert(max_e_q8 < 0.02f);
        assert(max_e_q4 < 0.15f);
        std::printf("[PASS] CPU GQA d128 attention (F16 vs Q8_0 max_err=%.4f, Q4_0 max_err=%.4f)\n", max_e_q8, max_e_q4);
    }

    // Verify Gemma 4 thought-channel prompt rendering, non-streaming parse, and byte-by-byte SSE splitter
    {
        rookery::json msgs = rookery::json::array({
            {{"role", "user"}, {"content", "Hi"}},
        });
        const std::string p_off = rookery::render_gemma4_chat_prompt(msgs, false);
        assert(p_off.find("<|turn>model\n") != std::string::npos);
        assert(p_off.find("<|channel>thought") == std::string::npos);
        const std::string p_on = rookery::render_gemma4_chat_prompt(msgs, true);
        assert(p_on.find("<|turn>model\n<|channel>thought\n") != std::string::npos);
        assert(p_on.find("<channel|>") == std::string::npos);

        auto p1 = rookery::parse_gemma4_output("<|channel>thought\n<channel|>The Seine flows through Paris.", false);
        assert(p1.reasoning_content.empty());
        assert(p1.content == "The Seine flows through Paris.");

        auto p2 = rookery::parse_gemma4_output("step 1\nstep 2<channel|>\nAnswer.", true);
        assert(p2.reasoning_content == "step 1\nstep 2");
        assert(p2.content == "Answer.");

        // Fuzz byte-by-byte streaming split (including empty thought emitted when enable_thinking=false!)
        auto run_stream = [](const std::string & raw, bool started_in_think, size_t chunk_sz) {
            rookery::Gemma4StreamSplitter sp(started_in_think);
            std::string pending, out_c, out_r;
            auto safe_pfx = [](const std::string &, size_t n) { return n; };
            auto emit = [&](const char * field, const std::string & s) -> bool {
                if (std::string(field) == "content") out_c += s;
                else out_r += s;
                return true;
            };
            for (size_t i = 0; i < raw.size(); i += chunk_sz) {
                pending += raw.substr(i, std::min(chunk_sz, raw.size() - i));
                assert(sp.flush(pending, false, 0, emit, safe_pfx));
            }
            assert(sp.flush(pending, true, 0, emit, safe_pfx));
            return std::make_pair(out_r, out_c);
        };
        for (size_t csz : {size_t(1), size_t(3), size_t(7), size_t(32)}) {
            auto [r0, c0] = run_stream("<|channel>thought\n<channel|>The Seine flows through Paris.", false, csz);
            assert(r0.empty());
            assert(c0 == "The Seine flows through Paris.");

            auto [r1, c1] = run_stream("plan A<channel|>\nFinal reply.", true, csz);
            assert(r1 == "plan A");
            assert(c1 == "Final reply.");

            auto [r2, c2] = run_stream("Prefix <|channel>thought\nsecret<channel|>Suffix.", false, csz);
            assert(r2 == "secret");
            assert(c2 == "Prefix Suffix.");
        }
    }

    // Verify CPU Q4_0 KV cache accuracy vs F16 reference (Gemma 4 GQA + BailingMoE3 MLA)
    {
        const int n_head = 8, n_kv = 2, hd = 256, max_slots = 64, seq_len = 24;
        std::vector<float> q(static_cast<size_t>(n_head) * hd);
        std::vector<float> k_seq(static_cast<size_t>(seq_len) * n_kv * hd);
        std::vector<float> v_seq(static_cast<size_t>(seq_len) * n_kv * hd);
        std::vector<int32_t> pos(seq_len);
        for (int i = 0; i < seq_len; ++i) pos[i] = i;
        for (size_t i = 0; i < q.size(); ++i)     q[i]     = 0.3f * std::sin(static_cast<float>(i) * 0.13f);
        for (size_t i = 0; i < k_seq.size(); ++i) k_seq[i] = 0.4f * std::cos(static_cast<float>(i) * 0.09f);
        for (size_t i = 0; i < v_seq.size(); ++i) v_seq[i] = 0.5f * std::sin(static_cast<float>(i) * 0.07f + 0.3f);

        const size_t f16_bytes = static_cast<size_t>(max_slots) * n_kv * hd * sizeof(uint16_t);
        const size_t q8_stride = static_cast<size_t>(hd) + (((static_cast<size_t>(hd >> 5) * 2u) + 31u) & ~size_t(31));
        const size_t q4_stride = static_cast<size_t>(hd >> 1) + (((static_cast<size_t>(hd >> 5) * 2u) + 31u) & ~size_t(31));
        std::vector<uint8_t> kf(f16_bytes, 0), vf(f16_bytes, 0);
        std::vector<uint8_t> k8(static_cast<size_t>(max_slots) * n_kv * q8_stride, 0), v8(k8.size(), 0);
        std::vector<uint8_t> k4(static_cast<size_t>(max_slots) * n_kv * q4_stride, 0), v4(k4.size(), 0);

        rookery::cpu::gemma4_vnorm_kv_write(k_seq.data(), v_seq.data(), kf.data(), vf.data(), rookery::KvCacheType::F16,  pos.data(), seq_len, n_kv, hd, max_slots, false, 1e-6f);
        rookery::cpu::gemma4_vnorm_kv_write(k_seq.data(), v_seq.data(), k8.data(), v8.data(), rookery::KvCacheType::Q8_0, pos.data(), seq_len, n_kv, hd, max_slots, false, 1e-6f);
        rookery::cpu::gemma4_vnorm_kv_write(k_seq.data(), v_seq.data(), k4.data(), v4.data(), rookery::KvCacheType::Q4_0, pos.data(), seq_len, n_kv, hd, max_slots, false, 1e-6f);

        int32_t qpos = seq_len - 1;
        std::vector<float> out_f(q.size()), out_8(q.size()), out_4(q.size());
        rookery::cpu::gemma4_gqa_attn(pool, q.data(), kf.data(), vf.data(), rookery::KvCacheType::F16,  &qpos, out_f.data(), 1, n_head, n_kv, hd, max_slots, 0);
        rookery::cpu::gemma4_gqa_attn(pool, q.data(), k8.data(), v8.data(), rookery::KvCacheType::Q8_0, &qpos, out_8.data(), 1, n_head, n_kv, hd, max_slots, 0);
        rookery::cpu::gemma4_gqa_attn(pool, q.data(), k4.data(), v4.data(), rookery::KvCacheType::Q4_0, &qpos, out_4.data(), 1, n_head, n_kv, hd, max_slots, 0);

        auto cos_sim = [](const std::vector<float> & a, const std::vector<float> & b) {
            double dot = 0, na = 0, nb = 0;
            for (size_t i = 0; i < a.size(); ++i) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
            return dot / (std::sqrt(na * nb) + 1e-12);
        };
        assert(cos_sim(out_f, out_8) > 0.999);
        assert(cos_sim(out_f, out_4) > 0.970);

        // MLA Q4_0 KV accuracy vs F16
        const int mla_h = 8;
        std::vector<float> mla_kv_in(static_cast<size_t>(seq_len) * 576);
        std::vector<float> mla_nw(512, 1.0f);
        std::vector<float> mla_q(static_cast<size_t>(mla_h) * 576);
        for (size_t i = 0; i < mla_kv_in.size(); ++i) mla_kv_in[i] = 0.4f * std::cos(static_cast<float>(i) * 0.05f);
        for (size_t i = 0; i < mla_q.size(); ++i)     mla_q[i]     = 0.3f * std::sin(static_cast<float>(i) * 0.11f);

        std::vector<uint16_t> mla_f(static_cast<size_t>(max_slots) * 576, 0);
        std::vector<rookery::cpu::CpuMlaKv576Q8Pos> mla_8(max_slots);
        std::vector<rookery::cpu::CpuMlaKv576Q4Pos> mla_4(max_slots);
        rookery::cpu::mla_kv_norm_rope_write_576(mla_kv_in.data(), mla_nw.data(), pos.data(), mla_f.data(), rookery::KvCacheType::F16,  seq_len, 1e-6f, 10000.0f);
        rookery::cpu::mla_kv_norm_rope_write_576(mla_kv_in.data(), mla_nw.data(), pos.data(), mla_8.data(), rookery::KvCacheType::Q8_0, seq_len, 1e-6f, 10000.0f);
        rookery::cpu::mla_kv_norm_rope_write_576(mla_kv_in.data(), mla_nw.data(), pos.data(), mla_4.data(), rookery::KvCacheType::Q4_0, seq_len, 1e-6f, 10000.0f);

        std::vector<float> mo_f(static_cast<size_t>(mla_h) * 512), mo_8(mo_f.size()), mo_4(mo_f.size());
        const float sc = 1.0f / std::sqrt(192.0f);
        rookery::cpu::mla_attn_d576_v512(pool, mla_q.data(), mla_f.data(), rookery::KvCacheType::F16,  &qpos, mo_f.data(), 1, mla_h, sc);
        rookery::cpu::mla_attn_d576_v512(pool, mla_q.data(), mla_8.data(), rookery::KvCacheType::Q8_0, &qpos, mo_8.data(), 1, mla_h, sc);
        rookery::cpu::mla_attn_d576_v512(pool, mla_q.data(), mla_4.data(), rookery::KvCacheType::Q4_0, &qpos, mo_4.data(), 1, mla_h, sc);
        assert(cos_sim(mo_f, mo_8) > 0.999);
        assert(cos_sim(mo_f, mo_4) > 0.990);
    }

    std::printf("[PASS] CPU quantized kernels (16 dtypes, decode & prefill), Q4_0 KV, Gemma 4 stream filter, GeGLU-tanh, softcap unit tests\n");
}

static void test_raon1_chat_template() {
    using json = nlohmann::json;
    json msgs = json::array({
        {{"role", "system"}, {"content", "Be helpful."}},
        {{"role", "user"}, {"content", "Hi"}},
        {{"role", "assistant"}, {"reasoning_content", "plan"}, {"content", "Hello!"}},
        {{"role", "user"}, {"content", "Next"}}
    });
    const std::string p_think = rookery::render_raon1_chat_prompt(msgs, json::array(), true);
    assert(p_think.find("<|im_start|>system\nBe helpful.<|im_end|>\n") != std::string::npos);
    assert(p_think.find("<|im_start|>assistant\n<think>\nplan\n</think>\n\nHello!<|im_end|>\n") != std::string::npos);
    assert(p_think.ends_with("<|im_start|>assistant\n<think>\n"));

    const std::string p_nothink = rookery::render_raon1_chat_prompt(msgs, json::array(), false);
    assert(p_nothink.ends_with("<|im_start|>assistant\n<think></think>\n\n"));

    auto parsed = rookery::parse_maxwell_output("thought\n</think>\n\nAnswer", true);
    assert(parsed.reasoning_content == "thought");
    assert(parsed.content == "Answer");
    std::printf("[PASS] Raon-1-mini chat template unit tests\n");
}

static void test_gpt4o_inkling2_pretokenizer() {
    // Verify GPT-4o / o200k_base / Inkling-2 pretokenizer preserves punctuation+newline
    // tokens (e.g. "):\n"), whitespace+newline runs, CamelCase splits, and contractions.
    const auto p1 = rookery::BpeTokenizer::pretokenize_gpt4o("def fooBar(x=123456):\n    return don't\n\n");
    const std::vector<std::string> exp1 = {
        "def", " foo", "Bar", "(x", "=", "123", "456", "):\n", "   ", " return", " don't", "\n\n"
    };
    assert(p1 == exp1);

    // Trailing '/' attaches to punctuation; CJK + ASCII uppercase splits cleanly
    const auto p2 = rookery::BpeTokenizer::pretokenize_gpt4o("a // b\r\n");
    const std::vector<std::string> exp2 = {"a", " //", " b", "\r\n"};
    assert(p2 == exp2);

    std::printf("[PASS] GPT-4o / Inkling-2 pretokenizer unit tests\n");
}

int main(int argc, char ** argv) {
    test_chat_parser();
    test_raon1_chat_template();
    test_gpt4o_inkling2_pretokenizer();
    test_engram_rkad_v3_contract();
    test_kv_cache_types();
    test_cpu_quant_kernels();

    // Optional tokenizer check mode: rookery_unit_test <model.gguf> <text1> [text2 ...]
    if (argc >= 3) {
        auto gguf = rookery::GgufFile::open(argv[1]);
        rookery::BpeTokenizer tok;
        tok.load(gguf);
        assert(tok.is_special_token(tok.bos_id()) || tok.token_type(tok.bos_id()) != 1);
        for (int i = 2; i < argc; ++i) {
            auto ids = tok.encode(argv[i], true, true);
            std::printf("TOK[%d]:", i - 2);
            for (int32_t id : ids) std::printf(" %d%s", id, tok.is_special_token(id) ? "*" : "");
            std::printf("\n");
        }
    }
    return 0;
}
