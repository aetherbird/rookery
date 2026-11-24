#include "engram/engram_sidecar.h"
#include "gguf/reader.h"
#include "server/chat_template.h"
#include "tokenizer/bpe.h"

#include <cassert>
#include <cstdio>
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

    // 3. Per-sequence rollback_to preserves exact byte ends and safely rewinds committed cursor
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

    std::printf("[PASS] RKAD v3 manifest contract & EngramSeqState unit tests\n");
}

int main(int argc, char ** argv) {
    test_chat_parser();
    test_engram_rkad_v3_contract();

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
