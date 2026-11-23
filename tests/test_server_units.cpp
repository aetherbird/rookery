#include "gguf/reader.h"
#include "server/chat_template.h"
#include "tokenizer/bpe.h"

#include <cassert>
#include <cstdio>
#include <iostream>
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

int main(int argc, char ** argv) {
    test_chat_parser();

    // Optional tokenizer check mode: rookery_unit_test <model.gguf> <text1> [text2 ...]
    if (argc >= 3) {
        auto gguf = rookery::GgufFile::open(argv[1]);
        rookery::BpeTokenizer tok;
        tok.load(gguf);
        for (int i = 2; i < argc; ++i) {
            auto ids = tok.encode(argv[i], true, true);
            std::printf("TOK[%d]:", i - 2);
            for (int32_t id : ids) std::printf(" %d", id);
            std::printf("\n");
        }
    }
    return 0;
}
