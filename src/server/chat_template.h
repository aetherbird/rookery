#pragma once

#include "nlohmann/json.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace rookery {

using json = nlohmann::json;

// Generates a globally unique OpenAI tool_call_id across turns, requests, and restarts.
inline std::string next_tool_call_id() {
    static const uint64_t boot_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    static std::atomic<uint64_t> seq{0};
    const uint64_t n = seq.fetch_add(1, std::memory_order_relaxed) + 1;
    char buf[48];
    std::snprintf(buf, sizeof(buf), "call_rk_%llx_%llu",
                  static_cast<unsigned long long>(boot_us & 0xFFFFFFFFULL),
                  static_cast<unsigned long long>(n));
    return std::string(buf);
}

struct ParsedAssistantResponse {
    std::string reasoning_content;
    std::string content;
    json        tool_calls = json::array();
};

// Renders OpenAI chat messages (+ optional tools) into exact Maxwell-V1-Flash prompt format.
inline std::string render_maxwell_chat_prompt(const json & messages,
                                              const json & tools,
                                              bool enable_thinking = true) {
    std::string out;

    bool has_tools = tools.is_array() && !tools.empty();
    bool first_is_sys = messages.is_array() && !messages.empty() && messages[0].value("role", "") == "system";

    auto extract_text = [](const json & c) -> std::string {
        if (c.is_string()) return c.get<std::string>();
        if (c.is_array()) {
            std::string s;
            for (const auto & part : c) {
                if (part.is_object() && part.value("type", "") == "text") {
                    s += part.value("text", "");
                }
            }
            return s;
        }
        return "";
    };

    if (has_tools) {
        out += "<|maxwell_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n";
        for (const auto & tool : tools) {
            const json & fn = tool.contains("function") ? tool["function"] : tool;
            out += "<function>\n";
            out += "<name>" + fn.value("name", "") + "</name>\n";
            if (fn.contains("description")) {
                out += "<description>" + fn.value("description", "") + "</description>\n";
            }
            if (fn.contains("parameters")) {
                out += "<parameters>\n" + fn["parameters"].dump() + "\n</parameters>\n";
            }
            out += "</function>\n";
        }
        out += "</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
               "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>\n";
        if (first_is_sys) {
            std::string sys = extract_text(messages[0]["content"]);
            if (!sys.empty()) out += "\n\n" + sys;
        }
        out += "<|maxwell_end|>\n";
    } else if (first_is_sys) {
        out += "<|maxwell_start|>system\n" + extract_text(messages[0]["content"]) + "<|maxwell_end|>\n";
    }

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & m = messages[i];
        const std::string role = m.value("role", "");
        if (role == "system") continue;

        if (role == "user") {
            out += "<|maxwell_start|>user\n" + extract_text(m["content"]) + "<|maxwell_end|>\n";
        } else if (role == "assistant") {
            std::string rc  = m.contains("reasoning_content") && m["reasoning_content"].is_string()
                                  ? m["reasoning_content"].get<std::string>() : "";
            std::string cnt = m.contains("content") ? extract_text(m["content"]) : "";
            out += "<|maxwell_start|>assistant\n<think>\n" + rc + "\n</think>\n\n" + cnt;
            if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                for (size_t ti = 0; ti < m["tool_calls"].size(); ++ti) {
                    const auto & tc = m["tool_calls"][ti];
                    const auto & fn = tc.contains("function") ? tc["function"] : tc;
                    if (ti == 0 && !cnt.empty()) out += "\n\n";
                    else if (ti > 0) out += "\n";
                    out += "<tool_call>\n<function=" + fn.value("name", "") + ">\n";
                    json args = json::object();
                    if (fn.contains("arguments")) {
                        if (fn["arguments"].is_string()) {
                            try { args = json::parse(fn["arguments"].get<std::string>()); } catch (...) {}
                        } else if (fn["arguments"].is_object()) {
                            args = fn["arguments"];
                        }
                    }
                    for (auto it = args.begin(); it != args.end(); ++it) {
                        out += "<parameter=" + it.key() + ">\n";
                        out += it.value().is_string() ? it.value().get<std::string>() : it.value().dump();
                        out += "\n</parameter>\n";
                    }
                    out += "</function>\n</tool_call>";
                }
            }
            out += "<|maxwell_end|>\n";
        } else if (role == "tool") {
            bool prev_tool = (i > 0 && messages[i - 1].value("role", "") == "tool");
            bool next_tool = (i + 1 < messages.size() && messages[i + 1].value("role", "") == "tool");
            if (!prev_tool) out += "<|maxwell_start|>user";
            out += "\n<tool_response>\n" + extract_text(m["content"]) + "\n</tool_response>";
            if (!next_tool) out += "<|maxwell_end|>\n";
        }
    }

    out += "<|maxwell_start|>assistant\n";
    if (!enable_thinking) {
        out += "<think>\n\n</think>\n\n";
    } else {
        out += "<think>\n";
    }
    return out;
}

// Parses Maxwell output into reasoning_content, content, and tool_calls.
// Crucially fixes upstream llama.cpp's qwen3-coder.cpp XML leak: if the model omits </parameter>
// before the next <parameter=...> tag, we cleanly split at the next <parameter= boundary!
inline ParsedAssistantResponse parse_maxwell_output(const std::string & raw, bool started_in_think = true) {
    ParsedAssistantResponse res;
    std::string rest = raw;

    if (started_in_think) {
        size_t end_think = rest.find("</think>");
        size_t tc_think  = rest.find("<tool_call>");
        if (end_think != std::string::npos && (tc_think == std::string::npos || end_think < tc_think)) {
            res.reasoning_content = rest.substr(0, end_think);
            while (!res.reasoning_content.empty() && (res.reasoning_content.back() == '\n' || res.reasoning_content.back() == ' ')) {
                res.reasoning_content.pop_back();
            }
            rest = rest.substr(end_think + 8);
            while (!rest.empty() && rest.front() == '\n') rest.erase(rest.begin());
        } else if (tc_think != std::string::npos) {
            res.reasoning_content = rest.substr(0, tc_think);
            while (!res.reasoning_content.empty() && (res.reasoning_content.back() == '\n' || res.reasoning_content.back() == ' ')) {
                res.reasoning_content.pop_back();
            }
            rest = rest.substr(tc_think);
        } else {
            // Output hit max_tokens (or stopped) before closing </think>: everything emitted so far is reasoning!
            res.reasoning_content = rest;
            res.content.clear();
            return res;
        }
    }

    size_t tc_pos = rest.find("<tool_call>");
    if (tc_pos == std::string::npos) {
        res.content = rest;
        return res;
    }

    res.content = rest.substr(0, tc_pos);
    while (!res.content.empty() && (res.content.back() == '\n' || res.content.back() == ' ')) {
        res.content.pop_back();
    }

    size_t cursor = tc_pos;
    while ((cursor = rest.find("<function=", cursor)) != std::string::npos) {
        size_t name_end = rest.find('>', cursor + 10);
        if (name_end == std::string::npos) break;
        std::string fn_name = rest.substr(cursor + 10, name_end - (cursor + 10));

        size_t fn_end = rest.find("</function>", name_end);
        if (fn_end == std::string::npos) fn_end = rest.find("</tool_call>", name_end);
        if (fn_end == std::string::npos) fn_end = rest.size();

        std::string body = rest.substr(name_end + 1, fn_end - (name_end + 1));
        json args = json::object();

        size_t pcur = 0;
        while ((pcur = body.find("<parameter=", pcur)) != std::string::npos) {
            size_t pname_end = body.find('>', pcur + 11);
            if (pname_end == std::string::npos) break;
            std::string pname = body.substr(pcur + 11, pname_end - (pcur + 11));
            size_t val_start = pname_end + 1;
            if (val_start < body.size() && body[val_start] == '\n') ++val_start;

            // Stop at </parameter> OR next <parameter= (prevents upstream XML leak bug!)
            size_t pclose = body.find("</parameter>", val_start);
            size_t pnext  = body.find("<parameter=", val_start);
            size_t val_end = std::min(pclose, pnext);
            if (val_end == std::string::npos) val_end = body.size();

            std::string val = body.substr(val_start, val_end - val_start);
            while (!val.empty() && val.back() == '\n') val.pop_back();

            try {
                if (!val.empty() && (val.front() == '{' || val.front() == '[' || val == "true" || val == "false" || val == "null")) {
                    args[pname] = json::parse(val);
                } else {
                    args[pname] = val;
                }
            } catch (...) {
                args[pname] = val;
            }

            pcur = (pclose != std::string::npos && pclose <= pnext) ? (pclose + 12) : pnext;
            if (pcur == std::string::npos) break;
        }

        res.tool_calls.push_back({
            {"id", next_tool_call_id()},
            {"type", "function"},
            {"function", {
                {"name", fn_name},
                {"arguments", args.dump()}
            }}
        });
        cursor = fn_end;
    }

    return res;
}

// Renders OpenAI chat messages (+ optional tools) into exact GLM-5.3-Flash prompt format.
inline std::string render_glm5_chat_prompt(const json & messages,
                                           const json & tools,
                                           bool enable_thinking = true,
                                           const std::string & reasoning_effort = "max") {
    auto extract_text = [](const json & c) -> std::string {
        if (c.is_string()) return c.get<std::string>();
        if (c.is_array()) {
            std::string s;
            for (const auto & part : c) {
                if (part.is_object() && part.value("type", "") == "text") s += part.value("text", "");
                else if (part.is_string()) s += part.get<std::string>();
            }
            return s;
        }
        return "";
    };

    std::string eff = "Max";
    if (reasoning_effort == "low") eff = "Low";
    else if (reasoning_effort == "high") eff = "High";

    std::string out = "[gMASK]<sop>\n<|system|>Reasoning Effort: " + eff;

    if (tools.is_array() && !tools.empty()) {
        out += "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
        for (const auto & tool : tools) {
            const json & fn = tool.contains("function") ? tool["function"] : tool;
            out += fn.dump() + "\n";
        }
        out += "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n"
               "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
               "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";
    }

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & m = messages[i];
        const std::string role = m.value("role", "");
        const std::string text = m.contains("content") ? extract_text(m["content"]) : "";
        if (role == "system") {
            out += "<|system|>" + text;
        } else if (role == "user") {
            out += "<|user|>" + text;
        } else if (role == "assistant") {
            std::string rc = m.contains("reasoning_content") && m["reasoning_content"].is_string()
                                 ? m["reasoning_content"].get<std::string>() : "";
            out += "<|assistant|><think>" + rc + "</think>" + text;
            if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                for (const auto & tc : m["tool_calls"]) {
                    const auto & fn = tc.contains("function") ? tc["function"] : tc;
                    out += "<tool_call>" + fn.value("name", "");
                    json args = json::object();
                    if (fn.contains("arguments")) {
                        if (fn["arguments"].is_string()) {
                            try { args = json::parse(fn["arguments"].get<std::string>()); } catch (...) {}
                        } else if (fn["arguments"].is_object()) {
                            args = fn["arguments"];
                        }
                    }
                    for (auto it = args.begin(); it != args.end(); ++it) {
                        out += "<arg_key>" + it.key() + "</arg_key><arg_value>";
                        out += it.value().is_string() ? it.value().get<std::string>() : it.value().dump();
                        out += "</arg_value>";
                    }
                    out += "</tool_call>";
                }
            }
        } else if (role == "tool") {
            bool prev_tool = (i > 0 && messages[i - 1].value("role", "") == "tool");
            if (!prev_tool) out += "<|observation|>";
            out += "<tool_response>" + text + "</tool_response>";
        }
    }

    out += "<|assistant|>";
    if (enable_thinking) out += "<think>";
    else out += "<think></think>";
    return out;
}

// Parses GLM-5.3-Flash output into reasoning_content, content, and <tool_call>fn<arg_key>...</tool_call>.
inline ParsedAssistantResponse parse_glm5_output(const std::string & raw, bool started_in_think = true) {
    ParsedAssistantResponse res;
    std::string rest = raw;

    if (started_in_think) {
        size_t end_think = rest.find("</think>");
        size_t tc_think  = rest.find("<tool_call>");
        if (end_think != std::string::npos && (tc_think == std::string::npos || end_think < tc_think)) {
            res.reasoning_content = rest.substr(0, end_think);
            while (!res.reasoning_content.empty() && (res.reasoning_content.back() == '\n' || res.reasoning_content.back() == ' ')) {
                res.reasoning_content.pop_back();
            }
            rest = rest.substr(end_think + 8);
            while (!rest.empty() && rest.front() == '\n') rest.erase(rest.begin());
        } else if (tc_think != std::string::npos) {
            res.reasoning_content = rest.substr(0, tc_think);
            while (!res.reasoning_content.empty() && (res.reasoning_content.back() == '\n' || res.reasoning_content.back() == ' ')) {
                res.reasoning_content.pop_back();
            }
            rest = rest.substr(tc_think);
        } else {
            res.reasoning_content = rest;
            return res;
        }
    }

    size_t tc_pos = rest.find("<tool_call>");
    if (tc_pos == std::string::npos) {
        res.content = rest;
        return res;
    }

    res.content = rest.substr(0, tc_pos);
    while (!res.content.empty() && (res.content.back() == '\n' || res.content.back() == ' ')) {
        res.content.pop_back();
    }

    size_t cursor = tc_pos;
    while ((cursor = rest.find("<tool_call>", cursor)) != std::string::npos) {
        size_t body_start = cursor + 11;
        size_t tc_end = rest.find("</tool_call>", body_start);
        if (tc_end == std::string::npos) tc_end = rest.size();
        std::string body = rest.substr(body_start, tc_end - body_start);

        size_t first_key = body.find("<arg_key>");
        std::string fn_name = (first_key == std::string::npos) ? body : body.substr(0, first_key);
        while (!fn_name.empty() && (fn_name.front() == '\n' || fn_name.front() == ' ')) fn_name.erase(fn_name.begin());
        while (!fn_name.empty() && (fn_name.back() == '\n' || fn_name.back() == ' ')) fn_name.pop_back();

        json args = json::object();
        size_t kcur = (first_key == std::string::npos) ? body.size() : first_key;
        while ((kcur = body.find("<arg_key>", kcur)) != std::string::npos) {
            size_t kend = body.find("</arg_key>", kcur + 9);
            if (kend == std::string::npos) break;
            std::string k = body.substr(kcur + 9, kend - (kcur + 9));
            size_t vstart = body.find("<arg_value>", kend + 10);
            if (vstart == std::string::npos) break;
            vstart += 11;
            size_t vend = body.find("</arg_value>", vstart);
            if (vend == std::string::npos) vend = body.size();
            std::string v = body.substr(vstart, vend - vstart);
            try {
                if (!v.empty() && (v.front() == '{' || v.front() == '[' || v == "true" || v == "false" || v == "null")) {
                    args[k] = json::parse(v);
                } else {
                    args[k] = v;
                }
            } catch (...) {
                args[k] = v;
            }
            kcur = (vend < body.size()) ? (vend + 12) : body.size();
        }

        res.tool_calls.push_back({
            {"id", next_tool_call_id()},
            {"type", "function"},
            {"function", {
                {"name", fn_name},
                {"arguments", args.dump()}
            }}
        });
        cursor = (tc_end < rest.size()) ? (tc_end + 12) : rest.size();
    }

    return res;
}

// Renders OpenAI chat messages (+ optional tools) into exact Inkling 2 Mini prompt format.
inline std::string render_inkling2_chat_prompt(const json & messages,
                                               const json & tools,
                                               bool enable_thinking = true,
                                               const std::string & reasoning_effort = "high") {
    auto extract_text = [](const json & c) -> std::string {
        if (c.is_string()) return c.get<std::string>();
        if (c.is_array()) {
            std::string s;
            for (const auto & part : c) {
                if (part.is_string()) s += part.get<std::string>();
                else if (part.is_object()) {
                    const std::string tp = part.value("type", "text");
                    if (tp == "text" || tp == "input_text") s += part.value("text", "");
                }
            }
            return s;
        }
        return "";
    };

    auto emit_effort = [&]() -> std::string {
        std::string num = "0.9";
        if (!enable_thinking) {
            num = "0";
        } else if (reasoning_effort == "none") {
            num = "0";
        } else if (reasoning_effort == "minimal") {
            num = "0.1";
        } else if (reasoning_effort == "low") {
            num = "0.2";
        } else if (reasoning_effort == "medium") {
            num = "0.7";
        } else if (reasoning_effort == "high") {
            num = "0.9";
        } else if (reasoning_effort == "xhigh" || reasoning_effort == "max") {
            num = "0.99";
        }
        return "<|message_system|><|content_text|>Thinking effort level: " + num + "<|end_message|>";
    };

    std::string out;
    if (tools.is_array() && !tools.empty()) {
        json specs = json::array();
        for (const auto & tool : tools) {
            const json & fn = tool.contains("function") ? tool["function"] : tool;
            specs.push_back({
                {"description", fn.contains("description") && fn["description"].is_string() ? fn["description"].get<std::string>() : ""},
                {"name", fn.value("name", "")},
                {"parameters", fn.contains("parameters") && fn["parameters"].is_object() ? fn["parameters"] : json::object()},
                {"type", tool.contains("type") && tool["type"].is_string() ? tool["type"].get<std::string>() : "function"}
            });
        }
        out += "<|message_system|>tool_declare<|content_xml|>" + specs.dump() + "<|end_message|>";
    }

    bool effort_emitted = false;
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & m = messages[i];
        const std::string role = m.value("role", "");
        if (!effort_emitted && role != "system") {
            out += emit_effort();
            effort_emitted = true;
        }

        std::string rtok;
        if (role == "system")         rtok = "<|message_system|>";
        else if (role == "user")      rtok = "<|message_user|>";
        else if (role == "assistant") {
            rtok = "<|message_model|>";
            if (m.contains("name") && m["name"].is_string() && !m["name"].get<std::string>().empty()) {
                rtok += m["name"].get<std::string>();
            }
        } else if (role == "tool") {
            rtok = "<|message_tool|>";
            if (m.contains("name") && m["name"].is_string() && !m["name"].get<std::string>().empty()) {
                rtok += m["name"].get<std::string>();
            } else if (m.contains("tool_call_id") && m["tool_call_id"].is_string()) {
                const std::string tcid = m["tool_call_id"].get<std::string>();
                for (const auto & prev : messages) {
                    if (prev.value("role", "") == "assistant" && prev.contains("tool_calls") && prev["tool_calls"].is_array()) {
                        for (const auto & tc : prev["tool_calls"]) {
                            if (tc.value("id", "") == tcid && tc.contains("function") && tc["function"].contains("name")) {
                                rtok += tc["function"].value("name", "");
                                break;
                            }
                        }
                    }
                }
            }
        } else {
            continue;
        }

        const std::string rc = (role == "assistant" && m.contains("reasoning_content") && m["reasoning_content"].is_string())
                                   ? m["reasoning_content"].get<std::string>() : "";
        const std::string txt = m.contains("content") ? extract_text(m["content"]) : "";
        const bool has_tc = (role == "assistant" && m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty());

        if (!rc.empty()) {
            out += rtok + "<|content_thinking|>" + rc + "<|end_message|>";
        }
        if (!txt.empty() || !(role == "assistant" && (has_tc || !rc.empty()))) {
            out += rtok + "<|content_text|>" + txt + "<|end_message|>";
        }
        if (has_tc) {
            for (const auto & tc : m["tool_calls"]) {
                const auto & fn = tc.contains("function") ? tc["function"] : tc;
                const std::string fname = fn.value("name", "");
                json args = json::object();
                if (fn.contains("arguments")) {
                    if (fn["arguments"].is_string()) {
                        try { args = json::parse(fn["arguments"].get<std::string>()); } catch (...) {}
                    } else if (fn["arguments"].is_object()) {
                        args = fn["arguments"];
                    }
                }
                out += "<|message_model|>" + fname + "<|content_invoke_tool_json|>{\"name\":" +
                       json(fname).dump() + ",\"args\":" + args.dump() + "}<|end_message|>";
            }
        }
        if (role == "assistant") {
            out += "<|content_model_end_sampling|>";
        }
    }

    if (!effort_emitted) {
        out += emit_effort();
    }
    if (!enable_thinking) {
        out += "<|message_model|><|content_text|>";
    }
    return out;
}

// Parses Inkling 2 Mini output blocks (<|message_model|><|content_thinking|>/<|content_text|>/<|content_invoke_tool_json|>...<|end_message|>).
inline ParsedAssistantResponse parse_inkling2_output(const std::string & raw) {
    ParsedAssistantResponse res;
    size_t pos = 0;
    const std::string kThink = "<|content_thinking|>";
    const std::string kText  = "<|content_text|>";
    const std::string kTool  = "<|content_invoke_tool_json|>";
    const std::string kEnd   = "<|end_message|>";

    while (pos < raw.size()) {
        size_t p_th = raw.find(kThink, pos);
        size_t p_tx = raw.find(kText,  pos);
        size_t p_tl = raw.find(kTool,  pos);
        size_t p_min = std::min(p_th, std::min(p_tx, p_tl));
        if (p_min == std::string::npos) {
            // Fallback: plain text with no Inkling content tags
            if (pos == 0 && res.reasoning_content.empty() && res.content.empty() && res.tool_calls.empty()) {
                res.content = raw;
            }
            break;
        }
        int kind = (p_min == p_th) ? 0 : ((p_min == p_tx) ? 1 : 2);
        size_t body_start = p_min + (kind == 0 ? kThink.size() : (kind == 1 ? kText.size() : kTool.size()));
        size_t end_p = raw.find(kEnd, body_start);
        size_t body_end = (end_p == std::string::npos) ? raw.size() : end_p;
        std::string body = raw.substr(body_start, body_end - body_start);

        if (kind == 0) {
            res.reasoning_content += body;
        } else if (kind == 1) {
            res.content += body;
        } else {
            try {
                json j = json::parse(body);
                std::string fname = j.value("name", "");
                json args = j.contains("args") ? j["args"] : (j.contains("arguments") ? j["arguments"] : json::object());
                std::string args_str = args.is_string() ? args.get<std::string>() : args.dump();
                res.tool_calls.push_back({
                    {"id", next_tool_call_id()},
                    {"type", "function"},
                    {"function", {{"name", fname}, {"arguments", args_str}}}
                });
            } catch (...) {}
        }
        pos = (end_p == std::string::npos) ? raw.size() : (end_p + kEnd.size());
    }
    return res;
}

// Renders OpenAI chat messages (+ optional tools) into exact Qwen3.5 / Qwen3.6 / Qwen3.8 ChatML format.
inline std::string render_qwen3_chat_prompt(const json & messages,
                                            const json & tools,
                                            bool enable_thinking = true) {
    auto extract_text = [](const json & c) -> std::string {
        if (c.is_string()) return c.get<std::string>();
        if (c.is_array()) {
            std::string s;
            for (const auto & part : c) {
                if (part.is_string()) s += part.get<std::string>();
                else if (part.is_object() && part.contains("text") && part["text"].is_string()) {
                    s += part["text"].get<std::string>();
                }
            }
            return s;
        }
        return "";
    };

    std::string out;
    const bool has_tools = tools.is_array() && !tools.empty();
    const bool first_is_sys = messages.is_array() && !messages.empty() && messages[0].value("role", "") == "system";

    if (has_tools) {
        out += "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto & tool : tools) {
            out += "\n" + tool.dump();
        }
        out += "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
               "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
               "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n"
               "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
               "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
               "- Required parameters MUST be specified\n"
               "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
               "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
        if (first_is_sys && messages[0].contains("content")) {
            std::string sys = extract_text(messages[0]["content"]);
            if (!sys.empty()) out += "\n\n" + sys;
        }
        out += "<|im_end|>\n";
    } else if (first_is_sys && messages[0].contains("content")) {
        out += "<|im_start|>system\n" + extract_text(messages[0]["content"]) + "<|im_end|>\n";
    }

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & m = messages[i];
        const std::string role = m.value("role", "");
        if (role == "system") continue;
        const std::string txt = m.contains("content") ? extract_text(m["content"]) : "";

        if (role == "user") {
            out += "<|im_start|>user\n" + txt + "<|im_end|>\n";
        } else if (role == "assistant") {
            const std::string rc = (m.contains("reasoning_content") && m["reasoning_content"].is_string())
                                       ? m["reasoning_content"].get<std::string>() : "";
            out += "<|im_start|>assistant\n";
            if (!rc.empty()) {
                out += "<think>\n" + rc;
                if (rc.back() != '\n') out += "\n";
                out += "</think>";
                if (!txt.empty() && txt.front() != '\n') out += "\n\n";
                out += txt;
            } else {
                out += txt;
            }
            if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                for (size_t ti = 0; ti < m["tool_calls"].size(); ++ti) {
                    const auto & tc = m["tool_calls"][ti];
                    const auto & fn = tc.contains("function") ? tc["function"] : tc;
                    if (ti == 0 && !txt.empty()) out += "\n\n";
                    else if (ti > 0) out += "\n";
                    out += "<tool_call>\n<function=" + fn.value("name", "") + ">\n";
                    json args = json::object();
                    if (fn.contains("arguments")) {
                        if (fn["arguments"].is_string()) {
                            try { args = json::parse(fn["arguments"].get<std::string>()); } catch (...) {}
                        } else if (fn["arguments"].is_object()) {
                            args = fn["arguments"];
                        }
                    }
                    for (auto it = args.begin(); it != args.end(); ++it) {
                        out += "<parameter=" + it.key() + ">\n";
                        out += it.value().is_string() ? it.value().get<std::string>() : it.value().dump();
                        out += "\n</parameter>\n";
                    }
                    out += "</function>\n</tool_call>";
                }
            }
            out += "<|im_end|>\n";
        } else if (role == "tool") {
            const bool prev_tool = (i > 0 && messages[i - 1].value("role", "") == "tool");
            const bool next_tool = (i + 1 < messages.size() && messages[i + 1].value("role", "") == "tool");
            if (!prev_tool) out += "<|im_start|>user";
            out += "\n<tool_response>\n" + txt + "\n</tool_response>";
            if (!next_tool) out += "<|im_end|>\n";
        }
    }

    out += "<|im_start|>assistant\n";
    if (enable_thinking) out += "<think>\n";
    else                 out += "<think>\n\n</think>\n\n";
    return out;
}

// Renders OpenAI chat messages (+ optional tools) into exact Raon-1-mini ChatML format.
inline std::string render_raon1_chat_prompt(const json & messages,
                                            const json & tools,
                                            bool enable_thinking = true) {
    auto trim = [](std::string s) -> std::string {
        size_t a = 0;
        while (a < s.size() && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r')) ++a;
        size_t b = s.size();
        while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r')) --b;
        return s.substr(a, b - a);
    };
    auto extract_text = [](const json & c) -> std::string {
        if (c.is_string()) return c.get<std::string>();
        if (c.is_array()) {
            std::string s;
            for (const auto & part : c) {
                if (part.is_string()) s += part.get<std::string>();
                else if (part.is_object() && part.contains("text") && part["text"].is_string()) {
                    s += part["text"].get<std::string>();
                }
            }
            return s;
        }
        return "";
    };

    std::string out;
    const bool has_tools    = tools.is_array() && !tools.empty();
    const bool first_is_sys = messages.is_array() && !messages.empty() && messages[0].value("role", "") == "system";

    if (has_tools) {
        out += "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto & tool : tools) {
            out += "\n" + tool.dump();
        }
        out += "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
               "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
               "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n"
               "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
               "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
               "- Required parameters MUST be specified\n"
               "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
               "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
        if (first_is_sys && messages[0].contains("content")) {
            std::string sys = trim(extract_text(messages[0]["content"]));
            if (!sys.empty()) out += "\n\n" + sys;
        }
        out += "<|im_end|>\n";
    } else if (first_is_sys && messages[0].contains("content")) {
        out += "<|im_start|>system\n" + trim(extract_text(messages[0]["content"])) + "<|im_end|>\n";
    }

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & m = messages[i];
        const std::string role = m.value("role", "");
        if (role == "system" && i == 0) continue;
        std::string txt = m.contains("content") ? trim(extract_text(m["content"])) : "";

        if (role == "system") {
            out += "<|im_start|>system\n" + txt + "<|im_end|>\n";
        } else if (role == "user") {
            out += "<|im_start|>user\n" + txt + "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string rc = (m.contains("reasoning_content") && m["reasoning_content"].is_string())
                                 ? trim(m["reasoning_content"].get<std::string>()) : "";
            if (rc.empty() && txt.compare(0, 7, "<think>") == 0) {
                size_t end_t = txt.find("</think>");
                if (end_t != std::string::npos) {
                    rc  = trim(txt.substr(7, end_t - 7));
                    txt = trim(txt.substr(end_t + 8));
                }
            }
            out += "<|im_start|>assistant\n";
            if (!rc.empty()) out += "<think>\n" + rc + "\n</think>\n\n";
            else             out += "<think></think>\n\n";
            out += txt;
            if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                for (size_t ti = 0; ti < m["tool_calls"].size(); ++ti) {
                    const auto & tc = m["tool_calls"][ti];
                    const auto & fn = tc.contains("function") ? tc["function"] : tc;
                    if (ti == 0 && !txt.empty()) out += "\n\n";
                    else if (ti > 0) out += "\n";
                    out += "<tool_call>\n<function=" + fn.value("name", "") + ">\n";
                    json args = json::object();
                    if (fn.contains("arguments")) {
                        if (fn["arguments"].is_string()) {
                            try { args = json::parse(fn["arguments"].get<std::string>()); } catch (...) {}
                        } else if (fn["arguments"].is_object()) {
                            args = fn["arguments"];
                        }
                    }
                    for (auto it = args.begin(); it != args.end(); ++it) {
                        out += "<parameter=" + it.key() + ">\n";
                        out += it.value().is_string() ? it.value().get<std::string>() : it.value().dump();
                        out += "\n</parameter>\n";
                    }
                    out += "</function>\n</tool_call>";
                }
            }
            out += "<|im_end|>\n";
        } else if (role == "tool") {
            const bool prev_tool = (i > 0 && messages[i - 1].value("role", "") == "tool");
            const bool next_tool = (i + 1 < messages.size() && messages[i + 1].value("role", "") == "tool");
            if (!prev_tool) out += "<|im_start|>user";
            out += "\n<tool_response>\n" + txt + "\n</tool_response>";
            if (!next_tool) out += "<|im_end|>\n";
        }
    }

    out += "<|im_start|>assistant\n";
    if (enable_thinking) out += "<think>\n";
    else                 out += "<think></think>\n\n";
    return out;
}

// Renders OpenAI chat messages (+ optional tools) into exact Qwen2 / Qwen2.5 Instruct ChatML format (non-thinking).
inline std::string render_qwen2_chat_prompt(const json & messages, const json & tools) {
    auto extract_text = [](const json & c) -> std::string {
        if (c.is_string()) return c.get<std::string>();
        if (c.is_array()) {
            std::string s;
            for (const auto & part : c) {
                if (part.is_string()) s += part.get<std::string>();
                else if (part.is_object() && part.contains("text") && part["text"].is_string()) {
                    s += part["text"].get<std::string>();
                }
            }
            return s;
        }
        return "";
    };

    std::string out;
    const bool has_tools    = tools.is_array() && !tools.empty();
    const bool first_is_sys = messages.is_array() && !messages.empty() && messages[0].value("role", "") == "system";
    const std::string sys_txt = (first_is_sys && messages[0].contains("content"))
                                    ? extract_text(messages[0]["content"])
                                    : "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    out += "<|im_start|>system\n" + sys_txt;
    if (has_tools) {
        out += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const auto & tool : tools) {
            out += "\n" + tool.dump();
        }
        out += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
               "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
    }
    out += "<|im_end|>\n";

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & m = messages[i];
        const std::string role = m.value("role", "");
        if (role == "system" && i == 0) continue;
        const std::string txt = m.contains("content") ? extract_text(m["content"]) : "";

        if (role == "system") {
            out += "<|im_start|>system\n" + txt + "<|im_end|>\n";
        } else if (role == "user") {
            out += "<|im_start|>user\n" + txt + "<|im_end|>\n";
        } else if (role == "assistant") {
            out += "<|im_start|>assistant\n" + txt;
            if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                for (const auto & tc : m["tool_calls"]) {
                    const auto & fn = tc.contains("function") ? tc["function"] : tc;
                    json args = json::object();
                    if (fn.contains("arguments")) {
                        if (fn["arguments"].is_string()) {
                            try { args = json::parse(fn["arguments"].get<std::string>()); } catch (...) {}
                        } else if (fn["arguments"].is_object()) {
                            args = fn["arguments"];
                        }
                    }
                    json obj = {{"name", fn.value("name", "")}, {"arguments", args}};
                    out += "\n<tool_call>\n" + obj.dump() + "\n</tool_call>";
                }
            }
            out += "<|im_end|>\n";
        } else if (role == "tool") {
            const bool prev_tool = (i > 0 && messages[i - 1].value("role", "") == "tool");
            const bool next_tool = (i + 1 < messages.size() && messages[i + 1].value("role", "") == "tool");
            if (!prev_tool) out += "<|im_start|>user";
            out += "\n<tool_response>\n" + txt + "\n</tool_response>";
            if (!next_tool) out += "<|im_end|>\n";
        }
    }

    out += "<|im_start|>assistant\n";
    return out;
}

// Parses Qwen2 / Qwen2.5 Instruct output (plain text + optional JSON <tool_call>{...}</tool_call>).
inline ParsedAssistantResponse parse_qwen2_output(const std::string & raw) {
    ParsedAssistantResponse res;
    size_t tc_pos = raw.find("<tool_call>");
    if (tc_pos == std::string::npos) {
        res.content = raw;
        return res;
    }
    res.content = raw.substr(0, tc_pos);
    while (!res.content.empty() && (res.content.back() == '\n' || res.content.back() == ' ')) {
        res.content.pop_back();
    }
    size_t cursor = tc_pos;
    while ((cursor = raw.find("<tool_call>", cursor)) != std::string::npos) {
        const size_t body_start = cursor + 11;
        size_t tc_end = raw.find("</tool_call>", body_start);
        if (tc_end == std::string::npos) tc_end = raw.size();
        std::string body = raw.substr(body_start, tc_end - body_start);
        try {
            json tj = json::parse(body);
            std::string fname = tj.value("name", "");
            json args = tj.contains("arguments") ? tj["arguments"] : json::object();
            std::string args_s = args.is_string() ? args.get<std::string>() : args.dump();
            res.tool_calls.push_back({
                {"id", next_tool_call_id()},
                {"type", "function"},
                {"function", {{"name", fname}, {"arguments", args_s}}}
            });
        } catch (...) {}
        cursor = (tc_end < raw.size()) ? (tc_end + 12) : raw.size();
    }
    return res;
}

// Renders OpenAI chat messages (+ optional tools) into exact Bailing V3 (Ling-3.0 / Ling-3.2-Air) format.
inline std::string render_bailing3_chat_prompt(const json & messages,
                                               const json & tools,
                                               bool enable_thinking = true) {
    auto extract_text = [](const json & c) -> std::string {
        if (c.is_string()) return c.get<std::string>();
        if (c.is_array()) {
            std::string s;
            for (const auto & item : c) {
                if (item.is_object() && item.value("type", "") == "text") s += item.value("text", "");
            }
            return s;
        }
        return "";
    };

    const std::string thinking_opt = enable_thinking ? "on" : "off";
    const bool has_tools = tools.is_array() && !tools.empty();
    const bool first_sys = messages.is_array() && !messages.empty() && messages[0].value("role", "") == "system";
    const std::string sys_txt = first_sys ? extract_text(messages[0]["content"]) : "";
    const bool sys_has_think = (sys_txt.find("detailed thinking on") != std::string::npos ||
                                sys_txt.find("detailed thinking off") != std::string::npos);

    std::string out = "<role>SYSTEM</role>";
    if (has_tools) {
        if (first_sys) {
            out += sys_txt;
            if (!sys_txt.empty() && sys_txt.back() != '\n') out += "\n";
        }
        out += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const auto & t : tools) out += "\n" + t.dump();
        out += "\n</tools>\n\nIf none of the functions can be used, point it out. "
               "If the given question lacks the parameters required by the function, also point it out.\n"
               "If you need to use a function, for each function call, output the function name and arguments within the following XML format:\n"
               "<tool_call>{function-name}\n"
               "<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n"
               "<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>\n";
        if (sys_has_think) out += "<|role_end|>";
        else               out += "detailed thinking " + thinking_opt + "<|role_end|>";
    } else {
        if (first_sys) {
            out += sys_txt;
            if (!sys_has_think) {
                if (!sys_txt.empty() && sys_txt.back() != '\n') out += "\n";
                out += "detailed thinking " + thinking_opt;
            }
            out += "<|role_end|>";
        } else {
            out += "detailed thinking " + thinking_opt + "<|role_end|>";
        }
    }

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & m = messages[i];
        const std::string role = m.value("role", "");
        std::string content = m.contains("content") ? extract_text(m["content"]) : "";

        if (role == "system") {
            if (i > 0) out += "<role>SYSTEM</role>" + content + "<|role_end|>";
        } else if (role == "user") {
            out += "<role>HUMAN</role>" + content + "<|role_end|>";
        } else if (role == "assistant") {
            std::string rc = (m.contains("reasoning_content") && m["reasoning_content"].is_string())
                                 ? m["reasoning_content"].get<std::string>() : "";
            if (rc.empty()) {
                size_t et = content.find("</think>");
                if (et != std::string::npos) {
                    std::string before = content.substr(0, et);
                    size_t st = before.rfind("<think>");
                    rc = (st != std::string::npos) ? before.substr(st + 7) : before;
                    content = content.substr(et + 8);
                }
            }
            while (!rc.empty() && rc.front() == '\n') rc.erase(rc.begin());
            while (!rc.empty() && rc.back() == '\n') rc.pop_back();
            while (!content.empty() && content.front() == '\n') content.erase(content.begin());

            out += "<role>ASSISTANT</role>\n<think>" + rc + "</think>" + content;
            if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                for (size_t ti = 0; ti < m["tool_calls"].size(); ++ti) {
                    const auto & tc = m["tool_calls"][ti];
                    const auto & fn = tc.contains("function") ? tc["function"] : tc;
                    if ((ti == 0 && !content.empty()) || ti > 0) out += "\n";
                    out += "<tool_call>" + fn.value("name", "");
                    json args = json::object();
                    if (fn.contains("arguments")) {
                        if (fn["arguments"].is_string()) {
                            try { args = json::parse(fn["arguments"].get<std::string>()); } catch (...) {}
                        } else if (fn["arguments"].is_object()) {
                            args = fn["arguments"];
                        }
                    }
                    for (auto it = args.begin(); it != args.end(); ++it) {
                        out += "\n<arg_key>" + it.key() + "</arg_key>\n<arg_value>";
                        out += it.value().is_string() ? it.value().get<std::string>() : it.value().dump();
                        out += "</arg_value>";
                    }
                    out += "\n</tool_call>";
                }
            }
            out += "<|role_end|>";
        } else if (role == "tool") {
            const bool prev_tool = (i > 0 && messages[i - 1].value("role", "") == "tool");
            const bool next_tool = (i + 1 < messages.size() && messages[i + 1].value("role", "") == "tool");
            if (!prev_tool) out += "<role>OBSERVATION</role>";
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (!next_tool) out += "<|role_end|>";
        }
    }

    out += "<role>ASSISTANT</role>";
    if (enable_thinking) out += "\n<think>";
    else                 out += "\n<think></think>";
    return out;
}

// Renders OpenAI chat messages into Gemma 4 turn format (<|turn>role\n...<turn|>\n).
inline std::string render_gemma4_chat_prompt(const json & messages, bool enable_thinking = true) {
    auto extract_text = [](const json & c) -> std::string {
        if (c.is_string()) return c.get<std::string>();
        if (c.is_array()) {
            std::string s;
            for (const auto & item : c) {
                if (item.is_object() && item.value("type", "") == "text") s += item.value("text", "");
            }
            return s;
        }
        return "";
    };

    std::string out;
    bool has_sys = false;
    for (const auto & m : messages) {
        if (m.value("role", "") == "system" || m.value("role", "") == "developer") {
            has_sys = true;
            break;
        }
    }
    if (enable_thinking && !has_sys) {
        out += "<|turn>system\n<|think|>\n<turn|>\n";
    }
    for (const auto & m : messages) {
        const std::string role = m.value("role", "");
        const std::string txt  = m.contains("content") ? extract_text(m["content"]) : "";
        if (role == "system" || role == "developer") {
            out += "<|turn>system\n";
            if (enable_thinking && txt.find("<|think|>") == std::string::npos) out += "<|think|>\n";
            out += txt + "<turn|>\n";
        } else if (role == "user") {
            out += "<|turn>user\n" + txt + "<turn|>\n";
        } else if (role == "assistant") {
            out += "<|turn>model\n" + txt + "<turn|>\n";
        }
    }
    out += "<|turn>model\n";
    if (enable_thinking) {
        out += "<|channel>thought\n";
    }
    return out;
}

inline ParsedAssistantResponse parse_gemma4_output(const std::string & raw, bool started_in_think = true) {
    ParsedAssistantResponse res;
    std::string rest = raw;
    static const std::string kStartCh = "<|channel>thought";
    static const std::string kEndCh   = "<channel|>";
    bool in_think = started_in_think;

    auto trim_ws_back = [](std::string & s) {
        while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    };

    // Strip leading thought blocks (including empty <|channel>thought\n<channel|> emitted when thinking is off)
    while (true) {
        if (!in_think) {
            if (rest.compare(0, kStartCh.size(), kStartCh) == 0) {
                rest.erase(0, kStartCh.size());
                if (!rest.empty() && rest.front() == '\n') rest.erase(rest.begin());
                in_think = true;
            } else {
                break;
            }
        }
        size_t pos = rest.find(kEndCh);
        if (pos != std::string::npos) {
            std::string rc = rest.substr(0, pos);
            trim_ws_back(rc);
            if (!rc.empty()) {
                if (!res.reasoning_content.empty()) res.reasoning_content += "\n";
                res.reasoning_content += rc;
            }
            rest = rest.substr(pos + kEndCh.size());
            while (!rest.empty() && rest.front() == '\n') rest.erase(rest.begin());
            in_think = false;
        } else {
            trim_ws_back(rest);
            if (!rest.empty()) {
                if (!res.reasoning_content.empty()) res.reasoning_content += "\n";
                res.reasoning_content += rest;
            }
            return res;
        }
    }
    res.content = rest;
    return res;
}

// Incremental SSE stream splitter for Gemma 4 <|channel>thought ... <channel|> blocks.
// Guarantees partial tags never leak into delta.content even when enable_thinking == false.
struct Gemma4StreamSplitter {
    bool in_think = false;
    bool strip_leading_nl = false;

    explicit Gemma4StreamSplitter(bool started_in_think = false) : in_think(started_in_think) {}

    template <typename EmitFn, typename SafePrefixFn>
    bool flush(std::string & pending, bool is_final, size_t max_stop_len,
               EmitFn && emit, SafePrefixFn && utf8_safe_prefix) {
        static const std::string kStartCh = "<|channel>thought";
        static const std::string kEndCh   = "<channel|>";
        while (true) {
            if (in_think) {
                size_t end_t = pending.find(kEndCh);
                const size_t hold = std::max<size_t>(kEndCh.size(), max_stop_len);
                if (end_t != std::string::npos) {
                    std::string r = pending.substr(0, end_t);
                    while (!r.empty() && (r.back() == '\n' || r.back() == ' ')) r.pop_back();
                    if (!r.empty() && !emit("reasoning_content", r)) return false;
                    pending.erase(0, end_t + kEndCh.size());
                    while (!pending.empty() && pending.front() == '\n') pending.erase(pending.begin());
                    in_think = false;
                    strip_leading_nl = pending.empty();
                    continue;
                }
                if (is_final) {
                    if (!pending.empty() && !emit("reasoning_content", pending)) return false;
                    pending.clear();
                    return true;
                }
                if (pending.size() > hold) {
                    size_t n = utf8_safe_prefix(pending, pending.size() - hold);
                    if (n > 0) {
                        if (!emit("reasoning_content", pending.substr(0, n))) return false;
                        pending.erase(0, n);
                    }
                }
                return true;
            } else {
                if (strip_leading_nl) {
                    while (!pending.empty() && pending.front() == '\n') pending.erase(pending.begin());
                    if (!pending.empty()) strip_leading_nl = false;
                }
                size_t st = pending.find(kStartCh);
                const size_t hold = std::max<size_t>(kStartCh.size() + 1, max_stop_len);
                if (st != std::string::npos) {
                    // Wait for 1 more byte if not final and tag sits right at end (to consume optional '\n')
                    if (!is_final && st + kStartCh.size() == pending.size()) {
                        if (st > 0) {
                            if (!emit("content", pending.substr(0, st))) return false;
                            pending.erase(0, st);
                        }
                        return true;
                    }
                    if (st > 0 && !emit("content", pending.substr(0, st))) return false;
                    pending.erase(0, st + kStartCh.size());
                    if (!pending.empty() && pending.front() == '\n') pending.erase(pending.begin());
                    in_think = true;
                    continue;
                }
                if (is_final) {
                    if (!pending.empty() && !emit("content", pending)) return false;
                    pending.clear();
                    return true;
                }
                if (pending.size() > hold) {
                    size_t n = utf8_safe_prefix(pending, pending.size() - hold);
                    if (n > 0) {
                        if (!emit("content", pending.substr(0, n))) return false;
                        pending.erase(0, n);
                    }
                }
                return true;
            }
        }
    }
};

} // namespace rookery
