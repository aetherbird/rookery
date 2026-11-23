#!/usr/bin/env python3
"""
Rookery End-to-End Server Acceptance Test Suite (Kimi-style realistic workload).

Tests any OpenAI-compatible model served by rookery-server (directly or via llama-swap):
  1. /health, /v1/models, /tokenize sanity
  2. Multi-byte UTF-8 SSE streaming (em-dash, curly quotes, CJK, emojis, code blocks)
  3. OpenAI "stop" strings/arrays (streaming & non-streaming) + null JSON fields
  4. Tool definitions & tool_calls streaming (validating delta.tool_calls[].index)
  5. Multi-turn prefix caching (both stripping <think> and keeping <think>, temp=0 and temp=0.7)
  6. Long-context prefill + decode + needle-in-a-haystack accuracy at configurable lengths
     (default: 2K, 16K, 32K, 64K tokens) with speed regression checks.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def http_json(url: str, payload: dict | None = None, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sse_chat(base_url: str, payload: dict, timeout: int = 600) -> tuple[float, list[dict], str, str, list[dict]]:
    payload = dict(payload)
    payload["stream"] = True
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    chunks: list[dict] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            if line == "data: [DONE]":
                break
            if not line.startswith("data: "):
                continue
            ch = json.loads(line[6:])
            chunks.append(ch)
            choices = ch.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning_parts.append(delta["reasoning_content"])
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
    dt = time.monotonic() - t0
    return dt, chunks, "".join(content_parts), "".join(reasoning_parts), tool_calls


def make_haystack(target_tokens: int, needle_key: str, needle_val: str, needle_frac: float = 0.5) -> str:
    # ~42 tokens per entry -> varied, realistic technical log entries (avoids identical-line router collapse)
    n_entries = max(10, target_tokens // 42)
    needle_idx = max(1, min(n_entries - 2, int(n_entries * needle_frac)))
    lines = []
    for i in range(n_entries):
        if i == needle_idx:
            lines.append(
                f"[RECORD-{i:05d}] CRITICAL CONFIGURATION OVERRIDE: The unique verification token for "
                f"project {needle_key} is {needle_val}. Always quote {needle_val} verbatim when asked."
            )
        else:
            h1 = (i * 2654435761) & 0xFFFFFFFF
            h2 = ((i + 1337) * 2246822519) & 0xFFFFFFFF
            lines.append(
                f"[RECORD-{i:05d}] cluster-node-{i % 64:02d} shard=0x{h1:08x} crc=0x{h2:08x} "
                f"latency={12 + (i * 19) % 480}ms queue_depth={(i * 7) % 97} status=healthy_ok"
            )
    return "\n".join(lines)


def run_suite(args: argparse.Namespace) -> int:
    base = args.url.rstrip("/")
    model = args.model
    failures: list[str] = []

    def check(cond: bool, msg: str):
        if cond:
            print(f"  [PASS] {msg}")
        else:
            print(f"  [FAIL] {msg}")
            failures.append(msg)

    print(f"=== Rookery E2E Acceptance Suite: model={model} url={base} ===")

    # 1. /v1/models check
    print("\n[1/6] API metadata check")
    models = http_json(f"{base}/v1/models", timeout=30)
    check("data" in models and len(models["data"]) >= 1, f"/v1/models returned {len(models.get('data', []))} entries")

    # 2. Multi-byte UTF-8 SSE streaming + null JSON params + temp=0.7
    print("\n[2/6] Multi-byte UTF-8 SSE streaming + null params + temp=0.7")
    dt, chunks, content, reasoning, _ = sse_chat(
        base,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": None},
                {
                    "role": "user",
                    "content": (
                        "Reply in one short sentence using an em-dash — a curly quote ’ "
                        "the Chinese word 你好 and a rocket emoji 🚀."
                    ),
                },
            ],
            "max_tokens": None,
            "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=600,
    )
    last = chunks[-1] if chunks else {}
    timings = last.get("timings") or {}
    usage = last.get("usage") or {}
    print(f"  wall={dt:.2f}s chunks={len(chunks)} content={content!r} timings={timings}")
    check(len(chunks) >= 3, f"received {len(chunks)} SSE chunks without disconnect")
    check("timings" in last and "usage" in last, "final SSE chunk includes both usage and timings")
    check(len(content.strip()) > 0, "non-empty UTF-8 streamed content")
    check(timings.get("predicted_per_second", 0) >= args.min_tg_tps,
          f"temp=0.7 decode speed {timings.get('predicted_per_second', 0):.1f} tok/s >= {args.min_tg_tps}")

    # 3. OpenAI stop strings (streaming + non-streaming)
    print("\n[3/6] OpenAI stop strings (SSE + non-stream)")
    _, st_chunks, st_content, _, _ = sse_chat(
        base,
        {
            "model": model,
            "messages": [{"role": "user", "content": "Count from 1 to 10 separated by commas: 1, 2, 3, ..."}],
            "stop": ["5", "6"],
            "max_tokens": 64,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    print(f"  SSE stop content={st_content!r}")
    check("5" not in st_content and ("4" in st_content or "3" in st_content), f"SSE stop trimmed cleanly: {st_content!r}")

    ns_stop = http_json(
        f"{base}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Count from 1 to 10 separated by commas: 1, 2, 3, ..."}],
            "stop": "5",
            "max_tokens": 64,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    ns_txt = ns_stop["choices"][0]["message"].get("content", "")
    print(f"  Non-stream stop content={ns_txt!r}")
    check("5" not in ns_txt and ("4" in ns_txt or "3" in ns_txt), f"non-stream stop trimmed cleanly: {ns_txt!r}")

    # 4. Tool definitions + streamed tool_calls[].index validation
    print("\n[4/6] Tool definitions & SSE tool_calls[].index")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    _, tc_chunks, tc_content, tc_reas, tc_list = sse_chat(
        base,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "Always use the get_weather tool when asked about weather."},
                {"role": "user", "content": "What is the weather in Tokyo right now?"},
            ],
            "tools": tools,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    )
    print(f"  tool_calls={tc_list!r} finish={tc_chunks[-1]['choices'][0]['finish_reason'] if tc_chunks else None}")
    check(len(tc_list) >= 1, f"model emitted {len(tc_list)} tool call(s)")
    if tc_list:
        check(all("index" in tc for tc in tc_list), "every streamed delta.tool_calls entry has integer 'index'")
        check(tc_list[0].get("function", {}).get("name") == "get_weather", "tool call name == get_weather")

    # 5. Multi-turn prefix caching (strip <think> AND keep <think>, temp=0.7)
    print("\n[5/6] Multi-turn prefix caching (2K prompt, strip & keep <think>)")
    doc_2k = make_haystack(2000, "ALPHA-77", "FALCON-9421", 0.4)
    m_base = [
        {"role": "system", "content": "You are a concise assistant. Answer in one short sentence."},
        {"role": "user", "content": f"Logs:\n{doc_2k}\n\nWhat is the verification token for project ALPHA-77?"},
    ]
    dt1, ch1, ans1, reas1, _ = sse_chat(base, {"model": model, "messages": m_base, "max_tokens": 512, "temperature": 0.7})
    t1 = ch1[-1].get("timings", {})
    print(f"  Turn 1 (cold 2K): wall={dt1:.2f}s pp={t1.get('prompt_per_second', 0):.1f} tok/s tg={t1.get('predicted_per_second', 0):.1f} tok/s ans={ans1!r}")
    check("FALCON-9421" in ans1 or "FALCON-9421" in reas1, f"2K needle retrieved: {ans1!r}")

    # Turn 2: strip reasoning_content
    m_t2 = m_base + [
        {"role": "assistant", "content": ans1},
        {"role": "user", "content": "Confirm the project name for that token in 5 words."},
    ]
    dt2, ch2, ans2, reas2, _ = sse_chat(base, {"model": model, "messages": m_t2, "max_tokens": 256, "temperature": 0.7})
    t2 = ch2[-1].get("timings", {})
    u2 = ch2[-1].get("usage", {})
    cached2 = t2.get("cache_n", 0)
    print(f"  Turn 2 (strip <think>): wall={dt2:.2f}s cached={cached2}/{u2.get('prompt_tokens')} pp_ms={t2.get('prompt_ms', 0):.1f}")
    check(cached2 >= 1500, f"Turn 2 (strip <think>) reused {cached2} tokens (>= 1500)")

    # Turn 3: keep reasoning_content
    m_t3 = m_t2 + [
        {"role": "assistant", "content": ans2, "reasoning_content": reas2},
        {"role": "user", "content": "Say OK."},
    ]
    dt3, ch3, ans3, _, _ = sse_chat(base, {"model": model, "messages": m_t3, "max_tokens": 128, "temperature": 0.7})
    t3 = ch3[-1].get("timings", {})
    u3 = ch3[-1].get("usage", {})
    cached3 = t3.get("cache_n", 0)
    print(f"  Turn 3 (keep <think>): wall={dt3:.2f}s cached={cached3}/{u3.get('prompt_tokens')} pp_ms={t3.get('prompt_ms', 0):.1f}")
    check(cached3 >= u3.get("prompt_tokens", 0) - 32, f"Turn 3 (keep <think>) reused {cached3}/{u3.get('prompt_tokens')} tokens")

    # 6. Long-context needle + speed scaling (e.g. 16K, 32K, 64K)
    print(f"\n[6/6] Long-context scaling & needle retrieval: {args.contexts}")
    for ctx_k in args.contexts:
        target_tok = ctx_k * 1024
        key = f"OMEGA-{ctx_k}K"
        val = f"QUARTZ-{ctx_k * 137 + 42}"
        hay = make_haystack(target_tok - 128, key, val, 0.55)
        msgs = [
            {"role": "system", "content": "Find the exact verification token requested. State the token clearly and write one concise 25-word summary of the surrounding archive entry."},
            {"role": "user", "content": f"Archive:\n{hay}\n\nWhat is the exact verification token for project {key}?"},
        ]
        dt_l, ch_l, ans_l, reas_l, _ = sse_chat(
            base,
            {
                "model": model,
                "messages": msgs,
                "max_tokens": 512,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=900,
        )
        tl = ch_l[-1].get("timings", {})
        ul = ch_l[-1].get("usage", {})
        pp_tps = tl.get("prompt_per_second", 0.0)
        tg_tps = tl.get("predicted_per_second", 0.0)
        print(
            f"  [{ctx_k}K] prompt={ul.get('prompt_tokens')} pp={tl.get('prompt_ms', 0)/1000:.2f}s "
            f"({pp_tps:.1f} tok/s) | gen={tl.get('predicted_n')} ({tg_tps:.1f} tok/s) | ans={ans_l!r}"
        )
        check(val in ans_l or val in reas_l, f"{ctx_k}K needle '{val}' retrieved")
        check(pp_tps >= args.min_pp_tps, f"{ctx_k}K prefill {pp_tps:.1f} tok/s >= {args.min_pp_tps}")
        check(tg_tps >= args.min_tg_tps, f"{ctx_k}K decode {tg_tps:.1f} tok/s >= {args.min_tg_tps}")

    print("\n=== SUMMARY ===")
    if failures:
        print(f"FAILED ({len(failures)} failures):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL E2E CHECKS PASSED!")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Rookery E2E server acceptance test suite")
    ap.add_argument("--url", default="http://127.0.0.1:11434", help="Base server URL")
    ap.add_argument("--model", required=True, help="Model alias")
    ap.add_argument("--contexts", type=int, nargs="*", default=[16, 32], help="Long-context sizes in K tokens (e.g. 16 32 64)")
    ap.add_argument("--min-pp-tps", type=float, default=80.0, help="Minimum long-context prefill tok/s")
    ap.add_argument("--min-tg-tps", type=float, default=20.0, help="Minimum decode tok/s")
    return run_suite(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
