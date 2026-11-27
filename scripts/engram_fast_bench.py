#!/usr/bin/env python3
"""
Fast, zero-leak, multi-suite live benchmark runner for Rookery Engram/PLE evaluation.

Evaluates any OpenAI-compatible rookery-server endpoint (greedy, temp=0.0) across:
  - humaneval : All 164 OpenAI HumanEval Python coding problems (sandboxed execution, pass@1)
  - mmlu_pro  : Stratified 300-question 10-choice TIGER-Lab/MMLU-Pro slice
  - gsm8k     : 200-question openai/gsm8k test slice (exact final numeric match)

Designed to run in parallel across independent GPUs/ports (~4-10 min/model) as a
high-signal fast screening & optimization benchmark before full core_v14_local runs.
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def chat_complete(base_url: str, model: str, messages: list, max_tokens: int = 1024, timeout: int = 180, disable_thinking: bool = False) -> tuple[str, str, int]:
    """Call /v1/chat/completions with greedy decoding (temp=0). Returns (content, reasoning, completion_tokens)."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["reasoning_effort"] = "none"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = data.get("usage") or {}
    ctok = int(usage.get("completion_tokens") or 0)
    return content, reasoning, ctok


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.S)
    return text.strip()


def extract_python_code(text: str, entry_point: str, prompt: str) -> str:
    """Extract Python solution from model output and build executable test script."""
    text = strip_thinking(text)
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    code = m[-1] if m else text
    # If model wrote a full function definition including def entry_point, prepend imports from prompt
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, flags=re.M):
        # Grab any helper imports/defs before def entry_point in prompt
        pre = prompt.split(f"def {entry_point}")[0]
        return pre + "\n" + code
    # Otherwise treat as function body completion appended to prompt
    lines = code.splitlines()
    if lines and not lines[0].startswith((" ", "\t")):
        code = "\n".join(("    " + ln if ln.strip() else "") for ln in lines)
    return prompt + "\n" + code


def run_humaneval_test(program: str, test_code: str, entry_point: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    full = f"{program}\n\n{test_code}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(full)
        tpath = tf.name
    try:
        res = subprocess.run(
            [sys.executable, "-I", tpath],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if res.returncode == 0:
            return True, "ok"
        err = (res.stderr or res.stdout or "").strip().splitlines()
        return False, (err[-1][:140] if err else f"rc={res.returncode}")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:140]
    finally:
        try:
            os.unlink(tpath)
        except OSError:
            pass


def eval_humaneval(base_url: str, model: str, data_path: str, limit: int, max_tokens: int, no_think: bool) -> dict:
    items = []
    opener = gzip.open if data_path.endswith(".gz") else open
    with opener(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    if limit > 0:
        items = items[:limit]

    t0 = time.time()
    passed = 0
    results = []
    sys_msg = (
        "You are an expert Python programmer. Complete the given Python function accurately. "
        "Output ONLY a single ```python``` code block containing the complete function implementation (including signature). No explanation."
    )
    if no_think:
        sys_msg += " /no_think"

    for i, item in enumerate(items):
        prompt = item["prompt"]
        ep = item["entry_point"]
        user_msg = f"Complete the following Python function:\n\n```python\n{prompt}```"
        try:
            content, reasoning, ctok = chat_complete(
                base_url,
                model,
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                max_tokens=max_tokens,
                disable_thinking=no_think,
            )
            raw = content if content.strip() else reasoning
            prog = extract_python_code(raw, ep, prompt)
            ok, reason = run_humaneval_test(prog, item["test"], ep)
        except Exception as e:
            ok, reason, ctok = False, f"api_err:{str(e)[:100]}", 0
        if ok:
            passed += 1
        results.append({"task_id": item["task_id"], "passed": ok, "reason": reason, "ctok": ctok})
        if (i + 1) % 40 == 0 or (i + 1) == len(items):
            print(f"  [HumanEval] {i+1}/{len(items)}: {passed}/{i+1} ({100.0*passed/(i+1):.1f}%) elapsed={time.time()-t0:.1f}s", flush=True)

    return {
        "suite": "humaneval",
        "passed": passed,
        "total": len(items),
        "acc": round(100.0 * passed / max(1, len(items)), 2),
        "elapsed_s": round(time.time() - t0, 1),
        "items": results,
    }


def extract_choice_letter(text: str, n_opts: int) -> str:
    text = strip_thinking(text)
    valid = "ABCDEFGHIJ"[:n_opts]
    # Look for explicit Answer: X or \boxed{X} or final standalone letter
    for pat in [
        rf"(?:final\s+answer|answer)\s*[:\-is]+\s*\(?([{valid}])\)?\b",
        rf"\\boxed\{{([{valid}])\}}",
        rf"\*\*([{valid}])\*\*",
        rf"^\s*\(?([{valid}])\)?(?:[\.\)\:]|\s|$)",
    ]:
        ms = re.findall(pat, text, flags=re.I | re.M)
        if ms:
            return ms[-1].upper()
    ms = re.findall(rf"\b([{valid}])\b", text)
    return ms[-1].upper() if ms else "?"


def eval_mmlu_pro(base_url: str, model: str, data_path: str, limit: int, max_tokens: int, no_think: bool) -> dict:
    items = json.load(open(data_path, "r", encoding="utf-8"))
    if limit > 0:
        items = items[:limit]

    t0 = time.time()
    passed = 0
    by_cat = {}
    results = []
    sys_msg = (
        "Answer the multiple-choice question carefully. End your response with `Answer: X` "
        "where X is the single capital letter of the correct option."
    )
    if no_think:
        sys_msg += " /no_think"

    for i, item in enumerate(items):
        opts = item["options"]
        opt_lines = "\n".join(f"{chr(65+j)}. {o}" for j, o in enumerate(opts))
        user_msg = f"Question ({item.get('category','general')}):\n{item['question']}\n\nOptions:\n{opt_lines}\n\nRespond with `Answer: <LETTER>`."
        gold = item["answer"].strip().upper()
        cat = item.get("category", "other")
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][1] += 1
        try:
            content, reasoning, ctok = chat_complete(
                base_url,
                model,
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                max_tokens=max_tokens,
                disable_thinking=no_think,
            )
            pred = extract_choice_letter(content if content.strip() else reasoning, len(opts))
            ok = (pred == gold)
        except Exception as e:
            pred, ok, ctok = f"ERR:{str(e)[:40]}", False, 0
        if ok:
            passed += 1
            by_cat[cat][0] += 1
        results.append({"qid": item.get("question_id", i), "cat": cat, "gold": gold, "pred": pred, "passed": ok, "ctok": ctok})
        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            print(f"  [MMLU-Pro] {i+1}/{len(items)}: {passed}/{i+1} ({100.0*passed/(i+1):.1f}%) elapsed={time.time()-t0:.1f}s", flush=True)

    return {
        "suite": "mmlu_pro",
        "passed": passed,
        "total": len(items),
        "acc": round(100.0 * passed / max(1, len(items)), 2),
        "elapsed_s": round(time.time() - t0, 1),
        "by_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())},
        "items": results,
    }


def extract_gsm8k_num(text: str) -> str:
    text = strip_thinking(text)
    for pat in [
        r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)",
        r"\\boxed\{([-+]?\$?\d[\d,]*(?:\.\d+)?)\}",
        r"(?:final\s+answer|answer)\s*[:\-is]+\s*\$?([-+]?\d[\d,]*(?:\.\d+)?)",
    ]:
        ms = re.findall(pat, text, flags=re.I)
        if ms:
            return ms[-1].replace(",", "").replace("$", "").strip().rstrip(".")
    ms = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return ms[-1].replace(",", "").strip().rstrip(".") if ms else ""


def norm_num(s: str) -> str:
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else f"{v:.6g}"
    except Exception:
        return s.strip()


def eval_gsm8k(base_url: str, model: str, data_path: str, limit: int, max_tokens: int, no_think: bool) -> dict:
    items = json.load(open(data_path, "r", encoding="utf-8"))
    if limit > 0:
        items = items[:limit]

    t0 = time.time()
    passed = 0
    results = []
    sys_msg = (
        "Solve the math problem step by step and end your response with `#### <number>` "
        "containing only the final numeric answer."
    )
    if no_think:
        sys_msg += " /no_think"

    for i, item in enumerate(items):
        gold = norm_num(item["answer"].split("####")[-1].replace(",", "").strip())
        try:
            content, reasoning, ctok = chat_complete(
                base_url,
                model,
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": item["question"]}],
                max_tokens=max_tokens,
                disable_thinking=no_think,
            )
            pred = norm_num(extract_gsm8k_num(content if content.strip() else reasoning))
            ok = (pred == gold)
        except Exception as e:
            pred, ok, ctok = f"ERR:{str(e)[:40]}", False, 0
        if ok:
            passed += 1
        results.append({"idx": i, "gold": gold, "pred": pred, "passed": ok, "ctok": ctok})
        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            print(f"  [GSM8K] {i+1}/{len(items)}: {passed}/{i+1} ({100.0*passed/(i+1):.1f}%) elapsed={time.time()-t0:.1f}s", flush=True)

    return {
        "suite": "gsm8k",
        "passed": passed,
        "total": len(items),
        "acc": round(100.0 * passed / max(1, len(items)), 2),
        "elapsed_s": round(time.time() - t0, 1),
        "items": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--suites", default="humaneval,mmlu_pro", help="Comma-separated: humaneval,mmlu_pro,gsm8k")
    ap.add_argument("--bench-dir", default="./engram-bench")
    ap.add_argument("--limit-humaneval", type=int, default=164)
    ap.add_argument("--limit-mmlu", type=int, default=300)
    ap.add_argument("--limit-gsm8k", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    summary = {"model": args.model, "base_url": args.base_url, "suites": {}}
    for s in suites:
        if s == "humaneval":
            res = eval_humaneval(
                args.base_url, args.model,
                os.path.join(args.bench_dir, "HumanEval.jsonl.gz"),
                args.limit_humaneval, args.max_tokens, args.no_think,
            )
        elif s == "mmlu_pro":
            res = eval_mmlu_pro(
                args.base_url, args.model,
                os.path.join(args.bench_dir, "mmlu_pro_300.json"),
                args.limit_mmlu, args.max_tokens, args.no_think,
            )
        elif s == "gsm8k":
            res = eval_gsm8k(
                args.base_url, args.model,
                os.path.join(args.bench_dir, "gsm8k_200.json"),
                args.limit_gsm8k, args.max_tokens, args.no_think,
            )
        else:
            raise ValueError(f"Unknown suite {s}")
        summary["suites"][s] = res
        print(f"[SUMMARY] {args.model} | {s}: {res['passed']}/{res['total']} ({res['acc']}%) in {res['elapsed_s']}s", flush=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
