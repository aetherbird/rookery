#!/usr/bin/env python3
"""Held-out deterministic technical/factual knowledge QA benchmark for measuring Engram value.

Evaluates any OpenAI-compatible server (`rookery-server`) on 60 exact-match Linux man-page,
POSIX, Python stdlib, HTTP/RFC, networking, and Git plumbing questions to measure factual
recall delta with and without Engram sidecar grafting.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

QUESTIONS: list[dict[str, object]] = [
    # Linux / POSIX / CLI flags & signals
    {"id": "lx01", "cat": "linux", "q": "What numeric signal number is SIGKILL on Linux x86_64? Reply with just the integer.", "ans": ["9"]},
    {"id": "lx02", "cat": "linux", "q": "What numeric signal number is SIGTERM on Linux x86_64? Reply with just the integer.", "ans": ["15"]},
    {"id": "lx03", "cat": "linux", "q": "What numeric signal number is SIGHUP on Linux x86_64? Reply with just the integer.", "ans": ["1"]},
    {"id": "lx04", "cat": "linux", "q": "What standard POSIX shell exit code indicates 'command not found'? Reply with just the integer.", "ans": ["127"]},
    {"id": "lx05", "cat": "linux", "q": "What standard POSIX shell exit code indicates a command was found but is not executable? Reply with just the integer.", "ans": ["126"]},
    {"id": "lx06", "cat": "linux", "q": "In `tar`, which single-letter flag extracts an archive? Reply with just the single letter (no dash).", "ans": ["x"]},
    {"id": "lx07", "cat": "linux", "q": "In `tar`, which single-letter flag filters the archive through xz compression? Reply with just the single letter (no dash).", "ans": ["J"]},
    {"id": "lx08", "cat": "linux", "q": "In `tar`, which single-letter flag filters the archive through bzip2 compression? Reply with just the single letter (no dash).", "ans": ["j"]},
    {"id": "lx09", "cat": "linux", "q": "In `chmod`, what octal value represents the sticky bit (e.g. on `/tmp`)? Reply with 4 octal digits like 1000.", "ans": ["1000", "01000"]},
    {"id": "lx10", "cat": "linux", "q": "In `chmod`, what 4-digit octal mode sets setuid + rwxr-xr-x? Reply with just the 4 digits.", "ans": ["4755"]},
    {"id": "lx11", "cat": "linux", "q": "In `chmod`, what 4-digit octal mode sets setgid + rwxr-xr-x? Reply with just the 4 digits.", "ans": ["2755"]},
    {"id": "lx12", "cat": "linux", "q": "In GNU `grep`, which single-letter flag enables Perl-compatible regular expressions (PCRE)? Reply with just the single letter (no dash).", "ans": ["P"]},
    {"id": "lx13", "cat": "linux", "q": "In GNU `grep`, which single-letter flag prints only the filenames with matching lines? Reply with just the single letter (no dash).", "ans": ["l"]},
    {"id": "lx14", "cat": "linux", "q": "In `rsync`, which single-letter flag enables archive mode (`-rlptgoD`)? Reply with just the single letter (no dash).", "ans": ["a"]},
    {"id": "lx15", "cat": "linux", "q": "In `find`, what test primary matches files modified strictly more than N 24-hour periods ago? Reply with just the primary name including leading dash.", "ans": ["-mtime"]},
    {"id": "lx16", "cat": "linux", "q": "What Linux `/proc` file lists currently mounted filesystems in classic fstab format? Reply with just the absolute path.", "ans": ["/proc/mounts"]},
    {"id": "lx17", "cat": "linux", "q": "What POSIX errno macro name means 'No space left on device'? Reply with just the macro name.", "ans": ["ENOSPC"]},
    {"id": "lx18", "cat": "linux", "q": "What POSIX errno macro name means 'Too many open files' (per-process limit)? Reply with just the macro name.", "ans": ["EMFILE"]},
    {"id": "lx19", "cat": "linux", "q": "What POSIX errno macro name means 'Interrupted system call'? Reply with just the macro name.", "ans": ["EINTR"]},
    {"id": "lx20", "cat": "linux", "q": "In `ss` / `netstat`, what single-letter flag displays listening sockets only? Reply with just the single letter (no dash).", "ans": ["l"]},
    # Networking / HTTP / RFCs
    {"id": "nt01", "cat": "net", "q": "What well-known TCP port is assigned to HTTPS? Reply with just the integer.", "ans": ["443"]},
    {"id": "nt02", "cat": "net", "q": "What well-known TCP port is assigned to PostgreSQL by default? Reply with just the integer.", "ans": ["5432"]},
    {"id": "nt03", "cat": "net", "q": "What well-known TCP port is assigned to Redis by default? Reply with just the integer.", "ans": ["6379"]},
    {"id": "nt04", "cat": "net", "q": "What well-known TCP/UDP port is assigned to DNS? Reply with just the integer.", "ans": ["53"]},
    {"id": "nt05", "cat": "net", "q": "What HTTP status code means 'Too Many Requests'? Reply with just the 3-digit integer.", "ans": ["429"]},
    {"id": "nt06", "cat": "net", "q": "What HTTP status code means 'Conflict'? Reply with just the 3-digit integer.", "ans": ["409"]},
    {"id": "nt07", "cat": "net", "q": "What HTTP status code means 'No Content'? Reply with just the 3-digit integer.", "ans": ["204"]},
    {"id": "nt08", "cat": "net", "q": "What HTTP status code means 'Permanent Redirect' (preserving request method)? Reply with just the 3-digit integer.", "ans": ["308"]},
    {"id": "nt09", "cat": "net", "q": "What HTTP status code means 'Payload Too Large' / 'Content Too Large'? Reply with just the 3-digit integer.", "ans": ["413"]},
    {"id": "nt10", "cat": "net", "q": "How many usable host addresses are in an IPv4 `/26` subnet (excluding network and broadcast)? Reply with just the integer.", "ans": ["62"]},
    {"id": "nt11", "cat": "net", "q": "How many total IP addresses (including network and broadcast) are in an IPv4 `/24` block? Reply with just the integer.", "ans": ["256"]},
    {"id": "nt12", "cat": "net", "q": "What is the IPv6 loopback address in compressed canonical form? Reply with just the address.", "ans": ["::1"]},
    {"id": "nt13", "cat": "net", "q": "Which RFC defines JSON Pointer? Reply with just the RFC number (integer only).", "ans": ["6901"]},
    {"id": "nt14", "cat": "net", "q": "Which RFC defines UUIDs (the 2024 revision introducing UUIDv6/v7/v8)? Reply with just the RFC number (integer only).", "ans": ["9562"]},
    {"id": "nt15", "cat": "net", "q": "What DNS record type maps an IP address back to a domain name (reverse DNS)? Reply with just the record type.", "ans": ["PTR"]},
    # Python stdlib / internals
    {"id": "py01", "cat": "python", "q": "In Python's `itertools`, what function yields successive r-length combinations WITH replacement? Reply with just the function name.", "ans": ["combinations_with_replacement"]},
    {"id": "py02", "cat": "python", "q": "In Python's `functools`, what decorator transforms a class defining `__eq__` and one ordering method into full rich comparisons? Reply with just the decorator name (no `@`).", "ans": ["total_ordering"]},
    {"id": "py03", "cat": "python", "q": "In Python's `bisect` module, what function returns the insertion point AFTER any existing entries of x? Reply with just the function name.", "ans": ["bisect_right", "bisect"]},
    {"id": "py04", "cat": "python", "q": "In Python's `heapq`, what function pushes an item onto a heap and then pops and returns the smallest item in one operation? Reply with just the function name.", "ans": ["heappushpop"]},
    {"id": "py05", "cat": "python", "q": "In Python's `heapq`, what function pops and returns the smallest item and then pushes a new item? Reply with just the function name.", "ans": ["heapreplace"]},
    {"id": "py06", "cat": "python", "q": "In Python 3.11+, what standard library module provides read-only TOML parsing? Reply with just the module name.", "ans": ["tomllib"]},
    {"id": "py07", "cat": "python", "q": "In Python's `contextlib`, what class manages a dynamic stack of context managers? Reply with just the class name.", "ans": ["ExitStack"]},
    {"id": "py08", "cat": "python", "q": "In Python's `subprocess.run`, what keyword argument raises `CalledProcessError` if exit code is non-zero? Reply with just the argument name.", "ans": ["check"]},
    {"id": "py09", "cat": "python", "q": "In Python's `dataclasses.dataclass`, what boolean parameter generates `__slots__` automatically (Python 3.10+)? Reply with just the parameter name.", "ans": ["slots"]},
    {"id": "py10", "cat": "python", "q": "In Python's `urllib.parse`, what function parses a query string into a dict mapping keys to lists of values? Reply with just the function name.", "ans": ["parse_qs"]},
    {"id": "py11", "cat": "python", "q": "In Python's `struct` module, what format character represents an unsigned 64-bit integer? Reply with just the single character.", "ans": ["Q"]},
    {"id": "py12", "cat": "python", "q": "In Python's `struct` module, what prefix character denotes little-endian with standard sizes and no alignment? Reply with just the single character.", "ans": ["<"]},
    {"id": "py13", "cat": "python", "q": "In Python's `collections`, what container groups multiple dicts into a single updateable view? Reply with just the class name.", "ans": ["ChainMap"]},
    {"id": "py14", "cat": "python", "q": "In Python's `re` module, what flag makes `.` match newline characters as well? Reply with just the flag name without `re.`.", "ans": ["DOTALL", "S"]},
    {"id": "py15", "cat": "python", "q": "In Python's `hashlib`, what function computes PBKDF2-HMAC key derivation? Reply with just the function name.", "ans": ["pbkdf2_hmac"]},
    # Git / SQL / Systems facts
    {"id": "gt01", "cat": "git", "q": "What Git plumbing command prints the content, type, or size of repository objects? Reply with just the subcommand name (without `git`).", "ans": ["cat-file"]},
    {"id": "gt02", "cat": "git", "q": "What Git plumbing command resolves a ref or revision expression into a full SHA hash? Reply with just the subcommand name (without `git`).", "ans": ["rev-parse"]},
    {"id": "gt03", "cat": "git", "q": "What Git command uses binary search over commit history to find the commit that introduced a bug? Reply with just the subcommand name (without `git`).", "ans": ["bisect"]},
    {"id": "gt04", "cat": "git", "q": "In `git log`, which `--pretty` / `--format` placeholder prints the full commit hash? Reply with just the 2-character placeholder.", "ans": ["%H"]},
    {"id": "gt05", "cat": "git", "q": "In SQLite, what PRAGMA enables Write-Ahead Logging mode? Reply with just `journal_mode=WAL` or the PRAGMA name `journal_mode`.", "ans": ["journal_mode", "journal_mode=WAL"]},
    {"id": "gt06", "cat": "sys", "q": "How many bits are in a SHA-256 digest? Reply with just the integer.", "ans": ["256"]},
    {"id": "gt07", "cat": "sys", "q": "How many bytes are in a UUID? Reply with just the integer.", "ans": ["16"]},
    {"id": "gt08", "cat": "sys", "q": "In IEEE 754 binary32 (float32), how many exponent bits are there? Reply with just the integer.", "ans": ["8"]},
    {"id": "gt09", "cat": "sys", "q": "In IEEE 754 bfloat16, how many significand (explicit mantissa) bits are there? Reply with just the integer.", "ans": ["7"]},
    {"id": "gt10", "cat": "sys", "q": "In UTF-8, what is the maximum number of bytes used to encode a single Unicode scalar value? Reply with just the integer.", "ans": ["4"]},
]


def norm_ans(s: str) -> str:
    s = s.strip().strip("`\"'.").strip()
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate held-out technical knowledge QA against an OpenAI-compatible server")
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    passed = 0
    by_cat: dict[str, list[int]] = {}
    rows = []
    for item in QUESTIONS:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": item["q"]}],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        req = urllib.request.Request(
            f"{args.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode())
        content = norm_ans(d["choices"][0]["message"].get("content") or "")
        ok = any(content.lower() == a.lower() for a in item["ans"])
        passed += int(ok)
        by_cat.setdefault(str(item["cat"]), []).append(int(ok))
        rows.append({"id": item["id"], "cat": item["cat"], "got": content, "exp": item["ans"], "pass": ok})

    summary = {
        "model": args.model,
        "score": f"{passed}/{len(QUESTIONS)}",
        "pct": round(100.0 * passed / len(QUESTIONS), 2),
        "by_cat": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cat.items())},
        "results": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[engram-qa] {args.model}: {summary['score']} ({summary['pct']}%) {summary['by_cat']}")


if __name__ == "__main__":
    main()
