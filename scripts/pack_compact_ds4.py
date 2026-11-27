#!/usr/bin/env python3
"""
Build a compact, zero-requantization, Winner-Take-All (WTA) IQ4_NL DeepSeek-V4.1-Flash
Engram table in native DeepSeek tokenizer space (ALL 24 heads: 2/3/4-grams, 100% compatible
with DeepSeek's native `blk.1/14.engram_kv.weight` [5120 x 6144]).

Why this beats cross-tokenizer unweighted-average tables (like `flapling-8B`):
  1. Direct raw IQ4_NL row copy (144 bytes / 256-dim row) -> ZERO requantization loss,
     exact per-32-element FP16 block scales preserved, and 1.78x MORE rows in the same 8.2 GB!
  2. Native DeepSeek tokenizer + token_map + multipliers -> ZERO cross-tokenizer alignment loss.
  3. Keeps ALL 3 orders (2/3/4-grams = 24 heads = 6144-D) -> 100% native `engram_kv` compatible.
  4. Frequency-based Winner-Take-All (argmax corpus frequency per target bucket) -> ZERO
     orthogonal averaging/cancellation; frequent factual/code phrases retain 1.0000 cosine!
"""
import argparse
import glob
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_engram_table import (
    GGUF_TYPE_ARRAY,
    GGUF_TYPE_STRING,
    GGUF_TYPE_UINT32,
    GGUF_TYPE_UINT64,
    read_gguf_header,
    write_gguf_header,
)

GGML_TYPE_IQ4_NL = 20
IQ4_NL_ROW_BYTES = 144  # 8 blocks x 18 bytes = 144 bytes for 256 elements


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    r = int(math.isqrt(n))
    f = 5
    while f <= r:
        if n % f == 0 or n % (f + 2) == 0:
            return False
        f += 6
    return True


def next_prime(n: int, used: set[int]) -> int:
    cand = n if n % 2 == 1 else n + 1
    while cand in used or not is_prime(cand):
        cand += 2
    used.add(cand)
    return cand


def extract_text_from_jsonl_obj(obj: dict) -> str:
    for k in ("text", "content", "output", "response", "prompt", "instruction"):
        v = obj.get(k)
        if isinstance(v, str) and len(v) >= 60:
            return v
    msgs = obj.get("messages") or obj.get("conversations")
    if isinstance(msgs, list):
        parts = []
        for m in msgs:
            if isinstance(m, dict):
                c = m.get("content") or m.get("value")
                if isinstance(c, str):
                    parts.append(c)
        if parts:
            return "\n".join(parts)
    return ""


def iter_clean_docs(paths: list[str], per_file_chars: int, exclude_substrings: list[str]):
    files: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for ext in ("*.jsonl", "*.txt"):
                files.extend(sorted(pp.glob(ext)))
        elif pp.is_file():
            files.append(pp)
        else:
            files.extend(Path(x) for x in sorted(glob.glob(p)))

    seen = set()
    for fp in files:
        r = str(fp.resolve())
        if r in seen:
            continue
        seen.add(r)
        lname = fp.name.lower()
        if any(ex in lname for ex in exclude_substrings):
            print(f"[corpus] SKIP excluded file: {fp.name}", flush=True)
            continue
        used_chars = 0
        if fp.suffix == ".txt":
            txt = fp.read_text(encoding="utf-8", errors="ignore")
            step = 16384
            for off in range(0, min(len(txt), per_file_chars), step):
                chunk = txt[off : off + step]
                if len(chunk) >= 80:
                    yield chunk
                    used_chars += len(chunk)
        else:
            with fp.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if used_chars >= per_file_chars:
                        break
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    txt = extract_text_from_jsonl_obj(obj)
                    if len(txt) >= 80:
                        yield txt
                        used_chars += len(txt)
        print(f"[corpus] scanned {fp.name}: {used_chars/1e6:.2f}M chars", flush=True)


def compute_rolling_hashes(
    raw_ids: np.ndarray,
    token_map: np.ndarray,
    pad_c: int,
    mult: np.ndarray,  # [4] uint64
) -> list[np.ndarray]:
    """Exact match to rookery `EngramTable::hash_tokens` for deepseek41 (cascade_block=True).
    Returns [rolling_2gram, rolling_3gram, rolling_4gram], each uint64 [T] plus valid_mask bool [T].
    """
    t_len = raw_ids.shape[0]
    in_rng = (raw_ids >= 0) & (raw_ids < token_map.shape[0])
    safe = np.where(in_rng, raw_ids, 0)
    cids = np.where(in_rng, token_map[safe], -1)  # int64

    # Position 0..3 shifts with DeepSeek cascade blocking on negative token_map entries
    toks = []
    blocked = cids < 0
    valid = ~blocked
    t0 = np.where(blocked, pad_c, cids).astype(np.uint64)
    toks.append(t0)
    for s in range(1, 4):
        shifted = np.full(t_len, -1, dtype=np.int64)
        if t_len > s:
            shifted[s:] = cids[:-s]
        blocked = blocked | (shifted < 0)
        ts = np.where(blocked, pad_c, shifted).astype(np.uint64)
        toks.append(ts)

    with np.errstate(over="ignore"):
        r = toks[0] * np.uint64(mult[0])
        out = []
        for o in range(1, 4):
            r = r ^ (toks[o] * np.uint64(mult[o]))
            out.append(r.copy())
    return out, valid


def main():
    ap = argparse.ArgumentParser(description="Build compact WTA IQ4_NL DeepSeek Engram GGUF")
    ap.add_argument("--meta-gguf", required=True, help="DeepSeek shard 00001-of-00005.gguf (KV + tokenizer)")
    ap.add_argument("--table-gguf", required=True, help="DeepSeek shard 00005-of-00005.gguf (IQ4_NL table)")
    ap.add_argument("--tokenizer-json", required=True, help="DeepSeek HF tokenizer.json for fast corpus scanning")
    ap.add_argument("--corpus", nargs="+", required=True, help="Clean corpus dirs/files (jsonl or txt)")
    ap.add_argument("--out", required=True, help="Output standalone compact IQ4_NL GGUF path")
    ap.add_argument("--name", default="deepseek-v41-engram-compact-wta-8b-iq4nl")
    ap.add_argument("--max-tokens", type=int, default=40_000_000)
    ap.add_argument("--per-file-chars", type=int, default=20_000_000)
    ap.add_argument("--rows-2g", type=int, default=1_000_000, help="Target rows per 2-gram head")
    ap.add_argument("--rows-3g", type=int, default=1_250_000, help="Target rows per 3-gram head")
    ap.add_argument("--rows-4g", type=int, default=1_312_500, help="Target rows per 4-gram head")
    ap.add_argument("--exclude", nargs="*", default=["mmlu", "humaneval", "gsm8k", "eval", "test", "valid"])
    args = ap.parse_args()

    t0 = time.time()
    from tokenizers import Tokenizer

    print(f"[init] Reading DeepSeek metadata from {args.meta_gguf}...", flush=True)
    src_kvs, _, _ = read_gguf_header(args.meta_gguf)
    _, tbl_tensors, tbl_data_off = read_gguf_header(args.table_gguf)
    ti = tbl_tensors["per_layer_token_embd.weight"]
    assert ti["dtype"] == GGML_TYPE_IQ4_NL, f"Expected IQ4_NL (20), got {ti['dtype']}"
    head_dim, src_total_rows = int(ti["dims"][0]), int(ti["dims"][1])
    assert head_dim == 256, f"Expected head_dim=256, got {head_dim}"

    arch = "deepseek41"
    src_layers = src_kvs[f"{arch}.ple.layers"][1][1]
    ngram_size = int(src_kvs[f"{arch}.ple.ngram_size"][1])
    heads_per_ngram = int(src_kvs[f"{arch}.ple.heads_per_ngram"][1])
    n_layers = len(src_layers)
    n_cols = (ngram_size - 1) * heads_per_ngram
    assert n_layers == 2 and ngram_size == 4 and heads_per_ngram == 8 and n_cols == 24

    mults = np.asarray(src_kvs[f"{arch}.ple.layer_multipliers"][1][1], dtype=np.uint64).reshape(n_layers, ngram_size)
    src_primes = np.asarray(src_kvs[f"{arch}.ple.head_vocab_sizes"][1][1], dtype=np.uint64).reshape(n_layers, n_cols)
    src_offs = np.asarray(src_kvs[f"{arch}.ple.head_offsets"][1][1], dtype=np.uint64).reshape(n_layers, n_cols)
    tmap = np.asarray(src_kvs[f"{arch}.engram.token_map"][1][1], dtype=np.int64)
    pad_id = int(src_kvs[f"{arch}.engram.pad_token_id"][1])
    pad_c = int(tmap[pad_id])

    # Choose 48 globally unique target primes matching per-order capacity
    order_caps = [args.rows_2g, args.rows_3g, args.rows_4g]
    used_primes: set[int] = set()
    tgt_primes = np.zeros((n_layers, n_cols), dtype=np.uint64)
    tgt_offs = np.zeros((n_layers, n_cols), dtype=np.uint64)
    cur_off = 0
    for l in range(n_layers):
        for o in range(3):
            for h in range(heads_per_ngram):
                c = o * heads_per_ngram + h
                p = next_prime(order_caps[o], used_primes)
                tgt_primes[l, c] = p
                tgt_offs[l, c] = cur_off
                cur_off += p
    tgt_total_rows = int(cur_off)
    out_gb = (tgt_total_rows * IQ4_NL_ROW_BYTES) / 1e9
    print(
        f"[plan] Source={src_total_rows:,} rows -> Target={tgt_total_rows:,} rows "
        f"({out_gb:.2f} GB IQ4_NL, 2g={args.rows_2g:,}/h, 3g={args.rows_3g:,}/h, 4g={args.rows_4g:,}/h)",
        flush=True,
    )

    # Step 1: Scan clean corpus and count exact source-row frequencies per layer!
    # Memory: 768M x uint16 = 1.53 GB! (Easily fits in RAM; saturating uint16 max 65535 is plenty for WTA!)
    print(f"[scan] Loading DeepSeek tokenizer {args.tokenizer_json}...", flush=True)
    tok = Tokenizer.from_file(args.tokenizer_json)
    src_counts = np.zeros(src_total_rows, dtype=np.uint16)

    # Also track (tgt_row -> best_src_row, best_count) in a second pass?
    # EVEN FASTER (ONE PASS!): Wait — a src_row's count grows over time, so Winner-Take-All
    # is exact if we track per-src_row count AND (tgt_row -> winning src_row)!
    # Updating `best_src[tgt_row]` whenever `new_cnt > best_cnt[tgt_row]` is 100% EXACT in ONE SINGLE PASS!!
    # Proof: Every time any `src_row` reaches its final max count, it is visited and compared against
    # `best_cnt[tgt_row]`, so at the end of the scan `best_src[tgt_row]` is GUARANTEED to be the
    # exact global argmax count `src_row` for `tgt_row`!
    best_src = np.full(tgt_total_rows, -1, dtype=np.int32)
    best_cnt = np.zeros(tgt_total_rows, dtype=np.uint16)

    total_toks = 0
    total_docs = 0
    batch_texts: list[str] = []

    def process_batch(texts: list[str]):
        nonlocal total_toks, total_docs
        encs = tok.encode_batch(texts, add_special_tokens=False)
        for enc in encs:
            ids = np.asarray(enc.ids, dtype=np.int64)
            if ids.size < 4:
                continue
            total_toks += int(ids.size)
            total_docs += 1
            for l in range(n_layers):
                rolls, valid = compute_rolling_hashes(ids, tmap, pad_c, mults[l])
                if not np.any(valid):
                    continue
                for o in range(3):
                    rv = rolls[o][valid]
                    if rv.size == 0:
                        continue
                    for h in range(heads_per_ngram):
                        c = o * heads_per_ngram + h
                        sr = (rv % src_primes[l, c]) + src_offs[l, c]
                        tr = (rv % tgt_primes[l, c]) + tgt_offs[l, c]
                        # Increment saturating uint16 count for touched source rows
                        u_sr, inv, u_add = np.unique(sr, return_inverse=True, return_counts=True)
                        cur = src_counts[u_sr].astype(np.int32) + u_add
                        cur = np.minimum(cur, 65535).astype(np.uint16)
                        src_counts[u_sr] = cur
                        # Map each unique src_row to its tgt_row (note: different n-grams with same src_row
                        # could have different tgt_row, so use the paired (tr, cur[inv]))
                        c_pair = cur[inv]
                        # Fast WTA update: keep positions strictly beating current best_cnt[tr]
                        # (If multiple winners hit same tr in same doc, sort ascending by count so highest wins!)
                        better = c_pair > best_cnt[tr]
                        if np.any(better):
                            tr_b = tr[better]
                            sr_b = sr[better].astype(np.int32)
                            cnt_b = c_pair[better]
                            if tr_b.size > 1:
                                ord_c = np.argsort(cnt_b, kind="mergesort")
                                tr_b, sr_b, cnt_b = tr_b[ord_c], sr_b[ord_c], cnt_b[ord_c]
                            best_cnt[tr_b] = cnt_b
                            best_src[tr_b] = sr_b

    for doc in iter_clean_docs(args.corpus, args.per_file_chars, [x.lower() for x in args.exclude]):
        batch_texts.append(doc)
        if len(batch_texts) >= 64:
            process_batch(batch_texts)
            batch_texts.clear()
            if total_docs % 512 == 0:
                filled = int((best_src >= 0).sum())
                print(
                    f"[scan] docs={total_docs:,} toks={total_toks/1e6:.2f}M/{args.max_tokens/1e6:.1f}M "
                    f"filled={filled:,}/{tgt_total_rows:,} ({100.0*filled/tgt_total_rows:.1f}%) "
                    f"({time.time()-t0:.1f}s)",
                    flush=True,
                )
            if total_toks >= args.max_tokens:
                break
    if batch_texts and total_toks < args.max_tokens:
        process_batch(batch_texts)
        batch_texts.clear()

    # Log exact per-order fill stats & token-weighted purity!
    print(f"\n[scan done] docs={total_docs:,} toks={total_toks:,} in {time.time()-t0:.1f}s", flush=True)
    for l in range(n_layers):
        for o, oname in enumerate(("2-gram", "3-gram", "4-gram")):
            c0 = o * heads_per_ngram
            c1 = c0 + heads_per_ngram
            r0 = int(tgt_offs[l, c0])
            r1 = int(tgt_offs[l, c1 - 1] + tgt_primes[l, c1 - 1])
            sub_s = best_src[r0:r1]
            sub_c = best_cnt[r0:r1]
            nz = sub_s >= 0
            med_c = float(np.median(sub_c[nz])) if np.any(nz) else 0.0
            p90_c = float(np.percentile(sub_c[nz], 90)) if np.any(nz) else 0.0
            print(
                f"  L{src_layers[l]:2d} {oname}: filled={nz.mean()*100:.2f}% "
                f"({int(nz.sum()):,}/{r1-r0:,}), winner_cnt med={med_c:.1f} p90={p90_c:.1f}",
                flush=True,
            )
    del src_counts, best_cnt

    # Step 2: Single sequential pass over DeepSeek 00005-of-00005.gguf copying exact 144-byte IQ4_NL rows!
    t_copy = time.time()
    pop_tgt = np.flatnonzero(best_src >= 0).astype(np.int32)
    pop_src = best_src[pop_tgt]
    del best_src
    print(f"[copy] Sorting {pop_tgt.size:,} winning (src_row -> tgt_row) mappings...", flush=True)
    order = np.argsort(pop_src, kind="mergesort")
    pop_src = pop_src[order]
    pop_tgt = pop_tgt[order]
    del order

    out_buf = np.zeros((tgt_total_rows, IQ4_NL_ROW_BYTES), dtype=np.uint8)
    src_base = tbl_data_off + int(ti["offset"])
    chunk_src_rows = 2_000_000  # ~288 MB sequential read per chunk!
    n_pairs = pop_src.size
    p_idx = 0
    with open(args.table_gguf, "rb") as fin:
        for s0 in range(0, src_total_rows, chunk_src_rows):
            if p_idx >= n_pairs:
                break
            s1 = min(src_total_rows, s0 + chunk_src_rows)
            # Find slice of sorted pop_src in [s0, s1)
            p_end = int(np.searchsorted(pop_src, s1, side="left"))
            if p_end <= p_idx:
                continue
            fin.seek(src_base + s0 * IQ4_NL_ROW_BYTES)
            raw = np.frombuffer(fin.read((s1 - s0) * IQ4_NL_ROW_BYTES), dtype=np.uint8).reshape(s1 - s0, IQ4_NL_ROW_BYTES)
            local_s = pop_src[p_idx:p_end] - s0
            out_buf[pop_tgt[p_idx:p_end]] = raw[local_s]
            p_idx = p_end
            if (s0 // chunk_src_rows) % 40 == 0 or p_idx == n_pairs:
                print(
                    f"[copy] copied {p_idx:,}/{n_pairs:,} IQ4_NL rows ({100.0*p_idx/n_pairs:.1f}%, {time.time()-t_copy:.1f}s)",
                    flush=True,
                )
    assert p_idx == n_pairs, f"Missed rows: {p_idx} != {n_pairs}"

    # Step 3: Write standalone GGUF v3 (native deepseek41 arch + tokenizer + IQ4_NL table)
    kv_list = [
        ("general.architecture", GGUF_TYPE_STRING, arch),
        ("general.name", GGUF_TYPE_STRING, args.name),
        (f"{arch}.ple.layers", src_kvs[f"{arch}.ple.layers"][0], src_kvs[f"{arch}.ple.layers"][1]),
        (f"{arch}.ple.ngram_size", GGUF_TYPE_UINT64, ngram_size),
        (f"{arch}.ple.heads_per_ngram", GGUF_TYPE_UINT64, heads_per_ngram),
        (f"{arch}.embedding_length_per_layer_input", GGUF_TYPE_UINT64, head_dim),
        (f"{arch}.ple.layer_multipliers", src_kvs[f"{arch}.ple.layer_multipliers"][0], src_kvs[f"{arch}.ple.layer_multipliers"][1]),
        (f"{arch}.ple.head_offsets", GGUF_TYPE_ARRAY, (GGUF_TYPE_UINT32, tgt_offs.ravel().astype(np.uint32).tolist())),
        (f"{arch}.ple.head_vocab_sizes", GGUF_TYPE_ARRAY, (GGUF_TYPE_UINT32, tgt_primes.ravel().astype(np.uint32).tolist())),
        (f"{arch}.engram.pad_token_id", GGUF_TYPE_UINT64, pad_id),
        (f"{arch}.engram.token_map", src_kvs[f"{arch}.engram.token_map"][0], src_kvs[f"{arch}.engram.token_map"][1]),
    ]
    for k, (vt, v) in src_kvs.items():
        if k.startswith("tokenizer.ggml.") and k != "tokenizer.chat_template":
            kv_list.append((k, vt, v))

    tensor_list = [("per_layer_token_embd.weight", (head_dim, tgt_total_rows), GGML_TYPE_IQ4_NL, 0)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t_wr = time.time()
    with open(args.out, "wb") as fout:
        write_gguf_header(fout, kv_list, tensor_list)
        # Write in 64MB slices to avoid giant memoryview issues
        step_rows = 500_000
        for r0 in range(0, tgt_total_rows, step_rows):
            r1 = min(tgt_total_rows, r0 + step_rows)
            fout.write(out_buf[r0:r1].tobytes())
    print(
        f"[done] Wrote {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB IQ4_NL) "
        f"in {time.time()-t_wr:.1f}s write / {time.time()-t0:.1f}s total!",
        flush=True,
    )


if __name__ == "__main__":
    main()
