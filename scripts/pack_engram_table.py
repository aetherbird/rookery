#!/usr/bin/env python3
"""
Pack compact standalone Engram / PLE provider tables into mmap-able GGUF v3 files
for Rookery's EngramTable + `engram-z-v3` canonical bridge.

Modes:
  1. `flapling-fp8`:
     Packs a 2-layer FP8 E4M3FN NGRAM_HASH table (e.g. flapling-22b 8.19B FP8, ~7.64 GiB)
     together with its BPE tokenizer & compressed token_map into a single standalone GGUF
     with `per_layer_token_embd.weight` stored directly as GGML F8_E4M3 (type 40, zero requant!).

  2. `gemma4-ple`:
     Extracts `per_layer_token_embd.weight` (Q6_K or F16/BF16, [n_layers * ple_dim, vocab_size])
     and the Gemma-4 tokenizer from a Gemma-4 E2B/E4B GGUF, selects `K` evenly-spaced depth layers
     (default K=16 -> 4096-dim multi-depth unigram vector), per-layer RMS-normalizes each 256-dim
     slice, quantizes to GGML F8_E4M3 (~1.00 GiB for 262,144 x 4096), and writes a standalone
     UNIGRAM_PLE GGUF readable directly by `rookery::EngramTable`!
"""
import argparse
import json
import os
import struct
import time
from pathlib import Path

import numpy as np

# GGML / GGUF constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGN = 32

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q6_K = 14
GGML_TYPE_BF16 = 30
GGML_TYPE_F8_E4M3 = 43


def _read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="surrogateescape")


def read_gguf_header(path: str):
    """Parse all KV fields and tensor descriptors from a GGUF v3 file (no tensor data loaded)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {path}")
        ver, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        kvs = {}
        scalar_fmt = {
            0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
            6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d",
        }
        for _ in range(n_kv):
            k = _read_str(f)
            (vt,) = struct.unpack("<I", f.read(4))
            if vt == GGUF_TYPE_STRING:
                kvs[k] = (vt, _read_str(f))
            elif vt == GGUF_TYPE_ARRAY:
                et, ln = struct.unpack("<IQ", f.read(12))
                if et == GGUF_TYPE_STRING:
                    arr = [_read_str(f) for _ in range(ln)]
                else:
                    fmt = scalar_fmt[et]
                    sz = struct.calcsize(fmt)
                    raw = f.read(sz * ln)
                    arr = list(struct.unpack(f"<{ln}{fmt[1]}", raw))
                kvs[k] = (vt, (et, arr))
            else:
                fmt = scalar_fmt[vt]
                (val,) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
                kvs[k] = (vt, bool(val) if vt == GGUF_TYPE_BOOL else val)

        tensors = {}
        for _ in range(n_tensors):
            name = _read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            dtype, off = struct.unpack("<IQ", f.read(12))
            tensors[name] = {"dims": dims, "dtype": dtype, "offset": off}

        align = GGUF_DEFAULT_ALIGN
        if "general.alignment" in kvs:
            align = int(kvs["general.alignment"][1])
        pos = f.tell()
        data_offset = (pos + align - 1) // align * align
    return kvs, tensors, data_offset


def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8", errors="surrogateescape")
    return struct.pack("<Q", len(b)) + b


def _pack_kv(key: str, vt: int, val) -> bytes:
    out = _pack_str(key) + struct.pack("<I", vt)
    scalar_fmt = {
        0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
        6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d",
    }
    if vt == GGUF_TYPE_STRING:
        return out + _pack_str(val)
    if vt == GGUF_TYPE_ARRAY:
        et, arr = val
        out += struct.pack("<IQ", et, len(arr))
        if et == GGUF_TYPE_STRING:
            return out + b"".join(_pack_str(x) for x in arr)
        fmt = scalar_fmt[et]
        return out + struct.pack(f"<{len(arr)}{fmt[1]}", *arr)
    fmt = scalar_fmt[vt]
    return out + struct.pack(fmt, int(val) if vt == GGUF_TYPE_BOOL else val)


def write_gguf_header(fout, kv_list, tensor_list, align=GGUF_DEFAULT_ALIGN):
    """Write GGUF v3 header, KV section, tensor info section, and pad to alignment."""
    fout.write(GGUF_MAGIC)
    fout.write(struct.pack("<IQQ", GGUF_VERSION, len(tensor_list), len(kv_list)))
    for k, vt, v in kv_list:
        fout.write(_pack_kv(k, vt, v))
    for name, dims, dtype, off in tensor_list:
        fout.write(_pack_str(name))
        fout.write(struct.pack("<I", len(dims)))
        fout.write(struct.pack(f"<{len(dims)}Q", *dims))
        fout.write(struct.pack("<IQ", dtype, off))
    pos = fout.tell()
    pad = (align - (pos % align)) % align
    if pad:
        fout.write(b"\x00" * pad)


def dequant_q6_k_rows(raw_bytes: bytes, n_rows: int, row_elems: int) -> np.ndarray:
    """Vectorized exact GGML Q6_K dequantization -> float32 [n_rows, row_elems].
    Q6_K block: 256 elems, 210 bytes:
      ql[128] (uint8), qh[64] (uint8), scales[16] (int8), d (float16, 2 bytes).
    """
    assert row_elems % 256 == 0
    n_blk = row_elems // 256
    buf = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_rows * n_blk, 210)
    ql = buf[:, 0:128]
    qh = buf[:, 128:192]
    sc = buf[:, 192:208].view(np.int8).astype(np.float32)  # [B, 16]
    d = buf[:, 208:210].view(np.float16).astype(np.float32)  # [B, 1]

    out = np.empty((n_rows * n_blk, 256), dtype=np.float32)
    for half in range(2):
        ql_h = ql[:, half * 64 : (half + 1) * 64]
        qh_h = qh[:, half * 32 : (half + 1) * 32]
        sc_h = sc[:, half * 8 : (half + 1) * 8]
        # Each half produces 4 groups of 32 elements: [0..31], [32..63], [64..95], [96..127]
        q1 = ((ql_h[:, 0:32] & 0x0F) | (((qh_h >> 0) & 3) << 4)).astype(np.int16) - 32
        q2 = ((ql_h[:, 32:64] & 0x0F) | (((qh_h >> 2) & 3) << 4)).astype(np.int16) - 32
        q3 = ((ql_h[:, 0:32] >> 4) | (((qh_h >> 4) & 3) << 4)).astype(np.int16) - 32
        q4 = ((ql_h[:, 32:64] >> 4) | (((qh_h >> 6) & 3) << 4)).astype(np.int16) - 32

        # Sub-scales: each 32-elem group uses 2 scales (elems 0..15 and 16..31)
        s0 = d * np.repeat(sc_h[:, 0:2], 16, axis=1)
        s1 = d * np.repeat(sc_h[:, 2:4], 16, axis=1)
        s2 = d * np.repeat(sc_h[:, 4:6], 16, axis=1)
        s3 = d * np.repeat(sc_h[:, 6:8], 16, axis=1)

        base = half * 128
        out[:, base + 0 : base + 32] = s0 * q1
        out[:, base + 32 : base + 64] = s1 * q2
        out[:, base + 64 : base + 96] = s2 * q3
        out[:, base + 96 : base + 128] = s3 * q4

    return out.reshape(n_rows, row_elems)


def f32_to_fp8_e4m3fn(x: np.ndarray) -> np.ndarray:
    """Fast vectorized float32 -> IEEE FP8 E4M3FN (uint8) using a 65536-entry FP16 LUT."""
    if not hasattr(f32_to_fp8_e4m3fn, "_lut"):
        # Build exact FP16 -> FP8 E4M3FN LUT once (takes ~15ms)
        vals = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
        sign = (vals < 0).astype(np.uint8) << 7
        av = np.abs(vals)
        # Build all 127 finite non-negative E4M3FN levels
        pos_codes = np.arange(127, dtype=np.uint8)  # 0..126 (127 is NaN)
        e = (pos_codes >> 3) & 0xF
        m = pos_codes & 0x7
        levels = np.where(e == 0, np.ldexp(m.astype(np.float32), -9),
                          np.ldexp(1.0 + m.astype(np.float32) * 0.125, e.astype(np.int32) - 7))
        # Clamp av to max finite E4M3FN (448.0), NaN -> 0
        av_clean = np.where(np.isfinite(av), np.minimum(av, 448.0), 0.0)
        idx = np.searchsorted(levels, av_clean)
        idx = np.clip(idx, 0, len(levels) - 1)
        prev = np.maximum(idx - 1, 0)
        choose_prev = np.abs(av_clean - levels[prev]) <= np.abs(av_clean - levels[idx])
        best = np.where(choose_prev, prev, idx).astype(np.uint8)
        f32_to_fp8_e4m3fn._lut = (sign | best).astype(np.uint8)
    h = np.clip(x, -448.0, 448.0).astype(np.float16).view(np.uint16)
    return f32_to_fp8_e4m3fn._lut[h]


def pack_flapling_fp8(args):
    t0 = time.time()
    spec = json.loads(Path(args.spec).read_text())
    layer_ids = sorted(int(x) for x in spec["layers"])
    ngram_sizes = [int(x) for x in spec["ngram_sizes"]]
    max_ngram = max(ngram_sizes)
    heads_per_ngram = int(spec["heads_per_ngram"])
    head_dim = int(spec["head_dim"])

    all_mults, all_offsets, all_primes = [], [], []
    layer_rows = {}
    global_base = 0
    for lid in layer_ids:
        lo = spec["target_layers"][str(lid)]
        rows = int(lo["rows"])
        layer_rows[lid] = rows
        all_mults.extend(int(m) for m in lo["multipliers"])
        all_primes.extend(int(p) for p in lo["head_primes"])
        all_offsets.extend(global_base + int(o) for o in lo["head_offsets"])
        global_base += rows
    total_rows = global_base

    # Copy tokenizer KV fields verbatim from flapling-14b GGUF
    src_kvs, _, _ = read_gguf_header(args.tokenizer_gguf)
    arch = "bailingmoe3"
    vocab_size = len(src_kvs["tokenizer.ggml.tokens"][1][1])

    lut_raw = np.load(args.lookup).astype(np.int64)
    token_map = np.zeros(vocab_size, dtype=np.int64)
    n_copy = min(vocab_size, lut_raw.shape[0])
    token_map[:n_copy] = lut_raw[:n_copy]

    kv_list = [
        ("general.architecture", GGUF_TYPE_STRING, arch),
        ("general.name", GGUF_TYPE_STRING, "flapling-22b-engram-8b-fp8"),
        (f"{arch}.ple.layers", GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, layer_ids)),
        (f"{arch}.ple.ngram_size", GGUF_TYPE_UINT64, max_ngram),
        (f"{arch}.ple.heads_per_ngram", GGUF_TYPE_UINT64, heads_per_ngram),
        (f"{arch}.embedding_length_per_layer_input", GGUF_TYPE_UINT64, head_dim),
        (f"{arch}.ple.layer_multipliers", GGUF_TYPE_ARRAY, (GGUF_TYPE_UINT64, all_mults)),
        (f"{arch}.ple.head_offsets", GGUF_TYPE_ARRAY, (GGUF_TYPE_UINT64, all_offsets)),
        (f"{arch}.ple.head_vocab_sizes", GGUF_TYPE_ARRAY, (GGUF_TYPE_UINT64, all_primes)),
        (f"{arch}.engram.pad_token_id", GGUF_TYPE_UINT64, 0),
        (f"{arch}.engram.token_map", GGUF_TYPE_ARRAY, (GGUF_TYPE_INT64, token_map.tolist())),
    ]
    for k, (vt, v) in src_kvs.items():
        if k.startswith("tokenizer.ggml.") and k != "tokenizer.chat_template":
            kv_list.append((k, vt, v))

    tensor_list = [("per_layer_token_embd.weight", (head_dim, total_rows), GGML_TYPE_F8_E4M3, 0)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as fout:
        write_gguf_header(fout, kv_list, tensor_list)
        chunk_bytes = 64 * 1024 * 1024
        total_written = 0
        for lid in layer_ids:
            q_path = Path(args.tables_root) / f"target_l{lid}" / "table.fp8_e4m3.u8"
            exp_bytes = layer_rows[lid] * head_dim
            assert q_path.stat().st_size == exp_bytes, f"{q_path} size mismatch"
            with open(q_path, "rb") as fin:
                while True:
                    buf = fin.read(chunk_bytes)
                    if not buf:
                        break
                    fout.write(buf)
                    total_written += len(buf)
            print(f"[pack] streamed target_l{lid} ({layer_rows[lid]:,} x {head_dim} F8_E4M3, {time.time()-t0:.1f}s)", flush=True)
        assert total_written == total_rows * head_dim
    print(f"[done] wrote {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB) in {time.time()-t0:.1f}s")


def pack_gemma4_ple(args):
    t0 = time.time()
    src_kvs, tensors, data_off = read_gguf_header(args.src_gguf)
    ti = tensors["per_layer_token_embd.weight"]
    dims = ti["dims"]  # (n_layers * ple_dim, vocab_size) = (10752, 262144)
    dtype = ti["dtype"]
    n_total_cols, n_vocab = int(dims[0]), int(dims[1])
    ple_dim = int(src_kvs.get("gemma4.embedding_length_per_layer_input", (0, 256))[1])
    n_src_layers = n_total_cols // ple_dim
    assert n_src_layers * ple_dim == n_total_cols

    k_layers = min(args.num_layers, n_src_layers)
    sel_layers = np.linspace(0, n_src_layers - 1, k_layers).round().astype(int).tolist()
    out_dim = k_layers * ple_dim
    print(f"[gemma4-ple] src dims={dims} dtype={dtype} ({n_src_layers}L x {ple_dim}) -> selecting {k_layers} layers {sel_layers} -> out_dim={out_dim}", flush=True)

    kv_list = [
        ("general.architecture", GGUF_TYPE_STRING, "gemma4"),
        ("general.name", GGUF_TYPE_STRING, f"gemma4-e4b-ple-{k_layers}L-fp8"),
        ("gemma4.block_count", GGUF_TYPE_UINT64, 1),
        ("gemma4.embedding_length_per_layer_input", GGUF_TYPE_UINT64, out_dim),
        ("gemma4.ple.selected_layers", GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, sel_layers)),
    ]
    for k, (vt, v) in src_kvs.items():
        if k.startswith("tokenizer.ggml.") and k != "tokenizer.chat_template":
            kv_list.append((k, vt, v))

    tensor_list = [("per_layer_token_embd.weight", (out_dim, n_vocab), GGML_TYPE_F8_E4M3, 0)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    row_bytes_src = (n_total_cols // 256) * 210 if dtype == GGML_TYPE_Q6_K else (n_total_cols * 2)
    chunk_rows = 4096

    with open(args.src_gguf, "rb") as fin, open(args.out, "wb") as fout:
        write_gguf_header(fout, kv_list, tensor_list)
        fin.seek(data_off + ti["offset"])
        for start in range(0, n_vocab, chunk_rows):
            n = min(chunk_rows, n_vocab - start)
            raw = fin.read(n * row_bytes_src)
            if dtype == GGML_TYPE_Q6_K:
                f32 = dequant_q6_k_rows(raw, n, n_total_cols)  # [n, n_src_layers * ple_dim]
            elif dtype == GGML_TYPE_F16:
                f32 = np.frombuffer(raw, dtype=np.float16).reshape(n, n_total_cols).astype(np.float32)
            elif dtype == GGML_TYPE_BF16:
                u32 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                f32 = u32.view(np.float32).reshape(n, n_total_cols)
            else:
                raise ValueError(f"Unsupported Gemma-4 PLE dtype: {dtype}")

            # Reshape to [n, n_src_layers, ple_dim], pick selected layers, RMS-normalize each 256-dim slice
            # so every depth contributes equally, then scale into FP8 E4M3FN sweet spot (RMS ~ 16.0)
            sub = f32.reshape(n, n_src_layers, ple_dim)[:, sel_layers, :]  # [n, k_layers, ple_dim]
            rms = np.sqrt(np.mean(sub * sub, axis=-1, keepdims=True) + 1e-8)
            sub_norm = (sub / rms) * 16.0
            fp8 = f32_to_fp8_e4m3fn(sub_norm.reshape(n, out_dim))
            fout.write(fp8.tobytes())
            if (start // chunk_rows) % 16 == 0 or start + n == n_vocab:
                print(f"[gemma4-ple] packed {start+n:,}/{n_vocab:,} rows ({time.time()-t0:.1f}s)", flush=True)

    print(f"[done] wrote {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB) in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="Pack compact Engram / PLE tables into standalone GGUFs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("flapling-fp8")
    p1.add_argument("--spec", required=True)
    p1.add_argument("--lookup", required=True)
    p1.add_argument("--tables-root", required=True)
    p1.add_argument("--tokenizer-gguf", required=True)
    p1.add_argument("--out", required=True)

    p2 = sub.add_parser("gemma4-ple")
    p2.add_argument("--src-gguf", required=True)
    p2.add_argument("--num-layers", type=int, default=16)
    p2.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "flapling-fp8":
        pack_flapling_fp8(args)
    else:
        pack_gemma4_ple(args)


if __name__ == "__main__":
    main()
