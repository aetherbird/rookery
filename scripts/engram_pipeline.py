#!/usr/bin/env python3
"""
End-to-End Automated Cross-Model Engram Pipeline for rookery:
    inspect -> verify addressing (100% parity gate) -> dump -> calibrate/train (RKAD v3)
    -> 4-way quality gate (Base / Disabled / Real / Shuffled) -> latency bench -> register (.json card)

Works across any supported Engram provider (deepseek41, qwen4exp, hy_v4_flash, unigram_ple)
and any hooked host model (maxwell-moe, qwen35moe, glm5_next, qwen4exp, inkling2_loop).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, env=None, capture=True):
    print(f"+ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True,
    )
    if capture and proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}")
    return proc.stdout or ""


def inspect_table(engram_test_bin: str, engram_path: str) -> dict:
    out = run_cmd([engram_test_bin, "--engram", engram_path, "--table-only"])
    m_hdr = re.search(r"Loaded EngramTable \[([^,\]]+),\s*mode=([^,\]]+),\s*fp=(0x[0-9a-fA-F]+)\]", out)
    m_dim = re.search(r"layers=(\d+),\s*ngram=(\d+),\s*heads=(\d+),\s*head_dim=(\d+)\s*\(engram_dim=(\d+)\),\s*rows=(\d+),\s*dtype=([^\s(]+)", out)
    if not m_hdr or not m_dim:
        raise RuntimeError("Failed to parse EngramTable metadata from --table-only output")
    return {
        "provider_arch": m_hdr.group(1),
        "provider_mode": m_hdr.group(2),
        "provider_fingerprint": m_hdr.group(3),
        "n_layers": int(m_dim.group(1)),
        "ngram_size": int(m_dim.group(2)),
        "n_cols": int(m_dim.group(3)),
        "head_dim": int(m_dim.group(4)),
        "engram_dim": int(m_dim.group(5)),
        "total_rows": int(m_dim.group(6)),
        "dtype": m_dim.group(7),
    }


def inspect_host_gguf(gguf_path: str) -> dict:
    import struct

    def read_str(f):
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def read_val(f, t):
        if t in (0, 1, 7):
            return struct.unpack("<B", f.read(1))[0]
        if t in (2, 3):
            return struct.unpack("<H", f.read(2))[0]
        if t == 4:
            return struct.unpack("<I", f.read(4))[0]
        if t == 5:
            return struct.unpack("<i", f.read(4))[0]
        if t == 6:
            return struct.unpack("<f", f.read(4))[0]
        if t == 10:
            return struct.unpack("<Q", f.read(8))[0]
        if t == 11:
            return struct.unpack("<q", f.read(8))[0]
        if t == 12:
            return struct.unpack("<d", f.read(8))[0]
        if t == 8:
            return read_str(f)
        if t == 9:
            et, cnt = struct.unpack("<IQ", f.read(12))
            for _ in range(cnt):
                read_val(f, et)
            return None
        return None

    kv = {}
    with open(gguf_path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"Not a GGUF file: {gguf_path}")
        f.read(4)
        _, nkv = struct.unpack("<QQ", f.read(16))
        for _ in range(nkv):
            k = read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            v = read_val(f, t)
            if v is not None:
                kv[k] = v

    arch = str(kv.get("general.architecture", "unknown"))
    pfx = arch + "."
    n_layers = int(kv.get(pfx + "block_count", 0))
    n_mtp = int(kv.get(pfx + "nextn_predict_layers", 0))
    # Models with MTP layers include nextn_predict_layers in block_count (e.g. maxwell-moe 45+1=46, qwen35moe 40+1=41);
    # subtract n_mtp so host_n_layers matches backbone layer count (engine.layer_devices().size()).
    if n_mtp > 0 and n_layers > n_mtp:
        n_layers -= n_mtp
    d_embd = int(kv.get(pfx + "embedding_length", 0))
    hc_mult = int(kv.get(pfx + "hyper_connection.count", kv.get(pfx + "hc_mult", 4 if arch in ("maxwell-moe", "glm5_next", "qwen4exp") else 1)))
    return {
        "host_arch": arch,
        "host_n_layers": n_layers,
        "host_d_embd": d_embd,
        "host_hc_mult": hc_mult,
    }


def parse_4way_eval(out: str, is_factual: bool = False) -> dict:
    res = {"base_ppl": None, "disabled_delta": None, "scales": []}
    m_base = re.search(r"\[BASE\s*\].*?(?:ans_)?PPL=\s*([0-9.]+)", out)
    m_dis = re.search(r"\[DISABLED\].*?delta vs base=([+-][0-9.]+)", out)
    if m_base:
        res["base_ppl"] = float(m_base.group(1))
    if m_dis:
        res["disabled_delta"] = float(m_dis.group(1))

    real_pat = re.compile(r"\[REAL\s*\]\s*global_scale=\s*([0-9.]+).*?(?:ans_)?PPL=\s*([0-9.]+).*?\(vs base=([+-][0-9.]+)")
    shuf_pat = re.compile(r"\[SHUFFLED\]\s*global_scale=\s*([0-9.]+).*?(?:ans_)?PPL=\s*([0-9.]+).*?\(shuf-real=([+-][0-9.]+)")

    reals = {float(m.group(1)): (float(m.group(2)), float(m.group(3))) for m in real_pat.finditer(out)}
    shufs = {float(m.group(1)): (float(m.group(2)), float(m.group(3))) for m in shuf_pat.finditer(out)}
    for sc in sorted(reals.keys()):
        rp, rdelta = reals[sc]
        sp, gap = shufs.get(sc, (None, None))
        res["scales"].append({
            "scale": sc,
            "real_ppl": rp,
            "vs_base": rdelta,
            "shuf_ppl": sp,
            "shuf_minus_real": gap,
            "beats_base": rdelta < 0.0,
            "beats_shuffled": (gap is not None and gap > 0.0),
        })
    return res


def cmd_inspect(args):
    engram_test = str(Path(args.build_dir) / "rookery_engram_test")
    info = {}
    if args.engram:
        info["provider"] = inspect_table(engram_test, args.engram)
    if args.model:
        info["host"] = inspect_host_gguf(args.model)
    if args.adapter:
        run_cmd([engram_test, "--inspect-adapter", args.adapter])
    print(json.dumps(info, indent=2))


def cmd_build(args):
    t0 = time.time()
    build_dir = Path(args.build_dir)
    engram_test = str(build_dir / "rookery_engram_test")
    fit_bin = str(build_dir / "rookery-fit-adapter")
    train_py = str(Path(__file__).resolve().parent / "train_engram_bridge.py")

    print("=== [Stage 1/5] Inspect Provider & Host Contract ===")
    prov = inspect_table(engram_test, args.engram)
    host = inspect_host_gguf(args.model)
    print(json.dumps({"provider": prov, "host": host}, indent=2))

    print("\n=== [Stage 2/5] Strict 100% Causal Addressing Parity Gate ===")
    v_out = run_cmd([engram_test, "-m", args.model, "--engram", args.engram, "--verify-addressing"])
    if "Result: PASS (100% exact)" not in v_out:
        raise RuntimeError("Addressing parity gate FAILED — refusing to calibrate misaligned table!")

    print(f"\n=== [Stage 3/5] Calibrate / Train RKAD v3 ({args.mode}) ===")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dump = str(out_path.with_suffix(".rkpr" if args.mode == "level2" else ".rkce"))

    if args.mode == "level2":
        run_cmd([
            engram_test, "-m", args.model, "--engram", args.engram,
            "--gpus", args.gpus, "--split", args.split,
            "--dump-pairs", tmp_dump, "--dump-delta",
            "--corpus", args.corpus, "--max-tokens", str(args.max_tokens),
        ])
        run_cmd([
            fit_bin, "--pairs", tmp_dump, "--out", str(out_path),
            "--rank", str(args.rank), "--auto-lam", "--scale", str(args.base_scale),
            "--center", "--device", "0", "--rkad-v3",
            "--provider-arch", prov["provider_arch"],
            "--provider-fp", prov["provider_fingerprint"],
            "--host-arch", host["host_arch"],
            "--host-n-layers", str(host["host_n_layers"]),
            "--host-hc-mult", str(host["host_hc_mult"]),
        ])
    else:
        ce_layer = args.ce_layer if args.ce_layer >= 0 else (host["host_n_layers"] - 1)
        if not args.reuse_rkce or not os.path.exists(tmp_dump):
            run_cmd([
                engram_test, "-m", args.model, "--engram", args.engram,
                "--gpus", args.gpus, "--split", args.split,
                "--dump-tail-ce", tmp_dump, "--ce-layer", str(ce_layer),
                "--doc-ctx", "4096", "--max-tokens", str(args.max_tokens),
                "--corpus", args.corpus,
            ])
        tr_cmd = [
            args.python, train_py,
            "--rkce", tmp_dump, "--out", str(out_path),
            "--src-layer", str(args.src_layer), "--rank", str(args.rank), "--scale", str(args.base_scale),
            "--epochs", str(args.epochs), "--batch-size", "256", "--lr", "8e-4", "--wd", "2e-2",
            "--calibrate-base-temp", "--kl-beta", "0.5", "--shuf-kl-beta", "1.5", "--delta-reg", "1.0",
            "--dropout", "0.25", "--rkad-v3",
            "--provider-arch", prov["provider_arch"],
            "--provider-fingerprint", prov["provider_fingerprint"],
            "--host-arch", host["host_arch"],
            "--host-n-layers", str(host["host_n_layers"]),
            "--delta-clip", str(args.delta_clip),
        ]
        if args.safetensors_dir:
            tr_cmd += ["--safetensors-dir", args.safetensors_dir]
        else:
            tr_cmd += ["--gguf", args.model]
        env = os.environ.copy()
        if args.train_cuda_device:
            env["CUDA_VISIBLE_DEVICES"] = args.train_cuda_device
        run_cmd(tr_cmd, env=env)

    print("\n=== [Stage 4/5] Mandatory 4-Way Quality Gate (Base / Disabled / Real / Shuffled) ===")
    evals = {}
    if args.eval_tech:
        out_t = run_cmd([
            engram_test, "-m", args.model, "--engram", args.engram,
            "--engram-adapter", str(out_path), "--gpus", args.gpus, "--split", args.split,
            "--eval-ppl", args.eval_tech, "--eval-tokens", str(args.eval_tokens),
            "--scale-sweep", args.scale_sweep, "--shuffle-salt", "0xDEADBEEF1337CAFE",
        ])
        evals["eval_tech"] = parse_4way_eval(out_t, is_factual=False)

    if args.eval_prose:
        out_p = run_cmd([
            engram_test, "-m", args.model, "--engram", args.engram,
            "--engram-adapter", str(out_path), "--gpus", args.gpus, "--split", args.split,
            "--eval-ppl", args.eval_prose, "--eval-tokens", str(args.eval_tokens),
            "--scale-sweep", args.scale_sweep, "--shuffle-salt", "0xDEADBEEF1337CAFE",
        ])
        evals["eval_prose"] = parse_4way_eval(out_p, is_factual=False)

    if args.eval_factual:
        out_f = run_cmd([
            engram_test, "-m", args.model, "--engram", args.engram,
            "--engram-adapter", str(out_path), "--gpus", args.gpus, "--split", args.split,
            "--eval-factual", args.eval_factual,
            "--scale-sweep", args.scale_sweep, "--shuffle-salt", "0xDEADBEEF1337CAFE",
        ])
        evals["eval_factual"] = parse_4way_eval(out_f, is_factual=True)

    # Compute promotion gate: disabled invariance + exists scale beating BOTH base and shuffled on primary evals
    disabled_ok = all(ev.get("disabled_delta") == 0.0 for ev in evals.values() if ev.get("disabled_delta") is not None)
    best_scale = None
    for sc in [float(x) for x in args.scale_sweep.split(",") if x.strip()]:
        wins = 0
        checks = 0
        for name, ev in evals.items():
            row = next((r for r in ev["scales"] if abs(r["scale"] - sc) < 1e-6), None)
            if not row:
                continue
            checks += 1
            if row["beats_base"] and row["beats_shuffled"]:
                wins += 1
        if checks > 0 and wins == checks and best_scale is None:
            best_scale = sc
    if best_scale is None and "eval_tech" in evals:
        for row in evals["eval_tech"]["scales"]:
            if row["beats_base"] and row["beats_shuffled"]:
                best_scale = row["scale"]
                break

    promoted = bool(disabled_ok and best_scale is not None)

    print("\n=== [Stage 5/5] Register Adapter Card ===")
    card = {
        "schema": "rookery.engram.registry.v1",
        "adapter_path": str(out_path),
        "mode": args.mode,
        "provider": prov,
        "host": host,
        "addressing_parity": "100.00%",
        "evals": evals,
        "recommended_scale": best_scale if best_scale is not None else 0.0,
        "promoted": promoted,
        "elapsed_sec": round(time.time() - t0, 1),
        "serve_command": (
            f"{build_dir / 'rookery-server'} -m {args.model} --gpus {args.gpus} -ts {args.split} "
            f"--engram {args.engram} --engram-adapter {out_path} --engram-scale {best_scale or 1.0}"
        ),
    }
    card_path = str(out_path) + ".json"
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"Registered adapter card -> {card_path} (promoted={promoted}, recommended_scale={card['recommended_scale']})")


def main():
    ap = argparse.ArgumentParser(description="End-to-end rookery Engram calibration, 4-way gating & registration pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect", help="Inspect provider table, host GGUF, and/or .rkad adapter")
    p_ins.add_argument("--build-dir", default="build")
    p_ins.add_argument("--engram", default="")
    p_ins.add_argument("--model", default="")
    p_ins.add_argument("--adapter", default="")
    p_ins.set_defaults(func=cmd_inspect)

    p_bld = sub.add_parser("build", help="Run inspect -> verify -> calibrate/train -> 4-way evaluate -> register")
    p_bld.add_argument("--build-dir", default="build")
    p_bld.add_argument("--model", required=True)
    p_bld.add_argument("--engram", required=True)
    p_bld.add_argument("--mode", choices=["level2", "level3"], default="level2")
    p_bld.add_argument("--corpus", required=True)
    p_bld.add_argument("--out", required=True)
    p_bld.add_argument("--gpus", default="0,1")
    p_bld.add_argument("--split", default="20,28")
    p_bld.add_argument("--max-tokens", type=int, default=16000)
    p_bld.add_argument("--rank", type=int, default=128)
    p_bld.add_argument("--base-scale", type=float, default=0.02)
    p_bld.add_argument("--src-layer", type=int, default=0)
    p_bld.add_argument("--ce-layer", type=int, default=-1)
    p_bld.add_argument("--epochs", type=int, default=6)
    p_bld.add_argument("--delta-clip", type=float, default=0.15)
    p_bld.add_argument("--safetensors-dir", default="")
    p_bld.add_argument("--python", default=sys.executable)
    p_bld.add_argument("--train-cuda-device", default="")
    p_bld.add_argument("--reuse-rkce", action="store_true")
    p_bld.add_argument("--eval-tech", default="")
    p_bld.add_argument("--eval-prose", default="")
    p_bld.add_argument("--eval-factual", default="")
    p_bld.add_argument("--eval-tokens", type=int, default=2048)
    p_bld.add_argument("--scale-sweep", default="0.5,1.0,1.5")
    p_bld.set_defaults(func=cmd_build)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
