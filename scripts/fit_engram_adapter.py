#!/usr/bin/env python3
"""
Closed-form Ridge / SVD low-rank adapter fitter (RKPR v1 -> RKAD v3) + Multi-Hook RKAD v3 Merger.

Supports:
  1. Mean-centered delta-Ridge with held-out 10% validation R^2 (`--auto-lam`) and per-hook auto-gating.
  2. Optional `--engram-kv-gguf`: pre-decodes raw Engram hash vectors through the provider's own
     pre-trained `blk.<L>.engram_kv.weight` (W_V_ds), solves Ridge in the provider's clean semantic space,
     and algebraically folds `A_v @ W_V_norm` directly into a zero-overhead RKAD v3 adapter!
  3. `--merge a.rkad b.rkad ... --out multi.rkad`: merges multiple compatible RKAD v3 adapters
     (e.g. early-layer L1 linear Ridge + final-layer L_final gated bridge) into a single RKAD v3 file.
"""
import argparse
import json
import math
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_engram_bridge import load_provider_engram_kv


def read_rkpr(path: str):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"RKPR", f"Bad magic: {magic}"
        ver, n_hooks, d_engram, d_host = struct.unpack("<4I", f.read(16))
        _total_records = struct.unpack("<Q", f.read(8))[0]
        hook_meta = [struct.unpack("<2i", f.read(8)) for _ in range(n_hooks)]

        per_hook_E = [[] for _ in range(n_hooks)]
        per_hook_H = [[] for _ in range(n_hooks)]
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            hi, nt = struct.unpack("<2i", hdr)
            e_bytes = f.read(nt * d_engram * 4)
            h_bytes = f.read(nt * d_host * 4)
            E = torch.frombuffer(bytearray(e_bytes), dtype=torch.float32).view(nt, d_engram)
            H = torch.frombuffer(bytearray(h_bytes), dtype=torch.float32).view(nt, d_host)
            per_hook_E[hi].append(E)
            per_hook_H[hi].append(H)

    Es = [torch.cat(chunks, dim=0) for chunks in per_hook_E]
    Hs = [torch.cat(chunks, dim=0) for chunks in per_hook_H]
    return ver, n_hooks, d_engram, d_host, hook_meta, Es, Hs


def fit_ridge_centered_autolam(
    E: torch.Tensor,
    H: torch.Tensor,
    rank: int,
    lams: list[float],
    device: str,
    W_V_norm: torch.Tensor = None,
):
    N = E.shape[0]
    n_val = max(256, int(N * 0.1))
    n_tr = N - n_val

    E_tr = E[:n_tr].to(device=device, dtype=torch.float32)
    H_tr = H[:n_tr].to(device=device, dtype=torch.float32)
    E_va = E[n_tr:].to(device=device, dtype=torch.float32)
    H_va = H[n_tr:].to(device=device, dtype=torch.float32)

    mu_e = E_tr.mean(dim=0, keepdim=True)
    mu_h = H_tr.mean(dim=0, keepdim=True)
    Ec_tr = E_tr - mu_e
    Hc_tr = H_tr - mu_h
    Ec_va = E_va - mu_e
    Hc_va = H_va - mu_h

    if W_V_norm is not None:
        X_tr = Ec_tr @ W_V_norm.T  # [n_tr, d_prov]
        X_va = Ec_va @ W_V_norm.T  # [n_val, d_prov]
    else:
        X_tr = Ec_tr
        X_va = Ec_va

    XtX = X_tr.T @ X_tr
    XtH = X_tr.T @ Hc_tr
    diag_mean = XtX.diagonal().mean().clamp(min=1e-6).item()
    ss_tot_tr = (Hc_tr ** 2).sum().item()
    ss_tot_va = (Hc_va ** 2).sum().item()

    best_lam = lams[0]
    best_r2_va = -1e9
    best_W = None
    best_r2_tr = -1e9

    for lam in lams:
        A = XtX.clone()
        A.diagonal().add_(lam * diag_mean)
        W = torch.linalg.solve(A, XtH)  # [d_in, d_host]
        r2_va = 1.0 - ((Hc_va - X_va @ W) ** 2).sum().item() / max(ss_tot_va, 1e-12)
        if r2_va > best_r2_va:
            best_r2_va = r2_va
            best_lam = lam
            best_W = W
            best_r2_tr = 1.0 - ((Hc_tr - X_tr @ W) ** 2).sum().item() / max(ss_tot_tr, 1e-12)

    # Low-rank SVD factorization of W^T [d_host, d_in] = B_v[d_host, r] @ A_v[r, d_in]
    Wt = best_W.T  # [d_host, d_in]
    d_host, d_in = Wt.shape
    if rank > 0 and rank < min(d_host, d_in):
        U, S, Vh = torch.linalg.svd(Wt, full_matrices=False)
        s_sqrt = torch.sqrt(S[:rank])
        B_v = (U[:, :rank] * s_sqrt.unsqueeze(0)).contiguous()
        A_v = (s_sqrt.unsqueeze(1) * Vh[:rank, :]).contiguous()
        r2_lr = 1.0 - ((Hc_va - (X_va @ A_v.T) @ B_v.T) ** 2).sum().item() / max(ss_tot_va, 1e-12)
    else:
        A_v = Wt.contiguous()
        B_v = None
        r2_lr = best_r2_va

    # Algebraically fold provider decoder W_V_norm back into A_v: [rank, d_prov] @ [d_prov, d_eng] -> [rank, d_eng]!
    if W_V_norm is not None:
        A_v = (A_v @ W_V_norm).contiguous()

    return (
        mu_e.squeeze(0).cpu().numpy(),
        A_v.cpu().numpy(),
        B_v.cpu().numpy() if B_v is not None else None,
        best_lam,
        best_r2_tr,
        best_r2_va,
        r2_lr,
    )


def read_rkad_v3_raw(path: str):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"RKAD":
            raise ValueError(f"Not an RKAD file: {path}")
        ver, n_hooks, d_eng, d_host, def_rank, g_flags = struct.unpack("<6I", f.read(24))
        if ver != 3:
            raise ValueError(f"merge requires RKAD v3, got v{ver} in {path}")
        (mlen,) = struct.unpack("<I", f.read(4))
        manifest = json.loads(f.read(mlen).decode("utf-8"))
        hc = int(manifest.get("host_hc_mult", 1))
        hooks = []
        for _ in range(n_hooks):
            hdr_bytes = f.read(24)
            target_l, src_l, scale, h_rank, h_flags, d_clip = struct.unpack("<2if2If", hdr_bytes)
            has_mu = (h_flags & 1) != 0
            has_g = (h_flags & 2) != 0
            n_floats = (d_eng if has_mu else 0) + (h_rank * d_eng + d_host * h_rank if h_rank > 0 else d_host * d_eng)
            if has_g:
                n_floats += (h_rank * d_eng + d_host * h_rank if h_rank > 0 else d_host * d_eng) + hc * d_host
            payload = f.read(n_floats * 4)
            hooks.append((target_l, src_l, scale, h_rank, h_flags, d_clip, hdr_bytes + payload))
    return dict(d_eng=d_eng, d_host=d_host, def_rank=def_rank, g_flags=g_flags, manifest=manifest, hooks=hooks)


def merge_rkad_v3(paths: list[str], out_path: str, scale_overrides: list[float] = None):
    parsed = [read_rkad_v3_raw(p) for p in paths]
    ref = parsed[0]
    for i, item in enumerate(parsed[1:], 1):
        for k in ("d_eng", "d_host"):
            if item[k] != ref[k]:
                raise ValueError(f"Mismatch in {k} between {paths[0]} and {paths[i]}")
        for mk in ("provider_arch", "provider_fingerprint", "host_arch", "host_n_layers", "host_hc_mult"):
            if item["manifest"].get(mk) != ref["manifest"].get(mk):
                raise ValueError(f"Manifest mismatch on '{mk}': {ref['manifest'].get(mk)} vs {item['manifest'].get(mk)}")

    merged_hooks = []
    seen_layers = set()
    for pi, item in enumerate(parsed):
        sc_mult = scale_overrides[pi] if (scale_overrides and pi < len(scale_overrides)) else 1.0
        for target_l, src_l, scale, h_rank, h_flags, d_clip, blob in item["hooks"]:
            if scale == 0.0:
                continue  # skip auto-disabled hooks
            if target_l in seen_layers:
                raise ValueError(f"Duplicate target_host_layer={target_l} while merging {paths}")
            seen_layers.add(target_l)
            if sc_mult != 1.0:
                new_hdr = struct.pack("<2if2If", target_l, src_l, float(scale * sc_mult), h_rank, h_flags, d_clip)
                blob = new_hdr + blob[24:]
            merged_hooks.append((target_l, src_l, scale * sc_mult, h_rank, h_flags, d_clip, blob))

    manifest = dict(ref["manifest"])
    manifest["calibration"] = "merged:" + "+".join(p["manifest"].get("calibration", "rkad") for p in parsed)
    m_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

    with open(out_path, "wb") as f:
        f.write(b"RKAD")
        f.write(struct.pack("<6I", 3, len(merged_hooks), ref["d_eng"], ref["d_host"], ref["def_rank"], ref["g_flags"]))
        f.write(struct.pack("<I", len(m_bytes)))
        f.write(m_bytes)
        for _, _, _, _, _, _, blob in merged_hooks:
            f.write(blob)

    print(f"Merged {len(merged_hooks)} active hook(s) into RKAD v3 -> {out_path}")
    for i, (tl, sl, sc, hr, hf, dc, _) in enumerate(merged_hooks):
        print(f"  hook[{i}]: src_layer={sl} -> host_layer={tl}, rank={hr}, scale={sc:.4f}, gated={bool(hf & 2)}, clip={dc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="", help="Input .rkpr calibration pair dump")
    ap.add_argument("--merge", nargs="+", default=[], help="Merge two or more RKAD v3 files into one multi-hook RKAD v3")
    ap.add_argument("--merge-scales", default="", help="Optional comma-separated per-input scale multipliers for --merge")
    ap.add_argument("--out", required=True, help="Output .rkad adapter binary")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--auto-lam", action="store_true")
    ap.add_argument("--scale", type=float, default=0.02)
    ap.add_argument("--engram-kv-gguf", default="", help="Optional provider GGUF shard with blk.<L>.engram_kv.weight")
    ap.add_argument("--engram-kv-hc", type=int, default=4)
    ap.add_argument("--provider-arch", default="deepseek41")
    ap.add_argument("--provider-fp", default="")
    ap.add_argument("--host-arch", default="bailingmoe3")
    ap.add_argument("--host-n-layers", type=int, default=24)
    ap.add_argument("--host-hc-mult", type=int, default=1)
    ap.add_argument("--delta-clip", type=float, default=0.15)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.merge:
        sc_ov = [float(x) for x in args.merge_scales.split(",") if x.strip()] if args.merge_scales else None
        merge_rkad_v3(args.merge, args.out, sc_ov)
        return

    if not args.pairs:
        raise ValueError("Must provide --pairs <dump.rkpr> or --merge <a.rkad> <b.rkad>")

    ver, n_hooks, d_engram, d_host, hook_meta, Es, Hs = read_rkpr(args.pairs)
    print(f"Loaded {args.pairs}: {n_hooks} hook(s), N={Es[0].shape[0]:,}, d_engram={d_engram} -> d_host={d_host}", flush=True)

    lams = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0] if args.auto_lam else [args.lam]
    kv_tag = "_kvfold" if args.engram_kv_gguf else ""

    manifest = {
        "schema": "rookery.engram.adapter.v3",
        "provider_arch": args.provider_arch,
        "provider_mode": "ngram_hash",
        "provider_fingerprint": args.provider_fp,
        "host_arch": args.host_arch,
        "host_n_layers": args.host_n_layers,
        "host_d_embd": d_host,
        "host_hc_mult": args.host_hc_mult,
        "calibration": f"level2_ridge_N{Es[0].shape[0]}{kv_tag}",
    }
    m_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    hook_flags = 1 | (4 if args.delta_clip > 0.0 else 0)  # centered, non-gated, optional delta_clip

    with open(args.out, "wb") as f:
        f.write(b"RKAD")
        f.write(struct.pack("<6I", 3, n_hooks, d_engram, d_host, args.rank, 0))
        f.write(struct.pack("<I", len(m_bytes)))
        f.write(m_bytes)

        for hi in range(n_hooks):
            target_layer, src_layer = hook_meta[hi]
            W_V_norm = None
            if args.engram_kv_gguf:
                W_V_raw, _, kv_name = load_provider_engram_kv(
                    args.engram_kv_gguf, src_layer, "", args.engram_kv_hc, torch.device(args.device)
                )
                with torch.no_grad():
                    e_sub = Es[hi][:: max(1, Es[hi].shape[0] // 8192)].to(args.device)
                    e_sub = e_sub - e_sub.mean(dim=0, keepdim=True)
                    rms_v = math.sqrt(max((e_sub @ W_V_raw.T).pow(2).mean().item(), 1e-12))
                    W_V_norm = W_V_raw / rms_v
                print(f"  [ProviderKV] hook[{hi}] using {kv_name} (rms_v={rms_v:.4f})", flush=True)

            mu_e, A_v, B_v, lam, r2_tr, r2_va, r2_lr = fit_ridge_centered_autolam(
                Es[hi], Hs[hi], args.rank, lams, args.device, W_V_norm
            )
            eff_scale = args.scale if r2_lr > 0.005 else 0.0
            print(
                f"  hook[{hi}] src={src_layer} -> host L{target_layer}: lam={lam}, "
                f"train R^2={r2_tr:+.4f}, val R^2={r2_va:+.4f}, rank-{args.rank} val R^2={r2_lr:+.4f} "
                f"-> {'ENABLED (scale=' + str(eff_scale) + ')' if eff_scale > 0 else 'AUTO-DISABLED'}",
                flush=True,
            )
            f.write(struct.pack("<2if2If", target_layer, src_layer, float(eff_scale), args.rank, hook_flags, float(args.delta_clip)))
            f.write(mu_e.astype(np.float32).tobytes())
            f.write(A_v.astype(np.float32).tobytes())
            if B_v is not None:
                f.write(B_v.astype(np.float32).tobytes())

    print(f"Wrote RKAD v3 adapter -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
