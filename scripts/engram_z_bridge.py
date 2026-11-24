#!/usr/bin/env python3
"""
Canonical Shared Memory Representation (`engram-z-v1`) for True N + M Cross-Model Engram Composition.

Architecture:
  Engram Provider E_i  --[Exporter .rkex]-->  Canonical Space Z (engram-z-v1, d_z)  --[Importer .rkim]-->  Host Model H_j

  1. `fit-exporter`: Maps any Engram table's rows into the shared canonical space Z (`z = (e - mu_E) @ W_ex^T`),
                     using either the anchor table's whitened semantic PCA/CCA basis or Ridge alignment onto Z
                     on a shared probe corpus. Independent of all host models!
  2. `fit-importer`: Maps canonical Z into a host model's residual space (Level-2 delta-Ridge OR Level-3 gated
                     temp-calibrated CE+KL). Independent of all Engram providers!
  3. `compose`:      Algebraically folds ANY (Exporter .rkex, Importer .rkim) pair sharing the same canonical
                     `z_space_id` and `d_z` into a standard, zero-overhead `RKAD v3` adapter file (`A_folded = A_im @ W_ex`)!
                     Works even for (Provider, Host) pairs that have NEVER been trained together!
"""
import argparse
import json
import math
import os
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F

from train_engram_bridge import (
    RookeryGatedBridge,
    evaluate as eval_l3,
    load_gguf_tail_head,
    load_maxwell_head,
    load_rkce,
    maxwell_head_logits,
)


def read_rkpr(path: str, hook_idx: int = 0):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"RKPR":
            raise ValueError(f"Invalid RKPR magic in {path}: {magic!r}")
        ver, n_hooks, d_eng, d_host = struct.unpack("<4I", f.read(16))
        (total_rec,) = struct.unpack("<Q", f.read(8))
        metas = [struct.unpack("<2i", f.read(8)) for _ in range(n_hooks)]
        e_chunks, h_chunks = [], []
        while True:
            raw = f.read(8)
            if len(raw) < 8:
                break
            hi, nt = struct.unpack("<2i", raw)
            if nt <= 0:
                break
            eb = f.read(nt * d_eng * 4)
            hb = f.read(nt * d_host * 4)
            if len(eb) < nt * d_eng * 4 or len(hb) < nt * d_host * 4:
                break
            if hi == hook_idx:
                e_chunks.append(np.frombuffer(eb, dtype=np.float32).reshape(nt, d_eng).copy())
                h_chunks.append(np.frombuffer(hb, dtype=np.float32).reshape(nt, d_host).copy())
    E = np.concatenate(e_chunks, axis=0)
    H = np.concatenate(h_chunks, axis=0)
    return E, H, dict(target_layer=metas[hook_idx][0], src_layer=metas[hook_idx][1], d_eng=d_eng, d_host=d_host)


def solve_ridge_gpu(X_tr: np.ndarray, Y_tr: np.ndarray, X_va: np.ndarray, Y_va: np.ndarray,
                    lams=(0.2, 0.5, 1.0, 2.0, 5.0, 15.0, 50.0), device="cuda:0"):
    """Closed-form GPU Ridge regression with automatic validation R^2 lambda selection.
    Returns W [d_out, d_in], best_lam, train_r2, val_r2.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    Xtr = torch.from_numpy(X_tr).to(dev, dtype=torch.float32)
    Ytr = torch.from_numpy(Y_tr).to(dev, dtype=torch.float32)
    Xva = torch.from_numpy(X_va).to(dev, dtype=torch.float32)
    Yva = torch.from_numpy(Y_va).to(dev, dtype=torch.float32)

    d_in = Xtr.shape[1]
    XtX = Xtr.t() @ Xtr
    XtY = Xtr.t() @ Ytr
    mean_diag = float(XtX.diag().mean().item())
    eye = torch.eye(d_in, device=dev, dtype=torch.float32)

    yva_c = Yva - Yva.mean(dim=0, keepdim=True)
    va_tot = float((yva_c ** 2).sum().item())

    best_lam = lams[0]
    best_vr2 = -1e9
    for lam in lams:
        reg = max(1e-5, lam * mean_diag)
        L = torch.linalg.cholesky(XtX + reg * eye)
        W_t = torch.cholesky_solve(XtY, L)  # [d_in, d_out]
        pred = Xva @ W_t
        vr2 = 1.0 - float(((Yva - pred) ** 2).sum().item()) / max(va_tot, 1e-12)
        if vr2 > best_vr2:
            best_vr2 = vr2
            best_lam = lam

    # Refit on full (train + val) with best_lam
    Xall = torch.cat([Xtr, Xva], dim=0)
    Yall = torch.cat([Ytr, Yva], dim=0)
    XtX_a = Xall.t() @ Xall
    XtY_a = Xall.t() @ Yall
    reg_a = max(1e-5, best_lam * float(XtX_a.diag().mean().item()))
    L_a = torch.linalg.cholesky(XtX_a + reg_a * eye)
    W_t = torch.cholesky_solve(XtY_a, L_a)  # [d_in, d_out]
    pred_a = Xall @ W_t
    yall_c = Yall - Yall.mean(dim=0, keepdim=True)
    tr_r2 = 1.0 - float(((Yall - pred_a) ** 2).sum().item()) / max(float((yall_c ** 2).sum().item()), 1e-12)
    return W_t.t().contiguous().cpu().numpy().astype(np.float32), best_lam, tr_r2, best_vr2


def save_json_npz(path: str, meta: dict, arrays: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Single self-contained binary file: magic[4] + u32 json_len + json + float32 arrays in manifest order
    magic = meta.pop("_magic").encode("ascii")
    meta["arrays"] = {k: list(v.shape) for k, v in arrays.items()}
    mj = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", len(mj)))
        f.write(mj)
        for k, arr in arrays.items():
            f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def load_json_npz(path: str, expected_magic: str):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != expected_magic.encode("ascii"):
            raise ValueError(f"Expected magic {expected_magic}, got {magic!r} in {path}")
        (mlen,) = struct.unpack("<I", f.read(4))
        meta = json.loads(f.read(mlen).decode("utf-8"))
        arrays = {}
        for k, shape in meta["arrays"].items():
            n = 1
            for d in shape:
                n *= d
            arrays[k] = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(shape).copy()
    return meta, arrays


def cmd_fit_anchor_exporter(args):
    """Define canonical Z (engram-z-v1) via whitened top-d_z SVD subspace of the anchor Engram table,
    and save both the anchor Exporter (.rkex) and the canonical probe targets Z_probe.npy!
    """
    t0 = time.time()
    E, _, info = read_rkpr(args.rkpr, args.hook_idx)
    N, d_eng = E.shape
    mu_e = E.mean(axis=0, dtype=np.float64).astype(np.float32)
    Ec = E - mu_e

    # Compute top-d_z principal directions of Ec on GPU via covariance eigendecomposition
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(Ec).to(dev, dtype=torch.float32)
    cov = (X.t() @ X) / float(N)
    evals, evecs = torch.linalg.eigh(cov)
    idx = torch.argsort(evals, descending=True)[: args.d_z]
    top_vals = torch.clamp(evals[idx], min=1e-6)
    top_vecs = evecs[:, idx]  # [d_eng, d_z]
    var_kept = float(top_vals.sum().item() / torch.clamp(evals, min=0.0).sum().item())

    # Unit-variance whitened canonical basis: Z = Ec @ W_ex^T has Cov(Z) = I_{d_z}
    W_ex = (top_vecs / torch.sqrt(top_vals).unsqueeze(0)).t().contiguous().cpu().numpy().astype(np.float32)  # [d_z, d_eng]
    Z_probe = (Ec @ W_ex.T).astype(np.float32)

    meta = {
        "_magic": "RKEX",
        "schema": "rookery.engram.exporter.v1",
        "z_space_id": args.z_space_id,
        "d_z": args.d_z,
        "provider_arch": args.provider_arch,
        "provider_fingerprint": args.provider_fp,
        "src_engram_layer": info["src_layer"],
        "d_engram": d_eng,
        "is_anchor": True,
        "variance_explained": round(var_kept, 4),
    }
    save_json_npz(args.out, meta, {"mu_e": mu_e, "W_ex": W_ex})
    if args.save_z_probe:
        np.save(args.save_z_probe, Z_probe)
    print(f"[Anchor Exporter] Saved {args.out} (z_space={args.z_space_id}, d_engram={d_eng} -> d_z={args.d_z}, "
          f"var_kept={100*var_kept:.1f}%, N={N}) in {time.time()-t0:.2f}s")


def cmd_align_exporter(args):
    """Align a NEW independent Engram provider E_new onto an existing canonical Z_probe.npy!
    Knows NOTHING about any host model — pure provider -> canonical Z alignment.
    """
    t0 = time.time()
    E, _, info = read_rkpr(args.rkpr, args.hook_idx)
    Z_probe = np.load(args.z_probe)
    N = min(E.shape[0], Z_probe.shape[0])
    E = E[:N]
    Z = Z_probe[:N]
    d_eng = E.shape[1]
    d_z = Z.shape[1]

    n_val = max(256, N // 10)
    n_tr = N - n_val
    mu_e = E[:n_tr].mean(axis=0, dtype=np.float64).astype(np.float32)
    Ec = E - mu_e

    W_ex, best_lam, tr_r2, val_r2 = solve_ridge_gpu(Ec[:n_tr], Z[:n_tr], Ec[n_tr:], Z[n_tr:], device=args.device)

    # Also compute mean token-wise cosine similarity on held-out validation split!
    pred_va = Ec[n_tr:] @ W_ex.T
    num = (pred_va * Z[n_tr:]).sum(axis=-1)
    den = np.linalg.norm(pred_va, axis=-1) * np.linalg.norm(Z[n_tr:], axis=-1) + 1e-8
    val_cos = float((num / den).mean())

    meta = {
        "_magic": "RKEX",
        "schema": "rookery.engram.exporter.v1",
        "z_space_id": args.z_space_id,
        "d_z": d_z,
        "provider_arch": args.provider_arch,
        "provider_fingerprint": args.provider_fp,
        "src_engram_layer": info["src_layer"],
        "d_engram": d_eng,
        "is_anchor": False,
        "align_lam": best_lam,
        "align_train_r2": round(tr_r2, 4),
        "align_val_r2": round(val_r2, 4),
        "align_val_cos": round(val_cos, 4),
    }
    save_json_npz(args.out, meta, {"mu_e": mu_e, "W_ex": W_ex})
    print(f"[Aligned Exporter] Saved {args.out} ({args.provider_arch} {d_eng} -> {args.z_space_id} d_z={d_z}, "
          f"lam={best_lam}, train_R2={tr_r2:.4f}, val_R2={val_r2:.4f}, val_cos={val_cos:.4f}) in {time.time()-t0:.2f}s")


def cmd_fit_importer_l2(args):
    """Fit a Host Model Level-2 Delta-Ridge Importer (.rkim): Z -> H_host_delta.
    Takes either precomputed Z_probe OR projects E from rkpr via --anchor-exporter!
    """
    t0 = time.time()
    E, H, info = read_rkpr(args.rkpr, args.hook_idx)
    if args.z_probe:
        Z_probe = np.load(args.z_probe)
    elif args.anchor_exporter:
        ex_meta, ex_arr = load_json_npz(args.anchor_exporter, "RKEX")
        Z_probe = ((E - ex_arr["mu_e"]) @ ex_arr["W_ex"].T).astype(np.float32)
    else:
        raise ValueError("fit-importer-l2 requires --z-probe or --anchor-exporter")
    N = min(H.shape[0], Z_probe.shape[0])
    H = H[:N]
    Z = Z_probe[:N]
    d_z = Z.shape[1]
    d_host = H.shape[1]

    n_val = max(256, N // 10)
    n_tr = N - n_val
    mu_h = H[:n_tr].mean(axis=0, dtype=np.float64).astype(np.float32)
    Hc = H - mu_h

    W_im, best_lam, tr_r2, val_r2 = solve_ridge_gpu(Z[:n_tr], Hc[:n_tr], Z[n_tr:], Hc[n_tr:], device=args.device)

    meta = {
        "_magic": "RKIM",
        "schema": "rookery.engram.importer.v1",
        "z_space_id": args.z_space_id,
        "d_z": d_z,
        "host_arch": args.host_arch,
        "host_n_layers": args.host_n_layers,
        "host_d_embd": d_host,
        "host_hc_mult": args.host_hc_mult,
        "target_host_layer": info["target_layer"],
        "mode": "level2_linear",
        "scale": args.scale,
        "delta_clip": args.delta_clip,
        "lam": best_lam,
        "train_r2": round(tr_r2, 4),
        "val_r2": round(val_r2, 4),
    }
    save_json_npz(args.out, meta, {"W_im": W_im})
    print(f"[Host Importer L2] Saved {args.out} ({args.z_space_id} d_z={d_z} -> {args.host_arch} L{info['target_layer']} "
          f"d={d_host}, lam={best_lam}, train_R2={tr_r2:.4f}, val_R2={val_r2:.4f}) in {time.time()-t0:.2f}s")


def cmd_fit_importer_l3(args):
    """Fit a Host Model Level-3 Temp-Calibrated Gated Importer (.rkim): Z -> H_host (L44/tail CE+KL).
    Trains directly on canonical Z = (E_anchor - mu_anchor) @ W_ex_anchor^T!
    """
    t0 = time.time()
    ex_meta, ex_arr = load_json_npz(args.anchor_exporter, "RKEX")
    mu_anchor = ex_arr["mu_e"]
    W_ex = ex_arr["W_ex"]  # [d_z, d_eng]
    d_z = ex_meta["d_z"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    h_all, e_all, y_all, meta_ce = load_rkce(args.rkce, args.src_layer)
    # Project anchor E -> canonical Z on the fly (chunked to keep RAM tidy)
    N = h_all.shape[0]
    z_all = np.empty((N, d_z), dtype=np.float16)
    W_ex_t = torch.from_numpy(W_ex).to(device=device, dtype=torch.float32)
    mu_t = torch.from_numpy(mu_anchor).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, N, 4096):
            j = min(N, i + 4096)
            eb = torch.from_numpy(e_all[i:j]).to(device=device, dtype=torch.float32) - mu_t
            zb = F.linear(eb, W_ex_t)
            z_all[i:j] = zb.to(torch.float16).cpu().numpy()

    n_val = max(256, int(N * 0.1))
    n_tr = N - n_val
    h_tr, z_tr, y_tr = h_all[:n_tr], z_all[:n_tr], y_all[:n_tr]
    h_va, z_va, y_va = h_all[n_tr:], z_all[n_tr:], y_all[n_tr:]
    zero_mu_z = torch.zeros(d_z, device=device, dtype=torch.float32)

    if args.safetensors_dir:
        head_tensors = load_maxwell_head(args.safetensors_dir, device)
    else:
        head_tensors = load_gguf_tail_head(args.gguf, device)
    hc_fn, hc_scale, hc_base, out_norm, lm_head = head_tensors

    # Optimal scalar base temperature tau*
    sub_n = min(n_tr, 16384)
    taus = [0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4]
    nll_sums = [0.0 for _ in taus]
    with torch.no_grad():
        for i in range(0, sub_n, 256):
            j = min(sub_n, i + 256)
            hb = torch.from_numpy(h_tr[i:j]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y_tr[i:j]).to(device=device, dtype=torch.long)
            bl = maxwell_head_logits(hb, hc_fn, hc_scale, hc_base, out_norm, lm_head)
            for ti, tv in enumerate(taus):
                nll_sums[ti] += F.cross_entropy(bl / tv, yb, reduction="sum").item()
    train_tau = taus[int(np.argmin(nll_sums))]
    print(f"[Host Importer L3] Canonical Z (d_z={d_z}) -> {args.host_arch} L{meta_ce['target_layer']}, tau*={train_tau:.2f}")

    bridge = RookeryGatedBridge(d_z, meta_ce["d_host"], meta_ce["hc"], args.rank, args.scale, args.dropout).to(device)
    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=2e-2)
    total_steps = args.epochs * ((n_tr + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=args.lr * 0.05)

    b_nll, b_ppl, b_t1, _, _, _ = eval_l3(bridge, h_va, z_va, y_va, zero_mu_z, head_tensors)
    print(f"[Epoch 0] val BASE: PPL={b_ppl:.3f} top1={b_t1:.2f}%")

    best_score = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}
    rng = np.random.default_rng(42)

    for ep in range(1, args.epochs + 1):
        bridge.train()
        order = rng.permutation(n_tr)
        for i in range(0, n_tr, args.batch_size):
            idx = order[i : min(n_tr, i + args.batch_size)]
            hb = torch.from_numpy(h_tr[idx]).to(device=device, dtype=torch.float32)
            zb = torch.from_numpy(z_tr[idx]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y_tr[idx]).to(device=device, dtype=torch.long)

            with torch.no_grad():
                base_logits = maxwell_head_logits(hb, hc_fn, hc_scale, hc_base, out_norm, lm_head)
                base_logp = F.log_softmax(base_logits / train_tau, dim=-1)
                base_p = base_logp.exp()

            h_out, rel_norm, _ = bridge(hb, zb, return_stats=True)
            logp = F.log_softmax(maxwell_head_logits(h_out, hc_fn, hc_scale, hc_base, out_norm, lm_head) / train_tau, dim=-1)
            ce = F.nll_loss(logp, yb)
            kl = (base_p * (base_logp - logp)).sum(dim=-1).mean()
            loss = ce + args.kl_beta * kl + args.delta_reg * (rel_norm ** 2)

            if args.shuf_kl_beta > 0.0 and zb.shape[0] > 1:
                perm = torch.roll(torch.arange(zb.shape[0], device=device), shifts=1)
                shuf_logp = F.log_softmax(
                    maxwell_head_logits(bridge(hb, zb[perm]), hc_fn, hc_scale, hc_base, out_norm, lm_head) / train_tau, dim=-1
                )
                skl = (base_p * (base_logp - shuf_logp)).sum(dim=-1).mean()
                loss = loss + args.shuf_kl_beta * skl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
            opt.step()
            sched.step()

        _, _, _, r_nll, r_ppl, r_t1 = eval_l3(bridge, h_va, z_va, y_va, zero_mu_z, head_tensors, shuffled=False)
        _, _, _, s_nll, s_ppl, _ = eval_l3(bridge, h_va, z_va, y_va, zero_mu_z, head_tensors, shuffled=True)
        print(f"[Epoch {ep:2d}] val REAL: PPL={r_ppl:.3f} (vs base {r_ppl-b_ppl:+.3f}, top1={r_t1:.2f}%) | "
              f"SHUF: PPL={s_ppl:.3f} (shuf-real={s_ppl-r_ppl:+.3f})")
        if r_nll < s_nll and r_nll < best_score:
            best_score = r_nll
            best_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}

    bridge.load_state_dict(best_state)
    meta = {
        "_magic": "RKIM",
        "schema": "rookery.engram.importer.v1",
        "z_space_id": ex_meta["z_space_id"],
        "d_z": d_z,
        "host_arch": args.host_arch,
        "host_n_layers": args.host_n_layers,
        "host_d_embd": meta_ce["d_host"],
        "host_hc_mult": meta_ce["hc"],
        "target_host_layer": meta_ce["target_layer"],
        "mode": "level3_gated",
        "rank": args.rank,
        "scale": args.scale,
        "delta_clip": args.delta_clip,
        "val_ppl": round(math.exp(best_score), 4),
    }
    arrays = {
        "A_v": bridge.A_v.detach().cpu().float().numpy(),
        "B_v": bridge.B_v.detach().cpu().float().numpy(),
        "A_k": bridge.A_k.detach().cpu().float().numpy(),
        "B_k": bridge.B_k.detach().cpu().float().numpy(),
        "gate_w": bridge.gate_w.detach().cpu().float().numpy(),
    }
    save_json_npz(args.out, meta, arrays)
    print(f"[Host Importer L3] Saved {args.out} (best val PPL={math.exp(best_score):.3f} vs base {b_ppl:.3f}) in {time.time()-t0:.1f}s")


def cmd_compose(args):
    """Algebraically fold ANY Exporter (.rkex) + ANY Importer (.rkim) sharing canonical z_space_id & d_z
    into a standard, ZERO-OVERHEAD RKAD v3 adapter file!
    """
    ex_meta, ex_arr = load_json_npz(args.exporter, "RKEX")
    im_meta, im_arr = load_json_npz(args.importer, "RKIM")

    if ex_meta["z_space_id"] != im_meta["z_space_id"]:
        raise ValueError(f"Canonical Z space mismatch: exporter={ex_meta['z_space_id']} vs importer={im_meta['z_space_id']}")
    if ex_meta["d_z"] != im_meta["d_z"]:
        raise ValueError(f"Canonical Z dim mismatch: exporter d_z={ex_meta['d_z']} vs importer d_z={im_meta['d_z']}")

    mu_e = ex_arr["mu_e"]  # [d_eng]
    W_ex = ex_arr["W_ex"]  # [d_z, d_eng]
    d_eng = ex_meta["d_engram"]
    d_host = im_meta["host_d_embd"]
    mode = im_meta["mode"]

    manifest = {
        "schema": "rookery.engram.adapter.v3",
        "provider_arch": ex_meta["provider_arch"],
        "provider_mode": "ngram_hash",
        "provider_fingerprint": ex_meta["provider_fingerprint"],
        "host_arch": im_meta["host_arch"],
        "host_n_layers": im_meta["host_n_layers"],
        "host_d_embd": d_host,
        "host_hc_mult": im_meta["host_hc_mult"],
        "calibration": f"canonical_z_folded:{ex_meta['z_space_id']}:{mode}",
    }
    m_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

    if mode == "level2_linear":
        # Exact linear fold: W_eff = W_im [d_host, d_z] @ W_ex [d_z, d_eng] -> SVD into rank r!
        W_im = im_arr["W_im"]
        W_eff = W_im @ W_ex  # [d_host, d_eng]
        U, S, Vt = np.linalg.svd(W_eff, full_matrices=False)
        r = min(args.rank, len(S))
        sq = np.sqrt(np.maximum(S[:r], 0.0)).astype(np.float32)
        B_v = (U[:, :r] * sq[None, :]).astype(np.float32)
        A_v = (sq[:, None] * Vt[:r, :]).astype(np.float32)

        delta_clip = float(im_meta.get("delta_clip", 0.0))
        hflags = 1 | (4 if delta_clip > 0.0 else 0)
        with open(args.out, "wb") as f:
            f.write(b"RKAD")
            f.write(struct.pack("<6I", 3, 1, d_eng, d_host, r, 0))
            f.write(struct.pack("<I", len(m_bytes)))
            f.write(m_bytes)
            f.write(struct.pack("<2if2If", im_meta["target_host_layer"], ex_meta["src_engram_layer"],
                                float(im_meta["scale"]), r, hflags, delta_clip))
            f.write(mu_e.astype(np.float32).tobytes())
            f.write(A_v.tobytes())
            f.write(B_v.tobytes())
    elif mode == "level3_gated":
        # Exact affine-before-nonlinearity fold:
        #   A_v_fold = A_v [r, d_z] @ W_ex [d_z, d_eng] -> [r, d_eng]
        #   A_k_fold = A_k [r, d_z] @ W_ex [d_z, d_eng] -> [r, d_eng]
        #   B_v, B_k, gate_w are UNCHANGED! Zero extra compute at inference!
        A_v_fold = (im_arr["A_v"] @ W_ex).astype(np.float32)
        A_k_fold = (im_arr["A_k"] @ W_ex).astype(np.float32)
        B_v = im_arr["B_v"].astype(np.float32)
        B_k = im_arr["B_k"].astype(np.float32)
        gate_w = im_arr["gate_w"].astype(np.float32)
        r = int(im_meta["rank"])
        delta_clip = float(im_meta.get("delta_clip", 0.0))
        hflags = 1 | 2 | (4 if delta_clip > 0.0 else 0)
        with open(args.out, "wb") as f:
            f.write(b"RKAD")
            f.write(struct.pack("<6I", 3, 1, d_eng, d_host, r, 0))
            f.write(struct.pack("<I", len(m_bytes)))
            f.write(m_bytes)
            f.write(struct.pack("<2if2If", im_meta["target_host_layer"], ex_meta["src_engram_layer"],
                                float(im_meta["scale"]), r, hflags, delta_clip))
            f.write(mu_e.astype(np.float32).tobytes())
            f.write(A_v_fold.tobytes())
            f.write(B_v.tobytes())
            f.write(A_k_fold.tobytes())
            f.write(B_k.tobytes())
            f.write(gate_w.tobytes())
    else:
        raise ValueError(f"Unknown importer mode: {mode}")

    print(f"[Compose N+M -> RKAD v3] Folded ({ex_meta['provider_arch']} -> {ex_meta['z_space_id']} -> "
          f"{im_meta['host_arch']} L{im_meta['target_host_layer']}, mode={mode}) -> {args.out}")


def cmd_compose_multi(args):
    """Algebraically fold ONE Exporter (.rkex) with MULTIPLE Importers (.rkim, e.g. L1 linear + L44 gated)
    into a single multi-hook, ZERO-OVERHEAD RKAD v3 adapter file!
    """
    ex_meta, ex_arr = load_json_npz(args.exporter, "RKEX")
    mu_e = ex_arr["mu_e"]
    W_ex = ex_arr["W_ex"]
    d_eng = ex_meta["d_engram"]

    importers = [load_json_npz(p, "RKIM") for p in args.importers]
    if not importers:
        raise ValueError("compose-multi requires at least one --importer")

    im0_meta = importers[0][0]
    d_host = im0_meta["host_d_embd"]
    for im_meta, _ in importers:
        if ex_meta["z_space_id"] != im_meta["z_space_id"] or ex_meta["d_z"] != im_meta["d_z"]:
            raise ValueError(f"Canonical Z mismatch: exporter={ex_meta['z_space_id']}({ex_meta['d_z']}) "
                             f"vs importer={im_meta['z_space_id']}({im_meta['d_z']})")
        if im_meta["host_arch"] != im0_meta["host_arch"] or im_meta["host_d_embd"] != d_host:
            raise ValueError("All importers in compose-multi must target the same host model")

    modes = "+".join(m["mode"] for m, _ in importers)
    manifest = {
        "schema": "rookery.engram.adapter.v3",
        "provider_arch": ex_meta["provider_arch"],
        "provider_mode": "ngram_hash",
        "provider_fingerprint": ex_meta["provider_fingerprint"],
        "host_arch": im0_meta["host_arch"],
        "host_n_layers": im0_meta["host_n_layers"],
        "host_d_embd": d_host,
        "host_hc_mult": im0_meta["host_hc_mult"],
        "calibration": f"canonical_z_multi:{ex_meta['z_space_id']}:{modes}",
    }
    m_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

    with open(args.out, "wb") as f:
        f.write(b"RKAD")
        f.write(struct.pack("<6I", 3, len(importers), d_eng, d_host, args.rank, 0))
        f.write(struct.pack("<I", len(m_bytes)))
        f.write(m_bytes)
        for idx, (im_meta, im_arr) in enumerate(importers):
            mode = im_meta["mode"]
            sc_mult = args.scale_mults[idx] if idx < len(args.scale_mults) else 1.0
            eff_scale = float(im_meta["scale"]) * sc_mult
            delta_clip = float(im_meta.get("delta_clip", 0.0))
            if mode == "level2_linear":
                W_eff = im_arr["W_im"] @ W_ex
                U, S, Vt = np.linalg.svd(W_eff, full_matrices=False)
                r = min(args.rank, len(S))
                sq = np.sqrt(np.maximum(S[:r], 0.0)).astype(np.float32)
                B_v = (U[:, :r] * sq[None, :]).astype(np.float32)
                A_v = (sq[:, None] * Vt[:r, :]).astype(np.float32)
                hflags = 1 | (4 if delta_clip > 0.0 else 0)
                f.write(struct.pack("<2if2If", im_meta["target_host_layer"], ex_meta["src_engram_layer"],
                                    eff_scale, r, hflags, delta_clip))
                f.write(mu_e.astype(np.float32).tobytes())
                f.write(A_v.tobytes())
                f.write(B_v.tobytes())
            elif mode == "level3_gated":
                A_v_fold = (im_arr["A_v"] @ W_ex).astype(np.float32)
                A_k_fold = (im_arr["A_k"] @ W_ex).astype(np.float32)
                r = int(im_meta["rank"])
                hflags = 1 | 2 | (4 if delta_clip > 0.0 else 0)
                f.write(struct.pack("<2if2If", im_meta["target_host_layer"], ex_meta["src_engram_layer"],
                                    eff_scale, r, hflags, delta_clip))
                f.write(mu_e.astype(np.float32).tobytes())
                f.write(A_v_fold.tobytes())
                f.write(im_arr["B_v"].astype(np.float32).tobytes())
                f.write(A_k_fold.tobytes())
                f.write(im_arr["B_k"].astype(np.float32).tobytes())
                f.write(im_arr["gate_w"].astype(np.float32).tobytes())
            else:
                raise ValueError(f"Unknown importer mode: {mode}")

    layers_str = ",".join(f"L{m['target_host_layer']}({m['mode']})" for m, _ in importers)
    print(f"[Compose-Multi N+M -> RKAD v3] Folded ({ex_meta['provider_arch']} -> {ex_meta['z_space_id']} -> "
          f"{im0_meta['host_arch']} [{layers_str}]) -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Canonical Shared Memory Z (engram-z-v1) N+M Exporter/Importer/Composer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("fit-anchor-exporter")
    p1.add_argument("--rkpr", required=True)
    p1.add_argument("--hook-idx", type=int, default=0)
    p1.add_argument("--z-space-id", default="engram-z-v1")
    p1.add_argument("--d-z", type=int, default=256)
    p1.add_argument("--provider-arch", required=True)
    p1.add_argument("--provider-fp", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--save-z-probe", default="")
    p1.add_argument("--device", default="cuda:0")
    p1.set_defaults(func=cmd_fit_anchor_exporter)

    p2 = sub.add_parser("align-exporter")
    p2.add_argument("--rkpr", required=True)
    p2.add_argument("--hook-idx", type=int, default=0)
    p2.add_argument("--z-probe", required=True)
    p2.add_argument("--z-space-id", default="engram-z-v1")
    p2.add_argument("--provider-arch", required=True)
    p2.add_argument("--provider-fp", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--device", default="cuda:0")
    p2.set_defaults(func=cmd_align_exporter)

    p3 = sub.add_parser("fit-importer-l2")
    p3.add_argument("--rkpr", required=True)
    p3.add_argument("--hook-idx", type=int, default=0)
    p3.add_argument("--z-probe", default="")
    p3.add_argument("--anchor-exporter", default="")
    p3.add_argument("--z-space-id", default="engram-z-v1")
    p3.add_argument("--host-arch", required=True)
    p3.add_argument("--host-n-layers", type=int, required=True)
    p3.add_argument("--host-hc-mult", type=int, required=True)
    p3.add_argument("--scale", type=float, default=0.02)
    p3.add_argument("--delta-clip", type=float, default=0.0)
    p3.add_argument("--out", required=True)
    p3.add_argument("--device", default="cuda:0")
    p3.set_defaults(func=cmd_fit_importer_l2)

    p4 = sub.add_parser("fit-importer-l3")
    p4.add_argument("--anchor-exporter", required=True)
    p4.add_argument("--rkce", required=True)
    p4.add_argument("--src-layer", type=int, default=0)
    p4.add_argument("--safetensors-dir", default="")
    p4.add_argument("--gguf", default="")
    p4.add_argument("--host-arch", required=True)
    p4.add_argument("--host-n-layers", type=int, required=True)
    p4.add_argument("--rank", type=int, default=128)
    p4.add_argument("--scale", type=float, default=0.05)
    p4.add_argument("--delta-clip", type=float, default=0.15)
    p4.add_argument("--epochs", type=int, default=6)
    p4.add_argument("--batch-size", type=int, default=256)
    p4.add_argument("--lr", type=float, default=8e-4)
    p4.add_argument("--kl-beta", type=float, default=0.5)
    p4.add_argument("--shuf-kl-beta", type=float, default=1.5)
    p4.add_argument("--delta-reg", type=float, default=1.0)
    p4.add_argument("--dropout", type=float, default=0.25)
    p4.add_argument("--out", required=True)
    p4.add_argument("--device", default="cuda:0")
    p4.set_defaults(func=cmd_fit_importer_l3)

    p5 = sub.add_parser("compose")
    p5.add_argument("--exporter", required=True)
    p5.add_argument("--importer", required=True)
    p5.add_argument("--rank", type=int, default=128)
    p5.add_argument("--out", required=True)
    p5.set_defaults(func=cmd_compose)

    p6 = sub.add_parser("compose-multi")
    p6.add_argument("--exporter", required=True)
    p6.add_argument("--importers", nargs="+", required=True)
    p6.add_argument("--scale-mults", nargs="*", type=float, default=[])
    p6.add_argument("--rank", type=int, default=128)
    p6.add_argument("--out", required=True)
    p6.set_defaults(func=cmd_compose_multi)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
