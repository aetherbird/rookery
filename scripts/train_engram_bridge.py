#!/usr/bin/env python3
"""
Level-3 Supervised Gated Bridge Trainer for Cross-Model Engram Grafting (RKCE v1 -> RKAD v2).

Trains a lightweight low-rank gated Engram bridge (A_v, B_v, A_k, B_k, gate_w) directly against
next-token Cross-Entropy + Host-KL regularization through the host model's EXACT frozen tail
(hc_head + output_norm + lm_head), matching rookery's CUDA kernels formula-for-formula!

Exports directly to RKAD v2 (centered, has_gate=1) for immediate zero-overhead inference in rookery.
"""
import argparse
import json
import math
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


def load_rkce(path: str, src_layer: int):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"RKCE":
            raise ValueError(f"Invalid RKCE magic: {magic!r}")
        ver, n_eng, d_eng, d_host, hc, target_layer = struct.unpack("<IIIIII", f.read(24))
        (n_tokens_hdr,) = struct.unpack("<Q", f.read(8))
        if ver != 1:
            raise ValueError(f"Unsupported RKCE version {ver}")
        if src_layer < 0 or src_layer >= n_eng:
            raise ValueError(f"src_layer {src_layer} out of range [0, {n_eng})")

        targets_list = []
        h_list = []
        e_list = []
        while True:
            raw = f.read(4)
            if len(raw) < 4:
                break
            (nt,) = struct.unpack("<i", raw)
            if nt <= 0:
                break
            tgts = np.frombuffer(f.read(nt * 4), dtype=np.int32).copy()
            h_f16 = np.frombuffer(f.read(nt * hc * d_host * 2), dtype=np.float16).reshape(nt, hc, d_host).copy()
            e_sel = None
            for el in range(n_eng):
                buf = f.read(nt * d_eng * 2)
                if el == src_layer:
                    e_sel = np.frombuffer(buf, dtype=np.float16).reshape(nt, d_eng).copy()
            targets_list.append(tgts)
            h_list.append(h_f16)
            e_list.append(e_sel)

    targets = np.concatenate(targets_list, axis=0)
    h_all = np.concatenate(h_list, axis=0)
    e_all = np.concatenate(e_list, axis=0)
    valid = targets >= 0
    return (
        h_all[valid],
        e_all[valid],
        targets[valid].astype(np.int64),
        dict(n_eng=n_eng, d_eng=d_eng, d_host=d_host, hc=hc, target_layer=target_layer),
    )


def load_maxwell_head(safetensors_dir: str, device: torch.device):
    idx_path = os.path.join(safetensors_dir, "model.safetensors.index.json")
    wmap = json.load(open(idx_path))["weight_map"]

    def get_t(key: str) -> torch.Tensor:
        shard = os.path.join(safetensors_dir, wmap[key])
        with safe_open(shard, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    hc_fn = get_t("model.hc_head.hc_head_fn").to(device=device, dtype=torch.float32)       # [4, hc*d] or [hc*d, 4]
    hc_scale = get_t("model.hc_head.hc_head_scale").to(device=device, dtype=torch.float32) # [1] or scalar
    hc_base = get_t("model.hc_head.hc_head_base").to(device=device, dtype=torch.float32)   # [4]
    out_norm = get_t("model.norm.weight").to(device=device, dtype=torch.float32)           # [d_host]
    lm_head = get_t("lm_head.weight").to(device=device, dtype=torch.float16)               # [vocab, d_host]

    # Normalize hc_fn orientation to [4, hc * d_host]
    if hc_fn.shape[0] != 4 and hc_fn.shape[1] == 4:
        hc_fn = hc_fn.t().contiguous()
    hc_scale = hc_scale.reshape(-1)[0]
    hc_base = hc_base.reshape(4)
    return hc_fn, hc_scale, hc_base, out_norm, lm_head


def maxwell_head_logits(
    h_hc: torch.Tensor,  # [B, 4, d_host] fp32
    hc_fn: torch.Tensor,  # [4, 4*d_host] fp32
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,  # [4] fp32
    out_norm: torch.Tensor,  # [d_host] fp32
    lm_head: torch.Tensor,  # [vocab, d_host] fp16
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
) -> torch.Tensor:
    B, hc, D = h_hc.shape
    flat = h_hc.reshape(B, hc * D)
    inv_flat = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(flat * inv_flat, hc_fn)  # [B, 4]
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps  # [B, 4]
    mixed = (pre.unsqueeze(-1) * h_hc).sum(dim=1)  # [B, D]
    inv_rms = torch.rsqrt(mixed.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
    x_norm = mixed * inv_rms * out_norm
    logits = F.linear(x_norm.to(lm_head.dtype), lm_head).float()
    return logits


class RookeryGatedBridge(nn.Module):
    """Exact PyTorch replica of rookery's engram_gated_inject_kernel (RKAD v2 has_gate=1)."""

    def __init__(self, d_eng: int, d_host: int, hc: int, rank: int, init_scale: float = 0.05, dropout: float = 0.0):
        super().__init__()
        self.d_eng = d_eng
        self.d_host = d_host
        self.hc = hc
        self.rank = rank
        self.fixed_scale = init_scale
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.A_v = nn.Parameter(torch.randn(rank, d_eng) * (1.0 / math.sqrt(d_eng)))
        self.B_v = nn.Parameter(torch.zeros(d_host, rank))  # exact zero-init -> starts at baseline!
        self.A_k = nn.Parameter(torch.randn(rank, d_eng) * (1.0 / math.sqrt(d_eng)))
        self.B_k = nn.Parameter(torch.randn(d_host, rank) * (1.0 / math.sqrt(rank)))
        self.gate_w = nn.Parameter(torch.ones(hc, d_host))

    def forward(self, h_hc: torch.Tensor, e_c: torch.Tensor, rms_eps: float = 1e-6, return_stats: bool = False):
        e_in = self.drop(e_c)
        # v: [B, d_host], k: [B, d_host]
        v = F.linear(self.drop(F.linear(e_in, self.A_v)), self.B_v)
        k = F.linear(self.drop(F.linear(e_in, self.A_k)), self.B_k)

        # Exact engram_gated_inject_kernel math per stream s in 0..hc-1
        k_norm = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + rms_eps)  # [B, D]
        h_norm = h_hc * torch.rsqrt(h_hc.pow(2).mean(dim=-1, keepdim=True) + rms_eps)  # [B, hc, D]
        dot = (h_norm * k_norm.unsqueeze(1) * self.gate_w.unsqueeze(0)).sum(dim=-1)  # [B, hc]
        scaled = dot / math.sqrt(float(self.d_host))
        mag = torch.sqrt(torch.clamp(scaled.abs(), min=1e-6))
        sgn = torch.where(scaled >= 0.0, torch.ones_like(scaled), -torch.ones_like(scaled))
        g = torch.sigmoid(sgn * mag) * self.fixed_scale  # [B, hc]
        delta = g.unsqueeze(-1) * v.unsqueeze(1)
        out = h_hc + delta
        if return_stats:
            h_rms = torch.sqrt(h_hc.pow(2).mean(dim=-1) + rms_eps)
            d_rms = torch.sqrt(delta.pow(2).mean(dim=-1) + rms_eps)
            rel_norm = (d_rms / h_rms).mean()
            return out, rel_norm, g.mean()
        return out


def write_rkad_v3(
    out_path: str,
    target_layer: int,
    src_layer: int,
    mu_e: np.ndarray,
    bridge: RookeryGatedBridge,
    manifest: dict,
    delta_clip: float = 0.0,
):
    m_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    hook_flags = 1 | 2 | (4 if delta_clip > 0.0 else 0)  # bit0=mu_e, bit1=gate, bit2=delta_clip
    with open(out_path, "wb") as f:
        f.write(b"RKAD")
        # version=3, n_hooks=1, d_engram, d_host, default_rank, global_flags=0
        f.write(struct.pack("<6I", 3, 1, bridge.d_eng, bridge.d_host, bridge.rank, 0))
        f.write(struct.pack("<I", len(m_bytes)))
        f.write(m_bytes)
        f.write(struct.pack("<2if2If", target_layer, src_layer, float(bridge.fixed_scale),
                            bridge.rank, hook_flags, float(delta_clip)))
        f.write(mu_e.astype(np.float32).tobytes())
        f.write(bridge.A_v.detach().cpu().float().numpy().tobytes())
        f.write(bridge.B_v.detach().cpu().float().numpy().tobytes())
        f.write(bridge.A_k.detach().cpu().float().numpy().tobytes())
        f.write(bridge.B_k.detach().cpu().float().numpy().tobytes())
        f.write(bridge.gate_w.detach().cpu().float().numpy().tobytes())


def write_rkad_v2(
    out_path: str,
    target_layer: int,
    src_layer: int,
    mu_e: np.ndarray,
    bridge: RookeryGatedBridge,
):
    with open(out_path, "wb") as f:
        f.write(b"RKAD")
        # version=2, n_hooks=1, d_engram, d_host, rank, has_gate=1
        f.write(struct.pack("<6I", 2, 1, bridge.d_eng, bridge.d_host, bridge.rank, 1))
        f.write(struct.pack("<2if", target_layer, src_layer, float(bridge.fixed_scale)))
        f.write(mu_e.astype(np.float32).tobytes())
        f.write(bridge.A_v.detach().cpu().float().numpy().tobytes())
        f.write(bridge.B_v.detach().cpu().float().numpy().tobytes())
        f.write(bridge.A_k.detach().cpu().float().numpy().tobytes())
        f.write(bridge.B_k.detach().cpu().float().numpy().tobytes())
        f.write(bridge.gate_w.detach().cpu().float().numpy().tobytes())


@torch.no_grad()
def evaluate(bridge, h_val, e_val, y_val, mu_e_t, head_tensors, batch_size=256, shuffled=False):
    bridge.eval()
    hc_fn, hc_scale, hc_base, out_norm, lm_head = head_tensors
    device = hc_fn.device
    N = h_val.shape[0]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(1337)) if shuffled else None
    tot_nll = 0.0
    tot_base_nll = 0.0
    hit1 = 0
    base_hit1 = 0
    for i in range(0, N, batch_size):
        j = min(N, i + batch_size)
        hb = torch.from_numpy(h_val[i:j]).to(device=device, dtype=torch.float32)
        eb_idx = perm[i:j].numpy() if shuffled else slice(i, j)
        eb = torch.from_numpy(e_val[eb_idx]).to(device=device, dtype=torch.float32) - mu_e_t
        yb = torch.from_numpy(y_val[i:j]).to(device=device, dtype=torch.long)

        base_logits = maxwell_head_logits(hb, hc_fn, hc_scale, hc_base, out_norm, lm_head)
        tot_base_nll += F.cross_entropy(base_logits, yb, reduction="sum").item()
        base_hit1 += (base_logits.argmax(dim=-1) == yb).sum().item()

        h_out = bridge(hb, eb)
        logits = maxwell_head_logits(h_out, hc_fn, hc_scale, hc_base, out_norm, lm_head)
        tot_nll += F.cross_entropy(logits, yb, reduction="sum").item()
        hit1 += (logits.argmax(dim=-1) == yb).sum().item()

    return (
        tot_base_nll / N,
        math.exp(tot_base_nll / N),
        100.0 * base_hit1 / N,
        tot_nll / N,
        math.exp(tot_nll / N),
        100.0 * hit1 / N,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rkce", required=True, help="Path to RKCE v1 dump")
    ap.add_argument("--safetensors-dir", required=True, help="Directory with Maxwell safetensors + index.json")
    ap.add_argument("--out", required=True, help="Output RKAD v2 path")
    ap.add_argument("--src-layer", type=int, default=0, help="Engram source layer index (0 or 1)")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=2e-2)
    ap.add_argument("--kl-beta", type=float, default=0.25)
    ap.add_argument("--shuf-kl-beta", type=float, default=0.0,
                    help="Contrastive shuffled-batch KL(p_base || p_shuf) penalty to prevent RMSNorm logit-softening shortcuts")
    ap.add_argument("--delta-reg", type=float, default=0.0,
                    help="L2 penalty on mean ||delta||_rms / ||h||_rms to keep perturbations small and well-calibrated")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--rkad-v3", action="store_true", help="Emit self-describing RKAD v3 with strict JSON contract manifest")
    ap.add_argument("--provider-arch", default="deepseek41")
    ap.add_argument("--provider-fingerprint", default="")
    ap.add_argument("--host-arch", default="maxwell-moe")
    ap.add_argument("--host-n-layers", type=int, default=45)
    ap.add_argument("--delta-clip", type=float, default=0.0)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    h_all, e_all, y_all, meta = load_rkce(args.rkce, args.src_layer)
    N = h_all.shape[0]
    n_val = max(256, int(N * args.val_frac))
    n_tr = N - n_val
    print(f"Loaded {args.rkce}: N={N} valid tokens (train={n_tr}, val={n_val}), "
          f"src_layer={args.src_layer}, target_layer={meta['target_layer']}, "
          f"d_eng={meta['d_eng']}, hc={meta['hc']}, d_host={meta['d_host']}")

    h_tr, e_tr, y_tr = h_all[:n_tr], e_all[:n_tr], y_all[:n_tr]
    h_va, e_va, y_va = h_all[n_tr:], e_all[n_tr:], y_all[n_tr:]

    # Exact training-set mean centering vector mu_E
    mu_e = e_tr.astype(np.float64).mean(axis=0).astype(np.float32)
    mu_e_t = torch.from_numpy(mu_e).to(device=device, dtype=torch.float32)

    head_tensors = load_maxwell_head(args.safetensors_dir, device)
    hc_fn, hc_scale, hc_base, out_norm, lm_head = head_tensors

    bridge = RookeryGatedBridge(meta["d_eng"], meta["d_host"], meta["hc"], args.rank, args.scale, args.dropout).to(device)
    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * ((n_tr + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=args.lr * 0.05)

    b_nll, b_ppl, b_t1, r_nll, r_ppl, r_t1 = evaluate(bridge, h_va, e_va, y_va, mu_e_t, head_tensors)
    print(f"[Epoch 0] val BASE: NLL={b_nll:.5f} PPL={b_ppl:.3f} top1={b_t1:.2f}% | init PPL={r_ppl:.3f}")

    best_val_score = float("inf")
    best_val_nll = b_nll
    best_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}

    rng = np.random.default_rng(42)
    for ep in range(1, args.epochs + 1):
        bridge.train()
        order = rng.permutation(n_tr)
        tr_ce_sum = 0.0
        tr_kl_sum = 0.0
        tr_skl_sum = 0.0
        tr_rel_sum = 0.0
        steps = 0

        for i in range(0, n_tr, args.batch_size):
            idx = order[i : min(n_tr, i + args.batch_size)]
            hb = torch.from_numpy(h_tr[idx]).to(device=device, dtype=torch.float32)
            eb = torch.from_numpy(e_tr[idx]).to(device=device, dtype=torch.float32) - mu_e_t
            yb = torch.from_numpy(y_tr[idx]).to(device=device, dtype=torch.long)

            with torch.no_grad():
                base_logits = maxwell_head_logits(hb, hc_fn, hc_scale, hc_base, out_norm, lm_head)
                base_logp = F.log_softmax(base_logits, dim=-1)
                base_p = base_logp.exp()

            h_out, rel_norm, _ = bridge(hb, eb, return_stats=True)
            logits = maxwell_head_logits(h_out, hc_fn, hc_scale, hc_base, out_norm, lm_head)
            logp = F.log_softmax(logits, dim=-1)

            ce = F.nll_loss(logp, yb)
            # KL(p_base || p_grafted) = sum p_base * (log p_base - log p_grafted)
            kl = (base_p * (base_logp - logp)).sum(dim=-1).mean()
            loss = ce + args.kl_beta * kl

            skl_val = 0.0
            if args.shuf_kl_beta > 0.0 and eb.shape[0] > 1:
                perm = torch.roll(torch.arange(eb.shape[0], device=device), shifts=1)
                h_shuf = bridge(hb, eb[perm])
                shuf_logits = maxwell_head_logits(h_shuf, hc_fn, hc_scale, hc_base, out_norm, lm_head)
                shuf_logp = F.log_softmax(shuf_logits, dim=-1)
                # Force shuffled Engram inputs to leave base distribution unchanged (prevents isotropic softening!)
                skl = (base_p * (base_logp - shuf_logp)).sum(dim=-1).mean()
                loss = loss + args.shuf_kl_beta * skl
                skl_val = skl.item()

            if args.delta_reg > 0.0:
                loss = loss + args.delta_reg * (rel_norm ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
            opt.step()
            sched.step()

            tr_ce_sum += ce.item()
            tr_kl_sum += kl.item()
            tr_skl_sum += skl_val
            tr_rel_sum += rel_norm.item()
            steps += 1

        _, _, _, r_nll, r_ppl, r_t1 = evaluate(bridge, h_va, e_va, y_va, mu_e_t, head_tensors, shuffled=False)
        _, _, _, s_nll, s_ppl, s_t1 = evaluate(bridge, h_va, e_va, y_va, mu_e_t, head_tensors, shuffled=True)
        print(
            f"[Epoch {ep:2d}] tr_CE={tr_ce_sum/steps:.4f} tr_KL={tr_kl_sum/steps:.4f} "
            f"shuf_KL={tr_skl_sum/steps:.4f} ||d||/||h||={tr_rel_sum/steps:.4f} | "
            f"val REAL: PPL={r_ppl:.3f} (vs base {r_ppl-b_ppl:+.3f}, top1={r_t1:.2f}%) | "
            f"SHUF: PPL={s_ppl:.3f} (shuf-real={s_ppl-r_ppl:+.3f})"
        )
        # Model selection: require genuine transfer (REAL must beat both BASE and SHUF when shuf_kl_beta > 0)
        sel_score = r_nll if (args.shuf_kl_beta <= 0.0 or r_nll < s_nll) else float("inf")
        if sel_score < best_val_score:
            best_val_score = sel_score
            best_val_nll = r_nll
            best_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}

    bridge.load_state_dict(best_state)
    if args.rkad_v3:
        manifest = {
            "schema": "rookery.engram.adapter.v3",
            "provider_arch": args.provider_arch,
            "provider_mode": "ngram_hash",
            "provider_fingerprint": args.provider_fingerprint,
            "host_arch": args.host_arch,
            "host_n_layers": args.host_n_layers,
            "host_d_embd": meta["d_host"],
            "host_hc_mult": meta["hc"],
            "calibration": f"level3_ce_kl_N{n_tr}_drop{args.dropout}",
        }
        write_rkad_v3(args.out, meta["target_layer"], args.src_layer, mu_e, bridge, manifest, args.delta_clip)
        print(f"Saved best Level-3 gated RKAD v3 -> {args.out} (best val PPL={math.exp(best_val_nll):.3f} vs base {b_ppl:.3f}) in {time.time()-t0:.1f}s")
    else:
        write_rkad_v2(args.out, meta["target_layer"], args.src_layer, mu_e, bridge)
        print(f"Saved best Level-3 gated RKAD v2 -> {args.out} (best val PPL={math.exp(best_val_nll):.3f} vs base {b_ppl:.3f}) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
