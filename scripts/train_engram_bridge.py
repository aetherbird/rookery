#!/usr/bin/env python3
"""
Level-3 Supervised Gated Bridge Trainer for Cross-Model Engram Grafting (RKCE v1 -> RKAD v2/v3).

Trains a lightweight low-rank gated Engram bridge (A_v, B_v, A_k, B_k, gate_w) directly against
next-token Cross-Entropy + Host-KL + Contrastive Shuffled-KL regularization through the host model's
EXACT frozen tail (hc_head + output_norm + lm_head), matching rookery's CUDA kernels formula-for-formula!

Optional `--engram-kv-gguf`: loads the Engram provider's own pre-trained Engram decoder
(e.g. DeepSeek-V4.1-Flash `blk.1.engram_kv.weight` / `blk.14.engram_kv.weight`), unit-RMS normalizes
the decoded semantic value/key subspaces, trains the semantic coordinate translator in the provider's
decoded space, and algebraically folds `A_v @ W_V_norm` and `A_k @ W_K_norm` directly into a standard
zero-overhead RKAD v3 adapter (no provider projection shard required at inference time!).
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
    """Pre-allocated low-RAM loader for RKCE v1 dumps (supports single- or multi-layer dumps)."""
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

        if n_tokens_hdr > 0:
            targets = np.empty((n_tokens_hdr,), dtype=np.int32)
            h_all = np.empty((n_tokens_hdr, hc, d_host), dtype=np.float16)
            e_all = np.empty((n_tokens_hdr, d_eng), dtype=np.float16)
            off = 0
            while off < n_tokens_hdr:
                raw = f.read(4)
                if len(raw) < 4:
                    break
                (nt,) = struct.unpack("<i", raw)
                if nt <= 0:
                    break
                targets[off : off + nt] = np.frombuffer(f.read(nt * 4), dtype=np.int32)
                h_all[off : off + nt] = np.frombuffer(
                    f.read(nt * hc * d_host * 2), dtype=np.float16
                ).reshape(nt, hc, d_host)
                for el in range(n_eng):
                    nbytes = nt * d_eng * 2
                    if el == src_layer:
                        e_all[off : off + nt] = np.frombuffer(f.read(nbytes), dtype=np.float16).reshape(nt, d_eng)
                    else:
                        f.seek(nbytes, 1)
                off += nt
            targets, h_all, e_all = targets[:off], h_all[:off], e_all[:off]
        else:
            targets_list, h_list, e_list = [], [], []
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
                    nbytes = nt * d_eng * 2
                    if el == src_layer:
                        e_sel = np.frombuffer(f.read(nbytes), dtype=np.float16).reshape(nt, d_eng).copy()
                    else:
                        f.seek(nbytes, 1)
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


def load_gguf_tensors(gguf_path: str, tensor_names: list[str]) -> dict[str, torch.Tensor]:
    """Load and dequantize named tensors from a GGUF file (F32, F16, BF16, Q8_0, Q4_K, Q6_K)."""
    def read_str(f):
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def skip_val(f, t):
        if t in (0, 1, 7):
            f.read(1)
        elif t in (2, 3):
            f.read(2)
        elif t in (4, 5, 6):
            f.read(4)
        elif t in (10, 11, 12):
            f.read(8)
        elif t == 8:
            read_str(f)
        elif t == 9:
            et, cnt = struct.unpack("<IQ", f.read(12))
            for _ in range(cnt):
                skip_val(f, et)

    with open(gguf_path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"Not a GGUF file: {gguf_path}")
        f.read(4)
        nt, nkv = struct.unpack("<QQ", f.read(16))
        align = 32
        for _ in range(nkv):
            k = read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            if k == "general.alignment" and t == 4:
                (align,) = struct.unpack("<I", f.read(4))
            else:
                skip_val(f, t)
        tensors = {}
        for _ in range(nt):
            name = read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            tt, off = struct.unpack("<IQ", f.read(12))
            tensors[name] = (dims, tt, off)
        pos = f.tell()
        data_off = ((pos + align - 1) // align) * align

        out_map = {}
        for name in tensor_names:
            if name not in tensors:
                raise KeyError(f"Tensor '{name}' not found in {gguf_path}")
            dims, tt, off = tensors[name]
            ne = 1
            for d in dims:
                ne *= d
            f.seek(data_off + off)
            if tt == 0:  # F32
                arr = np.frombuffer(f.read(ne * 4), dtype=np.float32).copy()
            elif tt == 1:  # F16
                arr = np.frombuffer(f.read(ne * 2), dtype=np.float16).astype(np.float32)
            elif tt == 30:  # BF16
                u16 = np.frombuffer(f.read(ne * 2), dtype=np.uint16).astype(np.uint32) << 16
                arr = u16.view(np.float32).copy()
            elif tt == 8:  # Q8_0
                nblk = ne // 32
                raw = np.frombuffer(f.read(nblk * 34), dtype=np.uint8).reshape(nblk, 34)
                d = raw[:, :2].copy().view(np.float16).astype(np.float32)
                qs = raw[:, 2:].view(np.int8).astype(np.float32)
                arr = (d * qs).reshape(ne)
            elif tt == 12:  # Q4_K
                nblk = ne // 256
                raw = np.frombuffer(f.read(nblk * 144), dtype=np.uint8).reshape(nblk, 144)
                d = raw[:, 0:2].copy().view(np.float16).astype(np.float32)
                dmin = raw[:, 2:4].copy().view(np.float16).astype(np.float32)
                sc_b = raw[:, 4:16]
                qs = raw[:, 16:144]
                sc = np.empty((nblk, 8), dtype=np.float32)
                mn = np.empty((nblk, 8), dtype=np.float32)
                sc[:, 0:4] = sc_b[:, 0:4] & 63
                mn[:, 0:4] = sc_b[:, 4:8] & 63
                sc[:, 4:8] = (sc_b[:, 8:12] & 0x0F) | ((sc_b[:, 0:4] >> 6) << 4)
                mn[:, 4:8] = (sc_b[:, 8:12] >> 4) | ((sc_b[:, 4:8] >> 6) << 4)
                out = np.empty((nblk, 256), dtype=np.float32)
                for j in range(4):
                    q = qs[:, 32 * j : 32 * (j + 1)]
                    out[:, 64 * j : 64 * j + 32] = (d * sc[:, 2 * j : 2 * j + 1]) * (q & 0x0F).astype(np.float32) - (dmin * mn[:, 2 * j : 2 * j + 1])
                    out[:, 64 * j + 32 : 64 * j + 64] = (d * sc[:, 2 * j + 1 : 2 * j + 2]) * (q >> 4).astype(np.float32) - (dmin * mn[:, 2 * j + 1 : 2 * j + 2])
                arr = out.reshape(ne)
            elif tt == 14:  # Q6_K
                nblk = ne // 256
                raw = np.frombuffer(f.read(nblk * 210), dtype=np.uint8).reshape(nblk, 210)
                ql = raw[:, 0:128]
                qh = raw[:, 128:192]
                sc = raw[:, 192:208].copy().view(np.int8).astype(np.float32)
                d = raw[:, 208:210].copy().view(np.float16).astype(np.float32)
                out = np.empty((nblk, 256), dtype=np.float32)
                for n in range(2):
                    ql_n = ql[:, 64 * n : 64 * (n + 1)]
                    qh_n = qh[:, 32 * n : 32 * (n + 1)]
                    sc_n = sc[:, 8 * n : 8 * (n + 1)]
                    q1 = ((ql_n[:, 0:32] & 0x0F) | (((qh_n >> 0) & 3) << 4)).astype(np.int32) - 32
                    q2 = ((ql_n[:, 32:64] & 0x0F) | (((qh_n >> 2) & 3) << 4)).astype(np.int32) - 32
                    q3 = ((ql_n[:, 0:32] >> 4) | (((qh_n >> 4) & 3) << 4)).astype(np.int32) - 32
                    q4 = ((ql_n[:, 32:64] >> 4) | (((qh_n >> 6) & 3) << 4)).astype(np.int32) - 32
                    out[:, 128 * n + 0 : 128 * n + 32] = d * np.repeat(sc_n[:, 0:2], 16, axis=1) * q1
                    out[:, 128 * n + 32 : 128 * n + 64] = d * np.repeat(sc_n[:, 2:4], 16, axis=1) * q2
                    out[:, 128 * n + 64 : 128 * n + 96] = d * np.repeat(sc_n[:, 4:6], 16, axis=1) * q3
                    out[:, 128 * n + 96 : 128 * n + 128] = d * np.repeat(sc_n[:, 6:8], 16, axis=1) * q4
                arr = out.reshape(ne)
            else:
                raise ValueError(f"Unsupported GGUF tensor type {tt} for {name}")
            shape = tuple(reversed(dims))
            out_map[name] = torch.from_numpy(arr.reshape(shape))
    return out_map


def load_provider_engram_kv(
    gguf_path: str,
    src_layer: int,
    tensor_name: str = "",
    hc_provider: int = 4,
    device: torch.device = torch.device("cuda:0"),
    max_cols: int = 0,
):
    """
    Load provider's native pre-trained Engram projection (e.g. DeepSeek-V4.1-Flash `blk.1.engram_kv.weight`).
    Layout in DeepSeek-V4.1-Flash: [ (hc + 1) * d_prov, d_eng ], where:
      - rows [0 : hc * d_prov] = K_0..K_{hc-1} (averaged across streams -> W_K_prov [d_prov, d_eng])
      - rows [hc * d_prov : (hc + 1) * d_prov] = V (W_V_prov [d_prov, d_eng])
    Optional `max_cols` slices leading columns [:, :max_cols] (e.g. 4096 for 2+3-gram sub-tables like flapling-8B).
    """
    if not tensor_name:
        blk = 1 if src_layer == 0 else 14
        tensor_name = f"blk.{blk}.engram_kv.weight"
    kv = load_gguf_tensors(gguf_path, [tensor_name])[tensor_name]  # [(hc+1)*d_prov, d_eng]
    n_rows, d_eng = kv.shape
    if n_rows % (hc_provider + 1) != 0:
        raise ValueError(f"Unexpected {tensor_name} shape {tuple(kv.shape)} for hc_provider={hc_provider}")
    d_prov = n_rows // (hc_provider + 1)
    if max_cols > 0 and max_cols < d_eng:
        kv = kv[:, :max_cols]
        d_eng = max_cols
    W_K = kv[: hc_provider * d_prov].reshape(hc_provider, d_prov, d_eng).mean(dim=0).to(device=device, dtype=torch.float32)
    W_V = kv[hc_provider * d_prov :].contiguous().to(device=device, dtype=torch.float32)
    return W_V, W_K, tensor_name


def load_maxwell_head(safetensors_dir: str, device: torch.device):
    idx_path = os.path.join(safetensors_dir, "model.safetensors.index.json")
    wmap = json.load(open(idx_path))["weight_map"]

    def get_t(key: str) -> torch.Tensor:
        shard = os.path.join(safetensors_dir, wmap[key])
        with safe_open(shard, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    hc_fn = get_t("model.hc_head.hc_head_fn").to(device=device, dtype=torch.float32)
    hc_scale = get_t("model.hc_head.hc_head_scale").to(device=device, dtype=torch.float32)
    hc_base = get_t("model.hc_head.hc_head_base").to(device=device, dtype=torch.float32)
    out_norm = get_t("model.norm.weight").to(device=device, dtype=torch.float32)
    lm_head = get_t("lm_head.weight").to(device=device, dtype=torch.float16)

    if hc_fn.shape[0] != 4 and hc_fn.shape[1] == 4:
        hc_fn = hc_fn.t().contiguous()
    hc_scale = hc_scale.reshape(-1)[0]
    hc_base = hc_base.reshape(4)
    return hc_fn, hc_scale, hc_base, out_norm, lm_head


def load_gguf_tail_head(gguf_path: str, device: torch.device):
    """Load frozen tail from GGUF (single or multi-shard) for:
      - Maxwell hc=4 models (output_hc_fn/scale/base + output_norm + output)
      - Standard hc=1 models (output_norm + output)
      - Post-output-hc-mix models with no separate output_norm (e.g. qwen4exp: output only)
    """
    import glob, re
    shards = [gguf_path]
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", gguf_path)
    if m:
        cand = sorted(glob.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf"))
        if cand:
            shards = cand

    def find_tensor(name: str):
        for sp in shards:
            try:
                return load_gguf_tensors(sp, [name])[name]
            except KeyError:
                continue
        raise KeyError(f"Tensor '{name}' not found across {len(shards)} shard(s) of {gguf_path}")

    try:
        lm_head = find_tensor("output.weight").to(device=device, dtype=torch.float16)
    except KeyError:
        lm_head = find_tensor("token_embd.weight").to(device=device, dtype=torch.float16)
    try:
        out_norm = find_tensor("output_norm.weight").to(device=device, dtype=torch.float32)
    except KeyError:
        out_norm = None

    try:
        hc_fn = find_tensor("output_hc_fn.weight").to(device=device, dtype=torch.float32)
        if hc_fn.shape[0] != 4 and hc_fn.shape[1] == 4:
            hc_fn = hc_fn.t().contiguous()
        hc_scale = find_tensor("output_hc_scale.weight").to(device=device, dtype=torch.float32).reshape(-1)[0]
        hc_base = find_tensor("output_hc_base.weight").to(device=device, dtype=torch.float32).reshape(4)
        return hc_fn, hc_scale, hc_base, out_norm, lm_head
    except KeyError:
        pass

    # Check if Qwen4Exp output_hc_mix tensors exist (output_hc_norm + output_hc_down + output_hc_up, no output_norm)
    try:
        q_norm = find_tensor("output_hc_norm.weight").to(device=device, dtype=torch.float32)
        q_down = find_tensor("output_hc_down.weight").to(device=device, dtype=torch.float32)
        q_up = find_tensor("output_hc_up.weight").to(device=device, dtype=torch.float32)
        return ("qwen4exp", q_norm, q_down, q_up), None, None, None, lm_head
    except KeyError:
        return None, None, None, out_norm, lm_head


def maxwell_head_logits(
    h_hc: torch.Tensor,
    hc_fn,
    hc_scale,
    hc_base,
    out_norm: torch.Tensor,
    lm_head: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    logit_softcap: float = 0.0,
) -> torch.Tensor:
    B, hc, D = h_hc.shape
    if isinstance(hc_fn, tuple) and hc_fn[0] == "qwen4exp":
        _, q_norm, q_down, q_up = hc_fn
        inv = torch.rsqrt(h_hc.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
        w = q_norm.reshape(1, hc, D) if q_norm.numel() == hc * D else q_norm.reshape(1, 1, D)
        xn = h_hc * inv * w
        lo = F.silu(F.linear(xn.reshape(B, hc * D), q_down) * (1.0 / float(hc)))
        gate = F.linear(lo, q_up).reshape(B, hc, D)
        mixed = (xn * torch.sigmoid(gate)).mean(dim=1)
        x_norm = mixed
    else:
        if hc_fn is not None:
            flat = h_hc.reshape(B, hc * D)
            inv_flat = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
            mixes = F.linear(flat * inv_flat, hc_fn)
            pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
            mixed = (pre.unsqueeze(-1) * h_hc).sum(dim=1)
        else:
            mixed = h_hc.mean(dim=1)
        if out_norm is not None:
            inv_rms = torch.rsqrt(mixed.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
            x_norm = mixed * inv_rms * out_norm
        else:
            x_norm = mixed
    logits = F.linear(x_norm.to(lm_head.dtype), lm_head).float()
    if logit_softcap > 0.0:
        logits = logit_softcap * torch.tanh(logits / logit_softcap)
    return logits


class RookeryGatedBridge(nn.Module):
    """Exact PyTorch replica of rookery's engram_gated_inject_kernel (RKAD v2/v3 has_gate=1)."""

    def __init__(
        self,
        d_in: int,
        d_host: int,
        hc: int,
        rank: int,
        init_scale: float = 0.05,
        dropout: float = 0.0,
        gate_bias: float = 0.0,
    ):
        super().__init__()
        self.d_eng = d_in
        self.d_host = d_host
        self.hc = hc
        self.rank = rank
        self.fixed_scale = init_scale
        self.gate_bias = float(gate_bias)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.A_v = nn.Parameter(torch.randn(rank, d_in) * (1.0 / math.sqrt(d_in)))
        self.B_v = nn.Parameter(torch.zeros(d_host, rank))  # exact zero-init -> starts at baseline!
        self.A_k = nn.Parameter(torch.randn(rank, d_in) * (1.0 / math.sqrt(d_in)))
        self.B_k = nn.Parameter(torch.randn(d_host, rank) * (1.0 / math.sqrt(rank)))
        self.gate_w = nn.Parameter(torch.ones(hc, d_host))

    def forward(
        self,
        h_hc: torch.Tensor,
        e_v: torch.Tensor,
        e_k: torch.Tensor = None,
        rms_eps: float = 1e-6,
        return_stats: bool = False,
    ):
        if e_k is None:
            e_k = e_v
        v = F.linear(self.drop(F.linear(self.drop(e_v), self.A_v)), self.B_v)
        k = F.linear(self.drop(F.linear(self.drop(e_k), self.A_k)), self.B_k)

        k_norm = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
        h_norm = h_hc * torch.rsqrt(h_hc.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
        dot = (h_norm * k_norm.unsqueeze(1) * self.gate_w.unsqueeze(0)).sum(dim=-1)
        scaled = dot / math.sqrt(float(self.d_host))
        mag = torch.sqrt(torch.clamp(scaled.abs(), min=1e-6))
        sgn = torch.where(scaled >= 0.0, torch.ones_like(scaled), -torch.ones_like(scaled))
        g = torch.sigmoid(sgn * mag + self.gate_bias) * self.fixed_scale
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
    W_V_fold: torch.Tensor = None,
    W_K_fold: torch.Tensor = None,
):
    A_v = (bridge.A_v @ W_V_fold) if W_V_fold is not None else bridge.A_v
    A_k = (bridge.A_k @ W_K_fold) if W_K_fold is not None else bridge.A_k
    d_eng_out = int(A_v.shape[1])

    m_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    has_mu = mu_e is not None
    hook_flags = (1 if has_mu else 0) | 2 | (4 if delta_clip > 0.0 else 0)
    with open(out_path, "wb") as f:
        f.write(b"RKAD")
        f.write(struct.pack("<6I", 3, 1, d_eng_out, bridge.d_host, bridge.rank, 0))
        f.write(struct.pack("<I", len(m_bytes)))
        f.write(m_bytes)
        f.write(
            struct.pack(
                "<2if2If",
                target_layer,
                src_layer,
                float(bridge.fixed_scale),
                bridge.rank,
                hook_flags,
                float(delta_clip),
            )
        )
        if has_mu:
            f.write(mu_e.astype(np.float32).tobytes())
        f.write(A_v.detach().cpu().float().numpy().tobytes())
        f.write(bridge.B_v.detach().cpu().float().numpy().tobytes())
        f.write(A_k.detach().cpu().float().numpy().tobytes())
        f.write(bridge.B_k.detach().cpu().float().numpy().tobytes())
        f.write(bridge.gate_w.detach().cpu().float().numpy().tobytes())


def write_rkad_v2(
    out_path: str,
    target_layer: int,
    src_layer: int,
    mu_e: np.ndarray,
    bridge: RookeryGatedBridge,
    W_V_fold: torch.Tensor = None,
    W_K_fold: torch.Tensor = None,
):
    A_v = (bridge.A_v @ W_V_fold) if W_V_fold is not None else bridge.A_v
    A_k = (bridge.A_k @ W_K_fold) if W_K_fold is not None else bridge.A_k
    d_eng_out = int(A_v.shape[1])
    with open(out_path, "wb") as f:
        f.write(b"RKAD")
        f.write(struct.pack("<6I", 2, 1, d_eng_out, bridge.d_host, bridge.rank, 1))
        f.write(struct.pack("<2if", target_layer, src_layer, float(bridge.fixed_scale)))
        f.write(mu_e.astype(np.float32).tobytes())
        f.write(A_v.detach().cpu().float().numpy().tobytes())
        f.write(bridge.B_v.detach().cpu().float().numpy().tobytes())
        f.write(A_k.detach().cpu().float().numpy().tobytes())
        f.write(bridge.B_k.detach().cpu().float().numpy().tobytes())
        f.write(bridge.gate_w.detach().cpu().float().numpy().tobytes())


@torch.no_grad()
def evaluate(
    bridge,
    h_val,
    e_val,
    y_val,
    mu_e_t,
    head_tensors,
    W_V_norm=None,
    W_K_norm=None,
    batch_size=256,
    shuffled=False,
    logit_softcap=0.0,
):
    bridge.eval()
    hc_fn, hc_scale, hc_base, out_norm, lm_head = head_tensors
    device = lm_head.device
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

        base_logits = maxwell_head_logits(hb, hc_fn, hc_scale, hc_base, out_norm, lm_head, logit_softcap=logit_softcap)
        tot_base_nll += F.cross_entropy(base_logits, yb, reduction="sum").item()
        base_hit1 += (base_logits.argmax(dim=-1) == yb).sum().item()

        ev = F.linear(eb, W_V_norm) if W_V_norm is not None else eb
        ek = F.linear(eb, W_K_norm) if W_K_norm is not None else ev
        h_out = bridge(hb, ev, ek)
        logits = maxwell_head_logits(h_out, hc_fn, hc_scale, hc_base, out_norm, lm_head, logit_softcap=logit_softcap)
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
    ap.add_argument("--safetensors-dir", default="", help="Directory with Maxwell safetensors + index.json")
    ap.add_argument("--gguf", default="", help="Host GGUF path for frozen tail loading (hc=1 or Maxwell hc=4)")
    ap.add_argument("--engram-kv-gguf", default="", help="Optional provider GGUF shard with blk.<L>.engram_kv.weight for pre-decoded semantic folding")
    ap.add_argument("--engram-kv-tensor", default="", help="Explicit tensor name in --engram-kv-gguf (default: auto blk.1/blk.14.engram_kv.weight)")
    ap.add_argument("--engram-kv-hc", type=int, default=4, help="Provider hc_mult for splitting engram_kv into [K_0..K_{hc-1}, V]")
    ap.add_argument("--pca-dim", type=int, default=0, help="Optional unsupervised whitened PCA projection dim (folded algebraically into RKAD v3)")
    ap.add_argument("--no-center", action="store_true", help="Do not subtract corpus mean mu_e (keeps e=0 -> delta=0 exact, eliminates phantom -mu_e bias on unseen/shuffled rows)")
    ap.add_argument("--out", required=True, help="Output RKAD v2/v3 path")
    ap.add_argument("--src-layer", type=int, default=0, help="Engram source layer index in RKCE (0 or 1)")
    ap.add_argument("--export-src-layer", type=int, default=-1, help="Engram source layer index written to RKAD (default: same as --src-layer)")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--wd", type=float, default=2e-2)
    ap.add_argument("--kl-beta", type=float, default=0.5)
    ap.add_argument("--shuf-kl-beta", type=float, default=1.5)
    ap.add_argument("--delta-reg", type=float, default=1.0)
    ap.add_argument("--calibrate-base-temp", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-val", type=int, default=40000)
    ap.add_argument("--rkad-v3", action="store_true")
    ap.add_argument("--provider-arch", default="deepseek41")
    ap.add_argument("--provider-mode", default="ngram_hash")
    ap.add_argument("--provider-fingerprint", default="")
    ap.add_argument("--host-arch", default="maxwell-moe")
    ap.add_argument("--host-n-layers", type=int, default=45)
    ap.add_argument("--delta-clip", type=float, default=0.15)
    ap.add_argument("--gate-bias", type=float, default=0.0, help="Additive gate logit bias (negative = closed by default)")
    ap.add_argument("--surprisal-alpha", type=float, default=0.0, help="Dual surprisal weighting strength (0 = uniform)")
    ap.add_argument("--surprisal-thresh", type=float, default=1.5, help="Base NLL threshold separating confident vs uncertain tokens")
    ap.add_argument("--logit-softcap", type=float, default=-1.0, help="Optional final logit softcap (default: auto 30.0 for inkling2_loop, 0 otherwise)")
    args = ap.parse_args()
    logit_softcap = args.logit_softcap if args.logit_softcap >= 0.0 else (30.0 if args.host_arch in ("inkling2_loop", "inkling2", "gemma4") else 0.0)

    t0 = time.time()
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    h_all, e_all, y_all, meta = load_rkce(args.rkce, args.src_layer)
    N = h_all.shape[0]
    n_val = min(args.max_val, max(256, int(N * args.val_frac)))
    # Interleaved 1024-token block split so train and val share identical domain mixture (no tail-domain shift!)
    blk_sz = 1024
    n_blks = (N + blk_sz - 1) // blk_sz
    n_val_blks = max(1, min(n_blks // 3, int(round(n_val / blk_sz))))
    val_step = max(2, n_blks // n_val_blks)
    val_mask = np.zeros(N, dtype=bool)
    for b in range(val_step - 1, n_blks, val_step):
        val_mask[b * blk_sz : min(N, (b + 1) * blk_sz)] = True
    if val_mask.sum() > args.max_val:
        idx_on = np.where(val_mask)[0]
        val_mask[idx_on[args.max_val :]] = False
    tr_mask = ~val_mask
    n_tr = int(tr_mask.sum())
    n_val = int(val_mask.sum())
    export_src = args.export_src_layer if args.export_src_layer >= 0 else args.src_layer
    print(
        f"Loaded {args.rkce}: N={N:,} valid tokens (train={n_tr:,}, val={n_val:,} interleaved), "
        f"src_layer={args.src_layer} (export={export_src}), target_layer={meta['target_layer']}, "
        f"d_eng={meta['d_eng']}, hc={meta['hc']}, d_host={meta['d_host']}",
        flush=True,
    )

    h_tr, e_tr, y_tr = h_all[tr_mask], e_all[tr_mask], y_all[tr_mask]
    h_va, e_va, y_va = h_all[val_mask], e_all[val_mask], y_all[val_mask]

    # Optional training-set mean centering vector mu_E (use --no-center for exact e=0 -> delta=0 invariance)
    if args.no_center:
        mu_e = np.zeros(meta["d_eng"], dtype=np.float32)
    else:
        step_mu = max(1, n_tr // 200000)
        mu_e = e_tr[::step_mu].astype(np.float64).mean(axis=0).astype(np.float32)
    mu_e_t = torch.from_numpy(mu_e).to(device=device, dtype=torch.float32)

    # Optional provider pre-decoder W_V / W_K with exact unit-RMS normalization!
    W_V_norm, W_K_norm = None, None
    d_bridge_in = meta["d_eng"]
    kv_tag = ""
    if args.engram_kv_gguf:
        W_V_raw, W_K_raw, kv_name = load_provider_engram_kv(
            args.engram_kv_gguf, export_src, args.engram_kv_tensor, args.engram_kv_hc, device
        )
        # Measure exact RMS of projected training vectors so bridge inputs have exact unit RMS 1.0!
        with torch.no_grad():
            sub_e = torch.from_numpy(e_tr[:: max(1, n_tr // 16384)]).to(device=device, dtype=torch.float32) - mu_e_t
            v_ss = F.linear(sub_e, W_V_raw).pow(2).mean().item()
            k_ss = F.linear(sub_e, W_K_raw).pow(2).mean().item()
            rms_v = math.sqrt(max(v_ss, 1e-12))
            rms_k = math.sqrt(max(k_ss, 1e-12))
            W_V_norm = W_V_raw / rms_v
            W_K_norm = W_K_raw / rms_k
        d_bridge_in = int(W_V_norm.shape[0])
        kv_tag = f"_kvfold"
        print(
            f"[ProviderKV] Loaded {kv_name} ({d_bridge_in}x{meta['d_eng']}): "
            f"pre-decoded rms_v={rms_v:.4f}, rms_k={rms_k:.4f} -> unit-RMS normalized for folding",
            flush=True,
        )
    elif args.pca_dim > 0 and args.pca_dim < meta["d_eng"]:
        with torch.no_grad():
            step_p = max(1, n_tr // 32768)
            X_sub = torch.from_numpy(e_tr[::step_p]).to(device=device, dtype=torch.float32) - mu_e_t
            # Exact SVD / eigendecomposition of covariance in float32
            cov = (X_sub.T @ X_sub) / float(X_sub.shape[0])
            evals, evecs = torch.linalg.eigh(cov)
            idx_desc = torch.argsort(evals, descending=True)
            evals = evals[idx_desc]
            evecs = evecs[:, idx_desc]
            k = min(args.pca_dim, int((evals > 1e-7).sum().item()))
            var_exp = evals[:k].sum().item() / evals.clamp(min=0.0).sum().item()
            inv_std = torch.rsqrt(evals[:k].clamp(min=1e-5))
            # Whitened PCA basis W_pca [k, d_eng] -> isotropic unit-RMS coordinates, folded into RKAD v3!
            W_pca = (evecs[:, :k] * inv_std.unsqueeze(0)).T.contiguous()
            W_V_norm = W_pca
            W_K_norm = W_pca
            d_bridge_in = k
            kv_tag = f"_pca{k}"
            print(
                f"[DirectPCA] Fitted native whitened PCA ({k}x{meta['d_eng']}): "
                f"explained_var={100.0*var_exp:.1f}% -> algebraic fold into RKAD v3",
                flush=True,
            )

    if args.safetensors_dir:
        head_tensors = load_maxwell_head(args.safetensors_dir, device)
    elif args.gguf:
        head_tensors = load_gguf_tail_head(args.gguf, device)
    else:
        raise ValueError("Must specify either --safetensors-dir or --gguf")
    hc_fn, hc_scale, hc_base, out_norm, lm_head = head_tensors

    bridge = RookeryGatedBridge(d_bridge_in, meta["d_host"], meta["hc"], args.rank, args.scale, args.dropout, gate_bias=args.gate_bias).to(device)
    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * ((n_tr + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=args.lr * 0.05)

    b_nll, b_ppl, b_t1, r_nll, r_ppl, r_t1 = evaluate(
        bridge, h_va, e_va, y_va, mu_e_t, head_tensors, W_V_norm, W_K_norm, logit_softcap=logit_softcap
    )
    print(f"[Epoch 0] val BASE: NLL={b_nll:.5f} PPL={b_ppl:.3f} top1={b_t1:.2f}% | init PPL={r_ppl:.3f}", flush=True)

    train_tau = 1.0
    if args.calibrate_base_temp:
        sub_n = min(n_tr, 16384)
        taus = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50]
        nll_sums = [0.0 for _ in taus]
        with torch.no_grad():
            for i in range(0, sub_n, 256):
                j = min(sub_n, i + 256)
                hb = torch.from_numpy(h_tr[i:j]).to(device=device, dtype=torch.float32)
                yb = torch.from_numpy(y_tr[i:j]).to(device=device, dtype=torch.long)
                bl = maxwell_head_logits(hb, hc_fn, hc_scale, hc_base, out_norm, lm_head, logit_softcap=logit_softcap)
                for ti, tval in enumerate(taus):
                    nll_sums[ti] += F.cross_entropy(bl / tval, yb, reduction="sum").item()
        best_ti = int(np.argmin(nll_sums))
        train_tau = taus[best_ti]
        print(
            f"[TempCalib] Optimal base temperature tau*={train_tau:.2f} "
            f"(base NLL@1.0={nll_sums[taus.index(1.00)]/sub_n:.4f} -> @{train_tau:.2f}={nll_sums[best_ti]/sub_n:.4f})",
            flush=True,
        )

    best_clean_score = float("inf")
    best_clean_state = None
    best_clean_nll = b_nll
    best_any_score = float("inf")
    best_any_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}
    best_any_nll = b_nll

    rng = np.random.default_rng(42)
    for ep in range(1, args.epochs + 1):
        bridge.train()
        order = rng.permutation(n_tr)
        tr_ce_sum = tr_kl_sum = tr_skl_sum = tr_rel_sum = 0.0
        steps = 0

        for i in range(0, n_tr, args.batch_size):
            idx = order[i : min(n_tr, i + args.batch_size)]
            hb = torch.from_numpy(h_tr[idx]).to(device=device, dtype=torch.float32)
            eb = torch.from_numpy(e_tr[idx]).to(device=device, dtype=torch.float32) - mu_e_t
            yb = torch.from_numpy(y_tr[idx]).to(device=device, dtype=torch.long)

            with torch.no_grad():
                base_logits = maxwell_head_logits(hb, hc_fn, hc_scale, hc_base, out_norm, lm_head, logit_softcap=logit_softcap)
                base_logp = F.log_softmax(base_logits / train_tau, dim=-1)
                base_p = base_logp.exp()
                if args.surprisal_alpha > 0.0:
                    base_nll = F.cross_entropy(base_logits, yb, reduction="none")
                    u_tok = torch.sigmoid((base_nll - args.surprisal_thresh) / 0.5)
                    ce_w = 1.0 + args.surprisal_alpha * u_tok
                    ce_w = ce_w / ce_w.mean().clamp(min=1e-6)
                    kl_w = 1.0 + args.surprisal_alpha * (1.0 - u_tok)
                    kl_w = kl_w / kl_w.mean().clamp(min=1e-6)
                else:
                    ce_w = kl_w = None
                ev = F.linear(eb, W_V_norm) if W_V_norm is not None else eb
                ek = F.linear(eb, W_K_norm) if W_K_norm is not None else ev

            h_out, rel_norm, mean_g = bridge(hb, ev, ek, return_stats=True)
            logits = maxwell_head_logits(h_out, hc_fn, hc_scale, hc_base, out_norm, lm_head, logit_softcap=logit_softcap)
            logp = F.log_softmax(logits / train_tau, dim=-1)

            if ce_w is not None:
                ce = (F.nll_loss(logp, yb, reduction="none") * ce_w).mean()
                kl = ((base_p * (base_logp - logp)).sum(dim=-1) * kl_w).mean()
            else:
                ce = F.nll_loss(logp, yb)
                kl = (base_p * (base_logp - logp)).sum(dim=-1).mean()
            loss = ce + args.kl_beta * kl

            skl_val = 0.0
            if args.shuf_kl_beta > 0.0 and eb.shape[0] > 1:
                perm = torch.roll(torch.arange(eb.shape[0], device=device), shifts=1)
                h_shuf = bridge(hb, ev[perm], ek[perm])
                shuf_logits = maxwell_head_logits(h_shuf, hc_fn, hc_scale, hc_base, out_norm, lm_head, logit_softcap=logit_softcap)
                shuf_logp = F.log_softmax(shuf_logits / train_tau, dim=-1)
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

        _, _, _, r_nll, r_ppl, r_t1 = evaluate(
            bridge, h_va, e_va, y_va, mu_e_t, head_tensors, W_V_norm, W_K_norm, shuffled=False, logit_softcap=logit_softcap
        )
        _, _, _, s_nll, s_ppl, s_t1 = evaluate(
            bridge, h_va, e_va, y_va, mu_e_t, head_tensors, W_V_norm, W_K_norm, shuffled=True, logit_softcap=logit_softcap
        )
        print(
            f"[Epoch {ep:2d}] tr_CE={tr_ce_sum/steps:.4f} tr_KL={tr_kl_sum/steps:.4f} "
            f"shuf_KL={tr_skl_sum/steps:.4f} ||d||/||h||={tr_rel_sum/steps:.4f} | "
            f"val REAL: PPL={r_ppl:.3f} (vs base {r_ppl-b_ppl:+.3f}, top1={r_t1:.2f}%) | "
            f"SHUF: PPL={s_ppl:.3f} (shuf-real={s_ppl-r_ppl:+.3f})",
            flush=True,
        )
        if args.shuf_kl_beta <= 0.0 or r_nll < s_nll:
            if r_nll < best_any_score:
                best_any_score = r_nll
                best_any_nll = r_nll
                best_any_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}
            # Prefer strictly non-softening checkpoints (SHUF NLL >= BASE NLL - 0.002)
            if s_nll >= b_nll - 0.002 and r_nll < best_clean_score:
                best_clean_score = r_nll
                best_clean_nll = r_nll
                best_clean_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}

    if best_clean_state is not None:
        bridge.load_state_dict(best_clean_state)
        best_val_nll = best_clean_nll
    else:
        bridge.load_state_dict(best_any_state)
        best_val_nll = best_any_nll

    if args.rkad_v3:
        manifest = {
            "schema": "rookery.engram.adapter.v3",
            "provider_arch": args.provider_arch,
            "provider_mode": args.provider_mode,
            "provider_fingerprint": args.provider_fingerprint,
            "host_arch": args.host_arch,
            "host_n_layers": args.host_n_layers,
            "host_d_embd": meta["d_host"],
            "host_hc_mult": meta["hc"],
            "default_gate_bias": float(args.gate_bias),
            "calibration": f"level3_ce_kl_N{n_tr}_drop{args.dropout}_gb{args.gate_bias:g}_sa{args.surprisal_alpha:g}{kv_tag}{'_nc' if args.no_center else ''}",
        }
        write_rkad_v3(
            args.out,
            meta["target_layer"],
            export_src,
            None if args.no_center else mu_e,
            bridge,
            manifest,
            args.delta_clip,
            W_V_norm,
            W_K_norm,
        )
        print(
            f"Saved best Level-3 gated RKAD v3 -> {args.out} "
            f"(best val PPL={math.exp(best_val_nll):.3f} vs base {b_ppl:.3f}) in {time.time()-t0:.1f}s",
            flush=True,
        )
    else:
        write_rkad_v2(args.out, meta["target_layer"], export_src, mu_e, bridge, W_V_norm, W_K_norm)
        print(
            f"Saved best Level-3 gated RKAD v2 -> {args.out} "
            f"(best val PPL={math.exp(best_val_nll):.3f} vs base {b_ppl:.3f}) in {time.time()-t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
