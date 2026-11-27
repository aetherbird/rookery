#!/usr/bin/env python3
"""
End-to-End Frozen-Host Gated Mid-Layer Bridge Trainer (RKCE v1 -> RKAD v3).

Instead of injecting Engram vectors only at the final residual stream (where only
output_norm + lm_head see the perturbation), this trainer injects a RookeryGatedBridge
at an earlier host layer L_k (e.g. Ling-3.0-tiny L22 -> frozen blk.23 MLA Full Attention
+ 128-expert MoE FFN + Shared Expert + output_norm + lm_head) and backpropagates
Surprisal-Weighted CE + Host-KL + Contrastive Shuffled-KL + Delta-Reg end-to-end
through the frozen downstream host Transformer layer(s) over contiguous sequence windows!

Exports directly to standard zero-overhead RKAD v3 (target_host_layer = L_k).
"""
import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_engram_bridge import RookeryGatedBridge, load_provider_engram_kv, write_rkad_v3


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv * w.float()).to(x.dtype)


def dequant_gguf_tensor(raw_bytes: bytes, dims: list[int], tt: int) -> torch.Tensor:
    """Dequantize GGUF tensor (F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q6_K) to float32."""
    ne = 1
    for d in dims:
        ne *= d
    if tt == 0:  # F32
        arr = np.frombuffer(raw_bytes, dtype=np.float32).copy()
    elif tt == 1:  # F16
        arr = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)
    elif tt == 30:  # BF16
        u16 = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.uint32) << 16
        arr = u16.view(np.float32).copy()
    elif tt == 6:  # Q5_0: 32 elems/block, 22 bytes/block (f16 d, u32 qh, u8 qs[16])
        nblk = ne // 32
        raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(nblk, 22)
        d = raw[:, 0:2].copy().view(np.float16).astype(np.float32)
        qh = raw[:, 2:6].copy().view(np.uint32)  # [nblk, 1]
        qs = raw[:, 6:22]  # [nblk, 16]
        shifts_lo = np.arange(16, dtype=np.uint32)[None, :]
        shifts_hi = (np.arange(16, dtype=np.uint32) + 16)[None, :]
        xh_0 = ((qh >> shifts_lo) & 1) << 4
        xh_1 = ((qh >> shifts_hi) & 1) << 4
        x0 = ((qs & 0x0F).astype(np.uint32) | xh_0).astype(np.int32) - 16
        x1 = ((qs >> 4).astype(np.uint32) | xh_1).astype(np.int32) - 16
        out = np.empty((nblk, 32), dtype=np.float32)
        out[:, :16] = d * x0.astype(np.float32)
        out[:, 16:] = d * x1.astype(np.float32)
        arr = out.reshape(ne)
    elif tt == 8:  # Q8_0
        nblk = ne // 32
        raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(nblk, 34)
        d = raw[:, :2].copy().view(np.float16).astype(np.float32)
        qs = raw[:, 2:].view(np.int8).astype(np.float32)
        arr = (d * qs).reshape(ne)
    elif tt == 12:  # Q4_K
        nblk = ne // 256
        raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(nblk, 144)
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
            out[:, 64 * j : 64 * j + 32] = (
                (d * sc[:, 2 * j : 2 * j + 1]) * (q & 0x0F).astype(np.float32)
                - (dmin * mn[:, 2 * j : 2 * j + 1])
            )
            out[:, 64 * j + 32 : 64 * j + 64] = (
                (d * sc[:, 2 * j + 1 : 2 * j + 2]) * (q >> 4).astype(np.float32)
                - (dmin * mn[:, 2 * j + 1 : 2 * j + 2])
            )
        arr = out.reshape(ne)
    elif tt == 14:  # Q6_K
        nblk = ne // 256
        raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(nblk, 210)
        ql = raw[:, 0:128]
        qh = raw[:, 128:192]
        sc = raw[:, 192:208].view(np.int8).astype(np.float32)
        d = raw[:, 208:210].copy().view(np.float16).astype(np.float32)
        out = np.empty((nblk, 256), dtype=np.float32)
        for n128 in range(2):
            ql_s = ql[:, 64 * n128 : 64 * (n128 + 1)]
            qh_s = qh[:, 32 * n128 : 32 * (n128 + 1)]
            sc_s = sc[:, 8 * n128 : 8 * (n128 + 1)]
            for l in range(32):
                is_idx = l // 16
                q1 = ((ql_s[:, l] & 0x0F) | (((qh_s[:, l] >> 0) & 3) << 4)).astype(np.int32) - 32
                q2 = ((ql_s[:, l + 32] & 0x0F) | (((qh_s[:, l] >> 2) & 3) << 4)).astype(np.int32) - 32
                q3 = ((ql_s[:, l] >> 4) | (((qh_s[:, l] >> 4) & 3) << 4)).astype(np.int32) - 32
                q4 = ((ql_s[:, l + 32] >> 4) | (((qh_s[:, l] >> 6) & 3) << 4)).astype(np.int32) - 32
                out[:, 128 * n128 + l + 0] = (d[:, 0] * sc_s[:, is_idx + 0]) * q1
                out[:, 128 * n128 + l + 32] = (d[:, 0] * sc_s[:, is_idx + 2]) * q2
                out[:, 128 * n128 + l + 64] = (d[:, 0] * sc_s[:, is_idx + 4]) * q3
                out[:, 128 * n128 + l + 96] = (d[:, 0] * sc_s[:, is_idx + 6]) * q4
        arr = out.reshape(ne)
    else:
        raise ValueError(f"Unsupported GGUF tensor type {tt}")
    # GGUF dims are innermost-first; reverse for C/PyTorch row-major shape
    shape = list(reversed(dims))
    return torch.from_numpy(arr.reshape(shape))


def load_gguf_subset(gguf_path: str, prefixes: tuple[str, ...], exact: tuple[str, ...] = ()):
    """Load GGUF KV metadata + all tensors matching prefixes or exact names."""
    def read_str(f):
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def read_val(f, t):
        fmts = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in fmts:
            return struct.unpack(fmts[t], f.read(struct.calcsize(fmts[t])))[0]
        if t == 8:
            return read_str(f)
        if t == 9:
            et, cnt = struct.unpack("<IQ", f.read(12))
            if cnt > 256:
                for _ in range(cnt):
                    read_val(f, et)
                return None
            return [read_val(f, et) for _ in range(cnt)]
        raise ValueError(t)

    bsize = {0: (1, 4), 1: (1, 2), 30: (1, 2), 6: (32, 22), 8: (32, 34), 12: (256, 144), 14: (256, 210)}
    with open(gguf_path, "rb") as f:
        assert f.read(4) == b"GGUF"
        f.read(4)
        nt, nkv = struct.unpack("<QQ", f.read(16))
        kv = {}
        align = 32
        for _ in range(nkv):
            k = read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            v = read_val(f, t)
            if k == "general.alignment":
                align = int(v)
            if not k.startswith("tokenizer."):
                kv[k] = v
        tinfo = {}
        for _ in range(nt):
            name = read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            tt, off = struct.unpack("<IQ", f.read(12))
            tinfo[name] = (dims, tt, off)
        data_off = ((f.tell() + align - 1) // align) * align

        out = {}
        for name, (dims, tt, off) in tinfo.items():
            if name in exact or any(name.startswith(p) for p in prefixes):
                ne = 1
                for d in dims:
                    ne *= d
                bs, nb = bsize[tt]
                nbytes = (ne // bs) * nb
                f.seek(data_off + off)
                out[name] = dequant_gguf_tensor(f.read(nbytes), dims, tt)
    return kv, out


def load_rkce_windows(path: str, src_layer: int, win_len: int):
    """Load RKCE v1 preserving EXACT contiguous doc_ctx windows (no filtering of -1 boundary tokens!)."""
    with open(path, "rb") as f:
        assert f.read(4) == b"RKCE"
        ver, n_eng, d_eng, d_host, hc, target_layer = struct.unpack("<IIIIII", f.read(24))
        (n_tokens_hdr,) = struct.unpack("<Q", f.read(8))
        assert ver == 1
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
            h_all[off : off + nt] = np.frombuffer(f.read(nt * hc * d_host * 2), dtype=np.float16).reshape(nt, hc, d_host)
            for el in range(n_eng):
                nbytes = nt * d_eng * 2
                if el == src_layer:
                    e_all[off : off + nt] = np.frombuffer(f.read(nbytes), dtype=np.float16).reshape(nt, d_eng)
                else:
                    f.seek(nbytes, 1)
            off += nt
    n_win = off // win_len
    N = n_win * win_len
    h_w = h_all[:N].reshape(n_win, win_len, hc, d_host)
    e_w = e_all[:N].reshape(n_win, win_len, d_eng)
    y_w = targets[:N].reshape(n_win, win_len).astype(np.int64)
    meta = dict(n_eng=n_eng, d_eng=d_eng, d_host=d_host, hc=hc, target_layer=target_layer, n_win=n_win, win_len=win_len)
    return h_w, e_w, y_w, meta


class FrozenBailing3MlaMoeLayer(nn.Module):
    """Exact PyTorch replica of rookery's BailingMoe3 MLA Full-Attention + 128-Expert MoE layer."""

    def __init__(self, kv: dict, tensors: dict, layer_idx: int, device: torch.device, dtype=torch.float16):
        super().__init__()
        pfx = f"blk.{layer_idx}."
        self.n_head = int(kv.get("bailingmoe3.attention.head_count", 16))
        self.q_lora = int(kv.get("bailingmoe3.attention.q_lora_rank", 256))
        self.kv_lora = int(kv.get("bailingmoe3.attention.kv_lora_rank", 512))
        self.rope_dim = int(kv.get("bailingmoe3.rope.dimension_count", 64))
        self.qk_head_dim = int(kv.get("bailingmoe3.attention.key_length_mla", 192))
        self.qk_nope = self.qk_head_dim - self.rope_dim
        self.v_head_dim = int(kv.get("bailingmoe3.attention.value_length_mla", 128))
        self.rope_base = float(kv.get("bailingmoe3.rope.freq_base", 6000000.0))
        self.rms_eps = float(kv.get("bailingmoe3.attention.layer_norm_rms_epsilon", 1e-6))
        self.n_expert = int(kv.get("bailingmoe3.expert_count", 128))
        self.n_expert_used = int(kv.get("bailingmoe3.expert_used_count", 8))
        self.n_groups = int(kv.get("bailingmoe3.expert_group_count", 8))
        self.n_group_used = int(kv.get("bailingmoe3.expert_group_used_count", 4))
        self.exp_scale = float(kv.get("bailingmoe3.expert_weights_scale", 2.5))
        self.exp_norm = bool(kv.get("bailingmoe3.expert_weights_norm", True))

        def reg(name, t, dt=dtype):
            self.register_buffer(name, t.to(device=device, dtype=dt))

        reg("attn_norm", tensors[pfx + "attn_norm.weight"], torch.float32)
        reg("ffn_norm", tensors[pfx + "ffn_norm.weight"], torch.float32)
        reg("wq_a", tensors[pfx + "attn_q_a.weight"])
        reg("wq_a_norm", tensors[pfx + "attn_q_a_norm.weight"], torch.float32)
        reg("wq_b", tensors[pfx + "attn_q_b.weight"])
        reg("wkv_a", tensors[pfx + "attn_kv_a_mqa.weight"])
        reg("wkv_a_norm", tensors[pfx + "attn_kv_a_norm.weight"], torch.float32)
        reg("wk_b", tensors[pfx + "attn_k_b.weight"])  # [16, 512, 128]
        reg("wv_b", tensors[pfx + "attn_v_b.weight"])  # [16, 128, 512]
        reg("attn_gate", tensors[pfx + "attn_gate.weight"])
        reg("wo", tensors[pfx + "attn_output.weight"])

        reg("router_w", tensors[pfx + "ffn_gate_inp.weight"], torch.float32)
        reg("exp_bias", tensors[pfx + "exp_probs_b.bias"], torch.float32)
        reg("gate_exps", tensors[pfx + "ffn_gate_exps.weight"])  # [128, 512, 1536]
        reg("up_exps", tensors[pfx + "ffn_up_exps.weight"])      # [128, 512, 1536]
        reg("down_exps", tensors[pfx + "ffn_down_exps.weight"])  # [128, 1536, 512]
        reg("gate_sh", tensors[pfx + "ffn_gate_shexp.weight"])
        reg("up_sh", tensors[pfx + "ffn_up_shexp.weight"])
        reg("down_sh", tensors[pfx + "ffn_down_shexp.weight"])

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # Normal-pair RoPE matching rookery's mla_rope_split_q_kernel: pairs (2*d, 2*d+1)
        S = x.shape[1]
        pos = torch.arange(S, device=x.device, dtype=torch.float32)
        d_idx = torch.arange(self.rope_dim // 2, device=x.device, dtype=torch.float32)
        freqs = self.rope_base ** (-2.0 * d_idx / float(self.rope_dim))
        theta = pos[:, None] * freqs[None, :]  # [S, 32]
        c = torch.cos(theta)
        s = torch.sin(theta)
        if x.dim() == 4:  # [B, S, H, 64]
            c = c[None, :, None, :]
            s = s[None, :, None, :]
        else:  # [B, S, 64]
            c = c[None, :, :]
            s = s[None, :, :]
        xf = x.float()
        x0 = xf[..., 0::2]
        x1 = xf[..., 1::2]
        y0 = x0 * c - x1 * s
        y1 = x0 * s + x1 * c
        return torch.stack([y0, y1], dim=-1).flatten(-2).to(x.dtype)

    def _route_grouped_sigmoid(self, x_norm: torch.Tensor):
        # x_norm: [N, d_embd]
        logits = F.linear(x_norm.float(), self.router_w)
        p = torch.sigmoid(logits)
        s = p + self.exp_bias.unsqueeze(0)
        if self.n_groups > 1 and 0 < self.n_group_used < self.n_groups:
            n_per = self.n_expert // self.n_groups
            sg = s.view(-1, self.n_groups, n_per)
            top2_sum = sg.topk(2, dim=-1).values.sum(dim=-1)  # [N, n_groups]
            _, g_idx = top2_sum.topk(self.n_group_used, dim=-1)  # [N, n_group_used]
            g_mask = torch.zeros_like(top2_sum, dtype=torch.bool).scatter_(1, g_idx, True)
            e_mask = g_mask.unsqueeze(-1).expand(-1, self.n_groups, n_per).reshape(-1, self.n_expert)
            s = s.masked_fill(~e_mask, -1e9)
        _, top_ids = s.topk(self.n_expert_used, dim=-1)  # [N, k]
        top_w = torch.gather(p, 1, top_ids)
        if self.exp_norm:
            top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        top_w = top_w * self.exp_scale
        return top_ids, top_w

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, S, d_embd] float32
        B, S, D = h.shape
        dt = self.wq_a.dtype

        # 1. Absorbed RoPE-576 MLA Full Causal Attention
        xn = rmsnorm(h, self.attn_norm, self.rms_eps).to(dt)
        qr = rmsnorm(F.linear(xn, self.wq_a), self.wq_a_norm, self.rms_eps).to(dt)
        q_all = F.linear(qr, self.wq_b).view(B, S, self.n_head, self.qk_head_dim)
        q_nope = q_all[..., : self.qk_nope]  # [B, S, H, 128]
        q_pe = self._apply_rope(q_all[..., self.qk_nope :])  # [B, S, H, 64]

        kv_all = F.linear(xn, self.wkv_a)  # [B, S, 576]
        kv_lat = rmsnorm(kv_all[..., : self.kv_lora], self.wkv_a_norm, self.rms_eps).to(dt)  # [B, S, 512]
        k_pe = self._apply_rope(kv_all[..., self.kv_lora :])  # [B, S, 64]
        kv576 = torch.cat([kv_lat, k_pe], dim=-1)  # [B, S, 576]

        q_abs = torch.einsum("bshk,hmk->bshm", q_nope, self.wk_b)  # [B, S, H, 512]
        q576 = torch.cat([q_abs, q_pe], dim=-1)  # [B, S, H, 576]

        scale = 1.0 / math.sqrt(float(self.qk_head_dim))
        scores = torch.einsum("bshd,btd->bhst", q576.float(), kv576.float()) * scale
        causal = torch.triu(torch.ones(S, S, device=h.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))
        attn_w = torch.softmax(scores, dim=-1).to(dt)
        o_lat = torch.einsum("bhst,btd->bshd", attn_w, kv_lat)  # [B, S, H, 512]
        o = torch.einsum("bshk,hmk->bshm", o_lat, self.wv_b)  # [B, S, H, 128]
        gate = torch.sigmoid(F.linear(xn, self.attn_gate).float()).to(dt)  # [B, S, H]
        o = (o * gate.unsqueeze(-1)).reshape(B, S, self.n_head * self.v_head_dim)
        h = h + F.linear(o, self.wo).float()

        # 2. MoE FFN (grouped top-8 of 128 experts) + Shared Expert
        xn2 = rmsnorm(h, self.ffn_norm, self.rms_eps)
        xn2_flat = xn2.reshape(-1, D)
        xn2_dt = xn2_flat.to(dt)
        top_ids, top_w = self._route_grouped_sigmoid(xn2_flat)  # [N, k]

        # Shared expert (fast dense matmul)
        sh_hid = F.silu(F.linear(xn2_dt, self.gate_sh)) * F.linear(xn2_dt, self.up_sh)
        out_flat = F.linear(sh_hid, self.down_sh).float()

        # Routed experts (sparse dispatch by active expert ID — fast and differentiable!)
        flat_tok = torch.arange(xn2_flat.shape[0], device=h.device).unsqueeze(1).expand_as(top_ids).reshape(-1)
        flat_exp = top_ids.reshape(-1)
        flat_w = top_w.reshape(-1)
        order = torch.argsort(flat_exp)
        s_tok = flat_tok[order]
        s_exp = flat_exp[order]
        s_w = flat_w[order]
        uniq_e, counts = torch.unique_consecutive(s_exp, return_counts=True)
        offsets = torch.cat([torch.zeros(1, device=h.device, dtype=counts.dtype), counts.cumsum(0)])
        uniq_list = uniq_e.tolist()
        off_list = offsets.tolist()

        routed_acc = torch.zeros_like(out_flat)
        for idx, e_id in enumerate(uniq_list):
            st, en = off_list[idx], off_list[idx + 1]
            t_idx = s_tok[st:en]
            x_e = xn2_dt[t_idx]  # [c, D]
            g_e = F.linear(x_e, self.gate_exps[e_id])
            u_e = F.linear(x_e, self.up_exps[e_id])
            d_e = F.linear(F.silu(g_e) * u_e, self.down_exps[e_id]).float()
            routed_acc.index_add_(0, t_idx, d_e * s_w[st:en].unsqueeze(-1))

        h = h + (out_flat + routed_acc).view(B, S, D)
        return h


class FrozenInkling2Layer(nn.Module):
    """Exact PyTorch replica of a single Inkling-2-Mini postlude layer (SWA or Global Attn + ShortConv + 128-Expert SiTU MoE)."""

    def __init__(self, kv: dict, tensors: dict, layer_idx: int, device: torch.device, dtype=torch.float16):
        super().__init__()
        pfx = f"blk.{layer_idx}."
        arch = kv.get("general.architecture", "inkling2_loop")
        apfx = arch + "."

        def _first_int(key, default):
            v = kv.get(apfx + key, default)
            if isinstance(v, list):
                return int(v[layer_idx]) if len(v) > layer_idx else (int(v[-1]) if v else int(default))
            return int(v)

        self.H = _first_int("embedding_length", 2304)
        self.n_head = _first_int("attention.head_count", 24)
        self.n_kv = _first_int("attention.head_count_kv", 8)
        self.head_dim = _first_int("attention.key_length", 128)
        self.n_rel = _first_int("attention.relative_length", 16)
        self.n_ext = _first_int("attention.relative_extent", 1024)
        self.qk_clip = float(kv.get(apfx + "attention.qk_clip", 22.627417))
        self.rel_clip = float(kv.get(apfx + "attention.relative_clip", 8.0))
        self.log_floor = float(kv.get(apfx + "attention.log_scale_floor", 8192))
        self.log_alpha = float(kv.get(apfx + "attention.log_scale_alpha", 0.1))
        self.n_expert = _first_int("expert_count", 128)
        self.n_expert_used = _first_int("expert_used_count", 8)
        self.ff_exp = _first_int("expert_feed_forward_length", 1152)
        self.situ_beta = float(kv.get(apfx + "activation.situ_beta", 4.0))
        self.situ_lb = float(kv.get(apfx + "activation.situ_linear_beta", 25.0))
        self.exp_scale = float(kv.get(apfx + "expert_weights_scale", 8.0))
        self.rms_eps = float(kv.get(apfx + "attention.layer_norm_rms_epsilon", 1e-6))
        self.is_swa = (pfx + "attn_norm.weight") in tensors

        def reg(name, t, dt=dtype):
            self.register_buffer(name, t.to(device=device, dtype=dt))

        if self.is_swa:
            reg("attn_res_q", tensors[pfx + "attn_res_query.weight"].reshape(self.H), torch.float32)
            reg("attn_norm", tensors[pfx + "attn_norm.weight"], torch.float32)
        else:
            reg("attn_res_q", tensors[pfx + "attn_res_query.weight"].reshape(3, self.H), torch.float32)
            reg("q_norm", tensors[pfx + "attn_q_inp_norm.weight"], torch.float32)
            reg("k_norm", tensors[pfx + "attn_k_inp_norm.weight"], torch.float32)
            reg("v_norm", tensors[pfx + "attn_v_inp_norm.weight"], torch.float32)
            reg("wk_dyn", tensors[pfx + "attn_k_conv_proj.weight"])
            reg("wv_dyn", tensors[pfx + "attn_v_conv_proj.weight"])

        reg("ffn_res_q", tensors[pfx + "ffn_res_query.weight"].reshape(self.H), torch.float32)
        reg("ffn_norm", tensors[pfx + "ffn_norm.weight"], torch.float32)

        reg("wq", tensors[pfx + "attn_q.weight"])
        reg("wk", tensors[pfx + "attn_k.weight"])
        reg("wv", tensors[pfx + "attn_v.weight"])
        reg("wo", tensors[pfx + "attn_output.weight"])
        reg("wgate", tensors[pfx + "attn_gate.weight"])
        reg("wrel", tensors[pfx + "attn_rel.weight"])
        reg("wsink", tensors[pfx + "attn_sink_proj.weight"])

        reg("q_clip_w", tensors[pfx + "attn_q_clip.weight"].reshape(self.n_head, self.head_dim), torch.float32)
        reg("rel_clip_w", tensors[pfx + "attn_rel_clip.weight"].reshape(self.n_head, self.n_rel), torch.float32)
        reg("rel_bias", tensors[pfx + "attn_rel_bias.weight"], torch.float32)
        reg("k_conv", tensors[pfx + "attn_k_conv.weight"], torch.float32)
        reg("v_conv", tensors[pfx + "attn_v_conv.weight"], torch.float32)
        reg("attn_pconv", tensors[pfx + "attn_post_conv.weight"], torch.float32)
        reg("ffn_pconv", tensors[pfx + "ffn_post_conv.weight"], torch.float32)

        reg("router_w", tensors[pfx + "ffn_gate_inp.weight"], torch.float32)
        exp_b = tensors.get(pfx + "exp_probs_b.bias", torch.zeros(self.n_expert))
        reg("exp_bias", exp_b, torch.float32)
        reg("gu_exps", tensors[pfx + "ffn_gate_up_exps.weight"])
        reg("dn_exps", tensors[pfx + "ffn_down_exps.weight"])

    def _depth_mix(self, blocks: torch.Tensor, cur: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        states = torch.cat([blocks, cur.unsqueeze(2)], dim=2)
        inv_rms = torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + self.rms_eps)
        st_norm = states * inv_rms
        q = query.unsqueeze(0) if query.dim() == 1 else query
        scores = torch.einsum("bsnh,qh->bsqn", st_norm, q) * (1.0 / math.sqrt(float(self.H)))
        w = torch.softmax(scores, dim=-1)
        out = torch.einsum("bsqn,bsnh->bsqh", w, st_norm)
        return out.squeeze(2) if query.dim() == 1 else out

    def _shortconv(self, x: torch.Tensor, w: torch.Tensor, dyn: torch.Tensor = None) -> torch.Tensor:
        B, S, C = x.shape
        x_pad = F.pad(x, (0, 0, 3, 0))
        s0 = x_pad[:, 0:S, :]
        s1 = x_pad[:, 1:S+1, :]
        s2 = x_pad[:, 2:S+2, :]
        xt = x_pad[:, 3:S+3, :]
        acc = s0 * w[:, 0] + s1 * w[:, 1] + s2 * w[:, 2] + xt * w[:, 3]
        if dyn is not None:
            hd = C // self.n_kv
            d_exp = dyn.unsqueeze(3).expand(B, S, self.n_kv, hd, 4).reshape(B, S, C, 4)
            acc = acc + d_exp[..., 0] * xt + d_exp[..., 1] * s2 + d_exp[..., 2] * s1 + d_exp[..., 3] * s0
            inv_scale = 1.0 / 3.0
        else:
            inv_scale = 1.0 / math.sqrt(5.0)
        return (xt + acc) * inv_scale

    def _l2_clip(self, x: torch.Tensor, scale_w: torch.Tensor, thresh: float, apply_tau: bool) -> torch.Tensor:
        xf = x.float()
        if scale_w is not None:
            xf = xf * scale_w[None, None, :, :]
        norm = torch.sqrt(xf.pow(2).sum(dim=-1, keepdim=True)) + 1e-6
        mul = torch.clamp(thresh / norm, max=1.0)
        if apply_tau:
            S = x.shape[1]
            pos = torch.arange(1, S + 1, device=x.device, dtype=torch.float32)
            tau = 1.0 + self.log_alpha * torch.log(torch.clamp(pos / self.log_floor, min=1.0))
            mul = mul * tau[None, :, None, None]
        return xf * mul

    def forward(self, h: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        B, S, H = h.shape
        dt = self.wq.dtype

        if self.is_swa:
            mixed1 = self._depth_mix(blocks, h, self.attn_res_q)
            nq = nk = nv = rmsnorm(mixed1, self.attn_norm, self.rms_eps).to(dt)
            k_dyn = v_dyn = None
        else:
            mixed3 = self._depth_mix(blocks, h, self.attn_res_q)
            nq = rmsnorm(mixed3[:, :, 0, :], self.q_norm, self.rms_eps).to(dt)
            nk = rmsnorm(mixed3[:, :, 1, :], self.k_norm, self.rms_eps).to(dt)
            nv = rmsnorm(mixed3[:, :, 2, :], self.v_norm, self.rms_eps).to(dt)
            k_dyn = F.linear(nk, self.wk_dyn).float().view(B, S, self.n_kv, 4)
            v_dyn = F.linear(nv, self.wv_dyn).float().view(B, S, self.n_kv, 4)

        q = F.linear(nq, self.wq).float().view(B, S, self.n_head, self.head_dim)
        r = F.linear(nq, self.wrel).float().view(B, S, self.n_head, self.n_rel)
        sinks = F.linear(nq, self.wsink).float()
        attn_gate = F.linear(nq, self.wgate).float()

        k_raw = F.linear(nk, self.wk).float()
        v_raw = F.linear(nv, self.wv).float()

        k = self._shortconv(k_raw, self.k_conv, k_dyn).view(B, S, self.n_kv, self.head_dim)
        v = self._shortconv(v_raw, self.v_conv, v_dyn).view(B, S, self.n_kv, self.head_dim)

        q = self._l2_clip(q, self.q_clip_w, self.qk_clip, apply_tau=(not self.is_swa))
        k = self._l2_clip(k, None, self.qk_clip, apply_tau=False)
        v = v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + self.rms_eps)
        r = self._l2_clip(r, self.rel_clip_w, self.rel_clip, apply_tau=(not self.is_swa))

        gqa = self.n_head // self.n_kv
        k_exp = k.unsqueeze(3).expand(B, S, self.n_kv, gqa, self.head_dim).reshape(B, S, self.n_head, self.head_dim)
        v_exp = v.unsqueeze(3).expand(B, S, self.n_kv, gqa, self.head_dim).reshape(B, S, self.n_head, self.head_dim)
        scores = torch.einsum("bshd,bthd->bhst", q, k_exp) * (1.0 / math.sqrt(float(self.head_dim)))

        rb_all = torch.einsum("bshr,dr->bhsd", r, self.rel_bias[:S]) - sinks.permute(0, 2, 1).unsqueeze(-1)
        s_idx = torch.arange(S, device=h.device)[:, None]
        t_idx = torch.arange(S, device=h.device)[None, :]
        dist = (s_idx - t_idx).clamp(min=0)
        rel_mat = rb_all.gather(-1, dist[None, None, :, :].expand(B, self.n_head, S, S))
        scores = scores + rel_mat
        causal = t_idx > s_idx
        scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))

        sink_col = torch.zeros(B, self.n_head, S, 1, device=h.device, dtype=scores.dtype)
        probs = torch.softmax(torch.cat([scores, sink_col], dim=-1), dim=-1)[..., :S]
        attn_out = torch.einsum("bhst,bthd->bshd", probs, v_exp).reshape(B, S, self.n_head * self.head_dim)
        attn_out = (attn_out * torch.sigmoid(attn_gate)).to(dt)
        wo_raw = F.linear(attn_out, self.wo).float()
        h = h + self._shortconv(wo_raw, self.attn_pconv, None)

        ffn_x = self._depth_mix(blocks, h, self.ffn_res_q)
        xn2 = rmsnorm(ffn_x, self.ffn_norm, self.rms_eps)
        xn2_flat = xn2.reshape(-1, H)
        xn2_dt = xn2_flat.to(dt)

        rlogits = F.linear(xn2_flat, self.router_w)
        p = torch.sigmoid(rlogits)
        s = p + self.exp_bias.unsqueeze(0)
        _, top_ids = s.topk(self.n_expert_used, dim=-1)
        top_w = torch.gather(p, 1, top_ids)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)) * self.exp_scale

        flat_tok = torch.arange(xn2_flat.shape[0], device=h.device).unsqueeze(1).expand_as(top_ids).reshape(-1)
        flat_exp = top_ids.reshape(-1)
        flat_w = top_w.reshape(-1)
        order = torch.argsort(flat_exp)
        s_tok, s_exp, s_w = flat_tok[order], flat_exp[order], flat_w[order]
        uniq_e, counts = torch.unique_consecutive(s_exp, return_counts=True)
        offsets = torch.cat([torch.zeros(1, device=h.device, dtype=counts.dtype), counts.cumsum(0)])
        routed_acc = torch.zeros_like(xn2_flat)
        ff = self.ff_exp
        for idx, e_id in enumerate(uniq_e.tolist()):
            st, en = offsets[idx].item(), offsets[idx + 1].item()
            t_idx_e = s_tok[st:en]
            x_e = xn2_dt[t_idx_e]
            gu = F.linear(x_e, self.gu_exps[e_id]).float()
            g_e, u_e = gu[:, :ff], gu[:, ff:]
            act = (self.situ_beta * torch.tanh(g_e / self.situ_beta) * torch.sigmoid(g_e)) * (
                self.situ_lb * torch.tanh(u_e / self.situ_lb)
            )
            d_e = F.linear(act.to(dt), self.dn_exps[e_id]).float()
            routed_acc.index_add_(0, t_idx_e, d_e * s_w[st:en].unsqueeze(-1))

        moe_raw = routed_acc.view(B, S, H)
        h = h + self._shortconv(moe_raw, self.ffn_pconv, None)
        return h


class FrozenInkling2GlobalMoeTail(nn.Module):
    """Exact PyTorch replica of Inkling-2-Mini postlude layers [start_layer..23] -> output_depth_mix -> output_norm -> lm_head."""

    def __init__(self, kv: dict, tensors: dict, device: torch.device, dtype=torch.float16, start_layer: int = 23):
        super().__init__()
        arch = kv.get("general.architecture", "inkling2_loop")
        apfx = arch + "."
        self.H = int(kv.get(apfx + "embedding_length", 2304))
        self.logit_softcap = float(kv.get(apfx + "final_logit_softcapping", 30.0))
        self.n_vocab_unpadded = int(kv.get(apfx + "vocab_size_unpadded", 200058))
        self.rms_eps = float(kv.get(apfx + "attention.layer_norm_rms_epsilon", 1e-6))

        self.layers = nn.ModuleList([
            FrozenInkling2Layer(kv, tensors, il, device=device, dtype=dtype)
            for il in range(start_layer, 24)
        ])
        self.register_buffer("out_res_q", tensors["output_res_query.weight"].reshape(self.H).to(device=device, dtype=torch.float32))
        self.register_buffer("out_norm", tensors["output_norm.weight"].to(device=device, dtype=torch.float32))
        self.register_buffer("lm_head", tensors["output.weight"][: self.n_vocab_unpadded].to(device=device, dtype=dtype))

    def forward(self, h: torch.Tensor, blocks: torch.Tensor):
        for layer in self.layers:
            if h.requires_grad and len(self.layers) > 1:
                h = torch.utils.checkpoint.checkpoint(layer, h, blocks, use_reentrant=False)
            else:
                h = layer(h, blocks)
        h_mixed = self.layers[-1]._depth_mix(blocks, h, self.out_res_q)
        x_out = rmsnorm(h_mixed, self.out_norm, self.rms_eps).to(self.lm_head.dtype)
        logits = F.linear(x_out, self.lm_head).float()
        if self.logit_softcap > 0.0:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return h_mixed, logits


class FrozenSenseNovaGqaLayer(nn.Module):
    """Exact PyTorch replica of SenseNova-Loop-9B postlude GQA Full-Attention + Dense SwiGLU layer (visit 55 / blk.31 or blk.55)."""

    def __init__(self, kv: dict, tensors: dict, blk_pfx: str, device: torch.device, dtype=torch.float16):
        super().__init__()
        apfx = "sensenova_loop."
        self.n_head = int(kv.get(apfx + "attention.head_count", 16))
        self.n_kv = int(kv.get(apfx + "attention.head_count_kv", 4))
        self.head_dim = int(kv.get(apfx + "attention.key_length", 256))
        self.rope_dim = int(kv.get(apfx + "rope.dimension_count", 64))
        self.rope_base = float(kv.get(apfx + "rope.freq_base", 10000000.0))
        rope_sc = str(kv.get(apfx + "rope.scaling.type", "none"))
        self.yarn_factor = float(kv.get(apfx + "rope.scaling.factor", 1.0)) if rope_sc == "yarn" else 1.0
        self.orig_ctx = int(kv.get(apfx + "rope.scaling.original_context_length", 262144))
        self.rms_eps = float(kv.get(apfx + "attention.layer_norm_rms_epsilon", 1e-6))

        half = self.rope_dim // 2
        if self.yarn_factor <= 1.0:
            self.yarn_mscale = 1.0
            inv_freq = [self.rope_base ** (-2.0 * i / self.rope_dim) for i in range(half)]
        else:
            def _corr(nrot):
                return (self.rope_dim * math.log(float(self.orig_ctx) / (nrot * 2.0 * math.pi))) / (2.0 * math.log(self.rope_base))
            low = max(math.floor(_corr(32.0)), 0.0)
            high = min(math.ceil(_corr(1.0)), float(self.rope_dim - 1))
            span = 0.001 if abs(high - low) < 1e-9 else (high - low)
            self.yarn_mscale = 0.1 * math.log(self.yarn_factor) + 1.0
            inv_freq = []
            for i in range(half):
                fe = self.rope_base ** (-2.0 * i / self.rope_dim)
                fi = fe / self.yarn_factor
                ramp = max(0.0, min(1.0, (float(i) - low) / span))
                inv_freq.append(fi * ramp + fe * (1.0 - ramp))
        self.register_buffer("inv_freq", torch.tensor(inv_freq, device=device, dtype=torch.float32))

        def reg(name, t, dt=dtype):
            self.register_buffer(name, t.to(device=device, dtype=dt))

        reg("attn_norm", tensors[blk_pfx + "attn_norm.weight"], torch.float32)
        reg("post_attn_norm", tensors[blk_pfx + "post_attention_norm.weight"], torch.float32)
        reg("wq", tensors[blk_pfx + "attn_q.weight"])
        reg("wk", tensors[blk_pfx + "attn_k.weight"])
        reg("wv", tensors[blk_pfx + "attn_v.weight"])
        reg("wo", tensors[blk_pfx + "attn_output.weight"])
        reg("q_norm", tensors[blk_pfx + "attn_q_norm.weight"], torch.float32)
        reg("k_norm", tensors[blk_pfx + "attn_k_norm.weight"], torch.float32)
        reg("ffn_gate", tensors[blk_pfx + "ffn_gate.weight"])
        reg("ffn_up", tensors[blk_pfx + "ffn_up.weight"])
        reg("ffn_down", tensors[blk_pfx + "ffn_down.weight"])

    def _apply_yarn_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, n_h, head_dim] float32; NeoX half-split on [0:rope_dim]
        S = x.shape[1]
        half = self.rope_dim // 2
        pos = torch.arange(S, device=x.device, dtype=torch.float32)
        ang = pos[:, None] * self.inv_freq[None, :]  # [S, half]
        c = (torch.cos(ang) * self.yarn_mscale)[None, :, None, :]
        s = (torch.sin(ang) * self.yarn_mscale)[None, :, None, :]
        x0 = x[..., :half]
        x1 = x[..., half : self.rope_dim]
        y0 = x0 * c - x1 * s
        y1 = x0 * s + x1 * c
        if self.rope_dim < self.head_dim:
            return torch.cat([y0, y1, x[..., self.rope_dim :]], dim=-1)
        return torch.cat([y0, y1], dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, S, D = h.shape
        xn = rmsnorm(h, self.attn_norm, self.rms_eps).to(self.wq.dtype)
        q_full = F.linear(xn, self.wq).float().view(B, S, self.n_head, 2 * self.head_dim)
        q = rmsnorm(q_full[..., : self.head_dim], self.q_norm, self.rms_eps)
        gate = q_full[..., self.head_dim :]
        k = F.linear(xn, self.wk).float().view(B, S, self.n_kv, self.head_dim)
        k = rmsnorm(k, self.k_norm, self.rms_eps)
        v = F.linear(xn, self.wv).float().view(B, S, self.n_kv, self.head_dim)

        q = self._apply_yarn_rope(q)
        k = self._apply_yarn_rope(k)

        rep = self.n_head // self.n_kv
        q_t = q.permute(0, 2, 1, 3)  # [B, H, S, Dh]
        k_t = k.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1)
        v_t = v.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1)
        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        attn_out = attn_out.permute(0, 2, 1, 3) * torch.sigmoid(gate)
        sub = F.linear(attn_out.reshape(B, S, self.n_head * self.head_dim).to(self.wo.dtype), self.wo).float()
        h = h + sub

        fn = rmsnorm(h, self.post_attn_norm, self.rms_eps).to(self.ffn_gate.dtype)
        g = F.silu(F.linear(fn, self.ffn_gate).float())
        u = F.linear(fn, self.ffn_up).float()
        h = h + F.linear((g * u).to(self.ffn_down.dtype), self.ffn_down).float()
        return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rkce", required=True, help="Contiguous doc_ctx-windowed RKCE at mid-layer L_k (e.g. L22)")
    ap.add_argument("--blocks-path", default="", help="FP16 [N, 4, H] depth-mix blocks file for inkling2_loop")
    ap.add_argument("--parity-rkce", default="", help="Optional L23 reference RKCE on same prefix for mandatory parity gate")
    ap.add_argument("--gguf", required=True, help="Host model GGUF")
    ap.add_argument("--out", required=True, help="Output RKAD v3 adapter path")
    ap.add_argument("--win-len", type=int, default=256)
    ap.add_argument("--src-layer", type=int, default=0)
    ap.add_argument("--export-src-layer", type=int, default=-1, help="Override engram src_layer written into RKAD header (default: --src-layer)")
    ap.add_argument("--engram-kv-gguf", default="", help="Optional provider shard GGUF for native W_V/W_K pre-decoding (folded algebraically into RKAD v3)")
    ap.add_argument("--engram-kv-tensor", default="", help="Optional explicit engram_kv tensor name (e.g. blk.1.engram_kv.weight or blk.14.engram_kv.weight)")
    ap.add_argument("--engram-kv-hc", type=int, default=4, help="Provider HC stream count for engram_kv split (default: 4 for DeepSeek)")
    ap.add_argument("--provider-arch", required=True)
    ap.add_argument("--provider-mode", required=True)
    ap.add_argument("--provider-fingerprint", required=True)
    ap.add_argument("--host-arch", default="bailingmoe3")
    ap.add_argument("--host-n-layers", type=int, default=24)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--pca-dim", type=int, default=0, help="Optional whitened PCA dim folded algebraically into RKAD v3")
    ap.add_argument("--scale", type=float, default=0.08)
    ap.add_argument("--delta-clip", type=float, default=0.10)
    ap.add_argument("--gate-bias", type=float, default=-1.5)
    ap.add_argument("--surprisal-alpha", type=float, default=3.0)
    ap.add_argument("--surprisal-thresh", type=float, default=1.5)
    ap.add_argument("--kl-beta", type=float, default=1.5)
    ap.add_argument("--shuf-kl-beta", type=float, default=1.5)
    ap.add_argument("--delta-reg", type=float, default=0.02)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-windows", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--val-windows", type=int, default=24)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    h_w, e_w, y_w, meta = load_rkce_windows(args.rkce, args.src_layer, args.win_len)
    n_win = meta["n_win"]
    val_step = max(2, n_win // max(4, args.val_windows))
    val_mask = np.zeros(n_win, dtype=bool)
    val_mask[val_step - 1 :: val_step] = True
    val_idx = np.where(val_mask)[0]
    tr_idx = np.where(~val_mask)[0]
    n_tr, n_val = len(tr_idx), len(val_idx)
    tgt_l = int(meta["target_layer"])
    if args.host_arch == "sensenova_loop":
        assert tgt_l == 54, f"sensenova_loop expects target_layer=54 (before visit 55 / blk.31), got {tgt_l}"
        tail_layers = [31, 55]
    else:
        assert 18 <= tgt_l <= 22, f"Currently expects target_layer in [18..22], got {tgt_l}"
        tail_layers = list(range(tgt_l + 1, 24))
    print(f"[MidLayer] Loaded {args.rkce}: {n_win} windows x {args.win_len} toks "
          f"(tr={n_tr}, val={n_val} interleaved), L{tgt_l} -> frozen blk.{tail_layers[0]}..{tail_layers[-1]}, "
          f"d_eng={meta['d_eng']}, d_host={meta['d_host']}", flush=True)

    print(f"[MidLayer] Loading frozen blk.{tail_layers[0]}..{tail_layers[-1]} + output_norm + output.weight from {args.gguf}...", flush=True)
    kv, tensors = load_gguf_subset(
        args.gguf,
        prefixes=tuple(f"blk.{il}." for il in tail_layers),
        exact=("output_norm.weight", "output.weight", "token_embd.weight", "output_res_query.weight"),
    )
    gguf_arch = str(kv.get("general.architecture", args.host_arch))
    is_inkling2 = gguf_arch in ("inkling2_loop", "inkling2")
    is_sensenova = gguf_arch == "sensenova_loop"

    blocks_w = None
    if is_sensenova:
        blk_pfx = "blk.55." if "blk.55.attn_norm.weight" in tensors else "blk.31."
        layer_sn = FrozenSenseNovaGqaLayer(kv, tensors, blk_pfx, device=device, dtype=torch.float16).eval()
        for p in layer_sn.parameters():
            p.requires_grad_(False)
        out_norm = tensors["output_norm.weight"].to(device=device, dtype=torch.float32)
        lm_head_w = (tensors["output.weight"] if "output.weight" in tensors else tensors["token_embd.weight"]).to(
            device=device, dtype=torch.float16
        )
        rms_eps = layer_sn.rms_eps

        def forward_tail(h_in: torch.Tensor, blk_in: torch.Tensor = None):
            h_out = layer_sn(h_in)
            xn = rmsnorm(h_out, out_norm, rms_eps).to(lm_head_w.dtype)
            logits = F.linear(xn, lm_head_w).float()
            return h_out, logits

        def ref_logits_from_h23(h_ref: torch.Tensor):
            return F.linear(rmsnorm(h_ref, out_norm, rms_eps).to(lm_head_w.dtype), lm_head_w).float()
    elif is_inkling2:
        assert args.blocks_path, "--blocks-path is required for inkling2_loop / inkling2"
        n_blocks_exp = 4
        raw_blk = np.memmap(args.blocks_path, dtype=np.float16, mode="r")
        n_tok_blk = raw_blk.size // (n_blocks_exp * meta["d_host"])
        assert n_tok_blk >= n_win * args.win_len, (
            f"blocks file has {n_tok_blk} tokens < expected {n_win * args.win_len}"
        )
        blocks_w = raw_blk[: n_win * args.win_len * n_blocks_exp * meta["d_host"]].reshape(
            n_win, args.win_len, n_blocks_exp, meta["d_host"]
        )
        tail_ink2 = FrozenInkling2GlobalMoeTail(
            kv, tensors, device=device, dtype=torch.float16, start_layer=tgt_l + 1
        ).eval()
        for p in tail_ink2.parameters():
            p.requires_grad_(False)
        out_norm = tail_ink2.out_norm
        lm_head_w = tail_ink2.lm_head
        rms_eps = tail_ink2.rms_eps
        logit_softcap = tail_ink2.logit_softcap

        def forward_tail(h_in: torch.Tensor, blk_in: torch.Tensor = None):
            assert blk_in is not None
            return tail_ink2(h_in, blk_in)

        def ref_logits_from_h23(h23_ref: torch.Tensor):
            xn = rmsnorm(h23_ref, out_norm, rms_eps).to(lm_head_w.dtype)
            lg = F.linear(xn, lm_head_w).float()
            if logit_softcap > 0.0:
                lg = logit_softcap * torch.tanh(lg / logit_softcap)
            return lg
    else:
        layer23 = FrozenBailing3MlaMoeLayer(kv, tensors, 23, device=device, dtype=torch.float16).eval()
        for p in layer23.parameters():
            p.requires_grad_(False)
        out_norm = tensors["output_norm.weight"].to(device=device, dtype=torch.float32)
        lm_head_w = (tensors["output.weight"] if "output.weight" in tensors else tensors["token_embd.weight"]).to(
            device=device, dtype=torch.float16
        )
        rms_eps = float(kv.get("bailingmoe3.attention.layer_norm_rms_epsilon", 1e-6))

        def forward_tail(h_in: torch.Tensor, blk_in: torch.Tensor = None):
            h23 = layer23(h_in)
            xn = rmsnorm(h23, out_norm, rms_eps).to(lm_head_w.dtype)
            logits = F.linear(xn, lm_head_w).float()
            return h23, logits

        def ref_logits_from_h23(h23_ref: torch.Tensor):
            return F.linear(rmsnorm(h23_ref, out_norm, rms_eps).to(lm_head_w.dtype), lm_head_w).float()

    # MANDATORY PARITY GATE before Epoch 1!
    if args.parity_rkce:
        hp_w, _, yp_w, _ = load_rkce_windows(args.parity_rkce, args.src_layer, args.win_len)
        with torch.no_grad():
            h22_0 = torch.from_numpy(h_w[:1, :, 0, :]).to(device=device, dtype=torch.float32)
            blk_0 = (
                torch.from_numpy(np.array(blocks_w[:1], copy=True)).to(device=device, dtype=torch.float32)
                if blocks_w is not None
                else None
            )
            h23_ref = torch.from_numpy(hp_w[:1, :, 0, :]).to(device=device, dtype=torch.float32)
            y_ref = torch.from_numpy(yp_w[:1]).to(device=device)
            h23_py, logits_py = forward_tail(h22_0, blk_0)
            logits_ref = ref_logits_from_h23(h23_ref)
            cos = F.cosine_similarity(h23_py.reshape(-1, meta["d_host"]), h23_ref.reshape(-1, meta["d_host"]), dim=-1).mean().item()
            max_h_diff = (h23_py - h23_ref).abs().max().item()
            valid_m = y_ref >= 0
            nll_py = F.cross_entropy(logits_py[valid_m], y_ref[valid_m]).item()
            nll_ref = F.cross_entropy(logits_ref[valid_m], y_ref[valid_m]).item()
            max_l_diff = (logits_py - logits_ref).abs().max().item()
            top1_agree = (logits_py.argmax(-1) == logits_ref.argmax(-1)).float().mean().item() * 100.0
            print(f"[PARITY GATE] h23 cos={cos:.6f}, max|dh|={max_h_diff:.4f}, max|dlogit|={max_l_diff:.4f}, "
                  f"top1_agree={top1_agree:.2f}%, NLL py={nll_py:.5f} vs rookery={nll_ref:.5f} (diff={abs(nll_py-nll_ref):.5f})",
                  flush=True)
            if cos < 0.995 or abs(nll_py - nll_ref) > 0.05:
                raise RuntimeError(f"PARITY GATE FAILED! cos={cos:.6f}, |dNLL|={abs(nll_py-nll_ref):.5f}")

    # Mean-center Engram embeddings on training windows (+ optional native provider W_V/W_K OR whitened PCA fold!)
    mu_e = e_w[tr_idx].astype(np.float32).reshape(-1, meta["d_eng"]).mean(axis=0)
    mu_e_t = torch.from_numpy(mu_e).to(device=device, dtype=torch.float32)

    W_V_fold = None
    W_K_fold = None
    d_bridge_in = meta["d_eng"]
    pca_tag = ""
    if args.engram_kv_gguf:
        W_V_fold, W_K_fold, kv_tname = load_provider_engram_kv(
            args.engram_kv_gguf,
            src_layer=args.export_src_layer if args.export_src_layer >= 0 else args.src_layer,
            tensor_name=args.engram_kv_tensor,
            hc_provider=args.engram_kv_hc,
            device=device,
            max_cols=meta["d_eng"],
        )
        # Normalize columns of W_V / W_K so pre-decoded features have unit RMS on training set
        with torch.no_grad():
            e_tr_flat = e_w[tr_idx].reshape(-1, meta["d_eng"])
            step_p = max(1, e_tr_flat.shape[0] // 32768)
            X_sub = torch.from_numpy(e_tr_flat[::step_p]).to(device=device, dtype=torch.float32) - mu_e_t
            v_rms = F.linear(X_sub, W_V_fold).pow(2).mean().sqrt().clamp(min=1e-5)
            k_rms = F.linear(X_sub, W_K_fold).pow(2).mean().sqrt().clamp(min=1e-5)
            W_V_fold = (W_V_fold / v_rms).contiguous()
            W_K_fold = (W_K_fold / k_rms).contiguous()
        d_bridge_in = int(W_V_fold.shape[0])
        pca_tag = f"_nativekv{d_bridge_in}"
        print(f"[MidLayer NativeKV] Loaded provider W_V/W_K ({d_bridge_in}x{meta['d_eng']}) from {args.engram_kv_gguf}", flush=True)
    elif args.pca_dim > 0 and args.pca_dim < meta["d_eng"]:
        with torch.no_grad():
            e_tr_flat = e_w[tr_idx].reshape(-1, meta["d_eng"])
            step_p = max(1, e_tr_flat.shape[0] // 32768)
            X_sub = torch.from_numpy(e_tr_flat[::step_p]).to(device=device, dtype=torch.float32) - mu_e_t
            cov = (X_sub.T @ X_sub) / float(X_sub.shape[0])
            evals, evecs = torch.linalg.eigh(cov)
            idx_desc = torch.argsort(evals, descending=True)
            evals, evecs = evals[idx_desc], evecs[:, idx_desc]
            k = min(args.pca_dim, int((evals > 1e-7).sum().item()))
            var_exp = evals[:k].sum().item() / evals.clamp(min=0.0).sum().item()
            inv_std = torch.rsqrt(evals[:k].clamp(min=1e-5))
            W_pca = (evecs[:, :k] * inv_std.unsqueeze(0)).T.contiguous()
            W_V_fold = W_pca
            W_K_fold = W_pca
            d_bridge_in = k
            pca_tag = f"_pca{k}"
            print(f"[MidLayer PCA] Fitted whitened PCA ({k}x{meta['d_eng']}): explained_var={100.0*var_exp:.1f}%", flush=True)

    bridge = RookeryGatedBridge(
        d_in=d_bridge_in,
        d_host=meta["d_host"],
        hc=meta["hc"],
        rank=args.rank,
        init_scale=args.scale,
        dropout=args.dropout,
        gate_bias=args.gate_bias,
    ).to(device)

    @torch.no_grad()
    def eval_val(shuffled: bool = False):
        bridge.eval()
        tot_base_nll, tot_nll, tot_tok = 0.0, 0.0, 0
        base_h1, h1 = 0, 0
        g_sum, rel_sum, n_b = 0.0, 0.0, 0
        gen = torch.Generator().manual_seed(1337)
        for i in range(0, n_val, args.batch_windows):
            idx = val_idx[i : i + args.batch_windows]
            hb = torch.from_numpy(h_w[idx]).to(device=device, dtype=torch.float32)  # [B, S, 1, D]
            blkb = (
                torch.from_numpy(np.array(blocks_w[idx], copy=True)).to(device=device, dtype=torch.float32)
                if blocks_w is not None
                else None
            )
            eb = torch.from_numpy(e_w[idx]).to(device=device, dtype=torch.float32) - mu_e_t
            yb = torch.from_numpy(y_w[idx]).to(device=device)
            if shuffled:
                perm = torch.randperm(eb.shape[0] * eb.shape[1], generator=gen)
                eb = eb.reshape(-1, meta["d_eng"])[perm].reshape_as(eb)
            valid = yb >= 0
            _, base_logits = forward_tail(hb[:, :, 0, :], blkb)
            eb_in = eb.reshape(-1, meta["d_eng"])
            eb_v = F.linear(eb_in, W_V_fold) if W_V_fold is not None else eb_in
            eb_k = F.linear(eb_in, W_K_fold) if W_K_fold is not None else eb_v
            h_inj, rel_n, gm = bridge(hb.reshape(-1, meta["hc"], meta["d_host"]), eb_v, eb_k, return_stats=True)
            h_inj = h_inj.reshape(hb.shape[0], hb.shape[1], meta["d_host"])
            _, logits = forward_tail(h_inj, blkb)
            tot_base_nll += F.cross_entropy(base_logits[valid], yb[valid], reduction="sum").item()
            tot_nll += F.cross_entropy(logits[valid], yb[valid], reduction="sum").item()
            base_h1 += (base_logits[valid].argmax(-1) == yb[valid]).sum().item()
            h1 += (logits[valid].argmax(-1) == yb[valid]).sum().item()
            tot_tok += int(valid.sum().item())
            g_sum += gm.item()
            rel_sum += rel_n.item()
            n_b += 1
        return (
            tot_base_nll / tot_tok,
            math.exp(tot_base_nll / tot_tok),
            100.0 * base_h1 / tot_tok,
            tot_nll / tot_tok,
            math.exp(tot_nll / tot_tok),
            100.0 * h1 / tot_tok,
            rel_sum / max(1, n_b),
            g_sum / max(1, n_b),
        )

    b_nll, b_ppl, b_t1, _, _, _, _, _ = eval_val(shuffled=False)
    print(f"[Baseline Val L22->L23] NLL={b_nll:.5f}, PPL={b_ppl:.3f}, top1={b_t1:.2f}%", flush=True)

    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = max(1, args.epochs * max(1, (n_tr + args.batch_windows - 1) // args.batch_windows))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr * 0.05)

    best_clean_nll = 1e9
    best_clean_state = None
    best_any_nll = b_nll
    best_any_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}

    for ep in range(1, args.epochs + 1):
        bridge.train()
        perm_w = tr_idx[np.random.permutation(n_tr)]
        tr_ce, tr_kl, tr_skl, tr_rel, steps = 0.0, 0.0, 0.0, 0.0, 0
        for i in range(0, n_tr, args.batch_windows):
            idx = perm_w[i : i + args.batch_windows]
            hb = torch.from_numpy(h_w[idx]).to(device=device, dtype=torch.float32)  # [B, S, 1, D]
            blkb = (
                torch.from_numpy(np.array(blocks_w[idx], copy=True)).to(device=device, dtype=torch.float32)
                if blocks_w is not None
                else None
            )
            eb = torch.from_numpy(e_w[idx]).to(device=device, dtype=torch.float32) - mu_e_t
            yb = torch.from_numpy(y_w[idx]).to(device=device)
            valid = yb >= 0

            with torch.no_grad():
                _, base_logits = forward_tail(hb[:, :, 0, :], blkb)
                base_lp = F.log_softmax(base_logits[valid], dim=-1)
                base_p = base_lp.exp()
                base_nll_tok = -base_lp.gather(1, yb[valid].unsqueeze(1)).squeeze(1)
                if args.surprisal_alpha > 0.0:
                    surp = torch.sigmoid((base_nll_tok - args.surprisal_thresh) * 2.0)
                    ce_w = 1.0 + args.surprisal_alpha * surp
                    ce_w = ce_w / ce_w.mean().clamp(min=1e-6)
                    kl_w = 1.0 + args.surprisal_alpha * (1.0 - surp)
                    kl_w = kl_w / kl_w.mean().clamp(min=1e-6)
                else:
                    ce_w = torch.ones_like(base_nll_tok)
                    kl_w = torch.ones_like(base_nll_tok)

            B_cur, S_cur = hb.shape[0], hb.shape[1]
            eb_flat = eb.reshape(-1, meta["d_eng"])
            eb_v = F.linear(eb_flat, W_V_fold) if W_V_fold is not None else eb_flat
            eb_k = F.linear(eb_flat, W_K_fold) if W_K_fold is not None else eb_v
            h_inj, rel_norm, _ = bridge(
                hb.reshape(-1, meta["hc"], meta["d_host"]),
                eb_v,
                eb_k,
                return_stats=True,
            )
            h_inj = h_inj.reshape(B_cur, S_cur, meta["d_host"])
            _, logits = forward_tail(h_inj, blkb)
            lp = F.log_softmax(logits[valid], dim=-1)
            ce_tok = -lp.gather(1, yb[valid].unsqueeze(1)).squeeze(1)
            ce = (ce_tok * ce_w).mean()
            kl_tok = (base_p * (base_lp - lp)).sum(dim=-1)
            kl = (kl_tok * kl_w).mean()
            loss = ce + args.kl_beta * kl + args.delta_reg * (rel_norm ** 2)

            skl_val = 0.0
            if args.shuf_kl_beta > 0.0:
                perm_s = torch.randperm(eb_v.shape[0], device=device)
                h_shuf = bridge(hb.reshape(-1, meta["hc"], meta["d_host"]), eb_v[perm_s], eb_k[perm_s]).reshape(B_cur, S_cur, meta["d_host"])
                _, logits_s = forward_tail(h_shuf, blkb)
                lp_s = F.log_softmax(logits_s[valid], dim=-1)
                skl = (base_p * (base_lp - lp_s)).sum(dim=-1).mean()
                loss = loss + args.shuf_kl_beta * skl
                skl_val = skl.item()

            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            sched.step()

            tr_ce += ce.item()
            tr_kl += kl.item()
            tr_skl += skl_val
            tr_rel += rel_norm.item()
            steps += 1

        _, _, _, r_nll, r_ppl, r_t1, r_rel, r_g = eval_val(shuffled=False)
        _, _, _, s_nll, s_ppl, s_t1, _, _ = eval_val(shuffled=True)
        print(
            f"[Epoch {ep:2d}] tr_CE={tr_ce/steps:.4f} tr_KL={tr_kl/steps:.4f} shuf_KL={tr_skl/steps:.4f} "
            f"||d||/||h||={r_rel:.4f} g={r_g:.4f} | "
            f"val REAL: PPL={r_ppl:.3f} (vs base {r_ppl-b_ppl:+.3f}, top1={r_t1:.2f}%) | "
            f"SHUF: PPL={s_ppl:.3f} (shuf-real={s_ppl-r_ppl:+.3f})",
            flush=True,
        )
        if r_nll < s_nll:
            if r_nll < best_any_nll:
                best_any_nll = r_nll
                best_any_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}
            if s_nll >= b_nll - 0.006 and (s_nll - r_nll) >= 0.01 and r_nll < best_clean_nll:
                best_clean_nll = r_nll
                best_clean_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}

    if best_clean_state is not None:
        bridge.load_state_dict(best_clean_state)
        best_nll = best_clean_nll
    else:
        bridge.load_state_dict(best_any_state)
        best_nll = best_any_nll

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
        "calibration": f"e2e_midlayer_L{meta['target_layer']}_W{n_tr}x{args.win_len}_gb{args.gate_bias:g}_sa{args.surprisal_alpha:g}{pca_tag}",
    }
    write_rkad_v3(
        args.out,
        meta["target_layer"],
        args.export_src_layer if args.export_src_layer >= 0 else args.src_layer,
        mu_e,
        bridge,
        manifest,
        args.delta_clip,
        W_V_fold,
        W_K_fold,
    )
    print(
        f"[MidLayer] Saved E2E frozen-host mid-layer RKAD v3 -> {args.out} "
        f"(best val PPL={math.exp(best_nll):.3f} vs base {b_ppl:.3f}) in {time.time()-t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
