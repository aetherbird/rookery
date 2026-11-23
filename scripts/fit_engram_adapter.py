#!/usr/bin/env python3
"""
Closed-form Ridge / SVD low-rank adapter fitter for rookery Decoupled Engram Sidecar.

Reads an "RKPR" v1 calibration pair dump (produced by `rookery_engram_test --dump-pairs out.rkpr`),
solves W_v = argmin || E @ W_v^T - target ||_F^2 + lambda * ||W_v||_F^2 via SVD / Ridge per hook layer,
compresses W_v into low-rank B_v @ A_v (rank r, e.g. 128), and writes an "RKAD" v1 binary adapter
loadable directly by `rookery-server --engram <table.gguf> --engram-adapter <out.rkad>`.
"""
import argparse
import struct
import sys
import torch


def read_rkpr(path: str):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"RKPR", f"Bad magic: {magic}"
        ver, n_hooks, d_engram, d_host = struct.unpack("<4I", f.read(16))
        total_records = struct.unpack("<Q", f.read(8))[0]
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


def fit_ridge_lowrank(E: torch.Tensor, H: torch.Tensor, rank: int, lam: float, device: str):
    E = E.to(device=device, dtype=torch.float32)
    H = H.to(device=device, dtype=torch.float32)
    N, d_e = E.shape
    _, d_h = H.shape

    # Center target slightly? Keep uncentered residual alignment so injected vector matches hidden manifold
    # Solve (E^T E + lam * I) W = E^T H  where W is [d_e, d_h]
    EtE = E.T @ E
    diag_mean = EtE.diagonal().mean().clamp(min=1e-6)
    EtE.diagonal().add_(lam * diag_mean)
    EtH = E.T @ H
    W = torch.linalg.solve(EtE, EtH)  # [d_e, d_h]

    # Compute explained variance R^2 on calibration set
    pred = E @ W
    ss_res = ((H - pred) ** 2).sum().item()
    ss_tot = ((H - H.mean(dim=0, keepdim=True)) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Low-rank SVD factorization of W^T [d_h, d_e] = B_v[d_h, r] @ A_v[r, d_e]
    Wt = W.T  # [d_h, d_e]
    if rank > 0 and rank < min(d_h, d_e):
        U, S, Vh = torch.linalg.svd(Wt, full_matrices=False)
        s_sqrt = torch.sqrt(S[:rank])
        B_v = (U[:, :rank] * s_sqrt.unsqueeze(0)).contiguous().cpu()   # [d_h, rank]
        A_v = (s_sqrt.unsqueeze(1) * Vh[:rank, :]).contiguous().cpu()  # [rank, d_e]
        pred_lr = (E @ A_v.to(device).T) @ B_v.to(device).T
        r2_lr = 1.0 - ((H - pred_lr) ** 2).sum().item() / max(ss_tot, 1e-12)
        return A_v, B_v, r2, r2_lr
    return Wt.contiguous().cpu(), None, r2, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="Input .rkpr calibration pair dump")
    ap.add_argument("--out", required=True, help="Output .rkad adapter binary")
    ap.add_argument("--rank", type=int, default=128, help="Low-rank bottleneck (0 = full dense)")
    ap.add_argument("--lam", type=float, default=1e-2, help="Ridge regularization factor")
    ap.add_argument("--scale", type=float, default=0.05, help="Default residual injection scale per hook")
    ap.add_argument("--gated", action="store_true", help="Also emit cosine-RMSNorm key projection W_k = W_v")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ver, n_hooks, d_engram, d_host, hook_meta, Es, Hs = read_rkpr(args.pairs)
    print(f"Loaded {args.pairs}: {n_hooks} hooks, N={Es[0].shape[0]} tokens, d_engram={d_engram} -> d_host={d_host} (device={args.device})")

    with open(args.out, "wb") as f:
        f.write(b"RKAD")
        f.write(struct.pack("<6I", 1, n_hooks, d_engram, d_host, args.rank, 1 if args.gated else 0))

        for hi in range(n_hooks):
            target_layer, src_layer = hook_meta[hi]
            A_v, B_v, r2, r2_lr = fit_ridge_lowrank(Es[hi], Hs[hi], args.rank, args.lam, args.device)
            print(f"  hook[{hi}] src={src_layer} -> host_layer={target_layer}: Ridge R^2={r2:.4f}, rank-{args.rank} R^2={r2_lr:.4f}")

            f.write(struct.pack("<2if", target_layer, src_layer, args.scale))
            f.write(A_v.numpy().tobytes())
            if B_v is not None:
                f.write(B_v.numpy().tobytes())
            if args.gated:
                # Initialize key projection aligned to value manifold & unit gate weights across hc=4 streams
                f.write(A_v.numpy().tobytes())
                if B_v is not None:
                    f.write(B_v.numpy().tobytes())
                gate_w = torch.ones((4, d_host), dtype=torch.float32)
                f.write(gate_w.numpy().tobytes())

    print(f"Wrote RKAD v1 adapter -> {args.out}")


if __name__ == "__main__":
    main()
