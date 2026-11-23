# M0 Baseline Benchmarks (`llama.cpp-maxwell`)

- **Date:** 2026-11-22
- **Oracle binary:** `llama-bench` (`llama.cpp-maxwell`, build `9581154`)
- **Model:** `maxwell-Q8_0-mtp.gguf` (`36.55 GiB`, `36.81B` params)
- **Settings:** `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -r 2`

## Results

| GPU Setup | Split | `pp512` (tok/s) | `pp4096` (tok/s) | `tg128` (tok/s) | Notes |
|---|---|---|---|---|---|
| **RTX 3090 (`sm_86`) + V100 (`sm_70`)** | `24/32` | `2,303.14 ± 21.53` | `2,954.61 ± 2.28` | **`79.07 ± 0.76`** | Matches `llama-swap` config |
| **2× V100-32GB (`sm_70` homogeneous)** | `32/32` | `1,891.02 ± 22.04` | `2,246.85 ± 0.58` | **`65.55 ± 0.07`** | Both GPUs ~900 GB/s HBM2; hit by BF16 emulation + unfused mHC |

## Track 2 Complete — `rookery` v0.1 (`maxwell-moe`) vs `llama.cpp-maxwell` (2026-11-23, commit `91ac903`)

| Benchmark (`maxwell-Q8_0-mtp.gguf`, Q8_0 KV) | `llama.cpp-maxwell` | `rookery` (no MTP) | `rookery` (+ `blk.45` MTP) | Speedup vs `llama.cpp` |
|---|---:|---:|---:|---:|
| **Short ctx `tg128` — RTX 3090 + V100 (`24/32`)** | `79.07 tok/s` | **`98.04 tok/s`** (`+24.0%`) | **`136.31 tok/s`** | **+72.4%** (`1.72×`) |
| **Short ctx `tg128` — 2× V100 (`32/32`, `sm_70`)** | `65.55 tok/s` | **`89.34 tok/s`** (`+36.3%`) | **`128.52 tok/s`** | **+96.1%** (`1.96×`) |
| **Short ctx HTTP sampled (`temp=0.6`, 3090+V100)** | `79.07 tok/s` | **`96.88 tok/s`** (`+22.5%`) | — | **+22.5%** |
| **Short ctx HTTP tool-call (`temp=0`, 3090+V100)** | `79.07 tok/s` | — | **`141.45 tok/s`** | **+78.9%** (`1.79×`) |
| **20K ctx MSA `tg64 @ d20000` — RTX 3090 + V100** | `31.31 tok/s` | **`53.88 tok/s`** (`+72.1%`) | **`84.56 tok/s`** | **+170.1%** (`2.70×`) |

### Progression Log (`3090+V100` / `2× V100`, short ctx `tg128`)
1. First unoptimized draft (`1881796`): `54.18` / `47.11 tok/s`
2. Lossless BF16 unpack + smem Q8_0 GEMV + fused MoE+shared + GPU argmax + CUDA Graphs (`4398119`): `68.67` / `57.76 tok/s`
3. Coalesced 8-thread-per-`BlockQ8_0` (`ef57b7f`): `72.17` / `62.56 tok/s`
4. `nsys`-guided fused 24-block mHC `rmsnorm+bf16_gemv` + 16-lane warp Sinkhorn + 256-thread parallel MoE router + zero-sync warp FlashAttn (`be623b7`): `91.15` / `81.86 tok/s`
5. **Sprint 2A** — Structure-of-Arrays (SoA) Q8_0 weights with 100% coalesced 128-bit `LDG.128` loads, zero extra VRAM (`9d74b7a`): **`98.04` / `89.34 tok/s`**
6. **Sprint 2B** — `blk.45` NextN MTP speculative drafting + `T=2` weight-reuse SoA GEMV + `T=2` CUDA Graph + GPU-event peer sync (`83cd10a`): **`136.31` / `128.52 tok/s`** (exact bit-identical greedy tokens)
7. **Sprint 2C + 2D** — Split-Block MSA indexer (1,024 blocks) + Split-K=16 zero-copy FlashAttention (`53.88` / `84.56 tok/s` @ 20K ctx vs `31.31 tok/s` in `llama.cpp`), O(V) top-K sampler (`96.88 tok/s` sampled HTTP), real-time token-by-token SSE streaming, exact Maxwell BPE tokenizer parity, and XML-leak-safe tool-call parser (`91ac903`).

## Roofline Analysis

- Active weights per decode token ≈ `~3.2 GB` (`~3B` active params @ `Q8_0`).
- Memory bandwidth ceiling on ~900 GB/s HBM2 / 936 GB/s GDDR6X ≈ **`~280 tok/s`**.
- `llama.cpp` achieves only **23%–28%** of memory bandwidth on decode — leaving massive headroom for `rookery`'s `sm_70` `BF16→F16` conversion, fused mHC/Sinkhorn, fused Q8_0 MoE, zero-copy MSA FlashDecoding, and CUDA Graphs.
