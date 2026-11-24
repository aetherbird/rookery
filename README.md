# rookery

A pure C++20 + CUDA LLM inference and serving engine built for modern MoE and hybrid linear/sparse-attention architectures on heterogeneous multi-GPU setups (Maxwell `sm_50`/`sm_52`, Pascal `sm_60`/`sm_61`, Volta `sm_70`, and Ampere `sm_86`).

`rookery` loads standard GGUF models directly (including sharded multi-file GGUFs) and exposes a drop-in OpenAI-compatible HTTP/SSE server (`rookery-server`) compatible with `llama-swap` and standard OpenAI clients.

## Supported Architectures

| GGUF Architecture (`general.architecture`) | Model Family | Key Features |
|---|---|---|
| `maxwell-moe` | Maxwell-V1-Flash | mHC-4 (Sinkhorn), MSA sparse attention, NextN MTP speculative decoding, SoA `Q8_0` |
| `glm5-next` / `glm5next` | GLM-5.3-Flash | Hybrid KDA linear attention + MLA full attention, MoE (`Q8_0`/`K-quant`/`IQ`), MTP speculative decoding |
| `inkling2_loop` / `inkling2` | Inkling 2 Mini | Recurrent loop-transformer with SWA FlashAttention (`T >= 8` Tensor Core prefill), `Q8_0` MoE |
| `qwen35moe` | Qwen3.5 / Qwen3.6-35B-A3B-MTP | Hybrid Gated Delta Net (GDN) + GQA (`d=256`), grouped `__dp4a` MoE, NextN MTP speculative decoding, host-RAM GDN checkpoints |
| `qwen4exp` | Qwen3.8-Flash-Next | mHC-4 + host-RAM mmap Per-Layer Embedding (PLE) + QSA sparse indexer (`top_k=2048`) + GDN + `Q6_K`/`Q8_0` MoE |
| `bailingmoe3` | Ling-3.0-tiny / Ling-3.0-Flash / Ling-3.2-Air (incl. REAP-408) | Hybrid KDA linear attention + Gated RoPE-576 MLA (`P_TILE=8` zero-bank-conflict Split-K + Tensor Core FlashMLA), grouped top-k `Q6_K`/`Q8_0` SoA MoE with SwiGLU clamping, NextN MTP speculative decoding, host-RAM KDA checkpoints |

Additional engine features:
- **Heterogeneous multi-GPU pipeline execution:** Automatically discovers GPU topology (`sm_70` + `sm_86`), supports `--tensor-split` / `--gpus`, and converts `BF16` weights to `FP16` on Volta (`sm_70`) while keeping native `BF16` on Ampere (`sm_86`).
- **Structure-of-Arrays (SoA) quantized weights:** Coalesced 128-bit `LDG.128` loads with native `__dp4a` INT8 dot products and `cuBLAS` FP16 Tensor Core batched prefill.
- **Static CUDA Graphs:** Zero-launch-overhead `T=1` and `T=2` (MTP) decode graphs with GPU-event peer synchronization.
- **Prefix caching & recurrent checkpoints:** Automatic KV-cache reuse plus periodic host-RAM recurrent/conv state checkpoints for GDN/KDA/PLE/loop models (including multi-turn `<think>`-stripped conversations).
- **OpenAI-compatible server (`rookery-server`):** `/health`, `/v1/models`, `/tokenize`, `/v1/completions`, `/v1/chat/completions` with real-time UTF-8-safe SSE streaming, `reasoning_content` separation, XML/JSON `tool_calls` parsing, `stop` strings, and client-disconnect cancellation.
- **Cross-Model Engram Sidecar (`--engram`, `--engram-adapter`):** Optional host-RAM `mmap` n-gram Engram table grafting across heterogeneous tokenizers with closed-form GPU Ridge adapter fitting (`rookery-fit-adapter`).

## Benchmarks

Measured against `llama.cpp` (`-fa 1 -ctk q8_0 -ctv q8_0`) on heterogeneous multi-GPU hardware (`RTX 3090 24GB`, `RTX 3060 12GB`, `Tesla V100-PCIE-32GB`):

| Model | GPU Setup | Metric | `llama.cpp` | `rookery` | Speedup |
|---|---|---|---:|---:|---:|
| **Maxwell-V1-Flash** (`Q8_0`, MTP) | 3090 + V100 | Decode `tg128` (no MTP / + MTP) | `79.1 tok/s` | `98.0` / **`136.3 tok/s`** | **+24% / +72%** |
| **Maxwell-V1-Flash** (`Q8_0`, MTP) | 2× V100 (`sm_70`) | Decode `tg128` (no MTP / + MTP) | `65.6 tok/s` | `89.3` / **`128.5 tok/s`** | **+36% / +96%** |
| **Maxwell-V1-Flash** (`Q8_0`, MTP) | 3090 + V100 | 20K ctx decode (no MTP / + MTP) | `31.3 tok/s` | `53.9` / **`84.6 tok/s`** | **+72% / +170%** |
| **Qwen3.6-35B-A3B-MTP** (`Q8_0`) | 3090 + V100 | Decode (no MTP / + MTP) | `113.2 tok/s` | `134.0` / **`195.6 tok/s`** | **+18% / +73%** |
| **Qwen3.6-35B-A3B-MTP** (`Q8_0`) | 3090 + V100 | Prefill `pp2048` | — | **`1,887.4 tok/s`** | `0.000000` ckpt parity |
| **Qwen3.8-Flash-Next** (`UD-Q6_K_XL`) | 3090 + 2×3060 + 3×V100 | Decode (`ctx=5` / `ctx=4096`) | `41.6 tok/s` | **`50.5` / `50.1 tok/s`** | **+21%** |
| **Ling-3.2-Air** (`Q6_K`, `256K` ctx, MTP) | 3090 + 3×V100 | Decode `tg128` (no MTP / + MTP) | `55.4 tok/s` | `97.7` / **`118.2–124.7 tok/s`** | **+76% / +113%–125%** |
| **Ling-3.2-Air** (`Q6_K`, `256K` ctx, MTP) | 3090 + 3×V100 | Prefill `pp2048` / `pp8192` | `579.2 tok/s` | **`1,556.8` / `1,479.6 tok/s`** (`1,609.9` no MTP) | **2.69×–2.78×** |
| **Ling-3.2-Air** (`Q6_K`, `256K` ctx, MTP) | 3090 + 3×V100 | Long-ctx decode (`16K` / `32K` / `64K`) | `50.1 tok/s` (`32K`) | **`88.6` / `67.5` / `53.5 tok/s`** (100% needle up to `82K`) | **+35% at 32K** |
| **Ling-3.2-Air REAP-408** (`Q8_0`, `256K` ctx) | 3090 + 3×V100 | Prefill `pp2048` / Decode `tg128` (no MTP) | — | **`1,826.4` / `93.9 tok/s`** | **pure `Q8_0` SoA** |
| **Ling-3.0-tiny** (`Q4_K_M`) | RTX 3090 (`sm_86`) | Prefill `pp128` / `pp512` | `3,809.9` / `9,174.8 tok/s` | **`6,555.2` / `9,471.4 tok/s`** | **+72% / +3%** |
| **Ling-3.0-tiny** (`Q4_K_M`) | RTX 3090 (`sm_86`) | Decode `tg128` (`d=0` / `4K` / `16K`) | `241.4` / `239.2` / `220.8 tok/s` | **`383.9` / `355.2` / `294.0 tok/s`** | **+59% / +49% / +33%** |
| **Qwen3.6-35B-A3B** (`UD-Q3_K_XL`) | P100 + GTX 1060 (`sm_60/61`) | Decode `tg128` | `38.0 tok/s` | **`53.4 tok/s`** | **+41%** |
| **Maxwell-V1-Flash** (`Q3_K_M`) | P100 + GTX 1060 (`sm_60/61`) | Decode `tg64` | Unsupported | **`26.3 tok/s`** | — |
| **Inkling-2-Mini** (`IQ4_NL`, `36K` ctx) | P100 + GTX 1060 (`sm_60/61`) | Prefill `pp512` / `pp8192` | Unsupported | **`543.7` / `504.7 tok/s`** | **7.87× vs baseline** |
| **Inkling-2-Mini** (`IQ4_NL`, `36K` ctx) | P100 + GTX 1060 (`sm_60/61`) | Decode `tg128` (`d=0` / `16K` / `32K`) | Unsupported | **`64.0` / `54.6` / `49.0 tok/s`** | **2.31×–2.57× vs baseline** |

See [bench/baseline.md](bench/baseline.md), [bench/engram_audit_v1.md](bench/engram_audit_v1.md), and [bench/engram_vertical_slice_v2.md](bench/engram_vertical_slice_v2.md) for detailed logs and Engram vertical-slice audits.

## Building

Requirements:
- CMake >= 3.24
- C++20 compiler (GCC 12+ / Clang 16+)
- CUDA Toolkit 12.x (`nvcc`, `cuBLAS`, `cuSOLVER`)
- NVIDIA GPUs with compute capability `>= 5.0` (default build targets `sm_70;sm_86`; override with `-DCMAKE_CUDA_ARCHITECTURES="50;52;60;61"` for Maxwell/Pascal)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
# Override target CUDA architectures if needed (default: "70;86"):
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
cmake --build build -j$(nproc)
```

This builds:
- `build/rookery-server` — OpenAI-compatible inference server
- `build/rookery-fit-adapter` — GPU Ridge adapter fitter for cross-model Engram grafting
- `build/rookery_unit_test` — Chat template, tokenizer, and tool-parser unit tests
- `build/rookery_{maxwell,glm5,inkling2,qwen35,qwen4exp,bailingmoe3,engram}_test` — Model parity & benchmark binaries

## Usage

### Running `rookery-server`

```bash
./build/rookery-server \
  -m /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --alias my-model \
  -c 65536 \
  --tensor-split 24,32 \
  -ctk q8_0 -ctv q8_0
```

Common flags:
- `-m, --model <path>` — Path to GGUF file (or first shard `-00001-of-NNNNN.gguf`)
- `--host <addr>` / `--port <port>` — Bind address and port (default `127.0.0.1:8080`)
- `--alias <name>` — Model ID reported by `/v1/models`
- `-c, --ctx-size <n>` — Maximum context length (default `32768`)
- `--gpus <0,1,...>` / `-ts, --tensor-split <w0,w1,...>` — Explicit GPU selection and layer split weights
- `-ctk, --cache-type-k <q8_0|f16>` — KV cache quantization (default `q8_0`)
- `--mtp` / `--no-mtp` — Enable/disable NextN Multi-Token Prediction speculative decoding (default on when present)
- `--temp`, `--top-p`, `--top-k` — Default sampling parameters
- `--engram <table.gguf>`, `--engram-adapter <adapter.rkad>`, `--engram-scale <f>` — Cross-model Engram sidecar options

`rookery-server` also accepts common `llama-server` compatibility flags (`-ngl`, `-fa`, `--jinja`, `--reasoning-format`, etc.) for drop-in `llama-swap` configuration.

### Cross-Model Engram Sidecar (End-to-End)

`rookery` supports zero-VRAM host-RAM `mmap` Engram/PLE grafting across decoupled providers (`deepseek41`, `qwen4exp`, `hy_v4_flash`, `unigram_ple`) and all 5 host model architectures (`maxwell-moe`, `qwen35moe`, `glm5_next`, `qwen4exp`, `inkling2_loop`) using strict self-describing `RKAD v3` manifests (`provider_fingerprint`, host architecture/layer/stream contract checks), an incremental boundary-anchored BPE commit cache (`100.00%` causal parity, `~7.9 µs/tok`), and sub-1% warm-RAM decode overhead (`+0.90%` decode / `+0.49%` prefill on `DeepSeek-V4.1-Flash` 103 GiB → `Maxwell-V1-Flash`).

```bash
# 1. Inspect a provider table, host GGUF, or .rkad adapter
python3 scripts/engram_pipeline.py inspect \
  --engram /path/to/DeepSeek-V4.1-Flash-00005-of-00005.gguf \
  --model /path/to/host-model.gguf

# 2. End-to-end automated pipeline:
#    inspect -> 100% addressing gate -> dump -> RKAD v3 calibrate/train -> 4-way quality gate -> register .json card
python3 scripts/engram_pipeline.py build \
  --model /path/to/host-model.gguf \
  --engram /path/to/engram-table.gguf \
  --mode level2 \
  --corpus /path/to/train_corpus.txt \
  --eval-tech /path/to/eval_tech.txt \
  --eval-prose /path/to/eval_prose.txt \
  --out /path/to/adapter_v3.rkad

# (For Level-3 temp-calibrated contrastive gated training, pass --mode level3 --max-tokens 200000 --base-scale 0.05)

# 3. Optional N + M canonical Z composition (`engram-z-v1`, zero extra runtime overhead):
#    Fit provider exporters (.rkex) and host importers (.rkim) independently, then fold any pair into RKAD v3
python3 scripts/engram_z_bridge.py compose \
  --exporter /path/to/exporter_provider.rkex \
  --importer /path/to/importer_host.rkim \
  --out /path/to/composed_zeroshot_v3.rkad

# 4. Serve with strict RKAD v3 contract validation & zero-recapture per-request runtime controls
./build/rookery-server -m /path/to/host-model.gguf --gpus 0,1 -ts 20,28 \
  --engram /path/to/engram-table.gguf \
  --engram-adapter /path/to/adapter_v3.rkad \
  --engram-scale 1.0
```

Per-request JSON fields in `/v1/completions` and `/v1/chat/completions` override Engram behavior dynamically without recapturing CUDA Graphs:
- `"engram_scale"` (float, `0.0` = exact bit-identical disable)
- `"engram_gate_bias"` (float, additive logit bias inside sigmoid gate)
- `"engram_delta_clip"` (float, caps `||delta||_rms / ||h||_rms`)

### Testing

```bash
# Fast unit tests (no model required)
./build/rookery_unit_test

# Full end-to-end HTTP/SSE/tool-call/prefix-cache/long-context acceptance suite
python3 tests/test_server_e2e.py --url http://127.0.0.1:8080 --model my-model --contexts 16 32
```

## Project Layout

- `src/core/` — CUDA device topology discovery, memory helpers, and core types
- `src/gguf/` — Zero-dependency GGUF v2/v3 parser, multi-shard reader, and SoA weight loader
- `src/kernels/` — Custom CUDA kernels (`Q8_0`/`K-quant`/`IQ` GEMV, grouped `__dp4a` MoE, FlashAttention, GDN/KDA, MLA, mHC Sinkhorn, QSA/MSA sparse attention)
- `src/models/` — Model forward/decode/MTP/checkpoint implementations
- `src/tokenizer/` — Exact BPE tokenizer with Llama3/Qwen3/GLM/Maxwell pre-tokenization
- `src/server/` — Chat templates and XML/JSON tool-call & reasoning parsers
- `src/engram/` — Cross-model host-RAM Engram sidecar and CUDA injection kernels
- `tools/` & `scripts/` — Engram adapter fitting tools
- `tests/` — Unit, forward parity, and end-to-end HTTP acceptance tests
- `bench/` — Reproducible benchmarks and Engram audit reports

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option. See `LICENSE-APACHE` and `LICENSE-MIT`.

Third-party libraries under `third_party/` (`cpp-httplib` and `nlohmann/json`) are licensed under their respective MIT licenses.
