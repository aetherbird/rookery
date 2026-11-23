# rookery

A pure C++20 + CUDA LLM inference and serving engine built for modern MoE and hybrid linear/sparse-attention architectures on heterogeneous multi-GPU setups (Volta `sm_70` + Ampere `sm_86`).

`rookery` loads standard GGUF models directly (including sharded multi-file GGUFs) and exposes a drop-in OpenAI-compatible HTTP/SSE server (`rookery-server`) compatible with `llama-swap` and standard OpenAI clients.

## Supported Architectures

| GGUF Architecture (`general.architecture`) | Model Family | Key Features |
|---|---|---|
| `maxwell-moe` | Maxwell-V1-Flash | mHC-4 (Sinkhorn), MSA sparse attention, NextN MTP speculative decoding, SoA `Q8_0` |
| `glm5-next` / `glm5next` | GLM-5.3-Flash | Hybrid KDA linear attention + MLA full attention, MoE (`Q8_0`/`K-quant`/`IQ`), MTP speculative decoding |
| `inkling2_loop` / `inkling2` | Inkling 2 Mini | Recurrent loop-transformer with SWA FlashAttention (`T >= 8` Tensor Core prefill), `Q8_0` MoE |
| `qwen35moe` | Qwen3.5 / Qwen3.6-35B-A3B-MTP | Hybrid Gated Delta Net (GDN) + GQA (`d=256`), grouped `__dp4a` MoE, NextN MTP speculative decoding, host-RAM GDN checkpoints |
| `qwen4exp` | Qwen3.8-Flash-Next | mHC-4 + host-RAM mmap Per-Layer Embedding (PLE) + QSA sparse indexer (`top_k=2048`) + GDN + `Q6_K`/`Q8_0` MoE |

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

See [bench/baseline.md](bench/baseline.md), [bench/engram_audit_v1.md](bench/engram_audit_v1.md), and [bench/engram_vertical_slice_v2.md](bench/engram_vertical_slice_v2.md) for detailed logs and Engram vertical-slice audits.

## Building

Requirements:
- CMake >= 3.24
- C++20 compiler (GCC 12+ / Clang 16+)
- CUDA Toolkit 12.x (`nvcc`, `cuBLAS`, `cuSOLVER`)
- NVIDIA GPUs with compute capability `>= 7.0` (`sm_70`, `sm_86`, etc.)

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
- `build/rookery_{maxwell,glm5,inkling2,qwen35,qwen4exp,engram}_test` — Model parity & benchmark binaries

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
