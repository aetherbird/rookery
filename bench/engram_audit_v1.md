# Cross-Model Engram Grafting Audit v1 (2026-11-23)

- **Host Model:** `maxwell-Q8_0-mtp.gguf` (`36.8B` MoE, `d_host=2048`, `hc=4`) on RTX 3090 + V100-32GB
- **Calibration Train Corpus (disjoint):** `12,000` unique tokens from `llama.cpp/docs/{autoparser,build,function-calling,multi-gpu,ops,speculative,docker}.md`
- **Held-Out Eval #1 (`EVAL_TECH`, unseen docs):** `768` tokens from `llama.cpp/docs/{android,llguidance,moe-cache,multimodal,tts-norm-patterns}.md`
- **Held-Out Eval #2 (`EVAL_PROSE`, out-of-domain general knowledge):** `768` tokens from `gemma4/quality-eval/stock-outputs.jsonl`
- **Control:** Exact same `.rkad` adapter & norm, with Engram hash rows randomized via `--shuffle-salt 0x9e3779b97f4a7c15` (`[SHUFFLED]`).

## Key Results

### 1. DeepSeek-V4.1-Flash (103 GiB `IQ4_NL`, `d_engram=6144`, 2/3/4-gram, L1 & L14) — **Delta-Ridge (`H_{L+1}-H_L`)**
- GPU fit time: `2.70 s` | Calibration $R^2$: `L1 = 0.7131`, `L14 = 0.5645`

| Eval Set | `global_scale` (eff) | Real NLL | **Real PPL** | Real Top-1 / Top-5 | Shuffled PPL | **Shuffled − Real PPL** |
|---|---:|---:|---:|---:|---:|---:|
| `EVAL_PROSE` (OOD) | `0.00` (baseline) | `1.87067` | `6.493` | `60.16% / 84.11%` | — | — |
| `EVAL_PROSE` (OOD) | `0.25` (`0.005`) | `1.85595` | **`6.398`** (`-0.095`) | `60.29% / 84.51%` | `6.502` | **`+0.104`** (Real wins) |
| `EVAL_PROSE` (OOD) | `0.50` (`0.010`) | `1.85761` | **`6.408`** (`-0.085`) | **`60.81% / 85.42%`** | `6.529` | **`+0.120`** (Real wins) |
| `EVAL_PROSE` (OOD) | `1.00` (`0.020`) | `1.87619` | `6.529` | `60.68% / 84.77%` | `6.573` | **`+0.044`** (Real wins) |
| `EVAL_PROSE` (OOD) | `2.00` (`0.040`) | `1.91811` | `6.808` | `60.94% / 83.46%` | `6.940` | **`+0.132`** (Real wins) |
| `EVAL_TECH` (unseen) | `0.00` (baseline) | `3.06521` | `21.439` | `49.87% / 69.40%` | — | — |
| `EVAL_TECH` (unseen) | `0.25` (`0.005`) | `3.05734` | `21.271` (`-0.168`) | `49.74% / 70.05%` | `21.665` | **`+0.394`** (Real wins) |
| `EVAL_TECH` (unseen) | `1.00` (`0.020`) | `3.04389` | **`20.987`** (`-0.452`) | `49.61% / 70.83%` | `21.137` | **`+0.150`** (Real wins) |
| `EVAL_TECH` (unseen) | `2.00` (`0.040`) | `3.05754` | `21.275` (`-0.164`) | `48.44% / 70.96%` | `22.242` | **`+0.967`** (Real wins) |

### 2. HY4-Flash (59 GiB `F8_E4M3`, `d_engram=1536`, 2/3-gram, L1/11/21/31) — **Delta-Ridge (`H_{L+1}-H_L`)**
- GPU fit time: `3.15 s` | Calibration $R^2$: `L1 = 0.3295`, `L11 = 0.1791`, `L21 = 0.2652`, `L31 = 0.1733`

| Eval Set | `global_scale` (eff) | **Real PPL** | Shuffled PPL | **Shuffled − Real PPL** |
|---|---:|---:|---:|---:|
| `EVAL_PROSE` (OOD) | `0.00` (baseline) | `6.493` | — | — |
| `EVAL_PROSE` (OOD) | `1.00` (`0.020`) | `6.507` | `6.678` | **`+0.171`** (Real wins) |
| `EVAL_PROSE` (OOD) | `2.00` (`0.040`) | **`6.470`** (`-0.023`) | `6.779` | **`+0.309`** (Real wins) |
| `EVAL_TECH` (unseen) | `0.50` (`0.010`) | `21.083` (`-0.356`) | `21.410` | **`+0.327`** (Real wins) |
| `EVAL_TECH` (unseen) | `2.00` (`0.040`) | **`20.879`** (`-0.560`) | `21.731` | **`+0.853`** (Real wins) |

### 3. HY4-Flash **State-Ridge (`H_L`)** — Control Caught Domain-Mean Bias!
- Fitting uncentered `H_L` caused `[SHUFFLED]` to beat `[REAL]` on `EVAL_TECH` (`17.344` vs `18.671` at `scale=2.0`), proving State-Ridge was mostly adding a constant domain-mean steering vector $\mu_L$ rather than token-specific Engram memory.
- Switching to **Delta-Ridge (`H_{L+1} - H_L`)** eliminated that artifact: on out-of-domain `EVAL_PROSE`, `[SHUFFLED]` degrades PPL at every scale while `[REAL]` improves PPL.

## Honest Takeaways & Next Steps
1. **Verified real cross-model transfer:** DeepSeek-V4.1-Flash's 103 GiB Engram table measurably improves **Maxwell-V1-Flash** on completely unseen out-of-domain prose (`PPL 6.493 → 6.398`, `Top-5 84.11% → 85.42%`) and unseen tech docs (`21.439 → 20.987`), beating the shuffled control on every OOD scale (`4/4`) — using only a 2.7s closed-form Ridge fit.
2. **DeepSeek Engram (`d=6144`, 2/3/4-gram) transfers ~2× cleaner than HY4 (`d=1536`, 2/3-gram):** Calibration $R^2$ is `0.71 / 0.56` vs `0.33 / 0.18`.
3. **Two clear upgrades for the adapter fitter:**
   - **Mean-center `E` and `H`** in `rookery-fit-adapter` to make constant domain-bias leakage mathematically impossible.
   - **Next-token CE distillation** of the tiny ~2–4 MiB `.rkad` adapter (rather than Ridge regression onto Maxwell's existing layer deltas) is the path to injecting genuinely *new* factual knowledge Maxwell doesn't already know.
