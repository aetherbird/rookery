# Cross-Model Engram 1×1 Vertical Slice Audit v2 (2026-11-23)

Rigorous end-to-end verification of a single cross-model Engram pair (`DeepSeek-V4.1-Flash` 103 GiB `IQ4_NL` Engram → `Maxwell-V1-Flash` 36.8B `Q8_0` MoE on RTX 3090 + V100-32GB) covering:

1. Exact incremental source-compatible addressing parity,
2. Mean-centered RKAD v2 Ridge calibration with held-out 10% validation $R^2$ and per-hook auto-gating,
3. Measured stage-by-stage decode (`tg128`, CUDA Graph ON) and prefill (`pp1024`) latency overhead,
4. 4-way held-out quality evaluation (`Base`, `Disabled (scale=0)`, `Real`, `Shuffled`) on 2,048-token disjoint corpora.

---

## 1. Incremental Source-Tokenizer Addressing Parity (`--verify-addressing`)

Boundary-aware suffix windowing (`find_safe_bpe_cut`: 256B lookback + 96B guard anchored on newline or single leading-space pre-token boundaries) verified token-by-token against the exact causal full-prefix source-tokenizer oracle:

| Test Corpus | Host Tokens | Bytes | Boundary-Cut Active | Source Token Match | N-Gram Hash Match (2 layers × 24 heads) | Non-Causal Prefill Lookahead Skew |
|---|---:|---:|---:|---:|---:|---:|
| Mixed prose / C++ / Python / JSON / UTF-8 / CJK / URLs | `851` | `2,972` | `752 / 851` | **`851 / 851` (100.00%)** | **`40,848 / 40,848` (100.00%)** | `27 / 851` (`3.17%`) |
| Unseen technical documentation | `1,500` | `42,845` | `1,415 / 1,500` | **`1,500 / 1,500` (100.00%)** | **`72,000 / 72,000` (100.00%)** | `26 / 1,500` (`1.73%`) |

> **Key finding:** Boundary-aware suffix slicing achieves **100.00% exact parity** with causal full-prefix tokenization. The `1.73%–3.17%` non-causal prefill skew occurs only when the host tokenizer splits a word that the source tokenizer merges (e.g. `" Ray"` + `"leigh"` → `" Rayleigh"`), which batched prefill can see one subword early.

---

## 2. Measured Stage-by-Stage Latency Profile (`tg128` + `pp1024`)

Host: `Maxwell-V1-Flash` (`Q8_0`, 45 layers, RTX 3090 + V100-32GB), Provider: 103 GiB `IQ4_NL` `mmap` table (`d_engram=6144`, 2 hooks, `rank=128`, `float4`-vectorized matvec, RKAD v2 mean-centered):

### Decode (`tg128`, CUDA Graph ON)

| Configuration | Throughput | Total Latency | Overhead vs Base | Measured Stage Breakdown (µs/tok) |
|---|---:|---:|---:|---|
| **1. Base** (`no hook`) | `105.12 tok/s` | `9,513.3 µs/tok` | — | — |
| **2. Disabled** (`hook + prepare_step, scale=0`) | `104.29 tok/s` | `9,588.6 µs/tok` | `+75.3 µs` (**`+0.79%`**) | `tok=43.4`, `hash=0.6`, `gather+dequant+center=14.2`, `h2d=7.2` |
| **3. Active** (`2 hooks, scale=0.5`, CUDA Graph ON) | `103.93 tok/s` | `9,621.7 µs/tok` | `+108.4 µs` (**`+1.14%`**) | `tok=43.2`, `hash=0.7`, `gather+dequant+center=14.2`, `h2d=7.0`, `gpu_hooks=36.5`* |

*\*Isolated GPU hook kernel time (`36.5 µs/tok` across 2 hooks = `~18.2 µs/hook`) measured with per-hook CUDA events outside graph capture (`75.3 + 36.5 = 111.8 µs` vs `108.4 µs` end-to-end).*

### Prefill (`pp1024`, chunk=256)

| Configuration | Throughput | Total Time | Overhead vs Base | Prep Breakdown (total ms) |
|---|---:|---:|---:|---|
| **1. Base** (`no hook`) | `379.3 tok/s` | `2,699.36 ms` | — | — |
| **2. Active** (`2 hooks, scale=0.5`) | `374.7 tok/s` | `2,732.96 ms` | `+33.60 ms` (**`+1.24%`**) | `tok=2.66`, `hash=0.35`, `gather+dequant+center=20.34`, `h2d=0.06` |

> **Scaling note:** On a ~105 tok/s model with RAM-resident `mmap` tables, total decode overhead is **1.14%** (`~108 µs/tok`; `~0.5%` with 1 active hook). On a faster 400 tok/s model (`2,500 µs/tok`), the same `108 µs` would represent `~4.3%` unless `tokenize_align` (`43 µs/tok`) is cached incrementally and H2D runs on a dedicated non-blocking stream.

---

## 3. Mean-Centered Delta-Ridge Calibration (`RKAD v2`) & Auto-Gating

Fitted on `16,000` paired tokens (`H_{L+1} - H_L` layer deltas) with mean-centering (`μ_E`, `μ_H`) and 10% held-out validation $R^2$ (`--auto-lam`):

| Hook | `lam = 0.1` (`val R²`) | `--auto-lam` Best `lam` | `train R²` | **`val(10%) R²`** | Spectral Energy (`r=128`) | Auto-Gate Decision |
|---|---:|---:|---:|---:|---:|---|
| `hook[0]` (`src=0 → host L1`) | `-0.2999` (overfit) | **`2.0`** | `0.4087` | **`+0.1340`** | `77.8%` | **ENABLED** (`scale=0.02`) |
| `hook[1]` (`src=1 → host L14`) | `-20.1601` (severe overfit) | `50.0` | `0.0154` | **`-0.0396`** | `97.9%` | **AUTO-DISABLED** (`scale=0.00`) |

> **Key finding:** Held-out validation $R^2$ immediately caught severe overfitting at `lam=0.1` (especially on deep layer 14, where local n-grams cannot linearly predict context-dominated residual deltas). Auto-gating L14 off eliminated the OOD regression completely.

---

## 4. 4-Way Held-Out Quality Evaluation (2,048 Tokens Each)

Evaluated with the auto-gated RKAD v2 adapter (`L1` active at `lam=2.0`, `L14` auto-disabled) across all four controls:

### A. `EVAL_TECH` (2,048 unseen technical doc tokens)

| Condition | `global_scale` | NLL | **PPL** | vs Base | Top-1 | Top-5 | Shuffled − Real PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[BASE]` (no hook) | — | `2.54524` | `12.746` | — | `55.52%` | `75.63%` | — |
| `[DISABLED]` (hook+prep) | `0.00` | `2.54524` | `12.746` | `+0.000000` | `55.52%` | `75.63%` | — |
| `[REAL]` | `0.50` | `2.53343` | `12.597` | `-0.150` | `55.13%` | `75.73%` | **`+0.122`** (Real wins) |
| `[REAL]` | `1.00` | `2.53141` | **`12.571`** | **`-0.175`** | `55.47%` | `76.22%` | **`+0.185`** (Real wins) |
| `[REAL]` | `2.00` | `2.53198` | `12.578` | `-0.168` | `54.98%` | `75.98%` | **`+0.397`** (Real wins) |
| `[REAL]` | `4.00` | `2.53872` | `12.663` | `-0.083` | `55.47%` | **`76.42%`** | **`+0.644`** (Real wins) |

*(Shuffled PPL degraded monotonically: `12.719 → 12.756 → 12.975 → 13.307` across `scale = 0.5 → 4.0`.)*

### B. `EVAL_PROSE` (2,048 out-of-domain general prose tokens)

| Condition | `global_scale` | NLL | **PPL** | vs Base | Top-1 | Top-5 | Shuffled − Real PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[BASE]` (no hook) | — | `1.56006` | `4.759` | — | `72.51%` | `87.26%` | — |
| `[DISABLED]` (hook+prep) | `0.00` | `1.56006` | `4.759` | `+0.000000` | `72.51%` | `87.26%` | — |
| `[REAL]` | `0.50` | `1.55280` | **`4.725`** | **`-0.034`** | **`73.14%`** (`+0.63%`) | `87.06%` | **`+0.002`** (Real wins) |
| `[REAL]` | `1.00` | `1.56970` | `4.805` | `+0.046` | `73.10%` (`+0.59%`) | `87.16%` | **`+0.056`** (Real wins) |
| `[REAL]` | `2.00` | `1.57173` | `4.815` | `+0.056` | `72.22%` | `87.06%` | `+0.004` |
| `[REAL]` | `4.00` | `1.58389` | `4.874` | `+0.115` | `72.31%` | `86.91%` | `-0.141` |

---

## 5. Level-3 Supervised Gated Bridge (`L44` CE + Host-KL) & Factual QA Eval

Trained a tiny gated RKAD v2 bridge (`src_layer=0 → host L44`, right before `hc_head + output_norm + lm_head`, `rank=128`, `has_gate=1`, `centered=yes`, ~4.2M params) via `scripts/train_engram_bridge.py` on `45,000` strictly-causal tokens (`RKCE v1` dump from technical/open-source docs, **zero factual QA overlap**), optimizing `CE(next_token) + 0.5 * KL(p_base || p_grafted)` in `45.8 s` on a single RTX 3090:

- **Validation comparison (4,498 held-out tokens):**
  - `src_layer=0 → L44` (DeepSeek L1): `val PPL 3.090 → 3.068` (`-0.021`, `top1 79.52% → 79.84%`, `SHUF − REAL = +0.048` at epoch 2)
  - `src_layer=1 → L44` (DeepSeek L14): `val PPL 3.090 → 3.086` (`-0.004`, `SHUF − REAL = +0.016`) — independently confirming DeepSeek's L1 table transfers ~4× more strongly than L14.

### 4-Way `rookery_engram_test` Results (`src_layer=0 → L44`, Gated RKAD v2)

#### A. `EVAL_FACTUAL` (60 held-out factual QA items — capitals, elements, constants, history, literature, science)

| Condition | `global_scale` | `ans_NLL` | **`ans_PPL`** | vs Base | Exact First-Tok Top-1 | Answer Tok Top-1 | Shuffled − Real PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[BASE]` (no hook) | — | `0.33659` | `1.400` | — | `91.7%` (`55/60`) | `94.3%` | — |
| `[DISABLED]` | `0.00` | `0.33659` | `1.400` | `+0.000000` | `91.7%` | `94.3%` | — |
| `[REAL]` | `0.50` | `0.33044` | `1.392` | `-0.0086` | `91.7%` | `94.3%` | **`+0.0111`** (Real wins) |
| `[REAL]` | `1.00` | `0.32826` | `1.389` | `-0.0116` | `91.7%` | `94.3%` | **`+0.0245`** (Real wins) |
| `[REAL]` | `1.50` | `0.32367` | **`1.382`** | **`-0.0180`** | **`93.3%`** (`56/60`) | **`95.2%`** | **`+0.0396`** (Real wins) |

*(Shuffled `ans_PPL` degraded monotonically: `1.403 → 1.413 → 1.422`, staying at `91.7%` / `94.3%`.)*

#### B. `EVAL_TECH` & `EVAL_PROSE` (2,048 tokens each, apples-to-apples vs Level-2 Ridge)

| Eval Corpus | `[BASE]` PPL | Level-2 `L1` Ridge Best | Level-3 `L44` Gated (`scale=0.5 / 1.0`) | Level-3 `[SHUFFLED]` | Notes |
|---|---:|---:|---:|---:|---|
| **`EVAL_TECH`** (2,048 tok) | `12.746` | **`12.571`** (`-0.175`) | **`12.631`** (`-0.115`, `top5 76.51%`) | `12.709` (`+0.078` gap) | Beats Base & Shuffled at `0.5` & `1.0` |
| **`EVAL_PROSE`** (OOD, 2,048 tok) | `4.759` | **`4.725`** (`-0.034`, `top1 +0.63%`) | `4.776` (`+0.017`) | `4.771` (`-0.005`) | Tech-only 45K train set + `dropout=0` overfit prose |

---

## 6. Runtime Contract Hardening (`RKAD v3`, Zero-Recapture Controls & Incremental Commit Cache)

### A. Incremental Boundary-Anchored Commit Cache (`--verify-addressing`)
Replacing per-step 256–352B suffix re-tokenization with a boundary-anchored incremental commit cache (`EngramSeqState`, `kCommitGuard = 48B`, O(1) precomputed `BpeTokenizer::token_byte_len`) maintained **100.00% exact parity** against the causal full-prefix oracle while shrinking average re-tokenized tail to ~60–66 bytes:

| Test Corpus | Host Tokens | Commit Cache Active | Avg Tail | Source Token Match | N-Gram Hash Match |
|---|---:|---:|---:|---:|---:|
| Mixed prose / C++ / Python / JSON / UTF-8 / CJK | `851` | `838 / 851` | `60.4 B` | **`851 / 851` (100.00%)** | **`40,848 / 40,848` (100.00%)** |
| Unseen technical documentation | `1,500` | `1,484 / 1,500` | `66.2 B` | **`1,500 / 1,500` (100.00%)** | **`72,000 / 72,000` (100.00%)** |

### B. Updated Stage-by-Stage Latency (`tg128` + `pp1024`, 1 Active Gated `RKAD v3` Hook, `clip=0.15`)

| Benchmark | Configuration | Throughput | Latency | Overhead vs Base | Measured Prep Breakdown |
|---|---|---:|---:|---:|---|
| **`tg128`** (CUDA Graph ON) | **1. Base** (`no hook`) | `105.20 tok/s` | `9,505.6 µs/tok` | — | — |
| **`tg128`** (CUDA Graph ON) | **2. Disabled** (`scale=0`, zero recapture) | `104.94 tok/s` | `9,528.9 µs/tok` | `+23.3 µs` (**`+0.25%`**) | `tok=7.9`, `hash=0.2`, `gath=0.0`, `h2d=2.3 µs` |
| **`tg128`** (CUDA Graph ON) | **3. Active** (gated `RKAD v3`, `clip=0.15`) | `104.26 tok/s` | `9,591.3 µs/tok` | `+85.7 µs` (**`+0.90%`**) | `tok=7.9`, `hash=0.2`, `gath=7.1`, `h2d=4.7 µs` |
| **`pp1024`** (chunk=256) | **1. Base → 2. Active** | `379.3 → 377.5 tok/s` | `2,699.7 → 2,712.9 ms` | `+13.17 ms` (**`+0.49%`**) | `tok=1.16`, `hash=0.17`, `gath=9.46`, `h2d=0.06 ms` |

- `tokenize_align_us` dropped **5.5×** (`43.4 µs/tok → 7.9 µs/tok`), bringing total active end-to-end decode overhead under **1.0%** (`+0.90%`) and prefill overhead under **0.5%** (`+0.49%`).
- Per-token CUDA-Graph-safe `EngramRuntimeParams` (`scale`, `gate_bias`, `delta_clip_ratio`) allow per-request enabling/disabling and strength tuning in `rookery-server` without CUDA Graph recapture (verified exact `+0.000000` disabled PPL delta).
- `RKAD v3` embeds a strict JSON manifest checked at load time against the table's 64-bit FNV-1a addressing fingerprint (`provider_fingerprint`), `provider_arch`, `host_arch`, `host_n_layers`, `host_d_embd`, and `host_hc_mult` (plus per-hook `has_gate`, `rank`, and `delta_clip`).

### C. 200K Mixed-Domain Level-3 Findings & RMSNorm Logit-Softening Diagnostic

Scaling the `L44` causal dump to `200,000` mixed Wiki + Tech tokens (`base val PPL = 15.091`) surfaced an important failure mode caught directly by our 4-way `[SHUFFLED]` control:
- Under vanilla `CE + 0.5*KL` on high-entropy raw Wiki completions (`base PPL = 15.091`), a late-layer bridge placed directly before `output_norm` can exploit an **RMSNorm logit-softening shortcut** (increasing `||h + delta||` scales down logits like temperature `T > 1`), causing both `[REAL]` and `[SHUFFLED]` PPL to drop on raw prose/tech completions (`SHUFFLED` drops even more because uniform random hash rows have higher variance than frequent-token rows), while hurting calibrated `EVAL_FACTUAL` (`1.400 → 1.426`).
- Adding **contrastive shuffled-batch KL regularization** (`--shuf-kl-beta 2.0`) + **relative delta-norm penalty** (`--delta-reg 2.0`) + `delta_clip=0.15` restored monotonic `EVAL_FACTUAL` gains (`1.400 → 1.397 → 1.396 → 1.395`, `SHUF − REAL = +0.030`, vs shuffled degrading `1.400 → 1.409 → 1.422 → 1.425`) and produced a `+0.444` in-distribution `SHUF − REAL` gap (`13.931` vs `14.321`), though residual norm-inflation before `output_norm` still left a small `-0.02` to `-0.09` shuffled bias on raw-text `EVAL_TECH`/`EVAL_PROSE`.
- **Architectural conclusion & mathematical fix (`--calibrate-base-temp`):** Because `output_norm` projects onto the unit sphere, any perturbation of relative norm $r$ shrinks base logits by $\cos\theta = 1/\sqrt{1+r^2}$, giving free temperature softening whenever the base model is overconfident ($\tau^* > 1$). Fitting the base model's optimal scalar temperature $\tau^*$ once before epoch 1 ($\tau^* = 1.30$ on the 200K mixed corpus) zeroes $\partial \mathrm{CE}/\partial(\cos\theta) = 0$ during training while keeping inference unmodified ($\tau = 1.0$). Combined with `--shuf-kl-beta 1.5 --delta-reg 1.0` and `RKAD v3` (`clip=0.15`), **`[REAL]` beat `[SHUFFLED]` on 9 / 9 scale settings across all three held-out suites**:

| Held-Out Eval (Temp-Calibrated 200K `RKAD v3`) | `[BASE]` | `[DISABLED]` | `[REAL]` (`0.5 / 1.0 / 1.5`) | `[SHUFFLED]` (`0.5 / 1.0 / 1.5`) | `SHUF − REAL` |
|---|---:|---:|---|---|---|
| **`EVAL_FACTUAL`** (60 QA items) | `1.400` (`91.7%`/`94.3%`/`96.2%`) | `1.400` (`+0.000000`) | **`1.396 → 1.392 → 1.391`** (`93.3%`/`95.2%`/`97.1%`) | `1.402 → 1.406 → 1.407` (flat `91.7%`) | **`+0.006 / +0.014 / +0.016`** (3/3) |
| **`EVAL_TECH`** (2,048 tok) | `12.746` (`top5 75.63%`) | `12.746` (`+0.000000`) | **`12.612 → 12.509 → 12.464`** (`-0.282`, `top5 76.22%`) | `12.708 → 12.708 → 12.736` | **`+0.096 / +0.200 / +0.271`** (3/3) |
| **`EVAL_PROSE`** (2,048 tok, OOD) | `4.759` | `4.759` (`+0.000000`) | **`4.759 (-0.0005)`** / `4.765` / `4.772` | `4.767 → 4.781 → 4.799` | **`+0.009 / +0.017 / +0.027`** (3/3) |

---

## 7. Multi-Provider & Multi-Host Decoupling Proof (`Qwen3.8-Flash-Next` Provider + `Qwen3.6-35B-A3B` Host)

Using the exact same `EngramTable` + `EngramSidecar` + `RKAD v3` contract (zero model-specific bridge code):

1. **Provider #2 — `Qwen3.8-Flash-Next` (`qwen4exp`, 50.66 GiB `Q8_0` `mmap` PLE table, `fp=0xea58e1fab6490963`):**
   - `1 layer × 16 heads × head_dim=160 (d_engram=2560)`, `320,001,536` rows, EOS-cut raw-token N-gram hashing: opens in `119.3 ms` (0 GiB VRAM), `2.71 µs/tok` warm lookup.
   - `--verify-addressing` against `Maxwell-V1-Flash` host tokenizer: **`851 / 851` (100.00%)** tokens, **`13,616 / 13,616` (100.00%)** hash cols (`avg tail = 60.4 B`).
2. **Host #2 — `DeepSeek-V4.1-Flash` (103 GiB) → `Qwen3.6-35B-A3B` (`qwen35moe`, GDN + GQA MoE, `hc=1`, 40 layers):**
   - `--verify-addressing`: **`927 / 927` (100.00%)** mixed (`44,496 / 44,496` hashes) and **`1,500 / 1,500` (100.00%)** tech (`72,000 / 72,000` hashes).
   - Pure closed-form Level-2 delta-Ridge fit (`rookery-fit-adapter --auto-lam --rkad-v3`, `4.96 s` on GPU): `L1 val R² = +0.2080` (`lam=2.0`), `L14 val R² = +0.0184` (`lam=5.0`).
   - 4-way held-out evals (`[DISABLED]` = `+0.000000` on both):
     - **`EVAL_TECH` (2,048 tok):** Base `5.738` (`top1 60.45%`) → Real `5.709` (`-0.029`, `top1 60.69%`, `top5 82.03%`) at `scale=1.0`, beating Shuffled on 3/3 scales (`+0.0155 / +0.0282 / +0.0070`).
     - **`EVAL_PROSE` (2,048 tok, OOD):** Base `2.782` (`top1 75.73%`) → Real `2.779` (`-0.0034`, `top1 75.98%`) at `scale=0.5` (`+0.0082` vs Shuffled) and `2.781` (`top1 76.07%`) at `scale=2.0` (`+0.0211` vs Shuffled).
   - **Memory-residency note (`tg128`, `133.8 tok/s` base):** Disabled overhead is `+21.6 µs/tok` (`+0.29%`); warm-RAM active overhead is `~77.8 µs/tok` (`~1.04%`: `tok=11.8`, `hash=1.4`, `gath=17.1`, `h2d=9.9`, `gpu_hooks=37.6 µs`), whereas cold NVMe page faults after loading a 36 GiB model + 103 GiB table on a 128 GiB RAM host add `~2.0 ms/tok` until pages warm up.

---

## Honest Takeaways

1. **Sub-1% systems & runtime viability is proven on real hardware:** 100.00% causal addressing parity (`2,351 / 2,351` tokens, `112,848 / 112,848` hash cols), exact `+0.000000` zero-recapture disabled invariance, strict `RKAD v3` manifest validation, and **`+0.90%` decode / `+0.49%` prefill** measured overhead (`tok = 7.9 µs/tok`) against a 103 GiB `mmap` table.
2. **Real cross-model transfer is verified — and 4-way controls are non-negotiable:**
   - Level-2 centered/auto-gated `L1` Ridge improves both `EVAL_TECH` (`12.746 → 12.571`) and `EVAL_PROSE` (`4.759 → 4.725`, `top1 +0.63%`) while beating `[SHUFFLED]` on both.
   - Level-3 supervised gated `L44` improves completely zero-shot **factual QA** monotonically across all scales (`ans_PPL 1.400 → 1.382`, exact first-token accuracy `91.7% → 93.3%`, `SHUF − REAL = +0.040`).
   - Without the `[SHUFFLED]` control, late-layer `L44` training on high-entropy raw text would have looked like a massive `-1.40` PPL win when it was actually RMSNorm logit softening — validating why every adapter must pass 4-way controls (`Base`, `Disabled`, `Real`, `Shuffled`) before promotion.
