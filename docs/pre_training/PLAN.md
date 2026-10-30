# Draft Pre-training Plan — Data Recipe, Tree Drafting, Draft and Target Size

Status: proposed (2026-10-28). Integration branch `pre-training` (off `base-paper`); sub-task branches
`pt/NN-<slug>` → `pre-training`, issues and PRs titled `[PT-NN] …`, label `pre-training`, GitHub Project
"Draft pre-training". Scope: the **offline** EAGLE-3 draft that BP-04 produces and BP-06 / BP-08 start
from. Builds on BP-04 (`mini_aurora/baseline/`), the 9K data-scale run (`bp/04-data-scale`) and tree drafting
(`bp/05-tree-drafting`). The online loop, the loss and the GRPO project are out of scope.

## 1. Motivation

BP-08 showed online adaptation works, but every arm starts from a weak draft:

| draft | train data | held-out acc (pos 1 / 2 / 3) | τ̄ (held-out chat, K = 3) | α1 / α2 / α3 |
|---|---|---|---|---|
| `bp04-static-v1` | 900 ShareGPT chat, 8 ep | 0.311 / 0.271 / 0.248 | 0.315 | 0.243 / 0.058 / 0.014 |
| `kaggle_9k` | 9,000 ShareGPT chat, 8 ep | 0.516 / 0.451 / 0.417 | 0.623 | 0.401 / 0.159 / 0.063 |

With a chain draft and a roughly constant per-guess acceptance p, τ̄ ≈ p + p² + p³. Both drafts fit this model:
p ≈ 0.24 → 0.31 and p ≈ 0.40 → 0.62. On the Mac a round costs ~2.2 target-only steps (BP-07), so breaking even
needs τ̄ ≈ 1.2, i.e. p ≈ 0.55. On the T4, online arms are already at ×1.04–1.08.

This plan raises p through the training data (§5), recovers more of each round through tree drafting (§7), and
measures how **target size** and **draft size** change τ̄, latency and speed-up (§6).

## 2. Research questions

| RQ | question | knob | primary metric |
|---|---|---|---|
| RQ1 | Does training on the target's own outputs raise acceptance? | self-distilled vs original responses | τ̄, α_i |
| RQ2 | Does mixing code / math into training help on the mixed stream, at a fixed conversation count? | domain mix | per-domain τ̄ |
| RQ3 | Does a larger **target** change τ̄, per-round latency or speed-up? | Qwen3-0.6B / 1.7B / 4B | τ̄, R, S |
| RQ4 | Does a deeper **draft** pay for its own cost? | 1 vs 2 draft decoder layers | τ̄, t_d, S |
| RQ5 | How much does tree drafting add per checkpoint, and at what verify cost? | top-k / beam | τ̄, t_v(n), S |

### Latency model

Every result is decomposed so that a τ̄ change can be told apart from a cost change:

```
R = (K · t_d + t_v(n + 1) + t_o) / t_t        round cost, in target-only decode steps
S = (1 + τ̄) / R                               speed-up over target-only decoding
c = t_d / t_t                                  draft-to-target step ratio (BP-07: 0.099 on the Mac)
```

t_t = one target-only decode step; t_d = one draft step; t_v(n + 1) = one target verify pass over the root plus n
drafted nodes (n = K for a chain); t_o = per-round bookkeeping (cache crop, acceptance, host syncs).

Which terms each knob moves:

| knob | τ̄ | t_d | t_v / t_t | c |
|---|---|---|---|---|
| data recipe (§5) | ↑ expected | – | – | – |
| bigger target (§6) | unknown sign: measured | ↑ (draft width follows target width) | ≈ 1 | ↓ only where the target step is memory-bound |
| deeper draft (§6) | ↑ expected, mainly α2 / α3 | ↑ | – | ↑ |
| tree (§7) | ↑ | – | ↑ with n; ≈ free when memory-bound | – |

### Hypotheses about scale (stated before measuring)

- **H-target-τ.** At equal data and draft depth, τ̄ changes little with target size (±0.1). Larger targets give
  richer features but have a different output distribution, so the sign is not predicted.
- **H-target-latency.** At batch size 1 under HF eager, the 0.6B step is overhead-bound on every device we have.
  Measured target-only step times are 18 ms on the Mac, 30 ms on the A100 MIG slice and 40 ms on the T4. The
  weight-read floor for 2.4 GB of fp32 weights is ~7.5 ms on the T4 and ~1.6 ms on a full A100. So t_t grows
  with **layer count** (28 → 28 → 36) more than with width, until the weights become bandwidth-bound. 4B fp32
  (16 GB) passes that point on the T4 (~50 ms floor) but not on the A100 (~11 ms). The draft stays
  overhead-bound everywhere. Prediction: c falls ~1.5–2× from 0.6B to 4B on the T4 and the Mac, and only ~1.2×
  on the A100. So a bigger target raises S on the T4 / Mac even at constant τ̄, but barely on the A100.
- **H-draft.** A second draft layer raises t_d by ~30–50% (one more decoder layer next to the fc, norm and 24K
  `lm_head` launches). It is worth it only if (1 + τ̄₂) / (1 + τ̄₁) > R₂ / R₁.

## 3. Fixed controls

Unless an arm says otherwise, everything matches BP-04 / BP-08 so results stay comparable:

| setting | value |
|---|---|
| draft vocab | 24K mixed (`outputs/vocab/qwen3_24k_mixed.pt`). The same `d2t` is reused for every target, since Qwen3 shares one tokenizer. |
| TTT length / K | 3 / 3 (trees: depth 3) |
| loss | Σ 0.8ⁱ · forward-KL_i against the target's pruned distribution |
| optimizer | AdamW, LR 1e-4, warmup 1.5% + cosine, grad clip 0.5, batch 2 × accumulation 4 |
| precision | fp32 draft weights + gradient checkpointing; fp32 target for dumps and τ̄ |
| sequence | ≤ 1024 tokens per training sequence |
| aux layers | EAGLE-3 rule `[1, L/2 − 1, L − 4]`: 28 layers → 1 / 13 / 24, 36 layers → 1 / 17 / 32 |
| draft init | embedding copied from the target and frozen; everything else random (as BP-04) |
| chat template | `enable_thinking=False`, identical in generation, dump and serving |
| serving | greedy, bs = 1, 128 new tokens, KV-cached target, `--check` on |

LR 3e-4 won E4, but it is deliberately **not** adopted here, so every gain can be attributed to one knob.

## 4. Evaluation protocol

**Eval set E300.** 100 requests per domain, fixed IDs, none in any training pool:
- chat: `chat-00900` … `chat-00999`, the BP-04 held-out split, also excluded from the 9K / 27K pools;
- code: 100 CodeAlpaca stream rows;
- math: 100 GSM8K stream rows.

The extra code and math training prompts (§5) exclude every stream `src_id`. Selection uses the **mean of the
three per-domain τ̄**, which matches the mixed stream.

**Metrics per served configuration:**
- τ̄ with a 95% bootstrap CI over requests, α_i, and per-domain τ̄;
- lossless count, which must be 300/300 or the run is invalid;
- tok/s, and measured S against target-only on the same device and dtype;
- teacher-forced top-k accuracy (k = 1 / 2 / 4 / 8) per position on the held-out dumps; top-k at position 1
  predicts depth-1 tree acceptance.

**Latency benchmark** (new, PT-5): per checkpoint × device, the median over 200 timed calls after 20 warm-up calls
of t_t, t_d, t_v(n + 1) for n ∈ {3, 6, 12, 24}, and t_o, at context lengths 256 and 768. From these it reports c,
R and the predicted S, next to the measured S. It also records whether t_t is above the weight-read floor
(overhead-bound) or at it (bandwidth-bound).

**Devices.** τ̄ is hardware-independent, so each checkpoint is served once (fp32, A100) for τ̄. Latency is
benchmarked on the A100 (fp32 and bf16), the T4 (fp32, plus fp16 for 4B) and the Mac (fp32 up to 1.7B, bf16 for 4B).

**Training-seed variance.** One seed per arm. The Stage B 0.6B / 1-layer cell is repeated with a second seed, and
that spread is the noise floor for every comparison.

## 5. Stage A — data recipe on Qwen3-0.6B (RQ1, RQ2)

### Self-distilled data (RQ1)

For each training conversation, keep the user turns and regenerate every assistant turn **sequentially** with the
target: turn i is generated given the user turns and the target's own earlier answers.
- greedy decoding, `enable_thinking=False`, ≤ 512 new tokens per turn;
- truncate to 1024 tokens;
- loss mask on the regenerated assistant tokens, including the end-of-turn token;
- drop exact-duplicate responses, but keep repetitive ones: greedy serving produces them too;
- log the share of turns that hit the 512 cap or loop.

### Arms

| arm | prompts | responses | conversations (chat / code / math) | epochs | status |
|---|---|---|---|---|---|
| A0 | BP-02 chat pool (900) | ShareGPT original | 900 / 0 / 0 | 8 | done (`bp04-static-v1`) |
| A1 | 9K extended chat pool | ShareGPT original | 9,000 / 0 / 0 | 8 | done (`kaggle_9k`): serve on E300 only |
| A2 | A1's prompts | **target greedy** | 9,000 / 0 / 0 | 8 | new: A2 − A1 isolates self-distillation |
| A3 | 5K of A2's chat + CodeAlpaca + GSM8K | target greedy | 5,000 / 2,000 / 2,000 | 8 | new: A3 − A2 isolates domain mix at fixed count |
| A4 | winner of A2 / A3 at full size: all 27,475 filtered ShareGPT chat (+ ~6K code / ~6K math if A3 wins) | target greedy | ~27K–39K | 8 | new: scaling point |

- Assistant-token counts are reported for every arm. If A3's count differs from A2's by more than 20%, trim chat
  prompts to match.
- **Recipe R\*** is the A2 / A3 winner on mean per-domain τ̄. Ties within 0.02 go to A2, the simpler recipe.
- R\* is the recipe used in Stage B, at 9K size to bound cost. A4 runs in parallel with Stage B and only adds a
  point to the data-scaling curve (900 / 9K / 27K).

## 6. Stage B — target size × draft depth (RQ3, RQ4)

Recipe R\*, 9K conversations regenerated **by each target** (self-distilled data is target-specific). Every cell
trains for **4 epochs**, the cosine schedule spanning 4 epochs. Comparisons stay within Stage B; the 0.6B / 1-layer
cell also measures the 4- vs 8-epoch gap against Stage A. One dump per target serves both draft depths.

| target | layers / hidden | draft config (1 layer) | trainable draft params, 1 layer → 2 layers |
|---|---|---|---|
| Qwen3-0.6B | 28 / 1024 | 16 heads / 8 KV (head_dim 64), intermediate 3072 | 42.4M → 55.0M |
| Qwen3-1.7B | 28 / 2048 | 16 heads / 8 KV (head_dim 128), intermediate 6144 | ~120M → ~171M |
| Qwen3-4B | 36 / 2560 | 20 heads / 4 KV, intermediate 9728 (Aurora's `examples/qwen3-4b-*/draft_config.json`) | ~181M → ~271M |
| Qwen3-8B (stretch, 1 layer only) | 36 / 4096 | Aurora's `examples/qwen3-8b-*/draft_config.json` | – |

- Parameter counts include the 24K `lm_head`; the 0.6B number is the measured BP-04 count.
- **Prior result:** BP-09 trained a 1.7B-target draft on the same 900 conversations: held-out acc 0.35 vs 0.31,
  but the same τ̄ (0.319 vs 0.315), consistent with H-target-τ (`mini_aurora/zaratan/README.md`).
- **Grid:** 3 targets × {1, 2} draft layers = 6 cells, plus the second-seed 0.6B / 1-layer repeat.
- **Optional 8B cell:** calibrates against the setting Aurora reports.
- **Draft depth.** Layer 0 stays Aurora's `midlayer` (`aurora-main/aurora/models/draft/llama3_eagle.py:1746`),
  which takes the 2H concatenation of embedding and fc output. Layers 1 … n−1 are standard H-input decoder layers.
  The TTT unroll and the server's drafter feed the **last** layer's output to the next step.
- **Outputs:**
  - τ̄ vs target size, one line per draft depth;
  - R decomposed into K·t_d / t_v / t_o as a stacked bar per cell and device;
  - S vs τ̄ per device, with iso-R curves.

## 7. Stage C — tree drafting (RQ5)

Serve-only, no retraining (`mini_aurora/server/serve.py`, flags `-k / --tree-topk / --tree-beam`). It runs on A1,
R\*, A4 and all Stage B checkpoints.

| config | `-k` | `--tree-topk` | `--tree-beam` | drafted nodes n (≤) |
|---|---|---|---|---|
| T0 chain | 3 | 1 | 1 | 3 |
| T1 | 3 | 2 | 2 | 6 |
| T2 | 3 | 4 | 4 | 12 |
| T3 | 3 | 8 | 8 | 24 |

- Depth stays at 3 to match TTT 3.
- Report τ̄, t_v(n + 1) / t_t and S per config × device.
- Compare depth-1 acceptance with teacher-forced top-k accuracy at position 1, to check that the offline number
  predicts the tree gain.
- Expect trees to pay off most where the verify pass is memory-bound (4B on the T4) and least on MPS fp32.

## 8. Work items

| ID | scope | deliverables | branch |
|---|---|---|---|
| PT-01 | Stage A (§5) | `mini_aurora/data/regenerate.py` (sequential greedy regeneration, resumable shards); CodeAlpaca / GSM8K training prompts excluding stream `src_id`s; E300 manifest; arms A2–A4 | `pt/01-self-distill-data` |
| PT-02 | Stage B (§6) | per-target aux layers and draft config for 4B (BP-09 already covers 1.7B); `num_hidden_layers > 1` in the draft, TTT path and drafter with depth-2 losslessness tests; the 6-cell grid + seed repeat | `pt/02-size-grid` |
| PT-03 | Stage C (§7) and §4 | `mini_aurora/eval/bench_latency.py`; `evaluate.py --topk`; tree configs T0–T3 on every checkpoint; results table and figures | `pt/03-tree-latency-bench` |

PT-01 and PT-03 can start immediately; PT-02's grid waits for PT-01's recipe R\* and uses PT-03's benchmark.

## 9. Compute and disk budget

These are estimates scaled from the measured 9K run (dump 58 min, train 6.1 h on a T4). PT-5's measurements replace
them.

| stage | where | GPU time | disk (bf16 dumps, peak) |
|---|---|---|---|
| A2, A3 | Kaggle T4 | ~9 h each (generate ~1.5, dump ~1, train ~6.1, E300 ~0.75) | ~40 GB each, sequential |
| A4 | Zaratan A100 | ~12 h | ~120 GB |
| B: generate + dump | Zaratan A100 | 0.6B ~1 h, 1.7B ~2 h, 4B ~3 h | ~40 / ~80 / ~100 GB (4 × H × 2 bytes per token), deleted after both depths train |
| B: train (4 ep, 6 cells + repeat) | Zaratan A100 | ~20 h | – |
| B + C: serve E300 + bench | A100 (0.6B on MIG) | ~12 h | – |
| **total Zaratan** | | **~50 A100-h ≈ 2,400 SU** at 48 SU/h | |

**If over allocation:** drop the 4B / 2-layer cell first, then T1, then the seed repeat.

## 10. Decision rules and success criteria

- **RQ1 / RQ2.** A recipe counts as better if its mean per-domain τ̄ beats the other's by more than the larger of
  0.02 and the training-seed spread.
- **RQ3 / RQ4.** Both are answered by the measured S on each device, not by τ̄ alone. A deeper draft counts as
  worth it if S rises on at least 2 of the 3 devices. A larger target counts as worth it per device. The expected
  outcome differs by device (H-target-latency).
- **Validity.** Any configuration with fewer than 300/300 lossless outputs is excluded and debugged.
- **Headline goal (not a gate).** A 0.6B draft with chain τ̄ ≥ 1.0 on E300, and S > 1 on the Mac with the best
  tree config.

## 11. Risks

- **Regime-dependent latency.** HF eager at bs = 1 is overhead-bound, so conclusions about c may not carry over to
  compiled or CUDA-graph serving. The benchmark records the regime (§4).
- **Cache-free drafter.** `mini_aurora/server/drafter.py` recomputes the draft over the full context every step,
  which penalises deeper drafts and long contexts. If a draft KV cache lands before Stage B, PT-5 is rerun and both
  sets of numbers are reported.
- **Multi-layer TTT.** Aurora's TTT path assumes one `midlayer`. PT-4 must give each layer its own per-step
  attention cache, and must pass the depth-2 losslessness test before any training.
- **Greedy self-distillation.** It can over-represent repetitive 0.6B loops. The loop share is logged (§5); if it
  exceeds ~10% of turns, add an arm with sampled regeneration (T = 0.7).
- **4B fp32 memory.** 16 GB of weights rules out Kaggle T4 dumps and A100 MIG serving. All 4B work runs on full
  A100s.
- **Comparability with BP-08.** New drafts are out-of-sample for every stream, unlike `bp04-static-v1`. Rerunning
  BP-08's E2 / E3 static arms with R\* is a natural follow-up, but stays outside this plan.
