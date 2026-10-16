# Base Paper Implementation — Scaled-Down Aurora on Apple Silicon

Reproduction of **Aurora** ([arXiv:2602.06932](https://arxiv.org/abs/2602.06932), code in `aurora-main/`)
at laptop scale. **Scope: base paper only (forward-KL online EAGLE-3 training) — no GRPO.**
The GRPO extension from our proposal is a separate, later project; this plan leaves hooks for it
(per-step accepted lengths, top-k drafting, draft-version logging).

## 1. Why scaled down

The full Aurora setup needs ~6× 80 GB GPUs for Qwen3-8B (4 for SGLang at TP=4, 2 for FSDP training)
and ~650–700 GPU-hours for the experiments in our proposal. The stack (SGLang, Mooncake, Ray, FSDP2,
FlexAttention/FlashAttention, CUDA graphs) is CUDA-only. We keep Aurora's algorithm and reuse its
model/loss/data code, and replace only the infrastructure.

## 2. Component mapping

| Aurora component | Mac replacement |
|---|---|
| SGLang server + EAGLE-3 patch | HF `transformers` draft→verify loop on MPS (`mini_aurora/server`) |
| Aux hidden-state capture (patched SGLang) | `HFTargetModel.generate_eagle3_data` forward hooks (`aurora/models/target/eagle3_target_model.py`) |
| Mooncake transfer store | In-process bounded queue (same `max_sample_pool_size` / buffer-threshold semantics) |
| Ray controller + placement groups | Plain Python loop (sync mode) or two threads (async mode) |
| FSDP2 `Eagle3Trainer` (hard-coded `.cuda()`) | Slim single-device trainer wrapping `Eagle3Model` (`aurora/models/eagle3.py`) |
| FlexAttention / FlashAttention | `attention_backend="sdpa"` |
| `torch.compile` | Eager (`TORCHDYNAMO_DISABLE=1`) |
| Shared-FS weight sync + hot-swap | In-memory `state_dict` copy with a version counter |
| Qwen3-8B / 4B target | Qwen3-0.6B (stretch: Qwen3-1.7B) — same tokenizer and 152K vocab |
| W&B | CSV + matplotlib (W&B offline optional) |

## 3. Config: paper vs. Mac

| Setting | Paper | Mac |
|---|---|---|
| Target | Qwen3-8B / 4B | Qwen3-0.6B (stretch 1.7B) |
| Draft | hidden 4096, 32K draft vocab | hidden 1024, 16K draft vocab (~40M trainable) |
| K / `ttt_length` | 5 / 5 | 5 / 3 |
| `max_seq_length` / `max_new_tokens` | 2048 / 512 | 512–1024 / 128–256 |
| Concurrency / micro-batch | 12 / 4 | 1–4 / 1–2 (+ grad accumulation 4) |
| Requests per run | ~20K | 1–3K |
| Weight-sync interval | 50 steps | 10–20 steps |
| LR | 1e-4 (scratch), 1e-5 (with draft) | same |
| `max_sample_pool_size` | 1000 | **128** (pool shares unified memory) |

**Headline metric:** mean acceptance length τ̄ (hardware-independent). Wall-clock speed-up on MPS
will not match the paper; we also report the estimated speed-up `(1+τ̄)/(1+K·c)`, where
`c` = measured draft/target step-latency ratio.

## 4. Hardware budget (M5 Pro, 24 GB unified, ~16–18 GB usable by the GPU)

Estimates from model configs; PR BP-03 replaces them with measurements.

**Fixed (one-time disk):** env ~2 GB, Qwen3-0.6B ~1.5 GB, raw datasets ~1–2 GB, offline hidden states
for the static baseline ~5 GB (optional — can be computed on the fly). **≈10 GB** (≈25 GB with 1.7B).

**Running memory per experiment:**

| Component | 0.6B | 1.7B |
|---|---|---|
| Python/torch/MPS runtime | ~1–1.5 GB | ~1–1.5 GB |
| Target weights (bf16) | 1.2 GB | 4.1 GB |
| KV cache (112 KB/token × 1–4K tokens) | 0.1–0.5 GB | 0.1–0.5 GB |
| Target forward activations | 0.3–0.5 GB | 0.5–1 GB |
| Server-side draft snapshot | ~0.2 GB | ~0.4 GB |
| Trainer weights + grads + Adam (16 B/param) | ~0.65 GB | ~1.7 GB |
| Trainer activations (vocab logits × TTT) | 1–1.5 GB | 1.5–3 GB |
| Sample pool (128 samples) | ~1 GB | ~2 GB |
| **Peak** | **~6–8 GB** | **~11–14 GB** |

**Experiment disk:** ~0.5 GB per draft checkpoint (0.6B); ~6–18 GB for ~12 runs. **Total ≤ ~30 GB** (≤ ~50 GB with 1.7B).
**Time:** ~12 runs × 1–3 h + development ≈ 30–50 Mac-hours. Run experiments sequentially, under `caffeinate -i`.

## 5. Sub-tasks (one issue + one PR each)

Merge order: **BP-01 → (BP-02, BP-03) → (BP-04, BP-05) → BP-06 → BP-07 → BP-08 → BP-09**.

### BP-01 — Mac environment + device layer (`bp/01-env-mac`)
- `uv venv --python 3.12`; `requirements-mac.txt` (torch, `transformers==4.57.1`, datasets, accelerate, omegaconf, numba, matplotlib, pytest).
- `pip install -e aurora-main --no-deps` (skips Mooncake / Ray / sglang-router).
- `mini_aurora/device.py::get_device()` → cuda → mps → cpu.
- Env: `PYTORCH_ENABLE_MPS_FALLBACK=1`, `TORCHDYNAMO_DISABLE=1`.
- **Done when:** `aurora.models.draft.llama3_eagle` and `aurora.models.eagle3` import on Mac; `pytest tests/test_parse.py tests/test_vocab_mapping.py` passes.

### BP-02 — Data subsets and request streams (`bp/02-data-streams`)
- Public replacements for Aurora's private mixed dataset: ShareGPT (`Aeala/ShareGPT_Vicuna_unfiltered`), CodeAlpaca, GSM8K; ~1K prompts per domain.
- Emit **mixed** (shuffled), **ordered** (grouped by domain), **domain-shift** (chat→code→math) JSONL streams.
- Reuse `qwen` chat template, `compute_assistant_loss_mask`, `generate_vocab_mapping` → 16K `d2t`/`t2d`.
- **Done when:** three streams + cached vocab mapping on disk (CPU only).

### BP-03 — Target + EAGLE-3 draft on MPS (`bp/03-models-mps`)
- Load Qwen3-0.6B via `HFTargetModel` (aux layers [1, 13, 24]).
- `examples/mac-qwen3-0.6b/draft_config.json`: hidden 1024, 16 heads, 8 KV heads, intermediate 3072, 1 layer, `draft_vocab_size: 16000`.
- `Eagle3Model(draft, length=3, attention_backend="sdpa")`; frozen embedding and target `lm_head` shared with the target (tied weights).
- **Done when:** overfit test drives loss on one batch toward 0 in ~200 steps; step time and peak memory recorded.

### BP-04 — Offline static EAGLE-3 baseline (`bp/04-static-baseline`)
- No public EAGLE-3 draft for 0.6B, so train one: dump target hidden states on the **chat-only** split (mirrors `OfflineEagle3Dataset`), train 1–2 epochs, freeze.
- This checkpoint is both the *static* baseline and the init for *with-draft* mode.
- **Done when:** τ̄ clearly > 0 on held-out chat data in BP-05's loop.

### BP-05 — Mini speculative-decoding server (`bp/05-spec-server`)
- Chain drafting (topk=1, K steps): `fc(concat(3 aux hidden states)) + embed(next token)`, then draft's own hidden states for later steps.
- One target verify pass over K+1 positions with HF `DynamicCache` (crop on reject). Greedy (exact match) and sampling (`min(1, p/q)`, residual resample) acceptance.
- Log per-step accepted length + draft version; emit training samples (prompt + generation, aux hidden states, loss mask) = Aurora's `train_with_decode`.
- Start cache-free (recompute) for correctness, then add KV cache.
- **Done when:** greedy output is token-identical to `model.generate` on 50 prompts (bs=1, fp32).

### BP-06 — Online loop (`bp/06-online-loop`)
- Bounded sample pool (`max_sample_pool_size`, `buffer_threshold`), BP-03 trainer, versioned weight publish every N steps.
- Modes: **sync** (serve B requests ↔ train M steps; fixed Δ) and **async** (two threads; real Δ).
- Starts: **day-0** (random draft) and **with-draft** (BP-04 checkpoint).
- **Done when:** a 300-request day-0 run on the mixed stream shows τ̄ rising.

### BP-07 — Evaluation and metrics (`bp/07-eval-metrics`)
- Rolling τ̄ vs. request index, per-position α_i, acceptance rate, tokens/s vs. target-only, estimated speed-up, staleness histogram, loss curve.
- CSV writer + plotting script; losslessness regression test.

### BP-08 — Base-paper experiments (`bp/08-experiments`)

| ID | Paper claim | Experiment |
|---|---|---|
| E1 | Day-0 deployment works | Day-0 online vs target-only, mixed stream |
| E2 | Online adaptation beats static under shift | Static BP-04 draft vs with-draft online, chat→code→math |
| E3 | Ordered-stream behaviour | E2 on the ordered stream |
| E4 | Sensitivity | Sync interval N ∈ {5, 20, 50}, LR, sync vs async |

Results table + plots go back into this document.

### BP-09 — (Optional) CUDA portability / scale check (`bp/09-cuda-scale-check`)
- Same `mini_aurora` code on Colab/cluster CUDA with Qwen3-1.7B/4B and FlexAttention re-enabled.
- If 3 CUDA GPUs are available, run Aurora's `examples/qwen3-4b-external-*` once to calibrate mini results against the real system.

## 6. Hooks for the later GRPO project
- Reward = per-step accepted length (BP-05 logs).
- Groups = top-k > 1 tree paths (BP-05 drafting option).
- Staleness Δ = served draft version vs. trainer version (BP-06 logs).
