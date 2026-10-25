# [BP-05] Mini speculative-decoding server

Tracks #12. Branch: `bp/05-spec-server`.

- Chain drafting (topk=1, K steps): `fc(concat(3 aux hidden states)) + embed(next token)`, then draft's own hidden states for later steps.
- One target verify pass over K+1 positions with HF `DynamicCache` (crop on reject). Greedy (exact match) and sampling (`min(1, p/q)`, residual resample) acceptance.
- Log per-step accepted length + draft version; emit training samples (prompt + generation, aux hidden states, loss mask) = Aurora's `train_with_decode`.
- Start cache-free (recompute) for correctness, then add KV cache.
- **Done when:** greedy output is token-identical to `model.generate` on 50 prompts (bs=1, fp32).

## Progress

- [x] **1. Reference** (`reference.py`): `reference_generate()` = greedy HF `model.generate`, one token per pass;
  `encode_prompt()`, `eos_token_ids()` (Qwen3: 151643, 151645). Run the equivalence test with an **fp32** target
  (`load_target(dtype=torch.float32)`): a K+1-token verify pass rounds differently from one token per pass, and in
  bf16 that can flip near-tied argmaxes.
- [x] **2. Target step** (`target_step.py`): `target_forward()` returns logits, the concatenated aux states of
  layers [1, 13, 24] and the final-norm output, using the same hooks as Aurora's `generate_eagle3_data`
  (identical tensors). Optional HF `DynamicCache` and `num_logits` (keep only the last N positions' logits).
  A cached greedy loop through it reproduces `model.generate` exactly.
- [x] **3. Drafter** (`drafter.py`): `Drafter.propose()` = top-1 chain of K guesses, the TTT training unroll read
  off at the last position (step 0: `fc(aux)` + embed of the target's token; later steps: the draft's own hidden
  state + its previous guess, RoPE offset per step). Greedy or sampled; returns target IDs (via `d2t`),
  draft IDs and each step's distribution q. Cache-free: the draft recomputes the context each round.
- [x] **4. Verify + acceptance** (`verify.py`): the target scores `[x_n, g_1..g_K]` in one pass; row i checks
  g_{i+1} and row a gives the extra token, so each round emits a + 1 tokens. `verify_greedy()`: keep g_{i+1} iff
  it is the target's argmax. `verify_sampling()`: keep it with probability min(1, p/q), else draw from
  norm(max(0, p − q)); if all K pass, draw from p_K. Output is distributed exactly as target sampling (q is the
  distribution the draft *sampled* from, 0 outside the draft vocab).
- [x] **5. Cache-free loop** (`spec_decode.py`, `use_cache=False`): prefill, then draft → verify → commit until EOS
  or `max_new_tokens`; the target re-reads the whole sequence every round. Accepted tokens' aux states are kept
  for the next draft. Returns `new_ids`, `accepted` per round (τ̄ = `mean_accepted`) and the draft version.
- [x] **6. KV cache** (`use_cache=True`): the target keeps an HF `DynamicCache`, runs only the K + 1 new tokens
  per round and `crop()`s the cache back to the accepted length. The drafter stays cache-free.
- [x] **7. Logging** (`log.py`): one JSON line per request with `accepted` and `versions` per round, `k`, token
  counts, time and tok/s (plus `domain`, stream `pos`, `match`). `acceptance_by_position()` gives α_i. τ̄, α_i and
  staleness for BP-07 can all be recomputed from the log.
- [x] **8. Training samples** (`speculative_generate(..., return_sample=True)`): an `Eagle3TargetOutput` laid out
  as Aurora's `generate_with_decode`: prompt + completion minus its last token, their aux and final-norm states
  (reused from the verify passes, so no extra target pass), loss mask on the completion. `forward_step` takes it
  as-is; matches a fresh `capture_features` pass to 1e-3 in fp32.
- [x] **9. Draft hot-swap** (`drafter.py`): `Drafter.load_weights(state, version, d2t)` copies a trainer's
  `fc` / layer / `norm` / `lm_head` into the served draft in place (Aurora's `update_weights_from_disk`) and
  refuses weights trained on another vocab mapping (`d2t` or `t2d` differs); `Drafter.load_file()` reads a BP-04
  checkpoint (`mini_aurora.baseline.checkpoint.save_draft`, the one on-disk format) or a checkpoint directory via
  `latest.txt`, and uses its `step` as the version. Every round logs the version that proposed it.
- [x] **10. CLI + 50-prompt test** (`serve.py`): `serve()` runs stream requests one at a time (bs = 1), with optional
  `check` against target-only greedy, a `RequestLog`, and an `on_sample` callback for BP-06; `summarize()` gives
  τ̄ / α_i / tok/s per domain. `test_serve_50_prompts_lossless`: **50/50 token-identical** on the first 50 rows of
  `data/streams/mixed.jsonl` (18 chat / 17 code / 15 math, 32 new tokens, fp32).

**Alignment check** (`test_drafter_matches_training_alignment`): overfit a draft on a target-greedy sequence, then
draft after every anchor. Hit rate is **1.0 / 1.0 / 1.0** for guesses 1–3 whenever the true tokens are in the
24K vocab (the reference used here has 2 of 48 tokens outside it, " mapped" and " delet", which cause every miss).

**Greedy losslessness** (`test_speculative_greedy_is_lossless`): token-identical to `model.generate` on 3 prompts
with a day-0 draft and on the overfit draft's sequence, with and without KV cache. `test_verify_sampling_*` checks by
Monte Carlo that sampled acceptance reproduces the target distribution p, not the draft's q.

**Speed** (M5 Pro, fp32 target and draft, K = 3, same 55-token answer each run):

| Setup | tok/s | τ̄ | Target passes |
|---|---|---|---|
| Target only (`model.generate`) | 59.9 | – | 55 |
| Day-0 draft, cache-free | 25.5 | 0.00 | 54 |
| Day-0 draft, KV cache | 33.4 | 0.00 | 54 |
| Overfit draft, cache-free | 94.2 | 2.60 | 15 |
| Overfit draft, KV cache | **122.3** | 2.60 | 15 |

A useless draft is a slowdown (every round costs a draft + a verify pass for one token); a good one gives 2× here.
The overfit draft has seen this answer, so this is an upper bound. **BP-04's trained static draft** on the 100
held-out chat requests (`--heldout --check`, up to 128 new tokens):

| Draft | τ̄ | α1 / α2 / α3 | Tokens per target pass | tok/s vs target alone | Lossless |
|---|---|---|---|---|---|
| 2 epochs (step 224) | 0.181 | 0.164 / 0.016 / 0.001 | 1.19 | 29.1 vs 54.9 (×0.53) | 100/100 |
| **8 epochs (step 896)** | **0.313** | **0.242 / 0.058 / 0.013** | **1.32** | **32.1 vs 54.8 (×0.59)** | **100/100** |

A round takes ~41 ms (~2.3 target-only steps), so breaking even needs τ̄ ≈ 1.3.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest -q -s mini_aurora/tests/test_server.py   # 14 tests, ~2 min
python -m mini_aurora.server.serve -n 50 --check                       # done-when; exits 1 on any mismatch
python -m mini_aurora.server.serve -n 100 --heldout --draft outputs/baseline/checkpoints_8ep --check   # BP-04 τ̄
python -m mini_aurora.server.serve -n 20 --draft outputs/baseline/checkpoints_8ep --log outputs/server/run.jsonl
python -m mini_aurora.server.serve -n 20 --sampling --temperature 0.7 --seed 0 --bf16
```

Without `--draft`, `serve` uses a day-0 draft: output is still exact, but τ̄ ≈ 0 and it runs at about half the
target-only speed. BP-04's static baseline plugs in with `--draft outputs/baseline/checkpoints_8ep` (train it with
`python -m mini_aurora.baseline.train --epochs 8 --out outputs/baseline/checkpoints_8ep`); `--heldout` serves only
the 100 chat conversations BP-04 never trained on (its done-when: τ̄ clearly > 0 there, met at 0.313).
