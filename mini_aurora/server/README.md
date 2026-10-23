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
- [ ] 7. Logging  - [ ] 8. Training samples  - [ ] 9. Draft hot-swap  - [ ] 10. CLI + 50-prompt test

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
The overfit draft has seen this answer, so this is an upper bound; BP-04's trained draft gives the real number.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest -q -s mini_aurora/tests/test_server.py   # 9 tests, ~35 s
```
