# [BP-05] Mini speculative-decoding server

Tracks #12. Branch: `bp/05-spec-server`.

- Chain drafting (topk=1, K steps): `fc(concat(3 aux hidden states)) + embed(next token)`, then draft's own hidden states for later steps.
- One target verify pass over K+1 positions with HF `DynamicCache` (crop on reject). Greedy (exact match) and sampling (`min(1, p/q)`, residual resample) acceptance.
- Log per-step accepted length + draft version; emit training samples (prompt + generation, aux hidden states, loss mask) = Aurora's `train_with_decode`.
- Start cache-free (recompute) for correctness, then add KV cache.
- **Done when:** greedy output is token-identical to `model.generate` on 50 prompts (bs=1, fp32).
