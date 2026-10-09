# [BP-03] Target + EAGLE-3 draft on MPS

Tracks #8. Branch: `bp/03-models-mps`.

- Load Qwen3-0.6B via `HFTargetModel` (aux layers [1, 13, 24]).
- `examples/mac-qwen3-0.6b/draft_config.json`: hidden 1024, 16 heads, 8 KV heads, intermediate 3072, 1 layer, `draft_vocab_size: 16000`.
- `Eagle3Model(draft, length=3, attention_backend="sdpa")`; frozen embedding and target `lm_head` shared with the target (tied weights).
- **Done when:** overfit test drives loss on one batch toward 0 in ~200 steps; step time and peak memory recorded.
