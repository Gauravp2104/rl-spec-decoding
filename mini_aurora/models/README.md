# [BP-03] Target + EAGLE-3 draft on MPS

Tracks #8. Branch: `bp/03-models-mps`.

- Load Qwen3-0.6B via `HFTargetModel` (aux layers [1, 13, 24]).
- `examples/mac-qwen3-0.6b/draft_config.json`: hidden 1024, 16 heads, 8 KV heads, intermediate 3072, 1 layer, `draft_vocab_size: 16000`.
- `Eagle3Model(draft, length=3, attention_backend="sdpa")`; frozen embedding and target `lm_head` shared with the target (tied weights).
- **Done when:** overfit test drives loss on one batch toward 0 in ~200 steps; step time and peak memory recorded.

## Progress

- [x] **Target model** (`target.py`): `load_target()` loads Qwen3-0.6B through Aurora's `HFTargetModel` on `get_device()`
  (bf16, `attn_implementation="sdpa"`, eval mode, all parameters frozen) with aux layers `[1, 13, 24]`.
  `generate()` does greedy target-only decoding; `capture_features()` returns `generate_eagle3_data` output
  (`hidden_states` = 3 × 1024 concatenated aux states, `last_hidden_states` = final-norm output).
- [ ] Draft config `examples/mac-qwen3-0.6b/draft_config.json`
- [ ] `Eagle3Model(draft, length=3, attention_backend="sdpa")` with shared frozen embedding / `lm_head`
- [ ] Overfit test, step time and peak memory

## Run a few queries on MPS

```bash
source .venv/bin/activate && source mini_aurora/env.sh
python -m mini_aurora.models.query_target                                  # 3 built-in prompts (chat / code / math)
python -m mini_aurora.models.query_target -q "What is 17 * 23?" -q "Write a haiku about GPUs." --max-new-tokens 200
python -m mini_aurora.models.query_target --thinking                       # Qwen3 thinking mode
python -m mini_aurora.models.query_target --device cpu                     # compare against CPU
python -m pytest -q mini_aurora/tests/test_target.py                       # 3 tests; skipped if the model isn't cached
```

The first run downloads Qwen3-0.6B (~1.5 GB) to `~/.cache/huggingface`. The `torch_dtype is deprecated` warning
comes from Aurora's `HFTargetModel` and is harmless.

Measured on M5 Pro (24 GB), bf16, bs=1: load 0.8 s, weights 1.20 GB, greedy decoding ~70–76 tok/s.
