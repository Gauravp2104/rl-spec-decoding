# [BP-03] Target + EAGLE-3 draft on MPS

Tracks #8. Branch: `bp/03-models-mps`.

- Load Qwen3-0.6B via `HFTargetModel` (aux layers [1, 13, 24]).
- `examples/mac-qwen3-0.6b/draft_config.json`: hidden 1024, 16 heads, 8 KV heads, intermediate 3072, 1 layer, `draft_vocab_size: 16000`.
- `Eagle3Model(draft, length=3, attention_backend="sdpa")`; embedding copied from the target and frozen. The draft has its own trainable 16K-row `lm_head` (as in Aurora); the target `lm_head` only produces the training labels.
- **Done when:** overfit test drives loss on one batch toward 0 in ~200 steps; step time and peak memory recorded.

## Progress

- [x] **Target model** (`target.py`): `load_target()` loads Qwen3-0.6B through Aurora's `HFTargetModel` on `get_device()`
  (bf16, `attn_implementation="sdpa"`, eval mode, all parameters frozen) with aux layers `[1, 13, 24]`.
  `generate()` does greedy target-only decoding; `capture_features()` returns `generate_eagle3_data` output
  (`hidden_states` = 3 × 1024 concatenated aux states, `last_hidden_states` = final-norm output).
- [x] **Draft config** (`examples/mac-qwen3-0.6b/draft_config.json`): `LlamaForCausalLMEagle3`, hidden 1024,
  16 heads / 8 KV heads, intermediate 3072, 1 layer, `vocab_size` 151,936, `draft_vocab_size` 16,000
  (same layout as Aurora's `examples/qwen3-*` configs).
- [x] **Vocab mapping** (`vocab.py`): `build_vocab_mapping()` tokenizes conversations with the `qwen` template and
  assistant-only loss mask, then calls Aurora's `generate_vocab_mapping` (top-16K assistant tokens → `d2t` offsets,
  `t2d` mask); `save_vocab_mapping` / `load_vocab_mapping` cache it; `coverage()` checks per-domain coverage.
  Temporary mapping from the built-in samples (`samples.py`, 2 chat + 2 code + 2 math) until BP-02's streams exist.
- [x] **Draft + wrapper** (`draft.py`): `build_draft()` builds the draft from the config, copies the target embedding
  and freezes it, and sets the vocab buffers; `wrap_eagle3()` = `Eagle3Model(draft, length=3, attention_backend="sdpa")`.
  `forward_step()` mirrors `Eagle3Trainer._forward` (shift tokens / target states by one, pruned target probs,
  3-step TTT unroll); `weighted_loss()` = Σ 0.8^i · loss_i.
  As in Aurora, the draft has its **own trainable 16K-row `lm_head`**; the target `lm_head` only makes the labels.
  Parameters: 34.2M trainable (`fc`, decoder layer, `norm`, `lm_head`), 155.6M frozen (embedding).
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

## Build the vocab mapping and the draft

```bash
python -m mini_aurora.models.vocab                                         # temp mapping → outputs/vocab/qwen3_16k_samples.pt
python -m mini_aurora.models.vocab --jsonl <stream>.jsonl --out outputs/vocab/qwen3_16k_mixed.pt   # after BP-02
python -m pytest -q mini_aurora/tests/test_vocab.py mini_aurora/tests/test_draft.py
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest -q mini_aurora/tests/test_draft.py   # proves no silent CPU fallback
```

```python
from mini_aurora.models.target import load_target, load_tokenizer, capture_features
from mini_aurora.models.vocab import load_vocab_mapping, tokenize_conversations
from mini_aurora.models.draft import build_draft, wrap_eagle3, forward_step, weighted_loss, target_lm_head_weight
from mini_aurora.models.samples import SAMPLE_CONVERSATIONS

target, tokenizer = load_target(), load_tokenizer()
eagle3 = wrap_eagle3(build_draft(target, load_vocab_mapping("outputs/vocab/qwen3_16k_samples.pt")))
data = tokenize_conversations(tokenizer, SAMPLE_CONVERSATIONS[:1])
features = capture_features(target, data["input_ids"][0].to(target.model.device), data["loss_mask"][0])
plosses, acces = forward_step(eagle3, features, target_lm_head_weight(target))
weighted_loss(plosses).backward()      # random init: loss ≈ 9.9 per position (ln 16000 ≈ 9.68), acc 0
```

JSONL rows for `--jsonl` hold a conversation under `conversations` or `messages` (`role`/`content` or ShareGPT
`from`/`value`). The temporary mapping has only ~180 real tokens; the rest of the 16K are filler IDs, so
rebuild it from BP-02's mixed stream before BP-04.

The first run downloads Qwen3-0.6B (~1.5 GB) to `~/.cache/huggingface`. The `torch_dtype is deprecated` warning
comes from Aurora's `HFTargetModel` and is harmless.

Measured on M5 Pro (24 GB), bf16, bs=1: load 0.8 s, weights 1.20 GB, greedy decoding ~70–76 tok/s.
Target + draft + one forward/backward on an 82-token sample: 2.68 GB MPS driver memory.
