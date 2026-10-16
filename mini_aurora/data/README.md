# [BP-02] Data subsets and request streams

Tracks #6. Branch: `bp/02-data-streams`.

- Public replacements for Aurora's private mixed dataset: ShareGPT (`Aeala/ShareGPT_Vicuna_unfiltered`), CodeAlpaca, GSM8K; ~1K prompts per domain.
- Emit **mixed** (shuffled), **ordered** (grouped by domain), **domain-shift** (chat→code→math) JSONL streams.
- Reuse `qwen` chat template, `compute_assistant_loss_mask`, `generate_vocab_mapping` → 16K `d2t`/`t2d`.
- **Done when:** three streams + cached vocab mapping on disk (CPU only).
