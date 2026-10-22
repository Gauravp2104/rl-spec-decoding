# [BP-02] Data subsets and request streams

Tracks #6. Branch: `bp/02-data-streams`.

- Public replacements for Aurora's private mixed dataset: ShareGPT (`Aeala/ShareGPT_Vicuna_unfiltered`), CodeAlpaca, GSM8K; ~1K prompts per domain.
- Emit **mixed** (shuffled), **ordered** (grouped by domain), **domain-shift** (chat→code→math) JSONL streams.
- Reuse `qwen` chat template, `compute_assistant_loss_mask`, `generate_vocab_mapping` → 24K `d2t`/`t2d` (16K until BP-04).
- **Done when:** three streams + cached vocab mapping on disk (CPU only).

## Run

```bash
source .venv/bin/activate && source mini_aurora/env.sh
python -m mini_aurora.data.download          # ~5 min: ShareGPT json (464 MB) + two small sets -> data/raw/*.jsonl
python -m mini_aurora.data.streams           # -> data/streams/{mixed,ordered,domain_shift}.jsonl
python -m mini_aurora.data.vocab --compare   # -> outputs/vocab/qwen3_24k_mixed.{pt,json}   (needs Aurora, see below)
python -m pytest -q mini_aurora/tests/test_data.py
```

`download` and `streams` are pure Python (tokenizer only, no torch). `vocab` imports Aurora, which needs
torch >= 2.3; on a machine that cannot run that natively (Intel Mac: the last x86_64 wheel is torch 2.2) use
the Linux CPU container:

```bash
docker build -f mini_aurora/docker/Dockerfile.cpu -t mini-aurora-cpu .      # once, ~2.6 GB
bash mini_aurora/docker/run_cpu.sh python -m mini_aurora.data.vocab --compare
bash mini_aurora/docker/run_cpu.sh python -m pytest -q                        # whole suite on CPU
```

Everything is seeded (`SEED = 0` in `mini_aurora/data/__init__.py`); re-running reproduces the same files.

## Row format

One JSON object per line, the full conversation plus tags:

```json
{"id": "code-00017", "domain": "code", "source": "codealpaca", "src_id": 4212,
 "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "n_tokens": 81, "pos": 1337}
```

- `messages` is role/content, starts with `user`, alternates, and ends on an `assistant` turn, which is what
  Aurora's `GeneralParser` requires (it truncates anything else, silently).
- The **server** (BP-05) takes the first user turn as the request. The **offline baseline** (BP-04) and the
  **vocab mapping** use the assistant turns. Keeping the whole conversation in one file serves all three.
- `n_tokens` is the length of the Qwen3 chat-templated conversation; `pos` is the index within a stream.
- `python -m mini_aurora.models.overfit --stream ... --domain ...` (BP-03) reads this format directly (`messages` key).

## Pools (`data/raw/`, `stats.json`)

| domain | source | split | kept | tokens p50 / p90 / max | turns |
|---|---|---|---|---|---|
| chat | ShareGPT V4.3 split | seeded sample | 1000 | 495 / 899 / 1020 | 3.9 |
| code | CodeAlpaca-20k | train | 1000 | 81 / 161 / 649 | 2 |
| math | GSM8K main | train | 1000 | 190 / 290 / 456 | 2 |

Filters run in order after a seeded shuffle until 1000 rows are kept. Only ShareGPT loses rows:

| reason | chat rows dropped | why |
|---|---|---|
| `not_user_first` | 939 | the "split" ShareGPT file chops long threads into chunks; many start mid-conversation (`gpt` first) |
| `too_long` | 613 | > 1024 tokens after keeping at most 6 turns (PLAN.md: `max_seq_length` 512–1024) |
| `non_english` | 6 | < 90% ASCII letters |

CodeAlpaca rows are `instruction` (+ `input`) → `output`. GSM8K rows are `question` → `answer`, keeping the
worked solution and the final `#### <number>` line: that is the text the target generates, so it belongs in the
assistant turn for training and for vocab counting.

## Streams (`data/streams/`)

| stream | order | rows |
|---|---|---|
| `mixed` | all pools, one seeded shuffle (2021 domain runs) | 3000 |
| `ordered` | chat → code → math, each pool in its own order | 3000 |
| `domain_shift` | 600 chat → 300 code → 300 math (`SHIFT_SEGMENTS`) | 1200 |

The shift stream starts with a long chat run so the drafter settles before the first shift (E2 in BP-08).
Segment lengths are a constant in `mini_aurora/data/__init__.py`.

## Vocab mapping (`outputs/vocab/qwen3_24k_mixed.pt`)

Built from the assistant tokens of the **mixed** stream with Aurora's `generate_vocab_mapping` (top-24K by
frequency) and saved as `{"d2t", "t2d", "meta"}`, the format `mini_aurora.models.vocab` loads. Tokens outside
the draft vocab can never be drafted, so per-domain coverage is the number that matters
(`docs/concepts/vocabulary.md`). `--compare` also builds a chat-only mapping for comparison:

| mapping built from | chat | code | math |
|---|---|---|---|
| **mixed (all domains)**, shipped (24K) | 0.993 | 0.995 | 0.998 |
| chat only (24K) | 0.998 | 0.929 | 0.962 |
| mixed (16K, BP-02) | 0.971 | 0.977 | 0.992 |
| chat only (16K, BP-02) | 0.977 | 0.911 | 0.951 |

All-domains costs 0.6 points on chat and gains 6.6 on code and 3.6 on math (24K), so it is the one to use.

The mixed stream has 27.6K distinct assistant tokens and the top 16K cover 97.6% of occurrences, so only math
reached the 99% target from the concept notes at 16K. Coverage by draft vocab size (`draft_vocab_size` in
`examples/mac-qwen3-0.6b/draft_config.json`, a BP-03 setting; ranking by frequency, ties broken arbitrarily):

| size | chat | code | math | all |
|---|---|---|---|---|
| 16000 | 0.972 | 0.971 | 0.991 | 0.976 |
| 20000 | 0.985 | 0.985 | 0.996 | 0.987 |
| 24000 | 0.995 | 0.985 | 0.996 | 0.994 |
| 28000 | 1.000 | 1.000 | 1.000 | 1.000 |

These are in-sample numbers (the mapping is scored on the stream it was built from); held-out coverage will
be a little lower. **BP-04 switched the default to 24K** (every domain ≥ 99%): it costs ~8M more draft
`lm_head` parameters (1024 × 8K) and ~1 GB peak memory, recovered with bf16 + gradient checkpointing
(`mini_aurora/models/README.md`). The sweep table above used an earlier tie-breaking, hence small differences.

## Done when

- [x] three streams on disk (`data/streams/`)
- [x] cached vocab mapping (`outputs/vocab/qwen3_24k_mixed.pt`; 16K in BP-02) with per-domain coverage (`.json` next to it)
- [x] CPU only (streams: any machine; vocab: Apple Silicon, CUDA, or the CPU container)
