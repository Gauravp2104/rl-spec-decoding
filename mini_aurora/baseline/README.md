# [BP-04] Offline static EAGLE-3 baseline

Tracks #10. Branch: `bp/04-static-baseline`.

- No public EAGLE-3 draft for 0.6B, so train one: dump target hidden states on the **chat-only** split (mirrors `OfflineEagle3Dataset`), train 1–2 epochs, freeze (recorded baseline: 8 epochs, see Results).
- This checkpoint is both the *static* baseline and the init for *with-draft* mode.
- **Done when:** τ̄ clearly > 0 on held-out chat data in BP-05's loop.

## Run

```bash
source .venv/bin/activate && source mini_aurora/env.sh          # or: bash mini_aurora/docker/run_cpu.sh <cmd>
python -m mini_aurora.baseline.dump                              # chat pool -> outputs/baseline/hidden_states/{train,heldout}/
python -m mini_aurora.baseline.train                             # 8 epochs -> outputs/baseline/checkpoints/
python -m mini_aurora.baseline.evaluate                          # latest checkpoint on the heldout split
python -m pytest -q mini_aurora/tests/test_baseline.py           # 3 unit tests + 1 end-to-end smoke (needs the model cached)
```

Smoke test of the whole path on a few conversations: `dump --limit 20`, then `train --limit 20 --epochs 1`.

Needs BP-02's chat pool (`data/raw/chat.jsonl`) and vocab cache (`outputs/vocab/qwen3_24k_mixed.pt`), and BP-03's
target / draft modules. The feature dump for 1000 conversations is ~4 GB in bf16 (PLAN.md budgets ~5 GB).

## What each module does

| module | role | Aurora counterpart |
|---|---|---|
| `dump.py` | frozen target over each conversation once; saves `input_ids`, `loss_mask`, `aux_hidden_states` (layers 1/13/24), `hidden_states` (final norm) per conversation | offline data generation, `OfflineEagle3Dataset` keys |
| `data.py` | dataset over the dump, right-padding collator → `Eagle3TargetOutput` + attention mask, so BP-03's `forward_step` is used unchanged | `DataCollatorWithPadding` |
| `train.py` | AdamW, lr 1e-4, warmup 1.5% + cosine, grad clip 0.5, loss Σ 0.8ⁱ·ploss_i, batch 2 × accumulation 4, fp32 draft weights + gradient checkpointing (`--dtype bf16`, `--no-grad-ckpt`); CSV log, per-epoch held-out eval and checkpoint | `Eagle3Trainer` + `TrainingConfig` defaults |
| `checkpoint.py` | `draft_stepNNNNNN.pt` (trainable weights + `d2t`/`t2d` + metadata, embedding excluded) and `latest.txt`; `load_draft` rebuilds from the target | `checkpoint.py` tracker file |
| `evaluate.py` | forward-only loss / accuracy per TTT position on a split | `eval_forward` |

## Held-out split

The last `HELDOUT = 100` rows of the chat pool (`chat-00900` … `chat-00999`) go to `heldout/` and are never
trained on. Per-epoch held-out loss and accuracy go to `eval.json`. Position-1 accuracy (draft argmax equals target
argmax, over assistant positions the draft vocab can produce) is the teacher-forced proxy for first-token acceptance
under greedy decoding; the done-when number, τ̄ on the live loop, comes from BP-05.

## Results

Full run on a Kaggle T4 (`mini_aurora/baseline/kaggle/push.py`): chat pool, 900 train / 100 held-out conversations
(470,938 / 52,959 tokens; 366,622 / 41,535 assistant tokens), 24K vocab (`qwen3_24k_mixed.pt`), 2 epochs =
224 optimizer steps of batch 2 × accumulation 4, lr 1e-4 warmup + cosine, gradient checkpointing on. Held-out
loss is cross-entropy against the target's pruned distribution per TTT position (random init: ln 24000 = 10.09);
accuracy is draft argmax = target argmax.

| draft weights | held-out loss (pos 1 / 2 / 3) | held-out acc (pos 1 / 2 / 3) | train acc at end | step time | peak GPU |
|---|---|---|---|---|---|
| **fp32** (default) | **6.35 / 6.59 / 6.71** | **0.192 / 0.152 / 0.133** | 0.19 / 0.14 / 0.13 | 1.9 s | 2.4 GB |
| bf16 | 6.63 / 6.86 / 6.97 | 0.170 / 0.128 / 0.108 | 0.16 / 0.12 / 0.10 | 2.7 s | 1.8 GB |

Timings (fp32 run): install 16 s, dump 373 s (~1,260 tokens/s), train 470 s, evaluate 32 s. Outputs (checkpoint,
`train_log.csv`, `eval.json`, `evaluate.json`, `manifest.json`, `timings.json`) are in `outputs/baseline/kaggle/<label>/`.

- Both runs are still improving between epoch 1 and 2 (fp32: 0.171 → 0.192 position-1 accuracy) and train accuracy
  equals held-out accuracy, so the baseline is under-trained rather than overfitted; 224 steps is far below Aurora's
  schedule. More epochs are the lever if BP-05 needs a stronger day-0 draft.
- fp32 weights beat bf16 on every number and are faster on a T4 (bf16 is emulated there). With pure-bf16 AdamW,
  updates below ~0.4% of a weight's magnitude round away at lr 1e-4; Aurora avoids this by keeping fp32 master
  weights under FSDP mixed precision. Hence fp32 is the default here; the memory saving from bf16 is 0.6 GB.
- The checkpoint is the day-0 draft for BP-05 / BP-06. τ̄ (the done-when number) needs BP-05's loop.

### 8 epochs on MPS: the static baseline

Same data, vocab and settings on the M5 Pro (24 GB, MPS, fp32 draft, gradient checkpointing), `--epochs 8` =
896 optimizer steps; the warmup + cosine schedule stretches over the longer run. A 2-epoch MPS run reproduces the
T4 numbers above (held-out acc 0.171 → 0.192), so the two devices agree.

```bash
python -m mini_aurora.baseline.dump                                              # 70 s on MPS (900 + 100 conversations)
caffeinate -i python -m mini_aurora.baseline.train --epochs 8 --out outputs/baseline/checkpoints_8ep   # 19.4 min
python -m mini_aurora.baseline.evaluate --ckpt outputs/baseline/checkpoints_8ep/draft_step000896.pt
```

| epoch | step | held-out loss (pos 1 / 2 / 3) | held-out acc (pos 1 / 2 / 3) |
|---|---|---|---|
| 1 | 112 | 6.46 / 6.72 / 6.85 | 0.183 / 0.138 / 0.119 |
| 2 | 224 | 5.75 / 5.96 / 6.11 | 0.245 / 0.209 / 0.191 |
| 3 | 336 | 5.39 / 5.61 / 5.77 | 0.274 / 0.239 / 0.217 |
| 4 | 448 | 5.19 / 5.43 / 5.58 | 0.292 / 0.254 / 0.232 |
| 5 | 560 | 5.07 / 5.31 / 5.47 | 0.302 / 0.263 / 0.241 |
| 6 | 672 | 5.00 / 5.25 / 5.41 | 0.308 / 0.268 / 0.245 |
| 7 | 784 | 4.98 / 5.23 / 5.39 | 0.310 / 0.270 / 0.247 |
| **8** | **896** | **4.98 / 5.22 / 5.38** | **0.310 / 0.270 / 0.248** |

Median step 1.25 s, peak allocated memory 2.4 GB; train accuracy at the end 0.33 / 0.29 / 0.27.

- Held-out position-1 accuracy rises from 0.192 (2 epochs) to **0.310** and flattens after epoch 6, partly because
  the cosine schedule has decayed the learning rate to ~0 by then. Train is only ~2 points above held-out, so this
  is still not overfitting.
- Epoch 2 of this run (0.245) beats the 2-epoch run's final (0.192) at the same step count: with a longer schedule
  the learning rate stays higher for longer.

### Frozen static baseline: release `bp04-static-v1`

The 8-epoch checkpoint everyone downstream (BP-06 with-draft start, BP-08 E2/E3, the GRPO arm) should load is
published as a GitHub release, so all runs start from the same bytes:

```bash
gh release download bp04-static-v1 -R Gauravp2104/rl-spec-decoding -D outputs/baseline/checkpoints_8ep
shasum -a 256 outputs/baseline/checkpoints_8ep/bp04_static_draft_step000896.pt   # 89f66983d6d0cdc28e81ee9a34c29c1b5254c0e1b66a678a97da5c11b67b7e4d
python -m mini_aurora.server.serve -n 100 --heldout --check --draft outputs/baseline/checkpoints_8ep/bp04_static_draft_step000896.pt
```

It was trained on a Kaggle T4 with the defaults above (`mini_aurora/baseline/kaggle/push.py`): held-out acc
0.311 / 0.271 / 0.248, BP-05 τ̄ 0.315 (α 0.243 / 0.058 / 0.014), 100/100 lossless, reproducing the MPS numbers
below to three decimals. The release also carries `eval.json`, `train_log.csv`, the server check log and timings
(dump 418 s, train 2148 s, serve 934 s on the T4).

### Live acceptance on held-out chat (BP-05): done-when

BP-05's server (`bp/05-spec-server`) on the 100 held-out chat conversations (`chat-00900` … `chat-00999`), greedy,
K = 3, fp32 target and draft, up to 128 new tokens, with the target-only comparison:

```bash
python -m mini_aurora.server.serve -n 100 --heldout --draft outputs/baseline/checkpoints_8ep --check \
    --log outputs/server/heldout_checkpoints_8ep.jsonl
```

| draft | τ̄ | α1 / α2 / α3 | tokens per target pass | tok/s (speculative vs target only) | lossless |
|---|---|---|---|---|---|
| 2 epochs (step 224) | 0.181 | 0.164 / 0.016 / 0.001 | 1.19 | 29.1 vs 54.9 (×0.53) | 100/100 |
| **8 epochs (step 896)** | **0.313** | **0.242 / 0.058 / 0.013** | **1.32** | **32.1 vs 54.8 (×0.59)** | **100/100** |

τ̄ = mean guesses accepted per round; α_i = share of rounds whose first i guesses were all accepted.

- **Done-when met:** τ̄ is clearly > 0 (0.31), and every output is token-identical to target-only greedy decoding.
- **The 8-epoch checkpoint (`checkpoints_8ep/draft_step000896.pt`) is the static baseline** for BP-06 (with-draft
  start) and BP-08 (E2 / E3). It is frozen from here on.
- Acceptance collapses after the first guess (α2 = 0.06): a chain stops at the first miss, and guesses 2–3 are
  built on the draft's own hidden states. Teacher-forced accuracy (0.31 / 0.27 / 0.25) overstates live acceptance
  because it scores each position given the *true* previous tokens.
- Serving is still **slower** than the target alone (×0.59): a round takes ~41 ms (~2.3 target-only steps of
  18 ms) but yields 1.32 tokens. Breaking even needs τ̄ ≈ 1.3 at this round cost, and the static draft won't get
  there on its own; online training (BP-06) and a cheaper round (draft KV cache) are the levers.
