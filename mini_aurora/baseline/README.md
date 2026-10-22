# [BP-04] Offline static EAGLE-3 baseline

Tracks #10. Branch: `bp/04-static-baseline`.

- No public EAGLE-3 draft for 0.6B, so train one: dump target hidden states on the **chat-only** split (mirrors `OfflineEagle3Dataset`), train 1–2 epochs, freeze.
- This checkpoint is both the *static* baseline and the init for *with-draft* mode.
- **Done when:** τ̄ clearly > 0 on held-out chat data in BP-05's loop.

## Run

```bash
source .venv/bin/activate && source mini_aurora/env.sh          # or: bash mini_aurora/docker/run_cpu.sh <cmd>
python -m mini_aurora.baseline.dump                              # chat pool -> outputs/baseline/hidden_states/{train,heldout}/
python -m mini_aurora.baseline.train                             # 2 epochs -> outputs/baseline/checkpoints/
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
| `train.py` | AdamW, lr 1e-4, warmup 1.5% + cosine, grad clip 0.5, loss Σ 0.8ⁱ·ploss_i, batch 2 × accumulation 4, bf16 draft + gradient checkpointing (BP-04 default, `--dtype fp32 --no-grad-ckpt` to disable); CSV log, per-epoch held-out eval and checkpoint | `Eagle3Trainer` + `TrainingConfig` defaults |
| `checkpoint.py` | `draft_stepNNNNNN.pt` (trainable weights + `d2t`/`t2d` + metadata, embedding excluded) and `latest.txt`; `load_draft` rebuilds from the target | `checkpoint.py` tracker file |
| `evaluate.py` | forward-only loss / accuracy per TTT position on a split | `eval_forward` |

## Held-out split

The last `HELDOUT = 100` rows of the chat pool (`chat-00900` … `chat-00999`) go to `heldout/` and are never
trained on. Per-epoch held-out loss and accuracy go to `eval.json`. Position-1 accuracy (draft argmax equals target
argmax, over assistant positions the draft vocab can produce) is the teacher-forced proxy for first-token acceptance
under greedy decoding; the done-when number, τ̄ on the live loop, comes from BP-05.

## Results

<!-- filled in after the full run -->
