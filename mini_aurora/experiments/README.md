# [BP-08] Base-paper experiments

Tracks #18. Branch: `bp/08-experiments`.

| ID | Paper claim | Experiment | Stream | Arms |
|---|---|---|---|---|
| E1 | Day-0 deployment works | Day-0 online vs target-only | mixed | `target_only`, `day0`, `online_day0` |
| E2 | Online adaptation beats static under shift | Static BP-04 draft vs with-draft online, chat→code→math | domain_shift | `target_only`, `static`, `online_static` |
| E3 | Ordered-stream behaviour | E2 on the ordered stream | ordered | `target_only`, `static`, `online_static` |
| E4 | Sensitivity | Sync interval N ∈ {5, 20, 50}, LR, sync vs async | mixed | `online_day0` |

Results table + plots go back into PLAN.md.

## Run

```bash
python -m mini_aurora.experiments.run E2 target_only                 # target-only reference over the shift stream
python -m mini_aurora.experiments.run E2 static --check              # BP-04 static draft, lossless check on
python -m mini_aurora.experiments.run E1 day0 -n 300                 # random draft, no training
python -m mini_aurora.experiments.table                              # results table from outputs/experiments/
python -m pytest -q mini_aurora/tests/test_experiments.py
```

`run.py` serves one arm over one stream through BP-05's server (`serve`, bs = 1, greedy, K = 3, 128 new tokens)
and writes `outputs/experiments/<E>/<arm>_<stream>[_<tag>].jsonl` (one record per request, the server log
format) and `.json` (summary: overall, per domain, and per 100 requests along the stream for τ̄ vs request
index). `table.py` collects the summaries; speed-up is tokens/s relative to the `target_only` arm of the same
experiment and stream.

## Status

| arm | needs | status |
|---|---|---|
| `target_only` | BP-05 server | runnable |
| `static` | BP-04 checkpoint (`outputs/baseline/checkpoints_8ep`) | runnable |
| `day0` | – | runnable |
| `online_day0`, `online_static` | BP-06 online loop (#14) | refused by `run.py` until BP-06 lands |

The static halves of E2 / E3 and the target-only references for E1–E3 can be produced now; the online arms and
all of E4 wait for BP-06. BP-07's metrics (rolling τ̄, staleness, plots) read the same `.jsonl` logs.

## Results

<!-- filled in by table.py once the arms have run -->
