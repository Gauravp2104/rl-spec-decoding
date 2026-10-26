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
gh release download bp04-static-v1 -R Gauravp2104/rl-spec-decoding -D outputs/baseline/checkpoints_8ep   # frozen BP-04 draft
python -m mini_aurora.experiments.run E2 target_only                 # target-only reference over the shift stream
python -m mini_aurora.experiments.run E2 static --check              # BP-04 static draft, lossless check on
python -m mini_aurora.experiments.run E2 online_static               # BP-06 loop from the BP-04 draft (sync, N = 10)
python -m mini_aurora.experiments.run E1 online_day0 -n 300          # BP-06 loop from a random draft
python -m mini_aurora.experiments.run E4 online_day0 --grid N5       # one E4 setting: N5 / N20 / N50 / lr3e-4 / lr3e-5 / async
python -m mini_aurora.experiments.table                              # results table from outputs/experiments/
python -m pytest -q mini_aurora/tests/test_experiments.py
```

`run.py` runs one arm over one stream into `outputs/experiments/<E>/<arm>_<stream>[_<tag>]/`:

- static arms (`target_only`, `day0`, `static`) go through BP-05's `serve` (bs = 1, greedy, K = 3, 128 new
  tokens) and write `requests.jsonl` (the server log format);
- online arms (`online_day0`, `online_static`) run BP-06's loop (`python -m mini_aurora.online.loop --out <dir>`,
  `--start day0` or `--start draft --draft outputs/baseline/checkpoints_8ep`; extra loop arguments via `--grid`
  or `--loop-args`), so the directory also holds its `train_log.csv`, `config.json` and the loop's own
  `summary.json` is folded in. BP-07's metrics and plots read the same directory
  (`python -m mini_aurora.eval.plot <dir>`).

Every arm gets a `summary.json` with the same shape (overall, per domain, per 100 requests along the stream for
τ̄ vs request index), which `table.py` collects; speed-up is tokens/s relative to the `target_only` arm of the
same experiment and stream.

## Status

| arm | needs | status |
|---|---|---|
| `target_only` | BP-05 server | runnable |
| `static` | BP-04 release `bp04-static-v1` | runnable |
| `day0` | – | runnable |
| `online_day0`, `online_static` | BP-06 loop (#15) and BP-07 metrics (#17) merged | wired; `run.py` refuses them while `mini_aurora.online` is missing |

## Results

<!-- filled in by table.py once the arms have run -->
