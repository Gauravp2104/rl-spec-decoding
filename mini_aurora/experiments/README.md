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
| `online_day0`, `online_static` | BP-06 loop (#15, #29) and BP-07 metrics (#17), merged | runnable (`python -m mini_aurora.online.loop` under the hood) |

## Running the arms

One arm per GPU job. Three places have run them so far, all fp32 target and served draft, K = 3, 128 new tokens,
the frozen BP-04 draft from release `bp04-static-v1`:

- **Zaratan A100 MIG slices** (`GPU=a100_1g.5gb:1 PART=gpu-a100_1g.5gb`, 12 SU/h, no queue): the static arms
  (`target_only`, `static`, `day0`) fit the 5 GB slice; ~12–17 requests/min, so 1200 requests ≈ 1.3–1.7 h and
  3000 ≈ 3–5 h (set `-t` accordingly: the untrained `day0` arm is the slowest). Online arms do not fit a slice in fp32.
- **Zaratan full A100s** (`--gpus=a100:1`, 48 SU/h): the online arms; queue is first-come (fairshare weight 0).
- **Kaggle T4** (`python -m mini_aurora.experiments.kaggle --user <u> E2 online_static`): one arm at a time, fp32,
  16 GB; E2 online_static (1200 requests, 150 optimizer steps) took 88 min.

```bash
bash mini_aurora/zaratan/submit_bp08.sh                    # 15 jobs: E1-E3 (3 arms each) + E4 (6 settings), full streams
ONLY="E2 static" bash mini_aurora/zaratan/submit_bp08.sh   # one arm
squeue -u $USER                                            # progress; outputs in ~/rl-work/outputs/experiments/
python -m mini_aurora.experiments.table --out-dir ~/rl-work/outputs/experiments
```

Each Zaratan arm is one `bp08_arm.sbatch` job (`msai635-class`). Online arms use the loop's defaults (sync,
publish every 10 steps, batch 2 × accumulation 4, LR 1e-4 day-0 / 1e-5 with-draft) unless an E4 setting
overrides one of them. There is no checkpoint/resume: a job killed by its walltime leaves `requests.jsonl` but no
summary, and is rerun from the start.

## Results

<!-- table.py output goes here once the arms have run; per-experiment notes below -->

| exp | arm | stream | n | τ̄ | τ̄ first/last 100 | α1 / α2 / α3 | tok/s | × target-only | device | lossless | draft / loop args |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | target_only | mixed | 3000 | – | – | – | 33.0 | ×1.00 | NVIDIA A100 MIG 1g.5gb | – | – |
| E2 | online_static | domain_shift | 1200 | 0.375 | 0.319 / 0.385 | 0.276 / 0.079 / 0.020 | 26.8 | ×1.06 | Tesla T4 | – | checkpoints_8ep |
| E2 | static | domain_shift | 1200 | 0.327 | 0.315 / 0.242 | 0.246 / 0.065 / 0.016 | 25.1 | ×1.00 | Tesla T4 | – | draft_step000896.pt |
| E2 | static | domain_shift | 1200 | 0.327 | 0.315 / 0.242 | 0.246 / 0.065 / 0.016 | 26.9 | ×0.84 | NVIDIA A100 MIG 1g.5gb | – | draft_step000896.pt |
| E2 | target_only | domain_shift | 1200 | – | – | – | 25.2 | ×1.00 | Tesla T4 | – | – |
| E2 | target_only | domain_shift | 1200 | – | – | – | 32.1 | ×1.00 | NVIDIA A100 MIG 1g.5gb | – | – |
| E3 | static | ordered | 3000 | 0.326 | 0.315 / 0.258 | 0.244 / 0.066 / 0.016 | 28.6 | ×0.87 | NVIDIA A100 MIG 1g.5gb | – | draft_step000896.pt |
| E3 | target_only | ordered | 3000 | – | – | – | 32.8 | ×1.00 | NVIDIA A100 MIG 1g.5gb | – | – |

Results files: `results/<E>/<arm>_<stream>__<machine>.json` (the `summary.json` of each run; `requests.jsonl`
logs stay with the run directories). The same static arm run on a T4 and on an A100 slice gives identical τ̄, α
and per-window curves (greedy, same weights), so τ̄ is compared across machines and tok/s only within one.

### E1 — day-0 deployment (mixed stream)

Target-only reference: 33.0 tok/s (slice). `day0` (untrained draft, no training) hit its 4 h walltime at
2743/3000 requests and is rerun with 6 h; `online_day0` pending (Zaratan A100 queue). BP-06's 300-request fp32
day-0 run on the same stream (`mini_aurora/online/README.md`): τ̄ 0.039 → 0.246 over 300 requests.

### E2 — static vs online under domain shift (chat → code → math)

600 chat → 300 code → 300 math, both draft arms from the same frozen BP-04 draft, on a Kaggle T4:

| arm | τ̄ all | chat | code | math | τ̄ per 100 requests through the math shift |
|---|---|---|---|---|---|
| `static` (no training) | 0.327 | 0.324 | 0.419 | 0.259 | 0.44 → 0.26, 0.27, 0.24 |
| `online_static` (sync, N = 10, LR 1e-5) | **0.375** | 0.343 | 0.454 | 0.370 | 0.49 → 0.34, 0.39, 0.39 |

Online adaptation beats the static draft on every segment, and the gap opens at the shift: the static draft falls
to 0.26 on math and stays there, the online draft dips to 0.34 and recovers to 0.39 within 100 requests
(150 optimizer steps, 15 publishes). Math: +43 % τ̄ relative. Speed on the T4: target-only 25.2 tok/s, static 25.1
(×1.00), online 26.8 (×1.06); on the A100 slice the static draft is ×0.84 (26.9 vs 32.1), so the throughput side
is hardware-bound while τ̄ is not.

### E3 — ordered stream

1000 chat → 1000 code → 1000 math (slice). `static`: τ̄ 0.326 overall, chat 0.320 / code 0.422 / math 0.258, flat
within each block and equal to E2's per-segment numbers: the frozen draft's acceptance is a property of the domain,
not of the request order. Target-only 32.8 tok/s, static 28.6 (×0.87). `online_static` pending.

### E4 — sensitivity (publish interval, learning rate, sync vs async)

Six `online_day0` settings on the mixed stream (`--grid N5 | N20 | N50 | lr3e-4 | lr3e-5 | async`), pending.
