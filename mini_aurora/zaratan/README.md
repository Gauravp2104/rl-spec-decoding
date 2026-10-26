# [BP-09] CUDA portability / scale check on Zaratan

Tracks #20. Branch: `bp/09-cuda-scale-check`.

Portability of `mini_aurora` beyond the Mac, in three parts:

| part | where | status |
|---|---|---|
| BP-04 on CUDA (0.6B) | Kaggle T4, `mini_aurora/baseline/kaggle/` | done: T4 numbers reproduce MPS (baseline README) |
| BP-05 server + test suite on CUDA | Kaggle T4 | held-out τ̄ run via `push.py` (serve step); test suite pending |
| Scale: Qwen3-1.7B target on an A100 | UMD Zaratan, this folder | job script ready, not yet run |
| Online loop on CUDA | – | waits for BP-06 |

## 1.7B on Zaratan

`MINI_AURORA_TARGET` and `MINI_AURORA_DRAFT_CONFIG` (BP-09 additions in `models/target.py` / `models/draft.py`)
switch the target and the draft config without code changes; everything else (24K vocab, aux layers 1/13/24,
TTT length 3, trainer defaults) stays as in BP-04. The 1.7B draft config (`examples/zaratan-qwen3-1.7b/`) is the
0.6B one with hidden 2048 / intermediate 6144 (Qwen3-1.7B's sizes); 1.7B also has 28 layers, so the aux layers
are unchanged.

```bash
ssh zaratan                                                   # MFA, once
git clone https://github.com/Gauravp2104/rl-spec-decoding ~/rl-spec-decoding && cd ~/rl-spec-decoding
bash mini_aurora/zaratan/setup_env.sh                         # login node: venv in $HOME, data + models in scratch
sbatch mini_aurora/zaratan/bp09_1p7b.sbatch                   # smoke: 20 conversations, 1 epoch, ≤ 25 min walltime
sbatch -t 2:00:00 --export=ALL,LIMIT=,EPOCHS=8 mini_aurora/zaratan/bp09_1p7b.sbatch   # full run + held-out τ̄
squeue -u $USER; sacct -j <jobid> --format=JobID,Elapsed,AllocTRES%40,State -X
```

Cost: one A100 hour = 48 SU, charged for elapsed time. The smoke run is capped at 25 min (≤ 20 SU) and prints
its stage timings, from which the full run's cost is extrapolated before submitting it (expected 1–1.5 h,
50–75 SU). Outputs go to `~/scratch.msml612/rl-spec-decoding/outputs/1p7b*/` (checkpoint, eval.json,
evaluate.json, serve_heldout.*, timings.json); the 1.7B feature dump is deleted at the end of the job.

## Results

<!-- smoke timings, then the full-run table (held-out acc, τ̄, step time, peak memory) vs the 0.6B baseline -->
