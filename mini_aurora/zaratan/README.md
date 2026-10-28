# [BP-09] CUDA portability / scale check

Tracks #20. Branch: `bp/09-cuda-scale-check`.

Portability of `mini_aurora` beyond the Mac, in three parts:

| part | where | status |
|---|---|---|
| BP-04 on CUDA (0.6B) | Kaggle T4, `mini_aurora/baseline/kaggle/` | done: T4 numbers reproduce MPS (baseline README) |
| BP-05 server + test suite on CUDA | Kaggle T4, `kaggle/push_bp09.py tests` | done: 68/68 after one CUDA-tolerance fix (`results/tests/`) |
| Scale: Qwen3-1.7B target | Kaggle T4, `kaggle/push_bp09.py 1p7b` | done: τ̄ 0.319, 100/100 lossless (`results/1p7b/`, table below) |
| Online loop on CUDA | Kaggle T4, Zaratan A100 + MIG slice | done in BP-08 (#31): identical τ̄ on all three GPUs |
| Zaratan job scripts | this folder | written and tested for setup; jobs not run (see below) |

## Run on Kaggle (what produced the results)

```bash
python -m mini_aurora.baseline.kaggle.push_bp09 --user <u> tests                          # pytest on the T4, ~7 min
python -m mini_aurora.baseline.kaggle.push_bp09 --user <u> 1p7b --limit 20 --epochs 1    # smoke, ~3 min
python -m mini_aurora.baseline.kaggle.push_bp09 --user <u> 1p7b --epochs 8               # full run, ~2 h
```

Same `MINI_AURORA_TARGET` / `MINI_AURORA_DRAFT_CONFIG` overrides as the Zaratan job; results land in
`outputs/bp09_kaggle/<mode>/` and the committed copies are in `results/`.

## 1.7B on Zaratan (alternative, not run)

`MINI_AURORA_TARGET` and `MINI_AURORA_DRAFT_CONFIG` (BP-09 additions in `models/target.py` / `models/draft.py`)
switch the target and the draft config without code changes; everything else (24K vocab, aux layers 1/13/24,
TTT length 3, trainer defaults) stays as in BP-04. The 1.7B draft config (`examples/zaratan-qwen3-1.7b/`) is the
0.6B one with hidden 2048 / intermediate 6144 (Qwen3-1.7B's sizes); 1.7B also has 28 layers, so the aux layers
are unchanged.

```bash
ssh zaratan                                                   # MFA, once
mkdir -p ~/scratch.msai635/rl-spec-decoding && git clone https://github.com/Gauravp2104/rl-spec-decoding ~/scratch.msai635/rl-spec-decoding/repo && cd ~/scratch.msai635/rl-spec-decoding/repo
bash mini_aurora/zaratan/setup_env.sh                         # login node: uv + CPython 3.12 venv (Aurora needs >= 3.12; the cluster module is 3.10), data + models, all in scratch
sbatch mini_aurora/zaratan/bp09_1p7b.sbatch                   # smoke: 20 conversations, 1 epoch, ≤ 25 min walltime
sbatch -t 2:00:00 --export=ALL,LIMIT=,EPOCHS=8 mini_aurora/zaratan/bp09_1p7b.sbatch   # full run + held-out τ̄
squeue -u $USER; sacct -j <jobid> --format=JobID,Elapsed,AllocTRES%40,State -X
```

Cost: one A100 hour = 48 SU, charged for elapsed time. The smoke run is capped at 25 min (≤ 20 SU) and prints
its stage timings, from which the full run's cost is extrapolated before submitting it (expected 1–1.5 h,
50–75 SU). Jobs are charged to `msai635-class`. **Blocked (2026-10-26):** the `msai635` project scratch is over its 900K file-count quota (class-wide;
`beegfs-ctl --getquota --gid zt-msai635`), so no files can be created there until it is raised or cleaned. Outputs go to `~/scratch.msai635/rl-spec-decoding/outputs/1p7b*/` (checkpoint, eval.json,
evaluate.json, serve_heldout.*, timings.json); the 1.7B feature dump is deleted at the end of the job.

The Zaratan path was set up (environment, test-suite job on a MIG slice, 1.7B job on an A100) but the runs were
moved to Kaggle: the fp32 1.7B job does not fit a 4.75 GB MIG slice, the full-A100 queue (fairshare weight 0 for
`msai635-class`) waited for hours, and the project scratch was over its file quota. The Kaggle runs cost no SU.

## Results

### Test suite on CUDA (Kaggle T4, 2026-10-28)

68 tests, 429 s. First run: 67 passed, 1 failed, `test_server.py::test_target_forward_matches_capture_and_reference`.
The test compared the hidden states from `target_forward` and `capture_features` with `torch.equal`; on CUDA the two
forward paths round differently in fp32 (identical on CPU and MPS). Fixed by comparing with `allclose`
(atol/rtol 1e-4) on non-CPU devices; nothing in the serving or training path changed. Log: `results/tests/pytest.txt`.

### Qwen3-1.7B scale check (Kaggle T4, fp32, 2026-10-28)

BP-04's recipe unchanged (chat pool, 900 train / 100 held-out conversations, 24K vocab, aux layers 1/13/24,
8 epochs = 896 steps, fp32 weights + gradient checkpointing), then BP-05's held-out τ̄ with K = 3 and `--check`.
Smoke first (20 conversations, 1 epoch: dump 84 s, train 32 s, evaluate 20 s), then the full run.

| target | draft hidden / intermediate | held-out acc (1/2/3) | τ̄ | α1 / α2 / α3 | tok/s | lossless | dump / train / serve | s/step |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B (BP-04) | 1024 / 3072 | 0.311 / 0.271 / 0.248 | 0.315 | 0.243 / 0.058 / 0.014 | 23.2 | 100/100 | 418 / 2148 / 934 s | 2.4 |
| Qwen3-1.7B (BP-09) | 2048 / 6144 | 0.354 / 0.292 / 0.260 | **0.319** | 0.244 / 0.059 / 0.015 | 12.0 | 100/100 | 965 / 4976 / 1529 s | 5.6 |

- The same code, config and step count train a draft for a 3× larger target with no change; the whole job is
  one T4 kernel of 2 h 5 min.
- The 1.7B draft predicts its target's next token better (acc 0.35 vs 0.31) but serves the same τ̄ (0.32): the
  acceptance ceiling comes from the static draft and K = 3, not from the target size. Files: `results/1p7b/`.
- tok/s halves with the bigger target (every verification pass is a 1.7B forward); not compared to target-only here.
