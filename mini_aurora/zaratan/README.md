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

The BP-04 training run (same chat pool, 24K vocab, aux layers 1/13/24, 8 epochs, fp32 weights with gradient
checkpointing) with Qwen3-1.7B as the target, followed by BP-05's held-out τ̄ measurement (K = 3, lossless check on).
Smoke first (20 conversations, 1 epoch: dump 84 s, train 32 s, evaluate 20 s), then the full run.

| target | draft hidden / intermediate | held-out acc (1/2/3) | τ̄ | α1 / α2 / α3 | tok/s | lossless | dump / train / serve | s/step |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B (BP-04) | 1024 / 3072 | 0.311 / 0.271 / 0.248 | 0.315 | 0.243 / 0.058 / 0.014 | 23.2 | 100/100 | 418 / 2148 / 934 s | 2.4 |
| Qwen3-1.7B (BP-09) | 2048 / 6144 | 0.354 / 0.292 / 0.260 | **0.319** | 0.244 / 0.059 / 0.015 | 12.0 | 100/100 | 965 / 4976 / 1529 s | 5.6 |

- The same code trains a draft for a 3× larger target without changes; the whole job is one T4 kernel of
  about 2 hours.
- The 1.7B draft predicts its target's next token better (acc 0.35 vs 0.31) but serves the same τ̄ (0.32): the
  acceptance ceiling comes from the static draft and K = 3, not from the target size. Files: `results/1p7b/`.
- tok/s halves with the bigger target (every verification pass is a 1.7B forward); not compared to target-only here.

# [PT-02] Self-distilled 1.7B draft at ~27K conversations (B1)

Plan: `docs/pre_training/PLAN.md` §2, §4–§7. A job chain for arm B1, plus the PLAN §5 evaluation of A1 (PT-01,
release `pt01-a1-v1`) and B1 with the same harness. Environment layout as in `docs/zaratan_setup.md` (`~/rl-work`
for venv, repo, HF cache). Large outputs go under `$PT02` (default `~/scratch.msai635/rl-spec-decoding/pt02`).
Every path and knob is in `pt02_common.sh`.

```bash
bash mini_aurora/zaratan/pt02_prepare.sh                 # login node, once: 27K pool, 8 parts, A1 checkpoint, df check
SMOKE=1 bash mini_aurora/zaratan/submit_pt02.sh          # tiny end-to-end chain under $PT02/b1_smoke (~1 h of A100 in total)
bash mini_aurora/zaratan/submit_pt02.sh                  # full chain
DRY=1 bash mini_aurora/zaratan/submit_pt02.sh            # print the sbatch commands only
```

```
pt02_regen.sbatch  (array 0-7, 6 h)  ──afterok──> pt02_dump.sbatch (8 h) ──afterok──> pt02_train.sbatch (16 h) ──> pt02_eval b1
                                                                         └──afterok──> pt02_eval a1  (and a0 with A0=<BP-09 ckpt dir>)
```

| job | does | outputs (`$PT02/b1/`) |
|---|---|---|
| `pt02_prepare.sh` (login) | seeded download of all filtered ShareGPT chat; `extend_pool` drops the 1000 BP-02 rows and appends `chat-00900…00999` last; 8 contiguous parts; rows 800–999 of `data/raw/chat.jsonl` for a ShareGPT held-out dump; A1 checkpoint from the release | `../text/`, `../a1/` |
| `pt02_regen.sbatch` | `mini_aurora.data.regenerate` on one part: greedy, fp32, thinking off, ≤ 512 new tokens (as A1); resumable shards | `regen/part_NN.jsonl` |
| `pt02_dump.sbatch` | merges the parts (row order checked, held-out rows last), whole-pool stats; dumps the self-distilled pool and the ShareGPT held-out rows, packed into shards of 512 conversations (~110 files in all) | `chat_27k_self_1p7b{.jsonl,.stats.json}`, `dump_self/`, `dump_sharegpt_heldout/heldout/` |
| `pt02_train.sbatch` | BP-09 / A1 recipe, 4 epochs, seed 0; deletes `dump_self/train` when done | `checkpoints/` (+ `eval.json`, `train_log.csv`) |
| `pt02_eval.sbatch` | teacher-forced acc on both held-out dumps; `serve --check` on held-out chat, 100 code and 100 math rows (τ̄, α_i, lossless, target-only tok/s); with `TREES=1`, tree configs 2 / 4 | `eval/<a1\|b1\|a0>/` |

- **Resuming.** A timed-out regen task picks up at its last finished shard: run `FROM=regen` again, or
  `sbatch --array=<i> …`. A dump with a `manifest.json` is skipped; an interrupted dump resumes at its first unfinished shard (rerun `FROM=dump`). Training cannot resume, so delete
  `checkpoints/` and run `FROM=train`. `FROM=eval EVALS=b1` reruns one evaluation. `AFTER=<jobid>` makes the
  first stage wait for another job.
- **Memory.** `GEN_BATCH=64` assumes an 80 GB A100. On a 40 GB card, use `GEN_BATCH=32` (fp32 KV is ~230 KB per
  token). `GEN_BATCH` is not part of the resume config, so it can be changed between resubmits. `SHARD_SIZE` is
  part of it and must stay the same.
- **File quota.** The dumps are packed (`dump --shard-size $DUMP_SHARD_SIZE`, default 512): each shard is a raw
  `shard_NNNNN.bin` plus a small `shard_NNNNN.idx.pt`, so the full run writes ~110 dump files instead of ~27K
  `.ckpt` files. Training reads the shards through a memory map and shuffles per conversation as before.
  `DUMP_SHARD_SIZE` is part of the dump's resume config: keep it fixed for a run, or delete the dump.
- **Disk.** `dump_self/` peaks at ~240 GB on `$PT02`. `pt02_prepare.sh` prints `df` and the BeeGFS quota. If it
  doesn't fit, use the PLAN §7 fallback: set `PT02` elsewhere, or run on the first 9K rows.
- **Cost (estimate, to be replaced by the smoke and first-shard timings):** regen ~5–10 A100-h across the 8
  tasks, dump ~3 h, train ~6–8 h, eval ~1.5 h per draft. That is ~17–24 A100-h ≈ 800–1,150 SU at 48 SU/h.
  Walltimes are caps, and only elapsed time is charged.
