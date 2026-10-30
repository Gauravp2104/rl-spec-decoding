# [BP-06] Online loop

Tracks #14. Branch: `bp/06-online-loop`.

- Bounded sample pool (`max_sample_pool_size`, `buffer_threshold`), BP-03 trainer, versioned weight publish every N steps.
- Modes: **sync** (serve B requests ↔ train M steps; fixed Δ) and **async** (two threads; real Δ).
- Starts: **day-0** (random draft) and **with-draft** (BP-04 checkpoint).
- **Done when:** a 300-request day-0 run on the mixed stream shows τ̄ rising.

## Progress

- [x] **1. Sample pool** (`pool.py`): `SamplePool`, thread-safe FIFO bounded by `max_sample_pool_size` (default 128).
  `put()` blocks while the pool is full (Aurora's `_await_pool_capacity` backpressure: generation waits, samples
  are never dropped); `take(n)` removes the n oldest. Samples are copied to CPU on `put` (a 1K-token fp32 sample is
  ~16 MB). `buffer_threshold` = minimum pool size before training starts (latched). Note: Aurora's
  `inference_buffer_threshold` is something else, the low watermark of the inference side's *prompt* buffer; its
  trainer starts as soon as the pool holds one dispatch batch.
- [x] **2. Online trainer** (`trainer.py`): `OnlineTrainer` owns its own draft (`build_draft` →
  `wrap_eagle3(gradient_checkpointing=True)`, fp32 weights as in BP-04), AdamW, grad clip 0.5, `weighted_loss`
  (0.8^i). `step()` takes `batch_size × accumulation` samples, right-pads each micro-batch with BP-04's `collate`,
  runs `forward_step` + backward. Samples are prepared like Aurora's `OfflineEagle3Dataset`: truncated to
  `max_seq_len` (1024) and the last position's loss mask zeroed (its target state would be padding). Constant LR,
  no warmup (Aurora's online configs): 1e-4 day-0, 1e-5 with-draft. Writes `train_log.csv`: step, loss, loss_i,
  acc_i, grad_norm, lr, pool_size, samples, tokens, step_s, mem_gb (MPS driver memory of the whole process).
- [x] **3. Weight publish** (`publish.py`): every N steps the trainer snapshots `trainable_state(draft)` (a copy)
  into a single pending slot, versioned by its step count; a newer publish replaces an unapplied one. Before each
  request the server applies it with `Drafter.load_weights(state, version=step, d2t, t2d)`.
- [x] **4. Start modes** (`start.py`): `day0` = one fresh draft, served copy starts from the same weights
  (version 0); `draft` = `Drafter.load_file(ckpt)` for serving and `load_draft(target, ckpt)` for the trainer, whose
  step counter starts at the checkpoint's `step`, so served and trainer versions are one counter.
- [x] **5. Sync mode** (`loop.py`): serve B requests through BP-05's `serve(..., on_sample=...)`, train M steps,
  publish when due. Single-threaded, so when a sample arrives at a full pool the loop trains first instead of
  blocking forever.
- [x] **6. Async mode**: a serving thread and a training thread share the pool and the slot. PyTorch's MPS backend
  aborts when two threads issue GPU work at once, so a FIFO `DeviceLock` serializes it (one request / one optimizer
  step at a time); `put` / `take` stay outside the lock. Each request record gets `trainer_version` (and
  `trainer_version_end`), so staleness = `trainer_version − versions[r]` per round.
- [x] **7. CLI**: writes `requests.jsonl` (BP-05 `RequestLog` lines + `index`, `trainer_version`,
  `trainer_version_end`, `pool_size`), `train_log.csv`, `config.json` and `summary.json`.
- [x] **8. Tests** (`mini_aurora/tests/test_online.py`, 16 tests, ~3 min): the pool blocks when full / keeps FIFO
  / gates on the threshold / stores on CPU; a publish makes the served weights equal the trainer's and bumps the
  version; both start modes; a tiny sync run; **fp32 greedy with `check=True` stays token-identical to target-only
  decoding across hot-swaps** (sync: 10 requests, 4 swaps; async: 10 requests); a failing trainer doesn't hang the
  server.
- [x] **9. 300-request day-0 run** (below).

## 300-request day-0 run

MacBook Air (16 GB), sync mode, first 300 requests of `data/streams/mixed.jsonl` (104 chat / 100 code / 96 math),
up to 128 new tokens (mean 114), K = 3, greedy. **fp32 target and served draft** (the CLI default, as in BP-05),
fp32 trainer with gradient checkpointing, batch 2 × accumulation 1 (one optimizer step per 2 requests, each sample
used once), LR 1e-4, publish every 5 steps, pool 128 (it never held more than 2). `--check` compares every output
with target-only greedy decoding.

```bash
caffeinate -i python -m mini_aurora.online.loop --mode sync --start day0 -n 300 --max-new-tokens 128 \
    --batch-size 2 --accumulation 1 --sync-every 5 --check --out outputs/online/day0_sync_300_fp32
```

**Lossless: 300/300 outputs token-identical to target-only greedy**, across 30 weight publishes.

| Requests | 1–50 | 51–100 | 101–150 | 151–200 | 201–250 | 251–300 |
|---|---|---|---|---|---|---|
| Rolling τ̄ (trailing 50) at the end of the block | 0.038 | 0.113 | 0.162 | 0.163 | 0.208 | **0.245** |

| | First 50 | Last 50 | All 300 |
|---|---|---|---|
| τ̄ | 0.039 | **0.246** | 0.152 |
| α1 / α2 / α3 | 0.038 / 0 / 0 | 0.215 / 0.028 / 0.003 | 0.141 / 0.009 / 0.001 |
| tok/s, speculative / target-only (`--check`) | 17.6 / 33.2 | 15.7 / 29.3 | 16.9 / 31.8 |

Per domain, τ̄ of the first vs last third of its requests: chat 0.050 → 0.126, code 0.074 → 0.226,
math 0.115 → 0.318. Training: 150 steps, 30 publishes, weighted loss 23.7 → 14.9 (mean of the first / last 10
steps), training accuracy 0.24 / 0.16 / 0.11 at the end; staleness 0–4 steps (mean 1.96). **Peak MPS 7.52 GB**,
median step 0.74 s; **wall time 54.6 min**: 33.7 min serving, 2.3 min training and ~18 min of `--check` reference
decoding. Metrics, plots and the speed-up estimate: `mini_aurora/eval/README.md` (BP-07).

Both throughputs drop ~12% over the hour (the fanless Air slowing under sustained load) while their ratio stays at
0.53–0.54, so compare tok/s within a run, not across runs; τ̄ is the hardware-independent number.

τ̄ climbs from 0 to ~0.25 in 300 requests (~34K completion tokens) and is still rising; BP-04's static draft
reaches 0.31 on held-out chat after 896 offline steps. The draft is still a slowdown (×0.53 of target-only), as
BP-05 found: a round costs ~68 ms ≈ 2.2 target-only steps, so breaking even needs τ̄ ≈ 1.2 with this cache-free
drafter.

An earlier run with `--bf16` (bf16 target and served draft, same settings, no `--check`) gave the same curve:
τ̄ 0.038 → 0.247 (first / last 50), 0.154 overall, peak MPS 6.04 GB, 39.1 min.

```bash
python -m pytest -q mini_aurora/tests/test_online.py                                         # 16 tests, ~3 min
python -m mini_aurora.online.loop --mode sync --start day0 -n 20 --max-new-tokens 32 --out outputs/online/dev
python -m mini_aurora.online.loop --mode async --start day0 -n 20 --max-new-tokens 32 --out outputs/online/dev_async
python -m mini_aurora.online.loop --mode sync --start draft --draft outputs/baseline/checkpoints_8ep -n 300 \
    --out outputs/online/draft_sync                                                          # LR 1e-5 by default
```

`--check` compares every output with target-only greedy (fp32, slow); `--overwrite` replaces an existing run in
`--out`; `--save-draft` saves the trainer's final draft in BP-04's checkpoint format. Memory knobs:
`--batch-size`, `--accumulation`, `--max-seq-len`, `--max-pool-size`.

### With-draft start

The BP-04 static draft (8 epochs = step 896, held-out acc 0.311, τ̄ 0.315 in BP-05's loop) is GitHub release
`bp04-static-v1` (`draft_step000896.pt` + `latest.txt`, so `--draft` takes the directory):

```bash
gh release download bp04-static-v1 -R Gauravp2104/rl-spec-decoding -D outputs/baseline/checkpoints_8ep
shasum -a 256 outputs/baseline/checkpoints_8ep/draft_step000896.pt   # 89f66983...b67b7e4d (release notes)
python -m mini_aurora.online.loop --mode sync --start draft --draft outputs/baseline/checkpoints_8ep \
    -n 20 --max-new-tokens 32 --out outputs/online/draft_smoke
```

Smoke run (MacBook Air, fp32, 20 mixed requests): served and trainer versions start at 896 (trainer steps 897,
898, ...), LR 1e-5, **τ̄ 0.26 from the first requests** (day-0 starts at 0), peak MPS 6.6 GB.
`test_start_draft_with_bp04_release` checks this when the file is present and is skipped otherwise.

### Done when

- [x] A 300-request day-0 run on the mixed stream shows τ̄ rising: τ̄ 0.039 (first 50) → 0.246 (last 50), fp32,
  300/300 lossless.
