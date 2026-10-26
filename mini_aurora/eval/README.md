# [BP-07] Evaluation and metrics

Tracks #16. Branch: `bp/07-eval-metrics`.

- Rolling τ̄ vs. request index, per-position α_i, acceptance rate, tokens/s vs. target-only, estimated speed-up, staleness histogram, loss curve.
- CSV writer + plotting script; losslessness regression test.

## Progress

- [x] **1. Metrics** (`metrics.py`), recomputed from a BP-06 run's `requests.jsonl` and `train_log.csv`:
  τ̄ per request and rolling over the trailing W requests (round-weighted, as BP-05's `summarize`); α_i overall and
  per W-request window (BP-05's `acceptance_by_position`); acceptance rate = Σ accepted / (rounds × K); tok/s vs
  target-only (`reference_tokens_per_s` of `--check` runs, else the bench); estimated speed-up
  (1 + τ̄) / (1 + K·c); staleness Δ = trainer version − served version per round; the loss curve.
- [x] **Latency bench** (`bench.py`): c = draft-step / target-step latency with the run's dtype and K, on the first
  prompts of its stream. Target step = one cached single-token decode pass; draft step = one `Drafter.propose`
  of K guesses at the prompt's context ÷ K; also the K + 1-token verify pass and target-only tok/s
  (`reference_generate`). Writes `<run>/eval/bench.json`.
- [x] **2. CSV writer**: `<run>/eval/metrics.csv` (one row per request: its τ̄, acceptance rate, tok/s, served and
  trainer version, staleness, plus rolling τ̄ / acceptance rate / tok/s / α_i columns) and `<run>/eval/summary.json`
  (`summarize()`'s shape, one entry per domain and `all`, with extra keys: acceptance rate, staleness, target tok/s,
  measured and estimated speed-up, first/last-W τ̄, staleness histogram, c, final training loss/accuracy). Written
  under `<run>/eval/` so BP-06's own `summary.json` is never overwritten.
- [x] **3. Plots** (`plot.py`): rolling τ̄ with each request's domain as a strip underneath (contiguous segments
  read as blocks on ordered / domain-shift streams), α_i bars (first W, last W, all), staleness histogram, loss and
  per-position accuracy (raw steps faint, 10-step mean on top), tokens/s against target-only.
- [x] **4. Tests** (`mini_aurora/tests/test_eval.py`, 10 tests, ~1.5 min): every metric on a 4-request hand-written
  log with hand-computed values, the CSV / summary writer, the plots, the bench, and the **losslessness regression
  test**: a short fp32 greedy online run (8 requests, a publish after every step, sync and async) whose every output
  equals `reference_generate`.
- [x] **5. Results** on BP-06's 300-request day-0 run (below).

```bash
python -m mini_aurora.eval.bench outputs/online/day0_sync_300      # c and target-only tok/s; GPU otherwise idle
python -m mini_aurora.eval.plot outputs/online/day0_sync_300       # metrics.csv, summary.json and the plots
python -m mini_aurora.eval.metrics outputs/online/day0_sync_300 --window 25    # metrics only
python -m pytest -q mini_aurora/tests/test_eval.py
```

## BP-06 day-0 run: 300 mixed requests

Run settings in `mini_aurora/online/README.md` (MacBook Air, sync, bf16 target / served draft, K = 3, 150 optimizer
steps, publish every 5). Bench (bf16, 5 prompts): target step 24.2 ms, draft step 3.0 ms, **c = 0.123**, verify
pass 29.0 ms, target-only **36.6 tok/s**.

| | Requests | τ̄ | α1 / α2 / α3 | Acceptance rate | tok/s | vs target-only | Estimated speed-up | Δ mean / max |
|---|---|---|---|---|---|---|---|---|
| chat | 104 | 0.094 | 0.091 / 0.003 / 0.000 | 0.032 | 12.4 | ×0.34 | ×0.80 | 2.07 / 4 |
| code | 100 | 0.142 | 0.136 / 0.006 / 0.001 | 0.048 | 17.5 | ×0.48 | ×0.84 | 1.92 / 4 |
| math | 96 | 0.233 | 0.211 / 0.019 / 0.003 | 0.078 | 18.2 | ×0.50 | ×0.90 | 1.88 / 4 |
| **all** | **300** | **0.154** | 0.144 / 0.009 / 0.001 | 0.051 | 15.5 | ×0.42 | ×0.84 | 1.96 / 4 |

τ̄ over the first 50 requests 0.038, over the last 50 **0.247** (estimated speed-up ×0.91 at that τ̄).

![Rolling τ̄](figures/day0_sync_300/rolling_tau.png)

Rolling τ̄ rises over the whole run and is still rising at request 300. Math accepts most, chat least.

![Per-position acceptance](figures/day0_sync_300/alpha.png)

Almost all acceptance is the first guess: α1 goes 0.038 → 0.216 between the first and last 50 requests, α2 only
reaches 0.028. Training accuracy (below) shows positions 2 and 3 learning, but they are still at 0.16 / 0.11.

![Training loss and accuracy](figures/day0_sync_300/loss.png)

![Staleness](figures/day0_sync_300/staleness.png)

Sync mode with a step every 2 requests and a publish every 5 steps: the served draft is 0–4 steps behind, about
uniformly.

![Throughput](figures/day0_sync_300/tokens_per_s.png)

**Measured vs estimated speed-up.** The estimate (×0.84) counts a round as one target step plus K draft steps;
measured, a round takes ~74 ms ≈ 2.7 target-only steps (×0.42). The formula leaves out the verify pass over K + 1
tokens (29 ms vs a 24 ms decode step), the cache-free drafter re-reading a context that grows through the request
(the bench measures it at prompt length, 38–96 tokens, against 125 + 114 tokens on average in the run), and
per-round overhead outside the two models. τ̄ is the hardware-independent number; a drafter KV cache is the main
lever for the measured one.
