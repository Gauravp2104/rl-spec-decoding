"""BP-07: metrics, the metrics.csv / summary.json writer and the plots on a small hand-written online log, and a
losslessness regression test for the online loop.

The hand-written log has 4 requests (K = 3, chat then code). Over all 8 rounds the accepted lengths are
[0, 1 | 3, 3 | 2 | 0, 0, 1], so τ̄ = 10 / 8, α = (5/8, 3/8, 2/8), acceptance rate = 10 / 24, and staleness per round
(trainer version − served version) is [0, 0 | 1, 1 | 0 | 3, 3, 3].
"""

import csv
import json
from pathlib import Path

import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from mini_aurora.eval.metrics import (
    acceptance_rate,
    alpha_windows,
    domain_segments,
    estimated_speedup,
    evaluate_run,
    load_run,
    loss_curve,
    request_rows,
    rolling_tau,
    staleness,
    staleness_histogram,
    summarize_run,
    target_tokens_per_s,
    tau,
    tokens_per_s,
)
from mini_aurora.models.target import DEFAULT_TARGET, load_target, load_tokenizer
from mini_aurora.models.vocab import load_vocab_mapping
from mini_aurora.online.loop import OnlineLoop
from mini_aurora.online.pool import SamplePool
from mini_aurora.online.start import start
from mini_aurora.server.log import RequestLog, acceptance_by_position
from mini_aurora.server.serve import DEFAULT_STREAM, load_requests, summarize

RECORDS = [
    {"index": 0, "request_id": "a", "domain": "chat", "prompt_tokens": 10, "new_tokens": 5, "k": 3,
     "accepted": [0, 1], "versions": [0, 0], "seconds": 1.0, "tokens_per_s": 5.0, "trainer_version": 0},
    {"index": 1, "request_id": "b", "domain": "chat", "prompt_tokens": 12, "new_tokens": 8, "k": 3,
     "accepted": [3, 3], "versions": [1, 1], "seconds": 1.0, "tokens_per_s": 8.0, "trainer_version": 2},
    {"index": 2, "request_id": "c", "domain": "code", "prompt_tokens": 7, "new_tokens": 3, "k": 3,
     "accepted": [2], "versions": [2], "seconds": 0.5, "tokens_per_s": 6.0, "trainer_version": 2},
    {"index": 3, "request_id": "d", "domain": "code", "prompt_tokens": 9, "new_tokens": 4, "k": 3,
     "accepted": [0, 0, 1], "versions": [2, 2, 2], "seconds": 2.0, "tokens_per_s": 2.0, "trainer_version": 5},
]
for _r in RECORDS:
    _r["rounds"] = len(_r["accepted"])
    _r["mean_accepted"] = sum(_r["accepted"]) / len(_r["accepted"])

TRAIN = [
    {"step": 1, "loss": 25.0, "loss_0": 10.0, "loss_1": 10.0, "loss_2": 10.0, "acc_0": 0.0, "acc_1": 0.0, "acc_2": 0.0,
     "grad_norm": 14.0, "lr": 1e-4, "pool_size": 0, "samples": 2, "tokens": 200, "step_s": 1.2, "mem_gb": 5.0},
    {"step": 2, "loss": 20.0, "loss_0": 8.0, "loss_1": 8.0, "loss_2": 8.0, "acc_0": 0.1, "acc_1": 0.05, "acc_2": 0.0,
     "grad_norm": 9.0, "lr": 1e-4, "pool_size": 1, "samples": 2, "tokens": 180, "step_s": 1.0, "mem_gb": 5.1},
]


def test_tau_alpha_and_acceptance_rate():
    assert tau(RECORDS) == pytest.approx(10 / 8)
    assert acceptance_by_position(RECORDS) == pytest.approx([5 / 8, 3 / 8, 2 / 8])
    assert acceptance_rate(RECORDS) == pytest.approx(10 / 24) == pytest.approx(tau(RECORDS) / 3)
    assert acceptance_rate(RECORDS[1:2]) == 1.0


def test_rolling_tau_and_alpha_windows():
    assert rolling_tau(RECORDS, 2) == pytest.approx([1 / 2, 7 / 4, 8 / 3, 3 / 4])
    assert rolling_tau(RECORDS, 100)[-1] == pytest.approx(tau(RECORDS))
    windows = alpha_windows(RECORDS, 2)
    assert [(w["start"], w["end"]) for w in windows] == [(0, 2), (2, 4)]
    assert windows[0]["alpha"] == pytest.approx([3 / 4, 2 / 4, 2 / 4])
    assert windows[1]["alpha"] == pytest.approx([2 / 4, 1 / 4, 0])


def test_staleness_tokens_per_s_and_speedup():
    assert [staleness(r) for r in RECORDS] == [[0, 0], [1, 1], [0], [3, 3, 3]]
    assert staleness_histogram(RECORDS) == {0: 3, 1: 2, 3: 3}
    assert staleness({"versions": [3]}) == []  # plain BP-05 serve log: no trainer version
    assert tokens_per_s(RECORDS) == pytest.approx(20 / 4.5)
    assert estimated_speedup(1.25, 3, 0.1) == pytest.approx(2.25 / 1.3)
    assert estimated_speedup(0.0, 3, 0.0) == 1.0
    assert target_tokens_per_s(RECORDS) is None
    assert target_tokens_per_s(RECORDS, {"target_tokens_per_s": 12.0}) == 12.0
    checked = [dict(r, reference_tokens_per_s=v) for r, v in zip(RECORDS, (10.0, 14.0, 12.0, 12.0))]
    assert target_tokens_per_s(checked, {"target_tokens_per_s": 99.0}) == 12.0  # checked runs measure it per request


def test_domain_segments_and_loss_curve():
    assert domain_segments(RECORDS) == [(0, 2, "chat"), (2, 4, "code")]
    assert domain_segments([]) == []
    curve = loss_curve(TRAIN)
    assert curve["step"] == [1, 2] and curve["loss"] == [25.0, 20.0] and curve["acc_0"] == [0.0, 0.1]


def test_request_rows_and_summary_shape():
    rows = request_rows(RECORDS, window=2)
    assert len(rows) == 4
    assert [row["rolling_tau_2"] for row in rows] == pytest.approx([0.5, 1.75, 2.6667, 0.75], abs=1e-4)
    assert rows[3]["staleness_mean"] == 3 and rows[1]["staleness_max"] == 1 and rows[3]["served_version"] == 2
    assert rows[1]["rolling_alpha3_2"] == pytest.approx(0.5)

    summary = summarize_run(RECORDS, TRAIN, {"c": 0.1, "target_tokens_per_s": 10.0}, window=2)
    base = summarize(RECORDS)
    assert summary.keys() == base.keys() == {"chat", "code", "all"}
    for name in base:  # summarize()'s shape and values, plus extra keys
        assert {k: summary[name][k] for k in base[name]} == base[name]
    assert summary["all"]["acceptance_rate"] == pytest.approx(10 / 24, abs=1e-4)
    assert summary["chat"]["staleness_mean"] == 0.5 and summary["code"]["staleness_max"] == 3
    assert summary["all"]["estimated_speedup"] == pytest.approx(round(estimated_speedup(1.25, 3, 0.1), 3))
    assert summary["all"]["speedup_vs_target"] == pytest.approx(round(summary["all"]["tokens_per_s"] / 10.0, 3))
    assert summary["all"]["staleness_histogram"] == {0: 3, 1: 2, 3: 3}
    assert summary["all"]["tau_first_2"] == 1.75 and summary["all"]["tau_last_2"] == 0.75
    assert summary["all"]["train_steps"] == 2 and summary["all"]["final_acc"] == [0.1, 0.05, 0.0]
    assert summarize_run(RECORDS)["all"]["estimated_speedup"] is None  # no bench, no c


def test_evaluate_run_writes_csv_and_summary(tmp_path):
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    with open(tmp_path / "train_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TRAIN[0]))
        writer.writeheader()
        writer.writerows(TRAIN)
    (tmp_path / "config.json").write_text(json.dumps({"k": 3}))
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "bench.json").write_text(json.dumps({"c": 0.2, "target_tokens_per_s": 11.0}))

    run = load_run(tmp_path)
    assert len(run["records"]) == 4 and run["train"][1]["loss"] == 20.0 and run["bench"]["c"] == 0.2
    summary = evaluate_run(tmp_path, window=2)
    assert json.loads((tmp_path / "eval" / "summary.json").read_text()) == json.loads(json.dumps(summary))
    with open(tmp_path / "eval" / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert [row["request_id"] for row in rows] == ["a", "b", "c", "d"]
    assert {"tau", "acceptance_rate", "staleness_mean", "rolling_tau_2", "rolling_alpha1_2"} <= set(rows[0])
    assert summary["all"]["c"] == 0.2 and summary["all"]["target_tokens_per_s"] == 11.0


def test_plot_run_writes_every_figure(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    from mini_aurora.eval.plot import moving_average, plot_run

    assert moving_average([1, 3, 5, 7], 2) == [1, 2, 4, 6]
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    with open(tmp_path / "train_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TRAIN[0]))
        writer.writeheader()
        writer.writerows(TRAIN)
    paths = plot_run(tmp_path, window=2)
    assert [p.name for p in paths] == ["rolling_tau.png", "alpha.png", "staleness.png", "loss.png", "tokens_per_s.png"]
    assert all(p.stat().st_size > 1000 for p in paths)
    assert (tmp_path / "eval" / "metrics.csv").exists() and (tmp_path / "eval" / "summary.json").exists()


needs_models = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(DEFAULT_TARGET, "config.json"), str) or not Path(DEFAULT_STREAM).exists(),
    reason=f"{DEFAULT_TARGET} not in the HF cache or BP-02 stream missing",
)


@needs_models
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_online_run_is_lossless(mode, tmp_path):
    """Regression: every output of a short fp32 greedy online run equals ``reference_generate``, across hot-swaps."""
    torch.manual_seed(0)
    target, tokenizer = load_target(dtype=torch.float32), load_tokenizer()
    drafter, trainer = start(target, "day0", load_vocab_mapping(), lr=1e-3, batch_size=2, accumulation=1,
                             log_path=tmp_path / "train_log.csv")
    with RequestLog(tmp_path / "requests.jsonl") as log:
        loop = OnlineLoop(target, tokenizer, drafter, trainer, SamplePool(8, 2), sync_every=1, max_new_tokens=16,
                          check=True, log=log)
        rows = load_requests(DEFAULT_STREAM, 8)
        if mode == "sync":
            loop.run_sync(rows, serve_batch=2)
        else:
            loop.run_async(rows)
    trainer.close()
    summary = evaluate_run(tmp_path, window=4)
    assert summary["all"]["matches"] == "8/8"
    assert len({v for r in load_run(tmp_path)["records"] for v in r["versions"]}) >= 2  # weights were swapped

