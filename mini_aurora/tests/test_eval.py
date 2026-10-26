"""BP-07: metrics on a small hand-written online log.

The hand-written log has 4 requests (K = 3, chat then code). Over all 8 rounds the accepted lengths are
[0, 1 | 3, 3 | 2 | 0, 0, 1], so τ̄ = 10 / 8, α = (5/8, 3/8, 2/8), acceptance rate = 10 / 24, and staleness per round
(trainer version − served version) is [0, 0 | 1, 1 | 0 | 3, 3, 3].
"""

import pytest

from mini_aurora.eval.metrics import (
    acceptance_rate,
    alpha_windows,
    domain_segments,
    estimated_speedup,
    load_run,
    loss_curve,
    rolling_tau,
    staleness,
    staleness_histogram,
    target_tokens_per_s,
    tau,
    tokens_per_s,
)
from mini_aurora.server.log import acceptance_by_position

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
