"""BP-03 done-when: the draft memorises one fixed batch in 200 steps (~25 s on an M5 Pro)."""

import math

import pytest
from huggingface_hub import try_to_load_from_cache

from mini_aurora.models.overfit import run_overfit
from mini_aurora.models.target import DEFAULT_TARGET

pytestmark = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(DEFAULT_TARGET, "config.json"), str),
    reason=f"{DEFAULT_TARGET} not in the HF cache (run python -m mini_aurora.models.query_target once)",
)


def test_overfit_one_batch():
    result = run_overfit(steps=200, log_every=0)
    first, last = result.loss[0], result.final_loss
    assert all(math.isfinite(x) for row in result.loss for x in row)
    # Random init starts near ln(24000) ≈ 10.09 and must drop to the target-entropy floor.
    assert all(f > 9.0 for f in first)
    assert all(kl < 0.05 for kl in result.final_kl), result.final_kl
    assert all(loss < f / 5 for loss, f in zip(last, first))
    # Each sequence's last assistant position predicts past the end (no real label), so ~98% is the ceiling.
    assert all(acc >= 0.95 for acc in result.final_acc), result.final_acc
    assert result.peak_mps_gb < 8.0
