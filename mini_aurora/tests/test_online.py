"""BP-06: sample pool (backpressure, threshold gate, CPU storage)."""

import threading
import time

import pytest
import torch

from mini_aurora.device import get_device
from mini_aurora.online.pool import PoolClosed, SamplePool

from aurora.models.target.eagle3_target_model import Eagle3TargetOutput  # noqa: I001


def _wait_blocked(thread: threading.Thread, seconds: float = 0.2) -> bool:
    thread.join(seconds)
    return thread.is_alive()


def test_pool_put_blocks_when_full():
    pool = SamplePool(max_size=2)
    pool.put(0)
    pool.put(1)
    thread = threading.Thread(target=pool.put, args=(2,))
    thread.start()
    assert _wait_blocked(thread), "put on a full pool must block, not drop or overflow"
    assert len(pool) == 2
    assert pool.take(1) == [0]
    thread.join(2)
    assert not thread.is_alive()
    assert pool.take(2) == [1, 2]  # nothing dropped, FIFO order kept
    assert pool.put_wait_s > 0.1
    pool.put(3)
    pool.put(4)
    with pytest.raises(TimeoutError):
        pool.put(5, timeout=0.05)


def test_pool_take_removes_n_fifo():
    pool = SamplePool(max_size=8)
    for i in range(5):
        pool.put(i)
    assert pool.take(3) == [0, 1, 2]
    assert len(pool) == 2 and pool.total_put == 5 and pool.total_taken == 3
    with pytest.raises(ValueError):
        pool.take(9)  # can never fit
    with pytest.raises(TimeoutError):
        pool.take(3, timeout=0.05)


def test_pool_buffer_threshold_gates_first_take():
    pool = SamplePool(max_size=8, buffer_threshold=3)
    pool.put("a")
    pool.put("b")
    assert not pool.started and not pool.can_take(1)
    with pytest.raises(TimeoutError):
        pool.take(1, timeout=0.05)
    pool.put("c")
    assert pool.started and pool.can_take(3)
    assert pool.take(3) == ["a", "b", "c"]
    pool.put("d")  # below the threshold again, but training has started: the gate stays open
    assert pool.can_take(1) and pool.take(1) == ["d"]


def test_pool_stores_samples_on_cpu():
    device = get_device()
    sample = Eagle3TargetOutput(
        hidden_states=torch.randn(1, 4, 6, device=device),
        input_ids=torch.arange(4, device=device).view(1, 4),
        last_hidden_states=torch.randn(1, 4, 2, device=device),
        loss_mask=torch.ones(1, 4, dtype=torch.long, device=device),
    )
    pool = SamplePool()
    pool.put(sample)
    (out,) = pool.take(1)
    assert isinstance(out, Eagle3TargetOutput) and out.target is None
    for name in ("hidden_states", "input_ids", "last_hidden_states", "loss_mask"):
        assert getattr(out, name).device.type == "cpu"
        assert torch.equal(getattr(out, name), getattr(sample, name).cpu())


def test_pool_close_wakes_waiters():
    pool = SamplePool(max_size=1)
    errors = []

    def take():
        try:
            pool.take(1)
        except PoolClosed as e:
            errors.append(e)

    thread = threading.Thread(target=take)
    thread.start()
    assert _wait_blocked(thread)
    pool.close()
    thread.join(2)
    assert not thread.is_alive() and len(errors) == 1
    with pytest.raises(PoolClosed):
        pool.put(0)


def test_pool_producer_consumer_threads():
    pool, n, seen, sizes = SamplePool(max_size=4, buffer_threshold=2), 60, [], []

    def produce():
        for i in range(n):
            pool.put(i)
            sizes.append(len(pool))

    def consume():
        while len(seen) < n:
            seen.extend(pool.take(3 if n - len(seen) >= 3 else n - len(seen)))
            time.sleep(0.001)

    threads = [threading.Thread(target=produce), threading.Thread(target=consume)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert seen == list(range(n))
    assert max(sizes) <= 4
