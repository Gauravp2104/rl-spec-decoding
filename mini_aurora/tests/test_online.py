"""BP-06: sample pool (backpressure, threshold gate, CPU storage), the online trainer, weight publish,
the day-0 / with-draft start modes and the sync and async loops (lossless across
hot-swaps).

Model tests run the target and drafts in fp32, as BP-05's losslessness checks need.
"""

import csv
import gc
import threading
import time
from pathlib import Path

import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from mini_aurora.device import empty_cache, get_device
from mini_aurora.baseline.checkpoint import save_draft
from mini_aurora.models.draft import build_draft
from mini_aurora.models.target import DEFAULT_TARGET, load_target, load_tokenizer
from mini_aurora.models.vocab import load_vocab_mapping
from mini_aurora.online.loop import DeviceLock, OnlineLoop
from mini_aurora.online.pool import PoolClosed, SamplePool
from mini_aurora.online.publish import WeightSlot, snapshot
from mini_aurora.online.start import start
from mini_aurora.online.trainer import OnlineTrainer, collate_samples
from mini_aurora.server.drafter import Drafter, trainable_state
from mini_aurora.server.log import RequestLog, read_log
from mini_aurora.server.serve import DEFAULT_STREAM, load_requests, serve

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


def test_device_lock_is_fifo():
    lock, order = DeviceLock(), []
    lock.__enter__()  # held, so the workers queue up in start order

    def worker(i):
        with lock:
            order.append(i)

    threads = []
    for i in range(5):
        threads.append(threading.Thread(target=worker, args=(i,)))
        threads[-1].start()
        time.sleep(0.05)
    lock.__exit__(None, None, None)
    for t in threads:
        t.join(2)
    assert order == list(range(5))


@pytest.fixture(autouse=True, scope="module")
def release_device_memory():
    """Return this module's cached MPS memory when it ends: later tests (BP-03's overfit check) measure the
    process-wide driver allocation and would otherwise count the targets loaded here."""
    yield
    gc.collect()
    empty_cache(get_device())


needs_models = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(DEFAULT_TARGET, "config.json"), str) or not Path(DEFAULT_STREAM).exists(),
    reason=f"{DEFAULT_TARGET} not in the HF cache or BP-02 stream missing",
)


@pytest.fixture(scope="module")
def models():
    target, tokenizer = load_target(dtype=torch.float32), load_tokenizer()
    vocab = load_vocab_mapping()
    rows = load_requests(DEFAULT_STREAM, 4)
    drafter = Drafter(build_draft(target, vocab, dtype=torch.float32), k=3)
    samples = []
    serve(target, drafter, tokenizer, rows, max_new_tokens=16, on_sample=samples.append)
    return target, tokenizer, vocab, rows, samples


@needs_models
def test_collate_samples_pads_truncates_and_masks_last(models):
    samples = models[-1]
    lengths = [s.input_ids.shape[1] for s in samples[:2]]
    features, attention_mask = collate_samples(samples[:2])
    assert features.input_ids.shape == attention_mask.shape == (2, max(lengths))
    for b, n in enumerate(lengths):
        assert attention_mask[b, :n].all() and not attention_mask[b, n:].any()
        assert torch.equal(features.input_ids[b, :n], samples[b].input_ids[0].cpu())
        assert features.loss_mask[b, n - 1] == 0 and features.loss_mask[b, n - 2] == 1  # last position masked
        assert torch.equal(features.hidden_states[b, :n], samples[b].hidden_states[0].cpu())
    features, _ = collate_samples(samples[:1], max_seq_len=10)
    assert features.input_ids.shape == (1, 10)


@needs_models
def test_trainer_step_fits_pool_samples(models, tmp_path):
    target, _, vocab, _, samples = models
    torch.manual_seed(0)
    pool = SamplePool(max_size=8, buffer_threshold=4)
    trainer = OnlineTrainer(target, vocab, lr=1e-3, batch_size=2, accumulation=2, log_path=tmp_path / "train_log.csv")
    assert trainer.samples_per_step == 4
    losses = []
    for _ in range(12):
        for s in samples:
            pool.put(s)
        losses.append(trainer.step(pool)["loss"])
    trainer.close()
    assert len(pool) == 0 and trainer.step_count == 12
    assert losses[-1] < 0.7 * losses[0], losses
    with open(tmp_path / "train_log.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12 and rows[-1]["step"] == "12"
    assert {"loss_0", "loss_2", "acc_0", "acc_2", "grad_norm", "lr", "pool_size", "step_s", "mem_gb"} <= set(rows[0])
    assert float(rows[0]["lr"]) == 1e-3 and int(rows[0]["samples"]) == 4


def _same_weights(a, b) -> bool:
    sa, sb = trainable_state(a), trainable_state(b)
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@needs_models
def test_publish_swaps_trainer_weights_into_served_draft(models):
    target, _, vocab, _, samples = models
    torch.manual_seed(1)
    trainer = OnlineTrainer(target, vocab, lr=1e-3, batch_size=2, accumulation=2)
    drafter = Drafter(build_draft(target, vocab, dtype=torch.float32), k=3)
    slot = WeightSlot()
    assert slot.apply(drafter) is None and drafter.version == 0
    assert not _same_weights(trainer.draft, drafter.draft)

    trainer.step(samples=samples)
    slot.publish(snapshot(trainer.draft, trainer.step_count))
    assert slot.published_version == 1 and drafter.version == 0  # nothing changes until the server applies it
    assert slot.apply(drafter) == 1
    assert drafter.version == 1 and _same_weights(trainer.draft, drafter.draft)
    assert slot.apply(drafter) is None  # the slot is emptied by apply

    trainer.step(samples=samples)  # later trainer steps don't leak into the served copy
    assert not _same_weights(trainer.draft, drafter.draft) and drafter.version == 1

    trainer.step(samples=samples)
    slot.publish(snapshot(trainer.draft, 2))
    slot.publish(snapshot(trainer.draft, trainer.step_count))  # replaces the unapplied version 2
    assert slot.replaced == 1 and slot.apply(drafter) == 3
    assert drafter.version == 3 and _same_weights(trainer.draft, drafter.draft)


@needs_models
def test_start_day0_and_draft_modes(models, tmp_path):
    target, _, vocab, _, samples = models
    drafter, trainer = start(target, "day0", vocab, batch_size=2, accumulation=2)
    assert drafter.version == trainer.step_count == 0 and trainer.lr == 1e-4
    assert _same_weights(trainer.draft, drafter.draft)  # both copies start from the same fresh weights

    for _ in range(2):
        trainer.step(samples=samples)
    save_draft(trainer.draft, tmp_path, trainer.step_count)
    drafter, resumed = start(target, "draft", vocab, draft_path=tmp_path, batch_size=2, accumulation=2)
    assert drafter.version == resumed.step_count == 2 and resumed.lr == 1e-5
    assert _same_weights(trainer.draft, drafter.draft) and _same_weights(trainer.draft, resumed.draft)
    assert resumed.step(samples=samples)["step"] == 3  # versions continue from the checkpoint's step
    with pytest.raises(ValueError):
        start(target, "draft", vocab)


@needs_models
def test_sync_run_is_lossless_across_hot_swaps(models, tmp_path):
    target, tokenizer, vocab, _, _ = models
    torch.manual_seed(0)
    rows = load_requests(DEFAULT_STREAM, 10)
    drafter, trainer = start(target, "day0", vocab, lr=1e-3, batch_size=2, accumulation=1,
                             log_path=tmp_path / "train_log.csv")
    pool = SamplePool(max_size=8, buffer_threshold=2)
    with RequestLog(tmp_path / "requests.jsonl") as log:
        loop = OnlineLoop(target, tokenizer, drafter, trainer, pool, sync_every=1, max_new_tokens=16, check=True, log=log)
        summary = loop.run_sync(rows, serve_batch=2, train_steps=1)
    trainer.close()
    records = read_log(tmp_path / "requests.jsonl")
    assert len(records) == summary["requests"] == 10
    assert all(r["match"] for r in records), "greedy output must equal target-only decoding across hot-swaps"
    assert summary["train_steps"] == summary["publishes"] == 5
    assert drafter.version == 4 and loop.slot.published_version == 5  # the last publish came after the last request
    served = [r["versions"][0] for r in records if r["versions"]]
    assert served == sorted(served) and len(set(served)) >= 4  # the served draft changed between requests
    assert all(v == r["trainer_version"] for r in records for v in r["versions"])  # sync: a publish lands at once
    assert [r["index"] for r in records] == list(range(10))


@needs_models
def test_sync_full_pool_trains_instead_of_blocking(models):
    target, tokenizer, vocab, _, _ = models
    rows = load_requests(DEFAULT_STREAM, 6)
    drafter, trainer = start(target, "day0", vocab, batch_size=2, accumulation=1)
    pool = SamplePool(max_size=2, buffer_threshold=2)
    loop = OnlineLoop(target, tokenizer, drafter, trainer, pool, sync_every=100, max_new_tokens=8)
    summary = loop.run_sync(rows, serve_batch=100)  # the schedule never trains: only backpressure does
    assert summary["requests"] == 6 and summary["train_steps"] == 2 and len(pool) == 2
    assert pool.total_put == 6 and pool.total_taken == 4  # nothing dropped


@needs_models
def test_async_run_is_lossless_and_logs_staleness(models):
    target, tokenizer, vocab, _, _ = models
    torch.manual_seed(0)
    rows = load_requests(DEFAULT_STREAM, 10)
    drafter, trainer = start(target, "day0", vocab, lr=1e-3, batch_size=2, accumulation=1)
    pool = SamplePool(max_size=4, buffer_threshold=2)
    loop = OnlineLoop(target, tokenizer, drafter, trainer, pool, sync_every=1, max_new_tokens=16, check=True)
    summary = loop.run_async(rows)
    records = loop.records
    assert summary["requests"] == 10 and all(r["match"] for r in records)
    assert summary["train_steps"] >= 1 and summary["publishes"] == summary["train_steps"]
    assert pool.total_put == 10 and pool.total_taken + len(pool) == 10  # nothing dropped
    for r in records:
        assert r["trainer_version"] <= r["trainer_version_end"]
        assert all(0 <= r["trainer_version"] - v for v in r["versions"])  # the server never runs ahead


@needs_models
def test_async_trainer_failure_does_not_hang(models):
    target, tokenizer, vocab, _, _ = models
    rows = load_requests(DEFAULT_STREAM, 6)
    drafter, trainer = start(target, "day0", vocab, batch_size=2, accumulation=1)

    def broken_step(*args, **kwargs):
        raise FloatingPointError("boom")

    trainer.step = broken_step
    loop = OnlineLoop(target, tokenizer, drafter, trainer, SamplePool(max_size=2, buffer_threshold=2), max_new_tokens=8)
    with pytest.raises(RuntimeError, match="training thread failed"):
        loop.run_async(rows)
    assert len(loop.records) < 6  # serving stopped once the pool was closed
