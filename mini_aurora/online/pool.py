"""Bounded sample pool between serving and training (BP-06 step 1).

Aurora's controller keeps finished requests in ``sample_pool`` (a deque); its inference manager stops dispatching
new requests while the pool holds ``max_sample_pool_size`` samples (``_await_pool_capacity``), so generation waits
for training and no sample is ever dropped. Here the same backpressure is a blocking ``put``: the serving side
stalls while the pool is full, and ``take(n)`` removes the n oldest samples (FIFO, as ``sample_pool.popleft()``).

Samples are moved to CPU on ``put``: a 1K-token fp32 sample is ~16 MB of aux + final-norm states, so 128 of them
would otherwise sit in the MPS allocator next to the target, the served draft and the trainer.

``buffer_threshold`` is the minimum pool size before training starts: ``take`` waits until the pool has held at
least that many samples once (the gate then stays open, so later takes only need ``n`` samples). This is *not*
Aurora's ``inference_buffer_threshold``, which is the low watermark of the inference side's prompt buffer
(``AsyncInferenceManager._refill_buffer``); Aurora's trainer starts as soon as the pool holds one dispatch batch.
"""

import dataclasses
import threading
import time
from collections import deque

import torch


class PoolClosed(Exception):
    """Raised by ``put`` after ``close``, and by ``take`` when the pool is closed with fewer than ``n`` samples."""


def to_cpu(sample):
    """Copy every tensor field of a sample (e.g. an ``Eagle3TargetOutput``) to CPU; other values pass through."""
    if isinstance(sample, torch.Tensor):
        return sample.detach().cpu()
    if dataclasses.is_dataclass(sample):
        return dataclasses.replace(sample, **{f.name: to_cpu(getattr(sample, f.name)) for f in dataclasses.fields(sample)})
    if isinstance(sample, dict):
        return {k: to_cpu(v) for k, v in sample.items()}
    return sample


class SamplePool:
    """Thread-safe FIFO of training samples, bounded by ``max_size`` (Aurora's ``max_sample_pool_size``)."""

    def __init__(self, max_size: int = 128, buffer_threshold: int = 1) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if not 1 <= buffer_threshold <= max_size:
            raise ValueError(f"buffer_threshold must be in [1, {max_size}], got {buffer_threshold}")
        self.max_size = max_size
        self.buffer_threshold = buffer_threshold
        self._samples: deque = deque()
        self._cond = threading.Condition()
        self._closed = False
        self._started = False
        self.total_put = 0
        self.total_taken = 0
        self.put_wait_s = 0.0  # time ``put`` spent blocked on a full pool (backpressure)

    def __len__(self) -> int:
        with self._cond:
            return len(self._samples)

    @property
    def started(self) -> bool:
        """True once the pool has reached ``buffer_threshold`` (training may start)."""
        with self._cond:
            return self._started

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def can_take(self, n: int) -> bool:
        """Whether ``take(n)`` would return without waiting (used by the single-threaded sync loop)."""
        with self._cond:
            return self._started and len(self._samples) >= n

    def put(self, sample, timeout: float | None = None) -> None:
        """Add one sample (copied to CPU), blocking while the pool is full. Never drops a sample."""
        sample = to_cpu(sample)
        with self._cond:
            start = time.perf_counter()
            if not self._cond.wait_for(lambda: self._closed or len(self._samples) < self.max_size, timeout):
                self.put_wait_s += time.perf_counter() - start
                raise TimeoutError(f"sample pool full ({self.max_size}) for {timeout}s")
            self.put_wait_s += time.perf_counter() - start
            if self._closed:
                raise PoolClosed("put on a closed sample pool")
            self._samples.append(sample)
            self.total_put += 1
            if len(self._samples) >= self.buffer_threshold:
                self._started = True
            self._cond.notify_all()

    def take(self, n: int, timeout: float | None = None) -> list:
        """Remove and return the ``n`` oldest samples, waiting for the threshold gate and ``n`` samples."""
        if not 1 <= n <= self.max_size:
            raise ValueError(f"n must be in [1, {self.max_size}], got {n}")  # more could never fit: deadlock
        with self._cond:
            ready = lambda: self._closed or (self._started and len(self._samples) >= n)  # noqa: E731
            if not self._cond.wait_for(ready, timeout):
                raise TimeoutError(f"fewer than {n} samples (or threshold {self.buffer_threshold} unmet) for {timeout}s")
            if not (self._started and len(self._samples) >= n):
                raise PoolClosed(f"sample pool closed with {len(self._samples)} < {n} samples")
            out = [self._samples.popleft() for _ in range(n)]
            self.total_taken += n
            self._cond.notify_all()
            return out

    def close(self) -> None:
        """Wake every waiter; later ``put`` calls fail, ``take`` drains what is left in full batches."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
