"""Online loop (BP-06 steps 5-6): serve stream requests, push their training samples into the pool, train, publish.

Sync mode alternates: serve ``serve_batch`` (B) requests through BP-05's ``serve(..., on_sample=pool.put)``, then
take ``train_steps`` (M) optimizer steps if the pool holds enough samples, publishing every ``sync_every`` (N)
steps. Before every request the server applies whatever is in the pending weight slot, so a publish reaches the
very next request and staleness is fixed by B, M and N.

Backpressure in sync mode: the loop is single-threaded, so a blocking ``put`` on a full pool would never return.
Instead, when a sample arrives and the pool is full, the loop trains first (Aurora's generation waits for
training); no sample is ever dropped.

Async mode runs a serving thread and a training thread that share the pool and the pending weight slot: the
server blocks on ``put`` while the pool is full (backpressure), the trainer blocks on ``take`` until one optimizer
step's worth of samples is there, and staleness (trainer version − served version) is whatever the two speeds
make it. When the stream ends the trainer stops after its current step; samples left in the pool stay untrained,
as in sync mode. If either thread fails, the pool is closed so the other one wakes up, and the error is re-raised.

The threads share one device, and PyTorch's MPS backend aborts when two threads issue work at once (one command
stream). So GPU work is serialized by a FIFO ``DeviceLock``: the server holds it for one request (weight swap +
decoding), the trainer for one optimizer step (+ snapshot). ``put`` / ``take`` happen outside it, so a full pool
can't deadlock. On one GPU, real overlap would not add throughput anyway; what async keeps is that the trainer
and the server interleave by their own speeds rather than a fixed B / M schedule, so Δ is measured, not set.

Each request is logged in BP-05's ``RequestLog`` format plus ``index`` (position in this run),
``trainer_version`` / ``trainer_version_end`` (trainer steps when the request started / finished; the served
version of each round is in ``versions``) and ``pool_size`` after its sample was added.
"""

import threading
import time

from mini_aurora.online.pool import PoolClosed, SamplePool
from mini_aurora.online.publish import WeightSlot, snapshot
from mini_aurora.online.trainer import OnlineTrainer
from mini_aurora.server.drafter import Drafter
from mini_aurora.server.log import RequestLog
from mini_aurora.server.serve import serve


class DeviceLock:
    """FIFO (ticket) lock: waiters get the device in arrival order, so neither thread starves the other."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._next_ticket = 0
        self._now_serving = 0

    def __enter__(self) -> "DeviceLock":
        with self._cond:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._cond.wait_for(lambda: self._now_serving == ticket)
        return self

    def __exit__(self, *exc) -> None:
        with self._cond:
            self._now_serving += 1
            self._cond.notify_all()


def mean_tau(records: list[dict]) -> float:
    """τ̄ over all rounds of ``records`` (round-weighted, as ``summarize``)."""
    rounds = [a for r in records for a in r["accepted"]]
    return sum(rounds) / len(rounds) if rounds else 0.0


class OnlineLoop:
    def __init__(
        self,
        target,
        tokenizer,
        drafter: Drafter,
        trainer: OnlineTrainer,
        pool: SamplePool,
        sync_every: int = 10,
        max_new_tokens: int = 128,
        greedy: bool = True,
        temperature: float = 1.0,
        seed: int = 0,
        check: bool = False,
        log: RequestLog | None = None,
        verbose: bool = False,
    ) -> None:
        self.target, self.tokenizer, self.drafter, self.trainer, self.pool = target, tokenizer, drafter, trainer, pool
        self.slot = WeightSlot()
        self.device_lock = DeviceLock()
        self.sync_every, self.max_new_tokens = sync_every, max_new_tokens
        self.greedy, self.temperature, self.seed, self.check = greedy, temperature, seed, check
        self.log, self.verbose = log, verbose
        self.records: list[dict] = []
        self.train_rows: list[dict] = []

    def serve_one(self, row: dict, index: int, put=None) -> dict:
        """Apply pending weights, serve one request, push its sample with ``put`` (default ``pool.put``)."""
        samples = []
        with self.device_lock:
            self.slot.apply(self.drafter)
            trainer_version = self.trainer.step_count
            (record,) = serve(
                self.target,
                self.drafter,
                self.tokenizer,
                [row],
                max_new_tokens=self.max_new_tokens,
                greedy=self.greedy,
                temperature=self.temperature,
                seed=self.seed + index,
                check=self.check,
                on_sample=samples.append,
            )
        (put or self.pool.put)(samples[0])  # outside the lock: may block on a full pool
        record = {"index": index, **record, "trainer_version": trainer_version,
                  "trainer_version_end": self.trainer.step_count, "pool_size": len(self.pool)}
        self.records.append(record)
        if self.log is not None:
            self.log.write(record)
        if self.verbose:
            match = f"  match={record['match']}" if self.check else ""
            print(f"req {index:4d}  {record['domain'] or '':4s}  {record['new_tokens']:4d} tok  "
                  f"τ̄ {record['mean_accepted']:.2f}  v{record['versions'][-1] if record['versions'] else self.drafter.version}"
                  f"/t{trainer_version}  pool {record['pool_size']:3d}  {record['tokens_per_s']:5.1f} tok/s{match}", flush=True)
        return record

    def train_one(self, timeout: float | None = None) -> dict:
        """One optimizer step from the pool; publish if ``sync_every`` steps have passed."""
        samples = self.pool.take(self.trainer.samples_per_step, timeout)
        with self.device_lock:
            row = self.trainer.step(self.pool, samples=samples)
            if self.trainer.step_count % self.sync_every == 0:
                self.slot.publish(snapshot(self.trainer.draft, self.trainer.step_count))
                row["published"] = True
        self.train_rows.append(row)
        if self.verbose:
            print(f"  step {row['step']:4d}  loss {row['loss']:.3f}  acc {' '.join(f'{x:.2f}' for x in row['acces'])}  "
                  f"pool {row['pool_size']:3d}  {row['step_s']:.1f}s  {row['mem_gb']:.1f} GB"
                  + ("  -> publish" if row.get("published") else ""), flush=True)
        return row

    def _put_sync(self, sample) -> None:
        while len(self.pool) >= self.pool.max_size:  # full pool => the threshold gate is open and a step fits
            self.train_one()
        self.pool.put(sample)

    def run_sync(self, rows: list[dict], serve_batch: int | None = None, train_steps: int = 1) -> dict:
        """Serve B requests, train M steps, repeat. B defaults to one optimizer step's worth of samples."""
        serve_batch = serve_batch or self.trainer.samples_per_step
        start = time.perf_counter()
        for i, row in enumerate(rows):
            self.serve_one(row, i, put=self._put_sync)
            if (i + 1) % serve_batch == 0:
                for _ in range(train_steps):
                    if not self.pool.can_take(self.trainer.samples_per_step):
                        break
                    self.train_one()
        return self.summary(time.perf_counter() - start)

    def run_async(self, rows: list[dict]) -> dict:
        """Serving and training in two threads until every row has been served."""
        stop, errors = threading.Event(), {}

        def serving() -> None:
            try:
                for i, row in enumerate(rows):
                    self.serve_one(row, i)
            except PoolClosed:
                pass  # the trainer failed and closed the pool; its error is raised below
            except BaseException as e:  # noqa: BLE001
                errors["serving"] = e
            finally:
                stop.set()

        def training() -> None:
            try:
                while not stop.is_set():
                    try:
                        self.train_one(timeout=0.5)
                    except TimeoutError:
                        continue
            except PoolClosed:
                pass
            except BaseException as e:  # noqa: BLE001
                errors["training"] = e
                self.pool.close()  # unblock a server waiting on a full pool

        start = time.perf_counter()
        threads = [threading.Thread(target=serving, name="serving"), threading.Thread(target=training, name="training")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for name, error in errors.items():
            raise RuntimeError(f"{name} thread failed") from error
        return self.summary(time.perf_counter() - start)

    def summary(self, seconds: float, window: int = 50) -> dict:
        records = self.records
        return {
            "requests": len(records),
            "train_steps": len(self.train_rows),
            "publishes": self.slot.publishes,
            "tau": round(mean_tau(records), 4),
            f"tau_first_{window}": round(mean_tau(records[:window]), 4),
            f"tau_last_{window}": round(mean_tau(records[-window:]), 4),
            "final_served_version": self.drafter.version,
            "final_trainer_version": self.trainer.step_count,
            "samples_left_in_pool": len(self.pool),
            "pool_put_wait_s": round(self.pool.put_wait_s, 2),
            "peak_mem_gb": round(self.trainer.peak_mem_gb, 2),
            "wall_s": round(seconds, 1),
        }
