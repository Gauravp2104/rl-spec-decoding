"""Online draft trainer (BP-06 step 2): BP-04's training step, fed from the sample pool instead of a dump.

The trainer owns its own draft (``build_draft`` -> ``wrap_eagle3(gradient_checkpointing=True)``), separate from
the served one: serving only sees the trainer's weights when they are published (step 3). Each ``step()`` takes
``batch_size * accumulation`` samples from the pool, right-pads each micro-batch with BP-04's ``collate``
(Aurora's ``DataCollatorWithPadding``), runs ``forward_step`` and backward, then one AdamW update with grad clip
0.5 on ``sum_i 0.8^i * ploss_i``.

LR is constant with no warmup, as in Aurora's online example configs (``lr_decay_style: constant``,
``warmup_ratio: 0.0``): 1e-4 from scratch (day-0), 1e-5 when starting from a trained draft.

Samples are prepared as Aurora's ``OfflineEagle3Dataset.process_data`` does: truncated to the first
``max_seq_len`` tokens, and the last position's loss mask zeroed. Position t learns the target distribution at
t + 1, which the last position of a sample does not have.

``train_log.csv`` gets one row per optimizer step: step, loss, loss_i, acc_i, grad_norm, lr, pool_size (after
the take), samples, tokens (loss positions), step_s and mem_gb (MPS driver memory, i.e. the whole process's
GPU footprint: target, served draft, trainer and activations).
"""

import csv
import time
from pathlib import Path

import torch

from mini_aurora.baseline.data import collate, to_device
from mini_aurora.device import get_device, memory_allocated_gb
from mini_aurora.models.draft import build_draft, forward_step, target_lm_head_weight, weighted_loss, wrap_eagle3
from mini_aurora.online.pool import SamplePool

from aurora.models.target.eagle3_target_model import Eagle3TargetOutput  # noqa: I001

DAY0_LR = 1e-4
DRAFT_LR = 1e-5


def gpu_memory_gb(device: torch.device) -> float:
    """MPS driver-allocated memory (what counts against the unified-memory budget); allocated memory elsewhere."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return memory_allocated_gb(device)


def collate_samples(
    samples: list[Eagle3TargetOutput], max_seq_len: int | None = None, feature_dtype: torch.dtype = torch.float32
) -> tuple[Eagle3TargetOutput, torch.Tensor]:
    """Online samples -> one right-padded batch ``(features, attention_mask)`` on CPU, via BP-04's ``collate``."""
    items = []
    for s in samples:
        n = s.input_ids.shape[1] if max_seq_len is None else min(s.input_ids.shape[1], max_seq_len)
        loss_mask = s.loss_mask[0, :n].clone()
        loss_mask[-1] = 0  # no target distribution after the last position
        items.append({
            "input_ids": s.input_ids[0, :n],
            "loss_mask": loss_mask,
            "aux_hidden_states": s.hidden_states[:, :n],
            "hidden_states": s.last_hidden_states[:, :n],
        })
    return collate(items, feature_dtype)


class OnlineTrainer:
    def __init__(
        self,
        target,
        vocab_mapping: tuple[torch.Tensor, torch.Tensor] | None = None,
        draft=None,
        lr: float = DAY0_LR,
        batch_size: int = 2,
        accumulation: int = 4,
        max_grad_norm: float = 0.5,
        max_seq_len: int | None = 1024,
        gradient_checkpointing: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        log_path: str | Path | None = None,
        step: int = 0,
    ) -> None:
        """``draft`` (e.g. from ``load_draft``) is trained in place; otherwise a day-0 draft is built."""
        self.device = device or get_device()
        if draft is None:
            if vocab_mapping is None:
                raise ValueError("A day-0 draft needs vocab_mapping")
            draft = build_draft(target, vocab_mapping, device=self.device, dtype=dtype)
        self.draft = draft
        self.eagle3 = wrap_eagle3(draft, gradient_checkpointing=gradient_checkpointing).train()
        self.lm_head = target_lm_head_weight(target)
        self.params = [p for p in self.eagle3.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.0)
        self.lr, self.batch_size, self.accumulation = lr, batch_size, accumulation
        self.max_grad_norm, self.max_seq_len = max_grad_norm, max_seq_len
        self.step_count = step  # optimizer steps so far; the version a publish carries
        self.peak_mem_gb = 0.0
        self._log = self._writer = None
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(path, "w", newline="")
            self._writer = csv.writer(self._log)
            length = self.eagle3.length
            self._writer.writerow(["step", "loss", *[f"loss_{i}" for i in range(length)],
                                   *[f"acc_{i}" for i in range(length)], "grad_norm", "lr", "pool_size",
                                   "samples", "tokens", "step_s", "mem_gb"])

    @property
    def samples_per_step(self) -> int:
        return self.batch_size * self.accumulation

    def step(self, pool: SamplePool | None = None, samples: list | None = None, timeout: float | None = None) -> dict:
        """One optimizer step on ``samples``, or on ``samples_per_step`` samples taken from ``pool``."""
        start = time.perf_counter()
        if samples is None:
            samples = pool.take(self.samples_per_step, timeout)
        micro_batches = [samples[i : i + self.batch_size] for i in range(0, len(samples), self.batch_size)]
        loss_sum, plosses_sum, acces_sum, tokens = 0.0, None, None, 0
        self.optimizer.zero_grad(set_to_none=True)
        for micro in micro_batches:
            features, attention_mask = collate_samples(micro, self.max_seq_len, self.lm_head.dtype)
            features, attention_mask = to_device(features, attention_mask, self.device)
            tokens += int(features.loss_mask.sum())
            plosses, acces = forward_step(self.eagle3, features, self.lm_head, attention_mask)
            loss = weighted_loss(plosses) / len(micro_batches)
            loss.backward()
            loss_sum += loss.item()
            pl = [x.item() / len(micro_batches) for x in plosses]
            ac = [x.item() / len(micro_batches) for x in acces]
            plosses_sum = pl if plosses_sum is None else [a + b for a, b in zip(plosses_sum, pl)]
            acces_sum = ac if acces_sum is None else [a + b for a, b in zip(acces_sum, ac)]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm).item()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step_count += 1
        mem = gpu_memory_gb(self.device)
        self.peak_mem_gb = max(self.peak_mem_gb, mem)
        row = {
            "step": self.step_count,
            "loss": loss_sum,
            "plosses": plosses_sum,
            "acces": acces_sum,
            "grad_norm": grad_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
            "pool_size": len(pool) if pool is not None else 0,
            "samples": len(samples),
            "tokens": tokens,
            "step_s": time.perf_counter() - start,
            "mem_gb": mem,
        }
        if self._writer is not None:
            self._writer.writerow([row["step"], f"{loss_sum:.4f}", *[f"{x:.4f}" for x in plosses_sum],
                                   *[f"{x:.4f}" for x in acces_sum], f"{grad_norm:.3f}", f"{row['lr']:.3e}",
                                   row["pool_size"], row["samples"], tokens, f"{row['step_s']:.2f}", f"{mem:.2f}"])
            self._log.flush()
        return row

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = self._writer = None
