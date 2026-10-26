"""Start modes for the online loop (BP-06 step 4): Aurora's no-draft and with-draft examples.

- ``day0``: one fresh draft for the trainer, and the served draft starts from the same weights (version 0), so the
  two copies only differ by what has not been published yet.
- ``draft``: a BP-04 checkpoint for both copies, ``Drafter.load_file`` for serving and ``load_draft`` for the
  trainer. The trainer's step counter starts at the checkpoint's ``step`` (the served version ``load_file``
  reports), so served and trainer versions stay one counter and staleness is their difference.

LR defaults follow PLAN.md: 1e-4 from scratch, 1e-5 with a draft.
"""

from pathlib import Path

import torch

from mini_aurora.baseline.checkpoint import load_draft
from mini_aurora.models.draft import build_draft
from mini_aurora.online.publish import snapshot
from mini_aurora.online.trainer import DAY0_LR, DRAFT_LR, OnlineTrainer
from mini_aurora.server.drafter import Drafter

MODES = ("day0", "draft")


def start(
    target,
    mode: str,
    vocab_mapping: tuple[torch.Tensor, torch.Tensor],
    draft_path: str | Path | None = None,
    k: int = 3,
    serve_dtype: torch.dtype = torch.float32,
    lr: float | None = None,
    device: torch.device | None = None,
    **trainer_kwargs,
) -> tuple[Drafter, OnlineTrainer]:
    """Served ``Drafter`` and ``OnlineTrainer`` for ``mode``; ``trainer_kwargs`` go to ``OnlineTrainer``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    device = device or target.model.device
    drafter = Drafter(build_draft(target, vocab_mapping, device=device, dtype=serve_dtype), k=k)
    if mode == "day0":
        trainer = OnlineTrainer(target, vocab_mapping, lr=DAY0_LR if lr is None else lr, device=device, **trainer_kwargs)
        snap = snapshot(trainer.draft, 0)
        drafter.load_weights(snap.state, version=0, d2t=snap.d2t, t2d=snap.t2d)
    else:
        if draft_path is None:
            raise ValueError("mode 'draft' needs draft_path (a BP-04 checkpoint file or directory)")
        path = drafter.load_file(draft_path)
        draft, ckpt = load_draft(target, path, device=device, dtype=trainer_kwargs.pop("dtype", torch.float32))
        trainer = OnlineTrainer(target, draft=draft, lr=DRAFT_LR if lr is None else lr, device=device,
                                step=ckpt["step"], **trainer_kwargs)
    return drafter, trainer
