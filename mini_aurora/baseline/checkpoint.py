"""Save and load draft checkpoints (single file, versioned by step).

A checkpoint holds the draft's trainable weights, its vocab buffers and metadata. The frozen embedding is
not stored (0.6 GB in fp32): ``load_draft`` rebuilds the draft from the target and the saved vocab
buffers, then loads the weights. Layout, mirroring Aurora's tracker file:

    <dir>/draft_step000123.pt
    <dir>/latest.txt            # "123"
"""

import json
from pathlib import Path

import torch

from mini_aurora.models.draft import build_draft

EMBED_PREFIX = "embed_tokens."
TRACKER = "latest.txt"


def save_draft(draft, out_dir: str | Path, step: int, **meta) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in draft.state_dict().items() if not k.startswith(EMBED_PREFIX)}
    path = out_dir / f"draft_step{step:06d}.pt"
    torch.save({"state_dict": state, "d2t": draft.d2t.cpu(), "t2d": draft.t2d.cpu(), "step": step, "meta": meta}, path)
    (out_dir / TRACKER).write_text(str(step))
    (out_dir / f"draft_step{step:06d}.json").write_text(json.dumps({"step": step, **meta}, indent=2, default=str))
    return path


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    tracker = Path(ckpt_dir) / TRACKER
    if not tracker.exists():
        return None
    return Path(ckpt_dir) / f"draft_step{int(tracker.read_text().strip()):06d}.pt"


def load_draft(target, path: str | Path, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """Rebuild the draft from ``target`` and load the checkpoint at ``path``. Returns ``(draft, checkpoint)``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    draft = build_draft(target, (ckpt["d2t"], ckpt["t2d"]), device=device, dtype=dtype)
    missing, unexpected = draft.load_state_dict(ckpt["state_dict"], strict=False)
    missing = [k for k in missing if not k.startswith(EMBED_PREFIX)]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint {path} does not match the draft: missing={missing} unexpected={unexpected}")
    return draft, ckpt
