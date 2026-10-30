"""Forward-only evaluation of a draft on a feature dump split.

    python -m mini_aurora.baseline.evaluate                                 # latest checkpoint, heldout split
    python -m mini_aurora.baseline.evaluate --ckpt outputs/baseline/checkpoints/draft_step000450.pt

Reports, per TTT position, the mean loss (cross-entropy against the target's pruned distribution) and
accuracy (draft argmax == target argmax, over assistant positions whose target token is in the draft vocab).
Position-1 accuracy is the teacher-forced proxy for the first-token acceptance rate under greedy decoding;
the real acceptance length comes from BP-05's serving loop.
"""

import argparse
import json
from pathlib import Path

import torch

from mini_aurora.baseline import CKPT_DIR, DUMP_DIR
from mini_aurora.baseline.checkpoint import latest_checkpoint, load_draft
from mini_aurora.baseline.data import make_loader, to_device
from mini_aurora.device import get_device, get_dtype
from mini_aurora.models.draft import forward_step, target_lm_head_weight, wrap_eagle3
from mini_aurora.models.target import load_target


@torch.no_grad()
def evaluate(eagle3, lm_head: torch.Tensor, loader, device: torch.device) -> dict:
    """Token-weighted mean loss and accuracy per TTT position over ``loader``."""
    eagle3.eval()
    length = eagle3.length
    loss_sum, acc_sum, weight = [0.0] * length, [0.0] * length, 0.0
    for features, attention_mask in loader:
        features, attention_mask = to_device(features, attention_mask, device)
        plosses, acces = forward_step(eagle3, features, lm_head, attention_mask)
        w = float(features.loss_mask.sum())
        for i in range(length):
            loss_sum[i] += plosses[i].item() * w
            acc_sum[i] += acces[i].item() * w
        weight += w
    eagle3.train()
    return {"loss": [s / max(weight, 1) for s in loss_sum], "acc": [s / max(weight, 1) for s in acc_sum],
            "assistant_tokens": int(weight), "batches": len(loader)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help=f"default: latest in {CKPT_DIR}")
    p.add_argument("--dump-dir", default=DUMP_DIR)
    p.add_argument("--split", default="heldout")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="write the result as JSON")
    args = p.parse_args()

    device = get_device(args.device)
    ckpt_path = Path(args.ckpt) if args.ckpt else latest_checkpoint(CKPT_DIR)
    if ckpt_path is None or not ckpt_path.exists():
        raise SystemExit(f"no checkpoint found ({args.ckpt or CKPT_DIR})")
    target = load_target(device=device)
    draft, ckpt = load_draft(target, ckpt_path, device=device)  # in the checkpoint's dtype, as trained
    eagle3 = wrap_eagle3(draft)
    loader = make_loader(Path(args.dump_dir) / args.split, args.batch_size, shuffle=False, feature_dtype=get_dtype(device))
    result = {"checkpoint": str(ckpt_path), "step": ckpt["step"], "split": args.split,
              **evaluate(eagle3, target_lm_head_weight(target), loader, device)}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
