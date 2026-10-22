"""Train the static EAGLE-3 draft on a feature dump (Aurora's offline training, single device).

    python -m mini_aurora.baseline.train                            # 2 epochs on outputs/baseline/hidden_states/train
    python -m mini_aurora.baseline.train --limit 20 --epochs 1       # smoke test
    python -m mini_aurora.baseline.train --dtype bf16 --out outputs/baseline/checkpoints_bf16

Defaults follow Aurora's ``TrainingConfig`` where it applies: AdamW, lr 1e-4 (PLAN.md "scratch"), no weight
decay, 1.5% linear warmup then cosine decay, grad clip 0.5, loss ``sum_i 0.8^i * ploss_i``, batch 2 with
gradient accumulation 4 (PLAN.md micro-batch 1-2, accumulation 4). The draft starts from the target's
embedding (frozen) and the cached 24K vocab mapping; everything else is random (day-0). Draft weights are fp32
(Aurora keeps fp32 master weights under FSDP mixed precision; pure-bf16 AdamW trained worse, see README) with
gradient checkpointing on, as in mini_aurora/models/README.md.

Writes per-step metrics to ``<out>/train_log.csv``, a held-out evaluation per epoch to ``<out>/eval.json``,
and a checkpoint per epoch (``mini_aurora.baseline.checkpoint``).
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from mini_aurora.baseline import CKPT_DIR, DUMP_DIR
from mini_aurora.baseline.checkpoint import save_draft
from mini_aurora.baseline.data import make_loader, to_device
from mini_aurora.baseline.evaluate import evaluate
from mini_aurora.device import empty_cache, get_device, get_dtype, memory_allocated_gb
from mini_aurora.models.draft import build_draft, forward_step, param_counts, target_lm_head_weight, weighted_loss, wrap_eagle3
from mini_aurora.models.target import load_target
from mini_aurora.models.vocab import DEFAULT_VOCAB, load_vocab_mapping

from aurora.training.lr_scheduler import LRSchedulerWithWarmup  # noqa: I001


def train(
    dump_dir: str = DUMP_DIR,
    out_dir: str = CKPT_DIR,
    vocab_path: str = DEFAULT_VOCAB,
    epochs: int = 2,
    lr: float = 1e-4,
    batch_size: int = 2,
    accumulation: int = 4,
    warmup_ratio: float = 0.015,
    max_grad_norm: float = 0.5,
    bf16: bool = False,
    gradient_checkpointing: bool = True,
    limit: int | None = None,
    seed: int = 0,
    log_every: int = 10,
    device: torch.device | None = None,
) -> dict:
    torch.manual_seed(seed)
    device = device or get_device()
    feature_dtype = get_dtype(device)
    draft_dtype = torch.bfloat16 if bf16 else torch.float32
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    target = load_target(device=device)
    draft = build_draft(target, load_vocab_mapping(vocab_path), device=device, dtype=draft_dtype)
    eagle3 = wrap_eagle3(draft, gradient_checkpointing=gradient_checkpointing).train()
    lm_head = target_lm_head_weight(target)
    trainable, frozen = param_counts(eagle3)

    train_loader = make_loader(Path(dump_dir) / "train", batch_size, shuffle=True, seed=seed, limit=limit, feature_dtype=feature_dtype)
    heldout_dir = Path(dump_dir) / "heldout"
    eval_loader = make_loader(heldout_dir, batch_size, shuffle=False, limit=limit, feature_dtype=feature_dtype) if heldout_dir.exists() else None

    steps_per_epoch = max(1, len(train_loader) // accumulation)
    total_steps = steps_per_epoch * epochs
    params = [p for p in eagle3.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = LRSchedulerWithWarmup(optimizer, max_lr=lr, total_steps=total_steps,
                                      warmup_steps=int(warmup_ratio * total_steps), decay_style="cosine")
    config = {"dump_dir": dump_dir, "vocab": vocab_path, "epochs": epochs, "lr": lr, "batch_size": batch_size,
              "accumulation": accumulation, "warmup_ratio": warmup_ratio, "max_grad_norm": max_grad_norm,
              "draft_dtype": str(draft_dtype).split(".")[-1], "gradient_checkpointing": gradient_checkpointing, "device": device.type, "seed": seed, "limit": limit,
              "train_conversations": len(train_loader.dataset), "steps_per_epoch": steps_per_epoch,
              "total_steps": total_steps, "trainable_params": trainable, "frozen_params": frozen}
    print(json.dumps(config, indent=2))

    log = open(out / "train_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "epoch", "lr", "loss", *[f"loss_{i}" for i in range(eagle3.length)],
                     *[f"acc_{i}" for i in range(eagle3.length)], "grad_norm", "step_s", "mem_gb"])
    evals, step, start = [], 0, time.perf_counter()
    for epoch in range(epochs):
        micro, acc_loss, acc_pl, acc_ac = 0, 0.0, None, None
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        for features, attention_mask in train_loader:
            features, attention_mask = to_device(features, attention_mask, device)
            plosses, acces = forward_step(eagle3, features, lm_head, attention_mask)
            loss = weighted_loss(plosses) / accumulation
            loss.backward()
            acc_loss += loss.item()
            acc_pl = [x.item() / accumulation for x in plosses] if acc_pl is None else [a + x.item() / accumulation for a, x in zip(acc_pl, plosses)]
            acc_ac = [x.item() / accumulation for x in acces] if acc_ac is None else [a + x.item() / accumulation for a, x in zip(acc_ac, acces)]
            micro += 1
            if micro % accumulation:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            writer.writerow([step, epoch, f"{scheduler.get_last_lr()[0]:.3e}", f"{acc_loss:.4f}",
                             *[f"{x:.4f}" for x in acc_pl], *[f"{x:.4f}" for x in acc_ac],
                             f"{grad_norm:.3f}", f"{time.perf_counter() - step_start:.2f}", f"{memory_allocated_gb(device):.2f}"])
            if log_every and (step % log_every == 0 or step == total_steps):
                print(f"step {step:5d}/{total_steps}  epoch {epoch}  loss {acc_loss:.3f}  "
                      f"acc {' '.join(f'{x:.3f}' for x in acc_ac)}  lr {scheduler.get_last_lr()[0]:.2e}  "
                      f"{time.perf_counter() - step_start:.2f}s/step")
            acc_loss, acc_pl, acc_ac = 0.0, None, None
            step_start = time.perf_counter()
            log.flush()
        result = {"epoch": epoch, "step": step, "elapsed_s": round(time.perf_counter() - start, 1)}
        if eval_loader is not None:
            result["heldout"] = evaluate(eagle3, lm_head, eval_loader, device)
            print(f"epoch {epoch} heldout: loss {' '.join(f'{x:.3f}' for x in result['heldout']['loss'])}  "
                  f"acc {' '.join(f'{x:.3f}' for x in result['heldout']['acc'])}")
        evals.append(result)
        save_draft(draft, out, step, epoch=epoch, config=config, heldout=result.get("heldout"))
        (out / "eval.json").write_text(json.dumps({"config": config, "epochs": evals}, indent=2))
    log.close()
    empty_cache(device)
    return {"config": config, "epochs": evals}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump-dir", default=DUMP_DIR)
    p.add_argument("--out", default=CKPT_DIR)
    p.add_argument("--vocab", default=DEFAULT_VOCAB)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.015)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32", help="draft weights (default fp32)")
    p.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing")
    p.add_argument("--limit", type=int, default=None, help="use only the first N dump files per split (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()
    train(args.dump_dir, args.out, args.vocab, args.epochs, args.lr, args.batch_size, args.accumulation,
          args.warmup_ratio, args.max_grad_norm, args.dtype == "bf16", not args.no_grad_ckpt, args.limit, args.seed, args.log_every,
          device=get_device(args.device))


if __name__ == "__main__":
    main()
