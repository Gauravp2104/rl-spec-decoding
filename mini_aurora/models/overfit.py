"""BP-03 overfit sanity test: train a day-0 draft on one fixed batch until it memorises it.

The target features for the batch are captured once (the target is frozen), then the draft takes
``steps`` AdamW steps on that same batch. Passing means the whole data → target → draft → loss path is
wired correctly; it says nothing about generalisation (that is BP-04).

    python -m mini_aurora.models.overfit                         # 200 steps on 4 built-in samples
    python -m mini_aurora.models.overfit --steps 300 --lr 5e-4 --out outputs/overfit/run.json
    python -m mini_aurora.models.overfit --vocab outputs/vocab/qwen3_16k_mixed.pt \
        --stream data/streams/mixed.jsonl --domain chat          # real mapping + real chat-length batch

By default the batch and the vocab mapping both come from ``samples.py`` (no downloaded data needed).

The loss is forward KL as cross-entropy against the target's soft distribution, so its floor is the
target's entropy, not 0; ``kl`` (loss − entropy) is the part that should go to ~0, and accuracy to ~1.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from mini_aurora.device import empty_cache, get_device
from mini_aurora.models.draft import (
    build_draft,
    forward_step,
    param_counts,
    target_lm_head_weight,
    weighted_loss,
    wrap_eagle3,
)
from mini_aurora.models.samples import SAMPLE_CONVERSATIONS
from mini_aurora.models.target import capture_features, load_target, load_tokenizer
from mini_aurora.models.vocab import build_vocab_mapping, coverage, load_vocab_mapping, tokenize_conversations

from aurora.models.eagle3 import compute_target_p_padded  # noqa: I001
from aurora.utils.tensor import padding


@dataclass
class OverfitResult:
    steps: int
    lr: float
    batch_size: int
    seq_len: int
    assistant_tokens: int
    trainable_params: int
    frozen_params: int
    target_entropy: list[float]  # per TTT position: floor of the loss
    vocab: str = "samples"  # mapping file, or "samples" (built in memory)
    data: str = "samples"  # stream file (+ domain), or "samples"
    batch_coverage: float = 1.0  # assistant tokens of the batch inside the draft vocab
    loss: list[list[float]] = field(default_factory=list)  # [step][position]
    acc: list[list[float]] = field(default_factory=list)
    step_ms: list[float] = field(default_factory=list)
    peak_mps_gb: float = 0.0  # max driver-allocated memory seen after any step

    @property
    def final_loss(self) -> list[float]:
        return self.loss[-1]

    @property
    def final_acc(self) -> list[float]:
        return self.acc[-1]

    @property
    def final_kl(self) -> list[float]:
        return [loss - h for loss, h in zip(self.final_loss, self.target_entropy)]

    @property
    def median_step_ms(self) -> float:
        # Skip warm-up steps (kernel compilation / allocator growth).
        timed = sorted(self.step_ms[5:] or self.step_ms)
        return timed[len(timed) // 2]


def load_stream_conversations(path: str | Path, n: int, domain: str | None = None) -> list:
    """First ``n`` conversations (``messages``) of a BP-02 stream, optionally from one domain."""
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if domain is None or row["domain"] == domain:
                rows.append(row["messages"])
                if len(rows) == n:
                    break
    return rows


def make_batch(tokenizer, conversations: list, device: torch.device) -> dict[str, torch.Tensor]:
    """Right-pad to one batch with zeros, as Aurora's ``DataCollatorWithPadding`` does."""
    data = tokenize_conversations(tokenizer, conversations)
    seqs = list(zip(data["input_ids"], data["loss_mask"]))
    max_len = max(ids.shape[1] for ids, _ in seqs)
    batch = {"input_ids": [], "loss_mask": [], "attention_mask": []}
    for ids, mask in seqs:
        pad = max_len - ids.shape[1]
        batch["input_ids"].append(torch.nn.functional.pad(ids, (0, pad)))
        batch["loss_mask"].append(torch.nn.functional.pad(mask, (0, pad)))
        batch["attention_mask"].append(torch.nn.functional.pad(torch.ones_like(ids), (0, pad)))
    return {k: torch.cat(v).to(device) for k, v in batch.items()}


@torch.no_grad()
def target_entropy(features, lm_head_weight: torch.Tensor, t2d: torch.Tensor, length: int) -> list[float]:
    """Entropy of the pruned target distribution, averaged exactly as ``Eagle3Model`` averages the loss.

    Same target (``compute_target_p_padded``), same mask (its ``position_mask``) and the same per-step
    slicing / mask shift, so ``loss − entropy`` is the KL the draft can still remove.
    """
    loss_mask = features.loss_mask.squeeze(-1)
    target = compute_target_p_padded(
        padding(features.last_hidden_states, left=False), lm_head_weight, t2d, loss_mask, length
    )
    p = target.target_p_padded
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)  # (B, T + length)
    mask, seq_len, out = target.position_mask, loss_mask.shape[1], []
    for idx in range(length):
        step = entropy[:, idx : idx + seq_len]
        out.append(((step * mask).sum() / mask.sum().clamp_min(1)).item())
        mask = padding(mask, left=False)
    return out


def run_overfit(
    steps: int = 200,
    lr: float = 1e-3,
    batch_size: int = 4,
    draft_dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    log_every: int = 20,
    seed: int = 0,
    vocab_path: str | Path | None = None,
    stream: str | Path | None = None,
    domain: str | None = None,
) -> OverfitResult:
    """Overfit one batch. ``vocab_path`` / ``stream`` swap in a saved mapping / a BP-02 stream for the samples."""
    torch.manual_seed(seed)
    device = device or get_device()
    target, tokenizer = load_target(device=device), load_tokenizer()
    if stream:
        conversations = load_stream_conversations(stream, batch_size, domain)
    else:
        conversations = SAMPLE_CONVERSATIONS[:batch_size]
    if vocab_path:
        vocab = load_vocab_mapping(vocab_path)
    else:
        vocab = build_vocab_mapping(tokenizer, SAMPLE_CONVERSATIONS, target.model.config.vocab_size)
    draft = build_draft(target, vocab, device=device, dtype=draft_dtype)
    eagle3 = wrap_eagle3(draft).train()
    lm_head = target_lm_head_weight(target)

    batch = make_batch(tokenizer, conversations, device)
    features = capture_features(target, batch["input_ids"], batch["loss_mask"], batch["attention_mask"])
    trainable, frozen = param_counts(eagle3)
    result = OverfitResult(
        steps=steps,
        lr=lr,
        batch_size=len(conversations),
        seq_len=batch["input_ids"].shape[1],
        assistant_tokens=int(batch["loss_mask"].sum()),
        trainable_params=trainable,
        frozen_params=frozen,
        target_entropy=target_entropy(features, lm_head, draft.t2d, eagle3.length),
        vocab=str(vocab_path or "samples"),
        data=f"{stream}" + (f" ({domain})" if domain else "") if stream else "samples",
        batch_coverage=coverage(draft.t2d, batch["input_ids"], batch["loss_mask"]),
    )

    optimizer = torch.optim.AdamW([p for p in eagle3.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    for step in range(steps):
        start = time.perf_counter()
        plosses, acces = forward_step(eagle3, features, lm_head, batch["attention_mask"])
        loss = weighted_loss(plosses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eagle3.parameters(), 1.0)
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
            result.peak_mps_gb = max(result.peak_mps_gb, torch.mps.driver_allocated_memory() / 1e9)
        result.step_ms.append(1000 * (time.perf_counter() - start))
        result.loss.append([x.item() for x in plosses])
        result.acc.append([x.item() for x in acces])
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(
                f"step {step:4d}  loss {_fmt(result.loss[-1])}  acc {_fmt(result.acc[-1])}  "
                f"{result.step_ms[-1]:6.1f} ms"
            )

    del target, draft, eagle3, optimizer, features
    empty_cache(device)
    return result


def _fmt(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4, help=f"≤ {len(SAMPLE_CONVERSATIONS)} built-in samples")
    parser.add_argument("--vocab", default=None, help="saved mapping, e.g. outputs/vocab/qwen3_16k_mixed.pt")
    parser.add_argument("--stream", default=None, help="BP-02 stream JSONL to take the batch from")
    parser.add_argument("--domain", default=None, choices=["chat", "code", "math"], help="filter the stream")
    parser.add_argument("--bf16", action="store_true", help="bf16 draft weights (default fp32)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="write the full history as JSON")
    args = parser.parse_args()

    result = run_overfit(
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        draft_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device=get_device(args.device),
        vocab_path=args.vocab,
        stream=args.stream,
        domain=args.domain,
    )
    print(f"\nvocab {result.vocab}, data {result.data}, batch coverage {result.batch_coverage:.3f}")
    print(
        f"batch {result.batch_size} × {result.seq_len} tokens ({result.assistant_tokens} assistant), "
        f"{result.trainable_params / 1e6:.1f}M trainable / {result.frozen_params / 1e6:.1f}M frozen"
    )
    print(f"final loss {_fmt(result.final_loss)}  (floor = target entropy {_fmt(result.target_entropy)})")
    print(f"final KL   {_fmt(result.final_kl)}  final acc {_fmt(result.final_acc)}")
    print(f"step time  median {result.median_step_ms:.1f} ms   peak MPS {result.peak_mps_gb:.2f} GB")
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), indent=1))
        print(f"history → {path}")


if __name__ == "__main__":
    main()
