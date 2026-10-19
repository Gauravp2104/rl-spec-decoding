"""Draft vocab mapping (``d2t`` / ``t2d``): build from conversations, cache to disk, check coverage.

Thin layer over Aurora's ``generate_vocab_mapping``, which keeps the ``draft_vocab_size`` most frequent
assistant tokens. Until BP-02 lands, build a temporary mapping from the built-in samples:

    python -m mini_aurora.models.vocab                                   # → outputs/vocab/qwen3_16k_samples.pt
    python -m mini_aurora.models.vocab --jsonl stream.jsonl --out outputs/vocab/qwen3_16k_mixed.pt
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, PreTrainedTokenizer

import mini_aurora  # noqa: F401  (sets MPS env vars before Aurora imports)

from aurora.data.preprocessing import generate_vocab_mapping, preprocess_conversations  # noqa: I001
from aurora.data.template import TEMPLATE_REGISTRY

DRAFT_VOCAB_SIZE = 16000
DEFAULT_OUT = "outputs/vocab/qwen3_16k_samples.pt"


def tokenize_conversations(
    tokenizer: PreTrainedTokenizer,
    conversations: list,
    chat_template: str = "qwen",
    max_length: int = 2048,
) -> dict:
    """Aurora's preprocessing: ``input_ids`` (1, T) and assistant-only ``loss_mask`` (1, T) per conversation."""
    return preprocess_conversations(
        tokenizer,
        conversations,
        TEMPLATE_REGISTRY.get(chat_template),
        max_length=max_length,
        use_packed_loss_mask=False,
    )


def build_vocab_mapping(
    tokenizer: PreTrainedTokenizer,
    conversations: list,
    target_vocab_size: int,
    draft_vocab_size: int = DRAFT_VOCAB_SIZE,
    chat_template: str = "qwen",
    max_length: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(d2t, t2d)`` from the assistant-token frequencies of ``conversations``.

    ``target_vocab_size`` is the target's ``config.vocab_size`` (151,936 for Qwen3), not ``len(tokenizer)``
    (151,669): ``t2d`` must have one entry per target ``lm_head`` row.
    """
    data = tokenize_conversations(tokenizer, conversations, chat_template, max_length)
    # generate_vocab_mapping counts all tokens unless given a packed mask, so drop non-assistant tokens here.
    prompts = [{"input_ids": ids[mask.bool()]} for ids, mask in zip(data["input_ids"], data["loss_mask"])]
    return generate_vocab_mapping(prompts, target_vocab_size, draft_vocab_size)


def coverage(t2d: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor | None = None) -> float:
    """Fraction of (masked) tokens that the draft vocab can output. Check per domain; aim for ≥ 0.99."""
    ids = input_ids.flatten()
    if loss_mask is not None:
        ids = ids[loss_mask.flatten().bool()]
    return t2d.to(ids.device)[ids].float().mean().item() if ids.numel() else 0.0


def save_vocab_mapping(path: str | Path, d2t: torch.Tensor, t2d: torch.Tensor, **meta) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"d2t": d2t.cpu(), "t2d": t2d.cpu(), "meta": meta}, path)
    return path


def load_vocab_mapping(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    return state["d2t"], state["t2d"]


def _read_jsonl(paths: list[str]) -> list:
    """Each line holds a conversation under ``conversations`` or ``messages`` (role/content or from/value)."""
    conversations = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    conversations.append(row.get("conversations") or row.get("messages"))
    return conversations


def main() -> None:
    from mini_aurora.models.samples import SAMPLE_CONVERSATIONS
    from mini_aurora.models.target import DEFAULT_TARGET, load_tokenizer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_TARGET)
    parser.add_argument("--jsonl", action="append", help="repeatable; default: built-in samples")
    parser.add_argument("--draft-vocab-size", type=int, default=DRAFT_VOCAB_SIZE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    target_vocab_size = AutoConfig.from_pretrained(args.model).vocab_size
    conversations = _read_jsonl(args.jsonl) if args.jsonl else SAMPLE_CONVERSATIONS
    d2t, t2d = build_vocab_mapping(tokenizer, conversations, target_vocab_size, args.draft_vocab_size)
    source = args.jsonl or ["mini_aurora.models.samples"]
    path = save_vocab_mapping(args.out, d2t, t2d, source=source, model=args.model, conversations=len(conversations))
    print(f"Saved d2t {tuple(d2t.shape)}, t2d {tuple(t2d.shape)} ({int(t2d.sum())} kept) to {path}")


if __name__ == "__main__":
    main()
