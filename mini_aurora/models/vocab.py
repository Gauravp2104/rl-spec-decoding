"""Draft vocab mapping (``d2t`` / ``t2d``): build from conversations, cache to disk, check coverage.

Thin layer over Aurora's ``generate_vocab_mapping``, which keeps the ``draft_vocab_size`` most frequent
assistant tokens. The real mapping is built from BP-02's mixed stream by the CLI in ``mini_aurora.data.vocab``:

    python -m mini_aurora.data.vocab --compare             # → outputs/vocab/qwen3_24k_mixed.{pt,json}
"""

from pathlib import Path

import torch
from transformers import PreTrainedTokenizer

import mini_aurora  # noqa: F401  (sets MPS env vars before Aurora imports)

from aurora.data.preprocessing import generate_vocab_mapping, preprocess_conversations  # noqa: I001
from aurora.data.template import TEMPLATE_REGISTRY

DRAFT_VOCAB_SIZE = 24000
DEFAULT_VOCAB = "outputs/vocab/qwen3_24k_mixed.pt"


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


def assistant_ids(tokenizer: PreTrainedTokenizer, conversations: list, max_length: int = 2048) -> list:
    """Per conversation, the ``input_ids`` restricted to assistant tokens (1-D)."""
    data = tokenize_conversations(tokenizer, conversations, max_length=max_length)
    return [ids[mask.bool()] for ids, mask in zip(data["input_ids"], data["loss_mask"])]


def build_vocab_mapping(
    tokenizer: PreTrainedTokenizer,
    conversations: list,
    target_vocab_size: int,
    draft_vocab_size: int = DRAFT_VOCAB_SIZE,
    max_length: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(d2t, t2d)`` from the assistant-token frequencies of ``conversations``.

    ``target_vocab_size`` is the target's ``config.vocab_size`` (151,936 for Qwen3), not ``len(tokenizer)``
    (151,669): ``t2d`` must have one entry per target ``lm_head`` row.
    """
    # generate_vocab_mapping counts all tokens unless given a packed mask, so drop non-assistant tokens here.
    prompts = [{"input_ids": ids} for ids in assistant_ids(tokenizer, conversations, max_length)]
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


def load_vocab_mapping(path: str | Path = DEFAULT_VOCAB) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it with: python -m mini_aurora.data.vocab")
    state = torch.load(path, map_location="cpu")
    return state["d2t"], state["t2d"]
