"""EAGLE-3 draft for the Qwen3 target, wrapped in Aurora's ``Eagle3Model`` for TTT training.

Mirrors ``aurora/training/eagle3_trainer.py`` without FSDP / Ray / Mooncake:

- draft built from ``examples/mac-qwen3-0.6b/draft_config.json`` (1 layer, 24K draft vocab);
- embedding copied from the target and frozen; ``fc``, the decoder layer, ``norm`` and the
  24K-row draft ``lm_head`` are trained (the target ``lm_head`` only produces the labels);
- ``d2t`` / ``t2d`` vocab buffers set from a cached mapping;
- ``forward_step`` = ``Eagle3Trainer._forward``: shift tokens and target states by one, build pruned
  target probabilities, run the ``length``-step TTT unroll.
"""

from pathlib import Path

import torch

from mini_aurora.device import get_device, get_dtype  # sets MPS env vars before Aurora imports

from aurora.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel  # noqa: I001
from aurora.models.draft.base import Eagle3DraftModel
from aurora.models.eagle3 import Eagle3Model, compute_target_p_padded
from aurora.models.target.eagle3_target_model import Eagle3TargetOutput, HFTargetModel
from aurora.utils.tensor import padding

DRAFT_CONFIG = Path(__file__).resolve().parents[2] / "examples" / "mac-qwen3-0.6b" / "draft_config.json"
TTT_LENGTH = 3
# Aurora's per-position loss weights: 0.8 ** i.
PLOSS_DECAY = 0.8


def build_draft(
    target: HFTargetModel,
    vocab_mapping: tuple[torch.Tensor, torch.Tensor] | None = None,
    config_path: str | Path = DRAFT_CONFIG,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Eagle3DraftModel:
    """Fresh (day-0) draft with the target's embedding (frozen) and the given ``(d2t, t2d)``."""
    device = device or get_device()
    dtype = dtype or get_dtype(device)
    config = AutoDraftModelConfig.from_file(str(config_path))
    draft = AutoEagle3DraftModel.from_config(config, attention_backend="sdpa", torch_dtype=dtype)

    target_embedding = target.model.get_input_embeddings().weight
    if target_embedding.shape != draft.embed_tokens.weight.shape:
        raise ValueError(
            f"Draft embedding {tuple(draft.embed_tokens.weight.shape)} != target {tuple(target_embedding.shape)}"
        )
    with torch.no_grad():
        draft.embed_tokens.weight.copy_(target_embedding)
    draft.freeze_embedding()

    if vocab_mapping is not None:
        d2t, t2d = vocab_mapping
        draft.set_vocab_buffers(d2t, t2d)
    return draft.to(device)


def wrap_eagle3(
    draft: Eagle3DraftModel, length: int = TTT_LENGTH, gradient_checkpointing: bool = False
) -> Eagle3Model:
    """``Eagle3Model`` with SDPA attention.

    ``gradient_checkpointing`` recomputes the per-step loss (draft logits × 3 TTT steps) in backward instead of
    storing it: with a bf16 draft at 4 × 1K tokens, peak MPS drops from 9.3 to 7.2 GB for ~20% more step time.
    """
    return Eagle3Model(draft, length=length, attention_backend="sdpa", gradient_checkpointing=gradient_checkpointing)


def target_lm_head_weight(target: HFTargetModel) -> torch.Tensor:
    """Frozen target ``lm_head`` weight (151,936 × 1024); only used to build the labels."""
    return target.model.get_output_embeddings().weight.detach()


def forward_step(
    eagle3: Eagle3Model,
    features: Eagle3TargetOutput,
    lm_head_weight: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """One TTT forward pass on target features; returns per-position ``(plosses, acces)``.

    Position t is trained to predict token t+1, so ``input_ids`` and ``last_hidden_states`` are shifted
    left by one (Aurora's ``padding(left=False)``) while the aux ``hidden_states`` stay at t.
    """
    draft = eagle3.draft_model
    if not draft.has_vocab_pruning:
        raise ValueError("Draft vocab buffers are unset; pass vocab_mapping to build_draft")

    input_ids = padding(features.input_ids, left=False)
    target_hidden = padding(features.last_hidden_states, left=False)
    loss_mask = features.loss_mask
    if loss_mask.dim() == 3:
        loss_mask = loss_mask.squeeze(-1)
    if attention_mask is None:
        attention_mask = torch.ones_like(features.input_ids)

    target = compute_target_p_padded(
        target_hidden_states=target_hidden,
        target_lm_head_weight=lm_head_weight,
        t2d=draft.t2d,
        loss_mask=loss_mask,
        length=eagle3.length,
    )
    hidden_states = features.hidden_states.to(next(draft.fc.parameters()).dtype)
    plosses, _, acces = eagle3(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target=target,
        loss_mask=loss_mask,
        hidden_states=hidden_states,
    )
    return plosses, acces


def weighted_loss(plosses: list[torch.Tensor]) -> torch.Tensor:
    """Aurora's training objective: ``sum_i 0.8**i * ploss_i``."""
    return sum(PLOSS_DECAY**i * loss for i, loss in enumerate(plosses))


def param_counts(module: torch.nn.Module) -> tuple[int, int]:
    """``(trainable, frozen)`` parameter counts."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return trainable, frozen
