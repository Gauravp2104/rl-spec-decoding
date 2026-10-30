"""One target forward pass that also returns the EAGLE-3 aux hidden states.

Same hooks as Aurora's ``HFTargetModel.generate_eagle3_data`` (outputs of the aux layers [1, 13, 24],
concatenated, plus the final-norm output), so the draft sees exactly the features it was trained on. Unlike
``generate_eagle3_data`` it also returns the logits and can run on top of an HF ``DynamicCache``:

- prefill: ``target_forward(target, prompt_ids)``;
- verify: ``target_forward(target, last_token + guesses, cache, num_logits=K + 1)``.
"""

from dataclasses import dataclass

import torch
from transformers import Cache

from aurora.models.target.eagle3_target_model import HFTargetModel


@dataclass
class TargetStep:
    logits: torch.Tensor  # (1, num_logits or T, V_target)
    aux_hidden: torch.Tensor  # (1, T, 3 * hidden): draft input, one row per input token
    last_hidden: torch.Tensor  # (1, T, hidden): final-norm output (training labels in BP-06)
    past_key_values: Cache | None


@torch.no_grad()
def target_forward(
    target: HFTargetModel,
    input_ids: torch.Tensor,
    past_key_values: Cache | None = None,
    num_logits: int = 0,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> TargetStep:
    """Forward ``input_ids`` (1, T) after the tokens already in ``past_key_values`` (if any).

    ``num_logits`` keeps only the last N positions' logits (0 = all); a 1K-token prefill would otherwise
    materialise a 1K × 151,936 logits tensor. ``attention_mask`` (additive, (1, 1, T, past + T)) and
    ``position_ids`` (1, T) override the causal default: tree drafting verifies several branches in one pass.
    """
    captured, handles = {}, []

    def hook(name):
        def fn(module, inputs, output):
            captured[name] = output[0] if isinstance(output, tuple) else output

        return fn

    layers = target._get_transformer_layers()
    for idx in target.aux_hidden_states_layers:
        handles.append(layers[idx].register_forward_hook(hook(idx)))
    handles.append(target._get_final_norm().register_forward_hook(hook("norm")))
    try:
        extra = {}
        if attention_mask is not None:
            past = past_key_values.get_seq_length() if past_key_values is not None else 0
            extra["attention_mask"] = attention_mask
            extra["position_ids"] = position_ids
            extra["cache_position"] = torch.arange(past, past + input_ids.shape[1], device=input_ids.device)
        out = target.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
            logits_to_keep=num_logits,
            **extra,
        )
    finally:
        for handle in handles:
            handle.remove()

    aux = torch.cat([captured[idx] for idx in target.aux_hidden_states_layers], dim=-1)
    return TargetStep(out.logits, aux, captured["norm"], out.past_key_values)
