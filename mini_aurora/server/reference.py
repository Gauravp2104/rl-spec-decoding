"""Correctness reference: greedy target-only decoding, one token per forward pass (HF ``model.generate``).

Speculative decoding must reproduce this output token for token. The equivalence test runs the target in
fp32 (``load_target(dtype=torch.float32)``): verifying K+1 tokens in one pass rounds differently from one
token per pass, and in bf16 that can flip near-tied argmaxes even when the algorithm is correct.
"""

import time
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer

from mini_aurora.models.target import _sync, build_chat_prompt

from aurora.models.target.eagle3_target_model import HFTargetModel  # noqa: I001


@dataclass
class ReferenceOutput:
    prompt_ids: torch.Tensor  # (1, n)
    new_ids: torch.Tensor  # (m,), includes the EOS token if one was generated
    seconds: float

    @property
    def tokens_per_s(self) -> float:
        return self.new_ids.numel() / self.seconds if self.seconds > 0 else 0.0


def encode_prompt(
    tokenizer: PreTrainedTokenizer, question: str, device: torch.device, thinking: bool = False
) -> torch.Tensor:
    """Qwen3 chat prompt for one user turn as ``input_ids`` (1, n)."""
    return tokenizer(build_chat_prompt(tokenizer, question, thinking), return_tensors="pt")["input_ids"].to(device)


def eos_token_ids(target: HFTargetModel) -> set[int]:
    """Token IDs that end generation (Qwen3: ``<|im_end|>`` and ``<|endoftext|>``)."""
    eos = target.model.generation_config.eos_token_id
    return set(eos if isinstance(eos, list) else [eos])


@torch.no_grad()
def reference_generate(target: HFTargetModel, input_ids: torch.Tensor, max_new_tokens: int = 128) -> ReferenceOutput:
    """Greedy decoding with HF ``generate``; stops at EOS or after ``max_new_tokens``."""
    model = target.model
    start = time.perf_counter()
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        pad_token_id=model.generation_config.pad_token_id,
    )
    _sync(model.device)
    return ReferenceOutput(input_ids, out[0, input_ids.shape[1] :], time.perf_counter() - start)
