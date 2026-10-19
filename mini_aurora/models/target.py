"""Frozen Qwen3 target model on the local device, wrapped in Aurora's ``HFTargetModel``.

``HFTargetModel.generate_eagle3_data`` hooks the three aux layers and the final norm, so the
same object serves plain generation (``target.model.generate``) and EAGLE-3 feature capture.
"""

import time
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from mini_aurora.device import get_device, get_dtype  # sets MPS env vars before Aurora imports

from aurora.models.target.eagle3_target_model import HFTargetModel  # noqa: I001

DEFAULT_TARGET = "Qwen/Qwen3-0.6B"
# Aurora's default for a 28-layer model: [1, L // 2 - 1, L - 4].
AUX_LAYERS = [1, 13, 24]


def load_tokenizer(model_path: str = DEFAULT_TARGET, **kwargs) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_path, **kwargs)


def load_target(
    model_path: str = DEFAULT_TARGET,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    aux_layers: list[int] | None = AUX_LAYERS,
    **kwargs,
) -> HFTargetModel:
    """Load a frozen, eval-mode target on ``device`` (default: ``get_device()``) with SDPA attention."""
    device = device or get_device()
    dtype = dtype or get_dtype(device)
    kwargs.setdefault("attn_implementation", "sdpa")
    target = HFTargetModel.from_pretrained(model_path, torch_dtype=dtype, device=str(device), **kwargs)
    target.model.eval().requires_grad_(False)
    target.set_aux_hidden_states_layers(aux_layers)
    return target


def build_chat_prompt(tokenizer: PreTrainedTokenizer, question: str, thinking: bool = False) -> str:
    """Qwen3 chat prompt for one user turn; thinking mode off by default (shorter, chat-style answers)."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )


@dataclass
class GenerationResult:
    prompt_tokens: int
    new_tokens: int
    seconds: float
    text: str

    @property
    def tokens_per_s(self) -> float:
        return self.new_tokens / self.seconds if self.seconds > 0 else 0.0


@torch.no_grad()
def generate(
    target: HFTargetModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    max_new_tokens: int = 128,
    thinking: bool = False,
) -> GenerationResult:
    """Greedy target-only decoding: the reference output and speed for BP-05/BP-07."""
    model = target.model
    inputs = tokenizer(build_chat_prompt(tokenizer, question, thinking), return_tensors="pt").to(model.device)
    start = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, top_p=None, top_k=None, temperature=None)
    _sync(model.device)
    seconds = time.perf_counter() - start
    new = out[0, inputs["input_ids"].shape[1] :]
    return GenerationResult(
        prompt_tokens=inputs["input_ids"].shape[1],
        new_tokens=new.shape[0],
        seconds=seconds,
        text=tokenizer.decode(new, skip_special_tokens=True),
    )


def capture_features(target: HFTargetModel, input_ids: torch.Tensor, loss_mask: torch.Tensor | None = None):
    """Run ``generate_eagle3_data`` on ``input_ids`` (full attention; loss mask defaults to all tokens)."""
    ones = torch.ones_like(input_ids)
    loss_mask = ones if loss_mask is None else loss_mask.to(input_ids.device)
    return target.generate_eagle3_data(input_ids=input_ids, attention_mask=ones, loss_mask=loss_mask)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


__all__ = [
    "AUX_LAYERS",
    "DEFAULT_TARGET",
    "GenerationResult",
    "build_chat_prompt",
    "capture_features",
    "generate",
    "load_target",
    "load_tokenizer",
]
