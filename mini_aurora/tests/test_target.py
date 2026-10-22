"""BP-03: the Qwen3 target loads on the local device, generates, and yields EAGLE-3 aux features."""

import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from mini_aurora.device import get_device
from mini_aurora.models.target import AUX_LAYERS, DEFAULT_TARGET, capture_features, generate, load_target, load_tokenizer

pytestmark = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(DEFAULT_TARGET, "config.json"), str),
    reason=f"{DEFAULT_TARGET} not in the HF cache (run python -m mini_aurora.models.query_target once)",
)


@pytest.fixture(scope="module")
def target_and_tokenizer():
    return load_target(), load_tokenizer()


def test_target_is_frozen_on_device(target_and_tokenizer):
    target, _ = target_and_tokenizer
    assert target.model.device.type == get_device().type
    assert target.aux_hidden_states_layers == AUX_LAYERS
    assert not any(p.requires_grad for p in target.model.parameters())


def test_generate_greedy(target_and_tokenizer):
    target, tokenizer = target_and_tokenizer
    result = generate(target, tokenizer, "What is 2 + 2? Answer with one number.", max_new_tokens=8)
    assert result.new_tokens > 0
    assert "4" in result.text


def test_capture_features_shapes(target_and_tokenizer):
    target, tokenizer = target_and_tokenizer
    ids = tokenizer("Hello, world!", return_tensors="pt").input_ids.to(target.model.device)
    out = capture_features(target, ids)
    hidden = target.model.config.hidden_size
    assert out.hidden_states.shape == (1, ids.shape[1], 3 * hidden)
    assert out.last_hidden_states.shape == (1, ids.shape[1], hidden)
    assert torch.isfinite(out.hidden_states).all()
