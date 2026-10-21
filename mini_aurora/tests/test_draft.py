"""BP-03: draft config, vocab mapping and Eagle3Model wrapping on the local device (no training loop)."""

import json

import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from mini_aurora.models.draft import (
    DRAFT_CONFIG,
    TTT_LENGTH,
    build_draft,
    forward_step,
    param_counts,
    target_lm_head_weight,
    weighted_loss,
    wrap_eagle3,
)
from mini_aurora.models.samples import SAMPLE_CONVERSATIONS
from mini_aurora.models.target import DEFAULT_TARGET, capture_features, load_target, load_tokenizer
from mini_aurora.models.vocab import build_vocab_mapping, coverage, tokenize_conversations

needs_target = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(DEFAULT_TARGET, "config.json"), str),
    reason=f"{DEFAULT_TARGET} not in the HF cache (run python -m mini_aurora.models.query_target once)",
)


def test_draft_config_matches_plan():
    config = json.loads(DRAFT_CONFIG.read_text())
    assert config["architectures"] == ["LlamaForCausalLMEagle3"]
    assert (config["hidden_size"], config["num_attention_heads"], config["num_key_value_heads"]) == (1024, 16, 8)
    assert (config["intermediate_size"], config["num_hidden_layers"]) == (3072, 1)
    assert (config["vocab_size"], config["draft_vocab_size"]) == (151936, 16000)


@pytest.fixture(scope="module")
def setup():
    target, tokenizer = load_target(), load_tokenizer()
    vocab = build_vocab_mapping(tokenizer, SAMPLE_CONVERSATIONS, target.model.config.vocab_size)
    draft = build_draft(target, vocab)
    return target, tokenizer, vocab, draft, wrap_eagle3(draft)


@needs_target
def test_vocab_mapping_from_samples(setup):
    target, tokenizer, (d2t, t2d), _, _ = setup
    assert d2t.shape == (16000,) and t2d.shape == (target.model.config.vocab_size,)
    assert int(t2d.sum()) == 16000
    # d2t offsets map draft IDs onto exactly the kept target IDs.
    assert torch.equal(torch.arange(16000) + d2t, t2d.nonzero().squeeze(-1))
    data = tokenize_conversations(tokenizer, SAMPLE_CONVERSATIONS)
    for ids, mask in zip(data["input_ids"], data["loss_mask"]):
        assert coverage(t2d, ids, mask) == 1.0


@needs_target
def test_draft_embedding_copied_and_frozen(setup):
    target, _, _, draft, eagle3 = setup
    assert torch.equal(draft.embed_tokens.weight, target.model.get_input_embeddings().weight)
    assert not draft.embed_tokens.weight.requires_grad
    assert draft.lm_head.weight.shape == (16000, 1024) and draft.lm_head.weight.requires_grad
    assert draft.fc.weight.shape == (1024, 3072)
    assert draft.has_vocab_pruning and eagle3.vocab_pruning and eagle3.length == TTT_LENGTH
    trainable, frozen = param_counts(eagle3)
    assert frozen == target.model.get_input_embeddings().weight.numel()
    assert 30e6 < trainable < 40e6


@needs_target
def test_forward_backward_one_batch(setup):
    target, tokenizer, _, draft, eagle3 = setup
    data = tokenize_conversations(tokenizer, SAMPLE_CONVERSATIONS[:1])
    ids = data["input_ids"][0].to(target.model.device)
    features = capture_features(target, ids, data["loss_mask"][0])
    plosses, acces = forward_step(eagle3, features, target_lm_head_weight(target))
    assert len(plosses) == len(acces) == TTT_LENGTH
    assert all(torch.isfinite(loss) for loss in plosses)

    draft.zero_grad(set_to_none=True)
    weighted_loss(plosses).backward()
    assert draft.embed_tokens.weight.grad is None
    for name in ("fc", "lm_head"):
        grad = getattr(draft, name).weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    draft.zero_grad(set_to_none=True)
