"""BP-05 steps 1-6: reference, target step, chain drafter, verification, speculative loop with / without KV cache.

Run in fp32 (target and draft) so greedy speculative output can be compared token for token with ``generate``.
"""

import pytest
import torch
from huggingface_hub import try_to_load_from_cache
from transformers import DynamicCache

from mini_aurora.models.draft import build_draft, forward_step, target_lm_head_weight, weighted_loss, wrap_eagle3
from mini_aurora.models.target import DEFAULT_TARGET, capture_features, load_target, load_tokenizer
from mini_aurora.models.vocab import load_vocab_mapping
from mini_aurora.server.drafter import Drafter
from mini_aurora.server.reference import encode_prompt, eos_token_ids, reference_generate
from mini_aurora.server.spec_decode import speculative_generate
from mini_aurora.server.target_step import target_forward
from mini_aurora.server.verify import verify_greedy, verify_sampling

pytestmark = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(DEFAULT_TARGET, "config.json"), str),
    reason=f"{DEFAULT_TARGET} not in the HF cache (run python -m mini_aurora.models.query_target once)",
)


@pytest.fixture(scope="module")
def setup():
    target, tokenizer = load_target(dtype=torch.float32), load_tokenizer()
    vocab = load_vocab_mapping()
    prompt = encode_prompt(tokenizer, "Explain what a hash map is in three sentences.", target.model.device)
    ref = reference_generate(target, prompt, max_new_tokens=48)
    return target, vocab, prompt, ref


def test_reference_greedy(setup):
    target, _, prompt, ref = setup
    assert 0 < ref.new_ids.numel() <= 48
    assert eos_token_ids(target) == {151643, 151645}
    # Deterministic: a second run gives the same tokens.
    assert torch.equal(reference_generate(target, prompt, max_new_tokens=48).new_ids, ref.new_ids)


def test_target_forward_matches_capture_and_reference(setup):
    target, _, prompt, ref = setup
    full = target_forward(target, prompt)
    features = capture_features(target, prompt)
    assert torch.equal(full.aux_hidden, features.hidden_states)
    assert torch.equal(full.last_hidden, features.last_hidden_states)
    assert full.logits.shape == (1, prompt.shape[1], target.model.config.vocab_size)

    # Greedy decoding through target_forward on a DynamicCache reproduces model.generate.
    step = target_forward(target, prompt, DynamicCache(), num_logits=1)
    assert step.logits.shape[1] == 1
    out = []
    for _ in range(ref.new_ids.numel()):
        token = step.logits[0, -1].argmax()
        out.append(int(token))
        step = target_forward(target, token.view(1, 1), step.past_key_values, num_logits=1)
    assert out == ref.new_ids.tolist()


def test_drafter_proposal(setup):
    target, vocab, prompt, ref = setup
    drafter = Drafter(build_draft(target, vocab, dtype=torch.float32), k=3)
    aux = target_forward(target, prompt).aux_hidden
    p = drafter.propose(prompt, aux, ref.new_ids[0])
    assert p.tokens.shape == p.draft_ids.shape == (3,)
    assert p.probs.shape == (3, drafter.draft.lm_head.out_features)
    assert torch.allclose(p.probs.sum(-1), torch.ones(3, device=p.probs.device))
    assert drafter.draft.t2d[p.tokens].all()  # every guess is a real target ID inside the draft vocab
    assert torch.equal(p.draft_ids, p.probs.argmax(-1))

    def sample(seed):
        gen = torch.Generator(device=prompt.device).manual_seed(seed)
        return drafter.propose(prompt, aux, ref.new_ids[0], greedy=False, generator=gen).tokens

    assert torch.equal(sample(0), sample(0))


@pytest.fixture(scope="module")
def trained(setup):
    """A draft overfit on the reference sequence (prompt + the target's greedy answer)."""
    target, vocab, prompt, ref = setup
    torch.manual_seed(0)
    seq = torch.cat([prompt, ref.new_ids.view(1, -1)], dim=1)
    n_prompt = prompt.shape[1]
    loss_mask = torch.zeros_like(seq)
    loss_mask[:, n_prompt:] = 1
    features = capture_features(target, seq, loss_mask)

    draft = build_draft(target, vocab, dtype=torch.float32)
    eagle3 = wrap_eagle3(draft).train()
    lm_head = target_lm_head_weight(target)
    optimizer = torch.optim.AdamW([p for p in eagle3.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.0)
    for _ in range(150):
        plosses, _ = forward_step(eagle3, features, lm_head)
        optimizer.zero_grad(set_to_none=True)
        weighted_loss(plosses).backward()
        optimizer.step()
    return draft, seq, features


def test_drafter_matches_training_alignment(setup, trained):
    """Drafting with the overfit draft must reproduce the sequence it was trained on.

    The labels are the target's own greedy tokens, so if drafting lines up with the TTT training unroll
    (same shifts, same aux positions, same RoPE offsets), the K guesses after any anchor are the next K tokens.
    """
    _, vocab, prompt, _ = setup
    draft, seq, features = trained
    n_prompt = prompt.shape[1]

    # Score guess i only if tokens 1..i after the anchor are all in the draft vocab: a token outside it can
    # never be drafted, and every later guess in that chain is conditioned on the wrong token.
    drafter = Drafter(draft, k=3)
    hits, scored = torch.zeros(3), torch.zeros(3)
    for n in range(n_prompt + 1, seq.shape[1] - 3):
        guesses = drafter.propose(seq[:, :n], features.hidden_states[:, :n], seq[0, n]).tokens.cpu()
        real = seq[0, n + 1 : n + 4].cpu()
        draftable = vocab[1][real].long().cumprod(0).bool()
        hits += ((guesses == real) & draftable).float()
        scored += draftable.float()
    rate = hits / scored
    # Measured: 1.0 at every position; a misaligned shift / aux position / RoPE offset gives ~0.
    print("drafting hit rate per position", rate.tolist())
    assert (rate >= 0.95).all(), rate


def test_verify_greedy():
    logits = torch.full((4, 10), -1.0)
    for row, token in enumerate([3, 5, 7, 2]):
        logits[row, token] = 1.0  # target's greedy choices after x_n, g_1, g_2, g_3
    assert verify_greedy(logits, torch.tensor([3, 5, 7])).tokens.tolist() == [3, 5, 7, 2]  # all kept + extra
    v = verify_greedy(logits, torch.tensor([3, 9, 7]))
    assert v.accepted == 1 and v.tokens.tolist() == [3, 5]  # g_2 wrong: keep g_1, target supplies 5
    v = verify_greedy(logits, torch.tensor([1, 5, 7]))
    assert v.accepted == 0 and v.tokens.tolist() == [3]


def test_verify_sampling_matches_target_distribution():
    """With guesses drawn from q, the first emitted token must follow the target's p (not q)."""
    gen = torch.Generator().manual_seed(0)
    p = torch.tensor([0.10, 0.05, 0.40, 0.25, 0.20])  # target over 5 tokens
    draft_to_target = torch.tensor([0, 2, 3])  # draft vocab: 3 of them
    q = torch.tensor([0.50, 0.20, 0.30])  # deliberately unlike p
    logits = p.log().repeat(2, 1)  # K = 1: rows check g_1 / give the extra token
    n, counts = 20000, torch.zeros(5)
    for _ in range(n):
        g = draft_to_target[torch.multinomial(q, 1, generator=gen)]
        counts[verify_sampling(logits, g, q.view(1, -1), draft_to_target, generator=gen).tokens[0]] += 1
    assert torch.allclose(counts / n, p, atol=0.015), counts / n


PROMPTS = [
    "Explain what a hash map is in three sentences.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "If a shirt costs $25 after a 20% discount, what was the original price?",
]


@pytest.mark.parametrize("use_cache", [False, True], ids=["cache-free", "kv-cache"])
def test_speculative_greedy_is_lossless(setup, trained, use_cache):
    """Greedy speculative output == model.generate, with a random (day-0) and an overfit draft."""
    target, vocab, prompt, ref = setup
    tokenizer = load_tokenizer()
    day0 = Drafter(build_draft(target, vocab, dtype=torch.float32), k=3)
    for question in PROMPTS:
        ids = encode_prompt(tokenizer, question, target.model.device)
        expected = reference_generate(target, ids, max_new_tokens=48).new_ids
        out = speculative_generate(target, day0, ids, max_new_tokens=48, use_cache=use_cache)
        assert torch.equal(out.new_ids, expected), question

    # Overfit draft on its own training sequence: most guesses are accepted, output still identical.
    out = speculative_generate(target, Drafter(trained[0], k=3), prompt, max_new_tokens=48, use_cache=use_cache)
    assert torch.equal(out.new_ids, ref.new_ids)
    print(f"overfit draft ({'cache' if use_cache else 'no cache'}): accepted per round {out.accepted}")
    assert out.mean_accepted > 2.0, out.accepted
    assert len(out.accepted) < ref.new_ids.numel() / 2  # fewer target passes than tokens


def test_speculative_sampling(setup, trained):
    target, _, prompt, _ = setup
    drafter = Drafter(trained[0], k=3)

    def run(seed):
        gen = torch.Generator(device=prompt.device).manual_seed(seed)
        return speculative_generate(target, drafter, prompt, max_new_tokens=32, greedy=False, generator=gen)

    a, b = run(0), run(0)
    assert torch.equal(a.new_ids, b.new_ids) and a.accepted == b.accepted  # reproducible with a seed
    assert 0 < a.new_ids.numel() <= 32
    assert all(0 <= x <= 3 for x in a.accepted)
