"""BP-05 steps 1-10: reference, target step, drafter, verification, speculative loop (with / without KV cache),
logging, training samples, draft hot-swap, and the 50-prompt losslessness check on the BP-02 stream.

Run in fp32 (target and draft) so greedy speculative output can be compared token for token with ``generate``.
"""

from pathlib import Path

import pytest
import torch
from huggingface_hub import try_to_load_from_cache
from transformers import DynamicCache

from mini_aurora.models.draft import build_draft, forward_step, target_lm_head_weight, weighted_loss, wrap_eagle3
from mini_aurora.models.target import DEFAULT_TARGET, capture_features, load_target, load_tokenizer
from mini_aurora.models.vocab import load_vocab_mapping
from mini_aurora.baseline.checkpoint import save_draft
from mini_aurora.server.drafter import Drafter, trainable_state
from mini_aurora.server.log import RequestLog, acceptance_by_position, read_log, request_record
from mini_aurora.server.reference import encode_prompt, eos_token_ids, reference_generate
from mini_aurora.server.serve import DEFAULT_STREAM, heldout_ids, load_requests, serve, summarize
from mini_aurora.server.spec_decode import speculative_generate
from mini_aurora.server.target_step import target_forward
from mini_aurora.server.tree import DraftTree, cache_keep, tree_attention_mask, tree_position_ids, walk_greedy
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


def test_log_records(setup, trained, tmp_path):
    target, _, prompt, _ = setup
    out = speculative_generate(target, Drafter(trained[0], k=3, version=7), prompt, max_new_tokens=48)
    record = request_record(out, "req-0", domain="chat", pos=0)
    assert record["accepted"] == out.accepted and set(record["versions"]) == {7}
    assert record["new_tokens"] == out.new_ids.numel() and record["rounds"] == len(out.accepted)
    with RequestLog(tmp_path / "log.jsonl") as log:
        log.write(record)
        log.write(record)
    assert read_log(tmp_path / "log.jsonl") == [record, record]
    alpha = acceptance_by_position([record])
    assert len(alpha) == 3 and alpha[0] >= alpha[1] >= alpha[2]
    assert alpha[0] == sum(a >= 1 for a in out.accepted) / len(out.accepted)


def test_training_sample_matches_capture(setup, trained):
    """The sample reuses the verify passes' states; they must equal a fresh target pass over the same tokens."""
    target, _, prompt, ref = setup
    draft = trained[0]
    out = speculative_generate(target, Drafter(draft, k=3), prompt, max_new_tokens=48, return_sample=True)
    sample, m = out.sample, out.new_ids.numel()
    assert sample.input_ids.shape == (1, prompt.shape[1] + m - 1)
    assert torch.equal(sample.input_ids[0, prompt.shape[1] :], out.new_ids[:-1])
    assert int(sample.loss_mask.sum()) == m - 1 and not sample.loss_mask[0, : prompt.shape[1]].any()

    fresh = capture_features(target, sample.input_ids, sample.loss_mask)
    assert torch.allclose(sample.hidden_states, fresh.hidden_states, atol=1e-3)
    assert torch.allclose(sample.last_hidden_states, fresh.last_hidden_states, atol=1e-3)
    # Usable as-is by the trainer (BP-06).
    plosses, _ = forward_step(wrap_eagle3(draft), sample, target_lm_head_weight(target))
    assert all(torch.isfinite(loss) for loss in plosses)


def test_hot_swap(setup, trained, tmp_path):
    target, vocab, prompt, ref = setup
    served = Drafter(build_draft(target, vocab, dtype=torch.float32), k=3)  # day-0, version 0
    before = speculative_generate(target, served, prompt, max_new_tokens=48)
    assert before.mean_accepted < 0.5 and set(before.versions) == {0}

    trainer_draft = trained[0]
    served.load_weights(trainable_state(trainer_draft), version=1, d2t=trainer_draft.d2t)
    after = speculative_generate(target, served, prompt, max_new_tokens=48)
    assert torch.equal(after.new_ids, ref.new_ids) and set(after.versions) == {1}
    assert after.mean_accepted > 2.0

    # The served copy is independent of the trainer's weights.
    assert served.draft.fc.weight.data_ptr() != trainer_draft.fc.weight.data_ptr()

    # Round trip through a BP-04 checkpoint (file and directory); the step becomes the version.
    path = save_draft(trainer_draft, tmp_path / "ckpt", step=2)
    fresh = Drafter(build_draft(target, vocab, dtype=torch.float32), k=3)
    assert fresh.load_file(path) == path
    assert fresh.version == 2 and torch.equal(fresh.draft.lm_head.weight, trainer_draft.lm_head.weight)
    save_draft(trainer_draft, tmp_path / "ckpt", step=5)
    assert fresh.load_file(tmp_path / "ckpt").name == "draft_step000005.pt" and fresh.version == 5
    with pytest.raises(FileNotFoundError):
        fresh.load_file(tmp_path)

    # Refuse weights trained with another vocab mapping.
    with pytest.raises(ValueError, match="d2t differs"):
        fresh.load_weights(trainable_state(trainer_draft), 3, d2t=torch.roll(trainer_draft.d2t, 1))
    with pytest.raises(ValueError, match="t2d differs"):
        fresh.load_weights(trainable_state(trainer_draft), 3, t2d=torch.roll(trainer_draft.t2d, 1))
    other = torch.load(path, weights_only=False)
    other["t2d"] = torch.roll(other["t2d"], 1)
    torch.save(other, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="vocab mapping"):
        fresh.load_file(tmp_path / "other.pt")


@pytest.mark.skipif(not Path(DEFAULT_STREAM).exists(), reason="BP-02 stream missing (python -m mini_aurora.data.streams)")
def test_heldout_requests():
    """--heldout serves exactly BP-04's held-out chat conversations (the 100 the static draft never saw)."""
    ids = heldout_ids()
    assert len(ids) == 100 and all(i.startswith("chat-") for i in ids)
    rows = load_requests(DEFAULT_STREAM, 1000, ids=ids)
    assert len(rows) == 100 and {r["id"] for r in rows} == ids


@pytest.mark.skipif(not Path(DEFAULT_STREAM).exists(), reason="BP-02 stream missing (python -m mini_aurora.data.streams)")
def test_serve_50_prompts_lossless(setup, trained):
    """BP-05 done-when: greedy speculative output == model.generate on 50 stream prompts (bs = 1, fp32)."""
    target, _, _, _ = setup
    rows = load_requests(DEFAULT_STREAM, 50)
    samples = []
    records = serve(
        target, Drafter(trained[0], k=3), load_tokenizer(), rows, max_new_tokens=32, check=True,
        on_sample=samples.append,
    )
    summary = summarize(records)
    print("50-prompt summary", summary)
    assert [r["request_id"] for r in records] == [r["id"] for r in rows]
    assert all(r["match"] for r in records), [r["request_id"] for r in records if not r["match"]]
    assert summary["all"]["matches"] == "50/50" and len(samples) == 50
    assert {"chat", "code", "math"} <= set(summary)


# --- tree drafting (top-k > 1) -------------------------------------------------------------------------------


def _toy_tree():
    """Root x_n with children A (node 0) and B (node 1); A has children C (2) and D (3); B has child E (4)."""
    return DraftTree(
        tokens=torch.tensor([10, 11, 20, 21, 30]),
        draft_ids=torch.tensor([10, 11, 20, 21, 30]),
        parent=torch.tensor([0, 0, 1, 1, 2]),
        depth=torch.tensor([1, 1, 2, 2, 2]),
        score=torch.zeros(5),
    )


def test_tree_mask_and_positions():
    tree = _toy_tree()
    mask = tree_attention_mask(tree.parent, past=3, dtype=torch.float32)
    assert mask.shape == (1, 1, 6, 9)
    allowed = mask[0, 0] == 0
    assert allowed[:, :3].all()  # every tree position sees the whole context
    # verify positions: 0 root, 1 A, 2 B, 3 C, 4 D, 5 E (offset by the 3 context slots)
    assert allowed[4, 3:].tolist() == [True, True, False, False, True, False]  # D: root, A, itself
    assert allowed[5, 3:].tolist() == [True, False, True, False, False, True]  # E: root, B, itself
    assert allowed[0, 3:].tolist() == [True, False, False, False, False, False]
    full = tree_attention_mask(tree.parent, past=3, dtype=torch.float32, prefix_rows=True)
    assert full.shape == (1, 1, 9, 9)
    assert (full[0, 0, :3, :3] == 0).tril().sum() == 6 and (full[0, 0, :3, 3:] != 0).all()
    assert torch.equal(full[0, 0, 3:], mask[0, 0])
    assert tree_position_ids(tree.depth, 3).tolist() == [[3, 4, 4, 5, 5, 5]]


def test_walk_greedy_toy():
    tree = _toy_tree()
    logits = torch.full((6, 100), -1.0)
    for row, token in enumerate([11, 99, 30, 99, 99, 7]):  # root → B, A → nothing, B → E, ..., E → 7
        logits[row, token] = 1.0
    v = walk_greedy(logits, tree)
    assert (v.accepted, v.keep, v.tokens.tolist()) == (2, [0, 2, 5], [11, 30, 7])
    logits[0, 11], logits[0, 5] = -1.0, 1.0  # root's argmax is not a child: nothing accepted, extra = 5
    v = walk_greedy(logits, tree)
    assert (v.accepted, v.keep, v.tokens.tolist()) == (0, [0], [5])


def test_cache_keep():
    cache = DynamicCache()
    keys = torch.arange(9.0).view(1, 1, 9, 1)
    cache.update(keys, keys * 10, 0)
    cache_keep(cache, past=3, keep=[0, 2, 5])
    kept = cache.layers[0].keys if hasattr(cache, "layers") else cache.key_cache[0]
    assert kept.flatten().tolist() == [0, 1, 2, 3, 5, 8]
    assert cache.get_seq_length() == 6


def test_propose_tree_topk1_is_the_chain(setup, trained):
    target, _, prompt, ref = setup
    drafter = Drafter(trained[0], k=3)
    aux = target_forward(target, prompt).aux_hidden
    chain = drafter.propose(prompt, aux, ref.new_ids[0])
    tree = drafter.propose_tree(prompt, aux, ref.new_ids[0], topk=1, beam=1)
    assert torch.equal(tree.tokens, chain.tokens) and tree.parent.tolist() == [0, 1, 2] and tree.depth.tolist() == [1, 2, 3]
    wide = drafter.propose_tree(prompt, aux, ref.new_ids[0], k=3, topk=2, beam=3)
    assert wide.size == 2 + 3 + 3 and wide.depth.tolist() == [1, 1, 2, 2, 2, 3, 3, 3]
    assert (wide.parent[2:5] <= 2).all() and (wide.parent[5:] >= 3).all()
    assert torch.equal(wide.tokens[0], chain.tokens[0])  # the chain's first guess is the root's best child
    assert drafter.draft.t2d[wide.tokens].all()


@pytest.mark.parametrize("use_cache", [False, True], ids=["cache-free", "kv-cache"])
def test_tree_greedy_is_lossless(setup, trained, use_cache):
    """Tree rounds (top-k 2, beam 3, depth 3) give model.generate's tokens with a day-0 and an overfit draft."""
    target, vocab, prompt, ref = setup
    tokenizer = load_tokenizer()
    day0 = Drafter(build_draft(target, vocab, dtype=torch.float32), k=3)
    for question in PROMPTS[:2]:
        ids = encode_prompt(tokenizer, question, target.model.device)
        expected = reference_generate(target, ids, max_new_tokens=32).new_ids
        out = speculative_generate(target, day0, ids, max_new_tokens=32, use_cache=use_cache, tree_topk=2, tree_beam=3)
        assert torch.equal(out.new_ids, expected), question

    chain = speculative_generate(target, Drafter(trained[0], k=3), prompt, max_new_tokens=48, use_cache=use_cache)
    out = speculative_generate(target, Drafter(trained[0], k=3), prompt, max_new_tokens=48, use_cache=use_cache,
                               tree_topk=2, tree_beam=3)
    assert torch.equal(out.new_ids, ref.new_ids)
    print(f"overfit draft, tree ({'cache' if use_cache else 'no cache'}): accepted per round {out.accepted}")
    assert out.mean_accepted >= chain.mean_accepted  # a tree never accepts less than its own best chain
