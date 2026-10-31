"""PT-01: self-distilled regeneration (``mini_aurora.data.regenerate``).

The turn logic runs on a fake generator (no torch); the last tests use the real default target if it is cached.
"""

import json

import pytest

from mini_aurora.data import regenerate as rg
from mini_aurora.data.regenerate import Reply, is_looped, regenerate_rows, summarize


def _row(i, n_pairs):
    msgs = []
    for t in range(n_pairs):
        msgs += [{"role": "user", "content": f"u{i}.{t}"}, {"role": "assistant", "content": f"gt{i}.{t}"}]
    return {"id": f"chat-{i:05d}", "domain": "chat", "source": "sharegpt", "src_id": f"s{i}", "messages": msgs}


class FakeGen:
    """Answers ``ans(<last user>|<n earlier assistant turns>)``; records every history it was asked about."""

    def __init__(self, new_tokens=5):
        self.calls, self.new_tokens = [], new_tokens

    def __call__(self, histories):
        self.calls.append(histories)
        return [Reply(f"ans({h[-1]['content']}|{sum(m['role'] == 'assistant' for m in h)})", self.new_tokens,
                      False, False) for h in histories]


def _len(history):  # 10 tokens per message
    return 10 * len(history)


def test_keeps_user_turns_ids_and_order():
    rows = [_row(0, 2), _row(1, 1), _row(2, 3)]
    out = regenerate_rows(rows, FakeGen(), _len)
    assert [r["id"] for r in out] == [r["id"] for r in rows]
    for r, o in zip(rows, out):
        assert rg.user_turns(o["messages"]) == rg.user_turns(r["messages"])
        assert [m["role"] for m in o["messages"]] == [m["role"] for m in r["messages"]]
        assert not any(m["content"].startswith("gt") for m in o["messages"])
        assert o["regen"]["dropped_pairs"] == 0


def test_turns_condition_on_own_answers_and_batch_by_turn():
    gen = FakeGen()
    out = regenerate_rows([_row(0, 2), _row(1, 3)], gen, _len)
    assert [len(c) for c in gen.calls] == [2, 2, 1]  # batched by turn index
    second_turn_history = gen.calls[1][0]
    assert second_turn_history[1]["content"] == "ans(u0.0|0)"  # the target's answer, not "gt0.0"
    assert out[1]["messages"][-1]["content"] == "ans(u1.2|2)"


def test_cuts_pairs_past_max_tokens_but_keeps_first():
    # turn 0: 10 + 5 + 2 = 17 tokens; turn 1: 30 + 5 + 2 = 37; turn 2 prompt 50.
    out = regenerate_rows([_row(0, 3)], FakeGen(), _len, max_tokens=40)
    assert len(out[0]["messages"]) == 4 and out[0]["regen"]["dropped_pairs"] == 1
    assert out[0]["n_tokens"] == 37
    out = regenerate_rows([_row(0, 3)], FakeGen(new_tokens=100), _len, max_tokens=40)
    assert len(out[0]["messages"]) == 2 and out[0]["regen"]["dropped_pairs"] == 2  # first pair kept even if long


def test_is_looped():
    assert is_looped([1, 2, 3, 4] * 20)
    assert not is_looped(list(range(100)))
    assert not is_looped([7] * 10)  # too short to judge


def test_summarize_counts():
    rows = [_row(0, 2)]
    out = regenerate_rows(rows, FakeGen(), _len)
    out[0]["regen"]["turns"][0]["hit_cap"] = True
    s = summarize(rows, out, lambda text: 3)
    assert s["turns_original"] == s["turns_generated"] == 2
    assert s["hit_cap_rate"] == 0.5 and s["assistant_tokens_original"] == 6 and s["assistant_tokens_generated"] == 10
    json.dumps(s)


def test_run_resumes_from_shards(tmp_path, monkeypatch):
    src = tmp_path / "chat.jsonl"
    src.write_text("".join(json.dumps(_row(i, 2)) + "\n" for i in range(5)))

    class Fake(FakeGen):
        def __init__(self, *a, **k):
            super().__init__()
            self.tokenizer = lambda s, **k: {"input_ids": s.split()}

        prompt_len = staticmethod(_len)

    monkeypatch.setattr(rg, "TargetGenerator", Fake)
    out = tmp_path / "self.jsonl"
    stats = rg.run(str(src), str(out), "fake", "cpu", "float32", shard_size=2)
    assert stats["conversations"] == 5 and len(list((tmp_path / "self.shards").glob("shard_*.jsonl"))) == 3
    (tmp_path / "self.shards" / "shard_00002.jsonl").unlink()

    class Counting(Fake):
        built = 0

        def __init__(self, *a, **k):
            super().__init__()
            Counting.built += 1

    monkeypatch.setattr(rg, "TargetGenerator", Counting)
    rg.run(str(src), str(out), "fake", "cpu", "float32", shard_size=2)
    assert Counting.built == 1  # only the missing shard was regenerated
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == [f"chat-{i:05d}" for i in range(5)]
    with pytest.raises(SystemExit):
        rg.run(str(src), str(out), "fake", "cpu", "float32", shard_size=3)  # config changed


# --- real target (skipped if not cached) ---------------------------------------------------------------------

def _cached():
    try:
        from huggingface_hub import try_to_load_from_cache

        from mini_aurora.models.target import DEFAULT_TARGET
        return isinstance(try_to_load_from_cache(DEFAULT_TARGET, "config.json"), str)
    except ImportError:
        return False


@pytest.mark.skipif(not _cached(), reason="default target not in the HF cache")
def test_real_target_batched_greedy_matches_single():
    import torch

    from mini_aurora.device import get_device
    from mini_aurora.models.target import DEFAULT_TARGET

    gen = rg.TargetGenerator(DEFAULT_TARGET, get_device(), torch.float32, max_new_tokens=24, batch_size=4)
    hs = [[{"role": "user", "content": "Name three primary colors."}],
          [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"},
           {"role": "user", "content": "Write one sentence about the sea and nothing else, please."}]]
    batched = gen(hs)
    gen.batch_size = 1
    single = gen(hs)
    for b, s in zip(batched, single):
        assert b.new_tokens > 0 and "<think>" not in b.text and "<|im_end|>" not in b.text
        assert b.text[:40] == s.text[:40]  # left padding must not change greedy output
