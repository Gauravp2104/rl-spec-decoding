"""BP-02: conversation normalisation, filters and stream construction (no network, no torch)."""

import json

import pytest

from mini_aurora.data import DOMAINS, SHIFT_SEGMENTS, STREAMS
from mini_aurora.data.download import is_english, normalize
from mini_aurora.data.streams import build_streams


def test_normalize_sharegpt_roles_and_system_drop():
    turns = [{"from": "system", "value": "sys"}, {"from": "human", "value": " hi "}, {"from": "gpt", "value": "hello"}]
    assert normalize(turns) == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


@pytest.mark.parametrize("turns, reason", [
    ([{"from": "gpt", "value": "a"}, {"from": "human", "value": "b"}], "not_user_first"),
    ([{"from": "human", "value": "a"}, {"from": "human", "value": "b"}, {"from": "gpt", "value": "c"}], "bad_alternation"),
    ([{"from": "human", "value": "a"}, {"from": "gpt", "value": "   "}], "empty"),
    ([{"from": "human", "value": "a"}], "not_user_first"),
])
def test_normalize_rejects(turns, reason):
    assert normalize(turns) is None
    assert normalize.reason == reason


def test_normalize_truncates_and_ends_on_assistant():
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(9)]
    out = normalize(turns)
    assert len(out) == 6 and out[-1]["role"] == "assistant"


def test_is_english():
    assert is_english([{"role": "user", "content": "How do I sort a list in Python?"}])
    assert not is_english([{"role": "user", "content": "Привет, как дела сегодня?"}])
    assert not is_english([{"role": "user", "content": "これは日本語の文章です。"}])


def _pool(domain, n):
    return [{"id": f"{domain}-{i:05d}", "domain": domain, "messages": [], "n_tokens": 1} for i in range(n)]


def test_build_streams_shapes_and_order():
    pools = {d: _pool(d, 700) for d in DOMAINS}
    streams = build_streams(pools, seed=0)
    assert set(streams) == set(STREAMS)
    assert len(streams["mixed"]) == 2100 and len(streams["ordered"]) == 2100
    assert sorted(r["id"] for r in streams["mixed"]) == sorted(r["id"] for r in streams["ordered"])
    assert [r["domain"] for r in streams["ordered"]] == [d for d in DOMAINS for _ in range(700)]
    expected = [d for d, n in SHIFT_SEGMENTS for _ in range(n)]
    assert [r["domain"] for r in streams["domain_shift"]] == expected
    assert [r["pos"] for r in streams["domain_shift"]] == list(range(len(expected)))
    # a shuffle, not the ordered sequence
    assert [r["domain"] for r in streams["mixed"]] != [r["domain"] for r in streams["ordered"]]
    assert build_streams(pools, seed=0)["mixed"][0]["id"] == streams["mixed"][0]["id"]


def test_build_streams_rejects_small_pool():
    pools = {d: _pool(d, 100) for d in DOMAINS}
    with pytest.raises(ValueError):
        build_streams(pools)


def test_rows_are_json_serialisable():
    row = build_streams({d: _pool(d, 600) for d in DOMAINS})["domain_shift"][0]
    assert json.loads(json.dumps(row))["domain"] == "chat"
