"""BP-08: summary windows and the results table, on synthetic records (no model)."""

import json

from mini_aurora.experiments import EXPERIMENTS, ONLINE_ARMS
from mini_aurora.experiments.run import summarize
from mini_aurora.experiments.table import load_summaries, markdown, rows


def _rec(i, domain, accepted=None, match=None):
    r = {"request_id": f"{domain}-{i:05d}", "domain": domain, "pos": i, "prompt_tokens": 10,
         "new_tokens": 8, "seconds": 0.5, "tokens_per_s": 16.0}
    if accepted is not None:
        r.update({"k": 3, "rounds": len(accepted), "accepted": accepted, "versions": [0] * len(accepted),
                  "mean_accepted": sum(accepted) / len(accepted)})
    if match is not None:
        r["match"] = match
    return r


def test_experiments_registry():
    for exp in EXPERIMENTS.values():
        assert exp["stream"] in ("mixed", "ordered", "domain_shift") and exp["arms"]
    assert ONLINE_ARMS <= {a for e in EXPERIMENTS.values() for a in e["arms"]}


def test_summarize_windows_and_domains():
    recs = [_rec(i, "chat", [1, 0, 2, 3], True) for i in range(3)] + [_rec(i, "code", [0, 0], True) for i in range(3, 5)]
    s = summarize(recs, window=3)
    assert s["overall"]["requests"] == 5 and s["overall"]["matches"] == "5/5"
    assert s["by_domain"]["chat"]["tau"] == 1.5 and s["by_domain"]["code"]["tau"] == 0.0
    assert s["by_domain"]["chat"]["alpha"] == [0.75, 0.5, 0.25]
    assert [w["end"] for w in s["by_window"]] == [3, 5] and s["by_window"][1]["domains"] == ["code"]
    # target-only records (no accepted) still summarize
    t = summarize([_rec(i, "chat") for i in range(2)], window=100)
    assert "tau" not in t["overall"] and t["overall"]["tokens_per_s"] == 16.0


def test_table_speedup_against_target_only(tmp_path):
    (tmp_path / "E2").mkdir()
    base = {"experiment": "E2", "stream": "domain_shift", "claim": "", "arm_desc": ""}
    (tmp_path / "E2" / "target_only_domain_shift.json").write_text(json.dumps(
        {**base, "arm": "target_only", "draft": None, "overall": {"requests": 4, "new_tokens": 32, "tokens_per_s": 50.0}}))
    (tmp_path / "E2" / "static_domain_shift.json").write_text(json.dumps(
        {**base, "arm": "static", "draft": "outputs/baseline/checkpoints_8ep/draft_step000896.pt",
         "overall": {"requests": 4, "new_tokens": 32, "tokens_per_s": 30.0, "tau": 0.3, "alpha": [0.24, 0.06, 0.01], "matches": "4/4"}}))
    table = rows(load_summaries(tmp_path))
    assert [r[1] for r in table] == ["static", "target_only"]
    assert table[0][7] == "×0.60" and table[0][8] == "4/4" and table[0][9] == "draft_step000896.pt"
    assert table[1][4] == "–"
    md = markdown(table)
    assert md.startswith("| exp |") and md.count("\n") == 3
