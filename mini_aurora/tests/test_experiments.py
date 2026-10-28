"""BP-08: summary windows and the results table, on synthetic records (no model)."""

import json

from mini_aurora.experiments import E4_GRID, EXPERIMENTS, ONLINE_ARMS
from mini_aurora.experiments.run import loop_command, summarize
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
    assert {"N5", "N20", "N50", "async"} <= set(E4_GRID)


def test_loop_command_for_online_arms():
    day0 = loop_command("online_day0", "data/streams/mixed.jsonl", "out/E1/x", 300, None, 3, 128, False, E4_GRID["N5"])
    assert day0[1:3] == ["-m", "mini_aurora.online.loop"] and "--start" in day0 and day0[day0.index("--start") + 1] == "day0"
    assert day0[-2:] == ["--sync-every", "5"] and "--check" not in day0 and day0[day0.index("-n") + 1] == "300"
    static = loop_command("online_static", "s.jsonl", "o", None, "ckpt_dir", 3, 128, True)
    assert static[static.index("--start") + 1:static.index("--start") + 3] == ["draft", "--draft"] and "ckpt_dir" in static
    assert "--check" in static and static[static.index("-n") + 1] == str(10**9)


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
    base = {"experiment": "E2", "stream": "domain_shift", "claim": "", "arm_desc": "", "device_name": "T4"}
    for name, body in {
        "target_only_domain_shift": {**base, "arm": "target_only", "draft": None,
                                     "overall": {"requests": 4, "new_tokens": 32, "tokens_per_s": 50.0}},
        "static_domain_shift": {**base, "arm": "static", "draft": "outputs/baseline/checkpoints_8ep/draft_step000896.pt",
                                "overall": {"requests": 4, "new_tokens": 32, "tokens_per_s": 30.0, "tau": 0.3,
                                            "alpha": [0.24, 0.06, 0.01], "matches": "4/4"},
                                "by_window": [{"tau": 0.2}, {"tau": 0.4}]},
        "online_static_domain_shift_N5": {**base, "arm": "online_static", "draft": "ckpt/draft_step000896.pt",
                                          "loop_args": ["--sync-every", "5"],
                                          "overall": {"requests": 4, "new_tokens": 32, "tokens_per_s": 25.0, "tau": 0.5,
                                                      "alpha": [0.4, 0.1, 0.0]}},
    }.items():
        (tmp_path / "E2" / name).mkdir(parents=True)
        (tmp_path / "E2" / name / "summary.json").write_text(json.dumps(body))
    table = rows(load_summaries(tmp_path))
    assert [r[1] for r in table] == ["online_static", "static", "target_only"]
    static = table[1]
    assert static[5] == "0.200 / 0.400" and static[8] == "×0.60" and static[9] == "T4" and static[10] == "4/4" and static[11] == "draft_step000896.pt"
    assert table[0][11] == "draft_step000896.pt --sync-every 5" and table[0][8] == "×0.50"
    assert table[2][4] == "–" and table[2][5] == "–"
    md = markdown(table)
    assert md.startswith("| exp |") and md.count("\n") == 4
