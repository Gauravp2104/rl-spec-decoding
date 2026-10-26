"""Metrics for an online run (BP-07 steps 1-2), recomputed from BP-06's logs without re-running anything.

    python -m mini_aurora.eval.metrics outputs/online/day0_sync_300            # window 50
    python -m mini_aurora.eval.metrics outputs/online/day0_sync_300 --window 25

Reads ``<run>/requests.jsonl`` (BP-05 ``RequestLog`` lines + BP-06's ``trainer_version``), ``<run>/train_log.csv``,
``<run>/config.json`` and, if present, ``<run>/eval/bench.json`` (``mini_aurora.eval.bench``: the draft / target
step-latency ratio c and target-only tok/s). Writes ``<run>/eval/metrics.csv`` (one row per request plus trailing-
window columns) and ``<run>/eval/summary.json`` (``mini_aurora.server.serve.summarize()``'s shape: one entry per
domain and ``all``, with extra keys).

Definitions (K = guesses per round):

- τ̄: accepted guesses per round, round-weighted (``summarize``); rolling τ̄ over the trailing W requests;
- α_i: fraction of rounds whose guess i was accepted (``acceptance_by_position``), overall and per W-request window;
- acceptance rate: sum(accepted) / (rounds × K) = τ̄ / K;
- tok/s vs target-only: new tokens / seconds, against ``reference_tokens_per_s`` (``--check`` runs) or the bench;
- estimated speed-up: (1 + τ̄) / (1 + K·c), c = draft-step latency / target-step latency (PLAN.md §3);
- staleness Δ per round: trainer version when the request started − served draft version of that round.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from mini_aurora.server.log import acceptance_by_position, read_log
from mini_aurora.server.serve import summarize

EVAL_DIR = "eval"


def load_run(run_dir: str | Path) -> dict:
    """``records``, ``train`` (train_log.csv rows as floats), ``config`` and ``bench`` (or None) of a run."""
    run = Path(run_dir)
    train_path, config_path, bench_path = run / "train_log.csv", run / "config.json", run / EVAL_DIR / "bench.json"
    return {
        "records": read_log(run / "requests.jsonl"),
        "train": read_train_log(train_path) if train_path.exists() else [],
        "config": json.loads(config_path.read_text()) if config_path.exists() else {},
        "bench": json.loads(bench_path.read_text()) if bench_path.exists() else None,
    }


def read_train_log(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def tau(records: list[dict]) -> float:
    rounds = [a for r in records for a in r["accepted"]]
    return sum(rounds) / len(rounds) if rounds else 0.0


def request_tau(record: dict) -> float:
    return tau([record])


def acceptance_rate(records: list[dict]) -> float:
    """sum(accepted) / (rounds × K)."""
    slots = sum(len(r["accepted"]) * r["k"] for r in records)
    return sum(sum(r["accepted"]) for r in records) / slots if slots else 0.0


def tokens_per_s(records: list[dict]) -> float:
    seconds = sum(r["seconds"] for r in records)
    return sum(r["new_tokens"] for r in records) / seconds if seconds else 0.0


def trailing(records: list[dict], window: int) -> list[list[dict]]:
    """For each request i, the last ``window`` requests up to and including i (shorter at the start)."""
    return [records[max(0, i - window + 1) : i + 1] for i in range(len(records))]


def rolling_tau(records: list[dict], window: int) -> list[float]:
    return [tau(w) for w in trailing(records, window)]


def alpha_windows(records: list[dict], window: int) -> list[dict]:
    """α_i per consecutive, non-overlapping block of ``window`` requests (the last block may be shorter)."""
    return [{"start": i, "end": min(i + window, len(records)), "alpha": acceptance_by_position(records[i : i + window])}
            for i in range(0, len(records), window)]


def staleness(record: dict) -> list[int]:
    """Δ per round; empty for logs without ``trainer_version`` (plain BP-05 ``serve`` runs)."""
    if "trainer_version" not in record:
        return []
    return [record["trainer_version"] - v for v in record["versions"]]


def staleness_histogram(records: list[dict]) -> dict[int, int]:
    return dict(sorted(Counter(d for r in records for d in staleness(r)).items()))


def estimated_speedup(tau_bar: float, k: int, c: float) -> float:
    """(1 + τ̄) / (1 + K·c): tokens per round over the round's cost in target steps (verify counted as one step)."""
    return (1 + tau_bar) / (1 + k * c)


def target_tokens_per_s(records: list[dict], bench: dict | None = None) -> float | None:
    """Target-only tok/s: mean ``reference_tokens_per_s`` of checked requests, else the bench's, else None."""
    refs = [r["reference_tokens_per_s"] for r in records if "reference_tokens_per_s" in r]
    if refs:
        return sum(refs) / len(refs)
    return bench.get("target_tokens_per_s") if bench else None


def loss_curve(train: list[dict]) -> dict[str, list[float]]:
    """Columns of train_log.csv as lists (``step``, ``loss``, ``loss_i``, ``acc_i``, ``grad_norm``, ...)."""
    return {k: [row[k] for row in train] for k in (train[0] if train else {})}


def domain_segments(records: list[dict]) -> list[tuple[int, int, str]]:
    """Maximal runs of consecutive requests from one domain: ``(start, end_exclusive, domain)``."""
    segments = []
    for i, r in enumerate(records):
        if segments and segments[-1][2] == r.get("domain"):
            segments[-1] = (segments[-1][0], i + 1, segments[-1][2])
        else:
            segments.append((i, i + 1, r.get("domain")))
    return segments


def request_rows(records: list[dict], window: int = 50) -> list[dict]:
    """One row per request for metrics.csv: its own metrics and the trailing-``window`` ones."""
    k = max((r["k"] for r in records), default=0)
    rows = []
    for i, (r, w) in enumerate(zip(records, trailing(records, window))):
        delta = staleness(r)
        alpha = acceptance_by_position(w)
        rows.append({
            "index": r.get("index", i),
            "request_id": r["request_id"],
            "domain": r.get("domain"),
            "prompt_tokens": r["prompt_tokens"],
            "new_tokens": r["new_tokens"],
            "rounds": len(r["accepted"]),
            "accepted": sum(r["accepted"]),
            "tau": round(request_tau(r), 4),
            "acceptance_rate": round(acceptance_rate([r]), 4),
            "tokens_per_s": r["tokens_per_s"],
            "served_version": r["versions"][-1] if r["versions"] else "",
            "trainer_version": r.get("trainer_version", ""),
            "staleness_mean": round(sum(delta) / len(delta), 3) if delta else "",
            "staleness_max": max(delta) if delta else "",
            f"rolling_tau_{window}": round(tau(w), 4),
            f"rolling_acceptance_rate_{window}": round(acceptance_rate(w), 4),
            f"rolling_tokens_per_s_{window}": round(tokens_per_s(w), 2),
            **{f"rolling_alpha{j + 1}_{window}": round(alpha[j], 4) if j < len(alpha) else "" for j in range(k)},
        })
    return rows


def write_metrics_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(records: list[dict], train: list[dict] | None = None, bench: dict | None = None,
                  window: int = 50) -> dict:
    """``summarize()`` per domain and ``all``, plus acceptance rate, staleness, speed-up and (``all``) training."""
    summary = summarize(records)
    c = bench.get("c") if bench else None
    for name, group in summary.items():
        rs = records if name == "all" else [r for r in records if r["domain"] == name]
        k = max(r["k"] for r in rs)
        deltas = [d for r in rs for d in staleness(r)]
        target_tps = target_tokens_per_s(rs, bench)
        group["acceptance_rate"] = round(acceptance_rate(rs), 4)
        group["staleness_mean"] = round(sum(deltas) / len(deltas), 3) if deltas else None
        group["staleness_max"] = max(deltas) if deltas else None
        group["target_tokens_per_s"] = round(target_tps, 1) if target_tps else None
        group["speedup_vs_target"] = round(group["tokens_per_s"] / target_tps, 3) if target_tps else None
        group["estimated_speedup"] = round(estimated_speedup(group["tau"], k, c), 3) if c is not None else None
    summary["all"].update({
        f"tau_first_{window}": round(tau(records[:window]), 4),
        f"tau_last_{window}": round(tau(records[-window:]), 4),
        "staleness_histogram": staleness_histogram(records),
        "c": c,
    })
    if train:
        last = train[-1]
        summary["all"]["train_steps"] = len(train)
        summary["all"]["final_loss"] = last["loss"]
        summary["all"]["final_acc"] = [last[key] for key in last if key.startswith("acc_")]
    return summary


def evaluate_run(run_dir: str | Path, window: int = 50) -> dict:
    """Write ``<run>/eval/metrics.csv`` and ``<run>/eval/summary.json``; returns the summary."""
    run = load_run(run_dir)
    out = Path(run_dir) / EVAL_DIR
    write_metrics_csv(request_rows(run["records"], window), out / "metrics.csv")
    summary = summarize_run(run["records"], run["train"], run["bench"], window)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="online run directory (BP-06 --out)")
    p.add_argument("--window", type=int, default=50)
    args = p.parse_args()
    summary = evaluate_run(args.run, args.window)
    print(json.dumps(summary, indent=2))
    print(f"metrics -> {Path(args.run) / EVAL_DIR}")


if __name__ == "__main__":
    main()
