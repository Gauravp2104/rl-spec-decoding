"""Per-request serving log (BP-05 step 7): one JSON line per request, read by BP-07's metrics.

Each line keeps the per-round accepted lengths and the draft version of every round, so τ̄ over time, the
per-position acceptance α_i, and staleness (served version vs. trainer version, BP-06) can all be recomputed
from the log without re-running anything.
"""

import json
from pathlib import Path

from mini_aurora.server.spec_decode import SpecOutput


def request_record(out: SpecOutput, request_id: str, **extra) -> dict:
    """``extra`` holds request metadata, e.g. ``domain``, stream ``pos``, or ``match`` from a losslessness check."""
    return {
        "request_id": request_id,
        **extra,
        "prompt_tokens": out.prompt_ids.shape[1],
        "new_tokens": out.new_ids.numel(),
        "k": out.k,
        "rounds": len(out.accepted),
        "accepted": out.accepted,
        "versions": out.versions,
        "mean_accepted": round(out.mean_accepted, 4),
        "seconds": round(out.seconds, 4),
        "tokens_per_s": round(out.tokens_per_s, 2),
    }


def acceptance_by_position(records: list[dict]) -> list[float]:
    """α_i: fraction of rounds whose guess i was accepted (needs guesses 1..i-1 accepted too)."""
    rounds = [a for r in records for a in r["accepted"]]
    k = max((r["k"] for r in records), default=0)
    return [sum(a >= i for a in rounds) / len(rounds) if rounds else 0.0 for i in range(1, k + 1)]


class RequestLog:
    """Append-only JSONL file; one ``write`` per finished request."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a")

    def write(self, record: dict) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "RequestLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_log(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
