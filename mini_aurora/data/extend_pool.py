"""Build a bigger chat training pool that keeps the BP-02 pool fully held out.

    python -m mini_aurora.data.download --data-dir data_10k --per-domain 10000 --domains chat   # same seed/filters
    python -m mini_aurora.data.extend_pool                                                       # -> data_10k/chat_9k.jsonl

The seeded download of 10K conversations starts with the same 1000 rows as ``data/raw/chat.jsonl`` (the BP-02 pool
that every request stream is drawn from). This script drops those 1000, keeps the next ``--train`` rows for training,
and appends the BP-02 pool's last ``--heldout`` rows (BP-04's held-out split) so the dump's "last rows are held out"
rule still gives the same 100 held-out conversations and τ̄ stays comparable with the 900-conversation draft.
"""

import argparse
import json
from pathlib import Path

from mini_aurora.data.download import read_jsonl, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="data/raw/chat.jsonl", help="BP-02 chat pool (excluded from training)")
    p.add_argument("--big", default="data_10k/raw/chat.jsonl", help="seeded download with --per-domain > 1000")
    p.add_argument("--out", default="data_10k/chat_9k.jsonl")
    p.add_argument("--train", type=int, default=9000)
    p.add_argument("--heldout", type=int, default=100)
    a = p.parse_args()

    base = read_jsonl(Path(a.base))
    big = read_jsonl(Path(a.big))
    base_ids = {r["src_id"] for r in base}
    assert [r["src_id"] for r in big[: len(base)]] == [r["src_id"] for r in base], "big pool must extend the base pool (same seed)"
    train = [r for r in big if r["src_id"] not in base_ids][: a.train]
    if len(train) < a.train:
        raise SystemExit(f"only {len(train)} new conversations after excluding the base pool; download more")
    heldout = base[len(base) - a.heldout :]
    rows = train + heldout
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"
    write_jsonl(Path(a.out), rows)
    stats = {"train": len(train), "heldout": len(heldout), "train_tokens": sum(r["n_tokens"] for r in train),
             "heldout_ids": [heldout[0]["id"], heldout[-1]["id"]], "base": a.base, "big": a.big}
    Path(a.out).with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
