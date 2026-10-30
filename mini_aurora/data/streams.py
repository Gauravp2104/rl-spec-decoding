"""Build the three request streams from the per-domain pools (CPU, no torch).

    python -m mini_aurora.data.streams            # -> data/streams/{mixed,ordered,domain_shift}.jsonl

| stream | order | size |
|---|---|---|
| ``mixed`` | all three pools, one seeded shuffle | 3 x PER_DOMAIN |
| ``ordered`` | chat, then code, then math (each pool in its own order) | 3 x PER_DOMAIN |
| ``domain_shift`` | first 600 chat, then 300 code, then 300 math (``SHIFT_SEGMENTS``) | 1200 |

Rows keep the full conversation (``messages``). The serving loop takes the first user turn as the request;
the offline baseline (BP-04) and the vocab mapping use the assistant turns. Each row also gets a ``pos``
field (its index in the stream) so plots of acceptance vs. request index need no extra bookkeeping.
"""

import argparse
import random
from pathlib import Path

from mini_aurora.data import DATA_DIR, DOMAINS, SEED, SHIFT_SEGMENTS, STREAMS
from mini_aurora.data.download import read_jsonl, write_jsonl


def load_pools(data_dir: str | Path) -> dict:
    raw = Path(data_dir) / "raw"
    return {d: read_jsonl(raw / f"{d}.jsonl") for d in DOMAINS}


def build_streams(pools: dict, seed: int = SEED, segments=SHIFT_SEGMENTS) -> dict:
    rng = random.Random(seed)
    mixed = [r for d in DOMAINS for r in pools[d]]
    rng.shuffle(mixed)
    ordered = [r for d in DOMAINS for r in pools[d]]
    shift = []
    for domain, n in segments:
        if n > len(pools[domain]):
            raise ValueError(f"domain_shift needs {n} {domain} rows, pool has {len(pools[domain])}")
        shift.extend(pools[domain][:n])
    streams = {"mixed": mixed, "ordered": ordered, "domain_shift": shift}
    return {name: [dict(r, pos=i) for i, r in enumerate(rows)] for name, rows in streams.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    streams = build_streams(load_pools(args.data_dir), args.seed)
    out = Path(args.data_dir) / "streams"
    for name in STREAMS:
        write_jsonl(out / f"{name}.jsonl", streams[name])
        domains = [r["domain"] for r in streams[name]]
        runs = [domains[0]] + [d for a, d in zip(domains, domains[1:]) if d != a]
        print(f"{name}: {len(domains)} rows, {len(runs)} domain runs"
              f"{'' if len(runs) > 10 else ' ' + ' -> '.join(runs)}")


if __name__ == "__main__":
    main()
