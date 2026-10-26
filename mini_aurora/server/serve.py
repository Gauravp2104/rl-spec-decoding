"""Serve requests from a BP-02 stream with speculative decoding (BP-05 step 10).

    python -m mini_aurora.server.serve -n 50 --check                     # done-when: greedy == generate on 50
    python -m mini_aurora.server.serve -n 100 --heldout --draft outputs/baseline/checkpoints_8ep --check   # BP-04 τ̄
    python -m mini_aurora.server.serve -n 20 --draft outputs/baseline/checkpoints_8ep --log outputs/server/run.jsonl
    python -m mini_aurora.server.serve -n 20 --sampling --temperature 0.7 --seed 0

Each request is a stream conversation minus its final assistant answer (``encode_messages``). Without
``--draft`` the draft is day-0 (random), which is correct but accepts ~nothing. ``--draft`` takes a BP-04
checkpoint file or directory (``outputs/baseline/checkpoints``, resolved through ``latest.txt``); ``--heldout``
keeps only the chat conversations BP-04 never trained on (the same split as ``mini_aurora.baseline.dump``). Target and draft run in fp32 by
default, as the losslessness check needs; ``--bf16`` is faster but can flip near-tied argmaxes.
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch

from mini_aurora.baseline import HELDOUT
from mini_aurora.baseline.dump import DEFAULT_JSONL as CHAT_POOL
from mini_aurora.baseline.dump import split_rows
from mini_aurora.data.download import read_jsonl
from mini_aurora.device import get_device
from mini_aurora.models.draft import build_draft
from mini_aurora.models.target import load_target, load_tokenizer
from mini_aurora.models.vocab import DEFAULT_VOCAB, load_vocab_mapping
from mini_aurora.server.drafter import Drafter
from mini_aurora.server.log import RequestLog, acceptance_by_position, request_record
from mini_aurora.server.reference import encode_messages, reference_generate
from mini_aurora.server.spec_decode import speculative_generate

DEFAULT_STREAM = "data/streams/mixed.jsonl"


def heldout_ids(pool: str | Path = CHAT_POOL, heldout: int = HELDOUT) -> set[str]:
    """IDs of the chat conversations BP-04 held out of training (the last ``heldout`` rows of the pool)."""
    return {row["id"] for row in split_rows(read_jsonl(Path(pool)), heldout)["heldout"]}


def load_requests(path: str | Path, n: int, domain: str | None = None, ids: set[str] | None = None) -> list[dict]:
    """First ``n`` stream rows (``id``, ``domain``, ``pos``, ``messages``), optionally from one domain / ID set."""
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if (domain is None or row["domain"] == domain) and (ids is None or row["id"] in ids):
                rows.append(row)
                if len(rows) == n:
                    break
    return rows


def serve(
    target,
    drafter: Drafter,
    tokenizer,
    rows: list[dict],
    max_new_tokens: int = 128,
    greedy: bool = True,
    temperature: float = 1.0,
    use_cache: bool = True,
    seed: int = 0,
    check: bool = False,
    log: RequestLog | None = None,
    on_sample: Callable | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Serve ``rows`` one at a time (bs = 1); returns one log record per request.

    ``check`` also runs target-only greedy decoding and records ``match``; ``on_sample`` receives each
    request's training sample (BP-06 pushes these into its sample pool).
    """
    if check and not greedy:
        raise ValueError("The token-for-token check only applies to greedy decoding")
    device = target.model.device
    generator = None if greedy else torch.Generator(device=device).manual_seed(seed)
    records = []
    for row in rows:
        ids = encode_messages(tokenizer, row["messages"], device)
        out = speculative_generate(
            target,
            drafter,
            ids,
            max_new_tokens=max_new_tokens,
            greedy=greedy,
            temperature=temperature,
            use_cache=use_cache,
            generator=generator,
            return_sample=on_sample is not None,
        )
        extra = {"domain": row.get("domain"), "pos": row.get("pos")}
        if check:
            ref = reference_generate(target, ids, max_new_tokens)
            extra["match"] = torch.equal(out.new_ids, ref.new_ids)
            extra["reference_tokens_per_s"] = round(ref.tokens_per_s, 2)
        record = request_record(out, row["id"], **extra)
        records.append(record)
        if log is not None:
            log.write(record)
        if on_sample is not None:
            on_sample(out.sample)
        if verbose:
            match = f"  match={record['match']}" if check else ""
            print(
                f"{record['request_id']:>12s}  {record['new_tokens']:4d} tok  τ̄ {record['mean_accepted']:.2f}  "
                f"{record['tokens_per_s']:6.1f} tok/s{match}"
            )
    return records


def summarize(records: list[dict]) -> dict:
    """τ̄ (mean accepted guesses per round), α_i and tok/s per domain and overall; matches if checked."""
    groups = defaultdict(list)
    for r in records:
        groups[r["domain"]].append(r)
        groups["all"].append(r)
    summary = {}
    for name, group in groups.items():
        rounds = [a for r in group for a in r["accepted"]]
        tokens, seconds = sum(r["new_tokens"] for r in group), sum(r["seconds"] for r in group)
        summary[name] = {
            "requests": len(group),
            "tau": round(sum(rounds) / len(rounds), 3) if rounds else 0.0,
            "alpha": [round(a, 3) for a in acceptance_by_position(group)],
            "tokens_per_s": round(tokens / seconds, 1) if seconds else 0.0,
        }
        if "match" in group[0]:
            summary[name]["matches"] = f"{sum(r['match'] for r in group)}/{len(group)}"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", default=DEFAULT_STREAM)
    parser.add_argument("-n", type=int, default=50, help="number of requests")
    parser.add_argument("--domain", default=None, choices=["chat", "code", "math"])
    parser.add_argument("--heldout", action="store_true", help="only chat requests BP-04 never trained on")
    parser.add_argument("--draft", default=None, help="BP-04 checkpoint file or directory (default: day-0 draft)")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("-k", type=int, default=3, help="guesses per round")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sampling", action="store_true", help="speculative sampling instead of greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true", help="cache-free target (slow; for debugging)")
    parser.add_argument("--bf16", action="store_true", help="bf16 target and draft (default fp32)")
    parser.add_argument("--check", action="store_true", help="compare each output with target-only greedy")
    parser.add_argument("--log", default=None, help="append one JSON line per request")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    target, tokenizer = load_target(device=device, dtype=dtype), load_tokenizer()
    drafter = Drafter(build_draft(target, load_vocab_mapping(args.vocab), device=device, dtype=dtype), k=args.k)
    if args.draft:
        print(f"draft {drafter.load_file(args.draft)} (version {drafter.version})")
    ids = heldout_ids() if args.heldout else None
    rows = load_requests(args.stream, args.n, args.domain, ids)
    if args.heldout:
        print(f"{len(rows)} held-out chat requests")

    log = RequestLog(args.log) if args.log else None
    try:
        records = serve(
            target,
            drafter,
            tokenizer,
            rows,
            max_new_tokens=args.max_new_tokens,
            greedy=not args.sampling,
            temperature=args.temperature,
            use_cache=not args.no_cache,
            seed=args.seed,
            check=args.check,
            log=log,
            verbose=True,
        )
    finally:
        if log is not None:
            log.close()
    print(json.dumps(summarize(records), indent=2))
    if args.log:
        print(f"log → {args.log}")
    if args.check and not all(r["match"] for r in records):
        sys.exit(1)


if __name__ == "__main__":
    main()
