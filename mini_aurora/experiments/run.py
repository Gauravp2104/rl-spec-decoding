"""Run one arm of one base-paper experiment over a stream and write its log + summary.

    python -m mini_aurora.experiments.run E2 static                         # BP-04 draft over the domain-shift stream
    python -m mini_aurora.experiments.run E2 target_only                    # target-only speed reference, same stream
    python -m mini_aurora.experiments.run E1 day0 -n 300 --check            # random draft, lossless check on
    python -m mini_aurora.experiments.run E3 static --draft outputs/baseline/checkpoints_8ep --tag 8ep

Writes ``outputs/experiments/<E>/<arm>[_<tag>].jsonl`` (one record per request, BP-05's log format) and
``.json`` (summary: overall, per domain, per ``WINDOW`` requests along the stream). ``online_*`` arms need
BP-06's loop and are refused here until it lands.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from mini_aurora.data import DATA_DIR
from mini_aurora.device import get_device
from mini_aurora.experiments import ARMS, EXPERIMENTS, ONLINE_ARMS, OUT_DIR, STATIC_DRAFT, WINDOW
from mini_aurora.models.draft import build_draft
from mini_aurora.models.target import load_target, load_tokenizer
from mini_aurora.models.vocab import DEFAULT_VOCAB, load_vocab_mapping
from mini_aurora.server.drafter import Drafter
from mini_aurora.server.log import RequestLog, acceptance_by_position
from mini_aurora.server.reference import encode_messages, reference_generate
from mini_aurora.server.serve import load_requests, serve


def serve_target_only(target, tokenizer, rows: list[dict], max_new_tokens: int, log: RequestLog | None = None) -> list[dict]:
    """Target greedy decoding alone; records carry the same keys as the server log minus the draft fields."""
    records = []
    for row in rows:
        ids = encode_messages(tokenizer, row["messages"], target.model.device)
        ref = reference_generate(target, ids, max_new_tokens)
        record = {"request_id": row["id"], "domain": row.get("domain"), "pos": row.get("pos"),
                  "prompt_tokens": ids.shape[1], "new_tokens": ref.new_ids.numel(),
                  "seconds": round(ref.seconds, 4), "tokens_per_s": round(ref.tokens_per_s, 2)}
        records.append(record)
        if log is not None:
            log.write(record)
    return records


def _stats(group: list[dict]) -> dict:
    tokens, seconds = sum(r["new_tokens"] for r in group), sum(r["seconds"] for r in group)
    out = {"requests": len(group), "new_tokens": tokens, "tokens_per_s": round(tokens / seconds, 1) if seconds else 0.0}
    if group and "accepted" in group[0]:
        rounds = [a for r in group for a in r["accepted"]]
        out["tau"] = round(sum(rounds) / len(rounds), 3) if rounds else 0.0
        out["alpha"] = [round(a, 3) for a in acceptance_by_position(group)]
        out["tokens_per_round"] = round(tokens / len(rounds), 3) if rounds else 0.0
    if group and "match" in group[0]:
        out["matches"] = f"{sum(r['match'] for r in group)}/{len(group)}"
    return out


def summarize(records: list[dict], window: int = WINDOW) -> dict:
    """Overall, per-domain and per-window (stream order) statistics of a run's records."""
    by_domain = defaultdict(list)
    for r in records:
        by_domain[r.get("domain")].append(r)
    windows = []
    for start in range(0, len(records), window):
        chunk = records[start:start + window]
        windows.append({"start": start, "end": start + len(chunk),
                        "domains": sorted({r.get("domain") for r in chunk if r.get("domain")}), **_stats(chunk)})
    return {"overall": _stats(records), "by_domain": {d: _stats(g) for d, g in by_domain.items()}, "by_window": windows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=sorted(EXPERIMENTS))
    p.add_argument("arm", choices=sorted(ARMS))
    p.add_argument("--stream", default=None, help="stream name under data/streams (default: the experiment's)")
    p.add_argument("-n", type=int, default=None, help="requests to serve (default: the whole stream)")
    p.add_argument("--draft", default=STATIC_DRAFT, help="BP-04 checkpoint file or directory for the static arm")
    p.add_argument("--vocab", default=DEFAULT_VOCAB)
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--check", action="store_true", help="also decode target-only and record losslessness")
    p.add_argument("--tag", default=None, help="suffix for the output files")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    exp = EXPERIMENTS[args.experiment]
    if args.arm not in exp["arms"]:
        p.error(f"{args.experiment} has arms {exp['arms']}")
    if args.arm in ONLINE_ARMS:
        raise SystemExit(f"{args.arm} needs BP-06's online loop (#14); not available yet")
    stream = args.stream or exp["stream"]
    rows = load_requests(Path(DATA_DIR) / "streams" / f"{stream}.jsonl", args.n or 10**9)
    out = Path(args.out_dir) / args.experiment
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.arm}_{stream}" + (f"_{args.tag}" if args.tag else "")

    device = get_device(args.device)
    target, tokenizer = load_target(device=device, dtype=torch.float32), load_tokenizer()
    meta = {"experiment": args.experiment, "claim": exp["claim"], "arm": args.arm, "arm_desc": ARMS[args.arm],
            "stream": stream, "requests": len(rows), "k": args.k, "max_new_tokens": args.max_new_tokens,
            "device": device.type, "draft": None}
    print(json.dumps(meta, indent=2), flush=True)
    t0 = time.perf_counter()
    with RequestLog(out / f"{name}.jsonl") as log:
        if args.arm == "target_only":
            records = serve_target_only(target, tokenizer, rows, args.max_new_tokens, log)
        else:
            drafter = Drafter(build_draft(target, load_vocab_mapping(args.vocab), device=device, dtype=torch.float32), k=args.k)
            if args.arm == "static":
                meta["draft"] = str(drafter.load_file(args.draft))
                meta["draft_version"] = drafter.version
            records = serve(target, drafter, tokenizer, rows, max_new_tokens=args.max_new_tokens,
                            check=args.check, log=log, verbose=False)
    summary = {**meta, "elapsed_s": round(time.perf_counter() - t0, 1), **summarize(records)}
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("overall", "by_domain")}, indent=2))
    print(f"→ {out / name}.json")


if __name__ == "__main__":
    main()
