"""Run one arm of one base-paper experiment over a stream and write its log + summary.

    python -m mini_aurora.experiments.run E2 static                         # BP-04 draft over the domain-shift stream
    python -m mini_aurora.experiments.run E2 target_only                    # target-only speed reference, same stream
    python -m mini_aurora.experiments.run E1 day0 -n 300 --check            # random draft, lossless check on
    python -m mini_aurora.experiments.run E3 static --draft outputs/baseline/checkpoints_8ep --tag 8ep
    python -m mini_aurora.experiments.run E2 online_static                  # BP-06 loop, --start draft, sync
    python -m mini_aurora.experiments.run E4 online_day0 --grid N5            # one E4 setting (E4_GRID)
    python -m mini_aurora.experiments.run E1 online_day0 --loop-args "--mode async --sync-every 1"

Writes ``outputs/experiments/<E>/<arm>_<stream>[_<tag>]/``: ``requests.jsonl`` (one record per request, BP-05's
log format) and ``summary.json`` (overall, per domain, per ``WINDOW`` requests along the stream). Online arms
run ``python -m mini_aurora.online.loop`` (BP-06) with that directory as ``--out``, so it also holds the loop's
``train_log.csv`` / ``config.json`` and BP-07's ``eval/`` can be produced from it
(``python -m mini_aurora.eval.plot <dir>``).
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from mini_aurora.data import DATA_DIR
from mini_aurora.device import get_device
from mini_aurora.experiments import ARMS, E4_GRID, EXPERIMENTS, ONLINE_ARMS, OUT_DIR, STATIC_DRAFT, WINDOW
from mini_aurora.models.draft import build_draft
from mini_aurora.models.target import load_target, load_tokenizer
from mini_aurora.models.vocab import DEFAULT_VOCAB, load_vocab_mapping
from mini_aurora.server.drafter import Drafter
from mini_aurora.server.log import RequestLog, acceptance_by_position, read_log
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


def loop_command(arm: str, stream_path: str | Path, out_dir: str | Path, n: int | None, draft: str | None,
                 k: int, max_new_tokens: int, check: bool, extra: list[str] | None = None) -> list[str]:
    """BP-06 loop invocation for an online arm (``--start day0`` or ``--start draft --draft <ckpt>``)."""
    cmd = [sys.executable, "-m", "mini_aurora.online.loop", "--stream", str(stream_path), "--out", str(out_dir),
           "--overwrite", "-k", str(k), "--max-new-tokens", str(max_new_tokens), "--quiet"]
    cmd += ["-n", str(n if n else 10**9)]
    cmd += ["--start", "day0"] if arm == "online_day0" else ["--start", "draft", "--draft", str(draft)]
    if check:
        cmd.append("--check")
    return cmd + list(extra or [])


def run_online(arm: str, stream_path: Path, out_dir: Path, n: int | None, draft: str | None, k: int,
               max_new_tokens: int, check: bool, extra: list[str] | None = None) -> tuple[list[dict], dict]:
    """Run BP-06's loop into ``out_dir`` and return its records and its own summary."""
    if importlib.util.find_spec("mini_aurora.online") is None:
        raise SystemExit(f"{arm} needs BP-06's online loop (mini_aurora.online, #14), which is not in this checkout")
    cmd = loop_command(arm, stream_path, out_dir, n, draft, k, max_new_tokens, check, extra)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return read_log(out_dir / "requests.jsonl"), json.loads((out_dir / "summary.json").read_text())


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
    p.add_argument("--tag", default=None, help="suffix for the output directory")
    p.add_argument("--grid", default=None, choices=sorted(E4_GRID), help="E4 setting (adds its loop arguments and tag)")
    p.add_argument("--loop-args", default="", help="extra arguments for mini_aurora.online.loop (online arms)")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    exp = EXPERIMENTS[args.experiment]
    if args.arm not in exp["arms"]:
        p.error(f"{args.experiment} has arms {exp['arms']}")
    if args.grid and args.experiment != "E4":
        p.error("--grid is for E4")
    extra = (E4_GRID[args.grid] if args.grid else []) + args.loop_args.split()
    tag = "_".join(t for t in (args.grid, args.tag) if t)
    stream = args.stream or exp["stream"]
    stream_path = Path(DATA_DIR) / "streams" / f"{stream}.jsonl"
    rows = load_requests(stream_path, args.n or 10**9)
    out = Path(args.out_dir) / args.experiment / (f"{args.arm}_{stream}" + (f"_{tag}" if tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    meta = {"experiment": args.experiment, "claim": exp["claim"], "arm": args.arm, "arm_desc": ARMS[args.arm],
            "stream": stream, "requests": len(rows), "k": args.k, "max_new_tokens": args.max_new_tokens,
            "device": device.type, "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
            "draft": str(args.draft) if args.arm in ("static", "online_static") else None,
            "loop_args": extra if args.arm in ONLINE_ARMS else None}
    print(json.dumps(meta, indent=2), flush=True)
    t0 = time.perf_counter()
    if args.arm in ONLINE_ARMS:
        records, loop_summary = run_online(args.arm, stream_path, out, args.n, args.draft, args.k,
                                           args.max_new_tokens, args.check, extra)
        meta["loop"] = {k: v for k, v in loop_summary.items() if k != "domains"}
    else:
        target, tokenizer = load_target(device=device, dtype=torch.float32), load_tokenizer()
        with RequestLog(out / "requests.jsonl") as log:
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
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("overall", "by_domain")}, indent=2))
    print(f"→ {out / 'summary.json'}")


if __name__ == "__main__":
    main()
