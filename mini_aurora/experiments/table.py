"""Collect run summaries into the BP-08 results table (markdown).

    python -m mini_aurora.experiments.table                    # print
    python -m mini_aurora.experiments.table --out results.md   # write

Speed-up is tokens/s relative to the ``target_only`` arm of the same experiment, stream and device, when present
(tok/s is hardware-bound, so arms are only compared on the same machine; τ̄ is the hardware-independent number).
"""

import argparse
import json
from pathlib import Path

from mini_aurora.experiments import OUT_DIR

COLUMNS = ["exp", "arm", "stream", "n", "τ̄", "τ̄ first/last 100", "α1 / α2 / α3", "tok/s", "× target-only", "device", "lossless", "draft / loop args"]


def load_summaries(out_dir: str | Path = OUT_DIR) -> list[dict]:
    return sorted((json.loads(p.read_text()) for p in Path(out_dir).glob("E*/*/summary.json")),
                  key=lambda s: (s["experiment"], s["stream"], s["arm"], " ".join(s.get("loop_args") or []), s.get("draft") or ""))


def rows(summaries: list[dict]) -> list[list[str]]:
    reference = {(s["experiment"], s["stream"], s.get("device_name") or s.get("device")): s["overall"]["tokens_per_s"]
                 for s in summaries if s["arm"] == "target_only"}
    table = []
    for s in summaries:
        o = s["overall"]
        device = s.get("device_name") or s.get("device") or "–"
        ref = reference.get((s["experiment"], s["stream"], device))
        windows = s.get("by_window") or []
        first_last = (f"{windows[0]['tau']:.3f} / {windows[-1]['tau']:.3f}"
                      if len(windows) > 1 and "tau" in windows[0] and "tau" in windows[-1] else "–")
        setting = " ".join(s["loop_args"]) if s.get("loop_args") else ""
        draft = Path(s["draft"]).name if s.get("draft") else ""
        table.append([
            s["experiment"], s["arm"], s["stream"], str(o["requests"]),
            f"{o['tau']:.3f}" if "tau" in o else "–",
            first_last,
            " / ".join(f"{a:.3f}" for a in o["alpha"]) if "alpha" in o else "–",
            f"{o['tokens_per_s']:.1f}",
            f"×{o['tokens_per_s'] / ref:.2f}" if ref else "–",
            device,
            o.get("matches", "–"),
            " ".join(x for x in (draft, setting) if x) or "–",
        ])
    return table


def markdown(table: list[list[str]], columns: list[str] = COLUMNS) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(r) + " |" for r in table]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    text = markdown(rows(load_summaries(args.out_dir)))
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
