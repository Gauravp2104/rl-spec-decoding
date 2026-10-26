"""Collect run summaries into the BP-08 results table (markdown).

    python -m mini_aurora.experiments.table                    # print
    python -m mini_aurora.experiments.table --out results.md   # write

Speed-up is tokens/s relative to the ``target_only`` arm of the same experiment and stream, when present.
"""

import argparse
import json
from pathlib import Path

from mini_aurora.experiments import OUT_DIR

COLUMNS = ["exp", "arm", "stream", "n", "τ̄", "α1 / α2 / α3", "tok/s", "× target-only", "lossless", "draft"]


def load_summaries(out_dir: str | Path = OUT_DIR) -> list[dict]:
    return sorted((json.loads(p.read_text()) for p in Path(out_dir).glob("E*/*.json")),
                  key=lambda s: (s["experiment"], s["stream"], s["arm"], s.get("draft") or ""))


def rows(summaries: list[dict]) -> list[list[str]]:
    reference = {(s["experiment"], s["stream"]): s["overall"]["tokens_per_s"]
                 for s in summaries if s["arm"] == "target_only"}
    table = []
    for s in summaries:
        o = s["overall"]
        ref = reference.get((s["experiment"], s["stream"]))
        table.append([
            s["experiment"], s["arm"], s["stream"], str(o["requests"]),
            f"{o['tau']:.3f}" if "tau" in o else "–",
            " / ".join(f"{a:.3f}" for a in o["alpha"]) if "alpha" in o else "–",
            f"{o['tokens_per_s']:.1f}",
            f"×{o['tokens_per_s'] / ref:.2f}" if ref else "–",
            o.get("matches", "–"),
            Path(s["draft"]).name if s.get("draft") else "–",
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
