"""Overlay plots for the BP-08 experiments: rolling τ̄ of several arms over the same stream.

    python -m mini_aurora.experiments.plot_compare E2 outputs/experiments_kaggle/E2/static_domain_shift \
        outputs/experiments_kaggle/E2/online_static_domain_shift --labels static online --out figures/E2_rolling_tau.png

Each run directory holds a ``requests.jsonl`` (BP-05 log format). The domain of each request is drawn as a strip
under the axis (contiguous segments read as blocks on the ordered / domain-shift streams). Rolling τ̄ is
round-weighted over the trailing ``--window`` requests, as in ``mini_aurora.eval.metrics``.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mini_aurora.server.log import read_log  # noqa: E402

COLORS = ["#8c8c8c", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
DOMAIN_COLORS = {"chat": "#3b78d8", "code": "#e06a2b", "math": "#2aa876"}


def rolling_tau(records: list[dict], window: int) -> list[float]:
    out = []
    for i in range(len(records)):
        rounds = [a for r in records[max(0, i - window + 1): i + 1] for a in r["accepted"]]
        out.append(sum(rounds) / len(rounds) if rounds else 0.0)
    return out


def plot(title: str, runs: list[Path], labels: list[str], out: Path, window: int = 100) -> Path:
    fig, (ax, strip) = plt.subplots(2, 1, figsize=(9, 4.6), gridspec_kw={"height_ratios": [12, 1]}, sharex=True)
    first = None
    for i, (run, label) in enumerate(zip(runs, labels)):
        recs = read_log(run / "requests.jsonl")
        first = first or recs
        ax.plot(range(len(recs)), rolling_tau(recs, window), color=COLORS[i % len(COLORS)], lw=2, label=label)
    ax.set_ylabel(f"rolling τ̄ (last {window})")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    start = 0
    for i in range(1, len(first) + 1):
        if i == len(first) or first[i].get("domain") != first[start].get("domain"):
            strip.axvspan(start, i, color=DOMAIN_COLORS.get(first[start].get("domain"), "#999"), lw=0)
            start = i
    strip.set_yticks([])
    strip.set_xlabel("request index")
    for spine in strip.spines.values():
        spine.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DOMAIN_COLORS.values()]
    strip.legend(handles, list(DOMAIN_COLORS), ncol=3, frameon=False, loc="upper right", bbox_to_anchor=(1, -0.6), fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("title")
    p.add_argument("runs", nargs="+")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=100)
    a = p.parse_args()
    if len(a.labels) != len(a.runs):
        p.error("one label per run")
    print("plot ->", plot(a.title, [Path(r) for r in a.runs], a.labels, Path(a.out), a.window))


if __name__ == "__main__":
    main()
