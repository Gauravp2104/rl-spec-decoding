"""Plots for an online run (BP-07 step 3), from the same metrics as ``mini_aurora.eval.metrics``.

    python -m mini_aurora.eval.plot outputs/online/day0_sync_300
    python -m mini_aurora.eval.plot outputs/online/day0_sync_300 --window 25

Recomputes ``<run>/eval/metrics.csv`` and ``<run>/eval/summary.json``, then writes to ``<run>/eval/``:

- ``rolling_tau.png``: rolling τ̄ (trailing W requests) vs request index, per-request τ̄ as dots, and a strip of
  each request's domain underneath (contiguous same-domain segments read as blocks on ordered / shift streams);
- ``alpha.png``: α_i for the first and last W requests and overall;
- ``staleness.png``: histogram of Δ = trainer version − served version, per round;
- ``loss.png``: weighted training loss and per-position accuracy vs optimizer step (two panels, one axis each;
  raw steps faint, a ``SMOOTH``-step moving average on top: each step sees only a few samples);
- ``tokens_per_s.png``: rolling tok/s vs request index, against target-only decoding when known.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from mini_aurora.eval.metrics import (  # noqa: E402
    EVAL_DIR,
    evaluate_run,
    load_run,
    loss_curve,
    rolling_tau,
    staleness_histogram,
    target_tokens_per_s,
    tokens_per_s,
    trailing,
)
from mini_aurora.server.log import acceptance_by_position  # noqa: E402

# Light-mode chart tokens; categorical slots in fixed order (validated all-pairs for 3 slots).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DOMAINS = ("chat", "code", "math")
SMOOTH = 10  # optimizer steps in the loss / accuracy moving average


def moving_average(values: list[float], window: int) -> list[float]:
    """Trailing mean over the last ``window`` values (shorter at the start)."""
    return [sum(values[max(0, i - window + 1) : i + 1]) / (i - max(0, i - window + 1) + 1) for i in range(len(values))]


def _style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def _title(ax, text: str) -> None:
    ax.set_title(text, color=INK, fontsize=11, loc="left")


def domain_colors(records: list[dict]) -> dict:
    seen = [d for d in DOMAINS if any(r.get("domain") == d for r in records)]
    seen += sorted({r.get("domain") for r in records} - set(seen), key=str)
    return {d: SERIES[i % len(SERIES)] for i, d in enumerate(seen)}


def plot_rolling_tau(records: list[dict], window: int, path: Path) -> None:
    fig, (ax, strip) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True, gridspec_kw={"height_ratios": [12, 1], "hspace": 0.08}, facecolor=SURFACE)
    x = list(range(len(records)))
    ax.scatter(x, [r["mean_accepted"] for r in records], s=8, color=AXIS, linewidths=0, label="per request", zorder=2)
    ax.plot(x, rolling_tau(records, window), color=INK, linewidth=2, label=f"rolling τ̄ (last {window})", zorder=3)
    _style(ax, "", "accepted guesses per round (τ̄)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _title(ax, f"Rolling τ̄ over {len(records)} requests")

    colors = domain_colors(records)
    strip.bar(x, [1] * len(x), width=1.0, color=[colors[r.get("domain")] for r in records], linewidth=0)
    strip.set_yticks([])
    strip.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        strip.spines[side].set_visible(False)
    strip.spines["bottom"].set_color(AXIS)
    strip.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    strip.set_xlabel("request index", color=INK_2, fontsize=10)
    strip.set_xlim(-0.5, len(records) - 0.5)
    fig.legend(handles=[Patch(color=c, label=str(d)) for d, c in colors.items()], frameon=False, fontsize=9,
               labelcolor=INK_2, loc="lower right", ncol=len(colors), bbox_to_anchor=(0.98, 0.0))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_alpha(records: list[dict], window: int, path: Path) -> None:
    groups = [(f"first {window}", records[:window]), (f"last {window}", records[-window:]), ("all", records)]
    k = max(r["k"] for r in records)
    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor=SURFACE)
    width = 0.26
    for j, (name, rs) in enumerate(groups):
        alpha = acceptance_by_position(rs)
        xs = [i + (j - 1) * (width + 0.02) for i in range(k)]
        bars = ax.bar(xs, alpha, width=width, color=SERIES[j], label=name, linewidth=0)
        ax.bar_label(bars, labels=[f"{a:.3f}" for a in alpha], fontsize=8, color=INK_2, padding=2)
    ax.set_xticks(range(k), [f"α{i + 1}" for i in range(k)])
    _style(ax, "guess position", "fraction of rounds accepted")
    ax.set_ylim(0, max(0.05, ax.get_ylim()[1] * 1.15))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    _title(ax, "Per-position acceptance αᵢ")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_staleness(records: list[dict], path: Path) -> None:
    hist = staleness_histogram(records)
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
    if hist:
        total = sum(hist.values())
        bars = ax.bar(list(hist), [n / total for n in hist.values()], width=0.8, color=SERIES[0], linewidth=0)
        ax.bar_label(bars, labels=[f"{n / total:.0%}" for n in hist.values()], fontsize=8, color=INK_2, padding=2)
        ax.set_xticks(list(hist))
    else:
        ax.text(0.5, 0.5, "no trainer versions in this log", ha="center", va="center", color=MUTED, transform=ax.transAxes)
    _style(ax, "Δ = trainer version − served version (optimizer steps)", "fraction of rounds")
    _title(ax, "Staleness of the served draft")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_loss(train: list[dict], path: Path) -> None:
    curve = loss_curve(train)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    if curve:
        steps = curve["step"]
        ax_loss.plot(steps, curve["loss"], color=AXIS, linewidth=1, label="per step")
        ax_loss.plot(steps, moving_average(curve["loss"], SMOOTH), color=INK, linewidth=2, label=f"mean of last {SMOOTH}")
        ax_loss.legend(frameon=False, fontsize=9, labelcolor=INK_2)
        accs = sorted(k for k in curve if k.startswith("acc_"))
        for i, key in enumerate(accs):
            ax_acc.plot(steps, curve[key], color=SERIES[i], linewidth=0.8, alpha=0.3)
            ax_acc.plot(steps, moving_average(curve[key], SMOOTH), color=SERIES[i], linewidth=2,
                        label=f"position {int(key[4:]) + 1}")
        ax_acc.legend(frameon=False, fontsize=9, labelcolor=INK_2, title=f"mean of last {SMOOTH} steps",
                      title_fontsize=8)
        for ax in (ax_loss, ax_acc):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(ax_loss, "optimizer step", "weighted loss  Σ 0.8^i · loss_i")
    _style(ax_acc, "optimizer step", "draft top-1 accuracy (training batch)")
    ax_acc.set_ylim(bottom=0)
    _title(ax_loss, "Training loss")
    _title(ax_acc, "Training accuracy per TTT position")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_tokens_per_s(records: list[dict], window: int, bench: dict | None, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    x = list(range(len(records)))
    ax.plot(x, [tokens_per_s(w) for w in trailing(records, window)], color=INK, linewidth=2)
    reference = target_tokens_per_s(records, bench)
    if reference:
        ax.axhline(reference, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(f"target only {reference:.1f} tok/s", (len(records) - 1, reference), xytext=(0, 4),
                    textcoords="offset points", ha="right", va="bottom", fontsize=9, color=INK_2)
    _style(ax, "request index", f"tokens/s (last {window} requests)")
    ax.set_ylim(bottom=0)
    _title(ax, "Decoding throughput")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_run(run_dir: str | Path, window: int = 50) -> list[Path]:
    """Write the metrics and every plot for ``run_dir``; returns the image paths."""
    summary = evaluate_run(run_dir, window)
    run = load_run(run_dir)
    records, out = run["records"], Path(run_dir) / EVAL_DIR
    paths = [out / name for name in ("rolling_tau.png", "alpha.png", "staleness.png", "loss.png", "tokens_per_s.png")]
    plot_rolling_tau(records, window, paths[0])
    plot_alpha(records, window, paths[1])
    plot_staleness(records, paths[2])
    plot_loss(run["train"], paths[3])
    plot_tokens_per_s(records, window, run["bench"], paths[4])
    print(f"τ̄ {summary['all']['tau']}  first {window}: {summary['all'][f'tau_first_{window}']}  "
          f"last {window}: {summary['all'][f'tau_last_{window}']}")
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="online run directory (BP-06 --out)")
    p.add_argument("--window", type=int, default=50)
    args = p.parse_args()
    for path in plot_run(args.run, args.window):
        print(f"plot -> {path}")


if __name__ == "__main__":
    main()
