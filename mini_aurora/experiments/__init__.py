"""BP-08: the base-paper experiments (PLAN.md E1-E4) on BP-02's streams.

Each experiment is a set of *arms* served over one stream; ``run.py`` runs one arm and writes a per-request
log plus a summary, ``table.py`` collects the summaries into the results table. Static arms go through BP-05's
``serve``; online arms run BP-06's loop (``python -m mini_aurora.online.loop``) and summarize its
``requests.jsonl``; BP-07's metrics and plots read the same run directory.
"""

OUT_DIR = "outputs/experiments"
WINDOW = 100  # requests per window for τ̄ vs request index
STATIC_DRAFT = "outputs/baseline/checkpoints_8ep"  # BP-04's frozen static baseline

# arm -> how the draft starts and whether it trains while serving
ARMS = {
    "target_only": "no draft: target greedy decoding (the speed reference)",
    "day0": "random (day-0) draft, no training",
    "static": "BP-04 static draft, no training",
    "online_day0": "random draft, trained online (BP-06 loop, --start day0)",
    "online_static": "BP-04 draft, trained online (BP-06 loop, --start draft)",
}
ONLINE_ARMS = {"online_day0", "online_static"}

EXPERIMENTS = {
    "E1": {"claim": "Day-0 deployment works", "stream": "mixed", "arms": ["target_only", "day0", "online_day0"]},
    "E2": {"claim": "Online adaptation beats static under shift", "stream": "domain_shift",
           "arms": ["target_only", "static", "online_static"]},
    "E3": {"claim": "Ordered-stream behaviour", "stream": "ordered", "arms": ["target_only", "static", "online_static"]},
    "E4": {"claim": "Sensitivity (sync interval, LR, sync vs async)", "stream": "mixed", "arms": ["online_day0"]},
}

# E4 grid (PLAN.md): publish interval N, learning rate, sync vs async; each a tag + extra loop arguments.
E4_GRID = {
    "N5": ["--sync-every", "5"],
    "N20": ["--sync-every", "20"],
    "N50": ["--sync-every", "50"],
    "lr3e-4": ["--lr", "3e-4"],
    "lr3e-5": ["--lr", "3e-5"],
    "async": ["--mode", "async"],
}
