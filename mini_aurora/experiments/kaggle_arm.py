"""One BP-08 arm on Kaggle (T4, fp32): downloads the frozen BP-04 draft from the release, runs the arm, keeps the run dir.

Kernel script; BP08_EXP / BP08_ARM / BP08_GRID / BP08_N / BP08_LOOP_ARGS are set by ``kaggle.py``. The source
dataset ``rl-spec-decoding-src`` (``mini_aurora/baseline/kaggle/push.py``) must be attached and internet on.
Output: ``/kaggle/working/bp08/<EXP>/<arm>_<stream>[_<tag>]/`` (requests.jsonl, summary.json, loop files).
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "bp08"
REPO = Path("/tmp/repo")
RELEASE = "https://github.com/Gauravp2104/rl-spec-decoding/releases/download/bp04-static-v1"


def find_source() -> Path:
    for p in Path("/kaggle/input").rglob("pyproject.toml"):
        if (p.parent / "mini_aurora").is_dir():
            return p.parent
    raise SystemExit("rl-spec-decoding-src dataset not attached")


def run(cmd: list[str], **env) -> float:
    print("+", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=REPO, env={**os.environ, **env})
    return round(time.perf_counter() - t0, 1)


def main() -> None:
    src = find_source()
    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(src, REPO, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    OUT.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "omegaconf", "pydantic", "numba",
         "psutil", "wandb", "accelerate", "datasets", "numpy<2.4"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "aurora-main", "--no-deps", "-e", ".", "--no-deps"])
    ckpt = REPO / "outputs" / "baseline" / "checkpoints_8ep"
    ckpt.mkdir(parents=True, exist_ok=True)
    for name in ("draft_step000896.pt", "latest.txt"):
        run(["curl", "-sSL", "-o", str(ckpt / name), f"{RELEASE}/{name}"])
    subprocess.run(["sha256sum", str(ckpt / "draft_step000896.pt")], check=True)

    exp, arm = os.environ["BP08_EXP"], os.environ["BP08_ARM"]
    cmd = [sys.executable, "-m", "mini_aurora.experiments.run", exp, arm, "--out-dir", str(OUT),
           "--draft", "outputs/baseline/checkpoints_8ep"]
    if os.environ.get("BP08_GRID"):
        cmd += ["--grid", os.environ["BP08_GRID"]]
    if os.environ.get("BP08_N"):
        cmd += ["-n", os.environ["BP08_N"]]
    if os.environ.get("BP08_LOOP_ARGS"):
        cmd += ["--loop-args", os.environ["BP08_LOOP_ARGS"]]
    env = {"MINI_AURORA_DEVICE": "cuda", "WANDB_MODE": "offline", "HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false"}
    seconds = run(cmd, **env)
    print(f"arm done in {seconds} s")


if __name__ == "__main__":
    main()
