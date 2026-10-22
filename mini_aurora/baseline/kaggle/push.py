"""Run BP-04 on Kaggle's free T4 and pull the results back.

1. Packages the repo source (mini_aurora/, aurora-main/aurora, examples/, pyproject files) together with
   ``data/raw/chat.jsonl`` and ``outputs/vocab/`` into the private Kaggle dataset ``<user>/rl-spec-decoding-src``
   (kernels can attach datasets, not git branches).
2. Pushes ``run_bp04.py`` as a private GPU script kernel ``<user>/rl-spec-decoding-bp04`` with internet on
   (Qwen3-0.6B download), polls until it finishes, then downloads ``/kaggle/working/bp04/`` into
   ``outputs/baseline/kaggle/``.

    python mini_aurora/baseline/kaggle/push.py --user <kaggle user>                # full run: 2 epochs, fp32 draft
    python mini_aurora/baseline/kaggle/push.py --user <kaggle user> --limit 20     # smoke run
    python mini_aurora/baseline/kaggle/push.py --user <kaggle user> --bf16
    python mini_aurora/baseline/kaggle/push.py --user <kaggle user> --pull-only

Needs the Kaggle CLI (``pip install kaggle``) authenticated via ~/.kaggle/kaggle.json or ~/.kaggle/access_token.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
KAGGLE = [sys.executable, "-m", "kaggle"]
DATASET = "rl-spec-decoding-src"
KERNEL = "rl-spec-decoding-bp04"


def kaggle(*args: str, check: bool = True) -> str:
    r = subprocess.run([*KAGGLE, *args], text=True, capture_output=True)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def push_source(user: str) -> str:
    slug = f"{user}/{DATASET}"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "outputs", "data")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        shutil.copytree(ROOT / "mini_aurora", d / "mini_aurora", ignore=ignore)
        shutil.copytree(ROOT / "examples", d / "examples", ignore=ignore)
        shutil.copytree(ROOT / "aurora-main" / "aurora", d / "aurora-main" / "aurora", ignore=ignore)
        for f in ("pyproject.toml", "version.txt", "README.md"):
            shutil.copy(ROOT / "aurora-main" / f, d / "aurora-main" / f)
        shutil.copy(ROOT / "pyproject.toml", d / "pyproject.toml")
        (d / "data" / "raw").mkdir(parents=True)
        shutil.copy(ROOT / "data" / "raw" / "chat.jsonl", d / "data" / "raw" / "chat.jsonl")
        shutil.copytree(ROOT / "outputs" / "vocab", d / "outputs" / "vocab")
        (d / "dataset-metadata.json").write_text(json.dumps({"title": DATASET, "id": slug, "licenses": [{"name": "CC0-1.0"}]}))
        exists = DATASET in kaggle("datasets", "list", "--mine", check=False)
        out = kaggle("datasets", "version" if exists else "create", "-p", str(d), *(["-m", "update"] if exists else []),
                     "--dir-mode", "zip", check=False)
        print(out.strip().splitlines()[-1])
    time.sleep(20)  # the new version takes a moment to become attachable
    return slug


def push_kernel(user: str, src_slug: str, epochs: int, bf16: bool, limit: int | None) -> str:
    kid = f"{user}/{KERNEL}"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        script = (HERE / "run_bp04.py").read_text()
        env = {"BP04_EPOCHS": str(epochs), "BP04_BF16": "1" if bf16 else "0", "BP04_LIMIT": str(limit or "")}
        header = "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
        (d / "run_bp04.py").write_text(header + script)
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": KERNEL, "code_file": "run_bp04.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [src_slug], "competition_sources": [], "kernel_sources": []}))
        print(kaggle("kernels", "push", "-p", str(d)).strip().splitlines()[-1])
    return kid


def wait(kid: str, poll: int = 60) -> str:
    t0 = time.time()
    while time.time() - t0 < 300:  # the status endpoint lags behind a push
        status = kaggle("kernels", "status", kid, check=False).strip().splitlines()[-1].lower()
        if "queued" in status or "running" in status:
            break
        time.sleep(10)
    while True:
        status = kaggle("kernels", "status", kid, check=False).strip().splitlines()[-1]
        print(f"  {time.time() - t0:6.0f} s  {status}", flush=True)
        if any(k in status.lower() for k in ("complete", "error", "cancel")):
            return status
        time.sleep(poll)


def pull(kid: str) -> None:
    out = ROOT / "outputs" / "baseline" / "kaggle"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    print(kaggle("kernels", "output", kid, "-p", str(out)).strip().splitlines()[-1])
    log = next(out.glob("*.log"), None)
    if log:
        lines = []
        for line in log.read_text().splitlines():
            try:
                o = json.loads(line.strip().lstrip(","))
            except json.JSONDecodeError:
                continue
            if o.get("stream_name") == "stdout" or "rror" in o.get("data", ""):
                lines.append(o["data"].rstrip())
        (out / "kaggle_run.log").write_text("\n".join(lines))
        print("\n".join(lines[-30:]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="Kaggle username")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bf16", action="store_true", help="bf16 draft weights (default fp32)")
    ap.add_argument("--limit", type=int, default=None, help="conversations to dump (smoke run)")
    ap.add_argument("--pull-only", action="store_true")
    a = ap.parse_args()
    kid = f"{a.user}/{KERNEL}"
    if not a.pull_only:
        slug = push_source(a.user)
        kid = push_kernel(a.user, slug, a.epochs, a.bf16, a.limit)
        print("final:", wait(kid))
    pull(kid)


if __name__ == "__main__":
    main()
