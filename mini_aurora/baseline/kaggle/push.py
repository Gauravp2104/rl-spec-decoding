"""Run BP-04 on Kaggle's free T4 and pull the results back.

1. Packages the repo source (mini_aurora/, aurora-main/aurora, examples/, pyproject files) together with
   ``data/raw/chat.jsonl``, ``data/streams/mixed.jsonl`` (for BP-05's server) and ``outputs/vocab/`` (24K mapping) into the private Kaggle dataset ``<user>/rl-spec-decoding-src``
   (kernels can attach datasets, not git branches).
2. Pushes ``run_bp04.py`` as a private GPU script kernel ``<user>/rl-spec-decoding-bp04`` with internet on
   (Qwen3-0.6B download), polls until it finishes, then downloads ``/kaggle/working/bp04/`` into
   ``outputs/baseline/kaggle/<label>/`` (``--label``, default ``<dtype>[_ckpt]_<epochs>ep``; an existing folder is replaced).

    python mini_aurora/baseline/kaggle/push.py --user <kaggle user>                # full run: 8 epochs, fp32 draft + grad ckpt
    python mini_aurora/baseline/kaggle/push.py --user <kaggle user> --limit 20     # smoke run
    python mini_aurora/baseline/kaggle/push.py --user <kaggle user> --dtype bf16
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


def push_source(user: str, chat_jsonl: Path | None = None, tag: str = "") -> str:
    slug = f"{user}/{DATASET}{tag}"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        shutil.copytree(ROOT / "mini_aurora", d / "mini_aurora", ignore=ignore)
        shutil.copytree(ROOT / "examples", d / "examples", ignore=ignore)
        shutil.copytree(ROOT / "aurora-main" / "aurora", d / "aurora-main" / "aurora", ignore=ignore)
        for f in ("pyproject.toml", "version.txt", "README.md"):
            shutil.copy(ROOT / "aurora-main" / f, d / "aurora-main" / f)
        shutil.copy(ROOT / "pyproject.toml", d / "pyproject.toml")
        (d / "data" / "raw").mkdir(parents=True)
        shutil.copy(chat_jsonl or ROOT / "data" / "raw" / "chat.jsonl", d / "data" / "raw" / "chat.jsonl")
        (d / "data" / "streams").mkdir()
        for f in (ROOT / "data" / "streams").glob("*.jsonl"):
            shutil.copy(f, d / "data" / "streams" / f.name)
        shutil.copytree(ROOT / "outputs" / "vocab", d / "outputs" / "vocab")
        (d / "dataset-metadata.json").write_text(json.dumps({"title": DATASET + tag, "id": slug, "licenses": [{"name": "CC0-1.0"}]}))
        exists = slug in kaggle("datasets", "list", "--mine", check=False)
        out = kaggle("datasets", "version" if exists else "create", "-p", str(d), *(["-m", "update"] if exists else []),
                     "--dir-mode", "zip", check=False)
        print(out.strip().splitlines()[-1])
    time.sleep(20)  # the new version takes a moment to become attachable
    return slug


def push_kernel(user: str, src_slug: str, epochs: int, dtype: str, grad_ckpt: bool, limit: int | None, serve: bool, tag: str = "",
                batch: int = 2, accum: int = 4) -> str:
    kid = f"{user}/{KERNEL}{tag}"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        script = (HERE / "run_bp04.py").read_text()
        env = {"BP04_EPOCHS": str(epochs), "BP04_DTYPE": dtype, "BP04_GRAD_CKPT": "1" if grad_ckpt else "0", "BP04_LIMIT": str(limit or ""), "BP04_SERVE": "1" if serve else "0",
               "BP04_BATCH": str(batch), "BP04_ACCUM": str(accum)}
        header = "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
        (d / "run_bp04.py").write_text(header + script)
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": KERNEL + tag, "code_file": "run_bp04.py", "language": "python", "kernel_type": "script",
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


def pull(kid: str, label: str) -> None:
    out = ROOT / "outputs" / "baseline" / "kaggle" / label
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
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32", help="draft weights (default fp32)")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--limit", type=int, default=None, help="conversations to dump (smoke run)")
    ap.add_argument("--pull-only", action="store_true")
    ap.add_argument("--label", default=None, help="output folder under outputs/baseline/kaggle/")
    ap.add_argument("--no-serve", action="store_true", help="skip BP-05's held-out τ̄ check after training")
    ap.add_argument("--chat-jsonl", default=None, help="train on this chat pool instead of data/raw/chat.jsonl (e.g. data_10k/chat_9k.jsonl)")
    ap.add_argument("--tag", default="", help="suffix for the dataset and kernel names, e.g. -9k (keeps the default runs untouched)")
    ap.add_argument("--batch-size", type=int, default=2, help="micro-batch (effective batch = batch-size x accumulation)")
    ap.add_argument("--accumulation", type=int, default=4)
    a = ap.parse_args()
    kid = f"{a.user}/{KERNEL}{a.tag}"
    if not a.pull_only:
        slug = push_source(a.user, Path(a.chat_jsonl) if a.chat_jsonl else None, a.tag)
        kid = push_kernel(a.user, slug, a.epochs, a.dtype, not a.no_grad_ckpt, a.limit, not a.no_serve, a.tag, a.batch_size, a.accumulation)
        print("final:", wait(kid))
    pull(kid, a.label or f"{a.dtype}{'' if a.no_grad_ckpt else '_ckpt'}_{a.epochs}ep{'_limit' + str(a.limit) if a.limit else ''}")


if __name__ == "__main__":
    main()
