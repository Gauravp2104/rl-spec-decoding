"""BP-04 on Kaggle (T4): feature dump -> 2-epoch static draft training -> held-out evaluation.

Runs as a Kaggle script kernel with the private dataset ``rl-spec-decoding-src`` attached (pushed by
``push.py``: the repo source plus ``data/raw/chat.jsonl`` and ``outputs/vocab/``). Internet must be on for
the Qwen3-0.6B download. Everything worth keeping lands in ``/kaggle/working/bp04/``:

    manifest.json        dump sizes and timings
    train_log.csv        per-step loss / acc / lr / grad norm / step time
    eval.json            per-epoch held-out loss and accuracy
    draft_step*.pt/json  final checkpoint (trainable weights + vocab buffers, no embedding)
    evaluate.json        standalone evaluation of the final checkpoint
    timings.json         wall-clock per stage

Knobs via environment variables (set by push.py): BP04_EPOCHS (2), BP04_DTYPE (bf16/fp32), BP04_GRAD_CKPT (1/0), BP04_LIMIT (all),
BP04_HELDOUT (100), BP04_BATCH (2), BP04_ACCUM (4).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "bp04"
REPO = Path("/tmp/repo")  # outside /kaggle/working so the kernel output is only bp04/


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
    timings = {}

    timings["install_s"] = run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "omegaconf",
                                "pydantic", "numba", "psutil", "wandb", "accelerate", "datasets", "numpy<2.4"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "aurora-main", "--no-deps", "-e", ".", "--no-deps"])
    subprocess.run([sys.executable, "-c", "import torch;print('torch', torch.__version__, torch.cuda.get_device_name(0))"], check=True)

    epochs = os.environ.get("BP04_EPOCHS", "2")
    limit = os.environ.get("BP04_LIMIT", "")
    heldout = os.environ.get("BP04_HELDOUT", "100")
    common_env = {"MINI_AURORA_DEVICE": "cuda", "WANDB_MODE": "offline", "HF_HOME": "/tmp/hf"}

    dump = [sys.executable, "-m", "mini_aurora.baseline.dump", "--jsonl", "data/raw/chat.jsonl", "--heldout", heldout]
    if limit:
        dump += ["--limit", limit]
    timings["dump_s"] = run(dump, **common_env)

    train = [sys.executable, "-m", "mini_aurora.baseline.train", "--epochs", epochs,
             "--batch-size", os.environ.get("BP04_BATCH", "2"), "--accumulation", os.environ.get("BP04_ACCUM", "4")]
    train += ["--dtype", os.environ.get("BP04_DTYPE", "bf16")]
    if os.environ.get("BP04_GRAD_CKPT", "1") == "0":
        train.append("--no-grad-ckpt")
    timings["train_s"] = run(train, **common_env)

    timings["evaluate_s"] = run([sys.executable, "-m", "mini_aurora.baseline.evaluate", "--out", "outputs/baseline/evaluate.json"], **common_env)

    base = REPO / "outputs" / "baseline"
    shutil.copy(base / "hidden_states" / "manifest.json", OUT / "manifest.json")
    shutil.copy(base / "evaluate.json", OUT / "evaluate.json")
    ckpt = base / "checkpoints"
    for name in ("train_log.csv", "eval.json", "latest.txt"):
        shutil.copy(ckpt / name, OUT / name)
    step = int((ckpt / "latest.txt").read_text().strip())
    for ext in ("pt", "json"):
        shutil.copy(ckpt / f"draft_step{step:06d}.{ext}", OUT / f"draft_step{step:06d}.{ext}")
    timings["epochs"], timings["limit"] = int(epochs), limit or None
    timings["dtype"], timings["grad_ckpt"] = os.environ.get("BP04_DTYPE", "bf16"), os.environ.get("BP04_GRAD_CKPT", "1") == "1"
    (OUT / "timings.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings, indent=2))
    print(json.dumps(json.loads((OUT / "eval.json").read_text())["epochs"], indent=2))
    shutil.rmtree(REPO / "outputs" / "baseline" / "hidden_states", ignore_errors=True)  # ~4 GB, not kept


if __name__ == "__main__":
    main()
