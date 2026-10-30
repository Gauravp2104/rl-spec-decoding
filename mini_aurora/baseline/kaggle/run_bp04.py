"""BP-04 on Kaggle (T4): feature dump -> 2-epoch static draft training -> held-out evaluation.

Runs as a Kaggle script kernel with the private dataset ``rl-spec-decoding-src`` attached (pushed by
``push.py``: the repo source plus ``data/raw/chat.jsonl`` and ``outputs/vocab/``). Internet must be on for
the Qwen3-0.6B download. Everything worth keeping lands in ``/kaggle/working/bp04/``:

    manifest.json        dump sizes and timings
    train_log.csv        per-step loss / acc / lr / grad norm / step time
    eval.json            per-epoch held-out loss and accuracy
    draft_step*.pt/json  final checkpoint (trainable weights + vocab buffers, no embedding)
    evaluate.json        standalone evaluation of the final checkpoint
    serve_heldout.txt    BP-05 server on the held-out chat requests with the trained draft (τ̄, --check), BP04_SERVE=1
    serve_heldout.jsonl  one JSON line per request from that run
    timings.json         wall-clock per stage

Knobs via environment variables (set by push.py): BP04_EPOCHS (8), BP04_DTYPE (fp32/bf16), BP04_GRAD_CKPT (1/0), BP04_LIMIT (all),
BP04_HELDOUT (100), BP04_BATCH (2), BP04_ACCUM (4), BP04_SERVE (1), BP04_K (3),
BP04_SERVE_ONLY (0: with 1, skip dump/train and serve the checkpoint of the attached kernel output at BP04_K),
BP04_TREE_TOPK (1) / BP04_TREE_BEAM (1): tree drafting in the serve step.
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


def run(cmd: list[str], tee: Path | None = None, **env) -> float:
    print("+", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    if tee is None:
        subprocess.run(cmd, check=True, cwd=REPO, env={**os.environ, **env})
    else:
        r = subprocess.run(cmd, cwd=REPO, env={**os.environ, **env}, text=True, capture_output=True)
        print(r.stdout, r.stderr[-3000:], flush=True)
        tee.write_text(r.stdout)
        r.check_returncode()
    return round(time.perf_counter() - t0, 1)


def serve(k: str, heldout: str, common_env: dict) -> float:
    """BP-05's loop on the held-out chat requests with the trained draft; K guesses per round."""
    cmd = [sys.executable, "-m", "mini_aurora.server.serve", "-n", heldout, "--heldout", "--check", "-k", k,
           "--draft", "outputs/baseline/checkpoints", "--log", "outputs/server/heldout.jsonl"]
    if os.environ.get("BP04_TREE_TOPK", "1") != "1":
        cmd += ["--tree-topk", os.environ["BP04_TREE_TOPK"], "--tree-beam", os.environ.get("BP04_TREE_BEAM", "1")]
    seconds = run(cmd, tee=OUT / "serve_heldout.txt", **common_env)
    shutil.copy(REPO / "outputs" / "server" / "heldout.jsonl", OUT / "serve_heldout.jsonl")
    return seconds


def serve_only(k: str, heldout: str, timings: dict, common_env: dict) -> None:
    """No dump / train: copy the checkpoint from the attached kernel output and serve it at K = ``k``."""
    latest = next(Path("/kaggle/input").rglob("bp04/latest.txt"), None)
    if latest is None:
        raise SystemExit("no finished BP-04 kernel attached as a kernel source")
    ckpt = REPO / "outputs" / "baseline" / "checkpoints"
    ckpt.mkdir(parents=True)
    step = int(latest.read_text().strip())
    for name in ("latest.txt", f"draft_step{step:06d}.pt", f"draft_step{step:06d}.json"):
        shutil.copy(latest.parent / name, ckpt / name)
    timings["serve_s"] = serve(k, heldout, common_env)
    timings.update(k=int(k), tree_topk=int(os.environ.get("BP04_TREE_TOPK", "1")), tree_beam=int(os.environ.get("BP04_TREE_BEAM", "1")),
                   checkpoint=str(latest.parent), step=step)
    (OUT / "timings.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings, indent=2))


def main() -> None:
    src = find_source()
    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(src, REPO, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["df", "-h", "/tmp", str(WORK)])  # the feature dump lives under /tmp/repo (~8 KB per token)
    timings = {}

    timings["install_s"] = run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "omegaconf",
                                "pydantic", "numba", "psutil", "wandb", "accelerate", "datasets", "numpy<2.4"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "aurora-main", "--no-deps", "-e", ".", "--no-deps"])
    subprocess.run([sys.executable, "-c", "import torch;print('torch', torch.__version__, torch.cuda.get_device_name(0))"], check=True)

    epochs = os.environ.get("BP04_EPOCHS", "8")
    limit = os.environ.get("BP04_LIMIT", "")
    heldout = os.environ.get("BP04_HELDOUT", "100")
    k = os.environ.get("BP04_K", "3")
    common_env = {"MINI_AURORA_DEVICE": "cuda", "WANDB_MODE": "offline", "HF_HOME": "/tmp/hf"}

    if os.environ.get("BP04_SERVE_ONLY", "0") == "1":  # re-serve a finished kernel's checkpoint (attached as a kernel source)
        serve_only(k, heldout, timings, common_env)
        return

    dump = [sys.executable, "-m", "mini_aurora.baseline.dump", "--jsonl", "data/raw/chat.jsonl", "--heldout", heldout]
    if limit:
        dump += ["--limit", limit]
    timings["dump_s"] = run(dump, **common_env)

    train = [sys.executable, "-m", "mini_aurora.baseline.train", "--epochs", epochs,
             "--batch-size", os.environ.get("BP04_BATCH", "2"), "--accumulation", os.environ.get("BP04_ACCUM", "4")]
    train += ["--dtype", os.environ.get("BP04_DTYPE", "fp32")]
    if os.environ.get("BP04_GRAD_CKPT", "1") == "0":
        train.append("--no-grad-ckpt")
    timings["train_s"] = run(train, **common_env)

    timings["evaluate_s"] = run([sys.executable, "-m", "mini_aurora.baseline.evaluate", "--out", "outputs/baseline/evaluate.json"], **common_env)

    if os.environ.get("BP04_SERVE", "1") == "1" and not limit:
        timings["serve_s"] = serve(k, heldout, common_env)

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
    timings["dtype"], timings["grad_ckpt"] = os.environ.get("BP04_DTYPE", "fp32"), os.environ.get("BP04_GRAD_CKPT", "1") == "1"
    (OUT / "timings.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings, indent=2))
    print(json.dumps(json.loads((OUT / "eval.json").read_text())["epochs"], indent=2))
    shutil.rmtree(REPO / "outputs" / "baseline" / "hidden_states", ignore_errors=True)  # ~4 GB, not kept


if __name__ == "__main__":
    main()
