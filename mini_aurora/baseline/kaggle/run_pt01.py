"""PT-01 on Kaggle (T4): arm A1, a Qwen3-1.7B draft trained on the target's own answers.

Kernel script; settings come from environment variables written by push_pt01.py:
PT01_LIMIT (conversations, "" = all 1000), PT01_EPOCHS (8 = BP-09), PT01_SEED (train seed), PT01_GEN_BATCH.

Pipeline (BP-09's, with a regeneration step in front):
    regenerate (data/raw/chat.jsonl -> chat_self_1p7b.jsonl) -> dump -> train -> evaluate -> serve --heldout --check
The source dataset ``rl-spec-decoding-src-pt01`` must be attached and internet on. Outputs: /kaggle/working/pt01/.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "pt01"
REPO = Path("/tmp/repo")
TARGET = "Qwen/Qwen3-1.7B"


def find_source() -> Path:
    for p in Path("/kaggle/input").rglob("pyproject.toml"):
        if (p.parent / "mini_aurora").is_dir():
            return p.parent
    raise SystemExit("rl-spec-decoding-src-pt01 dataset not attached")


def run(cmd, check=True, **env) -> tuple[int, float]:
    print("+", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=REPO, env={**os.environ, **env})
    if check:
        r.check_returncode()
    return r.returncode, round(time.perf_counter() - t0, 1)


def main() -> None:
    src = find_source()
    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(src, REPO, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    OUT.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "omegaconf", "pydantic", "numba",
         "psutil", "wandb", "accelerate", "datasets", "numpy<2.4", "pytest", "matplotlib"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "aurora-main", "--no-deps", "-e", ".", "--no-deps"])
    env = {"MINI_AURORA_DEVICE": "cuda", "WANDB_MODE": "offline", "HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MINI_AURORA_TARGET": TARGET,
           "MINI_AURORA_DRAFT_CONFIG": str(REPO / "examples" / "zaratan-qwen3-1.7b" / "draft_config.json")}
    subprocess.run([sys.executable, "-c", "import torch;print('torch', torch.__version__, torch.cuda.get_device_name(0))"],
                   check=True)

    limit = os.environ.get("PT01_LIMIT", "20")
    epochs = os.environ.get("PT01_EPOCHS", "1")
    seed = os.environ.get("PT01_SEED", "0")
    gen_batch = os.environ.get("PT01_GEN_BATCH", "16")
    lim = ["--limit", limit] if limit else []
    data = REPO / "data" / "raw" / "chat_self_1p7b.jsonl"
    dump, ckpts = Path("/tmp/pt01_dump"), OUT / "checkpoints"
    timings = {"arm": "A1", "target": TARGET, "limit": limit or None, "epochs": int(epochs), "seed": int(seed)}

    # 1. regenerate every assistant turn with the target (greedy, fp32, thinking off)
    _, timings["regenerate_s"] = run([sys.executable, "-m", "mini_aurora.data.regenerate", "--model", TARGET,
                                      "--out", str(data), "--batch-size", gen_batch, *lim], **env)
    for f in (data, data.with_name(data.stem + ".stats.json")):
        shutil.copy(f, OUT / f.name)

    # 2-4. BP-09's pipeline on the regenerated pool; everything else unchanged
    _, timings["dump_s"] = run([sys.executable, "-m", "mini_aurora.baseline.dump", "--jsonl", str(data),
                                "--out", str(dump), *lim], **env)
    _, timings["train_s"] = run([sys.executable, "-m", "mini_aurora.baseline.train", "--dump-dir", str(dump),
                                 "--out", str(ckpts), "--epochs", epochs, "--seed", seed], **env)
    step = int((ckpts / "latest.txt").read_text().strip())
    final = ckpts / f"draft_step{step:06d}.pt"
    _, timings["evaluate_s"] = run([sys.executable, "-m", "mini_aurora.baseline.evaluate", "--dump-dir", str(dump),
                                    "--ckpt", str(final), "--out", str(OUT / "evaluate_self_heldout.json")], **env)

    # 5. τ̄ on the BP-04 held-out chat requests (same ids as A0), lossless check on
    if not limit:
        _, timings["serve_s"] = run([sys.executable, "-m", "mini_aurora.server.serve", "-n", "100", "--heldout", "--check",
                                     "--draft", str(ckpts), "--log", str(OUT / "serve_heldout.jsonl")], **env)

    shutil.copy(dump / "manifest.json", OUT / "manifest.json")
    shutil.rmtree(dump, ignore_errors=True)
    for f in ckpts.glob("draft_step*.pt"):
        if f != final:
            f.unlink()
    (OUT / "timings.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
