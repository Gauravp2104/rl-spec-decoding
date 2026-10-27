"""BP-09 on Kaggle (T4): the test suite on CUDA, or the 1.7B scale check (BP-04 pipeline + BP-05 held-out τ̄).

Kernel script; BP09_MODE = "tests" | "1p7b", BP09_LIMIT (conversations, "" = all), BP09_EPOCHS, set by push_bp09.py.
The source dataset ``rl-spec-decoding-src`` must be attached and internet on. Outputs: /kaggle/working/bp09/.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "bp09"
REPO = Path("/tmp/repo")
RELEASE = "https://github.com/Gauravp2104/rl-spec-decoding/releases/download/bp04-static-v1"


def find_source() -> Path:
    for p in Path("/kaggle/input").rglob("pyproject.toml"):
        if (p.parent / "mini_aurora").is_dir():
            return p.parent
    raise SystemExit("rl-spec-decoding-src dataset not attached")


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
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    subprocess.run([sys.executable, "-c", "import torch;print('torch', torch.__version__, torch.cuda.get_device_name(0))"], check=True)
    mode = os.environ.get("BP09_MODE", "tests")
    timings = {"mode": mode}

    if mode == "tests":
        ckpt = REPO / "outputs" / "baseline" / "checkpoints_8ep"
        ckpt.mkdir(parents=True, exist_ok=True)
        for name in ("draft_step000896.pt", "latest.txt"):
            run(["curl", "-sSL", "-o", str(ckpt / name), f"{RELEASE}/{name}"])
        run([sys.executable, "-m", "mini_aurora.models.query_target"], check=False, **env)  # warms the model cache
        log = OUT / "pytest.txt"
        with open(log, "w") as f:
            t0 = time.perf_counter()
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "mini_aurora/tests"], cwd=REPO,
                               env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT)
            timings["pytest_s"], timings["pytest_exit"] = round(time.perf_counter() - t0, 1), r.returncode
        print(log.read_text()[-4000:])
    else:
        limit, epochs = os.environ.get("BP09_LIMIT", "20"), os.environ.get("BP09_EPOCHS", "1")
        env.update({"MINI_AURORA_TARGET": "Qwen/Qwen3-1.7B",
                    "MINI_AURORA_DRAFT_CONFIG": str(REPO / "examples" / "zaratan-qwen3-1.7b" / "draft_config.json")})
        dump = Path("/tmp/bp09_dump")
        _, timings["dump_s"] = run([sys.executable, "-m", "mini_aurora.baseline.dump", "--out", str(dump),
                                    *(["--limit", limit] if limit else [])], **env)
        _, timings["train_s"] = run([sys.executable, "-m", "mini_aurora.baseline.train", "--dump-dir", str(dump),
                                     "--out", str(OUT / "checkpoints"), "--epochs", epochs], **env)
        step = (OUT / "checkpoints" / "latest.txt").read_text().strip()
        _, timings["evaluate_s"] = run([sys.executable, "-m", "mini_aurora.baseline.evaluate", "--dump-dir", str(dump),
                                        "--ckpt", str(OUT / "checkpoints" / f"draft_step{int(step):06d}.pt"),
                                        "--out", str(OUT / "evaluate.json")], **env)
        if not limit:
            _, timings["serve_s"] = run([sys.executable, "-m", "mini_aurora.server.serve", "-n", "100", "--heldout", "--check",
                                         "--draft", str(OUT / "checkpoints"), "--log", str(OUT / "serve_heldout.jsonl")], **env)
        shutil.copy(dump / "manifest.json", OUT / "manifest.json")
        shutil.rmtree(dump, ignore_errors=True)
        for f in (OUT / "checkpoints").glob("draft_step*.pt"):
            if f.name != f"draft_step{int(step):06d}.pt":
                f.unlink()
        timings.update({"limit": limit or None, "epochs": int(epochs), "target": "Qwen/Qwen3-1.7B"})
    (OUT / "timings.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
