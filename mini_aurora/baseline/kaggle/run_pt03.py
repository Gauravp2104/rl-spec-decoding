"""PT-03 on Kaggle (T4): serve one finished draft with chain and tree configs on the held-out chat requests.

Kernel script; set by push_pt03.py:
    PT03_DRAFT    a0 (BP-09 1.7B draft, attached bp09 kernel output) | a1 (PT-01 release) | 15k6ep (BP-04 15K/6ep, attached bp04 kernel output)
    PT03_CONFIGS  comma-separated k:topk:beam, default 3:1:1,3:2:2,3:2:4,3:4:2,3:4:4
    PT03_N        held-out requests (100)
Outputs: /kaggle/working/pt03/<k{k}_topk{t}_beam{b}>/serve_heldout.{txt,jsonl} and results.json.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "pt03"
REPO = Path("/tmp/repo")
A1_RELEASE = "https://github.com/Gauravp2104/rl-spec-decoding/releases/download/pt01-a1-v1"
TARGET_1P7B = {"MINI_AURORA_TARGET": "Qwen/Qwen3-1.7B",
               "MINI_AURORA_DRAFT_CONFIG": str(REPO / "examples" / "zaratan-qwen3-1.7b" / "draft_config.json")}


def find_source() -> Path:
    for p in Path("/kaggle/input").rglob("pyproject.toml"):
        if (p.parent / "mini_aurora").is_dir():
            return p.parent
    raise SystemExit("rl-spec-decoding-src dataset not attached")


def run(cmd, check=True, tee: Path | None = None, **env) -> tuple[int, float, str]:
    print("+", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=REPO, env={**os.environ, **env}, text=True, capture_output=tee is not None)
    if tee is not None:
        print(r.stdout[-6000:], r.stderr[-3000:], flush=True)
        tee.write_text(r.stdout + ("\n" + r.stderr if r.returncode else ""))
    if check:
        r.check_returncode()
    return r.returncode, round(time.perf_counter() - t0, 1), r.stdout or ""


def fetch_checkpoint(draft: str) -> Path:
    ckpt = REPO / "outputs" / "baseline" / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    if draft == "a1":
        for name in ("latest.txt", "draft_step000896.pt", "draft_step000896.json"):
            run(["curl", "-sSL", "-o", str(ckpt / name), f"{A1_RELEASE}/{name}"])
        return ckpt
    pattern = {"a0": "bp09/checkpoints/latest.txt", "15k6ep": "bp04/latest.txt"}[draft]
    latest = next(Path("/kaggle/input").rglob(pattern), None)
    if latest is None:
        raise SystemExit(f"no finished kernel with {pattern} attached as a kernel source")
    step = int(latest.read_text().strip())
    for name in ("latest.txt", f"draft_step{step:06d}.pt", f"draft_step{step:06d}.json"):
        shutil.copy(latest.parent / name, ckpt / name)
    return ckpt


def summary(stdout: str) -> dict:
    """The last JSON object serve.py prints."""
    end = stdout.rfind("\n}")
    start = stdout.rfind("\n{", 0, end)
    try:
        return json.loads(stdout[start:end + 2])
    except (ValueError, json.JSONDecodeError):
        return {}


def main() -> None:
    src = find_source()
    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(src, REPO, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    OUT.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "omegaconf", "pydantic", "numba",
         "psutil", "wandb", "accelerate", "datasets", "numpy<2.4"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "aurora-main", "--no-deps", "-e", ".", "--no-deps"])
    subprocess.run([sys.executable, "-c", "import torch;print('torch', torch.__version__, torch.cuda.get_device_name(0))"], check=True)

    draft = os.environ.get("PT03_DRAFT", "a1")
    n = os.environ.get("PT03_N", "100")
    configs = [tuple(int(x) for x in c.split(":")) for c in os.environ.get("PT03_CONFIGS", "3:1:1,3:2:2,3:2:4,3:4:2,3:4:4").split(",")]
    env = {"MINI_AURORA_DEVICE": "cuda", "WANDB_MODE": "offline", "HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if draft in ("a0", "a1"):
        env.update(TARGET_1P7B)
    ckpt = fetch_checkpoint(draft)
    step = int((ckpt / "latest.txt").read_text().strip())

    results = {"draft": draft, "step": step, "target": env.get("MINI_AURORA_TARGET", "Qwen/Qwen3-0.6B"), "n": int(n), "configs": []}
    for k, topk, beam in configs:
        label = f"k{k}_topk{topk}_beam{beam}"
        d = OUT / label
        d.mkdir(exist_ok=True)
        cmd = [sys.executable, "-m", "mini_aurora.server.serve", "-n", n, "--heldout", "--check", "-k", str(k),
               "--draft", str(ckpt), "--log", str(d / "serve_heldout.jsonl")]
        if topk > 1:
            cmd += ["--tree-topk", str(topk), "--tree-beam", str(beam)]
        code, seconds, stdout = run(cmd, check=False, tee=d / "serve_heldout.txt", **env)
        row = {"label": label, "k": k, "tree_topk": topk, "tree_beam": beam, "serve_s": seconds, "exit": code, **summary(stdout)}
        results["configs"].append(row)
        (OUT / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
