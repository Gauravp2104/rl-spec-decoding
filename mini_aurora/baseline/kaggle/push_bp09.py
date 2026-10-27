"""Run BP-09 on Kaggle's T4: the test suite on CUDA, or the 1.7B scale check.

    python -m mini_aurora.baseline.kaggle.push_bp09 --user <u> tests
    python -m mini_aurora.baseline.kaggle.push_bp09 --user <u> 1p7b --limit 20 --epochs 1     # smoke
    python -m mini_aurora.baseline.kaggle.push_bp09 --user <u> 1p7b --epochs 8                # full run
    python -m mini_aurora.baseline.kaggle.push_bp09 --user <u> tests --pull-only

Reuses push.py's dataset push / wait / log parsing; results land in outputs/bp09_kaggle/<mode>/.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from mini_aurora.baseline.kaggle.push import ROOT, kaggle, push_source, wait

HERE = Path(__file__).resolve().parent


def push_kernel(user: str, src_slug: str, mode: str, limit: int | None, epochs: int) -> str:
    kid = f"{user}/rl-spec-decoding-bp09-{mode}"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        env = {"BP09_MODE": mode, "BP09_LIMIT": str(limit) if limit else "", "BP09_EPOCHS": str(epochs)}
        header = "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
        (d / "run_bp09.py").write_text(header + (HERE / "run_bp09.py").read_text())
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": kid.split("/")[1], "code_file": "run_bp09.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [src_slug], "competition_sources": [], "kernel_sources": []}))
        print(kaggle("kernels", "push", "-p", str(d)).strip().splitlines()[-1])
    return kid


def pull(kid: str, mode: str) -> None:
    out = ROOT / "outputs" / "bp09_kaggle" / mode
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(kaggle("kernels", "output", kid, "-p", tmp).strip().splitlines()[-1])
        src = Path(tmp) / "bp09"
        if src.exists():
            shutil.copytree(src, out, dirs_exist_ok=True)
        log = next(Path(tmp).glob("*.log"), None)
        if log:
            lines = [json.loads(l.strip().lstrip(",")).get("data", "").rstrip() for l in log.read_text().splitlines()
                     if l.strip().lstrip(",").startswith("{")]
            (out / "kaggle_run.log").write_text("\n".join(x for x in lines if x))
    print("pulled ->", out, sorted(p.name for p in out.iterdir()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tests", "1p7b"])
    ap.add_argument("--user", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--pull-only", action="store_true")
    a = ap.parse_args()
    kid = f"{a.user}/rl-spec-decoding-bp09-{a.mode}"
    if not a.pull_only:
        kid = push_kernel(a.user, push_source(a.user), a.mode, a.limit, a.epochs)
        print("final:", wait(kid, poll=120))
    pull(kid, a.mode)


if __name__ == "__main__":
    main()
