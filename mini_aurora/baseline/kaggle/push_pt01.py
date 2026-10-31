"""Run PT-01's A1 arm on Kaggle's T4 and pull the results back.

    python -m mini_aurora.baseline.kaggle.push_pt01 --user <u> --limit 20 --epochs 1     # smoke
    python -m mini_aurora.baseline.kaggle.push_pt01 --user <u>                           # full run: 1000 rows, 8 epochs
    python -m mini_aurora.baseline.kaggle.push_pt01 --user <u> --seed 1                  # second seed (PLAN §8)
    python -m mini_aurora.baseline.kaggle.push_pt01 --user <u> --pull-only

Packages the source as the private dataset ``<u>/rl-spec-decoding-src-pt01`` (push.py's ``push_source``, kept apart
from BP-04/BP-09's dataset), pushes ``run_pt01.py`` as a GPU kernel and waits for it. Results land in
``outputs/pt01_kaggle/<label>/`` (default label ``a1_<epochs>ep_seed<seed>`` or ``smoke``).
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from mini_aurora.baseline.kaggle.push import ROOT, kaggle, push_source, wait

HERE = Path(__file__).resolve().parent
TAG = "-pt01"


def kernel_id(user: str, smoke: bool, seed: int) -> str:
    return f"{user}/rl-spec-decoding-pt01-a1" + ("-smoke" if smoke else "") + (f"-seed{seed}" if seed else "")


def push_kernel(kid: str, src_slug: str, limit: int | None, epochs: int, seed: int, gen_batch: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        env = {"PT01_LIMIT": str(limit) if limit else "", "PT01_EPOCHS": str(epochs), "PT01_SEED": str(seed),
               "PT01_GEN_BATCH": str(gen_batch)}
        header = "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
        (d / "run_pt01.py").write_text(header + (HERE / "run_pt01.py").read_text())
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": kid.split("/")[1], "code_file": "run_pt01.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [src_slug], "competition_sources": [], "kernel_sources": []}))
        print(kaggle("kernels", "push", "-p", str(d)).strip().splitlines()[-1])


def pull(kid: str, label: str) -> Path:
    out = ROOT / "outputs" / "pt01_kaggle" / label
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(kaggle("kernels", "output", kid, "-p", tmp).strip().splitlines()[-1])
        src = Path(tmp) / "pt01"
        if src.exists():
            shutil.copytree(src, out, dirs_exist_ok=True)
        log = next(Path(tmp).glob("*.log"), None)
        if log:
            lines = [json.loads(l.strip().lstrip(",")).get("data", "").rstrip() for l in log.read_text().splitlines()
                     if l.strip().lstrip(",").startswith("{")]
            (out / "kaggle_run.log").write_text("\n".join(x for x in lines if x))
    print("pulled ->", out, sorted(p.name for p in out.iterdir()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N conversations (smoke); default all")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-batch", type=int, default=16, help="regeneration batch size")
    ap.add_argument("--label", default=None)
    ap.add_argument("--pull-only", action="store_true")
    a = ap.parse_args()
    kid = kernel_id(a.user, a.limit is not None, a.seed)
    if not a.pull_only:
        push_kernel(kid, push_source(a.user, tag=TAG), a.limit, a.epochs, a.seed, a.gen_batch)
        print("final:", wait(kid, poll=120))
    pull(kid, a.label or ("smoke" if a.limit else f"a1_{a.epochs}ep_seed{a.seed}"))


if __name__ == "__main__":
    main()
