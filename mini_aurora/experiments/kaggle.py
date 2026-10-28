"""Run one BP-08 arm on Kaggle's T4 (fp32, 16 GB) and pull the run directory back.

    python -m mini_aurora.experiments.kaggle --user <kaggle user> E2 online_static
    python -m mini_aurora.experiments.kaggle --user <kaggle user> E4 online_day0 --grid N5
    python -m mini_aurora.experiments.kaggle --user <kaggle user> E2 online_static --pull-only

Reuses the BP-04 Kaggle runner's dataset push, wait and output download (``mini_aurora/baseline/kaggle/push.py``);
the kernel is ``<user>/rl-spec-decoding-bp08-<exp>-<arm>[-<grid>]`` running ``kaggle_arm.py``. Results land in
``outputs/experiments_kaggle/<EXP>/...`` and feed ``mini_aurora.experiments.table --out-dir outputs/experiments_kaggle``.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from mini_aurora.baseline.kaggle.push import ROOT, kaggle, push_source, wait

HERE = Path(__file__).resolve().parent


def push_kernel(user: str, src_slug: str, exp: str, arm: str, grid: str | None, n: int | None, loop_args: str) -> str:
    kid = f"{user}/rl-spec-decoding-bp08-{exp}-{arm}{'-' + grid if grid else ''}".lower().replace("_", "-")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        env = {"BP08_EXP": exp, "BP08_ARM": arm, "BP08_GRID": grid or "", "BP08_N": str(n or ""), "BP08_LOOP_ARGS": loop_args}
        header = "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
        (d / "kaggle_arm.py").write_text(header + (HERE / "kaggle_arm.py").read_text())
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": kid.split("/")[1], "code_file": "kaggle_arm.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [src_slug], "competition_sources": [], "kernel_sources": []}))
        print(kaggle("kernels", "push", "-p", str(d)).strip().splitlines()[-1])
    return kid


def pull(kid: str) -> None:
    out = ROOT / "outputs" / "experiments_kaggle"
    with tempfile.TemporaryDirectory() as tmp:
        print(kaggle("kernels", "output", kid, "-p", tmp).strip().splitlines()[-1])
        src = Path(tmp) / "bp08"
        if src.exists():
            shutil.copytree(src, out, dirs_exist_ok=True)
            print("pulled:", *[str(p.relative_to(src)) for p in src.glob("E*/*/summary.json")])
        log = next(Path(tmp).glob("*.log"), None)
        if log:
            lines = [json.loads(l.strip().lstrip(",")).get("data", "").rstrip() for l in log.read_text().splitlines()
                     if l.strip().lstrip(",").startswith("{")]
            print("\n".join(x for x in lines if x)[-2500:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("arm")
    ap.add_argument("--user", required=True)
    ap.add_argument("--grid", default=None)
    ap.add_argument("-n", type=int, default=None)
    ap.add_argument("--loop-args", default="")
    ap.add_argument("--pull-only", action="store_true")
    a = ap.parse_args()
    kid = f"{a.user}/rl-spec-decoding-bp08-{a.experiment}-{a.arm}{'-' + a.grid if a.grid else ''}".lower().replace("_", "-")
    if not a.pull_only:
        slug = push_source(a.user)
        kid = push_kernel(a.user, slug, a.experiment, a.arm, a.grid, a.n, a.loop_args)
        print("final:", wait(kid, poll=120))
    pull(kid)


if __name__ == "__main__":
    main()
