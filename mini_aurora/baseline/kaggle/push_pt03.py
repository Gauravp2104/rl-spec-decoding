"""Run PT-03 on Kaggle's T4: chain and tree serves of a finished draft on the held-out chat requests.

    python -m mini_aurora.baseline.kaggle.push_pt03 --user <u> a1                 # PT-01 A1 (release), 1.7B target
    python -m mini_aurora.baseline.kaggle.push_pt03 --user <u> a0                 # BP-09 A0 (attached bp09-1p7b kernel), 1.7B target
    python -m mini_aurora.baseline.kaggle.push_pt03 --user <u> 15k6ep             # BP-04 15K/6ep (attached bp04-15k-6ep kernel), 0.6B target
    python -m mini_aurora.baseline.kaggle.push_pt03 --user <u> a1 -n 3            # smoke
    python -m mini_aurora.baseline.kaggle.push_pt03 --user <u> a1 --pull-only

Reuses push.py's dataset push / wait; results land in outputs/pt03_kaggle/<draft>/.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from mini_aurora.baseline.kaggle.push import ROOT, kaggle, push_source, wait

HERE = Path(__file__).resolve().parent
SOURCES = {"a0": "rl-spec-decoding-bp09-1p7b", "15k6ep": "rl-spec-decoding-bp04-15k-6ep", "a1": None}
CONFIGS = "3:1:1,3:2:2,3:2:4,3:4:2,3:4:4"


def push_kernel(user: str, src_slug: str, draft: str, n: int, configs: str) -> str:
    kid = f"{user}/rl-spec-decoding-pt03-{draft}"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        env = {"PT03_DRAFT": draft, "PT03_N": str(n), "PT03_CONFIGS": configs}
        header = "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
        (d / "run_pt03.py").write_text(header + (HERE / "run_pt03.py").read_text())
        src_kernel = SOURCES[draft]
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": kid.split("/")[1], "code_file": "run_pt03.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [src_slug], "competition_sources": [],
            "kernel_sources": [f"{user}/{src_kernel}"] if src_kernel else []}))
        print(kaggle("kernels", "push", "-p", str(d)).strip().splitlines()[-1])
    return kid


def pull(kid: str, draft: str) -> None:
    out = ROOT / "outputs" / "pt03_kaggle" / draft
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(kaggle("kernels", "output", kid, "-p", tmp).strip().splitlines()[-1])
        src = Path(tmp) / "pt03"
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
    ap.add_argument("draft", choices=sorted(SOURCES))
    ap.add_argument("--user", required=True)
    ap.add_argument("-n", type=int, default=100, help="held-out requests")
    ap.add_argument("--configs", default=CONFIGS, help="comma-separated k:topk:beam")
    ap.add_argument("--pull-only", action="store_true")
    a = ap.parse_args()
    kid = f"{a.user}/rl-spec-decoding-pt03-{a.draft}"
    if not a.pull_only:
        kid = push_kernel(a.user, push_source(a.user, tag="-pt03"), a.draft, a.n, a.configs)
        print("final:", wait(kid, poll=120))
    pull(kid, a.draft)


if __name__ == "__main__":
    main()
