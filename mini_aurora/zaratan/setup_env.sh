#!/usr/bin/env bash
# One-time setup on a Zaratan LOGIN node (has internet): venv, data and models in scratch (home is 10 GB).
#   bash mini_aurora/zaratan/setup_env.sh
# Afterwards: sbatch mini_aurora/zaratan/bp09_1p7b.sbatch
set -euo pipefail
cd "$(dirname "$0")/../.."
SCRATCH="${SCRATCH:-$HOME/scratch.msai635/rl-spec-decoding}"
VENV="${VENV:-$SCRATCH/venv}"

source /etc/profile
module load python/3.10.10
unset PYTHONPATH                      # the module leaks its own site-packages
[ -d "$VENV" ] || python -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124
grep -v '^torch\|^#' mini_aurora/requirements-mac.txt | sed 's/#.*//' | xargs pip install -q
pip install -q -e aurora-main --no-deps -e . --no-deps

mkdir -p "$SCRATCH/hf" "$SCRATCH/outputs"
export HF_HOME="$SCRATCH/hf" WANDB_MODE=offline
python -m mini_aurora.data.download && python -m mini_aurora.data.streams          # data/raw, data/streams
MINI_AURORA_DEVICE=cpu python -m mini_aurora.data.vocab                               # outputs/vocab/qwen3_24k_mixed.pt
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"):
    print(m, snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
PY
echo "setup done: venv $VENV, HF cache $SCRATCH/hf"
