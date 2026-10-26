#!/usr/bin/env bash
# One-time setup on a Zaratan LOGIN node (has internet): venv, data and models in scratch (home is 10 GB).
#   bash mini_aurora/zaratan/setup_env.sh
# Afterwards: sbatch mini_aurora/zaratan/bp09_1p7b.sbatch
set -eo pipefail   # no -u: /etc/profile and the module system reference unset variables
cd "$(dirname "$0")/../.."
SCRATCH="${SCRATCH:-$HOME/scratch.msml612/rl-spec-decoding}"
VENV="${VENV:-$SCRATCH/venv}"

# Aurora needs Python >= 3.12 and Zaratan's newest python module is 3.10, so uv fetches a standalone CPython 3.12
# (everything under scratch: home is 10 GB).
mkdir -p "$SCRATCH/tmp"
export TMPDIR="$SCRATCH/tmp" UV_LINK_MODE=copy UV_NO_MODIFY_PATH=1   # keep temp files and wheels out of the 10 GB home
export UV_INSTALL_DIR="$SCRATCH/bin" UV_PYTHON_INSTALL_DIR="$SCRATCH/uv-python" UV_CACHE_DIR="$SCRATCH/uv-cache"
[ -x "$SCRATCH/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
UV="$SCRATCH/bin/uv"
[ -d "$VENV" ] || "$UV" venv --python 3.12 "$VENV"
source "$VENV/bin/activate"
"$UV" pip install -q torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124
grep -v '^torch\|^#' mini_aurora/requirements-mac.txt | sed 's/#.*//' | xargs "$UV" pip install -q
"$UV" pip install -q -e aurora-main --no-deps -e . --no-deps

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
