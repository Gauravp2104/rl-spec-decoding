#!/usr/bin/env bash
# Run a command inside the CPU container with the repo mounted at /work and the HF cache shared.
#   bash mini_aurora/docker/run_cpu.sh python -m mini_aurora.data.vocab --compare
#   bash mini_aurora/docker/run_cpu.sh python -m pytest -q
set -euo pipefail
cd "$(dirname "$0")/../.."
docker run --rm -it \
  -v "$PWD":/work \
  -v "${HF_HOME:-$HOME/.cache/huggingface}":/hf \
  mini-aurora-cpu "$@"
