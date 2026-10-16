#!/usr/bin/env bash
# One-time environment setup for BP-01. Run from the repo root: bash mini_aurora/setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || uv venv --python 3.12 .venv   # reuse an existing venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r mini_aurora/requirements-mac.txt
uv pip install -e aurora-main --no-deps   # Aurora model/loss/data code without Mooncake / Ray / SGLang
uv pip install -e . --no-deps             # mini_aurora package

echo "Done. Activate with: source .venv/bin/activate && source mini_aurora/env.sh"
