#!/usr/bin/env bash
# One-time environment setup for BP-01. Run from the repo root: bash mini_aurora/setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# The venv lives in .venv.nosync (iCloud "Desktop & Documents" sync skips *.nosync and would otherwise
# mark the venv's .pth files hidden, which Python 3.12+ ignores). .venv is a symlink to it.
if [ -d .venv ] && [ ! -L .venv ]; then
  echo "Replacing plain .venv with .venv.nosync"; rm -rf .venv
fi
[ -d .venv.nosync ] || uv venv --python 3.12 .venv.nosync
ln -sfn .venv.nosync .venv
# shellcheck disable=SC1091
source .venv.nosync/bin/activate
uv pip install -r mini_aurora/requirements-mac.txt
uv pip install -e aurora-main --no-deps   # Aurora model/loss/data code without Mooncake / Ray / SGLang
uv pip install -e . --no-deps             # mini_aurora package

# Python 3.12+ skips .pth files with the macOS "hidden" flag, which breaks the editable installs above.
chflags nohidden .venv.nosync/lib/python3.12/site-packages/*.pth

echo "Done. Activate with: source .venv/bin/activate && source mini_aurora/env.sh"
