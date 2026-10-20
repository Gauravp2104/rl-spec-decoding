#!/usr/bin/env bash
# Editable installs against the mounted repo (fast, no deps), then run the given command.
set -euo pipefail
pip install -q -e aurora-main --no-deps -e . --no-deps
exec "$@"
