#!/bin/bash
# Send user requests to the external sglang server — Qwen3-Coder-Next (with draft)
#
# Sends dataset prompts to the sglang server, which triggers hidden state
# collection and training sample creation via the callback mechanism.
#
# Prerequisites:
#   - run.sh must be running (both training and sglang server must be up)
#
# Usage:
#   bash examples/qwen3-next-coder-external-with-draft/send_requests.sh [EXTRA_ARGS...]
#
# Environment variables:
#   SGLANG_URL     - sglang server URL (default: http://localhost:30000)
#   NUM_SAMPLES    - Number of samples to send, 0=all (default: 0)
#   NUM_WORKERS    - Concurrent request workers (default: 12)
#   MAX_TOKENS     - Max generation tokens per request (default: 512)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
DATASET="${DATASET:-$ROOT_DIR/datasets/onlinesd/merged/merged_code_train_shuffled.jsonl}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-Next}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
NUM_WORKERS="${NUM_WORKERS:-12}"
MAX_TOKENS="${MAX_TOKENS:-512}"

# Kill any leftover send_user_requests processes
pkill -f "send_user_requests.py" 2>/dev/null && echo "Killed previous send_user_requests processes" || true
sleep 1

python3 "$ROOT_DIR/examples/send_user_requests.py" \
    --dataset "$DATASET" \
    --server-url "$SGLANG_URL" \
    --model "$MODEL" \
    --chat-template "$CHAT_TEMPLATE" \
    --num-samples "$NUM_SAMPLES" \
    --num-workers "$NUM_WORKERS" \
    --max-tokens "$MAX_TOKENS" \
    --temperature 0  --no-shuffle \
    "$@"
