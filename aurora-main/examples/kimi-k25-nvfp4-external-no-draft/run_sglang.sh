#!/bin/bash
# Kimi-K2.5 NVFP4 external training — Node 2: SGLang Server
#
# Run this on the INFERENCE node (8 GPUs for sglang server with TP=8).
# Uses NVFP4 quantization for efficient inference.
# The sglang server connects to the mooncake master and callback server
# running on the trainer node (Node 1).
#
# Prerequisites:
#   - run_trainer.sh must be running on Node 1 and the callback server must be ready
#   - TRAINER_IP must be set to Node 1's IP address
#
# GPU allocation:
#   - All 8 GPUs on this node: sglang server (TP=8, NVFP4 quantized target model)
#
# Usage:
#   TRAINER_IP=<node1_ip> bash examples/kimi-k25-nvfp4-external-no-draft/run_sglang.sh
#
# Environment variables:
#   TRAINER_IP            - IP of the trainer node (REQUIRED)
#   CUDA_VISIBLE_DEVICES  - GPUs for sglang (default: 0,1,2,3,4,5,6,7)
#   SGLANG_PORT           - sglang server port (default: 30000)
#   CALLBACK_PORT         - Trainer callback port on Node 1 (default: 18080)
#   TARGET_MODEL          - Target model path (default: nvidia/Kimi-K2.5-NVFP4)

set -euo pipefail
set -x

if [ -z "${TRAINER_IP:-}" ]; then
    echo "ERROR: TRAINER_IP is required. Set it to the IP of the trainer node (Node 1)."
    echo "Usage: TRAINER_IP=<node1_ip> bash $0"
    exit 1
fi

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export HF_HOME="${HF_HOME:-/scratch/shared/huggingface}"
export PYTHONPATH="$ROOT_DIR/_sglang/python:${PYTHONPATH:-}"

SGLANG_PORT="${SGLANG_PORT:-30000}"
CALLBACK_PORT="${CALLBACK_PORT:-18080}"
MOONCAKE_GRPC_PORT="${MOONCAKE_GRPC_PORT:-50052}"
MOONCAKE_META_PORT="${MOONCAKE_META_PORT:-8090}"
TARGET_MODEL="${TARGET_MODEL:-nvidia/Kimi-K2.5-NVFP4}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-${#GPU_ARRAY[@]}}"

CALLBACK_URL="http://${TRAINER_IP}:${CALLBACK_PORT}/push_sample"
LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SGLANG_LOG="$LOG_DIR/kimi_k25_nvfp4_sglang_${TIMESTAMP}.log"
SGLANG_REQ_LOG_DIR="$LOG_DIR/requests_${TIMESTAMP}"
mkdir -p "$SGLANG_REQ_LOG_DIR"

echo "=============================================="
echo "Kimi-K2.5 NVFP4 External Training — SGLang Node"
echo "=============================================="
echo "Target model: $TARGET_MODEL"
echo "SGLang GPUs: $CUDA_VISIBLE_DEVICES (TP=$SGLANG_TP_SIZE)"
echo "Trainer IP: $TRAINER_IP"
echo "Callback URL: $CALLBACK_URL"
echo "Local IP: $LOCAL_IP"
echo "Log: $SGLANG_LOG"
echo "=============================================="

# --- Verify trainer callback server is reachable ---
echo "Checking trainer callback server at http://${TRAINER_IP}:${CALLBACK_PORT}/health ..."
if ! curl -s --connect-timeout 5 "http://${TRAINER_IP}:${CALLBACK_PORT}/health" > /dev/null 2>&1; then
    echo "WARNING: Trainer callback server not reachable yet. Make sure run_trainer.sh is running on Node 1."
    echo "Proceeding anyway — sglang will retry connections."
fi

# --- Mooncake connection to trainer node ---
export MOONCAKE_MASTER_SERVER="${TRAINER_IP}:${MOONCAKE_GRPC_PORT}"
export MOONCAKE_METADATA_SERVER="http://${TRAINER_IP}:${MOONCAKE_META_PORT}/metadata"
export MOONCAKE_LOCAL_HOSTNAME="${LOCAL_IP}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((16 * 1024 * 1024 * 1024))}"
export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-$((2 * 1024 * 1024 * 1024))}"

# --- Start sglang server (NVFP4 quantization, NO speculative decoding) ---
echo "Starting sglang server on port $SGLANG_PORT..."
python -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --port "$SGLANG_PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --mem-fraction-static "$MEM_FRACTION" \
    --tp-size "$SGLANG_TP_SIZE" \
    --context-length "$CONTEXT_LENGTH" \
    --quantization modelopt_fp4 \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend trtllm_mla \
    --moe-runner-backend flashinfer_trtllm \
    --disable-flashinfer-autotune \
    --disable-radix-cache \
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --enable-spec-training-mooncake \
    --enable-aux-hidden-states \
    --enable-return-hidden-states \
    --spec-training-callback-url "$CALLBACK_URL" \
    --log-requests \
    --log-requests-target "$SGLANG_REQ_LOG_DIR" \
    2>&1 | tee "$SGLANG_LOG"
