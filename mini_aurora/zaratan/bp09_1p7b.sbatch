#!/bin/bash
#SBATCH --job-name=bp09-1p7b
#SBATCH -A msai635-class
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 0:25:00
#SBATCH --output=slurm-%x-%j.out
# BP-09 scale check: BP-04's pipeline + BP-05's held-out τ̄ with Qwen3-1.7B as the target, on one A100.
#   smoke (default, ~10 min, ≤ 20 SU):   sbatch mini_aurora/zaratan/bp09_1p7b.sbatch
#   full  (~1-1.5 h, ~50-75 SU):          sbatch -t 2:00:00 --export=ALL,LIMIT=,EPOCHS=8 mini_aurora/zaratan/bp09_1p7b.sbatch
# Walltime is the SU cap: 1 A100 hour = 48 SU, charged for the time used.

SCRATCH="${SCRATCH:-$HOME/scratch.msml612/rl-spec-decoding}"
source "${VENV:-$SCRATCH/venv}/bin/activate"
cd "${REPO:-$SCRATCH/repo}"

LIMIT="${LIMIT-20}"          # LIMIT= (empty) for the whole chat pool
EPOCHS="${EPOCHS:-1}"
OUT="$SCRATCH/outputs/1p7b${LIMIT:+_limit$LIMIT}_${EPOCHS}ep"
export HF_HOME="$SCRATCH/hf" HF_HUB_OFFLINE=1 WANDB_MODE=offline MINI_AURORA_DEVICE=cuda
export MINI_AURORA_TARGET="Qwen/Qwen3-1.7B"
export MINI_AURORA_DRAFT_CONFIG="$PWD/examples/zaratan-qwen3-1.7b/draft_config.json"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch;print('torch',torch.__version__,torch.version.cuda)"

t0=$(date +%s)
python -m mini_aurora.baseline.dump --out "$OUT/hidden_states" ${LIMIT:+--limit $LIMIT}
t1=$(date +%s)
python -m mini_aurora.baseline.train --dump-dir "$OUT/hidden_states" --out "$OUT/checkpoints" --epochs "$EPOCHS"
t2=$(date +%s)
python -m mini_aurora.baseline.evaluate --dump-dir "$OUT/hidden_states" --ckpt "$OUT/checkpoints/$(printf 'draft_step%06d.pt' "$(cat "$OUT/checkpoints/latest.txt")")" --out "$OUT/evaluate.json"
t3=$(date +%s)
if [ -z "$LIMIT" ]; then
  python -m mini_aurora.server.serve -n 100 --heldout --check --draft "$OUT/checkpoints" --log "$OUT/serve_heldout.jsonl" | tee "$OUT/serve_heldout.txt"
fi
t4=$(date +%s)
echo "{\"dump_s\": $((t1-t0)), \"train_s\": $((t2-t1)), \"evaluate_s\": $((t3-t2)), \"serve_s\": $((t4-t3)), \"limit\": \"$LIMIT\", \"epochs\": $EPOCHS}" | tee "$OUT/timings.json"
rm -rf "$OUT/hidden_states"      # 1.7B dump is ~8 GB at full size; not kept
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,AllocTRES%40,State -X

