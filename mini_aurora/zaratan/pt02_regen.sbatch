#!/bin/bash
#SBATCH --job-name=pt02-regen
#SBATCH -A msai635-class
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH --output=slurm-%x-%A_%a.out
# PT-02 step 1, one array task per part: regenerate every assistant turn of $TEXT/parts/part_<i>.jsonl with
# Qwen3-1.7B (greedy, fp32, thinking off, <= 512 new tokens per turn; the PT-01 settings).
# Submitted by submit_pt02.sh with --array=0-<PARTS-1>. Resumable: finished shards are skipped, so a timed-out task
# is simply resubmitted (FROM=regen bash mini_aurora/zaratan/submit_pt02.sh, or sbatch --array=<i> this file).
set -eo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/mini_aurora/zaratan/pt02_common.sh"
job_header
i=$(printf '%02d' "${SLURM_ARRAY_TASK_ID:-0}")
mkdir -p "$REGEN"
t0=$(date +%s)
python -m mini_aurora.data.regenerate --model "$MINI_AURORA_TARGET" --jsonl "$TEXT/parts/part_$i.jsonl" \
  --out "$REGEN/part_$i.jsonl" --batch-size "$GEN_BATCH" --shard-size "$SHARD_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" ${LIMIT:+--limit $LIMIT}
echo "part $i: regenerate_s $(( $(date +%s) - t0 ))"
job_footer
