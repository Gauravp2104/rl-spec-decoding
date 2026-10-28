#!/bin/bash
#SBATCH --job-name=bp08
#SBATCH -A msai635-class
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --output=slurm-%x-%j.out
# One BP-08 arm on one A100:  sbatch --export=ALL,EXP=E2,ARM=static [,GRID=N5] [,N=300] mini_aurora/zaratan/bp08_arm.sbatch
# Outputs: $WORK/outputs/experiments/<EXP>/<arm>_<stream>[_<grid>]/ ; submit all arms with submit_bp08.sh.
set -eo pipefail
WORK="${WORK:-$HOME/rl-work}"
source "$WORK/venv/bin/activate"
cd "$WORK/repo"
export HF_HOME="$WORK/hf" HF_HUB_OFFLINE=1 WANDB_MODE=offline MINI_AURORA_DEVICE=cuda TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$WORK/outputs/experiments"
ARGS=()
[ -n "${GRID:-}" ] && ARGS+=(--grid "$GRID")
[ -n "${N:-}" ] && ARGS+=(-n "$N")
[ -n "${LOOP_ARGS:-}" ] && ARGS+=(--loop-args "$LOOP_ARGS")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "== $EXP $ARM ${GRID:-} n=${N:-all} ==  $(date)"
t0=$(date +%s)
python -m mini_aurora.experiments.run "$EXP" "$ARM" --out-dir "$OUT" --draft outputs/baseline/checkpoints_8ep "${ARGS[@]}"
echo "elapsed_s $(( $(date +%s) - t0 ))"
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,AllocTRES%30,State -X

