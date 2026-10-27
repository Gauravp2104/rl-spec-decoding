#!/bin/bash
#SBATCH --job-name=bp09-tests
#SBATCH -A msai635-class
#SBATCH --partition=gpu-a100_1g.5gb
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 0:40:00
#SBATCH --output=slurm-%x-%j.out
# BP-09: the whole mini_aurora test suite on a CUDA device (A100 MIG slice; ~12 SU/h).
set -eo pipefail
WORK="${WORK:-$HOME/rl-work}"
source "$WORK/venv/bin/activate"
cd "$WORK/repo"
export HF_HOME="$WORK/hf" HF_HUB_OFFLINE=1 WANDB_MODE=offline MINI_AURORA_DEVICE=cuda TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
python -m pytest -q mini_aurora/tests 2>&1 | tail -25
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,AllocTRES%30,State -X

