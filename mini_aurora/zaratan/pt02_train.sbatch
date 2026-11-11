#!/bin/bash
#SBATCH --job-name=pt02-train
#SBATCH -A msai635-class
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH --output=slurm-%x-%j.out
# PT-02 step 3 (after the dump): train B1, the BP-09 / A1 recipe (1.7B draft config, 24K vocab, aux 1/13/24, fp32 +
# gradient checkpointing, LR 1e-4, batch 2 x 4) for $EPOCHS = 4 epochs, cosine over the whole run. One checkpoint and
# one held-out evaluation per epoch (train.py). The trainer cannot resume, so a failed run must be restarted from
# scratch: delete $CKPT first (this job refuses to overwrite a run). On success the dump's train/ split (~240 GB)
# is deleted; heldout/ stays for pt02_eval.sbatch. KEEP_DUMP=1 keeps it.
set -eo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/mini_aurora/zaratan/pt02_common.sh"
job_header
if [ -e "$CKPT/train_log.csv" ]; then echo "$CKPT already has a run; delete it to retrain"; exit 1; fi
t0=$(date +%s)
python -m mini_aurora.baseline.train --dump-dir "$DUMP" --out "$CKPT" --epochs "$EPOCHS" --seed "$SEED"
echo "{\"train_s\": $(( $(date +%s) - t0 )), \"epochs\": $EPOCHS, \"seed\": $SEED}" | tee "$RUN_DIR/timings_train.json"
[ -n "${KEEP_DUMP:-}" ] || rm -rf "$DUMP/train"
job_footer
