#!/bin/bash
#SBATCH --job-name=pt02-build
#SBATCH -A msai635-class
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 20:00:00
#SBATCH --output=slurm-%x-%j.out
# PT-02 steps 2 + 3 in ONE job, for LOCAL=1 (no course scratch): the feature dumps (~240 GB) and per-epoch checkpoints
# live on this node's local disk ($TMPDIR, 12 TB on the A100 nodes), which is wiped when the job ends, so everything
# that reads them runs here:
#   1. pt02_dump.sbatch   merge the regenerated parts (-> $RUN_DIR, home) and dump both pools -> $NODE_TMP
#   2. pt02_train.sbatch  4 epochs on $NODE_TMP; the final checkpoint + logs are copied to $CKPT (home)
#   3. teacher-forced acc on both held-out dumps for every draft in $EVALS -> $EVAL/<name>/evaluate_*.json
# pt02_eval.sbatch then only serves (no dumps needed). Nothing here resumes: if the job fails, delete $CKPT and
# resubmit (FROM=dump LOCAL=1 bash mini_aurora/zaratan/submit_pt02.sh). Submitted by submit_pt02.sh when LOCAL=1.
set -eo pipefail
S="${REPO:-$SLURM_SUBMIT_DIR}/mini_aurora/zaratan"
source "$S/pt02_common.sh"
[ -n "$LOCAL" ] || { echo "pt02_build.sbatch is the LOCAL=1 path; without LOCAL use pt02_dump + pt02_train"; exit 1; }
job_header
mkdir -p "$NODE_TMP"
trap 'rm -rf "$NODE_TMP"' EXIT
df -h "$NODE_TMP"

bash "$S/pt02_dump.sbatch"
bash "$S/pt02_train.sbatch"

t0=$(date +%s)
for name in ${EVALS:-a1 b1}; do
  case $name in
    a1) draft="$A1_DIR" ;;
    b1) draft="$CKPT" ;;
    a0) draft="${A0:?A0=<BP-09 checkpoint dir> needed for a0}" ;;
    *) echo "unknown eval $name"; exit 1 ;;
  esac
  ck="$draft/$(printf 'draft_step%06d.pt' "$(cat "$draft/latest.txt")")"
  mkdir -p "$EVAL/$name"
  python -m mini_aurora.baseline.evaluate --dump-dir "$DUMP" --ckpt "$ck" --out "$EVAL/$name/evaluate_self_heldout.json"
  python -m mini_aurora.baseline.evaluate --dump-dir "$DUMP_SG" --ckpt "$ck" --out "$EVAL/$name/evaluate_sharegpt_heldout.json"
done
echo "{\"evaluate_s\": $(( $(date +%s) - t0 )), \"evals\": \"${EVALS:-a1 b1}\"}" | tee "$RUN_DIR/timings_build_evaluate.json"
du -sh "$NODE_TMP" "$RUN_DIR"
job_footer
