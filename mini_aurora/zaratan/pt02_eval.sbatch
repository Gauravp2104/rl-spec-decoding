#!/bin/bash
#SBATCH --job-name=pt02-eval
#SBATCH -A msai635-class
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH --output=slurm-%x-%j.out
# PT-02 step 4, one job per draft (PLAN §5, same harness for every arm):
#   sbatch --export=ALL,NAME=b1,DRAFT=<checkpoint dir> mini_aurora/zaratan/pt02_eval.sbatch   (submit_pt02.sh does this)
# Writes $EVAL/<NAME>/:
#   evaluate_self_heldout.json      teacher-forced acc on the self-distilled held-out dump
#   evaluate_sharegpt_heldout.json  teacher-forced acc on the ShareGPT held-out dump (A0's data)
#   serve_{heldout,code,math}.{jsonl,txt}   τ̄, α_i, lossless (--check) and target-only tok/s, K = 3 chain
#   serve_heldout_tree{2,4}.*       with TREES=1: -k 3 --tree-topk k --tree-beam k
set -eo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/mini_aurora/zaratan/pt02_common.sh"
job_header
: "${NAME:?set NAME}" "${DRAFT:?set DRAFT (checkpoint directory with latest.txt)}"
CK="$DRAFT/$(printf 'draft_step%06d.pt' "$(cat "$DRAFT/latest.txt")")"
OUT="$EVAL/$NAME"
mkdir -p "$OUT"
echo "draft $CK -> $OUT"

t0=$(date +%s)
python -m mini_aurora.baseline.evaluate --dump-dir "$DUMP" --ckpt "$CK" --out "$OUT/evaluate_self_heldout.json"
python -m mini_aurora.baseline.evaluate --dump-dir "$DUMP_SG" --ckpt "$CK" --out "$OUT/evaluate_sharegpt_heldout.json"
t1=$(date +%s)

serve() {  # tag serve-args...
  local tag=$1; shift
  rm -f "$OUT/serve_$tag.jsonl"   # --log appends
  python -m mini_aurora.server.serve -n "$N_SERVE" --check --draft "$DRAFT" --log "$OUT/serve_$tag.jsonl" "$@" \
    | tee "$OUT/serve_$tag.txt"
}
serve heldout --heldout
serve code --domain code
serve math --domain math
if [ -n "$TREES" ]; then
  for k in 2 4; do serve "heldout_tree$k" --heldout -k 3 --tree-topk "$k" --tree-beam "$k"; done
fi
t2=$(date +%s)
echo "{\"draft\": \"$CK\", \"evaluate_s\": $((t1-t0)), \"serve_s\": $((t2-t1)), \"n\": $N_SERVE, \"trees\": \"$TREES\"}" \
  | tee "$OUT/timings.json"
job_footer
