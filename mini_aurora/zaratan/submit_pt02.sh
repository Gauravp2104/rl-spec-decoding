#!/usr/bin/env bash
# Submit the PT-02 chain (B1: ~26.5K conversations self-distilled by Qwen3-1.7B, 4 epochs) as dependent A100 jobs.
# Run from a login node after pt02_prepare.sh:
#
#   SMOKE=1 bash mini_aurora/zaratan/submit_pt02.sh     # tiny end-to-end chain (8 rows per part, 1 epoch, 5 requests)
#   bash mini_aurora/zaratan/submit_pt02.sh             # full chain
#
#   regen (array, one task per part) ──afterok──> dump ──afterok──> train ──afterok──> eval b1
#                                                    └──afterok──> eval a1   (A1 from PT-01, needs only the dumps)
#                                                    └──afterok──> eval a0   (only with A0=<BP-09 checkpoint dir>)
#
# Options (environment):
#   FROM=dump|train|eval   skip the earlier stages (their outputs must exist); FROM=regen also resumes a partial regen
#   AFTER=<jobid>          make the first submitted stage wait for this job
#   EVALS="a1 b1"          which drafts to evaluate (default a1 b1, plus a0 when A0 is set)
#   TREES=1                eval also serves tree configs (top-k / beam 2 and 4) on held-out chat
#   GPU=h100:1 PART=...    other GPU type / partition (default a100:1 on gpu)
#   DRY=1                  print the sbatch commands instead of submitting
# Paths and the remaining knobs (PARTS, GEN_BATCH, EPOCHS, PT02, WORK ...) are in pt02_common.sh.
set -eo pipefail
cd "$(dirname "$0")/../.."
export REPO="$PWD"
source mini_aurora/zaratan/pt02_common.sh
export SMOKE TREES   # the jobs re-source pt02_common.sh and derive the same RUN / LIMIT / paths

FROM="${FROM:-regen}"
A0="${A0:-}"
EVALS="${EVALS:-a1 b1${A0:+ a0}}"
order() { case $1 in regen) echo 1 ;; dump) echo 2 ;; train) echo 3 ;; eval) echo 4 ;; *) echo 0 ;; esac; }
[ "$(order "$FROM")" -gt 0 ] || { echo "FROM must be regen, dump, train or eval"; exit 1; }
runs() { [ "$(order "$1")" -ge "$(order "$FROM")" ]; }

n_parts=$(( $(ls "$TEXT"/parts/part_*.jsonl 2>/dev/null | wc -l) ))
SFX="${SMOKE:+-smoke}"
[ "$n_parts" -gt 0 ] || { echo "no parts in $TEXT/parts: run pt02_prepare.sh first"; exit 1; }
[ -s "$A1_DIR/latest.txt" ] || [[ " $EVALS " != *" a1 "* ]] || { echo "A1 checkpoint missing in $A1_DIR"; exit 1; }
mkdir -p "$LOGS"

S=mini_aurora/zaratan
GPU_ARGS=()
[ -n "${GPU:-}" ] && GPU_ARGS+=(--gpus="$GPU")
[ -n "${PART:-}" ] && GPU_ARGS+=(-p "$PART")
if [ -n "$SMOKE" ]; then T_REGEN=0:30:00 T_DUMP=0:30:00 T_TRAIN=0:30:00 T_EVAL=0:45:00
else T_REGEN=6:00:00 T_DUMP=8:00:00 T_TRAIN=16:00:00 T_EVAL=4:00:00; fi

submit() {  # walltime after-jobid log-pattern sbatch-args...
  local t=$1 after=$2 log=$3; shift 3
  local cmd=(sbatch --parsable -t "$t" -o "$LOGS/$log" "${GPU_ARGS[@]}" ${after:+--dependency=afterok:$after} "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY$RANDOM"; else "${cmd[@]}"; fi
}

prev="${AFTER:-}"
if runs regen; then
  prev=$(submit "$T_REGEN" "$prev" '%x-%A_%a.out' --array=0-$((n_parts - 1)) -J "pt02-regen$SFX" "$S/pt02_regen.sbatch")
  echo "regen  $prev  ($n_parts tasks)"
fi
if runs dump; then
  prev=$(submit "$T_DUMP" "$prev" '%x-%j.out' -J "pt02-dump$SFX" "$S/pt02_dump.sbatch")
  echo "dump   $prev"
fi
dumped="$prev"
for name in $EVALS; do
  case $name in
    a1) draft="$A1_DIR" ;;
    a0) draft="${A0:?A0=<BP-09 checkpoint dir> needed for a0}" ;;
    b1) continue ;;
    *) echo "unknown eval $name"; exit 1 ;;
  esac
  if runs eval; then
    j=$(submit "$T_EVAL" "$dumped" '%x-%j.out' -J "pt02-eval-$name$SFX" --export="ALL,NAME=$name,DRAFT=$draft" "$S/pt02_eval.sbatch")
    echo "eval   $j  ($name)"
  fi
done
if runs train; then
  prev=$(submit "$T_TRAIN" "$prev" '%x-%j.out' -J "pt02-train$SFX" "$S/pt02_train.sbatch")
  echo "train  $prev"
fi
if [[ " $EVALS " == *" b1 "* ]] && runs eval; then
  j=$(submit "$T_EVAL" "$prev" '%x-%j.out' -J "pt02-eval-b1$SFX" --export="ALL,NAME=b1,DRAFT=$CKPT" "$S/pt02_eval.sbatch")
  echo "eval   $j  (b1)"
fi
echo "logs: $LOGS   outputs: $RUN_DIR"
[ -n "${DRY:-}" ] || squeue -u "$USER" -o "%.10i %.28j %.8T %.10M %.6D %R"
