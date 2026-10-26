#!/usr/bin/env bash
# Submit every BP-08 arm as its own A100 job (PLAN.md E1-E4, full streams).
#   bash mini_aurora/zaratan/submit_bp08.sh            # 15 jobs
#   N=300 bash mini_aurora/zaratan/submit_bp08.sh      # shorter runs
#   ONLY="E2 static" bash mini_aurora/zaratan/submit_bp08.sh
#   GPU=a100_1g.5gb:1 PART=gpu-a100_1g.5gb bash mini_aurora/zaratan/submit_bp08.sh   # MIG slices (7 SU/h, short queue)
set -eo pipefail
GPU="${GPU:-a100:1}"; PART="${PART:-gpu}"
cd "$(dirname "$0")/../.."
S=mini_aurora/zaratan/bp08_arm.sbatch
submit() {  # exp arm [grid] [walltime]
  local exp=$1 arm=$2 grid=${3:-} t=${4:-4:00:00}
  if [ -n "${ONLY:-}" ] && [ "${ONLY}" != "$exp $arm${grid:+ $grid}" ]; then return; fi
  sbatch -t "$t" -p "$PART" --gpus="$GPU" -J "bp08-$exp-$arm${grid:+-$grid}" --export=ALL,EXP=$exp,ARM=$arm,GRID=$grid,N=${N:-} "$S"
}
submit E1 target_only "" 3:00:00
submit E1 day0        "" 4:00:00
submit E1 online_day0 "" 6:00:00
submit E2 target_only "" 2:00:00
submit E2 static      "" 2:00:00
submit E2 online_static "" 3:00:00
submit E3 target_only "" 3:00:00
submit E3 static      "" 4:00:00
submit E3 online_static "" 6:00:00
for g in N5 N20 N50 lr3e-4 lr3e-5 async; do submit E4 online_day0 "$g" 6:00:00; done
squeue -u "$USER" -o "%.10i %.22j %.8T %.10M %.6D %R"
