# PT-02 shared settings, sourced by pt02_prepare.sh, submit_pt02.sh and every pt02_*.sbatch (not run directly).
# Override any of these in the environment before submitting; sbatch's default --export=ALL passes them to the jobs,
# which source this file again and derive the same paths.
#
#   WORK   venv, repo clone, HF cache (docs/zaratan_setup.md layout; home, ~8 GB)
#   PT02   text pools, regenerated data, feature dumps (~240 GB), checkpoints, eval results (course scratch)
#   SMOKE  non-empty -> tiny end-to-end run under $PT02/b1_smoke (LIMIT rows per part, 1 epoch, 5 requests)

WORK="${WORK:-$HOME/rl-work}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$WORK/repo}}"
VENV="${VENV:-$WORK/venv}"
PT02="${PT02:-$HOME/scratch.msai635/rl-spec-decoding/pt02}"

if [ -n "${SMOKE:-}" ]; then
  RUN="${RUN:-b1_smoke}"; LIMIT="${LIMIT:-8}"; EPOCHS="${EPOCHS:-1}"; N_SERVE="${N_SERVE:-5}"
else
  RUN="${RUN:-b1}"; LIMIT="${LIMIT:-}"; EPOCHS="${EPOCHS:-4}"; N_SERVE="${N_SERVE:-100}"
fi
SEED="${SEED:-0}"
PARTS="${PARTS:-8}"                     # regeneration array size (fixed by pt02_prepare.sh)
GEN_BATCH="${GEN_BATCH:-64}"            # regenerate --batch-size; lower it (e.g. 32) on a 40 GB A100 if it OOMs
SHARD_SIZE="${SHARD_SIZE:-256}"         # regenerate --shard-size; part of the resume config, keep it fixed per run
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}" # same as A1 (PT-01), so B1 - A1 is data scale only
TREES="${TREES:-}"                      # non-empty -> eval also serves tree configs top-k / beam 2 and 4

TEXT="$PT02/text"                                   # prepare: pool, parts, ShareGPT held-out pool (shared by runs)
POOL="$TEXT/chat_27k.jsonl"                         # ~26.5K train rows + BP-04's 100 held-out rows, last
SG_POOL="$TEXT/sharegpt_heldout_200.jsonl"          # rows 800-999 of data/raw/chat.jsonl: dump keeps the last 100
A1_DIR="$PT02/a1"                                   # PT-01 A1 checkpoint (release pt01-a1-v1)
RUN_DIR="$PT02/$RUN"
REGEN="$RUN_DIR/regen"                              # one regenerated jsonl (+ resumable shards) per part
SELF_POOL="$RUN_DIR/chat_27k_self_1p7b.jsonl"       # merged regenerated pool
DUMP="$RUN_DIR/dump_self"                           # B1 features; train/ is deleted after training
DUMP_SG="$RUN_DIR/dump_sharegpt_heldout"            # ShareGPT held-out features (heldout/ only)
CKPT="$RUN_DIR/checkpoints"
EVAL="$RUN_DIR/eval"
LOGS="$RUN_DIR/logs"

export HF_HOME="$WORK/hf" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export MINI_AURORA_DEVICE="${MINI_AURORA_DEVICE:-cuda}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MINI_AURORA_TARGET="Qwen/Qwen3-1.7B"
export MINI_AURORA_DRAFT_CONFIG="$REPO/examples/zaratan-qwen3-1.7b/draft_config.json"

source "$VENV/bin/activate"
cd "$REPO"

job_header() {  # print what the job runs on
  echo "== $SLURM_JOB_NAME ${SLURM_ARRAY_TASK_ID:+task $SLURM_ARRAY_TASK_ID }run=$RUN $(date)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
  python -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
}

job_footer() {
  sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,AllocTRES%40,State -X 2>/dev/null || true
}
