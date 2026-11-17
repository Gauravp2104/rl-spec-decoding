#!/usr/bin/env bash
# PT-02 step 0, on a Zaratan LOGIN node (needs internet; CPU only, ~10-20 min). Run once, after setup_env.sh:
#   SCRATCH=$HOME/rl-work bash mini_aurora/zaratan/setup_env.sh     # if not done yet (docs/zaratan_setup.md)
#   bash mini_aurora/zaratan/pt02_prepare.sh                         # then: bash mini_aurora/zaratan/submit_pt02.sh
#
# 1. all filtered ShareGPT chat (seeded BP-02 download, no row cap) -> $PT02/text/raw/chat.jsonl
# 2. extend_pool: drop the 1000 BP-02 rows, keep every other row for training, append BP-04's 100 held-out rows
#    (chat-00900 ... chat-00999) last, so the dump's "last 100 rows are held out" rule matches A0 / A1
# 3. split the pool into $PARTS contiguous parts for the regeneration array
# 4. rows 800-999 of data/raw/chat.jsonl, for a ShareGPT-answer held-out dump (teacher-forced acc on both dumps)
# 5. the A1 checkpoint from release pt01-a1-v1, Qwen3-1.7B in the HF cache, and a disk check
set -eo pipefail
cd "$(dirname "$0")/../.."
export REPO="$PWD" HF_HUB_OFFLINE=0
source mini_aurora/zaratan/pt02_common.sh
mkdir -p "$TEXT/parts" "$A1_DIR"
[ -s data/raw/chat.jsonl ] || { echo "data/raw/chat.jsonl missing: run setup_env.sh first"; exit 1; }

# 1. full pool (download stops at --per-domain kept rows; 100000 is above the 27,475 that pass the filters)
if [ ! -s "$TEXT/raw/chat.jsonl" ]; then
  MINI_AURORA_DEVICE=cpu python -m mini_aurora.data.download --data-dir "$TEXT" --per-domain 100000 --domains chat
fi

# 2. training pool with the held-out rows last
N_TRAIN=$(( $(wc -l < "$TEXT/raw/chat.jsonl") - $(wc -l < data/raw/chat.jsonl) ))
[ -s "$POOL" ] || python -m mini_aurora.data.extend_pool --base data/raw/chat.jsonl --big "$TEXT/raw/chat.jsonl" \
  --out "$POOL" --train "$N_TRAIN" --heldout 100

# 3 + 4. regeneration parts and the ShareGPT held-out pool
python - "$POOL" "$TEXT/parts" "$PARTS" "$SG_POOL" <<'PY'
import sys
from pathlib import Path
from mini_aurora.data.download import read_jsonl, write_jsonl

pool, parts_dir, n, sg = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
rows = read_jsonl(pool)
assert [r["id"] for r in rows[-100:]] == [f"chat-{i:05d}" for i in range(900, 1000)], "held-out rows must be last"
existing = sorted(parts_dir.glob("part_*.jsonl"))
if existing:
    got = [r["id"] for p in existing for r in read_jsonl(p)]
    assert got == [r["id"] for r in rows], f"{parts_dir} does not match {pool}; delete it to re-split"
    print(f"parts: {len(existing)} already split")
else:
    size = -(-len(rows) // n)
    for i in range(n):
        write_jsonl(parts_dir / f"part_{i:02d}.jsonl", rows[i * size : (i + 1) * size])
    print(f"parts: {n} x <= {size} rows")
base = read_jsonl(Path("data/raw/chat.jsonl"))
write_jsonl(sg, base[-200:])
print(f"pool: {len(rows)} rows ({len(rows) - 100} train + 100 held-out); ShareGPT held-out pool: {sg}")
PY

# 5. A1 checkpoint (PT-01), target weights, disk
REL=https://github.com/Gauravp2104/rl-spec-decoding/releases/download/pt01-a1-v1
for f in draft_step000896.pt draft_step000896.json latest.txt; do
  [ -s "$A1_DIR/$f" ] || curl -fsSL -o "$A1_DIR/$f" "$REL/$f"
done
python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-1.7B', allow_patterns=['*.json','*.safetensors','*.txt']))"

echo "== disk: the self-distilled dump needs ~240 GB (bf16, ~16 KB per token) under $PT02"
df -h "$PT02" || true
beegfs-ctl --getquota --gid zt-msai635 2>/dev/null || true
echo "prepare done: $(ls "$TEXT"/parts/part_*.jsonl | wc -l) parts in $TEXT/parts; next: bash mini_aurora/zaratan/submit_pt02.sh"
