#!/bin/bash
#SBATCH --job-name=pt02-dump
#SBATCH -A msai635-class
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH --output=slurm-%x-%j.out
# PT-02 step 2 (after every regen task): merge the regenerated parts, then dump Qwen3-1.7B features for
#   - the merged self-distilled pool -> $DUMP/{train,heldout}   (~240 GB at full size)
#   - the ShareGPT held-out pool     -> $DUMP_SG/heldout          (train/ deleted; for teacher-forced acc on A0-style data)
# A dump with a manifest.json is complete and skipped on resubmit; an incomplete one is deleted and redone.
set -eo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/mini_aurora/zaratan/pt02_common.sh"
job_header
mkdir -p "$RUN_DIR"

# merge: parts in order, the same rows as the pool (all of them unless smoke), stats over the whole pool
python - "$TEXT/parts" "$REGEN" "$SELF_POOL" "${LIMIT:-}" <<'PY'
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer
from mini_aurora.data.download import read_jsonl, write_jsonl
from mini_aurora.data.regenerate import summarize

parts_dir, regen_dir, out, limit = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
parts = sorted(parts_dir.glob("part_*.jsonl"))
orig, new, seconds = [], [], 0.0
for p in parts:
    o = read_jsonl(p)[: int(limit)] if limit else read_jsonl(p)
    r = regen_dir / p.name
    if not r.exists():
        raise SystemExit(f"{r} missing: regen task for {p.name} did not finish")
    n = read_jsonl(r)
    assert [x["id"] for x in n] == [x["id"] for x in o], f"{r} rows differ from {p}"
    orig += o
    new += n
    seconds += json.loads(r.with_name(r.stem + ".stats.json").read_text())["seconds"]
if not limit:
    assert new[-100:][0]["id"] == "chat-00900" and new[-1]["id"] == "chat-00999", "held-out rows must stay last"
write_jsonl(out, new)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
stats = summarize(orig, new, lambda s: len(tok(s, add_special_tokens=False)["input_ids"]))
stats.update(regenerate_gpu_seconds=round(seconds, 1), parts=len(parts), limit_per_part=limit or None)
out.with_name(out.stem + ".stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
PY

dump() {  # jsonl out [limit]
  if [ -s "$2/manifest.json" ]; then echo "$2 complete, skipped"; return; fi
  rm -rf "$2"
  python -m mini_aurora.baseline.dump --jsonl "$1" --out "$2" --heldout 100 ${3:+--limit $3}
}
t0=$(date +%s)
dump "$SELF_POOL" "$DUMP" "${LIMIT:+$(( LIMIT * $(ls "$TEXT"/parts/part_*.jsonl | wc -l) ))}"
t1=$(date +%s)
dump "$SG_POOL" "$DUMP_SG" "${LIMIT:+40}"
rm -rf "$DUMP_SG/train"
t2=$(date +%s)
echo "{\"dump_self_s\": $((t1-t0)), \"dump_sharegpt_heldout_s\": $((t2-t1))}" | tee "$RUN_DIR/timings_dump.json"
du -sh "$DUMP" "$DUMP_SG"
job_footer
