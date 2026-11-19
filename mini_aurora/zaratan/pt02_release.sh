#!/usr/bin/env bash
# PT-02 step 5: publish B1 as a GitHub release, the fixed checkpoint GRPO starts from. Two subcommands:
#
#   bash mini_aurora/zaratan/pt02_release.sh stage              # Zaratan login node, after `eval b1` finished
#   bash mini_aurora/zaratan/pt02_release.sh publish [DIR]      # anywhere with an authenticated `gh` (default DIR: stage dir)
#
# stage   checks the run (B1 checkpoint, every serve lossless, the GRPO baseline tree present, 100 requests) and copies
#         the release files flat into $RUN_DIR/release/ (~0.6 GB): the latest checkpoint (.pt/.json/latest.txt), train
#         eval.json / train_log.csv, the regenerated pool + stats, eval/b1 (summaries, per-request logs), timings,
#         plus NOTES.md (generated from the results, with A1 for comparison if eval/a1 exists) and SHA256SUMS.
#         Needs only bash + python3 (no venv, no GPU).
# publish creates the release as a DRAFT (`gh release create --draft`) pointing at the commit that trained B1. Check it
#         on GitHub, then make it public: gh release edit $TAG -R $GH_REPO --draft=false
#
# If `gh` is not on Zaratan, copy the stage dir to your machine and publish from a repo checkout there:
#   rsync -av zaratan:scratch.msai635/rl-spec-decoding/pt02/b1/release/ outputs/pt02_b1_release/
#   bash mini_aurora/zaratan/pt02_release.sh publish outputs/pt02_b1_release
#
# Options (environment): LOCAL=1 (run made with LOCAL=1: PT02 defaults to ~/rl-work/pt02), PT02, RUN (default b1; b1_smoke stages the smoke run for a dry check), TAG (pt02-b1-v1),
# GH_REPO (Gauravp2104/rl-spec-decoding), ALLOW_SMOKE=1 (stage/publish a run with < 100 requests), DRY=1 (publish:
# print the gh command only).
set -eo pipefail
cd "$(dirname "$0")/../.."
if [ -n "${LOCAL:-}" ]; then PT02="${PT02:-$HOME/rl-work/pt02}"; else PT02="${PT02:-$HOME/scratch.msai635/rl-spec-decoding/pt02}"; fi
RUN="${RUN:-b1}"
RUN_DIR="$PT02/$RUN"
TAG="${TAG:-pt02-b1-v1}"
GH_REPO="${GH_REPO:-Gauravp2104/rl-spec-decoding}"
cmd="${1:-}"

stage() {
  local ck="$RUN_DIR/checkpoints" ev="$RUN_DIR/eval/b1" out="$RUN_DIR/release"
  [ -s "$ck/latest.txt" ] || { echo "no checkpoint in $ck: train has not finished"; exit 1; }
  [ -s "$ev/summary.json" ] || { echo "no $ev/summary.json: eval b1 has not finished"; exit 1; }
  local step; step=$(printf '%06d' "$(cat "$ck/latest.txt")")
  rm -rf "$out" && mkdir -p "$out"
  cp "$ck/draft_step$step.pt" "$ck/draft_step$step.json" "$ck/latest.txt" "$ck/eval.json" "$ck/train_log.csv" "$out/"
  cp "$RUN_DIR/chat_27k_self_1p7b.jsonl" "$RUN_DIR/chat_27k_self_1p7b.stats.json" "$out/"
  cp "$RUN_DIR/timings_dump.json" "$RUN_DIR/timings_train.json" "$out/"
  cp "$ev/timings.json" "$out/timings_eval.json"
  cp "$ev"/summary.json "$ev"/evaluate_self_heldout.json "$ev"/evaluate_sharegpt_heldout.json "$ev"/serve_*.jsonl "$out/"
  for f in "$ev"/serve_*.txt; do [ -e "$f" ] && cp "$f" "$out/"; done
  local a1="$RUN_DIR/eval/a1/summary.json"
  python3 - "$out" "$step" "$TAG" "${ALLOW_SMOKE:-}" "$([ -s "$a1" ] && echo "$a1")" <<'PY'
import json
import sys
from pathlib import Path

out, step, tag, allow_smoke, a1_path = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
load = lambda name: json.loads((out / name).read_text())
summary, stats, train = load("summary.json"), load("chat_27k_self_1p7b.stats.json"), load("timings_train.json")
a1 = json.loads(Path(a1_path).read_text()) if a1_path else {}

bad = [k for k, v in summary.items() if v.get("matches", "").split("/")[0] != str(v["requests"])]
if bad:
    sys.exit(f"not lossless: {bad}; do not release")
grpo = [k for k in summary if k.startswith("heldout_k") and "_topk" in k]
if not grpo:
    sys.exit("no GRPO baseline tree serve (serve_heldout_k*_topk*_beam*) in eval/b1: rerun FROM=eval EVALS=b1")
small = [k for k, v in summary.items() if v["requests"] < 100]
if small and not allow_smoke:
    sys.exit(f"serves with < 100 requests: {small} (smoke run?); ALLOW_SMOKE=1 to stage anyway")

def label(key):
    if key == "heldout":
        return "held-out chat, chain K = 3"
    if key in ("code", "math"):
        return f"{key} (out of domain), chain K = 3"
    if key.startswith("heldout_tree"):
        t = key.removeprefix("heldout_tree")
        return f"held-out chat, tree K = 3, top-k {t}, beam {t}"
    d, t, b = key.removeprefix("heldout_k").replace("_topk", " ").replace("_beam", " ").split()
    return f"held-out chat, tree K = {d}, top-k {t}, beam {b} (**GRPO baseline**)"

order = ["heldout", *grpo, *sorted(k for k in summary if k.startswith("heldout_tree")), "code", "math"]
rows = []
for key in [k for k in order if k in summary]:
    v = summary[key]
    a = a1.get(key, {}).get("tau")
    rows.append(f"| {label(key)} | **{v['tau']:.3f}** | {'' if a is None else f'{a:.3f}'} | "
                f"{' / '.join(f'{x:.3f}' for x in v['alpha'])} | {v['tokens_per_s']} | {v['matches']} |")
acc = lambda name: " / ".join(f"{x:.3f}" for x in load(name)["acc"])
g = summary[grpo[0]]
notes = f"""PT-02 arm B1: EAGLE-3 draft for **Qwen3-1.7B** trained on **self-distilled** chat data at scale, the offline \
draft the GRPO project starts from. Every filtered ShareGPT chat conversation outside the BP-02 pool keeps its user \
turns, and every assistant turn is regenerated by Qwen3-1.7B (greedy, fp32, `enable_thinking=False`, ≤ 512 new tokens \
per turn; `mini_aurora/data/regenerate.py`, the PT-01 settings). The 100 BP-04 held-out conversations \
(`chat-00900`…`chat-00999`) are last and held out, as for A0 / A1.

- data: {stats['conversations']} conversations ({stats['conversations'] - 100} train + 100 held-out), \
{stats['assistant_tokens_generated']:,} generated assistant tokens, hit-cap rate {stats['hit_cap_rate']}, \
looped rate {stats['looped_rate']}
- training: BP-09 / A1 recipe (`examples/zaratan-qwen3-1.7b/draft_config.json`, 24K vocab `qwen3_24k_mixed`, fp32 + \
gradient checkpointing), {train['epochs']} epochs = {step} steps, seed {train['seed']}, Zaratan A100, \
{train['train_s'] / 3600:.1f} h; commit `{train.get('commit', '?')}`

**GRPO baseline: τ̄ {g['tau']:.3f}** on held-out chat at {label(grpo[0]).split(', ', 1)[1].removesuffix(' (**GRPO baseline**)')}, \
{g['matches']} lossless.

Serve results (greedy, 128 new tokens, fp32, A100; A1 = release `pt01-a1-v1` on the same harness):

| setting | τ̄ B1 | τ̄ A1 | α1 / α2 / … (B1) | tok/s | lossless |
|---|---|---|---|---|---|
{chr(10).join(rows)}

Teacher-forced acc per position: self-distilled held-out {acc('evaluate_self_heldout.json')}; ShareGPT held-out \
{acc('evaluate_sharegpt_heldout.json')}.

Assets:
- `draft_step{step:06d}.pt`: load with `mini_aurora.baseline.checkpoint.load_draft` or \
`MINI_AURORA_TARGET=Qwen/Qwen3-1.7B MINI_AURORA_DRAFT_CONFIG=examples/zaratan-qwen3-1.7b/draft_config.json \
python -m mini_aurora.server.serve --draft <dir>` (`latest.txt` is included)
- `chat_27k_self_1p7b.jsonl` (+ `.stats.json`): the regenerated pool
- `summary.json`, `serve_*.jsonl` (per-request acceptance logs), `evaluate_*.json`, `eval.json`, `train_log.csv`, timings

Download: `gh release download {tag} -R Gauravp2104/rl-spec-decoding -D outputs/pt02_b1 && (cd outputs/pt02_b1 && shasum -a 256 -c SHA256SUMS)`
"""
(out / "NOTES.md").write_text(notes)
print(notes)
PY
  (cd "$out" && shasum -a 256 $(ls | grep -v -e '^NOTES.md$' -e '^SHA256SUMS$') > SHA256SUMS)
  printf '\n## sha256\n\n```\n%s\n```\n' "$(cat "$out/SHA256SUMS")" >> "$out/NOTES.md"
  du -sh "$out"; ls "$out" | wc -l
  echo "staged in $out; next: bash mini_aurora/zaratan/pt02_release.sh publish   (or rsync it and publish elsewhere)"
}

publish() {
  local dir="${1:-$RUN_DIR/release}"
  [ -s "$dir/NOTES.md" ] && [ -s "$dir/SHA256SUMS" ] || { echo "$dir is not staged: run '$0 stage' first"; exit 1; }
  (cd "$dir" && shasum -a 256 -c --quiet SHA256SUMS) || { echo "checksum mismatch in $dir"; exit 1; }
  local commit; commit=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('commit',''))" "$dir/timings_train.json")
  local files=(); while IFS= read -r f; do files+=("$dir/$f"); done < <(ls "$dir" | grep -v '^NOTES.md$')
  local cmd=(gh release create "$TAG" -R "$GH_REPO" --draft --notes-file "$dir/NOTES.md"
             --title "PT-02 B1: self-distilled Qwen3-1.7B draft, ~26.5K conversations" ${commit:+--target "$commit"} "${files[@]}")
  if [ -n "${DRY:-}" ]; then printf '%q ' "${cmd[@]}"; echo; return; fi
  command -v gh >/dev/null || { echo "gh not found: rsync $dir to a machine with gh (see the header)"; exit 1; }
  "${cmd[@]}"
  echo "draft release $TAG created; review it, then: gh release edit $TAG -R $GH_REPO --draft=false"
}

case "$cmd" in
  stage) stage ;;
  publish) shift; publish "$@" ;;
  *) echo "usage: $0 stage | publish [dir]"; exit 1 ;;
esac
