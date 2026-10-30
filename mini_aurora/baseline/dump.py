"""Dump EAGLE-3 target features for the chat pool (mirrors Aurora's offline data generation).

    python -m mini_aurora.baseline.dump                       # data/raw/chat.jsonl -> outputs/baseline/hidden_states/{train,heldout}/
    python -m mini_aurora.baseline.dump --limit 20            # smoke test on 20 conversations

One file per conversation, with the keys Aurora's ``OfflineEagle3Dataset`` expects:

| key | shape | meaning |
|---|---|---|
| ``input_ids`` | (T,) | chat-templated conversation |
| ``loss_mask`` | (T,) | 1 on assistant tokens (last position zeroed: it has no next token) |
| ``aux_hidden_states`` | (1, T, 3 x H) | target layers 1, 13, 24 concatenated: the draft's input |
| ``hidden_states`` | (1, T, H) | target final-norm output: ``lm_head`` of this gives the labels |

The last ``HELDOUT`` rows of the pool go to ``heldout/`` and are never trained on; the rest go to
``train/``. A ``manifest.json`` records counts, token totals, dtype and timing.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from mini_aurora.baseline import DUMP_DIR, DUMP_DTYPE, HELDOUT
from mini_aurora.data.download import read_jsonl
from mini_aurora.device import get_device
from mini_aurora.models.target import capture_features, load_target, load_tokenizer
from mini_aurora.models.vocab import tokenize_conversations

DEFAULT_JSONL = "data/raw/chat.jsonl"


def split_rows(rows: list, heldout: int = HELDOUT) -> dict:
    if heldout >= len(rows):
        raise ValueError(f"heldout={heldout} leaves no training rows out of {len(rows)}")
    return {"train": rows[: len(rows) - heldout], "heldout": rows[len(rows) - heldout :]}


@torch.no_grad()
def dump_conversation(target, tokenizer, messages: list, device: torch.device, dtype: torch.dtype) -> dict | None:
    """Tokenize one conversation, capture features, return CPU tensors (None if it has no assistant tokens)."""
    data = tokenize_conversations(tokenizer, [messages])
    if not data["input_ids"]:
        return None
    ids, mask = data["input_ids"][0].to(device), data["loss_mask"][0].to(device)
    features = capture_features(target, ids, mask)
    mask = mask.clone()
    mask[0, -1] = 0  # Aurora's OfflineEagle3Dataset: the last position has no next token
    return {
        "input_ids": ids[0].cpu(),
        "loss_mask": mask[0].cpu(),
        "aux_hidden_states": features.hidden_states.to(dtype).cpu(),
        "hidden_states": features.last_hidden_states.to(dtype).cpu(),
    }


def run_dump(jsonl: str, out_dir: str, heldout: int, limit: int | None, dtype: torch.dtype,
             device: torch.device | None = None) -> dict:
    device = device or get_device()
    rows = read_jsonl(Path(jsonl))
    if limit:
        rows = rows[:limit]
    splits = split_rows(rows, min(heldout, max(1, len(rows) // 5)) if limit else heldout)
    target, tokenizer = load_target(device=device), load_tokenizer()
    out = Path(out_dir)
    manifest = {"source": jsonl, "dtype": str(dtype).split(".")[-1], "model": target.model.config._name_or_path,
                "aux_layers": target.aux_hidden_states_layers, "splits": {}}
    for split, split_rows_ in splits.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        n_tokens = n_assistant = skipped = 0
        start = time.perf_counter()
        for r in split_rows_:
            item = dump_conversation(target, tokenizer, r["messages"], device, dtype)
            if item is None:
                skipped += 1
                continue
            torch.save(item, out / split / f"{r['id']}.ckpt")
            n_tokens += item["input_ids"].numel()
            n_assistant += int(item["loss_mask"].sum())
        seconds = time.perf_counter() - start
        manifest["splits"][split] = {"conversations": len(split_rows_) - skipped, "skipped": skipped,
                                     "tokens": n_tokens, "assistant_tokens": n_assistant,
                                     "seconds": round(seconds, 1)}
        print(f"{split}: {len(split_rows_) - skipped} conversations, {n_tokens} tokens "
              f"({n_assistant} assistant), {seconds:.0f}s -> {out / split}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", default=DEFAULT_JSONL)
    p.add_argument("--out", default=DUMP_DIR)
    p.add_argument("--heldout", type=int, default=HELDOUT)
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    p.add_argument("--dtype", default=DUMP_DTYPE, choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    args = p.parse_args()
    run_dump(args.jsonl, args.out, args.heldout, args.limit, getattr(torch, args.dtype), get_device(args.device))


if __name__ == "__main__":
    main()
