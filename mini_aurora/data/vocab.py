"""Cache the 24K draft vocab mapping from a stream and report per-domain coverage (needs Aurora: torch >= 2.3).

    python -m mini_aurora.data.vocab                       # mixed stream -> outputs/vocab/qwen3_24k_mixed.pt
    python -m mini_aurora.data.vocab --compare             # also build chat-only and report both coverages

Building, coverage and the ``{"d2t", "t2d", "meta"}`` file format live in ``mini_aurora.models.vocab`` (BP-03).
The mapping is built from the **mixed** stream (all three domains): tokens outside the 24K can never be
drafted, so a chat-only mapping would cap acceptance on the code and math segments of the shift stream
(docs/concepts/vocabulary.md). ``--compare`` prints the numbers behind that choice.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from mini_aurora.data import DATA_DIR, DOMAINS
from mini_aurora.data.download import TOKENIZER, read_jsonl
from mini_aurora.models.vocab import (
    DEFAULT_VOCAB,
    DRAFT_VOCAB_SIZE,
    assistant_ids,
    build_vocab_mapping,
    coverage,
    save_vocab_mapping,
)


def domain_coverage(t2d: torch.Tensor, per_domain_ids: dict) -> dict:
    return {d: round(coverage(t2d, torch.cat(ids)), 4) for d, ids in per_domain_ids.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--stream", default="mixed")
    p.add_argument("--model", default=TOKENIZER)
    p.add_argument("--draft-vocab-size", type=int, default=DRAFT_VOCAB_SIZE)
    p.add_argument("--out", default=DEFAULT_VOCAB)
    p.add_argument("--compare", action="store_true", help="also build chat-only and report both coverages")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    target_vocab_size = AutoConfig.from_pretrained(args.model).vocab_size
    rows = read_jsonl(Path(args.data_dir) / "streams" / f"{args.stream}.jsonl")
    per_domain = {d: [r["messages"] for r in rows if r["domain"] == d] for d in DOMAINS}
    per_domain_ids = {d: assistant_ids(tokenizer, convs) for d, convs in per_domain.items()}

    d2t, t2d = build_vocab_mapping(tokenizer, [r["messages"] for r in rows], target_vocab_size, args.draft_vocab_size)
    report = {"stream": args.stream, "rows": len(rows), "kept_tokens": int(t2d.sum()),
              "coverage": domain_coverage(t2d, per_domain_ids)}
    if args.compare:
        _, t2d_c = build_vocab_mapping(tokenizer, per_domain["chat"], target_vocab_size, args.draft_vocab_size)
        report["coverage_chat_only"] = domain_coverage(t2d_c, per_domain_ids)
    path = save_vocab_mapping(args.out, d2t, t2d, source=[f"{args.data_dir}/streams/{args.stream}.jsonl"],
                model=args.model, conversations=len(rows), coverage=report["coverage"])
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved d2t {tuple(d2t.shape)}, t2d {tuple(t2d.shape)} to {path}")


if __name__ == "__main__":
    main()
