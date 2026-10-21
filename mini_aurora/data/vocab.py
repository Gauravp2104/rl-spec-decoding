"""Cache the 16K draft vocab mapping from a stream and report per-domain coverage (needs Aurora: torch >= 2.3).

    python -m mini_aurora.data.vocab                       # mixed stream -> outputs/vocab/qwen3_16k_mixed.pt
    python -m mini_aurora.data.vocab --compare             # also build chat-only and report both coverages

Output format is the one ``mini_aurora.models.vocab`` (BP-03) reads: ``{"d2t", "t2d", "meta"}``.
The mapping is built from the **mixed** stream (all three domains): tokens outside the 16K can never be
drafted, so a chat-only mapping would cap acceptance on the code and math segments of the shift stream
(docs/concepts/vocabulary.md). ``--compare`` prints the numbers behind that choice.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

import mini_aurora  # noqa: F401  (env defaults before Aurora imports)

from aurora.data.preprocessing import generate_vocab_mapping, preprocess_conversations  # noqa: I001
from aurora.data.template import TEMPLATE_REGISTRY

from mini_aurora.data import DATA_DIR, DOMAINS
from mini_aurora.data.download import TOKENIZER, read_jsonl

DRAFT_VOCAB_SIZE = 16000
DEFAULT_OUT = "outputs/vocab/qwen3_16k_mixed.pt"


def assistant_ids(tokenizer, conversations: list, max_length: int = 2048) -> list:
    """``input_ids`` restricted to assistant tokens, per conversation (Aurora's qwen template + loss mask)."""
    data = preprocess_conversations(tokenizer, conversations, TEMPLATE_REGISTRY.get("qwen"),
                                    max_length=max_length, use_packed_loss_mask=False)
    return [ids[mask.bool()] for ids, mask in zip(data["input_ids"], data["loss_mask"])]


def build(tokenizer, conversations: list, target_vocab_size: int, draft_vocab_size: int = DRAFT_VOCAB_SIZE):
    prompts = [{"input_ids": ids} for ids in assistant_ids(tokenizer, conversations)]
    return generate_vocab_mapping(prompts, target_vocab_size, draft_vocab_size)


def coverage(t2d: torch.Tensor, ids_list: list) -> float:
    """Fraction of assistant tokens the draft vocab can produce (same definition as BP-03's ``coverage``)."""
    ids = torch.cat(ids_list)
    return t2d[ids].float().mean().item() if ids.numel() else 0.0


def save(path: str | Path, d2t: torch.Tensor, t2d: torch.Tensor, **meta) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"d2t": d2t.cpu(), "t2d": t2d.cpu(), "meta": meta}, path)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--stream", default="mixed")
    p.add_argument("--model", default=TOKENIZER)
    p.add_argument("--draft-vocab-size", type=int, default=DRAFT_VOCAB_SIZE)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--compare", action="store_true", help="also build chat-only and report both coverages")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    target_vocab_size = AutoConfig.from_pretrained(args.model).vocab_size
    rows = read_jsonl(Path(args.data_dir) / "streams" / f"{args.stream}.jsonl")
    per_domain = {d: [r["messages"] for r in rows if r["domain"] == d] for d in DOMAINS}
    per_domain_ids = {d: assistant_ids(tokenizer, convs) for d, convs in per_domain.items()}

    d2t, t2d = build(tokenizer, [r["messages"] for r in rows], target_vocab_size, args.draft_vocab_size)
    report = {"stream": args.stream, "rows": len(rows), "kept_tokens": int(t2d.sum()),
              "coverage": {d: round(coverage(t2d, ids), 4) for d, ids in per_domain_ids.items()}}
    if args.compare:
        d2t_c, t2d_c = build(tokenizer, per_domain["chat"], target_vocab_size, args.draft_vocab_size)
        report["coverage_chat_only"] = {d: round(coverage(t2d_c, ids), 4) for d, ids in per_domain_ids.items()}
    path = save(args.out, d2t, t2d, source=[f"{args.data_dir}/streams/{args.stream}.jsonl"],
                model=args.model, conversations=len(rows), coverage=report["coverage"])
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved d2t {tuple(d2t.shape)}, t2d {tuple(t2d.shape)} to {path}")


if __name__ == "__main__":
    main()
