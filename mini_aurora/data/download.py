"""Download the three public datasets and convert them to tagged conversations (CPU, no torch).

    python -m mini_aurora.data.download            # -> data/raw/{chat,code,math}.jsonl + data/raw/stats.json

Public replacements for Aurora's private mixed corpus (PLAN.md BP-02):

| domain | source | HF dataset |
|---|---|---|
| chat | ShareGPT | ``Aeala/ShareGPT_Vicuna_unfiltered`` (``ShareGPT_V4.3_unfiltered_cleaned_split.json``) |
| code | CodeAlpaca | ``sahil2801/CodeAlpaca-20k`` |
| math | GSM8K | ``openai/gsm8k`` (``main`` config, train split) |

Every kept row is ``{"id", "domain", "source", "messages", "n_tokens"}`` where ``messages`` is a list of
``{"role", "content"}`` turns starting with ``user`` and alternating, which is what Aurora's ``GeneralParser``
requires (it truncates anything else). ``n_tokens`` is the length of the Qwen3 chat-templated conversation.

Filters (counts are written to ``stats.json``):
- ``not_user_first``: conversation does not start with a user turn (system turns are dropped first).
- ``bad_alternation``: two consecutive turns with the same role, or an unknown role.
- ``empty``: an empty turn.
- ``non_english``: fewer than 90% of the letters are ASCII (ShareGPT only).
- ``too_long``: more than ``MAX_TOKENS`` tokens after keeping at most ``MAX_TURNS`` turns.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from mini_aurora.data import DATA_DIR, DOMAINS, MAX_TOKENS, MAX_TURNS, PER_DOMAIN, SEED

SHAREGPT_REPO = "Aeala/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V4.3_unfiltered_cleaned_split.json"
CODEALPACA = "sahil2801/CodeAlpaca-20k"
GSM8K = "openai/gsm8k"
TOKENIZER = "Qwen/Qwen3-0.6B"

_ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)


def normalize(turns: list) -> list | None:
    """ShareGPT ``from``/``value`` or ``role``/``content`` -> role/content; drop system turns.

    Returns ``None`` (with the reason in ``normalize.reason``) if the conversation cannot be used.
    """
    out = []
    for t in turns:
        role = _ROLE.get(t.get("from", t.get("role")))
        content = t.get("value", t.get("content"))
        if role is None:
            normalize.reason = "bad_alternation"
            return None
        if role == "system":
            continue
        if not isinstance(content, str) or not content.strip():
            normalize.reason = "empty"
            return None
        out.append({"role": role, "content": content.strip()})
    if not out or out[0]["role"] != "user":
        normalize.reason = "not_user_first"
        return None
    for a, b in zip(out, out[1:]):
        if a["role"] == b["role"]:
            normalize.reason = "bad_alternation"
            return None
    out = out[:MAX_TURNS]
    if out[-1]["role"] == "user":  # end on an assistant turn so the loss mask is non-empty
        out = out[:-1]
    if len(out) < 2:
        normalize.reason = "not_user_first"
        return None
    return out


def is_english(messages: list, min_ascii: float = 0.9) -> bool:
    text = " ".join(m["content"] for m in messages)
    letters = _LETTERS.findall(text)
    if not letters:
        return False
    ascii_letters = sum(1 for ch in letters if ch.isascii())
    return ascii_letters / len(letters) >= min_ascii


def load_sharegpt(cache_dir: str | None = None) -> list:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset", cache_dir=cache_dir)
    with open(path) as f:
        rows = json.load(f)
    return [{"source": "sharegpt", "src_id": r.get("id"), "turns": r["conversations"]} for r in rows]


def load_codealpaca(cache_dir: str | None = None) -> list:
    from datasets import load_dataset

    ds = load_dataset(CODEALPACA, split="train", cache_dir=cache_dir)
    rows = []
    for i, r in enumerate(ds):
        prompt = r["instruction"].strip()
        if r.get("input", "").strip():
            prompt += "\n\n" + r["input"].strip()
        rows.append({"source": "codealpaca", "src_id": i, "turns": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": r["output"]},
        ]})
    return rows


def load_gsm8k(cache_dir: str | None = None) -> list:
    from datasets import load_dataset

    ds = load_dataset(GSM8K, "main", split="train", cache_dir=cache_dir)
    # The answer keeps GSM8K's worked solution and the final "#### <number>" line: that is the text the
    # target is expected to generate, so it belongs in the assistant turn for training and vocab counting.
    return [{"source": "gsm8k", "src_id": i, "turns": [
        {"role": "user", "content": r["question"]},
        {"role": "assistant", "content": r["answer"]},
    ]} for i, r in enumerate(ds)]


LOADERS = {"chat": load_sharegpt, "code": load_codealpaca, "math": load_gsm8k}


def count_tokens(tokenizer, messages: list) -> int:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_domain(domain: str, tokenizer, per_domain: int, seed: int, cache_dir: str | None) -> tuple[list, Counter]:
    """Shuffle the source with ``seed``, filter in order, stop at ``per_domain`` kept rows."""
    rows = LOADERS[domain](cache_dir)
    random.Random(seed).shuffle(rows)
    kept, reasons = [], Counter()
    for r in rows:
        if len(kept) >= per_domain:
            break
        messages = normalize(r["turns"])
        if messages is None:
            reasons[normalize.reason] += 1
            continue
        if domain == "chat" and not is_english(messages):
            reasons["non_english"] += 1
            continue
        n_tokens = count_tokens(tokenizer, messages)
        if n_tokens > MAX_TOKENS:
            reasons["too_long"] += 1
            continue
        kept.append({
            "id": f"{domain}-{len(kept):05d}",
            "domain": domain,
            "source": r["source"],
            "src_id": r["src_id"],
            "messages": messages,
            "n_tokens": n_tokens,
        })
    reasons["kept"] = len(kept)
    reasons["seen"] = sum(reasons.values()) - len(kept)
    return kept, reasons


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def token_stats(rows: list) -> dict:
    n = sorted(r["n_tokens"] for r in rows)
    pick = lambda q: n[min(len(n) - 1, int(q * len(n)))]  # noqa: E731
    return {"min": n[0], "p50": pick(0.5), "p90": pick(0.9), "max": n[-1],
            "mean_turns": round(sum(len(r["messages"]) for r in rows) / len(rows), 2)}


def main() -> None:
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--per-domain", type=int, default=PER_DOMAIN)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    p.add_argument("--hf-cache", default=None, help="HF cache dir (default: ~/.cache/huggingface)")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    raw = Path(args.data_dir) / "raw"
    stats_path = raw / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}  # merge per domain on reruns
    for domain in args.domains:
        rows, reasons = build_domain(domain, tokenizer, args.per_domain, args.seed, args.hf_cache)
        write_jsonl(raw / f"{domain}.jsonl", rows)
        stats[domain] = {"filters": dict(reasons), "tokens": token_stats(rows)}
        print(f"{domain}: kept {len(rows)}  filters={dict(reasons)}  tokens={stats[domain]['tokens']}")
    stats["config"] = {"per_domain": args.per_domain, "seed": args.seed, "max_tokens": MAX_TOKENS,
                       "max_turns": MAX_TURNS, "tokenizer": TOKENIZER}
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
