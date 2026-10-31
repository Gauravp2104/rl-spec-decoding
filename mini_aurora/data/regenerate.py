"""PT-01: self-distilled chat data, every assistant turn rewritten by the target itself.

    python -m mini_aurora.data.regenerate --limit 3                       # smoke test (default target)
    MINI_AURORA_TARGET=Qwen/Qwen3-1.7B python -m mini_aurora.data.regenerate \\
        --out data/raw/chat_self_1p7b.jsonl                               # all 1000 rows of data/raw/chat.jsonl

User turns are kept; assistant turn i is regenerated **given the user turns and the target's own answers 0..i-1**
(not the ShareGPT ones). Decoding is greedy with ``enable_thinking=False`` in Qwen3's native chat template, the
prompt the target sees at serving time (``mini_aurora.models.target.build_chat_prompt``). The dump re-wraps the
text in Aurora's ``qwen`` template exactly as it does for the ShareGPT rows, so A0 and A1 differ only in the
assistant text.

Row count and order are preserved, so ``dump.split_rows`` holds out the same ``chat-00900``… rows as for A0.
Rows are never dropped; only trailing turn pairs that push a conversation past ``MAX_TOKENS`` are cut (the
first pair is always kept). An empty answer is kept as-is, since that is what the target produced.

Work is done in shards of ``--shard-size`` conversations (``<out>.shards/``) and is resumable: finished shards
are skipped. Within a shard, conversations are batched at the same turn index (left padding, sorted by length).
Outputs: ``<out>`` (same schema as ``chat.jsonl`` plus a ``regen`` field) and ``<out minus .jsonl>.stats.json``.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mini_aurora.data import MAX_TOKENS
from mini_aurora.data.download import read_jsonl, write_jsonl

DEFAULT_JSONL = "data/raw/chat.jsonl"
MAX_NEW_TOKENS = 512
SHARD_SIZE = 100
BATCH_SIZE = 8
END_OF_TURN = 2  # "<|im_end|>\n" closes every assistant turn in the template


@dataclass
class Reply:
    text: str
    new_tokens: int  # excluding the end-of-turn token
    hit_cap: bool  # stopped by max_new_tokens, not by end-of-turn
    looped: bool


def is_looped(ids: list, n: int = 4, min_len: int = 32, threshold: float = 0.5) -> bool:
    """Degenerate repetition: more than ``threshold`` of the token n-grams repeat an earlier one."""
    if len(ids) < min_len:
        return False
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / len(grams) > threshold


def user_turns(messages: list) -> list:
    return [m["content"] for m in messages if m["role"] == "user"]


def regenerate_rows(
    rows: list,
    generate: Callable[[list], list],
    prompt_len: Callable[[list], int],
    max_tokens: int = MAX_TOKENS,
) -> list:
    """Regenerate every assistant turn of ``rows``, turn index by turn index.

    ``generate(histories) -> [Reply]`` answers a batch of chat histories that each end on a user turn;
    ``prompt_len(history) -> int`` is the token length of a history plus the generation prompt.
    """
    users = [user_turns(r["messages"]) for r in rows]
    history = [[] for _ in rows]
    meta = [{"turns": [], "dropped_pairs": 0} for _ in rows]
    n_tokens = [0] * len(rows)
    done = [False] * len(rows)
    for t in range(max((len(u) for u in users), default=0)):
        active, prompts = [], []
        for i, u in enumerate(users):
            if done[i] or t >= len(u):
                continue
            h = history[i] + [{"role": "user", "content": u[t]}]
            if t > 0 and prompt_len(h) + END_OF_TURN > max_tokens:  # no room for an answer
                done[i] = True
                continue
            active.append(i)
            prompts.append(h)
        if not active:
            break
        for i, h, reply in zip(active, prompts, generate(prompts)):
            total = prompt_len(h) + reply.new_tokens + END_OF_TURN
            if t > 0 and total > max_tokens:  # answer overflowed: cut this pair, stop the conversation
                done[i] = True
                continue
            history[i] = h + [{"role": "assistant", "content": reply.text}]
            n_tokens[i] = total
            meta[i]["turns"].append({"new_tokens": reply.new_tokens, "hit_cap": reply.hit_cap, "looped": reply.looped})
            if total >= max_tokens:
                done[i] = True
    out = []
    for r, h, m, n, u in zip(rows, history, meta, n_tokens, users):
        m["dropped_pairs"] = len(u) - len(m["turns"])
        out.append({"id": r["id"], "domain": r["domain"], "source": r["source"], "src_id": r["src_id"],
                    "messages": h, "n_tokens": n, "regen": m})
    return out


class TargetGenerator:
    """Batched greedy generation with the target's own chat template (``enable_thinking=False``)."""

    def __init__(self, model_path: str, device, dtype, max_new_tokens: int = MAX_NEW_TOKENS,
                 batch_size: int = BATCH_SIZE):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa")
        self.model.to(device).eval().requires_grad_(False)
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        self.stop_ids = {self.tokenizer.convert_tokens_to_ids("<|im_end|>"), self.tokenizer.eos_token_id, self.pad_id}

    def prompt_ids(self, history: list) -> list:
        return self.tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False)

    def prompt_len(self, history: list) -> int:
        return len(self.prompt_ids(history))

    def __call__(self, histories: list) -> list:
        torch = self.torch
        encoded = [self.prompt_ids(h) for h in histories]
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
        replies = [None] * len(encoded)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            width = max(len(encoded[i]) for i in idx)
            input_ids = torch.full((len(idx), width), self.pad_id, dtype=torch.long)
            attention = torch.zeros((len(idx), width), dtype=torch.long)
            for b, i in enumerate(idx):  # left padding
                input_ids[b, width - len(encoded[i]) :] = torch.tensor(encoded[i])
                attention[b, width - len(encoded[i]) :] = 1
            with torch.no_grad():
                out = self.model.generate(input_ids=input_ids.to(self.device), attention_mask=attention.to(self.device),
                                          max_new_tokens=self.max_new_tokens, do_sample=False, top_p=None, top_k=None,
                                          temperature=None, pad_token_id=self.pad_id)
            for b, i in enumerate(idx):
                new = out[b, width:].tolist()
                stop = next((k for k, tok in enumerate(new) if tok in self.stop_ids), None)
                ids = new if stop is None else new[:stop]
                replies[i] = Reply(text=self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
                                   new_tokens=len(ids), hit_cap=stop is None, looped=is_looped(ids))
        return replies


def summarize(orig: list, new: list, count_tokens: Callable[[str], int]) -> dict:
    """Generation statistics for the PT-01 report (``count_tokens`` counts the tokens of one answer)."""
    turns = [t for r in new for t in r["regen"]["turns"]]
    answers = [m["content"] for r in new for m in r["messages"] if m["role"] == "assistant"]
    orig_answers = [m["content"] for r in orig for m in r["messages"] if m["role"] == "assistant"]
    n = max(len(turns), 1)
    return {
        "conversations": len(new),
        "turns_original": len(orig_answers),
        "turns_generated": len(turns),
        "dropped_pairs": sum(r["regen"]["dropped_pairs"] for r in new),
        "conversations_truncated": sum(r["regen"]["dropped_pairs"] > 0 for r in new),
        "hit_cap_rate": round(sum(t["hit_cap"] for t in turns) / n, 4),
        "looped_rate": round(sum(t["looped"] for t in turns) / n, 4),
        "empty_answers": sum(not a for a in answers),
        "duplicate_answers": len(answers) - len(set(answers)),
        "assistant_tokens_original": sum(count_tokens(a) for a in orig_answers),
        "assistant_tokens_generated": sum(t["new_tokens"] for t in turns),
        "mean_answer_tokens_original": round(sum(count_tokens(a) for a in orig_answers) / max(len(orig_answers), 1), 1),
        "mean_answer_tokens_generated": round(sum(t["new_tokens"] for t in turns) / n, 1),
    }


def shard_dir_for(out: Path) -> Path:
    return out.with_name(out.stem + ".shards")


def run(jsonl: str, out: str, model_path: str, device, dtype, max_new_tokens: int = MAX_NEW_TOKENS,
        max_tokens: int = MAX_TOKENS, batch_size: int = BATCH_SIZE, shard_size: int = SHARD_SIZE,
        limit: int | None = None) -> dict:
    rows = read_jsonl(Path(jsonl))[:limit] if limit else read_jsonl(Path(jsonl))
    out = Path(out)
    shard_dir = shard_dir_for(out)
    shard_dir.mkdir(parents=True, exist_ok=True)
    config = {"source": jsonl, "model": model_path, "dtype": str(dtype).split(".")[-1], "decoding": "greedy",
              "enable_thinking": False, "max_new_tokens": max_new_tokens, "max_tokens": max_tokens,
              "shard_size": shard_size, "limit": limit}
    cfg_path = shard_dir / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != config:
        raise SystemExit(f"{shard_dir} was produced with a different config; delete it or change --out")
    cfg_path.write_text(json.dumps(config, indent=2))

    generator, seconds = None, 0.0
    for start in range(0, len(rows), shard_size):
        path = shard_dir / f"shard_{start:05d}.jsonl"
        timing = path.with_suffix(".json")
        if path.exists() and timing.exists():
            seconds += json.loads(timing.read_text())["seconds"]
            continue
        if generator is None:
            generator = TargetGenerator(model_path, device, dtype, max_new_tokens, batch_size)
        t0 = time.perf_counter()
        shard = regenerate_rows(rows[start : start + shard_size], generator, generator.prompt_len, max_tokens)
        dt = time.perf_counter() - t0
        tmp = path.with_suffix(".tmp")
        write_jsonl(tmp, shard)
        tmp.rename(path)  # a shard counts as finished only once fully written
        timing.write_text(json.dumps({"seconds": round(dt, 1)}))
        seconds += dt
        print(f"{path.name}: {len(shard)} conversations, {dt:.0f}s", flush=True)

    new = [r for start in range(0, len(rows), shard_size)
           for r in read_jsonl(shard_dir / f"shard_{start:05d}.jsonl")]
    assert [r["id"] for r in new] == [r["id"] for r in rows], "row order changed; the held-out split would shift"
    write_jsonl(out, new)

    if generator is not None:
        tokenizer = generator.tokenizer
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    stats = summarize(rows, new, lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"]))
    stats["seconds"] = round(seconds, 1)
    stats["config"] = config
    out.with_name(out.stem + ".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    from mini_aurora.device import get_device
    from mini_aurora.models.target import DEFAULT_TARGET

    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", default=DEFAULT_JSONL)
    p.add_argument("--out", default=None, help="default: data/raw/chat_self_<model>.jsonl")
    p.add_argument("--model", default=DEFAULT_TARGET)
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="conversation length cap (tokens)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    p.add_argument("--dtype", default="float32", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    args = p.parse_args()
    out = args.out or f"data/raw/chat_self_{args.model.split('/')[-1].lower().replace('qwen3-', '').replace('.', 'p')}.jsonl"
    run(args.jsonl, out, args.model, get_device(args.device), getattr(torch, args.dtype), args.max_new_tokens,
        args.max_tokens, args.batch_size, args.shard_size, args.limit)


if __name__ == "__main__":
    main()
