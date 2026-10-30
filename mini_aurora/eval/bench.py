"""Latency benchmark for the estimated speed-up (BP-07 step 1): c = draft-step latency / target-step latency.

    python -m mini_aurora.eval.bench outputs/online/day0_sync_300          # same dtype / K / stream as the run
    python -m mini_aurora.eval.bench outputs/online/day0_sync_300 --prompts 10 --repeats 20

On the first ``--prompts`` requests of the run's stream, with the run's target dtype and K:

- target step: one cached single-token decode pass (``target_forward`` on a ``DynamicCache``), after the prompt;
- draft step: one ``Drafter.propose`` of K guesses at the prompt's context, divided by K. The drafter is
  cache-free, so this grows with the context; measuring at the prompt length matches the start of a request;
- verify pass: the K + 1-token cached target pass of a speculative round (reported, not used in c);
- target-only tok/s: ``reference_generate`` (greedy ``model.generate``) with the run's ``max_new_tokens``.

Draft latency does not depend on the weights, so a day-0 draft is used. Writes ``<run>/eval/bench.json``, which
``mini_aurora.eval.metrics`` picks up. Run it when nothing else is using the GPU.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import DynamicCache

from mini_aurora.device import get_device
from mini_aurora.models.draft import build_draft
from mini_aurora.models.target import _sync, load_target, load_tokenizer
from mini_aurora.models.vocab import DEFAULT_VOCAB, load_vocab_mapping
from mini_aurora.server.drafter import Drafter
from mini_aurora.server.reference import encode_messages, reference_generate
from mini_aurora.server.serve import DEFAULT_STREAM, load_requests
from mini_aurora.server.target_step import target_forward

from mini_aurora.eval.metrics import EVAL_DIR  # noqa: I001


def _timed(fn, device: torch.device, repeats: int) -> float:
    """Median seconds of ``fn()`` over ``repeats`` runs, after one warm-up run."""
    fn()
    times = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


@torch.no_grad()
def bench_prompt(target, drafter: Drafter, ids: torch.Tensor, k: int, repeats: int = 10) -> dict:
    """Target decode-step, draft-step and verify-pass latency (seconds) at the context ``ids`` (1, n)."""
    device = ids.device
    prefill = target_forward(target, ids, DynamicCache(), num_logits=1)
    cache, n = prefill.past_key_values, ids.shape[1]
    token = prefill.logits[0, -1].argmax().view(1, 1)

    def target_step():
        target_forward(target, token, cache, num_logits=1)
        cache.crop(n)

    proposal = drafter.propose(ids, prefill.aux_hidden, token, k=k)
    verify_ids = torch.cat([token, proposal.tokens.view(1, -1)], dim=1)

    def verify():
        target_forward(target, verify_ids, cache, num_logits=k + 1)
        cache.crop(n)

    return {
        "prompt_tokens": n,
        "target_step_s": _timed(target_step, device, repeats),
        "draft_step_s": _timed(lambda: drafter.propose(ids, prefill.aux_hidden, token, k=k), device, repeats) / k,
        "verify_s": _timed(verify, device, repeats),
    }


def run_bench(target, drafter: Drafter, tokenizer, rows: list[dict], k: int, max_new_tokens: int,
              repeats: int = 10) -> dict:
    per_prompt, tokens, seconds = [], 0, 0.0
    for row in rows:
        ids = encode_messages(tokenizer, row["messages"], target.model.device)
        per_prompt.append(bench_prompt(target, drafter, ids, k, repeats))
        ref = reference_generate(target, ids, max_new_tokens)
        tokens, seconds = tokens + ref.new_ids.numel(), seconds + ref.seconds
    mean = {key: statistics.mean(p[key] for p in per_prompt) for key in ("target_step_s", "draft_step_s", "verify_s")}
    return {
        "c": round(mean["draft_step_s"] / mean["target_step_s"], 4),
        "k": k,
        "target_step_ms": round(1000 * mean["target_step_s"], 2),
        "draft_step_ms": round(1000 * mean["draft_step_s"], 2),
        "verify_ms": round(1000 * mean["verify_s"], 2),
        "target_tokens_per_s": round(tokens / seconds, 2),
        "max_new_tokens": max_new_tokens,
        "prompts": len(rows),
        "per_prompt": [{key: round(v, 5) if isinstance(v, float) else v for key, v in p.items()} for p in per_prompt],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="online run directory (its config.json gives dtype, K, stream)")
    p.add_argument("--prompts", type=int, default=5)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    config = json.loads((Path(args.run) / "config.json").read_text())
    device = get_device(args.device)
    dtype = torch.bfloat16 if config.get("bf16") else torch.float32
    k = config.get("k", 3)
    target, tokenizer = load_target(device=device, dtype=dtype), load_tokenizer()
    drafter = Drafter(build_draft(target, load_vocab_mapping(config.get("vocab", DEFAULT_VOCAB)), device=device,
                                  dtype=dtype), k=k)
    rows = load_requests(config.get("stream", DEFAULT_STREAM), args.prompts, config.get("domain"))
    result = run_bench(target, drafter, tokenizer, rows, k, config.get("max_new_tokens", 128), args.repeats)
    result["dtype"] = str(dtype).split(".")[-1]
    out = Path(args.run) / EVAL_DIR / "bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: v for key, v in result.items() if key != "per_prompt"}, indent=2))
    print(f"bench -> {out}")


if __name__ == "__main__":
    main()
