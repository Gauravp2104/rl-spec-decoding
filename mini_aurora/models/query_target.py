"""Load the target on MPS and run a few chat queries; prints answers, speed, memory and aux-feature shapes.

    python -m mini_aurora.models.query_target
    python -m mini_aurora.models.query_target -q "What is 17 * 23?" -q "Write a Python fizzbuzz." --max-new-tokens 200
"""

import argparse
import time

from mini_aurora.device import get_device, memory_allocated_gb
from mini_aurora.models.target import DEFAULT_TARGET, capture_features, generate, load_target, load_tokenizer

DEFAULT_QUESTIONS = [
    "Explain speculative decoding in two sentences.",
    "Write a Python function that checks whether a string is a palindrome.",
    "A train travels 120 km in 1.5 hours. What is its average speed in km/h?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_TARGET)
    parser.add_argument("--device", default=None, help="mps / cpu / cuda (default: auto)")
    parser.add_argument("-q", "--question", action="append", help="repeatable; default: 3 built-in prompts")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode")
    args = parser.parse_args()

    device = get_device(args.device)
    start = time.perf_counter()
    tokenizer = load_tokenizer(args.model)
    target = load_target(args.model, device=device)
    model = target.model
    print(f"Loaded {args.model} on {model.device} ({model.dtype}, attn={model.config._attn_implementation}) "
          f"in {time.perf_counter() - start:.1f}s; weights {memory_allocated_gb(device):.2f} GB")
    print(f"Aux layers: {target.aux_hidden_states_layers} of {model.config.num_hidden_layers}\n")

    for question in args.question or DEFAULT_QUESTIONS:
        result = generate(target, tokenizer, question, args.max_new_tokens, args.thinking)
        print(f"Q: {question}\nA: {result.text.strip()}")
        print(f"   [{result.prompt_tokens} prompt + {result.new_tokens} new tokens, "
              f"{result.seconds:.1f}s, {result.tokens_per_s:.1f} tok/s]\n")

    ids = tokenizer(DEFAULT_QUESTIONS[0], return_tensors="pt").input_ids.to(device)
    features = capture_features(target, ids)
    print(f"EAGLE-3 features for {ids.shape[1]} tokens: hidden_states {tuple(features.hidden_states.shape)} "
          f"(3 x {model.config.hidden_size}), last_hidden_states {tuple(features.last_hidden_states.shape)}")
    print(f"Memory now {memory_allocated_gb(device):.2f} GB allocated")


if __name__ == "__main__":
    main()
