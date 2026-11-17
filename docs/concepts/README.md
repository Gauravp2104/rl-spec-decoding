# Concepts

| doc | covers |
|---|---|
| [speculative_decoding.md](speculative_decoding.md) | Draft ↔ target (EAGLE-3, aux hidden states, 24K draft vocab), verifying a round (greedy, sampling, KV cache), tree drafting, metrics and the speedup break-even, results so far, offline / online / GRPO training, code map |
| [training_plan.md](training_plan.md) | The plan from here: train and release B1 (1.7B, 26.5K self-distilled), build GRPO on sampled trees, compare static / online forward-KL / online GRPO for τ̄ and speedup |

Figures are [Mermaid](https://mermaid.js.org/) blocks; they render on GitHub and in VS Code's Markdown preview.
