# Concepts

Plain-language explanations of how our scaled-down Aurora works. Code-level details live next to the code
(`mini_aurora/*/README.md`); the plan lives in [`../base_paper/PLAN.md`](../base_paper/PLAN.md).

Figures are [Mermaid](https://mermaid.js.org/) blocks, so they render on GitHub and in VS Code's Markdown preview.

| Doc | Covers |
|---|---|
| [draft_target_interaction.md](draft_target_interaction.md) | What passes between the target and the EAGLE-3 draft: shared weights, aux hidden states, draft → verify rounds, training samples, online weight sync |
| [vocabulary.md](vocabulary.md) | Why a vocabulary is needed, the draft's pruned 24K vocab, `d2t` / `t2d` with a toy example, and how it fits the RL (GRPO) project |
