# Off-Policy GRPO for Online Speculative Decoding

MSAI-635 Group 2 project: Gaurav Prakash, Sanjana Kommareddy, Simran Kharbanda.
Proposal: [docs/MSAI635_project_proposal_Group2.pdf](docs/MSAI635_project_proposal_Group2.pdf).
How it works: [docs/concepts/speculative_decoding.md](docs/concepts/speculative_decoding.md). The plan from here: [docs/concepts/training_plan.md](docs/concepts/training_plan.md).

## What this is

Speculative decoding speeds up LLM inference by letting a small draft model propose a block of tokens that the large target model verifies in one forward pass. The speed-up depends almost entirely on how many drafted tokens the target accepts.

[Aurora](https://arxiv.org/abs/2602.06932) trains an EAGLE-3 drafter *online*, from the hidden states and logits the inference server already produces, and hot-swaps the updated weights back into the server. It frames this as asynchronous reinforcement learning, but the loss it optimizes is still forward-KL distillation.

We first reproduce Aurora at laptop scale, then replace the distillation loss with an off-policy GRPO objective so the drafter optimizes acceptance length directly, using the accepted lengths the verifier already reports as the reward.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 1. Base paper reproduction | Scaled-down Aurora with the original forward-KL loss: HF draft/verify loop, online training, day-0 and domain-shift experiments. Tracking issue [#2](https://github.com/Gauravp2104/rl-spec-decoding/issues/2), plan in [docs/base_paper/PLAN.md](docs/base_paper/PLAN.md), conventions in [docs/base_paper/CONVENTIONS.md](docs/base_paper/CONVENTIONS.md). | Done (BP-01 … BP-09), [project board](https://github.com/users/Gauravp2104/projects/2) |
| 1b. Pre-training | Self-distilled 1.7B drafts: A1 (900 conversations, release `pt01-a1-v1`), B1 (26.5K, PT-02, the GRPO starting checkpoint) and tree serving. Plan in [docs/pre_training/PLAN.md](docs/pre_training/PLAN.md). | B1 job chain ready on Zaratan |
| 2. GRPO extension | Stochastic draft-tree sampling with behavior log-probs, reward and group-normalized advantages, off-policy GRPO loss with the KL term kept as an anchor, staleness filtering. Compared against B1 static and online forward-KL on the same streams ([training_plan.md](docs/concepts/training_plan.md)). | Planned |

## Citation

J. Wang et al. *When RL meets adaptive speculative training: A unified training-serving system.* arXiv:2602.06932, 2026. Upstream code is vendored under `aurora-main/` (its large-model `examples/` are removed; we use only the model, loss and data code).
