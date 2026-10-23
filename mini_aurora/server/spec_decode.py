"""Speculative decoding loop (bs = 1, chain drafting): prefill, then draft → verify → commit until done.

State between rounds: ``context`` (1, n) = tokens whose target aux states are known (``aux`` (1, n, 3 * hidden)),
and ``next_token`` = the target's latest token x_n, not yet run through the target. A round:

1. the drafter guesses g_1..g_K after x_n from ``context`` / ``aux`` / x_n;
2. the target runs ``[x_n, g_1..g_K]`` once (``num_logits=K + 1``);
3. ``verify_*`` keeps g_1..g_a and adds the target's own token t;
4. x_n, g_1..g_a join ``context`` (their aux states are valid: each depends only on accepted tokens), and t
   becomes the new ``next_token``.

``use_cache=False`` re-runs the target over the whole sequence every round (BP-05 step 5, simplest to trust);
``use_cache=True`` keeps an HF ``DynamicCache`` and crops it back to ``context`` after each round (step 6).
The drafter is cache-free either way.

Each round records the guesses kept and the draft version that proposed them (step 7). With
``return_sample=True`` the output also carries a training sample (step 8), laid out as Aurora's
``generate_with_decode``: prompt + completion minus its last token (never run through the target), aux and
final-norm states for each of those tokens, and a loss mask over the completion. The states are the ones the
verify passes already computed, so a sample costs no extra target pass.
"""

import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from mini_aurora.models.target import _sync
from mini_aurora.server.drafter import Drafter
from mini_aurora.server.reference import eos_token_ids
from mini_aurora.server.target_step import target_forward
from mini_aurora.server.verify import sample_token, verify_greedy, verify_sampling

from aurora.models.target.eagle3_target_model import Eagle3TargetOutput, HFTargetModel  # noqa: I001


@dataclass
class SpecOutput:
    prompt_ids: torch.Tensor  # (1, n)
    new_ids: torch.Tensor  # (m,), includes the EOS token if one was generated
    seconds: float
    accepted: list[int] = field(default_factory=list)  # guesses kept per round (0..K)
    versions: list[int] = field(default_factory=list)  # draft version that proposed each round
    k: int = 3
    sample: Eagle3TargetOutput | None = None  # training sample (return_sample=True)

    @property
    def draft_version(self) -> int:
        return self.versions[-1] if self.versions else -1

    @property
    def tokens_per_s(self) -> float:
        return self.new_ids.numel() / self.seconds if self.seconds > 0 else 0.0

    @property
    def mean_accepted(self) -> float:
        """τ̄: mean accepted guesses per round (each round also emits one target token)."""
        return sum(self.accepted) / len(self.accepted) if self.accepted else 0.0


@torch.no_grad()
def speculative_generate(
    target: HFTargetModel,
    drafter: Drafter,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    k: int | None = None,
    greedy: bool = True,
    temperature: float = 1.0,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
    return_sample: bool = False,
) -> SpecOutput:
    """Generate after ``input_ids`` (1, n). Greedy output equals ``reference_generate`` token for token."""
    k = k or drafter.k
    eos = eos_token_ids(target)
    draft_to_target = drafter.to_target_ids(torch.arange(drafter.draft.lm_head.out_features, device=input_ids.device))
    start = time.perf_counter()

    step = target_forward(target, input_ids, DynamicCache() if use_cache else None, num_logits=1)
    context, aux, last, cache = input_ids, step.aux_hidden, step.last_hidden, step.past_key_values
    next_token = sample_token(step.logits[0, -1], greedy, temperature, generator)
    out, accepted, versions = [int(next_token)], [], []

    while len(out) < max_new_tokens and out[-1] not in eos:
        proposal = drafter.propose(
            context, aux, next_token, k=k, greedy=greedy, temperature=temperature, generator=generator
        )
        verify_ids = torch.cat([next_token.view(1, 1), proposal.tokens.view(1, -1)], dim=1)
        if use_cache:
            step = target_forward(target, verify_ids, cache, num_logits=k + 1)
        else:
            step = target_forward(target, torch.cat([context, verify_ids], dim=1), num_logits=k + 1)
        new_aux, new_last = step.aux_hidden[:, -(k + 1) :], step.last_hidden[:, -(k + 1) :]

        logits = step.logits[0]
        if greedy:
            verdict = verify_greedy(logits, proposal.tokens)
        else:
            verdict = verify_sampling(logits, proposal.tokens, proposal.probs, draft_to_target, temperature, generator)

        a = verdict.accepted
        context = torch.cat([context, verify_ids[:, : a + 1]], dim=1)
        aux = torch.cat([aux, new_aux[:, : a + 1]], dim=1)
        last = torch.cat([last, new_last[:, : a + 1]], dim=1)
        if use_cache:
            cache = step.past_key_values
            cache.crop(context.shape[1])  # drop the rejected guesses (and the extra token's slot)
        next_token = verdict.tokens[-1]
        accepted.append(a)
        versions.append(drafter.version)
        for token in verdict.tokens.tolist():
            out.append(token)
            if token in eos or len(out) == max_new_tokens:
                break

    _sync(input_ids.device)
    new_ids = torch.tensor(out, device=input_ids.device)
    seconds = time.perf_counter() - start
    sample = _training_sample(input_ids, new_ids, aux, last) if return_sample else None
    return SpecOutput(input_ids, new_ids, seconds, accepted, versions, k, sample)


def _training_sample(
    prompt: torch.Tensor, new_ids: torch.Tensor, aux: torch.Tensor, last: torch.Tensor
) -> Eagle3TargetOutput:
    """Prompt + completion[:-1] with their target states; loss on the completion (Aurora's decode samples)."""
    seq = torch.cat([prompt, new_ids[:-1].view(1, -1)], dim=1)
    n = seq.shape[1]  # every token here went through a verify pass, so aux / last cover it
    loss_mask = torch.zeros_like(seq)
    loss_mask[:, prompt.shape[1] :] = 1
    return Eagle3TargetOutput(hidden_states=aux[:, :n], input_ids=seq, last_hidden_states=last[:, :n], loss_mask=loss_mask)
