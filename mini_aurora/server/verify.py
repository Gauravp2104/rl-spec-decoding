"""Verification: keep the longest correct prefix of the K guesses, then add one token from the target.

The target scores ``[x_n, g_1, ..., g_K]`` in one pass. Row i of its logits is its distribution for the token
after position i, so row i checks guess g_{i+1}, and row a (the first rejected guess, or K if all pass) gives
the extra token. Each round therefore emits a + 1 tokens for one target pass.

- greedy: g_{i+1} is kept iff it equals the target's argmax; the extra token is the argmax of row a.
- sampling (speculative sampling): g_{i+1} is kept with probability min(1, p_i(g) / q_i(g)); on a reject the
  extra token is drawn from norm(max(0, p_i - q_i)), and if all K pass, from p_K. Every emitted token is then
  distributed exactly as sampling from the target at the same temperature.
"""

from dataclasses import dataclass

import torch


@dataclass
class Verdict:
    accepted: int  # a: guesses kept (0..K)
    tokens: torch.Tensor  # (a + 1,) target IDs: g_1..g_a, then the target's own token


def verify_greedy(logits: torch.Tensor, guesses: torch.Tensor) -> Verdict:
    """``logits`` (K + 1, V_target) from the verify pass, ``guesses`` (K,) target IDs."""
    choice = logits.argmax(-1)
    a = int((choice[:-1] == guesses).long().cumprod(0).sum())
    return Verdict(a, torch.cat([guesses[:a], choice[a : a + 1]]))


def verify_sampling(
    logits: torch.Tensor,
    guesses: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_to_target: torch.Tensor,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> Verdict:
    """Speculative-sampling acceptance.

    ``draft_probs`` (K, V_draft) are the distributions q_i the guesses were *sampled* from (not argmax'd), and
    ``draft_to_target`` (V_draft,) maps draft IDs to target IDs; q is 0 for target tokens outside the draft vocab.
    """
    k = guesses.numel()
    p = torch.softmax(logits.float() / temperature, dim=-1)  # (K + 1, V_target)
    q = p.new_zeros(k, p.shape[-1])
    q[:, draft_to_target] = draft_probs.float()
    rows = torch.arange(k, device=p.device)
    p_g, q_g = p[rows, guesses], q[rows, guesses]
    u = torch.rand(k, device=p.device, generator=generator)
    a = int((u * q_g < p_g).long().cumprod(0).sum())  # u < p/q, without dividing by q

    if a < k:
        residual = (p[a] - q[a]).clamp_min(0)
        # residual == 0 only if p_a == q_a, where a reject has probability 0; guard against rounding.
        dist = residual if residual.sum() > 0 else p[a]
    else:
        dist = p[k]
    extra = torch.multinomial(dist / dist.sum(), 1, generator=generator)
    return Verdict(a, torch.cat([guesses[:a], extra]))


def sample_token(logits: torch.Tensor, greedy: bool, temperature: float = 1.0, generator=None) -> torch.Tensor:
    """One token (0-d) from ``logits`` (V,): argmax, or a sample at ``temperature``."""
    if greedy:
        return logits.argmax()
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, 1, generator=generator)[0]
