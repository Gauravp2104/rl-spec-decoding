"""EAGLE-3 chain drafter (top-1, K steps), using the same backbone calls as training.

Training (``Eagle3Model.forward``) runs ``length`` TTT steps over the whole sequence. At step ``i``,
position t reads the previous step's hidden state (step 0: ``fc(aux_t)``) plus the embedding of token
t+1+i, and predicts token t+2+i. Drafting is that unroll read off at the last position t = n-1:

- step 0: ``fc(aux of x_0..x_{n-1})`` + embed(x_1..x_n), where x_n is the token the target just produced;
  the last position's logits give guess g_1;
- step i: the draft's own hidden states + the ids shifted once more with g_i appended; the last
  position gives g_{i+1}.

The draft attention's TTT cache adds one key per step at the same position (RoPE offset by the step), so
the last position at step i sees exactly what it saw in training. Earlier positions are recomputed and
ignored: cache-free, simple and correct first (BP-05 step 5); a KV cache comes later.

Hot-swap (step 9): ``Drafter.load_weights`` copies a trainer's new weights into the served draft in place and
bumps ``version``; the loop reads the version every round, so logs show which draft proposed each guess
(staleness in BP-06). The served draft is its own copy: later trainer steps don't leak into it.
``Drafter.load_file`` reads BP-04's checkpoint format (``mini_aurora.baseline.checkpoint.save_draft``), the one
format for draft weights on disk; the checkpoint's ``step`` becomes the version.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from mini_aurora.baseline.checkpoint import latest_checkpoint

from aurora.models.draft.base import Eagle3DraftModel  # noqa: I001

VOCAB_BUFFERS = ("d2t", "t2d")


@dataclass
class DraftProposal:
    tokens: torch.Tensor  # (K,) guesses as target IDs
    draft_ids: torch.Tensor  # (K,) the same guesses as draft-vocab IDs
    probs: torch.Tensor  # (K, V_draft) draft distribution q_i each guess was taken from


class Drafter:
    def __init__(self, draft: Eagle3DraftModel, k: int = 3, version: int = 0) -> None:
        if not draft.has_vocab_pruning:
            raise ValueError("Draft vocab buffers are unset; pass vocab_mapping to build_draft")
        self.draft = draft.eval()
        self.k = k
        self.version = version

    @torch.no_grad()
    def load_weights(
        self,
        state: dict[str, torch.Tensor],
        version: int,
        d2t: torch.Tensor | None = None,
        t2d: torch.Tensor | None = None,
    ) -> None:
        """Replace the trainable weights (``fc``, decoder layer, ``norm``, ``lm_head``) with ``state``.

        Aurora's ``update_weights_from_disk``. The frozen embedding and the vocab mapping stay; if the sender's
        ``d2t`` / ``t2d`` are given they must match, since a draft trained on another mapping would emit wrong IDs.
        """
        for name, theirs in (("d2t", d2t), ("t2d", t2d)):
            ours = getattr(self.draft, name)
            if theirs is not None and not torch.equal(theirs.to(ours.device), ours):
                raise ValueError(f"Draft weights were trained with a different vocab mapping ({name} differs)")
        params = dict(self.draft.named_parameters())
        expected = set(trainable_state(self.draft))
        if set(state) != expected:
            raise ValueError(f"Weight names differ: missing {expected - set(state)}, unexpected {set(state) - expected}")
        for name, value in state.items():
            params[name].copy_(value)
        self.version = version

    def load_file(self, path: str | Path) -> Path:
        """Hot-swap from a BP-04 checkpoint file, or a checkpoint directory (its ``latest.txt``).

        The format is ``{"state_dict", "d2t", "t2d", "step", "meta"}`` from ``save_draft``; ``state_dict`` holds
        the trainable weights plus the vocab buffers, which are checked against the served draft, not loaded.
        Returns the file that was loaded.
        """
        path = Path(path)
        if path.is_dir():
            resolved = latest_checkpoint(path)
            if resolved is None:
                raise FileNotFoundError(f"no latest.txt in {path}")
            path = resolved
        ckpt = torch.load(path, map_location=self.draft.d2t.device, weights_only=False)
        state = {k: v for k, v in ckpt["state_dict"].items() if k not in VOCAB_BUFFERS}
        d2t = ckpt.get("d2t", ckpt["state_dict"].get("d2t"))
        t2d = ckpt.get("t2d", ckpt["state_dict"].get("t2d"))
        self.load_weights(state, ckpt.get("version", ckpt["step"]), d2t, t2d)
        return path

    def to_target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Draft-vocab IDs → target IDs (``target_id = draft_id + d2t[draft_id]``)."""
        return draft_ids + self.draft.d2t[draft_ids]

    @torch.no_grad()
    def propose(
        self,
        input_ids: torch.Tensor,
        aux_hidden: torch.Tensor,
        next_token: torch.Tensor | int,
        k: int | None = None,
        greedy: bool = True,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> DraftProposal:
        """Guess the K tokens after ``next_token``.

        ``input_ids`` (1, n) are the tokens so far, ``aux_hidden`` (1, n, 3 * hidden) the target's aux states
        for them, and ``next_token`` the target's token at position n (not yet in ``input_ids``).
        """
        draft, k = self.draft, k or self.k
        n = input_ids.shape[1]
        if aux_hidden.shape[1] != n:
            raise ValueError(f"aux_hidden covers {aux_hidden.shape[1]} tokens, input_ids {n}")
        dtype = next(draft.fc.parameters()).dtype
        next_token = torch.as_tensor(next_token, device=input_ids.device).view(1, 1)

        hidden = draft.project_hidden_states(aux_hidden.to(dtype))
        position_ids = torch.arange(n, device=input_ids.device).unsqueeze(0)
        mask = draft.prepare_decoder_attention_mask(
            attention_mask=torch.ones_like(input_ids),
            hidden_states=hidden,
            batch_size=1,
            seq_length=n,
            past_key_values_length=0,
        )
        ids = torch.cat([input_ids[:, 1:], next_token], dim=1)  # = padding(input_ids, left=False) in training
        cache_keys = cache_values = None
        draft_ids, probs = [], []
        for _ in range(k):
            embeds = draft.embed_input_ids(ids.clamp(0, draft.target_vocab_size - 1)).to(dtype)
            hidden, cache_keys, cache_values = draft.backbone(
                input_embeds=embeds,
                hidden_states=hidden,
                attention_mask=mask,
                position_ids=position_ids,
                cache_keys=cache_keys,
                cache_values=cache_values,
                use_cache=True,
            )
            logits = draft.compute_logits(hidden[:, -1]).float()[0]  # (V_draft,)
            if greedy:
                q = torch.softmax(logits, dim=-1)
                draft_id = q.argmax()
            else:
                q = torch.softmax(logits / temperature, dim=-1)
                draft_id = torch.multinomial(q, 1, generator=generator)[0]
            draft_ids.append(draft_id)
            probs.append(q)
            ids = torch.cat([ids[:, 1:], self.to_target_ids(draft_id).view(1, 1)], dim=1)

        draft_ids = torch.stack(draft_ids)
        return DraftProposal(self.to_target_ids(draft_ids), draft_ids, torch.stack(probs))


def trainable_state(draft: Eagle3DraftModel) -> dict[str, torch.Tensor]:
    """The weights a trainer updates and the server swaps in (not the frozen embedding or the vocab buffers)."""
    return {name: p.detach() for name, p in draft.named_parameters() if p.requires_grad}

