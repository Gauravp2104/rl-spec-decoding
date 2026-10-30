"""Tree drafting (top-k > 1): the verify-side pieces that the chain loop does not need.

A chain proposes one token per step, so a miss at guess i throws away guesses i+1..K. A tree keeps the draft's
top-k tokens at each step (EAGLE-2 style, pruned to ``beam`` nodes per depth by cumulative log-probability), so a
round proposes several alternatives for the same position and the target verifies all of them in one pass.

Layout of one round: the verify input is ``[x_n, node_1, ..., node_T]`` with the nodes in breadth-first order;
``parent[j]`` is the verify-sequence index of node j's parent (0 = the root x_n). A node attends to the whole
context, the root, and its own ancestors (``tree_attention_mask``); its RoPE position is ``n + depth``
(``tree_position_ids``). Greedy verification (``walk_greedy``) descends from the root, at each node following the
child whose token equals the target's argmax there, and stops at the first miss: that path is the accepted
prefix, and the target's argmax at the last accepted node is the extra token. ``cache_keep`` then drops every
other node's slot from the target's KV cache, the tree analogue of ``DynamicCache.crop``.

With top-k 1 (beam 1) the tree is a chain and the round is token-for-token the chain round.
"""

from dataclasses import dataclass

import torch
from transformers import Cache


@dataclass
class DraftTree:
    tokens: torch.Tensor  # (T,) node tokens as target IDs, breadth-first
    draft_ids: torch.Tensor  # (T,) the same as draft-vocab IDs
    parent: torch.Tensor  # (T,) verify-sequence index of each node's parent: 0 = root x_n, j + 1 = node j
    depth: torch.Tensor  # (T,) 1..K
    score: torch.Tensor  # (T,) cumulative draft log-probability along the path

    @property
    def size(self) -> int:
        return int(self.tokens.numel())


@dataclass
class TreeVerdict:
    accepted: int  # a: nodes on the accepted path (0..K)
    keep: list[int]  # verify-sequence indices to commit: [0 (root), accepted nodes...]
    tokens: torch.Tensor  # (a + 1,) target IDs: the path's tokens, then the target's own token


def ancestor_matrix(parent: torch.Tensor) -> torch.Tensor:
    """(T + 1, T + 1) bool: ``anc[i, j]`` iff verify position j is i itself or one of its ancestors."""
    size = parent.numel() + 1
    anc = torch.eye(size, dtype=torch.bool, device=parent.device)
    for j in range(parent.numel()):  # breadth-first order: a parent's row is final before its children's
        anc[j + 1] |= anc[int(parent[j])]
    return anc


def tree_attention_mask(parent: torch.Tensor, past: int, dtype: torch.dtype, prefix_rows: bool = False) -> torch.Tensor:
    """Additive 4D mask for the target's verify pass over ``[x_n, nodes]`` after ``past`` context tokens.

    Shape (1, 1, T + 1, past + T + 1): 0 where a query may attend, the dtype's minimum elsewhere. Every tree
    position attends to all ``past`` context slots. With ``prefix_rows`` (cache-free verify, where the context is
    re-fed too) the context rows are prepended as an ordinary causal block, giving (1, 1, past + T + 1, past + T + 1).
    """
    anc = ancestor_matrix(parent)
    size = anc.shape[0]
    allowed = torch.cat([torch.ones(size, past, dtype=torch.bool, device=parent.device), anc], dim=1)
    if prefix_rows:
        causal = torch.ones(past, past + size, dtype=torch.bool, device=parent.device).tril(0)
        allowed = torch.cat([causal, allowed], dim=0)
    mask = torch.zeros(allowed.shape, dtype=dtype, device=parent.device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None]


def tree_position_ids(depth: torch.Tensor, n: int) -> torch.Tensor:
    """(1, T + 1) RoPE positions: the root x_n sits at ``n``, a node at depth d at ``n + d``."""
    return torch.cat([torch.zeros(1, dtype=torch.long, device=depth.device), depth.long()])[None] + n


def walk_greedy(logits: torch.Tensor, tree: DraftTree) -> TreeVerdict:
    """``logits`` (T + 1, V_target), one row per verify position. Follow the target's argmax down the tree."""
    choice = logits.argmax(-1)
    cur, keep, tokens = 0, [0], []
    while True:
        hit = ((tree.parent == cur) & (tree.tokens == choice[cur])).nonzero()
        if hit.numel() == 0:
            break
        j = int(hit[0])
        tokens.append(tree.tokens[j])
        cur = j + 1
        keep.append(cur)
    extra = choice[cur]
    return TreeVerdict(len(tokens), keep, torch.stack([*tokens, extra]))


def cache_keep(cache: Cache, past: int, keep: list[int]) -> None:
    """Keep the first ``past`` slots plus the verify-sequence slots ``keep`` (in order); drop the rest, in place."""
    idx = torch.tensor([past + i for i in keep], dtype=torch.long)
    layers = getattr(cache, "layers", None)
    if layers is not None:  # transformers >= 4.56
        for layer in layers:
            if layer.keys is None:
                continue
            index = torch.cat([torch.arange(past, device=layer.keys.device), idx.to(layer.keys.device)])
            layer.keys = layer.keys.index_select(2, index)
            layer.values = layer.values.index_select(2, index)
    else:  # legacy key_cache / value_cache lists
        for i in range(len(cache.key_cache)):
            keys = cache.key_cache[i]
            index = torch.cat([torch.arange(past, device=keys.device), idx.to(keys.device)])
            cache.key_cache[i] = keys.index_select(2, index)
            cache.value_cache[i] = cache.value_cache[i].index_select(2, index)
