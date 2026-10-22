"""BP-03: vocab mapping from conversations follows Aurora's d2t offset / t2d mask convention."""

import torch

from mini_aurora.models.vocab import coverage, load_vocab_mapping, save_vocab_mapping


def test_toy_mapping_roundtrip(tmp_path):
    # docs/concepts/vocabulary.md toy example: target IDs 0, 1, 3, 5 kept.
    t2d = torch.zeros(10, dtype=torch.bool)
    t2d[[0, 1, 3, 5]] = True
    used = t2d.nonzero().squeeze(-1)
    d2t = used - torch.arange(len(used))
    assert d2t.tolist() == [0, 0, 1, 2]
    assert (torch.arange(4) + d2t).tolist() == [0, 1, 3, 5]

    path = save_vocab_mapping(tmp_path / "vocab.pt", d2t, t2d, source="toy")
    d2t2, t2d2 = load_vocab_mapping(path)
    assert torch.equal(d2t, d2t2) and torch.equal(t2d, t2d2)


def test_coverage_uses_loss_mask():
    t2d = torch.tensor([True, True, False, True])
    ids = torch.tensor([[0, 2, 2, 3]])
    assert coverage(t2d, ids) == 0.5
    assert coverage(t2d, ids, loss_mask=torch.tensor([[1, 0, 0, 1]])) == 1.0
