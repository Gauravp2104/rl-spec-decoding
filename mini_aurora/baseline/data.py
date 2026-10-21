"""Dataset and collator over a feature dump (``mini_aurora.baseline.dump``).

Batches are right-padded with zeros, as Aurora's ``DataCollatorWithPadding`` does, and returned as an
``Eagle3TargetOutput`` plus an attention mask so they feed ``mini_aurora.models.draft.forward_step`` unchanged.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from aurora.models.target.eagle3_target_model import Eagle3TargetOutput  # noqa: I001


class DumpDataset(Dataset):
    def __init__(self, split_dir: str | Path, limit: int | None = None):
        self.paths = sorted(Path(split_dir).glob("*.ckpt"))
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no .ckpt files under {split_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        return torch.load(self.paths[i], map_location="cpu", weights_only=False)


def collate(items: list, feature_dtype: torch.dtype = torch.float32) -> tuple[Eagle3TargetOutput, torch.Tensor]:
    """Right-pad to the longest sequence; returns ``(features, attention_mask)`` on CPU."""
    max_len = max(it["input_ids"].numel() for it in items)
    B, aux_dim, hid_dim = len(items), items[0]["aux_hidden_states"].shape[-1], items[0]["hidden_states"].shape[-1]
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    loss_mask = torch.zeros(B, max_len, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    aux = torch.zeros(B, max_len, aux_dim, dtype=feature_dtype)
    last = torch.zeros(B, max_len, hid_dim, dtype=feature_dtype)
    for b, it in enumerate(items):
        T = it["input_ids"].numel()
        input_ids[b, :T] = it["input_ids"]
        loss_mask[b, :T] = it["loss_mask"]
        attention_mask[b, :T] = 1
        aux[b, :T] = it["aux_hidden_states"][0].to(feature_dtype)
        last[b, :T] = it["hidden_states"][0].to(feature_dtype)
    features = Eagle3TargetOutput(hidden_states=aux, input_ids=input_ids, last_hidden_states=last, loss_mask=loss_mask)
    return features, attention_mask


def to_device(features: Eagle3TargetOutput, attention_mask: torch.Tensor, device: torch.device):
    return (
        Eagle3TargetOutput(
            hidden_states=features.hidden_states.to(device),
            input_ids=features.input_ids.to(device),
            last_hidden_states=features.last_hidden_states.to(device),
            loss_mask=features.loss_mask.to(device),
        ),
        attention_mask.to(device),
    )


def make_loader(split_dir: str | Path, batch_size: int, shuffle: bool, seed: int = 0,
                limit: int | None = None, feature_dtype: torch.dtype = torch.float32) -> DataLoader:
    ds = DumpDataset(split_dir, limit)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen,
                      collate_fn=lambda items: collate(items, feature_dtype), num_workers=0)
