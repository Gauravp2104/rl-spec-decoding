"""Packed feature-dump shards: a few large files per split instead of one ``.ckpt`` per conversation.

A 27K-conversation dump in the per-conversation layout is ~27K files, which a file-count quota (Zaratan's class
scratch) may not allow. ``dump --shard-size N`` packs N conversations per shard as two files:

| file | content |
|---|---|
| ``shard_<k>.bin`` | raw feature rows, one per token: ``[aux_hidden_states (3H) | hidden_states (H)]`` in the dump dtype |
| ``shard_<k>.idx.pt`` | conversation ids, token offsets, concatenated ``input_ids`` / ``loss_mask``, dims, dtype, counts |

The index is written last (atomically), so a shard without one is incomplete and is rewritten on resume. Reading
memory-maps the ``.bin`` and returns the same dict as a per-conversation ``.ckpt`` (``mini_aurora.baseline.dump``),
so ``DumpDataset`` gives random access per conversation and training shuffles exactly as before.
"""

import os
from pathlib import Path

import numpy as np
import torch

# raw storage: each float dtype is written as the same-width integer and viewed back on read (numpy has no bf16)
_INT = {torch.bfloat16: (torch.int16, np.int16), torch.float16: (torch.int16, np.int16),
        torch.float32: (torch.int32, np.int32)}
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def shard_paths(split_dir: str | Path, k: int) -> tuple[Path, Path]:
    d = Path(split_dir)
    return d / f"shard_{k:05d}.bin", d / f"shard_{k:05d}.idx.pt"


def complete_shards(split_dir: str | Path) -> list[Path]:
    """Index files of the finished shards, in shard order."""
    return sorted(Path(split_dir).glob("shard_*.idx.pt"))


class ShardWriter:
    """Append conversations (``dump_conversation`` dicts) to one shard; ``close`` writes the index."""

    def __init__(self, split_dir: str | Path, k: int, dtype: torch.dtype):
        if dtype not in _INT:
            raise ValueError(f"unsupported dump dtype {dtype}")
        self.bin_path, self.idx_path = shard_paths(split_dir, k)
        self.dtype, self.int_dtype = dtype, _INT[dtype][0]
        self.f = open(self.bin_path, "wb")
        self.ids, self.input_ids, self.loss_mask, self.offsets = [], [], [], [0]
        self.dims = None
        self.skipped = 0

    def add(self, conv_id: str, item: dict | None) -> None:
        if item is None:
            self.skipped += 1
            return
        aux, last = item["aux_hidden_states"][0], item["hidden_states"][0]
        dims = (aux.shape[-1], last.shape[-1])
        if self.dims is None:
            self.dims = dims
        elif dims != self.dims:
            raise ValueError(f"{conv_id}: feature dims {dims} differ from the shard's {self.dims}")
        rows = torch.cat([aux, last], dim=-1).to(self.dtype).contiguous()
        rows.view(self.int_dtype).numpy().tofile(self.f)
        self.ids.append(conv_id)
        self.input_ids.append(item["input_ids"])
        self.loss_mask.append(item["loss_mask"])
        self.offsets.append(self.offsets[-1] + item["input_ids"].numel())

    def close(self) -> dict:
        self.f.close()
        index = {
            "ids": self.ids,
            "offsets": torch.tensor(self.offsets, dtype=torch.long),
            "input_ids": torch.cat(self.input_ids) if self.input_ids else torch.zeros(0, dtype=torch.long),
            "loss_mask": torch.cat(self.loss_mask) if self.loss_mask else torch.zeros(0, dtype=torch.long),
            "aux_dim": self.dims[0] if self.dims else 0,
            "hid_dim": self.dims[1] if self.dims else 0,
            "dtype": str(self.dtype).split(".")[-1],
            "skipped": self.skipped,
        }
        index["tokens"] = int(index["offsets"][-1])
        index["assistant_tokens"] = int(index["loss_mask"].sum())
        if os.path.getsize(self.bin_path) != index["tokens"] * sum(self.dims or (0, 0)) * self.dtype.itemsize:
            raise RuntimeError(f"{self.bin_path}: size does not match the index")
        tmp = self.idx_path.with_suffix(".tmp")
        torch.save(index, tmp)
        os.replace(tmp, self.idx_path)  # the shard counts as complete only once this exists
        return index


class ShardReader:
    """Random access to the conversations of every complete shard in a split directory."""

    def __init__(self, split_dir: str | Path):
        self.indexes, self.bins = [], []
        for idx_path in complete_shards(split_dir):
            self.indexes.append(torch.load(idx_path, map_location="cpu", weights_only=False))
            self.bins.append(idx_path.with_name(idx_path.name.replace(".idx.pt", ".bin")))
        # same order as the per-conversation layout, which sorts "<id>.ckpt" file names
        entries = [(cid, s, j) for s, ix in enumerate(self.indexes) for j, cid in enumerate(ix["ids"])]
        self.entries = sorted(entries, key=lambda e: f"{e[0]}.ckpt")
        self._maps = {}  # opened lazily, so the reader pickles cheaply into DataLoader workers

    def __len__(self) -> int:
        return len(self.entries)

    def _map(self, s: int) -> np.memmap:
        if s not in self._maps:
            ix = self.indexes[s]
            np_int = _INT[_DTYPES[ix["dtype"]]][1]
            self._maps[s] = np.memmap(self.bins[s], dtype=np_int, mode="r",
                                      shape=(ix["tokens"], ix["aux_dim"] + ix["hid_dim"]))
        return self._maps[s]

    def __getitem__(self, i: int) -> dict:
        _, s, j = self.entries[i]
        ix = self.indexes[s]
        a, b = int(ix["offsets"][j]), int(ix["offsets"][j + 1])
        rows = torch.from_numpy(np.array(self._map(s)[a:b])).view(_DTYPES[ix["dtype"]])
        aux_dim = ix["aux_dim"]
        return {
            "input_ids": ix["input_ids"][a:b].clone(),
            "loss_mask": ix["loss_mask"][a:b].clone(),
            "aux_hidden_states": rows[:, :aux_dim].unsqueeze(0).contiguous(),
            "hidden_states": rows[:, aux_dim:].unsqueeze(0).contiguous(),
        }
