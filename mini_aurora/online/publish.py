"""Versioned weight publish (BP-06 step 3): Aurora's shared-FS weight sync, as one in-memory pending slot.

Every N optimizer steps the trainer snapshots ``trainable_state(draft)`` (a copy, so later steps don't leak into
it) and publishes it with its step count as the version. The slot holds at most one snapshot: a newer publish
replaces one the server has not applied yet, since only the latest weights matter. Before each request the
serving side calls ``apply(drafter)``, which hot-swaps the pending weights in with ``Drafter.load_weights``
(Aurora's ``update_weights_from_disk``); the swap happens between requests, never inside one.
"""

import threading
from dataclasses import dataclass

import torch

from mini_aurora.server.drafter import Drafter, trainable_state


@dataclass
class Snapshot:
    state: dict[str, torch.Tensor]
    version: int
    d2t: torch.Tensor
    t2d: torch.Tensor


def snapshot(draft, version: int) -> Snapshot:
    """Copy of a draft's trainable weights plus its vocab mapping (checked by ``load_weights``)."""
    state = {name: p.clone() for name, p in trainable_state(draft).items()}
    return Snapshot(state, version, draft.d2t, draft.t2d)


class WeightSlot:
    """Single pending-weights slot shared by the trainer (``publish``) and the server (``apply``)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Snapshot | None = None
        self.published_version = -1  # latest version published (applied or not)
        self.publishes = 0
        self.replaced = 0  # snapshots overwritten before the server applied them

    def publish(self, snap: Snapshot) -> None:
        with self._lock:
            if self._pending is not None:
                self.replaced += 1
            self._pending = snap
            self.published_version = snap.version
            self.publishes += 1

    def apply(self, drafter: Drafter) -> int | None:
        """Hot-swap the pending snapshot into ``drafter``; returns its version, or None if nothing was pending."""
        with self._lock:
            snap, self._pending = self._pending, None
        if snap is None:
            return None
        drafter.load_weights(snap.state, version=snap.version, d2t=snap.d2t, t2d=snap.t2d)
        return snap.version
