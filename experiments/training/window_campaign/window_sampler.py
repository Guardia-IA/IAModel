"""B3 — hierarchical clip-capped sampler."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Sampler

from window_index import WindowExample


class HierarchicalClipCapSampler(Sampler[int]):
    """
    Sample indices with per-clip cap and positive share target.
    Skips label_state == ignore via weight 0 indices still in set but lower priority.
    """

    def __init__(
        self,
        windows: Sequence[WindowExample],
        *,
        num_samples: int,
        positive_share: float = 0.35,
        max_windows_per_clip: int = 4,
        seed: int = 42,
    ):
        self.windows = list(windows)
        self.num_samples = int(num_samples)
        self.positive_share = float(positive_share)
        self.max_windows_per_clip = int(max_windows_per_clip)
        self.seed = int(seed)
        self._epoch = 0

        self._pos: List[int] = []
        self._neg: List[int] = []
        self._by_clip: Dict[str, List[int]] = defaultdict(list)
        for i, w in enumerate(self.windows):
            if w.label_state == "ignore" or w.train_label < 0 or w.sample_weight <= 0:
                continue
            self._by_clip[w.clip_uid].append(i)
            if int(w.train_label) == 1:
                self._pos.append(i)
            else:
                self._neg.append(i)
        self._pos_target = min(
            int(self.num_samples * self.positive_share),
            max(1, len(self._pos) * self.max_windows_per_clip),
        )

    def stats_line(self) -> str:
        return (
            f"sampler: num_samples={self.num_samples} pos_pool={len(self._pos)} "
            f"neg_pool={len(self._neg)} pos_target/epoch={self._pos_target} "
            f"(positive_share={self.positive_share})"
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        clip_counts: Dict[str, int] = defaultdict(int)
        n_pos_target = int(self._pos_target)
        emitted = 0
        pos_emitted = 0

        def pick_from(pool: List[int]) -> int | None:
            rng.shuffle(pool)
            for idx in pool:
                w = self.windows[idx]
                if clip_counts[w.clip_uid] >= self.max_windows_per_clip:
                    continue
                clip_counts[w.clip_uid] += 1
                return idx
            return None

        while emitted < self.num_samples:
            want_pos = pos_emitted < n_pos_target and self._pos
            pool = self._pos if want_pos else self._neg
            if not pool:
                pool = self._neg or self._pos
            idx = pick_from(pool)
            if idx is None:
                alt = self._neg if pool is self._pos else self._pos
                if alt:
                    idx = pick_from(alt)
                if idx is None:
                    break
            if int(self.windows[idx].train_label) == 1:
                pos_emitted += 1
            emitted += 1
            yield idx

        while emitted < self.num_samples:
            idx = rng.randint(0, len(self.windows) - 1)
            w = self.windows[idx]
            if w.label_state == "ignore" or w.sample_weight <= 0:
                continue
            yield idx
            emitted += 1

    def __len__(self) -> int:
        return self.num_samples