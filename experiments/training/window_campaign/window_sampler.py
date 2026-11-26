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
        pos_per_clip: Dict[str, int] = defaultdict(int)
        for i, w in enumerate(self.windows):
            if w.label_state == "ignore" or w.train_label < 0 or w.sample_weight <= 0:
                continue
            self._by_clip[w.clip_uid].append(i)
            if int(w.train_label) == 1:
                self._pos.append(i)
                pos_per_clip[w.clip_uid] += 1
            else:
                self._neg.append(i)
        # El mismo índice puede repetirse hasta el tope por clip (varias pasadas).
        feasible_pos = len(pos_per_clip) * self.max_windows_per_clip
        self._pos_target = min(int(self.num_samples * self.positive_share), max(0, feasible_pos))

    def stats_line(self) -> str:
        return (
            f"sampler: num_samples={self.num_samples} pos_pool={len(self._pos)} "
            f"neg_pool={len(self._neg)} pos_target/epoch={self._pos_target} "
            f"(positive_share={self.positive_share})"
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _draw(
        self,
        rng: random.Random,
        pool: List[int],
        n: int,
        clip_counts: Dict[str, int],
    ) -> List[int]:
        """Toma n índices del pool sin pasarse del tope por clip. No rellena con la otra clase."""
        if n <= 0 or not pool:
            return []
        out: List[int] = []
        stalls = 0
        while len(out) < n and stalls < 4:
            order = pool[:]
            rng.shuffle(order)
            before = len(out)
            for idx in order:
                if len(out) >= n:
                    break
                uid = self.windows[idx].clip_uid
                if clip_counts[uid] >= self.max_windows_per_clip:
                    continue
                clip_counts[uid] += 1
                out.append(idx)
            if len(out) == before:
                stalls += 1
            else:
                stalls = 0
        return out

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        clip_counts: Dict[str, int] = defaultdict(int)
        # Primero los positivos: el tope por clip no se lo comen los negativos.
        pos_drawn = self._draw(rng, self._pos, int(self._pos_target), clip_counts)
        neg_target = max(0, self.num_samples - len(pos_drawn))
        neg_drawn = self._draw(rng, self._neg, neg_target, clip_counts)
        drawn = pos_drawn + neg_drawn
        rng.shuffle(drawn)
        self.last_epoch_pos = len(pos_drawn)
        self.last_epoch_neg = len(neg_drawn)
        for idx in drawn:
            yield idx

    def __len__(self) -> int:
        return self.num_samples