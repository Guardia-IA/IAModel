"""Dataset PyTorch: una ventana de 96 frames (sin resize del clip completo)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from train_model_operations import (  # type: ignore
    _apply_deterministic_ops,
    _augment_poses_on_the_fly,
    _stable_str_hash,
    _stable_uid_hash,
    add_velocity,
    normalize_sequence,
    temporal_resize,
    SEED,
)

from window_index import WindowExample
from window_preprocess import add_velocity_v2, extract_window


class WindowPoseDataset(Dataset):
    def __init__(
        self,
        windows: List[WindowExample],
        label_to_idx: Dict[int, int],
        window_len: int = 96,
        pad_mode: str = "repeat_last",
        mask_policy: str = "keep_grid",
        augment_on_the_fly: bool = False,
        augment_prob: float = 0.5,
        augment_max_ops: int = 3,
        augment_op_probs: Optional[Dict[str, float]] = None,
        augment_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        augment_seed: int = SEED,
        dataset_split: str = "train",
        deterministic_variants: Optional[List[Dict[str, Any]]] = None,
        train_deterministic_prob: float = 0.35,
        use_deterministic_in_train: bool = True,
        per_clip_variants: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        self.windows = windows
        self.label_to_idx = label_to_idx
        self.window_len = int(window_len)
        self.pad_mode = pad_mode
        self.mask_policy = mask_policy
        self.augment_on_the_fly = augment_on_the_fly
        self.augment_prob = augment_prob
        self.augment_max_ops = max(1, augment_max_ops)
        self.augment_op_probs = augment_op_probs or {
            "mirror": 0.26,
            "rotate": 0.24,
            "scale": 0.14,
            "shift": 0.14,
            "noise": 0.08,
            "speed": 0.14,
        }
        self.augment_ranges = augment_ranges or {}
        self.augment_seed = int(augment_seed)
        self.dataset_split = dataset_split
        self.deterministic_variants = deterministic_variants or [{"name": "identity", "ops": []}]
        self.train_deterministic_prob = float(train_deterministic_prob)
        self.use_deterministic_in_train = bool(use_deterministic_in_train)
        self.per_clip_variants = per_clip_variants or {}

    def _variants_for_clip(self, clip_uid: str) -> List[Dict[str, Any]]:
        if clip_uid in self.per_clip_variants:
            return self.per_clip_variants[clip_uid]
        return self.deterministic_variants

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        w = self.windows[idx]
        poses_full = np.load(w.pose_path)
        vm = None
        if w.valid_mask_path is not None and w.valid_mask_path.exists():
            vm = np.load(w.valid_mask_path)

        est = w.effective_start if w.effective_start is not None else w.start
        eend = w.effective_end if w.effective_end is not None else w.end

        poses, _, _pad = extract_window(
            poses_full,
            vm,
            est,
            eend,
            window_len=self.window_len,
            pad_mode=self.pad_mode,
            mask_policy=self.mask_policy,
        )

        uid = w.window_id
        forced = w.forced_ops
        worker = get_worker_info()
        base_seed = worker.seed if worker is not None else self.augment_seed
        sample_seed = (int(base_seed) + int(idx) * 1000003) & 0xFFFFFFFF
        rng = np.random.default_rng(sample_seed)

        if forced is not None:
            det_seed = (
                _stable_uid_hash(uid) * 1009 + _stable_str_hash(repr(forced)) + int(idx) * 131
            ) & 0xFFFFFFFF
            rng_det = np.random.default_rng(int(det_seed))
            poses = _apply_deterministic_ops(poses, forced, rng=rng_det)
            if self.dataset_split == "train" and self.augment_on_the_fly and len(forced) == 0:
                poses = _augment_poses_on_the_fly(
                    poses=poses,
                    rng=rng,
                    augment_prob=self.augment_prob,
                    max_ops=self.augment_max_ops,
                    op_probs=self.augment_op_probs,
                    ranges=self.augment_ranges,
                )
        elif self.dataset_split == "train":
            variants = self._variants_for_clip(w.clip_uid)
            n_var = len(variants)
            if self.use_deterministic_in_train and n_var > 0 and rng.random() < self.train_deterministic_prob:
                vidx = int(rng.integers(0, n_var))
                spec = variants[vidx]
                det_seed = (sample_seed ^ (vidx * 7919)) & 0xFFFFFFFF
                rng_det = np.random.default_rng(int(det_seed))
                poses = _apply_deterministic_ops(poses, spec.get("ops", []), rng=rng_det)
            if self.augment_on_the_fly:
                poses = _augment_poses_on_the_fly(
                    poses=poses,
                    rng=rng,
                    augment_prob=self.augment_prob,
                    max_ops=self.augment_max_ops,
                    op_probs=self.augment_op_probs,
                    ranges=self.augment_ranges,
                )

        poses = normalize_sequence(poses)
        if self.mask_policy == "keep_grid_v2":
            poses = add_velocity_v2(poses)
        else:
            poses = add_velocity(poses)
        if poses.shape[0] != self.window_len:
            poses = temporal_resize(poses, self.window_len)
        t, j, d = poses.shape
        x = torch.from_numpy(poses.reshape(t, j * d).astype(np.float32))
        tl = int(w.train_label) if w.train_label >= 0 else 0
        y = self.label_to_idx.get(tl, self.label_to_idx.get(int(w.label), 0))
        sw = float(w.sample_weight) if w.label_state != "ignore" else 0.0
        return x, y, sw