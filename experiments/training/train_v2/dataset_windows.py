"""PyTorch dataset sobre mmap Campaign2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import random

from train_v2.augment import augment_pose_window
from train_v2.features import PREPROCESS_TORSO, window_to_features
from train_v2.window_index import WindowRef, build_window_index


class WindowManifestDataset(Dataset):
    def __init__(
        self,
        manifest: Dict[str, Any],
        window_refs: List[WindowRef],
        *,
        window_frames: int,
        seq_len: int,
        min_valid_joint_ratio: float = 0.25,
        train_augment: bool = False,
        augment_cfg: Dict[str, Any] | None = None,
        normalize_mode: str = PREPROCESS_TORSO,
        augment_seed: int = 0,
    ) -> None:
        self.manifest = manifest
        self.refs = window_refs
        self.window_frames = window_frames
        self.seq_len = seq_len
        self.min_valid_joint_ratio = min_valid_joint_ratio
        self.train_augment = train_augment
        self.augment_cfg = dict(augment_cfg or {})
        self.normalize_mode = normalize_mode
        self._rng = random.Random(int(augment_seed))

        poses_path = Path(manifest["poses_mmap"])
        valid_path = Path(manifest["valid_mmap"])
        tf = int(manifest["total_frames"])
        nj = int(manifest["num_joints"])
        self.poses_mm = np.memmap(poses_path, dtype=np.float32, mode="r", shape=(tf, nj, 2))
        self.valid_mm = np.memmap(valid_path, dtype=np.uint8, mode="r", shape=(tf, nj))
        self.entries = manifest["entries"]

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = self.refs[idx]
        ent = self.entries[ref.entry_index]
        offset = int(ent["frame_offset"])
        t = int(ent["num_frames"])
        start = ref.start_frame
        end = min(start + self.window_frames, t)
        poses = np.array(self.poses_mm[offset + start : offset + end], copy=True)
        valid = np.array(self.valid_mm[offset + start : offset + end], copy=True)
        if poses.shape[0] < self.window_frames:
            pad = self.window_frames - poses.shape[0]
            poses = np.concatenate([poses, np.repeat(poses[-1:], pad, axis=0)], axis=0)
            valid = np.concatenate([valid, np.repeat(valid[-1:], pad, axis=0)], axis=0)

        if self.train_augment:
            poses, valid = augment_pose_window(poses, valid, self._rng, self.augment_cfg)

        x, fmask, ok = window_to_features(
            poses,
            valid,
            seq_len=self.seq_len,
            min_valid_joint_ratio=self.min_valid_joint_ratio,
            normalize_mode=self.normalize_mode,
        )
        y = float(ref.label_theft)
        if not ok:
            fmask *= 0.0
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
            torch.from_numpy(fmask),
            ref.uid,
            float(ref.clip_is_robbery),
        )


def collate_window_batch(batch):
    xs, ys, ms, uids, crob = zip(*batch)
    return (
        torch.stack(xs, dim=0),
        torch.stack(ys, dim=0),
        torch.stack(ms, dim=0),
        list(uids),
        torch.tensor(crob, dtype=torch.float32),
    )


def load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def manifest_entries_for_split(manifest: Dict[str, Any], split_name: str) -> List[Dict[str, Any]]:
    return [e for e in (manifest.get("entries") or []) if str(e.get("split")) == split_name]


def make_datasets(
    manifest_path: Path,
    *,
    train_split: str = "train",
    val_split: str = "val_dev",
    temporal: Dict[str, Any],
    robbery_class: int = 6,
    max_windows_per_clip: int = 32,
    min_valid_joint_ratio: float = 0.25,
    train_augment: bool = True,
    augment_cfg: Dict[str, Any] | None = None,
    normalize_mode: str = PREPROCESS_TORSO,
    augment_seed: int = 42,
    window_label_policy: str = "tail_positive",
    robbery_positive_tail_seconds: float = 3.0,
) -> Tuple[WindowManifestDataset, WindowManifestDataset, Dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    hz = float(temporal.get("target_hz", 12.0))
    window_frames = int(round(float(temporal.get("window_seconds", 8.0)) * hz))
    stride_frames = max(1, int(round(float(temporal.get("stride_seconds", 1.0)) * hz)))
    seq_len = int(temporal.get("seq_len", 96))
    target_hz = float(temporal.get("target_hz", manifest.get("target_hz", 12.0)))
    entries = manifest.get("entries") or []

    train_refs = build_window_index(
        entries,
        split=train_split,
        window_frames=window_frames,
        stride_frames=stride_frames,
        robbery_class=robbery_class,
        max_windows_per_clip=max_windows_per_clip,
        target_hz=target_hz,
        window_label_policy=window_label_policy,
        robbery_positive_tail_seconds=robbery_positive_tail_seconds,
    )
    val_refs = build_window_index(
        entries,
        split=val_split,
        window_frames=window_frames,
        stride_frames=stride_frames,
        robbery_class=robbery_class,
        max_windows_per_clip=0,
        target_hz=target_hz,
        window_label_policy=window_label_policy,
        robbery_positive_tail_seconds=robbery_positive_tail_seconds,
    )
    train_ds = WindowManifestDataset(
        manifest,
        train_refs,
        window_frames=window_frames,
        seq_len=seq_len,
        min_valid_joint_ratio=min_valid_joint_ratio,
        train_augment=train_augment,
        augment_cfg=augment_cfg,
        normalize_mode=normalize_mode,
        augment_seed=augment_seed,
    )
    val_ds = WindowManifestDataset(
        manifest,
        val_refs,
        window_frames=window_frames,
        seq_len=seq_len,
        min_valid_joint_ratio=min_valid_joint_ratio,
        train_augment=False,
        normalize_mode=normalize_mode,
    )
    meta = {
        "train_windows": len(train_refs),
        "val_windows": len(val_refs),
        "input_dim": int(manifest.get("num_joints", 8)) * 5,
        "seq_len": seq_len,
        "window_frames": window_frames,
        "window_label_policy": window_label_policy,
        "robbery_positive_tail_seconds": robbery_positive_tail_seconds,
        "target_hz": target_hz,
    }
    return train_ds, val_ds, meta
