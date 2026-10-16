"""PyTorch dataset sobre mmap Campaign2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from train_v2.features import window_to_features
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
    ) -> None:
        self.manifest = manifest
        self.refs = window_refs
        self.window_frames = window_frames
        self.seq_len = seq_len
        self.min_valid_joint_ratio = min_valid_joint_ratio

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

        x, fmask, ok = window_to_features(
            poses,
            valid,
            seq_len=self.seq_len,
            min_valid_joint_ratio=self.min_valid_joint_ratio,
        )
        y = float(ref.label_theft)
        if not ok:
            fmask *= 0.0
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
            torch.from_numpy(fmask),
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
) -> Tuple[WindowManifestDataset, WindowManifestDataset, Dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    hz = float(temporal.get("target_hz", 12.0))
    window_frames = int(round(float(temporal.get("window_seconds", 8.0)) * hz))
    stride_frames = max(1, int(round(float(temporal.get("stride_seconds", 1.0)) * hz)))
    seq_len = int(temporal.get("seq_len", 96))
    entries = manifest.get("entries") or []

    train_refs = build_window_index(
        entries,
        split=train_split,
        window_frames=window_frames,
        stride_frames=stride_frames,
        robbery_class=robbery_class,
        max_windows_per_clip=max_windows_per_clip,
    )
    val_refs = build_window_index(
        entries,
        split=val_split,
        window_frames=window_frames,
        stride_frames=stride_frames,
        robbery_class=robbery_class,
        max_windows_per_clip=0,
    )
    train_ds = WindowManifestDataset(
        manifest,
        train_refs,
        window_frames=window_frames,
        seq_len=seq_len,
        min_valid_joint_ratio=min_valid_joint_ratio,
    )
    val_ds = WindowManifestDataset(
        manifest,
        val_refs,
        window_frames=window_frames,
        seq_len=seq_len,
        min_valid_joint_ratio=min_valid_joint_ratio,
    )
    meta = {
        "train_windows": len(train_refs),
        "val_windows": len(val_refs),
        "input_dim": int(manifest.get("num_joints", 8)) * 5,
        "seq_len": seq_len,
        "window_frames": window_frames,
    }
    return train_ds, val_ds, meta
