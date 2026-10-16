"""
Construcción de caché contigua 12 Hz en memoria mapeada (Sprint 1).

No modifica el legacy; solo lee .npy / meta.json existentes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


@dataclass
class CacheBuildResult:
    manifest_path: Path
    poses_mmap_path: Path
    valid_mmap_path: Path
    total_frames: int
    num_clips: int
    bytes_on_disk: int


def _fps_from_meta(meta: Dict[str, Any], default: float = 12.0) -> float:
    for key in ("fps", "pose_fps", "video_fps"):
        v = meta.get(key)
        if v is not None:
            try:
                f = float(v)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                pass
    return default


def _linear_resample_1d(y: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    out = np.full(t_dst.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        if valid.any():
            out[:] = float(y[valid][0])
        return out
    yv = y[valid].astype(np.float64)
    tv = t_src[valid].astype(np.float64)
    order = np.argsort(tv)
    tv = tv[order]
    yv = yv[order]
    for i, td in enumerate(t_dst):
        if td <= tv[0]:
            out[i] = yv[0]
        elif td >= tv[-1]:
            out[i] = yv[-1]
        else:
            out[i] = np.interp(td, tv, yv)
    return out


def resample_poses_to_hz(
    poses: np.ndarray,
    fps: float,
    target_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    poses: [T, J, 2] float, NaN allowed.
    Returns resampled [T2, J, 2] and valid [T2, J] bool (finite xy).
    """
    poses = np.asarray(poses, dtype=np.float32)
    t, j, _ = poses.shape
    if t == 0:
        return poses, np.zeros((0, j), dtype=bool)
    if t == 1 or fps <= 0 or target_hz <= 0:
        valid = np.isfinite(poses).all(axis=-1)
        return poses.copy(), valid

    duration = (t - 1) / float(fps)
    n_out = max(2, int(round(duration * target_hz)) + 1)
    t_src = np.arange(t, dtype=np.float64) / float(fps)
    t_dst = np.linspace(0.0, duration, n_out, dtype=np.float64)

    out = np.full((n_out, j, 2), np.nan, dtype=np.float32)
    for ji in range(j):
        for ci in range(2):
            channel = poses[:, ji, ci]
            out[:, ji, ci] = _linear_resample_1d(channel, t_src, t_dst)
    valid = np.isfinite(out).all(axis=-1)
    return out, valid


def _clip_group_id(clip_dir: Path, meta: Dict[str, Any]) -> str:
    vs = str(meta.get("video_source") or meta.get("video") or "")
    name = clip_dir.name
    raw = f"{vs}|{name}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _assign_split(group_id: str, ratios: Dict[str, float], seed: int) -> str:
    keys = ("train", "val_dev", "threshold_calib", "frozen_test")
    weights = [float(ratios.get(k, 0.0)) for k in keys]
    s = sum(weights) or 1.0
    weights = [w / s for w in weights]
    h = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16)
    x = (h % 10_000) / 10_000.0
    acc = 0.0
    for k, w in zip(keys, weights):
        acc += w
        if x < acc:
            return k
    return keys[-1]


def collect_clip_records(
    data_roots: List[Path],
    *,
    pose_source: str = "full",
    single_user_only: bool = False,
    robbery_class: int = 6,
) -> List[Dict[str, Any]]:
    from train_model_operations import collect_examples, _example_uid  # type: ignore

    records: List[Dict[str, Any]] = []
    for root in data_roots:
        examples = collect_examples(
            pose_source=pose_source,
            single_user_only=single_user_only,
            data_root=str(root),
        )
        for ex in examples:
            clip_dir = Path(ex.pose_path).parent.parent
            meta_path = clip_dir / "meta.json"
            meta: Dict[str, Any] = {}
            if meta_path.is_file():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            records.append(
                {
                    "uid": _example_uid(ex),
                    "label": int(ex.label),
                    "is_robbery": int(ex.label) == int(robbery_class),
                    "pose_path": str(ex.pose_path),
                    "valid_mask_path": str(ex.valid_mask_path) if ex.valid_mask_path else None,
                    "clip_dir": str(clip_dir),
                    "data_root": str(root),
                    "meta": meta,
                    "group_id": _clip_group_id(clip_dir, meta),
                }
            )
    return records


def build_pose_cache(
    records: List[Dict[str, Any]],
    cache_dir: Path,
    *,
    target_hz: float = 12.0,
    split_ratios: Optional[Dict[str, float]] = None,
    split_seed: int = 42,
    use_shm: bool = True,
    shm_basename: str = "kanvis_pose_12hz",
) -> CacheBuildResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_ratios = split_ratios or {
        "train": 0.7,
        "val_dev": 0.1,
        "threshold_calib": 0.1,
        "frozen_test": 0.1,
    }

    resampled: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = []
    for rec in records:
        poses = np.load(rec["pose_path"])
        if rec.get("valid_mask_path") and Path(rec["valid_mask_path"]).is_file():
            vm = np.load(rec["valid_mask_path"])
            if vm.dtype == bool and vm.shape[0] == poses.shape[0]:
                poses = poses.copy()
                poses[~vm] = np.nan
        fps = _fps_from_meta(rec.get("meta") or {})
        p2, valid = resample_poses_to_hz(poses, fps=fps, target_hz=target_hz)
        rec2 = dict(rec)
        rec2["T_resampled"] = int(p2.shape[0])
        rec2["fps_source"] = fps
        rec2["split"] = _assign_split(rec["group_id"], split_ratios, split_seed)
        resampled.append((p2, valid, rec2))

    total_frames = sum(int(p.shape[0]) for p, _, _ in resampled)
    if total_frames == 0:
        raise RuntimeError("Caché vacía: ningún clip válido")

    j = resampled[0][0].shape[1]
    poses_path = cache_dir / "poses.f32.mmap"
    valid_path = cache_dir / "valid.u8.mmap"

    if use_shm:
        shm = Path("/dev/shm")
        if shm.is_dir() and os.access(shm, os.W_OK):
            tag = hashlib.sha1(str(cache_dir).encode()).hexdigest()[:10]
            poses_path = shm / f"{shm_basename}_{tag}_poses.f32.mmap"
            valid_path = shm / f"{shm_basename}_{tag}_valid.u8.mmap"

    poses_mm = np.memmap(poses_path, dtype=np.float32, mode="w+", shape=(total_frames, j, 2))
    valid_mm = np.memmap(valid_path, dtype=np.uint8, mode="w+", shape=(total_frames, j))

    manifest_entries: List[Dict[str, Any]] = []
    offset = 0
    for p2, valid, rec2 in resampled:
        t = p2.shape[0]
        poses_mm[offset : offset + t] = p2
        valid_mm[offset : offset + t] = valid.astype(np.uint8)
        manifest_entries.append(
            {
                "uid": rec2.get("uid"),
                "label": rec2["label"],
                "is_robbery": rec2["is_robbery"],
                "group_id": rec2["group_id"],
                "split": rec2["split"],
                "clip_dir": rec2["clip_dir"],
                "pose_path": rec2["pose_path"],
                "frame_offset": offset,
                "num_frames": t,
                "target_hz": target_hz,
            }
        )
        offset += t

    poses_mm.flush()
    valid_mm.flush()

    manifest = {
        "version": 1,
        "target_hz": target_hz,
        "num_joints": j,
        "total_frames": total_frames,
        "num_clips": len(manifest_entries),
        "poses_mmap": str(poses_path),
        "valid_mmap": str(valid_path),
        "entries": manifest_entries,
    }
    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    bytes_on_disk = poses_path.stat().st_size + valid_path.stat().st_size
    return CacheBuildResult(
        manifest_path=manifest_path,
        poses_mmap_path=poses_path,
        valid_mmap_path=valid_path,
        total_frames=total_frames,
        num_clips=len(manifest_entries),
        bytes_on_disk=bytes_on_disk,
    )
