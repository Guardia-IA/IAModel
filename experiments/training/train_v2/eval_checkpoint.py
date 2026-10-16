"""Evaluación ventanas P0.B-style para checkpoints train_v2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from train_v2.dataset_windows import load_manifest, manifest_entries_for_split
from train_v2.features import PREPROCESS_TORSO, window_to_features
from train_v2.metrics import binary_metrics, clip_metrics_from_window_probs
from train_v2.models.gru_pooled import GRUPooledClassifier
from train_v2.models.masked_tcn import MaskedTCNClassifier
from train_v2.legacy_preprocess import window_starts
from train_v2.train_job import _build_model


def _load_v2_model(checkpoint: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    cfg = checkpoint.get("config") or {}
    arch = str(cfg.get("arch", "masked_tcn64"))
    input_dim = int(checkpoint["input_dim"])
    model = _build_model(arch, input_dim, cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def run_v2_window_eval(
    checkpoint_path: Path,
    manifest_path: Path,
    *,
    eval_split: str = "val_dev",
    temporal: Dict[str, Any] | None = None,
    batch_size: int = 64,
    min_valid_joint_ratio: float = 0.25,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    temporal = temporal or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    manifest = load_manifest(manifest_path)
    entries = manifest_entries_for_split(manifest, eval_split)

    hz = float(temporal.get("target_hz", manifest.get("target_hz", 12.0)))
    window_frames = int(round(float(temporal.get("window_seconds", 8.0)) * hz))
    stride_frames = max(1, int(round(float(temporal.get("stride_seconds", 1.0)) * hz)))
    seq_len = int(temporal.get("seq_len", checkpoint.get("seq_len", 96)))
    num_joints = int(manifest["num_joints"])

    model = _load_v2_model(checkpoint, device)
    normalize_mode = str(checkpoint.get("preprocess") or PREPROCESS_TORSO)

    poses_path = Path(manifest["poses_mmap"])
    valid_path = Path(manifest["valid_mmap"])
    tf = int(manifest["total_frames"])
    poses_mm = np.memmap(poses_path, dtype=np.float32, mode="r", shape=(tf, num_joints, 2))
    valid_mm = np.memmap(valid_path, dtype=np.uint8, mode="r", shape=(tf, num_joints))

    window_probs: Dict[int, List[float]] = {}
    batch_x: List[torch.Tensor] = []
    batch_m: List[torch.Tensor] = []
    batch_meta: List[Tuple[int, str, int]] = []

    def flush() -> None:
        nonlocal batch_x, batch_m, batch_meta
        if not batch_x:
            return
        x = torch.stack(batch_x, dim=0).to(device)
        m = torch.stack(batch_m, dim=0).to(device)
        logits = model(x, m)
        probs = torch.sigmoid(logits).cpu().numpy()
        for i, (cidx, uid, _w) in enumerate(batch_meta):
            window_probs.setdefault(cidx, []).append(float(probs[i]))
        batch_x, batch_m, batch_meta = [], [], []

    for cidx, ent in enumerate(entries):
        offset = int(ent["frame_offset"])
        t = int(ent["num_frames"])
        uid = str(ent["uid"])
        yt = 1 if ent.get("is_robbery") else 0
        starts = window_starts(t, window_frames, stride_frames)
        slice_poses = poses_mm[offset : offset + t]
        slice_valid = valid_mm[offset : offset + t]
        for widx, start in enumerate(starts):
            end = min(start + window_frames, t)
            chunk = np.array(slice_poses[start:end], copy=True)
            vchunk = np.array(slice_valid[start:end], copy=True)
            if chunk.shape[0] < window_frames:
                pad = window_frames - chunk.shape[0]
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)], axis=0)
                vchunk = np.concatenate([vchunk, np.repeat(vchunk[-1:], pad, axis=0)], axis=0)
            x_np, fmask, ok = window_to_features(
                chunk,
                vchunk,
                seq_len=seq_len,
                min_valid_joint_ratio=min_valid_joint_ratio,
                normalize_mode=normalize_mode,
            )
            if not ok:
                continue
            batch_x.append(torch.from_numpy(x_np))
            batch_m.append(torch.from_numpy(fmask))
            batch_meta.append((cidx, uid, widx))
            if len(batch_x) >= batch_size:
                flush()
        flush()
    flush()

    clip_records: List[Dict[str, Any]] = []
    y_true: List[int] = []
    probs: List[float] = []
    uids: List[str] = []
    for cidx, ent in enumerate(entries):
        uid = str(ent["uid"])
        yt = 1 if ent.get("is_robbery") else 0
        wprobs = window_probs.get(cidx, [])
        pclip = float(max(wprobs)) if wprobs else 0.0
        clip_records.append({"uid": uid, "true_label": yt, "prob_pos": pclip, "num_windows": len(wprobs)})
        y_true.append(yt)
        probs.append(pclip)
        uids.append(uid)

    yt_arr = np.array(y_true, dtype=np.int64)
    pr_arr = np.array(probs, dtype=np.float64)
    pred = (pr_arr >= threshold).astype(np.int64)
    return {
        "checkpoint": str(checkpoint_path),
        "eval_split": eval_split,
        "threshold": threshold,
        "window_metrics": binary_metrics(yt_arr, pred),
        "clip_max_metrics": clip_metrics_from_window_probs(uids, yt_arr, pr_arr, threshold=threshold),
        "n_clips": len(entries),
        "predictions": clip_records,
        "job_id": checkpoint.get("job_id"),
        "seed": checkpoint.get("seed"),
    }


def run_p1_eval_job(params: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    ckpt = Path(str(params["checkpoint_path"])).expanduser()
    manifest = Path(str(params["manifest_path"])).expanduser()
    temporal = params.get("temporal") or {}
    out = run_v2_window_eval(
        ckpt,
        manifest,
        eval_split=str(params.get("eval_split", "val_dev")),
        temporal=temporal,
        batch_size=int(params.get("batch_size", 64)),
        min_valid_joint_ratio=float((params.get("p1_train") or {}).get("min_valid_joint_ratio", 0.25)),
        threshold=float(params.get("threshold", 0.5)),
    )
    out["job_type"] = "p1_eval"
    out["source_job_id"] = params.get("source_job_id")
    with open(output_dir / "p1_eval_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    preds = out.pop("predictions", [])
    with open(output_dir / "clip_predictions.json", "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out
