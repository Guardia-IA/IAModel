#!/usr/bin/env python3
"""Evaluación P0: incumbent #41/#48 — clip legacy (P0.A) y ventanas 8s/12Hz (P0.B)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
CAMPAIGN_DIR = TRAINING_DIR / "campaign"
for p in (TRAINING_DIR, CAMPAIGN2_DIR, CAMPAIGN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from p0_examples import (  # noqa: E402
    collect_examples_for_manifest,
    load_manifest,
    manifest_entries_for_split,
    map_examples_to_checkpoint_labels,
)
from train_model_operations import build_model  # noqa: E402
from train_v2.legacy_preprocess import poses_to_model_input, window_starts  # noqa: E402


def _import_binary_metrics():
    from evaluate_campaign import _binary_metrics  # type: ignore

    return _binary_metrics


def _import_collect_binary():
    from evaluate_campaign import collect_binary_predictions  # type: ignore

    return collect_binary_predictions


def _dummy_plan_path(output_dir: Path) -> Path:
    p = output_dir / "dummy_training_plan.json"
    if not p.is_file():
        p.write_text(json.dumps({"split": "val_dev", "note": "P0 placeholder"}), encoding="utf-8")
    return p


def _metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, Any]:
    pred = (probs >= float(thr)).astype(np.int64)
    return _import_binary_metrics()(y_true, pred)


@torch.no_grad()
def _run_windows_eval(
    model_path: Path,
    manifest: Dict[str, Any],
    entries: List[Dict[str, Any]],
    *,
    window_frames: int,
    stride_frames: int,
    batch_size: int,
    min_valid_joint_ratio: float = 0.25,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    label_to_idx = checkpoint["label_to_idx"]
    seq_len = int(checkpoint.get("seq_len", 64))
    cfg = checkpoint.get("config", {})
    arch = cfg.get("arch", "tcn")
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint.get("num_classes", len(label_to_idx)))

    poses_path = Path(manifest["poses_mmap"])
    valid_path = Path(manifest["valid_mmap"])
    total_frames = int(manifest["total_frames"])
    num_joints = int(manifest["num_joints"])
    poses_mm = np.memmap(
        poses_path, dtype=np.float32, mode="r", shape=(total_frames, num_joints, 2)
    )
    valid_mm = np.memmap(valid_path, dtype=np.uint8, mode="r", shape=(total_frames, num_joints))

    model = build_model(arch, input_dim, num_classes, cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    clip_records: List[Dict[str, Any]] = []
    y_true_list: List[int] = []
    probs_list: List[float] = []
    n_windows_total = 0

    batch_x: List[np.ndarray] = []
    batch_meta: List[Tuple[int, str, int]] = []  # clip_idx, uid, window_idx
    window_probs: Dict[int, List[float]] = {}

    def flush_batch() -> None:
        nonlocal batch_x, batch_meta
        if not batch_x:
            return
        x = torch.from_numpy(np.stack(batch_x, axis=0)).to(device)
        logits = model(x)
        prob = F.softmax(logits, dim=1)
        if num_classes == 2:
            p_pos = prob[:, 1].cpu().numpy()
        else:
            p_pos = prob[:, 0].cpu().numpy()
        for i, (cidx, _uid, _widx) in enumerate(batch_meta):
            window_probs.setdefault(cidx, []).append(float(p_pos[i]))
        batch_x = []
        batch_meta = []

    for cidx, ent in enumerate(entries):
        offset = int(ent["frame_offset"])
        t = int(ent["num_frames"])
        uid = str(ent["uid"])
        lab = int(ent["label"])
        if lab in label_to_idx:
            yt = int(label_to_idx[lab])
        else:
            yt = 1 if ent.get("is_robbery") else 0

        starts = window_starts(t, window_frames, stride_frames)
        if not starts:
            continue
        slice_poses = poses_mm[offset : offset + t]
        slice_valid = valid_mm[offset : offset + t]

        for widx, start in enumerate(starts):
            end = min(start + window_frames, t)
            chunk = np.array(slice_poses[start:end], copy=True)
            vchunk = slice_valid[start:end]
            if chunk.shape[0] < window_frames:
                chunk_padded = np.zeros((window_frames, num_joints, 2), dtype=np.float32)
                chunk_padded[: chunk.shape[0]] = chunk
                if chunk.shape[0] > 0:
                    chunk_padded[chunk.shape[0] :] = chunk[-1]
                chunk = chunk_padded
                vpad = np.zeros((window_frames, num_joints), dtype=np.uint8)
                vpad[: vchunk.shape[0]] = vchunk
                vchunk = vpad
            valid_ratio = float(vchunk.mean()) if vchunk.size else 0.0
            if valid_ratio < min_valid_joint_ratio:
                continue
            feat = poses_to_model_input(chunk, seq_len, pad_to_len=window_frames)
            batch_x.append(feat)
            batch_meta.append((cidx, uid, widx))
            n_windows_total += 1
            if len(batch_x) >= batch_size:
                flush_batch()

    flush_batch()

    for cidx, ent in enumerate(entries):
        uid = str(ent["uid"])
        lab = int(ent["label"])
        if lab in label_to_idx:
            yt = int(label_to_idx[lab])
        else:
            yt = 1 if ent.get("is_robbery") else 0
        wprobs = window_probs.get(cidx, [])
        if not wprobs:
            clip_prob = 0.0
        else:
            clip_prob = float(max(wprobs))
        clip_records.append(
            {
                "uid": uid,
                "true_label": yt,
                "prob_pos": clip_prob,
                "num_windows": len(wprobs),
                "label_raw": int(ent["label"]),
                "is_robbery": bool(ent.get("is_robbery")),
            }
        )
        y_true_list.append(yt)
        probs_list.append(clip_prob)

    y_true = np.array(y_true_list, dtype=np.int64)
    probs = np.array(probs_list, dtype=np.float64)
    meta = {"n_windows_scored": n_windows_total, "window_frames": window_frames, "stride_frames": stride_frames}
    return clip_records, y_true, probs, meta


def run_p0_eval(params: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    mode = str(params.get("mode", "legacy_clip"))
    model_path = Path(str(params["model_path"])).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"model_path no encontrado: {model_path}")

    manifest_path = Path(str(params["manifest_path"]))
    manifest = load_manifest(manifest_path)
    data_roots = [Path(p).expanduser().resolve() for p in (params.get("data_roots") or [])]
    eval_split = str(params.get("eval_split", "val_dev"))
    single_user_only = bool(params.get("single_user_only", False))
    pose_source = str(params.get("pose_source", "full"))
    batch_size = int(params.get("batch_size", 64))
    frozen_thr = float(params.get("frozen_threshold", 0.8))
    model_id = str(params.get("model_id", model_path.stem))

    temporal = params.get("temporal") or {}
    window_sec = float(temporal.get("window_seconds", 8.0))
    hz = float(temporal.get("target_hz", 12.0))
    stride_sec = float(temporal.get("stride_seconds", 1.0))
    window_frames = int(round(window_sec * hz))
    stride_frames = max(1, int(round(stride_sec * hz)))

    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "legacy_clip":
        device_peek = torch.device("cpu")
        ck_peek = torch.load(model_path, map_location=device_peek, weights_only=False)
        label_to_idx = ck_peek["label_to_idx"]
        robbery_class = int(params.get("robbery_class", 6))
        examples = collect_examples_for_manifest(
            data_roots,
            manifest,
            eval_split=eval_split,
            pose_source=pose_source,
            single_user_only=single_user_only,
        )
        examples = map_examples_to_checkpoint_labels(
            examples, label_to_idx, robbery_class=robbery_class
        )
        collect_binary_predictions = _import_collect_binary()
        plan = _dummy_plan_path(output_dir)
        records, y_true, probs, _margins = collect_binary_predictions(
            model_path,
            examples,
            training_plan_path=plan,
            split_name="val",
            batch_size=batch_size,
            num_workers=0,
        )
        clip_records = [
            {
                "uid": r["uid"],
                "true_label": int(r["true_label"]),
                "prob_pos": float(r["prob_pos"]),
                "label_raw": int(r.get("folder_category", r["true_label"])),
            }
            for r in records
        ]
        extra_meta: Dict[str, Any] = {"n_clips": len(examples), "preprocess": "legacy_pose_dataset_val"}
    elif mode == "windows":
        entries = manifest_entries_for_split(manifest, eval_split)
        clip_records, y_true, probs, extra_meta = _run_windows_eval(
            model_path,
            manifest,
            entries,
            window_frames=window_frames,
            stride_frames=stride_frames,
            batch_size=batch_size,
        )
        extra_meta["preprocess"] = "mmap_12hz_windows_max_aggregate"
    else:
        raise ValueError(f"mode desconocido: {mode}")

    metrics_thr = _metrics_at_threshold(y_true, probs, frozen_thr)
    metrics_argmax = _metrics_at_threshold(y_true, probs, 0.5)

    pred_path = output_dir / "clip_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(clip_records, f, indent=2, ensure_ascii=False)
        f.write("\n")

    payload = {
        "job_type": "p0_eval",
        "mode": mode,
        "model_id": model_id,
        "model_path": str(model_path),
        "eval_split": eval_split,
        "single_user_only": single_user_only,
        "frozen_threshold": frozen_thr,
        "pose_source": params.get("pose_source"),
        "metrics_at_frozen_threshold": metrics_thr,
        "metrics_softmax_0_5": metrics_argmax,
        "support": {"n_clips": len(clip_records), **extra_meta},
        "predictions_file": str(pred_path),
    }
    with open(output_dir / "p0_result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return payload
