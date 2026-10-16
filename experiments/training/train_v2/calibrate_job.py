"""Calibración de umbral en split threshold_calib (T) — solo negativos."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from train_v2.eval_checkpoint import run_v2_window_eval
from train_v2.metrics import (
    clip_metrics_at_fp_budget,
    threshold_from_negative_order_statistic,
)


def run_p1_calibrate(params: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    ckpt = Path(str(params["checkpoint_path"])).expanduser()
    manifest = Path(str(params["manifest_path"])).expanduser()
    temporal = params.get("temporal") or {}
    calib_split = str(params.get("calib_split", "threshold_calib"))
    selection = dict(params.get("selection") or {})
    max_fp_clips = int(
        params.get("max_fp_clips")
        or selection.get("max_val_fp_clips")
        or (params.get("p1_train") or {}).get("max_val_fp_clips")
        or 11
    )
    fp_budget_per_1000 = selection.get("fp_budget_per_1000_sessions")
    batch_size = int(params.get("batch_size", 64))
    min_valid = float((params.get("p1_train") or {}).get("min_valid_joint_ratio", 0.25))

    # Recoger scores clip-max sin umbral fijo (threshold=0 para logits path — usamos probs raw)
    raw = run_v2_window_eval(
        ckpt,
        manifest,
        eval_split=calib_split,
        temporal=temporal,
        batch_size=batch_size,
        min_valid_joint_ratio=min_valid,
        threshold=0.0,
    )
    preds = raw.get("predictions") or []
    if not preds:
        raise RuntimeError(f"Sin clips en split {calib_split!r} para calibración")

    uids = [str(p["uid"]) for p in preds]
    yt = np.array([int(p["true_label"]) for p in preds], dtype=np.int64)
    pr = np.array([float(p["prob_pos"]) for p in preds], dtype=np.float64)
    n_neg = int((yt == 0).sum())
    if fp_budget_per_1000 is not None and n_neg > 0:
        max_fp_clips = max(1, int(round(float(fp_budget_per_1000) * n_neg / 1000.0)))

    thr = threshold_from_negative_order_statistic(yt, pr, max_fp_clips=max_fp_clips)
    calib_metrics = clip_metrics_at_fp_budget(
        uids, yt, pr, max_fp_clips=max_fp_clips, threshold=thr
    )

    threshold_doc = {
        "version": 1,
        "split": calib_split,
        "threshold": float(thr),
        "max_fp_clips": int(max_fp_clips),
        "n_clips": len(preds),
        "n_neg_clips": n_neg,
        "metrics_at_threshold": calib_metrics,
        "source_checkpoint": str(ckpt),
        "job_id": params.get("source_job_id"),
    }
    out_thr = output_dir / "threshold.json"
    with open(out_thr, "w", encoding="utf-8") as f:
        json.dump(threshold_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Copia junto al checkpoint de entrenamiento para eval posterior
    train_job_dir = ckpt.parent
    with open(train_job_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    result = {
        "job_type": "p1_calibrate",
        "source_job_id": params.get("source_job_id"),
        "calib_split": calib_split,
        "threshold": float(thr),
        "max_fp_clips": int(max_fp_clips),
        "metrics": calib_metrics,
        "threshold_path": str(out_thr),
    }
    with open(output_dir / "p1_calibrate_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return result
