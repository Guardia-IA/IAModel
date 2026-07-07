"""Utilidades compartidas: splits, estimación disco/tiempo, selección de modelos."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from train_model_operations import verify_split_uid_disjoint
except ImportError:
    from ..train_model_operations import verify_split_uid_disjoint  # type: ignore


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [{k: row[k] for k in row} for row in csv.DictReader(f)]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def verify_mass_augment_split(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Comprueba train/val/test disjuntos y que mass-aug solo afecta train."""
    split_uids = plan.get("split_uids") or {}
    issues: List[str] = []
    passed = True

    try:
        verify_split_uid_disjoint(split_uids)
    except ValueError as exc:
        passed = False
        issues.append(str(exc))

    train = {str(u) for u in split_uids.get("train", [])}
    val = {str(u) for u in split_uids.get("val", [])}
    test = {str(u) for u in split_uids.get("test", [])}
    ratios = plan.get("split_ratios") or {}

    stats = plan.get("split_stats_by_category") or {}
    train_clips = sum(int(v) for v in (stats.get("train") or {}).values())
    val_clips = sum(int(v) for v in (stats.get("val") or {}).values())
    test_clips = sum(int(v) for v in (stats.get("test") or {}).values())

    mass = plan.get("mass_augmentation") or {}
    if not mass.get("enabled"):
        issues.append("mass_augmentation.enabled no está activo en el plan")
        passed = False

    apply_splits = (plan.get("mass_augmentation") or {}).get("apply_to_splits") or ["train"]
    if "val" in apply_splits or "test" in apply_splits:
        issues.append("mass_augmentation no debe aplicarse a val/test")
        passed = False

    rob_train = int((plan.get("split_stats_by_category", {}).get("train") or {}).get("6", 0))
    if (plan.get("mass_augmentation") or {}).get("keep_all_robbery_in_train", True) and rob_train <= 0:
        issues.append("keep_all_robbery: no hay clips de robo (cat.6) en train")
        passed = False

    return {
        "passed": passed and len(issues) == 0,
        "passed_disjoint": passed,
        "issues": issues,
        "uids_train": len(train),
        "uids_val": len(val),
        "uids_test": len(test),
        "clips_train": train_clips,
        "clips_val": val_clips,
        "clips_test": test_clips,
        "split_ratios": ratios,
        "mass_augment_enabled": bool(mass.get("enabled")),
        "projected_train_rows": int((mass.get("plan") or {}).get("projected_total_rows", 0)),
        "robbery_clips_train": rob_train,
    }


def estimate_mass_augment_storage(
    *,
    n_experiments: int,
    n_cells: int,
    train_rows_per_cell: int,
    export_fp_videos: bool = False,
) -> Dict[str, Any]:
    """Heurística de espacio en disco (GB)."""
    model_mb = 18.0
    models_gb = n_experiments * n_cells * model_mb / 1024.0
    plans_mb = n_cells * 3.0
    sweep_csv_mb = n_cells * n_experiments * 0.08
    logs_gb = max(0.5, n_cells * 0.3)
    reports_gb = n_cells * 0.15
    fp_clips_gb = n_cells * 2.0 if export_fp_videos else 0.0
    cache_gb = 0.5
    total_gb = models_gb + plans_mb / 1024.0 + sweep_csv_mb / 1024.0 + logs_gb + reports_gb + fp_clips_gb + cache_gb
    recommended_gb = total_gb * 1.25

    return {
        "models_gb": round(models_gb, 2),
        "logs_gb": round(logs_gb, 2),
        "reports_gb": round(reports_gb, 2),
        "fp_clips_gb": round(fp_clips_gb, 2),
        "misc_gb": round(plans_mb / 1024.0 + sweep_csv_mb / 1024.0 + cache_gb, 2),
        "total_gb": round(total_gb, 2),
        "recommended_free_gb": round(recommended_gb, 2),
        "train_rows_per_cell": int(train_rows_per_cell),
        "note": "Los datos augmentados no se materializan en disco; se generan on-the-fly en train.",
    }


def estimate_mass_augment_eval_seconds(
    *,
    n_experiments: int,
    n_cells: int,
    val_rows: int,
    clips_x: int = 40,
    variants_y: int = 8,
    n_synthetic_selections: int = 4,
    batch_size: int = 64,
    cuda: Optional[bool] = None,
) -> Dict[str, Any]:
    """Tiempo heurístico de evaluate_mass_augment (real + sintético + sweep)."""
    if cuda is None:
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
        except ImportError:
            cuda = False

    val_batches = max(1, math.ceil(max(1, val_rows) / batch_size))
    forward_gpu = 0.004
    forward_cpu = forward_gpu * 8.0
    t_batch = forward_gpu if cuda else forward_cpu

    sweep_steps = 18
    real_per_model = val_batches * t_batch * (1 + sweep_steps)
    real_per_cell = n_experiments * real_per_model

    syn_rows = 2 * clips_x * variants_y
    syn_batches = max(1, math.ceil(syn_rows / batch_size))
    syn_per_cell = n_synthetic_selections * syn_batches * t_batch

    ensemble_overhead = 0.15 * real_per_cell
    per_cell = real_per_cell + syn_per_cell + ensemble_overhead
    total = per_cell * n_cells

    return {
        "seconds": float(total),
        "human": _fmt(total),
        "hours": round(total / 3600.0, 2),
        "per_cell_seconds": float(per_cell),
        "real_eval_seconds": float(real_per_cell * n_cells),
        "synthetic_eval_seconds": float(syn_per_cell * n_cells),
        "val_rows": int(val_rows),
        "experiments": int(n_experiments),
        "cells": int(n_cells),
    }


def _fmt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"~{h}h {m}m {s}s"
    if m:
        return f"~{m}m {s}s"
    return f"~{s}s"


def pick_best_f1_from_sweep(sweep_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rows = [r for r in sweep_rows if r.get("decision_mode") == "softmax_thr"]
    if not rows:
        rows = list(sweep_rows)
    if not rows:
        return None
    return max(rows, key=lambda r: _f(r, "f1_pct"))


def pick_best_min_fp_from_sweep(
    sweep_rows: List[Dict[str, Any]],
    *,
    min_recall_pct: float = 60.0,
) -> Optional[Dict[str, Any]]:
    rows = [r for r in sweep_rows if r.get("decision_mode") == "softmax_thr"]
    if not rows:
        return None
    meets = [r for r in rows if _f(r, "recall_pct") >= min_recall_pct]
    pool = meets or rows
    return min(pool, key=lambda r: (_f(r, "fp_rate_pct"), -_f(r, "f1_pct")))


def pick_best_from_leaderboard(
    leaderboard_rows: List[Dict[str, Any]],
    *,
    key: str = "f1_pct",
    minimize: bool = False,
    min_recall_pct: float = 0.0,
) -> Optional[Dict[str, Any]]:
    singles = [
        r for r in leaderboard_rows
        if not str(r.get("decision_mode", "")).startswith("ensemble_")
    ]
    if min_recall_pct > 0:
        filtered = [r for r in singles if _f(r, "recall_pct") >= min_recall_pct]
        singles = filtered or singles
    if not singles:
        return None
    if minimize:
        return min(singles, key=lambda r: (_f(r, key), -_f(r, "f1_pct")))
    return max(singles, key=lambda r: _f(r, key))


def pick_best_ensemble_from_grid(
    ensemble_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not ensemble_rows:
        return None
    target_rec = float(config.get("target_recall_pct", 60.0))
    target_fp = float(config.get("target_fp_rate_pct", 0.01))
    meets_rec = [r for r in ensemble_rows if _f(r, "recall_pct") >= target_rec]
    if meets_rec:
        meets_both = [r for r in meets_rec if _f(r, "fp_rate_pct") <= target_fp]
        pool = meets_both or meets_rec
        return min(pool, key=lambda r: (_f(r, "fp_rate_pct"), -_f(r, "f1_pct")))
    return max(ensemble_rows, key=lambda r: _f(r, "f1_pct"))


def load_eval_artifacts(reports_dir: Path, split: str) -> Tuple[List[Dict], List[Dict], List[Dict], Optional[Dict]]:
    sweep = _read_csv(reports_dir / f"{split}_decision_sweep.csv")
    leader = _read_csv(reports_dir / f"{split}_leaderboard.csv")
    ensemble = _read_csv(reports_dir / f"{split}_ensemble_grid.csv")
    best_ens_path = reports_dir / f"{split}_best_ensemble.json"
    best_ens = json.loads(best_ens_path.read_text(encoding="utf-8")) if best_ens_path.is_file() else None
    return sweep, leader, ensemble, best_ens


def _resolve_model_path(models_dir: Path, name: str) -> Path:
    n = str(name).strip()
    if not n.endswith(".pt"):
        n = f"{n}.pt"
    return models_dir / n


def fp_by_category_from_manifest(fp_manifest_path: Path) -> List[Dict[str, Any]]:
    """Agrega falsos positivos reales por categoría de carpeta."""
    rows = _read_csv(fp_manifest_path)
    if not rows:
        return []
    by_cat: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cat = str(r.get("folder_category") or r.get("category_str") or "?")
        block = by_cat.setdefault(cat, {"folder_category": cat, "fp_count": 0})
        block["fp_count"] = int(block["fp_count"]) + 1
    out = sorted(by_cat.values(), key=lambda x: -int(x["fp_count"]))
    for row in out:
        row["fp_pct_of_total"] = round(
            100.0 * int(row["fp_count"]) / max(len(rows), 1), 2
        )
    return out


def synthetic_gate_check(
    real_fp_rate_pct: float,
    synthetic_fp_rate_pct: float,
    *,
    max_ratio: float = 2.0,
) -> Dict[str, Any]:
    """Alerta si FP sintético >> FP real (overfit a augment)."""
    real = float(real_fp_rate_pct or 0)
    syn = float(synthetic_fp_rate_pct or 0)
    ratio = syn / max(real, 1e-6) if syn > 0 else 0.0
    passed = syn <= real * max_ratio or syn <= real + 0.5
    return {
        "passed": passed,
        "real_fp_rate_pct": real,
        "synthetic_fp_rate_pct": syn,
        "ratio": round(ratio, 2),
        "max_ratio": max_ratio,
        "warning": None if passed else (
            f"FP sintético ({syn:.2f}%) >> real ({real:.2f}%), ratio={ratio:.1f}x — posible overfit augment"
        ),
    }


def pick_best_operational_from_sweep(
    sweep_rows: List[Dict[str, Any]],
    *,
    min_recall_pct: float = 60.0,
    target_fp_rate_pct: float = 0.01,
) -> Optional[Dict[str, Any]]:
    """Min FP con recall mínimo; prioriza cumplir target FP si existe."""
    rows = [r for r in sweep_rows if r.get("decision_mode") == "softmax_thr"]
    if not rows:
        return None
    meets_rec = [r for r in rows if _f(r, "recall_pct") >= min_recall_pct]
    pool = meets_rec or rows
    meets_fp = [r for r in pool if _f(r, "fp_rate_pct") <= target_fp_rate_pct]
    if meets_fp:
        return max(meets_fp, key=lambda r: (_f(r, "f1_pct"), -_f(r, "fp_rate_pct")))
    return min(pool, key=lambda r: (_f(r, "fp_rate_pct"), -_f(r, "f1_pct")))


def selection_specs_from_eval(
    reports_dir: Path,
    split: str,
    config: Dict[str, Any],
    models_dir: Path,
) -> List[Dict[str, Any]]:
    """Devuelve selecciones: best_f1, best_min_fp, best_operational, best_ensemble."""
    sweep, leader, ensemble, best_ens_json = load_eval_artifacts(reports_dir, split)
    min_rec = float(config.get("target_recall_pct", 60.0))
    ma = config.get("mass_augment") or {}
    deploy = ma.get("deploy_targets") or {}
    target_fp = float(deploy.get("fp_rate_pct") or config.get("target_fp_rate_pct", 0.01))
    deploy_rec = float(deploy.get("recall_pct") or min_rec)
    specs: List[Dict[str, Any]] = []

    f1_row = pick_best_f1_from_sweep(sweep) or pick_best_from_leaderboard(leader, key="f1_pct")
    if f1_row and f1_row.get("model"):
        mname = str(f1_row["model"])
        specs.append({
            "selection": "best_f1",
            "kind": "single_model",
            "model": mname,
            "model_path": str(_resolve_model_path(models_dir, mname)),
            "threshold": _f(f1_row, "threshold", 0.5),
            "f1_pct": _f(f1_row, "f1_pct"),
            "fp_rate_pct": _f(f1_row, "fp_rate_pct"),
            "recall_pct": _f(f1_row, "recall_pct"),
        })

    fp_row = pick_best_min_fp_from_sweep(sweep, min_recall_pct=min_rec)
    if fp_row and fp_row.get("model"):
        fp_model = str(fp_row["model"])
        if not any(s.get("model") == fp_model and s["selection"] == "best_f1" for s in specs):
            specs.append({
                "selection": "best_min_fp",
                "kind": "single_model",
                "model": fp_model,
                "model_path": str(_resolve_model_path(models_dir, fp_model)),
                "threshold": _f(fp_row, "threshold", 0.5),
                "f1_pct": _f(fp_row, "f1_pct"),
                "fp_rate_pct": _f(fp_row, "fp_rate_pct"),
                "recall_pct": _f(fp_row, "recall_pct"),
            })
        elif len(specs) == 1:
            specs[0]["selection"] = "best_f1_and_min_fp_same_model"

    op_row = pick_best_operational_from_sweep(
        sweep, min_recall_pct=deploy_rec, target_fp_rate_pct=target_fp,
    )
    if op_row and op_row.get("model"):
        op_model = str(op_row["model"])
        if not any(s.get("model") == op_model and s.get("selection") == "best_operational" for s in specs):
            specs.append({
                "selection": "best_operational",
                "kind": "single_model",
                "model": op_model,
                "model_path": str(_resolve_model_path(models_dir, op_model)),
                "threshold": _f(op_row, "threshold", 0.5),
                "f1_pct": _f(op_row, "f1_pct"),
                "fp_rate_pct": _f(op_row, "fp_rate_pct"),
                "recall_pct": _f(op_row, "recall_pct"),
            })

    ens_row = pick_best_ensemble_from_grid(ensemble, config)
    if ens_row:
        models = [m.strip() for m in str(ens_row.get("models", "")).split("|") if m.strip()]
        specs.append({
            "selection": "best_ensemble",
            "kind": "ensemble",
            "models": models,
            "model_paths": [str(_resolve_model_path(models_dir, m)) for m in models],
            "rule": str(ens_row.get("decision_mode", "")).replace("ensemble_", ""),
            "thresholds": str(ens_row.get("thresholds", "")),
            "f1_pct": _f(ens_row, "f1_pct"),
            "fp_rate_pct": _f(ens_row, "fp_rate_pct"),
            "recall_pct": _f(ens_row, "recall_pct"),
        })
    elif best_ens_json:
        models = list(best_ens_json.get("models") or [])
        specs.append({
            "selection": "best_ensemble",
            "kind": "ensemble",
            "models": models,
            "model_paths": [str(_resolve_model_path(models_dir, m)) for m in models],
            "rule": best_ens_json.get("rule", "mean"),
            "thresholds": best_ens_json.get("thresholds"),
            "f1_pct": _f(best_ens_json, "f1_pct"),
            "fp_rate_pct": _f(best_ens_json, "fp_rate_pct"),
            "recall_pct": _f(best_ens_json, "recall_pct"),
        })

    return specs
