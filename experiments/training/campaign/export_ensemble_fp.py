#!/usr/bin/env python3
"""
Exporta FP/FN de un ensemble binario de campaña.

Por defecto usa el mejor ensemble detectado en evaluate_campaign.py
({split}_best_ensemble.json). Ya no asume modelo_06+modelo_14.

Uso:
  cd experiments/training/campaign

  # Tras evaluate_campaign.py (lee best_ensemble.json)
  python export_ensemble_fp.py --split val --outcomes errors

  # Ensemble explícito
  python export_ensemble_fp.py --cell bin_full --models modelo_06 modelo_49 \\
      --rule mean --threshold 0.68 --split val
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

try:
    from campaign_paths import ensure_cell_dirs, training_plan_path, load_campaign_config
    from evaluate_validation import load_split_uids, build_split_examples
    from evaluate_campaign import (
        collect_binary_predictions,
        _binary_metrics,
        load_best_ensemble_spec,
    )
    from export_fp_artifacts import export_fp_from_records, example_export_paths
except ImportError as exc:
    raise SystemExit(f"Import error (¿entorno con torch?): {exc}") from exc


def _normalize_model_name(name: str) -> str:
    name = str(name).strip()
    if name.isdigit():
        return f"modelo_{int(name):02d}.pt"
    if not name.endswith(".pt"):
        if name.startswith("modelo_"):
            return name + ".pt"
        return f"modelo_{name}.pt"
    return name


def _resolve_model_paths(models_dir: Path, names: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for n in names:
        p = Path(n)
        if p.is_file():
            paths.append(p.resolve())
            continue
        candidate = models_dir / _normalize_model_name(n)
        if not candidate.is_file():
            raise FileNotFoundError(f"No existe modelo: {candidate}")
        paths.append(candidate.resolve())
    return paths


def _ensemble_predict(
    prob_matrix: np.ndarray,
    rule: str,
    threshold: float,
    *,
    cascade_low: float = 0.4,
    cascade_high: float = 0.55,
) -> np.ndarray:
    """prob_matrix shape (n_models, n_examples)."""
    rule = rule.lower()
    if rule == "mean":
        p = prob_matrix.mean(axis=0)
        return (p >= threshold).astype(np.int64)
    if rule == "and":
        mask = np.ones(prob_matrix.shape[1], dtype=bool)
        for i in range(prob_matrix.shape[0]):
            mask &= prob_matrix[i] >= threshold
        return mask.astype(np.int64)
    if rule == "or":
        mask = np.zeros(prob_matrix.shape[1], dtype=bool)
        for i in range(prob_matrix.shape[0]):
            mask |= prob_matrix[i] >= threshold
        return mask.astype(np.int64)
    if rule == "cascade":
        if prob_matrix.shape[0] < 2:
            raise ValueError("cascade requiere al menos 2 modelos")
        pred = np.zeros(prob_matrix.shape[1], dtype=np.int64)
        for j in range(prob_matrix.shape[1]):
            if prob_matrix[0, j] >= cascade_low and prob_matrix[1, j] >= cascade_high:
                pred[j] = 1
        return pred
    raise ValueError(f"Regla desconocida: {rule!r} (mean|and|or|cascade)")


def _threshold_from_spec(spec: Dict[str, Any], rule: str) -> float:
    thr = spec.get("threshold")
    if isinstance(thr, list):
        return float(thr[0])
    if thr is not None:
        return float(thr)
    thrs = spec.get("thresholds")
    if isinstance(thrs, list) and thrs:
        return float(thrs[0])
    if isinstance(thrs, str) and thrs.strip():
        return float(thrs.split("|")[0])
    return 0.5


def resolve_ensemble_args(
    *,
    cell_id: str,
    split: str,
    model_names: Optional[Sequence[str]],
    rule: Optional[str],
    threshold: Optional[float],
    run_id: Optional[str],
) -> Tuple[List[str], str, float, Dict[str, Any]]:
    arts = ensure_cell_dirs(cell_id, run_id=run_id)
    spec = load_best_ensemble_spec(arts["reports_dir"], split)
    if model_names:
        models = list(model_names)
        ens_rule = str(rule or (spec or {}).get("rule") or "mean")
        ens_thr = float(threshold if threshold is not None else _threshold_from_spec(spec or {}, ens_rule))
        return models, ens_rule, ens_thr, spec or {}

    if spec is None:
        raise FileNotFoundError(
            f"No hay {split}_best_ensemble.json en {arts['reports_dir']}. "
            "Ejecuta evaluate_campaign.py --all primero."
        )
    models = [str(m) for m in spec.get("models") or []]
    if not models:
        raise ValueError(f"best_ensemble.json sin modelos: {arts['reports_dir']}")
    ens_rule = str(rule or spec.get("rule") or "mean")
    ens_thr = float(threshold if threshold is not None else _threshold_from_spec(spec, ens_rule))
    return models, ens_rule, ens_thr, spec


def run_ensemble_export(
    *,
    cell_id: str,
    model_names: Optional[Sequence[str]],
    split: str,
    rule: Optional[str],
    threshold: Optional[float],
    cascade_low: float,
    cascade_high: float,
    single_user_only: Optional[bool],
    batch_size: int,
    export_videos: bool,
    use_symlink: bool,
    outcomes: str,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_models, ens_rule, ens_thr, spec = resolve_ensemble_args(
        cell_id=cell_id,
        split=split,
        model_names=model_names,
        rule=rule,
        threshold=threshold,
        run_id=run_id,
    )

    arts = ensure_cell_dirs(cell_id, run_id=run_id)
    models_dir = arts["models_dir"]
    plan_path = training_plan_path(cell_id, run_id=run_id)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Falta plan: {plan_path}. Ejecuta preflight_campaign.py --write-all")

    config = load_campaign_config()
    cell = next((c for c in config.get("cells", []) if c["id"] == cell_id), None)
    if cell is None:
        raise ValueError(f"Celda desconocida en campaign_config.json: {cell_id!r}")

    model_paths = _resolve_model_paths(models_dir, resolved_models)
    split_uids, split_meta = load_split_uids(split_name=split, training_plan_path=plan_path)
    split_meta["split_name"] = split
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    split_meta["split_uids_all"] = {
        k: [str(x) for x in v] for k, v in plan.get("split_uids", {}).items()
    }

    su = single_user_only if single_user_only is not None else bool(config.get("single_user_only", True))
    examples, pool_info = build_split_examples(
        split_uids=split_uids,
        split_meta=split_meta,
        pose_source=cell["pose_source"],
        single_user_only=su,
        task=cell["task"],
    )
    if not examples:
        raise RuntimeError(f"No hay ejemplos en split {split!r}")
    print(
        f"  Export ensemble: {[p.name for p in model_paths]} | rule={ens_rule} @ {ens_thr} | "
        f"{split} clips={len(examples)}",
        flush=True,
    )

    prob_rows: List[np.ndarray] = []
    base_records: Optional[List[Dict[str, Any]]] = None
    y_true: Optional[np.ndarray] = None
    model_labels: List[str] = []

    for mp in model_paths:
        print(f"  → Inferencia {mp.name} ({len(examples)} clips)...", flush=True)
        recs, yt, probs, _margins = collect_binary_predictions(
            mp,
            examples,
            training_plan_path=plan_path,
            split_name=split,
            batch_size=batch_size,
        )
        prob_rows.append(probs)
        model_labels.append(mp.name)
        if base_records is None:
            base_records = recs
            y_true = yt
        elif len(recs) != len(base_records):
            raise RuntimeError(f"Desalineación de predicciones: {mp.name}")
        print(f"  ✓ {mp.name} listo", flush=True)

    assert base_records is not None and y_true is not None
    prob_matrix = np.stack(prob_rows, axis=0)
    p_mean = prob_matrix.mean(axis=0)
    y_pred = _ensemble_predict(
        prob_matrix,
        ens_rule,
        ens_thr,
        cascade_low=cascade_low,
        cascade_high=cascade_high,
    )
    metrics = _binary_metrics(y_true, y_pred)

    all_rows: List[Dict[str, Any]] = []
    for i, base in enumerate(base_records):
        yt = int(y_true[i])
        yp = int(y_pred[i])
        if yp == 1 and yt == 0:
            outcome = "FP"
        elif yp == 0 and yt == 1:
            outcome = "FN"
        elif yp == 1 and yt == 1:
            outcome = "TP"
        else:
            outcome = "TN"

        paths = example_export_paths(examples[i])
        row: Dict[str, Any] = {
            "uid": paths["uid"],
            "uid_absolute": paths["uid_absolute"],
            "clip_name": paths["clip_name"],
            "category_str": paths["category_str"],
            "folder_category": paths["folder_category"],
            "true_label": yt,
            "pred_label": yp,
            "outcome": outcome,
            "p_mean": round(float(p_mean[i]), 6),
            "threshold": ens_thr,
            "rule": ens_rule,
            "clip_video_path": paths["clip_video_path"],
            "clip_video_exists": paths["clip_video_exists"],
            "clip_dir": paths["clip_dir"],
            "meta_json_path": paths["meta_json_path"],
            "user_dir": paths["user_dir"],
            "pose_path": paths["pose_path"],
            "clip_path": paths["clip_video_path"] or paths["clip_dir"],
        }
        for label, probs in zip(model_labels, prob_rows):
            key = label.replace(".pt", "")
            row[f"p_{key}"] = round(float(probs[i]), 6)
        all_rows.append(row)

    if outcomes == "fp":
        selected = [r for r in all_rows if r["outcome"] == "FP"]
    elif outcomes == "errors":
        selected = [r for r in all_rows if r["outcome"] in ("FP", "FN")]
    elif outcomes == "alarms":
        selected = [r for r in all_rows if int(r["pred_label"]) == 1]
    else:
        selected = all_rows

    reports_dir = arts["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_tag = "|".join(m.replace(".pt", "") for m in model_labels)
    csv_path = reports_dir / f"{split}_ensemble_fp_{ens_rule}_{model_tag}_t{ens_thr:.2f}.csv".replace("|", "+")

    fieldnames: List[str] = [
        "outcome",
        "folder_category",
        "category_str",
        "clip_name",
        "clip_video_path",
        "clip_dir",
        "meta_json_path",
        "user_dir",
        "pose_path",
        "uid",
        "uid_absolute",
        "p_mean",
        "threshold",
        "rule",
        "true_label",
        "pred_label",
        "clip_video_exists",
        "clip_path",
    ]
    for r in all_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(selected)

    fn_paths = [
        r.get("clip_video_path") or r.get("clip_dir") or ""
        for r in selected
        if r.get("outcome") == "FN"
    ]
    fp_paths = [
        r.get("clip_video_path") or r.get("clip_dir") or ""
        for r in selected
        if r.get("outcome") == "FP"
    ]

    video_list_path = csv_path.with_suffix(".videos.txt")
    with open(video_list_path, "w", encoding="utf-8") as f:
        for r in selected:
            video = r.get("clip_video_path") or r.get("clip_dir") or ""
            if video:
                f.write(f"{video}\n")

    fn_list_path = csv_path.with_suffix(".fn.txt")
    fp_list_path = csv_path.with_suffix(".fp.txt")
    with open(fn_list_path, "w", encoding="utf-8") as f:
        f.write(f"# FN robos no detectados — {ens_rule} {model_tag} @ {ens_thr}\n")
        for p in fn_paths:
            if p:
                f.write(f"{p}\n")
    with open(fp_list_path, "w", encoding="utf-8") as f:
        f.write(f"# FP falsos positivos — {ens_rule} {model_tag} @ {ens_thr}\n")
        for p in fp_paths:
            if p:
                f.write(f"{p}\n")

    logs_dir = arts["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    fn_log = logs_dir / f"{split}_{cell_id}_ensemble_fn.txt"
    fp_log = logs_dir / f"{split}_{cell_id}_ensemble_fp.txt"
    fn_log.write_text(fn_list_path.read_text(encoding="utf-8"), encoding="utf-8")
    fp_log.write_text(fp_list_path.read_text(encoding="utf-8"), encoding="utf-8")

    summary_path = reports_dir / f"{split}_ensemble_fp_summary.json"
    summary = {
        "cell_id": cell_id,
        "split": split,
        "models": [str(p) for p in model_paths],
        "rule": ens_rule,
        "threshold": ens_thr,
        "source_spec": spec,
        "cascade_low": cascade_low,
        "cascade_high": cascade_high,
        "metrics": metrics,
        "pool_info": pool_info,
        "exported_rows": len(selected),
        "outcomes_filter": outcomes,
        "csv_path": str(csv_path.resolve()),
        "video_list_path": str(video_list_path.resolve()),
        "fn_list_path": str(fn_list_path.resolve()),
        "fp_list_path": str(fp_list_path.resolve()),
        "fn_log_path": str(fn_log.resolve()),
        "fp_log_path": str(fp_log.resolve()),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    fp_export_dir: Optional[Path] = None
    if export_videos and selected:
        fp_export_dir = arts["fp_clips_dir"] / f"ensemble_{ens_rule}_t{ens_thr:.2f}"
        fp_records = []
        for r in selected:
            fp_records.append({
                **r,
                "prob_pos": r["p_mean"],
                "model_path": model_tag,
            })
        export_fp_from_records(
            fp_records,
            fp_export_dir,
            cell_id=cell_id,
            use_symlink=use_symlink,
        )

    return {
        "csv_path": csv_path,
        "video_list_path": video_list_path,
        "fn_list_path": fn_list_path,
        "fp_list_path": fp_list_path,
        "fn_log_path": fn_log,
        "fp_log_path": fp_log,
        "summary_path": summary_path,
        "metrics": metrics,
        "selected": selected,
        "models": resolved_models,
        "rule": ens_rule,
        "threshold": ens_thr,
        "fp_export_dir": fp_export_dir,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Exporta FP/FN de un ensemble binario (auto desde best_ensemble.json)"
    )
    ap.add_argument("--cell", type=str, default="bin_full_hardened")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Modelos .pt (si omites, usa {split}_best_ensemble.json de evaluate)",
    )
    ap.add_argument("--rule", choices=["mean", "and", "or", "cascade"], default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--cascade-low", type=float, default=0.4)
    ap.add_argument("--cascade-high", type=float, default=0.55)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--single-user-only", action="store_true", default=None)
    ap.add_argument("--no-single-user-only", action="store_true")
    ap.add_argument(
        "--outcomes",
        choices=["fp", "errors", "alarms", "all"],
        default="errors",
        help="errors=FP+FN (default) | fp=solo FP | alarms=TP+FP",
    )
    ap.add_argument("--export-videos", action="store_true", help="Symlink clip.mp4 en fp_clips/")
    ap.add_argument("--copy-videos", action="store_true", help="Copiar vídeos en lugar de symlink")
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Artefactos bajo artifacts/runs/<run-id>/",
    )
    args = ap.parse_args()

    su: Optional[bool] = None
    if args.single_user_only:
        su = True
    elif args.no_single_user_only:
        su = False

    try:
        result = run_ensemble_export(
            cell_id=args.cell,
            model_names=args.models,
            split=args.split,
            rule=args.rule,
            threshold=args.threshold,
            cascade_low=args.cascade_low,
            cascade_high=args.cascade_high,
            single_user_only=su,
            batch_size=args.batch_size,
            export_videos=args.export_videos,
            use_symlink=not args.copy_videos,
            outcomes=args.outcomes,
            run_id=str(args.run_id).strip() if args.run_id else None,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    m = result["metrics"]
    print(
        f"\n=== Ensemble {result['rule']} @ {result['threshold']} — split {args.split} ==="
    )
    print(f"Modelos: {result['models']}")
    print(f"Celda: {args.cell}")
    print(
        f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} | "
        f"F1={m['f1_pct']:.1f}% Rec={m['recall_pct']:.1f}% FP={m['fp_rate_pct']:.2f}%"
    )
    print(f"\nExportado ({args.outcomes}): {len(result['selected'])} filas")
    print(f"CSV: {result['csv_path'].resolve()}")
    print(f"FN (robos no detectados): {result['fn_list_path'].resolve()}")
    print(f"FP (falsos positivos): {result['fp_list_path'].resolve()}")
    print(f"Logs FN: {result['fn_log_path'].resolve()}")
    print(f"Logs FP: {result['fp_log_path'].resolve()}")
    print(f"Resumen JSON: {result['summary_path'].resolve()}")
    if result["fp_export_dir"]:
        print(f"Vídeos FP: {result['fp_export_dir'].resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
