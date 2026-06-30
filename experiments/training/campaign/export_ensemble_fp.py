#!/usr/bin/env python3
"""
Exporta la lista de falsos positivos (y opcionalmente FN/TP) de un ensemble binario.

Pensado para la config recomendada:
  bin_full_hardened, MEAN(modelo_06 + modelo_14) >= 0.68

Uso:
  cd experiments/training/campaign

  # Config recomendada (defaults)
  python export_ensemble_fp.py --split val

  # Con vídeos symlink
  python export_ensemble_fp.py --split val --export-videos

  # Otro ensemble
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
    from evaluate_campaign import collect_binary_predictions, _binary_metrics
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


def run_ensemble_export(
    *,
    cell_id: str,
    model_names: Sequence[str],
    split: str,
    rule: str,
    threshold: float,
    cascade_low: float,
    cascade_high: float,
    single_user_only: Optional[bool],
    batch_size: int,
    export_videos: bool,
    use_symlink: bool,
    outcomes: str,
) -> Dict[str, Any]:
    arts = ensure_cell_dirs(cell_id)
    models_dir = arts["models_dir"]
    plan_path = training_plan_path(cell_id)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Falta plan: {plan_path}. Ejecuta preflight_campaign.py --write-all")

    config = load_campaign_config()
    cell = next((c for c in config.get("cells", []) if c["id"] == cell_id), None)
    if cell is None:
        raise ValueError(f"Celda desconocida en campaign_config.json: {cell_id!r}")

    model_paths = _resolve_model_paths(models_dir, model_names)
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

    prob_rows: List[np.ndarray] = []
    base_records: Optional[List[Dict[str, Any]]] = None
    y_true: Optional[np.ndarray] = None
    model_labels: List[str] = []

    for mp in model_paths:
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

    assert base_records is not None and y_true is not None
    prob_matrix = np.stack(prob_rows, axis=0)
    p_mean = prob_matrix.mean(axis=0)
    y_pred = _ensemble_predict(
        prob_matrix,
        rule,
        threshold,
        cascade_low=cascade_low,
        cascade_high=cascade_high,
    )
    metrics = _binary_metrics(y_true, y_pred)

    all_rows: List[Dict[str, Any]] = []
    for i, (base, ex) in enumerate(zip(base_records, examples)):
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

        paths = example_export_paths(ex)
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
            "threshold": threshold,
            "rule": rule,
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
    else:
        selected = all_rows

    reports_dir = arts["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_tag = "|".join(m.replace(".pt", "") for m in model_labels)
    csv_path = reports_dir / f"{split}_ensemble_fp_{rule}_{model_tag}_t{threshold:.2f}.csv".replace("|", "+")

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

    # Lista plana de vídeos (ruta completa por línea)
    video_list_path = csv_path.with_suffix(".videos.txt")
    with open(video_list_path, "w", encoding="utf-8") as f:
        for r in selected:
            video = r.get("clip_video_path") or r.get("clip_dir") or ""
            if video:
                f.write(f"{video}\n")

    summary_path = reports_dir / f"{split}_ensemble_fp_summary.json"
    summary = {
        "cell_id": cell_id,
        "split": split,
        "models": [str(p) for p in model_paths],
        "rule": rule,
        "threshold": threshold,
        "cascade_low": cascade_low,
        "cascade_high": cascade_high,
        "metrics": metrics,
        "pool_info": pool_info,
        "exported_rows": len(selected),
        "outcomes_filter": outcomes,
        "csv_path": str(csv_path.resolve()),
        "video_list_path": str(video_list_path.resolve()),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    fp_export_dir: Optional[Path] = None
    if export_videos and selected:
        fp_export_dir = arts["fp_clips_dir"] / f"ensemble_{rule}_t{threshold:.2f}"
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
        "summary_path": summary_path,
        "metrics": metrics,
        "selected": selected,
        "fp_export_dir": fp_export_dir,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Exporta FP (y opcionalmente FN) de un ensemble binario de campaña"
    )
    ap.add_argument("--cell", type=str, default="bin_full_hardened")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["modelo_06", "modelo_14"],
        help="Nombres o rutas .pt (default: modelo_06 modelo_14)",
    )
    ap.add_argument("--rule", choices=["mean", "and", "or", "cascade"], default="mean")
    ap.add_argument("--threshold", type=float, default=0.68)
    ap.add_argument("--cascade-low", type=float, default=0.4)
    ap.add_argument("--cascade-high", type=float, default=0.55)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--single-user-only", action="store_true", default=None)
    ap.add_argument("--no-single-user-only", action="store_true")
    ap.add_argument(
        "--outcomes",
        choices=["fp", "errors", "all"],
        default="fp",
        help="Qué filas exportar (default: solo FP)",
    )
    ap.add_argument("--export-videos", action="store_true", help="Symlink clip.mp4 en fp_clips/")
    ap.add_argument("--copy-videos", action="store_true", help="Copiar vídeos en lugar de symlink")
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
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    m = result["metrics"]
    print(f"\n=== Ensemble {args.rule} @ {args.threshold} — split {args.split} ===")
    print(f"Modelos: {args.models}")
    print(f"Celda: {args.cell}")
    print(
        f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} | "
        f"F1={m['f1_pct']:.1f}% Rec={m['recall_pct']:.1f}% FP={m['fp_rate_pct']:.2f}%"
    )
    print(f"\nExportado ({args.outcomes}): {len(result['selected'])} filas")
    print(f"CSV: {result['csv_path'].resolve()}")
    print(f"Lista vídeos (ruta completa/línea): {result['video_list_path'].resolve()}")
    print(f"Resumen JSON: {result['summary_path'].resolve()}")
    if result["fp_export_dir"]:
        print(f"Vídeos FP: {result['fp_export_dir'].resolve()}")

    if result["selected"] and args.outcomes == "fp":
        print("\n--- Falsos positivos (ruta completa al vídeo) ---")
        for r in sorted(result["selected"], key=lambda x: -float(x["p_mean"])):
            video = r.get("clip_video_path") or r.get("clip_dir") or "?"
            probs = " ".join(f"{k}={v}" for k, v in r.items() if k.startswith("p_modelo"))
            print(
                f"\n  cat={r['folder_category']} clip={r.get('clip_name','?')} "
                f"p_mean={r['p_mean']:.3f} {probs}"
            )
            print(f"  vídeo: {video}")
            print(f"  carpeta: {r.get('clip_dir', '')}")
            print(f"  uid: {r.get('uid', '')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
