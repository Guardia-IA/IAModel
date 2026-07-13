#!/usr/bin/env python3
"""
Evaluación unificada del pipeline FP: modelos → ensemble → verificador → heurísticas.

Tras train, un solo comando produce F1/FN/FP finales por modelo, ensemble y pipeline 3 etapas.

Uso:
  cd experiments/training/campaign
  RUN_ID=fp_pipeline_v1 python evaluate_fp_pipeline.py --all

  # Si ya corriste evaluate_campaign.py (eval):
  RUN_ID=fp_pipeline_v1 python evaluate_fp_pipeline.py --skip-model-eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

from campaign_paths import ensure_cell_dirs, filter_cells, load_merged_campaign_config, master_reports_dir
from evaluate_campaign import evaluate_cell, load_best_ensemble_spec
from export_ensemble_fp import run_ensemble_export
from merge_verifier_probs import merge_verifier_csv
from pose_robbery_heuristics import run_pipeline_sweep


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _metrics_row(label: str, m: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = {
        "label": label,
        "f1_pct": round(float(m.get("f1_pct", 0)), 2),
        "recall_pct": round(float(m.get("recall_pct", 0)), 2),
        "fp_rate_pct": round(float(m.get("fp_rate_pct", 0)), 4),
        "fp": int(m.get("fp", 0)),
        "fn": int(m.get("fn", 0)),
        "tp": int(m.get("tp", 0)),
        "tn": int(m.get("tn", 0)),
    }
    if extra:
        row.update(extra)
    return row


def _print_summary_table(rows: List[Dict[str, Any]], *, split: str, run_id: str) -> None:
    print(f"\n{'=' * 88}")
    print(f"Evaluación pipeline — run_id={run_id} split={split}")
    print(f"{'=' * 88}")
    print(f"{'Etapa':<28} {'F1%':>7} {'Rec%':>7} {'FP':>4} {'FN':>4} {'FP%':>8}  Detalle")
    print("-" * 88)
    for row in rows:
        detail = row.get("detail", "")
        print(
            f"{row['label']:<28} "
            f"{row['f1_pct']:>7.1f} "
            f"{row['recall_pct']:>7.1f} "
            f"{row['fp']:>4} "
            f"{row['fn']:>4} "
            f"{row['fp_rate_pct']:>7.2f}%  "
            f"{detail}"
        )
    print(f"{'=' * 88}\n")


def run_model_eval(
    *,
    config: Dict[str, Any],
    run_id: str,
    split: str,
    export_fp_videos: bool,
) -> List[Dict[str, Any]]:
    cells = filter_cells(config, None)
    if not cells:
        raise RuntimeError("campaign_config_fp_pipeline.json sin celdas")
    summary: List[Dict[str, Any]] = []
    print(f"\n=== Eval modelos — {len(cells)} celdas, split={split}, run_id={run_id} ===\n")
    for cell in cells:
        row = evaluate_cell(
            cell,
            config,
            split=split,
            export_fp=export_fp_videos,
            run_id=run_id,
        )
        summary.append(row)
    master = master_reports_dir(run_id) / f"{split}_eval_cells.json"
    master.parent.mkdir(parents=True, exist_ok=True)
    with open(master, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nResumen eval modelos: {master}")
    return summary


def build_pipeline_summary(
    *,
    run_id: str,
    split: str,
    stage1_cell: str,
    ensemble_export: Dict[str, Any],
    merge_result: Dict[str, Any],
    pipeline_result: Dict[str, Any],
) -> Dict[str, Any]:
    stage1_reports = ensure_cell_dirs(stage1_cell, run_id=run_id)["reports_dir"]
    verifier_reports = ensure_cell_dirs("bin_verifier_234", run_id=run_id)["reports_dir"]

    per_model = _load_json(stage1_reports / f"{split}_per_model_best.json") or []
    best_ens = _load_json(stage1_reports / f"{split}_best_ensemble.json") or {}
    verifier_per_model = _load_json(verifier_reports / f"{split}_per_model_best.json") or []
    pipeline_best = pipeline_result.get("best") or {}

    comparison: List[Dict[str, Any]] = []

    for pm in sorted(per_model, key=lambda r: -float(r.get("f1_pct") or 0)):
        comparison.append(
            _metrics_row(
                f"modelo {pm.get('model', '?')}",
                {
                    "f1_pct": pm.get("f1_pct"),
                    "recall_pct": pm.get("recall_pct"),
                    "fp_rate_pct": pm.get("fp_rate_pct"),
                    "fp": pm.get("fp_count"),
                    "fn": pm.get("fn_count"),
                },
                {
                    "stage": "1_model",
                    "threshold": pm.get("threshold"),
                    "detail": f"umbral={pm.get('threshold')}",
                },
            )
        )

    ens_metrics = ensemble_export.get("metrics") or {}
    ens_models = "|".join(str(m).replace(".pt", "") for m in ensemble_export.get("models", []))
    comparison.append(
        _metrics_row(
            f"ensemble {ensemble_export.get('rule', 'mean')}",
            ens_metrics,
            {
                "stage": "1_ensemble",
                "models": ensemble_export.get("models"),
                "rule": ensemble_export.get("rule"),
                "threshold": ensemble_export.get("threshold"),
                "detail": f"{ens_models} @ {ensemble_export.get('threshold')}",
            },
        )
    )

    if best_ens:
        comparison.append(
            {
                "label": "best_ensemble (grid eval)",
                "stage": "1_ensemble_grid",
                "f1_pct": float(best_ens.get("f1_pct") or 0),
                "recall_pct": float(best_ens.get("recall_pct") or 0),
                "fp_rate_pct": float(best_ens.get("fp_rate_pct") or 0),
                "fp": int(best_ens.get("fp") or 0),
                "fn": int(best_ens.get("fn") or 0),
                "models": best_ens.get("models"),
                "rule": best_ens.get("rule"),
                "threshold": best_ens.get("threshold"),
                "detail": f"{'|'.join(best_ens.get('models') or [])} @ {best_ens.get('threshold')}",
            }
        )

    if verifier_per_model:
        vp = max(verifier_per_model, key=lambda r: float(r.get("f1_pct") or 0))
        comparison.append(
            _metrics_row(
                f"verificador {vp.get('model', '?')}",
                {
                    "f1_pct": vp.get("f1_pct"),
                    "recall_pct": vp.get("recall_pct"),
                    "fp_rate_pct": vp.get("fp_rate_pct"),
                    "fp": vp.get("fp_count"),
                    "fn": vp.get("fn_count"),
                },
                {
                    "stage": "2_verifier_only",
                    "threshold": vp.get("threshold"),
                    "detail": "solo celda bin_verifier_234 (no pipeline)",
                },
            )
        )

    if pipeline_best:
        comparison.append(
            _metrics_row(
                "pipeline 3 etapas (best sweep)",
                pipeline_best,
                {
                    "stage": "3_full_pipeline",
                    "rule": pipeline_best.get("rule"),
                    "t_stage1": pipeline_best.get("t_stage1"),
                    "t_kin": pipeline_best.get("t_kin"),
                    "t_verifier": pipeline_best.get("t_verifier"),
                    "detail": (
                        f"t1={pipeline_best.get('t_stage1')} "
                        f"tk={pipeline_best.get('t_kin')} "
                        f"tv={pipeline_best.get('t_verifier')}"
                    ),
                },
            )
        )

    out_dir = stage1_reports / "fp_pipeline"
    summary_path = out_dir / f"{split}_pipeline_eval_summary.json"
    payload = {
        "run_id": run_id,
        "split": split,
        "stage1_cell": stage1_cell,
        "comparison": comparison,
        "artifacts": {
            "ensemble_csv": str(ensemble_export.get("csv_path")),
            "ensemble_with_verifier_csv": str(merge_result.get("out")),
            "verifier_model": str(merge_result.get("verifier_model")),
            "pipeline_sweep_csv": pipeline_result.get("sweep_csv"),
            "pipeline_best_summary": str(out_dir / "pipeline_best_summary.json"),
            "heuristics_features_csv": pipeline_result.get("heuristics_csv"),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    _print_summary_table(comparison, split=split, run_id=run_id)
    print(f"Resumen JSON: {summary_path}")
    return payload


def run_fp_pipeline_eval(
    *,
    run_id: str,
    split: str = "val",
    config_path: Optional[Path] = None,
    stage1_cell: str = "bin_filtered_hardened",
    verifier_cell: str = "bin_verifier_234",
    skip_model_eval: bool = False,
    export_fp_videos: bool = False,
    model_names: Optional[List[str]] = None,
    ensemble_rule: Optional[str] = None,
    ensemble_threshold: Optional[float] = None,
    pipeline_rule: str = "mean",
    t_verifier: float = 0.55,
    min_conceal_sustain: float = 0.08,
    require_kin: bool = True,
    t_stage1_grid: Optional[List[float]] = None,
    t_kin_grid: Optional[List[float]] = None,
) -> Dict[str, Any]:
    cfg_path = config_path or (CAMPAIGN_DIR / "campaign_config_fp_pipeline.json")
    config = load_merged_campaign_config(path=cfg_path)

    if not skip_model_eval:
        run_model_eval(
            config=config,
            run_id=run_id,
            split=split,
            export_fp_videos=export_fp_videos,
        )
    else:
        stage1_reports = ensure_cell_dirs(stage1_cell, run_id=run_id)["reports_dir"]
        if not (stage1_reports / f"{split}_best_ensemble.json").is_file():
            raise FileNotFoundError(
                f"No hay eval previo en {stage1_reports}. "
                "Ejecuta sin --skip-model-eval o corre ./run_fp_pipeline.sh eval"
            )

    print(f"\n=== Export ensemble etapa 1 (split completo) ===", flush=True)
    try:
        ensemble_export = run_ensemble_export(
        cell_id=stage1_cell,
        model_names=model_names,
        split=split,
        rule=ensemble_rule,
        threshold=ensemble_threshold,
        outcomes="all",
        run_id=run_id,
        )
    except Exception as exc:
        print(f"ERROR en export ensemble etapa 1: {exc}", file=sys.stderr, flush=True)
        raise

    print(f"\n=== Merge verificador etapa 2 ===", flush=True)
    merge_result = merge_verifier_csv(
        ensemble_export["csv_path"],
        run_id=run_id,
        split=split,
        verifier_cell=verifier_cell,
    )
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in merge_result.items()}, indent=2))

    out_dir = ensure_cell_dirs(stage1_cell, run_id=run_id)["reports_dir"] / "fp_pipeline"
    t1_grid = t_stage1_grid or [0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78]
    tk_grid = t_kin_grid or [0.30, 0.35, 0.40, 0.42, 0.45, 0.50, 0.55]

    print(f"\n=== Pipeline sweep etapas 1+2+3 (heurísticas H1–H12) ===")
    pipeline_result = run_pipeline_sweep(
        merge_result["out"],
        out_dir=out_dir,
        t_stage1_grid=t1_grid,
        t_kin_grid=tk_grid,
        rule=pipeline_rule,
        t_verifier=t_verifier,
        min_conceal_sustain=min_conceal_sustain,
        require_kin=require_kin,
    )

    return build_pipeline_summary(
        run_id=run_id,
        split=split,
        stage1_cell=stage1_cell,
        ensemble_export=ensemble_export,
        merge_result=merge_result,
        pipeline_result=pipeline_result,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluación unificada pipeline FP (modelos + ensemble + heurísticas)")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--stage1-cell", default="bin_filtered_hardened")
    ap.add_argument("--verifier-cell", default="bin_verifier_234")
    ap.add_argument("--skip-model-eval", action="store_true", help="Reutiliza eval ya hecho (evaluate_campaign.py)")
    ap.add_argument("--export-fp-videos", action="store_true")
    ap.add_argument("--models", nargs="*", default=None, help="Ensemble etapa 1 (default: best_ensemble.json)")
    ap.add_argument("--rule", choices=["mean", "and", "cascade"], default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--pipeline-rule", choices=["mean", "and", "cascade"], default="mean")
    ap.add_argument("--t-verifier", type=float, default=0.55)
    ap.add_argument("--min-conceal-sustain", type=float, default=0.08)
    ap.add_argument("--no-require-kin", action="store_true")
    args = ap.parse_args()

    try:
        run_fp_pipeline_eval(
            run_id=str(args.run_id).strip(),
            split=args.split,
            config_path=args.config,
            stage1_cell=args.stage1_cell,
            verifier_cell=args.verifier_cell,
            skip_model_eval=args.skip_model_eval,
            export_fp_videos=args.export_fp_videos,
            model_names=args.models,
            ensemble_rule=args.rule,
            ensemble_threshold=args.threshold,
            pipeline_rule=args.pipeline_rule,
            t_verifier=args.t_verifier,
            min_conceal_sustain=args.min_conceal_sustain,
            require_kin=not args.no_require_kin,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
