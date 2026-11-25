#!/usr/bin/env python3
"""Preflight window_campaign — informe legible + plan JSON + estimación GPU/CPU."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

WINDOW_DIR = Path(__file__).resolve().parent
TRAINING_DIR = WINDOW_DIR.parent
for p in (TRAINING_DIR, WINDOW_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_config import EXPERIMENTS  # type: ignore
from train_model_operations import (  # type: ignore
    collect_debug_examples,
    collect_examples,
    get_data_result_roots,
    make_binary_examples,
)
from training_time_estimate import estimate_all_experiments, fmt_duration  # type: ignore

from core_labels import build_core_index_for_examples, load_core_index
from preflight_stats import build_preflight_stats
from window_paths import WINDOW_ROOT, load_window_config, window_plan_path


# ─── ANSI (si TTY) ───
def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(s: str) -> str:
    return _c("1", s)


def cyan(s: str) -> str:
    return _c("36", s)


def yellow(s: str) -> str:
    return _c("33", s)


def green(s: str) -> str:
    return _c("32", s)


def _hline(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _section(title: str) -> None:
    print()
    _hline("═")
    print(bold(f"  {title}"))
    _hline("─")


def _resolve_data_root(cli: str | None) -> Path | None:
    if cli:
        return Path(cli).expanduser().resolve()
    roots = get_data_result_roots(None)
    return roots[0] if roots else None


def _core_path(cfg: dict) -> Path:
    pl = cfg.get("positive_labeling") or {}
    p = Path(pl.get("core_index_path", "artifacts/core_index.json"))
    return p if p.is_absolute() else WINDOW_ROOT / p


def main() -> None:
    ap = argparse.ArgumentParser(description="Preflight window_campaign (solo binario)")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--write-plan", action="store_true")
    ap.add_argument("--cell-id", type=str, default=None)
    ap.add_argument("--debug-clips", type=int, default=0, help="Si >0, estadísticas sobre N clips (modo debug)")
    ap.add_argument(
        "--experiments",
        type=int,
        nargs="*",
        default=None,
        help="IDs para estimación de tiempo (default: experiment_benchmark en config)",
    )
    args = ap.parse_args()

    cfg = load_window_config()
    cell_id = args.cell_id or cfg.get("default_cell_id", "win_bin_full_baseline")
    data_root = _resolve_data_root(args.data_root)
    if data_root is None:
        raise SystemExit("Indica --data-root o configura data_result en experiments/config.py")

    pose_source = cfg.get("pose_source", "full")
    single_user = bool(cfg.get("single_user_only", True))
    rob = cfg.get("robbery_classes", [6, 14])

    _section("WINDOW CAMPAIGN — PREFLIGHT (binario únicamente)")
    print(f"  data_root     {cyan(str(data_root))}")
    print(f"  cell_id       {cell_id}")
    print(f"  pose_source   {pose_source} | mask {cfg.get('mask_policy')}")
    print(f"  positivos     carpetas {rob}")

    if args.debug_clips > 0:
        examples = collect_debug_examples(
            n=args.debug_clips,
            pose_source=pose_source,
            single_user_only=single_user,
            data_root=data_root,
        )
        examples = make_binary_examples(examples, positive_classes=rob)
        print(yellow(f"  [DEBUG] Solo {len(examples)} clips para el informe"))
    else:
        examples = collect_examples(
            pose_source=pose_source,
            single_user_only=single_user,
            data_root=data_root,
        )
        examples = make_binary_examples(examples, positive_classes=rob)

    core_path = _core_path(cfg)
    if (cfg.get("positive_labeling") or {}).get("mode") == "core_aware":
        if not core_path.is_file():
            build_core_index_for_examples(examples, core_path, rob)
        core_index = load_core_index(core_path)
    else:
        core_index = {}

    stats = build_preflight_stats(examples, cfg, core_index=core_index, data_root_str=str(data_root))

    _section("1) Clips (.npy por acción)")
    c = stats["clips"]
    print(f"  Total clips              {bold(str(c['total']))}")
    print(f"  Split train / val / test {c['train']} / {c['val']} / {c['test']}")
    print(f"  Clips con ≥2 chunks      {green(str(c['with_multiple_chunks']))}  (segmentación temporal)")
    print(f"  Clips con 1 sola ventana {c['with_single_chunk_only']}")

    _section("2) Chunks base (ventanas 96f, stride 48)")
    ch = stats["chunks_base"]
    print(f"  Total chunks (todos)     {bold(str(ch['total_all_splits']))}")
    print(f"  Train / val / test       {ch['train']} / {ch['val']} / {ch['test']}")
    lbl = ch.get("by_label") or {}
    print(f"  Por etiqueta binaria     neg(0)={lbl.get(0, 0)} | pos(1)={lbl.get(1, 0)}")

    _section("3) Chunks en train — core-aware + augment (cota superior)")
    seg = stats["chunks_segmented_train"]
    print(f"  Entrenables (≠ ignore)   {bold(str(seg['trainable_after_core']))}")
    print(f"  ignore (solo de pie 6/14) {seg['ignored_stand_only']}")
    print(f"  pos_core / partial / neg  {seg['pos_core']} / {seg['pos_partial']} / {seg['neg']}")
    print(
        f"  Filas train (1+{cfg.get('synthetic_variants', {}).get('extra_manifest_views_per_clip', 1)} syn) "
        f"≤ {seg['expanded_upper_bound_syn']}"
    )
    print(f"  Muestreo ≈ filas/época     {cyan(str(seg['sampler_rows_per_epoch_est']))}")

    exp_ids = args.experiments
    if not exp_ids:
        exp_ids = list(cfg.get("experiment_benchmark") or cfg.get("experiment_stages", {}).get("stage2", [48, 49]))
    train_rows = int(stats["train_rows_per_epoch"])
    val_rows = max(1, int(stats["val_rows_per_epoch"]))
    epochs_cfg = int((cfg.get("train_defaults") or {}).get("epochs", 25))

    est = estimate_all_experiments(
        train_rows=train_rows,
        val_rows=val_rows,
        experiment_ids=exp_ids,
    )

    _section(f"4) Arquitecturas a entrenar ({len(exp_ids)} modelos)")
    for eid in exp_ids:
        if 1 <= eid <= len(EXPERIMENTS):
            ec = EXPERIMENTS[eid - 1]
            ep = epochs_cfg if cfg.get("train_defaults", {}).get("epochs") else ec.get("epochs", 25)
            print(
                f"  exp {eid:02d}  {ec.get('arch'):16s}  seq={cfg.get('window_len', 96)}  "
                f"epochs≈{ep}  batch={ec.get('batch_size', 32)}"
            )

    _section("5) Tiempo estimado (todos los experimentos, en serie)")
    # Rescale: estimate uses EXPERIMENTS epochs; override with train_defaults.epochs
    gpu_t = 0.0
    cpu_t = 0.0
    for row in est.get("per_experiment", []):
        ratio = epochs_cfg / max(1, int(row.get("epochs", 25)))
        gpu_t += float(row.get("gpu_seconds", 0)) * ratio
        cpu_t += float(row.get("cpu_seconds", 0)) * ratio

    on_gpu = est.get("cuda_available", False)
    print(f"  Filas/época train/val     {train_rows} / {val_rows}")
    print(f"  CUDA disponible           {green('sí') if on_gpu else yellow('no')}")
    print(f"  {bold('GPU')} (suma {len(exp_ids)} modelos, {epochs_cfg} épocas c/u)  {cyan(fmt_duration(gpu_t))}")
    print(f"  {bold('CPU')} (misma batería, referencia)        {fmt_duration(cpu_t)}")
    if on_gpu:
        print(f"  {bold('Recomendado')}: lanzar en GPU y dormir {cyan(fmt_duration(gpu_t))}")
    else:
        print(f"  {bold('Recomendado')}: usar máquina con CUDA; en CPU sería {yellow(fmt_duration(cpu_t))}")

    _hline("═")
    print()

    if args.write_plan:
        plan = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cell_id": cell_id,
            "task": "binary",
            "experiment_ids": exp_ids,
            "stats": stats,
            "time_estimate": {
                "gpu_seconds": gpu_t,
                "cpu_seconds": cpu_t,
                "gpu_human": fmt_duration(gpu_t),
                "cpu_human": fmt_duration(cpu_t),
                "cuda_available": on_gpu,
                "epochs_per_model": epochs_cfg,
            },
            "config_snapshot": {
                "window_len": cfg.get("window_len"),
                "stride": cfg.get("stride"),
                "synthetic_variants": cfg.get("synthetic_variants"),
            },
        }
        out = window_plan_path(cell_id)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        print(f"Plan → {out}")


if __name__ == "__main__":
    main()