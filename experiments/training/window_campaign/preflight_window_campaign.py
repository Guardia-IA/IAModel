#!/usr/bin/env python3
"""Preflight window_campaign — informe legible + plan JSON + estimación GPU/CPU."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Feedback inmediato (imports pesados — PyTorch — pueden tardar)
print("[preflight] Arranque… (importando dependencias; la 1ª vez puede tardar ~1 min)", flush=True)

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


def _say(msg: str = "", *, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stdout
    if msg:
        print(msg, flush=True, file=stream)
    else:
        print(flush=True, file=stream)


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
    _say(char * width)


def _section(title: str) -> None:
    _say()
    _hline("═")
    _say(bold(f"  {title}"))
    _hline("─")


def _resolve_data_root(cli: str | None) -> Path | None:
    if cli:
        return Path(cli).expanduser().resolve()
    roots = get_data_result_roots(None)
    return roots[0] if roots else None


def _validate_data_root(data_root: Path, *, cli_path: str | None) -> None:
    _say(f"[preflight] Comprobando data_root: {data_root}")
    if not data_root.exists():
        hint = (
            f"La ruta no existe: {data_root}\n"
            f"  Usa --data-root con la carpeta que contiene categorías (p. ej. …/6_robo/, …/3_caminar/).\n"
        )
        if cli_path:
            hint += f"  Recibido: --data-root {cli_path!r}\n"
        else:
            hint += "  No pasaste --data-root; se tomó la ruta por defecto de experiments/config.py.\n"
        raise SystemExit(hint)
    if not data_root.is_dir():
        raise SystemExit(f"data_root no es una carpeta: {data_root}")
    try:
        children = list(data_root.iterdir())
    except OSError as e:
        raise SystemExit(f"No se puede leer data_root ({data_root}): {e}") from e
    cat_dirs = [p for p in children if p.is_dir()]
    with_meta = 0
    sample_cats: list[str] = []
    for cat in sorted(cat_dirs)[:30]:
        sample_cats.append(cat.name)
        try:
            for clip in cat.iterdir():
                if clip.is_dir() and (clip / "meta.json").is_file():
                    with_meta += 1
                    break
        except OSError:
            continue
    _say(
        f"[preflight] OK: carpeta accesible | subcarpetas={len(cat_dirs)} | "
        f"categorías con al menos 1 clip≈{with_meta} (muestra hasta 30 cats)"
    )
    if sample_cats:
        _say(f"[preflight] Ejemplos de nombres: {', '.join(sample_cats[:8])}")
    if not cat_dirs:
        raise SystemExit(
            f"data_root vacío o sin subcarpetas de categoría: {data_root}\n"
            "  Esperado: …/data_result/6_robo/<clip>/meta.json"
        )
    if with_meta == 0:
        _say(
            yellow(
                "[preflight] AVISO: no vi ningún clip con meta.json en las primeras "
                "categorías; puede que la ruta apunte a un nivel incorrecto."
            )
        )


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

    _say("[preflight] Leyendo window_config.json …")
    cfg = load_window_config()
    cell_id = args.cell_id or cfg.get("default_cell_id", "win_bin_full_baseline")
    data_root = _resolve_data_root(args.data_root)
    if data_root is None:
        raise SystemExit(
            "Indica --data-root /ruta/data_result o configura data_result en experiments/config.py"
        )
    _validate_data_root(data_root, cli_path=args.data_root)

    pose_source = cfg.get("pose_source", "full")
    single_user = bool(cfg.get("single_user_only", True))
    rob = cfg.get("robbery_classes", [6, 14])

    _section("WINDOW CAMPAIGN — PREFLIGHT (binario únicamente)")
    _say(f"  data_root     {cyan(str(data_root))}")
    _say(f"  cell_id       {cell_id}")
    _say(f"  pose_source   {pose_source} | mask {cfg.get('mask_policy')}")
    _say(f"  positivos     carpetas {rob}")

    if args.debug_clips > 0:
        _say(f"[preflight] Modo debug: recolectando ≤{args.debug_clips} clips …")
        examples = collect_debug_examples(
            n=args.debug_clips,
            pose_source=pose_source,
            single_user_only=single_user,
            data_root=data_root,
        )
        examples = make_binary_examples(examples, positive_classes=rob)
        _say(yellow(f"  [DEBUG] {len(examples)} clips en el informe"))
    else:
        cf = cfg.get("clip_filters") or {}
        _say("[preflight] Recorriendo corpus completo (puede tardar; ver líneas [progreso]) …")
        examples = collect_examples(
            pose_source=pose_source,
            single_user_only=single_user,
            data_root=data_root,
            min_clip_seconds=float(cf.get("min_clip_seconds", 0)),
            min_valid_frames=int(cf.get("min_valid_frames", 0)),
            min_valid_pct=float(cf.get("min_valid_pct", 0)),
            max_occlusion_ratio=float(cf.get("max_occlusion_ratio", 90)),
        )
        examples = make_binary_examples(examples, positive_classes=rob)
        _say(f"[preflight] Corpus: {len(examples)} ejemplos (usuarios/clips) listos")

    core_path = _core_path(cfg)
    if (cfg.get("positive_labeling") or {}).get("mode") == "core_aware":
        if not core_path.is_file():
            _say(f"[preflight] Construyendo core_index (6/14) → {core_path} …")
            build_core_index_for_examples(examples, core_path, rob)
        else:
            _say(f"[preflight] Reutilizando core_index: {core_path}")
        core_index = load_core_index(core_path)
    else:
        core_index = {}

    _say("[preflight] Ventanas + quality gate + estadísticas …")
    stats = build_preflight_stats(
        examples,
        cfg,
        core_index=core_index,
        data_root_str=str(data_root),
        progress=True,
    )
    _say("[preflight] Estadísticas listas.")

    _section("1) Clips (.npy por acción)")
    c = stats["clips"]
    _say(f"  Total clips              {bold(str(c['total']))}")
    _say(f"  Split train / val / test {c['train']} / {c['val']} / {c['test']}")
    _say(f"  Clips con ≥2 chunks      {green(str(c['with_multiple_chunks']))}  (segmentación temporal)")
    _say(f"  Clips con 1 sola ventana {c['with_single_chunk_only']}")

    _section("2) Chunks base (ventanas 96f, stride 48)")
    ch = stats["chunks_base"]
    _say(f"  Total chunks (todos)     {bold(str(ch['total_all_splits']))}")
    _say(f"  Train / val / test       {ch['train']} / {ch['val']} / {ch['test']}")
    lbl = ch.get("by_label") or {}
    _say(f"  Por etiqueta binaria     neg(0)={lbl.get(0, 0)} | pos(1)={lbl.get(1, 0)}")

    qg = stats.get("quality_gate") or {}
    _section("2b) Filtro calidad por chunk (esqueleto)")
    _say(f"  Chunks antes del filtro        {qg.get('chunks_before', '?')}")
    _say(
        f"  Descartados train / val / test {qg.get('dropped_train', 0)} / "
        f"{qg.get('dropped_val', 0)} / {qg.get('dropped_test', 0)}"
    )
    rc = qg.get("reason_counts_train") or {}
    if rc:
        _say(f"  Motivos (train)                {rc}")
    cf = cfg.get("clip_filters") or {}
    _say(
        f"  Clips: min duración            {cf.get('min_clip_seconds', 0)} s "
        f"(collect_examples; último chunk corto permitido)"
    )
    qcfg = cfg.get("quality_gate") or {}
    _say(
        f"  Chunk: min contenido           {qcfg.get('min_content_seconds')} s "
        f"(padded tail ≥ {qcfg.get('min_padded_content_seconds')} s)"
    )

    _section("3) Chunks en train — core-aware + augment (cota superior)")
    seg = stats["chunks_segmented_train"]
    _say(f"  Entrenables (≠ ignore)   {bold(str(seg['trainable_after_core']))}")
    _say(f"  ignore (solo de pie 6/14) {seg['ignored_stand_only']}")
    _say(f"  pos_core / partial / neg  {seg['pos_core']} / {seg['pos_partial']} / {seg['neg']}")
    _say(
        f"  Filas train (1+{cfg.get('synthetic_variants', {}).get('extra_manifest_views_per_clip', 1)} syn) "
        f"≤ {seg['expanded_upper_bound_syn']}"
    )
    _say(f"  Muestreo ≈ filas/época     {cyan(str(seg['sampler_rows_per_epoch_est']))}")

    exp_ids = args.experiments
    if not exp_ids:
        exp_ids = list(cfg.get("experiment_benchmark") or cfg.get("experiment_stages", {}).get("stage2", [48, 49]))
    train_rows = int(stats["train_rows_per_epoch"])
    val_rows = max(1, int(stats["val_rows_per_epoch"]))
    epochs_cfg = int((cfg.get("train_defaults") or {}).get("epochs", 25))

    _say("[preflight] Estimación de tiempos por arquitectura …")
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
            _say(
                f"  exp {eid:02d}  {ec.get('arch'):16s}  seq={cfg.get('window_len', 96)}  "
                f"epochs≈{ep}  batch={ec.get('batch_size', 32)}"
            )

    _section("5) Tiempo estimado (todos los experimentos, en serie)")
    gpu_t = 0.0
    cpu_t = 0.0
    for row in est.get("per_experiment", []):
        ratio = epochs_cfg / max(1, int(row.get("epochs", 25)))
        gpu_t += float(row.get("gpu_seconds", 0)) * ratio
        cpu_t += float(row.get("cpu_seconds", 0)) * ratio

    on_gpu = est.get("cuda_available", False)
    _say(f"  Filas/época train/val     {train_rows} / {val_rows}")
    _say(f"  CUDA disponible           {green('sí') if on_gpu else yellow('no')}")
    _say(f"  {bold('GPU')} (suma {len(exp_ids)} modelos, {epochs_cfg} épocas c/u)  {cyan(fmt_duration(gpu_t))}")
    _say(f"  {bold('CPU')} (misma batería, referencia)        {fmt_duration(cpu_t)}")
    if on_gpu:
        _say(f"  {bold('Recomendado')}: lanzar en GPU y dormir {cyan(fmt_duration(gpu_t))}")
    else:
        _say(f"  {bold('Recomendado')}: usar máquina con CUDA; en CPU sería {yellow(fmt_duration(cpu_t))}")

    _hline("═")
    _say()

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
        _say(f"Plan → {out}")

    _say("[preflight] Fin.")


if __name__ == "__main__":
    main()