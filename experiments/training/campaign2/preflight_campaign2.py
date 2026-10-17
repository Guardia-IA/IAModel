#!/usr/bin/env python3
"""
Preflight Campaign2 (Sprint 1): inventario, caché 12 Hz (mmap), cola de jobs opcional.

NO entrena modelos Kanvis. El entrenamiento real llega en Sprint 2 (train_v2 + jobs P0/P1).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
for p in (TRAINING_DIR, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import (  # noqa: E402
    job_queue_path,
    load_campaign2_config,
    read_current_run_id,
    resolve_data_roots,
    run_layout,
    run_meta_path,
)
from dataset_cache import (  # noqa: E402
    build_pose_cache,
    collect_clip_records,
    subsample_records_stratified,
    swap_pose_source_on_records,
)
from gpu_scheduler import JobSpec  # noqa: E402
from time_estimate import build_time_estimate_report, estimate_job_queue_wall  # noqa: E402
from campaign2_paths import parallel_recommendation_path  # noqa: E402
from training_time_estimate import fmt_duration  # noqa: E402
from preflight_display import render_preflight_summary  # noqa: E402
from dataset_plan_summary import SPLIT_LABELS, SPLIT_ORDER, summarize_training_dataset  # noqa: E402
from p0_jobs import build_p0_job_queue  # noqa: E402
from p1_jobs import build_p1_job_queue  # noqa: E402
from matrix_jobs import build_matrix_job_queue, cache_dir_for_pose, matrix_settings  # noqa: E402
from manifest_paths import write_run_meta  # noqa: E402
from fps_audit import audit_clip_timing  # noqa: E402
from hard_neg_manifest import (  # noqa: E402
    build_hard_neg_manifest,
    merge_external_manifest,
    write_hard_neg_manifest,
)

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


def _log_line(log_path: Optional[Path], msg: str) -> None:
    print(msg, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _split_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for e in entries:
        sp = e.get("split", "unknown")
        out.setdefault(sp, {"clips": 0, "robbery": 0})
        out[sp]["clips"] += 1
        if e.get("is_robbery"):
            out[sp]["robbery"] += 1
    return out


def _category_table(records: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter(int(r["label"]) for r in records)
    return {str(k): v for k, v in sorted(c.items())}


def _category_table_from_manifest(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter(int(e.get("label", -1)) for e in entries)
    return {str(k): v for k, v in sorted(c.items()) if k >= 0}


def _inventory_audits(
    records: List[Dict[str, Any]],
    layout: Dict[str, Path],
    cfg: Dict[str, Any],
    log_fn,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    fps_audit = audit_clip_timing(records)
    fps_path = layout["reports_dir"] / "fps_audit.json"
    with open(fps_path, "w", encoding="utf-8") as f:
        json.dump(fps_audit, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if fps_audit["n_issues"]:
        log_fn(
            f"  {YELLOW}FPS/meta: {fps_audit['n_issues']} clips con duración ≠ meta "
            f"→ {fps_path}{RESET}"
        )
    else:
        log_fn(
            f"  {GREEN}FPS/meta: sin discrepancias graves (±{fps_audit['tolerance_sec']}s){RESET}"
        )
    p1_cfg = cfg.get("p1_train") or {}
    hn_doc = build_hard_neg_manifest(
        records,
        hard_neg_classes=[int(x) for x in (p1_cfg.get("hard_neg_classes") or [2, 3, 4, 14])],
        robbery_class=int(cfg.get("robbery_class", 6)),
    )
    p3 = cfg.get("p3_hard_neg") or {}
    ext_hn = str(p3.get("external_manifest_path") or "").strip()
    if ext_hn:
        hn_doc = merge_external_manifest(hn_doc, Path(ext_hn))
    write_hard_neg_manifest(layout["reports_dir"] / "hard_neg_manifest.json", hn_doc)
    log_fn(f"  Hard neg manifest: {hn_doc['n_clips']} clips → reports/hard_neg_manifest.json")
    return fps_audit, hn_doc


def _cache_info_from_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    if not manifest_path.is_file():
        return None
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    poses = man.get("poses_mmap", "")
    try:
        sz = Path(poses).stat().st_size if poses and Path(poses).is_file() else 0
    except OSError:
        sz = 0
    return {
        "manifest": str(manifest_path),
        "num_clips": man.get("num_clips"),
        "total_frames": man.get("total_frames"),
        "bytes_on_disk": sz,
        "poses_mmap": poses,
        "reused": True,
    }


def build_smoke_job_queue(
    cfg: Dict[str, Any],
    *,
    manifest_path: Optional[Path],
    run_id: str,
    enable_smoke: bool,
) -> List[JobSpec]:
    n = int(cfg.get("sprint1_smoke_jobs", 0)) if enable_smoke else 0
    jobs: List[JobSpec] = []
    for i in range(n):
        jtype = "smoke_train" if i == 0 else "smoke"
        params: Dict[str, Any] = {}
        if jtype == "smoke_train":
            par = cfg.get("parallelism") or {}
            params = {
                "batch_size": int(par.get("probe_batch_size", 64)),
                "seq_len": int((cfg.get("temporal") or {}).get("seq_len", 96)),
                "feature_dim": int(par.get("probe_feature_dim", 62)),
                "steps": 8,
            }
        else:
            params = {"sleep_seconds": 0.5 + 0.1 * i}
        jobs.append(JobSpec(job_id=f"smoke_{i+1:02d}", job_type=jtype, params=params, priority=0))

    if manifest_path and manifest_path.is_file() and enable_smoke:
        jobs.append(
            JobSpec(
                job_id="cache_touch_01",
                job_type="cache_touch",
                params={"manifest_path": str(manifest_path)},
                priority=1,
            )
        )
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight Campaign2 (Sprint 1 — infra, no entrenamiento Kanvis)")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument(
        "--data-root",
        action="append",
        default=None,
        help="Carpeta data_result(s) o carpeta principal (escanea hijos datos*/data_result)",
    )
    ap.add_argument(
        "--pose-source",
        choices=("full", "filtered"),
        default=None,
        help="poses_full.npy vs poses.npy (default: campaign2_config pose_source). Legacy probaba ambas en celdas distintas.",
    )
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--max-clips", type=int, default=0, help="Limit clips (debug)")
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help="Añade jobs smoke al scheduler (probar paralelismo). No entrena el modelo de producción.",
    )
    ap.add_argument(
        "--p0",
        action="store_true",
        help="Sprint 2: encola eval P0 (#41/#48, P0.A clip + P0.B ventanas). Requiere rutas en p0_incumbent o env CAMPAIGN2_MODEL_*.",
    )
    ap.add_argument(
        "--p1",
        action="store_true",
        help="Sprint 3: encola entrenamiento train_v2 (matriz p1_job_matrix × semillas).",
    )
    ap.add_argument(
        "--p1-smoke",
        action="store_true",
        help="Un job P1 mínimo (2 épocas) para validar train_v2 + GPU.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Modo rápido: ~10 clips (debug.max_clips), P1 corto. Usa con --p1 o --matrix.",
    )
    ap.add_argument(
        "--matrix",
        action="store_true",
        help="Campaña matriz: full+filtered, P0 legacy + P1 v2, una cola (ver campaign_matrix en config).",
    )
    args = ap.parse_args()

    if args.debug and not args.p1 and not args.p1_smoke:
        args.p1 = True

    cfg = load_campaign2_config(Path(args.config) if args.config else None)
    if args.debug:
        dbg = cfg.get("debug") or {}
        if args.max_clips <= 0:
            args.max_clips = int(dbg.get("max_clips", 10))
    phase = str(cfg.get("phase") or "sprint1_infra")
    if args.debug:
        phase = "debug_p1"
    run_id = args.run_id or read_current_run_id()
    if not run_id:
        print("ERROR: sin RUN_ID. Ejecuta: ./run_campaign2.sh new-run", file=sys.stderr)
        return 2

    layout = run_layout(run_id)
    log_path = layout["logs_dir"] / "preflight.log"
    log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        _log_line(log_path, msg)

    log(f"{BOLD}{'=' * 72}{RESET}")
    log(f"{BOLD}Campaign2 PREFLIGHT — {phase}{RESET}")
    log(f"RUN_ID={CYAN}{run_id}{RESET}")
    if args.p1 or args.p1_smoke or phase.startswith("sprint3"):
        log(f"{CYAN}Sprint 3 P1: caché + cola entrenamiento train_v2 (ventanas 8s/12Hz).{RESET}")
    elif args.p0 or phase.startswith("sprint2"):
        log(f"{CYAN}Sprint 2 P0: inventario + caché + cola eval incumbent (no entrena v2 aún).{RESET}")
    else:
        log(f"{YELLOW}Sprint 1: inventario + caché 12 Hz. NO es entrenamiento del modelo Kanvis.{RESET}")
    log(f"Log: {log_path}")
    if args.debug:
        log(f"{MAGENTA}DEBUG: max_clips={args.max_clips} — pipeline rápido (métricas no representativas).{RESET}")
    log(f"{BOLD}{'=' * 72}{RESET}")

    data_roots = resolve_data_roots(cfg, args.data_root)
    log(f"\n{BOLD}1) Raíces de datos{RESET}")
    if not data_roots:
        log(f"  {YELLOW}WARN: sin data_roots. Usa --data-root o campaign2_config.json → data_roots{RESET}")
    else:
        for r in data_roots:
            log(f"  {CYAN}{r}{RESET}")

    temporal = cfg.get("temporal") or {}
    cache_cfg = cfg.get("cache") or {}
    split_cfg = cfg.get("split") or {}
    min_rob_eval = int(split_cfg.get("min_robbery_per_eval_split", 5))
    if args.debug:
        min_rob_eval = min(min_rob_eval, 1)

    records: List[Dict[str, Any]] = []
    fps_audit_report: Optional[Dict[str, Any]] = None
    hard_neg_summary: Optional[Dict[str, Any]] = None
    cache_result = None
    manifest_by_pose: Dict[str, Path] = {}
    t0 = time.perf_counter()
    matrix_jobs_built = False

    if (
        args.matrix
        and data_roots
        and not args.skip_cache
        and cache_cfg.get("enabled", True)
    ):
        ms = matrix_settings(cfg)
        log(f"\n{BOLD}2–3) Modo MATRIZ: caché por pose_source {ms['pose_sources']}{RESET}")
        single_user_only = bool(cfg.get("single_user_only", False))
        robbery_class = int(cfg.get("robbery_class", 6))
        dbg_cfg = cfg.get("debug") or {}
        inventory_limit = int(dbg_cfg.get("inventory_limit", 6000)) if args.debug else 0
        reuse_inv = bool(dbg_cfg.get("reuse_inventory_across_pose_sources", True))
        if inventory_limit > 0:
            log(
                f"  {YELLOW}DEBUG inventario rápido: máx {inventory_limit} clips/scan "
                f"(reuse_pose={reuse_inv}){RESET}"
            )
        rec_sampled: Optional[List[Dict[str, Any]]] = None
        for ps in ms["pose_sources"]:
            log(f"\n  {BOLD}pose_source={CYAN}{ps}{RESET}")
            t_inv = time.perf_counter()
            if rec_sampled is not None and reuse_inv and args.debug:
                rec_ps = swap_pose_source_on_records(rec_sampled, str(ps))
                log(
                    f"    Inventario reutilizado ({len(rec_ps)} clips, sin re-escaneo) "
                    f"en {time.perf_counter() - t_inv:.1f}s"
                )
            else:
                rec_ps = collect_clip_records(
                    data_roots,
                    pose_source=str(ps),
                    single_user_only=single_user_only,
                    robbery_class=robbery_class,
                    split_strategy=str(split_cfg.get("strategy", "stratified_group_hybrid")),
                    inventory_limit=inventory_limit,
                    log_fn=log,
                )
                log(f"    Inventario {len(rec_ps)} clips en {time.perf_counter() - t_inv:.1f}s")
            if args.max_clips > 0:
                dbg = cfg.get("debug") or {}
                min_rob = int(dbg.get("min_robbery_clips", 8)) if args.debug else 0
                if min_rob > 0:
                    rec_ps = subsample_records_stratified(
                        rec_ps,
                        args.max_clips,
                        robbery_class=robbery_class,
                        min_robbery=min_rob,
                        seed=int(split_cfg.get("seed", 42)),
                    )
                else:
                    rec_ps = rec_ps[: args.max_clips]
            if rec_sampled is None and args.debug and reuse_inv:
                rec_sampled = list(rec_ps)
            if not rec_ps:
                log(f"    {YELLOW}WARN: 0 clips para {ps}{RESET}")
                continue
            cdir = cache_dir_for_pose(layout["cache_dir"], str(ps))
            cr = build_pose_cache(
                rec_ps,
                cdir,
                target_hz=float(temporal.get("target_hz", 12.0)),
                split_ratios=dict(split_cfg.get("ratios") or {}),
                split_seed=int(split_cfg.get("seed", 42)),
                min_robbery_per_eval_split=min_rob_eval,
                use_shm=bool(cache_cfg.get("use_shm", True)),
                shm_basename=f"{cache_cfg.get('shm_basename', 'kanvis_pose_12hz')}_{ps}",
                log_fn=log,
            )
            manifest_by_pose[str(ps)] = cr.manifest_path
            if cache_result is None:
                cache_result = cr
            records.extend(rec_ps)
        if manifest_by_pose:
            manifest_path = next(iter(manifest_by_pose.values()))
        if records:
            log(f"\n{BOLD}Auditoría inventario (fps + hard neg P3){RESET}")
            fps_audit_report, hard_neg_summary = _inventory_audits(records, layout, cfg, log)
    elif data_roots and not args.skip_cache and cache_cfg.get("enabled", True):
        log(f"\n{BOLD}2) Inventario de clips (collect_examples legacy, solo lectura){RESET}")
        t_inv = time.perf_counter()
        pose_source = str(args.pose_source or cfg.get("pose_source", "full"))
        single_user_only = bool(cfg.get("single_user_only", False))
        robbery_class = int(cfg.get("robbery_class", 6))
        dbg_cfg = cfg.get("debug") or {}
        inventory_limit = int(dbg_cfg.get("inventory_limit", 6000)) if args.debug else 0
        if inventory_limit > 0:
            log(f"  {YELLOW}DEBUG inventario rápido: máx {inventory_limit} clips{RESET}")
        records = collect_clip_records(
            data_roots,
            pose_source=pose_source,
            single_user_only=single_user_only,
            robbery_class=robbery_class,
            split_strategy=str(split_cfg.get("strategy", "stratified_group_hybrid")),
            inventory_limit=inventory_limit,
            log_fn=log,
        )
        log(f"  Inventario en {time.perf_counter() - t_inv:.1f}s — {len(records)} registros")
        if records:
            log(f"  Por categoría (label): {json.dumps(_category_table(records))}")

        if args.max_clips > 0:
            dbg = cfg.get("debug") or {}
            min_rob = int(dbg.get("min_robbery_clips", 8)) if args.debug else 0
            if min_rob > 0:
                records = subsample_records_stratified(
                    records,
                    args.max_clips,
                    robbery_class=robbery_class,
                    min_robbery=min_rob,
                    seed=int(split_cfg.get("seed", 42)),
                )
                log(
                    f"  {YELLOW}--max-clips={args.max_clips} (estratificado, ≥{min_rob} robos cat.{robbery_class}){RESET}"
                )
            else:
                records = records[: args.max_clips]
                log(f"  {YELLOW}--max-clips={args.max_clips} (primeros N del inventario){RESET}")

        if not records:
            log(f"  {YELLOW}WARN: 0 clips; caché omitida.{RESET}")
        else:
            log(f"\n{BOLD}Auditoría inventario (fps + hard neg P3){RESET}")
            fps_audit_report, hard_neg_summary = _inventory_audits(records, layout, cfg, log)
            log(f"\n{BOLD}3) Caché poses @ {temporal.get('target_hz', 12)} Hz (mmap){RESET}")
            t_cache = time.perf_counter()
            cache_result = build_pose_cache(
                records,
                layout["cache_dir"],
                target_hz=float(temporal.get("target_hz", 12.0)),
                split_ratios=dict(split_cfg.get("ratios") or {}),
                split_seed=int(split_cfg.get("seed", 42)),
                min_robbery_per_eval_split=min_rob_eval,
                use_shm=bool(cache_cfg.get("use_shm", True)),
                shm_basename=str(cache_cfg.get("shm_basename", "kanvis_pose_12hz")),
                log_fn=log,
            )
            mb = cache_result.bytes_on_disk / (1024 * 1024)
            log(
                f"  {GREEN}OK{RESET} clips={cache_result.num_clips} frames={cache_result.total_frames} "
                f"size≈{mb:.1f} MiB"
            )
            log(f"  manifest: {cache_result.manifest_path}")
            log(f"  poses mmap: {cache_result.poses_mmap_path}")
            log(f"  Caché en {time.perf_counter() - t_cache:.1f}s")
    elif args.skip_cache:
        log(f"\n{YELLOW}--skip-cache: reutiliza manifest/mmap existente si hay.{RESET}")

    manifest_path = layout["cache_dir"] / "manifest.json"
    if cache_result:
        manifest_path = cache_result.manifest_path

    split_sum = _split_summary(
        json.loads(manifest_path.read_text(encoding="utf-8")).get("entries", [])
        if manifest_path.is_file()
        else []
    )
    manifest_entries_early: List[Dict[str, Any]] = []
    if manifest_path.is_file():
        manifest_entries_early = json.loads(manifest_path.read_text(encoding="utf-8")).get("entries", [])

    training_data_summary: Optional[Dict[str, Any]] = None
    if manifest_entries_early:
        training_data_summary = summarize_training_dataset(
            manifest_entries_early,
            cfg,
            debug_mode=bool(args.debug),
        )
        log(f"\n{BOLD}4) Dataset para entrenamiento v2 (clips + ventanas 8 s){RESET}")
        log(
            f"  {BOLD}Total clips:{RESET} {GREEN}{training_data_summary['num_clips_total']:,}{RESET} "
            f"(manifest/caché)"
        )
        log(f"  {BOLD}Por categoría (label carpeta):{RESET}")
        for cat_s, n in sorted(
            (training_data_summary.get("category_counts") or {}).items(),
            key=lambda x: int(x[0]),
        ):
            mark = f" {MAGENTA}← robo{RESET}" if int(cat_s) == int(cfg.get("robbery_class", 6)) else ""
            log(f"    cat {cat_s}: {n:,}{mark}")
        win = training_data_summary.get("windows") or {}
        log(
            f"  {BOLD}Ventanas indexadas:{RESET} "
            f"{win.get('window_seconds')}s stride {win.get('stride_seconds')}s, "
            f"policy={win.get('window_label_policy')}, max/clip={win.get('max_windows_per_clip')}"
        )
        for sp in SPLIT_ORDER:
            sc = (training_data_summary.get("split_clips") or {}).get(sp)
            if not sc:
                continue
            nw = int((win.get("by_split") or {}).get(sp, 0))
            np_ = int((win.get("positive_labels_by_split") or {}).get(sp, 0))
            log(
                f"    {CYAN}{SPLIT_LABELS.get(sp, sp)}{RESET}: "
                f"{sc['clips']:,} clips ({sc['robbery']:,} robos) → "
                f"{nw:,} ventanas (y=1: {np_:,})"
            )
        aug = training_data_summary.get("augment") or {}
        if aug.get("enabled"):
            tw = int((win.get("by_split") or {}).get("train", 0))
            log(
                f"  {BOLD}Sintético (augment):{RESET} {YELLOW}on-the-fly{RESET} en train — "
                f"0 filas extra en disco; ~{tw:,} ventanas × transform./época"
            )
        else:
            log(f"  {BOLD}Sintético (augment):{RESET} desactivado")

    if split_sum:
        log(f"\n{BOLD}4b) Split resumido (clips / robos){RESET}")
        for sp, counts in sorted(split_sum.items()):
            log(f"  {sp}: clips={counts['clips']} robbery={counts['robbery']}")

    enable_smoke = args.smoke_test or int(cfg.get("sprint1_smoke_jobs", 0)) > 0
    jobs = build_smoke_job_queue(
        cfg,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        run_id=run_id,
        enable_smoke=enable_smoke
        and not args.p0
        and not args.p1
        and not args.p1_smoke
        and not args.matrix,
    )

    p0_warnings: List[str] = []
    queue_purpose = "smoke_test" if jobs else "infra_only"
    queue_phase = phase

    if args.matrix and manifest_by_pose:
        mx_jobs, mx_warn = build_matrix_job_queue(
            cfg,
            manifest_by_pose=manifest_by_pose,
            data_roots=data_roots,
            run_id=run_id,
            smoke=bool(args.p1_smoke),
            debug=bool(args.debug),
        )
        for w in mx_warn:
            log(f"  {YELLOW}MATRIZ: {w}{RESET}")
        if mx_jobs:
            jobs = mx_jobs
            matrix_jobs_built = True
            queue_purpose = "campaign_matrix"
            queue_phase = "campaign_matrix"
            log(f"\n{BOLD}7m) Cola MATRIZ (full + filtered + …){RESET}")
            log(f"  {GREEN}{len(jobs)} jobs totales{RESET} (P0 legacy eval + P1 train según config)")
            for j in jobs[:10]:
                log(f"    - {j.job_id} ({j.job_type})")
            if len(jobs) > 10:
                log(f"    … +{len(jobs) - 10} más")

    if args.p0 and not args.matrix:
        if not manifest_path.is_file():
            log(f"\n{YELLOW}--p0: falta manifest (ejecuta preflight con --data-root y sin --skip-cache).{RESET}")
        elif not data_roots:
            log(f"\n{YELLOW}--p0: sin data_roots (necesarios para P0.A legacy_clip).{RESET}")
        else:
            p0_jobs, p0_warnings = build_p0_job_queue(
                cfg,
                manifest_path=manifest_path,
                data_roots=data_roots,
                run_id=run_id,
            )
            for w in p0_warnings:
                log(f"  {YELLOW}P0: {w}{RESET}")
            if p0_jobs:
                jobs = p0_jobs
                queue_purpose = "p0_baseline"
                queue_phase = "sprint2_p0"
                log(f"\n{BOLD}7b) Cola P0 (Sprint 2){RESET}")
                log(f"  {GREEN}{len(jobs)} jobs eval{RESET} (legacy_clip + ventanas × single_user on/off)")
                for j in jobs:
                    log(f"    - {j.job_id} ({j.params.get('mode')}, model {j.params.get('model_id')})")
            else:
                log(f"\n{YELLOW}--p0: cola vacía (configura rutas a modelo_41/modelo_48).{RESET}")

    if (args.p1 or args.p1_smoke) and manifest_path.is_file() and not args.matrix:
        p1_jobs, p1_warn = build_p1_job_queue(
            cfg,
            manifest_path=str(manifest_path),
            run_id=run_id,
            smoke=bool(args.p1_smoke),
            debug=bool(args.debug),
        )
        for w in p1_warn:
            log(f"  {YELLOW}P1: {w}{RESET}")
        if p1_jobs:
            jobs.extend(p1_jobs)
            queue_purpose = "p1_train" if queue_purpose == "infra_only" else queue_purpose + "+p1"
            queue_phase = "sprint3_p1"
            log(f"\n{BOLD}7c) Cola P1 train_v2{RESET}")
            log(f"  {GREEN}{len(p1_jobs)} jobs entrenamiento{RESET}")
            for j in p1_jobs[:6]:
                log(f"    - {j.job_id} ({j.params.get('arch')}, seed={j.params.get('seed')})")
            if len(p1_jobs) > 6:
                log(f"    … +{len(p1_jobs) - 6} más")
    elif args.p1 or args.p1_smoke:
        log(f"\n{YELLOW}--p1: falta manifest/caché (preflight con --data-root).{RESET}")

    manifest_entries: List[Dict[str, Any]] = []
    if manifest_path.is_file():
        manifest_entries = json.loads(manifest_path.read_text(encoding="utf-8")).get("entries", [])

    if manifest_entries:
        category_counts = _category_table_from_manifest(manifest_entries)
    elif records and not matrix_jobs_built:
        category_counts = _category_table(records)
    elif records:
        category_counts = _category_table_from_manifest(
            records[: len(records) // max(1, len(manifest_by_pose) or 1)]
        )
    else:
        category_counts = {}
    if not cache_result and manifest_path.is_file():
        cache_result_reuse = _cache_info_from_manifest(manifest_path)
    else:
        cache_result_reuse = None

    max_parallel = int((cfg.get("parallelism") or {}).get("max_parallel_jobs", 8))
    rec_path = parallel_recommendation_path(run_id)
    if rec_path.is_file():
        with open(rec_path, "r", encoding="utf-8") as f:
            max_parallel = int(json.load(f).get("max_parallel_jobs", max_parallel))
        log(f"\n{BOLD}5) Paralelismo (desde probe){RESET}")
        log(f"  max_parallel_jobs = {CYAN}{max_parallel}{RESET} (ejecuta ./run_campaign2.sh probe si aún no)")
    else:
        log(f"\n{BOLD}5) Paralelismo{RESET}")
        log(f"  {YELLOW}Sin probe aún; uso config max_parallel_jobs={max_parallel}{RESET}")
        log(f"  Recomendado: ./run_campaign2.sh probe antes del preflight para afinar slots GPU")

    time_est = build_time_estimate_report(
        cfg,
        manifest_entries=manifest_entries,
        max_parallel_jobs=max_parallel,
        run_id=run_id,
    )
    log(f"\n{BOLD}6) Estimación tiempo entrenamiento (plan Sprint 2 P1, NO ejecutado aún){RESET}")
    log(f"  Ventanas/época (train / val_dev): {time_est['window_train_rows_per_epoch']} / {time_est['window_val_rows_per_epoch']}")
    log(f"  Jobs planificados: {time_est['p1_planned_jobs']} × semillas → {time_est['p1_planned_training_runs']} entrenamientos")
    log(f"  Factor calibración (subestimación legacy): ×{time_est['calibration_factor_applied']}")
    log(f"  {BOLD}P1 secuencial GPU (1 job tras otro):{RESET} {CYAN}{time_est.get('p1_serial_gpu_human', time_est['p1_serial_human'])}{RESET}")
    log(
        f"  {BOLD}P1 paralelo GPU (×{max_parallel} jobs):{RESET} {GREEN}{time_est.get('p1_parallel_gpu_human', time_est['p1_parallel_human'])}{RESET} "
        f"({time_est.get('p1_parallel_gpu_seconds', time_est['p1_parallel_seconds']) / 3600:.2f} h)"
    )
    if time_est.get("p1_serial_cpu_human"):
        log(
            f"  {BOLD}P1 referencia CPU{RESET} (×8 vs GPU heurístico; train_v2 no soporta CPU hoy): "
            f"sec {time_est['p1_serial_cpu_human']} → par×{max_parallel} {time_est.get('p1_parallel_cpu_human', '?')}"
        )
    if time_est.get("legacy_campaign_20_models_reference"):
        leg = time_est["legacy_campaign_20_models_reference"]
        log(
            f"  Referencia legacy (20 arq. #41-like, 1 celda, calibrado): "
            f"secuencial {fmt_duration(leg['serial_seconds_calibrated'])} → "
            f"paralelo×{max_parallel} {fmt_duration(leg['parallel_seconds'])}"
        )
    log(f"  {YELLOW}{time_est['note']}{RESET}")
    log(
        f"  {DIM}(§6 = plan config time_estimate.p1_job_matrix: "
        f"{time_est['p1_planned_training_runs']} entrenamientos agrupados por tipo de job; "
        f"la cola §7 puede tener más jobs — 1 job = 1 semilla.){RESET}"
    )

    log(f"\n{BOLD}7) Cola de jobs{RESET}")
    if not jobs:
        log(f"  {GREEN}Cola vacía.{RESET}")
        if not args.p0:
            log(f"  Sprint 1: usa preflight --p0 tras configurar checkpoints para baseline.")
            log(f"  Scheduler: preflight --smoke-test && ./run_campaign2.sh train")
    elif queue_purpose == "p0_baseline":
        log(f"  Siguiente: ./run_campaign2.sh train && ./run_campaign2.sh p0-summary")
    elif "p1" in queue_purpose or queue_purpose == "campaign_matrix":
        log(f"  Siguiente: ./run_campaign2.sh train  (checkpoints en jobs/<job_id>/best.pt)")
    elif enable_smoke and jobs:
        log(f"  {YELLOW}Modo smoke-test:{RESET} {len(jobs)} jobs (scheduler, NO modelo Kanvis)")
    elif jobs:
        log(f"  {len(jobs)} jobs en cola ({queue_purpose})")
    else:
        log(f"  {YELLOW}Modo smoke-test:{RESET} {len(jobs)} jobs (scheduler, NO modelo Kanvis)")
        for j in jobs:
            log(f"    - {j.job_id} ({j.job_type})")

    queue_wall: Optional[Dict[str, Any]] = None
    if jobs:
        queue_wall = estimate_job_queue_wall(
            jobs,
            train_rows=int(time_est["window_train_rows_per_epoch"]),
            val_rows=int(time_est["window_val_rows_per_epoch"]),
            calibration_factor=float(time_est["calibration_factor_applied"]),
            max_parallel_jobs=max_parallel,
        )
        time_est["queue_wall_estimate"] = queue_wall
        log(f"\n{BOLD}8) Tiempo total estimado — cola real (train paralelo){RESET}")
        log(
            f"  Jobs en cola: {queue_wall['num_jobs_total']} "
            f"(p1_train={queue_wall['num_p1_train_jobs']}, p0_eval={queue_wall['num_p0_eval_jobs']})"
        )
        log(
            f"  {BOLD}Suma GPU (1 job tras otro):{RESET} {CYAN}{queue_wall['serial_gpu_human']}{RESET} "
            f"({queue_wall['serial_gpu_seconds'] / 3600:.2f} h)"
        )
        log(
            f"  {BOLD}Pared con ×{max_parallel} GPUs:{RESET} {GREEN}{queue_wall['parallel_gpu_human']}{RESET} "
            f"({queue_wall['parallel_gpu_seconds'] / 3600:.2f} h)  ← tiempo de ./train con esta cola"
        )
        if queue_wall["num_p1_train_jobs"]:
            log(
                f"  Solo entrenamiento v2 ({queue_wall['num_p1_train_jobs']} jobs): "
                f"sec {queue_wall['p1_train_serial_gpu_human']} → "
                f"par×{max_parallel} {GREEN}{queue_wall['p1_train_parallel_gpu_human']}{RESET}"
            )
        log(f"  {DIM}{queue_wall['formula']}{RESET}")
        post_min = 20 + max(0, queue_wall["num_p1_train_jobs"]) * 2
        log(
            f"  {DIM}+ post-train (calibrate + p1-eval, no en cola inicial): "
            f"~{post_min}–{post_min + 30} min extra{RESET}"
        )

    queue_payload = {
        "run_id": run_id,
        "phase": queue_phase,
        "purpose": queue_purpose,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [
            {"job_id": j.job_id, "job_type": j.job_type, "params": j.params, "priority": j.priority}
            for j in jobs
        ],
    }
    qpath = job_queue_path(run_id)
    with open(qpath, "w", encoding="utf-8") as f:
        json.dump(queue_payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    preflight_report = {
        "run_id": run_id,
        "phase": phase,
        "purpose": queue_purpose,
        "data_roots": [str(p) for p in data_roots],
        "num_clip_records": len(records) or len(manifest_entries),
        "category_counts": category_counts,
        "fps_audit": fps_audit_report,
        "hard_neg_manifest": (
            {"n_clips": hard_neg_summary.get("n_clips"), "path": "reports/hard_neg_manifest.json"}
            if hard_neg_summary
            else None
        ),
        "job_queue_summary": {
            "purpose": queue_purpose,
            "num_jobs": len(jobs),
            "job_ids": [j.job_id for j in jobs[:12]],
        },
        "cache": None
        if not cache_result and not cache_result_reuse
        else (
            {
                "manifest": str(cache_result.manifest_path),
                "total_frames": cache_result.total_frames,
                "num_clips": cache_result.num_clips,
                "bytes_on_disk": cache_result.bytes_on_disk,
                "poses_mmap": str(cache_result.poses_mmap_path),
            }
            if cache_result
            else cache_result_reuse
        ),
        "split_summary": split_sum,
        "training_data_summary": training_data_summary,
        "job_queue": str(qpath),
        "num_jobs": len(jobs),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "log_file": str(log_path),
        "max_parallel_jobs": max_parallel,
        "time_estimate": time_est,
        "queue_wall_estimate": queue_wall,
    }
    manifest_meta = {k: str(v) for k, v in manifest_by_pose.items()}
    if not manifest_meta and manifest_path.is_file():
        manifest_meta = {"default": str(manifest_path)}
    write_run_meta(
        run_id,
        {
            "run_id": run_id,
            "purpose": queue_purpose,
            "matrix_mode": bool(manifest_by_pose),
            "manifest_by_pose": manifest_meta,
            "default_manifest": str(manifest_path) if manifest_path.is_file() else None,
            "debug": bool(args.debug),
        },
    )

    report_path = layout["reports_dir"] / "preflight_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(preflight_report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if training_data_summary:
        tdp = layout["reports_dir"] / "training_data_plan.json"
        with open(tdp, "w", encoding="utf-8") as f:
            json.dump(training_data_summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
        log(f"{DIM}Plan dataset: {tdp}{RESET}")

    meta = {
        "run_id": run_id,
        "phase": "preflight_done",
        "campaign_phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config) if args.config else "campaign2_config.json"),
    }
    with open(run_meta_path(run_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    preflight_report["_json_path"] = str(report_path)
    pretty = render_preflight_summary(
        preflight_report,
        robbery_class=int(cfg.get("robbery_class", 6)),
    )
    log(pretty)
    log(f"{DIM}(JSON guardado en {report_path}){RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
