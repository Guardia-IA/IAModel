"""Salida legible del preflight Campaign2 (tablas y tiempos destacados)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from dataset_plan_summary import SPLIT_LABELS, SPLIT_ORDER

# ANSI
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"


def _hline(width: int = 72, char: str = "─") -> str:
    return char * width


def _box_title(title: str, width: int = 72) -> str:
    pad = max(0, width - len(title) - 4)
    return f"{BOLD}{CYAN}┌─ {title} {'─' * pad}┐{RESET}"


def _box_end(width: int = 72) -> str:
    return f"{BOLD}{CYAN}└{'─' * (width - 2)}┘{RESET}"


def _row(label: str, value: str, width: int = 72, value_color: str = WHITE) -> str:
    lab = f"  {label}"
    dots = max(1, width - len(lab) - len(value) - 4)
    return f"{DIM}│{RESET}{lab}{DIM}{'.' * dots}{RESET} {value_color}{value}{RESET} {DIM}│{RESET}"


def format_category_table(category_counts: Dict[str, int], robbery_class: int = 6) -> List[str]:
    lines: List[str] = []
    w = 72
    lines.append(_box_title("Clips por categoría", w))
    lines.append(f"{DIM}│{RESET}  {'Cat':>4}  {'Clips':>8}  {'%':>7}  {'Nota':<40}{DIM}│{RESET}")
    lines.append(f"{DIM}│{RESET}  {'────':>4}  {'────────':>8}  {'───────':>7}  {'─' * 40}{DIM}│{RESET}")
    total = sum(category_counts.values()) or 1
    names = {
        0: "pasa sin parar",
        1: "mira, no compra",
        2: "coge y deja",
        3: "coge → cesta/compra",
        4: "coge, lleva en mano",
        5: "camina con objetos",
        6: "ROBO / ocultación",
        7: "empleado reposición",
        9: "mira con objetos",
        13: "empleado limpieza",
        14: "mete en bolsa propia",
    }
    for cat_s in sorted(category_counts.keys(), key=lambda x: int(x)):
        cat = int(cat_s)
        n = category_counts[cat_s]
        pct = 100.0 * n / total
        note = names.get(cat, "")
        col = GREEN if cat == robbery_class else WHITE
        if n < 50:
            col = YELLOW
        lines.append(
            f"{DIM}│{RESET}  {col}{cat:>4}{RESET}  {col}{n:>8,}{RESET}  {pct:>6.2f}%  "
            f"{DIM}{note[:40]:<40}{RESET}{DIM}│{RESET}"
        )
    lines.append(_row("TOTAL", f"{total:,} clips", w, GREEN + BOLD))
    lines.append(_box_end(w))
    return lines


def format_dataset_overview_block(summary: Dict[str, Any], robbery_class: int = 6) -> List[str]:
    """Clips totales, ventanas 8 s y augment (sintético on-the-fly)."""
    lines: List[str] = []
    w = 72
    total = int(summary.get("num_clips_total") or 0)
    lines.append("")
    lines.append(_box_title("Inventario total (clips .npy)", w))
    lines.append(_row("Clips en manifest/caché", f"{total:,}", w, GREEN + BOLD))
    lines.append(_box_end(w))

    win = summary.get("windows") or {}
    aug = summary.get("augment") or {}
    lines.append("")
    lines.append(_box_title("Ventanas 8 s @ 12 Hz (muestras train_v2)", w))
    lines.append(
        _row(
            "Contrato",
            f"{win.get('window_seconds', 8)} s, stride {win.get('stride_seconds', 1)} s, "
            f"máx {win.get('max_windows_per_clip', '?')}/clip",
            w,
            CYAN,
        )
    )
    lines.append(_row("Política label", str(win.get("window_label_policy", "?")), w, CYAN))
    by_split = win.get("by_split") or {}
    pos_split = win.get("positive_labels_by_split") or {}
    neg_split = win.get("negative_labels_by_split") or {}
    lines.append(
        f"{DIM}│{RESET}  {'Partición':<26} {'Ventanas':>10} {'y=1':>8} {'y=0':>8}  {DIM}│{RESET}"
    )
    lines.append(f"{DIM}│{RESET}  {'─' * 26} {'─' * 10} {'─' * 8} {'─' * 8}  {DIM}│{RESET}")
    for key in SPLIT_ORDER:
        if key not in by_split:
            continue
        nv = int(by_split[key])
        np_ = int(pos_split.get(key, 0))
        nn = int(neg_split.get(key, 0))
        lines.append(
            f"{DIM}│{RESET}  {CYAN}{SPLIT_LABELS.get(key, key):<26}{RESET} "
            f"{GREEN}{nv:>10,}{RESET} {MAGENTA}{np_:>8,}{RESET} {nn:>8,}  {DIM}│{RESET}"
        )
    train_w = int(by_split.get("train", 0))
    lines.append(_box_end(w))

    lines.append("")
    lines.append(_box_title("Datos sintéticos (augment)", w))
    if aug.get("enabled"):
        lines.append(_row("Modo", "on-the-fly solo en train", w, GREEN))
        lines.append(
            _row(
                "Filas extra en disco",
                "0 (no mass-augment legacy)",
                w,
                GREEN,
            )
        )
        lines.append(
            _row(
                "Efecto por época",
                f"~{train_w:,} ventanas × transform. aleatorias",
                w,
                YELLOW,
            )
        )
    else:
        lines.append(_row("Augment", "desactivado", w, YELLOW))
    note = str(aug.get("note_es") or "")
    if note:
        wrapped = note[: w - 6]
        lines.append(f"{DIM}│{RESET}  {DIM}{wrapped}{RESET}{' ' * max(0, w - len(wrapped) - 4)}{DIM}│{RESET}")
    lines.append(_box_end(w))

    cat_by_sp = summary.get("category_counts_by_split") or {}
    if cat_by_sp:
        lines.append("")
        lines.append(_box_title("Clips por categoría y split", w))
        for key in SPLIT_ORDER:
            if key not in cat_by_sp:
                continue
            cats = cat_by_sp[key]
            if not cats:
                continue
            parts = [f"{c}:{n}" for c, n in sorted(cats.items(), key=lambda x: int(x[0]))]
            line = f"{SPLIT_LABELS.get(key, key)}: " + ", ".join(parts)
            if len(line) > w - 4:
                line = line[: w - 7] + "..."
            lines.append(f"{DIM}│{RESET}  {line:<{w - 4}}{DIM}│{RESET}")
        lines.append(_box_end(w))

    return lines


def format_split_table(split_summary: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    w = 72
    order = ("train", "val_dev", "threshold_calib", "frozen_test")
    labels = {
        "train": "Train (70%)",
        "val_dev": "Val desarrollo (10%)",
        "threshold_calib": "Umbral T (10%)",
        "frozen_test": "Test congelado U (10%)",
    }
    lines.append("")
    lines.append(_box_title("Split provisional (por group_id)", w))
    lines.append(
        f"{DIM}│{RESET}  {'Partición':<22} {'Clips':>10} {'Robos':>10} {'% robo':>10}  {DIM}│{RESET}"
    )
    lines.append(f"{DIM}│{RESET}  {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 10}  {DIM}│{RESET}")
    for key in order:
        if key not in split_summary:
            continue
        c = split_summary[key]
        clips = int(c.get("clips", 0))
        rob = int(c.get("robbery", 0))
        pct = 100.0 * rob / clips if clips else 0.0
        lines.append(
            f"{DIM}│{RESET}  {CYAN}{labels.get(key, key):<22}{RESET} "
            f"{GREEN}{clips:>10,}{RESET} {MAGENTA}{rob:>10,}{RESET} {pct:>9.1f}%  {DIM}│{RESET}"
        )
    lines.append(_box_end(w))
    return lines


def format_cache_block(cache: Optional[Dict[str, Any]], wall_preflight_s: float) -> List[str]:
    lines: List[str] = []
    w = 72
    lines.append("")
    lines.append(_box_title("Caché 12 Hz (preflight)", w))
    if not cache:
        lines.append(_row("Estado", "no construida (--skip-cache o sin datos)", w, YELLOW))
    elif cache.get("reused"):
        mb = int(cache.get("bytes_on_disk", 0)) / (1024 * 1024)
        lines.append(_row("Estado", "reutilizada (manifest existente)", w, GREEN))
        lines.append(_row("Clips en caché", f"{cache.get('num_clips', 0):,}", w, GREEN))
        lines.append(_row("Frames totales", f"{cache.get('total_frames', 0):,}", w, GREEN))
        if mb > 0:
            lines.append(_row("Tamaño mmap", f"{mb:.1f} MiB", w, CYAN))
    else:
        mb = int(cache.get("bytes_on_disk", 0)) / (1024 * 1024)
        lines.append(_row("Clips en caché", f"{cache.get('num_clips', 0):,}", w, GREEN))
        lines.append(_row("Frames totales", f"{cache.get('total_frames', 0):,}", w, GREEN))
        lines.append(_row("Tamaño mmap", f"{mb:.1f} MiB", w, CYAN))
        lines.append(_row("Tiempo preflight", f"{wall_preflight_s:.1f} s", w, CYAN))
        mmap = str(cache.get("poses_mmap", ""))
        if len(mmap) > 48:
            mmap = "…" + mmap[-44:]
        lines.append(_row("poses mmap", mmap, w, DIM))
    lines.append(_box_end(w))
    return lines


def format_time_estimate_block(te: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    w = 72
    lines.append("")
    lines.append(_box_title("Tiempo estimado — plan P1 train_v2", w))
    lines.append(
        f"{DIM}│{RESET}  {YELLOW}Heurística + calibración ×{te.get('calibration_factor_applied', 1)}{RESET}"
        f"  ·  paralelo ×{te.get('max_parallel_jobs', '?')} jobs  {DIM}│{RESET}"
    )
    lines.append(f"{DIM}│{RESET}  Ventanas/época: train {GREEN}{te.get('window_train_rows_per_epoch', 0):,}{RESET}"
                   f"  ·  val {CYAN}{te.get('window_val_rows_per_epoch', 0):,}{RESET}  {DIM}│{RESET}")
    lines.append(f"{DIM}│{RESET}  Entrenamientos planificados: "
                   f"{BOLD}{te.get('p1_planned_training_runs', 0)}{RESET} "
                   f"({te.get('p1_planned_jobs', 0)} jobs × semillas)  {DIM}│{RESET}")
    lines.append(f"{DIM}│{'─' * (w - 2)}│{RESET}")

    serial_h = te.get("p1_serial_human", "?")
    par_h = te.get("p1_parallel_human", "?")
    par_h_num = te.get("p1_parallel_seconds", 0) / 3600.0

    lines.append(
        f"{DIM}│{RESET}  {BOLD}1 GPU, secuencial:{RESET}     "
        f"{DIM}{serial_h:>24}{RESET}  {DIM}│{RESET}"
    )
    lines.append(
        f"{DIM}│{RESET}  {BOLD}{GREEN}Paralelo ({te.get('max_parallel_jobs')} jobs):{RESET}  "
        f"{GREEN}{BOLD}{par_h:>24}{RESET}  {DIM}│{RESET}"
    )
    if par_h_num >= 1:
        lines.append(
            f"{DIM}│{RESET}  {BLUE}≈ {par_h_num:.1f} horas de pared en RTX 5090 (estimado){RESET}"
            f"{' ' * max(0, 18 - len(f'{par_h_num:.1f}'))}{DIM}│{RESET}"
        )
    else:
        mins = te.get("p1_parallel_seconds", 0) / 60.0
        lines.append(
            f"{DIM}│{RESET}  {BLUE}≈ {mins:.0f} minutos de pared (estimado){RESET}"
            f"{' ' * 28}{DIM}│{RESET}"
        )

    per = te.get("per_job") or []
    if per:
        lines.append(f"{DIM}│{'─' * (w - 2)}│{RESET}")
        lines.append(
            f"{DIM}│{RESET}  {BOLD}Plan config (por tipo × semillas en p1_job_matrix):{RESET}{' ' * 8}{DIM}│{RESET}"
        )
        for j in per:
            jid = str(j.get("job_id", ""))[:28]
            sec = j.get("gpu_seconds_all_seeds", 0)
            h = sec / 3600
            if h >= 1:
                tstr = f"~{h:.1f} h"
            else:
                tstr = f"~{sec / 60:.0f} min"
            lines.append(
                f"{DIM}│{RESET}    {CYAN}{jid:<28}{RESET} {GREEN}{tstr:>12}{RESET}  {DIM}│{RESET}"
            )

    qw = te.get("queue_wall_estimate") or {}
    if qw.get("num_jobs_total"):
        lines.append(f"{DIM}│{'─' * (w - 2)}│{RESET}")
        lines.append(
            f"{DIM}│{RESET}  {BOLD}{GREEN}COLA REAL (§7/§8 preflight — usar este total):{RESET}{' ' * 12}{DIM}│{RESET}"
        )
        lines.append(
            f"{DIM}│{RESET}  Jobs: {qw.get('num_jobs_total')} "
            f"(train {qw.get('num_p1_train_jobs', 0)}, p0 {qw.get('num_p0_eval_jobs', 0)})  {DIM}│{RESET}"
        )
        lines.append(
            f"{DIM}│{RESET}  1 GPU secuencial: {CYAN}{qw.get('serial_gpu_human', '?')}{RESET}"
            f"{' ' * max(0, 24 - len(str(qw.get('serial_gpu_human', ''))))}{DIM}│{RESET}"
        )
        lines.append(
            f"{DIM}│{RESET}  {BOLD}Pared ×{qw.get('max_parallel_jobs', '?')} GPUs:{RESET} "
            f"{GREEN}{BOLD}{qw.get('parallel_gpu_human', '?')}{RESET}"
            f"  ({qw.get('parallel_gpu_seconds', 0) / 3600:.2f} h)  {DIM}│{RESET}"
        )
        if qw.get("p1_train_parallel_gpu_human"):
            lines.append(
                f"{DIM}│{RESET}  Solo p1_train par: {GREEN}{qw.get('p1_train_parallel_gpu_human')}{RESET}"
                f"{' ' * 36}{DIM}│{RESET}"
            )
        lines.append(f"{DIM}│{RESET}  {DIM}{qw.get('formula', '')[:64]}{RESET}  {DIM}│{RESET}")

    leg = te.get("legacy_campaign_20_models_reference")
    if leg:
        lines.append(f"{DIM}│{'─' * (w - 2)}│{RESET}")
        from training_time_estimate import fmt_duration

        lines.append(
            f"{DIM}│{RESET}  Referencia legacy (20 modelos, 1 celda): "
            f"sec {fmt_duration(leg.get('serial_seconds_calibrated', 0))} → "
            f"par {GREEN}{fmt_duration(leg.get('parallel_seconds', 0))}{RESET}  {DIM}│{RESET}"
        )

    note = te.get("note") or "Heurística; confirma con p1-smoke antes de campaña larga."
    lines.append(_row("Nota", note[:52], w, YELLOW))
    lines.append(_box_end(w))
    return lines


def format_queue_block(qs: Optional[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    if not qs or not qs.get("num_jobs"):
        return lines
    w = 72
    lines.append("")
    lines.append(_box_title("Cola de jobs (próximo train)", w))
    lines.append(_row("Propósito", str(qs.get("purpose", "?")), w, CYAN))
    lines.append(_row("Nº jobs", str(qs.get("num_jobs", 0)), w, GREEN))
    for jid in qs.get("job_ids") or []:
        lines.append(f"{DIM}│{RESET}    • {CYAN}{jid}{RESET}{' ' * max(0, 50 - len(str(jid)))}{DIM}│{RESET}")
    extra = int(qs.get("num_jobs", 0)) - len(qs.get("job_ids") or [])
    if extra > 0:
        lines.append(_row("…", f"+{extra} más", w, DIM))
    lines.append(_row("Monitor", "./run_campaign2.sh p1-status", w, GREEN))
    lines.append(_box_end(w))
    return lines


def render_preflight_summary(
    report: Dict[str, Any],
    *,
    robbery_class: int = 6,
) -> str:
    parts: List[str] = []
    parts.append("")
    parts.append(f"{BOLD}{GREEN}{'═' * 72}{RESET}")
    parts.append(f"{BOLD}{GREEN}  CAMPAIGN2 PREFLIGHT — RESUMEN{RESET}")
    parts.append(f"{DIM}  RUN_ID: {report.get('run_id')}{RESET}")
    parts.append(f"{BOLD}{GREEN}{'═' * 72}{RESET}")

    parts.extend(format_category_table(report.get("category_counts") or {}, robbery_class))
    if report.get("training_data_summary"):
        parts.extend(format_dataset_overview_block(report["training_data_summary"], robbery_class))
    parts.extend(format_split_table(report.get("split_summary") or {}))
    parts.extend(format_cache_block(report.get("cache"), float(report.get("wall_seconds", 0))))
    if report.get("time_estimate"):
        parts.extend(format_time_estimate_block(report["time_estimate"]))
    parts.extend(format_queue_block(report.get("job_queue_summary")))

    qw = report.get("queue_wall_estimate") or (report.get("time_estimate") or {}).get("queue_wall_estimate")
    pre_s = float(report.get("wall_seconds") or 0)
    parts.append("")
    parts.append(f"{BOLD}{MAGENTA}{'═' * 72}{RESET}")
    parts.append(f"{BOLD}{MAGENTA}  TIEMPO TOTAL ESTIMADO (resumen){RESET}")
    parts.append(f"{BOLD}{MAGENTA}{'═' * 72}{RESET}")
    parts.append(
        f"  {BOLD}Preflight{RESET} (inventario + caché, ya ejecutado): "
        f"{CYAN}{pre_s / 60:.1f} min{RESET} ({pre_s:.0f} s)"
    )
    if qw and qw.get("num_jobs_total"):
        par_h = float(qw.get("parallel_gpu_seconds", 0)) / 3600.0
        post_h = max(0.3, qw.get("num_p1_train_jobs", 0) * 2 / 60.0)
        total_h = pre_s / 3600.0 + par_h + post_h
        parts.append(
            f"  {BOLD}Train{RESET} (./run_campaign2.sh train, cola {qw.get('num_jobs_total')} jobs, "
            f"×{qw.get('max_parallel_jobs')} GPU): "
            f"{GREEN}{BOLD}{qw.get('parallel_gpu_human')}{RESET} ({par_h:.2f} h)"
        )
        parts.append(
            f"  {BOLD}Post-train{RESET} (calibrate + eval, aprox.): "
            f"{YELLOW}~{post_h * 60:.0f}–{post_h * 60 + 30:.0f} min{RESET}"
        )
        parts.append(
            f"  {BOLD}TOTAL hasta modelos evaluados:{RESET} "
            f"{GREEN}{BOLD}~{total_h:.1f} h{RESET} de pared "
            f"(preflight + train paralelo + post)"
        )
        te_plan = report.get("time_estimate") or {}
        n_plan = te_plan.get("p1_planned_training_runs", "?")
        parts.append(
            f"  {DIM}Nota: §6 del log = plan config ({n_plan} runs agrupados por tipo); "
            f"el TOTAL de arriba = cola real (1 job = 1 entrenamiento).{RESET}"
        )
    else:
        te = report.get("time_estimate") or {}
        parts.append(
            f"  {BOLD}Train P1 (plan config, ×{te.get('max_parallel_jobs', '?')} GPU):{RESET} "
            f"{GREEN}{te.get('p1_parallel_gpu_human', te.get('p1_parallel_human', '?'))}{RESET}"
        )
    parts.append(f"{BOLD}{MAGENTA}{'═' * 72}{RESET}")

    parts.append("")
    parts.append(
        f"{DIM}JSON completo:{RESET} {CYAN}{report.get('_json_path', 'reports/preflight_summary.json')}{RESET}"
    )
    parts.append(f"{DIM}Log detallado:{RESET} {report.get('log_file', '')}{RESET}")
    parts.append("")
    return "\n".join(parts)
