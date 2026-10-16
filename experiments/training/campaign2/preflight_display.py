"""Salida legible del preflight Campaign2 (tablas y tiempos destacados)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    lines.append(_box_title("Tiempo estimado — plan Sprint 2 P1 (aún NO entrena)", w))
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
        lines.append(f"{DIM}│{RESET}  {BOLD}Por job (GPU, todas las semillas):{RESET}{' ' * 28}{DIM}│{RESET}")
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

    leg = te.get("legacy_campaign_20_models_reference")
    if leg:
        lines.append(f"{DIM}│{'─' * (w - 2)}│{RESET}")
        from training_time_estimate import fmt_duration

        lines.append(
            f"{DIM}│{RESET}  Referencia legacy (20 modelos, 1 celda): "
            f"sec {fmt_duration(leg.get('serial_seconds_calibrated', 0))} → "
            f"par {GREEN}{fmt_duration(leg.get('parallel_seconds', 0))}{RESET}  {DIM}│{RESET}"
        )

    lines.append(_row("Nota", "Sprint 1 solo caché; train Kanvis en Sprint 2", w, YELLOW))
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
    parts.extend(format_split_table(report.get("split_summary") or {}))
    parts.extend(format_cache_block(report.get("cache"), float(report.get("wall_seconds", 0))))
    if report.get("time_estimate"):
        parts.extend(format_time_estimate_block(report["time_estimate"]))

    parts.append("")
    parts.append(
        f"{DIM}JSON completo:{RESET} {CYAN}{report.get('_json_path', 'reports/preflight_summary.json')}{RESET}"
    )
    parts.append(f"{DIM}Log detallado:{RESET} {report.get('log_file', '')}{RESET}")
    parts.append("")
    return "\n".join(parts)
