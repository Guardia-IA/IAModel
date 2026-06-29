#!/usr/bin/env python3
"""
Convierte CSV con category_segments al formato interno de pose_extractor_clean.

Entrada (columnas):
  id, retail_id, camera_id, category_id, max_persons_in_frame, category_segments

Ruta del clip:
  {base_path}/{retail_id}/{camera_id}/{id_sin_guiones}/{clip_filename}

Por defecto clip_filename = clip_buffer.mp4 (mismo layout que parse_oldweb_videos.py).

Salida:
  video_path, inicio, fin, #clasificacion

Antes de generar el CSV ejecuta un preflight (salvo --no-preflight):
  - carpeta del clip existe
  - clip_buffer.mp4 (o --clip-filename) existe y tiene duración legible
  - merged.npy presente (sin él = sin personas → se omite la fila)
  - category_segments válido y tiempos coherentes con el vídeo
  - clasificación en rango [0, max_clas] (security.py)

Por defecto, al generar el CSV se omiten las filas con errores de preflight
y se escriben las válidas. Usa --strict para abortar si hay cualquier error.

Uso:
  python parse_category_segments.py entrada.csv --base-path /ruta/videos
  python parse_category_segments.py entrada.csv -b /data/clips --preflight-only
  python parse_category_segments.py entrada.csv -b /data/clips -o salida.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _SCRIPT_DIR.parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

try:
    from security import DEFAULT_MAX_CLAS, DEFAULT_MIN_CLAS  # type: ignore[attr-defined]
except ImportError:
    DEFAULT_MIN_CLAS = 0
    DEFAULT_MAX_CLAS = 14

REQUIRED_COLUMNS = ("id", "retail_id", "camera_id", "category_id", "category_segments")
OUTPUT_COLUMNS = ("video_path", "inicio", "fin", "#clasificacion")
FULL_CLIP_TIME = "00:00:00"
DEFAULT_CLIP_FILENAME = "clip_buffer.mp4"
MERGED_FILENAME = "merged.npy"
MULTI_SEGMENT_CATEGORY = -2

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass
class PreflightIssue:
    row_num: int
    row_id: str
    level: str  # "error" | "warn"
    code: str
    message: str


@dataclass
class RowPlan:
    row_num: int
    row: dict[str, str]
    clip_dir: Path
    clip_file: Path
    merged_file: Path
    video_path: str
    cat_flag: int
    segments: list[dict[str, Any]]
    output_rows: list[dict[str, str | int]]
    video_duration: float = -1.0
    skip_reason: str | None = None


@dataclass
class PreflightReport:
    base_path: Path
    input_rows: int
    planned_output_rows: int
    issues: list[PreflightIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[PreflightIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[PreflightIssue]:
        return [i for i in self.issues if i.level == "warn"]

    @property
    def ok(self) -> bool:
        return not self.errors


def _read_input_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError("CSV vacío o sin cabecera")
        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        missing = [c for c in REQUIRED_COLUMNS if c not in field_map]
        if missing:
            raise ValueError(
                f"Columnas obligatorias ausentes: {missing}. "
                f"Encontradas: {list(reader.fieldnames)}"
            )
        rows: list[dict[str, str]] = []
        for raw in reader:
            row = {col: (raw.get(field_map[col]) or "").strip() for col in REQUIRED_COLUMNS}
            if "max_persons_in_frame" in field_map:
                row["max_persons_in_frame"] = (raw.get(field_map["max_persons_in_frame"]) or "").strip()
            rows.append(row)
        return rows


def _uuid_without_dashes(value: str) -> str:
    return value.strip().replace("-", "")


def _build_clip_dir(base_path: Path, retail_id: str, camera_id: str, row_id: str) -> Path:
    folder_id = _uuid_without_dashes(row_id)
    return base_path / retail_id.strip() / camera_id.strip() / folder_id


def _build_video_path(
    base_path: Path,
    retail_id: str,
    camera_id: str,
    row_id: str,
    clip_filename: str,
) -> str:
    return str(
        (_build_clip_dir(base_path, retail_id, camera_id, row_id) / clip_filename).as_posix()
    )


def _parse_segments(raw: str) -> list[dict[str, Any]]:
    if not raw:
        raise ValueError("category_segments vacío")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("category_segments debe ser un array JSON")
    if not data:
        raise ValueError("category_segments sin elementos")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"segmento {i} no es un objeto")
        out.append(item)
    return out


def _seconds_to_hms(seconds: float) -> str:
    total = max(0, int(float(seconds) + 0.5))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _hms_to_seconds(t: str) -> int:
    h, m, s = map(int, str(t).strip().split(":"))
    return h * 3600 + m * 60 + s


def _video_duration_seconds(video_path: Path) -> float:
    if not video_path.is_file():
        return -1.0
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return -1.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        if fps > 0 and frames > 0:
            return float(frames / fps)
    except Exception:
        pass
    return -1.0


def _is_near(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def _is_full_clip_segment(start: float, end: float, video_duration: float, margin_sec: float) -> bool:
    if video_duration <= 0:
        return False
    if not _is_near(start, 0.0, 0.01):
        return False
    return _is_near(end, video_duration, margin_sec) or end >= video_duration - margin_sec


def _rows_from_single_category(
    video_path: str,
    clip_file: Path,
    segments: list[dict[str, Any]],
    margin_sec: float,
) -> list[dict[str, str | int]]:
    if len(segments) != 1:
        raise ValueError(f"category_id>=0 espera 1 segmento, hay {len(segments)}")

    seg = segments[0]
    start = float(seg["start"])
    end = float(seg["end"])
    clas = int(seg.get("category_id", seg.get("categoryId", 0)))

    duration = _video_duration_seconds(clip_file)
    if _is_full_clip_segment(start, end, duration, margin_sec):
        inicio, fin = FULL_CLIP_TIME, FULL_CLIP_TIME
    else:
        inicio = _seconds_to_hms(start)
        fin = _seconds_to_hms(end)

    return [
        {
            "video_path": video_path,
            "inicio": inicio,
            "fin": fin,
            "#clasificacion": clas,
        }
    ]


def _rows_from_multi_segments(
    video_path: str,
    segments: list[dict[str, Any]],
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        clas = int(seg.get("category_id", seg.get("categoryId", 0)))
        rows.append(
            {
                "video_path": video_path,
                "inicio": _seconds_to_hms(start),
                "fin": _seconds_to_hms(end),
                "#clasificacion": clas,
            }
        )
    return rows


def _plan_row(
    row: dict[str, str],
    row_num: int,
    base: Path,
    *,
    clip_filename: str,
    margin_sec: float,
) -> RowPlan:
    clip_dir = _build_clip_dir(base, row["retail_id"], row["camera_id"], row["id"])
    clip_file = clip_dir / clip_filename
    merged_file = clip_dir / MERGED_FILENAME
    video_path = _build_video_path(base, row["retail_id"], row["camera_id"], row["id"], clip_filename)

    plan = RowPlan(
        row_num=row_num,
        row=row,
        clip_dir=clip_dir,
        clip_file=clip_file,
        merged_file=merged_file,
        video_path=video_path,
        cat_flag=0,
        segments=[],
        output_rows=[],
    )

    if not row["id"] or not row["retail_id"] or not row["camera_id"]:
        plan.skip_reason = "id, retail_id o camera_id vacío"
        return plan

    try:
        plan.cat_flag = int(row["category_id"])
    except (TypeError, ValueError):
        plan.skip_reason = f"category_id inválido ({row['category_id']!r})"
        return plan

    try:
        plan.segments = _parse_segments(row["category_segments"])
    except (json.JSONDecodeError, ValueError) as exc:
        plan.skip_reason = f"category_segments inválido: {exc}"
        return plan

    if plan.cat_flag == MULTI_SEGMENT_CATEGORY:
        plan.output_rows = _rows_from_multi_segments(video_path, plan.segments)
    elif plan.cat_flag >= 0:
        plan.output_rows = _rows_from_single_category(
            video_path, clip_file, plan.segments, margin_sec=margin_sec
        )
    else:
        plan.skip_reason = f"category_id no soportado ({plan.cat_flag})"
        return plan

    if clip_file.is_file():
        plan.video_duration = _video_duration_seconds(clip_file)

    return plan


def _issue(row_num: int, row_id: str, level: str, code: str, message: str) -> PreflightIssue:
    return PreflightIssue(row_num=row_num, row_id=row_id, level=level, code=code, message=message)


def run_preflight(
    plans: list[RowPlan],
    base_path: Path,
    *,
    max_clas: int = DEFAULT_MAX_CLAS,
    min_clas: int = DEFAULT_MIN_CLAS,
    time_margin_sec: float = 1.0,
) -> PreflightReport:
    issues: list[PreflightIssue] = []
    output_rows = 0

    for plan in plans:
        row_id = plan.row.get("id", "?")
        if plan.skip_reason:
            issues.append(
                _issue(plan.row_num, row_id, "error", "row_invalid", plan.skip_reason)
            )
            continue

        output_rows += len(plan.output_rows)

        if not plan.clip_dir.is_dir():
            issues.append(
                _issue(
                    plan.row_num,
                    row_id,
                    "error",
                    "missing_clip_dir",
                    f"No existe carpeta: {plan.clip_dir}",
                )
            )

        if not plan.clip_file.is_file():
            issues.append(
                _issue(
                    plan.row_num,
                    row_id,
                    "error",
                    "missing_clip_video",
                    f"No existe vídeo: {plan.clip_file}",
                )
            )
        elif plan.video_duration <= 0:
            issues.append(
                _issue(
                    plan.row_num,
                    row_id,
                    "error",
                    "unreadable_video_duration",
                    f"No se pudo leer duración de {plan.clip_file.name} (ffprobe/opencv)",
                )
            )

        if not plan.merged_file.is_file():
            issues.append(
                _issue(
                    plan.row_num,
                    row_id,
                    "error",
                    "missing_merged_npy",
                    f"Sin {MERGED_FILENAME} (sin personas detectadas): {plan.merged_file}",
                )
            )

        for seg_idx, seg in enumerate(plan.segments):
            try:
                start = float(seg["start"])
                end = float(seg["end"])
                clas = int(seg.get("category_id", seg.get("categoryId", 0)))
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(
                    _issue(
                        plan.row_num,
                        row_id,
                        "error",
                        "bad_segment",
                        f"Segmento {seg_idx}: {exc}",
                    )
                )
                continue

            if start < 0:
                issues.append(
                    _issue(
                        plan.row_num,
                        row_id,
                        "error",
                        "segment_start_negative",
                        f"Segmento {seg_idx}: start={start} < 0",
                    )
                )
            if end <= start:
                issues.append(
                    _issue(
                        plan.row_num,
                        row_id,
                        "error",
                        "segment_end_before_start",
                        f"Segmento {seg_idx}: end={end} <= start={start}",
                    )
                )
            if clas < min_clas or clas > max_clas:
                issues.append(
                    _issue(
                        plan.row_num,
                        row_id,
                        "error",
                        "classification_out_of_range",
                        f"Segmento {seg_idx}: clasificación {clas} fuera de [{min_clas}, {max_clas}]",
                    )
                )

            if plan.video_duration > 0 and end > plan.video_duration + time_margin_sec:
                issues.append(
                    _issue(
                        plan.row_num,
                        row_id,
                        "error",
                        "segment_end_beyond_video",
                        (
                            f"Segmento {seg_idx}: end={end:.3f}s > duración vídeo "
                            f"{plan.video_duration:.3f}s (+{time_margin_sec}s margen)"
                        ),
                    )
                )

        if plan.cat_flag >= 0 and len(plan.segments) != 1:
            issues.append(
                _issue(
                    plan.row_num,
                    row_id,
                    "error",
                    "segment_count_mismatch",
                    f"category_id={plan.cat_flag} pero hay {len(plan.segments)} segmentos",
                )
            )

        for out_idx, out in enumerate(plan.output_rows, start=1):
            inicio = str(out["inicio"])
            fin = str(out["fin"])
            is_full = inicio == FULL_CLIP_TIME and fin == FULL_CLIP_TIME
            if is_full:
                if plan.video_duration <= 0:
                    issues.append(
                        _issue(
                            plan.row_num,
                            row_id,
                            "error",
                            "full_clip_no_duration",
                            f"Salida {out_idx}: clip completo 00:00:00–00:00:00 sin duración legible",
                        )
                    )
            else:
                if _hms_to_seconds(fin) <= _hms_to_seconds(inicio):
                    issues.append(
                        _issue(
                            plan.row_num,
                            row_id,
                            "error",
                            "invalid_hms_range",
                            f"Salida {out_idx}: inicio={inicio} fin={fin} (fin <= inicio)",
                        )
                    )
                if plan.video_duration > 0 and _hms_to_seconds(fin) > int(plan.video_duration + time_margin_sec):
                    issues.append(
                        _issue(
                            plan.row_num,
                            row_id,
                            "error",
                            "hms_end_beyond_video",
                            (
                                f"Salida {out_idx}: fin={fin} > duración vídeo "
                                f"{plan.video_duration:.1f}s"
                            ),
                        )
                    )

    return PreflightReport(
        base_path=base_path,
        input_rows=len(plans),
        planned_output_rows=output_rows,
        issues=issues,
    )


def _issues_by_row_id(issues: list[PreflightIssue]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for issue in issues:
        out.setdefault(issue.row_id, set()).add(issue.code)
    return out


def _issue_codes_for_row(report: PreflightReport, row_num: int) -> list[str]:
    return sorted({i.code for i in report.errors if i.row_num == row_num})


def _print_preflight(report: PreflightReport, *, clip_filename: str, quiet: bool) -> None:
    failed_by_id = _issues_by_row_id(report.errors)

    if not quiet:
        print(f"\n{BOLD}{'=' * 72}")
        print("PREFLIGHT — comprobaciones antes de pose_extractor_clean")
        print(f"{'=' * 72}{RESET}")
        print(f"  Vídeo esperado por clip: {clip_filename}")
        print(f"  merged.npy obligatorio:  sí (sin él = sin personas, se omite)")
        print(f"  Clasificación válida:    [{DEFAULT_MIN_CLAS}, {DEFAULT_MAX_CLAS}]")
        print(f"  Filas CSV entrada:       {report.input_rows}")
        print(f"  Filas CSV salida prev.:  {report.planned_output_rows}")
        print(f"  Base-path:               {report.base_path}")

        err_codes = Counter(i.code for i in report.errors)

        if report.errors:
            print(f"\n{RED}Errores ({len(report.errors)}):{RESET}")
            for issue in report.errors[:50]:
                print(
                    f"  [{issue.code}] fila {issue.row_num} id={issue.row_id} — {issue.message}"
                )
            if len(report.errors) > 50:
                print(f"  … y {len(report.errors) - 50} mensajes más")
            print(f"\n  Resumen errores: {dict(err_codes)}")

        if report.warnings:
            print(f"\n{YELLOW}Avisos ({len(report.warnings)}):{RESET}")
            for issue in report.warnings[:20]:
                print(
                    f"  [{issue.code}] fila {issue.row_num} id={issue.row_id} — {issue.message}"
                )
            if len(report.warnings) > 20:
                print(f"  … y {len(report.warnings) - 20} más")

    if failed_by_id:
        print(f"\n{RED}Clips con error — id completo ({len(failed_by_id)}):{RESET}")
        for row_id in sorted(failed_by_id):
            codes = ", ".join(sorted(failed_by_id[row_id]))
            print(f"  {row_id}  [{codes}]")

    if report.ok:
        if not quiet:
            print(f"\n{GREEN}[OK] Preflight superado: listo para generar CSV y pasar a pose_extractor_clean.{RESET}")
    else:
        print(
            f"\n{RED}[X] Preflight: {len(failed_by_id)} clip(s) con error, "
            f"{len(report.warnings)} aviso(s).{RESET}"
        )


def _print_skipped_clips_summary(
    skipped: list[dict[str, Any]],
    *,
    title: str,
) -> None:
    if not skipped:
        return
    print(f"\n{YELLOW}{title} ({len(skipped)} clip(s)):{RESET}")
    code_counts: Counter[str] = Counter()
    for item in skipped:
        for code in item["codes"]:
            code_counts[code] += 1
    if code_counts:
        parts = ", ".join(f"{code}: {n}" for code, n in code_counts.most_common())
        print(f"  Motivos: {parts}")
    for item in skipped:
        codes = ", ".join(item["codes"])
        reason = item.get("reason")
        extra = f" — {reason}" if reason else ""
        print(f"  {item['id']}  fila {item['row_num']}  [{codes}]{extra}")


def parse_category_segments_csv(
    input_csv: str | Path,
    base_path: str | Path,
    output_csv: str | Path | None = None,
    *,
    clip_filename: str = DEFAULT_CLIP_FILENAME,
    margin_sec: float = 1.0,
    time_margin_sec: float = 1.0,
    max_clas: int = DEFAULT_MAX_CLAS,
    run_preflight_check: bool = True,
    preflight_only: bool = False,
    strict: bool = False,
    skip_invalid_rows: bool = True,
    quiet: bool = False,
) -> Path | None:
    input_csv = Path(input_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"CSV no encontrado: {input_csv}")

    base = Path(base_path).expanduser().resolve()
    if not quiet:
        print(f"Base-path: {base}")
        if not base.is_dir():
            print(f"{YELLOW}[WARN] El base-path no existe o no es carpeta: {base}{RESET}")

    rows_in = _read_input_csv(input_csv)
    plans = [
        _plan_row(row, i + 2, base, clip_filename=clip_filename, margin_sec=margin_sec)
        for i, row in enumerate(rows_in)
    ]

    report: PreflightReport | None = None
    if run_preflight_check:
        report = run_preflight(
            plans,
            base,
            max_clas=max_clas,
            time_margin_sec=time_margin_sec,
        )
        _print_preflight(report, clip_filename=clip_filename, quiet=quiet)

        if preflight_only and not report.ok:
            raise RuntimeError(
                f"Preflight fallido con {len(report.errors)} error(es)."
            )
        if not preflight_only and strict and not report.ok:
            raise RuntimeError(
                f"Preflight fallido con {len(report.errors)} error(es). "
                "Corrige los datos o ejecuta sin --strict para omitir filas con error."
            )

    if preflight_only:
        return None

    rows_out: list[dict[str, str | int]] = []
    skipped_parse = 0
    skipped_preflight = 0
    skipped_details: list[dict[str, Any]] = []
    err_rows = {e.row_num for e in report.errors} if report else set()

    for plan in plans:
        if plan.skip_reason:
            skipped_parse += 1
            if skip_invalid_rows:
                skipped_details.append(
                    {
                        "id": plan.row.get("id", "?"),
                        "row_num": plan.row_num,
                        "codes": ["row_invalid"],
                        "reason": plan.skip_reason,
                    }
                )
                continue
            continue

        if report and plan.row_num in err_rows:
            skipped_preflight += 1
            skipped_details.append(
                {
                    "id": plan.row.get("id", "?"),
                    "row_num": plan.row_num,
                    "codes": _issue_codes_for_row(report, plan.row_num),
                    "reason": None,
                }
            )
            continue

        if not quiet and not run_preflight_check:
            print(f"\n--- Fila {plan.row_num} ---")
            for j, out in enumerate(plan.output_rows, start=1):
                print(
                    f"  → {j}: {out['video_path']}, {out['inicio']}, {out['fin']}, "
                    f"{out['#clasificacion']}"
                )

        rows_out.extend(plan.output_rows)

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_parsed.csv")
    else:
        output_csv = Path(output_csv)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows_out)

    if not quiet:
        print(f"\n{BOLD}{'=' * 50}{RESET}")
        print(f"Filas entrada:              {len(rows_in)}")
        print(f"Omitidas (parse inválido):  {skipped_parse}")
        print(f"Omitidas (preflight error): {skipped_preflight}")
        print(f"Filas CSV escritas:         {len(rows_out)}")
        print(f"CSV generado:               {output_csv}")

    _print_skipped_clips_summary(
        skipped_details,
        title="Clips NO incluidos en el CSV (id completo)",
    )

    if not quiet and not rows_out:
        print(f"{RED}[WARN] CSV vacío: ninguna fila pasó el preflight.{RESET}")
    elif quiet and skipped_details:
        print(
            f"{YELLOW}Resumen: {len(skipped_details)} clip(s) omitido(s), "
            f"{len(rows_out)} fila(s) en CSV.{RESET}"
        )

    return output_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CSV category_segments → video_path,inicio,fin,#clasificacion "
            "(con preflight antes de generar salida)"
        ),
    )
    parser.add_argument("csv", type=Path, help="CSV de entrada")
    parser.add_argument(
        "--base-path",
        "-b",
        type=Path,
        required=True,
        help=(
            "Raíz: {base}/{retail_id}/{camera_id}/{id_sin_guiones}/{clip_filename}"
        ),
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="CSV de salida")
    parser.add_argument(
        "--clip-filename",
        default=DEFAULT_CLIP_FILENAME,
        help=f"Nombre del vídeo en cada carpeta (default: {DEFAULT_CLIP_FILENAME})",
    )
    parser.add_argument(
        "--margin-sec",
        type=float,
        default=1.0,
        help="Margen (s) para detectar clip completo 00:00:00/00:00:00 (default 1)",
    )
    parser.add_argument(
        "--time-margin-sec",
        type=float,
        default=1.0,
        help="Margen (s) preflight: segmento fin vs duración vídeo (default 1)",
    )
    parser.add_argument(
        "--max-clas",
        type=int,
        default=DEFAULT_MAX_CLAS,
        help=f"Clasificación máxima válida (default {DEFAULT_MAX_CLAS})",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Solo ejecutar preflight, no escribir CSV de salida",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Saltar preflight (no recomendado)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abortar sin escribir CSV si el preflight tiene errores (default: omitir filas con error)",
    )
    parser.add_argument(
        "--include-invalid-rows",
        action="store_true",
        help="Incluir filas con category_id/segments inválidos al escribir CSV (no recomendado)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Menos salida (preflight solo resumen final)",
    )
    args = parser.parse_args()

    try:
        result = parse_category_segments_csv(
            args.csv,
            base_path=args.base_path,
            output_csv=args.output,
            clip_filename=str(args.clip_filename),
            margin_sec=float(args.margin_sec),
            time_margin_sec=float(args.time_margin_sec),
            max_clas=int(args.max_clas),
            run_preflight_check=not args.no_preflight,
            preflight_only=bool(args.preflight_only),
            strict=bool(args.strict),
            skip_invalid_rows=not args.include_invalid_rows,
            quiet=args.quiet,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.preflight_only:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
