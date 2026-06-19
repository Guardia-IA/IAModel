#!/usr/bin/env python3
"""
Convierte export TXT/CSV de la web antigua al formato interno.

Entrada (columnas):
  id, retail_id, camera_id, created_at, category_id

Salida:
  video_path, inicio, fin, #clasificacion

video_path = {base_path}/{retail_id}/{camera_id}/{id_sin_guiones}/clip_buffer.mp4

El id en la ruta se normaliza: minúsculas y sin guiones.

Para clips ya recortados (clip_buffer.mp4):
  inicio = fin = 00:00:00  →  el extractor usa el fichero completo sin recortar

Uso:
  python parse_oldweb_videos.py entrada.txt --base-path /ruta/videos
  python parse_oldweb_videos.py entrada.txt -b /data/clips -o salida.csv --pause
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

INPUT_COLUMNS = ("id", "retail_id", "camera_id", "created_at", "category_id")
OUTPUT_COLUMNS = ("video_path", "inicio", "fin", "#clasificacion")
CLIP_FILENAME = "clip_buffer.mp4"
INICIO_FULL_CLIP = "00:00:00"
FIN_FULL_CLIP = "00:00:00"


def seconds_to_hms(seconds: int | float) -> str:
    total = max(0, int(round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _read_input_file(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError("Fichero vacío o sin cabecera")
        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        missing = [c for c in INPUT_COLUMNS if c not in field_map]
        if missing:
            raise ValueError(
                f"Columnas obligatorias ausentes: {missing}. "
                f"Encontradas: {list(reader.fieldnames)}"
            )
        rows = []
        for raw in reader:
            rows.append({col: (raw.get(field_map[col]) or "").strip() for col in INPUT_COLUMNS})
        return rows


def _normalize_id(raw_id: str) -> str:
    return raw_id.strip().lower().replace("-", "")


def _build_video_path(base_path: Path, row: dict[str, str]) -> Path:
    folder_id = _normalize_id(row["id"])
    return base_path / row["retail_id"] / row["camera_id"] / folder_id / CLIP_FILENAME


def _video_duration_hms(video_path: Path) -> str | None:
    if not video_path.is_file():
        return None
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    try:
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if frames <= 0 or fps <= 0:
            return None
        return seconds_to_hms(frames / fps)
    finally:
        cap.release()


def _print_row(
    index: int,
    total: int,
    row: dict[str, str],
    video_path: Path,
    inicio: str,
    fin: str,
    clasificacion: int,
    exists: bool,
    duration_ok: bool,
) -> None:
    print(f"\n--- Fila {index + 1}/{total} ---")
    print(f"  id:          {row['id']}")
    print(f"  retail_id:   {row['retail_id']}")
    print(f"  camera_id:   {row['camera_id']}")
    print(f"  created_at:  {row['created_at']}")
    print(f"  category_id: {row['category_id']}")
    print(f"  → salida:    {video_path}, {inicio}, {fin}, {clasificacion}")
    clip_ok = "✓" if exists else "✗"
    print(f"  clip:        [{clip_ok}] {video_path}")
    if exists and not duration_ok:
        print("  [WARN] No se pudo leer la duración del vídeo (opencv o metadatos)")


def parse_oldweb_videos(
    input_file: str | Path,
    base_path: str | Path,
    output_csv: str | Path | None = None,
    pause: bool = False,
    quiet: bool = False,
    skip_missing: bool = False,
    default_fin: str = "00:00:10",
) -> Path:
    input_file = Path(input_file)
    if not input_file.is_file():
        raise FileNotFoundError(f"Fichero no encontrado: {input_file}")

    prefix = Path(base_path).expanduser().resolve()
    if not prefix.is_dir() and not quiet:
        print(f"[WARN] El base-path no existe o no es carpeta: {prefix}")

    rows_in = _read_input_file(input_file)
    rows_out: list[dict[str, str | int]] = []
    total = len(rows_in)
    skipped = 0

    if not quiet:
        print(f"Base-path: {prefix}")

    for i, row in enumerate(rows_in):
        for key in ("id", "retail_id", "camera_id"):
            if not row[key]:
                skipped += 1
                print(f"\n[skip] Fila {i + 1}: {key} vacío")
                break
        else:
            try:
                clasificacion = int(float(row["category_id"]))
            except (TypeError, ValueError):
                skipped += 1
                print(f"\n[skip] Fila {i + 1}: category_id inválido ({row['category_id']!r})")
                continue

            video_path = _build_video_path(prefix, row)
            exists = video_path.is_file()
            fin = _video_duration_hms(video_path)
            duration_ok = fin is not None

            if skip_missing and not exists:
                skipped += 1
                if not quiet:
                    print(f"\n[skip] Fila {i + 1}: no existe {video_path}")
                continue

            if not quiet:
                _print_row(
                    i, total, row, video_path, INICIO_FULL_CLIP, FIN_FULL_CLIP,
                    clasificacion, exists, duration_ok,
                )
                if pause:
                    try:
                        input("  [Enter para continuar, Ctrl+C para abortar] ")
                    except KeyboardInterrupt:
                        print("\nInterrumpido por el usuario.")
                        break

            rows_out.append(
                {
                    "video_path": str(video_path.as_posix()),
                    "inicio": INICIO_FULL_CLIP,
                    "fin": FIN_FULL_CLIP,
                    "#clasificacion": clasificacion,
                }
            )

    if output_csv is None:
        output_csv = input_file.with_name(input_file.stem + "_parsed.csv")
    else:
        output_csv = Path(output_csv)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows_out)

    if not quiet:
        print(f"\n{'=' * 50}")
        print(f"Filas entrada:   {total}")
        print(f"Filas omitidas:  {skipped}")
        print(f"Filas escritas:  {len(rows_out)}")
        print(f"CSV generado:    {output_csv}")

    return output_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "TXT/CSV old web (id,retail_id,camera_id,created_at,category_id) "
            "→ video_path,inicio,fin,#clasificacion"
        ),
    )
    parser.add_argument("input", type=Path, help="TXT/CSV de entrada")
    parser.add_argument(
        "--base-path",
        "-b",
        type=Path,
        required=True,
        help="Prefijo: {base}/{retail_id}/{camera_id}/{id_normalizado}/clip_buffer.mp4",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="CSV de salida")
    parser.add_argument(
        "--default-fin",
        default="00:00:10",
        help="Fin por defecto si no se puede leer la duración (default: 00:00:10)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Omitir filas cuyo clip_buffer.mp4 no exista",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Pausar tras cada fila (Enter para continuar)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="No mostrar filas una a una",
    )
    args = parser.parse_args()

    try:
        parse_oldweb_videos(
            args.input,
            base_path=args.base_path,
            output_csv=args.output,
            pause=args.pause,
            quiet=args.quiet,
            skip_missing=args.skip_missing,
            default_fin=args.default_fin,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
