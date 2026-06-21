#!/usr/bin/env python3
"""
Convierte CSV exportado de register actions (web) al formato interno.

Entrada (columnas):
  id, camera_id, category_id

Salida:
  video_path, inicio, fin, #clasificacion

video_path = {base_path}/{id}_{camera_id}.mp4
inicio y fin = 00:00:00 por defecto (clip completo sin recortar)

Uso:
  python parse_registeractions_web.py entrada.csv --base-path /ruta/videos
  python parse_registeractions_web.py entrada.csv -b /data/clips -o salida.csv
  python parse_registeractions_web.py entrada.csv -b /data/clips --inicio 00:00:02 --fin 00:00:10
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

INPUT_COLUMNS = ("id", "camera_id", "category_id")
OUTPUT_COLUMNS = ("video_path", "inicio", "fin", "#clasificacion")
DEFAULT_TIME = "00:00:00"


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


def _build_video_path(base_path: Path, row_id: str, camera_id: str) -> str:
    filename = f"{row_id}_{camera_id}.mp4"
    return str((base_path / filename).as_posix())


def _print_row(
    index: int,
    total: int,
    row: dict[str, str],
    video_path: str,
    inicio: str,
    fin: str,
    clasificacion: int,
    exists: bool | None,
) -> None:
    print(f"\n--- Fila {index + 1}/{total} ---")
    print(f"  id:          {row['id']}")
    print(f"  camera_id:   {row['camera_id']}")
    print(f"  category_id: {row['category_id']}")
    print(f"  → salida:    {video_path}, {inicio}, {fin}, {clasificacion}")
    if exists is not None:
        mark = "✓" if exists else "✗"
        print(f"  fichero:     [{mark}] {video_path}")


def parse_registeractions_web_csv(
    input_csv: str | Path,
    base_path: str | Path,
    output_csv: str | Path | None = None,
    inicio: str = DEFAULT_TIME,
    fin: str = DEFAULT_TIME,
    skip_missing: bool = False,
    pause: bool = False,
    quiet: bool = False,
) -> Path:
    input_csv = Path(input_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"CSV no encontrado: {input_csv}")

    base = Path(base_path).expanduser().resolve()
    if not quiet:
        print(f"Prefijo base-path: {base}")
        if not base.is_dir():
            print(f"[WARN] El prefijo no existe o no es carpeta: {base}")

    rows_in = _read_input_csv(input_csv)
    rows_out: list[dict[str, str | int]] = []
    total = len(rows_in)
    skipped = 0

    for i, row in enumerate(rows_in):
        if not row["id"] or not row["camera_id"]:
            skipped += 1
            if not quiet:
                print(f"\n[skip] Fila {i + 1}: id o camera_id vacío")
            continue

        try:
            clasificacion = int(row["category_id"])
        except (TypeError, ValueError):
            skipped += 1
            if not quiet:
                print(f"\n[skip] Fila {i + 1}: category_id inválido ({row['category_id']!r})")
            continue

        video_path = _build_video_path(base, row["id"], row["camera_id"])
        video_file = Path(video_path)
        exists = video_file.is_file() if base.is_dir() else None

        if skip_missing and exists is False:
            skipped += 1
            if not quiet:
                print(f"\n[skip] Fila {i + 1}: no existe {video_path}")
            continue

        if not quiet:
            _print_row(i, total, row, video_path, inicio, fin, clasificacion, exists)
            if pause:
                try:
                    input("  [Enter para continuar, Ctrl+C para abortar] ")
                except KeyboardInterrupt:
                    print("\nInterrumpido por el usuario.")
                    break

        rows_out.append(
            {
                "video_path": video_path,
                "inicio": inicio,
                "fin": fin,
                "#clasificacion": clasificacion,
            }
        )

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_parsed.csv")
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
            "CSV register actions (id,camera_id,category_id) "
            "→ video_path,inicio,fin,#clasificacion"
        ),
    )
    parser.add_argument("csv", type=Path, help="CSV de entrada")
    parser.add_argument(
        "--base-path",
        "-b",
        type=Path,
        required=True,
        help="Prefijo para video_path: {base}/{id}_{camera_id}.mp4",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="CSV de salida")
    parser.add_argument(
        "--inicio",
        default=DEFAULT_TIME,
        help=f"Inicio por defecto para todas las filas (default: {DEFAULT_TIME})",
    )
    parser.add_argument(
        "--fin",
        default=DEFAULT_TIME,
        help=f"Fin por defecto para todas las filas (default: {DEFAULT_TIME})",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Omitir filas cuyo .mp4 no exista en base-path",
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
        parse_registeractions_web_csv(
            args.csv,
            base_path=args.base_path,
            output_csv=args.output,
            inicio=args.inicio,
            fin=args.fin,
            skip_missing=args.skip_missing,
            pause=args.pause,
            quiet=args.quiet,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
