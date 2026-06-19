#!/usr/bin/env python3
"""
Convierte CSV exportado de la web de etiquetado de robos al formato interno.

Entrada (columnas):
  id, name, video_url, start, end, deleted, num_users, labeled

Salida:
  video_path, inicio, fin, #clasificacion

Ejemplo:
  name=..._conservas_001536_001546_6, start=2, end=9
  → video_path=..._conservas_001536_001546_6/clip.mp4, 00:00:02, 00:00:09, 6

Uso:
  python parser_theft_web.py entrada.csv
  python parser_theft_web.py entrada.csv -o salida.csv --base-path /ruta/videos
  python parser_theft_web.py entrada.csv --base-path /data/clips --pause
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

CLASIFICACION_ROBO = 6

INPUT_COLUMNS = ("id", "name", "video_url", "start", "end", "deleted", "num_users", "labeled")
OUTPUT_COLUMNS = ("video_path", "inicio", "fin", "#clasificacion")


def seconds_to_hms(seconds: int | float) -> str:
    total = int(seconds)
    if total < 0:
        raise ValueError(f"Segundos negativos: {seconds}")
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _is_truthy(val: str | None) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"t", "true", "1", "yes", "y", "si", "sí"}


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


CLIP_FILENAME = "clip.mp4"


def _video_path_with_prefix(base_path: Path | None, name: str) -> str:
    name = name.strip()
    if not base_path:
        return f"{name}/{CLIP_FILENAME}"
    return str((base_path / name / CLIP_FILENAME).as_posix())


def _check_paths(base_path: Path | None, name: str) -> dict[str, bool | str]:
    name = name.strip()
    info: dict[str, bool | str] = {"name": name}
    if not base_path:
        info["base_path"] = "(sin prefijo)"
        info["dir_exists"] = False
        info["clip_exists"] = False
        return info

    dir_path = base_path / name
    clip_path = dir_path / CLIP_FILENAME
    info["base_path"] = str(base_path)
    info["dir_path"] = str(dir_path)
    info["clip_path"] = str(clip_path)
    info["dir_exists"] = dir_path.is_dir()
    info["clip_exists"] = clip_path.is_file()
    return info


def _print_row(
    index: int,
    total: int,
    row: dict[str, str],
    video_path: str,
    inicio: str,
    fin: str,
    path_info: dict,
) -> None:
    print(f"\n--- Fila {index + 1}/{total} ---")
    print(f"  id:         {row['id']}")
    print(f"  name:       {row['name']}")
    print(f"  start/end:  {row['start']} → {row['end']} s")
    print(f"  deleted:    {row['deleted']}  |  labeled: {row['labeled']}")
    print(f"  → salida:   {video_path}, {inicio}, {fin}, {CLASIFICACION_ROBO}")

    if path_info.get("base_path") != "(sin prefijo)":
        dir_ok = "✓" if path_info["dir_exists"] else "✗"
        clip_ok = "✓" if path_info["clip_exists"] else "✗"
        print(f"  carpeta:    [{dir_ok}] {path_info.get('dir_path', '')}")
        print(f"  clip.mp4:   [{clip_ok}] {path_info.get('clip_path', '')}")
    else:
        print("  (sin --base-path: no se comprueba existencia en disco)")


def parse_theft_web_csv(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    base_path: str | Path | None = None,
    skip_deleted: bool = True,
    only_labeled: bool = False,
    pause: bool = False,
    quiet: bool = False,
) -> Path:
    input_csv = Path(input_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"CSV no encontrado: {input_csv}")

    rows_in = _read_input_csv(input_csv)
    prefix = Path(base_path).expanduser().resolve() if base_path else None
    if prefix and not quiet:
        print(f"Prefijo base-path: {prefix}")
        if not prefix.is_dir():
            print(f"[WARN] El prefijo no existe o no es carpeta: {prefix}")

    rows_out: list[dict[str, str | int]] = []
    total = len(rows_in)
    skipped = 0

    for i, row in enumerate(rows_in):
        if skip_deleted and _is_truthy(row["deleted"]):
            skipped += 1
            if not quiet:
                print(f"\n[skip] Fila {i + 1}: deleted={row['deleted']}")
            continue
        if only_labeled and not _is_truthy(row["labeled"]):
            skipped += 1
            if not quiet:
                print(f"\n[skip] Fila {i + 1}: labeled={row['labeled']}")
            continue

        try:
            start = int(float(row["start"]))
            end = int(float(row["end"]))
        except (TypeError, ValueError) as exc:
            skipped += 1
            print(f"\n[skip] Fila {i + 1}: start/end inválidos ({exc})")
            continue

        if end < start:
            skipped += 1
            print(f"\n[skip] Fila {i + 1}: end ({end}) < start ({start})")
            continue

        name = row["name"]
        video_path = _video_path_with_prefix(prefix, name)
        inicio = seconds_to_hms(start)
        fin = seconds_to_hms(end)
        path_info = _check_paths(prefix, name)

        if not quiet:
            _print_row(i, total, row, video_path, inicio, fin, path_info)
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
                "#clasificacion": CLASIFICACION_ROBO,
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
        description="CSV web (id,name,video_url,start,end,...) → video_path,inicio,fin,#clasificacion",
    )
    parser.add_argument("csv", type=Path, help="CSV de entrada")
    parser.add_argument("-o", "--output", type=Path, default=None, help="CSV de salida")
    parser.add_argument(
        "--base-path",
        "-b",
        type=Path,
        default=None,
        help="Prefijo para video_path (p. ej. /data/clips). Comprueba carpeta y clip.mp4",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Incluir filas con deleted=true (por defecto se omiten)",
    )
    parser.add_argument(
        "--only-labeled",
        action="store_true",
        help="Solo filas con labeled=true",
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
        parse_theft_web_csv(
            args.csv,
            output_csv=args.output,
            base_path=args.base_path,
            skip_deleted=not args.include_deleted,
            only_labeled=args.only_labeled,
            pause=args.pause,
            quiet=args.quiet,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
