"""
Parsea un CSV con columnas Archivo, Hora inicio, Hora fin, Clasificación
y genera otro con video, inicio, fin, clasificacion.
Convierte rutas Windows (\) a formato Linux (/) y extrae camX/archivo.avi.
"""
import argparse
import os
import re

import pandas as pd


def path_to_linux_video(archivo: str) -> str:
    """
    De '2025.12.15\\cam1\\17_00___21_00.avi' extrae 'cam1/17_00___21_00.avi'.
    Toma las últimas dos partes del path (carpeta + fichero) y usa /.
    """
    s = str(archivo).strip().replace("\\", "/")
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return s if parts else ""


def _looks_like_time(val: str) -> bool:
    """Comprueba si el valor parece un tiempo (MM:SS o HH:MM:SS), no una fecha/header."""
    s = str(val).strip()
    if ":" not in s or "." in s:
        return False
    parts = s.split(":")
    if len(parts) not in (2, 3):
        return False
    try:
        for p in parts:
            int(p)
        return True
    except ValueError:
        return False


def parse_mm_ss_to_hms(t: str) -> str:
    """Convierte 'MM:SS' o 'M:SS' a 'HH:mm:ss'. Si ya tiene H:MM:SS, lo normaliza."""
    s = str(t).strip()
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = int(parts[0]), int(parts[1])
        h = 0
    elif len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        raise ValueError(f"Formato de tiempo no válido: {s}")
    total_secs = h * 3600 + m * 60 + sec
    h = total_secs // 3600
    m = (total_secs % 3600) // 60
    s = total_secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_csv(input_csv: str, output_csv: str | None = None, skip_rows: int = 0) -> str:
    """
    Lee CSV con: Archivo, Hora inicio, Hora fin, Clasificación
    y genera: video, inicio, fin, clasificacion
    skip_rows: filas a saltar al inicio (p. ej. 1 si la primera fila es título/metadatos)
    """
    df = pd.read_csv(input_csv, sep=None, engine="python", skiprows=skip_rows)
    df.columns = df.columns.str.strip()

    # Aceptar variantes de nombres de columna
    col_inicio = next((c for c in df.columns if "inicio" in c.lower() or "hora inicio" in c.lower()), None)
    col_fin = next((c for c in df.columns if "fin" in c.lower() and "hora" in c.lower()), None)
    if not col_fin:
        col_fin = next((c for c in df.columns if "fin" in c.lower()), None)
    col_archivo = next((c for c in df.columns if "archivo" in c.lower()), None)
    col_clas = next((c for c in df.columns if "clasificación" in c.lower() or "clasificacion" in c.lower()), None)

    if not all([col_archivo, col_inicio, col_fin, col_clas]):
        raise ValueError("Faltan columnas. Esperadas: Archivo, Hora inicio, Hora fin, Clasificación")

    videos, inicios, fines, clasifs = [], [], [], []
    for idx, row in df.iterrows():
        val_inicio = row[col_inicio]
        val_fin = row[col_fin]
        if not _looks_like_time(val_inicio) or not _looks_like_time(val_fin):
            continue  # Saltar filas que no parecen datos (header repetido, metadatos, etc.)
        try:
            videos.append(path_to_linux_video(row[col_archivo]))
            inicios.append(parse_mm_ss_to_hms(val_inicio))
            fines.append(parse_mm_ss_to_hms(val_fin))
            clasifs.append(int(row[col_clas]))
        except (ValueError, TypeError) as e:
            print(f"  [skip] Fila {idx + 1 + skip_rows}: {e}")

    out_df = pd.DataFrame({
        "video": videos,
        "inicio": inicios,
        "fin": fines,
        "clasificacion": clasifs,
    })

    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = base + "_parsed.csv"

    out_df.to_csv(output_csv, index=False)
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Parsea CSV con Archivo, Hora inicio, Hora fin, Clasificación → video, inicio, fin, clasificacion",
    )
    parser.add_argument("csv", type=str, help="Ruta al CSV de entrada")
    parser.add_argument("--output", "-o", type=str, default=None, help="Ruta del CSV de salida")
    parser.add_argument("--skip-rows", "-s", type=int, default=0, help="Filas a saltar al inicio (ej. 1 si la primera es título)")
    args = parser.parse_args()

    out_path = parse_csv(args.csv, args.output, skip_rows=args.skip_rows)
    print(f"CSV generado: {out_path}")


if __name__ == "__main__":
    main()
