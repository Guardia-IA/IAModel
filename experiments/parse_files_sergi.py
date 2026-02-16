"""
Parsea un CSV con columnas CLIP, INICIO, SEGUNDOS, CLASIFICACIÓN
y genera otro con video, inicio, fin, clasificacion.
"""
import argparse
import os
import re

import pandas as pd

HMS_PATTERN = re.compile(r"^(\d+):(\d{2}):(\d{2})$")


def parse_hms(hms_str: str) -> int:
    """Convierte 'H:MM:SS' o 'HH:MM:SS' a segundos totales."""
    s = str(hms_str).strip()
    m = HMS_PATTERN.match(s)
    if not m:
        # Intentar variantes como 0:00:09
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + m * 60 + sec
        raise ValueError(f"Formato de tiempo no válido: {s}")
    h, m, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + m * 60 + sec


def seconds_to_hms(total_seconds: int) -> str:
    """Convierte segundos a cadena HH:mm:ss."""
    total_seconds = int(total_seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_csv(input_csv: str, video_name: str, output_csv: str | None = None) -> str:
    """
    Lee un CSV con columnas: CLIP, INICIO, SEGUNDOS, CLASIFICACIÓN
    y genera otro con: video, inicio, fin, clasificacion
    """
    df = pd.read_csv(input_csv, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    required_cols = ["INICIO", "SEGUNDOS", "CLASIFICACIÓN"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Columna requerida no encontrada: '{col}'")

    videos = []
    inicios = []
    fines = []
    clasifs = []

    for idx, row in df.iterrows():
        inicio_secs = parse_hms(row["INICIO"])
        segundos = int(row["SEGUNDOS"])
        fin_secs = inicio_secs + segundos

        videos.append(video_name)
        inicios.append(seconds_to_hms(inicio_secs))
        fines.append(seconds_to_hms(fin_secs))
        clasifs.append(int(row["CLASIFICACIÓN"]))

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
        description="Parsea CSV con CLIP, INICIO, SEGUNDOS, CLASIFICACIÓN → video, inicio, fin, clasificacion",
    )
    parser.add_argument("csv", type=str, help="Ruta al CSV de entrada")
    parser.add_argument("--video", "-v", type=str, required=True, help="Nombre del fichero de vídeo")
    parser.add_argument("--output", "-o", type=str, default=None, help="Ruta del CSV de salida")
    args = parser.parse_args()

    out_path = parse_csv(args.csv, args.video, args.output)
    print(f"CSV generado: {out_path}")


if __name__ == "__main__":
    main()
