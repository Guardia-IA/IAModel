import argparse
import os

import pandas as pd


def seconds_to_hms(total_seconds: int) -> str:
    """Convierte segundos a cadena HH:MM:SS."""
    total_seconds = int(total_seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_csv(input_csv: str, video_name: str, output_csv: str | None = None) -> str:
    """
    Lee un CSV con columnas:
      Clip, Hora, Minuto, Segundo, Duración, Código

    y genera un CSV con columnas:
      video, inicio, fin, clasificacion
    """
    # Detectar separador automáticamente (coma, tab, etc.)
    df = pd.read_csv(input_csv, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    required_cols = ["Hora", "Minuto", "Segundo", "Duración", "Código"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Columna requerida no encontrada en el CSV: '{col}'")

    videos = []
    inicios = []
    fines = []
    clasifs = []

    for _, row in df.iterrows():
        try:
            h = int(row["Hora"])
            m = int(row["Minuto"])
            s = int(row["Segundo"])
            dur = int(row["Duración"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error convirtiendo tiempo en fila {_}: {e}")

        inicio_secs = h * 3600 + m * 60 + s
        fin_secs = inicio_secs + dur

        videos.append(video_name)
        inicios.append(seconds_to_hms(inicio_secs))
        fines.append(seconds_to_hms(fin_secs))
        clasifs.append(int(row["Código"]))

    out_df = pd.DataFrame(
        {
            "video": videos,
            "inicio": inicios,
            "fin": fines,
            "clasificacion": clasifs,
        }
    )

    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = base + "_parsed.csv"

    out_df.to_csv(output_csv, index=False)
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parsea un CSV de clips con columnas "
            "'Clip, Hora, Minuto, Segundo, Duración, Código' "
            "y genera 'video,inicio,fin,clasificacion'."
        )
    )
    parser.add_argument("csv", type=str, help="Ruta al CSV de entrada")
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        required=True,
        help="Nombre del fichero de vídeo (se rellenará en la columna 'video')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Ruta del CSV de salida (por defecto: <csv>_parsed.csv)",
    )
    args = parser.parse_args()

    out_path = parse_csv(args.csv, args.video, args.output)
    print(f"CSV generado: {out_path}")


if __name__ == "__main__":
    main()

