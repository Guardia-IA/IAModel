"""
Escala poses de un archivo .npy segun un porcentaje.

Uso:
    python Scale.py ./user_14058/poses_full.npy ./user_14058/poses_full_scaled.npy 120
"""

import argparse
import os

import numpy as np


def Scale(input_npy_path: str, output_npy_path: str, percentage: float) -> None:
    """
    Escala puntos (x, y) alrededor del centro (0.5, 0.5).

    percentage:
    - 100  -> mantiene el tamano original
    - 120  -> aumenta un 20%
    - 80   -> reduce un 20%

    Para evitar coordenadas invalidas, recorta (clip) el resultado a [0, 1].
    Soporta arreglos con ultima dimension de al menos (x, y), por ejemplo:
    - (T, J, 2)
    - (T, 2, J, 2)
    """
    data = np.load(input_npy_path)

    if data.ndim < 2 or data.shape[-1] < 2:
        raise ValueError(
            f"Formato no soportado: {data.shape}. "
            "Se esperaba que la ultima dimension tuviera al menos 2 valores (x, y)."
        )

    factor = percentage / 100.0
    scaled = data.copy()

    x = scaled[..., 0]
    y = scaled[..., 1]

    x = (x - 0.5) * factor + 0.5
    y = (y - 0.5) * factor + 0.5

    # Evita puntos fuera de imagen tras el escalado.
    scaled[..., 0] = np.clip(x, 0.0, 1.0)
    scaled[..., 1] = np.clip(y, 0.0, 1.0)

    output_dir = os.path.dirname(output_npy_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.save(output_npy_path, scaled)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Escala un archivo .npy de poses segun un porcentaje."
    )
    parser.add_argument("input_npy", type=str, help="Ruta del archivo .npy de entrada")
    parser.add_argument("output_npy", type=str, help="Ruta del archivo .npy de salida")
    parser.add_argument(
        "percentage",
        type=float,
        help="Porcentaje de escala (100 = original, 120 = +20%, 80 = -20%)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_npy):
        print(f"Archivo de entrada no encontrado: {args.input_npy}")
        raise SystemExit(1)

    Scale(args.input_npy, args.output_npy, args.percentage)
    print(
        f"Archivo escalado ({args.percentage}%) guardado en: {args.output_npy}"
    )


if __name__ == "__main__":
    main()
