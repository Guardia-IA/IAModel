"""
Rota poses de un archivo .npy por un angulo en grados.

Uso:
    python rotate_X.py ./user_14058/poses_full.npy ./user_14058/poses_full_rotated.npy 15
"""

import argparse
import math
import os

import numpy as np


def rotate_X(input_npy_path: str, output_npy_path: str, degrees: float) -> None:
    """
    Rota puntos (x, y) por `degrees` alrededor del centro (0.5, 0.5).

    Soporta arreglos donde la ultima dimension contiene al menos (x, y),
    por ejemplo:
    - (T, J, 2)
    - (T, 2, J, 2)
    """
    data = np.load(input_npy_path)

    if data.ndim < 2 or data.shape[-1] < 2:
        raise ValueError(
            f"Formato no soportado: {data.shape}. "
            "Se esperaba que la ultima dimension tuviera al menos 2 valores (x, y)."
        )

    rotated = data.copy()

    theta = math.radians(degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Traslada al origen en el centro de la imagen normalizada.
    x = rotated[..., 0] - 0.5
    y = rotated[..., 1] - 0.5

    # Rotacion 2D:
    # x' = x*cos - y*sin
    # y' = x*sin + y*cos
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t

    # Devuelve al sistema original.
    rotated[..., 0] = x_rot + 0.5
    rotated[..., 1] = y_rot + 0.5

    output_dir = os.path.dirname(output_npy_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.save(output_npy_path, rotated)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rota un archivo .npy de poses por un angulo dado."
    )
    parser.add_argument("input_npy", type=str, help="Ruta del archivo .npy de entrada")
    parser.add_argument("output_npy", type=str, help="Ruta del archivo .npy de salida")
    parser.add_argument("degrees", type=float, help="Grados de rotacion")
    args = parser.parse_args()

    if not os.path.exists(args.input_npy):
        print(f"Archivo de entrada no encontrado: {args.input_npy}")
        raise SystemExit(1)

    rotate_X(args.input_npy, args.output_npy, args.degrees)
    print(
        f"Archivo rotado ({args.degrees} grados) guardado en: {args.output_npy}"
    )


if __name__ == "__main__":
    main()
