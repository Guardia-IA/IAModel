"""
Desplaza poses de un archivo .npy en X e Y (coordenadas normalizadas).

Uso:
    python Shift.py ./user_14058/poses_full.npy ./user_14058/poses_shifted.npy 0.05 -0.03
"""

import argparse
import os

import numpy as np


def Shift(
    input_npy_path: str, output_npy_path: str, dx: float, dy: float
) -> None:
    """
    Desplaza puntos (x, y) sumando dx y dy en espacio normalizado [0, 1].

    dx, dy son fracciones del ancho/alto (ej. 0.05 = 5% hacia la derecha en x;
    dy positivo baja el esqueleto si y crece hacia abajo).

    Recorta el resultado a [0, 1] para evitar coordenadas invalidas.
    Soporta (T, J, 2) y (T, 2, J, 2), etc., con ultima dimension >= 2.
    """
    data = np.load(input_npy_path)

    if data.ndim < 2 or data.shape[-1] < 2:
        raise ValueError(
            f"Formato no soportado: {data.shape}. "
            "Se esperaba que la ultima dimension tuviera al menos 2 valores (x, y)."
        )

    shifted = data.copy()
    shifted[..., 0] = np.clip(shifted[..., 0] + dx, 0.0, 1.0)
    shifted[..., 1] = np.clip(shifted[..., 1] + dy, 0.0, 1.0)

    output_dir = os.path.dirname(output_npy_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.save(output_npy_path, shifted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Desplaza un archivo .npy de poses en X e Y."
    )
    parser.add_argument("input_npy", type=str, help="Ruta del archivo .npy de entrada")
    parser.add_argument("output_npy", type=str, help="Ruta del archivo .npy de salida")
    parser.add_argument(
        "dx",
        type=float,
        help="Desplazamiento en X (normalizado, ej. 0.05 = 5%% a la derecha)",
    )
    parser.add_argument(
        "dy",
        type=float,
        help="Desplazamiento en Y (normalizado, ej. -0.03 hacia arriba si y crece abajo)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_npy):
        print(f"Archivo de entrada no encontrado: {args.input_npy}")
        raise SystemExit(1)

    Shift(args.input_npy, args.output_npy, args.dx, args.dy)
    print(f"Archivo desplazado guardado en: {args.output_npy}")


if __name__ == "__main__":
    main()
