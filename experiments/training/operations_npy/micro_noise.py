"""
Anade ruido gaussiano (microruido) a las coordenadas X e Y de un .npy.

Uso:
    python micro_noise.py ./user_14058/poses_full.npy ./user_14058/poses_noisy.npy 0.005 0.005
"""

import argparse
import os

import numpy as np


def micro_noise(
    input_npy_path: str,
    output_npy_path: str,
    sigma_x: float,
    sigma_y: float,
) -> None:
    """
    Suma ruido gaussiano independiente a x e y: N(0, sigma_x^2) y N(0, sigma_y^2).

    sigma_x y sigma_y estan en la misma escala normalizada que las poses [0, 1]
    (ej. 0.005 ~ medio punto por mil en cada eje).

    Recorta el resultado a [0, 1]. Soporta (T, J, 2), (T, 2, J, 2), etc.
    """
    if sigma_x < 0 or sigma_y < 0:
        raise ValueError("sigma_x y sigma_y deben ser >= 0")

    data = np.load(input_npy_path)

    if data.ndim < 2 or data.shape[-1] < 2:
        raise ValueError(
            f"Formato no soportado: {data.shape}. "
            "Se esperaba que la ultima dimension tuviera al menos 2 valores (x, y)."
        )

    rng = np.random.default_rng()
    noisy = data.astype(np.float64, copy=True)

    shape_xy = noisy[..., 0].shape
    noisy[..., 0] = noisy[..., 0] + rng.normal(0.0, sigma_x, size=shape_xy)
    noisy[..., 1] = noisy[..., 1] + rng.normal(0.0, sigma_y, size=shape_xy)

    noisy[..., 0] = np.clip(noisy[..., 0], 0.0, 1.0)
    noisy[..., 1] = np.clip(noisy[..., 1], 0.0, 1.0)

    if data.dtype != np.float64:
        noisy = noisy.astype(data.dtype, copy=False)

    output_dir = os.path.dirname(output_npy_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.save(output_npy_path, noisy)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anade microruido gaussiano a un archivo .npy de poses."
    )
    parser.add_argument("input_npy", type=str, help="Ruta del archivo .npy de entrada")
    parser.add_argument("output_npy", type=str, help="Ruta del archivo .npy de salida")
    parser.add_argument(
        "sigma_x",
        type=float,
        help="Desviacion tipica del ruido en X (espacio normalizado, ej. 0.005)",
    )
    parser.add_argument(
        "sigma_y",
        type=float,
        help="Desviacion tipica del ruido en Y (espacio normalizado, ej. 0.005)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_npy):
        print(f"Archivo de entrada no encontrado: {args.input_npy}")
        raise SystemExit(1)

    micro_noise(args.input_npy, args.output_npy, args.sigma_x, args.sigma_y)
    print(f"Archivo con ruido guardado en: {args.output_npy}")


if __name__ == "__main__":
    main()
