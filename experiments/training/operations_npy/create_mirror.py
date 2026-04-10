"""
Genera una versión espejo horizontal de un archivo de poses .npy.

Uso:
    python create_mirror.py ./user_14058/poses_full.npy ./user_14058/poses_full_mirror.npy
"""

import argparse
import os

import numpy as np


def create_mirror(input_npy_path: str, output_npy_path: str) -> None:
    """
    Crea un archivo .npy espejado horizontalmente.

    Asume que la coordenada X está normalizada en [0, 1], por lo que aplica:
        x_mirror = 1.0 - x

    Soporta arreglos donde la última dimensión contiene al menos (x, y),
    por ejemplo:
    - (T, J, 2)
    - (T, 2, J, 2)

    Si J==8 (KEEP_KPS = [5,6,7,8,9,10,11,12]), además permuta pares
    izquierda/derecha para mantener coherencia semántica de landmarks.
    """
    data = np.load(input_npy_path)

    if data.ndim < 2 or data.shape[-1] < 2:
        raise ValueError(
            f"Formato no soportado: {data.shape}. "
            "Se esperaba que la última dimensión tuviera al menos 2 valores (x, y)."
        )

    mirrored = data.copy()
    mirrored[..., 0] = 1.0 - mirrored[..., 0]
    if mirrored.ndim >= 3 and mirrored.shape[-2] == 8:
        lr_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
        for li, ri in lr_pairs:
            tmp = mirrored[..., li, :].copy()
            mirrored[..., li, :] = mirrored[..., ri, :]
            mirrored[..., ri, :] = tmp

    output_dir = os.path.dirname(output_npy_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.save(output_npy_path, mirrored)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crear un archivo .npy con efecto espejo horizontal."
    )
    parser.add_argument("input_npy", type=str, help="Ruta del archivo .npy de entrada")
    parser.add_argument("output_npy", type=str, help="Ruta del archivo .npy de salida")
    args = parser.parse_args()

    if not os.path.exists(args.input_npy):
        print(f"Archivo de entrada no encontrado: {args.input_npy}")
        raise SystemExit(1)

    create_mirror(args.input_npy, args.output_npy)
    print(f"Archivo espejado guardado en: {args.output_npy}")


if __name__ == "__main__":
    main()
