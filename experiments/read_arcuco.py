#!/usr/bin/env python3
"""
read_arcuco.py

Detecta marcadores ArUco en una imagen y muestra sus IDs numéricos.
Usa la misma lógica de detección que split_videos.py.

Uso:
    python read_arcuco.py /ruta/a/imagen.jpg
    python read_arcuco.py /ruta/a/imagen.jpg --dict DICT_6X6_250
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

try:
    from cv2 import aruco
except ImportError as exc:
    raise ImportError(
        "Se necesita opencv-contrib-python para ArUco. "
        "Instala con: pip install opencv-contrib-python"
    ) from exc


def _get_aruco_detector(aruco_dict, params):
    """Devuelve el detector ArUco según la versión de OpenCV."""
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(aruco_dict, params)
    return None


def _detect_aruco(frame, aruco_dict, params):
    """
    Detecta ArUcos en el frame.
    Returns: (corners, ids) donde corners es lista de arrays 4x2, ids es array 1D.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    detector = _get_aruco_detector(aruco_dict, params)
    if detector is not None:
        _, ids, _ = detector.detectMarkers(gray)
    else:
        _, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return ids


def _resolve_dict_name(dict_name: str) -> str:
    """
    Resuelve el nombre del diccionario ArUco de forma tolerante a mayúsculas.
    Ejemplo: DICT_6x6_250 -> DICT_6X6_250.
    """
    name = str(dict_name).strip()
    if not name:
        raise ValueError("Debes indicar un diccionario ArUco válido.")

    normalized = name.upper()
    if hasattr(aruco, normalized):
        return normalized

    for attr in dir(aruco):
        if attr.startswith("DICT_") and attr.upper() == normalized:
            return attr

    raise ValueError(
        f"Diccionario ArUco no válido: {dict_name}. "
        "Ejemplo correcto: DICT_6X6_250"
    )


def detect_aruco_ids(image_path: Path, dict_name: str) -> list[int]:
    """Devuelve una lista ordenada con IDs ArUco detectados en la imagen."""
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"No se pudo leer la imagen (formato no soportado o corrupta): {image_path}")

    resolved_dict_name = _resolve_dict_name(dict_name)
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, resolved_dict_name))
    params = aruco.DetectorParameters()
    if hasattr(params, "adaptiveThreshWinSizeMin"):
        params.adaptiveThreshWinSizeMin = 3
    if hasattr(params, "adaptiveThreshWinSizeMax"):
        params.adaptiveThreshWinSizeMax = 23

    ids = _detect_aruco(image, aruco_dict, params)
    if ids is None or ids.size == 0:
        return []
    return sorted(int(marker_id) for marker_id in ids.flatten())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecta ArUcos en una imagen y muestra sus IDs numéricos."
    )
    parser.add_argument("image", type=str, help="Ruta de la imagen a analizar")
    parser.add_argument(
        "--dict",
        dest="dict_name",
        type=str,
        default="DICT_6X6_250",
        help="Diccionario ArUco a usar (default: DICT_6X6_250)",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    ids = detect_aruco_ids(image_path, args.dict_name)

    if not ids:
        print("No se detectaron ArUcos en la imagen.")
        return

    print(f"ArUcos detectados ({len(ids)}): {ids}")
    print("IDs:", ", ".join(str(x) for x in ids))


if __name__ == "__main__":
    main()
