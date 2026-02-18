"""
Generador de marcadores ArUco en PDF tamaño A4.
Máxima calidad (300 DPI), bordes optimizados para que el marcador sea lo más grande posible.
"""
import argparse
import io
from pathlib import Path

import numpy as np

# ArUco requiere opencv-contrib-python (no solo opencv-python)
try:
    import cv2
    from cv2 import aruco
except ImportError:
    raise ImportError(
        "Se necesita opencv-contrib-python para ArUco. Instala con: pip install opencv-contrib-python"
    )

# ReportLab para PDF de alta calidad
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
except ImportError:
    raise ImportError("Se necesita reportlab. Instala con: pip install reportlab")

# A4 en mm
A4_W_MM = 210
A4_H_MM = 297
# DPI para máxima calidad
DPI = 300
# Márgenes mínimos en mm (optimizados para maximizar el marcador)
MARGIN_MM = 5


def _generate_aruco_image(marker_id: int, size_px: int, dictionary_type: int = aruco.DICT_6X6_250) -> np.ndarray:
    """Genera imagen del marcador ArUco a alta resolución."""
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    # generateImageMarker (OpenCV 4.7+) o drawMarker (versiones anteriores)
    if hasattr(aruco, "generateImageMarker"):
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id, size_px, borderBits=1)
    elif hasattr(aruco, "drawMarker"):
        marker_img = aruco.drawMarker(aruco_dict, marker_id, size_px, borderBits=1)
    else:
        raise RuntimeError("OpenCV ArUco: ni generateImageMarker ni drawMarker disponibles. Usa opencv-contrib-python.")
    return marker_img


def generate_aruco_pdf(
    marker_id: int,
    output_path: str | Path,
    margin_mm: float = MARGIN_MM,
    dpi: int = DPI,
    dictionary_type: int = aruco.DICT_6X6_250,
) -> Path:
    """
    Genera un PDF A4 con el marcador ArUco centrado y lo más grande posible.

    Args:
        marker_id: ID del marcador (0-249 para DICT_6X6_250).
        output_path: Ruta del PDF de salida.
        margin_mm: Márgenes en mm (menor = marcador más grande).
        dpi: Resolución para calidad de impresión (300 = alta).
        dictionary_type: Diccionario ArUco (p. ej. aruco.DICT_6X6_250).

    Returns:
        Path del PDF generado.
    """
    output_path = Path(output_path)

    # Área usable en mm
    usable_w = A4_W_MM - 2 * margin_mm
    usable_h = A4_H_MM - 2 * margin_mm
    # El marcador es cuadrado: limitado por el lado corto del área usable
    marker_mm = min(usable_w, usable_h)

    # Pixels del marcador para esa dimensión física a la DPI dada
    # 1 inch = 25.4 mm  ->  pixels = mm * (dpi / 25.4)
    marker_px = int(marker_mm * dpi / 25.4)
    # Múltiplo de 8 para evitar aliasing
    marker_px = (marker_px // 8) * 8
    marker_px = max(marker_px, 64)

    # Generar imagen del marcador
    marker_img = _generate_aruco_image(marker_id, marker_px, dictionary_type)

    # Guardar a PNG temporal (o en memoria) para insertar en PDF
    # ReportLab trabaja en puntos: 1 punto = 1/72 inch
    # Tamaño en puntos: mm * (72/25.4)
    mm_to_pt = 72 / 25.4
    marker_w_pt = marker_mm * mm_to_pt
    marker_h_pt = marker_mm * mm_to_pt

    # Página A4 en puntos
    page_w, page_h = A4
    # Centrar el marcador
    x_pt = (page_w - marker_w_pt) / 2
    y_pt = (page_h - marker_h_pt) / 2

    # Crear PDF
    c = canvas.Canvas(str(output_path), pagesize=A4)

    # Guardar imagen a bytes (PNG)
    _, buf = cv2.imencode(".png", marker_img)
    img_bytes = buf.tobytes()

    img_reader = ImageReader(io.BytesIO(img_bytes))

    # Dibujar imagen centrada
    c.drawImage(img_reader, x_pt, y_pt, width=marker_w_pt, height=marker_h_pt)

    c.save()
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Genera un PDF A4 con marcador ArUco de máxima calidad y bordes optimizados."
    )
    parser.add_argument("id", type=int, help="ID del marcador ArUco (0-249 para DICT_6X6_250)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Ruta del PDF de salida (default: arcuco_ID.pdf)")
    parser.add_argument("--margin", type=float, default=MARGIN_MM, help=f"Márgenes en mm (default: {MARGIN_MM})")
    parser.add_argument("--dpi", type=int, default=DPI, help=f"Resolución para impresión (default: {DPI})")
    args = parser.parse_args()

    if args.id < 0 or args.id > 249:
        print("Advertencia: DICT_6X6_250 tiene IDs 0-249. IDs fuera de rango pueden dar error.")

    output = args.output or f"arcuco_{args.id}.pdf"
    path = generate_aruco_pdf(
        marker_id=args.id,
        output_path=output,
        margin_mm=args.margin,
        dpi=args.dpi,
    )
    print(f"PDF generado: {path}")
    print(f"  Marcador ID: {args.id}")
    marker_size_mm = min(A4_W_MM - 2*args.margin, A4_H_MM - 2*args.margin)
    print(f"  Tamaño del marcador: {marker_size_mm:.0f}×{marker_size_mm:.0f} mm")


if __name__ == "__main__":
    main()
