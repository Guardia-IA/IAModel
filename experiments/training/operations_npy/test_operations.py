"""
Compara visualmente un .npy original vs transformado en tiempo real.

Uso basico:
    python test_operations.py /ruta/poses.npy mirror
    python test_operations.py /ruta/poses.npy rotate --degrees 15
    python test_operations.py /ruta/poses.npy scale --percentage 120
    python test_operations.py /ruta/poses.npy shift --dx 0.05 --dy -0.03
    python test_operations.py /ruta/poses.npy noise --sigma-x 0.005 --sigma-y 0.005

Pipeline (varias operaciones en orden):
    python test_operations.py /ruta/poses.npy "mirror,rotate:12,noise:0.003:0.003"
    python test_operations.py /ruta/poses.npy "scale:110,shift:0.03:-0.02"
"""

import argparse
import os
import math
import tkinter as tk
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageTk


# Conexiones: hombros, codos, muñecas, cadera (sin piernas)
CONNECTIONS = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7)]
W, H = 600, 600


def draw_skeleton_pil(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    color_line,
    color_pt,
    x_offset: int = 0,
) -> None:
    for start, end in CONNECTIONS:
        if start < len(points) and end < len(points):
            x1 = int(points[start][0] * W) + x_offset
            y1 = int(points[start][1] * H)
            x2 = int(points[end][0] * W) + x_offset
            y2 = int(points[end][1] * H)
            draw.line([(x1, y1), (x2, y2)], fill=color_line, width=2)
    r = 4
    for pt in points:
        x = int(pt[0] * W) + x_offset
        y = int(pt[1] * H)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color_pt, outline=color_pt)


def _check_pose_shape(data: np.ndarray) -> None:
    if data.ndim < 3 or data.shape[-1] < 2:
        raise ValueError(
            f"Formato no soportado: {data.shape}. Se espera algo como (T, J, 2) o (T, U, J, 2)."
        )


def op_mirror(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    out[..., 0] = 1.0 - out[..., 0]
    # KEEP_KPS = [5,6,7,8,9,10,11,12] => intercambiar izquierda/derecha
    # tras el espejo para preservar semántica de joints.
    if out.ndim >= 3 and out.shape[-2] == 8:
        lr_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
        for li, ri in lr_pairs:
            tmp = out[..., li, :].copy()
            out[..., li, :] = out[..., ri, :]
            out[..., ri, :] = tmp
    return out


def op_rotate(data: np.ndarray, degrees: float) -> np.ndarray:
    out = data.copy()
    theta = math.radians(degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x = out[..., 0] - 0.5
    y = out[..., 1] - 0.5
    out[..., 0] = x * cos_t - y * sin_t + 0.5
    out[..., 1] = x * sin_t + y * cos_t + 0.5
    return np.clip(out, 0.0, 1.0)


def op_scale(data: np.ndarray, percentage: float) -> np.ndarray:
    out = data.copy()
    factor = percentage / 100.0
    out[..., 0] = (out[..., 0] - 0.5) * factor + 0.5
    out[..., 1] = (out[..., 1] - 0.5) * factor + 0.5
    return np.clip(out, 0.0, 1.0)


def op_shift(data: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = data.copy()
    out[..., 0] = np.clip(out[..., 0] + dx, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + dy, 0.0, 1.0)
    return out


def op_noise(data: np.ndarray, sigma_x: float, sigma_y: float, seed: int | None = None) -> np.ndarray:
    if sigma_x < 0 or sigma_y < 0:
        raise ValueError("sigma_x y sigma_y deben ser >= 0")
    rng = np.random.default_rng(seed)
    out = data.astype(np.float64, copy=True)
    shape_xy = out[..., 0].shape
    out[..., 0] = out[..., 0] + rng.normal(0.0, sigma_x, size=shape_xy)
    out[..., 1] = out[..., 1] + rng.normal(0.0, sigma_y, size=shape_xy)
    out[..., 0] = np.clip(out[..., 0], 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1], 0.0, 1.0)
    if data.dtype != np.float64:
        out = out.astype(data.dtype, copy=False)
    return out


def apply_operation(data: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, str]:
    op = args.operation.lower()
    if op == "mirror":
        return op_mirror(data), "mirror (x -> 1-x)"
    if op == "rotate":
        return op_rotate(data, args.degrees), f"rotate ({args.degrees} deg)"
    if op == "scale":
        return op_scale(data, args.percentage), f"scale ({args.percentage}%)"
    if op == "shift":
        return op_shift(data, args.dx, args.dy), f"shift (dx={args.dx}, dy={args.dy})"
    if op == "noise":
        return op_noise(data, args.sigma_x, args.sigma_y, seed=args.seed), (
            f"noise (sx={args.sigma_x}, sy={args.sigma_y})"
        )
    raise ValueError(f"Operacion no soportada: {args.operation}")


def apply_pipeline(data: np.ndarray, pipeline: str, seed: int | None = None) -> Tuple[np.ndarray, str]:
    """
    Aplica una cadena de operaciones separadas por comas.
    Sintaxis de cada paso:
      - mirror
      - rotate:grados
      - scale:porcentaje
      - shift:dx:dy
      - noise:sigma_x:sigma_y
    """
    text = (pipeline or "").strip()
    if not text:
        raise ValueError("Pipeline vacío.")

    out = data.copy()
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("Pipeline vacío.")

    desc_parts = []
    for step in parts:
        toks = [t.strip() for t in step.split(":")]
        op = toks[0].lower()
        if op == "mirror":
            out = op_mirror(out)
            desc_parts.append("mirror")
        elif op == "rotate":
            if len(toks) != 2:
                raise ValueError(f"Paso inválido '{step}'. Esperado: rotate:grados")
            deg = float(toks[1])
            out = op_rotate(out, deg)
            desc_parts.append(f"rotate:{deg}")
        elif op == "scale":
            if len(toks) != 2:
                raise ValueError(f"Paso inválido '{step}'. Esperado: scale:porcentaje")
            pct = float(toks[1])
            out = op_scale(out, pct)
            desc_parts.append(f"scale:{pct}")
        elif op == "shift":
            if len(toks) != 3:
                raise ValueError(f"Paso inválido '{step}'. Esperado: shift:dx:dy")
            dx = float(toks[1])
            dy = float(toks[2])
            out = op_shift(out, dx, dy)
            desc_parts.append(f"shift:{dx}:{dy}")
        elif op == "noise":
            if len(toks) != 3:
                raise ValueError(f"Paso inválido '{step}'. Esperado: noise:sigma_x:sigma_y")
            sx = float(toks[1])
            sy = float(toks[2])
            out = op_noise(out, sx, sy, seed=seed)
            desc_parts.append(f"noise:{sx}:{sy}")
        else:
            raise ValueError(f"Operación desconocida en pipeline: '{op}'")

    return out, " | ".join(desc_parts)


def draw_side(img: Image.Image, frame_data: np.ndarray, title: str, x_offset: int) -> None:
    draw = ImageDraw.Draw(img)
    draw.rectangle([x_offset, 0, x_offset + W, H], fill=(0, 0, 0))
    draw_skeleton_pil(
        draw,
        frame_data,
        color_line=(0, 255, 180),
        color_pt=(255, 255, 255),
        x_offset=x_offset,
    )
    draw.text((x_offset + 10, 10), title, fill=(255, 255, 255))


def visualize_comparison(original: np.ndarray, transformed: np.ndarray, op_desc: str, fps: float) -> None:
    if original.shape != transformed.shape:
        raise ValueError(
            f"El original y transformado deben tener misma forma. "
            f"Original={original.shape}, transformado={transformed.shape}"
        )

    if original.ndim != 3:
        raise ValueError(
            f"Este comparador espera un solo usuario: shape (T, J, 2). Recibido: {original.shape}"
        )

    n_frames = original.shape[0]
    delay_ms = max(1, int(1000 / fps)) if fps > 0 else 50
    frame_idx = [0]

    root = tk.Tk()
    root.title("Comparador de Operaciones NPY (Q/Esc para salir)")
    root.resizable(False, False)

    label = tk.Label(root)
    label.pack()

    total_w = W * 2

    def build_frame(idx: int) -> Image.Image:
        img = Image.new("RGB", (total_w, H), (20, 20, 20))
        draw_side(img, original[idx], "Original", 0)
        draw_side(img, transformed[idx], f"Transformado: {op_desc}", W)
        draw = ImageDraw.Draw(img)
        draw.line([(W, 0), (W, H)], fill=(80, 80, 80), width=2)
        draw.text((10, H - 30), f"Frame: {idx}/{n_frames - 1}", fill=(255, 255, 255))
        return img

    def update():
        idx = frame_idx[0] % n_frames
        img = build_frame(idx)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo
        frame_idx[0] += 1
        root.after(delay_ms, update)

    def on_key(event):
        if event.char == "q" or event.keysym == "Escape":
            root.quit()
            root.destroy()

    root.bind("<KeyPress>", on_key)
    root.after(0, update)
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara visualmente un .npy original vs transformado por una operacion."
    )
    parser.add_argument("npy_path", type=str, help="Ruta al archivo .npy de entrada")
    parser.add_argument(
        "operation",
        type=str,
        help=(
            "Operacion simple (mirror|rotate|scale|shift|noise) "
            "o pipeline, p.ej: mirror,rotate:12,noise:0.003:0.003"
        ),
    )
    parser.add_argument("--fps", type=float, default=20.0, help="FPS de reproducción")

    # Params opcionales por operación
    parser.add_argument("--degrees", type=float, default=15.0, help="Grados para rotate")
    parser.add_argument("--percentage", type=float, default=120.0, help="Porcentaje para scale")
    parser.add_argument("--dx", type=float, default=0.05, help="Desplazamiento X para shift")
    parser.add_argument("--dy", type=float, default=-0.03, help="Desplazamiento Y para shift")
    parser.add_argument("--sigma-x", type=float, default=0.005, help="Sigma X para noise")
    parser.add_argument("--sigma-y", type=float, default=0.005, help="Sigma Y para noise")
    parser.add_argument("--seed", type=int, default=None, help="Seed opcional para noise")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.npy_path):
        print(f"Archivo no encontrado: {args.npy_path}")
        raise SystemExit(1)

    data = np.load(args.npy_path)
    _check_pose_shape(data)

    # Para comparador lado a lado simplificado usamos solo formato (T, J, 2).
    if data.ndim != 3:
        raise SystemExit(
            f"Este script espera (T, J, 2). Se recibió {data.shape}. "
            "Si tienes multiusuario (p.ej. T,2,J,2), extrae un usuario antes."
        )

    op_text = args.operation.strip().lower()
    if "," in op_text or ":" in op_text:
        transformed, op_desc = apply_pipeline(data, args.operation, seed=args.seed)
    else:
        transformed, op_desc = apply_operation(data, args)
    visualize_comparison(data, transformed, op_desc, fps=args.fps)


if __name__ == "__main__":
    main()
