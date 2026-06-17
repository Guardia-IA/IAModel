"""
Visualiza poses desde un archivo .npy usando Tkinter (sin OpenCV para la ventana).
Soporta formato [T, J, 2] (un usuario) y [T, 2, J, 2] (dos usuarios).

Opción --save: escribe MP4 H.264 + yuv420p + faststart (compatible con WhatsApp).
Requiere ffmpeg en PATH; si no hay ffmpeg, intenta OpenCV (cv2.VideoWriter).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageTk

# Conexiones: hombros, codos, muñecas, cadera (sin piernas)
CONNECTIONS = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7)]

W, H = 600, 600


def _kp_visible(pt) -> bool:
    """Keypoint dibujable: coordenadas finitas y no (0,0) ausente."""
    if pt is None or len(pt) < 2:
        return False
    x, y = float(pt[0]), float(pt[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    return not (x == 0.0 and y == 0.0)


def draw_skeleton_pil(draw, points, color_line, color_pt):
    """Dibuja un esqueleto con PIL ImageDraw (omite keypoints NaN o ausentes)."""
    for start, end in CONNECTIONS:
        if start >= len(points) or end >= len(points):
            continue
        if not _kp_visible(points[start]) or not _kp_visible(points[end]):
            continue
        x1 = int(points[start][0] * W)
        y1 = int(points[start][1] * H)
        x2 = int(points[end][0] * W)
        y2 = int(points[end][1] * H)
        draw.line([(x1, y1), (x2, y2)], fill=color_line, width=2)
    r = 4
    for pt in points:
        if not _kp_visible(pt):
            continue
        x = int(pt[0] * W)
        y = int(pt[1] * H)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color_pt, outline=color_pt)


def load_pose_array(npy_path: str) -> Tuple[np.ndarray, bool, int]:
    data = np.load(npy_path)
    if data.ndim == 3:
        return data, False, len(data)
    if data.ndim == 4 and data.shape[1] == 2:
        return data, True, len(data)
    raise ValueError(f"Formato no soportado. Esperado (T,J,2) o (T,2,J,2), recibido {data.shape}")


def render_frame(data: np.ndarray, idx: int, multi_user: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    if multi_user:
        pts1 = data[idx, 0]
        pts2 = data[idx, 1]
        draw_skeleton_pil(draw, pts1, (0, 255, 0), (0, 255, 100))
        draw_skeleton_pil(draw, pts2, (255, 165, 0), (255, 200, 0))
        draw.text((10, 8), "Usuario 1 (robo)=verde | Usuario 2=naranja", fill=(255, 255, 255))
    else:
        draw_skeleton_pil(draw, data[idx], (255, 255, 0), (0, 0, 255))

    draw.text((10, 38), f"Frame: {idx}", fill=(255, 255, 255))
    return img


def iter_frames_pil(data: np.ndarray, multi_user: bool, n_frames: int) -> Iterator[Image.Image]:
    for idx in range(n_frames):
        yield render_frame(data, idx, multi_user)


def _encode_mp4_ffmpeg(frames: Iterator[Image.Image], out_path: Path, fps: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    fps_i = max(1, min(60, int(round(fps)))) if fps > 0 else 25
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{W}x{H}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps_i),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        for img in frames:
            proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait(timeout=600)
        if code != 0:
            print(f"[ERROR] ffmpeg falló (código {code}). stderr:\n{err[-2000:]}", file=sys.stderr)
        return code == 0
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"[ERROR] ffmpeg: {e}", file=sys.stderr)
        return False


def _encode_mp4_cv2(frames: Iterator[Image.Image], out_path: Path, fps: float) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    fps_i = float(max(1, min(60, round(fps)))) if fps > 0 else 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_i, (W, H))
    if not writer.isOpened():
        return False
    try:
        for img in frames:
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        return True
    finally:
        writer.release()


def save_animation_mp4(
    data: np.ndarray,
    multi_user: bool,
    n_frames: int,
    out_path: Path,
    fps: float,
) -> bool:
    """MP4 orientado a compatibilidad WhatsApp (H.264 + yuv420p con ffmpeg)."""
    out_path = out_path.expanduser().resolve()
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = iter_frames_pil(data, multi_user, n_frames)
    if _encode_mp4_ffmpeg(frames, out_path, fps):
        return True
    frames = iter_frames_pil(data, multi_user, n_frames)
    if _encode_mp4_cv2(frames, out_path, fps):
        print("[INFO] Guardado con OpenCV (mp4v). Para mejor compatibilidad WhatsApp, instala ffmpeg.")
        return True
    print(
        "[ERROR] No se pudo escribir vídeo: instala ffmpeg (recomendado) o opencv-python.",
        file=sys.stderr,
    )
    return False


def visualize_skeleton(npy_path: str, fps: float = 20.0) -> None:
    try:
        data, multi_user, n_frames = load_pose_array(npy_path)
    except ValueError as e:
        print(e)
        return
    print(f"Visualizando: {npy_path}")
    print(f"Dimensiones: {data.shape}")

    delay_ms = max(1, int(1000 / fps)) if fps > 0 else 50
    frame_idx = [0]  # lista para poder modificar desde el closure

    root = tk.Tk()
    root.title("Verificador de Esqueletos")
    root.resizable(False, False)

    label = tk.Label(root)
    label.pack()

    def update():
        idx = frame_idx[0] % n_frames
        img = render_frame(data, idx, multi_user)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza poses desde un archivo .npy")
    parser.add_argument("npy_path", type=str, help="Ruta al archivo poses.npy")
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="FPS a los que reproducir la secuencia (por defecto: 20)",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="OUT.mp4",
        help=(
            "Guarda la animación como MP4 (H.264, yuv420p, faststart; apto WhatsApp). "
            "Si solo indicas --save, el fichero será <npy_stem>_vis.mp4 junto al .npy."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.npy_path):
        print(f"Archivo no encontrado: {args.npy_path}")
        sys.exit(1)

    if args.save is not None:
        try:
            data, multi_user, n_frames = load_pose_array(args.npy_path)
        except ValueError as e:
            print(e)
            sys.exit(1)
        npy_p = Path(args.npy_path)
        if args.save == "__auto__":
            out = npy_p.with_name(f"{npy_p.stem}_vis.mp4")
        else:
            out = Path(args.save)
        print(f"Dimensiones: {data.shape} | frames={n_frames} | fps={args.fps}")
        if save_animation_mp4(data, multi_user, n_frames, out, args.fps):
            print(f"[OK] Vídeo guardado: {out.resolve()}")
        else:
            sys.exit(1)
        sys.exit(0)

    visualize_skeleton(args.npy_path, fps=args.fps)
