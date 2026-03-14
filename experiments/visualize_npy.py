"""
Visualiza poses desde un archivo .npy usando Tkinter (sin OpenCV/Qt).
Soporta formato [T, J, 2] (un usuario) y [T, 2, J, 2] (dos usuarios).
"""
import argparse
import os
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

# Conexiones: hombros, codos, muñecas, cadera (sin piernas)
CONNECTIONS = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7)]

W, H = 600, 600


def draw_skeleton_pil(draw, points, color_line, color_pt):
    """Dibuja un esqueleto con PIL ImageDraw."""
    for start, end in CONNECTIONS:
        if start < len(points) and end < len(points):
            x1 = int(points[start][0] * W)
            y1 = int(points[start][1] * H)
            x2 = int(points[end][0] * W)
            y2 = int(points[end][1] * H)
            draw.line([(x1, y1), (x2, y2)], fill=color_line, width=2)
    r = 4
    for pt in points:
        x = int(pt[0] * W)
        y = int(pt[1] * H)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color_pt, outline=color_pt)


def visualize_skeleton(npy_path, fps: float = 20.0):
    data = np.load(npy_path)
    print(f"Visualizando: {npy_path}")
    print(f"Dimensiones: {data.shape}")

    if data.ndim == 3:
        n_frames = len(data)
        multi_user = False
    elif data.ndim == 4 and data.shape[1] == 2:
        n_frames = len(data)
        multi_user = True
    else:
        print(f"Formato no soportado. Esperado (T,J,2) o (T,2,J,2), recibido {data.shape}")
        return

    delay_ms = max(1, int(1000 / fps)) if fps > 0 else 50
    frame_idx = [0]  # lista para poder modificar desde el closure

    root = tk.Tk()
    root.title("Verificador de Esqueletos")
    root.resizable(False, False)

    label = tk.Label(root)
    label.pack()

    def build_frame(idx):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza poses desde un archivo .npy")
    parser.add_argument("npy_path", type=str, help="Ruta al archivo poses.npy")
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="FPS a los que reproducir la secuencia (por defecto: 20)",
    )
    args = parser.parse_args()

    if os.path.exists(args.npy_path):
        visualize_skeleton(args.npy_path, fps=args.fps)
    else:
        print(f"Archivo no encontrado: {args.npy_path}")
        exit(1)
