"""
Visualiza un vídeo MP4 en Tkinter con la pose (solo parte superior: hombros, codos, muñecas, cadera).
Opcional: fichero .npy para usar poses precalculadas en lugar de YOLO.
Botón para alternar entre "vídeo + esqueleto" y "solo esqueleto".

Uso:
    python visualize_video_pose.py video.mp4
    python visualize_video_pose.py video.mp4 poses.npy
    python visualize_video_pose.py video.mp4 --npy poses.npy --model yolo11n-pose.pt

Nota sobre el .npy: debe corresponder al mismo clip que el vídeo.
  - El .npy viene de pose_extractor (poses.npy o poses_full.npy en user_X/).
  - Las coordenadas están normalizadas 0-1.
  - Si vídeo y .npy no coinciden: comprueba que el vídeo sea el CLIP que generó ese npy,
    no el vídeo original completo. El npy está asociado a un clip cortado (meta.json tiene
    clip_name, t_start, t_end). Usa el clip en temp_clips o regenera el clip para comparar.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

# OpenCV solo para lectura de vídeo e inferencia YOLO (no se usa imshow)
import cv2

# Índices COCO: 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
# 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip
UPPER_KPS = [5, 6, 7, 8, 9, 10, 11, 12]
# Conexiones entre índices de UPPER_KPS (0..7): hombro-codo-muñeca, hombros, torso, cadera
CONNECTIONS = [(0, 2), (2, 4), (1, 3), (3, 5), (0, 1), (0, 6), (1, 7), (6, 7)]

DISPLAY_W, DISPLAY_H = 640, 480
MIN_CONF = 0.25  # confianza mínima para dibujar un keypoint

# Escala para YOLO (igual que pose_extractor: CLIP_SCALE_HEIGHT=1080).
# Permite comparar si YOLO a 1080p produce las mismas poses que el .npy extraído.
YOLO_SCALE_HEIGHT = 1080


def load_pose_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Se necesita 'ultralytics' para detección de poses. Ejecuta: pip install ultralytics")
        sys.exit(1)
    path = Path(model_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / model_path
    if not path.exists():
        print(f"Modelo no encontrado: {path}")
        sys.exit(1)
    return YOLO(str(path))


def get_upper_pose(results, frame_shape, choose_biggest: bool = True):
    """
    Extrae keypoints de la parte superior del cuerpo para la primera persona
    o la de bbox más grande. Devuelve (points_xy, confs) o (None, None).
    points_xy: np array (8, 2) en coordenadas de imagen.
    """
    if not results or len(results) == 0:
        return None, None
    r = results[0]
    if r.keypoints is None or r.boxes is None:
        return None, None
    kpts = r.keypoints.xy.cpu().numpy()   # (N, 17, 2) o (N, 17, 3)
    confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones((kpts.shape[0], 17))
    boxes = r.boxes.xyxy.cpu().numpy()    # (N, 4)

    if kpts.shape[0] == 0:
        return None, None

    if choose_biggest and kpts.shape[0] > 1:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
    else:
        idx = 0

    kp = kpts[idx]   # (17, 2) o (17, 3)
    if kp.shape[-1] >= 3:
        kp = kp[:, :2]
    conf = confs[idx] if idx < len(confs) else np.ones(17)
    upper_xy = kp[UPPER_KPS]   # (8, 2)
    upper_conf = conf[UPPER_KPS]
    return upper_xy, upper_conf


def draw_upper_skeleton(draw, points_xy, confs, w: int, h: int, color_line=(0, 255, 0), color_pt=(0, 255, 100)):
    """Dibuja solo la parte superior del esqueleto. points_xy (8,2) en píxeles."""
    if points_xy is None or len(points_xy) == 0:
        return
    pts = np.asarray(points_xy, dtype=float)
    if confs is None:
        confs = np.ones(len(pts))
    r = 5
    for start, end in CONNECTIONS:
        if start < len(pts) and end < len(pts):
            c1, c2 = confs[start], confs[end]
            if c1 >= MIN_CONF and c2 >= MIN_CONF:
                x1, y1 = int(pts[start][0]), int(pts[start][1])
                x2, y2 = int(pts[end][0]), int(pts[end][1])
                draw.line([(x1, y1), (x2, y2)], fill=color_line, width=3)
    for i, pt in enumerate(pts):
        if confs[i] >= MIN_CONF:
            x, y = int(pt[0]), int(pt[1])
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color_pt, outline=color_pt)


def frame_to_pil(bgr_frame):
    """Convierte frame BGR (numpy) a PIL Image RGB."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_npy_poses(npy_path: str):
    """
    Carga poses desde .npy. Devuelve [T, 8, 2] con coords normalizadas 0-1.
    Soporta: [T, 8, 2] (pose_extractor), [T, 2, 8, 2] (merged_demo, usa user 0), [T, 17, 2].
    """
    data = np.load(npy_path)
    if data.ndim == 3:
        T, J, C = data.shape
        arr = data[:, :, :2]
        if J == 8:
            return arr
        if J >= 13:
            return arr[:, np.array(UPPER_KPS), :]
        return arr
    if data.ndim == 4 and data.shape[1] in (1, 2):
        return data[:, 0, :, :2]
    return data


def run_app(video_path: str, model_path: str = "yolo11n-pose.pt", npy_path: str | None = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el vídeo: {video_path}")
        return

    npy_poses = None
    if npy_path:
        npy_poses = load_npy_poses(npy_path)
        print(f"Poses cargadas desde {npy_path}: shape={npy_poses.shape}")

    model = load_pose_model(model_path) if npy_poses is None else None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = max(1, int(1000 / fps))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    npy_frames = len(npy_poses) if npy_poses is not None else 0

    root = tk.Tk()
    title = "Vídeo + Pose (npy)" if npy_poses is not None else "Vídeo + Pose"
    root.title(title)
    root.resizable(True, True)

    show_video = tk.BooleanVar(value=True)

    label = tk.Label(root)
    label.pack(padx=4, pady=4)

    def toggle():
        show_video.set(not show_video.get())
        # El botón indica qué verás al pulsar: "solo esqueleto" o "vídeo + esqueleto"
        btn.config(text="Ver vídeo + esqueleto" if show_video.get() else "Ver solo esqueleto")

    btn = tk.Button(root, text="Ver solo esqueleto", command=toggle)
    btn.pack(pady=4)

    status = tk.Label(root, text="", font=("", 9))
    status.pack(pady=2)

    current_frame = [0]
    paused = [False]

    def on_key(event):
        if event.keysym == "space":
            paused[0] = not paused[0]
        elif event.keysym == "Escape" or event.char == "q":
            root.quit()
            root.destroy()

    root.bind("<KeyPress>", on_key)

    def update():
        if not cap.isOpened():
            return
        if paused[0]:
            root.after(50, update)
            return

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame[0] = 0
            root.after(delay_ms, update)
            return

        h_orig, w_orig = frame.shape[:2]

        if npy_poses is not None:
            npy_len = len(npy_poses)
            if npy_len > 0:
                # Mapear frame de vídeo a frame de npy (pueden tener distinta longitud)
                idx = current_frame[0] % max(1, video_frames or 1)
                if video_frames and npy_len != video_frames:
                    j = int(round(idx * (npy_len - 1) / max(1, video_frames - 1)))
                else:
                    j = min(idx, npy_len - 1)
                j = max(0, min(j, npy_len - 1))
                pts_norm = npy_poses[j]
                points_xy = pts_norm.copy()
                points_xy[:, 0] = pts_norm[:, 0] * w_orig
                points_xy[:, 1] = pts_norm[:, 1] * h_orig
                confs = np.ones(8)
            else:
                points_xy, confs = None, np.zeros(8)
        else:
            # Escalar a 1080p para YOLO (igual que pose_extractor) y así comparar con .npy
            h_orig, w_orig = frame.shape[:2]
            if YOLO_SCALE_HEIGHT and h_orig != YOLO_SCALE_HEIGHT:
                scale = YOLO_SCALE_HEIGHT / h_orig
                w_1080 = int(round(w_orig * scale))
                h_1080 = YOLO_SCALE_HEIGHT
                frame_yolo = cv2.resize(frame, (w_1080, h_1080), interpolation=cv2.INTER_LINEAR)
            else:
                frame_yolo = frame
                w_1080, h_1080 = w_orig, h_orig
            results = model(frame_yolo, verbose=False)
            points_xy, confs = get_upper_pose(results, frame_yolo.shape)
            if confs is None:
                confs = np.zeros(8)
            elif points_xy is not None and (w_1080 != w_orig or h_1080 != h_orig):
                # Escalar keypoints de vuelta al frame original para dibujar
                points_xy[:, 0] = points_xy[:, 0] * (w_orig / w_1080)
                points_xy[:, 1] = points_xy[:, 1] * (h_orig / h_1080)

        if show_video.get():
            img = frame_to_pil(frame)
        else:
            img = Image.new("RGB", (w_orig, h_orig), (0, 0, 0))

        draw = ImageDraw.Draw(img)
        if points_xy is not None:
            draw_upper_skeleton(draw, points_xy, confs, w_orig, h_orig)

        img = img.resize((DISPLAY_W, DISPLAY_H), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo

        current_frame[0] += 1
        src = "npy" if npy_poses is not None else "yolo"
        status.config(text=f"Frame {current_frame[0]} | {src} | {'vídeo+esqueleto' if show_video.get() else 'solo esqueleto'} | Esp=pausa Q=salir")
        root.after(delay_ms, update)

    root.after(0, update)
    root.mainloop()
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Visualiza vídeo MP4 con pose (parte superior) en Tkinter")
    parser.add_argument("video", type=str, help="Ruta al archivo MP4")
    parser.add_argument("npy", type=str, nargs="?", default=None, help="Opcional: fichero .npy con poses precalculadas (del mismo clip)")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="Modelo YOLO pose si no se usa .npy")
    args = parser.parse_args()

    npy_path = args.npy
    if npy_path and not os.path.exists(npy_path):
        print(f"Fichero .npy no encontrado: {npy_path}")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"Archivo no encontrado: {args.video}")
        sys.exit(1)

    run_app(args.video, model_path=args.model, npy_path=npy_path)


if __name__ == "__main__":
    main()
