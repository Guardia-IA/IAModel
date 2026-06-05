"""
Genera un .mp4 con SOLO el esqueleto (parte superior del cuerpo) sobre fondo negro,
a partir de un vídeo original. Extrae la pose con YOLO frame a frame (o usa un .npy
precalculado) y escribe el resultado a fichero (sin ventana, apto para servidor headless).

Uso:
    python export_skeleton_video.py video.mp4
    python export_skeleton_video.py video.mp4 -o salida_esqueleto.mp4
    python export_skeleton_video.py video.mp4 --model yolo11n-pose.pt
    python export_skeleton_video.py video.mp4 --npy user_0/poses.npy   # usa poses ya extraídas
    python export_skeleton_video.py video.mp4 --over-video             # esqueleto encima del vídeo

Notas:
  - Reutiliza la definición de keypoints/conexiones y el dibujo del visor visualize_video_pose.py.
  - El .npy debe corresponder al MISMO clip que el vídeo (coords normalizadas 0-1).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw

from visualize_video_pose import (
    UPPER_KPS,
    MIN_CONF,
    load_pose_model,
    get_upper_pose,
    draw_upper_skeleton,
    load_npy_poses,
)


def resolve_device(choice: str) -> str:
    """Devuelve el device para YOLO ('cuda:0' o 'cpu') según la elección y la disponibilidad."""
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False

    if choice in ("gpu", "cuda"):
        if not has_cuda:
            print("[AVISO] Se pidió GPU pero no hay CUDA disponible; se usa CPU.")
            return "cpu"
        return "cuda:0"
    if choice == "cpu":
        return "cpu"
    # auto
    return "cuda:0" if has_cuda else "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Exporta un .mp4 con solo el esqueleto (sobre fondo negro) a partir de un vídeo."
    )
    parser.add_argument("video", type=str, help="Ruta al vídeo original (.mp4)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Ruta del .mp4 de salida (def: <video>_esqueleto.mp4)")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt",
                        help="Modelo YOLO pose si no se usa --npy (def: yolo11n-pose.pt)")
    parser.add_argument("--npy", type=str, default=None,
                        help="Opcional: .npy con poses precalculadas (normalizadas 0-1) del mismo clip")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de entrada YOLO (def: 640)")
    parser.add_argument("--over-video", action="store_true",
                        help="Dibuja el esqueleto sobre el vídeo original en vez de fondo negro")
    parser.add_argument("--device", choices=["auto", "gpu", "cuda", "cpu"], default="auto",
                        help="Dispositivo YOLO: auto (GPU si hay, si no CPU), gpu/cuda, cpu (def: auto)")
    args = parser.parse_args()

    device = resolve_device(args.device)

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        print(f"Vídeo no encontrado: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"No se pudo abrir el vídeo: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = Path(args.output).expanduser().resolve() if args.output \
        else video_path.with_name(f"{video_path.stem}_esqueleto.mp4")

    npy_poses = None
    if args.npy:
        npy_poses = load_npy_poses(str(args.npy))
        print(f"Poses cargadas desde {args.npy}: shape={npy_poses.shape}")

    model = None
    if npy_poses is None:
        model = load_pose_model(args.model)
        try:
            model.to(device)
        except Exception as e:
            print(f"[AVISO] No se pudo mover el modelo a {device}: {e}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print("No se pudo crear el VideoWriter (códec mp4v no disponible).")
        sys.exit(1)

    print(f"Vídeo: {w}x{h} @ {fps:.1f} fps | frames: {total or '?'}")
    print(f"Modo: {'esqueleto sobre vídeo' if args.over_video else 'solo esqueleto (fondo negro)'}")
    fuente = "npy precalculado" if npy_poses is not None else f"YOLO ({args.model}) en {device}"
    print(f"Fuente de pose: {fuente}")
    print(f"Salida: {out_path}")

    idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.over_video:
            base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            base = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(base)

        if npy_poses is not None:
            if idx < len(npy_poses):
                pts_norm = np.asarray(npy_poses[idx], dtype=float)  # (8,2) en 0-1
                pts_px = pts_norm.copy()
                pts_px[:, 0] *= w
                pts_px[:, 1] *= h
                # keypoints ausentes (0,0) -> conf 0 para no dibujarlos
                confs = np.where((pts_norm[:, 0] == 0) & (pts_norm[:, 1] == 0), 0.0, 1.0)
                draw_upper_skeleton(draw, pts_px, confs, w, h)
        else:
            results = model.predict(frame, imgsz=args.imgsz, device=device, verbose=False)
            pts, confs = get_upper_pose(results, frame.shape, choose_biggest=True)
            if pts is not None:
                draw_upper_skeleton(draw, pts, confs, w, h)

        out_frame = cv2.cvtColor(np.asarray(base), cv2.COLOR_RGB2BGR)
        writer.write(out_frame)
        written += 1
        idx += 1
        if total and idx % 50 == 0:
            print(f"  procesados {idx}/{total} frames...")

    cap.release()
    writer.release()
    print(f"Listo. {written} frames escritos en: {out_path}")


if __name__ == "__main__":
    main()
