#!/usr/bin/env python3
"""
split_videos.py - Detecta secuencias ArUco inicio→fin y extrae clips.

Flujo: usuario muestra ArUco inicio (ej. 1) → lo guarda (desaparece) → hace acción
       → muestra ArUco fin (ej. 42) → desaparece.

Clip extraído: desde (último frame ArUco inicio + 3 s) hasta (último frame ArUco fin - 5 s).

Uso:
  python split_videos.py VIDEO.mp4 -o /path/output
  python split_videos.py VIDEO.mp4 -o /path/output --inicio 1 --fin 42
  python split_videos.py VIDEO.mp4 --preview   # Solo visualizar detección (1080p, bbox + ID)

Salida: output/robos_split_YYYYMMDD_HHMMSS/clip_001.mp4, clip_002.mp4, ...
"""
import os
# Evitar avisos Qt/Wayland: forzar X11 (xcb)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from cv2 import aruco
except ImportError:
    raise ImportError(
        "Se necesita opencv-contrib-python para ArUco. Instala con: pip install opencv-contrib-python"
    )

# Constantes
ARUCO_DICT = aruco.DICT_6X6_250
FRAMES_SIN_MARCADOR = 8  # Frames consecutivos sin ver el ArUco para considerarlo "desaparecido"
OFFSET_INICIO_SEC = 3.0  # Segundos después de que desaparece el ArUco de inicio
OFFSET_FIN_SEC = 5.0     # Segundos antes de que desaparezca el ArUco de fin (retroceder)
# Al buscar ArUco: saltar N frames para acelerar (0 = procesar todos). En visible: siempre frame a frame.
OFFSET_FRAMES = 5


def _sec_to_hhmmss(sec: float) -> str:
    """Convierte segundos a HH:MM:SS."""
    h = int(sec) // 3600
    m = (int(sec) % 3600) // 60
    s = int(sec) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


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
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids if ids is not None else np.array([]).reshape(0, 1)


def _detect_aruco_ids(frame, aruco_dict, params) -> set[int]:
    """Devuelve los IDs de ArUco detectados en el frame."""
    _, ids = _detect_aruco(frame, aruco_dict, params)
    if ids.size == 0:
        return set()
    return set(int(x) for x in ids.flatten())


def find_aruco_sequences(
    video_path: str | Path,
    aruco_inicio: int,
    aruco_fin: int,
    fps: float | None = None,
) -> list[tuple[float, float]]:
    """
    Escanea el vídeo y devuelve lista de (start_sec, end_sec) para cada secuencia
    ArUco inicio → acción → ArUco fin.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el vídeo: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or (fps or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_sec = total_frames / vid_fps if total_frames else 0
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    params = aruco.DetectorParameters()
    if hasattr(params, "adaptiveThreshWinSizeMin"):
        params.adaptiveThreshWinSizeMin = 3
    if hasattr(params, "adaptiveThreshWinSizeMax"):
        params.adaptiveThreshWinSizeMax = 23

    state = "buscar_inicio"  # buscar_inicio | inicio_visible | buscar_fin | fin_visible
    last_frame_inicio: int | None = None
    last_frame_fin: int | None = None
    frames_sin_inicio = 0
    frames_sin_fin = 0
    sequences: list[tuple[float, float]] = []

    pbar = tqdm(
        total=total_frames,
        unit="fr",
        desc="Analizando vídeo",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t_sec = frame_idx / vid_fps
            pbar.n = min(frame_idx, total_frames)
            pbar.set_postfix_str(f"{_sec_to_hhmmss(t_sec)} / {_sec_to_hhmmss(total_sec)}")
            pbar.refresh()

            ids = _detect_aruco_ids(frame, aruco_dict, params)

            if state == "buscar_inicio":
                if aruco_inicio in ids:
                    print(f"ArUco con ID {aruco_inicio} encontrado en {_sec_to_hhmmss(t_sec)}")
                    state = "inicio_visible"
                    last_frame_inicio = frame_idx
                    frames_sin_inicio = 0
                # else: sigue buscando

            elif state == "inicio_visible":
                if aruco_inicio in ids:
                    last_frame_inicio = frame_idx
                    frames_sin_inicio = 0
                else:
                    frames_sin_inicio += 1
                    if frames_sin_inicio >= FRAMES_SIN_MARCADOR:
                        # ArUco inicio desapareció → clip_start = último frame + 3 s
                        t_inicio_last = last_frame_inicio / vid_fps
                        clip_start = t_inicio_last + OFFSET_INICIO_SEC
                        state = "buscar_fin"
                        last_frame_fin = None
                        frames_sin_fin = 0

            elif state == "buscar_fin":
                if aruco_fin in ids:
                    print(f"ArUco con ID {aruco_fin} encontrado en {_sec_to_hhmmss(t_sec)}")
                    state = "fin_visible"
                    last_frame_fin = frame_idx
                    frames_sin_fin = 0
                # else: sigue buscando

            elif state == "fin_visible":
                if aruco_fin in ids:
                    last_frame_fin = frame_idx
                    frames_sin_fin = 0
                else:
                    frames_sin_fin += 1
                    if frames_sin_fin >= FRAMES_SIN_MARCADOR and last_frame_fin is not None:
                        # ArUco fin desapareció → clip_end = último frame - 5 s
                        t_fin_last = last_frame_fin / vid_fps
                        clip_end = max(clip_start + 1.0, t_fin_last - OFFSET_FIN_SEC)
                        if clip_end > clip_start:
                            sequences.append((clip_start, clip_end))
                        state = "buscar_inicio"
                        last_frame_inicio = None
                        frames_sin_inicio = 0

            # Avanzar: en buscar_* saltar OFFSET_FRAMES si > 0; en visible frame a frame
            if state in ("buscar_inicio", "buscar_fin") and OFFSET_FRAMES > 0:
                frame_idx += OFFSET_FRAMES
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            else:
                frame_idx += 1
    finally:
        pbar.n = total_frames
        pbar.close()
        cap.release()

    return sequences


PREVIEW_HEIGHT = 1080


def run_preview(video_path: Path, aruco_inicio: int, aruco_fin: int) -> None:
    """
    Muestra el vídeo redimensionado a 1080p con bounding boxes e IDs de cualquier
    ArUco detectado (DICT_6X6_250).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el vídeo: {video_path}")

    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    params = aruco.DetectorParameters()
    if hasattr(params, "adaptiveThreshWinSizeMin"):
        params.adaptiveThreshWinSizeMin = 3
    if hasattr(params, "adaptiveThreshWinSizeMax"):
        params.adaptiveThreshWinSizeMax = 23

    print(f"Preview: {video_path}")
    print(f"Buscando ArUco inicio={aruco_inicio}, fin={aruco_fin}. Mostrando todos los detectados.")
    print("Q = salir, Espacio = pausa")
    print()

    paused = False
    frame = None
    ids_prev: set[int] = set()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
        if frame is None:
            break

        # Redimensionar a 1080p de altura
        h, w = frame.shape[:2]
        scale = PREVIEW_HEIGHT / h if h > PREVIEW_HEIGHT else 1.0
        if scale != 1.0:
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, PREVIEW_HEIGHT), interpolation=cv2.INTER_LINEAR)

        corners, ids = _detect_aruco(frame, aruco_dict, params)
        ids_act = set(int(x) for x in ids.flatten()) if ids.size > 0 else set()

        # Mostrar en terminal cuando se detecta ArUco inicio/fin (solo al aparecer)
        for mid in (aruco_inicio, aruco_fin):
            if mid in ids_act and mid not in ids_prev:
                t_sec = (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps
                print(f"ArUco con ID {mid} encontrado en {_sec_to_hhmmss(t_sec)}")
        ids_prev = ids_act

        # Dibujar bounding box e ID para cada ArUco detectado
        if ids.size > 0:
            ids_flat = ids.flatten()
            for i, (pts, marker_id) in enumerate(zip(corners, ids_flat)):
                pts = pts.astype(np.int32)
                # Bounding box
                color = (0, 255, 0) if marker_id in (aruco_inicio, aruco_fin) else (0, 165, 255)
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                # Centro para el texto
                cx = int(pts[0].mean(axis=0)[0])
                cy = int(pts[0].mean(axis=0)[1])
                label = f"ID:{int(marker_id)}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (cx - tw//2 - 4, cy - th - 8), (cx + tw//2 + 4, cy + 4), color, -1)
                cv2.putText(frame, label, (cx - tw//2, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Info en pantalla
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        t_sec = frame_idx / fps
        cv2.putText(frame, f"t={t_sec:.1f}s | Detectados: {list(ids.flatten()) if ids.size else []}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"t={t_sec:.1f}s | Detectados: {list(ids.flatten()) if ids.size else []}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.imshow("ArUco Preview - Q=salir Espacio=pausa", frame)
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


def cut_clip(video_in: Path, start_sec: float, end_sec: float, video_out: Path) -> bool:
    """Recorta el vídeo con FFmpeg. Stream copy: misma resolución y codecs, sin recodificar."""
    cmd = [
        "ffmpeg", "-y", "-threads", "0",
        "-ss", str(start_sec), "-to", str(end_sec),
        "-i", str(video_in),
        "-c", "copy",  # copy = sin recodificar, mantiene resolución y codecs originales
        "-loglevel", "error", str(video_out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and r.stderr:
        print(f"  [FFmpeg] {r.stderr.strip()[:200]}")
    return r.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Detecta secuencias ArUco inicio→fin en un vídeo y extrae clips."
    )
    parser.add_argument("video", type=str, help="Ruta del vídeo de entrada")
    parser.add_argument("-o", "--output", type=str, default=None, help="Ruta de salida (requerido si no --preview)")
    parser.add_argument("--preview", action="store_true", help="Solo mostrar vídeo con detección de ArUcos (bbox + ID)")
    parser.add_argument("--inicio", type=int, default=1, help="ID ArUco de inicio (default: 1)")
    parser.add_argument("--fin", type=int, default=42, help="ID ArUco de fin (default: 42)")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo no encontrado: {video_path}")

    if args.preview:
        run_preview(video_path, args.inicio, args.fin)
        return

    if not args.output:
        parser.error("-o/--output es requerido cuando no se usa --preview")
    output_base = Path(args.output).resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_base / f"robos_split_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analizando: {video_path}")
    print(f"ArUco inicio: {args.inicio}, ArUco fin: {args.fin}")
    print(f"Salida: {out_dir}\n")

    sequences = find_aruco_sequences(video_path, args.inicio, args.fin)
    print(f"Secuencias encontradas: {len(sequences)}")

    for i, (start_sec, end_sec) in enumerate(sequences, 1):
        clip_path = out_dir / f"clip_{i:03d}.mp4"
        ok = cut_clip(video_path, start_sec, end_sec, clip_path)
        status = "OK" if ok else "ERROR"
        print(f"  Clip {i}: {start_sec:.1f}s - {end_sec:.1f}s → {clip_path.name} [{status}]")

    print(f"\nListo. Clips en: {out_dir}")


if __name__ == "__main__":
    main()
