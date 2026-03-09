#!/usr/bin/env python3
"""
split_videos.py - Detecta secuencias ArUco inicio→fin y extrae clips.

Flujo básico por vídeo:
  - Usuario muestra ArUco inicio (ej. 1) → lo guarda (desaparece) → hace acción.
  - Más tarde muestra ArUco fin (ej. 42) → desaparece.

Clip extraído: desde (último frame ArUco inicio + 3 s) hasta (último frame ArUco fin - 5 s).

Modos de uso:
  1) Modo simple (un solo vídeo):
       python split_videos.py VIDEO.mp4 -o /path/output --inicio 1 --fin 42

  2) Preview (no genera ficheros, solo visualización):
       python split_videos.py VIDEO.mp4 --preview --inicio 1 --fin 42

  3) Modo configuración JSON (varios vídeos + rangos de búsqueda):
       python split_videos.py --config config_split.json

     Estructura esperada del JSON:
       {
         "output_dir": "/ruta/salida/general",
         "clips": [
           {
             "video": "/ruta/completa/video1.mp4",
             "search_start": 0.0,          # segundos (opcional)
             "search_end": null,           # segundos (opcional, null = hasta el final)
             "aruco_inicio": 1,
             "aruco_fin": 42
           }
         ]
       }

En modo JSON todos los clips se guardan en la misma carpeta output_dir como:
  clip1.mp4, clip2.mp4, ...
y se genera además un CSV clips_robos2.csv con columnas:
  video,inicio,fin,clasificacion
"""
import os
# Evitar avisos Qt/Wayland: forzar X11 (xcb)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    """Convierte segundos (float) a HH:MM:SS."""
    if sec is None or sec < 0:
        return "00:00:00"
    h = int(sec) // 3600
    m = (int(sec) % 3600) // 60
    s = int(sec) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _hhmmss_to_sec(value) -> Optional[float]:
    """
    Convierte "HH:MM:SS" (o segundos numéricos) a float en segundos.
    Devuelve None si value es None.
    """
    if value is None:
        return None
    # Si ya es número, lo devolvemos como float
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"Valor de tiempo no válido (esperado HH:MM:SS o segundos): {value!r}")
    text = value.strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 3:
        # Intento de parsear como segundos sueltos en texto
        try:
            return float(text)
        except ValueError as e:
            raise ValueError(f"Formato de tiempo no válido (esperado HH:MM:SS): {text!r}") from e
    try:
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2])
    except ValueError as e:
        raise ValueError(f"Formato de tiempo no válido (esperado HH:MM:SS): {text!r}") from e
    if m < 0 or m >= 60 or s < 0 or s >= 60:
        raise ValueError(f"Minutos/segundos fuera de rango en tiempo: {text!r}")
    return float(h * 3600 + m * 60 + s)


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
    search_start: Optional[float] = None,
    search_end: Optional[float] = None,
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
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_sec_video = total_frames_video / vid_fps if total_frames_video else 0.0
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Rango de búsqueda en segundos
    start_sec = max(0.0, float(search_start)) if search_start is not None else 0.0
    end_sec = float(search_end) if search_end is not None else total_sec_video
    if end_sec <= 0.0 or end_sec > total_sec_video:
        end_sec = total_sec_video

    start_frame = int(start_sec * vid_fps)
    end_frame = int(end_sec * vid_fps) if end_sec > 0 else total_frames_video

    # Rango de frames efectivo para la barra de progreso
    if total_frames_video > 0:
        total_frames_range = max(
            0,
            min(end_frame, total_frames_video) - min(start_frame, total_frames_video),
        )
    else:
        total_frames_range = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    params = aruco.DetectorParameters()
    if hasattr(params, "adaptiveThreshWinSizeMin"):
        params.adaptiveThreshWinSizeMin = 3
    if hasattr(params, "adaptiveThreshWinSizeMax"):
        params.adaptiveThreshWinSizeMax = 23

    state = "buscar_inicio"  # buscar_inicio | inicio_visible | buscar_fin | fin_visible
    last_frame_inicio: int | None = None
    # Para el ArUco de fin queremos:
    #   - inicio del clip: primer frame SIN ArUco de inicio (tras su desaparición)
    #   - fin del clip: último frame ANTES del primer frame donde aparece el ArUco de fin
    first_frame_fin: int | None = None
    last_frame_fin: int | None = None
    frames_sin_inicio = 0
    frames_sin_fin = 0
    sequences: list[tuple[float, float]] = []

    pbar = tqdm(
        total=total_frames_range or total_frames_video,
        unit="fr",
        desc="Analizando vídeo",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    frame_idx = start_frame
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t_sec = frame_idx / vid_fps

            if frame_idx > end_frame:
                break

            # Progreso relativo al rango de búsqueda
            if total_frames_range > 0:
                pbar.n = min(frame_idx - start_frame, total_frames_range)
            else:
                pbar.n = min(frame_idx, total_frames_video)
            pbar.set_postfix_str(f"{_sec_to_hhmmss(t_sec)} / {_sec_to_hhmmss(end_sec)}")
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
                        # ArUco inicio desapareció → clip_start = primer frame sin el marcador
                        # Si el último frame con marcador es F, el primero sin marcador es F+1.
                        clip_start = (last_frame_inicio + 1) / vid_fps
                        state = "buscar_fin"
                        first_frame_fin = None
                        last_frame_fin = None
                        frames_sin_fin = 0

            elif state == "buscar_fin":
                if aruco_fin in ids:
                    print(f"ArUco con ID {aruco_fin} encontrado en {_sec_to_hhmmss(t_sec)}")
                    state = "fin_visible"
                    # Guardamos el PRIMER frame donde aparece el ArUco de fin
                    first_frame_fin = frame_idx
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
                        # ArUco fin ha desaparecido.
                        # El primer frame donde apareció fue first_frame_fin (N),
                        # así que el último frame SIN el marcador es N-1.
                        if first_frame_fin is not None:
                            t_end = max(0.0, (first_frame_fin - 1) / vid_fps)
                        else:
                            # Fallback muy raro: no se registró first_frame_fin, usar último frame visto.
                            t_end = max(0.0, last_frame_fin / vid_fps)

                        # Aseguramos que el clip tenga al menos 1 frame de duración
                        min_duration = 1.0 / vid_fps
                        clip_end = max(clip_start + min_duration, t_end)
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
        if total_frames_range > 0:
            pbar.n = total_frames_range
            pbar.refresh()
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


@dataclass
class ConfigClip:
    video: Path
    search_start: Optional[float]
    search_end: Optional[float]
    aruco_inicio: int
    aruco_fin: int


def _load_config(config_path: Path) -> tuple[Path, list[ConfigClip]]:
    """Carga el JSON de configuración para el modo batch."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir_raw = data.get("output_dir", "")
    if not output_dir_raw:
        raise ValueError("El JSON debe contener la clave 'output_dir' con la carpeta de salida.")
    output_dir = Path(output_dir_raw).expanduser().resolve()

    clips_raw = data.get("clips", [])
    if not isinstance(clips_raw, list) or not clips_raw:
        raise ValueError("El JSON debe contener una lista 'clips' con al menos un objeto.")

    clips_cfg: list[ConfigClip] = []
    for i, item in enumerate(clips_raw, start=1):
        if "video" not in item:
            raise ValueError(f"Clip {i}: falta la clave 'video'.")
        video_path = Path(item["video"]).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Clip {i}: vídeo no encontrado: {video_path}")

        # search_start / search_end pueden venir como segundos o como "HH:MM:SS"
        raw_start = item.get("search_start")
        raw_end = item.get("search_end")
        search_start = _hhmmss_to_sec(raw_start)
        search_end = _hhmmss_to_sec(raw_end)

        clips_cfg.append(
            ConfigClip(
                video=video_path,
                search_start=search_start,
                search_end=search_end,
                aruco_inicio=int(item.get("aruco_inicio", 1)),
                aruco_fin=int(item.get("aruco_fin", 42)),
            )
        )

    return output_dir, clips_cfg


def run_from_config(config_path: Path, debug_max_clips: Optional[int] = None) -> None:
    """
    Ejecuta el procesamiento en modo configuración JSON.

    - Recorre todos los clips del JSON.
    - Genera clips numerados: clip1.mp4, clip2.mp4, ...
    - Genera CSV clips_robos2.csv con columnas:
        video,inicio,fin,clasificacion
      donde:
        video         = nombre del clip (p.ej. clip1.mp4)
        inicio        = 00:00:00
        fin           = duración del clip en HH:MM:SS
        clasificacion = 6
    """
    output_dir, clips_cfg = _load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "clips_robos2.csv"
    csv_file = open(csv_path, "w", encoding="utf-8", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["video", "inicio", "fin", "clasificacion"])

    clip_counter = 1
    try:
        for cfg in clips_cfg:
            print(f"\n=== Procesando vídeo: {cfg.video} ===")
            txt_start = _sec_to_hhmmss(cfg.search_start) if cfg.search_start is not None else "00:00:00"
            txt_end = _sec_to_hhmmss(cfg.search_end) if cfg.search_end is not None else "fin"
            print(f"  Rango búsqueda: {txt_start} → {txt_end}")
            print(f"  ArUco inicio={cfg.aruco_inicio}, fin={cfg.aruco_fin}")

            sequences = find_aruco_sequences(
                cfg.video,
                cfg.aruco_inicio,
                cfg.aruco_fin,
                search_start=cfg.search_start,
                search_end=cfg.search_end,
            )
            print(f"  Secuencias encontradas: {len(sequences)}")

            for (start_sec, end_sec) in sequences:
                duration = max(0.0, end_sec - start_sec)
                clip_name = f"clip{clip_counter}.mp4"
                clip_path = output_dir / clip_name

                ok = cut_clip(cfg.video, start_sec, end_sec, clip_path)
                status = "OK" if ok else "ERROR"
                print(
                    f"    Clip {clip_counter}: {start_sec:.1f}s - {end_sec:.1f}s "
                    f"({duration:.1f}s) → {clip_name} [{status}]"
                )

                if ok:
                    writer.writerow(
                        [
                            clip_name,
                            "00:00:00",
                            _sec_to_hhmmss(duration),
                            "6",
                        ]
                    )
                    # Forzar volcado al disco para poder ver el CSV crecer en tiempo real
                    csv_file.flush()
                    clip_counter += 1

                    # Modo debug: parar tras N clips totales
                    if debug_max_clips is not None and clip_counter > debug_max_clips:
                        print(f"\n[DEBUG] Límite de clips alcanzado ({debug_max_clips}). Deteniendo procesamiento.")
                        return
    finally:
        csv_file.close()
        print(f"\nCSV generado: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detecta secuencias ArUco inicio→fin en un vídeo y extrae clips."
    )
    parser.add_argument("video", type=str, nargs="?", help="Ruta del vídeo de entrada (modo simple)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Ruta de salida (requerido en modo simple si no se usa --preview)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Ruta a JSON de configuración para procesar múltiples vídeos (modo batch).",
    )
    parser.add_argument("--preview", action="store_true", help="Solo mostrar vídeo con detección de ArUcos (bbox + ID)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Modo debug: en modo JSON sólo genera los primeros 3 clips (para revisar el CSV).",
    )
    parser.add_argument("--inicio", type=int, default=1, help="ID ArUco de inicio (default: 1)")
    parser.add_argument("--fin", type=int, default=42, help="ID ArUco de fin (default: 42)")
    args = parser.parse_args()

    # Modo batch con JSON de configuración
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"JSON de configuración no encontrado: {config_path}")
        debug_max = 3 if args.debug else None
        run_from_config(config_path, debug_max_clips=debug_max)
        return

    # Modo simple (un solo vídeo)
    if not args.video:
        parser.error("Debes proporcionar un vídeo (modo simple) o bien usar --config (modo batch).")

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo no encontrado: {video_path}")

    if args.preview:
        run_preview(video_path, args.inicio, args.fin)
        return

    if not args.output:
        parser.error("-o/--output es requerido en modo simple cuando no se usa --preview")

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
