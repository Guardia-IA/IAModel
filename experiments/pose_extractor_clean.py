import argparse
import os
import re
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
import subprocess
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIGURACIÓN DE RUTAS (edita config.py) ---
from config import (
    get_experiments, get_path_roots, PATH_ROOT, CSV_PATH, OUTPUT_BASE, LOGS_SUBDIR,
    CLIP_SCALE_HEIGHT, VAAPI_DEVICE, YOLO_POSE_MODEL,
)
from security import validate_csv_files, validate_folder

# Base de salida: OUTPUT_BASE de config (temp_clips/ y data_result/ dentro)
OUTPUT = Path(OUTPUT_BASE) if OUTPUT_BASE else Path(__file__).parent / "output"

# Config opcional de límites por categoría (máx. clips a procesar por clase).
# Se busca en el directorio actual desde donde se ejecuta el script.
POSE_EXTRACTION_CONFIG = "config_pose_extraction.json"


def _load_category_limits() -> dict[str, int]:
    """
    Lee config_pose_extraction.json si existe y devuelve un dict {cat_str: max_clips}.
    Valores:
      - null/None  -> sin límite (se procesan todos)
      - 0          -> no se procesa ninguno
      - >0         -> máximo N clips para esa categoría
    """ 
    limits: dict[str, int] = {}
    cfg_path = Path(POSE_EXTRACTION_CONFIG)
    if not cfg_path.exists():
        return limits
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return limits
    cat_cfg = data.get("category_limits", {})
    if isinstance(cat_cfg, dict):
        for k, v in cat_cfg.items():
            try:
                cat_str = str(int(k))
            except (TypeError, ValueError):
                continue
            if v is None:
                continue
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if n >= 0:
                limits[cat_str] = n
    return limits

# --- PARÁMETROS DE CONTROL ---
DEBUG_MODE = False      # True: Solo procesa N vídeos
N_DEBUG = 5            # Número de vídeos en modo debug
MODEL_PATH = YOLO_POSE_MODEL     # Definido en config.py
# Tracker junto a este script (cwd puede ser training/ u otro → ruta relativa falla).
_CUSTOM_TRACKER_YAML = Path(__file__).resolve().parent / "custom_tracker.yaml"
DELETE_TEMP_VIDEOS = True       # Si True, borra los vídeos temporales al terminar
# Si True, guarda una copia del clip procesado en data_result/{cat}/{clip_name}/clip.mp4
# para poder visualizar poses con el vídeo exacto (mismo nº de frames que poses_full.npy)
SAVE_PROCESSED_CLIP = True

# --- FILTROS DE CALIDAD ---
# Umbral de confianza: keypoints con conf <= MIN_KP_CONF se guardan como NaN (no se infieren).
MIN_KP_CONF = 0.5
RELIABILITY_THR = 0.9                  # Ratio mínimo de frames válidos para guardar (90%)
KEEP_KPS = [5, 6, 7, 8, 9, 10, 11, 12]  # Hombros, codos, muñecas, cadera (sin piernas)
CRITICAL_KPS = [7, 8, 9, 10]           # Muñecas y codos (estadística de oclusión)

# --- FILTRO POR COBERTURA EN TIEMPO ---
MIN_COVERAGE_RATIO = 0.8               # Mínimo % de la duración con buena calidad (80%)
DEFAULT_FPS = 12                       # FPS por defecto si no se puede leer del vídeo

# Filtro de duración mínima por usuario (en segundos). 0 = desactivado.
MIN_USER_SECONDS = 0.0

# Solo procesar clips con exactamente un track/persona detectada; si hay más, se omite la fila CSV.
SINGLE_USER_ONLY = True

# Filtro de visibilidad corporal: no guardar usuarios que solo muestren mano, cabeza, etc.
# Un frame cuenta como "cuerpo visible" si al menos BODY_VISIBLE_MIN_KPS keypoints están
# por encima de MIN_KP_CONF (torso, brazos, caderas). No se guarda el usuario si no cumple:
BODY_VISIBLE_MIN_KPS = 5               # Mín. keypoints visibles por frame (de 8: hombros, codos, muñecas, cadera)
BODY_VISIBLE_MIN_FRAMES = 5            # Mín. frames con cuerpo visible
BODY_VISIBLE_MIN_RATIO = 0.2           # Mín. ratio de frames con cuerpo visible (20%)

# Umbral de confianza para considerar un punto "ocluso"
OCCLUSION_CONF_THR = 0.3              # keypoints con conf < esto se consideran ocluidos

# Pose sentinela para mantener 1:1 frame–pose cuando no hay detección (evita desfase vídeo/poses)
NAN_POSE = np.full((len(KEEP_KPS), 2), np.nan, dtype=np.float32)
EMPTY_KP_MASK = np.zeros(len(KEEP_KPS), dtype=bool)


def _build_masked_pose(kpts_row: np.ndarray, confs_row: np.ndarray) -> np.ndarray:
    """[8, 2] con NaN en keypoints con conf <= MIN_KP_CONF (no guardar inferencias dudosas)."""
    out = kpts_row[np.array(KEEP_KPS, dtype=int)].astype(np.float32, copy=True)
    for j, coco_idx in enumerate(KEEP_KPS):
        if float(confs_row[coco_idx]) <= MIN_KP_CONF:
            out[j] = np.nan
    return out


def _keypoint_mask_from_pose(kpt_pose: np.ndarray) -> np.ndarray:
    """True por keypoint visible (sin NaN en x/y). Forma [8]."""
    return ~np.isnan(kpt_pose).any(axis=1)


def _visible_kp_count(kpt_pose: np.ndarray) -> int:
    return int(_keypoint_mask_from_pose(kpt_pose).sum())


def _frame_body_visible(kpt_pose: np.ndarray) -> bool:
    return _visible_kp_count(kpt_pose) >= BODY_VISIBLE_MIN_KPS


def _frame_usable(kpt_pose: np.ndarray) -> bool:
    """Frame útil: persona detectada con suficientes keypoints visibles (p. ej. robo de perfil)."""
    if np.isnan(kpt_pose).all():
        return False
    return _frame_body_visible(kpt_pose)


def _save_user_pose_files(user_dir: Path, info: dict) -> None:
    """Guarda poses_full, poses, keypoint_mask y valid_mask."""
    poses_full_arr = np.array(info["poses_full"], dtype=np.float32)
    np.save(str(user_dir / "poses_full.npy"), poses_full_arr)
    np.save(str(user_dir / "poses.npy"), np.array(info["poses"], dtype=np.float32))
    keypoint_mask_arr = np.array(info["keypoint_masks"], dtype=bool)
    np.save(str(user_dir / "keypoint_mask.npy"), keypoint_mask_arr)
    visible_per_frame = np.sum(keypoint_mask_arr, axis=1)
    valid_mask = visible_per_frame >= BODY_VISIBLE_MIN_KPS
    np.save(str(user_dir / "valid_mask.npy"), valid_mask)


def _get_device() -> str:
    """Detecta GPU CUDA o usa CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _setup_logging(log_dir: str | Path | None = None):
    """
    Redirige stdout a terminal + fichero log<timestamp>.txt.
    Logs en OUTPUT/LOGS_SUBDIR (p. ej. OUTPUT_BASE/logs). Todo lo que se imprima
    después quedará en el log (útil para ejecutar por SSH y revisar luego).
    Devuelve (log_file, original_stdout, log_path) o (None, None, None) si falla.
    """
    base = (OUTPUT / LOGS_SUBDIR) if OUTPUT_BASE else Path(__file__).parent / "logs"
    log_dir = Path(log_dir or base).resolve()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None, None, None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"log{timestamp}.txt"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def _safe_flush(self, s):
            try:
                if hasattr(s, "flush"):
                    s.flush()
            except (ValueError, OSError):
                pass

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    self._safe_flush(s)
                except (ValueError, OSError):
                    pass

        def flush(self):
            for s in self.streams:
                self._safe_flush(s)

    try:
        log_file = open(log_path, "w", encoding="utf-8")
    except OSError:
        return None, None, None
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)
    return log_file, original_stdout, log_path


def _tracker_yaml_path() -> str:
    """Ruta absoluta al YAML de tracking, o bytetrack por defecto de Ultralytics."""
    if _CUSTOM_TRACKER_YAML.exists():
        return str(_CUSTOM_TRACKER_YAML)
    return "bytetrack.yaml"


def _resolve_model_path(base: str) -> str:
    """Busca .engine en engine/ y, si no existe, usa .pt."""
    stem = Path(base).stem
    engine_dir = Path(__file__).resolve().parent / "engine"
    engine_path = engine_dir / f"{stem}.engine"
    if engine_path.exists():
        return str(engine_path)
    return base


# Carga del modelo (se hace una sola vez)
DEVICE = _get_device()
_MODEL_RESOLVED = _resolve_model_path(MODEL_PATH)
print(f"Cargando modelo pose: {_MODEL_RESOLVED}")
model = YOLO(_MODEL_RESOLVED)  # Modelo pose para keypoints


# Patrón para detectar timestamps HH:MM:SS
HMS_PATTERN = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")


def is_hms_format(val) -> bool:
    """Comprueba si el valor tiene formato HH:MM:SS."""
    s = str(val).strip().strip('"').strip("'")
    return bool(HMS_PATTERN.match(s))


def find_start_row(csv_path: str) -> int:
    """
    Encuentra la primera fila donde la segunda columna tiene formato HH:MM:SS.
    Devuelve el número de fila (1-based). Si no encuentra, devuelve 1.
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            parts = line.strip().split(",")
            if len(parts) >= 2 and is_hms_format(parts[1]):
                return i
    return 1


def hms_to_seconds(t: str) -> float:
    """Convierte 'HH:MM:SS' en segundos."""
    h, m, s = map(int, str(t).split(":"))
    return h * 3600 + m * 60 + s


def seconds_to_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _normalize_hms(t: str) -> str:
    return str(t).strip().strip('"').strip("'")


def _is_full_clip_range(t_start: str, t_end: str) -> bool:
    """00:00:00 + 00:00:00 = usar el fichero de vídeo completo sin recortar."""
    return _normalize_hms(t_start) == "00:00:00" and _normalize_hms(t_end) == "00:00:00"


def _resolve_video_path(videos_dir: str, video_rel_path: str) -> str:
    p = Path(str(video_rel_path).strip().strip('"').strip("'"))
    if p.is_absolute():
        return str(p.resolve())
    return str((Path(videos_dir) / p).resolve())


def _video_duration_from_file(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps is None or fps <= 0:
            fps = DEFAULT_FPS
        if frames is None or frames <= 0:
            return 0.0
        return float(frames) / float(fps)
    finally:
        cap.release()


def _hms_to_compact(t: str) -> str:
    """Convierte 'HH:MM:SS' a 'HHMMSS' (sin dos puntos)."""
    return str(t).replace(":", "").strip()


def make_clip_name(
    video_rel_path: str,
    t_start: str,
    t_end: str,
    category: str,
    used_names: set,
    dir_rel_path: str | None = None,
) -> str:
    """
    Genera nombre único de clip: [dir_rel_path_]video_HHMMSS_HHMMSS_cat
    dir_rel_path = ruta del directorio (fecha, cámara) ej. 12Diciembre2025_cam1
    Si hay duplicados, añade _2, _3, etc.
    """
    p = Path(video_rel_path)
    path_parts = list(p.parent.parts) + [p.stem]
    path_norm = "_".join(path_parts) if path_parts else p.stem or "clip"
    path_norm = path_norm.replace(" ", "_").replace("/", "_")

    if dir_rel_path:
        dir_norm = str(dir_rel_path).replace(" ", "_").replace("/", "_").replace("\\", "_")
        if dir_norm and dir_norm != ".":
            path_norm = f"{dir_norm}_{path_norm}"

    t1 = _hms_to_compact(t_start)
    t2 = _hms_to_compact(t_end)
    if _is_full_clip_range(t_start, t_end):
        base = f"{path_norm}_full_{category}"
    else:
        base = f"{path_norm}_{t1}_{t2}_{category}"

    name = base
    n = 2
    while name in used_names:
        name = f"{base}_{n}"
        n += 1
    used_names.add(name)
    return name


def cut_clip(video_in, start, end, video_out) -> bool:
    """
    Recorta el vídeo con FFmpeg. Escala a CLIP_SCALE_HEIGHT si está configurado.
    Usa VAAPI cuando VAAPI_DEVICE está definido (Intel/AMD).
    """
    scale_h = CLIP_SCALE_HEIGHT
    vaapi = VAAPI_DEVICE

    if vaapi and scale_h:
        # Pipeline VAAPI: decode → scale_vaapi → encode h264_vaapi
        scale_expr = f"scale_vaapi=-2:{scale_h}"  # -2 = ancho automático (mantiene aspect)
        command = [
            'ffmpeg', '-y', '-threads', '0', '-hwaccel', 'vaapi',
            '-hwaccel_device', str(vaapi),
            '-hwaccel_output_format', 'vaapi',
            '-ss', str(start), '-to', str(end), '-i', video_in,
            '-vf', scale_expr,
            '-c:v', 'h264_vaapi',
            '-vaapi_device', str(vaapi),
            '-loglevel', 'error', video_out
        ]
    elif scale_h:
        # Software: scale + libx264
        scale_expr = f"scale=-2:{scale_h}"
        command = [
            'ffmpeg', '-y', '-threads', '0', '-ss', str(start), '-to', str(end),
            '-i', video_in, '-vf', scale_expr,
            '-c:v', 'libx264', '-loglevel', 'error', video_out
        ]
    else:
        # Sin escalado: copia directa (como antes)
        command = [
            'ffmpeg', '-y', '-threads', '0', '-ss', str(start), '-to', str(end),
            '-i', video_in, '-c', 'copy', '-loglevel', 'error', video_out
        ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 and vaapi and scale_h:
        # Fallback a software si VAAPI falla (ej. solo NVIDIA, sin Intel/AMD)
        if result.stderr:
            print(f"[FFmpeg VAAPI falló, usando software] {result.stderr.strip()[:200]}")
        command_fb = [
            'ffmpeg', '-y', '-threads', '0', '-ss', str(start), '-to', str(end),
            '-i', video_in, '-vf', f"scale=-2:{scale_h}",
            '-c:v', 'libx264', '-loglevel', 'error', video_out
        ]
        result = subprocess.run(command_fb, capture_output=True, text=True)
    if result.returncode != 0 and result.stderr:
        print(f"[FFmpeg] {result.stderr.strip()[:400]}")
    return result.returncode == 0 and os.path.isfile(video_out)


def scale_video(video_in: str, video_out: str, height: int | None = None) -> bool:
    """
    Escala un vídeo a la altura indicada (sin recortar).
    Usa el mismo pipeline que cut_clip (VAAPI o software).
    Devuelve True si ok, False si falló.
    """
    scale_h = height or CLIP_SCALE_HEIGHT
    vaapi = VAAPI_DEVICE
    if scale_h is None:
        scale_h = 720  # fallback
    scale_expr = f"scale=-2:{scale_h}"
    if vaapi and scale_h:
        cmd = [
            'ffmpeg', '-y', '-threads', '0', '-hwaccel', 'vaapi',
            '-hwaccel_device', str(vaapi), '-hwaccel_output_format', 'vaapi',
            '-i', video_in, '-vf', f"scale_vaapi=-2:{scale_h}",
            '-c:v', 'h264_vaapi', '-vaapi_device', str(vaapi),
            '-loglevel', 'error', video_out
        ]
    else:
        cmd = [
            'ffmpeg', '-y', '-threads', '0', '-i', video_in,
            '-vf', scale_expr, '-c:v', 'libx264', '-loglevel', 'error', video_out
        ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and vaapi:
        cmd_fb = [
            'ffmpeg', '-y', '-threads', '0', '-i', video_in,
            '-vf', scale_expr, '-c:v', 'libx264', '-loglevel', 'error', video_out
        ]
        r = subprocess.run(cmd_fb, capture_output=True, text=True)
    return r.returncode == 0


def run_debug_extract(video_path: str, yolo_pose_model: str | None = None, output_dir: str | None = None) -> Path | None:
    """
    Modo debug/test: extrae poses de un único vídeo y guarda en carpeta temporal.
    Genera poses_full.npy y poses.npy (normalizados 0-1) por usuario, igual que el flujo normal.

    yolo_pose_model: si se indica (p. ej. \"yolo11n-pose.pt\"), usa ese modelo YOLO pose para
    esta extracción en lugar del global cargado desde config (p. ej. yolo11x-pose.pt).

    output_dir: si se indica, guarda los .npy en ese directorio (se crea si no existe) en lugar
    del directorio por defecto basado en OUTPUT de config.py.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        print(f"[DEBUG] Vídeo no encontrado: {video_path}")
        return None
    stem = video_path.stem or "clip"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
    else:
        out_dir = OUTPUT / "debug_extract" / f"{timestamp}_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_clip = out_dir / "_temp_scaled.mp4"
    print(f"[DEBUG] Procesando: {video_path}")
    print(f"[DEBUG] Salida: {out_dir}")

    if not scale_video(str(video_path), str(temp_clip)):
        print("[DEBUG] Error al escalar el vídeo")
        return None

    cap = cv2.VideoCapture(str(temp_clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    cap.release()

    override = (yolo_pose_model or "").strip()
    if override:
        yolo_path = _resolve_model_path(override)
        print(f"[DEBUG] YOLO pose (override): {yolo_path}")
        track_model = YOLO(yolo_path)
        yolo_meta_path = yolo_path
    else:
        track_model = model
        yolo_meta_path = str(_MODEL_RESOLVED)

    results = track_model.track(
        source=str(temp_clip),
        tracker=_tracker_yaml_path(),
        persist=True,
        verbose=False,
        stream=True,
        device=DEVICE,
        half=True,
    )
    temp_person_data = {}
    # 1:1 frame–pose: cada iteración = un frame; si no hay detección se rellena con NAN_POSE.

    for r in tqdm(results, desc="Extrayendo poses"):
        frame = getattr(r, 'orig_img', None)
        if frame is None or r.keypoints is None or r.boxes.id is None:
            for tid in list(temp_person_data.keys()):
                temp_person_data[tid]['poses_full'].append(NAN_POSE.copy())
                temp_person_data[tid]['keypoint_masks'].append(EMPTY_KP_MASK.copy())
                temp_person_data[tid]['total'] += 1
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist()
        kpts = r.keypoints.xyn.cpu().numpy()
        confs = r.keypoints.conf.cpu().numpy()
        # Tracks ya vistos pero no detectados en este frame: 1:1 con NaN
        for tid in list(temp_person_data.keys()):
            if tid not in ids:
                temp_person_data[tid]['poses_full'].append(NAN_POSE.copy())
                temp_person_data[tid]['keypoint_masks'].append(EMPTY_KP_MASK.copy())
                temp_person_data[tid]['total'] += 1
        for i, track_id in enumerate(ids):
            if track_id not in temp_person_data:
                c_t, c_b = get_color_attributes(frame, boxes[i])
                temp_person_data[track_id] = {
                    'poses_full': [], 'poses': [], 'keypoint_masks': [],
                    'v_cnt': 0, 'total': 0, 'body_visible_cnt': 0,
                    'clothes': {'top': c_t, 'bottom': c_b},
                    'kp_conf_sum': 0.0, 'occluded_frames': 0, 'bbox_ratios': [],
                }
            conf_kp = confs[i][KEEP_KPS]
            kpt_pose = _build_masked_pose(kpts[i], confs[i])
            body_visible = int(_frame_body_visible(kpt_pose))
            temp_person_data[track_id]['body_visible_cnt'] += body_visible
            usable = _frame_usable(kpt_pose)
            temp_person_data[track_id]['poses_full'].append(kpt_pose)
            temp_person_data[track_id]['keypoint_masks'].append(_keypoint_mask_from_pose(kpt_pose))
            if usable:
                temp_person_data[track_id]['poses'].append(kpt_pose)
                temp_person_data[track_id]['v_cnt'] += 1
            temp_person_data[track_id]['total'] += 1
            visible_confs = conf_kp[conf_kp > MIN_KP_CONF]
            temp_person_data[track_id]['kp_conf_sum'] += (
                float(np.mean(visible_confs)) if visible_confs.size else 0.0
            )
            if any(confs[i][idx] < OCCLUSION_CONF_THR for idx in CRITICAL_KPS):
                temp_person_data[track_id]['occluded_frames'] += 1
            bbox = boxes[i]
            h_bbox = bbox[3] - bbox[1]
            w_bbox = max(bbox[2] - bbox[0], 1e-6)
            temp_person_data[track_id]['bbox_ratios'].append(float(h_bbox / w_bbox))

    if not temp_person_data:
        print("[DEBUG] No se detectaron personas")
        return None

    users_meta = []
    for tid, info in temp_person_data.items():
        total = info['total']
        if total == 0:
            continue
        body_vis = info['body_visible_cnt']
        if body_vis < BODY_VISIBLE_MIN_FRAMES or (body_vis / total) < BODY_VISIBLE_MIN_RATIO:
            continue  # No guardar: solo mano, cabeza o visibilidad insuficiente
        valid = info['v_cnt']
        rel = valid / total
        user_dir = out_dir / f"user_{tid}"
        user_dir.mkdir(exist_ok=True)
        _save_user_pose_files(user_dir, info)
        users_meta.append({
            "track_id": int(tid),
            "valid_pct": round(rel * 100, 1),
            "rel": round(rel, 2),
            "valid_frames": int(valid),
            "total_frames": int(total),
            "poses_full_count": len(info['poses_full']),
            "poses_filtered_count": len(info['poses']),
            "clothes": info['clothes'],
        })

    clip_mp4 = out_dir / "clip.mp4"
    if temp_clip.exists():
        shutil.copy2(temp_clip, clip_mp4)
        temp_clip.unlink()
    else:
        scale_video(str(video_path), str(clip_mp4))

    frame_count = max(info['total'] for info in temp_person_data.values()) if temp_person_data else 0
    meta = {
        "clip_name": stem,
        "video_source": str(video_path),
        "debug_mode": True,
        "fps": fps,
        "frame_count": frame_count,
        "yolo_model": str(yolo_meta_path),
        "yolo_backend": "engine" if str(yolo_meta_path).endswith(".engine") else "pt",
        "users": users_meta,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[DEBUG] Listo. Salida: {out_dir}")
    if users_meta:
        uid = users_meta[0]["track_id"]
        print(f"        Visualizar: python visualize_video_pose.py {clip_mp4} {out_dir / f'user_{uid}' / 'poses_full.npy'}")
    else:
        print("        (No se guardaron usuarios: ninguno cumple visibilidad de torso/brazos)")
    return out_dir


def get_color_attributes(frame, bbox):
    """Extrae colores RGB promedios de la ropa."""
    x1, y1, x2, y2 = map(int, bbox)
    person_img = frame[max(0, y1):y2, max(0, x1):x2]
    if person_img.size == 0: return "unknown", "unknown"
    h, w = person_img.shape[:2]
    top = person_img[0:int(h*0.4), :]
    bottom = person_img[int(h*0.4):h, :]
    def avg_col(roi):
        c = roi.mean(axis=(0, 1))
        return f"RGB({int(c[2])},{int(c[1])},{int(c[0])})"
    return avg_col(top), avg_col(bottom)


def _resolve_track_model(yolo_pose_model: str | None = None):
    """Modelo YOLO pose para tracking (override opcional)."""
    override = (yolo_pose_model or "").strip()
    if override:
        yolo_path = _resolve_model_path(override)
        return YOLO(yolo_path), yolo_path
    return model, str(_MODEL_RESOLVED)


def _read_clip_timing(clip_path: str) -> tuple[float, float, int]:
    """Devuelve (fps, clip_duration_seconds, video_frame_count)."""
    cap = cv2.VideoCapture(clip_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps is None or fps <= 0:
        fps = DEFAULT_FPS
    clip_duration = video_frame_count / fps if fps > 0 and video_frame_count > 0 else 0.0
    return float(fps), clip_duration, video_frame_count


def _track_poses_on_clip(
    clip_path: str,
    track_model,
    *,
    clip_duration: float | None = None,
) -> tuple[dict, float, int, int, int]:
    """
    Ejecuta YOLO track sobre un clip ya recortado.
    Devuelve temp_person_data, fps, min_valid_frames, processed_frame_count, video_frame_count.
    """
    results = track_model.track(
        source=clip_path,
        tracker=_tracker_yaml_path(),
        persist=True,
        verbose=False,
        stream=True,
        device=DEVICE,
        half=True,
    )
    temp_person_data: dict = {}
    fps, duration_from_file, video_frame_count = _read_clip_timing(clip_path)
    if clip_duration is None or clip_duration <= 0:
        clip_duration = duration_from_file

    min_valid_seconds = clip_duration * MIN_COVERAGE_RATIO
    min_valid_frames = int(min_valid_seconds * fps) if clip_duration > 0 else 0

    for r in results:
        frame = getattr(r, "orig_img", None)
        if frame is None or r.keypoints is None or r.boxes.id is None:
            for tid in list(temp_person_data.keys()):
                temp_person_data[tid]["poses_full"].append(NAN_POSE.copy())
                temp_person_data[tid]["keypoint_masks"].append(EMPTY_KP_MASK.copy())
                temp_person_data[tid]["total"] += 1
            continue

        boxes_gpu = r.boxes.xyxy
        ids_gpu = r.boxes.id.int()
        kpts_gpu = r.keypoints.xyn
        confs_gpu = r.keypoints.conf
        ids = ids_gpu.cpu().tolist()
        kpts = kpts_gpu.cpu().numpy()
        confs = confs_gpu.cpu().numpy()
        boxes = boxes_gpu.cpu().numpy()

        for tid in list(temp_person_data.keys()):
            if tid not in ids:
                temp_person_data[tid]["poses_full"].append(NAN_POSE.copy())
                temp_person_data[tid]["keypoint_masks"].append(EMPTY_KP_MASK.copy())
                temp_person_data[tid]["total"] += 1
        for i, track_id in enumerate(ids):
            if track_id not in temp_person_data:
                c_t, c_b = get_color_attributes(frame, boxes[i])
                temp_person_data[track_id] = {
                    "poses_full": [],
                    "poses": [],
                    "keypoint_masks": [],
                    "v_cnt": 0,
                    "total": 0,
                    "body_visible_cnt": 0,
                    "clothes": {"top": c_t, "bottom": c_b},
                    "kp_conf_sum": 0.0,
                    "occluded_frames": 0,
                    "bbox_ratios": [],
                }
            conf_kp = confs[i][KEEP_KPS]
            kpt_pose = _build_masked_pose(kpts[i], confs[i])
            body_visible = int(_frame_body_visible(kpt_pose))
            temp_person_data[track_id]["body_visible_cnt"] += body_visible
            usable = _frame_usable(kpt_pose)
            temp_person_data[track_id]["poses_full"].append(kpt_pose)
            temp_person_data[track_id]["keypoint_masks"].append(_keypoint_mask_from_pose(kpt_pose))
            if usable:
                temp_person_data[track_id]["poses"].append(kpt_pose)
                temp_person_data[track_id]["v_cnt"] += 1
            temp_person_data[track_id]["total"] += 1
            visible_confs = conf_kp[conf_kp > MIN_KP_CONF]
            temp_person_data[track_id]["kp_conf_sum"] += (
                float(np.mean(visible_confs)) if visible_confs.size else 0.0
            )
            if any(confs[i][idx] < OCCLUSION_CONF_THR for idx in CRITICAL_KPS):
                temp_person_data[track_id]["occluded_frames"] += 1
            h_bbox = boxes[i][3] - boxes[i][1]
            w_bbox = max(boxes[i][2] - boxes[i][0], 1e-6)
            temp_person_data[track_id]["bbox_ratios"].append(float(h_bbox / w_bbox))

    processed_frame_count = max((info["total"] for info in temp_person_data.values()), default=0)
    return temp_person_data, fps, min_valid_frames, processed_frame_count, video_frame_count


def _save_clip_pose_artifacts(
    *,
    data_dir: Path,
    clip_name: str,
    category: str,
    clip_path: str,
    temp_person_data: dict,
    fps: float,
    min_valid_frames: int,
    processed_frame_count: int,
    video_frame_count: int,
    clip_duration: float,
    yolo_meta_path: str,
    meta_fields: dict,
    old_users_meta: list | None = None,
    copy_clip: bool = True,
) -> list[dict]:
    """Guarda user_X/, meta.json y opcionalmente clip.mp4. Devuelve users_meta."""
    data_dir.mkdir(parents=True, exist_ok=True)
    old_by_tid: dict[int, dict] = {}
    if old_users_meta:
        for u in old_users_meta:
            try:
                old_by_tid[int(u["track_id"])] = u
            except (KeyError, TypeError, ValueError):
                continue

    users_meta: list[dict] = []
    for tid, info in temp_person_data.items():
        total_frames = info["total"]
        valid_frames = info["v_cnt"]
        if total_frames == 0:
            continue

        user_seconds = total_frames / fps if fps > 0 else 0.0
        if MIN_USER_SECONDS > 0 and user_seconds < MIN_USER_SECONDS:
            print(
                f"[DESCARTADO usuario] Clip '{clip_name}' user_{tid} | "
                f"duración={user_seconds:.2f}s < {MIN_USER_SECONDS}s"
            )
            continue

        body_vis = info.get("body_visible_cnt", 0)
        if body_vis < BODY_VISIBLE_MIN_FRAMES or (body_vis / total_frames) < BODY_VISIBLE_MIN_RATIO:
            print(
                f"[DESCARTADO usuario] Clip '{clip_name}' user_{tid} | "
                f"solo mano/cabeza/visibilidad insuficiente (body_visible={body_vis}/{total_frames})"
            )
            continue

        rel = valid_frames / total_frames
        valid_pct = round(rel * 100, 1)
        passes_filters = rel >= RELIABILITY_THR and valid_frames >= min_valid_frames
        kp_conf_avg = info["kp_conf_sum"] / total_frames if total_frames > 0 else 0.0
        occlusion_ratio = round(info["occluded_frames"] / total_frames * 100, 1) if total_frames > 0 else 0.0
        bbox_aspect_ratio = round(float(np.mean(info["bbox_ratios"])), 3) if info["bbox_ratios"] else 0.0
        poses_full_arr = np.array(info["poses_full"])
        if len(poses_full_arr) > 1:
            d = np.diff(poses_full_arr, axis=0)
            velocity_mag = np.sqrt((d ** 2).sum(axis=-1))
            subject_velocity = round(float(np.mean(velocity_mag)), 5)
        else:
            subject_velocity = 0.0

        prev = old_by_tid.get(int(tid), {})
        user_meta = {
            "track_id": int(tid),
            "valid_pct": valid_pct,
            "rel": round(rel, 2),
            "valid_frames": int(valid_frames),
            "total_frames": int(total_frames),
            "poses_full_count": len(info["poses_full"]),
            "poses_filtered_count": len(info["poses"]),
            "passes_filters": passes_filters,
            "keypoint_confidence_avg": round(kp_conf_avg, 3),
            "occlusion_ratio": occlusion_ratio,
            "bbox_aspect_ratio": bbox_aspect_ratio,
            "subject_velocity": subject_velocity,
            "clothes": info["clothes"],
            "user_cat": int(prev.get("user_cat", category)),
        }
        if "is_primary_robber" in prev:
            user_meta["is_primary_robber"] = prev["is_primary_robber"]
        users_meta.append(user_meta)

        user_dir = data_dir / f"user_{tid}"
        user_dir.mkdir(exist_ok=True)
        _save_user_pose_files(user_dir, info)
        if not passes_filters:
            print(
                f"[SIN filtros] Clip '{clip_name}' user_{tid} | "
                f"valid_pct={valid_pct}% | valid_frames={valid_frames}, min={min_valid_frames}"
            )

    try:
        clip_cat_int = int(category)
    except (TypeError, ValueError):
        clip_cat_int = None
    if clip_cat_int == 6 and users_meta and not any("is_primary_robber" in u for u in users_meta):
        for idx_u, u in enumerate(users_meta):
            u["is_primary_robber"] = idx_u == 0

    if video_frame_count and video_frame_count != processed_frame_count:
        print(
            f"[AVISO] Clip '{clip_name}': frames en vídeo={video_frame_count}, "
            f"frames con poses={processed_frame_count}"
        )

    meta = {
        "clip_name": clip_name,
        "fps": fps,
        "frame_count": processed_frame_count,
        "video_frame_count": video_frame_count,
        "min_valid_frames": int(min_valid_frames),
        "cat": category,
        "clip_duration": clip_duration,
        "yolo_model": str(yolo_meta_path),
        "yolo_backend": "engine" if str(yolo_meta_path).endswith(".engine") else "pt",
        "users": users_meta,
        **meta_fields,
    }
    if copy_clip and SAVE_PROCESSED_CLIP and clip_path and os.path.exists(clip_path):
        meta["clip_video"] = "clip.mp4"
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    if copy_clip and SAVE_PROCESSED_CLIP and clip_path and os.path.exists(clip_path):
        dest_clip = data_dir / "clip.mp4"
        try:
            shutil.copy2(clip_path, dest_clip)
        except Exception as e:
            print(f"[AVISO] No se pudo copiar clip en data_result: {e}")

    return users_meta


def _resolve_data_result_root(path: Path) -> Path:
    """Acepta .../data_result o OUTPUT_BASE (con data_result dentro)."""
    p = path.expanduser().resolve()
    if p.name == "data_result":
        return p
    nested = p / "data_result"
    if nested.is_dir():
        return nested
    return p


def iter_existing_data_result_clips(data_result_root: Path):
    """Recorre categorías numéricas y clips con meta.json + clip.mp4."""
    base = _resolve_data_result_root(data_result_root)
    if not base.is_dir():
        raise FileNotFoundError(f"No existe data_result: {base}")
    for cat_dir in sorted(base.iterdir()):
        if not cat_dir.is_dir():
            continue
        try:
            int(cat_dir.name)
        except ValueError:
            continue
        category = cat_dir.name
        for clip_dir in sorted(cat_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            meta_path = clip_dir / "meta.json"
            clip_mp4 = clip_dir / "clip.mp4"
            if meta_path.is_file() and clip_mp4.is_file():
                yield category, clip_dir.name, clip_dir, meta_path, clip_mp4


def process_from_data_result(
    source_data_result: Path,
    dest_data_result_base: Path,
    *,
    track_model,
    yolo_meta_path: str,
    failed_clips: list,
    category_limits: dict[str, int] | None = None,
    category_counters: dict[str, int] | None = None,
    single_user_only: bool = False,
    max_clips: int | None = None,
) -> None:
    """
    Re-extrae poses desde clip.mp4 ya existentes (sin CSV ni FFmpeg).
    Copia clip.mp4 a dest_data_result_base/{cat}/{clip}/ y regenera user_X/ + meta.json.
    """
    if category_limits is None:
        category_limits = {}
    if category_counters is None:
        category_counters = {}

    src_root = _resolve_data_result_root(source_data_result)
    dest_root = Path(dest_data_result_base)
    dest_root.mkdir(parents=True, exist_ok=True)
    print(f"Re-extracción desde: {src_root}")
    print(f"Salida data_result:   {dest_root}")

    clips = list(iter_existing_data_result_clips(src_root))
    if max_clips is not None:
        clips = clips[: max_clips]
    print(f"Clips encontrados: {len(clips)}")

    for category, clip_name, source_dir, meta_path, clip_mp4 in tqdm(clips, desc="Re-extrayendo poses"):
        limit = category_limits.get(category)
        if limit is not None and category_counters.get(category, 0) >= limit:
            print(f"[OMITIDO] cat={category} límite {limit} alcanzado")
            continue

        dest_dir = dest_root / category / clip_name
        if dest_dir.is_dir() and (dest_dir / "meta.json").is_file():
            print(f"[OMITIDO] Ya existe {dest_dir} (reanudación)")
            category_counters[category] = category_counters.get(category, 0) + 1
            continue

        try:
            with open(meta_path, encoding="utf-8") as f:
                old_meta = json.load(f)
        except Exception as e:
            failed_clips.append({
                "clip": str(source_dir),
                "error": f"No se pudo leer meta.json: {e}",
            })
            continue

        clip_duration = float(old_meta.get("clip_duration") or 0.0)
        if clip_duration <= 0:
            _, clip_duration, _ = _read_clip_timing(str(clip_mp4))

        try:
            temp_person_data, fps, min_valid_frames, processed_frame_count, video_frame_count = (
                _track_poses_on_clip(str(clip_mp4), track_model, clip_duration=clip_duration)
            )
        except Exception as e:
            failed_clips.append({"clip": str(source_dir), "error": str(e)})
            print(f"[ERROR] {clip_name}: {e}")
            continue

        if not temp_person_data:
            failed_clips.append({"clip": str(source_dir), "error": "No se detectaron personas"})
            continue

        if single_user_only and len(temp_person_data) > 1:
            print(f"[OMITIDO] {clip_name}: multiusuario (n={len(temp_person_data)} tracks)")
            continue

        preserve_keys = (
            "video_source",
            "row_csv",
            "t_start",
            "t_end",
            "full_clip",
        )
        meta_fields = {k: old_meta[k] for k in preserve_keys if k in old_meta}
        meta_fields["reextracted_from"] = str(source_dir.resolve())
        if old_meta.get("yolo_model"):
            meta_fields["yolo_model_source"] = old_meta.get("yolo_model")

        _save_clip_pose_artifacts(
            data_dir=dest_dir,
            clip_name=clip_name,
            category=category,
            clip_path=str(clip_mp4),
            temp_person_data=temp_person_data,
            fps=fps,
            min_valid_frames=min_valid_frames,
            processed_frame_count=processed_frame_count,
            video_frame_count=video_frame_count,
            clip_duration=clip_duration,
            yolo_meta_path=yolo_meta_path,
            meta_fields=meta_fields,
            old_users_meta=old_meta.get("users") or [],
            copy_clip=True,
        )
        category_counters[category] = category_counters.get(category, 0) + 1


def process_single_csv(
    CSV_PATH,
    VIDEOS_DIR_RES,
    DATA_RESULT_BASE,
    TEMP_CLIPS_BASE,
    failed_clips: list,
    used_clip_names: set,
    dir_rel_path: str | None = None,
    category_limits: dict[str, int] | None = None,
    category_counters: dict[str, int] | None = None,
    track_model=None,
    yolo_meta_path_override: str | None = None,
):
    """Procesa un único CSV. Errores por clip se añaden a failed_clips."""
    pose_model = track_model or model
    yolo_path_for_meta = yolo_meta_path_override or str(_MODEL_RESOLVED)
    # 1. Detectar fila de inicio (primera con HH:MM:SS en col 2) y leer CSV
    start_row = find_start_row(CSV_PATH)
    print(f"Inicio de datos en fila CSV: {start_row}")
    print(f"Vídeos: {VIDEOS_DIR_RES} | Temp clips: {TEMP_CLIPS_BASE} | Data: {DATA_RESULT_BASE}")
    # header=None: la primera fila leída es dato, no encabezado (evita saltarnos una fila)
    df = pd.read_csv(CSV_PATH, skiprows=range(0, start_row - 1), header=None)
    
    if DEBUG_MODE:
        df = df.head(N_DEBUG)  # Limitar filas (clips) de este CSV

    # Cada iteración = un clip: estado fresco (temp_person_data, etc.). Nada se arrastra del clip anterior.
    if category_limits is None:
        category_limits = {}
    if category_counters is None:
        category_counters = {}
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Procesando CSV"):
        video_rel_path = str(row.iloc[0]).strip().strip('"').strip("'")
        t_start = _normalize_hms(str(row.iloc[1]))
        t_end = _normalize_hms(str(row.iloc[2]))
        category = str(int(row.iloc[3]))
        use_full_clip = _is_full_clip_range(t_start, t_end)

        # fila_csv = número de fila en el CSV
        fila_csv = start_row + int(index)

        # Límite por categoría (máx. clips a procesar por clase). Si se supera, se omite el clip.
        limit = category_limits.get(category)
        if limit is not None and category_counters.get(category, 0) >= limit:
            print(
                f"[OMITIDO] Clip fila {fila_csv} (cat={category}) porque alcanzó el "
                f"límite de {limit} clips para esa categoría."
            )
            continue

        video_full_path = _resolve_video_path(VIDEOS_DIR_RES, video_rel_path)
        if not os.path.exists(video_full_path):
            failed_clips.append({
                "csv": str(CSV_PATH), "row": fila_csv, "video": video_rel_path,
                "error": "Archivo no encontrado",
            })
            continue

        if use_full_clip:
            clip_duration = _video_duration_from_file(video_full_path)
            if clip_duration <= 0:
                failed_clips.append({
                    "csv": str(CSV_PATH), "row": fila_csv, "video": video_rel_path,
                    "error": "Modo clip completo (00:00:00–00:00:00): no se pudo leer duración del vídeo",
                })
                continue
        else:
            clip_duration = max(0.0, hms_to_seconds(t_end) - hms_to_seconds(t_start))
            if clip_duration <= 0:
                failed_clips.append({
                    "csv": str(CSV_PATH), "row": fila_csv, "video": video_rel_path,
                    "error": f"Duración inválida: inicio={t_start}, fin={t_end}",
                })
                continue

        cap = None
        clip_path = None
        try:
            # 2. Nombre único del clip (incluye ruta dir: fecha_cam) y ruta temporal
            clip_name = make_clip_name(
                video_rel_path, t_start, t_end, category, used_clip_names, dir_rel_path
            )
            data_dir = Path(DATA_RESULT_BASE) / category / clip_name
            if data_dir.is_dir():
                print(
                    f"[OMITIDO] Clip '{clip_name}' fila {fila_csv} ya existe en "
                    f"{data_dir}; se omite (reanudación)."
                )
                category_counters[category] = category_counters.get(category, 0) + 1
                continue

            temp_cat_dir = Path(TEMP_CLIPS_BASE) / category
            temp_cat_dir.mkdir(parents=True, exist_ok=True)
            clip_is_temp = False

            if use_full_clip:
                clip_path = video_full_path
                print(
                    f"[Clip COMPLETO] {clip_name} | Fila CSV: {fila_csv} | "
                    f"Duración: {seconds_to_hms(clip_duration)} | Video: {video_rel_path}"
                )
            else:
                clip_path = str(temp_cat_dir / f"{clip_name}.mp4")
                print(
                    f"[Clip] {clip_name} | Fila CSV: {fila_csv} | "
                    f"Inicio: {t_start} | Fin: {t_end} | Video: {video_rel_path}"
                )
                if not cut_clip(video_full_path, t_start, t_end, clip_path):
                    failed_clips.append({
                        "csv": str(CSV_PATH), "row": fila_csv, "video": video_rel_path,
                        "error": f"FFmpeg no generó el clip temporal: {clip_path}",
                    })
                    continue
                clip_is_temp = True

            # 3. YOLO pose tracking sobre el clip (ya recortado o temporal)
            temp_person_data, fps, min_valid_frames, processed_frame_count, video_frame_count = (
                _track_poses_on_clip(clip_path, pose_model, clip_duration=clip_duration)
            )

            if not temp_person_data:
                failed_clips.append({
                    "csv": str(CSV_PATH), "row": fila_csv, "video": video_rel_path,
                    "error": "No se detectaron personas",
                })
                continue

            if SINGLE_USER_ONLY and len(temp_person_data) > 1:
                print(
                    f"[OMITIDO] Clip '{clip_name}' | Fila CSV: {fila_csv} | "
                    f"multiusuario detectado (n={len(temp_person_data)} tracks) — se salta la fila"
                )
                continue

            data_dir = Path(DATA_RESULT_BASE) / category / clip_name
            _save_clip_pose_artifacts(
                data_dir=data_dir,
                clip_name=clip_name,
                category=category,
                clip_path=clip_path,
                temp_person_data=temp_person_data,
                fps=fps,
                min_valid_frames=min_valid_frames,
                processed_frame_count=processed_frame_count,
                video_frame_count=video_frame_count,
                clip_duration=clip_duration,
                yolo_meta_path=yolo_path_for_meta,
                meta_fields={
                    "video_source": str(Path(video_full_path).resolve()),
                    "row_csv": int(fila_csv),
                    "t_start": str(t_start),
                    "t_end": seconds_to_hms(clip_duration) if use_full_clip else str(t_end),
                    "full_clip": use_full_clip,
                },
                copy_clip=SAVE_PROCESSED_CLIP,
            )

            if DELETE_TEMP_VIDEOS and clip_is_temp and clip_path and os.path.exists(clip_path):
                os.remove(clip_path)

            category_counters[category] = category_counters.get(category, 0) + 1

        except Exception as e:
            failed_clips.append({
                "csv": str(CSV_PATH), "row": fila_csv, "video": video_rel_path,
                "error": str(e),
            })
            print(f"[ERROR] Clip fila {fila_csv} ({video_rel_path}): {e}")


def main():
    # Log a fichero desde el inicio (todo lo que se imprima irá a terminal + log; útil para SSH)
    log_file, original_stdout, log_path = None, None, None
    log_result = _setup_logging()
    if log_result:
        log_file, original_stdout, log_path = log_result
        print(f"Log guardado en: {log_path}")
        print(f"Modelo pose: {_MODEL_RESOLVED}")

    parser = argparse.ArgumentParser(description="Extractor de poses YOLO para clips")
    parser.add_argument("--debug", "--test", dest="debug_video", metavar="VIDEO", help="Modo debug: extrae poses de un único vídeo en carpeta temporal (poses_full.npy, poses.npy)")
    parser.add_argument("--limit", "-n", type=int, default=None, metavar="N", help="Solo procesar los primeros N clips (CSV o --from-data-result)")
    parser.add_argument(
        "--from-data-result",
        dest="from_data_result",
        default=None,
        metavar="DIR",
        help="Re-extrae poses desde data_result existente (meta.json + clip.mp4), sin CSV ni FFmpeg",
    )
    parser.add_argument(
        "--output-base",
        dest="output_base",
        default=None,
        metavar="DIR",
        help="Carpeta OUTPUT_BASE de salida (data_result/ dentro). Obligatorio distinta del origen en --from-data-result",
    )
    parser.add_argument(
        "--yolo-pose-model",
        dest="yolo_pose_model",
        default=None,
        metavar="MODELO",
        help="Modelo YOLO pose (p. ej. yolo26s-pose.pt). Sobrescribe config.py",
    )
    parser.add_argument(
        "--single-user-only",
        action="store_true",
        help="En --from-data-result: omitir clips con más de un track",
    )
    parser.add_argument("--output", "-o", dest="output_dir", default=None, metavar="DIR", help="Directorio donde guardar los .npy en modo debug. Sobrescribe la ruta por defecto de config.py")
    parser.add_argument(
        "--continue-on-missing",
        "--skip-missing-videos",
        dest="continue_on_missing",
        action="store_true",
        help="Si hay vídeos no encontrados en los CSV, omitirlos y continuar con los que sí existen",
    )
    args = parser.parse_args()

    output_root = Path(args.output_base).expanduser().resolve() if args.output_base else OUTPUT

    if args.limit is not None and args.debug_video is None:
        global N_DEBUG, DEBUG_MODE
        N_DEBUG = args.limit
        DEBUG_MODE = True

    if args.debug_video:
        run_debug_extract(args.debug_video, yolo_pose_model=args.yolo_pose_model, output_dir=args.output_dir)
        if log_file:
            if original_stdout is not None:
                sys.stdout = original_stdout
            log_file.close()
        return

    track_model, yolo_meta_path = _resolve_track_model(args.yolo_pose_model)
    if args.yolo_pose_model:
        print(f"Modelo pose (override): {yolo_meta_path}")

    if args.from_data_result:
        source = Path(args.from_data_result).expanduser().resolve()
        src_data = _resolve_data_result_root(source)
        dest_data = output_root / "data_result"
        try:
            if src_data.resolve() == dest_data.resolve():
                print("ERROR: --output-base debe apuntar a otra carpeta distinta del origen.")
                if log_file:
                    if original_stdout is not None:
                        sys.stdout = original_stdout
                    log_file.close()
                return
        except FileNotFoundError:
            pass

        print(f"Dispositivo: {DEVICE}")
        print(f"Modo: re-extracción desde data_result (sin CSV)")
        print(f"Salida: {output_root} (data_result/)")

        failed_clips: list = []
        category_limits = _load_category_limits()
        category_counters: dict[str, int] = {}
        max_clips = N_DEBUG if DEBUG_MODE else None
        process_from_data_result(
            source,
            dest_data,
            track_model=track_model,
            yolo_meta_path=yolo_meta_path,
            failed_clips=failed_clips,
            category_limits=category_limits,
            category_counters=category_counters,
            single_user_only=args.single_user_only,
            max_clips=max_clips,
        )
        if failed_clips:
            print("\n" + "=" * 60)
            print("RESUMEN DE CLIPS CON ERRORES")
            print("=" * 60)
            for fc in failed_clips:
                print(f"  Clip: {fc.get('clip', fc.get('csv', '?'))}")
                print(f"    Error: {fc.get('error')}")
            print(f"\nTotal clips con error: {len(failed_clips)}")
        else:
            print("\nTodos los clips se procesaron correctamente.")
        print(f"\n[FIN] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Re-extracción terminada.")
        if log_file:
            if original_stdout is not None:
                sys.stdout = original_stdout
            log_file.close()
        return

    # 2. Validación previa (security): un único resumen sobre todos los CSVs a procesar
    allow_missing = args.continue_on_missing
    if allow_missing:
        print("Modo --continue-on-missing: los vídeos no encontrados se omitirán.")

    experiments_for_validation = get_experiments()
    if experiments_for_validation:
        csv_paths = [exp["csv"] for exp in experiments_for_validation]
        path_roots = get_path_roots()
        if path_roots:
            print(
                f"Validando {len(csv_paths)} CSV(s) en "
                f"{len(path_roots)} PATH_ROOT(S)..."
            )
        else:
            print(f"Validando {len(csv_paths)} CSV(s)...")
        validation = validate_csv_files(
            csv_paths,
            allow_missing_videos=allow_missing,
        )
        if not validation.get("ok"):
            print("Abortando: hay errores en los CSVs. Corrígelos antes de ejecutar el extractor.")
            if log_file:
                if original_stdout is not None:
                    sys.stdout = original_stdout
                log_file.close()
            return
    else:
        path_to_validate = os.path.dirname(os.path.abspath(CSV_PATH or "."))
        validation = validate_folder(path_to_validate, allow_missing_videos=allow_missing)
        if not validation.get("ok"):
            print("Abortando: hay errores en los CSVs. Corrígelos antes de ejecutar el extractor.")
            if log_file:
                if original_stdout is not None:
                    sys.stdout = original_stdout
                log_file.close()
            return

    # 3. Device y ruta de salida
    print(f"Dispositivo: {DEVICE}")
    if args.output_base:
        print(f"Salida: {output_root} (temp_clips/ + data_result/)")
    else:
        print(f"Salida: {OUTPUT} (temp_clips/ + data_result/)")

    experiments = get_experiments()
    if not experiments:
        print("Error: no se encontraron CSV. Configura PATH_ROOTS o CSV_PATH en config.py")
        if log_file:
            if original_stdout is not None:
                sys.stdout = original_stdout
            log_file.close()
        return

    path_roots = get_path_roots()
    if path_roots:
        print(f"PATH_ROOTS ({len(path_roots)}): " + ", ".join(str(p) for p in path_roots))

    if DEBUG_MODE:
        experiments = experiments[:1]  # Solo primer CSV
        print(f"[DEBUG] Procesando solo {N_DEBUG} clips del primer CSV")

    temp_clips_base = output_root / "temp_clips"
    data_result_base = output_root / "data_result"
    temp_clips_base.mkdir(parents=True, exist_ok=True)
    data_result_base.mkdir(parents=True, exist_ok=True)

    # Destinos por experimento (varios PATH_ROOTS pueden compartir o separar data_result)
    dest_dirs: set[str] = set()
    for exp in experiments:
        dr = exp.get("data_result_dir") or str(data_result_base)
        dest_dirs.add(dr)
        Path(dr).mkdir(parents=True, exist_ok=True)
        tc = exp.get("temp_clips_dir") or str(temp_clips_base)
        Path(tc).mkdir(parents=True, exist_ok=True)
    if len(dest_dirs) > 1:
        print(f"Salidas data_result ({len(dest_dirs)}):")
        for d in sorted(dest_dirs):
            print(f"  - {d}")
    else:
        print(f"Salida data_result: {next(iter(dest_dirs))}")

    # Límite global opcional por categoría (máx. clips a procesar por clase, en todos los CSV)
    category_limits = _load_category_limits()
    if category_limits:
        print(f"Límites por categoría (config_pose_extraction.json): {category_limits}")

    track_model, yolo_meta_path = _resolve_track_model(args.yolo_pose_model)
    if args.yolo_pose_model:
        print(f"Modelo pose (override): {yolo_meta_path}")

    failed_clips = []
    used_clip_names_by_dest: dict[str, set] = {}
    category_counters: dict[str, int] = {}
    for i, exp in enumerate(experiments):
        exp_data_result = exp.get("data_result_dir") or str(data_result_base)
        exp_temp = exp.get("temp_clips_dir") or str(temp_clips_base)
        used_names = used_clip_names_by_dest.setdefault(exp_data_result, set())
        print(f"\n{'='*60}")
        print(f"[Experimento {i+1}/{len(experiments)}] CSV: {exp['csv']}")
        if exp.get("path_root"):
            print(f"  PATH_ROOT: {exp['path_root']}")
        print(f"  data_result: {exp_data_result}")
        print(f"{'='*60}")
        process_single_csv(
            exp["csv"],
            exp["videos"],
            exp_data_result,
            exp_temp,
            failed_clips,
            used_names,
            dir_rel_path=exp.get("rel_path"),
            category_limits=category_limits,
            category_counters=category_counters,
            track_model=track_model if args.yolo_pose_model else None,
            yolo_meta_path_override=yolo_meta_path if args.yolo_pose_model else None,
        )

    # 4. Resumen de clips fallidos
    if failed_clips:
        print("\n" + "=" * 60)
        print("RESUMEN DE CLIPS CON ERRORES")
        print("=" * 60)
        for fc in failed_clips:
            print(f"  CSV: {fc['csv']} | Fila: {fc['row']} | Video: {fc['video']}")
            print(f"    Error: {fc['error']}")
        print(f"\nTotal clips con error: {len(failed_clips)}")
    else:
        print("\nTodos los clips se procesaron correctamente.")

    print(f"\n[FIN] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Proceso terminado.")
    if log_file:
        if original_stdout is not None:
            sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    main()