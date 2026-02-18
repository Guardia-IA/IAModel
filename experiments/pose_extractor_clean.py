import os
import re
import json
import sys
from datetime import datetime
from pathlib import Path
from itertools import zip_longest

import numpy as np
import cv2
import pandas as pd
import subprocess
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIGURACIÓN DE RUTAS (edita config.py) ---
from config import get_experiments, PATH_ROOT, CSV_PATH, OUTPUT_BASE, LOGS_SUBDIR
from security import validate_folder

# Base de salida: OUTPUT_BASE de config (temp_clips/ y data_result/ dentro)
OUTPUT = Path(OUTPUT_BASE) if OUTPUT_BASE else Path(__file__).parent / "output"

# --- PARÁMETROS DE CONTROL ---
DEBUG_MODE = True      # True: Solo procesa N vídeos
N_DEBUG = 5            # Número de vídeos en modo debug
MODEL_PATH = 'yolo11x-pose.pt'   # m= rápido+preciso; s/n= más rápido; l/x= más preciso
DELETE_TEMP_VIDEOS = False       # Si True, borra los vídeos temporales al terminar

# --- FILTROS DE CALIDAD ---
MIN_KP_CONF = 0.5
RELIABILITY_THR = 0.9                  # Ratio mínimo de frames válidos para guardar (90%)
KEEP_KPS = [5, 6, 7, 8, 9, 10, 11, 12]  # Hombros, codos, muñecas, cadera (sin piernas)
CRITICAL_KPS = [7, 8, 9, 10]           # Muñecas y codos

# --- FILTRO POR COBERTURA EN TIEMPO ---
MIN_COVERAGE_RATIO = 0.8               # Mínimo % de la duración con buena calidad (80%)
DEFAULT_FPS = 12                       # FPS por defecto si no se puede leer del vídeo

# Filtro de duración mínima por usuario (en segundos)
MIN_USER_SECONDS = 2.0                 # Usuarios con menos de esto se ignoran

# Umbral de confianza para considerar un punto "ocluso"
OCCLUSION_CONF_THR = 0.3              # keypoints con conf < esto se consideran ocluidos


def _get_device() -> str:
    """Detecta GPU CUDA o usa CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _setup_logging(log_dir: str | Path | None = None) -> object | None:
    """Redirige stdout a terminal + fichero log<timestamp>.txt. Logs en OUTPUT/LOGS_SUBDIR (OUTPUT_BASE/logs)."""
    base = (OUTPUT / LOGS_SUBDIR) if OUTPUT_BASE else Path(__file__).parent / "logs"
    log_dir = Path(log_dir or base).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
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
                pass  # archivo cerrado u otro error de I/O

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

    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)
    return log_file, original_stdout


# Carga del modelo (se hace una sola vez)
DEVICE = _get_device()
print(f"Cargando modelo pose: {MODEL_PATH}")
model = YOLO(MODEL_PATH)  # Modelo pose para keypoints
print("Cargando modelo de segmentación: yolo11x-seg.pt (para extraer siluetas)")
model_seg = YOLO('yolo11x-seg.pt')  # Modelo seg para máscaras
print("✓ Modelos cargados correctamente")


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
    base = f"{path_norm}_{t1}_{t2}_{category}"

    name = base
    n = 2
    while name in used_names:
        name = f"{base}_{n}"
        n += 1
    used_names.add(name)
    return name


def cut_clip(video_in, start, end, video_out):
    """Recorta el vídeo usando FFmpeg (sin re-codificar, solo copia de codecs)."""
    command = [
        'ffmpeg', '-y', '-ss', str(start), '-to', str(end),
        '-i', video_in, '-c', 'copy', '-loglevel', 'error', video_out
    ]
    subprocess.run(command)

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

def process_single_csv(
    CSV_PATH,
    VIDEOS_DIR_RES,
    DATA_RESULT_BASE,
    TEMP_CLIPS_BASE,
    failed_clips: list,
    used_clip_names: set,
    dir_rel_path: str | None = None,
):
    """Procesa un único CSV. Errores por clip se añaden a failed_clips."""
    # 1. Detectar fila de inicio (primera con HH:MM:SS en col 2) y leer CSV
    start_row = find_start_row(CSV_PATH)
    print(f"Inicio de datos en fila CSV: {start_row}")
    print(f"Vídeos: {VIDEOS_DIR_RES} | Temp clips: {TEMP_CLIPS_BASE} | Data: {DATA_RESULT_BASE}")
    # header=None: la primera fila leída es dato, no encabezado (evita saltarnos una fila)
    df = pd.read_csv(CSV_PATH, skiprows=range(0, start_row - 1), header=None)
    
    if DEBUG_MODE:
        df = df.head(N_DEBUG)  # Limitar filas (clips) de este CSV

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Procesando CSV"):
        video_rel_path = row.iloc[0]  # Nombre del vídeo en la primera columna
        t_start = row.iloc[1]        # HH:MM:SS
        t_end = row.iloc[2]          # HH:MM:SS
        category = str(int(row.iloc[3]))

        # fila_csv = número de fila en el CSV
        fila_csv = start_row + int(index)

        # Duración teórica del clip según el CSV
        clip_duration = max(0.0, hms_to_seconds(t_end) - hms_to_seconds(t_start))
        
        video_full_path = os.path.join(VIDEOS_DIR_RES, video_rel_path)
        if not os.path.exists(video_full_path):
            failed_clips.append({
                "csv": str(CSV_PATH), "row": fila_csv, "video": video_rel_path,
                "error": "Archivo no encontrado",
            })
            continue

        cap = None
        clip_path = None
        try:
            # 2. Nombre único del clip (incluye ruta dir: fecha_cam) y ruta temporal
            clip_name = make_clip_name(
                video_rel_path, t_start, t_end, category, used_clip_names, dir_rel_path
            )
            temp_cat_dir = Path(TEMP_CLIPS_BASE) / category
            temp_cat_dir.mkdir(parents=True, exist_ok=True)
            clip_path = str(temp_cat_dir / f"{clip_name}.avi")

            print(f"[Clip] {clip_name} | Fila CSV: {fila_csv} | Inicio: {t_start} | Fin: {t_end} | Video: {video_rel_path}")
            cut_clip(video_full_path, t_start, t_end, clip_path)

            # 3. Procesar Pose Tracking en el clip (stream=True reduce uso de memoria)
            results = model.track(
                source=clip_path,
                tracker="custom_tracker.yaml",
                persist=True,
                verbose=False,
                stream=True,
                device=DEVICE,
            )
            # Procesar segmentación en paralelo para obtener siluetas
            results_seg = model_seg.track(
                source=clip_path,
                tracker="custom_tracker.yaml",
                persist=True,
                verbose=False,
                stream=True,
                device=DEVICE,
            )
            temp_person_data = {}
            cap = cv2.VideoCapture(clip_path)

            # FPS del clip (si falla, usamos DEFAULT_FPS)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = DEFAULT_FPS

            # Frames mínimos "válidos" exigidos según duración*MIN_COVERAGE_RATIO
            min_valid_seconds = clip_duration * MIN_COVERAGE_RATIO
            min_valid_frames = int(min_valid_seconds * fps) if clip_duration > 0 else 0

            # Sincronizar iteradores de pose y segmentación frame por frame
            for r, r_seg in zip_longest(results, results_seg, fillvalue=None):
                success, frame = cap.read()
                if not success:
                    continue
                    
                # Si no hay resultados de pose, saltar este frame
                if r is None or r.keypoints is None or r.boxes.id is None:
                    continue

                ids = r.boxes.id.int().cpu().tolist()
                kpts = r.keypoints.xyn.cpu().numpy()
                confs = r.keypoints.conf.cpu().numpy()
                boxes = r.boxes.xyxy.cpu().numpy()
                
                # Obtener dimensiones del frame
                h_frame, w_frame = frame.shape[:2]
                
                # Obtener máscaras de segmentación si están disponibles
                seg_masks = {}
                seg_boxes = {}
                
                if r_seg is not None and r_seg.masks is not None and r_seg.boxes.id is not None:
                    seg_ids = r_seg.boxes.id.int().cpu().tolist()
                    seg_boxes_seg = r_seg.boxes.xyxy.cpu().numpy()
                    
                    for seg_i, seg_track_id in enumerate(seg_ids):
                        if seg_i < len(r_seg.masks.data):
                            try:
                                # Obtener máscara de YOLO (puede venir como tensor o array)
                                mask_tensor = r_seg.masks.data[seg_i]
                                if hasattr(mask_tensor, 'cpu'):
                                    mask_np = mask_tensor.cpu().numpy()
                                else:
                                    mask_np = np.array(mask_tensor)
                                
                                # Las máscaras de YOLO pueden venir en diferentes formatos
                                # Si es un array 2D, ya está en formato de imagen
                                # Si tiene más dimensiones, tomar la primera
                                if len(mask_np.shape) > 2:
                                    mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np
                                
                                # Redimensionar al tamaño del frame si es necesario
                                if mask_np.shape != (h_frame, w_frame):
                                    mask_np = cv2.resize(mask_np, (w_frame, h_frame), interpolation=cv2.INTER_NEAREST)
                                
                                # Convertir a binario (0 o 255)
                                mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                                seg_masks[seg_track_id] = mask_binary
                                seg_boxes[seg_track_id] = seg_boxes_seg[seg_i]
                            except Exception as e:
                                # Si hay error extrayendo la máscara, continuar sin ella
                                print(f"[WARNING] Error extrayendo máscara para track {seg_track_id}: {e}")
                                continue

                for i, track_id in enumerate(ids):
                    if track_id not in temp_person_data:
                        c_t, c_b = get_color_attributes(frame, boxes[i])
                        temp_person_data[track_id] = {
                            'poses_full': [],
                            'poses': [],
                            'masks_full': [],  # Máscaras de todos los frames
                            'masks': [],       # Máscaras solo de frames válidos
                            'mask_areas': [],  # Áreas de máscaras para análisis volumétrico
                            'v_cnt': 0, 'total': 0, 'clothes': {'top': c_t, 'bottom': c_b},
                            'kp_conf_sum': 0.0, 'occluded_frames': 0, 'bbox_ratios': []
                        }

                    # Filtro de confianza en puntos críticos (muñecas, codos)
                    valid = all(confs[i][idx] > MIN_KP_CONF for idx in CRITICAL_KPS)
                    kpt_pose = kpts[i][KEEP_KPS]
                    conf_kp = confs[i][KEEP_KPS]
                    bbox = boxes[i]

                    # poses_full: todo el tracking
                    temp_person_data[track_id]['poses_full'].append(kpt_pose)

                    # Extraer máscara de segmentación (silueta) para este usuario
                    mask_frame = None
                    if track_id in seg_masks:
                        # Usar máscara directamente si el track_id coincide
                        mask_frame = seg_masks[track_id]
                    else:
                        # Si no hay coincidencia exacta, buscar por IoU de bboxes
                        best_iou = 0.0
                        best_seg_id = None
                        for seg_id, seg_box in seg_boxes.items():
                            # Calcular IoU entre bboxes
                            x1_i = max(bbox[0], seg_box[0])
                            y1_i = max(bbox[1], seg_box[1])
                            x2_i = min(bbox[2], seg_box[2])
                            y2_i = min(bbox[3], seg_box[3])
                            if x2_i > x1_i and y2_i > y1_i:
                                inter_area = (x2_i - x1_i) * (y2_i - y1_i)
                                box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                seg_area = (seg_box[2] - seg_box[0]) * (seg_box[3] - seg_box[1])
                                union_area = box_area + seg_area - inter_area
                                iou = inter_area / union_area if union_area > 0 else 0.0
                                if iou > best_iou and iou > 0.5:  # Umbral mínimo de IoU
                                    best_iou = iou
                                    best_seg_id = seg_id
                        if best_seg_id is not None:
                            mask_frame = seg_masks[best_seg_id]
                    
                    # Si no se encontró máscara, crear una máscara vacía
                    if mask_frame is None:
                        mask_frame = np.zeros((h_frame, w_frame), dtype=np.uint8)
                    
                    # Calcular área de la máscara (píxeles del cuerpo)
                    mask_area = int(np.sum(mask_frame > 0))
                    temp_person_data[track_id]['mask_areas'].append(mask_area)
                    
                    # masks_full: todas las máscaras
                    temp_person_data[track_id]['masks_full'].append(mask_frame)

                    # poses: solo frames válidos
                    if valid:
                        temp_person_data[track_id]['poses'].append(kpt_pose)
                        temp_person_data[track_id]['masks'].append(mask_frame)  # Máscara solo si frame válido
                        temp_person_data[track_id]['v_cnt'] += 1
                    temp_person_data[track_id]['total'] += 1

                    # Metadatos enriquecidos
                    temp_person_data[track_id]['kp_conf_sum'] += float(np.mean(conf_kp))
                    if any(confs[i][idx] < OCCLUSION_CONF_THR for idx in CRITICAL_KPS):
                        temp_person_data[track_id]['occluded_frames'] += 1
                    h_bbox = bbox[3] - bbox[1]
                    w_bbox = max(bbox[2] - bbox[0], 1e-6)
                    temp_person_data[track_id]['bbox_ratios'].append(float(h_bbox / w_bbox))

            cap.release()
            cap = None

            # 4. Guardar: data_result/{cat}/{clip_name}/ con meta.json y user_X/
            if not temp_person_data:
                failed_clips.append({
                    "csv": str(CSV_PATH), "row": fila_csv, "video": video_rel_path,
                    "error": "No se detectaron personas",
                })
                continue

            data_dir = Path(DATA_RESULT_BASE) / category / clip_name
            data_dir.mkdir(parents=True, exist_ok=True)

            users_meta = []
            for tid, info in temp_person_data.items():
                total_frames = info['total']
                valid_frames = info['v_cnt']

                if total_frames == 0:
                    continue

                # Duración del usuario en segundos (para filtrar personas de paso rápido / tracks rotos)
                user_seconds = total_frames / fps if fps > 0 else 0.0
                if user_seconds < MIN_USER_SECONDS:
                    print(
                        f"[DESCARTADO usuario] Clip '{clip_name}' user_{tid} | "
                        f"duración={user_seconds:.2f}s < {MIN_USER_SECONDS}s"
                    )
                    continue

                rel = valid_frames / total_frames
                valid_pct = round(rel * 100, 1)

                passes_filters = rel >= RELIABILITY_THR and valid_frames >= min_valid_frames

                kp_conf_avg = info['kp_conf_sum'] / total_frames if total_frames > 0 else 0.0
                occlusion_ratio = round(info['occluded_frames'] / total_frames * 100, 1) if total_frames > 0 else 0.0
                bbox_aspect_ratio = round(float(np.mean(info['bbox_ratios'])), 3) if info['bbox_ratios'] else 0.0
                poses_full_arr = np.array(info['poses_full'])
                if len(poses_full_arr) > 1:
                    d = np.diff(poses_full_arr, axis=0)
                    velocity_mag = np.sqrt((d ** 2).sum(axis=-1))
                    subject_velocity = round(float(np.mean(velocity_mag)), 5)
                else:
                    subject_velocity = 0.0

                # Métricas de silueta (máscara)
                mask_areas = info.get('mask_areas', [])
                mask_area_avg = round(float(np.mean(mask_areas)), 1) if mask_areas else 0.0
                mask_area_std = round(float(np.std(mask_areas)), 1) if len(mask_areas) > 1 else 0.0
                # Variación de área (detecta cambios volumétricos como objetos ocultos)
                mask_area_cv = round(mask_area_std / mask_area_avg * 100, 2) if mask_area_avg > 0 else 0.0

                coverage_seconds = valid_frames / fps if fps > 0 else 0.0
                user_meta = {
                    "track_id": int(tid),
                    "valid_pct": valid_pct,
                    "rel": round(rel, 2),
                    "valid_frames": int(valid_frames),
                    "total_frames": int(total_frames),
                    "poses_full_count": len(info['poses_full']),
                    "poses_filtered_count": len(info['poses']),
                    "masks_full_count": len(info.get('masks_full', [])),
                    "masks_filtered_count": len(info.get('masks', [])),
                    "passes_filters": passes_filters,
                    "keypoint_confidence_avg": round(kp_conf_avg, 3),
                    "occlusion_ratio": occlusion_ratio,
                    "bbox_aspect_ratio": bbox_aspect_ratio,
                    "subject_velocity": subject_velocity,
                    "mask_area_avg": mask_area_avg,
                    "mask_area_std": mask_area_std,
                    "mask_area_cv": mask_area_cv,  # Coeficiente de variación (detecta cambios volumétricos)
                    "clothes": info['clothes'],
                }
                users_meta.append(user_meta)

                # Carpetas user_X con poses y máscaras
                user_dir = data_dir / f"user_{tid}"
                user_dir.mkdir(exist_ok=True)
                np.save(str(user_dir / "poses_full.npy"), np.array(info['poses_full']))
                np.save(str(user_dir / "poses.npy"), np.array(info['poses']))
                # Guardar máscaras (siluetas) de segmentación
                if info.get('masks_full'):
                    np.save(str(user_dir / "masks_full.npy"), np.array(info['masks_full']))
                if info.get('masks'):
                    np.save(str(user_dir / "masks.npy"), np.array(info['masks']))

                if not passes_filters:
                    print(
                        f"[SIN filtros] Clip '{clip_name}' user_{tid} | "
                        f"valid_pct={valid_pct}% | valid_frames={valid_frames}, min={min_valid_frames}"
                    )

            # meta.json único por clip
            meta = {
                "clip_name": clip_name,
                "video_source": video_rel_path,
                "row_csv": int(fila_csv),
                "t_start": str(t_start),
                "t_end": str(t_end),
                "clip_duration": clip_duration,
                "fps": fps,
                "min_valid_frames": int(min_valid_frames),
                "cat": category,
                "users": users_meta,
            }
            with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)

            # Limpiar clip temporal (opcional)
            if DELETE_TEMP_VIDEOS and clip_path and os.path.exists(clip_path):
                os.remove(clip_path)

        except Exception as e:
            failed_clips.append({
                "csv": str(CSV_PATH), "row": fila_csv, "video": video_rel_path,
                "error": str(e),
            })
            print(f"[ERROR] Clip fila {fila_csv} ({video_rel_path}): {e}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass


def main():
    # 1. Log en fichero además del terminal (antes de validación para registrar todo)
    log_result = _setup_logging()
    log_file = original_stdout = None
    if log_result:
        log_file, original_stdout = log_result
        print(f"Log guardado en: {log_file.name}")

    # 2. Validación previa (security): comprobar que CSVs y vídeos existen, tiempos correctos
    path_to_validate = PATH_ROOT if PATH_ROOT else (os.path.dirname(os.path.abspath(CSV_PATH or ".")))
    validation = validate_folder(path_to_validate)
    if not validation.get("ok"):
        print("Abortando: hay errores en los CSVs. Corrígelos antes de ejecutar el extractor.")
        if log_file:
            if original_stdout is not None:
                sys.stdout = original_stdout
            log_file.close()
        return

    # 3. Device y ruta de salida
    print(f"Dispositivo: {DEVICE}")
    print(f"Salida: {OUTPUT} (temp_clips/ + data_result/)")

    experiments = get_experiments()
    if not experiments:
        print("Error: no se encontraron CSV. Configura PATH_ROOT o CSV_PATH en config.py")
        return

    if DEBUG_MODE:
        experiments = experiments[:1]  # Solo primer CSV
        print(f"[DEBUG] Procesando solo {N_DEBUG} clips del primer CSV")

    temp_clips_base = OUTPUT / "temp_clips"
    data_result_base = OUTPUT / "data_result"
    temp_clips_base.mkdir(parents=True, exist_ok=True)
    data_result_base.mkdir(parents=True, exist_ok=True)

    failed_clips = []
    used_clip_names = set()
    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"[Experimento {i+1}/{len(experiments)}] CSV: {exp['csv']}")
        print(f"{'='*60}")
        process_single_csv(
            exp["csv"],
            exp["videos"],
            str(data_result_base),
            str(temp_clips_base),
            failed_clips,
            used_clip_names,
            dir_rel_path=exp.get("rel_path"),
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

    if log_file:
        if original_stdout is not None:
            sys.stdout = original_stdout  # Restaurar antes de cerrar para evitar flush en archivo cerrado
        log_file.close()


if __name__ == "__main__":
    main()