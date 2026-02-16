import os
import re
import json
import numpy as np
import cv2
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIGURACIÓN DE RUTAS (edita config.py) ---
from config import get_experiments

# --- PARÁMETROS DE CONTROL ---
DEBUG_MODE = True      # True: Solo procesa N vídeos
N_DEBUG = 2            # Número de vídeos en modo debug
MODEL_PATH = 'yolo26m-pose.pt'   # m= rápido+preciso; s/n= más rápido; l/x= más preciso
DELETE_TEMP_VIDEOS = False       # Si True, borra los vídeos temporales al terminar

# --- FILTROS DE CALIDAD ---
MIN_KP_CONF = 0.5
RELIABILITY_THR = 0.9                  # Ratio mínimo de frames válidos para guardar (90%)
KEEP_KPS = [5, 6, 7, 8, 9, 10, 11, 12]  # Hombros, codos, muñecas, cadera (sin piernas)
CRITICAL_KPS = [7, 8, 9, 10]           # Muñecas y codos

# --- FILTRO POR COBERTURA EN TIEMPO ---
MIN_COVERAGE_RATIO = 0.8               # Mínimo % de la duración con buena calidad (80%)
DEFAULT_FPS = 12                       # FPS por defecto si no se puede leer del vídeo

# Umbral de confianza para considerar un punto "ocluso"
OCCLUSION_CONF_THR = 0.3              # keypoints con conf < esto se consideran ocluidos


# Carga del modelo (se hace una sola vez)
model = YOLO(MODEL_PATH)


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

def process_single_csv(CSV_PATH, VIDEOS_DIR_RES, OUTPUT_DIR_RES, TEMP_CLIPS_RES):
    """Procesa un único CSV. Rutas ya resueltas."""
    # 1. Detectar fila de inicio (primera con HH:MM:SS en col 2) y leer CSV
    start_row = find_start_row(CSV_PATH)
    print(f"Inicio de datos en fila CSV: {start_row}")
    print(f"Vídeos: {VIDEOS_DIR_RES} | Salida: {OUTPUT_DIR_RES} | Temp: {TEMP_CLIPS_RES}")
    # header=None: la primera fila leída es dato, no encabezado (evita saltarnos una fila)
    df = pd.read_csv(CSV_PATH, skiprows=range(0, start_row - 1), header=None)
    
    if DEBUG_MODE:
        df = df.head(N_DEBUG)

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
            print(f"Archivo no encontrado: {video_full_path}")
            continue

        # 2. Crear Clip Temporal: temp_<fila_csv>_<categoria>.avi
        Path(TEMP_CLIPS_RES).mkdir(exist_ok=True)
        clip_path = os.path.join(TEMP_CLIPS_RES, f"temp_{fila_csv}_{category}.avi")

        print(f"[Clip] Fila CSV: {fila_csv} | Inicio: {t_start} | Fin: {t_end} | Video: {video_rel_path}")
        cut_clip(video_full_path, t_start, t_end, clip_path)

        # 3. Procesar Pose Tracking en el clip (stream=True reduce uso de memoria)
        results = model.track(clip_path, persist=True, tracker="bytetrack.yaml", verbose=False, stream=True)
        
        temp_person_data = {}
        cap = cv2.VideoCapture(clip_path)

        # FPS del clip (si falla, usamos DEFAULT_FPS)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = DEFAULT_FPS

        # Frames mínimos "válidos" exigidos según duración*MIN_COVERAGE_RATIO
        min_valid_seconds = clip_duration * MIN_COVERAGE_RATIO
        min_valid_frames = int(min_valid_seconds * fps) if clip_duration > 0 else 0

        for r in results:
            success, frame = cap.read()
            if not success or r.keypoints is None or r.boxes.id is None: continue

            ids = r.boxes.id.int().cpu().tolist()
            kpts = r.keypoints.xyn.cpu().numpy()
            confs = r.keypoints.conf.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()

            for i, track_id in enumerate(ids):
                if track_id not in temp_person_data:
                    c_t, c_b = get_color_attributes(frame, boxes[i])
                    temp_person_data[track_id] = {
                        'poses_full': [],
                        'poses': [],
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

                # poses: solo frames válidos
                if valid:
                    temp_person_data[track_id]['poses'].append(kpt_pose)
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

        # 4. Guardar: poses_full.npy siempre; poses.npy solo si pasa filtros
        video_name = Path(video_rel_path).stem
        for tid, info in temp_person_data.items():
            total_frames = info['total']
            valid_frames = info['v_cnt']

            if total_frames == 0:
                continue

            rel = valid_frames / total_frames
            valid_pct = round(rel * 100, 1)  # % frames válidos respecto al total

            passes_filters = rel >= RELIABILITY_THR and valid_frames >= min_valid_frames

            # Metadatos enriquecidos
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

            p_dir = Path(OUTPUT_DIR_RES) / category / f"{video_name}_idx{fila_csv}_p{tid}"
            p_dir.mkdir(parents=True, exist_ok=True)

            # poses_full.npy: siempre (todo el tracking)
            np.save(str(p_dir / "poses_full.npy"), np.array(info['poses_full']))

            # poses.npy: siempre, con solo los frames válidos (filtrados)
            np.save(str(p_dir / "poses.npy"), np.array(info['poses']))

            # meta.json: siempre (incluye valid_pct y metadatos enriquecidos)
            coverage_seconds = valid_frames / fps if fps > 0 else 0.0
            meta = {
                "video_source": video_rel_path,
                "row_csv": int(fila_csv),
                "t_start": str(t_start),
                "t_end": str(t_end),
                "valid_pct": valid_pct,
                "rel": round(rel, 2),
                "valid_frames": int(valid_frames),
                "total_frames": int(total_frames),
                "poses_full_count": len(info['poses_full']),
                "poses_filtered_count": len(info['poses']),
                "passes_filters": passes_filters,
                "min_valid_frames": int(min_valid_frames),
                "coverage_seconds": round(coverage_seconds, 2),
                "clip_duration": clip_duration,
                "fps": fps,
                "keypoint_confidence_avg": round(kp_conf_avg, 3),
                "occlusion_ratio": occlusion_ratio,
                "bbox_aspect_ratio": bbox_aspect_ratio,
                "subject_velocity": subject_velocity,
                "clothes": info['clothes'],
                "cat": category,
            }
            with open(p_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=4)

            if not passes_filters:
                print(
                    f"[SIN poses.npy] Clip '{video_name}' fila {fila_csv} track {tid} | "
                    f"valid_pct={valid_pct}% | valid_frames={valid_frames}, min={min_valid_frames}"
                )

        # Limpiar clip temporal (opcional)
        if DELETE_TEMP_VIDEOS and os.path.exists(clip_path):
            os.remove(clip_path)


def main():
    experiments = get_experiments()
    if not experiments:
        print("Error: no se encontraron CSV. Configura PATH_ROOT o CSV_PATH en config.py")
        return

    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"[Experimento {i+1}/{len(experiments)}] CSV: {exp['csv']}")
        print(f"{'='*60}")
        process_single_csv(
            exp["csv"],
            exp["videos"],
            exp["output"],
            exp["temp"],
        )


if __name__ == "__main__":
    main()