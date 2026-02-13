import os
import json
import numpy as np
import cv2
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIGURACIÓN DE RUTAS ---
CSV_PATH = "/home/debian/Vídeos/ScriptPrueba/clips.csv"        # Ruta de tu archivo CSV
BASE_DIR = os.path.dirname(CSV_PATH)   # Directorio del CSV (base para vídeos y salidas)
VIDEOS_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_final_limpio")
TEMP_CLIPS = os.path.join(BASE_DIR, "temp_clips")

# --- PARÁMETROS DE CONTROL ---
DEBUG_MODE = True      # True: Solo procesa N vídeos
N_DEBUG = 3            # Número de vídeos en modo debug
SKIP_ROWS = 20          # Filas a saltar en el CSV (metadatos)
MODEL_PATH = 'yolo26m-pose.pt'   # m= rápido+preciso; s/n= más rápido; l/x= más preciso

# --- FILTROS DE CALIDAD ---
MIN_KP_CONF = 0.5
RELIABILITY_THR = 0.9
KEEP_KPS = [5, 6, 7, 8, 9, 10, 11, 12] # Hombros, codos, muñecas, cadera
CRITICAL_KPS = [7, 8, 9, 10]           # Muñecas y codos

# Carga del modelo (se hace una sola vez)
model = YOLO(MODEL_PATH)

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

def main():
    # 1. Leer CSV saltando filas de metadatos
    # Asumimos columnas: 0:ruta_video, 1:inicio, 2:fin, 3:categoria
    df = pd.read_csv(CSV_PATH, skiprows=SKIP_ROWS)
    
    if DEBUG_MODE:
        df = df.head(N_DEBUG)

    clip_id = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Procesando CSV"):
        video_rel_path = row.iloc[0]  # Nombre del vídeo en la primera columna
        t_start = row.iloc[1]        # HH:MM:SS
        t_end = row.iloc[2]          # HH:MM:SS
        category = str(int(row.iloc[3]))
        
        video_full_path = os.path.join(VIDEOS_DIR, video_rel_path)
        if not os.path.exists(video_full_path):
            print(f"Archivo no encontrado: {video_full_path}")
            continue

        # 2. Crear Clip Temporal (temp_<id_clip>_<fila_csv>_<categoria>.avi)
        Path(TEMP_CLIPS).mkdir(exist_ok=True)
        clip_path = os.path.join(TEMP_CLIPS, f"temp_{clip_id}_{index}_{category}.avi")
        clip_id += 1
        cut_clip(video_full_path, t_start, t_end, clip_path)

        # 3. Procesar Pose Tracking en el clip
        results = model.track(clip_path, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        temp_person_data = {}
        cap = cv2.VideoCapture(clip_path)

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
                    temp_person_data[track_id] = {'poses': [], 'v_cnt': 0, 'total': 0, 'clothes': {'top': c_t, 'bottom': c_b}}
                
                # Filtro de confianza en puntos críticos
                valid = all(confs[i][idx] > MIN_KP_CONF for idx in CRITICAL_KPS)
                if valid:
                    temp_person_data[track_id]['poses'].append(kpts[i][KEEP_KPS])
                    temp_person_data[track_id]['v_cnt'] += 1
                temp_person_data[track_id]['total'] += 1
        
        cap.release()

        # 4. Guardar si cumple fiabilidad del 90%
        video_name = Path(video_rel_path).stem
        for tid, info in temp_person_data.items():
            if info['total'] > 0 and (info['v_cnt'] / info['total']) >= RELIABILITY_THR:
                p_dir = Path(OUTPUT_DIR) / category / f"{video_name}_idx{index}_p{tid}"
                p_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(p_dir / "poses.npy"), np.array(info['poses']))
                
                with open(p_dir / "meta.json", "w") as f:
                    meta = {"rel": round(info['v_cnt']/info['total'], 2), "clothes": info['clothes'], "cat": category}
                    json.dump(meta, f, indent=4)

        # Limpiar clip temporal (opcional)
        if os.path.exists(clip_path): os.remove(clip_path)

if __name__ == "__main__":
    main()