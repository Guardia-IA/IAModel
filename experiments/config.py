"""
Configuración centralizada de rutas para pose_extractor_clean y build_clip_index.

ENTRADA (dónde están los datos):
  PATH_ROOT = carpeta con tus datos (12Diciembre, 15Diciembre, cam1, etc.). Busca *.csv recursivamente.
  CSV_PATH  = solo si PATH_ROOT es None: ruta a un CSV concreto.

SALIDA: OUTPUT_BASE = carpeta base. Dentro: temp_clips/, data_result/, logs/
"""
import os
from pathlib import Path

# --- RUTAS ---
PATH_ROOT = "/home/debian/Proyectos/GuardIA/Grabaciones/Videos/"  # Carpeta raíz: busca *.csv recursivamente. None = modo CSV_PATH
CSV_PATH = None    # Usado solo si PATH_ROOT es None
VIDEOS_DIR = None  # Solo modo CSV_PATH: None = mismo dir que el CSV
OUTPUT_DIR = None  # Solo modo CSV_PATH: None = dataset_final_limpio en dir del CSV
TEMP_CLIPS = None  # Solo modo CSV_PATH: None = temp_clips en dir del CSV

# --- SALIDA ---
# OUTPUT_BASE = carpeta donde van temp_clips/ y data_result/
#   temp_clips/{0,1,2,...}/ = clips por categoría
#   data_result/{0,1,2,...}/{clip_name}/ = meta.json + user_X/
OUTPUT_BASE = "/home/debian/Proyectos/GuardIA/Grabaciones/output_test_11_n_v3/"   # None = output junto al script
TEMP_BASE = None     # Legacy: solo para get_experiments si no usas OUTPUT_BASE unificado.

CLIP_META_PATH = None  # build_clip_index genera clip_meta.csv (índice de todos los meta.json).
                       # Si None, se crea en OUTPUT_BASE/clip_meta.csv o junto al primer output.

# Subcarpetas bajo OUTPUT_BASE
OUTPUT_SUBDIR = "dataset_final_limpio"  # Legacy: para get_experiments
TEMP_SUBDIR = "temp_clips"              # temp_clips/{0,1,2,...}/
LOGS_SUBDIR = "logs"                    # logs/ (ficheros log<timestamp>.txt)

# --- RECORTE DE CLIPS (FFmpeg) ---
# Resolución objetivo al extraer clips (reduce tiempo de YOLO). None = mantener original.
CLIP_SCALE_HEIGHT = 1080   # Altura en píxeles. Ancho se calcula manteniendo aspect ratio (-2).

# Aceleración VAAPI (Intel iGPU / AMD). None = usar software (libx264).
# Recomendado: usar la gráfica INTEGRADA (/dev/dri/renderD128 suele ser Intel)
# para no competir con la GPU NVIDIA donde corre YOLO.
VAAPI_DEVICE = "/dev/dri/renderD128"   # None = sin VAAPI


def _find_csv_files(root: str):
    """Busca todos los archivos .csv bajo root de forma recursiva."""
    root_path = Path(root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        return []
    return sorted(root_path.rglob("*.csv"))


def get_experiments():
    """
    Devuelve lista de experimentos a procesar.
    Cada elemento: {"csv": path, "videos": path, "output": path, "temp": path}
    """
    if PATH_ROOT:
        csv_files = _find_csv_files(PATH_ROOT)
        if not csv_files:
            return []
        root_path = Path(PATH_ROOT).resolve()
        experiments = []
        for csv_path in csv_files:
            base = csv_path.parent
            try:
                rel = base.relative_to(root_path)
            except ValueError:
                rel = base.name
            rel_str = str(rel) if rel != Path(".") else base.name

            if OUTPUT_BASE:
                output = str(Path(OUTPUT_BASE) / rel_str / OUTPUT_SUBDIR)
            else:
                output = str(base / OUTPUT_SUBDIR)

            if TEMP_BASE:
                temp = str(Path(TEMP_BASE) / rel_str / TEMP_SUBDIR)
            else:
                temp = str(base / TEMP_SUBDIR)

            experiments.append({
                "csv": str(csv_path),
                "videos": str(base),
                "output": output,
                "temp": temp,
                "rel_path": rel_str,  # ruta desde PATH_ROOT (ej. 12Diciembre/cam1)
            })
        return experiments

    # Modo CSV_PATH
    if not CSV_PATH or not os.path.isfile(CSV_PATH):
        return []
    base = os.path.dirname(os.path.abspath(CSV_PATH))
    out = OUTPUT_DIR if OUTPUT_DIR else os.path.join(base, OUTPUT_SUBDIR)
    temp = TEMP_CLIPS if TEMP_CLIPS else os.path.join(base, TEMP_SUBDIR)
    videos = VIDEOS_DIR if VIDEOS_DIR else base
    rel = Path(base).name if base else "."
    return [{"csv": CSV_PATH, "videos": videos, "output": out, "temp": temp, "rel_path": str(rel)}]


def resolve_paths():
    """Resuelve rutas para build_clip_index. data_result = OUTPUT_BASE/data_result."""
    data_result = str(Path(OUTPUT_BASE) / "data_result") if OUTPUT_BASE else None
    if CLIP_META_PATH:
        clip_meta = CLIP_META_PATH
    elif OUTPUT_BASE:
        clip_meta = str(Path(OUTPUT_BASE) / "clip_meta.csv")
    else:
        exps = get_experiments()
        out = exps[0]["output"] if exps else os.path.join(os.path.dirname(CSV_PATH or "."), OUTPUT_SUBDIR)
        clip_meta = os.path.join(out, "clip_meta.csv")
    exps = get_experiments()
    return {
        "csv": exps[0]["csv"] if exps else None,
        "videos": exps[0]["videos"] if exps else None,
        "output": data_result or (exps[0]["output"] if exps else None),
        "temp": exps[0]["temp"] if exps else None,
        "clip_meta": clip_meta,
        "all_outputs": [data_result] if data_result else [e["output"] for e in exps],
    }
