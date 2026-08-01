"""
Configuración centralizada de rutas para pose_extractor_clean y build_clip_index.

ENTRADA (dónde están los datos):
  PATH_ROOTS = lista de carpetas con CSVs/vídeos (busca *.csv recursivamente en cada una).
  PATH_ROOT  = primer elemento de PATH_ROOTS (compatibilidad legacy).
  CSV_PATH   = solo si no hay PATH_ROOTS ni PATH_ROOT: ruta a un CSV concreto.

SALIDA:
  OUTPUT_BASE = carpeta base unificada. Dentro: temp_clips/, data_result/, logs/
  Si hay varios PATH_ROOTS y OUTPUT_BASE es None, cada raíz escribe en
  {PATH_ROOT}/DEFAULT_OUTPUT_SUBDIR/data_result (p. ej. datos_1/data_yolo26m/data_result).
"""
import os
from pathlib import Path
from typing import List, Optional

# --- RUTAS DE ENTRADA ---
PATH_ROOTS: List[str] = [
    "/media/8TB/DatosEntrenamiento/datos_2/csv_buenos",
    "/media/8TB/DatosEntrenamiento/datos_1",
    "/home/angel/videos"
]
PATH_ROOT = PATH_ROOTS[0] if PATH_ROOTS else None  # Legacy: primer root
CSV_PATH = None    # Usado solo si no hay PATH_ROOTS ni PATH_ROOT
VIDEOS_DIR = None  # Solo modo CSV_PATH: None = mismo dir que el CSV
OUTPUT_DIR = None  # Solo modo CSV_PATH: None = dataset_final_limpio en dir del CSV
TEMP_CLIPS = None  # Solo modo CSV_PATH: None = temp_clips en dir del CSV

# --- SALIDA ---
# OUTPUT_BASE = carpeta donde van temp_clips/ y data_result/ (recomendado con varios PATH_ROOTS)
OUTPUT_BASE = "/media/8TB/DatosEntrenamiento/data_clips/data_yolo26s_010826"
# Subcarpeta bajo cada PATH_ROOT cuando OUTPUT_BASE es None y hay varios PATH_ROOTS
DEFAULT_OUTPUT_SUBDIR = "data_yolo26s"
TEMP_BASE = None     # Legacy: solo para get_experiments si no usas OUTPUT_BASE unificado.

CLIP_META_PATH = None  # build_clip_index genera clip_meta.csv (índice de todos los meta.json).
                       # Si None, se crea en OUTPUT_BASE/clip_meta.csv o junto al primer output.

# Subcarpetas bajo OUTPUT_BASE
OUTPUT_SUBDIR = "dataset_final_limpio"  # Legacy: para get_experiments
TEMP_SUBDIR = "temp_clips"              # temp_clips/{0,1,2,...}/
LOGS_SUBDIR = "logs"                    # logs/ (ficheros log<timestamp>.txt)

# --- RECORTE DE CLIPS (FFmpeg) ---
CLIP_SCALE_HEIGHT = 1080   # Altura en píxeles. Ancho se calcula manteniendo aspect ratio (-2).

# Aceleración VAAPI (Intel iGPU / AMD). None = usar software (libx264).
VAAPI_DEVICE = None   # None = sin VAAPI

# --- MODELO YOLO POSE ---
YOLO_POSE_MODEL = "yolo26s-pose.pt"


def get_path_roots() -> List[Path]:
    """Raíces de entrada activas (PATH_ROOTS o PATH_ROOT legacy)."""
    raw: List[str] = []
    if PATH_ROOTS:
        raw.extend(str(p) for p in PATH_ROOTS if p)
    elif PATH_ROOT:
        raw.append(str(PATH_ROOT))
    out: List[Path] = []
    seen: set[str] = set()
    for item in raw:
        p = Path(item).expanduser().resolve()
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def output_base_for_path_root(path_root: Path) -> Optional[Path]:
    """
    Carpeta OUTPUT_BASE para una raíz de entrada.
    None = usar salida legacy del script (experiments/output) en extracción mono-root.
    """
    if OUTPUT_BASE:
        return Path(OUTPUT_BASE).expanduser().resolve()
    if len(get_path_roots()) > 1:
        return path_root / DEFAULT_OUTPUT_SUBDIR
    return None


def get_data_result_roots() -> List[Path]:
    """
    Carpetas data_result usadas por entrenamiento/evaluación.
    - OUTPUT_BASE definido → un único data_result merged.
    - Varios PATH_ROOTS sin OUTPUT_BASE → uno por raíz ({root}/data_yolo26m/data_result).
    """
    out: List[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        rp = p.expanduser().resolve()
        key = str(rp)
        if key not in seen:
            seen.add(key)
            out.append(rp)

    if OUTPUT_BASE:
        _add(Path(OUTPUT_BASE) / "data_result")
        return out

    path_roots = get_path_roots()
    if len(path_roots) > 1:
        for pr in path_roots:
            _add(pr / DEFAULT_OUTPUT_SUBDIR / "data_result")
    return out


def data_result_tag(data_result_root: Path) -> str:
    """Etiqueta estable para UIDs cuando hay varios data_result."""
    dr = data_result_root.expanduser().resolve()
    for pr in get_path_roots():
        ob = output_base_for_path_root(pr)
        if ob is not None and (ob / "data_result").resolve() == dr:
            return pr.name
    return dr.parent.name if dr.parent.name else dr.name


def _find_csv_files(root: str):
    """Busca todos los archivos .csv bajo root de forma recursiva."""
    root_path = Path(root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        return []
    return sorted(root_path.rglob("*.csv"))


def get_experiments():
    """
    Devuelve lista de experimentos a procesar (todos los PATH_ROOTS).
    Cada elemento incluye csv, videos, rel_path, data_result_dir, temp_clips_dir, path_root.
    """
    path_roots = get_path_roots()
    if path_roots:
        experiments = []
        multi_root = len(path_roots) > 1
        for root_path in path_roots:
            csv_files = _find_csv_files(str(root_path))
            root_tag = root_path.name
            ob = output_base_for_path_root(root_path)
            for csv_path in csv_files:
                base = csv_path.parent
                try:
                    rel = base.relative_to(root_path)
                except ValueError:
                    rel = base.name
                rel_str = str(rel) if rel != Path(".") else base.name
                if multi_root:
                    rel_str = f"{root_tag}/{rel_str}"

                if ob is not None:
                    temp = str(ob / TEMP_SUBDIR / rel_str) if not TEMP_BASE else str(Path(TEMP_BASE) / rel_str / TEMP_SUBDIR)
                    data_result_dir = str(ob / "data_result")
                    temp_clips_dir = str(ob / TEMP_SUBDIR)
                    output = str(ob / rel_str / OUTPUT_SUBDIR)
                else:
                    if TEMP_BASE:
                        temp = str(Path(TEMP_BASE) / rel_str / TEMP_SUBDIR)
                    else:
                        temp = str(base / TEMP_SUBDIR)
                    output = str(base / OUTPUT_SUBDIR)
                    data_result_dir = None
                    temp_clips_dir = None

                experiments.append({
                    "csv": str(csv_path),
                    "videos": str(base),
                    "output": output,
                    "temp": temp,
                    "rel_path": rel_str,
                    "path_root": str(root_path),
                    "path_root_tag": root_tag,
                    "data_result_dir": data_result_dir,
                    "temp_clips_dir": temp_clips_dir,
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
    return [{
        "csv": CSV_PATH,
        "videos": videos,
        "output": out,
        "temp": temp,
        "rel_path": str(rel),
        "path_root": base,
        "path_root_tag": rel,
        "data_result_dir": None,
        "temp_clips_dir": None,
    }]


def resolve_paths():
    """Resuelve rutas para build_clip_index."""
    data_results = get_data_result_roots()
    if data_results:
        all_outputs = [str(p) for p in data_results]
        data_result = str(data_results[0])
    elif OUTPUT_BASE:
        data_result = str(Path(OUTPUT_BASE) / "data_result")
        all_outputs = [data_result]
    else:
        data_result = None
        all_outputs = []

    if CLIP_META_PATH:
        clip_meta = CLIP_META_PATH
    elif OUTPUT_BASE:
        clip_meta = str(Path(OUTPUT_BASE) / "clip_meta.csv")
    else:
        exps = get_experiments()
        out = exps[0]["output"] if exps else os.path.join(os.path.dirname(CSV_PATH or "."), OUTPUT_SUBDIR)
        clip_meta = os.path.join(out, "clip_meta.csv")

    exps = get_experiments()
    if not all_outputs:
        all_outputs = [data_result] if data_result else [e["output"] for e in exps if e.get("output")]

    return {
        "csv": exps[0]["csv"] if exps else None,
        "videos": exps[0]["videos"] if exps else None,
        "output": data_result or (exps[0]["output"] if exps else None),
        "temp": exps[0]["temp"] if exps else None,
        "clip_meta": clip_meta,
        "all_outputs": all_outputs,
    }
