"""
Configuración centralizada de rutas para pose_extractor_clean y build_clip_index.

Modo 1 - PATH_ROOT (recomendado para experimentos):
  PATH_ROOT = "/ruta/carpeta"  → busca recursivamente *.csv dentro, procesa cada uno.
  Donde encuentre un CSV, los vídeos estarán en el mismo directorio.
  Salida/temp: si OUTPUT_BASE y TEMP_BASE están definidos, van ahí (otra unidad).
               Si None, se crean junto al CSV (dataset_final_limpio, temp_clips).

Modo 2 - CSV_PATH (modo simple):
  PATH_ROOT = None y CSV_PATH = "/ruta/clips.csv"  → procesa solo ese CSV.
  VIDEOS_DIR, OUTPUT_DIR, TEMP_CLIPS: None = relativo al directorio del CSV.
"""
import os
from pathlib import Path

# --- RUTAS ---
PATH_ROOT = "/home/debian/Vídeos/ScriptPrueba"  # Carpeta raíz: busca *.csv recursivamente. None = modo CSV_PATH
CSV_PATH = None    # Usado solo si PATH_ROOT es None
VIDEOS_DIR = None  # Solo modo CSV_PATH: None = mismo dir que el CSV
OUTPUT_DIR = None  # Solo modo CSV_PATH: None = dataset_final_limpio en dir del CSV
TEMP_CLIPS = None  # Solo modo CSV_PATH: None = temp_clips en dir del CSV

# --- SALIDA EN OTRA UNIDAD (solo con PATH_ROOT) ---
OUTPUT_BASE = None   # None = salida junto a cada CSV. Si defines ruta (ej. /mnt/disco2/dataset), va ahí
TEMP_BASE = None     # None = temp junto a cada CSV. Si defines ruta (ej. /mnt/disco2/temp), va ahí
# Se conserva la estructura: OUTPUT_BASE/exp1/, OUTPUT_BASE/exp2/, etc. (relativo a PATH_ROOT)

CLIP_META_PATH = None  # None = clip_meta.csv en OUTPUT_BASE o PATH_ROOT
OUTPUT_SUBDIR = "dataset_final_limpio"  # Subcarpeta dentro de cada output
TEMP_SUBDIR = "temp_clips"              # Subcarpeta dentro de cada temp


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
            })
        return experiments

    # Modo CSV_PATH
    if not CSV_PATH or not os.path.isfile(CSV_PATH):
        return []
    base = os.path.dirname(os.path.abspath(CSV_PATH))
    out = OUTPUT_DIR if OUTPUT_DIR else os.path.join(base, OUTPUT_SUBDIR)
    temp = TEMP_CLIPS if TEMP_CLIPS else os.path.join(base, TEMP_SUBDIR)
    videos = VIDEOS_DIR if VIDEOS_DIR else base
    return [{"csv": CSV_PATH, "videos": videos, "output": out, "temp": temp}]


def resolve_paths():
    """Resuelve rutas para build_clip_index."""
    exps = get_experiments()
    if exps:
        out = exps[0]["output"]
        if CLIP_META_PATH:
            clip_meta = CLIP_META_PATH
        elif PATH_ROOT and len(exps) > 1:
            clip_meta = str(Path(OUTPUT_BASE or PATH_ROOT) / "clip_meta.csv")
        else:
            clip_meta = os.path.join(out, "clip_meta.csv")
    else:
        base = os.path.dirname(os.path.abspath(CSV_PATH or "."))
        out = OUTPUT_DIR or os.path.join(base, OUTPUT_SUBDIR)
        clip_meta = CLIP_META_PATH or os.path.join(out, "clip_meta.csv")
    return {
        "csv": exps[0]["csv"] if exps else None,
        "videos": exps[0]["videos"] if exps else None,
        "output": out,
        "temp": exps[0]["temp"] if exps else None,
        "clip_meta": clip_meta,
        "all_outputs": [e["output"] for e in exps],
    }
