"""
Genera un índice CSV (clip_meta.csv) a partir de todos los meta.json generados
por pose_extractor_clean.py. Útil para filtrar clips por calidad, categoría, etc.
"""
import argparse
import json
import os
from pathlib import Path


# --- CONFIGURACIÓN DE RUTAS (edita config.py) ---
try:
    from config import resolve_paths
    _paths = resolve_paths()
    DEFAULT_INPUT_DIRS = _paths.get("all_outputs")  # lista de directorios o None
    if not DEFAULT_INPUT_DIRS:
        DEFAULT_INPUT_DIRS = [_paths["output"]] if _paths.get("output") else []
    DEFAULT_OUTPUT_FILE = _paths.get("clip_meta") or (
        os.path.join(DEFAULT_INPUT_DIRS[0], "clip_meta.csv") if DEFAULT_INPUT_DIRS else None
    )
except ImportError:
    DEFAULT_INPUT_DIRS = []
    DEFAULT_OUTPUT_FILE = None


def build_clip_index(input_dirs, output_file: str) -> None:
    """
    Recorre input_dirs buscando meta.json, extrae campos y escribe clip_meta.csv.
    input_dirs: str (una ruta) o lista de rutas.
    """
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]

    metas = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"  [skip] No existe: {input_dir}")
            continue
        for meta_path in input_path.rglob("meta.json"):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                rel_path = meta_path.parent.relative_to(input_path)
                meta["_path"] = str(meta_path.parent)
                meta["_rel_path"] = str(rel_path)
                metas.append(meta)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [skip] {meta_path}: {e}")

    if not metas:
        print(f"No se encontraron meta.json en {input_dirs}")
        return

    # Columnas del CSV (campos relevantes para experimentos)
    cols = [
        "video_source", "row_csv", "t_start", "t_end", "cat",
        "valid_pct", "rel", "valid_frames", "total_frames",
        "poses_full_count", "poses_filtered_count", "passes_filters",
        "coverage_seconds", "clip_duration", "fps",
        "keypoint_confidence_avg", "occlusion_ratio", "bbox_aspect_ratio", "subject_velocity",
        "_rel_path", "_path",
    ]

    # Escribir CSV
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for m in metas:
            row = {k: m.get(k, "") for k in cols}
            writer.writerow(row)

    print(f"Índice guardado: {out_path} ({len(metas)} clips)")


def main():
    parser = argparse.ArgumentParser(description="Genera clip_meta.csv a partir de meta.json")
    parser.add_argument(
        "--input", "-i",
        type=str,
        action="append",
        default=None,
        help="Directorio(s) donde buscar meta.json (puede repetirse). Si no se pasa, usa config.py",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Ruta de salida para clip_meta.csv",
    )
    args = parser.parse_args()

    input_dirs = args.input or DEFAULT_INPUT_DIRS
    output_file = args.output or DEFAULT_OUTPUT_FILE

    if not input_dirs:
        print("Error: define --input o configura PATH_ROOT/CSV_PATH en config.py")
        return 1
    if not output_file:
        output_file = os.path.join(input_dirs[0], "clip_meta.csv")

    build_clip_index(input_dirs, output_file)
    return 0


if __name__ == "__main__":
    exit(main())
