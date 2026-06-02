"""
Ejecuta en lote la inferencia de `test_model2.py` sobre una lista de experimentos
definida en un fichero de configuración JSON, y vuelca el resultado a un CSV.

Configuración (JSON): una lista de objetos, cada uno con:
    - "nombre":        nombre del vídeo (etiqueta de la FILA en el CSV).
    - "modelo":        ruta al checkpoint .pt del clasificador.
    - "modelo_nombre": (opcional) etiqueta de la COLUMNA en el CSV.
                       Por defecto se usa el nombre de fichero del .pt (sin extensión).
    - "merged_npy":    ruta al fichero merged.npy de poses de un único usuario ([T, J, 2]).

Resultado (CSV):
    - Una fila por nombre de vídeo.
    - Una columna por modelo.
    - Cada celda contiene el LOGIT CRUDO de clase[1] (índice 1 de la salida del modelo).

Antes de ejecutar nada se valida que existan TODOS los modelos y TODOS los merged.npy
referenciados; si falta alguno, se aborta sin lanzar el experimento.

Uso:
    python batch_logits_runner.py --config batch_logits_config.json --output resultados_logits.csv
    python batch_logits_runner.py --config batch_logits_config.json --output resultados_logits.csv --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Aseguramos que tanto experiments/ como experiments/training/ están en el path,
# de forma que test_model2 pueda resolver sus imports (train_model_operations, etc.).
_THIS_DIR = Path(__file__).resolve().parent          # experiments/training
_EXPERIMENTS_DIR = _THIS_DIR.parent                  # experiments
for _p in (str(_THIS_DIR), str(_EXPERIMENTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_model2 as tm  # noqa: E402  (reutilizamos su lógica de inferencia)


def load_config(config_path: Path) -> List[Dict[str, Any]]:
    """Lee la configuración. Acepta una lista de experimentos o un objeto con clave 'experimentos'."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("experimentos", data.get("experiments", []))
    if not isinstance(data, list):
        raise SystemExit("El fichero de configuración debe ser una lista de objetos JSON.")
    return data


def build_model_from_checkpoint(
    model_path: Path, device: torch.device
) -> Tuple[Dict[str, Any], torch.nn.Module]:
    """Carga el checkpoint y construye el modelo igual que test_model2.main()."""
    checkpoint = torch.load(str(model_path), map_location="cpu")
    cfg = checkpoint.get("config", {})
    arch = cfg.get("arch", "tcn")
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint.get("num_classes", len(checkpoint["label_to_idx"])))
    model = tm.build_model(arch, input_dim, num_classes, cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def build_clip_from_merged(merged_path: Path) -> Path:
    """
    Crea un directorio temporal tipo clip (user_0/poses.npy + meta.json) a partir
    de un merged.npy de un único usuario con forma [T, J, 2].
    """
    arr = np.load(str(merged_path), allow_pickle=False)
    if getattr(arr, "ndim", 0) != 3 or int(arr.shape[2]) != 2:
        raise ValueError(
            f"Se esperaba un merged.npy de un usuario con forma [T, J, 2]; "
            f"shape={getattr(arr, 'shape', None)}"
        )
    arr = np.ascontiguousarray(arr.astype(np.float32))

    root = Path(tempfile.mkdtemp(prefix="batch_logits_"))
    user_dir = root / "user_0"
    user_dir.mkdir(parents=True, exist_ok=False)
    np.save(str(user_dir / "poses.npy"), arr)
    meta = {
        "clip_name": merged_path.stem or "merged",
        "users": [{"track_id": 0, "total_frames": int(arr.shape[0])}],
        "source_npy": str(merged_path.resolve()),
    }
    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return root


def logit_class1_for_merged(
    *,
    checkpoint: Dict[str, Any],
    model: torch.nn.Module,
    merged_path: Path,
    clip_name: str,
    device: torch.device,
) -> float:
    """Ejecuta la inferencia (pipeline por defecto de test_model2) y devuelve el logit crudo de clase[1]."""
    clip_dir = build_clip_from_merged(merged_path)
    tmp_root = clip_dir / ".infer_tmp"
    try:
        user_dirs = tm._find_user_dirs(clip_dir)
        if not user_dirs:
            raise RuntimeError(f"No se generaron usuarios para {merged_path}")
        # Solo el primer usuario (user_0).
        ud = user_dirs[0]
        _tid, _prob, logits, _clf_ms, _total_ms, cleanup = tm.infer_user_track(
            checkpoint=checkpoint,
            model=model,
            user_dir=ud,
            pose_source="filtered",
            clip_name=clip_name,
            device=device,
            simple_preprocess=False,
            use_temp_npy=True,
            tmp_dir=tmp_root,
        )
        tm._safe_unlink(cleanup)
        logits_1d = logits.detach().float().cpu().reshape(-1)
        if int(logits_1d.shape[0]) < 2:
            raise RuntimeError(
                f"El modelo solo produjo {int(logits_1d.shape[0])} logit(s); no existe clase[1]."
            )
        return float(logits_1d[1].item())
    finally:
        shutil.rmtree(clip_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aplica una lista de (modelo, merged.npy) y genera un CSV de logits de clase[1]."
    )
    parser.add_argument("--config", required=True, help="Fichero JSON con la lista de experimentos.")
    parser.add_argument("--output", required=True, help="Ruta del CSV de salida.")
    parser.add_argument("--device", default=None, help="cpu | cuda (autodetección si se omite).")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    output_path = Path(args.output).expanduser().resolve()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[INFO] Device: {device}")

    experiments = load_config(config_path)
    if not experiments:
        raise SystemExit("La configuración no contiene experimentos.")
    print(f"[INFO] Experimentos a ejecutar: {len(experiments)}")

    # --- Comprobación previa: estructura + existencia de TODOS los ficheros ---
    errores: List[str] = []
    for i, exp in enumerate(experiments, start=1):
        nombre = str(exp.get("nombre", "")).strip()
        modelo = str(exp.get("modelo", "")).strip()
        merged = str(exp.get("merged_npy", "")).strip()
        if not nombre:
            errores.append(f"Experimento #{i}: falta 'nombre'.")
        if not modelo:
            errores.append(f"Experimento #{i}: falta 'modelo'.")
        elif not Path(modelo).expanduser().resolve().is_file():
            errores.append(f"Experimento #{i}: no existe el modelo: {modelo}")
        if not merged:
            errores.append(f"Experimento #{i}: falta 'merged_npy'.")
        elif not Path(merged).expanduser().resolve().is_file():
            errores.append(f"Experimento #{i}: no existe el merged.npy: {merged}")

    if errores:
        print(f"[ABORTADO] Se encontraron {len(errores)} problema(s) en la configuración:")
        for msg in errores:
            print(f"  - {msg}")
        raise SystemExit(1)
    print("[INFO] Comprobación previa OK: todos los modelos y merged.npy existen.")

    # Caché de modelos por ruta resuelta (evita recargar el mismo .pt varias veces).
    model_cache: Dict[str, Tuple[Dict[str, Any], torch.nn.Module]] = {}

    # (fila, columna) -> valor. Conservamos orden de aparición de filas y columnas.
    cell_values: Dict[Tuple[str, str], str] = {}
    row_order: List[str] = []
    col_order: List[str] = []

    for i, exp in enumerate(experiments, start=1):
        nombre = str(exp.get("nombre", "")).strip()
        modelo = str(exp.get("modelo", "")).strip()
        merged = str(exp.get("merged_npy", "")).strip()
        if not nombre or not modelo or not merged:
            print(f"[WARN] Experimento #{i} incompleto (nombre/modelo/merged_npy). Se omite.")
            continue

        model_path = Path(modelo).expanduser().resolve()
        merged_path = Path(merged).expanduser().resolve()
        col_name = str(exp.get("modelo_nombre") or model_path.stem)

        if nombre not in row_order:
            row_order.append(nombre)
        if col_name not in col_order:
            col_order.append(col_name)

        print(f"[INFO] ({i}/{len(experiments)}) video='{nombre}' modelo='{col_name}'")

        if not model_path.exists():
            print(f"[WARN]   No existe el modelo: {model_path}")
            cell_values[(nombre, col_name)] = "ERROR:modelo_no_encontrado"
            continue
        if not merged_path.exists():
            print(f"[WARN]   No existe el merged.npy: {merged_path}")
            cell_values[(nombre, col_name)] = "ERROR:merged_no_encontrado"
            continue

        try:
            key = str(model_path)
            if key not in model_cache:
                model_cache[key] = build_model_from_checkpoint(model_path, device)
            checkpoint, model = model_cache[key]
            logit_c1 = logit_class1_for_merged(
                checkpoint=checkpoint,
                model=model,
                merged_path=merged_path,
                clip_name=nombre,
                device=device,
            )
            cell_values[(nombre, col_name)] = repr(logit_c1)
            print(f"[OK]     logit clase[1] = {logit_c1:+.6f}")
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR]  {e}")
            cell_values[(nombre, col_name)] = f"ERROR:{type(e).__name__}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video"] + col_order)
        for row in row_order:
            writer.writerow([row] + [cell_values.get((row, col), "") for col in col_order])

    print(f"[INFO] CSV escrito en: {output_path}")
    print(f"[INFO] Filas (vídeos): {len(row_order)} | Columnas (modelos): {len(col_order)}")


if __name__ == "__main__":
    main()
