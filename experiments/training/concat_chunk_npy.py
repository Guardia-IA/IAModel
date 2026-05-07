"""
Concatena arrays NumPy de subcarpetas chunk_NNN (mismo criterio que test_model2.py).

Por defecto lee en cada chunk el fichero indicado (poses.npy), comprueba meta.json para
frames válidos, omite chunks vacíos o con 0 frames válidos, y guarda un único .npy [T,J,...].
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Carpeta con chunk_001, chunk_002, ... → un solo .npy concatenado por orden temporal."
        )
    )
    p.add_argument(
        "--chunks-dir",
        required=True,
        type=Path,
        help="Directorio raíz que contiene subcarpetas chunk_NNN.",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Ruta del fichero de salida (.npy). Si no termina en .npy, numpy añade la extensión.",
    )
    p.add_argument(
        "--pose-source",
        choices=["filtered", "full"],
        default=None,
        help=(
            "Atajo para elegir el fichero por chunk: "
            "'filtered' -> poses.npy, 'full' -> poses_full.npy. "
            "Si se indica junto a --npy-name, prevalece --npy-name."
        ),
    )
    p.add_argument(
        "--npy-name",
        default="poses.npy",
        help="Nombre del fichero .npy dentro de cada chunk (por defecto poses.npy, como test_model2).",
    )
    p.add_argument(
        "--no-meta",
        action="store_true",
        help="No exigir meta.json: usa toda la T de cada array; sigue omitiendo arrays vacíos.",
    )
    return p.parse_args()


def _chunk_subdir_sort_key(path: Path) -> Tuple[int, str]:
    m = re.match(r"^chunk_(\d+)$", path.name, flags=re.IGNORECASE)
    if m:
        return (int(m.group(1)), path.name.lower())
    return (10**9, path.name.lower())


def _read_valid_frames_from_meta(meta: Dict[str, Any]) -> Optional[int]:
    for k in ("frames_validos", "valid_frames", "valid_frame_count", "filtered_frames"):
        if k in meta and meta[k] is not None:
            return int(meta[k])
    users = meta.get("users")
    if isinstance(users, list) and users:
        u0 = users[0]
        if isinstance(u0, dict):
            for k in ("valid_frames", "poses_filtered_count"):
                if k in u0 and u0[k] is not None:
                    return int(u0[k])
    return None


def _list_chunk_dirs(root: Path) -> List[Path]:
    subdirs = [p for p in root.iterdir() if p.is_dir() and re.match(r"^chunk_\d+$", p.name, re.I)]
    subdirs.sort(key=_chunk_subdir_sort_key)
    return subdirs


def concatenate_chunks(
    root: Path,
    npy_name: str,
    require_meta: bool,
) -> Tuple[np.ndarray, List[str], List[str]]:
    subdirs = _list_chunk_dirs(root)
    if not subdirs:
        raise SystemExit(
            f"No hay subcarpetas chunk_NNN en {root} (esperado p. ej. chunk_001, chunk_002)."
        )

    parts: List[np.ndarray] = []
    used: List[str] = []
    skipped: List[str] = []
    j_ref: Optional[int] = None
    ndim_ref: Optional[int] = None

    for ch in subdirs:
        name = ch.name
        npy_p = ch / npy_name
        meta_p = ch / "meta.json"

        if not npy_p.exists():
            skipped.append(f"{name} (sin {npy_name})")
            continue
        if require_meta and not meta_p.exists():
            skipped.append(f"{name} (sin meta.json)")
            continue

        meta: Dict[str, Any] = {}
        if meta_p.exists():
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                skipped.append(f"{name} (meta.json ilegible: {e})")
                continue
        elif require_meta:
            skipped.append(f"{name} (sin meta.json)")
            continue

        vf = _read_valid_frames_from_meta(meta) if meta else None

        try:
            arr = np.load(str(npy_p), allow_pickle=False)
        except Exception as e:
            skipped.append(f"{name} ({npy_name}: {e})")
            continue

        if vf is None:
            vf = int(arr.shape[0]) if arr.ndim >= 1 else 0
        if vf == 0:
            skipped.append(f"{name} (frames válidos = 0)")
            continue
        if arr.size == 0 or (arr.ndim >= 1 and arr.shape[0] == 0):
            skipped.append(f"{name} (array vacío)")
            continue

        if ndim_ref is None:
            ndim_ref = int(arr.ndim)
        elif int(arr.ndim) != ndim_ref:
            raise SystemExit(
                f"Incompatibilidad de ndim entre chunks: {name} tiene ndim={arr.ndim}, "
                f"antes ndim={ndim_ref}."
            )

        if j_ref is None and arr.ndim >= 2:
            j_ref = int(arr.shape[1])
        elif arr.ndim >= 2 and j_ref is not None and int(arr.shape[1]) != j_ref:
            raise SystemExit(
                f"Incompatibilidad de forma entre chunks: {name} tiene shape[1]={arr.shape[1]}, "
                f"antes {j_ref}."
            )

        parts.append(arr.astype(np.float32, copy=False))
        used.append(name)

    if not parts:
        raise SystemExit(
            "Ningún chunk aportó datos válidos (todos omitidos o vacíos). "
            f"Omitidos: {skipped}"
        )

    stacked = np.concatenate(parts, axis=0)
    return stacked, used, skipped


def main() -> None:
    args = parse_args()
    root = args.chunks_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"No es un directorio: {root}")

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    npy_name = args.npy_name
    if args.pose_source and args.npy_name == "poses.npy":
        npy_name = "poses_full.npy" if args.pose_source == "full" else "poses.npy"
    elif args.pose_source and args.npy_name != "poses.npy":
        print(
            "[WARN] Se indicó --pose-source junto con --npy-name; "
            "se usará --npy-name."
        )

    stacked, used, skipped = concatenate_chunks(
        root,
        npy_name=npy_name,
        require_meta=not args.no_meta,
    )

    np.save(str(out), stacked)

    print(f"[OK] Guardado: {out} (shape={stacked.shape}, dtype={stacked.dtype})")
    print(f"[INFO] Chunks usados ({len(used)}): {', '.join(used)}")
    if skipped:
        print(f"[INFO] Chunks omitidos ({len(skipped)}): {' | '.join(skipped)}")


if __name__ == "__main__":
    main()
