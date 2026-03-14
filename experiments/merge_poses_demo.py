"""
Demostración: une dos clips (uno clase 6, otro clase !=6) en un único .npy
para simular multiusuario donde solo uno roba. Permite visualizar el resultado.

Uso:
    python merge_poses_demo.py
    # Genera experiments/output/merged_demo.npy

Luego:
    python visualize_npy.py output/merged_demo.npy
"""
import json
import random
from pathlib import Path

import numpy as np

try:
    from .training.model_config import DATA_RESULT_ROOT
except ImportError:
    try:
        from training.model_config import DATA_RESULT_ROOT
    except ImportError:
        try:
            from model_config import DATA_RESULT_ROOT
        except ImportError:
            DATA_RESULT_ROOT = Path(__file__).resolve().parent.parent.parent / "ResultadosExperimentos" / "data_result"

MIN_SEQ_LEN = 4
OUTPUT_PATH = Path(__file__).parent / "output" / "merged_demo.npy"


def collect_single_user_clips(pose_source: str = "filtered") -> list[tuple[Path, int]]:
    """Recopila (ruta poses.npy, clase) para clips con un único usuario."""
    root = DATA_RESULT_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Data root no encontrado: {root}")
    items: list[tuple[Path, int]] = []

    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_str = cat_dir.name
        try:
            cat_int = int(cat_str)
        except ValueError:
            continue

        for clip_dir in sorted(cat_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            meta_path = clip_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            users = meta.get("users", [])
            if not users:
                continue

            if cat_str != "6":
                if len(users) != 1:
                    continue
                chosen = users[0]
            else:
                chosen = max(users, key=lambda u: u.get("total_frames", 0))

            track_id = chosen.get("track_id")
            if track_id is None:
                continue

            user_dir = clip_dir / f"user_{track_id}"
            fn = "poses.npy" if pose_source == "filtered" else "poses_full.npy"
            pose_path = user_dir / fn
            if not pose_path.exists():
                continue

            try:
                p = np.load(pose_path)
            except Exception:
                continue
            if p.ndim != 3 or p.shape[-1] != 2 or p.shape[0] < MIN_SEQ_LEN:
                continue

            items.append((pose_path, cat_int))

    return items


def merge_two_poses(path_a: Path, path_b: Path, offset_second: float = 0.35) -> np.ndarray:
    """
    Carga dos poses.npy, alinea temporalmente al mínimo de frames,
    y los combina en [T, 2, J, 2]. El segundo esqueleto se desplaza en x para no solaparse.
    """
    a = np.load(path_a).astype(np.float32)  # [T1, J, 2]
    b = np.load(path_b).astype(np.float32)  # [T2, J, 2]
    t1, j1, _ = a.shape
    t2, j2, _ = b.shape
    t = min(t1, t2)
    j = min(j1, j2)
    a = a[:t, :j, :]
    b = b[:t, :j, :]

    # Muestreo si tienen distinta longitud (ya recortamos a t)
    # Desplazar segundo esqueleto en x para separarlos visualmente
    b_offset = b.copy()
    b_offset[..., 0] += offset_second  # desplazar en x
    b_offset = np.clip(b_offset, 0.0, 1.0)

    merged = np.stack([a, b_offset], axis=1)  # [T, 2, J, 2]
    return merged


def main():
    random.seed(42)
    items = collect_single_user_clips("filtered")
    if not items:
        print("No se encontraron clips válidos (un usuario por clip).")
        return

    by_cat: dict[int, list[tuple[Path, int]]] = {}
    for path, cat in items:
        by_cat.setdefault(cat, []).append((path, cat))

    cat6 = by_cat.get(6, [])
    others = [(p, c) for c in by_cat if c != 6 for (p, _) in by_cat[c]]

    if not cat6:
        print("No hay clips de clase 6 (robos).")
        return
    if not others:
        print("No hay clips de otras clases.")
        return

    path_robo, _ = random.choice(cat6)
    path_otro, cat_otro = random.choice(others)

    path_robo_full = path_robo.resolve()
    path_otro_full = path_otro.resolve()
    print("Clips utilizados para la mezcla (rutas completas):")
    print(f"  Robo (clase 6): {path_robo_full}")
    print(f"  Otra clase ({cat_otro}): {path_otro_full}")

    merged = merge_two_poses(path_robo, path_otro)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_PATH), merged)
    print(f"Guardado: {OUTPUT_PATH} | shape={merged.shape} (frames, 2_usuarios, joints, xy)")
    print("")
    print("Para visualizar:")
    print(f"  python visualize_npy.py {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
