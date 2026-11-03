#!/usr/bin/env python3
"""Muestra N clips al azar de data_result y comprueba el contrato del npy.

No compara con el merged.npy de C1 (eso es compare_c1_golden y exige la sesión).
Comprueba lo que tiene que cumplir una extracción correcta: forma, T, máscaras,
fps, duración y coordenadas normalizadas.

Uso:
  ./run_extract.sh audit-sample --data-result /ruta/data_result --n 10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from triton_pose_extract.validate_output import validate_clip_dir  # noqa: E402


def _resolve_root(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_dir():
        raise SystemExit(f"No existe la carpeta: {path}")
    nested = path / "data_result"
    if nested.is_dir():
        print(f"Usando {nested}", flush=True)
        return nested
    return path


def _has_poses_full(clip: Path) -> bool:
    try:
        with os.scandir(clip) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False) or not entry.name.startswith("user_"):
                    continue
                if os.path.isfile(os.path.join(entry.path, "poses_full.npy")):
                    return True
    except OSError:
        return False
    return False


def _category_names(root: Path) -> list[str]:
    with os.scandir(root) as it:
        return sorted(e.name for e in it if e.is_dir(follow_symlinks=False))


def _clip_names(cat_dir: Path) -> list[str]:
    with os.scandir(cat_dir) as it:
        return [e.name for e in it if e.is_dir(follow_symlinks=False)]


def _sample_with_npy(root: Path, n: int, rng: random.Random) -> tuple[list[Path], list[str]]:
    """Elige N clips sin recorrer todas las categorías.

    Solo lista los nombres de una categoría cuando sale en el sorteo.
    """
    cats = _category_names(root)
    print(f"Categorías ({len(cats)}): {', '.join(cats)}", flush=True)
    if not cats:
        return [], []
    remaining = list(cats)
    names: dict[str, list[str]] = {}
    chosen: list[Path] = []
    tried = 0
    limit = max(n * 40, n)
    while len(chosen) < n and remaining and tried < limit:
        tried += 1
        cat = rng.choice(remaining)
        if cat not in names:
            print(f"Nombres de la categoría {cat} …", flush=True)
            names[cat] = _clip_names(root / cat)
            print(f"  {cat}: {len(names[cat])} clips", flush=True)
        pool = names[cat]
        if not pool:
            remaining.remove(cat)
            continue
        clip_name = pool.pop(rng.randrange(len(pool)))
        clip = root / cat / clip_name
        if not _has_poses_full(clip):
            continue
        chosen.append(clip)
        print(f"  elegido {len(chosen)}/{n}: {cat}/{clip_name}", flush=True)
    if len(chosen) < n:
        print(f"Solo {len(chosen)} clips con npy (intentos {tried}).", flush=True)
    return chosen, list(names)


def _as_bool(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.bool_:
        return arr
    return arr.astype(np.int64) != 0


def _finite_kps(poses: np.ndarray) -> np.ndarray:
    """(T, 8) True si x e y del keypoint son finitos."""
    return np.isfinite(poses).all(axis=-1)


def _check_user(clip: Path, user_dir: Path, meta: dict) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {"user": user_dir.name}

    poses_path = user_dir / "poses_full.npy"
    if not poses_path.is_file():
        errors.append(f"{user_dir.name}: falta poses_full.npy")
        return errors, warnings, stats
    poses = np.load(poses_path)
    stats["shape"] = list(poses.shape)
    if poses.ndim != 3 or poses.shape[1:] != (8, 2):
        errors.append(f"{user_dir.name}: poses_full {tuple(poses.shape)} (esperado T,8,2)")
        return errors, warnings, stats
    if not np.issubdtype(poses.dtype, np.floating):
        errors.append(f"{user_dir.name}: poses_full dtype {poses.dtype} (esperado float)")
    t = int(poses.shape[0])
    stats["T"] = t
    if t <= 0:
        errors.append(f"{user_dir.name}: T=0")

    finite_xy = poses[np.isfinite(poses)]
    if finite_xy.size == 0:
        errors.append(f"{user_dir.name}: poses_full sin ninguna coordenada finita")
    else:
        lo = float(finite_xy.min())
        hi = float(finite_xy.max())
        stats["coord_min"] = round(lo, 4)
        stats["coord_max"] = round(hi, 4)
        if lo < -0.02 or hi > 1.02:
            errors.append(
                f"{user_dir.name}: coordenadas fuera de 0–1 (min={lo:.4f} max={hi:.4f})"
            )

    kp_finite = _finite_kps(poses)
    valid_from_pose = kp_finite.sum(axis=1) >= 5

    vm_path = user_dir / "valid_mask.npy"
    if not vm_path.is_file():
        errors.append(f"{user_dir.name}: falta valid_mask.npy")
        valid = valid_from_pose
    else:
        vm = np.load(vm_path)
        if vm.ndim != 1 or int(vm.shape[0]) != t:
            errors.append(f"{user_dir.name}: valid_mask {tuple(vm.shape)} (esperado ({t},))")
            valid = valid_from_pose
        else:
            valid = _as_bool(vm)
            if not np.array_equal(valid, valid_from_pose):
                mismatch = int(np.count_nonzero(valid != valid_from_pose))
                errors.append(
                    f"{user_dir.name}: valid_mask no coincide con ≥5 KP finitos ({mismatch} frames)"
                )

    km_path = user_dir / "keypoint_mask.npy"
    if not km_path.is_file():
        errors.append(f"{user_dir.name}: falta keypoint_mask.npy")
    else:
        km = np.load(km_path)
        if km.shape != (t, 8):
            errors.append(f"{user_dir.name}: keypoint_mask {tuple(km.shape)} (esperado ({t}, 8))")
        elif not np.array_equal(_as_bool(km), kp_finite):
            errors.append(f"{user_dir.name}: keypoint_mask no coincide con coordenadas finitas")

    n_valid = int(np.count_nonzero(valid))
    stats["valid_frames"] = n_valid
    stats["valid_pct"] = round(100.0 * n_valid / max(t, 1), 1)
    if n_valid <= 0:
        errors.append(f"{user_dir.name}: ningún frame válido")

    poses_f = user_dir / "poses.npy"
    if not poses_f.is_file():
        errors.append(f"{user_dir.name}: falta poses.npy")
    else:
        filtered = np.load(poses_f)
        if n_valid == 0:
            if filtered.size != 0:
                errors.append(f"{user_dir.name}: poses.npy no vacío con 0 frames válidos")
        elif filtered.ndim != 3 or filtered.shape != (n_valid, 8, 2):
            errors.append(
                f"{user_dir.name}: poses.npy {tuple(filtered.shape)} (esperado ({n_valid}, 8, 2))"
            )

    fc = int(meta.get("frame_count") or 0)
    pose_t = int(meta.get("poses_full_T") or 0)
    if fc and t != fc:
        errors.append(f"{user_dir.name}: T={t} != meta.frame_count={fc}")
    if pose_t and t != pose_t:
        errors.append(f"{user_dir.name}: T={t} != meta.poses_full_T={pose_t}")

    users_meta = meta.get("users") or []
    mine = next((u for u in users_meta if f"user_{u.get('track_id')}" == user_dir.name), None)
    if mine is not None:
        meta_valid = int(mine.get("valid_frames") or 0)
        if meta_valid != n_valid:
            errors.append(
                f"{user_dir.name}: meta.valid_frames={meta_valid} != máscara {n_valid}"
            )
    return errors, warnings, stats


def _check_clip(clip: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    meta_path = clip / "meta.json"
    if not meta_path.is_file():
        return {"clip": str(clip), "pass": False, "errors": ["falta meta.json"], "warnings": []}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    errors.extend(validate_clip_dir(clip))

    fps = meta.get("fps")
    try:
        fps_f = float(fps)
    except (TypeError, ValueError):
        fps_f = 0.0
    if not (1.0 <= fps_f <= 120.0):
        errors.append(f"fps={fps!r} fuera de 1–120")
    elif not (8.0 <= fps_f <= 60.0):
        warnings.append(f"fps={fps_f:.3f} fuera de lo habitual (8–60)")

    fc = int(meta.get("frame_count") or 0)
    dur = meta.get("clip_duration")
    if fps_f > 0 and fc > 0 and dur is not None:
        expect = fc / fps_f
        if abs(float(dur) - expect) > max(2.0 / fps_f, 0.05):
            errors.append(
                f"clip_duration={dur} != frame_count/fps={expect:.3f}"
            )

    backend = meta.get("decode_backend_used")
    if backend is None:
        warnings.append("meta sin decode_backend_used")
    elif str(backend) != "opencv":
        errors.append(f"decode_backend_used={backend!r} (esperado opencv)")

    engine = meta.get("iamodel_extract_engine")
    if engine is not None and str(engine) != "c1_like":
        warnings.append(f"iamodel_extract_engine={engine!r}")

    user_stats = []
    for user_dir in sorted(clip.glob("user_*")):
        u_err, u_warn, u_stats = _check_user(clip, user_dir, meta)
        errors.extend(u_err)
        warnings.extend(u_warn)
        user_stats.append(u_stats)

    # validate_clip_dir ya recorre los npy; quitar duplicados conservando orden
    dedup_e: list[str] = []
    for item in errors:
        if item not in dedup_e:
            dedup_e.append(item)
    dedup_w: list[str] = []
    for item in warnings:
        if item not in dedup_w:
            dedup_w.append(item)

    return {
        "clip": str(clip),
        "name": clip.name,
        "cat": clip.parent.name,
        "pass": not dedup_e,
        "fps": fps_f,
        "frame_count": fc,
        "decode_backend_used": backend,
        "errors": dedup_e,
        "warnings": dedup_w,
        "users": user_stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audita N extracciones aleatorias de data_result")
    ap.add_argument("--data-result", type=Path, required=True, help="Carpeta data_result o su padre")
    ap.add_argument("--n", type=int, required=True, help="Cuántos clips al azar")
    ap.add_argument("--seed", type=int, default=None, help="Semilla (reproducible)")
    ap.add_argument("--json", type=Path, default=None, help="Informe JSON")
    args = ap.parse_args()
    if args.n < 1:
        print("N tiene que ser >= 1", file=sys.stderr)
        return 2

    print("Auditoría: categorías al azar, sin recorrer todo data_result.", flush=True)
    root = _resolve_root(args.data_result)
    rng = random.Random(args.seed)
    sample, cats_opened = _sample_with_npy(root, args.n, rng)
    if not sample:
        print(f"No hay user_*/poses_full.npy bajo {root}", file=sys.stderr)
        return 2
    n = len(sample)

    print(
        f"Raíz: {root} | categorías abiertas: {', '.join(cats_opened) or '-'} | muestra: {n}"
        + (f" | seed={args.seed}" if args.seed is not None else ""),
        flush=True,
    )

    reports = []
    n_pass = 0
    for i, clip in enumerate(sample, start=1):
        print(f"[{i}/{n}] leyendo {clip.parent.name}/{clip.name} …", flush=True)
        rep = _check_clip(clip)
        reports.append(rep)
        mark = "PASS" if rep["pass"] else "FAIL"
        if rep["pass"]:
            n_pass += 1
        users = rep.get("users") or []
        valid = ",".join(f"{u.get('valid_pct', '?')}%" for u in users) or "-"
        print(
            f"[{i}/{n}] {mark}  cat={rep.get('cat')}  {rep.get('name')}  "
            f"T={rep.get('frame_count')}  fps={rep.get('fps')}  "
            f"valid={valid}  decode={rep.get('decode_backend_used')}"
        )
        for err in rep["errors"]:
            print(f"         - {err}")
        for warn in rep["warnings"]:
            print(f"         ~ {warn}")

    print(f"Resumen: {n_pass}/{n} PASS")
    payload = {
        "data_result": str(root),
        "categories_opened": cats_opened,
        "n": n,
        "seed": args.seed,
        "n_pass": n_pass,
        "n_fail": n - n_pass,
        "clips": reports,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Informe: {args.json}")
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
