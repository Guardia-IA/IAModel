#!/usr/bin/env python3
"""
Compara poses_full.npy de un extract antiguo (p. ej. pose_extractor_clean.py)
con una extracción nueva del mismo MP4 vía triton_pose_extract (batch, sin --session-close).

Uso:
  export TRITON_EXTRACT_DECODE_BACKEND=opencv   # recomendado si el npy viejo es de clip.mp4 data_result
  python3 scripts/compare_legacy_pose_npy.py \\
    --video /path/clip.mp4 \\
    --reference /path/data_result/.../user_2/poses_full.npy \\
    [--epsilon 1e-3] [--json report.json]

  # Solo comparar dos npy ya generados (sin Triton):
  python3 scripts/compare_legacy_pose_npy.py \\
    --reference ref/poses_full.npy --fresh /tmp/out/.../user_0/poses_full.npy --skip-extract

Criterio PASS (por defecto): mismas filas T (±0), valid_mask idéntico, max|Δ| coords < epsilon
en filas donde ambos valid_mask son true.

Nota: YOLO local (clean) vs Triton TRT pueden diferir ligeramente aunque el pipeline sea equivalente;
epsilon 1e-3 suele bastar para coords normalizadas; sube a 1e-2 si solo buscas paridad operativa.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

EPS_DEFAULT = 1e-3


def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[1:] != (8, 2):
        raise ValueError(f"poses_full esperado (T,8,2), got {arr.shape} @ {path}")
    return arr.astype(np.float32)


def _load_valid_mask(user_dir: Path, poses: np.ndarray) -> np.ndarray:
    vm_path = user_dir / "valid_mask.npy"
    if vm_path.is_file():
        vm = np.load(vm_path)
        if vm.shape[0] != poses.shape[0]:
            raise ValueError(f"valid_mask T={vm.shape[0]} != poses T={poses.shape[0]}")
        return vm.astype(bool)
    # Misma regla que pose_extractor_clean / pose_core (≥5 KP visibles)
    out = np.zeros(poses.shape[0], dtype=bool)
    for t in range(poses.shape[0]):
        row = poses[t]
        vis = (~np.isnan(row).any(axis=1)).sum()
        out[t] = vis >= 5
    return out


def _compare_poses(
    ref: np.ndarray,
    ref_vm: np.ndarray,
    fresh: np.ndarray,
    fresh_vm: np.ndarray,
    *,
    epsilon: float,
) -> dict[str, Any]:
    t_ref, t_f = ref.shape[0], fresh.shape[0]
    n = min(t_ref, t_f)
    report: dict[str, Any] = {
        "reference_T": t_ref,
        "fresh_T": t_f,
        "rows_compared": n,
        "T_match": t_ref == t_f,
        "epsilon": epsilon,
    }
    if t_ref != t_f:
        report["T_warning"] = (
            "Distinto número de frames; se comparan las primeras min(T). "
            "Revisa decode (opencv vs ffmpeg) o que el MP4 sea el mismo clip."
        )

    ref, ref_vm = ref[:n], ref_vm[:n]
    fresh, fresh_vm = fresh[:n], fresh_vm[:n]

    vm_mismatch = int(np.sum(ref_vm != fresh_vm))
    both = ref_vm & fresh_vm
    n_both = int(both.sum())
    max_delta = 0.0
    mean_delta = 0.0
    if n_both > 0:
        diff = np.abs(ref[both] - fresh[both])
        max_delta = float(np.nanmax(diff))
        mean_delta = float(np.nanmean(diff))

    nan_pattern_mismatch = 0
    for t in range(n):
        ref_nan = np.isnan(ref[t]).all()
        fresh_nan = np.isnan(fresh[t]).all()
        if ref_nan != fresh_nan:
            nan_pattern_mismatch += 1

    pass_mask = vm_mismatch == 0
    pass_coords = (n_both == 0) or (max_delta < epsilon)
    report.update(
        {
            "valid_mask_mismatch_rows": vm_mismatch,
            "rows_both_valid": n_both,
            "nan_all_mismatch_rows": nan_pattern_mismatch,
            "max_abs_delta_coords": max_delta,
            "mean_abs_delta_coords": mean_delta,
            "pass_valid_mask": pass_mask,
            "pass_coords": pass_coords,
            "pass": pass_mask and pass_coords and t_ref == t_f,
        }
    )
    return report


def _score_user_vs_reference(ref: np.ndarray, ref_vm: np.ndarray, cand: np.ndarray, cand_vm: np.ndarray) -> float:
    n = min(ref.shape[0], cand.shape[0])
    both = ref_vm[:n] & cand_vm[:n]
    if not both.any():
        return float("inf")
    diff = np.abs(ref[:n][both] - cand[:n][both])
    return float(np.nanmean(diff))


def _pick_fresh_user_dir(clip_root: Path, ref: np.ndarray, ref_vm: np.ndarray) -> Path:
    users = sorted(clip_root.glob("user_*"))
    if not users:
        raise FileNotFoundError(f"Sin user_* en {clip_root}")
    if len(users) == 1:
        return users[0]
    best = min(
        users,
        key=lambda ud: _score_user_vs_reference(
            ref, ref_vm, _load_npy(ud / "poses_full.npy"), _load_valid_mask(ud, _load_npy(ud / "poses_full.npy"))
        ),
    )
    return best


def _run_fresh_extract(
    video: Path,
    work: Path,
    *,
    config: Path | None,
    triton_url: str | None,
    cat: str,
) -> Path:
    from extract_clips import _load_raw_config
    from triton_pose_extract.c1_like_extract import extract_clip_c1_like
    from triton_pose_extract.extract_stack import build_extract_stack
    from triton_pose_extract.paths import ensure_pose_core_importable
    from triton_pose_extract.pipeline_config import load_pipeline_config
    from triton_pose_extract.settings import ExtractSettings

    class _Args:
        config = str(config) if config else ""

    raw = _load_raw_config(_Args())
    cfg_path = config or PKG / "triton_pose_extract_config.json"
    settings = ExtractSettings.load(
        cfg_path,
        triton_url=triton_url or None,
        require_pose_repo=True,
    )
    pipe = load_pipeline_config(raw, output_base=work)
    pipe.filters.session_close_export = False
    pipe.filters.subject_bbox_json = None
    pipe.filters.export_track_id = None
    strict = pipe.filters.single_user_only and not pipe.filters.export_primary_user_on_multi_track
    settings.single_user_only = strict
    ensure_pose_core_importable(settings.triton_pose_repo)

    stack = build_extract_stack(settings)
    stack.tracker.reset()
    result, _timings = extract_clip_c1_like(
        stack,
        video,
        out_dir=work / "compare_out",
        cat=cat,
        clip_folder=video.stem,
        pipe_cfg=pipe.extract,
        filters=pipe.filters,
    )
    if not result.user_dirs:
        raise RuntimeError(
            "Extract sin user_* (multi-persona filtrada, decode vacío o single_user_only). "
            "Prueba --no-single-user-only en extract manual o revisa el clip."
        )
    return Path(result.out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Comparar poses_full antiguo vs extract Triton actual")
    ap.add_argument("--video", type=Path, default=None, help="MP4 (mismo archivo que en su día)")
    ap.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="poses_full.npy de referencia (p. ej. data_result/.../user_X/poses_full.npy)",
    )
    ap.add_argument(
        "--fresh",
        type=Path,
        default=None,
        help="poses_full.npy ya generado (salta extract si --skip-extract)",
    )
    ap.add_argument("--fresh-user-dir", type=Path, default=None, help="Carpeta user_* con npy nuevo")
    ap.add_argument("--skip-extract", action="store_true", help="No llamar Triton; usar --fresh o --fresh-user-dir")
    ap.add_argument("--work-dir", type=Path, default=None, help="Salida extract (default: temp)")
    ap.add_argument("--config", type=Path, default=None, help="triton_pose_extract_config.json")
    ap.add_argument("--triton-url", type=str, default="")
    ap.add_argument("--cat", type=str, default="compare", help="Subcarpeta cat en salida")
    ap.add_argument("--epsilon", type=float, default=EPS_DEFAULT)
    ap.add_argument("--json", type=Path, default=None, help="Escribir informe JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    ref_path = args.reference.expanduser().resolve()
    if not ref_path.is_file():
        print(f"ERROR: reference no existe: {ref_path}", file=sys.stderr)
        return 2

    ref = _load_npy(ref_path)
    ref_user_dir = ref_path.parent
    ref_vm = _load_valid_mask(ref_user_dir, ref)

    fresh_path: Path | None = None
    fresh_user_dir: Path | None = None

    if args.skip_extract:
        if args.fresh:
            fresh_path = args.fresh.expanduser().resolve()
        elif args.fresh_user_dir:
            fresh_user_dir = args.fresh_user_dir.expanduser().resolve()
            fresh_path = fresh_user_dir / "poses_full.npy"
        else:
            print("ERROR: --skip-extract requiere --fresh o --fresh-user-dir", file=sys.stderr)
            return 2
    else:
        if not args.video or not args.video.is_file():
            print("ERROR: indica --video existente (o --skip-extract)", file=sys.stderr)
            return 2
        from triton_pose_extract.triton_health import check_model_ready
        from triton_pose_extract.settings import ExtractSettings

        cfg_path = args.config or PKG / "triton_pose_extract_config.json"
        settings = ExtractSettings.load(
            cfg_path,
            triton_url=args.triton_url or None,
            require_pose_repo=True,
        )
        ready = check_model_ready(settings)
        if not ready.get("ok"):
            print(f"ERROR: Triton no ready: {ready}", file=sys.stderr)
            return 2

        if args.work_dir:
            work = args.work_dir.expanduser().resolve()
            work.mkdir(parents=True, exist_ok=True)
            cleanup = False
        else:
            work = Path(tempfile.mkdtemp(prefix="compare_legacy_pose_"))
            cleanup = True

        try:
            clip_root = _run_fresh_extract(
                args.video.expanduser().resolve(),
                work,
                config=args.config,
                triton_url=args.triton_url or None,
                cat=args.cat,
            )
            if args.fresh_user_dir:
                fresh_user_dir = args.fresh_user_dir.resolve()
                fresh_path = fresh_user_dir / "poses_full.npy"
            else:
                hint_user = ref_user_dir.name if ref_user_dir.name.startswith("user_") else None
                if hint_user and (clip_root / hint_user / "poses_full.npy").is_file():
                    fresh_user_dir = clip_root / hint_user
                else:
                    fresh_user_dir = _pick_fresh_user_dir(clip_root, ref, ref_vm)
                fresh_path = fresh_user_dir / "poses_full.npy"
            if args.verbose:
                print(f"Extract → {clip_root}", flush=True)
                print(f"Comparando con user {fresh_user_dir}", flush=True)
        finally:
            if cleanup and args.verbose:
                print(f"(work dir temporal: {work})", flush=True)

    assert fresh_path is not None and fresh_path.is_file()
    fresh = _load_npy(fresh_path)
    fresh_vm = _load_valid_mask(fresh_path.parent, fresh)

    report = _compare_poses(ref, ref_vm, fresh, fresh_vm, epsilon=args.epsilon)
    report["reference_path"] = str(ref_path)
    report["fresh_path"] = str(fresh_path)
    report["video"] = str(args.video.resolve()) if args.video else None

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.json:
        args.json.expanduser().write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if report["pass"]:
        print("\nPASS: los datos antiguos son equivalentes al extract actual (misma T, máscara y coords).")
        return 0
    print("\nFAIL: hay diferencias. Revisa T, valid_mask o sube --epsilon si solo cambió el backend.", file=sys.stderr)
    if not report["T_match"]:
        print("  → Distinto T: usa el mismo clip.mp4 y TRITON_EXTRACT_DECODE_BACKEND=opencv.", file=sys.stderr)
    if not report["pass_valid_mask"]:
        print("  → valid_mask distinto: tracker/modelo o filtros BODY_VISIBLE.", file=sys.stderr)
    if not report["pass_coords"]:
        print(
            f"  → coords: max Δ={report['max_abs_delta_coords']:.6g} (ε={args.epsilon}).",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
