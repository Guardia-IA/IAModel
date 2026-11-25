#!/usr/bin/env python3
"""
Motion (IAModel 48+49) + veto HAR_v2 pocket (Kanvis vendor/har).

No usa har_legacy: solo ``pocket_infer`` / ``c3_har_pocket_v2``.

Ejemplos:

  # Un npy (probe 4 s chunk002)
  python3 motion_har_veto_demo.py \\
    --poses-full /home/debian/Descargas/merged/ses2/poses_probe_chunk002_seg0_4s.npy \\
    --valid-mask /home/debian/Descargas/merged/ses2/valid_mask_probe_chunk002_seg0_4s.npy \\
    --model-a /home/debian/Documentos/Kanvis/modelos_v2/modelo_48.pt \\
    --model-b /home/debian/Documentos/Kanvis/modelos_v2/modelo_49.pt

  # Sesión entera (chunks + overlaps en p1.json)
  python3 motion_har_veto_demo.py \\
    --session-dir /home/debian/Descargas/merged/ses2 \\
    --p1-json /home/debian/Descargas/merged/ses2/p1.json \\
    --model-a ... --model-b ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

CAMPAIGN_DIR = Path(__file__).resolve().parent
DEFAULT_HAR_ROOT = Path(
    "/home/debian/dev/Proyectos/Kanvis-Orchestrator/kanvis-orchestrator/app/vendor/har"
)


def _ensure_har_path(har_root: Path) -> None:
    root = har_root.expanduser().resolve()
    if not (root / "pocket_infer.py").is_file():
        raise SystemExit(f"No encuentro HAR v2 en {root} (pocket_infer.py)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def run_har_v2(
    poses: np.ndarray,
    *,
    har_root: Path,
    fps: float,
    valid_mask: np.ndarray | None,
    timestamps: np.ndarray | None,
) -> dict[str, Any]:
    _ensure_har_path(har_root)
    from pocket_infer import infer_from_poses_array, result_to_c3_dict

    poses = np.nan_to_num(poses, nan=0.0).astype(np.float32)
    result = infer_from_poses_array(
        poses,
        fps=float(fps),
        aspect=16.0 / 9.0,
        row_mask=valid_mask,
        timestamps=timestamps,
    )
    payload = result_to_c3_dict(result)
    payload["schema_version"] = "c3_har_pocket_v2"
    return payload


def run_motion(
    poses_path: Path,
    valid_mask: Path | None,
    *,
    model_a: Path,
    model_b: Path,
    threshold: float,
    use_mask: bool,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(CAMPAIGN_DIR / "score_pose_ensemble.py"),
        "--poses-full",
        str(poses_path),
        "--model-a",
        str(model_a),
        "--model-b",
        str(model_b),
        "--rule",
        "mean",
        "--threshold",
        str(threshold),
        "--json",
    ]
    if valid_mask and valid_mask.is_file():
        cmd.extend(["--valid-mask", str(valid_mask)])
    if not use_mask:
        cmd.append("--no-valid-mask-filter")
    return json.loads(subprocess.check_output(cmd, cwd=str(CAMPAIGN_DIR)))


def motion_alarm_after_har_veto(motion: dict[str, Any], har: dict[str, Any]) -> dict[str, Any]:
    """Política demo alineada con orchestrator (sin SAM3): no alarma si HAR_v2 no confirma."""
    raw_alarm = bool(motion.get("robbery_alarm"))
    htb = har.get("hand_to_body") is True
    branch = str(har.get("orchestrator_branch") or "")
    suppress = har.get("motion_suppress") is True
    resultado = str(har.get("resultado") or "")
    motivo = str(har.get("motivo") or "")

    har_blocks = (
        suppress
        or branch in ("DISCARD_SUPPRESS", "DISCARD_OTHER")
        or (not htb and branch != "CONFIRM")
    )
    veto_reason: list[str] = []
    if suppress:
        veto_reason.append("motion_suppress=true")
    if branch and branch not in ("CONFIRM", "REVIEW_RAW"):
        veto_reason.append(f"orchestrator_branch={branch}")
    if not htb:
        veto_reason.append("hand_to_body=false")
    if resultado == "INCOMPATIBLE" and motivo:
        veto_reason.append(f"resultado=INCOMPATIBLE motivo={motivo}")

    final_alarm = raw_alarm and not har_blocks
    return {
        "motion_robbery_alarm_raw": raw_alarm,
        "motion_robbery_score": motion.get("robbery_score"),
        "har_hand_to_body": htb,
        "har_motion_suppress": suppress,
        "har_orchestrator_branch": branch,
        "har_resultado": resultado,
        "har_motivo": motivo,
        "har_accion_propuesta": har.get("accion_propuesta"),
        "final_robbery_alarm_after_har_veto": final_alarm,
        "veto_applied": raw_alarm and har_blocks,
        "veto_reason": veto_reason,
    }


def _load_mask_ts(poses_path: Path, valid_mask: Path | None, timestamps: Path | None):
    vm = np.load(valid_mask) if valid_mask and valid_mask.is_file() else None
    ts = np.load(timestamps) if timestamps and timestamps.is_file() else None
    if vm is None:
        side = poses_path.parent / "valid_mask.npy"
        if side.is_file():
            vm = np.load(side)
    if ts is None:
        side = poses_path.parent / "pose_frame_timestamps.npy"
        if side.is_file():
            ts = np.load(side)
    return vm, ts


def analyze_one(
    poses_path: Path,
    *,
    label: str,
    valid_mask: Path | None,
    timestamps: Path | None,
    model_a: Path,
    model_b: Path,
    threshold: float,
    use_mask: bool,
    har_root: Path,
    fps: float,
) -> dict[str, Any]:
    poses = np.load(poses_path)
    vm_path = valid_mask
    vm, ts = _load_mask_ts(poses_path, vm_path, timestamps)
    har = run_har_v2(poses, har_root=har_root, fps=fps, valid_mask=vm, timestamps=ts)
    motion = run_motion(poses_path, vm_path, model_a=model_a, model_b=model_b, threshold=threshold, use_mask=use_mask)
    merged = motion_alarm_after_har_veto(motion, har)
    wh_min = None
    p = np.nan_to_num(poses)
    if p.shape[0] > 0:
        wh = np.linalg.norm(p[:, 4:6].mean(axis=1) - p[:, 6:8].mean(axis=1), axis=1)
        wh_min = float(np.nanmin(wh)) if wh.size else None
    return {
        "label": label,
        "poses_full": str(poses_path.resolve()),
        "T_raw": int(poses.shape[0]),
        "min_wrist_hip_norm": wh_min,
        "motion": motion,
        "har_pocket_v2": har,
        "veto": merged,
    }


def _windows_from_session(
    session_dir: Path, p1: dict[str, Any]
) -> list[tuple[str, Path, Path | None, bool]]:
    """(label, poses_path, valid_mask_path, use_valid_mask_filter)."""
    rows: list[tuple[str, Path, Path | None, bool]] = []
    for key in sorted(p1.keys()):
        if not key.startswith("chunk_"):
            continue
        entry = p1[key]
        if not isinstance(entry, dict):
            continue
        use_mask = bool(entry.get("used_valid_mask_filter", True))
        if key.count("_") == 1 and key.replace("chunk_", "").isdigit():
            chunk = session_dir / key
            poses = chunk / "poses_full.npy"
            if poses.is_file():
                rows.append((key, poses, chunk / "valid_mask.npy", use_mask))
        elif "_" in key.replace("chunk_", "", 1):
            parts = key.split("_")
            if len(parts) == 3 and parts[0] == "chunk":
                a, b = int(parts[1]), int(parts[2])
                ov = session_dir / f"poses_overlap_{a:03d}_{b:03d}_dense.npy"
                vm = session_dir / f"valid_mask_overlap_{a:03d}_{b:03d}_dense.npy"
                if ov.is_file():
                    rows.append(
                        (
                            key,
                            ov,
                            vm if vm.is_file() else None,
                            bool(entry.get("used_valid_mask_filter", False)),
                        )
                    )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo motion + veto HAR_v2 pocket")
    ap.add_argument("--poses-full", type=Path, default=None)
    ap.add_argument("--valid-mask", type=Path, default=None)
    ap.add_argument("--timestamps", type=Path, default=None)
    ap.add_argument("--session-dir", type=Path, default=None, help="Carpeta sesión (chunks)")
    ap.add_argument("--p1-json", type=Path, default=None)
    ap.add_argument("--model-a", type=Path, required=True)
    ap.add_argument("--model-b", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--no-valid-mask-filter", action="store_true")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--har-root", type=Path, default=DEFAULT_HAR_ROOT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    use_mask = not args.no_valid_mask_filter
    results: list[dict[str, Any]] = []

    if args.poses_full:
        if not args.poses_full.is_file():
            raise SystemExit(f"No existe {args.poses_full}")
        results.append(
            analyze_one(
                args.poses_full,
                label=args.poses_full.stem,
                valid_mask=args.valid_mask,
                timestamps=args.timestamps,
                model_a=args.model_a,
                model_b=args.model_b,
                threshold=args.threshold,
                use_mask=use_mask,
                har_root=args.har_root,
                fps=args.fps,
            )
        )
    elif args.session_dir and args.p1_json:
        p1 = json.loads(args.p1_json.read_text(encoding="utf-8"))
        default_mask = use_mask
        for label, poses_path, vm_path, use_mask_flag in _windows_from_session(
            args.session_dir, p1
        ):
            if not vm_path or not vm_path.is_file():
                vm_path = None
            results.append(
                analyze_one(
                    poses_path,
                    label=label,
                    valid_mask=vm_path,
                    timestamps=None,
                    model_a=args.model_a,
                    model_b=args.model_b,
                    threshold=args.threshold,
                    use_mask=default_mask and use_mask_flag,
                    har_root=args.har_root,
                    fps=args.fps,
                )
            )
    else:
        raise SystemExit("Indica --poses-full o (--session-dir y --p1-json)")

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2, ensure_ascii=False))
        return 0

    for row in results:
        v = row["veto"]
        print(f"\n=== {row['label']} ===")
        print(f"poses: {row['poses_full']}")
        print(f"T={row['T_raw']}  min_wrist_hip(norm)≈{row.get('min_wrist_hip_norm')}")
        print(
            f"Motion MEAN={v['motion_robbery_score']:.4f}  "
            f"raw_alarm={v['motion_robbery_alarm_raw']}"
        )
        print(
            f"HAR_v2: hand_to_body={v['har_hand_to_body']}  "
            f"motion_suppress={v['har_motion_suppress']}  "
            f"branch={v['har_orchestrator_branch']}"
        )
        print(f"       resultado={v['har_resultado']}  motivo={v['har_motivo']}")
        print(f"       accion_propuesta={v.get('har_accion_propuesta')}")
        if v["veto_applied"]:
            print(f"VETO → final_alarm=False  ({'; '.join(v['veto_reason'])})")
        else:
            print(f"final_alarm_after_veto={v['final_robbery_alarm_after_har_veto']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())