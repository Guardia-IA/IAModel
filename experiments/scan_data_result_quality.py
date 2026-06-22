#!/usr/bin/env python3
"""
Escanea una carpeta tipo data_result y lista clips con calidad aceptable:

  - un único usuario en meta.json
  - duración del clip entre min_sec y max_sec (default 5–10 s)
  - poses.npy coherente con esa duración (frames ≈ duración × fps)

Estructura esperada (recursiva):
  .../{cat}/{clip_name}/meta.json
  .../{cat}/{clip_name}/user_{track_id}/poses.npy

Uso:
  python scan_data_result_quality.py /ruta/a/data_result
  python scan_data_result_quality.py /ruta/a/data_result --list-ok
  python scan_data_result_quality.py /ruta/a/data_result --csv ok_clips.csv
  python scan_data_result_quality.py /ruta/a/data_result --json report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Se requiere numpy: pip install numpy") from exc


DEFAULT_MIN_SEC = 5.0
DEFAULT_MAX_SEC = 10.0
DEFAULT_FPS = 12.5
DEFAULT_POSE_TOLERANCE = 0.15  # ±15 % en conteo de frames vs duración×fps


@dataclass
class ClipReport:
    path: Path
    category: str
    clip_name: str
    ok: bool
    reasons: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    fps: float = DEFAULT_FPS
    n_users: int = 0
    track_id: Optional[int] = None
    pose_frames: int = 0
    valid_frames: int = 0
    pose_duration_sec: float = 0.0

    def as_dict(self, root: Path) -> Dict[str, Any]:
        try:
            rel = str(self.path.relative_to(root))
        except ValueError:
            rel = str(self.path)
        return {
            "path": rel,
            "category": self.category,
            "clip_name": self.clip_name,
            "ok": self.ok,
            "reasons": list(self.reasons),
            "duration_sec": round(self.duration_sec, 3),
            "fps": round(self.fps, 3),
            "n_users": self.n_users,
            "track_id": self.track_id,
            "pose_frames": self.pose_frames,
            "valid_frames": self.valid_frames,
            "pose_duration_sec": round(self.pose_duration_sec, 3),
        }


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_category_from_path(clip_dir: Path, root: Path) -> str:
    try:
        rel = clip_dir.relative_to(root)
        parts = rel.parts
        if len(parts) >= 2:
            return parts[0]
    except ValueError:
        pass
    return "?"


def _expected_frame_bounds(
    duration_sec: float,
    fps: float,
    tolerance: float,
) -> tuple[int, int]:
    center = duration_sec * fps
    lo = int(max(1, center * (1.0 - tolerance)))
    hi = int(max(lo, center * (1.0 + tolerance)))
    return lo, hi


def _find_pose_file(user_dir: Path, pose_source: str) -> Optional[Path]:
    if pose_source == "full":
        for name in ("poses_full.npy", "poses.npy"):
            p = user_dir / name
            if p.is_file():
                return p
        return None
    for name in ("poses.npy", "poses_full.npy"):
        p = user_dir / name
        if p.is_file():
            return p
    return None


def evaluate_clip(
    clip_dir: Path,
    root: Path,
    *,
    min_sec: float,
    max_sec: float,
    pose_tolerance: float,
    pose_source: str,
    min_valid_pct: float,
) -> ClipReport:
    meta_path = clip_dir / "meta.json"
    category = _parse_category_from_path(clip_dir, root)
    clip_name = clip_dir.name
    report = ClipReport(path=clip_dir, category=category, clip_name=clip_name, ok=False)

    if not meta_path.is_file():
        report.reasons.append("missing_meta_json")
        return report

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        report.reasons.append("invalid_meta_json")
        return report

    clip_name = str(meta.get("clip_name", clip_name))
    report.clip_name = clip_name

    duration = _to_float(meta.get("clip_duration"), default=0.0)
    fps = _to_float(meta.get("fps"), default=DEFAULT_FPS)
    if fps <= 0:
        fps = DEFAULT_FPS
    report.duration_sec = duration
    report.fps = fps

    if duration < min_sec or duration > max_sec:
        report.reasons.append(
            f"duration_out_of_range ({duration:.2f}s not in [{min_sec}, {max_sec}])"
        )

    users = meta.get("users") or []
    report.n_users = len(users)
    if len(users) != 1:
        report.reasons.append(f"multi_user (n={len(users)})")
        return report

    user = users[0]
    track_id = user.get("track_id")
    if track_id is None:
        report.reasons.append("missing_track_id")
        return report
    report.track_id = int(track_id)

    user_dir = clip_dir / f"user_{track_id}"
    pose_path = _find_pose_file(user_dir, pose_source)
    if pose_path is None:
        report.reasons.append("missing_poses_npy")
        return report

    try:
        poses = np.load(pose_path)
    except Exception:
        report.reasons.append("invalid_poses_npy")
        return report

    if poses.ndim != 3 or poses.shape[-1] != 2:
        report.reasons.append(f"bad_poses_shape {getattr(poses, 'shape', None)}")
        return report

    pose_len = int(poses.shape[0])
    report.pose_frames = pose_len
    report.pose_duration_sec = pose_len / fps

    total_frames = _to_int(user.get("total_frames"), default=pose_len)
    valid_frames = _to_int(user.get("valid_frames"), default=pose_len)
    report.valid_frames = valid_frames

    if total_frames <= 0:
        total_frames = pose_len
    if abs(pose_len - total_frames) > max(2, int(total_frames * pose_tolerance)):
        report.reasons.append(
            f"pose_meta_frame_mismatch (poses={pose_len}, meta total_frames={total_frames})"
        )

    lo, hi = _expected_frame_bounds(duration, fps, pose_tolerance)
    if pose_len < lo or pose_len > hi:
        report.reasons.append(
            f"pose_frames_out_of_range ({pose_len} not in [{lo}, {hi}] for {duration:.1f}s @ {fps:.1f}fps)"
        )

    if report.pose_duration_sec < min_sec or report.pose_duration_sec > max_sec:
        report.reasons.append(
            f"pose_duration_out_of_range ({report.pose_duration_sec:.2f}s not in [{min_sec}, {max_sec}])"
        )

    if total_frames > 0 and valid_frames > 0:
        valid_pct = 100.0 * valid_frames / total_frames
        if valid_pct < min_valid_pct:
            report.reasons.append(f"low_valid_pct ({valid_pct:.1f}% < {min_valid_pct}%)")

    report.ok = len(report.reasons) == 0
    return report


def iter_clip_dirs(root: Path) -> Iterable[Path]:
    for meta_path in sorted(root.rglob("meta.json")):
        yield meta_path.parent


def scan_root(
    root: Path,
    *,
    min_sec: float,
    max_sec: float,
    pose_tolerance: float,
    pose_source: str,
    min_valid_pct: float,
) -> List[ClipReport]:
    if not root.is_dir():
        raise FileNotFoundError(f"No es una carpeta: {root}")
    reports: List[ClipReport] = []
    for clip_dir in iter_clip_dirs(root):
        reports.append(
            evaluate_clip(
                clip_dir,
                root,
                min_sec=min_sec,
                max_sec=max_sec,
                pose_tolerance=pose_tolerance,
                pose_source=pose_source,
                min_valid_pct=min_valid_pct,
            )
        )
    return reports


def _reason_bucket(reason: str) -> str:
    return reason.split(" ", 1)[0].split("(", 1)[0]


def print_summary(reports: List[ClipReport], root: Path) -> None:
    total = len(reports)
    ok_reports = [r for r in reports if r.ok]
    n_ok = len(ok_reports)
    pct = 100.0 * n_ok / total if total else 0.0

    print(f"\nRaíz: {root.resolve()}")
    print(f"Clips escaneados: {total}")
    print(f"Aceptables: {n_ok} ({pct:.1f}%)")

    by_cat_total: Dict[str, int] = defaultdict(int)
    by_cat_ok: Dict[str, int] = defaultdict(int)
    reject_reasons: Counter[str] = Counter()

    for r in reports:
        by_cat_total[r.category] += 1
        if r.ok:
            by_cat_ok[r.category] += 1
        else:
            for reason in r.reasons:
                reject_reasons[_reason_bucket(reason)] += 1

    if by_cat_total:
        print("\nPor categoría (aceptables / total):")
        for cat in sorted(by_cat_total.keys(), key=lambda x: (not x.isdigit(), x)):
            print(f"  cat {cat:>3}: {by_cat_ok[cat]:5d} / {by_cat_total[cat]:5d}")

    if reject_reasons:
        print("\nMotivos de rechazo (un clip puede tener varios):")
        for reason, count in reject_reasons.most_common():
            print(f"  {reason}: {count}")

    if ok_reports:
        durs = [r.duration_sec for r in ok_reports]
        poses = [r.pose_frames for r in ok_reports]
        print(
            f"\nAceptables — duración media: {sum(durs)/len(durs):.2f}s | "
            f"frames pose medios: {sum(poses)/len(poses):.1f}"
        )


def write_csv(path: Path, reports: List[ClipReport], root: Path, only_ok: bool) -> None:
    rows = [r for r in reports if r.ok or not only_ok]
    fields = [
        "path", "category", "clip_name", "ok", "duration_sec", "fps",
        "track_id", "pose_frames", "valid_frames", "pose_duration_sec", "reasons",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = r.as_dict(root)
            d["reasons"] = "; ".join(r.reasons)
            w.writerow({k: d.get(k) for k in fields})


def write_json(path: Path, reports: List[ClipReport], root: Path) -> None:
    payload = {
        "root": str(root.resolve()),
        "total": len(reports),
        "acceptable": sum(1 for r in reports if r.ok),
        "clips": [r.as_dict(root) for r in reports],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Escanea data_result y lista clips con calidad aceptable (1 usuario, 5–10 s, poses coherentes)."
    )
    parser.add_argument(
        "root",
        type=str,
        help="Carpeta raíz (p. ej. data_result)",
    )
    parser.add_argument("--min-sec", type=float, default=DEFAULT_MIN_SEC, help="Duración mínima (default 5)")
    parser.add_argument("--max-sec", type=float, default=DEFAULT_MAX_SEC, help="Duración máxima (default 10)")
    parser.add_argument(
        "--pose-tolerance",
        type=float,
        default=DEFAULT_POSE_TOLERANCE,
        help="Tolerancia relativa en frames vs duración×fps (default 0.15)",
    )
    parser.add_argument(
        "--min-valid-pct",
        type=float,
        default=0.0,
        help="Mínimo valid_pct del usuario (0 = no filtrar; p. ej. 20)",
    )
    parser.add_argument(
        "--pose-source",
        choices=["filtered", "full"],
        default="filtered",
        help="Preferir poses.npy (filtered) o poses_full.npy (full)",
    )
    parser.add_argument(
        "--list-ok",
        action="store_true",
        help="Imprimir ruta de cada clip aceptable",
    )
    parser.add_argument("--csv", type=str, default=None, help="Exportar resultados a CSV")
    parser.add_argument("--json", type=str, default=None, help="Exportar informe completo a JSON")
    parser.add_argument(
        "--csv-all",
        action="store_true",
        help="Con --csv, incluir también clips rechazados (default: solo OK)",
    )
    args = parser.parse_args()

    if args.min_sec <= 0 or args.max_sec <= 0 or args.min_sec > args.max_sec:
        print("min-sec debe ser > 0 y <= max-sec", file=sys.stderr)
        return 2

    root = Path(args.root).expanduser().resolve()
    try:
        reports = scan_root(
            root,
            min_sec=float(args.min_sec),
            max_sec=float(args.max_sec),
            pose_tolerance=float(args.pose_tolerance),
            pose_source=args.pose_source,
            min_valid_pct=float(args.min_valid_pct),
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not reports:
        print(f"No se encontraron meta.json bajo {root}")
        return 0

    print_summary(reports, root)

    if args.list_ok:
        print("\nClips aceptables:")
        for r in sorted(reports, key=lambda x: str(x.path)):
            if not r.ok:
                continue
            rel = r.as_dict(root)["path"]
            print(
                f"  {rel} | {r.duration_sec:.1f}s | pose={r.pose_frames} frames "
                f"({r.pose_duration_sec:.1f}s) | user_{r.track_id}"
            )

    if args.csv:
        write_csv(Path(args.csv), reports, root, only_ok=not args.csv_all)

    if args.json:
        write_json(Path(args.json), reports, root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
