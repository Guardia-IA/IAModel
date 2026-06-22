#!/usr/bin/env python3
"""
Escanea una carpeta tipo data_result y lista clips con calidad aceptable.

Filtro por defecto (laxo):
  - un único usuario en meta.json
  - duración del clip entre min_sec y max_sec (default 5–10 s)

Con --strict añade comprobaciones de poses.npy coherentes con la duración.

Estructura esperada (recursiva):
  .../{cat}/{clip_name}/meta.json

Uso:
  python scan_data_result_quality.py /ruta/a/data_result
  python scan_data_result_quality.py /ruta/a/data_result --list-ok
  python scan_data_result_quality.py /ruta/a/data_result --csv ok_clips.csv
  python scan_data_result_quality.py /ruta/a/data_result --strict
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

DEFAULT_MIN_SEC = 5.0
DEFAULT_MAX_SEC = 10.0
DEFAULT_FPS = 12.5
DEFAULT_POSE_TOLERANCE = 0.15


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


def _expected_frame_bounds(duration_sec: float, fps: float, tolerance: float) -> tuple[int, int]:
    center = duration_sec * fps
    lo = int(max(1, center * (1.0 - tolerance)))
    hi = int(max(lo, center * (1.0 + tolerance)))
    return lo, hi


def _find_pose_file(user_dir: Path, pose_source: str) -> Optional[Path]:
    if pose_source == "full":
        names = ("poses_full.npy", "poses.npy")
    else:
        names = ("poses.npy", "poses_full.npy")
    for name in names:
        p = user_dir / name
        if p.is_file():
            return p
    return None


def _check_pose_strict(
    report: ClipReport,
    clip_dir: Path,
    user: Dict[str, Any],
    *,
    min_sec: float,
    max_sec: float,
    pose_tolerance: float,
    pose_source: str,
    min_valid_pct: float,
) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        report.reasons.append("numpy_required_for_strict")
        raise exc

    track_id = user.get("track_id")
    if track_id is None:
        report.reasons.append("missing_track_id")
        return
    report.track_id = int(track_id)

    pose_path = _find_pose_file(clip_dir / f"user_{track_id}", pose_source)
    if pose_path is None:
        report.reasons.append("missing_poses_npy")
        return

    try:
        poses = np.load(pose_path)
    except Exception:
        report.reasons.append("invalid_poses_npy")
        return

    if poses.ndim != 3 or poses.shape[-1] != 2:
        report.reasons.append(f"bad_poses_shape {getattr(poses, 'shape', None)}")
        return

    pose_len = int(poses.shape[0])
    report.pose_frames = pose_len
    report.pose_duration_sec = pose_len / report.fps if report.fps > 0 else 0.0

    total_frames = _to_int(user.get("total_frames"), default=pose_len)
    valid_frames = _to_int(user.get("valid_frames"), default=pose_len)
    report.valid_frames = valid_frames
    if total_frames <= 0:
        total_frames = pose_len

    if abs(pose_len - total_frames) > max(2, int(total_frames * pose_tolerance)):
        report.reasons.append(
            f"pose_meta_frame_mismatch (poses={pose_len}, meta total_frames={total_frames})"
        )

    lo, hi = _expected_frame_bounds(report.duration_sec, report.fps, pose_tolerance)
    if pose_len < lo or pose_len > hi:
        report.reasons.append(
            f"pose_frames_out_of_range ({pose_len} not in [{lo}, {hi}])"
        )

    if report.pose_duration_sec < min_sec or report.pose_duration_sec > max_sec:
        report.reasons.append(
            f"pose_duration_out_of_range ({report.pose_duration_sec:.2f}s not in [{min_sec}, {max_sec}])"
        )

    if total_frames > 0 and valid_frames > 0 and min_valid_pct > 0:
        valid_pct = 100.0 * valid_frames / total_frames
        if valid_pct < min_valid_pct:
            report.reasons.append(f"low_valid_pct ({valid_pct:.1f}% < {min_valid_pct}%)")


def evaluate_clip(
    clip_dir: Path,
    root: Path,
    *,
    min_sec: float,
    max_sec: float,
    strict: bool,
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

    report.clip_name = str(meta.get("clip_name", clip_name))
    report.duration_sec = _to_float(meta.get("clip_duration"), default=0.0)
    report.fps = _to_float(meta.get("fps"), default=DEFAULT_FPS)
    if report.fps <= 0:
        report.fps = DEFAULT_FPS

    if report.duration_sec < min_sec or report.duration_sec > max_sec:
        report.reasons.append(
            f"duration_out_of_range ({report.duration_sec:.2f}s not in [{min_sec}, {max_sec}])"
        )

    users = meta.get("users") or []
    report.n_users = len(users)
    if len(users) != 1:
        report.reasons.append(f"multi_user (n={len(users)})")
    else:
        user = users[0]
        track_id = user.get("track_id")
        if track_id is not None:
            report.track_id = int(track_id)
        report.valid_frames = _to_int(user.get("valid_frames"), default=0)
        if strict:
            _check_pose_strict(
                report,
                clip_dir,
                user,
                min_sec=min_sec,
                max_sec=max_sec,
                pose_tolerance=pose_tolerance,
                pose_source=pose_source,
                min_valid_pct=min_valid_pct,
            )

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
    strict: bool,
    pose_tolerance: float,
    pose_source: str,
    min_valid_pct: float,
) -> List[ClipReport]:
    if not root.is_dir():
        raise FileNotFoundError(f"No es una carpeta: {root}")
    return [
        evaluate_clip(
            clip_dir,
            root,
            min_sec=min_sec,
            max_sec=max_sec,
            strict=strict,
            pose_tolerance=pose_tolerance,
            pose_source=pose_source,
            min_valid_pct=min_valid_pct,
        )
        for clip_dir in iter_clip_dirs(root)
    ]


def _reason_bucket(reason: str) -> str:
    return reason.split(" ", 1)[0].split("(", 1)[0]


def print_summary(reports: List[ClipReport], root: Path, *, strict: bool) -> None:
    total = len(reports)
    ok_reports = [r for r in reports if r.ok]
    n_ok = len(ok_reports)
    pct = 100.0 * n_ok / total if total else 0.0

    mode = "estricto (duración + 1 usuario + poses)" if strict else "laxo (duración + 1 usuario)"
    print(f"\nRaíz: {root.resolve()}")
    print(f"Modo: {mode}")
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
        print(f"\nAceptables — duración media: {sum(durs) / len(durs):.2f}s")


def write_csv(path: Path, reports: List[ClipReport], root: Path, only_ok: bool) -> None:
    rows = [r for r in reports if r.ok or not only_ok]
    fields = [
        "path", "category", "clip_name", "ok", "duration_sec", "fps",
        "n_users", "track_id", "valid_frames", "reasons",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = r.as_dict(root)
            d["reasons"] = "; ".join(r.reasons)
            w.writerow({k: d.get(k) for k in fields})


def write_json(path: Path, reports: List[ClipReport], root: Path, *, strict: bool) -> None:
    payload = {
        "root": str(root.resolve()),
        "mode": "strict" if strict else "lax",
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
        description="Escanea data_result: por defecto 1 usuario y duración 5–10 s (filtro laxo)."
    )
    parser.add_argument("root", type=str, help="Carpeta raíz (p. ej. data_result)")
    parser.add_argument("--min-sec", type=float, default=DEFAULT_MIN_SEC, help="Duración mínima (default 5)")
    parser.add_argument("--max-sec", type=float, default=DEFAULT_MAX_SEC, help="Duración máxima (default 10)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Añadir comprobaciones de poses.npy coherentes con la duración",
    )
    parser.add_argument(
        "--pose-tolerance",
        type=float,
        default=DEFAULT_POSE_TOLERANCE,
        help="Solo con --strict: tolerancia en frames vs duración×fps (default 0.15)",
    )
    parser.add_argument(
        "--min-valid-pct",
        type=float,
        default=0.0,
        help="Solo con --strict: mínimo valid_pct (0 = no filtrar)",
    )
    parser.add_argument(
        "--pose-source",
        choices=["filtered", "full"],
        default="filtered",
        help="Solo con --strict: poses.npy o poses_full.npy",
    )
    parser.add_argument("--list-ok", action="store_true", help="Imprimir ruta de cada clip aceptable")
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
            strict=bool(args.strict),
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

    print_summary(reports, root, strict=bool(args.strict))

    if args.list_ok:
        print("\nClips aceptables:")
        for r in sorted(reports, key=lambda x: str(x.path)):
            if not r.ok:
                continue
            rel = r.as_dict(root)["path"]
            extra = f" | user_{r.track_id}" if r.track_id is not None else ""
            print(f"  {rel} | {r.duration_sec:.1f}s{extra}")

    if args.csv:
        write_csv(Path(args.csv), reports, root, only_ok=not args.csv_all)

    if args.json:
        write_json(Path(args.json), reports, root, strict=bool(args.strict))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
