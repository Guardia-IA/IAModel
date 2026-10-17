#!/usr/bin/env python3
"""CLI IAModel → kanvis-triton-pose (extracción paralela por clip)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from triton_pose_extract.capacity import capacity_report  # noqa: E402
from triton_pose_extract.discovery import discover_videos  # noqa: E402
from triton_pose_extract.health import check_model_ready, print_health, smoke_infer_optional  # noqa: E402
from triton_pose_extract.parallel_extract import run_parallel_extract  # noqa: E402
from triton_pose_extract.paths import ensure_pose_core_importable, default_triton_pose_repo  # noqa: E402
from triton_pose_extract.settings import ExtractSettings  # noqa: E402
from triton_pose_extract.validate_output import validate_clip_dir  # noqa: E402
from triton_pose_extract.pipeline_config import load_pipeline_config  # noqa: E402


def _config_path(args: argparse.Namespace) -> Path | None:
    return Path(args.config) if getattr(args, "config", "") else None


def _resolve_input_roots(args: argparse.Namespace, raw: dict) -> list[Path]:
    roots: list[Path] = []
    for p in getattr(args, "input_root", None) or []:
        if p:
            roots.append(Path(p).expanduser().resolve())
    pipe = raw.get("pipeline") or {}
    for p in pipe.get("input_roots") or []:
        roots.append(Path(str(p)).expanduser().resolve())
    if pipe.get("input_root"):
        roots.append(Path(str(pipe["input_root"])).expanduser().resolve())
    seen: set[str] = set()
    uniq: list[Path] = []
    for r in roots:
        k = str(r)
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    if not uniq:
        print("ERROR: indica al menos una carpeta (--input-root DIR, repetible)", file=sys.stderr)
        raise SystemExit(2)
    return uniq


def _load_raw_config(args: argparse.Namespace) -> dict:
    p = _config_path(args)
    path = p or PKG / "triton_pose_extract_config.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _cmd_check(args: argparse.Namespace) -> int:
    settings = ExtractSettings.load(
        Path(args.config) if args.config else None,
        triton_url=args.triton_url or None,
        require_pose_repo=False,
    )
    code = print_health(settings)
    if code != 0:
        return code
    print(json.dumps({"capacity": capacity_report(settings)}, indent=2))
    if args.smoke:
        print(json.dumps({"smoke": smoke_infer_optional(settings)}, indent=2))
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    settings = ExtractSettings.load(
        Path(args.config) if args.config else None,
        triton_url=args.triton_url or None,
        parallel_clips=args.parallel if args.parallel > 0 else None,
        require_pose_repo=True,
    )
    ensure_pose_core_importable(settings.triton_pose_repo)

    if not args.skip_ready:
        ready = check_model_ready(settings)
        if not ready.get("ok"):
            print(f"ERROR: Triton/modelo no ready: {ready}", file=sys.stderr)
            return 2

    jobs = discover_videos(
        video=Path(args.video) if args.video else None,
        video_dir=Path(args.video_dir) if args.video_dir else None,
        cat=args.cat,
        limit=int(args.limit or 0),
    )
    if not jobs:
        print("ERROR: indica --video o --video-dir", file=sys.stderr)
        return 2

    cap = capacity_report(settings)
    print(
        f"Extracción: {len(jobs)} clip(s), parallel={args.parallel or cap['parallel_clips']}, "
        f"batch={settings.triton_batch_size}, url={settings.triton_url}",
        flush=True,
    )

    report = run_parallel_extract(
        settings,
        jobs,
        Path(args.out),
        parallel=args.parallel if args.parallel > 0 else None,
    )
    if args.validate:
        for row in report.get("results") or []:
            if not row.get("ok") or not row.get("out_dir"):
                continue
            errs = validate_clip_dir(Path(row["out_dir"]))
            if errs:
                print(f"VALIDATE WARN {row['out_dir']}: {errs}", file=sys.stderr)
            else:
                print(f"VALIDATE OK {row['out_dir']}", flush=True)

    return 0 if report.get("ok") else 1


def _cmd_preflight_csv(args: argparse.Namespace) -> int:
    from triton_pose_extract.preflight_csv import (
        print_preflight_report,
        run_preflight_report,
        save_preflight_json,
    )

    raw = _load_raw_config(args)
    ob = Path(args.output_base).expanduser() if args.output_base else None
    pipe = load_pipeline_config(raw, output_base=ob)
    input_roots = _resolve_input_roots(args, raw)
    print(f"Preflight: {len(input_roots)} carpeta(s), salida → {pipe.paths.data_result_dir}", flush=True)
    settings = ExtractSettings.load(
        _config_path(args),
        triton_url=args.triton_url or None,
        require_pose_repo=False,
    )
    report = run_preflight_report(
        input_roots,
        pipe,
        settings,
        probe_videos=bool(args.probe_videos),
        measure_video_duration=bool(getattr(args, "measure_video_duration", False)),
        verbose=not getattr(args, "quiet", False),
    )
    print_preflight_report(report)
    if args.save_json:
        save_preflight_json(report, Path(args.save_json).expanduser())
    inv = report["inventory"]
    if inv["total_clips"] <= 0:
        return 1
    return 0


def _cmd_extract_csv(args: argparse.Namespace) -> int:
    from triton_pose_extract.batch_csv import run_csv_batch

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    raw = _load_raw_config(args)
    ob = Path(args.output_base).expanduser() if args.output_base else None
    pipe = load_pipeline_config(raw, output_base=ob)
    input_roots = _resolve_input_roots(args, raw)

    settings = ExtractSettings.load(
        _config_path(args),
        triton_url=args.triton_url or None,
        parallel_clips=args.parallel if args.parallel > 0 else None,
        require_pose_repo=True,
    )
    settings.single_user_only = pipe.filters.single_user_only
    ensure_pose_core_importable(settings.triton_pose_repo)

    if not args.skip_ready:
        ready = check_model_ready(settings)
        if not ready.get("ok"):
            print(f"ERROR: Triton/modelo no ready: {ready}", file=sys.stderr)
            return 2

    report = run_csv_batch(
        input_roots,
        pipe,
        settings,
        parallel=args.parallel if args.parallel > 0 else None,
        limit=int(args.limit or 0),
    )
    return 0 if report.get("ok") else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="IAModel extracción pose vía Triton (kanvis-triton-pose)")
    sub = ap.add_subparsers(dest="command", required=True)

    chk = sub.add_parser("check-triton", help="HTTP ready + capacidad paralela")
    chk.add_argument("--config", type=str, default="", help="triton_pose_extract_config.json")
    chk.add_argument("--triton-url", type=str, default="", help="gRPC host:port")
    chk.add_argument("--smoke", action="store_true", help="check_triton_pose_batch.py (lento)")
    chk.set_defaults(func=_cmd_check)

    ext = sub.add_parser("extract", help="Extraer clips en paralelo")
    ext.add_argument("--config", type=str, default="", help="triton_pose_extract_config.json")
    ext.add_argument("--triton-url", type=str, default="", help="gRPC host:port")
    ext.add_argument("--video", type=str, default="")
    ext.add_argument("--video-dir", type=str, default="")
    ext.add_argument("--out", type=str, required=True)
    ext.add_argument("--cat", type=str, default=None)
    ext.add_argument("--limit", type=int, default=0)
    ext.add_argument("--parallel", type=int, default=0, help="Workers simultáneos (0=auto)")
    ext.add_argument("--skip-ready", action="store_true")
    ext.add_argument("--validate", action="store_true", help="Comprobar npy tras cada clip")
    ext.add_argument("-v", "--verbose", action="store_true")
    ext.set_defaults(func=_cmd_extract)

    pf = sub.add_parser("preflight-csv", help="Inventario CSV + tiempos Triton paralelo")
    pf.add_argument("--config", type=str, default="")
    pf.add_argument("--triton-url", type=str, default="")
    pf.add_argument(
        "--input-root",
        action="append",
        default=[],
        help="Carpeta con *.csv (repetible; busca recursivo). Ej: .../DatosOnline_3/sessions",
    )
    pf.add_argument("--output-base", type=str, default="", help="OUTPUT_BASE (data_result/ dentro)")
    pf.add_argument("--probe-videos", action="store_true", help="ffprobe en clips 00:00:00–00:00:00 (más lento)")
    pf.add_argument(
        "--measure-video-duration",
        action="store_true",
        help="OpenCV en cada vídeo (MUY lento; solo debug)",
    )
    pf.add_argument("--quiet", action="store_true", help="Solo resumen final")
    pf.add_argument("--save-json", type=str, default="", help="Guardar informe JSON")
    pf.set_defaults(func=_cmd_preflight_csv)

    excsv = sub.add_parser("extract-csv", help="Pipeline CSV → FFmpeg → Triton (como pose_extractor_clean)")
    excsv.add_argument("--config", type=str, default="")
    excsv.add_argument("--triton-url", type=str, default="")
    excsv.add_argument(
        "--input-root",
        action="append",
        default=[],
        help="Carpeta(s) con CSVs (repetible)",
    )
    excsv.add_argument("--output-base", type=str, default="")
    excsv.add_argument("--limit", type=int, default=0)
    excsv.add_argument("--parallel", type=int, default=0)
    excsv.add_argument("--skip-ready", action="store_true")
    excsv.add_argument("-v", "--verbose", action="store_true")
    excsv.set_defaults(func=_cmd_extract_csv)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
