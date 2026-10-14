#!/usr/bin/env python3
"""
Exporta un modelo YOLO pose a TensorRT (.engine) en un directorio del host.

Pensado para un único artefacto compartido entre contenedores (C1, extracción IAModel, etc.).

Requisitos: GPU NVIDIA + CUDA + TensorRT + ultralytics (ver engine/requirements.txt).

Ejemplos:
  python export_yolo_pose_engine.py \\
    --output-dir /opt/kanvis/models/yolo26s-pose/v1 \\
    --yolo yolo26s-pose

  python export_yolo_pose_engine.py \\
    -o ./engine \\
    --yolo yolo26s-pose.pt \\
    --imgsz 640 --device 0

Salida:
  <output-dir>/<stem>.engine
  <output-dir>/<stem>.engine.manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Reutiliza la lógica de export del repo
_ENGINE_DIR = Path(__file__).resolve().parent
if str(_ENGINE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR.parent))

from engine.build_engines import build_engine, detect_gpu  # noqa: E402


def _normalize_yolo_arg(yolo: str) -> tuple[str, str]:
    """Devuelve (ruta o nombre para YOLO, stem sin extensión)."""
    yolo = yolo.strip()
    if not yolo:
        raise ValueError("--yolo no puede estar vacío")
    p = Path(yolo)
    if p.suffix.lower() in (".pt", ".yaml"):
        stem = p.stem
        return yolo, stem
    # "yolo26s-pose" → yolo26s-pose.pt
    name = yolo if yolo.endswith(".pt") else f"{yolo}.pt"
    return name, Path(name).stem


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_versions() -> dict:
    out: dict = {"python": platform.python_version()}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        out["torch"] = None
    try:
        import ultralytics

        out["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
    except ImportError:
        out["ultralytics"] = None
    try:
        import tensorrt as trt

        out["tensorrt"] = trt.__version__
    except Exception:
        out["tensorrt"] = None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera yolo26s-pose.engine (u otro YOLO pose) en un directorio del host"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directorio de salida en el host (se crea si no existe)",
    )
    parser.add_argument(
        "--yolo",
        required=True,
        help='Versión/modelo, p. ej. "yolo26s-pose" o ruta a "yolo26s-pose.pt"',
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de entrada (default 640)")
    parser.add_argument("-d", "--device", type=int, default=0, help="GPU id (default 0)")
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="FP16 (default)",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="FP32 en lugar de FP16",
    )
    parser.add_argument(
        "--engine-name",
        type=str,
        default=None,
        help="Nombre del .engine (default: <stem>.engine)",
    )
    args = parser.parse_args()

    gpu = detect_gpu()
    if not gpu["cuda_available"]:
        print("ERROR: se necesita GPU NVIDIA con CUDA para exportar .engine", file=sys.stderr)
        return 1

    model_ref, stem = _normalize_yolo_arg(args.yolo)
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    half = args.half and not args.no_half
    print(f"GPU: {gpu['device_name']}")
    print(f"Modelo: {model_ref}")
    print(f"Salida: {out_dir}")
    print(f"FP16: {half}, imgsz: {args.imgsz}, device: {args.device}")
    print()

    built = build_engine(
        model_ref,
        output_dir=out_dir,
        device=args.device,
        half=half,
        imgsz=args.imgsz,
        verbose=True,
    )
    if built is None:
        return 1

    engine_path = Path(built)
    dest_name = args.engine_name or f"{stem}.engine"
    if not dest_name.endswith(".engine"):
        dest_name = f"{dest_name}.engine"
    final_engine = out_dir / dest_name
    if engine_path.resolve() != final_engine.resolve():
        shutil.move(str(engine_path), str(final_engine))
        engine_path = final_engine

    manifest_path = final_engine.with_suffix(final_engine.suffix + ".manifest.json")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "yolo_model_ref": model_ref,
        "engine_file": final_engine.name,
        "engine_path": str(final_engine),
        "engine_sha256": _sha256_file(final_engine),
        "export": {
            "format": "tensorrt",
            "half": half,
            "imgsz": args.imgsz,
            "device_id": args.device,
            "workspace_gb": 4,
        },
        "host": {
            "hostname": platform.node(),
            "gpu_detected": gpu,
        },
        "versions": _collect_versions(),
        "notes": (
            "Engine ligado a GPU/CUDA/TensorRT de esta máquina. "
            "Montar solo lectura en contenedores; no recompilar en cada deploy."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print("=" * 60)
    print(f"Engine:   {final_engine}")
    print(f"Manifest: {manifest_path}")
    print(f"SHA256:   {manifest['engine_sha256']}")
    print()
    print("Montaje Docker sugerido (host → contenedor):")
    print(f"  {out_dir}:/models/yolo-pose:ro")
    print("Variable sugerida:")
    print(f"  YOLO_POSE_ENGINE=/models/yolo-pose/{final_engine.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
