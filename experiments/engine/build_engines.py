#!/usr/bin/env python3
"""
Detecta la GPU y genera ficheros .engine (TensorRT) para modelos YOLO pose.

Los .engine son específicos de la GPU y aceleran la inferencia ~2-5x respecto a .pt.
Hay que regenerarlos si cambias de GPU o de versión de CUDA/TensorRT.

Uso:
    python build_engines.py
    python build_engines.py --model yolo11x-pose.pt
    python build_engines.py --models yolo11n-pose.pt yolo11m-pose.pt --half
"""
import argparse
import sys
from pathlib import Path

# Añadir padre para imports si hace falta
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def detect_gpu() -> dict:
    """Detecta la GPU NVIDIA disponible y devuelve info."""
    info = {"cuda_available": False, "device_name": None, "device_id": 0}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_id"] = 0
    except ImportError:
        pass
    return info


def build_engine(
    model_path: str,
    output_dir: Path | None = None,
    device: int = 0,
    half: bool = True,
    imgsz: int = 640,
    verbose: bool = True,
) -> Path | None:
    """
    Exporta un modelo YOLO a TensorRT .engine.
    Devuelve la ruta del .engine generado o None si falló.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: necesita 'ultralytics'. Ejecuta: pip install ultralytics")
        return None

    model_path = Path(model_path)
    if not model_path.exists():
        # Descargar automáticamente si es un nombre conocido
        model_path = str(model_path)
    else:
        model_path = str(model_path.resolve())

    out_dir = output_dir or Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Cargando {model_path}...")
    model = YOLO(model_path)

    if verbose:
        print(f"  Exportando a TensorRT (FP16={half}, imgsz={imgsz})...")
    try:
        engine_path = model.export(
            format="engine",
            half=half,
            imgsz=imgsz,
            device=device,
            verbose=verbose,
            workspace=4,
        )
        engine_path = Path(engine_path)
        dest = out_dir / engine_path.name
        if engine_path.resolve() != dest.resolve():
            import shutil
            shutil.copy2(str(engine_path), str(dest))
        return dest
    except Exception as e:
        print(f"  [ERROR] Fallo al exportar: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Genera .engine (TensorRT) para YOLO pose en la GPU detectada"
    )
    parser.add_argument(
        "--model", "--models",
        dest="models",
        nargs="+",
        default=["yolo11n-pose.pt", "yolo11m-pose.pt"],
        help="Modelo(s) a exportar (ej: yolo11n-pose.pt yolo11x-pose.pt)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Carpeta donde guardar los .engine",
    )
    parser.add_argument("--half", action="store_true", default=True, help="Usar FP16 (por defecto: True)")
    parser.add_argument("--no-half", action="store_true", help="Usar FP32 en lugar de FP16")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de entrada (default 640)")
    parser.add_argument("-d", "--device", type=int, default=0, help="ID de GPU (default 0)")
    args = parser.parse_args()

    gpu = detect_gpu()
    if not gpu["cuda_available"]:
        print("ERROR: No se detectó GPU NVIDIA con CUDA.")
        print("Los .engine solo se pueden generar en máquinas con GPU NVIDIA.")
        sys.exit(1)

    print(f"GPU detectada: {gpu['device_name']}")
    print(f"Salida: {args.output}")
    print()

    args.output.mkdir(parents=True, exist_ok=True)
    half = args.half and not args.no_half

    built = []
    for m in args.models:
        print(f"[{m}]")
        p = build_engine(
            m,
            output_dir=args.output,
            device=args.device,
            half=half,
            imgsz=args.imgsz,
        )
        if p:
            built.append(p)
            print(f"  -> {p}\n")
        else:
            print(f"  -> falló\n")

    if built:
        print("=" * 50)
        print(f"Generados {len(built)} .engine en {args.output}")
        print("Usa el .engine en pose_extractor_clean.py cambiando MODEL_PATH.")
    else:
        print("No se generó ningún .engine.")
        sys.exit(1)


if __name__ == "__main__":
    main()
