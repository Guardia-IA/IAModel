#!/usr/bin/env python3
"""
Convierte el checkpoint del CLASIFICADOR de poses (.pt) a un .engine de TensorRT.

NO uses engine/build_engines.py para esto: aquel es solo para modelos YOLO-pose
(entrada de imagen 640x640). El clasificador es un TCN propio cuya entrada es
[1, seq_len, input_dim] (poses), asi que hay que exportarlo a ONNX y de ahi
construir el engine.

Flujo:
    .pt (checkpoint) --torch.onnx.export--> .onnx --TensorRT--> .engine

El .engine resultante es compatible con el runtime TensorRTClassifier de
logits_extraction.py (entrada [1, seq_len, input_dim] float32, salida [1, num_classes]).

Requisitos (en la maquina/cloud con GPU NVIDIA):
    - torch
    - tensorrt
    - El paquete con las clases del modelo (train_model_operations.py), localizable
      via --training-dir (o variable de entorno IA_TRAINING_DIR).

Uso tipico:
    python build_engine_modelo_54.py \
        --model /home/debian/Documentos/UGR/modelosaux/modelo_54.pt \
        --training-dir /ruta/al/repo/experiments/training \
        --fp32 --verify

Notas:
    - Por defecto FP32 (logits identicos al .pt, ideal para analisis de umbrales).
      Usa --fp16 para un engine mas rapido (puede variar ligeramente los logits).
    - Batch fijo a 1 (la inferencia es por usuario), shapes estaticas.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch

_THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convierte modelo_54.pt (clasificador) a .engine TensorRT.")
    p.add_argument(
        "--model",
        default=str(_THIS_DIR / "modelo_54.pt"),
        help="Checkpoint .pt del clasificador (por defecto modelo_54.pt junto a este script).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Ruta del .engine de salida (por defecto: mismo nombre que el .pt con extension .engine).",
    )
    p.add_argument(
        "--training-dir",
        default=os.environ.get("IA_TRAINING_DIR", ""),
        help=(
            "Carpeta que contiene train_model_operations.py (o train_model.py) para "
            "reconstruir la arquitectura. Tambien se intenta autodetectar."
        ),
    )
    p.add_argument("--fp16", action="store_true", help="Construir engine en FP16 (mas rapido).")
    p.add_argument("--fp32", action="store_true", help="Forzar FP32 (por defecto si no se indica nada).")
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="Opset de ONNX para la exportacion (por defecto 17).",
    )
    p.add_argument(
        "--workspace-mb",
        type=int,
        default=2048,
        help="Memoria de trabajo del builder de TensorRT en MB (por defecto 2048).",
    )
    p.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Conservar el .onnx intermedio (por defecto se borra).",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Tras construir, compara logits del engine vs .pt con una entrada aleatoria.",
    )
    return p.parse_args()


def _add_training_dir_to_path(training_dir: str) -> None:
    """Anade a sys.path la carpeta con train_model_operations.py."""
    candidates: List[Path] = []
    if training_dir:
        candidates.append(Path(training_dir).expanduser().resolve())
    candidates += [
        Path("/home/debian/dev/Proyectos/GuadIA-IAModel/IAModel/experiments/training"),
        Path("/home/debian/dev/Proyectos/GuadIA-IAModel/IAModel/experiments"),
        _THIS_DIR,
    ]
    for c in candidates:
        if c.is_dir() and (
            (c / "train_model_operations.py").exists() or (c / "train_model.py").exists()
        ):
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            if str(c.parent) not in sys.path:
                sys.path.insert(0, str(c.parent))
            return
    raise SystemExit(
        "No encuentro train_model_operations.py / train_model.py. "
        "Indica la carpeta con --training-dir (o IA_TRAINING_DIR)."
    )


def _import_build_model():
    try:
        from train_model_operations import build_model  # type: ignore[attr-defined]
        return build_model
    except ImportError:
        from train_model import build_model  # type: ignore[attr-defined]
        return build_model


def build_torch_model(checkpoint: Dict[str, Any]) -> torch.nn.Module:
    build_model = _import_build_model()
    cfg = checkpoint.get("config", {})
    arch = cfg.get("arch", "tcn")
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint.get("num_classes", len(checkpoint["label_to_idx"])))
    model = build_model(arch, input_dim, num_classes, cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def export_onnx(model: torch.nn.Module, seq_len: int, input_dim: int, onnx_path: Path, opset: int) -> None:
    """
    Exporta a ONNX. Preferimos el exportador *legacy* (TorchScript, dynamo=False):
    mete los pesos inline en un solo .onnx (sin .onnx.data externo) y respeta el
    opset, que es lo que TensorRT parsea sin problemas. Si fallara, caemos al
    exportador dynamo (que puede generar pesos externos; eso se maneja luego con
    parser.parse_from_file).
    """
    dummy = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    # 1) Exportador legacy (recomendado para TensorRT).
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["poses"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
        print(f"[OK] ONNX exportado (exportador legacy, pesos inline): {onnx_path}")
        return
    except TypeError:
        # torch antiguo sin el parametro 'dynamo': comportamiento legacy por defecto.
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["poses"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )
        print(f"[OK] ONNX exportado: {onnx_path}")
        return
    except Exception as e:  # noqa: BLE001 - fallback intencionado al exportador dynamo
        print(f"[WARN] El exportador legacy fallo ({e}). Pruebo el exportador dynamo...")

    # 2) Exportador dynamo (puede crear <onnx>.data; opset >=18 para evitar la
    #    conversion fallida a versiones anteriores).
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["poses"],
        output_names=["logits"],
        opset_version=max(int(opset), 18),
        dynamo=True,
    )
    print(f"[OK] ONNX exportado (exportador dynamo): {onnx_path}")


def build_engine_from_onnx(onnx_path: Path, engine_path: Path, fp16: bool, workspace_mb: int) -> None:
    import tensorrt as trt  # type: ignore[import-not-found]

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    # parse_from_file resuelve pesos externos (.onnx.data) relativos a la ruta del
    # .onnx; parse(bytes) los buscaria en el cwd y fallaria.
    if hasattr(parser, "parse_from_file"):
        ok = parser.parse_from_file(str(onnx_path))
    else:
        with open(onnx_path, "rb") as f:
            ok = parser.parse(f.read())
    if not ok:
        msgs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise SystemExit("Fallo al parsear el ONNX:\n" + "\n".join(msgs))

    config = builder.create_builder_config()
    ws_bytes = int(workspace_mb) * 1024 * 1024
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_bytes)
    except AttributeError:
        config.max_workspace_size = ws_bytes  # type: ignore[attr-defined]

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 habilitado.")
        else:
            print("[WARN] La plataforma no acelera FP16; se construye en FP32.")

    print("[INFO] Construyendo engine (puede tardar)...")
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise SystemExit("TensorRT no pudo construir el engine (build_serialized_network devolvio None).")
        with open(engine_path, "wb") as f:
            f.write(serialized)
    else:
        engine = builder.build_engine(network, config)  # type: ignore[attr-defined]
        if engine is None:
            raise SystemExit("TensorRT no pudo construir el engine (build_engine devolvio None).")
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
    print(f"[OK] Engine escrito: {engine_path}")


def verify_engine(engine_path: Path, model: torch.nn.Module, seq_len: int, input_dim: int) -> None:
    import numpy as np
    import tensorrt as trt  # type: ignore[import-not-found]

    if not torch.cuda.is_available():
        print("[VERIFY] No hay GPU CUDA disponible; me salto la verificacion del engine.")
        return
    device = torch.device("cuda")

    x = torch.randn(1, seq_len, input_dim, dtype=torch.float32)
    with torch.no_grad():
        ref = model(x).detach().cpu().numpy().reshape(-1)

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    in_name = out_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            in_name = name
        else:
            out_name = name

    # Buffers CUDA con PyTorch (sin pycuda).
    x_cuda = x.to(device, dtype=torch.float32).contiguous()
    context.set_input_shape(in_name, tuple(x_cuda.shape))
    out_shape = tuple(context.get_tensor_shape(out_name))
    out_cuda = torch.empty(out_shape, dtype=torch.float32, device=device)
    context.set_tensor_address(in_name, int(x_cuda.data_ptr()))
    context.set_tensor_address(out_name, int(out_cuda.data_ptr()))
    stream = torch.cuda.current_stream(device)
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    eng = out_cuda.detach().cpu().numpy().reshape(-1)
    max_abs = float(np.max(np.abs(eng - ref)))
    print(f"[VERIFY] logits .pt    : {np.round(ref, 4)}")
    print(f"[VERIFY] logits engine : {np.round(eng, 4)}")
    msg = "OK" if max_abs < 1e-2 else "REVISAR (FP16 puede explicar diferencias)"
    print(f"[VERIFY] max |diff| = {max_abs:.6f} ({msg})")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"No existe el checkpoint: {model_path}")
    if model_path.suffix != ".pt":
        raise SystemExit(f"Se esperaba un .pt del clasificador: {model_path}")

    engine_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else model_path.with_suffix(".engine")
    )
    fp16 = bool(args.fp16) and not bool(args.fp32)

    _add_training_dir_to_path(args.training_dir)

    print(f"[INFO] Checkpoint: {model_path}")
    print(f"[INFO] Salida engine: {engine_path}")
    print(f"[INFO] Precision: {'FP16' if fp16 else 'FP32'}")

    checkpoint = torch.load(str(model_path), map_location="cpu")
    seq_len = int(checkpoint.get("seq_len", 64))
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint.get("num_classes", len(checkpoint["label_to_idx"])))
    arch = checkpoint.get("config", {}).get("arch", "tcn")
    print(
        f"[INFO] arch={arch} | seq_len={seq_len} | input_dim={input_dim} | "
        f"num_classes={num_classes} | task={checkpoint.get('task', '?')}"
    )

    model = build_torch_model(checkpoint)

    tmp_onnx = Path(tempfile.mkstemp(prefix="modelo_54_", suffix=".onnx")[1])
    onnx_path = model_path.with_suffix(".onnx") if args.keep_onnx else tmp_onnx
    try:
        export_onnx(model, seq_len, input_dim, onnx_path, args.opset)
        build_engine_from_onnx(onnx_path, engine_path, fp16, args.workspace_mb)
        if args.verify:
            verify_engine(engine_path, model, seq_len, input_dim)
    finally:
        if not args.keep_onnx:
            for p in (onnx_path, Path(str(onnx_path) + ".data")):
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass

    print(f"[FIN] Engine listo: {engine_path}")


if __name__ == "__main__":
    main()
