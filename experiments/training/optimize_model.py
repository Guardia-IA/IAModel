import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

try:
    from .train_model_operations import build_model  # type: ignore[attr-defined]
except ImportError:
    try:
        from train_model_operations import build_model  # type: ignore[attr-defined]
    except ImportError:
        try:
            from .train_model import build_model  # type: ignore[attr-defined]
        except ImportError:
            from train_model import build_model  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimiza un checkpoint .pt para inferencia con los pasos disponibles en el equipo. "
            "Siempre funciona en CPU y deja preparados pasos de GPU para cuando haya hardware."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Ruta al checkpoint de entrada (.pt).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        type=str,
        help="Carpeta de salida (default: output).",
    )
    parser.add_argument(
        "--final-name",
        default="optimized_model.pt",
        type=str,
        help="Nombre del único archivo final optimizado.",
    )
    parser.add_argument(
        "--prune-amount",
        default=0.10,
        type=float,
        help="Ratio de poda global no estructurada (0..1). Default: 0.10",
    )
    parser.add_argument(
        "--disable-prune",
        action="store_true",
        help="Desactiva la poda.",
    )
    parser.add_argument(
        "--disable-int8",
        action="store_true",
        help="Desactiva cuantización dinámica INT8.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Conserva archivos intermedios de candidatos.",
    )
    parser.add_argument(
        "--try-compile",
        action="store_true",
        help="Intenta torch.compile (solo benchmark en sesión; no guarda artefacto portable).",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("El archivo .pt no contiene un checkpoint tipo dict esperado.")
    required = ["model_state_dict", "input_dim"]
    missing = [k for k in required if k not in checkpoint]
    if missing:
        raise RuntimeError(f"Checkpoint inválido, faltan claves: {missing}")
    return checkpoint


def build_from_checkpoint(checkpoint: Dict[str, Any]) -> Tuple[nn.Module, int, int]:
    cfg = checkpoint.get("config", {})
    arch = cfg.get("arch", "tcn")
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint.get("num_classes", len(checkpoint.get("label_to_idx", {})) or 2))
    seq_len = int(checkpoint.get("seq_len", 64))

    model = build_model(arch, input_dim, num_classes, cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, seq_len, input_dim


def save_scripted_model(model: nn.Module, seq_len: int, input_dim: int, out_path: Path) -> str:
    example = torch.randn(1, seq_len, input_dim)
    traced = torch.jit.trace(model, example, strict=False)
    frozen = torch.jit.freeze(traced)
    optimized = torch.jit.optimize_for_inference(frozen)
    optimized.save(str(out_path))
    return str(out_path)


def try_dynamic_quant(model: nn.Module) -> nn.Module:
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},
        dtype=torch.qint8,
    )


def apply_global_pruning(model: nn.Module, amount: float) -> int:
    amount = float(max(0.0, min(1.0, amount)))
    to_prune = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            to_prune.append((m, "weight"))
    if not to_prune or amount <= 0.0:
        return 0
    prune.global_unstructured(
        to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, _name in to_prune:
        prune.remove(module, "weight")
    return len(to_prune)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = output_dir / args.final_name
    intermediates_dir = output_dir / ".optimize_model_intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "input_model": str(model_path),
        "output_dir": str(output_dir),
        "final_model": str(final_model_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device_cuda_available": bool(torch.cuda.is_available()),
        "artifacts": {},
        "steps_applied": [],
        "steps_skipped": [],
        "warnings": [],
    }

    checkpoint = load_checkpoint(model_path)
    model, seq_len, input_dim = build_from_checkpoint(checkpoint)

    base_name = model_path.stem
    arch = checkpoint.get("config", {}).get("arch", "unknown")
    report["arch"] = arch
    report["seq_len"] = seq_len
    report["input_dim"] = input_dim
    report["requested_prune_amount"] = float(max(0.0, min(1.0, args.prune_amount)))

    candidates: Dict[str, Path] = {}

    # 1) Baseline TorchScript FP32
    try:
        out_ts_fp32 = intermediates_dir / f"{base_name}_torchscript_fp32.pt"
        saved = save_scripted_model(model, seq_len, input_dim, out_ts_fp32)
        report["artifacts"]["torchscript_fp32"] = saved
        report["steps_applied"].append("torchscript_fp32")
        candidates["torchscript_fp32"] = out_ts_fp32
        print(f"[OK] TorchScript FP32 guardado en: {saved}")
    except Exception as e:
        report["steps_skipped"].append("torchscript_fp32")
        report["warnings"].append(f"Falló TorchScript FP32: {e}")
        print(f"[WARN] No se pudo generar TorchScript FP32: {e}")

    # 2) Poda + export
    model_pruned = model
    if args.disable_prune:
        report["steps_skipped"].append("pruning")
    else:
        try:
            pruned_modules = apply_global_pruning(model_pruned, args.prune_amount)
            if pruned_modules > 0:
                out_ts_pruned = intermediates_dir / f"{base_name}_torchscript_pruned_fp32.pt"
                saved_p = save_scripted_model(model_pruned, seq_len, input_dim, out_ts_pruned)
                report["artifacts"]["torchscript_pruned_fp32"] = saved_p
                report["steps_applied"].append("pruning")
                report["steps_applied"].append("torchscript_pruned_fp32")
                report["pruned_modules"] = int(pruned_modules)
                candidates["torchscript_pruned_fp32"] = out_ts_pruned
                print(f"[OK] Poda + TorchScript FP32 guardado en: {saved_p}")
            else:
                report["steps_skipped"].append("pruning")
                report["warnings"].append("No se encontraron capas podables para pruning.")
        except Exception as e:
            report["steps_skipped"].append("pruning")
            report["warnings"].append(f"No se pudo aplicar poda: {e}")
            print(f"[WARN] Se omite poda: {e}")

    # 3) Cuantización dinámica INT8 en CPU (cuando aplica)
    if args.disable_int8:
        report["steps_skipped"].append("torchscript_dynamic_int8")
    else:
        try:
            model_q = try_dynamic_quant(model_pruned)
            out_ts_int8 = intermediates_dir / f"{base_name}_torchscript_dynamic_int8.pt"
            saved_q = save_scripted_model(model_q, seq_len, input_dim, out_ts_int8)
            report["artifacts"]["torchscript_dynamic_int8"] = saved_q
            report["steps_applied"].append("torchscript_dynamic_int8")
            candidates["torchscript_dynamic_int8"] = out_ts_int8
            print(f"[OK] TorchScript dinámico INT8 guardado en: {saved_q}")
        except Exception as e:
            report["steps_skipped"].append("torchscript_dynamic_int8")
            report["warnings"].append(f"No se pudo aplicar cuantización dinámica INT8: {e}")
            print(f"[WARN] Se omite cuantización dinámica INT8: {e}")

    # 4) torch.compile (si se solicita) - no se guarda artefacto; solo comprobación
    if args.try_compile:
        if hasattr(torch, "compile"):
            try:
                compiled = torch.compile(model)
                with torch.no_grad():
                    _ = compiled(torch.randn(1, seq_len, input_dim))
                report["steps_applied"].append("torch_compile_runtime_check")
                print("[OK] torch.compile ejecutado (runtime check).")
            except Exception as e:
                report["steps_skipped"].append("torch_compile_runtime_check")
                report["warnings"].append(f"torch.compile no disponible/estable en este entorno: {e}")
                print(f"[WARN] torch.compile no aplicable: {e}")
        else:
            report["steps_skipped"].append("torch_compile_runtime_check")
            report["warnings"].append("La versión de PyTorch no soporta torch.compile.")
            print("[WARN] torch.compile no está disponible en esta versión de PyTorch.")
    else:
        report["steps_skipped"].append("torch_compile_runtime_check")

    # 5) TensorRT (engine) - preparación informativa si no hay GPU
    if torch.cuda.is_available():
        report["warnings"].append(
            "GPU detectada: para TensorRT exporta a ONNX y compila engine (.engine) en este mismo equipo."
        )
    else:
        report["steps_skipped"].append("tensorrt_engine")
        report["warnings"].append(
            "No hay GPU CUDA en este equipo; se omite TensorRT (.engine). "
            "Puedes ejecutar ese paso en una máquina con NVIDIA."
        )
        print("[INFO] TensorRT omitido: no hay GPU CUDA en este equipo.")

    # 6) Destilación: requiere dataset + entrenamiento (no se puede one-shot desde un único .pt)
    report["steps_skipped"].append("distillation")
    report["warnings"].append(
        "Destilación omitida: requiere dataset de entrenamiento/validación y proceso de reentrenado "
        "(teacher->student), no es posible hacerla solo con un checkpoint."
    )

    # 7) Selección de único artefacto final
    if not candidates:
        raise RuntimeError("No se pudo generar ningún candidato optimizado.")
    preferred_order = [
        "torchscript_dynamic_int8",
        "torchscript_pruned_fp32",
        "torchscript_fp32",
    ]
    chosen_key = None
    for key in preferred_order:
        if key in candidates:
            chosen_key = key
            break
    if chosen_key is None:
        chosen_key = next(iter(candidates.keys()))
    chosen_path = candidates[chosen_key]
    shutil.copy2(chosen_path, final_model_path)
    report["selected_candidate"] = chosen_key
    report["selected_candidate_path"] = str(chosen_path)
    report["final_model_path"] = str(final_model_path)
    report["steps_applied"].append("single_final_artifact")
    print(f"[OK] Modelo final único guardado en: {final_model_path} (origen: {chosen_key})")

    if not args.keep_intermediates:
        for p in intermediates_dir.glob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass
        try:
            intermediates_dir.rmdir()
        except OSError:
            pass

    report_path = output_dir / f"{base_name}_optimization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] Reporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
