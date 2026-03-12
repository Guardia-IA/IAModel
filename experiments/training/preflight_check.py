import sys
import platform
import importlib
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import json
import math

# Soportar ejecución como módulo (-m training.preflight_check) y como script (python training/preflight_check.py)
try:
    from .model_config import DATA_RESULT_ROOT, EXPERIMENTS  # type: ignore[attr-defined]
except ImportError:
    from model_config import DATA_RESULT_ROOT, EXPERIMENTS  # type: ignore[attr-defined]

# Intento de import opcional de tqdm para barras de progreso
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - entorno sin tqdm
    tqdm = None


# Colores ANSI para hacer la salida más clara
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"


TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
MIN_SEQ_LEN = 4  # debe coincidir con train_model


def color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def ok(msg: str) -> None:
    print(f"{color('[OK]', GREEN)} {msg}")


def fail(msg: str) -> None:
    print(f"{color('[X]', RED)} {msg}")


def warn(msg: str) -> None:
    print(f"{color('[!]', YELLOW)} {msg}")


def header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{BOLD}{line}\n{title}\n{line}{RESET}")


def check_python() -> None:
    header("1) Comprobación de entorno Python")
    v = sys.version.split()[0]
    ok(f"Versión de Python: {v}")


def check_module(name: str, import_name: str | None = None) -> Tuple[bool, Any]:
    mod_name = import_name or name
    try:
        module = importlib.import_module(mod_name)
        ok(f"Librería '{name}' disponible (import {mod_name})")
        return True, module
    except ImportError:
        fail(f"Librería '{name}' NO encontrada (import {mod_name} falló)")
        return False, None


def check_dependencies() -> Dict[str, Any]:
    header("2) Comprobación de dependencias")

    deps: Dict[str, Any] = {}

    # Núcleo numérico y DL
    for lib, imp in [
        ("numpy", "numpy"),
        ("torch", "torch"),
    ]:
        ok_, module = check_module(lib, imp)
        deps[lib] = module if ok_ else None

    # Utilidades recomendadas
    for lib, imp in [
        ("tqdm (barras de progreso)", "tqdm"),
        ("pandas (opcional)", "pandas"),
        ("opencv-python (cv2, opcional)", "cv2"),
    ]:
        name, imp_name = lib, imp
        try:
            module = importlib.import_module(imp_name)
            ok(f"Librería '{name}' disponible (import {imp_name})")
            deps[imp_name] = module
        except ImportError:
            warn(f"Librería '{name}' no encontrada. Solo necesaria para algunas utilidades.")

    return deps


def check_gpu(torch_mod) -> Dict[str, Any]:
    header("3) Dispositivo de cómputo (GPU/CPU)")

    info: Dict[str, Any] = {"device": "cpu", "gpus": []}

    if torch_mod is None:
        fail("PyTorch no está disponible. Solo se podrá usar CPU.")
        return info

    if not torch_mod.cuda.is_available():
        warn("CUDA NO disponible. Se utilizará CPU para el entrenamiento.")
        return info

    num_devices = torch_mod.cuda.device_count()
    for idx in range(num_devices):
        name = torch_mod.cuda.get_device_name(idx)
        cap = torch_mod.cuda.get_device_capability(idx)
        info["gpus"].append({"index": idx, "name": name, "capability": cap})

    if info["gpus"]:
        main_gpu = info["gpus"][0]
        info["device"] = f"cuda:{main_gpu['index']}"
        ok(
            f"CUDA disponible. GPU principal: {main_gpu['name']} "
            f"(compute capability {main_gpu['capability'][0]}.{main_gpu['capability'][1]})"
        )
        if num_devices > 1:
            warn(f"Se han detectado {num_devices} GPUs. Este script usará solo la GPU 0 para la estimación.")
    else:
        warn("No se ha podido enumerar GPUs, se utilizará CPU.")

    return info


def get_data_result_root() -> Path:
    root = DATA_RESULT_ROOT
    if not root.exists():
        raise RuntimeError(f"No se encontró la carpeta data_result en: {root}")
    return root


def scan_embeddings(pose_source: str = "filtered") -> Dict[str, Any]:
    header("4) Escaneo de embeddings en data_result")

    root = get_data_result_root()
    print(f"Carpeta de datos: {root}")

    total_examples = 0
    per_cat_counts: Dict[str, int] = {}
    per_cat_frames: Dict[str, List[int]] = {}

    cat_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    iterable = cat_dirs
    if tqdm is not None:
        iterable = tqdm(cat_dirs, desc="Recorriendo categorías", unit="cat")

    for cat_dir in iterable:
        cat_str = cat_dir.name  # e.g. "0", "1", ...
        per_cat_counts.setdefault(cat_str, 0)
        per_cat_frames.setdefault(cat_str, [])

        for clip_dir in sorted(cat_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            meta_path = clip_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            users = meta.get("users", [])
            if not users:
                continue

            chosen_user = None
            if cat_str != "6":
                # Solo clips con un único usuario
                if len(users) != 1:
                    continue
                chosen_user = users[0]
            else:
                # cat == 6: nos quedamos con el usuario con más frames
                chosen_user = max(users, key=lambda u: u.get("total_frames", 0))

            if not chosen_user:
                continue

            track_id = chosen_user.get("track_id")
            if track_id is None:
                continue

            user_dir = clip_dir / f"user_{track_id}"
            pose_filename = "poses.npy" if pose_source == "filtered" else "poses_full.npy"
            pose_path = user_dir / pose_filename
            if not pose_path.exists():
                continue

            try:
                poses = np.load(pose_path)
            except Exception:
                continue

            if poses.ndim != 3 or poses.shape[-1] != 2:
                continue
            if poses.shape[0] < MIN_SEQ_LEN:
                continue

            frames = poses.shape[0]

            per_cat_counts[cat_str] += 1
            per_cat_frames[cat_str].append(frames)
            total_examples += 1

    if total_examples == 0:
        fail("No se encontraron embeddings válidos para entrenamiento.")
        return {"total": 0, "per_cat_counts": {}, "per_cat_avg_frames": {}, "avg_frames": 0.0}

    # Calcular medias de frames
    per_cat_avg_frames: Dict[str, float] = {}
    all_frames: List[int] = []
    for cat, frames_list in per_cat_frames.items():
        if frames_list:
            avg_f = sum(frames_list) / len(frames_list)
            per_cat_avg_frames[cat] = avg_f
            all_frames.extend(frames_list)
        else:
            per_cat_avg_frames[cat] = 0.0

    avg_frames = sum(all_frames) / len(all_frames) if all_frames else 0.0

    # Mostrar resumen
    print(f"\nResumen de embeddings (solo clips con un único usuario, regla especial para clase 6):")
    print(f"{BOLD}{'Cat':>4} | {'#clips':>8} | {'%':>6} | {'frames_medios':>14}{RESET}")
    print("-" * 40)
    for cat in sorted(per_cat_counts.keys(), key=lambda x: int(x) if x.isdigit() else x):
        cnt = per_cat_counts[cat]
        pct = 100.0 * cnt / total_examples if total_examples > 0 else 0.0
        avg_f = per_cat_avg_frames.get(cat, 0.0)
        print(f"{cat:>4} | {cnt:8d} | {pct:6.2f} | {avg_f:14.2f}")

    print(f"\nTotal de embeddings válidos: {color(str(total_examples), CYAN)}")
    print(f"Frames medios por embedding: {color(f'{avg_frames:.2f}', CYAN)}")

    return {
        "total": total_examples,
        "per_cat_counts": per_cat_counts,
        "per_cat_avg_frames": per_cat_avg_frames,
        "avg_frames": avg_frames,
    }


def estimate_times(
    n_examples: int,
    avg_frames: float,
    device_info: Dict[str, Any],
) -> None:
    header("5) Estimación aproximada de tiempos por experimento")

    if n_examples == 0:
        warn("No hay ejemplos para estimar tiempos.")
        return

    device = device_info.get("device", "cpu")
    on_gpu = device.startswith("cuda")

    # Coeficientes base (segundos por batch) muy aproximados
    # Ajustados para una GPU potente; en CPU multiplicamos por un factor.
    base_time_per_batch = {
        "tcn": 0.004,             # ~4 ms/batch
        "res_tcn": 0.005,         # algo más profundo que TCN simple
        "stgcn": 0.005,           # un poco más caro que TCN
        "lstm": 0.006,            # ~6 ms/batch
        "transformer": 0.008,     # ~8 ms/batch
        "pose_cnn2d": 0.006,      # similar a LSTM en coste
        "joint_attn": 0.008,      # parecido a transformer
        "dilated_tcn": 0.005,     # similar a res_tcn
        "tcn_lstm": 0.007,        # TCN + BiLSTM combinado
    }
    cpu_multiplier = 8.0  # CPU mucho más lenta que una 5090

    frame_factor = avg_frames / 64.0 if avg_frames > 0 else 1.0

    total_est_seconds = 0.0

    print(
        f"Dispositivo estimado: {color(device, CYAN)} "
        f"({'GPU' if on_gpu else 'CPU'})"
    )
    print(
        f"Ejemplos totales (antes de split): {color(str(n_examples), CYAN)}, "
        f"frames medios ~ {color(f'{avg_frames:.1f}', CYAN)}"
    )

    # Ordenar experimentos: primero por arquitectura, luego por nº de epochs
    sorted_exps = sorted(
        enumerate(EXPERIMENTS, start=1),
        key=lambda pair: (pair[1].get("arch", ""), int(pair[1].get("epochs", 0))),
    )
    iter_exps = sorted_exps
    if tqdm is not None:
        iter_exps = tqdm(iter_exps, desc="Estimando tiempos", unit="exp")

    print(f"\n{BOLD}{'ID':>3} | {'Arch':>11} | {'Epochs':>6} | {'Batch':>5} | {'SeqLen':>6} | {'t_exp (min)':>11}{RESET}")
    print("-" * 60)

    for i, cfg in iter_exps:
        if cfg.get("done", False):
            print(f"{i:3d} | {cfg['arch']:>11} | {cfg.get('epochs', 0):6d} | "
                  f"{cfg.get('batch_size', 0):5d} | {cfg.get('seq_len', 0):6d} | {'(saltado)':>11}")
            continue

        arch = cfg["arch"]
        epochs = int(cfg.get("epochs", 20))
        batch_size = int(cfg.get("batch_size", 32))

        # Estimación de batches por época (train + val)
        n_train = int(n_examples * TRAIN_RATIO)
        n_val = int(n_examples * VAL_RATIO)
        train_batches = max(1, math.ceil(n_train / batch_size))
        val_batches = max(1, math.ceil(n_val / batch_size))
        batches_per_epoch = train_batches + val_batches

        t_per_batch = base_time_per_batch.get(arch, 0.006)
        if not on_gpu:
            t_per_batch *= cpu_multiplier
        t_per_batch *= frame_factor

        epoch_time = batches_per_epoch * t_per_batch
        exp_time = epoch_time * epochs
        total_est_seconds += exp_time

        exp_time_min = exp_time / 60.0
        print(f"{i:3d} | {arch:>11} | {epochs:6d} | {batch_size:5d} | "
              f"{cfg.get('seq_len', 0):6d} | {exp_time_min:11.2f}")

    # Resumen total
    hours = int(total_est_seconds // 3600)
    rem = total_est_seconds % 3600
    minutes = int(rem // 60)
    seconds = int(rem % 60)

    print(
        f"\nTiempo TOTAL estimado para todos los experimentos (no marcados como done): "
        f"{color(f'~{hours}h {minutes}m {seconds}s', CYAN)}"
    )
    warn("Estas estimaciones son aproximadas y asumen un uso eficiente de la GPU/CPU.")


def main() -> None:
    start = time.time()

    check_python()
    deps = check_dependencies()

    torch_mod = deps.get("torch")
    gpu_info = check_gpu(torch_mod)

    # Necesitamos numpy para el escaneo
    global np
    ok_np, np_mod = (deps.get("numpy") is not None), deps.get("numpy")
    if not ok_np:
        fail("numpy es obligatorio para continuar.")
        return
    np = np_mod  # type: ignore[assignment]

    data_info = scan_embeddings(pose_source="filtered")

    estimate_times(
        n_examples=data_info["total"],
        avg_frames=data_info["avg_frames"],
        device_info=gpu_info,
    )

    elapsed = time.time() - start
    print(f"\nScript de pre-chequeo completado en {elapsed:.1f} segundos.")


if __name__ == "__main__":
    main()

