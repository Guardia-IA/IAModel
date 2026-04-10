import sys
import importlib
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import math

try:
    from .model_config import DATA_RESULT_ROOT, EXPERIMENTS  # type: ignore[attr-defined]
except ImportError:
    from model_config import DATA_RESULT_ROOT, EXPERIMENTS  # type: ignore[attr-defined]

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
MIN_SEQ_LEN = 4
MIN_CLIP_SECONDS = 3.0
MIN_VALID_FRAMES = 12
MIN_VALID_PCT = 20.0
MAX_OCCLUSION_RATIO = 90.0


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
    ok(f"Versión de Python: {sys.version.split()[0]}")


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
    for lib, imp in [("numpy", "numpy"), ("torch", "torch")]:
        ok_, module = check_module(lib, imp)
        deps[lib] = module if ok_ else None
    return deps


def check_gpu(torch_mod) -> Dict[str, Any]:
    header("3) Dispositivo de cómputo (GPU/CPU)")
    info: Dict[str, Any] = {"device": "cpu", "gpus": []}
    if torch_mod is None:
        fail("PyTorch no está disponible. Solo se podrá usar CPU.")
        return info
    if not torch_mod.cuda.is_available():
        warn("CUDA NO disponible. Se estimará CPU y GPU de forma teórica.")
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
    return info


def get_data_result_root() -> Path:
    root = DATA_RESULT_ROOT
    if not root.exists():
        raise RuntimeError(f"No se encontró la carpeta data_result en: {root}")
    return root


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _user_quality_ok(user_meta: Dict[str, Any], meta: Dict[str, Any], pose_len: int) -> bool:
    valid_frames = _to_int(user_meta.get("valid_frames"), default=pose_len)
    total_frames = _to_int(user_meta.get("total_frames"), default=pose_len)
    valid_pct = _to_float(user_meta.get("valid_pct"), default=100.0 if total_frames <= 0 else 0.0)
    occlusion_ratio = _to_float(user_meta.get("occlusion_ratio"), default=0.0)
    if valid_pct <= 0 and total_frames > 0 and valid_frames > 0:
        valid_pct = 100.0 * (valid_frames / total_frames)
    clip_duration = _to_float(meta.get("clip_duration"), default=0.0)
    if clip_duration > 0 and clip_duration < MIN_CLIP_SECONDS:
        return False
    if pose_len < MIN_SEQ_LEN:
        return False
    if valid_frames < MIN_VALID_FRAMES:
        return False
    if valid_pct < MIN_VALID_PCT:
        return False
    if occlusion_ratio > MAX_OCCLUSION_RATIO:
        return False
    if user_meta.get("passes_filters") is False:
        return False
    return True


def scan_embeddings(pose_source: str = "filtered", single_user_only: bool = False) -> Dict[str, Any]:
    header("4) Escaneo de embeddings en data_result")
    root = get_data_result_root()
    print(f"Carpeta de datos: {root}")
    total_examples = 0
    per_cat_counts: Dict[str, int] = {}
    per_cat_frames: Dict[str, List[int]] = {}
    cat_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    iterable = tqdm(cat_dirs, desc="Recorriendo categorías", unit="cat") if tqdm is not None else cat_dirs
    for cat_dir in iterable:
        cat_str = cat_dir.name
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
            if single_user_only and len(users) != 1:
                continue
            for user in users:
                track_id = user.get("track_id")
                if track_id is None:
                    continue
                user_dir = clip_dir / f"user_{track_id}"
                pose_file = "poses.npy" if pose_source == "filtered" else "poses_full.npy"
                pose_path = user_dir / pose_file
                if not pose_path.exists():
                    continue
                try:
                    poses = np.load(pose_path)
                except Exception:
                    continue
                if poses.ndim != 3 or poses.shape[-1] != 2:
                    continue
                if not _user_quality_ok(user, meta, poses.shape[0]):
                    continue
                user_cat = user.get("user_cat")
                if user_cat is not None:
                    label_cat = str(_to_int(user_cat, default=_to_int(meta.get("cat", cat_str), default=0)))
                else:
                    label_cat = str(_to_int(meta.get("cat", cat_str), default=0))
                per_cat_counts.setdefault(label_cat, 0)
                per_cat_frames.setdefault(label_cat, [])
                per_cat_counts[label_cat] += 1
                per_cat_frames[label_cat].append(poses.shape[0])
                total_examples += 1
    if total_examples == 0:
        fail("No se encontraron embeddings válidos para entrenamiento.")
        return {"total": 0, "per_cat_counts": {}, "per_cat_avg_frames": {}, "avg_frames": 0.0}
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
    print("\nResumen de embeddings (incluye multiusuario y etiqueta por user_cat cuando existe):")
    print(f"{BOLD}{'Cat':>4} | {'#clips':>8} | {'%':>6} | {'frames_medios':>14}{RESET}")
    print("-" * 40)
    for cat in sorted(per_cat_counts.keys(), key=lambda x: int(x) if x.isdigit() else x):
        cnt = per_cat_counts[cat]
        pct = 100.0 * cnt / total_examples if total_examples > 0 else 0.0
        avg_f = per_cat_avg_frames.get(cat, 0.0)
        print(f"{cat:>4} | {cnt:8d} | {pct:6.2f} | {avg_f:14.2f}")
    print(f"\nTotal embeddings válidos (originales): {color(str(total_examples), CYAN)}")
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
    augment_on_the_fly: bool = False,
    augment_prob: float = 0.65,
    maintain_class_ratio: bool = False,
    target_neg_pos_ratio: Optional[float] = None,
    per_cat_counts: Optional[Dict[str, int]] = None,
) -> None:
    header("5) Estimación aproximada de tiempos por experimento")
    if n_examples == 0:
        warn("No hay ejemplos para estimar tiempos.")
        return
    base_time_per_batch = {
        "tcn": 0.004,
        "res_tcn": 0.005,
        "stgcn": 0.005,
        "lstm": 0.006,
        "transformer": 0.008,
        "pose_cnn2d": 0.006,
        "joint_attn": 0.008,
        "dilated_tcn": 0.005,
        "tcn_lstm": 0.007,
    }
    cpu_multiplier = 8.0
    frame_factor = avg_frames / 64.0 if avg_frames > 0 else 1.0
    extra_aug = int(round(n_examples * max(0.0, min(1.0, augment_prob)))) if augment_on_the_fly else 0
    effective_examples = n_examples + extra_aug
    print(
        f"Clips efectivos por época => originales={n_examples}, augment_virtual={extra_aug}, "
        f"total={color(str(effective_examples), CYAN)}"
    )
    if maintain_class_ratio and per_cat_counts is not None:
        neg = sum(v for k, v in per_cat_counts.items() if str(k) != "6")
        pos = per_cat_counts.get("6", 0)
        if pos > 0:
            obs = neg / pos
            tgt = target_neg_pos_ratio if target_neg_pos_ratio is not None else obs
            print(f"Ratio no-robo/robo observado={obs:.3f} | objetivo sampler={tgt:.3f}")

    sorted_exps = sorted(enumerate(EXPERIMENTS, start=1), key=lambda p: (p[1].get("arch", ""), int(p[1].get("epochs", 0))))
    iter_exps = tqdm(sorted_exps, desc="Estimando tiempos", unit="exp") if tqdm is not None else sorted_exps
    print(f"\n{BOLD}{'ID':>3} | {'Arch':>11} | {'Epochs':>6} | {'Batch':>5} | {'SeqLen':>6} | {'GPU(min)':>9} | {'CPU(min)':>9}{RESET}")
    print("-" * 86)

    total_gpu = 0.0
    total_cpu = 0.0
    for i, cfg in iter_exps:
        if cfg.get("done", False):
            print(f"{i:3d} | {cfg['arch']:>11} | {cfg.get('epochs', 0):6d} | {cfg.get('batch_size', 0):5d} | {cfg.get('seq_len', 0):6d} | {'saltado':>9} | {'saltado':>9}")
            continue
        arch = cfg["arch"]
        epochs = int(cfg.get("epochs", 20))
        batch_size = int(cfg.get("batch_size", 32))
        n_train = int(effective_examples * TRAIN_RATIO)
        n_val = int(effective_examples * VAL_RATIO)
        train_batches = max(1, math.ceil(n_train / batch_size))
        val_batches = max(1, math.ceil(n_val / batch_size))
        batches_per_epoch = train_batches + val_batches
        t_gpu = base_time_per_batch.get(arch, 0.006) * frame_factor
        t_cpu = t_gpu * cpu_multiplier
        exp_gpu = batches_per_epoch * t_gpu * epochs
        exp_cpu = batches_per_epoch * t_cpu * epochs
        total_gpu += exp_gpu
        total_cpu += exp_cpu
        print(f"{i:3d} | {arch:>11} | {epochs:6d} | {batch_size:5d} | {cfg.get('seq_len', 0):6d} | {exp_gpu/60.0:9.2f} | {exp_cpu/60.0:9.2f}")

    def fmt_hms(total_seconds: float) -> str:
        h = int(total_seconds // 3600)
        rem = total_seconds % 3600
        m = int(rem // 60)
        s = int(rem % 60)
        return f"~{h}h {m}m {s}s"

    print("\nTiempo TOTAL estimado (experimentos no done):")
    print(f"- GPU: {color(fmt_hms(total_gpu), CYAN)}")
    print(f"- CPU: {color(fmt_hms(total_cpu), CYAN)}")
    warn("Estimación aproximada; depende de I/O, workers y saturación real.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight de train_model_operations.")
    parser.add_argument("--pose-source", choices=["filtered", "full"], default="filtered")
    parser.add_argument("--single-user-only", action="store_true")
    parser.add_argument("--augment-on-the-fly", action="store_true")
    parser.add_argument("--augment-prob", type=float, default=0.65)
    parser.add_argument("--maintain-class-ratio", action="store_true")
    parser.add_argument("--target-neg-pos-ratio", type=float, default=None)
    parser.add_argument("--min-clip-seconds", type=float, default=MIN_CLIP_SECONDS)
    parser.add_argument("--min-valid-frames", type=int, default=MIN_VALID_FRAMES)
    parser.add_argument("--min-valid-pct", type=float, default=MIN_VALID_PCT)
    parser.add_argument("--max-occlusion-ratio", type=float, default=MAX_OCCLUSION_RATIO)
    return parser.parse_args()


def main() -> None:
    start = time.time()
    args = parse_args()
    global MIN_CLIP_SECONDS, MIN_VALID_FRAMES, MIN_VALID_PCT, MAX_OCCLUSION_RATIO
    MIN_CLIP_SECONDS = args.min_clip_seconds
    MIN_VALID_FRAMES = args.min_valid_frames
    MIN_VALID_PCT = args.min_valid_pct
    MAX_OCCLUSION_RATIO = args.max_occlusion_ratio

    check_python()
    deps = check_dependencies()
    torch_mod = deps.get("torch")
    check_gpu(torch_mod)
    global np
    ok_np, np_mod = (deps.get("numpy") is not None), deps.get("numpy")
    if not ok_np:
        fail("numpy es obligatorio para continuar.")
        return
    np = np_mod  # type: ignore[assignment]
    data_info = scan_embeddings(pose_source=args.pose_source, single_user_only=args.single_user_only)
    estimate_times(
        n_examples=data_info["total"],
        avg_frames=data_info["avg_frames"],
        augment_on_the_fly=args.augment_on_the_fly,
        augment_prob=args.augment_prob,
        maintain_class_ratio=args.maintain_class_ratio,
        target_neg_pos_ratio=args.target_neg_pos_ratio,
        per_cat_counts=data_info["per_cat_counts"],
    )
    print(f"\nScript de pre-chequeo completado en {time.time() - start:.1f} segundos.")


if __name__ == "__main__":
    main()

