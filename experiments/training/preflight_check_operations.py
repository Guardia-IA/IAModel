import sys
import importlib
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import math

try:
    from .model_config import (  # type: ignore[attr-defined]
        DATA_RESULT_ROOT,
        EXPERIMENTS,
        SPLIT_RATIO_TRAIN,
        SPLIT_RATIO_VAL,
        suggest_split_ratios,
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
        AUGMENT_PROB,
        PREFLIGHT_AUG_VARIANTS_PER_CLIP,
        PREFLIGHT_MIRROR_COMPOSE_RATIO_ESTIMATE,
        MAX_DETERMINISTIC_VARIANTS,
        TRAIN_DETERMINISTIC_PROB,
        VALIDATE_NPY_MIRROR_COMPOSE_RATIO,
        EXTRA_MANIFEST_VIEWS_PER_CLIP_DEFAULT,
    )
except ImportError:
    from model_config import (  # type: ignore[attr-defined]
        DATA_RESULT_ROOT,
        EXPERIMENTS,
        SPLIT_RATIO_TRAIN,
        SPLIT_RATIO_VAL,
        suggest_split_ratios,
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
        AUGMENT_PROB,
        PREFLIGHT_AUG_VARIANTS_PER_CLIP,
        PREFLIGHT_MIRROR_COMPOSE_RATIO_ESTIMATE,
        MAX_DETERMINISTIC_VARIANTS,
        TRAIN_DETERMINISTIC_PROB,
        VALIDATE_NPY_MIRROR_COMPOSE_RATIO,
        EXTRA_MANIFEST_VIEWS_PER_CLIP_DEFAULT,
    )

# Misma ruta que train_model_operations.MANIFEST_CACHE_DIR; evita importar torch/train al ejecutar este script directamente.
_MANIFEST_CACHE_DIR = Path(__file__).resolve().parent / "operations_npy" / "manifest_cache"

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

TRAIN_RATIO = float(SPLIT_RATIO_TRAIN)
VAL_RATIO = float(SPLIT_RATIO_VAL)
# MIN_* / MAX_OCCLUSION_* por defecto desde model_config (alineado con train_model_operations).


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


def dataset_info_from_collect_examples(
    pose_source: str,
    single_user_only: bool,
    min_clip_seconds: float,
    min_valid_frames: int,
    min_valid_pct: float,
    max_occlusion_ratio: float,
) -> Dict[str, Any]:
    """
    Misma lista que train_model_operations.build_datasets_and_loaders → collect_examples.
    Incluye passes_filters, poses_full + valid_mask, etc.
    """
    header("4) Dataset (solo collect_examples — mismas reglas que train_model_operations)")
    try:
        from .train_model_operations import collect_examples
    except ImportError:
        from train_model_operations import collect_examples

    root = get_data_result_root()
    print(f"Carpeta de datos: {root}")
    print(
        f"{BOLD}Filtros:{RESET} idénticos a build_datasets_and_loaders → collect_examples "
        "(_user_quality_ok: min_clip/min_valid/pct/occlusion, passes_filters, poses_full+valid_mask, etc.)."
    )
    examples = collect_examples(
        pose_source=pose_source,
        single_user_only=single_user_only,
        min_clip_seconds=float(min_clip_seconds),
        min_valid_frames=int(min_valid_frames),
        min_valid_pct=float(min_valid_pct),
        max_occlusion_ratio=float(max_occlusion_ratio),
    )
    n = len(examples)
    per_cat_counts: Dict[str, int] = {}
    per_cat_frames: Dict[str, List[int]] = {}
    all_frames: List[int] = []
    iterable = tqdm(examples, desc="Midiendo frames .npy", unit="emb") if tqdm is not None else examples
    for ex in iterable:
        lk = str(ex.label)
        per_cat_counts[lk] = per_cat_counts.get(lk, 0) + 1
        try:
            arr = np.load(ex.pose_path)
            t = int(arr.shape[0])
        except Exception:
            t = 64
        per_cat_frames.setdefault(lk, []).append(t)
        all_frames.append(t)

    per_cat_avg_frames: Dict[str, float] = {}
    for cat, frames_list in per_cat_frames.items():
        per_cat_avg_frames[cat] = (sum(frames_list) / len(frames_list)) if frames_list else 0.0
    avg_frames = sum(all_frames) / len(all_frames) if all_frames else 0.0

    print(f"\n{BOLD}Resumen por etiqueta (tras collect_examples):{RESET}")
    print(f"{BOLD}{'label':>6} | {'#':>8} | {'%':>6} | {'frames_medios':>14}{RESET}")
    print("-" * 44)
    for cat in sorted(per_cat_counts.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        cnt = per_cat_counts[cat]
        pct = 100.0 * cnt / n if n > 0 else 0.0
        avg_f = per_cat_avg_frames.get(cat, 0.0)
        print(f"{cat:>6} | {cnt:8d} | {pct:6.2f} | {avg_f:14.2f}")
    print(f"\nTotal ejemplos (N para estimación de tiempos y mismo pool que train): {color(str(n), CYAN)}")
    print(f"Frames medios por embedding: {color(f'{avg_frames:.2f}', CYAN)}")
    return {
        "total": n,
        "per_cat_counts": per_cat_counts,
        "per_cat_avg_frames": per_cat_avg_frames,
        "avg_frames": avg_frames,
    }


def check_manifest_cache_section(
    manifest_cache_dir: Optional[str],
    pose_source: str,
    single_user_only: bool,
    min_clip_seconds: float,
    min_valid_frames: int,
    min_valid_pct: float,
    max_occlusion_ratio: float,
    legacy_selection: bool = False,
) -> None:
    """Alineación entre collect_examples (mismos filtros que train) y JSON en manifest_cache."""
    header("5) Caché de manifests validate_npy (opcional)")
    print(
        "Si entrenas con --manifest-cache-dir, cada UID estable (ruta del .npy relativa a data_result, "
        "o legado absoluta) puede tener un JSON de validate_npy; el resto usa la rejilla global."
    )
    print(f"Directorio por defecto en el proyecto: {_MANIFEST_CACHE_DIR}")
    if legacy_selection:
        warn("Activaste --legacy-selection (no afecta a collect_examples; ver aviso en main).")
    if not manifest_cache_dir:
        warn("No pasaste --manifest-cache-dir: el entrenamiento usará solo la rejilla global (sin manifests por fichero).")
        return
    mdir = Path(manifest_cache_dir).expanduser().resolve()
    if mdir.exists() and not mdir.is_dir():
        fail(f"No es una carpeta (es un fichero): {mdir}")
        return
    if not mdir.exists():
        try:
            mdir.mkdir(parents=True, exist_ok=True)
            ok(f"Carpeta de caché creada (vacía; rellénala con validate_npy / batch_build): {mdir}")
        except OSError as exc:
            fail(f"No se pudo crear la carpeta de caché: {mdir} ({exc})")
            return
    json_n = len(list(mdir.glob("*.json")))
    ok(f"Carpeta de caché: {mdir} | ficheros *.json = {json_n}")
    try:
        from .train_model_operations import (  # type: ignore[attr-defined]
            collect_examples,
            manifest_cache_path_for_uid,
            _example_uid,
        )
    except ImportError:
        from train_model_operations import (  # type: ignore[attr-defined]
            collect_examples,
            manifest_cache_path_for_uid,
            _example_uid,
        )
    ex = collect_examples(
        pose_source=pose_source,
        single_user_only=single_user_only,
        min_clip_seconds=float(min_clip_seconds),
        min_valid_frames=int(min_valid_frames),
        min_valid_pct=float(min_valid_pct),
        max_occlusion_ratio=float(max_occlusion_ratio),
    )
    print(
        f"collect_examples => {len(ex)} ejemplos (mismo N que la sección 4 si los flags coinciden)."
    )
    hits = 0
    seen: set[str] = set()
    for e in ex:
        uid = _example_uid(e)
        if uid in seen:
            continue
        seen.add(uid)
        p = manifest_cache_path_for_uid(mdir, uid)
        if p.exists():
            hits += 1
    uids = len(seen)
    pct = (100.0 * hits / uids) if uids else 0.0
    ok(f"UIDs únicos (mismo criterio que train): {uids} | con JSON en caché: {hits} ({pct:.1f}%)")
    if hits < uids:
        warn(
            f"Faltan manifests para {uids - hits} UIDs. Genera caché con "
            "batch_build_manifest_cache.py (o validate_npy.py por fichero)."
        )


def estimate_times(
    n_examples: int,
    avg_frames: float,
    augment_on_the_fly: bool = False,
    augment_prob: float = 0.65,
    aug_variants_per_clip: float = 0.0,
    mirror_compose_ratio_estimate: float = 0.0,
    maintain_class_ratio: bool = False,
    target_neg_pos_ratio: Optional[float] = None,
    per_cat_counts: Optional[Dict[str, int]] = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    extra_manifest_views_per_clip: int = 0,
) -> None:
    header("6) Estimación aproximada de tiempos por experimento")
    if n_examples == 0:
        warn("No hay ejemplos para estimar tiempos.")
        return
    base_time_per_batch = {
        "tcn": 0.004,
        "ms_tcn": 0.0058,
        "tcn_attn": 0.0060,
        "res_tcn": 0.005,
        "stgcn": 0.005,
        "gat_tcn": 0.0072,
        "lstm": 0.006,
        "gru": 0.0055,
        "gru_attn": 0.0062,
        "transformer": 0.008,
        "conformer_lite": 0.009,
        "pose_cnn2d": 0.006,
        "joint_attn": 0.008,
        "dilated_tcn": 0.005,
        "tcn_lstm": 0.007,
        "tcn_gru": 0.0065,
    }
    cpu_multiplier = 8.0
    frame_factor = avg_frames / 64.0 if avg_frames > 0 else 1.0
    tr_eff = float(max(0.0, min(1.0, train_ratio)))
    vr_eff = float(max(0.0, min(1.0, val_ratio)))
    if tr_eff + vr_eff > 1.0:
        tr_eff, vr_eff = float(TRAIN_RATIO), float(VAL_RATIO)

    if aug_variants_per_clip > 0:
        avg_var_base = max(0.0, float(aug_variants_per_clip))
        avg_var_mirror_extra = avg_var_base * max(0.0, min(1.0, float(mirror_compose_ratio_estimate)))
        avg_var_total = avg_var_base + avg_var_mirror_extra
        expansion_factor = 1.0 + avg_var_total
        effective_examples = int(round(n_examples * expansion_factor))
        extra_aug = effective_examples - n_examples
        print(
            "Clips efectivos por época => "
            f"originales={n_examples}, variantes_por_clip_base={avg_var_base:.2f}, "
            f"extra_mirror_promedio={avg_var_mirror_extra:.2f}, "
            f"variantes_por_clip_total={avg_var_total:.2f}, "
            f"factor={expansion_factor:.2f}, total={color(str(effective_examples), CYAN)}"
        )
        print(
            f"Promedio por NPY real => base={avg_var_base:.2f}, "
            f"mirror_extra={avg_var_mirror_extra:.2f}, total_variantes={avg_var_total:.2f}"
        )
    elif int(extra_manifest_views_per_clip) > 0:
        exv = int(extra_manifest_views_per_clip)
        n = n_examples
        n_tr = int(n * tr_eff)
        n_va = int(n * vr_eff)
        n_te = max(0, n - n_tr - n_va)
        effective_examples = n_te + (n_tr + n_va) * (1 + exv)
        extra_aug = effective_examples - n_examples
        print(
            f"Expansión validate_npy (--extra-manifest-views-per-clip={exv}, alineado con train_model_operations): "
            f"clips únicos={n} | n_tr≈{n_tr} n_val≈{n_va} n_test≈{n_te} | "
            f"filas train+val ≈ (n_tr+n_val)*(1+{exv}), test sin expandir | "
            f"filas totales/época ≈ {color(str(effective_examples), CYAN)}"
        )
        if augment_on_the_fly:
            warn(
                "Con --augment-on-the-fly el coste real puede ser algo mayor; esta línea solo cuenta filas del dataset."
            )
    else:
        extra_aug = int(round(n_examples * max(0.0, min(1.0, augment_prob)))) if augment_on_the_fly else 0
        effective_examples = n_examples + extra_aug
        print(
            f"Clips efectivos por época => originales={n_examples}, augment_virtual={extra_aug}, "
            f"total={color(str(effective_examples), CYAN)}"
        )
        if not augment_on_the_fly and extra_aug == 0:
            print(
                f"Nota: augment_virtual solo refleja --augment-on-the-fly o --aug-variants-per-clip>0. "
                f"En train, variantes determinísticas (hasta {MAX_DETERMINISTIC_VARIANTS}, "
                f"p={TRAIN_DETERMINISTIC_PROB} en train) no aumentan len(dataset); diversifican cada muestra en __getitem__."
            )
            print(
                "Para expansión explícita por filas como train --extra-manifest-views-per-clip N, "
                "pasa ese mismo N aquí. Para heurística antigua usa "
                f"--aug-variants-per-clip K (ref. validate_npy mirror {VALIDATE_NPY_MIRROR_COMPOSE_RATIO})."
            )
        if augment_on_the_fly:
            print(
                "Heurística on-the-fly: no duplica ítems en el DataLoader; es coste extra aproximado. "
                "Para expansión explícita por manifest usa --extra-manifest-views-per-clip."
            )
    if maintain_class_ratio and per_cat_counts is not None:
        neg = sum(v for k, v in per_cat_counts.items() if str(k) != "6")
        pos = per_cat_counts.get("6", 0)
        if pos > 0:
            obs = neg / pos
            tgt = target_neg_pos_ratio if target_neg_pos_ratio is not None else obs
            print(f"Ratio no-robo/robo observado={obs:.3f} | objetivo sampler={tgt:.3f}")
            # Mostrar conteos explícitos por clase (original y virtual estimado)
            print(
                f"Conteo original por clase => no-robo={neg} | robo={pos} | total={neg + pos}"
            )
            if (neg + pos) > 0:
                virt_neg = int(round(effective_examples * (neg / (neg + pos))))
                virt_pos = int(max(0, effective_examples - virt_neg))
                print(
                    f"Conteo virtual estimado por clase => no-robo={virt_neg} | "
                    f"robo={virt_pos} | total={effective_examples}"
                )

    tr_b = float(max(0.0, min(1.0, train_ratio)))
    vr_b = float(max(0.0, min(1.0, val_ratio)))
    if tr_b + vr_b > 1.0:
        tr_b, vr_b = float(TRAIN_RATIO), float(VAL_RATIO)
    use_manifest_row_split = int(extra_manifest_views_per_clip) > 0 and aug_variants_per_clip <= 0
    if use_manifest_row_split:
        exv_m = int(extra_manifest_views_per_clip)
        n_tr0 = int(n_examples * tr_b)
        n_va0 = int(n_examples * vr_b)
        batch_train_rows = n_tr0 * (1 + exv_m)
        batch_val_rows = n_va0 * (1 + exv_m)
    else:
        batch_train_rows = -1
        batch_val_rows = -1

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
        tr = float(max(0.0, min(1.0, train_ratio)))
        vr = float(max(0.0, min(1.0, val_ratio)))
        if tr + vr > 1.0:
            tr, vr = TRAIN_RATIO, VAL_RATIO
        if batch_train_rows >= 0:
            n_train = batch_train_rows
            n_val = batch_val_rows
        else:
            n_train = int(effective_examples * tr)
            n_val = int(effective_examples * vr)
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
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="(Obsoleto) Modo actual ya usa todos los usuarios para cat!=6.",
    )
    parser.add_argument(
        "--legacy-selection",
        action="store_true",
        help="Activa selección antigua de preflight_check.py (no-cat6 único usuario, cat6 usuario mayoritario).",
    )
    parser.add_argument("--augment-on-the-fly", action="store_true")
    parser.add_argument(
        "--augment-prob",
        type=float,
        default=AUGMENT_PROB,
        help=f"Probabilidad augment on-the-fly (alineado con train; default {AUGMENT_PROB}).",
    )
    parser.add_argument(
        "--aug-variants-per-clip",
        type=float,
        default=PREFLIGHT_AUG_VARIANTS_PER_CLIP,
        help=(
            "Si >0, estima tiempo como expansión explícita por clip: total = N * (1 + valor). "
            f"Default model_config: {PREFLIGHT_AUG_VARIANTS_PER_CLIP}. Ej: 75 => original + 75 variantes."
        ),
    )
    parser.add_argument(
        "--mirror-compose-ratio-estimate",
        type=float,
        default=PREFLIGHT_MIRROR_COMPOSE_RATIO_ESTIMATE,
        help=(
            "Fracción [0..1] de variantes base a las que además se aplica mirror "
            f"en la estimación por expansión explícita (default {PREFLIGHT_MIRROR_COMPOSE_RATIO_ESTIMATE})."
        ),
    )
    parser.add_argument("--maintain-class-ratio", action="store_true")
    parser.add_argument("--target-neg-pos-ratio", type=float, default=None)
    parser.add_argument("--min-clip-seconds", type=float, default=MIN_CLIP_SECONDS)
    parser.add_argument("--min-valid-frames", type=int, default=MIN_VALID_FRAMES)
    parser.add_argument("--min-valid-pct", type=float, default=MIN_VALID_PCT)
    parser.add_argument("--max-occlusion-ratio", type=float, default=MAX_OCCLUSION_RATIO)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help=(
            "Fracción train para estimación de batches. Omitir junto con --val-ratio para usar "
            f"suggest_split_ratios(N) (igual que train sin flags). Referencia: {SPLIT_RATIO_TRAIN:.3f}."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Fracción val. Omitir junto con --train-ratio para heurística automática.",
    )
    parser.add_argument(
        "--manifest-cache-dir",
        type=str,
        default=None,
        help="Si se indica, cuenta cuántos ejemplos de collect_examples tienen JSON en esa carpeta (mismo uso que train --manifest-cache-dir). "
        "Si no existe, se crea vacía.",
    )
    parser.add_argument(
        "--extra-manifest-views-per-clip",
        type=int,
        default=EXTRA_MANIFEST_VIEWS_PER_CLIP_DEFAULT,
        help=(
            "Mismo valor que train_model_operations --extra-manifest-views-per-clip: filas train+val "
            f"≈ (n_tr+n_val)*(1+N), test sin expandir. Default {EXTRA_MANIFEST_VIEWS_PER_CLIP_DEFAULT}."
        ),
    )
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
    if args.legacy_selection:
        warn(
            "--legacy-selection no está implementado en collect_examples/train; "
            "se ignora. Quita el flag o implementa la lógica en train_model_operations si la necesitas."
        )
    data_info = dataset_info_from_collect_examples(
        pose_source=args.pose_source,
        single_user_only=args.single_user_only,
        min_clip_seconds=float(args.min_clip_seconds),
        min_valid_frames=int(args.min_valid_frames),
        min_valid_pct=float(args.min_valid_pct),
        max_occlusion_ratio=float(args.max_occlusion_ratio),
    )
    n_tot = int(data_info["total"])
    if (args.train_ratio is None) ^ (args.val_ratio is None):
        fail("Indica ambos --train-ratio y --val-ratio, o ninguno (heurística suggest_split_ratios).")
        return

    if args.train_ratio is None and args.val_ratio is None:
        tr_eff, va_eff, _te_eff = suggest_split_ratios(max(n_tot, 1))
    else:
        tr_eff = float(args.train_ratio)
        va_eff = float(args.val_ratio)

    if n_tot > 0:
        s_tr, s_va, s_te = suggest_split_ratios(n_tot)
        print(
            f"\n{BOLD}Reparto sugerido (heurística por N={n_tot}): "
            f"train/val/test = {s_tr:.3f} / {s_va:.3f} / {s_te:.3f}{RESET}"
        )
        if args.train_ratio is None and args.val_ratio is None:
            te_cur = s_te
            print(
                f"{BOLD}Reparto en uso (automático, mismo que train sin --train-ratio/--val-ratio): "
                f"{s_tr:.3f} / {s_va:.3f} / {te_cur:.3f}{RESET}"
            )
        else:
            te_cur = 1.0 - float(args.train_ratio) - float(args.val_ratio)
            print(
                f"{BOLD}Reparto en uso (args explícitos): "
                f"{float(args.train_ratio):.3f} / {float(args.val_ratio):.3f} / {te_cur:.3f}{RESET}"
            )
    check_manifest_cache_section(
        manifest_cache_dir=args.manifest_cache_dir,
        pose_source=args.pose_source,
        single_user_only=args.single_user_only,
        min_clip_seconds=args.min_clip_seconds,
        min_valid_frames=args.min_valid_frames,
        min_valid_pct=args.min_valid_pct,
        max_occlusion_ratio=args.max_occlusion_ratio,
        legacy_selection=args.legacy_selection,
    )
    estimate_times(
        n_examples=data_info["total"],
        avg_frames=data_info["avg_frames"],
        augment_on_the_fly=args.augment_on_the_fly,
        augment_prob=args.augment_prob,
        aug_variants_per_clip=args.aug_variants_per_clip,
        mirror_compose_ratio_estimate=args.mirror_compose_ratio_estimate,
        maintain_class_ratio=args.maintain_class_ratio,
        target_neg_pos_ratio=args.target_neg_pos_ratio,
        per_cat_counts=data_info["per_cat_counts"],
        train_ratio=tr_eff,
        val_ratio=va_eff,
        extra_manifest_views_per_clip=int(args.extra_manifest_views_per_clip),
    )
    print(f"\nScript de pre-chequeo completado en {time.time() - start:.1f} segundos.")


if __name__ == "__main__":
    main()

