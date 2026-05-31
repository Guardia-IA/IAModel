"""
Extracción masiva de logits del clasificador de poses para análisis de umbrales.

Estructura de entrada esperada (input_dir):

    input_dir/
      0/                      <- categoría/acción (nombre = entero)
        clip_aaa/             <- un clip
          clip.mp4            (ignorado: NO se re-extraen poses)
          meta.json
          user_0/poses.npy
          user_1/poses.npy
        clip_bbb/
          ...
      1/
        ...
      6/                      <- categoría de ROBO (configurable)
        ...

Por cada (clip, usuario) se ejecuta el modelo sobre los `poses.npy` ya extraídos
(mismo preprocesado que entrenamiento/eval) y se escribe una fila en el CSV con
los logits crudos, softmax, sigmoide por clase, el gap top1-top2 y la decisión.

Modelo: se indica un `.pt` (checkpoint del clasificador, imprescindible por sus
metadatos). Si junto a él existe un `.engine` con el mismo nombre y hay runtime
TensorRT disponible, se usa el `.engine` solo para el forward; si no, el `.pt`.

Config: JSON (ver logits.conf). Uso:

    python logits_extraction.py --config logits.conf
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# --- Imports del pipeline de entrenamiento/eval (carpeta training hermana) ---
_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
_TRAINING_DIR = _EXPERIMENTS_DIR / "training"
for _p in (str(_TRAINING_DIR), str(_EXPERIMENTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from train_model_operations import (  # type: ignore[attr-defined]
        PoseExample,
        build_model,
        build_pose_dataset_for_eval,
        normalize_sequence,
        add_velocity,
        temporal_resize,
    )
    _HAS_OPERATIONS = True
except ImportError:
    _HAS_OPERATIONS = False
    from train_model import (  # type: ignore[attr-defined]
        build_model,
        normalize_sequence,
        add_velocity,
        temporal_resize,
    )


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extrae logits del clasificador para muchos clips y los vuelca a CSV."
    )
    p.add_argument(
        "--config",
        default=str(_THIS_DIR / "logits.conf"),
        help="Ruta al fichero de configuración JSON (por defecto logits.conf junto al script).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Modo debug: corta tras debug_max_clips clips (por defecto 10). Sobrescribe la config.",
    )
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="No preguntar tras el preflight; lanzar directamente.",
    )
    return p.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"No existe el fichero de configuración: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"logits.conf no es JSON válido: {e}") from e
    for key in ("input_dir", "output_csv", "model_path"):
        if not cfg.get(key):
            raise SystemExit(f"Falta '{key}' (obligatorio) en {path}")
    cfg.setdefault("robbery_category", 6)
    cfg.setdefault("threshold_robbery", 0.8)
    cfg.setdefault("pose_source", "filtered")
    cfg.setdefault("simple_preprocess", False)
    cfg.setdefault("device", None)
    cfg.setdefault("use_engine_if_available", True)
    cfg.setdefault("csv_overwrite", True)
    cfg.setdefault("debug", False)
    cfg.setdefault("debug_max_clips", 10)
    cfg.setdefault("preflight", True)
    cfg.setdefault("preflight_sample", 8)
    cfg.setdefault("assume_yes", False)
    return cfg


# ---------------------------------------------------------------------------
# Backends de inferencia: Torch (siempre) y TensorRT (.engine, si disponible)
# ---------------------------------------------------------------------------
class TorchClassifier:
    """Forward con el modelo PyTorch reconstruido desde el checkpoint."""

    def __init__(self, checkpoint: Dict[str, Any], device: torch.device) -> None:
        cfg = checkpoint.get("config", {})
        arch = cfg.get("arch", "tcn")
        input_dim = int(checkpoint["input_dim"])
        num_classes = int(checkpoint.get("num_classes", len(checkpoint["label_to_idx"])))
        self.model = build_model(arch, input_dim, num_classes, cfg).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.device = device

    def logits(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            out = self.model(x.to(self.device))[0]
        return out.detach().float().cpu().numpy()


class TensorRTClassifier:
    """
    Forward con un .engine TensorRT usando tensores CUDA de PyTorch como buffers
    (sin pycuda). Asume entrada [1, seq_len, input_dim] float32 y salida [1, C].
    Si algo falla en la construcción, el llamador captura la excepción y cae a Torch.
    """

    def __init__(
        self,
        engine_path: Path,
        input_dim: int,
        seq_len: int,
        num_classes: int,
        device: torch.device,
    ) -> None:
        import tensorrt as trt  # type: ignore[import-not-found]

        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("El backend TensorRT requiere una GPU CUDA disponible.")

        self._trt = trt
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.device = device

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"No se pudo deserializar el engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # API por nombre de tensor (TensorRT 8.5+/10). Detecta input/output.
        self._input_name = None
        self._output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_name = name
        if self._input_name is None or self._output_name is None:
            raise RuntimeError("No se identificaron tensores de entrada/salida del engine.")

    def logits(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.device, dtype=torch.float32).contiguous()
        self.context.set_input_shape(self._input_name, tuple(x.shape))
        out_shape = tuple(self.context.get_tensor_shape(self._output_name))
        out = torch.empty(out_shape, dtype=torch.float32, device=self.device)

        self.context.set_tensor_address(self._input_name, int(x.data_ptr()))
        self.context.set_tensor_address(self._output_name, int(out.data_ptr()))
        stream = torch.cuda.current_stream(self.device)
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        return out.detach().cpu().numpy().reshape(-1)


def build_backend(
    checkpoint: Dict[str, Any],
    ckpt_path: Path,
    device: torch.device,
    use_engine_if_available: bool,
) -> Tuple[Any, str]:
    """
    Devuelve (backend, nombre_backend). Intenta engine si hay .engine hermano y
    runtime TensorRT; si no, Torch.
    """
    engine_path = ckpt_path.with_suffix(".engine")
    if use_engine_if_available and engine_path.exists():
        try:
            input_dim = int(checkpoint["input_dim"])
            seq_len = int(checkpoint.get("seq_len", 64))
            num_classes = int(checkpoint.get("num_classes", len(checkpoint["label_to_idx"])))
            backend = TensorRTClassifier(engine_path, input_dim, seq_len, num_classes, device)
            return backend, f"engine ({engine_path.name})"
        except Exception as e:  # noqa: BLE001 - fallback intencionado a Torch
            print(f"[WARN] No se pudo usar el engine ({engine_path.name}): {e}. Se usa el .pt.")
    return TorchClassifier(checkpoint, device), "torch (.pt)"


# ---------------------------------------------------------------------------
# Preprocesado de poses -> tensor [1, seq_len, input_dim]
# ---------------------------------------------------------------------------
def _tensor_simple_pipeline(poses: np.ndarray, seq_len: int) -> torch.Tensor:
    if np.any(np.isnan(poses)):
        poses = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0)
    poses = normalize_sequence(poses)
    poses = add_velocity(poses)
    poses = temporal_resize(poses, seq_len)
    t, j, d = poses.shape
    return torch.from_numpy(poses.reshape(t, j * d).astype(np.float32)).unsqueeze(0)


def _load_poses_array(user_dir: Path, pose_source: str) -> np.ndarray:
    if pose_source == "filtered":
        pose_path = user_dir / "poses.npy"
        valid_mask_path: Optional[Path] = None
    else:
        pose_path = user_dir / "poses_full.npy"
        valid_mask_path = user_dir / "valid_mask.npy"
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    poses = np.load(pose_path)
    if valid_mask_path is not None and valid_mask_path.exists():
        vm = np.load(valid_mask_path)
        poses = poses[vm].copy()
    if np.any(np.isnan(poses)):
        poses = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0)
    return poses


def build_input_tensor(
    *,
    user_dir: Path,
    pose_source: str,
    clip_name: str,
    checkpoint: Dict[str, Any],
    seq_len: int,
    simple_preprocess: bool,
    track_id: int,
) -> torch.Tensor:
    """Construye el tensor de entrada igual que test_model2 (eval alineado)."""
    label_to_idx: Dict[Any, int] = checkpoint["label_to_idx"]
    dummy_label = int(next(iter(label_to_idx.keys())))

    if _HAS_OPERATIONS and not simple_preprocess:
        pose_path = (user_dir / "poses.npy") if pose_source == "filtered" else (user_dir / "poses_full.npy")
        mask_path = user_dir / "valid_mask.npy"
        ex = PoseExample(
            pose_path=pose_path.resolve(),
            label=dummy_label,
            track_id=track_id,
            clip_name=clip_name,
            category_str="infer",
            valid_mask_path=(mask_path.resolve() if (pose_source == "full" and mask_path.exists()) else None),
            users_in_clip=1,
        )
        try:
            ds = build_pose_dataset_for_eval(
                [ex], label_to_idx, seq_len, dataset_split="test", checkpoint=checkpoint
            )
            xb, _y = ds[0]
            return xb.unsqueeze(0)
        except Exception:
            poses = _load_poses_array(user_dir, pose_source)
            return _tensor_simple_pipeline(poses, seq_len)

    poses = _load_poses_array(user_dir, pose_source)
    return _tensor_simple_pipeline(poses, seq_len)


# ---------------------------------------------------------------------------
# Cálculo de métricas a partir de logits
# ---------------------------------------------------------------------------
def _label_to_class_index(label_to_idx: Dict[Any, Any], label: int) -> Optional[int]:
    if label in label_to_idx:
        return int(label_to_idx[label])
    ls = str(label)
    if ls in label_to_idx:
        return int(label_to_idx[ls])
    return None


def robbery_class_index(checkpoint: Dict[str, Any]) -> int:
    """Índice de la clase 'robo' en el vector de logits."""
    task = checkpoint.get("task", "multiclass")
    label_to_idx: Dict[Any, Any] = checkpoint["label_to_idx"]
    if task == "binary":
        idx = _label_to_class_index(label_to_idx, 1)
        return idx if idx is not None else int(label_to_idx.get(1, 1))
    positive_class = int(checkpoint.get("positive_class", 6))
    idx = _label_to_class_index(label_to_idx, positive_class)
    return idx if idx is not None else 1


def logits_metrics(logits: np.ndarray, robo_idx: int) -> Dict[str, float]:
    """softmax, sigmoide por clase, gap top1-top2 y prob de robo."""
    logits = logits.astype(np.float64).reshape(-1)
    m = float(np.max(logits))
    exp = np.exp(logits - m)
    softmax = exp / np.sum(exp)
    sigmoid = 1.0 / (1.0 + np.exp(-logits))
    if logits.shape[0] >= 2:
        gap = float(np.max(logits) - np.partition(logits, -2)[-2])
    else:
        gap = 0.0
    out: Dict[str, float] = {"gap": gap}
    for i in range(logits.shape[0]):
        out[f"clase{i}_logit"] = float(logits[i])
        out[f"clase{i}_softmax"] = float(softmax[i])
        out[f"clase{i}_sigmoide"] = float(sigmoid[i])
    out["prob_robo"] = float(softmax[robo_idx]) if 0 <= robo_idx < softmax.shape[0] else float(np.max(softmax))
    return out


def build_csv_columns(num_classes: int) -> List[str]:
    cols = ["clip_path", "clip_name", "categoria", "is_robo", "num_usuarios", "usuario"]
    for i in range(num_classes):
        cols.append(f"clase{i}_logit")
    for i in range(num_classes):
        cols.append(f"clase{i}_softmax")
    for i in range(num_classes):
        cols.append(f"clase{i}_sigmoide")
    cols += ["gap", "prob_robo", "decision", "backend"]
    return cols


# ---------------------------------------------------------------------------
# Recorrido del dataset
# ---------------------------------------------------------------------------
def _category_dirs(input_dir: Path) -> List[Tuple[int, Path]]:
    """Subcarpetas cuyo nombre es un entero -> (categoria, ruta), ordenadas."""
    out: List[Tuple[int, Path]] = []
    for d in input_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            out.append((int(d.name), d))
    out.sort(key=lambda x: x[0])
    return out


def _clip_dirs(category_dir: Path) -> List[Path]:
    """Subcarpetas de clip: las que contienen al menos un user_*/."""
    out: List[Path] = []
    for d in sorted(category_dir.iterdir(), key=lambda x: x.name):
        if d.is_dir() and any(u.is_dir() for u in d.glob("user_*")):
            out.append(d)
    return out


def _user_dirs(clip_dir: Path) -> List[Path]:
    return sorted([u for u in clip_dir.glob("user_*") if u.is_dir()], key=lambda x: x.name)


def _clip_name_from_meta(clip_dir: Path) -> str:
    meta_path = clip_dir / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return str(meta.get("clip_name", clip_dir.name))
        except (OSError, json.JSONDecodeError):
            pass
    return clip_dir.name


def gather_clips(input_dir: Path) -> Tuple[List[Tuple[int, Path]], List[Dict[str, Any]]]:
    """
    Escanea el árbol y devuelve (categorias, clips). Cada clip:
    {categoria, clip_dir, clip_name, user_dirs}. Solo lista directorios (rápido).
    """
    categories = _category_dirs(input_dir)
    clips: List[Dict[str, Any]] = []
    for categoria, cat_dir in categories:
        for clip_dir in _clip_dirs(cat_dir):
            clips.append(
                {
                    "categoria": categoria,
                    "clip_dir": clip_dir,
                    "clip_name": _clip_name_from_meta(clip_dir),
                    "user_dirs": _user_dirs(clip_dir),
                }
            )
    return categories, clips


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.1f} min"
    return f"{minutes / 60.0:.2f} h"


def _device_label(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            idx = device.index if device.index is not None else torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            return f"GPU CUDA: {name} ({total_gb:.1f} GB)"
        except Exception:  # noqa: BLE001
            return "GPU CUDA (detalle no disponible)"
    return "CPU"


def preflight(
    *,
    clips: List[Dict[str, Any]],
    categories: List[Tuple[int, Path]],
    backend: Any,
    backend_name: str,
    checkpoint: Dict[str, Any],
    device: torch.device,
    pose_source: str,
    simple_preprocess: bool,
    seq_len: int,
    robo_idx: int,
    sample_size: int,
    debug: bool,
    debug_max_clips: int,
) -> None:
    """Resumen del trabajo + estimación de tiempo cronometrando unas inferencias."""
    total_clips = len(clips)
    total_users = sum(len(c["user_dirs"]) for c in clips)
    per_cat: Dict[int, Tuple[int, int]] = {}
    for c in clips:
        nc, nu = per_cat.get(c["categoria"], (0, 0))
        per_cat[c["categoria"]] = (nc + 1, nu + len(c["user_dirs"]))

    print("\n========== PREFLIGHT ==========")
    print(f"[PRE] {_device_label(device)}")
    print(f"[PRE] Backend: {backend_name}")
    print(
        f"[PRE] Modelo: task={checkpoint.get('task', '?')} | "
        f"num_classes={checkpoint.get('num_classes', '?')} | seq_len={seq_len} | "
        f"input_dim={checkpoint.get('input_dim', '?')} | indice_clase_robo={robo_idx}"
    )
    print(f"[PRE] Preprocesado: {'operations (eval)' if (_HAS_OPERATIONS and not simple_preprocess) else 'simple'}")
    print(f"[PRE] Categorías: {len(categories)} -> {[c for c, _ in categories]}")
    for cat in sorted(per_cat):
        nc, nu = per_cat[cat]
        print(f"[PRE]   categoría {cat}: {nc} clips, {nu} usuarios (filas)")
    print(f"[PRE] TOTAL: {total_clips} clips, {total_users} usuarios (filas a generar)")

    # --- Muestra cronometrada (build_input_tensor + forward) por usuario ---
    sample: List[Tuple[Dict[str, Any], Path, int]] = []
    for c in clips:
        for ud in c["user_dirs"]:
            try:
                tid = int(ud.name.split("_")[1])
            except (IndexError, ValueError):
                tid = -1
            sample.append((c, ud, tid))
            if len(sample) >= max(1, int(sample_size)) + 1:  # +1 para warmup
                break
        if len(sample) >= max(1, int(sample_size)) + 1:
            break

    if not sample:
        print("[PRE] No hay usuarios para cronometrar; no se puede estimar el tiempo.")
        print("================================\n")
        return

    times: List[float] = []
    for k, (c, ud, tid) in enumerate(sample):
        try:
            t0 = time.perf_counter()
            x = build_input_tensor(
                user_dir=ud,
                pose_source=pose_source,
                clip_name=c["clip_name"],
                checkpoint=checkpoint,
                seq_len=seq_len,
                simple_preprocess=simple_preprocess,
                track_id=tid,
            )
            backend.logits(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            print(f"[PRE][WARN] muestra falló en {ud}: {e}")
            continue
        if k == 0:
            continue  # descarta el primero (warmup: caches, kernels CUDA, etc.)
        times.append(dt)

    if not times:
        print("[PRE] No se pudo cronometrar ninguna muestra válida; estimación no disponible.")
        print("================================\n")
        return

    per_user = float(np.mean(times))
    units = debug_max_clips if debug else total_clips
    units_users = total_users
    if debug:
        units_users = sum(len(c["user_dirs"]) for c in clips[: max(0, int(debug_max_clips))])
    est_total = per_user * units_users
    print(
        f"[PRE] Tiempo medio por usuario (n={len(times)}): "
        f"{per_user * 1000.0:.1f} ms (incluye preprocesado + forward)"
    )
    if debug:
        print(
            f"[PRE] MODO DEBUG: se procesarán {min(units, total_clips)} clips "
            f"(~{units_users} usuarios)."
        )
    print(f"[PRE] Estimación de tiempo total: ~{_fmt_duration(est_total)} "
          f"para {units_users} usuarios.")
    print("================================\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).expanduser().resolve())
    if args.debug:
        cfg["debug"] = True
    if args.yes:
        cfg["assume_yes"] = True
    debug = bool(cfg["debug"])
    debug_max_clips = int(cfg["debug_max_clips"])

    input_dir = Path(cfg["input_dir"]).expanduser().resolve()
    output_csv = Path(cfg["output_csv"]).expanduser().resolve()
    model_path = Path(cfg["model_path"]).expanduser().resolve()
    robo_category = int(cfg["robbery_category"])
    thr = float(np.clip(float(cfg["threshold_robbery"]), 0.0, 1.0))
    pose_source = str(cfg["pose_source"])
    simple_preprocess = bool(cfg["simple_preprocess"])
    use_engine = bool(cfg["use_engine_if_available"])

    if not input_dir.is_dir():
        raise SystemExit(f"input_dir no es una carpeta: {input_dir}")
    # El checkpoint .pt es imprescindible (metadatos + pesos / fallback).
    ckpt_path = model_path if model_path.suffix == ".pt" else model_path.with_suffix(".pt")
    if not ckpt_path.exists():
        raise SystemExit(
            f"Hace falta el checkpoint .pt con los metadatos del modelo: {ckpt_path} "
            "(aunque uses .engine, el .pt aporta label_to_idx/seq_len/config)."
        )

    device = torch.device(
        cfg["device"] if cfg["device"] else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[INFO] input_dir={input_dir}")
    print(f"[INFO] output_csv={output_csv}")
    print(f"[INFO] checkpoint={ckpt_path}")
    print(f"[INFO] device={device} | pose_source={pose_source} | simple_preprocess={simple_preprocess}")
    print(f"[INFO] categoria_robo={robo_category} | umbral_robo={thr:.0%} | eval_operations={_HAS_OPERATIONS}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    seq_len = int(checkpoint.get("seq_len", 64))
    num_classes = int(checkpoint.get("num_classes", len(checkpoint["label_to_idx"])))
    robo_idx = robbery_class_index(checkpoint)
    print(
        f"[INFO] task={checkpoint.get('task', '?')} | num_classes={num_classes} | "
        f"label_to_idx={checkpoint.get('label_to_idx')} | indice_clase_robo={robo_idx}"
    )

    backend, backend_name = build_backend(checkpoint, ckpt_path, device, use_engine)
    print(f"[INFO] backend de inferencia: {backend_name}")

    categories, clips = gather_clips(input_dir)
    if not categories:
        raise SystemExit(f"No hay subcarpetas numéricas (0,1,2,...) en {input_dir}")
    if not clips:
        raise SystemExit(f"No se encontraron clips con carpetas user_* en {input_dir}")

    if bool(cfg["preflight"]):
        preflight(
            clips=clips,
            categories=categories,
            backend=backend,
            backend_name=backend_name,
            checkpoint=checkpoint,
            device=device,
            pose_source=pose_source,
            simple_preprocess=simple_preprocess,
            seq_len=seq_len,
            robo_idx=robo_idx,
            sample_size=int(cfg["preflight_sample"]),
            debug=debug,
            debug_max_clips=debug_max_clips,
        )
        if not bool(cfg["assume_yes"]):
            if sys.stdin is not None and sys.stdin.isatty():
                resp = input("¿Continuar y generar el CSV? [y/N]: ").strip().lower()
                if resp not in ("y", "yes", "s", "si", "sí"):
                    print("[FIN] Cancelado por el usuario tras el preflight.")
                    return
            else:
                print("[INFO] Sin terminal interactiva; continuo (usa assume_yes/--yes para silenciar).")

    if debug:
        clips = clips[: max(0, debug_max_clips)]
        print(f"[INFO] MODO DEBUG activo: limitado a {len(clips)} clips.")

    columns = build_csv_columns(num_classes)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if bool(cfg["csv_overwrite"]) else "a"
    write_header = bool(cfg["csv_overwrite"]) or not output_csv.exists()

    n_rows = 0
    n_clips = 0
    n_errors = 0
    t0 = time.perf_counter()

    with open(output_csv, mode, newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=columns)
        if write_header:
            writer.writeheader()

        for clip in clips:
            n_clips += 1
            categoria = clip["categoria"]
            clip_dir = clip["clip_dir"]
            clip_name = clip["clip_name"]
            user_dirs = clip["user_dirs"]
            num_users = len(user_dirs)
            for ud in user_dirs:
                try:
                    tid = int(ud.name.split("_")[1])
                except (IndexError, ValueError):
                    tid = -1
                try:
                    x = build_input_tensor(
                        user_dir=ud,
                        pose_source=pose_source,
                        clip_name=clip_name,
                        checkpoint=checkpoint,
                        seq_len=seq_len,
                        simple_preprocess=simple_preprocess,
                        track_id=tid,
                    )
                    logits = backend.logits(x)
                    metrics = logits_metrics(logits, robo_idx)
                except Exception as e:  # noqa: BLE001
                    n_errors += 1
                    print(f"[WARN] Falló {ud}: {e}")
                    continue

                prob_robo = metrics["prob_robo"]
                row: Dict[str, Any] = {
                    "clip_path": str(clip_dir),
                    "clip_name": clip_name,
                    "categoria": categoria,
                    "is_robo": int(categoria == robo_category),
                    "num_usuarios": num_users,
                    "usuario": ud.name,
                    "gap": round(metrics["gap"], 6),
                    "prob_robo": round(prob_robo, 8),
                    "decision": "ROBO" if prob_robo >= thr else "NO_ROBO",
                    "backend": backend_name,
                }
                for i in range(num_classes):
                    row[f"clase{i}_logit"] = round(metrics[f"clase{i}_logit"], 6)
                    row[f"clase{i}_softmax"] = round(metrics[f"clase{i}_softmax"], 8)
                    row[f"clase{i}_sigmoide"] = round(metrics[f"clase{i}_sigmoide"], 8)
                writer.writerow(row)
                n_rows += 1
            if n_clips % 25 == 0:
                fcsv.flush()
                print(f"[INFO] Progreso: {n_clips}/{len(clips)} clips, {n_rows} filas...")

    dt = time.perf_counter() - t0
    print(
        f"[FIN] {n_rows} filas escritas | {n_clips} clips | {n_errors} errores | "
        f"{dt:.1f}s -> {output_csv}"
    )


if __name__ == "__main__":
    main()
