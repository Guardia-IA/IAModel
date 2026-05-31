"""
Inferencia alineada con entrenamiento/evaluación:

1) Entrada: vídeo (extracción con `pose_extractor_clean.run_debug_extract`) **o** un `.npy` de poses
   `[T, J, 2]` (sin re-extraer; se monta un clip temporal `user_0/poses.npy`).
2) Copia las entradas a `.npy` temporales bajo `.infer_tmp` (opcional) y los borra tras inferir.
3) Usa `build_pose_dataset_for_eval` cuando el checkpoint es operations (como evaluate_singleuser);
   con `--simple-preprocess`, el pipeline de `train_model.PoseDataset`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

try:
    from pose_extractor_clean import run_debug_extract  # type: ignore[import-untyped]
except ImportError as e:
    run_debug_extract = None  # type: ignore[assignment,misc]
    _IMPORT_ERR = str(e)
else:
    _IMPORT_ERR = ""

try:
    from .train_model_operations import (  # type: ignore[attr-defined]
        PoseExample,
        build_model,
        build_pose_dataset_for_eval,
        normalize_sequence,
        add_velocity,
        temporal_resize,
    )
    _HAS_OPERATIONS = True
except ImportError:
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
        try:
            from .train_model import (  # type: ignore[attr-defined]
                build_model,
                normalize_sequence,
                add_velocity,
                temporal_resize,
            )
        except ImportError:
            from train_model import (  # type: ignore[attr-defined]
                build_model,
                normalize_sequence,
                add_velocity,
                temporal_resize,
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Vídeo o .npy [T,J,2] -> clasificador (eval alineado con evaluate_singleuser si operations). "
            "Con vídeo: extracción pose_extractor_clean. Con .npy: sin re-extraer."
        )
    )
    p.add_argument("--model", required=True, help="Checkpoint .pt del clasificador.")
    p.add_argument(
        "--input",
        default="",
        help=(
            "Vídeo, .npy [T,J,2], o carpeta con subcarpetas chunk_001, chunk_002, ... "
            "(cada una con poses.npy + meta.json; se concatenan en orden los que tengan frames válidos > 0)."
        ),
    )
    p.add_argument(
        "--video",
        default="",
        help="(Compat) Vídeo; equivalente a --input si no pasas --input.",
    )
    p.add_argument(
        "--pose-source",
        choices=["filtered", "full"],
        default="filtered",
        help="'filtered' = poses.npy | 'full' = poses_full.npy + valid_mask.npy.",
    )
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--extract-dir", type=str, default=None)
    p.add_argument("--keep-extract", action="store_true")
    p.add_argument("--threshold-robbery", type=float, default=0.8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--simple-preprocess",
        action="store_true",
        help="Solo normalize+velocity+temporal_resize (train_model), sin manifest operations.",
    )
    p.add_argument(
        "--no-temp-npy",
        action="store_true",
        help="No copiar a .npy temporal; leer directamente los ficheros del clip (menos fiel a np.load desde servicio).",
    )
    p.add_argument(
        "--yolo-pose-model",
        type=str,
        default="yolo11n-pose.pt",
        help=(
            "Modelo YOLO pose para run_debug_extract (extracción). Por defecto yolo11n-pose.pt "
            "(más rápido); en config suele usarse yolo11x-pose.pt. "
            "Ignorado con --skip-extract, .npy, o carpeta chunk_*."
        ),
    )
    return p.parse_args()


def _resolve_media_path(args: argparse.Namespace) -> Tuple[Optional[Path], str]:
    """
    Devuelve (ruta, kind) con kind en {\"skip\", \"npy\", \"video\", \"chunk_dir\"}.
    skip = --skip-extract (no hay vídeo/npy de entrada única).
    """
    if args.skip_extract:
        return None, "skip"
    inp = (args.input or "").strip()
    vid = (args.video or "").strip()
    if inp and vid and Path(inp).expanduser().resolve() != Path(vid).expanduser().resolve():
        raise SystemExit("Usa solo --input o solo --video cuando las rutas son distintas.")
    raw = inp or vid
    if not raw:
        raise SystemExit("Indica --input (vídeo o .npy), --video, o bien --skip-extract con --extract-dir.")
    p = Path(raw).expanduser().resolve()
    if p.is_dir():
        return p, "chunk_dir"
    if not p.is_file():
        raise FileNotFoundError(f"No existe o no es fichero/carpeta usable: {p}")
    if p.suffix.lower() == ".npy":
        return p, "npy"
    return p, "video"


def _clip_dir_from_input_npy(npy_path: Path, pose_source: str) -> Path:
    """Crea un directorio tipo clip con user_0/poses.npy (temporal)."""
    if pose_source != "filtered":
        raise SystemExit(
            "Con un único .npy de entrada usa --pose-source filtered (secuencia [T,J,2] como poses.npy). "
            "Para poses_full.npy + valid_mask.npy usa --skip-extract con --extract-dir apuntando al clip."
        )
    try:
        arr = np.load(str(npy_path), allow_pickle=False)
    except Exception as e:
        raise SystemExit(f"No se pudo leer el .npy: {e}") from e
    if getattr(arr, "ndim", 0) != 3 or int(arr.shape[2]) != 2:
        raise SystemExit(f"Se esperaba array [T, J, 2]; shape={getattr(arr, 'shape', None)}")
    root = Path(tempfile.mkdtemp(prefix="test_model2_npy_"))
    ud = root / "user_0"
    ud.mkdir(parents=True, exist_ok=False)
    shutil.copy2(npy_path, ud / "poses.npy")
    meta = {
        "clip_name": npy_path.stem or "poses_input",
        "users": [{"track_id": 0, "total_frames": int(arr.shape[0])}],
        "source_npy": str(npy_path.resolve()),
    }
    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return root


def _chunk_subdir_sort_key(path: Path) -> Tuple[int, str]:
    m = re.match(r"^chunk_(\d+)$", path.name, flags=re.IGNORECASE)
    if m:
        return (int(m.group(1)), path.name.lower())
    return (10**9, path.name.lower())


def _read_valid_frames_from_meta(meta: Dict[str, Any]) -> Optional[int]:
    """
    Número de frames válidos declarado en meta.json.
    None = no hay campo reconocido (se usará len(poses.npy) como respaldo).
    """
    for k in ("frames_validos", "valid_frames", "valid_frame_count", "filtered_frames"):
        if k in meta and meta[k] is not None:
            return int(meta[k])
    users = meta.get("users")
    if isinstance(users, list) and users:
        u0 = users[0]
        if isinstance(u0, dict):
            for k in ("valid_frames", "poses_filtered_count"):
                if k in u0 and u0[k] is not None:
                    return int(u0[k])
    return None


def _concatenate_chunk_poses(root: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Recorre root/chunk_* en orden, concatena poses.npy donde frames válidos > 0.
    Devuelve (array [T,J,2], nombres de chunks usados, mensajes de chunks omitidos).
    """
    subdirs = [p for p in root.iterdir() if p.is_dir() and re.match(r"^chunk_\d+$", p.name, re.I)]
    subdirs.sort(key=_chunk_subdir_sort_key)
    if not subdirs:
        raise SystemExit(
            f"No hay subcarpetas chunk_NNN en {root} (esperado p. ej. chunk_001, chunk_002)."
        )
    parts: List[np.ndarray] = []
    used: List[str] = []
    skipped: List[str] = []
    j_ref: Optional[int] = None

    for ch in subdirs:
        name = ch.name
        poses_p = ch / "poses.npy"
        meta_p = ch / "meta.json"
        if not poses_p.exists():
            skipped.append(f"{name} (sin poses.npy)")
            continue
        if not meta_p.exists():
            skipped.append(f"{name} (sin meta.json)")
            continue
        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            skipped.append(f"{name} (meta.json ilegible: {e})")
            continue
        vf = _read_valid_frames_from_meta(meta)
        try:
            arr = np.load(str(poses_p), allow_pickle=False)
        except Exception as e:
            skipped.append(f"{name} (poses.npy: {e})")
            continue
        if vf is None:
            vf = int(arr.shape[0])
        if vf == 0:
            skipped.append(f"{name} (frames válidos = 0)")
            continue
        if arr.size == 0 or arr.shape[0] == 0:
            skipped.append(f"{name} (poses.npy vacío)")
            continue
        if arr.ndim != 3 or int(arr.shape[2]) != 2:
            skipped.append(f"{name} (shape {arr.shape}, se esperaba [T,J,2])")
            continue
        if j_ref is None:
            j_ref = int(arr.shape[1])
        elif int(arr.shape[1]) != j_ref:
            raise SystemExit(
                f"Incompatibilidad de J entre chunks: {name} tiene J={arr.shape[1]}, "
                f"antes J={j_ref}."
            )
        parts.append(arr.astype(np.float32, copy=False))
        used.append(name)

    if not parts:
        raise SystemExit(
            "Ningún chunk aportó poses válidos (todos omitidos o vacíos). "
            f"Omitidos: {skipped}"
        )
    stacked = np.concatenate(parts, axis=0)
    return stacked, used, skipped


def _clip_dir_from_concatenated_array(
    arr: np.ndarray,
    folder_name: str,
    chunks_used: List[str],
    chunks_skipped: List[str],
) -> Path:
    """Escribe user_0/poses.npy + meta.json para secuencia ya concatenada."""
    if arr.ndim != 3 or int(arr.shape[2]) != 2:
        raise SystemExit(f"Tras concatenar, shape inválida: {arr.shape}")
    out = Path(tempfile.mkdtemp(prefix="test_model2_chunks_"))
    ud = out / "user_0"
    ud.mkdir(parents=True, exist_ok=False)
    np.save(str(ud / "poses.npy"), arr)
    meta: Dict[str, Any] = {
        "clip_name": folder_name,
        "source": "chunk_concat",
        "chunks_concat_order": chunks_used,
        "chunks_skipped": chunks_skipped,
        "users": [{"track_id": 0, "total_frames": int(arr.shape[0])}],
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return out


def _find_user_dirs(clip_dir: Path) -> List[Path]:
    return sorted(clip_dir.glob("user_*"), key=lambda x: x.name)


def _label_to_class_index(label_to_idx: Dict[Any, Any], label: int) -> Optional[int]:
    """Resuelve índice de clase; acepta claves int o str (p. ej. JSON antiguo)."""
    if label in label_to_idx:
        return int(label_to_idx[label])
    ls = str(label)
    if ls in label_to_idx:
        return int(label_to_idx[ls])
    return None


def _format_prob_robo(prob: float) -> str:
    """
    Formatea P(robo) para comparar con inferencia .engine (probabilidad cruda)
    y, en paralelo, el porcentaje con decimales extra si es muy bajo.
    """
    pct = prob * 100.0
    if pct == 0.0:
        pct_fmt = "0.00%"
    elif abs(pct) < 1e-6:
        pct_fmt = f"{pct:.6e}%"
    elif abs(pct) < 0.0001:
        pct_fmt = f"{pct:.8f}%"
    elif abs(pct) < 0.05:
        pct_fmt = f"{pct:.6f}%"
    else:
        pct_fmt = f"{pct:.2f}%"
    return f"{prob} ({pct_fmt})"


def _format_logits_detail(logits_1d: torch.Tensor, checkpoint: Dict[str, Any]) -> str:
    """
    Detalle por clase: logit crudo, softmax (multiclase) y sigmoide (one-vs-rest)
    de cada salida. Útil para distinguir si un 99% viene de un logit realmente
    alto o de que softmax reparte casi todo a una clase aunque los logits estén
    muy juntos (clases poco separadas).
    """
    label_to_idx: Dict[Any, Any] = checkpoint["label_to_idx"]
    idx_to_label: Dict[int, Any] = {}
    for lbl, idx in label_to_idx.items():
        try:
            idx_to_label[int(idx)] = lbl
        except (TypeError, ValueError):
            continue

    logits = logits_1d.detach().float().cpu().numpy()
    softmax = torch.softmax(logits_1d, dim=0).detach().float().cpu().numpy()
    sigmoid = torch.sigmoid(logits_1d).detach().float().cpu().numpy()

    parts: List[str] = []
    for i in range(int(logits.shape[0])):
        lbl = idx_to_label.get(i, "?")
        parts.append(
            f"clase[{i}]={lbl}: logit={logits[i]:+.4f}, "
            f"softmax={softmax[i] * 100.0:.2f}%, sigmoide={sigmoid[i] * 100.0:.2f}%"
        )
    logit_gap = float(np.max(logits) - np.partition(logits, -2)[-2]) if logits.shape[0] >= 2 else 0.0
    return " || ".join(parts) + f" || gap(top1-top2)={logit_gap:+.4f}"


def _prob_from_logits(logits_1d: torch.Tensor, checkpoint: Dict[str, Any]) -> float:
    task = checkpoint.get("task", "multiclass")
    label_to_idx: Dict[Any, Any] = checkpoint["label_to_idx"]
    positive_class = int(checkpoint.get("positive_class", 6))
    probs = torch.softmax(logits_1d, dim=0)
    if task == "binary":
        pos_idx = _label_to_class_index(label_to_idx, 1)
        if pos_idx is None:
            pos_idx = int(label_to_idx.get(1, 1))
        return float(probs[pos_idx].item())
    idx = _label_to_class_index(label_to_idx, positive_class)
    if idx is not None:
        return float(probs[idx].item())
    return float(probs.max().item())


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


def _materialize_temp_npy(
    user_dir: Path,
    pose_source: str,
    tmp_dir: Path,
    tid: int,
) -> Tuple[Path, Optional[Path], List[Path], int]:
    """Escribe .npy temporales (mismo contenido que entrenamiento) y devuelve rutas + buffer_len=T."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cleanup: List[Path] = []
    if pose_source == "filtered":
        dst = tmp_dir / f"_tmp_{tid}_poses.npy"
        shutil.copy2(user_dir / "poses.npy", dst)
        cleanup.append(dst)
        arr = np.load(dst)
        return dst, None, cleanup, int(arr.shape[0])
    pf = user_dir / "poses_full.npy"
    vmf = user_dir / "valid_mask.npy"
    if not pf.exists() or not vmf.exists():
        raise FileNotFoundError("Modo 'full' requiere poses_full.npy y valid_mask.npy")
    dst_p = tmp_dir / f"_tmp_{tid}_poses_full.npy"
    dst_m = tmp_dir / f"_tmp_{tid}_valid_mask.npy"
    shutil.copy2(pf, dst_p)
    shutil.copy2(vmf, dst_m)
    cleanup.extend([dst_p, dst_m])
    arr = np.load(dst_p)
    return dst_p, dst_m, cleanup, int(arr.shape[0])


def _safe_unlink(paths: List[Path]) -> None:
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass


def infer_user_track(
    *,
    checkpoint: Dict[str, Any],
    model: torch.nn.Module,
    user_dir: Path,
    pose_source: str,
    clip_name: str,
    device: torch.device,
    simple_preprocess: bool,
    use_temp_npy: bool,
    tmp_dir: Path,
) -> Tuple[int, float, torch.Tensor, float, float, List[Path]]:
    """
    Devuelve (track_id, prob_robo, logits_1d, clf_ms, total_ms, cleanup_paths).
    total_ms = desde lectura/prepare (o ds[0]) hasta softmax; clf_ms = solo forward+softmax.
    """
    tid = int(user_dir.name.split("_")[1])
    seq_len = int(checkpoint.get("seq_len", 64))
    label_to_idx: Dict[Any, int] = checkpoint["label_to_idx"]
    dummy_label = next(iter(label_to_idx.keys()))
    cleanup: List[Path] = []
    mask_p: Optional[Path] = None

    if use_temp_npy:
        pose_p, mask_p, cleanup, _buf = _materialize_temp_npy(
            user_dir, pose_source, tmp_dir, tid
        )
    else:
        pose_p = (user_dir / "poses.npy") if pose_source == "filtered" else (user_dir / "poses_full.npy")
        if pose_source == "full":
            vm = user_dir / "valid_mask.npy"
            mask_p = vm if vm.exists() else None

    try:
        with torch.no_grad():
            if _HAS_OPERATIONS and not simple_preprocess:
                ex = PoseExample(
                    pose_path=pose_p.resolve(),
                    label=int(dummy_label),
                    track_id=tid,
                    clip_name=clip_name,
                    category_str="infer",
                    valid_mask_path=mask_p.resolve() if mask_p else None,
                    users_in_clip=1,
                )
                try:
                    t_total0 = time.perf_counter()
                    ds = build_pose_dataset_for_eval(
                        [ex],
                        label_to_idx,
                        seq_len,
                        dataset_split="test",
                        checkpoint=checkpoint,
                    )
                    xb, _y = ds[0]
                    x = xb.unsqueeze(0).to(device)
                    t_clf0 = time.perf_counter()
                    logits = model(x)[0]
                    prob = _prob_from_logits(logits, checkpoint)
                    t_end = time.perf_counter()
                except Exception:
                    poses = _load_poses_array(user_dir, pose_source)
                    t_total0 = time.perf_counter()
                    x = _tensor_simple_pipeline(poses, seq_len).to(device)
                    t_clf0 = time.perf_counter()
                    logits = model(x)[0]
                    prob = _prob_from_logits(logits, checkpoint)
                    t_end = time.perf_counter()
            else:
                if use_temp_npy:
                    if pose_source == "filtered":
                        poses = np.load(pose_p)
                    else:
                        poses = np.load(pose_p)
                        if mask_p is not None:
                            vm = np.load(mask_p)
                            poses = poses[vm].copy()
                    if np.any(np.isnan(poses)):
                        poses = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    poses = _load_poses_array(user_dir, pose_source)
                t_total0 = time.perf_counter()
                x = _tensor_simple_pipeline(poses, seq_len).to(device)
                t_clf0 = time.perf_counter()
                logits = model(x)[0]
                prob = _prob_from_logits(logits, checkpoint)
                t_end = time.perf_counter()

        clf_ms = (t_end - t_clf0) * 1000.0
        total_ms = (t_end - t_total0) * 1000.0
        return tid, prob, logits.detach().cpu(), clf_ms, total_ms, cleanup
    except Exception:
        _safe_unlink(cleanup)
        raise


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    thr = float(np.clip(args.threshold_robbery, 0.0, 1.0))
    use_temp = not args.no_temp_npy

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Modelo clasificación: {model_path}")
    print(f"[INFO] Umbral robo: {thr:.0%}")
    print(f"[INFO] pose_source={args.pose_source} | temp_npy={use_temp} | simple_preprocess={args.simple_preprocess}")
    print(f"[INFO] eval_pipeline_operations={_HAS_OPERATIONS}")

    if args.skip_extract:
        if not args.extract_dir:
            raise SystemExit("--skip-extract requiere --extract-dir")
        clip_dir = Path(args.extract_dir).expanduser().resolve()
        if not clip_dir.is_dir():
            raise FileNotFoundError(clip_dir)
        extracted_here = False
    else:
        media_path, media_kind = _resolve_media_path(args)
        if media_kind == "npy":
            assert media_path is not None
            print(f"[INFO] Entrada poses (.npy): {media_path}")
            print("[INFO] Extracción YOLO omitida (se usa el .npy tal cual).")
            clip_dir = _clip_dir_from_input_npy(media_path, args.pose_source)
            extracted_here = True
        elif media_kind == "chunk_dir":
            assert media_path is not None
            if args.pose_source != "filtered":
                raise SystemExit("Entrada carpeta chunk_*: usa --pose-source filtered.")
            print(f"[INFO] Carpeta con chunks: {media_path}")
            print("[INFO] Extracción YOLO omitida (solo lectura/concat de poses).")
            stacked, used, skipped = _concatenate_chunk_poses(media_path)
            print(f"[INFO] Chunks usados ({len(used)}): {', '.join(used)}")
            if skipped:
                print(f"[INFO] Chunks omitidos ({len(skipped)}): {' | '.join(skipped)}")
            print(
                f"[INFO] Secuencia concatenada: T={stacked.shape[0]} frames, "
                f"J={stacked.shape[1]} joints"
            )
            clip_dir = _clip_dir_from_concatenated_array(
                stacked, media_path.name, used, skipped
            )
            extracted_here = True
        else:
            assert media_path is not None
            print(f"[INFO] Video: {media_path}")
            print(f"[INFO] YOLO pose (extracción): {args.yolo_pose_model}")
            if run_debug_extract is None:
                raise SystemExit(f"No se pudo importar pose_extractor_clean: {_IMPORT_ERR}")
            out = run_debug_extract(str(media_path), yolo_pose_model=args.yolo_pose_model)
            if out is None:
                raise SystemExit("Extracción fallida o sin usuarios válidos.")
            clip_dir = Path(out).resolve()
            extracted_here = True

    checkpoint = torch.load(model_path, map_location="cpu")
    print(
        "[INFO] Checkpoint: "
        f"task={checkpoint.get('task', '?')} | "
        f"num_classes={checkpoint.get('num_classes', '?')} | "
        f"label_to_idx={checkpoint.get('label_to_idx')} | "
        f"positive_class={checkpoint.get('positive_class', '?')}"
    )
    cfg = checkpoint.get("config", {})
    arch = cfg.get("arch", "tcn")
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint.get("num_classes", len(checkpoint["label_to_idx"])))
    model = build_model(arch, input_dim, num_classes, cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    clip_name = clip_dir.name
    meta_path = clip_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        clip_name = str(meta.get("clip_name", clip_name))

    user_dirs = _find_user_dirs(clip_dir)
    if not user_dirs:
        raise SystemExit(f"No hay user_* en {clip_dir}")

    print(f"[INFO] clip_dir={clip_dir}")

    tmp_root = clip_dir / ".infer_tmp"
    infer_times_ms: List[float] = []
    infer_total_times_ms: List[float] = []
    first_model_use_t0: Optional[float] = None
    first_track: Optional[int] = None
    first_eval_pass: Optional[int] = None
    eval_pass = 0
    results: List[Tuple[int, float, str]] = []
    first_result_line = True

    try:
        for ud in user_dirs:
            eval_pass += 1
            tid_preview = int(ud.name.split("_")[1])
            arr_t = _load_poses_array(ud, args.pose_source)
            if first_model_use_t0 is None:
                first_track = tid_preview
                first_eval_pass = eval_pass
                first_model_use_t0 = time.perf_counter()
                print(
                    f"[TRACE] Inicio uso modelo clf_model: frame={eval_pass}, "
                    f"track={tid_preview}, buffer_len={arr_t.shape[0]}"
                )

            tid, prob, logits, clf_ms, total_ms, cleanup = infer_user_track(
                checkpoint=checkpoint,
                model=model,
                user_dir=ud,
                pose_source=args.pose_source,
                clip_name=clip_name,
                device=device,
                simple_preprocess=args.simple_preprocess,
                use_temp_npy=use_temp,
                tmp_dir=tmp_root,
            )

            infer_times_ms.append(clf_ms)
            infer_total_times_ms.append(total_ms)
            res_tag = "Primer resultado clf_model" if first_result_line else "Resultado clf_model"
            first_result_line = False
            prob_fmt = _format_prob_robo(prob)
            logits_detail = _format_logits_detail(logits, checkpoint)
            print(
                f"[TRACE] {res_tag}: frame={eval_pass}, track={tid}, "
                f"P(robo)={prob_fmt} | clf={clf_ms:.2f} ms | total={total_ms:.2f} ms"
            )
            pred = "ROBO" if prob >= thr else "NO_ROBO"
            print(
                f"[RESULT] user_{tid} | P(robo)={prob_fmt} | decision={pred} "
                f"(umbral={thr*100:.0f}%) | clf {clf_ms:.2f} ms | total {total_ms:.2f} ms"
            )
            print(f"[LOGITS] user_{tid} | {logits_detail}")
            results.append((tid, prob, pred))
            _safe_unlink(cleanup)
    finally:
        if tmp_root.exists():
            try:
                shutil.rmtree(tmp_root, ignore_errors=True)
            except OSError:
                pass

    # [FIN] al estilo test_model: un track “principal” = mayor P(robo)
    best_tid, best_p, _ = max(results, key=lambda x: x[1])
    robbed_any = any(p >= thr for _, p, _ in results)

    if not robbed_any:
        print("[FIN] No se detecto robo por encima del umbral.")
    else:
        print(f"[FIN] Robo detectado: track_id={best_tid} | P(robo)={_format_prob_robo(best_p)}")
        if first_model_use_t0 is not None:
            elapsed_from_first_use_ms = (time.perf_counter() - first_model_use_t0) * 1000.0
            print(
                "[TRACE] Tiempo desde primer uso del modelo hasta FIN: "
                f"{elapsed_from_first_use_ms:.2f} ms"
            )

    if infer_times_ms:
        infer_arr = np.array(infer_times_ms, dtype=np.float64)
        print(
            "[PERF] Inference clf_model (por ventana): "
            f"n={infer_arr.size} | mean={infer_arr.mean():.2f} ms | "
            f"p95={np.percentile(infer_arr, 95):.2f} ms | "
            f"min={infer_arr.min():.2f} ms | max={infer_arr.max():.2f} ms"
        )
    else:
        print("[PERF] No hubo inferencias de clasificación para reportar tiempos.")

    if infer_total_times_ms:
        infer_total_arr = np.array(infer_total_times_ms, dtype=np.float64)
        print(
            "[PERF] Total seq->prob (preprocess + clf + softmax, por ventana): "
            f"n={infer_total_arr.size} | mean={infer_total_arr.mean():.2f} ms | "
            f"p95={np.percentile(infer_total_arr, 95):.2f} ms | "
            f"min={infer_total_arr.min():.2f} ms | max={infer_total_arr.max():.2f} ms"
        )
    else:
        print("[PERF] No hubo mediciones de tiempo total seq->prob.")

    if first_model_use_t0 is not None and first_track is not None and first_eval_pass is not None:
        print(
            f"[TRACE] Primer uso global del modelo en frame={first_eval_pass}, "
            f"track={first_track}."
        )
    else:
        print("[TRACE] El modelo de clasificación no llegó a ejecutarse.")

    if extracted_here and not args.keep_extract:
        shutil.rmtree(clip_dir, ignore_errors=True)
        print(f"[INFO] Carpeta de extracción eliminada: {clip_dir} (--keep-extract para conservarla)")


if __name__ == "__main__":
    main()
