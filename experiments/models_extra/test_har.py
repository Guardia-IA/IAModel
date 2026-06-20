#!/usr/bin/env python3
"""
Reconocimiento de acciones (HAR) con esqueleto: vídeo → YOLO pose → PySKL.

Pipeline:
  1. Extrae poses COCO-17 por frame con YOLO (mismo stack que movement_probe).
  2. Convierte al formato ``fake_anno`` que espera PySKL.
  3. Clasifica con PoseC3D preentrenado (por defecto NTU RGB+D 120).

PySKL: https://github.com/kennymckormick/pyskl

Instalación (entorno aparte; no viene en experiments/venv):

    git clone https://github.com/kennymckormick/pyskl.git
    cd pyskl
    conda env create -f pyskl.yaml   # o pyskl_310.yaml con Python 3.10
    conda activate pyskl
    pip install -e .
    export PYSKL_ROOT=/ruta/a/pyskl

Uso:

    python test_har.py /ruta/video.mp4
    python test_har.py video.mp4 --pyskl-root ~/pyskl --topk 5
    python test_har.py video.mp4 --save-poses poses.npz
    python test_har.py video.mp4 --load-poses poses.npz --device cpu

Notas:
  - Modelo por defecto: PoseC3D entrenado en NTU-120 (acciones de laboratorio).
    En vídeos de tienda las etiquetas son orientativas (p. ej. "reach into pocket",
    "pickup") pero no hay fine-tuning para tu dominio.
  - Se usa la persona con bbox más grande en cada frame.
  - PySKL usa mmcv (no mmengine). Requiere el entorno conda del repo.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
DEFAULT_LABEL_MAP = SCRIPT_DIR / "label_map_ntu120.txt"
DEFAULT_YOLO = "yolo26n-pose.pt"

DEFAULT_CHECKPOINT = (
    "https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/"
    "slowonly_r50_ntu120_xsub/joint.pth"
)
DEFAULT_CONFIG_REL = "configs/posec3d/slowonly_r50_ntu120_xsub/joint.py"

NUM_COCO_KEYPOINTS = 17


def _resolve_yolo_device(requested: str) -> str:
    req = (requested or "0").strip().lower()
    if req == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return requested if req != "cpu" else "0"
    except ImportError:
        pass
    print("[INFO] CUDA no disponible para YOLO; usando CPU.")
    return "cpu"


def _resolve_pyskl_device(requested: str) -> str:
    req = (requested or "0").strip().lower()
    if req == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            if req.isdigit():
                return f"cuda:{req}"
            return req if req.startswith("cuda") else "cuda:0"
    except ImportError:
        pass
    print("[INFO] CUDA no disponible para PySKL; usando CPU.")
    return "cpu"


def _resolve_pyskl_root(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"PYSKL root no encontrado: {root}")
        return root
    env = os.environ.get("PYSKL_ROOT", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        if root.is_dir():
            return root
    try:
        import pyskl

        candidate = Path(pyskl.__file__).resolve().parent.parent
        if (candidate / "configs").is_dir():
            return candidate
    except ImportError:
        pass
    raise FileNotFoundError(
        "No se encuentra PySKL. Clona el repo y define PYSKL_ROOT, o pasa --pyskl-root.\n"
        "  git clone https://github.com/kennymckormick/pyskl.git\n"
        "  cd pyskl && conda env create -f pyskl.yaml && conda activate pyskl\n"
        "  pip install -e ."
    )


def _load_label_map(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Label map no encontrado: {path}")
    labels = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not labels:
        raise ValueError(f"Label map vacío: {path}")
    return labels


def _pick_main_person(result) -> Optional[int]:
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def extract_poses_from_video(
    video_path: Path,
    yolo_model,
    device: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> Tuple[List[Dict[str, np.ndarray]], Tuple[int, int], float]:
    """Lista por frame: {'keypoints': (1,17,2), 'keypoint_scores': (1,17)} en píxeles."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    img_shape: Optional[Tuple[int, int]] = None
    pose_results: List[Dict[str, np.ndarray]] = []
    frame_i = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_i % max(1, stride) != 0:
            frame_i += 1
            continue

        h, w = bgr.shape[:2]
        img_shape = (h, w)
        res = yolo_model(bgr, verbose=False, device=device)[0]

        empty_kp = np.zeros((1, NUM_COCO_KEYPOINTS, 2), dtype=np.float32)
        empty_sc = np.zeros((1, NUM_COCO_KEYPOINTS), dtype=np.float32)

        person_idx = _pick_main_person(res)
        if person_idx is None:
            pose_results.append({"keypoints": empty_kp, "keypoint_scores": empty_sc})
        else:
            kpts = res.keypoints.xy.cpu().numpy()[person_idx : person_idx + 1].astype(np.float32)
            if res.keypoints.conf is not None:
                scores = res.keypoints.conf.cpu().numpy()[person_idx : person_idx + 1].astype(np.float32)
            else:
                scores = np.ones((1, NUM_COCO_KEYPOINTS), dtype=np.float32)
            if kpts.shape[1] != NUM_COCO_KEYPOINTS:
                raise RuntimeError(
                    f"Se esperaban {NUM_COCO_KEYPOINTS} keypoints COCO; got {kpts.shape[1]}"
                )
            pose_results.append({"keypoints": kpts, "keypoint_scores": scores})

        if max_frames is not None and len(pose_results) >= max_frames:
            break
        frame_i += 1

    cap.release()
    if not pose_results or img_shape is None:
        raise RuntimeError("No se extrajo ningún frame del vídeo.")
    return pose_results, img_shape, fps


def save_poses_npz(path: Path, pose_results: Sequence[Dict[str, np.ndarray]], img_shape: Tuple[int, int]) -> None:
    t = len(pose_results)
    kpts = np.zeros((t, NUM_COCO_KEYPOINTS, 2), dtype=np.float32)
    scores = np.zeros((t, NUM_COCO_KEYPOINTS), dtype=np.float32)
    for i, frm in enumerate(pose_results):
        kpts[i] = frm["keypoints"][0]
        scores[i] = frm["keypoint_scores"][0]
    np.savez(
        path,
        keypoints=kpts,
        keypoint_scores=scores,
        img_shape=np.array(img_shape, dtype=np.int32),
    )


def load_poses_npz(path: Path) -> Tuple[List[Dict[str, np.ndarray]], Tuple[int, int]]:
    data = np.load(path)
    kpts = data["keypoints"]
    sc = data["keypoint_scores"]
    h, w = [int(x) for x in data["img_shape"]]
    pose_results = []
    for i in range(len(kpts)):
        pose_results.append(
            {
                "keypoints": kpts[i : i + 1].astype(np.float32),
                "keypoint_scores": sc[i : i + 1].astype(np.float32),
            }
        )
    return pose_results, (h, w)


def build_pyskl_anno(
    pose_results: Sequence[Dict[str, np.ndarray]],
    img_shape: Tuple[int, int],
    num_person: int = 1,
) -> dict:
    """Construye fake_anno para ``pyskl.apis.inference_recognizer`` (formato PoseC3D)."""
    h, w = img_shape
    num_frame = len(pose_results)
    keypoint = np.zeros((num_person, num_frame, NUM_COCO_KEYPOINTS, 2), dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, NUM_COCO_KEYPOINTS), dtype=np.float16)

    for t, frm in enumerate(pose_results):
        keypoint[0, t] = frm["keypoints"][0]
        keypoint_score[0, t] = frm["keypoint_scores"][0]

    return dict(
        frame_dir="",
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality="Pose",
        total_frames=num_frame,
        keypoint=keypoint,
        keypoint_score=keypoint_score,
    )


def _require_pyskl():
    try:
        import mmcv  # noqa: F401
        import pyskl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Faltan PySKL / mmcv. Instala el entorno del repo:\n"
            "  git clone https://github.com/kennymckormick/pyskl.git\n"
            "  cd pyskl && conda env create -f pyskl.yaml && conda activate pyskl\n"
            "  pip install -e ."
        ) from exc


def init_pyskl_model(config_path: Path, checkpoint: str, device: str):
    _require_pyskl()
    import mmcv
    from pyskl.apis import init_recognizer

    config = mmcv.Config.fromfile(str(config_path))
    pipeline = config.data.test.pipeline
    config.data.test.pipeline = [x for x in pipeline if x.get("type") != "DecompressPose"]
    return init_recognizer(config, checkpoint, device)


def predict_action(model, fake_anno: dict) -> List[Tuple[int, float]]:
    _require_pyskl()
    from pyskl.apis import inference_recognizer

    if fake_anno.get("keypoint") is None:
        raise RuntimeError("No hay keypoints válidos para inferencia.")
    top5 = inference_recognizer(model, fake_anno)
    return [(int(idx), float(score)) for idx, score in top5]


def format_topk(
    top: Sequence[Tuple[int, float]],
    labels: Sequence[str],
    topk: int,
) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for idx, score in top[:topk]:
        name = labels[idx] if 0 <= idx < len(labels) else f"clase_{idx}"
        out.append((name, score))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HAR con YOLO pose + PySKL (skeleton).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", type=Path, help="Vídeo de entrada (.mp4, etc.)")
    p.add_argument(
        "--pyskl-root",
        type=Path,
        default=None,
        help="Raíz del repo clonado pyskl (o env PYSKL_ROOT)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config PySKL (.py). Por defecto PoseC3D NTU-120 joint",
    )
    p.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint PySKL (URL o ruta local)",
    )
    p.add_argument(
        "--label-map",
        type=Path,
        default=DEFAULT_LABEL_MAP,
        help="Etiquetas NTU (una por línea)",
    )
    p.add_argument("--yolo-model", default=DEFAULT_YOLO, help="Modelo YOLO pose (.pt)")
    p.add_argument("--device", default="0", help="Dispositivo YOLO/PySKL: 0, cpu, cuda:0…")
    p.add_argument("--stride", type=int, default=1, help="Usar 1 de cada N frames")
    p.add_argument("--max-frames", type=int, default=None, help="Límite de frames")
    p.add_argument("--topk", type=int, default=5, help="Mostrar top-K acciones")
    p.add_argument("--save-poses", type=Path, default=None, help="Guardar poses (.npz)")
    p.add_argument("--load-poses", type=Path, default=None, help="Cargar poses (.npz), saltar YOLO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.is_file():
        print(f"[ERROR] Vídeo no encontrado: {args.video}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(EXPERIMENTS_DIR))
    try:
        from config import YOLO_POSE_MODEL
    except ImportError:
        YOLO_POSE_MODEL = DEFAULT_YOLO

    yolo_name = args.yolo_model or YOLO_POSE_MODEL
    yolo_path = Path(yolo_name)
    if not yolo_path.is_file():
        for base in (EXPERIMENTS_DIR, Path.cwd(), SCRIPT_DIR):
            candidate = base / yolo_name
            if candidate.is_file():
                yolo_path = candidate
                break

    yolo_device = _resolve_yolo_device(args.device)
    pyskl_device = _resolve_pyskl_device(args.device)

    if args.load_poses:
        print(f"[1/3] Cargando poses desde {args.load_poses} …")
        pose_results, img_shape = load_poses_npz(args.load_poses)
    else:
        from ultralytics import YOLO

        print(f"[1/3] Extrayendo poses con YOLO ({yolo_path.name}) …")
        yolo = YOLO(str(yolo_path))
        pose_results, img_shape, fps = extract_poses_from_video(
            args.video,
            yolo,
            device=yolo_device,
            stride=args.stride,
            max_frames=args.max_frames,
        )
        valid = sum(1 for f in pose_results if f["keypoint_scores"][0].max() > 0.05)
        print(f"      Frames: {len(pose_results)}  |  con persona: {valid}  |  {img_shape[1]}x{img_shape[0]}")
        if fps > 0:
            print(f"      FPS vídeo: {fps:.1f}  |  stride: {args.stride}")
        if args.save_poses:
            save_poses_npz(args.save_poses, pose_results, img_shape)
            print(f"      Poses guardadas en {args.save_poses}")

    if len(pose_results) < 8:
        print("[WARN] Muy pocos frames; PoseC3D funciona mejor con ≥48 frames.", file=sys.stderr)

    pyskl_root = _resolve_pyskl_root(str(args.pyskl_root) if args.pyskl_root else None)
    config_path = args.config or (pyskl_root / DEFAULT_CONFIG_REL)
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        print(f"[ERROR] Config no encontrado: {config_path}", file=sys.stderr)
        return 1

    labels = _load_label_map(args.label_map)
    fake_anno = build_pyskl_anno(pose_results, img_shape)

    print("[2/3] Cargando PySKL …")
    print(f"      config: {config_path.relative_to(pyskl_root) if config_path.is_relative_to(pyskl_root) else config_path.name}")
    print(f"      checkpoint: {args.checkpoint}")
    print(f"      device: {pyskl_device}")
    model = init_pyskl_model(config_path, args.checkpoint, pyskl_device)

    print("[3/3] Inferencia skeleton …")
    try:
        top_raw = predict_action(model, fake_anno)
    except Exception as exc:
        print(f"[ERROR] Inferencia fallida: {exc}", file=sys.stderr)
        return 1

    top = format_topk(top_raw, labels, args.topk)
    if not top:
        print("[ERROR] Sin predicciones.", file=sys.stderr)
        return 1

    # Normalizar scores a porcentaje relativo dentro del top (PySKL devuelve logits/probs crudas)
    total = sum(s for _, s in top) or 1.0

    print()
    print(f"Vídeo: {args.video}")
    print(f"Modelo: PoseC3D / NTU-120 ({len(labels)} clases)")
    print()
    print("Top acciones:")
    for rank, (label, score) in enumerate(top, start=1):
        pct = 100.0 * score / total
        bar = "█" * int(min(pct, 100) / 5)
        print(f"  {rank}. {label:<45} {pct:5.1f}%  {bar}")
    print()
    print(f"→ Predicción: {top[0][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
