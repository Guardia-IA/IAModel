#!/usr/bin/env python3
"""
Genera un vídeo con esqueleto superpuesto usando distintos motores de pose.

Backends soportados (según dependencias instaladas):
  - yolo        : Ultralytics YOLO pose (recomendado en este proyecto)
  - mediapipe   : pip install mediapipe
  - openpifpaf  : pip install openpifpaf  (estilo OpenPose, sin binario CMU)
  - openpose    : binario externo CMU OpenPose (OPENPOSE_BIN o --openpose-bin)

Uso:
    python test_poses.py video.mp4
    python test_poses.py video.mp4 --backend yolo --model yolo26n
    python test_poses.py video.mp4 --backend mediapipe
    python test_poses.py video.mp4 --compare
    python test_poses.py video.mp4 --list-backends
    python test_poses.py video.mp4 --list-yolo-models

Genera además un .txt junto al vídeo de salida (mismo nombre) con tiempo (s),
frames totales y frames con las 8 keypoints del tronco válidas (confianza alta).

Solo se dibujan keypoints y segmentos con confianza >= umbral (default 0.50).
Si un brazo no se ve bien, no se pinta ese punto ni la línea que lo une.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# KEEP_KPS de pose_extractor_clean: hombros, codos, muñecas, cadera (sin cabeza ni piernas)
# Local 0..7: hombro_L, hombro_R, codo_L, codo_R, muñeca_L, muñeca_R, cadera_L, cadera_R
NUM_UPPER_KPS = 8
UPPER_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 2), (2, 4), (1, 3), (3, 5), (0, 1), (0, 6), (1, 7), (6, 7),
]

# Índices en cada formato de backend -> orden local 0..7
COCO_UPPER_INDICES = [5, 6, 7, 8, 9, 10, 11, 12]
MEDIAPIPE_UPPER_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]
# OpenPose BODY_25: hombros, codos, muñecas, caderas
OPENPOSE_BODY25_UPPER_INDICES = [5, 2, 6, 3, 7, 4, 12, 9]

DEFAULT_VALID_CONF = 0.5

YOLO_MODEL_PRESETS: Dict[str, str] = {
    "yolo11n": "yolo11n-pose.pt",
    "yolo11s": "yolo11s-pose.pt",
    "yolo11m": "yolo11m-pose.pt",
    "yolo11l": "yolo11l-pose.pt",
    "yolo11x": "yolo11x-pose.pt",
    "yolo26n": "yolo26n-pose.pt",
    "yolo26s": "yolo26s-pose.pt",
    "yolo26m": "yolo26m-pose.pt",
    "yolo26l": "yolo26l-pose.pt",
    "yolo26x": "yolo26x-pose.pt",
    "yolov8n": "yolov8n-pose.pt",
    "yolov8s": "yolov8s-pose.pt",
    "yolov8m": "yolov8m-pose.pt",
    "yolov8l": "yolov8l-pose.pt",
    "yolov8x": "yolov8x-pose.pt",
}


@dataclass
class PoseDetection:
    """Una persona detectada en un frame."""
    keypoints: np.ndarray  # [J, 3] => x, y, conf (píxeles)
    connections: List[Tuple[int, int]]


@dataclass
class FrameTiming:
    read_ms: float = 0.0
    infer_ms: float = 0.0
    draw_ms: float = 0.0


@dataclass
class RunStats:
    backend: str
    model: str
    frames: int = 0
    persons_total: int = 0
    timings: List[FrameTiming] = field(default_factory=list)
    output_path: Optional[str] = None
    timing_txt_path: Optional[str] = None
    total_seconds: float = 0.0
    frames_valid: int = 0
    valid_conf_threshold: float = DEFAULT_VALID_CONF
    draw_conf_threshold: float = DEFAULT_VALID_CONF

    @property
    def infer_ms_avg(self) -> float:
        if not self.timings:
            return 0.0
        return float(np.mean([t.infer_ms for t in self.timings]))

    @property
    def total_ms_avg(self) -> float:
        if not self.timings:
            return 0.0
        return float(np.mean([t.read_ms + t.infer_ms + t.draw_ms for t in self.timings]))

    @property
    def fps_effective(self) -> float:
        avg = self.total_ms_avg
        return 1000.0 / avg if avg > 0 else 0.0


def _resolve_yolo_model(model: str) -> str:
    preset = YOLO_MODEL_PRESETS.get(model.lower())
    if preset:
        model = preset
    path = Path(model)
    if path.exists():
        return str(path.resolve())
    local = SCRIPT_DIR / model
    if local.exists():
        return str(local.resolve())
    return model


def _upper_indices_for_keypoints(n_joints: int, backend: str) -> List[int]:
    if backend == "mediapipe" or n_joints == 33:
        return MEDIAPIPE_UPPER_INDICES
    if backend == "openpose" and n_joints >= 25:
        return OPENPOSE_BODY25_UPPER_INDICES
    if n_joints >= 17:
        return COCO_UPPER_INDICES
    if n_joints >= 25:
        return OPENPOSE_BODY25_UPPER_INDICES
    return COCO_UPPER_INDICES


def _to_upper_body(det: PoseDetection, backend: str) -> PoseDetection:
    n = len(det.keypoints)
    indices = _upper_indices_for_keypoints(n, backend)
    if n <= max(indices):
        empty = np.zeros((NUM_UPPER_KPS, 3), dtype=np.float64)
        return PoseDetection(keypoints=empty, connections=UPPER_CONNECTIONS)
    upper = det.keypoints[np.array(indices, dtype=int)].astype(np.float64, copy=True)
    if upper.shape[1] == 2:
        upper = np.column_stack([upper, np.ones(len(upper))])
    return PoseDetection(keypoints=upper, connections=UPPER_CONNECTIONS)


def _normalize_detections(detections: Sequence[PoseDetection], backend: str) -> List[PoseDetection]:
    return [_to_upper_body(det, backend) for det in detections]


def _choose_main_person(detections: Sequence[PoseDetection]) -> Optional[PoseDetection]:
    if not detections:
        return None
    best: Optional[PoseDetection] = None
    best_area = -1.0
    for det in detections:
        kps = det.keypoints
        visible = (kps[:, 2] > 0) & (kps[:, 0] > 0) & (kps[:, 1] > 0)
        if not np.any(visible):
            continue
        xs = kps[visible, 0]
        ys = kps[visible, 1]
        area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
        if area > best_area:
            best_area = area
            best = det
    return best if best is not None else detections[0]


def _keypoint_visible(x: float, y: float, c: float, min_conf: float) -> bool:
    """Keypoint dibujable: coordenadas válidas y confianza >= umbral."""
    if np.isnan(x) or np.isnan(y) or np.isnan(c):
        return False
    if float(x) == 0.0 and float(y) == 0.0:
        return False
    return float(c) >= min_conf


def _frame_has_valid_upper_pose(det: Optional[PoseDetection], valid_conf: float) -> bool:
    """Frame válido = los 8 keypoints del tronco visibles con confianza >= umbral."""
    if det is None or det.keypoints.shape[0] != NUM_UPPER_KPS:
        return False
    for x, y, c in det.keypoints:
        if not _keypoint_visible(x, y, c, valid_conf):
            return False
    return True


def _draw_detections(
    frame: np.ndarray,
    detections: Sequence[PoseDetection],
    min_conf: float,
) -> None:
    for det in detections:
        kps = det.keypoints
        for i, j in det.connections:
            if i >= len(kps) or j >= len(kps):
                continue
            xi, yi, ci = kps[i]
            xj, yj, cj = kps[j]
            if not _keypoint_visible(xi, yi, ci, min_conf):
                continue
            if not _keypoint_visible(xj, yj, cj, min_conf):
                continue
            p1 = (int(xi), int(yi))
            p2 = (int(xj), int(yj))
            cv2.line(frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        for x, y, c in kps:
            if not _keypoint_visible(x, y, c, min_conf):
                continue
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)


def _overlay_hud(
    frame: np.ndarray,
    backend: str,
    model: str,
    frame_idx: int,
    timing: FrameTiming,
    stats: RunStats,
) -> None:
    lines = [
        f"backend: {backend}  model: {model}",
        f"frame: {frame_idx}  validos: {stats.frames_valid}/{stats.frames}",
        f"infer: {timing.infer_ms:.1f} ms  total frame: {timing.read_ms + timing.infer_ms + timing.draw_ms:.1f} ms",
        f"media infer: {stats.infer_ms_avg:.1f} ms  FPS~: {stats.fps_effective:.1f}",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


# ---------------------------------------------------------------------------
# Backend: YOLO
# ---------------------------------------------------------------------------

def _yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


def _make_yolo_predictor(model_path: str, device: str, imgsz: int):
    from ultralytics import YOLO

    model = YOLO(_resolve_yolo_model(model_path))
    use_half = device != "cpu"

    def predict(frame: np.ndarray) -> List[PoseDetection]:
        results = model.predict(
            source=frame,
            verbose=False,
            device=device,
            imgsz=imgsz,
            half=use_half,
        )
        detections: List[PoseDetection] = []
        if not results:
            return detections
        r = results[0]
        if r.keypoints is None or r.boxes is None:
            return detections
        kpts_xy = r.keypoints.xy.cpu().numpy()
        confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kpts_xy.shape[:2])
        for person_idx in range(kpts_xy.shape[0]):
            xy = kpts_xy[person_idx]
            conf = confs[person_idx]
            arr = np.column_stack([xy[:, 0], xy[:, 1], conf[: xy.shape[0]]])
            detections.append(PoseDetection(keypoints=arr, connections=[]))
        return detections

    return predict


# ---------------------------------------------------------------------------
# Backend: MediaPipe
# ---------------------------------------------------------------------------

MEDIAPIPE_MODELS_DIR = SCRIPT_DIR / "mediapipe_models"
MEDIAPIPE_TASK_MODELS: Dict[int, Tuple[str, str]] = {
    0: (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ),
    1: (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    ),
    2: (
        "pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    ),
}


def _mediapipe_tasks_available() -> bool:
    try:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks.python import vision  # noqa: F401
        return hasattr(mp, "tasks") and hasattr(vision, "PoseLandmarker")
    except ImportError:
        return False


def _mediapipe_legacy_available() -> bool:
    try:
        import mediapipe as mp  # noqa: F401
        return hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
    except ImportError:
        return False


def _mediapipe_available() -> bool:
    return _mediapipe_tasks_available() or _mediapipe_legacy_available()


def _ensure_mediapipe_task_model(model_complexity: int) -> Path:
    complexity = int(model_complexity)
    if complexity not in MEDIAPIPE_TASK_MODELS:
        raise ValueError(f"mediapipe-complexity debe ser 0, 1 o 2 (recibido: {model_complexity})")
    filename, url = MEDIAPIPE_TASK_MODELS[complexity]
    MEDIAPIPE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MEDIAPIPE_MODELS_DIR / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"[mediapipe] Descargando {filename} ...")
    tmp = dest.with_suffix(".task.part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return dest


def _landmarks_to_array(person_lms, width: int, height: int) -> np.ndarray:
    kps = []
    for lm in person_lms:
        vis = lm.visibility if lm.visibility is not None else lm.presence
        if vis is None:
            vis = 1.0
        kps.append([float(lm.x) * width, float(lm.y) * height, float(vis)])
    return np.asarray(kps, dtype=np.float64)


MEDIAPIPE_CONNECTIONS: List[Tuple[int, int]] = []  # no usado: se normaliza a UPPER_CONNECTIONS


def _make_mediapipe_predictor_tasks(model_complexity: int, fps: float):
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    model_path = str(_ensure_mediapipe_task_model(model_complexity))
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=5,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    ms_per_frame = 1000.0 / max(float(fps), 1.0)
    timestamp_ms = 0

    def predict(frame: np.ndarray) -> List[PoseDetection]:
        nonlocal timestamp_ms
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
        timestamp_ms += ms_per_frame
        if not result.pose_landmarks:
            return []
        detections: List[PoseDetection] = []
        for person_lms in result.pose_landmarks:
            arr = _landmarks_to_array(person_lms, w, h)
            detections.append(PoseDetection(keypoints=arr, connections=[]))
        return detections

    predict.close = landmarker.close  # type: ignore[attr-defined]
    return predict


def _make_mediapipe_predictor_legacy(model_complexity: int):
    import mediapipe as mp

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    def predict(frame: np.ndarray) -> List[PoseDetection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            return []
        h, w = frame.shape[:2]
        kps = []
        for lm in res.pose_landmarks.landmark:
            kps.append([lm.x * w, lm.y * h, lm.visibility])
        arr = np.asarray(kps, dtype=np.float64)
        return [PoseDetection(keypoints=arr, connections=[])]

    predict.close = pose.close  # type: ignore[attr-defined]
    return predict


def _make_mediapipe_predictor(model_complexity: int = 1, fps: float = 25.0):
    if _mediapipe_tasks_available():
        return _make_mediapipe_predictor_tasks(model_complexity, fps)
    if _mediapipe_legacy_available():
        return _make_mediapipe_predictor_legacy(model_complexity)
    raise RuntimeError("MediaPipe instalado pero sin API compatible (tasks ni solutions).")


# ---------------------------------------------------------------------------
# Backend: OpenPifPaf (OpenPose-style, pip)
# ---------------------------------------------------------------------------

def _openpifpaf_available() -> bool:
    try:
        import openpifpaf  # noqa: F401
        return True
    except ImportError:
        return False


def _make_openpifpaf_predictor(checkpoint: str = "shufflenetv2k30") -> Callable[[np.ndarray], List[PoseDetection]]:
    import openpifpaf
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = openpifpaf.Predictor(checkpoint=checkpoint, device=device)

    skeleton = predictor.model.meta.skeleton
    connections = [(int(a), int(b)) for a, b in skeleton]

    def predict(frame: np.ndarray) -> List[PoseDetection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions, _, _ = predictor.numpy_image(rgb)
        detections: List[PoseDetection] = []
        for pred in predictions:
            if pred.data is None:
                continue
            data = np.asarray(pred.data, dtype=np.float64)
            if data.shape[1] >= 3:
                arr = data[:, :3]
            else:
                arr = np.column_stack([data[:, 0], data[:, 1], np.ones(len(data))])
            detections.append(PoseDetection(keypoints=arr, connections=[]))
        return detections

    return predict


# ---------------------------------------------------------------------------
# Backend: CMU OpenPose (binario externo)
# ---------------------------------------------------------------------------

def _find_openpose_bin(explicit: Optional[str] = None) -> Optional[str]:
    if explicit and Path(explicit).exists():
        return explicit
    env = os.environ.get("OPENPOSE_BIN")
    if env and Path(env).exists():
        return env
    return shutil.which("openpose") or shutil.which("OpenPoseDemo")


def _openpose_available(openpose_bin: Optional[str] = None) -> bool:
    return _find_openpose_bin(openpose_bin) is not None


def _run_openpose_batch(
    video_path: str,
    openpose_bin: str,
    openpose_models: Optional[str] = None,
) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="openpose_out_"))
    cmd = [
        openpose_bin,
        "--video", video_path,
        "--write_json", str(tmp),
        "--display", "0",
        "--render_pose", "0",
    ]
    if openpose_models:
        cmd.extend(["--model_folder", openpose_models])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "OpenPose falló.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr: {proc.stderr[-2000:]}"
        )
    return tmp


def _load_openpose_json(json_path: Path, frame_shape: Tuple[int, int]) -> List[PoseDetection]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    h, w = frame_shape
    detections: List[PoseDetection] = []
    for person in data.get("people", []):
        flat = person.get("pose_keypoints_2d") or person.get("body_keypoints_2d")
        if not flat:
            continue
        vals = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
        if vals[:, 0].max() <= 1.5 and vals[:, 1].max() <= 1.5:
            vals[:, 0] *= w
            vals[:, 1] *= h
        n = len(vals)
        detections.append(PoseDetection(keypoints=vals, connections=[]))
    return detections


def _make_openpose_predictor_from_dir(json_dir: Path, frame_shape: Tuple[int, int], fps: float):
    json_files = sorted(json_dir.glob("*_keypoints.json"))
    if not json_files:
        json_files = sorted(json_dir.glob("*.json"))

    cache: Dict[int, List[PoseDetection]] = {}

    def predict(frame: np.ndarray) -> List[PoseDetection]:
        idx = len(cache)
        if idx < len(json_files):
            cache[idx] = _load_openpose_json(json_files[idx], frame_shape)
        return cache.get(idx, [])

    return predict


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class BackendInfo:
    name: str
    description: str
    available: Callable[[], bool]
    install_hint: str


BACKEND_INFO: Dict[str, BackendInfo] = {
    "yolo": BackendInfo(
        "yolo",
        "Ultralytics YOLO pose (usado en pose_extractor_clean)",
        _yolo_available,
        "pip install ultralytics",
    ),
    "mediapipe": BackendInfo(
        "mediapipe",
        "Google MediaPipe Pose (33 landmarks, API Tasks >=0.10)",
        _mediapipe_available,
        "pip install mediapipe",
    ),
    "openpifpaf": BackendInfo(
        "openpifpaf",
        "OpenPifPaf (Python, estilo OpenPose sin binario CMU)",
        _openpifpaf_available,
        "pip install openpifpaf",
    ),
    "openpose": BackendInfo(
        "openpose",
        "CMU OpenPose (binario externo, OPENPOSE_BIN)",
        lambda: _openpose_available(None),
        "Compila/instala OpenPose y exporta OPENPOSE_BIN=/ruta/OpenPoseDemo",
    ),
}


def _available_backends(openpose_bin: Optional[str] = None) -> List[str]:
    names = []
    for name, info in BACKEND_INFO.items():
        if name == "openpose":
            if _openpose_available(openpose_bin):
                names.append(name)
        elif info.available():
            names.append(name)
    return names


def _build_predictor(args: argparse.Namespace) -> Tuple[Callable[[np.ndarray], List[PoseDetection]], str]:
    backend = args.backend.lower()
    if backend == "yolo":
        if not _yolo_available():
            raise RuntimeError("YOLO no disponible. pip install ultralytics")
        model = args.model or "yolo26n-pose.pt"
        return _make_yolo_predictor(model, args.device, args.imgsz), model

    if backend == "mediapipe":
        if not _mediapipe_available():
            raise RuntimeError("MediaPipe no disponible. pip install mediapipe")
        model = f"complexity={args.mediapipe_complexity}"
        fps = float(getattr(args, "video_fps", 25.0))
        return _make_mediapipe_predictor(args.mediapipe_complexity, fps=fps), model

    if backend == "openpifpaf":
        if not _openpifpaf_available():
            raise RuntimeError("OpenPifPaf no disponible. pip install openpifpaf")
        model = args.openpifpaf_checkpoint
        return _make_openpifpaf_predictor(model), model

    if backend == "openpose":
        bin_path = _find_openpose_bin(args.openpose_bin)
        if not bin_path:
            raise RuntimeError(
                "OpenPose binario no encontrado. "
                "Define OPENPOSE_BIN o usa --openpose-bin / --backend openpifpaf"
            )
        raise RuntimeError(
            "OpenPose batch se prepara en process_video(); no usar _build_predictor directamente."
        )

    raise ValueError(f"Backend desconocido: {backend}")


def _default_output_path(video_path: str, backend: str, model: str) -> str:
    stem = Path(video_path).stem
    safe_model = Path(model).stem.replace("/", "_")
    out_dir = Path(video_path).parent
    return str(out_dir / f"{stem}_poses_{backend}_{safe_model}.mp4")


def _timing_txt_path(output_path: str) -> str:
    return str(Path(output_path).with_suffix(".txt"))


def _write_timing_txt(stats: RunStats) -> str:
    if not stats.output_path:
        return ""
    txt_path = _timing_txt_path(stats.output_path)
    pct = (100.0 * stats.frames_valid / stats.frames) if stats.frames else 0.0
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"tiempo_segundos={stats.total_seconds:.3f}\n")
        f.write(f"frames_total={stats.frames}\n")
        f.write(f"frames_validos={stats.frames_valid}\n")
        f.write(f"porcentaje_validos={pct:.1f}\n")
        f.write(f"umbral_confianza_valido={stats.valid_conf_threshold:.2f}\n")
        f.write(f"umbral_confianza_dibujo={stats.draw_conf_threshold:.2f}\n")
        f.write(
            "criterio_valido=8 keypoints tronco (hombros,codos,muñecas,caderas) "
            "visibles con confianza >= umbral\n"
        )
        f.write(
            "criterio_dibujo=solo puntos y segmentos con confianza >= umbral_dibujo; "
            "sin inventar brazos ocultos\n"
        )
    stats.timing_txt_path = txt_path
    return txt_path


def process_video(args: argparse.Namespace) -> RunStats:
    run_started = time.perf_counter()
    video_path = str(Path(args.video).resolve())
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Vídeo no encontrado: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    args.video_fps = fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    backend = args.backend.lower()
    openpose_json_dir: Optional[Path] = None
    openpose_prep_ms = 0.0

    if backend == "openpose":
        bin_path = _find_openpose_bin(args.openpose_bin)
        if not bin_path:
            raise RuntimeError("OpenPose binario no encontrado.")
        print(f"[openpose] Ejecutando binario batch (puede tardar): {bin_path}")
        t0 = time.perf_counter()
        openpose_json_dir = _run_openpose_batch(video_path, bin_path, args.openpose_models)
        openpose_prep_ms = (time.perf_counter() - t0) * 1000.0
        predictor = _make_openpose_predictor_from_dir(openpose_json_dir, (height, width), fps)
        model_label = "OpenPoseDemo"
    else:
        predictor, model_label = _build_predictor(args)

    output_path = args.output or _default_output_path(video_path, backend, model_label)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo crear vídeo de salida: {output_path}")

    stats = RunStats(
        backend=backend,
        model=model_label,
        output_path=output_path,
        valid_conf_threshold=args.valid_conf,
        draw_conf_threshold=args.draw_conf,
    )
    frame_idx = 0
    max_frames = args.max_frames if args.max_frames > 0 else total_frames or 10**9

    print(f"Procesando: {video_path}")
    print(f"  backend={backend}  model={model_label}  salida={output_path}")
    if total_frames:
        print(f"  resolución={width}x{height}  fps={fps:.2f}  frames~={total_frames}")

    try:
        while frame_idx < max_frames:
            t_read0 = time.perf_counter()
            ok, frame = cap.read()
            t_read1 = time.perf_counter()
            if not ok:
                break

            t_inf0 = time.perf_counter()
            raw_detections = predictor(frame)
            t_inf1 = time.perf_counter()

            t_draw0 = time.perf_counter()
            detections = _normalize_detections(raw_detections, backend)
            main_person = _choose_main_person(detections)
            if _frame_has_valid_upper_pose(main_person, args.valid_conf):
                stats.frames_valid += 1
            _draw_detections(frame, detections, min_conf=args.draw_conf)
            timing = FrameTiming(
                read_ms=(t_read1 - t_read0) * 1000.0,
                infer_ms=(t_inf1 - t_inf0) * 1000.0,
                draw_ms=0.0,
            )
            _overlay_hud(frame, backend, model_label, frame_idx, timing, stats)
            timing.draw_ms = (time.perf_counter() - t_draw0) * 1000.0
            stats.timings.append(timing)
            stats.frames += 1
            stats.persons_total += len(detections)

            writer.write(frame)
            frame_idx += 1

            if args.progress_every > 0 and frame_idx % args.progress_every == 0:
                print(
                    f"  frame {frame_idx}  infer={stats.infer_ms_avg:.1f} ms  "
                    f"FPS~={stats.fps_effective:.1f}"
                )
    finally:
        cap.release()
        writer.release()
        close_fn = getattr(predictor, "close", None)
        if callable(close_fn):
            close_fn()
        if openpose_json_dir and openpose_json_dir.exists() and not args.keep_openpose_json:
            shutil.rmtree(openpose_json_dir, ignore_errors=True)

    if backend == "openpose" and openpose_prep_ms > 0:
        print(f"  openpose batch prep: {openpose_prep_ms:.0f} ms (no incluido en infer/frame)")

    stats.total_seconds = time.perf_counter() - run_started
    _write_timing_txt(stats)
    return stats


def _print_stats(stats: RunStats) -> None:
    print("\n=== Resumen ===")
    print(f"Backend:     {stats.backend}")
    print(f"Modelo:      {stats.model}")
    print(f"Frames:      {stats.frames} total, {stats.frames_valid} validos (conf>={stats.valid_conf_threshold:.2f})")
    print(f"Dibujo:      solo keypoints/segmentos con conf>={stats.draw_conf_threshold:.2f}")
    print(f"Personas:    {stats.persons_total} (detecciones acumuladas)")
    print(f"Infer media: {stats.infer_ms_avg:.2f} ms/frame")
    print(f"Total media: {stats.total_ms_avg:.2f} ms/frame")
    print(f"FPS~:        {stats.fps_effective:.2f}")
    if stats.output_path:
        print(f"Salida:      {stats.output_path}")
    if stats.timing_txt_path:
        print(f"Tiempo (s):  {stats.total_seconds:.3f}  ->  {stats.timing_txt_path}")
        print(f"  frames_total={stats.frames}  frames_validos={stats.frames_valid}")


def _run_compare(args: argparse.Namespace) -> None:
    backends = _available_backends(args.openpose_bin)
    if not backends:
        print("No hay backends disponibles. Instala al menos ultralytics (YOLO).")
        sys.exit(1)

    selected = args.compare_backends or backends
    summary = []
    print("Backends a comparar:", ", ".join(selected))

    for backend in selected:
        if backend not in BACKEND_INFO:
            print(f"[skip] backend desconocido: {backend}")
            continue
        if backend == "openpose" and not _openpose_available(args.openpose_bin):
            print(f"[skip] {backend} no disponible")
            continue
        if backend != "openpose" and not BACKEND_INFO[backend].available():
            print(f"[skip] {backend} no instalado ({BACKEND_INFO[backend].install_hint})")
            continue

        run_args = argparse.Namespace(**vars(args))
        run_args.backend = backend
        run_args.output = None
        print(f"\n--- {backend} ---")
        try:
            stats = process_video(run_args)
            _print_stats(stats)
            summary.append(stats)
        except Exception as exc:
            print(f"[error] {backend}: {exc}")

    if len(summary) >= 2:
        print("\n=== Comparativa (inferencia) ===")
        summary.sort(key=lambda s: s.infer_ms_avg)
        for rank, s in enumerate(summary, start=1):
            print(
                f"{rank}. {s.backend:12s}  infer={s.infer_ms_avg:7.2f} ms  "
                f"FPS~={s.fps_effective:6.2f}  model={s.model}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera vídeo con esqueleto usando YOLO, MediaPipe, OpenPifPaf u OpenPose."
    )
    parser.add_argument("video", type=str, nargs="?", default=None, help="Ruta al vídeo de entrada")
    parser.add_argument(
        "--backend",
        choices=list(BACKEND_INFO.keys()),
        default="yolo",
        help="Motor de pose (default: yolo)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n-pose.pt",
        help=(
            "Modelo YOLO pose (ruta o preset: yolo26n, yolo26s, yolo11m, ...). "
            "Solo aplica con --backend yolo."
        ),
    )
    parser.add_argument("--output", type=str, default=None, help="Vídeo de salida (.mp4)")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Dispositivo YOLO/OpenPifPaf: 0, cpu, cuda (default: 0)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño inferencia YOLO (default: 640)")
    parser.add_argument(
        "--mediapipe-complexity",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="MediaPipe model_complexity (0=lite, 1=full, 2=heavy)",
    )
    parser.add_argument(
        "--openpifpaf-checkpoint",
        type=str,
        default="shufflenetv2k30",
        help="Checkpoint OpenPifPaf (default: shufflenetv2k30)",
    )
    parser.add_argument(
        "--openpose-bin",
        type=str,
        default=None,
        help="Ruta al binario OpenPoseDemo (alternativa a OPENPOSE_BIN)",
    )
    parser.add_argument(
        "--openpose-models",
        type=str,
        default=None,
        help="Carpeta model_folder de OpenPose (--model_folder)",
    )
    parser.add_argument(
        "--keep-openpose-json",
        action="store_true",
        help="No borrar JSON temporales de OpenPose",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=None,
        dest="draw_conf_arg",
        help=(
            "Confianza mínima para dibujar punto o segmento (default: mismo que --valid-conf). "
            "Si un brazo tiene confianza baja, no se dibuja."
        ),
    )
    parser.add_argument(
        "--valid-conf",
        type=float,
        default=DEFAULT_VALID_CONF,
        help=(
            "Confianza mínima en los 8 keypoints del tronco para contar frame válido "
            f"(default: {DEFAULT_VALID_CONF})"
        ),
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Limitar frames (0=todo)")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Imprimir progreso cada N frames (0=off, default: 50)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Ejecuta todos los backends instalados y compara tiempos",
    )
    parser.add_argument(
        "--compare-backends",
        nargs="+",
        default=None,
        help="Subconjunto para --compare (ej: yolo mediapipe openpifpaf)",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Lista backends y si están instalados",
    )
    parser.add_argument(
        "--list-yolo-models",
        action="store_true",
        help="Lista presets YOLO disponibles",
    )
    return parser.parse_args()


def _apply_conf_defaults(args: argparse.Namespace) -> None:
    """Por defecto, dibujar solo keypoints con la misma confianza alta que cuenta como válido."""
    args.draw_conf = (
        args.draw_conf_arg if args.draw_conf_arg is not None else args.valid_conf
    )


def main() -> None:
    args = parse_args()

    if args.list_yolo_models:
        print("Presets YOLO (--model <nombre>):")
        for key, val in YOLO_MODEL_PRESETS.items():
            print(f"  {key:10s} -> {val}")
        return

    if args.list_backends:
        print("Backends:")
        for name, info in BACKEND_INFO.items():
            ok = info.available() if name != "openpose" else _openpose_available(args.openpose_bin)
            status = "OK" if ok else "no instalado"
            print(f"  {name:12s} [{status:12s}]  {info.description}")
            if not ok:
                print(f"               -> {info.install_hint}")
        return

    if not args.video:
        print("Indica la ruta al vídeo o usa --list-backends / --list-yolo-models")
        sys.exit(1)

    if args.compare:
        _apply_conf_defaults(args)
        _run_compare(args)
        return

    _apply_conf_defaults(args)
    try:
        stats = process_video(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_stats(stats)


if __name__ == "__main__":
    main()
