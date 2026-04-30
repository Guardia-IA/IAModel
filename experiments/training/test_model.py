import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, Deque, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    # Prioriza pipeline "operations": soporta arquitecturas nuevas (p.ej. ms_tcn)
    from .train_model_operations import (  # type: ignore[attr-defined]
        build_model,
        normalize_sequence,
        add_velocity,
        temporal_resize,
    )
except ImportError:
    try:
        from train_model_operations import (  # type: ignore[attr-defined]
            build_model,
            normalize_sequence,
            add_velocity,
            temporal_resize,
        )
    except ImportError:
        try:
            from .train_model import build_model, normalize_sequence, add_velocity, temporal_resize  # type: ignore[attr-defined]
        except ImportError:
            from train_model import build_model, normalize_sequence, add_velocity, temporal_resize  # type: ignore[attr-defined]


# Keypoints usados en entrenamiento (torso + brazos + cadera)
KEEP_KPS = [5, 6, 7, 8, 9, 10, 11, 12]
CRITICAL_KPS = [7, 8, 9, 10]  # codos + muñecas
MIN_KP_CONF = 0.5

# Conectividad sobre los 8 keypoints KEEP_KPS (índices locales 0..7)
SKELETON_EDGES = [
    (0, 1),  # hombro izq - hombro dcha
    (0, 2),  # hombro izq - codo izq
    (2, 4),  # codo izq - muñeca izq
    (1, 3),  # hombro dcha - codo dcha
    (3, 5),  # codo dcha - muñeca dcha
    (0, 6),  # hombro izq - cadera izq
    (1, 7),  # hombro dcha - cadera dcha
    (6, 7),  # cadera izq - cadera dcha
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prueba de modelo de robo en video real con YOLO pose + tracking + buffer temporal."
    )
    parser.add_argument("--model", required=True, help="Ruta al modelo .pt de clasificación (train_model).")
    parser.add_argument("--video", required=True, help="Ruta al vídeo .mp4 a analizar.")
    parser.add_argument(
        "--yolo-model",
        default="yolo11n-pose.pt",
        help="Modelo YOLO pose para detección/tracking (default: yolo11n-pose.pt).",
    )
    parser.add_argument(
        "--threshold-robbery",
        type=float,
        default=0.8,
        help="Umbral de decisión de robo (P(robo) >= threshold).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de vídeo de salida con overlay. Si no se indica, no guarda archivo.",
    )
    parser.add_argument("--no-show", action="store_true", help="No abrir ventana de visualización.")
    parser.add_argument(
        "--stop-on-robbery",
        action="store_true",
        help="Detener procesamiento al detectar robo por encima del umbral.",
    )
    parser.add_argument(
        "--stale-frames",
        type=int,
        default=45,
        help="Frames sin ver un track antes de limpiar su buffer.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=1080,
        help="Altura de visualización en pantalla (default: 1080). Solo afecta a la ventana, no al video guardado.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.35,
        help="Suavizado EMA de probabilidad por track (0..1). Menor = mas estable.",
    )
    parser.add_argument(
        "--activate-frames",
        type=int,
        default=6,
        help="Frames consecutivos por encima del umbral para activar estado ROBO.",
    )
    parser.add_argument(
        "--release-frames",
        type=int,
        default=10,
        help="Frames consecutivos por debajo de umbral_clear para salir de estado ROBO.",
    )
    parser.add_argument(
        "--threshold-clear",
        type=float,
        default=0.65,
        help="Umbral inferior para desactivar ROBO (histeresis).",
    )
    parser.add_argument(
        "--decision-window",
        type=int,
        default=12,
        help="Ventana de frames para decision robusta por track.",
    )
    parser.add_argument(
        "--window-min-hits",
        type=int,
        default=8,
        help="Minimo de frames >= umbral dentro de decision-window para activar robo.",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=20,
        help="Minimo de frames acumulados por track antes de permitir activar robo.",
    )
    parser.add_argument(
        "--min-motion",
        type=float,
        default=0.010,
        help="Movimiento medio minimo (coords normalizadas/frame) para permitir activar robo.",
    )
    parser.add_argument(
        "--force-threshold",
        type=float,
        default=0.95,
        help="Umbral alto de P(robo) para activar por alta confianza (bypass de algunos filtros).",
    )
    parser.add_argument(
        "--force-frames",
        type=int,
        default=3,
        help="Frames consecutivos por encima de force-threshold para activar por alta confianza.",
    )
    parser.add_argument(
        "--roi-region",
        type=str,
        default=None,
        help="JSON de regiones (formato {'regions':[{'polygon': [[x,y], ...]}]}). Si se indica, habilita logica entrar/salir ROI.",
    )
    parser.add_argument(
        "--roi-index",
        type=int,
        default=0,
        help="Indice de region a usar dentro del JSON (default: 0).",
    )
    parser.add_argument(
        "--roi-margin-px",
        type=float,
        default=12.0,
        help="Margen en pixeles para considerar dentro de ROI cerca del borde.",
    )
    parser.add_argument(
        "--robbery-hold-seconds",
        type=float,
        default=1.0,
        help="Tras salir de ROI, segundos minimos continuos con P(robo)>=umbral para confirmar robo.",
    )
    parser.add_argument(
        "--blink-period-frames",
        type=int,
        default=8,
        help="Periodo de parpadeo del resaltado en frames cuando hay robo activo.",
    )
    return parser.parse_args()


def draw_skeleton(frame: np.ndarray, pts_norm: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> None:
    h, w = frame.shape[:2]
    pts = np.zeros((len(KEEP_KPS), 2), dtype=np.int32)
    for i in range(len(KEEP_KPS)):
        pts[i, 0] = int(np.clip(pts_norm[i, 0] * w, 0, w - 1))
        pts[i, 1] = int(np.clip(pts_norm[i, 1] * h, 0, h - 1))

    for a, b in SKELETON_EDGES:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(p), 4, color, -1, cv2.LINE_AA)


def make_feature_tensor(seq_poses: np.ndarray, seq_len: int, device: torch.device) -> torch.Tensor:
    # seq_poses: [T, 8, 2] en coordenadas normalizadas 0..1
    poses = normalize_sequence(seq_poses)
    poses = add_velocity(poses)            # [T, 8, 4]
    poses = temporal_resize(poses, seq_len)
    t, j, d = poses.shape
    poses = poses.reshape(t, j * d).astype(np.float32)
    x = torch.from_numpy(poses).unsqueeze(0).to(device)  # [1, T, F]
    return x


def motion_score(seq_poses: np.ndarray) -> float:
    """Movimiento medio en coords normalizadas/frame sobre muñecas+codos."""
    if seq_poses.shape[0] < 2:
        return 0.0
    # índices locales en KEEP_KPS: codo/codo/muñeca/muñeca
    motion_kps = [2, 3, 4, 5]
    pts = seq_poses[:, motion_kps, :]  # [T,4,2]
    d = np.diff(pts, axis=0)           # [T-1,4,2]
    speed = np.linalg.norm(d, axis=-1) # [T-1,4]
    return float(np.mean(speed))


def load_roi_polygon(roi_region: str, roi_index: int, frame_w: int, frame_h: int) -> np.ndarray:
    p = Path(roi_region).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"No existe ROI JSON: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    regions = data.get("regions") or []
    if not regions:
        raise RuntimeError(f"El JSON ROI no contiene regiones: {p}")
    idx = max(0, min(int(roi_index), len(regions) - 1))
    poly = regions[idx].get("polygon") or []
    if len(poly) < 3:
        raise RuntimeError(f"Region ROI invalida (se requieren >=3 puntos), indice={idx}")
    arr = np.array(poly, dtype=np.float32)
    max_x = float(np.max(arr[:, 0]))
    max_y = float(np.max(arr[:, 1]))

    # 1) Si vienen normalizadas 0..1.
    if max_x <= 1.5 and max_y <= 1.5:
        arr[:, 0] *= frame_w
        arr[:, 1] *= frame_h
    else:
        # 2) Si el JSON trae resolución base, usarla (misma lógica que guardia-firmware).
        res = data.get("resolution")
        if isinstance(res, (list, tuple)) and len(res) == 2:
            try:
                base_w = float(res[0])
                base_h = float(res[1])
                if base_w > 0 and base_h > 0:
                    sx = frame_w / base_w
                    sy = frame_h / base_h
                    arr[:, 0] *= sx
                    arr[:, 1] *= sy
            except Exception:
                pass
        # 3) Fallback heurístico si aun parece fuera de escala.
        max_x2 = float(np.max(arr[:, 0]))
        max_y2 = float(np.max(arr[:, 1]))
        if max_x2 > frame_w * 1.2 or max_y2 > frame_h * 1.2:
            arr[:, 0] *= (frame_w / max(max_x2, 1.0))
            arr[:, 1] *= (frame_h / max(max_y2, 1.0))

    # Clamp final al frame para asegurar dibujo.
    arr[:, 0] = np.clip(arr[:, 0], 0, frame_w - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, frame_h - 1)
    return arr.astype(np.int32)


def inside_poly(poly_i: np.ndarray, x: float, y: float, margin_px: float = 0.0) -> bool:
    dist = cv2.pointPolygonTest(poly_i, (float(x), float(y)), True)
    return dist >= -float(margin_px)


def wrists_inside_roi(
    pts_norm: np.ndarray,
    poly_i: np.ndarray,
    frame_w: int,
    frame_h: int,
    margin_px: float,
) -> bool:
    """
    Comprueba si alguna muñeca cae dentro ROI.
    Índices locales de KEEP_KPS: 4=muñeca izq, 5=muñeca dcha.
    """
    for k in (4, 5):
        x = float(np.clip(pts_norm[k, 0] * frame_w, 0, frame_w - 1))
        y = float(np.clip(pts_norm[k, 1] * frame_h, 0, frame_h - 1))
        if inside_poly(poly_i, x, y, margin_px=margin_px):
            return True
    return False


def main() -> None:
    args = parse_args()
    threshold = float(np.clip(args.threshold_robbery, 0.0, 1.0))
    threshold_clear = float(np.clip(args.threshold_clear, 0.0, 1.0))
    ema_alpha = float(np.clip(args.ema_alpha, 0.0, 1.0))
    activate_frames = max(1, int(args.activate_frames))
    release_frames = max(1, int(args.release_frames))
    decision_window = max(3, int(args.decision_window))
    window_min_hits = max(1, int(args.window_min_hits))
    min_track_frames = max(1, int(args.min_track_frames))
    min_motion = max(0.0, float(args.min_motion))
    force_threshold = float(np.clip(args.force_threshold, 0.0, 1.0))
    force_frames = max(1, int(args.force_frames))
    if threshold_clear > threshold:
        threshold_clear = threshold

    model_path = Path(args.model).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"No existe video: {video_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Modelo clasificación: {model_path}")
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Umbral robo: {threshold:.0%}")
    print(
        f"[INFO] Histeresis: activar={activate_frames} fr @ {threshold:.0%} | "
        f"desactivar={release_frames} fr @ {threshold_clear:.0%} | ema_alpha={ema_alpha:.2f}"
    )
    print(
        f"[INFO] Filtro robusto: ventana={decision_window}, min_hits={window_min_hits}, "
        f"min_track_frames={min_track_frames}, min_motion={min_motion:.4f}"
    )
    print(
        f"[INFO] Activacion alta confianza: >= {force_threshold:.0%} durante {force_frames} frames."
    )
    print(f"[INFO] Hold de robo tras salir de ROI: {args.robbery_hold_seconds:.2f}s")

    checkpoint = torch.load(model_path, map_location=device)
    label_to_idx = checkpoint["label_to_idx"]
    seq_len = int(checkpoint.get("seq_len", 64))
    task = checkpoint.get("task", "multiclass")
    positive_class = int(checkpoint.get("positive_class", 6))
    cfg = checkpoint.get("config", {})
    arch = cfg.get("arch", "tcn")
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint.get("num_classes", len(label_to_idx)))

    clf_model = build_model(arch, input_dim, num_classes, cfg).to(device)
    clf_model.load_state_dict(checkpoint["model_state_dict"])
    clf_model.eval()

    if task != "binary":
        print("[WARN] El checkpoint no es binario; se usará la probabilidad de la clase 6 como referencia.")

    pos_idx = int(label_to_idx.get(1, label_to_idx.get(positive_class, 1)))
    yolo = YOLO(args.yolo_model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    writer = None
    panel_w = 360
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w + panel_w, h))
        print(f"[INFO] Guardando salida en: {out_path}")

    roi_poly = None
    if args.roi_region:
        roi_poly = load_roi_polygon(args.roi_region, args.roi_index, w, h)
        xmin, ymin = int(roi_poly[:, 0].min()), int(roi_poly[:, 1].min())
        xmax, ymax = int(roi_poly[:, 0].max()), int(roi_poly[:, 1].max())
        print(
            f"[INFO] ROI activa: {Path(args.roi_region).expanduser().resolve()} "
            f"(index={args.roi_index}, margen={args.roi_margin_px}px)"
        )
        print(f"[INFO] ROI bounds escalado: x=[{xmin},{xmax}] y=[{ymin},{ymax}] en frame {w}x{h}")

    buffers: Dict[int, Deque[np.ndarray]] = {}
    probs_raw: Dict[int, float] = {}
    probs_smooth: Dict[int, float] = {}
    hit_counts: Dict[int, int] = {}
    miss_counts: Dict[int, int] = {}
    robbery_active: Dict[int, bool] = {}
    force_counts: Dict[int, int] = {}
    in_region_prev: Dict[int, bool] = {}
    collect_enabled: Dict[int, bool] = {}
    infer_enabled: Dict[int, bool] = {}
    exited_roi_once: Dict[int, bool] = {}
    post_exit_high_frames: Dict[int, int] = {}
    hit_windows: Dict[int, Deque[int]] = {}
    track_frames: Dict[int, int] = {}
    motion_by_track: Dict[int, float] = {}
    last_seen: Dict[int, int] = {}
    robbed_track: int | None = None
    fixed_detect_prob_by_track: Dict[int, float] = {}
    peak_prob_by_track: Dict[int, float] = {}
    infer_times_ms: list[float] = []
    last_infer_ms_by_track: Dict[int, float] = {}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        result = yolo.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
        ids = []
        boxes = np.empty((0, 4), dtype=np.float32)
        kpts = np.empty((0, len(KEEP_KPS), 2), dtype=np.float32)
        confs = np.empty((0, 17), dtype=np.float32)

        if result.boxes is not None and result.boxes.id is not None and result.keypoints is not None:
            ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()
            kpts_all = result.keypoints.xyn.cpu().numpy()    # [N,17,2]
            confs = result.keypoints.conf.cpu().numpy()      # [N,17]
            if len(kpts_all) > 0:
                kpts = kpts_all[:, KEEP_KPS, :]

        # Limpieza de tracks no vistos
        for tid in list(last_seen.keys()):
            if frame_idx - last_seen[tid] > args.stale_frames:
                last_seen.pop(tid, None)
                buffers.pop(tid, None)
                probs_raw.pop(tid, None)
                probs_smooth.pop(tid, None)
                hit_counts.pop(tid, None)
                miss_counts.pop(tid, None)
                robbery_active.pop(tid, None)
                force_counts.pop(tid, None)
                in_region_prev.pop(tid, None)
                collect_enabled.pop(tid, None)
                infer_enabled.pop(tid, None)
                exited_roi_once.pop(tid, None)
                post_exit_high_frames.pop(tid, None)
                hit_windows.pop(tid, None)
                track_frames.pop(tid, None)
                motion_by_track.pop(tid, None)

        # Actualizar buffers + inferencia por track
        for i, tid in enumerate(ids):
            last_seen[tid] = frame_idx
            if tid not in buffers:
                buffers[tid] = deque(maxlen=seq_len)
                hit_windows[tid] = deque(maxlen=decision_window)
                track_frames[tid] = 0
                force_counts[tid] = 0
                in_region_prev[tid] = False
                if roi_poly is None:
                    collect_enabled[tid] = True
                    infer_enabled[tid] = True
                    # Sin ROI: considerar "post-salida" activo desde el inicio.
                    exited_roi_once[tid] = True
                else:
                    collect_enabled[tid] = False
                    infer_enabled[tid] = False
                    exited_roi_once[tid] = False
                post_exit_high_frames[tid] = 0

            # Lógica ROI por track: entrada -> empieza a acumular, salida -> habilita inferencia
            if roi_poly is not None:
                inside = wrists_inside_roi(
                    kpts[i],
                    roi_poly,
                    w,
                    h,
                    margin_px=args.roi_margin_px,
                )
                was_inside = in_region_prev.get(tid, False)
                if inside and not was_inside:
                    # Nueva entrada: reinicia buffer para empezar exactamente desde entrada.
                    buffers[tid].clear()
                    hit_windows[tid].clear()
                    hit_counts[tid] = 0
                    miss_counts[tid] = 0
                    force_counts[tid] = 0
                    robbery_active[tid] = False
                    collect_enabled[tid] = True
                    infer_enabled[tid] = False
                    exited_roi_once[tid] = False
                    post_exit_high_frames[tid] = 0
                    print(f"[ROI] Track {tid} entra en region (frame={frame_idx}).")
                elif (not inside) and was_inside:
                    infer_enabled[tid] = True
                    exited_roi_once[tid] = True
                    post_exit_high_frames[tid] = 0
                    print(f"[ROI] Track {tid} sale de region (frame={frame_idx}) -> activando inferencia.")
                in_region_prev[tid] = inside

            valid = all(confs[i][idx] > MIN_KP_CONF for idx in CRITICAL_KPS)
            if valid and collect_enabled.get(tid, False):
                buffers[tid].append(kpts[i].astype(np.float32))
            track_frames[tid] = track_frames.get(tid, 0) + 1

            if infer_enabled.get(tid, False) and len(buffers[tid]) >= max(4, min(seq_len, len(buffers[tid]))):
                seq = np.array(buffers[tid], dtype=np.float32)  # [T,8,2]
                with torch.no_grad():
                    x = make_feature_tensor(seq, seq_len, device)
                    t0 = time.perf_counter()
                    logits = clf_model(x)[0]
                    p = float(torch.softmax(logits, dim=0)[pos_idx].item())
                    infer_ms = (time.perf_counter() - t0) * 1000.0
                    infer_times_ms.append(infer_ms)
                    last_infer_ms_by_track[tid] = infer_ms
                probs_raw[tid] = p
                motion_by_track[tid] = motion_score(seq)
                prev_s = probs_smooth.get(tid, p)
                p_s = (1.0 - ema_alpha) * prev_s + ema_alpha * p
                probs_smooth[tid] = p_s
                peak_prob_by_track[tid] = max(peak_prob_by_track.get(tid, 0.0), p_s)

                hit_windows[tid].append(1 if p_s >= threshold else 0)
                window_hits = int(sum(hit_windows[tid]))

                if p_s >= threshold:
                    hit_counts[tid] = hit_counts.get(tid, 0) + 1
                    miss_counts[tid] = 0
                elif p_s <= threshold_clear:
                    miss_counts[tid] = miss_counts.get(tid, 0) + 1
                    hit_counts[tid] = 0
                else:
                    # Zona intermedia: no incrementa ninguno; mantiene estado.
                    hit_counts[tid] = 0
                    miss_counts[tid] = 0

                if p_s >= force_threshold:
                    force_counts[tid] = force_counts.get(tid, 0) + 1
                else:
                    force_counts[tid] = 0

                # Condición solicitada: tras salir de ROI, mantener > umbral durante >= X segundos
                hold_frames_needed = max(1, int(round(float(args.robbery_hold_seconds) * fps)))
                if exited_roi_once.get(tid, False):
                    if p_s >= threshold:
                        post_exit_high_frames[tid] = post_exit_high_frames.get(tid, 0) + 1
                    else:
                        post_exit_high_frames[tid] = 0
                else:
                    post_exit_high_frames[tid] = 0
                enough_hold = post_exit_high_frames.get(tid, 0) >= hold_frames_needed

                enough_track = track_frames.get(tid, 0) >= min_track_frames
                enough_motion = motion_by_track.get(tid, 0.0) >= min_motion
                enough_window = window_hits >= window_min_hits
                enough_consecutive = hit_counts.get(tid, 0) >= activate_frames
                enough_force = force_counts.get(tid, 0) >= force_frames
                # Regla principal solicitada: tras salir de ROI, mantener >= umbral durante X segundos.
                enough_primary_hold = enough_hold

                if (
                    (not robbery_active.get(tid, False))
                    and (
                        enough_primary_hold
                        or (enough_track and enough_motion and enough_window and enough_consecutive and enough_hold)
                        or enough_force
                    )
                ):
                    robbery_active[tid] = True
                if robbery_active.get(tid, False) and miss_counts.get(tid, 0) >= release_frames:
                    robbery_active[tid] = False

        # Dibujado personas
        blink_period = max(1, int(args.blink_period_frames))
        blink_on = ((frame_idx // blink_period) % 2) == 0
        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i].astype(int).tolist()
            p = probs_smooth.get(tid, probs_raw.get(tid, 0.0))
            is_robber = robbery_active.get(tid, False) or (p >= threshold)
            if is_robber:
                # Parpadeo muy visible: rojo/amarillo con grosor alto.
                color = (0, 0, 255) if blink_on else (0, 255, 255)
                thick = 8 if blink_on else 5
            else:
                color = (0, 220, 120)
                thick = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

            tag = f"ID {tid} | Robo {p*100:.1f}%"
            if tid in last_infer_ms_by_track:
                tag += f" | {last_infer_ms_by_track[tid]:.1f} ms"
            cv2.putText(frame, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

            if is_robber and robbed_track is None:
                robbed_track = tid
            if is_robber and tid not in fixed_detect_prob_by_track:
                fixed_detect_prob_by_track[tid] = p

        # Panel lateral
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (panel_w, h), (20, 20, 20), -1)
        cv2.putText(panel, "DETECCION ROBO", (25, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"Umbral: {threshold*100:.0f}%", (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        if roi_poly is not None:
            anyone_inside = any(in_region_prev.get(tid, False) for tid in in_region_prev.keys())
            roi_color = (0, 180, 255) if anyone_inside else (255, 255, 0)
            # Relleno semitransparente
            overlay = frame.copy()
            cv2.fillPoly(overlay, [roi_poly], color=roi_color)
            alpha = 0.28 if anyone_inside else 0.20
            frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)
            cv2.polylines(frame, [roi_poly], isClosed=True, color=roi_color, thickness=4, lineType=cv2.LINE_AA)
            for p in roi_poly:
                cv2.circle(frame, (int(p[0]), int(p[1])), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{'ROI ACTIVA' if anyone_inside else 'ROI'} | UMBRAL {threshold*100:.0f}%",
                (int(roi_poly[0][0]), max(25, int(roi_poly[0][1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                roi_color,
                2,
                cv2.LINE_AA,
            )

        top_tid = None
        top_p = 0.0
        if probs_smooth:
            top_tid, top_p = max(probs_smooth.items(), key=lambda x: x[1])
        top_active = (top_tid is not None and (robbery_active.get(top_tid, False) or (top_p >= threshold)))
        big_color = ((0, 0, 255) if blink_on else (0, 255, 255)) if top_active else (0, 200, 255)
        status = "ROBO DETECTADO" if top_active else "MONITORIZANDO"
        cv2.putText(panel, status, (25, 135), cv2.FONT_HERSHEY_DUPLEX, 0.85, big_color, 2, cv2.LINE_AA)
        cv2.putText(panel, f"{top_p*100:05.1f}%", (25, 210), cv2.FONT_HERSHEY_DUPLEX, 2.0, big_color, 3, cv2.LINE_AA)
        if top_tid is not None:
            cv2.putText(panel, f"Track: {top_tid}", (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)

        cv2.putText(panel, "Robos detectados:", (25, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
        top_list = sorted(probs_smooth.items(), key=lambda x: x[1], reverse=True)[:6]
        ytxt = 345
        for tid, p in top_list:
            if tid in fixed_detect_prob_by_track:
                p = fixed_detect_prob_by_track[tid]
            c = ((0, 0, 255) if blink_on else (0, 255, 255)) if (robbery_active.get(tid, False) or (p >= threshold)) else (180, 220, 180)
            cv2.putText(panel, f"ID {tid:>3}: {p*100:5.1f}%", (25, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
            ytxt += 32

        out_frame = np.hstack([frame, panel])

        if writer is not None:
            writer.write(out_frame)
        if not args.no_show:
            show_frame = out_frame
            disp_h = max(240, int(args.display_height))
            if out_frame.shape[0] != disp_h:
                scale = disp_h / float(out_frame.shape[0])
                disp_w = max(320, int(out_frame.shape[1] * scale))
                show_frame = cv2.resize(out_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Test Modelo Robo (YOLO Pose + Buffer)", show_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        if robbed_track is not None and args.stop_on_robbery:
            prob_alert = fixed_detect_prob_by_track.get(
                robbed_track,
                peak_prob_by_track.get(robbed_track, probs_smooth.get(robbed_track, 0.0)),
            )
            print(
                f"[ALERTA] Robo detectado en track {robbed_track} "
                f"(P(robo)={prob_alert*100:.1f}%) -> deteniendo."
            )
            break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    if robbed_track is None:
        print("[FIN] No se detecto robo por encima del umbral.")
    else:
        prob_final = fixed_detect_prob_by_track.get(
            robbed_track,
            peak_prob_by_track.get(robbed_track, probs_smooth.get(robbed_track, 0.0)),
        )
        print(
            f"[FIN] Robo detectado: track_id={robbed_track} | "
            f"P(robo)={prob_final*100:.1f}%"
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


if __name__ == "__main__":
    main()
