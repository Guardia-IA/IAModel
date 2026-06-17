#!/usr/bin/env python3
"""
Visor de vídeo con esqueleto YOLO26n y clasificación heurística de movimientos.

Detecta movimientos resumidos por brazo (reposo, extendido, recogido + zona corporal).
Sin clasificación de robo: solo exploración de patrones kinemáticos.

Uso:
    python movement_probe.py /ruta/video.mp4
    python movement_probe.py /ruta/video.mp4 --model yolo26n-pose.pt

Controles:
    Espacio  — pausa / continuar
    ← / →    — frame anterior / siguiente
    q / Esc  — salir
"""

from __future__ import annotations

import argparse
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(EXPERIMENTS_DIR))

from models_extra.movement_states import (  # noqa: E402
    ArmFrameMetrics,
    ArmSummary,
    BodyZone,
    FrameMovementResult,
    MovementClassifier,
    ZONE_LABELS_ES,
    build_masked_pose,
)

# COCO indices → KEEP_KPS local 0..7
KEEP_KPS = [5, 6, 7, 8, 9, 10, 11, 12]
CONNECTIONS = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7)]

DEFAULT_MODEL = "yolo26n-pose.pt"
DISPLAY_MAX_SIDE = 1280  # reduce RAM y acelera UI en vídeos 4K


def _resolve_device(requested: str) -> str:
    """Usa CUDA si está disponible; si no, CPU (evita crash con device=0 sin GPU)."""
    req = (requested or "0").strip().lower()
    if req == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return requested if req != "cpu" else "0"
    except ImportError:
        pass
    print("[INFO] CUDA no disponible; usando CPU para YOLO.")
    return "cpu"


def _downscale_for_display(bgr: np.ndarray, max_side: int = DISPLAY_MAX_SIDE) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    scale = max_side / max(h, w)
    return cv2.resize(
        bgr,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _resolve_model(model: str) -> str:
    p = Path(model)
    if p.exists():
        return str(p.resolve())
    for base in (EXPERIMENTS_DIR, SCRIPT_DIR, Path.cwd()):
        cand = base / model
        if cand.exists():
            return str(cand.resolve())
    return model


def _kp_visible_pt(pt: np.ndarray) -> bool:
    if pt is None or len(pt) < 2:
        return False
    x, y = float(pt[0]), float(pt[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    return not (x == 0.0 and y == 0.0)


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    points_px: np.ndarray,
    color_line: Tuple[int, int, int],
    color_pt: Tuple[int, int, int],
) -> None:
    for i, j in CONNECTIONS:
        if i >= len(points_px) or j >= len(points_px):
            continue
        if not _kp_visible_pt(points_px[i]) or not _kp_visible_pt(points_px[j]):
            continue
        draw.line(
            [(int(points_px[i][0]), int(points_px[i][1])), (int(points_px[j][0]), int(points_px[j][1]))],
            fill=color_line,
            width=3,
        )
    r = 5
    for pt in points_px:
        if not _kp_visible_pt(pt):
            continue
        x, y = int(pt[0]), int(pt[1])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color_pt, outline=color_pt)


@dataclass
class CachedFrame:
    bgr: np.ndarray
    pose_norm: Optional[np.ndarray]  # [8,2]
    confs: Optional[np.ndarray]  # [8]
    movement: Optional[FrameMovementResult]


class PoseExtractor:
    def __init__(self, model_path: str, device: str = "0") -> None:
        from ultralytics import YOLO

        self.model = YOLO(_resolve_model(model_path))
        self.device = _resolve_device(device)

    def extract_person(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        h, w = frame_bgr.shape[:2]
        results = self.model.predict(
            source=frame_bgr,
            verbose=False,
            device=self.device,
            imgsz=640,
            half=self.device != "cpu",
        )
        if not results or results[0].keypoints is None or results[0].boxes is None:
            return None, None
        r = results[0]
        if len(r.boxes) == 0:
            return None, None

        kpts_xy = r.keypoints.xyn.cpu().numpy()
        confs_all = r.keypoints.conf.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        best_i = 0
        best_area = -1.0
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            area = float((x2 - x1) * (y2 - y1))
            if area > best_area:
                best_area = area
                best_i = i

        pose_norm, local_confs = build_masked_pose(kpts_xy[best_i], confs_all[best_i], KEEP_KPS)
        return pose_norm, local_confs


class MovementProbeApp:
    def __init__(self, video_path: str, model_path: str, device: str) -> None:
        self.video_path = video_path
        self.model_path = model_path
        self.device = _resolve_device(device)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frames: List[CachedFrame] = []
        self._frames_lock = threading.Lock()
        self._slider_guard = False
        self.classifier = MovementClassifier()
        self.current_idx = 0
        self.playing = False
        self.extract_done = False
        self._extract_thread: Optional[threading.Thread] = None

        self.root = tk.Tk()
        self.root.title("Movement Probe — patrones de robo")
        self.root.geometry("1100x780")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._bind_keys()
        self.root.after(50, self._preview_first_frame)
        self._start_extraction()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text=f"Vídeo: {Path(self.video_path).name}").pack(side=tk.LEFT)
        ttk.Label(top, text=f"  Modelo: {Path(self.model_path).name}").pack(side=tk.LEFT)
        ttk.Label(top, text=f"  Device: {self.device}").pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(top, mode="determinate", length=220)
        self.progress.pack(side=tk.RIGHT, padx=8)
        self.progress_label = ttk.Label(top, text="Extrayendo poses…")
        self.progress_label.pack(side=tk.RIGHT)

        body = ttk.Frame(self.root, padding=8)
        body.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(body)
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = ttk.Frame(body, width=340)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))

        ttk.Label(side, text="Estado actual", font=("", 11, "bold")).pack(anchor=tk.W)
        self.state_text = tk.Text(side, height=8, width=42, wrap=tk.WORD, font=("Consolas", 10))
        self.state_text.pack(fill=tk.X, pady=4)

        ttk.Label(side, text="Historial (solo cambios)", font=("", 11, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.event_log = tk.Listbox(side, height=22, width=42, font=("Consolas", 10))
        self.event_log.pack(fill=tk.BOTH, expand=True, pady=4)

        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(fill=tk.X)

        self.btn_play = ttk.Button(ctrl, text="▶ Play", command=self._toggle_play)
        self.btn_play.pack(side=tk.LEFT)

        ttk.Button(ctrl, text="◀ Frame", command=lambda: self._step(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Frame ▶", command=lambda: self._step(1)).pack(side=tk.LEFT)

        self.frame_slider = ttk.Scale(
            ctrl,
            from_=0,
            to=max(0, self.total_frames - 1),
            orient=tk.HORIZONTAL,
            command=self._on_slider,
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12)

        self.frame_info = ttk.Label(ctrl, text="frame 0 / 0")
        self.frame_info.pack(side=tk.RIGHT)

        self.progress["maximum"] = max(1, self.total_frames)
        self.progress["value"] = 0

        self._seen_events: set[str] = set()

    def _frame_count(self) -> int:
        with self._frames_lock:
            return len(self.frames)

    def _get_frame(self, idx: int) -> Optional[CachedFrame]:
        with self._frames_lock:
            if 0 <= idx < len(self.frames):
                return self.frames[idx]
        return None

    def _preview_first_frame(self) -> None:
        """Muestra el primer frame al instante (sin pose) mientras arranca YOLO."""
        ok, bgr = self.cap.read()
        if not ok:
            return
        bgr = _downscale_for_display(bgr)
        cf = CachedFrame(bgr=bgr, pose_norm=None, confs=None, movement=None)
        with self._frames_lock:
            self.frames = [cf]
        self._update_slider_range(1)
        self._show_frame(0)
        self.progress_label.config(text="Extrayendo poses… (0/?)")

    def _update_slider_range(self, n_available: int) -> None:
        upper = max(0, n_available - 1)
        self.frame_slider.config(to=upper)
        if self.total_frames > 0:
            self.progress["maximum"] = self.total_frames

    def _set_slider_value(self, idx: int) -> None:
        n = self._frame_count()
        if n <= 0:
            return
        idx = int(np.clip(idx, 0, n - 1))
        self._slider_guard = True
        try:
            self.frame_slider.set(idx)
        finally:
            self._slider_guard = False

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<Left>", lambda e: self._step(-1))
        self.root.bind("<Right>", lambda e: self._step(1))
        self.root.bind("<Key-q>", lambda e: self._on_close())
        self.root.bind("<Escape>", lambda e: self._on_close())

    def _start_extraction(self) -> None:
        self._extract_thread = threading.Thread(target=self._extract_all_poses, daemon=True)
        self._extract_thread.start()

    def _extract_all_poses(self) -> None:
        try:
            extractor = PoseExtractor(self.model_path, self.device)
            classifier = MovementClassifier()
            cap = cv2.VideoCapture(self.video_path)
            idx = 0

            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                bgr = _downscale_for_display(bgr)
                pose_norm, confs = extractor.extract_person(bgr)
                movement = None
                if pose_norm is not None and confs is not None:
                    movement = classifier.classify(pose_norm, confs)
                cf = CachedFrame(bgr=bgr, pose_norm=pose_norm, confs=confs, movement=movement)
                with self._frames_lock:
                    if idx < len(self.frames):
                        self.frames[idx] = cf
                    else:
                        self.frames.append(cf)
                idx += 1
                self.root.after(0, self._on_frame_processed, idx)

            cap.release()
            self.classifier = classifier
            self.extract_done = True
            self.root.after(0, self._on_extraction_done, idx)
        except Exception as exc:
            msg = str(exc)
            self.root.after(0, lambda m=msg: self._show_extraction_error(m))

    def _on_frame_processed(self, n: int) -> None:
        self._update_slider_range(n)
        self.progress["value"] = n
        total = self.total_frames or n
        self.progress_label.config(text=f"Extrayendo… {n}/{total}")
        if self.playing:
            target = n - 1
            if self.current_idx < target:
                self._show_frame(self.current_idx + 1)
        elif n == 1:
            self._show_frame(0)

    def _show_extraction_error(self, message: str) -> None:
        self.progress_label.config(text="Error en extracción")
        messagebox.showerror("Error", message)

    def _on_extraction_done(self, n: int) -> None:
        self.total_frames = n
        self.progress["value"] = n
        self.progress_label.config(text=f"Listo — {n} frames")
        self._update_slider_range(n)
        self.extract_done = True
        if not self.playing:
            self._show_frame(min(self.current_idx, max(0, n - 1)))

    def _toggle_play(self) -> None:
        if self._frame_count() == 0:
            return
        self.playing = not self.playing
        self.btn_play.config(text="⏸ Pausa" if self.playing else "▶ Play")
        if self.playing:
            self._play_tick()

    def _play_tick(self) -> None:
        if not self.playing:
            return
        n = self._frame_count()
        if n == 0:
            self.root.after(100, self._play_tick)
            return
        if self.current_idx >= n - 1:
            if self.extract_done:
                self.playing = False
                self.btn_play.config(text="▶ Play")
                return
            self.root.after(100, self._play_tick)
            return
        self.current_idx += 1
        self._show_frame(self.current_idx)
        delay = max(20, int(1000 / self.fps))
        self.root.after(delay, self._play_tick)

    def _step(self, delta: int) -> None:
        n = self._frame_count()
        if n == 0:
            return
        self.playing = False
        self.btn_play.config(text="▶ Play")
        self.current_idx = int(np.clip(self.current_idx + delta, 0, n - 1))
        self._show_frame(self.current_idx)

    def _on_slider(self, _value: str) -> None:
        if self._slider_guard or self._frame_count() == 0:
            return
        self.playing = False
        self.btn_play.config(text="▶ Play")
        n = self._frame_count()
        self.current_idx = int(np.clip(float(_value), 0, n - 1))
        self._show_frame(self.current_idx, update_slider=False)

    def _render_frame_image(self, cf: CachedFrame) -> Image.Image:
        rgb = cv2.cvtColor(cf.bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        h, w = cf.bgr.shape[:2]

        if cf.pose_norm is not None:
            pts_px = cf.pose_norm.copy()
            pts_px[:, 0] *= w
            pts_px[:, 1] *= h
            draw_skeleton(draw, pts_px, (0, 255, 255), (255, 80, 80))

        if cf.movement is not None:
            y = 8
            for label in self._movement_lines(cf.movement):
                draw.text((10, y), label, fill=(255, 255, 255))
                y += 18
        status = f"Frame {self.current_idx + 1}"
        if not self.extract_done:
            status += "  [extrayendo poses…]"
        draw.text((10, h - 28), status, fill=(200, 200, 200))

        max_side = 720
        if max(img.size) > max_side:
            scale = max_side / max(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
        return img

    def _movement_lines(self, mv: FrameMovementResult) -> List[str]:
        if mv.status_line:
            return [mv.status_line]
        return ["—"]

    def _show_frame(self, idx: int, update_slider: bool = True) -> None:
        cf = self._get_frame(idx)
        if cf is None:
            return
        self.current_idx = idx
        img = self._render_frame_image(cf)
        photo = ImageTk.PhotoImage(img)
        self.video_label.config(image=photo)
        self.video_label.image = photo

        if update_slider:
            self._set_slider_value(idx)
        n = self._frame_count()
        total_txt = str(self.total_frames) if self.total_frames > 0 else "?"
        self.frame_info.config(text=f"frame {idx + 1} / {n}  (vídeo ~{total_txt})")

        if cf.movement is not None:
            self._update_side_panel(cf.movement, idx)
        elif not self.extract_done:
            self.state_text.delete("1.0", tk.END)
            self.state_text.insert(tk.END, f"Frame {idx}\n\nProcesando poses…")

    def _summary_label(self, summary: ArmSummary, zone: BodyZone) -> str:
        if summary == ArmSummary.NO_DATA:
            return "sin datos"
        if summary == ArmSummary.CALM:
            return "tranquilo (pegado al cuerpo)"
        if summary == ArmSummary.EXTENDED_NEAR:
            return "extendido cerca del cuerpo (normal)"
        if summary == ArmSummary.EXTENDED_AWAY:
            if zone == BodyZone.LOW:
                return "extendido alejado (hacia abajo)"
            return "extendido alejado del cuerpo"
        if summary == ArmSummary.TOWARD_POCKET:
            where = ZONE_LABELS_ES.get(zone, "bolsillo/cadera")
            return f"hacia {where}"
        if summary == ArmSummary.RETRACTED:
            where = ZONE_LABELS_ES.get(zone, "cuerpo")
            return f"recogido → {where}"
        return summary.value

    def _metrics_debug(self, label: str, m: ArmFrameMetrics) -> str:
        def f(v: float) -> str:
            return f"{v:.2f}" if np.isfinite(v) else "—"

        elbow = f"{m.elbow_flex_deg:.0f}°" if np.isfinite(m.elbow_flex_deg) else "—"
        dy = f"{m.wrist_hip_dy:.2f}" if np.isfinite(m.wrist_hip_dy) else "—"
        return (
            f"  {label}: body={f(m.body_dist)} reach={f(m.reach)} hip={f(m.hip_side_dist)} "
            f"conceal={f(m.conceal)} codo={elbow} dy={dy} pkt={m.pocket_score}"
        )

    def _update_side_panel(self, mv: FrameMovementResult, idx: int) -> None:
        self.state_text.delete("1.0", tk.END)
        scale_line = ""
        if mv.person_norm is not None:
            n = mv.person_norm
            thr = n.pocket_thresholds()
            scale_line = (
                f"\nEscala persona: {n.scale_ratio:.2f}  "
                f"(torso={n.torso_h:.3f}, hombros={n.shoulder_w:.3f})\n"
                f"Umbrales bolsillo adaptados: hip≤{thr.hip_side_max:.2f}  "
                f"conceal≥{thr.conceal_min:.2f}  codo_oblig={thr.require_elbow_bend}\n"
            )
        text = (
            f"Frame {idx}\n\n"
            f"{mv.status_line}\n\n"
            f"Izquierdo:  {self._summary_label(mv.left_summary, mv.left_zone)}\n"
            f"Derecho:    {self._summary_label(mv.right_summary, mv.right_zone)}\n"
            f"{scale_line}\n"
            f"Métricas (calibración):\n"
            f"{self._metrics_debug('Izq', mv.left_metrics)}\n"
            f"{self._metrics_debug('Der', mv.right_metrics)}\n"
        )
        self.state_text.insert(tk.END, text)

        for event in mv.events:
            key = f"{idx}:{event}"
            if key in self._seen_events:
                continue
            self._seen_events.add(key)
            self.event_log.insert(tk.END, f"[{idx:04d}] {event}")
            self.event_log.yview_moveto(1.0)

    def _on_close(self) -> None:
        self.playing = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visor de movimientos con YOLO pose + Tkinter")
    parser.add_argument("video", nargs="?", default=None, help="Ruta al vídeo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Modelo YOLO pose (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", default="0", help="Dispositivo YOLO: 0, cpu (auto→cpu si no hay CUDA)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = args.video
    if not video:
        root = tk.Tk()
        root.withdraw()
        video = filedialog.askopenfilename(
            title="Selecciona un vídeo",
            filetypes=[("Vídeo", "*.mp4 *.avi *.mkv *.mov"), ("Todos", "*.*")],
        )
        root.destroy()
    if not video:
        print("Indica un vídeo o selecciónalo en el diálogo.")
        sys.exit(1)
    if not Path(video).exists():
        print(f"No existe: {video}")
        sys.exit(1)

    app = MovementProbeApp(video, args.model, args.device)
    app.run()


if __name__ == "__main__":
    main()
