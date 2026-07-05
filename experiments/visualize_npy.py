"""
Visualiza archivos .npy de poses o timestamps con Tkinter (sin OpenCV para la ventana).

Formatos soportados:
  - poses.npy / poses_full.npy: (T, J, 2) coords normalizadas 0-1
  - (T, J, 3) o (T, 2, J, 3): x, y + confianza (píxeles o 0-1)
  - (T, 2, J, 2): dos usuarios
  - pose_frame_timestamps.npy: (T,) timestamps por frame (epoch segundos)

Opción --save poses: MP4 H.264 + yuv420p + faststart (compatible con WhatsApp).
Opción --save timestamps: CSV frame,timestamp_epoch,timestamp_iso,delta_ms
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk
from typing import Iterator, Literal, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageTk

NpyKind = Literal["poses_single", "poses_multi", "timestamps"]

# Conexiones: hombros, codos, muñecas, cadera (sin piernas)
CONNECTIONS = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7)]

W, H = 600, 600
MIN_KPT_CONF = 0.25  # si hay canal conf (…, 3), omitir keypoints por debajo
YELLOW = "\033[93m"
RESET = "\033[0m"


def _kp_visible(pt) -> bool:
    """Keypoint dibujable: coordenadas finitas y no (0,0) ausente."""
    if pt is None or len(pt) < 2:
        return False
    x, y = float(pt[0]), float(pt[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    return not (x == 0.0 and y == 0.0)


def draw_skeleton_pil(draw, points, color_line, color_pt):
    """Dibuja un esqueleto con PIL ImageDraw (omite keypoints NaN o ausentes)."""
    for start, end in CONNECTIONS:
        if start >= len(points) or end >= len(points):
            continue
        if not _kp_visible(points[start]) or not _kp_visible(points[end]):
            continue
        x1 = int(points[start][0] * W)
        y1 = int(points[start][1] * H)
        x2 = int(points[end][0] * W)
        y2 = int(points[end][1] * H)
        draw.line([(x1, y1), (x2, y2)], fill=color_line, width=2)
    r = 4
    for pt in points:
        if not _kp_visible(pt):
            continue
        x = int(pt[0] * W)
        y = int(pt[1] * H)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color_pt, outline=color_pt)


def _looks_like_unix_timestamps(data: np.ndarray) -> bool:
    if data.size < 2:
        return False
    finite = data[np.isfinite(data)]
    if finite.size < 2:
        return False
    lo, hi = float(np.min(finite)), float(np.max(finite))
    return 1.0e8 <= lo <= 1.0e11 and 1.0e8 <= hi <= 1.0e11


def _normalize_xy_to_unit(arr: np.ndarray) -> np.ndarray:
    """Convierte coords a 0-1 si parecen píxeles (max > 1.5)."""
    out = np.asarray(arr, dtype=np.float32).copy()
    finite = out[np.isfinite(out)]
    if finite.size == 0:
        return out
    if float(np.nanmax(finite)) <= 1.5:
        return out
    for ax in (0, 1):
        m = float(np.nanmax(out[..., ax]))
        if m > 1.0:
            out[..., ax] /= m
    return out


def _apply_conf_mask(xy: np.ndarray, conf: np.ndarray, min_conf: float = MIN_KPT_CONF) -> np.ndarray:
    """Pone NaN en keypoints con confianza baja."""
    out = xy.copy()
    bad = conf < float(min_conf)
    out[bad] = np.nan
    return out


def _parse_pose_array(data: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Normaliza a (T,J,2) o (T,2,J,2) en coords 0-1 para dibujo.
    Acepta último dim 2 (xy) o 3 (xy + conf).
    """
    data = np.asarray(data)

    if data.ndim == 3 and data.shape[-1] == 3:
        xy = _normalize_xy_to_unit(data[..., :2])
        conf = data[..., 2]
        xy = _apply_conf_mask(xy, conf)
        return xy, False

    if data.ndim == 3 and data.shape[-1] == 2:
        return _normalize_xy_to_unit(data), False

    if data.ndim == 4 and data.shape[1] == 2 and data.shape[-1] == 3:
        xy = _normalize_xy_to_unit(data[..., :2])
        conf = data[..., 2]
        for u in range(xy.shape[1]):
            xy[:, u] = _apply_conf_mask(xy[:, u], conf[:, u])
        return xy, True

    if data.ndim == 4 and data.shape[1] == 2 and data.shape[-1] == 2:
        return _normalize_xy_to_unit(data), True

    raise ValueError(f"shape de poses no reconocida: {data.shape}")


def detect_npy_kind(data: np.ndarray, npy_path: str) -> NpyKind:
    name = Path(npy_path).name.lower()
    if data.ndim == 3 and data.shape[-1] in (2, 3):
        return "poses_single"
    if data.ndim == 4 and data.shape[1] == 2 and data.shape[-1] in (2, 3):
        return "poses_multi"
    if data.ndim == 1:
        if "timestamp" in name or _looks_like_unix_timestamps(data):
            return "timestamps"
    raise ValueError(
        "Formato no soportado. Esperado (T,J,2), (T,J,3), (T,2,J,2), (T,2,J,3) "
        f"o timestamps (T,). Recibido {data.shape} en {Path(npy_path).name}"
    )


def load_pose_array(npy_path: str) -> Tuple[np.ndarray, bool, int]:
    raw = np.load(npy_path)
    kind = detect_npy_kind(raw, npy_path)
    if kind == "poses_single":
        data, multi = _parse_pose_array(raw)
        return data, multi, len(data)
    if kind == "poses_multi":
        data, multi = _parse_pose_array(raw)
        return data, multi, len(data)
    raise ValueError(
        f"{Path(npy_path).name} es timestamps (T,), no poses. "
        "Usa la visualización de timestamps o pasa poses.npy / poses_full.npy."
    )


def load_timestamps(npy_path: str) -> np.ndarray:
    data = np.load(npy_path)
    if detect_npy_kind(data, npy_path) != "timestamps":
        raise ValueError(
            f"{Path(npy_path).name} no parece pose_frame_timestamps: shape={data.shape}"
        )
    return np.asarray(data, dtype=np.float64).reshape(-1)


def _format_epoch(ts: float) -> str:
    if not np.isfinite(ts):
        return "NaN"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"
    except (OverflowError, OSError, ValueError):
        return str(ts)


def summarize_timestamps(ts: np.ndarray) -> dict:
    n = int(ts.size)
    deltas = np.diff(ts) if n > 1 else np.array([], dtype=np.float64)
    delta_ms = deltas * 1000.0 if deltas.size else np.array([], dtype=np.float64)
    summary = {
        "count": n,
        "min_epoch": float(np.nanmin(ts)) if n else None,
        "max_epoch": float(np.nanmax(ts)) if n else None,
        "duration_sec": float(ts[-1] - ts[0]) if n > 1 else 0.0,
        "mean_delta_ms": float(np.mean(delta_ms)) if delta_ms.size else None,
        "median_delta_ms": float(np.median(delta_ms)) if delta_ms.size else None,
        "est_fps": float(1000.0 / np.median(delta_ms)) if delta_ms.size and np.median(delta_ms) > 0 else None,
    }
    return summary


def _timestamp_rows(ts: np.ndarray) -> list[dict[str, Union[int, float, str]]]:
    rows: list[dict[str, Union[int, float, str]]] = []
    prev: float | None = None
    for i, raw in enumerate(ts):
        t = float(raw)
        delta_ms = (t - prev) * 1000.0 if prev is not None and np.isfinite(t) and np.isfinite(prev) else None
        rows.append(
            {
                "frame": i,
                "timestamp_epoch": t,
                "timestamp_iso": _format_epoch(t),
                "delta_ms": round(delta_ms, 3) if delta_ms is not None else "",
            }
        )
        prev = t
    return rows


def print_timestamp_summary(npy_path: str, ts: np.ndarray) -> None:
    s = summarize_timestamps(ts)
    print(f"Timestamps: {npy_path}")
    print(f"  frames: {s['count']}")
    if s["min_epoch"] is not None:
        print(f"  inicio: {s['min_epoch']:.6f}  ({_format_epoch(s['min_epoch'])})")
        print(f"  fin:    {s['max_epoch']:.6f}  ({_format_epoch(s['max_epoch'])})")
    print(f"  duración: {s['duration_sec']:.3f} s")
    if s["mean_delta_ms"] is not None:
        print(f"  Δt medio: {s['mean_delta_ms']:.2f} ms | mediana: {s['median_delta_ms']:.2f} ms")
    if s["est_fps"] is not None:
        print(f"  FPS estimado (mediana Δt): {s['est_fps']:.2f}")

    sibling_poses = []
    parent = Path(npy_path).parent
    for name in ("poses_full.npy", "poses.npy"):
        p = parent / name
        if p.is_file():
            try:
                poses = np.load(p)
                sibling_poses.append(f"{name}={poses.shape[0]} frames")
            except Exception:
                sibling_poses.append(f"{name}=?")
    if sibling_poses:
        print(f"  poses en la misma carpeta: {', '.join(sibling_poses)}")
        pf = parent / "poses_full.npy"
        if pf.is_file():
            n_poses = int(np.load(pf).shape[0])
            if n_poses != s["count"]:
                print(
                    f"  {YELLOW}[AVISO]{RESET} timestamps ({s['count']}) ≠ poses_full "
                    f"({n_poses}); pueden ser secuencias distintas."
                )


def save_timestamps_csv(ts: np.ndarray, out_path: Path) -> None:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _timestamp_rows(ts)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["frame", "timestamp_epoch", "timestamp_iso", "delta_ms"],
        )
        w.writeheader()
        w.writerows(rows)


def visualize_timestamps(npy_path: str) -> None:
    ts = load_timestamps(npy_path)
    print_timestamp_summary(npy_path, ts)
    rows = _timestamp_rows(ts)

    root = tk.Tk()
    root.title(f"Timestamps — {Path(npy_path).name}")
    root.geometry("920x520")

    summary = summarize_timestamps(ts)
    header = (
        f"{summary['count']} frames | "
        f"duración {summary['duration_sec']:.2f}s"
    )
    if summary["est_fps"] is not None:
        header += f" | ~{summary['est_fps']:.1f} FPS"
    tk.Label(root, text=header, anchor="w").pack(fill="x", padx=8, pady=6)

    cols = ("frame", "timestamp_epoch", "timestamp_iso", "delta_ms")
    tree = ttk.Treeview(root, columns=cols, show="headings", height=22)
    tree.heading("frame", text="frame")
    tree.heading("timestamp_epoch", text="epoch (s)")
    tree.heading("timestamp_iso", text="UTC")
    tree.heading("delta_ms", text="Δt (ms)")
    tree.column("frame", width=60, anchor="e")
    tree.column("timestamp_epoch", width=160, anchor="e")
    tree.column("timestamp_iso", width=280, anchor="w")
    tree.column("delta_ms", width=90, anchor="e")
    scroll = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 8))
    scroll.pack(side="right", fill="y", padx=(0, 8), pady=(0, 8))

    for row in rows:
        tree.insert(
            "",
            "end",
            values=(
                row["frame"],
                f"{row['timestamp_epoch']:.6f}",
                row["timestamp_iso"],
                row["delta_ms"],
            ),
        )

    def on_key(event):
        if event.char == "q" or event.keysym == "Escape":
            root.quit()
            root.destroy()

    root.bind("<KeyPress>", on_key)
    root.mainloop()


def render_frame(data: np.ndarray, idx: int, multi_user: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    if multi_user:
        pts1 = data[idx, 0]
        pts2 = data[idx, 1]
        draw_skeleton_pil(draw, pts1, (0, 255, 0), (0, 255, 100))
        draw_skeleton_pil(draw, pts2, (255, 165, 0), (255, 200, 0))
        draw.text((10, 8), "Usuario 1 (robo)=verde | Usuario 2=naranja", fill=(255, 255, 255))
    else:
        draw_skeleton_pil(draw, data[idx], (255, 255, 0), (0, 0, 255))

    draw.text((10, 38), f"Frame: {idx}", fill=(255, 255, 255))
    return img


def iter_frames_pil(data: np.ndarray, multi_user: bool, n_frames: int) -> Iterator[Image.Image]:
    for idx in range(n_frames):
        yield render_frame(data, idx, multi_user)


def _encode_mp4_ffmpeg(frames: Iterator[Image.Image], out_path: Path, fps: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    fps_i = max(1, min(60, int(round(fps)))) if fps > 0 else 25
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{W}x{H}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps_i),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        for img in frames:
            proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait(timeout=600)
        if code != 0:
            print(f"[ERROR] ffmpeg falló (código {code}). stderr:\n{err[-2000:]}", file=sys.stderr)
        return code == 0
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"[ERROR] ffmpeg: {e}", file=sys.stderr)
        return False


def _encode_mp4_cv2(frames: Iterator[Image.Image], out_path: Path, fps: float) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    fps_i = float(max(1, min(60, round(fps)))) if fps > 0 else 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_i, (W, H))
    if not writer.isOpened():
        return False
    try:
        for img in frames:
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        return True
    finally:
        writer.release()


def save_animation_mp4(
    data: np.ndarray,
    multi_user: bool,
    n_frames: int,
    out_path: Path,
    fps: float,
) -> bool:
    """MP4 orientado a compatibilidad WhatsApp (H.264 + yuv420p con ffmpeg)."""
    out_path = out_path.expanduser().resolve()
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = iter_frames_pil(data, multi_user, n_frames)
    if _encode_mp4_ffmpeg(frames, out_path, fps):
        return True
    frames = iter_frames_pil(data, multi_user, n_frames)
    if _encode_mp4_cv2(frames, out_path, fps):
        print("[INFO] Guardado con OpenCV (mp4v). Para mejor compatibilidad WhatsApp, instala ffmpeg.")
        return True
    print(
        "[ERROR] No se pudo escribir vídeo: instala ffmpeg (recomendado) o opencv-python.",
        file=sys.stderr,
    )
    return False


def visualize_skeleton(npy_path: str, fps: float = 20.0) -> None:
    try:
        data, multi_user, n_frames = load_pose_array(npy_path)
    except ValueError as e:
        print(e)
        return
    print(f"Visualizando: {npy_path}")
    print(f"Dimensiones: {data.shape}")

    delay_ms = max(1, int(1000 / fps)) if fps > 0 else 50
    frame_idx = [0]  # lista para poder modificar desde el closure

    root = tk.Tk()
    root.title("Verificador de Esqueletos")
    root.resizable(False, False)

    label = tk.Label(root)
    label.pack()

    def update():
        idx = frame_idx[0] % n_frames
        img = render_frame(data, idx, multi_user)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo
        frame_idx[0] += 1
        root.after(delay_ms, update)

    def on_key(event):
        if event.char == "q" or event.keysym == "Escape":
            root.quit()
            root.destroy()

    root.bind("<KeyPress>", on_key)
    root.after(0, update)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualiza poses (.npy) o timestamps (pose_frame_timestamps.npy)"
    )
    parser.add_argument(
        "npy_path",
        type=str,
        help="poses.npy, poses_full.npy, (T,J,3) xy+conf, o pose_frame_timestamps.npy",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="FPS a los que reproducir la secuencia (por defecto: 20)",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="OUT.mp4",
        help=(
            "Poses: guarda animación MP4 (H.264). "
            "Timestamps: guarda CSV frame,timestamp_epoch,timestamp_iso,delta_ms. "
            "Sin ruta: <stem>_vis.mp4 o <stem>.csv"
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.npy_path):
        print(f"Archivo no encontrado: {args.npy_path}")
        sys.exit(1)

    raw = np.load(args.npy_path)
    try:
        kind = detect_npy_kind(raw, args.npy_path)
    except ValueError as e:
        print(e)
        sys.exit(1)

    if kind == "timestamps":
        ts = np.asarray(raw, dtype=np.float64).reshape(-1)
        if args.save is not None:
            npy_p = Path(args.npy_path)
            out = npy_p.with_suffix(".csv") if args.save == "__auto__" else Path(args.save)
            save_timestamps_csv(ts, out)
            print_timestamp_summary(args.npy_path, ts)
            print(f"[OK] CSV guardado: {out.resolve()}")
            sys.exit(0)
        visualize_timestamps(args.npy_path)
        sys.exit(0)

    if args.save is not None:
        try:
            data, multi_user, n_frames = load_pose_array(args.npy_path)
        except ValueError as e:
            print(e)
            sys.exit(1)
        npy_p = Path(args.npy_path)
        if args.save == "__auto__":
            out = npy_p.with_name(f"{npy_p.stem}_vis.mp4")
        else:
            out = Path(args.save)
        print(f"Dimensiones: {data.shape} | frames={n_frames} | fps={args.fps}")
        if save_animation_mp4(data, multi_user, n_frames, out, args.fps):
            print(f"[OK] Vídeo guardado: {out.resolve()}")
        else:
            sys.exit(1)
        sys.exit(0)

    visualize_skeleton(args.npy_path, fps=args.fps)
