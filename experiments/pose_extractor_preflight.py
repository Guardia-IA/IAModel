#!/usr/bin/env python3
"""
Pre-chequeo antes de lanzar pose_extractor_clean.py.

Modo CSV (por defecto): busca CSVs bajo PATH_ROOTS, cuenta clips (sin ffprobe).
Modo data_result (--from-data-result): recorre data_result existente (meta.json + clip.mp4).

Uso rápido (CSV):
  python pose_extractor_preflight.py
  python pose_extractor_preflight.py --check-exists          # solo ¿existe el .mp4?
  python pose_extractor_preflight.py --save-txt informe.txt  # guardar + pantalla
  python pose_extractor_preflight.py --probe-videos          # ffprobe (lento, estimación precisa)

Uso data_result:
  python pose_extractor_preflight.py --from-data-result /ruta/data_result
  python pose_extractor_preflight.py --from-data-result /ruta --count-only
  python pose_extractor_preflight.py --from-data-result /ruta --probe-videos
"""
import re
import subprocess
import sys
import json
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import IO, List, TextIO

# Sin cargar YOLO ni pose_extractor_clean (evitar carga pesada)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PATH_ROOT,
    CSV_PATH,
    OUTPUT_BASE,
    CLIP_SCALE_HEIGHT,
    YOLO_POSE_MODEL,
    get_experiments,
    get_path_roots,
    get_data_result_roots,
)
from security import validate_csv_files

# Misma lógica que pose_extractor para detectar fila de inicio del CSV
HMS_PATTERN = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
DEFAULT_FPS = 12.0
DEFAULT_MAX_CLIP_SEC = 10.0


class _ReportTee:
    """Escribe en terminal y opcionalmente en un fichero de texto."""

    def __init__(self, stdout: TextIO, report_path: Path | None):
        self.stdout = stdout
        self.report_path = report_path
        self._file: TextIOWrapper | None = None
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = report_path.open("w", encoding="utf-8")
            self._file.write(f"# pose_extractor_preflight — {datetime.now().isoformat(timespec='seconds')}\n\n")

    def write(self, data: str) -> int:
        self.stdout.write(data)
        if self._file is not None:
            self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self.stdout.flush()
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


def _setup_report_tee(save_txt: Path) -> tuple[TextIO, _ReportTee]:
    original = sys.stdout
    tee = _ReportTee(original, save_txt.expanduser().resolve())
    sys.stdout = tee  # type: ignore[assignment]
    return original, tee


def _restore_stdout(original: TextIO, tee: _ReportTee | None) -> None:
    if tee is not None:
        sys.stdout = original
        tee.close()
        print(f"Informe guardado en: {tee.report_path}")


def _resolve_video_path(video_rel: str, base_dir: Path) -> Path:
    p = Path(str(video_rel).strip().strip('"').strip("'"))
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _csv_display_row(start_row: int, row_idx: int) -> int:
    """Número de fila en el CSV (1-based, coherente con pose_extractor_clean)."""
    return start_row + int(row_idx)


def _path_root_containing(csv_path: Path, path_roots: List[Path]) -> Path | None:
    """PATH_ROOT que contiene este CSV (el más específico / más largo)."""
    cp = csv_path.resolve()
    best: Path | None = None
    best_len = -1
    for pr in path_roots:
        rp = pr.expanduser().resolve()
        try:
            cp.relative_to(rp)
        except ValueError:
            continue
        n = len(rp.parts)
        if n > best_len:
            best_len = n
            best = rp
    return best


def _csv_display_path(csv_path: Path, path_roots: List[Path]) -> str:
    """
    Ruta legible del CSV para informes: relativa al PATH_ROOT que lo contiene e incluye .csv.
    """
    cp = csv_path.resolve()
    owner = _path_root_containing(cp, path_roots)
    if owner is not None:
        try:
            return cp.relative_to(owner).as_posix()
        except ValueError:
            pass
    return cp.name


def is_hms_format(val) -> bool:
    s = str(val).strip().strip('"').strip("'")
    return bool(HMS_PATTERN.match(s))


def find_start_row(csv_path: str) -> int:
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            parts = line.strip().split(",")
            if len(parts) >= 2 and is_hms_format(parts[1]):
                return i
    return 1


def hms_to_seconds(t: str) -> float:
    h, m, s = map(int, str(t).split(":"))
    return h * 3600 + m * 60 + s


def _is_full_clip_range(t_start: str, t_end: str) -> bool:
    return t_start == "00:00:00" and t_end == "00:00:00"


def get_video_fps(video_path: Path, cache: dict) -> float:
    """Obtiene FPS del vídeo con ffprobe. cache[str(path)] = fps. Si falla, devuelve 12.0."""
    key = str(video_path.resolve())
    if key in cache:
        return cache[key]
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
                key,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0 or not out.stdout or not out.stdout.strip():
            cache[key] = 12.0
            return 12.0
        rate = out.stdout.strip()
        if "/" in rate:
            num, den = rate.split("/", 1)
            cache[key] = float(num) / float(den)
        else:
            cache[key] = float(rate)
        return cache[key]
    except Exception:
        cache[key] = 12.0
        return 12.0


# Colores ANSI
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"{GREEN}[OK]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}[X]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[!]{RESET} {msg}")


def header(title: str) -> None:
    line = "=" * 60
    print(f"\n{BOLD}{line}\n{title}\n{line}{RESET}")


def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# Config opcional de límites por categoría (máx. clips a procesar por clase),
# compartida con pose_extractor_clean.py.
# Se busca en el directorio actual desde donde se ejecuta el script.
POSE_EXTRACTION_CONFIG = "config_pose_extraction.json"


def _load_category_limits() -> dict[str, int]:
    """
    Lee config_pose_extraction.json si existe y devuelve un dict {cat_str: max_clips}.
    Valores:
      - null/None  -> sin límite (se procesan todos)
      - 0          -> no se procesa ninguno
      - >0         -> máximo N clips para esa categoría
    """
    limits: dict[str, int] = {}
    cfg_path = Path(POSE_EXTRACTION_CONFIG)
    if not cfg_path.exists():
        return limits
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return limits
    cat_cfg = data.get("category_limits", {})
    if isinstance(cat_cfg, dict):
        for k, v in cat_cfg.items():
            try:
                cat_str = str(int(k))
            except (TypeError, ValueError):
                continue
            if v is None:
                continue
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if n >= 0:
                limits[cat_str] = n
    return limits


# 8 keypoints = KEEP_KPS en pose_extractor_clean: hombros (5,6), codos (7,8), muñecas (9,10), cadera (11,12). No los 17.
# Tiempo medio por frame: solo inferencia pose (esos 8 keypoints). Unidad: s/frame. Referencia 1080p.
# pt = modelo YOLO .pt (PyTorch); engine = TensorRT (.engine) generado en engine/.
SEC_PER_FRAME_PT = {
    "cpu": {"n": 1.2, "s": 1.6, "m": 2.2, "l": 3.5, "x": 5.0},
    "cuda": {"n": 0.020, "s": 0.028, "m": 0.040, "l": 0.070, "x": 0.12},
}
# Valores orientativos: TensorRT suele ser ~2–3× más rápido en GPU.
SEC_PER_FRAME_ENGINE = {
    "cpu": SEC_PER_FRAME_PT["cpu"],  # en CPU no aplica engine; igual que .pt
    "cuda": {"n": 0.010, "s": 0.016, "m": 0.022, "l": 0.035, "x": 0.06},
}
FFMPEG_SEC_PER_CLIP = 3.0
COPY_SEC_PER_CLIP = 0.5  # re-extracción: copia clip.mp4, sin FFmpeg


def _resolve_data_result_root(path: Path) -> Path:
    """Acepta .../data_result o OUTPUT_BASE (con data_result dentro)."""
    p = path.expanduser().resolve()
    if p.name == "data_result":
        return p
    nested = p / "data_result"
    if nested.is_dir():
        return nested
    return p


def _clip_duration_from_meta(meta: dict | None) -> tuple[float, float, int]:
    """Duración/fps/frames solo desde meta.json (sin ffprobe)."""
    fps = 12.0
    frames = 0
    dur = 0.0
    if not meta:
        return dur, fps, frames
    try:
        fps = float(meta.get("fps") or 0.0)
    except (TypeError, ValueError):
        fps = 0.0
    for key in ("frame_count", "video_frame_count"):
        try:
            frames = int(meta.get(key) or 0)
        except (TypeError, ValueError):
            frames = 0
        if frames > 0:
            break
    try:
        dur = float(meta.get("clip_duration") or 0.0)
    except (TypeError, ValueError):
        dur = 0.0
    if dur <= 0 and fps > 0 and frames > 0:
        dur = frames / fps
    if fps <= 0:
        fps = 12.0
    if frames <= 0 and dur > 0:
        frames = int(dur * fps + 0.5)
    if dur <= 0 and frames > 0 and fps > 0:
        dur = frames / fps
    return dur, fps, frames


def _clip_duration_seconds(clip_mp4: Path, meta: dict | None, fps_cache: dict) -> tuple[float, float, int]:
    """Devuelve (duración_s, fps, frames_aprox). Usa meta; ffprobe solo si faltan datos."""
    dur, fps, frames = _clip_duration_from_meta(meta)
    if dur > 0 and frames > 0:
        return dur, fps, frames

    fps_probe = get_video_fps(clip_mp4, fps_cache)
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(clip_mp4),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        dur_probe = float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else 0.0
    except Exception:
        dur_probe = 0.0
    if dur_probe > 0:
        dur = dur_probe
    if fps <= 0:
        fps = fps_probe
    if frames <= 0 and dur > 0 and fps > 0:
        frames = int(dur * fps + 0.5)
    if fps <= 0:
        fps = 12.0
    if frames <= 0 and dur > 0:
        frames = int(dur * fps + 0.5)
    if dur <= 0 and frames > 0:
        dur = frames / fps
    return dur, fps, frames


def scan_data_result_inventory(
    data_result_root: Path,
    *,
    count_only: bool = False,
    probe_videos: bool = False,
) -> dict:
    """
    Recorre data_result/{cat}/{clip}/ con meta.json + clip.mp4.

    count_only=True: solo comprueba que existan ambos ficheros (rápido, sin leer meta ni ffprobe).
    probe_videos=True: si meta no tiene frames/duración, usa ffprobe (lento).
    Por defecto: lee meta.json pero no llama ffprobe.
    """
    base = _resolve_data_result_root(data_result_root)
    if not base.is_dir():
        raise FileNotFoundError(f"No existe data_result: {base}")

    by_category: dict[int, int] = {}
    total_clips = 0
    total_frames = 0
    total_seconds = 0.0
    fps_cache: dict = {}
    incomplete: list[str] = []
    meta_missing_stats = 0

    for cat_dir in sorted(base.iterdir()):
        if not cat_dir.is_dir():
            continue
        try:
            cat = int(cat_dir.name)
        except ValueError:
            continue
        for clip_dir in sorted(cat_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            meta_path = clip_dir / "meta.json"
            clip_mp4 = clip_dir / "clip.mp4"
            has_meta = meta_path.is_file()
            has_clip = clip_mp4.is_file()
            if has_meta and has_clip:
                by_category[cat] = by_category.get(cat, 0) + 1
                total_clips += 1
                if count_only:
                    continue
                meta = None
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    meta = None
                if probe_videos:
                    dur, _fps, frames = _clip_duration_seconds(clip_mp4, meta, fps_cache)
                else:
                    dur, _fps, frames = _clip_duration_from_meta(meta)
                    if frames <= 0 and dur <= 0:
                        meta_missing_stats += 1
                total_frames += max(frames, 0)
                total_seconds += max(dur, 0.0)
            elif has_meta or has_clip:
                incomplete.append(str(clip_dir.relative_to(base)))

    return {
        "base": base,
        "by_category": by_category,
        "total_clips": total_clips,
        "total_folders": total_clips + len(incomplete),
        "total_frames": total_frames,
        "total_seconds": total_seconds,
        "incomplete": incomplete,
        "meta_missing_stats": meta_missing_stats,
        "count_only": count_only,
    }


def _print_time_estimates(
    *,
    total_clips: int,
    total_frames: int,
    device: str,
    model_stem: str,
    ffmpeg_sec_per_clip: float,
    label_suffix: str = "",
) -> None:
    sec_per_frame_pt = get_sec_per_frame(device, model_stem, backend="pt")
    inf_pt = total_frames * sec_per_frame_pt
    ffmpeg_pt = total_clips * ffmpeg_sec_per_clip
    total_pt = inf_pt + ffmpeg_pt

    model_size = get_model_size(model_stem)
    print(f"  [YOLO .pt]    Tiempo medio por frame (8 keypoints): {CYAN}{sec_per_frame_pt:.3f} s/frame{RESET} (YOLO-{model_size}, {device.upper()})")
    if ffmpeg_sec_per_clip > 0:
        print(f"               Frames: {total_frames} → Inferencia: ~{inf_pt/60:.0f} min  |  FFmpeg: ~{ffmpeg_pt/60:.0f} min")
    else:
        print(f"               Frames: {total_frames} → Inferencia: ~{inf_pt/60:.0f} min  |  Copia clips: ~{ffmpeg_pt/60:.0f} min")

    sec_per_frame_eng = get_sec_per_frame(device, model_stem, backend="engine")
    inf_eng = total_frames * sec_per_frame_eng
    ffmpeg_eng = total_clips * ffmpeg_sec_per_clip
    total_eng = inf_eng + ffmpeg_eng
    print(f"  [YOLO .engine] Tiempo medio por frame (estimado): {CYAN}{sec_per_frame_eng:.3f} s/frame{RESET} (YOLO-{model_size}, {device.upper()})")
    if ffmpeg_sec_per_clip > 0:
        print(f"                 Frames: {total_frames} → Inferencia: ~{inf_eng/60:.0f} min  |  FFmpeg: ~{ffmpeg_eng/60:.0f} min")
    else:
        print(f"                 Frames: {total_frames} → Inferencia: ~{inf_eng/60:.0f} min  |  Copia clips: ~{ffmpeg_eng/60:.0f} min")

    hours_pt = int(total_pt // 3600)
    mins_pt = int((total_pt % 3600) // 60)
    secs_pt = int(total_pt % 60)
    hours_eng = int(total_eng // 3600)
    mins_eng = int((total_eng % 3600) // 60)
    secs_eng = int(total_eng % 60)
    print(f"\n  Tiempo estimado total YOLO .pt{label_suffix}:     {CYAN}~{hours_pt}h {mins_pt}m {secs_pt}s{RESET}")
    print(f"  Tiempo estimado total YOLO .engine{label_suffix}:{CYAN}~{hours_eng}h {mins_eng}m {secs_eng}s{RESET}")
    if total_clips > 0:
        print(f"  Tiempo medio por clip (.pt):      {CYAN}~{(total_pt/total_clips)/60:.1f} min{RESET} (~{total_pt/total_clips:.0f} s)")
        print(f"  Tiempo medio por clip (.engine):  {CYAN}~{(total_eng/total_clips)/60:.1f} min{RESET} (~{total_eng/total_clips:.0f} s)")


def run_preflight_data_result(
    source: Path,
    *,
    output_base: Path | None,
    yolo_pose_model: str | None,
    count_only: bool = False,
    probe_videos: bool = False,
) -> int:
    header("1) Configuración (re-extracción data_result)")
    print(f"  Origen:          {source}")
    if output_base:
        print(f"  OUTPUT_BASE:     {output_base} (destino propuesto)")
    else:
        print(f"  OUTPUT_BASE:     {OUTPUT_BASE or '(config.py)'}")
    model_name = yolo_pose_model or YOLO_POSE_MODEL
    print(f"  Modelo YOLO:     {model_name}")
    if count_only:
        print(f"  Modo:            conteo rápido (solo existencia meta.json + clip.mp4)")
    elif probe_videos:
        print(f"  Modo:            meta + ffprobe por clip (lento, estimación precisa)")
    else:
        print(f"  Modo:            meta.json only (sin ffprobe)")

    try:
        inv = scan_data_result_inventory(
            source,
            count_only=count_only,
            probe_videos=probe_videos,
        )
    except FileNotFoundError as e:
        fail(str(e))
        return 1

    header("2) Inventario (existencia meta.json + clip.mp4, sin ffprobe)")
    print(f"  Raíz escaneada: {inv['base']}")
    n_incomplete = len(inv["incomplete"])
    n_correct = inv["total_clips"]
    n_total = inv.get("total_folders", n_correct + n_incomplete)
    print(f"\n  {BOLD}Carpetas de clip encontradas:{RESET} {n_total}")
    print(f"  {GREEN}Correctas{RESET} (meta.json + clip.mp4):     {n_correct}")
    if n_incomplete:
        warn(f"Incompletas (falta meta o clip):           {n_incomplete}")
        for rel in inv["incomplete"][:10]:
            print(f"    - {rel}")
        if n_incomplete > 10:
            print(f"    ... y {n_incomplete - 10} más")
    else:
        ok("Todas las carpetas tienen meta.json y clip.mp4")

    if n_correct == 0:
        fail("No hay clips correctos (meta.json + clip.mp4).")
        return 1

    by_cat = inv["by_category"]
    print(f"\n  Desglose por categoría (solo correctos):")
    for cat in sorted(by_cat):
        print(f"    cat {cat:>2}: {by_cat[cat]:>5} clips")

    if not count_only:
        src = "meta.json" if not probe_videos else "meta.json + ffprobe (fallback)"
        print(f"  Duración total ({src}): {inv['total_seconds']/60:.1f} min")
        print(f"  Frames totales (estimados):   {inv['total_frames']}")
        if inv.get("meta_missing_stats"):
            warn(
                f"{inv['meta_missing_stats']} clips correctos sin frames/duración en meta "
                f"(estimación parcial; usa --probe-videos solo si necesitas precisión)"
            )

    category_limits = _load_category_limits()
    if category_limits:
        print(f"\n  Límites config_pose_extraction.json (referencia): {category_limits}")
        limited_total = 0
        for cat, n in sorted(by_cat.items()):
            lim = category_limits.get(str(cat))
            if lim is None:
                limited_total += n
            else:
                limited_total += min(n, lim)
        print(f"  Clips que procesaría con esos límites: {limited_total}")

    if count_only:
        header("RESUMEN")
        print(f"  1) Clips: {n_total} carpetas → {n_correct} correctos, {n_incomplete} incompletos")
        print(f"  2) Tiempo estimado: omitido (--count-only no lee meta.json)")
        print()
        warn("Para estimar tiempo: quita --count-only y pasa --yolo-pose-model")
        return 0

    header("3) Validación")
    ok(f"{n_correct} clips listos para re-extracción")

    header("4) Dispositivo y modelo")
    device = get_device()
    model_stem = Path(model_name).stem
    if device == "cuda":
        try:
            import torch
            ok(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            ok("GPU: CUDA disponible")
    else:
        warn("CPU: CUDA no disponible. La extracción será lenta.")

    header("5) Estimación de tiempo total")
    _print_time_estimates(
        total_clips=inv["total_clips"],
        total_frames=inv["total_frames"],
        device=device,
        model_stem=model_stem,
        ffmpeg_sec_per_clip=COPY_SEC_PER_CLIP,
        label_suffix=" (re-extracción)",
    )
    warn("Estimación aproximada (CPU/GPU y carga del sistema pueden variar).")

    header("RESUMEN")
    print(f"  1) Clips: {n_total} carpetas → {GREEN}{n_correct} correctos{RESET}, {n_incomplete} incompletos")
    device = get_device()
    model_stem = Path(model_name).stem
    sec_pt = get_sec_per_frame(device, model_stem, backend="pt")
    total_pt = inv["total_frames"] * sec_pt + n_correct * COPY_SEC_PER_CLIP
    h, rem = divmod(int(total_pt), 3600)
    m, s = divmod(rem, 60)
    print(
        f"  2) Tiempo estimado ({model_name}, {device.upper()}, .pt): "
        f"{CYAN}~{h}h {m}m {s}s{RESET} "
        f"({inv['total_frames']} frames × {sec_pt:.3f} s/frame + copia clips)"
    )
    print()
    return 0


def get_model_size(model_stem: str) -> str:
    """Extrae tamaño del modelo: n, s, m, l, x (p. ej. yolo11n-pose -> n)."""
    s = model_stem.lower().strip()
    parts = s.split("-")
    if parts:
        first = parts[0]
        if len(first) >= 1 and first[-1] in "nsmlx":
            return first[-1]
    if "xlarge" in s or "xlarge" in model_stem.lower():
        return "x"
    if "large" in s:
        return "l"
    if "medium" in s:
        return "m"
    if "small" in s:
        return "s"
    return "n"


def get_sec_per_frame(device: str, model_stem: str, backend: str = "pt") -> float:
    """Tiempo medio (s) por frame para extracción de pose (8 keypoints), según modelo, dispositivo y backend ('pt' o 'engine')."""
    size = get_model_size(model_stem)
    if backend == "engine":
        table = SEC_PER_FRAME_ENGINE
    else:
        table = SEC_PER_FRAME_PT
    return table.get(device, table["cpu"]).get(size, table["cpu"]["n"])


def estimate_time(
    total_frames: int,
    total_clips: int,
    device: str,
    model_stem: str,
    backend: str = "pt",
):
    """
    Estima tiempo total: inferencia YOLO (solo pose) + FFmpeg para un backend dado.
    Devuelve (segundos_inferencia, segundos_ffmpeg, segundos_por_frame_usado).
    """
    sec_per_frame = get_sec_per_frame(device, model_stem, backend=backend)
    inference_sec = total_frames * sec_per_frame
    ffmpeg_sec = total_clips * FFMPEG_SEC_PER_CLIP
    return inference_sec, ffmpeg_sec, sec_per_frame


def _scan_csv_inventory(
    csv_files: List[Path],
    path_roots: List[Path],
    *,
    check_exists_only: bool = False,
    probe_videos: bool = False,
) -> dict:
    """
    Recorre CSVs y cuenta clips. Por defecto sin ffprobe (rápido).
    check_exists_only: solo comprueba existencia del fichero de vídeo.
    probe_videos: usa ffprobe para duración/FPS (lento).
    """
    import pandas as pd

    category_limits = _load_category_limits()
    category_counters: dict[str, int] = {}

    total_clips = 0
    by_category: dict[int, int] = {}
    total_seconds_of_video = 0.0
    total_frames_approx = 0
    fps_cache: dict = {}
    exists_count = 0
    missing_count = 0
    missing_entries: list[dict] = []
    full_clip_without_probe = 0

    for csv_path in csv_files:
        start_row = find_start_row(str(csv_path))
        csv_display = _csv_display_path(csv_path, path_roots)
        try:
            df = pd.read_csv(str(csv_path), skiprows=range(0, start_row - 1), header=None)
        except Exception as e:
            warn(f"CSV no legible: {csv_path.name} — {e}")
            continue

        base_dir = Path(csv_path).resolve().parent
        n_clips = 0
        cat_counts: dict[int, int] = {}

        for row_idx, row in df.iterrows():
            if len(row) < 4 or pd.isna(row.iloc[0]):
                continue
            try:
                cat = int(row.iloc[3])
            except (ValueError, TypeError):
                continue
            if cat < 0:
                continue

            cat_str = str(cat)
            limit = category_limits.get(cat_str) if category_limits else None
            if limit is not None and category_counters.get(cat_str, 0) >= limit:
                continue

            inicio_s = str(row.iloc[1]).strip().strip('"').strip("'")
            fin_s = str(row.iloc[2]).strip().strip('"').strip("'")
            if not is_hms_format(inicio_s) or not is_hms_format(fin_s):
                continue

            video_rel = str(row.iloc[0]).strip().strip('"').strip("'")
            video_full_path = _resolve_video_path(video_rel, base_dir)
            fila_csv = _csv_display_row(start_row, row_idx)

            if check_exists_only:
                n_clips += 1
                if video_full_path.is_file():
                    exists_count += 1
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
                    category_counters[cat_str] = category_counters.get(cat_str, 0) + 1
                else:
                    missing_count += 1
                    missing_entries.append({
                        "csv": str(csv_path.resolve()),
                        "csv_display": csv_display,
                        "row": fila_csv,
                        "category": cat,
                        "video_rel": video_rel,
                        "video_path": str(video_full_path),
                    })
                continue

            if not video_full_path.is_file():
                missing_count += 1
                missing_entries.append({
                    "csv": str(csv_path.resolve()),
                    "csv_display": csv_display,
                    "row": fila_csv,
                    "category": cat,
                    "video_rel": video_rel,
                    "video_path": str(video_full_path),
                })
                continue

            if _is_full_clip_range(inicio_s, fin_s):
                if probe_videos:
                    fps = get_video_fps(video_full_path, fps_cache)
                    try:
                        out = subprocess.run(
                            [
                                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(video_full_path),
                            ],
                            capture_output=True, text=True, timeout=10, check=False,
                        )
                        dur = float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else 0.0
                    except Exception:
                        dur = 0.0
                    if dur <= 0:
                        continue
                else:
                    dur = DEFAULT_MAX_CLIP_SEC
                    fps = DEFAULT_FPS
                    full_clip_without_probe += 1
            else:
                dur = hms_to_seconds(fin_s) - hms_to_seconds(inicio_s)
                fps = DEFAULT_FPS
                if probe_videos:
                    fps = get_video_fps(video_full_path, fps_cache)

            if dur <= 0:
                continue

            n_clips += 1
            total_seconds_of_video += dur
            total_frames_approx += int(min(dur, DEFAULT_MAX_CLIP_SEC) * fps + 0.5)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            category_counters[cat_str] = category_counters.get(cat_str, 0) + 1

        for k, v in cat_counts.items():
            by_category[k] = by_category.get(k, 0) + v
        total_clips += n_clips

        if check_exists_only:
            print(f"  {csv_display}: {n_clips} filas CSV")
        else:
            print(f"  {csv_display}: {n_clips} clips")

    return {
        "total_clips": total_clips,
        "by_category": by_category,
        "total_seconds_of_video": total_seconds_of_video,
        "total_frames_approx": total_frames_approx,
        "exists_count": exists_count,
        "missing_count": missing_count,
        "missing_entries": missing_entries,
        "full_clip_without_probe": full_clip_without_probe,
        "check_exists_only": check_exists_only,
        "probe_videos": probe_videos,
    }


def _print_by_category(by_category: dict[int, int], *, title: str = "Por categoría") -> None:
    if not by_category:
        return
    print(f"\n  {BOLD}{title}:{RESET}")
    for cat in sorted(by_category):
        print(f"    cat {cat:>2}: {by_category[cat]:>5} clips")


def _print_missing_videos(missing_entries: list[dict], *, limit: int = 30) -> None:
    if not missing_entries:
        return
    print(f"\n  {BOLD}Vídeos no encontrados ({len(missing_entries)}):{RESET}")
    for entry in missing_entries[:limit]:
        print(
            f"    - CSV: {entry['csv_display']} | Fila: {entry['row']} | "
            f"cat={entry['category']} | {entry['video_rel']}"
        )
        print(f"      ({entry['csv']})")
        print(f"      → {entry['video_path']}")
    if len(missing_entries) > limit:
        print(f"    ... y {len(missing_entries) - limit} más")


def run_preflight_csv(
    *,
    path: str | None,
    yolo_pose_model: str | None,
    check_exists_only: bool = False,
    probe_videos: bool = False,
) -> int:
    path_roots = get_path_roots()
    path_to_scan = path or (str(path_roots[0]) if path_roots else None)
    if not path_to_scan and CSV_PATH:
        path_to_scan = str(Path(CSV_PATH).resolve().parent)

    header("1) Configuración")
    if path_roots:
        print(f"  PATH_ROOTS ({len(path_roots)}):")
        for pr in path_roots:
            print(f"    - {pr}")
    else:
        print(f"  PATH_ROOT / CSV: {path_to_scan or PATH_ROOT or CSV_PATH}")
    print(f"  OUTPUT_BASE:     {OUTPUT_BASE or '(script dir)/output'}")
    dr_roots = get_data_result_roots()
    if dr_roots:
        print(f"  data_result ({len(dr_roots)}):")
        for dr in dr_roots:
            print(f"    - {dr}")
    print(f"  Escala clips:    {CLIP_SCALE_HEIGHT or 'original'} px altura")
    if check_exists_only:
        print(f"  Modo:            {CYAN}--check-exists{RESET} (solo existencia de vídeos, sin ffprobe)")
    elif probe_videos:
        print(f"  Modo:            ffprobe por vídeo (lento)")
    else:
        print(f"  Modo:            rápido (sin ffprobe; FPS={DEFAULT_FPS:g}, clip completo≈{DEFAULT_MAX_CLIP_SEC:g}s)")

    header("2) CSVs y clips por categoría")
    experiments = get_experiments()
    if path_to_scan and not experiments:
        root = Path(path_to_scan).resolve()
        csv_files = sorted(root.rglob("*.csv")) if root.is_dir() else []
    elif experiments:
        csv_files = [Path(e["csv"]) for e in experiments]
        root = Path(experiments[0]["path_root"]).resolve() if experiments[0].get("path_root") else None
    else:
        csv_files = [Path(CSV_PATH).resolve()] if CSV_PATH and Path(CSV_PATH).exists() else []
        root = csv_files[0].parent if csv_files else None

    if not csv_files:
        fail("No se encontraron CSVs. Revisa PATH_ROOTS o CSV_PATH en config.py.")
        return 1

    category_limits = _load_category_limits()
    if category_limits:
        print(f"Límites por categoría (config_pose_extraction.json): {category_limits}")
    else:
        print("Límites por categoría: sin límites (no hay config_pose_extraction.json o está vacío).")

    inv = _scan_csv_inventory(
        csv_files,
        path_roots if path_roots else ([root] if root is not None else []),
        check_exists_only=check_exists_only,
        probe_videos=probe_videos,
    )

    if inv["total_clips"] == 0:
        fail("No hay filas de clips válidas en los CSVs.")
        return 1

    if check_exists_only:
        print(f"\n  {BOLD}Filas CSV analizadas:{RESET} {inv['total_clips']}")
        print(f"  {GREEN}Vídeos encontrados:{RESET}  {inv['exists_count']}")
        if inv["missing_count"]:
            fail(f"Vídeos NO encontrados: {inv['missing_count']}")
            _print_missing_videos(inv["missing_entries"])
        else:
            ok("Todos los vídeos referenciados existen en disco.")
        header("RESUMEN")
        print(
            f"  Filas CSV totales: {inv['total_clips']} | "
            f"Procesables (vídeo existe): {inv['exists_count']} | "
            f"Faltan: {inv['missing_count']}"
        )
        _print_by_category(inv["by_category"], title="Clips por categoría (vídeo existente)")
        print()
        return 1 if inv["missing_count"] else 0

    print(f"\n  {BOLD}Total clips:{RESET} {inv['total_clips']}")
    _print_by_category(inv["by_category"])
    if inv["missing_count"]:
        warn(f"{inv['missing_count']} filas omitidas: vídeo no encontrado (no entran en el conteo)")
        _print_missing_videos(inv["missing_entries"])
    if probe_videos:
        print(f"  Duración total vídeo (ffprobe + CSV): {inv['total_seconds_of_video']/60:.1f} min")
    else:
        print(f"  Duración estimada (CSV, sin ffprobe): {inv['total_seconds_of_video']/60:.1f} min")
        if inv["full_clip_without_probe"]:
            warn(
                f"{inv['full_clip_without_probe']} clips completos (00:00:00–00:00:00) "
                f"contados con {DEFAULT_MAX_CLIP_SEC:g}s de estimación; usa --probe-videos para duración real"
            )
    fps_label = "real" if probe_videos else f"{DEFAULT_FPS:g}"
    print(
        f"  Frames estimados (FPS={fps_label}, "
        f"≤{DEFAULT_MAX_CLIP_SEC:g} s/clip): {inv['total_frames_approx']}"
    )

    header("3) Validación de CSV (estructura y vídeos)")
    if csv_files:
        print(f"  Validando {len(csv_files)} CSV(s)...")
        validation = validate_csv_files(
            [str(p) for p in csv_files],
            allow_missing_videos=True,
        )
        if not validation.get("ok"):
            warn("Hay errores en CSVs (no relacionados con vídeos faltantes). Corrígelos antes de ejecutar.")
        else:
            ok("Todos los CSVs validados (vídeos faltantes omitidos del conteo).")
    else:
        ok("Omisión de validación (sin CSVs).")

    header("4) Dispositivo y modelo")
    device = get_device()
    model_stem = Path(yolo_pose_model or YOLO_POSE_MODEL).stem
    if device == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            ok(f"GPU: {name}")
        except Exception:
            ok("GPU: CUDA disponible")
    else:
        warn("CPU: CUDA no disponible. La extracción será lenta.")
    print(f"  Modelo (estimación): {model_stem}")

    header("5) Estimación de tiempo total")
    _print_time_estimates(
        total_clips=inv["total_clips"],
        total_frames=inv["total_frames_approx"],
        device=device,
        model_stem=model_stem,
        ffmpeg_sec_per_clip=FFMPEG_SEC_PER_CLIP,
    )
    if not probe_videos:
        warn("Estimación de frames aproximada (sin ffprobe). Usa --probe-videos si necesitas precisión.")
    else:
        warn("Estimación aproximada (CPU/GPU y carga del sistema pueden variar).")

    header("RESUMEN")
    print(f"  Clips procesables: {inv['total_clips']}")
    if inv["missing_count"]:
        print(f"  Vídeos no encontrados (omitidos): {inv['missing_count']}")
    _print_by_category(inv["by_category"], title="Clips por categoría")
    print()
    return 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pre-chequeo para pose_extractor_clean")
    parser.add_argument("--path", type=str, default=None, help="Sobrescribir carpeta de búsqueda de CSVs (modo CSV)")
    parser.add_argument(
        "--from-data-result",
        dest="from_data_result",
        default=None,
        metavar="DIR",
        help="Inventario de clips en data_result existente (meta.json + clip.mp4), sin buscar CSVs",
    )
    parser.add_argument(
        "--output-base",
        dest="output_base",
        default=None,
        metavar="DIR",
        help="OUTPUT_BASE destino propuesto (solo informativo en preflight)",
    )
    parser.add_argument(
        "--yolo-pose-model",
        dest="yolo_pose_model",
        default=None,
        metavar="MODELO",
        help="Modelo YOLO para estimación de tiempo (p. ej. yolo26s-pose.pt)",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Solo contar clips (existencia meta.json + clip.mp4). Sin leer meta ni ffprobe — muy rápido",
    )
    parser.add_argument(
        "--probe-videos",
        action="store_true",
        help="Usar ffprobe por vídeo/clip para duración y FPS (lento; solo si necesitas estimación precisa)",
    )
    parser.add_argument(
        "--check-exists",
        action="store_true",
        help="Solo comprobar si existe cada fichero de vídeo del CSV (sin ffprobe, sin estimación de tiempo)",
    )
    parser.add_argument(
        "--save-txt",
        type=Path,
        default=None,
        metavar="FILE",
        help="Guardar el informe completo en un fichero .txt además de mostrarlo en pantalla",
    )
    args = parser.parse_args()

    original_stdout = sys.stdout
    tee: _ReportTee | None = None
    if args.save_txt:
        original_stdout, tee = _setup_report_tee(args.save_txt)

    exit_code = 0
    try:
        if args.from_data_result:
            out = Path(args.output_base).expanduser().resolve() if args.output_base else None
            exit_code = run_preflight_data_result(
                Path(args.from_data_result),
                output_base=out,
                yolo_pose_model=args.yolo_pose_model,
                count_only=args.count_only,
                probe_videos=args.probe_videos,
            )
        else:
            if args.check_exists and args.probe_videos:
                warn("--check-exists ignora --probe-videos (no se usa ffprobe).")
            exit_code = run_preflight_csv(
                path=args.path,
                yolo_pose_model=args.yolo_pose_model,
                check_exists_only=args.check_exists,
                probe_videos=args.probe_videos and not args.check_exists,
            )
    finally:
        _restore_stdout(original_stdout, tee)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
