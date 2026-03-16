#!/usr/bin/env python3
"""
Pre-chequeo antes de lanzar pose_extractor_clean.py.

Desde la ruta de búsqueda de CSVs (config.PATH_ROOT o CSV_PATH):
  1) Lista los CSVs encontrados y total de vídeos/clips por categoría.
  2) Comprueba que los vídeos existan y que los CSV sean válidos.
  3) Estima el tiempo total de procesamiento según GPU/CPU y modelo YOLO.

Uso:
  python pose_extractor_preflight.py
  python pose_extractor_preflight.py --path /ruta/alternativa
"""
import re
import subprocess
import sys
from pathlib import Path

# Sin cargar YOLO ni pose_extractor_clean (evitar carga pesada)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PATH_ROOT,
    CSV_PATH,
    OUTPUT_BASE,
    CLIP_SCALE_HEIGHT,
    YOLO_POSE_MODEL,
)
from security import validate_folder

# Misma lógica que pose_extractor para detectar fila de inicio del CSV
HMS_PATTERN = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")


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


def get_model_size(model_stem: str) -> str:
    """Extrae tamaño del modelo: n, s, m, l, x (p. ej. yolo11n-pose -> n)."""
    s = model_stem.lower().strip()
    # Estilo yolo11n-pose: última letra del primer bloque
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pre-chequeo para pose_extractor_clean")
    parser.add_argument("--path", type=str, default=None, help="Sobrescribir carpeta de búsqueda de CSVs")
    args = parser.parse_args()

    path_to_scan = args.path or PATH_ROOT
    if not path_to_scan and CSV_PATH:
        path_to_scan = str(Path(CSV_PATH).resolve().parent)

    header("1) Configuración")
    print(f"  PATH_ROOT / CSV: {path_to_scan or PATH_ROOT or CSV_PATH}")
    print(f"  OUTPUT_BASE:     {OUTPUT_BASE or '(script dir)/output'}")
    print(f"  Escala clips:    {CLIP_SCALE_HEIGHT or 'original'} px altura")

    header("2) CSVs y clips por categoría")
    if path_to_scan:
        root = Path(path_to_scan).resolve()
        csv_files = sorted(root.rglob("*.csv")) if root.is_dir() else []
    else:
        csv_files = [Path(CSV_PATH).resolve()] if CSV_PATH and Path(CSV_PATH).exists() else []
        root = csv_files[0].parent if csv_files else None

    if not csv_files:
        fail("No se encontraron CSVs. Revisa PATH_ROOT o CSV_PATH en config.py.")
        sys.exit(1)

    total_clips = 0
    by_category: dict[int, int] = {}
    total_seconds_of_video = 0.0
    total_frames_approx = 0
    max_clip_sec = 10.0
    fps_cache = {}

    # Límite global opcional por categoría (máx. clips a considerar por clase),
    # compartido con pose_extractor_clean (mismos números que luego se procesarán).
    category_limits = _load_category_limits()
    category_counters: dict[str, int] = {}
    if category_limits:
        print(f"Límites por categoría (config_pose_extraction.json): {category_limits}")
    else:
        print("Límites por categoría: sin límites (no hay config_pose_extraction.json o está vacío).")

    for csv_path in csv_files:
        start_row = find_start_row(str(csv_path))
        try:
            import pandas as pd
            df = pd.read_csv(str(csv_path), skiprows=range(0, start_row - 1), header=None)
        except Exception as e:
            warn(f"CSV no legible: {csv_path.name} — {e}")
            continue

        base_dir = Path(csv_path).resolve().parent
        n_clips = 0
        cat_counts: dict[int, int] = {}
        for _, row in df.iterrows():
            if len(row) < 4 or pd.isna(row.iloc[0]):
                continue
            try:
                cat = int(row.iloc[3])
            except (ValueError, TypeError):
                continue
            if cat < 0:
                continue
            cat_str = str(cat)
            # Respetar límite por categoría si existe (mismos N primeros que procesará pose_extractor_clean)
            limit = category_limits.get(cat_str) if category_limits else None
            if limit is not None and category_counters.get(cat_str, 0) >= limit:
                continue
            if not is_hms_format(str(row.iloc[1]).strip()) or not is_hms_format(str(row.iloc[2]).strip()):
                continue
            dur = hms_to_seconds(str(row.iloc[2]).strip()) - hms_to_seconds(str(row.iloc[1]).strip())
            if dur <= 0:
                continue
            video_rel = str(row.iloc[0]).strip().strip('"').strip("'")
            video_full_path = (base_dir / video_rel).resolve()
            fps = get_video_fps(video_full_path, fps_cache)
            n_clips += 1
            total_seconds_of_video += dur
            total_frames_approx += int(min(dur, max_clip_sec) * fps + 0.5)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            category_counters[cat_str] = category_counters.get(cat_str, 0) + 1

        for k, v in cat_counts.items():
            by_category[k] = by_category.get(k, 0) + v
        total_clips += n_clips
        rel = csv_path.relative_to(root) if root and root in csv_path.parents else csv_path.name
        print(f"  {rel}: {n_clips} clips")

    if total_clips == 0:
        fail("No hay filas de clips válidas en los CSVs.")
        sys.exit(1)

    print(f"\n  {BOLD}Total clips: {total_clips}{RESET}")
    print(f"  Por categoría: {dict(sorted(by_category.items()))}")
    print(f"  Duración total vídeo (suma CSV): {total_seconds_of_video/60:.1f} min")
    print(f"  Frames totales (FPS real por vídeo, ≤{max_clip_sec:.0f} s/clip): {total_frames_approx}")

    header("3) Validación de vídeos y CSV")
    path_validate = path_to_scan or (str(Path(CSV_PATH).parent) if CSV_PATH else "")
    if path_validate:
        validation = validate_folder(path_validate)
        if not validation.get("ok"):
            warn("Hay errores en CSVs o vídeos. Corrígelos antes de ejecutar pose_extractor_clean.")
    else:
        ok("Omisión de validación (sin PATH_ROOT ni CSV_PATH).")

    header("4) Dispositivo y modelo")
    device = get_device()
    model_stem = Path(YOLO_POSE_MODEL).stem
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
    model_size = get_model_size(model_stem)
    # Estimación con modelo .pt (PyTorch)
    sec_per_frame_pt = get_sec_per_frame(device, model_stem, backend="pt")
    inf_pt, ffmpeg_pt, _ = estimate_time(total_frames_approx, total_clips, device, model_stem, backend="pt")
    total_pt = inf_pt + ffmpeg_pt

    print(f"  [YOLO .pt]    Tiempo medio por frame (8 keypoints, KEEP_KPS): {CYAN}{sec_per_frame_pt:.3f} s/frame{RESET} (YOLO-{model_size}, {device.upper()})")
    print(f"               Frames: {total_frames_approx} → Inferencia: ~{inf_pt/60:.0f} min  |  FFmpeg: ~{ffmpeg_pt/60:.0f} min")

    # Estimación con .engine (TensorRT) — sólo aporta mejora real en CUDA
    sec_per_frame_eng = get_sec_per_frame(device, model_stem, backend="engine")
    inf_eng, ffmpeg_eng, _ = estimate_time(total_frames_approx, total_clips, device, model_stem, backend="engine")
    total_eng = inf_eng + ffmpeg_eng
    print(f"  [YOLO .engine] Tiempo medio por frame (estimado): {CYAN}{sec_per_frame_eng:.3f} s/frame{RESET} (YOLO-{model_size}, {device.upper()})")
    print(f"                 Frames: {total_frames_approx} → Inferencia: ~{inf_eng/60:.0f} min  |  FFmpeg: ~{ffmpeg_eng/60:.0f} min")

    # Resumen comparativo
    hours_pt = int(total_pt // 3600)
    mins_pt = int((total_pt % 3600) // 60)
    secs_pt = int(total_pt % 60)
    hours_eng = int(total_eng // 3600)
    mins_eng = int((total_eng % 3600) // 60)
    secs_eng = int(total_eng % 60)
    print(f"\n  Tiempo estimado total YOLO .pt:     {CYAN}~{hours_pt}h {mins_pt}m {secs_pt}s{RESET}")
    print(f"  Tiempo estimado total YOLO .engine:{CYAN}~{hours_eng}h {mins_eng}m {secs_eng}s{RESET}")

    if total_clips > 0:
        sec_per_clip_pt = total_pt / total_clips
        sec_per_clip_eng = total_eng / total_clips
        print(f"  Tiempo medio por clip (.pt):      {CYAN}~{sec_per_clip_pt/60:.1f} min{RESET} (~{sec_per_clip_pt:.0f} s)")
        print(f"  Tiempo medio por clip (.engine):  {CYAN}~{sec_per_clip_eng/60:.1f} min{RESET} (~{sec_per_clip_eng:.0f} s)")

    warn("Estimación aproximada (CPU/GPU y carga del sistema pueden variar).")

    print()


if __name__ == "__main__":
    main()
