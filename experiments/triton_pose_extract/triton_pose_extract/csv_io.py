"""Lectura de CSVs de anotación (mismo contrato que pose_extractor_clean)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

HMS_PATTERN = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")


def is_hms_format(val) -> bool:
    s = str(val).strip().strip('"').strip("'")
    return bool(HMS_PATTERN.match(s))


def find_start_row(csv_path: Path) -> int:
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            parts = line.strip().split(",")
            if len(parts) >= 2 and is_hms_format(parts[1]):
                return i
    return 1


def hms_to_seconds(t: str) -> float:
    h, m, s = map(int, str(t).split(":"))
    return h * 3600 + m * 60 + s


def normalize_hms(t: str) -> str:
    return str(t).strip().strip('"').strip("'")


def is_full_clip_range(t_start: str, t_end: str) -> bool:
    return normalize_hms(t_start) == "00:00:00" and normalize_hms(t_end) == "00:00:00"


def resolve_video_path(video_rel: str, base_dir: Path) -> Path:
    p = Path(str(video_rel).strip().strip('"').strip("'"))
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def discover_csv_files(input_root: Path) -> List[Path]:
    root = input_root.expanduser().resolve()
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.csv"))


def discover_csv_files_many(input_roots: List[Path]) -> List[Path]:
    """Varias carpetas → todos los *.csv (sin duplicados)."""
    seen: set[str] = set()
    out: List[Path] = []
    for root in input_roots:
        for p in discover_csv_files(root):
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append(p)
    return sorted(out, key=lambda p: str(p))


def csv_display_row(start_row: int, row_idx: int) -> int:
    return start_row + int(row_idx)


def resolve_clip_timing(
    t_start: str,
    t_end: str,
    video_path: Path,
    *,
    video_duration_fn,
    end_equals_full_epsilon: float = 0.35,
) -> tuple[bool, str, str, float]:
    """
    Como pose_extractor_clean + regla extra:
    - 00:00:00 + 00:00:00 → vídeo entero sin FFmpeg.
    - Si inicio ~0 y fin ~duración del fichero → entero sin recorte.
    - Si no → segmento [t_start, t_end] con FFmpeg.
    """
    if is_full_clip_range(t_start, t_end):
        dur = float(video_duration_fn(video_path)) if video_duration_fn else 0.0
        return True, t_start, t_end, dur

    start_s = hms_to_seconds(t_start)
    end_s = hms_to_seconds(t_end)
    dur_seg = max(0.0, end_s - start_s)
    total = float(video_duration_fn(video_path)) if video_duration_fn else 0.0
    if total > 0 and start_s <= end_equals_full_epsilon and end_s >= total - end_equals_full_epsilon:
        return True, "00:00:00", seconds_to_hms(total), total
    if dur_seg <= 0:
        return False, t_start, t_end, 0.0
    return False, t_start, t_end, dur_seg


def seconds_to_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass(frozen=True)
class CsvClipRow:
    csv_path: Path
    csv_display: str
    row_csv: int
    video_rel: str
    video_path: Path
    t_start: str
    t_end: str
    category: str
    use_full_clip: bool
    duration_sec: float
    dir_rel_path: str
    videos_dir: Path
    ffmpeg_start: str
    ffmpeg_end: str


def _hms_to_compact(t: str) -> str:
    return str(t).replace(":", "").strip()


def make_clip_name(
    video_rel_path: str,
    t_start: str,
    t_end: str,
    category: str,
    used_names: set,
    dir_rel_path: str | None = None,
) -> str:
    p = Path(video_rel_path)
    path_parts = list(p.parent.parts) + [p.stem]
    path_norm = "_".join(path_parts) if path_parts else p.stem or "clip"
    path_norm = path_norm.replace(" ", "_").replace("/", "_")
    if dir_rel_path:
        dir_norm = str(dir_rel_path).replace(" ", "_").replace("/", "_").replace("\\", "_")
        if dir_norm and dir_norm != ".":
            path_norm = f"{dir_norm}_{path_norm}"
    t1 = _hms_to_compact(t_start)
    t2 = _hms_to_compact(t_end)
    if is_full_clip_range(t_start, t_end):
        base = f"{path_norm}_full_{category}"
    else:
        base = f"{path_norm}_{t1}_{t2}_{category}"
    name = base
    n = 2
    while name in used_names:
        name = f"{base}_{n}"
        n += 1
    used_names.add(name)
    return name


def iter_csv_clip_rows(
    csv_path: Path,
    *,
    path_root: Path | None = None,
    category_limits: Optional[Dict[str, int]] = None,
    category_counters: Optional[Dict[str, int]] = None,
    video_duration_fn=None,
    end_equals_full_epsilon: float = 0.35,
) -> Iterator[CsvClipRow]:
    """Genera filas válidas del CSV (sin aplicar filtros de duración/calidad)."""
    category_limits = category_limits or {}
    category_counters = category_counters if category_counters is not None else {}
    start_row = find_start_row(csv_path)
    base_dir = csv_path.resolve().parent
    if path_root is None:
        path_root = base_dir
    try:
        rel = base_dir.relative_to(path_root)
        dir_rel = str(rel) if rel != Path(".") else base_dir.name
    except ValueError:
        dir_rel = base_dir.name

    csv_display = csv_path.name
    if path_root:
        try:
            csv_display = csv_path.resolve().relative_to(path_root.resolve()).as_posix()
        except ValueError:
            pass

    df = pd.read_csv(str(csv_path), skiprows=range(0, start_row - 1), header=None)
    for row_idx, row in df.iterrows():
        if len(row) < 4 or pd.isna(row.iloc[0]):
            continue
        try:
            category = str(int(row.iloc[3]))
        except (ValueError, TypeError):
            continue
        limit = category_limits.get(category)
        if limit is not None and category_counters.get(category, 0) >= limit:
            continue

        t_start = normalize_hms(str(row.iloc[1]))
        t_end = normalize_hms(str(row.iloc[2]))
        if not is_hms_format(t_start) or not is_hms_format(t_end):
            continue

        video_rel = str(row.iloc[0]).strip().strip('"').strip("'")
        video_path = resolve_video_path(video_rel, base_dir)
        row_csv = csv_display_row(start_row, row_idx)
        use_full, ff_start, ff_end, duration_sec = resolve_clip_timing(
            t_start,
            t_end,
            video_path,
            video_duration_fn=video_duration_fn,
            end_equals_full_epsilon=end_equals_full_epsilon,
        )

        yield CsvClipRow(
            csv_path=csv_path,
            csv_display=csv_display,
            row_csv=row_csv,
            video_rel=video_rel,
            video_path=video_path,
            t_start=t_start,
            t_end=t_end,
            category=category,
            use_full_clip=use_full,
            duration_sec=duration_sec,
            dir_rel_path=dir_rel,
            videos_dir=base_dir,
            ffmpeg_start=ff_start,
            ffmpeg_end=ff_end,
        )
