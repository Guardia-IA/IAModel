"""Jobs de extracción desde filas CSV."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from triton_pose_extract.csv_io import CsvClipRow, discover_csv_files, iter_csv_clip_rows, make_clip_name
from triton_pose_extract.ffmpeg_cut import video_duration_cv2
from triton_pose_extract.pipeline_config import PipelineConfig, PipelineFilters
from triton_pose_extract.primary_user import has_complete_export


@dataclass
class ExtractClipJob:
    row: CsvClipRow
    clip_name: str
    data_dir: Path
    work_video: Path
    clip_is_temp: bool


@dataclass
class SkippedRow:
    row: CsvClipRow
    reason: str


def _apply_prefilters(
    r: CsvClipRow,
    *,
    filters: PipelineFilters,
    data_dir: Path,
) -> Optional[str]:
    if not r.video_path.is_file():
        return "video_no_encontrado"
    if filters.skip_existing_output and has_complete_export(data_dir):
        return "ya_existe_salida"
    dur = r.duration_sec
    if r.use_full_clip and dur <= 0:
        dur = video_duration_cv2(r.video_path)
    if filters.min_clip_duration_sec.enabled and dur < filters.min_clip_duration_sec.value:
        return f"muy_corto_{dur:.2f}s"
    if filters.max_clip_duration_sec.enabled and dur > filters.max_clip_duration_sec.value:
        return f"muy_largo_{dur:.2f}s"
    if dur <= 0:
        return "duracion_invalida"
    return None


def build_jobs_from_csvs(
    csv_files: List[Path],
    pipe: PipelineConfig,
    *,
    path_root: Path | None = None,
    limit: int = 0,
) -> Tuple[List[ExtractClipJob], List[SkippedRow], Dict[str, int]]:
    used_names: set = set()
    category_counters: Dict[str, int] = {}
    jobs: List[ExtractClipJob] = []
    skipped: List[SkippedRow] = []
    by_category: Dict[str, int] = {}

    def _dur_fn(p: Path) -> float:
        return video_duration_cv2(p)

    eps = float(getattr(pipe, "end_equals_full_epsilon", 0.35))

    for csv_path in csv_files:
        for r in iter_csv_clip_rows(
            csv_path,
            path_root=path_root,
            category_limits=pipe.category_limits,
            category_counters=category_counters,
            video_duration_fn=_dur_fn,
            end_equals_full_epsilon=eps,
        ):
            clip_name = make_clip_name(
                r.video_rel,
                r.t_start,
                r.t_end,
                r.category,
                used_names,
                r.dir_rel_path,
            )
            data_dir = pipe.paths.data_result_dir / r.category / clip_name
            reason = _apply_prefilters(r, filters=pipe.filters, data_dir=data_dir)
            if reason:
                skipped.append(SkippedRow(r, reason))
                continue

            if r.use_full_clip:
                work = r.video_path
                is_temp = False
            else:
                temp_cat = pipe.paths.temp_clips_dir / r.category
                temp_cat.mkdir(parents=True, exist_ok=True)
                work = temp_cat / f"{clip_name}.mp4"
                is_temp = True

            jobs.append(
                ExtractClipJob(
                    row=r,
                    clip_name=clip_name,
                    data_dir=data_dir,
                    work_video=work,
                    clip_is_temp=is_temp,
                )
            )
            by_category[r.category] = by_category.get(r.category, 0) + 1
            category_counters[r.category] = category_counters.get(r.category, 0) + 1
            if limit > 0 and len(jobs) >= limit:
                return jobs, skipped, by_category

    return jobs, skipped, by_category


def resolve_csv_list(input_roots: List[Path]) -> List[Path]:
    from triton_pose_extract.csv_io import discover_csv_files_many

    return discover_csv_files_many(input_roots)
