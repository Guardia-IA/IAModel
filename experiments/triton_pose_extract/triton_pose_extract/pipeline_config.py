"""Config pipeline CSV + filtros (activar/desactivar como pose_extractor_clean)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class FilterToggle:
    enabled: bool = True
    value: float = 0.0


@dataclass
class PipelineFilters:
    single_user_only: bool = True
    export_primary_user_on_multi_track: bool = True
    min_clip_duration_sec: FilterToggle = field(default_factory=lambda: FilterToggle(True, 0.5))
    max_clip_duration_sec: FilterToggle = field(default_factory=lambda: FilterToggle(True, 120.0))
    min_user_seconds: FilterToggle = field(default_factory=lambda: FilterToggle(False, 0.0))
    min_coverage_ratio: FilterToggle = field(default_factory=lambda: FilterToggle(True, 0.8))
    skip_existing_output: bool = True
    min_kp_conf: float = 0.5


@dataclass
class C1LikeExtractConfig:
    """Motor IAModel tipo C1 (NVDEC + preprocess paralelo); no toca el repo vendor."""

    engine: str = "clip_extractor"
    """clip_extractor | c1_like"""
    decode_backend: str = "ffmpeg_nvdec"
    """ffmpeg_nvdec | opencv"""
    decode_c1_pixel_parity: bool = True
    """Gold N2: C1 re-infer suele decodificar con OpenCV; preferir OpenCV si T=ffprobe."""
    preprocess_workers: int = 8
    fallback_opencv_decode: bool = True
    ffmpeg_hwaccel: str = "cuda"

    @classmethod
    def from_raw(cls, raw: dict | None) -> "C1LikeExtractConfig":
        ex = raw or {}
        engine = str(ex.get("engine") or "clip_extractor").strip().lower()
        backend = str(ex.get("decode_backend") or "ffmpeg_nvdec").strip().lower()
        if backend not in ("ffmpeg_nvdec", "opencv"):
            backend = "ffmpeg_nvdec"
        return cls(
            engine=engine if engine in ("clip_extractor", "c1_like") else "clip_extractor",
            decode_backend=backend,
            decode_c1_pixel_parity=bool(ex.get("decode_c1_pixel_parity", True)),
            preprocess_workers=max(1, min(int(ex.get("preprocess_workers") or 8), 32)),
            fallback_opencv_decode=bool(ex.get("fallback_opencv_decode", True)),
            ffmpeg_hwaccel=str(ex.get("ffmpeg_hwaccel") or "cuda"),
        )

    def use_c1_like(self) -> bool:
        return self.engine == "c1_like"


@dataclass
class FfmpegConfig:
    clip_scale_height: int = 0
    vaapi_device: Optional[str] = None
    delete_temp_videos: bool = True


@dataclass
class PipelinePaths:
    output_base: Path
    data_result_dir: Path
    temp_clips_dir: Path


@dataclass
class PipelineConfig:
    input_roots: List[Path]
    filters: PipelineFilters
    extract: C1LikeExtractConfig
    ffmpeg: FfmpegConfig
    paths: PipelinePaths
    category_limits: Dict[str, int]
    estimate_full_clip_sec: float = 10.0
    estimate_fps: float = 12.0
    triton_sec_per_frame: float = 0.022
    ffmpeg_sec_per_clip: float = 6.0
    end_equals_full_epsilon: float = 0.35


def _toggle(raw: Any, default_enabled: bool, default_value: float) -> FilterToggle:
    if isinstance(raw, dict):
        return FilterToggle(
            enabled=bool(raw.get("enabled", default_enabled)),
            value=float(raw.get("value", default_value)),
        )
    if raw is None:
        return FilterToggle(default_enabled, default_value)
    return FilterToggle(default_enabled, float(raw))


def load_pipeline_config(raw: Dict[str, Any], *, output_base: Path | None = None) -> PipelineConfig:
    pipe = raw.get("pipeline") or {}
    filt = raw.get("filters") or {}
    ff = raw.get("ffmpeg") or {}

    ob = output_base or Path(str(pipe.get("output_base") or raw.get("output_base") or "./output_triton"))
    ob = ob.expanduser().resolve()
    dr_name = str(pipe.get("data_result_subdir") or "data_result")
    temp_name = str(pipe.get("temp_clips_subdir") or "temp_clips")

    limits: Dict[str, int] = {}
    lim_path = raw.get("category_limits_path") or "../config_pose_extraction.json"
    lim_file = Path(lim_path)
    if not lim_file.is_file():
        lim_file = Path(__file__).resolve().parents[2] / "config_pose_extraction.json"
    if lim_file.is_file():
        try:
            data = json.loads(lim_file.read_text(encoding="utf-8"))
            for k, v in (data.get("category_limits") or {}).items():
                if v is None:
                    continue
                limits[str(int(k))] = int(v)
        except Exception:
            pass
    for k, v in (raw.get("category_limits") or {}).items():
        if v is None:
            continue
        limits[str(int(k))] = int(v)

    filters = PipelineFilters(
        single_user_only=bool(filt.get("single_user_only", True))
        if not isinstance(filt.get("single_user_only"), dict)
        else bool((filt.get("single_user_only") or {}).get("enabled", True)),
        min_clip_duration_sec=_toggle(filt.get("min_clip_duration_sec"), True, 0.5),
        max_clip_duration_sec=_toggle(filt.get("max_clip_duration_sec"), True, 120.0),
        min_user_seconds=_toggle(filt.get("min_user_seconds"), False, 0.0),
        min_coverage_ratio=_toggle(filt.get("min_coverage_ratio"), True, 0.8),
        skip_existing_output=bool(filt.get("skip_existing_output", True)),
        min_kp_conf=float(filt.get("min_kp_conf", 0.5)),
        export_primary_user_on_multi_track=bool(
            filt.get("export_primary_user_on_multi_track", True)
        ),
    )

    est = raw.get("time_estimate") or {}
    roots: List[Path] = []
    for item in pipe.get("input_roots") or []:
        roots.append(Path(str(item)).expanduser().resolve())
    if pipe.get("input_root"):
        roots.append(Path(str(pipe["input_root"])).expanduser().resolve())
    seen: set[str] = set()
    input_roots: List[Path] = []
    for p in roots:
        k = str(p)
        if k not in seen:
            seen.add(k)
            input_roots.append(p)

    return PipelineConfig(
        input_roots=input_roots,
        filters=filters,
        extract=C1LikeExtractConfig.from_raw(raw.get("extract")),
        ffmpeg=FfmpegConfig(
            clip_scale_height=int(ff["clip_scale_height"])
            if "clip_scale_height" in ff
            else 0,
            vaapi_device=ff.get("vaapi_device") or None,
            delete_temp_videos=bool(ff.get("delete_temp_videos", True)),
        ),
        paths=PipelinePaths(
            output_base=ob,
            data_result_dir=ob / dr_name,
            temp_clips_dir=ob / temp_name,
        ),
        category_limits=limits,
        estimate_full_clip_sec=float(est.get("full_clip_default_sec") or 10.0),
        estimate_fps=float(est.get("default_fps") or 12.0),
        triton_sec_per_frame=float(est.get("triton_sec_per_frame") or 0.022),
        ffmpeg_sec_per_clip=float(est.get("ffmpeg_sec_per_clip") or 6.0),
        end_equals_full_epsilon=float(pipe.get("end_equals_full_epsilon") or 0.35),
    )
