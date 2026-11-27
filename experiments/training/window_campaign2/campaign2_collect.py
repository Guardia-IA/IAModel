"""Stratified clip subsampling for smoke tests (enough 6/14 clips)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from train_model_operations import (  # type: ignore
    PoseExample,
    collect_debug_examples,
    collect_examples,
)


def _robbery_clip_examples(
    n: int,
    robbery_classes: List[int],
    *,
    pose_source: str,
    single_user_only: bool,
    data_root: str | Path | None,
    **cf: Any,
) -> List[PoseExample]:
    """Robbery folders only (faster than scanning full corpus)."""
    skip = set(range(0, 30)) - {int(x) for x in robbery_classes}
    got = collect_debug_examples(
        n=max(n * 3, n + 20),
        pose_source=pose_source,
        single_user_only=single_user_only,
        data_root=data_root,
        skip_categories=skip,
        **cf,
    )
    pos: List[PoseExample] = []
    for ex in got:
        try:
            cat = int(str(ex.category_str).split("_")[0].split("/")[-1])
        except (ValueError, IndexError):
            continue
        if cat in robbery_classes:
            pos.append(ex)
        if len(pos) >= n:
            break
    return pos[:n]


def collect_smoke_examples(
    *,
    n_total: int,
    robbery_classes: List[int],
    data_root: str | Path | None,
    min_robbery_clips: Optional[int] = None,
    **clip_filter_kwargs: Any,
) -> List[PoseExample]:
    """Take n_total clips with at least min_robbery_clips from robbery folders."""
    pose_source = str(clip_filter_kwargs.pop("pose_source", "full"))
    single_user = bool(clip_filter_kwargs.pop("single_user_only", True))
    rob = {int(x) for x in robbery_classes}

    min_rob = min_robbery_clips
    if min_rob is None:
        min_rob = max(24, int(n_total * 0.35))
    n_rob = min(int(min_rob), n_total)
    n_neg = max(0, n_total - n_rob)

    pos = _robbery_clip_examples(
        n_rob,
        robbery_classes,
        pose_source=pose_source,
        single_user_only=single_user,
        data_root=data_root,
        **clip_filter_kwargs,
    )
    rob_skip = {int(x) for x in robbery_classes}
    neg = collect_debug_examples(
        n=max(n_neg, 1),
        pose_source=pose_source,
        single_user_only=single_user,
        data_root=data_root,
        skip_categories=rob_skip,
        **clip_filter_kwargs,
    )[:n_neg]

    if len(pos) < n_rob:
        print(
            f"[WIN2-SMOKE] AVISO: solo {len(pos)} clips robo (objetivo {n_rob})",
            flush=True,
        )

    out = pos + neg
    print(
        f"[WIN2-SMOKE] Submuestra {len(out)} clips (robo={n_rob}, neg={n_neg}) "
        f"(muestreo estratificado)",
        flush=True,
    )
    if n_rob < 8:
        print(
            "[WIN2-SMOKE] Muy pocos clips 6/14: el smoke no validará recall. "
            "Sube --debug-clips o revisa data_root.",
            flush=True,
        )
    return out