"""Loaders for campaign2 — symmetric gate + v2 labels."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CAMPAIGN2_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_ROOT.parent
WC = TRAINING_DIR / "window_campaign"
for p in (TRAINING_DIR, WC, CAMPAIGN2_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from torch.utils.data import DataLoader  # noqa: E402

from train_model_operations import (  # type: ignore
    AUGMENT_CONFIG_PATH,
    AUGMENT_MAX_OPS,
    AUGMENT_PROB,
    AUGMENT_PROFILE_DEFAULT,
    MANIFEST_VARIANT_SET_DEFAULT,
    MAX_DETERMINISTIC_VARIANTS,
    SEED,
    TRAIN_DETERMINISTIC_PROB,
    PoseExample,
    _load_augment_profile,
    build_deterministic_variant_specs,
    build_label_mapping,
    build_per_uid_variant_map,
    collect_debug_examples,
    collect_examples,
    make_binary_examples,
    split_uids_from_example_lists,
)

from campaign2_labels import apply_campaign2_labels, ensure_core_index_file  # noqa: E402
from campaign2_paths import CAMPAIGN2_ROOT as ROOT  # noqa: E402
from core_labels import apply_padded_negative_simulation  # type: ignore
from window_build_loaders import (  # type: ignore
    _aug_ranges_from_profile,
    _clip_filter_kwargs,
    _collate_xy,
    _collate_xyw,
    _count_label_states,
)
from window_dataset import WindowPoseDataset  # type: ignore
from window_index import WindowExample, build_windows_from_examples, summarize_windows  # type: ignore
from window_quality import filter_windows_by_pose_quality  # type: ignore
from window_sampler import HierarchicalClipCapSampler  # type: ignore
from campaign2_split_policy import split_clips_from_config, write_split_manifest  # noqa: E402
from window_split_policy import assert_eval_windows_not_augmented  # type: ignore
from window_synthetic import expand_windows_with_manifest_extra_views  # type: ignore


def build_campaign2_datasets_and_loaders(
    *,
    data_root: str | Path | None,
    cfg: Dict[str, Any],
    batch_size: int = 32,
    num_workers: int = 2,
    augment_on_the_fly: bool = True,
    augment_config_path: Path = AUGMENT_CONFIG_PATH,
    augment_profile: str = AUGMENT_PROFILE_DEFAULT,
    augment_prob: float = AUGMENT_PROB,
    augment_max_ops: int = AUGMENT_MAX_OPS,
    augment_seed: int = SEED,
    max_deterministic_variants: int = MAX_DETERMINISTIC_VARIANTS,
    train_deterministic_prob: float = TRAIN_DETERMINISTIC_PROB,
    use_deterministic_in_train: bool = True,
    manifest_cache_dir: Optional[Path] = None,
    manifest_variant_set: str = MANIFEST_VARIANT_SET_DEFAULT,
    extra_manifest_views_per_window: int = 1,
    debug_max_clips: int = 0,
    force_new_split: bool = False,
    cell_id: str | None = None,
) -> Tuple[Dict[str, DataLoader], int, Dict[int, int], Dict[str, Any]]:
    window_len = int(cfg.get("window_len", 96))
    stride = int(cfg.get("stride", 48))
    min_tail = int(cfg.get("min_tail_for_final_window", 24))
    pad_mode = str(cfg.get("pad_mode", "repeat_last"))
    mask_policy = str(cfg.get("mask_policy", "keep_grid_v2"))
    pose_source = str(cfg.get("pose_source", "full"))
    single_user = bool(cfg.get("single_user_only", True))
    task = str(cfg.get("task_default", "binary"))
    positive_classes = cfg.get("robbery_classes", [6, 14])
    pl_cfg = cfg.get("positive_labeling") or {}
    pad_cfg = cfg.get("padding") or {}
    sampler_cfg = cfg.get("sampler") or {}
    cf = _clip_filter_kwargs(cfg)

    if debug_max_clips > 0:
        examples = collect_debug_examples(
            n=int(debug_max_clips),
            pose_source=pose_source,
            single_user_only=single_user,
            data_root=data_root,
            **cf,
        )
        print(f"[WIN2-DEBUG] {len(examples)} clips (debug_max_clips={debug_max_clips})")
    else:
        examples = collect_examples(
            pose_source=pose_source,
            single_user_only=single_user,
            data_root=data_root,
            **cf,
        )
    if task == "binary":
        examples = make_binary_examples(examples, positive_classes=positive_classes)

    train_clips, val_clips, test_clips, manifest_path, used_frozen = split_clips_from_config(
        examples,
        cfg,
        cell_id=cell_id or cfg.get("default_cell_id"),
        force_new_split=force_new_split,
    )
    split_uids = split_uids_from_example_lists(train_clips, val_clips, test_clips)
    if not used_frozen:
        write_split_manifest(
            manifest_path,
            train_clips=train_clips,
            val_clips=val_clips,
            test_clips=test_clips,
            cfg=cfg,
        )

    all_clips = train_clips + val_clips + test_clips
    core_index = ensure_core_index_file(cfg, all_clips)

    def _wins(clips: List[PoseExample]) -> List[WindowExample]:
        return build_windows_from_examples(
            clips,
            window_len=window_len,
            stride=stride,
            pose_source=pose_source,
            min_tail_for_final_window=min_tail,
        )

    train_w = _wins(train_clips)
    val_w = _wins(val_clips)
    test_w = _wins(test_clips)

    if pl_cfg.get("mode", "core_aware_v2").startswith("core"):
        train_w = apply_campaign2_labels(train_w, core_index, robbery_cats=positive_classes, cfg=cfg)
        val_w = apply_campaign2_labels(val_w, core_index, robbery_cats=positive_classes, cfg=cfg)
        test_w = apply_campaign2_labels(test_w, core_index, robbery_cats=positive_classes, cfg=cfg)

    q_summary: Dict[str, Any] = {}
    if (cfg.get("quality_gate") or {}).get("enabled", True):
        train_w, q_tr = filter_windows_by_pose_quality(train_w, cfg=cfg)
        val_w, q_va = filter_windows_by_pose_quality(val_w, cfg=cfg, verbose=False)
        test_w, q_te = filter_windows_by_pose_quality(test_w, cfg=cfg, verbose=False)
        q_summary = {"train": q_tr, "val": q_va, "test": q_te}

    train_w = apply_padded_negative_simulation(
        train_w,
        prob=float(pad_cfg.get("padded_negative_prob", 0.1)),
        len_range=tuple(pad_cfg.get("padded_len_range", [48, 95])),
        seed=SEED,
        window_len=window_len,
    )

    if manifest_cache_dir is not None and extra_manifest_views_per_window > 0:
        train_w = expand_windows_with_manifest_extra_views(
            train_w,
            Path(manifest_cache_dir),
            manifest_variant_set,
            extra_manifest_views_per_window,
        )

    assert_eval_windows_not_augmented(val_w, test_w)

    core_path = ROOT / (pl_cfg.get("core_index_path") or "artifacts/core_index.json")
    summary = {
        "clips": {"train": len(train_clips), "val": len(val_clips), "test": len(test_clips)},
        "windows": {
            "train": summarize_windows(train_w),
            "val": summarize_windows(val_w),
            "test": summarize_windows(test_w),
        },
        "label_states": {
            "train": _count_label_states(train_w),
            "val": _count_label_states(val_w),
            "test": _count_label_states(test_w),
        },
        "split_uids": split_uids,
        "split_manifest_path": str(manifest_path),
        "split_used_frozen_manifest": used_frozen,
        "core_index_path": str(core_path),
        "mask_policy": mask_policy,
        "quality_gate": q_summary,
        "windows_raw": {"train": train_w, "val": val_w, "test": test_w},
    }
    print(
        f"[WIN2] clips T/V/Te={len(train_clips)}/{len(val_clips)}/{len(test_clips)} | "
        f"windows T/V/Te={len(train_w)}/{len(val_w)}/{len(test_w)} | "
        f"label_states test={summary['label_states']['test']}",
        flush=True,
    )

    label_to_idx = build_label_mapping(examples)
    aug_cfg = _load_augment_profile(Path(augment_config_path), augment_profile) or {}
    aug_ranges = _aug_ranges_from_profile(aug_cfg)
    det_specs = build_deterministic_variant_specs(
        aug_cfg if isinstance(aug_cfg, dict) else {},
        max_variants=max(1, int(max_deterministic_variants)),
    )
    per_clip_map: Optional[Dict[str, List[Dict[str, Any]]]] = None
    if manifest_cache_dir is not None:
        per_clip_map = build_per_uid_variant_map(
            all_clips,
            Path(manifest_cache_dir),
            variant_set=manifest_variant_set,
            verify_source_path=True,
        )

    ds_kw = dict(
        label_to_idx=label_to_idx,
        window_len=window_len,
        pad_mode=pad_mode,
        mask_policy=mask_policy,
        augment_ranges=aug_ranges,
        augment_seed=augment_seed,
        deterministic_variants=det_specs,
        per_clip_variants=per_clip_map,
    )

    train_ds = WindowPoseDataset(
        train_w,
        augment_on_the_fly=augment_on_the_fly,
        augment_prob=augment_prob,
        augment_max_ops=augment_max_ops,
        dataset_split="train",
        train_deterministic_prob=train_deterministic_prob,
        use_deterministic_in_train=use_deterministic_in_train,
        **ds_kw,
    )

    def _eval_only(ws: List[WindowExample]) -> List[WindowExample]:
        return [w for w in ws if w.label_state != "ignore" and w.train_label >= 0]

    val_eval = _eval_only(val_w)
    test_eval = _eval_only(test_w)

    val_ds = WindowPoseDataset(
        val_eval,
        augment_on_the_fly=False,
        use_deterministic_in_train=False,
        train_deterministic_prob=0.0,
        dataset_split="val",
        **ds_kw,
    )
    test_ds = WindowPoseDataset(
        test_eval,
        augment_on_the_fly=False,
        use_deterministic_in_train=False,
        train_deterministic_prob=0.0,
        dataset_split="test",
        **ds_kw,
    )

    use_hier = str(sampler_cfg.get("type", "hierarchical")) == "hierarchical"
    if use_hier and task == "binary":
        default_spb = int(sampler_cfg.get("samples_per_epoch_batches", 100))
        n_train = int(sampler_cfg.get("samples_per_epoch", batch_size * default_spb))
        n_train = max(batch_size, min(n_train, len(train_ds)))
        sampler = HierarchicalClipCapSampler(
            train_w,
            num_samples=n_train,
            positive_share=float(sampler_cfg.get("positive_share", 0.22)),
            max_windows_per_clip=int(sampler_cfg.get("max_windows_per_clip_per_epoch", 4)),
            seed=SEED,
        )
        print(f"[WIN2-SAMPLER] {sampler.stats_line()}", flush=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=_collate_xyw,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate_xyw,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_xy
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_xy
    )

    sample_x, _, _ = train_ds[0]
    input_dim = int(sample_x.shape[-1])
    return {"train": train_loader, "val": val_loader, "test": test_loader}, input_dim, label_to_idx, summary