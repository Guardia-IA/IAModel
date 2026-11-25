"""Política de split y comprobaciones val/test sin augment (window_campaign)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from train_model_operations import (  # type: ignore
    PoseExample,
    _example_uid,
    assert_no_uid_leak_between_splits,
    split_examples_by_uid_manifest,
    split_uids_from_example_lists,
    verify_split_uid_disjoint,
)

from window_index import WindowExample


def split_manifest_path(cfg: Dict[str, Any], cell_id: str | None = None) -> Path:
    split = cfg.get("split") or {}
    rel = split.get(
        "manifest_path",
        f"artifacts/plans/{cell_id or cfg.get('default_cell_id', 'win_bin_full_baseline')}_split_manifest.json",
    )
    from window_paths import WINDOW_ROOT

    p = Path(rel)
    return p if p.is_absolute() else WINDOW_ROOT / p


def load_split_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_clips_from_config(
    examples: List[PoseExample],
    cfg: Dict[str, Any],
    *,
    cell_id: str | None = None,
    force_new_split: bool = False,
) -> Tuple[List[PoseExample], List[PoseExample], List[PoseExample], Path, bool]:
    """
    Reparte clips. Si existe manifiesto y freeze está activo → mismos UIDs siempre (val aislada).
    Devuelve (train, val, test, manifest_path, used_frozen_manifest).
    """
    from train_model_operations import split_examples_stratified_by_uid, SEED

    split = cfg.get("split") or {}
    path = split_manifest_path(cfg, cell_id=cell_id)
    use_frozen = bool(split.get("use_frozen_manifest", True)) and not force_new_split

    if use_frozen and path.is_file():
        data = load_split_manifest(path)
        su = data.get("split_uids")
        if not su:
            raise ValueError(f"Manifiesto sin split_uids: {path}")
        verify_split_uid_disjoint(su)
        train, val, test = split_examples_by_uid_manifest(examples, su)
        certify_disjoint_splits(train, val, test, label="(manifiesto congelado)")
        return train, val, test, path, True

    train, val, test = split_examples_stratified_by_uid(
        examples,
        seed=SEED,
        train_ratio=float(split.get("train_ratio", 0.8)),
        val_ratio=float(split.get("val_ratio", 0.1)),
    )
    certify_disjoint_splits(train, val, test, label="(split nuevo)")
    return train, val, test, path, False


def certify_disjoint_splits(
    train_clips: List[PoseExample],
    val_clips: List[PoseExample],
    test_clips: List[PoseExample],
    *,
    label: str = "",
) -> Dict[str, Any]:
    """
    Garantía fuerte: ningún clip (UID ni ruta .npy) aparece en más de un split.
    train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅.
    """
    assert_no_uid_leak_between_splits(train_clips, val_clips, test_clips)

    def paths(exs: List[PoseExample]) -> Set[str]:
        return {str(ex.pose_path.resolve()) for ex in exs}

    def uids(exs: List[PoseExample]) -> Set[str]:
        return {_example_uid(ex) for ex in exs}

    tr_u, va_u, te_u = uids(train_clips), uids(val_clips), uids(test_clips)
    tr_p, va_p, te_p = paths(train_clips), paths(val_clips), paths(test_clips)

    leaks = {
        "uid_train_val": tr_u & va_u,
        "uid_train_test": tr_u & te_u,
        "uid_val_test": va_u & te_u,
        "path_train_val": tr_p & va_p,
        "path_train_test": tr_p & te_p,
        "path_val_test": va_p & te_p,
    }
    for key, inter in leaks.items():
        if inter:
            raise RuntimeError(
                f"SPLIT LEAK {label}: {key} tiene {len(inter)} elementos en común, p.ej. {next(iter(inter))!r}"
            )

    report = {
        "ok": True,
        "clips_train": len(train_clips),
        "clips_val": len(val_clips),
        "clips_test": len(test_clips),
        "uids_train": len(tr_u),
        "uids_val": len(va_u),
        "uids_test": len(te_u),
        "train_intersect_val": 0,
        "train_intersect_test": 0,
        "val_intersect_test": 0,
    }
    print(
        f"[SPLIT-OK]{label} clips T/V/Te={report['clips_train']}/{report['clips_val']}/{report['clips_test']} | "
        f"UIDs disjuntos: train∩val=0 train∩test=0 val∩test=0"
    )
    return report


def assert_eval_windows_not_augmented(
    val_w: List[WindowExample],
    test_w: List[WindowExample],
) -> None:
    for split_name, rows in (("val", val_w), ("test", test_w)):
        for w in rows:
            if w.forced_ops is not None:
                raise RuntimeError(
                    f"Fuga augment: split {split_name} tiene forced_ops en {w.window_id}"
                )
            if getattr(w, "simulate_padded_negative", False):
                raise RuntimeError(
                    f"Fuga augment: split {split_name} tiene simulate_padded_negative en {w.window_id}"
                )


def write_split_manifest(
    path: Path,
    *,
    train_clips: list,
    val_clips: list,
    test_clips: list,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    certify_disjoint_splits(train_clips, val_clips, test_clips, label="(escribiendo manifiesto)")
    split_uids = split_uids_from_example_lists(train_clips, val_clips, test_clips)
    payload = {
        "version": 1,
        "policy": (
            "Split a nivel clip UID. train ∩ val ∩ test = ∅ (comprobado). "
            "Ventanas del mismo clip siempre en el mismo split. "
            "Augment solo en train."
        ),
        "train_ratio": (cfg.get("split") or {}).get("train_ratio", 0.8),
        "val_ratio": (cfg.get("split") or {}).get("val_ratio", 0.1),
        "split_uids": split_uids,
        "counts": {
            "clips_train": len(train_clips),
            "clips_val": len(val_clips),
            "clips_test": len(test_clips),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload