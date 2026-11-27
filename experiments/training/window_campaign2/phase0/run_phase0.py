#!/usr/bin/env python3
"""Phase 0 — measurements only (no training). Writes artifacts/reports/phase0_report.md."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TR = ROOT.parent
WC = TR / "window_campaign"
for p in (TR, WC, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_build_loaders import build_campaign2_datasets_and_loaders  # noqa: E402
from campaign2_metrics import (  # noqa: E402
    binary_window_metrics,
    event_recall_or,
    fp_buckets_in_robbery_files,
)
from campaign2_paths import CAMPAIGN2_ROOT, load_campaign2_config, reports_dir, resolve_checkpoint_roots  # noqa: E402
from campaign2_scoring import (  # noqa: E402
    forward_prob_robbery,
    load_model_from_checkpoint,
    preprocess_clip_resampled,
    preprocess_native_window,
)
from campaign2_split_policy import load_split_manifest, split_manifest_path  # noqa: E402
from window_index import WindowExample  # type: ignore


def _clip_uid(w: WindowExample) -> str:
    return w.clip_uid


def _window_meta(w: WindowExample, core_index: Dict[str, Any]) -> Dict[str, Any]:
    core = core_index.get(w.clip_uid) or {}
    cs = int(core.get("core_start", 0))
    dist_sec = abs(w.start - cs) / 12.0 if w.start >= 0 else 0.0
    neg_source = "neg_folder"
    if int(w.label) == 1 and w.train_label != 1:
        neg_source = "in_robbery_file"
    return {
        "clip_uid": w.clip_uid,
        "window_id": w.window_id,
        "label_state": w.label_state,
        "train_label": w.train_label,
        "start": w.start,
        "end": w.end,
        "neg_source": neg_source,
        "core_distance_sec": dist_sec,
    }


@torch.no_grad()
def score_native_windows(
    model,
    device,
    windows: List[WindowExample],
    *,
    window_len: int,
    mask_policy: str,
) -> np.ndarray:
    probs = []
    for w in windows:
        poses = np.load(w.pose_path)
        vm = np.load(w.valid_mask_path) if w.valid_mask_path and w.valid_mask_path.exists() else None
        x = preprocess_native_window(
            poses, vm, w.start, w.end, window_len=window_len, mask_policy=mask_policy
        )
        probs.append(forward_prob_robbery(model, x, device))
    return np.asarray(probs, dtype=np.float64)


def p0_short_clips(cfg: Dict[str, Any], data_root: Path) -> Dict[str, Any]:
    from train_model_operations import collect_examples  # type: ignore

    ex = collect_examples(data_root=data_root, pose_source=cfg.get("pose_source", "full"))
    rob = {int(x) for x in cfg.get("robbery_classes", [6, 14])}
    short_pos = short_neg = 0
    for e in ex:
        try:
            meta = json.loads((e.pose_path.parent.parent / "meta.json").read_text())
            dur = float(meta.get("clip_duration", 999))
        except Exception:
            dur = 999.0
        if dur >= 4.0:
            continue
        from campaign2_labels import apply_campaign2_labels as _  # noqa: F401
        from core_labels import _parse_cat  # type: ignore

        cat = _parse_cat(e.category_str)
        if cat in rob:
            short_pos += 1
        else:
            short_neg += 1
    return {"short_under_4s_pos": short_pos, "short_under_4s_neg": short_neg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--full-clip-checkpoint-dir", type=str, default=None)
    ap.add_argument("--exp-ids", type=int, nargs="+", default=[48, 49, 53, 11])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.8])
    args = ap.parse_args()

    cfg = load_campaign2_config()
    p0 = cfg.get("phase0") or {}
    if args.full_clip_checkpoint_dir:
        p0 = {**p0, "full_clip_checkpoint_dir": args.full_clip_checkpoint_dir}
        cfg["phase0"] = p0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = int((cfg.get("train_defaults") or {}).get("batch_size", 32))
    data_root = Path(args.data_root).resolve()

    print("[P0] Building eval windows (campaign2 labels + symmetric gate)…", flush=True)
    loaders, _dim, _map, summary = build_campaign2_datasets_and_loaders(
        data_root=data_root,
        cfg=cfg,
        batch_size=batch,
        num_workers=0,
        augment_on_the_fly=False,
        cell_id=cfg.get("default_cell_id"),
    )
    split_key = args.split
    raw: List[WindowExample] = summary["windows_raw"][split_key]
    eval_windows = [w for w in raw if w.label_state != "ignore" and w.train_label >= 0]

    from core_labels import load_core_index  # type: ignore

    rel = (cfg.get("positive_labeling") or {}).get("core_index_path", "artifacts/core_index.json")
    core_path = Path(rel) if Path(rel).is_absolute() else CAMPAIGN2_ROOT / rel
    core_index = load_core_index(core_path)

    metas = [_window_meta(w, core_index) for w in eval_windows]
    y = np.array([int(w.train_label) for w in eval_windows], dtype=np.int64)

    roots = resolve_checkpoint_roots(cfg)
    if args.full_clip_checkpoint_dir:
        roots["full_clip"] = Path(args.full_clip_checkpoint_dir).resolve()
    v1 = roots["window_v1"]
    fc = roots.get("full_clip")

    lines: List[str] = ["# Phase 0 report (window_campaign2)\n"]

    # P0-7
    short = p0_short_clips(cfg, data_root)
    lines.append("## P0-7 Short clips (<4s in meta)\n")
    lines.append(json.dumps(short, indent=2) + "\n")

    # P0-1 with window v1 checkpoints (all gated windows, incl. ignore in robbery files)
    lines.append("## P0-1 FP decomposition (window v1 checkpoints)\n")
    raw_metas = [_window_meta(w, core_index) for w in raw]
    for eid in args.exp_ids:
        path = v1 / f"modelo_{eid:02d}.pt"
        if not path.is_file():
            lines.append(f"- modelo_{eid:02d}: missing {path}\n")
            continue
        model, _ = load_model_from_checkpoint(path, device, kind="window_native")
        probs_all = score_native_windows(
            model, device, raw, window_len=int(cfg["window_len"]), mask_policy=cfg["mask_policy"]
        )
        for thr in args.thresholds:
            buckets = fp_buckets_in_robbery_files(raw_metas, probs_all, thr)
            lines.append(f"\n### v1 exp {eid:02d} @ {thr}\n```json\n{json.dumps(buckets, indent=2)}\n```\n")
        probs_eval = score_native_windows(
            model, device, eval_windows, window_len=int(cfg["window_len"]), mask_policy=cfg["mask_policy"]
        )
        m50 = binary_window_metrics(y, probs_eval >= 0.5)
        lines.append(f"Scored-train windows @0.5: {json.dumps(m50)}\n")
        del model

    # P0-2 paired matrix C@U1 vs C@U2
    lines.append("\n## P0-2 Paired matrix (full-clip ckpt)\n")
    if fc is None or not fc.is_dir():
        lines.append("_Set `--full-clip-checkpoint-dir` to campaign bin_full models for C@U1/C@U2._\n")
    else:
        by_clip: Dict[str, List[WindowExample]] = defaultdict(list)
        for w in eval_windows:
            by_clip[w.clip_uid].append(w)

        for eid in args.exp_ids:
            path = fc / f"modelo_{eid:02d}.pt"
            if not path.is_file():
                continue
            model, _ = load_model_from_checkpoint(path, device, kind="full_clip")
            u1_probs: List[float] = []
            u1_y: List[int] = []
            u2_all: List[float] = []
            u2_y: List[int] = []
            clip_flags: List[bool] = []
            clip_uids_list: List[str] = []

            for uid, wins in sorted(by_clip.items()):
                w0 = wins[0]
                poses = np.load(w0.pose_path)
                vm = np.load(w0.valid_mask_path) if w0.valid_mask_path and w0.valid_mask_path.exists() else None
                x = preprocess_clip_resampled(poses, vm, seq_len=int(cfg["window_len"]), mask_policy=cfg["mask_policy"])
                p_clip = forward_prob_robbery(model, x, device)
                is_pos = any(int(w.train_label) == 1 for w in wins)
                u1_probs.append(p_clip)
                u1_y.append(1 if is_pos else 0)

                for w in wins:
                    xw = preprocess_native_window(
                        poses, vm, w.start, w.end, window_len=int(cfg["window_len"]), mask_policy=cfg["mask_policy"]
                    )
                    pw = forward_prob_robbery(model, xw, device)
                    u2_all.append(pw)
                    u2_y.append(int(w.train_label))
                    clip_uids_list.append(uid)
                    clip_flags.append(is_pos)

            u1p = np.asarray(u1_probs)
            u1y = np.asarray(u1_y)
            u2p = np.asarray(u2_all)
            u2y = np.asarray(u2_y)
            m_u1 = binary_window_metrics(u1y, u1p >= 0.5)
            m_u2 = binary_window_metrics(u2y, u2p >= 0.5)
            ev = event_recall_or(
                clip_uids=clip_uids_list,
                clip_is_positive=clip_flags,
                window_probs=u2p,
                window_clip_uid=clip_uids_list,
                threshold=0.5,
            )
            lines.append(
                f"**exp {eid:02d}** C@U1 clip-level @0.5: rec={m_u1['recall_pct']:.1f}% FP={m_u1['fp_rate_pct']:.2f}% "
                f"(n_clips={len(u1y)})\n"
            )
            lines.append(
                f"**exp {eid:02d}** C@U2 window-level @0.5: rec={m_u2['recall_pct']:.1f}% FP={m_u2['fp_rate_pct']:.2f}% "
                f"(n_win={len(u2y)})\n"
            )
            lines.append(f"**exp {eid:02d}** C@U2 event OR @0.5: {json.dumps(ev)}\n")
            del model

    # P0-6 gate summary from loader build
    qg = summary.get("quality_gate") or {}
    lines.append("\n## P0-6 Quality gate (symmetric, no positive protect)\n")
    lines.append("```json\n" + json.dumps(qg, indent=2) + "\n```\n")

    out_md = reports_dir() / "phase0_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[P0] Wrote {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())