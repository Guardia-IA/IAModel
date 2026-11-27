"""Score full-clip vs native-window units (Phase 0)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

CAMPAIGN2_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_ROOT.parent
WC = TRAINING_DIR / "window_campaign"
for p in (TRAINING_DIR, WC, CAMPAIGN2_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_config import EXPERIMENTS  # type: ignore
from train_model_operations import (  # type: ignore
    add_velocity,
    build_model,
    normalize_sequence,
    temporal_resize,
)
from window_preprocess import add_velocity_v2, normalize_visible_sequence  # type: ignore
from window_preprocess import extract_window  # type: ignore

CheckpointKind = Literal["full_clip", "window_native"]


def _exp_cfg(exp_id: int) -> Dict[str, Any]:
    if exp_id < 1 or exp_id > len(EXPERIMENTS):
        raise ValueError(f"exp_id {exp_id} out of range")
    return dict(EXPERIMENTS[exp_id - 1])


def load_model_from_checkpoint(
    path: Path,
    device: torch.device,
    *,
    kind: CheckpointKind,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if kind == "full_clip":
        cfg = dict(ckpt.get("config") or {})
        arch = str(cfg.get("arch") or ckpt.get("arch"))
        exp_id = int(ckpt.get("exp_id") or cfg.get("exp_id") or cfg.get("experiment_id") or 0)
        if exp_id < 1:
            # Infer from filename modelo_XX.pt
            stem = path.stem
            if stem.startswith("modelo_"):
                try:
                    exp_id = int(stem.split("_")[1])
                except ValueError:
                    exp_id = 0
        if exp_id >= 1:
            exp_cfg = _exp_cfg(exp_id)
            exp_cfg.update({k: v for k, v in cfg.items() if k in exp_cfg})
        else:
            exp_cfg = cfg
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
        input_dim = int(ckpt.get("input_dim") or cfg.get("input_dim", 136))
        model = build_model(arch, input_dim, 2, exp_cfg)
        model.load_state_dict(state)
        meta = {"arch": arch, "kind": kind, "exp_cfg": exp_cfg, "exp_id": exp_id}
    else:
        exp_id = int(ckpt.get("experiment_id") or ckpt.get("exp_id") or 0)
        exp_cfg = _exp_cfg(exp_id) if exp_id else {}
        arch = str(ckpt.get("arch") or exp_cfg.get("arch"))
        if exp_cfg:
            model = build_model(arch, int(ckpt["input_dim"]), 2, exp_cfg)
        else:
            model = build_model(arch, int(ckpt["input_dim"]), 2, {"seq_len": int(ckpt.get("seq_len", 96))})
        model.load_state_dict(ckpt["state_dict"])
        meta = {"arch": arch, "kind": kind, "exp_id": exp_id}
    return model.to(device).eval(), meta


def preprocess_clip_resampled(
    poses: np.ndarray,
    valid_mask: Optional[np.ndarray],
    *,
    seq_len: int = 96,
    mask_policy: str = "keep_grid_v2",
) -> torch.Tensor:
    poses = poses.astype(np.float32)
    if mask_policy == "keep_grid_v2":
        poses = normalize_visible_sequence(poses)
        poses = add_velocity_v2(poses)
    else:
        poses = normalize_sequence(poses)
        poses = add_velocity(poses)
    poses = temporal_resize(poses, seq_len)
    t, j, d = poses.shape
    x = torch.from_numpy(poses.reshape(t, j * d).astype(np.float32))
    return x.unsqueeze(0)


def preprocess_native_window(
    poses_full: np.ndarray,
    valid_mask: Optional[np.ndarray],
    start: int,
    end: int,
    *,
    window_len: int = 96,
    pad_mode: str = "repeat_last",
    mask_policy: str = "keep_grid_v2",
) -> torch.Tensor:
    poses, _, _ = extract_window(
        poses_full,
        valid_mask,
        start,
        end,
        window_len=window_len,
        pad_mode=pad_mode,
        mask_policy=mask_policy,
    )
    if mask_policy == "keep_grid_v2":
        poses = normalize_visible_sequence(poses)
        poses = add_velocity_v2(poses)
    else:
        poses = normalize_sequence(poses)
        poses = add_velocity(poses)
    if poses.shape[0] != window_len:
        poses = temporal_resize(poses, window_len)
    t, j, d = poses.shape
    x = torch.from_numpy(poses.reshape(t, j * d).astype(np.float32))
    return x.unsqueeze(0)


@torch.no_grad()
def forward_prob_robbery(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> float:
    logits = model(x.to(device))
    if logits.shape[-1] == 1:
        return float(torch.sigmoid(logits).item())
    return float(F.softmax(logits, dim=1)[0, 1].item())