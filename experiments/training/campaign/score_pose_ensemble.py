#!/usr/bin/env python3
"""
Score P(robo) con checkpoints de campaña (prod: exp 41 + 48, AND @ 0.80).

Misma cadena que eval/train: poses_full → (opcional valid_mask) → normalize → velocity → temporal_resize.

Ejemplos:

  # Chunk export C1
  python score_pose_ensemble.py \\
    --poses-full /path/chunk_001/poses_full.npy \\
    --valid-mask /path/chunk_001/valid_mask.npy \\
    --model-a artifacts/models/bin_full/modelo_41.pt \\
    --model-b artifacts/models/bin_full/modelo_48.pt

  # Salida extract IAModel (user_*)
  python score_pose_ensemble.py \\
    --poses-full out_smoke/data_result/0/clip_xxx/user_0/poses_full.npy \\
    --valid-mask out_smoke/data_result/0/clip_xxx/user_0/valid_mask.npy \\
    --model-a ~/IAModel/experiments/training/campaign/artifacts/models/bin_full/modelo_41.pt \\
    --model-b ~/IAModel/experiments/training/campaign/artifacts/models/bin_full/modelo_48.pt

  # Solo poses.npy (filas ya válidas, sin máscara)
  python score_pose_ensemble.py --poses-full /path/poses.npy --model-a ... --model-b ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from train_model_operations import (  # noqa: E402
    add_velocity,
    build_model,
    normalize_sequence,
    temporal_resize,
)


def load_poses(poses_full: Path, valid_mask: Path | None, use_mask: bool) -> np.ndarray:
    poses = np.load(poses_full)
    if poses.ndim != 3:
        raise SystemExit(f"Se esperaba [T,J,2], got {poses.shape}")
    if use_mask and valid_mask is not None and valid_mask.is_file():
        vm = np.load(valid_mask)
        if vm.shape[0] != poses.shape[0]:
            raise SystemExit(f"valid_mask len {vm.shape[0]} != poses T {poses.shape[0]}")
        poses = poses[vm.astype(bool)].copy()
    if np.any(np.isnan(poses)):
        poses = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0)
    return poses


@torch.no_grad()
def prob_robo(model_path: Path, poses: np.ndarray, device: torch.device) -> dict:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    arch = cfg.get("arch", "tcn")
    seq_len = int(ckpt.get("seq_len", 64))
    input_dim = int(ckpt["input_dim"])
    num_classes = int(ckpt.get("num_classes", 2))

    model = build_model(arch, input_dim, num_classes, cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    x = normalize_sequence(poses.copy())
    x = add_velocity(x)
    x = temporal_resize(x, seq_len)
    t, j, d = x.shape
    tensor = torch.from_numpy(x.reshape(1, t, j * d).astype(np.float32)).to(device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)
    p = float(probs[0, 1].item())
    return {
        "model": str(model_path),
        "arch": arch,
        "seq_len": seq_len,
        "input_T_after_mask": int(poses.shape[0]),
        "p_robo": p,
        "logit0": float(logits[0, 0].item()),
        "logit1": float(logits[0, 1].item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Score ensemble binario 41|48 (AND @ umbral)")
    ap.add_argument("--poses-full", type=Path, required=True, help="poses_full.npy o poses.npy [T,8,2]")
    ap.add_argument("--valid-mask", type=Path, default=None, help="valid_mask.npy (misma T que poses_full)")
    ap.add_argument(
        "--no-valid-mask-filter",
        action="store_true",
        help="No filtrar filas con valid_mask (usa grilla completa; comparar con C2)",
    )
    ap.add_argument("--model-a", type=Path, required=True, help="p.ej. modelo_41.pt")
    ap.add_argument("--model-b", type=Path, required=True, help="p.ej. modelo_48.pt")
    ap.add_argument("--threshold", type=float, default=0.80, help="AND: ambos >= threshold")
    ap.add_argument("--rule", choices=("and", "mean"), default="and")
    ap.add_argument("--json", action="store_true", help="Salida JSON")
    args = ap.parse_args()

    if not args.poses_full.is_file():
        raise SystemExit(f"No existe {args.poses_full}")
    for mp in (args.model_a, args.model_b):
        if not mp.is_file():
            raise SystemExit(f"No existe checkpoint: {mp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_mask = not args.no_valid_mask_filter
    poses = load_poses(args.poses_full, args.valid_mask, use_mask)

    ra = prob_robo(args.model_a.resolve(), poses, device)
    rb = prob_robo(args.model_b.resolve(), poses, device)
    pa, pb = ra["p_robo"], rb["p_robo"]

    if args.rule == "and":
        score = min(pa, pb)
        alarm = pa >= args.threshold and pb >= args.threshold
    else:
        score = (pa + pb) / 2.0
        alarm = score >= args.threshold

    out = {
        "poses_full": str(args.poses_full.resolve()),
        "valid_mask": str(args.valid_mask.resolve()) if args.valid_mask else None,
        "used_valid_mask_filter": use_mask,
        "T_raw": int(np.load(args.poses_full).shape[0]),
        "T_after_mask": int(poses.shape[0]),
        "p41": pa,
        "p48": pb,
        "rule": args.rule,
        "threshold": args.threshold,
        "robbery_score": score,
        "robbery_alarm": alarm,
        "model_a_detail": ra,
        "model_b_detail": rb,
    }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"poses: {args.poses_full}")
        print(f"T raw={out['T_raw']} after_mask={out['T_after_mask']} (filter={use_mask})")
        print(f"p41={pa:.6f}  p48={pb:.6f}  {args.rule} score={score:.6f}  alarm={alarm}")
        print(f"device={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
