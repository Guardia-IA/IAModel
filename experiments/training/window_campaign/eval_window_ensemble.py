#!/usr/bin/env python3
"""
Ensemble sobre el test de ventanas (96 frames), sin reentrenar.

No usa score_pose_ensemble.py: ese script reescala el .npy entero y espera
checkpoints de campaign/. Aquí se carga el test de window_campaign y se
combinan las probabilidades de dos modelo_XX.pt.

  python3 eval_window_ensemble.py --data-root "$DATA"
  python3 eval_window_ensemble.py --data-root "$DATA" --pairs 48,49 49,53
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WINDOW_DIR = Path(__file__).resolve().parent
TRAINING_DIR = WINDOW_DIR.parent
for p in (TRAINING_DIR, WINDOW_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_config import EXPERIMENTS  # type: ignore
from train_model_operations import build_model  # type: ignore
from window_build_loaders import build_window_datasets_and_loaders
from window_paths import load_window_config, models_dir


def _metrics(y: np.ndarray, pred: np.ndarray) -> str:
    pos = y == 1
    neg = ~pos
    tp = int((pred & pos).sum())
    fp = int((pred & neg).sum())
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    rec = 100.0 * tp / max(n_pos, 1)
    fpr = 100.0 * fp / max(n_neg, 1)
    acc = 100.0 * ((pred == pos).sum()) / max(len(y), 1)
    return f"acc={acc:.1f}% rec_robo={rec:.1f}% FP={fpr:.1f}% n_pos={n_pos}"


@torch.no_grad()
def _probs(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    ps: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        logits = model(x)
        p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        ps.append(p)
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def _load(exp_id: int, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    exp_cfg = dict(EXPERIMENTS[exp_id - 1])
    model = build_model(str(ckpt.get("arch") or exp_cfg["arch"]), int(ckpt["input_dim"]), 2, exp_cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensemble mean/AND en el test de ventanas")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=["48,49", "49,53", "48,53", "49,57"],
        help="Pares exp,exp",
    )
    args = ap.parse_args()

    cfg = load_window_config()
    cell = str(cfg.get("default_cell_id", "win_bin_full_baseline"))
    mdir = models_dir(cell)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = int((cfg.get("train_defaults") or {}).get("batch_size", 32))

    print("[ENSEMBLE] Montando el test (misma lectura que el entrenamiento) …", flush=True)
    loaders, _dim, _map, summary = build_window_datasets_and_loaders(
        data_root=Path(args.data_root).resolve(),
        cfg_window=cfg,
        batch_size=batch,
        num_workers=0,
        augment_on_the_fly=False,
        cell_id=cell,
    )
    print(f"[ENSEMBLE] test label_states={summary['label_states']['test']}", flush=True)

    needed = sorted({int(x) for pair in args.pairs for x in pair.split(",")})
    probs: dict[int, np.ndarray] = {}
    y_ref: np.ndarray | None = None
    for eid in needed:
        path = mdir / f"modelo_{eid:02d}.pt"
        if not path.is_file():
            raise SystemExit(f"Falta {path}")
        print(f"[ENSEMBLE] forward exp {eid:02d}", flush=True)
        model = _load(eid, path, device)
        p, y = _probs(model, loaders["test"], device)
        probs[eid] = p
        if y_ref is None:
            y_ref = y
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert y_ref is not None

    print("\n[ENSEMBLE] Umbral suelto = cada modelo >= 0.50. Umbral estricto = >= 0.80.\n", flush=True)
    for eid in needed:
        p = probs[eid]
        print(f"  solo {eid:02d} @0.50  {_metrics(y_ref, p >= 0.50)}")
        print(f"  solo {eid:02d} @0.80  {_metrics(y_ref, p >= 0.80)}")

    for pair in args.pairs:
        a_s, b_s = pair.split(",")
        a, b = int(a_s), int(b_s)
        pa, pb = probs[a], probs[b]
        mean = (pa + pb) / 2.0
        print(f"\n  par {a:02d}+{b:02d}")
        print(f"    mean @0.50  {_metrics(y_ref, mean >= 0.50)}")
        print(f"    mean @0.80  {_metrics(y_ref, mean >= 0.80)}")
        print(f"    AND  @0.50  {_metrics(y_ref, (pa >= 0.50) & (pb >= 0.50))}")
        print(f"    AND  @0.80  {_metrics(y_ref, (pa >= 0.80) & (pb >= 0.80))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())