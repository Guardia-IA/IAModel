import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .train_model_operations import (
        PoseExample,
        _example_uid,
        add_velocity,
        build_model,
        collect_examples,
        make_binary_examples,
        normalize_sequence,
        temporal_resize,
    )
except ImportError:
    from train_model_operations import (  # type: ignore
        PoseExample,
        _example_uid,
        add_velocity,
        build_model,
        collect_examples,
        make_binary_examples,
        normalize_sequence,
        temporal_resize,
    )


def _load_split_uids(split_manifest_path: Optional[Path], split_name: str) -> Optional[set]:
    if split_manifest_path is None:
        return None
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    split = data.get("split", {})
    uids = split.get(split_name)
    if not isinstance(uids, list):
        raise RuntimeError(f"No existe split '{split_name}' en {split_manifest_path}")
    return set(str(x) for x in uids)


def _load_examples_filtered(
    pose_source: str,
    split_manifest_path: Optional[Path],
    split_name: str,
    positive_class: int,
) -> List[PoseExample]:
    examples = collect_examples(pose_source=pose_source, single_user_only=False)
    examples = make_binary_examples(examples, positive_class=positive_class)
    split_uids = _load_split_uids(split_manifest_path, split_name)
    if split_uids is None:
        return examples
    out = [ex for ex in examples if _example_uid(ex) in split_uids]
    if not out:
        raise RuntimeError("No quedaron ejemplos tras filtrar por split manifest.")
    return out


def _forward_prob_pos(
    model: torch.nn.Module,
    ex: PoseExample,
    seq_len: int,
    device: torch.device,
    temperature: float = 1.0,
) -> float:
    poses = np.load(ex.pose_path)
    if ex.valid_mask_path is not None and ex.valid_mask_path.exists():
        valid_mask = np.load(ex.valid_mask_path)
        poses = poses[valid_mask].copy()
    poses = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0)
    x = normalize_sequence(poses)
    x = add_velocity(x)
    x = temporal_resize(x, seq_len)
    x = x.reshape(x.shape[0], -1).astype(np.float32)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(xt) / float(max(1e-6, temperature))
        probs = torch.softmax(logits, dim=1)
    return float(probs[0, 1].item())


def _fit_temperature_grid(logits: torch.Tensor, y: torch.Tensor) -> float:
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.5, 3.0, 51):
        nll = F.cross_entropy(logits / float(t), y, reduction="mean").item()
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def main() -> None:
    ap = argparse.ArgumentParser(description="Herramientas post-training: ensemble, calibración, OOD, conformal.")
    ap.add_argument("--model-path", action="append", required=True, help="Ruta a checkpoint .pt (repetible).")
    ap.add_argument("--split-manifest", type=str, default=None, help="split_manifest para subset fijo.")
    ap.add_argument("--split-name", type=str, default="val", help="Subset: train|val|test")
    ap.add_argument(
        "--positive-class",
        type=int,
        default=None,
        help="Clase original positiva (p.ej. 6). Si no se indica, se usa la del primer checkpoint.",
    )
    ap.add_argument("--pose-source", type=str, default="filtered", choices=["filtered", "full"])
    ap.add_argument("--conformal-alpha", type=float, default=0.1, help="Riesgo permitido (FPR aprox).")
    ap.add_argument("--ood-entropy-quantile", type=float, default=0.95, help="Cuantil de entropía para gate OOD.")
    ap.add_argument("--output-json", type=str, default="post_training_binary_tools.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = [Path(p) for p in args.model_path]
    cps: List[Dict[str, Any]] = [torch.load(p, map_location=device) for p in model_paths]
    for cp in cps:
        if cp.get("num_classes", 2) != 2:
            raise RuntimeError("Esta herramienta es para binario (2 clases).")

    pos_cls = int(args.positive_class) if args.positive_class is not None else int(cps[0].get("positive_class", 6))

    examples = _load_examples_filtered(
        pose_source=args.pose_source,
        split_manifest_path=(Path(args.split_manifest) if args.split_manifest else None),
        split_name=args.split_name,
        positive_class=pos_cls,
    )

    # Cargar modelos una sola vez
    loaded_models: List[Tuple[torch.nn.Module, int, Dict[str, Any]]] = []
    for cp in cps:
        cfg = cp.get("config", {})
        model = build_model(cfg["arch"], cp["input_dim"], 2, cfg).to(device)
        model.load_state_dict(cp["model_state_dict"])
        model.eval()
        loaded_models.append((model, int(cp.get("seq_len", 64)), cp))

    # Temperatura por modelo (grid en logits del subset)
    temperatures: List[float] = []
    for model, seq_len, _cp in loaded_models:
        logits_all: List[torch.Tensor] = []
        y_all: List[int] = []
        for ex in examples:
            p = _forward_prob_pos(model, ex, seq_len, device, temperature=1.0)
            p = min(max(p, 1e-6), 1.0 - 1e-6)
            logit = np.log(p / (1.0 - p))
            logits_all.append(torch.tensor([0.0, logit], dtype=torch.float32))
            y_all.append(1 if ex.label == 6 else 0)
        logits_t = torch.stack(logits_all).to(device)
        y_t = torch.tensor(y_all, dtype=torch.long, device=device)
        temperatures.append(_fit_temperature_grid(logits_t, y_t))

    # Ensemble calibrado
    probs: List[float] = []
    y_true: List[int] = []
    entropies: List[float] = []
    for ex in examples:
        per_model = []
        for (model, seq_len, _cp), t in zip(loaded_models, temperatures):
            per_model.append(_forward_prob_pos(model, ex, seq_len, device, temperature=t))
        p = float(np.mean(per_model))
        probs.append(p)
        y_true.append(1 if ex.label == 1 else 0)
        p2 = max(1e-6, min(1.0 - 1e-6, p))
        entropies.append(float(-(p2 * np.log(p2) + (1.0 - p2) * np.log(1.0 - p2))))

    # OOD gate por entropía
    q = float(max(0.0, min(1.0, args.ood_entropy_quantile)))
    ood_entropy_thr = float(np.quantile(np.array(entropies, dtype=np.float64), q))

    # Conformal: umbral sobre p(robo) en negativos para controlar falsas alarmas
    alpha = float(max(1e-6, min(0.99, args.conformal_alpha)))
    neg_probs = [p for p, y in zip(probs, y_true) if y == 0]
    if neg_probs:
        conf_thr = float(np.quantile(np.array(neg_probs, dtype=np.float64), 1.0 - alpha))
    else:
        conf_thr = 0.5

    # Hard negatives del subset (no-robo con prob alta de robo)
    hard_negs = []
    for ex, p, y in zip(examples, probs, y_true):
        if y == 0:
            hard_negs.append({"uid": _example_uid(ex), "path": str(ex.pose_path), "prob_robo": float(p)})
    hard_negs = sorted(hard_negs, key=lambda r: r["prob_robo"], reverse=True)[:200]

    payload = {
        "positive_class_original": pos_cls,
        "models": [str(p) for p in model_paths],
        "temperatures": [float(t) for t in temperatures],
        "ensemble": {"type": "mean_prob"},
        "ood": {"entropy_threshold": ood_entropy_thr, "entropy_quantile": q},
        "conformal": {"alpha": alpha, "prob_threshold_for_alarm": conf_thr},
        "hard_negatives_top": hard_negs,
        "usage_recommended": {
            "alarm_if": "prob_robo >= conformal.prob_threshold_for_alarm AND entropy <= ood.entropy_threshold",
            "note": "Si entropy supera el umbral, tratar como incertidumbre/OOD.",
        },
    }
    out_path = Path(args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[OK] Guardado: {out_path}")


if __name__ == "__main__":
    main()

