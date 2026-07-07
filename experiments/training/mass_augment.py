"""Augmentación masiva: recetas fijas por clip, balanceo a ~100k filas, eval sintética."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .model_config import ROBBERY_CLASS, SEED
    from .train_model_operations import (
        PoseExample,
        _category_augment_ops_for_clip,
        _copy_pose_example,
        _example_folder_category,
        _example_uid,
        build_model,
        build_pose_dataset_for_eval,
    )
except ImportError:
    from model_config import ROBBERY_CLASS, SEED  # type: ignore
    from train_model_operations import (  # type: ignore
        PoseExample,
        _category_augment_ops_for_clip,
        _copy_pose_example,
        _example_folder_category,
        _example_uid,
        build_model,
        build_pose_dataset_for_eval,
    )

DEFAULT_MASS_AUG_CONFIG_PATH = Path(__file__).parent / "config_mass_augmentation.json"


def load_mass_augment_config(path: str | Path | None = None) -> Dict[str, Any]:
    p = Path(path or DEFAULT_MASS_AUG_CONFIG_PATH)
    if not p.is_file():
        return {"enabled": False, "recipes": [], "variants_per_clip": 0}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"enabled": False, "recipes": [], "variants_per_clip": 0}
    return data


def recipe_bank_from_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    recipes = cfg.get("recipes")
    if isinstance(recipes, list) and recipes:
        return list(recipes)
    return []


def compute_mass_augment_plan(
    train_counts: Dict[int, int],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Calcula variantes base + extras por categoría hacia target_train_rows."""
    variants_per_clip = int(cfg.get("variants_per_clip", len(recipe_bank_from_config(cfg)) or 15))
    target_rows = int(cfg.get("target_train_rows", 100_000))
    include_identity = bool(cfg.get("include_identity", False))
    max_extra = int(cfg.get("max_extra_variants_per_clip", 3))
    n_clips = sum(int(v) for v in train_counts.values())
    if n_clips <= 0:
        return {
            "variants_per_clip": variants_per_clip,
            "include_identity": include_identity,
            "target_train_rows": target_rows,
            "projected_base_rows": 0,
            "projected_total_rows": 0,
            "category_variants": {},
            "extra_variants_by_category": {},
            "train_clips": 0,
        }

    base_mult = 1 + variants_per_clip if include_identity else variants_per_clip
    base_rows = n_clips * base_mult
    extra_by_cat: Dict[int, int] = {int(c): 0 for c in train_counts}
    cap_at_target = bool(cfg.get("cap_at_target", False))

    if cap_at_target and base_rows > target_rows and not include_identity:
        variants_per_clip = max(1, target_rows // n_clips)
        base_mult = variants_per_clip
        base_rows = n_clips * variants_per_clip
        deficit = 0
    else:
        deficit = max(0, target_rows - base_rows)

    if deficit > 0 and train_counts:
        hn_boost = cfg.get("hard_negative_category_boost") or {}
        weights: Dict[int, float] = {}
        for cat, v in train_counts.items():
            w = 1.0 / max(int(v), 1)
            boost = hn_boost.get(str(int(cat)), hn_boost.get(int(cat), 1.0))
            weights[int(cat)] = w * float(boost)
        w_sum = sum(weights.values()) or 1.0
        assigned = 0
        for cat in train_counts:
            share = int(deficit * weights[int(cat)] / w_sum)
            cat_boost = float(hn_boost.get(str(int(cat)), hn_boost.get(int(cat), 1.0)))
            cap_mult = max_extra * max(1.0, cat_boost)
            cap = int(train_counts[cat]) * int(cap_mult)
            extra_by_cat[int(cat)] = min(share, cap)
            assigned += extra_by_cat[int(cat)]
        remainder = deficit - assigned
        cats_by_size = sorted(train_counts.keys(), key=lambda c: train_counts[c])
        idx = 0
        while remainder > 0 and cats_by_size:
            cat = int(cats_by_size[idx % len(cats_by_size)])
            cap = int(train_counts[cat]) * max_extra
            if extra_by_cat[cat] < cap:
                extra_by_cat[cat] += 1
                remainder -= 1
            idx += 1
            if idx > len(cats_by_size) * max_extra * 2:
                break

    category_variants: Dict[str, int] = {}
    robbery_class = int(cfg.get("robbery_class", 6))
    rob_extra = int(cfg.get("robbery_category_extra_variants", 0))
    for cat, n in train_counts.items():
        extra = extra_by_cat.get(int(cat), 0)
        if int(cat) == robbery_class and rob_extra > 0:
            extra += rob_extra
        category_variants[str(int(cat))] = variants_per_clip + extra

    projected_total = 0
    for cat, n in train_counts.items():
        mult = (1 + category_variants[str(int(cat))]) if include_identity else category_variants[str(int(cat))]
        projected_total += int(n) * mult

    return {
        "variants_per_clip": variants_per_clip,
        "include_identity": include_identity,
        "target_train_rows": target_rows,
        "cap_at_target": cap_at_target,
        "projected_base_rows": base_rows,
        "projected_total_rows": projected_total,
        "category_variants": category_variants,
        "extra_variants_by_category": {str(k): v for k, v in extra_by_cat.items()},
        "train_clips": n_clips,
    }


def apply_mass_plan_to_training_plan(
    plan: Dict[str, Any],
    mass_plan: Dict[str, Any],
    cat_cfg: Dict[str, Any],
) -> None:
    """Sustituye dataset_stats/totals del preflight estándar por proyección mass-augment."""
    try:
        from .train_model_operations import summarize_category_aug_on_train
    except ImportError:
        from train_model_operations import summarize_category_aug_on_train  # type: ignore

    train_counts = {
        int(k): int(v) for k, v in (plan.get("split_stats_by_category", {}).get("train") or {}).items()
    }
    val_counts = {
        int(k): int(v) for k, v in (plan.get("split_stats_by_category", {}).get("val") or {}).items()
    }
    test_counts = {
        int(k): int(v) for k, v in (plan.get("split_stats_by_category", {}).get("test") or {}).items()
    }
    rows_detail = summarize_category_aug_on_train(train_counts, cat_cfg)
    include_identity = bool(mass_plan.get("include_identity", cat_cfg.get("include_identity", False)))
    projected = int(mass_plan.get("projected_total_rows", 0))
    real_train = int(mass_plan.get("train_clips", sum(train_counts.values())))

    by_cat: Dict[str, Dict[str, int]] = {}
    total_rows = 0
    total_synthetic = 0
    cats = sorted(set(train_counts) | set(val_counts) | set(test_counts), key=lambda c: int(c))
    for cat in cats:
        clips = int(train_counts.get(cat, 0))
        detail = rows_detail.get(int(cat), {})
        rows = int(detail.get("train_rows", 0))
        synthetic = max(0, rows - clips) if include_identity else rows
        by_cat[str(int(cat))] = {
            "clips_real_train": clips,
            "clips_real_val": int(val_counts.get(cat, 0)),
            "clips_real_test": int(test_counts.get(cat, 0)),
            "aug_per_clip": int(detail.get("aug_per_clip", 0)),
            "rows_synthetic_train": synthetic,
            "rows_total_train": rows,
        }
        total_rows += rows
        total_synthetic += synthetic

    if projected <= 0:
        projected = total_rows
    if include_identity:
        total_synthetic = max(0, projected - real_train)
    else:
        total_synthetic = projected

    plan["dataset_stats"] = {
        "by_category": by_cat,
        "totals": {
            "clips_real_train": real_train,
            "clips_real_val": sum(int(v) for v in val_counts.values()),
            "clips_real_test": sum(int(v) for v in test_counts.values()),
            "rows_synthetic_train": total_synthetic,
            "rows_total_train": projected,
            "include_identity_in_rows": include_identity,
        },
    }
    plan.setdefault("totals", {})
    plan["totals"]["rows_train_proposed"] = projected
    plan["totals"]["rows_synthetic_train"] = total_synthetic
    plan["totals"]["uids_train"] = plan["totals"].get("uids_train") or len(
        plan.get("split_uids", {}).get("train") or []
    )


def mass_aug_to_category_config(cfg: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    cats = plan.get("category_variants") or {}
    return {
        "enabled": True,
        "include_identity": bool(plan.get("include_identity", cfg.get("include_identity", False))),
        "default": 0,
        "categories": {str(k): int(v) for k, v in sorted(cats.items(), key=lambda x: int(x[0]))},
        "_mass_augment": True,
        "_variants_per_clip_base": int(plan.get("variants_per_clip", 0)),
        "_target_train_rows": int(plan.get("target_train_rows", 0)),
    }


def expand_examples_with_mass_augmentation(
    examples: List[PoseExample],
    cfg: Dict[str, Any],
    *,
    mass_plan: Optional[Dict[str, Any]] = None,
    augment_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    seed: int = SEED,
) -> List[PoseExample]:
    """Expansión masiva SOLO train con recetas fijas del config."""
    if not cfg.get("enabled", True):
        return examples

    bank = recipe_bank_from_config(cfg)
    if not bank:
        return examples

    plan = mass_plan or {}
    include_identity = bool(plan.get("include_identity", cfg.get("include_identity", False)))
    use_all = bool(cfg.get("use_all_recipes", True))
    base_n = int(plan.get("variants_per_clip", cfg.get("variants_per_clip", len(bank))))
    cat_variants = plan.get("category_variants") or {}

    ranges = augment_ranges or {
        "rotate_degrees": (-12.0, 12.0),
        "scale_percentage": (92.0, 110.0),
        "shift_dx": (-0.05, 0.05),
        "shift_dy": (-0.05, 0.05),
        "noise_sigma_x": (0.0, 0.004),
        "noise_sigma_y": (0.0, 0.004),
        "speed_factor": (0.85, 1.15),
    }

    out: List[PoseExample] = []
    expanded = 0
    added = 0

    for ex in examples:
        if ex.forced_ops is not None:
            out.append(ex)
            continue

        cat = int(_example_folder_category(ex))
        uid = _example_uid(ex)
        n_req = int(cat_variants.get(str(cat), base_n)) if cat_variants else base_n

        if use_all:
            recipes = bank[: min(base_n, len(bank))]
            ops_list = [list(r.get("ops") or []) for r in recipes]
            if n_req > len(ops_list):
                ops_list.extend(
                    _category_augment_ops_for_clip(
                        uid, n_req - len(ops_list), bank, ranges, seed=seed,
                    )
                )
        else:
            ops_list = _category_augment_ops_for_clip(uid, n_req, bank, ranges, seed=seed)

        if include_identity:
            out.append(_copy_pose_example(ex, forced_ops=[]))
        for ops in ops_list:
            out.append(_copy_pose_example(ex, forced_ops=list(ops)))
            added += 1
        expanded += 1

    print(
        f"[MASS-AUG] Clips expandidos: {expanded} | variantes añadidas: {added} | "
        f"filas totales: {len(out)} (antes {len(examples)})"
    )
    return out


def build_synthetic_eval_examples(
    base_examples: Sequence[PoseExample],
    *,
    cfg: Dict[str, Any],
    robbery_class: int = ROBBERY_CLASS,
    clips_x: int = 40,
    variants_y: int = 8,
    seed: int = SEED,
) -> Tuple[List[PoseExample], List[PoseExample]]:
    """X clips no-robo y X robos; Y variantes sintéticas por clip."""
    bank = recipe_bank_from_config(cfg)
    if not bank:
        return [], []

    rng = random.Random(int(seed))
    neg = [ex for ex in base_examples if int(_example_folder_category(ex)) != int(robbery_class)]
    pos = [ex for ex in base_examples if int(_example_folder_category(ex)) == int(robbery_class)]

    def _sample(pool: List[PoseExample], n: int) -> List[PoseExample]:
        if not pool:
            return []
        if len(pool) <= n:
            return list(pool)
        return rng.sample(pool, n)

    neg_pick = _sample(neg, clips_x)
    pos_pick = _sample(pos, clips_x)
    y = min(variants_y, len(bank))

    def _variants(ex: PoseExample) -> List[PoseExample]:
        rows: List[PoseExample] = []
        for i in range(y):
            ops = list(bank[i].get("ops") or [])
            rows.append(_copy_pose_example(ex, forced_ops=ops))
        return rows

    syn_neg = [v for ex in neg_pick for v in _variants(ex)]
    syn_pos = [v for ex in pos_pick for v in _variants(ex)]
    return syn_neg, syn_pos


def _predict_examples(
    model_path: Path,
    examples: List[PoseExample],
    *,
    label_to_idx: Dict[int, int],
    task: str,
    seq_len: int,
    arch_cfg: Dict[str, Any],
    device: Any,
    batch_size: int = 64,
    threshold: float = 0.5,
    temperature: float = 1.0,
) -> Tuple[List[int], List[int], List[float]]:
    import torch
    from torch.utils.data import DataLoader

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)
    input_dim = 34 * 2

    ds = build_pose_dataset_for_eval(
        examples,
        label_to_idx,
        seq_len,
        dataset_split="val",
        checkpoint={"config": arch_cfg},
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(
        str(arch_cfg.get("arch", "tcn")),
        input_dim,
        num_classes,
        arch_cfg,
    ).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    ex_iter = iter(examples)
    pos_class = int(arch_cfg.get("positive_class", ROBBERY_CLASS))
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device)
            logits = model(x) / float(max(1e-6, temperature))
            if task == "binary":
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = (probs >= threshold).long().cpu().tolist()
                prob_list = probs.cpu().tolist()
            else:
                probs_pos = torch.softmax(logits, dim=1)
                if num_classes == 2:
                    prob_list = probs_pos[:, 1].cpu().tolist()
                else:
                    idx_pos = label_to_idx.get(pos_class, 1)
                    prob_list = probs_pos[:, int(idx_pos)].cpu().tolist()
                preds = logits.argmax(dim=1).cpu().tolist()
            for i in range(len(preds)):
                ex = next(ex_iter)
                true_cat = int(_example_folder_category(ex))
                if task == "binary":
                    y_true.append(1 if true_cat == pos_class else 0)
                    y_pred.append(int(preds[i]))
                else:
                    y_true.append(true_cat)
                    pred_label = int(idx_to_label.get(int(preds[i]), int(preds[i])))
                    y_pred.append(pred_label)
                y_prob.append(float(prob_list[i]))
    return y_true, y_pred, y_prob


def synthetic_eval_metrics(
    y_true: List[int],
    y_pred: List[int],
    *,
    task: str,
    robbery_class: int = ROBBERY_CLASS,
) -> Dict[str, Any]:
    if not y_true:
        return {"n": 0}

    if task == "binary":
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        n_neg = max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return {
            "n": len(y_true),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "f1_pct": 100.0 * f1,
            "recall_pct": 100.0 * rec,
            "precision_pct": 100.0 * prec,
            "fp_rate_pct": 100.0 * fp / n_neg,
        }

    y_true_bin = [1 if int(t) == int(robbery_class) else 0 for t in y_true]
    y_pred_bin = [1 if int(p) == int(robbery_class) else 0 for p in y_pred]
    return synthetic_eval_metrics(
        y_true_bin,
        y_pred_bin,
        task="binary",
        robbery_class=robbery_class,
    )


def run_synthetic_battery_for_model(
    model_path: Path,
    *,
    val_examples: List[PoseExample],
    cfg: Dict[str, Any],
    task: str,
    label_to_idx: Dict[int, int],
    arch_cfg: Dict[str, Any],
    robbery_class: int = ROBBERY_CLASS,
    device: Any = None,
    threshold: float = 0.5,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    import torch

    syn_cfg = cfg.get("synthetic_eval") or {}
    clips_x = int(syn_cfg.get("clips_per_category_pool", 40))
    variants_y = int(syn_cfg.get("variants_per_clip", 8))
    seed = int(syn_cfg.get("seed", SEED))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    syn_neg, syn_pos = build_synthetic_eval_examples(
        val_examples,
        cfg=cfg,
        robbery_class=robbery_class,
        clips_x=clips_x,
        variants_y=variants_y,
        seed=seed,
    )

    arch = dict(arch_cfg)
    arch["positive_class"] = robbery_class
    seq_len = int(arch.get("seq_len", 64))

    neg_true, neg_pred, _ = _predict_examples(
        model_path, syn_neg, label_to_idx=label_to_idx, task=task,
        seq_len=seq_len, arch_cfg=arch, device=device, threshold=threshold, temperature=temperature,
    )
    pos_true, pos_pred, _ = _predict_examples(
        model_path, syn_pos, label_to_idx=label_to_idx, task=task,
        seq_len=seq_len, arch_cfg=arch, device=device, threshold=threshold, temperature=temperature,
    )

    return {
        "clips_x": clips_x,
        "variants_y": variants_y,
        "threshold": threshold,
        "temperature": temperature,
        "synthetic_negatives": synthetic_eval_metrics(
            neg_true, neg_pred, task=task, robbery_class=robbery_class,
        ),
        "synthetic_robbery": synthetic_eval_metrics(
            pos_true, pos_pred, task=task, robbery_class=robbery_class,
        ),
        "n_synthetic_neg_rows": len(syn_neg),
        "n_synthetic_pos_rows": len(syn_pos),
    }


def run_synthetic_battery_for_ensemble(
    model_paths: List[Path],
    *,
    val_examples: List[PoseExample],
    cfg: Dict[str, Any],
    task: str,
    rule: str = "mean",
    thresholds: Any = 0.5,
    robbery_class: int = ROBBERY_CLASS,
    device: Any = None,
) -> Dict[str, Any]:
    """Batería sintética con ensemble (mean/and/cascade simplificado)."""
    import torch

    if not model_paths:
        return {"available": False, "error": "Sin modelos ensemble"}

    syn_cfg = cfg.get("synthetic_eval") or {}
    clips_x = int(syn_cfg.get("clips_per_category_pool", 40))
    variants_y = int(syn_cfg.get("variants_per_clip", 8))
    seed = int(syn_cfg.get("seed", SEED))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    syn_neg, syn_pos = build_synthetic_eval_examples(
        val_examples, cfg=cfg, robbery_class=robbery_class,
        clips_x=clips_x, variants_y=variants_y, seed=seed,
    )

    if isinstance(thresholds, (list, tuple)):
        thr_list = [float(t) for t in thresholds]
    elif isinstance(thresholds, str) and thresholds.strip():
        thr_list = [float(t) for t in thresholds.split("|") if t.strip()]
    else:
        thr_list = [float(thresholds or 0.5)]

    def _ensemble_predict(examples: List[PoseExample]) -> Tuple[List[int], List[int]]:
        prob_stack: List[List[float]] = []
        y_true_ref: List[int] = []
        for mp in model_paths:
            ckpt = torch.load(mp, map_location=device, weights_only=False)
            label_to_idx = ckpt["label_to_idx"]
            arch_cfg = dict(ckpt.get("config") or {})
            arch_cfg["seq_len"] = int(ckpt.get("seq_len", 64))
            arch_cfg["positive_class"] = robbery_class
            yt, _, probs = _predict_examples(
                mp, examples, label_to_idx=label_to_idx, task=task,
                seq_len=int(arch_cfg["seq_len"]), arch_cfg=arch_cfg, device=device,
                threshold=0.5,
            )
            if not y_true_ref:
                y_true_ref = yt
            prob_stack.append(probs)

        n = len(y_true_ref)
        preds: List[int] = []
        for i in range(n):
            vals = [float(p[i]) for p in prob_stack]
            if rule == "and":
                thrs = thr_list * len(vals) if len(thr_list) < len(vals) else thr_list[: len(vals)]
                preds.append(1 if all(v >= t for v, t in zip(vals, thrs)) else 0)
            elif rule == "cascade" and len(vals) >= 2:
                low, high = 0.4, 0.55
                preds.append(1 if vals[0] >= low and vals[1] >= high else 0)
            else:
                thr = thr_list[0] if thr_list else 0.5
                mean_p = sum(vals) / max(len(vals), 1)
                preds.append(1 if mean_p >= thr else 0)
        return y_true_ref, preds

    neg_true, neg_pred = _ensemble_predict(syn_neg)
    pos_true, pos_pred = _ensemble_predict(syn_pos)

    return {
        "clips_x": clips_x,
        "variants_y": variants_y,
        "rule": rule,
        "thresholds": thr_list,
        "synthetic_negatives": synthetic_eval_metrics(
            neg_true, neg_pred, task=task, robbery_class=robbery_class,
        ),
        "synthetic_robbery": synthetic_eval_metrics(
            pos_true, pos_pred, task=task, robbery_class=robbery_class,
        ),
        "n_synthetic_neg_rows": len(syn_neg),
        "n_synthetic_pos_rows": len(syn_pos),
    }
