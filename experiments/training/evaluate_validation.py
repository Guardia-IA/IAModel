#!/usr/bin/env python3
"""
Evaluación en el split de validación (o test) usando training_plan.json o split_manifest.

Garantiza evaluar solo clips del split elegido (p. ej. val), sin mezclar train.
En multiclase muestra métricas por categoría de carpeta (0–14).
En binario compara softmax-argmax, softmax-umbral y margen en logits.

Uso:
  # Un modelo
  python evaluate_validation.py --model models-operation-single/modelo_08.pt \\
      --training-plan training_plan.json --split val --single-user-only

  # Todos los modelos de la carpeta + tabla resumen F1/FP clase 6
  python evaluate_validation.py --models-dir models-operation-single \\
      --training-plan training_plan.json --split val --single-user-only \\
      --summary-json reports/val_all.json --quiet
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .model_config import (  # type: ignore[attr-defined]
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
        DEFAULT_BINARY_SOFTMAX_THRESHOLD,
        DEFAULT_BINARY_LOGIT_MARGIN,
        ROBBERY_CLASS,
        TRAINING_PLAN_PATH,
    )
    from .train_model_operations import (  # type: ignore[attr-defined]
        MODELS_DIR,
        MODELS_SINGLE_DIR,
        PoseExample,
        collect_examples,
        make_binary_examples,
        build_model,
        build_pose_dataset_for_eval,
        example_in_split_set,
        split_examples_by_uid_manifest,
        load_training_plan_json,
        verify_split_uid_disjoint,
        evaluate_with_metrics,
        format_binary_metrics_line,
        _example_folder_category,
        _example_uid,
        SEED,
    )
except ImportError:
    from model_config import (  # type: ignore[attr-defined]
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
        DEFAULT_BINARY_SOFTMAX_THRESHOLD,
        DEFAULT_BINARY_LOGIT_MARGIN,
        ROBBERY_CLASS,
        TRAINING_PLAN_PATH,
    )
    from train_model_operations import (  # type: ignore[attr-defined]
        MODELS_DIR,
        MODELS_SINGLE_DIR,
        PoseExample,
        collect_examples,
        make_binary_examples,
        build_model,
        build_pose_dataset_for_eval,
        example_in_split_set,
        split_examples_by_uid_manifest,
        load_training_plan_json,
        verify_split_uid_disjoint,
        evaluate_with_metrics,
        format_binary_metrics_line,
        _example_folder_category,
        _example_uid,
        SEED,
    )

RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"


def header(title: str) -> None:
    line = "=" * 72
    print(f"\n{BOLD}{line}\n{title}\n{line}{RESET}")


def load_split_uids(
    *,
    split_name: str,
    training_plan_path: Optional[Path] = None,
    split_manifest_path: Optional[Path] = None,
) -> Tuple[Set[str], Dict[str, Any]]:
    """Carga UIDs del split desde training_plan.json o split_manifest."""
    meta: Dict[str, Any] = {"split_name": split_name, "source": None}

    if training_plan_path is not None:
        plan = load_training_plan_json(training_plan_path)
        if split_name not in plan.get("split_uids", {}):
            raise ValueError(f"Split {split_name!r} no está en training_plan.split_uids")
        uids = {str(u) for u in plan["split_uids"][split_name]}
        meta["source"] = "training_plan"
        meta["training_plan_path"] = str(training_plan_path)
        meta["task"] = plan.get("task")
        meta["positive_class"] = plan.get("positive_class")
        meta["filters"] = plan.get("filters", {})
        meta["data_root"] = plan.get("data_root")
        meta["pose_source"] = plan.get("pose_source", "filtered")
        meta["single_user_only"] = plan.get("single_user_only", False)
        meta["split_uids_all"] = {
            k: [str(x) for x in v] for k, v in plan.get("split_uids", {}).items()
        }
        meta["evaluation"] = plan.get("evaluation", {})
        return uids, meta

    if split_manifest_path is not None:
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        split = data.get("split", {})
        if "split_uids_clips" in split:
            block = split["split_uids_clips"]
            if split_name not in block:
                raise ValueError(
                    f"Split {split_name!r} no está en split.split_uids_clips"
                )
            uids = {str(u) for u in block[split_name]}
        elif split_name in split:
            uids = {str(u) for u in split[split_name]}
        else:
            raise ValueError(
                f"Split {split_name!r} no encontrado en {split_manifest_path}"
            )
        meta["source"] = "split_manifest"
        meta["split_manifest_path"] = str(split_manifest_path)
        meta["task"] = data.get("task")
        meta["positive_class"] = data.get("positive_class")
        meta["filters"] = data.get("filters", {})
        meta["pose_source"] = data.get("pose_source", "filtered")
        meta["single_user_only"] = data.get("single_user_only", False)
        return uids, meta

    raise ValueError("Indica --training-plan o --split-manifest")


def assert_split_disjoint_from_train(
    examples: List[PoseExample],
    split_uids_all: Dict[str, List[str]],
    eval_split: str,
) -> None:
    train_uids = set(split_uids_all.get("train", []))
    eval_uids = set(split_uids_all.get(eval_split, []))
    if train_uids & eval_uids:
        raise ValueError(
            f"Fuga train∩{eval_split}: {len(train_uids & eval_uids)} UIDs compartidos"
        )
    for ex in examples:
        uid = _example_uid(ex)
        if uid in train_uids:
            raise ValueError(
                f"Ejemplo de eval en split train (UID): {uid[:80]}..."
            )


def build_split_examples(
    *,
    split_uids: Set[str],
    split_meta: Dict[str, Any],
    pose_source: Optional[str] = None,
    single_user_only: Optional[bool] = None,
    task: Optional[str] = None,
    positive_class: int = 6,
    min_clip_seconds: float = MIN_CLIP_SECONDS,
    min_valid_frames: int = MIN_VALID_FRAMES,
    min_valid_pct: float = MIN_VALID_PCT,
    max_occlusion_ratio: float = MAX_OCCLUSION_RATIO,
    data_root: Optional[Path] = None,
) -> Tuple[List[PoseExample], Dict[str, Any]]:
    filters = split_meta.get("filters") or {}
    ps = pose_source or split_meta.get("pose_source", "filtered")
    su = single_user_only if single_user_only is not None else bool(
        split_meta.get("single_user_only", False)
    )
    t_task = task or split_meta.get("task", "multiclass")
    pos = int(split_meta.get("positive_class", positive_class))

    root = data_root
    if root is None and split_meta.get("data_root"):
        root = Path(str(split_meta["data_root"]))

    collect_kw: Dict[str, Any] = dict(
        pose_source=ps,
        single_user_only=su,
        min_clip_seconds=float(filters.get("min_clip_seconds", min_clip_seconds)),
        min_valid_frames=int(filters.get("min_valid_frames", min_valid_frames)),
        min_valid_pct=float(filters.get("min_valid_pct", min_valid_pct)),
        max_occlusion_ratio=float(filters.get("max_occlusion_ratio", max_occlusion_ratio)),
    )
    if root is not None:
        collect_kw["data_root"] = root

    all_examples = collect_examples(**collect_kw)
    if t_task == "binary":
        all_examples = make_binary_examples(all_examples, positive_class=pos)

    split_uids_dict = split_meta.get("split_uids_all")
    if split_uids_dict:
        verify_split_uid_disjoint(split_uids_dict)
        _, val_ex, test_ex = split_examples_by_uid_manifest(
            all_examples, split_uids_dict
        )
        if split_meta.get("split_name") == "val":
            examples = val_ex
        elif split_meta.get("split_name") == "test":
            examples = test_ex
        else:
            examples = [ex for ex in all_examples if example_in_split_set(ex, split_uids)]
        assert_split_disjoint_from_train(
            examples, split_uids_dict, str(split_meta.get("split_name", "val"))
        )
    else:
        examples = [ex for ex in all_examples if example_in_split_set(ex, split_uids)]

    info = {
        "pool_total": len(all_examples),
        "split_examples": len(examples),
        "unique_uids": len({_example_uid(ex) for ex in examples}),
        "pose_source": ps,
        "single_user_only": su,
        "task": t_task,
        "positive_class": pos,
    }
    return examples, info


@torch.no_grad()
def metrics_by_folder_category(
    model: nn.Module,
    loader: DataLoader,
    examples: List[PoseExample],
    label_to_idx: Dict[int, int],
    device: torch.device,
) -> Dict[str, Any]:
    """Métricas por categoría de carpeta (0–14)."""
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    model.eval()

    conf: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    support: Dict[int, int] = defaultdict(int)
    ex_iter = iter(examples)

    for x, _y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().tolist()
        for i in range(len(preds)):
            ex = next(ex_iter)
            true_cat = int(_example_folder_category(ex))
            pred_label = int(idx_to_label.get(int(preds[i]), int(preds[i])))
            support[true_cat] += 1
            conf[true_cat][pred_label] += 1

    per_cat: Dict[str, Dict[str, Any]] = {}
    total = sum(support.values())
    for cat in sorted(support.keys()):
        sup = int(support[cat])
        tp = int(conf[cat].get(cat, 0))
        fn = sup - tp
        fp = int(sum(conf[t].get(cat, 0) for t in support if t != cat))
        support_neg = int(total - sup)
        fp_rate = fp / max(support_neg, 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_cat[str(cat)] = {
            "support": sup,
            "support_neg": support_neg,
            "correct": tp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "accuracy_pct": float(100.0 * tp / sup) if sup > 0 else None,
            "precision_pct": float(100.0 * prec),
            "recall_pct": float(100.0 * rec),
            "f1_pct": float(100.0 * f1),
            "false_positive_rate_pct": float(100.0 * fp_rate),
            "top_confusions": _top_confusions_for_cat(conf, cat, limit=3),
        }

    correct = sum(conf[c].get(c, 0) for c in support)
    return {
        "overall_accuracy_pct": float(100.0 * correct / total) if total else 0.0,
        "total_examples": int(total),
        "per_category": per_cat,
        "confusion_by_folder_category": {
            str(k): {str(pk): int(pv) for pk, pv in v.items()} for k, v in conf.items()
        },
    }


def _top_confusions_for_cat(
    conf: Dict[int, Dict[int, int]],
    cat: int,
    limit: int = 3,
) -> List[Dict[str, int]]:
    row = conf.get(cat, {})
    items = [(pred, cnt) for pred, cnt in row.items() if pred != cat and cnt > 0]
    items.sort(key=lambda x: -x[1])
    return [{"predicted_as": int(p), "count": int(c)} for p, c in items[:limit]]


def compute_robbery_class_metrics(
    conf: Dict[int, Dict[int, int]],
    support: Dict[int, int],
    robbery_class: int = ROBBERY_CLASS,
) -> Dict[str, Any]:
    """Compat: métricas one-vs-rest de la clase robo a partir de la matriz de confusión."""
    total = sum(support.values())
    cat = int(robbery_class)
    sup = int(support.get(cat, 0))
    tp = int(conf.get(cat, {}).get(cat, 0))
    fn = sup - tp
    fp = int(sum(conf.get(t, {}).get(cat, 0) for t in support if t != cat))
    support_neg = int(total - sup)
    fp_rate = fp / max(support_neg, 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fp_by: Dict[str, int] = {}
    for true_cat in sorted(support.keys()):
        if true_cat == cat:
            continue
        n = int(conf.get(true_cat, {}).get(cat, 0))
        if n > 0:
            fp_by[str(true_cat)] = n
    return {
        "robbery_class": cat,
        "support_pos": sup,
        "support_neg": support_neg,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_pct": float(100.0 * prec),
        "recall_pct": float(100.0 * rec),
        "f1_pct": float(100.0 * f1),
        "false_positive_rate_pct": float(100.0 * fp_rate),
        "false_positives_by_true_category": fp_by,
    }


def print_multiclass_category_table(folder_metrics: Dict[str, Any]) -> None:
    header("F1 y tasa de FP por categoría (one-vs-rest, argmax)")
    print(f"  Accuracy global: {GREEN}{folder_metrics['overall_accuracy_pct']:.2f}%{RESET}")
    print(f"  Ejemplos: {folder_metrics['total_examples']}")
    print(
        f"\n  {YELLOW}FP%{RESET} = % de clips cuya categoría real NO es K pero se predice K."
    )
    print(
        f"\n{'cat':>4} | {'n':>5} | {'F1%':>6} | {'FP%':>7} | {'Rec%':>6} | {'Prec%':>6} | confusiones"
    )
    print("-" * 78)
    per = folder_metrics.get("per_category", {})
    for cat in sorted(per.keys(), key=lambda x: int(x)):
        m = per[cat]
        confs = m.get("top_confusions") or []
        conf_s = ", ".join(f"→{c['predicted_as']}({c['count']})" for c in confs) or "-"
        if m["support"] < 5:
            conf_s = f"{YELLOW}{conf_s}{RESET}"
        fp_s = f"{m['false_positive_rate_pct']:7.2f}"
        if int(cat) == ROBBERY_CLASS:
            fp_s = f"{YELLOW}{fp_s}{RESET}"
        print(
            f"{int(cat):4d} | {m['support']:5d} | {m['f1_pct']:6.1f} | {fp_s} | "
            f"{m['recall_pct']:6.1f} | {m['precision_pct']:6.1f} | {conf_s}"
        )


def print_standard_metrics(
    task: str,
    loss: float,
    acc: float,
    metrics: Dict[str, Any],
) -> None:
    header("Métricas globales (softmax argmax)")
    print(f"  loss={loss:.4f} | accuracy={acc:.4f} | macro_f1={metrics.get('macro_f1', 0):.4f}")
    if task == "binary" and metrics.get("binary"):
        print(f"\n  {format_binary_metrics_line(metrics['binary'])}")
    per = metrics.get("per_class_accuracy_pct")
    if per and task == "multiclass":
        parts = [
            f"c{k}={v:.0f}%"
            for k, v in sorted(per.items(), key=lambda x: int(x[0]))
            if v is not None
        ]
        if parts:
            print(
                f"  Por índice de modelo: {' '.join(parts[:10])}"
                + (" ..." if len(parts) > 10 else "")
            )


@torch.no_grad()
def evaluate_validation(
    model_path: Path,
    *,
    split_name: str = "val",
    training_plan_path: Optional[Path] = None,
    split_manifest_path: Optional[Path] = None,
    pose_source: Optional[str] = None,
    single_user_only: Optional[bool] = None,
    batch_size: int = 64,
    num_workers: int = 2,
    binary_softmax_threshold: Optional[float] = None,
    binary_logit_margin: Optional[float] = None,
    output_json: Optional[Path] = None,
    robbery_class: int = ROBBERY_CLASS,
    quiet: bool = False,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not quiet:
        print(f"Dispositivo: {device}")
        print(f"Modelo: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    label_to_idx = checkpoint["label_to_idx"]
    seq_len = int(checkpoint.get("seq_len", 64))
    task = str(checkpoint.get("task", "multiclass"))
    positive_class = int(checkpoint.get("positive_class", 6))
    cfg = checkpoint.get("config", {})
    arch = cfg.get("arch", "tcn")
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint.get("num_classes", len(label_to_idx)))

    split_uids, split_meta = load_split_uids(
        split_name=split_name,
        training_plan_path=training_plan_path,
        split_manifest_path=split_manifest_path,
    )
    split_meta["split_name"] = split_name

    examples, pool_info = build_split_examples(
        split_uids=split_uids,
        split_meta=split_meta,
        pose_source=pose_source,
        single_user_only=single_user_only,
        task=task,
        positive_class=positive_class,
    )

    if not quiet:
        header(f"Split '{split_name}' — clips de evaluación")
        print(f"  Fuente UIDs: {CYAN}{split_meta.get('source')}{RESET}")
        print(f"  Ejemplos: {pool_info['split_examples']} | UIDs únicos: {pool_info['unique_uids']}")
        print(f"  Pool total tras collect: {pool_info['pool_total']}")
        print(f"  {GREEN}Sin augment{RESET} (mismo criterio que val en entrenamiento)")

    if not examples:
        raise RuntimeError(f"No hay ejemplos en split {split_name!r}")

    examples = [ex for ex in examples if ex.label in label_to_idx]
    if not quiet:
        print(f"  Tras filtrar clases del modelo: {len(examples)}")

    ds = build_pose_dataset_for_eval(
        examples,
        label_to_idx,
        seq_len,
        dataset_split=split_name,
        checkpoint=checkpoint,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(arch, input_dim, num_classes, cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_cfg = split_meta.get("evaluation") or {}
    bs_thr = float(
        binary_softmax_threshold
        if binary_softmax_threshold is not None
        else eval_cfg.get("binary_softmax_threshold", DEFAULT_BINARY_SOFTMAX_THRESHOLD)
    )
    bs_margin = float(
        binary_logit_margin
        if binary_logit_margin is not None
        else eval_cfg.get("binary_logit_margin", DEFAULT_BINARY_LOGIT_MARGIN)
    )

    criterion = nn.CrossEntropyLoss()
    loss, acc, metrics = evaluate_with_metrics(
        model,
        loader,
        criterion,
        device,
        num_classes=num_classes,
        task=task,
        binary_softmax_threshold=bs_thr,
        binary_logit_margin=bs_margin,
    )

    if not quiet:
        print_standard_metrics(task, loss, acc, metrics)

    folder_metrics: Optional[Dict[str, Any]] = None
    robbery_metrics: Optional[Dict[str, Any]] = None
    if task == "multiclass":
        loader_cat = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        folder_metrics = metrics_by_folder_category(
            model, loader_cat, examples, label_to_idx, device
        )
        robbery_metrics = compute_robbery_class_metrics(
            {
                int(k): {int(pk): pv for pk, pv in v.items()}
                for k, v in folder_metrics["confusion_by_folder_category"].items()
            },
            {int(k): folder_metrics["per_category"][k]["support"] for k in folder_metrics["per_category"]},
            robbery_class=robbery_class,
        )
        folder_metrics["robbery_class_metrics"] = robbery_metrics
        if not quiet:
            print_multiclass_category_table(folder_metrics)
    elif task == "binary" and metrics.get("binary") and not quiet:
        b = metrics["binary"]["softmax_argmax"]
        header(f"Binario — clase positiva {positive_class}")
        f1_pos = 2 * b["precision_robbery_pct"] * b["recall_robbery_pct"] / max(
            b["precision_robbery_pct"] + b["recall_robbery_pct"], 1e-9
        )
        print(
            f"  F1≈{f1_pos:.1f}% | Recall={b['recall_robbery_pct']:.1f}% | "
            f"Precision={b['precision_robbery_pct']:.1f}% | "
            f"FP rate={b['false_positive_rate_pct']:.2f}%"
        )
        print(f"  {format_binary_metrics_line(metrics['binary'])}")

    report: Dict[str, Any] = {
        "model_path": str(model_path.resolve()),
        "split_name": split_name,
        "split_source": split_meta.get("source"),
        "task": task,
        "positive_class": positive_class,
        "pool_info": pool_info,
        "split_uids_count": len(split_uids),
        "evaluated_examples": len(examples),
        "loss": float(loss),
        "accuracy": float(acc),
        "metrics": metrics,
        "folder_category_metrics": folder_metrics,
        "robbery_class_metrics": robbery_metrics,
        "binary_softmax_threshold": bs_thr,
        "binary_logit_margin": bs_margin,
        "seed": SEED,
        "arch": arch,
        "exp_id": _exp_id_from_model_path(model_path),
    }

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\n{GREEN}Informe guardado:{RESET} {output_json}")

    return report


def _exp_id_from_model_path(model_path: Path) -> Optional[int]:
    stem = model_path.stem
    if stem.startswith("modelo_"):
        try:
            return int(stem.split("_", 1)[1])
        except ValueError:
            return None
    return None


def _categories_in_results(results: List[Dict[str, Any]]) -> List[int]:
    cats: set[int] = set()
    for r in results:
        per = (r.get("folder_category_metrics") or {}).get("per_category") or {}
        cats.update(int(k) for k in per.keys())
    if cats:
        return sorted(cats)
    return list(range(15))


def _metric_cell(per: Dict[str, Dict[str, Any]], cat: int, key: str) -> str:
    m = per.get(str(cat))
    if not m or m.get("support", 0) <= 0:
        return "  n/a"
    val = float(m.get(key, 0.0))
    return f"{val:5.1f}"


def print_batch_summary(results: List[Dict[str, Any]], split_name: str) -> None:
    cats = _categories_in_results(results)
    header(f"Resumen — {len(results)} modelos en split '{split_name}'")

    print(
        f"\n{BOLD}{'Modelo':>14} | {'Arch':>10} | {'Acc%':>6} | {'MacroF1':>7}{RESET}"
    )
    print("-" * 46)
    for r in sorted(results, key=lambda x: -float(x.get("accuracy", 0.0))):
        name = Path(r["model_path"]).name
        arch = str(r.get("arch", "?"))[:10]
        acc = 100.0 * float(r.get("accuracy", 0.0))
        macro = float((r.get("metrics") or {}).get("macro_f1", 0.0))
        print(f"{name:>14} | {arch:>10} | {acc:6.1f} | {macro:7.3f}")

    header("F1 (%) por categoría — one-vs-rest, argmax")
    hdr = f"{BOLD}{'Modelo':>14} | " + " | ".join(f"c{c:>2}" for c in cats) + f"{RESET}"
    print(hdr)
    print("-" * min(120, len(hdr)))
    for r in sorted(
        results,
        key=lambda x: -float(
            ((x.get("folder_category_metrics") or {}).get("per_category") or {})
            .get(str(ROBBERY_CLASS), {})
            .get("f1_pct", 0.0)
        ),
    ):
        per = (r.get("folder_category_metrics") or {}).get("per_category") or {}
        name = Path(r["model_path"]).name[:14]
        cells = [_metric_cell(per, c, "f1_pct") for c in cats]
        print(f"{name:>14} | " + " | ".join(cells))

    header("Tasa de FP (%) por categoría — % de no-K predichos como K")
    print(hdr)
    print("-" * min(120, len(hdr)))
    for r in sorted(
        results,
        key=lambda x: float(
            ((x.get("folder_category_metrics") or {}).get("per_category") or {})
            .get(str(ROBBERY_CLASS), {})
            .get("false_positive_rate_pct", 999.0)
        ),
    ):
        per = (r.get("folder_category_metrics") or {}).get("per_category") or {}
        name = Path(r["model_path"]).name[:14]
        cells = [_metric_cell(per, c, "false_positive_rate_pct") for c in cats]
        print(f"{name:>14} | " + " | ".join(cells))

    print(
        f"\n  Columna c{ROBBERY_CLASS}: clase robo. "
        f"{YELLOW}FP%{RESET} bajo en c6 = menos normales marcados como robo."
    )


def write_summary_csv(results: List[Dict[str, Any]], path: Path) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "exp_id",
                "arch",
                "split",
                "accuracy_pct",
                "macro_f1",
                "category",
                "support",
                "f1_pct",
                "false_positive_rate_pct",
                "recall_pct",
                "precision_pct",
            ]
        )
        for r in results:
            per = (r.get("folder_category_metrics") or {}).get("per_category") or {}
            base = [
                Path(r["model_path"]).name,
                r.get("exp_id", ""),
                r.get("arch", ""),
                r.get("split_name", ""),
                round(100.0 * float(r.get("accuracy", 0.0)), 4),
                round(float((r.get("metrics") or {}).get("macro_f1", 0.0)), 4),
            ]
            if not per:
                w.writerow(base + ["", "", "", "", "", ""])
                continue
            for cat in sorted(per.keys(), key=lambda x: int(x)):
                m = per[cat]
                w.writerow(
                    base
                    + [
                        int(cat),
                        m.get("support", 0),
                        round(float(m.get("f1_pct", 0.0)), 2),
                        round(float(m.get("false_positive_rate_pct", 0.0)), 4),
                        round(float(m.get("recall_pct", 0.0)), 2),
                        round(float(m.get("precision_pct", 0.0)), 2),
                    ]
                )


def resolve_model_paths(
    model: Optional[str],
    models_dir: Optional[str],
    single_user_only: bool,
) -> List[Path]:
    if model:
        return [Path(model).expanduser().resolve()]
    base = Path(models_dir).expanduser().resolve() if models_dir else (
        MODELS_SINGLE_DIR if single_user_only else MODELS_DIR
    )
    paths = sorted(base.glob("modelo_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No hay modelo_*.pt en {base}")
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evalúa uno o todos los modelos en val/test con split fijo."
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Un checkpoint modelo_XX.pt (opcional si usas --models-dir).",
    )
    p.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help=(
            "Carpeta con modelo_*.pt para evaluar todos. "
            "Default: models-operation-single si --single-user-only, si no models-operation."
        ),
    )
    p.add_argument(
        "--training-plan",
        type=str,
        default=None,
        help=f"training_plan.json (recomendado). Default: {TRAINING_PLAN_PATH.name} si existe.",
    )
    p.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="split_manifest_exp_XX.json del entrenamiento (alternativa al plan).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Subset a evaluar (default: val).",
    )
    p.add_argument("--pose-source", choices=["filtered", "full"], default=None)
    p.add_argument("--single-user-only", action="store_true")
    p.add_argument("--no-single-user-only", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--binary-softmax-threshold", type=float, default=None)
    p.add_argument("--binary-logit-margin", type=float, default=None)
    p.add_argument(
        "--robbery-class",
        type=int,
        default=ROBBERY_CLASS,
        help=f"Categoría robo para F1/FP en multiclase (default {ROBBERY_CLASS}).",
    )
    p.add_argument("--output-json", type=str, default=None, help="JSON por modelo (solo con --model).")
    p.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="JSON agregado al evaluar varios modelos (--models-dir).",
    )
    p.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="CSV largo: una fila por (modelo, categoría) con F1% y FP%.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Menos salida por modelo (útil con --models-dir).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model and not args.models_dir:
        print("Indica --model o --models-dir.", file=sys.stderr)
        return 1

    training_plan = Path(args.training_plan).expanduser() if args.training_plan else None
    split_manifest = Path(args.split_manifest).expanduser() if args.split_manifest else None

    if training_plan is None and split_manifest is None:
        default_plan = Path(__file__).parent / TRAINING_PLAN_PATH.name
        if default_plan.is_file():
            training_plan = default_plan
            if not args.quiet:
                print(f"Usando training plan por defecto: {training_plan}")
        else:
            print(
                "Indica --training-plan o --split-manifest (no hay training_plan.json por defecto).",
                file=sys.stderr,
            )
            return 1

    su: Optional[bool] = None
    if args.single_user_only:
        su = True
    elif args.no_single_user_only:
        su = False

    try:
        model_paths = resolve_model_paths(
            args.model,
            args.models_dir,
            single_user_only=bool(su if su is not None else args.single_user_only),
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    batch_mode = len(model_paths) > 1 or (args.models_dir and not args.model)
    results: List[Dict[str, Any]] = []
    errors: List[str] = []

    for mp in model_paths:
        if batch_mode and not args.quiet:
            print("\n" + "#" * 80)
            print(f"# {mp.name}")
            print("#" * 80)
        try:
            report = evaluate_validation(
                mp,
                split_name=args.split,
                training_plan_path=training_plan,
                split_manifest_path=split_manifest,
                pose_source=args.pose_source,
                single_user_only=su,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                binary_softmax_threshold=args.binary_softmax_threshold,
                binary_logit_margin=args.binary_logit_margin,
                output_json=(Path(args.output_json) if args.output_json and not batch_mode else None),
                robbery_class=args.robbery_class,
                quiet=args.quiet and batch_mode,
            )
            results.append(report)
        except Exception as exc:
            errors.append(f"{mp.name}: {exc}")
            print(f"Error en {mp.name}: {exc}", file=sys.stderr)

    if batch_mode and results:
        print_batch_summary(results, args.split)

    if args.summary_json and results:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": args.split,
            "models_evaluated": len(results),
            "errors": errors,
            "results": results,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\n{GREEN}Resumen JSON:{RESET} {out}")

    if args.summary_csv and results:
        csv_out = Path(args.summary_csv)
        write_summary_csv(results, csv_out)
        print(f"{GREEN}Resumen CSV (F1/FP todas las categorías):{RESET} {csv_out}")

    if errors and not results:
        return 1
    return 0 if not errors else 0


if __name__ == "__main__":
    sys.exit(main())
