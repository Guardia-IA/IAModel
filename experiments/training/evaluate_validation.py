#!/usr/bin/env python3
"""
Evaluación en el split de validación (o test) usando training_plan.json o split_manifest.

Garantiza evaluar solo clips del split elegido (p. ej. val), sin mezclar train.
En multiclase muestra métricas por categoría de carpeta (0–14).
En binario compara softmax-argmax, softmax-umbral y margen en logits.

Uso:
  python evaluate_validation.py --model models-operation-single/modelo_01.pt \\
      --training-plan training_plan.json --split val

  python evaluate_validation.py --model modelo_01.pt --split-manifest splits/split_manifest_exp_01.json \\
      --split-name val --single-user-only
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
        TRAINING_PLAN_PATH,
    )
    from .train_model_operations import (  # type: ignore[attr-defined]
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
        TRAINING_PLAN_PATH,
    )
    from train_model_operations import (  # type: ignore[attr-defined]
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
    for cat in sorted(support.keys()):
        sup = support[cat]
        tp = conf[cat].get(cat, 0)
        fn = sup - tp
        fp = sum(conf[t].get(cat, 0) for t in conf if t != cat)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_cat[str(cat)] = {
            "support": int(sup),
            "correct": int(tp),
            "accuracy_pct": float(100.0 * tp / sup) if sup > 0 else None,
            "precision_pct": float(100.0 * prec),
            "recall_pct": float(100.0 * rec),
            "f1_pct": float(100.0 * f1),
            "top_confusions": _top_confusions_for_cat(conf, cat, limit=3),
        }

    total = sum(support.values())
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


def print_multiclass_category_table(folder_metrics: Dict[str, Any]) -> None:
    header("Métricas por categoría (split evaluado)")
    print(f"  Accuracy global (por carpeta): {GREEN}{folder_metrics['overall_accuracy_pct']:.2f}%{RESET}")
    print(f"  Ejemplos: {folder_metrics['total_examples']}")
    print(f"\n{'cat':>4} | {'n':>5} | {'acc%':>6} | {'prec%':>6} | {'rec%':>6} | {'f1%':>6} | confusiones")
    print("-" * 72)
    per = folder_metrics.get("per_category", {})
    for cat in sorted(per.keys(), key=lambda x: int(x)):
        m = per[cat]
        acc = m["accuracy_pct"]
        acc_s = f"{acc:6.1f}" if acc is not None else "   n/a"
        confs = m.get("top_confusions") or []
        conf_s = ", ".join(f"→{c['predicted_as']}({c['count']})" for c in confs) or "-"
        if m["support"] < 5:
            conf_s = f"{YELLOW}{conf_s}{RESET}"
        print(
            f"{int(cat):4d} | {m['support']:5d} | {acc_s} | "
            f"{m['precision_pct']:6.1f} | {m['recall_pct']:6.1f} | {m['f1_pct']:6.1f} | {conf_s}"
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
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    header(f"Split '{split_name}' — clips de evaluación")
    print(f"  Fuente UIDs: {CYAN}{split_meta.get('source')}{RESET}")
    print(f"  Ejemplos: {pool_info['split_examples']} | UIDs únicos: {pool_info['unique_uids']}")
    print(f"  Pool total tras collect: {pool_info['pool_total']}")
    print(f"  {GREEN}Sin augment{RESET} (mismo criterio que val en entrenamiento)")

    if not examples:
        raise RuntimeError(f"No hay ejemplos en split {split_name!r}")

    examples = [ex for ex in examples if ex.label in label_to_idx]
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

    print_standard_metrics(task, loss, acc, metrics)

    folder_metrics: Optional[Dict[str, Any]] = None
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
        print_multiclass_category_table(folder_metrics)

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
        "binary_softmax_threshold": bs_thr,
        "binary_logit_margin": bs_margin,
        "seed": SEED,
    }

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\n{GREEN}Informe guardado:{RESET} {output_json}")

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evalúa un modelo en val/test con split fijo (training_plan o split_manifest)."
    )
    p.add_argument("--model", type=str, required=True, help="Ruta al checkpoint modelo_XX.pt")
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
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        print(f"No existe el modelo: {model_path}", file=sys.stderr)
        return 1

    training_plan = Path(args.training_plan).expanduser() if args.training_plan else None
    split_manifest = Path(args.split_manifest).expanduser() if args.split_manifest else None

    if training_plan is None and split_manifest is None:
        default_plan = Path(__file__).parent / TRAINING_PLAN_PATH.name
        if default_plan.is_file():
            training_plan = default_plan
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
        evaluate_validation(
            model_path,
            split_name=args.split,
            training_plan_path=training_plan,
            split_manifest_path=split_manifest,
            pose_source=args.pose_source,
            single_user_only=su,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            binary_softmax_threshold=args.binary_softmax_threshold,
            binary_logit_margin=args.binary_logit_margin,
            output_json=Path(args.output_json) if args.output_json else None,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
