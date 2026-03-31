import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

# Soportar ejecución como módulo (-m training.evaluate_singleuser) y como script (python training/evaluate_singleuser.py)
try:
    from .model_config import DATA_RESULT_ROOT  # type: ignore[attr-defined]
    from .train_model import PoseExample, PoseDataset  # type: ignore[attr-defined]
except ImportError:
    from model_config import DATA_RESULT_ROOT  # type: ignore[attr-defined]
    from train_model import PoseExample, PoseDataset  # type: ignore[attr-defined]


MIN_TOTAL_FRAMES_SINGLE = 10


def get_data_result_root() -> Path:
    root = DATA_RESULT_ROOT
    if not root.exists():
        raise RuntimeError(f"No se encontró la carpeta data_result en: {root}")
    return root


def collect_singleuser_examples(pose_source: str = "filtered") -> List[PoseExample]:
    """Construye ejemplos SOLO a partir de clips con exactamente un usuario."""
    root = get_data_result_root()
    examples: List[PoseExample] = []

    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_str = cat_dir.name
        for clip_dir in sorted(cat_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            meta_path = clip_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            users = meta.get("users", [])
            if len(users) != 1:
                continue

            u = users[0]
            track_id = u.get("track_id")
            total_frames = u.get("total_frames", 0)
            if track_id is None or total_frames < MIN_TOTAL_FRAMES_SINGLE:
                continue

            user_dir = clip_dir / f"user_{track_id}"
            pose_filename = "poses.npy" if pose_source == "filtered" else "poses_full.npy"
            pose_path = user_dir / pose_filename
            if not pose_path.exists():
                continue

            try:
                poses = np.load(pose_path)
            except Exception:
                continue

            if poses.ndim != 3 or poses.shape[-1] != 2:
                continue

            try:
                clip_cat_int = int(meta.get("cat", cat_str))
            except Exception:
                clip_cat_int = int(cat_str)

            examples.append(
                PoseExample(
                    pose_path=pose_path,
                    label=clip_cat_int,
                    track_id=int(track_id),
                    clip_name=str(meta.get("clip_name", clip_dir.name)),
                    category_str=cat_str,
                )
            )

    if not examples:
        raise RuntimeError("No se encontraron ejemplos single-user válidos en data_result.")
    return examples


@torch.no_grad()
def evaluate_model_on_singleuser(
    model_path: Path,
    pose_source: str = "filtered",
    *,
    balanced_eval: bool = False,
    balanced_ratio: float = 1.0,
    seed: int = 42,
    apply_threshold: float | None = None,
    threshold_sweep: bool = False,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.05,
) -> Dict[str, Any]:
    print(f"\nEvaluando modelo en single-user: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)

    checkpoint = torch.load(model_path, map_location=device)
    model_state = checkpoint["model_state_dict"]
    label_to_idx = checkpoint["label_to_idx"]
    seq_len = checkpoint.get("seq_len", 64)
    task = checkpoint.get("task", "multiclass")
    positive_class = checkpoint.get("positive_class", 6)

    try:
        from .train_model import build_model  # type: ignore[attr-defined]
    except ImportError:
        from train_model import build_model  # type: ignore[attr-defined]

    cfg = checkpoint.get("config", {})
    arch = cfg.get("arch", "tcn")
    input_dim = checkpoint["input_dim"]
    num_classes = int(checkpoint.get("num_classes", len(label_to_idx)))

    model = build_model(arch, input_dim, num_classes, cfg).to(device)
    model.load_state_dict(model_state)
    model.eval()

    print(f"Construyendo dataset single-user (pose_source='{pose_source}')...")
    examples = collect_singleuser_examples(pose_source=pose_source)
    total_single = len(examples)
    print(f"Ejemplos single-user: {total_single}")

    if task == "binary":
        try:
            from .train_model import make_binary_examples  # type: ignore[attr-defined]
        except ImportError:
            from train_model import make_binary_examples  # type: ignore[attr-defined]
        print(f"[BINARIO] Reetiquetando single-user con positive_class={positive_class}")
        examples = make_binary_examples(examples, positive_class=positive_class)

        if balanced_eval:
            pos = [ex for ex in examples if ex.label == 1]
            neg = [ex for ex in examples if ex.label == 0]
            if len(pos) > 0 and len(neg) > 0:
                target_neg = int(len(pos) * balanced_ratio)
                target_neg = max(1, target_neg)
                target_neg = min(target_neg, len(neg))
                neg_sel = random.sample(neg, target_neg) if target_neg < len(neg) else neg
                examples = pos + neg_sel
                random.shuffle(examples)
                print(
                    f"[BALANCED-EVAL] pos={len(pos)} neg_total={len(neg)} "
                    f"neg_kept={target_neg} total_eval={len(examples)}"
                )
            else:
                print(f"[BALANCED-EVAL] pos={len(pos)} neg={len(neg)} => sin balance (pocos datos)")

    examples = [ex for ex in examples if ex.label in label_to_idx]
    filtered_single = len(examples)
    print(f"Ejemplos después de filtrar por clases conocidas: {filtered_single}")

    ds = PoseDataset(examples, label_to_idx, seq_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    total = 0
    correct = 0
    conf_mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    top3_correct = 0
    sample_offset = 0
    y_true_bin: List[int] = []
    y_pos_prob: List[float] = []
    pos_idx_internal = 1 if task == "binary" else None
    false_negative_video_paths: List[str] = []

    for x, y_idx in loader:
        x = x.to(device)
        y_idx = y_idx.to(device)
        logits = model(x)

        if task == "binary" and logits.size(1) >= 2 and pos_idx_internal is not None:
            probs = torch.softmax(logits, dim=1)
            pos_probs = probs[:, pos_idx_internal]
            y_pos_prob.extend(pos_probs.detach().cpu().tolist())
            y_true_bin.extend((y_idx == pos_idx_internal).long().cpu().tolist())
            if apply_threshold is not None:
                preds_idx = (pos_probs >= float(apply_threshold)).long()
            else:
                preds_idx = logits.argmax(dim=1)
        else:
            preds_idx = logits.argmax(dim=1)

        total += y_idx.size(0)
        correct += (preds_idx == y_idx).sum().item()

        for yt, yp in zip(y_idx.tolist(), preds_idx.tolist()):
            if 0 <= yt < num_classes and 0 <= yp < num_classes:
                conf_mat[yt][yp] += 1

        # Falsos negativos (binario): etiquetado robo (1) pero predicho no-robo (0).
        if task == "binary" and pos_idx_internal is not None:
            y_true_list = y_idx.tolist()
            y_pred_list = preds_idx.tolist()
            for j, (yt, yp) in enumerate(zip(y_true_list, y_pred_list)):
                if yt == pos_idx_internal and yp != pos_idx_internal:
                    ex_idx = sample_offset + j
                    if 0 <= ex_idx < len(examples):
                        ex = examples[ex_idx]
                        clip_dir = ex.pose_path.parent.parent
                        clip_video_path = clip_dir / "clip.mp4"
                        full_path = clip_video_path.resolve() if clip_video_path.exists() else clip_dir.resolve()
                        false_negative_video_paths.append(str(full_path))

        sample_offset += y_idx.size(0)

        if logits.size(1) >= 3:
            top3 = logits.topk(3, dim=1).indices
            for yt, topk in zip(y_idx.tolist(), top3.tolist()):
                if yt in topk:
                    top3_correct += 1

    overall_acc = correct / max(total, 1)
    top3_acc = top3_correct / max(total, 1) if total > 0 else 0.0

    per_class: Dict[int, Dict[str, Any]] = {}
    supports = []
    f1s = []
    for c in range(num_classes):
        tp = conf_mat[c][c]
        fn = sum(conf_mat[c][j] for j in range(num_classes)) - tp
        fp = sum(conf_mat[i][c] for i in range(num_classes)) - tp
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(support)}
        supports.append(support)
        f1s.append(f1)

    total_support = sum(supports) if supports else 0
    macro_f1 = float(sum(f1s) / max(len(f1s), 1)) if total_support > 0 else 0.0
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_support) if total_support > 0 else 0.0

    print(f"Accuracy global single-user: {overall_acc:.4f} ({correct}/{total})")
    print(f"Top-3 accuracy single-user: {top3_acc:.4f}")
    print(f"Macro-F1 single-user: {macro_f1:.4f}")
    print(f"Weighted-F1 single-user: {weighted_f1:.4f}")

    if task == "binary":
        print("Métricas por clase (binario 0=no-robo, 1=robo):")
        for idx, stats in sorted(per_class.items()):
            if stats["support"] == 0:
                continue
            print(
                f"  Clase {idx}: "
                f"prec={stats['precision']:.3f}, rec={stats['recall']:.3f}, "
                f"f1={stats['f1']:.3f}, support={stats['support']}"
            )

        pos_stats = per_class.get(1, {})
        neg_stats = per_class.get(0, {})
        c6_prec = float(pos_stats.get("precision", 0.0))
        c6_rec = float(pos_stats.get("recall", 0.0))
        c6_f1 = float(pos_stats.get("f1", 0.0))
        c6_sup = int(pos_stats.get("support", 0))
        n_prec = float(neg_stats.get("precision", 0.0))
        n_rec = float(neg_stats.get("recall", 0.0))
        n_f1 = float(neg_stats.get("f1", 0.0))
        n_sup = int(neg_stats.get("support", 0))

        print(f"Métricas clase positiva (1=robo): prec={c6_prec:.3f}, rec={c6_rec:.3f}, f1={c6_f1:.3f}, support={c6_sup}")
        print(f"Métricas clase negativa (0=no robo): prec={n_prec:.3f}, rec={n_rec:.3f}, f1={n_f1:.3f}, support={n_sup}")
        if false_negative_video_paths:
            print("\n[FALSE-NEGATIVES] Robos etiquetados no detectados (ruta completa):")
            for p in sorted(set(false_negative_video_paths)):
                print(f"  - {p}")
        else:
            print("\n[FALSE-NEGATIVES] No hay robos etiquetados no detectados.")

        best_thr = None
        best_prec = 0.0
        best_rec = 0.0
        best_f1 = 0.0
        if threshold_sweep and y_true_bin and y_pos_prob:
            t_min = max(0.0, min(1.0, float(threshold_min)))
            t_max = max(0.0, min(1.0, float(threshold_max)))
            t_step = max(1e-6, float(threshold_step))
            if t_max < t_min:
                t_min, t_max = t_max, t_min
            thr = t_min
            while thr <= (t_max + 1e-12):
                y_pred_thr = [1 if p >= thr else 0 for p in y_pos_prob]
                tp = sum(1 for yt, yp in zip(y_true_bin, y_pred_thr) if yt == 1 and yp == 1)
                fp = sum(1 for yt, yp in zip(y_true_bin, y_pred_thr) if yt == 0 and yp == 1)
                fn = sum(1 for yt, yp in zip(y_true_bin, y_pred_thr) if yt == 1 and yp == 0)
                prec_t = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec_t = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_t = (2 * prec_t * rec_t / (prec_t + rec_t)) if (prec_t + rec_t) > 0 else 0.0
                if (
                    (f1_t > best_f1)
                    or (abs(f1_t - best_f1) <= 1e-12 and prec_t > best_prec)
                    or (
                        abs(f1_t - best_f1) <= 1e-12
                        and abs(prec_t - best_prec) <= 1e-12
                        and (best_thr is None or thr < best_thr)
                    )
                ):
                    best_thr = thr
                    best_prec = prec_t
                    best_rec = rec_t
                    best_f1 = f1_t
                thr += t_step
            if best_thr is not None:
                print(
                    "[THRESHOLD-SWEEP] "
                    f"mejor umbral={best_thr:.3f} | "
                    f"prec_pos={best_prec:.3f} rec_pos={best_rec:.3f} f1_pos={best_f1:.3f}"
                )
    else:
        inv_map = {v: k for k, v in label_to_idx.items()}
        print("Métricas por clase (label original):")
        for idx, stats in sorted(per_class.items()):
            lab = inv_map.get(idx, idx)
            if stats["support"] == 0:
                continue
            print(
                f"  Clase {lab}: "
                f"prec={stats['precision']:.3f}, rec={stats['recall']:.3f}, "
                f"f1={stats['f1']:.3f}, support={stats['support']}"
            )
        class6_idx = None
        for idx, lab in inv_map.items():
            if lab == 6:
                class6_idx = idx
                break
        if class6_idx is not None and class6_idx in per_class:
            c6_stats = per_class[class6_idx]
            c6_prec = float(c6_stats["precision"])
            c6_rec = float(c6_stats["recall"])
            c6_f1 = float(c6_stats["f1"])
            c6_sup = int(c6_stats["support"])
        else:
            c6_prec = 0.0
            c6_rec = 0.0
            c6_f1 = 0.0
            c6_sup = 0

    result: Dict[str, Any] = {
        "model_path": str(model_path),
        "arch": arch,
        "task": task,
        "total_single": int(total_single),
        "filtered_single": int(filtered_single),
        "accuracy": float(overall_acc),
        "top3_acc": float(top3_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "class6_precision": c6_prec,
        "class6_recall": c6_rec,
        "class6_f1": c6_f1,
        "class6_support": c6_sup,
    }

    if task == "binary":
        result.update(
            {
                "pos_precision": c6_prec,
                "pos_recall": c6_rec,
                "pos_f1": c6_f1,
                "neg_precision": n_prec,
                "neg_recall": n_rec,
                "neg_f1": n_f1,
                "neg_support": n_sup,
                "applied_threshold": (float(apply_threshold) if apply_threshold is not None else None),
                "threshold_sweep_enabled": bool(threshold_sweep),
                "threshold_sweep_min": float(threshold_min),
                "threshold_sweep_max": float(threshold_max),
                "threshold_sweep_step": float(threshold_step),
                "best_threshold": (float(best_thr) if best_thr is not None else None),
                "best_pos_precision": float(best_prec),
                "best_pos_recall": float(best_rec),
                "best_pos_f1": float(best_f1),
            }
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Evalúa modelos en clips de un único usuario.")
    parser.add_argument(
        "--pose-source",
        choices=["filtered", "full"],
        default="filtered",
        help="Fuente de poses: 'filtered' (poses.npy) o 'full' (poses_full.npy).",
    )
    parser.add_argument("--balanced", action="store_true", help="En modo binario, balancea eval recortando la clase no-robo.")
    parser.add_argument("--balanced-ratio", type=float, default=1.0, help="neg_kept = pos_count * ratio (solo si --balanced).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestreo balanceado.")
    parser.add_argument("--apply-threshold", type=float, default=None, help="En binario, umbral fijo de probabilidad para decidir robo (p>=thr => robo).")
    parser.add_argument("--threshold-sweep", action="store_true", help="En binario, barre umbrales de probabilidad y reporta el mejor F1_pos.")
    parser.add_argument("--threshold-min", type=float, default=0.05, help="Umbral mínimo del barrido (0..1).")
    parser.add_argument("--threshold-max", type=float, default=0.95, help="Umbral máximo del barrido (0..1).")
    parser.add_argument("--threshold-step", type=float, default=0.05, help="Paso del barrido de umbral.")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Carpeta de modelos a evaluar. Por defecto usa training/models-single.",
    )
    args = parser.parse_args()

    default_models_dir = Path(__file__).parent / "models-single"
    models_dir = Path(args.models_dir).expanduser().resolve() if args.models_dir else default_models_dir
    model_paths = sorted(models_dir.glob("modelo_*.pt"))
    if not model_paths:
        print(f"No se encontraron modelos modelo_*.pt en {models_dir}")
        return

    results: List[Dict[str, Any]] = []
    for mp in model_paths:
        res = evaluate_model_on_singleuser(
            mp,
            pose_source=args.pose_source,
            balanced_eval=args.balanced,
            balanced_ratio=args.balanced_ratio,
            seed=args.seed,
            apply_threshold=args.apply_threshold,
            threshold_sweep=args.threshold_sweep,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
        results.append(res)

    if not results:
        return

    print("\n" + "=" * 80)
    print("RESUMEN MODELOS EN SINGLE-USER (enfocado en clase positiva / clase 6)")
    print("=" * 80)
    header = (
        f"{'ID':>3} | {'Modelo':>12} | {'Arch':>11} | "
        f"{'N_single':>8} | {'N_usados':>8} | "
        f"{'Acc':>6} | {'MacroF1':>8} | "
        f"{'F1_pos':>7} | {'Rec_pos':>7} | {'Prec_pos':>8} | {'Sup_pos':>7}"
    )
    print(header)
    print("-" * len(header))

    sorted_results = sorted(
        enumerate(results, start=1),
        key=lambda p: (-p[1]["class6_f1"], -p[1]["macro_f1"]),
    )

    for idx, r in sorted_results:
        name = Path(r["model_path"]).name
        pos_prec = r.get("pos_precision", r["class6_precision"])
        print(
            f"{idx:3d} | {name:>12} | {r['arch']:>11} | "
            f"{r['total_single']:8d} | {r['filtered_single']:8d} | "
            f"{r['accuracy']:6.3f} | {r['macro_f1']:8.3f} | "
            f"{r['class6_f1']:7.3f} | {r['class6_recall']:7.3f} | {pos_prec:8.3f} | {r['class6_support']:7d}"
        )


if __name__ == "__main__":
    main()
