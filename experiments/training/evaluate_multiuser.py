import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

# Soportar ejecución como módulo (-m training.evaluate_multiuser) y como script (python training/evaluate_multiuser.py)
try:
    from .model_config import DATA_RESULT_ROOT  # type: ignore[attr-defined]
    from .train_model import PoseExample, PoseDataset  # type: ignore[attr-defined]
except ImportError:
    from model_config import DATA_RESULT_ROOT  # type: ignore[attr-defined]
    from train_model import PoseExample, PoseDataset  # type: ignore[attr-defined]


# Umbral mínimo de frames totales por usuario para considerar el track en test multiusuario
MIN_TOTAL_FRAMES_MULTI = 10


def get_data_result_root() -> Path:
    root = DATA_RESULT_ROOT
    if not root.exists():
        raise RuntimeError(f"No se encontró la carpeta data_result en: {root}")
    return root


def collect_multiuser_examples(pose_source: str = "filtered") -> List[PoseExample]:
    """
    Construye un conjunto de ejemplos SOLO a partir de clips con más de un usuario en meta["users"].
    Para cada clip multiusuario, se añaden los usuarios cuyo total_frames >= MIN_TOTAL_FRAMES_MULTI.
    """
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
            # Solo nos interesan clips con más de un usuario anotado
            if len(users) <= 1:
                continue

            for u in users:
                track_id = u.get("track_id")
                total_frames = u.get("total_frames", 0)
                if track_id is None or total_frames < MIN_TOTAL_FRAMES_MULTI:
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

                examples.append(
                    PoseExample(
                        pose_path=pose_path,
                        label=int(meta.get("cat", cat_str)),
                        track_id=int(track_id),
                        clip_name=str(meta.get("clip_name", clip_dir.name)),
                        category_str=cat_str,
                    )
                )

    if not examples:
        raise RuntimeError("No se encontraron ejemplos multiusuario válidos en data_result.")
    return examples


@torch.no_grad()
def evaluate_model_on_multiuser(model_path: Path, pose_source: str = "filtered") -> Dict[str, Any]:
    print(f"\nEvaluando modelo en multiusuario: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    model_state = checkpoint["model_state_dict"]
    label_to_idx = checkpoint["label_to_idx"]
    seq_len = checkpoint.get("seq_len", 64)
    task = checkpoint.get("task", "multiclass")
    positive_class = checkpoint.get("positive_class", 6)

    # Reconstruir modelo según config guardada
    try:
        # Cuando se ejecuta como módulo: python -m training.evaluate_multiuser
        from .train_model import build_model  # type: ignore[attr-defined]
    except ImportError:
        # Cuando se ejecuta como script: python training/evaluate_multiuser.py
        from train_model import build_model  # type: ignore[attr-defined]

    cfg = checkpoint.get("config", {})
    arch = cfg.get("arch", "tcn")
    input_dim = checkpoint["input_dim"]
    # Para compatibilidad: si viene num_classes en el checkpoint, lo usamos; si no, inferimos.
    num_classes = int(checkpoint.get("num_classes", len(label_to_idx)))

    model = build_model(arch, input_dim, num_classes, cfg).to(device)
    model.load_state_dict(model_state)
    model.eval()

    print(f"Construyendo dataset multiusuario (pose_source='{pose_source}')...")
    examples = collect_multiuser_examples(pose_source=pose_source)
    total_multi = len(examples)
    print(f"Ejemplos multiusuario: {total_multi}")

    # Si el modelo es binario, reetiquetamos a 0/1 igual que en train_model
    if task == "binary":
        try:
            # Ejecución como módulo: python -m training.evaluate_multiuser
            from .train_model import make_binary_examples  # type: ignore[attr-defined]
        except ImportError:
            # Ejecución como script: python training/evaluate_multiuser.py
            from train_model import make_binary_examples  # type: ignore[attr-defined]
        print(f"[BINARIO] Reetiquetando multiusuario con positive_class={positive_class}")
        examples = make_binary_examples(examples, positive_class=positive_class)

    # Filtrar ejemplos a solo labels que el modelo conoce
    examples = [ex for ex in examples if ex.label in label_to_idx]
    filtered_multi = len(examples)
    print(f"Ejemplos después de filtrar por clases conocidas: {filtered_multi}")

    # Dataset y loader
    ds = PoseDataset(examples, label_to_idx, seq_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    num_classes = len(label_to_idx)

    total = 0
    correct = 0
    conf_mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    top3_correct = 0

    for x, y_idx in loader:
        x = x.to(device)
        y_idx = y_idx.to(device)
        logits = model(x)
        preds_idx = logits.argmax(dim=1)

        total += y_idx.size(0)
        correct += (preds_idx == y_idx).sum().item()

        # Matriz de confusión en espacio de índices internos
        for yt, yp in zip(y_idx.tolist(), preds_idx.tolist()):
            if 0 <= yt < num_classes and 0 <= yp < num_classes:
                conf_mat[yt][yp] += 1

        # Top-3 accuracy
        if logits.size(1) >= 3:
            top3 = logits.topk(3, dim=1).indices
            for yt, topk in zip(y_idx.tolist(), top3.tolist()):
                if yt in topk:
                    top3_correct += 1

    overall_acc = correct / max(total, 1)
    top3_acc = top3_correct / max(total, 1) if total > 0 else 0.0

    # Métricas por clase en espacio de índices
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
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        per_class[c] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(support),
        }

        supports.append(support)
        f1s.append(f1)

    total_support = sum(supports) if supports else 0
    if total_support > 0:
        macro_f1 = float(sum(f1s) / max(len(f1s), 1))
        weighted_f1 = float(
            sum(f * s for f, s in zip(f1s, supports)) / total_support
        )
    else:
        macro_f1 = 0.0
        weighted_f1 = 0.0

    print(f"Accuracy global multiusuario: {overall_acc:.4f} ({correct}/{total})")
    print(f"Top-3 accuracy multiusuario: {top3_acc:.4f}")
    print(f"Macro-F1 multiusuario: {macro_f1:.4f}")
    print(f"Weighted-F1 multiusuario: {weighted_f1:.4f}")

    # Mostrar métricas por clase
    if task == "binary":
        # En binario: índice 1 = clase positiva (robos)
        print("Métricas por clase (binario 0=no-robo, 1=robo):")
        for idx, stats in sorted(per_class.items()):
            if stats["support"] == 0:
                continue
            print(
                f"  Clase {idx}: "
                f"prec={stats['precision']:.3f}, "
                f"rec={stats['recall']:.3f}, "
                f"f1={stats['f1']:.3f}, "
                f"support={stats['support']}"
            )

        # Métricas específicas para la clase positiva (1)
        pos_idx = 1
        if pos_idx in per_class:
            c_stats = per_class[pos_idx]
            c_prec = float(c_stats["precision"])
            c_rec = float(c_stats["recall"])
            c_f1 = float(c_stats["f1"])
            c_sup = int(c_stats["support"])
        else:
            c_prec = 0.0
            c_rec = 0.0
            c_f1 = 0.0
            c_sup = 0

        # Métricas de la clase negativa (0)
        neg_idx = 0
        if neg_idx in per_class:
            n_stats = per_class[neg_idx]
            n_prec = float(n_stats["precision"])
            n_rec = float(n_stats["recall"])
            n_f1 = float(n_stats["f1"])
            n_sup = int(n_stats["support"])
        else:
            n_prec = 0.0
            n_rec = 0.0
            n_f1 = 0.0
            n_sup = 0

        print(
            f"Métricas clase positiva (1=robo): "
            f"prec={c_prec:.3f}, rec={c_rec:.3f}, f1={c_f1:.3f}, support={c_sup}"
        )
        print(
            f"Métricas clase negativa (0=no robo): "
            f"prec={n_prec:.3f}, rec={n_rec:.3f}, f1={n_f1:.3f}, support={n_sup}"
        )

        c6_prec = c_prec
        c6_rec = c_rec
        c6_f1 = c_f1
        c6_sup = c_sup
    else:
        # Multiclase: mapeamos de índice interno a label original
        inv_map = {v: k for k, v in label_to_idx.items()}
        print("Métricas por clase (label original):")
        for idx, stats in sorted(per_class.items()):
            lab = inv_map.get(idx, idx)
            if stats["support"] == 0:
                continue
            print(
                f"  Clase {lab}: "
                f"prec={stats['precision']:.3f}, "
                f"rec={stats['recall']:.3f}, "
                f"f1={stats['f1']:.3f}, "
                f"support={stats['support']}"
            )

        # Métricas específicas para la clase 6 (robos)
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

        print(
            f"Métricas clase 6 (robos): "
            f"prec={c6_prec:.3f}, rec={c6_rec:.3f}, f1={c6_f1:.3f}, support={c6_sup}"
        )

    result: Dict[str, Any] = {
        "model_path": str(model_path),
        "arch": arch,
        "task": task,
        "total_multi": int(total_multi),
        "filtered_multi": int(filtered_multi),
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
                "neg_precision": n_prec,
                "neg_recall": n_rec,
                "neg_f1": n_f1,
                "neg_support": n_sup,
            }
        )

    return result


def main():
    models_dir = Path(__file__).parent / "models"
    model_paths = sorted(models_dir.glob("modelo_*.pt"))
    if not model_paths:
        print(f"No se encontraron modelos modelo_*.pt en {models_dir}")
        return

    # Evalúa todos los modelos en multiusuario con poses filtradas
    results: List[Dict[str, Any]] = []
    for mp in model_paths:
        res = evaluate_model_on_multiuser(mp, pose_source="filtered")
        results.append(res)

    if not results:
        return

    # Resumen en tabla enfocado en la clase 6 (robos)
    print("\n" + "=" * 80)
    print("RESUMEN MODELOS EN MULTIUSUARIO (enfocado en clase positiva / clase 6)")
    print("=" * 80)
    header = (
        f"{'ID':>3} | {'Modelo':>12} | {'Arch':>11} | "
        f"{'N_multi':>8} | {'N_usados':>8} | "
        f"{'Acc':>6} | {'MacroF1':>8} | "
        f"{'F1_pos':>7} | {'Rec_pos':>7} | {'Sup_pos':>7}"
    )
    print(header)
    print("-" * len(header))

    # Ordenar por F1 de clase positiva/robos descendente y luego por macro_f1
    sorted_results = sorted(
        enumerate(results, start=1),
        key=lambda p: (-p[1]["class6_f1"], -p[1]["macro_f1"]),
    )

    for idx, r in sorted_results:
        name = Path(r["model_path"]).name
        print(
            f"{idx:3d} | {name:>12} | {r['arch']:>11} | "
            f"{r['total_multi']:8d} | {r['filtered_multi']:8d} | "
            f"{r['accuracy']:6.3f} | {r['macro_f1']:8.3f} | "
            f"{r['class6_f1']:7.3f} | {r['class6_recall']:7.3f} | {r['class6_support']:7d}"
        )


if __name__ == "__main__":
    main()

