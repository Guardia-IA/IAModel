"""
Análisis del CSV generado por logits_extraction.py para elegir umbrales que
reduzcan falsos positivos.

Qué hace:
  1. Estadísticos (percentiles) de las métricas clave por categoría y, sobre todo,
     separando ROBO (is_robo=1) vs NO_ROBO (is_robo=0).
  2. Barrido de umbrales sobre una puntuación (por defecto clase1_logit, el logit
     de la clase robo) con precisión/recall/F1/FP/FN, a nivel de fila (usuario) y
     a nivel de clip (máximo entre usuarios del clip).
  3. Umbrales sugeridos: el mínimo que alcanza una precisión objetivo (maximizando
     recall) y el que maximiza F0.5 (favorece precisión = menos falsos positivos).
  4. Desglose de los falsos positivos por categoría en el umbral sugerido (para
     ver qué acciones confunden al modelo).
  5. (Opcional, --plots) histogramas ROBO vs NO_ROBO si matplotlib está instalado.

Uso:
    python analyze_logits.py --config logits.conf
    python analyze_logits.py --csv /ruta/logits.csv --score clase1_logit --min-precision 0.97
    python analyze_logits.py --csv /ruta/logits.csv --plots --outdir analisis/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Carga del CSV
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analiza el CSV de logits para elegir umbrales.")
    p.add_argument(
        "--config",
        default=str(_THIS_DIR / "logits.conf"),
        help="logits.conf para localizar output_csv si no pasas --csv.",
    )
    p.add_argument("--csv", default="", help="Ruta del CSV (si no, se lee output_csv de la config).")
    p.add_argument(
        "--score",
        default="clase1_logit",
        help="Columna usada como puntuación de robo para el barrido (def: clase1_logit).",
    )
    p.add_argument(
        "--extra-scores",
        nargs="*",
        default=["gap", "prob_robo"],
        help="Otras columnas para las que también imprimir barrido (def: gap prob_robo).",
    )
    p.add_argument(
        "--min-precision",
        type=float,
        default=0.97,
        help="Precisión objetivo para el umbral sugerido (def: 0.97).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Nº de umbrales del barrido (def: 30).",
    )
    p.add_argument("--plots", action="store_true", help="Guardar histogramas (necesita matplotlib).")
    p.add_argument("--outdir", default="", help="Carpeta para los PNG de --plots (def: junto al CSV).")
    return p.parse_args()


def resolve_csv_path(args: argparse.Namespace) -> Path:
    if args.csv:
        return Path(args.csv).expanduser().resolve()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise SystemExit("Pasa --csv o un --config válido con output_csv.")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    out = cfg.get("output_csv")
    if not out:
        raise SystemExit(f"output_csv no está en {cfg_path}; usa --csv.")
    return Path(out).expanduser().resolve()


def load_csv(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"No existe el CSV: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"CSV vacío: {path}")

    cols = reader.fieldnames or []
    data: Dict[str, np.ndarray] = {}
    float_cols = {
        c
        for c in cols
        if c.endswith("_logit") or c.endswith("_softmax") or c.endswith("_sigmoide")
        or c in ("gap", "prob_robo")
    }
    int_cols = {"categoria", "is_robo", "num_usuarios"}
    for c in cols:
        vals = [r.get(c, "") for r in rows]
        if c in float_cols:
            data[c] = np.array([_to_float(v) for v in vals], dtype=np.float64)
        elif c in int_cols:
            data[c] = np.array([_to_int(v) for v in vals], dtype=np.int64)
        else:
            data[c] = np.array(vals, dtype=object)
    return data


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _to_int(v: Any) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return -1


# ---------------------------------------------------------------------------
# Estadísticos
# ---------------------------------------------------------------------------
_PCTS = [0, 5, 25, 50, 75, 90, 95, 99, 100]


def _stats_line(name: str, x: np.ndarray) -> str:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return f"  {name:<16} (sin datos)"
    pct = np.percentile(x, _PCTS)
    pct_str = " ".join(f"p{p}={v:+.3f}" for p, v in zip(_PCTS, pct))
    return f"  {name:<16} n={x.size:<6} mean={x.mean():+.3f} std={x.std():.3f} | {pct_str}"


def print_distributions(data: Dict[str, np.ndarray], score: str, extra: List[str]) -> None:
    metrics = [m for m in dict.fromkeys([score] + list(extra)) if m in data]
    is_robo = data.get("is_robo")

    print("\n================ DISTRIBUCIONES (ROBO vs NO_ROBO) ================")
    if is_robo is None:
        print("[WARN] No hay columna is_robo; muestro global.")
        for m in metrics:
            print(f"[{m}]")
            print(_stats_line("global", data[m]))
        return
    robo_mask = is_robo == 1
    for m in metrics:
        print(f"[{m}]")
        print(_stats_line("NO_ROBO", data[m][~robo_mask]))
        print(_stats_line("ROBO", data[m][robo_mask]))

    print("\n================ POR CATEGORÍA ================")
    cats = sorted(set(int(c) for c in data["categoria"].tolist())) if "categoria" in data else []
    for cat in cats:
        mask = data["categoria"] == cat
        tag = "ROBO" if (is_robo[mask][0] == 1 if mask.any() else False) else "no_robo"
        print(f"[categoría {cat} ({tag}), {int(mask.sum())} filas]")
        for m in metrics:
            print(_stats_line(m, data[m][mask]))


# ---------------------------------------------------------------------------
# Barrido de umbrales
# ---------------------------------------------------------------------------
def confusion(scores: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, float]:
    pred = scores >= thr
    pos = labels == 1
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    tn = int(np.sum(~pred & ~pos))
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec > 0 and rec > 0) else 0.0
    return {"thr": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


def _fbeta(prec: float, rec: float, beta: float) -> float:
    if not (prec > 0 and rec > 0):
        return 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def threshold_grid(scores: np.ndarray, steps: int) -> np.ndarray:
    s = scores[~np.isnan(scores)]
    if s.size == 0:
        return np.array([0.0])
    lo, hi = float(np.min(s)), float(np.max(s))
    if hi <= lo:
        return np.array([lo])
    qs = np.percentile(s, np.linspace(1, 99, steps))
    lin = np.linspace(lo, hi, steps)
    return np.unique(np.concatenate([qs, lin]))


def sweep(scores: np.ndarray, labels: np.ndarray, steps: int, title: str, min_precision: float) -> None:
    valid = ~np.isnan(scores)
    scores, labels = scores[valid], labels[valid]
    grid = threshold_grid(scores, steps)
    rows = [confusion(scores, labels, t) for t in grid]

    print(f"\n---------------- BARRIDO: {title} ----------------")
    print(f"  {'umbral':>10} {'prec':>7} {'recall':>7} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>6}")
    for r in rows:
        print(
            f"  {r['thr']:>10.3f} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>6.3f} "
            f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>5} {r['tn']:>6}"
        )

    # Sugerencia 1: precisión >= objetivo con máximo recall.
    feasible = [r for r in rows if not np.isnan(r["precision"]) and r["precision"] >= min_precision]
    if feasible:
        best = max(feasible, key=lambda r: (r["recall"], -r["thr"]))
        print(
            f"  >> Para precisión >= {min_precision:.2f}: umbral={best['thr']:.3f} "
            f"=> precisión={best['precision']:.3f}, recall={best['recall']:.3f}, "
            f"FP={best['fp']}, FN={best['fn']}"
        )
    else:
        print(f"  >> Ningún umbral alcanza precisión >= {min_precision:.2f} en este barrido.")

    # Sugerencia 2: maximizar F0.5 (penaliza más los falsos positivos).
    best_f05 = max(rows, key=lambda r: _fbeta(r["precision"] if r["precision"] > 0 else 0.0, r["recall"] if r["recall"] > 0 else 0.0, 0.5))
    print(
        f"  >> Máx F0.5 (favorece precisión): umbral={best_f05['thr']:.3f} "
        f"=> precisión={best_f05['precision']:.3f}, recall={best_f05['recall']:.3f}, "
        f"FP={best_f05['fp']}, FN={best_f05['fn']}"
    )


def fp_breakdown_by_category(
    scores: np.ndarray, labels: np.ndarray, cats: np.ndarray, thr: float
) -> None:
    pred = scores >= thr
    fp_mask = pred & (labels == 0)
    if not fp_mask.any():
        print(f"  Sin falsos positivos en umbral {thr:.3f}.")
        return
    print(f"  Falsos positivos en umbral {thr:.3f} por categoría:")
    uniq = sorted(set(int(c) for c in cats[fp_mask].tolist()))
    total = int(fp_mask.sum())
    for cat in uniq:
        n = int(np.sum(fp_mask & (cats == cat)))
        print(f"    categoría {cat}: {n} FP ({100.0 * n / total:.1f}%)")


# ---------------------------------------------------------------------------
# Agregación a nivel de clip
# ---------------------------------------------------------------------------
def aggregate_per_clip(data: Dict[str, np.ndarray], score: str) -> Dict[str, np.ndarray]:
    """Por clip: max(score) entre usuarios, con su is_robo y categoría."""
    clip_paths = data.get("clip_path")
    if clip_paths is None:
        return {}
    scores = data[score]
    is_robo = data.get("is_robo", np.zeros(len(scores), dtype=np.int64))
    cats = data.get("categoria", np.full(len(scores), -1, dtype=np.int64))

    by_clip: Dict[str, Dict[str, float]] = {}
    for i in range(len(scores)):
        cp = str(clip_paths[i])
        s = float(scores[i])
        cur = by_clip.get(cp)
        if cur is None or (not np.isnan(s) and s > cur["score"]):
            by_clip[cp] = {"score": s, "is_robo": int(is_robo[i]), "categoria": int(cats[i])}

    out_scores = np.array([v["score"] for v in by_clip.values()], dtype=np.float64)
    out_robo = np.array([v["is_robo"] for v in by_clip.values()], dtype=np.int64)
    out_cats = np.array([v["categoria"] for v in by_clip.values()], dtype=np.int64)
    return {"score": out_scores, "is_robo": out_robo, "categoria": out_cats}


# ---------------------------------------------------------------------------
# Histogramas opcionales
# ---------------------------------------------------------------------------
def make_plots(data: Dict[str, np.ndarray], metrics: List[str], outdir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib no está instalado; me salto --plots.")
        return
    is_robo = data.get("is_robo")
    outdir.mkdir(parents=True, exist_ok=True)
    for m in metrics:
        if m not in data:
            continue
        x = data[m]
        plt.figure(figsize=(8, 5))
        if is_robo is not None:
            plt.hist(x[is_robo == 0][~np.isnan(x[is_robo == 0])], bins=60, alpha=0.6, label="NO_ROBO", density=True)
            plt.hist(x[is_robo == 1][~np.isnan(x[is_robo == 1])], bins=60, alpha=0.6, label="ROBO", density=True)
            plt.legend()
        else:
            plt.hist(x[~np.isnan(x)], bins=60, alpha=0.8)
        plt.title(f"Distribución de {m}")
        plt.xlabel(m)
        plt.ylabel("densidad")
        out = outdir / f"hist_{m}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(args)
    print(f"[INFO] CSV: {csv_path}")
    data = load_csv(csv_path)
    n = len(next(iter(data.values())))
    print(f"[INFO] Filas: {n} | columnas: {list(data.keys())}")

    if args.score not in data:
        raise SystemExit(f"La columna --score '{args.score}' no está en el CSV. Disponibles: {list(data.keys())}")
    if "is_robo" not in data:
        raise SystemExit("El CSV no tiene columna is_robo; no puedo evaluar precisión/recall.")

    extra = [c for c in args.extra_scores if c in data]
    print_distributions(data, args.score, extra)

    labels = data["is_robo"]
    cats = data.get("categoria", np.full(n, -1, dtype=np.int64))

    # --- Nivel fila (por usuario) ---
    print("\n################ NIVEL FILA (por usuario) ################")
    sweep(data[args.score], labels, args.steps, f"{args.score} (fila)", args.min_precision)
    for c in extra:
        sweep(data[c], labels, args.steps, f"{c} (fila)", args.min_precision)

    # --- Nivel clip (máximo entre usuarios) ---
    clip = aggregate_per_clip(data, args.score)
    if clip:
        print("\n################ NIVEL CLIP (máx entre usuarios) ################")
        print(f"[INFO] Clips: {clip['score'].size} (de {n} filas)")
        sweep(clip["score"], clip["is_robo"], args.steps, f"{args.score} (clip)", args.min_precision)

        # Desglose de FP por categoría en el umbral de precisión objetivo (clip).
        grid = threshold_grid(clip["score"], args.steps)
        feas = [confusion(clip["score"], clip["is_robo"], t) for t in grid]
        feas = [r for r in feas if not np.isnan(r["precision"]) and r["precision"] >= args.min_precision]
        if feas:
            chosen = max(feas, key=lambda r: r["recall"])["thr"]
            print(f"\n[INFO] Desglose FP (clip) en umbral de precisión>= {args.min_precision:.2f}:")
            fp_breakdown_by_category(clip["score"], clip["is_robo"], clip["categoria"], chosen)

    if args.plots:
        outdir = Path(args.outdir).expanduser().resolve() if args.outdir else csv_path.parent / "analisis_logits"
        make_plots(data, [args.score] + extra, outdir)

    print("\n[FIN] Análisis completado.")


if __name__ == "__main__":
    main()
