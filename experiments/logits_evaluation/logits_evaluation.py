"""
Evalúa varios CSV de logits (salida de batch_logits_dir_runner.py) y dice qué
modelo acierta más y cuál tiene más falsos positivos.

Cada CSV representa una CATEGORÍA de clips:
    - Uno (o varios) marcados como "robo": se esperan logits POSITIVOS (clase[1]).
    - El resto, "no robo": se esperan logits NEGATIVOS.

Formato de cada CSV: cabecera con columnas de identificación (clip, usuario,
video, muestra...) y una columna por modelo; cada celda es el logit de clase[1].

Criterio (umbral T, por defecto 0.0):
    - Predicción ROBO  si logit  > T
    - Predicción NO-ROBO si logit <= T
    - Acierto: robo con logit>T  (TP)  ó  no-robo con logit<=T (TN)
    - Falso positivo (FP): no-robo con logit>T
    - Falso negativo (FN): robo con logit<=T

Entradas:
    1) Config JSON (--config):
        {
          "threshold": 0.0,
          "csvs": [
            { "path": "/ruta/robos.csv",   "robo": true },
            { "path": "/ruta/compras.csv", "robo": false },
            { "path": "/ruta/paseos.csv",  "robo": false }
          ]
        }
       (También se acepta una lista directa de objetos {path, robo}.)
    2) Por línea de comandos:
        --csvs a.csv b.csv c.csv --robo-csv a.csv [--threshold 0.0]

Salida:
    - Tabla por modelo en consola (aciertos, FP, % acierto, recall...).
    - Modelo con más aciertos y modelo con más falsos positivos.
    - Umbral óptimo por modelo (maximiza acierto).
    - Opcional --output resumen.csv | resumen.json

Análisis extra para reducir falsos positivos (todos opcionales):
    --target-fpr X        umbral por modelo con FPR <= X%% (maximiza recall).
    --target-precision X  umbral por modelo con precisión >= X%% (maximiza recall).
    --fp-breakdown        FP por CSV (categoría) y modelo: ve qué categoría los causa.
    --consensus M1 M2 ... consenso AND/OR entre modelos (cada uno con su umbral).
    --consensus-search    prueba TODAS las combinaciones (--consensus-size, def 2) y
                          elige la mejor según --consensus-rank (accuracy/f1/precision/recall/fp).
    --abstain-model M --abstain-low L --abstain-high H
                          zona de abstención: <L no-robo, >H robo, en medio "incierto"
                          (candidatos a revisar, p. ej. con un VLM en cascada).

Uso:
    python logits_evaluation.py --config logits_evaluation_config.json
    python logits_evaluation.py --csvs robos.csv compras.csv --robo-csv robos.csv --threshold 0.5
    python logits_evaluation.py --config cfg.json --output resumen.csv
    python logits_evaluation.py --config cfg.json --target-fpr 1.0 --fp-breakdown
    python logits_evaluation.py --config cfg.json --consensus modelo_54 modelo_25
    python logits_evaluation.py --config cfg.json --abstain-model modelo_54 --abstain-low 0 --abstain-high 2.3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Columnas que NO son modelos (identificación de la muestra).
ID_COLUMNS = {"clip", "usuario", "user", "video", "muestra", "sample", "name", "nombre"}


def _to_float(value: str) -> Optional[float]:
    """Convierte una celda a float; None si está vacía o no es numérica (p. ej. 'ERROR:...')."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_csv_specs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], float]:
    """
    Devuelve (lista de {path, robo}, threshold) combinando config y/o CLI.
    """
    specs: List[Dict[str, Any]] = []
    threshold = 0.0

    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        if not cfg_path.is_file():
            raise SystemExit(f"No existe el config: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            threshold = float(data.get("threshold", 0.0))
            robo_csv = data.get("robo_csv")
            entries = data.get("csvs", [])
        else:
            entries = data
            robo_csv = None
        base = cfg_path.parent
        for e in entries:
            if isinstance(e, str):
                path = e
                robo = (robo_csv is not None and Path(e).name == Path(robo_csv).name)
            elif isinstance(e, dict):
                path = str(e.get("path", e.get("csv", "")))
                robo = bool(e.get("robo", e.get("robbery", False)))
            else:
                raise SystemExit("Cada entrada de 'csvs' debe ser una ruta o un objeto {path, robo}.")
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = (base / p)
            specs.append({"path": p.resolve(), "robo": robo})

    if args.csvs:
        robo_name = Path(args.robo_csv).name if args.robo_csv else None
        for c in args.csvs:
            p = Path(c).expanduser().resolve()
            specs.append({"path": p, "robo": (robo_name is not None and p.name == robo_name)})

    if args.threshold is not None:
        threshold = float(args.threshold)

    return specs, threshold


def read_logits_csv(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Lee un CSV de logits. Devuelve (nombres_modelo, filas).
    Cada fila es {"_id": <identificador>, <modelo>: float|None, ...}.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV vacío o sin cabecera: {path}")
        headers = list(reader.fieldnames)
        model_cols = [h for h in headers if h not in ID_COLUMNS]
        id_cols = [h for h in headers if h in ID_COLUMNS]
        rows: List[Dict[str, Any]] = []
        for raw in reader:
            ident = "/".join(str(raw.get(c, "")) for c in id_cols) or f"row{len(rows)}"
            rec: Dict[str, Any] = {"_id": ident}
            for m in model_cols:
                rec[m] = _to_float(raw.get(m))
            rows.append(rec)
    return model_cols, rows


def _counts_to_metrics(t: float, tp: int, tn: int, fp: int, fn: int) -> Dict[str, Any]:
    total = tp + tn + fp + fn
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold": t, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy_pct": (tp + tn) / total * 100.0 if total else 0.0,
        "recall_pct": recall * 100.0,
        "precision_pct": precision * 100.0,
        "f1_pct": f1 * 100.0,
        "fpr_pct": fp / (fp + tn) * 100.0 if (fp + tn) else 0.0,
    }


def _build_thresholds(
    report: Dict[str, Any], models_sel: List[str], threshold_mode: str
) -> Dict[str, float]:
    """Umbral por modelo para el consenso: el óptimo de cada uno, o el fijo del report."""
    if threshold_mode == "fixed":
        return {m: report["threshold"] for m in models_sel}
    out: Dict[str, float] = {}
    for m in models_sel:
        opt = report["stats"][m].get("optimo")
        out[m] = opt["threshold"] if opt else report["threshold"]
    return out


def threshold_sweep(pairs: List[Tuple[float, bool]]) -> List[Dict[str, Any]]:
    """
    Barrido O(n log n) de todos los umbrales relevantes (robo si valor > T).
    Devuelve una lista de métricas (umbral asc): accuracy, recall, precision, fpr...
    """
    if not pairs:
        return []
    total = len(pairs)
    n_robo = sum(1 for _, r in pairs if r)
    n_norobo = total - n_robo
    sp = sorted(pairs, key=lambda x: x[0], reverse=True)

    out: List[Dict[str, Any]] = []
    out.append(_counts_to_metrics(sp[0][0] + 1.0, 0, n_norobo, 0, n_robo))  # todo NO-ROBO

    tp = fp = 0
    i = 0
    while i < total:
        v = sp[i][0]
        j = i
        while j < total and sp[j][0] == v:
            if sp[j][1]:
                tp += 1
            else:
                fp += 1
            j += 1
        next_v = sp[j][0] if j < total else (v - 1.0)
        t = (v + next_v) / 2.0
        out.append(_counts_to_metrics(t, tp, n_norobo - fp, fp, n_robo - tp))
        i = j

    out.sort(key=lambda d: d["threshold"])
    return out


def optimal_threshold(pairs: List[Tuple[float, bool]]) -> Optional[Dict[str, Any]]:
    """Umbral que maximiza el % de acierto. Empates: menos FP y umbral central (máx. margen)."""
    sweep = threshold_sweep(pairs)
    if not sweep:
        return None
    best_acc = max(d["accuracy_pct"] for d in sweep)
    cand = [d for d in sweep if d["accuracy_pct"] == best_acc]
    min_fp = min(d["FP"] for d in cand)
    cand = sorted((d for d in cand if d["FP"] == min_fp), key=lambda d: d["threshold"])
    return dict(cand[len(cand) // 2])


def threshold_for_fpr(pairs: List[Tuple[float, bool]], target_fpr_pct: float) -> Optional[Dict[str, Any]]:
    """Menor umbral con FPR <= objetivo (maximiza recall sin pasarse de falsos positivos)."""
    sweep = threshold_sweep(pairs)
    feasible = [d for d in sweep if d["fpr_pct"] <= target_fpr_pct + 1e-9]
    if not feasible:
        return None
    return dict(max(feasible, key=lambda d: d["recall_pct"]))


def threshold_for_precision(pairs: List[Tuple[float, bool]], target_prec_pct: float) -> Optional[Dict[str, Any]]:
    """Umbral con precisión >= objetivo que maximiza recall (debe predecir algún positivo)."""
    sweep = threshold_sweep(pairs)
    feasible = [d for d in sweep if (d["TP"] + d["FP"]) > 0 and d["precision_pct"] >= target_prec_pct - 1e-9]
    if not feasible:
        return None
    return dict(max(feasible, key=lambda d: d["recall_pct"]))


def evaluate(specs: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """Calcula métricas por modelo agregando todas las muestras de todos los CSV."""
    # Carga de todos los CSV y unión de modelos.
    loaded: List[Tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]] = []
    model_order: List[str] = []
    total_rows = 0
    print(f"[INFO] Leyendo {len(specs)} CSV...")
    for idx, spec in enumerate(specs, start=1):
        cols, rows = read_logits_csv(spec["path"])
        total_rows += len(rows)
        for c in cols:
            if c not in model_order:
                model_order.append(c)
        loaded.append((spec, cols, rows))
        tag = "ROBO" if spec["robo"] else "NO-ROBO"
        print(f"[INFO]   ({idx}/{len(specs)}) [{tag}] {spec['path'].name}: "
              f"{len(rows)} filas, {len(cols)} modelos")
    print(f"[INFO] Total: {total_rows} filas | {len(model_order)} modelos detectados.")

    # Inicializa contadores y pares (logit, is_robo) por modelo.
    stats: Dict[str, Dict[str, int]] = {
        m: {"TP": 0, "FN": 0, "TN": 0, "FP": 0, "NA": 0} for m in model_order
    }
    pairs_by_model: Dict[str, List[Tuple[float, bool]]] = {m: [] for m in model_order}

    print("[INFO] Agregando muestras por modelo...")
    for spec, cols, rows in loaded:
        is_robo = bool(spec["robo"])
        for row in rows:
            for m in model_order:
                val = row.get(m)
                if val is None:
                    stats[m]["NA"] += 1
                    continue
                pairs_by_model[m].append((val, is_robo))
                pred_robo = val > threshold
                if is_robo:
                    stats[m]["TP" if pred_robo else "FN"] += 1
                else:
                    stats[m]["FP" if pred_robo else "TN"] += 1

    # Deriva métricas.
    print(f"[INFO] Calculando métricas y umbral óptimo de {len(model_order)} modelos...")
    results: Dict[str, Any] = {}
    for mi, m in enumerate(model_order, start=1):
        print(f"[INFO]   ({mi}/{len(model_order)}) {m}...")
        s = stats[m]
        tp, fn, tn, fp, na = s["TP"], s["FN"], s["TN"], s["FP"], s["NA"]
        evaluables = tp + fn + tn + fp
        aciertos = tp + tn
        acc = (aciertos / evaluables * 100.0) if evaluables else 0.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) else 0.0
        especificidad = (tn / (tn + fp) * 100.0) if (tn + fp) else 0.0
        precision = (tp / (tp + fp) * 100.0) if (tp + fp) else 0.0
        results[m] = {
            "TP": tp, "FN": fn, "TN": tn, "FP": fp, "NA": na,
            "evaluables": evaluables, "aciertos": aciertos,
            "accuracy_pct": acc, "recall_pct": recall,
            "especificidad_pct": especificidad, "precision_pct": precision,
            "optimo": optimal_threshold(pairs_by_model[m]),
        }

    return {
        "threshold": threshold,
        "models": model_order,
        "stats": results,
        "pairs_by_model": pairs_by_model,
        "loaded": loaded,
    }


def print_report(report: Dict[str, Any], specs: List[Dict[str, Any]]) -> None:
    thr = report["threshold"]
    models = report["models"]
    stats = report["stats"]

    n_robo = sum(1 for s in specs if s["robo"])
    print("=" * 78)
    print(f"Umbral (logit > T => ROBO): T = {thr}")
    print(f"CSVs: {len(specs)} | robo: {n_robo} | no-robo: {len(specs) - n_robo}")
    for s in specs:
        tag = "ROBO   " if s["robo"] else "NO-ROBO"
        print(f"  [{tag}] {s['path']}")
    print("=" * 78)

    header = f"{'modelo':<16} {'acierto%':>9} {'aciertos':>9} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} {'recall%':>8} {'NA':>4}"
    print(header)
    print("-" * len(header))
    for m in models:
        s = stats[m]
        print(
            f"{m:<16} {s['accuracy_pct']:>8.1f}% {s['aciertos']:>9} "
            f"{s['TP']:>5} {s['TN']:>5} {s['FP']:>5} {s['FN']:>5} {s['recall_pct']:>7.1f}% {s['NA']:>4}"
        )
    print("-" * len(header))

    if not models:
        print("No hay columnas de modelo en los CSV.")
        return

    best_acc = max(models, key=lambda m: (stats[m]["accuracy_pct"], -stats[m]["FP"]))
    most_fp = max(models, key=lambda m: stats[m]["FP"])
    least_fp = min(models, key=lambda m: (stats[m]["FP"], -stats[m]["accuracy_pct"]))

    print()
    print(f">> Modelo con MÁS aciertos : {best_acc} "
          f"({stats[best_acc]['accuracy_pct']:.1f}% de acierto, "
          f"{stats[best_acc]['aciertos']}/{stats[best_acc]['evaluables']})")
    print(f">> Modelo con MÁS falsos positivos : {most_fp} (FP={stats[most_fp]['FP']})")
    print(f">> Modelo con MENOS falsos positivos: {least_fp} (FP={stats[least_fp]['FP']})")

    # --- Umbral óptimo por modelo (el que maximiza el acierto) ---
    print()
    print("Umbral óptimo por modelo (maximiza % de acierto):")
    h2 = f"{'modelo':<16} {'umbral_opt':>11} {'acierto_opt%':>13} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}"
    print(h2)
    print("-" * len(h2))
    best_opt_model = None
    best_opt_acc = -1.0
    for m in models:
        opt = stats[m].get("optimo")
        if not opt:
            print(f"{m:<16} {'(sin datos)':>11}")
            continue
        print(
            f"{m:<16} {opt['threshold']:>11.4f} {opt['accuracy_pct']:>12.1f}% "
            f"{opt['TP']:>5} {opt['TN']:>5} {opt['FP']:>5} {opt['FN']:>5}"
        )
        if opt["accuracy_pct"] > best_opt_acc:
            best_opt_acc = opt["accuracy_pct"]
            best_opt_model = m
    print("-" * len(h2))
    if best_opt_model is not None:
        opt = stats[best_opt_model]["optimo"]
        print(
            f">> Mejor con su umbral óptimo: {best_opt_model} "
            f"(umbral={opt['threshold']:.4f}, acierto={opt['accuracy_pct']:.1f}%, FP={opt['FP']})"
        )
    print("=" * 78)


def print_target_thresholds(
    report: Dict[str, Any],
    target_fpr: Optional[float],
    target_prec: Optional[float],
) -> None:
    """Umbral por modelo que cumple un FPR máximo o una precisión mínima, maximizando recall."""
    models = report["models"]
    pbm = report["pairs_by_model"]

    if target_fpr is not None:
        print()
        print(f"Umbral por FPR objetivo (FPR <= {target_fpr:.2f}%), maximizando recall:")
        h = (f"{'modelo':<16} {'umbral':>10} {'recall%':>8} {'FPR%':>7} "
             f"{'prec%':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
        print(h)
        print("-" * len(h))
        for m in models:
            d = threshold_for_fpr(pbm[m], target_fpr)
            if d is None:
                print(f"{m:<16} {'(imposible)':>10}")
                continue
            print(f"{m:<16} {d['threshold']:>10.4f} {d['recall_pct']:>7.1f}% "
                  f"{d['fpr_pct']:>6.2f}% {d['precision_pct']:>6.1f}% "
                  f"{d['TP']:>6} {d['FP']:>6} {d['FN']:>6}")
        print("-" * len(h))

    if target_prec is not None:
        print()
        print(f"Umbral por precisión objetivo (precisión >= {target_prec:.2f}%), maximizando recall:")
        h = (f"{'modelo':<16} {'umbral':>10} {'recall%':>8} {'prec%':>7} "
             f"{'FPR%':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
        print(h)
        print("-" * len(h))
        for m in models:
            d = threshold_for_precision(pbm[m], target_prec)
            if d is None:
                print(f"{m:<16} {'(imposible)':>10}")
                continue
            print(f"{m:<16} {d['threshold']:>10.4f} {d['recall_pct']:>7.1f}% "
                  f"{d['precision_pct']:>6.1f}% {d['fpr_pct']:>6.2f}% "
                  f"{d['TP']:>6} {d['FP']:>6} {d['FN']:>6}")
        print("-" * len(h))


def print_fp_breakdown(report: Dict[str, Any]) -> None:
    """Falsos positivos por CSV (categoría) y modelo, al umbral fijo del report."""
    models = report["models"]
    loaded = report["loaded"]
    thr = report["threshold"]
    norobo = [(spec, rows) for spec, cols, rows in loaded if not spec["robo"]]
    if not norobo:
        return
    print()
    print(f"Desglose de FALSOS POSITIVOS por CSV (no-robo) al umbral T={thr}:")
    h = f"{'csv':<22} {'neg':>6} " + " ".join(f"{m:>10}" for m in models)
    print(h)
    print("-" * len(h))
    totals = {m: 0 for m in models}
    for spec, rows in norobo:
        name = spec["path"].name
        neg = 0
        cells = {m: 0 for m in models}
        for row in rows:
            counted = False
            for m in models:
                v = row.get(m)
                if v is None:
                    continue
                counted = True
                if v > thr:
                    cells[m] += 1
                    totals[m] += 1
            if counted:
                neg += 1
        print(f"{name:<22} {neg:>6} " + " ".join(f"{cells[m]:>10}" for m in models))
    print("-" * len(h))
    print(f"{'TOTAL FP':<22} {'':>6} " + " ".join(f"{totals[m]:>10}" for m in models))
    print("-" * len(h))


def consensus_eval(
    report: Dict[str, Any],
    models_sel: List[str],
    thresholds: Dict[str, float],
    mode: str,
) -> Dict[str, Any]:
    """Combina varios modelos por AND/OR (cada uno con su umbral) sobre cada muestra."""
    loaded = report["loaded"]
    tp = tn = fp = fn = skipped = 0
    for spec, cols, rows in loaded:
        is_robo = bool(spec["robo"])
        for row in rows:
            vals = [row.get(m) for m in models_sel]
            if any(v is None for v in vals):
                skipped += 1
                continue
            preds = [v > thresholds[m] for v, m in zip(vals, models_sel)]
            pred = all(preds) if mode == "and" else any(preds)
            if is_robo:
                tp += pred
                fn += not pred
            else:
                fp += pred
                tn += not pred
    out = _counts_to_metrics(0.0, tp, tn, fp, fn)
    out["skipped"] = skipped
    out["mode"] = mode
    return out


def print_consensus(
    report: Dict[str, Any],
    models_sel: List[str],
    modes: List[str],
    threshold_mode: str,
) -> None:
    models = report["models"]
    sel = [m for m in models_sel if m in models]
    missing = [m for m in models_sel if m not in models]
    if missing:
        print(f"[WARN] Modelos de consenso no encontrados (se ignoran): {missing}")
    if len(sel) < 2:
        print("[WARN] El consenso necesita al menos 2 modelos válidos. Se omite.")
        return

    thresholds = _build_thresholds(report, sel, threshold_mode)

    print()
    print(f"Consenso entre modelos {sel} | umbrales={threshold_mode}:")
    for m in sel:
        print(f"    {m}: umbral={thresholds[m]:.4f}")
    h = (f"{'modo':<6} {'acierto%':>9} {'recall%':>8} {'prec%':>7} {'FPR%':>7} "
         f"{'TP':>6} {'TN':>7} {'FP':>6} {'FN':>6} {'skip':>5}")
    print(h)
    print("-" * len(h))
    for mode in modes:
        d = consensus_eval(report, sel, thresholds, mode)
        label = "AND" if mode == "and" else "OR"
        print(f"{label:<6} {d['accuracy_pct']:>8.1f}% {d['recall_pct']:>7.1f}% "
              f"{d['precision_pct']:>6.1f}% {d['fpr_pct']:>6.2f}% "
              f"{d['TP']:>6} {d['TN']:>7} {d['FP']:>6} {d['FN']:>6} {d['skipped']:>5}")
    print("-" * len(h))


def _rank_key(d: Dict[str, Any], rank_by: str):
    """Clave de ordenación (menor = mejor) según la métrica elegida."""
    if rank_by == "accuracy":
        return (-d["accuracy_pct"], d["FP"])
    if rank_by == "f1":
        return (-d["f1_pct"], d["FP"])
    if rank_by == "precision":
        return (-d["precision_pct"], -d["recall_pct"])
    if rank_by == "recall":
        return (-d["recall_pct"], d["FP"])
    if rank_by == "fp":
        return (d["FP"], -d["recall_pct"])
    return (-d["accuracy_pct"], d["FP"])


def print_consensus_search(
    report: Dict[str, Any],
    size: int,
    modes: List[str],
    threshold_mode: str,
    rank_by: str,
    top_k: int,
) -> None:
    """Prueba TODAS las combinaciones de 'size' modelos, evalúa el consenso y las ordena."""
    import itertools

    models = report["models"]
    if len(models) < size:
        print(f"[WARN] Se piden combinaciones de {size} pero solo hay {len(models)} modelos.")
        return

    thr_all = _build_thresholds(report, models, threshold_mode)

    rows: List[Dict[str, Any]] = []
    for combo in itertools.combinations(models, size):
        sub_thr = {m: thr_all[m] for m in combo}
        for mode in modes:
            d = consensus_eval(report, list(combo), sub_thr, mode)
            d["combo"] = combo
            d["mode_label"] = "AND" if mode == "and" else "OR"
            rows.append(d)

    rows.sort(key=lambda d: _rank_key(d, rank_by))

    print()
    print(f"Búsqueda de consenso: combinaciones de {size} modelos | umbrales={threshold_mode} | "
          f"ordenado por '{rank_by}' (mejor arriba). Top {top_k}:")
    combo_w = max(28, 2 + sum(len(m) + 1 for m in models[:size]))
    h = (f"{'combinacion':<{combo_w}} {'modo':<4} {'acierto%':>9} {'recall%':>8} "
         f"{'prec%':>7} {'F1%':>7} {'FPR%':>7} {'FP':>6} {'FN':>6}")
    print(h)
    print("-" * len(h))
    for d in rows[:top_k]:
        combo_str = "+".join(d["combo"])
        print(f"{combo_str:<{combo_w}} {d['mode_label']:<4} {d['accuracy_pct']:>8.1f}% "
              f"{d['recall_pct']:>7.1f}% {d['precision_pct']:>6.1f}% {d['f1_pct']:>6.1f}% "
              f"{d['fpr_pct']:>6.2f}% {d['FP']:>6} {d['FN']:>6}")
    print("-" * len(h))

    best = rows[0]
    print(f">> Mejor combinación ({rank_by}): {'+'.join(best['combo'])} [{best['mode_label']}] "
          f"-> acierto={best['accuracy_pct']:.1f}%, recall={best['recall_pct']:.1f}%, FP={best['FP']}")
    print(f"   Reprodúcela con: --consensus {' '.join(best['combo'])} "
          f"--consensus-mode {best['mode_label'].lower()} --consensus-threshold-mode {threshold_mode}")


def print_abstain(report: Dict[str, Any], model: str, low: float, high: float) -> None:
    """Zona de abstención para un modelo: <low => no-robo, >high => robo, en medio => incierto."""
    if model not in report["models"]:
        print(f"[WARN] --abstain-model '{model}' no está entre los modelos. Se omite.")
        return
    pairs = report["pairs_by_model"][model]
    tp = tn = fp = fn = unc_robo = unc_norobo = 0
    for val, is_robo in pairs:
        if val > high:
            pred = 1
        elif val < low:
            pred = 0
        else:
            if is_robo:
                unc_robo += 1
            else:
                unc_norobo += 1
            continue
        if is_robo:
            tp += pred == 1
            fn += pred == 0
        else:
            fp += pred == 1
            tn += pred == 0
    decided = tp + tn + fp + fn
    incierto = unc_robo + unc_norobo
    total = decided + incierto
    d = _counts_to_metrics(0.0, tp, tn, fp, fn)
    print()
    print(f"Zona de abstención para '{model}': no-robo si <{low}, robo si >{high}, "
          f"incierto en medio (candidatos a revisar, p. ej. con el VLM).")
    print(f"    Muestras decididas : {decided}/{total} ({decided / total * 100.0:.1f}%)" if total else "    sin datos")
    print(f"    Inciertas (a revisar): {incierto}/{total} "
          f"({incierto / total * 100.0:.1f}%) -> robos={unc_robo}, no-robos={unc_norobo}" if total else "")
    print(f"    Sobre las DECIDIDAS: acierto={d['accuracy_pct']:.1f}% | recall={d['recall_pct']:.1f}% | "
          f"precisión={d['precision_pct']:.1f}% | FP={fp} | FN={fn}")


def write_output(output_path: Path, report: Dict[str, Any]) -> None:
    models = report["models"]
    stats = report["stats"]
    fmt = output_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == ".json":
        payload = {"threshold": report["threshold"], "resultados": {m: stats[m] for m in models}}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    elif fmt == ".csv":
        cols = ["modelo", "accuracy_pct", "aciertos", "evaluables",
                "TP", "TN", "FP", "FN", "recall_pct", "especificidad_pct", "precision_pct", "NA",
                "umbral_optimo", "accuracy_optimo_pct", "FP_optimo"]
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for m in models:
                s = stats[m]
                opt = s.get("optimo") or {}
                w.writerow([m, f"{s['accuracy_pct']:.4f}", s["aciertos"], s["evaluables"],
                            s["TP"], s["TN"], s["FP"], s["FN"],
                            f"{s['recall_pct']:.4f}", f"{s['especificidad_pct']:.4f}",
                            f"{s['precision_pct']:.4f}", s["NA"],
                            f"{opt.get('threshold'):.4f}" if opt else "",
                            f"{opt.get('accuracy_pct'):.4f}" if opt else "",
                            opt.get("FP", "")])
    else:
        raise SystemExit("--output debe terminar en .csv o .json")
    print(f"[INFO] Resumen escrito en: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa CSVs de logits: aciertos y falsos positivos por modelo."
    )
    parser.add_argument("--config", help="Config JSON con la lista de CSVs y cuál es el de robos.")
    parser.add_argument("--csvs", nargs="+", help="Lista de CSVs (alternativa al config).")
    parser.add_argument("--robo-csv", help="Ruta del CSV que contiene los robos (con --csvs).")
    parser.add_argument("--threshold", type=float, default=None, help="Umbral del logit (por defecto 0.0).")
    parser.add_argument("--output", help="Resumen de salida (.csv o .json).")

    # Análisis adicionales para reducir falsos positivos.
    parser.add_argument("--target-fpr", type=float, default=None,
                        help="FPR máximo en %% (p. ej. 1.0): umbral por modelo que lo cumple maximizando recall.")
    parser.add_argument("--target-precision", type=float, default=None,
                        help="Precisión mínima en %% (p. ej. 95): umbral por modelo que la cumple maximizando recall.")
    parser.add_argument("--fp-breakdown", action="store_true",
                        help="Muestra los falsos positivos por CSV (categoría) y modelo al umbral fijo.")
    parser.add_argument("--consensus", nargs="+", default=None,
                        help="Lista de modelos para evaluar consenso AND/OR (al menos 2).")
    parser.add_argument("--consensus-search", action="store_true",
                        help="Prueba TODAS las combinaciones y elige automáticamente la mejor.")
    parser.add_argument("--consensus-size", type=int, default=2,
                        help="Nº de modelos por combinación en --consensus-search (por defecto 2).")
    parser.add_argument("--consensus-rank", choices=["accuracy", "f1", "precision", "recall", "fp"],
                        default="accuracy",
                        help="Métrica para ordenar las combinaciones en la búsqueda (por defecto accuracy).")
    parser.add_argument("--consensus-top", type=int, default=10,
                        help="Cuántas combinaciones mostrar en la búsqueda (por defecto 10).")
    parser.add_argument("--consensus-mode", choices=["and", "or", "both"], default="both",
                        help="Modo de consenso a mostrar (por defecto ambos).")
    parser.add_argument("--consensus-threshold-mode", choices=["optimal", "fixed"], default="optimal",
                        help="Umbral por modelo en el consenso: óptimo (def) o el fijo --threshold.")
    parser.add_argument("--abstain-model", default=None,
                        help="Modelo para la zona de abstención (histéresis).")
    parser.add_argument("--abstain-low", type=float, default=None, help="Umbral bajo de la abstención.")
    parser.add_argument("--abstain-high", type=float, default=None, help="Umbral alto de la abstención.")
    args = parser.parse_args()

    if not args.config and not args.csvs:
        raise SystemExit("Indica --config o --csvs.")

    specs, threshold = load_csv_specs(args)
    if not specs:
        raise SystemExit("No se especificó ningún CSV.")

    faltan = [str(s["path"]) for s in specs if not s["path"].is_file()]
    if faltan:
        print("[ABORTADO] No existen estos CSV:")
        for f in faltan:
            print(f"  - {f}")
        raise SystemExit(1)

    if not any(s["robo"] for s in specs):
        raise SystemExit(
            "Ningún CSV está marcado como de robos. Usa 'robo': true en el config "
            "o --robo-csv en la línea de comandos."
        )

    report = evaluate(specs, threshold)
    print_report(report, specs)

    if args.target_fpr is not None or args.target_precision is not None:
        print_target_thresholds(report, args.target_fpr, args.target_precision)

    if args.fp_breakdown:
        print_fp_breakdown(report)

    if args.consensus:
        modes = ["and", "or"] if args.consensus_mode == "both" else [args.consensus_mode]
        print_consensus(report, args.consensus, modes, args.consensus_threshold_mode)

    if args.consensus_search:
        modes = ["and", "or"] if args.consensus_mode == "both" else [args.consensus_mode]
        if args.consensus_size < 2:
            raise SystemExit("--consensus-size debe ser >= 2.")
        print_consensus_search(
            report, args.consensus_size, modes,
            args.consensus_threshold_mode, args.consensus_rank, args.consensus_top,
        )

    if args.abstain_model is not None:
        if args.abstain_low is None or args.abstain_high is None:
            raise SystemExit("--abstain-model requiere --abstain-low y --abstain-high.")
        if args.abstain_low > args.abstain_high:
            raise SystemExit("--abstain-low debe ser <= --abstain-high.")
        print_abstain(report, args.abstain_model, args.abstain_low, args.abstain_high)

    if args.output:
        write_output(Path(args.output).expanduser().resolve(), report)


if __name__ == "__main__":
    main()
