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
    - Opcional --output resumen.csv | resumen.json

Uso:
    python logits_evaluation.py --config logits_evaluation_config.json
    python logits_evaluation.py --csvs robos.csv compras.csv --robo-csv robos.csv --threshold 0.5
    python logits_evaluation.py --config cfg.json --output resumen.csv
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


def _counts_at(pairs: List[Tuple[float, bool]], thr: float) -> Tuple[int, int, int, int]:
    """Devuelve (TP, TN, FP, FN) para un umbral dado (robo si logit > thr)."""
    tp = tn = fp = fn = 0
    for val, is_robo in pairs:
        pred_robo = val > thr
        if is_robo:
            tp += pred_robo
            fn += not pred_robo
        else:
            fp += pred_robo
            tn += not pred_robo
    return tp, tn, fp, fn


def optimal_threshold(pairs: List[Tuple[float, bool]]) -> Optional[Dict[str, Any]]:
    """
    Busca el umbral que maximiza el % de acierto (TP+TN). Empates: menos FP y,
    entre los óptimos, el umbral central (máximo margen) para mayor robustez.
    """
    if not pairs:
        return None
    vals = sorted({v for v, _ in pairs})
    # Candidatos: por debajo del mínimo, puntos medios entre valores y por encima del máximo.
    cands = [vals[0] - 1.0]
    for i in range(len(vals) - 1):
        cands.append((vals[i] + vals[i + 1]) / 2.0)
    cands.append(vals[-1] + 1.0)

    scored = []
    for t in cands:
        tp, tn, fp, fn = _counts_at(pairs, t)
        n = tp + tn + fp + fn
        acc = (tp + tn) / n if n else 0.0
        scored.append((acc, fp, t, tp, tn, fn))

    best_acc = max(s[0] for s in scored)
    cand_acc = [s for s in scored if s[0] == best_acc]
    min_fp = min(s[1] for s in cand_acc)
    cand_fp = sorted((s for s in cand_acc if s[1] == min_fp), key=lambda s: s[2])
    acc, fp, t, tp, tn, fn = cand_fp[len(cand_fp) // 2]  # umbral central del plateau óptimo
    return {
        "threshold": t, "accuracy_pct": acc * 100.0,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def evaluate(specs: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """Calcula métricas por modelo agregando todas las muestras de todos los CSV."""
    # Carga de todos los CSV y unión de modelos.
    loaded: List[Tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]] = []
    model_order: List[str] = []
    for spec in specs:
        cols, rows = read_logits_csv(spec["path"])
        for c in cols:
            if c not in model_order:
                model_order.append(c)
        loaded.append((spec, cols, rows))

    # Inicializa contadores y pares (logit, is_robo) por modelo.
    stats: Dict[str, Dict[str, int]] = {
        m: {"TP": 0, "FN": 0, "TN": 0, "FP": 0, "NA": 0} for m in model_order
    }
    pairs_by_model: Dict[str, List[Tuple[float, bool]]] = {m: [] for m in model_order}

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
    results: Dict[str, Any] = {}
    for m in model_order:
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

    return {"threshold": threshold, "models": model_order, "stats": results}


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

    if args.output:
        write_output(Path(args.output).expanduser().resolve(), report)


if __name__ == "__main__":
    main()
