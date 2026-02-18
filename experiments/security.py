"""
Comprobaciones previas antes de ejecutar experimentos.
Valida estructura de CSVs: video existe, horas HH:mm:ss, inicio <= fin, clasificación 0-13.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd

HMS_PATTERN = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
DEFAULT_MIN_CLAS = 0
DEFAULT_MAX_CLAS = 13


def _is_valid_hms(val) -> bool:
    """Comprueba que el valor tenga formato HH:mm:ss."""
    s = str(val).strip()
    if not HMS_PATTERN.match(s):
        return False
    parts = s.split(":")
    h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    return 0 <= m < 60 and 0 <= sec < 60 and h >= 0


def _hms_to_seconds(val: str) -> int:
    """Convierte HH:mm:ss a segundos."""
    parts = str(val).strip().split(":")
    h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + sec


def validate_csv(
    csv_path: str,
    base_dir: str | None = None,
    min_clas: int = DEFAULT_MIN_CLAS,
    max_clas: int = DEFAULT_MAX_CLAS,
) -> dict:
    """
    Valida un CSV con columnas: video, inicio, fin, clasificacion.

    Comprueba:
    1) Primera columna = video, y que existe (solo cuando cambia el nombre)
    2) inicio y fin en formato HH:mm:ss
    3) inicio <= fin
    4) clasificacion es entero en [min_clas, max_clas]
    5) No hay celdas vacías

    Devuelve JSON con errores o con total_rows y by_type.
    """
    base_dir = base_dir or os.path.dirname(os.path.abspath(csv_path))
    errors = []
    last_video = None

    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", header=0)
    except Exception as e:
        return {"ok": False, "errors": [f"No se pudo leer el CSV: {e}"], "file": csv_path}

    if df.shape[1] < 4:
        return {"ok": False, "errors": ["CSV debe tener al menos 4 columnas"], "file": csv_path}

    # Primera fila = cabecera. Validamos desde la segunda fila (índices 0, 1, 2... en df)
    # Nombre de columnas por índice: 0=video, 1=inicio, 2=fin, 3=clasificacion
    by_type = {}
    for idx, row in df.iterrows():
        row_num = idx + 2  # +2: fila 1=cabecera, fila 2=primer dato
        video = row.iloc[0]
        inicio = row.iloc[1]
        fin = row.iloc[2]
        clas = row.iloc[3]
        row_errors = []

        # 5) Celdas vacías
        if pd.isna(video) or pd.isna(inicio) or pd.isna(fin) or pd.isna(clas):
            row_errors.append({"row": row_num, "msg": "Celda vacía", "row_data": row.tolist()})
            errors.extend(row_errors)
            continue

        video = str(video).strip()
        inicio_str = str(inicio).strip()
        fin_str = str(fin).strip()

        # 5) Celdas vacías (string)
        if not video or not inicio_str or not fin_str:
            row_errors.append({"row": row_num, "msg": "Celda vacía", "row_data": row.tolist()})
            errors.extend(row_errors)
            continue

        # Clasificación < 0: ignorar fila (no procesar, no error)
        try:
            if int(clas) < 0:
                continue
        except (ValueError, TypeError):
            pass  # se reportará como error más abajo

        # 2) Formato HH:mm:ss
        if not _is_valid_hms(inicio_str):
            row_errors.append({"row": row_num, "msg": f"Inicio no válido (HH:mm:ss): '{inicio}'"})
        if not _is_valid_hms(fin_str):
            row_errors.append({"row": row_num, "msg": f"Fin no válido (HH:mm:ss): '{fin}'"})

        # 3) inicio <= fin
        if _is_valid_hms(inicio_str) and _is_valid_hms(fin_str):
            if _hms_to_seconds(inicio_str) > _hms_to_seconds(fin_str):
                row_errors.append({"row": row_num, "msg": f"Inicio ({inicio_str}) > Fin ({fin_str})"})

        # 4) clasificación entero en rango
        try:
            c = int(clas)
            if c < min_clas or c > max_clas:
                row_errors.append({"row": row_num, "msg": f"Clasificación {c} fuera de [{min_clas}, {max_clas}]"})
        except (ValueError, TypeError):
            row_errors.append({"row": row_num, "msg": f"Clasificación no es entero: '{clas}'"})

        # 1) Video existe (solo cuando cambia)
        if video != last_video:
            last_video = video
            full_path = os.path.join(base_dir, video)
            if not os.path.isfile(full_path):
                row_errors.append({"row": row_num, "msg": f"Vídeo no encontrado: {full_path}", "video": video})

        errors.extend(row_errors)
        if not row_errors:
            c = int(clas)
            by_type[c] = by_type.get(c, 0) + 1

    if errors:
        return {"ok": False, "errors": errors, "file": csv_path}

    return {
        "ok": True,
        "file": csv_path,
        "total_rows": len(df),
        "by_type": dict(sorted(by_type.items())),
    }


def validate_folder(
    folder_path: str,
    min_clas: int = DEFAULT_MIN_CLAS,
    max_clas: int = DEFAULT_MAX_CLAS,
) -> dict:
    """
    Recorre recursivamente folder_path, valida cada CSV y concatena resultados.
    Devuelve JSON global con resumen.
    """
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        return {"ok": False, "errors": [f"No existe o no es directorio: {folder_path}"], "csvs": []}

    csv_files = sorted(folder.rglob("*.csv"))
    results = []
    all_ok = True
    total_clips = 0
    global_by_type = {}

    for csv_path in csv_files:
        rel_path = csv_path.relative_to(folder) if folder != csv_path.parent else csv_path.name
        base_dir = str(csv_path.parent)

        result = validate_csv(str(csv_path), base_dir=base_dir, min_clas=min_clas, max_clas=max_clas)
        results.append(result)

        print(f"\n--- CSV: {csv_path} ---")
        if result["ok"]:
            print("  Estado: OK")
            total_clips += result["total_rows"]
            for k, v in result["by_type"].items():
                global_by_type[k] = global_by_type.get(k, 0) + v
        else:
            all_ok = False
            print("  Estado: ERROR")
            for e in result["errors"]:
                if isinstance(e, dict):
                    print(f"  - Fila {e.get('row', '?')}: {e.get('msg', e)}")
                else:
                    print(f"  - {e}")

    summary = {
        "ok": all_ok,
        "total_csvs": len(csv_files),
        "total_clips": total_clips,
        "by_type": dict(sorted(global_by_type.items())),
    }
    if not all_ok:
        summary["csvs"] = [{"file": r["file"], "ok": r["ok"]} for r in results]

    print("\n" + "=" * 60)
    print("RESUMEN GLOBAL")
    print("=" * 60)
    if all_ok:
        by_type_str = ", ".join(f"tipo {k}={v}" for k, v in summary["by_type"].items())
        print(f"Todos los ficheros son coherentes. Existen un total de {total_clips} clips ({by_type_str}).")
    else:
        print("Hay errores en uno o más CSVs. Revisa los mensajes anteriores.")
    print(f"\nJSON resumen:\n{json.dumps(summary, indent=2, ensure_ascii=False)}")

    return summary


def count_by_category(folder_path: str, category: int | None = None) -> dict:
    """
    Recorre recursivamente todos los CSV en folder_path y cuenta clips por categoría.
    Ignora filas con clasificación < 0 (igual que validate_csv).
    Útil para comprobar que el conteo de validate_folder es correcto.

    Args:
        folder_path: Ruta a la carpeta con CSVs.
        category: Si se pasa, devuelve solo el conteo de esa categoría (clave extra en el resultado).

    Returns:
        {"by_type": {0: n, 1: m, ...}, "total": N, "total_csvs": k}
        Si category se especifica, añade "category": cat, "count": n.
    """
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        return {"error": f"No existe o no es directorio: {folder_path}"}

    csv_files = sorted(folder.rglob("*.csv"))
    by_type = {}

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python", header=0)
        except Exception:
            continue
        if df.shape[1] < 4:
            continue

        for _, row in df.iterrows():
            clas = row.iloc[3]
            if pd.isna(clas):
                continue
            try:
                c = int(clas)
            except (ValueError, TypeError):
                continue
            if c < 0:
                continue
            by_type[c] = by_type.get(c, 0) + 1

    result = {
        "by_type": dict(sorted(by_type.items())),
        "total": sum(by_type.values()),
        "total_csvs": len(csv_files),
    }
    if category is not None:
        result["category"] = category
        result["count"] = by_type.get(category, 0)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Valida CSVs antes de experimentos")
    parser.add_argument("path", type=str, help="Ruta al CSV o carpeta a revisar")
    parser.add_argument("--min-clas", type=int, default=DEFAULT_MIN_CLAS, help="Clasificación mínima (default 0)")
    parser.add_argument("--max-clas", type=int, default=DEFAULT_MAX_CLAS, help="Clasificación máxima (default 13)")
    parser.add_argument("--count", "-c", type=int, default=None, metavar="N", help="Solo contar clips de categoría N en la carpeta")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir() and args.count is not None:
        result = count_by_category(str(path), category=args.count)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if path.is_file():
        result = validate_csv(str(path), min_clas=args.min_clas, max_clas=args.max_clas)
        if not result["ok"]:
            print("Estado: ERROR")
            for e in result["errors"]:
                if isinstance(e, dict):
                    print(f"  Fila {e.get('row', '?')}: {e.get('msg', e)}")
                else:
                    print(f"  {e}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif path.is_dir():
        validate_folder(str(path), min_clas=args.min_clas, max_clas=args.max_clas)
    else:
        print(f"Error: no existe {path}")


if __name__ == "__main__":
    main()
