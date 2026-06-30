"""Utilidades para runs de mejora (hard negatives, augment uniforme, boost por categoría FP)."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def load_fp_manifest_csv(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe manifest FP: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def extract_uids_from_fp_rows(rows: List[Dict[str, str]]) -> List[str]:
    uids: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for key in ("uid", "uid_absolute"):
            u = str(row.get(key) or "").strip()
            if u and u not in seen:
                seen.add(u)
                uids.append(u)
                break
    return uids


def fp_category_counts(rows: List[Dict[str, str]]) -> Dict[int, int]:
    c: Counter[int] = Counter()
    for row in rows:
        try:
            cat = int(row.get("folder_category") or row.get("category_str") or -1)
        except (TypeError, ValueError):
            continue
        if cat >= 0:
            c[cat] += 1
    return dict(sorted(c.items()))


def apply_uniform_ops_per_clip(
    proposed: Dict[int, int],
    ops_per_clip: int,
    *,
    robbery_class: int = 6,
    cap_robbery: bool = True,
) -> Dict[int, int]:
    """Fija el mismo número de variantes augment por categoría (experimento uniforme)."""
    n = max(0, int(ops_per_clip))
    out = {int(k): n for k in proposed.keys()}
    if cap_robbery and robbery_class in out and n > 0:
        out[int(robbery_class)] = max(0, int(round(n * 0.85)))
    return out


def boost_aug_from_fp_categories(
    proposed: Dict[int, int],
    fp_rows: List[Dict[str, str]],
    *,
    boost_factor: float = 1.5,
    extra_per_fp_cat: int = 1,
    robbery_class: int = 6,
    max_aug: int = 15,
) -> Dict[int, int]:
    """Sube augment en categorías que aparecen en el manifest de falsos positivos."""
    counts = fp_category_counts(fp_rows)
    out = {int(k): int(v) for k, v in proposed.items()}
    for cat, n_fp in counts.items():
        if cat == robbery_class:
            continue
        cur = out.get(cat, 0)
        boosted = int(round(cur * boost_factor)) + int(extra_per_fp_cat)
        out[cat] = min(max_aug, max(cur, boosted))
    return out


def write_hard_negative_manifest(
    uids: List[str],
    path: Path,
    *,
    source_csv: Optional[Path] = None,
    weight: float = 3.0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "uids": uids,
        "uid_weight": float(weight),
        "source_csv": str(source_csv.resolve()) if source_csv else None,
        "count": len(uids),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def load_hard_negative_manifest(path: Path) -> tuple[Set[str], float]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    uids = {str(u) for u in data.get("uids", [])}
    weight = float(data.get("uid_weight", 3.0))
    return uids, weight
