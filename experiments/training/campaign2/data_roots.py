"""Descubrimiento de carpetas data_result / data_results."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set

DATA_RESULT_DIR_NAMES = ("data_result", "data_results")


def _as_data_result_dir(path: Path) -> Path | None:
    p = path.expanduser().resolve()
    if not p.is_dir():
        return None
    if p.name.lower() in DATA_RESULT_DIR_NAMES:
        return p
    for name in DATA_RESULT_DIR_NAMES:
        child = p / name
        if child.is_dir():
            return child
    return None


def discover_data_result_roots(raw_paths: Iterable[str | Path]) -> List[Path]:
    """
    Acepta:
      - Ruta directa a .../data_result(s)
      - Carpeta intermedia .../datos1/ con data_result dentro
      - Carpeta principal con hijos datos1/, datos2/, cada uno con data_result(s)
    """
    found: List[Path] = []
    seen: Set[str] = set()

    def _add(p: Path) -> None:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            found.append(p.resolve())

    for raw in raw_paths:
        base = Path(raw).expanduser().resolve()
        if not base.is_dir():
            continue
        direct = _as_data_result_dir(base)
        if direct is not None:
            _add(direct)
            continue
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            inner = _as_data_result_dir(sub)
            if inner is not None:
                _add(inner)
    return found
