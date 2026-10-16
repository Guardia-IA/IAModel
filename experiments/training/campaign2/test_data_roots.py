"""Tests ligeros discover_data_result_roots."""
from __future__ import annotations

import tempfile
from pathlib import Path

from data_roots import discover_data_result_roots


def test_discover_parent_and_direct() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "datos1" / "data_result").mkdir(parents=True)
        (root / "datos2" / "data_results").mkdir(parents=True)
        direct = root / "solo" / "data_result"
        direct.mkdir(parents=True)
        found = discover_data_result_roots([str(root), str(direct.parent)])
        names = sorted(p.name for p in found)
        assert "data_result" in names or "data_results" in names
        assert len(found) >= 3


if __name__ == "__main__":
    test_discover_parent_and_direct()
    print("OK data_roots")
