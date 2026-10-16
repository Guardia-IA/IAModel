"""Rutas y utilidades de Campaign2 (independiente de campaign/)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGN2_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_ROOT.parent
CONFIG_PATH = CAMPAIGN2_ROOT / "campaign2_config.json"
ARTIFACTS_ROOT = CAMPAIGN2_ROOT / "artifacts"
JOBS_DIR = CAMPAIGN2_ROOT / "jobs"


def load_campaign2_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_class_maps_dir(config: Dict[str, Any]) -> Path:
    raw = config.get("class_maps_dir") or "../campaign/class_maps"
    p = Path(raw)
    if not p.is_absolute():
        p = (CAMPAIGN2_ROOT / p).resolve()
    return p


def artifacts_root(run_id: Optional[str] = None) -> Path:
    if run_id:
        return ARTIFACTS_ROOT / "runs" / str(run_id)
    return ARTIFACTS_ROOT


def current_run_pointer_path() -> Path:
    return ARTIFACTS_ROOT / "runs" / ".current_run"


def read_current_run_id() -> Optional[str]:
    p = current_run_pointer_path()
    if not p.is_file():
        return None
    rid = p.read_text(encoding="utf-8").strip()
    return rid or None


def write_current_run_id(run_id: str) -> Path:
    p = current_run_pointer_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(run_id).strip() + "\n", encoding="utf-8")
    return p


def new_run_id(prefix: str = "campaign2") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def run_layout(run_id: str) -> Dict[str, Path]:
    root = artifacts_root(run_id)
    layout = {
        "run_id": run_id,
        "root": root,
        "cache_dir": root / "cache",
        "jobs_dir": root / "jobs",
        "logs_dir": root / "logs",
        "plans_dir": root / "plans",
        "models_dir": root / "models",
        "reports_dir": root / "reports",
    }
    for d in layout.values():
        if isinstance(d, Path):
            d.mkdir(parents=True, exist_ok=True)
    return layout


def run_meta_path(run_id: str) -> Path:
    return artifacts_root(run_id) / "run_meta.json"


def job_queue_path(run_id: str) -> Path:
    return artifacts_root(run_id) / "jobs" / "queue.json"


def probe_report_path(run_id: str) -> Path:
    return artifacts_root(run_id) / "reports" / "gpu_probe.json"


def parallel_recommendation_path(run_id: str) -> Path:
    return artifacts_root(run_id) / "reports" / "parallel_recommendation.json"


def scheduler_log_path(run_id: str) -> Path:
    return artifacts_root(run_id) / "logs" / "scheduler_events.jsonl"


def resolve_data_roots(config: Dict[str, Any], cli_roots: Optional[List[str]] = None) -> List[Path]:
    if cli_roots:
        return [Path(r).expanduser().resolve() for r in cli_roots]
    roots_cfg = config.get("data_roots") or []
    out = [Path(r).expanduser().resolve() for r in roots_cfg if str(r).strip()]
    if out:
        return out
    try:
        import sys

        if str(TRAINING_DIR) not in sys.path:
            sys.path.insert(0, str(TRAINING_DIR))
        from train_model_operations import get_data_result_roots  # type: ignore

        legacy = get_data_result_roots(None)
        return [Path(r).resolve() for r in legacy if Path(r).is_dir()]
    except Exception:
        return []
