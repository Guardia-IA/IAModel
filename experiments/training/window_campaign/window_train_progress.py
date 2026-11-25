"""Estado de entrenamiento legible (log + JSON para revisar por la mañana)."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

WINDOW_DIR = Path(__file__).resolve().parent
DEFAULT_PROGRESS_PATH = WINDOW_DIR / "artifacts" / "logs" / "train_progress.json"

_lock = threading.Lock()

try:
    import fcntl  # Unix: workers paralelos comparten el mismo JSON

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_progress(path: Path = DEFAULT_PROGRESS_PATH) -> Dict[str, Any]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def merge_progress(path: Path, patch: Dict[str, Any]) -> Dict[str, Any]:
    """Lee JSON actual, fusiona patch (experiments por clave) y guarda (flock si hay)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        if path.is_file() and path.stat().st_size > 0:
            try:
                base = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                base = {}
        else:
            base = {}
        if "experiments" in patch and isinstance(patch["experiments"], dict):
            exp = base.setdefault("experiments", {})
            for k, v in patch["experiments"].items():
                if isinstance(v, dict) and isinstance(exp.get(k), dict):
                    exp[k] = {**exp[k], **v}
                else:
                    exp[k] = v
            rest = {k: v for k, v in patch.items() if k != "experiments"}
            base.update(rest)
        else:
            base.update(patch)
        base["updated_at"] = _utc_now()
        completed = sum(1 for v in base.get("experiments", {}).values() if v.get("status") == "done")
        base["completed_count"] = completed
        total = len(base.get("experiment_ids_total") or [])
        base["pending_count"] = max(0, total - completed)
        base["current_experiment_ids"] = _running_ids(base)

        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            if _HAS_FCNTL:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except OSError:
                    pass
            json.dump(base, f, indent=2, ensure_ascii=False)
            f.flush()
        tmp.replace(path)
        return base


def write_progress(path: Path, data: Dict[str, Any]) -> None:
    merge_progress(path, data)


def init_batch(
    *,
    path: Path,
    stage: str,
    exp_ids: List[int],
    parallel: int,
    cell_id: str,
    data_root: str,
) -> Dict[str, Any]:
    data = {
        "phase": "train",
        "stage": stage,
        "cell_id": cell_id,
        "data_root": data_root,
        "parallel_jobs": parallel,
        "experiment_ids_total": list(exp_ids),
        "experiments": {
            str(eid): {"status": "pending", "epoch": 0, "epochs_total": None}
            for eid in exp_ids
        },
        "started_at": _utc_now(),
    }
    write_progress(path, data)
    return data


def set_experiment_running(
    path: Path,
    exp_id: int,
    *,
    epochs_total: int,
    arch: str,
) -> None:
    merge_progress(
        path,
        {
            "experiments": {
                str(exp_id): {
                    "status": "running",
                    "epoch": 0,
                    "epochs_total": epochs_total,
                    "arch": arch,
                    "started_at": _utc_now(),
                }
            }
        },
    )


def tick_epoch(
    path: Path,
    exp_id: int,
    epoch: int,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    patch: Dict[str, Any] = {
        "status": "running",
        "epoch": int(epoch),
    }
    if extra:
        patch["last_metrics"] = extra
    merge_progress(path, {"experiments": {str(exp_id): patch}})


def finish_experiment(
    path: Path,
    exp_id: int,
    *,
    ok: bool = True,
    wall_seconds: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    patch: Dict[str, Any] = {
        "status": "done" if ok else "failed",
        "finished_at": _utc_now(),
    }
    if wall_seconds is not None:
        patch["wall_seconds"] = round(wall_seconds, 1)
    if error:
        patch["error"] = error[:2000]
    merge_progress(path, {"experiments": {str(exp_id): patch}})


def _running_ids(data: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    for k, v in (data.get("experiments") or {}).items():
        if v.get("status") == "running":
            try:
                out.append(int(k))
            except ValueError:
                continue
    return sorted(out)


def format_status_line(data: Dict[str, Any]) -> str:
    total = len(data.get("experiment_ids_total") or [])
    done = int(data.get("completed_count") or 0)
    running = data.get("current_experiment_ids") or []
    parts = [f"[PROGRESS] {done}/{total} modelos listos"]
    if running:
        chunks = []
        for eid in running:
            ex = (data.get("experiments") or {}).get(str(eid), {})
            ep = ex.get("epoch", "?")
            et = ex.get("epochs_total", "?")
            chunks.append(f"exp{eid:02d} ep {ep}/{et}")
        parts.append("en curso: " + ", ".join(chunks))
    return " | ".join(parts)


def attach_heartbeat(
    *,
    path: Path,
    interval_sec: float = 60.0,
    started_at: Optional[float] = None,
) -> threading.Event:
    """Hilo que escribe línea de estado + ETA (lee JSON en cada tick)."""
    stop = threading.Event()
    t0 = started_at if started_at is not None else time.perf_counter()

    def _loop() -> None:
        while not stop.wait(interval_sec):
            data = load_progress(path)
            done = int(data.get("completed_count") or 0)
            total = max(1, len(data.get("experiment_ids_total") or []))
            elapsed = time.perf_counter() - t0
            if done > 0:
                eta = (elapsed / done) * (total - done)
                eta_s = f"ETA ~{int(eta // 3600)}h {int((eta % 3600) // 60)}m"
            else:
                eta_s = "ETA tras 1er modelo"
            line = (
                format_status_line(data)
                + f" | {eta_s} | elapsed {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"
            )
            print(line, flush=True)
            merge_progress(path, {"heartbeat": line, "elapsed_seconds": round(elapsed, 1)})

    threading.Thread(target=_loop, name="window-train-heartbeat", daemon=True).start()
    return stop