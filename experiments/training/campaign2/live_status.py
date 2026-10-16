"""Estado en vivo de jobs (P1 y otros) — mismo terminal que train."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from campaign2_paths import job_queue_path, run_layout

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


def _last_progress(progress_path: Path) -> Optional[Dict[str, Any]]:
    if not progress_path.is_file():
        return None
    last = None
    with open(progress_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    pass
    return last


def _job_row(jobs_dir: Path, job_id: str) -> Dict[str, Any]:
    d = jobs_dir / job_id
    row: Dict[str, Any] = {"job_id": job_id, "state": "pending"}
    res = d / "p1_train_result.json"
    if res.is_file():
        with open(res, "r", encoding="utf-8") as f:
            r = json.load(f)
        row["state"] = "done"
        row["epochs_ran"] = r.get("epochs_ran")
        row["best_score"] = r.get("best_score")
        le = r.get("last_epoch") or {}
        row["val_clip_recall_pct"] = le.get("val_clip_recall_pct")
        row["val_clip_fp"] = le.get("val_clip_fp")
        row["val_clip_recall_at_fp_pct"] = le.get("val_clip_recall_at_fp_pct")
        row["wall_seconds"] = r.get("wall_seconds")
        return row
    fail = d / "result.json"
    if fail.is_file():
        with open(fail, "r", encoding="utf-8") as f:
            wrap = json.load(f)
        if wrap.get("status") != "ok":
            row["state"] = "failed"
            err = wrap.get("error") or (wrap.get("result") or {}).get("error", "")
            row["error"] = str(err)[:120]
            return row
    prog = _last_progress(d / "progress.jsonl")
    if prog:
        row["state"] = "running"
        row.update(prog)
    elif (d / "spec.json").is_file():
        row["state"] = "queued_or_starting"
    return row


def _queue_job_ids(run_id: str, jobs_dir: Path) -> tuple[List[str], str]:
    qpath = job_queue_path(run_id)
    purpose = "?"
    job_ids: List[str] = []
    if qpath.is_file():
        q = json.loads(qpath.read_text(encoding="utf-8"))
        purpose = str(q.get("purpose", "?"))
        job_ids = [str(j["job_id"]) for j in q.get("jobs", [])]
    if not job_ids and jobs_dir.is_dir():
        job_ids = sorted(p.name for p in jobs_dir.iterdir() if p.is_dir())
    return job_ids, purpose


def render_live_status(
    run_id: str,
    *,
    title: str = "Campaign2 — en vivo",
    hint: str | None = None,
) -> str:
    layout = run_layout(run_id)
    jobs_dir = layout["jobs_dir"]
    job_ids, purpose = _queue_job_ids(run_id, jobs_dir)

    lines: List[str] = []
    lines.append(f"{BOLD}{'─' * 78}{RESET}")
    lines.append(f"{BOLD}{title}{RESET}  run={CYAN}{run_id}{RESET}  cola={purpose}")
    if hint:
        lines.append(f"{DIM}{hint}{RESET}")

    if not job_ids:
        lines.append(f"{YELLOW}Sin jobs en cola.{RESET}")
        lines.append(f"{BOLD}{'─' * 78}{RESET}")
        return "\n".join(lines)

    rows = [_job_row(jobs_dir, jid) for jid in job_ids]
    done = sum(1 for r in rows if r["state"] == "done")
    fail = sum(1 for r in rows if r["state"] == "failed")
    run = sum(1 for r in rows if r["state"] == "running")
    lines.append(
        f"Jobs: {GREEN}{done}{RESET}/{len(job_ids)} OK  "
        f"{CYAN}{run}{RESET} activos  {RED}{fail}{RESET} fallidos"
    )

    for r, jid in zip(rows, job_ids):
        st = r["state"]
        if st == "done":
            icon = f"{GREEN}✓{RESET}"
            detail = (
                f"épocas={r.get('epochs_ran')} "
                f"rec@FP={r.get('val_clip_recall_at_fp_pct', r.get('val_clip_recall_pct', 0)):.1f}% "
                f"score={r.get('best_score', 0):.2f}"
            )
        elif st == "running":
            icon = f"{CYAN}…{RESET}"
            rec_fp = r.get("val_clip_recall_at_fp_pct")
            rec_s = f" rec@FP={rec_fp:.1f}%" if rec_fp is not None else ""
            detail = (
                f"época {r.get('epoch', '?')}/{r.get('epochs_total', '?')} "
                f"loss={r.get('train_loss', 0):.4f} "
                f"val_rec={r.get('val_clip_recall_pct', 0):.1f}% "
                f"FP={r.get('val_clip_fp', '?')}{rec_s}"
            )
        elif st == "failed":
            icon = f"{RED}✗{RESET}"
            detail = str(r.get("error", ""))[:70]
        else:
            icon = f"{DIM}○{RESET}"
            detail = "en cola / arrancando"
        lines.append(f"  {icon} {BOLD}{jid:<36}{RESET} {detail}")

    lines.append(f"{BOLD}{'─' * 78}{RESET}")
    return "\n".join(lines)


class LiveStatusThread:
    """Refresca el panel en stderr para no mezclar con líneas [job_id] de stdout."""

    def __init__(self, run_id: str, *, interval_sec: float = 4.0) -> None:
        self.run_id = run_id
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.active = False

    def start(self) -> None:
        if not sys.stderr.isatty():
            return
        self._thread = threading.Thread(target=self._loop, name="live_status", daemon=True)
        self._thread.start()
        self.active = True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_sec + 2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            block = render_live_status(
                self.run_id,
                title="Campaign2 — progreso (mismo terminal)",
                hint="Las líneas [job_id] abajo son logs de entrenamiento/eval.",
            )
            print(block, file=sys.stderr, flush=True)
            self._stop.wait(self.interval_sec)


def watch_forever(run_id: str, interval_sec: float = 4.0) -> None:
    """Modo p1-status --watch (un solo proceso, Ctrl+C para salir)."""
    try:
        while True:
            if sys.stdout.isatty():
                print("\033[H\033[J", end="", flush=True)
            print(render_live_status(run_id, title="Campaign2 P1 — watch"))
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\n(watch detenido)")
