#!/usr/bin/env python3
"""Estado en vivo de jobs P1 (tail-friendly)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
for p in (TRAINING_DIR, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import job_queue_path, read_current_run_id, run_layout  # noqa: E402

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
        row["wall_seconds"] = r.get("wall_seconds")
        row["checkpoint"] = r.get("best_checkpoint")
        return row
    fail = d / "result.json"
    if fail.is_file():
        with open(fail, "r", encoding="utf-8") as f:
            wrap = json.load(f)
        if wrap.get("status") != "ok":
            row["state"] = "failed"
            row["error"] = (wrap.get("error") or wrap.get("result") or {}).get("error", "")[:120]
            return row
    prog = _last_progress(d / "progress.jsonl")
    if prog:
        row["state"] = "running"
        row.update(prog)
    elif (d / "spec.json").is_file():
        row["state"] = "queued_or_starting"
    return row


def render_status(run_id: str, *, watch: bool = False) -> str:
    layout = run_layout(run_id)
    jobs_dir = layout["jobs_dir"]
    qpath = job_queue_path(run_id)
    job_ids: List[str] = []
    purpose = "?"
    if qpath.is_file():
        q = json.loads(qpath.read_text(encoding="utf-8"))
        purpose = str(q.get("purpose", "?"))
        job_ids = [str(j["job_id"]) for j in q.get("jobs", []) if j.get("job_type") == "p1_train"]
    if not job_ids and jobs_dir.is_dir():
        job_ids = sorted(
            p.name for p in jobs_dir.iterdir() if p.is_dir() and p.name.startswith(("p1_", "v2_"))
        )

    lines: List[str] = []
    lines.append(f"{BOLD}{'═' * 78}{RESET}")
    lines.append(f"{BOLD}Campaign2 P1 — estado{RESET}  run={CYAN}{run_id}{RESET}  cola={purpose}")
    lines.append(f"{DIM}Comando: watch -n 5 ./run_campaign2.sh p1-status{RESET}")
    lines.append(f"{BOLD}{'═' * 78}{RESET}")

    if not job_ids:
        lines.append(f"{YELLOW}Sin jobs P1 en cola ni carpetas v2_*/p1_*.{RESET}")
        return "\n".join(lines)

    done = sum(1 for j in job_ids if _job_row(jobs_dir, j)["state"] == "done")
    fail = sum(1 for j in job_ids if _job_row(jobs_dir, j)["state"] == "failed")
    run = sum(1 for j in job_ids if _job_row(jobs_dir, j)["state"] == "running")
    lines.append(f"Progreso: {GREEN}{done}{RESET}/{len(job_ids)} OK  "
                   f"{CYAN}{run}{RESET} en curso  {RED}{fail}{RESET} fallidos\n")

    for jid in job_ids:
        r = _job_row(jobs_dir, jid)
        st = r["state"]
        if st == "done":
            icon = f"{GREEN}✓{RESET}"
            detail = (
                f"épocas={r.get('epochs_ran')} "
                f"val_rec={r.get('val_clip_recall_pct', 0):.1f}% "
                f"val_FP={r.get('val_clip_fp')} "
                f"score={r.get('best_score', 0):.2f}"
            )
        elif st == "running":
            icon = f"{CYAN}…{RESET}"
            detail = (
                f"época {r.get('epoch', '?')}/{r.get('epochs_total', '?')} "
                f"loss={r.get('train_loss', 0):.4f} "
                f"val_rec={r.get('val_clip_recall_pct', 0):.1f}% "
                f"val_FP={r.get('val_clip_fp', '?')}"
            )
        elif st == "failed":
            icon = f"{RED}✗{RESET}"
            detail = str(r.get("error", ""))[:70]
        else:
            icon = f"{DIM}○{RESET}"
            detail = "en cola / arrancando"
        lines.append(f"  {icon} {BOLD}{jid:<32}{RESET} {detail}")

    lines.append(f"\n{BOLD}{'═' * 78}{RESET}")
    if done == len(job_ids) and done > 0:
        lines.append(f"Listo: {GREEN}./run_campaign2.sh p1-summary{RESET}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()
    run_id = args.run_id or read_current_run_id()
    if not run_id:
        print("ERROR: sin RUN_ID", file=sys.stderr)
        return 2
    print(render_status(run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
