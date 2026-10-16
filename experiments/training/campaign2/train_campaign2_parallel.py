#!/usr/bin/env python3
"""Ejecuta la cola de jobs Campaign2 en paralelo (Sprint 1)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
for p in (TRAINING_DIR, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import (  # noqa: E402
    job_queue_path,
    load_campaign2_config,
    parallel_recommendation_path,
    read_current_run_id,
    run_layout,
    scheduler_log_path,
)
from gpu_scheduler import GpuJobScheduler, JobSpec  # noqa: E402
from live_status import LiveStatusThread  # noqa: E402


def _load_max_parallel(cfg: dict, run_id: str, override: Optional[int]) -> int:
    if override is not None and override > 0:
        return override
    rec_path = parallel_recommendation_path(run_id)
    if rec_path.is_file():
        with open(rec_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        return int(rec.get("max_parallel_jobs", cfg.get("parallelism", {}).get("max_parallel_jobs", 4)))
    return int(cfg.get("parallelism", {}).get("max_parallel_jobs", 4))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument(
        "--no-live-status",
        action="store_true",
        help="No mostrar panel de progreso periódico en stderr",
    )
    args = ap.parse_args()

    cfg = load_campaign2_config(Path(args.config) if args.config else None)
    run_id = args.run_id or read_current_run_id()
    if not run_id:
        print("ERROR: sin RUN_ID", file=sys.stderr)
        return 2

    qpath = job_queue_path(run_id)
    if not qpath.is_file():
        print(f"ERROR: falta cola {qpath}. Ejecuta preflight_campaign2.py", file=sys.stderr)
        return 2

    with open(qpath, "r", encoding="utf-8") as f:
        queue = json.load(f)

    jobs = [
        JobSpec(
            job_id=str(j["job_id"]),
            job_type=str(j["job_type"]),
            params=dict(j.get("params") or {}),
            priority=int(j.get("priority", 0)),
        )
        for j in queue.get("jobs", [])
    ]
    purpose = queue.get("purpose", "unknown")
    phase = queue.get("phase", "sprint1_infra")

    if not jobs:
        print(
            "\n=== Campaign2 TRAIN (scheduler) ===\n"
            "Cola vacía — no hay nada que ejecutar.\n"
            "Sprint 1 solo prepara caché/inventario; el entrenamiento Kanvis llega en Sprint 2.\n"
            "Para probar GPU en paralelo: preflight --smoke-test && train\n"
        )
        return 0

    if purpose == "campaign_matrix" or phase == "campaign_matrix":
        print(
            "\n"
            + "=" * 72
            + "\n"
            "Campaign2 MATRIZ: jobs P0 (legacy .pt × full/filtered) + P1 train_v2 en paralelo.\n"
            "Después: p1-eval + ensemble-leaderboard\n"
            + "=" * 72
            + "\n",
            flush=True,
        )
    elif purpose == "p1_eval" or phase == "sprint3_p1_eval":
        print(
            "\n"
            + "=" * 72
            + "\n"
            "Campaign2 P1 eval: forward ventanas sobre best.pt\n"
            "Tras terminar: ./run_campaign2.sh p1-eval-summary\n"
            + "=" * 72
            + "\n",
            flush=True,
        )
    elif purpose == "p1_train" or ("p1" in purpose and "eval" not in purpose) or phase == "sprint3_p1":
        print(
            "\n"
            + "=" * 72
            + "\n"
            "Campaign2 P1: entrenamiento train_v2 (ventanas, checkpoints por job).\n"
            "Artefactos: artifacts/runs/<RUN_ID>/jobs/<job_id>/best.pt\n"
            + "=" * 72
            + "\n",
            flush=True,
        )
    elif purpose == "p0_baseline" or phase == "sprint2_p0":
        print(
            "\n"
            + "=" * 72
            + "\n"
            "Campaign2 P0: evaluación incumbent #41/#48 (forward, sin re-entrenar).\n"
            "Tras terminar: ./run_campaign2.sh p0-summary\n"
            + "=" * 72
            + "\n",
            flush=True,
        )
    elif purpose == "smoke_test" or phase == "sprint1_infra":
        print(
            "\n"
            + "=" * 72
            + "\n"
            "AVISO: estos jobs son PRUEBA DE INFRA (smoke/smoke_train/cache_touch).\n"
            "NO entrenan el modelo Kanvis ni guardan modelo_41/modelo_48.\n"
            + "=" * 72
            + "\n",
            flush=True,
        )

    max_par = _load_max_parallel(cfg, run_id, args.max_parallel)
    par_cfg = cfg.get("parallelism") or {}
    layout = run_layout(run_id)
    scheduler = GpuJobScheduler(
        run_id=run_id,
        campaign2_dir=CAMPAIGN2_DIR,
        max_parallel=max_par,
        log_path=scheduler_log_path(run_id),
        cuda_device_index=int(par_cfg.get("cuda_device_index", 0)),
    )

    print(f"Campaign2 train-parallel run_id={run_id} jobs={len(jobs)} max_parallel={max_par}")
    live = LiveStatusThread(run_id, interval_sec=4.0)
    if not args.no_live_status:
        live.start()
        if live.active:
            print(
                "(Panel de jobs cada ~4s arriba; logs con prefijo [job_id] en stdout)\n",
                flush=True,
            )
    try:
        results = scheduler.run_queue(jobs, layout["jobs_dir"])
    finally:
        live.stop()

    summary = {
        "run_id": run_id,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "max_parallel": max_par,
        "results": [
            {
                "job_id": r.job_id,
                "status": r.status,
                "returncode": r.returncode,
                "wall_seconds": round(r.wall_seconds, 3),
                "output_dir": r.output_dir,
                "error": r.error,
            }
            for r in results
        ],
    }
    out_path = layout["reports_dir"] / "train_parallel_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    failed = [r for r in results if r.status != "ok"]
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
