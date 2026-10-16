#!/usr/bin/env python3
"""
Preflight Campaign2: inventario, caché 12 Hz (mmap), cola de jobs Sprint 1.

No escribe en campaign/ ni modifica datos de entrenamiento legacy.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
for p in (TRAINING_DIR, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import (  # noqa: E402
    job_queue_path,
    load_campaign2_config,
    read_current_run_id,
    resolve_data_roots,
    run_layout,
    run_meta_path,
)
from dataset_cache import build_pose_cache, collect_clip_records  # noqa: E402
from gpu_scheduler import JobSpec  # noqa: E402


def _split_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for e in entries:
        sp = e.get("split", "unknown")
        out.setdefault(sp, {"clips": 0, "robbery": 0})
        out[sp]["clips"] += 1
        if e.get("is_robbery"):
            out[sp]["robbery"] += 1
    return out


def build_smoke_job_queue(
    cfg: Dict[str, Any],
    *,
    manifest_path: Optional[Path],
    run_id: str,
) -> List[JobSpec]:
    n = int(cfg.get("sprint1_smoke_jobs", 3))
    jobs: List[JobSpec] = []
    for i in range(n):
        jtype = "smoke_train" if i == 0 else "smoke"
        params: Dict[str, Any] = {}
        if jtype == "smoke_train":
            par = cfg.get("parallelism") or {}
            params = {
                "batch_size": int(par.get("probe_batch_size", 64)),
                "seq_len": int((cfg.get("temporal") or {}).get("seq_len", 96)),
                "feature_dim": int(par.get("probe_feature_dim", 62)),
                "steps": 8,
            }
        else:
            params = {"sleep_seconds": 0.5 + 0.1 * i}
        jobs.append(JobSpec(job_id=f"smoke_{i+1:02d}", job_type=jtype, params=params, priority=0))

    if manifest_path and manifest_path.is_file():
        jobs.append(
            JobSpec(
                job_id="cache_touch_01",
                job_type="cache_touch",
                params={"manifest_path": str(manifest_path)},
                priority=1,
            )
        )
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight Campaign2")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--data-root", action="append", default=None, help="Override data_result (repetible)")
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--max-clips", type=int, default=0, help="Limit clips (debug)")
    args = ap.parse_args()

    cfg = load_campaign2_config(Path(args.config) if args.config else None)
    run_id = args.run_id or read_current_run_id()
    if not run_id:
        print("ERROR: sin RUN_ID. Ejecuta: ./run_campaign2.sh new-run", file=sys.stderr)
        return 2

    layout = run_layout(run_id)
    data_roots = resolve_data_roots(cfg, args.data_root)
    if not data_roots:
        print(
            "WARN: no hay data_roots. Añade rutas en campaign2_config.json o --data-root.",
            file=sys.stderr,
        )

    temporal = cfg.get("temporal") or {}
    cache_cfg = cfg.get("cache") or {}
    split_cfg = cfg.get("split") or {}

    records: List[Dict[str, Any]] = []
    cache_result = None
    if data_roots and not args.skip_cache and cache_cfg.get("enabled", True):
        records = collect_clip_records(
            data_roots,
            pose_source=str(cfg.get("pose_source", "full")),
            single_user_only=bool(cfg.get("single_user_only", False)),
            robbery_class=int(cfg.get("robbery_class", 6)),
        )
        if args.max_clips > 0:
            records = records[: args.max_clips]
        if not records:
            print("WARN: 0 clips encontrados; caché omitida.", file=sys.stderr)
        elif records:
            cache_result = build_pose_cache(
                records,
                layout["cache_dir"],
                target_hz=float(temporal.get("target_hz", 12.0)),
                split_ratios=dict(split_cfg.get("ratios") or {}),
                split_seed=int(split_cfg.get("seed", 42)),
                use_shm=bool(cache_cfg.get("use_shm", True)),
                shm_basename=str(cache_cfg.get("shm_basename", "kanvis_pose_12hz")),
            )

    manifest_path = layout["cache_dir"] / "manifest.json"
    if cache_result:
        manifest_path = cache_result.manifest_path

    jobs = build_smoke_job_queue(cfg, manifest_path=manifest_path if manifest_path.is_file() else None, run_id=run_id)
    queue_payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [
            {"job_id": j.job_id, "job_type": j.job_type, "params": j.params, "priority": j.priority}
            for j in jobs
        ],
    }
    qpath = job_queue_path(run_id)
    with open(qpath, "w", encoding="utf-8") as f:
        json.dump(queue_payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    preflight_report = {
        "run_id": run_id,
        "data_roots": [str(p) for p in data_roots],
        "num_clip_records": len(records),
        "cache": None
        if not cache_result
        else {
            "manifest": str(cache_result.manifest_path),
            "total_frames": cache_result.total_frames,
            "num_clips": cache_result.num_clips,
            "bytes_on_disk": cache_result.bytes_on_disk,
            "poses_mmap": str(cache_result.poses_mmap_path),
        },
        "split_summary": _split_summary(
            json.loads(manifest_path.read_text(encoding="utf-8")).get("entries", [])
            if manifest_path.is_file()
            else []
        ),
        "job_queue": str(qpath),
        "num_jobs": len(jobs),
    }
    report_path = layout["reports_dir"] / "preflight_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(preflight_report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    meta = {
        "run_id": run_id,
        "phase": "preflight_done",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config) if args.config else "campaign2_config.json"),
    }
    with open(run_meta_path(run_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(json.dumps(preflight_report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
