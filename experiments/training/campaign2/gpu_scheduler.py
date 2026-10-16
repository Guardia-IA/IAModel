"""
Cola de jobs Campaign2 con ejecución paralela en subprocess (Sprint 1).

Cada job corre en job_worker.py con directorio de salida aislado.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class JobSpec:
    job_id: str
    job_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0


@dataclass
class JobResult:
    job_id: str
    status: str
    returncode: int
    wall_seconds: float
    output_dir: str
    error: Optional[str] = None


class GpuJobScheduler:
    def __init__(
        self,
        *,
        run_id: str,
        campaign2_dir: Path,
        max_parallel: int,
        log_path: Path,
        cuda_device_index: int = 0,
    ) -> None:
        self.run_id = run_id
        self.campaign2_dir = campaign2_dir
        self.max_parallel = max(1, int(max_parallel))
        self.log_path = log_path
        self.cuda_device_index = cuda_device_index
        self._lock = threading.Lock()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _run_one(self, job: JobSpec, jobs_root: Path) -> JobResult:
        out_dir = jobs_root / job.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        spec_path = out_dir / "spec.json"
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_id": job.job_id,
                    "job_type": job.job_type,
                    "run_id": self.run_id,
                    "params": job.params,
                },
                f,
                indent=2,
            )
            f.write("\n")

        worker = self.campaign2_dir / "job_worker.py"
        import os

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device_index)
        env["CAMPAIGN2_RUN_ID"] = self.run_id

        cmd = [
            sys.executable,
            str(worker),
            "--spec",
            str(spec_path),
            "--output-dir",
            str(out_dir),
        ]
        t0 = time.perf_counter()
        self._log("job_start", {"job_id": job.job_id, "cmd": cmd})
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.campaign2_dir.parent),
                env=env,
                capture_output=True,
                text=True,
            )
            wall = time.perf_counter() - t0
            (out_dir / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
            (out_dir / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
            status = "ok" if proc.returncode == 0 else "failed"
            self._log(
                "job_end",
                {"job_id": job.job_id, "status": status, "returncode": proc.returncode, "wall_seconds": wall},
            )
            return JobResult(
                job_id=job.job_id,
                status=status,
                returncode=proc.returncode,
                wall_seconds=wall,
                output_dir=str(out_dir),
                error=None if proc.returncode == 0 else (proc.stderr or "")[:2000],
            )
        except Exception as exc:
            wall = time.perf_counter() - t0
            self._log("job_error", {"job_id": job.job_id, "error": str(exc)})
            return JobResult(
                job_id=job.job_id,
                status="error",
                returncode=-1,
                wall_seconds=wall,
                output_dir=str(out_dir),
                error=str(exc),
            )

    def run_queue(
        self,
        jobs: List[JobSpec],
        jobs_root: Path,
        on_complete: Optional[Callable[[JobResult], None]] = None,
    ) -> List[JobResult]:
        jobs_sorted = sorted(jobs, key=lambda j: (-j.priority, j.job_id))
        results: List[JobResult] = []
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futures = {pool.submit(self._run_one, job, jobs_root): job for job in jobs_sorted}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                mark = "OK" if res.status == "ok" else "FAIL"
                print(
                    f"[scheduler] {mark} {res.job_id} "
                    f"{res.wall_seconds:.1f}s → {res.output_dir}",
                    flush=True,
                )
                if res.status != "ok" and res.error:
                    print(f"  error: {res.error[:300]}", flush=True)
                elif res.status != "ok":
                    err_file = Path(res.output_dir) / "result.json"
                    if err_file.is_file():
                        try:
                            wrap = json.loads(err_file.read_text(encoding="utf-8"))
                            err = wrap.get("error") or (wrap.get("result") or {}).get("error")
                            if err:
                                print(f"  error: {str(err)[:300]}", flush=True)
                        except Exception:
                            pass
                if on_complete:
                    on_complete(res)
        return results
