"""Progreso y ETA en vivo durante extract-csv (actualización periódica)."""
from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field

from triton_pose_extract.bench_estimate import fmt_duration


@dataclass
class BatchProgressMonitor:
    """Contadores thread-safe + hilo que imprime ETA cada ``interval_sec``."""

    total: int
    workers: int
    interval_sec: float = 20.0
    enabled: bool = True

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _t0: float = field(default=0.0, repr=False)
    _done: int = 0
    _ok: int = 0
    _skip: int = 0
    _fail: int = 0
    _job_seconds: list[float] = field(default_factory=list)
    _timing_sums: dict = field(default_factory=dict)
    _timing_n: int = 0

    def start(self) -> None:
        if not self.enabled or self.total <= 0:
            return
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._reporter_loop,
            name="extract_progress",
            daemon=True,
        )
        self._thread.start()
        print(
            f"[PROGRESS] Monitor ETA cada {self.interval_sec:.0f}s · "
            f"{self.total} clips · {self.workers} workers",
            flush=True,
        )

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 5.0)
            self._thread = None
        line = self._format_line(final=True)
        if line:
            print(line, flush=True)

    def on_job_done(self, *, seconds: float, status: str, timings: dict | None = None) -> None:
        with self._lock:
            self._done += 1
            self._job_seconds.append(float(seconds))
            if status == "ok":
                self._ok += 1
            elif status == "fail_infra":
                self._fail += 1
            else:
                self._skip += 1
            if timings:
                self._timing_n += 1
                for k, v in timings.items():
                    try:
                        self._timing_sums[k] = self._timing_sums.get(k, 0.0) + float(v)
                    except (TypeError, ValueError):
                        continue

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "total": self.total,
                "done": self._done,
                "ok": self._ok,
                "skip": self._skip,
                "fail": self._fail,
                "pct": round(100.0 * self._done / self.total, 2) if self.total else 0.0,
                "elapsed_sec": round(time.perf_counter() - self._t0, 1) if self._t0 else 0.0,
            }

    def _reporter_loop(self) -> None:
        while not self._stop.wait(timeout=max(1.0, self.interval_sec)):
            line = self._format_line(final=False)
            if line:
                print(line, flush=True)

    def _format_line(self, *, final: bool) -> str:
        with self._lock:
            done = self._done
            total = self.total
            ok, skip, fail = self._ok, self._skip, self._fail
            secs = list(self._job_seconds)
            t0 = self._t0
            tsums = dict(self._timing_sums)
            tn = self._timing_n

        if total <= 0:
            return ""
        elapsed = time.perf_counter() - t0 if t0 else 0.0
        pct = 100.0 * done / total
        prefix = "[PROGRESS · fin]" if final else "[PROGRESS]"
        eta_str = self._eta_text(done, total, elapsed, secs)
        rate_str = ""
        if elapsed > 0 and done > 0:
            jph = done / elapsed * 3600.0
            rate_str = f" · {jph:.1f} clips/h"

        line = (
            f"{prefix} {pct:5.1f}% ({done}/{total}) · "
            f"OK={ok} SKIP={skip} FAIL={fail} · "
            f"transcurrido {fmt_duration(elapsed)} · ETA {eta_str}{rate_str}"
        )
        if tn > 0 and tsums:
            order = (
                "ffmpeg_cut_sec",
                "probe_sec",
                "decode_sec",
                "decode_wall_sec",
                "preprocess_sec",
                "infer_track_export_sec",
                "infer_wait_sec",
                "triton_rtt_sec",
                "track_sec",
                "export_sec",
                "post_sec",
                "total_sec",
            )
            parts = []
            for k in order:
                if k in tsums:
                    parts.append(f"{k.removesuffix('_sec')}={tsums[k] / tn:.1f}s")
            if parts:
                line += "\n           medias/clip: " + " ".join(parts)
        return line

    def _eta_text(
        self,
        done: int,
        total: int,
        elapsed: float,
        job_seconds: list[float],
    ) -> str:
        remaining = max(0, total - done)
        if remaining == 0:
            return "0s (completado)"
        if done <= 0:
            return "… (esperando primer clip)"
        if elapsed <= 0:
            elapsed = 1e-6
        w = max(1, self.workers)
        # Throughput real del pool (incluye SKIP; refleja pared observada).
        eta_tp = remaining / (done / elapsed)
        parts = [fmt_duration(eta_tp)]
        if len(job_seconds) >= 2:
            med = statistics.median(job_seconds)
            eta_med = remaining * med / w
            parts.append(f"alt. {fmt_duration(eta_med)} (mediana×jobs/workers)")
        elif len(job_seconds) == 1:
            eta_one = remaining * job_seconds[0] / w
            parts.append(f"alt. {fmt_duration(eta_one)} (1 clip/workers)")
        return " · ".join(parts)
