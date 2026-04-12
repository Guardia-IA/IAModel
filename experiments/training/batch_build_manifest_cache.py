#!/usr/bin/env python3
"""
Genera manifests validate_npy (un JSON por UID) en manifest_cache/.

El nombre de fichero es md5(UTF-8 del UID estable del .npy), igual que
train_model_operations.manifest_cache_path_for_uid (ruta relativa a data_result
si aplica, o absoluta como legado).

Uso (desde esta carpeta):
  python batch_build_manifest_cache.py --pose-source filtered
  python batch_build_manifest_cache.py --single-user-only --skip-existing
  python batch_build_manifest_cache.py --log-file ./manifest_batch.log --quiet --jobs 8

Requiere las mismas rutas de datos que train (model_config.DATA_RESULT_ROOT).
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TextIO

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - entorno sin tqdm
    tqdm = None  # type: ignore[misc, assignment]

_TRAINING_DIR = Path(__file__).resolve().parent
_VALIDATE_NPY = _TRAINING_DIR / "operations_npy" / "validate_npy.py"
_DEFAULT_VALIDATE_JSON = _TRAINING_DIR / "operations_npy" / "validate_npy.json"


class _RunLog:
    """Traza con marca temporal, flush inmediato y tee opcional a consola (para `cat` del fichero en background)."""

    def __init__(self, path: Path | None, *, tee_stdout: bool = True) -> None:
        self._path = path
        self._tee = tee_stdout
        self._fp: TextIO | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = open(path, "a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        if self._fp:
            self._fp.close()
            self._fp = None

    def line(self, msg: str, *, also_print: bool | None = None) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{ts}] {msg}"
        if self._fp:
            self._fp.write(text + "\n")
            self._fp.flush()
        if also_print is None:
            also_print = self._tee if self._fp else True
        if also_print:
            print(text, flush=True)

    def __enter__(self) -> _RunLog:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _pool_run_validate(payload: tuple[list[str], bool]) -> tuple[int, str | None]:
    """Ejecuta validate_npy en subproceso; definido a nivel de módulo para ProcessPoolExecutor."""
    cmd, quiet = payload
    if quiet:
        r = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return r.returncode, (r.stderr.strip() if r.stderr else None)
    r = subprocess.run(cmd)
    return r.returncode, None


def _import_ops():
    try:
        from model_config import (  # type: ignore[attr-defined]
            MIN_CLIP_SECONDS,
            MIN_VALID_FRAMES,
            MIN_VALID_PCT,
            MAX_OCCLUSION_RATIO,
            AUGMENT_PROFILE_DEFAULT,
            VALIDATE_NPY_MIRROR_COMPOSE_RATIO,
            VALIDATE_NPY_COMPOSE_LIGHT_RATIO,
        )
        from train_model_operations import (  # type: ignore[attr-defined]
            collect_examples,
            manifest_cache_path_for_uid,
            _example_uid,
            MANIFEST_CACHE_DIR,
        )
    except ImportError:
        sys.path.insert(0, str(_TRAINING_DIR))
        from model_config import (  # type: ignore[attr-defined]
            MIN_CLIP_SECONDS,
            MIN_VALID_FRAMES,
            MIN_VALID_PCT,
            MAX_OCCLUSION_RATIO,
            AUGMENT_PROFILE_DEFAULT,
            VALIDATE_NPY_MIRROR_COMPOSE_RATIO,
            VALIDATE_NPY_COMPOSE_LIGHT_RATIO,
        )
        from train_model_operations import (  # type: ignore[attr-defined]
            collect_examples,
            manifest_cache_path_for_uid,
            _example_uid,
            MANIFEST_CACHE_DIR,
        )
    return (
        collect_examples,
        manifest_cache_path_for_uid,
        _example_uid,
        MANIFEST_CACHE_DIR,
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
        AUGMENT_PROFILE_DEFAULT,
        VALIDATE_NPY_MIRROR_COMPOSE_RATIO,
        VALIDATE_NPY_COMPOSE_LIGHT_RATIO,
    )


def main() -> None:
    (
        collect_examples,
        manifest_cache_path_for_uid,
        _example_uid,
        default_cache,
        def_min_clip,
        def_min_vf,
        def_min_pct,
        def_occ,
        def_profile,
        def_mirror_ratio,
        def_compose_light,
    ) = _import_ops()

    p = argparse.ArgumentParser(description="Rellena operations_npy/manifest_cache con validate_npy por UID.")
    p.add_argument(
        "--cache-dir",
        type=str,
        default=str(default_cache),
        help="Salida: un .json por UID (md5).",
    )
    p.add_argument("--pose-source", choices=["filtered", "full"], default="filtered")
    p.add_argument("--single-user-only", action="store_true")
    p.add_argument(
        "--profile",
        type=str,
        default=def_profile,
        help=f"Perfil en validate_npy.json (alinear con train; default {def_profile}).",
    )
    p.add_argument("--config", type=str, default=str(_DEFAULT_VALIDATE_JSON))
    p.add_argument("--skip-existing", action="store_true", help="No regenerar si ya existe el JSON.")
    p.add_argument("--dry-run", action="store_true", help="Lista UIDs sin ejecutar validate_npy.")
    p.add_argument("--max", type=int, default=None, help="Procesar como mucho N UIDs únicos.")
    p.add_argument("--min-clip-seconds", type=float, default=float(def_min_clip))
    p.add_argument("--min-valid-frames", type=int, default=int(def_min_vf))
    p.add_argument("--min-valid-pct", type=float, default=float(def_min_pct))
    p.add_argument("--max-occlusion-ratio", type=float, default=float(def_occ))
    p.add_argument(
        "--mirror-compose-ratio",
        type=float,
        default=def_mirror_ratio,
        help=f"Reenviado a validate_npy (default model_config {def_mirror_ratio}).",
    )
    p.add_argument(
        "--compose-light-ratio",
        type=float,
        default=def_compose_light,
        help=f"Reenviado a validate_npy (default model_config {def_compose_light}).",
    )
    p.add_argument("--step", type=float, default=None, help="Opcional: step global en validate_npy.")
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Silencia la salida de validate_npy (menos I/O en terminal; suele ahorrar tiempo).",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Procesos en paralelo (cada uno ejecuta validate_npy en subproceso). Default 1.",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Fichero de traza (una línea por evento, flush inmediato). Ideal para nohup y `tail -f` / `cat`.",
    )
    p.add_argument(
        "--no-console",
        action="store_true",
        help="Con --log-file, no duplicar las líneas en stdout (solo fichero).",
    )
    args = p.parse_args()

    if int(args.jobs) < 1:
        raise SystemExit("--jobs debe ser >= 1")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.config).expanduser().resolve()
    if not _VALIDATE_NPY.is_file():
        raise SystemExit(f"No se encuentra validate_npy.py en {_VALIDATE_NPY}")
    if not cfg_path.is_file():
        raise SystemExit(f"No se encuentra config: {cfg_path}")

    log_path = Path(args.log_file).expanduser().resolve() if args.log_file else None
    log = _RunLog(log_path, tee_stdout=not args.no_console)
    t_wall0 = time.perf_counter()

    try:
        log.line(
            f"inicio | pid={os.getpid()} | cache_dir={cache_dir} | profile={args.profile} | "
            f"jobs={int(args.jobs)} | quiet={bool(args.quiet)} | dry_run={bool(args.dry_run)}"
        )
        log.line("fase: collect_examples (puede tardar)…")

        examples = collect_examples(
            pose_source=args.pose_source,
            single_user_only=args.single_user_only,
            min_clip_seconds=float(args.min_clip_seconds),
            min_valid_frames=int(args.min_valid_frames),
            min_valid_pct=float(args.min_valid_pct),
            max_occlusion_ratio=float(args.max_occlusion_ratio),
        )
        seen: set[str] = set()
        # (uid estable para nombre md5, ruta absoluta para np.load en validate_npy)
        jobs: list[tuple[str, str]] = []
        for ex in examples:
            uid = _example_uid(ex)
            if uid in seen:
                continue
            seen.add(uid)
            jobs.append((uid, str(ex.pose_path.resolve())))

        if args.max is not None:
            jobs = jobs[: max(0, int(args.max))]

        log.line(f"UIDs únicos en dataset (tras dedup): {len(jobs)}")

        def build_cmd(uid: str, abs_npy: str, out: Path) -> list[str]:
            c = [
                sys.executable,
                str(_VALIDATE_NPY),
                abs_npy,
                "--config",
                str(cfg_path),
                "--profile",
                args.profile,
                "--manifest",
                str(out),
                "--mirror-compose-ratio",
                str(args.mirror_compose_ratio),
                "--compose-light-ratio",
                str(args.compose_light_ratio),
            ]
            if args.step is not None:
                c.extend(["--step", str(args.step)])
            return c

        ok = 0
        skipped = 0
        failed = 0
        pending: list[tuple[str, str, list[str]]] = []

        for uid, abs_npy in jobs:
            out = manifest_cache_path_for_uid(cache_dir, uid)
            if args.skip_existing and out.exists():
                skipped += 1
                continue
            if args.dry_run:
                log.line(f"[dry-run] uid={uid!r} npy={abs_npy} -> {out.name}")
                ok += 1
                continue
            pending.append((uid, abs_npy, build_cmd(uid, abs_npy, out)))

        quiet = bool(args.quiet)
        n_jobs = int(args.jobs)
        total_pending = len(pending)

        log.line(
            f"resumen | a_generar_validate_npy={total_pending} | omitidos_skip_existing={skipped}"
            + (f" | dry_run_lineas={ok}" if args.dry_run else "")
        )

        if args.dry_run:
            dur = time.perf_counter() - t_wall0
            log.line(
                f"FIN | modo=dry-run | lineas_registradas={ok} | omitidos_skip_existing={skipped} | "
                f"duracion_total_s={dur:.1f}"
            )
        elif not pending:
            log.line("nada que ejecutar (0 pendientes tras filtros / skip-existing).")
            dur = time.perf_counter() - t_wall0
            log.line(
                f"FIN | generados_ok=0 | omitidos_skip_existing={skipped} | fallidos=0 | duracion_total_s={dur:.1f}"
            )
        else:

            def run_cmd(cmd: list[str]) -> tuple[int, str | None]:
                if quiet:
                    r = subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    return r.returncode, (r.stderr.strip() if r.stderr else None)
                r = subprocess.run(cmd)
                return r.returncode, None

            def _log_progress(done: int, uid: str, abs_npy: str) -> None:
                elapsed = time.perf_counter() - t_batch0
                pct = 100.0 * done / total_pending if total_pending else 100.0
                eta = (elapsed / done) * (total_pending - done) if done > 0 and total_pending > done else 0.0
                log.line(
                    f"progreso {done}/{total_pending} ({pct:.1f}%) | ok={ok} fail={failed} | "
                    f"t={elapsed:.1f}s eta≈{eta:.1f}s | uid={uid[:24]} | {Path(abs_npy).name}"
                )

            t_batch0 = time.perf_counter()

            if n_jobs <= 1:
                seq: Any = pending
                use_pbar = tqdm is not None and sys.stderr.isatty()
                if use_pbar:
                    seq = tqdm(
                        pending,
                        total=total_pending,
                        desc="validate_npy",
                        unit="clip",
                        file=sys.stderr,
                        mininterval=0.5,
                    )
                done = 0
                for uid, abs_npy, cmd in seq:
                    code, err = run_cmd(cmd)
                    if code == 0:
                        ok += 1
                    else:
                        failed += 1
                        log.line(f"ERROR validate_npy falló | uid={uid!r} | {abs_npy}")
                        if err:
                            log.line(f"ERROR stderr: {err[:2000]}")
                    done += 1
                    _log_progress(done, uid, abs_npy)
            else:
                with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                    futs = {
                        ex.submit(_pool_run_validate, (cmd, quiet)): (uid, abs_npy)
                        for uid, abs_npy, cmd in pending
                    }
                    iterator: Any = as_completed(futs)
                    use_pbar = tqdm is not None and sys.stderr.isatty()
                    if use_pbar:
                        iterator = tqdm(
                            iterator,
                            total=total_pending,
                            desc="validate_npy",
                            unit="clip",
                            file=sys.stderr,
                            mininterval=0.5,
                        )
                    done = 0
                    for fut in iterator:
                        uid, abs_npy = futs[fut]
                        try:
                            code, err = fut.result()
                        except Exception as e:
                            failed += 1
                            log.line(f"ERROR excepción en worker | uid={uid!r} | {abs_npy} | {e}")
                            done += 1
                            _log_progress(done, uid, abs_npy)
                            continue
                        if code == 0:
                            ok += 1
                        else:
                            failed += 1
                            log.line(f"ERROR validate_npy falló | uid={uid!r} | {abs_npy}")
                            if err:
                                log.line(f"ERROR stderr: {err[:2000]}")
                        done += 1
                        _log_progress(done, uid, abs_npy)

            dur = time.perf_counter() - t_wall0
            log.line(
                f"FIN | generados_ok={ok} | omitidos_skip_existing={skipped} | fallidos={failed} | "
                f"duracion_total_s={dur:.1f}"
            )
    finally:
        log.close()


if __name__ == "__main__":
    main()
