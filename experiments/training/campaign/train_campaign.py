#!/usr/bin/env python3
"""
Entrena los experimentos seleccionados para cada celda de la campaña.

Uso:
  python train_campaign.py --cells bin_full
  python train_campaign.py --all
  python train_campaign.py --all --resume
  python train_campaign.py --run-id improve_v1 --config campaign_config_improve.json --all
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

try:
    from campaign_paths import (
        load_merged_campaign_config,
        filter_cells,
        ensure_cell_dirs,
        training_plan_path,
        category_aug_path,
        hard_negative_manifest_path,
        resolve_experiment_ids,
        robbery_class_ids,
    )
    from model_config import EXPERIMENTS, DEFAULT_BINARY_SOFTMAX_THRESHOLD, DEFAULT_BINARY_LOGIT_MARGIN
    from train_model_operations import (
        run_experiment,
        load_training_plan_json,
        AUGMENT_CONFIG_PATH,
        AUGMENT_PROFILE_DEFAULT,
        AUGMENT_PROB,
        AUGMENT_MAX_OPS,
        SEED,
        MAX_DETERMINISTIC_VARIANTS,
        TRAIN_DETERMINISTIC_PROB,
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
    )
except ImportError as exc:
    raise SystemExit(f"Import error (¿entorno con torch?): {exc}") from exc


def _aug_profile(config: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
    return dict((config.get("aug_profiles") or {})[profile_id])


def _train_opts_for_cell(cell: Dict[str, Any], config: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    aug_id = plan.get("campaign", {}).get("aug_profile") or cell.get("aug_profile")
    if not aug_id and cell.get("improve_profile"):
        ip = (config.get("improve_profiles") or {})[cell["improve_profile"]]
        aug_id = ip.get("base_aug_profile", "fp_hardened")
    aug_prof = _aug_profile(config, str(aug_id))
    train_opts = dict(aug_prof.get("train_opts") or {})
    improve = plan.get("campaign", {}).get("improve") or {}
    ip_id = improve.get("improve_profile") or cell.get("improve_profile")
    if ip_id:
        ip = (config.get("improve_profiles") or {}).get(ip_id, {})
        train_opts.update(ip.get("train_opts") or {})
    return train_opts


def _model_exists(models_dir: Path, exp_id: int) -> bool:
    return (models_dir / f"modelo_{exp_id:02d}.pt").is_file()


def _parallel_jobs(config: Dict[str, Any], override: Optional[int]) -> int:
    if override is not None:
        return max(1, int(override))
    par = config.get("parallelism") or {}
    return max(1, int(par.get("max_parallel_jobs", 2)))


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _child_env(config: Dict[str, Any], slots: int) -> Dict[str, str]:
    par = config.get("parallelism") or {}
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(int(par.get("cuda_device_index", 0)))
    env["CAMPAIGN_NUM_WORKERS"] = str(int(par.get("dataloader_workers", 2)))
    env["CAMPAIGN_TRAIN_WORKER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    cpus = os.cpu_count() or 8
    threads = max(1, cpus // max(1, slots * 2))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(key, str(threads))
    return env


def _train_one_experiment(
    *,
    exp_id: int,
    cell: Dict[str, Any],
    config: Dict[str, Any],
    plan: Dict[str, Any],
    plan_path: Path,
    models_dir: Path,
    splits_dir: Path,
    train_opts: Dict[str, Any],
    hn_manifest: Optional[Path],
    hn_weight: float,
    device: torch.device,
    bs_thr: float,
    bs_margin: float,
    run_id: Optional[str],
) -> Dict[str, Any]:
    cell_id = cell["id"]
    cfg = dict(EXPERIMENTS[exp_id - 1])
    t0 = time.perf_counter()
    try:
        res = run_experiment(
            exp_id,
            cfg,
            device,
            task=cell["task"],
            positive_class=robbery_class_ids(config)[0],
            positive_classes=robbery_class_ids(config),
            pose_source_override=cell["pose_source"],
            single_user_only=bool(config.get("single_user_only", True)),
            models_dir=models_dir,
            min_clip_seconds=MIN_CLIP_SECONDS,
            min_valid_frames=MIN_VALID_FRAMES,
            min_valid_pct=MIN_VALID_PCT,
            max_occlusion_ratio=MAX_OCCLUSION_RATIO,
            split_manifest_out=(splits_dir / f"split_manifest_exp_{exp_id:02d}.json"),
            category_aug_config_path=category_aug_path(cell_id, run_id=run_id),
            training_plan_path=plan_path,
            loss_type=str(train_opts.get("loss_type", "ce")),
            hard_negative_mining=bool(train_opts.get("hard_negative_mining", False)),
            target_neg_pos_ratio=train_opts.get("target_neg_pos_ratio"),
            maintain_class_ratio=bool(train_opts.get("maintain_class_ratio", False)),
            hard_negative_uid_manifest=hn_manifest,
            hard_negative_uid_weight=hn_weight,
            augment_config_path=AUGMENT_CONFIG_PATH,
            augment_profile=AUGMENT_PROFILE_DEFAULT,
            augment_prob=AUGMENT_PROB,
            augment_max_ops=AUGMENT_MAX_OPS,
            augment_seed=SEED,
            max_deterministic_variants=MAX_DETERMINISTIC_VARIANTS,
            train_deterministic_prob=TRAIN_DETERMINISTIC_PROB,
            binary_softmax_threshold=bs_thr,
            binary_logit_margin=bs_margin,
        )
        res["cell_id"] = cell_id
        res["run_id"] = run_id
        res["wall_seconds"] = time.perf_counter() - t0
        return res
    except Exception as exc:
        print(f"[Exp {exp_id:02d}] ERROR: {exc}")
        return {
            "exp_id": exp_id,
            "cell_id": cell_id,
            "run_id": run_id,
            "error": str(exc),
            "wall_seconds": time.perf_counter() - t0,
        }


def _write_summary(
    arts: Dict[str, Path],
    *,
    cell_id: str,
    run_id: Optional[str],
    wall_total: float,
    exp_list: List[int],
    hn_manifest: Optional[Path],
    results: List[Dict[str, Any]],
    parallel: int,
) -> Path:
    summary_path = arts["reports_dir"] / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_id": cell_id,
        "run_id": run_id,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "wall_seconds": wall_total,
        "parallel_jobs": parallel,
        "experiment_ids": exp_list,
        "hard_negative_manifest": str(hn_manifest) if hn_manifest else None,
        "results": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return summary_path


def _run_exp_subprocess(
    *,
    cell_id: str,
    exp_id: int,
    config_path: Optional[Path],
    run_id: Optional[str],
    resume: bool,
    job_result: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--cells",
        cell_id,
        "--exp-ids",
        str(exp_id),
        "--parallel",
        "1",
        "--job-result",
        str(job_result),
    ]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    if run_id:
        cmd.extend(["--run-id", run_id])
    if resume:
        cmd.append("--resume")

    label = f"{cell_id}/exp{exp_id:02d}"
    t0 = time.perf_counter()
    print(f"[paralelo] inicio {label}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(CAMPAIGN_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        chunks.append(line)
        print(f"[{label}] {line}", end="", flush=True)
    proc.wait()
    wall = time.perf_counter() - t0
    if job_result.is_file():
        try:
            payload = json.loads(job_result.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"error": "job-result ilegible", "exp_id": exp_id, "cell_id": cell_id}
    else:
        tail = "".join(chunks)[-2000:]
        payload = {
            "exp_id": exp_id,
            "cell_id": cell_id,
            "run_id": run_id,
            "error": tail or f"worker exit {proc.returncode} sin job-result",
        }
    if proc.returncode != 0 and "error" not in payload:
        payload["error"] = f"worker exit {proc.returncode}"
    payload["wall_seconds"] = float(payload.get("wall_seconds") or wall)
    payload["returncode"] = proc.returncode
    return payload


def _train_cell_parallel(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    exp_list: List[int],
    resume: bool,
    run_id: Optional[str],
    parallel: int,
    config_path: Optional[Path],
    arts: Dict[str, Path],
) -> Tuple[List[Dict[str, Any]], float]:
    cell_id = cell["id"]
    models_dir = arts["models_dir"]
    jobs_dir = arts["reports_dir"] / "train_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    pending: List[int] = []
    for exp_id in exp_list:
        if exp_id < 1 or exp_id > len(EXPERIMENTS):
            print(f"[Exp {exp_id:02d}] Fuera de rango EXPERIMENTS, omitido.")
            continue
        if resume and _model_exists(models_dir, exp_id):
            print(f"[Exp {exp_id:02d}] Ya existe, resume → omitido.")
            continue
        pending.append(exp_id)

    if not pending:
        return [], 0.0

    env = _child_env(config, parallel)
    print(
        f"Paralelo celda {cell_id}: {len(pending)} experimentos, "
        f"{parallel} a la vez, CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}, "
        f"dataloader workers={env.get('CAMPAIGN_NUM_WORKERS')}"
    )

    results: List[Dict[str, Any]] = []
    wall_sum = 0.0
    lock = threading.Lock()
    inflight: List[str] = []
    done = 0
    total = len(pending)
    t_wall0 = time.perf_counter()
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(30):
            with lock:
                running = list(inflight)
                n_done = done
                elapsed = time.perf_counter() - t_wall0
            if n_done <= 0:
                eta = "esperando el primer experimento"
            else:
                eta = _fmt_duration(elapsed / n_done * (total - n_done))
            print(
                f"[paralelo] {cell_id} {n_done}/{total} listos | "
                f"en curso: {', '.join(running) or '-'} | ETA {eta}",
                flush=True,
            )

    beater = threading.Thread(target=_beat, name=f"eta-{cell_id}", daemon=True)
    beater.start()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _wrapped(exp_id: int, label: str, job_result: Path) -> Dict[str, Any]:
        with lock:
            inflight.append(label)
        try:
            return _run_exp_subprocess(
                cell_id=cell_id,
                exp_id=exp_id,
                config_path=config_path,
                run_id=run_id,
                resume=resume,
                job_result=job_result,
                env=env,
            )
        finally:
            with lock:
                if label in inflight:
                    inflight.remove(label)

    futures = {}
    try:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for exp_id in pending:
                label = f"exp{exp_id:02d}"
                job_result = jobs_dir / f"exp_{exp_id:02d}.json"
                fut = pool.submit(_wrapped, exp_id, label, job_result)
                futures[fut] = (exp_id, label)
            for fut in as_completed(futures):
                exp_id, label = futures[fut]
                payload = fut.result()
                with lock:
                    done += 1
                    n_done = done
                    elapsed = time.perf_counter() - t_wall0
                wall_sum += float(payload.get("wall_seconds") or 0.0)
                results.append(payload)
                status = "FAIL" if payload.get("error") else "OK"
                eta = _fmt_duration(elapsed / n_done * (total - n_done)) if n_done < total else "0s"
                print(
                    f"[paralelo] {status} {cell_id}/exp{exp_id:02d} "
                    f"{_fmt_duration(float(payload.get('wall_seconds') or 0.0))} | "
                    f"{n_done}/{total} | ETA {eta}",
                    flush=True,
                )
    finally:
        stop.set()

    results.sort(key=lambda r: int(r.get("exp_id") or 0))
    return results, wall_sum


def train_cell(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    resume: bool = False,
    exp_ids: Optional[List[int]] = None,
    run_id: Optional[str] = None,
    parallel: int = 1,
    config_path: Optional[Path] = None,
    write_summary: bool = True,
) -> Dict[str, Any]:
    cell_id = cell["id"]
    arts = ensure_cell_dirs(cell_id, run_id=run_id)
    models_dir = arts["models_dir"]
    splits_dir = arts["splits_dir"]
    plan_path = training_plan_path(cell_id, run_id=run_id)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Falta plan: {plan_path}. Ejecuta preflight_campaign.py --write-all")

    plan = load_training_plan_json(plan_path)
    if exp_ids is not None:
        exp_list = resolve_experiment_ids(exp_ids)
    else:
        from learning_curve_utils import experiment_ids_for_cell

        exp_list = experiment_ids_for_cell(cell, config)
    train_opts = _train_opts_for_cell(cell, config, plan)

    hn_manifest: Optional[Path] = None
    hn_weight = 3.0
    hn_path_str = plan.get("campaign", {}).get("hard_negative_manifest")
    if hn_path_str and Path(hn_path_str).is_file():
        hn_manifest = Path(hn_path_str)
    elif hard_negative_manifest_path(cell_id, run_id=run_id).is_file():
        hn_manifest = hard_negative_manifest_path(cell_id, run_id=run_id)
    if hn_manifest:
        with open(hn_manifest, "r", encoding="utf-8") as f:
            hn_data = json.load(f)
        hn_weight = float(hn_data.get("uid_weight", hn_weight))

    bs_thr = float(plan.get("evaluation", {}).get("binary_softmax_threshold", DEFAULT_BINARY_SOFTMAX_THRESHOLD))
    bs_margin = float(plan.get("evaluation", {}).get("binary_logit_margin", DEFAULT_BINARY_LOGIT_MARGIN))

    results: List[Dict[str, Any]] = []
    wall_total = 0.0

    print(f"\n{'=' * 80}\nCelda {cell_id} | task={cell['task']} pose={cell['pose_source']}")
    if run_id:
        print(f"Run: {run_id}")
    print(f"{'=' * 80}")
    print(f"Plan: {plan_path}")
    print(f"Modelos: {models_dir}")
    if hn_manifest:
        print(f"Hard negatives: {hn_manifest}")
    print(f"Experimentos: {exp_list} | paralelo={max(1, int(parallel))}")

    if int(parallel) > 1 and not os.environ.get("CAMPAIGN_TRAIN_WORKER"):
        results, wall_total = _train_cell_parallel(
            cell,
            config,
            exp_list=exp_list,
            resume=resume,
            run_id=run_id,
            parallel=max(1, int(parallel)),
            config_path=config_path,
            arts=arts,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for exp_id in exp_list:
            if exp_id < 1 or exp_id > len(EXPERIMENTS):
                print(f"[Exp {exp_id:02d}] Fuera de rango EXPERIMENTS, omitido.")
                continue
            if resume and _model_exists(models_dir, exp_id):
                print(f"[Exp {exp_id:02d}] Ya existe, resume → omitido.")
                continue
            results.append(
                _train_one_experiment(
                    exp_id=exp_id,
                    cell=cell,
                    config=config,
                    plan=plan,
                    plan_path=plan_path,
                    models_dir=models_dir,
                    splits_dir=splits_dir,
                    train_opts=train_opts,
                    hn_manifest=hn_manifest,
                    hn_weight=hn_weight,
                    device=device,
                    bs_thr=bs_thr,
                    bs_margin=bs_margin,
                    run_id=run_id,
                )
            )
            wall_total += float(results[-1].get("wall_seconds") or 0.0)

    payload = {
        "cell_id": cell_id,
        "run_id": run_id,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "wall_seconds": wall_total,
        "parallel_jobs": max(1, int(parallel)),
        "experiment_ids": exp_list,
        "hard_negative_manifest": str(hn_manifest) if hn_manifest else None,
        "results": results,
    }
    if write_summary:
        summary_path = _write_summary(
            arts,
            cell_id=cell_id,
            run_id=run_id,
            wall_total=wall_total,
            exp_list=exp_list,
            hn_manifest=hn_manifest,
            results=results,
            parallel=max(1, int(parallel)),
        )
        print(f"\nResumen train → {summary_path} ({wall_total / 60:.1f} min)")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrenamiento por celdas de campaña")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="Todas las celdas del config")
    ap.add_argument(
        "--mass-all",
        action="store_true",
        help="Celdas de mass_augment.cells (pipeline run_mass_augment.sh; no mc_*_merge)",
    )
    ap.add_argument("--resume", action="store_true", help="Omite modelos ya entrenados")
    ap.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Experimentos a la vez en la misma GPU (default: parallelism.max_parallel_jobs del config, 2). 1 = serie.",
    )
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--job-result", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--exp-ids", type=int, nargs="*", default=None, help="Override experiment_ids")
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="ID de run (artefactos en artifacts/runs/<run-id>/)",
    )
    ap.add_argument(
        "--learning-curve",
        "--prediction",
        dest="learning_curve",
        action="store_true",
        help="Entrena cada tamaño de la curva (run_id lc_<N>)",
    )
    ap.add_argument(
        "--train-sizes",
        nargs="+",
        default=None,
        metavar="N|max",
        help="Tamaños train (entero o 'max'; sin flag → manifiesto del preflight)",
    )
    args = ap.parse_args()

    config = load_merged_campaign_config(Path(args.config) if args.config else None)
    config_path = Path(args.config) if args.config else None
    parallel = 1 if args.worker else _parallel_jobs(config, args.parallel)
    write_summary = not args.worker

    if args.learning_curve:
        from learning_curve_utils import (
            get_learning_curve_train_sizes,
            resolve_learning_curve_cells,
            run_id_for_train_size,
            experiment_ids_for_cell,
        )

        try:
            cells = resolve_learning_curve_cells(config, args.cells)
            train_sizes = get_learning_curve_train_sizes(
                cli_sizes=args.train_sizes,
                config=config,
            )
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        for cell in cells:
            cid = cell["id"]
            exp_ids = args.exp_ids or experiment_ids_for_cell(cell, config)
            for n in train_sizes:
                rid = run_id_for_train_size(n)
                print(f"\n{'#' * 80}\n# LC train — {cid} size={n} run_id={rid}\n{'#' * 80}")
                train_cell(
                    cell,
                    config,
                    resume=args.resume,
                    exp_ids=exp_ids,
                    run_id=rid,
                    parallel=parallel,
                    config_path=config_path,
                    write_summary=write_summary,
                )
        return 0

    run_id = str(args.run_id).strip() if args.run_id else None
    if args.mass_all:
        from campaign_paths import resolve_mass_cells

        cells = resolve_mass_cells(config, args.cells)
    elif args.cells:
        cells = filter_cells(config, args.cells)
    elif args.all:
        cells = filter_cells(config, None)
    else:
        print("Indica --all, --mass-all o --cells id1 id2 ...", file=sys.stderr)
        return 1

    if not cells:
        print("Indica --all, --mass-all o --cells id1 id2 ...", file=sys.stderr)
        return 1

    failed = False
    for cell in cells:
        cell_exp_ids = args.exp_ids or None
        payload = train_cell(
            cell,
            config,
            resume=args.resume,
            exp_ids=cell_exp_ids,
            run_id=run_id,
            parallel=parallel,
            config_path=config_path,
            write_summary=write_summary,
        )
        if args.job_result:
            results = payload.get("results") or []
            body = results[0] if results else payload
            out = Path(args.job_result)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(body, f, indent=2, ensure_ascii=False)
                f.write("\n")
        if any(isinstance(r, dict) and r.get("error") for r in payload.get("results") or []):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
