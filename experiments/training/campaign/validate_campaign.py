#!/usr/bin/env python3
"""
Comprueba que la campaña puede ejecutarse sin sorpresas (imports, datos, planes, GPU).

Uso:
  python validate_campaign.py
  python validate_campaign.py --data-root /ruta/data_result
  python validate_campaign.py --strict   # exige CUDA

Salida: 0 si todo OK, 1 si hay errores bloqueantes.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import py_compile
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    fatal: bool = True


@dataclass
class ValidationReport:
    results: List[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", *, fatal: bool = True) -> None:
        self.results.append(CheckResult(name, ok, detail, fatal))

    @property
    def passed(self) -> bool:
        return all(r.ok or not r.fatal for r in self.results)

    @property
    def blocking_failures(self) -> List[CheckResult]:
        return [r for r in self.results if not r.ok and r.fatal]


def _compile_py(path: Path) -> Optional[str]:
    try:
        py_compile.compile(str(path), doraise=True)
        return None
    except py_compile.PyCompileError as exc:
        return str(exc)


def check_python_syntax(report: ValidationReport) -> None:
    scripts = [
        CAMPAIGN_DIR / "preflight_campaign.py",
        CAMPAIGN_DIR / "train_campaign.py",
        CAMPAIGN_DIR / "evaluate_campaign.py",
        CAMPAIGN_DIR / "export_fp_artifacts.py",
        CAMPAIGN_DIR / "summarize_campaign.py",
        CAMPAIGN_DIR / "validate_campaign.py",
        CAMPAIGN_DIR / "campaign_paths.py",
        TRAINING_DIR / "class_map_utils.py",
        TRAINING_DIR / "preflight_train_plan.py",
        TRAINING_DIR / "train_model_operations.py",
        TRAINING_DIR / "evaluate_validation.py",
        TRAINING_DIR / "training_time_estimate.py",
    ]
    errors: List[str] = []
    for p in scripts:
        if not p.is_file():
            errors.append(f"Falta {p}")
            continue
        err = _compile_py(p)
        if err:
            errors.append(f"{p.name}: {err}")
    report.add(
        "Sintaxis Python (scripts campaña + core)",
        not errors,
        "; ".join(errors) if errors else f"{len(scripts)} ficheros OK",
    )


def check_third_party(report: ValidationReport) -> None:
    modules = ["torch", "numpy"]
    missing: List[str] = []
    versions: List[str] = []
    for mod in modules:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            versions.append(f"{mod}={ver}")
        except ImportError:
            missing.append(mod)
    report.add(
        "Dependencias Python (torch, numpy)",
        not missing,
        ", ".join(versions) if not missing else f"Faltan: {', '.join(missing)}",
    )


def check_campaign_imports(report: ValidationReport) -> None:
    """Importa los módulos que usarán train/eval (detecta NameError, imports rotos)."""
    blocks: List[tuple[str, Callable[[], None]]] = []

    def _campaign_paths():
        from campaign_paths import load_campaign_config, filter_cells  # noqa: F401

    def _train_stack():
        from model_config import EXPERIMENTS  # noqa: F401
        from train_model_operations import (  # noqa: F401
            run_experiment,
            collect_examples,
            build_model,
            build_datasets_and_loaders,
            AUGMENT_CONFIG_PATH,
        )

    def _eval_stack():
        from evaluate_validation import evaluate_validation  # noqa: F401

    def _preflight_stack():
        from preflight_train_plan import build_training_plan  # noqa: F401

    blocks = [
        ("campaign_paths", _campaign_paths),
        ("train_model_operations + model_config", _train_stack),
        ("evaluate_validation", _eval_stack),
        ("preflight_train_plan", _preflight_stack),
    ]
    errors: List[str] = []
    for label, fn in blocks:
        try:
            fn()
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    report.add(
        "Imports campaña (train / eval / preflight)",
        not errors,
        errors[0] if len(errors) == 1 else ("\n    ".join(errors) if errors else "OK"),
    )


def check_config(report: ValidationReport, config_path: Optional[Path]) -> Dict[str, Any]:
    try:
        from campaign_paths import load_campaign_config, filter_cells, class_map_path
        from model_config import EXPERIMENTS

        config = load_campaign_config(config_path)
        cells = filter_cells(config, None)
        exp_ids = list(config.get("experiment_ids", []))
        errors: List[str] = []
        if not cells:
            errors.append("cells vacío")
        for cid in exp_ids:
            if cid < 1 or cid > len(EXPERIMENTS):
                errors.append(f"experiment_id {cid} fuera de rango 1..{len(EXPERIMENTS)}")
        for cell in cells:
            try:
                class_map_path(cell["class_map_id"])
            except FileNotFoundError as exc:
                errors.append(str(exc))
            prof = cell.get("aug_profile")
            if prof not in (config.get("aug_profiles") or {}):
                errors.append(f"celda {cell['id']}: aug_profile {prof!r} desconocido")
        report.add(
            "campaign_config.json + class_maps",
            not errors,
            f"{len(cells)} celdas, exp_ids={exp_ids}" if not errors else "; ".join(errors),
        )
        return config
    except Exception as exc:
        report.add("campaign_config.json + class_maps", False, str(exc))
        return {}


def check_data_root(
    report: ValidationReport,
    data_root: Optional[Path],
    *,
    single_user_only: bool,
) -> None:
    try:
        from train_model_operations import collect_examples, get_data_result_root

        root = data_root or get_data_result_root()
        if not root.is_dir():
            report.add("data_result existe", False, f"No es directorio: {root}")
            return

        for ps in ("filtered", "full"):
            try:
                ex = collect_examples(
                    pose_source=ps,
                    single_user_only=single_user_only,
                    data_root=root,
                )
                n = len(ex)
                if n == 0:
                    report.add(
                        f"collect_examples pose_source={ps}",
                        False,
                        f"0 ejemplos en {root}",
                    )
                else:
                    report.add(
                        f"collect_examples pose_source={ps}",
                        True,
                        f"{n} ejemplos en {root}",
                    )
            except Exception as exc:
                report.add(f"collect_examples pose_source={ps}", False, str(exc))
    except Exception as exc:
        report.add("data_result / collect_examples", False, str(exc))


def check_augment_files(report: ValidationReport) -> None:
    try:
        from train_model_operations import AUGMENT_CONFIG_PATH

        p = Path(AUGMENT_CONFIG_PATH)
        ok = p.is_file()
        report.add(
            "validate_npy.json (augment)",
            ok,
            str(p) if ok else f"No encontrado: {p}",
        )
    except Exception as exc:
        report.add("validate_npy.json (augment)", False, str(exc))


def check_plans_and_augments(
    report: ValidationReport,
    config: Dict[str, Any],
    *,
    require_written: bool,
) -> None:
    if not config:
        return
    try:
        from campaign_paths import filter_cells, training_plan_path, category_aug_path

        cells = filter_cells(config, None)
        missing: List[str] = []
        for cell in cells:
            cid = cell["id"]
            plan_p = training_plan_path(cid)
            aug_p = category_aug_path(cid)
            if require_written:
                if not plan_p.is_file():
                    missing.append(f"{cid}: falta {plan_p.name}")
                elif not aug_p.is_file():
                    missing.append(f"{cid}: falta {aug_p.name}")
                else:
                    with open(plan_p, "r", encoding="utf-8") as f:
                        plan = json.load(f)
                    if not plan.get("split_uids"):
                        missing.append(f"{cid}: plan sin split_uids")
        report.add(
            "Planes preflight escritos (--write-all)",
            not missing,
            f"{len(cells)} celdas OK" if not missing else "; ".join(missing[:5])
            + (f" (+{len(missing)-5} más)" if len(missing) > 5 else ""),
            fatal=require_written,
        )
    except Exception as exc:
        report.add("Planes preflight", False, str(exc), fatal=require_written)


def check_preflight_smoke(
    report: ValidationReport,
    config: Dict[str, Any],
    data_root: Optional[Path],
) -> None:
    """Ejecuta build_training_plan en la 1ª celda (sin escribir) para detectar errores de runtime."""
    if not config:
        return
    try:
        from campaign_paths import filter_cells, class_map_path, category_aug_path
        from class_map_utils import load_class_map
        from preflight_train_plan import build_training_plan

        cells = filter_cells(config, None)
        if not cells:
            report.add("Preflight smoke (1 celda)", False, "Sin celdas")
            return
        cell = cells[0]
        class_map_spec = load_class_map(class_map_path(cell["class_map_id"]))
        exp_ids = list(config.get("experiment_ids", []))
        build_training_plan(
            task=cell["task"],
            pose_source=cell["pose_source"],
            single_user_only=bool(config.get("single_user_only", True)),
            data_root=data_root,
            category_aug_config=category_aug_path(cell["id"]),
            skip_time_estimate=True,
            class_map_spec=class_map_spec,
            experiment_ids=exp_ids if exp_ids else None,
        )
        report.add(
            "Preflight smoke (build_training_plan 1ª celda)",
            True,
            f"celda={cell['id']} OK",
        )
    except Exception as exc:
        tb = traceback.format_exc(limit=3)
        report.add(
            "Preflight smoke (build_training_plan 1ª celda)",
            False,
            f"{exc}\n    {tb.replace(chr(10), chr(10)+'    ')}",
        )


def check_model_forward_smoke(report: ValidationReport, config: Dict[str, Any]) -> None:
    """Construye un modelo de campaña y hace forward en CPU (tensor aleatorio)."""
    if not config:
        return
    try:
        import torch
        from model_config import EXPERIMENTS
        from train_model_operations import build_model

        exp_ids = list(config.get("experiment_ids", [6]))
        exp_id = exp_ids[0]
        cfg = dict(EXPERIMENTS[exp_id - 1])
        input_dim = 34 * 2 * 2  # orden de magnitud típico pose+velocity
        num_classes = 2 if any(c.get("task") == "binary" for c in config.get("cells", [])) else 15
        model = build_model(cfg["arch"], input_dim, num_classes, cfg)
        model.eval()
        seq_len = int(cfg.get("seq_len", 64))
        x = torch.randn(2, seq_len, input_dim)
        with torch.no_grad():
            y = model(x)
        report.add(
            "Forward smoke (build_model + inferencia CPU)",
            y.shape[0] == 2,
            f"exp {exp_id}, arch={cfg['arch']}, out={tuple(y.shape)}",
        )
    except Exception as exc:
        report.add("Forward smoke (build_model + inferencia CPU)", False, str(exc))


def check_cuda(report: ValidationReport, *, strict: bool) -> None:
    try:
        import torch

        avail = torch.cuda.is_available()
        if avail:
            name = torch.cuda.get_device_name(0)
            detail = f"CUDA OK — {name}"
        else:
            detail = "CUDA no disponible (entrenará en CPU, mucho más lento)"
        report.add("GPU / CUDA", avail or not strict, detail, fatal=strict)
    except Exception as exc:
        report.add("GPU / CUDA", False, str(exc), fatal=strict)


def check_artifacts_writable(report: ValidationReport) -> None:
    try:
        from campaign_paths import ARTIFACTS_ROOT, ensure_cell_dirs

        test_dir = ARTIFACTS_ROOT / "logs"
        test_dir.mkdir(parents=True, exist_ok=True)
        probe = test_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        ensure_cell_dirs("_validate_probe")
        report.add("Escritura en artifacts/", True, str(ARTIFACTS_ROOT))
    except Exception as exc:
        report.add("Escritura en artifacts/", False, str(exc))


def check_disk_space(report: ValidationReport, min_gb: float = 5.0) -> None:
    try:
        from campaign_paths import ARTIFACTS_ROOT

        stat = os.statvfs(ARTIFACTS_ROOT if ARTIFACTS_ROOT.exists() else CAMPAIGN_DIR)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        ok = free_gb >= min_gb
        report.add(
            "Espacio en disco",
            ok,
            f"~{free_gb:.1f} GB libres (mínimo recomendado {min_gb} GB)",
            fatal=False,
        )
    except Exception as exc:
        report.add("Espacio en disco", True, f"No comprobado: {exc}", fatal=False)


def run_validation(
    *,
    config_path: Optional[Path] = None,
    data_root: Optional[Path] = None,
    strict_cuda: bool = False,
    require_plans: bool = False,
    skip_smoke: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    check_python_syntax(report)
    check_third_party(report)
    check_campaign_imports(report)
    config = check_config(report, config_path)
    check_augment_files(report)
    check_artifacts_writable(report)
    check_disk_space(report)
    check_cuda(report, strict=strict_cuda)

    su = bool(config.get("single_user_only", True)) if config else True
    check_data_root(report, data_root, single_user_only=su)

    if not skip_smoke and report.passed:
        check_preflight_smoke(report, config, data_root)
        check_model_forward_smoke(report, config)

    check_plans_and_augments(report, config, require_written=require_plans)
    return report


def print_report(report: ValidationReport) -> None:
    print(f"\n{BOLD}{'=' * 72}")
    print("VALIDACIÓN CAMPAÑA")
    print(f"{'=' * 72}{RESET}\n")
    for r in report.results:
        icon = f"{GREEN}OK{RESET}" if r.ok else (f"{RED}FAIL{RESET}" if r.fatal else f"{YELLOW}WARN{RESET}")
        print(f"  [{icon}] {r.name}")
        if r.detail:
            for line in str(r.detail).splitlines():
                print(f"        {line}")

    fails = report.blocking_failures
    print(f"\n{BOLD}{'=' * 72}{RESET}")
    if report.passed:
        print(f"{GREEN}{BOLD}Resultado: LISTO para lanzar train/eval.{RESET}")
    else:
        print(f"{RED}{BOLD}Resultado: {len(fails)} error(es) bloqueante(s). No lances train hasta corregir.{RESET}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Validación previa de la campaña")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--strict", action="store_true", help="Exige CUDA (falla sin GPU)")
    ap.add_argument("--require-plans", action="store_true", help="Exige que preflight --write-all ya se ejecutó")
    ap.add_argument("--skip-smoke", action="store_true", help="Omite build_training_plan y forward smoke")
    args = ap.parse_args()

    report = run_validation(
        config_path=Path(args.config) if args.config else None,
        data_root=Path(args.data_root) if args.data_root else None,
        strict_cuda=args.strict,
        require_plans=args.require_plans,
        skip_smoke=args.skip_smoke,
    )
    print_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
