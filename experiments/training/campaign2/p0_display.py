"""Tablas ANSI para informe P0."""
from __future__ import annotations

from typing import Any, Dict, List


BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _fmt_metrics(m: Dict[str, Any]) -> str:
    return (
        f"F1={m.get('f1_pct', 0):.1f}% "
        f"Rec={m.get('recall_pct', 0):.1f}% "
        f"Prec={m.get('precision_pct', 0):.1f}% "
        f"FP={m.get('fp', 0)} FN={m.get('fn', 0)} "
        f"(neg={m.get('support_neg', '?')})"
    )


def render_p0_baseline(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"{BOLD}{'=' * 72}{RESET}")
    lines.append(f"{BOLD}Campaign2 P0 baseline — run {report.get('run_id')}{RESET}")
    lines.append(f"Split eval: {CYAN}{report.get('eval_split')}{RESET} | jobs OK: {report.get('num_jobs_ok', 0)}")
    lines.append(f"{BOLD}{'=' * 72}{RESET}")

    jobs = report.get("per_model_jobs") or []
    if not jobs:
        lines.append(f"{YELLOW}Sin resultados p0_result.json — ejecuta train tras preflight --p0{RESET}")
    else:
        lines.append(f"\n{BOLD}Por modelo (umbral congelado en cada job):{RESET}")
        for j in jobs:
            mode = "P0.A clip" if j.get("mode") == "legacy_clip" else "P0.B ventanas"
            su = "single_user" if j.get("single_user_only") else "multi_user"
            m = j.get("metrics_at_frozen_threshold") or {}
            thr = j.get("frozen_threshold", 0.8)
            lines.append(
                f"  {CYAN}{j.get('job_id')}{RESET} [{mode}, {su}] "
                f"m{j.get('model_id')} @ {thr:.2f} → {GREEN}{_fmt_metrics(m)}{RESET}"
            )

    ens = report.get("ensembles") or []
    if ens:
        lines.append(f"\n{BOLD}Ensemble AND (umbrales congelados):{RESET}")
        for e in ens:
            mode = "P0.A" if e.get("mode") == "legacy_clip" else "P0.B"
            su = "single_user" if e.get("single_user_only") else "multi_user"
            th = e.get("thresholds") or {}
            lines.append(
                f"  {BOLD}{mode}{RESET} {su} #41@{th.get('41', 0.8):.2f} AND #48@{th.get('48', 0.8):.2f} → "
                f"{GREEN}{_fmt_metrics(e.get('metrics') or {})}{RESET}"
            )
    lines.append(f"{BOLD}{'=' * 72}{RESET}")
    return "\n".join(lines)
