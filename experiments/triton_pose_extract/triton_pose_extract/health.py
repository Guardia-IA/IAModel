"""Comprobaciones Triton (HTTP ready + opcional smoke gRPC)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from triton_pose_extract.settings import ExtractSettings


def check_model_ready(settings: ExtractSettings, *, timeout: float = 10.0) -> dict[str, Any]:
    model = settings.model_name
    url = f"{settings.http_health_base}/v2/models/{model}/ready"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = resp.status == 200
            body = resp.read(256).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return {"ok": False, "url": url, "status": e.code, "error": str(e)}
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e)}
    return {"ok": ok, "url": url, "status": 200, "body": body.strip()}


def smoke_infer_optional(settings: ExtractSettings) -> dict[str, Any]:
    """Ejecuta check_triton_pose_batch.py del vendor si está disponible."""
    script = settings.triton_pose_repo / "scripts" / "check_triton_pose_batch.py"
    if not script.is_file():
        return {"skipped": True, "reason": "sin check_triton_pose_batch.py"}
    import subprocess
    import sys

    cmd = [
        sys.executable,
        str(script),
        "--url",
        settings.triton_url,
        "--model",
        settings.model_name,
        "--no-config",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(120, settings.infer_timeout_seconds))
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def print_health(settings: ExtractSettings) -> int:
    rep = check_model_ready(settings)
    print(json.dumps({"ready": rep, "triton_url": settings.triton_url}, indent=2))
    if not rep.get("ok"):
        return 1
    return 0
