#!/usr/bin/env python3
"""
Diagnóstico offline: solape chunk k-1 / k, saltos de pose, máscara y score ensemble.

Uso (tus ficheros en Descargas/merged):

  cd IAModel/experiments/training/campaign
  python3 diagnose_chunk_overlap.py \\
    --poses-a /home/debian/Descargas/merged/poses_full1.npy \\
    --mask-a  /home/debian/Descargas/merged/valid_mask1.npy \\
    --poses-b /home/debian/Descargas/merged/poses_full2.npy \\
    --mask-b  /home/debian/Descargas/merged/valid_mask2.npy \\
    --model-a /home/debian/Documentos/Kanvis/modelos_v2/modelo_48.pt \\
    --model-b /home/debian/Documentos/Kanvis/modelos_v2/modelo_49.pt

Opcional: --plot out.png
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CAMPAIGN_DIR = Path(__file__).resolve().parent


def nanfix(poses: np.ndarray) -> np.ndarray:
    return np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def visible_joints(frame: np.ndarray) -> int:
    return int(np.sum(np.linalg.norm(frame, axis=1) > 1e-5))


def frame_l2_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((b - a).reshape(-1)))


def build_half_overlap(
    poses_a: np.ndarray, mask_a: np.ndarray | None,
    poses_b: np.ndarray, mask_b: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    ha, hb = poses_a.shape[0] // 2, poses_b.shape[0] // 2
    overlap = np.concatenate([poses_a[ha:], poses_b[:hb]], axis=0)
    vm: np.ndarray | None = None
    if mask_a is not None and mask_b is not None:
        vm = np.concatenate([mask_a[ha:], mask_b[:hb]], axis=0)
    return overlap, vm


def discontinuity_report(poses: np.ndarray, mask: np.ndarray | None, stitch: int) -> dict:
    diffs = np.array([frame_l2_diff(poses[i], poses[i + 1]) for i in range(len(poses) - 1)])
    cent = poses.mean(axis=1)
    cdiff = np.linalg.norm(np.diff(cent, axis=0), axis=1)
    stitch_pose = diffs[stitch] if 0 <= stitch < len(diffs) else float("nan")
    stitch_cent = float(cdiff[stitch]) if 0 <= stitch < len(cdiff) else float("nan")
    med = float(np.median(diffs)) if len(diffs) else 0.0
    ratio = stitch_pose / med if med > 1e-9 else float("inf")
    inv = np.where(~mask.astype(bool))[0].tolist() if mask is not None else []
    vj_before = visible_joints(poses[stitch]) if stitch < len(poses) else 0
    vj_after = visible_joints(poses[stitch + 1]) if stitch + 1 < len(poses) else 0
    return {
        "stitch_index": stitch,
        "diff_median": med,
        "diff_max": float(diffs.max()) if len(diffs) else 0.0,
        "diff_max_frame": int(diffs.argmax()) if len(diffs) else -1,
        "stitch_pose_l2": stitch_pose,
        "stitch_pose_vs_median": ratio,
        "stitch_centroid_jump": stitch_cent,
        "visible_joints_stitch": [vj_before, vj_after],
        "invalid_mask_indices": inv,
    }


def apply_mask(poses: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return poses
    return poses[mask.astype(bool)].copy()


def teleport_after_mask(poses: np.ndarray, mask: np.ndarray | None) -> dict | None:
    """Salto tras eliminar frames inválidos consecutivos (típico inicio chunk B)."""
    if mask is None or mask.shape[0] != poses.shape[0]:
        return None
    vm = mask.astype(bool)
    if vm.all():
        return None
    filtered = poses[vm]
    fd = np.array([frame_l2_diff(filtered[i], filtered[i + 1]) for i in range(len(filtered) - 1)])
    if len(fd) == 0:
        return None
    i = int(fd.argmax())
    return {
        "max_jump_after_filter_frame": i,
        "max_jump_after_filter_l2": float(fd[i]),
        "median_after_filter": float(np.median(fd)),
    }


def run_score(
    poses_path: Path,
    mask_path: Path | None,
    use_mask: bool,
    model_a: Path,
    model_b: Path,
    threshold: float,
) -> dict:
    cmd = [
        sys.executable,
        str(CAMPAIGN_DIR / "score_pose_ensemble.py"),
        "--poses-full",
        str(poses_path),
        "--model-a",
        str(model_a),
        "--model-b",
        str(model_b),
        "--rule",
        "mean",
        "--threshold",
        str(threshold),
        "--json",
    ]
    if mask_path is not None:
        cmd.extend(["--valid-mask", str(mask_path)])
    if not use_mask:
        cmd.append("--no-valid-mask-filter")
    raw = subprocess.check_output(cmd, cwd=str(CAMPAIGN_DIR))
    return json.loads(raw)


def suggest_gates(rep: dict, teleport: dict | None, scores: dict) -> list[str]:
    hints: list[str] = []
    if rep["stitch_pose_vs_median"] > 8.0:
        hints.append(
            f"GATE stitch_jump: ratio {rep['stitch_pose_vs_median']:.1f}x mediana "
            "(>8) → no inferir solape o marcar low_confidence."
        )
    if rep["visible_joints_stitch"][0] >= 7 and rep["visible_joints_stitch"][1] <= 5:
        hints.append(
            "GATE visibility_drop en costura: posible cambio de track / persona distinta."
        )
    if teleport and teleport["max_jump_after_filter_l2"] > 3 * teleport["median_after_filter"]:
        hints.append(
            "GATE post_mask_teleport: tras filtrar máscara hay salto grande "
            "(frames malos eliminados entre dos poses válidas lejanas)."
        )
    s_chunk = scores.get("chunks_low", False)
    s_ov = scores.get("overlap_with_mask", {}).get("robbery_score", 0.0)
    if s_chunk and s_ov >= scores.get("threshold", 0.5):
        hints.append(
            "GATE chunk_consistency: chunks A y B con score bajo pero solape alto "
            "→ tratar solape como artefacto (no alarma stage-1)."
        )
    if not hints:
        hints.append("Sin regla automática disparada; revisar vídeo y timestamps.")
    return hints


def maybe_plot(poses: np.ndarray, mask: np.ndarray | None, stitch: int, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib no instalado; omito --plot)", file=sys.stderr)
        return

    raw_diff = np.array([frame_l2_diff(poses[i], poses[i + 1]) for i in range(len(poses) - 1)])
    cent = poses.mean(axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = np.arange(len(poses))
    axes[0].plot(t, cent[:, 0], label="cx")
    axes[0].plot(t, cent[:, 1], label="cy")
    axes[0].axvline(stitch + 0.5, color="red", linestyle="--", alpha=0.7, label="costura")
    axes[0].set_ylabel("centroide")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Grilla densa del solape (antes de filtrar máscara)")

    axes[1].plot(np.arange(len(raw_diff)), raw_diff, color="tab:orange")
    axes[1].axvline(stitch, color="red", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Δ pose L2")

    if mask is not None:
        filt = poses[mask.astype(bool)]
        fcent = filt.mean(axis=1)
        axes[2].plot(np.arange(len(fcent)), fcent[:, 0], label="cx filtrado")
        axes[2].plot(np.arange(len(fcent)), fcent[:, 1], label="cy filtrado")
        axes[2].set_ylabel("centroide tras mask")
        axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_xlabel("índice de frame")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Gráfico guardado: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnóstico solape entre chunks")
    ap.add_argument("--poses-a", type=Path, required=True)
    ap.add_argument("--poses-b", type=Path, required=True)
    ap.add_argument("--mask-a", type=Path, default=None)
    ap.add_argument("--mask-b", type=Path, default=None)
    ap.add_argument("--model-a", type=Path, required=True)
    ap.add_argument("--model-b", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--tmpdir", type=Path, default=Path("/tmp"))
    ap.add_argument("--plot", type=Path, default=None, help="PNG con centroides y saltos")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    pa = nanfix(np.load(args.poses_a))
    pb = nanfix(np.load(args.poses_b))
    ma = np.load(args.mask_a) if args.mask_a and args.mask_a.is_file() else None
    mb = np.load(args.mask_b) if args.mask_b and args.mask_b.is_file() else None

    overlap, vm = build_half_overlap(pa, ma, pb, mb)
    stitch = pa.shape[0] // 2 - 1  # último índice mitad A en solape

    rep = discontinuity_report(overlap, vm, stitch)
    tel = teleport_after_mask(overlap, vm)

    tmp_p = args.tmpdir / "diagnose_overlap_poses.npy"
    tmp_m = args.tmpdir / "diagnose_overlap_mask.npy"
    np.save(tmp_p, overlap)
    if vm is not None:
        np.save(tmp_m, vm)

    scores: dict = {"threshold": args.threshold}
    for tag, p, m in (
        ("chunk_a", args.poses_a, args.mask_a),
        ("chunk_b", args.poses_b, args.mask_b),
    ):
        if m is not None and Path(m).is_file():
            scores[tag] = run_score(p, m, True, args.model_a, args.model_b, args.threshold)
    if vm is not None:
        scores["overlap_with_mask"] = run_score(tmp_p, tmp_m, True, args.model_a, args.model_b, args.threshold)
        scores["overlap_no_mask_filter"] = run_score(tmp_p, tmp_m, False, args.model_a, args.model_b, args.threshold)
        ca = scores.get("chunk_a", {}).get("robbery_score", 0.0)
        cb = scores.get("chunk_b", {}).get("robbery_score", 0.0)
        scores["chunks_low"] = ca < args.threshold and cb < args.threshold

    gates = suggest_gates(rep, tel, scores)

    payload = {
        "poses_a": str(args.poses_a.resolve()),
        "poses_b": str(args.poses_b.resolve()),
        "overlap_T": int(overlap.shape[0]),
        "stitch": rep,
        "teleport_after_valid_mask": tel,
        "scores": scores,
        "suggested_gates": gates,
        "note_single_skeleton": (
            "Cada poses_full es UN track (una persona). El solape concatena fin(A)+inicio(B); "
            "en vídeo puede parecer dos personas andando, pero el modelo ve UNA secuencia "
            "con discontinuidades que generan velocidades atípicas."
        ),
    }

    if args.plot:
        maybe_plot(overlap, vm, stitch, args.plot)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("=== Solape mitad+mitad ===")
        print(f"T={payload['overlap_T']}  costura en frame {stitch}->{stitch + 1}")
        print(
            f"Salto L2 costura: {rep['stitch_pose_l2']:.4f}  "
            f"(×{rep['stitch_pose_vs_median']:.1f} mediana)  "
            f"joints visibles: {rep['visible_joints_stitch']}"
        )
        if tel:
            print(
                f"Tras máscara: max salto en frame {tel['max_jump_after_filter_frame']} "
                f"L2={tel['max_jump_after_filter_l2']:.4f}"
            )
        if "overlap_with_mask" in scores:
            ow = scores["overlap_with_mask"]
            on = scores["overlap_no_mask_filter"]
            print(
                f"Score MEAN con máscara: {ow['robbery_score']:.4f} alarm={ow['robbery_alarm']}  |  "
                f"sin filtro: {on['robbery_score']:.4f} alarm={on['robbery_alarm']}"
            )
        print("\n--- Reglas sugeridas ---")
        for g in gates:
            print(f"  • {g}")
        print(f"\n{payload['note_single_skeleton']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())