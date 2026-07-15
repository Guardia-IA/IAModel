#!/usr/bin/env python3
"""Exporta vídeos / rutas de clips con falsos positivos para revisión manual."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

try:
    from train_model_operations import PoseExample, _example_uid, _example_folder_category
except ImportError:
    PoseExample = Any  # type: ignore[misc,assignment]

    def _example_uid(ex: Any) -> str:  # type: ignore[misc]
        return str(getattr(ex, "pose_path", "unknown"))

    def _example_folder_category(ex: Any) -> int:  # type: ignore[misc]
        return int(getattr(ex, "label", 0))


def example_export_paths(ex: PoseExample) -> Dict[str, str]:
    """
    Rutas absolutas para localizar el clip y el vídeo en disco.
    Estructura típica: data_result/{cat}/{clip}/user_X/poses_full.npy
    """
    pose_path = Path(ex.pose_path).resolve()
    user_dir = pose_path.parent
    clip_dir = user_dir.parent
    clip_video = clip_dir / "clip.mp4"
    meta_json = clip_dir / "meta.json"
    valid_mask_s = ""
    vm = getattr(ex, "valid_mask_path", None)
    if vm is not None and Path(vm).is_file():
        valid_mask_s = str(Path(vm).resolve())
    elif pose_path.name == "poses_full.npy" and (user_dir / "valid_mask.npy").is_file():
        valid_mask_s = str((user_dir / "valid_mask.npy").resolve())

    video_path = clip_video.resolve() if clip_video.is_file() else None
    return {
        "uid": _example_uid(ex),
        "uid_absolute": str(pose_path),
        "clip_name": str(getattr(ex, "clip_name", clip_dir.name)),
        "category_str": str(getattr(ex, "category_str", "")),
        "folder_category": str(_example_folder_category(ex)),
        "clip_dir": str(clip_dir.resolve()),
        "clip_video_path": str(video_path) if video_path else "",
        "clip_video_exists": "1" if video_path else "0",
        "meta_json_path": str(meta_json.resolve()) if meta_json.is_file() else "",
        "pose_path": str(pose_path),
        "user_dir": str(user_dir.resolve()),
        "valid_mask_path": valid_mask_s,
    }


def clip_path_from_example(ex: PoseExample) -> str:
    """Ruta al vídeo si existe; si no, carpeta del clip (absoluta)."""
    paths = example_export_paths(ex)
    if paths["clip_video_path"]:
        return paths["clip_video_path"]
    return paths["clip_dir"]


def export_fp_from_records(
    records: List[Dict[str, Any]],
    dest_dir: Path,
    *,
    cell_id: str,
    copy_videos: bool = True,
    use_symlink: bool = True,
) -> Path:
    """
    Copia o enlaza clip.mp4 de cada FP a dest_dir/{uid}/.
    Escribe fp_export_index.json con metadatos.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    exported: List[Dict[str, Any]] = []

    for rec in records:
        uid = str(rec.get("uid", "unknown"))
        clip_video = str(rec.get("clip_video_path") or rec.get("clip_path") or "")
        clip_path = Path(clip_video) if clip_video else Path(str(rec.get("clip_dir", "")))
        out_sub = dest_dir / uid.replace("/", "_").replace("\\", "_")[:120]
        out_sub.mkdir(parents=True, exist_ok=True)

        meta_path = out_sub / "fp_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
            f.write("\n")

        video_dest = out_sub / "clip.mp4"
        if clip_path.is_file() and copy_videos:
            if use_symlink:
                if video_dest.exists() or video_dest.is_symlink():
                    video_dest.unlink(missing_ok=True)
                video_dest.symlink_to(clip_path.resolve())
            else:
                shutil.copy2(clip_path, video_dest)

        exported.append({
            "uid": uid,
            "folder_category": rec.get("folder_category"),
            "clip_name": rec.get("clip_name"),
            "prob_pos": rec.get("prob_pos") or rec.get("p_mean"),
            "clip_video_path": clip_video or str(clip_path),
            "clip_dir": rec.get("clip_dir"),
            "export_dir": str(out_sub),
            "model": rec.get("model_path"),
        })

    index_path = dest_dir / "fp_export_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"cell_id": cell_id, "count": len(exported), "items": exported}, f, indent=2)
        f.write("\n")

    playlist = dest_dir / "fp_playlist.txt"
    with open(playlist, "w", encoding="utf-8") as f:
        for item in exported:
            f.write(
                f"{item['clip_video_path']}\t"
                f"cat={item.get('folder_category')}\t"
                f"clip={item.get('clip_name')}\t"
                f"p={item.get('prob_pos')}\n"
            )

    return index_path


def export_from_manifest_csv(
    manifest_csv: Path,
    dest_dir: Path,
    *,
    cell_id: str,
    max_items: int = 50,
) -> Path:
    rows: List[Dict[str, Any]] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_items:
                break
            rows.append(dict(row))
    return export_fp_from_records(rows, dest_dir, cell_id=cell_id)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Export FP clips desde manifest CSV")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--dest", type=str, required=True)
    ap.add_argument("--cell-id", type=str, default="manual")
    ap.add_argument("--max", type=int, default=50)
    ap.add_argument("--copy", action="store_true", help="Copiar en lugar de symlink")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= args.max:
                break
            rows.append(dict(row))

    path = export_fp_from_records(
        rows,
        Path(args.dest),
        cell_id=args.cell_id,
        use_symlink=not args.copy,
    )
    print(f"Exportado: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
