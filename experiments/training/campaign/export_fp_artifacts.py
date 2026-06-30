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
    from train_model_operations import PoseExample
except ImportError:
    PoseExample = Any  # type: ignore[misc,assignment]


def clip_path_from_example(ex: PoseExample) -> str:
    clip_dir = ex.pose_path.parent.parent
    clip_video = clip_dir / "clip.mp4"
    if clip_video.is_file():
        return str(clip_video.resolve())
    return str(clip_dir.resolve())


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
        clip_path = Path(str(rec.get("clip_path", "")))
        out_sub = dest_dir / uid.replace("/", "_")[:120]
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
            "prob_pos": rec.get("prob_pos"),
            "clip_path": str(clip_path),
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
            f.write(f"{item['clip_path']}\tcat={item.get('folder_category')}\tp={item.get('prob_pos')}\n")

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
