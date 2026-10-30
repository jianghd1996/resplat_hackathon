#!/usr/bin/env python3
"""Replace marked frames_info conversions with PyCOLMAP and paired undistorted images.

Dry run by default. Existing models are archived outside the scene before replacement.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "module_a"))
from preprocess import run_pycolmap  # noqa: E402


def source_images(scene):
    keep = set()
    for line in (scene / "frames_info.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            fields = line.split()
            if len(fields) >= 15:
                keep.add(fields[1] + ".jpg")
    paths = [scene / "images" / name for name in sorted(keep)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{scene}: missing referenced images: {missing[:5]}")
    if not paths:
        raise RuntimeError(f"{scene}: frames_info.txt selected no JPG images")
    return paths


def rebuild(scene, backup_root):
    images = source_images(scene)
    staging = Path(tempfile.mkdtemp(prefix="._hw_pycolmap_", dir=scene))
    try:
        result = run_pycolmap([str(p) for p in images], str(staging / "reconstruction"))
        model_stage = staging / "sparse" / "0"
        model_stage.parent.mkdir()
        shutil.copytree(result["model_dir"], model_stage)
        (model_stage / "pycolmap_undistorted.json").write_text(json.dumps({
            "source": "pycolmap_run", "input_images": len(images),
            "registered_images": len(result["poses"]),
            "pycolmap_images": "images_undistorted",
        }, indent=2))
        images_stage = staging / "images_undistorted"
        shutil.copytree(result["images_dir"], images_stage)
        missing = [name for name in result["poses"]
                   if not (images_stage / name).is_file()]
        if missing:
            raise RuntimeError(f"Missing undistorted registered images: {missing[:5]}")

        backup = backup_root / scene.name
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}")
        backup.mkdir(parents=True)
        old_sparse = scene / "sparse"
        old_images = scene / "images_undistorted"
        try:
            if old_sparse.exists():
                os.replace(old_sparse, backup / "sparse")
            if old_images.exists():
                os.replace(old_images, backup / "images_undistorted")
            os.replace(staging / "sparse", scene / "sparse")
            os.replace(images_stage, old_images)
        except Exception:
            # Restore the previous usable pair if installation fails halfway.
            if (backup / "sparse").exists():
                if (scene / "sparse").exists():
                    shutil.rmtree(scene / "sparse")
                os.replace(backup / "sparse", scene / "sparse")
            if (backup / "images_undistorted").exists():
                if old_images.exists():
                    shutil.rmtree(old_images)
                os.replace(backup / "images_undistorted", old_images)
            raise
        print(f"[OK] {scene.name}: {len(result['poses'])}/{len(images)} registered; "
              f"old sparse archived at {backup}")
    finally:
        shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent / "datasets/hw")
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="reconstruct and archive marked old models")
    parser.add_argument("scenes", nargs="*", help="scene names (default: all directories under root)")
    args = parser.parse_args()
    root = args.root.resolve()
    scenes = [root / name for name in args.scenes] if args.scenes else sorted(p for p in root.iterdir() if p.is_dir())
    for scene in scenes:
        marker = scene / "sparse/0/conversion_summary.json"
        if not (scene / "frames_info.txt").is_file():
            print(f"[SKIP] {scene.name}: no frames_info.txt")
        elif not marker.is_file():
            print(f"[SKIP] {scene.name}: sparse/0 is not marked as a direct frames_info conversion")
        elif args.apply:
            rebuild(scene, args.backup_dir.resolve())
        else:
            print(f"[PLAN] {scene.name}: archive direct conversion, rebuild from "
                  f"{len(source_images(scene))} original JPGs, save undistorted images")
    if not args.apply:
        print("Dry run only; pass --apply to run reconstruction")


if __name__ == "__main__":
    main()
