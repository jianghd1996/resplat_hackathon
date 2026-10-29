#!/usr/bin/env python3
"""Batch sparse reconstruction for datasets/hw scenes with PyCOLMAP.

Each scene is expected to have:
    <root>/<scene>/images/*.{jpg,jpeg,png,...}

The best sparse model (the one with the most registered images) is written to:
    <root>/<scene>/sparse/0/{cameras,images,points3D}.{bin,txt}

Examples:
    # All scenes under datasets/hw
    python baseline/run_pycolmap_hw.py

    # One scene
    python baseline/run_pycolmap_hw.py basketball

    # Rebuild existing output
    python baseline/run_pycolmap_hw.py basketball --overwrite

PyCOLMAP installation:
    python -m pip install pycolmap
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

try:
    import pycolmap
except ImportError as exc:
    raise SystemExit(
        "PyCOLMAP is not installed in this Python environment.\n"
        "Install it with: python -m pip install pycolmap"
    ) from exc


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def image_names(image_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def pycolmap_device(name: str) -> Any:
    aliases = {
        "auto": "auto",
        "cpu": "cpu",
        "cuda": "cuda",
    }
    try:
        return getattr(pycolmap.Device, aliases[name])
    except AttributeError as exc:
        raise RuntimeError(
            f"PyCOLMAP {getattr(pycolmap, '__version__', 'unknown')} "
            f"does not expose Device.{aliases[name]}"
        ) from exc


def extract_features(
    database_path: Path,
    image_dir: Path,
    camera_model: str,
    device: Any,
) -> None:
    """Call the current API, with a fallback for PyCOLMAP 0.6/3.11."""
    try:
        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = camera_model
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            device=device,
        )
    except TypeError:
        # Older releases accepted camera_model directly.
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model=camera_model,
            device=device,
        )


def match_features(
    database_path: Path,
    matcher: str,
    device: Any,
    sequential_overlap: int,
) -> None:
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(
            database_path=database_path,
            device=device,
        )
        return

    try:
        pairing_options = pycolmap.SequentialPairingOptions()
        pairing_options.overlap = sequential_overlap
        if hasattr(pairing_options, "quadratic_overlap"):
            pairing_options.quadratic_overlap = True
        pycolmap.match_sequential(
            database_path=database_path,
            pairing_options=pairing_options,
            device=device,
        )
    except TypeError:
        # Compatibility with older releases.
        pycolmap.match_sequential(
            database_path=database_path,
            overlap=sequential_overlap,
            quadratic_overlap=True,
            device=device,
        )


def reconstruction_size(reconstruction: Any) -> int:
    images = reconstruction.images
    return len(images() if callable(images) else images)


def reconstruct_scene(scene_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    image_dir = scene_dir / "images"
    output_dir = scene_dir / "sparse" / "0"
    work_dir = scene_dir / ".pycolmap"
    database_path = work_dir / "database.db"
    candidate_dir = work_dir / "models"

    names = image_names(image_dir)
    if len(names) < 2:
        raise RuntimeError(f"{image_dir}: need at least 2 supported images")

    if output_dir.exists() and not args.overwrite:
        if (output_dir / "cameras.bin").exists() and (output_dir / "images.bin").exists():
            print(f"[SKIP] {scene_dir.name}: existing model at {output_dir}")
            return {
                "scene": scene_dir.name,
                "status": "skipped",
                "num_input_images": len(names),
                "output": str(output_dir),
            }
        raise RuntimeError(
            f"{output_dir} exists but is incomplete; use --overwrite to rebuild"
        )

    if args.overwrite:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if work_dir.exists():
            shutil.rmtree(work_dir)
    elif work_dir.exists():
        # Do not silently reuse a possibly partial database.
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    matcher = args.matcher
    if matcher == "auto":
        matcher = "exhaustive" if len(names) <= args.exhaustive_max_images else "sequential"

    print()
    print("=" * 72)
    print(f"[START] {scene_dir.name}")
    print(f"  images:       {len(names)} ({image_dir})")
    print(f"  camera model: {args.camera_model}, single camera")
    print(f"  matcher:      {matcher}")
    print(f"  device:       {args.device}")
    print(f"  output:       {output_dir}")
    print("=" * 72, flush=True)

    stage_times: dict[str, float] = {}
    t_total = time.perf_counter()
    device = pycolmap_device(args.device)

    t0 = time.perf_counter()
    extract_features(database_path, image_dir, args.camera_model, device)
    stage_times["feature_extraction_seconds"] = time.perf_counter() - t0
    print(f"[{scene_dir.name}] feature extraction: "
          f"{stage_times['feature_extraction_seconds']:.1f}s", flush=True)

    t0 = time.perf_counter()
    match_features(database_path, matcher, device, args.sequential_overlap)
    stage_times["matching_seconds"] = time.perf_counter() - t0
    print(f"[{scene_dir.name}] matching: "
          f"{stage_times['matching_seconds']:.1f}s", flush=True)

    t0 = time.perf_counter()
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=candidate_dir,
    )
    stage_times["mapping_seconds"] = time.perf_counter() - t0

    if not reconstructions:
        raise RuntimeError(
            f"{scene_dir.name}: incremental mapping produced no reconstruction"
        )

    candidates = [
        (int(index), reconstruction, reconstruction_size(reconstruction))
        for index, reconstruction in reconstructions.items()
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)
    best_index, best, num_registered = candidates[0]

    output_dir.mkdir(parents=True, exist_ok=False)
    best.write(output_dir)
    if args.write_text:
        best.write_text(output_dir)

    stage_times["total_seconds"] = time.perf_counter() - t_total
    summary = {
        "scene": scene_dir.name,
        "status": "ok",
        "pycolmap_version": getattr(pycolmap, "__version__", "unknown"),
        "camera_model": args.camera_model,
        "camera_mode": "SINGLE",
        "matcher": matcher,
        "device": args.device,
        "num_input_images": len(names),
        "num_models": len(candidates),
        "selected_model_index": best_index,
        "num_registered_images": num_registered,
        "registration_ratio": num_registered / len(names),
        "candidate_registered_images": {
            str(index): count for index, _, count in candidates
        },
        "output": str(output_dir),
        **{key: round(value, 3) for key, value in stage_times.items()},
    }
    with open(scene_dir / "pycolmap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.cleanup:
        shutil.rmtree(work_dir)

    print(
        f"[DONE] {scene_dir.name}: {num_registered}/{len(names)} registered "
        f"({100.0 * num_registered / len(names):.1f}%), "
        f"{stage_times['total_seconds']:.1f}s -> {output_dir}",
        flush=True,
    )
    return summary


def discover_scenes(root: Path, selected: list[str]) -> list[Path]:
    if selected:
        scenes = [root / name for name in selected]
    else:
        scenes = sorted(
            path for path in root.iterdir()
            if path.is_dir() and (path / "images").is_dir()
        )

    missing = [str(path) for path in scenes if not (path / "images").is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing scene images directories:\n  " + "\n  ".join(missing)
        )
    return scenes


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_root = repo_root / "datasets" / "hw"

    parser = argparse.ArgumentParser(
        description="Build sparse/0 for datasets/hw scenes with PyCOLMAP"
    )
    parser.add_argument(
        "scenes", nargs="*",
        help="scene names below --root; omit to process every scene with images/"
    )
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--camera-model", default="OPENCV",
        choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"],
    )
    parser.add_argument(
        "--matcher", choices=["auto", "exhaustive", "sequential"], default="auto"
    )
    parser.add_argument("--exhaustive-max-images", type=int, default=120)
    parser.add_argument("--sequential-overlap", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing sparse/0 and temporary PyCOLMAP state"
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="delete .pycolmap database and candidate models after success"
    )
    parser.add_argument(
        "--no-write-text", dest="write_text", action="store_false",
        help="only write COLMAP binary files"
    )
    parser.set_defaults(write_text=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HW dataset root does not exist: {root}")

    scenes = discover_scenes(root, args.scenes)
    if not scenes:
        print(f"No scenes containing images/ found under {root}")
        return 0

    print(f"PyCOLMAP version: {getattr(pycolmap, '__version__', 'unknown')}")
    print(f"Dataset root: {root}")
    print("Scenes: " + ", ".join(path.name for path in scenes))

    results = []
    failures = []
    for scene_dir in scenes:
        try:
            results.append(reconstruct_scene(scene_dir, args))
        except Exception as exc:  # continue the batch and report every failure
            failures.append({"scene": scene_dir.name, "error": repr(exc)})
            print(f"[FAILED] {scene_dir.name}: {exc}", file=sys.stderr, flush=True)

    batch_summary = {
        "root": str(root),
        "pycolmap_version": getattr(pycolmap, "__version__", "unknown"),
        "results": results,
        "failures": failures,
    }
    with open(root / "pycolmap_batch_summary.json", "w") as f:
        json.dump(batch_summary, f, indent=2)

    print()
    print(f"Finished: {len(results)} succeeded/skipped, {len(failures)} failed")
    print(f"Summary: {root / 'pycolmap_batch_summary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
