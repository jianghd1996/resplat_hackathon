#!/usr/bin/env python3
"""Diagnose how an HW frames_info pose dump relates to a COLMAP model.

This script is read-only with respect to the dataset. It matches images by
timestamp filename, tries several quaternion/axis conventions, aligns the two
world frames, and reports rotation and camera-center errors.

Example::

    python baseline/diagnose_frames_info_pose.py \
        --scene datasets/hw/basketball \
        --colmap datasets/hw/basketball/sparse_0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pycolmap
except ImportError as exc:
    raise SystemExit("Install PyCOLMAP first: python -m pip install pycolmap") from exc


def quat_xyzw_to_rotation(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    x, y, z, w = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_frames_info(path: Path) -> dict[str, dict[str, Any]]:
    frames: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 15:
                raise ValueError(
                    f"{path}:{line_no}: expected 15 fields, got {len(fields)}"
                )
            timestamp = fields[1]
            if timestamp in frames:
                raise ValueError(f"{path}:{line_no}: duplicate timestamp {timestamp}")
            frames[timestamp] = {
                "frame": int(fields[0]),
                "timestamp": timestamp,
                "fx": float(fields[2]),
                "fy": float(fields[3]),
                "cx": float(fields[4]),
                "cy": float(fields[5]),
                "width": int(float(fields[6])),
                "height": int(float(fields[7])),
                "center": np.asarray([float(v) for v in fields[8:11]]),
                "rotation_raw": quat_xyzw_to_rotation(
                    np.asarray([float(v) for v in fields[11:15]])
                ),
            }
    if not frames:
        raise RuntimeError(f"no frames found in {path}")
    return frames


def reconstruction_images(reconstruction: Any) -> list[Any]:
    images = reconstruction.images
    if callable(images):
        images = images()
    return list(images.values()) if hasattr(images, "values") else list(images)


def rigid_rotation_matrix(rigid: Any) -> np.ndarray:
    rotation = rigid.rotation
    if callable(rotation):
        rotation = rotation()
    matrix = rotation.matrix
    if callable(matrix):
        matrix = matrix()
    return np.asarray(matrix, dtype=np.float64)


def rigid_translation(rigid: Any) -> np.ndarray:
    translation = rigid.translation
    if callable(translation):
        translation = translation()
    return np.asarray(translation, dtype=np.float64).reshape(3)


def image_pose(image: Any) -> tuple[np.ndarray, np.ndarray]:
    rigid = image.cam_from_world
    if callable(rigid):
        rigid = rigid()
    rotation = rigid_rotation_matrix(rigid)
    translation = rigid_translation(rigid)
    center = -(rotation.T @ translation)
    return rotation, center


def find_models(root: Path) -> list[tuple[Path, Any, int]]:
    candidates: list[Path] = []
    if (root / "cameras.bin").is_file() or (root / "cameras.txt").is_file():
        candidates.append(root)
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        if (path / "cameras.bin").is_file() or (path / "cameras.txt").is_file():
            candidates.append(path)

    models = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            reconstruction = pycolmap.Reconstruction(str(path))
            count = len(reconstruction_images(reconstruction))
            models.append((path, reconstruction, count))
        except Exception as exc:
            print(f"[warn] cannot load COLMAP model {path}: {exc}")
    if not models:
        raise RuntimeError(f"no readable COLMAP model found below {root}")
    models.sort(key=lambda item: item[2], reverse=True)
    return models


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate target ~= scale * rotation @ source + translation."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance < 1e-15:
        raise RuntimeError("camera centers are degenerate")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def rotation_summary(errors: list[float]) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    return {
        "mean_deg": float(values.mean()),
        "median_deg": float(np.median(values)),
        "max_deg": float(values.max()),
    }


def evaluate_convention(
    name: str,
    frames_rotations: list[np.ndarray],
    frames_centers: np.ndarray,
    colmap_rotations: list[np.ndarray],
    colmap_centers: np.ndarray,
) -> dict[str, Any]:
    scale, world_rotation, translation = umeyama(frames_centers, colmap_centers)
    aligned_centers = (
        scale * (world_rotation @ frames_centers.T).T + translation[None]
    )
    center_errors = np.linalg.norm(aligned_centers - colmap_centers, axis=1)
    scene_extent = float(
        np.max(np.linalg.norm(colmap_centers - colmap_centers.mean(axis=0), axis=1))
    )

    errors_from_center_alignment = []
    for frame_rotation, colmap_rotation in zip(frames_rotations, colmap_rotations):
        predicted = frame_rotation @ world_rotation.T
        errors_from_center_alignment.append(
            rotation_error_degrees(predicted, colmap_rotation)
        )

    # Estimate a world-frame rotation using camera orientations only. From
    # R_colmap ~= R_frame @ A.T, every pair proposes A = R_colmap.T @ R_frame.
    orientation_world_rotation = project_to_rotation(
        sum(
            colmap_rotation.T @ frame_rotation
            for frame_rotation, colmap_rotation in zip(
                frames_rotations, colmap_rotations
            )
        )
    )
    errors_from_orientation_alignment = []
    for frame_rotation, colmap_rotation in zip(frames_rotations, colmap_rotations):
        predicted = frame_rotation @ orientation_world_rotation.T
        errors_from_orientation_alignment.append(
            rotation_error_degrees(predicted, colmap_rotation)
        )

    return {
        "convention": name,
        "orientation_only": rotation_summary(errors_from_orientation_alignment),
        "center_sim3_rotation": rotation_summary(errors_from_center_alignment),
        "center_alignment": {
            "scale": scale,
            "mean_error": float(center_errors.mean()),
            "median_error": float(np.median(center_errors)),
            "max_error": float(center_errors.max()),
            "scene_extent": scene_extent,
            "mean_error_over_extent": (
                float(center_errors.mean() / scene_extent) if scene_extent > 0 else None
            ),
        },
        "world_rotation_from_centers": world_rotation.tolist(),
        "world_rotation_from_orientations": orientation_world_rotation.tolist(),
    }


def camera_intrinsics_summary(reconstruction: Any) -> list[dict[str, Any]]:
    cameras = reconstruction.cameras
    if callable(cameras):
        cameras = cameras()
    items = cameras.values() if hasattr(cameras, "values") else cameras
    output = []
    for camera in items:
        model = camera.model
        if callable(model):
            model = model()
        model_name = getattr(model, "name", None)
        if callable(model_name):
            model_name = model_name()
        if model_name is None:
            model_name = str(model)
        output.append(
            {
                "camera_id": int(camera.camera_id),
                "model": str(model_name),
                "width": int(camera.width),
                "height": int(camera.height),
                "params": np.asarray(camera.params).tolist(),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare frames_info pose conventions against a COLMAP model"
    )
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="JSON report path (default: <scene>/frames_info_pose_diagnosis.json)",
    )
    args = parser.parse_args()

    scene = args.scene.resolve()
    frames = read_frames_info(scene / "frames_info.txt")
    models = find_models(args.colmap.resolve())
    model_path, reconstruction, model_image_count = models[0]
    print(f"Using largest model: {model_path} ({model_image_count} images)")

    colmap_by_stem: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for image in reconstruction_images(reconstruction):
        rotation, center = image_pose(image)
        name = str(image.name)
        colmap_by_stem[Path(name).stem] = (rotation, center, name)

    common = sorted(
        set(frames).intersection(colmap_by_stem),
        key=lambda timestamp: frames[timestamp]["frame"],
    )
    if len(common) < 3:
        raise RuntimeError(
            f"only {len(common)} timestamp filenames overlap frames_info and COLMAP"
        )

    frame_centers = np.asarray([frames[name]["center"] for name in common])
    colmap_centers = np.asarray([colmap_by_stem[name][1] for name in common])
    colmap_rotations = [colmap_by_stem[name][0] for name in common]

    axis_options = {
        "opencv": np.eye(3),
        "flip_yz": np.diag([1.0, -1.0, -1.0]),
        "flip_xz": np.diag([-1.0, 1.0, -1.0]),
        "flip_xy": np.diag([-1.0, -1.0, 1.0]),
    }
    results = []
    for direction in ("w2c", "c2w"):
        for axis_name, camera_axis_rotation in axis_options.items():
            candidate_rotations = []
            for timestamp in common:
                raw = frames[timestamp]["rotation_raw"]
                base_w2c = raw if direction == "w2c" else raw.T
                candidate_rotations.append(camera_axis_rotation @ base_w2c)
            results.append(
                evaluate_convention(
                    f"{direction}+{axis_name}",
                    candidate_rotations,
                    frame_centers,
                    colmap_rotations,
                    colmap_centers,
                )
            )

    results.sort(key=lambda item: item["orientation_only"]["mean_deg"])
    report = {
        "scene": str(scene),
        "frames_info": str(scene / "frames_info.txt"),
        "colmap_root": str(args.colmap.resolve()),
        "selected_model": str(model_path),
        "pycolmap_version": getattr(pycolmap, "__version__", "unknown"),
        "num_frames_info": len(frames),
        "num_colmap_images": model_image_count,
        "num_matched_images": len(common),
        "matched_timestamps": common,
        "unmatched_frames_info": sorted(set(frames) - set(common)),
        "unmatched_colmap_images": sorted(set(colmap_by_stem) - set(common)),
        "frames_info_intrinsics_first": {
            key: frames[common[0]][key]
            for key in ("fx", "fy", "cx", "cy", "width", "height")
        },
        "colmap_cameras": camera_intrinsics_summary(reconstruction),
        "conventions": results,
    }

    output = args.output or (scene / "frames_info_pose_diagnosis.json")
    output = output.resolve()
    with output.open("w") as handle:
        json.dump(report, handle, indent=2)

    print()
    print("convention          orient mean/median    center-Sim3 rot mean    center RMSE/extent")
    print("-" * 88)
    for item in results:
        orient = item["orientation_only"]
        center_rot = item["center_sim3_rotation"]
        center = item["center_alignment"]
        normalized = center["mean_error_over_extent"]
        print(
            f"{item['convention']:<20} "
            f"{orient['mean_deg']:7.3f}/{orient['median_deg']:7.3f} deg   "
            f"{center_rot['mean_deg']:9.3f} deg          "
            f"{normalized:9.5f}"
        )
    print()
    print(f"Report: {output}")
    print(f"Best orientation convention: {results[0]['convention']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
