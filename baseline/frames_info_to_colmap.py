#!/usr/bin/env python3
"""Convert an HW ``frames_info.txt`` pose dump to COLMAP text format.

Expected input layout::

    <scene>/frames_info.txt
    <scene>/images/<timestamp>.jpg

Output::

    <scene>/sparse/0/cameras.txt
    <scene>/sparse/0/images.txt
    <scene>/sparse/0/points3D.txt
    <scene>/sparse/0/conversion_summary.json

Default pose convention (matching the existing experimental HW loader):

* ``rotX rotY rotZ rotW`` is an OpenCV world-to-camera quaternion (xyzw).
* ``posX posY posZ`` is the OpenCV world-to-camera translation.
* Therefore it is written directly as COLMAP's ``TX TY TZ``.
* The corresponding camera center is ``C = -R.T @ t``.
* A zero principal point is interpreted as the image center.

The script only converts metadata; it does not run feature extraction, matching,
bundle adjustment, or triangulation. Consequently ``points3D.txt`` is empty.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Frame:
    frame: int
    timestamp: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    translation: np.ndarray
    quat_xyzw: np.ndarray


def read_frames_info(path: Path) -> list[Frame]:
    frames: list[Frame] = []
    seen_timestamps: set[str] = set()

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
            if timestamp in seen_timestamps:
                raise ValueError(f"{path}:{line_no}: duplicate timestamp {timestamp}")
            seen_timestamps.add(timestamp)

            quat = np.asarray([float(value) for value in fields[11:15]], dtype=np.float64)
            norm = float(np.linalg.norm(quat))
            if not math.isfinite(norm) or norm < 1e-12:
                raise ValueError(f"{path}:{line_no}: invalid quaternion")
            quat /= norm

            frames.append(
                Frame(
                    frame=int(fields[0]),
                    timestamp=timestamp,
                    fx=float(fields[2]),
                    fy=float(fields[3]),
                    cx=float(fields[4]),
                    cy=float(fields[5]),
                    width=int(float(fields[6])),
                    height=int(float(fields[7])),
                    translation=np.asarray(
                        [float(value) for value in fields[8:11]], dtype=np.float64
                    ),
                    quat_xyzw=quat,
                )
            )

    if not frames:
        raise RuntimeError(f"no frames found in {path}")
    frames.sort(key=lambda item: item.frame)
    return frames


def quat_xyzw_to_rotation(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to COLMAP's normalized wxyz quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )

    quat /= np.linalg.norm(quat)
    if quat[0] < 0:
        quat = -quat
    return quat


def index_images(image_dir: Path) -> tuple[dict[str, Path], list[str]]:
    by_stem: dict[str, Path] = {}
    duplicate_stems: list[str] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in by_stem:
            duplicate_stems.append(path.stem)
            continue
        by_stem[path.stem] = path
    if duplicate_stems:
        raise RuntimeError(
            "multiple supported images share these stems: "
            + ", ".join(sorted(set(duplicate_stems)))
        )
    return by_stem, sorted(path.name for path in by_stem.values())


def camera_key(frame: Frame, zero_principal_is_center: bool) -> tuple:
    cx = frame.cx
    cy = frame.cy
    if zero_principal_is_center and abs(cx) < 1e-12 and abs(cy) < 1e-12:
        cx = frame.width / 2.0
        cy = frame.height / 2.0
    return (frame.width, frame.height, frame.fx, frame.fy, cx, cy)


def convert(scene_dir: Path, overwrite: bool, zero_principal_is_center: bool) -> dict:
    frames_path = scene_dir / "frames_info.txt"
    image_dir = scene_dir / "images"
    output_dir = scene_dir / "sparse" / "0"

    if not frames_path.is_file():
        raise FileNotFoundError(frames_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    frames = read_frames_info(frames_path)
    images_by_stem, all_image_names = index_images(image_dir)

    matched: list[tuple[Frame, Path]] = []
    missing_images: list[str] = []
    for frame in frames:
        image = images_by_stem.get(frame.timestamp)
        if image is None:
            missing_images.append(frame.timestamp)
        else:
            matched.append((frame, image))
    if not matched:
        raise RuntimeError("no frames_info timestamps matched image filenames")

    matched_names = {image.name for _, image in matched}
    ignored_images = sorted(set(all_image_names) - matched_names)

    camera_ids: dict[tuple, int] = {}
    for frame, _ in matched:
        key = camera_key(frame, zero_principal_is_center)
        camera_ids.setdefault(key, len(camera_ids) + 1)

    with (output_dir / "cameras.txt").open("w") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(f"# Number of cameras: {len(camera_ids)}\n")
        for key, camera_id in camera_ids.items():
            width, height, fx, fy, cx, cy = key
            handle.write(
                f"{camera_id} PINHOLE {width} {height} "
                f"{fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n"
            )

    centers = []
    with (output_dir / "images.txt").open("w") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(matched)}, mean observations per image: 0\n")

        for image_id, (frame, image) in enumerate(matched, 1):
            rotation = quat_xyzw_to_rotation(frame.quat_xyzw)
            translation = frame.translation
            quat_wxyz = rotation_to_quat_wxyz(rotation)
            camera_id = camera_ids[camera_key(frame, zero_principal_is_center)]
            values = [*quat_wxyz.tolist(), *translation.tolist()]
            handle.write(
                f"{image_id} "
                + " ".join(f"{value:.12g}" for value in values)
                + f" {camera_id} {image.name}\n\n"
            )
            centers.append(-(rotation.T @ translation))

    with (output_dir / "points3D.txt").open("w") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        handle.write("# Number of points: 0, mean track length: 0\n")

    centers_array = np.asarray(centers)
    summary = {
        "scene": str(scene_dir),
        "frames_info": str(frames_path),
        "images_dir": str(image_dir),
        "output": str(output_dir),
        "pose_convention": {
            "rotation": "OpenCV world-to-camera quaternion in xyzw order",
            "position": "OpenCV world-to-camera translation",
            "colmap_translation": "t = [posX, posY, posZ]",
            "derived_camera_center": "C = -R.T @ t",
        },
        "zero_principal_is_center": zero_principal_is_center,
        "num_frames_info": len(frames),
        "num_matched_images": len(matched),
        "num_cameras": len(camera_ids),
        "missing_image_timestamps": missing_images,
        "ignored_images": ignored_images,
        "camera_center_min": centers_array.min(axis=0).tolist(),
        "camera_center_max": centers_array.max(axis=0).tolist(),
    }
    with (output_dir / "conversion_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an HW frames_info.txt file to COLMAP text format"
    )
    parser.add_argument("scene_dir", type=Path, help="scene containing images/ and frames_info.txt")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing sparse/0 directory"
    )
    parser.add_argument(
        "--zero-principal-is-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="interpret principalX=principalY=0 as the image center (default: true)",
    )
    args = parser.parse_args()

    summary = convert(
        args.scene_dir.resolve(), args.overwrite, args.zero_principal_is_center
    )
    print(
        f"converted {summary['num_matched_images']}/{summary['num_frames_info']} frames "
        f"using {summary['num_cameras']} camera(s) -> {summary['output']}"
    )
    if summary["missing_image_timestamps"]:
        print(f"missing images: {len(summary['missing_image_timestamps'])}")
    if summary["ignored_images"]:
        print("ignored images: " + ", ".join(summary["ignored_images"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
