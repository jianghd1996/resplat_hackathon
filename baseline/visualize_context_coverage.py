#!/usr/bin/env python3
"""Visualize how ReSplat's selected context views cover a reconstructed scene.

The script consumes a Module-A work directory and a world-space 3DGS PLY.  It
produces a camera/scene coverage figure, a contact sheet of the selected input
images, and machine-readable coverage statistics.

Coverage means that a sampled Gaussian center projects inside a selected
camera's image bounds with positive depth.  It deliberately ignores occlusion;
the result is a view-frustum coverage diagnostic, not a visibility renderer.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageOps  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def qvec_to_rotmat(qvec):
    """COLMAP wxyz quaternion to a world-to-camera rotation matrix."""
    w, x, y, z = np.asarray(qvec, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("zero-length COLMAP quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def read_camera(path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            continue
        camera_id = int(fields[0])
        model = fields[1]
        width, height = int(fields[2]), int(fields[3])
        params = [float(value) for value in fields[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise ValueError(
                f"Unsupported camera model {model}; run Module A first to create PINHOLE cameras.txt")
        return {
            "id": camera_id, "model": model, "width": width, "height": height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        }
    raise RuntimeError(f"No camera found in {path}")


def read_images(path):
    images = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        # An image record has IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME.
        # This also avoids interpreting a POINTS2D line as an image record.
        if len(fields) < 10 or Path(fields[9]).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        qvec = np.array([float(value) for value in fields[1:5]])
        tvec = np.array([float(value) for value in fields[5:8]])
        rotation = qvec_to_rotmat(qvec)
        center = -rotation.T @ tvec
        images[fields[9]] = {
            "id": int(fields[0]), "name": fields[9], "R": rotation,
            "t": tvec, "center": center,
            "forward": rotation.T @ np.array([0.0, 0.0, 1.0]),
        }
    if not images:
        raise RuntimeError(f"No image poses found in {path}")
    return images


def load_points(path, max_points, seed):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'plyfile'; install it with: pip install plyfile") from exc
    ply = PlyData.read(str(path), mmap="c")
    vertex = ply["vertex"].data
    missing = [name for name in ("x", "y", "z") if name not in vertex.dtype.names]
    if missing:
        raise ValueError(f"PLY has no Gaussian centers: missing properties {missing}")
    total = len(vertex)
    if total == 0:
        raise ValueError(f"PLY contains no vertices: {path}")
    if total > max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(total, size=max_points, replace=False))
        points = np.column_stack((vertex["x"][indices], vertex["y"][indices], vertex["z"][indices]))
    else:
        points = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
    points = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    return points[finite], total


def frustum_coverage(points, context, camera, near):
    counts = np.zeros(len(points), dtype=np.uint16)
    per_camera = {}
    for pose in context:
        camera_points = points @ pose["R"].T + pose["t"]
        depth = camera_points[:, 2]
        valid_depth = depth > near
        safe_depth = np.where(valid_depth, depth, 1.0)
        u = camera["fx"] * camera_points[:, 0] / safe_depth + camera["cx"]
        v = camera["fy"] * camera_points[:, 1] / safe_depth + camera["cy"]
        visible = (valid_depth & (u >= 0) & (u < camera["width"])
                   & (v >= 0) & (v < camera["height"]))
        counts += visible.astype(np.uint16)
        per_camera[pose["name"]] = int(visible.sum())
    return counts, per_camera


def principal_frame(points):
    center = np.median(points, axis=0)
    centered = points - center
    sample = centered
    if len(sample) > 50000:
        sample = sample[np.linspace(0, len(sample) - 1, 50000, dtype=int)]
    _, _, vh = np.linalg.svd(sample, full_matrices=False)
    basis = vh.T
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return center, basis


def angular_coordinates(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-12)
    azimuth = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
    elevation = np.degrees(np.arcsin(np.clip(unit[:, 2], -1.0, 1.0)))
    return azimuth, elevation, unit


def nearest_context_angles(all_unit, context_unit):
    similarities = np.clip(all_unit @ context_unit.T, -1.0, 1.0)
    return np.degrees(np.arccos(np.max(similarities, axis=1)))


def set_axes_equal_3d(axis, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def draw_frustum(axis, pose, camera, scale, color="#d62728", alpha=0.85):
    center = pose["center"]
    corners = np.array([
        [(0 - camera["cx"]) / camera["fx"], (0 - camera["cy"]) / camera["fy"], 1.0],
        [(camera["width"] - camera["cx"]) / camera["fx"],
         (0 - camera["cy"]) / camera["fy"], 1.0],
        [(camera["width"] - camera["cx"]) / camera["fx"],
         (camera["height"] - camera["cy"]) / camera["fy"], 1.0],
        [(0 - camera["cx"]) / camera["fx"],
         (camera["height"] - camera["cy"]) / camera["fy"], 1.0],
    ])
    corners = center + (corners @ pose["R"]) * scale
    for corner in corners:
        axis.plot(*np.vstack((center, corner)).T, color=color, alpha=alpha, linewidth=0.7)
    loop = np.vstack((corners, corners[0]))
    axis.plot(*loop.T, color=color, alpha=alpha, linewidth=0.8)


def make_contact_sheet(work, image_names, output, title_prefix="view"):
    columns = 4
    rows = math.ceil(len(image_names) / columns)
    cell_w, cell_h = 320, 260
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(image_names):
        path = work / "images" / name
        if not path.is_file():
            raise FileNotFoundError(f"Context image missing: {path}")
        with Image.open(path) as source:
            thumb = ImageOps.contain(source.convert("RGB"), (cell_w - 16, cell_h - 38))
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        px = x + (cell_w - thumb.width) // 2
        py = y + 8
        canvas.paste(thumb, (px, py))
        label = f"{title_prefix} {index:02d}  {name}"
        draw.text((x + 8, y + cell_h - 25), label, fill="black")
    canvas.save(output)


def make_figure(points, counts, poses, context_names, camera, center, basis,
                azimuth, elevation, gaps, test_names, output, plot_points):
    context_set = set(context_names)
    test_set = set(test_names)
    centers = np.stack([pose["center"] for pose in poses])
    context_poses = [pose for pose in poses if pose["name"] in context_set]
    context_centers = np.stack([pose["center"] for pose in context_poses])
    test_poses = [pose for pose in poses if pose["name"] in test_set]
    test_centers = (np.stack([pose["center"] for pose in test_poses])
                    if test_poses else np.empty((0, 3)))
    pca_points = (points - center) @ basis
    pca_centers = (centers - center) @ basis

    if len(points) > plot_points:
        indices = np.linspace(0, len(points) - 1, plot_points, dtype=int)
    else:
        indices = np.arange(len(points))

    fig = plt.figure(figsize=(17, 13), constrained_layout=True)
    axis3d = fig.add_subplot(2, 2, 1, projection="3d")
    scatter = axis3d.scatter(
        points[indices, 0], points[indices, 1], points[indices, 2],
        c=counts[indices], cmap="viridis", vmin=0, vmax=max(1, len(context_names)),
        s=1.0, alpha=0.55, rasterized=True)
    axis3d.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="0.65", s=9, label="all views")
    axis3d.scatter(context_centers[:, 0], context_centers[:, 1], context_centers[:, 2],
                   c="#d62728", marker="^", s=48, label="16 context views")
    if len(test_centers):
        axis3d.scatter(test_centers[:, 0], test_centers[:, 1], test_centers[:, 2],
                       c="#1f77b4", marker="x", s=42, linewidths=1.2,
                       label="test views")
    extent = np.percentile(np.linalg.norm(points - center, axis=1), 90)
    frustum_scale = max(float(extent) * 0.12, 1e-5)
    for pose in context_poses:
        draw_frustum(axis3d, pose, camera, frustum_scale)
    set_axes_equal_3d(axis3d, np.vstack((points[indices], centers)))
    axis3d.set_title("World-space Gaussian coverage and selected camera frustums")
    axis3d.set_xlabel("world X")
    axis3d.set_ylabel("world Y")
    axis3d.set_zlabel("world Z")
    axis3d.legend(loc="upper right")
    fig.colorbar(scatter, ax=axis3d, shrink=0.7, label="number of context frustums")

    top = fig.add_subplot(2, 2, 2)
    top.scatter(pca_points[indices, 0], pca_points[indices, 1], c=counts[indices],
                cmap="viridis", vmin=0, vmax=max(1, len(context_names)),
                s=1.0, alpha=0.45, rasterized=True)
    top.scatter(pca_centers[:, 0], pca_centers[:, 1], c="0.65", s=14)
    context_mask = np.array([pose["name"] in context_set for pose in poses])
    test_mask = np.array([pose["name"] in test_set for pose in poses])
    top.scatter(pca_centers[context_mask, 0], pca_centers[context_mask, 1],
                c="#d62728", marker="^", s=55)
    if test_mask.any():
        top.scatter(pca_centers[test_mask, 0], pca_centers[test_mask, 1],
                    c="#1f77b4", marker="x", s=48, linewidths=1.2,
                    label="test views")
    for index, xy in enumerate(pca_centers[context_mask, :2]):
        top.annotate(str(index), xy, xytext=(3, 3), textcoords="offset points", fontsize=7)
    if test_mask.any():
        top.legend(loc="best")
    top.set_aspect("equal", adjustable="datalim")
    top.set_title("PCA-aligned top view")
    top.set_xlabel("principal axis 1")
    top.set_ylabel("principal axis 2")
    top.grid(alpha=0.2)

    sphere = fig.add_subplot(2, 2, 3)
    sphere.scatter(azimuth, elevation, c=gaps, cmap="magma", s=24, alpha=0.75,
                   label="all cameras")
    sphere.scatter(azimuth[context_mask], elevation[context_mask], c="#20c55a",
                   edgecolors="black", linewidths=0.6, marker="^", s=85,
                   label="context")
    if test_mask.any():
        sphere.scatter(azimuth[test_mask], elevation[test_mask], c="#27a8e0",
                       linewidths=1.5, marker="x", s=80,
                       label="test")
    for index, (az, el) in enumerate(zip(azimuth[context_mask], elevation[context_mask])):
        sphere.annotate(str(index), (az, el), xytext=(4, 4), textcoords="offset points", fontsize=7)
    sphere.set_xlim(-180, 180)
    sphere.set_ylim(-90, 90)
    sphere.set_xticks(np.arange(-180, 181, 45))
    sphere.set_yticks(np.arange(-90, 91, 30))
    sphere.grid(alpha=0.25)
    sphere.set_xlabel("azimuth around model center (deg)")
    sphere.set_ylabel("elevation (deg)")
    sphere.set_title("Camera-orbit angular coverage\ncolor = nearest context angular gap")
    sphere.legend()

    hist = fig.add_subplot(2, 2, 4)
    bins = np.arange(-0.5, len(context_names) + 1.5, 1.0)
    hist.hist(counts, bins=bins, color="#4472c4", edgecolor="white")
    hist.set_yscale("log")
    hist.set_xlabel("number of selected context frustums containing a Gaussian center")
    hist.set_ylabel("sampled Gaussian centers (log scale)")
    hist.set_title("Frustum overlap distribution")
    hist.grid(axis="y", alpha=0.25)
    hist.text(
        0.98, 0.96,
        f"covered by >=1: {(counts >= 1).mean() * 100:.1f}%\n"
        f"covered by >=2: {(counts >= 2).mean() * 100:.1f}%\n"
        f"covered by >=4: {(counts >= 4).mean() * 100:.1f}%\n"
        f"camera gap mean/max: {gaps.mean():.1f}° / {gaps.max():.1f}°",
        transform=hist.transAxes, ha="right", va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})

    fig.suptitle(f"Context-view coverage: {len(context_names)} selected / {len(poses)} total views",
                 fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True,
                        help="Module-A work directory containing split.json, images/, sparse/0/")
    parser.add_argument("--ply", type=Path, required=True,
                        help="world-space ReSplat/3DGS PLY aligned with the work-dir cameras")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <PLY directory>/context_coverage)")
    parser.add_argument("--max-points", type=int, default=150000,
                        help="maximum Gaussian centers sampled for coverage calculations")
    parser.add_argument("--plot-points", type=int, default=50000,
                        help="maximum Gaussian centers drawn in the summary figure")
    parser.add_argument("--near", type=float, default=1e-4,
                        help="positive camera-space depth threshold")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    work = args.work.resolve()
    ply_path = args.ply.resolve()
    output = (args.out or (ply_path.parent / "context_coverage")).resolve()
    output.mkdir(parents=True, exist_ok=True)

    split_path = work / "split.json"
    camera_path = work / "sparse" / "0" / "cameras.txt"
    images_path = work / "sparse" / "0" / "images.txt"
    for required in (split_path, camera_path, images_path, ply_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    split = json.loads(split_path.read_text())
    context_names = list(split.get("context", []))
    if not context_names:
        raise RuntimeError(f"No context views in {split_path}")
    camera = read_camera(camera_path)
    pose_map = read_images(images_path)
    missing = [name for name in context_names if name not in pose_map]
    if missing:
        raise RuntimeError(f"Context views missing from COLMAP images.txt: {missing}")

    ordered_names = [name for name in split.get("images", []) if name in pose_map]
    ordered_names.extend(sorted(set(pose_map) - set(ordered_names)))
    poses = [pose_map[name] for name in ordered_names]
    context = [pose_map[name] for name in context_names]
    points, total_points = load_points(ply_path, args.max_points, args.seed)
    if len(points) == 0:
        raise RuntimeError("No finite Gaussian centers available for visualization")

    counts, per_camera = frustum_coverage(points, context, camera, args.near)
    center, basis = principal_frame(points)
    centers = np.stack([pose["center"] for pose in poses])
    context_centers = np.stack([pose["center"] for pose in context])
    _, _, all_unit = angular_coordinates((centers - center) @ basis)
    azimuth, elevation, _ = angular_coordinates((centers - center) @ basis)
    _, _, context_unit = angular_coordinates((context_centers - center) @ basis)
    gaps = nearest_context_angles(all_unit, context_unit)

    figure_path = output / "context_coverage.png"
    contact_path = output / "context_views.png"
    test_contact_path = output / "test_views.png"
    report_path = output / "context_coverage.json"
    test_names = list(split.get("test", []))
    make_figure(points, counts, poses, context_names, camera, center, basis,
                azimuth, elevation, gaps, test_names, figure_path, args.plot_points)
    make_contact_sheet(work, context_names, contact_path, "context")
    if test_names:
        make_contact_sheet(work, test_names, test_contact_path, "test")

    context_set = set(context_names)
    test_set = set(split.get("test", []))
    report = {
        "work": str(work),
        "ply": str(ply_path),
        "total_gaussians_in_ply": total_points,
        "sampled_finite_gaussians": len(points),
        "coverage_definition": "positive depth and projection inside image bounds; occlusion ignored",
        "num_total_views": len(poses),
        "num_context_views": len(context_names),
        "context_views": context_names,
        "gaussian_frustum_coverage": {
            "mean_context_views": float(counts.mean()),
            "median_context_views": float(np.median(counts)),
            "covered_by_at_least_1_ratio": float((counts >= 1).mean()),
            "covered_by_at_least_2_ratio": float((counts >= 2).mean()),
            "covered_by_at_least_4_ratio": float((counts >= 4).mean()),
            "uncovered_ratio": float((counts == 0).mean()),
            "histogram": {str(value): int((counts == value).sum())
                          for value in np.unique(counts)},
            "points_inside_each_context_frustum": per_camera,
        },
        "camera_orbit_nearest_context_gap_degrees": {
            "all_mean": float(gaps.mean()),
            "all_max": float(gaps.max()),
            "train_non_context_mean": float(np.mean([
                gap for name, gap in zip(ordered_names, gaps)
                if name not in context_set and name not in test_set
            ])) if any(name not in context_set and name not in test_set for name in ordered_names) else None,
            "test_mean": float(np.mean([
                gap for name, gap in zip(ordered_names, gaps) if name in test_set
            ])) if any(name in test_set for name in ordered_names) else None,
            "test_max": float(np.max([
                gap for name, gap in zip(ordered_names, gaps) if name in test_set
            ])) if any(name in test_set for name in ordered_names) else None,
            "per_view": {name: float(gap) for name, gap in zip(ordered_names, gaps)},
        },
        "outputs": {
            "figure": str(figure_path),
            "context_contact_sheet": str(contact_path),
            "test_contact_sheet": str(test_contact_path) if test_names else None,
        },
    }
    report_path.write_text(json.dumps(report, indent=2))

    print(f"[coverage] {len(context_names)}/{len(poses)} context/total views")
    print(f"[coverage] sampled {len(points):,}/{total_points:,} Gaussian centers")
    print(f"[coverage] >=1/2/4 frustums: {(counts >= 1).mean() * 100:.1f}% / "
          f"{(counts >= 2).mean() * 100:.1f}% / {(counts >= 4).mean() * 100:.1f}%")
    print(f"[coverage] camera angular gap mean/max: {gaps.mean():.2f} / {gaps.max():.2f} deg")
    print(f"[coverage] figure: {figure_path}")
    print(f"[coverage] context images: {contact_path}")
    if test_names:
        print(f"[coverage] test images: {test_contact_path}")
    print(f"[coverage] report: {report_path}")


if __name__ == "__main__":
    main()
