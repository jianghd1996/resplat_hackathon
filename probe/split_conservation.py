"""Probe whether FastGS split operators conserve a thin Gaussian surface.

This is an operator-level diagnostic, not a quality experiment. It reads the
three canonical ReSplat input PLYs, samples parents with a fixed local RNG and
compares three two-child constructions in the parent's two largest local axes.
The numerical footprint is an orthographic tangent-plane proxy; it deliberately
does not reproduce projection, visibility, alpha sorting or occlusion.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
    }


def load_sample(path, count, rng):
    vertex = PlyData.read(path, mmap="c")["vertex"].data
    indices = rng.choice(len(vertex), min(count, len(vertex)), replace=False)
    opacity_logit = np.asarray(vertex["opacity"])[indices].astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))
    scales = np.exp(np.column_stack([
        np.asarray(vertex[f"scale_{axis}"])[indices] for axis in range(3)
    ]).astype(np.float64))
    scales.sort(axis=1)
    return len(vertex), opacity, scales


def offsets(name, count, rng):
    if name == "current_isotropic":
        return rng.normal(size=(count, 2, 2)), 1.0 / 1.6
    if name == "current_long_axis":
        value = np.zeros((count, 2, 2), dtype=np.float64)
        value[:, :, 0] = rng.normal(size=(count, 2))
        return value, 1.0 / 1.6
    if name == "surface_conserving":
        angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
        direction = np.column_stack([np.cos(angle), np.sin(angle)])
        return np.stack([direction, -direction], axis=1), 1.0 / np.sqrt(2.0)
    raise ValueError(name)


def operator_moments(center, scale_ratio):
    centroid = center.mean(axis=1)
    second_about_parent = scale_ratio**2 + np.mean(center**2, axis=1)
    second_about_centroid = second_about_parent - centroid**2
    return {
        "linear_tangent_integral_ratio": float(2.0 * scale_ratio**2),
        "centroid_shift_normalized": describe(np.linalg.norm(centroid, axis=1)),
        "second_moment_about_parent_ratio": {
            "axis0": describe(second_about_parent[:, 0]),
            "axis1": describe(second_about_parent[:, 1]),
        },
        "second_moment_about_child_centroid_ratio": {
            "axis0": describe(second_about_centroid[:, 0]),
            "axis1": describe(second_about_centroid[:, 1]),
        },
    }


def footprint_proxy(opacity, center, scale_ratio, grid_size, batch=128):
    coordinate = np.linspace(-4.0, 4.0, grid_size, dtype=np.float64)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    parent_shape = np.exp(-0.5 * np.sum(grid**2, axis=1))
    integral_ratio = []
    nrmse = []
    for start in range(0, len(opacity), batch):
        stop = min(start + batch, len(opacity))
        alpha = opacity[start:stop, None]
        parent = alpha * parent_shape[None]
        child_alpha = []
        for child in range(2):
            delta = ((grid[None] - center[start:stop, child, None]) /
                     scale_ratio)
            shape = np.exp(-0.5 * np.sum(delta**2, axis=2))
            child_alpha.append(alpha * shape)
        combined = 1.0 - (1.0 - child_alpha[0]) * (1.0 - child_alpha[1])
        integral_ratio.extend((combined.sum(axis=1) /
                               np.maximum(parent.sum(axis=1), 1e-12)).tolist())
        numerator = np.sqrt(np.mean((combined - parent)**2, axis=1))
        denominator = np.sqrt(np.mean(parent**2, axis=1))
        nrmse.extend((numerator / np.maximum(denominator, 1e-12)).tolist())
    return {
        "composited_alpha_integral_ratio": describe(integral_ratio),
        "composited_alpha_nrmse": describe(nrmse),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--grid-size", type=int, default=65)
    parser.add_argument("--out", type=Path,
                        default=RESULTS / "probe_v1/split_conservation.json")
    args = parser.parse_args()
    if args.sample_count < 1 or args.grid_size < 9:
        parser.error("sample-count must be positive and grid-size must be at least 9")

    scenes = json.loads((ROOT / "baseline/scenes.json").read_text())
    report = {
        "seed": 0,
        "sample_count_requested": args.sample_count,
        "grid_size": args.grid_size,
        "scope": "operator-level orthographic tangent-plane proxy; no test images or metrics",
        "operators": {
            "current_isotropic": "two independent N(0,parent_scale) offsets; all scales /1.6",
            "current_long_axis": "two independent offsets on largest axis; all scales /1.6",
            "surface_conserving": "antithetic unit-ellipse tangent offsets; two tangent scales /sqrt(2), normal scale unchanged",
        },
        "scenes": [],
    }
    for scene_index, (group, info) in enumerate(scenes.items()):
        path = (RESULTS / "baseline_A_no_reset7000_B" / group / info["scene"] /
                "resplat/resplat_world.ply")
        parent_rng = np.random.default_rng(scene_index)
        total, opacity, scales = load_sample(path, args.sample_count, parent_rng)
        record = {
            "group": group,
            "scene": info["scene"],
            "input_ply": str(path.relative_to(ROOT)),
            "parent_count": total,
            "sample_count": len(opacity),
            "opacity": describe(opacity),
            "fraction_min_scale_at_1e_6_floor": float(
                np.isclose(scales[:, 0], 1e-6, rtol=1e-5, atol=0).mean()),
            "middle_to_max_scale": describe(scales[:, 1] / scales[:, 2]),
            "min_to_max_scale": describe(scales[:, 0] / scales[:, 2]),
            "operators": {},
        }
        for operator_index, name in enumerate(report["operators"]):
            operator_rng = np.random.default_rng(1000 * scene_index + operator_index)
            center, ratio = offsets(name, len(opacity), operator_rng)
            value = operator_moments(center, ratio)
            value.update(footprint_proxy(opacity, center, ratio, args.grid_size))
            value["normal_scale_ratio"] = 1.0 if name == "surface_conserving" else ratio
            record["operators"][name] = value
        report["scenes"].append(record)
        print(group, info["scene"][:12],
              "floor", round(record["fraction_min_scale_at_1e_6_floor"], 4),
              "nrmse", {name: round(value["composited_alpha_nrmse"]["mean"], 4)
                         for name, value in record["operators"].items()}, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
