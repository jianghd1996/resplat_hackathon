#!/usr/bin/env python3
"""Compare two ReSplat context selections and diagnose quality differences.

The comparison uses the same world-space reference Gaussian centers for both
strategies.  It measures camera-pose coverage, context-to-context frustum
overlap/connectivity, global multi-view coverage, and support available at each
held-out test view.  If metric JSON files are available, per-view PSNR changes
are correlated with the geometric changes.

Frustum coverage ignores occlusion.  The output is a diagnostic explanation,
not a proof of causality.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from visualize_context_coverage import load_points, read_camera, read_images  # noqa: E402


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "p10": None, "p50": None, "p90": None, "max": None}
    return {
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def rotation_angle_matrix(left, right):
    result = np.empty((len(left), len(right)), dtype=np.float64)
    for i, left_rotation in enumerate(left):
        relative = left_rotation[None, :, :] @ np.transpose(right, (0, 2, 1))
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        result[i] = np.degrees(np.arccos(cosine))
    return result


def angular_matrix(left, right):
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip(left @ right.T, -1.0, 1.0)))


def visibility_matrix(points, poses, camera, near):
    visibility = np.empty((len(poses), len(points)), dtype=bool)
    for index, pose in enumerate(poses):
        camera_points = points @ pose["R"].T + pose["t"]
        depth = camera_points[:, 2]
        positive = depth > near
        safe_depth = np.where(positive, depth, 1.0)
        u = camera["fx"] * camera_points[:, 0] / safe_depth + camera["cx"]
        v = camera["fy"] * camera_points[:, 1] / safe_depth + camera["cy"]
        visibility[index] = (
            positive & (u >= 0) & (u < camera["width"])
            & (v >= 0) & (v < camera["height"])
        )
    return visibility


def pairwise_jaccard(visibility):
    count = len(visibility)
    matrix = np.eye(count, dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            union = np.count_nonzero(visibility[i] | visibility[j])
            value = (np.count_nonzero(visibility[i] & visibility[j]) / union
                     if union else 0.0)
            matrix[i, j] = matrix[j, i] = value
    return matrix


def graph_stats(jaccard, threshold):
    adjacency = (jaccard >= threshold) & ~np.eye(len(jaccard), dtype=bool)
    degrees = adjacency.sum(axis=1)
    unseen = set(range(len(jaccard)))
    components = []
    while unseen:
        root = unseen.pop()
        stack = [root]
        component = [root]
        while stack:
            node = stack.pop()
            neighbors = set(np.flatnonzero(adjacency[node])) & unseen
            unseen -= neighbors
            stack.extend(neighbors)
            component.extend(neighbors)
        components.append(component)
    return {
        "jaccard_edge_threshold": float(threshold),
        "num_components": len(components),
        "component_sizes": sorted((len(value) for value in components), reverse=True),
        "isolated_contexts": int(np.count_nonzero(degrees == 0)),
        "mean_degree": float(degrees.mean()),
        "min_degree": int(degrees.min()),
    }


def load_case(name, work, metrics_path):
    work = work.resolve()
    split = json.loads((work / "split.json").read_text())
    camera = read_camera(work / "sparse/0/cameras.txt")
    pose_map = read_images(work / "sparse/0/images.txt")
    context_names = list(split["context"])
    test_names = list(split["test"])
    all_names = [value for value in split["images"] if value in pose_map]
    missing = [value for value in context_names + test_names if value not in pose_map]
    if missing:
        raise RuntimeError(f"{name}: views missing from COLMAP poses: {missing[:5]}")
    metrics = None
    if metrics_path is None:
        inferred = work.parent / "resplat_eval/metrics.json"
        metrics_path = inferred if inferred.is_file() else None
    if metrics_path is not None:
        metrics = json.loads(metrics_path.resolve().read_text())
    return {
        "name": name, "work": work, "split": split, "camera": camera,
        "pose_map": pose_map, "context_names": context_names,
        "test_names": test_names, "all_names": all_names, "metrics": metrics,
    }


def evaluate_case(case, points, scene_center, scene_scale, near, graph_threshold):
    context = [case["pose_map"][name] for name in case["context_names"]]
    tests = [case["pose_map"][name] for name in case["test_names"]]
    all_poses = [case["pose_map"][name] for name in case["all_names"]]

    context_centers = np.stack([pose["center"] for pose in context])
    test_centers = np.stack([pose["center"] for pose in tests])
    all_centers = np.stack([pose["center"] for pose in all_poses])
    context_rotations = np.stack([pose["R"] for pose in context])
    test_rotations = np.stack([pose["R"] for pose in tests])

    center_distance = np.linalg.norm(
        test_centers[:, None, :] - context_centers[None, :, :], axis=-1) / scene_scale
    rotation_distance = rotation_angle_matrix(test_rotations, context_rotations)
    orbit_distance = angular_matrix(
        test_centers - scene_center, context_centers - scene_center)
    all_orbit_distance = angular_matrix(
        all_centers - scene_center, context_centers - scene_center)
    composite = center_distance + 0.5 * rotation_distance / 180.0

    context_visibility = visibility_matrix(points, context, case["camera"], near)
    test_visibility = visibility_matrix(points, tests, case["camera"], near)
    coverage_count = context_visibility.sum(axis=0)
    context_union = coverage_count >= 1
    covered = max(int(context_union.sum()), 1)
    jaccard = pairwise_jaccard(context_visibility)
    upper = jaccard[np.triu_indices(len(jaccard), 1)]

    support_one = []
    support_two = []
    support_four = []
    test_visible_counts = []
    for visible in test_visibility:
        denominator = int(visible.sum())
        test_visible_counts.append(denominator)
        if denominator == 0:
            support_one.append(0.0)
            support_two.append(0.0)
            support_four.append(0.0)
            continue
        support_one.append(float(np.count_nonzero(visible & (coverage_count >= 1)) / denominator))
        support_two.append(float(np.count_nonzero(visible & (coverage_count >= 2)) / denominator))
        support_four.append(float(np.count_nonzero(visible & (coverage_count >= 4)) / denominator))

    per_test = {}
    for index, name in enumerate(case["test_names"]):
        per_test[name] = {
            "nearest_center_distance_normalized": float(center_distance[index].min()),
            "nearest_rotation_degrees": float(rotation_distance[index].min()),
            "nearest_orbit_degrees": float(orbit_distance[index].min()),
            "nearest_composite_pose_distance": float(composite[index].min()),
            "frustum_support_at_least_1": support_one[index],
            "frustum_support_at_least_2": support_two[index],
            "frustum_support_at_least_4": support_four[index],
            "reference_points_in_test_frustum": test_visible_counts[index],
        }

    return {
        "context_views": case["context_names"],
        "test_pose_coverage": {
            "nearest_center_distance_normalized": percentile_summary(center_distance.min(axis=1)),
            "nearest_rotation_degrees": percentile_summary(rotation_distance.min(axis=1)),
            "nearest_orbit_degrees": percentile_summary(orbit_distance.min(axis=1)),
            "nearest_composite_pose_distance": percentile_summary(composite.min(axis=1)),
        },
        "all_camera_orbit_coverage": {
            "nearest_context_degrees": percentile_summary(all_orbit_distance.min(axis=1)),
        },
        "gaussian_frustum_coverage": {
            "union_ratio": float((coverage_count >= 1).mean()),
            "covered_by_at_least_2_ratio": float((coverage_count >= 2).mean()),
            "covered_by_at_least_4_ratio": float((coverage_count >= 4).mean()),
            "covered_by_at_least_8_ratio": float((coverage_count >= 8).mean()),
            "mean_contexts_per_reference_point": float(coverage_count.mean()),
            "mean_contexts_per_covered_point": float(coverage_count.sum() / covered),
        },
        "context_frustum_overlap": {
            "pairwise_jaccard": percentile_summary(upper),
            "graph": graph_stats(jaccard, graph_threshold),
        },
        "test_view_frustum_support": {
            "at_least_1": percentile_summary(support_one),
            "at_least_2": percentile_summary(support_two),
            "at_least_4": percentile_summary(support_four),
        },
        "per_test_view": per_test,
    }


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def metric_comparison(case_a, case_b, result_a, result_b):
    if case_a["metrics"] is None or case_b["metrics"] is None:
        return None
    mean_a = case_a["metrics"]["mean"]
    mean_b = case_b["metrics"]["mean"]
    per_a = {value["name"]: value for value in case_a["metrics"]["per_view"]}
    per_b = {value["name"]: value for value in case_b["metrics"]["per_view"]}
    common = sorted(set(per_a) & set(per_b) & set(result_a["per_test_view"]) & set(result_b["per_test_view"]))
    rows = []
    for name in common:
        geometry_a = result_a["per_test_view"][name]
        geometry_b = result_b["per_test_view"][name]
        rows.append({
            "name": name,
            "psnr_delta_b_minus_a": per_b[name]["psnr"] - per_a[name]["psnr"],
            "ssim_delta_b_minus_a": per_b[name]["ssim"] - per_a[name]["ssim"],
            "lpips_delta_b_minus_a": per_b[name]["lpips"] - per_a[name]["lpips"],
            "pose_distance_improvement_a_minus_b": (
                geometry_a["nearest_composite_pose_distance"]
                - geometry_b["nearest_composite_pose_distance"]),
            "support2_improvement_b_minus_a": (
                geometry_b["frustum_support_at_least_2"]
                - geometry_a["frustum_support_at_least_2"]),
            "support4_improvement_b_minus_a": (
                geometry_b["frustum_support_at_least_4"]
                - geometry_a["frustum_support_at_least_4"]),
        })
    psnr_delta = [row["psnr_delta_b_minus_a"] for row in rows]
    return {
        "mean": {
            "a": mean_a,
            "b": mean_b,
            "delta_b_minus_a": {
                "psnr": float(mean_b["psnr"] - mean_a["psnr"]),
                "ssim": float(mean_b["ssim"] - mean_a["ssim"]),
                "lpips": float(mean_b["lpips"] - mean_a["lpips"]),
            },
        },
        "num_psnr_improved_views": sum(value > 0 for value in psnr_delta),
        "num_compared_views": len(rows),
        "correlation_with_psnr_delta": {
            "pose_distance_improvement": pearson(
                [row["pose_distance_improvement_a_minus_b"] for row in rows], psnr_delta),
            "support2_improvement": pearson(
                [row["support2_improvement_b_minus_a"] for row in rows], psnr_delta),
            "support4_improvement": pearson(
                [row["support4_improvement_b_minus_a"] for row in rows], psnr_delta),
        },
        "per_view": rows,
    }


def percent_change(before, after):
    return 100.0 * (after - before) / max(abs(before), 1e-12)


def diagnose(name_a, name_b, a, b, metrics):
    findings = []
    pose_a = a["test_pose_coverage"]["nearest_composite_pose_distance"]["mean"]
    pose_b = b["test_pose_coverage"]["nearest_composite_pose_distance"]["mean"]
    overlap_a = a["context_frustum_overlap"]["pairwise_jaccard"]["mean"]
    overlap_b = b["context_frustum_overlap"]["pairwise_jaccard"]["mean"]
    multi_a = a["test_view_frustum_support"]["at_least_2"]["mean"]
    multi_b = b["test_view_frustum_support"]["at_least_2"]["mean"]
    union_a = a["gaussian_frustum_coverage"]["union_ratio"]
    union_b = b["gaussian_frustum_coverage"]["union_ratio"]
    graph_a = a["context_frustum_overlap"]["graph"]
    graph_b = b["context_frustum_overlap"]["graph"]

    findings.append(
        f"{name_b} changes mean test-pose distance by {percent_change(pose_a, pose_b):+.1f}% "
        f"relative to {name_a} (lower is better).")
    findings.append(
        f"Mean context-frustum Jaccard changes by {percent_change(overlap_a, overlap_b):+.1f}%; "
        f"mean >=2-view test support changes by {percent_change(multi_a, multi_b):+.1f}%.")
    findings.append(
        f"Global reference-point union coverage changes by {percent_change(union_a, union_b):+.1f}%; "
        f"visibility-graph components change {graph_a['num_components']} -> {graph_b['num_components']}.")

    if metrics is not None:
        delta = metrics["mean"]["delta_b_minus_a"]
        findings.append(
            f"Observed quality delta ({name_b} - {name_a}): PSNR {delta['psnr']:+.3f} dB, "
            f"SSIM {delta['ssim']:+.4f}, LPIPS {delta['lpips']:+.4f}.")
        if pose_b < pose_a and delta["psnr"] <= 0:
            findings.append(
                "Test-pose proximity improved without improving PSNR, so nearest-pose distance "
                "alone is not a sufficient selection objective.")
        if delta["psnr"] < 0 and (overlap_b < overlap_a * 0.97 or multi_b < multi_a * 0.97
                                  or graph_b["num_components"] > graph_a["num_components"]):
            findings.append(
                "The degradation is consistent with reduced multi-view overlap/connectivity: "
                "the selected inputs are closer to targets but provide weaker shared geometry.")
        elif delta["psnr"] < 0 and abs(union_b - union_a) < 0.01:
            findings.append(
                "Global frustum union is nearly unchanged; the loss is more likely due to "
                "redundancy/overlap, image content, or ReSplat distribution mismatch than an "
                "uncovered-region problem.")
    findings.append(
        "These are projection-based correlations and ignore occlusion, texture, blur, and exposure; "
        "treat the explanation as evidence for the next ablation rather than proof.")
    return findings


def make_figure(name_a, name_b, a, b, metrics, output):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    colors = ["#4472c4", "#ed7d31"]

    labels = ["center\n(normalized)", "rotation\n(deg/30)", "orbit\n(deg/30)"]
    values_a = [
        a["test_pose_coverage"]["nearest_center_distance_normalized"]["mean"],
        a["test_pose_coverage"]["nearest_rotation_degrees"]["mean"] / 30.0,
        a["test_pose_coverage"]["nearest_orbit_degrees"]["mean"] / 30.0,
    ]
    values_b = [
        b["test_pose_coverage"]["nearest_center_distance_normalized"]["mean"],
        b["test_pose_coverage"]["nearest_rotation_degrees"]["mean"] / 30.0,
        b["test_pose_coverage"]["nearest_orbit_degrees"]["mean"] / 30.0,
    ]
    x = np.arange(len(labels))
    axes[0, 0].bar(x - 0.18, values_a, 0.36, label=name_a, color=colors[0])
    axes[0, 0].bar(x + 0.18, values_b, 0.36, label=name_b, color=colors[1])
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_title("Mean nearest-context pose gaps (lower is better)")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.2)

    labels = ["global union", ">=2 global", ">=4 global", "test >=1", "test >=2", "test >=4"]
    values_a = [
        a["gaussian_frustum_coverage"]["union_ratio"],
        a["gaussian_frustum_coverage"]["covered_by_at_least_2_ratio"],
        a["gaussian_frustum_coverage"]["covered_by_at_least_4_ratio"],
        a["test_view_frustum_support"]["at_least_1"]["mean"],
        a["test_view_frustum_support"]["at_least_2"]["mean"],
        a["test_view_frustum_support"]["at_least_4"]["mean"],
    ]
    values_b = [
        b["gaussian_frustum_coverage"]["union_ratio"],
        b["gaussian_frustum_coverage"]["covered_by_at_least_2_ratio"],
        b["gaussian_frustum_coverage"]["covered_by_at_least_4_ratio"],
        b["test_view_frustum_support"]["at_least_1"]["mean"],
        b["test_view_frustum_support"]["at_least_2"]["mean"],
        b["test_view_frustum_support"]["at_least_4"]["mean"],
    ]
    x = np.arange(len(labels))
    axes[0, 1].bar(x - 0.18, values_a, 0.36, label=name_a, color=colors[0])
    axes[0, 1].bar(x + 0.18, values_b, 0.36, label=name_b, color=colors[1])
    axes[0, 1].set_xticks(x, labels, rotation=20, ha="right")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Reference-Gaussian frustum support (higher is better)")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.2)

    overlap_labels = ["pair Jaccard", "mean degree / 15", "mean contexts / 16"]
    overlap_a = [
        a["context_frustum_overlap"]["pairwise_jaccard"]["mean"],
        a["context_frustum_overlap"]["graph"]["mean_degree"] / 15.0,
        a["gaussian_frustum_coverage"]["mean_contexts_per_covered_point"] / 16.0,
    ]
    overlap_b = [
        b["context_frustum_overlap"]["pairwise_jaccard"]["mean"],
        b["context_frustum_overlap"]["graph"]["mean_degree"] / 15.0,
        b["gaussian_frustum_coverage"]["mean_contexts_per_covered_point"] / 16.0,
    ]
    x = np.arange(len(overlap_labels))
    axes[1, 0].bar(x - 0.18, overlap_a, 0.36, label=name_a, color=colors[0])
    axes[1, 0].bar(x + 0.18, overlap_b, 0.36, label=name_b, color=colors[1])
    axes[1, 0].set_xticks(x, overlap_labels)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("Context overlap and connectivity proxies")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.2)

    scatter = axes[1, 1]
    if metrics is not None and metrics["per_view"]:
        pose_gain = np.array([row["pose_distance_improvement_a_minus_b"]
                              for row in metrics["per_view"]])
        psnr_delta = np.array([row["psnr_delta_b_minus_a"] for row in metrics["per_view"]])
        support_gain = np.array([row["support2_improvement_b_minus_a"]
                                 for row in metrics["per_view"]])
        points = scatter.scatter(pose_gain, psnr_delta, c=support_gain, cmap="coolwarm",
                                 s=48, edgecolors="black", linewidths=0.4)
        scatter.axhline(0, color="black", linewidth=0.8)
        scatter.axvline(0, color="black", linewidth=0.8)
        scatter.set_xlabel("pose-distance improvement A-B (positive favors B)")
        scatter.set_ylabel("PSNR delta B-A (dB)")
        scatter.set_title("Per-test-view geometry change vs quality")
        fig.colorbar(points, ax=scatter, label=">=2-view support delta B-A")
    else:
        scatter.text(0.5, 0.5, "Per-view metric JSON not available", ha="center", va="center")
        scatter.set_axis_off()
    scatter.grid(alpha=0.2)
    fig.suptitle(f"Context selection diagnosis: {name_a} vs {name_b}", fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_report(report):
    names = report["cases"]
    a = report["geometry"][names[0]]
    b = report["geometry"][names[1]]
    metrics = report["quality"]
    rows = [
        ("Test composite pose gap mean ↓",
         a["test_pose_coverage"]["nearest_composite_pose_distance"]["mean"],
         b["test_pose_coverage"]["nearest_composite_pose_distance"]["mean"]),
        ("Test orbit gap mean (deg) ↓",
         a["test_pose_coverage"]["nearest_orbit_degrees"]["mean"],
         b["test_pose_coverage"]["nearest_orbit_degrees"]["mean"]),
        ("Global union coverage ↑", a["gaussian_frustum_coverage"]["union_ratio"],
         b["gaussian_frustum_coverage"]["union_ratio"]),
        ("Global >=2-view coverage ↑",
         a["gaussian_frustum_coverage"]["covered_by_at_least_2_ratio"],
         b["gaussian_frustum_coverage"]["covered_by_at_least_2_ratio"]),
        ("Context pair Jaccard mean ↑",
         a["context_frustum_overlap"]["pairwise_jaccard"]["mean"],
         b["context_frustum_overlap"]["pairwise_jaccard"]["mean"]),
        ("Visibility graph components ↓",
         a["context_frustum_overlap"]["graph"]["num_components"],
         b["context_frustum_overlap"]["graph"]["num_components"]),
        ("Test >=2-view support mean ↑",
         a["test_view_frustum_support"]["at_least_2"]["mean"],
         b["test_view_frustum_support"]["at_least_2"]["mean"]),
        ("Test >=4-view support mean ↑",
         a["test_view_frustum_support"]["at_least_4"]["mean"],
         b["test_view_frustum_support"]["at_least_4"]["mean"]),
    ]
    lines = ["# Context Selection Diagnosis", "", f"Reference points: {report['reference_points']}", "",
             f"| Metric | {names[0]} | {names[1]} |", "|---|---:|---:|"]
    for label, value_a, value_b in rows:
        lines.append(f"| {label} | {value_a:.5f} | {value_b:.5f} |")
    if metrics is not None:
        mean = metrics["mean"]
        lines.extend([
            "", "## ReSplat Test Quality", "",
            f"| Metric | {names[0]} | {names[1]} | Delta B-A |",
            "|---|---:|---:|---:|",
            f"| PSNR ↑ | {mean['a']['psnr']:.4f} | {mean['b']['psnr']:.4f} | {mean['delta_b_minus_a']['psnr']:+.4f} |",
            f"| SSIM ↑ | {mean['a']['ssim']:.4f} | {mean['b']['ssim']:.4f} | {mean['delta_b_minus_a']['ssim']:+.4f} |",
            f"| LPIPS ↓ | {mean['a']['lpips']:.4f} | {mean['b']['lpips']:.4f} | {mean['delta_b_minus_a']['lpips']:+.4f} |",
            "", f"PSNR improved on {metrics['num_psnr_improved_views']}/{metrics['num_compared_views']} test views.",
        ])
    lines.extend(["", "## Diagnostic Findings", ""])
    lines.extend(f"- {value}" for value in report["findings"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-work", type=Path, required=True)
    parser.add_argument("--b-work", type=Path, required=True)
    parser.add_argument("--a-name", default="uniform16")
    parser.add_argument("--b-name", default="oracle_test_nearest")
    parser.add_argument("--a-metrics", type=Path)
    parser.add_argument("--b-metrics", type=Path)
    parser.add_argument("--reference-ply", type=Path, nargs="+", required=True,
                        help="one or more aligned PLYs; equal samples are combined")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--near", type=float, default=1e-4)
    parser.add_argument("--graph-jaccard-threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.max_points < len(args.reference_ply):
        parser.error("--max-points must be at least the number of reference PLYs")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case_a = load_case(args.a_name, args.a_work, args.a_metrics)
    case_b = load_case(args.b_name, args.b_work, args.b_metrics)
    if case_a["test_names"] != case_b["test_names"]:
        raise ValueError("The two work directories do not have identical test splits")

    samples = []
    each = args.max_points // len(args.reference_ply)
    ply_counts = {}
    for index, path in enumerate(args.reference_ply):
        points, total = load_points(path.resolve(), each, args.seed + index)
        samples.append(points)
        ply_counts[str(path.resolve())] = {"total": total, "sampled": len(points)}
    points = np.concatenate(samples, axis=0)

    all_centers = np.stack([
        case_a["pose_map"][name]["center"] for name in case_a["all_names"]
    ])
    scene_center = np.median(all_centers, axis=0)
    scene_scale = max(float(np.median(np.linalg.norm(all_centers - scene_center, axis=1))), 1e-8)
    result_a = evaluate_case(case_a, points, scene_center, scene_scale, args.near,
                             args.graph_jaccard_threshold)
    result_b = evaluate_case(case_b, points, scene_center, scene_scale, args.near,
                             args.graph_jaccard_threshold)
    quality = metric_comparison(case_a, case_b, result_a, result_b)
    findings = diagnose(args.a_name, args.b_name, result_a, result_b, quality)
    report = {
        "cases": [args.a_name, args.b_name],
        "reference_points": len(points),
        "reference_plys": ply_counts,
        "coverage_definition": "positive depth and projection inside image bounds; occlusion ignored",
        "geometry": {args.a_name: result_a, args.b_name: result_b},
        "quality": quality,
        "findings": findings,
    }

    json_path = output / "context_selection_comparison.json"
    markdown_path = output / "context_selection_comparison.md"
    figure_path = output / "context_selection_comparison.png"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    markdown_path.write_text(markdown_report(report))
    make_figure(args.a_name, args.b_name, result_a, result_b, quality, figure_path)
    print(markdown_report(report))
    print(f"[compare] JSON: {json_path}")
    print(f"[compare] Markdown: {markdown_path}")
    print(f"[compare] Figure: {figure_path}")


if __name__ == "__main__":
    main()
