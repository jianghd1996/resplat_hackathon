#!/usr/bin/env python3
"""Test whether context-view coverage has saturated at the selected budget.

The input is an output directory produced by ``batch_context_ablation.py``.
Every strategy is evaluated against the same reference geometry: an equal-size
pool sampled from all available strategy PLYs for a scene.  Coverage is based
on camera-frustum projection and therefore ignores occlusion.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compare_context_selections import angular_matrix, rotation_angle_matrix, visibility_matrix  # noqa: E402
from utils_A import STRATEGIES, select_context  # noqa: E402
from visualize_context_coverage import load_points, read_camera, read_images  # noqa: E402


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover_cases(root, groups, scenes, strategies):
    runs = root / "runs"
    available = sorted(path.name for path in runs.iterdir() if path.is_dir())
    selected_strategies = strategies or available
    missing = sorted(set(selected_strategies) - set(available))
    if missing:
        raise FileNotFoundError(f"strategy directories missing: {missing}")
    scene_filter = set(scenes or [])
    cases = defaultdict(dict)
    for strategy in selected_strategies:
        for group_dir in sorted((runs / strategy).iterdir()):
            if not group_dir.is_dir() or (groups and group_dir.name not in groups):
                continue
            for scene_dir in sorted(group_dir.iterdir()):
                key = (group_dir.name, scene_dir.name)
                if scene_filter and scene_dir.name not in scene_filter \
                        and f"{key[0]}/{key[1]}" not in scene_filter:
                    continue
                work = scene_dir / "work"
                ply = scene_dir / "resplat/resplat_world.ply"
                if (work / "split.json").is_file() and ply.is_file():
                    cases[key][strategy] = scene_dir
    return {key: value for key, value in cases.items() if value}


def pooled_reference(strategy_dirs, max_points, seed):
    items = sorted(strategy_dirs.items())
    per_source = max(1, math.ceil(max_points / len(items)))
    chunks = []
    sources = []
    for index, (strategy, run_dir) in enumerate(items):
        ply = run_dir / "resplat/resplat_world.ply"
        points, total = load_points(ply, per_source, seed + index)
        if len(points):
            chunks.append(points)
            sources.append({"strategy": strategy, "ply": str(ply),
                            "total_points": int(total), "sampled_points": int(len(points))})
    if not chunks:
        raise RuntimeError("no finite points found in reference PLYs")
    points = np.concatenate(chunks, axis=0)
    if len(points) > max_points:
        generator = np.random.default_rng(seed)
        points = points[generator.choice(len(points), max_points, replace=False)]
    return points, sources


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else 0.0


def coverage_values(selected_visibility, test_visibility, all_train_counts):
    counts = selected_visibility.sum(axis=0)
    result = {
        "global_union": float((counts >= 1).mean()),
        "global_support2": float((counts >= 2).mean()),
        "global_support4": float((counts >= 4).mean()),
        "mean_contexts_per_point": float(counts.mean()),
        "union_vs_all_train": safe_ratio(np.count_nonzero(counts >= 1),
                                          np.count_nonzero(all_train_counts >= 1)),
        "support2_vs_all_train": safe_ratio(np.count_nonzero(counts >= 2),
                                             np.count_nonzero(all_train_counts >= 2)),
        "support4_vs_all_train": safe_ratio(np.count_nonzero(counts >= 4),
                                             np.count_nonzero(all_train_counts >= 4)),
    }
    for threshold in (1, 2, 4):
        values = []
        supported = counts >= threshold
        for visible in test_visibility:
            denominator = np.count_nonzero(visible)
            values.append(safe_ratio(np.count_nonzero(visible & supported), denominator))
        result[f"test_support{threshold}"] = float(np.mean(values)) if values else 0.0
    return result


def pose_values(context_names, test_names, poses, all_names):
    context_centers = np.stack([poses[name]["center"] for name in context_names])
    test_centers = np.stack([poses[name]["center"] for name in test_names])
    all_centers = np.stack([poses[name]["center"] for name in all_names])
    scene_center = np.median(all_centers, axis=0)
    scale = max(float(np.median(np.linalg.norm(all_centers - scene_center, axis=1))), 1e-8)
    center_gap = np.linalg.norm(test_centers[:, None] - context_centers[None, :], axis=-1) / scale
    rotation_gap = rotation_angle_matrix(
        np.stack([poses[name]["R"] for name in test_names]),
        np.stack([poses[name]["R"] for name in context_names]))
    orbit_gap = angular_matrix(test_centers - scene_center, context_centers - scene_center)
    return {
        "test_center_gap_mean": float(center_gap.min(axis=1).mean()),
        "test_rotation_gap_mean": float(rotation_gap.min(axis=1).mean()),
        "test_orbit_gap_mean": float(orbit_gap.min(axis=1).mean()),
    }


def names_for_budget(strategy, budget, split, poses):
    order = list(split["images"])
    train_set = set(split["train"])
    test_set = set(split["test"])
    train_indices = [index for index, name in enumerate(order) if name in train_set]
    test_indices = [index for index, name in enumerate(order) if name in test_set]
    pose_tuples = {name: (poses[name]["R"], poses[name]["t"]) for name in order}
    positions = select_context(strategy, train_indices, budget, order=order,
                               test_indices=test_indices, poses=pose_tuples)
    return [order[train_indices[position]] for position in positions]


def metric_mean(run_dir):
    path = run_dir / "resplat_eval/metrics.json"
    if not path.is_file():
        return {"psnr": None, "ssim": None, "lpips": None}
    return read_json(path).get("mean", {"psnr": None, "ssim": None, "lpips": None})


def analyze_scene(key, strategy_dirs, args):
    group, scene = key
    points, sources = pooled_reference(strategy_dirs, args.max_points, args.seed)
    first_dir = next(iter(strategy_dirs.values()))
    first_work = first_dir / "work"
    base_split = read_json(first_work / "split.json")
    camera = read_camera(first_work / "sparse/0/cameras.txt")
    poses = read_images(first_work / "sparse/0/images.txt")
    order = [name for name in base_split["images"] if name in poses]
    train_names = [name for name in order if name in set(base_split["train"])]
    test_names = [name for name in order if name in set(base_split["test"])]
    visibility = visibility_matrix(points, [poses[name] for name in order], camera, args.near)
    visibility_by_name = {name: visibility[index] for index, name in enumerate(order)}
    train_visibility = np.stack([visibility_by_name[name] for name in train_names])
    test_visibility = np.stack([visibility_by_name[name] for name in test_names])
    all_train_counts = train_visibility.sum(axis=0)

    rows = []
    curves = []
    for strategy, run_dir in sorted(strategy_dirs.items()):
        split = read_json(run_dir / "work/split.json")
        final_names = [name for name in split["context"] if name in visibility_by_name]
        if not final_names:
            continue
        final_budget = len(final_names)
        budget_values = {}
        for budget in range(1, final_budget + 1):
            if budget == final_budget:
                names = final_names
            elif strategy in STRATEGIES:
                names = names_for_budget(strategy, budget, split, poses)
            else:
                names = final_names[:budget]
            selected = np.stack([visibility_by_name[name] for name in names])
            values = coverage_values(selected, test_visibility, all_train_counts)
            budget_values[budget] = values
            curves.append({"group": group, "scene": scene, "strategy": strategy,
                           "budget": budget, **values})
        final = budget_values[final_budget]
        previous = budget_values.get(max(1, final_budget - 4), final)
        quality = metric_mean(run_dir)
        row = {
            "group": group, "scene": scene, "strategy": strategy,
            "num_context": final_budget, **quality, **final,
            **pose_values(final_names, test_names, poses, order),
            # Selectors are rerun at each budget, so this is a 12->16 budget
            # effect rather than necessarily four nested, newly added views.
            "budget12_to16_union_gain": final["global_union"] - previous["global_union"],
            "budget12_to16_test_support2_gain": (
                final["test_support2"] - previous["test_support2"]),
        }
        row["coverage_saturated"] = bool(
            row["union_vs_all_train"] >= args.saturation_ratio
            and row["test_support1"] >= args.test_support_ratio
            and row["budget12_to16_union_gain"] <= args.max_budget_gain)
        rows.append(row)
    metadata = {"group": group, "scene": scene, "num_reference_points": len(points),
                "num_train_views": len(train_names), "num_test_views": len(test_names),
                "reference_sources": sources}
    return rows, curves, metadata


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def add_uniform_deltas(rows):
    by_scene = defaultdict(dict)
    for row in rows:
        by_scene[(row["group"], row["scene"])][row["strategy"]] = row
    for scene_rows in by_scene.values():
        uniform = scene_rows.get("uniform16")
        for row in scene_rows.values():
            for field in ("psnr", "global_union", "global_support2", "test_support1",
                          "test_support2", "union_vs_all_train"):
                row[f"delta_{field}_vs_uniform"] = (
                    None if uniform is None or row[field] is None or uniform[field] is None
                    else row[field] - uniform[field])


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["strategy"]].append(row)
    result = []
    fields = ("psnr", "ssim", "lpips", "global_union", "global_support2",
              "test_support1", "test_support2", "union_vs_all_train",
              "budget12_to16_union_gain", "delta_psnr_vs_uniform", "delta_global_union_vs_uniform",
              "delta_test_support2_vs_uniform")
    for strategy, values in sorted(groups.items()):
        item = {"strategy": strategy, "scenes": len(values),
                "saturated_scenes": sum(row["coverage_saturated"] for row in values)}
        item.update({f"{field}_mean": mean_or_none([row[field] for row in values])
                     for field in fields})
        result.append(item)
    return result


def plot_scene(rows, curves, output):
    strategies = [row["strategy"] for row in rows]
    colors = dict(zip(strategies, plt.cm.tab10(np.linspace(0, 1, max(len(strategies), 2)))))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for strategy in strategies:
        values = [row for row in curves if row["strategy"] == strategy]
        axes[0, 0].plot([row["budget"] for row in values],
                        [row["union_vs_all_train"] for row in values], marker="o",
                        markersize=2.5, label=strategy, color=colors[strategy])
        axes[0, 1].plot([row["budget"] for row in values],
                        [row["test_support2"] for row in values], marker="o",
                        markersize=2.5, label=strategy, color=colors[strategy])
    axes[0, 0].set_title("Global union / all-train union")
    axes[0, 1].set_title("Mean test-frustum support by >=2 contexts")
    for axis in axes[0]:
        axis.set_xlabel("context budget")
        axis.set_ylabel("ratio")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)

    x = np.arange(len(rows))
    axes[1, 0].bar(x - 0.2, [row["union_vs_all_train"] for row in rows], 0.4,
                   label="union / all train")
    axes[1, 0].bar(x + 0.2, [row["test_support2"] for row in rows], 0.4,
                   label="test support >=2")
    axes[1, 0].set_xticks(x, strategies, rotation=30, ha="right")
    axes[1, 0].set_ylim(0, 1.03)
    axes[1, 0].set_title("Final 16-view coverage")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.25)

    for row in rows:
        if row["psnr"] is None:
            continue
        axes[1, 1].scatter(row["test_support2"], row["psnr"], s=55,
                           color=colors[row["strategy"]])
        axes[1, 1].annotate(row["strategy"], (row["test_support2"], row["psnr"]),
                            xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1, 1].set_xlabel("test support >=2")
    axes[1, 1].set_ylabel("ReSplat-only PSNR")
    axes[1, 1].set_title("Coverage versus reconstruction quality")
    axes[1, 1].grid(alpha=0.25)
    fig.suptitle(f"Coverage saturation: {rows[0]['group']}/{rows[0]['scene']}", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_aggregate(rows, aggregate, output):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    uniform = [row for row in rows if row["strategy"] == "uniform16"]
    labels = [f"{row['group']}/{row['scene']}" for row in uniform]
    x = np.arange(len(uniform))
    axes[0].bar(x - 0.25, [row["union_vs_all_train"] for row in uniform], 0.25,
                label="union/all train")
    axes[0].bar(x, [row["test_support1"] for row in uniform], 0.25,
                label="test >=1")
    axes[0].bar(x + 0.25, [row["test_support2"] for row in uniform], 0.25,
                label="test >=2")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_title("uniform16 coverage by scene")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    for row in rows:
        if row["strategy"] == "uniform16" or row["delta_psnr_vs_uniform"] is None:
            continue
        axes[1].scatter(100 * row["delta_test_support2_vs_uniform"],
                        row["delta_psnr_vs_uniform"], s=48, alpha=0.8)
        axes[1].annotate(f"{row['scene']}:{row['strategy']}",
                         (100 * row["delta_test_support2_vs_uniform"],
                          row["delta_psnr_vs_uniform"]),
                         xytext=(3, 3), textcoords="offset points", fontsize=6)
    axes[1].axhline(0, color="0.5", linewidth=0.8)
    axes[1].axvline(0, color="0.5", linewidth=0.8)
    axes[1].set_xlabel("test >=2 support gain vs uniform (percentage points)")
    axes[1].set_ylabel("PSNR gain vs uniform (dB)")
    axes[1].set_title("Does extra coverage become extra quality?")
    axes[1].grid(alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def pearson_pairs(rows, x_field, y_field):
    pairs = [(row[x_field], row[y_field]) for row in rows
             if row["strategy"] != "uniform16"
             and row[x_field] is not None and row[y_field] is not None]
    if len(pairs) < 3:
        return None
    x, y = np.asarray(pairs, dtype=np.float64).T
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def report_markdown(rows, aggregate, correlations, args):
    uniform = [row for row in rows if row["strategy"] == "uniform16"]
    lines = ["# Context Coverage Saturation", "",
             "Coverage uses one shared, equally pooled ReSplat point set per scene. "
             "Frustum projection ignores occlusion.", "",
             "## uniform16 saturation", "",
             "| group/scene | union/all train | test >=1 | test >=2 | 12→16 union gain | saturated | PSNR |",
             "|---|---:|---:|---:|---:|:---:|---:|"]
    for row in uniform:
        psnr = "—" if row["psnr"] is None else f"{row['psnr']:.3f}"
        lines.append(f"| {row['group']}/{row['scene']} | {row['union_vs_all_train']:.3f} | "
                     f"{row['test_support1']:.3f} | {row['test_support2']:.3f} | "
                     f"{100 * row['budget12_to16_union_gain']:.2f} pp | "
                     f"{'yes' if row['coverage_saturated'] else 'no'} | {psnr} |")
    lines.extend(["", "## Strategy averages", "",
                  "| strategy | scenes | saturated | PSNR | union/all train | test >=2 | last 4 gain | ΔPSNR vs uniform | Δtest >=2 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in aggregate:
        def show(value, digits=3):
            return "—" if value is None else f"{value:.{digits}f}"
        lines.append(f"| {row['strategy']} | {row['scenes']} | {row['saturated_scenes']} | "
                     f"{show(row['psnr_mean'])} | {show(row['union_vs_all_train_mean'])} | "
                     f"{show(row['test_support2_mean'])} | "
                     f"{show(100 * row['budget12_to16_union_gain_mean'], 2)} pp | "
                     f"{show(row['delta_psnr_vs_uniform_mean'])} | "
                     f"{show(100 * row['delta_test_support2_vs_uniform_mean'], 2)} pp |")
    lines.extend(["", "## Cross-case correlations", "",
                  "Computed over non-uniform strategies as deltas relative to uniform16.", ""])
    for name, value in correlations.items():
        lines.append(f"- {name}: {'unavailable' if value is None else f'{value:.3f}'}")
    lines.extend(["", "## Saturation rule", "",
                  f"A case is marked saturated when union/all-train >= {args.saturation_ratio:.2f}, "
                  f"test support >=1 >= {args.test_support_ratio:.2f}, and the final four views add "
                  f"<= {100 * args.max_budget_gain:.1f} percentage points of union coverage.", "",
                  "This validates geometric view coverage, not texture, occlusion, exposure, or "
                  "ReSplat's learned input-distribution preference."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, required=True,
                        help="output directory from batch_context_ablation.py")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--groups", nargs="*")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--strategies", nargs="*")
    parser.add_argument("--max-points", type=int, default=140000)
    parser.add_argument("--near", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--saturation-ratio", type=float, default=0.95)
    parser.add_argument("--test-support-ratio", type=float, default=0.95)
    parser.add_argument("--max-budget-gain", type=float, default=0.02,
                        help="maximum union-coverage gain from budget 12 to 16")
    args = parser.parse_args()
    root = args.ablation_root.resolve()
    output = (args.out or root / "coverage_saturation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    cases = discover_cases(root, args.groups, args.scenes, args.strategies)
    if not cases:
        raise RuntimeError(f"no complete runs found below {root / 'runs'}")

    rows, curves, metadata = [], [], []
    for key, strategy_dirs in sorted(cases.items()):
        print(f"[coverage] {key[0]}/{key[1]}: {len(strategy_dirs)} strategies", flush=True)
        scene_rows, scene_curves, scene_metadata = analyze_scene(key, strategy_dirs, args)
        rows.extend(scene_rows)
        curves.extend(scene_curves)
        metadata.append(scene_metadata)
        plot_scene(scene_rows, scene_curves,
                   figure_dir / f"{key[0]}_{key[1]}_coverage.png")
    add_uniform_deltas(rows)
    aggregate = aggregate_rows(rows)
    correlations = {
        "corr(delta global union, delta PSNR)": pearson_pairs(
            rows, "delta_global_union_vs_uniform", "delta_psnr_vs_uniform"),
        "corr(delta test support >=2, delta PSNR)": pearson_pairs(
            rows, "delta_test_support2_vs_uniform", "delta_psnr_vs_uniform"),
    }
    plot_aggregate(rows, aggregate, output / "coverage_saturation_summary.png")
    write_csv(output / "per_scene_strategy.csv", rows)
    write_csv(output / "coverage_curves.csv", curves)
    write_csv(output / "strategy_summary.csv", aggregate)
    (output / "report.json").write_text(json.dumps({
        "definition": "shared pooled ReSplat points; positive-depth in-frame projection; no occlusion",
        "thresholds": {"saturation_ratio": args.saturation_ratio,
                       "test_support_ratio": args.test_support_ratio,
                       "max_budget_gain": args.max_budget_gain},
        "correlations": correlations, "scenes": metadata,
        "per_scene_strategy": rows, "strategy_summary": aggregate,
    }, indent=2))
    report = report_markdown(rows, aggregate, correlations, args)
    (output / "report.md").write_text(report)
    print(report)
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
