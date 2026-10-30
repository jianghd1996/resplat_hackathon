"""Print a portable summary from a server-side A/B experiment directory."""

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        return {}


DELTA_FIELDS = ("psnr", "ssim", "lpips", "A_seconds", "resplat_seconds",
                "B_seconds", "total_seconds")


def add_baseline_deltas(row, baseline_row):
    """Attach experiment minus same-machine baseline values to a scene row."""
    for key in DELTA_FIELDS:
        value = row.get(key)
        baseline = baseline_row.get(key)
        row[f"delta_{key}"] = (value - baseline
                               if isinstance(value, (int, float))
                               and isinstance(baseline, (int, float)) else None)
    return row


def collect(scene_dir, group, scene, reference_dir=None):
    report = read(scene_dir / "report.json")
    a = read(scene_dir / "work/module_a.json")
    r = read(scene_dir / "resplat/resplat.json")
    b = read(scene_dir / "module_b/module_b.json")
    metric = read(scene_dir / "module_b/metrics.json")
    reference_path = scene_dir / "reference_3dgs/module_b.json"
    reference = read(reference_path)
    reference_source = "same_experiment" if reference else None
    if not reference and reference_dir is not None:
        reference_path = reference_dir / "reference_3dgs/module_b.json"
        reference = read(reference_path)
        reference_source = "baseline_A_baseline_B" if reference else None
    reference_iterations = reference.get("iterations")
    reference_train_seconds = reference.get("train_seconds")
    reference_iter_seconds = (reference_train_seconds / reference_iterations
                              if isinstance(reference_train_seconds, (int, float))
                              and isinstance(reference_iterations, int)
                              and reference_iterations > 0 else None)
    mean = metric.get("mean") or report.get("metrics") or report
    times = {
        "A": a.get("total_seconds", report.get("time_A", report.get("A_seconds"))),
        "ReSplat": r.get("total_seconds", report.get("time_resplat", report.get("resplat_seconds"))),
        "B": b.get("total_seconds", report.get("time_B", report.get("B_seconds"))),
    }
    stage_complete = all(isinstance(stage.get("total_seconds"), (int, float))
                         for stage in (a, r, b))
    reference_budget = (30000 * reference_iter_seconds
                        if reference_iter_seconds is not None else None)
    if stage_complete:
        total = round(sum(times.values()), 3)
    else:
        total = report.get("time_total", report.get("total_seconds"))
        if total is None and all(isinstance(x, (int, float)) for x in times.values()):
            total = round(sum(times.values()), 3)
    return {
        "group": group, "scene": scene,
        "pose_source": a.get("pose_source", report.get("pose_source")),
        "test_views": metric.get("num_views", report.get("num_test_views", report.get("evaluated_views"))),
        "psnr": mean.get("psnr"), "ssim": mean.get("ssim"),
        "lpips": mean.get("lpips"), "A_seconds": times["A"],
        "resplat_seconds": times["ReSplat"], "B_seconds": times["B"],
        "total_seconds": total,
        "reference_30k_seconds": (reference.get("total_seconds")
                                  if reference_iterations == 30000 else None),
        "reference_30k_psnr": (reference.get("test_psnr_quick")
                               if reference_iterations == 30000 else None),
        "reference_T_iter_seconds": reference_iter_seconds,
        "reference_30k_B_budget_seconds": reference_budget,
        "reference_source": reference_source,
        "B_fraction_of_reference_budget": (times["B"] / reference_budget
                                           if reference_budget and isinstance(times["B"], (int, float))
                                           else None),
    }


def format_cell(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return str(round(value, 4))
    return str(value)


def format_with_delta(row, key):
    value = row.get(key)
    delta = row.get(f"delta_{key}")
    if value is None:
        return "—"
    if delta is None:
        return format_cell(value)
    precision = 4 if key in ("ssim", "lpips") else 3
    return f"{format_cell(value)} ({delta:+.{precision}f})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="baseline_A_baseline_B")
    parser.add_argument("--results-root", type=Path, default=REPO / "results")
    parser.add_argument("--format", choices=["markdown", "json", "both"], default="both")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+_A_[A-Za-z0-9_]+_B", args.experiment):
        parser.error("experiment must be named xx_A_xx_B")
    root = args.results_root / args.experiment
    if not root.is_dir():
        parser.error(f"experiment directory missing: {root}")
    rows = []
    for group in sorted(root.iterdir()):
        if not group.is_dir():
            continue
        for scene in sorted(group.iterdir()):
            if not scene.is_dir():
                continue
            baseline_dir = args.results_root / "baseline_A_baseline_B" / group.name / scene.name
            row = collect(scene, group.name, scene.name, baseline_dir)
            baseline_row = (row if args.experiment == "baseline_A_baseline_B"
                            else collect(baseline_dir, group.name, scene.name))
            rows.append(add_baseline_deltas(row, baseline_row))
    if args.format in ("markdown", "both"):
        print("| group | scene | PSNR (Δ) | SSIM (Δ) | LPIPS (Δ) | A(s) (Δ) | ReSplat(s) (Δ) | B(s) (Δ) | total(s) (Δ) | ref30k(s) / PSNR |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in rows:
            ref = (f"{format_cell(row['reference_30k_seconds'])} / "
                   f"{format_cell(row['reference_30k_psnr'])}")
            print("| " + " | ".join(format_with_delta(row, key)
                                    if key in DELTA_FIELDS else format_cell(row[key]) for key in (
                "group", "scene", "psnr", "ssim", "lpips", "A_seconds",
                "resplat_seconds", "B_seconds", "total_seconds")) + f" | {ref} |")
    if args.format in ("json", "both"):
        print(json.dumps({"experiment": args.experiment, "scenes": rows},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
