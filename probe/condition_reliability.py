"""Cross-validate condition-guided A decisions on all 13 local scenes.

Candidate generation and validation both use training-track observations. Track
IDs are deterministically split five ways; no test image or quality metric is
read. The output validates a proxy decision, not final rendering quality.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dataset_census import ROOT, load_colmap_poses, load_transforms_json, track_sets
from utils_A.condition_guided import (MIN_MEDIAN_COVERAGE_GAIN,
                                      MIN_MEDIAN_SUPPORT_GAIN,
                                      MIN_POSITIVE_FOLDS, NUM_FOLDS, diagnose)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/probe_v1/condition_reliability.json")
    args = parser.parse_args()
    census = json.loads((ROOT / "results/rethink_v0/dataset_census.json").read_text())
    rows = []
    for row in census["scenes"]:
        scene = ROOT / "datasets" / row["dataset"] / row["scene"]
        names = [value["name"] for value in row["per_train_image_proxies"]]
        info = load_colmap_poses(str(scene)) or load_transforms_json(str(scene))
        centers = np.stack([-info["poses"][name][0].T @ info["poses"][name][1]
                            for name in names])
        by_name = track_sets(scene, names)
        tracks = [by_name[name] for name in names] if by_name else None
        result = diagnose(range(len(names)), 16, centers=centers, track_sets=tracks)
        selected = result.pop("selected")
        candidate = result.pop("candidate")
        result.update(dataset=row["dataset"], scene=row["scene"],
                      selected_context=[names[index] for index in selected],
                      candidate_context=[names[index] for index in candidate],
                      selected_changed_slots=sum(a != b for a, b in zip(
                          [names[index] for index in selected],
                          row["strategies"]["uniform16"]["context"])))
        rows.append(result)
        print(row["dataset"], row["scene"][:12], result["reason"],
              round(result.get("median_coverage_gain", 0.0), 5), flush=True)
    report = {
        "scope": "training tracks only; no test images or quality metrics",
        "num_folds": NUM_FOLDS,
        "min_positive_folds": MIN_POSITIVE_FOLDS,
        "min_median_coverage_gain": MIN_MEDIAN_COVERAGE_GAIN,
        "min_median_support_p10_gain": MIN_MEDIAN_SUPPORT_GAIN,
        "scenes": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
