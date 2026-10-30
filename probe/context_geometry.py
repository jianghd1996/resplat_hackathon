"""Read-only A-selection audit. Outputs statistics, never selects using test images.

Run with the resplat Python from the repository root. Frustum coverage ignores
occlusion. The pooled point cloud is a sensitivity check, not ground truth.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))
from visualize_context_coverage import read_camera, read_images, load_points

STRATEGIES = {"uniform16": "baseline_A_baseline_B", "fps": "fps_A_baseline_B",
              "coverage_local": "coverage_local_A_baseline_B",
              "quality_filter": "quality_filter_A_baseline_B"}


def describe(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max()),
            "min": float(a.min())}


def coverage(points, poses, camera):
    masks = []
    for pose in poses:
        p = points @ pose["R"].T + pose["t"]
        z = p[:, 2]
        safe = np.where(z > 1e-4, z, 1)
        u = camera["fx"] * p[:, 0] / safe + camera["cx"]
        v = camera["fy"] * p[:, 1] / safe + camera["cy"]
        masks.append((z > 1e-4) & (u >= 0) & (u < camera["width"]) &
                     (v >= 0) & (v < camera["height"]))
    masks = np.asarray(masks)
    counts = masks.sum(axis=0)
    inter = masks.astype(float) @ masks.astype(float).T
    union = masks.sum(1)[:, None] + masks.sum(1)[None, :] - inter
    jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(jac, -1)
    return {"at_least_1": float((counts >= 1).mean()),
            "at_least_2": float((counts >= 2).mean()),
            "at_least_4": float((counts >= 4).mean()),
            "mean_count": float(counts.mean()),
            "histogram_0_to_16": np.bincount(counts, minlength=17).tolist(),
            "per_camera_fraction": masks.mean(1).tolist(),
            "nearest_pair_jaccard_mean": float(jac.max(1).mean())}


def tracks(source, train_names):
    path = source / "sparse/0/images.bin"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "probe_colmap_loader", ROOT / "baseline/3dgs/scene/colmap_loader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images = module.read_extrinsics_binary(str(path))
    by_stem = {Path(i.name).stem: set(map(int, i.point3D_ids[i.point3D_ids >= 0]))
               for i in images.values()}
    if any(Path(n).stem not in by_stem for n in train_names):
        return None
    return {n: by_stem[Path(n).stem] for n in train_names}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/rethink_v0/context_geometry.json")
    args = ap.parse_args()
    report = {"seed": 0, "reference_points": 20000,
              "definition": "frustum only; no occlusion; distances divided by train-camera bbox diagonal",
              "scenes": {}}
    scenes = json.loads((ROOT / "baseline/scenes.json").read_text())
    for group, info in scenes.items():
        rel = Path(group) / info["scene"]
        base = ROOT / "results/baseline_A_baseline_B" / rel
        split = json.loads((base / "work/split.json").read_text())
        pose_map = read_images(base / "work/sparse/0/images.txt")
        camera = read_camera(base / "work/sparse/0/cameras.txt")
        train = split["train"]
        centers = np.stack([pose_map[n]["center"] for n in train])
        forward = np.stack([pose_map[n]["forward"] for n in train])
        scale = max(float(np.linalg.norm(np.ptp(centers, axis=0))), 1e-12)
        points, total = load_points(base / "resplat/resplat_world.ply", 20000, 0)
        pooled = np.concatenate([load_points(ROOT / "results" / exp / rel /
                                             "resplat/resplat_world.ply", 5000, 0)[0]
                                 for exp in STRATEGIES.values()])
        step = np.linalg.norm(np.diff(centers, axis=0), axis=1) / scale
        t = tracks(ROOT / info["source"] / info["scene"], train)
        universe = sorted(set().union(*t.values())) if t else []
        row = {"train_views": len(train), "test_views": len(split["test"]),
               "camera_bbox_diagonal_native_units": scale,
               "adjacent_train_step_normalized": describe(step),
               "train_track_count": len(universe), "strategies": {}}
        for strategy, experiment in STRATEGIES.items():
            work = ROOT / "results" / experiment / rel / "work"
            other = json.loads((work / "split.json").read_text())
            assert other["train"] == train and other["test"] == split["test"]
            selected = other["context"]
            ids = [train.index(n) for n in selected]
            context = [pose_map[n] for n in selected]
            c = centers[ids]
            d = np.linalg.norm(centers[:, None] - c[None, :], axis=2) / scale
            pair = np.linalg.norm(c[:, None] - c[None, :], axis=2) / scale
            nearest_pair = pair.copy()
            np.fill_diagonal(nearest_pair, np.inf)
            fgap = np.degrees(np.arccos(np.clip(forward @ forward[ids].T, -1, 1))).min(1)
            orbit = centers - np.median(points, axis=0)
            orbit /= np.maximum(np.linalg.norm(orbit, axis=1, keepdims=True), 1e-12)
            ogap = np.degrees(np.arccos(np.clip(orbit @ orbit[ids].T, -1, 1))).min(1)
            not_selected = [i for i in range(len(train)) if i not in ids]
            result = {"context": selected, "train_index_gaps": describe(np.diff(sorted(ids))),
                      "train_to_context_distance_normalized": describe(d[not_selected].min(1)),
                      "context_pair_distance_normalized": describe(pair[np.triu_indices(len(c), 1)]),
                      "context_nearest_pair_distance_normalized": describe(nearest_pair.min(1)),
                      "train_to_context_forward_angle_deg": describe(fgap[not_selected]),
                      "train_to_context_orbit_angle_deg": describe(ogap[not_selected]),
                      "baseline_points": coverage(points, context, camera),
                      "pooled_points": coverage(pooled, context, camera)}
            if universe:
                selected_counts = np.zeros(len(universe), dtype=int)
                lookup = {key: i for i, key in enumerate(universe)}
                for name in selected:
                    for key in t[name]:
                        selected_counts[lookup[key]] += 1
                result["train_observed_track_coverage"] = {
                    "at_least_1": float((selected_counts >= 1).mean()),
                    "at_least_2": float((selected_counts >= 2).mean()),
                    "histogram": np.bincount(selected_counts, minlength=17).tolist()}
            row["strategies"][strategy] = result
            print(group, strategy, "C1/C2", round(result["baseline_points"]["at_least_1"]*100, 3),
                  round(result["baseline_points"]["at_least_2"]*100, 3),
                  "pooledC1", round(result["pooled_points"]["at_least_1"]*100, 3),
                  "nearest_dist", round(result["train_to_context_distance_normalized"]["mean"], 4),
                  "pairNN", round(result["context_nearest_pair_distance_normalized"]["mean"], 4),
                  "trackC1", result.get("train_observed_track_coverage", {}).get("at_least_1"))
        report["scenes"][group] = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
