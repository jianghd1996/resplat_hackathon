"""Audit every local scene using poses, train-only feature tracks and image proxies.

No reconstruction/training and no test-image scores are used for selection.
Sharpness is computed on raw images resized to 256 square, so it is a dataset
proxy, not a reproduction of the pipeline's two-stage resize quality filter.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "probe"))
from module_a.preprocess import load_colmap_poses, load_transforms_json
from utils_A import select_context
from context_geometry import describe


def track_sets(scene, names):
    path = scene / "sparse/0/images.bin"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "census_colmap", ROOT / "baseline/3dgs/scene/colmap_loader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images = module.read_extrinsics_binary(str(path))
    by_name = {i.name: set(map(int, i.point3D_ids[i.point3D_ids >= 0]))
               for i in images.values()}
    return {n: by_name[n] for n in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/rethink_v0/dataset_census.json")
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    for dataset in ("DL3DV", "hw", "mipnerf360"):
        for scene in sorted((ROOT / "datasets" / dataset).iterdir()):
            if not scene.is_dir():
                continue
            info = load_colmap_poses(str(scene)) or load_transforms_json(str(scene))
            if info is None:
                rows.append({"dataset": dataset, "scene": scene.name, "status": "no usable poses"})
                continue
            image_dir = Path(info.get("images_dir", scene / ("images_4" if dataset == "DL3DV" else "images")))
            files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            names = [p.name for p in files if p.name in info["poses"]]
            train = [n for i, n in enumerate(names) if i % 8 != 0]
            centers = np.stack([-info["poses"][n][0].T @ info["poses"][n][1] for n in train])
            forward = np.stack([info["poses"][n][0].T[:, 2] for n in train])
            scale = max(float(np.linalg.norm(np.ptp(centers, axis=0))), 1e-12)
            sv = np.linalg.svd(centers - centers.mean(0), compute_uv=False)**2
            track = track_sets(scene, train)
            universe = set().union(*track.values()) if track else set()
            features = []
            for name in train:
                im = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)
                if im is None:
                    raise FileNotFoundError(image_dir / name)
                gray = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
                features.append({"name": name, "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                                 "brightness": float(gray.mean()/255),
                                 "contrast": float(gray.std()/255),
                                 "near_clipped_fraction": float(((gray < 5) | (gray > 250)).mean())})
            lap = np.array([f["laplacian_variance"] for f in features])
            uniform = select_context("uniform16", range(len(train)), 16)
            row = {"dataset": dataset, "scene": scene.name, "status": "ok",
                   "pose_source": info["source"], "image_files": len(files),
                   "registered_available": len(names), "train": len(train), "test": len(names)-len(train),
                   "track_count_train_only": len(universe),
                   "fov_x_deg": float(np.degrees(2*np.arctan(info["W"]/(2*info["fx"])))),
                   "camera_pca_variance_ratio": (sv/sv.sum()).tolist(),
                   "adjacent_step_normalized": describe(np.linalg.norm(np.diff(centers, axis=0), axis=1)/scale),
                   "path_length_normalized": float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()/scale),
                   "adjacent_forward_angle_deg": describe(np.degrees(np.arccos(np.clip((forward[:-1]*forward[1:]).sum(1), -1, 1)))),
                   "image_proxies": {key: describe([f[key] for f in features]) for key in features[0] if key != "name"},
                   "uniform_low_laplacian_count": int((lap[uniform] < .8*np.median(lap)).sum()),
                   "per_train_image_proxies": features, "strategies": {}}
            for strategy in ("uniform16", "fps", "coverage_local"):
                ids = select_context(strategy, range(len(train)), 16, centers=centers, directions=forward)
                unselected = [i for i in range(len(train)) if i not in ids]
                d = np.linalg.norm(centers[:, None]-centers[ids][None, :], axis=2)/scale
                gap = np.degrees(np.arccos(np.clip(forward @ forward[ids].T, -1, 1))).min(1)
                value = {"context": [train[i] for i in ids],
                         "nearest_distance_normalized": describe(d[unselected].min(1)),
                         "nearest_forward_gap_deg": describe(gap[unselected]),
                         "train_index_gaps": describe(np.diff(ids))}
                if universe:
                    union = set().union(*(track[train[i]] for i in ids))
                    coverage = [len(track[n] & union)/len(track[n]) for n in train if track[n]]
                    value["track_coverage_fraction"] = len(union)/len(universe)
                    value["per_train_view_track_coverage"] = describe(coverage)
                    value["per_train_view_track_coverage_p10"] = float(np.quantile(coverage, .1))
                row["strategies"][strategy] = value
            rows.append(row)
            print(dataset, scene.name[:12], len(names), "tracks", len(universe),
                  "low_lap", row["uniform_low_laplacian_count"],
                  "d_uniform/fps/local", [round(v["nearest_distance_normalized"]["mean"], 4) for v in row["strategies"].values()], flush=True)
    report = {"llffhold": 8, "image_proxy_resolution": [256, 256],
              "scope": "all local scene directories, not the public datasets in their entirety",
              "elapsed_seconds": time.time()-t0, "scenes": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
