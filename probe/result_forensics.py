"""Audit saved metrics/PLYs and residuals; never changes or retrains an experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = {
    "fps_A_baseline_B": "baseline_A_baseline_B",
    "coverage_local_A_baseline_B": "baseline_A_baseline_B",
    "quality_filter_A_baseline_B": "baseline_A_baseline_B",
    "baseline_A_short_budget7000_B": "baseline_A_baseline_B",
    "baseline_A_short_schedule7000_B": "baseline_A_short_budget7000_B",
    "baseline_A_no_reset7000_B": "baseline_A_short_schedule7000_B",
    "baseline_A_no_reset5000_B": "baseline_A_short_schedule5000_B",
    "baseline_A_regularized7000_B": "baseline_A_short_schedule7000_B",
    "baseline_A_topk_growth7000_B": "baseline_A_no_reset7000_B",
    "baseline_A_coarse_to_fine7000_B": "baseline_A_no_reset7000_B",
    "baseline_A_long_axis_split7000_B": "baseline_A_no_reset7000_B",
    "baseline_A_perceptual_loss7000_B": "baseline_A_no_reset7000_B",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ply_stats(path):
    v = PlyData.read(path, mmap="c")["vertex"].data
    opacity = 1/(1+np.exp(-v["opacity"].astype(float)))
    scales = np.exp(np.column_stack([v[f"scale_{i}"] for i in range(3)]).astype(float))
    aspect = scales.max(1)/np.maximum(scales.min(1), 1e-15)
    return {"count": len(v), "opacity_quantiles": np.quantile(opacity, [0, .1, .5, .9, 1]).tolist(),
            "opacity_gt_08": float((opacity > .800001).mean()),
            "aspect_ratio_quantiles": np.quantile(aspect, [.5, .9, .99]).tolist(),
            "scale_sorted_medians": np.median(np.sort(scales, axis=1), axis=0).tolist(),
            "fraction_min_at_model_floor": float(np.isclose(scales.min(1), 1e-6, rtol=1e-5, atol=0).mean()),
            "fraction_min_at_export_floor": float((scales.min(1) < 1.0001e-9).mean()),
            "median_middle_to_max": float(np.median(np.sort(scales, axis=1)[:, 1]/scales.max(1))),
            "scale_max_quantiles": np.quantile(scales.max(1), [.5, .9, .99]).tolist()}


def residual_stats(model):
    rows = []
    for p in sorted((model / "test_renders").glob("*.png")):
        gt = np.asarray(Image.open(model / "test_gt" / p.name).convert("RGB"), dtype=float)/255
        image = np.asarray(Image.open(p).convert("RGB"), dtype=float)/255
        d = image-gt
        mse = float(np.mean(d*d))
        dc = float(np.mean(d.mean((0, 1))**2))
        h, w, _ = d.shape
        assert h % 16 == 0 and w % 16 == 0
        block = d.reshape(h//16, 16, w//16, 16, 3).mean((1, 3))
        block_energy = float(np.mean(block*block))
        rows.append({"name": p.name, "mse": mse, "dc_fraction": dc/max(mse, 1e-12),
                     "block16_mean_fraction": block_energy/max(mse, 1e-12),
                     "rgb_bias": d.mean((0, 1)).tolist()})
    return {"per_view": rows, "mean_mse": float(np.mean([r["mse"] for r in rows])),
            "mean_dc_fraction": float(np.mean([r["dc_fraction"] for r in rows])),
            "mean_block16_fraction": float(np.mean([r["block16_mean_fraction"] for r in rows]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/rethink_v0/result_forensics.json")
    args = ap.parse_args()
    comparisons = []
    for exp, ctrl in CONTROLS.items():
        for path in sorted((ROOT / "results" / exp).glob("*/*/module_b/metrics.json")):
            rel = path.parents[1].relative_to(ROOT / "results" / exp)
            other = ROOT / "results" / ctrl / rel
            if not (other / "module_b/metrics.json").exists():
                continue
            current = path.parents[1]
            a = json.loads((other / "module_b/metrics.json").read_text())
            b = json.loads(path.read_text())
            index = {v["name"]: v for v in a["per_view"]}
            assert set(index) == {v["name"] for v in b["per_view"]}
            record = {"experiment": exp, "control": ctrl, "scene": str(rel), "metrics": {}}
            for metric in ("psnr", "ssim", "lpips"):
                diff = np.array([v[metric]-index[v["name"]][metric] for v in b["per_view"]])
                worse = diff > 0 if metric == "lpips" else diff < 0
                worst = np.argsort(-diff if metric == "lpips" else diff)[:3]
                record["metrics"][metric] = {"mean_delta": float(diff.mean()), "median_delta": float(np.median(diff)),
                     "worse_views": int(worse.sum()), "views": len(diff),
                     "worst_three": [{"name": b["per_view"][int(i)]["name"], "delta": float(diff[i])} for i in worst]}
            record["same_resplat_ply"] = sha(current / "resplat/resplat_world.ply") == sha(other / "resplat/resplat_world.ply")
            record["same_context"] = json.loads((current / "work/split.json").read_text())["context"] == json.loads((other / "work/split.json").read_text())["context"]
            comparisons.append(record)
            print(exp, str(rel)[:18], "samePLY", record["same_resplat_ply"],
                  "PSNR median/worse", record["metrics"]["psnr"]["median_delta"], record["metrics"]["psnr"]["worse_views"],
                  "LPIPS worse", record["metrics"]["lpips"]["worse_views"], flush=True)
    gaussian = {}
    residual = {}
    for exp in ("baseline_A_no_reset7000_B", "baseline_A_long_axis_split7000_B"):
        for model in sorted((ROOT / "results" / exp).glob("*/*/module_b")):
            key = str(model.relative_to(ROOT / "results"))
            gaussian[key] = ply_stats(model / "point_cloud/iteration_7000/point_cloud.ply")
            if "dl3dv" in key:
                residual[key] = residual_stats(model)
    inventory = [{"path": str(p.relative_to(ROOT)), "mean": json.loads(p.read_text())["mean"]}
                 for p in sorted((ROOT / "results").glob("*_A_*_B/*/*/module_b/metrics.json"))]
    initial = {str(p.relative_to(ROOT)): ply_stats(p) for p in
               sorted((ROOT / "results/baseline_A_baseline_B").glob("*/*/resplat/resplat_world.ply"))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"comparisons": comparisons, "gaussian_stats": gaussian,
                                   "dl3dv_residuals": residual, "initial_gaussians": initial,
                                   "completed_inventory": inventory}, indent=2)+"\n")


if __name__ == "__main__":
    main()
