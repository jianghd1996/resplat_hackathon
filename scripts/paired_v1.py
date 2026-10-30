"""Audit v1 scene runs against same-seed control at per-view resolution."""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = json.loads((ROOT / "baseline/scenes.json").read_text())


def load_run(path):
    required = (path / "module_b/metrics.json", path / "module_b/module_b.json")
    if not all(p.exists() for p in required):
        return None
    return [json.loads(p.read_text()) for p in required]


def compare(candidate, control):
    cm, cb = candidate
    rm, rb = control
    ref = {v["name"]: v for v in rm["per_view"]}
    assert set(ref) == {v["name"] for v in cm["per_view"]}
    report = {"input_ply_equal": cb["init_ply_sha256"] == rb["init_ply_sha256"],
              "B_seconds_delta": round(cb["total_seconds"] - rb["total_seconds"], 2),
              "metrics": {}}
    for key in ("psnr", "ssim", "lpips"):
        diffs = sorted(v[key] - ref[v["name"]][key] for v in cm["per_view"])
        report["metrics"][key] = {"mean_delta": round(sum(diffs)/len(diffs), 6),
                                   "median_delta": round(statistics.median(diffs), 6),
                                   "worse_views": sum(v > 0 if key == "lpips" else v < 0 for v in diffs),
                                   "views": len(diffs), "worst_delta": round(diffs[-1] if key == "lpips" else diffs[0], 6)}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat", default="r0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = {}
    for backend in ("terminal_guard", "topology_freeze"):
        for group, info in SCENES.items():
            rel = Path(group) / info["scene"]
            candidate = load_run(ROOT / "results" / f"baseline_A_{backend}7000_seed{args.seed}{args.repeat}_B" / rel)
            control = load_run(ROOT / "results" / f"baseline_A_no_reset7000_seed{args.seed}{args.repeat}_B" / rel)
            if candidate and control:
                data[f"{backend}/{group}"] = compare(candidate, control)
    if args.repeat == "r1":
        for group, info in SCENES.items():
            rel = Path(group) / info["scene"]
            control0 = load_run(ROOT / "results" / "baseline_A_no_reset7000_seed0r0_B" / rel)
            control1 = load_run(ROOT / "results" / "baseline_A_no_reset7000_seed0r1_B" / rel)
            if control0 and control1:
                data[f"repeat_no_reset/{group}"] = compare(control1, control0)
    out = ROOT / "results/v1_diagnostics" / f"paired_seed{args.seed}{args.repeat}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
