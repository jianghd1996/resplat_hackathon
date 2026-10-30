"""Describe candidate Module-A conditions; do not fit or apply a routing rule.

Reads existing all-scene train-only census and guarded-swap diagnostics.
Angular gap / horizontal FOV is only a pose proxy, not visibility or overlap.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "results/rethink_v0"
    census = json.loads((folder / "dataset_census.json").read_text())["scenes"]
    swaps = {(r["dataset"], r["scene"]): r for r in
             json.loads((folder / "covisibility_swaps.json").read_text())["scenes"]}
    rows = []
    for scene in census:
        uniform = scene["strategies"]["uniform16"]
        candidate = swaps[(scene["dataset"], scene["scene"])]
        row = {
            "dataset": scene["dataset"], "scene": scene["scene"],
            "fov_x_deg": scene["fov_x_deg"],
            "uniform_forward_gap_p95_over_fov_x":
                uniform["nearest_forward_gap_deg"]["p95"] / scene["fov_x_deg"],
            "uniform_distance_p95_normalized": uniform["nearest_distance_normalized"]["p95"],
            "uniform_context_fraction": len(uniform["context"]) / scene["train"],
            "track_data_available": "track_coverage_fraction" in uniform,
            "uniform_track_coverage": uniform.get("track_coverage_fraction"),
            "uniform_per_view_support_p10": uniform.get("per_train_view_track_coverage_p10"),
            "candidate_track_gain_percentage_points": None,
            "candidate_support_p10_gain_percentage_points": None,
        }
        if "guarded" in candidate:
            row["candidate_track_gain_percentage_points"] = 100 * (
                candidate["guarded"]["track_coverage_fraction"] - candidate["uniform"]["track_coverage_fraction"])
            row["candidate_support_p10_gain_percentage_points"] = 100 * (
                candidate["guarded"]["per_train_view_support_p10"] - candidate["uniform"]["per_train_view_support_p10"])
        rows.append(row)
        print(scene["dataset"], scene["scene"][:12],
              f'FOV={row["fov_x_deg"]:.1f}',
              f'gap/FOV={row["uniform_forward_gap_p95_over_fov_x"]:.3f}',
              "track_gain_pp=", row["candidate_track_gain_percentage_points"])
    report = {"scope": "descriptive conditions only; no policy fit or new reconstruction metrics",
              "missing_tracks": "unknown, never zero coverage",
              "angular_proxy_limit": "forward direction ignores roll, translation, vertical FOV and occlusion",
              "scenes": rows}
    (folder / "condition_audit.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
