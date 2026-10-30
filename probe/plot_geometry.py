"""Render dataset-wide geometry comparisons from completed offline probes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "rethink_v0_geometry"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/rethink_v0/geometry.svg")
    args = ap.parse_args()
    rows = json.loads((ROOT / "results/rethink_v0/dataset_census.json").read_text())["scenes"]
    swaps = {(r["dataset"], r["scene"]): r for r in
             json.loads((ROOT / "results/rethink_v0/covisibility_swaps.json").read_text())["scenes"]}
    labels = [f'{r["dataset"]}/{r["scene"][:12]}' for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    styles = [("uniform16", "Uniform", "#64748b", "o", -.21),
              ("fps", "FPS", "#2563eb", "s", -.07),
              ("coverage_local", "Local pairs", "#c2410c", "^", .07),
              ("guarded", "Guarded swaps", "#15803d", "D", .21)]
    y = np.arange(len(rows))
    for key, label, color, marker, offset in styles:
        dist, coverage = [], []
        for r in rows:
            base = r["strategies"]["uniform16"]["nearest_distance_normalized"]["mean"]
            swap = swaps[(r["dataset"], r["scene"])]
            if key == "guarded":
                stats = swap.get("guarded")
                dist.append(stats["nearest_distance_normalized"]/base if stats else 1)
                coverage.append(stats["track_coverage_fraction"]*100 if stats else np.nan)
            else:
                stats = r["strategies"][key]
                dist.append(stats["nearest_distance_normalized"]["mean"]/base)
                coverage.append(stats.get("track_coverage_fraction", np.nan)*100)
        axes[0].scatter(dist, y+offset, color=color, marker=marker, s=24, label=label)
        axes[1].scatter(coverage, y+offset, color=color, marker=marker, s=24)
    axes[0].axvline(1, color="#94a3b8", linewidth=1, linestyle="--")
    axes[0].set(xlabel="Mean nearest-camera distance / uniform (lower is closer)",
                title="All 13 local scenes: spatial coverage", xlim=(.7, 2.05))
    axes[1].set(xlabel="Observed train-track coverage (%)", title="7 scenes with tracks: content-support proxy",
                xlim=(15, 92))
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        for separator in (5.5, 9.5):
            ax.axhline(separator, color="#94a3b8", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    for i in range(6):
        axes[1].text(52, i, "No track observations in supplied input", ha="center",
                     va="center", fontsize=8, color="#64748b")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Geometry probes only: improvements here are not reconstruction-quality results", fontsize=11)
    fig.tight_layout(rect=(0, .045, 1, .95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, metadata={"Date": None})
    plt.close(fig)
    if args.out.suffix.lower() == ".svg":
        args.out.write_text("\n".join(line.rstrip() for line in args.out.read_text().splitlines())+"\n")
    print(args.out)


if __name__ == "__main__":
    main()
