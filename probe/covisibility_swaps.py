"""Train-track-only directional probe: two guarded swaps starting from uniform16.

This is an offline hypothesis test, not a registered Module A or quality result.
No images, test tracks, ReSplat output, or rendering scores guide selection.
"""
import json
import time
from pathlib import Path

import numpy as np

from dataset_census import ROOT, load_colmap_poses, load_transforms_json, track_sets


def main():
    census = json.loads((ROOT / "results/rethink_v0/dataset_census.json").read_text())
    rows = []
    for row in census["scenes"]:
        started = time.perf_counter()
        scene = ROOT / "datasets" / row["dataset"] / row["scene"]
        names = [v["name"] for v in row["per_train_image_proxies"]]
        uniform = [names.index(n) for n in row["strategies"]["uniform16"]["context"]]
        selected = uniform.copy()
        track = track_sets(scene, names)
        result = {"dataset": row["dataset"], "scene": row["scene"], "swaps": []}
        if not track or not any(track.values()):
            result.update(status="uniform fallback: no train tracks", context=[names[i] for i in selected])
            rows.append(result)
            continue
        universe = sorted(set().union(*track.values()))
        lookup = {key: i for i, key in enumerate(universe)}
        indices = [np.array([lookup[key] for key in track[n]], dtype=int) for n in names]
        info = load_colmap_poses(str(scene)) or load_transforms_json(str(scene))
        centers = np.stack([-info["poses"][n][0].T @ info["poses"][n][1] for n in names])
        distance = np.linalg.norm(centers[:, None]-centers[None, :], axis=2)
        distance /= max(float(np.linalg.norm(np.ptp(centers, axis=0))), 1e-12)

        def stats(ids):
            counts = np.zeros(len(universe), dtype=np.int16)
            for i in ids:
                counts[indices[i]] += 1
            visible = counts > 0
            support = [visible[x].mean() for x in indices if len(x)]
            unselected = [i for i in range(len(names)) if i not in ids]
            return counts, {"track_coverage_fraction": float(visible.mean()),
                            "per_train_view_support_p10": float(np.quantile(support, .1)),
                            "nearest_distance_normalized": float(distance[unselected][:, ids].min(1).mean()),
                            "max_train_index_gap": int(np.diff(sorted(ids)).max())}

        counts, initial = stats(selected)
        boundaries = np.array([(uniform[i]+uniform[i+1])/2 for i in range(len(uniform)-1)])
        bins = np.searchsorted(boundaries, np.arange(len(names)), side="right")
        for step in range(2):
            candidates = []
            for slot in range(1, len(selected)-1):  # Preserve both endpoints.
                remaining = counts.copy()
                remaining[indices[selected[slot]]] -= 1
                lost = np.count_nonzero(counts)-np.count_nonzero(remaining)
                for candidate in np.flatnonzero(bins == slot):
                    if candidate in selected:
                        continue
                    gain = np.count_nonzero(remaining[indices[candidate]] == 0)-lost
                    if gain > 0:
                        candidates.append((-int(gain), slot, int(candidate)))
            accepted = False
            for neg_gain, slot, candidate in sorted(candidates):
                trial = selected.copy()
                trial[slot] = candidate
                new_counts, after = stats(trial)
                if (after["nearest_distance_normalized"] > 1.1*initial["nearest_distance_normalized"]
                        or after["per_train_view_support_p10"] < initial["per_train_view_support_p10"]
                        or after["max_train_index_gap"] > 2*initial["max_train_index_gap"]):
                    continue
                result["swaps"].append({"from": names[selected[slot]], "to": names[candidate],
                                        "new_tracks_net": -neg_gain})
                selected, counts, accepted = trial, new_counts, True
                break
            if not accepted:
                break
        _, final = stats(selected)
        result.update(status="offline only", uniform=initial, guarded=final,
                      context=[names[i] for i in selected], elapsed_seconds=time.perf_counter()-started)
        rows.append(result)
        print(row["dataset"], row["scene"], initial["track_coverage_fraction"],
              final["track_coverage_fraction"], len(result["swaps"]), flush=True)
    out = ROOT / "results/rethink_v0/covisibility_swaps.json"
    out.write_text(json.dumps({"max_swaps": 2, "distance_ratio_guard": 1.1,
                               "scope": "offline train-observation proxy only", "scenes": rows}, indent=2)+"\n")


if __name__ == "__main__":
    main()
