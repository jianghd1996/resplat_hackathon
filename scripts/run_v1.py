"""Run one v1 configuration across the three benchmark scenes on two GPUs.

Example: python scripts/run_v1.py --backend no_reset --repeat r0
The result directory includes backend, budget, seed and repeat; never overwrites.
"""
import argparse
import concurrent.futures
import datetime
import json
import os
import queue
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = json.loads((ROOT / "baseline/scenes.json").read_text())
RESPLAT_PY = Path("/home/wangyb/miniconda3/envs/resplat/bin/python")
FASTGS_PY = Path("/home/wangyb/miniconda3/envs/fastgs/bin/python")


def job(group, info, backend, seed, repeat, gpu, commit):
    label = f"baseline_A_{backend}7000_seed{seed}{repeat}_B"
    output = ROOT / "results" / label / group / info["scene"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)
    cmd = [str(RESPLAT_PY), str(ROOT / "baseline/run_pipeline.py"),
           "--scene", str(ROOT / info["source"] / info["scene"]),
           "--output", str(output), "--group", group,
           "--module-a", "uniform16", "--module-b", backend,
           "--iterations", "7000", "--fastgs-python", str(FASTGS_PY)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), V1_SEED=str(seed),
               PYTHONUNBUFFERED="1")
    manifest = {"backend": backend, "iterations": 7000, "seed": seed,
                "repeat": repeat, "gpu_physical_index": gpu, "git_commit": commit,
                "command": cmd, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (output / "v1_run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / "pipeline.log").open("w") as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                cwd=ROOT, check=False)
    manifest["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest["exit_code"] = result.returncode
    (output / "v1_run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"{group} failed on GPU {gpu}; inspect {output / 'pipeline.log'}")
    print(f"{group} complete on GPU {gpu}: {output}", flush=True)
    return str(output)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("no_reset", "terminal_guard", "topology_freeze"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", required=True, help="Unique identifier, e.g. r0 or r1")
    ap.add_argument("--groups", nargs="+", choices=tuple(SCENES), default=list(SCENES))
    args = ap.parse_args()
    if not args.repeat.isalnum():
        ap.error("--repeat must be alphanumeric")
    if not all(p.is_file() for p in (RESPLAT_PY, FASTGS_PY)):
        ap.error("configured conda Python missing")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tasks = [(group, SCENES[group]) for group in args.groups]
    for group, info in tasks:
        target = ROOT / "results" / f"baseline_A_{args.backend}7000_seed{args.seed}{args.repeat}_B" / group / info["scene"]
        if target.exists():
            ap.error(f"experiment output already exists: {target}")
    pending = queue.Queue()
    for item in tasks:
        pending.put(item)

    def gpu_worker(gpu):
        while True:
            try:
                group, info = pending.get_nowait()
            except queue.Empty:
                break
            try:
                job(group, info, args.backend, args.seed, args.repeat, gpu, commit)
            finally:
                pending.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(gpu_worker, gpu) for gpu in (0, 1)]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
