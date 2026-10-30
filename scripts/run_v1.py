"""Run one fixed-seed v1 configuration across the three benchmark scenes on two GPUs.

Example: python scripts/run_v1.py --backend terminal_guard
The result directory follows the established A/B module naming convention.
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


def job(group, info, backend, gpu, commit):
    label = f"baseline_A_{backend}7000_B"
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
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    manifest = {"backend": backend, "iterations": 7000, "seed": 0,
                "gpu_physical_index": gpu, "git_commit": commit,
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
    ap.add_argument("--backend", choices=("terminal_guard", "topology_freeze"), required=True)
    ap.add_argument("--groups", nargs="+", choices=tuple(SCENES), default=list(SCENES))
    ap.add_argument("--gpus", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    args = ap.parse_args()
    if not all(p.is_file() for p in (RESPLAT_PY, FASTGS_PY)):
        ap.error("configured conda Python missing")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tasks = [(group, SCENES[group]) for group in args.groups]
    for group, info in tasks:
        target = ROOT / "results" / f"baseline_A_{args.backend}7000_B" / group / info["scene"]
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
                job(group, info, args.backend, gpu, commit)
            finally:
                pending.task_done()

    if len(set(args.gpus)) != len(args.gpus):
        ap.error("GPU IDs must be unique")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(gpu_worker, gpu) for gpu in args.gpus]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
