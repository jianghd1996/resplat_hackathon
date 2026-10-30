"""Run baseline, quality SOTA, and per-scene classic-3DGS references.

Completed outputs are reused.  An existing but incomplete output is never
overwritten, so interrupted jobs remain available for diagnosis.
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import queue
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENES = json.loads((ROOT / "baseline/scenes_full.json").read_text())
RESPLAT_PY = Path("/home/wangyb/miniconda3/envs/resplat/bin/python")
FASTGS_PY = Path("/home/wangyb/miniconda3/envs/fastgs/bin/python")
BASELINE = ROOT / "results/baseline_A_baseline_B"
SOTA = ROOT / "results/baseline_A_topology_opacity7000_B"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def completed_pipeline(output, iterations):
    """Accept both current reports and the complete legacy symlink layout."""
    if (output / "report.json").is_file():
        return True
    required = (
        output / "work/module_a.json",
        output / "resplat/resplat.json",
        output / "module_b/module_b.json",
        output / "module_b/metrics.json",
        output / f"module_b/point_cloud/iteration_{iterations}/point_cloud.ply",
    )
    return all(path.is_file() for path in required)


def completed_reference(output):
    record = output / "module_b.json"
    ply = output / "point_cloud/iteration_30000/point_cloud.ply"
    if not record.is_file() or not ply.is_file():
        return False
    return json.loads(record.read_text()).get("iterations") == 30000


def pipeline_job(scene, variant, gpu, commit):
    if variant == "baseline":
        output, backend, iterations = (
            BASELINE / scene["group"] / scene["scene"], "fastgs", 10000)
    else:
        output, backend, iterations = (
            SOTA / scene["group"] / scene["scene"], "topology_opacity", 7000)
    if completed_pipeline(output, iterations):
        return f"reuse {variant} {scene['group']}/{scene['scene']}"
    if output.exists():
        raise FileExistsError(f"incomplete output exists; refusing overwrite: {output}")
    output.mkdir(parents=True)
    command = [
        str(RESPLAT_PY), str(ROOT / "baseline/run_pipeline.py"),
        "--scene", str(ROOT / scene["source"] / scene["scene"]),
        "--output", str(output), "--group", scene["group"],
        "--module-a", "uniform16", "--module-b", backend,
        "--iterations", str(iterations), "--fastgs-python", str(FASTGS_PY),
    ]
    manifest = {
        "variant": variant, "module_a": "uniform16", "module_b": backend,
        "iterations": iterations, "seed": 0, "gpu_physical_index": gpu,
        "git_commit": commit, "command": command, "started_utc": utc_now(),
    }
    manifest_path = output / "full_eval_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    with (output / "full_eval.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    manifest.update(finished_utc=utc_now(), exit_code=result.returncode)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"pipeline failed: {output / 'full_eval.log'}")
    return f"complete {variant} {scene['group']}/{scene['scene']} on GPU {gpu}"


def reference_job(scene, gpu, commit):
    experiment = BASELINE / scene["group"] / scene["scene"]
    work = experiment / "work"
    output = experiment / "reference_3dgs"
    if completed_reference(output):
        return f"reuse reference {scene['group']}/{scene['scene']}"
    if output.exists():
        raise FileExistsError(f"incomplete reference exists; refusing overwrite: {output}")
    if not (work / "split.json").is_file():
        raise FileNotFoundError(f"baseline work directory missing: {work}")
    output.mkdir(parents=True)
    command = [
        str(FASTGS_PY), str(ROOT / "baseline/module_b/train_from_ply.py"),
        "-s", str(work), "-m", str(output), "--eval", "-r", "1",
        "--sh_degree", "3", "--iterations", "30000",
        "--position_lr_max_steps", "30000", "--densify_until_iter", "15000",
        "--init_random", "--quiet",
    ]
    manifest = {
        "variant": "classic_3dgs_reference", "iterations": 30000, "seed": 0,
        "gpu_physical_index": gpu, "git_commit": commit, "command": command,
        "started_utc": utc_now(),
    }
    manifest_path = output / "full_eval_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTHONPATH=str(ROOT / "baseline/3dgs") + os.pathsep + env.get("PYTHONPATH", ""))
    with (output / "full_eval.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    manifest.update(finished_utc=utc_now(), exit_code=result.returncode)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"reference failed: {output / 'full_eval.log'}")
    return f"complete reference {scene['group']}/{scene['scene']} on GPU {gpu}"


def run_jobs(tasks, gpus, commit):
    pending = queue.Queue()
    for task in tasks:
        pending.put(task)

    def worker(gpu):
        while True:
            try:
                kind, scene, variant = pending.get_nowait()
            except queue.Empty:
                return
            try:
                if kind == "pipeline":
                    message = pipeline_job(scene, variant, gpu, commit)
                else:
                    message = reference_job(scene, gpu, commit)
                print(message, flush=True)
            finally:
                pending.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pipelines", "reference", "all"), default="all")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--groups", nargs="+", choices=("dl3dv", "hw", "mipnerf360"))
    args = parser.parse_args()
    if len(args.gpus) != len(set(args.gpus)):
        parser.error("GPU IDs must be unique")
    if not all(path.is_file() for path in (RESPLAT_PY, FASTGS_PY)):
        parser.error("configured conda Python missing")
    scenes = [s for s in SCENES if not args.groups or s["group"] in args.groups]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    if args.phase in ("pipelines", "all"):
        tasks = [("pipeline", scene, variant) for scene in scenes
                 for variant in ("baseline", "sota")]
        run_jobs(tasks, args.gpus, commit)
    if args.phase in ("reference", "all"):
        tasks = [("reference", scene, None) for scene in scenes]
        run_jobs(tasks, args.gpus, commit)


if __name__ == "__main__":
    main()
