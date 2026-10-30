#!/usr/bin/env python3
"""Batch ReSplat-only context-selection ablations with cached preprocessing.

Phases are deliberately separated: preprocess every scene once, generate all
ReSplat PLYs, render/evaluate all completed PLYs, then aggregate JSON/CSV/MD.
The script is resumable and supports one worker per listed GPU.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from utils_A import STRATEGIES, select_context  # noqa: E402
from visualize_context_coverage import read_images  # noqa: E402


GROUP_DIRS = {"dl3dv": "DL3DV", "mipnerf360": "mipnerf360"}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def valid_cached_work(work):
    return all((work / relative).exists() for relative in (
        "split.json", "module_a.json", "images", "sparse/0/cameras.txt",
        "sparse/0/images.txt"))


def discover_scenes(dataset_root, groups, filters):
    scenes = []
    allowed = set(filters or [])
    for group in groups:
        root = dataset_root / GROUP_DIRS[group]
        if not root.is_dir():
            raise FileNotFoundError(f"dataset group missing: {root}")
        for scene in sorted(path for path in root.iterdir() if path.is_dir()):
            if allowed and scene.name not in allowed and f"{group}/{scene.name}" not in allowed:
                continue
            scenes.append((group, scene.name, scene.resolve()))
    if not scenes:
        raise RuntimeError("no scenes selected")
    return scenes


def run_logged(command, log_path, env, prefix):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{prefix}] $ {' '.join(map(str, command))}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(value) for value in command], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{prefix}] {line}", end="", flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def prepare_cache(args, scenes):
    env = os.environ.copy()
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus[0]
    for group, scene_name, scene_path in scenes:
        work = args.output / "cache" / group / scene_name / "work"
        if valid_cached_work(work) and not args.force:
            print(f"[prepare] cached: {group}/{scene_name}")
            continue
        command = [
            args.resplat_python, ROOT / "module_a/preprocess.py",
            "--scene_dir", scene_path, "--group", group, "--out", work,
            "--module-a", "uniform16", "--num_context", args.num_context,
        ]
        run_logged(command, args.output / "logs" / f"prepare_{group}_{scene_name}.log",
                   env, f"prepare:{group}/{scene_name}")


def ensure_link(link, target):
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"expected a symlink managed by the batch script: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def materialize_selection(args, group, scene_name, strategy):
    cache = args.output / "cache" / group / scene_name / "work"
    if not valid_cached_work(cache):
        raise FileNotFoundError(f"preprocessed cache missing: {cache}")
    work = args.output / "runs" / strategy / group / scene_name / "work"
    work.mkdir(parents=True, exist_ok=True)
    ensure_link(work / "images", cache / "images")
    ensure_link(work / "sparse", cache / "sparse")

    split = read_json(cache / "split.json")
    order = list(split["images"])
    train_set = set(split["train"])
    test_set = set(split["test"])
    train_indices = [index for index, name in enumerate(order) if name in train_set]
    test_indices = [index for index, name in enumerate(order) if name in test_set]
    poses = read_images(cache / "sparse/0/images.txt")
    start = time.perf_counter()
    selected_positions = select_context(
        strategy, train_indices, args.num_context,
        order=order, test_indices=test_indices, poses={
            name: (poses[name]["R"], poses[name]["t"]) for name in order
        },
    )
    selection_seconds = time.perf_counter() - start
    context = [order[train_indices[position]] for position in selected_positions]
    if len(context) != args.num_context or len(set(context)) != len(context):
        raise RuntimeError(
            f"{strategy} returned {len(context)} views, expected {args.num_context} unique views")
    split["context"] = context
    split["module_a"] = strategy
    write_json(work / "split.json", split)

    cached_info = read_json(cache / "module_a.json")
    info = dict(cached_info)
    info.update({
        "module_a": strategy,
        "selection_seconds": selection_seconds,
        "cached_preprocess_seconds": cached_info.get("total_seconds"),
        "total_seconds": selection_seconds,
        "cache_work": str(cache.resolve()),
    })
    write_json(work / "module_a.json", info)
    write_json(work / "selection.json", {
        "strategy": strategy, "num_context": len(context),
        "selection_seconds": selection_seconds, "context": context,
    })
    return work


def build_tasks(args, scenes):
    tasks = []
    for strategy in args.strategies:
        for group, scene_name, _ in scenes:
            work = materialize_selection(args, group, scene_name, strategy)
            run_root = args.output / "runs" / strategy / group / scene_name
            tasks.append({
                "strategy": strategy, "group": group, "scene": scene_name,
                "work": work, "root": run_root,
            })
    return tasks


def run_gpu_partitions(tasks, gpus, worker, failures):
    partitions = [tasks[index::len(gpus)] for index in range(len(gpus))]

    def run_partition(gpu, partition):
        for task in partition:
            try:
                worker(task, gpu)
            except Exception as exc:  # continue large batches and summarize failures
                failure = {
                    "stage": worker.__name__, "gpu": gpu,
                    "strategy": task["strategy"], "group": task["group"],
                    "scene": task["scene"], "error": repr(exc),
                }
                failures.append(failure)
                print(f"[FAILED] {failure}", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_partition, gpu, partition)
                   for gpu, partition in zip(gpus, partitions)]
        for future in futures:
            future.result()


def infer_all(args, tasks, failures):
    def infer(task, gpu):
        out = task["root"] / "resplat"
        if (out / "resplat_world.ply").is_file() and (out / "resplat.json").is_file() \
                and not args.force:
            print(f"[infer] cached: {task['strategy']}/{task['group']}/{task['scene']}")
            return
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        command = [
            args.resplat_python, ROOT / "run_resplat.py",
            "--work", task["work"], "--out", out,
            "--checkpoint", args.checkpoint,
        ]
        name = f"{task['strategy']}_{task['group']}_{task['scene']}"
        run_logged(command, args.output / "logs" / f"infer_{name}.log", env,
                   f"gpu{gpu}:infer:{name}")

    run_gpu_partitions(tasks, args.gpus, infer, failures)


def link_eval_ply(eval_root, ply):
    point_dir = eval_root / "point_cloud/iteration_1"
    point_dir.mkdir(parents=True, exist_ok=True)
    link = point_dir / "point_cloud.ply"
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == ply.resolve():
            return
        link.unlink()
    link.symlink_to(ply.resolve())


def evaluate_all(args, tasks, failures):
    def evaluate(task, gpu):
        ply = task["root"] / "resplat/resplat_world.ply"
        if not ply.is_file():
            raise FileNotFoundError(f"ReSplat inference missing: {ply}")
        eval_root = task["root"] / "resplat_eval"
        metrics_path = eval_root / "metrics.json"
        if metrics_path.is_file() and not args.force:
            print(f"[evaluate] cached: {task['strategy']}/{task['group']}/{task['scene']}")
            return
        link_eval_ply(eval_root, ply)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        name = f"{task['strategy']}_{task['group']}_{task['scene']}"
        render = [
            args.fastgs_python, ROOT / "utils_B/fastgs.py", "render",
            "--work", task["work"], "--out", eval_root, "--iterations", "1",
        ]
        run_logged(render, args.output / "logs" / f"render_{name}.log", env,
                   f"gpu{gpu}:render:{name}")
        metric = [args.resplat_python, ROOT / "evaluate.py", "--model_path", eval_root]
        run_logged(metric, args.output / "logs" / f"metrics_{name}.log", env,
                   f"gpu{gpu}:metrics:{name}")

    run_gpu_partitions(tasks, args.gpus, evaluate, failures)


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def summarize(args, tasks, failures):
    rows = []
    for task in tasks:
        metrics_path = task["root"] / "resplat_eval/metrics.json"
        timing_path = task["root"] / "resplat/resplat.json"
        selection_path = task["work"] / "selection.json"
        cache_info_path = args.output / "cache" / task["group"] / task["scene"] / "work/module_a.json"
        status = "ok" if metrics_path.is_file() else "missing"
        row = {
            "group": task["group"], "scene": task["scene"],
            "strategy": task["strategy"], "status": status,
            "psnr": None, "ssim": None, "lpips": None,
            "num_test_views": None, "preprocess_seconds": None,
            "selection_seconds": None, "resplat_seconds": None,
        }
        if cache_info_path.is_file():
            row["preprocess_seconds"] = read_json(cache_info_path).get("total_seconds")
        if selection_path.is_file():
            row["selection_seconds"] = read_json(selection_path).get("selection_seconds")
        if timing_path.is_file():
            row["resplat_seconds"] = read_json(timing_path).get("total_seconds")
        if metrics_path.is_file():
            metrics = read_json(metrics_path)
            row.update(metrics["mean"])
            row["num_test_views"] = metrics["num_views"]
        rows.append(row)
        write_json(task["root"] / "report.json", row)

    successful = [row for row in rows if row["status"] == "ok"]
    aggregate = []
    for group in args.groups:
        for strategy in args.strategies:
            subset = [row for row in successful
                      if row["group"] == group and row["strategy"] == strategy]
            if not subset:
                continue
            psnr_mean, psnr_std = mean_std([row["psnr"] for row in subset])
            ssim_mean, ssim_std = mean_std([row["ssim"] for row in subset])
            lpips_mean, lpips_std = mean_std([row["lpips"] for row in subset])
            time_mean, time_std = mean_std([row["resplat_seconds"] for row in subset])
            aggregate.append({
                "group": group, "strategy": strategy,
                "scenes": len(subset), "psnr_mean": psnr_mean, "psnr_std": psnr_std,
                "ssim_mean": ssim_mean, "ssim_std": ssim_std,
                "lpips_mean": lpips_mean, "lpips_std": lpips_std,
                "resplat_seconds_mean": time_mean, "resplat_seconds_std": time_std,
            })

    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output / "per_scene.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "summary.json", {
        "per_scene": rows, "aggregate": aggregate, "failures": failures,
    })

    lines = [
        "# ReSplat Context-Selection Ablation", "",
        "## Aggregate by dataset", "",
        "| group | strategy | scenes | PSNR mean±std | SSIM mean±std | LPIPS mean±std | ReSplat s mean±std |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['group']} | {row['strategy']} | {row['scenes']} | "
            f"{row['psnr_mean']:.3f}±{row['psnr_std']:.3f} | "
            f"{row['ssim_mean']:.4f}±{row['ssim_std']:.4f} | "
            f"{row['lpips_mean']:.4f}±{row['lpips_std']:.4f} | "
            f"{row['resplat_seconds_mean']:.1f}±{row['resplat_seconds_std']:.1f} |")
    lines.extend([
        "", "## Per scene", "",
        "| group | scene | strategy | status | PSNR | SSIM | LPIPS | ReSplat s |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ])
    for row in rows:
        def show(value, digits):
            return "—" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {row['group']} | {row['scene']} | {row['strategy']} | {row['status']} | "
            f"{show(row['psnr'], 3)} | {show(row['ssim'], 4)} | "
            f"{show(row['lpips'], 4)} | {show(row['resplat_seconds'], 1)} |")
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{value}`" for value in failures)
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def resolve_checkpoint(value):
    if value is not None:
        return value.resolve()
    name = "resplat-base-dl3dv-540x960-view16-a72dc6d0.pth"
    for candidate in (REPO / "resplat/pretrained" / name, REPO / "pretrained" / name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("ReSplat checkpoint missing; pass --checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO / "datasets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=sorted(GROUP_DIRS),
                        default=["dl3dv", "mipnerf360"])
    parser.add_argument("--strategies", nargs="+", choices=sorted(STRATEGIES),
                        default=[
                            "uniform16", "random16", "pose_fps", "orbit_fps",
                            "oracle_test_nearest", "oracle_test_nearest_pairs",
                        ])
    parser.add_argument("--scenes", nargs="*",
                        help="optional scene names or group/name selectors")
    parser.add_argument("--num-context", type=int, default=16)
    parser.add_argument("--stage", choices=["all", "prepare", "infer", "evaluate", "summarize"],
                        default="all")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                        help="comma-separated physical GPU IDs; one sequential worker per GPU")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resplat-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--fastgs-python", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.dataset_root = args.dataset_root.resolve()
    args.output = args.output.resolve()
    args.gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    args.checkpoint = resolve_checkpoint(args.checkpoint)
    args.resplat_python = args.resplat_python.resolve()
    args.fastgs_python = (args.fastgs_python or args.resplat_python).resolve()
    if not args.gpus:
        parser.error("--gpus cannot be empty")
    if args.num_context < 1:
        parser.error("--num-context must be positive")
    scenes = discover_scenes(args.dataset_root, args.groups, args.scenes)
    failures = []

    if args.stage in ("all", "prepare"):
        prepare_cache(args, scenes)
    elif not all(valid_cached_work(args.output / "cache" / group / scene / "work")
                 for group, scene, _ in scenes):
        raise RuntimeError("cached preprocessing is incomplete; run --stage prepare first")

    tasks = build_tasks(args, scenes)
    if args.stage in ("all", "infer"):
        infer_all(args, tasks, failures)
    if args.stage in ("all", "evaluate"):
        evaluate_all(args, tasks, failures)
    if args.stage in ("all", "summarize"):
        summarize(args, tasks, failures)
    if failures:
        print(f"[batch] completed with {len(failures)} failures; see summary.json", file=sys.stderr)


if __name__ == "__main__":
    main()
