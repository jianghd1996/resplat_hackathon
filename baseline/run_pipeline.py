"""Single-scene A -> fixed ReSplat -> B -> render/evaluate entry point.

Run with the configured ReSplat Python; pass --fastgs-python only when the
FastGS extensions live in a different Python environment.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
from utils_A import STRATEGIES  # noqa: E402
from utils_B import BACKENDS  # noqa: E402


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def infer_group(scene):
    return {"DL3DV": "dl3dv", "hw": "hw", "mipnerf360": "mipnerf360"}.get(
        scene.parent.name, "custom")


def run(command, env=None):
    print("[pipeline]", " ".join(map(str, command)), flush=True)
    subprocess.run([str(item) for item in command], check=True, env=env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", help="Defaults to the scene parent directory")
    parser.add_argument("--module-a", choices=sorted(STRATEGIES), default="uniform16")
    parser.add_argument("--module-b", choices=sorted(BACKENDS), default="fastgs")
    parser.add_argument("--num-context", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument(
        "--resplat-only", action="store_true",
        help="stop after ReSplat, render its world-space PLY on test views, and evaluate",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resplat-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--fastgs-python", type=Path)
    args = parser.parse_args(argv)

    scene = args.scene.resolve()
    output = args.output.resolve()
    if not scene.is_dir():
        parser.error(f"scene directory does not exist: {scene}")
    if args.num_context < 1 or args.iterations < 1:
        parser.error("--num-context and --iterations must be positive")
    checkpoint = args.checkpoint
    if checkpoint is None:
        name = "resplat-base-dl3dv-540x960-view16-a72dc6d0.pth"
        checkpoint = next((p for p in (
            REPO / "resplat/pretrained" / name, REPO / "pretrained" / name)
            if p.is_file()), None)
    if checkpoint is None or not checkpoint.is_file():
        parser.error("ReSplat checkpoint missing; pass --checkpoint PATH")

    resplat_python = args.resplat_python.resolve()
    fastgs_python = (args.fastgs_python or args.resplat_python).resolve()
    for executable in (resplat_python, fastgs_python):
        if not executable.is_file():
            parser.error(f"Python executable missing: {executable}")

    group = args.group or infer_group(scene)
    work = output / "work"
    resplat = output / "resplat"
    model = output / "module_b"
    output.mkdir(parents=True, exist_ok=True)

    run([resplat_python, ROOT / "module_a/preprocess.py", "--scene_dir", scene,
         "--group", group, "--out", work, "--module-a", args.module_a,
         "--num_context", args.num_context])
    run([resplat_python, ROOT / "run_resplat.py", "--work", work,
         "--out", resplat, "--checkpoint", checkpoint])
    fastgs_env = os.environ.copy()
    fastgs_env["PYTHONPATH"] = str(ROOT / "fastgs") + os.pathsep + fastgs_env.get("PYTHONPATH", "")

    if args.resplat_only:
        # Reuse the standard FastGS renderer without running any optimization.
        # Scene expects the conventional point_cloud/iteration_N layout.
        eval_model = output / "resplat_eval"
        eval_iteration = 1
        eval_point_dir = eval_model / "point_cloud" / f"iteration_{eval_iteration}"
        eval_point_dir.mkdir(parents=True, exist_ok=True)
        resplat_ply = resplat / "resplat_world.ply"
        if not resplat_ply.is_file():
            raise FileNotFoundError(f"ReSplat PLY missing: {resplat_ply}")
        shutil.copy2(resplat_ply, eval_point_dir / "point_cloud.ply")
        run([fastgs_python, ROOT / "utils_B" / f"{args.module_b}.py", "render",
             "--work", work, "--out", eval_model,
             "--iterations", eval_iteration], fastgs_env)
        run([resplat_python, ROOT / "evaluate.py", "--model_path", eval_model])

        a = read_json(work / "module_a.json")
        r = read_json(resplat / "resplat.json")
        metrics = read_json(eval_model / "metrics.json")
        report = {
            "scene": str(scene), "group": group,
            "module_a": args.module_a, "module_b": args.module_b,
            "resplat_only": True,
            "checkpoint": str(checkpoint.resolve()),
            "num_test_views": metrics["num_views"],
            "metrics": metrics["mean"],
            "time_A": a["total_seconds"], "time_resplat": r["total_seconds"],
            "time_B": 0.0,
            "time_total": round(a["total_seconds"] + r["total_seconds"], 3),
            "pose_source": a["pose_source"],
            "model_ply": str(resplat_ply),
            "renders": str(eval_model / "test_renders"),
        }
        with open(output / "report.json", "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    run([fastgs_python, ROOT / "utils_B" / f"{args.module_b}.py", "train",
         "--work", work, "--out", model, "--init-ply", resplat / "resplat_world.ply",
         "--iterations", args.iterations], fastgs_env)
    run([fastgs_python, ROOT / "utils_B" / f"{args.module_b}.py", "render",
         "--work", work, "--out", model, "--iterations", args.iterations], fastgs_env)
    run([resplat_python, ROOT / "evaluate.py", "--model_path", model])

    trained_ply = model / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
    if not trained_ply.is_file():
        raise FileNotFoundError(f"trained PLY missing: {trained_ply}")
    shutil.copy2(trained_ply, output / "model.ply")
    a = read_json(work / "module_a.json")
    r = read_json(resplat / "resplat.json")
    b = read_json(model / "module_b.json")
    metrics = read_json(model / "metrics.json")
    report = {
        "scene": str(scene), "group": group,
        "module_a": args.module_a, "module_b": args.module_b,
        "checkpoint": str(checkpoint.resolve()),
        "num_test_views": metrics["num_views"],
        "metrics": metrics["mean"],
        "time_A": a["total_seconds"], "time_resplat": r["total_seconds"],
        "time_B": b["total_seconds"],
        "time_total": round(a["total_seconds"] + r["total_seconds"] + b["total_seconds"], 3),
        "pose_source": a["pose_source"],
        "model_ply": str(output / "model.ply"),
        "renders": str(model / "test_renders"),
    }
    with open(output / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
