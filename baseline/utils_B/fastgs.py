"""Baseline B adapter: invoke the unchanged FastGS trainer and renderer."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def train_command(work, out, init_ply, iterations):
    return [sys.executable, str(ROOT / "module_b/train_from_ply_fastgs.py"),
            "-s", str(work), "-m", str(out), "--eval", "-r", "1",
            "--sh_degree", "3", "--iterations", str(iterations),
            "--init_ply", str(init_ply), "--quiet"]


def render_command(work, out, iterations):
    return [sys.executable, str(ROOT / "render_test.py"),
            "-s", str(work), "-m", str(out), "--eval", "-r", "1",
            "--backend", "fastgs", "--iteration", str(iterations), "--quiet"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "render"])
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init-ply")
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    if args.action == "train":
        if not args.init_ply:
            parser.error("train requires --init-ply")
        command = train_command(args.work, args.out, args.init_ply, args.iterations)
    else:
        command = render_command(args.work, args.out, args.iterations)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "fastgs") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
