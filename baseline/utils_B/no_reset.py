"""Short-schedule FastGS with all scheduled opacity resets disabled."""

import argparse
import json
import os
import subprocess
from pathlib import Path

if __package__:
    from .fastgs import ROOT, render_command, train_command
else:
    from fastgs import ROOT, render_command, train_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "render"])
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init-ply")
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.iterations < 10000:
        parser.error("no-reset iterations must be between 1 and 9999")

    if args.action == "train":
        if not args.init_ply:
            parser.error("train requires --init-ply")
        command = train_command(args.work, args.out, args.init_ply, args.iterations)
        command.extend(["--skip_final_opacity_reset", "--position_lr_max_steps",
                        str(args.iterations), "--disable_opacity_reset", "--seed",
                        os.environ.get("V1_SEED", "0")])
    else:
        command = render_command(args.work, args.out, args.iterations)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "fastgs") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, env=env, check=True)

    if args.action == "train":
        path = Path(args.out) / "module_b.json"
        result = json.loads(path.read_text())
        result["adapter"] = "no_reset"
        result["budget_iterations"] = args.iterations
        result["position_lr_max_steps"] = args.iterations
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
