"""No-reset 7000-step continuation, skipping only the final topology event."""

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
    if args.iterations != 7000:
        parser.error("terminal_guard is defined at 7000 iterations in v1")
    if args.action == "train":
        if not args.init_ply:
            parser.error("train requires --init-ply")
        command = train_command(args.work, args.out, args.init_ply, args.iterations)
        command.extend(["--skip_final_opacity_reset", "--position_lr_max_steps", "7000",
                        "--disable_opacity_reset", "--skip_final_densification", "--seed",
                        os.environ.get("V1_SEED", "0")])
    else:
        command = render_command(args.work, args.out, args.iterations)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "fastgs") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, env=env, check=True)
    if args.action == "train":
        path = Path(args.out) / "module_b.json"
        record = json.loads(path.read_text())
        record.update(adapter="terminal_guard", budget_iterations=7000)
        path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
