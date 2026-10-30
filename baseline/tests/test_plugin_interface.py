import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils_A import select_context  # noqa: E402
from utils_B.fastgs import render_command, train_command  # noqa: E402
from collect_results import collect  # noqa: E402
import run_pipeline  # noqa: E402


class PluginInterfaceTest(unittest.TestCase):
    def test_uniform16_matches_previous_linspace_policy(self):
        train_indices = list(range(56))
        self.assertEqual(select_context("uniform16", train_indices, 16),
                         [0, 3, 7, 11, 14, 18, 22, 25, 29, 33, 36, 40, 44, 47, 51, 55])
        self.assertEqual(select_context("uniform16", list(range(8)), 16), list(range(8)))

    def test_fastgs_adapter_preserves_training_and_render_flags(self):
        train = train_command("work", "model", "init.ply", 10000)
        self.assertEqual(train[2:], ["-s", "work", "-m", "model", "--eval", "-r", "1",
                                      "--sh_degree", "3", "--iterations", "10000",
                                      "--init_ply", "init.ply", "--quiet"])
        render = render_command("work", "model", 10000)
        self.assertEqual(render[2:], ["-s", "work", "-m", "model", "--eval", "-r", "1",
                                       "--backend", "fastgs", "--iteration", "10000", "--quiet"])

    def test_collects_external_billboard_report_without_stage_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)
            (scene / "report.json").write_text(json.dumps({
                "psnr": 22.661, "ssim": 0.7985, "lpips": 0.1606,
                "A_seconds": 10.8169, "resplat_seconds": 35.86,
                "B_seconds": 196.68, "total_seconds": 243.4,
                "evaluated_views": 8, "pose_source": "colmap.bin",
            }))
            row = collect(scene, "hw", "billboard")
            self.assertEqual(row["psnr"], 22.661)
            self.assertEqual(row["test_views"], 8)
            self.assertEqual(row["total_seconds"], 243.4)

    def test_single_scene_entry_writes_standard_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "hw/billboard"
            scene.mkdir(parents=True)
            checkpoint = root / "weights.pth"
            checkpoint.write_bytes(b"stub")
            output = root / "results/baseline_A_baseline_B/hw/billboard"
            calls = []

            def fake_run(command, env=None):
                calls.append([str(item) for item in command])
                stage = Path(command[1]).name
                if stage == "preprocess.py":
                    work = output / "work"
                    work.mkdir(parents=True)
                    (work / "module_a.json").write_text(json.dumps({
                        "total_seconds": 1, "pose_source": "colmap.bin"}))
                elif stage == "run_resplat.py":
                    resplat = output / "resplat"
                    resplat.mkdir()
                    (resplat / "resplat.json").write_text('{"total_seconds": 2}')
                elif stage == "fastgs.py" and "train" in command:
                    model = output / "module_b"
                    ply = model / "point_cloud/iteration_10000/point_cloud.ply"
                    ply.parent.mkdir(parents=True)
                    ply.write_bytes(b"ply\n")
                    (model / "module_b.json").write_text('{"total_seconds": 3}')
                elif stage == "evaluate.py":
                    (output / "module_b/metrics.json").write_text(json.dumps({
                        "num_views": 8, "mean": {"psnr": 22.661,
                        "ssim": 0.7985, "lpips": 0.1606}}))

            with patch.object(run_pipeline, "run", side_effect=fake_run):
                run_pipeline.main(["--scene", str(scene), "--output", str(output),
                                   "--checkpoint", str(checkpoint)])
            self.assertEqual(len(calls), 5)
            self.assertEqual(calls[0][calls[0].index("--module-a") + 1], "uniform16")
            self.assertEqual((output / "model.ply").read_bytes(), b"ply\n")
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["time_total"], 6)
            self.assertEqual(report["metrics"]["lpips"], 0.1606)


if __name__ == "__main__":
    unittest.main()
