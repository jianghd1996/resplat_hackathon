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
from utils_B import short_budget  # noqa: E402
from collect_results import collect  # noqa: E402
import run_pipeline  # noqa: E402


class PluginInterfaceTest(unittest.TestCase):
    def test_uniform16_matches_previous_linspace_policy(self):
        train_indices = list(range(56))
        self.assertEqual(select_context("uniform16", train_indices, 16),
                         [0, 3, 7, 11, 14, 18, 22, 25, 29, 33, 36, 40, 44, 47, 51, 55])
        self.assertEqual(select_context("uniform16", list(range(8)), 16), list(range(8)))

    def test_fps_uses_training_camera_geometry_and_is_deterministic(self):
        train_indices = [1, 2, 3, 4]
        centers = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0]]
        self.assertEqual(select_context("fps", train_indices, 2, centers), [0, 3])
        self.assertEqual(select_context("fps", train_indices, 2, centers), [0, 3])
        self.assertEqual(select_context("fps", train_indices, 8, centers), [0, 1, 2, 3])

    def test_fastgs_adapter_preserves_training_and_render_flags(self):
        train = train_command("work", "model", "init.ply", 10000)
        self.assertEqual(train[2:], ["-s", "work", "-m", "model", "--eval", "-r", "1",
                                      "--sh_degree", "3", "--iterations", "10000",
                                      "--init_ply", "init.ply", "--quiet"])
        render = render_command("work", "model", 10000)
        self.assertEqual(render[2:], ["-s", "work", "-m", "model", "--eval", "-r", "1",
                                       "--backend", "fastgs", "--iteration", "10000", "--quiet"])

    def test_short_schedule_changes_only_learning_rate_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls = []

            def fake_run(command, **_):
                calls.append(command)
                (out / "module_b.json").write_text('{"total_seconds": 10}')

            args = ["short_budget.py", "train", "--work", "work", "--out", str(out),
                    "--init-ply", "init.ply", "--iterations", "7000"]
            with patch.object(short_budget.subprocess, "run", side_effect=fake_run):
                with patch.object(sys, "argv", args):
                    short_budget.main(short_schedule=False)
                with patch.object(sys, "argv", args):
                    short_budget.main(short_schedule=True)
            self.assertEqual(calls[0] + ["--position_lr_max_steps", "7000"], calls[1])
            self.assertIn("--skip_final_opacity_reset", calls[0])
            result = json.loads((out / "module_b.json").read_text())
            self.assertEqual(result["adapter"], "short_schedule")
            self.assertEqual(result["position_lr_max_steps"], 7000)

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

            # A new official run in the same directory must win over the
            # archived report, which may describe an older checkpoint.
            (scene / "module_b").mkdir()
            (scene / "module_b/metrics.json").write_text(json.dumps({
                "num_views": 9, "mean": {"psnr": 23.0, "ssim": 0.8, "lpips": 0.15}}))
            (scene / "work").mkdir()
            (scene / "work/module_a.json").write_text('{"total_seconds": 5}')
            (scene / "resplat").mkdir()
            (scene / "resplat/resplat.json").write_text('{"total_seconds": 6}')
            (scene / "module_b/module_b.json").write_text('{"total_seconds": 7}')
            refreshed = collect(scene, "hw", "billboard")
            self.assertEqual(refreshed["psnr"], 23.0)
            self.assertEqual(refreshed["test_views"], 9)
            self.assertEqual(refreshed["total_seconds"], 18)

    def test_reference_budget_uses_measured_training_iterations(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)
            reference = scene / "reference_3dgs/module_b.json"
            reference.parent.mkdir(parents=True)
            reference.write_text(json.dumps({"iterations": 1000,
                                             "train_seconds": 20.0,
                                             "total_seconds": 25.0}))
            probe = collect(scene, "hw", "billboard")
            self.assertEqual(probe["reference_T_iter_seconds"], 0.02)
            self.assertIsNone(probe["reference_30k_seconds"])
            reference.write_text(json.dumps({"iterations": 30000,
                                             "train_seconds": 600.0,
                                             "total_seconds": 615.0}))
            full = collect(scene, "hw", "billboard")
            self.assertEqual(full["reference_30k_B_budget_seconds"], 600.0)
            self.assertEqual(full["reference_30k_seconds"], 615.0)

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
