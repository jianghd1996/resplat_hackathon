import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module_a"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocess  # noqa: E402
import rebuild_hw_sparse  # noqa: E402


class FakeModel:
    images = {1: object()}

    def write(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class PycolmapPreprocessTest(unittest.TestCase):
    def test_rebuild_archives_marked_model_and_pairs_undistorted_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "billboard"
            (scene / "images").mkdir(parents=True)
            (scene / "sparse/0").mkdir(parents=True)
            (scene / "frames_info.txt").write_text(
                "0 123 1 1 0 0 64 64 0 0 0 0 0 0 1\n")
            Image.new("RGB", (64, 64)).save(scene / "images/123.jpg")
            (scene / "sparse/0/conversion_summary.json").write_text("{}")

            def fake_reconstruct(paths, work):
                self.assertEqual([Path(p).name for p in paths], ["123.jpg"])
                model = Path(work) / "model"
                images = Path(work) / "undistorted"
                model.mkdir(parents=True)
                images.mkdir()
                (model / "cameras.bin").write_bytes(b"camera")
                (model / "images.bin").write_bytes(b"poses")
                Image.new("RGB", (32, 32)).save(images / "123.jpg")
                return {"model_dir": str(model), "images_dir": str(images),
                        "poses": {"123.jpg": object()}}

            with patch.object(rebuild_hw_sparse, "run_pycolmap", fake_reconstruct):
                rebuild_hw_sparse.rebuild(scene, root / "backup")
            self.assertTrue((root / "backup/billboard/sparse/0/conversion_summary.json").is_file())
            self.assertTrue((scene / "sparse/0/pycolmap_undistorted.json").is_file())
            with Image.open(scene / "images_undistorted/123.jpg") as image:
                self.assertEqual(image.size, (32, 32))

    def test_cpu_fallback_undistortion_and_paired_images(self):
        calls = []

        def extract_features(**kwargs):
            calls.append(("extract", kwargs))
            if kwargs["device"] == "cuda":
                raise ValueError("CUDA SIFT unavailable")

        def undistort_images(**kwargs):
            calls.append(("undistort", kwargs))
            out = Path(kwargs["output_path"])
            (out / "sparse").mkdir(parents=True)
            (out / "images").mkdir()
            Image.new("RGB", (64, 64)).save(out / "images/view.jpg")

        fake = types.SimpleNamespace(
            Device=types.SimpleNamespace(cuda="cuda", cpu="cpu"),
            CameraMode=types.SimpleNamespace(SINGLE="single"),
            extract_features=extract_features,
            match_exhaustive=lambda **kw: calls.append(("match", kw)),
            incremental_mapping=lambda **kw: {0: FakeModel()},
            undistort_images=undistort_images,
        )
        cam = types.SimpleNamespace(model="PINHOLE", width=64, height=64,
                                    params=[50, 50, 32, 32])
        img = types.SimpleNamespace(name="view.jpg", qvec=np.array([1, 0, 0, 0]),
                                    tvec=np.zeros(3))
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene"
            scene.mkdir()
            Image.new("RGB", (80, 80)).save(scene / "view.jpg")
            with patch.dict(sys.modules, {"pycolmap": fake}), \
                 patch.object(preprocess, "read_intrinsics_binary", return_value={1: cam}), \
                 patch.object(preprocess, "read_extrinsics_binary", return_value={1: img}):
                result = preprocess.run_pycolmap([str(scene / "view.jpg")],
                                                 str(Path(tmp) / "work"))
            self.assertEqual(result["source"], "pycolmap_run")
            self.assertEqual([c[1]["device"] for c in calls if c[0] == "extract"],
                             ["cuda", "cpu"])
            self.assertEqual(next(c[1] for c in calls if c[0] == "extract")["camera_model"],
                             "OPENCV")
            self.assertEqual(next(c[1] for c in calls if c[0] == "match")["device"],
                             "cpu")
            self.assertTrue(os.path.isfile(os.path.join(result["images_dir"], "view.jpg")))
            self.assertEqual(preprocess.natural_image_list(str(scene), result)[0][1],
                             os.path.join(result["images_dir"], "view.jpg"))


if __name__ == "__main__":
    unittest.main()
