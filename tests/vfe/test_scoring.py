from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


from vfe.scoring import load_annotations, metrics
from score_paper import mean_scores


def _fixture():
    uid = "test_task:000007"
    episodes = {uid: {"source_episode_index": 7, "length": 3, "tasks": ["test task"]}}
    annotation = {
        "episode_uid": uid,
        "episode_index": 7,
        "num_frames": 3,
        "task": "test task",
        "fps": 30,
        "success": "success",
        "interpolation": "lead_window_smootherstep",
        "window_seconds": 2.5,
        "points": [
            {"frame_index": i, "timestamp": round(i / 30, 4), "progress": p}
            for i, p in enumerate((0.0, 50.0, 100.0))
        ],
    }
    return uid, episodes, annotation


class AnnotationSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_zip_and_directory_load_identical_dense_targets(self):
        tmp_path = self.root
        uid, episodes, annotation = _fixture()
        name = uid.replace(":", "__") + ".json"
        raw = json.dumps(annotation).encode()
        folder = tmp_path / "annotations"
        folder.mkdir()
        (folder / name).write_bytes(raw)
        archive = tmp_path / "annotations.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("nested/" + name, raw)
            z.writestr("nested/unused_episode.json", "{}")
        direct, direct_manifest = load_annotations(tmp_path, folder, episodes, 30)
        zipped, zip_manifest = load_annotations(tmp_path, archive, episodes, 30)
        np.testing.assert_array_equal(direct[uid], [-1.0, -0.5, 0.0])
        np.testing.assert_array_equal(zipped[uid], direct[uid])
        assert direct_manifest[0]["sha256"] == zip_manifest[0]["sha256"]
        assert zip_manifest[0]["window_seconds"] == 2.5
        assert zip_manifest[0]["path"] == "annotations.zip!/nested/" + name
        assert len(zip_manifest) == 1
        assert not (tmp_path / "nested").exists()


    def test_zip_missing_selected_episode_is_rejected(self):
        tmp_path = self.root
        _, episodes, _ = _fixture()
        archive = tmp_path / "missing.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("unrelated.json", "{}")
        with self.assertRaisesRegex(ValueError, "Missing annotation in ZIP"):
            load_annotations(tmp_path, archive, episodes, 30)


    def test_zip_duplicate_basename_is_rejected(self):
        tmp_path = self.root
        uid, episodes, annotation = _fixture()
        archive = tmp_path / "duplicate.zip"
        name = uid.replace(":", "__") + ".json"
        with zipfile.ZipFile(archive, "w") as z:
            for prefix in ("a/", "b/"):
                z.writestr(prefix + name, json.dumps(annotation))
        with self.assertRaisesRegex(ValueError, "Duplicate annotation basename"):
            load_annotations(tmp_path, archive, episodes, 30)


    def test_zip_misaligned_frame_is_rejected(self):
        uid, episodes, _ = _fixture()
        for field in ("frame_index", "timestamp"):
            with self.subTest(field=field):
                _, _, annotation = _fixture()
                annotation["points"][1][field] = 100
                archive = self.root / (field + ".zip")
                with zipfile.ZipFile(archive, "w") as z:
                    z.writestr(uid.replace(":", "__") + ".json", json.dumps(annotation))
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    load_annotations(self.root, archive, episodes, 30)


class KendallMetricTests(unittest.TestCase):
    def test_monotonic_and_reversed(self):
        target = np.arange(5, dtype=np.float32)
        self.assertAlmostEqual(metrics(target, target)["kendall"], 1.0)
        self.assertAlmostEqual(metrics(target, target[::-1])["kendall"], -1.0)

    def test_ties_in_one_or_both_curves(self):
        # Four concordant pairs, one target-only tie, and one joint tie.
        target = np.asarray([0, 0, 1, 1], dtype=np.float32)
        prediction = np.asarray([0, 1, 2, 2], dtype=np.float32)
        self.assertAlmostEqual(metrics(target, prediction)["kendall"], 4 / np.sqrt(4 * 5))
        # Four concordant pairs, one discordant pair, and one joint tie.
        target = np.asarray([0, 0, 1, 2], dtype=np.float32)
        prediction = np.asarray([0, 0, 2, 1], dtype=np.float32)
        self.assertAlmostEqual(metrics(target, prediction)["kendall"], 3 / 5)

    def test_degenerate_curves_are_zero(self):
        for target, prediction in (([0], [1]), ([0, 0], [0, 1]), ([0, 1], [1, 1])):
            self.assertEqual(metrics(np.asarray(target), np.asarray(prediction))["kendall"], 0.0)

    def test_context_then_query_equal_average(self):
        first_query = mean_scores([
            dict.fromkeys(("mae", "mse", "pearson", "kendall"), 0.2),
            dict.fromkeys(("mae", "mse", "pearson", "kendall"), 0.6),
        ])
        second_query = dict.fromkeys(("mae", "mse", "pearson", "kendall"), 1.0)
        self.assertAlmostEqual(mean_scores([first_query, second_query])["kendall"], 0.7)


if __name__ == "__main__":
    unittest.main()
