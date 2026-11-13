from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from vfe.data import VfeFrameDataset
from vfe.labels import (
    ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION,
    LEGACY_GROUND_TRUTH_VERSION,
    PER_DEMO_GROUND_TRUTH_VERSION,
    annotation_progress_values,
    discretize_values,
    ground_truth_values,
    relative_progress_values,
    remaining_step_values,
    validate_ground_truth_version,
)


def test_legacy_remaining_steps_use_task_max_demo_length() -> None:
    np.testing.assert_allclose(
        remaining_step_values(demo_length=3, task_max_length=5),
        [-0.4, -0.2, 0.0],
    )
    np.testing.assert_allclose(
        remaining_step_values(demo_length=5, task_max_length=5),
        [-0.8, -0.6, -0.4, -0.2, 0.0],
    )


def test_missing_semantics_is_rejected() -> None:
    try:
        validate_ground_truth_version(None)
    except ValueError as exc:
        assert "must be declared explicitly" in str(exc)
    else:
        raise AssertionError("Missing target semantics silently selected a default")


def test_value_target_dispatch_supports_both_definitions() -> None:
    legacy = ground_truth_values(
        ground_truth_version=LEGACY_GROUND_TRUTH_VERSION,
        demo_length=3,
        task_max_length=5,
    )
    per_demo = ground_truth_values(
        ground_truth_version=PER_DEMO_GROUND_TRUTH_VERSION,
        demo_length=3,
        task_max_length=5,
    )
    np.testing.assert_allclose(legacy, [-0.4, -0.2, 0.0])
    np.testing.assert_allclose(per_demo, [-1.0, -0.5, 0.0])


def test_annotation_progress_clamps_then_maps_to_value_range() -> None:
    progress = np.asarray([-100.0, -1.0, 0.0, 50.0, 100.0, 125.0])
    values = annotation_progress_values(progress)
    np.testing.assert_allclose(values, [-1.0, -1.0, -1.0, -0.5, 0.0, 0.0])
    np.testing.assert_array_equal(discretize_values(values), [0, 0, 0, 100, 200, 200])

    dispatched = ground_truth_values(
        ground_truth_version=ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION,
        demo_length=len(progress),
        task_max_length=len(progress),
        annotation_progress=progress,
    )
    np.testing.assert_array_equal(dispatched, values)


def test_annotation_progress_rejects_missing_misaligned_or_nonfinite_values() -> None:
    for progress, expected_message in (
        (None, "annotation_progress is required"),
        (np.asarray([0.0]), "expected demo_length=2"),
        (np.asarray([0.0, np.nan]), "finite values"),
    ):
        try:
            ground_truth_values(
                ground_truth_version=ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION,
                demo_length=2,
                task_max_length=2,
                annotation_progress=progress,
            )
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError("Invalid annotation progress was accepted")


def test_dataset_requires_matching_preprocessed_data_config() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        processed_dir = Path(temp_dir)
        (processed_dir / "tasks").mkdir()
        (processed_dir / "labels").mkdir()
        np.savez(
            processed_dir / "labels/demo_0.npz",
            value_continuous=np.asarray([-1.0, -0.5, 0.0], dtype=np.float32),
            value_bin=np.asarray([0, 100, 200], dtype=np.int64),
            data_split_preset=np.asarray("new_config"),
        )
        task = {
            "task_id": 0,
            "language_instruction": "test",
            "task_max_demo_length": 5,
            "value_range": [-1.0, 0.0],
            "num_bins": 201,
            "demos": [
                {
                    "demo_id": "demo_0",
                    "demo_split": "train",
                    "demo_length": 3,
                    "label_path": "labels/demo_0.npz",
                    "source_hdf5": "unused.hdf5",
                }
            ],
        }
        (processed_dir / "tasks/task_000.json").write_text(json.dumps(task), encoding="utf-8")
        summary = {
            "value_range": [-1.0, 0.0],
            "num_bins": 201,
            "data_split_preset": "new_config",
            "tasks": [{"task_id": 0, "metadata_path": "tasks/task_000.json"}],
        }
        (processed_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        try:
            VfeFrameDataset(
                processed_dir,
                split="train",
                data_split_preset="legacy_config",
            )
        except ValueError as exc:
            assert "was preprocessed for data_split_preset" in str(exc)
        else:
            raise AssertionError("Dataset accepted a different ground-truth version")

        new_dataset = VfeFrameDataset(
            processed_dir,
            split="train",
            data_split_preset="new_config",
        )
        np.testing.assert_allclose(
            new_dataset.load_label(0, "demo_0")["value_continuous"],
            [-1.0, -0.5, 0.0],
        )


def test_relative_progress_spans_full_range_for_every_demo_length() -> None:
    short = relative_progress_values(3)
    long = relative_progress_values(7)

    np.testing.assert_allclose(short, [-1.0, -0.5, 0.0])
    assert short[0] == long[0] == -1.0
    assert short[-1] == long[-1] == 0.0
    assert np.all(np.diff(short) > 0)
    assert np.all(np.diff(long) > 0)


def test_relative_progress_uses_configured_endpoints() -> None:
    values = relative_progress_values(3, value_min=-2.0, value_max=1.0)
    np.testing.assert_allclose(values, [-2.0, -0.5, 1.0])


def test_single_frame_demo_is_terminal() -> None:
    np.testing.assert_array_equal(relative_progress_values(1), [0.0])


def test_relative_progress_rejects_invalid_arguments() -> None:
    try:
        relative_progress_values(0)
    except ValueError as exc:
        assert "demo_length must be positive" in str(exc)
    else:
        raise AssertionError("A non-positive demo length was accepted")

    try:
        relative_progress_values(3, value_min=0.0, value_max=0.0)
    except ValueError as exc:
        assert "value_max must be greater" in str(exc)
    else:
        raise AssertionError("An empty value range was accepted")


def test_relative_progress_endpoints_map_to_endpoint_bins() -> None:
    bins = discretize_values(relative_progress_values(11), num_bins=201)
    assert bins[0] == 0
    assert bins[-1] == 200


if __name__ == "__main__":
    test_legacy_remaining_steps_use_task_max_demo_length()
    test_missing_semantics_is_rejected()
    test_value_target_dispatch_supports_both_definitions()
    test_annotation_progress_clamps_then_maps_to_value_range()
    test_annotation_progress_rejects_missing_misaligned_or_nonfinite_values()
    test_dataset_requires_matching_preprocessed_data_config()
    test_relative_progress_spans_full_range_for_every_demo_length()
    test_relative_progress_uses_configured_endpoints()
    test_single_frame_demo_is_terminal()
    test_relative_progress_rejects_invalid_arguments()
    test_relative_progress_endpoints_map_to_endpoint_bins()
