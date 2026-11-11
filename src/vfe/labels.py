from __future__ import annotations

import numpy as np


LEGACY_GROUND_TRUTH_VERSION = "task_max_normalized_remaining_steps_v1"
PER_DEMO_GROUND_TRUTH_VERSION = "per_demo_relative_progress_v1"
ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION = "annotation_progress_clamped_v1"
SUPPORTED_GROUND_TRUTH_VERSIONS = (
    LEGACY_GROUND_TRUTH_VERSION,
    PER_DEMO_GROUND_TRUTH_VERSION,
    ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION,
)


def bin_values(num_bins: int = 201, value_min: float = -1.0, value_max: float = 0.0) -> np.ndarray:
    """Return scalar values represented by distributional bins."""

    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    if value_max <= value_min:
        raise ValueError("value_max must be greater than value_min")
    return np.linspace(value_min, value_max, num_bins, dtype=np.float32)


def relative_progress_values(
    demo_length: int,
    value_min: float = -1.0,
    value_max: float = 0.0,
) -> np.ndarray:
    """Linearly map one successful demo from ``value_min`` to ``value_max``."""

    if demo_length <= 0:
        raise ValueError("demo_length must be positive")
    if value_max <= value_min:
        raise ValueError("value_max must be greater than value_min")

    # A one-frame successful demo contains only its terminal state, so the
    # terminal endpoint takes precedence over the usual initial endpoint.
    if demo_length == 1:
        return np.asarray([value_max], dtype=np.float32)
    return np.linspace(value_min, value_max, demo_length, dtype=np.float32)


def remaining_step_values(
    demo_length: int,
    task_max_length: int,
    value_min: float = -1.0,
    value_max: float = 0.0,
) -> np.ndarray:
    """Return the legacy task-max-normalized negative remaining-step labels."""

    if demo_length <= 0:
        raise ValueError("demo_length must be positive")
    if task_max_length <= 0:
        raise ValueError("task_max_length must be positive")
    frame_ids = np.arange(demo_length, dtype=np.float32)
    values = -((demo_length - 1 - frame_ids) / float(task_max_length))
    values = np.clip(values, value_min, value_max).astype(np.float32)
    values[-1] = 0.0
    return values


def annotation_progress_values(
    progress: np.ndarray,
    value_min: float = -1.0,
    value_max: float = 0.0,
) -> np.ndarray:
    """Map annotation progress percentages to the configured value range.

    Annotation values below zero are treated as zero. Values above 100 are
    treated as 100 so malformed percentages cannot produce out-of-range
    training targets.
    """

    if value_max <= value_min:
        raise ValueError("value_max must be greater than value_min")
    progress = np.asarray(progress, dtype=np.float32)
    if progress.ndim != 1:
        raise ValueError("annotation progress must be a one-dimensional array")
    if progress.size == 0:
        raise ValueError("annotation progress must not be empty")
    if not np.isfinite(progress).all():
        raise ValueError("annotation progress must contain only finite values")
    normalized = np.clip(progress, 0.0, 100.0) / 100.0
    return (value_min + normalized * (value_max - value_min)).astype(np.float32)


def ground_truth_values(
    *,
    ground_truth_version: str,
    demo_length: int,
    task_max_length: int,
    value_min: float = -1.0,
    value_max: float = 0.0,
    annotation_progress: np.ndarray | None = None,
) -> np.ndarray:
    """Construct continuous targets under one of the supported definitions."""

    if ground_truth_version == LEGACY_GROUND_TRUTH_VERSION:
        return remaining_step_values(
            demo_length=demo_length,
            task_max_length=task_max_length,
            value_min=value_min,
            value_max=value_max,
        )
    if ground_truth_version == PER_DEMO_GROUND_TRUTH_VERSION:
        return relative_progress_values(
            demo_length=demo_length,
            value_min=value_min,
            value_max=value_max,
        )
    if ground_truth_version == ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION:
        if annotation_progress is None:
            raise ValueError(
                "annotation_progress is required for annotation-based ground truth"
            )
        if len(annotation_progress) != demo_length:
            raise ValueError(
                f"annotation progress has {len(annotation_progress)} frames; "
                f"expected demo_length={demo_length}"
            )
        return annotation_progress_values(
            annotation_progress,
            value_min=value_min,
            value_max=value_max,
        )
    raise ValueError(
        f"Unsupported ground-truth version {ground_truth_version!r}; "
        f"expected one of {SUPPORTED_GROUND_TRUTH_VERSIONS}"
    )


def validate_ground_truth_version(ground_truth_version: str | None) -> str:
    """Validate an explicitly declared ground-truth label definition."""

    if ground_truth_version is None:
        raise ValueError(
            "The ground-truth version must be declared explicitly by the data config"
        )
    if ground_truth_version not in SUPPORTED_GROUND_TRUTH_VERSIONS:
        raise ValueError(
            f"Unsupported ground-truth version {ground_truth_version!r}; "
            f"expected one of {SUPPORTED_GROUND_TRUTH_VERSIONS}"
        )
    return ground_truth_version


def discretize_values(
    values: np.ndarray,
    num_bins: int = 201,
    value_min: float = -1.0,
    value_max: float = 0.0,
) -> np.ndarray:
    """Map continuous values in [value_min, value_max] to integer bin ids."""

    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    scale = (num_bins - 1) / float(value_max - value_min)
    bins = np.rint((np.asarray(values, dtype=np.float32) - value_min) * scale)
    return np.clip(bins, 0, num_bins - 1).astype(np.int64)
