from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data_splits import DEFAULT_PRESET_PATH, load_preset_preprocess_config, versioned_processed_dir
from .labels import (
    ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION,
    discretize_values,
    ground_truth_values,
    validate_ground_truth_version,
)
from .io import save_demo_labels, write_json


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_episode_annotation_progress(
    *,
    annotation_root: Path,
    episode_uid: str,
    demo_length: int,
) -> tuple[np.ndarray, Path]:
    """Load and strictly align one continuous annotation with a LeRobot episode."""

    annotation_path = annotation_root / f"{episode_uid.replace(':', '__')}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Missing annotation for episode_uid={episode_uid!r}: {annotation_path}"
        )
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if str(payload.get("episode_uid")) != episode_uid:
        raise ValueError(
            f"Annotation {annotation_path} records episode_uid={payload.get('episode_uid')!r}; "
            f"expected {episode_uid!r}"
        )
    points = payload.get("points")
    if not isinstance(points, list) or len(points) != demo_length:
        point_count = len(points) if isinstance(points, list) else None
        raise ValueError(
            f"Annotation {annotation_path} has {point_count} points; "
            f"expected demo_length={demo_length}"
        )
    frame_indices = [point.get("frame_index") for point in points]
    if frame_indices != list(range(demo_length)):
        raise ValueError(
            f"Annotation {annotation_path} frame_index values are not exactly "
            f"0..{demo_length - 1}"
        )
    try:
        progress = np.asarray([point["progress"] for point in points], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Annotation {annotation_path} has invalid progress values") from exc
    if not np.isfinite(progress).all():
        raise ValueError(f"Annotation {annotation_path} has non-finite progress values")
    return progress, annotation_path


def preprocess_lerobot_v21(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    seed: int = 0,
    demos_per_task: int | None = 50,
    train_demos_per_task: int | None = 40,
    test_demos_per_task: int = 10,
    num_bins: int = 201,
    value_min: float = -1.0,
    value_max: float = 0.0,
    strict_demo_count: bool = True,
    paper_reference_path: str = "",
    data_split_preset: str,
    ground_truth_version: str,
    annotation_root: str | Path | None = None,
) -> dict:
    """Precompute VFE labels and task metadata from LeRobot v2.1 metadata.

    Video and parquet payloads remain in the source dataset. The processed
    directory contains only versioned labels and references to those payloads.
    """

    ground_truth_version = validate_ground_truth_version(ground_truth_version)
    dataset_root = Path(dataset_root).expanduser().resolve()
    resolved_annotation_root = (
        Path(annotation_root).expanduser().resolve() if annotation_root is not None else None
    )
    if ground_truth_version == ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION:
        if resolved_annotation_root is None:
            raise ValueError(
                "annotation_root is required for annotation-based ground truth"
            )
        if not resolved_annotation_root.is_dir():
            raise FileNotFoundError(resolved_annotation_root)
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            f"Expected LeRobot v2.1 at {dataset_root}, got "
            f"{info.get('codebase_version')!r}"
        )
    use_all_demos = demos_per_task is None
    if use_all_demos:
        if train_demos_per_task is not None or test_demos_per_task != 0:
            raise ValueError(
                "LeRobot preprocessing with demos_per_task=null requires "
                "train_demos_per_task=null and test_demos_per_task=0"
            )
    else:
        if demos_per_task <= 0:
            raise ValueError("demos_per_task must be positive or null")
        if train_demos_per_task is None or train_demos_per_task <= 0:
            raise ValueError("train_demos_per_task must be positive")
        if test_demos_per_task < 0:
            raise ValueError("test_demos_per_task must be non-negative")
        if train_demos_per_task + test_demos_per_task != demos_per_task:
            raise ValueError("train_demos_per_task + test_demos_per_task must equal demos_per_task")

    base_output_dir = Path(output_dir).expanduser().resolve()
    processed_dir = versioned_processed_dir(base_output_dir, ground_truth_version).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = processed_dir / "tasks"
    labels_dir = processed_dir / "labels"

    task_rows = _read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    episode_rows = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    task_id_by_name = {str(row["task"]): int(row["task_index"]) for row in task_rows}
    episodes_by_task: dict[int, list[dict]] = {
        int(row["task_index"]): [] for row in task_rows
    }
    for episode in episode_rows:
        names = episode.get("tasks")
        if not isinstance(names, list) or len(names) != 1:
            raise ValueError(
                f"Episode {episode.get('episode_index')} must have exactly one task"
            )
        task_name = str(names[0])
        if task_name not in task_id_by_name:
            raise KeyError(f"Episode references unknown task {task_name!r}")
        episodes_by_task[task_id_by_name[task_name]].append(episode)

    chunks_size = int(info.get("chunks_size", 100))
    video_keys = [
        key for key, feature in info["features"].items() if feature.get("dtype") == "video"
    ]
    proprio_keys = [
        key
        for key, feature in info["features"].items()
        if key.startswith("observation.") and feature.get("dtype") != "video"
    ]
    manifest_rows: list[dict] = []
    task_summaries: list[dict] = []
    demos_processed = 0
    length_min: int | None = None
    length_max: int | None = None
    value_seen_min = float("inf")
    value_seen_max = float("-inf")
    bin_seen_min = int(10**9)
    bin_seen_max = int(-10**9)
    annotation_negative_count = 0
    annotation_above_100_count = 0

    for task_row in sorted(task_rows, key=lambda row: int(row["task_index"])):
        task_id = int(task_row["task_index"])
        task_name = str(task_row["task"])
        task_episodes = sorted(
            episodes_by_task[task_id], key=lambda row: int(row["episode_index"])
        )
        if not use_all_demos:
            assert demos_per_task is not None
            if strict_demo_count and len(task_episodes) != demos_per_task:
                raise ValueError(
                    f"Task {task_id} has {len(task_episodes)} episodes; expected {demos_per_task}"
                )
            if len(task_episodes) < demos_per_task:
                raise ValueError(
                    f"Task {task_id} has {len(task_episodes)} episodes; need {demos_per_task}"
                )
            task_episodes = task_episodes[:demos_per_task]
        task_train_demos = (
            len(task_episodes) if train_demos_per_task is None else train_demos_per_task
        )
        task_max_length = max(int(episode["length"]) for episode in task_episodes)
        demo_entries: list[dict] = []
        train_demo_ids: list[str] = []
        test_demo_ids: list[str] = []

        for local_index, episode in enumerate(task_episodes):
            episode_index = int(episode["episode_index"])
            demo_id = f"episode_{episode_index:06d}"
            demo_split = "train" if local_index < task_train_demos else "test"
            (train_demo_ids if demo_split == "train" else test_demo_ids).append(demo_id)
            demo_length = int(episode["length"])
            annotation_progress = None
            annotation_path = None
            if ground_truth_version == ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION:
                assert resolved_annotation_root is not None
                episode_uid = str(episode.get("episode_uid", ""))
                if not episode_uid:
                    raise KeyError(
                        f"Episode {episode_index} is missing episode_uid required for annotations"
                    )
                annotation_progress, annotation_path = _load_episode_annotation_progress(
                    annotation_root=resolved_annotation_root,
                    episode_uid=episode_uid,
                    demo_length=demo_length,
                )
                annotation_negative_count += int(np.count_nonzero(annotation_progress < 0.0))
                annotation_above_100_count += int(np.count_nonzero(annotation_progress > 100.0))
            values = ground_truth_values(
                ground_truth_version=ground_truth_version,
                demo_length=demo_length,
                task_max_length=task_max_length,
                value_min=value_min,
                value_max=value_max,
                annotation_progress=annotation_progress,
            )
            bins = discretize_values(
                values, num_bins=num_bins, value_min=value_min, value_max=value_max
            )
            if values.min() < value_min or values.max() > value_max:
                raise AssertionError(
                    f"Continuous labels out of range for task {task_id} {demo_id}"
                )
            if bins.min() < 0 or bins.max() >= num_bins:
                raise AssertionError(
                    f"Bin labels out of range for task {task_id} {demo_id}"
                )
            chunk = episode_index // chunks_size
            source_parquet = (
                dataset_root
                / "data"
                / f"chunk-{chunk:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            source_videos = {
                key: str(
                    dataset_root
                    / "videos"
                    / f"chunk-{chunk:03d}"
                    / key
                    / f"episode_{episode_index:06d}.mp4"
                )
                for key in video_keys
            }
            if not source_parquet.is_file():
                raise FileNotFoundError(source_parquet)
            missing_videos = [path for path in source_videos.values() if not Path(path).is_file()]
            if missing_videos:
                raise FileNotFoundError(missing_videos[0])

            label_path = labels_dir / f"task_{task_id:03d}" / f"{demo_id}.npz"
            save_demo_labels(
                label_path,
                value_continuous=values,
                value_bin=bins,
                demo_length=demo_length,
                demo_id=demo_id,
                task_id=task_id,
                task_name=task_name,
                demo_role="value",
                task_split="all",
                demo_split=demo_split,
                source_hdf5=str(source_parquet),
                num_bins=num_bins,
                value_min=value_min,
                value_max=value_max,
                data_split_preset=data_split_preset,
            )
            entry = {
                "demo_id": demo_id,
                "demo_role": "value",
                "demo_split": demo_split,
                "demo_length": demo_length,
                "label_path": str(label_path.relative_to(processed_dir)),
                "source_format": "lerobot_v21",
                "source_episode_index": episode_index,
                "source_parquet": str(source_parquet),
                "source_videos": source_videos,
                "success": str(episode.get("success", "unknown")),
            }
            if annotation_path is not None:
                entry["source_annotation"] = str(annotation_path)
            demo_entries.append(entry)
            manifest_rows.append(
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "suite": "icl_dataset",
                    "task_split": "all",
                    **entry,
                }
            )
            demos_processed += 1
            length_min = demo_length if length_min is None else min(length_min, demo_length)
            length_max = demo_length if length_max is None else max(length_max, demo_length)
            value_seen_min = min(value_seen_min, float(values.min()))
            value_seen_max = max(value_seen_max, float(values.max()))
            bin_seen_min = min(bin_seen_min, int(bins.min()))
            bin_seen_max = max(bin_seen_max, int(bins.max()))

        task_payload = {
            "task_id": task_id,
            "task_name": task_name,
            "suite": "icl_dataset",
            "language_instruction": task_name,
            "task_split": "all",
            "demo_ids": [entry["demo_id"] for entry in demo_entries],
            "train_demo_ids": train_demo_ids,
            "test_demo_ids": test_demo_ids,
            "task_max_demo_length": task_max_length,
            "value_range": [value_min, value_max],
            "data_split_preset": data_split_preset,
            "ground_truth_version": ground_truth_version,
            "num_bins": num_bins,
            "dataset_path": str(dataset_root),
            "source_format": "lerobot_v21",
            "fps": int(info["fps"]),
            "obs_keys": video_keys,
            "proprio_keys": proprio_keys,
            "demos": demo_entries,
        }
        task_path = tasks_dir / f"task_{task_id:03d}.json"
        write_json(task_path, task_payload)
        task_summaries.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "suite": "icl_dataset",
                "task_split": "all",
                "train_demo_ids": train_demo_ids,
                "test_demo_ids": test_demo_ids,
                "language_instruction": task_name,
                "metadata_path": str(task_path.relative_to(processed_dir)),
                "dataset_path": str(dataset_root),
            }
        )

    manifest_path = processed_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    summary = {
        "dataset": "LeRobot v2.1 ICL dataset",
        "source_format": "lerobot_v21",
        "num_total_tasks": len(task_rows),
        "num_train_tasks": len(task_rows),
        "num_test_tasks": len(task_rows) if test_demos_per_task > 0 else 0,
        "train_task_ids": [int(row["task_index"]) for row in task_rows],
        "test_task_ids": (
            [int(row["task_index"]) for row in task_rows]
            if test_demos_per_task > 0
            else []
        ),
        "num_demos_processed": demos_processed,
        "demos_per_task": demos_per_task,
        "train_demos_per_task": train_demos_per_task,
        "test_demos_per_task": test_demos_per_task,
        "value_range": [value_min, value_max],
        "base_output_dir": str(base_output_dir),
        "num_bins": num_bins,
        "task_split_mode": (
            "all_tasks_no_test" if use_all_demos else "manifest_task_70_30"
        ),
        "task_split_seed": seed,
        "demo_split_mode": (
            "all_source_episodes_train_no_test"
            if use_all_demos
            else f"first_{train_demos_per_task}_train_last_{test_demos_per_task}_test"
        ),
        "paper_reference_path": paper_reference_path,
        "data_split_preset": data_split_preset,
        "ground_truth_version": ground_truth_version,
        "dataset_root": str(dataset_root),
        "annotation_root": (
            str(resolved_annotation_root) if resolved_annotation_root is not None else None
        ),
        "annotation_negative_values_clamped": annotation_negative_count,
        "annotation_above_100_values_clamped": annotation_above_100_count,
        "output_dir": str(processed_dir),
        "tasks": task_summaries,
        "length_min": length_min,
        "length_max": length_max,
        "value_min_seen": value_seen_min,
        "value_max_seen": value_seen_max,
        "bin_min_seen": bin_seen_min,
        "bin_max_seen": bin_seen_max,
        "manifest_path": str(manifest_path),
    }
    write_json(processed_dir / "summary.json", summary)
    print("LeRobot VFE preprocessing complete")
    print(f"tasks processed: {len(task_rows)}")
    print(f"episodes processed: {demos_processed}")
    if resolved_annotation_root is not None:
        print(f"negative annotation values clamped: {annotation_negative_count}")
        print(f"annotation values above 100 clamped: {annotation_above_100_count}")
    print(f"summary: {processed_dir / 'summary.json'}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess a configured dataset for VFE.")
    parser.add_argument("--data-split-config", default=str(DEFAULT_PRESET_PATH))
    parser.add_argument("--data-split-preset", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_preset_preprocess_config(args.data_split_preset, args.data_split_config)
    dataset_format = config.pop("dataset_format", "lerobot_v21")
    if dataset_format != "lerobot_v21":
        raise ValueError(f"Unsupported dataset format: {dataset_format!r}")
    preprocess_lerobot_v21(**config, data_split_preset=args.data_split_preset)



if __name__ == "__main__":
    main()
