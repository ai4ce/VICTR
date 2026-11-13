from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .labels import (
    ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION,
    validate_ground_truth_version,
)


DEFAULT_PRESET_PATH = Path("configs/vfe/vfe_data_splits.json")
DEFAULT_CONTEXT_MANIFEST_PATH = Path("configs/vfe/icl_dataset_keep_true_all36_query80_ctx20.json")
GROUND_TRUTH_VERSION_FIELD = "value_" + "target_semantics"
PRESET_METADATA_KEYS = {
    "description",
    "manifest",
    "manifest_path",
    "task_manifest",
    "preprocess",
    GROUND_TRUTH_VERSION_FIELD,
}
COMMON_PREPROCESS_CONFIG_KEYS = {
    "dataset_root",
    "output_dir",
    "seed",
    "demos_per_task",
    "train_demos_per_task",
    "test_demos_per_task",
    "num_bins",
    "value_min",
    "value_max",
    "strict_demo_count",
    "paper_reference_path",
}
PREPROCESS_FORMAT_KEYS = {
    "lerobot_v21": set(),
}


def parse_id_spec(spec: str | list[int] | tuple[int, ...]) -> set[int]:
    if isinstance(spec, (list, tuple)):
        return {int(value) for value in spec}
    task_ids: set[int] = set()
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending id range: {chunk}")
            task_ids.update(range(start, end + 1))
        else:
            task_ids.add(int(chunk))
    return task_ids


def load_data_split_presets(path: str | Path = DEFAULT_PRESET_PATH) -> dict[str, dict[str, Any]]:
    preset_path = Path(path).expanduser()
    if not preset_path.exists():
        raise FileNotFoundError(f"Data split preset file not found: {preset_path}")
    return json.loads(preset_path.read_text(encoding="utf-8"))


def load_data_split_preset(
    preset_name: str,
    path: str | Path = DEFAULT_PRESET_PATH,
) -> dict[str, Any]:
    presets = load_data_split_presets(path)
    if preset_name not in presets:
        available = ", ".join(sorted(presets))
        raise KeyError(f"Unknown data split preset {preset_name!r}. Available: {available}")
    preset = presets[preset_name]
    if not isinstance(preset, dict):
        raise TypeError(f"Data split preset {preset_name!r} must be an object")
    return dict(preset)


def load_preset_preprocess_config(
    preset_name: str,
    path: str | Path = DEFAULT_PRESET_PATH,
) -> dict[str, Any]:
    preset = load_data_split_preset(preset_name, path)
    config = preset.get("preprocess")
    if not isinstance(config, dict):
        raise KeyError(f"Data split preset {preset_name!r} must define a preprocess object")
    dataset_format = str(config.get("dataset_format", "lerobot_v21"))
    if dataset_format not in PREPROCESS_FORMAT_KEYS:
        supported = ", ".join(sorted(PREPROCESS_FORMAT_KEYS))
        raise ValueError(
            f"Data split preset {preset_name!r} has unsupported preprocess "
            f"dataset_format={dataset_format!r}; expected one of: {supported}"
        )
    required = COMMON_PREPROCESS_CONFIG_KEYS | PREPROCESS_FORMAT_KEYS[dataset_format]
    missing = sorted(required.difference(config))
    if missing:
        raise KeyError(
            f"Data split preset {preset_name!r} preprocess object is missing: {', '.join(missing)}"
        )
    ground_truth_version = preset.get(GROUND_TRUTH_VERSION_FIELD)
    if ground_truth_version is None:
        raise KeyError(
            f"Data split preset {preset_name!r} must define its ground-truth version"
        )
    validated_ground_truth_version = validate_ground_truth_version(ground_truth_version)
    if validated_ground_truth_version == ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION:
        if dataset_format != "lerobot_v21":
            raise ValueError(
                "Annotation-based ground truth is currently supported only for LeRobot v2.1"
            )
        if not config.get("annotation_root"):
            raise KeyError(
                f"Data split preset {preset_name!r} must define preprocess.annotation_root "
                "for annotation-based ground truth"
            )
    result = dict(config)
    result["dataset_format"] = dataset_format
    result["ground_truth_version"] = validated_ground_truth_version
    return result


def load_preset_ground_truth_version(
    preset_name: str,
    path: str | Path = DEFAULT_PRESET_PATH,
) -> str:
    """Return the ground-truth definition owned by one data split preset."""

    preset = load_data_split_preset(preset_name, path)
    ground_truth_version = preset.get(GROUND_TRUTH_VERSION_FIELD)
    if ground_truth_version is None:
        raise KeyError(
            f"Data split preset {preset_name!r} must define its ground-truth version"
        )
    return validate_ground_truth_version(ground_truth_version)


def versioned_processed_dir(output_dir: str | Path, ground_truth_version: str) -> Path:
    """Return the non-overlapping processed-data directory for one label definition."""

    version = validate_ground_truth_version(ground_truth_version)
    return Path(output_dir).expanduser() / "value_targets" / version


def resolve_processed_dir(output_dir: str | Path, ground_truth_version: str) -> Path:
    """Require the fully preprocessed directory for the requested label definition."""

    base_dir = Path(output_dir).expanduser()
    requested_dir = versioned_processed_dir(base_dir, ground_truth_version)
    if not (requested_dir / "summary.json").is_file():
        raise FileNotFoundError(
            f"Missing preprocessed ground truth for {ground_truth_version!r}: {requested_dir}. "
            "Run scripts/vfe/preprocess_vfe.py with the matching data-split preset first."
        )
    return requested_dir


def _available_phases(preset: dict[str, Any]) -> str:
    return ", ".join(
        sorted(
            key
            for key, value in preset.items()
            if key not in PRESET_METADATA_KEYS and isinstance(value, dict)
        )
    )


def _resolve_relative_path(path: str | Path, base_path: Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return (base_path.parent / resolved).resolve()


def load_context_split_manifest(path: str | Path = DEFAULT_CONTEXT_MANIFEST_PATH) -> dict[str, Any]:
    manifest_path = Path(path).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Context split manifest not found: {manifest_path}. "
            "Restore the checked-in episode manifest or configure an explicit manifest path."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def _task_ids_from_records(records: list[dict[str, Any]]) -> set[int]:
    return {int(record["task_id"]) for record in records}


def _demo_map(records: list[dict[str, Any]], field: str) -> dict[int, list[str]]:
    return {int(record["task_id"]): [str(demo_id) for demo_id in record[field]] for record in records}


def _resolve_context_manifest_phase(
    *,
    preset_name: str,
    phase: str,
    preset: dict[str, Any],
    preset_path: Path,
) -> dict[str, Any]:
    manifest_key = preset.get("manifest", preset.get("manifest_path", DEFAULT_CONTEXT_MANIFEST_PATH))
    manifest = load_context_split_manifest(_resolve_relative_path(manifest_key, preset_path))
    phase_config = dict(preset.get(phase, {}))
    manifest_phase = phase_config.get("manifest_phase", phase)
    if manifest_phase in {"train", "train_context"}:
        task_records = manifest["train_tasks"]
        query_field = "train_query_demo_ids"
        context_field = "train_context_demo_ids"
        context_selection = "train_seeded"
    elif manifest_phase in {"eval_unseen", "unseen_eval"}:
        task_records = manifest["unseen_eval_tasks"]
        query_field = "eval_query_demo_ids"
        context_field = "eval_context_demo_ids"
        context_selection = "eval_fixed"
    else:
        raise KeyError(
            f"Unsupported context manifest phase {manifest_phase!r} for preset {preset_name!r}. "
            "Use 'train' or 'eval_unseen'."
        )

    return {
        "preset_name": preset_name,
        "phase": phase,
        "description": preset.get("description", ""),
        "demo_split": "all",
        "task_ids": sorted(_task_ids_from_records(task_records)),
        "task_id_set": _task_ids_from_records(task_records),
        "context_split_manifest": manifest.get("_manifest_path", str(manifest_key)),
        "context_split_name": manifest.get("split_name", preset_name),
        "context_split_seed": int(manifest.get("seed", 0)),
        "query_demo_ids_by_task": _demo_map(task_records, query_field),
        "context_demo_ids_by_task": _demo_map(task_records, context_field),
        "context_selection": context_selection,
    }


def _resolve_task_manifest_phase(
    *,
    preset_name: str,
    phase: str,
    preset: dict[str, Any],
    preset_path: Path,
) -> dict[str, Any]:
    """Resolve task IDs from a manifest without inheriting its demo lists.

    This supports controls that share the exact seed-selected 70/30 task split
    of a context experiment while retaining the dataset's ordinary demo_split
    metadata, e.g. 45 train / 5 test demos for selected training tasks.
    """
    manifest_key = preset["task_manifest"]
    manifest = load_context_split_manifest(_resolve_relative_path(manifest_key, preset_path))
    phase_config = dict(preset.get(phase, {}))
    manifest_phase = phase_config.get("manifest_phase", phase)
    if manifest_phase in {"train", "train_context"}:
        task_records = manifest["train_tasks"]
    elif manifest_phase in {"eval_unseen", "unseen_eval"}:
        task_records = manifest["unseen_eval_tasks"]
    else:
        raise KeyError(
            f"Unsupported task manifest phase {manifest_phase!r} for preset {preset_name!r}. "
            "Use 'train' or 'eval_unseen'."
        )
    if "demo_split" not in phase_config:
        raise KeyError(
            f"Task-manifest preset {preset_name!r} phase {phase!r} must define demo_split."
        )

    return {
        "preset_name": preset_name,
        "phase": phase,
        "description": preset.get("description", ""),
        "demo_split": phase_config["demo_split"],
        "task_ids": sorted(_task_ids_from_records(task_records)),
        "task_id_set": _task_ids_from_records(task_records),
        "task_split_manifest": manifest.get("_manifest_path", str(manifest_key)),
        "task_split_name": manifest.get("split_name", preset_name),
        "task_split_seed": int(manifest.get("seed", 0)),
    }


def resolve_data_split(
    *,
    preset_name: str,
    phase: str,
    preset_path: str | Path = DEFAULT_PRESET_PATH,
) -> dict[str, Any]:
    preset = load_data_split_preset(preset_name, preset_path)
    preset_path_obj = Path(preset_path).expanduser()
    if "task_manifest" in preset:
        if phase not in preset:
            available = _available_phases(preset)
            raise KeyError(f"Preset {preset_name!r} has no phase {phase!r}. Available: {available}")
        return _resolve_task_manifest_phase(
            preset_name=preset_name,
            phase=phase,
            preset=preset,
            preset_path=preset_path_obj,
        )
    if "manifest" in preset or "manifest_path" in preset:
        if phase not in preset:
            available = _available_phases(preset)
            raise KeyError(f"Preset {preset_name!r} has no phase {phase!r}. Available phases: {available}")
        return _resolve_context_manifest_phase(
            preset_name=preset_name,
            phase=phase,
            preset=preset,
            preset_path=preset_path_obj,
        )
    if phase not in preset:
        available = _available_phases(preset)
        raise KeyError(f"Preset {preset_name!r} has no phase {phase!r}. Available phases: {available}")
    config = dict(preset[phase])
    config["preset_name"] = preset_name
    config["phase"] = phase
    config["description"] = preset.get("description", "")
    if "task_ids" not in config:
        raise KeyError(f"Preset {preset_name!r} phase {phase!r} must define task_ids")
    if "demo_split" not in config:
        raise KeyError(f"Preset {preset_name!r} phase {phase!r} must define demo_split")
    config["task_id_set"] = parse_id_spec(config["task_ids"])
    return config
