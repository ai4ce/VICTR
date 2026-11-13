from __future__ import annotations

import tempfile
from pathlib import Path

from vfe.data import derive_context_max_timesteps
from vfe.data_splits import (
    DEFAULT_PRESET_PATH,
    GROUND_TRUTH_VERSION_FIELD,
    load_data_split_preset,
    load_data_split_presets,
    load_preset_preprocess_config,
    load_preset_ground_truth_version,
    resolve_processed_dir,
    resolve_data_split,
    versioned_processed_dir,
)
from vfe.icl_eval import build_arg_parser as build_evaluate_arg_parser
from vfe.labels import (
    ANNOTATION_PROGRESS_GROUND_TRUTH_VERSION,
    LEGACY_GROUND_TRUTH_VERSION,
    PER_DEMO_GROUND_TRUTH_VERSION,
)
from vfe.native_features import build_arg_parser as build_feature_arg_parser
from vfe.preprocess import build_arg_parser as build_preprocess_arg_parser
from vfe.train import apply_vfe_preset, build_arg_parser as build_train_arg_parser
from vfe.vfe_presets import (
    DEFAULT_VFE_PRESET_PATH,
    load_vfe_preset,
    load_vfe_presets,
    resolve_vfe_backbone,
)


REMOVED_DATA_OPTIONS = {
    "--processed-dir",
    "--obs-key",
    "--proprio-key",
    "--image-size",
    "--context-obs-key",
    "--context-image-size",
    "--context-stride",
    "--context-max-original-timesteps",
    "--context-native-feature-dir",
    "--native-vision-model-name",
    "--native-vision-model-revision",
    "--backbone",
    "--no-context",
    "--vision-model-name",
    "--context-vision-model-name",
    "--language-model-name",
    "--torch-dtype",
    "--freeze-backbone",
    "--freeze-native-vision",
    "--freeze-vision",
    "--lora",
    "--gradient-checkpointing",
    "--adapter-output-norm",
    "--context-patch-tokens-per-image",
    "--context-image-chunk-size",
}


def _options(parser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }


def test_every_data_split_preset_has_complete_preprocess_config() -> None:
    presets = load_data_split_presets(DEFAULT_PRESET_PATH)
    for preset_name in presets:
        load_preset_preprocess_config(preset_name, DEFAULT_PRESET_PATH)


def test_preprocessing_clis_expose_only_preset_selection() -> None:
    assert _options(build_preprocess_arg_parser()) == {
        "--data-split-config",
        "--data-split-preset",
    }
    assert _options(build_feature_arg_parser()) == {
        "--data-split-config",
        "--data-split-preset",
        "--vfe-config",
        "--vfe-preset",
    }


def test_preprocess_separates_value_target_versions() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        base = Path(temp_dir) / "processed"
        legacy_dir = versioned_processed_dir(base, LEGACY_GROUND_TRUTH_VERSION)
        new_dir = versioned_processed_dir(base, PER_DEMO_GROUND_TRUTH_VERSION)
        assert legacy_dir != new_dir
        assert legacy_dir.parent == new_dir.parent == base / "value_targets"

        new_dir.mkdir(parents=True)
        (new_dir / "summary.json").write_text("{}", encoding="utf-8")
        assert resolve_processed_dir(base, PER_DEMO_GROUND_TRUTH_VERSION) == new_dir
        try:
            resolve_processed_dir(base, LEGACY_GROUND_TRUTH_VERSION)
        except FileNotFoundError as exc:
            assert str(legacy_dir) in str(exc)
        else:
            raise AssertionError("A missing requested ground-truth version was silently substituted")


def test_training_and_evaluation_do_not_expose_preset_data_overrides() -> None:
    assert not (_options(build_train_arg_parser()) & REMOVED_DATA_OPTIONS)
    assert not (_options(build_evaluate_arg_parser()) & REMOVED_DATA_OPTIONS)


def test_icl_keep_true_manifest_is_all_tasks_with_80_20_episode_splits_and_no_test() -> None:
    preset_name = "icl_keep_true_all36_query80_ctx20_per_demo"
    preset = load_data_split_preset(preset_name)
    assert set(preset) - {
        "description",
        GROUND_TRUTH_VERSION_FIELD,
        "manifest",
        "preprocess",
    } == {"train"}

    train = resolve_data_split(preset_name=preset_name, phase="train")
    assert train["task_id_set"] == set(range(36))
    assert train["demo_split"] == "all"
    assert train["context_selection"] == "train_seeded"

    total_query = 0
    total_context = 0
    for task_id in train["task_ids"]:
        query = set(train["query_demo_ids_by_task"][task_id])
        context = set(train["context_demo_ids_by_task"][task_id])
        total = len(query) + len(context)
        assert query.isdisjoint(context)
        assert total in {48, 50, 100}
        assert len(query) == int(total * 0.8)
        assert len(context) == total - len(query)
        total_query += len(query)
        total_context += len(context)

    assert total_query == 1704
    assert total_context == 430
    assert total_query + total_context == 2134


def test_icl_annotation_preset_reuses_exact_keep_true_episode_split() -> None:
    linear = resolve_data_split(
        preset_name="icl_keep_true_all36_query80_ctx20_per_demo", phase="train"
    )
    annotated = resolve_data_split(
        preset_name="icl_keep_true_all36_query80_ctx20_annotation_progress", phase="train"
    )
    assert annotated["task_ids"] == linear["task_ids"]
    assert annotated["query_demo_ids_by_task"] == linear["query_demo_ids_by_task"]
    assert annotated["context_demo_ids_by_task"] == linear["context_demo_ids_by_task"]
