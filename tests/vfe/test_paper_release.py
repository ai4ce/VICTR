from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from vfe.data_splits import load_preset_preprocess_config
from vfe.train import build_arg_parser
from vfe.vfe_presets import load_vfe_preset, load_vfe_presets
from paper import build_command
from score_paper import correlations, mean_scores, score
from vfe.scoring import causal_ema

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs/vfe/paper_experiments.json").read_text())


def args_for(name, stage, **kwargs):
    return argparse.Namespace(experiment=name, stage=stage, resume=None, gpus=None,
                              exp_name=None, resume_model_only=False, allow_inexact_resume=False,
                              checkpoint=None, output=None, batch_size=4, source=None,
                              config=ROOT / "configs/vfe/paper_experiments.json", **kwargs)


def test_only_four_models_and_saved_source_settings_are_preserved():
    assert set(CONFIG["experiments"]) == {"ic_vfe", "ic_vfe_no_context", "ic_vfe_no_block_embedding", "recap"}
    assert set(load_vfe_presets()["presets"]) == {s["vfe_preset"] for s in CONFIG["experiments"].values()}
    for spec in CONFIG["experiments"].values():
        resolved = load_vfe_preset(spec["vfe_preset"])["model"]
        for key, expected in spec.get("source_model", {}).items():
            assert resolved[key] == expected, (spec["vfe_preset"], key)
        assert load_preset_preprocess_config(spec["data_split_preset"])["ground_truth_version"] == "annotation_progress_clamped_v1"


def test_no_block_ablation_changes_only_embedding_and_reuses_data_and_features():
    baseline = CONFIG["experiments"]["ic_vfe"]
    spec = CONFIG["experiments"]["ic_vfe_no_block_embedding"]
    main_preset = load_vfe_preset(baseline["vfe_preset"])
    ablation_preset = load_vfe_preset(spec["vfe_preset"])
    main, ablation = main_preset["model"], ablation_preset["model"]
    assert {k for k in main if main[k] != ablation[k]} == {"context_slot_embedding_enabled"}
    assert main["context_slot_embedding_enabled"] and not main["context_progress_enabled"]
    assert not ablation["context_progress_enabled"] and not ablation["context_slot_embedding_enabled"]
    assert main_preset["context_features"] == ablation_preset["context_features"]
    assert baseline["data_split_preset"] == spec["data_split_preset"]
    assert baseline["scoring"] == spec["scoring"]
    assert spec["status"] == "pending_training"
    assert not ({"checkpoint_step", "source_model", "source_checkpoint", "source_eval", "source_run", "source_run_config_sha256"} & spec.keys())
    with pytest.raises(KeyError, match="Unknown VFE preset"):
        load_vfe_preset("gemma3_270m_native_dualres_context177_stride30_3cam_freeze_vision_context_progress_rope_only")


@pytest.mark.parametrize("name", CONFIG["experiments"])
def test_training_commands_parse_and_preserve_global_batch(name):
    args = args_for(name, "train")
    if name == "ic_vfe":
        args.resume = "checkpoint_step_6800_bs320_cursor26.pt"
    command = build_command(args, CONFIG)
    parsed = build_arg_parser().parse_args(command[command.index("scripts/vfe/train_vfe.py") + 1:])
    training = CONFIG["experiments"][name]["training"]
    assert parsed.tasks_per_batch * parsed.samples_per_task * training["distributed"]["world_size"] == training["distributed"]["global_batch_size"]
    assert parsed.max_steps == training["max_steps"]


@pytest.mark.parametrize("stage,flag", [("evaluate", "checkpoint"), ("validate-data", "checkpoint"), ("score", "source")])
def test_pending_ablation_requires_new_artifacts(stage, flag):
    args = args_for("ic_vfe_no_block_embedding", stage)
    with pytest.raises(ValueError, match="pending training"):
        build_command(args, CONFIG)
    setattr(args, flag, "outputs/new_no_block_run")
    command = build_command(args, CONFIG)
    assert "outputs/new_no_block_run" in command
    assert not any("context_progress_rope_only" in part for part in command)


def test_main_training_requires_explicit_resume_and_invalid_gpu_count_fails():
    with pytest.raises(ValueError, match="continued training"):
        build_command(args_for("ic_vfe", "train"), CONFIG)
    args = args_for("recap", "train")
    args.gpus = 3
    with pytest.raises(ValueError, match="divide global batch"):
        build_command(args, CONFIG)


def test_context_free_evaluation_does_not_repeat_queries_or_cache_context():
    command = build_command(args_for("recap", "evaluate"), CONFIG)
    assert "--all-contexts" not in command and "--cache-context-tokens" not in command
    command = build_command(args_for("ic_vfe", "evaluate"), CONFIG)
    assert "--all-contexts" in command and "--cache-context-tokens" in command


def test_paper_constant_conventions_and_causal_ema():
    target = np.ones(4)
    prediction = np.arange(4)
    assert correlations(target, prediction, True)["pearson"] is None
    assert correlations(target, prediction, False)["pearson"] == 0
    np.testing.assert_allclose(causal_ema(np.array([0., 1., 0.]), .5), [0., .5, .25])
    assert mean_scores([correlations(target, prediction, True)])["kendall"] is None


@pytest.mark.parametrize("corruption", [None, "mixed_checkpoint", "duplicate", "already_smoothed", "missing_query"])
def test_saved_prediction_scoring_requires_one_complete_raw_run(tmp_path, corruption):
    config = deepcopy(CONFIG)
    config["vfe_config"] = str(ROOT / config["vfe_config"])
    config["evaluation"].update(dataset_root="dataset", human_annotations="annotations",
                                expected_queries=4, expected_queries_per_split=2)
    metadata_dir = tmp_path / "dataset/meta"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "info.json").write_text(json.dumps({"fps": 30}))
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    source = tmp_path / "predictions"
    source.mkdir()
    episodes = []
    for index in range(4):
        uid = f"task:{index}"
        task = f"task {index // 2}"
        episodes.append({"episode_uid": uid, "source_episode_index": index, "length": 3,
                         "tasks": [task], "keep": True, "success": True})
        annotation = {"episode_uid": uid, "episode_index": index, "num_frames": 3,
                      "task": task, "fps": 30, "success": "success",
                      "points": [{"frame_index": i, "timestamp": round(i / 30, 4), "progress": p}
                                 for i, p in enumerate([0., 50., 100.])]}
        (annotations / (uid.replace(":", "__") + ".json")).write_text(json.dumps(annotation))
        meta = {"episode_uid": uid, "context_episode_uid": None, "task_name": task,
                "vfe_preset": config["experiments"]["recap"]["vfe_preset"],
                "checkpoint": "checkpoint.pt", "task_novelty": "seen" if index < 2 else "unseen"}
        if index == 3 and corruption == "mixed_checkpoint":
            meta["checkpoint"] = "different.pt"
        if corruption == "already_smoothed":
            meta["ema_alpha"] = .5
        if index == 3 and corruption == "missing_query":
            continue
        np.savez(source / f"{index}.npz", metadata_json=np.asarray(json.dumps(meta)),
                 prediction=np.asarray([-1., -.5, 0.], dtype=np.float32),
                 timesteps=np.arange(3), context_demo_ids=np.asarray([], dtype=str))
    (metadata_dir / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))
    if corruption == "duplicate":
        (source / "duplicate.npz").write_bytes((source / "0.npz").read_bytes())
    if corruption:
        with pytest.raises(ValueError):
            score(config, "recap", source, root=tmp_path)
    else:
        result = score(config, "recap", source, root=tmp_path)
        assert result["aggregate"]["seen"]["metrics"]["pearson"] == pytest.approx(1)
        assert result["aggregate"]["unseen"]["num_scored_queries"] == 2


def test_main_10k_release_and_training_agree():
    spec = CONFIG["experiments"]["ic_vfe"]
    releases = json.loads((ROOT / "configs/vfe/checkpoints.json").read_text())
    published = releases["models"]["ic_vfe"]
    assert spec["checkpoint_step"] == published["step"] == 10000
    assert published["filename"] == "checkpoint_step_10000.pt"
    command = build_command(args_for("ic_vfe", "evaluate"), CONFIG)
    assert command[command.index("--checkpoint") + 1].endswith("checkpoint_step_10000.pt")
    args = args_for("ic_vfe", "train")
    args.resume = "checkpoint_step_6800_bs320_cursor26.pt"
    command = build_command(args, CONFIG)
    assert command[command.index("--max-steps") + 1] == "10000"
    command = build_command(args_for("ic_vfe", "pretrain"), CONFIG)
    assert command[command.index("--max-steps") + 1] == "30000"


@pytest.mark.parametrize("name", ["ic_vfe", "ic_vfe_no_context", "recap"])
def test_released_models_default_to_download_and_prediction_locations(name):
    spec = CONFIG["experiments"][name]
    releases = json.loads((ROOT / "configs/vfe/checkpoints.json").read_text())
    command = build_command(args_for(name, "evaluate"), CONFIG)
    assert command[command.index("--checkpoint") + 1] == f"checkpoints/{name}/{releases['models'][name]['filename']}"
    output = command[command.index("--output-dir") + 1]
    command = build_command(args_for(name, "score"), CONFIG)
    assert command[command.index("--source") + 1] == output == spec["prediction_dir"]
