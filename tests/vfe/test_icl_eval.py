from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from vfe.icl_eval import (
    IclContinuousEvalDataset,
    build_all_context_eval_work_items,
    choose_context_demo_id,
    load_existing_trajectory_record,
    load_continuous_eval_episode_entries,
    manual_annotation_progress,
    resolve_record_context_demo_id,
    shard_all_context_eval_work_items,
    task_output_dir_name,
)
from vfe.labels import annotation_progress_values


def test_manual_key_events_expand_to_step_ground_truth_and_clamp_terminal_boundary() -> None:
    annotation = {
        "key_events": [
            {"frame": 0, "value": 0.0},
            {"frame": 2, "value": 0.5},
            {"frame": 5, "value": 1.0},
        ]
    }
    progress = manual_annotation_progress(annotation, demo_length=5)
    np.testing.assert_allclose(progress, [0.0, 0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(
        annotation_progress_values(progress * 100.0),
        [-1.0, -1.0, -0.5, -0.5, 0.0],
    )


def test_task_output_dir_name_includes_id_and_safe_task_name() -> None:
    assert (
        task_output_dir_name(0, "hit the eggplant with the mallet")
        == "task_000+hit_the_eggplant_with_the_mallet"
    )
    assert task_output_dir_name(12, " put/circle: peg ") == "task_012+put_circle_peg"


def test_no_context_eval_retains_enumerated_context_identity() -> None:
    assert (
        resolve_record_context_demo_id(
            [],
            forced_context_demo_id="eval_episode_000001",
            model_context_enabled=False,
            query_demo_id="eval_episode_000000",
        )
        == "eval_episode_000001"
    )
    assert (
        resolve_record_context_demo_id(
            ["eval_episode_000001"],
            forced_context_demo_id="eval_episode_000001",
            model_context_enabled=True,
            query_demo_id="eval_episode_000000",
        )
        == "eval_episode_000001"
    )


def test_context_group_sharding_balances_frame_cost_and_keeps_groups_intact() -> None:
    class DatasetStub:
        lengths = {"q0": 100, "q1": 100, "q2": 190, "q3": 10, "q4": 10}

        def demo_entry(self, _task_id: int, demo_id: str) -> dict[str, int]:
            return {"demo_length": self.lengths[demo_id]}

    work_items = [
        (0, 0, 0, "c0", "q0"),
        (1, 0, 0, "c0", "q1"),
        (2, 1, 0, "c1", "q2"),
        (3, 2, 0, "c2", "q3"),
        (4, 3, 0, "c3", "q4"),
    ]
    shards = []
    for rank in range(2):
        shard, costs = shard_all_context_eval_work_items(
            work_items, DatasetStub(), rank=rank, world_size=2
        )
        shards.append(shard)
        assert costs == [210, 200]
    assert sorted(item[0] for shard in shards for item in shard) == list(range(5))
    group_ranks = {
        group_index: {
            rank
            for rank, shard in enumerate(shards)
            for item in shard
            if item[1] == group_index
        }
        for group_index in range(4)
    }
    assert all(len(ranks) == 1 for ranks in group_ranks.values())


def test_existing_trajectory_resume_strictly_validates_identity() -> None:
    for context_id, context_uid in [("context", "context_uid"), ("", None)]:
        _check_existing_trajectory_identity(context_id, context_uid)


def _check_existing_trajectory_identity(context_id, context_uid) -> None:
    metadata = {
        "trajectory_index": 3,
        "task_id": 2,
        "task_name": "task",
        "task_novelty": "unseen",
        "demo_id": "query",
        "episode_index": 8,
        "episode_uid": "query_uid",
        "dataset_success": "success",
        "effective_success": "success",
        "outcome_override": None,
        "training_overlap": "separate_dataset",
        "context_demo_id": context_id,
        "context_episode_index": 7,
        "context_episode_uid": context_uid,
        "num_source_frames": 5,
        "num_evaluated_frames": 5,
        "metrics": {"mae": 0.1},
        "checkpoint": "/tmp/checkpoint.pt",
        "ground_truth_version": "annotation_progress_clamped_v1",
        "vfe_preset": "preset",
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "curve.npz"
        np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata)))
        record = load_existing_trajectory_record(
            path,
            trajectory_index=3,
            task_id=2,
            demo_id="query",
            episode_uid="query_uid",
            context_demo_id=context_id,
            context_episode_uid=context_uid,
            checkpoint="/tmp/checkpoint.pt",
            ground_truth_version="annotation_progress_clamped_v1",
            vfe_preset="preset",
        )
        assert record["metrics"] == {"mae": 0.1}
        try:
            load_existing_trajectory_record(
                path,
                trajectory_index=4,
                task_id=2,
                demo_id="query",
                episode_uid="query_uid",
                context_demo_id=context_id,
                context_episode_uid=context_uid,
                checkpoint="/tmp/checkpoint.pt",
                ground_truth_version="annotation_progress_clamped_v1",
                vfe_preset="preset",
            )
        except ValueError as exc:
            assert "trajectory_index" in str(exc)
        else:
            raise AssertionError("Mismatched resume artifact was accepted")


def test_manual_key_events_require_frame_zero_and_strict_order() -> None:
    for events, message in (
        ([{"frame": 1, "value": 0.0}], "frame 0"),
        (
            [
                {"frame": 0, "value": 0.0},
                {"frame": 2, "value": 0.5},
                {"frame": 2, "value": 1.0},
            ],
            "strictly increasing",
        ),
    ):
        try:
            manual_annotation_progress({"key_events": events}, demo_length=4)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("Invalid manual key events were accepted")


def test_context_selection_is_fixed_and_skips_exact_query_episode() -> None:
    pool = ["context_0", "context_1", "context_2"]
    uid_by_demo = {
        "context_0": "uid_0",
        "context_1": "query_uid",
        "context_2": "uid_2",
    }
    selected = choose_context_demo_id(
        pool,
        local_eval_index=1,
        task_id=3,
        seed=0,
        strategy="round_robin",
        query_episode_uid="query_uid",
        context_uid_by_demo_id=uid_by_demo,
    )
    assert selected == "context_2"


def test_continuous_demo_eval_covers_seen_and_unseen_and_excludes_self_context() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "dataset"
        annotation_root = Path(temp_dir) / "annotations"
        (root / "meta").mkdir(parents=True)
        annotation_root.mkdir()
        info = {
            "codebase_version": "v2.1",
            "fps": 30,
            "chunks_size": 100,
            "features": {
                "observation.images.zed": {"dtype": "video", "shape": [2, 2, 3]},
                "observation.state": {"dtype": "float32", "shape": [15]},
            },
        }
        (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
        tasks = [
            {"task_index": 0, "task": "seen task"},
            {"task_index": 1, "task": "novel task"},
        ]
        (root / "meta/tasks.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8"
        )
        episodes = []
        for episode_index, task_name in enumerate(
            ["seen task", "seen task", "novel task", "novel task"]
        ):
            uid = f"episode_{episode_index}:000000"
            episodes.append(
                {
                    "episode_index": episode_index,
                    "episode_uid": uid,
                    "tasks": [task_name],
                    "length": 3,
                    "success": True,
                    "valid": True,
                    "keep": True,
                }
            )
            chunk = root / "data/chunk-000"
            video = root / "videos/chunk-000/observation.images.zed"
            chunk.mkdir(parents=True, exist_ok=True)
            video.mkdir(parents=True, exist_ok=True)
            (chunk / f"episode_{episode_index:06d}.parquet").write_bytes(b"test")
            (video / f"episode_{episode_index:06d}.mp4").write_bytes(b"test")
            annotation = {
                "episode_uid": uid,
                "episode_index": episode_index,
                "task": task_name,
                "success": "success",
                "fps": 30,
                "num_frames": 3,
                "points": [
                    {
                        "frame_index": frame_index,
                        "timestamp": round(frame_index / 30, 4),
                        "progress": frame_index * 50.0,
                    }
                    for frame_index in range(3)
                ],
            }
            (annotation_root / f"{uid.replace(':', '__')}.json").write_text(
                json.dumps(annotation), encoding="utf-8"
            )
        episodes.append(
            {
                "episode_index": 4,
                "episode_uid": "failed:000000",
                "tasks": ["seen task"],
                "length": 3,
                "success": False,
                "valid": True,
                "keep": False,
            }
        )
        (annotation_root / "failed__000000.json").write_text(
            json.dumps({"episode_uid": "failed:000000"}), encoding="utf-8"
        )
        (annotation_root / "other_dataset__000000.json").write_text(
            json.dumps({"episode_uid": "other_dataset:000000"}), encoding="utf-8"
        )
        (root / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
        )

        entries, labels, summary = load_continuous_eval_episode_entries(
            dataset_root=root,
            annotation_root=annotation_root,
            training_task_names={"seen task"},
            value_min=-1.0,
            value_max=0.0,
            num_bins=201,
            trajectories_per_task=1,
        )
        assert summary["num_tasks"] == 2
        assert summary["num_trajectories"] == 2
        assert summary["task_novelty_trajectories"] == {"seen": 1, "unseen": 1}
        assert summary["annotation_files"] == 6
        assert summary["annotation_files_matching_dataset"] == 5
        assert summary["annotation_files_outside_dataset"] == 1
        assert summary["annotation_files_for_keep_false"] == 1
        assert sum(len(task_entries) for task_entries in entries.values()) == 4
        np.testing.assert_allclose(
            labels[(0, "eval_episode_000000")]["value_continuous"],
            [-1.0, -0.5, 0.0],
        )

        dataset = IclContinuousEvalDataset(
            dataset_root=root,
            entries_by_task=entries,
            eval_labels=labels,
            obs_key="observation.images.zed",
            proprio_key="observation.state",
            image_size=0,
            query_history_offsets=[60, 30, 0],
            context_obs_key="observation.images.zed",
            context_image_size=0,
            context_max_timesteps=2,
            context_stride=30,
            context_max_original_timesteps=60,
            context_selection="round_robin",
            context_seed=0,
            contexts_per_task=2,
            all_contexts=False,
            proprio_dim=15,
        )
        assert dataset.query_history_offsets == [60, 30, 0]
        assert len(dataset.demo_by_task[0]) == 1
        assert dataset.select_context_demo_id(
            0,
            "eval_episode_000000",
            frame_idx=0,
            sample_index=0,
            epoch=0,
        ) == "eval_episode_000001"

        all_context_dataset = IclContinuousEvalDataset(
            dataset_root=root,
            entries_by_task=entries,
            eval_labels=labels,
            obs_key="observation.images.zed",
            proprio_key="observation.state",
            image_size=0,
            context_obs_key="observation.images.zed",
            context_image_size=0,
            context_max_timesteps=2,
            context_stride=30,
            context_max_original_timesteps=60,
            context_selection="round_robin",
            context_seed=0,
            contexts_per_task=2,
            all_contexts=True,
            proprio_dim=15,
        )
        assert len(all_context_dataset.context_demo_ids_by_task[0]) == 2
        all_context_dataset.set_fixed_context(0, "eval_episode_000001")
        assert all_context_dataset.select_context_demo_id(
            0,
            "eval_episode_000000",
            frame_idx=0,
            sample_index=0,
            epoch=0,
        ) == "eval_episode_000001"
        work_items, num_context_groups = build_all_context_eval_work_items(
            all_context_dataset
        )
        assert num_context_groups == 4
        assert len(work_items) == 2


if __name__ == "__main__":
    test_manual_key_events_expand_to_step_ground_truth_and_clamp_terminal_boundary()
    test_task_output_dir_name_includes_id_and_safe_task_name()
    test_no_context_eval_retains_enumerated_context_identity()
    test_context_group_sharding_balances_frame_cost_and_keeps_groups_intact()
    test_existing_trajectory_resume_strictly_validates_identity()
    test_manual_key_events_require_frame_zero_and_strict_order()
    test_context_selection_is_fixed_and_skips_exact_query_episode()
    test_continuous_demo_eval_covers_seen_and_unseen_and_excludes_self_context()
