from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vfe.evaluate import (
    aggregate_context_cache_summaries,
    aggregate_metric_dicts,
    distributed_barrier,
    initialize_distributed_evaluation,
    merge_sharded_trajectory_metrics,
    validate_checkpoint_ground_truth_provenance,
    save_trajectory_data,
    shard_trajectory_work_items,
)
from vfe.labels import LEGACY_GROUND_TRUTH_VERSION, PER_DEMO_GROUND_TRUTH_VERSION


def test_checkpoint_semantics_require_explicit_checkpoint_or_data_config_metadata() -> None:
    resolved = validate_checkpoint_ground_truth_provenance(
        checkpoint_legacy_ground_truth_version=None,
        checkpoint_data_split_preset="legacy_config",
        requested_data_split_preset="legacy_config",
        config_ground_truth_version=LEGACY_GROUND_TRUTH_VERSION,
    )
    assert resolved == LEGACY_GROUND_TRUTH_VERSION

    try:
        validate_checkpoint_ground_truth_provenance(
            checkpoint_legacy_ground_truth_version=None,
            checkpoint_data_split_preset=None,
            requested_data_split_preset="legacy_config",
            config_ground_truth_version=LEGACY_GROUND_TRUTH_VERSION,
        )
    except ValueError as exc:
        assert "does not record data_split_preset" in str(exc)
    else:
        raise AssertionError("Checkpoint with no ground-truth metadata silently selected legacy")


def test_checkpoint_semantics_must_match_requested_data_config() -> None:
    assert validate_checkpoint_ground_truth_provenance(
        checkpoint_legacy_ground_truth_version=PER_DEMO_GROUND_TRUTH_VERSION,
        checkpoint_data_split_preset="new_config",
        requested_data_split_preset="new_config",
        config_ground_truth_version=PER_DEMO_GROUND_TRUTH_VERSION,
    ) == PER_DEMO_GROUND_TRUTH_VERSION

    try:
        validate_checkpoint_ground_truth_provenance(
            checkpoint_legacy_ground_truth_version=PER_DEMO_GROUND_TRUTH_VERSION,
            checkpoint_data_split_preset="new_config",
            requested_data_split_preset="new_config",
            config_ground_truth_version=LEGACY_GROUND_TRUTH_VERSION,
        )
    except ValueError as exc:
        assert "declares" in str(exc)
    else:
        raise AssertionError("Checkpoint semantics mismatch was accepted")


def _metrics(mse: float) -> dict[str, float]:
    return {"mse": mse, "mae": mse + 1.0}


def test_trajectory_shards_cover_each_item_once() -> None:
    work_items = [(10, "demo_a"), (10, "demo_b"), (11, "demo_c"), (12, "demo_d"), (12, "demo_e")]
    shards = [shard_trajectory_work_items(work_items, rank=rank, world_size=3) for rank in range(3)]

    assert [entry[0] for shard in shards for entry in shard] == [0, 3, 1, 4, 2]
    assert sorted(entry[0] for shard in shards for entry in shard) == list(range(len(work_items)))
    assert [(task_id, demo_id) for shard in shards for _, task_id, demo_id in shard] == [
        (10, "demo_a"),
        (12, "demo_d"),
        (10, "demo_b"),
        (12, "demo_e"),
        (11, "demo_c"),
    ]


def test_merge_sharded_metrics_restores_global_order_and_task_means() -> None:
    rank_payloads = [
        [(0, 10, _metrics(1.0)), (3, 11, _metrics(4.0))],
        [(1, 10, _metrics(2.0))],
        [(2, 11, _metrics(3.0))],
    ]

    all_metrics, per_task = merge_sharded_trajectory_metrics(
        rank_payloads,
        task_ids=[10, 11],
        total_trajectories=4,
    )

    assert [metrics["mse"] for metrics in all_metrics] == [1.0, 2.0, 3.0, 4.0]
    assert per_task == {
        "10": aggregate_metric_dicts([_metrics(1.0), _metrics(2.0)]),
        "11": aggregate_metric_dicts([_metrics(3.0), _metrics(4.0)]),
    }


def test_context_cache_statistics_sum_across_ranks() -> None:
    summary = aggregate_context_cache_summaries(
        [
            {"enabled": True, "entries": 2, "requests": 5, "hits": 3, "misses": 2, "resident_bytes": 100},
            {"enabled": True, "entries": 3, "requests": 8, "hits": 6, "misses": 2, "resident_bytes": 200},
        ]
    )

    assert summary == {
        "enabled": True,
        "entries": 5,
        "requests": 13,
        "hits": 9,
        "misses": 4,
        "hit_rate": 9.0 / 13.0,
        "resident_bytes": 300,
    }


def test_trajectory_arrays_are_saved_next_to_curve_with_metadata() -> None:
    result = {
        "timesteps": np.asarray([0, 1], dtype=np.int64),
        "target": np.asarray([-1.0, 0.0], dtype=np.float32),
        "prediction": np.asarray([-0.8, -0.1], dtype=np.float32),
        "logits": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "context_demo_ids": np.asarray(["demo_7", "demo_7"]),
    }
    metadata = {"task_id": 3, "demo_id": "demo_2", "metrics": {"mse": 0.025}}
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "curves" / "task_003_demo_2.npz"
        save_trajectory_data(path, result, metadata)
        with np.load(path, allow_pickle=False) as saved:
            assert saved.files == ["timesteps", "target", "prediction", "logits", "metadata_json", "context_demo_ids"]
            assert saved["timesteps"].tolist() == [0, 1]
            assert saved["context_demo_ids"].tolist() == ["demo_7", "demo_7"]
            assert json.loads(saved["metadata_json"].item()) == metadata


def torchrun_collects_one_payload_per_rank() -> None:
    info = initialize_distributed_evaluation(None)
    try:
        assert info.enabled
        gathered: list[dict[str, int] | None] = [None] * info.world_size
        dist.all_gather_object(gathered, {"rank": info.rank})
        assert gathered == [{"rank": rank} for rank in range(info.world_size)]
        distributed_barrier(info)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    test_checkpoint_semantics_require_explicit_checkpoint_or_data_config_metadata()
    test_checkpoint_semantics_must_match_requested_data_config()
    test_trajectory_shards_cover_each_item_once()
    test_merge_sharded_metrics_restores_global_order_and_task_means()
    test_context_cache_statistics_sum_across_ranks()
    test_trajectory_arrays_are_saved_next_to_curve_with_metadata()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torchrun_collects_one_payload_per_rank()
