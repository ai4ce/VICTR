from __future__ import annotations

from vfe.data import MixedTaskBatchSampler, QueryFrameEpochBatchSampler


class DummyFrameDataset:
    def __init__(self, num_frames: int) -> None:
        self.num_frames = num_frames
        self.task_ids = [0, 1]
        self.by_task_indices = {
            0: list(range(0, num_frames // 2)),
            1: list(range(num_frames // 2, num_frames)),
        }

    def __len__(self) -> int:
        return self.num_frames


def test_query_frame_epoch_sampler_visits_every_frame_once_across_ranks() -> None:
    dataset = DummyFrameDataset(num_frames=14)
    samplers = [
        QueryFrameEpochBatchSampler(dataset, local_batch_size=3, seed=7, rank=rank, world_size=2)
        for rank in range(2)
    ]
    for sampler in samplers:
        sampler.set_epoch(4)

    rank_batches = [list(sampler) for sampler in samplers]
    assert [len(batch) for batch in rank_batches[0]] == [3, 3, 1]
    assert [len(batch) for batch in rank_batches[1]] == [3, 3, 1]
    observed = [sample_index for batches in rank_batches for batch in batches for sample_index, _ in batch]
    assert sorted(observed) == list(range(14))
    assert len(observed) == len(set(observed))
    assert {epoch for batches in rank_batches for batch in batches for _, epoch in batch} == {4}


def test_query_frame_epoch_sampler_can_randomly_pad_last_batch_to_static_shape() -> None:
    dataset = DummyFrameDataset(num_frames=14)
    samplers = [
        QueryFrameEpochBatchSampler(
            dataset,
            local_batch_size=3,
            seed=7,
            rank=rank,
            world_size=2,
            pad_last_batch=True,
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        sampler.set_epoch(4)

    rank_batches = [list(sampler) for sampler in samplers]
    assert [len(batch) for batch in rank_batches[0]] == [3, 3, 3]
    assert [len(batch) for batch in rank_batches[1]] == [3, 3, 3]
    observed = [sample_index for batches in rank_batches for batch in batches for sample_index, _ in batch]
    assert set(observed) == set(range(14))
    assert len(observed) == 18
    counts = sorted(observed.count(index) for index in range(14))
    assert counts == [1] * 10 + [2] * 4
    assert {epoch for batches in rank_batches for batch in batches for _, epoch in batch} == {4}

    repeated = [
        QueryFrameEpochBatchSampler(
            dataset,
            local_batch_size=3,
            seed=7,
            rank=rank,
            world_size=2,
            pad_last_batch=True,
        )
        for rank in range(2)
    ]
    for sampler in repeated:
        sampler.set_epoch(4)
    assert [list(sampler) for sampler in repeated] == rank_batches


def test_query_frame_epoch_sampler_can_pad_dataset_not_divisible_by_world_size() -> None:
    dataset = DummyFrameDataset(num_frames=13)
    samplers = [
        QueryFrameEpochBatchSampler(
            dataset,
            local_batch_size=3,
            seed=11,
            rank=rank,
            world_size=4,
            pad_last_batch=True,
        )
        for rank in range(4)
    ]

    rank_batches = [list(sampler) for sampler in samplers]
    assert all([len(batch) for batch in batches] == [3, 3] for batches in rank_batches)
    observed = [sample_index for batches in rank_batches for batch in batches for sample_index, _ in batch]
    assert set(observed) == set(range(13))
    assert len(observed) == 24

    try:
        QueryFrameEpochBatchSampler(
            dataset,
            local_batch_size=3,
            rank=0,
            world_size=4,
            pad_last_batch=False,
        )
    except ValueError as exc:
        assert "divisible by world_size" in str(exc)
    else:
        raise AssertionError("A non-divisible strict no-replacement split was accepted")


def test_query_frame_epoch_sampler_is_resumable_and_changes_context_epoch() -> None:
    dataset = DummyFrameDataset(num_frames=14)
    sampler = QueryFrameEpochBatchSampler(dataset, local_batch_size=3, seed=7, rank=0, world_size=2)

    sampler.set_epoch(2)
    full_epoch = list(sampler)
    sampler.set_epoch(2)
    sampler.set_start_step(1)
    assert list(sampler) == full_epoch[1:]

    sampler.set_epoch(3)
    next_epoch = list(sampler)
    assert {epoch for batch in next_epoch for _, epoch in batch} == {3}
    assert next_epoch != full_epoch


def test_legacy_task_balanced_sampler_uses_the_epoch_set_by_training() -> None:
    dataset = DummyFrameDataset(num_frames=12)
    sampler = MixedTaskBatchSampler(
        dataset,
        tasks_per_batch=2,
        samples_per_task=1,
        steps_per_epoch=3,
        seed=5,
    )
    sampler.set_epoch(6)
    first = list(sampler)
    sampler.set_epoch(6)
    assert list(sampler) == first
    assert {epoch for batch in first for _, epoch in batch} == {6}
