from __future__ import annotations

from collections.abc import Sequence


def normalize_query_history_offsets(offsets: Sequence[int]) -> list[int]:
    """Offsets are frame ages, ordered oldest to current (e.g. [60, 30, 0])."""

    offsets = list(offsets)
    if (
        not offsets
        or any(type(offset) is not int or offset < 0 for offset in offsets)
        or offsets[-1] != 0
        or any(left <= right for left, right in zip(offsets, offsets[1:]))
    ):
        raise ValueError(
            "query_history_offsets must be strictly decreasing nonnegative integers ending in 0"
        )
    return offsets


def query_history_frame_indices(frame_idx: int, offsets: Sequence[int]) -> list[int]:
    """Repeat the episode's first observation when a requested past frame is absent."""

    if frame_idx < 0:
        raise ValueError("query frame index must be nonnegative")
    return [max(0, frame_idx - offset) for offset in offsets]
