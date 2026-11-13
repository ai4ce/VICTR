from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")



def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))



def save_demo_labels(
    path: str | Path,
    *,
    value_continuous: np.ndarray,
    value_bin: np.ndarray,
    demo_length: int,
    demo_id: str,
    task_id: int,
    task_name: str,
    demo_role: str,
    task_split: str,
    demo_split: str,
    source_hdf5: str,
    num_bins: int,
    value_min: float,
    value_max: float,
    data_split_preset: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        value_continuous=np.asarray(value_continuous, dtype=np.float32),
        value_bin=np.asarray(value_bin, dtype=np.int64),
        demo_length=np.asarray(demo_length, dtype=np.int64),
        demo_id=np.asarray(demo_id),
        task_id=np.asarray(task_id, dtype=np.int64),
        task_name=np.asarray(task_name),
        demo_role=np.asarray(demo_role),
        task_split=np.asarray(task_split),
        demo_split=np.asarray(demo_split),
        source_hdf5=np.asarray(source_hdf5),
        num_bins=np.asarray(num_bins, dtype=np.int64),
        value_min=np.asarray(value_min, dtype=np.float32),
        value_max=np.asarray(value_max, dtype=np.float32),
        data_split_preset=np.asarray(data_split_preset),
    )
