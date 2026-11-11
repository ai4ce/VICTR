from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .io import read_json
from .observation_history import normalize_query_history_offsets, query_history_frame_indices


LIBERO_PROPRIO_KEYS = "joint_states,gripper_states"
CONTEXT_MAX_ORIGINAL_TIMESTEPS = 520
CONTEXT_STRIDE = 10
CONTEXT_IMAGE_SIZE = 196


def derive_context_max_timesteps(
    max_original_timesteps: int = CONTEXT_MAX_ORIGINAL_TIMESTEPS,
    stride: int = CONTEXT_STRIDE,
) -> int:
    if stride <= 0:
        raise ValueError("context stride must be positive")
    if max_original_timesteps <= 0:
        raise ValueError("context max original timesteps must be positive")
    return (int(max_original_timesteps) + int(stride) - 1) // int(stride)


CONTEXT_MAX_TIMESTEPS = derive_context_max_timesteps()


@dataclass(frozen=True)
class FrameRef:
    task_id: int
    demo_id: str
    frame_idx: int


def end_anchored_context_frame_indices(
    length: int,
    *,
    stride: int = CONTEXT_STRIDE,
    max_original_timesteps: int = CONTEXT_MAX_ORIGINAL_TIMESTEPS,
    max_sampled_timesteps: int | None = None,
) -> list[int]:
    """Sample context frames from the end, then return them chronologically."""

    if length <= 0:
        return []
    if stride <= 0:
        raise ValueError("context stride must be positive")
    sampled_limit = (
        int(max_sampled_timesteps)
        if max_sampled_timesteps is not None
        else derive_context_max_timesteps(max_original_timesteps, stride)
    )
    window_start = max(0, int(length) - int(max_original_timesteps))
    selected = list(range(int(length) - 1, window_start - 1, -int(stride)))[:sampled_limit]
    return sorted(selected)


def uniform_context_frame_indices(length: int, num_timesteps: int) -> list[int]:
    """Map the whole demo to fixed chronological slots, including both endpoints.

    Short demos repeat nearest frames; no interpolation or padding is used.
    """
    if length <= 0 or num_timesteps <= 0:
        raise ValueError("Uniform context requires a nonempty demo and positive slot count")
    return np.rint(np.linspace(0, length - 1, num_timesteps)).astype(np.int64).tolist()


class VfeFrameDataset(Dataset):
    """Frame-level target dataset for π*0.6-style value prediction."""

    def __init__(
        self,
        processed_dir: str | Path,
        data_split_preset: str,
        split: str = "train",
        obs_key: str = "agentview_rgb",
        proprio_key: str = "",
        image_size: int = 128,
        seed: int = 0,
        task_id_filter: set[int] | None = None,
        query_demo_ids_by_task: dict[int, list[str]] | None = None,
        context_demo_ids_by_task: dict[int, list[str]] | None = None,
        context_enabled: bool = False,
        context_progress_enabled: bool = False,
        context_selection: str = "train_seeded",
        context_seed: int = 0,
        context_obs_key: str = "",
        context_image_size: int = CONTEXT_IMAGE_SIZE,
        context_max_timesteps: int | None = None,
        context_sampling_mode: str = "end_anchored",
        context_stride: int = CONTEXT_STRIDE,
        context_max_original_timesteps: int = CONTEXT_MAX_ORIGINAL_TIMESTEPS,
        context_native_feature_dir: str | Path | None = None,
        native_vision_model_name: str = "google/gemma-3-4b-pt",
        native_vision_model_revision: str | None = None,
        context_feature_model_name: str | None = None,
        context_feature_model_revision: str | None = None,
        context_feature_tokens_per_image: int = 256,
        context_native_feature_stage: str | None = None,
        context_native_feature_dtype: str | None = None,
        query_history_offsets: tuple[int, ...] | list[int] = (0,),
    ) -> None:
        self.processed_dir = Path(processed_dir).expanduser().resolve()
        self.summary = read_json(self.processed_dir / "summary.json")
        self.data_split_preset = str(data_split_preset)
        stored_data_split_preset = self.summary.get("data_split_preset")
        if stored_data_split_preset != self.data_split_preset:
            raise ValueError(
                f"{self.processed_dir} was preprocessed for data_split_preset={stored_data_split_preset!r}, "
                f"but {self.data_split_preset!r} was requested. Run preprocessing for the requested data config."
            )
        self.split = split
        self.obs_key = obs_key
        self.obs_keys = [key.strip() for key in obs_key.split(",") if key.strip()]
        self.proprio_keys = [key.strip() for key in proprio_key.split(",") if key.strip()]
        self.proprio_key = ",".join(self.proprio_keys)
        self.image_size = image_size
        self.query_history_offsets = normalize_query_history_offsets(query_history_offsets)
        self.rng = random.Random(seed)
        self.context_enabled = context_enabled
        self.context_progress_enabled = bool(context_progress_enabled)
        if self.context_progress_enabled and not self.context_enabled:
            raise ValueError("context_progress_enabled requires context_enabled=True")
        self.context_selection = context_selection
        self.context_seed = int(context_seed)
        self.context_obs_keys = [key.strip() for key in (context_obs_key or obs_key).split(",") if key.strip()]
        self.context_image_size = int(context_image_size)
        self.context_stride = int(context_stride)
        self.context_max_original_timesteps = int(context_max_original_timesteps)
        self.context_max_timesteps = (
            int(context_max_timesteps)
            if context_max_timesteps is not None
            else derive_context_max_timesteps(self.context_max_original_timesteps, self.context_stride)
        )
        self.context_native_feature_dir = (
            Path(context_native_feature_dir).expanduser().resolve() if context_native_feature_dir else None
        )
        self.context_sampling_mode = context_sampling_mode
        if context_sampling_mode not in {"end_anchored", "uniform"}:
            raise ValueError(f"Unknown context sampling mode: {context_sampling_mode}")
        self.context_feature_store: Any | None = None
        if self.context_native_feature_dir is not None:
            if not self.context_enabled:
                raise ValueError("A context native-feature cache requires context_enabled=True")
            from .native_features import ContextNativeFeatureStore

            self.context_feature_store = ContextNativeFeatureStore(
                self.context_native_feature_dir,
                native_vision_model_name=native_vision_model_name,
                native_vision_model_revision=native_vision_model_revision,
                feature_model_name=context_feature_model_name,
                feature_model_revision=context_feature_model_revision,
                expected_tokens_per_image=context_feature_tokens_per_image,
                obs_keys=self.context_obs_keys,
                context_stride=self.context_stride,
                context_max_original_timesteps=self.context_max_original_timesteps,
                context_sampling_mode=self.context_sampling_mode,
                context_max_timesteps=self.context_max_timesteps,
                expected_feature_stage=context_native_feature_stage,
                expected_feature_dtype=context_native_feature_dtype,
            )
        self.query_demo_ids_by_task = {
            int(task_id): [str(demo_id) for demo_id in demo_ids]
            for task_id, demo_ids in (query_demo_ids_by_task or {}).items()
        }
        self.context_demo_ids_by_task = {
            int(task_id): [str(demo_id) for demo_id in demo_ids]
            for task_id, demo_ids in (context_demo_ids_by_task or {}).items()
        }
        self._h5_cache: dict[str, h5py.File] = {}
        self._video_cache: OrderedDict[str, Any] = OrderedDict()
        # Keep decoders alive across nearby increasing frame requests. Full
        # trajectory evaluation asks for frames in order, and context sampling
        # also advances chronologically. Reusing the decoder preserves the
        # source pixels while avoiding one keyframe seek per requested frame.
        self._video_sequential_state: dict[str, tuple[Any, int]] = {}
        self._parquet_column_cache: dict[tuple[str, str], np.ndarray] = {}
        self._label_cache: dict[str, dict[str, np.ndarray]] = {}

        self.tasks: dict[int, dict[str, Any]] = {}
        self.all_demo_by_task: dict[int, dict[str, dict[str, Any]]] = {}
        self.demo_by_task: dict[int, dict[str, dict[str, Any]]] = {}
        self.query_demo_local_index_by_task: dict[int, dict[str, int]] = {}
        self.frames: list[FrameRef] = []
        self.by_task_indices: dict[int, list[int]] = {}

        for task_stub in self.summary["tasks"]:
            task_path = self.processed_dir / task_stub["metadata_path"]
            task = read_json(task_path)
            task_id = int(task["task_id"])
            if task_id_filter is not None and task_id not in task_id_filter:
                continue
            all_demos = {demo["demo_id"]: demo for demo in task["demos"]}
            selected_demos = self._select_query_demos(task_id, task["demos"], split)
            if not selected_demos:
                continue
            self.tasks[task_id] = task
            self.all_demo_by_task[task_id] = all_demos
            demos = {demo["demo_id"]: demo for demo in selected_demos}
            self.demo_by_task[task_id] = demos
            self.by_task_indices[task_id] = []
            ordered_query_demo_ids = sorted(demos, key=self._demo_sort_key)
            self.query_demo_local_index_by_task[task_id] = {
                demo_id: index for index, demo_id in enumerate(ordered_query_demo_ids)
            }
            if self.context_enabled:
                context_ids = self.context_demo_ids_by_task.get(task_id, [])
                if not context_ids:
                    raise ValueError(f"Context is enabled but task {task_id} has no context demo ids")
                missing = [demo_id for demo_id in context_ids if demo_id not in all_demos]
                if missing:
                    raise KeyError(f"Task {task_id} context demos are not in metadata: {missing}")

            for demo in selected_demos:
                demo_id = demo["demo_id"]
                for frame_idx in range(int(demo["demo_length"])):
                    self.by_task_indices[task_id].append(len(self.frames))
                    self.frames.append(FrameRef(task_id=task_id, demo_id=demo_id, frame_idx=frame_idx))

        if not self.frames:
            raise ValueError(f"No target frames found for split={split} in {self.processed_dir}")
        if self.context_feature_store is not None:
            self.context_feature_store.require_entries(
                {
                    task_id: self.context_demo_ids_by_task[task_id]
                    for task_id in self.tasks
                }
            )

    def _select_query_demos(self, task_id: int, demos: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        if self.query_demo_ids_by_task:
            allowed = set(self.query_demo_ids_by_task.get(task_id, []))
            selected = [demo for demo in demos if demo["demo_id"] in allowed]
            missing = sorted(allowed.difference({demo["demo_id"] for demo in demos}), key=self._demo_sort_key)
            if missing:
                raise KeyError(f"Task {task_id} query demos are not in metadata: {missing}")
            return selected
        selected_demos = []
        for demo in demos:
            if split != "all" and "demo_split" not in demo:
                raise ValueError(
                    f"{self.processed_dir} does not contain per-demo train/test split metadata. "
                    "Re-run scripts/preprocess_vfe.py with the matching data preset."
                )
            demo_split = demo.get("demo_split", "all")
            if split == "all" or demo_split == split:
                selected_demos.append(demo)
        return selected_demos

    def _demo_sort_key(self, demo_id: str) -> tuple[int, str]:
        if demo_id.startswith("demo_"):
            suffix = demo_id[5:]
            if suffix.isdigit():
                return int(suffix), demo_id
        return 10**9, demo_id

    def __len__(self) -> int:
        return len(self.frames)

    def close(self) -> None:
        for h5_file in self._h5_cache.values():
            h5_file.close()
        self._h5_cache.clear()
        for container in self._video_cache.values():
            container.close()
        self._video_cache.clear()
        self._video_sequential_state.clear()
        self._parquet_column_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def task_ids(self) -> list[int]:
        return sorted(self.tasks.keys())

    def infer_proprio_dim(self) -> int:
        if not self.proprio_keys:
            return 0
        ref = self.frames[0]
        return int(self.load_frame_proprio(ref.task_id, ref.demo_id, ref.frame_idx).numel())

    def _h5(self, source_hdf5: str) -> h5py.File:
        if source_hdf5 not in self._h5_cache:
            self._h5_cache[source_hdf5] = h5py.File(source_hdf5, "r")
        return self._h5_cache[source_hdf5]

    def _labels(self, label_path: str) -> dict[str, np.ndarray]:
        abs_path = str(self.processed_dir / label_path)
        if abs_path not in self._label_cache:
            with np.load(abs_path, allow_pickle=False) as data:
                self._label_cache[abs_path] = {key: data[key] for key in data.files}
        return self._label_cache[abs_path]

    def demo_entry(self, task_id: int, demo_id: str) -> dict[str, Any]:
        return self.all_demo_by_task[int(task_id)][demo_id]

    def task_entry(self, task_id: int) -> dict[str, Any]:
        return self.tasks[int(task_id)]

    def load_label(self, task_id: int, demo_id: str) -> dict[str, np.ndarray]:
        demo = self.demo_entry(task_id, demo_id)
        labels = self._labels(demo["label_path"])
        recorded_preset = labels.get("data_split_preset")
        recorded_preset = str(np.asarray(recorded_preset).item()) if recorded_preset is not None else None
        if recorded_preset != self.data_split_preset:
            raise ValueError(
                f"{self.processed_dir / demo['label_path']} records data_split_preset={recorded_preset!r}, "
                f"but the dataset requires {self.data_split_preset!r}. Re-run preprocessing for this data config."
            )
        return labels

    def load_frame_image(
        self,
        task_id: int,
        demo_id: str,
        frame_idx: int,
        *,
        obs_keys: list[str] | None = None,
        image_size: int | None = None,
    ) -> torch.Tensor:
        keys = obs_keys or self.obs_keys
        images = [
            self._load_single_frame_image(task_id, demo_id, frame_idx, obs_key, image_size=image_size)
            for obs_key in keys
        ]
        if len(images) == 1:
            return images[0]
        return torch.stack(images, dim=0)

    def _load_single_frame_image(
        self,
        task_id: int,
        demo_id: str,
        frame_idx: int,
        obs_key: str,
        *,
        image_size: int | None = None,
    ) -> torch.Tensor:
        demo = self.demo_entry(task_id, demo_id)
        if demo.get("source_format") == "lerobot_v21":
            image = self._load_lerobot_video_frame(demo, frame_idx, obs_key)
            return self._image_to_tensor(
                image, image_size=self.image_size if image_size is None else image_size
            )
        h5_file = self._h5(demo["source_hdf5"])
        dataset_path = f"data/{demo_id}/obs/{obs_key}"
        if dataset_path not in h5_file:
            available = sorted(list(h5_file[f"data/{demo_id}/obs"].keys()))
            raise KeyError(f"{obs_key!r} not found in {demo['source_hdf5']}:{demo_id}. Available: {available}")
        image = h5_file[dataset_path][frame_idx]
        return self._image_to_tensor(image, image_size=self.image_size if image_size is None else image_size)

    def load_frame_proprio(self, task_id: int, demo_id: str, frame_idx: int) -> torch.Tensor:
        if not self.proprio_keys:
            return torch.empty(0, dtype=torch.float32)
        demo = self.demo_entry(task_id, demo_id)
        if demo.get("source_format") == "lerobot_v21":
            values = [
                torch.from_numpy(
                    np.asarray(self._lerobot_parquet_column(demo, key)[frame_idx])
                ).float().flatten()
                for key in self.proprio_keys
            ]
            return torch.cat(values, dim=0) if len(values) > 1 else values[0]
        h5_file = self._h5(demo["source_hdf5"])
        values = []
        for proprio_key in self.proprio_keys:
            for dataset_path in self._proprio_candidates(demo_id, proprio_key):
                if dataset_path in h5_file:
                    values.append(torch.from_numpy(np.asarray(h5_file[dataset_path][frame_idx])).float().flatten())
                    break
            else:
                root_keys = sorted(list(h5_file[f"data/{demo_id}"].keys()))
                obs_keys = sorted(list(h5_file[f"data/{demo_id}/obs"].keys()))
                raise KeyError(
                    f"{proprio_key!r} not found in {demo['source_hdf5']}:{demo_id}. "
                    f"Available root keys: {root_keys}; obs keys: {obs_keys}"
                )
        return torch.cat(values, dim=0) if len(values) > 1 else values[0]

    def _lerobot_parquet_column(self, demo: dict[str, Any], key: str) -> np.ndarray:
        parquet_path = str(demo["source_parquet"])
        cache_key = (parquet_path, key)
        if cache_key not in self._parquet_column_cache:
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError(
                    "LeRobot VFE data requires pyarrow>=24. Re-run scripts/setup_vfe_train.sh."
                ) from exc
            try:
                table = pq.read_table(parquet_path, columns=[key])
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read LeRobot proprio key {key!r} from {parquet_path}"
                ) from exc
            self._parquet_column_cache[cache_key] = np.asarray(
                table.column(key).to_pylist()
            )
        return self._parquet_column_cache[cache_key]

    def _load_lerobot_video_frame(
        self,
        demo: dict[str, Any],
        frame_idx: int,
        obs_key: str,
    ) -> np.ndarray:
        source_videos = demo.get("source_videos", {})
        if obs_key not in source_videos:
            available = ", ".join(sorted(source_videos))
            raise KeyError(
                f"LeRobot observation key {obs_key!r} is unavailable; choose one of: {available}"
            )
        video_path = str(source_videos[obs_key])
        try:
            import av
        except ImportError as exc:
            raise ImportError(
                "LeRobot VFE video loading requires PyAV. Re-run scripts/setup_vfe_train.sh."
            ) from exc
        if video_path not in self._video_cache:
            if len(self._video_cache) >= 16:
                stale_path, stale_container = self._video_cache.popitem(last=False)
                self._video_sequential_state.pop(stale_path, None)
                stale_container.close()
            self._video_cache[video_path] = av.open(video_path)
        else:
            self._video_cache.move_to_end(video_path)
        container = self._video_cache[video_path]
        stream = container.streams.video[0]

        # Starting near the beginning, moving forward one query frame at a
        # time, and advancing by the context stride are all substantially
        # faster as a single sequential decode. A gap cap keeps random-access
        # training samples from decoding a large unwanted interval.
        sequential_state = self._video_sequential_state.get(video_path)
        if frame_idx <= 64 and (
            sequential_state is None or int(frame_idx) < sequential_state[1]
        ):
            container.seek(0, stream=stream, backward=True, any_frame=False)
            sequential_state = (iter(container.decode(stream)), 0)
            self._video_sequential_state[video_path] = sequential_state
        if sequential_state is not None:
            decoder, next_index = sequential_state
            gap = int(frame_idx) - int(next_index)
            if 0 <= gap <= 64:
                try:
                    frame = None
                    for decoded_index in range(next_index, int(frame_idx) + 1):
                        frame = next(decoder)
                    assert frame is not None
                    self._video_sequential_state[video_path] = (
                        decoder,
                        int(frame_idx) + 1,
                    )
                    return frame.to_ndarray(format="rgb24")
                except StopIteration:
                    self._video_sequential_state.pop(video_path, None)

        # Fall back to timestamp-based random access for genuinely non-local
        # requests. Seeking invalidates any decoder tied to the old position.
        self._video_sequential_state.pop(video_path, None)
        fps = float(stream.average_rate or 30)
        target_seconds = float(frame_idx) / fps
        target_pts = int(target_seconds / float(stream.time_base))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        closest_frame = None
        closest_distance = float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_seconds = float(frame.pts * frame.time_base)
            decoded_index = int(round(frame_seconds * fps))
            distance = abs(decoded_index - int(frame_idx))
            if distance < closest_distance:
                closest_frame = frame
                closest_distance = distance
            if decoded_index >= frame_idx:
                break
        if closest_frame is None:
            raise IndexError(
                f"Could not decode frame {frame_idx} from LeRobot video {video_path}"
            )
        return closest_frame.to_ndarray(format="rgb24")

    def _proprio_candidates(self, demo_id: str, proprio_key: str) -> list[str]:
        if proprio_key.startswith("data/"):
            return [proprio_key]
        if "/" in proprio_key:
            return [f"data/{demo_id}/{proprio_key}"]
        return [
            f"data/{demo_id}/{proprio_key}",
            f"data/{demo_id}/obs/{proprio_key}",
        ]

    def _image_to_tensor(self, image: np.ndarray, *, image_size: int | None = None) -> torch.Tensor:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {arr.shape}")
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            tensor = torch.from_numpy(arr).float()
        else:
            tensor = torch.from_numpy(np.moveaxis(arr, -1, 0)).float()
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.max() > 2.0:
            tensor = tensor / 255.0
        tensor = tensor.clamp(0.0, 1.0)
        if image_size is not None and image_size > 0:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor.contiguous()

    def select_context_demo_id(
        self,
        task_id: int,
        query_demo_id: str,
        *,
        frame_idx: int,
        sample_index: int,
        epoch: int,
    ) -> str:
        context_pool = list(self.context_demo_ids_by_task[int(task_id)])
        if self.context_selection == "eval_fixed":
            local_index = self.query_demo_local_index_by_task[int(task_id)][query_demo_id]
            context_index = local_index // 4
            if context_index >= len(context_pool):
                raise IndexError(
                    f"Eval context index {context_index} out of range for task {task_id} context pool size {len(context_pool)}"
                )
            return context_pool[context_index]
        if self.context_selection != "train_seeded":
            raise ValueError(f"Unsupported context selection mode: {self.context_selection}")
        rng = random.Random(
            self.context_seed
            + int(epoch) * 1_000_003
            + int(sample_index) * 101
            + int(task_id) * 10_007
            + int(frame_idx)
        )
        return rng.choice(context_pool)

    def load_context_demo(self, task_id: int, context_demo_id: str) -> dict[str, torch.Tensor]:
        demo = self.demo_entry(task_id, context_demo_id)
        length = int(demo["demo_length"])
        if getattr(self, "context_sampling_mode", "end_anchored") == "uniform":
            frame_indices = uniform_context_frame_indices(length, self.context_max_timesteps)
        else:
            frame_indices = end_anchored_context_frame_indices(
                length,
                stride=self.context_stride,
                max_original_timesteps=self.context_max_original_timesteps,
                max_sampled_timesteps=self.context_max_timesteps,
            )
        num_slots = self.context_max_timesteps
        proprio_dim = self.infer_proprio_dim()
        context_proprio = torch.zeros(num_slots, proprio_dim, dtype=torch.float32)
        context_progress = torch.zeros(num_slots, dtype=torch.float32)
        context_valid_mask = torch.zeros(num_slots, dtype=torch.bool)
        start_slot = num_slots - len(frame_indices)
        context_slot_indices = torch.arange(start_slot, num_slots, dtype=torch.long)
        progress_values: np.ndarray | None = None
        if self.context_progress_enabled:
            labels = self.load_label(task_id, context_demo_id)
            progress_values = np.asarray(labels["value_continuous"], dtype=np.float32)
            if progress_values.shape != (length,):
                raise ValueError(
                    f"Context progress for task {task_id} {context_demo_id} has shape "
                    f"{progress_values.shape}; expected ({length},)"
                )
            if not np.isfinite(progress_values).all():
                raise ValueError(
                    f"Context progress for task {task_id} {context_demo_id} contains non-finite values"
                )
        for offset, frame_idx in enumerate(frame_indices):
            slot = start_slot + offset
            if proprio_dim > 0:
                context_proprio[slot] = self.load_frame_proprio(task_id, context_demo_id, frame_idx)
            if progress_values is not None:
                context_progress[slot] = float(progress_values[frame_idx])
            context_valid_mask[slot] = True

        if self.context_feature_store is not None:
            payload = self.context_feature_store.load(task_id, context_demo_id)
            cached_frame_indices = [int(value) for value in payload["frame_indices"].tolist()]
            if cached_frame_indices != frame_indices:
                raise ValueError(
                    f"Cached context frame indices for task {task_id} {context_demo_id} do not match runtime sampling"
                )
            output = {
                "context_native_features": payload["features"],
                "context_feature_slot_indices": context_slot_indices,
                "context_proprio": context_proprio,
                "context_valid_mask": context_valid_mask,
            }
            if self.context_progress_enabled:
                output["context_progress"] = context_progress
            return output

        num_views = len(self.context_obs_keys)
        if not frame_indices:
            raise ValueError(f"Context demo {context_demo_id} for task {task_id} has no sampled frames")
        # A size of zero preserves the raw LIBERO pixels. This is necessary when
        # query and context must be passed through the exact same HF image
        # processor, as in the Gemma-native-style 270M backbone.
        first_context_image = self.load_frame_image(
            task_id,
            context_demo_id,
            frame_indices[0],
            obs_keys=self.context_obs_keys,
            image_size=self.context_image_size,
        )
        context_images = torch.zeros(
            num_slots,
            *first_context_image.shape,
            dtype=torch.float32,
        )
        for offset, frame_idx in enumerate(frame_indices):
            slot = start_slot + offset
            context_images[slot] = (
                first_context_image
                if offset == 0
                else self.load_frame_image(
                    task_id,
                    context_demo_id,
                    frame_idx,
                    obs_keys=self.context_obs_keys,
                    image_size=self.context_image_size,
                )
            )
        output = {
            "context_images": context_images,
            "context_proprio": context_proprio,
            "context_valid_mask": context_valid_mask,
        }
        if self.context_progress_enabled:
            output["context_progress"] = context_progress
        return output

    def item_from_ref(
        self,
        ref: FrameRef,
        *,
        sample_index: int = 0,
        epoch: int = 0,
        include_context: bool = True,
    ) -> dict[str, Any]:
        task = self.task_entry(ref.task_id)
        target_demo = self.demo_entry(ref.task_id, ref.demo_id)
        labels = self.load_label(ref.task_id, ref.demo_id)
        history_offsets = getattr(self, "query_history_offsets", [0])
        history_indices = query_history_frame_indices(ref.frame_idx, history_offsets)
        # Decode repeated boundary frames only once. All lookups retain the query's demo ID.
        observations = {
            frame_idx: (
                self.load_frame_image(ref.task_id, ref.demo_id, frame_idx),
                self.load_frame_proprio(ref.task_id, ref.demo_id, frame_idx),
            )
            for frame_idx in dict.fromkeys(history_indices)
        }
        query_image, query_proprio = observations[ref.frame_idx]
        if len(history_offsets) > 1:
            query_image = torch.stack([
                image if image.dim() == 4 else image.unsqueeze(0)
                for image, _ in (observations[index] for index in history_indices)
            ])
            query_proprio = torch.stack([observations[index][1] for index in history_indices])
        item = {
            "query_image": query_image,
            "query_proprio": query_proprio,
            "language": task["language_instruction"],
            "value_bin": int(labels["value_bin"][ref.frame_idx]),
            "value_continuous": float(labels["value_continuous"][ref.frame_idx]),
            "task_id": int(ref.task_id),
            "target_demo_id": ref.demo_id,
            # Keep the legacy batch key for checkpoint/log compatibility, but
            # LeRobot episodes are backed by parquet rather than HDF5.
            "target_source_hdf5": target_demo.get(
                "source_hdf5", target_demo.get("source_parquet", "")
            ),
            "obs_keys": task.get("obs_keys", []),
            "proprio_keys": self.proprio_keys,
            "frame_idx": int(ref.frame_idx),
        }
        if self.context_enabled:
            context_demo_id = self.select_context_demo_id(
                ref.task_id,
                ref.demo_id,
                frame_idx=ref.frame_idx,
                sample_index=sample_index,
                epoch=epoch,
            )
            if include_context:
                item.update(self.load_context_demo(ref.task_id, context_demo_id))
            item["context_demo_id"] = context_demo_id
        return item

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        epoch = 0
        sample_index = int(index[0]) if isinstance(index, tuple) else int(index)
        if isinstance(index, tuple):
            epoch = int(index[1])
            index = int(index[0])
        return self.item_from_ref(self.frames[int(index)], sample_index=sample_index, epoch=epoch)


class QueryFrameEpochBatchSampler(Sampler[List[int | tuple[int, int]]]):
    """Shuffle every query frame once per epoch and shard each batch across ranks.

    Each emitted ``(frame_index, epoch)`` pair is consumed by
    :class:`VfeFrameDataset`, which uses ``epoch`` to choose a dynamic context
    demo. The sampler itself never materializes query/context pairs. Without
    final-batch padding, one shuffled pass is exactly one pass over the
    query-frame dataset.

    All ranks receive equally sized local batches, including the final partial
    batch. Consequently, strict no-replacement DDP requires the number of
    query frames to be divisible by ``world_size``. ``pad_last_batch`` trades
    strict no-replacement for a static shape by filling the final incomplete
    batch with a deterministic random sample from the same epoch's other query
    frames.
    """

    def __init__(
        self,
        dataset: VfeFrameDataset,
        *,
        local_batch_size: int,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        pad_last_batch: bool = False,
    ) -> None:
        self.dataset = dataset
        self.local_batch_size = int(local_batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.pad_last_batch = bool(pad_last_batch)
        self._epoch = 0
        self._start_step = 0
        if self.local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")
        if not self.pad_last_batch and len(self.dataset) % self.world_size != 0:
            raise ValueError(
                "query_frame_once requires the number of query frames to be divisible by world_size so every "
                "DDP rank receives equal no-replacement batches; "
                f"got {len(self.dataset)} frames and world_size={self.world_size}."
            )

    @property
    def steps_per_epoch(self) -> int:
        global_batch_size = self.local_batch_size * self.world_size
        return (len(self.dataset) + global_batch_size - 1) // global_batch_size

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self._epoch = int(epoch)
        self._start_step = 0

    def set_start_step(self, start_step: int) -> None:
        if start_step < 0 or start_step > self.steps_per_epoch:
            raise ValueError(
                f"start_step must be in [0, {self.steps_per_epoch}], got {start_step}"
            )
        self._start_step = int(start_step)

    def __len__(self) -> int:
        return self.steps_per_epoch - self._start_step

    def __iter__(self) -> Iterator[list[int | tuple[int, int]]]:
        epoch = self._epoch
        shuffled_indices = list(range(len(self.dataset)))
        epoch_rng = random.Random(self.seed + epoch)
        epoch_rng.shuffle(shuffled_indices)
        global_batch_size = self.local_batch_size * self.world_size

        for step_index, start in enumerate(range(0, len(shuffled_indices), global_batch_size)):
            if step_index < self._start_step:
                continue
            global_batch = shuffled_indices[start : start + global_batch_size]
            if self.pad_last_batch and len(global_batch) < global_batch_size:
                padding_size = global_batch_size - len(global_batch)
                padding_candidates = shuffled_indices[:start] or shuffled_indices
                if padding_size <= len(padding_candidates):
                    global_batch.extend(epoch_rng.sample(padding_candidates, padding_size))
                else:
                    global_batch.extend(epoch_rng.choices(padding_candidates, k=padding_size))
            if len(global_batch) % self.world_size != 0:
                raise AssertionError("Strict no-replacement batch cannot be evenly sharded across ranks")
            local_batch_size = len(global_batch) // self.world_size
            rank_start = self.rank * local_batch_size
            rank_indices = global_batch[rank_start : rank_start + local_batch_size]
            yield [(sample_index, epoch) for sample_index in rank_indices]


class MixedTaskBatchSampler(Sampler[List[int | tuple[int, int]]]):
    """Yield task-balanced batches with replacement (legacy sampling mode)."""

    def __init__(
        self,
        dataset: VfeFrameDataset,
        tasks_per_batch: int = 4,
        samples_per_task: int = 8,
        steps_per_epoch: int = 1000,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.tasks_per_batch = tasks_per_batch
        self.samples_per_task = samples_per_task
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0
        self._start_step = 0
        if tasks_per_batch <= 0 or samples_per_task <= 0:
            raise ValueError("tasks_per_batch and samples_per_task must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self._epoch = int(epoch)
        self._start_step = 0

    def set_start_step(self, start_step: int) -> None:
        if start_step < 0 or start_step > self.steps_per_epoch:
            raise ValueError(
                f"start_step must be in [0, {self.steps_per_epoch}], got {start_step}"
            )
        self._start_step = int(start_step)

    def __len__(self) -> int:
        return self.steps_per_epoch - self._start_step

    def __iter__(self) -> Iterator[list[int | tuple[int, int]]]:
        rng = random.Random(self.seed + self._epoch)
        epoch = self._epoch
        task_ids = self.dataset.task_ids
        global_tasks_per_batch = self.tasks_per_batch * self.world_size
        local_batch_size = self.tasks_per_batch * self.samples_per_task
        start_step = self._start_step
        for step_index in range(self.steps_per_epoch):
            if len(task_ids) >= global_tasks_per_batch:
                batch_tasks = rng.sample(task_ids, global_tasks_per_batch)
            else:
                batch_tasks = [rng.choice(task_ids) for _ in range(global_tasks_per_batch)]
            rank_batches: list[list[int | tuple[int, int]]] = []
            for rank_index in range(self.world_size):
                start_task = rank_index * self.tasks_per_batch
                local_tasks = batch_tasks[start_task : start_task + self.tasks_per_batch]
                local_batch: list[int | tuple[int, int]] = []
                for task_id in local_tasks:
                    candidates = self.dataset.by_task_indices[task_id]
                    for _ in range(self.samples_per_task):
                        local_batch.append((rng.choice(candidates), epoch))
                rng.shuffle(local_batch)
                if len(local_batch) != local_batch_size:
                    raise AssertionError("Distributed sampler produced an incomplete local batch")
                rank_batches.append(local_batch)
            if step_index >= start_step:
                yield rank_batches[self.rank]


def collate_vfe_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated = {
        "query_image": torch.stack([item["query_image"] for item in batch], dim=0),
        "query_proprio": torch.stack([item["query_proprio"] for item in batch], dim=0),
        "language": [item["language"] for item in batch],
        "value_bin": torch.tensor([item["value_bin"] for item in batch], dtype=torch.long),
        "value_continuous": torch.tensor([item["value_continuous"] for item in batch], dtype=torch.float32),
        "task_id": torch.tensor([item["task_id"] for item in batch], dtype=torch.long),
        "target_demo_id": [item["target_demo_id"] for item in batch],
        "target_source_hdf5": [item["target_source_hdf5"] for item in batch],
        "obs_keys": [item["obs_keys"] for item in batch],
        "proprio_keys": [item["proprio_keys"] for item in batch],
        "frame_idx": torch.tensor([item["frame_idx"] for item in batch], dtype=torch.long),
    }
    if "context_images" in batch[0]:
        collated.update(
            {
                "context_images": torch.stack([item["context_images"] for item in batch], dim=0),
                "context_proprio": torch.stack([item["context_proprio"] for item in batch], dim=0),
                "context_valid_mask": torch.stack([item["context_valid_mask"] for item in batch], dim=0),
                "context_demo_id": [item["context_demo_id"] for item in batch],
            }
        )
    elif "context_native_features" in batch[0]:
        packed_features = []
        packed_positions = []
        for batch_index, item in enumerate(batch):
            features = item["context_native_features"]
            slot_indices = item["context_feature_slot_indices"]
            if features.shape[0] != slot_indices.shape[0]:
                raise ValueError("Context native features and slot indices do not align")
            num_slots, num_views = features.shape[:2]
            packed_features.append(features.reshape(num_slots * num_views, *features.shape[2:]))
            packed_positions.append(
                torch.stack(
                    [
                        torch.full((num_slots * num_views,), batch_index, dtype=torch.long),
                        slot_indices.repeat_interleave(num_views),
                        torch.arange(num_views, dtype=torch.long).repeat(num_slots),
                    ],
                    dim=1,
                )
            )
        collated.update(
            {
                "context_native_features": torch.cat(packed_features, dim=0),
                "context_feature_positions": torch.cat(packed_positions, dim=0),
                "context_proprio": torch.stack([item["context_proprio"] for item in batch], dim=0),
                "context_valid_mask": torch.stack([item["context_valid_mask"] for item in batch], dim=0),
                "context_demo_id": [item["context_demo_id"] for item in batch],
            }
        )
    if "context_progress" in batch[0]:
        if not all("context_progress" in item for item in batch):
            raise ValueError("Either every item or no item must contain context_progress")
        collated["context_progress"] = torch.stack(
            [item["context_progress"] for item in batch], dim=0
        )
    return collated


def make_train_dataloader(
    processed_dir: str | Path,
    *,
    split: str = "train",
    obs_key: str = "agentview_rgb",
    proprio_key: str = "",
    image_size: int = 128,
    query_history_offsets: tuple[int, ...] | list[int] = (0,),
    tasks_per_batch: int = 4,
    samples_per_task: int = 8,
    steps_per_epoch: int = 1000,
    sampling_mode: str = "query_frame_once",
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    seed: int = 0,
    task_id_filter: set[int] | None = None,
    query_demo_ids_by_task: dict[int, list[str]] | None = None,
    context_demo_ids_by_task: dict[int, list[str]] | None = None,
    context_enabled: bool = False,
    context_progress_enabled: bool = False,
    context_selection: str = "train_seeded",
    context_seed: int = 0,
    context_obs_key: str = "",
    context_image_size: int = CONTEXT_IMAGE_SIZE,
    context_max_timesteps: int | None = None,
    context_sampling_mode: str = "end_anchored",
    context_stride: int = CONTEXT_STRIDE,
    context_max_original_timesteps: int = CONTEXT_MAX_ORIGINAL_TIMESTEPS,
    context_native_feature_dir: str | Path | None = None,
    native_vision_model_name: str = "google/gemma-3-4b-pt",
    native_vision_model_revision: str | None = None,
    context_feature_model_name: str | None = None,
    context_feature_model_revision: str | None = None,
    context_feature_tokens_per_image: int = 256,
    context_native_feature_stage: str | None = None,
    context_native_feature_dtype: str | None = None,
    data_split_preset: str,
    rank: int = 0,
    world_size: int = 1,
    pad_query_frame_once_last_batch: bool = False,
) -> DataLoader:
    dataset = VfeFrameDataset(
        processed_dir,
        split=split,
        obs_key=obs_key,
        proprio_key=proprio_key,
        image_size=image_size,
        query_history_offsets=query_history_offsets,
        seed=seed,
        task_id_filter=task_id_filter,
        query_demo_ids_by_task=query_demo_ids_by_task,
        context_demo_ids_by_task=context_demo_ids_by_task,
        context_enabled=context_enabled,
        context_progress_enabled=context_progress_enabled,
        context_selection=context_selection,
        context_seed=context_seed,
        context_obs_key=context_obs_key,
        context_image_size=context_image_size,
        context_max_timesteps=context_max_timesteps,
        context_sampling_mode=context_sampling_mode,
        context_stride=context_stride,
        context_max_original_timesteps=context_max_original_timesteps,
        context_native_feature_dir=context_native_feature_dir,
        native_vision_model_name=native_vision_model_name,
        native_vision_model_revision=native_vision_model_revision,
        context_feature_model_name=context_feature_model_name,
        context_feature_model_revision=context_feature_model_revision,
        context_feature_tokens_per_image=context_feature_tokens_per_image,
        context_native_feature_stage=context_native_feature_stage,
        context_native_feature_dtype=context_native_feature_dtype,
        data_split_preset=data_split_preset,
    )
    local_batch_size = tasks_per_batch * samples_per_task
    if sampling_mode == "query_frame_once":
        sampler: Sampler[List[int | tuple[int, int]]] = QueryFrameEpochBatchSampler(
            dataset,
            local_batch_size=local_batch_size,
            seed=seed,
            rank=rank,
            world_size=world_size,
            pad_last_batch=pad_query_frame_once_last_batch,
        )
    elif sampling_mode == "task_balanced_replacement":
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive for task_balanced_replacement")
        sampler = MixedTaskBatchSampler(
            dataset,
            tasks_per_batch=tasks_per_batch,
            samples_per_task=samples_per_task,
            steps_per_epoch=steps_per_epoch,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )
    else:
        raise ValueError(
            "sampling_mode must be one of: 'query_frame_once', 'task_balanced_replacement'; "
            f"got {sampling_mode!r}"
        )
    loader_kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": collate_vfe_batch,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
