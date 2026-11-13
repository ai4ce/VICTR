from __future__ import annotations

import argparse
import gc
import hashlib
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .data import (
    end_anchored_context_frame_indices,
    uniform_context_frame_indices,
)
from .data_splits import (
    DEFAULT_PRESET_PATH,
    load_preset_preprocess_config,
    resolve_data_split,
    resolve_processed_dir,
)
from .io import read_json, write_json
from .vfe_presets import DEFAULT_VFE_PRESET_PATH, load_vfe_preset


CONTEXT_NATIVE_FEATURE_FORMAT_VERSION = 1
CONTEXT_NATIVE_FEATURE_TYPE = "gemma3_native_projector_tokens"
CONTEXT_HF_VISION_FEATURE_TYPE = "hf_vision_patch_tokens"
CONTEXT_RECAP_DETAIL_FEATURE_TYPE = "recap_qwen2_detail_tokens"
CONTEXT_FEATURE_TYPES = {
    CONTEXT_NATIVE_FEATURE_TYPE,
    CONTEXT_HF_VISION_FEATURE_TYPE,
    CONTEXT_RECAP_DETAIL_FEATURE_TYPE,
}
CONTEXT_NATIVE_FEATURE_MANIFEST = "manifest.json"
NATIVE_VISION_POOLED_STAGE = "native_vision_pooled"
NATIVE_PROJECTOR_INPUT_STAGE = "native_projector_input"
NATIVE_PROJECTOR_OUTPUT_STAGE = "native_projector_output"
RECAP_VISION_PATCH_STAGE = "recap_vision_patches"
RECAP_QWEN2_DETAIL_STAGE = "recap_qwen2_detail"
NATIVE_FEATURE_STAGES = {
    NATIVE_VISION_POOLED_STAGE,
    NATIVE_PROJECTOR_INPUT_STAGE,
    NATIVE_PROJECTOR_OUTPUT_STAGE,
    RECAP_VISION_PATCH_STAGE,
    RECAP_QWEN2_DETAIL_STAGE,
}


def _feature_type_for_stage(feature_stage: str) -> str:
    if feature_stage == RECAP_VISION_PATCH_STAGE:
        return CONTEXT_HF_VISION_FEATURE_TYPE
    if feature_stage == RECAP_QWEN2_DETAIL_STAGE:
        return CONTEXT_RECAP_DETAIL_FEATURE_TYPE
    return CONTEXT_NATIVE_FEATURE_TYPE


def canonical_feature_dtype_name(name: str) -> str:
    normalized = str(name).lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported feature dtype: {name}")
    return aliases[normalized]


def context_feature_key(task_id: int, demo_id: str) -> str:
    return f"task_{int(task_id):03d}/{demo_id}"


def context_feature_relative_path(task_id: int, demo_id: str) -> Path:
    return Path(f"task_{int(task_id):03d}") / f"{demo_id}.pt"


def load_context_feature_manifest(feature_dir: str | Path) -> dict[str, Any]:
    root = Path(feature_dir).expanduser().resolve()
    path = root / CONTEXT_NATIVE_FEATURE_MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"Context native-feature manifest not found: {path}")
    manifest = read_json(path)
    if int(manifest.get("format_version", -1)) != CONTEXT_NATIVE_FEATURE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported context native-feature format version {manifest.get('format_version')!r} in {path}; "
            f"expected {CONTEXT_NATIVE_FEATURE_FORMAT_VERSION}"
        )
    if manifest.get("feature_type") not in CONTEXT_FEATURE_TYPES:
        raise ValueError(
            f"Unsupported context feature type {manifest.get('feature_type')!r} in {path}; "
            f"expected one of {sorted(CONTEXT_FEATURE_TYPES)}"
        )
    if manifest.get("feature_stage") not in NATIVE_FEATURE_STAGES:
        raise ValueError(
            f"Unsupported context feature stage {manifest.get('feature_stage')!r} in {path}; "
            f"expected one of {sorted(NATIVE_FEATURE_STAGES)}"
        )
    manifest["_feature_dir"] = str(root)
    manifest["_manifest_path"] = str(path)
    return manifest


def validate_context_feature_manifest(
    manifest: dict[str, Any],
    *,
    native_vision_model_name: str | None = None,
    native_vision_model_revision: str | None = None,
    feature_model_name: str | None = None,
    feature_model_revision: str | None = None,
    obs_keys: list[str],
    context_stride: int,
    context_max_original_timesteps: int,
    context_sampling_mode: str = "end_anchored",
    context_max_timesteps: int | None = None,
    expected_tokens_per_image: int = 256,
    expected_feature_stage: str | None = None,
    expected_feature_dtype: str | None = None,
) -> None:
    requested_model_name = feature_model_name or native_vision_model_name
    requested_model_revision = (
        feature_model_revision if feature_model_name is not None else native_vision_model_revision
    )
    if not requested_model_name:
        raise ValueError("Context feature validation requires a feature model name")
    cached_model_name = manifest.get("feature_model_name", manifest.get("native_vision_model_name"))
    cached_model_revision = manifest.get(
        "feature_model_revision", manifest.get("native_vision_model_revision")
    )
    expected = {
        "obs_keys": list(obs_keys),
        "selection": context_sampling_mode,
        "tokens_per_image": int(expected_tokens_per_image),
    }
    if context_sampling_mode == "uniform":
        if context_max_timesteps is None or context_max_timesteps <= 0:
            raise ValueError("Uniform cache validation requires a positive context_max_timesteps")
        expected["context_max_timesteps"] = int(context_max_timesteps)
        expected["uniform_sampling"] = "linspace_rint_v1"
    elif context_sampling_mode == "end_anchored":
        expected["context_stride"] = int(context_stride)
        expected["context_max_original_timesteps"] = int(context_max_original_timesteps)
    else:
        raise ValueError(f"Unknown context sampling mode: {context_sampling_mode}")
    if expected_feature_stage is not None:
        expected["feature_stage"] = expected_feature_stage
        expected["feature_type"] = _feature_type_for_stage(expected_feature_stage)
    if expected_feature_dtype is not None:
        expected["feature_dtype"] = canonical_feature_dtype_name(expected_feature_dtype)
    mismatches = []
    if cached_model_name != str(requested_model_name):
        mismatches.append(
            f"feature_model_name: cache={cached_model_name!r}, requested={str(requested_model_name)!r}"
        )
    if cached_model_revision != requested_model_revision:
        mismatches.append(
            "feature_model_revision: "
            f"cache={cached_model_revision!r}, requested={requested_model_revision!r}"
        )
    for key, value in expected.items():
        if manifest.get(key) != value:
            mismatches.append(f"{key}: cache={manifest.get(key)!r}, requested={value!r}")
    if mismatches:
        raise ValueError(
            f"Context native-feature cache {manifest.get('_manifest_path', '<unknown>')} is incompatible: "
            + "; ".join(mismatches)
        )


def _load_torch_shard(path: Path, *, mmap: bool) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": True}
    if mmap:
        kwargs["mmap"] = True
    try:
        payload = torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        payload = torch.load(path, **kwargs)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a tensor dictionary in {path}, got {type(payload).__name__}")
    return payload


class ContextNativeFeatureStore:
    """Read-only, mmap-backed access to per-demo frozen native visual features."""

    def __init__(
        self,
        feature_dir: str | Path,
        *,
        native_vision_model_name: str | None = None,
        obs_keys: list[str],
        context_stride: int,
        context_max_original_timesteps: int,
        context_sampling_mode: str = "end_anchored",
        context_max_timesteps: int | None = None,
        expected_tokens_per_image: int = 256,
        native_vision_model_revision: str | None = None,
        feature_model_name: str | None = None,
        feature_model_revision: str | None = None,
        expected_feature_stage: str | None = None,
        expected_feature_dtype: str | None = None,
        max_cached_payloads: int = 8,
    ) -> None:
        self.root = Path(feature_dir).expanduser().resolve()
        self.manifest = load_context_feature_manifest(self.root)
        validate_context_feature_manifest(
            self.manifest,
            native_vision_model_name=native_vision_model_name,
            native_vision_model_revision=native_vision_model_revision,
            feature_model_name=feature_model_name,
            feature_model_revision=feature_model_revision,
            obs_keys=obs_keys,
            context_stride=context_stride,
            context_max_original_timesteps=context_max_original_timesteps,
            context_sampling_mode=context_sampling_mode,
            context_max_timesteps=context_max_timesteps,
            expected_tokens_per_image=expected_tokens_per_image,
            expected_feature_stage=expected_feature_stage,
            expected_feature_dtype=expected_feature_dtype,
        )
        self.entries = dict(self.manifest.get("entries", {}))
        self.num_views = len(obs_keys)
        self.tokens_per_image = int(self.manifest["tokens_per_image"])
        self.feature_dim = int(self.manifest["feature_dim"])
        self.feature_stage = str(self.manifest["feature_stage"])
        self.max_cached_payloads = int(max_cached_payloads)
        if self.max_cached_payloads <= 0:
            raise ValueError("max_cached_payloads must be positive")
        self._payload_cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

    def require_entries(self, demo_ids_by_task: dict[int, list[str]]) -> None:
        missing = [
            context_feature_key(task_id, demo_id)
            for task_id, demo_ids in demo_ids_by_task.items()
            for demo_id in demo_ids
            if context_feature_key(task_id, demo_id) not in self.entries
        ]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = f" and {len(missing) - 8} more" if len(missing) > 8 else ""
            raise KeyError(f"Context native-feature cache is missing {preview}{suffix}")

    def load(self, task_id: int, demo_id: str) -> dict[str, torch.Tensor]:
        key = context_feature_key(task_id, demo_id)
        if key not in self.entries:
            raise KeyError(f"Context native-feature cache has no entry for {key}")
        if key not in self._payload_cache:
            path = self.root / self.entries[key]["path"]
            payload = _load_torch_shard(path, mmap=True)
            features = payload.get("features")
            frame_indices = payload.get("frame_indices")
            if not isinstance(features, torch.Tensor) or not isinstance(frame_indices, torch.Tensor):
                raise KeyError(f"{path} must contain tensor keys 'features' and 'frame_indices'")
            expected_tail = (self.num_views, self.tokens_per_image, self.feature_dim)
            if features.dim() != 4 or tuple(features.shape[1:]) != expected_tail:
                raise ValueError(
                    f"{path} features shape {tuple(features.shape)} does not match [slots, {expected_tail}]"
                )
            if frame_indices.dim() != 1 or frame_indices.shape[0] != features.shape[0]:
                raise ValueError(f"{path} frame_indices do not align with features")
            if self.manifest.get("feature_dtype") is not None:
                expected_dtype = _resolve_torch_dtype(self.manifest["feature_dtype"])
                if features.dtype != expected_dtype:
                    raise ValueError(f"{path} feature dtype does not match manifest")
            if self.manifest["selection"] == "uniform":
                expected_indices = uniform_context_frame_indices(
                    int(self.entries[key]["demo_length"]),
                    int(self.manifest["context_max_timesteps"]),
                )
                if frame_indices.tolist() != expected_indices:
                    raise ValueError(f"{path} does not match uniform context frame indices")
            self._payload_cache[key] = {
                "features": features,
                "frame_indices": frame_indices.to(dtype=torch.long),
            }
            while len(self._payload_cache) > self.max_cached_payloads:
                self._payload_cache.popitem(last=False)
        else:
            self._payload_cache.move_to_end(key)
        return self._payload_cache[key]


def _resolve_torch_dtype(name: str) -> torch.dtype:
    normalized = canonical_feature_dtype_name(name)
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    raise AssertionError(f"Unhandled canonical feature dtype: {normalized}")


def _to_processor_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max(initial=0.0) <= 2.0:
            arr = arr * 255.0
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _load_lerobot_video_frames(
    demo: dict[str, Any],
    *,
    obs_key: str,
    frame_indices: list[int],
) -> dict[int, np.ndarray]:
    source_videos = demo.get("source_videos", {})
    if obs_key not in source_videos:
        available = ", ".join(sorted(source_videos))
        raise KeyError(
            f"LeRobot observation key {obs_key!r} is unavailable; choose one of: {available}"
        )
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "LeRobot context feature preprocessing requires PyAV. "
            "Re-run scripts/vfe/setup_vfe_train.sh."
        ) from exc

    requested = set(int(index) for index in frame_indices)
    if not requested:
        return {}
    video_path = str(source_videos[obs_key])
    decoded: dict[int, np.ndarray] = {}
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        final_index = max(requested)
        for decoded_index, frame in enumerate(container.decode(stream)):
            if decoded_index in requested:
                decoded[decoded_index] = frame.to_ndarray(format="rgb24")
            if decoded_index >= final_index:
                break
    missing = sorted(requested.difference(decoded))
    if missing:
        preview = ", ".join(str(index) for index in missing[:8])
        suffix = f" and {len(missing) - 8} more" if len(missing) > 8 else ""
        raise IndexError(
            f"Could not decode requested frame indices {preview}{suffix} from {video_path}"
        )
    return decoded


def _load_context_processor_images(
    demo: dict[str, Any],
    *,
    demo_id: str,
    obs_keys: list[str],
    frame_indices: list[int],
) -> list[np.ndarray]:
    if demo.get("source_format") == "lerobot_v21":
        frames_by_key = {
            obs_key: _load_lerobot_video_frames(
                demo,
                obs_key=obs_key,
                frame_indices=frame_indices,
            )
            for obs_key in obs_keys
        }
        return [
            _to_processor_image(frames_by_key[obs_key][frame_idx])
            for frame_idx in frame_indices
            for obs_key in obs_keys
        ]

    with h5py.File(demo["source_hdf5"], "r") as h5_file:
        return [
            _to_processor_image(h5_file[f"data/{demo_id}/obs/{obs_key}"][frame_idx])
            for frame_idx in frame_indices
            for obs_key in obs_keys
        ]


class Gemma3NativeFeatureExtractor:
    """Frozen Gemma 3 vision tower plus parameter-free 256-token pooling."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Native feature preprocessing requires transformers") from exc

        pretrained_kwargs: dict[str, Any] = {}
        if revision:
            pretrained_kwargs["revision"] = revision
        processor = AutoProcessor.from_pretrained(model_name, **pretrained_kwargs)
        self.image_processor = getattr(processor, "image_processor", processor)
        native_model = AutoModel.from_pretrained(model_name, torch_dtype=model_dtype, **pretrained_kwargs)
        vision_tower = getattr(native_model, "vision_tower", None)
        projector = getattr(native_model, "multi_modal_projector", None)
        if vision_tower is None or projector is None:
            raise ValueError(f"{model_name!r} does not expose Gemma 3 native vision modules")
        native_config = native_model.config
        self.resolved_revision = getattr(native_config, "_commit_hash", None) or revision
        self.tokens_per_image = int(getattr(native_config, "mm_tokens_per_image", 256))
        vision_config = getattr(native_config, "vision_config", None)
        if vision_config is None:
            raise ValueError("Gemma 3 native model config does not expose vision_config")
        self.vision_tower = vision_tower.to(device).eval()
        self.projector = projector.to(device).eval()
        self.patches_per_image = int(vision_config.image_size) // int(vision_config.patch_size)
        tokens_per_side = math.isqrt(self.tokens_per_image)
        if tokens_per_side * tokens_per_side != self.tokens_per_image:
            raise ValueError("Gemma 3 mm_tokens_per_image must be a perfect square")
        if self.patches_per_image % tokens_per_side != 0:
            raise ValueError("Gemma 3 vision patch grid must be divisible by the pooled token grid")
        pool_kernel = self.patches_per_image // tokens_per_side
        self.pooler = torch.nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel)
        for module in (self.vision_tower, self.projector):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.device = device
        self.vision_hidden_size = int(vision_config.hidden_size)
        self.projected_hidden_size = int(projector.mm_input_projection_weight.shape[1])
        del native_model
        gc.collect()

    @torch.inference_mode()
    def encode(self, images: list[np.ndarray], *, feature_stage: str) -> torch.Tensor:
        processed = self.image_processor(
            images=images,
            return_tensors="pt",
            do_pan_and_scan=False,
        )
        pixel_values = processed["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = torch.stack(pixel_values, dim=0)
        pixel_values = pixel_values.to(self.device)
        outputs = self.vision_tower(pixel_values=pixel_values, return_dict=True)
        vision_outputs = outputs.last_hidden_state
        batch_size, _, hidden_size = vision_outputs.shape
        reshaped = vision_outputs.transpose(1, 2).reshape(
            batch_size,
            hidden_size,
            self.patches_per_image,
            self.patches_per_image,
        )
        pooled = self.pooler(reshaped.contiguous()).flatten(2).transpose(1, 2)
        if feature_stage == NATIVE_VISION_POOLED_STAGE:
            features = pooled
        elif feature_stage == NATIVE_PROJECTOR_INPUT_STAGE:
            features = self.projector.mm_soft_emb_norm(pooled)
        elif feature_stage == NATIVE_PROJECTOR_OUTPUT_STAGE:
            features = self.projector(vision_outputs)
        else:
            raise ValueError(f"Unsupported feature stage: {feature_stage}")
        if features.dim() != 3 or features.shape[1] != self.tokens_per_image:
            raise RuntimeError(
                f"Native vision pooler returned shape {tuple(features.shape)}; "
                f"expected [images, {self.tokens_per_image}, hidden]"
            )
        return features


class HfVisionPatchFeatureExtractor:
    """Frozen HF vision tower returning its full spatial patch sequence."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("HF vision feature preprocessing requires transformers") from exc

        pretrained_kwargs: dict[str, Any] = {}
        if revision:
            pretrained_kwargs["revision"] = revision
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            backend="torchvision",
            **pretrained_kwargs,
        )
        model = AutoModel.from_pretrained(model_name, torch_dtype=model_dtype, **pretrained_kwargs)
        self.vision_tower = getattr(model, "vision_model", model).to(device).eval()
        config = getattr(model.config, "vision_config", model.config)
        self.resolved_revision = getattr(model.config, "_commit_hash", None) or revision
        image_size = int(config.image_size)
        patch_size = int(config.patch_size)
        self.tokens_per_image = (image_size // patch_size) ** 2
        self.vision_hidden_size = int(config.hidden_size)
        self.projected_hidden_size = self.vision_hidden_size
        for parameter in self.vision_tower.parameters():
            parameter.requires_grad_(False)
        self.device = device
        del model
        gc.collect()

    @torch.inference_mode()
    def encode(self, images: list[np.ndarray], *, feature_stage: str) -> torch.Tensor:
        if feature_stage != RECAP_VISION_PATCH_STAGE:
            raise ValueError(f"HF patch extractor does not support feature stage {feature_stage!r}")
        image_tensor = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).to(self.device)
        processed = self.image_processor(
            images=image_tensor,
            return_tensors="pt",
            device=self.device,
        )
        pixel_values = processed["pixel_values"].to(self.device)
        outputs = self.vision_tower(pixel_values=pixel_values, return_dict=True)
        features = outputs.last_hidden_state
        if features.dim() != 3 or features.shape[1:] != (
            self.tokens_per_image,
            self.vision_hidden_size,
        ):
            raise RuntimeError(
                f"HF vision tower returned shape {tuple(features.shape)}; expected "
                f"[images, {self.tokens_per_image}, {self.vision_hidden_size}]"
            )
        return features


class RecapQwen2DetailFeatureExtractor:
    """Apply a checkpointed, frozen 2x2 merger to cached SigLIP patch tokens."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        state_prefix: str,
        device: torch.device,
        source_tokens_per_image: int,
        source_feature_dim: int,
        resolved_revision: str | None,
    ) -> None:
        from .model import Qwen2StyleSpatialPatchMerger

        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RECAP detail-merger checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        model_state = checkpoint.get("model_state")
        if not isinstance(model_state, dict):
            raise KeyError(f"Checkpoint {path} has no model_state dictionary")
        prefix = str(state_prefix).rstrip(".") + "."
        merger_state = {
            key[len(prefix) :]: value
            for key, value in model_state.items()
            if key.startswith(prefix)
        }
        required = {
            "norm.weight",
            "norm.bias",
            "mlp.0.weight",
            "mlp.0.bias",
            "mlp.2.weight",
            "mlp.2.bias",
        }
        if set(merger_state) != required:
            raise KeyError(
                f"Checkpoint {path} merger keys for {prefix!r} are "
                f"{sorted(merger_state)}; expected {sorted(required)}"
            )
        context_dim = int(merger_state["norm.weight"].numel())
        output_dim = int(merger_state["mlp.2.weight"].shape[0])
        if context_dim != int(source_feature_dim):
            raise ValueError(
                f"Merger expects hidden={context_dim}, but source cache has hidden={source_feature_dim}"
            )
        source_grid = math.isqrt(int(source_tokens_per_image))
        if source_grid * source_grid != int(source_tokens_per_image) or source_grid % 2:
            raise ValueError(
                f"Source token count {source_tokens_per_image} is not an even square grid"
            )
        self.merger = Qwen2StyleSpatialPatchMerger(
            context_dim=context_dim,
            output_dim=output_dim,
            spatial_merge_size=2,
        )
        self.merger.load_state_dict(merger_state, strict=True)
        self.merger.to(device).eval()
        for parameter in self.merger.parameters():
            parameter.requires_grad_(False)
        digest = hashlib.sha256()
        for key in sorted(merger_state):
            tensor = merger_state[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        self.projector_sha256 = digest.hexdigest()
        self.projector_checkpoint = str(path)
        self.projector_state_prefix = str(state_prefix).rstrip(".")
        self.source_tokens_per_image = int(source_tokens_per_image)
        self.tokens_per_image = (source_grid // 2) ** 2
        self.vision_hidden_size = context_dim
        self.projected_hidden_size = output_dim
        self.resolved_revision = resolved_revision
        self.device = device
        del checkpoint, model_state, merger_state
        gc.collect()

    @torch.inference_mode()
    def encode(self, patch_features: torch.Tensor, *, feature_stage: str) -> torch.Tensor:
        if feature_stage != RECAP_QWEN2_DETAIL_STAGE:
            raise ValueError(f"RECAP detail extractor does not support stage {feature_stage!r}")
        if patch_features.dim() != 3 or tuple(patch_features.shape[1:]) != (
            self.source_tokens_per_image,
            self.vision_hidden_size,
        ):
            raise ValueError(
                f"Source patch features have shape {tuple(patch_features.shape)}; expected "
                f"[images, {self.source_tokens_per_image}, {self.vision_hidden_size}]"
            )
        return self.merger(patch_features.to(device=self.device, dtype=torch.float32))


def _context_demo_records(
    *,
    processed_dir: Path,
    preset_name: str,
    phases: list[str],
    preset_path: str | Path,
) -> list[tuple[int, str, dict[str, Any]]]:
    summary = read_json(processed_dir / "summary.json")
    tasks = {
        int(stub["task_id"]): read_json(processed_dir / stub["metadata_path"])
        for stub in summary["tasks"]
    }
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    for phase in phases:
        split = resolve_data_split(preset_name=preset_name, phase=phase, preset_path=preset_path)
        context_by_task = split.get("context_demo_ids_by_task")
        if not context_by_task:
            raise ValueError(f"Data split preset {preset_name!r} phase {phase!r} has no context demos")
        for task_id, demo_ids in context_by_task.items():
            demos = {demo["demo_id"]: demo for demo in tasks[int(task_id)]["demos"]}
            for demo_id in demo_ids:
                if demo_id not in demos:
                    raise KeyError(f"Task {task_id} metadata has no context demo {demo_id}")
                selected[(int(task_id), str(demo_id))] = demos[demo_id]
    return [(task_id, demo_id, selected[(task_id, demo_id)]) for task_id, demo_id in sorted(selected)]


def apply_context_feature_config(args: argparse.Namespace) -> argparse.Namespace:
    preprocess = load_preset_preprocess_config(args.data_split_preset, args.data_split_config)
    vfe = load_vfe_preset(args.vfe_preset, args.vfe_config)
    model = vfe["model"]
    features = vfe["context_features"]
    if not bool(model["context_enabled"]):
        raise ValueError(f"VFE preset {args.vfe_preset!r} does not enable context")
    if not bool(features["enabled"]):
        raise ValueError(f"VFE preset {args.vfe_preset!r} does not enable context feature caching")
    args.processed_dir = str(
        resolve_processed_dir(preprocess["output_dir"], preprocess["ground_truth_version"])
    )
    args.output_dir = features["output_dir"]
    args.data_split_phases = features["phases"]
    args.context_obs_key = model["context_obs_key"]
    args.context_stride = model["context_stride"]
    args.context_max_original_timesteps = model["context_max_original_timesteps"]
    args.context_sampling_mode = model["context_sampling_mode"]
    args.context_max_timesteps = model["context_max_timesteps"]
    args.feature_stage = features["stage"]
    args.feature_model_name = features.get("model_name") or model["native_vision_model_name"]
    args.feature_model_revision = features.get("model_revision", model["native_vision_model_revision"])
    args.expected_tokens_per_image = int(features.get("tokens_per_image", 256))
    args.model_dtype = features["model_dtype"]
    args.feature_dtype = features["feature_dtype"]
    args.image_batch_size = features["image_batch_size"]
    args.device = features["device"]
    args.overwrite = features["overwrite"]
    args.source_feature_dir = features.get("source_feature_dir")
    args.projector_checkpoint = features.get("projector_checkpoint")
    args.projector_state_prefix = features.get("projector_state_prefix", "impl.visual_projector")
    args.resolved_vfe_preset = vfe
    return args


def precompute_context_native_features(args: argparse.Namespace) -> dict[str, Any]:
    args = apply_context_feature_config(args)
    processed_dir = Path(args.processed_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_values = (
        args.data_split_phases.split(",")
        if isinstance(args.data_split_phases, str)
        else args.data_split_phases
    )
    phases = [str(phase).strip() for phase in phase_values if str(phase).strip()]
    if not phases:
        raise ValueError(
            f"VFE preset {args.vfe_preset!r} must define at least one context feature phase"
        )
    obs_keys = [key.strip() for key in args.context_obs_key.split(",") if key.strip()]
    if not obs_keys:
        raise ValueError(
            f"VFE preset {args.vfe_preset!r} must define at least one context_obs_key"
        )
    if args.image_batch_size <= 0:
        raise ValueError(
            f"VFE preset {args.vfe_preset!r} context feature image_batch_size must be positive"
        )
    feature_stage = str(args.feature_stage)
    model_dtype = _resolve_torch_dtype(args.model_dtype)
    feature_dtype = _resolve_torch_dtype(args.feature_dtype)

    existing_manifest: dict[str, Any] | None = None
    manifest_path = output_dir / CONTEXT_NATIVE_FEATURE_MANIFEST
    if manifest_path.exists() and not args.overwrite:
        existing_manifest = load_context_feature_manifest(output_dir)
        validate_context_feature_manifest(
            existing_manifest,
            feature_model_name=args.feature_model_name,
            feature_model_revision=args.feature_model_revision,
            obs_keys=obs_keys,
            context_stride=args.context_stride,
            context_max_original_timesteps=args.context_max_original_timesteps,
            context_sampling_mode=args.context_sampling_mode,
            context_max_timesteps=args.context_max_timesteps,
            expected_tokens_per_image=args.expected_tokens_per_image,
        )
        expected_existing = {
            "feature_stage": feature_stage,
            "feature_dtype": str(feature_dtype).removeprefix("torch."),
            "model_dtype": str(model_dtype).removeprefix("torch."),
            "data_split_preset": args.data_split_preset,
            "vfe_preset": args.vfe_preset,
        }
        differences = [
            f"{key}: cache={existing_manifest.get(key)!r}, requested={value!r}"
            for key, value in expected_existing.items()
            if existing_manifest.get(key) != value
        ]
        if differences:
            raise ValueError("Existing context feature cache is incompatible: " + "; ".join(differences))
    elif not args.overwrite and next(output_dir.glob("task_*/*.pt"), None) is not None:
        raise FileExistsError(
            f"{output_dir} contains feature shards but no compatible manifest; pass --overwrite or use a new directory"
        )

    records = _context_demo_records(
        processed_dir=processed_dir,
        preset_name=args.data_split_preset,
        phases=phases,
        preset_path=args.data_split_config,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    source_store: ContextNativeFeatureStore | None = None
    if feature_stage == RECAP_QWEN2_DETAIL_STAGE:
        if not args.source_feature_dir or not args.projector_checkpoint:
            raise ValueError(
                "recap_qwen2_detail caching requires source_feature_dir and projector_checkpoint"
            )
        source_store = ContextNativeFeatureStore(
            args.source_feature_dir,
            feature_model_name=args.feature_model_name,
            feature_model_revision=args.feature_model_revision,
            obs_keys=obs_keys,
            context_stride=args.context_stride,
            context_max_original_timesteps=args.context_max_original_timesteps,
            context_sampling_mode=args.context_sampling_mode,
            context_max_timesteps=args.context_max_timesteps,
            expected_tokens_per_image=1024,
            expected_feature_stage=RECAP_VISION_PATCH_STAGE,
            expected_feature_dtype="bf16",
            max_cached_payloads=1,
        )
        source_store.require_entries(
            {
                task_id: [demo_id for record_task_id, demo_id, _ in records if record_task_id == task_id]
                for task_id in sorted({record_task_id for record_task_id, _, _ in records})
            }
        )
        extractor = RecapQwen2DetailFeatureExtractor(
            args.projector_checkpoint,
            state_prefix=args.projector_state_prefix,
            device=device,
            source_tokens_per_image=source_store.tokens_per_image,
            source_feature_dim=source_store.feature_dim,
            resolved_revision=source_store.manifest.get("resolved_feature_model_revision"),
        )
    else:
        extractor_cls = (
            HfVisionPatchFeatureExtractor
            if feature_stage == RECAP_VISION_PATCH_STAGE
            else Gemma3NativeFeatureExtractor
        )
        extractor = extractor_cls(
            args.feature_model_name,
            revision=args.feature_model_revision,
            device=device,
            model_dtype=model_dtype,
        )
    if int(extractor.tokens_per_image) != args.expected_tokens_per_image:
        raise ValueError(
            f"Feature model produces {extractor.tokens_per_image} tokens/image, but the preset expects "
            f"{args.expected_tokens_per_image}"
        )

    entries: dict[str, dict[str, Any]] = (
        dict(existing_manifest.get("entries", {})) if existing_manifest is not None else {}
    )
    expected_feature_dim = (
        extractor.projected_hidden_size
        if feature_stage in {NATIVE_PROJECTOR_OUTPUT_STAGE, RECAP_QWEN2_DETAIL_STAGE}
        else extractor.vision_hidden_size
    )
    feature_dim = int(existing_manifest["feature_dim"]) if existing_manifest is not None else expected_feature_dim
    if feature_dim != expected_feature_dim:
        raise ValueError(
            f"Existing cache feature_dim={feature_dim} does not match model/stage dimension {expected_feature_dim}"
        )
    if (
        existing_manifest is not None
        and existing_manifest.get(
            "resolved_feature_model_revision",
            existing_manifest.get("resolved_native_vision_revision"),
        )
        != extractor.resolved_revision
    ):
        raise ValueError(
            "Existing cache resolved model revision does not match the currently loaded model: "
            "cache="
            f"{existing_manifest.get('resolved_feature_model_revision', existing_manifest.get('resolved_native_vision_revision'))!r}, "
            f"loaded={extractor.resolved_revision!r}"
        )
    if existing_manifest is not None and feature_stage == RECAP_QWEN2_DETAIL_STAGE:
        expected_transform = {
            "source_feature_dir": str(Path(args.source_feature_dir).expanduser().resolve()),
            "source_feature_stage": RECAP_VISION_PATCH_STAGE,
            "projector_checkpoint": extractor.projector_checkpoint,
            "projector_state_prefix": extractor.projector_state_prefix,
            "projector_sha256": extractor.projector_sha256,
        }
        differences = [
            f"{key}: cache={existing_manifest.get(key)!r}, requested={value!r}"
            for key, value in expected_transform.items()
            if existing_manifest.get(key) != value
        ]
        if differences:
            raise ValueError(
                "Existing RECAP detail cache uses a different transform: " + "; ".join(differences)
            )

    all_phases = sorted(
        set(phases).union(existing_manifest.get("data_split_phases", []) if existing_manifest else [])
    )

    def write_manifest_snapshot() -> dict[str, Any]:
        total_slots = sum(int(entry["num_slots"]) for entry in entries.values())
        total_images = sum(int(entry["num_slots"]) * int(entry["num_views"]) for entry in entries.values())
        payload = {
            "format_version": CONTEXT_NATIVE_FEATURE_FORMAT_VERSION,
            "feature_type": _feature_type_for_stage(feature_stage),
            "feature_stage": feature_stage,
            "processed_dir": str(processed_dir),
            "data_split_config": str(Path(args.data_split_config).expanduser()),
            "data_split_preset": args.data_split_preset,
            "vfe_config": str(Path(args.vfe_config).expanduser()),
            "vfe_preset": args.vfe_preset,
            "resolved_vfe_preset": args.resolved_vfe_preset,
            "data_split_phases": all_phases,
            "feature_model_name": args.feature_model_name,
            "feature_model_revision": args.feature_model_revision,
            "resolved_feature_model_revision": extractor.resolved_revision,
            "obs_keys": obs_keys,
            "context_stride": int(args.context_stride),
            "context_max_original_timesteps": int(args.context_max_original_timesteps),
            "selection": args.context_sampling_mode,
            "context_max_timesteps": int(args.context_max_timesteps),
            "do_pan_and_scan": False,
            "feature_dtype": str(feature_dtype).removeprefix("torch."),
            "model_dtype": str(model_dtype).removeprefix("torch."),
            "tokens_per_image": int(extractor.tokens_per_image),
            "feature_dim": int(feature_dim),
            "num_demos": len(entries),
            "num_slots": total_slots,
            "num_images": total_images,
            "entries": entries,
        }
        if args.context_sampling_mode == "uniform":
            payload["uniform_sampling"] = "linspace_rint_v1"
        if feature_stage == RECAP_QWEN2_DETAIL_STAGE:
            payload.update(
                {
                    "source_feature_dir": str(Path(args.source_feature_dir).expanduser().resolve()),
                    "source_feature_stage": RECAP_VISION_PATCH_STAGE,
                    "projector_checkpoint": extractor.projector_checkpoint,
                    "projector_state_prefix": extractor.projector_state_prefix,
                    "projector_sha256": extractor.projector_sha256,
                }
            )
        if feature_stage not in {RECAP_VISION_PATCH_STAGE, RECAP_QWEN2_DETAIL_STAGE}:
            # Preserve the original manifest keys for existing tooling while
            # the generic feature-model keys support non-Gemma vision towers.
            payload.update(
                {
                    "native_vision_model_name": args.feature_model_name,
                    "native_vision_model_revision": args.feature_model_revision,
                    "resolved_native_vision_revision": extractor.resolved_revision,
                }
            )
        manifest_path = output_dir / CONTEXT_NATIVE_FEATURE_MANIFEST
        temp_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        write_json(temp_manifest_path, payload)
        temp_manifest_path.replace(manifest_path)
        return payload

    # Establish cache provenance before the first potentially long GPU encode.
    write_manifest_snapshot()
    for record_index, (task_id, demo_id, demo) in enumerate(records, start=1):
        if args.context_sampling_mode == "uniform":
            frame_indices = uniform_context_frame_indices(
                int(demo["demo_length"]), args.context_max_timesteps
            )
        else:
            frame_indices = end_anchored_context_frame_indices(
                int(demo["demo_length"]),
                stride=args.context_stride,
                max_original_timesteps=args.context_max_original_timesteps,
                max_sampled_timesteps=args.context_max_timesteps,
            )
        relative_path = context_feature_relative_path(task_id, demo_id)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not args.overwrite:
            payload = _load_torch_shard(output_path, mmap=True)
            features = payload["features"]
            saved_indices = payload["frame_indices"].tolist()
            if saved_indices != frame_indices:
                raise ValueError(f"Existing shard {output_path} uses different frame indices")
            expected_shape = (len(frame_indices), len(obs_keys), extractor.tokens_per_image, feature_dim)
            if tuple(features.shape) != expected_shape or features.dtype != feature_dtype:
                raise ValueError(f"Existing shard {output_path} has incompatible shape or dtype")
        else:
            encoded_chunks = []
            if source_store is not None:
                source_payload = source_store.load(task_id, demo_id)
                source_indices = source_payload["frame_indices"].tolist()
                if source_indices != frame_indices:
                    raise ValueError(
                        f"Source cache indices for {context_feature_key(task_id, demo_id)} do not match"
                    )
                source_features = source_payload["features"].reshape(
                    -1,
                    source_store.tokens_per_image,
                    source_store.feature_dim,
                )
                for start in range(0, source_features.shape[0], args.image_batch_size):
                    encoded_chunks.append(
                        extractor.encode(
                            source_features[start : start + args.image_batch_size],
                            feature_stage=feature_stage,
                        ).cpu()
                    )
            else:
                flat_images = _load_context_processor_images(
                    demo,
                    demo_id=demo_id,
                    obs_keys=obs_keys,
                    frame_indices=frame_indices,
                )
                for start in range(0, len(flat_images), args.image_batch_size):
                    encoded_chunks.append(
                        extractor.encode(
                            flat_images[start : start + args.image_batch_size],
                            feature_stage=feature_stage,
                        ).cpu()
                    )
            flat_features = torch.cat(encoded_chunks, dim=0).to(dtype=feature_dtype)
            features = flat_features.reshape(
                len(frame_indices),
                len(obs_keys),
                extractor.tokens_per_image,
                flat_features.shape[-1],
            ).contiguous()
            temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            torch.save(
                {
                    "features": features,
                    "frame_indices": torch.tensor(frame_indices, dtype=torch.int32),
                },
                temp_path,
            )
            temp_path.replace(output_path)

        current_feature_dim = int(features.shape[-1])
        if feature_dim != current_feature_dim:
            raise ValueError(f"Feature dimension changed from {feature_dim} to {current_feature_dim}")
        key = context_feature_key(task_id, demo_id)
        entries[key] = {
            "task_id": task_id,
            "demo_id": demo_id,
            "path": str(relative_path),
            "demo_length": int(demo["demo_length"]),
            "num_slots": len(frame_indices),
            "num_views": len(obs_keys),
            "frame_indices": frame_indices,
        }
        write_manifest_snapshot()
        print(
            f"[{record_index}/{len(records)}] {key}: slots={len(frame_indices)} "
            f"shape={tuple(features.shape)}",
            flush=True,
        )

    manifest = write_manifest_snapshot()
    print(f"context native-feature manifest: {output_dir / CONTEXT_NATIVE_FEATURE_MANIFEST}", flush=True)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute frozen Gemma 3 native features for context demos.")
    parser.add_argument("--data-split-config", default=str(DEFAULT_PRESET_PATH))
    parser.add_argument("--data-split-preset", required=True)
    parser.add_argument("--vfe-config", default=str(DEFAULT_VFE_PRESET_PATH))
    parser.add_argument("--vfe-preset", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    precompute_context_native_features(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
