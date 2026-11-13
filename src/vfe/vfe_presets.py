from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .observation_history import normalize_query_history_offsets


DEFAULT_VFE_PRESET_PATH = Path("configs/vfe/vfe_presets.json")
NO_CONTEXT_BACKBONE_MAP = {
    "siglip_gemma_context": "siglip_gemma",
    "gemma3_native_vision_context": "gemma3_native_vision",
    "gemma3_native_style_270m_context": "gemma3_native_style_270m",
}


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_vfe_presets(path: str | Path = DEFAULT_VFE_PRESET_PATH) -> dict[str, Any]:
    preset_path = Path(path).expanduser()
    if not preset_path.exists():
        raise FileNotFoundError(f"VFE preset file not found: {preset_path}")
    payload = json.loads(preset_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("defaults"), dict) or not isinstance(payload.get("presets"), dict):
        raise KeyError(f"{preset_path} must contain defaults and presets objects")
    return payload


def load_vfe_preset(
    preset_name: str,
    path: str | Path = DEFAULT_VFE_PRESET_PATH,
) -> dict[str, Any]:
    payload = load_vfe_presets(path)
    presets = payload["presets"]
    if preset_name not in presets:
        available = ", ".join(sorted(presets))
        raise KeyError(f"Unknown VFE preset {preset_name!r}. Available: {available}")
    resolved = _merge_dict(payload["defaults"], presets[preset_name])
    model = resolved.get("model")
    features = resolved.get("context_features")
    if not isinstance(model, dict) or not isinstance(features, dict):
        raise KeyError(f"VFE preset {preset_name!r} must resolve model and context_features objects")
    model["query_history_offsets"] = normalize_query_history_offsets(
        model.get("query_history_offsets", [0])
    )
    if int(model["context_stride"]) <= 0 or int(model["context_max_original_timesteps"]) <= 0:
        raise ValueError(f"VFE preset {preset_name!r} context stride and source window must be positive")
    model.setdefault("context_sampling_mode", "end_anchored")
    if model["context_sampling_mode"] == "uniform":
        count = model.get("context_max_timesteps")
        if type(count) is not int or count <= 0:
            raise ValueError("Uniform context requires a positive integer context_max_timesteps")
    elif model["context_sampling_mode"] == "end_anchored":
        window, stride = int(model["context_max_original_timesteps"]), int(model["context_stride"])
        model["context_max_timesteps"] = (window + stride - 1) // stride
    else:
        raise ValueError(f"Unknown context sampling mode: {model['context_sampling_mode']}")
    if bool(features["enabled"]):
        if not bool(model["context_enabled"]):
            raise ValueError(f"VFE preset {preset_name!r} enables context features without context")
        if not features["output_dir"] or not features["phases"]:
            raise ValueError(
                f"VFE preset {preset_name!r} enabled context features require output_dir and phases"
            )
        if int(features.get("tokens_per_image", 256)) <= 0:
            raise ValueError(
                f"VFE preset {preset_name!r} context feature tokens_per_image must be positive"
            )
        if features.get("stage") in {"recap_vision_patches", "recap_qwen2_detail"}:
            if model.get("native_vision_projector_mode") != "recap_style_qwen2_512_dualres":
                raise ValueError(
                    f"VFE preset {preset_name!r} uses RECAP context patches without the matching model mode"
                )
            if features.get("model_name") != model.get("vision_model_name"):
                raise ValueError(
                    f"VFE preset {preset_name!r} context feature model must match query vision_model_name"
                )
        if features.get("stage") == "recap_vision_patches":
            if int(features.get("tokens_per_image", 0)) != int(model.get("query_patch_tokens_per_image", 0)):
                raise ValueError(
                    f"VFE preset {preset_name!r} context/query patch token counts must match"
                )
        if features.get("stage") == "recap_qwen2_detail":
            if int(features.get("tokens_per_image", 0)) != 256:
                raise ValueError(
                    f"VFE preset {preset_name!r} RECAP detail cache must contain 256 tokens/image"
                )
            if not bool(model.get("freeze_recap_detail_merger", False)):
                raise ValueError(
                    f"VFE preset {preset_name!r} must freeze the merger used to create its detail cache"
                )
            if not features.get("source_feature_dir") or not features.get("projector_checkpoint"):
                raise ValueError(
                    f"VFE preset {preset_name!r} detail cache requires source_feature_dir and projector_checkpoint"
                )
    if bool(model.get("context_progress_enabled", False)) and not bool(model["context_enabled"]):
        raise ValueError(
            f"VFE preset {preset_name!r} enables context progress without context"
        )
    if model.get("loss_type", "hard_ce") not in {"hard_ce", "gaussian_soft_ce_huber"}:
        raise ValueError(
            f"VFE preset {preset_name!r} has unsupported loss_type={model.get('loss_type')!r}"
        )
    if (
        float(model.get("soft_label_sigma", 0.01)) <= 0
        or float(model.get("huber_beta", 0.02)) <= 0
        or float(model.get("huber_weight", 1.0)) < 0
    ):
        raise ValueError(f"VFE preset {preset_name!r} has invalid value-loss parameters")
    resolved["preset_name"] = preset_name
    return resolved


def resolve_vfe_backbone(model_config: dict[str, Any]) -> str:
    backbone = str(model_config["backbone"])
    if bool(model_config["context_enabled"]):
        return backbone
    return NO_CONTEXT_BACKBONE_MAP.get(backbone, backbone)
