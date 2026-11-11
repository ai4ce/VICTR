from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .data import (
    derive_context_max_timesteps,
    make_train_dataloader,
)
from .data_splits import load_preset_preprocess_config, resolve_data_split, resolve_processed_dir
from .io import read_json
from .model import (
    CONTEXT_BACKBONES,
    CUSTOM_CONTEXT_ATTENTION_MODES,
    GEMMA3_NATIVE_STYLE_270M_BACKBONE,
    GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE,
    GEMMA3_NATIVE_VISION_BACKBONE,
    GEMMA3_NATIVE_VISION_CONTEXT_BACKBONE,
    HF_IMAGE_BACKBONES,
    SIGLIP_GEMMA_BACKBONE,
    SIGLIP_GEMMA_CONTEXT_BACKBONE,
    ValueFunctionEstimator,
    VfeModelConfig,
)
from .vfe_presets import DEFAULT_VFE_PRESET_PATH, load_vfe_preset


WANDB_PROJECT = "vfe-libero100"
OUTPUT_ROOT = Path("outputs")
HARD_CE_LOSS = "hard_ce"
GAUSSIAN_SOFT_CE_HUBER_LOSS = "gaussian_soft_ce_huber"
SUPPORTED_VALUE_LOSSES = {HARD_CE_LOSS, GAUSSIAN_SOFT_CE_HUBER_LOSS}


def discretized_gaussian_targets(
    values: torch.Tensor,
    bin_values: torch.Tensor,
    *,
    sigma: float,
) -> torch.Tensor:
    """Project bounded scalar targets onto bins with Gaussian CDF masses.

    The first and last bins absorb the Gaussian tails outside the value range,
    avoiding the endpoint bias introduced by truncating and renormalizing a
    Gaussian evaluated only at bin centers.
    """

    if values.dim() != 1:
        raise ValueError(f"Gaussian soft targets require [batch] values, got {tuple(values.shape)}")
    if bin_values.dim() != 1 or bin_values.numel() < 2:
        raise ValueError("Gaussian soft targets require at least two one-dimensional bin values")
    if sigma <= 0:
        raise ValueError(f"Gaussian soft-label sigma must be positive, got {sigma}")

    centers = bin_values.to(device=values.device, dtype=torch.float32)
    if not bool(torch.all(centers[1:] > centers[:-1]).item()):
        raise ValueError("Gaussian soft-label bin values must be strictly increasing")
    midpoints = (centers[:-1] + centers[1:]) * 0.5
    edges = torch.cat(
        (
            torch.full((1,), -torch.inf, device=values.device),
            midpoints,
            torch.full((1,), torch.inf, device=values.device),
        )
    )
    standardized_edges = (edges.unsqueeze(0) - values.float().unsqueeze(1)) / float(sigma)
    cdf = torch.special.ndtr(standardized_edges)
    targets = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0)
    return targets / targets.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(targets.dtype).tiny)


def value_loss_components(
    logits: torch.Tensor,
    *,
    value_bins: torch.Tensor,
    value_continuous: torch.Tensor,
    bin_values: torch.Tensor,
    loss_type: str,
    soft_label_sigma: float,
    huber_beta: float,
    huber_weight: float,
) -> dict[str, torch.Tensor]:
    """Return the active value loss and comparable diagnostic components."""

    if loss_type not in SUPPORTED_VALUE_LOSSES:
        raise ValueError(f"Unsupported value loss {loss_type!r}; use one of {sorted(SUPPORTED_VALUE_LOSSES)}")
    if huber_beta <= 0:
        raise ValueError(f"Huber beta must be positive, got {huber_beta}")
    if huber_weight < 0:
        raise ValueError(f"Huber weight must be non-negative, got {huber_weight}")

    centers = bin_values.to(device=logits.device, dtype=torch.float32)
    # Keep the hard-CE path and expected-value diagnostic numerically identical
    # to the pre-ablation trainer.  The float32 log-softmax below is needed only
    # by the new soft-label objective.
    predicted_value = (F.softmax(logits, dim=-1) * centers).sum(dim=-1)
    hard_ce = F.cross_entropy(logits, value_bins)
    huber = F.smooth_l1_loss(
        predicted_value,
        value_continuous.float(),
        beta=float(huber_beta),
    )

    if loss_type == HARD_CE_LOSS:
        soft_ce = hard_ce
        loss = hard_ce
    else:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        soft_targets = discretized_gaussian_targets(
            value_continuous,
            centers,
            sigma=soft_label_sigma,
        )
        soft_ce = -(soft_targets * log_probs).sum(dim=-1).mean()
        loss = soft_ce + float(huber_weight) * huber

    return {
        "loss": loss,
        "hard_ce": hard_ce,
        "soft_ce": soft_ce,
        "huber": huber,
        "predicted_value": predicted_value,
    }


@dataclass(frozen=True)
class DistributedInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def output_dir_from_exp_name(exp_name: str) -> Path:
    exp_name = exp_name.strip()
    if not exp_name:
        raise ValueError("--exp-name must not be empty")
    if Path(exp_name).name != exp_name:
        raise ValueError("--exp-name must be a single directory name, not a path")
    return (OUTPUT_ROOT / exp_name).expanduser().resolve()


def initialize_distributed(args: argparse.Namespace) -> DistributedInfo:
    """Initialize torchrun/DDP when WORLD_SIZE is greater than one."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank if args.local_rank >= 0 else 0)))
    if world_size <= 1:
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        return DistributedInfo(rank=0, local_rank=0, world_size=1, device=device)

    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid distributed rank {rank} for WORLD_SIZE={world_size}")
    if torch.cuda.is_available():
        if args.device and not str(args.device).startswith("cuda"):
            raise ValueError("DDP with CUDA workers requires --device cuda (or no --device override)")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return DistributedInfo(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def distributed_barrier(info: DistributedInfo) -> None:
    if info.enabled:
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> ValueFunctionEstimator:
    if isinstance(model, DistributedDataParallel):
        return model.module
    if not isinstance(model, ValueFunctionEstimator):
        raise TypeError(f"Expected ValueFunctionEstimator, got {type(model)!r}")
    return model


def reduce_sum(value: torch.Tensor, info: DistributedInfo) -> torch.Tensor:
    result = value.detach().clone()
    if info.enabled:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def reduce_max(value: torch.Tensor, info: DistributedInfo) -> torch.Tensor:
    result = value.detach().clone()
    if info.enabled:
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result


def seed_process_rng(seed: int, rank: int) -> None:
    """Seed process-local RNGs deterministically before model construction."""

    process_seed = int(seed) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32))
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(process_seed)


def capture_local_rng_state(info: DistributedInfo) -> dict:
    """Capture serializable RNG state for one training rank."""

    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": bytes(torch.get_rng_state().cpu().tolist()),
    }
    if info.device.type == "cuda":
        state["torch_cuda"] = bytes(torch.cuda.get_rng_state(info.device).cpu().tolist())
    return state


def collect_rng_states(info: DistributedInfo) -> list[dict]:
    """Collect every rank's RNG state so a DDP checkpoint can resume exactly."""

    local_state = capture_local_rng_state(info)
    if not info.enabled:
        return [local_state]
    gathered: list[dict | None] = [None] * info.world_size
    dist.all_gather_object(gathered, local_state)
    if any(state is None for state in gathered):
        raise RuntimeError("Did not receive RNG state from every distributed rank")
    return [state for state in gathered if state is not None]


def restore_local_rng_state(
    rng_states_by_rank: list[dict] | None,
    info: DistributedInfo,
    *,
    allow_inexact: bool,
) -> bool:
    """Restore this rank's RNG state, or explicitly reject an inexact resume."""

    if not rng_states_by_rank:
        message = (
            "Checkpoint does not contain per-rank RNG states, so dropout and other stochastic operations "
            "cannot resume exactly."
        )
        if allow_inexact:
            warnings.warn(message + " Continuing because --allow-inexact-resume was set.", stacklevel=2)
            return False
        raise ValueError(message + " Use --allow-inexact-resume to accept this legacy checkpoint.")
    if len(rng_states_by_rank) != info.world_size:
        message = (
            f"Checkpoint contains RNG states for {len(rng_states_by_rank)} ranks, "
            f"but the current world size is {info.world_size}."
        )
        if allow_inexact:
            warnings.warn(message + " RNG state will not be restored.", stacklevel=2)
            return False
        raise ValueError(message)

    state = rng_states_by_rank[info.rank]
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch.tensor(list(state["torch_cpu"]), dtype=torch.uint8))
    if info.device.type == "cuda":
        cuda_state = state.get("torch_cuda")
        if cuda_state is None:
            message = "Checkpoint has no CUDA RNG state for a CUDA resume."
            if allow_inexact:
                warnings.warn(message, stacklevel=2)
                return False
            raise ValueError(message)
        torch.cuda.set_rng_state(torch.tensor(list(cuda_state), dtype=torch.uint8), info.device)
    return True


def resume_signature_differences(saved: object, current: object, prefix: str = "") -> list[str]:
    """Return readable leaf-level differences between two resume signatures."""

    if isinstance(saved, dict) and isinstance(current, dict):
        differences: list[str] = []
        for key in sorted(set(saved) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in saved:
                differences.append(f"{path}: missing from checkpoint")
            elif key not in current:
                differences.append(f"{path}: missing from current run")
            else:
                differences.extend(resume_signature_differences(saved[key], current[key], path))
        return differences
    if saved != current:
        return [f"{prefix}: checkpoint={saved!r}, current={current!r}"]
    return []


def normalize_resume_signature(signature: object) -> object:
    """Fill defaults for model fields added after older exact checkpoints were saved."""

    if not isinstance(signature, dict):
        return signature
    normalized = dict(signature)
    normalized.setdefault(
        "loss",
        {
            "type": HARD_CE_LOSS,
            "soft_label_sigma": 0.01,
            "huber_beta": 0.02,
            "huber_weight": 1.0,
        },
    )
    model_config = normalized.get("model_config")
    if not isinstance(model_config, dict):
        return normalized
    normalized_model_config = dict(model_config)
    normalized_model_config.setdefault("context_progress_enabled", False)
    normalized_model_config.setdefault("context_slot_embedding_enabled", True)
    normalized_model_config.setdefault("query_history_offsets", [0])
    normalized_model_config.setdefault("context_sampling_mode", "end_anchored")
    normalized_model_config.setdefault("freeze_recap_detail_merger", False)
    normalized_model_config.setdefault("query_patch_tokens_per_image", 729)
    normalized_model_config.setdefault("query_vision_projector_mode", "linear")
    normalized_model_config.setdefault("query_image_preprocess_mode", "hf_cpu")
    normalized["model_config"] = normalized_model_config
    return normalized


def load_model_only_state(
    model: ValueFunctionEstimator,
    checkpoint: dict,
) -> list[str]:
    """Warm-start a model, permitting only a newly added query-history embedding."""

    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise KeyError("Model-only checkpoint has no model_state dictionary")
    checkpoint_model_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_model_config, dict):
        raise KeyError("Model-only checkpoint has no model_config dictionary")

    history_key = "impl.query_history_embedding.weight"
    source_history = checkpoint_model_config.get("query_history_offsets", [0])
    target_history = list(model.config.query_history_offsets)
    target_has_history = history_key in model.state_dict()
    allowed_missing = set()
    if (
        source_history == [0]
        and len(target_history) > 1
        and target_has_history
        and history_key not in model_state
    ):
        allowed_missing.add(history_key)

    incompatible = model.load_state_dict(model_state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != allowed_missing or unexpected:
        raise RuntimeError(
            "Unsafe model-only checkpoint mismatch: "
            f"missing={sorted(missing)}, allowed_missing={sorted(allowed_missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    if history_key in allowed_missing:
        history_parameter = dict(model.named_parameters())[history_key]
        with torch.no_grad():
            history_parameter.zero_()
    return sorted(allowed_missing)


def load_composite_warmstart_state(
    model: ValueFunctionEstimator,
    *,
    base_checkpoint_path: str,
    visual_checkpoint_path: str,
    progress_checkpoint_path: str | None = None,
) -> dict[str, object]:
    """Compose a fresh-run initialization from compatible prior experiments.

    The context-aware annotation checkpoint supplies every shape-compatible
    target parameter.  The RECAP-512 checkpoint then overrides the first query
    Qwen2 merger, and the context-progress checkpoint supplies the progress
    encoder.  Both target shared query/context mergers are initialized from
    the trained native 64-to-16 merger.  Its 640-dimensional 2x2 Qwen2-merger
    parameterization is shape-compatible with both the new 256-to-64 and
    64-to-16 stages; the old native 256-to-64 merger itself is incompatible
    because it consumes 1152-dimensional vision features.
    """

    target_state = model.state_dict()

    def read_model_state(path: str) -> dict[str, torch.Tensor]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model_state = checkpoint.get("model_state")
        if not isinstance(model_state, dict):
            raise KeyError(f"Warm-start checkpoint {path!r} has no model_state dictionary")
        return model_state

    base_state = read_model_state(base_checkpoint_path)
    loaded_from_base: list[str] = []
    skipped_base_shape: list[str] = []
    for name, source in base_state.items():
        target = target_state.get(name)
        if target is None:
            continue
        if tuple(source.shape) != tuple(target.shape):
            skipped_base_shape.append(name)
            continue
        target_state[name] = source
        loaded_from_base.append(name)

    visual_state = read_model_state(visual_checkpoint_path)
    visual_prefix = "impl.visual_projector."
    expected_visual = sorted(name for name in target_state if name.startswith(visual_prefix))
    loaded_from_visual: list[str] = []
    for name in expected_visual:
        source = visual_state.get(name)
        if source is None or tuple(source.shape) != tuple(target_state[name].shape):
            raise RuntimeError(
                f"Visual warm start cannot initialize {name}: target={tuple(target_state[name].shape)}, "
                f"source={None if source is None else tuple(source.shape)}"
            )
        target_state[name] = source
        loaded_from_visual.append(name)
    if not expected_visual:
        raise RuntimeError("Target model has no impl.visual_projector parameters to warm start")

    progress_prefix = "impl.context_progress_encoder."
    expected_progress = sorted(name for name in target_state if name.startswith(progress_prefix))
    if expected_progress and not progress_checkpoint_path:
        raise ValueError("A model with context progress requires a progress warm-start checkpoint")
    if not expected_progress and progress_checkpoint_path:
        raise ValueError("A model without context progress must omit the progress warm-start checkpoint")
    progress_state = read_model_state(progress_checkpoint_path) if expected_progress else {}
    loaded_from_progress: list[str] = []
    for name in expected_progress:
        source = progress_state.get(name)
        if source is None or tuple(source.shape) != tuple(target_state[name].shape):
            raise RuntimeError(
                f"Progress warm start cannot initialize {name}: target={tuple(target_state[name].shape)}, "
                f"source={None if source is None else tuple(source.shape)}"
            )
        target_state[name] = source
        loaded_from_progress.append(name)
    source_merger_prefix = "impl.native_spatial_merger_stage2."

    def initialize_shared_merger(target_prefix: str, stage_description: str) -> list[str]:
        initialized: list[str] = []
        for target_name in sorted(name for name in target_state if name.startswith(target_prefix)):
            suffix = target_name.removeprefix(target_prefix)
            source_name = source_merger_prefix + suffix
            source = base_state.get(source_name)
            if source is None or tuple(source.shape) != tuple(target_state[target_name].shape):
                raise RuntimeError(
                    f"Shared query/context {stage_description} merger warm start cannot map "
                    f"{source_name} to {target_name}"
                )
            target_state[target_name] = source
            initialized.append(target_name)
        if not initialized:
            raise RuntimeError(
                f"Target model has no shared query/context {stage_description} merger "
                "parameters to warm start"
            )
        return initialized

    initialized_query_stage2 = initialize_shared_merger(
        "impl.recap_spatial_merger_stage2.", "256-to-64"
    )
    initialized_query_stage3 = initialize_shared_merger(
        "impl.recap_spatial_merger_stage3.", "64-to-16"
    )

    model.load_state_dict(target_state, strict=True)
    return {
        "base_checkpoint": base_checkpoint_path,
        "base_parameter_count": len(loaded_from_base),
        "base_shape_mismatches": sorted(skipped_base_shape),
        "visual_checkpoint": visual_checkpoint_path,
        "visual_parameters": loaded_from_visual,
        "progress_checkpoint": progress_checkpoint_path,
        "progress_parameters": loaded_from_progress,
        "shared_stage2_from_base_parameters": initialized_query_stage2,
        "shared_stage3_from_base_parameters": initialized_query_stage3,
    }


def validate_resume_checkpoint(
    checkpoint: dict,
    current_signature: dict,
    *,
    allow_inexact: bool,
) -> None:
    """Reject checkpoints whose data order cannot be resumed at the recorded step."""

    problems: list[str] = []
    if checkpoint.get("train_progress") is None:
        problems.append("checkpoint has no train_progress and cannot identify its exact step within the epoch")
    saved_signature = checkpoint.get("run_config", {}).get("resume_signature")
    if saved_signature is None:
        problems.append("checkpoint has no resume_signature")
    else:
        problems.extend(
            resume_signature_differences(
                normalize_resume_signature(saved_signature),
                normalize_resume_signature(current_signature),
            )
        )
    if checkpoint.get("rng_states_by_rank") is None:
        problems.append("checkpoint has no per-rank RNG state")

    if not problems:
        return
    message = "Resume compatibility check failed:\n- " + "\n- ".join(problems)
    if allow_inexact:
        warnings.warn(message + "\nContinuing because --allow-inexact-resume was set.", stacklevel=2)
        return
    raise ValueError(message + "\nUse --allow-inexact-resume only if an inexact legacy resume is intentional.")


def has_reached_max_steps(global_step: int, max_steps: int) -> bool:
    return max_steps > 0 and global_step >= max_steps


def saved_wandb_run_id(checkpoint: dict) -> str | None:
    """Read a W&B run ID embedded by a legacy checkpoint."""

    candidate = checkpoint.get("wandb_run_id")
    if not candidate:
        candidate = checkpoint.get("run_config", {}).get("wandb_run_id")
    if candidate is None:
        return None
    candidate = str(candidate).strip()
    return candidate or None


def wandb_run_id_from_checkpoint_config(checkpoint_path: str | Path) -> str | None:
    """Read the single experiment-level run_config.json beside a checkpoint."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    experiment_dir = (
        checkpoint_path.parent.parent
        if checkpoint_path.parent.name == "checkpoints"
        else checkpoint_path.parent
    )
    config_path = experiment_dir / "run_config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read resume config {config_path}: {exc}") from exc
    candidate = config.get("wandb_run_id")
    if candidate is None:
        return None
    candidate = str(candidate).strip()
    return candidate or None


def resolve_wandb_resume_id(
    checkpoint_run_id: str | None,
    explicit_run_id: str | None,
) -> str | None:
    """Prefer checkpoint identity and reject accidental cross-run resumes."""

    checkpoint_run_id = str(checkpoint_run_id).strip() if checkpoint_run_id else None
    explicit_run_id = str(explicit_run_id).strip() if explicit_run_id else None
    if checkpoint_run_id and explicit_run_id and checkpoint_run_id != explicit_run_id:
        raise ValueError(
            "Explicit --wandb-run-id does not match the run ID stored in the experiment config: "
            f"config={checkpoint_run_id!r}, explicit={explicit_run_id!r}."
        )
    return checkpoint_run_id or explicit_run_id


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def nonfinite_gradient_report(model: torch.nn.Module, max_examples: int = 20) -> dict:
    examples = []
    nonfinite_parameters = 0
    nonfinite_elements = 0
    gradient_elements = 0
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        gradient_elements += grad.numel()
        finite_mask = torch.isfinite(grad)
        if bool(finite_mask.all().item()):
            continue
        nonfinite_parameters += 1
        bad_count = int((~finite_mask).sum().item())
        nonfinite_elements += bad_count
        if len(examples) < max_examples:
            finite_values = grad[finite_mask]
            max_abs_finite = float(finite_values.detach().abs().max().item()) if finite_values.numel() else None
            examples.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(grad.dtype),
                    "nonfinite_elements": bad_count,
                    "total_elements": int(grad.numel()),
                    "max_abs_finite": max_abs_finite,
                }
            )
    return {
        "nonfinite_parameters": nonfinite_parameters,
        "nonfinite_elements": nonfinite_elements,
        "gradient_elements": gradient_elements,
        "examples": examples,
    }


def clip_gradients_and_maybe_report(
    model: torch.nn.Module,
    *,
    max_norm: float,
    check_nonfinite: bool,
) -> tuple[torch.Tensor, dict | None]:
    """Compute the total norm once and only scan individual gradients on failure."""

    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    gradients = [parameter.grad for parameter in parameters]
    grad_norm = torch.nn.utils.get_total_norm(gradients, norm_type=2.0)
    grad_report = None
    if check_nonfinite and not bool(torch.isfinite(grad_norm).item()):
        candidate_report = nonfinite_gradient_report(model)
        if candidate_report["nonfinite_parameters"] > 0:
            grad_report = candidate_report
    if max_norm > 0:
        torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, grad_norm)
    return grad_norm, grad_report


def batch_to_device(batch: dict, device: torch.device, *, keep_cpu_keys: set[str] | None = None) -> dict:
    keep_cpu_keys = keep_cpu_keys or set()
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key in keep_cpu_keys:
                moved[key] = value
            else:
                moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def save_checkpoint(
    path: str | Path,
    *,
    model: ValueFunctionEstimator,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    run_config: dict,
    train_progress: dict[str, int] | None = None,
    rng_states_by_rank: list[dict] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_run_config = dict(run_config)
    checkpoint_run_config.pop("wandb_run_id", None)
    payload = {
        "checkpoint_format_version": 2,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "run_config": checkpoint_run_config,
        "epoch": epoch,
        "step": step,
    }
    if train_progress is not None:
        payload["train_progress"] = train_progress
    if rng_states_by_rank is not None:
        payload["rng_states_by_rank"] = rng_states_by_rank
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def resolve_effective_backbone(requested_backbone: str, *, no_context: bool) -> str:
    """Resolve a context backbone to its current-query-only counterpart."""
    if not no_context:
        return requested_backbone
    if requested_backbone == SIGLIP_GEMMA_CONTEXT_BACKBONE:
        return SIGLIP_GEMMA_BACKBONE
    if requested_backbone == GEMMA3_NATIVE_VISION_CONTEXT_BACKBONE:
        return GEMMA3_NATIVE_VISION_BACKBONE
    if requested_backbone == GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE:
        return GEMMA3_NATIVE_STYLE_270M_BACKBONE
    return requested_backbone


def apply_vfe_preset(args: argparse.Namespace) -> dict:
    vfe = load_vfe_preset(args.vfe_preset, args.vfe_config)
    model = vfe["model"]
    direct_keys = {
        "backbone",
        "vision_model_name",
        "context_vision_model_name",
        "native_vision_model_name",
        "native_vision_model_revision",
        "native_vision_projector_mode",
        "attention_mode",
        "language_model_name",
        "torch_dtype",
        "freeze_backbone",
        "freeze_native_vision",
        "freeze_vision",
        "freeze_recap_detail_merger",
        "lora",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "gradient_checkpointing",
        "obs_key",
        "query_history_offsets",
        "proprio_key",
        "image_size",
        "context_obs_key",
        "context_image_size",
        "context_progress_enabled",
        "context_slot_embedding_enabled",
        "context_max_original_timesteps",
        "context_stride",
        "context_sampling_mode",
        "context_max_timesteps",
        "context_patch_tokens_per_image",
        "query_patch_tokens_per_image",
        "query_vision_projector_mode",
        "query_image_preprocess_mode",
        "context_image_chunk_size",
        "max_language_tokens",
        "loss_type",
        "soft_label_sigma",
        "huber_beta",
        "huber_weight",
        "adapter_init_std",
        "adapter_token_scale",
        "adapter_output_norm",
        "log_input_embedding_stats",
        "hidden_dim",
        "transformer_layers",
        "transformer_heads",
        "dropout",
        "vocab_size",
    }
    for key in direct_keys:
        value = model.get(key, VfeModelConfig.attention_mode) if key == "attention_mode" else model[key]
        setattr(args, key, value)
    args.no_context = not bool(model["context_enabled"])
    args.no_match_adapter_token_rms = not bool(model["match_adapter_token_rms"])
    args.no_source_embeddings = not bool(model["use_source_embeddings"])
    return vfe


def train(args: argparse.Namespace) -> None:
    distributed_info = initialize_distributed(args)
    seed_process_rng(args.seed, distributed_info.rank)
    vfe_config = apply_vfe_preset(args)
    model_preset = vfe_config["model"]
    feature_preset = vfe_config["context_features"]
    preprocess_config = load_preset_preprocess_config(
        args.data_split_preset,
        args.data_split_config,
    )
    ground_truth_version = preprocess_config["ground_truth_version"]
    requested_backbone = args.backbone
    effective_backbone = resolve_effective_backbone(requested_backbone, no_context=args.no_context)
    args.processed_dir = str(
        resolve_processed_dir(preprocess_config["output_dir"], ground_truth_version)
    )
    context_enabled = effective_backbone in CONTEXT_BACKBONES
    if args.loss_type not in SUPPORTED_VALUE_LOSSES:
        raise ValueError(
            f"VFE preset {args.vfe_preset!r} uses unsupported loss_type={args.loss_type!r}; "
            f"expected one of {sorted(SUPPORTED_VALUE_LOSSES)}"
        )
    if args.soft_label_sigma <= 0 or args.huber_beta <= 0 or args.huber_weight < 0:
        raise ValueError(
            "VFE loss parameters require soft_label_sigma > 0, huber_beta > 0, and huber_weight >= 0"
        )
    if context_enabled != bool(model_preset["context_enabled"]):
        raise ValueError(
            f"VFE preset {args.vfe_preset!r} context_enabled does not match "
            f"its effective backbone {effective_backbone!r}"
        )
    args.context_native_feature_dir = (
        feature_preset["output_dir"]
        if bool(feature_preset["enabled"])
        and context_enabled
        and effective_backbone == GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE
        and args.freeze_vision
        else None
    )
    if args.freeze_native_vision and effective_backbone not in {
        GEMMA3_NATIVE_VISION_BACKBONE,
        GEMMA3_NATIVE_VISION_CONTEXT_BACKBONE,
    }:
        raise ValueError("VFE preset freeze_native_vision requires a Gemma 3 native-vision backbone")
    if args.freeze_vision and effective_backbone not in {
        SIGLIP_GEMMA_BACKBONE,
        GEMMA3_NATIVE_STYLE_270M_BACKBONE,
        GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE,
    }:
        raise ValueError(
            "VFE preset freeze_vision requires the SigLIP no-context or "
            "gemma3_native_style_270m backbone"
        )
    if args.context_native_feature_dir:
        if effective_backbone != GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE:
            raise ValueError(
                "The preset context_native_feature_dir requires the "
                "gemma3_native_style_270m_context backbone"
            )
        if not args.freeze_vision:
            raise ValueError("The VFE context feature cache requires model.freeze_vision=true")
    if args.lora and effective_backbone not in {
        GEMMA3_NATIVE_VISION_BACKBONE,
        GEMMA3_NATIVE_VISION_CONTEXT_BACKBONE,
    }:
        raise ValueError("VFE preset lora requires a Gemma 3 native-vision backbone")
    if args.lora and args.freeze_backbone:
        raise ValueError("VFE preset lora cannot be combined with freeze_backbone")
    if args.wandb_run_id and not args.resume:
        raise ValueError("--wandb-run-id requires --resume so it cannot overwrite an unrelated W&B run")
    if args.resume_model_only and args.wandb_run_id:
        raise ValueError("--wandb-run-id cannot be combined with --resume-model-only, which starts a fresh run")
    composite_warmstart_paths = [
        args.warmstart_base_checkpoint,
        args.warmstart_visual_checkpoint,
        args.warmstart_progress_checkpoint,
    ]
    uses_composite_warmstart = any(composite_warmstart_paths)
    if uses_composite_warmstart and not all(composite_warmstart_paths[:2]):
        raise ValueError(
            "Composite warm start requires --warmstart-base-checkpoint, "
            "and --warmstart-visual-checkpoint together"
        )
    if uses_composite_warmstart and bool(args.context_progress_enabled) != bool(args.warmstart_progress_checkpoint):
        raise ValueError(
            "Composite warm start requires --warmstart-progress-checkpoint exactly when context progress is enabled"
        )
    if uses_composite_warmstart and (args.resume or args.resume_model_only or args.wandb_run_id):
        raise ValueError("Composite warm start cannot be combined with resume or W&B resume options")
    for checkpoint_path in composite_warmstart_paths:
        if checkpoint_path and not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Composite warm-start checkpoint does not exist: {checkpoint_path}")

    processed_dir = Path(args.processed_dir).expanduser().resolve()
    output_dir = output_dir_from_exp_name(args.exp_name)
    checkpoint_dir = output_dir / "checkpoints"
    if distributed_info.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(distributed_info)

    summary = read_json(processed_dir / "summary.json")
    value_min, value_max = summary["value_range"]
    data_split = resolve_data_split(
        preset_name=args.data_split_preset,
        phase="train",
        preset_path=args.data_split_config,
    )
    context_max_timesteps = args.context_max_timesteps
    dataloader = make_train_dataloader(
        processed_dir,
        split=data_split["demo_split"],
        obs_key=args.obs_key,
        proprio_key=args.proprio_key,
        image_size=args.image_size,
        query_history_offsets=args.query_history_offsets,
        tasks_per_batch=args.tasks_per_batch,
        samples_per_task=args.samples_per_task,
        steps_per_epoch=args.steps_per_epoch,
        sampling_mode=args.sampling_mode,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        task_id_filter=data_split["task_id_set"],
        query_demo_ids_by_task=data_split.get("query_demo_ids_by_task"),
        context_demo_ids_by_task=data_split.get("context_demo_ids_by_task"),
        context_enabled=context_enabled,
        context_progress_enabled=args.context_progress_enabled,
        context_selection=data_split.get("context_selection", "train_seeded"),
        context_seed=int(data_split.get("context_split_seed", args.seed)),
        context_obs_key=args.context_obs_key,
        context_image_size=args.context_image_size,
        context_max_timesteps=context_max_timesteps,
        context_sampling_mode=args.context_sampling_mode,
        context_stride=args.context_stride,
        context_max_original_timesteps=args.context_max_original_timesteps,
        context_native_feature_dir=args.context_native_feature_dir,
        native_vision_model_name=args.native_vision_model_name,
        native_vision_model_revision=args.native_vision_model_revision,
        context_feature_model_name=feature_preset.get("model_name"),
        context_feature_model_revision=feature_preset.get("model_revision"),
        context_feature_tokens_per_image=int(feature_preset.get("tokens_per_image", 256)),
        context_native_feature_stage=feature_preset["stage"],
        context_native_feature_dtype=feature_preset["feature_dtype"],
        data_split_preset=args.data_split_preset,
        rank=distributed_info.rank,
        world_size=distributed_info.world_size,
        pad_query_frame_once_last_batch=(
            args.pad_query_frame_once_last_batch
            or args.attention_mode in CUSTOM_CONTEXT_ATTENTION_MODES
        ),
    )
    train_sampler = dataloader.batch_sampler
    if not hasattr(train_sampler, "set_epoch") or not hasattr(train_sampler, "set_start_step"):
        raise TypeError("Training batch sampler must implement set_epoch() and set_start_step()")
    effective_steps_per_epoch = int(getattr(train_sampler, "steps_per_epoch", len(dataloader)))
    proprio_dim = dataloader.dataset.infer_proprio_dim()
    context_patch_tokens_per_image = (
        256
        if effective_backbone
        in {
            GEMMA3_NATIVE_VISION_CONTEXT_BACKBONE,
            GEMMA3_NATIVE_VISION_BACKBONE,
            GEMMA3_NATIVE_STYLE_270M_BACKBONE,
            GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE,
        }
        else args.context_patch_tokens_per_image
    )
    model_config = VfeModelConfig(
        backbone=effective_backbone,
        vision_model_name=args.vision_model_name,
        language_model_name=args.language_model_name,
        context_vision_model_name=args.context_vision_model_name,
        native_vision_model_name=args.native_vision_model_name,
        native_vision_model_revision=args.native_vision_model_revision,
        native_vision_projector_mode=args.native_vision_projector_mode,
        attention_mode=args.attention_mode,
        freeze_backbone=args.freeze_backbone,
        freeze_native_vision=args.freeze_native_vision,
        freeze_vision=args.freeze_vision,
        freeze_recap_detail_merger=args.freeze_recap_detail_merger,
        use_lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        torch_dtype=args.torch_dtype,
        num_bins=int(summary["num_bins"]),
        value_min=float(value_min),
        value_max=float(value_max),
        hidden_dim=args.hidden_dim,
        proprio_dim=proprio_dim,
        vocab_size=args.vocab_size,
        max_language_tokens=args.max_language_tokens,
        query_history_offsets=args.query_history_offsets,
        context_max_timesteps=context_max_timesteps,
        context_sampling_mode=args.context_sampling_mode,
        context_views=len([key for key in (args.context_obs_key or args.obs_key).split(",") if key.strip()]),
        context_progress_enabled=args.context_progress_enabled,
        context_slot_embedding_enabled=args.context_slot_embedding_enabled,
        context_image_size=args.context_image_size,
        context_patch_tokens_per_image=context_patch_tokens_per_image,
        query_patch_tokens_per_image=args.query_patch_tokens_per_image,
        query_vision_projector_mode=args.query_vision_projector_mode,
        query_image_preprocess_mode=args.query_image_preprocess_mode,
        context_image_chunk_size=args.context_image_chunk_size,
        gradient_checkpointing=args.gradient_checkpointing,
        adapter_init_std=args.adapter_init_std,
        adapter_token_scale=args.adapter_token_scale,
        adapter_output_norm=args.adapter_output_norm,
        match_adapter_token_rms=not args.no_match_adapter_token_rms,
        use_source_embeddings=not args.no_source_embeddings,
        log_input_embedding_stats=args.log_input_embedding_stats,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
    )
    resume_signature = {
        "format_version": 1,
        "sampling": {
            "mode": args.sampling_mode,
            "seed": args.seed,
            "dataset_length": len(dataloader.dataset),
            "effective_steps_per_epoch": effective_steps_per_epoch,
            "tasks_per_batch_per_rank": args.tasks_per_batch,
            "samples_per_task": args.samples_per_task,
            "global_batch_size": args.tasks_per_batch
            * args.samples_per_task
            * distributed_info.world_size,
            "world_size": distributed_info.world_size,
        },
        "data": {
            "resolved_split": {
                key: value for key, value in data_split.items() if key != "task_id_set"
            },
            "obs_key": args.obs_key,
            "proprio_key": args.proprio_key,
            "image_size": args.image_size,
            "context_obs_key": args.context_obs_key,
            "context_image_size": args.context_image_size,
            "context_stride": args.context_stride,
            "context_max_original_timesteps": args.context_max_original_timesteps,
        },
        "model_config": model_config.to_dict(),
        "optimizer": {
            "type": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "loss": {
            "type": args.loss_type,
            "soft_label_sigma": args.soft_label_sigma,
            "huber_beta": args.huber_beta,
            "huber_weight": args.huber_weight,
        },
    }

    # Let rank zero populate the local Hugging Face cache before the other ranks construct the same model.
    if distributed_info.enabled and not distributed_info.is_main_process:
        distributed_barrier(distributed_info)
    model: torch.nn.Module = ValueFunctionEstimator(model_config).to(distributed_info.device)
    if distributed_info.enabled and distributed_info.is_main_process:
        distributed_barrier(distributed_info)
    if distributed_info.enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[distributed_info.local_rank] if distributed_info.device.type == "cuda" else None,
            output_device=distributed_info.local_rank if distributed_info.device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    model_base = unwrap_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    checkpoint_wandb_run_id = None
    total_params = total_parameter_count(model_base)
    trainable_params = trainable_parameter_count(model_base)

    composite_warmstart_report = None
    if uses_composite_warmstart:
        composite_warmstart_report = load_composite_warmstart_state(
            model_base,
            base_checkpoint_path=args.warmstart_base_checkpoint,
            visual_checkpoint_path=args.warmstart_visual_checkpoint,
            progress_checkpoint_path=args.warmstart_progress_checkpoint,
        )
        if distributed_info.is_main_process:
            print(
                "Loaded composite model-only warm start with a fresh optimizer: "
                f"{json.dumps(composite_warmstart_report, sort_keys=True)}",
                flush=True,
            )

    if args.resume:
        if not args.resume_model_only:
            checkpoint_wandb_run_id = wandb_run_id_from_checkpoint_config(args.resume)
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not args.resume_model_only and checkpoint_wandb_run_id is None:
            # Compatibility only: checkpoints produced before the experiment-level
            # config became authoritative may contain an embedded run ID.
            checkpoint_wandb_run_id = saved_wandb_run_id(checkpoint)
        rng_states_by_rank = checkpoint.get("rng_states_by_rank")
        if not args.resume_model_only:
            validate_resume_checkpoint(
                checkpoint,
                resume_signature,
                allow_inexact=args.allow_inexact_resume,
            )
        if args.resume_model_only:
            initialized_parameters = load_model_only_state(model_base, checkpoint)
            if distributed_info.is_main_process:
                print(
                    "Loaded model-only warm start; zero-initialized new parameters: "
                    f"{initialized_parameters or 'none'}",
                    flush=True,
                )
        else:
            model_base.load_state_dict(checkpoint["model_state"])
        if not args.resume_model_only:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            train_progress = checkpoint.get("train_progress")
            if train_progress is None:
                start_epoch = int(checkpoint.get("epoch", 0))
            else:
                start_epoch = int(train_progress["next_epoch"])
                start_step_in_epoch = int(train_progress["next_step_in_epoch"])
            global_step = int(checkpoint.get("step", 0))
        del checkpoint
        gc.collect()
        if distributed_info.device.type == "cuda":
            torch.cuda.empty_cache()
        if not args.resume_model_only:
            restore_local_rng_state(
                rng_states_by_rank,
                distributed_info,
                allow_inexact=args.allow_inexact_resume,
            )
    effective_wandb_run_id = (
        None
        if args.resume_model_only or uses_composite_warmstart
        else resolve_wandb_resume_id(checkpoint_wandb_run_id, args.wandb_run_id)
    )
    if start_step_in_epoch < 0 or start_step_in_epoch > effective_steps_per_epoch:
        raise ValueError(
            "Checkpoint next_step_in_epoch is outside the current epoch length: "
            f"{start_step_in_epoch} not in [0, {effective_steps_per_epoch}]. "
            "Resume with the same split, batch size, and sampling mode used to create the checkpoint."
        )

    run_config = vars(args).copy()
    run_config["composite_warmstart_report"] = composite_warmstart_report
    run_config["requested_backbone"] = requested_backbone
    run_config["effective_backbone"] = effective_backbone
    run_config["context_enabled"] = context_enabled
    run_config["resolved_vfe_preset"] = vfe_config
    run_config["output_dir"] = str(output_dir)
    run_config["checkpoint_dir"] = str(checkpoint_dir)
    run_config["wandb_project"] = WANDB_PROJECT
    run_config["wandb_run_name"] = args.exp_name
    run_config["wandb_run_id"] = effective_wandb_run_id
    run_config["resolved_data_split"] = {
        key: value for key, value in data_split.items() if key != "task_id_set"
    }
    run_config["resolved_data_split"]["num_tasks"] = len(data_split["task_id_set"])
    run_config["model_config"] = model_config.to_dict()
    run_config["processed_summary"] = str(processed_dir / "summary.json")
    run_config["total_parameters"] = total_params
    run_config["trainable_parameters"] = trainable_params
    run_config["derived_context_max_timesteps"] = context_max_timesteps
    run_config["sampling_mode"] = args.sampling_mode
    run_config["effective_steps_per_epoch"] = effective_steps_per_epoch
    run_config["pad_query_frame_once_last_batch"] = bool(
        getattr(train_sampler, "pad_last_batch", False)
    )
    global_batch_size = (
        args.tasks_per_batch * args.samples_per_task * distributed_info.world_size
    )
    run_config["query_frame_once_padding_examples_per_epoch"] = (
        (-len(dataloader.dataset)) % global_batch_size
        if run_config["pad_query_frame_once_last_batch"]
        else 0
    )
    run_config["max_steps"] = args.max_steps
    run_config["resume_signature"] = resume_signature
    run_config["distributed"] = {
        "enabled": distributed_info.enabled,
        "world_size": distributed_info.world_size,
        "per_rank_batch_size": args.tasks_per_batch * args.samples_per_task,
        "global_batch_size": args.tasks_per_batch * args.samples_per_task * distributed_info.world_size,
    }
    if distributed_info.is_main_process:
        (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    wandb_run = None
    metrics_file = None
    metrics_path = output_dir / "metrics.jsonl"
    if distributed_info.is_main_process:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb is required because training always logs to vfe-libero100.") from exc
        wandb_init_kwargs: dict[str, object] = {
            "project": WANDB_PROJECT,
            "name": args.exp_name,
            "config": run_config,
            "dir": str(output_dir),
        }
        if effective_wandb_run_id:
            wandb_init_kwargs.update(id=effective_wandb_run_id, resume="must", allow_val_change=True)
        elif args.resume and not args.resume_model_only:
            warnings.warn(
                "Resume experiment config has no W&B run ID and no --wandb-run-id was provided; "
                "starting a new W&B run for this legacy checkpoint.",
                stacklevel=2,
            )
        wandb_run = wandb.init(**wandb_init_kwargs)
        if not wandb_run.id:
            raise RuntimeError("W&B initialized without a run ID, so the run cannot be resumed safely")
        actual_wandb_run_id = str(wandb_run.id).strip()
        if effective_wandb_run_id and actual_wandb_run_id != effective_wandb_run_id:
            raise RuntimeError(
                "W&B resumed an unexpected run ID: "
                f"expected {effective_wandb_run_id!r}, got {actual_wandb_run_id!r}"
            )
        run_config["wandb_run_id"] = actual_wandb_run_id
        wandb_run.config.update({"wandb_run_id": actual_wandb_run_id}, allow_val_change=True)
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics_file = metrics_path.open("a", encoding="utf-8")

        print(f"device: {distributed_info.device} rank=0/{distributed_info.world_size}")
        print(
            f"batch size: local={args.tasks_per_batch * args.samples_per_task} "
            f"global={args.tasks_per_batch * args.samples_per_task * distributed_info.world_size}"
        )
        print(f"sampling mode: {args.sampling_mode}")
        print(f"steps per epoch: {effective_steps_per_epoch}")
        if run_config["pad_query_frame_once_last_batch"]:
            print(
                "query_frame_once static final batch: "
                f"randomly pad {run_config['query_frame_once_padding_examples_per_epoch']} "
                "examples per epoch"
            )
        print(f"num workers per rank: {args.num_workers}")
        print(
            f"data split: preset={args.data_split_preset} phase=train "
            f"tasks={len(data_split['task_id_set'])} demo_split={data_split['demo_split']}"
        )
        print(f"obs keys: {args.obs_key}")
        print(f"proprio keys: {args.proprio_key or '<none>'} dim={proprio_dim}")
        print(
            f"backbone: requested={requested_backbone} effective={effective_backbone} "
            f"no_context={args.no_context}"
        )
        print(
            f"context: enabled={context_enabled} selection={data_split.get('context_selection', '<none>')} "
            f"manifest={data_split.get('context_split_manifest', '<none>')}"
        )
        print(
            f"context sampling: mode={args.context_sampling_mode} slots={context_max_timesteps}; "
            + ("whole demo, uniformly sampled (stride/source-window settings ignored)"
               if args.context_sampling_mode == "uniform" else
               f"max_original_timesteps={args.context_max_original_timesteps} stride={args.context_stride}")
        )
        print(f"context native-feature cache: {args.context_native_feature_dir or '<disabled>'}")
        print(f"gradient checkpointing: {args.gradient_checkpointing}")
        print(f"freeze backbone: {args.freeze_backbone}")
        print(f"freeze native vision inputs: {args.freeze_native_vision}")
        print(
            f"LoRA: enabled={args.lora} rank={args.lora_rank} alpha={args.lora_alpha} "
            f"dropout={args.lora_dropout} targets={args.lora_target_modules} "
            f"replaced_linears={getattr(model_base.impl, 'lora_module_count', 0)}"
        )
        print(
            f"adapter scale: init_std={args.adapter_init_std} token_scale={args.adapter_token_scale} "
            f"output_norm={args.adapter_output_norm} match_rms={not args.no_match_adapter_token_rms}"
        )
        print(
            f"value loss: type={args.loss_type} soft_label_sigma={args.soft_label_sigma} "
            f"huber_beta={args.huber_beta} huber_weight={args.huber_weight}"
        )
        print(f"parameters: total {total_params:,} trainable {trainable_params:,}")
        print(f"output dir: {output_dir}")
        print(f"data split preset: {args.data_split_preset}")
        print(f"processed ground truth: {processed_dir}")
        print(f"wandb: project={WANDB_PROJECT} run={args.exp_name}")
        print(f"metrics: {metrics_path}")
    try:
        if has_reached_max_steps(global_step, args.max_steps):
            if distributed_info.is_main_process:
                stop_metrics = {
                    "epoch": start_epoch + 1,
                    "step_in_epoch": start_step_in_epoch,
                    "global_step": global_step,
                    "max_steps": args.max_steps,
                    "event": "max_steps_already_reached",
                }
                if metrics_file is None:
                    raise RuntimeError("Rank zero did not open its metrics file")
                metrics_file.write(json.dumps(stop_metrics, sort_keys=True) + "\n")
                metrics_file.flush()
                if wandb_run is not None:
                    wandb_run.log(stop_metrics, step=global_step)
                print(
                    f"checkpoint is already at global_step={global_step}, "
                    f"which satisfies max_steps={args.max_steps}; no optimizer update was run"
                )
            return

        stop_after_max_steps = False
        for epoch in range(start_epoch, args.epochs):
            resume_step = start_step_in_epoch if epoch == start_epoch else 0
            train_sampler.set_epoch(epoch)
            train_sampler.set_start_step(resume_step)
            model.train()
            epoch_start = time.time()
            log_start = epoch_start
            next_batch_start = time.time()
            running_loss = None
            running_ce = None
            running_soft_ce = None
            running_huber = None
            running_mse = None
            running_mae = None
            running_entropy = None
            running_target_mean = None
            running_pred_mean = None
            running_target_std = None
            running_pred_std = None
            running_grad_norm = None
            running_data_time = 0.0
            running_compute_time = 0.0
            running_steps = 0
            running_samples = 0

            for step_in_epoch, batch in enumerate(dataloader, start=resume_step + 1):
                data_time = time.time() - next_batch_start
                compute_start = time.time()
                keep_cpu_keys = {"query_image"} if effective_backbone in HF_IMAGE_BACKBONES else set()
                if effective_backbone in CONTEXT_BACKBONES:
                    keep_cpu_keys.add("context_images")
                if "context_native_features" in batch:
                    # The unmerged 1024-patch cache is much larger than its
                    # 16-token context output. Keep it on CPU and let the model
                    # transfer only context_image_chunk_size images at a time.
                    keep_cpu_keys.add("context_native_features")
                batch = batch_to_device(batch, distributed_info.device, keep_cpu_keys=keep_cpu_keys)
                logits = model(batch=batch)
                loss_components = value_loss_components(
                    logits,
                    value_bins=batch["value_bin"],
                    value_continuous=batch["value_continuous"],
                    bin_values=model_base.bin_values,
                    loss_type=args.loss_type,
                    soft_label_sigma=args.soft_label_sigma,
                    huber_beta=args.huber_beta,
                    huber_weight=args.huber_weight,
                )
                loss = loss_components["loss"]
                ce_loss = loss_components["hard_ce"]
                soft_ce_loss = loss_components["soft_ce"]
                huber_loss = loss_components["huber"]
                pred_value = loss_components["predicted_value"]
                value_mse = F.mse_loss(pred_value, batch["value_continuous"]).detach()
                value_mae = F.l1_loss(pred_value, batch["value_continuous"]).detach()
                probs = F.softmax(logits.detach(), dim=-1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                check_nonfinite = args.log_nonfinite_grads or args.error_on_nonfinite_grad
                grad_norm, grad_report = clip_gradients_and_maybe_report(
                    model_base,
                    max_norm=args.grad_clip_norm,
                    check_nonfinite=check_nonfinite,
                )
                if check_nonfinite:
                    local_nonfinite = grad_report is not None
                    any_nonfinite = bool(
                        reduce_max(
                            torch.tensor(int(local_nonfinite), device=distributed_info.device),
                            distributed_info,
                        ).item()
                    )
                    if local_nonfinite:
                        print(
                            f"rank {distributed_info.rank} nonfinite gradient report:",
                            json.dumps(grad_report, indent=2, sort_keys=True),
                            flush=True,
                        )
                    if any_nonfinite and args.error_on_nonfinite_grad:
                        raise FloatingPointError("Non-finite gradients detected; see rank-specific reports above.")
                optimizer.step()

                global_step += 1
                batch_size = int(batch["value_bin"].shape[0])
                running_steps += 1
                running_samples += batch_size
                running_data_time += data_time
                running_compute_time += time.time() - compute_start
                loss_detached = loss.detach()
                ce_detached = ce_loss.detach()
                soft_ce_detached = soft_ce_loss.detach()
                huber_detached = huber_loss.detach()
                target_mean = batch["value_continuous"].detach().mean()
                pred_mean = pred_value.detach().mean()
                target_std = batch["value_continuous"].detach().std(unbiased=False)
                pred_std = pred_value.detach().std(unbiased=False)
                grad_norm_detached = (
                    grad_norm.detach()
                    if hasattr(grad_norm, "detach")
                    else torch.tensor(float(grad_norm), device=distributed_info.device)
                )
                running_loss = loss_detached if running_loss is None else running_loss + loss_detached
                running_ce = ce_detached if running_ce is None else running_ce + ce_detached
                running_soft_ce = (
                    soft_ce_detached if running_soft_ce is None else running_soft_ce + soft_ce_detached
                )
                running_huber = huber_detached if running_huber is None else running_huber + huber_detached
                running_mse = value_mse if running_mse is None else running_mse + value_mse
                running_mae = value_mae if running_mae is None else running_mae + value_mae
                running_entropy = entropy if running_entropy is None else running_entropy + entropy
                running_target_mean = target_mean if running_target_mean is None else running_target_mean + target_mean
                running_pred_mean = pred_mean if running_pred_mean is None else running_pred_mean + pred_mean
                running_target_std = target_std if running_target_std is None else running_target_std + target_std
                running_pred_std = pred_std if running_pred_std is None else running_pred_std + pred_std
                running_grad_norm = grad_norm_detached if running_grad_norm is None else running_grad_norm + grad_norm_detached

                if global_step % args.log_every == 0:
                    reduced = {
                        "loss": reduce_sum(running_loss, distributed_info),
                        "ce": reduce_sum(running_ce, distributed_info),
                        "soft_ce": reduce_sum(running_soft_ce, distributed_info),
                        "huber": reduce_sum(running_huber, distributed_info),
                        "value_mse": reduce_sum(running_mse, distributed_info),
                        "value_mae": reduce_sum(running_mae, distributed_info),
                        "prediction_entropy": reduce_sum(running_entropy, distributed_info),
                        "target_value_mean": reduce_sum(running_target_mean, distributed_info),
                        "predicted_value_mean": reduce_sum(running_pred_mean, distributed_info),
                        "target_value_std": reduce_sum(running_target_std, distributed_info),
                        "predicted_value_std": reduce_sum(running_pred_std, distributed_info),
                        "grad_norm": reduce_sum(running_grad_norm, distributed_info),
                        "data_time_seconds": reduce_sum(
                            torch.tensor(running_data_time, device=distributed_info.device), distributed_info
                        ),
                        "compute_time_seconds": reduce_sum(
                            torch.tensor(running_compute_time, device=distributed_info.device), distributed_info
                        ),
                        "samples": reduce_sum(
                            torch.tensor(running_samples, device=distributed_info.device), distributed_info
                        ),
                        "steps": reduce_sum(
                            torch.tensor(running_steps, device=distributed_info.device), distributed_info
                        ),
                    }
                    elapsed = reduce_max(
                        torch.tensor(time.time() - log_start, device=distributed_info.device), distributed_info
                    ).item()
                    elapsed = max(float(elapsed), 1e-9)
                    cuda_memory_gb = None
                    if torch.cuda.is_available() and distributed_info.device.type == "cuda":
                        cuda_memory_gb = reduce_max(
                            torch.tensor(
                                [
                                    torch.cuda.memory_allocated(distributed_info.device) / 1e9,
                                    torch.cuda.memory_reserved(distributed_info.device) / 1e9,
                                    torch.cuda.max_memory_allocated(distributed_info.device) / 1e9,
                                ],
                                device=distributed_info.device,
                            ),
                            distributed_info,
                        )
                    if distributed_info.is_main_process:
                        global_steps = max(float(reduced["steps"].item()), 1.0)
                        mean_data_time = float(reduced["data_time_seconds"].item()) / global_steps
                        mean_compute_time = float(reduced["compute_time_seconds"].item()) / global_steps
                        metrics = {
                            "epoch": epoch + 1,
                            "step_in_epoch": step_in_epoch,
                            "global_step": global_step,
                            "loss": float((reduced["loss"] / global_steps).item()),
                            "ce": float((reduced["ce"] / global_steps).item()),
                            "soft_ce": float((reduced["soft_ce"] / global_steps).item()),
                            "huber": float((reduced["huber"] / global_steps).item()),
                            "loss_type": args.loss_type,
                            "value_mse": float((reduced["value_mse"] / global_steps).item()),
                            "value_mae": float((reduced["value_mae"] / global_steps).item()),
                            "prediction_entropy": float((reduced["prediction_entropy"] / global_steps).item()),
                            "target_value_mean": float((reduced["target_value_mean"] / global_steps).item()),
                            "predicted_value_mean": float((reduced["predicted_value_mean"] / global_steps).item()),
                            "target_value_std": float((reduced["target_value_std"] / global_steps).item()),
                            "predicted_value_std": float((reduced["predicted_value_std"] / global_steps).item()),
                            "grad_norm": float((reduced["grad_norm"] / global_steps).item()),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "batch_size": batch_size * distributed_info.world_size,
                            "per_rank_batch_size": batch_size,
                            "tasks_per_batch_per_rank": args.tasks_per_batch,
                            "samples_per_task": args.samples_per_task,
                            "world_size": distributed_info.world_size,
                            "samples_per_second": float(reduced["samples"].item()) / elapsed,
                            "steps_per_second": running_steps / elapsed,
                            "data_time_seconds": mean_data_time,
                            "compute_time_seconds": mean_compute_time,
                            "data_time_fraction": mean_data_time / max(mean_data_time + mean_compute_time, 1e-9),
                            "total_parameters": total_params,
                            "trainable_parameters": trainable_params,
                        }
                        if cuda_memory_gb is not None:
                            metrics["cuda_memory_allocated_gb"] = float(cuda_memory_gb[0].item())
                            metrics["cuda_memory_reserved_gb"] = float(cuda_memory_gb[1].item())
                            metrics["cuda_max_memory_allocated_gb"] = float(cuda_memory_gb[2].item())
                        print(
                            f"epoch {epoch + 1} step {step_in_epoch}/{effective_steps_per_epoch} "
                            f"global {global_step} loss {metrics['loss']:.4f} "
                            f"ce {metrics['ce']:.4f} soft_ce {metrics['soft_ce']:.4f} "
                            f"huber {metrics['huber']:.6f} mse {metrics['value_mse']:.6f} "
                            f"mae {metrics['value_mae']:.6f} grad {metrics['grad_norm']:.3f}"
                        )
                        if metrics_file is None:
                            raise RuntimeError("Rank zero did not open its metrics file")
                        metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
                        metrics_file.flush()
                        if wandb_run is not None:
                            wandb_run.log(metrics, step=global_step)
                    running_loss = running_ce = running_soft_ce = running_huber = None
                    running_mse = running_mae = None
                    running_entropy = running_target_mean = running_pred_mean = None
                    running_target_std = running_pred_std = running_grad_norm = None
                    running_data_time = running_compute_time = 0.0
                    running_steps = running_samples = 0
                    log_start = time.time()

                checkpoint_due = args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0
                max_steps_reached = has_reached_max_steps(global_step, args.max_steps)
                save_max_steps_checkpoint = max_steps_reached and not args.no_save_final_checkpoint
                if checkpoint_due or save_max_steps_checkpoint:
                    next_epoch = epoch
                    next_step_in_epoch = step_in_epoch
                    if step_in_epoch == effective_steps_per_epoch:
                        next_epoch = epoch + 1
                        next_step_in_epoch = 0
                    train_progress = {
                        "next_epoch": next_epoch,
                        "next_step_in_epoch": next_step_in_epoch,
                    }
                    rng_states_by_rank = collect_rng_states(distributed_info)
                    if distributed_info.is_main_process:
                        if checkpoint_due:
                            save_checkpoint(
                                checkpoint_dir / f"checkpoint_step_{global_step}.pt",
                                model=model_base,
                                optimizer=optimizer,
                                epoch=epoch,
                                step=global_step,
                                run_config=run_config,
                                train_progress=train_progress,
                                rng_states_by_rank=rng_states_by_rank,
                            )
                        save_checkpoint(
                            checkpoint_dir / "latest.pt",
                            model=model_base,
                            optimizer=optimizer,
                            epoch=epoch,
                            step=global_step,
                            run_config=run_config,
                            train_progress=train_progress,
                            rng_states_by_rank=rng_states_by_rank,
                        )

                next_batch_start = time.time()

                if max_steps_reached:
                    stop_after_max_steps = True
                    break

            if stop_after_max_steps:
                if distributed_info.is_main_process:
                    stop_metrics = {
                        "epoch": epoch + 1,
                        "step_in_epoch": step_in_epoch,
                        "global_step": global_step,
                        "max_steps": args.max_steps,
                        "event": "max_steps_reached",
                    }
                    if metrics_file is None:
                        raise RuntimeError("Rank zero did not open its metrics file")
                    metrics_file.write(json.dumps(stop_metrics, sort_keys=True) + "\n")
                    metrics_file.flush()
                    if wandb_run is not None:
                        wandb_run.log(stop_metrics, step=global_step)
                    print(f"stopped after reaching max_steps={args.max_steps}")
                break

            if not args.no_save_final_checkpoint:
                rng_states_by_rank = collect_rng_states(distributed_info)
                if distributed_info.is_main_process:
                    save_checkpoint(
                        checkpoint_dir / "latest.pt",
                        model=model_base,
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        step=global_step,
                        run_config=run_config,
                        train_progress={"next_epoch": epoch + 1, "next_step_in_epoch": 0},
                        rng_states_by_rank=rng_states_by_rank,
                    )
            elapsed = time.time() - epoch_start
            if distributed_info.is_main_process:
                epoch_metrics = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "epoch_seconds": elapsed,
                    "epoch_steps": effective_steps_per_epoch,
                    "event": "epoch_end",
                }
                if metrics_file is None:
                    raise RuntimeError("Rank zero did not open its metrics file")
                metrics_file.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
                metrics_file.flush()
                if wandb_run is not None:
                    wandb_run.log(epoch_metrics, step=global_step)
                print(f"finished epoch {epoch + 1}/{args.epochs} in {elapsed:.1f}s")
            start_step_in_epoch = 0
    finally:
        if metrics_file is not None:
            metrics_file.close()
        if wandb_run is not None:
            wandb_run.finish()
        if distributed_info.enabled and dist.is_initialized():
            dist.destroy_process_group()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a paper VFE experiment.")
    parser.add_argument("--exp-name", required=True, dest="exp_name", help="Experiment name. Outputs go to outputs/<exp-name>; wandb run name uses the same value.")
    parser.add_argument("--data-split-config", default="configs/vfe_data_splits.json")
    parser.add_argument("--data-split-preset", default="icl_keep_true_all36_query80_ctx20_annotation_progress")
    parser.add_argument("--vfe-config", default=str(DEFAULT_VFE_PRESET_PATH))
    parser.add_argument("--vfe-preset", required=True)
    parser.add_argument(
        "--tasks-per-batch",
        type=int,
        default=1,
        help="Task count per rank for task_balanced_replacement; multiplied by samples_per_task for local batch size otherwise.",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=1,
        help="Samples per selected task for task_balanced_replacement; multiplied by tasks_per_batch for local batch size otherwise.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["query_frame_once", "task_balanced_replacement"],
        default="query_frame_once",
        help=(
            "query_frame_once shuffles every query frame without replacement and derives epoch length from the "
            "dataset; task_balanced_replacement preserves the legacy fixed-step task-balanced sampler."
        ),
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=1000,
        help="Required only by task_balanced_replacement; ignored by query_frame_once.",
    )
    parser.add_argument(
        "--pad-query-frame-once-last-batch",
        action="store_true",
        help=(
            "Pad the final query_frame_once global batch with a deterministic sample from the same epoch. "
            "This keeps every DDP rank's local batch size equal when the query-frame count is not divisible "
            "by world size or global batch size."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many total optimizer updates; 0 disables the limit and completes the requested epochs.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-nonfinite-grads", action="store_true")
    parser.add_argument("--error-on-nonfinite-grad", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--no-save-final-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=-1,
        help="Local rank supplied by torchrun; normally read from LOCAL_RANK automatically.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help=(
            "Optional W&B run ID for legacy checkpoints that do not store one. "
            "New checkpoints restore their W&B run ID automatically; an explicit mismatch is rejected."
        ),
    )
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help="Load only model weights from --resume and start a fresh optimizer/schedule.",
    )
    parser.add_argument(
        "--warmstart-base-checkpoint",
        default=None,
        help="Context-aware annotation checkpoint used as the primary composite model initialization.",
    )
    parser.add_argument(
        "--warmstart-visual-checkpoint",
        default=None,
        help="RECAP-512 checkpoint supplying the trained query visual_projector.",
    )
    parser.add_argument(
        "--warmstart-progress-checkpoint",
        default=None,
        help="Context-progress checkpoint supplying the trained progress encoder.",
    )
    parser.add_argument(
        "--allow-inexact-resume",
        action="store_true",
        help=(
            "Allow a legacy or configuration-mismatched checkpoint to resume without exact epoch/RNG continuity. "
            "Use only when the resulting repeated/skipped samples and changed stochastic stream are acceptable."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    train(parser.parse_args(argv))


if __name__ == "__main__":
    main()
