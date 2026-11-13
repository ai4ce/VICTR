from __future__ import annotations

import random
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from vfe.train import (
    DistributedInfo,
    capture_local_rng_state,
    clip_gradients_and_maybe_report,
    discretized_gaussian_targets,
    has_reached_max_steps,
    load_composite_warmstart_state,
    load_model_only_state,
    restore_local_rng_state,
    resolve_wandb_resume_id,
    save_checkpoint,
    saved_wandb_run_id,
    seed_process_rng,
    validate_resume_checkpoint,
    value_loss_components,
    wandb_run_id_from_checkpoint_config,
)

from vfe.model import VfeModelConfig


def _cpu_info() -> DistributedInfo:
    return DistributedInfo(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))


def test_gaussian_soft_value_loss_is_normalized_boundary_aware_and_differentiable() -> None:
    centers = torch.linspace(-1.0, 0.0, 201)
    values = torch.tensor([-1.0, -0.503, 0.0])
    targets = discretized_gaussian_targets(values, centers, sigma=0.01)

    torch.testing.assert_close(targets.sum(dim=-1), torch.ones(3))
    assert int(targets[0].argmax()) == 0
    assert int(targets[-1].argmax()) == 200
    # Gaussian probability outside the bounded support is accumulated into the
    # endpoint bins rather than discarded and renormalized away.
    assert float(targets[0, 0]) > 0.5
    assert float(targets[-1, -1]) > 0.5

    logits = torch.zeros(3, 201, requires_grad=True)
    components = value_loss_components(
        logits,
        value_bins=torch.tensor([0, 99, 200]),
        value_continuous=values,
        bin_values=centers,
        loss_type="gaussian_soft_ce_huber",
        soft_label_sigma=0.01,
        huber_beta=0.02,
        huber_weight=1.0,
    )
    torch.testing.assert_close(
        components["loss"], components["soft_ce"] + components["huber"]
    )
    components["loss"].backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())

    hard_logits = torch.randn(3, 201, requires_grad=True)
    hard_components = value_loss_components(
        hard_logits,
        value_bins=torch.tensor([0, 99, 200]),
        value_continuous=values,
        bin_values=centers,
        loss_type="hard_ce",
        soft_label_sigma=0.01,
        huber_beta=0.02,
        huber_weight=1.0,
    )
    expected_hard_ce = torch.nn.functional.cross_entropy(
        hard_logits.float(), torch.tensor([0, 99, 200])
    )
    torch.testing.assert_close(hard_components["loss"], expected_hard_ce)
    torch.testing.assert_close(hard_components["hard_ce"], expected_hard_ce)


def test_rng_state_round_trip() -> None:
    info = _cpu_info()
    seed_process_rng(123, rank=0)
    states = [capture_local_rng_state(info)]

    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    for _ in range(10):
        random.random()
        np.random.random()
        torch.rand(4)

    restore_local_rng_state(states, info, allow_inexact=False)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_resume_signature_rejects_changed_training_order() -> None:
    signature = {
        "sampling": {"mode": "query_frame_once", "world_size": 8, "global_batch_size": 160},
        "data": {"context_stride": 10},
    }
    checkpoint = {
        "train_progress": {"next_epoch": 1, "next_step_in_epoch": 227},
        "rng_states_by_rank": [{} for _ in range(8)],
        "run_config": {"resume_signature": signature},
    }
    validate_resume_checkpoint(checkpoint, signature, allow_inexact=False)

    changed = {
        "sampling": {"mode": "query_frame_once", "world_size": 4, "global_batch_size": 160},
        "data": {"context_stride": 20},
    }
    try:
        validate_resume_checkpoint(checkpoint, changed, allow_inexact=False)
    except ValueError as exc:
        message = str(exc)
        assert "sampling.world_size" in message
        assert "data.context_stride" in message
    else:
        raise AssertionError("Configuration-mismatched resume was not rejected")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_resume_checkpoint(checkpoint, changed, allow_inexact=True)
    assert caught and "Continuing because --allow-inexact-resume was set" in str(caught[0].message)


def test_legacy_checkpoint_requires_explicit_inexact_override() -> None:
    checkpoint = {"run_config": {}, "step": 10}
    try:
        validate_resume_checkpoint(checkpoint, {}, allow_inexact=False)
    except ValueError as exc:
        message = str(exc)
        assert "no train_progress" in message
        assert "no resume_signature" in message
        assert "no per-rank RNG state" in message
    else:
        raise AssertionError("Legacy inexact checkpoint was not rejected")


def test_model_only_warm_start_allows_only_new_history_embedding() -> None:
    class TinyEstimator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = VfeModelConfig(
                backbone="gemma3_native_style_270m_context",
                query_history_offsets=[60, 30, 0],
            )
            self.impl = nn.Module()
            self.impl.shared = nn.Linear(2, 2)
            self.impl.query_history_embedding = nn.Embedding(3, 2)

    source = TinyEstimator()
    source_state = {
        key: torch.full_like(value, 7)
        for key, value in source.state_dict().items()
        if key != "impl.query_history_embedding.weight"
    }
    target = TinyEstimator()
    initialized = load_model_only_state(
        target,
        {"model_state": source_state, "model_config": {}},
    )
    assert initialized == ["impl.query_history_embedding.weight"]
    torch.testing.assert_close(target.impl.shared.weight, torch.full_like(target.impl.shared.weight, 7))
    torch.testing.assert_close(
        target.impl.query_history_embedding.weight,
        torch.zeros_like(target.impl.query_history_embedding.weight),
    )

    unsafe_state = dict(source_state)
    del unsafe_state["impl.shared.bias"]
    try:
        load_model_only_state(
            TinyEstimator(),
            {"model_state": unsafe_state, "model_config": {}},
        )
    except RuntimeError as exc:
        assert "impl.shared.bias" in str(exc)
    else:
        raise AssertionError("Unsafe model-only missing key was accepted")


def test_composite_warm_start_reuses_native_640_dim_merger_for_both_recap_stages() -> None:
    _check_composite_warm_start(progress_enabled=True)


def test_composite_warm_start_without_progress_preserves_other_sources() -> None:
    _check_composite_warm_start(progress_enabled=False)


def _check_composite_warm_start(*, progress_enabled: bool) -> None:
    class TinyCompositeEstimator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.impl = nn.Module()
            self.impl.shared = nn.Linear(2, 2)
            self.impl.visual_projector = nn.Linear(2, 2)
            self.impl.context_progress_encoder = nn.Linear(1, 2) if progress_enabled else None
            self.impl.recap_spatial_merger_stage2 = nn.Linear(2, 2)
            self.impl.recap_spatial_merger_stage3 = nn.Linear(2, 2)

    target = TinyCompositeEstimator()
    base_state = {
        "impl.shared.weight": torch.full_like(target.impl.shared.weight, 1),
        "impl.shared.bias": torch.full_like(target.impl.shared.bias, 1),
        "impl.native_spatial_merger_stage2.weight": torch.full_like(
            target.impl.recap_spatial_merger_stage2.weight, 4
        ),
        "impl.native_spatial_merger_stage2.bias": torch.full_like(
            target.impl.recap_spatial_merger_stage2.bias, 4
        ),
    }
    visual_state = {
        name: torch.full_like(value, 2)
        for name, value in target.state_dict().items()
        if name.startswith("impl.visual_projector.")
    }
    progress_state = {
        name: torch.full_like(value, 3)
        for name, value in target.state_dict().items()
        if name.startswith("impl.context_progress_encoder.")
    }

    with TemporaryDirectory() as temp_dir:
        paths = [Path(temp_dir) / name for name in ("base.pt", "visual.pt", "progress.pt")]
        for path, state in zip(paths, (base_state, visual_state, progress_state), strict=True):
            torch.save({"model_state": state}, path)
        report = load_composite_warmstart_state(
            target,
            base_checkpoint_path=str(paths[0]),
            visual_checkpoint_path=str(paths[1]),
            progress_checkpoint_path=str(paths[2]) if progress_enabled else None,
        )

    torch.testing.assert_close(target.impl.shared.weight, torch.full_like(target.impl.shared.weight, 1))
    torch.testing.assert_close(
        target.impl.visual_projector.weight,
        torch.full_like(target.impl.visual_projector.weight, 2),
    )
    if progress_enabled:
        torch.testing.assert_close(
            target.impl.context_progress_encoder.weight,
            torch.full_like(target.impl.context_progress_encoder.weight, 3),
        )
    else:
        assert report["progress_checkpoint"] is None
        assert report["progress_parameters"] == []
    for merger in (
        target.impl.recap_spatial_merger_stage2,
        target.impl.recap_spatial_merger_stage3,
    ):
        torch.testing.assert_close(merger.weight, torch.full_like(merger.weight, 4))
        torch.testing.assert_close(merger.bias, torch.full_like(merger.bias, 4))
    assert len(report["shared_stage2_from_base_parameters"]) == 2
    assert len(report["shared_stage3_from_base_parameters"]) == 2


def test_resume_signature_fills_defaults_for_new_context_model_fields() -> None:
    legacy_signature = {
        "sampling": {"mode": "query_frame_once"},
        "model_config": {"backbone": "gemma3_native_style_270m"},
    }
    checkpoint = {
        "train_progress": {"next_epoch": 2, "next_step_in_epoch": 1504},
        "rng_states_by_rank": [{}, {}],
        "run_config": {"resume_signature": legacy_signature},
    }
    current_signature = {
        "sampling": {"mode": "query_frame_once"},
        "model_config": {
            "backbone": "gemma3_native_style_270m",
            "context_progress_enabled": False,
            "context_slot_embedding_enabled": True,
        },
        "loss": {
            "type": "hard_ce",
            "soft_label_sigma": 0.01,
            "huber_beta": 0.02,
            "huber_weight": 1.0,
        },
    }
    validate_resume_checkpoint(checkpoint, current_signature, allow_inexact=False)

    soft_loss_signature = {
        **current_signature,
        "loss": {
            **current_signature["loss"],
            "type": "gaussian_soft_ce_huber",
        },
    }
    try:
        validate_resume_checkpoint(checkpoint, soft_loss_signature, allow_inexact=False)
    except ValueError as exc:
        assert "loss.type" in str(exc)
    else:
        raise AssertionError("A checkpoint using a different value loss was incorrectly accepted")

    current_signature["model_config"]["context_progress_enabled"] = True
    try:
        validate_resume_checkpoint(checkpoint, current_signature, allow_inexact=False)
    except ValueError as exc:
        assert "model_config.context_progress_enabled" in str(exc)
    else:
        raise AssertionError("A non-default new model field was incorrectly accepted")


def test_max_steps_is_a_total_cap() -> None:
    assert not has_reached_max_steps(global_step=2999, max_steps=3000)
    assert has_reached_max_steps(global_step=3000, max_steps=3000)
    assert has_reached_max_steps(global_step=3001, max_steps=3000)
    assert not has_reached_max_steps(global_step=3000, max_steps=0)


def test_wandb_run_id_defaults_to_checkpoint_and_rejects_mismatch() -> None:
    assert saved_wandb_run_id({"wandb_run_id": "run-new"}) == "run-new"
    assert saved_wandb_run_id({"run_config": {"wandb_run_id": "run-legacy"}}) == "run-legacy"
    assert resolve_wandb_resume_id("run-new", None) == "run-new"
    assert resolve_wandb_resume_id(None, "run-legacy") == "run-legacy"
    assert resolve_wandb_resume_id("run-new", "run-new") == "run-new"
    try:
        resolve_wandb_resume_id("run-new", "wrong-run")
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("Mismatched explicit W&B run ID was not rejected")


def test_wandb_run_id_comes_from_single_experiment_config() -> None:
    with TemporaryDirectory() as temp_dir:
        experiment_dir = Path(temp_dir) / "experiment"
        checkpoint_dir = experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "latest.pt"
        checkpoint_path.touch()
        (experiment_dir / "run_config.json").write_text(
            '{"wandb_run_id": "config-run-id"}\n',
            encoding="utf-8",
        )
        assert wandb_run_id_from_checkpoint_config(checkpoint_path) == "config-run-id"


class _Config:
    def to_dict(self) -> dict[str, str]:
        return {"name": "test"}


class _Model(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 1)
        self.config = _Config()


def test_checkpoint_is_atomic_and_weights_only_readable() -> None:
    model = _Model()
    optimizer = torch.optim.AdamW(model.parameters())
    rng_states = [capture_local_rng_state(_cpu_info())]
    with TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "latest.pt"
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=12,
            run_config={"resume_signature": {"format_version": 1}, "wandb_run_id": "test-run-id"},
            train_progress={"next_epoch": 1, "next_step_in_epoch": 12},
            rng_states_by_rank=rng_states,
        )
        assert checkpoint_path.is_file()
        assert not checkpoint_path.with_suffix(".pt.tmp").exists()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["checkpoint_format_version"] == 2
        assert checkpoint["step"] == 12
        assert "wandb_run_id" not in checkpoint
        assert "wandb_run_id" not in checkpoint["run_config"]
        assert len(checkpoint["rng_states_by_rank"]) == 1


def test_gradient_clip_uses_total_norm_for_fast_nonfinite_check() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    with patch("vfe.train.nonfinite_gradient_report") as detailed_report:
        grad_norm, report = clip_gradients_and_maybe_report(
            model,
            max_norm=1.0,
            check_nonfinite=True,
        )
    detailed_report.assert_not_called()
    assert torch.allclose(grad_norm, torch.tensor(5.0))
    assert report is None
    assert torch.allclose(model.weight.grad.norm(), torch.tensor(1.0))

    model.weight.grad = torch.tensor([[float("nan"), 1.0]])
    grad_norm, report = clip_gradients_and_maybe_report(
        model,
        max_norm=0.0,
        check_nonfinite=True,
    )
    assert not bool(torch.isfinite(grad_norm).item())
    assert report is not None
    assert report["nonfinite_parameters"] == 1
    assert report["nonfinite_elements"] == 1


def test_gradient_clip_matches_torch_clip_grad_norm() -> None:
    expected = torch.nn.Linear(3, 2)
    actual = torch.nn.Linear(3, 2)
    actual.load_state_dict(expected.state_dict())
    expected.weight.grad = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected.bias.grad = torch.tensor([2.0, -3.0])
    actual.weight.grad = expected.weight.grad.clone()
    actual.bias.grad = expected.bias.grad.clone()

    expected_norm = torch.nn.utils.clip_grad_norm_(expected.parameters(), max_norm=0.75)
    actual_norm, report = clip_gradients_and_maybe_report(
        actual,
        max_norm=0.75,
        check_nonfinite=True,
    )
    assert report is None
    assert torch.equal(actual_norm, expected_norm)
    assert torch.equal(actual.weight.grad, expected.weight.grad)
    assert torch.equal(actual.bias.grad, expected.bias.grad)


if __name__ == "__main__":
    test_gaussian_soft_value_loss_is_normalized_boundary_aware_and_differentiable()
    test_rng_state_round_trip()
    test_resume_signature_rejects_changed_training_order()
    test_legacy_checkpoint_requires_explicit_inexact_override()
    test_model_only_warm_start_allows_only_new_history_embedding()
    test_composite_warm_start_reuses_native_640_dim_merger_for_both_recap_stages()
    test_resume_signature_fills_defaults_for_new_context_model_fields()
    test_max_steps_is_a_total_cap()
    test_wandb_run_id_defaults_to_checkpoint_and_rejects_mismatch()
    test_wandb_run_id_comes_from_single_experiment_config()
    test_checkpoint_is_atomic_and_weights_only_readable()
    test_gradient_clip_uses_total_norm_for_fast_nonfinite_check()
    test_gradient_clip_matches_torch_clip_grad_norm()
