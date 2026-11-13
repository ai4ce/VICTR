from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from vfe.data import collate_vfe_batch, end_anchored_context_frame_indices
from vfe.data_splits import GROUND_TRUTH_VERSION_FIELD
from vfe.labels import LEGACY_GROUND_TRUTH_VERSION
from vfe.model import (
    Qwen2StyleSpatialPatchMerger,
    SiglipGemmaVFE,
    build_context_block_causal_mask,
    build_independent_context_block_mask,
)
from vfe.native_features import (
    CONTEXT_HF_VISION_FEATURE_TYPE,
    CONTEXT_NATIVE_FEATURE_FORMAT_VERSION,
    CONTEXT_NATIVE_FEATURE_MANIFEST,
    CONTEXT_NATIVE_FEATURE_TYPE,
    ContextNativeFeatureStore,
    NATIVE_PROJECTOR_INPUT_STAGE,
    RecapQwen2DetailFeatureExtractor,
    RECAP_VISION_PATCH_STAGE,
    RECAP_QWEN2_DETAIL_STAGE,
    context_feature_key,
    precompute_context_native_features,
)


def test_end_anchored_context_indices_keep_terminal_frame() -> None:
    assert end_anchored_context_frame_indices(71, stride=10) == list(range(0, 71, 10))
    assert end_anchored_context_frame_indices(78, stride=10) == list(range(7, 78, 10))


def test_independent_context_block_mask_has_exact_retrieval_topology() -> None:
    # Two 3-token context timesteps, a 3-token current block (the last token is
    # padded), and one value token.
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 1]], dtype=torch.bool)
    block_mask = build_independent_context_block_mask(
        valid,
        context_token_count=6,
        context_timestep_count=2,
    )

    query_indices = torch.arange(10).view(10, 1)
    key_indices = torch.arange(10).view(1, 10)
    allowed = block_mask.mask_mod(
        torch.zeros((10, 10), dtype=torch.long),
        torch.tensor(0),
        query_indices,
        key_indices,
    )

    expected = torch.zeros((10, 10), dtype=torch.bool)
    expected[0:3, 0:3] = True
    expected[3:6, 3:6] = True
    expected[6:8, 0:8] = True
    expected[9, 0:8] = True
    expected[9, 9] = True
    assert torch.equal(allowed, expected)


def test_context_block_causal_mask_has_bidirectional_timestep_diagonal() -> None:
    # Two 3-token context timesteps, a 3-token current block (the last token is
    # padded), and one value token.
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 1]], dtype=torch.bool)
    block_mask = build_context_block_causal_mask(
        valid,
        context_token_count=6,
        context_timestep_count=2,
    )

    query_indices = torch.arange(10).view(10, 1)
    key_indices = torch.arange(10).view(1, 10)
    allowed = block_mask.mask_mod(
        torch.zeros((10, 10), dtype=torch.long),
        torch.tensor(0),
        query_indices,
        key_indices,
    )

    expected = torch.zeros((10, 10), dtype=torch.bool)
    expected[0:3, 0:3] = True
    expected[3:6, 0:6] = True
    expected[6:8, 0:8] = True
    expected[9, 0:8] = True
    expected[9, 9] = True
    assert torch.equal(allowed, expected)


def test_feature_store_validates_manifest_and_loads_bfloat16_shard() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        shard = root / "task_001" / "demo_9.pt"
        shard.parent.mkdir(parents=True)
        torch.save(
            {
                "features": torch.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4).to(torch.bfloat16),
                "frame_indices": torch.tensor([7, 17], dtype=torch.int32),
            },
            shard,
        )
        key = context_feature_key(1, "demo_9")
        manifest = {
            "format_version": CONTEXT_NATIVE_FEATURE_FORMAT_VERSION,
            "feature_type": CONTEXT_NATIVE_FEATURE_TYPE,
            "feature_stage": NATIVE_PROJECTOR_INPUT_STAGE,
            "native_vision_model_name": "native-model",
            "obs_keys": ["agentview_rgb", "eye_in_hand_rgb"],
            "context_stride": 10,
            "context_max_original_timesteps": 520,
            "selection": "end_anchored",
            "tokens_per_image": 3,
            "feature_dim": 4,
            "entries": {key: {"path": "task_001/demo_9.pt"}},
        }
        (root / CONTEXT_NATIVE_FEATURE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

        store = ContextNativeFeatureStore(
            root,
            native_vision_model_name="native-model",
            obs_keys=["agentview_rgb", "eye_in_hand_rgb"],
            context_stride=10,
            context_max_original_timesteps=520,
            expected_tokens_per_image=3,
        )
        store.require_entries({1: ["demo_9"]})
        payload = store.load(1, "demo_9")
        assert payload["features"].dtype == torch.bfloat16
        assert tuple(payload["features"].shape) == (2, 2, 3, 4)
        assert payload["frame_indices"].tolist() == [7, 17]


def test_feature_store_bounds_mmap_payload_cache_with_lru_eviction() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        entries = {}
        for demo_index in range(3):
            demo_id = f"demo_{demo_index}"
            relative_path = Path("task_001") / f"{demo_id}.pt"
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "features": torch.full(
                        (1, 1, 1, 2), demo_index, dtype=torch.bfloat16
                    ),
                    "frame_indices": torch.tensor([demo_index]),
                },
                path,
            )
            entries[context_feature_key(1, demo_id)] = {"path": str(relative_path)}
        manifest = {
            "format_version": CONTEXT_NATIVE_FEATURE_FORMAT_VERSION,
            "feature_type": CONTEXT_NATIVE_FEATURE_TYPE,
            "feature_stage": NATIVE_PROJECTOR_INPUT_STAGE,
            "native_vision_model_name": "native-model",
            "obs_keys": ["agentview_rgb"],
            "context_stride": 10,
            "context_max_original_timesteps": 520,
            "selection": "end_anchored",
            "tokens_per_image": 1,
            "feature_dim": 2,
            "entries": entries,
        }
        (root / CONTEXT_NATIVE_FEATURE_MANIFEST).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        store = ContextNativeFeatureStore(
            root,
            native_vision_model_name="native-model",
            obs_keys=["agentview_rgb"],
            context_stride=10,
            context_max_original_timesteps=520,
            expected_tokens_per_image=1,
            max_cached_payloads=2,
        )
        store.load(1, "demo_0")
        store.load(1, "demo_1")
        store.load(1, "demo_0")
        store.load(1, "demo_2")
        assert list(store._payload_cache) == [
            context_feature_key(1, "demo_0"),
            context_feature_key(1, "demo_2"),
        ]


def _cached_item(num_slots: int, slot_indices: list[int], offset: int) -> dict:
    features = torch.arange(num_slots * 2 * 3 * 4).reshape(num_slots, 2, 3, 4).to(torch.bfloat16)
    return {
        "query_image": torch.zeros(2, 3, 8, 8),
        "query_proprio": torch.zeros(0),
        "language": "instruction",
        "value_bin": 1,
        "value_continuous": -0.5,
        "task_id": offset,
        "target_demo_id": f"demo_{offset}",
        "target_source_hdf5": "source.hdf5",
        "obs_keys": ["agentview_rgb", "eye_in_hand_rgb"],
        "proprio_keys": [],
        "frame_idx": 0,
        "context_native_features": features + offset,
        "context_feature_slot_indices": torch.tensor(slot_indices, dtype=torch.long),
        "context_proprio": torch.zeros(5, 0),
        "context_progress": torch.linspace(-1.0, 0.0, 5),
        "context_valid_mask": torch.tensor([index in slot_indices for index in range(5)]),
        "context_demo_id": f"context_{offset}",
    }


def test_collate_packs_only_valid_context_features_and_positions() -> None:
    batch = collate_vfe_batch(
        [
            _cached_item(2, [3, 4], 0),
            _cached_item(1, [4], 1),
        ]
    )
    assert tuple(batch["context_native_features"].shape) == (6, 3, 4)
    assert batch["context_feature_positions"].tolist() == [
        [0, 3, 0],
        [0, 3, 1],
        [0, 4, 0],
        [0, 4, 1],
        [1, 4, 0],
        [1, 4, 1],
    ]
    assert tuple(batch["context_valid_mask"].shape) == (2, 5)
    assert tuple(batch["context_progress"].shape) == (2, 5)
    torch.testing.assert_close(batch["context_progress"][0], torch.linspace(-1.0, 0.0, 5))


def test_context_progress_adds_one_masked_token_per_timestep() -> None:
    class ProgressEncoder(nn.Module):
        def forward(self, progress, batch_size, device):
            assert tuple(progress.shape) == (batch_size, 1)
            return progress.to(device).expand(batch_size, 4)

    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        context_max_timesteps=2,
        context_views=1,
        proprio_dim=0,
        context_image_chunk_size=1,
        context_progress_enabled=True,
        use_source_embeddings=True,
    )
    model.vision_tower = nn.Linear(1, 1, bias=False)
    model.language_model = nn.Linear(1, 1, bias=False)
    model.native_tokens_per_image = 1
    model.native_context_tokens_per_image = 1
    model.native_vision_tower_hidden = 2
    model.uses_direct_native_270m_projector = True
    model.uses_qwen2_native_270m_merger = False
    model.uses_dualres_native_270m_merger = False
    model.visual_projector = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 4))
    model.value_query = nn.Parameter(torch.zeros(1, 1, 4))
    model.proprio_encoder = nn.Identity()
    model.context_progress_encoder = ProgressEncoder()
    model.segment_embedding = nn.Embedding(4, 4)
    model.modality_embedding = nn.Embedding(4, 4)
    model.camera_embedding = nn.Embedding(3, 4)
    model.context_slot_embedding = nn.Embedding(2, 4)
    for embedding in (
        model.segment_embedding,
        model.modality_embedding,
        model.camera_embedding,
        model.context_slot_embedding,
    ):
        nn.init.zeros_(embedding.weight)

    tokens, mask = model._encode_native_style_270m_context_tokens(
        None,
        torch.zeros(1, 2, 0),
        torch.tensor([[False, True]]),
        torch.tensor([[0.0, -0.25]]),
        context_native_features=torch.randn(1, 1, 2),
        context_feature_positions=torch.tensor([[0, 1, 0]]),
    )

    assert tuple(tokens.shape) == (1, 2 * 3, 4)
    assert mask.tolist() == [[False, False, False, True, True, True]]
    torch.testing.assert_close(tokens.reshape(1, 2, 3, 4)[0, 1, 2], torch.full((4,), -0.25))


def test_cached_native_features_backpropagate_through_trainable_connector() -> None:
    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        context_max_timesteps=3,
        context_views=2,
        proprio_dim=0,
        context_image_chunk_size=2,
        use_source_embeddings=True,
    )
    model.vision_tower = nn.Linear(1, 1, bias=False)
    model.language_model = nn.Linear(1, 1, bias=False)
    model.native_tokens_per_image = 3
    model.native_context_tokens_per_image = 3
    model.native_vision_tower_hidden = 4
    model.uses_direct_native_270m_projector = True
    model.uses_qwen2_native_270m_merger = False
    model.uses_dualres_native_270m_merger = False
    model.visual_projector = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 5))
    model.value_query = nn.Parameter(torch.zeros(1, 1, 5))
    model.proprio_encoder = nn.Identity()
    model.segment_embedding = nn.Embedding(4, 5)
    model.modality_embedding = nn.Embedding(4, 5)
    model.camera_embedding = nn.Embedding(3, 5)
    model.context_slot_embedding = nn.Embedding(3, 5)

    features = torch.randn(4, 3, 4, dtype=torch.bfloat16)
    positions = torch.tensor(
        [
            [0, 1, 0],
            [0, 1, 1],
            [0, 2, 0],
            [0, 2, 1],
        ],
        dtype=torch.long,
    )
    tokens, mask = model._encode_native_style_270m_context_tokens(
        None,
        torch.zeros(1, 3, 0),
        torch.tensor([[False, True, True]]),
        context_native_features=features,
        context_feature_positions=positions,
    )
    assert tuple(tokens.shape) == (1, 3 * (2 * 3 + 1), 5)
    assert int(mask.sum()) == 2 * (2 * 3 + 1)
    tokens.sum().backward()
    connector_weight = model.visual_projector[1].weight
    assert connector_weight.grad is not None
    assert torch.isfinite(connector_weight.grad).all()


def test_original_native_projector_chain_is_still_supported() -> None:
    class FrozenNativeProjector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mm_input_projection_weight = nn.Parameter(torch.randn(4, 6), requires_grad=False)

    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        context_max_timesteps=1,
        context_views=1,
        proprio_dim=0,
        context_image_chunk_size=1,
        use_source_embeddings=True,
    )
    model.vision_tower = nn.Linear(1, 1, bias=False)
    model.language_model = nn.Linear(1, 1, bias=False)
    model.native_vision_projector = FrozenNativeProjector()
    model.native_tokens_per_image = 3
    model.native_context_tokens_per_image = 3
    model.native_vision_tower_hidden = 4
    model.native_vision_hidden = 6
    model.uses_direct_native_270m_projector = False
    model.uses_qwen2_native_270m_merger = False
    model.uses_dualres_native_270m_merger = False
    model.visual_projector = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, 5))
    model.value_query = nn.Parameter(torch.zeros(1, 1, 5))
    model.proprio_encoder = nn.Identity()
    model.segment_embedding = nn.Embedding(4, 5)
    model.modality_embedding = nn.Embedding(4, 5)
    model.camera_embedding = nn.Embedding(3, 5)
    model.context_slot_embedding = nn.Embedding(1, 5)

    features = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    tokens, _ = model._encode_native_style_270m_context_tokens(
        None,
        torch.zeros(1, 1, 0),
        torch.tensor([[True]]),
        context_native_features=features,
        context_feature_positions=torch.tensor([[0, 0, 0]]),
    )
    tokens.sum().backward()
    assert model.visual_projector[1].weight.grad is not None


def test_native_style_pooling_keeps_vision_channels_and_reduces_patch_grid() -> None:
    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.native_vision_pooler = nn.AvgPool2d(kernel_size=2, stride=2)
    model.native_vision_patches_per_image = 4
    model.native_vision_tower_hidden = 3

    features = torch.arange(2 * 16 * 3, dtype=torch.float32).reshape(2, 16, 3).requires_grad_()
    pooled = model._pool_native_vision_features(features)

    assert tuple(pooled.shape) == (2, 4, 3)
    assert sum(parameter.numel() for parameter in model.native_vision_pooler.parameters()) == 0
    pooled.sum().backward()
    assert features.grad is not None


def test_qwen2_style_merger_groups_true_spatial_neighbors() -> None:
    merger = Qwen2StyleSpatialPatchMerger(context_dim=1, output_dim=1, spatial_merge_size=2)
    merger.norm = nn.Identity()
    merger.mlp = nn.Identity()
    grid = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)

    neighborhoods = merger(grid)

    assert tuple(neighborhoods.shape) == (1, 4, 4)
    assert neighborhoods[0].tolist() == [
        [0.0, 1.0, 4.0, 5.0],
        [2.0, 3.0, 6.0, 7.0],
        [8.0, 9.0, 12.0, 13.0],
        [10.0, 11.0, 14.0, 15.0],
    ]


def test_qwen2_style_merger_reduces_256_tokens_to_64_and_backpropagates() -> None:
    merger = Qwen2StyleSpatialPatchMerger(context_dim=3, output_dim=3, spatial_merge_size=2)
    features = torch.randn(2, 256, 3, requires_grad=True)

    merged = merger(features)

    assert tuple(merged.shape) == (2, 64, 3)
    merged.sum().backward()
    assert features.grad is not None
    assert merger.mlp[0].weight.grad is not None


def test_dualres_two_stage_merger_produces_256_plus_16_tokens() -> None:
    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.uses_qwen2_native_270m_merger = False
    model.uses_dualres_native_270m_merger = True
    model.native_spatial_merger = Qwen2StyleSpatialPatchMerger(3, 5, spatial_merge_size=2)
    model.native_spatial_merger_stage2 = Qwen2StyleSpatialPatchMerger(5, 5, spatial_merge_size=2)
    model.visual_projector = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, 5))
    pooled = torch.randn(2, 256, 3, requires_grad=True)

    query_tokens = model._project_native_pooled_features(pooled)
    context_tokens = model._project_native_pooled_features(pooled, alignment_only=True)

    assert tuple(query_tokens.shape) == (2, 272, 5)
    assert tuple(context_tokens.shape) == (2, 16, 5)
    query_tokens.sum().backward()
    assert pooled.grad is not None
    assert model.visual_projector[1].weight.grad is not None
    assert model.native_spatial_merger.mlp[0].weight.grad is not None
    assert model.native_spatial_merger_stage2.mlp[0].weight.grad is not None


def test_recap_style_dualres_combines_729_detail_and_16_alignment_tokens() -> None:
    class NativeProcessor:
        def __call__(self, *, images, return_tensors, do_pan_and_scan, device=None):
            assert return_tensors == "pt"
            assert not do_pan_and_scan
            return {"pixel_values": images.float()}

    class RecapProcessor:
        def __call__(self, *, images, return_tensors):
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros(len(images), 3, 4, 4)}

    class NativeTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, *, pixel_values, return_dict, **_kwargs):
            return SimpleNamespace(
                last_hidden_state=self.scale * torch.ones(pixel_values.shape[0], 4096, 3)
            )

    class RecapTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, *, pixel_values, return_dict, **_kwargs):
            return SimpleNamespace(
                last_hidden_state=self.scale * torch.ones(pixel_values.shape[0], 729, 3)
            )

    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(freeze_backbone=False, freeze_vision=True)
    model.language_model = nn.Linear(1, 5, bias=False)
    model.vision_tower = NativeTower()
    model.recap_vision_tower = RecapTower()
    model.image_processor = NativeProcessor()
    model.recap_image_processor = RecapProcessor()
    model.uses_direct_native_270m_projector = True
    model.uses_qwen2_native_270m_merger = False
    model.uses_dualres_native_270m_merger = True
    model.uses_recap_style_dualres = True
    model.native_vision_projector = None
    model.native_vision_pooler = nn.AvgPool2d(kernel_size=4, stride=4)
    model.native_vision_patches_per_image = 64
    model.native_vision_tower_hidden = 3
    model.native_pooled_tokens_per_image = 256
    model.recap_tokens_per_image = 729
    model.native_context_tokens_per_image = 16
    model.native_tokens_per_image = 745
    model.visual_projector = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, 5))
    model.native_spatial_merger = Qwen2StyleSpatialPatchMerger(3, 5, spatial_merge_size=2)
    model.native_spatial_merger_stage2 = Qwen2StyleSpatialPatchMerger(5, 5, spatial_merge_size=2)

    tokens = model._recap_style_dualres_image_features(torch.zeros(2, 3, 4, 4))

    assert tuple(tokens.shape) == (2, 745, 5)
    tokens.sum().backward()
    assert model.vision_tower.scale.grad is None
    assert model.recap_vision_tower.scale.grad is None
    assert model.visual_projector[1].weight.grad is not None
    assert model.native_spatial_merger.mlp[0].weight.grad is not None
    assert model.native_spatial_merger_stage2.mlp[0].weight.grad is not None


def test_recap_qwen2_512_dualres_derives_256_plus_16_query_tokens_from_siglip() -> None:
    class TensorProcessor:
        def __init__(self) -> None:
            self.images = None
            self.device = None

        def __call__(self, *, images, return_tensors, device):
            assert return_tensors == "pt"
            self.images = images.detach().clone()
            self.device = device
            return {"pixel_values": images.float()}

    class NativeTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, **_kwargs):
            raise AssertionError("Query alignment must not invoke the native vision tower")

    class RecapTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, *, pixel_values, return_dict, **_kwargs):
            return SimpleNamespace(
                last_hidden_state=self.scale
                * torch.ones(pixel_values.shape[0], 1024, 3, device=pixel_values.device)
            )

    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        freeze_backbone=False,
        freeze_vision=True,
        context_max_timesteps=1,
        context_views=3,
        proprio_dim=0,
        context_image_chunk_size=3,
        context_progress_enabled=False,
        use_source_embeddings=True,
    )
    model.language_model = nn.Linear(1, 5, bias=False)
    model.vision_tower = NativeTower()
    model.recap_vision_tower = RecapTower()
    model.image_processor = None
    model.recap_image_processor = TensorProcessor()
    model.uses_direct_native_270m_projector = True
    model.uses_qwen2_native_270m_merger = False
    model.uses_dualres_native_270m_merger = True
    model.uses_recap_style_dualres = True
    model.uses_recap_qwen2_512_dualres = True
    model.native_vision_projector = None
    model.native_vision_pooler = nn.AvgPool2d(kernel_size=4, stride=4)
    model.native_vision_patches_per_image = 64
    model.native_vision_tower_hidden = 3
    model.native_pooled_tokens_per_image = 256
    model.native_context_tokens_per_image = 16
    model.recap_tokens_per_image = 1024
    model.recap_detail_tokens_per_image = 256
    model.recap_alignment_tokens_per_image = 16
    model.native_tokens_per_image = 272
    model.native_spatial_merger = Qwen2StyleSpatialPatchMerger(3, 5, spatial_merge_size=2)
    model.native_spatial_merger_stage2 = Qwen2StyleSpatialPatchMerger(5, 5, spatial_merge_size=2)
    model.visual_projector = Qwen2StyleSpatialPatchMerger(3, 5, spatial_merge_size=2)
    model.recap_spatial_merger_stage2 = Qwen2StyleSpatialPatchMerger(5, 5, spatial_merge_size=2)
    model.recap_spatial_merger_stage3 = Qwen2StyleSpatialPatchMerger(5, 5, spatial_merge_size=2)
    model.value_query = nn.Parameter(torch.zeros(1, 1, 5))
    model.proprio_encoder = nn.Identity()
    model.segment_embedding = nn.Embedding(4, 5)
    model.modality_embedding = nn.Embedding(4, 5)
    model.camera_embedding = nn.Embedding(3, 5)
    model.context_slot_embedding = nn.Embedding(1, 5)
    with torch.no_grad():
        model.segment_embedding.weight.zero_()
        model.modality_embedding.weight.zero_()
        model.camera_embedding.weight.zero_()
        model.context_slot_embedding.weight.zero_()

    query_images = torch.zeros(1, 3, 3, 4, 4)
    tokens = model._encode_native_style_270m_visual_tokens(query_images)
    context_tokens, context_mask = model._encode_native_style_270m_context_tokens(
        query_images.unsqueeze(1),
        torch.zeros(1, 1, 0),
        torch.tensor([[True]]),
    )

    assert tuple(tokens.shape) == (1, 3 * 272, 5)
    query_alignment = tokens.reshape(1, 3, 272, 5)[:, :, -16:].reshape(1, 48, 5)
    torch.testing.assert_close(context_tokens[:, :48], query_alignment)
    assert int(context_mask.sum()) == 49
    assert tuple(model.recap_image_processor.images.shape) == (3, 3, 4, 4)
    assert model.recap_image_processor.images.dtype == torch.uint8
    assert model.recap_image_processor.device == next(model.recap_vision_tower.parameters()).device
    tokens.sum().backward()
    assert model.vision_tower.scale.grad is None
    assert model.recap_vision_tower.scale.grad is None
    assert model.visual_projector.mlp[0].weight.grad is not None
    assert model.recap_spatial_merger_stage2.mlp[0].weight.grad is not None
    assert model.recap_spatial_merger_stage3.mlp[0].weight.grad is not None


def test_recap_qwen2_cached_context_uses_exact_query_alignment_mergers() -> None:
    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        context_max_timesteps=1,
        context_views=1,
        proprio_dim=0,
        context_image_chunk_size=1,
        use_source_embeddings=True,
        context_progress_enabled=False,
    )
    model.vision_tower = nn.Linear(1, 1, bias=False)
    model.language_model = nn.Linear(1, 1, bias=False)
    model.native_tokens_per_image = 17
    model.native_context_tokens_per_image = 1
    model.native_pooled_tokens_per_image = 16
    model.native_vision_tower_hidden = 2
    model.recap_tokens_per_image = 64
    model.recap_vision_hidden = 2
    model.recap_detail_tokens_per_image = 16
    model.recap_alignment_tokens_per_image = 1
    model.uses_direct_native_270m_projector = True
    model.uses_qwen2_native_270m_merger = False
    model.uses_dualres_native_270m_merger = True
    model.uses_recap_qwen2_512_dualres = True
    model.native_spatial_merger = None
    model.native_spatial_merger_stage2 = None
    model.visual_projector = Qwen2StyleSpatialPatchMerger(2, 3, spatial_merge_size=2)
    model.recap_spatial_merger_stage2 = Qwen2StyleSpatialPatchMerger(3, 3, spatial_merge_size=2)
    model.recap_spatial_merger_stage3 = Qwen2StyleSpatialPatchMerger(3, 3, spatial_merge_size=2)
    model.value_query = nn.Parameter(torch.zeros(1, 1, 3))
    model.proprio_encoder = nn.Identity()
    model.segment_embedding = nn.Embedding(4, 3)
    model.modality_embedding = nn.Embedding(4, 3)
    model.camera_embedding = nn.Embedding(3, 3)
    model.context_slot_embedding = nn.Embedding(1, 3)
    with torch.no_grad():
        model.segment_embedding.weight.zero_()
        model.modality_embedding.weight.zero_()
        model.camera_embedding.weight.zero_()
        model.context_slot_embedding.weight.zero_()

    patch_features = torch.randn(1, 64, 2, dtype=torch.bfloat16)
    expected_alignment = model._recap_qwen2_alignment_from_patch_features(patch_features)
    tokens, mask = model._encode_native_style_270m_context_tokens(
        None,
        torch.zeros(1, 1, 0),
        torch.tensor([[True]]),
        context_native_features=patch_features,
        context_feature_positions=torch.tensor([[0, 0, 0]]),
    )

    torch.testing.assert_close(tokens[:, :1], expected_alignment)
    assert int(mask.sum()) == 2
    tokens.sum().backward()
    assert model.visual_projector.mlp[0].weight.grad is not None
    assert model.recap_spatial_merger_stage2.mlp[0].weight.grad is not None
    assert model.recap_spatial_merger_stage3.mlp[0].weight.grad is not None
    assert model.native_spatial_merger is None

    # A 256-token detail cache must bypass the frozen first merger while the
    # two downstream alignment mergers remain trainable.
    model.visual_projector.zero_grad(set_to_none=True)
    model.recap_spatial_merger_stage2.zero_grad(set_to_none=True)
    model.recap_spatial_merger_stage3.zero_grad(set_to_none=True)
    detail_features = model.visual_projector(patch_features.float()).detach().to(torch.bfloat16)
    expected_from_detail = model._recap_qwen2_alignment_from_detail_tokens(detail_features)
    detail_tokens, detail_mask = model._encode_native_style_270m_context_tokens(
        None,
        torch.zeros(1, 1, 0),
        torch.tensor([[True]]),
        context_native_features=detail_features,
        context_feature_positions=torch.tensor([[0, 0, 0]]),
    )
    torch.testing.assert_close(detail_tokens[:, :1], expected_from_detail)
    assert int(detail_mask.sum()) == 2
    detail_tokens.sum().backward()
    assert model.visual_projector.mlp[0].weight.grad is None
    assert model.recap_spatial_merger_stage2.mlp[0].weight.grad is not None
    assert model.recap_spatial_merger_stage3.mlp[0].weight.grad is not None


def test_recap_detail_extractor_loads_only_checkpointed_first_merger() -> None:
    merger = Qwen2StyleSpatialPatchMerger(2, 3, spatial_merge_size=2)
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "checkpoint.pt"
        torch.save(
            {
                "model_state": {
                    f"impl.visual_projector.{key}": value
                    for key, value in merger.state_dict().items()
                }
            },
            checkpoint_path,
        )
        extractor = RecapQwen2DetailFeatureExtractor(
            checkpoint_path,
            state_prefix="impl.visual_projector",
            device=torch.device("cpu"),
            source_tokens_per_image=64,
            source_feature_dim=2,
            resolved_revision="test-revision",
        )
        inputs = torch.randn(2, 64, 2, dtype=torch.bfloat16)
        outputs = extractor.encode(inputs, feature_stage=RECAP_QWEN2_DETAIL_STAGE)

    expected = merger(inputs.float())
    torch.testing.assert_close(outputs, expected)
    assert tuple(outputs.shape) == (2, 16, 3)
    assert len(extractor.projector_sha256) == 64
    assert all(not parameter.requires_grad for parameter in extractor.merger.parameters())


def test_recap_context_feature_manifest_rejects_old_native_cache() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        manifest = {
            "format_version": CONTEXT_NATIVE_FEATURE_FORMAT_VERSION,
            "feature_type": CONTEXT_NATIVE_FEATURE_TYPE,
            "feature_stage": NATIVE_PROJECTOR_INPUT_STAGE,
            "native_vision_model_name": "google/gemma-3-4b-pt",
            "native_vision_model_revision": None,
            "obs_keys": ["camera"],
            "context_stride": 30,
            "context_max_original_timesteps": 5295,
            "selection": "end_anchored",
            "tokens_per_image": 256,
            "feature_dim": 1152,
            "entries": {},
        }
        (root / CONTEXT_NATIVE_FEATURE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        try:
            ContextNativeFeatureStore(
                root,
                feature_model_name="google/siglip2-so400m-patch16-512",
                feature_model_revision=None,
                obs_keys=["camera"],
                context_stride=30,
                context_max_original_timesteps=5295,
                expected_tokens_per_image=1024,
                expected_feature_stage=RECAP_VISION_PATCH_STAGE,
            )
        except ValueError as exc:
            message = str(exc)
            assert "feature_model_name" in message
            assert "feature_stage" in message
            assert "tokens_per_image" in message
        else:
            raise AssertionError("Old Gemma-native cache was accepted for an all-RECAP context path")


def test_no_context_freezes_context_only_slot_embedding_for_ddp() -> None:
    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.uses_context = False
    model.context_slot_embedding = nn.Embedding(104, 5)

    model._freeze_unused_no_context_parameters()

    assert all(not parameter.requires_grad for parameter in model.context_slot_embedding.parameters())

    context_model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(context_model)
    context_model.uses_context = True
    context_model.context_slot_embedding = nn.Embedding(104, 5)
    context_model._freeze_unused_no_context_parameters()
    assert all(parameter.requires_grad for parameter in context_model.context_slot_embedding.parameters())


def test_no_context_without_source_embeddings_freezes_unused_embeddings_for_ddp() -> None:
    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.uses_context = False
    model.config = SimpleNamespace(use_source_embeddings=False)
    model.context_slot_embedding = nn.Embedding(104, 5)
    model.segment_embedding = nn.Embedding(4, 5)
    model.modality_embedding = nn.Embedding(4, 5)
    model.camera_embedding = nn.Embedding(3, 5)

    model._freeze_unused_no_context_parameters()

    for embedding in (
        model.context_slot_embedding,
        model.segment_embedding,
        model.modality_embedding,
        model.camera_embedding,
    ):
        assert all(not parameter.requires_grad for parameter in embedding.parameters())


def test_rope_only_context_returns_zero_slot_offsets_without_learned_parameters() -> None:
    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.value_query = nn.Parameter(torch.zeros(1, 1, 5))
    model.context_slot_embedding = None

    offsets = model._context_slot_offsets(
        177, device=torch.device("cpu"), dtype=torch.float32
    )

    assert tuple(offsets.shape) == (177, 5)
    assert torch.count_nonzero(offsets).item() == 0
    assert not any("context_slot_embedding" in name for name, _ in model.named_parameters())


def test_siglip_freeze_vision_keeps_connector_and_gemma_trainable() -> None:
    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.vision_tower = nn.Linear(4, 4)
    model.native_vision_projector = None
    model.visual_projector = nn.Linear(4, 5)
    model.language_model = nn.Linear(5, 5)

    model._freeze_vision_tower()

    assert all(not parameter.requires_grad for parameter in model.vision_tower.parameters())
    assert all(parameter.requires_grad for parameter in model.visual_projector.parameters())
    assert all(parameter.requires_grad for parameter in model.language_model.parameters())
    assert model.vision_tower.training is False


def test_siglip_512_query_merger_reduces_1024_patches_to_256_per_camera() -> None:
    class FakeProcessor:
        def __call__(self, *, images, return_tensors):
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros(len(images), 3, 8, 8)}

    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(freeze_backbone=False, freeze_vision=True)
    model.uses_native_gemma_vision = False
    model.uses_native_style_270m_vision = False
    model.vision_tower = nn.Linear(1, 1, bias=False)
    model.image_processor = FakeProcessor()
    model.visual_projector = Qwen2StyleSpatialPatchMerger(
        context_dim=3,
        output_dim=5,
        spatial_merge_size=2,
    )
    model._run_vision_tower = lambda processed: SimpleNamespace(
        last_hidden_state=torch.randn(processed["pixel_values"].shape[0], 1024, 3)
    )

    tokens = model._encode_visual_tokens(torch.zeros(1, 3, 3, 8, 8))

    assert tuple(tokens.shape) == (1, 3 * 256, 5)
    tokens.sum().backward()
    assert model.visual_projector.mlp[0].weight.grad is not None
    assert model.visual_projector.mlp[2].weight.grad is not None


def test_siglip_tensor_preprocessing_batches_resize_and_normalize_on_vision_device() -> None:
    class RecordingTensorProcessor:
        def __init__(self) -> None:
            self.images = None
            self.device = None

        def __call__(self, *, images, return_tensors, device):
            assert return_tensors == "pt"
            self.images = images.detach().clone()
            self.device = device
            resized = F.interpolate(images.float(), size=(4, 4), mode="bilinear")
            return {"pixel_values": resized.div(127.5).sub(1.0)}

    class RecordingVisionTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))
            self.pixel_values = None

        def forward(self, *, pixel_values, return_dict):
            self.pixel_values = pixel_values.detach().clone()
            features = self.scale * torch.ones(pixel_values.shape[0], 4, 3)
            return SimpleNamespace(last_hidden_state=features)

    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        freeze_backbone=False,
        freeze_vision=True,
        query_image_preprocess_mode="torch_gpu",
    )
    model.uses_native_gemma_vision = False
    model.uses_native_style_270m_vision = False
    model.vision_tower = RecordingVisionTower()
    model.image_processor = RecordingTensorProcessor()
    model.visual_projector = nn.Linear(3, 5)
    model._run_vision_tower = lambda processed: model.vision_tower(
        **processed, return_dict=True
    )

    query_images = torch.stack(
        [torch.zeros(3, 8, 8), torch.ones(3, 8, 8)], dim=0
    ).reshape(1, 2, 3, 8, 8)
    tokens = model._encode_visual_tokens(query_images)

    assert tuple(tokens.shape) == (1, 2 * 4, 5)
    assert tuple(model.image_processor.images.shape) == (2, 3, 8, 8)
    assert model.image_processor.images.dtype == torch.uint8
    assert model.image_processor.device == next(model.vision_tower.parameters()).device
    assert tuple(model.vision_tower.pixel_values.shape) == (2, 3, 4, 4)
    assert torch.allclose(model.vision_tower.pixel_values[0], torch.full((3, 4, 4), -1.0))
    assert torch.allclose(model.vision_tower.pixel_values[1], torch.full((3, 4, 4), 1.0))


def test_dualres_freeze_boundary_keeps_both_adapters_trainable() -> None:
    class FakeProcessor:
        def __call__(self, *, images, return_tensors, do_pan_and_scan):
            return {"pixel_values": torch.zeros(len(images), 3, 4, 4)}

    class FakeVisionTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, *, pixel_values, return_dict):
            features = self.scale * torch.ones(pixel_values.shape[0], 4096, 3)
            return SimpleNamespace(last_hidden_state=features)

    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(freeze_backbone=False, freeze_vision=True)
    model.vision_tower = FakeVisionTower()
    model.language_model = nn.Linear(1, 5, bias=False)
    model.image_processor = FakeProcessor()
    model.uses_direct_native_270m_projector = True
    model.uses_qwen2_native_270m_merger = False
    model.uses_dualres_native_270m_merger = True
    model.native_vision_projector = None
    model.native_vision_pooler = nn.AvgPool2d(kernel_size=4, stride=4)
    model.native_vision_patches_per_image = 64
    model.native_vision_tower_hidden = 3
    model.native_pooled_tokens_per_image = 256
    model.native_tokens_per_image = 272
    model.native_context_tokens_per_image = 16
    model.visual_projector = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, 5))
    model.native_spatial_merger = Qwen2StyleSpatialPatchMerger(3, 5, spatial_merge_size=2)
    model.native_spatial_merger_stage2 = Qwen2StyleSpatialPatchMerger(5, 5, spatial_merge_size=2)

    tokens = model._native_style_270m_image_features(
        [np.zeros((4, 4, 3), dtype=np.uint8)]
    )

    assert tuple(tokens.shape) == (1, 272, 5)
    tokens.sum().backward()
    assert model.vision_tower.scale.grad is None
    assert model.visual_projector[1].weight.grad is not None
    assert model.native_spatial_merger.mlp[0].weight.grad is not None
    assert model.native_spatial_merger_stage2.mlp[0].weight.grad is not None


def test_native_style_query_images_are_batched_and_preprocessed_on_the_vision_device() -> None:
    class RecordingProcessor:
        def __init__(self) -> None:
            self.images = None
            self.device = None

        def __call__(self, *, images, return_tensors, do_pan_and_scan, device=None):
            self.images = images.detach().clone()
            self.device = device
            assert return_tensors == "pt"
            assert not do_pan_and_scan
            return {"pixel_values": images.float()}

    class FakeVisionTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, *, pixel_values, return_dict):
            features = self.scale * torch.ones(pixel_values.shape[0], 16, 3, device=pixel_values.device)
            return SimpleNamespace(last_hidden_state=features)

    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(freeze_backbone=False, freeze_vision=True)
    model.vision_tower = FakeVisionTower()
    model.language_model = nn.Linear(1, 3, bias=False)
    model.image_processor = RecordingProcessor()
    model.uses_direct_native_270m_projector = True
    model.uses_qwen2_native_270m_merger = False
    model.uses_dualres_native_270m_merger = False
    model.native_vision_projector = None
    model.native_vision_pooler = nn.AvgPool2d(kernel_size=2, stride=2)
    model.native_vision_patches_per_image = 4
    model.native_vision_tower_hidden = 3
    model.native_pooled_tokens_per_image = 4
    model.native_tokens_per_image = 4
    model.native_context_tokens_per_image = 4
    model.visual_projector = nn.Identity()

    original_uint8 = torch.arange(2 * 2 * 3 * 4 * 4, dtype=torch.uint8).reshape(2, 2, 3, 4, 4)
    query_images = original_uint8.float() / 255.0
    tokens = model._encode_native_style_270m_visual_tokens(query_images)

    assert tuple(tokens.shape) == (2, 2 * 4, 3)
    assert isinstance(model.image_processor.images, torch.Tensor)
    assert tuple(model.image_processor.images.shape) == (4, 3, 4, 4)
    assert model.image_processor.images.dtype == torch.uint8
    assert torch.equal(model.image_processor.images.cpu(), original_uint8.reshape(4, 3, 4, 4))
    assert model.image_processor.device == next(model.vision_tower.parameters()).device


def test_dualres_cached_context_packs_104_slots_at_33_tokens_each() -> None:
    class FakeProprioEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(9, 4)

        def forward(self, proprio, batch_size, device):
            return self.proj(proprio)

    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        context_max_timesteps=104,
        context_views=2,
        proprio_dim=9,
        context_image_chunk_size=4,
        use_source_embeddings=True,
    )
    model.vision_tower = nn.Linear(1, 1, bias=False)
    model.language_model = nn.Linear(1, 1, bias=False)
    model.native_tokens_per_image = 272
    model.native_context_tokens_per_image = 16
    model.native_pooled_tokens_per_image = 256
    model.native_vision_tower_hidden = 3
    model.uses_direct_native_270m_projector = True
    model.uses_qwen2_native_270m_merger = False
    model.uses_dualres_native_270m_merger = True
    model.native_spatial_merger = Qwen2StyleSpatialPatchMerger(3, 4, spatial_merge_size=2)
    model.native_spatial_merger_stage2 = Qwen2StyleSpatialPatchMerger(4, 4, spatial_merge_size=2)
    model.visual_projector = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, 4))
    model.value_query = nn.Parameter(torch.zeros(1, 1, 4))
    model.proprio_encoder = FakeProprioEncoder()
    model.segment_embedding = nn.Embedding(4, 4)
    model.modality_embedding = nn.Embedding(4, 4)
    model.camera_embedding = nn.Embedding(3, 4)
    model.context_slot_embedding = nn.Embedding(104, 4)
    valid_mask = torch.zeros(1, 104, dtype=torch.bool)
    valid_mask[:, -1] = True

    tokens, mask = model._encode_native_style_270m_context_tokens(
        None,
        torch.zeros(1, 104, 9),
        valid_mask,
        context_native_features=torch.randn(2, 256, 3, dtype=torch.bfloat16),
        context_feature_positions=torch.tensor([[0, 103, 0], [0, 103, 1]]),
    )

    assert tuple(tokens.shape) == (1, 104 * 33, 4)
    assert tuple(mask.shape) == (1, 104 * 33)
    assert int(mask.sum()) == 33
    tokens.sum().backward()
    assert model.native_spatial_merger.mlp[0].weight.grad is not None
    assert model.native_spatial_merger_stage2.mlp[0].weight.grad is not None
    assert model.visual_projector[1].weight.grad is None


def test_cached_pooled_features_train_qwen2_merger_and_direct_connector() -> None:
    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        context_max_timesteps=1,
        context_views=1,
        proprio_dim=0,
        context_image_chunk_size=1,
        use_source_embeddings=True,
    )
    model.vision_tower = nn.Linear(1, 1, bias=False)
    model.language_model = nn.Linear(1, 1, bias=False)
    model.native_tokens_per_image = 1
    model.native_context_tokens_per_image = 1
    model.native_pooled_tokens_per_image = 4
    model.native_vision_tower_hidden = 2
    model.uses_direct_native_270m_projector = True
    model.uses_qwen2_native_270m_merger = True
    model.uses_dualres_native_270m_merger = False
    model.native_spatial_merger = Qwen2StyleSpatialPatchMerger(2, 3, spatial_merge_size=2)
    model.visual_projector = nn.Identity()
    model.value_query = nn.Parameter(torch.zeros(1, 1, 3))
    model.proprio_encoder = nn.Identity()
    model.segment_embedding = nn.Embedding(4, 3)
    model.modality_embedding = nn.Embedding(4, 3)
    model.camera_embedding = nn.Embedding(3, 3)
    model.context_slot_embedding = nn.Embedding(1, 3)

    tokens, mask = model._encode_native_style_270m_context_tokens(
        None,
        torch.zeros(1, 1, 0),
        torch.tensor([[True]]),
        context_native_features=torch.randn(1, 4, 2, dtype=torch.bfloat16),
        context_feature_positions=torch.tensor([[0, 0, 0]]),
    )

    assert tuple(tokens.shape) == (1, 2, 3)
    assert int(mask.sum()) == 2
    tokens.sum().backward()
    assert model.native_spatial_merger.mlp[0].weight.grad is not None
    assert model.native_spatial_merger.mlp[2].weight.grad is not None


def test_precompute_writes_end_anchored_shard_and_resumes() -> None:
    class FakeExtractor:
        tokens_per_image = 256
        vision_hidden_size = 4
        resolved_revision = "resolved-revision"

        def __init__(self, *args, **kwargs) -> None:
            pass

        def encode(self, images: list[np.ndarray], *, feature_stage: str) -> torch.Tensor:
            assert feature_stage == NATIVE_PROJECTOR_INPUT_STAGE
            return torch.arange(len(images) * 256 * 4).reshape(len(images), 256, 4).to(torch.bfloat16)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        processed_base = root / "processed"
        processed = processed_base / "value_targets" / LEGACY_GROUND_TRUTH_VERSION
        output = root / "features"
        source_hdf5 = root / "source.hdf5"
        processed.mkdir(parents=True)
        (processed / "tasks").mkdir()
        with h5py.File(source_hdf5, "w") as h5_file:
            obs = h5_file.create_group("data/demo_0/obs")
            obs.create_dataset("camera", data=np.zeros((12, 4, 4, 3), dtype=np.uint8))
        task = {
            "task_id": 0,
            "demos": [
                {
                    "demo_id": "demo_0",
                    "demo_length": 12,
                    "source_hdf5": str(source_hdf5),
                }
            ],
        }
        (processed / "tasks/task_000.json").write_text(json.dumps(task), encoding="utf-8")
        (processed / "summary.json").write_text(
            json.dumps({"tasks": [{"task_id": 0, "metadata_path": "tasks/task_000.json"}]}),
            encoding="utf-8",
        )
        split_manifest = {
            "seed": 0,
            "train_tasks": [
                {
                    "task_id": 0,
                    "train_query_demo_ids": [],
                    "train_context_demo_ids": ["demo_0"],
                }
            ],
        }
        (root / "split.json").write_text(json.dumps(split_manifest), encoding="utf-8")
        presets = {
            "tiny": {
                "manifest": "split.json",
                GROUND_TRUTH_VERSION_FIELD: LEGACY_GROUND_TRUTH_VERSION,
                "preprocess": {
                    "dataset_root": "unused",
                    "output_dir": str(processed_base),
                    "libero_root": "unused",
                    "seed": 0,
                    "demos_per_task": 1,
                    "train_demos_per_task": 1,
                    "test_demos_per_task": 0,
                    "num_bins": 201,
                    "value_min": -1.0,
                    "value_max": 0.0,
                    "strict_demo_count": True,
                    "paper_reference_path": "unused",
                },
                "train": {"manifest_phase": "train"},
            }
        }
        (root / "presets.json").write_text(json.dumps(presets), encoding="utf-8")
        vfe_presets = {
            "defaults": {
                "model": {
                    "context_enabled": True,
                    "context_obs_key": "camera",
                    "context_stride": 10,
                    "context_max_original_timesteps": 520,
                    "native_vision_model_name": "native-model",
                    "native_vision_model_revision": None,
                },
                "context_features": {
                    "enabled": True,
                    "output_dir": str(output),
                    "phases": ["train"],
                    "stage": NATIVE_PROJECTOR_INPUT_STAGE,
                    "model_dtype": "bf16",
                    "feature_dtype": "bf16",
                    "image_batch_size": 8,
                    "device": "cpu",
                    "overwrite": False,
                },
            },
            "presets": {"tiny_vfe": {}},
        }
        (root / "vfe_presets.json").write_text(json.dumps(vfe_presets), encoding="utf-8")
        args = SimpleNamespace(
            data_split_config=str(root / "presets.json"),
            data_split_preset="tiny",
            vfe_config=str(root / "vfe_presets.json"),
            vfe_preset="tiny_vfe",
        )
        with patch("vfe.native_features.Gemma3NativeFeatureExtractor", FakeExtractor):
            first_manifest = precompute_context_native_features(args)
            second_manifest = precompute_context_native_features(args)

        assert first_manifest["num_demos"] == 1
        assert second_manifest["num_slots"] == 2
        assert second_manifest["resolved_native_vision_revision"] == "resolved-revision"
        payload = torch.load(
            output / "task_000/demo_0.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert payload["frame_indices"].tolist() == [1, 11]
        assert tuple(payload["features"].shape) == (2, 1, 256, 4)


def test_recap_precompute_caches_unmerged_1024_patch_tokens() -> None:
    class FakeRecapExtractor:
        tokens_per_image = 1024
        vision_hidden_size = 1152
        projected_hidden_size = 1152
        resolved_revision = "siglip2-resolved-revision"

        def __init__(self, *args, **kwargs) -> None:
            pass

    class ForbiddenNativeExtractor:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("RECAP patch caching must not instantiate the native extractor")

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        processed = root / "processed"
        output = root / "features"
        processed.mkdir()
        args = SimpleNamespace(
            processed_dir=str(processed),
            output_dir=str(output),
            data_split_phases=["train"],
            context_obs_key="camera0,camera1,camera2",
            context_stride=30,
            context_max_original_timesteps=5295,
            context_sampling_mode="end_anchored",
            context_max_timesteps=177,
            feature_stage=RECAP_VISION_PATCH_STAGE,
            feature_model_name="google/siglip2-so400m-patch16-512",
            feature_model_revision=None,
            expected_tokens_per_image=1024,
            model_dtype="bf16",
            feature_dtype="bf16",
            image_batch_size=16,
            device="cpu",
            overwrite=False,
            data_split_config=str(root / "presets.json"),
            data_split_preset="tiny",
            vfe_config=str(root / "vfe_presets.json"),
            vfe_preset="recap",
            resolved_vfe_preset={"preset_name": "recap"},
        )
        with (
            patch("vfe.native_features.apply_context_feature_config", lambda value: value),
            patch("vfe.native_features._context_demo_records", return_value=[]),
            patch("vfe.native_features.HfVisionPatchFeatureExtractor", FakeRecapExtractor),
            patch("vfe.native_features.Gemma3NativeFeatureExtractor", ForbiddenNativeExtractor),
        ):
            manifest = precompute_context_native_features(args)

        assert manifest["feature_type"] == CONTEXT_HF_VISION_FEATURE_TYPE
        assert manifest["feature_stage"] == RECAP_VISION_PATCH_STAGE
        assert manifest["feature_model_name"] == args.feature_model_name
        assert manifest["tokens_per_image"] == 1024
        assert manifest["feature_dim"] == 1152
        assert "native_vision_model_name" not in manifest


if __name__ == "__main__":
    test_end_anchored_context_indices_keep_terminal_frame()
    test_independent_context_block_mask_has_exact_retrieval_topology()
    test_context_block_causal_mask_has_bidirectional_timestep_diagonal()
    test_feature_store_validates_manifest_and_loads_bfloat16_shard()
    test_feature_store_bounds_mmap_payload_cache_with_lru_eviction()
    test_collate_packs_only_valid_context_features_and_positions()
    test_context_progress_adds_one_masked_token_per_timestep()
    test_cached_native_features_backpropagate_through_trainable_connector()
    test_original_native_projector_chain_is_still_supported()
    test_native_style_pooling_keeps_vision_channels_and_reduces_patch_grid()
    test_qwen2_style_merger_groups_true_spatial_neighbors()
    test_qwen2_style_merger_reduces_256_tokens_to_64_and_backpropagates()
    test_dualres_two_stage_merger_produces_256_plus_16_tokens()
    test_recap_style_dualres_combines_729_detail_and_16_alignment_tokens()
    test_recap_qwen2_512_dualres_derives_256_plus_16_query_tokens_from_siglip()
    test_recap_qwen2_cached_context_uses_exact_query_alignment_mergers()
    test_recap_detail_extractor_loads_only_checkpointed_first_merger()
    test_recap_context_feature_manifest_rejects_old_native_cache()
    test_no_context_freezes_context_only_slot_embedding_for_ddp()
    test_no_context_without_source_embeddings_freezes_unused_embeddings_for_ddp()
    test_rope_only_context_returns_zero_slot_offsets_without_learned_parameters()
    test_siglip_freeze_vision_keeps_connector_and_gemma_trainable()
    test_siglip_512_query_merger_reduces_1024_patches_to_256_per_camera()
    test_siglip_tensor_preprocessing_batches_resize_and_normalize_on_vision_device()
    test_dualres_freeze_boundary_keeps_both_adapters_trainable()
    test_native_style_query_images_are_batched_and_preprocessed_on_the_vision_device()
    test_dualres_cached_context_packs_104_slots_at_33_tokens_each()
    test_cached_pooled_features_train_qwen2_merger_and_direct_connector()
    test_precompute_writes_end_anchored_shard_and_resumes()
    test_recap_precompute_caches_unmerged_1024_patch_tokens()
