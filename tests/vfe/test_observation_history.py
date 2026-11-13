from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from vfe.data import FrameRef, VfeFrameDataset, collate_vfe_batch
from vfe.model import SiglipGemmaVFE, VfeModelConfig
from vfe.observation_history import normalize_query_history_offsets, query_history_frame_indices


def test_history_indices_never_use_future_or_negative_frames() -> None:
    for t, expected in [(0, [0, 0, 0]), (10, [0, 0, 10]), (30, [0, 0, 30]),
                        (45, [0, 15, 45]), (60, [0, 30, 60]), (91, [31, 61, 91])]:
        assert query_history_frame_indices(t, [60, 30, 0]) == expected
    for invalid in [[], [0, 30, 60], [60, 30], [30, 30, 0], [-1, 0], [30.0, 0]]:
        try:
            normalize_query_history_offsets(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid history offsets: {invalid}")
    context_config = VfeModelConfig(
        backbone="gemma3_native_style_270m_context",
        query_history_offsets=[60, 30, 0],
    )
    assert context_config.query_history_offsets == [60, 30, 0]


class SyntheticHistoryDataset(VfeFrameDataset):
    def __init__(self, offsets=(60, 30, 0), views=3):
        self.query_history_offsets = list(offsets)
        self.context_enabled = False
        self.proprio_keys = ["state"]
        self.views = views
        self.loaded_images = []

    def task_entry(self, task_id):
        return {"language_instruction": "move the cube"}

    def demo_entry(self, task_id, demo_id):
        return {"source_hdf5": "unused"}

    def load_label(self, task_id, demo_id):
        return {"value_bin": np.arange(100), "value_continuous": np.arange(100) / 100 - 1}

    def load_frame_image(self, task_id, demo_id, frame_idx):
        assert (task_id, demo_id) == (7, "episode_a")
        self.loaded_images.append(frame_idx)
        images = torch.stack([torch.full((3, 2, 2), frame_idx + camera / 10)
                              for camera in range(self.views)])
        return images[0] if self.views == 1 else images

    def load_frame_proprio(self, task_id, demo_id, frame_idx):
        assert (task_id, demo_id) == (7, "episode_a")
        return torch.tensor([float(frame_idx)])


def test_history_dataset_keeps_cameras_proprio_and_current_targets_aligned() -> None:
    for views in [1, 3]:
        dataset = SyntheticHistoryDataset(views=views)
        items = [dataset.item_from_ref(FrameRef(7, "episode_a", t)) for t in [10, 45, 91]]
        batch = collate_vfe_batch(items)
        assert batch["query_image"].shape == (3, 3, views, 3, 2, 2)
        assert batch["query_proprio"].shape == (3, 3, 1)
        expected = torch.tensor([[0., 0., 10.], [0., 15., 45.], [31., 61., 91.]])
        torch.testing.assert_close(batch["query_image"][:, :, 0, 0, 0, 0], expected)
        torch.testing.assert_close(batch["query_proprio"][:, :, 0], expected)
        assert batch["value_bin"].tolist() == [10, 45, 91]
        torch.testing.assert_close(batch["value_continuous"], torch.tensor([-.9, -.55, -.09]))
        assert not any(key.startswith("context_") for key in batch)
        assert dataset.loaded_images[:2] == [0, 10]  # Repeat frame zero without decoding it twice.
    single = SyntheticHistoryDataset(offsets=[0]).item_from_ref(FrameRef(7, "episode_a", 45))
    assert single["query_image"].shape == (3, 3, 2, 2)
    assert single["query_proprio"].shape == (1,)


class CapturingLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, inputs_embeds, attention_mask, **kwargs):
        self.last_input = inputs_embeds
        self.last_mask = attention_mask
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1) * self.scale)


class ScalarProprio(nn.Module):
    def forward(self, values, batch_size, device):
        return values.to(device).expand(batch_size, 4)


def test_history_forward_preserves_time_camera_identity_and_backpropagates() -> None:
    # Exercise the real input assembly and forward path without downloading backbone weights.
    model = SiglipGemmaVFE.__new__(SiglipGemmaVFE)
    nn.Module.__init__(model)
    model.config = VfeModelConfig(
        backbone="gemma3_native_style_270m", query_history_offsets=[60, 30, 0],
        proprio_dim=1, match_adapter_token_rms=False,
    )
    model.uses_context = False
    model.language_needs_hidden_states = False
    model.language_model = CapturingLanguageModel()
    model.query_history_embedding = nn.Embedding(3, 4)
    model.segment_embedding = nn.Embedding(4, 4)
    model.modality_embedding = nn.Embedding(4, 4)
    model.camera_embedding = nn.Embedding(3, 4)
    model.adapter_output_norm = nn.Identity()
    model.proprio_encoder = ScalarProprio()
    model.value_query = nn.Parameter(torch.zeros(1, 1, 4))
    model.value_head = nn.Identity()
    with torch.no_grad():
        model.segment_embedding.weight.zero_()
        model.modality_embedding.weight.zero_()
        model.camera_embedding.weight.copy_(torch.arange(3).view(3, 1).expand(3, 4))
        model.query_history_embedding.weight.copy_(torch.tensor([100., 200., 300.]).view(3, 1).expand(3, 4))
    model._encode_text_embeddings = lambda language, device: (
        torch.zeros(len(language), 2, 4), torch.ones(len(language), 2, dtype=torch.long)
    )
    model._encode_visual_tokens = lambda images: images[:, :, 0, 0, 0].unsqueeze(-1).expand(-1, -1, 4)
    images = torch.tensor([10., 20., 30.]).view(1, 3, 1, 1, 1, 1).expand(2, 3, 3, 3, 2, 2)
    proprio = torch.tensor([1., 2., 3.]).view(1, 3, 1).expand(2, 3, 1)
    output = model(images, ["task", "task"], proprio)
    assert output.shape == (2, 4)
    assert model.language_model.last_input.shape == (2, 15, 4)
    # Each block contains camera 0/1/2, then its proprio token; text and value follow all blocks.
    expected = torch.tensor([110., 111., 112., 103., 220., 221., 222., 204.,
                             330., 331., 332., 305., 2., 2., 2.])
    torch.testing.assert_close(model.language_model.last_input[0, :, 0], expected)
    assert bool(model.language_model.last_mask.all())
    output.sum().backward()
    assert bool((model.query_history_embedding.weight.grad.abs().sum(dim=1) > 0).all())
    try:
        model(images[:, -1], ["task", "task"], proprio)
    except ValueError as exc:
        assert "Query history images" in str(exc)
    else:
        raise AssertionError("History model silently accepted a single observation")
