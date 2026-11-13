from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .observation_history import normalize_query_history_offsets


SIGLIP_GEMMA_BACKBONE = "siglip_gemma"
SIGLIP_GEMMA_CONTEXT_BACKBONE = "siglip_gemma_context"
GEMMA3_NATIVE_VISION_BACKBONE = "gemma3_native_vision"
GEMMA3_NATIVE_VISION_CONTEXT_BACKBONE = "gemma3_native_vision_context"
GEMMA3_NATIVE_STYLE_270M_BACKBONE = "gemma3_native_style_270m"
GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE = "gemma3_native_style_270m_context"
CAUSAL_ATTENTION_MODE = "causal"
INDEPENDENT_CONTEXT_ATTENTION_MODE = "independent_context_bidirectional_query"
HYBRID_CONTEXT_BLOCK_CAUSAL_ATTENTION_MODE = "independent_sliding_block_causal_full"
ATTENTION_MODES = {
    CAUSAL_ATTENTION_MODE,
    INDEPENDENT_CONTEXT_ATTENTION_MODE,
    HYBRID_CONTEXT_BLOCK_CAUSAL_ATTENTION_MODE,
}
CUSTOM_CONTEXT_ATTENTION_MODES = {
    INDEPENDENT_CONTEXT_ATTENTION_MODE,
    HYBRID_CONTEXT_BLOCK_CAUSAL_ATTENTION_MODE,
}
NATIVE_STYLE_270M_BACKBONES = {
    GEMMA3_NATIVE_STYLE_270M_BACKBONE,
    GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE,
}
NATIVE_270M_PROJECTOR_MODES = {
    "native_then_vfe",
    "direct",
    "qwen2_merger",
    "dualres_two_stage_merger",
    "recap_style_dualres",
    "recap_style_qwen2_512_dualres",
}
CONTEXT_BACKBONES = {
    SIGLIP_GEMMA_CONTEXT_BACKBONE,
    GEMMA3_NATIVE_VISION_CONTEXT_BACKBONE,
    GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE,
}
HF_IMAGE_BACKBONES = {
    SIGLIP_GEMMA_BACKBONE,
    SIGLIP_GEMMA_CONTEXT_BACKBONE,
    GEMMA3_NATIVE_VISION_BACKBONE,
    GEMMA3_NATIVE_VISION_CONTEXT_BACKBONE,
    GEMMA3_NATIVE_STYLE_270M_BACKBONE,
    GEMMA3_NATIVE_STYLE_270M_CONTEXT_BACKBONE,
}


@dataclass
class VfeModelConfig:
    backbone: str = "siglip_gemma_context"
    vision_model_name: str = "google/siglip-so400m-patch14-384"
    language_model_name: str = "google/gemma-3-4b-pt"
    context_vision_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    native_vision_model_name: str = "google/gemma-3-4b-pt"
    native_vision_model_revision: str | None = None
    native_vision_projector_mode: str = "native_then_vfe"
    attention_mode: str = CAUSAL_ATTENTION_MODE
    freeze_backbone: bool = False
    freeze_native_vision: bool = False
    freeze_vision: bool = False
    freeze_recap_detail_merger: bool = False
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    torch_dtype: str = "auto"
    num_bins: int = 201
    value_min: float = -1.0
    value_max: float = 0.0
    hidden_dim: int = 256
    proprio_dim: int = 0
    vocab_size: int = 8192
    max_language_tokens: int = 32
    query_history_offsets: list[int] = field(default_factory=lambda: [0])
    context_max_timesteps: int = 120
    context_sampling_mode: str = "end_anchored"
    context_views: int = 2
    context_progress_enabled: bool = False
    context_slot_embedding_enabled: bool = True
    context_image_size: int = 196
    context_patch_tokens_per_image: int = 144
    query_patch_tokens_per_image: int = 729
    query_vision_projector_mode: str = "linear"
    query_image_preprocess_mode: str = "hf_cpu"
    context_image_chunk_size: int = 64
    gradient_checkpointing: bool = False
    adapter_init_std: float = 0.02
    adapter_token_scale: float = 1.0
    adapter_output_norm: str = "layernorm"
    match_adapter_token_rms: bool = True
    use_source_embeddings: bool = True
    log_input_embedding_stats: bool = False
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        self.query_history_offsets = normalize_query_history_offsets(self.query_history_offsets)
        if self.context_sampling_mode not in {"end_anchored", "uniform"}:
            raise ValueError(f"Unknown context sampling mode: {self.context_sampling_mode}")
        uses_recap_qwen2_512 = (
            self.backbone in NATIVE_STYLE_270M_BACKBONES
            and self.native_vision_projector_mode == "recap_style_qwen2_512_dualres"
        )
        if self.freeze_recap_detail_merger and not uses_recap_qwen2_512:
            raise ValueError(
                "freeze_recap_detail_merger requires recap_style_qwen2_512_dualres"
            )
        if int(self.query_patch_tokens_per_image) <= 0:
            raise ValueError("query_patch_tokens_per_image must be positive")
        if self.query_vision_projector_mode not in {"linear", "qwen2_merger"}:
            raise ValueError(
                "query_vision_projector_mode must be 'linear' or 'qwen2_merger'"
            )
        if self.query_image_preprocess_mode not in {"hf_cpu", "torch_gpu"}:
            raise ValueError(
                "query_image_preprocess_mode must be 'hf_cpu' or 'torch_gpu'"
            )
        if (
            self.query_image_preprocess_mode != "hf_cpu"
            and self.backbone not in {SIGLIP_GEMMA_BACKBONE, SIGLIP_GEMMA_CONTEXT_BACKBONE}
            and not uses_recap_qwen2_512
        ):
            raise ValueError(
                "GPU query-image preprocessing currently requires a SigLIP+Gemma backbone"
            )
        if (
            self.query_vision_projector_mode != "linear"
            and self.backbone not in {SIGLIP_GEMMA_BACKBONE, SIGLIP_GEMMA_CONTEXT_BACKBONE}
            and not uses_recap_qwen2_512
        ):
            raise ValueError(
                "A non-linear query_vision_projector_mode currently requires a SigLIP+Gemma backbone"
            )
        if len(self.query_history_offsets) > 1 and self.backbone not in HF_IMAGE_BACKBONES:
            raise ValueError("Query observation history requires an HF image backbone")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "VfeModelConfig":
        values = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        if "context_image_size" not in values:
            patch_tokens = int(values.get("context_patch_tokens_per_image", cls.context_patch_tokens_per_image))
            if patch_tokens == 196:
                values["context_image_size"] = 224
            elif patch_tokens == 144:
                values["context_image_size"] = 196
        return cls(**values)


def build_independent_context_block_mask(
    valid_token_mask: torch.Tensor,
    *,
    context_token_count: int,
    context_timestep_count: int,
) -> Any:
    """Build the independent-context retrieval mask as a FlexAttention BlockMask.

    Every context timestep is a bidirectional block isolated from every other
    timestep. The current-observation block (query vision, query proprio, and
    language) attends bidirectionally within itself and unidirectionally to all
    valid context blocks. The final value token attends every valid token.
    """

    return _build_context_retrieval_block_mask(
        valid_token_mask,
        context_token_count=context_token_count,
        context_timestep_count=context_timestep_count,
        context_blocks_are_causal=False,
    )


def build_context_block_causal_mask(
    valid_token_mask: torch.Tensor,
    *,
    context_token_count: int,
    context_timestep_count: int,
) -> Any:
    """Build a timestep-level causal context mask with bidirectional diagonal blocks.

    Context timestep ``i`` reads every context timestep ``j <= i``. Tokens
    inside each timestep remain bidirectional. The current-observation block
    reads all context and itself bidirectionally, while the final value token
    reads the full valid sequence.
    """

    return _build_context_retrieval_block_mask(
        valid_token_mask,
        context_token_count=context_token_count,
        context_timestep_count=context_timestep_count,
        context_blocks_are_causal=True,
    )


def _build_context_retrieval_block_mask(
    valid_token_mask: torch.Tensor,
    *,
    context_token_count: int,
    context_timestep_count: int,
    context_blocks_are_causal: bool,
) -> Any:
    if valid_token_mask.dim() != 2:
        raise ValueError("valid_token_mask must have shape [batch, sequence]")
    batch_size, sequence_length = valid_token_mask.shape
    if context_timestep_count <= 0:
        raise ValueError("context_timestep_count must be positive")
    if context_token_count <= 0 or context_token_count % context_timestep_count != 0:
        raise ValueError(
            "context_token_count must be positive and divisible by context_timestep_count"
        )
    value_position = sequence_length - 1
    if context_token_count >= value_position:
        raise ValueError("The current-observation block must contain at least one token")

    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:
        raise RuntimeError("Custom context attention requires PyTorch FlexAttention") from exc

    valid = valid_token_mask.to(dtype=torch.bool)
    context_block_size = context_token_count // context_timestep_count

    def mask_mod(
        batch_index: torch.Tensor,
        _head_index: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        query_valid = valid[batch_index, query_index]
        key_valid = valid[batch_index, key_index]

        query_is_context = query_index < context_token_count
        key_is_context = key_index < context_token_count
        query_context_timestep = query_index // context_block_size
        key_context_timestep = key_index // context_block_size
        context_relation = (
            key_context_timestep <= query_context_timestep
            if context_blocks_are_causal
            else key_context_timestep == query_context_timestep
        )
        context_reads_context = (
            query_is_context & key_is_context & context_relation
        )

        query_is_current = (query_index >= context_token_count) & (query_index < value_position)
        key_is_prefix = key_index < value_position
        current_reads_context_and_current = query_is_current & key_is_prefix

        value_reads_everything = query_index == value_position
        allowed = context_reads_context | current_reads_context_and_current | value_reads_everything
        return query_valid & key_valid & allowed

    return create_block_mask(
        mask_mod=mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=valid.device,
        _compile=valid.device.type == "cuda",
    )


def tokenize_texts(texts: list[str], vocab_size: int, max_tokens: int, device: torch.device) -> torch.Tensor:
    token_ids = torch.zeros((len(texts), max_tokens), dtype=torch.long, device=device)
    for row, text in enumerate(texts):
        words = re.findall(r"[A-Za-z0-9]+", text.lower())
        for col, word in enumerate(words[:max_tokens]):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=4).digest()
            token_ids[row, col] = int.from_bytes(digest, byteorder="little") % (vocab_size - 1) + 1
    return token_ids


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable low-rank residual update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")

        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(self.dropout(inputs))) * self.scaling


class Qwen2StyleSpatialPatchMerger(nn.Module):
    """Merge each non-overlapping 2x2 spatial neighborhood with a Qwen2-VL MLP."""

    def __init__(self, context_dim: int, output_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        if spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.spatial_merge_size = int(spatial_merge_size)
        merged_hidden = self.context_dim * self.spatial_merge_size**2
        self.norm = nn.LayerNorm(self.context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(merged_hidden, merged_hidden),
            nn.GELU(),
            nn.Linear(merged_hidden, self.output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise RuntimeError(f"Spatial patch merger expects [batch, tokens, hidden], got {tuple(tokens.shape)}")
        batch_size, token_count, hidden_size = tokens.shape
        if hidden_size != self.context_dim:
            raise RuntimeError(
                f"Spatial patch merger hidden size is {hidden_size}; expected {self.context_dim}"
            )
        grid_size = math.isqrt(token_count)
        if grid_size * grid_size != token_count:
            raise RuntimeError(f"Spatial patch merger token count {token_count} is not a square grid")
        merge = self.spatial_merge_size
        if grid_size % merge != 0:
            raise RuntimeError(f"Spatial grid size {grid_size} is not divisible by merge size {merge}")

        normalized = self.norm(tokens).reshape(batch_size, grid_size, grid_size, hidden_size)
        merged_grid_size = grid_size // merge
        neighborhoods = normalized.reshape(
            batch_size,
            merged_grid_size,
            merge,
            merged_grid_size,
            merge,
            hidden_size,
        ).permute(0, 1, 3, 2, 4, 5)
        neighborhoods = neighborhoods.reshape(
            batch_size,
            merged_grid_size * merged_grid_size,
            merge * merge * hidden_size,
        )
        return self.mlp(neighborhoods)


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        mask = token_ids.ne(0).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (embedded * mask).sum(dim=1) / denom
        return self.proj(pooled)


class ProprioEncoder(nn.Module):
    """Encode query proprio into a backbone-size token."""

    def __init__(self, proprio_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        if proprio_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(proprio_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.net = None

    def forward(self, query_proprio: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.net is None or query_proprio is None or query_proprio.shape[-1] == 0:
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        return self.net(query_proprio)


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    if dtype_name in {"float16", "fp16"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _infer_hidden_size(model_config: Any) -> int:
    for attr in ("hidden_size", "projection_dim", "text_config", "language_config", "vision_config"):
        value = getattr(model_config, attr, None)
        if isinstance(value, int):
            return value
        nested = getattr(value, "hidden_size", None)
        if isinstance(nested, int):
            return nested
    raise ValueError("Could not infer hidden size from model config")


def _infer_vision_hidden_size(model_config: Any) -> int:
    vision_config = getattr(model_config, "vision_config", None)
    hidden_size = getattr(vision_config, "hidden_size", None)
    if isinstance(hidden_size, int):
        return hidden_size
    return _infer_hidden_size(model_config)


class LightweightVFE(nn.Module):
    """Small local fallback for π*0.6-style VFE input/output contract."""

    def __init__(self, config: VfeModelConfig) -> None:
        super().__init__()
        self.config = config
        cfg = self.config
        self.image_encoder = ImageEncoder(cfg.hidden_dim, cfg.dropout)
        self.text_encoder = TextEncoder(cfg.vocab_size, cfg.hidden_dim)
        self.proprio_encoder = ProprioEncoder(cfg.proprio_dim, cfg.hidden_dim, cfg.dropout)
        self.type_embedding = nn.Embedding(3, cfg.hidden_dim)
        self.position_embedding = nn.Embedding(512, cfg.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
        self.value_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_bins),
        )

    def encode_language(self, language: list[str] | torch.Tensor, device: torch.device) -> torch.Tensor:
        if isinstance(language, torch.Tensor):
            token_ids = language.to(device)
        else:
            token_ids = tokenize_texts(
                language,
                vocab_size=self.config.vocab_size,
                max_tokens=self.config.max_language_tokens,
                device=device,
            )
        return self.text_encoder(token_ids)

    def forward(
        self,
        query_image: torch.Tensor,
        language: list[str] | torch.Tensor,
        query_proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = query_image.shape[0]
        device = query_image.device
        if query_image.dim() == 5:
            query_views = query_image.shape[1]
            query_flat = query_image.reshape(batch_size * query_views, *query_image.shape[2:])
            query_token = self.image_encoder(query_flat).reshape(batch_size, query_views, -1).mean(dim=1, keepdim=True)
        else:
            query_token = self.image_encoder(query_image).unsqueeze(1)
        language_token = self.encode_language(language, device).unsqueeze(1)

        tokens = torch.cat([language_token, query_token], dim=1)
        type_ids = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=device),
                torch.ones((batch_size, 1), dtype=torch.long, device=device),
            ],
            dim=1,
        )
        if tokens.shape[1] > self.position_embedding.num_embeddings:
            raise ValueError(f"Too many visual/text tokens: {tokens.shape[1]}")
        pos_ids = torch.arange(tokens.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
        tokens = tokens + self.type_embedding(type_ids) + self.position_embedding(pos_ids)
        fused = self.fusion(tokens)
        proprio_token = self.proprio_encoder(query_proprio, batch_size, device)
        query_fused = fused[:, 1] + proprio_token
        return self.value_head(query_fused)


class SiglipGemmaVFE(nn.Module):
    """Value function with HF VLM image encoders and Gemma fusion."""

    def __init__(self, config: VfeModelConfig) -> None:
        super().__init__()
        self.config = config
        self.uses_context = config.backbone in CONTEXT_BACKBONES
        self.uses_native_gemma_vision = config.backbone in {
            GEMMA3_NATIVE_VISION_BACKBONE,
            GEMMA3_NATIVE_VISION_CONTEXT_BACKBONE,
        }
        self.uses_native_style_270m_vision = config.backbone in NATIVE_STYLE_270M_BACKBONES
        if config.context_progress_enabled and not self.uses_context:
            raise ValueError("context_progress_enabled requires a context backbone")
        if config.native_vision_projector_mode not in NATIVE_270M_PROJECTOR_MODES:
            raise ValueError(
                "native_vision_projector_mode must be one of: "
                f"{sorted(NATIVE_270M_PROJECTOR_MODES)}"
            )
        self.uses_direct_native_270m_projector = (
            self.uses_native_style_270m_vision
            and config.native_vision_projector_mode
            in {
                "direct",
                "qwen2_merger",
                "dualres_two_stage_merger",
                "recap_style_dualres",
                "recap_style_qwen2_512_dualres",
            }
        )
        self.uses_qwen2_native_270m_merger = (
            self.uses_native_style_270m_vision and config.native_vision_projector_mode == "qwen2_merger"
        )
        self.uses_recap_style_dualres = (
            self.uses_native_style_270m_vision
            and config.native_vision_projector_mode
            in {"recap_style_dualres", "recap_style_qwen2_512_dualres"}
        )
        self.uses_recap_qwen2_512_dualres = (
            self.uses_native_style_270m_vision
            and config.native_vision_projector_mode == "recap_style_qwen2_512_dualres"
        )
        self.uses_dualres_native_270m_merger = (
            self.uses_native_style_270m_vision
            and config.native_vision_projector_mode
            in {
                "dualres_two_stage_merger",
                "recap_style_dualres",
                "recap_style_qwen2_512_dualres",
            }
        )
        if config.attention_mode not in ATTENTION_MODES:
            raise ValueError(f"attention_mode must be one of: {sorted(ATTENTION_MODES)}")
        if config.attention_mode in CUSTOM_CONTEXT_ATTENTION_MODES and not (
            self.uses_context and self.uses_dualres_native_270m_merger
        ):
            raise ValueError(
                f"{INDEPENDENT_CONTEXT_ATTENTION_MODE} requires the dual-resolution "
                "Gemma 270M context backbone"
            )
        if config.freeze_native_vision and not self.uses_native_gemma_vision:
            raise ValueError("freeze_native_vision requires a Gemma 3 native-vision backbone")
        if config.freeze_vision and not (
            self.uses_native_style_270m_vision or config.backbone == SIGLIP_GEMMA_BACKBONE
        ):
            raise ValueError(
                "freeze_vision requires the SigLIP no-context or Gemma 3 native-style 270M vision backbone"
            )
        if config.use_lora and not self.uses_native_gemma_vision:
            raise ValueError("LoRA is implemented only for Gemma 3 native-vision backbones")
        if config.use_lora and config.freeze_backbone:
            raise ValueError("LoRA cannot be combined with freeze_backbone because LoRA parameters would be frozen")
        try:
            from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SigLIP+Gemma value function requires transformers. Install it "
                "in the active environment before training."
            ) from exc

        dtype = _resolve_torch_dtype(config.torch_dtype)
        self.recap_image_processor = None
        self.recap_vision_tower = None
        self.recap_tokens_per_image = 0
        self.recap_vision_hidden = 0
        if self.uses_native_gemma_vision:
            native_processor = AutoProcessor.from_pretrained(config.language_model_name)
            self.image_processor = getattr(native_processor, "image_processor", native_processor)
            self.context_image_processor = self.image_processor if self.uses_context else None
        elif self.uses_native_style_270m_vision:
            native_pretrained_kwargs = (
                {"revision": config.native_vision_model_revision}
                if config.native_vision_model_revision
                else {}
            )
            native_processor = AutoProcessor.from_pretrained(
                config.native_vision_model_name,
                **native_pretrained_kwargs,
            )
            self.image_processor = getattr(native_processor, "image_processor", native_processor)
            self.context_image_processor = self.image_processor if self.uses_context else None
            if self.uses_recap_style_dualres:
                recap_processor_kwargs = (
                    {"backend": "torchvision"}
                    if self.uses_recap_qwen2_512_dualres
                    and config.query_image_preprocess_mode == "torch_gpu"
                    else {}
                )
                self.recap_image_processor = AutoImageProcessor.from_pretrained(
                    config.vision_model_name,
                    **recap_processor_kwargs,
                )
        else:
            query_processor_kwargs = (
                {"backend": "torchvision"}
                if config.query_image_preprocess_mode == "torch_gpu"
                else {}
            )
            self.image_processor = AutoImageProcessor.from_pretrained(
                config.vision_model_name,
                **query_processor_kwargs,
            )
            self.context_image_processor = (
                AutoImageProcessor.from_pretrained(config.context_vision_model_name) if self.uses_context else None
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.native_vision_projector: nn.Module | None = None
        self.native_vision_pooler: nn.Module | None = None
        self.native_spatial_merger: nn.Module | None = None
        self.native_spatial_merger_stage2: nn.Module | None = None
        self.recap_spatial_merger_stage2: nn.Module | None = None
        self.recap_spatial_merger_stage3: nn.Module | None = None
        self.native_vision_patches_per_image = 0
        self.native_pooled_tokens_per_image = 0
        if self.uses_native_gemma_vision:
            self.language_model = AutoModel.from_pretrained(config.language_model_name, torch_dtype=dtype)
            if not hasattr(self.language_model, "get_image_features"):
                raise ValueError(
                    f"{config.language_model_name!r} does not expose Gemma3 native image features; "
                    "use a Gemma 3 multimodal checkpoint such as google/gemma-3-4b-pt."
                )
            self.vision_tower = getattr(self.language_model, "vision_tower", None)
            self.context_vision_tower = self.vision_tower if self.uses_context else None
        elif self.uses_native_style_270m_vision:
            self.language_model = self._load_gemma_language_model(
                AutoModel,
                AutoModelForCausalLM,
                config.language_model_name,
                dtype,
                attention_implementation=(
                    "flex_attention"
                    if config.attention_mode in CUSTOM_CONTEXT_ATTENTION_MODES
                    else None
                ),
            )
            native_vision_model = AutoModel.from_pretrained(
                config.native_vision_model_name,
                torch_dtype=dtype,
                **native_pretrained_kwargs,
            )
            self.vision_tower = getattr(native_vision_model, "vision_tower", None)
            if self.vision_tower is None:
                raise ValueError(
                    f"{config.native_vision_model_name!r} does not expose a Gemma 3 native vision tower; "
                    "use a Gemma 3 multimodal checkpoint such as google/gemma-3-4b-pt."
                )
            native_config = native_vision_model.config
            self.native_pooled_tokens_per_image = int(getattr(native_config, "mm_tokens_per_image", 256))
            if self.uses_qwen2_native_270m_merger:
                self.native_tokens_per_image = self.native_pooled_tokens_per_image // 4
            elif self.uses_recap_style_dualres:
                recap_model = AutoModel.from_pretrained(
                    config.vision_model_name,
                    torch_dtype=dtype,
                )
                recap_vision_config = getattr(
                    recap_model.config,
                    "vision_config",
                    recap_model.config,
                )
                # RECAP uses only SigLIP's visual encoder. Drop the unused text
                # tower from this mixed model while retaining the identical
                # vision forward path and connector input.
                self.recap_vision_tower = getattr(recap_model, "vision_model", recap_model)
                self.recap_vision_hidden = _infer_vision_hidden_size(self.recap_vision_tower.config)
                recap_image_size = int(getattr(recap_vision_config, "image_size"))
                recap_patch_size = int(getattr(recap_vision_config, "patch_size"))
                recap_grid_size = recap_image_size // recap_patch_size
                self.recap_tokens_per_image = recap_grid_size * recap_grid_size
                if self.recap_tokens_per_image != int(config.query_patch_tokens_per_image):
                    raise ValueError(
                        f"Vision model {config.vision_model_name!r} produces "
                        f"{self.recap_tokens_per_image} query patch tokens/image, but the VFE preset "
                        f"declares {config.query_patch_tokens_per_image}"
                    )
                if self.uses_recap_qwen2_512_dualres:
                    if recap_grid_size % 8 != 0:
                        raise ValueError(
                            "RECAP-style Qwen2 dual-resolution vision requires a patch grid "
                            "divisible by 8 for three consecutive 2x2 mergers"
                        )
                    self.recap_detail_tokens_per_image = (recap_grid_size // 2) ** 2
                    self.recap_alignment_tokens_per_image = (recap_grid_size // 8) ** 2
                else:
                    self.recap_detail_tokens_per_image = self.recap_tokens_per_image
                    self.recap_alignment_tokens_per_image = self.native_pooled_tokens_per_image // 16
                self.native_tokens_per_image = (
                    self.recap_detail_tokens_per_image
                    + self.recap_alignment_tokens_per_image
                )
            elif self.uses_dualres_native_270m_merger:
                self.native_tokens_per_image = (
                    self.native_pooled_tokens_per_image + self.native_pooled_tokens_per_image // 16
                )
            else:
                self.native_tokens_per_image = self.native_pooled_tokens_per_image
            self.native_context_tokens_per_image = (
                self.native_pooled_tokens_per_image // 16
                if self.uses_dualres_native_270m_merger
                else self.native_tokens_per_image
            )
            native_vision_projector = getattr(native_vision_model, "multi_modal_projector", None)
            if not self.uses_direct_native_270m_projector:
                if native_vision_projector is None:
                    raise ValueError(
                        f"{config.native_vision_model_name!r} does not expose a native multimodal projector"
                    )
                self.native_vision_projector = native_vision_projector
                self.native_vision_hidden = int(native_vision_projector.mm_input_projection_weight.shape[1])
            native_vision_config = getattr(native_config, "vision_config", None)
            if native_vision_config is None:
                raise ValueError("Gemma 3 native model config does not expose vision_config")
            self.native_vision_tower_hidden = _infer_vision_hidden_size(native_vision_config)
            image_size = int(native_vision_config.image_size)
            patch_size = int(native_vision_config.patch_size)
            self.native_vision_patches_per_image = image_size // patch_size
            tokens_per_side = math.isqrt(self.native_pooled_tokens_per_image)
            if tokens_per_side * tokens_per_side != self.native_pooled_tokens_per_image:
                raise ValueError("Gemma 3 mm_tokens_per_image must be a perfect square")
            if self.native_vision_patches_per_image % tokens_per_side != 0:
                raise ValueError("Gemma 3 vision patch grid must be divisible by the pooled token grid")
            pool_kernel = self.native_vision_patches_per_image // tokens_per_side
            self.native_vision_pooler = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel)
            if self.uses_qwen2_native_270m_merger or self.uses_dualres_native_270m_merger:
                if tokens_per_side % 2 != 0:
                    raise ValueError("Qwen2-style 2x2 merger requires an even pooled token grid")
            self.context_vision_tower = None
            # The direct mode retains only the tower and parameter-free pooler;
            # compatibility mode also retains Gemma's native 1152->2560 projector.
            # Neither mode retains the 4B decoder.
            del native_vision_model
        else:
            self.vision_tower = AutoModel.from_pretrained(config.vision_model_name, torch_dtype=dtype)
            self.context_vision_tower = (
                AutoModel.from_pretrained(config.context_vision_model_name, torch_dtype=dtype) if self.uses_context else None
            )
            self.language_model = self._load_gemma_language_model(AutoModel, AutoModelForCausalLM, config.language_model_name, dtype)
        self.language_needs_hidden_states = self.language_model.__class__.__name__.lower().endswith("forcausallm")
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        language_hidden = _infer_hidden_size(self.language_model.config)
        if self.uses_native_gemma_vision:
            self.visual_projector = None
            self.context_visual_projector = None
            native_config = self.language_model.config
            self.native_tokens_per_image = int(getattr(native_config, "mm_tokens_per_image", 256))
        elif self.uses_native_style_270m_vision:
            if self.uses_qwen2_native_270m_merger:
                self.native_spatial_merger = Qwen2StyleSpatialPatchMerger(
                    context_dim=self.native_vision_tower_hidden,
                    output_dim=language_hidden,
                    spatial_merge_size=2,
                )
                self.visual_projector = nn.Identity()
            elif self.uses_dualres_native_270m_merger:
                # The all-RECAP 512 path shares its three mergers between query
                # alignment tokens and context tokens, so the old Gemma-native
                # context-only mergers must not exist as trainable dead weights.
                if not self.uses_recap_qwen2_512_dualres:
                    self.native_spatial_merger = Qwen2StyleSpatialPatchMerger(
                        context_dim=self.native_vision_tower_hidden,
                        output_dim=language_hidden,
                        spatial_merge_size=2,
                    )
                    self.native_spatial_merger_stage2 = Qwen2StyleSpatialPatchMerger(
                        context_dim=language_hidden,
                        output_dim=language_hidden,
                        spatial_merge_size=2,
                    )
                detail_hidden = (
                    self.recap_vision_hidden
                    if self.uses_recap_style_dualres and self.recap_vision_tower is not None
                    else self.native_vision_tower_hidden
                )
                if self.uses_recap_qwen2_512_dualres:
                    self.visual_projector = Qwen2StyleSpatialPatchMerger(
                        context_dim=detail_hidden,
                        output_dim=language_hidden,
                        spatial_merge_size=2,
                    )
                    self.recap_spatial_merger_stage2 = Qwen2StyleSpatialPatchMerger(
                        context_dim=language_hidden,
                        output_dim=language_hidden,
                        spatial_merge_size=2,
                    )
                    self.recap_spatial_merger_stage3 = Qwen2StyleSpatialPatchMerger(
                        context_dim=language_hidden,
                        output_dim=language_hidden,
                        spatial_merge_size=2,
                    )
                else:
                    self.visual_projector = nn.Sequential(
                        nn.LayerNorm(detail_hidden),
                        nn.Linear(detail_hidden, language_hidden),
                    )
            else:
                visual_input_hidden = (
                    self.native_vision_tower_hidden
                    if self.uses_direct_native_270m_projector
                    else self.native_vision_hidden
                )
                self.visual_projector = nn.Sequential(
                    nn.LayerNorm(visual_input_hidden),
                    nn.Linear(visual_input_hidden, language_hidden),
                )
            self.context_visual_projector = None
        else:
            vision_hidden = _infer_vision_hidden_size(self.vision_tower.config)
            vision_config = getattr(self.vision_tower.config, "vision_config", self.vision_tower.config)
            vision_image_size = int(getattr(vision_config, "image_size"))
            vision_patch_size = int(getattr(vision_config, "patch_size"))
            query_grid_size = vision_image_size // vision_patch_size
            actual_query_tokens_per_image = query_grid_size * query_grid_size
            if actual_query_tokens_per_image != int(config.query_patch_tokens_per_image):
                raise ValueError(
                    f"Vision model {config.vision_model_name!r} produces "
                    f"{actual_query_tokens_per_image} query patch tokens/image, but the VFE preset "
                    f"declares {config.query_patch_tokens_per_image}"
                )
            if config.query_vision_projector_mode == "qwen2_merger":
                self.visual_projector = Qwen2StyleSpatialPatchMerger(
                    context_dim=vision_hidden,
                    output_dim=language_hidden,
                    spatial_merge_size=2,
                )
            else:
                self.visual_projector = nn.Sequential(
                    nn.LayerNorm(vision_hidden),
                    nn.Linear(vision_hidden, language_hidden),
                )
            self.native_tokens_per_image = 0
        if self.uses_context and not self.uses_native_gemma_vision and not self.uses_native_style_270m_vision:
            if self.context_vision_tower is None:
                raise RuntimeError("Context VFE requires a DINO context vision tower")
            context_hidden = _infer_vision_hidden_size(self.context_vision_tower.config)
            self.context_visual_projector = nn.Sequential(
                nn.LayerNorm(context_hidden),
                nn.Linear(context_hidden, language_hidden),
            )
        else:
            self.context_visual_projector = None
        self.proprio_encoder = ProprioEncoder(config.proprio_dim, language_hidden, config.dropout)
        self.context_progress_encoder = (
            ProprioEncoder(1, language_hidden, config.dropout)
            if config.context_progress_enabled
            else None
        )
        self.segment_embedding = nn.Embedding(4, language_hidden)
        self.modality_embedding = nn.Embedding(4, language_hidden)
        self.camera_embedding = nn.Embedding(3, language_hidden)
        self.query_history_embedding = (
            nn.Embedding(len(config.query_history_offsets), language_hidden)
            if len(config.query_history_offsets) > 1
            else None
        )
        self.context_slot_embedding = (
            nn.Embedding(config.context_max_timesteps, language_hidden)
            if config.context_slot_embedding_enabled
            else None
        )
        self.adapter_output_norm = self._make_adapter_output_norm(language_hidden)
        self.value_query = nn.Parameter(torch.empty(1, 1, language_hidden))
        self.value_head = nn.Sequential(
            nn.LayerNorm(language_hidden),
            nn.Linear(language_hidden, language_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(language_hidden, config.num_bins),
        )
        self._printed_input_embedding_stats = False
        self.lora_module_count = 0
        self._init_adapter_inputs()
        self._freeze_unused_no_context_parameters()

        if config.freeze_backbone:
            modules = [self.language_model]
            if not self.uses_native_gemma_vision and self.vision_tower is not None:
                modules.append(self.vision_tower)
            if not self.uses_native_gemma_vision and self.context_vision_tower is not None:
                modules.append(self.context_vision_tower)
            if self.native_vision_projector is not None:
                modules.append(self.native_vision_projector)
            if self.recap_vision_tower is not None:
                modules.append(self.recap_vision_tower)
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
                module.eval()
        elif config.use_lora:
            self._configure_native_gemma_lora()
        elif config.freeze_native_vision:
            self._freeze_native_vision_inputs()
        elif config.freeze_vision:
            self._freeze_vision_tower()
        if config.freeze_recap_detail_merger:
            self._freeze_recap_detail_merger()

    def _init_adapter_inputs(self) -> None:
        std = float(self.config.adapter_init_std)
        embeddings = [self.segment_embedding, self.modality_embedding, self.camera_embedding]
        if self.query_history_embedding is not None:
            embeddings.append(self.query_history_embedding)
        if self.context_slot_embedding is not None:
            embeddings.append(self.context_slot_embedding)
        for embedding in embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.value_query, mean=0.0, std=std)

    def _freeze_unused_no_context_parameters(self) -> None:
        """Exclude context-only parameters from no-context DDP reduction."""
        if self.uses_context:
            return
        if self.context_slot_embedding is not None:
            for parameter in self.context_slot_embedding.parameters():
                parameter.requires_grad_(False)
        if not bool(getattr(getattr(self, "config", None), "use_source_embeddings", True)):
            for embedding in (
                self.segment_embedding,
                self.modality_embedding,
                self.camera_embedding,
            ):
                for parameter in embedding.parameters():
                    parameter.requires_grad_(False)

    def _freeze_native_vision_inputs(self) -> None:
        if self.vision_tower is None:
            raise RuntimeError("Gemma 3 native vision backbone did not expose vision_tower")
        for parameter in self.vision_tower.parameters():
            parameter.requires_grad_(False)
        self.vision_tower.eval()

        input_embeddings = self.language_model.get_input_embeddings()
        if input_embeddings is None:
            raise RuntimeError("Gemma 3 native vision backbone did not expose input token embeddings")
        for parameter in input_embeddings.parameters():
            parameter.requires_grad_(False)

    def _configure_native_gemma_lora(self) -> None:
        """Freeze the 4B vision/text bases, train its native projector and decoder LoRA."""

        for parameter in self.language_model.parameters():
            parameter.requires_grad_(False)

        text_decoder = getattr(self.language_model, "language_model", None)
        if text_decoder is None:
            raise RuntimeError("Gemma 3 native VLM did not expose its text decoder as language_model")
        target_modules = {
            target.strip() for target in self.config.lora_target_modules.split(",") if target.strip()
        }
        if not target_modules:
            raise ValueError("lora_target_modules must contain at least one module name")

        replacements = 0
        # Materialize the module list before replacing children so the LoRA
        # wrappers themselves are not recursively considered as candidates.
        for parent in list(text_decoder.modules()):
            for child_name, child in list(parent.named_children()):
                if child_name in target_modules and isinstance(child, nn.Linear):
                    setattr(
                        parent,
                        child_name,
                        LoRALinear(
                            child,
                            rank=self.config.lora_rank,
                            alpha=self.config.lora_alpha,
                            dropout=self.config.lora_dropout,
                        ),
                    )
                    replacements += 1
        if replacements == 0:
            requested = ", ".join(sorted(target_modules))
            raise RuntimeError(
                "No Gemma text-decoder linear layers matched --lora-target-modules "
                f"({requested}). Check the Transformers Gemma 3 module names."
            )
        self.lora_module_count = replacements

        if self.vision_tower is None:
            raise RuntimeError("Gemma 3 native vision backbone did not expose vision_tower")
        self.vision_tower.eval()
        connector = getattr(self.language_model, "multi_modal_projector", None)
        if connector is None:
            raise RuntimeError("Gemma 3 native vision backbone did not expose multi_modal_projector")
        for parameter in connector.parameters():
            parameter.requires_grad_(True)
        connector.train()

    def _freeze_vision_tower(self) -> None:
        """Freeze image feature extraction while leaving VFE connectors trainable."""
        if self.vision_tower is None:
            raise RuntimeError("The selected backbone did not expose a vision tower")
        vision_towers = [self.vision_tower]
        recap_vision_tower = getattr(self, "recap_vision_tower", None)
        if recap_vision_tower is not None:
            vision_towers.append(recap_vision_tower)
        for tower in vision_towers:
            for parameter in tower.parameters():
                parameter.requires_grad_(False)
            tower.eval()
        # Compatibility-mode native-style vision treats Gemma's original
        # multimodal projector as part of the frozen image feature extractor.
        # The VFE visual_projector remains outside this boundary and trainable.
        if self.native_vision_projector is not None:
            for parameter in self.native_vision_projector.parameters():
                parameter.requires_grad_(False)
            self.native_vision_projector.eval()

    def _freeze_recap_detail_merger(self) -> None:
        """Freeze the shared 1024-to-256 merger used by query and cached context."""

        if not self.uses_recap_qwen2_512_dualres or self.visual_projector is None:
            raise RuntimeError(
                "A frozen RECAP detail merger requires recap_style_qwen2_512_dualres"
            )
        for parameter in self.visual_projector.parameters():
            parameter.requires_grad_(False)
        self.visual_projector.eval()

    def _make_adapter_output_norm(self, hidden_size: int) -> nn.Module:
        norm_name = self.config.adapter_output_norm.lower()
        if norm_name in {"none", "identity", ""}:
            return nn.Identity()
        if norm_name == "layernorm":
            return nn.LayerNorm(hidden_size)
        if norm_name == "rmsnorm":
            return nn.RMSNorm(hidden_size)
        raise ValueError("--adapter-output-norm must be one of: layernorm, rmsnorm, none")

    def _load_gemma_language_model(
        self,
        auto_model: Any,
        auto_causal_model: Any,
        model_name: str,
        dtype: torch.dtype | str,
        *,
        attention_implementation: str | None = None,
    ) -> nn.Module:
        errors: list[Exception] = []
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if attention_implementation is not None:
            load_kwargs["attn_implementation"] = attention_implementation
        for model_class in (auto_model, auto_causal_model):
            try:
                return model_class.from_pretrained(model_name, **load_kwargs)
            except (ValueError, OSError, TypeError) as exc:
                errors.append(exc)
        raise RuntimeError(f"Could not load Gemma language model {model_name!r}; last error: {errors[-1]}")

    def _enable_gradient_checkpointing(self) -> None:
        modules = [self.language_model]
        if not self.uses_native_gemma_vision and self.vision_tower is not None:
            modules.append(self.vision_tower)
        if not self.uses_native_gemma_vision and self.context_vision_tower is not None:
            modules.append(self.context_vision_tower)
        recap_vision_tower = getattr(self, "recap_vision_tower", None)
        if recap_vision_tower is not None:
            modules.append(recap_vision_tower)
        for module in modules:
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
            module_config = getattr(module, "config", None)
            if module_config is not None and hasattr(module_config, "use_cache"):
                module_config.use_cache = False

    def _tensor_to_image_array(self, image: torch.Tensor) -> np.ndarray:
        arr = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        return (arr * 255.0).round().astype("uint8")

    def _flatten_sample_images(self, query_image: torch.Tensor, row: int) -> list[np.ndarray]:
        query = query_image[row]
        query_flat = query.reshape(-1, *query.shape[-3:]) if query.dim() > 3 else query.unsqueeze(0)
        return [self._tensor_to_image_array(image) for image in query_flat]

    def _source_embedding(
        self,
        *,
        segment_id: int,
        modality_id: int,
        camera_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.config.use_source_embeddings:
            return torch.zeros(self.value_query.shape[-1], device=device, dtype=dtype)
        ids = torch.tensor([segment_id], dtype=torch.long, device=device)
        modality_ids = torch.tensor([modality_id], dtype=torch.long, device=device)
        camera_ids = torch.tensor([camera_id], dtype=torch.long, device=device)
        return (
            self.segment_embedding(ids)
            + self.modality_embedding(modality_ids)
            + self.camera_embedding(camera_ids)
        ).to(dtype).squeeze(0)

    def _context_slot_offsets(
        self,
        num_slots: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return learned absolute context-slot offsets, or zeros for the RoPE-only ablation."""

        hidden_size = int(self.value_query.shape[-1])
        if self.context_slot_embedding is None:
            return torch.zeros(num_slots, hidden_size, device=device, dtype=dtype)
        slot_ids = torch.arange(num_slots, dtype=torch.long, device=device)
        return self.context_slot_embedding(slot_ids).to(dtype)

    def _add_query_visual_embeddings(self, visual_tokens: torch.Tensor, query_image: torch.Tensor) -> torch.Tensor:
        batch_size = visual_tokens.shape[0]
        num_views = query_image.shape[1] if query_image.dim() == 5 else 1
        if visual_tokens.shape[1] % num_views != 0:
            raise ValueError(f"Query visual token count {visual_tokens.shape[1]} is not divisible by {num_views} views")
        tokens_per_view = visual_tokens.shape[1] // num_views
        output = visual_tokens.clone()
        for view_index in range(num_views):
            camera_id = view_index if view_index < 2 else 2
            offset = self._source_embedding(
                segment_id=1,
                modality_id=0,
                camera_id=camera_id,
                device=visual_tokens.device,
                dtype=visual_tokens.dtype,
            )
            start = view_index * tokens_per_view
            end = start + tokens_per_view
            output[:, start:end] = output[:, start:end] + offset.view(1, 1, -1)
        return output

    def _gemma_native_image_features(self, flat_images: list[np.ndarray]) -> torch.Tensor:
        if not flat_images:
            raise ValueError("Gemma3 native vision received no images to encode")
        device = next(self.language_model.parameters()).device
        processed = self.image_processor(
            images=flat_images,
            return_tensors="pt",
            do_pan_and_scan=False,
        )
        pixel_values = processed["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = torch.stack(pixel_values, dim=0)
        pixel_values = pixel_values.to(device)
        if pixel_values.dim() != 4:
            raise RuntimeError(
                "Gemma3 native image processor must return one pixel tensor per input image. "
                f"Got pixel_values shape {tuple(pixel_values.shape)}."
            )
        grad_enabled = torch.is_grad_enabled() and not self.config.freeze_backbone
        if (self.config.freeze_native_vision or self.config.use_lora) and not self.config.freeze_backbone:
            connector = getattr(self.language_model, "multi_modal_projector", None)
            if connector is None:
                raise RuntimeError("Gemma 3 native vision backbone did not expose multi_modal_projector")
            with torch.no_grad():
                vision_outputs = self.vision_tower(pixel_values=pixel_values, return_dict=True)
            with torch.set_grad_enabled(grad_enabled):
                image_features = connector(vision_outputs.last_hidden_state)
        else:
            with torch.set_grad_enabled(grad_enabled):
                image_outputs = self.language_model.get_image_features(pixel_values=pixel_values, return_dict=True)
            image_features = image_outputs.pooler_output
        if image_features.shape[1] != self.native_tokens_per_image:
            raise RuntimeError(
                f"Gemma3 native projector returned {image_features.shape[1]} tokens/image, "
                f"expected {self.native_tokens_per_image}"
            )
        return image_features

    def _merge_native_alignment_features(self, pooled_features: torch.Tensor) -> torch.Tensor:
        if self.native_spatial_merger is None:
            raise RuntimeError("Gemma3 native-style spatial merger is not initialized")
        merged = self.native_spatial_merger(pooled_features.float())
        if self.uses_dualres_native_270m_merger:
            if self.native_spatial_merger_stage2 is None:
                raise RuntimeError("Gemma3 native-style second-stage spatial merger is not initialized")
            merged = self.native_spatial_merger_stage2(merged.float())
        return merged

    def _project_native_pooled_features(
        self,
        pooled_features: torch.Tensor,
        *,
        alignment_only: bool = False,
    ) -> torch.Tensor:
        if alignment_only and not self.uses_dualres_native_270m_merger:
            raise ValueError("alignment_only is supported only by the dual-resolution native vision path")
        if self.uses_dualres_native_270m_merger:
            alignment_features = self._merge_native_alignment_features(pooled_features)
            if alignment_only:
                return alignment_features
            if getattr(self, "uses_recap_style_dualres", False):
                raise ValueError(
                    "recap_style_dualres obtains detail tokens from SigLIP; "
                    "native pooled features can only produce its alignment tokens"
                )
            detail_features = self.visual_projector(pooled_features.float())
            return torch.cat([detail_features, alignment_features], dim=1)
        if self.uses_qwen2_native_270m_merger:
            return self._merge_native_alignment_features(pooled_features)
        return self.visual_projector(pooled_features.float())

    def _native_style_270m_image_features(
        self,
        flat_images: list[np.ndarray] | torch.Tensor,
        *,
        alignment_only: bool = False,
    ) -> torch.Tensor:
        """Encode images through the 4B vision tower directly into the 270M space.

        Direct mode maps the fixed-pooled 1152-dimensional features to 640.
        Dual-resolution mode retains all 256 projected tokens and appends 16
        tokens from a shared pair of 2x2 spatial mergers.
        Compatibility mode preserves the original native 1152->2560 projector
        followed by the VFE 2560->640 connector.
        """
        if isinstance(flat_images, torch.Tensor):
            if flat_images.dim() != 4:
                raise ValueError(
                    "Gemma3 native-style tensor images must have shape [images, channels, height, width], "
                    f"got {tuple(flat_images.shape)}"
                )
            if flat_images.shape[0] == 0:
                raise ValueError("Gemma3 native-style 270M vision received no images to encode")
        elif not flat_images:
            raise ValueError("Gemma3 native-style 270M vision received no images to encode")
        if self.vision_tower is None or self.visual_projector is None:
            raise RuntimeError("Gemma3 native-style 270M vision modules are not initialized")
        if self.uses_direct_native_270m_projector and self.native_vision_pooler is None:
            raise RuntimeError("Gemma3 native-style direct pooler is not initialized")
        if self.uses_qwen2_native_270m_merger and self.native_spatial_merger is None:
            raise RuntimeError("Gemma3 native-style Qwen2 merger is not initialized")
        if self.uses_dualres_native_270m_merger and (
            self.native_spatial_merger is None or self.native_spatial_merger_stage2 is None
        ):
            raise RuntimeError("Gemma3 native-style two-stage merger is not initialized")
        if not self.uses_direct_native_270m_projector and self.native_vision_projector is None:
            raise RuntimeError("Gemma3 native-style native projector is not initialized")
        device = next(self.vision_tower.parameters()).device
        processor_kwargs: dict[str, Any] = {}
        if isinstance(flat_images, torch.Tensor):
            # Gemma3ImageProcessor uses the torchvision backend and can resize and
            # normalize a tensor batch directly on the target device. Query images
            # arrive as small pinned CPU tensors, so moving them before the 896x896
            # resize avoids the per-image NumPy loop and a large CPU-to-GPU copy.
            processor_kwargs["device"] = device
        processed = self.image_processor(
            images=flat_images,
            return_tensors="pt",
            do_pan_and_scan=False,
            **processor_kwargs,
        )
        pixel_values = processed["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = torch.stack(pixel_values, dim=0)
        pixel_values = pixel_values.to(device)
        if pixel_values.dim() != 4:
            raise RuntimeError(
                "Gemma3 native-style image processor must return one pixel tensor per input image. "
                f"Got pixel_values shape {tuple(pixel_values.shape)}."
            )

        vision_grad_enabled = torch.is_grad_enabled() and not self.config.freeze_backbone and not self.config.freeze_vision
        with torch.set_grad_enabled(vision_grad_enabled):
            vision_outputs = self.vision_tower(pixel_values=pixel_values, return_dict=True)
            if self.uses_direct_native_270m_projector:
                image_features = self._pool_native_vision_features(vision_outputs.last_hidden_state)
            else:
                image_features = self.native_vision_projector(vision_outputs.last_hidden_state)

        # The freeze boundary ends at the parameter-free 4x4 average pool.
        # Keep both dual-resolution adapters outside the vision no-grad scope.
        output_features = self._project_native_pooled_features(
            image_features,
            alignment_only=alignment_only,
        )

        expected_tokens = (
            self.native_context_tokens_per_image
            if alignment_only
            else self.native_tokens_per_image
        )
        if output_features.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Gemma3 native-style visual path returned {output_features.shape[1]} tokens/image, "
                f"expected {expected_tokens}"
            )
        return output_features.to(self._language_dtype())

    def _pool_native_vision_features(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        """Apply Gemma 3's fixed spatial pooling without its native projector."""
        if self.native_vision_pooler is None:
            raise RuntimeError("Gemma3 native-style 270M pooler is not initialized")
        if vision_outputs.dim() != 3:
            raise RuntimeError(
                "Gemma3 vision tower must return [images, patches, hidden], "
                f"got {tuple(vision_outputs.shape)}"
            )
        batch_size, patch_count, hidden_size = vision_outputs.shape
        patches_per_side = int(self.native_vision_patches_per_image)
        if patch_count != patches_per_side * patches_per_side:
            raise RuntimeError(
                f"Gemma3 vision tower returned {patch_count} patches; expected "
                f"{patches_per_side}x{patches_per_side}"
            )
        if hidden_size != int(self.native_vision_tower_hidden):
            raise RuntimeError(
                f"Gemma3 vision tower hidden size is {hidden_size}; expected "
                f"{self.native_vision_tower_hidden}"
            )
        reshaped = vision_outputs.transpose(1, 2).reshape(
            batch_size,
            hidden_size,
            patches_per_side,
            patches_per_side,
        )
        return self.native_vision_pooler(reshaped.contiguous()).flatten(2).transpose(1, 2)

    def _encode_visual_tokens(self, query_image: torch.Tensor) -> torch.Tensor:
        if self.uses_native_gemma_vision:
            return self._encode_native_query_visual_tokens(query_image)
        if self.uses_native_style_270m_vision:
            return self._encode_native_style_270m_visual_tokens(query_image)
        if self.visual_projector is None:
            raise RuntimeError("SigLIP visual projector is not initialized")
        if getattr(self.config, "query_image_preprocess_mode", "hf_cpu") == "torch_gpu":
            return self._encode_tensor_visual_tokens(query_image)
        device = next(self.vision_tower.parameters()).device
        batch_size = query_image.shape[0]
        image_counts: list[int] = []
        flat_images: list[np.ndarray] = []
        for row in range(batch_size):
            sample_images = self._flatten_sample_images(query_image, row)
            image_counts.append(len(sample_images))
            flat_images.extend(sample_images)

        processed = self.image_processor(images=flat_images, return_tensors="pt")
        processed = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in processed.items()}
        with torch.set_grad_enabled(not self.config.freeze_backbone and not self.config.freeze_vision):
            outputs = self._run_vision_tower(processed)
        image_features = self._extract_visual_token_features(outputs)
        visual_tokens = self.visual_projector(image_features.float())
        max_images = max(image_counts)
        tokens_per_image = visual_tokens.shape[1]
        padded = visual_tokens.new_zeros(batch_size, max_images * tokens_per_image, visual_tokens.shape[-1])
        cursor = 0
        for row, count in enumerate(image_counts):
            sample_tokens = visual_tokens[cursor : cursor + count].reshape(count * tokens_per_image, -1)
            padded[row, : sample_tokens.shape[0]] = sample_tokens
            cursor += count
        return padded

    def _encode_tensor_visual_tokens(self, query_image: torch.Tensor) -> torch.Tensor:
        """Resize and normalize a dense image batch on the vision-tower device."""
        if query_image.ndim == 4:
            query_image = query_image.unsqueeze(1)
        if query_image.ndim != 5:
            raise ValueError(
                "GPU query-image preprocessing expects [B,C,H,W] or [B,V,C,H,W], "
                f"got shape {tuple(query_image.shape)}"
            )
        batch_size, image_count, channels, height, width = query_image.shape
        if channels != 3:
            raise ValueError(f"GPU query-image preprocessing requires RGB input, got {channels} channels")

        device = next(self.vision_tower.parameters()).device
        flat_images = query_image.reshape(
            batch_size * image_count, channels, height, width
        ).to(device=device, non_blocking=True)
        if flat_images.is_floating_point():
            # Match the previous CPU path's quantization exactly before the
            # official torchvision processor performs batched resize/normalize.
            flat_images = flat_images.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        elif flat_images.dtype == torch.uint8:
            pass
        else:
            raise TypeError(
                "GPU query-image preprocessing expects uint8 or floating-point images, "
                f"got {flat_images.dtype}"
            )

        processed = self.image_processor(
            images=flat_images,
            return_tensors="pt",
            device=device,
        )
        processed = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in processed.items()
        }

        with torch.set_grad_enabled(not self.config.freeze_backbone and not self.config.freeze_vision):
            outputs = self._run_vision_tower(processed)
        image_features = self._extract_visual_token_features(outputs)
        visual_tokens = self.visual_projector(image_features.float())
        return visual_tokens.reshape(
            batch_size,
            image_count * visual_tokens.shape[1],
            visual_tokens.shape[-1],
        )

    def _encode_native_query_visual_tokens(self, query_image: torch.Tensor) -> torch.Tensor:
        batch_size = query_image.shape[0]
        image_counts: list[int] = []
        flat_images: list[np.ndarray] = []
        for row in range(batch_size):
            sample_images = self._flatten_sample_images(query_image, row)
            image_counts.append(len(sample_images))
            flat_images.extend(sample_images)

        visual_tokens = self._gemma_native_image_features(flat_images)
        max_images = max(image_counts)
        tokens_per_image = visual_tokens.shape[1]
        padded = visual_tokens.new_zeros(batch_size, max_images * tokens_per_image, visual_tokens.shape[-1])
        cursor = 0
        for row, count in enumerate(image_counts):
            sample_tokens = visual_tokens[cursor : cursor + count].reshape(count * tokens_per_image, -1)
            padded[row, : sample_tokens.shape[0]] = sample_tokens
            cursor += count
        return padded

    def _encode_native_style_270m_visual_tokens(self, query_image: torch.Tensor) -> torch.Tensor:
        batch_size = query_image.shape[0]
        if query_image.dim() == 5:
            num_views = query_image.shape[1]
            flat_images = query_image.reshape(batch_size * num_views, *query_image.shape[-3:])
        elif query_image.dim() == 4:
            num_views = 1
            flat_images = query_image
        else:
            raise ValueError(
                "Gemma3 native-style query images must have shape [batch, channels, height, width] "
                f"or [batch, views, channels, height, width], got {tuple(query_image.shape)}"
            )
        if flat_images.shape[1] != 3:
            raise ValueError(f"Gemma3 native-style query images must have 3 channels, got {flat_images.shape[1]}")

        if getattr(self, "uses_recap_style_dualres", False):
            visual_tokens = self._recap_style_dualres_image_features(flat_images)
            return visual_tokens.reshape(
                batch_size,
                num_views * visual_tokens.shape[1],
                visual_tokens.shape[-1],
            )

        device = next(self.vision_tower.parameters()).device
        flat_images = flat_images.to(device=device, non_blocking=True)
        if flat_images.is_floating_point():
            # Preserve the previous Tensor -> uint8 NumPy conversion exactly. The
            # processor expects uint8 here because its configured rescale factor is
            # 1/255; passing the dataset's [0, 1] floats would rescale twice.
            flat_images = (flat_images.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        elif flat_images.dtype != torch.uint8:
            flat_images = flat_images.clamp(0, 255).to(torch.uint8)

        visual_tokens = self._native_style_270m_image_features(flat_images)
        tokens_per_image = visual_tokens.shape[1]
        return visual_tokens.reshape(batch_size, num_views * tokens_per_image, visual_tokens.shape[-1])

    def _recap_style_dualres_image_features(self, flat_images: torch.Tensor) -> torch.Tensor:
        """Combine a SigLIP detail branch with a 16-token alignment branch."""

        if flat_images.dim() != 4 or flat_images.shape[1] != 3:
            raise ValueError("RECAP-style query images must be [images, 3, height, width]")
        if self.recap_image_processor is None or self.recap_vision_tower is None:
            raise RuntimeError("recap_style_dualres did not initialize the SigLIP vision path")
        if self.visual_projector is None:
            raise RuntimeError("recap_style_dualres did not initialize the SigLIP connector")

        recap_device = next(self.recap_vision_tower.parameters()).device
        processor_kwargs: dict[str, Any] = {}
        if getattr(self, "uses_recap_qwen2_512_dualres", False):
            recap_images = flat_images.to(device=recap_device, non_blocking=True)
            if recap_images.is_floating_point():
                recap_images = (
                    recap_images.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
                )
            elif recap_images.dtype != torch.uint8:
                recap_images = recap_images.clamp(0, 255).to(torch.uint8)
            processor_kwargs["device"] = recap_device
        else:
            recap_images = [self._tensor_to_image_array(image) for image in flat_images]
        processed = self.recap_image_processor(
            images=recap_images,
            return_tensors="pt",
            **processor_kwargs,
        )
        processed = {
            key: value.to(recap_device) if isinstance(value, torch.Tensor) else value
            for key, value in processed.items()
        }
        vision_grad_enabled = (
            torch.is_grad_enabled()
            and not self.config.freeze_backbone
            and not self.config.freeze_vision
        )
        with torch.set_grad_enabled(vision_grad_enabled):
            recap_outputs = self._run_vision_tower(processed, tower=self.recap_vision_tower)
        recap_features = self._extract_visual_token_features(recap_outputs)
        detail_tokens = self.visual_projector(recap_features.float()).to(self._language_dtype())
        expected_detail_tokens = int(
            getattr(self, "recap_detail_tokens_per_image", self.recap_tokens_per_image)
        )
        if detail_tokens.shape[1] != expected_detail_tokens:
            raise RuntimeError(
                f"RECAP SigLIP returned {detail_tokens.shape[1]} tokens/image; "
                f"expected {expected_detail_tokens}"
            )

        if getattr(self, "uses_recap_qwen2_512_dualres", False):
            alignment_tokens = self._recap_qwen2_alignment_from_detail_tokens(detail_tokens)
        else:
            native_device = next(self.vision_tower.parameters()).device
            native_images = flat_images.to(device=native_device, non_blocking=True)
            if native_images.is_floating_point():
                native_images = (
                    native_images.detach().clamp(0.0, 1.0) * 255.0
                ).round().to(torch.uint8)
            elif native_images.dtype != torch.uint8:
                native_images = native_images.clamp(0, 255).to(torch.uint8)
            alignment_tokens = self._native_style_270m_image_features(
                native_images,
                alignment_only=True,
            )
        combined = torch.cat([detail_tokens, alignment_tokens], dim=1)
        if combined.shape[1] != int(self.native_tokens_per_image):
            raise RuntimeError(
                f"RECAP dual-resolution path returned {combined.shape[1]} tokens/image; "
                f"expected {self.native_tokens_per_image}"
            )
        return combined

    def _recap_qwen2_alignment_from_patch_features(
        self,
        patch_features: torch.Tensor,
    ) -> torch.Tensor:
        """Use the exact shared 1024->256->64->16 RECAP path for context."""

        if self.visual_projector is None:
            raise RuntimeError("RECAP-style Qwen2 vision did not initialize the 1024-to-256 merger")
        detail_tokens = self.visual_projector(patch_features.float()).to(self._language_dtype())
        expected_detail_tokens = int(self.recap_detail_tokens_per_image)
        if detail_tokens.shape[1] != expected_detail_tokens:
            raise RuntimeError(
                f"RECAP Qwen2 detail branch returned {detail_tokens.shape[1]} tokens/image; "
                f"expected {expected_detail_tokens}"
            )
        return self._recap_qwen2_alignment_from_detail_tokens(detail_tokens)

    def _recap_qwen2_alignment_from_detail_tokens(
        self,
        detail_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.recap_spatial_merger_stage2 is None or self.recap_spatial_merger_stage3 is None:
            raise RuntimeError(
                "RECAP-style Qwen2 dual-resolution vision did not initialize both "
                "post-detail spatial mergers"
            )
        alignment_tokens = self.recap_spatial_merger_stage2(detail_tokens.float())
        alignment_tokens = self.recap_spatial_merger_stage3(alignment_tokens.float())
        alignment_tokens = alignment_tokens.to(self._language_dtype())
        expected_alignment_tokens = int(self.recap_alignment_tokens_per_image)
        if alignment_tokens.shape[1] != expected_alignment_tokens:
            raise RuntimeError(
                f"RECAP Qwen2 alignment branch returned {alignment_tokens.shape[1]} tokens/image; "
                f"expected {expected_alignment_tokens}"
            )
        return alignment_tokens

    def _run_vision_tower(self, processed: dict[str, torch.Tensor], tower: nn.Module | None = None) -> Any:
        tower = tower or self.vision_tower
        pixel_values = processed.get("pixel_values")
        if pixel_values is not None and hasattr(tower, "vision_model"):
            return tower.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            )
        try:
            return tower(**processed, output_hidden_states=False, return_dict=True)
        except (TypeError, ValueError):
            if pixel_values is not None and hasattr(tower, "get_image_features"):
                return {"image_embeds": tower.get_image_features(pixel_values=pixel_values)}
            raise

    def _extract_visual_token_features(self, outputs: Any) -> torch.Tensor:
        if isinstance(outputs, dict) and "image_embeds" in outputs:
            return outputs["image_embeds"].unsqueeze(1)
        vision_output = getattr(outputs, "vision_model_output", None)
        if vision_output is not None and getattr(vision_output, "last_hidden_state", None) is not None:
            return vision_output.last_hidden_state
        if getattr(outputs, "last_hidden_state", None) is not None:
            return outputs.last_hidden_state
        if getattr(outputs, "hidden_states", None) is not None:
            return outputs.hidden_states[-1]
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output.unsqueeze(1)
        if getattr(outputs, "image_embeds", None) is not None:
            return outputs.image_embeds.unsqueeze(1)
        raise RuntimeError("SigLIP vision tower did not return visual token features")

    def _extract_context_patch_features(self, outputs: Any) -> torch.Tensor:
        features = self._extract_visual_token_features(outputs)
        expected = int(self.config.context_patch_tokens_per_image)
        if features.shape[1] < expected:
            raise RuntimeError(
                f"DINO context tower returned {features.shape[1]} tokens, expected at least {expected} patch tokens"
            )
        return features[:, -expected:]

    def _encode_text_embeddings(self, language: list[str] | torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(language, torch.Tensor):
            raise TypeError("SigLIP+Gemma VFE expects raw language strings, not token ids")
        tokens = self.tokenizer(
            language,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_language_tokens,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        embeddings = self.language_model.get_input_embeddings()(tokens["input_ids"])
        return embeddings, tokens["attention_mask"]

    def _pool_hidden(self, outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        if getattr(outputs, "hidden_states", None) is not None:
            hidden = outputs.hidden_states[-1]
        elif getattr(outputs, "last_hidden_state", None) is not None:
            hidden = outputs.last_hidden_state
        else:
            raise RuntimeError("Gemma language model did not return hidden states")
        mask = attention_mask.to(hidden.device).to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _value_hidden(self, outputs: Any) -> torch.Tensor:
        if getattr(outputs, "hidden_states", None) is not None:
            hidden = outputs.hidden_states[-1]
        elif getattr(outputs, "last_hidden_state", None) is not None:
            hidden = outputs.last_hidden_state
        else:
            raise RuntimeError("Gemma language model did not return hidden states")
        return hidden[:, -1]

    def _language_dtype(self) -> torch.dtype:
        return next(self.language_model.parameters()).dtype

    def _reference_embedding_rms(self, text_embeddings: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        mask = text_mask.to(device=text_embeddings.device, dtype=torch.bool)
        if bool(mask.any().item()):
            values = text_embeddings.detach()[mask]
        else:
            values = text_embeddings.detach().reshape(-1, text_embeddings.shape[-1])
        return values.float().pow(2).mean().sqrt().clamp_min(1e-6)

    def _match_text_embedding_rms(
        self,
        tokens: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.match_adapter_token_rms:
            return tokens
        target_rms = self._reference_embedding_rms(text_embeddings, text_mask) * float(self.config.adapter_token_scale)
        token_rms = tokens.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        scale = (target_rms / token_rms).to(device=tokens.device, dtype=tokens.dtype)
        return tokens * scale

    def _normalize_adapter_tokens(
        self,
        tokens: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = tokens.dtype
        normalized = self.adapter_output_norm(tokens.float()).to(dtype=original_dtype)
        return self._match_text_embedding_rms(normalized, text_embeddings, text_mask)

    def _embedding_stats(self, name: str, tensor: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, object]:
        values = tensor.detach()
        if mask is not None:
            mask_bool = mask.to(device=values.device, dtype=torch.bool)
            if mask_bool.shape == values.shape[: mask_bool.dim()] and bool(mask_bool.any().item()):
                values = values[mask_bool]
        if values.dim() >= 2:
            values = values.reshape(-1, values.shape[-1])
            max_tokens = 4096
            if values.shape[0] > max_tokens:
                stride = max(1, values.shape[0] // max_tokens)
                values = values[::stride][:max_tokens]
        values_float = values.float()
        return {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "finite": bool(torch.isfinite(values_float).all().item()),
            "mean": float(values_float.mean().item()),
            "std": float(values_float.std(unbiased=False).item()),
            "rms": float(values_float.pow(2).mean().sqrt().item()),
            "absmax": float(values_float.abs().max().item()),
        }

    def _maybe_log_input_embedding_stats(
        self,
        parts: dict[str, tuple[torch.Tensor, torch.Tensor | None]],
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        if not self.config.log_input_embedding_stats or self._printed_input_embedding_stats:
            return
        stats = [self._embedding_stats(name, tensor, mask) for name, (tensor, mask) in parts.items()]
        stats.append(self._embedding_stats("all_input_embeddings", input_embeddings, attention_mask))
        print("input embedding stats:", stats, flush=True)
        self._printed_input_embedding_stats = True

    def _encode_context_tokens(
        self,
        context_images: torch.Tensor | None,
        context_proprio: torch.Tensor | None,
        context_valid_mask: torch.Tensor | None,
        context_progress: torch.Tensor | None = None,
        context_native_features: torch.Tensor | None = None,
        context_feature_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.uses_context:
            raise RuntimeError("_encode_context_tokens called for a no-context VFE")
        has_cached_features = context_native_features is not None or context_feature_positions is not None
        if (context_native_features is None) != (context_feature_positions is None):
            raise ValueError("Provide both context_native_features and context_feature_positions, or neither")
        if context_proprio is None or context_valid_mask is None:
            raise ValueError(
                f"{self.config.backbone} requires context_proprio and context_valid_mask in the batch"
            )
        if bool(getattr(self.config, "context_progress_enabled", False)):
            if context_progress is None:
                raise ValueError(
                    f"{self.config.backbone} requires context_progress when context progress is enabled"
                )
            if context_progress.shape != context_valid_mask.shape:
                raise ValueError(
                    "context_progress and context_valid_mask must both have shape [batch, slots]"
                )
        if has_cached_features:
            if context_images is not None:
                raise ValueError("Provide either context_images or cached context native features, not both")
            if not self.uses_native_style_270m_vision:
                raise ValueError("Cached context native features require a gemma3_native_style_270m context backbone")
            return self._encode_native_style_270m_context_tokens(
                None,
                context_proprio,
                context_valid_mask,
                context_progress,
                context_native_features=context_native_features,
                context_feature_positions=context_feature_positions,
            )
        if context_images is None:
            raise ValueError(f"{self.config.backbone} requires context_images when no native feature cache is used")
        if self.uses_native_gemma_vision:
            return self._encode_native_context_tokens(
                context_images, context_proprio, context_valid_mask, context_progress
            )
        if self.uses_native_style_270m_vision:
            return self._encode_native_style_270m_context_tokens(
                context_images, context_proprio, context_valid_mask, context_progress
            )
        if self.context_vision_tower is None or self.context_image_processor is None or self.context_visual_projector is None:
            raise RuntimeError("Context VFE was not initialized with DINO context modules")

        device = next(self.context_vision_tower.parameters()).device
        language_dtype = self._language_dtype()
        batch_size, num_slots, num_views = context_images.shape[:3]
        expected_slots = int(self.config.context_max_timesteps)
        expected_views = int(self.config.context_views)
        patch_tokens = int(self.config.context_patch_tokens_per_image)
        if num_slots != expected_slots:
            raise ValueError(f"context_images has {num_slots} slots, expected {expected_slots}")
        if num_views != expected_views:
            raise ValueError(f"context_images has {num_views} views, expected {expected_views}")
        if self.config.proprio_dim > 0 and context_proprio.shape[-1] != self.config.proprio_dim:
            raise ValueError(
                f"context_proprio dim {context_proprio.shape[-1]} does not match model proprio_dim {self.config.proprio_dim}"
            )

        progress_enabled = bool(getattr(self.config, "context_progress_enabled", False))
        block_tokens = num_views * patch_tokens + 1 + int(progress_enabled)
        proprio_position = num_views * patch_tokens
        progress_position = proprio_position + 1
        hidden_size = self.value_query.shape[-1]
        context_tokens = torch.zeros(
            batch_size,
            num_slots,
            block_tokens,
            hidden_size,
            device=device,
            dtype=language_dtype,
        )
        valid = context_valid_mask.to(device=device, dtype=torch.bool)

        flat_images: list[np.ndarray] = []
        flat_positions: list[tuple[int, int, int]] = []
        for batch_index in range(batch_size):
            for slot_index in range(num_slots):
                if not bool(valid[batch_index, slot_index].item()):
                    continue
                for view_index in range(num_views):
                    flat_images.append(self._tensor_to_image_array(context_images[batch_index, slot_index, view_index]))
                    flat_positions.append((batch_index, slot_index, view_index))

        chunk_size = max(1, int(self.config.context_image_chunk_size))
        context_image_size = int(self.config.context_image_size)
        for start in range(0, len(flat_images), chunk_size):
            end = min(start + chunk_size, len(flat_images))
            processed = self.context_image_processor(
                images=flat_images[start:end],
                size={"height": context_image_size, "width": context_image_size},
                return_tensors="pt",
            )
            processed = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in processed.items()}
            pixel_values = processed.get("pixel_values")
            if isinstance(pixel_values, torch.Tensor) and pixel_values.shape[-2:] != (context_image_size, context_image_size):
                raise RuntimeError(
                    "DINO context image processor returned pixel_values with spatial shape "
                    f"{tuple(pixel_values.shape[-2:])}, expected {(context_image_size, context_image_size)}"
                )
            with torch.set_grad_enabled(not self.config.freeze_backbone):
                outputs = self._run_vision_tower(processed, tower=self.context_vision_tower)
            patch_features = self._extract_context_patch_features(outputs)
            projected = self.context_visual_projector(patch_features.float()).to(language_dtype)
            for row, (batch_index, slot_index, view_index) in enumerate(flat_positions[start:end]):
                token_start = view_index * patch_tokens
                token_end = token_start + patch_tokens
                context_tokens[batch_index, slot_index, token_start:token_end] = projected[row]

        if self.config.proprio_dim > 0:
            context_proprio_flat = context_proprio.to(device=device, dtype=torch.float32).reshape(batch_size * num_slots, -1)
            proprio_tokens = self.proprio_encoder(context_proprio_flat, batch_size * num_slots, device).reshape(
                batch_size,
                num_slots,
                hidden_size,
            )
            context_tokens[:, :, proprio_position] = proprio_tokens.to(language_dtype)
        if progress_enabled:
            if context_progress is None or self.context_progress_encoder is None:
                raise ValueError("Context progress is enabled but no progress tensor/encoder is available")
            progress_flat = context_progress.to(device=device, dtype=torch.float32).reshape(
                batch_size * num_slots, 1
            )
            progress_tokens = self.context_progress_encoder(
                progress_flat, batch_size * num_slots, device
            ).reshape(batch_size, num_slots, hidden_size)
            context_tokens[:, :, progress_position] = progress_tokens.to(language_dtype)

        slot_offsets = self._context_slot_offsets(
            num_slots, device=device, dtype=language_dtype
        )
        context_segment_visual = (
            self.segment_embedding(torch.tensor([0], dtype=torch.long, device=device))
            + self.modality_embedding(torch.tensor([0], dtype=torch.long, device=device))
        ).to(language_dtype).squeeze(0)
        context_segment_proprio = self._source_embedding(
            segment_id=0,
            modality_id=1,
            camera_id=2,
            device=device,
            dtype=language_dtype,
        )
        context_segment_progress = self._source_embedding(
            segment_id=0,
            modality_id=3,
            camera_id=2,
            device=device,
            dtype=language_dtype,
        )
        for view_index in range(num_views):
            camera_offset = self.camera_embedding(
                torch.tensor([view_index if view_index < 2 else 2], dtype=torch.long, device=device)
            ).to(language_dtype).squeeze(0)
            token_start = view_index * patch_tokens
            token_end = token_start + patch_tokens
            context_tokens[:, :, token_start:token_end] = (
                context_tokens[:, :, token_start:token_end]
                + context_segment_visual.view(1, 1, 1, -1)
                + camera_offset.view(1, 1, 1, -1)
                + slot_offsets.view(1, num_slots, 1, -1)
            )
        context_tokens[:, :, proprio_position] = (
            context_tokens[:, :, proprio_position]
            + context_segment_proprio.view(1, 1, -1)
            + slot_offsets.view(1, num_slots, -1)
        )
        if progress_enabled:
            context_tokens[:, :, progress_position] = (
                context_tokens[:, :, progress_position]
                + context_segment_progress.view(1, 1, -1)
                + slot_offsets.view(1, num_slots, -1)
            )

        context_attention_mask = valid.unsqueeze(-1).expand(batch_size, num_slots, block_tokens).reshape(
            batch_size,
            num_slots * block_tokens,
        )
        return context_tokens.reshape(batch_size, num_slots * block_tokens, hidden_size), context_attention_mask

    @torch.no_grad()
    def encode_context_tokens_for_eval(
        self,
        context_images: torch.Tensor | None,
        context_proprio: torch.Tensor,
        context_valid_mask: torch.Tensor,
        context_progress: torch.Tensor | None = None,
        context_native_features: torch.Tensor | None = None,
        context_feature_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one or more fixed context demos for the evaluation-only token cache.

        The returned tokens are intentionally before adapter normalization.  That
        normalization is still applied in ``forward`` for every query so cached
        contexts preserve the exact normal training/evaluation computation.
        """

        if not self.uses_context:
            raise ValueError("Context-token caching requires a context VFE backbone")
        context_tokens, context_mask = self._encode_context_tokens(
            context_images,
            context_proprio,
            context_valid_mask,
            context_progress,
            context_native_features,
            context_feature_positions,
        )
        return context_tokens.detach(), context_mask.detach()

    def _encode_native_style_270m_context_tokens(
        self,
        context_images: torch.Tensor | None,
        context_proprio: torch.Tensor,
        context_valid_mask: torch.Tensor,
        context_progress: torch.Tensor | None = None,
        *,
        context_native_features: torch.Tensor | None = None,
        context_feature_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode context through the visual path shared with query alignment tokens."""
        if self.vision_tower is None:
            raise RuntimeError("Gemma3 native-style 270M backbone did not expose a vision tower")
        device = next(self.vision_tower.parameters()).device
        language_dtype = self._language_dtype()
        if context_valid_mask.dim() != 2:
            raise ValueError("context_valid_mask must have shape [batch, slots]")
        batch_size, num_slots = context_valid_mask.shape
        num_views = int(self.config.context_views)
        expected_slots = int(self.config.context_max_timesteps)
        expected_views = int(self.config.context_views)
        tokens_per_image = int(self.native_context_tokens_per_image)
        if context_images is not None:
            image_batch_size, image_slots, image_views = context_images.shape[:3]
            if image_batch_size != batch_size or image_slots != num_slots:
                raise ValueError("context_images and context_valid_mask batch/slot dimensions do not match")
            num_views = int(image_views)
        if num_slots != expected_slots:
            raise ValueError(f"context data has {num_slots} slots, expected {expected_slots}")
        if num_views != expected_views:
            raise ValueError(f"context data has {num_views} views, expected {expected_views}")
        if self.config.proprio_dim > 0 and context_proprio.shape[-1] != self.config.proprio_dim:
            raise ValueError(
                f"context_proprio dim {context_proprio.shape[-1]} does not match model proprio_dim {self.config.proprio_dim}"
            )

        progress_enabled = bool(getattr(self.config, "context_progress_enabled", False))
        block_tokens = num_views * tokens_per_image + 1 + int(progress_enabled)
        proprio_position = num_views * tokens_per_image
        progress_position = proprio_position + 1
        hidden_size = self.value_query.shape[-1]
        context_tokens = torch.zeros(
            batch_size,
            num_slots,
            block_tokens,
            hidden_size,
            device=device,
            dtype=language_dtype,
        )
        valid = context_valid_mask.to(device=device, dtype=torch.bool)

        chunk_size = max(1, int(self.config.context_image_chunk_size))
        if context_native_features is not None:
            if context_feature_positions is None:
                raise ValueError("Cached context features require context_feature_positions")
            if context_native_features.dim() != 3:
                raise ValueError("context_native_features must have shape [images, tokens, hidden]")
            if context_feature_positions.shape != (context_native_features.shape[0], 3):
                raise ValueError("context_feature_positions must have shape [images, 3]")
            feature_tail = tuple(context_native_features.shape[1:])
            if getattr(self, "uses_recap_qwen2_512_dualres", False):
                supported_feature_shapes = {
                    (int(self.recap_tokens_per_image), int(self.recap_vision_hidden)),
                    (int(self.recap_detail_tokens_per_image), int(self.value_query.shape[-1])),
                }
            elif self.uses_direct_native_270m_projector:
                cached_tokens_per_image = (
                    int(self.native_pooled_tokens_per_image)
                    if self.uses_qwen2_native_270m_merger or self.uses_dualres_native_270m_merger
                    else tokens_per_image
                )
                supported_feature_shapes = {
                    (cached_tokens_per_image, int(self.native_vision_tower_hidden))
                }
            else:
                supported_feature_shapes = {
                    (tokens_per_image, int(self.native_vision_tower_hidden)),
                    (tokens_per_image, int(self.native_vision_hidden)),
                }
            if feature_tail not in supported_feature_shapes:
                raise ValueError(
                    f"Cached context feature shape {feature_tail} does not match any supported stage: "
                    f"{sorted(supported_feature_shapes)}"
                )
            positions = context_feature_positions.detach().cpu().tolist()
            for batch_index, slot_index, view_index in positions:
                if not (0 <= batch_index < batch_size and 0 <= slot_index < num_slots and 0 <= view_index < num_views):
                    raise IndexError(f"Invalid cached context feature position {(batch_index, slot_index, view_index)}")
                if not bool(valid[batch_index, slot_index].item()):
                    raise ValueError(f"Cached context feature points to invalid slot {(batch_index, slot_index)}")
            for start in range(0, context_native_features.shape[0], chunk_size):
                end = min(start + chunk_size, context_native_features.shape[0])
                # DataLoader pinning makes this transfer genuinely asynchronous.
                # Enqueue cached context chunks without synchronizing the CPU on
                # every small H2D copy; subsequent merger ops stay ordered on the
                # same CUDA stream.
                cached_chunk = context_native_features[start:end].to(
                    device=device,
                    non_blocking=True,
                )
                if (
                    not self.uses_direct_native_270m_projector
                    and cached_chunk.shape[-1] == self.native_vision_tower_hidden
                ):
                    if self.native_vision_projector is None:
                        raise RuntimeError("Gemma3 native-style native projector is not initialized")
                    native_weight = self.native_vision_projector.mm_input_projection_weight
                    with torch.no_grad():
                        cached_chunk = torch.matmul(cached_chunk.to(native_weight.dtype), native_weight)
                if getattr(self, "uses_recap_qwen2_512_dualres", False):
                    if cached_chunk.shape[1] == int(self.recap_detail_tokens_per_image):
                        projected = self._recap_qwen2_alignment_from_detail_tokens(cached_chunk)
                    else:
                        projected = self._recap_qwen2_alignment_from_patch_features(
                            cached_chunk
                        )
                elif self.uses_qwen2_native_270m_merger or self.uses_dualres_native_270m_merger:
                    projected = self._merge_native_alignment_features(cached_chunk).to(language_dtype)
                else:
                    projected = self.visual_projector(cached_chunk.float()).to(language_dtype)
                for row, (batch_index, slot_index, view_index) in enumerate(positions[start:end]):
                    token_start = view_index * tokens_per_image
                    token_end = token_start + tokens_per_image
                    context_tokens[batch_index, slot_index, token_start:token_end] = projected[row]
        else:
            if context_images is None:
                raise ValueError("Raw context encoding requires context_images")
            flat_images: list[np.ndarray] = []
            flat_recap_images: list[torch.Tensor] = []
            flat_positions: list[tuple[int, int, int]] = []
            for batch_index in range(batch_size):
                for slot_index in range(num_slots):
                    if not bool(valid[batch_index, slot_index].item()):
                        continue
                    for view_index in range(num_views):
                        context_image = context_images[batch_index, slot_index, view_index]
                        if getattr(self, "uses_recap_qwen2_512_dualres", False):
                            flat_recap_images.append(context_image)
                        else:
                            flat_images.append(self._tensor_to_image_array(context_image))
                        flat_positions.append((batch_index, slot_index, view_index))

            image_count = (
                len(flat_recap_images)
                if getattr(self, "uses_recap_qwen2_512_dualres", False)
                else len(flat_images)
            )
            for start in range(0, image_count, chunk_size):
                end = min(start + chunk_size, image_count)
                if getattr(self, "uses_recap_qwen2_512_dualres", False):
                    recap_tokens = self._recap_style_dualres_image_features(
                        torch.stack(flat_recap_images[start:end], dim=0)
                    )
                    projected = recap_tokens[:, -tokens_per_image:]
                else:
                    projected = self._native_style_270m_image_features(
                        flat_images[start:end],
                        alignment_only=self.uses_dualres_native_270m_merger,
                    )
                for row, (batch_index, slot_index, view_index) in enumerate(flat_positions[start:end]):
                    token_start = view_index * tokens_per_image
                    token_end = token_start + tokens_per_image
                    context_tokens[batch_index, slot_index, token_start:token_end] = projected[row]

        if self.config.proprio_dim > 0:
            context_proprio_flat = context_proprio.to(device=device, dtype=torch.float32).reshape(batch_size * num_slots, -1)
            proprio_tokens = self.proprio_encoder(context_proprio_flat, batch_size * num_slots, device).reshape(
                batch_size,
                num_slots,
                hidden_size,
            )
            context_tokens[:, :, proprio_position] = proprio_tokens.to(language_dtype)
        if progress_enabled:
            if context_progress is None or self.context_progress_encoder is None:
                raise ValueError("Context progress is enabled but no progress tensor/encoder is available")
            progress_flat = context_progress.to(device=device, dtype=torch.float32).reshape(
                batch_size * num_slots, 1
            )
            progress_tokens = self.context_progress_encoder(
                progress_flat, batch_size * num_slots, device
            ).reshape(batch_size, num_slots, hidden_size)
            context_tokens[:, :, progress_position] = progress_tokens.to(language_dtype)

        slot_offsets = self._context_slot_offsets(
            num_slots, device=device, dtype=language_dtype
        )
        context_segment_visual = (
            self.segment_embedding(torch.tensor([0], dtype=torch.long, device=device))
            + self.modality_embedding(torch.tensor([0], dtype=torch.long, device=device))
        ).to(language_dtype).squeeze(0)
        context_segment_proprio = self._source_embedding(
            segment_id=0,
            modality_id=1,
            camera_id=2,
            device=device,
            dtype=language_dtype,
        )
        context_segment_progress = self._source_embedding(
            segment_id=0,
            modality_id=3,
            camera_id=2,
            device=device,
            dtype=language_dtype,
        )
        for view_index in range(num_views):
            camera_offset = self.camera_embedding(
                torch.tensor([view_index if view_index < 2 else 2], dtype=torch.long, device=device)
            ).to(language_dtype).squeeze(0)
            token_start = view_index * tokens_per_image
            token_end = token_start + tokens_per_image
            context_tokens[:, :, token_start:token_end] = (
                context_tokens[:, :, token_start:token_end]
                + context_segment_visual.view(1, 1, 1, -1)
                + camera_offset.view(1, 1, 1, -1)
                + slot_offsets.view(1, num_slots, 1, -1)
            )
        context_tokens[:, :, proprio_position] = (
            context_tokens[:, :, proprio_position]
            + context_segment_proprio.view(1, 1, -1)
            + slot_offsets.view(1, num_slots, -1)
        )
        if progress_enabled:
            context_tokens[:, :, progress_position] = (
                context_tokens[:, :, progress_position]
                + context_segment_progress.view(1, 1, -1)
                + slot_offsets.view(1, num_slots, -1)
            )

        context_attention_mask = valid.unsqueeze(-1).expand(batch_size, num_slots, block_tokens).reshape(
            batch_size,
            num_slots * block_tokens,
        )
        return context_tokens.reshape(batch_size, num_slots * block_tokens, hidden_size), context_attention_mask

    def _encode_native_context_tokens(
        self,
        context_images: torch.Tensor,
        context_proprio: torch.Tensor,
        context_valid_mask: torch.Tensor,
        context_progress: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.language_model.parameters()).device
        language_dtype = self._language_dtype()
        batch_size, num_slots, num_views = context_images.shape[:3]
        expected_slots = int(self.config.context_max_timesteps)
        expected_views = int(self.config.context_views)
        tokens_per_image = int(self.native_tokens_per_image)
        if num_slots != expected_slots:
            raise ValueError(f"context_images has {num_slots} slots, expected {expected_slots}")
        if num_views != expected_views:
            raise ValueError(f"context_images has {num_views} views, expected {expected_views}")
        if self.config.proprio_dim > 0 and context_proprio.shape[-1] != self.config.proprio_dim:
            raise ValueError(
                f"context_proprio dim {context_proprio.shape[-1]} does not match model proprio_dim {self.config.proprio_dim}"
            )

        progress_enabled = bool(getattr(self.config, "context_progress_enabled", False))
        block_tokens = num_views * tokens_per_image + 1 + int(progress_enabled)
        proprio_position = num_views * tokens_per_image
        progress_position = proprio_position + 1
        hidden_size = self.value_query.shape[-1]
        context_tokens = torch.zeros(
            batch_size,
            num_slots,
            block_tokens,
            hidden_size,
            device=device,
            dtype=language_dtype,
        )
        valid = context_valid_mask.to(device=device, dtype=torch.bool)

        flat_images: list[np.ndarray] = []
        flat_positions: list[tuple[int, int, int]] = []
        for batch_index in range(batch_size):
            for slot_index in range(num_slots):
                if not bool(valid[batch_index, slot_index].item()):
                    continue
                for view_index in range(num_views):
                    flat_images.append(self._tensor_to_image_array(context_images[batch_index, slot_index, view_index]))
                    flat_positions.append((batch_index, slot_index, view_index))

        chunk_size = max(1, int(self.config.context_image_chunk_size))
        for start in range(0, len(flat_images), chunk_size):
            end = min(start + chunk_size, len(flat_images))
            projected = self._gemma_native_image_features(flat_images[start:end]).to(language_dtype)
            for row, (batch_index, slot_index, view_index) in enumerate(flat_positions[start:end]):
                token_start = view_index * tokens_per_image
                token_end = token_start + tokens_per_image
                context_tokens[batch_index, slot_index, token_start:token_end] = projected[row]

        if self.config.proprio_dim > 0:
            context_proprio_flat = context_proprio.to(device=device, dtype=torch.float32).reshape(batch_size * num_slots, -1)
            proprio_tokens = self.proprio_encoder(context_proprio_flat, batch_size * num_slots, device).reshape(
                batch_size,
                num_slots,
                hidden_size,
            )
            context_tokens[:, :, proprio_position] = proprio_tokens.to(language_dtype)
        if progress_enabled:
            if context_progress is None or self.context_progress_encoder is None:
                raise ValueError("Context progress is enabled but no progress tensor/encoder is available")
            progress_flat = context_progress.to(device=device, dtype=torch.float32).reshape(
                batch_size * num_slots, 1
            )
            progress_tokens = self.context_progress_encoder(
                progress_flat, batch_size * num_slots, device
            ).reshape(batch_size, num_slots, hidden_size)
            context_tokens[:, :, progress_position] = progress_tokens.to(language_dtype)

        slot_offsets = self._context_slot_offsets(
            num_slots, device=device, dtype=language_dtype
        )
        context_segment_visual = (
            self.segment_embedding(torch.tensor([0], dtype=torch.long, device=device))
            + self.modality_embedding(torch.tensor([0], dtype=torch.long, device=device))
        ).to(language_dtype).squeeze(0)
        context_segment_proprio = self._source_embedding(
            segment_id=0,
            modality_id=1,
            camera_id=2,
            device=device,
            dtype=language_dtype,
        )
        context_segment_progress = self._source_embedding(
            segment_id=0,
            modality_id=3,
            camera_id=2,
            device=device,
            dtype=language_dtype,
        )
        for view_index in range(num_views):
            camera_offset = self.camera_embedding(
                torch.tensor([view_index if view_index < 2 else 2], dtype=torch.long, device=device)
            ).to(language_dtype).squeeze(0)
            token_start = view_index * tokens_per_image
            token_end = token_start + tokens_per_image
            context_tokens[:, :, token_start:token_end] = (
                context_tokens[:, :, token_start:token_end]
                + context_segment_visual.view(1, 1, 1, -1)
                + camera_offset.view(1, 1, 1, -1)
                + slot_offsets.view(1, num_slots, 1, -1)
            )
        context_tokens[:, :, proprio_position] = (
            context_tokens[:, :, proprio_position]
            + context_segment_proprio.view(1, 1, -1)
            + slot_offsets.view(1, num_slots, -1)
        )
        if progress_enabled:
            context_tokens[:, :, progress_position] = (
                context_tokens[:, :, progress_position]
                + context_segment_progress.view(1, 1, -1)
                + slot_offsets.view(1, num_slots, -1)
            )

        context_attention_mask = valid.unsqueeze(-1).expand(batch_size, num_slots, block_tokens).reshape(
            batch_size,
            num_slots * block_tokens,
        )
        return context_tokens.reshape(batch_size, num_slots * block_tokens, hidden_size), context_attention_mask

    def _encode_proprio_token(
        self,
        query_proprio: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.config.proprio_dim <= 0:
            return None
        if query_proprio is None or query_proprio.shape[-1] == 0:
            raise ValueError(
                "This checkpoint was configured with proprio_dim > 0, but the batch has no query_proprio. "
                "Pass --proprio-key with the same keys used for training."
            )
        if query_proprio.shape[-1] != self.config.proprio_dim:
            raise ValueError(
                f"query_proprio has dim {query_proprio.shape[-1]}, but model expects {self.config.proprio_dim}."
            )
        proprio = query_proprio.to(device=device, dtype=torch.float32)
        return self.proprio_encoder(proprio, batch_size, device).unsqueeze(1)

    def _encode_query_history_tokens(
        self,
        query_image: torch.Tensor,
        query_proprio: torch.Tensor | None,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode chronological observation blocks, sharing vision/proprio weights across time."""

        num_timesteps = len(self.config.query_history_offsets)
        if query_image.dim() != 6 or query_image.shape[1] != num_timesteps:
            raise ValueError(
                f"Query history images must be [batch, {num_timesteps}, views, channels, height, width]"
            )
        if self.config.proprio_dim > 0 and (
            query_proprio is None
            or query_proprio.dim() != 3
            or query_proprio.shape[:2] != query_image.shape[:2]
        ):
            raise ValueError("Query history proprio must be [batch, timesteps, proprio_dim]")
        device = text_embeddings.device
        dtype = self._language_dtype()
        blocks = []
        for time_index in range(num_timesteps):
            frame_images = query_image[:, time_index]
            time_offset = self.query_history_embedding.weight[time_index].to(dtype).view(1, 1, -1)
            visual_tokens = self._add_query_visual_embeddings(
                self._encode_visual_tokens(frame_images), frame_images
            ).to(dtype)
            blocks.append(self._normalize_adapter_tokens(
                visual_tokens + time_offset, text_embeddings, text_mask
            ))
            proprio_token = self._encode_proprio_token(
                query_proprio[:, time_index] if query_proprio is not None else None,
                query_image.shape[0],
                device,
            )
            if proprio_token is not None:
                source_offset = self._source_embedding(
                    segment_id=1, modality_id=1, camera_id=2, device=device, dtype=dtype
                ).view(1, 1, -1)
                blocks.append(self._normalize_adapter_tokens(
                    proprio_token.to(dtype) + source_offset + time_offset,
                    text_embeddings,
                    text_mask,
                ))
        return torch.cat(blocks, dim=1)

    def forward(
        self,
        query_image: torch.Tensor,
        language: list[str] | torch.Tensor,
        query_proprio: torch.Tensor | None = None,
        context_images: torch.Tensor | None = None,
        context_proprio: torch.Tensor | None = None,
        context_valid_mask: torch.Tensor | None = None,
        context_progress: torch.Tensor | None = None,
        context_native_features: torch.Tensor | None = None,
        context_feature_positions: torch.Tensor | None = None,
        precomputed_context_tokens: torch.Tensor | None = None,
        precomputed_context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = next(self.language_model.parameters()).device
        text_embeddings, text_mask = self._encode_text_embeddings(language, device)
        language_dtype = self._language_dtype()
        embedding_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []
        stats_parts: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        if (precomputed_context_tokens is None) != (precomputed_context_mask is None):
            raise ValueError("Provide both precomputed_context_tokens and precomputed_context_mask, or neither")
        if self.uses_context:
            if precomputed_context_tokens is not None:
                if precomputed_context_tokens.dim() != 3 or precomputed_context_mask.dim() != 2:
                    raise ValueError("Precomputed context tokens must be [batch, tokens, hidden] with a [batch, tokens] mask")
                if precomputed_context_tokens.shape[:2] != precomputed_context_mask.shape:
                    raise ValueError("Precomputed context token and mask shapes do not match")
                if precomputed_context_tokens.shape[0] != query_image.shape[0]:
                    raise ValueError("Precomputed context batch size does not match query_image")
                context_tokens = precomputed_context_tokens.to(device=device, dtype=language_dtype)
                context_mask = precomputed_context_mask.to(device=device, dtype=text_mask.dtype)
            else:
                context_tokens, context_mask = self._encode_context_tokens(
                    context_images,
                    context_proprio,
                    context_valid_mask,
                    context_progress,
                    context_native_features,
                    context_feature_positions,
                )
            context_tokens = self._normalize_adapter_tokens(context_tokens, text_embeddings, text_mask)
            embedding_parts.append(context_tokens)
            mask_parts.append(context_mask)
            stats_parts["context_tokens"] = (context_tokens, context_mask)
        elif precomputed_context_tokens is not None:
            raise ValueError("Precomputed context tokens require a context VFE backbone")

        has_query_history = len(getattr(self.config, "query_history_offsets", [0])) > 1
        if has_query_history:
            visual_tokens = self._encode_query_history_tokens(
                query_image, query_proprio, text_embeddings, text_mask
            )
            proprio_token = None  # Already interleaved with each historical visual block.
        else:
            visual_tokens = self._add_query_visual_embeddings(self._encode_visual_tokens(query_image), query_image)
            visual_tokens = self._normalize_adapter_tokens(visual_tokens, text_embeddings, text_mask)
            proprio_token = self._encode_proprio_token(query_proprio, query_image.shape[0], device)
        visual_mask = torch.ones(visual_tokens.shape[:2], dtype=text_mask.dtype, device=device)
        value_token = self.value_query.expand(query_image.shape[0], -1, -1).to(device=device)

        embedding_parts.append(visual_tokens.to(language_dtype))
        mask_parts.append(visual_mask)
        stats_parts["query_history_tokens" if has_query_history else "query_visual_tokens"] = (visual_tokens, visual_mask)
        if proprio_token is not None:
            query_proprio_offset = self._source_embedding(
                segment_id=1,
                modality_id=1,
                camera_id=2,
                device=device,
                dtype=language_dtype,
            )
            query_proprio_tokens = proprio_token.to(language_dtype) + query_proprio_offset.view(1, 1, -1)
            query_proprio_tokens = self._normalize_adapter_tokens(query_proprio_tokens, text_embeddings, text_mask)
            query_proprio_mask = torch.ones((query_image.shape[0], 1), dtype=text_mask.dtype, device=device)
            embedding_parts.append(query_proprio_tokens)
            mask_parts.append(query_proprio_mask)
            stats_parts["query_proprio_tokens"] = (query_proprio_tokens, query_proprio_mask)
        language_offset = self._source_embedding(
            segment_id=2,
            modality_id=2,
            camera_id=2,
            device=device,
            dtype=language_dtype,
        )
        value_offset = self._source_embedding(
            segment_id=3,
            modality_id=3,
            camera_id=2,
            device=device,
            dtype=language_dtype,
        )
        language_tokens = text_embeddings.to(language_dtype) + language_offset.view(1, 1, -1)
        value_tokens = value_token.to(language_dtype) + value_offset.view(1, 1, -1)
        value_tokens = self._normalize_adapter_tokens(value_tokens, text_embeddings, text_mask)
        embedding_parts.extend(
            [
                language_tokens,
                value_tokens,
            ]
        )
        value_mask = torch.ones((query_image.shape[0], 1), dtype=text_mask.dtype, device=device)
        mask_parts.extend([text_mask, value_mask])
        stats_parts["language_tokens"] = (language_tokens, text_mask)
        stats_parts["value_tokens"] = (value_tokens, value_mask)
        input_embeddings = torch.cat(embedding_parts, dim=1)
        attention_mask = torch.cat(mask_parts, dim=1).to(dtype=text_mask.dtype)
        self._maybe_log_input_embedding_stats(stats_parts, input_embeddings, attention_mask)
        language_attention_mask: torch.Tensor | dict[str, Any] = attention_mask
        if self.config.attention_mode in CUSTOM_CONTEXT_ATTENTION_MODES:
            if not self.uses_context:
                raise RuntimeError(f"{self.config.attention_mode} requires context tokens")
            context_token_count = int(context_tokens.shape[1])
            independent_block_mask = build_independent_context_block_mask(
                attention_mask,
                context_token_count=context_token_count,
                context_timestep_count=int(self.config.context_max_timesteps),
            )
            layer_types = set(getattr(self.language_model.config, "layer_types", ()))
            if not layer_types:
                raise RuntimeError("Gemma language model config does not expose layer_types")
            language_attention_mask = {
                layer_type: independent_block_mask for layer_type in layer_types
            }
            if self.config.attention_mode == HYBRID_CONTEXT_BLOCK_CAUSAL_ATTENTION_MODE:
                required_layer_types = {"sliding_attention", "full_attention"}
                missing_layer_types = required_layer_types - layer_types
                if missing_layer_types:
                    raise RuntimeError(
                        "Hybrid context attention requires Gemma sliding/full layer types; "
                        f"missing {sorted(missing_layer_types)}"
                    )
                block_causal_mask = build_context_block_causal_mask(
                    attention_mask,
                    context_token_count=context_token_count,
                    context_timestep_count=int(self.config.context_max_timesteps),
                )
                # Gemma indexes this mapping by each decoder layer's native
                # layer_type. This places block-causal mixing at layers 6, 12,
                # and 18 for the 270M [5 sliding + 1 full] x3 schedule.
                language_attention_mask["full_attention"] = block_causal_mask
        try:
            outputs = self.language_model(
                inputs_embeds=input_embeddings,
                attention_mask=language_attention_mask,
                output_hidden_states=self.language_needs_hidden_states,
                return_dict=True,
                use_cache=False,
            )
        except TypeError:
            outputs = self.language_model(
                inputs_embeds=input_embeddings,
                attention_mask=language_attention_mask,
                output_hidden_states=self.language_needs_hidden_states,
                return_dict=True,
            )
        value_hidden = self._value_hidden(outputs)
        return self.value_head(value_hidden.float())

    def train(self, mode: bool = True) -> "SiglipGemmaVFE":
        super().train(mode)
        if self.config.freeze_backbone:
            if self.vision_tower is not None:
                self.vision_tower.eval()
            if not self.uses_native_gemma_vision and self.context_vision_tower is not None:
                self.context_vision_tower.eval()
            if self.native_vision_projector is not None:
                self.native_vision_projector.eval()
            if mode and self.config.gradient_checkpointing:
                self.language_model.train()
            else:
                self.language_model.eval()
        elif self.config.use_lora:
            if mode:
                self.language_model.train()
            else:
                self.language_model.eval()
            if self.vision_tower is not None:
                self.vision_tower.eval()
            connector = getattr(self.language_model, "multi_modal_projector", None)
            if connector is not None:
                connector.train(mode)
        elif self.config.freeze_native_vision:
            if mode:
                self.language_model.train()
            else:
                self.language_model.eval()
            if self.vision_tower is not None:
                self.vision_tower.eval()
        elif self.config.freeze_vision:
            if self.vision_tower is not None:
                self.vision_tower.eval()
            recap_vision_tower = getattr(self, "recap_vision_tower", None)
            if recap_vision_tower is not None:
                recap_vision_tower.eval()
            if self.native_vision_projector is not None:
                self.native_vision_projector.eval()
        if self.config.freeze_recap_detail_merger and self.visual_projector is not None:
            self.visual_projector.eval()
        return self


class ValueFunctionEstimator(nn.Module):
    """RECAP/π*0.6-style value estimator.

    The value-function contract matches π*0.6 / RECAP: logits are a categorical
    distribution over value bins and scalar values are recovered by expectation.
    The context implementation prepends a full DINOv3-encoded demonstration
    context before the SigLIP-encoded current query observation.
    """

    def __init__(self, config: VfeModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or VfeModelConfig()
        if self.config.backbone in HF_IMAGE_BACKBONES:
            self.impl = SiglipGemmaVFE(self.config)
        elif self.config.backbone == "lightweight":
            self.impl = LightweightVFE(self.config)
        else:
            raise ValueError(
                "backbone must be one of: 'siglip_gemma', 'siglip_gemma_context', "
                "'gemma3_native_vision', 'gemma3_native_vision_context', "
                "'gemma3_native_style_270m', 'gemma3_native_style_270m_context', or 'lightweight'"
            )
        self.register_buffer(
            "bin_values",
            torch.linspace(self.config.value_min, self.config.value_max, self.config.num_bins),
            persistent=False,
        )

    def forward(
        self,
        query_image: torch.Tensor | None = None,
        language: list[str] | torch.Tensor | None = None,
        query_proprio: torch.Tensor | None = None,
        context_images: torch.Tensor | None = None,
        context_proprio: torch.Tensor | None = None,
        context_valid_mask: torch.Tensor | None = None,
        context_progress: torch.Tensor | None = None,
        context_native_features: torch.Tensor | None = None,
        context_feature_positions: torch.Tensor | None = None,
        precomputed_context_tokens: torch.Tensor | None = None,
        precomputed_context_mask: torch.Tensor | None = None,
        batch: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if batch is not None:
            if self.config.backbone == "lightweight":
                return self.impl(
                    batch["query_image"],
                    batch["language"],
                    batch.get("query_proprio"),
                )
            return self.impl(
                batch["query_image"],
                batch["language"],
                batch.get("query_proprio"),
                batch.get("context_images"),
                batch.get("context_proprio"),
                batch.get("context_valid_mask"),
                batch.get("context_progress"),
                batch.get("context_native_features"),
                batch.get("context_feature_positions"),
                precomputed_context_tokens,
                precomputed_context_mask,
            )
        if query_image is None or language is None:
            raise ValueError("Provide either batch=... or query_image and language.")
        if self.config.backbone == "lightweight":
            if precomputed_context_tokens is not None or precomputed_context_mask is not None:
                raise ValueError("Precomputed context tokens are unavailable for the lightweight backbone")
            return self.impl(query_image, language, query_proprio)
        return self.impl(
            query_image,
            language,
            query_proprio,
            context_images,
            context_proprio,
            context_valid_mask,
            context_progress,
            context_native_features,
            context_feature_positions,
            precomputed_context_tokens,
            precomputed_context_mask,
        )

    @torch.no_grad()
    def encode_context_tokens_for_eval(
        self,
        context_images: torch.Tensor | None,
        context_proprio: torch.Tensor,
        context_valid_mask: torch.Tensor,
        context_progress: torch.Tensor | None = None,
        context_native_features: torch.Tensor | None = None,
        context_feature_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.backbone == "lightweight":
            raise ValueError("Context-token caching is unavailable for the lightweight backbone")
        return self.impl.encode_context_tokens_for_eval(
            context_images,
            context_proprio,
            context_valid_mask,
            context_progress,
            context_native_features,
            context_feature_positions,
        )

    def expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_values.to(logits.device)).sum(dim=-1)
