from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from mmseg.registry import MODELS
from torch import Tensor

from ..structures import EmbeddingResult, ModelCapabilities
from .base import BaseGeoFMAdapter


S2_TESSERA_INDICES = (2, 0, 1, 3, 7, 4, 5, 6, 8, 9)
S2_MEAN = (
    1711.0938, 1308.8511, 1546.4543, 3010.1293, 3106.5083,
    2068.3044, 2685.0845, 2931.5889, 2514.6928, 1899.4922,
)
S2_STD = (
    1926.1026, 1862.9751, 1803.1792, 1741.7837, 1677.4543,
    1888.7862, 1736.3090, 1715.8104, 1514.5199, 1398.4779,
)
S1_MEAN = (5484.0407, 3003.7812)
S1_STD = (1871.2334, 1726.0670)


@MODELS.register_module()
class TESSERAAdapter(BaseGeoFMAdapter):
    """TESSERA S1/S2 per-pixel embedding adapter."""

    model_family = "tessera"

    def __init__(
        self,
        num_timesteps: int,
        model_variant: str = "v1",
        latent_dim: int = 128,
        out_channels: int = 128,
        temporal_pooling: str = "mean",
        use_pretrained_normalizer: bool = True,
        timestamp_month_base: int = 0,
        chunk_size: int = 8192,
        freeze: bool = True,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(model_variant=model_variant, init_cfg=init_cfg)
        if temporal_pooling not in {"mean", "max"}:
            raise ValueError("temporal_pooling must be 'mean' or 'max'.")
        if timestamp_month_base not in {0, 1}:
            raise ValueError("timestamp_month_base must be 0 or 1.")
        self.num_timesteps = int(num_timesteps)
        self.out_channels = int(out_channels)
        self.temporal_pooling = temporal_pooling
        self.use_pretrained_normalizer = use_pretrained_normalizer
        self.timestamp_month_base = timestamp_month_base
        self.freeze = freeze

        try:
            from projects.tessera.tessera.backbones.tessera_backbone import (
                TesseraBackbone,
            )
        except ImportError as exc:
            raise ImportError(
                "TESSERAAdapter requires the local projects.tessera project."
            ) from exc
        self.backbone = TesseraBackbone(
            sample_size_s2=num_timesteps,
            sample_size_s1=num_timesteps,
            latent_dim=latent_dim,
            out_channels=out_channels,
            chunk_size=chunk_size,
            frozen=freeze,
            init_cfg=init_cfg,
        )
        self.register_buffer("s2_mean", self._stats_tensor(S2_MEAN))
        self.register_buffer("s2_std", self._stats_tensor(S2_STD))
        self.register_buffer("s1_mean", self._stats_tensor(S1_MEAN))
        self.register_buffer("s1_std", self._stats_tensor(S1_STD))
        if freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @staticmethod
    def _stats_tensor(values) -> Tensor:
        return torch.tensor(values, dtype=torch.float32)[None, None, :, None, None]

    @property
    def capabilities(self) -> ModelCapabilities:
        modalities = frozenset({"sentinel1", "sentinel2_l2a"})
        return ModelCapabilities(
            supported_modalities=modalities,
            required_modalities=modalities,
            supports_global=True,
            supports_dense=True,
            supports_multitemporal=True,
            supports_multimodal=True,
            native_stride=1,
        )

    def init_weights(self) -> None:
        self.backbone.init_weights()
        self._is_init = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    @staticmethod
    def calculate_day_of_year(timestamps: Tensor, month_base: int = 0) -> Tensor:
        day = timestamps[..., 0].long()
        month = timestamps[..., 1].long() + (1 if month_base == 0 else 0)
        year = timestamps[..., 2].long()
        if ((month < 1) | (month > 12)).any():
            raise ValueError("Timestamp months must resolve to the range 1..12.")
        days_in_month = torch.tensor(
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
            device=timestamps.device,
        )
        cumulative = torch.cat(
            [
                torch.zeros(1, device=timestamps.device, dtype=torch.long),
                days_in_month.cumsum(0)[:-1],
            ]
        )
        is_leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
        return cumulative[month - 1] + day + (is_leap & (month > 2)).long()

    def _timestamps(
        self,
        inputs: Any,
        batch_metainfo: Sequence[dict[str, Any]] | None,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        value = inputs.get("timestamps") if isinstance(inputs, Mapping) else None
        if isinstance(value, Mapping):
            s2_timestamps = value.get("sentinel2_l2a")
            value = (
                s2_timestamps
                if s2_timestamps is not None
                else value.get("sentinel1")
            )
        if value is None and batch_metainfo:
            value = [metadata.get("timestamps") for metadata in batch_metainfo]
            if any(item is None for item in value):
                value = None
        if value is None:
            raise ValueError("TESSERA requires timestamps to calculate day of year.")
        tensor = torch.as_tensor(value, dtype=torch.long, device=device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).expand(batch_size, -1, -1)
        expected = (batch_size, self.num_timesteps, 3)
        if tensor.shape != expected:
            raise ValueError(
                f"TESSERA timestamps must have shape {expected}, "
                f"got {tuple(tensor.shape)}."
            )
        return tensor

    def prepare_inputs(
        self,
        inputs: Any,
        batch_metainfo: Sequence[dict[str, Any]] | None = None,
    ) -> Tensor:
        modalities = self.modality_tensors(inputs)
        s2 = modalities["sentinel2_l2a"]
        s1 = modalities["sentinel1"]
        if s2.ndim != 5 or s1.ndim != 5:
            raise ValueError("TESSERA inputs must have shape [B,T,C,H,W].")
        if s2.shape[:2] != s1.shape[:2] or s2.shape[-2:] != s1.shape[-2:]:
            raise ValueError("TESSERA S1 and S2 tensors must be temporally and spatially aligned.")
        if s2.shape[1] != self.num_timesteps or s1.shape[1] != self.num_timesteps:
            raise ValueError(f"TESSERA requires T={self.num_timesteps}.")
        if s2.shape[2] < 10 or s1.shape[2] != 2:
            raise ValueError("TESSERA requires at least 10 S2 bands and exactly 2 S1 bands.")

        s2 = s2[:, :, S2_TESSERA_INDICES]
        if self.use_pretrained_normalizer:
            s2 = (s2 - self.s2_mean) / self.s2_std
            s1 = (s1 - self.s1_mean) / self.s1_std
        timestamps = self._timestamps(
            inputs, batch_metainfo, s2.shape[0], s2.device
        )
        doy = self.calculate_day_of_year(
            timestamps, month_base=self.timestamp_month_base
        ).to(dtype=s2.dtype)
        doy = doy[:, :, None, None, None].expand(
            -1, -1, 1, s2.shape[-2], s2.shape[-1]
        )
        s2 = torch.cat([s2, doy], dim=2).flatten(1, 2)
        s1 = torch.cat([s1, doy], dim=2).flatten(1, 2)
        return torch.cat([s2, s1], dim=1)

    def extract_dense(self, prepared_inputs: Tensor) -> Tensor:
        return self.backbone(prepared_inputs)[0]

    def extract_global(self, prepared_inputs: Tensor) -> Tensor:
        dense = self.extract_dense(prepared_inputs)
        if self.temporal_pooling == "mean":
            return dense.mean(dim=(-2, -1))
        return dense.amax(dim=(-2, -1))

    def extract(self, inputs, batch_metainfo=None, mode="dense") -> EmbeddingResult:
        result = super().extract(inputs, batch_metainfo, mode)
        result.pooling = self.temporal_pooling if mode == "global" else None
        result.metadata.update(
            {
                "pixel_level": True,
                "use_pretrained_normalizer": self.use_pretrained_normalizer,
                "timestamp_month_base": self.timestamp_month_base,
            }
        )
        return result
