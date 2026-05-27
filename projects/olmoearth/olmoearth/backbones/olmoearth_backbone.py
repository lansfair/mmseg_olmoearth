from __future__ import annotations

from typing import Any

import torch
from mmseg.registry import MODELS
from torch import Tensor, nn

from ..utils import get_modality_bands, get_sample_field, load_olmoearth_model


def _import_olmoearth_types():
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

    return MaskedOlmoEarthSample, MaskValue, Modality


def _normalize_band_name(value: str) -> str:
    return str(value).strip().upper().replace("_", "").replace(" ", "")


@MODELS.register_module()
class OlmoEarthBackbone(nn.Module):
    """Dense OLMoEarth encoder backbone for MMSegmentation.

    The input tensor is flattened band-major as ``(B, C*T, H, W)``. Temporal
    metadata is supplied by ``OlmoEarthEncoderDecoder`` through
    ``set_batch_metainfo``.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        model_id: str | None = None,
        config_path: str | None = None,
        modality: str = "sentinel2_l2a",
        patch_size: int = 4,
        num_timesteps: int = 12,
        out_channels: int = 768,
        pooling_type: str = "mean",
        fast_pass: bool | None = None,
    ) -> None:
        super().__init__()
        self.modality = modality
        self.patch_size = patch_size
        self.num_timesteps = num_timesteps
        self.out_channels = out_channels
        self.pooling_type = pooling_type
        self.fast_pass = fast_pass
        self.band_names = list(get_modality_bands(modality))
        self.sample_field = get_sample_field(modality)
        self.model = load_olmoearth_model(
            checkpoint_path=checkpoint_path,
            model_id=model_id,
            config_path=config_path,
        )
        self.encoder = self.model.encoder
        self._batch_metainfo: list[dict[str, Any]] | None = None

    def set_batch_metainfo(
        self,
        batch_metainfo: list[dict[str, Any]] | None,
    ) -> None:
        self._batch_metainfo = batch_metainfo

    def _get_modality_enum(self):
        _, _, Modality = _import_olmoearth_types()
        return getattr(Modality, self.modality.upper(), None) or getattr(
            Modality, self.sample_field.upper()
        )

    def _get_bandsets(self) -> list[list[str]]:
        modality = self._get_modality_enum()
        for attr in ("band_sets", "bandsets", "band_groups"):
            if not hasattr(modality, attr):
                continue
            value = getattr(modality, attr)
            if value is None:
                continue
            resolved = []
            for group in value:
                if hasattr(group, "bands"):
                    resolved.append(
                        [_normalize_band_name(x) for x in group.bands]
                    )
                else:
                    resolved.append([_normalize_band_name(x) for x in group])
            return resolved
        return [[_normalize_band_name(band)] for band in self.band_names]

    def _default_timestamps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        timestamps = torch.tensor(
            [1, 1, 2025],
            dtype=torch.long,
            device=device,
        )
        return timestamps[None, None, :].repeat(
            batch_size,
            self.num_timesteps,
            1,
        )

    def _timestamps_from_metainfo(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if not self._batch_metainfo:
            return self._default_timestamps(batch_size, device)
        timestamps = []
        for meta in self._batch_metainfo:
            value = meta.get("timestamps")
            if value is None:
                timestamps.append(
                    self._default_timestamps(1, device).squeeze(0)
                )
                continue
            tensor = torch.as_tensor(value, dtype=torch.long, device=device)
            if tensor.ndim != 2 or tensor.shape[-1] != 3:
                raise ValueError(
                    "timestamps must have shape (T, 3), "
                    f"got {tuple(tensor.shape)}"
                )
            timestamps.append(tensor)
        return torch.stack(timestamps, dim=0)

    def _present_bands_from_metainfo(self, batch_size: int) -> list[set[str]]:
        if not self._batch_metainfo:
            all_bands = {
                _normalize_band_name(band)
                for band in self.band_names
            }
            return [all_bands for _ in range(batch_size)]
        out = []
        for meta in self._batch_metainfo:
            present = meta.get("present_bands") or self.band_names
            out.append({_normalize_band_name(band) for band in present})
        return out

    def _build_bandset_mask(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        _, MaskValue, _ = _import_olmoearth_types()
        bandsets = self._get_bandsets()
        present_by_sample = self._present_bands_from_metainfo(batch_size)
        mask = torch.full(
            (batch_size, height, width, self.num_timesteps, len(bandsets)),
            float(MaskValue.MISSING.value),
            dtype=torch.float32,
            device=device,
        )
        for sample_idx, present in enumerate(present_by_sample):
            for bandset_idx, bandset in enumerate(bandsets):
                if any(band in present for band in bandset):
                    mask[sample_idx, :, :, :, bandset_idx] = float(
                        MaskValue.ONLINE_ENCODER.value
                    )
        return mask

    def _has_missing_tokens(self, sample) -> bool:
        _, MaskValue, _ = _import_olmoearth_types()
        for name, value in sample.as_dict().items():
            if name.endswith("_mask") and value is not None:
                if (value == MaskValue.MISSING.value).any():
                    return True
        return False

    def _make_sample(self, inputs: Tensor):
        MaskedOlmoEarthSample, _, _ = _import_olmoearth_types()
        batch_size, channels, height, width = inputs.shape
        num_bands = len(self.band_names)
        expected_channels = num_bands * self.num_timesteps
        if channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels "
                f"({num_bands} bands x {self.num_timesteps} timesteps), "
                f"got {channels}"
            )
        image = inputs.reshape(
            batch_size,
            num_bands,
            self.num_timesteps,
            height,
            width,
        )
        image = image.permute(0, 3, 4, 2, 1).contiguous()
        bandset_mask = self._build_bandset_mask(
            batch_size, height, width, inputs.device
        )
        kwargs = {
            self.sample_field: image,
            f"{self.sample_field}_mask": bandset_mask,
            "timestamps": self._timestamps_from_metainfo(
                batch_size,
                inputs.device,
            ),
        }
        return MaskedOlmoEarthSample(**kwargs)

    def forward(self, inputs: Tensor) -> tuple[Tensor]:
        from olmoearth_pretrain.nn.pooling import (
            PoolingType,
            pool_unmasked_tokens,
        )

        sample = self._make_sample(inputs)
        fast_pass = self.fast_pass
        if fast_pass is None:
            fast_pass = not self._has_missing_tokens(sample)
        encoder_out = self.model(
            sample,
            fast_pass=fast_pass,
            patch_size=self.patch_size,
        )
        tokens_and_masks = encoder_out["tokens_and_masks"]
        pooled = pool_unmasked_tokens(
            tokens_and_masks,
            PoolingType(self.pooling_type),
            spatial_pooling=True,
            concat_features=False,
        )
        return (pooled.permute(0, 3, 1, 2).contiguous(),)
