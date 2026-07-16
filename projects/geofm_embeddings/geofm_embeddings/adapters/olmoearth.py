from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from mmengine.runner.checkpoint import CheckpointLoader
from mmseg.registry import MODELS
from torch import Tensor

from ..structures import EmbeddingMode, EmbeddingResult, ModelCapabilities
from .base import BaseGeoFMAdapter


def _import_olmoearth():
    try:
        from olmoearth_pretrain.data.constants import Modality
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
        from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens
    except ImportError as exc:
        raise ImportError(
            "OlmoEarthAdapter requires the local olmoearth_pretrain package."
        ) from exc
    return (
        MaskedOlmoEarthSample,
        MaskValue,
        Modality,
        PoolingType,
        pool_unmasked_tokens,
    )


def _build_olmoearth_model(model_config_path: str):
    try:
        from projects.olmoearth.olmoearth.utils import build_olmoearth_model
    except ImportError as exc:
        raise ImportError(
            "OlmoEarthAdapter requires projects.olmoearth.olmoearth."
        ) from exc
    return build_olmoearth_model(model_config_path)


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "OlmoEarth checkpoint must be a state dict or contain "
            "'state_dict'/'model'."
        )
    cleaned = {}
    for key, value in checkpoint.items():
        key = re.sub(r"^(module\.)+", "", key)
        key = re.sub(r"^(model\.)+", "", key)
        cleaned[key] = value
    return cleaned


@MODELS.register_module()
class OlmoEarthAdapter(BaseGeoFMAdapter):
    """Extract official global or dense OlmoEarth encoder embeddings.

    Canonical modality tensors use ``[B, T, C, H, W]``. A legacy flattened
    tensor ``[B, C*T, H, W]`` is accepted when exactly one modality is
    configured. The adapter converts both forms to OlmoEarth's native
    ``[B, H, W, T, C]`` layout before constructing ``MaskedOlmoEarthSample``.
    """

    model_family = "olmoearth"

    def __init__(
        self,
        model_config_path: str,
        modalities: Sequence[str] = ("sentinel2_l2a",),
        model_variant: str = "base",
        patch_size: int = 4,
        pooling_type: str = "mean",
        out_channels: int = 768,
        num_timesteps: int | Mapping[str, int] | None = None,
        fast_pass: bool | None = None,
        concat_dense_modalities: bool = False,
        freeze: bool = True,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(model_variant=model_variant, init_cfg=init_cfg)
        if not modalities:
            raise ValueError("At least one OlmoEarth modality is required.")
        self.modalities = tuple(str(name) for name in modalities)
        self.patch_size = int(patch_size)
        self.pooling_type = pooling_type
        self.out_channels = int(out_channels)
        self.num_timesteps = num_timesteps
        self.fast_pass = fast_pass
        self.concat_dense_modalities = concat_dense_modalities
        self.freeze = freeze
        self.model = _build_olmoearth_model(model_config_path)
        self.encoder = self.model.encoder
        if hasattr(self.encoder, "remove_masked_tokens"):
            self.encoder.remove_masked_tokens = self._remove_masked_tokens_sort_compat
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

        # Validate modality names while constructing the adapter, not at the
        # first expensive model forward.
        _, _, Modality, _, _ = _import_olmoearth()
        for modality in self.modalities:
            Modality.get(modality)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_modalities=frozenset(self.modalities),
            required_modalities=frozenset(self.modalities),
            supports_global=True,
            supports_dense=True,
            supports_multitemporal=True,
            supports_multimodal=True,
            native_stride=self.patch_size,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    @staticmethod
    def _remove_masked_tokens_sort_compat(
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sortable_mask = mask.to(torch.uint8) if mask.dtype == torch.bool else mask
        sorted_mask, indices = torch.sort(
            sortable_mask,
            dim=1,
            descending=True,
            stable=True,
        )
        sorted_mask = sorted_mask.bool()
        x = x.gather(1, indices[:, :, None].expand_as(x))
        x = x * sorted_mask.unsqueeze(-1)
        seq_lengths = sorted_mask.sum(-1)
        max_length = seq_lengths.max()
        return (
            x[:, :max_length],
            indices,
            sorted_mask[:, :max_length],
            seq_lengths,
            max_length,
        )

    def init_weights(self) -> None:
        if self.init_cfg is None:
            return
        if not isinstance(self.init_cfg, dict):
            raise TypeError("OlmoEarthAdapter init_cfg must be a dictionary.")
        if self.init_cfg.get("type") != "Pretrained":
            super().init_weights()
            return
        checkpoint_path = self.init_cfg.get("checkpoint")
        if checkpoint_path is None:
            raise ValueError("Pretrained init_cfg requires 'checkpoint'.")
        checkpoint = CheckpointLoader.load_checkpoint(
            checkpoint_path,
            map_location="cpu",
            logger=None,
        )
        self.model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
        self._is_init = True

    def _expected_timesteps(self, modality: str, tensor: Tensor) -> int:
        if isinstance(self.num_timesteps, Mapping):
            expected = self.num_timesteps.get(modality)
        else:
            expected = self.num_timesteps
        if expected is not None:
            return int(expected)
        if tensor.ndim == 5:
            return int(tensor.shape[1])

        _, _, Modality, _, _ = _import_olmoearth()
        spec = Modality.get(modality)
        return 12 if spec.is_multitemporal else 1

    @staticmethod
    def _coerce_modality_tensor(
        value: Tensor,
        modality: str,
        num_bands: int,
        num_timesteps: int,
    ) -> Tensor:
        if value.ndim == 5:
            if value.shape[1] != num_timesteps:
                raise ValueError(
                    f"{modality} expected T={num_timesteps}, "
                    f"got shape {tuple(value.shape)}."
                )
            if value.shape[2] != num_bands:
                raise ValueError(
                    f"{modality} expected C={num_bands}, "
                    f"got shape {tuple(value.shape)}."
                )
            return value.permute(0, 3, 4, 1, 2).contiguous()

        if value.ndim == 4:
            expected_channels = num_bands * num_timesteps
            if value.shape[1] != expected_channels:
                raise ValueError(
                    f"{modality} expected C*T={expected_channels}, "
                    f"got shape {tuple(value.shape)}."
                )
            batch, _, height, width = value.shape
            value = value.reshape(
                batch,
                num_bands,
                num_timesteps,
                height,
                width,
            )
            return value.permute(0, 3, 4, 2, 1).contiguous()

        raise ValueError(
            f"{modality} must have shape [B,T,C,H,W] or [B,C*T,H,W], "
            f"got {tuple(value.shape)}."
        )

    @staticmethod
    def _input_auxiliary(inputs: Any, key: str) -> Any:
        if isinstance(inputs, Mapping):
            return inputs.get(key)
        return None

    @staticmethod
    def _metadata_timestamps(
        batch_metainfo: Sequence[dict[str, Any]] | None,
        batch_size: int,
        num_timesteps: int,
        device: torch.device,
    ) -> Tensor:
        default = torch.tensor([1, 0, 2025], dtype=torch.long, device=device)
        if not batch_metainfo:
            return default[None, None].repeat(batch_size, num_timesteps, 1)
        timestamps = []
        for metadata in batch_metainfo:
            value = metadata.get("timestamps")
            if value is None:
                tensor = default[None].repeat(num_timesteps, 1)
            else:
                tensor = torch.as_tensor(value, dtype=torch.long, device=device)
            if tensor.shape != (num_timesteps, 3):
                raise ValueError(
                    "timestamps must have shape "
                    f"({num_timesteps}, 3), got {tuple(tensor.shape)}."
                )
            timestamps.append(tensor)
        return torch.stack(timestamps)

    def _resolve_timestamps(
        self,
        inputs: Any,
        batch_metainfo: Sequence[dict[str, Any]] | None,
        batch_size: int,
        num_timesteps: int,
        device: torch.device,
    ) -> Tensor:
        value = self._input_auxiliary(inputs, "timestamps")
        if isinstance(value, Mapping):
            values = [value[name] for name in self.modalities if name in value]
            if values:
                value = values[0]
                for other in values[1:]:
                    if not torch.equal(torch.as_tensor(value), torch.as_tensor(other)):
                        raise ValueError(
                            "OlmoEarth uses one shared timestamp sequence; "
                            "per-modality timestamps must be identical."
                        )
        if value is None:
            return self._metadata_timestamps(
                batch_metainfo,
                batch_size,
                num_timesteps,
                device,
            )
        tensor = torch.as_tensor(value, dtype=torch.long, device=device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).expand(batch_size, -1, -1)
        if tensor.shape != (batch_size, num_timesteps, 3):
            raise ValueError(
                "timestamps must have shape "
                f"[B,T,3]={batch_size, num_timesteps, 3}, "
                f"got {tuple(tensor.shape)}."
            )
        return tensor

    @staticmethod
    def _build_mask(native: Tensor, num_band_sets: int, online_value: int) -> Tensor:
        return torch.full(
            (*native.shape[:-1], num_band_sets),
            online_value,
            dtype=torch.float32,
            device=native.device,
        )

    def prepare_inputs(
        self,
        inputs: Any,
        batch_metainfo: Sequence[dict[str, Any]] | None = None,
    ):
        MaskedOlmoEarthSample, MaskValue, Modality, _, _ = _import_olmoearth()

        if isinstance(inputs, Tensor):
            if len(self.modalities) != 1:
                raise TypeError(
                    "Tensor input is supported only for a single configured modality."
                )
            modality_tensors = {self.modalities[0]: inputs}
        else:
            modality_tensors = self.modality_tensors(inputs)

        kwargs: dict[str, Tensor] = {}
        dynamic_timesteps: set[int] = set()
        batch_size = None
        device = None
        provided_masks = self._input_auxiliary(inputs, "masks")
        for modality in self.modalities:
            value = modality_tensors[modality]
            spec = Modality.get(modality)
            num_timesteps = self._expected_timesteps(modality, value)
            native = self._coerce_modality_tensor(
                value,
                modality,
                spec.num_bands,
                num_timesteps,
            )
            if spec.is_multitemporal:
                dynamic_timesteps.add(num_timesteps)
            if batch_size is None:
                batch_size = native.shape[0]
                device = native.device
            elif native.shape[0] != batch_size:
                raise ValueError("All modalities must have the same batch size.")
            kwargs[modality] = native

            mask = None
            if isinstance(provided_masks, Mapping):
                mask = provided_masks.get(modality)
            if mask is None:
                mask = self._build_mask(
                    native,
                    spec.num_band_sets,
                    MaskValue.ONLINE_ENCODER.value,
                )
            else:
                mask = torch.as_tensor(mask, device=native.device)
            kwargs[f"{modality}_mask"] = mask

        if len(dynamic_timesteps) > 1:
            raise ValueError(
                "OlmoEarth requires configured multitemporal modalities to "
                "share the same number of timesteps."
            )
        assert batch_size is not None and device is not None
        timestamp_count = next(iter(dynamic_timesteps), 1)
        kwargs["timestamps"] = self._resolve_timestamps(
            inputs,
            batch_metainfo,
            batch_size,
            timestamp_count,
            device,
        )
        return MaskedOlmoEarthSample(**kwargs)

    @staticmethod
    def _has_missing_tokens(sample) -> bool:
        _, MaskValue, _, _, _ = _import_olmoearth()
        for name, value in sample.as_dict().items():
            if name.endswith("_mask") and value is not None:
                if (value == MaskValue.MISSING.value).any():
                    return True
        return False

    def _tokens_and_masks(self, prepared_inputs):
        fast_pass = self.fast_pass
        if fast_pass is None:
            fast_pass = not self._has_missing_tokens(prepared_inputs)
        return self.encoder(
            prepared_inputs,
            patch_size=self.patch_size,
            fast_pass=fast_pass,
        )["tokens_and_masks"]

    def extract_global(self, prepared_inputs) -> Tensor:
        _, _, _, PoolingType, pool_unmasked_tokens = _import_olmoearth()
        return pool_unmasked_tokens(
            self._tokens_and_masks(prepared_inputs),
            PoolingType(self.pooling_type),
            spatial_pooling=False,
            concat_features=False,
        )

    def extract_dense(self, prepared_inputs) -> Tensor:
        _, _, _, PoolingType, pool_unmasked_tokens = _import_olmoearth()
        output = pool_unmasked_tokens(
            self._tokens_and_masks(prepared_inputs),
            PoolingType(self.pooling_type),
            spatial_pooling=True,
            concat_features=self.concat_dense_modalities,
        )
        if output.ndim == 5:
            batch, height, width, modalities, dim = output.shape
            output = output.reshape(batch, height, width, modalities * dim)
        return output.permute(0, 3, 1, 2).contiguous()

    def extract(
        self,
        inputs: Any,
        batch_metainfo: Sequence[dict[str, Any]] | None = None,
        mode: EmbeddingMode = "dense",
    ) -> EmbeddingResult:
        canonical_inputs = inputs
        if isinstance(inputs, Tensor):
            canonical_inputs = {
                "modalities": {self.modalities[0]: inputs},
            }
        result = super().extract(canonical_inputs, batch_metainfo, mode)
        result.pooling = self.pooling_type
        result.metadata.update(
            {
                "patch_size": self.patch_size,
                "concat_dense_modalities": self.concat_dense_modalities,
            }
        )
        return result
