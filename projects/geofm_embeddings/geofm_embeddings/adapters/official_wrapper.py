from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from mmseg.registry import MODELS
from torch import Tensor

from ..structures import EmbeddingResult, ModelCapabilities
from .base import BaseGeoFMAdapter


OFFICIAL_WRAPPER_PRESETS = {
    "anysat": {
        "wrapper_path": "olmoearth_pretrain.evals.models.anysat.anysat.AnySat",
        "supports_multimodal": True,
    },
    "clay": {
        "wrapper_path": "olmoearth_pretrain.evals.models.clay.clay.Clay",
        "supports_multimodal": True,
    },
    "croma": {
        "wrapper_path": "olmoearth_pretrain.evals.models.croma.croma.Croma",
        "supports_multimodal": True,
    },
    "galileo": {
        "wrapper_path": (
            "olmoearth_pretrain.evals.models.galileo.single_file_galileo."
            "GalileoWrapper"
        ),
        "supports_multimodal": True,
    },
    "panopticon": {
        "wrapper_path": (
            "olmoearth_pretrain.evals.models.panopticon.panopticon.Panopticon"
        ),
        "supports_multimodal": True,
        "dense_method": "forward_features",
        "uses_spatial_pool_argument": False,
    },
    "presto": {
        "wrapper_path": (
            "olmoearth_pretrain.evals.models.presto.presto.PrestoWrapper"
        ),
        "supports_multimodal": True,
    },
    "prithviv2": {
        "wrapper_path": (
            "olmoearth_pretrain.evals.models.prithviv2.prithviv2.PrithviV2"
        ),
        "supports_multimodal": False,
    },
    "satlas": {
        "wrapper_path": "olmoearth_pretrain.evals.models.satlas.satlas.Satlas",
        "supports_multimodal": False,
    },
    "terramind": {
        "wrapper_path": (
            "olmoearth_pretrain.evals.models.terramind.terramind.Terramind"
        ),
        "supports_multimodal": True,
    },
}


def _load_object(path: str):
    module_name, object_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), object_name)


def _olmoearth_types():
    try:
        from olmoearth_pretrain.data.constants import Modality
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
        from olmoearth_pretrain.nn.pooling import PoolingType
    except ImportError as exc:
        raise ImportError(
            "Official baseline adapters require olmoearth_pretrain."
        ) from exc
    return MaskedOlmoEarthSample, MaskValue, Modality, PoolingType


@MODELS.register_module()
class OfficialOlmoEarthWrapperAdapter(BaseGeoFMAdapter):
    """Expose an official ``olmoearth_pretrain.evals.models`` wrapper.

    This adapter is used for model families whose OlmoEarth evaluation wrapper
    already implements input normalization, token selection and temporal
    pooling. ``wrapper_path`` must point to the wrapper's ``nn.Module`` class;
    checkpoint and size arguments are passed through ``wrapper_kwargs``.
    """

    def __init__(
        self,
        model_variant: str,
        modalities: Sequence[str],
        out_channels: int,
        native_stride: int | None,
        preset: str | None = None,
        wrapper_path: str | None = None,
        model_family: str | None = None,
        wrapper_kwargs: dict[str, Any] | None = None,
        pooling_type: str = "mean",
        supports_multimodal: bool | None = None,
        supports_multitemporal: bool = True,
        dense_method: str | None = None,
        dense_layout: str = "bhwd",
        uses_spatial_pool_argument: bool | None = None,
        freeze: bool = True,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(model_variant=model_variant, init_cfg=init_cfg)
        preset_values = {}
        if preset is not None:
            if preset not in OFFICIAL_WRAPPER_PRESETS:
                raise ValueError(
                    "Unknown official wrapper preset: "
                    f"{preset}. Available: {sorted(OFFICIAL_WRAPPER_PRESETS)}"
                )
            preset_values = OFFICIAL_WRAPPER_PRESETS[preset]
        wrapper_path = wrapper_path or preset_values.get("wrapper_path")
        if wrapper_path is None:
            raise ValueError("Specify either preset or wrapper_path.")
        model_family = model_family or preset
        if model_family is None:
            raise ValueError("model_family is required without a preset.")
        if supports_multimodal is None:
            supports_multimodal = preset_values.get("supports_multimodal", True)
        if dense_method is None:
            dense_method = preset_values.get("dense_method", "forward")
        if uses_spatial_pool_argument is None:
            uses_spatial_pool_argument = preset_values.get(
                "uses_spatial_pool_argument", True
            )
        if not modalities:
            raise ValueError("At least one input modality must be configured.")
        if pooling_type not in {"mean", "max"}:
            raise ValueError("pooling_type must be 'mean' or 'max'.")
        if dense_method not in {"forward", "forward_features"}:
            raise ValueError("dense_method must be forward or forward_features.")
        if dense_layout not in {"bhwd", "bdhw"}:
            raise ValueError("dense_layout must be bhwd or bdhw.")
        self.model_family = model_family
        self.wrapper_path = wrapper_path
        self.modalities = tuple(modalities)
        self.out_channels = int(out_channels)
        self.native_stride = native_stride
        self.pooling_type = pooling_type
        self.supports_multimodal = supports_multimodal
        self.supports_multitemporal = supports_multitemporal
        self.dense_method = dense_method
        self.dense_layout = dense_layout
        self.uses_spatial_pool_argument = uses_spatial_pool_argument
        self.freeze = freeze
        wrapper_class = _load_object(wrapper_path)
        self.wrapper = wrapper_class(**(wrapper_kwargs or {}))
        if freeze:
            self.wrapper.requires_grad_(False)
            self.wrapper.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.wrapper.eval()
        return self

    @property
    def capabilities(self) -> ModelCapabilities:
        modalities = frozenset(self.modalities)
        return ModelCapabilities(
            supported_modalities=modalities,
            required_modalities=modalities,
            supports_global=True,
            supports_dense=True,
            supports_multitemporal=self.supports_multitemporal,
            supports_multimodal=self.supports_multimodal,
            native_stride=self.native_stride,
        )

    @staticmethod
    def _timestamps_from_inputs(
        inputs: Any,
        batch_metainfo: Sequence[dict[str, Any]] | None,
        batch_size: int,
        num_timesteps: int,
        device: torch.device,
    ) -> Tensor:
        value = inputs.get("timestamps") if isinstance(inputs, Mapping) else None
        if isinstance(value, Mapping):
            available = [item for item in value.values() if item is not None]
            value = available[0] if available else None
            for other in available[1:]:
                if not torch.equal(torch.as_tensor(value), torch.as_tensor(other)):
                    raise ValueError(
                        "Official wrappers require one shared timestamp sequence."
                    )
        if value is None and batch_metainfo:
            values = [metadata.get("timestamps") for metadata in batch_metainfo]
            if all(item is not None for item in values):
                value = values
        if value is None:
            default = torch.tensor([1, 0, 2025], device=device)
            return default[None, None].repeat(batch_size, num_timesteps, 1).long()
        tensor = torch.as_tensor(value, dtype=torch.long, device=device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).expand(batch_size, -1, -1)
        expected = (batch_size, num_timesteps, 3)
        if tensor.shape != expected:
            raise ValueError(
                f"timestamps must have shape {expected}, got {tuple(tensor.shape)}."
            )
        return tensor

    @staticmethod
    def _to_native(value: Tensor, modality: str, num_bands: int) -> Tensor:
        if value.ndim != 5:
            raise ValueError(f"{modality} must have shape [B,T,C,H,W].")
        if value.shape[2] != num_bands:
            raise ValueError(
                f"{modality} expected C={num_bands}, got C={value.shape[2]}."
            )
        return value.permute(0, 3, 4, 1, 2).contiguous()

    def prepare_inputs(
        self,
        inputs: Any,
        batch_metainfo: Sequence[dict[str, Any]] | None = None,
    ):
        MaskedSample, MaskValue, Modality, _ = _olmoearth_types()
        modality_tensors = self.modality_tensors(inputs)
        provided_masks = inputs.get("masks") if isinstance(inputs, Mapping) else None
        kwargs = {}
        batch_size = None
        device = None
        dynamic_timesteps = set()
        for modality in self.modalities:
            spec = Modality.get(modality)
            native = self._to_native(
                modality_tensors[modality], modality, spec.num_bands
            )
            kwargs[modality] = native
            if batch_size is None:
                batch_size = native.shape[0]
                device = native.device
            elif native.shape[0] != batch_size:
                raise ValueError("All modalities must have the same batch size.")
            if spec.is_multitemporal:
                dynamic_timesteps.add(native.shape[-2])
            mask = None
            if isinstance(provided_masks, Mapping):
                mask = provided_masks.get(modality)
            if mask is None:
                mask = torch.full(
                    (*native.shape[:-1], spec.num_band_sets),
                    MaskValue.ONLINE_ENCODER.value,
                    dtype=torch.float32,
                    device=native.device,
                )
            kwargs[f"{modality}_mask"] = mask
        if len(dynamic_timesteps) > 1:
            raise ValueError("Multitemporal modalities must share timestep count.")
        assert batch_size is not None and device is not None
        num_timesteps = next(iter(dynamic_timesteps), 1)
        kwargs["timestamps"] = self._timestamps_from_inputs(
            inputs,
            batch_metainfo,
            batch_size,
            num_timesteps,
            device,
        )
        return MaskedSample(**kwargs)

    def extract_global(self, prepared_inputs) -> Tensor:
        _, _, _, PoolingType = _olmoearth_types()
        kwargs = {"pooling": PoolingType(self.pooling_type)}
        if self.uses_spatial_pool_argument:
            kwargs["spatial_pool"] = False
        return self.wrapper(prepared_inputs, **kwargs)

    def extract_dense(self, prepared_inputs) -> Tensor:
        _, _, _, PoolingType = _olmoearth_types()
        pooling = PoolingType(self.pooling_type)
        if self.dense_method == "forward_features":
            output = self.wrapper.forward_features(
                prepared_inputs,
                pooling=pooling,
            )
        else:
            output = self.wrapper(
                prepared_inputs,
                pooling=pooling,
                spatial_pool=True,
            )
        if output.ndim != 4:
            raise ValueError(
                "Official dense wrapper must return a four-dimensional tensor, "
                f"got {tuple(output.shape)}."
            )
        if self.dense_layout == "bhwd":
            output = output.permute(0, 3, 1, 2).contiguous()
        return output

    def extract(self, inputs, batch_metainfo=None, mode="dense") -> EmbeddingResult:
        result = super().extract(inputs, batch_metainfo, mode)
        result.pooling = self.pooling_type
        result.metadata.update(
            {
                "official_wrapper": self.wrapper_path,
                "dense_method": self.dense_method,
            }
        )
        return result
