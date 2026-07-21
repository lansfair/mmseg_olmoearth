from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from mmengine.model import BaseDataPreprocessor
from mmseg.registry import MODELS


def _stack_nested(value: Any, path: str = "inputs") -> Any:
    if isinstance(value, Mapping):
        return {
            key: _stack_nested(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, torch.Tensor)
    ):
        if not value:
            return value
        if all(isinstance(item, torch.Tensor) for item in value):
            try:
                return torch.stack(list(value), dim=0)
            except RuntimeError as exc:
                raise ValueError(
                    f"Cannot batch {path}; sample tensors have different shapes."
                ) from exc
    return value


@MODELS.register_module()
class GeoFMDataPreprocessor(BaseDataPreprocessor):
    """Move nested multimodal inputs to the model device without flattening."""

    def forward(self, data: dict, training: bool = False) -> dict:
        data["inputs"] = _stack_nested(data.get("inputs"))
        return self.cast_data(data)


@MODELS.register_module()
class PotsdamGeoFMDataPreprocessor(BaseDataPreprocessor):
    """Convert packed Potsdam S2 proxies to canonical GeoFM inputs.

    The transform pipeline supplies a flattened ``[B, 12, H, W]`` tensor.
    Adapters receive a common ``[B, T, C, H, W]`` modality mapping.  TESSERA
    can additionally request a neutral synthetic Sentinel-1 pair because the
    RGB-only Potsdam dataset has no radar observation.
    """

    def __init__(
        self,
        num_s2_bands: int = 12,
        num_timesteps: int = 1,
        include_sentinel1: bool = False,
        input_representation: str = "normalized",
        timestamp: tuple[int, int, int] = (1, 0, 2025),
        s1_reflectance_fill: tuple[float, float] = (5484.0407, 3003.7812),
    ) -> None:
        super().__init__()
        if input_representation not in {"normalized", "reflectance"}:
            raise ValueError(
                "input_representation must be normalized or reflectance"
            )
        self.num_s2_bands = int(num_s2_bands)
        self.num_timesteps = int(num_timesteps)
        self.include_sentinel1 = bool(include_sentinel1)
        self.input_representation = input_representation
        self.timestamp = tuple(int(value) for value in timestamp)
        self.s1_reflectance_fill = tuple(float(value) for value in s1_reflectance_fill)

    def forward(self, data: dict, training: bool = False) -> dict:
        inputs = _stack_nested(data.get("inputs"))
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4:
            raise TypeError("Potsdam inputs must be a batched BCHW tensor")
        expected = self.num_s2_bands * self.num_timesteps
        if inputs.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} flattened S2 channels, got {inputs.shape[1]}"
            )
        batch, _, height, width = inputs.shape
        s2 = inputs.reshape(
            batch,
            self.num_s2_bands,
            self.num_timesteps,
            height,
            width,
        ).permute(0, 2, 1, 3, 4).contiguous()
        modalities: dict[str, torch.Tensor] = {"sentinel2_l2a": s2}
        if self.include_sentinel1:
            if self.input_representation == "normalized":
                fill = torch.zeros(2, dtype=s2.dtype, device=s2.device)
            else:
                fill = torch.tensor(
                    self.s1_reflectance_fill,
                    dtype=s2.dtype,
                    device=s2.device,
                )
            modalities["sentinel1"] = fill[None, None, :, None, None].expand(
                batch, self.num_timesteps, 2, height, width
            ).contiguous()
        timestamps = torch.tensor(
            self.timestamp, dtype=torch.long, device=s2.device
        )[None, None].expand(batch, self.num_timesteps, 3).contiguous()
        data["inputs"] = {"modalities": modalities, "timestamps": timestamps}
        return self.cast_data(data)
