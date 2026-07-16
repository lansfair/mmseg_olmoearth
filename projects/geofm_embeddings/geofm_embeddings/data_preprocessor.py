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
