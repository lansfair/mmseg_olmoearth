"""Data preprocessor for multimodal UniverSat inputs."""

from typing import Sequence

import torch

from mmengine.model import BaseDataPreprocessor
from mmseg.registry import MODELS


@MODELS.register_module()
class UniverSatDataPreprocessor(BaseDataPreprocessor):
    """Preprocessor that passes through a dict of modality tensors.

    Unlike the standard ``SegDataPreProcessor``, which expects a single image
    tensor, UniverSat receives a dict ``{modality: tensor}``. This preprocessor
    only casts tensors to the model's device/dtype and leaves the dict
    structure untouched.
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        """Forward function.

        Args:
            data: dict with ``inputs`` (dict of tensors) and optionally
                ``data_samples``.
            training: Whether in training mode.

        Returns:
            dict: The preprocessed data.

        Note:
            ``BaseDataPreprocessor.cast_data`` only moves tensors to the target
            device and preserves their dtypes. This keeps date tensors as
            ``long`` while floating-point modality tensors stay in the model's
            dtype.
        """
        inputs = data.get("inputs")
        if isinstance(inputs, dict):
            stacked = {}
            for key, value in inputs.items():
                if isinstance(value, Sequence) and not isinstance(value, torch.Tensor):
                    try:
                        value = torch.stack(list(value), dim=0)
                    except RuntimeError as exc:
                        raise ValueError(
                            f"Cannot batch UniverSat input {key!r}; all samples "
                            "must have the same shape. Use a project-specific "
                            "collate function for variable-length time series."
                        ) from exc
                stacked[key] = value
            data["inputs"] = stacked
        return self.cast_data(data)
