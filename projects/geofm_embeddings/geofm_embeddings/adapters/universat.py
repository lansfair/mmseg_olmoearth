from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import torch
from mmseg.registry import MODELS
from torch import Tensor, nn

from ..structures import EmbeddingResult, ModelCapabilities
from .base import BaseGeoFMAdapter


# Canonical OlmoEarth Sentinel-2 order starts with
# B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12. UniverSat uses
# B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12.
S2_OLMOEARTH_TO_UNIVERSAT = (0, 1, 2, 4, 5, 6, 3, 7, 8, 9)
UNIVERSAT_CHANNELS = {
    "sentinel2_l2a": 10,
    "sentinel1": 3,
}
UNIVERSAT_NAMES = {
    "sentinel2_l2a": "s2",
    "sentinel1": "s1",
}


def _timestamps_to_reference_days(
    timestamps: Tensor,
    *,
    reference_date: date = date(2018, 1, 1),
) -> Tensor:
    """Convert canonical [day, zero-based month, year] timestamps to days."""

    if timestamps.ndim != 3 or timestamps.shape[-1] != 3:
        raise ValueError(
            "UniverSat timestamps must have shape [B,T,3], got "
            f"{tuple(timestamps.shape)}."
        )
    flat = timestamps.detach().cpu().reshape(-1, 3).tolist()
    values: list[int] = []
    for day_value, month_zero, year_value in flat:
        try:
            current = date(
                int(year_value), int(month_zero) + 1, int(day_value)
            )
        except ValueError as error:
            raise ValueError(
                "Invalid UniverSat timestamp "
                f"[day={day_value}, month0={month_zero}, year={year_value}]"
            ) from error
        values.append((current - reference_date).days)
    return torch.tensor(
        values, dtype=torch.long, device=timestamps.device
    ).reshape(timestamps.shape[:2])


@MODELS.register_module()
class UniverSatAdapter(BaseGeoFMAdapter):
    """Official UniverSat dense-feature adapter.

    The official source tree is loaded locally and the released Hugging Face
    checkpoint may also be a local directory. Network access is therefore not
    required during extraction once both assets have been fetched.
    """

    model_family = "universat"

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        pretrained_model_dir: str | Path | None = None,
        repo_id: str = "g-astruc/UniverSat",
        modalities: Sequence[str] = ("sentinel2_l2a",),
        model_variant: str = "base",
        patch_size: float = 40.0,
        output_grid: int = 64,
        out_channels: int = 768,
        s2_input_order: str = "olmoearth",
        global_pooling: str = "mean",
        freeze: bool = True,
        model: nn.Module | None = None,
        model_cfg: dict[str, Any] | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(model_variant=model_variant, init_cfg=init_cfg)
        modalities = tuple(modalities)
        if not modalities or not set(modalities).issubset(UNIVERSAT_NAMES):
            raise ValueError(
                "UniverSat supports sentinel2_l2a and sentinel1 modalities."
            )
        if s2_input_order not in {"olmoearth", "universat"}:
            raise ValueError(
                "s2_input_order must be 'olmoearth' or 'universat'."
            )
        if global_pooling not in {"mean", "max"}:
            raise ValueError("global_pooling must be 'mean' or 'max'.")
        if patch_size <= 0 or output_grid < 1:
            raise ValueError("patch_size and output_grid must be positive.")

        self.modalities = modalities
        self.patch_size = float(patch_size)
        self.output_grid = int(output_grid)
        self.out_channels = int(out_channels)
        self.s2_input_order = s2_input_order
        self.global_pooling = global_pooling
        self.freeze = freeze
        self.repo_id = repo_id
        self.repo_dir = None if repo_dir is None else str(Path(repo_dir))
        self.pretrained_model_dir = (
            None
            if pretrained_model_dir is None
            else str(Path(pretrained_model_dir))
        )

        self.uses_native_backbone = False
        if model is None and model_cfg is not None:
            model = MODELS.build(model_cfg)
            if hasattr(model, "init_weights"):
                model.init_weights()
            self.uses_native_backbone = True
        if model is None:
            if self.repo_dir is None:
                raise ValueError(
                    "Specify model_cfg, inject model, or provide repo_dir."
                )
            hubconf = Path(self.repo_dir) / "hubconf.py"
            if not hubconf.is_file():
                raise FileNotFoundError(
                    f"UniverSat hubconf.py was not found under {self.repo_dir}."
                )
            checkpoint = self.pretrained_model_dir or self.repo_id
            model = torch.hub.load(
                self.repo_dir,
                "from_pretrained",
                source="local",
                repo_id=checkpoint,
            )
        self.model = model
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    @property
    def capabilities(self) -> ModelCapabilities:
        modalities = frozenset(self.modalities)
        return ModelCapabilities(
            supported_modalities=modalities,
            required_modalities=modalities,
            supports_global=True,
            supports_dense=True,
            supports_multitemporal=True,
            supports_multimodal=True,
            native_stride=None,
        )

    def init_weights(self) -> None:
        # The official hub loader has already loaded and audited the checkpoint.
        self._is_init = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    @staticmethod
    def _timestamp_value(
        inputs: Any,
        modality: str,
        batch_metainfo: Sequence[dict[str, Any]] | None,
    ) -> Any:
        value = inputs.get("timestamps") if isinstance(inputs, Mapping) else None
        if isinstance(value, Mapping):
            value = value.get(modality)
        if value is None and batch_metainfo:
            rows = [item.get("timestamps") for item in batch_metainfo]
            if all(item is not None for item in rows):
                value = rows
        return value

    def _prepare_modality(
        self,
        value: Tensor,
        modality: str,
    ) -> Tensor:
        if value.ndim != 5:
            raise ValueError(f"{modality} must have shape [B,T,C,H,W].")
        if modality == "sentinel2_l2a":
            if value.shape[2] < 10:
                raise ValueError(
                    "sentinel2_l2a requires at least ten spectral bands."
                )
            if self.s2_input_order == "olmoearth":
                value = value[:, :, S2_OLMOEARTH_TO_UNIVERSAT]
            else:
                value = value[:, :, :10]
        elif value.shape[2] != UNIVERSAT_CHANNELS[modality]:
            raise ValueError(
                f"{modality} expected C={UNIVERSAT_CHANNELS[modality]}, "
                f"got C={value.shape[2]}."
            )
        return value

    def prepare_inputs(
        self,
        inputs: Any,
        batch_metainfo: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Tensor]:
        tensors = self.modality_tensors(inputs)
        prepared: dict[str, Tensor] = {}
        batch_size: int | None = None
        spatial_size: tuple[int, int] | None = None
        for modality in self.modalities:
            value = self._prepare_modality(tensors[modality], modality)
            if batch_size is None:
                batch_size = value.shape[0]
                spatial_size = tuple(value.shape[-2:])
            elif value.shape[0] != batch_size or tuple(value.shape[-2:]) != spatial_size:
                raise ValueError(
                    "UniverSat modalities must share batch and spatial dimensions."
                )
            native_name = UNIVERSAT_NAMES[modality]
            prepared[native_name] = value
            timestamp_value = self._timestamp_value(
                inputs, modality, batch_metainfo
            )
            if timestamp_value is None:
                raise ValueError(
                    f"UniverSat requires timestamps for {modality}."
                )
            timestamps = torch.as_tensor(
                timestamp_value, dtype=torch.long, device=value.device
            )
            if timestamps.ndim == 2:
                timestamps = timestamps.unsqueeze(0).expand(
                    value.shape[0], -1, -1
                )
            expected = (value.shape[0], value.shape[1], 3)
            if tuple(timestamps.shape) != expected:
                raise ValueError(
                    f"{modality} timestamps must have shape {expected}, "
                    f"got {tuple(timestamps.shape)}."
                )
            prepared[f"{native_name}_dates"] = _timestamps_to_reference_days(
                timestamps
            )
        return prepared

    def extract_dense(self, prepared_inputs: dict[str, Tensor]) -> Tensor:
        if self.uses_native_backbone:
            output = self.model(prepared_inputs)
            if not isinstance(output, (list, tuple)) or len(output) != 1:
                raise ValueError(
                    "Native UniverSat backbone must return one feature map."
                )
            dense = output[0]
            if dense.ndim != 4 or dense.shape[1] != self.out_channels:
                raise ValueError(
                    "Native UniverSat output must be [B,D,H,W], got "
                    f"{tuple(dense.shape)}."
                )
            return dense
        tokens, _ = self.model.encode(
            prepared_inputs,
            patch_size=self.patch_size,
            output_grid=self.output_grid,
        )
        if tokens.ndim != 3 or tokens.shape[-1] != self.out_channels:
            raise ValueError(
                "UniverSat encode() must return [B,G*G,D], got "
                f"{tuple(tokens.shape)}."
            )
        expected_tokens = self.output_grid * self.output_grid
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} UniverSat spatial tokens, "
                f"got {tokens.shape[1]}."
            )
        return tokens.reshape(
            tokens.shape[0], self.output_grid, self.output_grid, self.out_channels
        ).permute(0, 3, 1, 2).contiguous()

    def extract_global(self, prepared_inputs: dict[str, Tensor]) -> Tensor:
        dense = self.extract_dense(prepared_inputs)
        if self.global_pooling == "mean":
            return dense.mean(dim=(-2, -1))
        return dense.amax(dim=(-2, -1))

    def extract(self, inputs, batch_metainfo=None, mode="dense") -> EmbeddingResult:
        result = super().extract(inputs, batch_metainfo, mode)
        result.pooling = self.global_pooling if mode == "global" else None
        result.metadata.update(
            {
                "official_repository": "https://github.com/gastruc/UniverSat",
                "checkpoint_repository": self.repo_id,
                "patch_size_metres": self.patch_size,
                "output_grid": self.output_grid,
                "s2_input_order": self.s2_input_order,
            }
        )
        return result
