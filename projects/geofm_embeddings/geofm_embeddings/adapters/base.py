from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from mmengine.model import BaseModule
from torch import Tensor

from ..structures import EmbeddingMode, EmbeddingResult, ModelCapabilities


class BaseGeoFMAdapter(BaseModule, ABC):
    """Base class for model-specific embedding extraction adapters."""

    model_family = "unknown"

    def __init__(
        self,
        model_variant: str,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.model_variant = model_variant
        self._geofm_frozen = False

    def set_frozen(self, frozen: bool) -> None:
        """Synchronize common model-specific freeze switches."""
        self._geofm_frozen = frozen
        for module in self.modules():
            for attribute in ("freeze", "frozen"):
                value = getattr(module, attribute, None)
                if isinstance(value, bool):
                    setattr(module, attribute, frozen)
            frozen_exclude = getattr(module, "frozen_exclude", None)
            if isinstance(frozen_exclude, (list, tuple, set)):
                module.frozen_exclude = [] if frozen else ["all"]
        self.requires_grad_(not frozen)
        if frozen:
            self.eval()
        else:
            self.train()

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Return the capabilities of this configured adapter."""

    @staticmethod
    def modality_tensors(inputs: Any) -> Mapping[str, Tensor]:
        """Return modality tensors from canonical or flat dictionary input."""
        if not isinstance(inputs, Mapping):
            raise TypeError(
                "GeoFM adapters expect a mapping with a 'modalities' mapping."
            )
        modalities = inputs.get("modalities", inputs)
        if not isinstance(modalities, Mapping):
            raise TypeError("inputs['modalities'] must be a mapping of tensors.")
        ignored = {"timestamps", "masks", "metadata"}
        return {
            str(name): value
            for name, value in modalities.items()
            if name not in ignored and isinstance(value, Tensor)
        }

    def validate_inputs(self, inputs: Any, mode: EmbeddingMode) -> tuple[str, ...]:
        modalities = tuple(sorted(self.modality_tensors(inputs)))
        if not modalities:
            raise ValueError("No modality tensors were supplied.")
        self.capabilities.validate(set(modalities), mode)
        if not self.capabilities.supports_multitemporal:
            for name, value in self.modality_tensors(inputs).items():
                if value.ndim == 5 and value.shape[1] > 1:
                    raise ValueError(
                        f"{self.model_family} does not support multiple "
                        f"timesteps, but {name!r} has T={value.shape[1]}."
                    )
        return modalities

    def extract(
        self,
        inputs: Any,
        batch_metainfo: Sequence[dict[str, Any]] | None = None,
        mode: EmbeddingMode = "dense",
    ) -> EmbeddingResult:
        modalities = self.validate_inputs(inputs, mode)
        prepared = self.prepare_inputs(inputs, batch_metainfo)
        if mode == "global":
            tensor = self.extract_global(prepared)
        else:
            tensor = self.extract_dense(prepared)
        return EmbeddingResult(
            tensor=tensor,
            mode=mode,
            model_family=self.model_family,
            model_variant=self.model_variant,
            modalities=modalities,
            native_stride=self.capabilities.native_stride,
        )

    @abstractmethod
    def prepare_inputs(
        self,
        inputs: Any,
        batch_metainfo: Sequence[dict[str, Any]] | None = None,
    ) -> Any:
        """Convert canonical inputs to the underlying model's native format."""

    @abstractmethod
    def extract_global(self, prepared_inputs: Any) -> Tensor:
        """Return one vector per sample with shape [B, D]."""

    @abstractmethod
    def extract_dense(self, prepared_inputs: Any) -> Tensor:
        """Return a feature map with shape [B, D, Hf, Wf]."""
