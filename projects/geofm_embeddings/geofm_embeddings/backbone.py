from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch import Tensor

from .adapters import BaseGeoFMAdapter
from .structures import EmbeddingMode, EmbeddingResult


@MODELS.register_module()
class GeoFMBackbone(BaseModule):
    """MMSeg backbone exposing a configured GeoFM embedding adapter."""

    def __init__(
        self,
        adapter: dict | BaseGeoFMAdapter,
        output_mode: EmbeddingMode = "dense",
        frozen: bool | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.adapter = (
            MODELS.build(adapter) if isinstance(adapter, dict) else adapter
        )
        if not isinstance(self.adapter, BaseGeoFMAdapter):
            raise TypeError("adapter must be a BaseGeoFMAdapter instance.")
        if output_mode not in {"global", "dense"}:
            raise ValueError(f"Unsupported output_mode: {output_mode}")
        self.output_mode = output_mode
        self.out_channels = getattr(self.adapter, "out_channels", None)
        self._batch_metainfo: Sequence[dict[str, Any]] | None = None
        self.frozen = (
            bool(getattr(self.adapter, "freeze", False))
            if frozen is None
            else frozen
        )
        self.set_frozen(self.frozen)

    @property
    def capabilities(self):
        return self.adapter.capabilities

    def set_batch_metainfo(
        self,
        batch_metainfo: Sequence[dict[str, Any]] | None,
    ) -> None:
        self._batch_metainfo = batch_metainfo

    def set_frozen(self, frozen: bool) -> None:
        """Apply one trainability policy to every model adapter."""
        self.frozen = frozen
        self.adapter.set_frozen(frozen)
        if not frozen:
            self.adapter.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.adapter.eval()
        return self

    def extract(self, inputs: Any) -> EmbeddingResult:
        return self.adapter.extract(
            inputs,
            batch_metainfo=self._batch_metainfo,
            mode=self.output_mode,
        )

    def forward(self, inputs: Any) -> tuple[Tensor]:
        return (self.extract(inputs).tensor,)
