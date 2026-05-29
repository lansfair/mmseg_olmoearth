from __future__ import annotations

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class OlmoEarthFeatureBackbone(BaseModule):
    """Backbone for precomputed dense OLMoEarth embeddings.

    This is used for reference-style linear probing: OLMoEarth embeddings are
    extracted once offline, then MMSeg trains only the probe head.
    """

    def __init__(
        self,
        out_channels: int = 768,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.out_channels = out_channels

    def forward(self, inputs: Tensor) -> tuple[Tensor]:
        if inputs.shape[1] != self.out_channels:
            raise ValueError(
                f"Expected {self.out_channels} embedding channels, "
                f"got {inputs.shape[1]}"
            )
        return (inputs,)
