"""Linear-probe segmentation head for UniverSat."""

import torch.nn as nn

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class UniverSatLinearProbeHead(BaseDecodeHead):
    """Pixel-wise linear probe head.

    Equivalent to a LayerNorm followed by a single linear classifier applied
    independently at every spatial location of the frozen backbone features.
    This matches the protocol used in the original ``src/LP_eval.py``.

    Args:
        in_channels: Number of input feature channels.
        output_size: Optional fixed output size ``(H, W)``.
        *args, **kwargs: forwarded to ``BaseDecodeHead``.
    """

    def __init__(self, in_channels: int = 768, output_size=None, *args, **kwargs):
        super().__init__(in_channels, in_channels, *args, **kwargs)
        self.output_size = output_size
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=True)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (list[Tensor]): list of feature maps from backbone.

        Returns:
            Tensor: segmentation logits of shape ``(B, num_classes, H, W)``.
        """
        x = self._transform_inputs(inputs)
        b, c, h, w = x.shape
        # LayerNorm over channel dimension.
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.cls_seg(x)
        if self.output_size is not None:
            x = nn.functional.interpolate(
                x, size=self.output_size, mode="bilinear", align_corners=False
            )
        return x
