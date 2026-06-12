from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module(force=True)
class LinearProbeHead(BaseDecodeHead):
    """A strict linear-probe head: one 1x1 classifier on one feature map."""

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        if self.dropout is not None:
            x = self.dropout(x)
        return self.cls_seg(x)
