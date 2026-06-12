from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module(force=True)
class LPHead(BaseDecodeHead):
    def forward(self, x): return self.cls_seg(x[0])