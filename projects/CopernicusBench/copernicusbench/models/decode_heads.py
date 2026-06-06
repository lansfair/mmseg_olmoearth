import torch.nn as nn

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class PatchLinearHead(BaseDecodeHead):
    """Linear patch head aligned with OlmoEarth segmentation probing.

    Each backbone spatial token predicts all pixels in its corresponding
    output patch with a single linear projection.
    """

    def __init__(self, output_patch_size=16, **kwargs):
        num_classes = kwargs.get('num_classes')
        if num_classes is None:
            raise ValueError('PatchLinearHead requires num_classes.')
        self.output_patch_size = output_patch_size
        kwargs.setdefault('out_channels', num_classes)
        super().__init__(**kwargs)
        self.conv_seg = nn.Conv2d(
            self.channels,
            self.num_classes * output_patch_size * output_patch_size,
            kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        logits = self.cls_seg(x)
        batch_size, _, height, width = logits.shape
        patch = self.output_patch_size
        expected_channels = self.num_classes * patch * patch
        if logits.shape[1] != expected_channels:
            raise ValueError(
                f'Expected {expected_channels} channels before patch '
                f'reshape, but got {logits.shape[1]}.')
        logits = logits.view(batch_size, self.num_classes, patch, patch,
                             height, width)
        logits = logits.permute(0, 1, 4, 2, 5, 3).contiguous()
        return logits.view(batch_size, self.num_classes, height * patch,
                           width * patch)
