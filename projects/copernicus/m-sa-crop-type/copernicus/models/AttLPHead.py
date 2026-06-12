import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from einops import rearrange
from mmseg.registry import MODELS

@MODELS.register_module(force=True)
class AttnPoolLinearProbe(BaseDecodeHead):
    """Attention Pooling Linear Probe for segmentation tasks.
    """
    def __init__(self, in_channels, channels, num_classes, num_queries=1, dropout_ratio=0.1, align_corners=False, ignore_index=255):
        """Initialize the attention pooling linear probe."""
        super().__init__(in_channels=in_channels, channels=channels, num_classes=num_classes, dropout_ratio=dropout_ratio, align_corners=align_corners, ignore_index=ignore_index)
        assert in_channels % 64 == 0, "in_channels must be divisible by 64"
        self.num_heads = in_channels // 64
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, in_channels))
        self.kv= nn.Linear(in_channels, in_channels * 2)
        self.output_proj = nn.Linear(in_channels, num_classes)
        self.dropout = nn.Dropout(dropout_ratio)
        self.init_weights()
    def init_weights(self):
        super().init_weights()
        """Initialize weights for the probe."""
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        if self.kv.bias is not None:
            nn.init.constant_(self.kv.bias, 0) 
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0)
    def forward(self, inputs) -> dict:
        if isinstance(inputs, (list, tuple)):
            x = inputs[-1]
        else:
            x = inputs
        B, C, H, W = x.shape
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        queries = self.query_tokens.expand(B, -1, -1)
        kv = self.kv(x_seq)
        k, v = kv.chunk(2, dim=-1)
        q = rearrange(queries, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        if self.num_queries > 1:
            attn_output = attn_output.squeeze(1)
            attn_output = attn_output.unsqueeze(-1).unsqueeze(-1)
            attn_output = attn_output.expand(-1, -1, H, W)
        else:
            attn_output = attn_output.mean(dim=1)
            attn_output = attn_output.unsqueeze(-1).unsqueeze(-1)
            attn_output = attn_output.expand(-1, -1, H, W)

        output = self.output_proj(attn_output.permute(0, 2, 3, 1))
        output = output.permute(0, 3, 1, 2)
        return output
    # def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
    #     seg_logits = self.forward(inputs)
    #     losses = self.losses(seg_logits, gt_semantic_seg)
    #     return losses
    # def forward_test(self, inputs, img_metas, gt_semantic_seg, test_cfg):
    #     return self.forward(inputs)
