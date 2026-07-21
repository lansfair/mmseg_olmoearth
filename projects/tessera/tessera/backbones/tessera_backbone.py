from __future__ import annotations

import math
import re
from typing import Any

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader
from mmseg.registry import MODELS
from torch import Tensor


class TesseraTemporalPositionalEncoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, doy: Tensor) -> Tensor:
        position = doy.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(
                0,
                self.d_model,
                2,
                dtype=torch.float,
                device=doy.device,
            )
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(
            doy.shape[0],
            doy.shape[1],
            self.d_model,
            device=doy.device,
            dtype=position.dtype,
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class TesseraTemporalAwarePooling(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, 1)
        self.temporal_context = nn.GRU(input_dim, input_dim, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        x_context, _ = self.temporal_context(x)
        weights = torch.softmax(self.query(x_context), dim=1)
        return (weights * x).sum(dim=1)


class TesseraTransformerEncoder(nn.Module):
    def __init__(
        self,
        band_num: int,
        latent_dim: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 8,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_model = latent_dim * 4
        self.embedding = nn.Sequential(
            nn.Linear(band_num, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.temporal_encoder = TesseraTemporalPositionalEncoder(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.attn_pool = TesseraTemporalAwarePooling(d_model)

    def forward(self, x: Tensor) -> Tensor:
        bands = x[:, :, :-1]
        doy = x[:, :, -1]
        x = self.embedding(bands) + self.temporal_encoder(doy)
        x = self.transformer_encoder(x)
        return self.attn_pool(x)


@MODELS.register_module()
class TesseraBackbone(BaseModule):
    """Online dense TESSERA v1-style encoder for MMSegmentation.

    The dataloader supplies a fixed-size per-pixel temporal tensor flattened as
    ``(B, C, H, W)`` where ``C = sample_size_s2 * 11 + sample_size_s1 * 3``.
    The backbone reshapes it into S2/S1 sequences, encodes every pixel, and
    returns one dense feature map ``(B, out_channels, H, W)``.
    """

    def __init__(
        self,
        sample_size_s2: int = 40,
        sample_size_s1: int = 40,
        latent_dim: int = 128,
        out_channels: int = 128,
        fusion_method: str = "concat",
        nhead: int = 8,
        num_encoder_layers: int = 8,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        chunk_size: int = 8192,
        frozen: bool = False,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if fusion_method not in {"concat", "sum"}:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")
        self.sample_size_s2 = int(sample_size_s2)
        self.sample_size_s1 = int(sample_size_s1)
        self.latent_dim = int(latent_dim)
        self.out_channels = int(out_channels)
        self.fusion_method = fusion_method
        self.chunk_size = int(chunk_size)
        self.frozen = bool(frozen)

        self.s2_backbone = TesseraTransformerEncoder(
            band_num=10,
            latent_dim=latent_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.s1_backbone = TesseraTransformerEncoder(
            band_num=2,
            latent_dim=latent_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        reducer_in = latent_dim * 8 if fusion_method == "concat" else latent_dim * 4
        self.dim_reducer = nn.Sequential(nn.Linear(reducer_in, out_channels))

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state", "model_state_dict"):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "TESSERA init_cfg checkpoint must be a state_dict or contain "
                "'state_dict'/'model'/'model_state'/'model_state_dict'."
            )
        cleaned = {}
        for key, value in checkpoint.items():
            key = re.sub(r"^(module\.)+", "", key)
            key = re.sub(r"^(_orig_mod\.)+", "", key)
            key = re.sub(r"^(model\.)+", "", key)
            if key.startswith("projector."):
                continue
            cleaned[key] = value
        return cleaned

    def init_weights(self) -> None:
        if self.init_cfg is None:
            return
        if not isinstance(self.init_cfg, dict):
            raise TypeError("TesseraBackbone init_cfg must be a dict.")
        if self.init_cfg.get("type") != "Pretrained":
            super().init_weights()
            return
        checkpoint_path = self.init_cfg.get("checkpoint")
        if checkpoint_path is None:
            raise ValueError("TesseraBackbone init_cfg requires checkpoint.")
        checkpoint = CheckpointLoader.load_checkpoint(
            checkpoint_path,
            map_location="cpu",
            logger=None,
        )
        state_dict = self._extract_state_dict(checkpoint)
        self.load_state_dict(state_dict, strict=False)
        self._is_init = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            super().train(False)
            for param in self.parameters():
                param.requires_grad = False
        return self

    def _encode_chunk(self, s2: Tensor, s1: Tensor) -> Tensor:
        s2_repr = self.s2_backbone(s2)
        s1_repr = self.s1_backbone(s1)
        if self.fusion_method == "concat":
            fused = torch.cat([s2_repr, s1_repr], dim=-1)
        else:
            fused = s2_repr + s1_repr
        return self.dim_reducer(fused)

    def forward(self, inputs: Tensor) -> tuple[Tensor]:
        expected_channels = self.sample_size_s2 * 11 + self.sample_size_s1 * 3
        if inputs.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} temporal channels, "
                f"got {inputs.shape[1]}"
            )

        batch_size, _, height, width = inputs.shape
        x = inputs.permute(0, 2, 3, 1).reshape(-1, expected_channels)
        s2_end = self.sample_size_s2 * 11
        s2 = x[:, :s2_end].reshape(-1, self.sample_size_s2, 11)
        s1 = x[:, s2_end:].reshape(-1, self.sample_size_s1, 3)

        outputs = []
        for start in range(0, s2.shape[0], self.chunk_size):
            end = start + self.chunk_size
            outputs.append(self._encode_chunk(s2[start:end], s1[start:end]))
        dense = torch.cat(outputs, dim=0)
        dense = dense.reshape(batch_size, height, width, self.out_channels)
        return (dense.permute(0, 3, 1, 2).contiguous(),)
