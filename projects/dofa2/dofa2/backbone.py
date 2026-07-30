"""DOFAv2 backbone adapted to MMSegmentation."""

from __future__ import annotations

import logging
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader
from mmseg.registry import MODELS
from timm.models.vision_transformer import VisionTransformer

from .utils import get_arch_setting, get_wavelengths


def get_1d_sincos_pos_embed(embed_dim: int,
                            positions: torch.Tensor) -> torch.Tensor:
    """Create the wavelength embedding used by DOFA."""
    if embed_dim % 2:
        raise ValueError('The wavelength embedding dimension must be even.')
    omega = torch.arange(
        embed_dim // 2, dtype=torch.float32, device=positions.device)
    omega = 1.0 / 10000**(omega / (embed_dim / 2.0))
    output = torch.einsum('m,d->md', positions.reshape(-1), omega)
    return torch.cat([torch.sin(output), torch.cos(output)], dim=1)


class TransformerWeightGenerator(nn.Module):
    """Generate a convolutional kernel for each input wavelength."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 embed_dim: int,
                 num_heads: int = 4,
                 num_layers: int = 1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation='gelu',
            norm_first=False,
            batch_first=False,
            dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.num_weight_tokens = 128
        self.weight_tokens = nn.Parameter(
            torch.empty(self.num_weight_tokens, input_dim))
        self.bias_token = nn.Parameter(torch.empty(1, input_dim))
        nn.init.normal_(self.weight_tokens, std=0.02)
        nn.init.normal_(self.bias_token, std=0.02)

    def forward(
            self, wavelength_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat(
            [self.weight_tokens, wavelength_embeddings, self.bias_token],
            dim=0,
        )
        output = self.transformer_encoder(tokens)
        weights = self.fc_weight(
            output[self.num_weight_tokens:-1] + wavelength_embeddings)
        bias = self.fc_bias(output[-1])
        return weights, bias


class FCResLayer(nn.Module):
    """Small residual MLP used to refine wavelength embeddings."""

    def __init__(self, dim: int = 128):
        super().__init__()
        # Keep the upstream parameter names so official DOFA checkpoints load
        # without a key-conversion table.
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.nonlin1(self.w1(inputs))
        output = self.nonlin2(self.w2(output))
        return inputs + output


class DynamicPatchEmbed(nn.Module):
    """DOFAv2 dynamic patch embedding.

    ``convert_patch_14_to_16`` is retained only for compatibility with some
    object-detection recipes. It is disabled by default because the published
    DOFAv2 segmentation implementation keeps a 14x14 kernel and stride.
    """

    def __init__(self,
                 wavelength_dim: int = 128,
                 kernel_size: int = 14,
                 embed_dim: int = 1024,
                 convert_patch_14_to_16: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1
        self.convert_patch_14_to_16 = convert_patch_14_to_16
        self.weight_generator = TransformerWeightGenerator(
            wavelength_dim,
            kernel_size * kernel_size * embed_dim,
            embed_dim,
        )
        self.fclayer = FCResLayer(wavelength_dim)
        self.scaler = 0.01
        self._init_weights()

    @staticmethod
    def _init_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)

    def _init_weights(self) -> None:
        self.weight_generator.apply(self._init_linear)
        self.fclayer.apply(self._init_linear)

    def forward(
            self, image: torch.Tensor,
            wavelengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wave_embeddings = get_1d_sincos_pos_embed(
            self.weight_generator.fc_weight.in_features,
            wavelengths * 1000,
        )
        wave_embeddings = self.fclayer(wave_embeddings)
        weights, bias = self.weight_generator(wave_embeddings)
        weights = weights.view(
            wavelengths.numel(), self.kernel_size, self.kernel_size,
            self.embed_dim)
        weights = weights.permute(3, 0, 1, 2) * self.scaler
        bias = bias.reshape(self.embed_dim) * self.scaler

        stride = self.kernel_size
        if self.convert_patch_14_to_16:
            if self.kernel_size != 14:
                raise ValueError(
                    'convert_patch_14_to_16 requires kernel_size=14.')
            stride = 16
            weights = F.interpolate(
                weights,
                size=(16, 16),
                mode='bicubic',
                align_corners=False,
            )

        tokens = F.conv2d(
            image, weights, bias=bias, stride=stride, padding=1)
        return tokens.flatten(2).transpose(1, 2), wave_embeddings


@MODELS.register_module()
class DOFAV2ViT(BaseModule):
    """DOFAv2 ViT backbone for MMSegmentation.

    The upstream checkpoint is loaded exactly once into the underlying timm
    ViT. ``freeze_backbone`` is an explicit boolean and replaces the ambiguous
    ``frozen_stages=0`` convention used by the initial migration.
    """

    def __init__(
        self,
        arch: str = 'large',
        img_size: int = 224,
        pretrain_img_size: int = 224,
        patch_size: int = 14,
        out_indices: tuple[int, ...] | list[int] | None = None,
        model_bands: tuple[str, ...] | list[str] = ('RED', 'GREEN', 'BLUE'),
        convert_patch_14_to_16: bool = False,
        wavelength_dim: int = 128,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        freeze_backbone: bool = False,
        frozen_stages: bool | int | None = None,
        init_cfg: dict | None = None,
    ):
        # Loading is handled below against self.model so that official,
        # unprefixed DOFA checkpoint keys match. Giving BaseModule the same
        # init_cfg would load the checkpoint a second time at the wrapper.
        super().__init__(init_cfg=None)
        self.pretrained_init_cfg = init_cfg
        settings = get_arch_setting(arch)
        self.embed_dim = settings['embed_dim']
        self.depth = settings['depth']
        self.out_indices = tuple(
            settings['default_out_indices']
            if out_indices is None else out_indices)
        self._validate_out_indices()

        if frozen_stages is not None:
            if frozen_stages not in (False, True, 0, 1):
                raise ValueError(
                    'Legacy frozen_stages only accepts 0/1. Use the explicit '
                    'freeze_backbone boolean for new configs.')
            warnings.warn(
                'frozen_stages is deprecated for DOFAV2ViT; use '
                'freeze_backbone=True/False. It is interpreted as a boolean.',
                DeprecationWarning,
                stacklevel=2,
            )
            freeze_backbone = bool(frozen_stages)
        self.freeze_backbone = freeze_backbone
        self.convert_patch_14_to_16 = convert_patch_14_to_16
        self.effective_patch_size = (
            16 if convert_patch_14_to_16 else patch_size)

        self.register_buffer(
            'wavelengths',
            torch.tensor(get_wavelengths(model_bands), dtype=torch.float32),
            persistent=False,
        )

        self.model = VisionTransformer(
            # Keep the positional embedding at the pretraining resolution.
            # dynamic_img_size resamples it at runtime for inputs such as
            # 512x512. Constructing it directly at 512 would make the official
            # 224x224 checkpoint fail with a pos_embed shape mismatch.
            img_size=pretrain_img_size,
            patch_size=patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=settings['num_heads'],
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.model.patch_embed = DynamicPatchEmbed(
            wavelength_dim=wavelength_dim,
            kernel_size=patch_size,
            embed_dim=self.embed_dim,
            convert_patch_14_to_16=convert_patch_14_to_16,
        )
        self.model.patch_embed.num_patches = (
            img_size // self.effective_patch_size)**2
        self._apply_freezing()

    def _validate_out_indices(self) -> None:
        if len(set(self.out_indices)) != len(self.out_indices):
            raise ValueError('out_indices must not contain duplicates.')
        invalid = [i for i in self.out_indices if not 0 <= i < self.depth]
        if invalid:
            raise ValueError(
                f'out_indices {invalid} are outside [0, {self.depth - 1}].')

    @staticmethod
    def _unwrap_checkpoint(checkpoint: dict) -> dict:
        for container_key in ('state_dict', 'model'):
            if (container_key in checkpoint
                    and isinstance(checkpoint[container_key], dict)):
                checkpoint = checkpoint[container_key]
        normalized_keys = []
        for raw_key in checkpoint:
            key = raw_key
            for prefix in ('module.', 'backbone.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            normalized_keys.append(key)
        has_nested_model_norm = any(
            key.startswith('model.norm.') for key in normalized_keys)

        cleaned = {}
        for raw_key, value in checkpoint.items():
            key = raw_key
            for prefix in ('module.', 'backbone.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            is_nested_model_key = key.startswith('model.')
            if is_nested_model_key:
                key = key[len('model.'):]
            elif has_nested_model_norm and key.startswith('norm.'):
                # The upstream wrapper contains an unused ``norm`` beside
                # ``model.norm``. Do not let it overwrite the actual timm
                # transformer norm after prefixes are removed.
                continue
            cleaned[key] = value
        return cleaned

    def init_weights(self) -> None:
        super().init_weights()
        init_cfg = self.pretrained_init_cfg
        if init_cfg is not None:
            if init_cfg.get('type') != 'Pretrained':
                raise ValueError(
                    'DOFAV2ViT only supports init_cfg type="Pretrained".')
            checkpoint_path = init_cfg.get('checkpoint')
            if not checkpoint_path:
                raise ValueError(
                    'A non-empty checkpoint path is required in init_cfg.')
            checkpoint = CheckpointLoader.load_checkpoint(
                checkpoint_path, map_location='cpu')
            if not isinstance(checkpoint, dict):
                raise TypeError('The DOFAv2 checkpoint must contain a dict.')
            state_dict = self._unwrap_checkpoint(checkpoint)
            message = self.model.load_state_dict(state_dict, strict=False)
            logging.getLogger(__name__).info(
                'Loaded DOFAv2 checkpoint %s: %s',
                checkpoint_path,
                message,
            )
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        if not self.freeze_backbone:
            return
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_freezing()
        return self

    def _patch_hw(self, image: torch.Tensor) -> tuple[int, int]:
        height, width = image.shape[-2:]
        patch = self.effective_patch_size
        return (
            (height + 2 - patch) // patch + 1,
            (width + 2 - patch) // patch + 1,
        )

    def _tokens_to_image(self, tokens: torch.Tensor,
                         hw_shape: tuple[int, int]) -> torch.Tensor:
        # timm prepends its class token in _pos_embed.
        tokens = tokens[:, 1:, :]
        tokens = tokens.reshape(
            tokens.shape[0], hw_shape[0], hw_shape[1], self.embed_dim)
        return tokens.permute(0, 3, 1, 2).contiguous()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if image.shape[1] != self.wavelengths.numel():
            raise ValueError(
                f'DOFAv2 received {image.shape[1]} channels but '
                f'{self.wavelengths.numel()} model_bands were configured.')

        hw_shape = self._patch_hw(image)
        wavelengths = self.wavelengths.to(
            device=image.device, dtype=torch.float32)
        tokens, _ = self.model.patch_embed(image, wavelengths)
        expected_tokens = hw_shape[0] * hw_shape[1]
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f'Expected {expected_tokens} patch tokens, '
                f'got {tokens.shape[1]}.')

        tokens = tokens.view(
            tokens.shape[0], hw_shape[0], hw_shape[1], tokens.shape[2])
        tokens = self.model._pos_embed(tokens)
        tokens = self.model.patch_drop(tokens)
        tokens = self.model.norm_pre(tokens)

        outputs = []
        for index, block in enumerate(self.model.blocks):
            tokens = block(tokens)
            if index in self.out_indices:
                outputs.append(self._tokens_to_image(tokens, hw_shape))
        return tuple(outputs)
