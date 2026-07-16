"""MMSegmentation 1.x backbone wrapper for UniverSat.

This file is a migration template for OpenMMLab projects. Copy the whole
``universat/`` project folder under ``mmsegmentation/projects/`` (or keep it in
``mmsegmentation/projects/universat/``) and run training with the provided
configs.

The underlying encoder code lives in ``universat_modules/`` and is kept as
close as possible to the original repository so that released checkpoints can
be loaded without key renaming beyond stripping the ``model.`` hub wrapper
prefix.
"""

import os
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from .universat_modules.UniverSat import UniverSat
from .universat_modules.UniversalPatchEncoder import UniversalPatchEncoder
from .universat_modules.utils.utils import RMSNorm
from .universat_modules.modality_registry import INPUT_RES, SUBPATCHES, WAVELENGTHS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unroll_block_list(blocks: List[str]) -> List[str]:
    """Same helper used by ``UniverSat.__init__`` for config compatibility."""
    out = []
    for b in blocks:
        if "x" in b:
            n = int(b.split("x")[1])
            bt = b.split("x")[0]
            out.extend([bt] * n)
        else:
            out.append(b)
    return out


def _load_checkpoint(path: str, map_location: str = "cpu"):
    """Load a checkpoint in either PyTorch or SafeTensors format."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "Loading .safetensors checkpoints requires `safetensors`. "
                "Install it with `pip install safetensors`."
            ) from exc
        return load_file(path, device=map_location)
    else:
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)


def _strip_hub_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the ``model.`` prefix added by ``_UniverSatHub``."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model.") :]
        new_state[k] = v
    return new_state


DEFAULT_MODALITIES_DICT = {
    "flair": ["spotRGBN", "aerialflair", "s2flair", "s1flair", "dem"],
    "pastishd": ["spot", "s2", "s1"],
    "planted": ["s2", "s1", "l7", "alos", "modis"],
    "tsaits": ["aerial", "s2", "s1"],
    "s2naip": ["naip", "l8", "s2", "s1"],
    "hyperglobal": ["EO1"],
    "earthview": ["rgbneon", "ndemneon", "neon"],
    "spectralearth": ["enmap"],
}


@MODELS.register_module()
class UniverSatBackbone(BaseModule):
    """MMSegmentation 1.x backbone wrapping the UniverSat multimodal encoder.

    The encoder receives a dict ``{modality_name: tensor}`` and returns a
    single-scale feature map of shape ``(B, embed_dim, H, W)`` where ``H x W``
    corresponds to the requested ``output_grid``.

    Args:
        modalities: List of modality names that will be fed to the encoder.
            Must match the keys present in the input dict.
        embed_dim: Token embedding dimension. Default 768 (Base).
        num_heads: Attention heads per trunk block. Default 12.
        mlp_ratio: Hidden-to-input ratio in trunk MLPs. Default 4.0.
        patch_size: Physical patch size in meters. Default 40.0.
        output_grid: Side length of the square output token grid. For example,
            ``36`` requests a 36 x 36 grid. If None, it is inferred from the
            input spatial resolution and the largest sub-patch factor.
        block_type: Ordered block kinds composing the trunk.
        n_registers: Number of register tokens prepended to the trunk.
        gating: Enable sigmoid gating in attention layers.
        proba_drop_modalities: Probability of dropping each non-MODIS
            modality at SSL pre-training time. Set to 0.0 for inference.
        modalities_dict: Dataset-to-modality mapping used to create the SSL
            projector heads. Required only when loading SSL checkpoints.
        wavelengths: Optional per-modality wavelength overrides.
        input_res: Optional per-modality resolution overrides (m/px).
        subpatches: Optional per-modality sub-patch factor overrides.
        keep_intermediate: Forward intermediate modality tokens.
        out_indices: Indices of the feature maps to return.
        multi_grids: Optional list of output grid side lengths for multi-scale
            feature extraction.
        freeze_backbone: Freeze the complete encoder for linear probing,
            matching the explicit switch used by the original UniverSat
            downstream implementation.
        frozen_stages: ``-1`` keeps the whole backbone trainable. ``0`` freezes
            the UPE and a positive value freezes the UPE plus that many leading
            trunk blocks. Use ``freeze_backbone=True`` to freeze everything.
        compile_encoder: Compile the two encoder stages with dynamic shapes.
            Disabled by default because variable-length satellite time series
            can otherwise trigger expensive recompilations.
        init_cfg: MMSegmentation init config, e.g.
            ``dict(type='Pretrained', checkpoint='path/to/model.safetensors')``.
    """

    def __init__(
        self,
        modalities: Sequence[str],
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: float = 40.0,
        output_grid: Optional[int] = None,
        block_type: Sequence[str] = ("Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub"),
        n_registers: int = 4,
        gating: bool = True,
        proba_drop_modalities: float = 0.0,
        modalities_dict: Optional[Dict[str, List[str]]] = None,
        wavelengths: Optional[Dict[str, List[Union[float, str]]]] = None,
        input_res: Optional[Dict[str, float]] = None,
        subpatches: Optional[Dict[str, int]] = None,
        keep_intermediate: bool = False,
        out_indices: Tuple[int, ...] = (-1,),
        multi_grids: Optional[Sequence[int]] = None,
        freeze_backbone: bool = False,
        frozen_stages: int = -1,
        compile_encoder: bool = False,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        if freeze_backbone and frozen_stages >= 0:
            raise ValueError(
                "freeze_backbone=True cannot be combined with partial "
                "frozen_stages. Set frozen_stages=-1."
            )
        self.freeze_backbone = freeze_backbone
        self.frozen_stages = frozen_stages
        self.compile_encoder = compile_encoder
        self.modalities = list(modalities)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.output_grid = output_grid
        self.block_type = _unroll_block_list(list(block_type))
        self.n_registers = n_registers
        self.gating = gating
        self.proba_drop_modalities = proba_drop_modalities
        self.modalities_dict = modalities_dict or DEFAULT_MODALITIES_DICT
        self.keep_intermediate = keep_intermediate
        self.out_indices = out_indices
        self.multi_grids = list(multi_grids) if multi_grids else None

        self.wavelengths = {}
        self.input_res = {}
        self.subpatches = {}
        for mod in self.modalities:
            if wavelengths is not None and mod in wavelengths:
                self.wavelengths[mod] = list(wavelengths[mod])
            elif mod in WAVELENGTHS:
                self.wavelengths[mod] = list(WAVELENGTHS[mod])
            else:
                raise KeyError(
                    f"Modality {mod!r} not in modality_registry and no "
                    f"`wavelengths` override provided."
                )

            if input_res is not None and mod in input_res:
                self.input_res[mod] = float(input_res[mod])
            elif mod in INPUT_RES:
                self.input_res[mod] = float(INPUT_RES[mod])
            else:
                raise KeyError(
                    f"Modality {mod!r} has no `input_res` in registry and no "
                    f"override provided."
                )

            if subpatches is not None and mod in subpatches:
                self.subpatches[mod] = int(subpatches[mod])
            else:
                self.subpatches[mod] = int(SUBPATCHES.get(mod, 1))

        spatial_encoder = partial(
            UniversalPatchEncoder,
            embed_dim=embed_dim // 8,
            final_dim=embed_dim,
            n_queries=[1, 1, 1, 1],
            expand_dim=[2, 2, 2, 2],
            order=["S1", "C", "T", "S"],
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop_rate=0.0,
            gating=gating,
        )

        self.model = UniverSat(
            spatial_encoder=spatial_encoder,
            block_type=self.block_type,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=False,
            n_registers=n_registers,
            pre_norm=False,
            drop_rate=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
            norm_layer=partial(RMSNorm),
            gating=gating,
            proba_drop_modalities=proba_drop_modalities,
            modalities_dict=self.modalities_dict,
        )

        if self.compile_encoder:
            self.model.UPE_forward = torch.compile(self.model.UPE_forward, dynamic=True)
            self.model.ViT_forward = torch.compile(self.model.ViT_forward, dynamic=True)

        # MMSegmentation necks read this attribute.
        self.out_channels = [embed_dim]

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze stages according to ``self.frozen_stages``.

        ``freeze_backbone=True`` freezes the complete encoder.
        ``frozen_stages=-1`` disables partial freezing, ``0`` freezes the UPE,
        and a positive value freezes the UPE plus the first N trunk blocks.
        """
        if self.freeze_backbone:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            return

        if self.frozen_stages < 0:
            return

        # Freeze UPE (stage 0)
        self.model.spatial_encoder.eval()
        for param in self.model.spatial_encoder.parameters():
            param.requires_grad = False

        # Freeze the first N trunk blocks.
        for i, block in enumerate(self.model.blocks):
            if i + 1 <= self.frozen_stages:
                block.eval()
                for param in block.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Override train so frozen stages stay in eval mode."""
        super().train(mode)
        self._freeze_stages()
        return self

    def init_weights(self) -> None:
        """Load pretrained weights from ``init_cfg['checkpoint']``."""
        if self.init_cfg is None:
            return

        checkpoint_path = self.init_cfg.get("checkpoint")
        if checkpoint_path is None:
            super().init_weights()
            return

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = _load_checkpoint(checkpoint_path, map_location="cpu")

        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]

        state_dict = _strip_hub_prefix(state_dict)

        ignored = [k for k in state_dict if k.startswith("projector__")]
        for k in ignored:
            del state_dict[k]
        if ignored:
            print_log(
                f"Ignored {len(ignored)} projector weights.",
                logger="current",
                level=30,  # WARNING
            )

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print_log(
                f"Missing keys when loading UniverSat backbone: {missing}",
                logger="current",
                level=30,
            )
        if unexpected:
            print_log(
                f"Unexpected keys when loading UniverSat backbone: {unexpected}",
                logger="current",
                level=30,
            )

        print_log(
            f"Loaded UniverSat weights from {checkpoint_path}",
            logger="current",
        )

    def _infer_latent_grid(self, x: Dict[str, torch.Tensor]) -> int:
        """Infer latent token count from the first spatial modality."""
        for mod in self.modalities:
            t = x[mod]
            if t.ndim >= 4:
                h, w = t.shape[-2:]
                if h != w:
                    raise ValueError(
                        f"UniverSat requires square inputs, but {mod!r} has "
                        f"spatial shape {(h, w)}."
                    )
                patch_px = max(int(self.patch_size / self.input_res[mod]), 1)
                if h % patch_px:
                    raise ValueError(
                        f"Input side {h} for {mod!r} is not divisible by its "
                        f"{patch_px}-pixel patch size."
                    )
                side = max(h // patch_px, 1)
                return side * side
        raise ValueError("Cannot infer latent_grid from inputs.")

    def _compute_output_grid(self, latent_grid: int) -> int:
        """Return output side length (G of GxG)."""
        if self.output_grid is not None:
            if self.output_grid <= 0:
                raise ValueError("output_grid must be a positive side length.")
            return self.output_grid
        latent_side = int(latent_grid**0.5)
        max_sub = max(self.subpatches.values()) if self.subpatches else 1
        return latent_side * max_sub

    def _run_once(
        self,
        x: Dict[str, torch.Tensor],
        latent_grid: int,
        output_side: int,
    ) -> torch.Tensor:
        """Single forward pass returning (B, C, H, W) features."""
        output_tokens = output_side * output_side
        scale = self.patch_size / 10.0

        tokens, _ = self.model(
            x=x,
            wavelengths=self.wavelengths,
            input_res=self.input_res,
            scale=scale,
            latent_grid=latent_grid,
            output_grid=output_tokens,
            subpatches=self.subpatches,
            keep_intermediate=self.keep_intermediate,
            mask_in=None,
            mask_out=None,
        )

        tokens = tokens[:, self.n_registers :]
        b, n, c = tokens.shape
        h_w = int(round(n**0.5))
        if h_w * h_w != n:
            raise RuntimeError(
                f"UniverSat returned {n} spatial tokens, which cannot be "
                "reshaped to a square feature map."
            )
        features = tokens.permute(0, 2, 1).reshape(b, c, h_w, h_w)
        return features

    def forward(
        self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> List[torch.Tensor]:
        """MMSeg-style forward.

        Args:
            x: Either a dict ``{modality: tensor}`` or a single tensor.

        Returns:
            list[torch.Tensor]: Feature maps.
        """
        if not isinstance(x, dict):
            if len(self.modalities) != 1:
                raise ValueError(
                    "Single-tensor input is only allowed when exactly one "
                    "modality is configured."
                )
            x = {self.modalities[0]: x}

        for mod in self.modalities:
            if mod not in x:
                raise KeyError(
                    f"Modality {mod!r} missing from input. Got keys: {list(x.keys())}"
                )

            dates_key = f"{mod}_dates"
            if dates_key in x:
                if x[mod].ndim != 5 or x[dates_key].ndim != 2:
                    raise ValueError(
                        f"Time-series {mod!r} must use data shape "
                        f"(B,T,C,H,W) and dates shape (B,T); got "
                        f"{tuple(x[mod].shape)} and {tuple(x[dates_key].shape)}."
                    )
                if x[mod].shape[:2] != x[dates_key].shape:
                    raise ValueError(
                        f"Time and date shapes differ for {mod!r}: "
                        f"{tuple(x[mod].shape[:2])} vs "
                        f"{tuple(x[dates_key].shape)}."
                    )
            elif x[mod].ndim != 4:
                raise ValueError(
                    f"Spatial-only {mod!r} must use shape (B,C,H,W); got "
                    f"{tuple(x[mod].shape)}. Time series also require "
                    f"the {dates_key!r} tensor."
                )

        latent_grid = self._infer_latent_grid(x)

        if self.multi_grids:
            features = [
                self._run_once(x, latent_grid, grid_side)
                for grid_side in self.multi_grids
            ]
            return [features[i] for i in self.out_indices]

        output_side = self._compute_output_grid(latent_grid)
        return [self._run_once(x, latent_grid, output_side)]
