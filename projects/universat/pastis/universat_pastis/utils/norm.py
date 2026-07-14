"""PASTIS-R normalization helpers mirroring the original UniverSat code."""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from mmengine.logging import print_log


def load_norm(
    norm_path: Optional[str],
    modalities: List[str],
) -> Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """Load per-modality normalisation tensors from ``norm_path``.

    Expected files::

        {norm_path}/NORM_{modality}_patch.json

    Each JSON file should contain ``{"mean": [...], "std": [...]}``.
    Missing files are skipped with a warning.

    Returns:
        dict mapping modality -> (mean, std) tensors, or None if
        ``norm_path`` is None.
    """
    if norm_path is None:
        return None

    norm = {}
    for mod in modalities:
        file_path = os.path.join(norm_path, f"NORM_{mod}_patch.json")
        if not os.path.exists(file_path):
            print_log(
                f"[UniverSatPASTIS] normalization file not found: {file_path}",
                logger="current",
                level=30,
            )
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            vals = json.load(f)
        norm[mod] = (
            torch.tensor(vals["mean"], dtype=torch.float32),
            torch.tensor(vals["std"], dtype=torch.float32),
        )
    return norm


def apply_norm(
    norm: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    output: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Standardise each modality tensor in ``output`` in place.

    Args:
        norm: ``{modality: (mean, std)}`` mapping.
        output: dict containing modality tensors. Spatial-only modalities have
            shape ``(C, H, W)``; time-series modalities have ``(T, C, H, W)``.

    Returns:
        The same dict with normalized tensors.
    """
    if norm is None:
        return output

    for modality, (mean, std) in norm.items():
        if modality not in output:
            continue
        data = output[modality]
        if data.ndim == 3:  # (C, H, W)
            m = mean.view(-1, 1, 1)
            s = std.view(-1, 1, 1)
        elif data.ndim == 4:  # (T, C, H, W)
            m = mean.view(1, -1, 1, 1)
            s = std.view(1, -1, 1, 1)
        else:
            continue
        output[modality] = (data - m) / s.clamp_min(1e-6)
    return output
