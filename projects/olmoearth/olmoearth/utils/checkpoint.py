from __future__ import annotations

from pathlib import Path
from typing import Any


def load_olmoearth_model(checkpoint_path: str | Path) -> Any:
    """Load OLMoEarth from an explicit checkpoint path."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint_path does not exist: {checkpoint_path}"
        )

    from olmoearth_pretrain.model_loader import load_model_from_path

    return load_model_from_path(str(checkpoint_path))
