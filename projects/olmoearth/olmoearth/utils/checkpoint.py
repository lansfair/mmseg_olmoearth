from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _build_model_from_config(config_path: Path) -> Any:
    from olmoearth_pretrain.config import Config
    from olmoearth_pretrain.model_loader import patch_legacy_encoder_config

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict = patch_legacy_encoder_config(config_dict)
    return Config.from_dict(config_dict["model"]).build()


def _unwrap_state_dict(value: Any) -> dict[str, torch.Tensor]:
    if isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]
    elif isinstance(value, dict) and "model" in value:
        value = value["model"]
    if not isinstance(value, dict):
        raise TypeError(
            "OLMoEarth checkpoint must be a state_dict or contain "
            "'state_dict'/'model'."
        )
    return value


def _load_model_from_pth(weights_path: Path, config_path: Path | None) -> Any:
    if config_path is None:
        config_path = weights_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "A standalone .pth needs a matching OLMoEarth config.json. "
            f"Expected {config_path}, or pass model.backbone.config_path."
        )
    model = _build_model_from_config(config_path)
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    return model


def _model_id_enum(model_id: str):
    from olmoearth_pretrain.model_loader import ModelID

    try:
        return ModelID(model_id)
    except ValueError as exc:
        supported = ", ".join(item.value for item in ModelID)
        raise ValueError(
            f"Unsupported OLMoEarth model_id={model_id!r}. "
            f"Supported values: {supported}"
        ) from exc


def load_olmoearth_model(
    checkpoint_path: str | Path | None = None,
    *,
    model_id: str | None = None,
    config_path: str | Path | None = None,
) -> Any:
    """Load OLMoEarth from a HF model id, artifact directory, or .pth file."""

    if model_id is not None:
        from olmoearth_pretrain.model_loader import load_model_from_id

        return load_model_from_id(_model_id_enum(model_id))

    if checkpoint_path is None:
        raise ValueError(
            "Either checkpoint_path or model_id must be provided."
        )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint_path does not exist: {checkpoint_path}"
        )

    if checkpoint_path.is_file():
        return _load_model_from_pth(
            weights_path=checkpoint_path,
            config_path=Path(config_path) if config_path is not None else None,
        )

    from olmoearth_pretrain.model_loader import load_model_from_path

    return load_model_from_path(str(checkpoint_path))
