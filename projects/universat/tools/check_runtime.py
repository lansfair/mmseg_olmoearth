"""Inspect one UniverSat sample and optionally run a backbone forward pass."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def _parse_cfg_options(items: list[str] | None) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"cfg option must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        try:
            options[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            options[key] = value
    return options


def _tensor_summary(value) -> dict[str, Any]:
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--cfg-options", nargs="*")
    args = parser.parse_args()

    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings
    from mmseg.registry import DATASETS, MODELS

    cfg = Config.fromfile(Path(args.config))
    cfg.merge_from_dict(_parse_cfg_options(args.cfg_options))
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get("default_scope", "mmseg"))

    dataset = DATASETS.build(cfg[f"{args.split}_dataloader"]["dataset"])
    item = dataset[args.index]
    inputs = {key: value.unsqueeze(0) for key, value in item["inputs"].items()}
    sample = item["data_samples"]

    summary: dict[str, Any] = {
        "config": str(Path(args.config)),
        "split": args.split,
        "index": args.index,
        "dataset_length": len(dataset),
        "inputs": {key: _tensor_summary(value) for key, value in inputs.items()},
        "gt_sem_seg": _tensor_summary(sample.gt_sem_seg.data),
    }

    if not args.skip_forward:
        model = MODELS.build(cfg.model)
        model.init_weights()
        model.to(args.device).eval()
        data = model.data_preprocessor(
            {"inputs": inputs, "data_samples": [sample]}, training=False
        )
        with torch.inference_mode():
            features = model.extract_feat(data["inputs"])
        summary["features"] = [_tensor_summary(value) for value in features]
        summary["trainable_parameters"] = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
