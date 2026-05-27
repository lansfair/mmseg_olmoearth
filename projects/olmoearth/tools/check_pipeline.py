from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def _shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return list(value.shape)
    return None


def _pixel_data_shape(sample, key: str) -> list[int] | None:
    if not hasattr(sample, key):
        return None
    value = getattr(sample, key)
    if hasattr(value, "data"):
        value = value.data
    return _shape(value)


def _pixel_data_range(sample, key: str) -> dict[str, int] | None:
    if not hasattr(sample, key):
        return None
    value = getattr(sample, key)
    if hasattr(value, "data"):
        value = value.data
    if value.numel() == 0:
        return None
    return {"min": int(value.min().item()), "max": int(value.max().item())}


def _valid_pixels(sample) -> dict[str, int] | None:
    if not hasattr(sample, "gt_valid_mask"):
        return None
    valid = sample.gt_valid_mask.data
    return {
        "valid": int((valid > 0).sum().item()),
        "total": int(valid.numel()),
    }


def _tensor_meta(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _parse_cfg_options(items: list[str] | None) -> dict[str, Any] | None:
    if items is None:
        return None
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"cfg option must be key=value, got: {item}")
        key, value = item.split("=", 1)
        if value.lower() == "none":
            parsed = None
        elif value.lower() in {"true", "false"}:
            parsed = value.lower() == "true"
        else:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = value
        out[key] = parsed
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an MMSeg dataset from config and inspect one sample."
        )
    )
    parser.add_argument("config")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        default=None,
        help="Override config options, e.g. key=value.",
    )
    args = parser.parse_args()

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings
    from mmseg.registry import DATASETS

    cfg = Config.fromfile(Path(args.config))
    cfg_options = _parse_cfg_options(args.cfg_options)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get("default_scope", "mmseg"))

    dataloader_cfg = cfg[f"{args.split}_dataloader"]
    dataset = DATASETS.build(dataloader_cfg["dataset"])
    item = dataset[args.index]
    inputs = item["inputs"]
    sample = item["data_samples"]

    summary = {
        "config": str(Path(args.config)),
        "split": args.split,
        "index": args.index,
        "dataset_length": len(dataset),
        "inputs_shape": _shape(inputs),
        "gt_sem_seg_shape": _pixel_data_shape(sample, "gt_sem_seg"),
        "gt_sem_seg_range": _pixel_data_range(sample, "gt_sem_seg"),
        "gt_valid_mask_shape": _pixel_data_shape(sample, "gt_valid_mask"),
        "valid_pixels": _valid_pixels(sample),
        "metainfo": {
            key: _tensor_meta(value)
            for key, value in sample.metainfo.items()
            if key in {"sample_id", "timestamps", "olmoearth_modality"}
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
