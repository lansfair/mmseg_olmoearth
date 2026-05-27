from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def _tensor_summary(value) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _loss_summary(losses: dict[str, Any]) -> dict[str, float | list[float]]:
    out: dict[str, float | list[float]] = {}
    for key, value in losses.items():
        if isinstance(value, list):
            out[key] = [float(v.detach().cpu().item()) for v in value]
        else:
            out[key] = float(value.detach().cpu().item())
    return out


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
            "Build OLMoEarth MMSeg dataset/model and run one loss step."
        )
    )
    parser.add_argument("config")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        default=None,
        help="Override config options, e.g. key=value.",
    )
    args = parser.parse_args()

    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings
    from mmseg.registry import DATASETS, MODELS

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

    model = MODELS.build(cfg.model)
    model.to(args.device)
    model.train()

    data = model.data_preprocessor(
        {
            "inputs": [item["inputs"]],
            "data_samples": [item["data_samples"]],
        },
        training=True,
    )

    with torch.no_grad():
        losses = model.loss(data["inputs"], data["data_samples"])

    summary = {
        "config": str(Path(args.config)),
        "split": args.split,
        "index": args.index,
        "input": _tensor_summary(data["inputs"]),
        "losses": _loss_summary(losses),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
