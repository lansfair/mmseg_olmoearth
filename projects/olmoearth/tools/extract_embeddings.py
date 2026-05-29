from __future__ import annotations

import argparse
import ast
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import save_geotiff, save_json


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


def _safe_sample_id(index: int, metainfo: dict[str, Any]) -> str:
    sample_id = metainfo.get("sample_id")
    if sample_id is None:
        sample_id = metainfo.get("img_path")
    if sample_id is None:
        sample_id = f"{index:06d}"
    return str(sample_id).replace("\\", "_").replace("/", "_")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _build_dataset(cfg, split: str, pipeline_key: str | None):
    from mmseg.registry import DATASETS

    dataloader_key = f"{split}_dataloader"
    if dataloader_key not in cfg:
        raise KeyError(f"Config does not define {dataloader_key}.")
    dataset_cfg = copy.deepcopy(cfg[dataloader_key]["dataset"])
    if pipeline_key is not None:
        if pipeline_key not in cfg:
            raise KeyError(f"Config does not define {pipeline_key}.")
        dataset_cfg["pipeline"] = copy.deepcopy(cfg[pipeline_key])
    return DATASETS.build(dataset_cfg)


def _build_backbone(cfg, device: torch.device):
    from mmseg.registry import MODELS

    backbone = MODELS.build(copy.deepcopy(cfg.model.backbone))
    backbone.init_weights()
    backbone.to(device)
    backbone.eval()
    return backbone


def _iter_batches(dataset, batch_size: int):
    batch = []
    for idx in range(len(dataset)):
        batch.append((idx, dataset[idx]))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _stack_inputs(batch_items: list[dict[str, Any]], device: torch.device):
    shapes = {tuple(item["inputs"].shape) for item in batch_items}
    if len(shapes) != 1:
        raise ValueError(
            "Embedding extraction batches require identical input shapes. "
            f"Got {sorted(shapes)}. Use --batch-size 1 or pre-tile the data."
        )
    return torch.stack([item["inputs"].float() for item in batch_items]).to(device)


def _extract_split(
    cfg,
    split: str,
    output_root: Path,
    batch_size: int,
    device: torch.device,
    pipeline_key: str | None,
) -> dict[str, Any]:
    dataset = _build_dataset(cfg, split, pipeline_key)
    backbone = _build_backbone(cfg, device)
    split_samples = []
    split_dir = output_root / split
    embedding_names = None

    with torch.inference_mode():
        for batch in _iter_batches(dataset, batch_size):
            indices = [idx for idx, _ in batch]
            items = [item for _, item in batch]
            inputs = _stack_inputs(items, device)
            metainfo = [
                item["data_samples"].metainfo
                for item in items
            ]
            if hasattr(backbone, "set_batch_metainfo"):
                backbone.set_batch_metainfo(metainfo)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                features = backbone(inputs)[0]
            features_np = features.float().cpu().numpy()

            for local_idx, item in enumerate(items):
                sample_meta = metainfo[local_idx]
                sample_id = _safe_sample_id(indices[local_idx], sample_meta)
                sample_dir = split_dir / sample_id
                embedding_rel = Path(split) / sample_id / "embedding.tif"
                label_rel = Path(split) / sample_id / "label.tif"
                feature = features_np[local_idx]
                label = (
                    item["data_samples"]
                    .gt_sem_seg
                    .data
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                if embedding_names is None:
                    embedding_names = [
                        f"embedding_{idx:04d}"
                        for idx in range(feature.shape[0])
                    ]
                save_geotiff(
                    output_root / embedding_rel,
                    feature.astype(np.float32, copy=False),
                    descriptions=embedding_names,
                )
                save_geotiff(output_root / label_rel, label)

                sample = {
                    "sample_id": sample_id,
                    "embedding_path": str(embedding_rel).replace("\\", "/"),
                    "seg_map_path": str(label_rel).replace("\\", "/"),
                    "dataset_name": sample_meta.get("dataset_name"),
                    "ori_shape": list(label.shape),
                    "embedding_shape": list(feature.shape),
                }
                if "timestamps" in sample_meta:
                    sample["timestamps"] = _jsonable(sample_meta["timestamps"])
                split_samples.append(sample)

    manifest = {
        "metainfo": {
            "source_config": str(getattr(cfg, "filename", "")),
            "split": split,
            "format": "olmoearth_embedding_geotiff_manifest",
        },
        "samples": split_samples,
    }
    save_json(output_root / f"{split}.json", manifest)
    return {
        "split": split,
        "num_samples": len(split_samples),
        "manifest": str(output_root / f"{split}.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract dense OLMoEarth embeddings for offline MMSeg probes."
    )
    parser.add_argument("config", help="Online OLMoEarth MMSeg config.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pipeline-key",
        default="test_pipeline",
        help=(
            "Pipeline to use for extraction. Use 'none' to keep each "
            "dataloader's configured pipeline."
        ),
    )
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

    cfg = Config.fromfile(Path(args.config))
    cfg_options = _parse_cfg_options(args.cfg_options)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get("default_scope", "mmseg"))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline_key = None if args.pipeline_key.lower() == "none" else args.pipeline_key
    device = torch.device(args.device)
    summaries = [
        _extract_split(
            cfg=cfg,
            split=split,
            output_root=output_root,
            batch_size=args.batch_size,
            device=device,
            pipeline_key=pipeline_key,
        )
        for split in args.splits
    ]
    save_json(output_root / "summary.json", {"splits": summaries})


if __name__ == "__main__":
    main()
