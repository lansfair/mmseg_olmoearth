from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _safe_id(value: Any, fallback: str) -> str:
    if value is None:
        value = fallback
    return str(value).replace("\\", "_").replace("/", "_")


def _sample_metadata(data_samples, batch_size: int) -> list[dict[str, Any]]:
    if data_samples is None:
        return [{} for _ in range(batch_size)]
    return [dict(sample.metainfo) for sample in data_samples]


def _input_hw(inputs: Any, metadata: dict[str, Any]) -> tuple[int, int] | None:
    shape = metadata.get("img_shape") or metadata.get("ori_shape")
    if shape is not None:
        return int(shape[0]), int(shape[1])
    if isinstance(inputs, torch.Tensor) and inputs.ndim >= 4:
        return int(inputs.shape[-2]), int(inputs.shape[-1])
    if isinstance(inputs, dict):
        modalities = inputs.get("modalities", inputs)
        if isinstance(modalities, dict):
            for value in modalities.values():
                if isinstance(value, torch.Tensor) and value.ndim >= 4:
                    return int(value.shape[-2]), int(value.shape[-1])
    return None


def _scaled_transform(
    metadata: dict[str, Any],
    source_hw: tuple[int, int] | None,
    destination_hw: tuple[int, int],
):
    transform = metadata.get("olmoearth_transform") or metadata.get("transform")
    if transform is None or source_hw is None:
        return None
    from rasterio.transform import Affine

    if not isinstance(transform, Affine):
        transform = Affine(*transform[:6])
    scale_x = source_hw[1] / destination_hw[1]
    scale_y = source_hw[0] / destination_hw[0]
    return transform * Affine.scale(scale_x, scale_y)


def _save_dense_geotiff(
    path: Path,
    embedding: torch.Tensor,
    metadata: dict[str, Any],
    source_hw: tuple[int, int] | None,
) -> None:
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "Dense GeoTIFF export requires rasterio. Use --dense-format pt "
            "or install rasterio."
        ) from exc

    array = embedding.detach().float().cpu().numpy()
    transform = _scaled_transform(metadata, source_hw, array.shape[-2:])
    crs = metadata.get("olmoearth_crs") or metadata.get("crs")
    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": "float32",
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    if transform is not None:
        profile["transform"] = transform
    if crs is not None:
        profile["crs"] = crs
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(array.astype(np.float32, copy=False))


def _save_embedding(
    path: Path,
    embedding: torch.Tensor,
    mode: str,
    dense_format: str,
    metadata: dict[str, Any],
    source_hw: tuple[int, int] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "dense" and dense_format == "geotiff":
        _save_dense_geotiff(path, embedding, metadata, source_hw)
        return
    torch.save(embedding.detach().float().cpu(), path)


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32" or device.type == "cpu":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def parse_args() -> argparse.Namespace:
    from mmengine.config import DictAction

    parser = argparse.ArgumentParser(
        description="Extract embeddings from a GeoFMBackbone config."
    )
    parser.add_argument("config")
    parser.add_argument("output_root")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--mode", choices=("global", "dense"), default=None)
    parser.add_argument(
        "--dense-format",
        choices=("geotiff", "pt"),
        default="geotiff",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def main() -> None:
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules

    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    register_all_modules(init_default_scope=True)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)

    device = torch.device(args.device)
    model = MODELS.build(cfg.model)
    model.init_weights()
    model.to(device).eval()
    if not hasattr(model.backbone, "extract"):
        raise TypeError("Configured backbone must be GeoFMBackbone-compatible.")
    if args.mode is not None:
        model.backbone.output_mode = args.mode
    mode = model.backbone.output_mode

    dataloader_cfg = cfg[f"{args.split}_dataloader"]
    dataloader = Runner.build_dataloader(dataloader_cfg)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    processed_count = 0

    with torch.inference_mode():
        for batch_index, data in enumerate(dataloader):
            processed = model.data_preprocessor(data, training=False)
            inputs = processed["inputs"]
            data_samples = processed.get("data_samples")

            if isinstance(inputs, torch.Tensor):
                batch_size = inputs.shape[0]
            else:
                modalities = inputs.get("modalities", inputs)
                first = next(
                    value
                    for value in modalities.values()
                    if isinstance(value, torch.Tensor)
                )
                batch_size = first.shape[0]
            metadata = _sample_metadata(data_samples, batch_size)
            model.backbone.set_batch_metainfo(metadata)
            with _autocast_context(device, args.precision):
                result = model.backbone.extract(inputs)
            embeddings = result.tensor
            if args.l2_normalize:
                embeddings = torch.nn.functional.normalize(embeddings, dim=1)
                result.normalized = True

            common_metadata = result.manifest_metadata()
            for item_index in range(batch_size):
                if args.limit is not None and processed_count >= args.limit:
                    break
                item_metadata = metadata[item_index]
                sample_id = _safe_id(
                    item_metadata.get("sample_id") or item_metadata.get("img_path"),
                    f"{batch_index:06d}_{item_index:03d}",
                )
                suffix = ".tif" if mode == "dense" and args.dense_format == "geotiff" else ".pt"
                relative_path = Path(args.split) / sample_id / f"embedding{suffix}"
                source_hw = _input_hw(inputs, item_metadata)
                _save_embedding(
                    output_root / relative_path,
                    embeddings[item_index],
                    mode,
                    args.dense_format,
                    item_metadata,
                    source_hw,
                )
                records.append(
                    {
                        "sample_id": sample_id,
                        "embedding_path": relative_path.as_posix(),
                        "embedding_shape": list(embeddings[item_index].shape),
                        **common_metadata,
                    }
                )
                processed_count += 1
            print(
                f"\rExtracted {processed_count} samples from {args.split}",
                end="",
                flush=True,
            )
            if args.limit is not None and processed_count >= args.limit:
                break

    print()
    manifest = {
        "format": "geofm_embedding_manifest_v1",
        "split": args.split,
        "count": len(records),
        "samples": records,
    }
    manifest_path = output_root / f"{args.split}.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
