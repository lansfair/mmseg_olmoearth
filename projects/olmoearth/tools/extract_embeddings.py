from __future__ import annotations

import argparse
import ast
import copy
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from common import save_geotiff, save_json


@dataclass(frozen=True)
class DistContext:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int


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


def _get_dist_context() -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistContext(
        is_distributed=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def _resolve_device(device_arg: str, ctx: DistContext) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        if ctx.is_distributed:
            return torch.device(f"cuda:{ctx.local_rank}")
        return torch.device("cuda")
    return torch.device("cpu")


def _init_distributed(ctx: DistContext, device: torch.device) -> None:
    if not ctx.is_distributed:
        return
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)


def _barrier(ctx: DistContext) -> None:
    if ctx.is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()


def _destroy_distributed(ctx: DistContext) -> None:
    if ctx.is_distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _configure_cuda_fast_math(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _clear_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


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


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _split_indices(length: int, ctx: DistContext) -> list[int]:
    if not ctx.is_distributed:
        return list(range(length))
    return list(range(ctx.rank, length, ctx.world_size))


def _embedding_shape(path: Path) -> list[int]:
    import rasterio

    with rasterio.open(path) as src:
        return [int(src.count), int(src.height), int(src.width)]


def _make_manifest_sample(
    index: int,
    split: str,
    sample_id: str,
    label: np.ndarray,
    feature_shape: list[int],
    metainfo: dict[str, Any],
    input_shape: list[int] | None = None,
) -> dict[str, Any]:
    embedding_rel = Path(split) / sample_id / "embedding.tif"
    label_rel = Path(split) / sample_id / "label.tif"
    sample = {
        "sample_id": sample_id,
        "source_index": int(index),
        "embedding_path": str(embedding_rel).replace("\\", "/"),
        "seg_map_path": str(label_rel).replace("\\", "/"),
        "dataset_name": metainfo.get("dataset_name"),
        "ori_shape": list(label.shape),
        "embedding_shape": feature_shape,
    }
    if input_shape is not None:
        input_rel = Path(split) / sample_id / "input.tif"
        sample["input_path"] = str(input_rel).replace("\\", "/")
        sample["input_shape"] = input_shape
    if "timestamps" in metainfo:
        sample["timestamps"] = _jsonable(metainfo["timestamps"])
    return sample


def _task_from_item(
    index: int,
    split: str,
    output_root: Path,
    item: dict[str, Any],
) -> dict[str, Any]:
    data_sample = item["data_samples"]
    metainfo = data_sample.metainfo
    sample_id = _safe_sample_id(index, metainfo)
    sample_dir = output_root / split / sample_id
    embedding_path = sample_dir / "embedding.tif"
    input_path = sample_dir / "input.tif"
    label_path = sample_dir / "label.tif"
    label = data_sample.gt_sem_seg.data.squeeze(0).cpu().numpy()
    return {
        "index": index,
        "item": item,
        "metainfo": metainfo,
        "sample_id": sample_id,
        "sample_dir": sample_dir,
        "embedding_path": embedding_path,
        "input_path": input_path,
        "label_path": label_path,
        "label": label,
    }


def _save_task_output(
    task: dict[str, Any],
    feature: np.ndarray,
    embedding_names: list[str],
    save_inputs: bool,
) -> dict[str, Any]:
    task["sample_dir"].mkdir(parents=True, exist_ok=True)
    save_geotiff(
        task["embedding_path"],
        feature.astype(np.float32, copy=False),
        descriptions=embedding_names,
    )
    input_shape = None
    if save_inputs:
        input_arr = task["item"]["inputs"].detach().cpu().numpy()
        save_geotiff(
            task["input_path"],
            input_arr.astype(np.float32, copy=False),
        )
        input_shape = list(input_arr.shape)
    save_geotiff(task["label_path"], task["label"])
    return _make_manifest_sample(
        index=task["index"],
        split=task["embedding_path"].parent.parent.name,
        sample_id=task["sample_id"],
        label=task["label"],
        feature_shape=list(feature.shape),
        metainfo=task["metainfo"],
        input_shape=input_shape,
    )


def _maybe_manifest_from_existing(
    task: dict[str, Any],
    split: str,
    skip_existing: bool,
    save_inputs: bool,
) -> dict[str, Any] | None:
    if not skip_existing:
        return None
    if not task["embedding_path"].exists() or not task["label_path"].exists():
        return None
    input_shape = None
    if save_inputs:
        if not task["input_path"].exists():
            return None
        input_shape = _embedding_shape(task["input_path"])
    return _make_manifest_sample(
        index=task["index"],
        split=split,
        sample_id=task["sample_id"],
        label=task["label"],
        feature_shape=_embedding_shape(task["embedding_path"]),
        metainfo=task["metainfo"],
        input_shape=input_shape,
    )


def _stack_inputs(tasks: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    shapes = {tuple(task["item"]["inputs"].shape) for task in tasks}
    if len(shapes) != 1:
        raise ValueError(f"Batch contains mixed input shapes: {sorted(shapes)}")
    inputs = torch.stack(
        [task["item"]["inputs"].float() for task in tasks],
        dim=0,
    )
    if device.type == "cuda":
        inputs = inputs.pin_memory()
    return inputs.to(device, non_blocking=device.type == "cuda")


def _forward_tasks(
    backbone,
    tasks: list[dict[str, Any]],
    device: torch.device,
    precision: str,
) -> list[np.ndarray]:
    inputs = _stack_inputs(tasks, device)
    metainfo = [task["metainfo"] for task in tasks]
    if hasattr(backbone, "set_batch_metainfo"):
        backbone.set_batch_metainfo(metainfo)
    with torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16" and device.type == "cuda",
    ):
        features = backbone(inputs)[0]
    return [feature.float().contiguous().cpu().numpy() for feature in features]


def _flush_shape_buckets(
    backbone,
    buckets: dict[tuple[int, ...], list[dict[str, Any]]],
    split_samples: list[dict[str, Any]],
    device: torch.device,
    precision: str,
    embedding_names: list[str] | None,
    save_inputs: bool,
    verbose: bool,
    rank: int,
) -> list[str] | None:
    for shape, tasks in list(buckets.items()):
        if not tasks:
            continue
        if verbose:
            print(f"[rank {rank}] extracting shape={shape}, batch={len(tasks)}")
        try:
            features = _forward_tasks(backbone, tasks, device, precision)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(tasks) == 1:
                raise
            print(
                f"[rank {rank}] OOM for batch size {len(tasks)}; "
                "retrying samples one by one."
            )
            _clear_cache(device)
            features = []
            for task in tasks:
                features.extend(
                    _forward_tasks(backbone, [task], device, precision)
                )

        if embedding_names is None:
            embedding_names = [
                f"embedding_{idx:04d}" for idx in range(features[0].shape[0])
            ]
        for task, feature in zip(tasks, features):
            split_samples.append(
                _save_task_output(task, feature, embedding_names, save_inputs)
            )
        buckets[shape] = []
        _clear_cache(device)
    return embedding_names


def _extract_split(
    cfg,
    split: str,
    output_root: Path,
    batch_size: int,
    device: torch.device,
    pipeline_key: str | None,
    ctx: DistContext,
    precision: str,
    skip_existing: bool,
    save_inputs: bool,
    verbose: bool,
) -> dict[str, Any]:
    dataset = _build_dataset(cfg, split, pipeline_key)
    backbone = _build_backbone(cfg, device)
    local_indices = _split_indices(len(dataset), ctx)
    split_samples: list[dict[str, Any]] = []
    buckets: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    embedding_names: list[str] | None = None
    skipped_count = 0

    with torch.inference_mode():
        for index in local_indices:
            task = _task_from_item(index, split, output_root, dataset[index])
            existing = _maybe_manifest_from_existing(
                task,
                split,
                skip_existing,
                save_inputs,
            )
            if existing is not None:
                split_samples.append(existing)
                skipped_count += 1
                continue

            shape = tuple(task["item"]["inputs"].shape)
            bucket = buckets.setdefault(shape, [])
            bucket.append(task)
            if len(bucket) >= batch_size:
                embedding_names = _flush_shape_buckets(
                    backbone=backbone,
                    buckets={shape: bucket},
                    split_samples=split_samples,
                    device=device,
                    precision=precision,
                    embedding_names=embedding_names,
                    save_inputs=save_inputs,
                    verbose=verbose,
                    rank=ctx.rank,
                )
                buckets[shape] = []

        embedding_names = _flush_shape_buckets(
            backbone=backbone,
            buckets=buckets,
            split_samples=split_samples,
            device=device,
            precision=precision,
            embedding_names=embedding_names,
            save_inputs=save_inputs,
            verbose=verbose,
            rank=ctx.rank,
        )

    split_samples.sort(key=lambda sample: int(sample["source_index"]))
    manifest = {
        "metainfo": {
            "source_config": str(getattr(cfg, "filename", "")),
            "split": split,
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "format": "olmoearth_embedding_geotiff_manifest",
        },
        "samples": split_samples,
    }
    manifest_path = output_root / (
        f"{split}_rank{ctx.rank}.json" if ctx.is_distributed else f"{split}.json"
    )
    save_json(manifest_path, manifest)
    return {
        "split": split,
        "rank": ctx.rank,
        "world_size": ctx.world_size,
        "assigned_samples": len(local_indices),
        "num_samples": len(split_samples),
        "skipped_existing": skipped_count,
        "save_inputs": save_inputs,
        "manifest": str(manifest_path),
    }


def _merge_split_manifests(
    output_root: Path,
    split: str,
    world_size: int,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    rank_manifests = []
    for rank in range(world_size):
        path = output_root / f"{split}_rank{rank}.json"
        payload = _load_json(path)
        rank_manifests.append(str(path))
        samples.extend(payload.get("samples", []))
    samples.sort(key=lambda sample: int(sample["source_index"]))
    manifest = {
        "metainfo": {
            "split": split,
            "world_size": world_size,
            "format": "olmoearth_embedding_geotiff_manifest",
            "rank_manifests": rank_manifests,
        },
        "samples": samples,
    }
    out_path = output_root / f"{split}.json"
    save_json(out_path, manifest)
    return {
        "split": split,
        "num_samples": len(samples),
        "manifest": str(out_path),
        "rank_manifests": rank_manifests,
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
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto' for cuda:LOCAL_RANK under torchrun.",
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp32"],
        default="bf16",
        help="CUDA inference precision. bf16 enables autocast and TF32.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Reuse existing outputs when present. With --save-inputs, "
            "input.tif must also exist."
        ),
    )
    parser.add_argument(
        "--save-inputs",
        action="store_true",
        help=(
            "Also save the pipeline input tensor as input.tif for inspection. "
            "For crop-type this is the normalized 12-band OLMoEarth input, "
            "not the raw GEO-Bench source array."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-batch logging.",
    )
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

    ctx = _get_dist_context()
    device = _resolve_device(args.device, ctx)
    _init_distributed(ctx, device)
    _configure_cuda_fast_math(device)

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

    try:
        local_summaries = [
            _extract_split(
                cfg=cfg,
                split=split,
                output_root=output_root,
                batch_size=args.batch_size,
                device=device,
                pipeline_key=pipeline_key,
                ctx=ctx,
                precision=args.precision,
                skip_existing=args.skip_existing,
                save_inputs=args.save_inputs,
                verbose=not args.quiet,
            )
            for split in args.splits
        ]
        save_json(
            output_root / f"summary_rank{ctx.rank}.json",
            {
                "rank": ctx.rank,
                "local_rank": ctx.local_rank,
                "world_size": ctx.world_size,
                "device": str(device),
                "precision": args.precision,
                "batch_size": args.batch_size,
                "save_inputs": args.save_inputs,
                "splits": local_summaries,
            },
        )
        _barrier(ctx)

        if ctx.rank == 0:
            if ctx.is_distributed:
                summaries = [
                    _merge_split_manifests(output_root, split, ctx.world_size)
                    for split in args.splits
                ]
            else:
                summaries = local_summaries
            save_json(
                output_root / "summary.json",
                {
                    "world_size": ctx.world_size,
                    "device": str(device),
                    "precision": args.precision,
                    "batch_size": args.batch_size,
                    "save_inputs": args.save_inputs,
                    "splits": summaries,
                },
            )
        _barrier(ctx)
    finally:
        _destroy_distributed(ctx)


if __name__ == "__main__":
    main()
