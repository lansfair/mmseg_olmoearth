from __future__ import annotations

import math
from pathlib import Path

from mmengine.config import Config
from mmengine.registry import init_default_scope
from tqdm.auto import tqdm

try:
    import rasterio
    from rasterio.windows import Window
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "large_image_inference.py requires rasterio to read/write GeoTIFFs."
    ) from exc

from .adapters import select_adapter
from .utils import (
    build_model,
    cfg_get_crop_size,
    get_pipeline,
    import_custom_modules,
    make_grids,
    num_classes,
    output_dtype,
    predict_batch,
    validate_large_inference_config,
)


def run_large_inference(args) -> None:
    """执行完整大图推理流程：读配置、建模型、滑窗预测并写出 GeoTIFF。"""

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_custom_modules(cfg)
    init_default_scope(cfg.get("default_scope", "mmseg"))
    validate_large_inference_config(cfg, args)

    raw_pipeline = get_pipeline(cfg)
    adapter = select_adapter(cfg, raw_pipeline, args)
    window_size = tuple(args.window_size) if args.window_size else cfg_get_crop_size(cfg)
    stride = tuple(args.stride) if args.stride else window_size

    model = build_model(cfg, args.checkpoint, args.device)
    out_dtype = output_dtype(num_classes(cfg))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.image) as src:
        adapter.prepare(src)
        grids = make_grids(src.width, src.height, window_size, stride)
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype=out_dtype,
            nodata=None,
            compress="lzw",
            BIGTIFF="IF_SAFER",
        )

        total_batches = math.ceil(len(grids) / args.batch_size)
        print(
            f"Adapter: {adapter.name}; windows: {len(grids)}; "
            f"batch size: {args.batch_size}; output: {output_path}"
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            progress = tqdm(
                range(total_batches),
                desc="Large image inference",
                unit="batch",
                dynamic_ncols=True,
            )
            for batch_idx in progress:
                batch_grids = grids[
                    batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size
                ]
                batch_items = [
                    adapter.pack(adapter.read(src, grid), src, grid)
                    for grid in batch_grids
                ]
                predictions = predict_batch(model, batch_items)
                for grid, pred in zip(batch_grids, predictions):
                    x, y, _, _, crop_x, crop_y, crop_w, crop_h = grid
                    cropped = pred[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                    dst.write(
                        cropped.astype(out_dtype, copy=False),
                        1,
                        window=Window(x + crop_x, y + crop_y, crop_w, crop_h),
                    )
