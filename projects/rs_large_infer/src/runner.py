from __future__ import annotations

import math
import queue
import threading
from pathlib import Path

from mmengine.config import Config
from mmengine.registry import init_default_scope
from tqdm.auto import tqdm

try:
    import rasterio
    from rasterio.windows import Window
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "large image inference requires rasterio to read/write GeoTIFFs."
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


_QUEUE_SENTINEL = object()


def _build_batch_items_serial(src, adapter, batch_grids):
    """串行构造一个 batch 的滑窗输入。"""

    return [
        (grid, adapter.pack(adapter.read(src, grid), src, grid))
        for grid in batch_grids
    ]


def _start_prefetch_thread(
    image_path: str,
    grids: list[tuple[int, int, int, int, int, int, int, int]],
    adapter,
    batch_size: int,
    max_prefetch_batches: int,
):
    """启动后台线程，提前读取并打包滑窗 batch。"""

    item_queue: queue.Queue = queue.Queue(maxsize=max(1, max_prefetch_batches))
    error_queue: queue.Queue = queue.Queue(maxsize=1)

    def producer() -> None:
        """在独立线程中读取 GeoTIFF 并准备 batch。"""

        try:
            with rasterio.open(image_path) as prefetch_src:
                for batch_start in range(0, len(grids), batch_size):
                    batch_grids = grids[batch_start:batch_start + batch_size]
                    item_queue.put(
                        _build_batch_items_serial(prefetch_src, adapter, batch_grids)
                    )
        except Exception as exc:  # pragma: no cover
            error_queue.put(exc)
        finally:
            item_queue.put(_QUEUE_SENTINEL)

    thread = threading.Thread(
        target=producer,
        name="rs-large-infer-prefetch",
        daemon=True,
    )
    thread.start()
    return thread, item_queue, error_queue


def _iter_prepared_batches(
    src,
    adapter,
    grids: list[tuple[int, int, int, int, int, int, int, int]],
    batch_size: int,
    prefetch_batches: int,
):
    """按配置选择串行或预取模式，逐个产出准备好的 batch。"""

    if prefetch_batches <= 0:
        for batch_start in range(0, len(grids), batch_size):
            yield _build_batch_items_serial(
                src,
                adapter,
                grids[batch_start:batch_start + batch_size],
            )
        return

    thread, item_queue, error_queue = _start_prefetch_thread(
        src.name,
        grids,
        adapter,
        batch_size,
        prefetch_batches,
    )
    try:
        while True:
            batch_items = item_queue.get()
            if batch_items is _QUEUE_SENTINEL:
                if not error_queue.empty():
                    raise error_queue.get()
                break
            yield batch_items
    finally:
        thread.join(timeout=1.0)


def run_large_inference(args) -> None:
    """执行完整大图推理流程：读配置、建模型、滑窗预测并写出 GeoTIFF。"""

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.prefetch_batches < 0:
        raise ValueError("--prefetch-batches must be non-negative.")

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
            prepared_batches = _iter_prepared_batches(
                src,
                adapter,
                grids,
                args.batch_size,
                args.prefetch_batches,
            )
            for batch_items in prepared_batches:
                progress.update(1)
                batch_grids = [grid for grid, _ in batch_items]
                batch_payload = [item for _, item in batch_items]
                predictions = predict_batch(model, batch_payload)
                for grid, pred in zip(batch_grids, predictions):
                    x, y, _, _, crop_x, crop_y, crop_w, crop_h = grid
                    cropped = pred[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                    dst.write(
                        cropped.astype(out_dtype, copy=False),
                        1,
                        window=Window(x + crop_x, y + crop_y, crop_w, crop_h),
                    )
            progress.close()
