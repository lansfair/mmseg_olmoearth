from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

try:
    from rasterio.warp import transform as transform_coords
    from rasterio.windows import Window
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "large image inference requires rasterio to read/write GeoTIFFs."
    ) from exc


S2_BANDS = [
    "B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12",
    "B01", "B09"
]
PACK_TYPES = {"PackSegInputs", "PackOlmoEarthSegInputs"}


def get_pipeline(cfg: Config) -> list[dict[str, Any]]:
    """从配置中取得推理 pipeline，优先使用顶层 test_pipeline。"""
    pipeline = cfg.get("test_pipeline")
    if pipeline is None:
        pipeline = cfg.test_dataloader.dataset.pipeline
    return [dict(transform) for transform in pipeline]


def cfg_get_crop_size(cfg: Config) -> tuple[int, int]:
    """从配置的 crop_size 或 data_preprocessor.size 推断默认窗口尺寸。"""
    size = None
    if cfg.get("crop_size") is not None:
        size = cfg.crop_size
    elif cfg.model.get("data_preprocessor", {}).get("size") is not None:
        size = cfg.model.data_preprocessor.size
    if size is None:
        raise ValueError(
            "Cannot infer window size from config. Pass --window-size WIDTH HEIGHT."
        )
    if isinstance(size, int):
        return int(size), int(size)
    if len(size) != 2:
        raise ValueError(f"Unsupported crop_size/data_preprocessor.size: {size}")
    return int(size[1]), int(size[0])


def find_transform(
    pipeline: list[dict[str, Any]],
    transform_type: str,
) -> dict[str, Any] | None:
    """在 pipeline 配置列表中查找指定类型的 transform。"""
    for transform in pipeline:
        if transform.get("type") == transform_type:
            return transform
    return None


def clean_pipeline(
    pipeline: list[dict[str, Any]],
    extra_drop_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    """移除加载、打包和不适合滑窗推理的 transform。"""
    extra_drop_types = extra_drop_types or set()
    cleaned = []
    for transform in pipeline:
        transform = dict(transform)
        transform_type = str(transform.get("type"))
        if (
            transform_type.startswith("Load")
            or transform_type in PACK_TYPES
            or transform_type.endswith("PackSegInputs")
            or transform_type in extra_drop_types
        ):
            continue
        if transform_type == "Resize":
            continue
        cleaned.append(transform)
    return cleaned


def import_custom_modules(cfg: Config) -> None:
    """按配置中的 custom_imports 导入项目自定义模块。"""
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg.custom_imports)


def build_model(cfg: Config, checkpoint: str, device: str):
    """构建 MMSeg 模型、加载 checkpoint 并切换到推理模式。"""
    model_type = str(cfg.model.get("type", ""))
    if "Siam" in model_type or "DualInput" in model_type:
        raise ValueError(
            "large_image_inference.py supports single-input semantic "
            "segmentation models only."
        )
    init_default_scope(cfg.get("default_scope", "mmseg"))
    model = MODELS.build(cfg.model)
    model.cfg = cfg
    load_checkpoint(model, checkpoint, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def backbone_cfg(cfg: Config) -> dict[str, Any]:
    """返回模型 backbone 配置，缺失或异常时返回空字典。"""
    backbone = cfg.model.get("backbone", {})
    if not isinstance(backbone, dict):
        return {}
    return backbone


def backbone_type(cfg: Config) -> str:
    """读取模型 backbone 的类型名称。"""
    return str(backbone_cfg(cfg).get("type", ""))


def uses_olmoearth_feature_backbone(cfg: Config) -> bool:
    """判断配置是否使用离线 OLMoEarth embedding backbone。"""
    return backbone_type(cfg) == "OlmoEarthFeatureBackbone"


def validate_large_inference_config(cfg: Config, args) -> None:
    """检查大图推理配置是否与输入模式兼容。"""
    if not uses_olmoearth_feature_backbone(cfg):
        return

    invalid_keys = sorted(
        set(backbone_cfg(cfg))
        - {
            "type",
            "out_channels",
        }
    )
    if invalid_keys:
        raise ValueError(
            "OlmoEarthFeatureBackbone is the offline-embedding backbone and "
            "does not build the OLMoEarth encoder. Remove these online "
            f"backbone options: {invalid_keys}. Use an OlmoEarthBackbone "
            "config for raw RGB/Sentinel-2 GeoTIFF inference."
        )
    if args.input_mode in {"rgb", "s2", "copernicus"}:
        raise ValueError(
            "OlmoEarthFeatureBackbone expects precomputed dense embedding "
            "inputs, not raw RGB/Sentinel-2/Copernicus imagery. Use "
            "--input-mode standard only if the input GeoTIFF already stores "
            "embedding channels, or switch to an online OlmoEarthBackbone "
            "config for raw image inference."
        )


def make_grids(
    width: int,
    height: int,
    window_size: tuple[int, int],
    stride: tuple[int, int],
) -> list[tuple[int, int, int, int, int, int, int, int]]:
    """按窗口尺寸和步长生成覆盖整幅大图的滑窗网格。"""
    win_w, win_h = window_size
    stride_x, stride_y = stride
    if win_w <= 0 or win_h <= 0 or stride_x <= 0 or stride_y <= 0:
        raise ValueError("window size and stride must be positive.")
    if win_w > width or win_h > height:
        raise ValueError(
            f"Window {window_size} is larger than image {(width, height)}. "
            "Use a smaller --window-size."
        )

    x_half_overlap = (win_w - stride_x + 1) // 2
    y_half_overlap = (win_h - stride_y + 1) // 2
    grids = []
    for y in range(0, height, stride_y):
        y_end = y + win_h >= height
        y_offset = height - win_h if y_end else y
        y_crop_off = 0 if y_offset == 0 else y_half_overlap
        y_crop_size = win_h if y_end else win_h - y_crop_off
        for x in range(0, width, stride_x):
            x_end = x + win_w >= width
            x_offset = width - win_w if x_end else x
            x_crop_off = 0 if x_offset == 0 else x_half_overlap
            x_crop_size = win_w if x_end else win_w - x_crop_off
            grids.append(
                (
                    x_offset,
                    y_offset,
                    win_w,
                    win_h,
                    x_crop_off,
                    y_crop_off,
                    x_crop_size,
                    y_crop_size,
                )
            )
    return grids


def read_window(
    src,
    grid: tuple[int, int, int, int, int, int, int, int],
    band_indices: list[int] | None = None,
    band_scales: list[float] | None = None,
    nan_to_num: bool = False,
    to_float32: bool = False,
) -> np.ndarray:
    """按滑窗和 band 设置从 GeoTIFF 读取 HWC 格式数组。"""
    x, y, w, h = grid[:4]
    indices = band_indices if band_indices is not None else list(range(1, src.count + 1))
    if not indices:
        raise ValueError("At least one band must be read.")
    if max(indices) > src.count or min(indices) < 1:
        raise ValueError(f"Band indices {indices} are out of range for {src.count} bands.")

    array = src.read(indices, window=Window(x, y, w, h))
    if to_float32:
        array = array.astype(np.float32, copy=False)
    if nan_to_num:
        array = np.nan_to_num(array)
    if band_scales is not None:
        if len(band_scales) != array.shape[0]:
            raise ValueError(
                "band_scales length must match loaded band count, got "
                f"{len(band_scales)} scales for {array.shape[0]} bands."
            )
        array = array * np.asarray(band_scales, dtype=np.float32).reshape(-1, 1, 1)
    return np.ascontiguousarray(array.transpose(1, 2, 0))


def data_sample(metainfo: dict[str, Any]) -> SegDataSample:
    """用给定 metainfo 构造 MMSeg SegDataSample。"""
    sample = SegDataSample()
    sample.set_metainfo(metainfo)
    return sample


def pack_window(
    results: dict[str, Any],
    pipeline: Compose,
    meta_keys: tuple[str, ...],
) -> dict[str, Any]:
    """运行 transform pipeline，并打包成 model.test_step 需要的数据项。"""
    results = pipeline(results)
    img = results["img"]
    if img.ndim < 3:
        img = np.expand_dims(img, -1)
    inputs = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1))).contiguous()
    metainfo = {key: results[key] for key in meta_keys if key in results}
    return {"inputs": inputs, "data_samples": data_sample(metainfo)}


def predict_batch(model, batch_items: list[dict[str, Any]]) -> list[np.ndarray]:
    """对一个 batch 的滑窗数据执行模型推理并返回 numpy 标签图。"""
    data = {
        "inputs": [item["inputs"] for item in batch_items],
        "data_samples": [item["data_samples"] for item in batch_items],
    }
    with torch.no_grad():
        results = model.test_step(data)
    return [
        result.pred_sem_seg.data.squeeze(0).detach().cpu().numpy()
        for result in results
    ]


def num_classes(cfg: Config) -> int | None:
    """从配置中读取类别数，用于决定输出 GeoTIFF 数据类型。"""
    if cfg.get("num_classes") is not None:
        return int(cfg.num_classes)
    decode_head = cfg.model.get("decode_head", {})
    if isinstance(decode_head, dict) and decode_head.get("num_classes") is not None:
        return int(decode_head["num_classes"])
    return None


def output_dtype(class_count: int | None) -> str:
    """根据类别数选择保存预测标签所需的最小整数类型。"""
    if class_count is not None and class_count <= 255:
        return "uint8"
    if class_count is not None and class_count <= 32767:
        return "int16"
    return "int32"


def normalize_band_name(name: str) -> str:
    """规范化 band 名称，便于和 OLMoEarth 标准 band 列表匹配。"""
    return str(name).strip().upper()


def band_indices_for_s2(
    source_band_names: list[str] | None,
    raster_count: int,
) -> list[int]:
    """根据输入 band 名称计算重排到 OLMoEarth S2 顺序的 1-based 索引。"""
    if source_band_names is None:
        if raster_count == len(S2_BANDS):
            return list(range(1, raster_count + 1))
        raise ValueError(
            "S2 input needs --source-band-names unless the GeoTIFF already "
            f"has exactly {len(S2_BANDS)} bands in OLMoEarth order: {S2_BANDS}"
        )
    normalized = [normalize_band_name(name) for name in source_band_names]
    missing = [band for band in S2_BANDS if band not in normalized]
    if missing:
        raise ValueError(f"Input is missing required S2 bands: {missing}")
    return [normalized.index(band) + 1 for band in S2_BANDS]


def as_float_list(value: Any) -> list[float] | None:
    """将可迭代数值转换成 float 列表，None 保持为 None。"""
    if value is None:
        return None
    return [float(item) for item in value]


def as_int_list(value: Any) -> list[int] | None:
    """将可迭代数值转换成 int 列表，None 保持为 None。"""
    if value is None:
        return None
    return [int(item) for item in value]


def parse_days_from_date(date_text: str | None) -> float:
    """把 YYYYMMDD 日期字符串转换成 days since 1970-01-01。"""
    if date_text is None:
        return float("nan")
    try:
        sensing_date = datetime.strptime(date_text[:8], "%Y%m%d").date()
    except ValueError:
        return float("nan")
    return float((sensing_date - date(1970, 1, 1)).days)


def parse_days_from_filename(
    filename: str,
    separator: str | None,
    token_index: int,
) -> float:
    """按分隔符和 token 位置从文件名中解析日期天数。"""
    if separator is None:
        return float("nan")
    basename = Path(filename).name
    try:
        token = basename.split(separator)[int(token_index)]
    except IndexError:
        return float("nan")
    return parse_days_from_date(token)


def window_lon_lat(
    src,
    grid: tuple[int, int, int, int, int, int, int, int],
    lon_lat_override: tuple[float, float] | None,
) -> tuple[float, float]:
    """计算滑窗中心点经纬度，可被命令行指定值覆盖。"""
    if lon_lat_override is not None:
        return lon_lat_override
    if src.transform is None:
        return float("nan"), float("nan")

    x, y, w, h = grid[:4]
    center_col = x + (w - 1) / 2
    center_row = y + (h - 1) / 2
    coord_x, coord_y = src.xy(center_row, center_col)
    if src.crs is not None:
        try:
            lon_values, lat_values = transform_coords(
                src.crs,
                "EPSG:4326",
                [coord_x],
                [coord_y],
            )
            return float(lon_values[0]), float(lat_values[0])
        except Exception:
            pass
    return float(coord_x), float(coord_y)
